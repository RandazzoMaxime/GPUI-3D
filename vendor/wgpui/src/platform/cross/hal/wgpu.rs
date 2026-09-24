//! Implémentation wgpu de [`Gpu`] : passe-plat vers l'API wgpu, dont le renderer
//! reprend la forme.

use std::mem::ManuallyDrop;
use std::num::NonZeroU64;
use std::ops::Range;

use wgpu::util::DeviceExt as _;

use super::{
    Acquire, BindEntry, BindResource, BindingKind, Blend, BufferUsage, Gpu, LayoutEntry, LoadOp,
    PassDesc, PipelineDesc, ShaderStages, TextureUsage, Topology,
};
use crate::{GpuSpecs, NativeDevice, NativeTexture, SurfaceFormat, WindowPresentMode};

/// Device wgpu partagé par l'UI et les moteurs 3D d'une application.
pub(crate) struct WgpuGpu {
    pub(crate) instance: wgpu::Instance,
    pub(crate) adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) desired_maximum_frame_latency: u32,
    /// wgpu ignore les soumissions natives sur sa `VkQueue` : ce verrou les exclut des
    /// siennes (`submit`, `present`).
    pub(crate) queue_lock: parking_lot::Mutex<()>,
}

pub(crate) struct WgpuSwapchain {
    surface: ManuallyDrop<wgpu::Surface<'static>>,
    configuration: wgpu::SurfaceConfiguration,
}

impl Drop for WgpuSwapchain {
    fn drop(&mut self) {
        // SAFETY: seule implémentation de Drop ; la surface n'a pas encore été libérée.
        // Libérée dans catch_unwind : Vulkan panique si une SurfaceTexture retient encore
        // un sémaphore de la swapchain (fenêtre fermée en pleine trame).
        let surface = unsafe { ManuallyDrop::take(&mut self.surface) };
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(surface))).is_err() {
            log::warn!("surface wgpu libérée avec une image de swapchain encore en vol");
        }
    }
}

fn buffer_usages(usage: BufferUsage) -> wgpu::BufferUsages {
    let mut usages = wgpu::BufferUsages::empty();
    for (flag, wgpu_flag) in [
        (BufferUsage::UNIFORM, wgpu::BufferUsages::UNIFORM),
        (BufferUsage::STORAGE, wgpu::BufferUsages::STORAGE),
        (BufferUsage::VERTEX, wgpu::BufferUsages::VERTEX),
        (BufferUsage::COPY_SRC, wgpu::BufferUsages::COPY_SRC),
        (BufferUsage::COPY_DST, wgpu::BufferUsages::COPY_DST),
    ] {
        if usage.contains(flag) {
            usages |= wgpu_flag;
        }
    }
    usages
}

fn texture_usages(usage: TextureUsage) -> wgpu::TextureUsages {
    let mut usages = wgpu::TextureUsages::empty();
    for (flag, wgpu_flag) in [
        (TextureUsage::RENDER_TARGET, wgpu::TextureUsages::RENDER_ATTACHMENT),
        (TextureUsage::SAMPLED, wgpu::TextureUsages::TEXTURE_BINDING),
        (TextureUsage::COPY_SRC, wgpu::TextureUsages::COPY_SRC),
        (TextureUsage::COPY_DST, wgpu::TextureUsages::COPY_DST),
    ] {
        if usage.contains(flag) {
            usages |= wgpu_flag;
        }
    }
    usages
}

fn shader_stages(stages: ShaderStages) -> wgpu::ShaderStages {
    match stages {
        ShaderStages::Vertex => wgpu::ShaderStages::VERTEX,
        ShaderStages::Fragment => wgpu::ShaderStages::FRAGMENT,
        ShaderStages::VertexFragment => wgpu::ShaderStages::VERTEX_FRAGMENT,
    }
}

fn present_mode(mode: WindowPresentMode) -> wgpu::PresentMode {
    match mode {
        WindowPresentMode::Fifo => wgpu::PresentMode::Fifo,
        WindowPresentMode::Mailbox => wgpu::PresentMode::Mailbox,
        WindowPresentMode::Immediate => wgpu::PresentMode::Immediate,
    }
}

fn window_present_mode(mode: wgpu::PresentMode) -> Option<WindowPresentMode> {
    match mode {
        wgpu::PresentMode::Fifo => Some(WindowPresentMode::Fifo),
        wgpu::PresentMode::Mailbox => Some(WindowPresentMode::Mailbox),
        wgpu::PresentMode::Immediate => Some(WindowPresentMode::Immediate),
        _ => None,
    }
}

impl Gpu for WgpuGpu {
    type Format = wgpu::TextureFormat;
    type Buffer = wgpu::Buffer;
    type Texture = wgpu::Texture;
    type TextureView = wgpu::TextureView;
    type Sampler = wgpu::Sampler;
    type BindGroupLayout = wgpu::BindGroupLayout;
    type BindGroup = wgpu::BindGroup;
    type Pipeline = wgpu::RenderPipeline;
    type Encoder = wgpu::CommandEncoder;
    type Pass<'a> = wgpu::RenderPass<'a>;
    type Swapchain = WgpuSwapchain;
    type Frame = wgpu::SurfaceTexture;
    #[cfg(feature = "flamegraph")]
    type Profiler = profiler::WgpuProfiler;

    const ATLAS_MONOCHROME: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
    const ATLAS_POLYCHROME: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    fn bytes_per_pixel(format: wgpu::TextureFormat) -> u32 {
        format.block_copy_size(None).unwrap_or(4)
    }

    fn gpu_specs(&self) -> GpuSpecs {
        let info = self.adapter.get_info();
        GpuSpecs {
            is_software_emulated: info.device_type == wgpu::DeviceType::Cpu,
            device_name: info.name,
            driver_name: info.driver,
            driver_info: info.driver_info,
        }
    }

    fn min_uniform_offset_alignment(&self) -> u32 {
        self.device.limits().min_uniform_buffer_offset_alignment
    }

    fn create_buffer(&self, label: &str, size: u64, usage: BufferUsage) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: buffer_usages(usage),
            mapped_at_creation: false,
        })
    }

    fn create_buffer_init(&self, label: &str, contents: &[u8], usage: BufferUsage) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: buffer_usages(usage),
        })
    }

    fn buffer_size(buffer: &wgpu::Buffer) -> u64 {
        buffer.size()
    }

    fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buffer, offset, data);
    }

    fn create_texture(
        &self,
        label: &str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: TextureUsage,
    ) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: texture_usages(usage),
            view_formats: &[],
        })
    }

    fn texture_size(texture: &wgpu::Texture) -> (u32, u32) {
        (texture.width(), texture.height())
    }

    fn create_view(texture: &wgpu::Texture) -> wgpu::TextureView {
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn write_texture(
        &self,
        texture: &wgpu::Texture,
        origin: (u32, u32),
        size: (u32, u32),
        bytes_per_pixel: u32,
        data: &[u8],
    ) {
        // wgpu exige des lignes alignées sur COPY_BYTES_PER_ROW_ALIGNMENT.
        let unpadded_bytes_per_row = (size.0 * bytes_per_pixel) as usize;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let height = size.1 as usize;
        let padded = (padded_bytes_per_row != unpadded_bytes_per_row).then(|| {
            let mut padded = vec![0u8; padded_bytes_per_row * height];
            for row in 0..height {
                let source = row * unpadded_bytes_per_row;
                let destination = row * padded_bytes_per_row;
                padded[destination..destination + unpadded_bytes_per_row]
                    .copy_from_slice(&data[source..source + unpadded_bytes_per_row]);
            }
            padded
        });
        // queue.write_texture plutôt qu'un buffer de staging : contourne des soucis de
        // pilote (repro helio/ship_flight).
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: origin.0, y: origin.1, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            padded.as_deref().unwrap_or(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row as u32),
                rows_per_image: None,
            },
            wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        );
    }

    fn create_linear_sampler(&self, label: &str) -> wgpu::Sampler {
        self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        })
    }

    fn create_bind_group_layout(&self, label: &str, entries: &[LayoutEntry]) -> wgpu::BindGroupLayout {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = entries
            .iter()
            .map(|entry| wgpu::BindGroupLayoutEntry {
                binding: entry.binding,
                visibility: shader_stages(entry.visibility),
                ty: match entry.kind {
                    BindingKind::Uniform { dynamic_offset, min_size } => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: dynamic_offset,
                        min_binding_size: min_size.and_then(NonZeroU64::new),
                    },
                    BindingKind::Storage => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    BindingKind::Texture => wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    BindingKind::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                },
                count: None,
            })
            .collect();
        self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        })
    }

    fn create_bind_group(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        entries: &[BindEntry<'_, Self>],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = entries
            .iter()
            .map(|entry| wgpu::BindGroupEntry {
                binding: entry.binding,
                resource: match &entry.resource {
                    BindResource::Buffer { buffer, offset, size } => {
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: *offset,
                            size: size.and_then(NonZeroU64::new),
                        })
                    }
                    BindResource::Texture(view) => wgpu::BindingResource::TextureView(view),
                    BindResource::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
                },
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }

    fn create_pipeline(&self, desc: &PipelineDesc<'_, Self>) -> wgpu::RenderPipeline {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(desc.label),
            source: wgpu::ShaderSource::Wgsl(super::super::shaders::wgsl_source(desc.shader.name()).into()),
        });
        let layouts: Vec<Option<&wgpu::BindGroupLayout>> = desc.layouts.iter().map(|layout| Some(*layout)).collect();
        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(desc.label),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });
        let blend = match desc.blend {
            Blend::Alpha => wgpu::BlendState::ALPHA_BLENDING,
            Blend::PremultipliedAlpha => wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
        };
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(desc.label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some(desc.vertex_entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: match desc.topology {
                    Topology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
                    Topology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
                },
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(desc.fragment_entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: desc.format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    fn create_encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    fn begin_pass<'a>(encoder: &'a mut wgpu::CommandEncoder, desc: &PassDesc<'_, Self>) -> wgpu::RenderPass<'a> {
        let load = match desc.load {
            LoadOp::Clear([r, g, b, a]) => wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
            LoadOp::Load => wgpu::LoadOp::Load,
        };
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(desc.label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: desc.target,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                resolve_target: None,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            #[cfg(feature = "flamegraph")]
            timestamp_writes: desc.timestamps.map(|reserved| reserved.writes()),
            #[cfg(not(feature = "flamegraph"))]
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn set_pipeline(pass: &mut wgpu::RenderPass<'_>, pipeline: &wgpu::RenderPipeline) {
        pass.set_pipeline(pipeline);
    }

    fn set_bind_group(pass: &mut wgpu::RenderPass<'_>, index: u32, group: &wgpu::BindGroup, dynamic_offsets: &[u32]) {
        pass.set_bind_group(index, group, dynamic_offsets);
    }

    fn set_viewport(pass: &mut wgpu::RenderPass<'_>, x: f32, y: f32, width: f32, height: f32) {
        pass.set_viewport(x, y, width, height, 0.0, 1.0);
    }

    fn set_scissor_rect(pass: &mut wgpu::RenderPass<'_>, x: u32, y: u32, width: u32, height: u32) {
        pass.set_scissor_rect(x, y, width, height);
    }

    fn draw(pass: &mut wgpu::RenderPass<'_>, vertices: Range<u32>, instances: Range<u32>) {
        pass.draw(vertices, instances);
    }

    fn copy_buffer_to_buffer(
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        source_offset: u64,
        destination: &wgpu::Buffer,
        destination_offset: u64,
        size: u64,
    ) {
        encoder.copy_buffer_to_buffer(source, source_offset, destination, destination_offset, size);
    }

    fn copy_texture_to_texture(
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        destination: &wgpu::Texture,
        width: u32,
        height: u32,
    ) {
        encoder.copy_texture_to_texture(
            source.as_image_copy(),
            destination.as_image_copy(),
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
    }

    fn create_swapchain(
        &self,
        window: raw_window_handle::RawWindowHandle,
        display: raw_window_handle::RawDisplayHandle,
        width: u32,
        height: u32,
    ) -> anyhow::Result<WgpuSwapchain> {
        // SAFETY: la fenêtre survit à sa swapchain (le renderer appartient à la fenêtre).
        let surface = unsafe {
            self.instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(display),
                raw_window_handle: window,
            })?
        };
        let capabilities = surface.get_capabilities(&self.adapter);
        // Les shaders (hsla_to_rgba) écrivent déjà du sRGB : un format non sRGB évite une
        // double conversion linéaire → sRGB.
        let format = capabilities
            .formats
            .iter()
            .find(|format| !format.is_srgb())
            .or(capabilities.formats.first())
            .copied()
            .ok_or_else(|| anyhow::anyhow!("surface sans format de présentation"))?;
        let alpha_mode = if capabilities.alpha_modes.contains(&wgpu::CompositeAlphaMode::PreMultiplied) {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else {
            capabilities
                .alpha_modes
                .first()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("surface sans mode alpha"))?
        };
        let configuration = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: self.desired_maximum_frame_latency,
        };
        Ok(WgpuSwapchain { surface: ManuallyDrop::new(surface), configuration })
    }

    fn swapchain_format(swapchain: &WgpuSwapchain) -> wgpu::TextureFormat {
        swapchain.configuration.format
    }

    fn swapchain_premultiplied(swapchain: &WgpuSwapchain) -> bool {
        swapchain.configuration.alpha_mode == wgpu::CompositeAlphaMode::PreMultiplied
    }

    fn swapchain_size(swapchain: &WgpuSwapchain) -> (u32, u32) {
        (swapchain.configuration.width, swapchain.configuration.height)
    }

    fn swapchain_present_mode(swapchain: &WgpuSwapchain) -> WindowPresentMode {
        window_present_mode(swapchain.configuration.present_mode).unwrap_or_default()
    }

    fn supported_present_modes(&self, swapchain: &WgpuSwapchain) -> Vec<WindowPresentMode> {
        swapchain
            .surface
            .get_capabilities(&self.adapter)
            .present_modes
            .into_iter()
            .filter_map(window_present_mode)
            .collect()
    }

    fn swapchain_frame_latency(swapchain: &WgpuSwapchain) -> u32 {
        swapchain.configuration.desired_maximum_frame_latency
    }

    fn configure_swapchain(&self, swapchain: &mut WgpuSwapchain, width: u32, height: u32, mode: WindowPresentMode) {
        swapchain.configuration.width = width;
        swapchain.configuration.height = height;
        swapchain.configuration.present_mode = present_mode(mode);
        swapchain.surface.configure(&self.device, &swapchain.configuration);
    }

    fn acquire(&self, swapchain: &mut WgpuSwapchain) -> Acquire<wgpu::SurfaceTexture> {
        match swapchain.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                Acquire::Frame(texture)
            }
            wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Validation => Acquire::Outdated,
            wgpu::CurrentSurfaceTexture::Timeout => Acquire::Skip("swap chain acquire timed out"),
            wgpu::CurrentSurfaceTexture::Occluded => Acquire::Skip("swap chain acquire occluded"),
        }
    }

    fn copy_texture_to_frame(encoder: &mut wgpu::CommandEncoder, source: &wgpu::Texture, frame: &wgpu::SurfaceTexture) {
        encoder.copy_texture_to_texture(
            source.as_image_copy(),
            frame.texture.as_image_copy(),
            wgpu::Extent3d {
                width: frame.texture.width(),
                height: frame.texture.height(),
                depth_or_array_layers: 1,
            },
        );
    }

    fn submit(&self, encoder: wgpu::CommandEncoder) {
        let command_buffer = encoder.finish();
        let _queue = self.queue_lock.lock();
        self.queue.submit(Some(command_buffer));
    }

    fn present(&self, frame: wgpu::SurfaceTexture) {
        let _queue = self.queue_lock.lock();
        self.queue.present(frame);
    }

    fn init_external_textures(&self, textures: [&wgpu::Texture; 3]) {
        // Un moteur natif écrit les tampons sans que wgpu le sache : sans cet effacement,
        // wgpu jugerait une texture vierge et la remettrait à zéro avant de l'échantillonner.
        // Contrat de layout : entre deux trames, chaque tampon est dans l'état `RESOURCE`
        // (Vulkan : `SHADER_READ_ONLY_OPTIMAL`) ; un moteur natif l'y laisse.
        let mut encoder = self.create_encoder("surface_buffers_init");
        for texture in textures {
            let view = Self::create_view(texture);
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("surface_buffer_clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        }
        encoder.transition_resources(
            std::iter::empty(),
            textures.into_iter().map(|texture| wgpu::TextureTransition {
                texture,
                selector: None,
                state: wgpu::TextureUses::RESOURCE,
            }),
        );
        self.submit(encoder);
    }

    fn surface_format(format: SurfaceFormat) -> wgpu::TextureFormat {
        match format {
            SurfaceFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
            SurfaceFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        }
    }

    fn queue_lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.queue_lock.lock()
    }

    fn native_device(&self) -> Option<NativeDevice> {
        #[cfg(target_vendor = "apple")]
        // SAFETY: poignées lues sans être détruites ; le device survit à l'appelant.
        if let Some(device) = unsafe { self.device.as_hal::<wgpu::hal::api::Metal>() } {
            // SAFETY: idem.
            let queue = unsafe { self.queue.as_hal::<wgpu::hal::api::Metal>() }?;
            return Some(NativeDevice::Metal {
                device: &**device.raw_device() as *const _ as *mut std::ffi::c_void,
                queue: queue.as_raw() as *const _ as *mut std::ffi::c_void,
            });
        }
        #[cfg(windows)]
        // SAFETY: poignées lues sans être détruites ; le device survit à l'appelant.
        if let Some(device) = unsafe { self.device.as_hal::<wgpu::hal::api::Dx12>() } {
            // SAFETY: idem.
            let queue = unsafe { self.queue.as_hal::<wgpu::hal::api::Dx12>() }?;
            return Some(NativeDevice::Dx12 {
                device: windows_core::Interface::as_raw(device.raw_device()),
                queue: windows_core::Interface::as_raw(queue.as_raw()),
            });
        }
        #[cfg(not(target_family = "wasm"))]
        // SAFETY: poignées lues sans être détruites ; le device survit à l'appelant.
        if let Some(device) = unsafe { self.device.as_hal::<wgpu::hal::api::Vulkan>() } {
            use ash::vk::Handle as _;
            return Some(NativeDevice::Vulkan {
                instance: device.shared_instance().raw_instance().handle().as_raw(),
                physical_device: device.raw_physical_device().as_raw(),
                device: device.raw_device().handle().as_raw(),
                queue: device.raw_queue().as_raw(),
                queue_family_index: device.queue_family_index(),
            });
        }
        None
    }

    fn native_texture(texture: &wgpu::Texture, view: &wgpu::TextureView) -> Option<NativeTexture> {
        #[cfg(target_vendor = "apple")]
        // SAFETY: poignée lue sans être détruite ; l'appelant retient la texture.
        if let Some(raw) = unsafe { texture.as_hal::<wgpu::hal::api::Metal>() } {
            return Some(NativeTexture::Metal(raw.raw_handle() as *const _ as *mut std::ffi::c_void));
        }
        #[cfg(windows)]
        // SAFETY: poignée lue sans être détruite ; l'appelant retient la texture.
        if let Some(raw) = unsafe { texture.as_hal::<wgpu::hal::api::Dx12>() } {
            // SAFETY: idem.
            return Some(NativeTexture::Dx12(windows_core::Interface::as_raw(unsafe { raw.raw_resource() })));
        }
        #[cfg(not(target_family = "wasm"))]
        {
            use ash::vk::Handle as _;
            // SAFETY: poignées lues sans être détruites ; l'appelant retient texture et vue.
            let image = unsafe { texture.as_hal::<wgpu::hal::api::Vulkan>()?.raw_handle() }.as_raw();
            // SAFETY: idem.
            let view = unsafe { view.as_hal::<wgpu::hal::api::Vulkan>()?.raw_handle() }.as_raw();
            return Some(NativeTexture::Vulkan { image, view });
        }
        #[allow(unreachable_code)]
        None
    }
}

#[cfg(feature = "flamegraph")]
pub(crate) mod profiler {
    use std::ops::Range;

    use super::{super::GpuProfiler, WgpuGpu};
    use crate::flamegraph_gpu::{DeepCapturePendingReadback, DeepCaptureRecorder, GpuQueryManager, ReservedTimestamps};

    /// Timestamps de passes et capture profonde (issues #57, #60), par renderer.
    #[derive(Default)]
    pub(crate) struct WgpuProfiler {
        query_manager: Option<GpuQueryManager>,
        submit_present: Option<ReservedTimestamps>,
        in_flight_capture: Option<DeepCapturePendingReadback>,
        finished_capture: Option<DeepCapturePendingReadback>,
    }

    impl GpuProfiler<WgpuGpu> for WgpuProfiler {
        type PassTimestamps = ReservedTimestamps;
        type DeepCapture = DeepCaptureRecorder;

        fn begin_frame(&mut self, gpu: &WgpuGpu, encoder: &mut wgpu::CommandEncoder) -> Option<DeepCaptureRecorder> {
            crate::flamegraph_gpu::sync_with_active_capture(&mut self.query_manager, &gpu.device, &gpu.queue);
            if let Some(manager) = self.query_manager.as_mut() {
                manager.poll_readback(&gpu.device);
                if let Some(frame_index) = crate::current_gpu_correlation_frame_index() {
                    manager.begin_frame(frame_index);
                }
            }
            self.submit_present = self.pass_timestamps("GpuSubmitPresent", crate::GpuPassKind::SubmitPresent);
            if let Some(reserved) = &self.submit_present {
                encoder.write_timestamp(reserved.query_set(), reserved.begin_index());
            }

            if let Some(pending) = self.in_flight_capture.as_mut()
                && let Some(capture) = pending.poll(&gpu.device)
            {
                crate::flamegraph::complete_deep_capture(capture);
                self.in_flight_capture = None;
            }
            (self.in_flight_capture.is_none() && crate::flamegraph::take_deep_capture_request())
                .then(DeepCaptureRecorder::new)
        }

        fn pass_timestamps(&mut self, name: &'static str, kind: crate::GpuPassKind) -> Option<ReservedTimestamps> {
            self.query_manager.as_mut()?.reserve_pair(crate::SpanName::Static(name), kind)
        }

        fn record_draw_call(
            capture: &mut DeepCaptureRecorder,
            kind: crate::DrawCallKind,
            pipeline: &'static str,
            pass: &'static str,
            vertices: Range<u32>,
            instances: Range<u32>,
            bind_groups: u32,
            buffer: Option<crate::DeepCaptureBufferKind>,
            texture: Option<u64>,
            surface: Option<u64>,
        ) {
            capture.record_draw_call(kind, pipeline, pass, vertices, instances, bind_groups, buffer, texture, surface);
        }

        fn end_frame(
            &mut self,
            gpu: &WgpuGpu,
            encoder: &mut wgpu::CommandEncoder,
            capture: Option<DeepCaptureRecorder>,
            buffers: &[(crate::DeepCaptureBufferKind, &wgpu::Buffer); 7],
            atlas: &crate::platform::cross::atlas::Atlas<WgpuGpu>,
            surfaces: &crate::platform::cross::surface_registry::SurfaceRegistry<WgpuGpu>,
        ) {
            if let Some(reserved) = self.submit_present.take() {
                encoder.write_timestamp(reserved.query_set(), reserved.end_index());
            }
            if let Some(manager) = self.query_manager.as_mut() {
                manager.finish_frame(encoder);
            }
            self.finished_capture =
                capture.map(|recorder| recorder.finish(&gpu.device, encoder, buffers, atlas, surfaces));
        }

        fn after_submit(&mut self) {
            if let Some(manager) = self.query_manager.as_mut() {
                manager.begin_readback();
            }
            if let Some(mut pending) = self.finished_capture.take() {
                pending.begin_readback();
                self.in_flight_capture = Some(pending);
            }
        }
    }
}
