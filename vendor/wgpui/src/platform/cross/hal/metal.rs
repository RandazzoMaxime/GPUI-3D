//! Implémentation Metal native (objc2-metal) de [`Gpu`], sans wgpu. macOS.
//!
//! Metal fait l'essentiel de ce que les autres implémentations écrivent à la main : les
//! ressources créées par le device sont « tracked » (dépendances entre encodeurs et
//! command buffers d'une même queue insérées par Metal), et chaque command buffer retient
//! les objets qu'il référence jusqu'à la fin de son exécution.
//!
//! Écritures (contrat `write_*` de [`super`]) : mises en attente, puis copiées par blit
//! depuis un tampon de staging dans un command buffer committé juste avant celui du
//! `submit` — Metal exécute les command buffers dans l'ordre de leur commit.
//!
//! L'espace clip Metal est celui de WebGPU, et `[[vertex_id]]`/`[[instance_id]]` incluent
//! le premier sommet et la première instance : rien à corriger.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ops::Range;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use objc2_06::msg_send;
use objc2_06::rc::{Retained, autoreleasepool};
use objc2_06::runtime::{AnyObject, ProtocolObject};
use objc2_core_foundation::CGSize;
use objc2_foundation_03::NSString;
use objc2_metal::*;
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use parking_lot::Mutex;

use super::{
    Acquire, BindEntry, BindResource, BindingKind, Blend, BufferUsage, Gpu, LayoutEntry, LoadOp, PassDesc,
    PipelineDesc, TextureUsage, Topology,
};
use crate::{GpuSpecs, NativeDevice, NativeTexture, SurfaceFormat, WindowPresentMode};

mod sources {
    include!(concat!(env!("OUT_DIR"), "/msl.rs"));
}

type Obj<T> = Retained<ProtocolObject<T>>;

const SWAPCHAIN_FORMAT: MTLPixelFormat = MTLPixelFormat::BGRA8Unorm;
/// Tampons de la couche : un de plus que les trames en vol.
const MAXIMUM_DRAWABLES: usize = 3;

/// Device Metal partagé par l'UI et les moteurs 3D d'une application.
#[derive(Clone)]
pub(crate) struct MetalGpu(Arc<Shared>);

struct Shared {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    /// L'unique sampler du HAL (linéaire, bords clampés).
    sampler: Obj<dyn MTLSamplerState>,
    /// Lié à `SIZES_BUFFER_SLOT` : déclaré par naga, jamais lu (bornes non vérifiées).
    sizes: Obj<dyn MTLBuffer>,
    libraries: Mutex<HashMap<&'static str, Obj<dyn MTLLibrary>>>,
    pending: Mutex<Vec<Upload>>,
    /// Contrat du trait : une queue Metal est thread-safe, rien à exclure.
    queue_lock: Mutex<()>,
}

// SAFETY: MTLDevice, MTLCommandQueue, MTLSamplerState et MTLLibrary sont thread-safe.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

enum Upload {
    Buffer { buffer: MtlBuffer, offset: u64, data: Vec<u8> },
    Texture { texture: MtlTexture, origin: (u32, u32), size: (u32, u32), bytes_per_pixel: u32, data: Vec<u8> },
}

impl Shared {
    fn command_buffer(&self) -> Obj<dyn MTLCommandBuffer> {
        self.queue.commandBuffer().expect("command buffer Metal")
    }

    /// Copie les écritures en attente dans un command buffer committé avant tout autre.
    fn flush_uploads(&self) {
        let uploads = std::mem::take(&mut *self.pending.lock());
        if uploads.is_empty() {
            return;
        }
        let mut offsets = Vec::with_capacity(uploads.len());
        let mut total = 0usize;
        for upload in &uploads {
            let length = match upload {
                Upload::Buffer { data, .. } | Upload::Texture { data, .. } => data.len(),
            };
            offsets.push(total);
            total = (total + length).next_multiple_of(16);
        }
        let staging = self
            .device
            .newBufferWithLength_options(total.max(16), MTLResourceOptions::StorageModeShared)
            .expect("tampon de staging Metal");
        let base = staging.contents().cast::<u8>();
        let command_buffer = self.command_buffer();
        let blit = command_buffer.blitCommandEncoder().expect("encodeur de copie Metal");
        for (upload, offset) in uploads.iter().zip(offsets) {
            // SAFETY: `offset + len <= total`, tampon partagé tout juste alloué ; copies
            // alignées sur 16 octets, multiples de la taille de pixel.
            unsafe {
                match upload {
                    Upload::Buffer { buffer, offset: destination, data } => {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), base.as_ptr().add(offset), data.len());
                        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            &staging,
                            offset,
                            &buffer.0.raw,
                            *destination as usize,
                            data.len(),
                        );
                    }
                    Upload::Texture { texture, origin, size, bytes_per_pixel, data } => {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), base.as_ptr().add(offset), data.len());
                        let row = (size.0 * bytes_per_pixel) as usize;
                        blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                            &staging,
                            offset,
                            row,
                            row * size.1 as usize,
                            MTLSize { width: size.0 as usize, height: size.1 as usize, depth: 1 },
                            &texture.0.raw,
                            0,
                            0,
                            MTLOrigin { x: origin.0 as usize, y: origin.1 as usize, z: 0 },
                        );
                    }
                }
            }
        }
        blit.endEncoding();
        command_buffer.commit();
    }

    fn library(&self, shader: &'static str) -> Obj<dyn MTLLibrary> {
        let mut libraries = self.libraries.lock();
        libraries
            .entry(shader)
            .or_insert_with(|| {
                let source = sources::msl(shader).unwrap_or_else(|| panic!("MSL absent : {shader}"));
                self.device
                    .newLibraryWithSource_options_error(&NSString::from_str(source), None)
                    .unwrap_or_else(|error| panic!("MSL « {shader} » : {}", error.localizedDescription()))
            })
            .clone()
    }
}

// --- Ressources -----------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct MtlBuffer(Arc<BufferInner>);

struct BufferInner {
    raw: Obj<dyn MTLBuffer>,
    size: u64,
}

// SAFETY: objet Metal sans état CPU partagé ; ses écritures passent par le HAL.
unsafe impl Send for BufferInner {}
unsafe impl Sync for BufferInner {}

#[derive(Clone)]
pub(crate) struct MtlTexture(Arc<TextureInner>);

struct TextureInner {
    raw: Obj<dyn MTLTexture>,
    format: MTLPixelFormat,
    width: u32,
    height: u32,
}

// SAFETY: objet Metal sans état CPU partagé ; ses écritures passent par le HAL.
unsafe impl Send for TextureInner {}
unsafe impl Sync for TextureInner {}

#[derive(Clone)]
pub(crate) struct MtlTextureView(MtlTexture);

pub(crate) struct MtlSampler;

pub(crate) struct MtlBindGroupLayout(Arc<Vec<LayoutEntry>>);

#[derive(Clone)]
pub(crate) struct MtlBindGroup(Arc<Vec<GroupEntry>>);

struct GroupEntry {
    binding: u32,
    resource: GroupResource,
}

enum GroupResource {
    Buffer { buffer: MtlBuffer, offset: u64, dynamic: bool },
    Texture(MtlTexture),
    Sampler,
}

pub(crate) struct MtlPipeline(Arc<PipelineInner>);

struct PipelineInner {
    raw: Obj<dyn MTLRenderPipelineState>,
    primitive: MTLPrimitiveType,
}

// SAFETY: MTLRenderPipelineState est thread-safe.
unsafe impl Send for PipelineInner {}
unsafe impl Sync for PipelineInner {}

pub(crate) struct MtlEncoder {
    shared: Arc<Shared>,
    command_buffer: Obj<dyn MTLCommandBuffer>,
}

pub(crate) struct MtlPass<'a> {
    encoder: Obj<dyn MTLRenderCommandEncoder>,
    shared: &'a Shared,
    target: (u32, u32),
    primitive: MTLPrimitiveType,
}

impl Drop for MtlPass<'_> {
    fn drop(&mut self) {
        self.encoder.endEncoding();
    }
}

pub(crate) struct MtlSwapchain {
    layer: Retained<CAMetalLayer>,
    width: u32,
    height: u32,
    present_mode: WindowPresentMode,
    configured: bool,
}

pub(crate) struct MtlFrame {
    drawable: Retained<ProtocolObject<dyn CAMetalDrawable>>,
}

impl MetalGpu {
    /// Device Metal par défaut du système. Échec explicite sans GPU Metal.
    pub(crate) fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| anyhow!("aucun device Metal"))?;
        let queue = device.newCommandQueue().ok_or_else(|| anyhow!("queue Metal"))?;
        let descriptor = MTLSamplerDescriptor::new();
        descriptor.setMinFilter(MTLSamplerMinMagFilter::Linear);
        descriptor.setMagFilter(MTLSamplerMinMagFilter::Linear);
        descriptor.setMipFilter(MTLSamplerMipFilter::NotMipmapped);
        descriptor.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        descriptor.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        descriptor.setRAddressMode(MTLSamplerAddressMode::ClampToEdge);
        let sampler = device.newSamplerStateWithDescriptor(&descriptor).ok_or_else(|| anyhow!("sampler Metal"))?;
        let sizes = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("tampon des tailles Metal"))?;
        Ok(Self(Arc::new(Shared {
            device,
            queue,
            sampler,
            sizes,
            libraries: Mutex::new(HashMap::new()),
            pending: Mutex::new(Vec::new()),
            queue_lock: Mutex::new(()),
        })))
    }
}

/// Lie un groupe aux deux étages, à l'emplacement `groupe * BINDING_STRIDE + binding` de
/// chaque classe (buffers, textures, samplers).
fn bind(pass: &MtlPass<'_>, index: u32, group: &MtlBindGroup, offsets: &[u32]) {
    let mut dynamic = offsets.iter();
    for entry in group.0.iter() {
        let slot = (index * sources::BINDING_STRIDE + entry.binding) as usize;
        // SAFETY: encodeur ouvert ; emplacements dans les limites de Metal (< 31 buffers,
        // < 16 samplers pour ce schéma) ; ressources retenues par le command buffer.
        unsafe {
            match &entry.resource {
                GroupResource::Buffer { buffer, offset, dynamic: has_dynamic } => {
                    let extra = if *has_dynamic { u64::from(dynamic.next().copied().unwrap_or(0)) } else { 0 };
                    let offset = (offset + extra) as usize;
                    pass.encoder.setVertexBuffer_offset_atIndex(Some(&buffer.0.raw), offset, slot);
                    pass.encoder.setFragmentBuffer_offset_atIndex(Some(&buffer.0.raw), offset, slot);
                }
                GroupResource::Texture(texture) => {
                    pass.encoder.setVertexTexture_atIndex(Some(&texture.0.raw), slot);
                    pass.encoder.setFragmentTexture_atIndex(Some(&texture.0.raw), slot);
                }
                GroupResource::Sampler => {
                    pass.encoder.setVertexSamplerState_atIndex(Some(&pass.shared.sampler), slot);
                    pass.encoder.setFragmentSamplerState_atIndex(Some(&pass.shared.sampler), slot);
                }
            }
        }
    }
}

fn copy_texture(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    source: &ProtocolObject<dyn MTLTexture>,
    destination: &ProtocolObject<dyn MTLTexture>,
    width: u32,
    height: u32,
) {
    let blit = command_buffer.blitCommandEncoder().expect("encodeur de copie Metal");
    // SAFETY: région dans les deux textures, mêmes formats.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            source,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize { width: width as usize, height: height as usize, depth: 1 },
            destination,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
        );
    }
    blit.endEncoding();
}

impl Gpu for MetalGpu {
    type Format = MTLPixelFormat;
    type Buffer = MtlBuffer;
    type Texture = MtlTexture;
    type TextureView = MtlTextureView;
    type Sampler = MtlSampler;
    type BindGroupLayout = MtlBindGroupLayout;
    type BindGroup = MtlBindGroup;
    type Pipeline = MtlPipeline;
    type Encoder = MtlEncoder;
    type Pass<'a> = MtlPass<'a>;
    type Swapchain = MtlSwapchain;
    type Frame = MtlFrame;
    #[cfg(feature = "flamegraph")]
    type Profiler = super::NoProfiler;

    const ATLAS_MONOCHROME: MTLPixelFormat = MTLPixelFormat::R8Unorm;
    const ATLAS_POLYCHROME: MTLPixelFormat = MTLPixelFormat::RGBA8Unorm;

    fn bytes_per_pixel(format: MTLPixelFormat) -> u32 {
        match format {
            MTLPixelFormat::R8Unorm => 1,
            MTLPixelFormat::RGBA16Float => 8,
            _ => 4,
        }
    }

    fn gpu_specs(&self) -> GpuSpecs {
        GpuSpecs {
            is_software_emulated: false,
            device_name: self.0.device.name().to_string(),
            driver_name: "Metal".into(),
            driver_info: String::new(),
        }
    }

    fn min_uniform_offset_alignment(&self) -> u32 {
        256
    }

    fn create_buffer(&self, label: &str, size: u64, _usage: BufferUsage) -> MtlBuffer {
        let size = size.max(4);
        let raw = self
            .0
            .device
            .newBufferWithLength_options(size as usize, MTLResourceOptions::StorageModePrivate)
            .unwrap_or_else(|| panic!("buffer Metal « {label} »"));
        MtlBuffer(Arc::new(BufferInner { raw, size }))
    }

    fn create_buffer_init(&self, label: &str, contents: &[u8], usage: BufferUsage) -> MtlBuffer {
        let buffer = self.create_buffer(label, contents.len() as u64, usage);
        self.write_buffer(&buffer, 0, contents);
        buffer
    }

    fn buffer_size(buffer: &MtlBuffer) -> u64 {
        buffer.0.size
    }

    fn write_buffer(&self, buffer: &MtlBuffer, offset: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        self.0.pending.lock().push(Upload::Buffer { buffer: buffer.clone(), offset, data: data.to_vec() });
    }

    fn create_texture(&self, label: &str, width: u32, height: u32, format: MTLPixelFormat, usage: TextureUsage) -> MtlTexture {
        let (width, height) = (width.max(1), height.max(1));
        // SAFETY: descripteur 2D sans mip, format valide.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format,
                width as usize,
                height as usize,
                false,
            )
        };
        let mut texture_usage = MTLTextureUsage::ShaderRead;
        if usage.contains(TextureUsage::RENDER_TARGET) {
            texture_usage |= MTLTextureUsage::RenderTarget;
        }
        descriptor.setUsage(texture_usage);
        descriptor.setStorageMode(MTLStorageMode::Private);
        let raw = self.0.device.newTextureWithDescriptor(&descriptor).unwrap_or_else(|| panic!("texture Metal « {label} »"));
        MtlTexture(Arc::new(TextureInner { raw, format, width, height }))
    }

    fn texture_size(texture: &MtlTexture) -> (u32, u32) {
        (texture.0.width, texture.0.height)
    }

    fn create_view(texture: &MtlTexture) -> MtlTextureView {
        MtlTextureView(texture.clone())
    }

    fn write_texture(&self, texture: &MtlTexture, origin: (u32, u32), size: (u32, u32), bytes_per_pixel: u32, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        self.0.pending.lock().push(Upload::Texture {
            texture: texture.clone(),
            origin,
            size,
            bytes_per_pixel,
            data: data.to_vec(),
        });
    }

    fn create_linear_sampler(&self, _label: &str) -> MtlSampler {
        MtlSampler
    }

    fn create_bind_group_layout(&self, _label: &str, entries: &[LayoutEntry]) -> MtlBindGroupLayout {
        MtlBindGroupLayout(Arc::new(entries.to_vec()))
    }

    fn create_bind_group(&self, _label: &str, layout: &MtlBindGroupLayout, entries: &[BindEntry<'_, Self>]) -> MtlBindGroup {
        let resource_of = |binding: u32| entries.iter().find(|entry| entry.binding == binding).map(|entry| &entry.resource);
        let entries = layout
            .0
            .iter()
            .map(|entry| {
                let resource = match (entry.kind, resource_of(entry.binding)) {
                    (BindingKind::Uniform { dynamic_offset, .. }, Some(BindResource::Buffer { buffer, offset, .. })) => {
                        GroupResource::Buffer { buffer: (*buffer).clone(), offset: *offset, dynamic: dynamic_offset }
                    }
                    (BindingKind::Storage, Some(BindResource::Buffer { buffer, offset, .. })) => {
                        GroupResource::Buffer { buffer: (*buffer).clone(), offset: *offset, dynamic: false }
                    }
                    (BindingKind::Texture, Some(BindResource::Texture(view))) => GroupResource::Texture(view.0.clone()),
                    (BindingKind::Sampler, _) => GroupResource::Sampler,
                    (kind, _) => panic!("groupe Metal : ressource absente ou incompatible pour {kind:?} @binding({})", entry.binding),
                };
                GroupEntry { binding: entry.binding, resource }
            })
            .collect();
        MtlBindGroup(Arc::new(entries))
    }

    fn create_pipeline(&self, desc: &PipelineDesc<'_, Self>) -> MtlPipeline {
        let shader = desc.shader.name();
        let library = self.0.library(shader);
        let function = |entry: &str| {
            let name = sources::msl_entry(shader, entry).unwrap_or_else(|| panic!("MSL absent : {shader}::{entry}"));
            library.newFunctionWithName(&NSString::from_str(name)).unwrap_or_else(|| panic!("fonction MSL {shader}::{name}"))
        };
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&function(desc.vertex_entry)));
        descriptor.setFragmentFunction(Some(&function(desc.fragment_entry)));
        // SAFETY: l'attachement 0 existe toujours.
        let color = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        color.setPixelFormat(desc.format);
        color.setBlendingEnabled(true);
        // Mêmes équations que `wgpu::BlendState::{ALPHA_BLENDING, PREMULTIPLIED_ALPHA_BLENDING}`.
        color.setSourceRGBBlendFactor(match desc.blend {
            Blend::Alpha => MTLBlendFactor::SourceAlpha,
            Blend::PremultipliedAlpha => MTLBlendFactor::One,
        });
        color.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        color.setRgbBlendOperation(MTLBlendOperation::Add);
        color.setSourceAlphaBlendFactor(MTLBlendFactor::One);
        color.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        color.setAlphaBlendOperation(MTLBlendOperation::Add);
        let raw = self
            .0
            .device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .unwrap_or_else(|error| panic!("pipeline Metal « {} » : {}", desc.label, error.localizedDescription()));
        MtlPipeline(Arc::new(PipelineInner {
            raw,
            primitive: match desc.topology {
                Topology::TriangleList => MTLPrimitiveType::Triangle,
                Topology::TriangleStrip => MTLPrimitiveType::TriangleStrip,
            },
        }))
    }

    fn create_encoder(&self, _label: &str) -> MtlEncoder {
        MtlEncoder { shared: self.0.clone(), command_buffer: self.0.command_buffer() }
    }

    fn begin_pass<'a>(encoder: &'a mut MtlEncoder, desc: &PassDesc<'_, Self>) -> MtlPass<'a> {
        let encoder: &'a MtlEncoder = encoder;
        let target = &desc.target.0;
        let pass = MTLRenderPassDescriptor::new();
        // SAFETY: l'attachement 0 existe toujours.
        let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        color.setTexture(Some(&target.0.raw));
        match desc.load {
            LoadOp::Clear([red, green, blue, alpha]) => {
                color.setLoadAction(MTLLoadAction::Clear);
                color.setClearColor(MTLClearColor { red, green, blue, alpha });
            }
            LoadOp::Load => color.setLoadAction(MTLLoadAction::Load),
        }
        color.setStoreAction(MTLStoreAction::Store);
        let render = encoder.command_buffer.renderCommandEncoderWithDescriptor(&pass).expect("encodeur de rendu Metal");
        // SAFETY: encodeur ouvert, emplacement réservé par `build.rs`.
        unsafe {
            render.setVertexBuffer_offset_atIndex(Some(&encoder.shared.sizes), 0, sources::SIZES_BUFFER_SLOT);
            render.setFragmentBuffer_offset_atIndex(Some(&encoder.shared.sizes), 0, sources::SIZES_BUFFER_SLOT);
        }
        MtlPass {
            encoder: render,
            shared: &encoder.shared,
            target: (target.0.width, target.0.height),
            primitive: MTLPrimitiveType::Triangle,
        }
    }

    fn set_pipeline(pass: &mut MtlPass<'_>, pipeline: &MtlPipeline) {
        pass.encoder.setRenderPipelineState(&pipeline.0.raw);
        pass.primitive = pipeline.0.primitive;
    }

    fn set_bind_group(pass: &mut MtlPass<'_>, index: u32, group: &MtlBindGroup, dynamic_offsets: &[u32]) {
        bind(pass, index, group, dynamic_offsets);
    }

    fn set_viewport(pass: &mut MtlPass<'_>, x: f32, y: f32, width: f32, height: f32) {
        pass.encoder.setViewport(MTLViewport {
            originX: f64::from(x),
            originY: f64::from(y),
            width: f64::from(width),
            height: f64::from(height),
            znear: 0.0,
            zfar: 1.0,
        });
    }

    fn set_scissor_rect(pass: &mut MtlPass<'_>, x: u32, y: u32, width: u32, height: u32) {
        // Metal refuse un rectangle qui déborde de la cible.
        let x = x.min(pass.target.0);
        let y = y.min(pass.target.1);
        let width = width.min(pass.target.0 - x);
        let height = height.min(pass.target.1 - y);
        pass.encoder.setScissorRect(MTLScissorRect { x: x as usize, y: y as usize, width: width as usize, height: height as usize });
    }

    fn draw(pass: &mut MtlPass<'_>, vertices: Range<u32>, instances: Range<u32>) {
        // SAFETY: encodeur ouvert, pipeline lié par le renderer avant tout draw.
        unsafe {
            pass.encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                pass.primitive,
                vertices.start as usize,
                (vertices.end - vertices.start) as usize,
                (instances.end - instances.start) as usize,
                instances.start as usize,
            );
        }
    }

    fn copy_buffer_to_buffer(
        encoder: &mut MtlEncoder,
        source: &MtlBuffer,
        source_offset: u64,
        destination: &MtlBuffer,
        destination_offset: u64,
        size: u64,
    ) {
        let blit = encoder.command_buffer.blitCommandEncoder().expect("encodeur de copie Metal");
        // SAFETY: plages dans les buffers ; Metal ordonne la copie avec le reste du flux.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &source.0.raw,
                source_offset as usize,
                &destination.0.raw,
                destination_offset as usize,
                size as usize,
            );
        }
        blit.endEncoding();
    }

    fn copy_texture_to_texture(encoder: &mut MtlEncoder, source: &MtlTexture, destination: &MtlTexture, width: u32, height: u32) {
        copy_texture(&encoder.command_buffer, &source.0.raw, &destination.0.raw, width, height);
    }

    fn create_swapchain(
        &self,
        window: raw_window_handle::RawWindowHandle,
        _display: raw_window_handle::RawDisplayHandle,
        width: u32,
        height: u32,
    ) -> Result<MtlSwapchain> {
        let raw_window_handle::RawWindowHandle::AppKit(handle) = window else {
            anyhow::bail!("Metal exige une vue AppKit");
        };
        // SAFETY: NSView vivante (elle survit au renderer), manipulée sur le fil principal.
        let view: &AnyObject = unsafe { handle.ns_view.cast::<AnyObject>().as_ref() };
        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&self.0.device));
        layer.setPixelFormat(SWAPCHAIN_FORMAT);
        // Le framebuffer y est copié par blit.
        layer.setFramebufferOnly(false);
        // Fenêtre opaque, comme wgpu-Metal (pas d'alpha prémultiplié).
        layer.setOpaque(true);
        layer.setMaximumDrawableCount(MAXIMUM_DRAWABLES);
        // SAFETY: messages AppKit standard sur une NSView et sa fenêtre.
        unsafe {
            let window: Option<&AnyObject> = msg_send![view, window];
            if let Some(window) = window {
                let scale: f64 = msg_send![window, backingScaleFactor];
                layer.setContentsScale(scale);
            }
            let _: () = msg_send![view, setLayer: &*layer];
            let _: () = msg_send![view, setWantsLayer: true];
        }
        Ok(MtlSwapchain { layer, width, height, present_mode: WindowPresentMode::Fifo, configured: false })
    }

    fn swapchain_format(_swapchain: &MtlSwapchain) -> MTLPixelFormat {
        SWAPCHAIN_FORMAT
    }

    fn swapchain_premultiplied(_swapchain: &MtlSwapchain) -> bool {
        false
    }

    fn swapchain_size(swapchain: &MtlSwapchain) -> (u32, u32) {
        (swapchain.width, swapchain.height)
    }

    fn swapchain_present_mode(swapchain: &MtlSwapchain) -> WindowPresentMode {
        swapchain.present_mode
    }

    fn supported_present_modes(&self, _swapchain: &MtlSwapchain) -> Vec<WindowPresentMode> {
        vec![WindowPresentMode::Fifo, WindowPresentMode::Immediate]
    }

    fn swapchain_frame_latency(_swapchain: &MtlSwapchain) -> u32 {
        (MAXIMUM_DRAWABLES - 1) as u32
    }

    fn configure_swapchain(&self, swapchain: &mut MtlSwapchain, width: u32, height: u32, mode: WindowPresentMode) {
        let (width, height) = (width.max(1), height.max(1));
        swapchain.layer.setDrawableSize(CGSize { width: f64::from(width), height: f64::from(height) });
        swapchain.layer.setDisplaySyncEnabled(mode == WindowPresentMode::Fifo);
        swapchain.width = width;
        swapchain.height = height;
        swapchain.present_mode = mode;
        swapchain.configured = true;
    }

    fn acquire(&self, swapchain: &mut MtlSwapchain) -> Acquire<MtlFrame> {
        if !swapchain.configured {
            return Acquire::Outdated;
        }
        // `nextDrawable` rend un objet autoreleased : le pool le relâche après la reprise.
        match autoreleasepool(|_| swapchain.layer.nextDrawable()) {
            Some(drawable) => Acquire::Frame(MtlFrame { drawable }),
            None => Acquire::Skip("swap chain acquire timed out"),
        }
    }

    fn copy_texture_to_frame(encoder: &mut MtlEncoder, source: &MtlTexture, frame: &MtlFrame) {
        let target = frame.drawable.texture();
        let width = source.0.width.min(target.width() as u32);
        let height = source.0.height.min(target.height() as u32);
        copy_texture(&encoder.command_buffer, &source.0.raw, &target, width, height);
    }

    fn submit(&self, encoder: MtlEncoder) {
        self.0.flush_uploads();
        encoder.command_buffer.commit();
    }

    fn present(&self, frame: MtlFrame) {
        let command_buffer = self.0.command_buffer();
        command_buffer.presentDrawable(ProtocolObject::from_ref(&*frame.drawable));
        command_buffer.commit();
    }

    fn init_external_textures(&self, textures: [&MtlTexture; 3]) {
        self.0.flush_uploads();
        let command_buffer = self.0.command_buffer();
        // Contrat des surfaces 3D : effacées avant le premier échantillonnage.
        for texture in textures {
            let pass = MTLRenderPassDescriptor::new();
            // SAFETY: l'attachement 0 existe toujours.
            let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            color.setTexture(Some(&texture.0.raw));
            color.setLoadAction(MTLLoadAction::Clear);
            color.setClearColor(MTLClearColor { red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0 });
            color.setStoreAction(MTLStoreAction::Store);
            command_buffer.renderCommandEncoderWithDescriptor(&pass).expect("encodeur de rendu Metal").endEncoding();
        }
        command_buffer.commit();
    }

    fn surface_format(format: SurfaceFormat) -> MTLPixelFormat {
        match format {
            SurfaceFormat::Bgra8UnormSrgb => MTLPixelFormat::BGRA8Unorm_sRGB,
            SurfaceFormat::Rgba8UnormSrgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        }
    }

    fn queue_lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.0.queue_lock.lock()
    }

    fn native_device(&self) -> Option<NativeDevice> {
        Some(NativeDevice::Metal {
            device: Retained::as_ptr(&self.0.device) as *mut c_void,
            queue: Retained::as_ptr(&self.0.queue) as *mut c_void,
        })
    }

    fn native_texture(texture: &MtlTexture, _view: &MtlTextureView) -> Option<NativeTexture> {
        Some(NativeTexture::Metal(Retained::as_ptr(&texture.0.raw) as *mut c_void))
    }

    fn read_texture_bgra(&self, texture: &MtlTexture) -> Option<Vec<u8>> {
        self.0.flush_uploads();
        let (width, height) = (texture.0.width as usize, texture.0.height as usize);
        let row = width * 4;
        let buffer = self.0.device.newBufferWithLength_options(row * height, MTLResourceOptions::StorageModeShared)?;
        let command_buffer = self.0.command_buffer();
        let blit = command_buffer.blitCommandEncoder()?;
        // SAFETY: copie de toute la texture dans un tampon de taille exacte.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &texture.0.raw,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width, height, depth: 1 },
                &buffer,
                0,
                row,
                row * height,
            );
        }
        blit.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        // SAFETY: tampon partagé, copie terminée (attente ci-dessus).
        let mut pixels = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>().as_ptr(), row * height) }.to_vec();
        if matches!(texture.0.format, MTLPixelFormat::RGBA8Unorm | MTLPixelFormat::RGBA8Unorm_sRGB) {
            pixels.chunks_exact_mut(4).for_each(|pixel| pixel.swap(0, 2));
        }
        Some(pixels)
    }
}
