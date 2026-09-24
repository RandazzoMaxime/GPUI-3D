//! Moteur 3D en Vulkan natif (ash), sans wgpu : GPUI fournit ses `VkInstance`,
//! `VkDevice`, `VkQueue` et l'image du tampon arrière ; mémoire, commandes, fences,
//! render pass et pipeline sont à nous. Même `VkQueue` que le compositeur (sous
//! `native_queue_lock`) ⇒ l'ordre de soumission porte la synchronisation.

use std::io::Cursor;

use ash::vk::{self, Handle as _};
use gpui3d_shell::{
    Backends, CLEAR_COLOR, CUBE_INDICES, CUBE_VERTICES, NativeBackBuffer, NativeDevice, NativeTexture, Renderer, Scene,
    Surface,
};

/// = `gpui3d_shell::SURFACE_FORMAT`.
const COLOR_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
const FRAMES_IN_FLIGHT: usize = 2;

struct VulkanCube {
    device: ash::Device,
    memory: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    pool: vk::CommandPool,
    frames: Vec<Frame>,
    frame: usize,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    vertices: (vk::Buffer, vk::DeviceMemory),
    indices: (vk::Buffer, vk::DeviceMemory),
    target: Option<Target>,
}

struct Frame {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// Retient le tampon arrière tant que le GPU l'écrit.
    back: Option<NativeBackBuffer>,
}

struct Target {
    size: (u32, u32),
    depth: (vk::Image, vk::DeviceMemory, vk::ImageView),
    framebuffers: Vec<(vk::ImageView, vk::Framebuffer)>,
}

impl Renderer for VulkanCube {
    fn new(surface: &Surface) -> Self {
        let Some(NativeDevice::Vulkan { instance, physical_device, device, queue, queue_family_index }) =
            surface.native_device()
        else {
            panic!("GPUI ne tourne pas sur Vulkan (MoltenVK absent sur macOS ?)");
        };
        let entry = unsafe { ash::Entry::load() }.expect("chargeur Vulkan");
        let instance = unsafe { ash::Instance::load(entry.static_fn(), vk::Instance::from_raw(instance)) };
        let device = unsafe { ash::Device::load(instance.fp_v1_0(), vk::Device::from_raw(device)) };
        let memory =
            unsafe { instance.get_physical_device_memory_properties(vk::PhysicalDevice::from_raw(physical_device)) };
        unsafe {
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family_index)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .expect("command pool");
            let cmds = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(FRAMES_IN_FLIGHT as u32),
                )
                .expect("command buffers");
            let frames = cmds
                .into_iter()
                .map(|cmd| Frame {
                    cmd,
                    fence: device
                        .create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)
                        .expect("fence"),
                    back: None,
                })
                .collect();
            let render_pass = create_render_pass(&device);
            let push = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::VERTEX, offset: 0, size: 64 }];
            let layout = device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push), None)
                .expect("pipeline layout");
            let pipeline = create_pipeline(&device, render_pass, layout);
            let usage = vk::BufferUsageFlags::VERTEX_BUFFER;
            let vertices = host_buffer(&device, &memory, bytemuck::cast_slice(&CUBE_VERTICES), usage);
            let indices =
                host_buffer(&device, &memory, bytemuck::cast_slice(&CUBE_INDICES), vk::BufferUsageFlags::INDEX_BUFFER);
            Self {
                device,
                memory,
                queue: vk::Queue::from_raw(queue),
                pool,
                frames,
                frame: 0,
                render_pass,
                layout,
                pipeline,
                vertices,
                indices,
                target: None,
            }
        }
    }

    fn render(&mut self, surface: &Surface, scene: &Scene) -> bool {
        let d = &self.device;
        let fence = self.frames[self.frame].fence;
        unsafe { d.wait_for_fences(&[fence], true, u64::MAX) }.expect("fence");
        self.frames[self.frame].back = None;
        let Some(back) = surface.native_back_buffer() else { return false };
        let NativeTexture::Vulkan { view, .. } = back.texture else { return false };
        let (w, h) = back.size;
        if self.target.as_ref().map(|t| t.size) != Some((w, h)) {
            self.recreate_target(w, h);
        }
        let d = &self.device;
        let target = self.target.as_mut().expect("cible");
        let view = vk::ImageView::from_raw(view);
        let framebuffer = match target.framebuffers.iter().find(|(v, _)| *v == view) {
            Some(&(_, fb)) => fb,
            None => {
                let attachments = [view, target.depth.2];
                let info = vk::FramebufferCreateInfo::default()
                    .render_pass(self.render_pass)
                    .attachments(&attachments)
                    .width(w)
                    .height(h)
                    .layers(1);
                let fb = unsafe { d.create_framebuffer(&info, None) }.expect("framebuffer");
                target.framebuffers.push((view, fb));
                fb
            }
        };

        let cmd = self.frames[self.frame].cmd;
        let mvp = scene.mvp(w, h);
        unsafe {
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).expect("reset");
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin");
            let [r, g, b, a] = CLEAR_COLOR.map(|c| c as f32);
            let clears = [
                vk::ClearValue { color: vk::ClearColorValue { float32: [r, g, b, a] } },
                vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
            ];
            let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: w, height: h } };
            let begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(framebuffer)
                .render_area(area)
                .clear_values(&clears);
            d.cmd_begin_render_pass(cmd, &begin, vk::SubpassContents::INLINE);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            let viewport = vk::Viewport { x: 0.0, y: 0.0, width: w as f32, height: h as f32, min_depth: 0.0, max_depth: 1.0 };
            d.cmd_set_viewport(cmd, 0, &[viewport]);
            d.cmd_set_scissor(cmd, 0, &[area]);
            d.cmd_bind_vertex_buffers(cmd, 0, &[self.vertices.0], &[0]);
            d.cmd_bind_index_buffer(cmd, self.indices.0, 0, vk::IndexType::UINT16);
            d.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::VERTEX, 0, bytemuck::bytes_of(&mvp));
            d.cmd_draw_indexed(cmd, CUBE_INDICES.len() as u32, 1, 0, 0, 0);
            d.cmd_end_render_pass(cmd);
            d.end_command_buffer(cmd).expect("end");

            d.reset_fences(&[fence]).expect("reset fence");
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let _queue = surface.native_queue_lock();
            d.queue_submit(self.queue, &[submit], fence).expect("vkQueueSubmit");
        }
        self.frames[self.frame].back = Some(back);
        self.frame = (self.frame + 1) % FRAMES_IN_FLIGHT;
        // Soumis avant la publication : le compositeur échantillonnera après nous.
        surface.swap_buffers();
        true
    }
}

impl VulkanCube {
    fn recreate_target(&mut self, w: u32, h: u32) {
        let d = &self.device;
        let fences: Vec<_> = self.frames.iter().map(|f| f.fence).collect();
        unsafe { d.wait_for_fences(&fences, true, u64::MAX) }.expect("fences");
        if let Some(old) = self.target.take() {
            destroy_target(d, old);
        }
        self.target = Some(Target { size: (w, h), depth: create_depth(d, &self.memory, w, h), framebuffers: Vec::new() });
    }
}

impl Drop for VulkanCube {
    fn drop(&mut self) {
        let d = &self.device;
        unsafe {
            let fences: Vec<_> = self.frames.iter().map(|f| f.fence).collect();
            let _ = d.wait_for_fences(&fences, true, u64::MAX);
            if let Some(target) = self.target.take() {
                destroy_target(d, target);
            }
            for frame in &self.frames {
                d.destroy_fence(frame.fence, None);
            }
            d.destroy_command_pool(self.pool, None);
            for (buffer, memory) in [self.vertices, self.indices] {
                d.destroy_buffer(buffer, None);
                d.free_memory(memory, None);
            }
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_render_pass(self.render_pass, None);
        }
    }
}

fn destroy_target(d: &ash::Device, target: Target) {
    unsafe {
        for (_, fb) in target.framebuffers {
            d.destroy_framebuffer(fb, None);
        }
        let (image, memory, view) = target.depth;
        d.destroy_image_view(view, None);
        d.destroy_image(image, None);
        d.free_memory(memory, None);
    }
}

fn memory_type(props: &vk::PhysicalDeviceMemoryProperties, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
    (0..props.memory_type_count)
        .find(|&i| bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(flags))
}

unsafe fn host_buffer(
    d: &ash::Device,
    props: &vk::PhysicalDeviceMemoryProperties,
    bytes: &[u8],
    usage: vk::BufferUsageFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    unsafe {
        let buffer = d
            .create_buffer(&vk::BufferCreateInfo::default().size(bytes.len() as u64).usage(usage), None)
            .expect("buffer");
        let req = d.get_buffer_memory_requirements(buffer);
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let index = memory_type(props, req.memory_type_bits, flags).expect("mémoire host-visible");
        let memory = d
            .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(index), None)
            .expect("allocation");
        d.bind_buffer_memory(buffer, memory, 0).expect("bind");
        let ptr = d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).expect("map");
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len());
        d.unmap_memory(memory);
        (buffer, memory)
    }
}

fn create_depth(
    d: &ash::Device,
    props: &vk::PhysicalDeviceMemoryProperties,
    w: u32,
    h: u32,
) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
    unsafe {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(DEPTH_FORMAT)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT);
        let image = d.create_image(&info, None).expect("image profondeur");
        let req = d.get_image_memory_requirements(image);
        // Mémoire tuile (lazily allocated) quand le GPU l'offre : Apple, mobiles.
        let lazy = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::LAZILY_ALLOCATED;
        let index = memory_type(props, req.memory_type_bits, lazy)
            .or_else(|| memory_type(props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL))
            .expect("mémoire profondeur");
        let memory = d
            .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(index), None)
            .expect("allocation profondeur");
        d.bind_image_memory(image, memory, 0).expect("bind profondeur");
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::DEPTH).level_count(1).layer_count(1);
        let view = d
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(DEPTH_FORMAT)
                    .subresource_range(range),
                None,
            )
            .expect("vue profondeur");
        (image, memory, view)
    }
}

/// Contrat de layout du fork : le tampon arrive et repart en `SHADER_READ_ONLY_OPTIMAL`.
fn create_render_pass(device: &ash::Device) -> vk::RenderPass {
    let attachments = [
        vk::AttachmentDescription::default()
            .format(COLOR_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(DEPTH_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];
    let color_ref = [vk::AttachmentReference { attachment: 0, layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL }];
    let depth_ref = vk::AttachmentReference { attachment: 1, layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_ref)
        .depth_stencil_attachment(&depth_ref)];
    let dependencies = [
        // Entrée : le compositeur a fini de lire ce tampon, la trame d'avant d'écrire la profondeur.
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
        // Sortie : l'échantillonnage du compositeur (wgpu n'émet aucune barrière RESOURCE→RESOURCE).
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ),
    ];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    unsafe { device.create_render_pass(&info, None) }.expect("render pass")
}

fn create_pipeline(device: &ash::Device, render_pass: vk::RenderPass, layout: vk::PipelineLayout) -> vk::Pipeline {
    let module = |spv: &[u8]| {
        let code = ash::util::read_spv(&mut Cursor::new(spv)).expect("SPIR-V");
        unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None) }
            .expect("shader module")
    };
    let vs = module(include_bytes!(concat!(env!("OUT_DIR"), "/cube.vert.spv")));
    let fs = module(include_bytes!(concat!(env!("OUT_DIR"), "/cube.frag.spv")));
    let stages = [
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vs).name(c"main"),
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fs).name(c"main"),
    ];
    let bindings = [vk::VertexInputBindingDescription { binding: 0, stride: 24, input_rate: vk::VertexInputRate::VERTEX }];
    let attributes = [
        vk::VertexInputAttributeDescription { location: 0, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 0 },
        vk::VertexInputAttributeDescription { location: 1, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 12 },
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let input_assembly =
        vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass);
    let pipeline = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
        .map_err(|(_, e)| e)
        .expect("pipeline Vulkan")[0];
    unsafe {
        device.destroy_shader_module(vs, None);
        device.destroy_shader_module(fs, None);
    }
    pipeline
}

fn main() {
    gpui3d_shell::run::<VulkanCube>("Vulkan natif", Backends::VULKAN);
}
