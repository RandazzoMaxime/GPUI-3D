//! Implémentation Vulkan native (ash) de [`Gpu`], sans wgpu. Vulkan 1.3 : dynamic
//! rendering et timeline semaphore.
//!
//! Trois mécanismes tiennent les contrats de [`super`] :
//! - Enregistreurs : chaque command buffer note, par image, le premier et le dernier layout
//!   qu'il utilise. Au `submit`, un command buffer de correction amène chaque image de son
//!   layout réel (suivi par le device) au premier layout attendu. Les uploads, enregistrés
//!   à n'importe quel moment mais exécutés avant le command buffer principal, restent ainsi
//!   corrects, comme `queue.write_*` de wgpu.
//! - Rétention : un enregistreur garde un `Arc` de chaque ressource qu'il référence, relâché
//!   quand le timeline semaphore atteint son numéro de soumission. La destruction effective
//!   d'une ressource (Drop) ne peut donc jamais précéder la fin de son usage GPU.
//! - Passes différées : les textures échantillonnées ne sont connues qu'au `set_bind_group`,
//!   et aucune barrière n'est permise dans une passe ; la passe mémorise ses commandes et les
//!   rejoue à sa fin, après les barrières.
//!
//! Flip Y comme wgpu-hal : viewport de hauteur négative, shaders SPIR-V sans ajustement.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context as _, Result, anyhow};
use ash::vk;
use parking_lot::Mutex;

use super::{
    Acquire, BindEntry, BindResource, BindingKind, Blend, BufferUsage, Gpu, LayoutEntry, LoadOp,
    PassDesc, PipelineDesc, ShaderId, ShaderStages, TextureUsage, Topology,
};
use crate::{GpuSpecs, WindowPresentMode};

const STAGING_CHUNK: u64 = 4 * 1024 * 1024;
const DESCRIPTOR_POOL_SETS: u32 = 1024;

fn spirv(shader: ShaderId) -> &'static [u8] {
    match shader {
        ShaderId::Quads => include_bytes!(concat!(env!("OUT_DIR"), "/quads.spv")),
        ShaderId::Shadows => include_bytes!(concat!(env!("OUT_DIR"), "/shadows.spv")),
        ShaderId::BackdropBlur => include_bytes!(concat!(env!("OUT_DIR"), "/backdrop_blur.spv")),
        ShaderId::Underlines => include_bytes!(concat!(env!("OUT_DIR"), "/underlines.spv")),
        ShaderId::MonoSprites => include_bytes!(concat!(env!("OUT_DIR"), "/mono_sprites.spv")),
        ShaderId::PolySprites => include_bytes!(concat!(env!("OUT_DIR"), "/poly_sprites.spv")),
        ShaderId::Surfaces => include_bytes!(concat!(env!("OUT_DIR"), "/surfaces.spv")),
        ShaderId::Paths => include_bytes!(concat!(env!("OUT_DIR"), "/paths.spv")),
    }
}

type Retained = Arc<dyn Any + Send + Sync>;

/// Device Vulkan partagé par l'UI et les moteurs 3D d'une application.
#[derive(Clone)]
pub(crate) struct VulkanGpu(Arc<Shared>);

struct Shared {
    entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    driver_name: String,
    driver_info: String,
    device: ash::Device,
    swapchain_loader: ash::khr::swapchain::Device,
    queue: vk::Queue,
    queue_family: u32,
    /// `VkQueue` exige une synchronisation externe : tout submit, present et attente.
    queue_lock: Mutex<()>,
    timeline: vk::Semaphore,
    next_image_id: AtomicU64,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    submitted: u64,
    uploads: Option<Recorder>,
    staging: Vec<StagingChunk>,
    in_flight: VecDeque<InFlight>,
    free_command_buffers: Vec<(vk::CommandPool, vk::CommandBuffer)>,
    free_staging: Vec<StagingChunk>,
    image_layouts: HashMap<u64, vk::ImageLayout>,
    descriptor_pools: Vec<vk::DescriptorPool>,
}

struct InFlight {
    serial: u64,
    command_buffers: Vec<(vk::CommandPool, vk::CommandBuffer)>,
    staging: Vec<StagingChunk>,
    retained: Vec<Retained>,
}

struct StagingChunk {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    size: u64,
    used: u64,
}

// SAFETY: `mapped` pointe dans une mémoire hôte cohérente possédée par le chunk ; tout accès
// passe par le verrou de `State`.
unsafe impl Send for StagingChunk {}

struct ImageUse {
    image: vk::Image,
    first: vk::ImageLayout,
    last: vk::ImageLayout,
}

/// Command buffer en cours d'enregistrement et tout ce qui doit survivre à son exécution.
struct Recorder {
    pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    images: HashMap<u64, ImageUse>,
    retained: Vec<Retained>,
    frame: Option<FrameSync>,
}

struct FrameSync {
    acquire: vk::Semaphore,
    render_done: vk::Semaphore,
    slot: Arc<AtomicU64>,
}

fn layout_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .old_layout(from)
        .new_layout(to)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(color_range());
    // SAFETY: command buffer en enregistrement, image vivante (retenue par l'enregistreur).
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn global_barrier(device: &ash::Device, command_buffer: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    // SAFETY: command buffer en enregistrement.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

impl Recorder {
    fn use_image(&mut self, device: &ash::Device, texture: &VkTexture, layout: vk::ImageLayout) {
        let inner = &texture.0;
        match self.images.get_mut(&inner.id) {
            Some(usage) => {
                if usage.last != layout {
                    layout_barrier(device, self.command_buffer, usage.image, usage.last, layout);
                    usage.last = layout;
                }
            }
            None => {
                self.images.insert(inner.id, ImageUse { image: inner.raw, first: layout, last: layout });
            }
        }
        self.retained.push(texture.0.clone());
    }
}

impl Shared {
    fn find_memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32> {
        (0..self.memory_properties.memory_type_count)
            .find(|&index| {
                bits & (1 << index) != 0
                    && self.memory_properties.memory_types[index as usize].property_flags.contains(flags)
            })
            .ok_or_else(|| anyhow!("aucun type de mémoire Vulkan {flags:?}"))
    }

    fn completed(&self) -> u64 {
        // SAFETY: timeline créée avec le device.
        unsafe { self.device.get_semaphore_counter_value(self.timeline) }.unwrap_or(0)
    }

    fn wait_serial(&self, serial: u64) {
        if serial == 0 || self.completed() >= serial {
            return;
        }
        let semaphores = [self.timeline];
        let values = [serial];
        let info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
        // SAFETY: timeline valide ; attente sans limite d'une valeur déjà soumise.
        if let Err(error) = unsafe { self.device.wait_semaphores(&info, u64::MAX) } {
            log::error!("attente de la timeline Vulkan : {error}");
        }
    }

    fn new_recorder(&self, state: &mut State) -> Recorder {
        let (pool, command_buffer) = state.free_command_buffers.pop().unwrap_or_else(|| {
            // SAFETY: device valide ; pool et command buffer détruits avec le device.
            unsafe {
                let pool = self
                    .device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(self.queue_family)
                            .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                        None,
                    )
                    .expect("command pool Vulkan");
                let command_buffer = self
                    .device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .expect("command buffer Vulkan")[0];
                (pool, command_buffer)
            }
        });
        // SAFETY: command buffer au repos (neuf ou pool réinitialisé).
        unsafe {
            self.device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin command buffer");
        }
        Recorder { pool, command_buffer, images: HashMap::new(), retained: Vec::new(), frame: None }
    }

    /// Relâche ce que le GPU a fini d'utiliser. Les `Arc` sont rendus à l'appelant, qui les
    /// libère hors du verrou (leurs Drop le reprennent).
    fn collect_completed(&self, state: &mut State) -> Vec<Retained> {
        let completed = self.completed();
        let mut released = Vec::new();
        while state.in_flight.front().is_some_and(|in_flight| in_flight.serial <= completed) {
            let Some(in_flight) = state.in_flight.pop_front() else { break };
            for (pool, command_buffer) in in_flight.command_buffers {
                // SAFETY: le GPU a fini ce command buffer (timeline atteinte).
                if unsafe { self.device.reset_command_pool(pool, vk::CommandPoolResetFlags::empty()) }.is_ok() {
                    state.free_command_buffers.push((pool, command_buffer));
                }
            }
            for mut chunk in in_flight.staging {
                chunk.used = 0;
                state.free_staging.push(chunk);
            }
            released.extend(in_flight.retained);
        }
        released
    }

    fn staging_write(&self, state: &mut State, data: &[u8]) -> (vk::Buffer, u64) {
        let size = data.len() as u64;
        let fits = |chunk: &StagingChunk| chunk.used.next_multiple_of(16) + size <= chunk.size;
        if !state.staging.last().is_some_and(fits) {
            let reusable = state.free_staging.iter().position(|chunk| chunk.size >= size);
            let chunk = match reusable {
                Some(index) => state.free_staging.swap_remove(index),
                None => self.create_staging_chunk(size.max(STAGING_CHUNK)),
            };
            state.staging.push(chunk);
        }
        let Some(chunk) = state.staging.last_mut() else { unreachable!("chunk ajouté ci-dessus") };
        let offset = chunk.used.next_multiple_of(16);
        // SAFETY: `offset + size <= chunk.size`, mémoire mappée pour toute la vie du chunk.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), chunk.mapped.add(offset as usize), data.len()) };
        chunk.used = offset + size;
        (chunk.buffer, offset)
    }

    fn create_staging_chunk(&self, size: u64) -> StagingChunk {
        // SAFETY: device valide ; ressources libérées dans `Drop for Shared`.
        unsafe {
            let buffer = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    None,
                )
                .expect("buffer de staging");
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            let memory_type = self
                .find_memory_type(
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .expect("mémoire hôte cohérente");
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(memory_type),
                    None,
                )
                .expect("allocation du staging");
            self.device.bind_buffer_memory(buffer, memory, 0).expect("bind staging");
            let mapped = self
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .expect("map staging") as *mut u8;
            StagingChunk { buffer, memory, mapped, size, used: 0 }
        }
    }

    fn uploads<'a>(&self, state: &'a mut State) -> &'a mut Recorder {
        if state.uploads.is_none() {
            let recorder = self.new_recorder(state);
            // Écriture après lecture : les copies attendent les lectures des trames précédentes.
            global_barrier(&self.device, recorder.command_buffer);
            state.uploads = Some(recorder);
        }
        let Some(recorder) = state.uploads.as_mut() else { unreachable!("créé ci-dessus") };
        recorder
    }

    /// Soumet `main`, précédé des uploads en attente et des corrections de layout.
    fn submit(&self, main: Recorder) {
        let mut state = self.state.lock();
        let serial = state.submitted + 1;
        let mut command_buffers = Vec::new();
        let mut owned = Vec::new();
        let mut retained = Vec::new();
        let uploads = state.uploads.take();
        for mut recorder in uploads.into_iter().chain(std::iter::once(main)) {
            if !recorder.images.is_empty() {
                let fixup = self.new_recorder(&mut state);
                for (id, usage) in &recorder.images {
                    let current = state.image_layouts.get(id).copied().unwrap_or(vk::ImageLayout::UNDEFINED);
                    if current != usage.first {
                        layout_barrier(&self.device, fixup.command_buffer, usage.image, current, usage.first);
                    }
                    state.image_layouts.insert(*id, usage.last);
                }
                // SAFETY: enregistrement terminé.
                unsafe { self.device.end_command_buffer(fixup.command_buffer) }.expect("fin du command buffer");
                command_buffers.push(fixup.command_buffer);
                owned.push((fixup.pool, fixup.command_buffer));
            }
            // Les écritures de ce command buffer précèdent toute lecture du suivant.
            global_barrier(&self.device, recorder.command_buffer);
            // SAFETY: enregistrement terminé.
            unsafe { self.device.end_command_buffer(recorder.command_buffer) }.expect("fin du command buffer");
            command_buffers.push(recorder.command_buffer);
            owned.push((recorder.pool, recorder.command_buffer));
            retained.append(&mut recorder.retained);
            if let Some(frame) = recorder.frame.take() {
                frame.slot.store(serial, Ordering::Release);
                self.queue_submit(&command_buffers, &[frame.acquire], &[frame.render_done, self.timeline], serial);
                command_buffers.clear();
            }
        }
        if !command_buffers.is_empty() {
            self.queue_submit(&command_buffers, &[], &[self.timeline], serial);
        }
        state.submitted = serial;
        let staging = std::mem::take(&mut state.staging);
        state.in_flight.push_back(InFlight { serial, command_buffers: owned, staging, retained });
        let released = self.collect_completed(&mut state);
        drop(state);
        drop(released);
    }

    fn queue_submit(
        &self,
        command_buffers: &[vk::CommandBuffer],
        wait: &[vk::Semaphore],
        signal: &[vk::Semaphore],
        serial: u64,
    ) {
        let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; wait.len()];
        let wait_values = vec![0; wait.len()];
        let signal_values: Vec<u64> =
            signal.iter().map(|semaphore| if *semaphore == self.timeline { serial } else { 0 }).collect();
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let submit = vk::SubmitInfo::default()
            .command_buffers(command_buffers)
            .wait_semaphores(wait)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(signal)
            .push_next(&mut timeline_info);
        let _queue = self.queue_lock.lock();
        // SAFETY: queue protégée par `queue_lock`, command buffers terminés.
        unsafe { self.device.queue_submit(self.queue, &[submit], vk::Fence::null()) }.expect("vkQueueSubmit");
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        let state = self.state.get_mut();
        // SAFETY: plus aucune ressource ne référence le device (elles retiennent `Shared`).
        unsafe {
            {
                let _queue = self.queue_lock.lock();
                if let Err(error) = self.device.queue_wait_idle(self.queue) {
                    log::error!("attente de la queue Vulkan à la fermeture : {error}");
                }
            }
            if let Some(recorder) = state.uploads.take() {
                self.device.destroy_command_pool(recorder.pool, None);
            }
            for in_flight in state.in_flight.drain(..) {
                for (pool, _) in in_flight.command_buffers {
                    self.device.destroy_command_pool(pool, None);
                }
                state.free_staging.extend(in_flight.staging);
            }
            for (pool, _) in state.free_command_buffers.drain(..) {
                self.device.destroy_command_pool(pool, None);
            }
            for chunk in state.staging.drain(..).chain(state.free_staging.drain(..)) {
                self.device.destroy_buffer(chunk.buffer, None);
                self.device.free_memory(chunk.memory, None);
            }
            for pool in state.descriptor_pools.drain(..) {
                self.device.destroy_descriptor_pool(pool, None);
            }
            self.device.destroy_semaphore(self.timeline, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// --- Ressources -----------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct VkBuffer(Arc<BufferInner>);

struct BufferInner {
    shared: Arc<Shared>,
    raw: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        // SAFETY: plus aucun command buffer en vol ne le retient.
        unsafe {
            self.shared.device.destroy_buffer(self.raw, None);
            self.shared.device.free_memory(self.memory, None);
        }
    }
}

#[derive(Clone)]
pub(crate) struct VkTexture(Arc<TextureInner>);

struct TextureInner {
    shared: Arc<Shared>,
    id: u64,
    raw: vk::Image,
    memory: vk::DeviceMemory,
    format: vk::Format,
    width: u32,
    height: u32,
}

impl Drop for TextureInner {
    fn drop(&mut self) {
        self.shared.state.lock().image_layouts.remove(&self.id);
        // SAFETY: plus aucun command buffer en vol ne la retient.
        unsafe {
            self.shared.device.destroy_image(self.raw, None);
            self.shared.device.free_memory(self.memory, None);
        }
    }
}

#[derive(Clone)]
pub(crate) struct VkTextureView(Arc<ViewInner>);

struct ViewInner {
    texture: VkTexture,
    raw: vk::ImageView,
}

impl Drop for ViewInner {
    fn drop(&mut self) {
        // SAFETY: plus aucun command buffer en vol ni descriptor set vivant ne la retient.
        unsafe { self.texture.0.shared.device.destroy_image_view(self.raw, None) };
    }
}

pub(crate) struct VkSampler(Arc<SamplerInner>);

struct SamplerInner {
    shared: Arc<Shared>,
    raw: vk::Sampler,
}

impl Drop for SamplerInner {
    fn drop(&mut self) {
        // SAFETY: plus aucun descriptor set vivant ne le retient.
        unsafe { self.shared.device.destroy_sampler(self.raw, None) };
    }
}

pub(crate) struct VkBindGroupLayout(Arc<LayoutInner>);

struct LayoutInner {
    shared: Arc<Shared>,
    raw: vk::DescriptorSetLayout,
    entries: Vec<LayoutEntry>,
}

impl Drop for LayoutInner {
    fn drop(&mut self) {
        // SAFETY: les pipelines et descriptor sets qui l'utilisent la retiennent.
        unsafe { self.shared.device.destroy_descriptor_set_layout(self.raw, None) };
    }
}

#[derive(Clone)]
pub(crate) struct VkBindGroup(Arc<GroupInner>);

struct GroupInner {
    shared: Arc<Shared>,
    pool: vk::DescriptorPool,
    raw: vk::DescriptorSet,
    /// Textures échantillonnées, à passer en `SHADER_READ_ONLY_OPTIMAL` avant la passe.
    sampled: Vec<VkTexture>,
    _resources: Vec<Retained>,
}

impl Drop for GroupInner {
    fn drop(&mut self) {
        let _state = self.shared.state.lock();
        // SAFETY: pool synchronisé par le verrou d'état ; plus aucun command buffer en vol.
        if let Err(error) = unsafe { self.shared.device.free_descriptor_sets(self.pool, &[self.raw]) } {
            log::error!("libération d'un descriptor set : {error}");
        }
    }
}

pub(crate) struct VkPipeline(Arc<PipelineInner>);

struct PipelineInner {
    shared: Arc<Shared>,
    raw: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_count: u32,
    _set_layouts: Vec<Arc<LayoutInner>>,
}

impl Drop for PipelineInner {
    fn drop(&mut self) {
        // SAFETY: plus aucun command buffer en vol ne le retient.
        unsafe {
            self.shared.device.destroy_pipeline(self.raw, None);
            self.shared.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

pub(crate) struct VkEncoder {
    shared: Arc<Shared>,
    recorder: Option<Recorder>,
}

impl VkEncoder {
    fn recorder(&mut self) -> &mut Recorder {
        let Some(recorder) = self.recorder.as_mut() else { unreachable!("encodeur consommé par submit") };
        recorder
    }
}

impl Drop for VkEncoder {
    fn drop(&mut self) {
        let Some(recorder) = self.recorder.take() else { return };
        // Trame abandonnée (acquisition ratée) : jamais soumis, donc au repos.
        // SAFETY: command buffer jamais soumis.
        unsafe {
            if self.shared.device.end_command_buffer(recorder.command_buffer).is_ok()
                && self
                    .shared
                    .device
                    .reset_command_pool(recorder.pool, vk::CommandPoolResetFlags::empty())
                    .is_ok()
            {
                self.shared.state.lock().free_command_buffers.push((recorder.pool, recorder.command_buffer));
            }
        }
    }
}

enum PassCommand {
    SetPipeline(Arc<PipelineInner>),
    SetBindGroup(u32, VkBindGroup, Vec<u32>),
    SetViewport(f32, f32, f32, f32),
    SetScissor(vk::Rect2D),
    Draw(Range<u32>, Range<u32>),
}

pub(crate) struct VkPass<'a> {
    encoder: &'a mut VkEncoder,
    target: VkTextureView,
    load: LoadOp,
    commands: Vec<PassCommand>,
}

impl Drop for VkPass<'_> {
    fn drop(&mut self) {
        let device = self.encoder.shared.device.clone();
        let commands = std::mem::take(&mut self.commands);
        let target = self.target.clone();
        let load = self.load;
        let recorder = self.encoder.recorder();

        for command in &commands {
            if let PassCommand::SetBindGroup(_, group, _) = command {
                for texture in &group.0.sampled {
                    recorder.use_image(&device, texture, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                }
            }
        }
        recorder.use_image(&device, &target.0.texture, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        global_barrier(&device, recorder.command_buffer);
        recorder.retained.push(target.0.clone());

        let (width, height) = (target.0.texture.0.width, target.0.texture.0.height);
        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width, height } };
        let (load_op, clear) = match load {
            LoadOp::Clear([r, g, b, a]) => (
                vk::AttachmentLoadOp::CLEAR,
                vk::ClearValue { color: vk::ClearColorValue { float32: [r as f32, g as f32, b as f32, a as f32] } },
            ),
            LoadOp::Load => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
        };
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(target.0.raw)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear)];
        let rendering = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&attachments);
        let command_buffer = recorder.command_buffer;
        // SAFETY: command buffer en enregistrement ; toutes les ressources utilisées sont
        // retenues par l'enregistreur jusqu'à la fin de leur usage GPU.
        unsafe {
            device.cmd_begin_rendering(command_buffer, &rendering);
            device.cmd_set_viewport(command_buffer, 0, &[flipped_viewport(0.0, 0.0, width as f32, height as f32)]);
            device.cmd_set_scissor(command_buffer, 0, &[area]);
            let mut pipeline: Option<Arc<PipelineInner>> = None;
            let mut bound: [Option<(VkBindGroup, Vec<u32>)>; 8] = Default::default();
            let mut dirty = [false; 8];
            for command in commands {
                match command {
                    PassCommand::SetPipeline(next) => {
                        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, next.raw);
                        dirty = [true; 8];
                        recorder.retained.push(next.clone());
                        pipeline = Some(next);
                    }
                    PassCommand::SetBindGroup(index, group, offsets) => {
                        if let Some(slot) = bound.get_mut(index as usize) {
                            recorder.retained.push(group.0.clone());
                            *slot = Some((group, offsets));
                            dirty[index as usize] = true;
                        }
                    }
                    PassCommand::SetViewport(x, y, width, height) => {
                        device.cmd_set_viewport(command_buffer, 0, &[flipped_viewport(x, y, width, height)]);
                    }
                    PassCommand::SetScissor(rect) => device.cmd_set_scissor(command_buffer, 0, &[rect]),
                    PassCommand::Draw(vertices, instances) => {
                        let Some(pipeline) = pipeline.as_ref() else {
                            log::error!("draw Vulkan sans pipeline");
                            continue;
                        };
                        for index in 0..pipeline.set_count as usize {
                            if !dirty[index] {
                                continue;
                            }
                            if let Some((group, offsets)) = &bound[index] {
                                device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    pipeline.layout,
                                    index as u32,
                                    &[group.0.raw],
                                    offsets,
                                );
                                dirty[index] = false;
                            }
                        }
                        device.cmd_draw(
                            command_buffer,
                            vertices.end - vertices.start,
                            instances.end - instances.start,
                            vertices.start,
                            instances.start,
                        );
                    }
                }
            }
            device.cmd_end_rendering(command_buffer);
        }
    }
}

/// Viewport de hauteur négative : l'espace clip WebGPU (Y vers le haut) sur le cadre Vulkan.
fn flipped_viewport(x: f32, y: f32, width: f32, height: f32) -> vk::Viewport {
    vk::Viewport { x, y: y + height, width, height: -height, min_depth: 0.0, max_depth: 1.0 }
}

pub(crate) struct VkSwapchain {
    shared: Arc<Shared>,
    surface: vk::SurfaceKHR,
    raw: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    render_done: Vec<vk::Semaphore>,
    acquire: Vec<(vk::Semaphore, Arc<AtomicU64>)>,
    next_acquire: usize,
    format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    composite_alpha: vk::CompositeAlphaFlagsKHR,
    extent: vk::Extent2D,
    present_mode: WindowPresentMode,
}

pub(crate) struct VkFrame {
    swapchain: vk::SwapchainKHR,
    image: vk::Image,
    index: u32,
    extent: vk::Extent2D,
    acquire: vk::Semaphore,
    render_done: vk::Semaphore,
    slot: Arc<AtomicU64>,
}

impl VkSwapchain {
    fn destroy_images(&mut self) {
        // SAFETY: appelé queue au repos.
        unsafe {
            for semaphore in self.render_done.drain(..) {
                self.shared.device.destroy_semaphore(semaphore, None);
            }
        }
        self.images.clear();
    }
}

impl Drop for VkSwapchain {
    fn drop(&mut self) {
        {
            let _queue = self.shared.queue_lock.lock();
            // SAFETY: queue protégée ; attendre que les présentations en cours se terminent.
            if let Err(error) = unsafe { self.shared.device.queue_wait_idle(self.shared.queue) } {
                log::error!("attente de la queue Vulkan : {error}");
            }
        }
        self.destroy_images();
        // SAFETY: queue au repos, plus aucune image de cette swapchain en usage.
        unsafe {
            for (semaphore, _) in self.acquire.drain(..) {
                self.shared.device.destroy_semaphore(semaphore, None);
            }
            if self.raw != vk::SwapchainKHR::null() {
                self.shared.swapchain_loader.destroy_swapchain(self.raw, None);
            }
            self.shared.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

fn present_mode(mode: WindowPresentMode) -> vk::PresentModeKHR {
    match mode {
        WindowPresentMode::Fifo => vk::PresentModeKHR::FIFO,
        WindowPresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        WindowPresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
    }
}

fn is_srgb(format: vk::Format) -> bool {
    matches!(format, vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB | vk::Format::A8B8G8R8_SRGB_PACK32)
}

fn descriptor_type(kind: BindingKind) -> vk::DescriptorType {
    match kind {
        BindingKind::Uniform { dynamic_offset: true, .. } => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        BindingKind::Uniform { dynamic_offset: false, .. } => vk::DescriptorType::UNIFORM_BUFFER,
        BindingKind::Storage => vk::DescriptorType::STORAGE_BUFFER,
        BindingKind::Texture => vk::DescriptorType::SAMPLED_IMAGE,
        BindingKind::Sampler => vk::DescriptorType::SAMPLER,
    }
}

fn allocate_descriptor_set(
    shared: &Shared,
    state: &mut State,
    layout: vk::DescriptorSetLayout,
) -> Result<(vk::DescriptorPool, vk::DescriptorSet)> {
    let layouts = [layout];
    if let Some(&pool) = state.descriptor_pools.last() {
        // SAFETY: pool synchronisé par le verrou d'état.
        if let Ok(sets) = unsafe {
            shared.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts),
            )
        } {
            return Ok((pool, sets[0]));
        }
    }
    let sizes = [
        vk::DescriptorType::UNIFORM_BUFFER,
        vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        vk::DescriptorType::STORAGE_BUFFER,
        vk::DescriptorType::SAMPLED_IMAGE,
        vk::DescriptorType::SAMPLER,
    ]
    .map(|ty| vk::DescriptorPoolSize { ty, descriptor_count: DESCRIPTOR_POOL_SETS * 2 });
    // SAFETY: device valide ; pool détruit avec `Shared`.
    let pool = unsafe {
        shared.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                .max_sets(DESCRIPTOR_POOL_SETS)
                .pool_sizes(&sizes),
            None,
        )
    }?;
    state.descriptor_pools.push(pool);
    // SAFETY: pool neuf, synchronisé par le verrou d'état.
    let sets = unsafe {
        shared
            .device
            .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))
    }?;
    Ok((pool, sets[0]))
}

// --- Création du device ---------------------------------------------------------------

impl VulkanGpu {
    /// Instance et device Vulkan 1.3 (dynamic rendering, timeline semaphore) sur le GPU
    /// discret de préférence. Échec explicite sans pilote ou sans GPU compatible.
    pub(crate) fn new() -> Result<Self> {
        // SAFETY: chargement dynamique du chargeur Vulkan du système.
        let entry = unsafe { ash::Entry::load() }.context("chargeur Vulkan introuvable")?;
        // SAFETY: entry valide.
        let available = unsafe { entry.enumerate_instance_extension_properties(None) }?;
        let has = |name: &CStr| available.iter().any(|extension| extension.extension_name_as_c_str() == Ok(name));
        let wanted: [&CStr; 7] = [
            ash::khr::surface::NAME,
            ash::khr::win32_surface::NAME,
            ash::khr::xlib_surface::NAME,
            ash::khr::xcb_surface::NAME,
            ash::khr::wayland_surface::NAME,
            ash::ext::metal_surface::NAME,
            ash::khr::portability_enumeration::NAME,
        ];
        let extensions: Vec<*const std::ffi::c_char> =
            wanted.iter().filter(|name| has(name)).map(|name| name.as_ptr()).collect();
        anyhow::ensure!(has(ash::khr::surface::NAME), "Vulkan sans VK_KHR_surface");
        let flags = if has(ash::khr::portability_enumeration::NAME) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };
        let application = vk::ApplicationInfo::default()
            .application_name(c"gpui")
            .engine_name(c"gpui")
            .api_version(vk::API_VERSION_1_3);
        // SAFETY: descripteurs valides pendant l'appel.
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&application)
                    .enabled_extension_names(&extensions)
                    .flags(flags),
                None,
            )
        }?;
        let result = Self::with_instance(entry, instance.clone());
        if result.is_err() {
            // SAFETY: aucun objet enfant n'a survécu à l'échec.
            unsafe { instance.destroy_instance(None) };
        }
        result
    }

    fn with_instance(entry: ash::Entry, instance: ash::Instance) -> Result<Self> {
        // SAFETY: instance valide.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        let candidates: Vec<(vk::PhysicalDevice, vk::PhysicalDeviceProperties, u32)> = physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                // SAFETY: handles issus de l'instance.
                let properties = unsafe { instance.get_physical_device_properties(physical_device) };
                if properties.api_version < vk::API_VERSION_1_3 {
                    return None;
                }
                // SAFETY: handle issu de l'instance.
                let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
                let family = families.iter().position(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))?;
                Some((physical_device, properties, family as u32))
            })
            .collect();
        let (physical_device, properties, queue_family) = candidates
            .iter()
            .copied()
            .min_by_key(|(_, properties, _)| properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU)
            .ok_or_else(|| anyhow!("aucun GPU Vulkan 1.3 avec une queue graphique"))?;

        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        // SAFETY: chaîne pNext valide pendant l'appel.
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };
        let driver_name =
            driver.driver_name_as_c_str().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        let driver_info =
            driver.driver_info_as_c_str().map(|info| info.to_string_lossy().into_owned()).unwrap_or_default();
        // SAFETY: handle issu de l'instance.
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // SAFETY: handle issu de l'instance.
        let available = unsafe { instance.enumerate_device_extension_properties(physical_device) }?;
        let has = |name: &CStr| available.iter().any(|extension| extension.extension_name_as_c_str() == Ok(name));
        anyhow::ensure!(has(ash::khr::swapchain::NAME), "GPU Vulkan sans VK_KHR_swapchain");
        let mut device_extensions = vec![ash::khr::swapchain::NAME.as_ptr()];
        if has(ash::khr::portability_subset::NAME) {
            device_extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(&priorities)];
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        // SAFETY: descripteurs valides pendant l'appel.
        let device = unsafe {
            instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&device_extensions)
                    .push_next(&mut vulkan12)
                    .push_next(&mut vulkan13),
                None,
            )
        }
        .context("création du device Vulkan (dynamic rendering, timeline semaphore)")?;
        // SAFETY: famille et index créés ci-dessus.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let mut timeline_type =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        // SAFETY: device valide.
        let timeline =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut timeline_type), None) }?;
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        Ok(Self(Arc::new(Shared {
            entry,
            instance,
            surface_loader,
            physical_device,
            properties,
            memory_properties,
            driver_name,
            driver_info,
            device,
            swapchain_loader,
            queue,
            queue_family,
            queue_lock: Mutex::new(()),
            timeline,
            next_image_id: AtomicU64::new(1),
            state: Mutex::new(State::default()),
        })))
    }

    fn create_raw_buffer(&self, label: &str, size: u64, usage: BufferUsage) -> VkBuffer {
        let shared = &self.0;
        let mut flags = vk::BufferUsageFlags::TRANSFER_DST;
        for (flag, vk_flag) in [
            (BufferUsage::UNIFORM, vk::BufferUsageFlags::UNIFORM_BUFFER),
            (BufferUsage::STORAGE, vk::BufferUsageFlags::STORAGE_BUFFER),
            (BufferUsage::VERTEX, vk::BufferUsageFlags::VERTEX_BUFFER),
            (BufferUsage::COPY_SRC, vk::BufferUsageFlags::TRANSFER_SRC),
        ] {
            if usage.contains(flag) {
                flags |= vk_flag;
            }
        }
        let size = size.max(4);
        // SAFETY: device valide ; libéré par `Drop for BufferInner`.
        unsafe {
            let raw = shared
                .device
                .create_buffer(&vk::BufferCreateInfo::default().size(size).usage(flags), None)
                .unwrap_or_else(|error| panic!("buffer Vulkan « {label} » : {error}"));
            let requirements = shared.device.get_buffer_memory_requirements(raw);
            let memory_type = shared
                .find_memory_type(requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .unwrap_or_else(|error| panic!("buffer « {label} » : {error}"));
            let memory = shared
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(memory_type),
                    None,
                )
                .unwrap_or_else(|error| panic!("mémoire du buffer « {label} » : {error}"));
            shared.device.bind_buffer_memory(raw, memory, 0).expect("bind buffer");
            VkBuffer(Arc::new(BufferInner { shared: shared.clone(), raw, memory, size }))
        }
    }
}

impl Gpu for VulkanGpu {
    type Format = vk::Format;
    type Buffer = VkBuffer;
    type Texture = VkTexture;
    type TextureView = VkTextureView;
    type Sampler = VkSampler;
    type BindGroupLayout = VkBindGroupLayout;
    type BindGroup = VkBindGroup;
    type Pipeline = VkPipeline;
    type Encoder = VkEncoder;
    type Pass<'a> = VkPass<'a>;
    type Swapchain = VkSwapchain;
    type Frame = VkFrame;
    #[cfg(feature = "flamegraph")]
    type Profiler = super::NoProfiler;

    const ATLAS_MONOCHROME: vk::Format = vk::Format::R8_UNORM;
    const ATLAS_POLYCHROME: vk::Format = vk::Format::R8G8B8A8_UNORM;

    fn bytes_per_pixel(format: vk::Format) -> u32 {
        match format {
            vk::Format::R8_UNORM => 1,
            vk::Format::R16G16B16A16_SFLOAT => 8,
            _ => 4,
        }
    }

    fn gpu_specs(&self) -> GpuSpecs {
        let properties = &self.0.properties;
        GpuSpecs {
            is_software_emulated: properties.device_type == vk::PhysicalDeviceType::CPU,
            device_name: properties
                .device_name_as_c_str()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            driver_name: self.0.driver_name.clone(),
            driver_info: self.0.driver_info.clone(),
        }
    }

    fn min_uniform_offset_alignment(&self) -> u32 {
        self.0.properties.limits.min_uniform_buffer_offset_alignment as u32
    }

    fn create_buffer(&self, label: &str, size: u64, usage: BufferUsage) -> VkBuffer {
        self.create_raw_buffer(label, size, usage)
    }

    fn create_buffer_init(&self, label: &str, contents: &[u8], usage: BufferUsage) -> VkBuffer {
        let buffer = self.create_raw_buffer(label, contents.len() as u64, usage);
        self.write_buffer(&buffer, 0, contents);
        buffer
    }

    fn buffer_size(buffer: &VkBuffer) -> u64 {
        buffer.0.size
    }

    fn write_buffer(&self, buffer: &VkBuffer, offset: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let shared = &self.0;
        let mut state = shared.state.lock();
        let (staging, staging_offset) = shared.staging_write(&mut state, data);
        let recorder = shared.uploads(&mut state);
        let region = vk::BufferCopy { src_offset: staging_offset, dst_offset: offset, size: data.len() as u64 };
        // SAFETY: command buffer d'upload en enregistrement, sous le verrou d'état.
        unsafe { shared.device.cmd_copy_buffer(recorder.command_buffer, staging, buffer.0.raw, &[region]) };
        recorder.retained.push(buffer.0.clone());
    }

    fn create_texture(&self, label: &str, width: u32, height: u32, format: vk::Format, usage: TextureUsage) -> VkTexture {
        let shared = &self.0;
        let mut flags = vk::ImageUsageFlags::empty();
        for (flag, vk_flag) in [
            (TextureUsage::RENDER_TARGET, vk::ImageUsageFlags::COLOR_ATTACHMENT),
            (TextureUsage::SAMPLED, vk::ImageUsageFlags::SAMPLED),
            (TextureUsage::COPY_SRC, vk::ImageUsageFlags::TRANSFER_SRC),
            (TextureUsage::COPY_DST, vk::ImageUsageFlags::TRANSFER_DST),
        ] {
            if usage.contains(flag) {
                flags |= vk_flag;
            }
        }
        // Effacement des surfaces 3D et uploads d'atlas : copies vers l'image.
        flags |= vk::ImageUsageFlags::TRANSFER_DST;
        let (width, height) = (width.max(1), height.max(1));
        // SAFETY: device valide ; libéré par `Drop for TextureInner`.
        unsafe {
            let raw = shared
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D { width, height, depth: 1 })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(flags)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .unwrap_or_else(|error| panic!("texture Vulkan « {label} » : {error}"));
            let requirements = shared.device.get_image_memory_requirements(raw);
            let memory_type = shared
                .find_memory_type(requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .unwrap_or_else(|error| panic!("texture « {label} » : {error}"));
            let memory = shared
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(memory_type),
                    None,
                )
                .unwrap_or_else(|error| panic!("mémoire de la texture « {label} » : {error}"));
            shared.device.bind_image_memory(raw, memory, 0).expect("bind image");
            VkTexture(Arc::new(TextureInner {
                shared: shared.clone(),
                id: shared.next_image_id.fetch_add(1, Ordering::Relaxed),
                raw,
                memory,
                format,
                width,
                height,
            }))
        }
    }

    fn texture_size(texture: &VkTexture) -> (u32, u32) {
        (texture.0.width, texture.0.height)
    }

    fn create_view(texture: &VkTexture) -> VkTextureView {
        let shared = &texture.0.shared;
        // SAFETY: image vivante (retenue par la vue).
        let raw = unsafe {
            shared.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(texture.0.raw)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(texture.0.format)
                    .subresource_range(color_range()),
                None,
            )
        }
        .expect("vue de texture Vulkan");
        VkTextureView(Arc::new(ViewInner { texture: texture.clone(), raw }))
    }

    fn write_texture(
        &self,
        texture: &VkTexture,
        origin: (u32, u32),
        size: (u32, u32),
        _bytes_per_pixel: u32,
        data: &[u8],
    ) {
        if data.is_empty() {
            return;
        }
        let shared = &self.0;
        let mut state = shared.state.lock();
        let (staging, staging_offset) = shared.staging_write(&mut state, data);
        let recorder = shared.uploads(&mut state);
        recorder.use_image(&shared.device, texture, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let region = vk::BufferImageCopy::default()
            .buffer_offset(staging_offset)
            .image_subresource(
                vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1),
            )
            .image_offset(vk::Offset3D { x: origin.0 as i32, y: origin.1 as i32, z: 0 })
            .image_extent(vk::Extent3D { width: size.0, height: size.1, depth: 1 });
        // SAFETY: command buffer d'upload en enregistrement, sous le verrou d'état.
        unsafe {
            shared.device.cmd_copy_buffer_to_image(
                recorder.command_buffer,
                staging,
                texture.0.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
    }

    fn create_linear_sampler(&self, _label: &str) -> VkSampler {
        let shared = &self.0;
        // SAFETY: device valide ; libéré par `Drop for SamplerInner`.
        let raw = unsafe {
            shared.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(32.0),
                None,
            )
        }
        .expect("sampler Vulkan");
        VkSampler(Arc::new(SamplerInner { shared: shared.clone(), raw }))
    }

    fn create_bind_group_layout(&self, label: &str, entries: &[LayoutEntry]) -> VkBindGroupLayout {
        let shared = &self.0;
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = entries
            .iter()
            .map(|entry| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(entry.binding)
                    .descriptor_type(descriptor_type(entry.kind))
                    .descriptor_count(1)
                    .stage_flags(match entry.visibility {
                        ShaderStages::Vertex => vk::ShaderStageFlags::VERTEX,
                        ShaderStages::Fragment => vk::ShaderStageFlags::FRAGMENT,
                        ShaderStages::VertexFragment => vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    })
            })
            .collect();
        // SAFETY: device valide.
        let raw = unsafe {
            shared
                .device
                .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)
        }
        .unwrap_or_else(|error| panic!("layout Vulkan « {label} » : {error}"));
        VkBindGroupLayout(Arc::new(LayoutInner { shared: shared.clone(), raw, entries: entries.to_vec() }))
    }

    fn create_bind_group(&self, label: &str, layout: &VkBindGroupLayout, entries: &[BindEntry<'_, Self>]) -> VkBindGroup {
        let shared = &self.0;
        let (pool, raw) = {
            let mut state = shared.state.lock();
            allocate_descriptor_set(shared, &mut state, layout.0.raw)
                .unwrap_or_else(|error| panic!("descriptor set « {label} » : {error}"))
        };
        let mut buffer_infos = Vec::new();
        let mut image_infos = Vec::new();
        let mut sampled = Vec::new();
        let mut resources: Vec<Retained> = vec![layout.0.clone()];
        for entry in entries {
            match &entry.resource {
                BindResource::Buffer { buffer, offset, size } => {
                    buffer_infos.push((
                        entry.binding,
                        vk::DescriptorBufferInfo {
                            buffer: buffer.0.raw,
                            offset: *offset,
                            range: size.unwrap_or(vk::WHOLE_SIZE),
                        },
                    ));
                    resources.push(buffer.0.clone());
                }
                BindResource::Texture(view) => {
                    image_infos.push((
                        entry.binding,
                        vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: view.0.raw,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                    ));
                    sampled.push(view.0.texture.clone());
                    resources.push(view.0.clone());
                }
                BindResource::Sampler(sampler) => {
                    image_infos.push((
                        entry.binding,
                        vk::DescriptorImageInfo {
                            sampler: sampler.0.raw,
                            image_view: vk::ImageView::null(),
                            image_layout: vk::ImageLayout::UNDEFINED,
                        },
                    ));
                    resources.push(sampler.0.clone());
                }
            }
        }
        let kind_of =
            |binding: u32| layout.0.entries.iter().find(|entry| entry.binding == binding).map(|entry| entry.kind);
        let mut writes = Vec::new();
        for (binding, info) in &buffer_infos {
            let Some(kind) = kind_of(*binding) else { continue };
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(raw)
                    .dst_binding(*binding)
                    .descriptor_type(descriptor_type(kind))
                    .buffer_info(std::slice::from_ref(info)),
            );
        }
        for (binding, info) in &image_infos {
            let Some(kind) = kind_of(*binding) else { continue };
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(raw)
                    .dst_binding(*binding)
                    .descriptor_type(descriptor_type(kind))
                    .image_info(std::slice::from_ref(info)),
            );
        }
        // SAFETY: set fraîchement alloué, jamais utilisé par le GPU.
        unsafe { shared.device.update_descriptor_sets(&writes, &[]) };
        VkBindGroup(Arc::new(GroupInner { shared: shared.clone(), pool, raw, sampled, _resources: resources }))
    }

    fn create_pipeline(&self, desc: &PipelineDesc<'_, Self>) -> VkPipeline {
        let shared = &self.0;
        let device = &shared.device;
        let set_layouts: Vec<vk::DescriptorSetLayout> = desc.layouts.iter().map(|layout| layout.0.raw).collect();
        // SAFETY: device valide ; module détruit après création du pipeline.
        unsafe {
            let layout = device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts), None)
                .unwrap_or_else(|error| panic!("pipeline layout « {} » : {error}", desc.label));
            let code = ash::util::read_spv(&mut std::io::Cursor::new(spirv(desc.shader))).expect("SPIR-V");
            let module = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .unwrap_or_else(|error| panic!("module SPIR-V « {} » : {error}", desc.label));
            let vertex_entry = CString::new(desc.vertex_entry).expect("nom d'entrée");
            let fragment_entry = CString::new(desc.fragment_entry).expect("nom d'entrée");
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(module)
                    .name(&vertex_entry),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(module)
                    .name(&fragment_entry),
            ];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(match desc.topology {
                Topology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
                Topology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
            });
            let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
            let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let multisample =
                vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
            // Mêmes équations que `wgpu::BlendState::{ALPHA_BLENDING, PREMULTIPLIED_ALPHA_BLENDING}`.
            let color_source = match desc.blend {
                Blend::Alpha => vk::BlendFactor::SRC_ALPHA,
                Blend::PremultipliedAlpha => vk::BlendFactor::ONE,
            };
            let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(color_source)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
            let formats = [desc.format];
            let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
            let info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport)
                .rasterization_state(&rasterization)
                .multisample_state(&multisample)
                .color_blend_state(&blend)
                .dynamic_state(&dynamic)
                .layout(layout)
                .push_next(&mut rendering);
            let raw = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, error)| error)
                .unwrap_or_else(|error| panic!("pipeline Vulkan « {} » : {error}", desc.label))[0];
            device.destroy_shader_module(module, None);
            VkPipeline(Arc::new(PipelineInner {
                shared: shared.clone(),
                raw,
                layout,
                set_count: set_layouts.len() as u32,
                _set_layouts: desc.layouts.iter().map(|layout| layout.0.clone()).collect(),
            }))
        }
    }

    fn create_encoder(&self, _label: &str) -> VkEncoder {
        let recorder = {
            let mut state = self.0.state.lock();
            self.0.new_recorder(&mut state)
        };
        VkEncoder { shared: self.0.clone(), recorder: Some(recorder) }
    }

    fn begin_pass<'a>(encoder: &'a mut VkEncoder, desc: &PassDesc<'_, Self>) -> VkPass<'a> {
        VkPass { encoder, target: desc.target.clone(), load: desc.load, commands: Vec::new() }
    }

    fn set_pipeline(pass: &mut VkPass<'_>, pipeline: &VkPipeline) {
        pass.commands.push(PassCommand::SetPipeline(pipeline.0.clone()));
    }

    fn set_bind_group(pass: &mut VkPass<'_>, index: u32, group: &VkBindGroup, dynamic_offsets: &[u32]) {
        pass.commands.push(PassCommand::SetBindGroup(index, group.clone(), dynamic_offsets.to_vec()));
    }

    fn set_viewport(pass: &mut VkPass<'_>, x: f32, y: f32, width: f32, height: f32) {
        pass.commands.push(PassCommand::SetViewport(x, y, width, height));
    }

    fn set_scissor_rect(pass: &mut VkPass<'_>, x: u32, y: u32, width: u32, height: u32) {
        pass.commands.push(PassCommand::SetScissor(vk::Rect2D {
            offset: vk::Offset2D { x: x as i32, y: y as i32 },
            extent: vk::Extent2D { width, height },
        }));
    }

    fn draw(pass: &mut VkPass<'_>, vertices: Range<u32>, instances: Range<u32>) {
        pass.commands.push(PassCommand::Draw(vertices, instances));
    }

    fn copy_buffer_to_buffer(
        encoder: &mut VkEncoder,
        source: &VkBuffer,
        source_offset: u64,
        destination: &VkBuffer,
        destination_offset: u64,
        size: u64,
    ) {
        let device = encoder.shared.device.clone();
        let recorder = encoder.recorder();
        global_barrier(&device, recorder.command_buffer);
        let region = vk::BufferCopy { src_offset: source_offset, dst_offset: destination_offset, size };
        // SAFETY: command buffer en enregistrement ; buffers retenus.
        unsafe { device.cmd_copy_buffer(recorder.command_buffer, source.0.raw, destination.0.raw, &[region]) };
        global_barrier(&device, recorder.command_buffer);
        recorder.retained.push(source.0.clone());
        recorder.retained.push(destination.0.clone());
    }

    fn copy_texture_to_texture(
        encoder: &mut VkEncoder,
        source: &VkTexture,
        destination: &VkTexture,
        width: u32,
        height: u32,
    ) {
        let device = encoder.shared.device.clone();
        let recorder = encoder.recorder();
        recorder.use_image(&device, source, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        recorder.use_image(&device, destination, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        global_barrier(&device, recorder.command_buffer);
        let layers = vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1);
        let region = vk::ImageCopy::default()
            .src_subresource(layers)
            .dst_subresource(layers)
            .extent(vk::Extent3D { width, height, depth: 1 });
        // SAFETY: command buffer en enregistrement ; images retenues, dans les bons layouts.
        unsafe {
            device.cmd_copy_image(
                recorder.command_buffer,
                source.0.raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                destination.0.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
        global_barrier(&device, recorder.command_buffer);
    }

    fn create_swapchain(
        &self,
        window: raw_window_handle::RawWindowHandle,
        display: raw_window_handle::RawDisplayHandle,
        width: u32,
        height: u32,
    ) -> Result<VkSwapchain> {
        let shared = &self.0;
        // SAFETY: handles de la fenêtre, qui survit au renderer (et donc à la swapchain).
        let surface = unsafe { ash_window::create_surface(&shared.entry, &shared.instance, display, window, None) }?;
        let destroy_surface = |error: anyhow::Error| {
            // SAFETY: surface créée ci-dessus, jamais utilisée.
            unsafe { shared.surface_loader.destroy_surface(surface, None) };
            error
        };
        // SAFETY: handles valides.
        let supported = unsafe {
            shared.surface_loader.get_physical_device_surface_support(shared.physical_device, shared.queue_family, surface)
        }
        .map_err(|error| destroy_surface(error.into()))?;
        if !supported {
            return Err(destroy_surface(anyhow!("la queue Vulkan ne peut pas présenter sur cette fenêtre")));
        }
        // SAFETY: handles valides.
        let formats = unsafe { shared.surface_loader.get_physical_device_surface_formats(shared.physical_device, surface) }
            .map_err(|error| destroy_surface(error.into()))?;
        // SAFETY: handles valides.
        let capabilities =
            unsafe { shared.surface_loader.get_physical_device_surface_capabilities(shared.physical_device, surface) }
                .map_err(|error| destroy_surface(error.into()))?;
        // Les shaders écrivent déjà du sRGB : format non sRGB (même choix que wgpu).
        let format = formats
            .iter()
            .find(|format| !is_srgb(format.format))
            .or(formats.first())
            .copied()
            .ok_or_else(|| destroy_surface(anyhow!("surface sans format de présentation")))?;
        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|mode| capabilities.supported_composite_alpha.contains(*mode))
        .ok_or_else(|| destroy_surface(anyhow!("surface sans mode alpha")))?;
        let acquire = (0..3)
            .map(|_| {
                // SAFETY: device valide ; détruits avec la swapchain.
                let semaphore = unsafe { shared.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                    .expect("sémaphore d'acquisition");
                (semaphore, Arc::new(AtomicU64::new(0)))
            })
            .collect();
        Ok(VkSwapchain {
            shared: shared.clone(),
            surface,
            raw: vk::SwapchainKHR::null(),
            images: Vec::new(),
            render_done: Vec::new(),
            acquire,
            next_acquire: 0,
            format: format.format,
            color_space: format.color_space,
            composite_alpha,
            extent: vk::Extent2D { width, height },
            present_mode: WindowPresentMode::Fifo,
        })
    }

    fn swapchain_format(swapchain: &VkSwapchain) -> vk::Format {
        swapchain.format
    }

    fn swapchain_premultiplied(swapchain: &VkSwapchain) -> bool {
        swapchain.composite_alpha == vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
    }

    fn swapchain_size(swapchain: &VkSwapchain) -> (u32, u32) {
        (swapchain.extent.width, swapchain.extent.height)
    }

    fn swapchain_present_mode(swapchain: &VkSwapchain) -> WindowPresentMode {
        swapchain.present_mode
    }

    fn supported_present_modes(&self, swapchain: &VkSwapchain) -> Vec<WindowPresentMode> {
        // SAFETY: handles valides.
        let modes = unsafe {
            self.0.surface_loader.get_physical_device_surface_present_modes(self.0.physical_device, swapchain.surface)
        }
        .unwrap_or_default();
        [WindowPresentMode::Fifo, WindowPresentMode::Mailbox, WindowPresentMode::Immediate]
            .into_iter()
            .filter(|mode| modes.contains(&present_mode(*mode)))
            .collect()
    }

    fn swapchain_frame_latency(swapchain: &VkSwapchain) -> u32 {
        swapchain.images.len().saturating_sub(1).max(1) as u32
    }

    fn configure_swapchain(&self, swapchain: &mut VkSwapchain, width: u32, height: u32, mode: WindowPresentMode) {
        let shared = &self.0;
        {
            let _queue = shared.queue_lock.lock();
            // SAFETY: queue protégée ; les images de l'ancienne swapchain ne sont plus en usage.
            if let Err(error) = unsafe { shared.device.queue_wait_idle(shared.queue) } {
                log::error!("attente de la queue Vulkan avant reconfiguration : {error}");
            }
        }
        // SAFETY: handles valides.
        let capabilities = match unsafe {
            shared.surface_loader.get_physical_device_surface_capabilities(shared.physical_device, swapchain.surface)
        } {
            Ok(capabilities) => capabilities,
            Err(error) => {
                log::error!("capacités de la surface Vulkan : {error}");
                return;
            }
        };
        let extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
                height: height.clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height),
            }
        };
        let mut image_count = capabilities.min_image_count.max(3);
        if capabilities.max_image_count != 0 {
            image_count = image_count.min(capabilities.max_image_count);
        }
        let old = swapchain.raw;
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(swapchain.surface)
            .min_image_count(image_count)
            .image_format(swapchain.format)
            .image_color_space(swapchain.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(swapchain.composite_alpha)
            .present_mode(present_mode(mode))
            .clipped(true)
            .old_swapchain(old);
        // SAFETY: descripteur valide ; l'ancienne swapchain est retirée puis détruite, queue au repos.
        let raw = match unsafe { shared.swapchain_loader.create_swapchain(&info, None) } {
            Ok(raw) => raw,
            Err(error) => {
                log::error!("création de la swapchain Vulkan : {error}");
                return;
            }
        };
        swapchain.destroy_images();
        if old != vk::SwapchainKHR::null() {
            // SAFETY: remplacée, et plus aucune image en usage (queue au repos).
            unsafe { shared.swapchain_loader.destroy_swapchain(old, None) };
        }
        swapchain.raw = raw;
        // SAFETY: swapchain valide.
        swapchain.images = unsafe { shared.swapchain_loader.get_swapchain_images(raw) }.unwrap_or_default();
        swapchain.render_done = swapchain
            .images
            .iter()
            .map(|_| {
                // SAFETY: device valide ; détruits avec les images.
                unsafe { shared.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                    .expect("sémaphore de rendu")
            })
            .collect();
        swapchain.extent = extent;
        swapchain.present_mode = mode;
    }

    fn acquire(&self, swapchain: &mut VkSwapchain) -> Acquire<VkFrame> {
        if swapchain.raw == vk::SwapchainKHR::null() {
            return Acquire::Outdated;
        }
        let (semaphore, slot) = swapchain.acquire[swapchain.next_acquire].clone();
        // Le sémaphore d'acquisition ne se réutilise qu'une fois sa soumission terminée.
        self.0.wait_serial(slot.load(Ordering::Acquire));
        // SAFETY: swapchain et sémaphore valides, sémaphore non signalé.
        match unsafe {
            self.0.swapchain_loader.acquire_next_image(swapchain.raw, u64::MAX, semaphore, vk::Fence::null())
        } {
            Ok((index, _suboptimal)) => {
                swapchain.next_acquire = (swapchain.next_acquire + 1) % swapchain.acquire.len();
                Acquire::Frame(VkFrame {
                    swapchain: swapchain.raw,
                    image: swapchain.images[index as usize],
                    index,
                    extent: swapchain.extent,
                    acquire: semaphore,
                    render_done: swapchain.render_done[index as usize],
                    slot,
                })
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::ERROR_SURFACE_LOST_KHR) => Acquire::Outdated,
            Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => Acquire::Skip("swap chain acquire timed out"),
            Err(error) => {
                log::error!("acquisition d'image Vulkan : {error}");
                Acquire::Skip("swap chain acquire failed")
            }
        }
    }

    fn copy_texture_to_frame(encoder: &mut VkEncoder, source: &VkTexture, frame: &VkFrame) {
        let device = encoder.shared.device.clone();
        let recorder = encoder.recorder();
        recorder.use_image(&device, source, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        layout_barrier(
            &device,
            recorder.command_buffer,
            frame.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        let layers = vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1);
        let region = vk::ImageCopy::default().src_subresource(layers).dst_subresource(layers).extent(vk::Extent3D {
            width: frame.extent.width.min(source.0.width),
            height: frame.extent.height.min(source.0.height),
            depth: 1,
        });
        // SAFETY: command buffer en enregistrement ; image de swapchain acquise, attendue au submit.
        unsafe {
            device.cmd_copy_image(
                recorder.command_buffer,
                source.0.raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                frame.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
        layout_barrier(
            &device,
            recorder.command_buffer,
            frame.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
        recorder.frame =
            Some(FrameSync { acquire: frame.acquire, render_done: frame.render_done, slot: frame.slot.clone() });
    }

    fn submit(&self, mut encoder: VkEncoder) {
        if let Some(recorder) = encoder.recorder.take() {
            self.0.submit(recorder);
        }
    }

    fn present(&self, frame: VkFrame) {
        let swapchains = [frame.swapchain];
        let indices = [frame.index];
        let wait = [frame.render_done];
        let info = vk::PresentInfoKHR::default().wait_semaphores(&wait).swapchains(&swapchains).image_indices(&indices);
        let _queue = self.0.queue_lock.lock();
        // SAFETY: queue protégée ; image acquise et rendue par la soumission qui signale `render_done`.
        match unsafe { self.0.swapchain_loader.queue_present(self.0.queue, &info) } {
            Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {}
            Err(error) => log::error!("présentation Vulkan : {error}"),
        }
    }

    fn init_external_textures(&self, textures: [&VkTexture; 3]) {
        let mut recorder = {
            let mut state = self.0.state.lock();
            self.0.new_recorder(&mut state)
        };
        let device = &self.0.device;
        for texture in textures {
            recorder.use_image(device, texture, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        }
        // Contrat des surfaces 3D : effacées, puis échantillonnables entre deux trames.
        let clear = vk::ClearColorValue { float32: [0.0; 4] };
        for texture in textures {
            // SAFETY: command buffer en enregistrement ; image en TRANSFER_DST_OPTIMAL.
            unsafe {
                device.cmd_clear_color_image(
                    recorder.command_buffer,
                    texture.0.raw,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &clear,
                    &[color_range()],
                );
            }
        }
        for texture in textures {
            recorder.use_image(device, texture, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
        self.0.submit(recorder);
    }
}
