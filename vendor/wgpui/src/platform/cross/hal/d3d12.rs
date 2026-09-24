//! Implémentation D3D12 native (windows-rs) de [`Gpu`], sans wgpu.
//!
//! Mêmes mécanismes que `vulkan.rs` pour tenir les contrats de [`super`] :
//! - Textures : chaque liste note, par texture, le premier et le dernier état qu'elle
//!   utilise ; au `submit`, une liste de correction amène chaque texture de son état réel
//!   (suivi par le device) au premier état attendu. Les uploads, enregistrés à tout moment
//!   mais exécutés avant la liste principale, restent ainsi corrects.
//! - Buffers : D3D12 les ramène à `COMMON` à la fin de chaque `ExecuteCommandLists` ; chaque
//!   liste est exécutée seule, son suivi part donc de `COMMON`, sans correction.
//! - Rétention : une liste garde un `Arc` de chaque ressource qu'elle référence jusqu'à ce
//!   que la fence atteigne son numéro de soumission.
//!
//! Liaisons : fixées au build (voir `build.rs`) — `@group(g) @binding(b)` → espace `g`,
//! registre `b`. Un seul sampler existe (linéaire, bords clampés) : le tas de samplers de
//! naga n'a qu'un descripteur et chaque tampon d'index de groupe est un tampon de zéros.
//! L'espace clip D3D12 est celui de WebGPU : aucun retournement.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context as _, Result, anyhow};
use parking_lot::Mutex;
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND, RECT, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::{
    D3D_FEATURE_LEVEL_11_0, D3D_PRIMITIVE_TOPOLOGY, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, ID3DBlob,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::WaitForSingleObjectEx;
use windows::core::Interface as _;

use super::{
    Acquire, BindEntry, BindResource, BindingKind, Blend, BufferUsage, Gpu, LayoutEntry, LoadOp,
    PassDesc, PipelineDesc, TextureUsage, Topology,
};
use crate::{GpuSpecs, NativeDevice, NativeTexture, SurfaceFormat, WindowPresentMode};

mod bytecode {
    include!(concat!(env!("OUT_DIR"), "/dxbc.rs"));
}

const STAGING_CHUNK: u64 = 4 * 1024 * 1024;
/// Descripteurs CBV/SRV visibles des shaders, alloués par blocs d'un groupe chacun.
const VIEW_HEAP_SIZE: u32 = 1 << 18;
const VIEW_BLOCK: u32 = 4;
const RTV_HEAP_SIZE: u32 = 1024;
/// Taille du tas de samplers déclaré par naga (`nagaSamplerHeap[2048]`).
const SAMPLER_HEAP_SIZE: u32 = 2048;
const SWAPCHAIN_FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;
const FRAME_LATENCY: u32 = 2;
/// Tampons de swapchain, un de plus que les trames en vol.
const SWAPCHAIN_BUFFERS: u32 = FRAME_LATENCY + 1;
/// Taille maximale d'une vue de constantes (4096 × float4).
const MAX_CBV_SIZE: u64 = 65536;

/// État « échantillonné » des textures, contrat des surfaces 3D natives compris.
const SAMPLED: D3D12_RESOURCE_STATES =
    D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0);
/// Buffers lus par les shaders : uniforms et storage.
const BUFFER_READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(
    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER.0
        | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0
        | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0,
);

type Retained = Arc<dyn Any + Send + Sync>;

/// Barrière de transition sur une ressource empruntée (pas d'AddRef, pas de Release).
fn transition(resource: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: copie non possédée ; la ressource survit à la liste (rétention).
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn subresource_location(resource: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: copie non possédée ; la ressource survit à la liste (rétention).
        pResource: unsafe { std::mem::transmute_copy(resource) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
    }
}

fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    }
}

fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: le blob possède `GetBufferSize` octets à `GetBufferPointer`.
    unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) }
}

/// Device D3D12 partagé par l'UI et les moteurs 3D d'une application.
#[derive(Clone)]
pub(crate) struct D3d12Gpu(Arc<Shared>);

struct Shared {
    factory: IDXGIFactory4,
    adapter_desc: DXGI_ADAPTER_DESC1,
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// Contrat du trait : une queue D3D12 est thread-safe, rien à exclure.
    queue_lock: Mutex<()>,
    tearing: bool,
    view_heap: ID3D12DescriptorHeap,
    view_increment: u32,
    sampler_heap: ID3D12DescriptorHeap,
    /// Tampon d'index de samplers de tous les groupes : des zéros, l'unique sampler.
    zero_buffer: ID3D12Resource,
    next_id: AtomicU64,
    state: Mutex<State>,
}

// SAFETY: les objets D3D12 et DXGI utilisés ici sont libres de threads ; les listes et
// l'état mutable passent par le verrou de `State`.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

#[derive(Default)]
struct State {
    submitted: u64,
    uploads: Option<Recorder>,
    staging: Vec<StagingChunk>,
    free_staging: Vec<StagingChunk>,
    in_flight: VecDeque<InFlight>,
    free_lists: Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    texture_states: HashMap<u64, D3D12_RESOURCE_STATES>,
    free_view_blocks: Vec<u32>,
    next_view_block: u32,
    rtv_heaps: Vec<ID3D12DescriptorHeap>,
    free_rtvs: Vec<(usize, u32)>,
}

struct InFlight {
    serial: u64,
    lists: Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    staging: Vec<StagingChunk>,
    retained: Vec<Retained>,
}

struct StagingChunk {
    buffer: ID3D12Resource,
    mapped: *mut u8,
    size: u64,
    used: u64,
}

struct TextureUse {
    resource: ID3D12Resource,
    first: D3D12_RESOURCE_STATES,
    last: D3D12_RESOURCE_STATES,
}

/// Liste en cours d'enregistrement et tout ce qui doit survivre à son exécution.
struct Recorder {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    textures: HashMap<u64, TextureUse>,
    buffers: HashMap<u64, D3D12_RESOURCE_STATES>,
    retained: Vec<Retained>,
    heaps_bound: bool,
}

impl Recorder {
    fn use_texture(&mut self, texture: &D3dTexture, state: D3D12_RESOURCE_STATES) {
        let inner = &texture.0;
        match self.textures.get_mut(&inner.id) {
            Some(usage) => {
                if usage.last != state {
                    // SAFETY: liste en enregistrement, ressource retenue.
                    unsafe { self.list.ResourceBarrier(&[transition(&usage.resource, usage.last, state)]) };
                    usage.last = state;
                }
            }
            None => {
                self.textures
                    .insert(inner.id, TextureUse { resource: inner.resource.clone(), first: state, last: state });
            }
        }
        self.retained.push(texture.0.clone());
    }

    fn use_buffer(&mut self, buffer: &D3dBuffer, state: D3D12_RESOURCE_STATES) {
        let current = self.buffers.get(&buffer.0.id).copied().unwrap_or(D3D12_RESOURCE_STATE_COMMON);
        if current != state {
            // SAFETY: liste en enregistrement, ressource retenue.
            unsafe { self.list.ResourceBarrier(&[transition(&buffer.0.resource, current, state)]) };
            self.buffers.insert(buffer.0.id, state);
        }
        self.retained.push(buffer.0.clone());
    }

    fn bind_heaps(&mut self, shared: &Shared) {
        if !self.heaps_bound {
            // SAFETY: liste en enregistrement ; les tas vivent autant que le device.
            unsafe {
                self.list.SetDescriptorHeaps(&[Some(shared.view_heap.clone()), Some(shared.sampler_heap.clone())]);
            }
            self.heaps_bound = true;
        }
    }
}

impl Shared {
    fn completed(&self) -> u64 {
        // SAFETY: fence valide.
        unsafe { self.fence.GetCompletedValue() }
    }

    fn wait_serial(&self, serial: u64) {
        if self.completed() >= serial {
            return;
        }
        // SAFETY: événement nul = attente synchrone jusqu'à la valeur.
        if let Err(error) = unsafe { self.fence.SetEventOnCompletion(serial, HANDLE::default()) } {
            log::error!("attente de la fence D3D12 : {error}");
        }
    }

    fn wait_idle(&self) {
        let serial = {
            let mut state = self.state.lock();
            state.submitted += 1;
            let serial = state.submitted;
            // SAFETY: fence et queue valides.
            if let Err(error) = unsafe { self.queue.Signal(&self.fence, serial) } {
                log::error!("signal de la fence D3D12 : {error}");
            }
            serial
        };
        self.wait_serial(serial);
    }

    fn new_recorder(&self, state: &mut State) -> Recorder {
        let (allocator, list) = match state.free_lists.pop() {
            Some((allocator, list)) => {
                // SAFETY: le GPU a fini cette liste (fence atteinte) avant son retour ici.
                unsafe {
                    allocator.Reset().expect("reset de l'allocateur D3D12");
                    list.Reset(&allocator, None).expect("reset de la liste D3D12");
                }
                (allocator, list)
            }
            None => {
                // SAFETY: device valide.
                unsafe {
                    let allocator: ID3D12CommandAllocator = self
                        .device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                        .expect("allocateur de commandes D3D12");
                    let list: ID3D12GraphicsCommandList = self
                        .device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                        .expect("liste de commandes D3D12");
                    (allocator, list)
                }
            }
        };
        Recorder { allocator, list, textures: HashMap::new(), buffers: HashMap::new(), retained: Vec::new(), heaps_bound: false }
    }

    /// Relâche ce que le GPU a fini d'utiliser. Les `Arc` sont rendus à l'appelant, qui les
    /// libère hors du verrou (leurs Drop le reprennent).
    fn collect_completed(&self, state: &mut State) -> Vec<Retained> {
        let completed = self.completed();
        let mut released = Vec::new();
        while state.in_flight.front().is_some_and(|in_flight| in_flight.serial <= completed) {
            let Some(in_flight) = state.in_flight.pop_front() else { break };
            state.free_lists.extend(in_flight.lists);
            for mut chunk in in_flight.staging {
                chunk.used = 0;
                state.free_staging.push(chunk);
            }
            released.extend(in_flight.retained);
        }
        released
    }

    /// Réserve `size` octets alignés sur `alignment` dans le staging et les remplit par `write`.
    fn staging_write(&self, state: &mut State, size: u64, alignment: u64, write: impl FnOnce(&mut [u8])) -> (ID3D12Resource, u64) {
        let fits = |chunk: &StagingChunk| chunk.used.next_multiple_of(alignment) + size <= chunk.size;
        if !state.staging.last().is_some_and(fits) {
            let reusable = state.free_staging.iter().position(|chunk| chunk.size >= size);
            let chunk = match reusable {
                Some(index) => state.free_staging.swap_remove(index),
                None => self.create_staging_chunk(size.max(STAGING_CHUNK)),
            };
            state.staging.push(chunk);
        }
        let Some(chunk) = state.staging.last_mut() else { unreachable!("chunk ajouté ci-dessus") };
        let offset = chunk.used.next_multiple_of(alignment);
        // SAFETY: `offset + size <= chunk.size`, mémoire mappée pour toute la vie du chunk.
        write(unsafe { std::slice::from_raw_parts_mut(chunk.mapped.add(offset as usize), size as usize) });
        chunk.used = offset + size;
        (chunk.buffer.clone(), offset)
    }

    fn create_staging_chunk(&self, size: u64) -> StagingChunk {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: descripteurs valides ; la mémoire reste mappée jusqu'à la libération.
        unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &buffer_desc(size),
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .expect("tampon de staging D3D12");
            let buffer = buffer.expect("tampon de staging D3D12");
            let mut mapped = std::ptr::null_mut();
            buffer.Map(0, None, Some(&mut mapped)).expect("map du staging D3D12");
            StagingChunk { buffer, mapped: mapped.cast(), size, used: 0 }
        }
    }

    fn uploads<'a>(&self, state: &'a mut State) -> &'a mut Recorder {
        if state.uploads.is_none() {
            let recorder = self.new_recorder(state);
            state.uploads = Some(recorder);
        }
        let Some(recorder) = state.uploads.as_mut() else { unreachable!("créé ci-dessus") };
        recorder
    }

    fn execute(&self, list: &ID3D12GraphicsCommandList) {
        // SAFETY: liste fermée ; queue libre de threads.
        unsafe {
            list.Close().expect("fermeture de la liste D3D12");
            self.queue.ExecuteCommandLists(&[Some(list.cast().expect("ID3D12CommandList"))]);
        }
    }

    /// Exécute les uploads en attente puis `main`, chacun précédé des corrections d'état
    /// de ses textures.
    fn submit(&self, main: Recorder) {
        let mut state = self.state.lock();
        let serial = state.submitted + 1;
        let mut lists = Vec::new();
        let mut retained = Vec::new();
        let uploads = state.uploads.take();
        for mut recorder in uploads.into_iter().chain(std::iter::once(main)) {
            if !recorder.textures.is_empty() {
                let fixup = self.new_recorder(&mut state);
                let mut barriers = Vec::new();
                for (id, usage) in &recorder.textures {
                    let current = state.texture_states.get(id).copied().unwrap_or(D3D12_RESOURCE_STATE_COMMON);
                    if current != usage.first {
                        barriers.push(transition(&usage.resource, current, usage.first));
                    }
                    state.texture_states.insert(*id, usage.last);
                }
                if !barriers.is_empty() {
                    // SAFETY: liste en enregistrement ; ressources retenues par `recorder`.
                    unsafe { fixup.list.ResourceBarrier(&barriers) };
                }
                self.execute(&fixup.list);
                lists.push((fixup.allocator, fixup.list));
            }
            self.execute(&recorder.list);
            lists.push((recorder.allocator, recorder.list));
            retained.append(&mut recorder.retained);
        }
        // SAFETY: fence et queue valides.
        if let Err(error) = unsafe { self.queue.Signal(&self.fence, serial) } {
            log::error!("signal de la fence D3D12 : {error}");
        }
        state.submitted = serial;
        let staging = std::mem::take(&mut state.staging);
        state.in_flight.push_back(InFlight { serial, lists, staging, retained });
        let released = self.collect_completed(&mut state);
        drop(state);
        drop(released);
    }

    fn allocate_view_block(&self) -> u32 {
        let mut state = self.state.lock();
        if let Some(block) = state.free_view_blocks.pop() {
            return block;
        }
        let block = state.next_view_block;
        assert!(
            (block + 1) * VIEW_BLOCK <= VIEW_HEAP_SIZE,
            "tas de descripteurs D3D12 épuisé ({VIEW_HEAP_SIZE} descripteurs)"
        );
        state.next_view_block += 1;
        block
    }

    fn view_cpu(&self, slot: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        // SAFETY: tas valide.
        let start = unsafe { self.view_heap.GetCPUDescriptorHandleForHeapStart() };
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start.ptr + (slot * self.view_increment) as usize }
    }

    fn view_gpu(&self, slot: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: tas valide.
        let start = unsafe { self.view_heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE { ptr: start.ptr + u64::from(slot * self.view_increment) }
    }

    fn allocate_rtv(&self) -> (usize, u32) {
        let mut state = self.state.lock();
        if let Some(slot) = state.free_rtvs.pop() {
            return slot;
        }
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: RTV_HEAP_SIZE,
            ..Default::default()
        };
        // SAFETY: device valide.
        let heap: ID3D12DescriptorHeap =
            unsafe { self.device.CreateDescriptorHeap(&desc) }.expect("tas de vues de rendu D3D12");
        let index = state.rtv_heaps.len();
        state.rtv_heaps.push(heap);
        state.free_rtvs.extend((1..RTV_HEAP_SIZE).rev().map(|slot| (index, slot)));
        (index, 0)
    }

    fn rtv_handle(&self, (heap, slot): (usize, u32)) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let state = self.state.lock();
        // SAFETY: device et tas valides.
        unsafe {
            let increment = self.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            let start = state.rtv_heaps[heap].GetCPUDescriptorHandleForHeapStart();
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start.ptr + (slot * increment) as usize }
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        self.wait_idle();
        let state = self.state.get_mut();
        state.uploads = None;
        state.in_flight.clear();
    }
}

// --- Ressources -----------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct D3dBuffer(Arc<BufferInner>);

struct BufferInner {
    id: u64,
    resource: ID3D12Resource,
    size: u64,
    address: u64,
}

// SAFETY: ressource D3D12 libre de threads.
unsafe impl Send for BufferInner {}
unsafe impl Sync for BufferInner {}

#[derive(Clone)]
pub(crate) struct D3dTexture(Arc<TextureInner>);

struct TextureInner {
    shared: Arc<Shared>,
    id: u64,
    resource: ID3D12Resource,
    format: DXGI_FORMAT,
    render_target: bool,
    width: u32,
    height: u32,
}

// SAFETY: ressource D3D12 libre de threads.
unsafe impl Send for TextureInner {}
unsafe impl Sync for TextureInner {}

impl Drop for TextureInner {
    fn drop(&mut self) {
        self.shared.state.lock().texture_states.remove(&self.id);
    }
}

#[derive(Clone)]
pub(crate) struct D3dTextureView(Arc<ViewInner>);

struct ViewInner {
    texture: D3dTexture,
    rtv: Option<(usize, u32)>,
}

impl Drop for ViewInner {
    fn drop(&mut self) {
        if let Some(slot) = self.rtv {
            self.texture.0.shared.state.lock().free_rtvs.push(slot);
        }
    }
}

/// L'unique sampler du HAL vit dans le tas de samplers ; l'objet n'est qu'un jeton.
pub(crate) struct D3dSampler;

pub(crate) struct D3dBindGroupLayout(Arc<Vec<LayoutEntry>>);

/// Entrées d'un groupe servies par sa table de descripteurs (ni offset dynamique, ni sampler).
fn table_entries(entries: &[LayoutEntry]) -> impl Iterator<Item = &LayoutEntry> {
    entries.iter().filter(|entry| {
        !matches!(entry.kind, BindingKind::Sampler | BindingKind::Uniform { dynamic_offset: true, .. })
    })
}

fn dynamic_entries(entries: &[LayoutEntry]) -> impl Iterator<Item = &LayoutEntry> {
    entries.iter().filter(|entry| matches!(entry.kind, BindingKind::Uniform { dynamic_offset: true, .. }))
}

#[derive(Clone)]
pub(crate) struct D3dBindGroup(Arc<GroupInner>);

struct GroupInner {
    shared: Arc<Shared>,
    block: Option<u32>,
    /// Adresse GPU de chaque uniform à offset dynamique, dans l'ordre du layout.
    dynamic: Vec<u64>,
    sampled: Vec<D3dTexture>,
    buffers: Vec<D3dBuffer>,
    _resources: Vec<Retained>,
}

impl Drop for GroupInner {
    fn drop(&mut self) {
        if let Some(block) = self.block {
            self.shared.state.lock().free_view_blocks.push(block);
        }
    }
}

/// Paramètres racine d'un groupe dans la root signature d'un pipeline.
struct GroupParams {
    table: Option<u32>,
    dynamic: Vec<u32>,
    sampler_index: Option<u32>,
}

pub(crate) struct D3dPipeline(Arc<PipelineInner>);

struct PipelineInner {
    pipeline: ID3D12PipelineState,
    root_signature: ID3D12RootSignature,
    topology: D3D_PRIMITIVE_TOPOLOGY,
    groups: Vec<GroupParams>,
}

/// Paramètres racine communs à tous les pipelines.
const SPECIAL_CONSTANTS_PARAMETER: u32 = 0;
const SAMPLER_HEAP_PARAMETER: u32 = 1;

// SAFETY: objets D3D12 libres de threads.
unsafe impl Send for PipelineInner {}
unsafe impl Sync for PipelineInner {}

pub(crate) struct D3dEncoder {
    shared: Arc<Shared>,
    recorder: Option<Recorder>,
}

impl D3dEncoder {
    fn recorder(&mut self) -> &mut Recorder {
        let Some(recorder) = self.recorder.as_mut() else { unreachable!("encodeur consommé par submit") };
        recorder
    }
}

impl Drop for D3dEncoder {
    fn drop(&mut self) {
        let Some(recorder) = self.recorder.take() else { return };
        // Trame abandonnée : jamais exécutée, la liste se ferme et retourne au pool.
        // SAFETY: liste jamais soumise.
        if unsafe { recorder.list.Close() }.is_ok() {
            self.shared.state.lock().free_lists.push((recorder.allocator, recorder.list));
        }
    }
}

pub(crate) struct D3dPass<'a> {
    encoder: &'a mut D3dEncoder,
    pipeline: Option<Arc<PipelineInner>>,
    bound: [Option<(D3dBindGroup, Vec<u32>)>; 8],
    dirty: [bool; 8],
}

pub(crate) struct D3dSwapchain {
    shared: Arc<Shared>,
    swapchain: IDXGISwapChain3,
    waitable: HANDLE,
    buffers: Vec<ID3D12Resource>,
    width: u32,
    height: u32,
    present_mode: WindowPresentMode,
    configured: bool,
}

impl Drop for D3dSwapchain {
    fn drop(&mut self) {
        self.shared.wait_idle();
        let released = self.shared.collect_completed(&mut self.shared.state.lock());
        drop(released);
        // SAFETY: handle rendu par `GetFrameLatencyWaitableObject`, fermé une seule fois.
        if let Err(error) = unsafe { CloseHandle(self.waitable) } {
            log::error!("fermeture de l'attente de swapchain D3D12 : {error}");
        }
    }
}

pub(crate) struct D3dFrame {
    swapchain: IDXGISwapChain3,
    back_buffer: ID3D12Resource,
    width: u32,
    height: u32,
    present_mode: WindowPresentMode,
}

/// Retient une ressource hors enveloppe (tampon arrière) jusqu'à la fin GPU.
struct RetainedResource {
    _resource: ID3D12Resource,
}

// SAFETY: ressource D3D12 libre de threads.
unsafe impl Send for RetainedResource {}
unsafe impl Sync for RetainedResource {}

// --- Création du device ---------------------------------------------------------------

/// Relaie les messages de la couche de debug D3D12 (avertissements et erreurs).
unsafe extern "system" fn debug_message(
    _category: D3D12_MESSAGE_CATEGORY,
    severity: D3D12_MESSAGE_SEVERITY,
    id: D3D12_MESSAGE_ID,
    description: windows::core::PCSTR,
    _context: *mut std::ffi::c_void,
) {
    // Effacer sans valeur optimisée à la création est voulu : les cibles changent de couleur
    // d'effacement d'une passe à l'autre.
    if id == D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE {
        return;
    }
    if severity.0 <= D3D12_MESSAGE_SEVERITY_WARNING.0 {
        // SAFETY: chaîne C fournie par la couche de debug, valide pendant l'appel.
        let text = unsafe { description.to_string() }.unwrap_or_default();
        eprintln!("[d3d12 {severity:?}] {text}");
    }
}

impl D3d12Gpu {
    /// Device D3D12 (niveau 11_0) sur le GPU matériel le plus performant ; jamais WARP.
    /// `GPUI_D3D12_DEBUG=1` active la couche de debug et relaie ses messages sur stderr.
    pub(crate) fn new() -> Result<Self> {
        let debug = std::env::var("GPUI_D3D12_DEBUG").is_ok_and(|value| value == "1");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            // SAFETY: sortie valide.
            unsafe { D3D12GetDebugInterface(&mut layer) }.context("couche de debug D3D12 (Outils graphiques)")?;
            if let Some(layer) = layer {
                // SAFETY: avant toute création de device.
                unsafe { layer.EnableDebugLayer() };
            }
        }
        let flags = if debug { DXGI_CREATE_FACTORY_DEBUG } else { DXGI_CREATE_FACTORY_FLAGS(0) };
        // SAFETY: appel sans pré-condition.
        let factory: IDXGIFactory6 = unsafe { CreateDXGIFactory2(flags) }.context("fabrique DXGI")?;
        let mut chosen = None;
        for index in 0.. {
            // SAFETY: énumération ; `DXGI_ERROR_NOT_FOUND` termine la liste.
            let Ok(adapter) = (unsafe {
                factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
            }) else {
                break;
            };
            // SAFETY: adaptateur valide.
            let desc = unsafe { adapter.GetDesc1() }?;
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                continue;
            }
            let mut device: Option<ID3D12Device> = None;
            // SAFETY: adaptateur valide, sortie valide.
            if unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device) }.is_ok()
                && let Some(device) = device
            {
                chosen = Some((desc, device));
                break;
            }
        }
        let (adapter_desc, device) = chosen.ok_or_else(|| anyhow!("aucun GPU D3D12 matériel (niveau 11_0)"))?;
        if debug && let Ok(info_queue) = device.cast::<ID3D12InfoQueue1>() {
            let mut cookie = 0;
            // SAFETY: rappel `extern "system"` sans contexte, valide toute la vie du processus.
            if let Err(error) = unsafe {
                info_queue.RegisterMessageCallback(
                    Some(debug_message),
                    D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                    std::ptr::null_mut(),
                    &mut cookie,
                )
            } {
                log::error!("relais des messages D3D12 : {error}");
            }
        }
        // SAFETY: descripteurs valides.
        unsafe {
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let view_heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: VIEW_HEAP_SIZE,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
            let view_increment = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let sampler_heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                NumDescriptors: SAMPLER_HEAP_SIZE,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
            device.CreateSampler(
                &D3D12_SAMPLER_DESC {
                    Filter: D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                    MaxAnisotropy: 1,
                    ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
                    MaxLOD: 32.0,
                    ..Default::default()
                },
                sampler_heap.GetCPUDescriptorHandleForHeapStart(),
            );
            let upload = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
            let mut zero_buffer: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(
                &upload,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(256),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut zero_buffer,
            )?;
            let zero_buffer = zero_buffer.ok_or_else(|| anyhow!("tampon de zéros D3D12"))?;
            let mut mapped = std::ptr::null_mut();
            zero_buffer.Map(0, None, Some(&mut mapped))?;
            std::ptr::write_bytes(mapped.cast::<u8>(), 0, 256);
            zero_buffer.Unmap(0, None);
            let mut allow_tearing = windows::core::BOOL::default();
            let tearing = factory
                .cast::<IDXGIFactory5>()
                .and_then(|factory| {
                    factory.CheckFeatureSupport(
                        DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                        (&mut allow_tearing as *mut windows::core::BOOL).cast(),
                        size_of::<windows::core::BOOL>() as u32,
                    )
                })
                .is_ok()
                && allow_tearing.as_bool();
            Ok(Self(Arc::new(Shared {
                factory: factory.cast()?,
                adapter_desc,
                device,
                queue,
                fence,
                queue_lock: Mutex::new(()),
                tearing,
                view_heap,
                view_increment,
                sampler_heap,
                zero_buffer,
                next_id: AtomicU64::new(1),
                state: Mutex::new(State::default()),
            })))
        }
    }

    fn create_raw_buffer(&self, label: &str, size: u64) -> D3dBuffer {
        let shared = &self.0;
        // Les vues de constantes couvrent des multiples de 256 octets.
        let size = size.max(1).next_multiple_of(256);
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: descripteurs valides.
        let resource = unsafe {
            shared
                .device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &buffer_desc(size),
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut resource,
                )
                .unwrap_or_else(|error| panic!("buffer D3D12 « {label} » : {error}"));
            resource.unwrap_or_else(|| panic!("buffer D3D12 « {label} »"))
        };
        // SAFETY: ressource valide.
        let address = unsafe { resource.GetGPUVirtualAddress() };
        D3dBuffer(Arc::new(BufferInner { id: shared.next_id.fetch_add(1, Ordering::Relaxed), resource, size, address }))
    }
}

impl Gpu for D3d12Gpu {
    type Format = DXGI_FORMAT;
    type Buffer = D3dBuffer;
    type Texture = D3dTexture;
    type TextureView = D3dTextureView;
    type Sampler = D3dSampler;
    type BindGroupLayout = D3dBindGroupLayout;
    type BindGroup = D3dBindGroup;
    type Pipeline = D3dPipeline;
    type Encoder = D3dEncoder;
    type Pass<'a> = D3dPass<'a>;
    type Swapchain = D3dSwapchain;
    type Frame = D3dFrame;
    #[cfg(feature = "flamegraph")]
    type Profiler = super::NoProfiler;

    const ATLAS_MONOCHROME: DXGI_FORMAT = DXGI_FORMAT_R8_UNORM;
    const ATLAS_POLYCHROME: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

    fn bytes_per_pixel(format: DXGI_FORMAT) -> u32 {
        match format {
            DXGI_FORMAT_R8_UNORM => 1,
            DXGI_FORMAT_R16G16B16A16_FLOAT => 8,
            _ => 4,
        }
    }

    fn gpu_specs(&self) -> GpuSpecs {
        let desc = &self.0.adapter_desc;
        let length = desc.Description.iter().position(|&unit| unit == 0).unwrap_or(desc.Description.len());
        GpuSpecs {
            is_software_emulated: false,
            device_name: String::from_utf16_lossy(&desc.Description[..length]),
            driver_name: "D3D12".into(),
            driver_info: format!("vendor 0x{:04x} device 0x{:04x}", desc.VendorId, desc.DeviceId),
        }
    }

    fn min_uniform_offset_alignment(&self) -> u32 {
        D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT
    }

    fn create_buffer(&self, label: &str, size: u64, _usage: BufferUsage) -> D3dBuffer {
        self.create_raw_buffer(label, size)
    }

    fn create_buffer_init(&self, label: &str, contents: &[u8], _usage: BufferUsage) -> D3dBuffer {
        let buffer = self.create_raw_buffer(label, contents.len() as u64);
        self.write_buffer(&buffer, 0, contents);
        buffer
    }

    fn buffer_size(buffer: &D3dBuffer) -> u64 {
        buffer.0.size
    }

    fn write_buffer(&self, buffer: &D3dBuffer, offset: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let shared = &self.0;
        let mut state = shared.state.lock();
        let (staging, staging_offset) =
            shared.staging_write(&mut state, data.len() as u64, 16, |target| target.copy_from_slice(data));
        let recorder = shared.uploads(&mut state);
        recorder.use_buffer(buffer, D3D12_RESOURCE_STATE_COPY_DEST);
        // SAFETY: liste d'upload en enregistrement, sous le verrou d'état ; ressources retenues.
        unsafe {
            recorder.list.CopyBufferRegion(&buffer.0.resource, offset, &staging, staging_offset, data.len() as u64);
        }
    }

    fn create_texture(&self, label: &str, width: u32, height: u32, format: DXGI_FORMAT, usage: TextureUsage) -> D3dTexture {
        let shared = &self.0;
        let render_target = usage.contains(TextureUsage::RENDER_TARGET);
        let (width, height) = (width.max(1), height.max(1));
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: if render_target { D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET } else { D3D12_RESOURCE_FLAG_NONE },
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: descripteurs valides.
        let resource = unsafe {
            shared
                .device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut resource)
                .unwrap_or_else(|error| panic!("texture D3D12 « {label} » : {error}"));
            resource.unwrap_or_else(|| panic!("texture D3D12 « {label} »"))
        };
        D3dTexture(Arc::new(TextureInner {
            shared: shared.clone(),
            id: shared.next_id.fetch_add(1, Ordering::Relaxed),
            resource,
            format,
            render_target,
            width,
            height,
        }))
    }

    fn texture_size(texture: &D3dTexture) -> (u32, u32) {
        (texture.0.width, texture.0.height)
    }

    fn create_view(texture: &D3dTexture) -> D3dTextureView {
        let shared = &texture.0.shared;
        let rtv = texture.0.render_target.then(|| {
            let slot = shared.allocate_rtv();
            // SAFETY: ressource et descripteur valides.
            unsafe { shared.device.CreateRenderTargetView(&texture.0.resource, None, shared.rtv_handle(slot)) };
            slot
        });
        D3dTextureView(Arc::new(ViewInner { texture: texture.clone(), rtv }))
    }

    fn write_texture(&self, texture: &D3dTexture, origin: (u32, u32), size: (u32, u32), bytes_per_pixel: u32, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let shared = &self.0;
        let row = (size.0 * bytes_per_pixel) as usize;
        let pitch = (row as u32).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
        let total = u64::from(pitch) * u64::from(size.1);
        let mut state = shared.state.lock();
        let (staging, staging_offset) =
            shared.staging_write(&mut state, total, u64::from(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT), |target| {
                for (source, destination) in data.chunks(row).zip(target.chunks_mut(pitch as usize)) {
                    destination[..source.len()].copy_from_slice(source);
                }
            });
        let recorder = shared.uploads(&mut state);
        recorder.use_texture(texture, D3D12_RESOURCE_STATE_COPY_DEST);
        let source = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: copie non possédée ; le staging survit à l'exécution.
            pResource: unsafe { std::mem::transmute_copy(&staging) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: staging_offset,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: texture.0.format,
                        Width: size.0,
                        Height: size.1,
                        Depth: 1,
                        RowPitch: pitch,
                    },
                },
            },
        };
        // SAFETY: liste d'upload en enregistrement, sous le verrou d'état.
        unsafe {
            recorder.list.CopyTextureRegion(
                &subresource_location(&texture.0.resource),
                origin.0,
                origin.1,
                0,
                &source,
                None,
            );
        }
    }

    fn create_linear_sampler(&self, _label: &str) -> D3dSampler {
        D3dSampler
    }

    fn create_bind_group_layout(&self, label: &str, entries: &[LayoutEntry]) -> D3dBindGroupLayout {
        assert!(
            table_entries(entries).count() as u32 <= VIEW_BLOCK,
            "layout D3D12 « {label} » : plus de {VIEW_BLOCK} descripteurs"
        );
        D3dBindGroupLayout(Arc::new(entries.to_vec()))
    }

    fn create_bind_group(&self, _label: &str, layout: &D3dBindGroupLayout, entries: &[BindEntry<'_, Self>]) -> D3dBindGroup {
        let shared = &self.0;
        let resource_of = |binding: u32| entries.iter().find(|entry| entry.binding == binding).map(|entry| &entry.resource);
        let mut resources: Vec<Retained> = Vec::new();
        let mut sampled = Vec::new();
        let mut buffers = Vec::new();
        let table: Vec<&LayoutEntry> = table_entries(&layout.0).collect();
        let block = (!table.is_empty()).then(|| shared.allocate_view_block());
        for (index, entry) in table.iter().enumerate() {
            let Some(block) = block else { break };
            let handle = shared.view_cpu(block * VIEW_BLOCK + index as u32);
            match (entry.kind, resource_of(entry.binding)) {
                (BindingKind::Uniform { .. }, Some(BindResource::Buffer { buffer, offset, size })) => {
                    let size = size.unwrap_or(buffer.0.size - offset).next_multiple_of(256).min(MAX_CBV_SIZE);
                    let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: buffer.0.address + offset, SizeInBytes: size as u32 };
                    // SAFETY: descripteur dans le bloc réservé à ce groupe.
                    unsafe { shared.device.CreateConstantBufferView(Some(&desc), handle) };
                    buffers.push((*buffer).clone());
                }
                (BindingKind::Storage, Some(BindResource::Buffer { buffer, offset, size })) => {
                    let size = size.unwrap_or(buffer.0.size - offset);
                    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_TYPELESS,
                        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_SRV {
                                FirstElement: offset / 4,
                                NumElements: (size / 4) as u32,
                                StructureByteStride: 0,
                                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                            },
                        },
                    };
                    // SAFETY: descripteur dans le bloc réservé à ce groupe.
                    unsafe { shared.device.CreateShaderResourceView(&buffer.0.resource, Some(&desc), handle) };
                    buffers.push((*buffer).clone());
                }
                (BindingKind::Texture, Some(BindResource::Texture(view))) => {
                    let texture = &view.0.texture;
                    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: texture.0.format,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                        },
                    };
                    // SAFETY: descripteur dans le bloc réservé à ce groupe.
                    unsafe { shared.device.CreateShaderResourceView(&texture.0.resource, Some(&desc), handle) };
                    sampled.push(texture.clone());
                    resources.push(view.0.clone());
                }
                (kind, _) => panic!("groupe D3D12 : ressource absente ou incompatible pour {kind:?} @binding({})", entry.binding),
            }
        }
        let dynamic = dynamic_entries(&layout.0)
            .map(|entry| match resource_of(entry.binding) {
                Some(BindResource::Buffer { buffer, offset, .. }) => {
                    buffers.push((*buffer).clone());
                    buffer.0.address + offset
                }
                _ => panic!("groupe D3D12 : buffer absent @binding({})", entry.binding),
            })
            .collect();
        D3dBindGroup(Arc::new(GroupInner { shared: shared.clone(), block, dynamic, sampled, buffers, _resources: resources }))
    }

    fn create_pipeline(&self, desc: &PipelineDesc<'_, Self>) -> D3dPipeline {
        let shared = &self.0;
        let space = |group: usize| group as u32;
        let ranges: Vec<Vec<D3D12_DESCRIPTOR_RANGE>> = desc
            .layouts
            .iter()
            .enumerate()
            .map(|(group, layout)| {
                table_entries(&layout.0)
                    .enumerate()
                    .map(|(index, entry)| D3D12_DESCRIPTOR_RANGE {
                        RangeType: match entry.kind {
                            BindingKind::Uniform { .. } => D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                            _ => D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                        },
                        NumDescriptors: 1,
                        BaseShaderRegister: entry.binding,
                        RegisterSpace: space(group),
                        OffsetInDescriptorsFromTableStart: index as u32,
                    })
                    .collect()
            })
            .collect();
        let sampler_range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
            NumDescriptors: SAMPLER_HEAP_SIZE,
            BaseShaderRegister: 0,
            RegisterSpace: bytecode::SAMPLER_HEAP_SPACE,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let parameter = |parameter_type, anonymous| D3D12_ROOT_PARAMETER {
            ParameterType: parameter_type,
            Anonymous: anonymous,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let descriptor = |register, space| D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: register, RegisterSpace: space },
        };
        // Ordre fixé par `SPECIAL_CONSTANTS_PARAMETER` et `SAMPLER_HEAP_PARAMETER`.
        let mut parameters = vec![
            parameter(
                D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: bytecode::SPECIAL_CONSTANTS_SPACE,
                        Num32BitValues: 3,
                    },
                },
            ),
            parameter(
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &sampler_range },
                },
            ),
        ];
        let mut groups = Vec::new();
        for (group, layout) in desc.layouts.iter().enumerate() {
            let table = (!ranges[group].is_empty()).then(|| {
                parameters.push(parameter(
                    D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    D3D12_ROOT_PARAMETER_0 {
                        DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                            NumDescriptorRanges: ranges[group].len() as u32,
                            pDescriptorRanges: ranges[group].as_ptr(),
                        },
                    },
                ));
                parameters.len() as u32 - 1
            });
            let dynamic = dynamic_entries(&layout.0)
                .map(|entry| {
                    parameters.push(parameter(D3D12_ROOT_PARAMETER_TYPE_CBV, descriptor(entry.binding, space(group))));
                    parameters.len() as u32 - 1
                })
                .collect();
            let sampler_index = layout.0.iter().any(|entry| entry.kind == BindingKind::Sampler).then(|| {
                parameters.push(parameter(
                    D3D12_ROOT_PARAMETER_TYPE_SRV,
                    descriptor(bytecode::SAMPLER_INDEX_REGISTER, space(group)),
                ));
                parameters.len() as u32 - 1
            });
            groups.push(GroupParams { table, dynamic, sampler_index });
        }
        let root_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: parameters.len() as u32,
            pParameters: parameters.as_ptr(),
            ..Default::default()
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        // SAFETY: descripteurs (et tableaux pointés) vivants pendant la sérialisation.
        let root_signature: ID3D12RootSignature = unsafe {
            if let Err(failure) = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, Some(&mut error)) {
                let message = error.as_ref().map(|blob| String::from_utf8_lossy(blob_bytes(blob)).into_owned());
                panic!("root signature D3D12 « {} » : {failure} {}", desc.label, message.unwrap_or_default());
            }
            let blob = blob.unwrap_or_else(|| panic!("root signature D3D12 « {} »", desc.label));
            shared
                .device
                .CreateRootSignature(0, blob_bytes(&blob))
                .unwrap_or_else(|error| panic!("root signature D3D12 « {} » : {error}", desc.label))
        };
        let shader = |entry: &str| {
            let code = bytecode::dxbc(desc.shader.name(), entry)
                .unwrap_or_else(|| panic!("DXBC absent : {}::{entry}", desc.shader.name()));
            D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() }
        };
        // Mêmes équations que `wgpu::BlendState::{ALPHA_BLENDING, PREMULTIPLIED_ALPHA_BLENDING}`.
        let color_source = match desc.blend {
            Blend::Alpha => D3D12_BLEND_SRC_ALPHA,
            Blend::PremultipliedAlpha => D3D12_BLEND_ONE,
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            LogicOpEnable: false.into(),
            SrcBlend: color_source,
            DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
        };
        let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
        formats[0] = desc.format;
        let pipeline_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: copie non possédée ; la root signature survit à l'appel.
            pRootSignature: unsafe { std::mem::transmute_copy(&root_signature) },
            VS: shader(desc.vertex_entry),
            PS: shader(desc.fragment_entry),
            BlendState: blend,
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: formats,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        // SAFETY: descripteur complet, bytecode statique.
        let pipeline: ID3D12PipelineState = unsafe { shared.device.CreateGraphicsPipelineState(&pipeline_desc) }
            .unwrap_or_else(|error| panic!("pipeline D3D12 « {} » : {error}", desc.label));
        D3dPipeline(Arc::new(PipelineInner {
            pipeline,
            root_signature,
            topology: match desc.topology {
                Topology::TriangleList => D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                Topology::TriangleStrip => D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            },
            groups,
        }))
    }

    fn create_encoder(&self, _label: &str) -> D3dEncoder {
        let recorder = {
            let mut state = self.0.state.lock();
            self.0.new_recorder(&mut state)
        };
        D3dEncoder { shared: self.0.clone(), recorder: Some(recorder) }
    }

    fn begin_pass<'a>(encoder: &'a mut D3dEncoder, desc: &PassDesc<'_, Self>) -> D3dPass<'a> {
        let shared = encoder.shared.clone();
        let target = desc.target;
        let texture = &target.0.texture;
        let rtv = shared.rtv_handle(target.0.rtv.expect("cible de passe D3D12 sans RENDER_TARGET"));
        let recorder = encoder.recorder();
        recorder.bind_heaps(&shared);
        recorder.use_texture(texture, D3D12_RESOURCE_STATE_RENDER_TARGET);
        recorder.retained.push(target.0.clone());
        let (width, height) = (texture.0.width, texture.0.height);
        // SAFETY: liste en enregistrement ; descripteur copié à l'enregistrement.
        unsafe {
            recorder.list.OMSetRenderTargets(1, Some(&rtv), false, None);
            if let LoadOp::Clear([r, g, b, a]) = desc.load {
                recorder.list.ClearRenderTargetView(rtv, &[r as f32, g as f32, b as f32, a as f32], None);
            }
            recorder.list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            recorder.list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: width as i32, bottom: height as i32 }]);
        }
        D3dPass { encoder, pipeline: None, bound: Default::default(), dirty: [false; 8] }
    }

    fn set_pipeline(pass: &mut D3dPass<'_>, pipeline: &D3dPipeline) {
        if pass.pipeline.as_ref().is_some_and(|current| Arc::ptr_eq(current, &pipeline.0)) {
            return;
        }
        let shared = pass.encoder.shared.clone();
        let recorder = pass.encoder.recorder();
        let inner = &pipeline.0;
        // SAFETY: liste en enregistrement ; objets retenus.
        unsafe {
            recorder.list.SetGraphicsRootSignature(&inner.root_signature);
            recorder.list.SetPipelineState(&inner.pipeline);
            recorder.list.IASetPrimitiveTopology(inner.topology);
            recorder
                .list
                .SetGraphicsRootDescriptorTable(SAMPLER_HEAP_PARAMETER, shared.sampler_heap.GetGPUDescriptorHandleForHeapStart());
        }
        recorder.retained.push(inner.clone());
        pass.pipeline = Some(inner.clone());
        pass.dirty = [true; 8];
    }

    fn set_bind_group(pass: &mut D3dPass<'_>, index: u32, group: &D3dBindGroup, dynamic_offsets: &[u32]) {
        let recorder = pass.encoder.recorder();
        for texture in &group.0.sampled {
            recorder.use_texture(texture, SAMPLED);
        }
        for buffer in &group.0.buffers {
            recorder.use_buffer(buffer, BUFFER_READ);
        }
        recorder.retained.push(group.0.clone());
        if let Some(slot) = pass.bound.get_mut(index as usize) {
            *slot = Some((group.clone(), dynamic_offsets.to_vec()));
            pass.dirty[index as usize] = true;
        }
    }

    fn set_viewport(pass: &mut D3dPass<'_>, x: f32, y: f32, width: f32, height: f32) {
        let viewport = D3D12_VIEWPORT { TopLeftX: x, TopLeftY: y, Width: width, Height: height, MinDepth: 0.0, MaxDepth: 1.0 };
        // SAFETY: liste en enregistrement.
        unsafe { pass.encoder.recorder().list.RSSetViewports(&[viewport]) };
    }

    fn set_scissor_rect(pass: &mut D3dPass<'_>, x: u32, y: u32, width: u32, height: u32) {
        let rect = RECT { left: x as i32, top: y as i32, right: (x + width) as i32, bottom: (y + height) as i32 };
        // SAFETY: liste en enregistrement.
        unsafe { pass.encoder.recorder().list.RSSetScissorRects(&[rect]) };
    }

    fn draw(pass: &mut D3dPass<'_>, vertices: Range<u32>, instances: Range<u32>) {
        let Some(pipeline) = pass.pipeline.clone() else {
            log::error!("draw D3D12 sans pipeline");
            return;
        };
        let shared = pass.encoder.shared.clone();
        // SAFETY: ressource valide.
        let zero_address = unsafe { shared.zero_buffer.GetGPUVirtualAddress() };
        let list = pass.encoder.recorder().list.clone();
        for (index, params) in pipeline.groups.iter().enumerate() {
            if !pass.dirty[index] {
                continue;
            }
            let Some((group, offsets)) = &pass.bound[index] else { continue };
            // SAFETY: liste en enregistrement ; le groupe est retenu par la liste.
            unsafe {
                if let (Some(parameter), Some(block)) = (params.table, group.0.block) {
                    list.SetGraphicsRootDescriptorTable(parameter, shared.view_gpu(block * VIEW_BLOCK));
                }
                for ((parameter, address), offset) in params.dynamic.iter().zip(&group.0.dynamic).zip(offsets) {
                    list.SetGraphicsRootConstantBufferView(*parameter, address + u64::from(*offset));
                }
                if let Some(parameter) = params.sampler_index {
                    list.SetGraphicsRootShaderResourceView(parameter, zero_address);
                }
            }
            pass.dirty[index] = false;
        }
        // `SV_VertexID`/`SV_InstanceID` n'incluent pas les premiers indices : naga les ajoute.
        let constants = [vertices.start, instances.start, 0];
        // SAFETY: liste en enregistrement.
        unsafe {
            list.SetGraphicsRoot32BitConstants(SPECIAL_CONSTANTS_PARAMETER, 3, constants.as_ptr().cast(), 0);
            list.DrawInstanced(
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            );
        }
    }

    fn copy_buffer_to_buffer(
        encoder: &mut D3dEncoder,
        source: &D3dBuffer,
        source_offset: u64,
        destination: &D3dBuffer,
        destination_offset: u64,
        size: u64,
    ) {
        let gpu = D3d12Gpu(encoder.shared.clone());
        let recorder = encoder.recorder();
        // SAFETY: liste en enregistrement ; ressources retenues.
        unsafe {
            if Arc::ptr_eq(&source.0, &destination.0) {
                // Une ressource ne peut pas être à la fois source et destination de copie.
                let scratch = gpu.create_raw_buffer("copy_scratch", size);
                recorder.use_buffer(source, D3D12_RESOURCE_STATE_COPY_SOURCE);
                recorder.use_buffer(&scratch, D3D12_RESOURCE_STATE_COPY_DEST);
                recorder.list.CopyBufferRegion(&scratch.0.resource, 0, &source.0.resource, source_offset, size);
                recorder.use_buffer(&scratch, D3D12_RESOURCE_STATE_COPY_SOURCE);
                recorder.use_buffer(destination, D3D12_RESOURCE_STATE_COPY_DEST);
                recorder.list.CopyBufferRegion(&destination.0.resource, destination_offset, &scratch.0.resource, 0, size);
            } else {
                recorder.use_buffer(source, D3D12_RESOURCE_STATE_COPY_SOURCE);
                recorder.use_buffer(destination, D3D12_RESOURCE_STATE_COPY_DEST);
                recorder.list.CopyBufferRegion(&destination.0.resource, destination_offset, &source.0.resource, source_offset, size);
            }
        }
    }

    fn copy_texture_to_texture(encoder: &mut D3dEncoder, source: &D3dTexture, destination: &D3dTexture, width: u32, height: u32) {
        let recorder = encoder.recorder();
        recorder.use_texture(source, D3D12_RESOURCE_STATE_COPY_SOURCE);
        recorder.use_texture(destination, D3D12_RESOURCE_STATE_COPY_DEST);
        let area = D3D12_BOX { left: 0, top: 0, front: 0, right: width, bottom: height, back: 1 };
        // SAFETY: liste en enregistrement ; textures retenues, dans les bons états.
        unsafe {
            recorder.list.CopyTextureRegion(
                &subresource_location(&destination.0.resource),
                0,
                0,
                0,
                &subresource_location(&source.0.resource),
                Some(&area),
            );
        }
    }

    fn create_swapchain(
        &self,
        window: raw_window_handle::RawWindowHandle,
        _display: raw_window_handle::RawDisplayHandle,
        width: u32,
        height: u32,
    ) -> Result<D3dSwapchain> {
        let raw_window_handle::RawWindowHandle::Win32(handle) = window else {
            anyhow::bail!("D3D12 exige une fenêtre Win32");
        };
        let shared = &self.0;
        let hwnd = HWND(handle.hwnd.get() as *mut std::ffi::c_void);
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width.max(1),
            Height: height.max(1),
            Format: SWAPCHAIN_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: SWAPCHAIN_BUFFERS,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_IGNORE,
            Flags: swapchain_flags(shared.tearing).0 as u32,
            ..Default::default()
        };
        // SAFETY: fenêtre vivante (elle survit au renderer) ; queue valide.
        unsafe {
            let swapchain: IDXGISwapChain3 = shared
                .factory
                .CreateSwapChainForHwnd(&shared.queue, hwnd, &desc, None, None::<&IDXGIOutput>)?
                .cast()?;
            if let Err(error) = shared.factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER) {
                log::debug!("association DXGI de la fenêtre : {error}");
            }
            swapchain.SetMaximumFrameLatency(FRAME_LATENCY)?;
            let waitable = swapchain.GetFrameLatencyWaitableObject();
            Ok(D3dSwapchain {
                shared: shared.clone(),
                swapchain,
                waitable,
                buffers: Vec::new(),
                width: desc.Width,
                height: desc.Height,
                present_mode: WindowPresentMode::Fifo,
                configured: false,
            })
        }
    }

    fn swapchain_format(_swapchain: &D3dSwapchain) -> DXGI_FORMAT {
        SWAPCHAIN_FORMAT
    }

    fn swapchain_premultiplied(_swapchain: &D3dSwapchain) -> bool {
        false
    }

    fn swapchain_size(swapchain: &D3dSwapchain) -> (u32, u32) {
        (swapchain.width, swapchain.height)
    }

    fn swapchain_present_mode(swapchain: &D3dSwapchain) -> WindowPresentMode {
        swapchain.present_mode
    }

    fn supported_present_modes(&self, _swapchain: &D3dSwapchain) -> Vec<WindowPresentMode> {
        let mut modes = vec![WindowPresentMode::Fifo, WindowPresentMode::Mailbox];
        if self.0.tearing {
            modes.push(WindowPresentMode::Immediate);
        }
        modes
    }

    fn swapchain_frame_latency(_swapchain: &D3dSwapchain) -> u32 {
        FRAME_LATENCY
    }

    fn configure_swapchain(&self, swapchain: &mut D3dSwapchain, width: u32, height: u32, mode: WindowPresentMode) {
        let shared = &self.0;
        // ResizeBuffers exige qu'aucune référence aux tampons ne subsiste, GPU compris.
        shared.wait_idle();
        let released = shared.collect_completed(&mut shared.state.lock());
        drop(released);
        swapchain.buffers.clear();
        let (width, height) = (width.max(1), height.max(1));
        // SAFETY: plus aucune référence aux tampons ; mêmes drapeaux qu'à la création.
        let resized = unsafe {
            swapchain.swapchain.ResizeBuffers(
                SWAPCHAIN_BUFFERS,
                width,
                height,
                SWAPCHAIN_FORMAT,
                swapchain_flags(shared.tearing),
            )
        };
        if let Err(error) = resized {
            log::error!("redimensionnement de la swapchain D3D12 : {error}");
            return;
        }
        swapchain.buffers = (0..SWAPCHAIN_BUFFERS)
            // SAFETY: index dans le nombre de tampons.
            .filter_map(|index| unsafe { swapchain.swapchain.GetBuffer::<ID3D12Resource>(index) }.ok())
            .collect();
        swapchain.width = width;
        swapchain.height = height;
        swapchain.present_mode = mode;
        swapchain.configured = true;
    }

    fn acquire(&self, swapchain: &mut D3dSwapchain) -> Acquire<D3dFrame> {
        if !swapchain.configured || swapchain.buffers.len() != SWAPCHAIN_BUFFERS as usize {
            return Acquire::Outdated;
        }
        // SAFETY: objet d'attente de la swapchain, vivant avec elle.
        if unsafe { WaitForSingleObjectEx(swapchain.waitable, 1000, true) } != WAIT_OBJECT_0 {
            return Acquire::Skip("swap chain acquire timed out");
        }
        // SAFETY: swapchain valide.
        let index = unsafe { swapchain.swapchain.GetCurrentBackBufferIndex() } as usize;
        Acquire::Frame(D3dFrame {
            swapchain: swapchain.swapchain.clone(),
            back_buffer: swapchain.buffers[index].clone(),
            width: swapchain.width,
            height: swapchain.height,
            present_mode: swapchain.present_mode,
        })
    }

    fn copy_texture_to_frame(encoder: &mut D3dEncoder, source: &D3dTexture, frame: &D3dFrame) {
        let recorder = encoder.recorder();
        recorder.use_texture(source, D3D12_RESOURCE_STATE_COPY_SOURCE);
        let area = D3D12_BOX {
            left: 0,
            top: 0,
            front: 0,
            right: frame.width.min(source.0.width),
            bottom: frame.height.min(source.0.height),
            back: 1,
        };
        // SAFETY: liste en enregistrement ; tampon arrière retenu jusqu'à la fin GPU.
        unsafe {
            recorder.list.ResourceBarrier(&[transition(
                &frame.back_buffer,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            recorder.list.CopyTextureRegion(
                &subresource_location(&frame.back_buffer),
                0,
                0,
                0,
                &subresource_location(&source.0.resource),
                Some(&area),
            );
            recorder.list.ResourceBarrier(&[transition(
                &frame.back_buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PRESENT,
            )]);
        }
        recorder.retained.push(Arc::new(RetainedResource { _resource: frame.back_buffer.clone() }));
    }

    fn submit(&self, mut encoder: D3dEncoder) {
        if let Some(recorder) = encoder.recorder.take() {
            self.0.submit(recorder);
        }
    }

    fn present(&self, frame: D3dFrame) {
        let (interval, flags) = match frame.present_mode {
            WindowPresentMode::Fifo => (1, DXGI_PRESENT(0)),
            WindowPresentMode::Mailbox => (0, DXGI_PRESENT(0)),
            WindowPresentMode::Immediate => (0, DXGI_PRESENT_ALLOW_TEARING),
        };
        // SAFETY: tampon arrière rendu par la soumission précédente sur la même queue.
        let result = unsafe { frame.swapchain.Present(interval, flags) };
        if result.is_err() {
            log::error!("présentation D3D12 : {result:?}");
        }
    }

    fn init_external_textures(&self, textures: [&D3dTexture; 3]) {
        let mut recorder = {
            let mut state = self.0.state.lock();
            self.0.new_recorder(&mut state)
        };
        // Contrat des surfaces 3D : effacées, puis échantillonnables entre deux trames.
        for texture in textures {
            let view = Self::create_view(texture);
            let Some(slot) = view.0.rtv else { continue };
            recorder.use_texture(texture, D3D12_RESOURCE_STATE_RENDER_TARGET);
            // SAFETY: liste en enregistrement ; texture en RENDER_TARGET.
            unsafe { recorder.list.ClearRenderTargetView(self.0.rtv_handle(slot), &[0.0; 4], None) };
            recorder.use_texture(texture, SAMPLED);
            recorder.retained.push(view.0.clone());
        }
        self.0.submit(recorder);
    }

    fn surface_format(format: SurfaceFormat) -> DXGI_FORMAT {
        match format {
            SurfaceFormat::Bgra8UnormSrgb => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
            SurfaceFormat::Rgba8UnormSrgb => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        }
    }

    fn queue_lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.0.queue_lock.lock()
    }

    fn native_device(&self) -> Option<NativeDevice> {
        Some(NativeDevice::Dx12 { device: self.0.device.as_raw(), queue: self.0.queue.as_raw() })
    }

    fn native_texture(texture: &D3dTexture, _view: &D3dTextureView) -> Option<NativeTexture> {
        Some(NativeTexture::Dx12(texture.0.resource.as_raw()))
    }
}

fn swapchain_flags(tearing: bool) -> DXGI_SWAP_CHAIN_FLAG {
    let mut flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    if tearing {
        flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    }
    flags
}
