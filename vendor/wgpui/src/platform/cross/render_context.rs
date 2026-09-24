use parking_lot::Mutex;
use std::sync::Arc;

use super::hal::{BufferUsage, Gpu};
use super::surface_registry::SurfaceRegistry;
#[cfg(feature = "wgpu")]
use super::hal::wgpu::WgpuGpu;

/// Contexte GPU d'une application : le device (partagé par l'UI et les moteurs 3D) et
/// les ressources communes à toutes ses fenêtres.
pub struct RenderContext<G: Gpu> {
    pub(crate) gpu: Arc<G>,

    // The globals and color-adjustment uniforms are NOT shared here, on
    // purpose: every window's renderer dedups their uploads against its own
    // last-written value (`uploaded_globals`), which is only sound if each
    // window owns the buffer it compares against. Sharing one buffer across
    // windows let window B's viewport overwrite window A's while A's dedup
    // guard still said "already uploaded" — A then rendered every frame
    // against B's viewport size until something re-wrote it, presenting as
    // shifted/scaled content with hit testing unaffected. They now live on
    // the renderer, one pair per window.
    pub(super) quads_buffer: Mutex<G::Buffer>,
    pub(super) shadows_buffer: Mutex<G::Buffer>,
    pub(super) backdrop_filters_buffer: Mutex<G::Buffer>,
    pub(super) underlines_buffer: Mutex<G::Buffer>,
    pub(super) mono_sprites_buffer: Mutex<G::Buffer>,
    pub(super) poly_sprites_buffer: Mutex<G::Buffer>,
    pub(super) paths_vertices_buffer: Mutex<G::Buffer>,

    pub(crate) surface_registry: Arc<SurfaceRegistry<G>>,

    /// Guards swapchain reconfiguration against concurrent queue submission.
    ///
    /// `Surface::configure` waits for the device to go idle and fails with
    /// `GpuWaitTimeout` if anything submits while it waits -- wgpu-core treats a
    /// non-empty queue after the wait as proof that "another thread is submitting
    /// at the same time". Because `WgpuSurfaceHandle` hands out clones of this
    /// device and queue, every external render thread is such a submitter.
    ///
    /// External render threads hold a **read** guard across render + submit
    /// (`WgpuSurfaceHandle::submit_guard`); swapchain configuration holds the
    /// **write** guard (`Renderer::reconfigure_surface`). Read guards do not block
    /// each other, so any number of surfaces keep rendering at independent rates;
    /// only reconfiguration (resize) serializes against them.
    ///
    /// The renderer's own submits deliberately do NOT take a read guard: they run
    /// on the same thread as `configure`, so guarding them would self-deadlock.
    pub(crate) gpu_submit_lock: Arc<parking_lot::RwLock<()>>,
}

impl<G: Gpu> std::ops::Deref for RenderContext<G> {
    type Target = G;

    fn deref(&self) -> &G {
        &self.gpu
    }
}

impl<G: Gpu> RenderContext<G> {
    pub(crate) fn with_gpu(gpu: G) -> Self {
        // Phase 4 of the profiling epic (issue #60/#71) reads these seven
        // fixed buffers back via `copy_buffer_to_buffer` during a triggered
        // GPU deep capture (see `DeepCaptureBufferKind`), which requires
        // `COPY_SRC` on the source buffer or wgpu's validator rejects the
        // encoder outright -- not a soft failure, a hard panic on the first
        // frame a capture is armed. Add it only when the capture code that
        // actually needs it is compiled in, so a non-`flamegraph` build's
        // buffers are byte-for-byte the same as before this fix.
        let instances = BufferUsage::VERTEX | BufferUsage::COPY_DST | BufferUsage::STORAGE;
        let storage = BufferUsage::STORAGE | BufferUsage::COPY_DST;
        #[cfg(feature = "flamegraph")]
        let (instances, storage) = (instances | BufferUsage::COPY_SRC, storage | BufferUsage::COPY_SRC);
        // Resized dynamically by ensure_buffer_size() when the initial allocation is exceeded.
        const INITIAL_SIZE: u64 = 8 * 1024 * 1024;
        let gpu = Arc::new(gpu);
        let buffer = |label: &str, usage| Mutex::new(gpu.create_buffer(label, INITIAL_SIZE, usage));
        Self {
            quads_buffer: buffer("Quads Buffer", instances),
            shadows_buffer: buffer("Shadows Buffer", instances),
            backdrop_filters_buffer: buffer("Backdrop Filters Buffer", storage),
            underlines_buffer: buffer("Underlines Buffer", instances),
            mono_sprites_buffer: buffer("Monosprites Buffer", instances),
            poly_sprites_buffer: buffer("Poly Sprites Buffer", instances),
            paths_vertices_buffer: buffer("Path Vertices Buffer", storage),
            surface_registry: Arc::new(SurfaceRegistry::new(gpu.clone())),
            gpu_submit_lock: Arc::new(parking_lot::RwLock::new(())),
            gpu,
        }
    }

    /// Sum of every fixed-size buffer's size (Phase 3 of the profiling epic,
    /// issue #59). The globals and color-adjustment uniforms are per-renderer
    /// (see the struct field comment); they are accounted by the renderer's own
    /// reporting, not here.
    #[cfg(feature = "flamegraph")]
    pub(crate) fn fixed_buffer_memory_usage(&self) -> u64 {
        [
            &self.quads_buffer,
            &self.shadows_buffer,
            &self.backdrop_filters_buffer,
            &self.underlines_buffer,
            &self.mono_sprites_buffer,
            &self.poly_sprites_buffer,
            &self.paths_vertices_buffer,
        ]
        .into_iter()
        .map(|buffer| G::buffer_size(&buffer.lock()))
        .sum()
    }
}

#[cfg(feature = "wgpu")]
pub type WgpuContext = RenderContext<WgpuGpu>;

#[cfg(feature = "wgpu")]
pub use wgpu_context::*;

#[cfg(feature = "wgpu")]
mod wgpu_context {
use parking_lot::Mutex;

use super::{WgpuContext, WgpuGpu};

/// Compat (voir src/compat.rs) : choix d'adaptateur par l'hôte.
/// Reçoit les infos des adaptateurs APTES (features UI remplies), dans l'ordre
/// de [`enumerate_qualifying_adapters`], et rend l'index retenu.
pub type AdapterSelector =
    std::sync::Arc<dyn Fn(&[wgpu::AdapterInfo]) -> Option<usize> + Send + Sync>;

/// Options for configuring the WGPU backend.
#[derive(Clone)]
pub struct WgpuOptions {
    /// Additional WGPU features to request when creating the device.
    /// These are OR'd with the features WGPUI itself requires.
    pub additional_features: wgpu::Features,

    /// Maximum number of frames the GPU can queue ahead before blocking.
    /// Higher values improve throughput at the cost of input latency.
    pub desired_maximum_frame_latency: u32,

    /// Compat (voir src/compat.rs) : sélecteur d'adaptateur.
    /// `None` = premier adaptateur apte (comportement amont). Ignoré sur
    /// macOS (chemin features optionnelles) et en WASM.
    pub adapter_selector: Option<AdapterSelector>,
}

impl Default for WgpuOptions {
    fn default() -> Self {
        Self {
            additional_features: wgpu::Features::empty(),
            desired_maximum_frame_latency: 2,
            adapter_selector: None,
        }
    }
}

// Compat (voir src/compat.rs) : les quatre poignées wgpu du
// renderer, adoptables par l'hôte (le globe et le calcul doivent rendre sur
// CE device — un second device, même adaptateur, a un autre registre d'ids).
#[cfg(not(target_family = "wasm"))]
static HOST_DEVICE: Mutex<Option<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)>> =
    Mutex::new(None);

/// Compat (voir src/compat.rs) : les poignées wgpu du renderer
/// hôte, une fois le device créé ([`WgpuContext::new`]). `None` avant.
#[cfg(not(target_family = "wasm"))]
pub fn host_device() -> Option<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    HOST_DEVICE.lock().clone()
}

/// Compat (voir src/compat.rs) : la même énumération d'adaptateurs
/// que la création du device UI (tous backends, filtrés par les features que
/// WGPUI exige) — la liste dans laquelle l'`AdapterSelector` choisit.
#[cfg(not(target_family = "wasm"))]
pub fn enumerate_qualifying_adapters() -> Vec<wgpu::Adapter> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let required_features = wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | wgpu::Features::INDIRECT_FIRST_INSTANCE;
    // Sur macOS ces features sont optionnelles à la création du device
    // (voir `WgpuContext::new`) : ne pas sur-filtrer l'énumération.
    #[cfg(not(target_os = "macos"))]
    let required_features = required_features
        | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .filter(|adapter| adapter.features().contains(required_features))
        .collect()
}

impl WgpuContext {
    pub fn new(options: &WgpuOptions) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });

        // On WASM, adapter enumeration is async and pollster::block_on panics
        // (Condvar::wait not supported). Return an error so CrossPlatform defers
        // to the spawn_local path in run().
        #[cfg(target_family = "wasm")]
        {
            let _ = instance;
            let _ = options;
            anyhow::bail!("WgpuContext requires async initialization on WASM");
        }

        // ============ Native-only path below ============
        #[cfg(not(target_family = "wasm"))]
        {
            // Features WGPUI itself needs for its rendering pipeline.
            // GPUI-3D : PRIMITIVE_INDEX retiré — aucun shader ne l'utilise, et MoltenVK ne l'a pas.
            let wgpui_features = wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                        | wgpu::Features::INDIRECT_FIRST_INSTANCE;

            let required_features = wgpui_features | options.additional_features;

            let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));

            // On macOS, some features are optional — prefer adapters that expose them
            // but do not require them, since Metal may not advertise them on all hardware.
            // On all other platforms, require them outright.
            #[cfg(target_os = "macos")]
            let (adapter, device_features) = {
                let optional_features = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
                    | wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
                let mut qualifying: Vec<wgpu::Adapter> = adapters
                    .into_iter()
                    .filter(|adapter| adapter.features().contains(required_features))
                    .collect();
                // GPUI-3D : le sélecteur vaut aussi sur macOS (Metal ou Vulkan/MoltenVK).
                let selected = options.adapter_selector.as_ref().and_then(|selector| {
                    let infos: Vec<wgpu::AdapterInfo> =
                        qualifying.iter().map(|adapter| adapter.get_info()).collect();
                    selector(&infos)
                });
                let adapter = match selected {
                    Some(index) if index < qualifying.len() => Some(qualifying.swap_remove(index)),
                    _ => qualifying
                        .into_iter()
                        // GPUI-3D : à égalité, Metal plutôt que Vulkan/MoltenVK.
                        .max_by_key(|adapter| {
                            (
                                adapter.features().contains(optional_features),
                                adapter.get_info().backend == wgpu::Backend::Metal,
                            )
                        }),
                };
                let adapter = adapter
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "No adapter available with required features: {:?}",
                            required_features
                        )
                    })?;
                let device_features = if adapter.features().contains(optional_features) {
                    required_features | optional_features
                } else {
                    required_features
                };
                (adapter, device_features)
            };

            // On non-macOS native, additionally require profiling features.
            #[cfg(all(not(target_os = "macos"), not(target_family = "wasm")))]
            let (adapter, device_features) = {
                let required_features = required_features
                    | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
                    | wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
                let mut qualifying: Vec<wgpu::Adapter> = adapters
                    .into_iter()
                    .filter(|adapter| adapter.features().contains(required_features))
                    .collect();
                anyhow::ensure!(
                    !qualifying.is_empty(),
                    "No adapter available with required features: {required_features:?}"
                );
                // Compat (voir src/compat.rs) : l'hôte choisit
                // parmi les adaptateurs aptes ; sans sélecteur, le premier.
                let index = options
                    .adapter_selector
                    .as_ref()
                    .and_then(|selector| {
                        let infos: Vec<wgpu::AdapterInfo> =
                            qualifying.iter().map(|adapter| adapter.get_info()).collect();
                        selector(&infos)
                    })
                    .unwrap_or(0);
                anyhow::ensure!(
                    index < qualifying.len(),
                    "adapter_selector returned out-of-range index {index}"
                );
                (qualifying.swap_remove(index), required_features)
            };

            // PIPELINE_STATISTICS_QUERY is best-effort on every platform (unlike
            // TIMESTAMP_QUERY/TIMESTAMP_QUERY_INSIDE_ENCODERS above, which are
            // already required outright on non-macOS): cross-backend wgpu 30
            // support for it is thin, so it's runtime-checked rather than folded
            // into `required_features` even on non-macOS.
            #[cfg(feature = "flamegraph")]
            let device_features = {
                let pipeline_statistics_query = wgpu::Features::PIPELINE_STATISTICS_QUERY;
                if adapter.features().contains(pipeline_statistics_query) {
                    device_features | pipeline_statistics_query
                } else {
                    device_features
                }
            };

            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features: device_features,
                    required_limits: wgpu::Limits {
                        max_binding_array_elements_per_shader_stage: 512,
                        ..adapter.limits()
                    },
                    ..Default::default()
                }))?;

            *HOST_DEVICE.lock() = Some((
                instance.clone(),
                adapter.clone(),
                device.clone(),
                queue.clone(),
            ));

            Ok(Self::with_gpu(WgpuGpu {
                instance,
                adapter,
                device,
                queue,
                desired_maximum_frame_latency: options.desired_maximum_frame_latency,
            }))
        } // end #[cfg(not(target_family = "wasm"))]
    }

    /// Async constructor for WASM. Enumerates adapters and creates device+buffers
    /// without blocking (pollster::block_on panics on WASM's no_threads impl).
    #[cfg(target_family = "wasm")]
    pub async fn new_async(options: &WgpuOptions) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });

        let wgpui_features = wgpu::Features::empty();

        let required_features = wgpui_features | options.additional_features;

        let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;

        let adapter = adapters
            .into_iter()
            .find(|adapter| adapter.features().contains(required_features))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No adapter available with required features: {:?}",
                    required_features
                )
            })?;

        let device_features = required_features;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: device_features,
                required_limits: wgpu::Limits {
                    max_binding_array_elements_per_shader_stage: 512,
                    ..adapter.limits()
                },
                ..Default::default()
            })
            .await?;

        Ok(Self::with_gpu(WgpuGpu {
            instance,
            adapter,
            device,
            queue,
            desired_maximum_frame_latency: options.desired_maximum_frame_latency,
        }))
    }

}

/// Bytes per texel for `format`, for GPU memory accounting (Phase 3 of the
/// profiling epic, issue #59). Falls back to 4 (the size of every format
/// WGPUI actually creates textures with) for the handful of combined
/// depth/stencil formats where `block_copy_size` needs an aspect to answer
/// -- none of WGPUI's own textures use those formats, so this only matters
/// for correctness-in-principle, not any real texture this crate creates.
#[cfg(feature = "flamegraph")]
pub(crate) fn texel_size(format: wgpu::TextureFormat) -> u64 {
    format.block_copy_size(None).unwrap_or(4) as u64
}

/// Total bytes backing a texture: every mip level's `width * height *
/// texel_size`, times array/depth layers. WGPUI only ever creates
/// single-mip, single-layer 2D textures today, so this is normally just
/// `width * height * texel_size`; the general form costs nothing extra to
/// write and stays correct if that ever changes.
#[cfg(feature = "flamegraph")]
pub(crate) fn texture_memory_bytes(texture: &wgpu::Texture) -> u64 {
    let bytes_per_texel = texel_size(texture.format());
    let mut total = 0u64;
    for mip in 0..texture.mip_level_count() {
        let width = (texture.width() >> mip).max(1) as u64;
        let height = (texture.height() >> mip).max(1) as u64;
        total += width * height * bytes_per_texel;
    }
    total * (texture.depth_or_array_layers() as u64)
}

} // mod wgpu_context

/// Ensures a buffer is large enough to hold the required size.
/// If the buffer is too small, it will be recreated with the new size.
pub(super) fn ensure_buffer_size<G: Gpu>(
    gpu: &G,
    buffer: &Mutex<G::Buffer>,
    required_size: u64,
    label: &str,
    usage: BufferUsage,
) {
    let mut buffer_guard = buffer.lock();
    let current_size = G::buffer_size(&buffer_guard);
    if current_size < required_size {
        // Recreate buffer with new size (add some headroom to avoid frequent reallocations)
        let new_size = (required_size * 3 / 2).max(required_size + 1024 * 1024);
        *buffer_guard = gpu.create_buffer(label, new_size, usage);
    }
}

#[cfg(all(test, feature = "flamegraph"))]
mod tests {
    use super::{WgpuContext, WgpuOptions, texel_size};

    // `texture_memory_bytes`/atlas/surface-registry memory accounting all
    // ultimately reduce to `texel_size`, and that's the one piece testable
    // without a real `wgpu::Device` (`wgpu::TextureFormat` is a plain enum,
    // no adapter/device required).
    #[test]
    fn texel_size_matches_known_format_sizes() {
        assert_eq!(texel_size(wgpu::TextureFormat::R8Unorm), 1);
        assert_eq!(texel_size(wgpu::TextureFormat::Rgba8Unorm), 4);
        assert_eq!(texel_size(wgpu::TextureFormat::Bgra8UnormSrgb), 4);
        assert_eq!(texel_size(wgpu::TextureFormat::Rgba16Float), 8);
    }

    // Regression test for a real crash: phase 4's own `finish_and_poll_...`
    // test in `flamegraph_gpu.rs` proved the *readback logic* works, but it
    // built its own throwaway source buffer with `COPY_SRC` set explicitly --
    // it never exercised these seven *actual* fixed buffers this file
    // creates, so it couldn't have caught (and didn't catch) that they were
    // missing `COPY_SRC` until this fix. That gap let a `copy_buffer_to_buffer`
    // from any of them hard-panic the whole app the first time a deep capture
    // ran, in any real (non-test) binary. This test goes through the real
    // `WgpuContext::new` construction path -- the actual bug location -- and
    // uses a push/pop error scope (rather than relying on `wgpu`'s default
    // uncaptured-error handler, which is what panicked in the field) so a
    // regression here fails the assertion cleanly instead of aborting the
    // test process.
    #[test]
    fn fixed_buffers_created_with_flamegraph_feature_support_copy_buffer_to_buffer_readback() {
        let Ok(context) = WgpuContext::new(&WgpuOptions::default()) else {
            eprintln!(
                "skipping fixed_buffers_created_with_flamegraph_feature_support_copy_buffer_to_buffer_readback: no wgpu adapter available in this environment"
            );
            return;
        };

        let buffers: [(&str, &parking_lot::Mutex<wgpu::Buffer>); 6] = [
            ("quads_buffer", &context.quads_buffer),
            ("shadows_buffer", &context.shadows_buffer),
            ("underlines_buffer", &context.underlines_buffer),
            ("mono_sprites_buffer", &context.mono_sprites_buffer),
            ("poly_sprites_buffer", &context.poly_sprites_buffer),
            ("backdrop_filters_buffer", &context.backdrop_filters_buffer),
        ];

        for (label, buffer) in buffers {
            assert_copy_buffer_to_buffer_read_accepted(&context, label, buffer);
        }

        // `paths_vertices_buffer` uses `STORAGE | COPY_DST` (no `VERTEX`) as its
        // base usage, so it's checked separately rather than folded into the
        // array above (which assumes the six `VERTEX`-based buffers' uniform
        // shape) -- still the same `deep_capture_readback` flag, same bug class.
        assert_copy_buffer_to_buffer_read_accepted(
            &context,
            "paths_vertices_buffer",
            &context.paths_vertices_buffer,
        );
    }

    fn assert_copy_buffer_to_buffer_read_accepted(
        context: &WgpuContext,
        label: &str,
        buffer: &parking_lot::Mutex<wgpu::Buffer>,
    ) {
        let error_scope = context
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);

        let source = buffer.lock();
        let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("regression_test_staging_buffer"),
            size: source.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&source, 0, &staging, 0, source.size());
        drop(source);
        context.queue.submit(Some(encoder.finish()));

        let error = pollster::block_on(error_scope.pop());
        assert!(
            error.is_none(),
            "{label} should accept a copy_buffer_to_buffer read (COPY_SRC), got: {error:?}"
        );
    }
}
