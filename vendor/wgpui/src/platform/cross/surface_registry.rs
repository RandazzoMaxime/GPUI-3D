use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

/// Ce que le registre fait d'une taille demandee. **Trois issues terminales et
/// aucune qui signifie « reessayer »** : c'est la propriete qui remplace
/// l'attente active `while !resize(..) { sleep(8ms) }`, laquelle ne pouvait pas
/// se terminer si `redraw_pending` restait pose par un bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResizeDecision {
    /// Taille deja en place. Une cible retenue devient caduque.
    AlreadyThere,
    /// Le compositeur lit les tampons : retenir la cible plutot que la perdre.
    /// `clear_redraw_pending` l'appliquera au point ou il les lache.
    Retain,
    /// Personne ne lit : reallouer maintenant.
    Apply,
}

/// La decision, sans GPU ni verrou — c'est la seule partie testable de
/// `SurfaceRegistry::resize`, le reste etant de l'allocation de texture.
pub(crate) fn decide_resize(
    current: (u32, u32),
    target: (u32, u32),
    redraw_pending: bool,
) -> ResizeDecision {
    if current == target {
        ResizeDecision::AlreadyThere
    } else if redraw_pending {
        ResizeDecision::Retain
    } else {
        ResizeDecision::Apply
    }
}

/// Quelle image conserver a travers une reallocation. Une seconde
/// reallocation sans trame intercalee ne doit PAS reporter l'affichage
/// courant : il est deja neuf, donc noir. On garde la premiere image encore
/// valide jusqu'a ce qu'une vraie trame la remplace.
fn stale_to_carry<T>(already_held: Option<T>, current_display: T) -> Option<T> {
    Some(already_held.unwrap_or(current_display))
}

// GPUI-3D : défini dans `scene` (indépendant du backend), réexporté ici.
pub use crate::scene::SurfaceId;

use super::hal::{Gpu, TextureUsage};

/// Triple-buffered surface for lock-free rendering.
///
/// Uses three buffers with atomic index swaps:
/// - `rendering`: Currently being rendered by external thread
/// - `ready`: Latest complete frame, ready to display
/// - `display`: Currently being composited by GPUI
///
/// This allows external thread and compositor to run independently without blocking.
struct TripleBuffer<G: Gpu> {
    textures: [G::Texture; 3],
    views: [G::TextureView; 3],

    // Packed state: 2 bits each for rendering/ready/display indices.
    // layout: [display(2-bit) | ready(2-bit) | rendering(2-bit)]
    state: AtomicU8,

    // Redraw coalescing: prevents flooding compositor with thousands of requests/sec
    redraw_pending: std::sync::atomic::AtomicBool,

    // Taille demandée pendant que le compositeur lisait les tampons. Appliquée
    // par `clear_redraw_pending`, le point où il n'y touche plus. Personne
    // n'attend : le redimensionnement est coopératif, pas sondé.
    pending_resize: Option<(u32, u32)>,

    // Dernier contenu VALIDE, garde en vie a travers une reallocation. Les trois
    // textures neuves sont noires : les presenter fait clignoter la fenetre
    // pendant un glissement de bordure (constat operateur du 2026-09-18). On
    // reaffiche donc l'ancienne image, etiree, jusqu'a la premiere vraie trame
    // a la nouvelle taille. La vue suffit a maintenir la texture en vie, wgpu
    // comptant les references.
    stale_display: Option<G::TextureView>,

    // Monotonic count of producer swaps (rendering → ready): one increment per
    // frame the external renderer presents.
    frame_generation: AtomicU64,
    // The `frame_generation` value the compositor last swapped into `display`.
    // The compositor swaps `ready → display` only when these differ, so a paint
    // with no newly produced frame holds the current display buffer instead of
    // rotating to a stale one.
    last_composited_generation: AtomicU64,

    width: u32,
    height: u32,
    format: G::Format,
}

impl<G: Gpu> TripleBuffer<G> {
    #[inline]
    fn pack_state(rendering: u8, ready: u8, display: u8) -> u8 {
        debug_assert!(rendering < 3 && ready < 3 && display < 3);
        debug_assert!(rendering != ready && ready != display && display != rendering);
        (display << 4) | (ready << 2) | rendering
    }

    #[inline]
    fn unpack_state(state: u8) -> (u8, u8, u8) {
        let rendering = state & 0x03;
        let ready = (state >> 2) & 0x03;
        let display = (state >> 4) & 0x03;
        (rendering, ready, display)
    }
}

/// Thread-safe registry of all active WGPU surfaces.
/// Maps `SurfaceId` to triple-buffered texture sets.
pub struct SurfaceRegistry<G: Gpu> {
    gpu: Arc<G>,
    surfaces: Mutex<HashMap<SurfaceId, TripleBuffer<G>>>,
    next_id: AtomicU64,
    seen_generations: AtomicU64,
    /// GPUI-3D : `VkQueue` exige une synchronisation externe ; tout `submit`/`present`
    /// wgpu du compositeur et tout `vkQueueSubmit` d'un moteur natif le prennent.
    queue_lock: parking_lot::Mutex<()>,
}

impl<G: Gpu> SurfaceRegistry<G> {
    /// GPUI-3D : les tampons sont initialisés à leur création pour les moteurs natifs
    /// (voir [`Gpu::init_external_textures`]).
    pub fn new(gpu: Arc<G>) -> Self {
        Self {
            gpu,
            surfaces: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            seen_generations: AtomicU64::new(0),
            queue_lock: parking_lot::Mutex::new(()),
        }
    }

    /// GPUI-3D : verrou de la queue partagée (voir le champ `queue_lock`).
    pub fn queue_lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.queue_lock.lock()
    }

    /// Create a new triple-buffered surface. Returns its `SurfaceId`.
    pub fn create(&self, width: u32, height: u32, format: G::Format) -> SurfaceId {
        let id = SurfaceId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let tb = self.create_triple_buffer(width, height, format);
        self.surfaces.lock().unwrap().insert(id, tb);
        id
    }

    /// Atomically swap rendering and ready buffers (called by external thread after rendering).
    ///
    /// This is the "present" operation - it makes the newly rendered frame available
    /// to the compositor and gives the external thread a recycled buffer to render into.
    ///
    /// Returns immediately without blocking.
    pub fn swap_rendering_ready(&self, id: SurfaceId) {
        if let Some(tb) = self.surfaces.lock().unwrap().get(&id) {
            let current = tb.state.load(Ordering::Acquire);
            let (rendering, ready, display) = TripleBuffer::<G>::unpack_state(current);

            log::trace!(
                "[surface_id={:?}] swap_rendering_ready called - state before: rendering={}, ready={}, display={}",
                id,
                rendering,
                ready,
                display
            );

            // Atomic swap: rendering ↔ ready
            let mut current = tb.state.load(Ordering::Acquire);
            loop {
                let (rendering, ready, display) = TripleBuffer::<G>::unpack_state(current);
                let next = TripleBuffer::<G>::pack_state(ready, rendering, display);
                match tb
                    .state
                    .compare_exchange(current, next, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => break,
                    Err(updated) => current = updated,
                }
            }

            // A newly rendered frame now sits in `ready`; advance the generation
            // so the compositor swaps it to `display` exactly once.
            tb.frame_generation.fetch_add(1, Ordering::Release);
        }
    }

    /// Get the rendering buffer's `TextureView` (what external code renders into).
    pub fn back_view(&self, id: SurfaceId) -> Option<G::TextureView> {
        let surfaces = self.surfaces.lock().unwrap();
        surfaces.get(&id).map(|tb| {
            let (rendering, _, _) = TripleBuffer::<G>::unpack_state(tb.state.load(Ordering::Acquire));
            tb.views[rendering as usize].clone()
        })
    }

    /// Compat (voir src/compat.rs) : la `Texture` du buffer de
    /// rendu, quand le rendu externe a besoin de plus qu'une vue (copies,
    /// dimensions, format).
    pub fn back_texture(&self, id: SurfaceId) -> Option<G::Texture> {
        let surfaces = self.surfaces.lock().unwrap();
        surfaces.get(&id).map(|tb| {
            let (rendering, _, _) = TripleBuffer::<G>::unpack_state(tb.state.load(Ordering::Acquire));
            tb.textures[rendering as usize].clone()
        })
    }

    /// Get the display buffer's `TextureView` (what the compositor reads from).
    pub fn front_view(&self, id: SurfaceId) -> Option<G::TextureView> {
        let surfaces = self.surfaces.lock().unwrap();
        surfaces.get(&id).map(|tb| {
            // Tant qu'aucune trame n'a ete rendue a la taille courante, le
            // tampon d'affichage est une texture neuve, donc noire.
            if let Some(stale) = &tb.stale_display {
                return stale.clone();
            }
            let (_, _, display) = TripleBuffer::<G>::unpack_state(tb.state.load(Ordering::Acquire));
            tb.views[display as usize].clone()
        })
    }

    /// Atomically retrieve both the rendering view and the corresponding texture
    /// dimensions. This is useful when a caller needs to create auxiliary
    /// resources (e.g. a depth buffer) that must exactly match the view's size.
    pub fn lock_and_get_back_with_size(
        &self,
        id: SurfaceId,
    ) -> Option<(G::TextureView, (u32, u32))> {
        let surfaces = self.surfaces.lock().unwrap();
        surfaces.get(&id).map(|tb| {
            let (rendering, _, _) = TripleBuffer::<G>::unpack_state(tb.state.load(Ordering::Acquire));
            (tb.views[rendering as usize].clone(), (tb.width, tb.height))
        })
    }

    /// Resize all three buffers, creating new textures with GPU synchronization.
    ///
    /// SAFETY: Waits for all pending GPU work to complete before destroying textures.
    /// This prevents use-after-free and ensures all GPU commands finish before
    /// texture resources are released.
    ///
    /// Also skips resize if compositor is actively using the buffers (redraw_pending).
    ///
    /// Rend `true` quand la nouvelle taille est en place. `false` veut dire
    /// « différée » et non « échouée » : la cible est retenue et
    /// [`clear_redraw_pending`](Self::clear_redraw_pending) l'applique dès que
    /// le compositeur lâche les tampons. L'appelant ne doit ni boucler ni
    /// attendre — une attente active sur ce drapeau tenait tout un glissement
    /// de bordure et ne pouvait pas se terminer si le drapeau restait posé.
    pub fn resize(&self, id: SurfaceId, width: u32, height: u32) -> bool {
        let mut surfaces = self.surfaces.lock().unwrap();
        if let Some(tb) = surfaces.get_mut(&id) {
            let pending = tb.redraw_pending.load(Ordering::Relaxed);
            match decide_resize((tb.width, tb.height), (width, height), pending) {
                ResizeDecision::AlreadyThere => tb.pending_resize = None,
                // CRITICAL: Don't resize while compositor is rendering this surface!
                ResizeDecision::Retain => {
                    tb.pending_resize = Some((width, height));
                    return false;
                }
                ResizeDecision::Apply => self.apply_resize(tb, width, height),
            }
            return true;
        }
        false
    }

    /// NOTE: We do NOT call device.poll() here because:
    /// 1. The render thread owns the device and may be actively using it
    /// 2. Calling poll from compositor thread causes device corruption
    /// 3. WGPU internally ref-counts textures, so old views remain valid until dropped
    /// 4. Callers only reach this with `redraw_pending` clear
    fn apply_resize(&self, tb: &mut TripleBuffer<G>, width: u32, height: u32) {
        // Reallouer d'abord, garder l'image ensuite : un glissement enchaine les
        // reallocations, et a la deuxieme le tampon d'affichage est deja neuf
        // donc noir. On reporte donc la PREMIERE image encore valide, pas la
        // courante, jusqu'a ce qu'une vraie trame la remplace.
        let (_, _, display) = TripleBuffer::<G>::unpack_state(tb.state.load(Ordering::Acquire));
        let carried = stale_to_carry(
            tb.stale_display.take(),
            tb.views[display as usize].clone(),
        );
        *tb = self.create_triple_buffer(width, height, tb.format);
        tb.stale_display = carried;
    }

    /// True tant qu'une taille demandée n'a pas été appliquée.
    pub fn has_pending_resize(&self, id: SurfaceId) -> bool {
        self.surfaces
            .lock()
            .unwrap()
            .get(&id)
            .is_some_and(|tb| tb.pending_resize.is_some())
    }

    /// Oublier une taille différée devenue obsolète.
    pub fn cancel_pending_resize(&self, id: SurfaceId) {
        if let Some(tb) = self.surfaces.lock().unwrap().get_mut(&id) {
            tb.pending_resize = None;
        }
    }

    /// Get the current size of a surface.
    pub fn size(&self, id: SurfaceId) -> Option<(u32, u32)> {
        let surfaces = self.surfaces.lock().unwrap();
        surfaces.get(&id).map(|tb| (tb.width, tb.height))
    }

    /// Remove a surface from the registry.
    pub fn remove(&self, id: SurfaceId) {
        self.surfaces.lock().unwrap().remove(&id);
    }

    /// Set the redraw pending flag, returning the previous value.
    /// Used by present() to coalesce multiple redraw requests.
    pub fn set_redraw_pending(&self, id: SurfaceId) -> bool {
        if let Some(tb) = self.surfaces.lock().unwrap().get(&id) {
            tb.redraw_pending.swap(true, Ordering::Relaxed)
        } else {
            false
        }
    }

    /// Clear the redraw pending flag.
    /// Called by the compositor after consuming a frame.
    ///
    /// C'est aussi le point de rendez-vous du redimensionnement coopératif :
    /// le compositeur ne lit plus les tampons, une taille retenue par
    /// [`resize`](Self::resize) s'applique donc ici. La vue déjà clonée dans la
    /// passe en cours reste valide — wgpu compte les références des textures.
    pub fn clear_redraw_pending(&self, id: SurfaceId) {
        if let Some(tb) = self.surfaces.lock().unwrap().get_mut(&id) {
            tb.redraw_pending.store(false, Ordering::Relaxed);
            if let Some((width, height)) = tb.pending_resize.take() {
                if (tb.width, tb.height) != (width, height) {
                    self.apply_resize(tb, width, height);
                }
            }
        }
    }

    /// Get all surfaces that have pending redraws.
    /// Used by the fast blit path to check which surfaces need updating.
    pub fn get_pending_surfaces(&self) -> Vec<SurfaceId> {
        let surfaces = self.surfaces.lock().unwrap();
        surfaces
            .iter()
            .filter(|(_, tb)| tb.redraw_pending.load(Ordering::Relaxed))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Swap `ready → display` only if the external renderer has presented a new
    /// frame since the last successful compositor swap. Returns `true` if a swap
    /// occurred; when it returns `false`, the caller should keep compositing the
    /// current `display` buffer (via [`front_view`](Self::front_view)) unchanged.
    ///
    /// This is the gated counterpart to [`swap_ready_display`](Self::swap_ready_display).
    /// The GPUI paint path composites a surface every frame regardless of whether
    /// the producer rendered anything, so an *ungated* swap there rotates `display`
    /// to a stale buffer whenever the producer skipped a frame (engine-lock
    /// contention, a pending resize, …), making the canvas strobe. Gating on the
    /// producer generation keeps `display` steady until a genuinely new frame is
    /// ready.
    ///
    /// Runs entirely under the surfaces mutex, so the generation compare, the
    /// buffer swap, and the generation store are atomic with respect to the
    /// producer's `swap_rendering_ready*`.
    pub fn swap_ready_display_if_new(&self, id: SurfaceId) -> bool {
        if let Some(tb) = self.surfaces.lock().unwrap().get_mut(&id) {
            let current_gen = tb.frame_generation.load(Ordering::Acquire);
            let last = tb.last_composited_generation.load(Ordering::Acquire);
            if !Self::should_composite_swap(current_gen, last) {
                return false;
            }
            // Voir `swap_ready_display` : l'image conservee cede a la vraie trame.
            tb.stale_display = None;

            // Atomic swap: ready ↔ display
            let mut current = tb.state.load(Ordering::Acquire);
            loop {
                let (rendering, ready, display) = TripleBuffer::<G>::unpack_state(current);
                let next = TripleBuffer::<G>::pack_state(rendering, display, ready);
                match tb
                    .state
                    .compare_exchange(current, next, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => break,
                    Err(updated) => current = updated,
                }
            }

            tb.last_composited_generation
                .store(current_gen, Ordering::Release);
            return true;
        }
        false
    }

    /// True if the producer has published a frame that the compositor has not
    /// promoted to `display` yet.
    ///
    /// The triple buffer holds exactly **one** `ready` frame: publishing a
    /// second before the first is consumed makes `swap_rendering_ready` recycle
    /// the unconsumed buffer as the next render target, discarding that frame's
    /// pixels. The GPU work, command buffers and per-frame allocations behind it
    /// are not free, though — so an external render thread that ignores this is
    /// an unbounded producer feeding a display-rate consumer.
    ///
    /// Render threads should use this for backpressure: skip producing while it
    /// returns true. Prefer a *bounded* wait — the fast-blit consumer
    /// ([`swap_ready_display`](Self::swap_ready_display)) does not advance the
    /// composited generation, so this can stay true indefinitely on that path.
    pub fn has_unconsumed_frame(&self, id: SurfaceId) -> bool {
        self.surfaces
            .lock()
            .unwrap()
            .get(&id)
            .is_some_and(|tb| {
                Self::should_composite_swap(
                    tb.frame_generation.load(Ordering::Acquire),
                    tb.last_composited_generation.load(Ordering::Acquire),
                )
            })
    }

    /// [`has_unconsumed_frame`](Self::has_unconsumed_frame) sur l'ensemble des
    /// surfaces : « une trame publiée attend d'être composée ».
    /// Vrai **une seule fois** par trame publiée, pour réveiller la boucle
    /// d'événements. Front et non niveau : une surface enregistrée dont
    /// l'élément n'est pas peint n'est jamais composée, donc son
    /// `any_unconsumed_frame` resterait vrai et ferait tourner la boucle au
    /// plafond de `IDLE_POLL_INTERVAL` en permanence.
    pub fn take_new_frame(&self) -> bool {
        let total: u64 = self
            .surfaces
            .lock()
            .unwrap()
            .values()
            .map(|tb| tb.frame_generation.load(Ordering::Acquire))
            .sum();
        self.seen_generations.swap(total, Ordering::AcqRel) != total
    }

    pub fn any_unconsumed_frame(&self) -> bool {
        self.surfaces.lock().unwrap().values().any(|tb| {
            Self::should_composite_swap(
                tb.frame_generation.load(Ordering::Acquire),
                tb.last_composited_generation.load(Ordering::Acquire),
            )
        })
    }

    /// Pure decision function used by [`swap_ready_display_if_new`](Self::swap_ready_display_if_new):
    /// the compositor should swap `ready → display` iff the producer has advanced
    /// the generation since the compositor last presented. Both start at `0`, so
    /// the first compositor paint before any frame is produced is a no-op (keeps
    /// the initial buffer). Split out so the gating logic is unit-testable without
    /// a GPU device.
    #[inline]
    pub fn should_composite_swap(current_generation: u64, last_composited: u64) -> bool {
        current_generation != last_composited
    }

    fn create_triple_buffer(&self, width: u32, height: u32, format: G::Format) -> TripleBuffer<G> {
        let w = width.max(1);
        let h = height.max(1);

        // Phase 4b of the profiling epic (issue #72) reads a surface's
        // currently-displayed triple-buffer texture back via
        // `copy_texture_to_buffer` during a triggered GPU deep capture,
        // which requires `COPY_SRC` on the source texture or wgpu's
        // validator rejects the encoder outright -- the exact same class of
        // hard, process-wide panic `render_context.rs`'s fixed buffers hit
        // before `COPY_SRC` was added to them (see that fix's commit
        // message for the full incident). Add it only when the capture code
        // that actually needs it is compiled in, so a non-`flamegraph`
        // build's surface textures are byte-for-byte the same as before
        // this change.
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED;
        #[cfg(feature = "flamegraph")]
        let usage = usage | TextureUsage::COPY_SRC;

        let textures = ["surface_buffer_0", "surface_buffer_1", "surface_buffer_2"]
            .map(|label| self.gpu.create_texture(label, w, h, format, usage));
        {
            let _queue = self.queue_lock();
            self.gpu.init_external_textures([&textures[0], &textures[1], &textures[2]]);
        }
        let views = [
            G::create_view(&textures[0]),
            G::create_view(&textures[1]),
            G::create_view(&textures[2]),
        ];

        TripleBuffer {
            textures,
            views,
            state: AtomicU8::new(TripleBuffer::<G>::pack_state(0, 1, 2)),
            redraw_pending: std::sync::atomic::AtomicBool::new(false),
            pending_resize: None,
            stale_display: None,
            frame_generation: AtomicU64::new(0),
            last_composited_generation: AtomicU64::new(0),
            width: w,
            height: h,
            format,
        }
    }
}

#[cfg(feature = "flamegraph")]
impl SurfaceRegistry<super::hal::wgpu::WgpuGpu> {
    /// One surface's currently-displayed triple-buffer texture, snapshotted
    /// for a triggered GPU deep capture (Phase 4b of the profiling epic,
    /// issue #72). Distinct from `front_view`, which only exposes a
    /// `TextureView` -- enough to bind the surfaces pipeline, but
    /// `copy_texture_to_buffer` needs the underlying `wgpu::Texture`
    /// directly, plus the pixel dimensions/texel size a caller needs to
    /// compute `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` row padding. A poisoned
    /// lock is treated as "nothing to snapshot" rather than propagating the
    /// panic -- this is a diagnostic-only read, matching `memory_usage`'s
    /// same choice just below.
    #[cfg(feature = "flamegraph")]
    pub(crate) fn front_texture_snapshot(&self, id: SurfaceId) -> Option<SurfaceTextureSnapshot> {
        let surfaces = self.surfaces.lock().ok()?;
        surfaces.get(&id).map(|tb| {
            let (_, _, display) = TripleBuffer::<super::hal::wgpu::WgpuGpu>::unpack_state(tb.state.load(Ordering::Acquire));
            SurfaceTextureSnapshot {
                texture: tb.textures[display as usize].clone(),
                width: tb.width,
                height: tb.height,
                bytes_per_pixel: super::render_context::texel_size(tb.format) as u32,
            }
        })
    }

    /// Sum of every registered surface's three triple-buffered textures
    /// (Phase 3 of the profiling epic, issue #59). A poisoned lock (some
    /// other thread already panicked while holding it) is treated as
    /// contributing zero rather than panicking here too.
    #[cfg(feature = "flamegraph")]
    pub(crate) fn memory_usage(&self) -> u64 {
        let surfaces = match self.surfaces.lock() {
            Ok(surfaces) => surfaces,
            Err(_) => return 0,
        };
        surfaces
            .values()
            .flat_map(|triple_buffer| triple_buffer.textures.iter())
            .map(super::render_context::texture_memory_bytes)
            .sum()
    }
}

/// See [`SurfaceRegistry::front_texture_snapshot`].
#[cfg(feature = "flamegraph")]
pub(crate) struct SurfaceTextureSnapshot {
    pub(crate) texture: wgpu::Texture,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bytes_per_pixel: u32,
}

#[cfg(all(test, feature = "flamegraph"))]
mod flamegraph_tests {
    use super::SurfaceRegistry;

    /// Creates a headless (surface-less) `wgpu::Device`/`Queue`, the same
    /// pattern `flamegraph_gpu.rs`/`flamegraph_replay.rs` already use for
    /// their own GPU-backed tests (see either module's `create_headless_device`
    /// doc comment for why `enumerate_adapters` + pick-first is used instead
    /// of `request_adapter`, and why a missing adapter skips rather than
    /// fails the test in this sandbox).
    use super::tests::headless_gpu;

    /// Regression test for the exact bug class `render_context.rs`'s fixed
    /// buffers hit before that fix landed (see `create_triple_buffer`'s
    /// `deep_capture_readback` comment): a surface's triple-buffer textures
    /// need `COPY_SRC` for a triggered GPU deep capture (issue #72) to read
    /// their content back via `copy_texture_to_buffer`, or wgpu's validator
    /// rejects the encoder outright -- a hard, process-wide panic by
    /// default, not a soft/recoverable error. This goes through the real
    /// `SurfaceRegistry::create` construction path -- the same one
    /// `create_wgpu_surface` uses in production -- and uses a push/pop error
    /// scope (rather than relying on wgpu's default uncaptured-error
    /// handler, which is what would panic) so a regression here fails the
    /// assertion cleanly instead of aborting the test process.
    #[test]
    fn surface_textures_created_with_flamegraph_feature_support_copy_texture_to_buffer_readback() {
        let Some(gpu) = headless_gpu() else {
            eprintln!(
                "skipping surface_textures_created_with_flamegraph_feature_support_copy_texture_to_buffer_readback: no wgpu adapter available in this environment"
            );
            return;
        };
        let (device, queue) = (gpu.device.clone(), gpu.queue.clone());

        let registry = SurfaceRegistry::new(gpu);
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let width = 16u32;
        let height = 16u32;
        let surface_id = registry.create(width, height, format);

        let snapshot = registry
            .front_texture_snapshot(surface_id)
            .expect("a surface just created should have a snapshot-able front texture");
        assert_eq!(snapshot.width, width);
        assert_eq!(snapshot.height, height);
        assert_eq!(snapshot.bytes_per_pixel, 4);

        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let bytes_per_pixel = snapshot.bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("regression_test_staging_buffer"),
            size: (padded_bytes_per_row as u64) * (height as u64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &snapshot.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let error = pollster::block_on(error_scope.pop());
        assert!(
            error.is_none(),
            "surface front texture should accept a copy_texture_to_buffer read (COPY_SRC), got: {error:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{SurfaceRegistry, TripleBuffer};

    type G = crate::platform::cross::hal::wgpu::WgpuGpu;

    /// [`create_headless_device`] wrapped as the renderer's GPU.
    pub(super) fn headless_gpu() -> Option<Arc<G>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .next()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
        Some(Arc::new(G { instance, adapter, device, queue, desired_maximum_frame_latency: 2 }))
    }

    /// Headless (surface-less) `wgpu::Device`/`Queue`; a missing adapter skips
    /// rather than fails the test in this sandbox.
    pub(super) fn create_headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .next()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    #[test]
    fn any_unconsumed_frame_follows_produce_then_composite() {
        let Some(gpu) = headless_gpu() else {
            eprintln!("skip any_unconsumed_frame_follows_produce_then_composite: pas d'adaptateur wgpu");
            return;
        };
        let registry = SurfaceRegistry::new(gpu);
        assert!(!registry.any_unconsumed_frame(), "registre vide");
        let id = registry.create(16, 16, wgpu::TextureFormat::Rgba8Unorm);
        assert!(!registry.any_unconsumed_frame(), "surface neuve, rien de publie");

        registry.swap_rendering_ready(id);
        assert!(registry.any_unconsumed_frame(), "trame publiee, pas encore composee");

        assert!(registry.swap_ready_display_if_new(id));
        assert!(!registry.any_unconsumed_frame(), "trame composee");
    }

    #[test]
    fn take_new_frame_is_an_edge_not_a_level() {
        let Some(gpu) = headless_gpu() else {
            eprintln!("skip take_new_frame_is_an_edge_not_a_level: pas d'adaptateur wgpu");
            return;
        };
        let registry = SurfaceRegistry::new(gpu);
        assert!(!registry.take_new_frame(), "registre vide");
        let id = registry.create(16, 16, wgpu::TextureFormat::Rgba8Unorm);
        assert!(!registry.take_new_frame(), "surface neuve, rien de publie");

        registry.swap_rendering_ready(id);
        assert!(registry.take_new_frame(), "trame publiee");
        // Jamais composee : le niveau reste vrai, le front doit etre retombe.
        assert!(registry.any_unconsumed_frame(), "toujours pas composee");
        assert!(!registry.take_new_frame(), "un front, pas un niveau");

        registry.swap_rendering_ready(id);
        assert!(registry.take_new_frame(), "trame suivante");
    }

    // Minimal, GPU-free model of the three-buffer roles. We track which "frame"
    // (a monotonically increasing id) currently lives in each physical buffer,
    // so we can assert what the compositor would actually display after a
    // sequence of producer/consumer swaps — the real textures are irrelevant to
    // the swap/gating logic under test.
    struct Model {
        state: u8,
        /// Frame id stored in each physical buffer (0 = never rendered).
        contents: [u32; 3],
        /// Producer generation (count of rendering→ready swaps).
        generation: u64,
        /// Generation the compositor last swapped into display.
        last_composited: u64,
    }

    impl Model {
        fn new() -> Self {
            Self {
                state: TripleBuffer::<G>::pack_state(0, 1, 2),
                contents: [0; 3],
                generation: 0,
                last_composited: 0,
            }
        }

        /// External renderer draws `frame` into the rendering buffer, then swaps
        /// rendering ↔ ready (mirrors `swap_rendering_ready*`).
        fn produce(&mut self, frame: u32) {
            let (rendering, ready, display) = TripleBuffer::<G>::unpack_state(self.state);
            self.contents[rendering as usize] = frame;
            self.state = TripleBuffer::<G>::pack_state(ready, rendering, display);
            self.generation += 1;
        }

        /// Old, ungated compositor: always swaps ready ↔ display.
        fn composite_ungated(&mut self) {
            let (rendering, ready, display) = TripleBuffer::<G>::unpack_state(self.state);
            self.state = TripleBuffer::<G>::pack_state(rendering, display, ready);
        }

        /// New, gated compositor: swaps only when a new frame was produced
        /// (mirrors `swap_ready_display_if_new`).
        fn composite_gated(&mut self) {
            if !SurfaceRegistry::<G>::should_composite_swap(self.generation, self.last_composited) {
                return;
            }
            let (rendering, ready, display) = TripleBuffer::<G>::unpack_state(self.state);
            self.state = TripleBuffer::<G>::pack_state(rendering, display, ready);
            self.last_composited = self.generation;
        }

        /// The frame the compositor would currently display.
        fn displayed_frame(&self) -> u32 {
            let (_, _, display) = TripleBuffer::<G>::unpack_state(self.state);
            self.contents[display as usize]
        }
    }

    #[test]
    fn should_composite_swap_only_on_new_generation() {
        assert!(!SurfaceRegistry::<G>::should_composite_swap(0, 0));
        assert!(!SurfaceRegistry::<G>::should_composite_swap(5, 5));
        assert!(SurfaceRegistry::<G>::should_composite_swap(1, 0));
        assert!(SurfaceRegistry::<G>::should_composite_swap(6, 5));
    }

    #[test]
    fn indices_stay_a_permutation_across_swaps() {
        // Any sequence of transpositions must keep the three roles distinct,
        // otherwise `pack_state`'s debug asserts would fire and buffers alias.
        let mut m = Model::new();
        for frame in 1..=20u32 {
            m.produce(frame);
            m.composite_gated();
            let (r, ready, d) = TripleBuffer::<G>::unpack_state(m.state);
            assert!(r != ready && ready != d && d != r, "roles collided: {:?}", (r, ready, d));
        }
    }

    #[test]
    fn ungated_compositor_regresses_to_stale_frame() {
        // Reproduces the bug: one produced frame, then the compositor paints
        // twice (as the GPUI path does every frame). The second, unpaired swap
        // rotates `display` to a buffer holding an older frame.
        let mut m = Model::new();
        m.produce(1);
        m.composite_ungated();
        assert_eq!(m.displayed_frame(), 1, "first composite shows the new frame");

        m.composite_ungated(); // unpaired paint, no new frame produced
        assert_ne!(
            m.displayed_frame(),
            1,
            "BUG: unpaired ungated swap regressed display off the latest frame"
        );
    }

    #[test]
    fn gated_compositor_holds_latest_frame_on_unpaired_paints() {
        // The fix: without a new produced frame, repeated compositor paints keep
        // showing the latest frame instead of strobing to a stale buffer.
        let mut m = Model::new();
        m.produce(1);
        m.composite_gated();
        assert_eq!(m.displayed_frame(), 1);

        for _ in 0..10 {
            m.composite_gated(); // unpaired paints (viewport skipped a frame)
            assert_eq!(
                m.displayed_frame(),
                1,
                "gated compositor must hold the last frame with no new production"
            );
        }
    }

    #[test]
    fn gated_compositor_tracks_new_frames() {
        // Normal 1:1 pairing advances the displayed frame each time.
        let mut m = Model::new();
        for frame in 1..=8u32 {
            m.produce(frame);
            m.composite_gated();
            assert_eq!(m.displayed_frame(), frame);
        }
    }

    #[test]
    fn gated_compositor_shows_latest_when_producer_outruns_compositor() {
        // Producer renders several frames before one composite; the compositor
        // should jump straight to the newest completed frame, never a stale one.
        let mut m = Model::new();
        m.produce(1);
        m.produce(2);
        m.produce(3);
        m.composite_gated();
        assert_eq!(m.displayed_frame(), 3);
    }
}

#[cfg(test)]
mod resize_decision_tests {
    use super::{decide_resize, ResizeDecision};

    #[test]
    fn la_garde_retient_la_cible_au_lieu_de_la_refuser() {
        assert_eq!(
            decide_resize((800, 600), (1024, 768), true),
            ResizeDecision::Retain,
            "compositeur en train de lire : la cible doit survivre a l'appel"
        );
        assert_eq!(
            decide_resize((800, 600), (1024, 768), false),
            ResizeDecision::Apply
        );
    }

    #[test]
    fn une_taille_deja_en_place_ne_retient_rien() {
        for pending in [false, true] {
            assert_eq!(
                decide_resize((1024, 768), (1024, 768), pending),
                ResizeDecision::AlreadyThere,
                "redraw_pending={pending}"
            );
        }
    }

    /// La propriete qui interdit le retour de l'attente active : quelle que
    /// soit l'entree, la decision est terminale. Aucune ne dit « reessayer »,
    /// donc aucun appelant ne peut en tirer une boucle.
    #[test]
    fn aucune_decision_ne_demande_de_reessayer() {
        for current in [(0, 0), (800, 600), (3440, 1369)] {
            for target in [(0, 0), (800, 600), (3440, 1369)] {
                for pending in [false, true] {
                    let d = decide_resize(current, target, pending);
                    assert!(
                        matches!(
                            d,
                            ResizeDecision::AlreadyThere
                                | ResizeDecision::Retain
                                | ResizeDecision::Apply
                        ),
                        "{current:?} -> {target:?} pending={pending} : {d:?}"
                    );
                    assert_eq!(
                        d == ResizeDecision::Retain,
                        pending && current != target,
                        "la retention n'est due qu'a la garde, {current:?} -> {target:?}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod stale_display_tests {
    use super::stale_to_carry;

    /// Sans reallocation prealable, c'est l'affichage courant qu'on garde.
    #[test]
    fn la_premiere_reallocation_reporte_l_affichage_courant() {
        assert_eq!(stale_to_carry(None, "image valide"), Some("image valide"));
    }

    /// Le cas qui produisait le clignotement noir : pendant un glissement, les
    /// reallocations s'enchainent et l'affichage courant est deja une texture
    /// neuve. Reporter celle-la afficherait du noir.
    #[test]
    fn une_reallocation_en_chaine_garde_la_premiere_image_valide() {
        let mut held = stale_to_carry(None, "image valide");
        for neuve in ["noir 1", "noir 2", "noir 3"] {
            held = stale_to_carry(held, neuve);
            assert_eq!(
                held,
                Some("image valide"),
                "une texture neuve ({neuve}) ne doit jamais devenir l'image conservee"
            );
        }
    }

    /// Une vraie trame libere l'image conservee ; la suivante repart de
    /// l'affichage, qui est alors valide.
    #[test]
    fn apres_une_vraie_trame_on_repart_de_l_affichage() {
        let held: Option<&str> = None; // remis a None par swap_ready_display
        assert_eq!(stale_to_carry(held, "trame rendue"), Some("trame rendue"));
    }
}
