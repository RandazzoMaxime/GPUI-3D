use std::sync::{Arc, Mutex};

use refineable::Refineable as _;

use crate::{
    App, Bounds, Corners, Element, ElementId, GlobalElementId, InspectorElementId, IntoElement, LayoutId,
    MouseButton, Pixels, Style, StyleRefinement, Styled, Window,
    platform::cross::{hal::wgpu::WgpuGpu, surface_registry::{SurfaceId, SurfaceRegistry}},
};

/// Inner state shared across clones of `WgpuSurfaceHandle`.
/// When the last clone is dropped, the surface is removed from the registry.
struct WgpuSurfaceHandleInner {
    surface_id: SurfaceId,
    registry: Arc<SurfaceRegistry<WgpuGpu>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    present_trigger: Arc<dyn Fn() + Send + Sync>,
    /// Optional direct handle to the winit window.  Having an `Arc` lets
    /// us call `request_redraw()` from another thread without touching the
    /// event bus.
    winit_window: Option<Arc<winit::window::Window>>,
    /// Device-level guard shared with the renderer. See
    /// `WgpuContext::gpu_submit_lock`'s doc comment for the full mechanism.
    gpu_submit_lock: Arc<parking_lot::RwLock<()>>,
    /// Repli quand la surface n'est plus dans le registre (après `remove`) :
    /// la taille vivante est celle du registre, pas une copie qui dérive.
    initial_size: (u32, u32),
    deferred_resize: Mutex<Option<(u32, u32)>>,
    format: wgpu::TextureFormat,
}

impl Drop for WgpuSurfaceHandleInner {
    fn drop(&mut self) {
        self.registry.remove(self.surface_id);
    }
}

/// A handle to a triple-buffered WGPU surface that perfectly emulates a Winit window.
///
/// External render threads use this to render continuously at their own pace:
/// 1. Get the back buffer with [`back_buffer_view()`](Self::back_buffer_view)
/// 2. Render to it
/// 3. Call [`present()`](Self::present) to swap buffers (non-blocking)
/// 4. Repeat immediately - no waiting required
///
/// The GPUI compositor runs independently, sampling the latest frame whenever it renders.
/// Frame drops and repeats are handled gracefully.
///
/// All rendering stays on the GPU — buffer swaps are atomic pointer swaps (no copy),
/// and the renderer samples textures directly in the shader.
///
/// This handle is `Clone + Send + Sync`.
#[derive(Clone)]
pub struct WgpuSurfaceHandle {
    inner: Arc<WgpuSurfaceHandleInner>,
}

impl WgpuSurfaceHandle {
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_id: SurfaceId,
        registry: Arc<SurfaceRegistry<WgpuGpu>>,
        present_trigger: Arc<dyn Fn() + Send + Sync>,
        winit_window: Option<Arc<winit::window::Window>>,
        gpu_submit_lock: Arc<parking_lot::RwLock<()>>,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            inner: Arc::new(WgpuSurfaceHandleInner {
                surface_id,
                registry,
                device,
                queue,
                present_trigger,
                winit_window,
                gpu_submit_lock,
                initial_size: (width, height),
                deferred_resize: Mutex::new(None),
                format,
            }),
        }
    }

    /// The wgpu `Device` for creating GPU resources and command encoders.
    pub fn device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    /// The wgpu `Queue` for submitting command buffers.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    /// Acquire permission to submit GPU work on this surface's device.
    ///
    /// **External render threads must hold this across encoding, `queue.submit()`
    /// and present.** The device and queue handed out by [`device()`](Self::device)
    /// and [`queue()`](Self::queue) are shared with the compositor, and
    /// `Surface::configure` (window resize) must observe an idle queue — wgpu
    /// aborts the process with `GpuWaitTimeout` if anything submits while it waits.
    ///
    /// This is a **read** guard: any number of surfaces may hold one simultaneously
    /// and none of them block each other, so each render thread keeps its own frame
    /// pacing. Only a resize takes the exclusive side, and only for as long as the
    /// swapchain takes to reconfigure.
    ///
    /// # Example
    /// ```no_run
    /// loop {
    ///     let guard = surface.submit_guard();
    ///     let (view, (w, h)) = surface.back_view_with_size()?;
    ///     // ... encode and submit ...
    ///     surface.present_synced_silent(submission_idx);
    ///     drop(guard);
    /// }
    /// ```
    pub fn submit_guard(&self) -> parking_lot::RwLockReadGuard<'_, ()> {
        self.inner.gpu_submit_lock.read()
    }

    /// Get a `TextureView` of the back buffer for use as a render target.
    /// Render into this, then call [`present()`](Self::present).
    pub fn back_buffer_view(&self) -> Option<wgpu::TextureView> {
        self.inner.registry.back_view(self.inner.surface_id)
    }

    /// Compat (voir src/compat.rs) : la `Texture` du back buffer,
    /// pour un rendu externe qui a besoin de la texture elle-même (taille,
    /// format, copies) et pas seulement d'une vue.
    pub fn back_buffer_texture(&self) -> Option<wgpu::Texture> {
        self.inner.registry.back_texture(self.inner.surface_id)
    }

    /// Atomically obtain the back buffer view _and_ its pixel dimensions.
    /// This avoids races where the surface is resized between separate calls
    /// to `back_buffer_view()` and `.size()`.
    pub fn back_view_with_size(&self) -> Option<(wgpu::TextureView, (u32, u32))> {
        self.inner
            .registry
            .lock_and_get_back_with_size(self.inner.surface_id)
    }

    /// Present the rendered frame with GPU synchronization (recommended).
    ///
    /// This atomically swaps the rendering and ready buffers, making your newly
    /// rendered frame available to the compositor, and triggers a window redraw request.
    ///
    /// The `submission_index` parameter (returned by `queue.submit()`) allows the
    /// compositor to poll the GPU and ensure rendering is complete before sampling
    /// the texture. This prevents visual artifacts from reading incomplete frames.
    ///
    /// **Returns immediately** - external threads can continue rendering the next
    /// frame without waiting for the compositor. This is the key difference from
    /// traditional blocking present models.
    ///
    /// # Example
    /// ```no_run
    /// // Render to the back buffer
    /// let view = surface.back_buffer_view()?;
    /// // ... encode commands ...
    /// let submission_idx = queue.submit([encoder.finish()]);
    /// drop(view);
    ///
    /// // Present with GPU sync
    /// surface.present_synced(submission_idx);
    /// ```
    pub fn present_synced(&self, _submission_index: wgpu::SubmissionIndex) {
        // Atomically swap rendering ↔ ready buffers with GPU sync
        self.inner
            .registry
            .swap_rendering_ready(self.inner.surface_id);

        // Track that this surface has new content to be composited
        self.inner
            .registry
            .set_redraw_pending(self.inner.surface_id);

        if let Some(winit) = &self.inner.winit_window {
            winit.request_redraw();
        } else {
            (self.inner.present_trigger)();
        }

        // Return immediately - no blocking
    }

    /// Swap the rendered frame into the ready slot with GPU synchronization, but
    /// without requesting a window redraw or marking the surface as needing one.
    ///
    /// Use this when the GPUI draw cycle already runs continuously (the element's
    /// view calls [`Window::request_animation_frame`]) and therefore drives
    /// compositing itself. The compositor promotes this frame to `display` on its
    /// next pass over the element.
    ///
    /// Prefer this over [`present_synced()`](Self::present_synced) for render
    /// threads embedded in a live GPUI view: `present_synced` drives repaints from
    /// the render thread, which forces a full window refresh per presented frame.
    pub fn present_synced_silent(&self, _submission_index: wgpu::SubmissionIndex) {
        self.inner
            .registry
            .swap_rendering_ready(self.inner.surface_id);
    }

    /// Réveille la boucle de fenêtre pour qu'elle compose la trame publiée par
    /// [`present_synced_silent()`](Self::present_synced_silent), sans poser le
    /// drapeau `redraw_pending` : celui-ci passe par le blit rapide (désactivé
    /// par défaut) et retombe en `force_render`, donc en dessin complet.
    pub fn request_window_redraw(&self) {
        match &self.inner.winit_window {
            Some(winit) => winit.request_redraw(),
            None => (self.inner.present_trigger)(),
        }
    }

    /// Present the rendered frame without GPU synchronization (deprecated).
    ///
    /// **DEPRECATED**: Use [`present_synced()`](Self::present_synced) instead for proper
    /// GPU synchronization. This method may cause visual artifacts if the compositor
    /// samples the texture before GPU rendering is complete.
    ///
    /// This method exists for backward compatibility only.
    #[deprecated(note = "Use present_synced() for proper GPU synchronization")]
    pub fn present(&self) {
        // Atomically swap rendering ↔ ready buffers (no GPU sync)
        self.inner
            .registry
            .swap_rendering_ready(self.inner.surface_id);

        // Track that this surface has new content to be composited
        self.inner
            .registry
            .set_redraw_pending(self.inner.surface_id);

        if let Some(winit) = &self.inner.winit_window {
            winit.request_redraw();
        } else {
            (self.inner.present_trigger)();
        }

        // Return immediately - no blocking
    }

    /// Silently swap the rendered buffer to the ready slot without triggering any
    /// redraw request or setting `redraw_pending`.
    ///
    /// Use this when the GPUI draw cycle (via `window.request_animation_frame()`) drives
    /// the compositor instead of the fast-blit path.  The compositor will pick up the
    /// latest ready buffer the next time it paints the `WgpuSurface` element.
    pub fn swap_buffers(&self) {
        self.inner
            .registry
            .swap_rendering_ready(self.inner.surface_id);
    }

    /// True if a frame published by [`present_synced`](Self::present_synced) or
    /// [`present_synced_silent`](Self::present_synced_silent) has not been
    /// composited yet.
    ///
    /// Use this to apply backpressure in an external render thread. The triple
    /// buffer keeps only one `ready` frame, so anything produced while this is
    /// true is discarded — the render is wasted work, but its GPU submission
    /// and per-frame allocations still cost memory. Without backpressure an
    /// uncapped render thread outruns the compositor without bound.
    ///
    /// Always pair this with a timeout rather than waiting indefinitely: the
    /// fast-blit presentation path does not advance the composited generation,
    /// so this can remain true while frames are in fact reaching the screen.
    pub fn has_unconsumed_frame(&self) -> bool {
        self.inner.registry.has_unconsumed_frame(self.inner.surface_id)
    }

    /// Current size in device pixels.
    pub fn size(&self) -> (u32, u32) {
        self.inner
            .registry
            .size(self.inner.surface_id)
            .unwrap_or(self.inner.initial_size)
    }

    /// The texture format used by this surface's buffers.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.inner.format
    }

    /// Returns true if a resize is pending, deferred, or currently in progress.
    pub fn is_resize_pending(&self) -> bool {
        if self.inner.registry.has_pending_resize(self.inner.surface_id) {
            return true;
        }
        self.inner.deferred_resize.lock().unwrap().is_some()
    }

    /// Defer a resize until a later time, without starting texture reallocation yet.
    pub fn defer_resize(&self, width: u32, height: u32) {
        let current_size = self.size();
        if current_size == (width, height) {
            return;
        }
        let mut deferred = self.inner.deferred_resize.lock().unwrap();
        if deferred.map_or(false, |pending_size| pending_size == (width, height)) {
            return;
        }
        *deferred = Some((width, height));
    }

    /// Take any deferred resize request, returning the target size.
    pub fn take_deferred_resize(&self) -> Option<(u32, u32)> {
        self.inner.deferred_resize.lock().unwrap().take()
    }

    /// Drop any resize retained by the registry so a now-stale size is never
    /// applied. The surface stays at its current committed size and the
    /// compositor keeps showing the stretched frame.
    pub fn cancel_pending_resize(&self) {
        self.inner
            .registry
            .cancel_pending_resize(self.inner.surface_id);
    }

    /// The `SurfaceId` for this handle (used internally by the element).
    pub fn id(&self) -> SurfaceId {
        self.inner.surface_id
    }

    /// Resize the surface's triple buffers to `width` x `height`.
    ///
    /// Applique la nouvelle taille tout de suite quand le compositeur ne lit
    /// pas les tampons, sinon la confie au registre, qui l'appliquera au point
    /// où le compositeur les lâche. **Aucun fil n'attend et rien n'est sondé.**
    ///
    /// La version précédente déportait le travail sur un fil qui bouclait sur
    /// `while !registry.resize(..) { sleep(8ms) }` : pendant un glissement de
    /// bordure la condition restait vraie tout le geste, la surface gardait ses
    /// anciennes dimensions, et `is_resize_pending()` restant vrai, le fil de
    /// rendu externe sortait sans dessiner. Un drapeau resté posé par un bug
    /// rendait en plus cette boucle infinie.
    pub fn request_resize(&self, width: u32, height: u32) {
        if width == 0 || height == 0 || self.size() == (width, height) {
            return;
        }
        self.inner
            .registry
            .resize(self.inner.surface_id, width, height);
    }
}

/// Create a `WgpuSurface` element from an existing handle.
pub fn wgpu_surface(handle: WgpuSurfaceHandle) -> WgpuSurface {
    WgpuSurface {
        handle,
        style: StyleRefinement::default(),
        on_resize: None,
        defer_resize_until_mouse_up: false,
    }
}

/// An element that displays content rendered externally via WGPU.
///
/// Acts as a drop-in replacement for a Winit window - external render threads
/// can render continuously at their own pace while GPUI composites around them.
///
/// On the WGPU platform, the renderer composites the surface's display buffer
/// texture directly (GPU → GPU, no copies). On other platforms this renders
/// as a fallback colored box.
pub struct WgpuSurface {
    handle: WgpuSurfaceHandle,
    style: StyleRefinement,
    on_resize: Option<Box<dyn Fn(u32, u32, &WgpuSurfaceHandle) + 'static>>,
    defer_resize_until_mouse_up: bool,
}

impl WgpuSurface {
    /// Register a callback invoked when the element's layout bounds change.
    /// The surface textures are automatically resized; use this to recreate
    /// any external resources that depend on the size.
    pub fn on_resize(
        mut self,
        callback: impl Fn(u32, u32, &WgpuSurfaceHandle) + 'static,
    ) -> Self {
        self.on_resize = Some(Box::new(callback));
        self
    }

    /// Enable deferred resize until left mouse release.
    pub fn defer_resize_until_mouse_up(mut self, enabled: bool) -> Self {
        self.defer_resize_until_mouse_up = enabled;
        self
    }
}

impl Element for WgpuSurface {
    type RequestLayoutState = Style;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.refine(&self.style);
        let layout_id = window.request_layout(style.clone(), [], cx);
        (layout_id, style)
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
        // Compute pixel size accounting for scale factor
        let scale = window.scale_factor();
        let pixel_w = (bounds.size.width.0 * scale).round() as u32;
        let pixel_h = (bounds.size.height.0 * scale).round() as u32;

        let (cur_w, cur_h) = self.handle.size();
        let left_pressed = window.pressed_mouse_button() == Some(MouseButton::Left);
        let window_resizing = window.is_window_resizing();

        if pixel_w != cur_w || pixel_h != cur_h {
            if self.defer_resize_until_mouse_up && (left_pressed || window_resizing) {
                // Une cible retenue par le registre deviendrait obsolète : on
                // l'oublie avant de différer. Le compositeur garde la trame
                // étirée en attendant.
                self.handle.cancel_pending_resize();
                self.handle.defer_resize(pixel_w, pixel_h);
            } else {
                self.handle.request_resize(pixel_w, pixel_h);
                if let Some(cb) = &self.on_resize {
                    cb(pixel_w, pixel_h, &self.handle);
                }
            }
        }

        if self.defer_resize_until_mouse_up && !left_pressed && !window_resizing {
            if let Some((pending_w, pending_h)) = self.handle.take_deferred_resize() {
                if (pending_w, pending_h) != (cur_w, cur_h) {
                    self.handle.request_resize(pending_w, pending_h);
                    if let Some(cb) = &self.on_resize {
                        cb(pending_w, pending_h, &self.handle);
                    }
                }
            }
        }
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        style: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        style.paint(bounds, window, cx, |window, _cx| {
            window.paint_wgpu_surface(bounds, Corners::default(), self.handle.id());
        });
    }
}

impl IntoElement for WgpuSurface {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Styled for WgpuSurface {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

// ---------------------------------------------------------------------------
// GPUI-3D : interop native. Un moteur 3D Metal, Vulkan ou D3D12 rend dans les tampons de la
// surface avec SA propre API, sur le device de l'UI (zéro copie). wgpu n'apparaît
// qu'ici, pour lire les poignées ; le moteur n'en dépend pas.
// ---------------------------------------------------------------------------

/// Poignées natives du device de l'UI. Valides tant que l'application vit.
#[derive(Clone, Copy, Debug)]
pub enum NativeDevice {
    /// `id<MTLDevice>` et `id<MTLCommandQueue>` du compositeur (même queue ⇒ ordre garanti).
    Metal {
        /// `id<MTLDevice>`.
        device: *mut std::ffi::c_void,
        /// `id<MTLCommandQueue>`.
        queue: *mut std::ffi::c_void,
    },
    /// Handles bruts (`vk::Handle::as_raw`). Toute soumission sur `queue` se fait sous
    /// [`WgpuSurfaceHandle::native_queue_lock`].
    Vulkan {
        /// `VkInstance`.
        instance: u64,
        /// `VkPhysicalDevice`.
        physical_device: u64,
        /// `VkDevice`.
        device: u64,
        /// `VkQueue`.
        queue: u64,
        /// Famille de `queue`.
        queue_family_index: u32,
    },
    /// `ID3D12Device*` et `ID3D12CommandQueue*` du compositeur (même queue ⇒ ordre
    /// garanti ; une queue D3D12 est thread-safe, aucun verrou requis). Pointeurs
    /// empruntés : `AddRef` pour les garder.
    Dx12 {
        /// `ID3D12Device*`.
        device: *mut std::ffi::c_void,
        /// `ID3D12CommandQueue*`.
        queue: *mut std::ffi::c_void,
    },
}

// SAFETY: poignées opaques ; `id<MTLDevice>`/`id<MTLCommandQueue>` et les objets D3D12
// sont thread-safe, `VkQueue` est protégée par `native_queue_lock`.
unsafe impl Send for NativeDevice {}
unsafe impl Sync for NativeDevice {}

/// Tampon arrière à rendre. Le garder en vie jusqu'à la fin GPU de la trame : il retient
/// la texture, que wgpu ne sait pas utilisée par des commandes natives.
pub struct NativeBackBuffer {
    /// `id<MTLTexture>` (Metal) ou `(VkImage, VkImageView)` (Vulkan).
    pub texture: NativeTexture,
    /// Taille en pixels.
    pub size: (u32, u32),
    _texture: wgpu::Texture,
    _view: wgpu::TextureView,
}

/// Voir [`NativeBackBuffer`]. Contrat Vulkan : l'image arrive et doit repartir en
/// `SHADER_READ_ONLY_OPTIMAL` (l'état `RESOURCE` que wgpu lui connaît). Contrat D3D12 :
/// même chose en `PIXEL_SHADER_RESOURCE | NON_PIXEL_SHADER_RESOURCE`.
#[derive(Clone, Copy, Debug)]
pub enum NativeTexture {
    /// `id<MTLTexture>`.
    Metal(*mut std::ffi::c_void),
    /// `VkImage` et `VkImageView` bruts.
    Vulkan {
        /// `VkImage`.
        image: u64,
        /// `VkImageView`.
        view: u64,
    },
    /// `ID3D12Resource*` emprunté.
    Dx12(*mut std::ffi::c_void),
}

unsafe impl Send for NativeBackBuffer {}

impl WgpuSurfaceHandle {
    /// Poignées natives du device de l'UI (`None` : backend GL ou WebGPU).
    pub fn native_device(&self) -> Option<NativeDevice> {
        #[cfg(target_vendor = "apple")]
        if let Some(device) = unsafe { self.inner.device.as_hal::<wgpu::hal::api::Metal>() } {
            let queue = unsafe { self.inner.queue.as_hal::<wgpu::hal::api::Metal>() }?;
            return Some(NativeDevice::Metal {
                device: &**device.raw_device() as *const _ as *mut std::ffi::c_void,
                queue: queue.as_raw() as *const _ as *mut std::ffi::c_void,
            });
        }
        #[cfg(windows)]
        if let Some(device) = unsafe { self.inner.device.as_hal::<wgpu::hal::api::Dx12>() } {
            let queue = unsafe { self.inner.queue.as_hal::<wgpu::hal::api::Dx12>() }?;
            return Some(NativeDevice::Dx12 {
                device: windows_core::Interface::as_raw(device.raw_device()),
                queue: windows_core::Interface::as_raw(queue.as_raw()),
            });
        }
        #[cfg(not(target_family = "wasm"))]
        if let Some(device) = unsafe { self.inner.device.as_hal::<wgpu::hal::api::Vulkan>() } {
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

    /// Le tampon arrière en poignées natives, à publier ensuite par
    /// [`swap_buffers`](Self::swap_buffers) une fois les commandes soumises.
    pub fn native_back_buffer(&self) -> Option<NativeBackBuffer> {
        let (view, size) = self.back_view_with_size()?;
        let texture = self.back_buffer_texture()?;
        #[cfg(target_vendor = "apple")]
        if let Some(raw) = unsafe { texture.as_hal::<wgpu::hal::api::Metal>() } {
            let texture_ptr = raw.raw_handle() as *const _ as *mut std::ffi::c_void;
            drop(raw);
            return Some(NativeBackBuffer {
                texture: NativeTexture::Metal(texture_ptr),
                size,
                _texture: texture,
                _view: view,
            });
        }
        #[cfg(windows)]
        if let Some(raw) = unsafe { texture.as_hal::<wgpu::hal::api::Dx12>() } {
            let resource = windows_core::Interface::as_raw(unsafe { raw.raw_resource() });
            drop(raw);
            return Some(NativeBackBuffer {
                texture: NativeTexture::Dx12(resource),
                size,
                _texture: texture,
                _view: view,
            });
        }
        #[cfg(not(target_family = "wasm"))]
        {
            use ash::vk::Handle as _;
            let image = unsafe { texture.as_hal::<wgpu::hal::api::Vulkan>()?.raw_handle() }.as_raw();
            let view_raw = unsafe { view.as_hal::<wgpu::hal::api::Vulkan>()?.raw_handle() }.as_raw();
            return Some(NativeBackBuffer {
                texture: NativeTexture::Vulkan { image, view: view_raw },
                size,
                _texture: texture,
                _view: view,
            });
        }
        #[allow(unreachable_code)]
        None
    }

    /// Verrou de la queue partagée, à tenir autour de tout `vkQueueSubmit` natif
    /// (le compositeur le prend autour de ses `submit`/`present`). Inutile en Metal et D3D12.
    pub fn native_queue_lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.inner.registry.queue_lock()
    }
}
