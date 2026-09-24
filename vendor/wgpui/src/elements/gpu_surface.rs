use std::sync::{Arc, Mutex};

use refineable::Refineable as _;

use crate::{
    App, Bounds, Corners, Element, ElementId, GlobalElementId, InspectorElementId, IntoElement, LayoutId,
    MouseButton, Pixels, Style, StyleRefinement, Styled, Window,
    platform::cross::surface_registry::{SurfaceId, SurfaceStore},
};
#[cfg(feature = "wgpu")]
use crate::platform::cross::{hal::wgpu::WgpuGpu, surface_registry::SurfaceRegistry};

/// Format des tampons d'une surface 3D, commun à tous les backends de rendu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceFormat {
    /// Metal `BGRA8Unorm_sRGB`, Vulkan `B8G8R8A8_SRGB`, D3D12 `B8G8R8A8_UNORM_SRGB`.
    Bgra8UnormSrgb,
    /// Metal `RGBA8Unorm_sRGB`, Vulkan `R8G8B8A8_SRGB`, D3D12 `R8G8B8A8_UNORM_SRGB`.
    Rgba8UnormSrgb,
}

/// Inner state shared across clones of `SurfaceHandle`.
/// When the last clone is dropped, the surface is removed from the registry.
struct SurfaceHandleInner {
    surface_id: SurfaceId,
    registry: Arc<dyn SurfaceStore>,
    present_trigger: Arc<dyn Fn() + Send + Sync>,
    /// Optional direct handle to the winit window.  Having an `Arc` lets
    /// us call `request_redraw()` from another thread without touching the
    /// event bus.
    winit_window: Option<Arc<winit::window::Window>>,
    /// Device-level guard shared with the renderer. See
    /// `RenderContext::gpu_submit_lock`'s doc comment for the full mechanism.
    gpu_submit_lock: Arc<parking_lot::RwLock<()>>,
    /// Repli quand la surface n'est plus dans le registre (après `remove`) :
    /// la taille vivante est celle du registre, pas une copie qui dérive.
    initial_size: (u32, u32),
    deferred_resize: Mutex<Option<(u32, u32)>>,
    /// Présent quand l'UI tourne sur wgpu (voir [`SurfaceHandle::as_wgpu`]).
    #[cfg(feature = "wgpu")]
    wgpu: Option<WgpuParts>,
}

#[cfg(feature = "wgpu")]
pub(crate) struct WgpuParts {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) registry: Arc<SurfaceRegistry<WgpuGpu>>,
    pub(crate) format: wgpu::TextureFormat,
}

impl Drop for SurfaceHandleInner {
    fn drop(&mut self) {
        self.registry.remove(self.surface_id);
    }
}

/// A handle to a triple-buffered GPU surface that perfectly emulates a Winit window,
/// on whichever backend renders the UI (wgpu or a native API).
///
/// External render threads use this to render continuously at their own pace, with
/// their own graphics API on the UI's device:
/// 1. Get the back buffer with [`native_back_buffer()`](Self::native_back_buffer)
/// 2. Render to it, submitting under [`native_queue_lock()`](Self::native_queue_lock)
/// 3. Call [`swap_buffers()`](Self::swap_buffers) (non-blocking), then
///    [`request_window_redraw()`](Self::request_window_redraw)
/// 4. Repeat immediately - no waiting required
///
/// The GPUI compositor runs independently, sampling the latest frame whenever it renders.
/// Frame drops and repeats are handled gracefully. All rendering stays on the GPU —
/// buffer swaps are atomic index swaps (no copy).
///
/// This handle is `Clone + Send + Sync`.
#[derive(Clone)]
pub struct SurfaceHandle {
    inner: Arc<SurfaceHandleInner>,
}

impl AsRef<SurfaceHandle> for SurfaceHandle {
    fn as_ref(&self) -> &SurfaceHandle {
        self
    }
}

impl SurfaceHandle {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        surface_id: SurfaceId,
        registry: Arc<dyn SurfaceStore>,
        present_trigger: Arc<dyn Fn() + Send + Sync>,
        winit_window: Option<Arc<winit::window::Window>>,
        gpu_submit_lock: Arc<parking_lot::RwLock<()>>,
        width: u32,
        height: u32,
        #[cfg(feature = "wgpu")] wgpu: Option<WgpuParts>,
    ) -> Self {
        Self {
            inner: Arc::new(SurfaceHandleInner {
                surface_id,
                registry,
                present_trigger,
                winit_window,
                gpu_submit_lock,
                initial_size: (width, height),
                deferred_resize: Mutex::new(None),
                #[cfg(feature = "wgpu")]
                wgpu,
            }),
        }
    }

    /// La même surface vue par l'API wgpu, si l'UI tourne sur wgpu.
    #[cfg(feature = "wgpu")]
    pub fn as_wgpu(&self) -> Option<WgpuSurfaceHandle> {
        WgpuSurfaceHandle::new(self.clone())
    }

    /// Acquire permission to submit GPU work on this surface's device.
    ///
    /// **External render threads must hold this across encoding, submission
    /// and present.** The device and queue are shared with the compositor, and
    /// swapchain reconfiguration (window resize) must observe an idle queue.
    ///
    /// This is a **read** guard: any number of surfaces may hold one simultaneously
    /// and none of them block each other, so each render thread keeps its own frame
    /// pacing. Only a resize takes the exclusive side, and only for as long as the
    /// swapchain takes to reconfigure.
    pub fn submit_guard(&self) -> parking_lot::RwLockReadGuard<'_, ()> {
        self.inner.gpu_submit_lock.read()
    }

    /// Publish the rendered frame: swap the rendering buffer into the ready slot
    /// without requesting a window redraw. The compositor picks it up on its next
    /// pass over the element; see [`request_window_redraw()`](Self::request_window_redraw).
    pub fn swap_buffers(&self) {
        self.inner.registry.swap_rendering_ready(self.inner.surface_id);
    }

    /// Réveille la boucle de fenêtre pour qu'elle compose la trame publiée par
    /// [`swap_buffers()`](Self::swap_buffers), sans poser le drapeau `redraw_pending` :
    /// celui-ci passe par le blit rapide (désactivé par défaut) et retombe en
    /// `force_render`, donc en dessin complet.
    pub fn request_window_redraw(&self) {
        match &self.inner.winit_window {
            Some(winit) => winit.request_redraw(),
            None => (self.inner.present_trigger)(),
        }
    }

    /// Swap, mark the surface as needing composition and request a redraw.
    #[cfg(feature = "wgpu")]
    fn present_with_redraw(&self) {
        self.swap_buffers();
        self.inner.registry.set_redraw_pending(self.inner.surface_id);
        self.request_window_redraw();
    }

    /// True if a published frame has not been composited yet.
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
        self.inner.registry.size(self.inner.surface_id).unwrap_or(self.inner.initial_size)
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
        self.inner.registry.cancel_pending_resize(self.inner.surface_id);
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
        self.inner.registry.resize(self.inner.surface_id, width, height);
    }

    /// Poignées natives du device de l'UI (`None` : backend GL ou WebGPU).
    pub fn native_device(&self) -> Option<NativeDevice> {
        self.inner.registry.native_device()
    }

    /// Le tampon arrière en poignées natives, à publier ensuite par
    /// [`swap_buffers`](Self::swap_buffers) une fois les commandes soumises.
    pub fn native_back_buffer(&self) -> Option<NativeBackBuffer> {
        self.inner.registry.native_back_buffer(self.inner.surface_id)
    }

    /// Verrou de la queue partagée, à tenir autour de tout `vkQueueSubmit` natif
    /// (le compositeur le prend autour de ses propres accès à la queue). Inutile en
    /// Metal et D3D12.
    pub fn native_queue_lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.inner.registry.queue_lock()
    }
}

/// [`SurfaceHandle`] d'une UI qui tourne sur wgpu, avec l'accès wgpu à son device
/// et à ses tampons. Obtenu par [`Window::create_wgpu_surface`] ou
/// [`SurfaceHandle::as_wgpu`].
#[cfg(feature = "wgpu")]
#[derive(Clone)]
pub struct WgpuSurfaceHandle(SurfaceHandle);

#[cfg(feature = "wgpu")]
impl std::ops::Deref for WgpuSurfaceHandle {
    type Target = SurfaceHandle;

    fn deref(&self) -> &SurfaceHandle {
        &self.0
    }
}

#[cfg(feature = "wgpu")]
impl AsRef<SurfaceHandle> for WgpuSurfaceHandle {
    fn as_ref(&self) -> &SurfaceHandle {
        &self.0
    }
}

#[cfg(feature = "wgpu")]
impl From<WgpuSurfaceHandle> for SurfaceHandle {
    fn from(handle: WgpuSurfaceHandle) -> Self {
        handle.0
    }
}

#[cfg(feature = "wgpu")]
impl WgpuSurfaceHandle {
    pub(crate) fn new(handle: SurfaceHandle) -> Option<Self> {
        handle.inner.wgpu.is_some().then_some(Self(handle))
    }

    fn parts(&self) -> &WgpuParts {
        let Some(parts) = &self.0.inner.wgpu else {
            unreachable!("WgpuSurfaceHandle n'est construit que sur une surface wgpu")
        };
        parts
    }

    /// The wgpu `Device` for creating GPU resources and command encoders.
    pub fn device(&self) -> &wgpu::Device {
        &self.parts().device
    }

    /// The wgpu `Queue` for submitting command buffers.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.parts().queue
    }

    /// Get a `TextureView` of the back buffer for use as a render target.
    /// Render into this, then call [`present_synced()`](Self::present_synced).
    pub fn back_buffer_view(&self) -> Option<wgpu::TextureView> {
        self.parts().registry.back_view(self.id())
    }

    /// Compat (voir src/compat.rs) : la `Texture` du back buffer,
    /// pour un rendu externe qui a besoin de la texture elle-même (taille,
    /// format, copies) et pas seulement d'une vue.
    pub fn back_buffer_texture(&self) -> Option<wgpu::Texture> {
        self.parts().registry.back_texture(self.id())
    }

    /// Atomically obtain the back buffer view _and_ its pixel dimensions.
    /// This avoids races where the surface is resized between separate calls
    /// to `back_buffer_view()` and `.size()`.
    pub fn back_view_with_size(&self) -> Option<(wgpu::TextureView, (u32, u32))> {
        self.parts().registry.lock_and_get_back_with_size(self.id())
    }

    /// The texture format used by this surface's buffers.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.parts().format
    }

    /// Present the rendered frame and trigger a window redraw.
    ///
    /// This atomically swaps the rendering and ready buffers, making your newly
    /// rendered frame available to the compositor. wgpu orders the compositor's
    /// sampling after `submission_index` on the shared queue.
    ///
    /// **Returns immediately** - external threads can continue rendering the next
    /// frame without waiting for the compositor.
    pub fn present_synced(&self, _submission_index: wgpu::SubmissionIndex) {
        self.present_with_redraw();
    }

    /// Swap the rendered frame into the ready slot without requesting a window
    /// redraw or marking the surface as needing one.
    ///
    /// Use this when the GPUI draw cycle already runs continuously (the element's
    /// view calls [`Window::request_animation_frame`]) or when the render thread
    /// calls [`request_window_redraw()`](SurfaceHandle::request_window_redraw)
    /// itself: `present_synced` drives repaints from the render thread, which
    /// forces a full window refresh per presented frame.
    pub fn present_synced_silent(&self, _submission_index: wgpu::SubmissionIndex) {
        self.swap_buffers();
    }

    /// Present the rendered frame and trigger a window redraw (deprecated).
    #[deprecated(note = "Use present_synced() for proper GPU synchronization")]
    pub fn present(&self) {
        self.present_with_redraw();
    }
}

/// Create a [`GpuSurface`] element from an existing handle.
pub fn gpu_surface<H: AsRef<SurfaceHandle> + 'static>(handle: H) -> GpuSurface<H> {
    GpuSurface { handle, style: StyleRefinement::default(), on_resize: None, defer_resize_until_mouse_up: false }
}

/// Create a `WgpuSurface` element from an existing handle.
#[cfg(feature = "wgpu")]
pub fn wgpu_surface(handle: WgpuSurfaceHandle) -> WgpuSurface {
    gpu_surface(handle)
}

/// Élément `GpuSurface` d'une surface wgpu.
#[cfg(feature = "wgpu")]
pub type WgpuSurface = GpuSurface<WgpuSurfaceHandle>;

/// An element that displays content rendered externally into a [`SurfaceHandle`].
///
/// Acts as a drop-in replacement for a Winit window - external render threads
/// can render continuously at their own pace while GPUI composites around them.
/// The renderer composites the surface's display buffer texture directly (GPU → GPU,
/// no copies).
pub struct GpuSurface<H: AsRef<SurfaceHandle> + 'static = SurfaceHandle> {
    handle: H,
    style: StyleRefinement,
    on_resize: Option<Box<dyn Fn(u32, u32, &H) + 'static>>,
    defer_resize_until_mouse_up: bool,
}

impl<H: AsRef<SurfaceHandle> + 'static> GpuSurface<H> {
    /// Register a callback invoked when the element's layout bounds change.
    /// The surface textures are automatically resized; use this to recreate
    /// any external resources that depend on the size.
    pub fn on_resize(mut self, callback: impl Fn(u32, u32, &H) + 'static) -> Self {
        self.on_resize = Some(Box::new(callback));
        self
    }

    /// Enable deferred resize until left mouse release.
    pub fn defer_resize_until_mouse_up(mut self, enabled: bool) -> Self {
        self.defer_resize_until_mouse_up = enabled;
        self
    }
}

impl<H: AsRef<SurfaceHandle> + 'static> Element for GpuSurface<H> {
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
        let surface = self.handle.as_ref();
        // Compute pixel size accounting for scale factor
        let scale = window.scale_factor();
        let pixel_w = (bounds.size.width.0 * scale).round() as u32;
        let pixel_h = (bounds.size.height.0 * scale).round() as u32;

        let (cur_w, cur_h) = surface.size();
        let left_pressed = window.pressed_mouse_button() == Some(MouseButton::Left);
        let window_resizing = window.is_window_resizing();

        if pixel_w != cur_w || pixel_h != cur_h {
            if self.defer_resize_until_mouse_up && (left_pressed || window_resizing) {
                // Une cible retenue par le registre deviendrait obsolète : on
                // l'oublie avant de différer. Le compositeur garde la trame
                // étirée en attendant.
                surface.cancel_pending_resize();
                surface.defer_resize(pixel_w, pixel_h);
            } else {
                surface.request_resize(pixel_w, pixel_h);
                if let Some(cb) = &self.on_resize {
                    cb(pixel_w, pixel_h, &self.handle);
                }
            }
        }

        if self.defer_resize_until_mouse_up && !left_pressed && !window_resizing {
            if let Some((pending_w, pending_h)) = surface.take_deferred_resize() {
                if (pending_w, pending_h) != (cur_w, cur_h) {
                    surface.request_resize(pending_w, pending_h);
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
        let surface_id = self.handle.as_ref().id();
        style.paint(bounds, window, cx, |window, _cx| {
            window.paint_wgpu_surface(bounds, Corners::default(), surface_id);
        });
    }
}

impl<H: AsRef<SurfaceHandle> + 'static> IntoElement for GpuSurface<H> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl<H: AsRef<SurfaceHandle> + 'static> Styled for GpuSurface<H> {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

// ---------------------------------------------------------------------------
// GPUI-3D : interop native. Un moteur 3D Metal, Vulkan ou D3D12 rend dans les tampons de la
// surface avec SA propre API, sur le device de l'UI (zéro copie). Le moteur ne dépend
// pas du backend de l'UI (wgpu ou natif).
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
    /// [`SurfaceHandle::native_queue_lock`].
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
/// la texture, que le compositeur ne sait pas utilisée par des commandes natives.
pub struct NativeBackBuffer {
    /// `id<MTLTexture>` (Metal), `(VkImage, VkImageView)` (Vulkan) ou `ID3D12Resource*`.
    pub texture: NativeTexture,
    /// Taille en pixels.
    pub size: (u32, u32),
    _retained: Box<dyn std::any::Any + Send>,
}

impl NativeBackBuffer {
    pub(crate) fn new(texture: NativeTexture, size: (u32, u32), retained: Box<dyn std::any::Any + Send>) -> Self {
        Self { texture, size, _retained: retained }
    }
}

/// Voir [`NativeBackBuffer`]. Contrat Vulkan : l'image arrive et doit repartir en
/// `SHADER_READ_ONLY_OPTIMAL`. Contrat D3D12 : même chose en
/// `PIXEL_SHADER_RESOURCE | NON_PIXEL_SHADER_RESOURCE`.
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

// SAFETY: la texture native est retenue par `_retained` ; les poignées sont opaques.
unsafe impl Send for NativeBackBuffer {}
