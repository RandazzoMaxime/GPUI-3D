//! Chrome GPUI + fil de rendu 3D, partagés par GPUI-WGPU et GPUI-METAL.
//! Recette du flight planner (`helix_gpui_shell::terra_render`) : le moteur rend sur
//! son propre fil dans une `WgpuSurface` triple-buffer du device de l'UI, publie par
//! `present_synced_silent` + `request_window_redraw` ⇒ la fenêtre recompose la scène
//! en cache, le chrome n'est jamais redessiné pour une trame 3D.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use glam::{Mat4, Vec3};
use gpui::{
    App, Application, Bounds, Context, FontWeight, MouseButton, MouseDownEvent, MouseMoveEvent,
    Point, Render, ScrollWheelEvent, TitlebarOptions, WgpuSurfaceHandle, Window, WindowBounds,
    WindowOptions, div, prelude::*, px, rgb, size, wgpu_surface,
};

pub use gpui::{NativeBackBuffer, NativeDevice, NativeTexture, WgpuSurfaceHandle as Surface};
pub use wgpu::Backends;

/// sRGB comme la surface Terra du flight planner (Metal : `BGRA8Unorm_sRGB`, Vulkan : `B8G8R8A8_SRGB`).
pub const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
pub const CLEAR_COLOR: [f64; 4] = [0.96, 0.96, 0.97, 1.0];

/// Moteur 3D branché sur la surface, appelé sur le fil de rendu (`submit_guard` tenu).
/// Il publie lui-même sa trame : `present_synced_silent` (wgpu) ou `swap_buffers` (natif).
pub trait Renderer: 'static {
    fn new(surface: &WgpuSurfaceHandle) -> Self;
    /// `true` si une trame a été publiée.
    fn render(&mut self, surface: &WgpuSurfaceHandle, scene: &Scene) -> bool;
}

#[derive(Clone, Copy)]
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { yaw: 0.6, pitch: 0.4, distance: 3.0 }
    }
}

impl OrbitCamera {
    fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.01;
        self.pitch = (self.pitch + dy * 0.01).clamp(-1.5, 1.5);
    }

    fn zoom(&mut self, dy: f32) {
        self.distance = (self.distance * (-dy * 0.002).exp()).clamp(1.2, 20.0);
    }

    fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, sp, cy * cp) * self.distance
    }
}

pub struct Scene {
    pub camera: OrbitCamera,
    pub time: f32,
}

impl Scene {
    /// Profondeur 0..1 : même convention clip pour wgpu et Metal.
    pub fn mvp(&self, width: u32, height: u32) -> [[f32; 4]; 4] {
        let aspect = width as f32 / height.max(1) as f32;
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 100.0);
        let view = Mat4::look_at_rh(self.camera.eye(), Vec3::ZERO, Vec3::Y);
        let model = Mat4::from_rotation_y(self.time * 0.8) * Mat4::from_rotation_x(self.time * 0.5);
        (proj * view * model).to_cols_array_2d()
    }
}

/// 24 sommets (4 par face) : position xyz + couleur rgb.
#[rustfmt::skip]
pub const CUBE_VERTICES: [[f32; 6]; 24] = [
    [-0.5,-0.5, 0.5, 0.95,0.30,0.30], [ 0.5,-0.5, 0.5, 0.95,0.30,0.30], [ 0.5, 0.5, 0.5, 1.00,0.55,0.55], [-0.5, 0.5, 0.5, 1.00,0.55,0.55],
    [ 0.5,-0.5,-0.5, 0.25,0.75,0.35], [-0.5,-0.5,-0.5, 0.25,0.75,0.35], [-0.5, 0.5,-0.5, 0.50,0.95,0.55], [ 0.5, 0.5,-0.5, 0.50,0.95,0.55],
    [-0.5,-0.5,-0.5, 0.25,0.40,0.95], [-0.5,-0.5, 0.5, 0.25,0.40,0.95], [-0.5, 0.5, 0.5, 0.50,0.65,1.00], [-0.5, 0.5,-0.5, 0.50,0.65,1.00],
    [ 0.5,-0.5, 0.5, 0.95,0.80,0.20], [ 0.5,-0.5,-0.5, 0.95,0.80,0.20], [ 0.5, 0.5,-0.5, 1.00,0.92,0.50], [ 0.5, 0.5, 0.5, 1.00,0.92,0.50],
    [-0.5, 0.5, 0.5, 0.20,0.85,0.90], [ 0.5, 0.5, 0.5, 0.20,0.85,0.90], [ 0.5, 0.5,-0.5, 0.50,1.00,1.00], [-0.5, 0.5,-0.5, 0.50,1.00,1.00],
    [-0.5,-0.5,-0.5, 0.85,0.30,0.90], [ 0.5,-0.5,-0.5, 0.85,0.30,0.90], [ 0.5,-0.5, 0.5, 1.00,0.55,1.00], [-0.5,-0.5, 0.5, 1.00,0.55,1.00],
];

#[rustfmt::skip]
pub const CUBE_INDICES: [u16; 36] = [
     0, 1, 2,  0, 2, 3,   4, 5, 6,  4, 6, 7,   8, 9,10,  8,10,11,
    12,13,14, 12,14,15,  16,17,18, 16,18,19,  20,21,22, 20,22,23,
];

/// Ouvre la fenêtre GPUI-3D et démarre le moteur `R` sur son fil. `backends` restreint
/// l'adaptateur du device GPUI — donc celui du moteur, c'est le même device.
pub fn run<R: Renderer>(label: &'static str, backends: Backends) {
    let adapter_selector: Option<gpui::AdapterSelector> = (backends != Backends::all()).then(|| {
        Arc::new(move |infos: &[wgpu::AdapterInfo]| infos.iter().position(|i| backends.contains(i.backend.into())))
            as _
    });
    Application::with_wgpu_options(gpui::WgpuOptions { adapter_selector, ..Default::default() }).run(move |cx: &mut App| {
        // Police embarquée : même rendu sur toutes les plateformes (« SF Pro » n'est pas une famille nommée sur macOS).
        cx.text_system()
            .add_fonts(vec![
                include_bytes!("../../vendor/wgpui/assets/fonts/ibm-plex-sans/IBMPlexSans-Regular.ttf").as_slice().into(),
                include_bytes!("../../vendor/wgpui/assets/fonts/ibm-plex-sans/IBMPlexSans-Bold.ttf").as_slice().into(),
            ])
            .expect("polices");
        let bounds = Bounds::centered(None, size(px(1280.), px(800.)), cx);
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            titlebar: Some(TitlebarOptions { title: Some("GPUI-3D".into()), ..Default::default() }),
            ..Default::default()
        };
        cx.open_window(options, |window, cx| {
            let scale = window.scale_factor();
            let vp = window.viewport_size();
            let (w, h) = ((f32::from(vp.width) * scale) as u32, (f32::from(vp.height) * scale) as u32);
            let surface = window
                .create_wgpu_surface(w.max(1), h.max(1), SURFACE_FORMAT)
                .expect("WgpuSurface indisponible sur cette plateforme");
            let hz = window
                .display(cx)
                .and_then(|d| d.refresh_rate_millihertz())
                .map_or(60.0, |mhz| f64::from(mhz) / 1000.0);
            let camera = Arc::new(Mutex::new(OrbitCamera::default()));
            let frames = Arc::new(AtomicU64::new(0));
            spawn_render_thread::<R>(surface.clone(), camera.clone(), frames.clone(), hz);
            let fps = cx.new(|cx| FpsLabel::new(frames, cx));
            cx.new(|_| Shell { backend: label, surface, camera, drag_from: Arc::new(Mutex::new(None)), fps })
        })
        .expect("ouverture fenêtre");
        cx.on_window_closed(|cx, _| cx.quit()).detach();
        cx.activate(true);
    });
}

fn spawn_render_thread<R: Renderer>(
    surface: WgpuSurfaceHandle,
    camera: Arc<Mutex<OrbitCamera>>,
    frames: Arc<AtomicU64>,
    hz: f64,
) {
    let period = Duration::from_secs_f64(1.0 / hz.max(1.0));
    eprintln!("[render-3d] cadence cible {hz:.0} Hz");
    std::thread::Builder::new()
        .name("render-3d".into())
        .spawn(move || {
            let mut renderer = R::new(&surface);
            let start = Instant::now();
            let mut deadline = start;
            loop {
                let scene = Scene { camera: *camera.lock().unwrap(), time: start.elapsed().as_secs_f32() };
                let guard = surface.submit_guard();
                if renderer.render(&surface, &scene) {
                    frames.fetch_add(1, Ordering::Relaxed);
                }
                drop(guard);
                // Jamais `present_synced` : il force un dessin complet du chrome par trame.
                surface.request_window_redraw();
                // Échéances absolues : un `sleep(reste)` par trame dérive et perd des trames.
                deadline = (deadline + period).max(Instant::now());
                std::thread::sleep(deadline - Instant::now());
            }
        })
        .expect("fil render-3d");
}

struct Shell {
    backend: &'static str,
    surface: WgpuSurfaceHandle,
    camera: Arc<Mutex<OrbitCamera>>,
    drag_from: Arc<Mutex<Option<Point<gpui::Pixels>>>>,
    fps: gpui::Entity<FpsLabel>,
}

impl Render for Shell {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        // Les gestes écrivent la caméra partagée sans `cx.notify()` : le fil de rendu la relit.
        let (drag_down, drag_move, drag_up) = (self.drag_from.clone(), self.drag_from.clone(), self.drag_from.clone());
        let (orbit, zoom) = (self.camera.clone(), self.camera.clone());
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(0xffffff))
            .font_family("IBM Plex Sans")
            .text_color(rgb(0x1a1a1f))
            .on_mouse_move(move |ev: &MouseMoveEvent, _, _| {
                let mut from = drag_move.lock().unwrap();
                if let (Some(prev), Some(MouseButton::Left)) = (*from, ev.pressed_button) {
                    let d = ev.position - prev;
                    orbit.lock().unwrap().orbit(f32::from(d.x), f32::from(d.y));
                    *from = Some(ev.position);
                } else if ev.pressed_button.is_none() {
                    *from = None;
                }
            })
            .on_mouse_up(MouseButton::Left, move |_, _, _| *drag_up.lock().unwrap() = None)
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .px_6()
                    .py_4()
                    .border_b_1()
                    .border_color(rgb(0xe6e6ea))
                    .child(div().text_2xl().font_weight(FontWeight::BOLD).child("GPUI-3D"))
                    .child(
                        div()
                            .flex()
                            .gap_4()
                            .text_sm()
                            .text_color(rgb(0x6b6b76))
                            .child(format!("moteur : {}", self.backend))
                            .child(self.fps.clone()),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .p_6()
                    .child(
                        div()
                            .id("viewport")
                            .size_full()
                            .relative()
                            .border_1()
                            .border_color(rgb(0xe6e6ea))
                            .on_mouse_down(MouseButton::Left, move |ev: &MouseDownEvent, _, _| {
                                *drag_down.lock().unwrap() = Some(ev.position);
                            })
                            .on_scroll_wheel(move |ev: &ScrollWheelEvent, _, _| {
                                zoom.lock().unwrap().zoom(f32::from(ev.delta.pixel_delta(px(16.)).y));
                            })
                            .child(wgpu_surface(self.surface.clone()).absolute().inset_0()),
                    ),
            )
            .child(
                div()
                    .px_6()
                    .pb_4()
                    .text_xs()
                    .text_color(rgb(0x9a9aa5))
                    .child("Glisser : orbite · Molette : zoom"),
            )
    }
}

/// Vue isolée : son `notify` 2×/s ne touche pas la surface 3D.
struct FpsLabel {
    text: String,
}

impl FpsLabel {
    fn new(frames: Arc<AtomicU64>, cx: &mut Context<Self>) -> Self {
        cx.spawn(async move |this, cx| {
            let mut last = (Instant::now(), 0u64);
            loop {
                cx.background_executor().timer(Duration::from_millis(500)).await;
                let n = frames.load(Ordering::Relaxed);
                let fps = (n - last.1) as f64 / last.0.elapsed().as_secs_f64();
                last = (Instant::now(), n);
                if this.update(cx, |l, cx| { l.text = format!("{fps:.0} fps"); cx.notify() }).is_err() {
                    break;
                }
            }
        })
        .detach();
        Self { text: "— fps".into() }
    }
}

impl Render for FpsLabel {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().child(self.text.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn la_camera_reste_dans_ses_bornes() {
        let mut cam = OrbitCamera::default();
        cam.orbit(0.0, 10_000.0);
        assert_eq!(cam.pitch, 1.5, "pas de passage par le pôle");
        cam.zoom(-1e6);
        assert_eq!(cam.distance, 20.0);
        cam.zoom(1e6);
        assert_eq!(cam.distance, 1.2);
        assert!((cam.eye().length() - 1.2).abs() < 1e-5, "l'œil est à `distance` de l'origine");
    }
}
