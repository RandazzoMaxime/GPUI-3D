//! GPUI-3D Starter : un petit éditeur de scène. Point de départ d'une app GPUI avec
//! un moteur 3D derrière une UI translucide. Les règles qui font la perf sont ici :
//!
//! - un device pour l'UI et la 3D, le moteur rend sur son fil dans une `WgpuSurface`
//!   (engine.rs) : une trame 3D ne redessine jamais l'UI ;
//! - l'état vit dans des entités (`Object`, `Model`) ; l'UI publie un instantané au
//!   moteur quand la scène change, jamais l'inverse ;
//! - chaque panneau et chaque ligne de liste est une vue `.cached()` : modifier un objet
//!   ne reconstruit que sa ligne et l'inspecteur, défiler ne reconstruit rien ;
//! - les gestes caméra écrivent la caméra partagée, sans `cx.notify()`.

mod engine;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use engine::{Instance, SELECTED, SPIN, Shared};
use gpui::{
    AnyView, App, Application, Bounds, Context, Entity, EntityId, FontWeight, Hsla, MouseButton,
    MouseDownEvent, MouseMoveEvent, Pixels, Point, Render, ScrollWheelEvent, SharedString,
    StyleRefinement, TitlebarOptions, WeakEntity, WgpuSurfaceHandle, Window, WindowBounds,
    WindowOptions, div, hsla, prelude::*, px, rgb, size, uniform_list, wgpu_surface,
};

const PALETTE: [(f32, f32, f32); 6] =
    [(0.0, 0.7, 0.55), (0.08, 0.8, 0.55), (0.14, 0.8, 0.5), (0.36, 0.5, 0.45), (0.58, 0.6, 0.5), (0.78, 0.5, 0.55)];
const ROW_HEIGHT: f32 = 44.;

// ---------------------------------------------------------------------------------
// État
// ---------------------------------------------------------------------------------

struct Object {
    name: SharedString,
    color: Hsla,
    position: [f32; 3],
    scale: f32,
    spin: bool,
    selected: bool,
}

impl Object {
    fn instance(&self) -> Instance {
        let rgba = self.color.to_rgb();
        let height = self.scale;
        Instance {
            position: [self.position[0], self.position[1] + height / 2. - 0.5, self.position[2]],
            scale: [self.scale, height, self.scale],
            color: [rgba.r, rgba.g, rgba.b],
            flags: if self.spin { SPIN } else { 0 } | if self.selected { SELECTED } else { 0 },
        }
    }
}

/// La scène : la liste des objets et la sélection. Publie un instantané au moteur à
/// chaque changement ; `cx.notify()` seulement quand la liste ou la sélection change.
struct Model {
    objects: Vec<Entity<Object>>,
    selected: Option<Entity<Object>>,
    created: usize,
    shared: Arc<Shared>,
}

impl Model {
    fn publish(&self, cx: &App) {
        self.shared.publish(self.objects.iter().map(|o| o.read(cx).instance()).collect());
    }

    fn add(&mut self, cx: &mut Context<Self>) {
        let n = self.created;
        self.created += 1;
        let angle = n as f32 * 2.4;
        let radius = 1.8 + (n % 5) as f32 * 0.9;
        let (h, s, l) = PALETTE[n % PALETTE.len()];
        let object = cx.new(|_| Object {
            name: format!("Cube {}", n + 1).into(),
            color: hsla(h, s, l, 1.),
            position: [radius * angle.cos(), 0., radius * angle.sin()],
            scale: 0.6 + (n % 3) as f32 * 0.3,
            spin: n % 2 == 0,
            selected: false,
        });
        self.objects.push(object.clone());
        self.select(Some(object), cx);
    }

    fn select(&mut self, object: Option<Entity<Object>>, cx: &mut Context<Self>) {
        for (entity, selected) in [(self.selected.take(), false), (object.clone(), true)] {
            if let Some(entity) = entity {
                entity.update(cx, |o, cx| {
                    o.selected = selected;
                    cx.notify();
                });
            }
        }
        self.selected = object;
        self.publish(cx);
        cx.notify();
    }

    fn remove_selected(&mut self, cx: &mut Context<Self>) {
        if let Some(selected) = self.selected.take() {
            self.objects.retain(|o| o != &selected);
            let next = self.objects.last().cloned();
            self.select(next, cx);
        }
    }

    /// Modifie l'objet sélectionné : seule sa ligne et l'inspecteur se reconstruisent.
    fn edit(&mut self, edit: impl FnOnce(&mut Object), cx: &mut Context<Self>) {
        if let Some(selected) = &self.selected {
            selected.update(cx, |o, cx| {
                edit(o);
                cx.notify();
            });
            self.publish(cx);
        }
    }
}

// ---------------------------------------------------------------------------------
// Vues
// ---------------------------------------------------------------------------------

fn panel() -> gpui::Div {
    div()
        .bg(hsla(0., 0., 1., 0.82))
        .border_1()
        .border_color(hsla(0., 0., 0., 0.08))
        .rounded_lg()
}

fn button(id: &'static str, label: impl Into<SharedString>) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .px_3()
        .py_1()
        .rounded_md()
        .border_1()
        .border_color(rgb(0xd5d7de))
        .bg(rgb(0xffffff))
        .text_sm()
        .cursor_pointer()
        .hover(|s| s.bg(rgb(0xf0f2f7)))
        .child(label.into())
}

/// Enveloppe une vue en `.cached()` : sa taille est connue sans la rendre.
fn cached(view: impl Into<AnyView>, width: impl Into<gpui::Length>, height: Option<gpui::Length>) -> AnyView {
    let mut style = StyleRefinement::default();
    style.size.width = Some(width.into());
    style.size.height = Some(height.unwrap_or_else(|| gpui::relative(1.).into()));
    view.into().cached(style)
}

/// Une ligne de la liste. Vue `.cached()` : ne se reconstruit que si son objet change ;
/// défilée, elle est rejouée translatée.
struct Row {
    object: Entity<Object>,
    model: WeakEntity<Model>,
}

impl Render for Row {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let o = self.object.read(cx);
        div()
            .id("row")
            .h(px(ROW_HEIGHT))
            .w_full()
            .px_3()
            .flex()
            .items_center()
            .gap_3()
            .rounded_md()
            .when(o.selected, |d| d.bg(hsla(0.6, 0.8, 0.6, 0.18)))
            .when(!o.selected, |d| d.hover(|s| s.bg(hsla(0., 0., 0., 0.04))))
            .child(div().size(px(14.)).rounded_sm().bg(o.color))
            .child(div().flex_1().text_sm().child(o.name.clone()))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x8a8a96))
                    .child(format!("{:.1}, {:.1}", o.position[0], o.position[2])),
            )
            .on_click(cx.listener(|this, _, _, cx| {
                let object = this.object.clone();
                this.model.update(cx, |m, cx| m.select(Some(object), cx)).ok();
            }))
    }
}

type RowCache = Rc<RefCell<HashMap<EntityId, Entity<Row>>>>;

/// Liste virtualisée (`uniform_list`) de lignes `.cached()`.
struct ObjectList {
    model: Entity<Model>,
    rows: RowCache,
}

impl Render for ObjectList {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let objects = self.model.read(cx).objects.clone();
        let (rows, model) = (self.rows.clone(), self.model.downgrade());
        rows.borrow_mut().retain(|id, _| objects.iter().any(|o| o.entity_id() == *id));
        let count = objects.len();
        div()
            .size_full()
            .flex()
            .flex_col()
            .p_2()
            .child(div().px_2().pb_2().text_xs().text_color(rgb(0x6b6b76)).child(format!("SCÈNE · {count} objets")))
            .child(
                uniform_list("objects", count, move |range, _, cx| {
                    let mut rows = rows.borrow_mut();
                    range
                        .map(|ix| {
                            let object = objects[ix].clone();
                            let row = rows
                                .entry(object.entity_id())
                                .or_insert_with(|| cx.new(|_| Row { object, model: model.clone() }))
                                .clone();
                            cached(row, gpui::relative(1.), Some(px(ROW_HEIGHT).into())).into_any_element()
                        })
                        .collect()
                })
                .flex_1(),
            )
    }
}

/// Propriétés de l'objet sélectionné.
struct Inspector {
    model: Entity<Model>,
}

impl Render for Inspector {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(selected) = self.model.read(cx).selected.clone() else {
            return div().p_4().text_sm().text_color(rgb(0x8a8a96)).child("Sélectionnez un objet");
        };
        let o = selected.read(cx);
        let model = self.model.clone();
        let edit = move |f: fn(&mut Object)| {
            let model = model.clone();
            move |_: &gpui::ClickEvent, _: &mut Window, cx: &mut App| {
                model.update(cx, |m, cx| m.edit(f, cx));
            }
        };
        let swatches = PALETTE.iter().enumerate().map(|(i, &(h, s, l))| {
            let model = self.model.clone();
            let color = hsla(h, s, l, 1.);
            div()
                .id(("swatch", i))
                .size(px(22.))
                .rounded_md()
                .bg(color)
                .cursor_pointer()
                .when(o.color == color, |d| d.border_2().border_color(rgb(0x1a1a1f)))
                .on_click(move |_, _, cx| model.update(cx, |m, cx| m.edit(|o| o.color = color, cx)))
        });
        let row = |label: &'static str| div().flex().items_center().justify_between().text_sm().child(label);
        div()
            .p_4()
            .flex()
            .flex_col()
            .gap_4()
            .child(div().text_lg().font_weight(FontWeight::BOLD).child(o.name.clone()))
            .child(div().flex().gap_2().children(swatches))
            .child(
                row("Taille").child(
                    div()
                        .flex()
                        .gap_1()
                        .child(button("smaller", "−").on_click(edit(|o| o.scale = (o.scale - 0.2).max(0.2))))
                        .child(button("bigger", "+").on_click(edit(|o| o.scale = (o.scale + 0.2).min(3.)))),
                ),
            )
            .child(
                row("Rotation").child(
                    button("spin", if o.spin { "activée" } else { "arrêtée" }).on_click(edit(|o| o.spin = !o.spin)),
                ),
            )
    }
}

/// Barre du haut : actions et fps (seule vue mise à jour en continu, 2×/s).
struct Toolbar {
    model: Entity<Model>,
    fps: Entity<Fps>,
}

impl Render for Toolbar {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let (add, remove) = (self.model.clone(), self.model.clone());
        panel()
            .size_full()
            .px_4()
            .flex()
            .items_center()
            .gap_2()
            .child(div().text_lg().font_weight(FontWeight::BOLD).mr_4().child("GPUI-3D Starter"))
            .child(button("add", "+ Cube").on_click(move |_, _, cx| add.update(cx, |m, cx| m.add(cx))))
            .child(button("remove", "Supprimer").on_click(move |_, _, cx| remove.update(cx, |m, cx| m.remove_selected(cx))))
            .child(div().flex_1())
            .child(div().text_sm().text_color(rgb(0x6b6b76)).child(self.fps.clone()))
    }
}

struct Fps {
    text: SharedString,
}

impl Fps {
    fn new(shared: Arc<Shared>, cx: &mut Context<Self>) -> Self {
        cx.spawn(async move |this, cx| {
            let mut last = (Instant::now(), 0);
            loop {
                cx.background_executor().timer(Duration::from_millis(500)).await;
                let frames = shared.frames.load(Ordering::Relaxed);
                let fps = (frames - last.1) as f64 / last.0.elapsed().as_secs_f64();
                last = (Instant::now(), frames);
                let text = format!("3D : {fps:.0} fps");
                if this.update(cx, |f, cx| { f.text = text.into(); cx.notify() }).is_err() {
                    break;
                }
            }
        })
        .detach();
        Self { text: "3D : — fps".into() }
    }
}

impl Render for Fps {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().child(self.text.clone())
    }
}

/// Racine : la 3D en fond, les panneaux par-dessus, la caméra au milieu.
struct Starter {
    surface: WgpuSurfaceHandle,
    shared: Arc<Shared>,
    toolbar: AnyView,
    list: AnyView,
    inspector: AnyView,
    drag_from: Arc<Mutex<Option<Point<Pixels>>>>,
}

impl Render for Starter {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let (down, moving, up) = (self.drag_from.clone(), self.drag_from.clone(), self.drag_from.clone());
        let (orbit, zoom) = (self.shared.clone(), self.shared.clone());
        div()
            .size_full()
            .relative()
            .font_family("IBM Plex Sans")
            .text_color(rgb(0x1a1a1f))
            // Le glisser continue au-dessus des panneaux : l'écoute est sur la racine.
            .on_mouse_move(move |ev: &MouseMoveEvent, _, _| {
                let mut from = moving.lock().unwrap();
                match (*from, ev.pressed_button) {
                    (Some(prev), Some(MouseButton::Left)) => {
                        let d = ev.position - prev;
                        orbit.camera.lock().unwrap().orbit(f32::from(d.x), f32::from(d.y));
                        *from = Some(ev.position);
                    }
                    (_, None) => *from = None,
                    _ => {}
                }
            })
            .on_mouse_up(MouseButton::Left, move |_, _, _| *up.lock().unwrap() = None)
            .child(wgpu_surface(self.surface.clone()).absolute().inset_0())
            .child(
                div()
                    .absolute()
                    .inset_0()
                    .p_3()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .child(div().h(px(52.)).child(self.toolbar.clone()))
                    .child(
                        div()
                            .flex_1()
                            .min_h_0()
                            .flex()
                            .gap_3()
                            .child(panel().h_full().child(self.list.clone()))
                            .child(
                                div()
                                    .id("viewport")
                                    .flex_1()
                                    .h_full()
                                    .on_mouse_down(MouseButton::Left, move |ev: &MouseDownEvent, _, _| {
                                        *down.lock().unwrap() = Some(ev.position);
                                    })
                                    .on_scroll_wheel(move |ev: &ScrollWheelEvent, _, _| {
                                        let dy = f32::from(ev.delta.pixel_delta(px(16.)).y);
                                        zoom.camera.lock().unwrap().zoom(dy);
                                    }),
                            )
                            .child(panel().h_full().child(self.inspector.clone())),
                    ),
            )
    }
}

fn main() {
    Application::new().run(|cx: &mut App| {
        cx.text_system()
            .add_fonts(vec![
                include_bytes!("../../vendor/wgpui/assets/fonts/ibm-plex-sans/IBMPlexSans-Regular.ttf").as_slice().into(),
                include_bytes!("../../vendor/wgpui/assets/fonts/ibm-plex-sans/IBMPlexSans-Bold.ttf").as_slice().into(),
            ])
            .expect("polices");
        let bounds = Bounds::centered(None, size(px(1280.), px(800.)), cx);
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            titlebar: Some(TitlebarOptions { title: Some("GPUI-3D Starter".into()), ..Default::default() }),
            ..Default::default()
        };
        cx.open_window(options, |window, cx| {
            let scale = window.scale_factor();
            let vp = window.viewport_size();
            let surface = window
                .create_wgpu_surface(
                    (f32::from(vp.width) * scale) as u32,
                    (f32::from(vp.height) * scale) as u32,
                    engine::SURFACE_FORMAT,
                )
                .expect("WgpuSurface");
            let hz = window
                .display(cx)
                .and_then(|d| d.refresh_rate_millihertz())
                .map_or(60., |mhz| f64::from(mhz) / 1000.);
            let shared = Arc::new(Shared::default());
            engine::spawn(surface.clone(), shared.clone(), hz);

            let model = cx.new(|_| Model { objects: Vec::new(), selected: None, created: 0, shared: shared.clone() });
            model.update(cx, |m, cx| (0..8).for_each(|_| m.add(cx)));
            let list = cx.new(|cx| {
                cx.observe(&model, |_, _, cx| cx.notify()).detach();
                ObjectList { model: model.clone(), rows: RowCache::default() }
            });
            let inspector = cx.new(|cx| {
                cx.observe(&model, |_, _, cx| cx.notify()).detach();
                Inspector { model: model.clone() }
            });
            let fps = cx.new(|cx| Fps::new(shared.clone(), cx));
            let toolbar = cx.new(|_| Toolbar { model: model.clone(), fps });
            cx.new(|_| Starter {
                surface,
                shared,
                toolbar: cached(toolbar, gpui::relative(1.), Some(px(52.).into())),
                list: cached(list, px(300.), None),
                inspector: cached(inspector, px(300.), None),
                drag_from: Arc::new(Mutex::new(None)),
            })
        })
        .expect("fenêtre");
        cx.on_window_closed(|cx, _| cx.quit()).detach();
        cx.activate(true);
    });
}
