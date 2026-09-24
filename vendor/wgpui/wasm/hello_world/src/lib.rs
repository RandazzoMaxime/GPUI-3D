use gpui::{
    div, prelude::*, px, rgb, size, App, Application, Bounds, Context, Window, WindowBounds,
    WindowOptions,
};
use wasm_bindgen::prelude::*;

fn elapsed_secs() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now() / 1000.0)
        .unwrap_or(0.0)
}

struct HelloWasm {
    start: f64,
    dropdown_open: bool,
    selected: usize,
    items: Vec<&'static str>,
}

impl gpui::Render for HelloWasm {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl gpui::IntoElement {
        cx.notify();
        let elapsed = (elapsed_secs() - self.start) as f32;
        let view = window.viewport_size();
        let max_w = (view.width - px(48.0)).max(px(0.0));
        let max_h = (view.height - px(48.0)).max(px(0.0));
        let tx = (elapsed * 1.2) % 2.0;
        let ty = (elapsed * 0.9) % 2.0;
        let prog_x = if tx < 1.0 { tx } else { 2.0 - tx };
        let prog_y = if ty < 1.0 { ty } else { 2.0 - ty };
        let x = max_w * prog_x;
        let y = max_h * prog_y;

        div()
            .relative()
            .size_full()
            .bg(rgb(0x1a1a2e))
            // Color swatches
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap_2()
                    .absolute()
                    .top(px(16.0))
                    .left(px(16.0))
                    .child(div().size_10().bg(gpui::red()).border_1().rounded_md())
                    .child(div().size_10().bg(gpui::green()).border_1().rounded_md())
                    .child(div().size_10().bg(gpui::blue()).border_1().rounded_md())
                    .child(div().size_10().bg(gpui::yellow()).border_1().rounded_md())
                    .child(
                        div()
                            .size_10()
                            .bg(rgb(0xffffff))
                            .border_1()
                            .border_color(gpui::black())
                            .rounded_md(),
                    ),
            )
            // Bouncing ball
            .child(
                div()
                    .absolute()
                    .left(x)
                    .top(y)
                    .size_12()
                    .bg(rgb(0xe94560))
                    .rounded_full()
                    .shadow_lg(),
            )
            // Dropdown menu
            .child(
                div()
                    .absolute()
                    .top(px(16.0))
                    .right(px(16.0))
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .id("dropdown-button")
                            .px_3()
                            .py_1()
                            .bg(rgb(0x16213e))
                            .border_1()
                            .border_color(rgb(0x4a9eff))
                            .rounded_md()
                            .cursor_pointer()
                            .hover(|style| style.bg(rgb(0x1a5276)))
                            .on_click(cx.listener(|this, _event, _window, cx| {
                                this.dropdown_open = !this.dropdown_open;
                                cx.notify();
                            }))
                            .child(format!("Option {} ▾", self.items[self.selected])),
                    )
                    .children(
                        if self.dropdown_open {
                            self.items.iter().enumerate().map(|(i, item)| {
                                let is_selected = i == self.selected;
                                div()
                                    .id(format!("dropdown-item-{i}"))
                                    .px_3()
                                    .py_1()
                                    .mt_px()
                                    .bg(if is_selected {
                                        rgb(0x2a4a7f)
                                    } else {
                                        rgb(0x0f3460)
                                    })
                                    .hover(|style| style.bg(rgb(0x1a5276)))
                                    .rounded_md()
                                    .cursor_pointer()
                                    .on_click(cx.listener(move |this, _event, _window, cx| {
                                        this.selected = i;
                                        this.dropdown_open = false;
                                        cx.notify();
                                    }))
                                    .child(*item)
                                    .into_any_element()
                            }).collect::<Vec<_>>()
                        } else {
                            vec![]
                        },
                    ),
            )
            // Label
            .child(
                div()
                    .absolute()
                    .bottom(px(16.0))
                    .left(px(16.0))
                    .bg(rgb(0x16213e))
                    .px_3()
                    .py_1()
                    .rounded_md()
                    .text_color(rgb(0xaaaaaa))
                    .child("WGPUI on WASM"),
            )
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(800.), px(600.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| HelloWasm {
                    start: elapsed_secs(),
                    dropdown_open: false,
                    selected: 0,
                    items: vec!["Red", "Green", "Blue", "Yellow", "White"],
                })
            },
        )
        .unwrap();
        cx.activate(true);
    });
}
