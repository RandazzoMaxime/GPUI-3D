//! Banc de charge CPU d'un défilement continu, UI type client mail (liste de 10 000
//! messages, barre latérale, barre d'outils, panneau de lecture). Mesure le CPU du
//! process (getrusage, tous fils) pendant `BENCH_SECS` de défilement ininterrompu.
//!
//! `BENCH_MODE=root`   : toute l'UI dans une seule vue, notifiée à chaque trame.
//! `BENCH_MODE=cached` : panneaux en vues `.cached()` (façon Zed), seule la liste
//!                       est notifiée.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

use gpui::{
    AnyElement, AnyView, App, Application, Bounds, Context, Entity, FontWeight, Hsla, Render,
    SharedString, StyleRefinement, UniformListScrollHandle, Window, WindowBounds, WindowOptions,
    div, hsla, point, prelude::*, px, rgb, size, uniform_list,
};

const ROWS: usize = 10_000;
const ROW_HEIGHT: f32 = 64.0;
const SPEED_PX_PER_FRAME: f32 = 6.0;

struct Mail {
    sender: SharedString,
    initials: SharedString,
    subject: SharedString,
    preview: SharedString,
    date: SharedString,
    color: Hsla,
    unread: bool,
    badges: Vec<SharedString>,
}

fn mails() -> Rc<Vec<Mail>> {
    const NAMES: [&str; 12] = [
        "Alice Martin", "Bruno Rossi", "Chloé Dubois", "David Chen", "Emma Laurent", "Farid Haddad",
        "Giulia Bianchi", "Hugo Moreau", "Inès Garcia", "Jules Petit", "Karim Benali", "Léa Fontaine",
    ];
    const SUBJECTS: [&str; 8] = [
        "Relevé LiDAR du secteur nord — validation", "Plan de vol : corridor Ajaccio",
        "Re: Calage altimétrique iAGL", "Facture septembre", "Mise à jour firmware M350",
        "Point hebdo équipe terrain", "Export WPML refusé par la RC", "Nouvelle zone NOTAM",
    ];
    const PREVIEWS: [&str; 4] = [
        "Bonjour, vous trouverez ci-joint les résultats du vol d'hier, avec la couverture et les écarts mesurés sur les points de contrôle.",
        "Petit rappel : la réunion est déplacée à jeudi 14h, merci de confirmer votre présence et d'apporter les relevés.",
        "Le fichier généré ne passe pas la validation, il manque l'altitude de sécurité sur deux waypoints du segment 3.",
        "Suite à notre échange, voici la proposition mise à jour avec le chiffrage détaillé et le planning prévisionnel.",
    ];
    const BADGES: [&str; 5] = ["Terrain", "Urgent", "Client", "Compta", "Drone"];
    (0..ROWS)
        .map(|i| {
            let name = NAMES[i % NAMES.len()];
            let initials: String = name.split(' ').filter_map(|w| w.chars().next()).collect();
            Mail {
                sender: name.into(),
                initials: initials.into(),
                subject: format!("{} #{i}", SUBJECTS[(i * 7) % SUBJECTS.len()]).into(),
                preview: PREVIEWS[(i * 3) % PREVIEWS.len()].into(),
                date: format!("{:02}/09 {:02}:{:02}", 1 + i % 28, (i * 5) % 24, (i * 13) % 60).into(),
                color: hsla((i % 12) as f32 / 12.0, 0.55, 0.5, 1.0),
                unread: i % 3 == 0,
                badges: (0..(i % 3)).map(|b| BADGES[(i + b) % BADGES.len()].into()).collect(),
            }
        })
        .collect::<Vec<_>>()
        .into()
}

fn mail_row(ix: usize, mail: &Mail) -> AnyElement {
    div()
        .id(ix)
        .h(px(ROW_HEIGHT))
        .w_full()
        .flex()
        .items_center()
        .gap_3()
        .px_4()
        .border_b_1()
        .border_color(rgb(0xececf0))
        .hover(|s| s.bg(rgb(0xf4f6fb)))
        .on_click(|_, _, _| {})
        .child(
            div()
                .size(px(36.))
                .rounded_full()
                .bg(mail.color)
                .flex()
                .items_center()
                .justify_center()
                .text_color(rgb(0xffffff))
                .text_sm()
                .child(mail.initials.clone()),
        )
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .child(
                    div()
                        .flex()
                        .justify_between()
                        .child(
                            div()
                                .font_weight(if mail.unread { FontWeight::BOLD } else { FontWeight::NORMAL })
                                .child(mail.sender.clone()),
                        )
                        .child(div().text_xs().text_color(rgb(0x8a8a96)).child(mail.date.clone())),
                )
                .child(div().text_sm().truncate().child(mail.subject.clone()))
                .child(div().text_xs().text_color(rgb(0x8a8a96)).truncate().child(mail.preview.clone())),
        )
        .child(div().flex().gap_1().children(mail.badges.iter().map(|b| {
            div().px_1p5().rounded_md().bg(rgb(0xe8eefc)).text_xs().child(b.clone())
        })))
        .into_any_element()
}

/// Une ligne de la liste en vue `.cached()` : son rendu n'est refait que si elle
/// est notifiée ; défilée, elle est rejouée (translatée) depuis la trame précédente.
struct RowView {
    mails: Rc<Vec<Mail>>,
    ix: usize,
}

impl Render for RowView {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        mail_row(self.ix, &self.mails[self.ix])
    }
}

type RowCache = Rc<RefCell<HashMap<usize, Entity<RowView>>>>;

fn cached_rows(mails: &Rc<Vec<Mail>>, cache: &RowCache, range: std::ops::Range<usize>, cx: &mut App) -> Vec<AnyElement> {
    let mut cache = cache.borrow_mut();
    cache.retain(|ix, _| *ix + 64 >= range.start && *ix < range.end + 64);
    let mut style = StyleRefinement::default();
    style.size.width = Some(gpui::relative(1.).into());
    style.size.height = Some(px(ROW_HEIGHT).into());
    range
        .map(|ix| {
            let row = cache
                .entry(ix)
                .or_insert_with(|| cx.new(|_| RowView { mails: mails.clone(), ix }))
                .clone();
            AnyView::from(row).cached(style.clone()).into_any_element()
        })
        .collect()
}

/// Avance le défilement et redemande une trame : ce que fait une molette continue.
fn mail_list(
    mails: &Rc<Vec<Mail>>,
    scroll: &UniformListScrollHandle,
    offset: &Cell<f32>,
    frames: &Cell<u64>,
    rows: &Option<RowCache>,
    window: &mut Window,
) -> AnyElement {
    let max = ROWS as f32 * ROW_HEIGHT - 2000.0;
    static SPEED: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| {
        std::env::var("BENCH_SPEED").ok().and_then(|v| v.parse().ok()).unwrap_or(SPEED_PX_PER_FRAME)
    });
    static STOP_AT: std::sync::LazyLock<Option<u64>> =
        std::sync::LazyLock::new(|| std::env::var("BENCH_STOP_AT").ok().and_then(|v| v.parse().ok()));
    // Arrêt figé pour la capture de parité pixel : plus de défilement ni de trame demandée.
    let running = STOP_AT.is_none_or(|stop| frames.get() < stop);
    if running {
        offset.set((offset.get() + *SPEED) % max);
        frames.set(frames.get() + 1);
        window.request_animation_frame();
    } else if frames.get() == STOP_AT.unwrap_or(0) {
        frames.set(frames.get() + 1);
        eprintln!("READY offset={}", offset.get());
    }
    scroll.0.borrow().base_handle.set_offset(point(px(0.), px(-offset.get())));
    let mails = mails.clone();
    let rows = rows.clone();
    uniform_list("mails", ROWS, move |range, _, cx| match &rows {
        Some(cache) => cached_rows(&mails, cache, range, cx),
        None => range.map(|ix| mail_row(ix, &mails[ix])).collect(),
    })
        .track_scroll(scroll)
        .size_full()
        .into_any_element()
}

fn sidebar() -> AnyElement {
    const FOLDERS: [&str; 10] = [
        "Boîte de réception", "Favoris", "Envoyés", "Brouillons", "Archives", "Terrain", "Clients",
        "Compta", "Spam", "Corbeille",
    ];
    div()
        .w(px(220.))
        .h_full()
        .flex()
        .flex_col()
        .gap_1()
        .p_3()
        .bg(rgb(0xf7f7f9))
        .border_r_1()
        .border_color(rgb(0xe6e6ea))
        .children((0..25).map(|i| {
            div()
                .id(("folder", i))
                .flex()
                .items_center()
                .gap_2()
                .px_2()
                .py_1()
                .rounded_md()
                .hover(|s| s.bg(rgb(0xececf2)))
                .child(div().size(px(12.)).rounded_sm().bg(hsla(i as f32 / 25.0, 0.5, 0.6, 1.0)))
                .child(div().flex_1().text_sm().child(FOLDERS[i % FOLDERS.len()]))
                .child(div().text_xs().text_color(rgb(0x8a8a96)).child(format!("{}", (i * 37) % 250)))
        }))
        .into_any_element()
}

fn toolbar() -> AnyElement {
    div()
        .h(px(52.))
        .w_full()
        .flex()
        .items_center()
        .gap_2()
        .px_4()
        .border_b_1()
        .border_color(rgb(0xe6e6ea))
        .child(div().text_xl().font_weight(FontWeight::BOLD).child("Courrier"))
        .child(
            div()
                .flex_1()
                .mx_4()
                .px_3()
                .py_1()
                .rounded_md()
                .bg(rgb(0xf0f0f4))
                .text_sm()
                .text_color(rgb(0x8a8a96))
                .child("Rechercher dans les messages…"),
        )
        .children(["Nouveau", "Répondre", "Transférer", "Archiver", "Supprimer", "Filtrer"].map(|label| {
            div()
                .id(label)
                .px_3()
                .py_1()
                .rounded_md()
                .border_1()
                .border_color(rgb(0xdcdce2))
                .text_sm()
                .hover(|s| s.bg(rgb(0xf0f0f4)))
                .child(label)
        }))
        .into_any_element()
}

fn reader() -> AnyElement {
    const PARAGRAPH: &str = "Suite au vol réalisé ce matin sur la zone de Porto-Vecchio, la couverture photogrammétrique \
        atteint 98 % de la surface demandée. Les écarts mesurés sur les points d'appui restent sous la tolérance \
        de 3 cm en planimétrie et 5 cm en altimétrie. Deux bandes seront refaites demain à cause du vent.";
    div()
        .w(px(460.))
        .h_full()
        .flex()
        .flex_col()
        .gap_3()
        .p_5()
        .border_l_1()
        .border_color(rgb(0xe6e6ea))
        .overflow_hidden()
        .child(div().text_lg().font_weight(FontWeight::BOLD).child("Relevé LiDAR du secteur nord — validation"))
        .child(div().text_sm().text_color(rgb(0x8a8a96)).child("Alice Martin · 24/09 09:12"))
        .children((0..14).map(|_| div().text_sm().child(PARAGRAPH)))
        .into_any_element()
}

fn status_bar(label: &'static str) -> AnyElement {
    div()
        .h(px(26.))
        .w_full()
        .flex()
        .items_center()
        .px_4()
        .border_t_1()
        .border_color(rgb(0xe6e6ea))
        .text_xs()
        .text_color(rgb(0x8a8a96))
        .child(format!("{ROWS} messages · {label}"))
        .into_any_element()
}

fn layout(toolbar: AnyElement, sidebar: AnyElement, list: AnyElement, reader: AnyElement, status: AnyElement) -> AnyElement {
    div()
        .size_full()
        .flex()
        .flex_col()
        .bg(rgb(0xffffff))
        .text_color(rgb(0x1a1a1f))
        .child(toolbar)
        .child(div().flex_1().min_h_0().flex().child(sidebar).child(div().flex_1().h_full().child(list)).child(reader))
        .child(status)
        .into_any_element()
}

struct ListState {
    mails: Rc<Vec<Mail>>,
    scroll: UniformListScrollHandle,
    offset: Cell<f32>,
    frames: Rc<Cell<u64>>,
    rows: Option<RowCache>,
}

impl ListState {
    fn new(frames: Rc<Cell<u64>>) -> Self {
        let rows = std::env::var("BENCH_ROWS").is_ok_and(|v| v == "1").then(RowCache::default);
        Self { mails: mails(), scroll: UniformListScrollHandle::new(), offset: Cell::new(0.), frames, rows }
    }
}

/// Tout dans une vue : chaque trame de défilement reconstruit la fenêtre entière.
struct RootMode(ListState);

impl Render for RootMode {
    fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let s = &self.0;
        let list = mail_list(&s.mails, &s.scroll, &s.offset, &s.frames, &s.rows, window);
        layout(toolbar(), sidebar(), list, reader(), status_bar("root"))
    }
}

/// Panneaux en `.cached()`, liste dans sa propre vue notifiée à chaque trame.
struct CachedMode {
    toolbar: AnyView,
    sidebar: AnyView,
    list: Entity<ListView>,
    reader: AnyView,
    status: AnyView,
}

struct ListView(ListState);
struct Static(fn() -> AnyElement);

impl Render for ListView {
    fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let s = &self.0;
        mail_list(&s.mails, &s.scroll, &s.offset, &s.frames, &s.rows, window)
    }
}

impl Render for Static {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        (self.0)()
    }
}

fn full() -> StyleRefinement {
    let mut style = StyleRefinement::default();
    style.size.width = Some(gpui::relative(1.).into());
    style.size.height = Some(gpui::relative(1.).into());
    style
}

impl Render for CachedMode {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let cached = |view: &AnyView, style: StyleRefinement| view.clone().cached(style).into_any_element();
        let mut toolbar_style = StyleRefinement::default();
        toolbar_style.size.width = Some(gpui::relative(1.).into());
        toolbar_style.size.height = Some(px(52.).into());
        let mut sidebar_style = StyleRefinement::default();
        sidebar_style.size.width = Some(px(220.).into());
        sidebar_style.size.height = Some(gpui::relative(1.).into());
        let mut reader_style = sidebar_style.clone();
        reader_style.size.width = Some(px(460.).into());
        let mut status_style = toolbar_style.clone();
        status_style.size.height = Some(px(26.).into());
        layout(
            cached(&self.toolbar, toolbar_style),
            cached(&self.sidebar, sidebar_style),
            AnyView::from(self.list.clone()).cached(full()).into_any_element(),
            cached(&self.reader, reader_style),
            cached(&self.status, status_style),
        )
    }
}

struct Root(AnyView);

impl Render for Root {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.0.clone()
    }
}

/// Instructions retirées et cycles du process (tous fils) : indépendants de la
/// fréquence CPU et du type de cœur, contrairement au temps CPU.
fn instructions_cycles() -> (u64, u64) {
    let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
    let ok = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as i32,
            libc::RUSAGE_INFO_V4,
            &mut info as *mut _ as *mut libc::rusage_info_t,
        )
    };
    assert_eq!(ok, 0, "proc_pid_rusage");
    (info.ri_instructions, info.ri_cycles)
}

fn cpu_seconds() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let t = |tv: libc::timeval| tv.tv_sec as f64 + tv.tv_usec as f64 / 1e6;
    t(usage.ru_utime) + t(usage.ru_stime)
}

fn main() {
    let mode = std::env::var("BENCH_MODE").unwrap_or_else(|_| "root".into());
    let secs: f64 = std::env::var("BENCH_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(8.0);
    let frames = Rc::new(Cell::new(0u64));
    Application::new().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1400.), px(900.)), cx);
        let options = WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        let frames_view = frames.clone();
        let mode_view = mode.clone();
        cx.open_window(options, move |_, cx: &mut App| {
            let view: AnyView = 
            if mode_view == "cached" {
                let list = cx.new(|_| ListView(ListState::new(frames_view)));
                let view = |f: fn() -> AnyElement, cx: &mut App| AnyView::from(cx.new(|_| Static(f)));
                cx.new(|cx| CachedMode {
                    toolbar: view(toolbar, cx),
                    sidebar: view(sidebar, cx),
                    list,
                    reader: view(reader, cx),
                    status: AnyView::from(cx.new(|_| Static(|| status_bar("cached")))),
                })
                .into()
            } else {
                cx.new(|_| RootMode(ListState::new(frames_view))).into()
            };
            cx.new(|_| Root(view))
        })
        .expect("fenêtre");
        cx.activate(true);
        cx.spawn(async move |cx| {
            let warmup = Duration::from_secs(2);
            cx.background_executor().timer(warmup).await;
            let (t0, cpu0, f0, (i0, c0)) = (Instant::now(), cpu_seconds(), frames.get(), instructions_cycles());
            cx.background_executor().timer(Duration::from_secs_f64(secs)).await;
            let (wall, cpu, n) = (t0.elapsed().as_secs_f64(), cpu_seconds() - cpu0, frames.get() - f0);
            let (i1, c1) = instructions_cycles();
            let per_frame = |v: u64| v as f64 / n.max(1) as f64 / 1e6;
            println!(
                "RESULT mode={mode} rows={} translate={} speed={} frames={n} fps={:.1} cpu_pct={:.1} cpu_ms_per_frame={:.3} minstr_per_frame={:.2} mcycles_per_frame={:.2}",
                std::env::var("BENCH_ROWS").unwrap_or_else(|_| "0".into()),
                std::env::var("GPUI_VIEW_TRANSLATE").unwrap_or_else(|_| "1".into()),
                std::env::var("BENCH_SPEED").unwrap_or_else(|_| "6".into()),
                n as f64 / wall,
                100.0 * cpu / wall,
                1000.0 * cpu / n.max(1) as f64,
                per_frame(i1 - i0),
                per_frame(c1 - c0),
            );
            cx.update(|cx| cx.quit());
        })
        .detach();
    });
}
