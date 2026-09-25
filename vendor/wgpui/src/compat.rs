//! Compatibilité : API d'un fork gpui antérieur, attendues par des crates qui
//! en dépendent (gpui-component notamment). Elles sont rétablies ici sur le
//! WGPUI amont (Far-Beyond-Pulsar/WGPUI) sous forme minimale ; à reporter lors
//! d'une mise à jour amont. Voir aussi les blocs « Compat » dans
//! `elements/div.rs`, `styled.rs` et `text_system/line.rs`, et les re-exports
//! accesskit dans `gpui.rs`.

use std::borrow::Cow;
use std::fmt::Debug;
use std::ops::{Add, Sub};
use std::time::{Duration, Instant};

use crate::{
    AnyElement, App, AvailableSpace, Axis, Bounds, BoxShadow, Corner, Element, ElementId, Global,
    GlobalElementId, GradientStop, Half, Hsla, InspectorElementId, IntoElement, LayoutId,
    LinearColorStop, Pixels, Point, SharedString, Size, Style, TouchPhase, Window, px, relative,
};

impl Pixels {
    /// Alias historique de [`Pixels::to_f32`].
    pub fn as_f32(self) -> f32 {
        self.to_f32()
    }
}

/// Point d'ancrage d'un rectangle : les quatre coins plus le milieu de chaque
/// côté. Généralise [`crate::Corner`] pour le positionnement des popups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Anchor {
    /// Coin haut-gauche.
    #[default]
    TopLeft,
    /// Milieu du bord haut.
    TopCenter,
    /// Coin haut-droit.
    TopRight,
    /// Milieu du bord gauche.
    LeftCenter,
    /// Milieu du bord droit.
    RightCenter,
    /// Coin bas-gauche.
    BottomLeft,
    /// Milieu du bord bas.
    BottomCenter,
    /// Coin bas-droit.
    BottomRight,
}

impl From<Corner> for Anchor {
    fn from(corner: Corner) -> Self {
        match corner {
            Corner::TopLeft => Anchor::TopLeft,
            Corner::TopRight => Anchor::TopRight,
            Corner::BottomLeft => Anchor::BottomLeft,
            Corner::BottomRight => Anchor::BottomRight,
        }
    }
}

impl<'a> IntoElement for Cow<'a, str> {
    type Element = SharedString;

    fn into_element(self) -> Self::Element {
        self.into()
    }
}

impl Anchor {
    /// Renvoie l'ancre symétrique le long de l'axe donné (haut/bas pour
    /// [`Axis::Vertical`], gauche/droite pour [`Axis::Horizontal`]).
    #[must_use]
    pub fn other_side_along(self, axis: Axis) -> Self {
        match axis {
            Axis::Vertical => match self {
                Anchor::TopLeft => Anchor::BottomLeft,
                Anchor::TopCenter => Anchor::BottomCenter,
                Anchor::TopRight => Anchor::BottomRight,
                Anchor::BottomLeft => Anchor::TopLeft,
                Anchor::BottomCenter => Anchor::TopCenter,
                Anchor::BottomRight => Anchor::TopRight,
                Anchor::LeftCenter | Anchor::RightCenter => self,
            },
            Axis::Horizontal => match self {
                Anchor::TopLeft => Anchor::TopRight,
                Anchor::TopRight => Anchor::TopLeft,
                Anchor::BottomLeft => Anchor::BottomRight,
                Anchor::BottomRight => Anchor::BottomLeft,
                Anchor::LeftCenter => Anchor::RightCenter,
                Anchor::RightCenter => Anchor::LeftCenter,
                Anchor::TopCenter | Anchor::BottomCenter => self,
            },
        }
    }
}

impl<T> Bounds<T>
where
    T: Clone + Debug + Default + PartialEq + Add<T, Output = T> + Sub<T, Output = T> + Half,
{
    /// Construit un `Bounds` dont le point `anchor` se trouve à `position`.
    pub fn from_anchor_and_size(anchor: Anchor, position: Point<T>, size: Size<T>) -> Self {
        let x = match anchor {
            Anchor::TopLeft | Anchor::LeftCenter | Anchor::BottomLeft => position.x,
            Anchor::TopCenter | Anchor::BottomCenter => position.x - size.width.half(),
            Anchor::TopRight | Anchor::RightCenter | Anchor::BottomRight => {
                position.x - size.width.clone()
            }
        };
        let y = match anchor {
            Anchor::TopLeft | Anchor::TopCenter | Anchor::TopRight => position.y,
            Anchor::LeftCenter | Anchor::RightCenter => position.y - size.height.half(),
            Anchor::BottomLeft | Anchor::BottomCenter | Anchor::BottomRight => {
                position.y - size.height.clone()
            }
        };
        Bounds {
            origin: Point { x, y },
            size,
        }
    }

    /// Milieu du bord haut.
    pub fn top_center(&self) -> Point<T> {
        Point {
            x: self.origin.x.clone() + self.size.width.half(),
            y: self.origin.y.clone(),
        }
    }
}

/// Verrouillage d'un geste de scroll sur son axe dominant. Chaque geste
/// (délimité par `TouchPhase::Started` ou une pause entre événements) choisit
/// l'axe du premier delta dominant et annule l'autre composante ensuite.
#[derive(Debug, Default)]
pub struct OngoingScroll {
    last_event: Option<Instant>,
    axis: Option<Axis>,
}

impl OngoingScroll {
    /// Crée un état sans geste en cours.
    pub fn new() -> Self {
        Self::default()
    }

    /// Filtre `delta` en le verrouillant sur l'axe du geste en cours.
    pub fn filter(&mut self, delta: &mut Point<Pixels>, touch_phase: TouchPhase) {
        const GESTURE_SEPARATION: Duration = Duration::from_millis(28);
        let now = Instant::now();
        let stale = self
            .last_event
            .is_none_or(|last| now.duration_since(last) > GESTURE_SEPARATION);
        if matches!(touch_phase, TouchPhase::Started) || stale {
            self.axis = None;
        }
        self.last_event = Some(now);

        if self.axis.is_none() {
            if delta.x.abs() > delta.y.abs() {
                self.axis = Some(Axis::Horizontal);
            } else if delta.y.abs() > delta.x.abs() {
                self.axis = Some(Axis::Vertical);
            }
        }
        match self.axis {
            Some(Axis::Horizontal) => delta.y = px(0.),
            Some(Axis::Vertical) => delta.x = px(0.),
            None => {}
        }
    }
}

/// Rabat une ancre 8 directions sur le [`Corner`] le plus proche, pour les API
/// amont (`anchored().anchor(…)`) qui ne connaissent que les quatre coins. Les
/// milieux de côté perdent leur centrage : approximation assumée.
impl From<Anchor> for Corner {
    fn from(anchor: Anchor) -> Self {
        match anchor {
            Anchor::TopLeft | Anchor::TopCenter | Anchor::LeftCenter => Corner::TopLeft,
            Anchor::TopRight | Anchor::RightCenter => Corner::TopRight,
            Anchor::BottomLeft | Anchor::BottomCenter => Corner::BottomLeft,
            Anchor::BottomRight => Corner::BottomRight,
        }
    }
}

impl From<LinearColorStop> for GradientStop {
    fn from(stop: LinearColorStop) -> Self {
        GradientStop {
            color: stop.color,
            position: stop.percentage,
        }
    }
}

impl BoxShadow {
    /// Crée une ombre décalée de (`offset_x`, `offset_y`), sans flou ni
    /// étalement ; à compléter avec [`BoxShadow::blur_radius`] et
    /// [`BoxShadow::spread_radius`].
    pub fn new(offset_x: Pixels, offset_y: Pixels, color: impl Into<Hsla>) -> Self {
        Self {
            color: color.into(),
            offset: Point {
                x: offset_x,
                y: offset_y,
            },
            blur_radius: px(0.),
            spread_radius: px(0.),
        }
    }

    /// Fixe le rayon de flou.
    #[must_use]
    pub fn blur_radius(mut self, radius: Pixels) -> Self {
        self.blur_radius = radius;
        self
    }

    /// Fixe le rayon d'étalement.
    #[must_use]
    pub fn spread_radius(mut self, radius: Pixels) -> Self {
        self.spread_radius = radius;
        self
    }
}

impl std::iter::Sum for Pixels {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(px(0.), Add::add)
    }
}

impl<'a> std::iter::Sum<&'a Pixels> for Pixels {
    fn sum<I: Iterator<Item = &'a Pixels>>(iter: I) -> Self {
        iter.copied().sum()
    }
}

impl Window {
    /// Vrai quand un client d'accessibilité écoute. Toujours faux : WGPUI n'a
    /// pas de backend accesskit.
    pub fn is_a11y_active(&self) -> bool {
        false
    }

    /// Peint `data` mappée sur `image_bounds`, rognée au rectangle `clip`
    /// (via le masque de contenu). Les coins arrondis s'appliquent au quad
    /// image, comme dans [`Window::paint_image`].
    pub fn paint_image_cropped(
        &mut self,
        clip: Bounds<Pixels>,
        image_bounds: Bounds<Pixels>,
        corner_radii: crate::Corners<Pixels>,
        data: std::sync::Arc<crate::RenderImage>,
        frame_index: usize,
        grayscale: bool,
    ) -> anyhow::Result<()> {
        self.with_content_mask(Some(crate::ContentMask { bounds: clip }), |window| {
            window.paint_image(image_bounds, corner_radii, data, frame_index, grayscale)
        })
    }
}

/// Élément qui construit son contenu en fonction de sa taille résolue
/// (équivalent minimal des container queries CSS). L'élément remplit l'espace
/// disponible, puis `render` est appelé au prepaint avec cette taille.
pub fn container_query<R, F>(render: F) -> ContainerQuery<F>
where
    R: IntoElement,
    F: FnOnce(Size<Pixels>, &mut Window, &mut App) -> R + 'static,
{
    ContainerQuery {
        render: Some(render),
    }
}

/// Voir [`container_query`].
pub struct ContainerQuery<F> {
    render: Option<F>,
}

impl<R, F> IntoElement for ContainerQuery<F>
where
    R: IntoElement,
    F: FnOnce(Size<Pixels>, &mut Window, &mut App) -> R + 'static,
{
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl<R, F> Element for ContainerQuery<F>
where
    R: IntoElement,
    F: FnOnce(Size<Pixels>, &mut Window, &mut App) -> R + 'static,
{
    type RequestLayoutState = ();
    type PrepaintState = AnyElement;

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
        style.size.width = relative(1.).into();
        style.size.height = relative(1.).into();
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let render = self.render.take().expect("prepaint appelé deux fois");
        let mut element = render(bounds.size, window, cx).into_any_element();
        element.layout_as_root(bounds.size.map(AvailableSpace::Definite), window, cx);
        element.prepaint_at(bounds.origin, window, cx);
        element
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        element: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        element.paint(window, cx);
    }
}

#[derive(Default)]
struct ReduceMotion(bool);

impl Global for ReduceMotion {}

impl App {
    /// Préférence « animations réduites ». Faux par défaut : WGPUI n'interroge
    /// pas le réglage système, seul [`App::set_reduce_motion`] la change.
    pub fn reduce_motion(&self) -> bool {
        self.try_global::<ReduceMotion>().is_some_and(|g| g.0)
    }

    /// Fixe la préférence « animations réduites ».
    pub fn set_reduce_motion(&mut self, reduce_motion: bool) {
        self.set_global(ReduceMotion(reduce_motion));
    }
}
