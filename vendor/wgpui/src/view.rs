use crate::{
    AnyElement, AnyEntity, AnyWeakEntity, App, Bounds, ContentMask, Context, Element, Point,
    ElementGeometry, ElementId, Entity, EntityId, GlobalElementId, InspectorElementId, IntoElement,
    LayerPolicy, LayoutId, PaintIndex, Pixels, PrepaintStateIndex, Render, Style, StyleRefinement,
    TextStyle, WeakEntity,
};
use crate::{Empty, Window};
use anyhow::Result;
use collections::FxHashSet;
use refineable::Refineable;
use std::rc::Rc;
use std::{any::TypeId, fmt, ops::Range};

struct AnyViewState {
    prepaint_range: Range<PrepaintStateIndex>,
    paint_range: Range<PaintIndex>,
    cache_key: ViewCacheKey,
    accessed_entities: FxHashSet<EntityId>,
    /// GPUI-3D : où le contenu enregistré est réellement peint : `cache_key.bounds.origin`
    /// calé au pixel physique (voir `snapped_origin`).
    drawn_origin: Point<Pixels>,
    /// GPUI-3D : emprise absolue de tout ce que la vue émet (primitives, y compris celles
    /// que le masque a éliminées, et hitboxes). `None` : rien d'émis.
    extent: Option<Bounds<Pixels>>,
    /// GPUI-3D : zone, en coordonnées locales à la vue, où la copie enregistrée est
    /// exacte (hors de là, le masque d'origine a pu couper ou éliminer du contenu).
    valid_local: Bounds<Pixels>,
    /// GPUI-3D : emprise des hitboxes d'une reconstruction, en attente du paint.
    prepaint_extent: Option<Bounds<Pixels>>,
    /// GPUI-3D : translation décidée au prepaint, appliquée au paint.
    translation: Option<(Point<Pixels>, Bounds<Pixels>)>,
}

/// GPUI-3D : désactivable par `GPUI_VIEW_TRANSLATE=0` pour comparer.
fn translated_reuse_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("GPUI_VIEW_TRANSLATE").map_or(true, |v| v != "0")
    });
    *ENABLED
}

/// GPUI-3D : `GPUI_VIEW_TRANSLATE=rebuild` garde le calage au pixel mais refuse tout rejeu
/// translaté : la référence « rendu neuf » des tests de parité pixel.
fn translated_replay_allowed() -> bool {
    static ALLOWED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("GPUI_VIEW_TRANSLATE").map_or(true, |v| v != "0" && v != "rebuild")
    });
    *ALLOWED
}

/// GPUI-3D : origine de peinture d'une vue en cache, calée au pixel physique. Une vue
/// reconstruite et une vue rejouée translatée se peignent ainsi au même endroit : tout
/// décalage entre deux trames est un nombre entier de pixels physiques, que les glyphes
/// (rastérisés calés au pixel) supportent sans écart. Absorbe aussi le bruit f32 de la
/// mise en page (une ligne mesurée 63,99997 px au lieu de 64).
fn snapped_origin(origin: Point<Pixels>, scale_factor: f32) -> Point<Pixels> {
    let snap = |v: Pixels| Pixels((v.0 * scale_factor).round() / scale_factor);
    Point { x: snap(origin.x), y: snap(origin.y) }
}

/// GPUI-3D : décalage (arrondi au pixel physique) qui rejoue la vue à `bounds`, ou la
/// raison pour laquelle le rejeu ne serait pas identique à un rendu neuf. Voir
/// `AnyViewState::valid_local`.
fn exact_translation(
    state: &AnyViewState,
    bounds: Bounds<Pixels>,
    content_mask: &ContentMask<Pixels>,
    mouse: Point<Pixels>,
    scale_factor: f32,
) -> Result<Point<Pixels>, &'static str> {
    if state.cache_key.bounds.size != bounds.size {
        return Err("view cache: translate refused (size)");
    }
    let delta = snapped_origin(bounds.origin, scale_factor) - state.drawn_origin;
    let Some(extent) = state.extent else {
        return Ok(delta);
    };
    let moved = Bounds { origin: extent.origin + delta, size: extent.size };
    // Le survol dépend du pointeur, pas d'une entité : une vue sous la souris se refait.
    if extent.contains(&mouse) || moved.contains(&mouse) {
        return Err("view cache: translate refused (pointer)");
    }
    let new_origin = state.drawn_origin + delta;
    let visible_local = Bounds {
        origin: content_mask.bounds.origin - new_origin,
        size: content_mask.bounds.size,
    };
    let extent_local = Bounds { origin: extent.origin - state.drawn_origin, size: extent.size };
    let required = extent_local.intersect(&visible_local);
    if required.is_empty() || covers(&state.valid_local, &required) {
        Ok(delta)
    } else {
        Err("view cache: translate refused (clipped at record)")
    }
}

fn covers(outer: &Bounds<Pixels>, inner: &Bounds<Pixels>) -> bool {
    inner.origin.x >= outer.origin.x
        && inner.origin.y >= outer.origin.y
        && inner.origin.x + inner.size.width <= outer.origin.x + outer.size.width
        && inner.origin.y + inner.size.height <= outer.origin.y + outer.size.height
}

#[derive(Default)]
struct ViewCacheKey {
    bounds: Bounds<Pixels>,
    content_mask: ContentMask<Pixels>,
    text_style: TextStyle,
}

impl<V: Render> Element for Entity<V> {
    type RequestLayoutState = AnyElement;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        Some(ElementId::View(self.entity_id()))
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut element = self.update(cx, |view, cx| view.render(window, cx).into_any_element());
        let layout_id = window.with_rendered_view(self.entity_id(), |window| {
            element.request_layout(window, cx)
        });
        (layout_id, element)
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        element: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) {
        window.set_view_id(self.entity_id());
        window.with_rendered_view(self.entity_id(), |window| element.prepaint(window, cx));
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        element: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        window.with_rendered_view(self.entity_id(), |window| element.paint(window, cx));
    }
}

/// Whether a cached view rebuilding is allowed to leave the cached views nested
/// inside it reusing, rather than forcing the whole subtree to rebuild.
///
/// This is what makes "no state change = no op" actually hold for nested
/// panels. Upstream forces the whole nested subtree to rebuild whenever any
/// ancestor cached view rebuilds, which in the level editor meant one genuinely
/// dirty view produced five collateral cache misses — measured at roughly 600ms
/// of every 890ms spent on the UI thread. With this on, a reuse costs ~0.003ms
/// against ~0.64ms for the rebuild it replaces.
///
/// Set `WGPUI_NESTED_VIEW_CACHE=0` to fall back to upstream behaviour.
///
/// **Why the escape hatch exists.** Enabling this repeatedly aborted the process
/// in `LineLayoutCache::reuse_layouts` — a stored reuse range outliving the
/// array it indexes. The ranges in `PrepaintStateIndex`/`PaintIndex` are
/// absolute offsets into per-frame arrays with nothing tying them to the array
/// they were recorded against. `Window::invalid_reuse_range` now bounds-checks
/// every one of them before a single byte is copied and treats a bad range as a
/// cache miss, so the failure mode is a slower frame rather than a crash. That
/// guard is what makes this safe to leave on; if something still goes wrong, the
/// env var reverts the behaviour without a rebuild.
///
/// Read once, at first use.
fn nested_view_cache_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("WGPUI_NESTED_VIEW_CACHE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true)
    });
    *ENABLED
}

/// Report the first cached-view reuse whose stored range had outlived the array
/// it indexes, then stay quiet.
///
/// One line is enough to identify the cause — which array, how far past the end,
/// and which view — without a per-frame flood. Everything after the first is
/// silently handled as a cache miss.
fn log_stale_reuse_range(entity: EntityId, array: &'static str, end: usize, len: usize) {
    // Counted on every occurrence, logged only on the first: the count is what
    // tells you whether this is a one-off at startup or happening continuously,
    // so it must not sit behind the log-once guard.
    crate::render_stats::count("view cache: stale range (rebuilt)");

    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    log::warn!(
        "[VIEW CACHE] stale reuse range discarded: entity={:?} array={} stored_end={} actual_len={}. \
         Rebuilding instead of replaying. This is the condition that used to panic in \
         `reuse_layouts`; further occurrences are handled silently.",
        entity,
        array,
        end,
        len,
    );
}

/// A dynamically-typed handle to a view, which can be downcast to a [Entity] for a specific type.
#[derive(Clone, Debug)]
pub struct AnyView {
    entity: AnyEntity,
    render: fn(&AnyView, &mut Window, &mut App) -> AnyElement,
    cached_style: Option<Rc<StyleRefinement>>,
}

impl<V: Render> From<Entity<V>> for AnyView {
    fn from(value: Entity<V>) -> Self {
        AnyView {
            entity: value.into_any(),
            render: any_view::render::<V>,
            cached_style: None,
        }
    }
}

impl AnyView {
    /// Indicate that this view should be cached when using it as an element.
    /// When using this method, the view's previous layout and paint will be recycled from the previous frame if [Context::notify] has not been called since it was rendered.
    /// The one exception is when [Window::refresh] is called, in which case caching is ignored.
    pub fn cached(mut self, style: StyleRefinement) -> Self {
        self.cached_style = Some(style.into());
        self
    }

    /// Convert this to a weak handle.
    pub fn downgrade(&self) -> AnyWeakView {
        AnyWeakView {
            entity: self.entity.downgrade(),
            render: self.render,
        }
    }

    /// Convert this to a [Entity] of a specific type.
    /// If this handle does not contain a view of the specified type, returns itself in an `Err` variant.
    pub fn downcast<T: 'static>(self) -> Result<Entity<T>, Self> {
        match self.entity.downcast() {
            Ok(entity) => Ok(entity),
            Err(entity) => Err(Self {
                entity,
                render: self.render,
                cached_style: self.cached_style,
            }),
        }
    }

    /// Gets the [TypeId] of the underlying view.
    pub fn entity_type(&self) -> TypeId {
        self.entity.entity_type
    }

    /// Gets the entity id of this handle.
    pub fn entity_id(&self) -> EntityId {
        self.entity.entity_id()
    }

    /// # Safety
    /// The caller must ensure the underlying entity is of type T.
    pub unsafe fn downgrade_unchecked<T: 'static>(&self) -> WeakEntity<T> {
        WeakEntity::from_raw(self.entity.downgrade())
    }
}

impl PartialEq for AnyView {
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity
    }
}

impl Eq for AnyView {}

impl Element for AnyView {
    type RequestLayoutState = Option<AnyElement>;
    type PrepaintState = Option<AnyElement>;

    fn id(&self) -> Option<ElementId> {
        Some(ElementId::View(self.entity_id()))
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
        window.with_rendered_view(self.entity_id(), |window| {
            // Disable caching when inspecting so that mouse_hit_test has all hitboxes.
            let caching_disabled = window.is_inspector_picking(cx);
            match self.cached_style.as_ref() {
                Some(style) if !caching_disabled => {
                    let mut root_style = Style::default();
                    root_style.refine(style);
                    let layout_id = window.request_layout(root_style, None, cx);
                    (layout_id, None)
                }
                _ => {
                    let mut element = {
                        let _t = crate::render_stats::scope("frame: render");
                        (self.render)(self, window, cx)
                    };
                    let layout_id = element.request_layout(window, cx);
                    (layout_id, Some(element))
                }
            }
        })
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        element: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<AnyElement> {
        window.set_view_id(self.entity_id());
        window.with_rendered_view(self.entity_id(), |window| {
            if let Some(mut element) = element.take() {
                element.prepaint(window, cx);
                return Some(element);
            }

            window.with_element_state::<AnyViewState, _>(
                global_id.unwrap(),
                |element_state, window| {
                    let content_mask = window.content_mask();
                    let text_style = window.text_style();

                    // Stored ranges are absolute offsets into per-frame arrays.
                    // Verify they still fit before anything is copied — a stale
                    // range would otherwise slice out of bounds and abort the
                    // process. Both ranges are checked here, at prepaint, because
                    // once prepaint commits to reusing, paint has no rebuild path
                    // left. A failure is just a cache miss: we fall through and
                    // rebuild, costing a frame's work rather than the process.
                    let stale_range = element_state.as_ref().and_then(|state| {
                        window.invalid_reuse_range(&state.prepaint_range, &state.paint_range)
                    });
                    if let Some((array, end, len)) = stale_range {
                        log_stale_reuse_range(self.entity_id(), array, end, len);
                    }

                    // `dirty_views` alone is not enough. It is built by walking
                    // the dispatch tree upward from each notified entity, so it
                    // only ever contains entities that own a dispatch node —
                    // views that were prepainted. Notifying anything else (a
                    // model, or any entity a view merely reads) marks nothing,
                    // and every cached view rendering that entity's data judges
                    // itself clean and replays, indefinitely. That is issue #83.
                    //
                    // So also ask whether anything in this view's *own* recorded
                    // dependency set changed. Nesting is covered for free: the
                    // set is cumulative over the whole subtree, so every cached
                    // layer above the view that actually reads the entity fails
                    // this test too and rebuilds, which is what gets the inner
                    // one prepainted at all.
                    let dependency_invalidated = element_state.as_ref().is_some_and(|state| {
                        window.accessed_entity_invalidated(&state.accessed_entities)
                    });

                    let miss_reason = match &element_state {
                        None => "view cache: miss (no state)",
                        Some(_) if stale_range.is_some() => "view cache: miss (stale range)",
                        Some(s) if s.cache_key.bounds != bounds => "view cache: miss (bounds)",
                        Some(s) if s.cache_key.content_mask != content_mask => {
                            "view cache: miss (mask)"
                        }
                        Some(s) if s.cache_key.text_style != text_style => {
                            "view cache: miss (text style)"
                        }
                        Some(_) if window.dirty_views.contains(&self.entity_id()) => {
                            "view cache: miss (dirty view)"
                        }
                        Some(_) if dependency_invalidated => "view cache: miss (dependency)",
                        Some(_) => "view cache: miss (unavailable)",
                    };
                    profiling::scope!(miss_reason);

                    if element_state.as_ref().is_some_and(|element_state| {
                        stale_range.is_none()
                            && element_state.cache_key.bounds == bounds
                            && element_state.cache_key.content_mask == content_mask
                            && element_state.cache_key.text_style == text_style
                    }) && !window.dirty_views.contains(&self.entity_id())
                        && !dependency_invalidated
                        && window.view_cache_available()
                        && let Some(mut element_state) = element_state
                    {
                        crate::render_stats::count("view cache: reused");
                        let _t = crate::render_stats::scope("view cache: reuse_prepaint");
                        let prepaint_start = window.prepaint_index();
                        window.reuse_prepaint(element_state.prepaint_range.clone());
                        cx.entities
                            .extend_accessed(&element_state.accessed_entities);

                        // `on_frame` effects still run on a cache hit — that is
                        // the whole point of the channel: side effects that must
                        // fire every frame regardless of caching. They are
                        // replayed from what the subtree recorded when it last
                        // rendered, *not* by rebuilding the subtree to find
                        // them again. Rebuilding would run `render` and a full
                        // `layout_as_root` on every reuse, which is most of
                        // what the cache is here to skip.
                        window.replay_frame_effects(&element_state.prepaint_range, cx);

                        let prepaint_end = window.prepaint_index();
                        element_state.prepaint_range = prepaint_start..prepaint_end;
                        element_state.translation = None;

                        return (None, element_state);
                    }

                    // GPUI-3D : même contenu, autre position ou autre découpe (défilement,
                    // panneau voisin redimensionné) → rejeu translaté au lieu d'un rebuild.
                    if translated_replay_allowed()
                        && stale_range.is_none()
                        && !window.dirty_views.contains(&self.entity_id())
                        && !dependency_invalidated
                        && window.view_cache_available()
                        && let Some(delta) = element_state.as_ref().and_then(|element_state| {
                            (element_state.cache_key.text_style == text_style
                                && window.translation_supported(
                                    &element_state.prepaint_range,
                                    &element_state.paint_range,
                                ))
                            .then(|| {
                                exact_translation(
                                    element_state,
                                    bounds,
                                    &content_mask,
                                    window.mouse_position(),
                                    window.scale_factor(),
                                )
                                .map_err(crate::render_stats::count)
                                .ok()
                            })
                            .flatten()
                        })
                        && let Some(mut element_state) = element_state
                    {
                        crate::render_stats::count("view cache: reused (translated)");
                        let clip = content_mask.bounds;
                        let prepaint_start = window.prepaint_index();
                        window.reuse_prepaint_translated(
                            element_state.prepaint_range.clone(),
                            delta,
                            clip,
                            cx,
                        );
                        cx.entities.extend_accessed(&element_state.accessed_entities);
                        element_state.prepaint_range = prepaint_start..window.prepaint_index();
                        element_state.drawn_origin += delta;
                        let visible_local = Bounds {
                            origin: clip.origin - element_state.drawn_origin,
                            size: clip.size,
                        };
                        element_state.valid_local = element_state.valid_local.intersect(&visible_local);
                        element_state.extent = element_state
                            .extent
                            .map(|e| Bounds { origin: e.origin + delta, size: e.size });
                        element_state.cache_key.bounds = bounds;
                        element_state.cache_key.content_mask = content_mask;
                        element_state.translation = Some((delta, clip));
                        return (None, element_state);
                    }

                    // Cache miss. If this fires every frame for a view whose
                    // content is static, something is calling `cx.notify()` on
                    // it or on one of its descendants — `mark_view_dirty` walks
                    // the ancestor path, so a chatty leaf invalidates every
                    // cached view above it.
                    crate::render_stats::count("view cache: rebuilt");
                    crate::render_stats::count(miss_reason);
                    if dependency_invalidated {
                        // Counted separately because this is the class of
                        // rebuild the #83 fix added. If it dominates, some
                        // entity read across a whole subtree is being notified
                        // every frame, and the fix to make is at that call site
                        // rather than here.
                        crate::render_stats::count("view cache: rebuilt (dependency changed)");
                    }
                    let _t = crate::render_stats::scope("view cache: rebuild");

                    // Rebuilding this view normally forces every cached view
                    // nested inside it to rebuild too. See
                    // `nested_view_cache_enabled` for why, and for the opt-in
                    // that lifts it.
                    let nested_cache_suppressed = window.nested_view_cache_suppressed;
                    if !nested_view_cache_enabled() {
                        window.nested_view_cache_suppressed = true;
                    }

                    let prepaint_start = window.prepaint_index();
                    let paint_origin = if translated_reuse_enabled() {
                        snapped_origin(bounds.origin, window.scale_factor())
                    } else {
                        bounds.origin
                    };
                    window.hitbox_extent_stack.push(None);
                    let (mut element, accessed_entities) = cx.detect_accessed_entities(|cx| {
                        // Split three ways: building the element tree is usually
                        // trivial next to laying it out and prepainting it, and
                        // conflating them hides which one to go after.
                        let mut element = {
                            let _t = crate::render_stats::scope("  rebuild: render");
                            // Also counted into the whole-frame bucket. A cached
                            // view renders from prepaint rather than from
                            // request_layout, so this is the one place where
                            // `frame: render` nests under `frame: prepaint`.
                            let _frame_render = crate::render_stats::scope("frame: render");
                            (self.render)(self, window, cx)
                        };
                        {
                            let _t = crate::render_stats::scope("  rebuild: layout");
                            element.layout_as_root(bounds.size.into(), window, cx);
                        }
                        {
                            let _t = crate::render_stats::scope("  rebuild: prepaint");
                            element.prepaint_at(paint_origin, window, cx);
                        }
                        element
                    });

                    let prepaint_end = window.prepaint_index();
                    window.nested_view_cache_suppressed = nested_cache_suppressed;
                    let prepaint_extent = window.hitbox_extent_stack.pop().flatten();
                    if let (Some(extent), Some(Some(parent))) =
                        (prepaint_extent, window.hitbox_extent_stack.last_mut().map(|p| p.as_mut()))
                    {
                        *parent = parent.union(&extent);
                    } else if let (Some(extent), Some(parent)) =
                        (prepaint_extent, window.hitbox_extent_stack.last_mut())
                    {
                        *parent = Some(extent);
                    }
                    let valid_local = Bounds {
                        origin: content_mask.bounds.origin - paint_origin,
                        size: content_mask.bounds.size,
                    };

                    (
                        Some(element),
                        AnyViewState {
                            accessed_entities,
                            prepaint_range: prepaint_start..prepaint_end,
                            paint_range: PaintIndex::default()..PaintIndex::default(),
                            cache_key: ViewCacheKey {
                                bounds,
                                content_mask,
                                text_style,
                            },
                            drawn_origin: paint_origin,
                            extent: None,
                            valid_local,
                            prepaint_extent,
                            translation: None,
                        },
                    )
                },
            )
        })
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        element: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        window.with_rendered_view(self.entity_id(), |window| {
            let caching_disabled = window.is_inspector_picking(cx);
            if self.cached_style.is_some() && !caching_disabled {
                let global_id = global_id.unwrap();
                // `cached` is a layer with a compat policy: every axis
                // invalidated together, primitive-retained. The decision about
                // *whether* to reuse was made in prepaint by `AnyViewState`,
                // which predates layers and reaches things layers cannot see
                // yet (recorded bounds, text style, the dispatch subtree). The
                // layer supplies only the retention — which is where the win
                // is, because it replaces re-inserting every primitive into a
                // `BoundsTree` with re-emitting orders that are already right.
                let (layer_key, layer_cache_key) = window.layer_identity(global_id, bounds);
                let layers_enabled = crate::layer::layers_enabled();

                window.with_element_state::<AnyViewState, _>(global_id, |element_state, window| {
                    let mut element_state = element_state.unwrap();

                    let paint_start = window.paint_index();

                    if let Some(element) = element {
                        // Paired with the prepaint path above.
                        let nested_cache_suppressed = window.nested_view_cache_suppressed;
                        if !nested_view_cache_enabled() {
                            window.nested_view_cache_suppressed = true;
                        }
                        window.next_frame.scene.begin_extent();
                        if layers_enabled {
                            window.record_layer(
                                layer_key,
                                layer_cache_key,
                                LayerPolicy::compat(),
                                |window| element.paint(window, cx),
                            );
                        } else {
                            element.paint(window, cx);
                        }
                        let scale = window.scale_factor();
                        let scene_extent = window.next_frame.scene.end_extent().map(|e| {
                            Bounds::new(
                                crate::point(crate::px(e.origin.x.0 / scale), crate::px(e.origin.y.0 / scale)),
                                crate::size(crate::px(e.size.width.0 / scale), crate::px(e.size.height.0 / scale)),
                            )
                        });
                        element_state.extent = match (element_state.prepaint_extent.take(), scene_extent) {
                            (Some(a), Some(b)) => Some(a.union(&b)),
                            (a, b) => a.or(b),
                        };
                        window.nested_view_cache_suppressed = nested_cache_suppressed;
                    } else if let Some((delta, clip)) = element_state.translation.take() {
                        window.reuse_paint_translated(&element_state.paint_range, delta, clip);
                        if let Some(extent) = element_state.extent {
                            let scale = window.scale_factor();
                            window.next_frame.scene.union_extent(extent.scale(scale));
                        }
                    } else {
                        if let Some(extent) = element_state.extent {
                            let scale = window.scale_factor();
                            window.next_frame.scene.union_extent(extent.scale(scale));
                        }
                        window.reuse_paint_except_scene(&element_state.paint_range);
                        // The layer can be gone even though prepaint committed
                        // to reusing — eviction is driven by draw age, and this
                        // view's element state outlives it. Falling back to the
                        // recorded scene range keeps that a slower frame rather
                        // than a missing panel.
                        if !window.try_composite_layer(layer_key) {
                            window.replay_scene_range(&element_state.paint_range);
                        }
                    }

                    let paint_end = window.paint_index();
                    element_state.paint_range = paint_start..paint_end;

                    ((), element_state)
                })
            } else {
                element.as_mut().unwrap().paint(window, cx);
            }
        });
    }
}

impl<V: 'static + Render> IntoElement for Entity<V> {
    type Element = Entity<V>;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl IntoElement for AnyView {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// A weak, dynamically-typed view handle that does not prevent the view from being released.
pub struct AnyWeakView {
    entity: AnyWeakEntity,
    render: fn(&AnyView, &mut Window, &mut App) -> AnyElement,
}

impl AnyWeakView {
    /// Convert to a strongly-typed handle if the referenced view has not yet been released.
    pub fn upgrade(&self) -> Option<AnyView> {
        let entity = self.entity.upgrade()?;
        Some(AnyView {
            entity,
            render: self.render,
            cached_style: None,
        })
    }
}

impl<V: 'static + Render> From<WeakEntity<V>> for AnyWeakView {
    fn from(view: WeakEntity<V>) -> Self {
        AnyWeakView {
            entity: view.into(),
            render: any_view::render::<V>,
        }
    }
}

impl PartialEq for AnyWeakView {
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity
    }
}

impl std::fmt::Debug for AnyWeakView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyWeakView")
            .field("entity_id", &self.entity.entity_id)
            .finish_non_exhaustive()
    }
}

mod any_view {
    use crate::{AnyElement, AnyView, App, IntoElement, Render, Window};

    pub(crate) fn render<V: 'static + Render>(
        view: &AnyView,
        window: &mut Window,
        cx: &mut App,
    ) -> AnyElement {
        profiling::scope!(std::any::type_name::<V>());
        let view = view.clone().downcast::<V>().unwrap();
        view.update(cx, |view, cx| view.render(window, cx).into_any_element())
    }
}

/// A view that renders nothing
pub struct EmptyView;

impl Render for EmptyView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        Empty
    }
}

/// GPUI-3D : rejeu translaté des vues en cache (défilement, voisin redimensionné).
#[cfg(test)]
mod translated_reuse_tests {
    use crate::{
        AnyView, AppContext as _, Context, DispatchPhase, Entity, IntoElement, Modifiers, MouseDownEvent,
        ParentElement as _, Pixels, Render, StyleRefinement, Styled as _, TestAppContext,
        Window, canvas, div, point, px, rgb,
    };
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    const ROW: f32 = 20.;
    const LIST_TOP: f32 = 100.;
    const VIEWPORT: f32 = 100.;

    struct Row {
        index: usize,
        renders: Rc<Cell<usize>>,
        hits: Rc<RefCell<Vec<usize>>>,
    }

    impl Render for Row {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            let (index, hits) = (self.index, self.hits.clone());
            // Le listener capture des bornes absolues, comme la plupart des éléments :
            // c'est exactement ce qu'un rejeu translaté doit préserver.
            div()
                .size_full()
                .bg(rgb(0x100000 * (index as u32 + 1)))
                .child(
                    canvas(
                        |_, _, _| {},
                        move |bounds, _, window, _| {
                            let hits = hits.clone();
                            window.on_mouse_event(move |event: &MouseDownEvent, phase, _, _| {
                                if phase == DispatchPhase::Bubble && bounds.contains(&event.position) {
                                    hits.borrow_mut().push(index);
                                }
                            });
                        },
                    )
                    .size_full(),
                )
        }
    }

    struct List {
        rows: Vec<Entity<Row>>,
        scroll: Pixels,
    }

    impl Render for List {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let row_style = || StyleRefinement::default().w(px(200.)).h(px(ROW));
            div().size_full().child(
                div()
                    .absolute()
                    .top(px(LIST_TOP))
                    .left(px(300.))
                    .w(px(200.))
                    .h(px(VIEWPORT))
                    .overflow_hidden()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .mt(self.scroll)
                            .children(self.rows.iter().map(|row| {
                                AnyView::from(row.clone()).cached(row_style()).into_any_element()
                            })),
                    ),
            )
        }
    }

    type Setup = (Entity<List>, Vec<Rc<Cell<usize>>>, Rc<RefCell<Vec<usize>>>);

    fn setup(cx: &mut TestAppContext) -> (Setup, &mut crate::VisualTestContext) {
        let hits = Rc::new(RefCell::new(Vec::new()));
        let renders: Vec<_> = (0..8).map(|_| Rc::new(Cell::new(0))).collect();
        let (hits_view, renders_view) = (hits.clone(), renders.clone());
        let (list, cx) = cx.add_window_view(move |_, cx| List {
            rows: (0..8)
                .map(|index| {
                    let (renders, hits) = (renders_view[index].clone(), hits_view.clone());
                    cx.new(|_| Row { index, renders, hits })
                })
                .collect(),
            scroll: px(0.),
        });
        ((list, renders, hits), cx)
    }

    fn scroll_to(list: &Entity<List>, y: f32, cx: &mut crate::VisualTestContext) {
        list.update(cx, |list, cx| {
            list.scroll = px(y);
            cx.notify();
        });
        cx.run_until_parked();
    }

    /// Ce que le GPU dessinera : chaque quad, sa couleur et la partie visible de ses
    /// bornes (le shader ne découpe que `bounds ∩ masque`), sans les ordres de tri.
    fn scene_quads(cx: &mut crate::VisualTestContext) -> Vec<String> {
        let mut quads: Vec<String> = cx.update(|window, _| {
            window
                .rendered_frame
                .scene
                .quads
                .iter()
                .map(|q| {
                    let visible = q.bounds.intersect(&q.content_mask.bounds);
                    format!("{:?} visible={:?} {:?}", q.bounds, visible, q.background)
                })
                .collect()
        });
        quads.sort();
        quads
    }

    fn rebuild_rows(list: &Entity<List>, cx: &mut crate::VisualTestContext) {
        let rows = list.read_with(cx, |list, _| list.rows.clone());
        for row in rows {
            row.update(cx, |_, cx| cx.notify());
        }
        cx.run_until_parked();
    }

    #[crate::test]
    fn moved_rows_replay_without_render_and_receive_clicks_where_they_are(cx: &mut TestAppContext) {
        let ((list, renders, hits), cx) = setup(cx);
        assert!(renders.iter().all(|r| r.get() == 1), "premier rendu");

        scroll_to(&list, 40., cx);
        assert!(
            renders.iter().all(|r| r.get() == 1),
            "déplacées, les lignes doivent être rejouées, pas re-rendues: {:?}",
            renders.iter().map(|r| r.get()).collect::<Vec<_>>()
        );

        // Ligne 1 : y = LIST_TOP + 40 + 1 * ROW = 160..180. Avant le défilement, ce point
        // appartenait à la ligne 3 ; un listener non translaté répondrait « 3 ».
        cx.simulate_click(point(px(310.), px(LIST_TOP + 40. + ROW + 5.)), Modifiers::none());
        assert_eq!(*hits.borrow(), vec![1]);
    }

    #[crate::test]
    fn translated_replay_draws_what_a_fresh_render_draws(cx: &mut TestAppContext) {
        let ((list, renders, _), cx) = setup(cx);

        // 50 px : la ligne 2 (90..110 local) est coupée par la fenêtre de 100 px.
        let snapshot = |renders: &[Rc<Cell<usize>>]| renders.iter().map(|r| r.get()).collect::<Vec<_>>();
        for y in [50., 13., -30.] {
            let before = snapshot(&renders);
            scroll_to(&list, y, cx);
            let after = snapshot(&renders);
            assert!(
                before.iter().zip(&after).any(|(b, a)| b == a),
                "à {y} px, aucune ligne n'a été rejouée : le test ne vérifierait rien"
            );
            let replayed = scene_quads(cx);
            rebuild_rows(&list, cx);
            let fresh = scene_quads(cx);
            assert_eq!(replayed, fresh, "défilement à {y} px");
        }
        // Retour en arrière : les lignes coupées au dernier rejeu n'ont plus leur contenu
        // complet et doivent se reconstruire, pas être rejouées incomplètes.
        scroll_to(&list, 50., cx);
        let replayed = scene_quads(cx);
        rebuild_rows(&list, cx);
        assert_eq!(replayed, scene_quads(cx), "retour en arrière");
        assert!(renders.iter().any(|r| r.get() > 1));
    }
}
