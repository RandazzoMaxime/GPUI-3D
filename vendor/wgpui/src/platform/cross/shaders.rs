//! Composition des shaders WGSL de l'UI, commune à tous les backends : wgpu les compile
//! à l'exécution, les backends natifs les traduisent hors ligne (naga, `build.rs`).
//! Autonome (aucun `crate::`) pour pouvoir être incluse telle quelle par un `build.rs`.

/// Shaders de l'UI : nom, groupe du transform de couche (`None` : shader non patché),
/// source brute.
pub(crate) const SHADERS: [(&str, Option<u32>, &str); 8] = [
    ("quads", Some(2), include_str!("shaders/quads.wgsl")),
    ("shadows", Some(2), include_str!("shaders/shadows.wgsl")),
    ("backdrop_blur", None, include_str!("shaders/backdrop_blur.wgsl")),
    ("underlines", Some(2), include_str!("shaders/underlines.wgsl")),
    ("mono_sprites", Some(4), include_str!("shaders/mono_sprites.wgsl")),
    ("poly_sprites", Some(3), include_str!("shaders/poly_sprites.wgsl")),
    ("surfaces", None, include_str!("shaders/surfaces.wgsl")),
    ("paths", Some(2), include_str!("shaders/paths.wgsl")),
];

/// Source WGSL finale du shader `name` (voir [`SHADERS`]).
pub(crate) fn wgsl_source(name: &str) -> String {
    let Some(&(_, group, body)) = SHADERS.iter().find(|(candidate, _, _)| *candidate == name) else {
        panic!("shader UI inconnu : {name}");
    };
    match group {
        Some(group) => slab_shader_source(name, group, body),
        None => body.to_owned(),
    }
}

/// Fragment-stage translate-undo edits, per shader: patterns that must occur
/// exactly once in that shader's body and get rewritten to route through
/// `layer_world_position`. Shaders absent from this list read no world-space
/// geometry in their fragment stages and must stay untouched.
const FRAGMENT_TRANSLATE_EDITS: &[(&str, &str, &str)] = &[
    (
        "quads",
        "gradient_color(quad.background, input.position.xy, quad.bounds,",
        "gradient_color(quad.background, layer_world_position(input.position.xy), quad.bounds,",
    ),
    (
        "quads",
        "let point = input.position.xy - quad.bounds.origin;",
        "let point = layer_world_position(input.position.xy) - quad.bounds.origin;",
    ),
    (
        "shadows",
        "let center_to_point = input.position.xy - center;",
        "let center_to_point = layer_world_position(input.position.xy) - center;",
    ),
    (
        "underlines",
        "let st = (input.position.xy - underline.bounds.origin)",
        "let st = (layer_world_position(input.position.xy) - underline.bounds.origin)",
    ),
    (
        "poly_sprites",
        "quad_sdf(input.position.xy, sprite.bounds, sprite.corner_radii)",
        "quad_sdf(layer_world_position(input.position.xy), sprite.bounds, sprite.corner_radii)",
    ),
];

/// Shaders whose vertex stage builds NDC positions through the shared
/// `to_device_position_impl` helper; `paths.wgsl` builds them inline instead.
const IMPL_VERTEX_SHADERS: &[&str] = &[
    "quads",
    "shadows",
    "underlines",
    "mono_sprites",
    "poly_sprites",
];

/// Shader source for a pipeline that can draw spliced layer-slab content:
/// the shared transform-uniform prelude ahead of the file's body, with exact
/// match-once edits threading the per-layer translate through the vertex
/// stage and undoing it in fragment stages that re-read world-space geometry.
///
/// The `.wgsl` files themselves stay byte-pristine: `flamegraph_replay`
/// renders them against its own bind-group layouts, so every slab-specific
/// reference must come from this composition step. Each edit asserts exactly
/// one match — a shader change that drifts past these patterns fails loudly
/// here instead of silently dropping the translate (or double-applying it).
pub(crate) fn slab_shader_source(name: &str, group: u32, body: &str) -> String {
    let mut source = include_str!("shaders/slab_transform.wgsl")
        .replace("{SLAB_TRANSFORM_GROUP}", &group.to_string());

    // Vertex stage: shift rasterized positions by the layer translate. Clip
    // distances are computed from untranslated bounds on purpose — they move
    // with the instance data, not the rasterized position.
    if IMPL_VERTEX_SHADERS.contains(&name) {
        const PATTERN: &str = "let device_position = position / globals.viewport_size";
        assert!(
            body.matches(PATTERN).count() == 1,
            "{name}: vertex-position pattern drifted"
        );
        source.push_str(&body.replacen(
            PATTERN,
            "let device_position = (position + layer_transform.translate) / globals.viewport_size",
            1,
        ));
    } else {
        const PATTERN: &str = "let device_pos = v.xy_position / globals.viewport_size";
        assert!(
            name == "paths" && body.matches(PATTERN).count() == 1,
            "{name}: no known vertex-position pattern; slab transform edits are stale"
        );
        source.push_str(&body.replacen(
            PATTERN,
            "let world_position = v.xy_position + layer_transform.translate;\n    let device_pos = world_position / globals.viewport_size",
            1,
        ));
    }

    for (shader, pattern, replacement) in FRAGMENT_TRANSLATE_EDITS {
        if *shader != name {
            continue;
        }
        assert_eq!(
            source.matches(pattern).count(),
            1,
            "{name}: fragment edit matched more than once: {pattern}"
        );
        source = source.replace(pattern, replacement);
    }

    source
}

