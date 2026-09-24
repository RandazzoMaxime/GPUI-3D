struct Globals {
    viewport_size: vec2<f32>,
    premultiplied_alpha: u32,
    pad: u32,
}

struct Bounds {
    origin: vec2<f32>,
    size: vec2<f32>,
}

struct Corners {
    top_left: f32,
    top_right: f32,
    bottom_right: f32,
    bottom_left: f32,
}

struct SurfaceParams {
    bounds: Bounds,
    content_mask: Bounds,
    /// Silhouette arrondie, appliquee au rectangle du masque de contenu : une
    /// surface peut couvrir la fenetre et n'etre visible que dans un panneau
    /// aux coins arrondis (champ de verre). Zero = rectangle net.
    corner_radii: Corners,
}

// SDF de rectangle arrondi, identique a celle de backdrop_blur.wgsl.
fn quad_sdf(point: vec2<f32>, bounds: Bounds, corner_radii: Corners) -> f32 {
    let center = bounds.origin + bounds.size / 2.0;
    let half_size = bounds.size / 2.0;
    var radii_size = vec2<f32>(0.0);
    if point.x < center.x {
        if point.y < center.y {
            radii_size = vec2<f32>(corner_radii.top_left);
        } else {
            radii_size = vec2<f32>(corner_radii.bottom_left);
        }
    } else {
        if point.y < center.y {
            radii_size = vec2<f32>(corner_radii.top_right);
        } else {
            radii_size = vec2<f32>(corner_radii.bottom_right);
        }
    }
    let q = abs(point - center) - half_size + radii_size;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - radii_size.x;
}

struct SurfaceVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) clip_distances: vec4<f32>,
    @location(2) pixel_position: vec2<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> params: SurfaceParams;
@group(1) @binding(1) var t_surface: texture_2d<f32>;
@group(1) @binding(2) var s_surface: sampler;

fn to_device_position(position: vec2<f32>) -> vec4<f32> {
    let device_position = position / globals.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    return vec4<f32>(device_position, 0.0, 1.0);
}

@vertex
fn vs_surface(@builtin(vertex_index) vertex_id: u32) -> SurfaceVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    // Quad rogné au masque : un champ de verre couvre la fenêtre et n'est visible que dans son
    // panneau ; sans ce rognage chaque panneau ombrait la fenêtre entière (−16 % fps sur UHD 770).
    let lo = max(params.bounds.origin, params.content_mask.origin);
    let hi = max(lo, min(params.bounds.origin + params.bounds.size, params.content_mask.origin + params.content_mask.size));
    let position = mix(lo, hi, unit_vertex);

    let clip_origin = params.content_mask.origin;
    let clip_size = params.content_mask.size;
    let tl = position - clip_origin;
    let br = clip_origin + clip_size - position;

    var out: SurfaceVarying;
    out.position = to_device_position(position);
    out.tex_coord = (position - params.bounds.origin) / max(params.bounds.size, vec2<f32>(1e-6));
    out.clip_distances = vec4<f32>(tl.x, br.x, tl.y, br.y);
    out.pixel_position = position;
    return out;
}

// `t_surface` is sampled from an sRGB-format texture, so `textureSample` below
// auto-decodes sRGB -> linear. The swapchain is non-sRGB (see renderer.rs), so
// writing linear values keeps blending correct and lets the display handle the
// final gamma curve.
@fragment
fn fs_surface(input: SurfaceVarying) -> @location(0) vec4<f32> {
    let inside = !any(input.clip_distances < vec4<f32>(0.0));
    let color = textureSample(t_surface, s_surface, input.tex_coord);
    // Rayons nuls => couverture 1 partout dans le masque : le rendu est celui
    // d'avant, au bit pres, pour toute surface qui n'en demande pas.
    let corner = saturate(0.5 - quad_sdf(input.pixel_position, params.content_mask, params.corner_radii));
    let alpha = color.a * corner;
    let multiplier = select(1.0, alpha, globals.premultiplied_alpha != 0u);
    let result = vec4<f32>(color.rgb * multiplier, alpha);
    return select(vec4<f32>(0.0), result, inside);
}
