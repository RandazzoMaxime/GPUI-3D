struct Frame {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    time: f32,
}
@group(0) @binding(0) var<uniform> frame: Frame;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) glow: f32,
}

fn rot_y(a: f32) -> mat3x3<f32> {
    let c = cos(a);
    let s = sin(a);
    return mat3x3<f32>(vec3(c, 0.0, -s), vec3(0.0, 1.0, 0.0), vec3(s, 0.0, c));
}

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) position: vec3<f32>,
    @location(3) scale: vec3<f32>,
    @location(4) color: vec3<f32>,
    @location(5) flags: u32,
) -> VsOut {
    let spin = (flags & 1u) != 0u;
    let selected = (flags & 2u) != 0u;
    let r = rot_y(select(0.0, frame.time * 0.9, spin));
    let world = r * (pos * scale) + position;
    let glow = select(0.0, 0.25 + 0.25 * sin(frame.time * 5.0), selected);
    return VsOut(frame.view_proj * vec4(world, 1.0), r * normal, color, glow);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let light = max(dot(normalize(in.normal), -frame.light_dir.xyz), 0.0);
    let base = in.color * (0.3 + 0.7 * light);
    return vec4(mix(base, vec3(1.0, 0.85, 0.3), in.glow), 1.0);
}
