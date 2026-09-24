#version 450 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 color;
// MVP partagée (colonnes) ; glClipControl(UPPER_LEFT, ZERO_TO_ONE) = convention D3D/wgpu.
layout(location = 0) uniform mat4 mvp;
out vec3 v_color;

void main() {
    gl_Position = mvp * vec4(pos, 1.0);
    v_color = color;
}
