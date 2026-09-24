#version 450
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 color;
layout(location = 0) out vec3 v_color;
layout(push_constant) uniform Push { mat4 mvp; } pc;

void main() {
    gl_Position = pc.mvp * vec4(pos, 1.0);
    // Clip Vulkan : Y vers le bas (la matrice partagée suit wgpu/Metal, Y vers le haut).
    gl_Position.y = -gl_Position.y;
    v_color = color;
}
