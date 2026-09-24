#[cfg(windows)]
mod opengl_cube;

#[cfg(windows)]
fn main() {
    // GL rend, D3D12 publie : le device GPUI doit être D3D12 (wgpu-GL n'a pas ce que GPUI exige).
    gpui3d_shell::run::<opengl_cube::OpenGlCube>("OpenGL natif", gpui3d_shell::Backends::DX12);
}

#[cfg(not(windows))]
fn main() {
    eprintln!("GPUI-OPENGL requiert Windows (interop OpenGL ↔ D3D12).");
}
