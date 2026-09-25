#[cfg(target_os = "macos")]
mod metal_cube;

#[cfg(target_os = "macos")]
fn main() {
    gpui3d_shell::run::<metal_cube::MetalCube>("Metal natif", gpui3d_shell::Ui::Metal);
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("GPUI-METAL requiert macOS : utiliser GPUI-WGPU ailleurs.");
}
