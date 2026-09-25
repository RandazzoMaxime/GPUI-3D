#[cfg(windows)]
mod opengl_cube;

#[cfg(windows)]
fn main() {
    gpui3d_shell::run::<opengl_cube::OpenGlCube>("OpenGL natif", gpui3d_shell::Ui::OpenGl);
}

#[cfg(not(windows))]
fn main() {
    eprintln!("GPUI-OPENGL requiert Windows (WGL).");
}
