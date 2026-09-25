//! Bindings OpenGL 4.5 core, générés depuis gl.xml.

use gl_generator::{Api, Fallbacks, Profile, Registry, StructGenerator};

fn main() {
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("gl.rs");
    Registry::new(Api::Gl, (4, 5), Profile::Core, Fallbacks::All, [])
        .write_bindings(StructGenerator, &mut std::fs::File::create(out).unwrap())
        .unwrap();
}
