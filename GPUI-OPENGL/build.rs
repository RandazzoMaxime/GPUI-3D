//! Bindings OpenGL 4.5 + interop mémoire/sémaphores externes, générés depuis gl.xml.

use gl_generator::{Api, Fallbacks, Profile, Registry, StructGenerator};

fn main() {
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("gl.rs");
    let extensions = ["GL_EXT_memory_object", "GL_EXT_memory_object_win32", "GL_EXT_semaphore", "GL_EXT_semaphore_win32"];
    Registry::new(Api::Gl, (4, 5), Profile::Core, Fallbacks::All, extensions)
        .write_bindings(StructGenerator, &mut std::fs::File::create(out).unwrap())
        .unwrap();
}
