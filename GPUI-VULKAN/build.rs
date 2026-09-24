//! GLSL → SPIR-V à la compilation : le binaire embarque les `.spv`, comme un moteur Vulkan.

use naga::{ShaderStage, back::spv, front::glsl, valid};

fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    for (name, stage) in [("cube.vert", ShaderStage::Vertex), ("cube.frag", ShaderStage::Fragment)] {
        let path = format!("src/{name}");
        println!("cargo:rerun-if-changed={path}");
        let source = std::fs::read_to_string(&path).unwrap();
        let module = glsl::Frontend::default()
            .parse(&glsl::Options::from(stage), &source)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let mut options = spv::Options::default();
        // Le flip Y Vulkan est écrit dans cube.vert : naga ne doit pas le refaire.
        options.flags.remove(spv::WriterFlags::ADJUST_COORDINATE_SPACE);
        let words = spv::write_vec(&module, &info, &options, None).unwrap();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(format!("{out}/{name}.spv"), bytes).unwrap();
    }
}
