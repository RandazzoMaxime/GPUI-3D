#![allow(clippy::disallowed_methods, reason = "build scripts are exempt")]

// GPUI-3D : composition des shaders WGSL de l'UI, partagée avec le runtime.
#[cfg(feature = "vulkan")]
#[path = "src/platform/cross/shaders.rs"]
#[allow(dead_code)]
mod shaders;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(gles)");
    #[cfg(feature = "vulkan")]
    compile_spirv();
}

/// GPUI-3D : WGSL → SPIR-V hors ligne pour le backend Vulkan natif, avec les réglages de
/// wgpu-hal (SPIR-V 1.6 pour Vulkan 1.3, bornes `Restrict`, pas d'ajustement de
/// coordonnées : le flip Y se fait par viewport négatif, comme wgpu).
#[cfg(feature = "vulkan")]
fn compile_spirv() {
    use naga::back::spv;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    println!("cargo::rerun-if-changed=src/platform/cross/shaders.rs");
    println!("cargo::rerun-if-changed=src/platform/cross/shaders");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    for (name, _, _) in shaders::SHADERS {
        let source = shaders::wgsl_source(name);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{name}.wgsl : {}", error.emit_to_string(&source)));
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name}.wgsl : {error:?}"));
        let options = spv::Options {
            lang_version: (1, 6),
            flags: spv::WriterFlags::LABEL_VARYINGS | spv::WriterFlags::FORCE_POINT_SIZE,
            bounds_check_policies: naga::proc::BoundsCheckPolicies {
                index: naga::proc::BoundsCheckPolicy::Restrict,
                buffer: naga::proc::BoundsCheckPolicy::Restrict,
                image_load: naga::proc::BoundsCheckPolicy::Restrict,
                binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
            },
            force_loop_bounding: true,
            ..spv::Options::default()
        };
        let words = spv::write_vec(&module, &info, &options, None)
            .unwrap_or_else(|error| panic!("{name}.wgsl → SPIR-V : {error:?}"));
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        std::fs::write(format!("{out}/{name}.spv"), bytes).expect("écriture SPIR-V");
    }
}
