#![allow(clippy::disallowed_methods, reason = "build scripts are exempt")]

// GPUI-3D : composition des shaders WGSL de l'UI, partagée avec le runtime.
#[cfg(any(feature = "vulkan", feature = "dx12", feature = "opengl"))]
#[path = "src/platform/cross/shaders.rs"]
#[allow(dead_code)]
mod shaders;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(gles)");
    #[cfg(feature = "vulkan")]
    compile_spirv();
    #[cfg(feature = "dx12")]
    compile_dxbc();
    #[cfg(feature = "opengl")]
    compile_glsl();
}

/// GPUI-3D : WGSL → GLSL 4.50 (naga) hors ligne pour le backend OpenGL natif, un source
/// par point d'entrée (compilé par le pilote au démarrage). Convention de wgpu-GL : Y
/// retourné dans le vertex shader, ligne 0 des textures en haut. Liaisons fixes :
/// `@group(g) @binding(b)` → point de liaison `g * BINDING_STRIDE + b` de sa classe.
#[cfg(feature = "opengl")]
fn compile_glsl() {
    use std::fmt::Write as _;

    use naga::back::glsl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    const BINDING_STRIDE: u32 = 4;

    println!("cargo::rerun-if-changed=src/platform/cross/shaders.rs");
    println!("cargo::rerun-if-changed=src/platform/cross/shaders");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let mut table = String::new();
    writeln!(table, "pub(crate) const BINDING_STRIDE: u32 = {BINDING_STRIDE};").ok();
    table.push_str("pub(crate) fn glsl(shader: &str, entry: &str) -> Option<&'static str> {\n    match (shader, entry) {\n");
    for (name, _, _) in shaders::SHADERS {
        let source = shaders::wgsl_source(name);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{name}.wgsl : {}", error.emit_to_string(&source)));
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name}.wgsl : {error:?}"));
        let mut binding_map = glsl::BindingMap::default();
        for (_, global) in module.global_variables.iter() {
            let Some(binding) = &global.binding else { continue };
            let slot = binding.group * BINDING_STRIDE + binding.binding;
            binding_map.insert(binding.clone(), u8::try_from(slot).expect("point de liaison GL"));
        }
        let options = glsl::Options {
            version: glsl::Version::Desktop(450),
            writer_flags: glsl::WriterFlags::ADJUST_COORDINATE_SPACE,
            binding_map,
            zero_initialize_workgroup_memory: false,
        };
        for entry_point in &module.entry_points {
            let pipeline_options = glsl::PipelineOptions {
                shader_stage: entry_point.stage,
                entry_point: entry_point.name.clone(),
                multiview: None,
            };
            let mut glsl_source = String::new();
            glsl::Writer::new(
                &mut glsl_source,
                &module,
                &info,
                &options,
                &pipeline_options,
                naga::proc::BoundsCheckPolicies::default(),
            )
            .and_then(|mut writer| writer.write())
            .unwrap_or_else(|error| panic!("{name}::{} → GLSL : {error:?}", entry_point.name));
            let file = format!("{name}.{}.glsl", entry_point.name);
            std::fs::write(format!("{out}/{file}"), glsl_source).expect("écriture GLSL");
            writeln!(
                table,
                "        ({name:?}, {:?}) => Some(include_str!(concat!(env!(\"OUT_DIR\"), {:?}))),",
                entry_point.name,
                format!("/{file}"),
            )
            .ok();
        }
    }
    table.push_str("        _ => None,\n    }\n}\n");
    std::fs::write(format!("{out}/glsl.rs"), table).expect("écriture glsl.rs");
}

/// GPUI-3D : WGSL → HLSL (naga, SM 5.1) → DXBC (FXC) hors ligne pour le backend D3D12
/// natif. Liaisons fixes, connues du runtime par les constantes du fichier généré :
/// `@group(g) @binding(b)` → espace `g`, registre `b` (b/t selon la classe) ; les
/// samplers passent par le tas de naga, indexé par un tampon par groupe.
#[cfg(feature = "dx12")]
fn compile_dxbc() {
    use std::fmt::Write as _;

    use naga::back::hlsl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    const SAMPLER_HEAP_SPACE: u8 = 100;
    const COMPARISON_SAMPLER_HEAP_SPACE: u8 = 101;
    const SPECIAL_CONSTANTS_SPACE: u8 = 102;
    const SAMPLER_INDEX_REGISTER: u32 = 64;

    println!("cargo::rerun-if-changed=src/platform/cross/shaders.rs");
    println!("cargo::rerun-if-changed=src/platform/cross/shaders");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let mut table = String::new();
    writeln!(table, "pub(crate) const SAMPLER_HEAP_SPACE: u32 = {SAMPLER_HEAP_SPACE};").ok();
    writeln!(table, "pub(crate) const SPECIAL_CONSTANTS_SPACE: u32 = {SPECIAL_CONSTANTS_SPACE};").ok();
    writeln!(table, "pub(crate) const SAMPLER_INDEX_REGISTER: u32 = {SAMPLER_INDEX_REGISTER};").ok();
    table.push_str("pub(crate) fn dxbc(shader: &str, entry: &str) -> Option<&'static [u8]> {\n    match (shader, entry) {\n");
    for (name, _, _) in shaders::SHADERS {
        let source = shaders::wgsl_source(name);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{name}.wgsl : {}", error.emit_to_string(&source)));
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name}.wgsl : {error:?}"));
        let target = |space: u8, register: u32| hlsl::BindTarget { space, register, ..Default::default() };
        let mut binding_map = hlsl::BindingMap::default();
        let mut sampler_buffer_binding_map = hlsl::SamplerIndexBufferBindingMap::default();
        for (_, global) in module.global_variables.iter() {
            let Some(binding) = &global.binding else { continue };
            binding_map.insert(binding.clone(), target(binding.group as u8, binding.binding));
            if matches!(module.types[global.ty].inner, naga::TypeInner::Sampler { .. }) {
                sampler_buffer_binding_map.insert(
                    hlsl::SamplerIndexBufferKey { group: binding.group },
                    target(binding.group as u8, SAMPLER_INDEX_REGISTER),
                );
            }
        }
        let options = hlsl::Options {
            shader_model: hlsl::ShaderModel::V5_1,
            binding_map,
            fake_missing_bindings: false,
            special_constants_binding: Some(target(SPECIAL_CONSTANTS_SPACE, 0)),
            sampler_heap_target: hlsl::SamplerHeapBindTargets {
                standard_samplers: target(SAMPLER_HEAP_SPACE, 0),
                comparison_samplers: target(COMPARISON_SAMPLER_HEAP_SPACE, 0),
            },
            sampler_buffer_binding_map,
            restrict_indexing: true,
            force_loop_bounding: true,
            ..Default::default()
        };
        let mut hlsl_source = String::new();
        let pipeline_options = hlsl::PipelineOptions::default();
        let reflection = hlsl::Writer::new(&mut hlsl_source, &options, &pipeline_options)
            .write(&module, &info, None)
            .unwrap_or_else(|error| panic!("{name}.wgsl → HLSL : {error:?}"));
        std::fs::write(format!("{out}/{name}.hlsl"), &hlsl_source).expect("écriture HLSL");
        for (index, entry_point) in module.entry_points.iter().enumerate() {
            let hlsl_entry = reflection.entry_point_names[index]
                .as_ref()
                .unwrap_or_else(|error| panic!("{name}::{} : {error:?}", entry_point.name));
            let profile = match entry_point.stage {
                naga::ShaderStage::Vertex => "vs_5_1",
                naga::ShaderStage::Fragment => "ps_5_1",
                stage => panic!("{name}::{} : étage {stage:?} inattendu", entry_point.name),
            };
            let bytecode = fxc_compile(&hlsl_source, name, hlsl_entry, profile);
            let file = format!("{name}.{}.dxbc", entry_point.name);
            std::fs::write(format!("{out}/{file}"), bytecode).expect("écriture DXBC");
            writeln!(
                table,
                "        ({name:?}, {:?}) => Some(include_bytes!(concat!(env!(\"OUT_DIR\"), {:?}))),",
                entry_point.name,
                format!("/{file}"),
            )
            .ok();
        }
    }
    table.push_str("        _ => None,\n    }\n}\n");
    std::fs::write(format!("{out}/dxbc.rs"), table).expect("écriture dxbc.rs");
}

#[cfg(feature = "dx12")]
fn fxc_compile(source: &str, name: &str, entry: &str, profile: &str) -> Vec<u8> {
    use windows::Win32::Graphics::Direct3D::Fxc::{
        D3DCOMPILE_ENABLE_STRICTNESS, D3DCOMPILE_OPTIMIZATION_LEVEL3, D3DCompile,
    };
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::core::PCSTR;

    let entry_c = std::ffi::CString::new(entry).expect("nom d'entrée");
    let profile_c = std::ffi::CString::new(profile).expect("profil");
    let name_c = std::ffi::CString::new(format!("{name}.hlsl")).expect("nom");
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: tampons valides pendant l'appel ; les blobs rendus sont possédés.
    let result = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR(name_c.as_ptr().cast()),
            None,
            None,
            PCSTR(entry_c.as_ptr().cast()),
            PCSTR(profile_c.as_ptr().cast()),
            D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    let blob_bytes = |blob: &ID3DBlob| {
        // SAFETY: le blob possède `GetBufferSize` octets à `GetBufferPointer`.
        unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()).to_vec() }
    };
    match (result, code) {
        (Ok(()), Some(code)) => blob_bytes(&code),
        (result, _) => {
            let message = errors.as_ref().map(|blob| String::from_utf8_lossy(&blob_bytes(blob)).into_owned());
            panic!("FXC {name}::{entry} ({profile}) : {result:?}\n{}", message.unwrap_or_default());
        }
    }
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
