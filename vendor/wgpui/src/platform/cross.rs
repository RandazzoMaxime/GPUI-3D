// Sans aucun backend de rendu compilé, la couche GPU générique n'a aucune instance : code
// mort et chemins inatteignables attendus.
#![cfg_attr(not(any(feature = "wgpu", feature = "vulkan", all(feature = "dx12", windows))), allow(dead_code, unreachable_code, unused_variables))]

pub mod atlas;
pub mod dispatcher;
pub mod gpu;
pub mod hal;
pub mod keyboard;
pub mod platform;
pub mod render_context;
pub mod renderer;
pub mod resize_detector;
pub mod slab;
// WGSL composé à l'exécution pour wgpu ; les backends natifs le reçoivent compilé par build.rs.
#[cfg(any(feature = "wgpu", test))]
pub mod shaders;
pub mod slab_gpu;
pub mod surface_registry;
pub mod text_system;
pub mod window;

/// Re-export so the `PlatformWindow::with_winit_window` trait method can name this type
/// without pulling winit into every file that uses `platform.rs`.
pub use winit::window::Window as WinitWindow;
