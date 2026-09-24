#[cfg(windows)]
mod dx12_cube;

#[cfg(windows)]
fn main() {
    gpui3d_shell::run::<dx12_cube::Dx12Cube>("D3D12 natif", gpui3d_shell::Ui::Dx12);
}

#[cfg(not(windows))]
fn main() {
    eprintln!("GPUI-DX12 requiert Windows : utiliser GPUI-WGPU ou GPUI-VULKAN ailleurs.");
}
