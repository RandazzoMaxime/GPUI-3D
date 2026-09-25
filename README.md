# GPUI-3D

## A repo to quickly start a GPUI project that needs 3D!

[GPUI](https://www.gpui.rs/) UI on top, your real-time 3D engine underneath — one GPU device,
zero copies, and the UI is never redrawn for a 3D frame. Pick a variant, copy its folder, replace
the cube.

| Vulkan | Direct3D 12 | OpenGL |
|---|---|---|
| ![Vulkan](docs/screenshots/vulkan-native.png) | ![Direct3D 12](docs/screenshots/d3d12-native.png) | ![OpenGL](docs/screenshots/opengl-native.png) |
| **Metal** | **wgpu** | **UI example (backdrop blur)** |
| ![Metal](docs/screenshots/metal-native.png) | ![wgpu](docs/screenshots/wgpu.png) | ![Backdrop blur](docs/screenshots/ui-blur-showcase.png) |

## Supported backends

Two ways to render the UI:

- **wgpu** — the UI goes through wgpu; your 3D engine uses wgpu or the native API behind it.
- **Full native** — the UI *and* your 3D engine both use the native API directly. No wgpu in the
  binary.

| Variant | 3D engine | UI renderer | Platform |
|---|---|---|---|
| [`GPUI-WGPU`](GPUI-WGPU/src/main.rs) | wgpu | wgpu | Windows, macOS (Linux untested) |
| [`GPUI-VULKAN`](GPUI-VULKAN/src/main.rs) | Vulkan | Vulkan (native) | Windows, Vulkan 1.3 (Linux untested) |
| [`GPUI-DX12`](GPUI-DX12/src/dx12_cube.rs) | Direct3D 12 | Direct3D 12 (native) | Windows 10+ |
| [`GPUI-OPENGL`](GPUI-OPENGL/src/opengl_cube.rs) | OpenGL 4.5 | OpenGL 4.5 (native) | Windows |
| [`GPUI-METAL`](GPUI-METAL/src/metal_cube.rs) | Metal | Metal (native) | macOS |

## Quickstart

You need stable Rust and a GPU driver for the API you pick (plus the Xcode command line tools on
macOS).

```bash
cargo run -p gpui-vulkan    # or gpui-dx12, gpui-opengl, gpui-metal, gpui-wgpu
```

Drag to orbit the cube, scroll to zoom. Run one variant at a time (`-p`).

To start your own project:

1. Copy the variant folder you want.
2. Replace the cube (`CUBE_*`) and the shader with your scene.
3. Keep the `Renderer` shape: `new(surface)` creates your GPU resources, `render(surface, scene)`
   draws one frame and publishes it.
4. Build your UI in `shell/src/lib.rs` (`Shell::render`).

## How it works

- GPUI creates the GPU device; your engine renders on the **same device** through native
  handles (`VkDevice`, `ID3D12Device`, a shared GL context, `MTLDevice`).
- Your engine draws into a **triple-buffered surface** that the UI composites directly — no
  readback, no copy.
- Your engine runs on **its own thread** and publishes frames with `swap_buffers()` +
  `request_window_redraw()`, so the UI is recomposed from cache instead of redrawn.

More details — the internal GPU layer, per-API synchronization rules, how each backend was
verified — in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Useful environment variables

| Variable | Effect |
|---|---|
| `GPUI3D_TIME=1.3` | Freezes the scene (handy for screenshots and comparisons). |
| `WGPU_BACKEND=vulkan\|dx12\|metal\|gl` | Forces the wgpu backend in `GPUI-WGPU`. |
| `GPUI_D3D12_DEBUG=1` | D3D12 debug layer, messages on stderr. |
| `GPUI_GL_DEBUG=1` | OpenGL debug context, errors on stderr. |
| `MTL_DEBUG_LAYER=1` | Metal API validation. |
| `GPUI_FRAME_DUMP=frame.bmp` | Saves every presented frame as a BMP (wgpu and Metal). |

## Known limitations

- OpenGL uses WGL: Windows only for now.
- Linux is untested.
- Window resizing has not been tested interactively on Metal.
- Backdrop blurs look slightly different between Vulkan and the other APIs (opaque vs.
  premultiplied window alpha) — the same difference you get with wgpu.

## Contributing

Fork the repo, work on your fork, open a pull request against `main`. Do not
create branches, push, or merge on this repository — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Original GPUI-3D code is [0BSD](LICENSE): reuse it for anything. A credit is
welcome, never required.

Third-party code in `vendor/` keeps its own license (`vendor/wgpui` is
Apache-2.0) — see [NOTICE](NOTICE).
