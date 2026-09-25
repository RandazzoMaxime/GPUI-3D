# Architecture

This document covers the parts of GPUI-3D that are not visible from the examples: the internal
GPU layer under GPUI's renderer, how shaders reach each API, the contract a native 3D engine
must follow, and how the backends were verified.

## Layers

```
 your 3D engine (wgpu / Vulkan / D3D12 / OpenGL / Metal)
        │  native handles + triple-buffered surface (SurfaceHandle)
        ▼
 shell/  ── GPUI app: chrome, camera, render thread
        │
 vendor/wgpui  ── GPUI fork
        │  Renderer<G: Gpu>  (one generic UI renderer)
        ▼
 platform/cross/hal.rs  ── trait Gpu
        ├── hal/wgpu.rs     (feature `wgpu`, default)
        ├── hal/vulkan.rs   (feature `vulkan`, ash)
        ├── hal/d3d12.rs    (feature `dx12`, windows-rs)
        ├── hal/gl.rs       (feature `opengl`, WGL + glow)
        └── hal/metal.rs    (feature `metal`, objc2-metal)
```

The backend is chosen at runtime among those compiled in, through
`RendererBackend` (`Application::with_renderer`). The shell exposes it as `Ui::Wgpu(backends)`,
`Ui::Vulkan`, `Ui::Dx12`, `Ui::OpenGl` or `Ui::Metal`, each behind a Cargo feature of
`gpui3d-shell`. Without a GPU the app fails with an explicit error: there is no software
fallback.

## The `Gpu` trait

`hal.rs` mirrors the subset of wgpu the UI renderer actually uses: storage buffers instead of
vertex buffers, a single draw call shape, uniforms (one with a dynamic offset), 2D textures
sampled with one linear clamp-to-edge sampler, clear/load render passes and copies. Every
implementation honours three contracts:

- `write_buffer` / `write_texture` take effect before the commands of the next `submit`, even
  if those commands were recorded earlier (wgpu's `queue.write_*` semantics). Each backend
  stages the writes and executes them in a command list submitted just before the main one.
- In `draw`, `vertex_index` and `instance_index` include the first vertex / first instance.
- Clip space is WebGPU's: Y up, `@builtin(position)` in pixels from the top-left, depth 0..1.

How each backend meets them:

| | Resource state | Lifetime | Base instance | Clip space |
|---|---|---|---|---|
| **Vulkan** | Per-image layout tracking; a fixup command buffer at submit moves each image from its real layout to the first one used | Command buffers retain `Arc`s until a timeline semaphore passes their serial | Built in (`InstanceIndex`) | Negative viewport height (as wgpu-hal) |
| **D3D12** | Per-texture state tracking with a fixup list; buffers tracked per list from `COMMON` (they decay at every `ExecuteCommandLists`) | Retained until a fence passes the submission | naga special constants (`first_vertex`, `first_instance`) as root constants | Same as WebGPU |
| **OpenGL** | Implicit (driver) | GL deletes objects when unused; Rust handles queue names for deletion | `naga_vs_first_instance` uniform | naga flips Y in vertex shaders; row 0 at the top; presentation blits flipped |
| **Metal** | Tracked resources (Metal inserts dependencies) | Command buffers retain referenced objects | Built in (`[[instance_id]]`) | Same as WebGPU |

OpenGL has one more constraint: a context is current on one thread at a time, while the HAL can
be called from any thread. Every GL access therefore runs in a *section* (lock, make the context
current on the calling thread, work, release it), and passes are recorded in memory and replayed
at `submit`, so a frame costs two sections.

## Shaders

The UI's shaders are WGSL (`platform/cross/shaders/*.wgsl`, composed by `shaders.rs`). For the
native backends, `vendor/wgpui/build.rs` translates them with naga at build time and generates a
lookup table in `OUT_DIR`:

| Backend | Output | Compiled | Bindings |
|---|---|---|---|
| Vulkan | SPIR-V 1.6 | build time | descriptor set = group, binding = binding |
| D3D12 | HLSL SM 5.1 → DXBC (FXC, `d3dcompiler_47`) | build time (Windows host) | space = group, register = binding; naga's sampler heap indexed by a zero buffer (one sampler) |
| OpenGL | GLSL 4.50, one source per entry point | driver, at startup | binding point = group × 4 + binding, per class |
| Metal | MSL 2.4, one source per shader | driver, at startup | slot = group × 4 + binding, per class; slot 30 reserved for naga's buffer-sizes argument |

The binding conventions are emitted as constants in the generated table, so `build.rs` is their
single source of truth.

## Native 3D engine contract

A native engine gets its device through `SurfaceHandle::native_device()` and the texture to render
into through `native_back_buffer()` (keep that `NativeBackBuffer` alive until the GPU is done with
the frame). It then publishes with `swap_buffers()` and wakes the window with
`request_window_redraw()`.

- **Vulkan:** `VkQueue` needs external synchronization, so every `vkQueueSubmit` happens under
  `native_queue_lock()` (the compositor takes it too). The image arrives and must leave in
  `SHADER_READ_ONLY_OPTIMAL`; render-pass subpass dependencies carry the synchronization with the
  compositor.
- **Direct3D 12:** same `ID3D12CommandQueue` as the compositor, so submission order is enough;
  a D3D12 queue is thread-safe, no lock. The resource arrives and must leave in
  `PIXEL_SHADER_RESOURCE | NON_PIXEL_SHADER_RESOURCE`.
- **OpenGL:** the engine creates its own context sharing objects with the UI's
  (`NativeDevice::OpenGl`, created under `native_queue_lock()`, on a window with the given pixel
  format), renders straight into the surface texture (`glClipControl(UPPER_LEFT, ZERO_TO_ONE)`
  keeps row 0 at the top) and calls `glFinish` before `swap_buffers()`: two contexts do not order
  their commands. The UI likewise finishes each presentation before a buffer goes back to the
  engine.
- **Metal:** same `MTLCommandQueue` as the compositor and tracked textures: no fence. Commit, then
  `swap_buffers()`.

Surfaces are cleared when they are created, so the compositor never samples uninitialized memory.

## Verification

Each backend was checked against the others pixel by pixel, not by eye:

- Six of the fork's examples (`shadow`, `text`, `gradient`, `svg`, `blur_showcase`,
  `emoji_display`) and every 3D variant with a frozen scene (`GPUI3D_TIME=1.3`).
- Windows: window captures with `PrintWindow` ([`tools/bench/shot.ps1`](../tools/bench/shot.ps1)),
  compared with [`tools/bench/imgdiff.ps1`](../tools/bench/imgdiff.ps1) (max channel difference
  and pixels above 2 LSB, `-Map` writes a difference image).
- macOS over SSH: no screen access, so frames are dumped by the GPU with `GPUI_FRAME_DUMP`
  ([`tools/bench/run_mac.sh`](../tools/bench/run_mac.sh)); the dump was checked to be identical
  to a screen capture of the same frame on Windows.
- Validation layers on: D3D12 debug layer, OpenGL debug context, Metal API validation — no
  messages. Vulkan was checked by pixel comparison only (no Vulkan SDK on the test machine).

Results: native Vulkan is bit-identical to wgpu on Vulkan; native D3D12 matches wgpu on D3D12
(bit-identical on 5 of 6 examples, 1 LSB on `gradient`); native OpenGL is bit-identical to native
D3D12; native Metal is bit-identical to wgpu on Metal. Opaque presentation (D3D12, OpenGL, Metal)
versus premultiplied alpha (Vulkan) changes backdrop blurs by up to 18 LSB, identically with wgpu.

## Adding a backend

1. Implement `Gpu` in `platform/cross/hal/<api>.rs`, behind a Cargo feature.
2. Translate the WGSL shaders in `build.rs` and emit the binding conventions as constants.
3. Add the variant to `RendererBackend`, `GpuContext`, `WindowAtlas`, `WindowRenderer` and the
   `dispatch!` macro (`gpu.rs`), and to `CrossWindow::create_surface` (`window.rs`).
4. Expose it in the shell (`Ui`) and verify it against an existing backend with the scripts in
   `tools/bench/`.
