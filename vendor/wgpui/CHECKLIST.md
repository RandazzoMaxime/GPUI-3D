# WASM Compilation Fix Checklist

## 1. `use util::*` → `use crate::util::*` ✅
- `src/app.rs` - `use crate::util::ResultExt` + `use crate::debug_panic`
- `src/elements/div.rs` - `use crate::util::ResultExt`
- `src/elements/img.rs` - `use crate::util::ResultExt`
- `src/elements/svg.rs` - `use crate::util::ResultExt`
- `src/elements/text.rs` - `use crate::util::ResultExt`
- `src/subscription.rs` - `use crate::util::post_inc`
- `src/window.rs` - `use crate::util::{post_inc, ResultExt, measure}`
- `src/executor.rs` - `use crate::util::TryFutureExt`
- `src/app/context.rs` - `use crate::util::Deferred`, `crate::util::defer(…)`
- `src/app/async_context.rs` - `crate::util::Deferred`, `crate::util::defer(…)`
- `src/platform/test/dispatcher.rs` - `use crate::util::post_inc`

## 2. `smol`, `parking`, `num_cpus` gates ✅
- `src/executor.rs` - `use smol::prelude::*;` gated
- `src/executor.rs` - Both `block_internal` functions gated
- `src/executor.rs` - `block`, `block_test`, `block_with_timeout` gated
- `src/executor.rs` - `num_cpus::get()` gated (returns 1 on WASM)
- `src/executor.rs` - `Scope::drop`'s blocking call gated (+ `use futures::StreamExt`)
- `src/executor.rs` - `waker_fn` import gated

## 3. `arboard`, `device_query`, `rfd`, `opener`, `open` gates ✅
- `src/platform/cross/platform.rs` - `arboard::Clipboard` import + usages gated
- `src/platform/cross/platform.rs` - `::open::that_detached` in `open_url`, `reveal_path`, `open_with_system` gated
- `src/platform/cross/platform.rs` - `opener::reveal` in `reveal_path` gated
- `src/platform/cross/platform.rs` - `rfd::AsyncFileDialog` in `prompt_for_paths`, `prompt_for_new_path` gated
- `src/platform/cross/resize_detector.rs` - `device_query::*` import + field + method body gated

## 4. `http_client` types gating ✅
- `src/app.rs` - `NullHttpClient` impl gated; `BoxedHttpClient` type alias for WASM
- `src/app.rs` - `http_client` field, `http_client()`, `set_http_client()` gated
- `src/app.rs` - `Application::new()`/`headless()`/`with_wgpu_options()` made WASM-compatible
- `src/elements/img.rs` - `is_uri` gated (returns `false` on WASM)
- `src/elements/img.rs` - `BadStatus` variant gated; `Resource::Uri` branch gated
- `src/elements/img.rs` - `client` variable declaration gated

## 5. SVG/usvg/resvg gating ✅
- `src/svg_renderer.rs` - Rewritten with `backend` module: non-WASM version uses usvg/resvg, WASM version returns errors
- `src/svg_renderer.rs` - `RenderSvgParams` derived traits added (Clone, PartialEq, Eq, Hash)
- `src/svg_renderer.rs` - `SMOOTH_SVG_SCALE_FACTOR` is `f32` (matching original)
- `src/elements/img.rs` - `Usvg` variant + `From<usvg::Error>` impl gated

## 6. `wasm-bindgen-futures` gating ✅
- `src/gpui.rs` - Timer re-export uses `#[cfg(feature = "wasm")]`
- `src/gpui.rs` - Removed unused `use web_sys::VideoFrame`

## 7. `TryFutureExt` fix ✅
- `src/util.rs` - Trait now bound on `Future<Output = Result<T, E>>`, uses `FutureExt::map`

## Additional fixes ✅
- `src/app.rs` - `debug_panic` macro import fixed (uses `#[macro_export]` from wasm_compat)
- `src/util.rs` - Added `defer` function to wasm_compat module
- `src/util.rs` - `ResultExt::log_err` returns `Option<T>` (matching external util crate behavior)
- `src/platform/cross/dispatcher.rs` - WASM paths fixed (discard unused results)
- `src/platform.rs` - SVG render_single_frame call fixed
- `src/executor.rs` - `poll` fix for newer Rust (Pin::new)
- `src/executor.rs` - `log_tracked_err` call deref fix
- `src/elements/img.rs` - Removed unused imports (Context, AsyncReadExt, FromStr gated)

## Remaining issues (pre-existing, not in original 7 categories)
- 9 errors in `platform/cross/atlas.rs` and `elements/wgpu_surface.rs`: `wgpu`'s WebGPU backend uses `Rc<...>` internally which is not `Send`/`Sync`. These need deeper platform gating or Send+Sync wrappers.
- All 7 requested categories are fully resolved.
