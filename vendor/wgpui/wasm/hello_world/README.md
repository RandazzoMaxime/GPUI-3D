# WGPUI WASM Hello World

A minimal test to verify GPUI rendering works in the browser via WebGPU.

## Prerequisites

- [Rust](https://rustup.rs/) with the `wasm32-unknown-unknown` target

The build script (`serve.ps1`) automatically installs `wasm-pack` and adds the
`wasm32` target if they're missing.

## Build and Serve

```powershell
powershell -File serve.ps1
```

This builds the WASM binary and starts a HTTP server on port 8080.

Open `http://localhost:8080` in a browser that supports WebGPU
(Chrome 113+, Edge 113+, Firefox Nightly).

## What It Tests

The example renders colored rectangles on a blue background using WebGPU.
A successful render shows the GPUI scene graph being drawn.
A black screen indicates the renderer failed to initialize — check the
browser's developer console (F12) for error messages.
