//! GPUI-3D : backend de rendu de l'UI, choisi à l'exécution parmi ceux compilés.
//!
//! Chaque backend (feature Cargo) fournit un contexte par application, un atlas et un
//! renderer par fenêtre. Les enums ci-dessous en font le dispatch ; leur variante
//! `Absent` (inhabitée) les garde valides quand aucun backend n'est compilé : l'app
//! démarre alors sans pouvoir ouvrir de fenêtre, avec une erreur explicite.

use std::convert::Infallible;
use std::sync::Arc;

use anyhow::Result;

use crate::{DevicePixels, GpuSpecs, LayerKey, PlatformAtlas, Scene, Size, WindowPresentMode};

#[cfg(feature = "wgpu")]
use super::{
    atlas::WgpuAtlas,
    render_context::{WgpuContext, WgpuOptions},
    renderer::WgpuRenderer,
};
#[cfg(any(feature = "vulkan", all(feature = "dx12", windows), all(feature = "opengl", windows), all(feature = "metal", target_os = "macos")))]
use super::{atlas::Atlas, render_context::RenderContext, renderer::Renderer};
#[cfg(all(feature = "metal", target_os = "macos"))]
use super::hal::metal::MetalGpu;
#[cfg(all(feature = "opengl", windows))]
use super::hal::gl::GlGpu;
#[cfg(all(feature = "dx12", windows))]
use super::hal::d3d12::D3d12Gpu;
#[cfg(feature = "vulkan")]
use super::hal::vulkan::VulkanGpu;

#[cfg(all(target_family = "wasm", not(feature = "wgpu")))]
compile_error!("gpui-ce en wasm exige la feature `wgpu` (seul backend disponible dans un navigateur).");

/// Backend de rendu de l'UI, et donc du device partagé avec les moteurs 3D.
#[derive(Clone)]
#[non_exhaustive]
pub enum RendererBackend {
    /// wgpu : abstraction multi-API (Vulkan, Metal, D3D12, GL), choix de l'adaptateur
    /// par [`WgpuOptions`].
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuOptions),
    /// Vulkan 1.3 natif (ash), sans wgpu : GPU discret de préférence.
    #[cfg(feature = "vulkan")]
    Vulkan,
    /// D3D12 natif (windows-rs), sans wgpu : GPU matériel le plus performant.
    #[cfg(all(feature = "dx12", windows))]
    Dx12,
    /// OpenGL 4.5 core natif (WGL), sans wgpu.
    #[cfg(all(feature = "opengl", windows))]
    OpenGl,
    /// Metal natif (objc2-metal), sans wgpu.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal,
    #[doc(hidden)]
    Absent(Infallible),
}

impl RendererBackend {
    /// Backend nommé par `GPUI_RENDERER` (`wgpu`, `vulkan`, `dx12`, `opengl`, `metal`) s'il est compilé, sinon le
    /// premier compilé dans l'ordre de déclaration ; `None` si aucun.
    pub fn compiled_default() -> Option<Self> {
        let requested = std::env::var("GPUI_RENDERER").unwrap_or_default();
        #[cfg(feature = "vulkan")]
        if requested.eq_ignore_ascii_case("vulkan") {
            return Some(Self::Vulkan);
        }
        #[cfg(all(feature = "dx12", windows))]
        if requested.eq_ignore_ascii_case("dx12") {
            return Some(Self::Dx12);
        }
        #[cfg(all(feature = "opengl", windows))]
        if requested.eq_ignore_ascii_case("opengl") {
            return Some(Self::OpenGl);
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if requested.eq_ignore_ascii_case("metal") {
            return Some(Self::Metal);
        }
        #[cfg(feature = "wgpu")]
        if requested.is_empty() || requested.eq_ignore_ascii_case("wgpu") {
            return Some(Self::Wgpu(WgpuOptions::default()));
        }
        if !requested.is_empty() {
            log::error!("GPUI_RENDERER={requested} : backend non compilé dans ce binaire");
            return None;
        }
        #[cfg(feature = "vulkan")]
        return Some(Self::Vulkan);
        #[cfg(all(feature = "dx12", windows, not(feature = "vulkan")))]
        return Some(Self::Dx12);
        #[cfg(all(feature = "opengl", windows, not(feature = "vulkan"), not(feature = "dx12")))]
        return Some(Self::OpenGl);
        #[cfg(all(feature = "metal", target_os = "macos", not(feature = "vulkan")))]
        return Some(Self::Metal);
        #[allow(unreachable_code)]
        None
    }
}

/// Contexte GPU de l'application : device, queue, ressources partagées entre fenêtres.
#[derive(Clone)]
pub(crate) enum GpuContext {
    #[cfg(feature = "wgpu")]
    Wgpu(Arc<WgpuContext>),
    #[cfg(feature = "vulkan")]
    Vulkan(Arc<RenderContext<VulkanGpu>>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Arc<RenderContext<D3d12Gpu>>),
    #[cfg(all(feature = "opengl", windows))]
    OpenGl(Arc<RenderContext<GlGpu>>),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(Arc<RenderContext<MetalGpu>>),
    #[allow(dead_code)]
    Absent(Infallible),
}

/// Atlas de sprites d'une fenêtre (glyphes, SVG, images).
pub(crate) enum WindowAtlas {
    #[cfg(feature = "wgpu")]
    Wgpu(Arc<WgpuAtlas>),
    #[cfg(feature = "vulkan")]
    Vulkan(Arc<Atlas<VulkanGpu>>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Arc<Atlas<D3d12Gpu>>),
    #[cfg(all(feature = "opengl", windows))]
    OpenGl(Arc<Atlas<GlGpu>>),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(Arc<Atlas<MetalGpu>>),
    #[allow(dead_code)]
    Absent(Infallible),
}

/// Renderer d'une fenêtre : dessine la scène et compose les surfaces 3D.
pub(crate) enum WindowRenderer {
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuRenderer),
    #[cfg(feature = "vulkan")]
    Vulkan(Renderer<VulkanGpu>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Renderer<D3d12Gpu>),
    #[cfg(all(feature = "opengl", windows))]
    OpenGl(Renderer<GlGpu>),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(Renderer<MetalGpu>),
    #[allow(dead_code)]
    Absent(Infallible),
}

/// Applique `$body` à la valeur interne de chaque variante compilée.
macro_rules! dispatch {
    ($value:expr, $enum:ident, $inner:ident => $body:expr) => {
        match $value {
            #[cfg(feature = "wgpu")]
            $enum::Wgpu($inner) => $body,
            #[cfg(feature = "vulkan")]
            $enum::Vulkan($inner) => $body,
            #[cfg(all(feature = "dx12", windows))]
            $enum::Dx12($inner) => $body,
            #[cfg(all(feature = "opengl", windows))]
            $enum::OpenGl($inner) => $body,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            $enum::Metal($inner) => $body,
            $enum::Absent(never) => match *never {},
        }
    };
}

impl GpuContext {
    /// Crée le contexte du backend demandé.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) fn new(backend: &RendererBackend) -> Result<Self> {
        match backend {
            #[cfg(feature = "wgpu")]
            RendererBackend::Wgpu(options) => Ok(Self::Wgpu(Arc::new(WgpuContext::new(options)?))),
            #[cfg(feature = "vulkan")]
            RendererBackend::Vulkan => Ok(Self::Vulkan(Arc::new(RenderContext::with_gpu(VulkanGpu::new()?)))),
            #[cfg(all(feature = "dx12", windows))]
            RendererBackend::Dx12 => Ok(Self::Dx12(Arc::new(RenderContext::with_gpu(D3d12Gpu::new()?)))),
            #[cfg(all(feature = "opengl", windows))]
            RendererBackend::OpenGl => Ok(Self::OpenGl(Arc::new(RenderContext::with_gpu(GlGpu::new()?)))),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            RendererBackend::Metal => Ok(Self::Metal(Arc::new(RenderContext::with_gpu(MetalGpu::new()?)))),
            RendererBackend::Absent(never) => match *never {},
        }
    }

    pub(crate) fn new_atlas(&self) -> WindowAtlas {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(context) => WindowAtlas::Wgpu(Arc::new(WgpuAtlas::new(context.gpu.clone()))),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(context) => WindowAtlas::Vulkan(Arc::new(Atlas::new(context.gpu.clone()))),
            #[cfg(all(feature = "dx12", windows))]
            Self::Dx12(context) => WindowAtlas::Dx12(Arc::new(Atlas::new(context.gpu.clone()))),
            #[cfg(all(feature = "opengl", windows))]
            Self::OpenGl(context) => WindowAtlas::OpenGl(Arc::new(Atlas::new(context.gpu.clone()))),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(context) => WindowAtlas::Metal(Arc::new(Atlas::new(context.gpu.clone()))),
            Self::Absent(never) => match *never {},
        }
    }

    /// Renderer d'une fenêtre de `width`×`height` pixels physiques, sur l'atlas de la fenêtre.
    pub(crate) fn new_renderer(
        &self,
        window: &winit::window::Window,
        atlas: &WindowAtlas,
        width: u32,
        height: u32,
    ) -> Result<WindowRenderer> {
        match (self, atlas) {
            #[cfg(feature = "wgpu")]
            (Self::Wgpu(context), WindowAtlas::Wgpu(atlas)) => Ok(WindowRenderer::Wgpu(WgpuRenderer::new(
                context.clone(),
                window,
                atlas.clone(),
                width,
                height,
            )?)),
            #[cfg(feature = "vulkan")]
            (Self::Vulkan(context), WindowAtlas::Vulkan(atlas)) => Ok(WindowRenderer::Vulkan(Renderer::new(
                context.clone(),
                window,
                atlas.clone(),
                width,
                height,
            )?)),
            #[cfg(all(feature = "dx12", windows))]
            (Self::Dx12(context), WindowAtlas::Dx12(atlas)) => Ok(WindowRenderer::Dx12(Renderer::new(
                context.clone(),
                window,
                atlas.clone(),
                width,
                height,
            )?)),
            #[cfg(all(feature = "opengl", windows))]
            (Self::OpenGl(context), WindowAtlas::OpenGl(atlas)) => Ok(WindowRenderer::OpenGl(Renderer::new(
                context.clone(),
                window,
                atlas.clone(),
                width,
                height,
            )?)),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            (Self::Metal(context), WindowAtlas::Metal(atlas)) => Ok(WindowRenderer::Metal(Renderer::new(
                context.clone(),
                window,
                atlas.clone(),
                width,
                height,
            )?)),
            (Self::Absent(never), _) => match *never {},
            #[allow(unreachable_patterns)]
            (_, WindowAtlas::Absent(never)) => match *never {},
            #[allow(unreachable_patterns)]
            _ => Err(anyhow::anyhow!("atlas et contexte GPU de backends différents")),
        }
    }
}

impl WindowAtlas {
    pub(crate) fn as_platform(&self) -> Arc<dyn PlatformAtlas> {
        dispatch!(self, WindowAtlas, atlas => atlas.clone())
    }
}

impl WindowRenderer {
    pub(crate) fn draw(&mut self, scene: &Scene) {
        dispatch!(self, WindowRenderer, renderer => renderer.draw(scene))
    }

    pub(crate) fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        dispatch!(self, WindowRenderer, renderer => renderer.update_drawable_size(size))
    }

    pub(crate) fn take_rerecord_requests(&mut self) -> Vec<LayerKey> {
        dispatch!(self, WindowRenderer, renderer => renderer.take_rerecord_requests())
    }

    pub(crate) fn set_present_mode(&mut self, mode: WindowPresentMode) -> bool {
        dispatch!(self, WindowRenderer, renderer => renderer.set_present_mode(mode))
    }

    pub(crate) fn lock_glass_backdrop(&mut self, key: u32) {
        dispatch!(self, WindowRenderer, renderer => renderer.lock_glass_backdrop(key))
    }

    pub(crate) fn gpu_specs(&self) -> GpuSpecs {
        dispatch!(self, WindowRenderer, renderer => renderer.gpu_specs())
    }

    pub(crate) fn has_pending_surfaces(&self) -> bool {
        dispatch!(self, WindowRenderer, renderer => renderer.has_pending_surfaces())
    }

    pub(crate) fn any_unconsumed_surface_frame(&self) -> bool {
        dispatch!(self, WindowRenderer, renderer => renderer.any_unconsumed_surface_frame())
    }

    pub(crate) fn take_new_surface_frame(&self) -> bool {
        dispatch!(self, WindowRenderer, renderer => renderer.take_new_surface_frame())
    }

    /// Profilage GPU : wgpu seulement.
    #[cfg(feature = "flamegraph")]
    pub(crate) fn gpu_memory_snapshot(&self) -> Option<crate::GpuMemorySnapshot> {
        match self {
            Self::Wgpu(renderer) => Some(renderer.gpu_memory_snapshot()),
            _ => None,
        }
    }

    #[cfg(feature = "flamegraph")]
    pub(crate) fn gpu_device_and_queue(&self) -> Option<(wgpu::Device, wgpu::Queue)> {
        match self {
            Self::Wgpu(renderer) => Some(renderer.gpu_device_and_queue()),
            _ => None,
        }
    }
}
