//! GPUI-3D : couche GPU interne du renderer de l'UI.
//!
//! Le renderer (`renderer.rs`) est générique sur [`Gpu`] ; chaque API en fournit une
//! implémentation (wgpu, Vulkan, D3D12, OpenGL, Metal). La forme reprend le
//! sous-ensemble de wgpu que le renderer utilise : pas de vertex/index buffer (les
//! instances sont lues en storage buffer), un seul type de draw, des uniforms dont un à
//! offset dynamique, des textures 2D échantillonnées en linéaire, des passes clear/load
//! et des copies.
//!
//! Contrats communs à toutes les implémentations :
//! - `write_buffer` / `write_texture` prennent effet avant les commandes du prochain
//!   `submit`, même enregistrées plus tôt dans l'encodeur (sémantique `queue.write_*`).
//! - `draw` : `vertex_index` et `instance_index` incluent le premier sommet / la première
//!   instance de la plage.
//! - Espace clip WebGPU : Y vers le haut, `@builtin(position)` en pixels depuis le coin
//!   haut-gauche, profondeur 0..1.

use std::ops::{BitOr, Range};

use crate::{GpuSpecs, WindowPresentMode};

#[cfg(feature = "wgpu")]
pub(crate) mod wgpu;
#[cfg(feature = "vulkan")]
pub(crate) mod vulkan;

/// Usages d'un buffer (combinables par `|`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BufferUsage(u8);

impl BufferUsage {
    pub(crate) const UNIFORM: Self = Self(1);
    pub(crate) const STORAGE: Self = Self(1 << 1);
    pub(crate) const VERTEX: Self = Self(1 << 2);
    pub(crate) const COPY_SRC: Self = Self(1 << 3);
    pub(crate) const COPY_DST: Self = Self(1 << 4);

    pub(crate) fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Usages d'une texture (combinables par `|`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TextureUsage(u8);

impl TextureUsage {
    pub(crate) const RENDER_TARGET: Self = Self(1);
    pub(crate) const SAMPLED: Self = Self(1 << 1);
    pub(crate) const COPY_SRC: Self = Self(1 << 2);
    pub(crate) const COPY_DST: Self = Self(1 << 3);

    pub(crate) fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl BitOr for TextureUsage {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShaderStages {
    Vertex,
    Fragment,
    VertexFragment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BindingKind {
    /// `min_size` : taille liée par draw quand `dynamic_offset` (sinon le buffer entier).
    Uniform { dynamic_offset: bool, min_size: Option<u64> },
    /// Lecture seule.
    Storage,
    /// `texture_2d<f32>` filtrable.
    Texture,
    /// Sampler filtrant.
    Sampler,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LayoutEntry {
    pub(crate) binding: u32,
    pub(crate) visibility: ShaderStages,
    pub(crate) kind: BindingKind,
}

pub(crate) enum BindResource<'a, G: Gpu> {
    /// `size: None` = jusqu'à la fin du buffer.
    Buffer { buffer: &'a G::Buffer, offset: u64, size: Option<u64> },
    Texture(&'a G::TextureView),
    Sampler(&'a G::Sampler),
}

pub(crate) struct BindEntry<'a, G: Gpu> {
    pub(crate) binding: u32,
    pub(crate) resource: BindResource<'a, G>,
}

/// Les shaders WGSL de l'UI (`shaders/*.wgsl`), un par pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ShaderId {
    Quads,
    Shadows,
    BackdropBlur,
    Underlines,
    MonoSprites,
    PolySprites,
    Surfaces,
    Paths,
}

impl ShaderId {
    /// Nom dans [`super::shaders::SHADERS`].
    #[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Quads => "quads",
            Self::Shadows => "shadows",
            Self::BackdropBlur => "backdrop_blur",
            Self::Underlines => "underlines",
            Self::MonoSprites => "mono_sprites",
            Self::PolySprites => "poly_sprites",
            Self::Surfaces => "surfaces",
            Self::Paths => "paths",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Topology {
    TriangleList,
    TriangleStrip,
}

/// Mélange de l'unique cible couleur.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Blend {
    Alpha,
    PremultipliedAlpha,
}

pub(crate) struct PipelineDesc<'a, G: Gpu> {
    pub(crate) label: &'static str,
    pub(crate) shader: ShaderId,
    pub(crate) vertex_entry: &'static str,
    pub(crate) fragment_entry: &'static str,
    pub(crate) topology: Topology,
    /// Un layout par groupe, dans l'ordre des `@group`.
    pub(crate) layouts: &'a [&'a G::BindGroupLayout],
    pub(crate) format: G::Format,
    pub(crate) blend: Blend,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LoadOp {
    Clear([f64; 4]),
    Load,
}

pub(crate) struct PassDesc<'a, G: Gpu> {
    #[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
    pub(crate) label: &'static str,
    pub(crate) target: &'a G::TextureView,
    pub(crate) load: LoadOp,
    #[cfg(feature = "flamegraph")]
    pub(crate) timestamps: Option<&'a <G::Profiler as GpuProfiler<G>>::PassTimestamps>,
}

/// Résultat d'une acquisition d'image de swapchain.
pub(crate) enum Acquire<F> {
    Frame(F),
    /// Swapchain à reconfigurer (taille, perte, validation), puis réessayer.
    Outdated,
    /// Trame à sauter (délai, fenêtre masquée).
    Skip(&'static str),
}

pub(crate) trait Gpu: Sized + Send + Sync + 'static {
    type Format: Copy + PartialEq + std::fmt::Debug + Send + Sync;
    type Buffer: Clone + Send + Sync;
    type Texture: Clone + Send + Sync;
    type TextureView: Clone + Send + Sync;
    type Sampler: Send + Sync;
    type BindGroupLayout: Send + Sync;
    type BindGroup: Clone + Send + Sync;
    type Pipeline: Send + Sync;
    type Encoder;
    type Pass<'a>;
    type Swapchain;
    type Frame;
    #[cfg(feature = "flamegraph")]
    type Profiler: GpuProfiler<Self>;

    /// Formats des pages d'atlas (R8 et RGBA8 unorm).
    const ATLAS_MONOCHROME: Self::Format;
    const ATLAS_POLYCHROME: Self::Format;

    fn bytes_per_pixel(format: Self::Format) -> u32;
    fn gpu_specs(&self) -> GpuSpecs;
    fn min_uniform_offset_alignment(&self) -> u32;

    fn create_buffer(&self, label: &str, size: u64, usage: BufferUsage) -> Self::Buffer;
    fn create_buffer_init(&self, label: &str, contents: &[u8], usage: BufferUsage) -> Self::Buffer;
    fn buffer_size(buffer: &Self::Buffer) -> u64;
    fn write_buffer(&self, buffer: &Self::Buffer, offset: u64, data: &[u8]);

    fn create_texture(
        &self,
        label: &str,
        width: u32,
        height: u32,
        format: Self::Format,
        usage: TextureUsage,
    ) -> Self::Texture;
    fn texture_size(texture: &Self::Texture) -> (u32, u32);
    fn create_view(texture: &Self::Texture) -> Self::TextureView;
    /// `data` : lignes jointives de `width * bytes_per_pixel` octets.
    fn write_texture(
        &self,
        texture: &Self::Texture,
        origin: (u32, u32),
        size: (u32, u32),
        bytes_per_pixel: u32,
        data: &[u8],
    );
    /// Filtrage linéaire, bords clampés, sans mip.
    fn create_linear_sampler(&self, label: &str) -> Self::Sampler;

    fn create_bind_group_layout(&self, label: &str, entries: &[LayoutEntry]) -> Self::BindGroupLayout;
    fn create_bind_group(
        &self,
        label: &str,
        layout: &Self::BindGroupLayout,
        entries: &[BindEntry<'_, Self>],
    ) -> Self::BindGroup;
    fn create_pipeline(&self, desc: &PipelineDesc<'_, Self>) -> Self::Pipeline;

    fn create_encoder(&self, label: &str) -> Self::Encoder;
    fn begin_pass<'a>(encoder: &'a mut Self::Encoder, desc: &PassDesc<'_, Self>) -> Self::Pass<'a>;
    fn set_pipeline(pass: &mut Self::Pass<'_>, pipeline: &Self::Pipeline);
    fn set_bind_group(pass: &mut Self::Pass<'_>, index: u32, group: &Self::BindGroup, dynamic_offsets: &[u32]);
    fn set_viewport(pass: &mut Self::Pass<'_>, x: f32, y: f32, width: f32, height: f32);
    fn set_scissor_rect(pass: &mut Self::Pass<'_>, x: u32, y: u32, width: u32, height: u32);
    fn draw(pass: &mut Self::Pass<'_>, vertices: Range<u32>, instances: Range<u32>);
    fn copy_buffer_to_buffer(
        encoder: &mut Self::Encoder,
        source: &Self::Buffer,
        source_offset: u64,
        destination: &Self::Buffer,
        destination_offset: u64,
        size: u64,
    );
    /// Copie de `width`×`height` depuis l'origine, textures de même format.
    fn copy_texture_to_texture(
        encoder: &mut Self::Encoder,
        source: &Self::Texture,
        destination: &Self::Texture,
        width: u32,
        height: u32,
    );

    /// Surface de présentation d'une fenêtre : format non sRGB (les shaders écrivent
    /// déjà du sRGB), alpha prémultiplié si disponible. Non configurée : l'appelant
    /// enchaîne sur [`Gpu::configure_swapchain`].
    fn create_swapchain(
        &self,
        window: raw_window_handle::RawWindowHandle,
        display: raw_window_handle::RawDisplayHandle,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Self::Swapchain>;
    fn swapchain_format(swapchain: &Self::Swapchain) -> Self::Format;
    fn swapchain_premultiplied(swapchain: &Self::Swapchain) -> bool;
    fn swapchain_size(swapchain: &Self::Swapchain) -> (u32, u32);
    fn swapchain_present_mode(swapchain: &Self::Swapchain) -> WindowPresentMode;
    fn supported_present_modes(&self, swapchain: &Self::Swapchain) -> Vec<WindowPresentMode>;
    /// Nombre d'images en vol demandé (estimation mémoire du profilage).
    #[cfg_attr(not(feature = "flamegraph"), allow(dead_code))]
    fn swapchain_frame_latency(swapchain: &Self::Swapchain) -> u32;
    /// Reconfigure à `width`×`height` et `present_mode` (supporté, vérifié par l'appelant).
    fn configure_swapchain(
        &self,
        swapchain: &mut Self::Swapchain,
        width: u32,
        height: u32,
        present_mode: WindowPresentMode,
    );
    fn acquire(&self, swapchain: &mut Self::Swapchain) -> Acquire<Self::Frame>;
    /// Copie `source` (même format et taille que la swapchain) dans l'image acquise.
    fn copy_texture_to_frame(encoder: &mut Self::Encoder, source: &Self::Texture, frame: &Self::Frame);
    fn submit(&self, encoder: Self::Encoder);
    fn present(&self, frame: Self::Frame);

    /// Efface les tampons d'une surface 3D et les laisse dans l'état « échantillonné »
    /// que les moteurs externes attendent entre deux trames.
    fn init_external_textures(&self, textures: [&Self::Texture; 3]);
}

/// Profilage GPU de la feature `flamegraph` (timestamps de passes, capture profonde).
/// Seul wgpu l'implémente ; les autres backends utilisent [`NoProfiler`].
#[cfg(feature = "flamegraph")]
pub(crate) trait GpuProfiler<G: Gpu>: Default {
    type PassTimestamps;
    type DeepCapture;

    /// Début de trame : synchronise la session, relève les lectures en cours, réserve la
    /// paire de timestamps « submit + present » et arme une capture profonde si demandée.
    fn begin_frame(&mut self, gpu: &G, encoder: &mut G::Encoder) -> Option<Self::DeepCapture>;
    fn pass_timestamps(&mut self, name: &'static str, kind: crate::GpuPassKind) -> Option<Self::PassTimestamps>;
    #[allow(clippy::too_many_arguments)]
    fn record_draw_call(
        capture: &mut Self::DeepCapture,
        kind: crate::DrawCallKind,
        pipeline: &'static str,
        pass: &'static str,
        vertices: Range<u32>,
        instances: Range<u32>,
        bind_groups: u32,
        buffer: Option<crate::DeepCaptureBufferKind>,
        texture: Option<u64>,
        surface: Option<u64>,
    );
    /// Fin de trame, avant `submit`.
    fn end_frame(
        &mut self,
        gpu: &G,
        encoder: &mut G::Encoder,
        capture: Option<Self::DeepCapture>,
        buffers: &[(crate::DeepCaptureBufferKind, &G::Buffer); 7],
        atlas: &super::atlas::Atlas<G>,
        surfaces: &super::surface_registry::SurfaceRegistry<G>,
    );
    fn after_submit(&mut self);
}

/// Profileur vide des backends sans profilage GPU.
#[cfg(feature = "flamegraph")]
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
#[derive(Default)]
pub(crate) struct NoProfiler;

#[cfg(feature = "flamegraph")]
impl<G: Gpu> GpuProfiler<G> for NoProfiler {
    type PassTimestamps = std::convert::Infallible;
    type DeepCapture = std::convert::Infallible;

    fn begin_frame(&mut self, _gpu: &G, _encoder: &mut G::Encoder) -> Option<Self::DeepCapture> {
        None
    }

    fn pass_timestamps(&mut self, _name: &'static str, _kind: crate::GpuPassKind) -> Option<Self::PassTimestamps> {
        None
    }

    fn record_draw_call(
        capture: &mut Self::DeepCapture,
        _kind: crate::DrawCallKind,
        _pipeline: &'static str,
        _pass: &'static str,
        _vertices: Range<u32>,
        _instances: Range<u32>,
        _bind_groups: u32,
        _buffer: Option<crate::DeepCaptureBufferKind>,
        _texture: Option<u64>,
        _surface: Option<u64>,
    ) {
        match *capture {}
    }

    fn end_frame(
        &mut self,
        _gpu: &G,
        _encoder: &mut G::Encoder,
        _capture: Option<Self::DeepCapture>,
        _buffers: &[(crate::DeepCaptureBufferKind, &G::Buffer); 7],
        _atlas: &super::atlas::Atlas<G>,
        _surfaces: &super::surface_registry::SurfaceRegistry<G>,
    ) {
    }

    fn after_submit(&mut self) {}
}
