use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use collections::{FxHashMap, FxHashSet};

#[cfg(feature = "flamegraph")]
use crate::platform::cross::hal::GpuProfiler;
use crate::{
    AtlasTextureId, BackdropFilter, DevicePixels, FilterBoundary, GpuSpecs, LayerKey,
    PrimitiveBatch, Scene, WindowPresentMode, geometry,
    platform::cross::{
        atlas::Atlas,
        hal::{
            Acquire, BindEntry, BindResource, BindingKind, Blend, BufferUsage, Gpu, LayoutEntry, LoadOp,
            PassDesc, PipelineDesc, ShaderId, ShaderStages, TextureUsage, Topology,
        },
        render_context::{RenderContext, ensure_buffer_size},
        slab::{SlabKind, MIN_CLASS},
        slab_gpu::{self, GpuLayerTransform, SlabGpuBuffers, SlabRegistry, SyncPlan},
    },
};

#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalParams {
    viewport_size: [f32; 2],
    premultimated_alpha: u32,
    pad: u32,
}

// Size of `Globals` in the uniform address space, where WGSL rounds a struct's
// byte size up to a multiple of its 16-byte binding alignment.
const _: () = assert!(std::mem::size_of::<GlobalParams>() == 16);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Bounds {
    origin: [f32; 2],
    size: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SurfaceParams {
    bounds: Bounds,
    content_mask: Bounds,
    corner_radii: [f32; 4],
}

/// Per-vertex data uploaded to the GPU for path rendering.
/// Layout must exactly match the `GpuPathVertex` struct in `paths.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuPathVertex {
    xy_position: [f32; 2],         // offset  0
    st_position: [f32; 2],         // offset  8
    hsla: [f32; 4],                // offset 16  (h, s, l, a)
    content_mask_origin: [f32; 2], // offset 32
    content_mask_size: [f32; 2],   // offset 40
} // stride  48

// Stride expected by `array<GpuPathVertex>` in paths.wgsl's storage buffer.
const _: () = assert!(std::mem::size_of::<GpuPathVertex>() == 48);

#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct ColorAdjustments {
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    _padding: [f32; 3],
}

// Size of the `ColorAdjustments` uniform in mono_sprites.wgsl, rounded up to
// the 16-byte uniform binding alignment (the `_padding` field supplies it).
const _: () = assert!(std::mem::size_of::<ColorAdjustments>() == 32);

struct Pipelines<G: Gpu> {
    quads_bind_group_layout: G::BindGroupLayout,
    shadows_bind_group_layout: G::BindGroupLayout,
    backdrop_filters_bind_group_layout: G::BindGroupLayout,
    backdrop_texture_bind_group_layout: G::BindGroupLayout,
    underlines_bind_group_layout: G::BindGroupLayout,
    sprites_bind_group_layout: G::BindGroupLayout,
    mono_sprites_bind_group_layout: G::BindGroupLayout,
    poly_sprites_bind_group_layout: G::BindGroupLayout,
    surfaces_bind_group_layout: G::BindGroupLayout,
    paths_bind_group_layout: G::BindGroupLayout,
    /// Per-layer translate, bound at the highest position of every pipeline
    /// that can draw slab content. Dynamic-offset so one small uniform serves
    /// all layers; slot 0 is permanently zero for legacy draws.
    layer_transform_bind_group_layout: G::BindGroupLayout,
    /// Stored (unlike every other one-off layout here) so a texture-retained
    /// layer's bake pass can build its own globals bind group, sized to the
    /// layer's texture rather than the window (#96) — see
    /// `emit_layer_texture_render`'s render pass.
    globals_bind_group_layout: G::BindGroupLayout,

    globals_bind_group: G::BindGroup,
    color_adjustments_bind_group: G::BindGroup,

    quads_pipeline: G::Pipeline,
    shadows_pipeline: G::Pipeline,
    backdrop_filters_pipeline: G::Pipeline,
    underlines_pipeline: G::Pipeline,
    mono_sprites_pipeline: G::Pipeline,
    poly_sprites_pipeline: G::Pipeline,
    surfaces_pipeline: G::Pipeline,
    paths_pipeline: G::Pipeline,
}

impl<G: Gpu> Pipelines<G> {
    pub fn new(
        gpu: &G,
        format: G::Format,
        premultiplied_alpha: bool,
        globals_buffer: &G::Buffer,
        color_adjustments_buffer: &G::Buffer,
    ) -> Self {
        use ShaderStages::{Fragment, Vertex, VertexFragment};
        let uniform = |binding, visibility| LayoutEntry {
            binding,
            visibility,
            kind: BindingKind::Uniform { dynamic_offset: false, min_size: None },
        };
        let storage = |visibility| LayoutEntry { binding: 0, visibility, kind: BindingKind::Storage };
        let texture = |binding, visibility| LayoutEntry { binding, visibility, kind: BindingKind::Texture };
        let sampler = |binding| LayoutEntry { binding, visibility: Fragment, kind: BindingKind::Sampler };

        let globals_bind_group_layout = gpu.create_bind_group_layout("globals", &[uniform(0, VertexFragment)]);
        let color_adjustments_bind_group_layout =
            gpu.create_bind_group_layout("color_adjustments_bind_group_layout", &[uniform(0, Fragment)]);
        // Vertex+fragment: the vertex stage reads `textureDimensions`.
        let sprites_bind_group_layout =
            gpu.create_bind_group_layout("sprite_bind_group_layout", &[texture(0, VertexFragment), sampler(1)]);
        let layer_transform_bind_group_layout = gpu.create_bind_group_layout(
            "layer_transform_bind_group_layout",
            &[LayoutEntry {
                binding: 0,
                visibility: VertexFragment,
                kind: BindingKind::Uniform {
                    dynamic_offset: true,
                    min_size: Some(std::mem::size_of::<GpuLayerTransform>() as u64),
                },
            }],
        );
        let quads_bind_group_layout = gpu.create_bind_group_layout("quads_bind_group_layout", &[storage(VertexFragment)]);
        let shadows_bind_group_layout =
            gpu.create_bind_group_layout("shadows_bind_group_layout", &[storage(VertexFragment)]);
        let backdrop_filters_bind_group_layout =
            gpu.create_bind_group_layout("backdrop_filters_bind_group_layout", &[storage(VertexFragment)]);
        let backdrop_texture_bind_group_layout =
            gpu.create_bind_group_layout("backdrop_texture_bind_group_layout", &[texture(0, Fragment), sampler(1)]);
        let underlines_bind_group_layout =
            gpu.create_bind_group_layout("underlines_bind_group_layout", &[storage(VertexFragment)]);
        let mono_sprites_bind_group_layout =
            gpu.create_bind_group_layout("Mono sprites bind group layout", &[storage(Vertex)]);
        let poly_sprites_bind_group_layout =
            gpu.create_bind_group_layout("Poly sprites bind group layout", &[storage(VertexFragment)]);
        let surfaces_bind_group_layout = gpu.create_bind_group_layout(
            "surfaces_bind_group_layout",
            &[uniform(0, VertexFragment), texture(1, Fragment), sampler(2)],
        );
        let paths_bind_group_layout = gpu.create_bind_group_layout("paths_bind_group_layout", &[storage(Vertex)]);

        let whole_buffer = |label: &str, layout: &G::BindGroupLayout, buffer: &G::Buffer| {
            gpu.create_bind_group(
                label,
                layout,
                &[BindEntry { binding: 0, resource: BindResource::Buffer { buffer, offset: 0, size: None } }],
            )
        };
        let globals_bind_group = whole_buffer("globals_bind_group", &globals_bind_group_layout, globals_buffer);
        let color_adjustments_bind_group = whole_buffer(
            "color_adjustments_bind_group",
            &color_adjustments_bind_group_layout,
            color_adjustments_buffer,
        );

        let blend = if premultiplied_alpha { Blend::PremultipliedAlpha } else { Blend::Alpha };
        let pipeline = |label: &'static str,
                        shader: ShaderId,
                        vertex_entry: &'static str,
                        fragment_entry: &'static str,
                        topology: Topology,
                        layouts: &[&G::BindGroupLayout]| {
            gpu.create_pipeline(&PipelineDesc {
                label,
                shader,
                vertex_entry,
                fragment_entry,
                topology,
                layouts,
                format,
                blend,
            })
        };
        let strip = Topology::TriangleStrip;
        let (globals, transform) = (&globals_bind_group_layout, &layer_transform_bind_group_layout);

        Self {
            quads_pipeline: pipeline(
                "quads",
                ShaderId::Quads,
                "vs_quad",
                "fs_quad",
                strip,
                &[globals, &quads_bind_group_layout, transform],
            ),
            shadows_pipeline: pipeline(
                "shadows",
                ShaderId::Shadows,
                "vs_shadow",
                "fs_shadow",
                strip,
                &[globals, &shadows_bind_group_layout, transform],
            ),
            backdrop_filters_pipeline: pipeline(
                "backdrop_filters",
                ShaderId::BackdropBlur,
                "vs_backdrop_filter",
                "fs_backdrop_filter",
                strip,
                &[globals, &backdrop_filters_bind_group_layout, &backdrop_texture_bind_group_layout],
            ),
            underlines_pipeline: pipeline(
                "underlines",
                ShaderId::Underlines,
                "vs_underline",
                "fs_underline",
                strip,
                &[globals, &underlines_bind_group_layout, transform],
            ),
            mono_sprites_pipeline: pipeline(
                "mono_sprites",
                ShaderId::MonoSprites,
                "vs_mono_sprite",
                "fs_mono_sprite",
                strip,
                &[
                    globals,
                    &color_adjustments_bind_group_layout,
                    &sprites_bind_group_layout,
                    &mono_sprites_bind_group_layout,
                    transform,
                ],
            ),
            poly_sprites_pipeline: pipeline(
                "poly_sprites",
                ShaderId::PolySprites,
                "vs_poly_sprite",
                "fs_poly_sprite",
                strip,
                &[globals, &sprites_bind_group_layout, &poly_sprites_bind_group_layout, transform],
            ),
            surfaces_pipeline: pipeline(
                "surfaces",
                ShaderId::Surfaces,
                "vs_surface",
                "fs_surface",
                strip,
                &[globals, &surfaces_bind_group_layout],
            ),
            paths_pipeline: pipeline(
                "paths",
                ShaderId::Paths,
                "vs_path",
                "fs_path",
                Topology::TriangleList,
                &[globals, &paths_bind_group_layout, transform],
            ),
            quads_bind_group_layout,
            shadows_bind_group_layout,
            backdrop_filters_bind_group_layout,
            backdrop_texture_bind_group_layout,
            underlines_bind_group_layout,
            sprites_bind_group_layout,
            mono_sprites_bind_group_layout,
            poly_sprites_bind_group_layout,
            surfaces_bind_group_layout,
            paths_bind_group_layout,
            layer_transform_bind_group_layout,
            globals_bind_group_layout,
            globals_bind_group,
            color_adjustments_bind_group,
        }
    }
}

struct RenderingParameters {
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
}

/// Bind state for one frame's slab draws, created only when the frame
/// actually carries spans.
struct SlabDrawGroups<G: Gpu> {
    quads: G::BindGroup,
    shadows: G::BindGroup,
    paths_vertices: G::BindGroup,
    underlines: G::BindGroup,
    mono_sprites: G::BindGroup,
    poly_sprites: G::BindGroup,
    layer_transform: G::BindGroup,
    /// Keyed by `(index, kind)`: `AtlasTextureId` carries no `Hash`, and the
    /// pair is what actually identifies a live page binding.
    sprite_textures: FxHashMap<(u32, crate::AtlasTextureKind), G::BindGroup>,
}

/// One merged stretch of a layer's slab stream awaiting its draw.
#[derive(Clone, Copy)]
struct SlabPendingRun {
    kind: SlabKind,
    texture_id: Option<AtlasTextureId>,
    start: u32,
    count: u32,
}

/// Which render pipeline is currently bound, as a semantic identity the bind
/// tracker can compare without touching wgpu resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DrawPipelineId {
    Quads,
    Shadows,
    Paths,
    Underlines,
    MonoSprites,
    PolySprites,
}

/// Which legacy fixed buffer a bind group wraps, per kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LegacyBuffer {
    Quads,
    Shadows,
    Underlines,
    MonoSprites,
    PolySprites,
    PathVertices,
}

/// Semantic identity of one bindable resource at one bind-group slot.
///
/// Two draws whose ids agree at every slot bind byte-identical GPU state, so
/// the second set is skippable without any pixel effect. Resources this
/// module does not model (per-surface groups, filter composites) are never
/// given an id: those draw paths reset the tracker instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundGroupId {
    Globals,
    ColorAdjustments,
    /// The layer transform uniform at a specific dynamic offset. Legacy
    /// draws always use offset 0 (the identity slot); slab draws use
    /// `slot * stride` for their layer.
    LayerTransform(u32),
    LegacyBuffer(LegacyBuffer),
    SlabStorage(SlabKind),
    SpriteTexture(u32, crate::AtlasTextureKind),
}

/// Upper bound on bind-group slots any pipeline here uses (mono sprites: 5).
const PASS_BIND_SLOTS: usize = 5;

/// What the current render pass has bound, so redundant
/// `set_pipeline`/`set_bind_group` calls can be skipped.
///
/// wgpu offers no way to query a pass's bound state, and driver-side state
/// churn is exactly what this tracks: consecutive same-kind runs (and split
/// legacy batches) re-issue identical binds today. Ids are semantic rather
/// than pointer-based, which keeps skipping sound even when equal-content
/// bind groups are distinct objects. Anything unmodeled must call [`Self::reset`]
/// before tracked draws resume.
#[derive(Default)]
struct PassBindState {
    pipeline: Option<DrawPipelineId>,
    groups: [Option<BoundGroupId>; PASS_BIND_SLOTS],
}

impl PassBindState {
    fn reset(&mut self) {
        self.pipeline = None;
        self.groups = [None; PASS_BIND_SLOTS];
    }

    fn set_pipeline<G: Gpu>(
        &mut self,
        pass: &mut G::Pass<'_>,
        id: DrawPipelineId,
        pipeline: &G::Pipeline,
    ) {
        if self.pipeline == Some(id) {
            return;
        }
        G::set_pipeline(pass, pipeline);
        self.pipeline = Some(id);
    }

    fn set_bind_group<G: Gpu>(
        &mut self,
        pass: &mut G::Pass<'_>,
        index: u32,
        id: BoundGroupId,
        group: &G::BindGroup,
        offsets: &[u32],
    ) {
        let slot = index as usize;
        if slot < PASS_BIND_SLOTS && self.groups[slot] == Some(id) {
            return;
        }
        G::set_bind_group(pass, index, group, offsets);
        if slot < PASS_BIND_SLOTS {
            self.groups[slot] = Some(id);
        }
    }
}

/// One merged slab stretch opened by a span that may stay open across span
/// boundaries until a non-continuing draw forces its flush.
///
/// Merging requires more than instance contiguity: both stretches must draw
/// through the same transform uniform offset, i.e. belong to the same layer.
/// Different layers occupy different transform slots, so their instances can
/// sit adjacent in a kind buffer yet still need separate draws — cross-layer
/// merging is impossible without changing pixels (or relocating resident
/// bytes to co-locate layers, a buffer reshuffle this deliberately avoids).
struct OpenSlabRun {
    key: LayerKey,
    slabs: crate::platform::cross::slab::LayerSlabs,
    transform_slot: u32,
    kind: SlabKind,
    texture_id: Option<AtlasTextureId>,
    start: u32,
    count: u32,
}

impl OpenSlabRun {
    /// Whether `run` continues this stretch: same layer, same kind, same
    /// texture, and exactly contiguous in the layer-wide instance stream.
    fn accepts(
        &self,
        key: LayerKey,
        slabs: &crate::platform::cross::slab::LayerSlabs,
        run: &crate::scene::SlabRun,
    ) -> bool {
        self.key == key
            && &self.slabs == slabs
            && self.kind == run.kind
            && self.texture_id == run.texture_id
            && self.start + self.count == run.start
    }

    fn as_pending(&self) -> SlabPendingRun {
        SlabPendingRun {
            kind: self.kind,
            texture_id: self.texture_id,
            start: self.start,
            count: self.count,
        }
    }
}

/// Frame-to-frame cache of the slab bind state: the six per-kind storage
/// groups plus the transform-uniform group are rebuilt only when
/// [`SlabGpuBuffers`] recreates their buffer, and atlas-page groups refresh
/// only when the referenced-page set changes. On a Clean-only frame this
/// costs a few handle clones instead of eight `create_bind_group` calls.
struct SlabGroupCache<G: Gpu> {
    kind_groups: [Option<G::BindGroup>; SlabKind::COUNT],
    transforms: Option<G::BindGroup>,
    /// The canonical (sorted) page set `sprite_textures` was built from.
    pages: Vec<(u32, crate::AtlasTextureKind)>,
    page_scratch: Vec<(u32, crate::AtlasTextureKind)>,
    sprite_textures: FxHashMap<(u32, crate::AtlasTextureKind), G::BindGroup>,
    #[cfg(test)]
    creations: u64,
}

impl<G: Gpu> Default for SlabGroupCache<G> {
    fn default() -> Self {
        SlabGroupCache {
            kind_groups: std::array::from_fn(|_| None),
            transforms: None,
            pages: Vec::new(),
            page_scratch: Vec::new(),
            sprite_textures: FxHashMap::default(),
            #[cfg(test)]
            creations: 0,
        }
    }
}

impl<G: Gpu> SlabGroupCache<G> {
    #[cfg(test)]
    fn creation_count(&self) -> u64 {
        self.creations
    }

    fn invalidate_kind(&mut self, kind: SlabKind) {
        self.kind_groups[kind.index()] = None;
    }

    fn invalidate_transforms(&mut self) {
        self.transforms = None;
    }

    fn kind_layout(pipelines: &Pipelines<G>, kind: SlabKind) -> &G::BindGroupLayout {
        match kind {
            SlabKind::Quads => &pipelines.quads_bind_group_layout,
            SlabKind::Shadows => &pipelines.shadows_bind_group_layout,
            SlabKind::Paths => &pipelines.paths_bind_group_layout,
            SlabKind::Underlines => &pipelines.underlines_bind_group_layout,
            SlabKind::MonoSprites => &pipelines.mono_sprites_bind_group_layout,
            SlabKind::PolySprites => &pipelines.poly_sprites_bind_group_layout,
        }
    }

    fn ensure_kind_group(
        &mut self,
        gpu: &G,
        pipelines: &Pipelines<G>,
        buffers: &slab_gpu::SlabGpuBuffers<G>,
        kind: SlabKind,
    ) {
        let index = kind.index();
        if self.kind_groups[index].is_some() {
            return;
        }
        let group = gpu.create_bind_group(
            "slab_kind_bind_group",
            Self::kind_layout(pipelines, kind),
            &[
                BindEntry { binding: 0, resource: BindResource::Buffer { buffer: buffers.kind_buffer(kind), offset: 0, size: None } },
            ],
        );
        self.kind_groups[index] = Some(group);
        #[cfg(test)]
        {
            self.creations += 1;
        }
    }

    fn ensure_transforms_group(
        &mut self,
        gpu: &G,
        pipelines: &Pipelines<G>,
        buffers: &slab_gpu::SlabGpuBuffers<G>,
    ) {
        if self.transforms.is_some() {
            return;
        }        let group = gpu.create_bind_group(
            "layer_transform_bind_group",
            &pipelines.layer_transform_bind_group_layout,
            &[
                BindEntry { binding: 0, resource: BindResource::Buffer { buffer: buffers.transforms_buffer(), offset: 0, size: Some(std::mem::size_of::<GpuLayerTransform>() as u64) } },
            ],
        );
        self.transforms = Some(group);
        #[cfg(test)]
        {
            self.creations += 1;
        }
    }

    /// The cached transform-uniform bind group, recreated only after the
    /// uniform buffer was. Legacy draws share this group with slab draws:
    /// both bind the same uniform, selecting slots via dynamic offsets.
    fn transforms_group(
        &mut self,
        gpu: &G,
        pipelines: &Pipelines<G>,
        buffers: &slab_gpu::SlabGpuBuffers<G>,
    ) -> G::BindGroup {
        self.ensure_transforms_group(gpu, pipelines, buffers);
        self.transforms.as_ref().expect("just ensured").clone()
    }

    /// Rebuild the page map only when this frame's referenced-page set differs
    /// from the cached one. Pages whose layers were poisoned by eviction stay
    /// in the map but are never bound: poisoned layers skip their draws before
    /// the texture check runs.
    fn sync_sprite_pages(
        &mut self,
        gpu: &G,
        pipelines: &Pipelines<G>,
        atlas: &Atlas<G>,
        atlas_sampler: &G::Sampler,
        scene: &Scene,
    ) {
        self.page_scratch.clear();
        for span in &scene.layer_slab_spans {
            for run in &span.runs {
                if let Some(texture_id) = run.texture_id {
                    let key = (texture_id.index, texture_id.kind);
                    if !self.page_scratch.contains(&key) {
                        self.page_scratch.push(key);
                    }
                }
            }
        }
        // Sorted so bind-group creation order stays deterministic under fuzzing.
        self.page_scratch.sort_by_key(|&(index, kind)| (index, kind as u8));
        if self.pages == self.page_scratch {
            return;
        }
        self.sprite_textures.clear();
        #[cfg(test)]
        let rebuilt = self.page_scratch.len() as u64;
        for &(texture_index, texture_kind) in &self.page_scratch {
            let tex_info = atlas.get_texture_info(AtlasTextureId {
                index: texture_index,
                kind: texture_kind,
            });
            let group = gpu.create_bind_group(
                "slab_sprite_texture_bind_group",
                &pipelines.sprites_bind_group_layout,
                &[
                    BindEntry { binding: 0, resource: BindResource::Texture(&tex_info.raw_view) },
                    BindEntry { binding: 1, resource: BindResource::Sampler(atlas_sampler) },
                ],
            );
            self.sprite_textures.insert((texture_index, texture_kind), group);
        }
        std::mem::swap(&mut self.pages, &mut self.page_scratch);
        #[cfg(test)]
        {
            self.creations += rebuilt;
        }
    }

    /// The frame's slab bind state, cloned out of the cache.
    fn frame_groups(
        &mut self,
        gpu: &G,
        pipelines: &Pipelines<G>,
        buffers: &slab_gpu::SlabGpuBuffers<G>,
        atlas: &Atlas<G>,
        atlas_sampler: &G::Sampler,
        scene: &Scene,
    ) -> SlabDrawGroups<G> {
        for kind in SlabKind::ALL {
            self.ensure_kind_group(gpu, pipelines, buffers, kind);
        }
        self.ensure_transforms_group(gpu, pipelines, buffers);
        self.sync_sprite_pages(gpu, pipelines, atlas, atlas_sampler, scene);
        let [quads, shadows, paths_vertices, underlines, mono_sprites, poly_sprites] =
            &self.kind_groups;
        SlabDrawGroups {
            quads: quads.as_ref().expect("kind group ensured above").clone(),
            shadows: shadows.as_ref().expect("kind group ensured above").clone(),
            paths_vertices: paths_vertices
                .as_ref()
                .expect("kind group ensured above")
                .clone(),
            underlines: underlines
                .as_ref()
                .expect("kind group ensured above")
                .clone(),
            mono_sprites: mono_sprites
                .as_ref()
                .expect("kind group ensured above")
                .clone(),
            poly_sprites: poly_sprites
                .as_ref()
                .expect("kind group ensured above")
                .clone(),
            layer_transform: self.transforms.as_ref().expect("transform group ensured above").clone(),
            sprite_textures: self.sprite_textures.clone(),
        }
    }
}

/// Bind state for one frame's slab draws, created only when the frame
/// actually carries slots. Free-standing so the GPU-tier tests drive the
/// exact production construction; the frame path uses [`SlabGroupCache`].
#[cfg(test)]
fn build_slab_draw_groups<G: Gpu>(
    gpu: &G,
    pipelines: &Pipelines<G>,
    buffers: &slab_gpu::SlabGpuBuffers<G>,
    atlas: &Atlas<G>,
    atlas_sampler: &G::Sampler,
    layer_transform_bind_group: &G::BindGroup,
    scene: &Scene,
) -> SlabDrawGroups<G> {
    let buffer_group = |label: &'static str,
                        layout: &G::BindGroupLayout,
                        buffer: &G::Buffer| -> G::BindGroup {
        gpu.create_bind_group(
            label,
            layout,
            &[
                BindEntry { binding: 0, resource: BindResource::Buffer { buffer: buffer, offset: 0, size: None } },
            ],
        )
    };

    let mut sprite_textures: FxHashMap<(u32, crate::AtlasTextureKind), G::BindGroup> =
        FxHashMap::default();
    let mut textures_this_frame: Vec<(u32, crate::AtlasTextureKind)> = Vec::new();
    for span in &scene.layer_slab_spans {
        for run in &span.runs {
            if let Some(texture_id) = run.texture_id {
                let key = (texture_id.index, texture_id.kind);
                if !textures_this_frame.contains(&key) {
                    textures_this_frame.push(key);
                }
            }
        }
    }
    // Sorted so bind-group creation order is deterministic under fuzzing.
    textures_this_frame.sort_by_key(|&(index, kind)| (index, kind as u8));
    for (texture_index, texture_kind) in textures_this_frame {
        let texture_id = AtlasTextureId {
            index: texture_index,
            kind: texture_kind,
        };
        let tex_info = atlas.get_texture_info(texture_id);
        let group = gpu.create_bind_group(
            "slab_sprite_texture_bind_group",
            &pipelines.sprites_bind_group_layout,
            &[
                BindEntry { binding: 0, resource: BindResource::Texture(&tex_info.raw_view) },
                BindEntry { binding: 1, resource: BindResource::Sampler(atlas_sampler) },
            ],
        );
        sprite_textures.insert((texture_index, texture_kind), group);
    }

    SlabDrawGroups {
        quads: buffer_group(
            "slab_quads_bind_group",
            &pipelines.quads_bind_group_layout,
            buffers.kind_buffer(SlabKind::Quads),
        ),
        shadows: buffer_group(
            "slab_shadows_bind_group",
            &pipelines.shadows_bind_group_layout,
            buffers.kind_buffer(SlabKind::Shadows),
        ),
        paths_vertices: buffer_group(
            "slab_paths_vertices_bind_group",
            &pipelines.paths_bind_group_layout,
            buffers.kind_buffer(SlabKind::Paths),
        ),
        underlines: buffer_group(
            "slab_underlines_bind_group",
            &pipelines.underlines_bind_group_layout,
            buffers.kind_buffer(SlabKind::Underlines),
        ),
        mono_sprites: buffer_group(
            "slab_mono_sprites_bind_group",
            &pipelines.mono_sprites_bind_group_layout,
            buffers.kind_buffer(SlabKind::MonoSprites),
        ),
        poly_sprites: buffer_group(
            "slab_poly_sprites_bind_group",
            &pipelines.poly_sprites_bind_group_layout,
            buffers.kind_buffer(SlabKind::PolySprites),
        ),
        layer_transform: layer_transform_bind_group.clone(),
        sprite_textures,
    }
}

/// Instances a legacy primitive batch would draw. Zero-instance batches
/// (the empty split halves `FrameBatchIterator` queues around spans) draw
/// nothing, so they must not break an open slab stretch's adjacency.
fn primitive_batch_instance_count(batch: &PrimitiveBatch<'_>) -> u32 {
    match batch {
        PrimitiveBatch::Quads(quads) => quads.len() as u32,
        PrimitiveBatch::Shadows(shadows) => shadows.len() as u32,
        PrimitiveBatch::Paths(paths) => paths.iter().map(|p| p.vertices.len() as u32).sum(),
        PrimitiveBatch::Underlines(underlines) => underlines.len() as u32,
        PrimitiveBatch::MonochromeSprites { sprites, .. } => sprites.len() as u32,
        PrimitiveBatch::PolychromeSprites { sprites, .. } => sprites.len() as u32,
        PrimitiveBatch::Surfaces(surfaces) => surfaces.len() as u32,
        PrimitiveBatch::BackdropFilters(backdrop_filters) => backdrop_filters.len() as u32,
        // One marker, one (degenerate) draw.
        PrimitiveBatch::FilterBoundary(_) => 1,
    }
}

/// One shared pass over the scene's spans, grouping every referenced atlas
/// page by owning layer.
///
/// Replaces the per-synced-layer rescan of all spans (O(spans²) with the
/// filter inside the sync loop); residency bookkeeping consumes the result
/// without changing what is recorded per layer.
fn collect_referenced_pages_by_layer(
    scene: &Scene,
) -> FxHashMap<LayerKey, Vec<(u32, crate::AtlasTextureKind)>> {
    let mut pages: FxHashMap<LayerKey, Vec<(u32, crate::AtlasTextureKind)>> =
        FxHashMap::default();
    for span in &scene.layer_slab_spans {
        for run in &span.runs {
            if let Some(texture_id) = run.texture_id {
                pages
                    .entry(span.key)
                    .or_default()
                    .push((texture_id.index, texture_id.kind));
            }
        }
    }
    pages
}

fn append_packed_kind_bytes(
    scratch: &mut Vec<u8>,
    kind: SlabKind,
    packed: &crate::scene_pack::PackedLayer,
) {
    match kind {
        SlabKind::Quads => scratch.extend_from_slice(bytemuck::cast_slice(&packed.quads)),
        SlabKind::Shadows => scratch.extend_from_slice(bytemuck::cast_slice(&packed.shadows)),
        // Path slabs hold the flattened GpuPathVertex stream (color and mask
        // baked per vertex), exactly what the legacy upload builds.
        SlabKind::Paths => {
            for path in &packed.paths {
                let color = path.color.solid;
                let cm = &path.content_mask.bounds;
                let cm_origin = [cm.origin.x.0, cm.origin.y.0];
                let cm_size = [cm.size.width.0, cm.size.height.0];
                for vertex in &path.vertices {
                    scratch.extend_from_slice(bytemuck::bytes_of(&GpuPathVertex {
                        xy_position: [vertex.xy_position.x.0, vertex.xy_position.y.0],
                        st_position: [vertex.st_position.x, vertex.st_position.y],
                        hsla: [color.h, color.s, color.l, color.a],
                        content_mask_origin: cm_origin,
                        content_mask_size: cm_size,
                    }));
                }
            }
        }
        SlabKind::Underlines => {
            scratch.extend_from_slice(bytemuck::cast_slice(&packed.underlines))
        }
        SlabKind::MonoSprites => {
            scratch.extend_from_slice(bytemuck::cast_slice(&packed.mono_sprites))
        }
        SlabKind::PolySprites => {
            scratch.extend_from_slice(bytemuck::cast_slice(&packed.poly_sprites))
        }
    }
}

impl RenderingParameters {
    fn from_env() -> Self {
        use std::env;

        let gamma = env::var("ZED_FONTS_GAMMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.8_f32)
            .clamp(1.0, 2.2);
        let gamma_ratios = crate::platform::get_gamma_correction_ratios(gamma);
        let grayscale_enhanced_contrast = env::var("ZED_FONTS_GRAYSCALE_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0_f32)
            .max(0.0);

        Self {
            gamma_ratios,
            grayscale_enhanced_contrast,
        }
    }
}

/// Maximum nesting depth supported for CSS-style content `filter` groups
/// (`with_filter_layer`). Groups nested deeper than this are painted inline,
/// unisolated and unblurred, rather than allocating unbounded offscreen textures.
const MAX_FILTER_DEPTH: usize = 4;

/// How many frames a layer texture may go unreferenced before the cache
/// drops it and posts a re-record request for its layer (#96). Generous, so
/// a briefly-scrolled-away buffer never thrashes.
const LAYER_TEXTURE_IDLE_FRAMES: u64 = 240;

/// One texture-retained layer's persistent offscreen texture (#96).
struct LayerTextureEntry<G: Gpu> {
    /// Kept with its view: native backends do not tie a texture's lifetime to its views.
    _texture: G::Texture,
    view: G::TextureView,
    width: u32,
    height: u32,
    /// The owning layer, for re-record requests when the entry dies.
    key: crate::LayerKey,
    /// The content generation baked in; compared against span tokens.
    content_token: u64,
    last_used_frame: u64,
}

/// Allocates the pool of full-surface-sized offscreen textures that
/// content-filter groups render into, one per supported nesting depth.
fn create_filter_group_textures<G: Gpu>(
    gpu: &G,
    width: u32,
    height: u32,
    format: G::Format,
) -> (Vec<G::Texture>, Vec<G::TextureView>) {
    (0..MAX_FILTER_DEPTH)
        .map(|_| {
            let texture = gpu.create_texture(
                "filter_group_texture",
                width,
                height,
                format,
                TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED,
            );
            let view = G::create_view(&texture);
            (texture, view)
        })
        .unzip()
}

pub struct Renderer<G: Gpu> {
    context: Arc<RenderContext<G>>,
    swapchain: G::Swapchain,
    atlas_sampler: G::Sampler,
    surface_sampler: G::Sampler,
    atlas: Arc<Atlas<G>>,
    pipelines: Pipelines<G>,
    rendering_parameters: RenderingParameters,

    // Persistent framebuffer for browser-canvas-style blitting
    persistent_framebuffer: Option<G::Texture>,
    persistent_framebuffer_view: Option<G::TextureView>,

    // Backdrop blur texture for capturing framebuffer content
    backdrop_blur_texture: Option<G::Texture>,
    backdrop_blur_texture_view: Option<G::TextureView>,
    backdrop_blur_sampler: G::Sampler,
    glass_backdrop_gen: AtomicU32,
    glass_backdrop_copied_gen: u32,

    // Pool of offscreen textures that content-filter groups (`with_filter_layer`)
    // render into so their content can be blurred and composited as a unit.
    // Indexed by nesting depth, sized to `MAX_FILTER_DEPTH`.
    group_textures: Vec<G::Texture>,
    group_views: Vec<G::TextureView>,

    // Per-layer persistent slab state (spec #94). The registry owns the
    // allocator and residency decisions; the buffers are the grow-only
    // storage the registry's ranges index into. Clean layers upload nothing;
    // dirty layers upload exactly their own slab.
    slab_registry: SlabRegistry,
    slab_buffers: SlabGpuBuffers<G>,
    /// Frame-to-frame cache of slab bind groups; invalidated when the
    /// underlying buffers are recreated (see `ensure_slab_buffer_capacities`).
    slab_group_cache: SlabGroupCache<G>,
    /// Reusable byte scratch for dirty-layer slab uploads.
    slab_upload_scratch: Vec<u8>,
    /// Reusable storage for per-frame dirty transform drains.
    transform_scratch: Vec<(u32, GpuLayerTransform)>,
    /// Persistent per-layer textures for texture-retained layers (#96),
    /// keyed by dense [`crate::LayerId`]. Created on first use, sampled by the
    /// surfaces pipeline on every clean composite frame, and dropped (with a
    /// re-record request) on resize or idleness.
    layer_textures: FxHashMap<crate::LayerId, LayerTextureEntry<G>>,
    /// Monotonic frame counter for `layer_textures` idle eviction.
    layer_texture_frame: u64,
    /// This window's frame-constant uniforms. Per-renderer on purpose: the
    /// upload dedup below compares against the last value this renderer
    /// pushed, which is only sound against a buffer no other window writes.
    /// A shared buffer let one window's viewport overwrite another's while
    /// the owner's dedup guard skipped the rewrite — content then rendered
    /// scaled/shifted against the wrong viewport until its own next resize.
    globals_buffer: G::Buffer,
    color_adjustments_buffer: G::Buffer,
    // Last values pushed into the frame-constant uniform buffers, so an idle
    // window issues zero `write_buffer` calls at all.
    uploaded_globals: Option<GlobalParams>,
    uploaded_color_adjustments: Option<ColorAdjustments>,

    /// GPU timestamps and deep capture of the `flamegraph` feature (issues #57, #60).
    #[cfg(feature = "flamegraph")]
    profiler: parking_lot::Mutex<G::Profiler>,
}

#[cfg(feature = "wgpu")]
pub type WgpuRenderer = Renderer<super::hal::wgpu::WgpuGpu>;

/// Mode de présentation voulu : `GPUI_PRESENT_MODE=mailbox|fifo|immediate`, sinon
/// `Immediate` si `GPUI_DISABLE_VSYNC` est posé, sinon `Fifo` (vsync).
fn requested_present_mode() -> WindowPresentMode {
    std::env::var("GPUI_PRESENT_MODE")
        .ok()
        .and_then(|mode| match mode.to_lowercase().as_str() {
            "mailbox" => Some(WindowPresentMode::Mailbox),
            "immediate" => Some(WindowPresentMode::Immediate),
            "fifo" => Some(WindowPresentMode::Fifo),
            _ => None,
        })
        .unwrap_or_else(|| {
            if std::env::var("GPUI_DISABLE_VSYNC").is_ok() {
                WindowPresentMode::Immediate
            } else {
                WindowPresentMode::Fifo
            }
        })
}

impl<G: Gpu> Renderer<G> {
    pub fn new<WindowHandle>(
        context: Arc<RenderContext<G>>,
        window: WindowHandle,
        atlas: Arc<Atlas<G>>,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Self>
    where
        WindowHandle: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
    {
        let swapchain = context.create_swapchain(
            window.window_handle()?.as_raw(),
            window.display_handle()?.as_raw(),
            width,
            height,
        )?;
        let format = G::swapchain_format(&swapchain);

        let supported_present_modes = context.supported_present_modes(&swapchain);
        crate::present_mode::set_supported_present_modes(supported_present_modes.iter().copied());
        let present_mode = Some(requested_present_mode())
            .filter(|mode| supported_present_modes.contains(mode))
            .unwrap_or(WindowPresentMode::Fifo);
        crate::present_mode::set_window_present_mode(present_mode);
        #[cfg(feature = "flamegraph")]
        crate::set_present_mode(match present_mode {
            WindowPresentMode::Fifo => crate::PresentMode::Fifo,
            WindowPresentMode::Mailbox => crate::PresentMode::Mailbox,
            WindowPresentMode::Immediate => crate::PresentMode::Immediate,
        });

        let atlas_sampler = context.create_linear_sampler("atlas_sampler");
        let surface_sampler = context.create_linear_sampler("surface_sampler");
        let backdrop_blur_sampler = context.create_linear_sampler("backdrop_blur_sampler");

        let globals_buffer = context.create_buffer(
            "Globals Buffer",
            std::mem::size_of::<[f32; 4]>() as u64,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        );
        let color_adjustments_buffer = context.create_buffer(
            "Color Adjustments Buffer",
            1024 * 16,
            BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::UNIFORM,
        );

        let pipelines = Pipelines::new(
            context.gpu.as_ref(),
            format,
            G::swapchain_premultiplied(&swapchain),
            &globals_buffer,
            &color_adjustments_buffer,
        );

        // Create persistent framebuffer for browser-canvas-style blitting
        let persistent_framebuffer = context.create_texture(
            "persistent_framebuffer",
            width,
            height,
            format,
            TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED | TextureUsage::COPY_SRC,
        );
        let persistent_framebuffer_view = G::create_view(&persistent_framebuffer);

        // Create backdrop blur texture for capturing framebuffer content
        let backdrop_blur_texture = context.create_texture(
            "backdrop_blur_texture",
            width,
            height,
            format,
            TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED | TextureUsage::COPY_DST,
        );
        let backdrop_blur_texture_view = G::create_view(&backdrop_blur_texture);

        let (group_textures, group_views) =
            create_filter_group_textures(context.gpu.as_ref(), width, height, format);

        let slab_buffers = SlabGpuBuffers::new(context.gpu.as_ref(), context.min_uniform_offset_alignment());

        let mut renderer = Self {
            context: context.clone(),
            swapchain,
            atlas,
            atlas_sampler,
            surface_sampler,
            backdrop_blur_sampler,
            pipelines,
            rendering_parameters: RenderingParameters::from_env(),
            persistent_framebuffer: Some(persistent_framebuffer),
            persistent_framebuffer_view: Some(persistent_framebuffer_view),
            backdrop_blur_texture: Some(backdrop_blur_texture),
            backdrop_blur_texture_view: Some(backdrop_blur_texture_view),
            glass_backdrop_gen: AtomicU32::new(0),
            glass_backdrop_copied_gen: 0,
            group_textures,
            group_views,
            slab_registry: SlabRegistry::new(),
            slab_buffers,
            slab_group_cache: SlabGroupCache::default(),
            slab_upload_scratch: Vec::new(),
            transform_scratch: Vec::new(),
            layer_textures: FxHashMap::default(),
            layer_texture_frame: 0,
            globals_buffer,
            color_adjustments_buffer,
            uploaded_globals: None,
            uploaded_color_adjustments: None,
            #[cfg(feature = "flamegraph")]
            profiler: Default::default(),
        };
        // Configure here: the initial same-size resize is skipped.
        renderer.reconfigure_surface(width, height, present_mode);
        Ok(renderer)
    }

    /// Reserve a timestamp-write pair against the current flamegraph GPU
    /// capture generation, if one is active and actively recording a frame.
    /// Returns `None` (cheaply, no wgpu resource allocation) otherwise, e.g.
    /// when no capture is running, or when `blit_surfaces_direct` runs
    /// outside a `draw()`-initiated frame.
    #[cfg(feature = "flamegraph")]
    fn reserve_gpu_timestamps(
        &self,
        name: &'static str,
        pass_kind: crate::GpuPassKind,
    ) -> Option<<G::Profiler as GpuProfiler<G>>::PassTimestamps> {
        self.profiler.lock().pass_timestamps(name, pass_kind)
    }

    /// Reconfigure the swapchain, excluding external render threads for the
    /// duration.
    ///
    /// `Surface::configure` waits for the device to go idle and fails fatally
    /// with `GpuWaitTimeout` if anything submits during that wait. Surfaces
    /// handed to external render threads (`WgpuSurfaceHandle`) share this
    /// device and queue, so the exclusive guard here is what keeps a window
    /// resize from racing them -- see `WgpuContext::gpu_submit_lock`'s doc
    /// comment for the full mechanism.
    ///
    /// All swapchain configuration in this renderer must go through here.
    fn reconfigure_surface(&mut self, width: u32, height: u32, present_mode: WindowPresentMode) {
        let _exclusive = self.context.gpu_submit_lock.write();
        self.context.configure_swapchain(&mut self.swapchain, width, height, present_mode);
    }

    fn surface_size(&self) -> (u32, u32) {
        G::swapchain_size(&self.swapchain)
    }

    // -------------------------------------------------------------------
    // Layer slabs (spec #94).
    // -------------------------------------------------------------------

    /// Grow slab storage to cover the allocator's arenas. Recreation loses
    /// bytes, so both events void residency: every layer re-uploads on its
    /// next sync rather than drawing against orphaned buffers.
    fn ensure_slab_buffer_capacities(&mut self) {
        let mut recreated_any_kind = false;
        for kind in SlabKind::ALL {
            let elements = self.slab_registry.arena_element_capacity(kind).max(MIN_CLASS);
            if self
                .slab_buffers
                .ensure_kind_capacity(self.context.gpu.as_ref(), kind, elements)
            {
                recreated_any_kind = true;
                // The bind group still targets the orphaned buffer until it is
                // rebuilt against the new one.
                self.slab_group_cache.invalidate_kind(kind);
            }
        }
        if recreated_any_kind {
            log::info!("slab buffer grew; re-uploading all resident layers");
            self.slab_registry.invalidate_all_residency();
        }

        let slots_needed = self.slab_registry.transforms_shared().slot_count() + 1;
        if self
            .slab_buffers
            .ensure_transform_capacity(self.context.gpu.as_ref(), slots_needed)
        {
            self.slab_registry.mark_all_transforms_dirty();
            self.slab_group_cache.invalidate_transforms();
        }
    }

    // -------------------------------------------------------------------
    // Layer textures (#96).
    // -------------------------------------------------------------------

    /// Make sure a persistent texture exists for `target` at the right size,
    /// creating (or recreating) it on first use and on resize. Returns the
    /// pixel size, or `None` for a degenerate extent.
    fn ensure_layer_texture(
        &mut self,
        target: &crate::scene::LayerTextureTarget,
    ) -> Option<(u32, u32)> {
        let width = target.texture_bounds.size.width.0.max(0.0).ceil() as u32;
        let height = target.texture_bounds.size.height.0.max(0.0).ceil() as u32;
        if width == 0 || height == 0 {
            return None;
        }
        if let Some(entry) = self.layer_textures.get(&target.layer_id) {
            if entry.width == width && entry.height == height {
                return Some((width, height));
            }
        }

        let texture = self.context.create_texture(
            "layer_texture",
            width,
            height,
            G::swapchain_format(&self.swapchain),
            TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED,
        );
        let view = G::create_view(&texture);
        crate::render_stats::count("layer: texture allocated");
        log::trace!(
            "layer texture for {:?} (key {:?}) allocated at {width}x{height}",
            target.layer_id,
            target.key
        );
        self.layer_textures.insert(
            target.layer_id,
            LayerTextureEntry {
                _texture: texture,
                view,
                width,
                height,
                key: target.key,
                content_token: target.content_token,
                last_used_frame: self.layer_texture_frame,
            },
        );
        Some((width, height))
    }

    /// Drop layer textures that went unreferenced for
    /// [`LAYER_TEXTURE_IDLE_FRAMES`] frames, posting re-record requests so the
    /// layers re-bake on their next composite instead of sampling a missing
    /// texture.
    fn gc_layer_textures(&mut self) {
        let frame = self.layer_texture_frame;
        let stale: Vec<crate::LayerId> = self
            .layer_textures
            .iter()
            .filter(|(_, entry)| frame.saturating_sub(entry.last_used_frame) > LAYER_TEXTURE_IDLE_FRAMES)
            .map(|(id, _)| *id)
            .collect();
        for id in stale {
            if let Some(entry) = self.layer_textures.remove(&id) {
                log::trace!("layer texture for {id:?} (key {:?}) evicted idle", entry.key);
                self.slab_registry.request_rerecord([entry.key]);
            }
        }
    }

    /// One sync decision per layer per frame. Clean layers cost zero
    /// `write_buffer`s; a dirty layer uploads exactly its own slab ranges.
    /// Runs before the render pass so the pass below only draws.
    fn resolve_slab_spans(&mut self, scene: &Scene) {
        profiling::scope!("wgpui: slab sync");
        let pages_by_layer = collect_referenced_pages_by_layer(scene);
        let mut synced_layers: FxHashSet<LayerKey> = FxHashSet::default();
        for span in &scene.layer_slab_spans {
            if self.slab_registry.is_awaiting_rerecord(span.key) {
                continue;
            }
            if !synced_layers.insert(span.key) {
                continue;
            }
            match self
                .slab_registry
                .plan_sync(span.key, span.content_token, span.totals)
            {
                Err(error) => {
                    slab_gpu::report_sync_overflow(error);
                    self.slab_registry.request_rerecord([span.key]);
                    continue;
                }
                Ok(SyncPlan::Clean) => self.slab_registry.note_span_drawn_clean(),
                Ok(SyncPlan::UploadAllOccupied) => {
                    self.upload_layer_slab_bytes(scene, span.key);
                }
            }
            // A texture-retained layer's pack was built at the texture origin,
            // so its spans draw in texture space with an identity translate
            // (#96); everything else translates from layer-local to window.
            if let Some(target) = &span.texture {
                if self.ensure_layer_texture(target).is_none() {
                    self.slab_registry.request_rerecord([span.key]);
                    continue;
                }
                self.slab_registry.set_layer_translate(span.key, [0.0, 0.0]);
            } else {
                self.slab_registry.set_layer_translate(span.key, span.origin);
            }

            if let Some(pages) = pages_by_layer.get(&span.key) {
                self.slab_registry.note_referenced_pages(span.key, pages.iter().copied());
            }
        }

        let transforms_buffer = self.slab_buffers.transforms_buffer().clone();
        let stride = self.slab_buffers.transform_slot_stride;
        let mut dirty_transforms = std::mem::take(&mut self.transform_scratch);
        self.slab_registry.take_dirty_transforms_into(&mut dirty_transforms);
        for &(slot, transform) in &dirty_transforms {
            self.context.write_buffer(
                &transforms_buffer,
                slot as u64 * stride,
                bytemuck::bytes_of(&transform),
            );
        }
        self.transform_scratch = dirty_transforms;
    }

    /// Upload every occupied kind range of one layer from its spans' packed
    /// arrays, at byte offsets derived from element-unit bases.
    ///
    /// The byte scratch is renderer-owned and reused across dirty syncs: a
    /// steady-state window re-uploads some layer every few frames, so a fresh
    /// `Vec` per sync would churn the allocator for bytes it just freed.
    fn upload_layer_slab_bytes(&mut self, scene: &Scene, key: LayerKey) {
        let Some(slabs) = self.slab_registry.entry_slabs(key) else {
            return;
        };
        let mut scratch = std::mem::take(&mut self.slab_upload_scratch);
        for kind in SlabKind::ALL {
            scratch.clear();
            for span in &scene.layer_slab_spans {
                if span.key != key {
                    continue;
                }
                append_packed_kind_bytes(&mut scratch, kind, &span.packed);
            }
            let range = slabs.slab(kind);
            if scratch.is_empty() || range.is_empty() {
                continue;
            }
            let stride = slab_gpu::instance_stride(kind);
            debug_assert_eq!(
                scratch.len() as u64,
                range.count as u64 * stride,
                "packed byte stream must match the reserved range"
            );
            self.context.write_buffer(
                self.slab_buffers.kind_buffer(kind),
                range.byte_offset(stride),
                &scratch,
            );
            crate::render_stats::add(slab_gpu::COUNTER_BYTES_UPLOADED, scratch.len() as u64);
        }
        self.slab_upload_scratch = scratch;
    }

    /// Bind groups for one frame's slab draws: per-kind storage bindings over
    /// the slab buffers plus every distinct atlas texture any span references.
    ///
    /// Sourced from [`Self::slab_group_cache`], so a Clean-only frame clones
    /// handles instead of rebuilding bind groups; the cache is invalidated by
    /// [`Self::ensure_slab_buffer_capacities`] whenever a backing buffer is
    /// recreated, and page groups refresh only when the page set changes.
    fn slab_draw_groups_for_frame(&mut self, scene: &Scene) -> SlabDrawGroups<G> {
        self.slab_group_cache.frame_groups(
            self.context.gpu.as_ref(),
            &self.pipelines,
            &self.slab_buffers,
            &self.atlas,
            &self.atlas_sampler,
            scene,
        )
    }

    /// Draw one spliced layer's runs as instanced draws into the batch
    /// stream, merging adjacent same-kind same-state runs where free —
    /// including stretches that continue across consecutive span boundaries
    /// of the same layer via `open_run`.
    ///
    /// Any state that would produce wrong pixels — an entry awaiting a
    /// re-record after eviction, missing registry state — skips the draws
    /// loudly instead; the posted re-record request rebuilds the layer.
    #[allow(clippy::too_many_arguments)]
    fn draw_layer_slab_span(
        &self,
        pass: &mut G::Pass<'_>,
        scene: &Scene,
        span_index: usize,
        groups: &SlabDrawGroups<G>,
        transform_slot_stride: u64,
        state: &mut PassBindState,
        open_run: &mut Option<OpenSlabRun>,
    ) {
        let Some(span) = scene.layer_slab_spans.get(span_index) else {
            debug_assert!(false, "frame_batches yielded an out-of-range span index");
            return;
        };
        if self.slab_registry.is_awaiting_rerecord(span.key) {
            self.slab_registry.note_span_skipped_awaiting_rerecord();
            return;
        }
        let Some(slabs) = self.slab_registry.entry_slabs(span.key) else {
            slab_gpu::report_missing_slab_state(&self.slab_registry, span.key);
            return;
        };
        let Some(transform_slot) = self.slab_registry.transform_slot(span.key) else {
            slab_gpu::report_missing_slab_state(&self.slab_registry, span.key);
            return;
        };
        for run in &span.runs {
            if let Some(texture_id) = run.texture_id {
                let key = (texture_id.index, texture_id.kind);
                if !groups.sprite_textures.contains_key(&key) {
                    // The atlas page died between resolution and draw; treat
                    // it exactly like eviction poisoning.
                    self.slab_registry.request_rerecord([span.key]);
                    self.slab_registry.note_span_skipped_awaiting_rerecord();
                    return;
                }
            }
        }

        for run in &span.runs {
            // Continue an open stretch: same layer, same kind, same texture,
            // exactly contiguous in the layer-wide instance stream. Their
            // instances draw as one call because everything bound between
            // them is identical.
            if let Some(open) = open_run
                && open.accepts(span.key, &slabs, run)
            {
                open.count += run.count;
                continue;
            }
            flush_open_slab_run(
                &self.pipelines,
                transform_slot_stride,
                pass,
                groups,
                state,
                open_run,
                &self.pipelines.globals_bind_group,
            );
            *open_run = Some(OpenSlabRun {
                key: span.key,
                slabs,
                transform_slot,
                kind: run.kind,
                texture_id: run.texture_id,
                start: run.start,
                count: run.count,
            });
        }
    }

    /// Draw one texture-retained layer's span runs into the CURRENT pass —
    /// the layer-texture pass the caller set up (#96). No cross-run merging:
    /// a texture bake is a refill-frame path, and the pass is torn down right
    /// after.
    fn draw_texture_span_runs(
        &self,
        pass: &mut G::Pass<'_>,
        span: &crate::scene::LayerSlabSpan,
        groups: &SlabDrawGroups<G>,
        transform_slot_stride: u64,
        state: &mut PassBindState,
        globals_bind_group: &G::BindGroup,
    ) {
        if self.slab_registry.is_awaiting_rerecord(span.key) {
            self.slab_registry.note_span_skipped_awaiting_rerecord();
            return;
        }
        let Some(slabs) = self.slab_registry.entry_slabs(span.key) else {
            slab_gpu::report_missing_slab_state(&self.slab_registry, span.key);
            return;
        };
        let Some(transform_slot) = self.slab_registry.transform_slot(span.key) else {
            slab_gpu::report_missing_slab_state(&self.slab_registry, span.key);
            return;
        };
        for run in &span.runs {
            if let Some(texture_id) = run.texture_id {
                let key = (texture_id.index, texture_id.kind);
                if !groups.sprite_textures.contains_key(&key) {
                    // The atlas page died between resolution and draw; treat
                    // it exactly like eviction poisoning.
                    self.slab_registry.request_rerecord([span.key]);
                    self.slab_registry.note_span_skipped_awaiting_rerecord();
                    return;
                }
            }
            let mut one = Some(OpenSlabRun {
                key: span.key,
                slabs,
                transform_slot,
                kind: run.kind,
                texture_id: run.texture_id,
                start: run.start,
                count: run.count,
            });
            flush_open_slab_run(
                &self.pipelines,
                transform_slot_stride,
                pass,
                groups,
                state,
                &mut one,
                globals_bind_group,
            );
        }
    }

    /// Drain this renderer's pending slab re-record requests.
    ///
    /// Called by the owning window at the start of its draw. Requests are
    /// per-renderer on purpose: a process-global queue would let another
    /// window's draw consume them, and a request that never reaches its owner
    /// leaves that owner's poisoned layers skipping draws indefinitely.
    pub fn take_rerecord_requests(&mut self) -> Vec<crate::LayerKey> {
        self.slab_registry.take_rerecord_requests()
    }

    pub fn draw(&mut self, scene: &Scene) {
        profiling::scope!("wgpui: renderer draw");
        log::trace!("Renderer::draw: starting frame");

        let mut command_encoder = self.context.create_encoder("main");

        // Flamegraph GPU capture (issues #57, #60): sync the query manager with
        // the active session, poll earlier readbacks, bracket the whole encoder
        // with a GpuSubmitPresent span and arm a deep capture if one was requested.
        #[cfg(feature = "flamegraph")]
        let mut deep_capture_recorder =
            self.profiler.get_mut().begin_frame(self.context.gpu.as_ref(), &mut command_encoder);

        // Slab residency upkeep (spec #94): eviction poisoning first — stale
        // tile ids must never reach the GPU this frame — then arena growth,
        // then advisory compaction while the frame is otherwise idle.
        let evicted_pages = self.atlas.drain_destroyed_pages();
        if !evicted_pages.is_empty() {
            let poisoned = self.slab_registry.poison_on_evicted_pages(&evicted_pages);
            if !poisoned.is_empty() {
                self.slab_registry.request_rerecord(poisoned);
            }
        }
        self.slab_registry.begin_frame();
        self.ensure_slab_buffer_capacities();
        // Layer-texture upkeep (#96): age the frame counter, drop entries that
        // have gone unreferenced too long (posting re-record requests), so an
        // evicted buffer's texture does not hold VRAM forever.
        self.layer_texture_frame += 1;
        self.gc_layer_textures();
        // Advisory compaction, gated three ways (all scheduling, never
        // correctness): kill switch, utilization heuristic, and — since the
        // arenas never shrink and GC keeps utilization low regardless — a
        // zero-move backoff plus an uploads-in-flight deferral so an idle
        // window stops rebuilding empty plans every frame.
        if slab_gpu::compaction_enabled()
            && self.slab_registry.should_compact(0.35)
            && self.slab_registry.compaction_gate_open()
        {
            let plan = self.slab_registry.compaction_plan();
            let moves = self.slab_registry.apply_compaction(&plan);
            if moves.is_empty() {
                self.slab_registry.note_zero_move_plan();
            } else {
                self.slab_registry.note_moves_applied();
            }
            for (kind, src, dst) in moves {
                let stride = slab_gpu::instance_stride(kind);
                G::copy_buffer_to_buffer(
                    &mut command_encoder,
                    self.slab_buffers.kind_buffer(kind),
                    src.byte_offset(stride),
                    self.slab_buffers.kind_buffer(kind),
                    dst.byte_offset(stride),
                    src.count as u64 * stride,
                );
            }
        }

        // keep track of which surface ids we rendered this frame
        let mut seen_surfaces: Vec<crate::platform::cross::surface_registry::SurfaceId> =
            Vec::new();

        // CRITICAL: Keep surface views alive until after the render pass ends
        // The bind groups reference these views, so they must not be dropped early
        let mut surface_views: Vec<G::TextureView> = Vec::new();
        // Surface bind groups also reference per-surface params buffers.
        let mut surface_param_buffers: Vec<G::Buffer> = Vec::new();

        // Covers every per-frame `write_buffer` below, up to the point the
        // swapchain image is acquired — with slabs live this is the residual
        // cost for legacy (unspliced) content only; clean slabbed layers
        // contribute nothing.
        {
            profiling::scope!("wgpui: gpu upload");
            let gpu_upload_timer = crate::render_stats::scope("frame: gpu upload");

            let color_adjustments = ColorAdjustments {
                gamma_ratios: self.rendering_parameters.gamma_ratios,
                grayscale_enhanced_contrast: self.rendering_parameters.grayscale_enhanced_contrast,
                _padding: [0.0; 3],
            };
            if self.uploaded_color_adjustments != Some(color_adjustments) {
                self.context.write_buffer(
                    &self.color_adjustments_buffer,
                    0,
                    bytemuck::bytes_of(&color_adjustments),
                );
                self.uploaded_color_adjustments = Some(color_adjustments);
            }

            let globals = GlobalParams {
                viewport_size: [self.surface_size().0 as f32, self.surface_size().1 as f32],
                premultimated_alpha: G::swapchain_premultiplied(&self.swapchain) as u32,
                pad: 0,
            };

            if self.uploaded_globals != Some(globals) {
                self.context.write_buffer(
                    &self.globals_buffer,
                    0,
                    bytemuck::bytes_of(&globals),
                );
                self.uploaded_globals = Some(globals);
            }

            if !scene.quads.is_empty() {
                let data = bytemuck::cast_slice(&scene.quads);
                ensure_buffer_size(
                    self.context.gpu.as_ref(),
                    &self.context.quads_buffer,
                    data.len() as u64,
                    "Quads Buffer",
                    BufferUsage::VERTEX
                        | BufferUsage::COPY_DST
                        | BufferUsage::STORAGE,
                );
                self.context.write_buffer(&self.context.quads_buffer.lock(), 0, data);
            }
            if !scene.shadows.is_empty() {
                let data = bytemuck::cast_slice(&scene.shadows);
                ensure_buffer_size(
                    self.context.gpu.as_ref(),
                    &self.context.shadows_buffer,
                    data.len() as u64,
                    "Shadows Buffer",
                    BufferUsage::VERTEX
                        | BufferUsage::COPY_DST
                        | BufferUsage::STORAGE,
                );
                self.context.write_buffer(&self.context.shadows_buffer.lock(), 0, data);
            }
            if !scene.backdrop_filters.is_empty() {
                let data = bytemuck::cast_slice(&scene.backdrop_filters);
                ensure_buffer_size(
                    self.context.gpu.as_ref(),
                    &self.context.backdrop_filters_buffer,
                    data.len() as u64,
                    "Backdrop Filters Buffer",
                    BufferUsage::STORAGE | BufferUsage::COPY_DST,
                );
                self.context.write_buffer(
                    &self.context.backdrop_filters_buffer.lock(),
                    0,
                    data,
                );
            }
            if !scene.underlines.is_empty() {
                let data = bytemuck::cast_slice(&scene.underlines);
                ensure_buffer_size(
                    self.context.gpu.as_ref(),
                    &self.context.underlines_buffer,
                    data.len() as u64,
                    "Underlines Buffer",
                    BufferUsage::VERTEX
                        | BufferUsage::COPY_DST
                        | BufferUsage::STORAGE,
                );
                self.context.write_buffer(
                    &self.context.underlines_buffer.lock(),
                    0,
                    data,
                );
            }
            if !scene.monochrome_sprites.is_empty() {
                let data = bytemuck::cast_slice(&scene.monochrome_sprites);
                ensure_buffer_size(
                    self.context.gpu.as_ref(),
                    &self.context.mono_sprites_buffer,
                    data.len() as u64,
                    "Monosprites Buffer",
                    BufferUsage::VERTEX
                        | BufferUsage::COPY_DST
                        | BufferUsage::STORAGE,
                );
                self.context.write_buffer(
                    &self.context.mono_sprites_buffer.lock(),
                    0,
                    data,
                );
            }
            if !scene.polychrome_sprites.is_empty() {
                let data = bytemuck::cast_slice(&scene.polychrome_sprites);
                ensure_buffer_size(
                    self.context.gpu.as_ref(),
                    &self.context.poly_sprites_buffer,
                    data.len() as u64,
                    "Poly Sprites Buffer",
                    BufferUsage::VERTEX
                        | BufferUsage::COPY_DST
                        | BufferUsage::STORAGE,
                );
                self.context.write_buffer(
                    &self.context.poly_sprites_buffer.lock(),
                    0,
                    data,
                );
            }

            // Build flat vertex array for all paths (color + content mask baked per-vertex)
            let mut flat_path_vertices: Vec<GpuPathVertex> = Vec::new();
            for path in &scene.paths {
                let color = path.color.solid;
                let cm = &path.content_mask.bounds;
                let cm_origin = [cm.origin.x.0, cm.origin.y.0];
                let cm_size = [cm.size.width.0, cm.size.height.0];
                for vertex in &path.vertices {
                    flat_path_vertices.push(GpuPathVertex {
                        xy_position: [vertex.xy_position.x.0, vertex.xy_position.y.0],
                        st_position: [vertex.st_position.x, vertex.st_position.y],
                        hsla: [color.h, color.s, color.l, color.a],
                        content_mask_origin: cm_origin,
                        content_mask_size: cm_size,
                    });
                }
            }
            if !flat_path_vertices.is_empty() {
                let data = bytemuck::cast_slice(&flat_path_vertices);
                ensure_buffer_size(
                    self.context.gpu.as_ref(),
                    &self.context.paths_vertices_buffer,
                    data.len() as u64,
                    "Path Vertices Buffer",
                    BufferUsage::STORAGE | BufferUsage::COPY_DST,
                );
                self.context.write_buffer(
                    &self.context.paths_vertices_buffer.lock(),
                    0,
                    data,
                );
            }

            // Slab span resolution happens inside the upload timer's scope: it is
            // exactly the per-frame upload work, and clean layers resolve here
            // with zero queue writes.
            self.resolve_slab_spans(scene);

            drop(gpu_upload_timer);
        }

        // Acquire the next swapchain image.  On the first frame after window
        // creation (or after a resize races with the GPU) the surface can be
        // reported as `Outdated` or `Other`.  Rather than panicking we
        // reconfigure and retry once; if the second attempt also fails we
        // simply drop this frame.
        let surface_texture = match self.context.acquire(&mut self.swapchain) {
            Acquire::Frame(frame) => frame,
            Acquire::Outdated => {
                // Reconfigure with the current known size and retry.
                let (width, height) = self.surface_size();
                let present_mode = G::swapchain_present_mode(&self.swapchain);
                self.reconfigure_surface(width, height, present_mode);
                match self.context.acquire(&mut self.swapchain) {
                    Acquire::Frame(frame) => frame,
                    Acquire::Outdated => {
                        log::warn!("Skipping frame: swap chain still outdated after reconfigure");
                        return;
                    }
                    Acquire::Skip(reason) => {
                        log::warn!("Skipping frame after reconfigure: {reason}");
                        return;
                    }
                }
            }
            Acquire::Skip(reason) => {
                log::warn!("Skipping frame: {reason}");
                return;
            }
        };

        // Slab bind state comes from the frame-to-frame cache, which needs
        // `&mut` access to the cache field — built before the legacy buffer
        // locks below are held for the rest of the function.
        let layer_transform_bind_group = self
            .slab_group_cache
            .transforms_group(self.context.gpu.as_ref(), &self.pipelines, &self.slab_buffers);

        // Only frames that actually carry spans pay for slab bind state; the
        // cache makes even those frames cheap when nothing was recreated.
        let slab_groups =
            (scene.slab_span_count() > 0).then(|| self.slab_draw_groups_for_frame(scene));

        // Borrow buffers for bind group creation - these borrows must live until bind groups are done
        let quads_buffer_ref = self.context.quads_buffer.lock();
        let shadows_buffer_ref = self.context.shadows_buffer.lock();
        let backdrop_filters_buffer_ref = self.context.backdrop_filters_buffer.lock();
        let underlines_buffer_ref = self.context.underlines_buffer.lock();
        let mono_sprites_buffer_ref = self.context.mono_sprites_buffer.lock();
        let poly_sprites_buffer_ref = self.context.poly_sprites_buffer.lock();
        let paths_vertices_buffer_ref = self.context.paths_vertices_buffer.lock();

        let quads_bind_group = self.context.create_bind_group(
            "quads_bind_group",
            &self.pipelines.quads_bind_group_layout,
            &[
                BindEntry { binding: 0, resource: BindResource::Buffer { buffer: &quads_buffer_ref, offset: 0, size: None } },
            ],
        );

        let shadows_bind_group =
            self.context.create_bind_group(
                "shadows_bind_group",
                &self.pipelines.shadows_bind_group_layout,
                &[
                    BindEntry { binding: 0, resource: BindResource::Buffer { buffer: &shadows_buffer_ref, offset: 0, size: None } },
                ],
            );

        let backdrop_filters_bind_group =
            self.context.create_bind_group(
                "backdrop_filters_bind_group",
                &self.pipelines.backdrop_filters_bind_group_layout,
                &[
                    BindEntry { binding: 0, resource: BindResource::Buffer { buffer: &backdrop_filters_buffer_ref, offset: 0, size: None } },
                ],
            );

        let backdrop_texture_bind_group =
            self.context.create_bind_group(
                "backdrop_texture_bind_group",
                &self.pipelines.backdrop_texture_bind_group_layout,
                &[
                    BindEntry { binding: 0, resource: BindResource::Texture(self.backdrop_blur_texture_view.as_ref().unwrap()) },
                    BindEntry { binding: 1, resource: BindResource::Sampler(&self.backdrop_blur_sampler) },
                ],
            );

        let underlines_bind_group =
            self.context.create_bind_group(
                "underlines_bind_group",
                &self.pipelines.underlines_bind_group_layout,
                &[
                    BindEntry { binding: 0, resource: BindResource::Buffer { buffer: &underlines_buffer_ref, offset: 0, size: None } },
                ],
            );

        let mono_sprites_bind_group =
            self.context.create_bind_group(
                "mono_sprites_bind_group",
                &self.pipelines.mono_sprites_bind_group_layout,
                &[
                    BindEntry { binding: 0, resource: BindResource::Buffer { buffer: &mono_sprites_buffer_ref, offset: 0, size: None } },
                ],
            );

        let poly_sprites_bind_group =
            self.context.create_bind_group(
                "poly_sprites_bind_group",
                &self.pipelines.poly_sprites_bind_group_layout,
                &[
                    BindEntry { binding: 0, resource: BindResource::Buffer { buffer: &poly_sprites_buffer_ref, offset: 0, size: None } },
                ],
            );

        let paths_bind_group = self.context.create_bind_group(
            "paths_bind_group",
            &self.pipelines.paths_bind_group_layout,
            &[
                BindEntry { binding: 0, resource: BindResource::Buffer { buffer: &paths_vertices_buffer_ref, offset: 0, size: None } },
            ],
        );

        let mut glass_copied = false;
        {
            #[cfg(feature = "flamegraph")]
            let flamegraph_main_pass =
                self.reserve_gpu_timestamps("main", crate::GpuPassKind::Main);

            let mut pass = G::begin_pass(
                &mut command_encoder,
                &PassDesc {
                    label: "main",
                    target: self.persistent_framebuffer_view.as_ref().expect("persistent framebuffer view must exist"),
                    load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                    #[cfg(feature = "flamegraph")]
                    timestamps: flamegraph_main_pass.as_ref(),
                },
            );

            let mut quads_first_instance: u32 = 0;
            let mut shadows_first_instance: u32 = 0;
            let mut backdrop_filters_first_instance: u32 = 0;
            let mut underlines_first_instance: u32 = 0;
            let mut mono_sprites_first_instance: u32 = 0;
            let mut poly_sprites_first_instance: u32 = 0;
            let mut paths_vertex_offset: u32 = 0;

            // Stack of active content-filter groups. Each entry pairs the group's
            // `FilterBoundary` start marker with the `group_textures`/`group_views`
            // slot its content is being rendered into (`None` if the group exceeded
            // `MAX_FILTER_DEPTH` and is being painted inline, unisolated).
            let mut filter_stack: Vec<(FilterBoundary, Option<usize>)> = Vec::new();

            // Deep capture (issue #60): tracks which render pass `pass`
            // currently points at, so each recorded draw call's `pass_label`
            // reflects reality even though `pass` gets reassigned mid-loop
            // (backdrop filters/filter groups end and re-begin it). Kept
            // outside the `#[cfg(feature = "flamegraph")]` recorder itself
            // since it's just a `&'static str`, cheaper than gating every one
            // of its several assignment sites.
            #[cfg(feature = "flamegraph")]
            let mut current_pass_label: &'static str = "main";

            // Bind-state tracker and cross-span merge slot for the whole
            // main pass. Both must reset wherever the pass is dropped and
            // re-begun (backdrop filters, filter groups) or where draws use
            // bind groups this module does not model (surfaces, filter
            // composites).
            let transform_slot_stride = self.slab_buffers.transform_slot_stride;
            let mut pass_state = PassBindState::default();
            let mut open_slab_run: Option<OpenSlabRun> = None;

            // Layer textures (#96) already cleared this frame: the first span
            // of a layer clears its texture, later spans load and accumulate.
            let mut layer_textures_cleared: FxHashSet<crate::LayerId> = FxHashSet::default();

            for frame_batch in scene.frame_batches() {
                let batch = match frame_batch {
                    crate::scene::SceneBatch::Primitives(batch) => {
                        // A pending slab stretch precedes this draw in paint
                        // order; flush it into the same pass before anything
                        // else renders. Zero-instance batches (empty split
                        // halves) draw nothing and must not break adjacency.
                        if primitive_batch_instance_count(&batch) > 0
                            && open_slab_run.is_some()
                            && let Some(groups) = slab_groups.as_ref()
                        {
                            flush_open_slab_run(
                                &self.pipelines,
                                transform_slot_stride,
                                &mut pass,
                                groups,
                                &mut pass_state,
                                &mut open_slab_run,
                                &self.pipelines.globals_bind_group,
                            );
                        }
                        batch
                    }
                    crate::scene::SceneBatch::LayerSlab(span_index) => {
                        let texture_target = scene
                            .layer_slab_spans
                            .get(span_index)
                            .and_then(|span| span.texture.clone());
                        if let Some(target) = texture_target {
                            // #96: this span bakes a texture-retained layer's
                            // content into its persistent texture. Flush the
                            // main pass's open run, redirect into the layer
                            // texture, then resume the main pass untouched.
                            if let Some(groups) = slab_groups.as_ref() {
                                flush_open_slab_run(
                                    &self.pipelines,
                                    transform_slot_stride,
                                    &mut pass,
                                    groups,
                                    &mut pass_state,
                                    &mut open_slab_run,
                                    &self.pipelines.globals_bind_group,
                                );
                            }
                            drop(pass);

                            // `resolve_slab_spans` already created the texture
                            // (and posted a re-record if it could not); this
                            // loop holds immutable buffer locks, so it only
                            // reads the cache here.
                            let texture_ready = {
                                let width = target.texture_bounds.size.width.0.max(0.0).ceil() as u32;
                                let height =
                                    target.texture_bounds.size.height.0.max(0.0).ceil() as u32;
                                self.layer_textures.get(&target.layer_id).is_some_and(|entry| {
                                    entry.width == width && entry.height == height
                                })
                            };
                            if !texture_ready {
                                pass = G::begin_pass(
                                    &mut command_encoder,
                                    &PassDesc {
                                        label: "main",
                                        target: self.persistent_framebuffer_view.as_ref().expect("framebuffer exists during draw"),
                                        load: LoadOp::Load,
                                        #[cfg(feature = "flamegraph")]
                                        timestamps: None,
                                    },
                                );
                                #[cfg(feature = "flamegraph")]
                                {
                                    current_pass_label = "main";
                                }
                                pass_state.reset();
                                continue;
                            }
                            let texture_view = self.layer_textures[&target.layer_id].view.clone();
                            if let Some(entry) = self.layer_textures.get_mut(&target.layer_id) {
                                entry.last_used_frame = self.layer_texture_frame;
                                entry.content_token = target.content_token;
                            }
                            let clear = layer_textures_cleared.insert(target.layer_id);

                            #[cfg(feature = "flamegraph")]
                            let flamegraph_layer_texture_pass = self.reserve_gpu_timestamps(
                                "layer_texture",
                                crate::GpuPassKind::FilterGroup,
                            );
                            pass = G::begin_pass(
                                &mut command_encoder,
                                &PassDesc {
                                    label: "layer_texture",
                                    target: &texture_view,
                                    load: if clear {
                                        LoadOp::Clear([0.0, 0.0, 0.0, 0.0])
                                    } else {
                                        LoadOp::Load
                                    },
                                    #[cfg(feature = "flamegraph")]
                                    timestamps: flamegraph_layer_texture_pass.as_ref(),
                                },
                            );
                            #[cfg(feature = "flamegraph")]
                            {
                                current_pass_label = "layer_texture";
                            }
                            pass_state.reset();

                            // The packed coordinates are already texture-relative
                            // (`resolve_slab_spans` gives a texture-backed span an
                            // identity layer translate, on top of a pack built
                            // origin-relative to `texture_bounds.origin` in the
                            // first place — see that translate assignment's own
                            // comment), so the viewport is a plain identity affine
                            // at origin (0, 0) — no `-texture_origin` offset (that
                            // shifted already-relative geometry a second time; see
                            // this pass's other fix, just above in git blame).
                            //
                            // What it must NOT be is the window-sized globals
                            // every other pass shares: every slab vertex shader
                            // divides by `globals.viewport_size` to reach NDC, and
                            // a buffered layer's texture — `viewport + 2 × margin`
                            // — routinely stands taller (or wider) than the
                            // window itself. Content past the window's own extent
                            // would produce an NDC coordinate outside [-1, 1] and
                            // get clipped by the rasterizer before the scissor
                            // ever runs, silently dropping exactly the far margin
                            // band a large-enough buffer needs. So this pass gets
                            // its own globals, sized to the texture rather than
                            // the window, bound in place of `globals_bind_group`
                            // for these draws only — confirmed by
                            // `overscroll_buffer_bake_and_composite_match_a_direct_render_at_the_same_scroll_position`
                            // in renderer_slab_tests.rs, which fails at extreme
                            // margins without this and passes with it.
                            let bake_viewport_size = [
                                target.texture_bounds.size.width.0.max(1.0),
                                target.texture_bounds.size.height.0.max(1.0),
                            ];
                            let bake_globals = GlobalParams {
                                viewport_size: bake_viewport_size,
                                premultimated_alpha: G::swapchain_premultiplied(&self.swapchain) as u32,
                                pad: 0,
                            };
                            let bake_globals_buffer = self.context.create_buffer_init("layer_texture_bake_globals", bytemuck::bytes_of(&bake_globals), BufferUsage::UNIFORM);
                            let bake_globals_bind_group = self.context.create_bind_group(
                                "layer_texture_bake_globals_bind_group",
                                &self.pipelines.globals_bind_group_layout,
                                &[
                                    BindEntry { binding: 0, resource: BindResource::Buffer { buffer: &bake_globals_buffer, offset: 0, size: None } },
                                ],
                            );

                            G::set_viewport(&mut pass, 0.0, 0.0, bake_viewport_size[0], bake_viewport_size[1]);
                            G::set_scissor_rect(&mut pass, 0, 0, target.texture_bounds.size.width.0.ceil() as u32, target.texture_bounds.size.height.0.ceil() as u32);

                            if let Some(groups) = slab_groups.as_ref() {
                                if let Some(span) = scene.layer_slab_spans.get(span_index) {
                                    self.draw_texture_span_runs(
                                        &mut pass,
                                        span,
                                        groups,
                                        transform_slot_stride,
                                        &mut pass_state,
                                        &bake_globals_bind_group,
                                    );
                                }
                            }

                            // Resume the main pass where the redirect left it.
                            drop(pass);
                            pass = G::begin_pass(
                                &mut command_encoder,
                                &PassDesc {
                                    label: "main",
                                    target: self.persistent_framebuffer_view.as_ref().expect("framebuffer exists during draw"),
                                    load: LoadOp::Load,
                                    #[cfg(feature = "flamegraph")]
                                    timestamps: None,
                                },
                            );
                            #[cfg(feature = "flamegraph")]
                            {
                                current_pass_label = "main";
                            }
                            pass_state.reset();
                            continue;
                        }
                        if let Some(groups) = slab_groups.as_ref() {
                            self.draw_layer_slab_span(
                                &mut pass,
                                scene,
                                span_index,
                                groups,
                                transform_slot_stride,
                                &mut pass_state,
                                &mut open_slab_run,
                            );
                        }
                        continue;
                    }
                };
                match batch {
                    PrimitiveBatch::Quads(quads) => {
                        let count = quads.len() as u32;
                        pass_state.set_pipeline::<G>(&mut pass, DrawPipelineId::Quads, &self.pipelines.quads_pipeline);
                        pass_state.set_bind_group::<G>(&mut pass, 0, BoundGroupId::Globals, &self.pipelines.globals_bind_group, &[]);
                        pass_state.set_bind_group::<G>(&mut pass, 1, BoundGroupId::LegacyBuffer(LegacyBuffer::Quads), &quads_bind_group, &[]);
                        // Dynamic offset 0: the permanently-zero identity
                        // slot, so absolute legacy coordinates draw unshifted.
                        pass_state.set_bind_group::<G>(&mut pass, 2, BoundGroupId::LayerTransform(0), &layer_transform_bind_group, &[0]);
                        G::draw(&mut pass, 0..4, quads_first_instance..quads_first_instance + count);
                        quads_first_instance += count;
                        #[cfg(feature = "flamegraph")]
                        crate::record_draw_call(crate::DrawCallKind::Quads, count);
                        #[cfg(feature = "flamegraph")]
                        if let Some(recorder) = deep_capture_recorder.as_mut() {
                            <G::Profiler as GpuProfiler<G>>::record_draw_call(
                                recorder,
                                crate::DrawCallKind::Quads,
                                "quads",
                                current_pass_label,
                                0..4,
                                quads_first_instance - count..quads_first_instance,
                                2,
                                Some(crate::flamegraph::DeepCaptureBufferKind::Quads),
                                None,
                                None,
                            );
                        }
                    }

                    PrimitiveBatch::MonochromeSprites {
                        texture_id,
                        sprites,
                    } => {
                        let count = sprites.len() as u32;
                        let tex_info = self.atlas.get_texture_info(texture_id);

                        let sprites_texture_bind_group =
                            self.context.create_bind_group(
                                "sprites_bind_group",
                                &self.pipelines.sprites_bind_group_layout,
                                &[
                                    BindEntry { binding: 0, resource: BindResource::Texture(&tex_info.raw_view) },
                                    BindEntry { binding: 1, resource: BindResource::Sampler(&self.atlas_sampler) },
                                ],
                            );

                        pass_state.set_pipeline::<G>(&mut pass, DrawPipelineId::MonoSprites, &self.pipelines.mono_sprites_pipeline);
                        pass_state.set_bind_group::<G>(&mut pass, 0, BoundGroupId::Globals, &self.pipelines.globals_bind_group, &[]);
                        pass_state.set_bind_group::<G>(&mut pass, 1, BoundGroupId::ColorAdjustments, &self.pipelines.color_adjustments_bind_group, &[]);
                        pass_state.set_bind_group::<G>(
                            &mut pass,
                            2,
                            BoundGroupId::SpriteTexture(texture_id.index, texture_id.kind),
                            &sprites_texture_bind_group,
                            &[],
                        );
                        pass_state.set_bind_group::<G>(&mut pass, 3, BoundGroupId::LegacyBuffer(LegacyBuffer::MonoSprites), &mono_sprites_bind_group, &[]);
                        pass_state.set_bind_group::<G>(&mut pass, 4, BoundGroupId::LayerTransform(0), &layer_transform_bind_group, &[0]);
                        G::draw(&mut pass, 
                            0..4,
                            mono_sprites_first_instance..mono_sprites_first_instance + count,
                        );
                        mono_sprites_first_instance += count;
                        #[cfg(feature = "flamegraph")]
                        crate::record_draw_call(crate::DrawCallKind::MonoSprites, count);
                        #[cfg(feature = "flamegraph")]
                        if let Some(recorder) = deep_capture_recorder.as_mut() {
                            <G::Profiler as GpuProfiler<G>>::record_draw_call(
                                recorder,
                                crate::DrawCallKind::MonoSprites,
                                "mono_sprites",
                                current_pass_label,
                                0..4,
                                mono_sprites_first_instance - count..mono_sprites_first_instance,
                                4,
                                Some(crate::flamegraph::DeepCaptureBufferKind::MonoSprites),
                                Some(((texture_id.kind as u64) << 32) | texture_id.index as u64),
                                None,
                            );
                        }
                    }
                    PrimitiveBatch::PolychromeSprites {
                        texture_id,
                        sprites,
                    } => {
                        let count = sprites.len() as u32;
                        let tex_info = self.atlas.get_texture_info(texture_id);

                        let sprites_texture_bind_group =
                            self.context.create_bind_group(
                                "poly_sprites_texture_bind_group",
                                &self.pipelines.sprites_bind_group_layout,
                                &[
                                    BindEntry { binding: 0, resource: BindResource::Texture(&tex_info.raw_view) },
                                    BindEntry { binding: 1, resource: BindResource::Sampler(&self.atlas_sampler) },
                                ],
                            );

                        pass_state.set_pipeline::<G>(&mut pass, DrawPipelineId::PolySprites, &self.pipelines.poly_sprites_pipeline);
                        pass_state.set_bind_group::<G>(&mut pass, 0, BoundGroupId::Globals, &self.pipelines.globals_bind_group, &[]);
                        pass_state.set_bind_group::<G>(
                            &mut pass,
                            1,
                            BoundGroupId::SpriteTexture(texture_id.index, texture_id.kind),
                            &sprites_texture_bind_group,
                            &[],
                        );
                        pass_state.set_bind_group::<G>(&mut pass, 2, BoundGroupId::LegacyBuffer(LegacyBuffer::PolySprites), &poly_sprites_bind_group, &[]);
                        pass_state.set_bind_group::<G>(&mut pass, 3, BoundGroupId::LayerTransform(0), &layer_transform_bind_group, &[0]);
                        G::draw(&mut pass, 
                            0..4,
                            poly_sprites_first_instance..poly_sprites_first_instance + count,
                        );
                        poly_sprites_first_instance += count;
                        #[cfg(feature = "flamegraph")]
                        crate::record_draw_call(crate::DrawCallKind::PolySprites, count);
                        #[cfg(feature = "flamegraph")]
                        if let Some(recorder) = deep_capture_recorder.as_mut() {
                            <G::Profiler as GpuProfiler<G>>::record_draw_call(
                                recorder,
                                crate::DrawCallKind::PolySprites,
                                "poly_sprites",
                                current_pass_label,
                                0..4,
                                poly_sprites_first_instance - count..poly_sprites_first_instance,
                                3,
                                Some(crate::flamegraph::DeepCaptureBufferKind::PolySprites),
                                Some(((texture_id.kind as u64) << 32) | texture_id.index as u64),
                                None,
                            );
                        }
                    }
                    PrimitiveBatch::Shadows(shadows) => {
                        let count = shadows.len() as u32;
                        pass_state.set_pipeline::<G>(&mut pass, DrawPipelineId::Shadows, &self.pipelines.shadows_pipeline);
                        pass_state.set_bind_group::<G>(&mut pass, 0, BoundGroupId::Globals, &self.pipelines.globals_bind_group, &[]);
                        pass_state.set_bind_group::<G>(&mut pass, 1, BoundGroupId::LegacyBuffer(LegacyBuffer::Shadows), &shadows_bind_group, &[]);
                        pass_state.set_bind_group::<G>(&mut pass, 2, BoundGroupId::LayerTransform(0), &layer_transform_bind_group, &[0]);
                        G::draw(&mut pass, 0..4, shadows_first_instance..shadows_first_instance + count);
                        shadows_first_instance += count;
                        #[cfg(feature = "flamegraph")]
                        crate::record_draw_call(crate::DrawCallKind::Shadows, count);
                        #[cfg(feature = "flamegraph")]
                        if let Some(recorder) = deep_capture_recorder.as_mut() {
                            <G::Profiler as GpuProfiler<G>>::record_draw_call(
                                recorder,
                                crate::DrawCallKind::Shadows,
                                "shadows",
                                current_pass_label,
                                0..4,
                                shadows_first_instance - count..shadows_first_instance,
                                2,
                                Some(crate::flamegraph::DeepCaptureBufferKind::Shadows),
                                None,
                                None,
                            );
                        }
                    }
                    PrimitiveBatch::BackdropFilters(backdrop_filters) => {
                        let count = backdrop_filters.len() as u32;

                        // End the current render pass to copy texture
                        drop(pass);

                        // Copy the persistent framebuffer (this frame's content so
                        // far) to backdrop_blur_texture for sampling. Never the
                        // swapchain: it only holds LAST frame's blit, and resuming
                        // on it gets clobbered by the final framebuffer blit
                        // (chrome invisible behind a full-window wgpu surface).
                        if let (Some(blur_texture), Some(framebuffer)) =
                            (&self.backdrop_blur_texture, &self.persistent_framebuffer)
                        {
                            let fb_size = G::texture_size(framebuffer);
                            let key = self.glass_backdrop_gen.load(Ordering::Relaxed);
                            let reuse = key != 0 && self.glass_backdrop_copied_gen == key;

                            // First frame of a lock key copies every batch so a
                            // nested sheet sees the sheet under it. Later frames
                            // reuse that last copy (covered UI does not move).
                            if !reuse && fb_size == G::texture_size(blur_texture) {
                                G::copy_texture_to_texture(
                                    &mut command_encoder,
                                    framebuffer,
                                    blur_texture,
                                    fb_size.0,
                                    fb_size.1,
                                );
                                glass_copied = true;
                            }
                        }

                        // Begin new render pass with Load to preserve existing content
                        #[cfg(feature = "flamegraph")]
                        let flamegraph_main_resumed_pass = self.reserve_gpu_timestamps(
                            "main_resumed",
                            crate::GpuPassKind::MainResumed,
                        );
                        pass = G::begin_pass(
                            &mut command_encoder,
                            &PassDesc {
                                label: "main_resumed",
                                target: self.persistent_framebuffer_view.as_ref().expect("persistent framebuffer view must exist"),
                                load: LoadOp::Load,
                                #[cfg(feature = "flamegraph")]
                                timestamps: flamegraph_main_resumed_pass.as_ref(),
                            },
                        );
                        #[cfg(feature = "flamegraph")]
                        {
                            current_pass_label = "main_resumed";
                        }
                        // Fresh pass: nothing is bound anymore.
                        pass_state.reset();

                        // Now render the backdrop blur quads
                        G::set_pipeline(&mut pass, &self.pipelines.backdrop_filters_pipeline);
                        G::set_bind_group(&mut pass, 0, &self.pipelines.globals_bind_group, &[]);
                        G::set_bind_group(&mut pass, 1, &backdrop_filters_bind_group, &[]);
                        G::set_bind_group(&mut pass, 2, &backdrop_texture_bind_group, &[]);
                        G::draw(&mut pass, 
                            0..4,
                            backdrop_filters_first_instance..backdrop_filters_first_instance + count,
                        );
                        backdrop_filters_first_instance += count;
                        #[cfg(feature = "flamegraph")]
                        crate::record_draw_call(crate::DrawCallKind::BackdropFilters, count);
                        #[cfg(feature = "flamegraph")]
                        if let Some(recorder) = deep_capture_recorder.as_mut() {
                            <G::Profiler as GpuProfiler<G>>::record_draw_call(
                                recorder,
                                crate::DrawCallKind::BackdropFilters,
                                "backdrop_filters",
                                current_pass_label,
                                0..4,
                                backdrop_filters_first_instance - count..backdrop_filters_first_instance,
                                3,
                                Some(crate::flamegraph::DeepCaptureBufferKind::BackdropFilters),
                                None,
                                None,
                            );
                        }
                    }
                    PrimitiveBatch::FilterBoundary(index) => {
                        let boundary = scene.filter_boundaries[index];

                        if boundary.is_start {
                            let depth = filter_stack.len();
                            if depth >= self.group_textures.len() {
                                // Exceeded the supported nesting depth: paint the group's
                                // content inline (unisolated/unblurred) rather than dropping it.
                                filter_stack.push((boundary, None));
                            } else {
                                drop(pass);

                                #[cfg(feature = "flamegraph")]
                                let flamegraph_filter_group_pass = self.reserve_gpu_timestamps(
                                    "filter_group",
                                    crate::GpuPassKind::FilterGroup,
                                );
                                pass = G::begin_pass(
                                    &mut command_encoder,
                                    &PassDesc {
                                        label: "filter_group",
                                        target: &self.group_views[depth],
                                        load: LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
                                        #[cfg(feature = "flamegraph")]
                                        timestamps: flamegraph_filter_group_pass.as_ref(),
                                    },
                                );
                                #[cfg(feature = "flamegraph")]
                                {
                                    current_pass_label = "filter_group";
                                }
                                // Fresh pass: nothing is bound anymore.
                                pass_state.reset();

                                filter_stack.push((boundary, Some(depth)));
                            }
                        } else {
                            let Some((start_boundary, depth)) = filter_stack.pop() else {
                                continue;
                            };

                            let Some(depth) = depth else {
                                // The group was painted inline; nothing to composite.
                                continue;
                            };

                            // End the group's pass: its content is now baked into
                            // `group_textures[depth]`.
                            drop(pass);

                            // Root parent = the persistent framebuffer, never the
                            // swapchain (the final blit would overwrite it).
                            let parent_view: &G::TextureView = match filter_stack.last() {
                                Some((_, Some(parent_depth))) => &self.group_views[*parent_depth],
                                _ => self
                                    .persistent_framebuffer_view
                                    .as_ref()
                                    .expect("persistent framebuffer view must exist"),
                            };

                            #[cfg(feature = "flamegraph")]
                            let flamegraph_filter_group_resumed_pass = self.reserve_gpu_timestamps(
                                "filter_group_resumed",
                                crate::GpuPassKind::FilterGroupResumed,
                            );
                            pass = G::begin_pass(
                                &mut command_encoder,
                                &PassDesc {
                                    label: "filter_group_resumed",
                                    target: parent_view,
                                    load: LoadOp::Load,
                                    #[cfg(feature = "flamegraph")]
                                    timestamps: flamegraph_filter_group_resumed_pass.as_ref(),
                                },
                            );
                            #[cfg(feature = "flamegraph")]
                            {
                                current_pass_label = "filter_group_resumed";
                            }
                            // Fresh pass: nothing is bound anymore.
                            pass_state.reset();

                            // Composite the blurred group content back over the parent using
                            // the same backdrop-filter pipeline, sampling from the group's
                            // offscreen texture instead of a surface snapshot.
                            let composite = BackdropFilter {
                                order: 0,
                                bounds: start_boundary.bounds,
                                content_mask: start_boundary.content_mask,
                                corner_radii: start_boundary.corner_radii,
                                blur_radius: start_boundary.blur_radius,
                                opacity: start_boundary.opacity,
                                _pad: 0,
                            };
                            let composite_buffer = self.context.create_buffer_init("filter_group_composite_buffer", bytemuck::cast_slice(std::slice::from_ref(
                                        &composite,
                                    )), BufferUsage::STORAGE);
                            let composite_bind_group = self.context.create_bind_group(
                                "filter_group_composite_bind_group",
                                &self.pipelines.backdrop_filters_bind_group_layout,
                                &[
                                    BindEntry { binding: 0, resource: BindResource::Buffer { buffer: &composite_buffer, offset: 0, size: None } },
                                ],
                            );
                            let composite_texture_bind_group = self.context.create_bind_group(
                                "filter_group_texture_bind_group",
                                &self.pipelines.backdrop_texture_bind_group_layout,
                                &[
                                    BindEntry { binding: 0, resource: BindResource::Texture(&self.group_views[depth]) },
                                    BindEntry { binding: 1, resource: BindResource::Sampler(&self.backdrop_blur_sampler) },
                                ],
                            );

                            G::set_pipeline(&mut pass, &self.pipelines.backdrop_filters_pipeline);
                            G::set_bind_group(&mut pass, 0, &self.pipelines.globals_bind_group, &[]);
                            G::set_bind_group(&mut pass, 1, &composite_bind_group, &[]);
                            G::set_bind_group(&mut pass, 2, &composite_texture_bind_group, &[]);
                            G::draw(&mut pass, 0..4, 0..1);
                        }
                    }
                    PrimitiveBatch::Underlines(underlines) => {
                        let count = underlines.len() as u32;
                        pass_state.set_pipeline::<G>(&mut pass, DrawPipelineId::Underlines, &self.pipelines.underlines_pipeline);
                        pass_state.set_bind_group::<G>(&mut pass, 0, BoundGroupId::Globals, &self.pipelines.globals_bind_group, &[]);
                        pass_state.set_bind_group::<G>(&mut pass, 1, BoundGroupId::LegacyBuffer(LegacyBuffer::Underlines), &underlines_bind_group, &[]);
                        pass_state.set_bind_group::<G>(&mut pass, 2, BoundGroupId::LayerTransform(0), &layer_transform_bind_group, &[0]);
                        G::draw(&mut pass, 
                            0..4,
                            underlines_first_instance..underlines_first_instance + count,
                        );
                        underlines_first_instance += count;
                        #[cfg(feature = "flamegraph")]
                        crate::record_draw_call(crate::DrawCallKind::Underlines, count);
                        #[cfg(feature = "flamegraph")]
                        if let Some(recorder) = deep_capture_recorder.as_mut() {
                            <G::Profiler as GpuProfiler<G>>::record_draw_call(
                                recorder,
                                crate::DrawCallKind::Underlines,
                                "underlines",
                                current_pass_label,
                                0..4,
                                underlines_first_instance - count..underlines_first_instance,
                                2,
                                Some(crate::flamegraph::DeepCaptureBufferKind::Underlines),
                                None,
                                None,
                            );
                        }
                    }
                    PrimitiveBatch::Surfaces(surfaces) => {
                        log::trace!("Renderer: processing {} surface(s)", surfaces.len());
                        for surface in surfaces {
                            if let crate::SurfaceContent::Wgpu(surface_id) = &surface.content {
                                // Swap ready → display ONLY if the external renderer produced
                                // a new frame since we last composited this surface. This paint
                                // path runs every GPUI frame whether or not the producer rendered
                                // anything (the viewport re-arms request_animation_frame each
                                // frame), so an unconditional swap here would rotate `display` to
                                // a stale buffer whenever the producer skipped a frame — engine
                                // lock contention or a pending resize — and the canvas strobes.
                                // The gate holds the current display buffer until a real frame is
                                // ready. The fast-blit path (Path B) is already gated via
                                // redraw_pending, so it keeps using the unconditional swap.
                                let _swapped = self
                                    .context
                                    .surface_registry
                                    .swap_ready_display_if_new(*surface_id);

                                if let Some(view) =
                                    self.context.surface_registry.front_view(*surface_id)
                                {
                                    let params = SurfaceParams {
                                        bounds: Bounds {
                                            origin: [
                                                surface.bounds.origin.x.0,
                                                surface.bounds.origin.y.0,
                                            ],
                                            size: [
                                                surface.bounds.size.width.0,
                                                surface.bounds.size.height.0,
                                            ],
                                        },
                                        content_mask: Bounds {
                                            origin: [
                                                surface.content_mask.bounds.origin.x.0,
                                                surface.content_mask.bounds.origin.y.0,
                                            ],
                                            size: [
                                                surface.content_mask.bounds.size.width.0,
                                                surface.content_mask.bounds.size.height.0,
                                            ],
                                        },
                                        corner_radii: [
                                            surface.corner_radii.top_left.0,
                                            surface.corner_radii.top_right.0,
                                            surface.corner_radii.bottom_right.0,
                                            surface.corner_radii.bottom_left.0,
                                        ],
                                    };

                                    let params_buffer = self.context.create_buffer_init("surface_params_buffer", bytemuck::bytes_of(&params), BufferUsage::UNIFORM);

                                    let surface_bind_group = self.context.create_bind_group(
                                        "surface_bind_group",
                                        &self.pipelines.surfaces_bind_group_layout,
                                        &[
                                            BindEntry { binding: 0, resource: BindResource::Buffer { buffer: &params_buffer, offset: 0, size: None } },
                                            BindEntry { binding: 1, resource: BindResource::Texture(&view) },
                                            BindEntry { binding: 2, resource: BindResource::Sampler(&self.surface_sampler) },
                                        ],
                                    );

                                    G::set_pipeline(&mut pass, &self.pipelines.surfaces_pipeline);
                                    G::set_bind_group(&mut pass, 0, &self.pipelines.globals_bind_group, &[]);
                                    G::set_bind_group(&mut pass, 1, &surface_bind_group, &[]);
                                    G::draw(&mut pass, 0..4, 0..1);
                                    // Per-surface groups are unique objects this
                                    // tracker does not model; drop all tracked
                                    // state so later draws rebind conservatively.
                                    pass_state.reset();
                                    #[cfg(feature = "flamegraph")]
                                    crate::record_draw_call(crate::DrawCallKind::Surfaces, 1);
                                    #[cfg(feature = "flamegraph")]
                                    if let Some(recorder) = deep_capture_recorder.as_mut() {
                                        <G::Profiler as GpuProfiler<G>>::record_draw_call(
                                recorder,
                                            crate::DrawCallKind::Surfaces,
                                            "surfaces",
                                            current_pass_label,
                                            0..4,
                                            0..1,
                                            2,
                                            None,
                                            None,
                                            Some(surface_id.0),
                                        );
                                    }

                                    // CRITICAL: Keep view alive until after render pass ends
                                    // The bind_group holds a reference to it
                                    surface_views.push(view);
                                    surface_param_buffers.push(params_buffer);

                                    // Clear redraw pending AFTER we're done with the view
                                    // This prevents the external thread from triggering another compositor
                                    // pass while we're still using this view
                                    self.context
                                        .surface_registry
                                        .clear_redraw_pending(*surface_id);

                                    seen_surfaces.push(*surface_id);
                                }
                            } else if let crate::SurfaceContent::Layer(layer_id) = &surface.content
                            {
                                // #96: composite a texture-retained layer's
                                // persistent texture. The surface's bounds are
                                // the buffer extent (shifted by the buffered
                                // element's scroll); the content mask clips to
                                // the layer's visible rect, so margin content
                                // never paints outside the layer.
                                let Some(entry) = self.layer_textures.get_mut(layer_id) else {
                                    log::trace!(
                                        "layer texture for {layer_id:?} missing at composite; \
                                         waiting for the posted re-record"
                                    );
                                    continue;
                                };
                                entry.last_used_frame = self.layer_texture_frame;
                                let view = entry.view.clone();

                                let params = SurfaceParams {
                                    bounds: Bounds {
                                        origin: [
                                            surface.bounds.origin.x.0,
                                            surface.bounds.origin.y.0,
                                        ],
                                        size: [
                                            surface.bounds.size.width.0,
                                            surface.bounds.size.height.0,
                                        ],
                                    },
                                    content_mask: Bounds {
                                        origin: [
                                            surface.content_mask.bounds.origin.x.0,
                                            surface.content_mask.bounds.origin.y.0,
                                        ],
                                        size: [
                                            surface.content_mask.bounds.size.width.0,
                                            surface.content_mask.bounds.size.height.0,
                                        ],
                                    },
                                    corner_radii: [
                                        surface.corner_radii.top_left.0,
                                        surface.corner_radii.top_right.0,
                                        surface.corner_radii.bottom_right.0,
                                        surface.corner_radii.bottom_left.0,
                                    ],
                                };

                                let params_buffer = self.context.create_buffer_init("layer_surface_params_buffer", bytemuck::bytes_of(&params), BufferUsage::UNIFORM);

                                let surface_bind_group = self.context.create_bind_group(
                                    "layer_surface_bind_group",
                                    &self.pipelines.surfaces_bind_group_layout,
                                    &[
                                        BindEntry { binding: 0, resource: BindResource::Buffer { buffer: &params_buffer, offset: 0, size: None } },
                                        BindEntry { binding: 1, resource: BindResource::Texture(&view) },
                                        BindEntry { binding: 2, resource: BindResource::Sampler(&self.surface_sampler) },
                                    ],
                                );

                                G::set_pipeline(&mut pass, &self.pipelines.surfaces_pipeline);
                                G::set_bind_group(&mut pass, 0, &self.pipelines.globals_bind_group, &[]);
                                G::set_bind_group(&mut pass, 1, &surface_bind_group, &[]);
                                G::draw(&mut pass, 0..4, 0..1);
                                pass_state.reset();
                                #[cfg(feature = "flamegraph")]
                                crate::record_draw_call(crate::DrawCallKind::Surfaces, 1);

                                // Keep the view alive until after the render pass ends.
                                surface_views.push(view);
                                surface_param_buffers.push(params_buffer);
                            }
                        }
                    }
                    PrimitiveBatch::Paths(paths) => {
                        let vertex_count: u32 = paths.iter().map(|p| p.vertices.len() as u32).sum();
                        if vertex_count > 0 {
                            pass_state.set_pipeline::<G>(&mut pass, DrawPipelineId::Paths, &self.pipelines.paths_pipeline);
                            pass_state.set_bind_group::<G>(&mut pass, 0, BoundGroupId::Globals, &self.pipelines.globals_bind_group, &[]);
                            pass_state.set_bind_group::<G>(&mut pass, 1, BoundGroupId::LegacyBuffer(LegacyBuffer::PathVertices), &paths_bind_group, &[]);
                            pass_state.set_bind_group::<G>(&mut pass, 2, BoundGroupId::LayerTransform(0), &layer_transform_bind_group, &[0]);
                            G::draw(&mut pass, 
                                paths_vertex_offset..paths_vertex_offset + vertex_count,
                                0..1,
                            );
                            paths_vertex_offset += vertex_count;
                            #[cfg(feature = "flamegraph")]
                            crate::record_draw_call(crate::DrawCallKind::Paths, paths.len() as u32);
                            #[cfg(feature = "flamegraph")]
                            if let Some(recorder) = deep_capture_recorder.as_mut() {
                                <G::Profiler as GpuProfiler<G>>::record_draw_call(
                                recorder,
                                    crate::DrawCallKind::Paths,
                                    "paths",
                                    current_pass_label,
                                    paths_vertex_offset - vertex_count..paths_vertex_offset,
                                    0..1,
                                    2,
                                    Some(crate::flamegraph::DeepCaptureBufferKind::Paths),
                                    None,
                                    None,
                                );
                            }
                        }
                    }
                }
            }

            // The final span's merged stretch flushes here: nothing may keep
            // an open run across the pass end.
            if let Some(groups) = slab_groups.as_ref() {
                flush_open_slab_run(
                    &self.pipelines,
                    transform_slot_stride,
                    &mut pass,
                    groups,
                    &mut pass_state,
                    &mut open_slab_run,
                    &self.pipelines.globals_bind_group,
                );
            }
        }

        // Blit persistent framebuffer to swapchain
        if let Some(ref persistent_framebuffer) = self.persistent_framebuffer {
            G::copy_texture_to_frame(&mut command_encoder, persistent_framebuffer, &surface_texture);
        }

        // Close out the GpuSubmitPresent bracket, resolve this frame's queries
        // and, if this frame was armed for a deep capture, record its readback
        // copies — all before the encoder is finished. `quads_buffer_ref`/etc.
        // are the guards already held for the whole duration of `draw`.
        #[cfg(feature = "flamegraph")]
        {
            let buffers: [(crate::DeepCaptureBufferKind, &G::Buffer); 7] = [
                (crate::DeepCaptureBufferKind::Quads, &quads_buffer_ref),
                (crate::DeepCaptureBufferKind::Shadows, &shadows_buffer_ref),
                (crate::DeepCaptureBufferKind::Underlines, &underlines_buffer_ref),
                (crate::DeepCaptureBufferKind::MonoSprites, &mono_sprites_buffer_ref),
                (crate::DeepCaptureBufferKind::PolySprites, &poly_sprites_buffer_ref),
                (crate::DeepCaptureBufferKind::BackdropFilters, &backdrop_filters_buffer_ref),
                (crate::DeepCaptureBufferKind::Paths, &paths_vertices_buffer_ref),
            ];
            self.profiler.lock().end_frame(
                self.context.gpu.as_ref(),
                &mut command_encoder,
                deep_capture_recorder.take(),
                &buffers,
                &self.atlas,
                &self.context.surface_registry,
            );
        }

        let glass_key = self.glass_backdrop_gen.load(Ordering::Relaxed);
        if glass_key != 0 && glass_copied {
            self.glass_backdrop_copied_gen = glass_key;
        } else if glass_key == 0 {
            self.glass_backdrop_copied_gen = 0;
        }

        log::trace!("Renderer::draw: submitting command buffer");
        let queue_guard = self.context.surface_registry.queue_lock();
        self.context.submit(command_encoder);
        log::trace!("Renderer::draw: presenting surface");
        self.context.present(surface_texture);
        drop(queue_guard);

        // Start the async readbacks now that their commands were submitted.
        #[cfg(feature = "flamegraph")]
        self.profiler.get_mut().after_submit();

        log::trace!("Renderer::draw: frame complete");
    }

    /// Get list of surfaces that have pending redraws
    /// Une surface a publié par `present_synced` une trame pas encore composée.
    pub fn has_pending_surfaces(&self) -> bool {
        !self.context.surface_registry.get_pending_surfaces().is_empty()
    }

    pub fn any_unconsumed_surface_frame(&self) -> bool {
        self.context.surface_registry.any_unconsumed_frame()
    }

    pub fn take_new_surface_frame(&self) -> bool {
        self.context.surface_registry.take_new_frame()
    }

    pub fn update_drawable_size(&mut self, size: geometry::Size<DevicePixels>) {
        // Windows répète des `WM_SIZE` de même taille pendant un glissement de
        // bordure. Reconfigurer prend le verrou exclusif de soumission (les
        // fils de rendu externes s'arrêtent) et recrée trois textures plein
        // écran : rien de tout ça n'est dû quand la taille n'a pas bougé.
        let (width, height) = (size.width.0 as u32, size.height.0 as u32);
        if self.surface_size() == (width, height) {
            crate::render_stats::count("resize: same size skipped");
            return;
        }
        crate::render_stats::count("resize: reconfigure");
        let present_mode = G::swapchain_present_mode(&self.swapchain);
        self.reconfigure_surface(width, height, present_mode);
        let format = G::swapchain_format(&self.swapchain);

        // Recreate persistent framebuffer at new size
        let persistent_framebuffer = self.context.create_texture(
            "persistent_framebuffer",
            width,
            height,
            format,
            TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED | TextureUsage::COPY_SRC,
        );
        self.persistent_framebuffer_view = Some(G::create_view(&persistent_framebuffer));
        self.persistent_framebuffer = Some(persistent_framebuffer);

        // Recreate backdrop blur capture texture at the new size so that
        // copy_texture_to_texture doesn't silently skip due to a size mismatch.
        let backdrop_blur_texture = self.context.create_texture(
            "backdrop_blur_texture",
            width,
            height,
            format,
            TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED | TextureUsage::COPY_DST,
        );
        self.backdrop_blur_texture_view = Some(G::create_view(&backdrop_blur_texture));
        self.backdrop_blur_texture = Some(backdrop_blur_texture);
        self.glass_backdrop_copied_gen = 0;

        // Recreate the content-filter group textures at the new size so they stay
        // pixel-aligned with the surface (group composites sample using
        // `pixel_position / globals.viewport_size` UVs, just like backdrop blur).
        let (group_textures, group_views) =
            create_filter_group_textures(self.context.gpu.as_ref(), width, height, format);
        self.group_textures = group_textures;
        self.group_views = group_views;

        // Layer textures (#96) are sized to their layer's buffer extent, not
        // the surface, so they survive a resize — but their content was
        // rasterized against the old scale/viewport, and the composite's NDC
        // mapping changed. Drop them all; the re-record requests make each
        // texture-retained layer re-bake on its next composite.
        let dropped_keys: Vec<crate::LayerKey> = self
            .layer_textures
            .drain()
            .map(|(_, entry)| entry.key)
            .collect();
        if !dropped_keys.is_empty() {
            log::trace!(
                "dropped {} layer textures for resize; requesting re-records",
                dropped_keys.len()
            );
            self.slab_registry.request_rerecord(dropped_keys);
        }

    }

    pub fn gpu_specs(&self) -> GpuSpecs {
        self.context.gpu_specs()
    }

    /// Rebascule la swapchain sur un autre mode de presentation, a chaud.
    ///
    /// Rend `false` si la surface ne l'accepte pas : `Mailbox` manque sur
    /// beaucoup de pilotes et `Surface::configure` avorte le process plutot
    /// que d'echouer proprement sur un mode non supporte.
    pub fn lock_glass_backdrop(&mut self, key: u32) {
        self.glass_backdrop_gen.store(key, Ordering::Relaxed);
    }

    pub fn set_present_mode(&mut self, mode: crate::WindowPresentMode) -> bool {
        if !self.context.supported_present_modes(&self.swapchain).contains(&mode) {
            return false;
        }
        if G::swapchain_present_mode(&self.swapchain) == mode {
            crate::present_mode::set_window_present_mode(mode);
            return true;
        }
        let (width, height) = self.surface_size();
        self.reconfigure_surface(width, height, mode);
        crate::present_mode::set_window_present_mode(mode);
        true
    }
}

#[cfg(all(feature = "wgpu", feature = "flamegraph"))]
impl WgpuRenderer {
    /// On-demand GPU memory snapshot for this renderer (Phase 3 of the
    /// profiling epic, issue #59): mostly summing sizes that already exist
    /// on already-owned wgpu resources, no new tracking required.
    #[cfg(feature = "flamegraph")]
    pub(crate) fn gpu_memory_snapshot(&self) -> crate::GpuMemorySnapshot {
        crate::GpuMemorySnapshot {
            fixed_buffer_bytes: self.context.fixed_buffer_memory_usage(),
            atlas_bytes: self.atlas.memory_usage(),
            surface_registry_bytes: self.context.surface_registry.memory_usage(),
            swapchain_bytes: self.swapchain_memory_usage(),
        }
    }

    /// The live `wgpu::Device`/`Queue` backing this renderer, for callers
    /// that want to drive GPU work outside the normal frame path -- e.g.
    /// `flamegraph_replay::render_deep_capture_step` (Phase 6 of the
    /// profiling epic, issue #62), so a deep-capture replay preview runs
    /// against the app's real device instead of spinning up a second,
    /// separate headless one on every call. `Device`/`Queue` are cheap
    /// `Clone` handles (wgpu itself reference-counts the underlying
    /// resources), so returning owned clones here is the idiomatic wgpu
    /// pattern, not a meaningful cost.
    #[cfg(feature = "flamegraph")]
    pub(crate) fn gpu_device_and_queue(&self) -> (wgpu::Device, wgpu::Queue) {
        (self.context.device.clone(), self.context.queue.clone())
    }

    /// Best-effort swapchain memory estimate from `surface_configuration`'s
    /// dimensions/format. wgpu doesn't expose the presentation engine's
    /// actual backing image count, so `desired_maximum_frame_latency` (the
    /// one buffering-depth signal WGPUI itself configures) stands in for it.
    #[cfg(feature = "flamegraph")]
    fn swapchain_memory_usage(&self) -> u64 {
        use super::hal::wgpu::WgpuGpu;
        let bytes_per_texel = WgpuGpu::bytes_per_pixel(WgpuGpu::swapchain_format(&self.swapchain)) as u64;
        let image_count = WgpuGpu::swapchain_frame_latency(&self.swapchain).max(1) as u64;
        let (width, height) = self.surface_size();
        (width as u64) * (height as u64) * bytes_per_texel * image_count
    }
}

/// Issue one merged slab run's instanced draw with full bind state.
///
/// Free-standing (rather than a `WgpuRenderer` method) so the GPU-tier tests
/// can drive the exact production draw function against a headless device.
/// Every call rebinds everything, which is what makes it the naive baseline;
/// production goes through [`flush_slab_run_with_state`] so redundant binds
/// are skipped instead.
#[cfg(test)]
fn flush_slab_run<G: Gpu>(
    pipelines: &Pipelines<G>,
    transform_slot_stride: u64,
    pass: &mut G::Pass<'_>,
    slabs: &crate::platform::cross::slab::LayerSlabs,
    groups: &SlabDrawGroups<G>,
    transform_slot: u32,
    run: &SlabPendingRun,
) {
    let mut untracked = PassBindState::default();
    flush_slab_run_with_state(
        pipelines,
        transform_slot_stride,
        pass,
        slabs,
        groups,
        transform_slot,
        run,
        &mut untracked,
        &pipelines.globals_bind_group,
    );
}

/// [`flush_slab_run`] with bind-state tracking: pipeline and bind-group sets
/// that would repeat what the pass already holds are skipped. Skipping is
/// pixel-neutral because tracked ids match only when the bound resource,
/// layout slot, and dynamic offsets are all identical.
///
/// `globals_bind_group` is a parameter rather than always
/// `&pipelines.globals_bind_group` so a texture-retained layer's bake pass
/// can bind its own, texture-sized globals instead of the shared,
/// window-sized one (#96) — see that pass's own call site for why.
fn flush_slab_run_with_state<G: Gpu>(
    pipelines: &Pipelines<G>,
    transform_slot_stride: u64,
    pass: &mut G::Pass<'_>,
    slabs: &crate::platform::cross::slab::LayerSlabs,
    groups: &SlabDrawGroups<G>,
    transform_slot: u32,
    run: &SlabPendingRun,
    state: &mut PassBindState,
    globals_bind_group: &G::BindGroup,
) {
    profiling::scope!("wgpui: flush slab runs");
    let dynamic_offsets = [(transform_slot as u64 * transform_slot_stride) as u32];
    let transform_id = BoundGroupId::LayerTransform(dynamic_offsets[0]);
    let range_base = slabs.slab(run.kind).base + run.start;
    match run.kind {
        SlabKind::Quads => {
            state.set_pipeline::<G>(pass, DrawPipelineId::Quads, &pipelines.quads_pipeline);
            state.set_bind_group::<G>(pass, 0, BoundGroupId::Globals, globals_bind_group, &[]);
            state.set_bind_group::<G>(pass, 1, BoundGroupId::SlabStorage(SlabKind::Quads), &groups.quads, &[]);
            state.set_bind_group::<G>(pass, 2, transform_id, &groups.layer_transform, &dynamic_offsets);
            G::draw(pass, 0..4, range_base..range_base + run.count);
        }
        SlabKind::Shadows => {
            state.set_pipeline::<G>(pass, DrawPipelineId::Shadows, &pipelines.shadows_pipeline);
            state.set_bind_group::<G>(pass, 0, BoundGroupId::Globals, globals_bind_group, &[]);
            state.set_bind_group::<G>(pass, 1, BoundGroupId::SlabStorage(SlabKind::Shadows), &groups.shadows, &[]);
            state.set_bind_group::<G>(pass, 2, transform_id, &groups.layer_transform, &dynamic_offsets);
            G::draw(pass, 0..4, range_base..range_base + run.count);
        }
            SlabKind::Underlines => {
                state.set_pipeline::<G>(pass, DrawPipelineId::Underlines, &pipelines.underlines_pipeline);
                state.set_bind_group::<G>(pass, 0, BoundGroupId::Globals, globals_bind_group, &[]);
                state.set_bind_group::<G>(pass, 1, BoundGroupId::SlabStorage(SlabKind::Underlines), &groups.underlines, &[]);
                state.set_bind_group::<G>(pass, 2, transform_id, &groups.layer_transform, &dynamic_offsets);
                G::draw(pass, 0..4, range_base..range_base + run.count);
            }
            SlabKind::MonoSprites => {
                let Some(texture_id) = run.texture_id else {
                    debug_assert!(false, "sprite runs carry a texture id");
                    return;
                };
                let Some(texture_group) = groups.sprite_textures.get(&(texture_id.index, texture_id.kind))
                else {
                    debug_assert!(false, "sprite textures validated before drawing");
                    return;
                };
                state.set_pipeline::<G>(pass, DrawPipelineId::MonoSprites, &pipelines.mono_sprites_pipeline);
                state.set_bind_group::<G>(pass, 0, BoundGroupId::Globals, globals_bind_group, &[]);
                state.set_bind_group::<G>(pass, 1, BoundGroupId::ColorAdjustments, &pipelines.color_adjustments_bind_group, &[]);
                state.set_bind_group::<G>(
                    pass,
                    2,
                    BoundGroupId::SpriteTexture(texture_id.index, texture_id.kind),
                    texture_group,
                    &[],
                );
                state.set_bind_group::<G>(pass, 3, BoundGroupId::SlabStorage(SlabKind::MonoSprites), &groups.mono_sprites, &[]);
                state.set_bind_group::<G>(pass, 4, transform_id, &groups.layer_transform, &dynamic_offsets);
                G::draw(pass, 0..4, range_base..range_base + run.count);
            }
            SlabKind::PolySprites => {
                let Some(texture_id) = run.texture_id else {
                    debug_assert!(false, "sprite runs carry a texture id");
                    return;
                };
                let Some(texture_group) = groups.sprite_textures.get(&(texture_id.index, texture_id.kind))
                else {
                    debug_assert!(false, "sprite textures validated before drawing");
                    return;
                };
                state.set_pipeline::<G>(pass, DrawPipelineId::PolySprites, &pipelines.poly_sprites_pipeline);
                state.set_bind_group::<G>(pass, 0, BoundGroupId::Globals, globals_bind_group, &[]);
                state.set_bind_group::<G>(
                    pass,
                    1,
                    BoundGroupId::SpriteTexture(texture_id.index, texture_id.kind),
                    texture_group,
                    &[],
                );
                state.set_bind_group::<G>(pass, 2, BoundGroupId::SlabStorage(SlabKind::PolySprites), &groups.poly_sprites, &[]);
                state.set_bind_group::<G>(pass, 3, transform_id, &groups.layer_transform, &dynamic_offsets);
                G::draw(pass, 0..4, range_base..range_base + run.count);
            }
            // Path runs address vertices, not instances: the layer's vertex
            // block sits at its Paths range inside the shared stream.
            SlabKind::Paths => {
                let base = slabs.slab(SlabKind::Paths).base;
                state.set_pipeline::<G>(pass, DrawPipelineId::Paths, &pipelines.paths_pipeline);
                state.set_bind_group::<G>(pass, 0, BoundGroupId::Globals, globals_bind_group, &[]);
                state.set_bind_group::<G>(pass, 1, BoundGroupId::SlabStorage(SlabKind::Paths), &groups.paths_vertices, &[]);
                state.set_bind_group::<G>(pass, 2, transform_id, &groups.layer_transform, &dynamic_offsets);
                G::draw(pass, base + run.start..base + run.start + run.count, 0..1);
            }
        }
        crate::render_stats::add(slab_gpu::COUNTER_DRAW_CALLS, 1);
        #[cfg(feature = "flamegraph")]
        crate::record_draw_call(flamegraph_kind(run.kind), run.count);
    }

/// Flush an open cross-span slab stretch, if any, with the shared bind-state
/// tracker. Called before any non-continuing draw and at end of pass.
fn flush_open_slab_run<G: Gpu>(
    pipelines: &Pipelines<G>,
    transform_slot_stride: u64,
    pass: &mut G::Pass<'_>,
    groups: &SlabDrawGroups<G>,
    state: &mut PassBindState,
    open: &mut Option<OpenSlabRun>,
    globals_bind_group: &G::BindGroup,
) {
    if let Some(open) = open.take() {
        let pending = open.as_pending();
        flush_slab_run_with_state(
            pipelines,
            transform_slot_stride,
            pass,
            &open.slabs,
            groups,
            open.transform_slot,
            &pending,
            state,
            globals_bind_group,
        );
    }
}

#[cfg(feature = "flamegraph")]
fn flamegraph_kind(kind: SlabKind) -> crate::DrawCallKind {
    match kind {
        SlabKind::Quads => crate::DrawCallKind::Quads,
        SlabKind::Shadows => crate::DrawCallKind::Shadows,
        SlabKind::Paths => crate::DrawCallKind::Paths,
        SlabKind::Underlines => crate::DrawCallKind::Underlines,
        SlabKind::MonoSprites => crate::DrawCallKind::MonoSprites,
        SlabKind::PolySprites => crate::DrawCallKind::PolySprites,
    }
}

#[cfg(test)]
#[path = "renderer_slab_tests.rs"]
mod slab_tests;
