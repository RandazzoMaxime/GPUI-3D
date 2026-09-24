use std::sync::Arc;

use collections::FxHashMap;
use etagere::BucketedAtlasAllocator;
use parking_lot::Mutex;

use crate::{
    AtlasKey, AtlasTextureId, AtlasTextureKind, AtlasTile, Bounds, DevicePixels, PlatformAtlas,
    Point, Size,
    platform::{AtlasTextureList, cross::hal::{Gpu, TextureUsage}},
};

pub(crate) struct Atlas<G: Gpu>(Mutex<AtlasState<G>>);

#[cfg(feature = "wgpu")]
pub(crate) type WgpuAtlas = Atlas<super::hal::wgpu::WgpuGpu>;

impl<G: Gpu> Atlas<G> {
    pub(crate) fn new(gpu: Arc<G>) -> Self {
        Atlas(Mutex::new(AtlasState {
            gpu,
            storage: AtlasStorage::default(),
            tiles_by_key: FxHashMap::default(),
            destroyed_pages: Vec::new(),
        }))
    }

    /// Drain every page whose contents changed under existing tile handles:
    /// fully-destroyed textures and tiles freed out of still-live pages alike.
    ///
    /// Layer-slab residency keys off these: any sprite run referencing an
    /// evicted page must not reach the GPU with stale texel ids, so the
    /// renderer poisons those layers and requests a re-record (see
    /// `slab_gpu`). Draining is destructive — each event is reported once.
    pub(crate) fn drain_destroyed_pages(&self) -> Vec<crate::AtlasTextureId> {
        std::mem::take(&mut self.0.lock().destroyed_pages)
    }

    pub(crate) fn get_texture_info(&self, texture_id: AtlasTextureId) -> TextureInfo<G> {
        let state = self.0.lock();
        let texture = &state.storage[texture_id];

        TextureInfo {
            raw_view: texture.raw_view.clone(),
        }
    }
}

#[cfg(feature = "flamegraph")]
impl WgpuAtlas {
    /// Sum of every live atlas texture's backing memory, monochrome and
    /// polychrome combined (Phase 3 of the profiling epic, issue #59).
    #[cfg(feature = "flamegraph")]
    pub(crate) fn memory_usage(&self) -> u64 {
        let state = self.0.lock();
        atlas_texture_list_memory_usage(&state.storage.monochrome_textures)
            + atlas_texture_list_memory_usage(&state.storage.polychrome_textures)
    }

    /// One atlas texture page's identity/metadata needed to read its pixel
    /// contents back for a triggered GPU deep capture (Phase 4b of the
    /// profiling epic, issue #72). Distinct from `get_texture_info`, which
    /// only exposes a `TextureView` -- enough to bind a sprite pipeline, but
    /// `copy_texture_to_buffer` needs the underlying `wgpu::Texture`
    /// directly, plus the pixel dimensions and texel size a caller needs to
    /// compute `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` row padding. Returns
    /// `None` if `texture_id` no longer refers to a live page (e.g. it was
    /// evicted between the draw call that touched it and this lookup --
    /// shouldn't happen within one frame's own command stream, but reported
    /// rather than panicking since there's no way to distinguish that from
    /// a genuinely stale id).
    #[cfg(feature = "flamegraph")]
    pub(crate) fn texture_snapshot(&self, texture_id: AtlasTextureId) -> Option<WgpuAtlasTextureSnapshot> {
        let state = self.0.lock();
        let texture = state.storage[texture_id.kind]
            .textures
            .get(texture_id.index as usize)?
            .as_ref()?;
        Some(WgpuAtlasTextureSnapshot {
            texture: texture.raw.clone(),
            width: texture.raw.width(),
            height: texture.raw.height(),
            // `texel_size` (not `WgpuAtlasTexture::bytes_per_pixel`, which
            // panics on any format outside the two this atlas currently
            // creates) so this stays correct rather than crashing if the
            // atlas ever grows a third texture kind.
            bytes_per_pixel: super::render_context::texel_size(texture.format) as u32,
        })
    }
}

/// See [`WgpuAtlas::texture_snapshot`].
#[cfg(feature = "flamegraph")]
pub(crate) struct WgpuAtlasTextureSnapshot {
    pub(crate) texture: wgpu::Texture,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bytes_per_pixel: u32,
}

#[cfg(feature = "flamegraph")]
fn atlas_texture_list_memory_usage(textures: &AtlasTextureList<AtlasTexture<super::hal::wgpu::WgpuGpu>>) -> u64 {
    textures
        .textures
        .iter()
        .flatten()
        .map(|texture| super::render_context::texture_memory_bytes(&texture.raw))
        .sum()
}

impl<G: Gpu> PlatformAtlas for Atlas<G> {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> anyhow::Result<
            Option<(Size<DevicePixels>, std::borrow::Cow<'a, [u8]>)>,
        >,
    ) -> anyhow::Result<Option<AtlasTile>> {
        let mut atlas = self.0.lock();

        match atlas.tiles_by_key.get(key) {
            Some(tile) => {
                #[cfg(feature = "flamegraph")]
                crate::record_atlas_cache_hit();
                Ok(Some(tile.clone()))
            }
            None => Ok({
                profiling::scope!("new tile");
                #[cfg(feature = "flamegraph")]
                crate::record_atlas_cache_miss();

                match build()? {
                    Some((size, bytes)) => {
                        let tile = atlas.allocate(size, key.texture_kind())?;
                        #[cfg(feature = "flamegraph")]
                        crate::record_atlas_tile_allocated();

                        atlas.upload_texture(tile.texture_id, tile.bounds, &bytes);
                        atlas.tiles_by_key.insert(key.clone(), tile.clone());

                        Some(tile)
                    }
                    None => None,
                }
            }),
        }
    }

    fn remove(&self, key: &AtlasKey) {
        let mut atlas = self.0.lock();

        let Some(id) = atlas.tiles_by_key.remove(key).map(|x| x.texture_id) else {
            return;
        };
        #[cfg(feature = "flamegraph")]
        crate::record_atlas_tile_evicted();

        // Both eviction shapes invalidate slab residency: a destroyed page
        // takes every tile with it, and a tile freed from a live page leaves
        // its region reusable by the next allocation.
        atlas.destroyed_pages.push(id);

        let Some(texture_slot) = atlas.storage[id.kind].textures.get_mut(id.index as usize) else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.decrement_ref_count();

            if texture.is_unreferenced() {
                atlas.storage[id.kind]
                    .free_list
                    .push(texture.id.index as usize);

                // Eagerly destroy to free GPU memory immediately.
            } else {
                *texture_slot = Some(texture);
            }
        }
    }
}

struct AtlasState<G: Gpu> {
    gpu: Arc<G>,
    storage: AtlasStorage<G>,
    tiles_by_key: FxHashMap<AtlasKey, AtlasTile>,
    /// Evictions not yet reported to the slab registry: destroyed pages and
    /// tiles freed from live pages, deduplicated at drain time.
    destroyed_pages: Vec<crate::AtlasTextureId>,
}

impl<G: Gpu> AtlasState<G> {
    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> anyhow::Result<AtlasTile> {
        {
            let textures = &mut self.storage[texture_kind];

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Ok(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);

        texture.allocate(size).ok_or_else(|| {
            anyhow::anyhow!(
                "newly created atlas texture of size {}x{} could not satisfy allocation of size {}x{}",
                G::texture_size(&texture.raw).0,
                G::texture_size(&texture.raw).1,
                size.width.0,
                size.height.0,
            )
        })
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> &mut AtlasTexture<G> {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };

        let size = min_size.max(&DEFAULT_ATLAS_SIZE);

        let format = match texture_kind {
            AtlasTextureKind::Monochrome => G::ATLAS_MONOCHROME,
            AtlasTextureKind::Polychrome => G::ATLAS_POLYCHROME,
        };
        let texture_raw = self.gpu.create_texture(
            "Atlas Texture",
            size.width.0 as u32,
            size.height.0 as u32,
            format,
            TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::SAMPLED,
        );
        let texture_raw_view = G::create_view(&texture_raw);

        let texture_list = &mut self.storage[texture_kind];

        let index = texture_list.free_list.pop();

        let atlas_texture = AtlasTexture {
            id: AtlasTextureId {
                kind: texture_kind,
                index: index.unwrap_or(texture_list.textures.len()) as u32,
            },
            allocator: BucketedAtlasAllocator::new(size.into()),
            raw: texture_raw,
            raw_view: texture_raw_view,
            format,
            live_atlas_keys: 0,
        };

        // If we popped a free slot from the free list, place the texture there;
        // otherwise append to the end.
        match index {
            Some(index) => {
                texture_list.textures[index] = Some(atlas_texture);
                texture_list
                    .textures
                    .get_mut(index)
                    .unwrap()
                    .as_mut()
                    .unwrap()
            }
            None => {
                texture_list.textures.push(Some(atlas_texture));
                texture_list.textures.last_mut().unwrap().as_mut().unwrap()
            }
        }
    }

    fn upload_texture(
        &mut self,
        texture_id: AtlasTextureId,
        bounds: Bounds<DevicePixels>,
        bytes: &[u8],
    ) {
        let texture = &self.storage[texture_id];
        self.gpu.write_texture(
            &texture.raw,
            (bounds.origin.x.into(), bounds.origin.y.into()),
            (bounds.size.width.into(), bounds.size.height.into()),
            G::bytes_per_pixel(texture.format),
            bytes,
        );
    }
}

pub(crate) struct AtlasTexture<G: Gpu> {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    raw: G::Texture,
    raw_view: G::TextureView,
    format: G::Format,
    live_atlas_keys: u32,
}

impl<G: Gpu> AtlasTexture<G> {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(size.into())?;

        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            padding: 0,
            bounds: Bounds {
                origin: allocation.rectangle.min.into(),
                size,
            },
        };

        self.live_atlas_keys += 1;

        Some(tile)
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys = self.live_atlas_keys.saturating_sub(1);
    }

    fn is_unreferenced(&self) -> bool {
        self.live_atlas_keys == 0
    }
}

impl<G: Gpu> std::ops::Index<AtlasTextureKind> for AtlasStorage<G> {
    type Output = AtlasTextureList<AtlasTexture<G>>;
    fn index(&self, kind: AtlasTextureKind) -> &Self::Output {
        match kind {
            crate::AtlasTextureKind::Monochrome => &self.monochrome_textures,
            crate::AtlasTextureKind::Polychrome => &self.polychrome_textures,
        }
    }
}

impl<G: Gpu> std::ops::IndexMut<AtlasTextureKind> for AtlasStorage<G> {
    fn index_mut(&mut self, kind: AtlasTextureKind) -> &mut Self::Output {
        match kind {
            crate::AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            crate::AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
        }
    }
}

impl<G: Gpu> std::ops::Index<AtlasTextureId> for AtlasStorage<G> {
    type Output = AtlasTexture<G>;
    fn index(&self, id: AtlasTextureId) -> &Self::Output {
        let textures = match id.kind {
            crate::AtlasTextureKind::Monochrome => &self.monochrome_textures,
            crate::AtlasTextureKind::Polychrome => &self.polychrome_textures,
        };

        textures[id.index as usize].as_ref().unwrap()
    }
}

struct AtlasStorage<G: Gpu> {
    monochrome_textures: AtlasTextureList<AtlasTexture<G>>,
    polychrome_textures: AtlasTextureList<AtlasTexture<G>>,
}

impl<G: Gpu> Default for AtlasStorage<G> {
    fn default() -> Self {
        Self {
            monochrome_textures: AtlasTextureList::default(),
            polychrome_textures: AtlasTextureList::default(),
        }
    }
}

pub(crate) struct TextureInfo<G: Gpu> {
    pub raw_view: G::TextureView,
}

impl From<Size<DevicePixels>> for etagere::Size {
    fn from(size: Size<DevicePixels>) -> Self {
        etagere::Size::new(size.width.into(), size.height.into())
    }
}

impl From<etagere::Point> for Point<DevicePixels> {
    fn from(value: etagere::Point) -> Self {
        Point {
            x: DevicePixels::from(value.x),
            y: DevicePixels::from(value.y),
        }
    }
}

impl From<etagere::Size> for Size<DevicePixels> {
    fn from(size: etagere::Size) -> Self {
        Size {
            width: DevicePixels::from(size.width),
            height: DevicePixels::from(size.height),
        }
    }
}

impl From<etagere::Rectangle> for Bounds<DevicePixels> {
    fn from(rectangle: etagere::Rectangle) -> Self {
        Bounds {
            origin: rectangle.min.into(),
            size: rectangle.size().into(),
        }
    }
}
