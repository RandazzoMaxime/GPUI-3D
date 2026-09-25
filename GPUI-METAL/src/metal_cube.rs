//! Full natif : l'UI de GPUI et ce moteur 3D rendent tous deux en Metal, sans wgpu dans
//! le binaire. GPUI fournit son `MTLDevice`, sa
//! `MTLCommandQueue` et la `MTLTexture` du tampon arrière ; tout le reste est l'API
//! Metal. Même queue que le compositeur ⇒ ordre garanti, textures « tracked » ⇒ Metal
//! gère le hazard rendu → échantillonnage, sans fence.

use std::ffi::c_void;
use std::ptr::NonNull;

use gpui3d_shell::{CLEAR_COLOR, CUBE_INDICES, CUBE_VERTICES, NativeDevice, NativeTexture, Renderer, Scene, Surface};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;

type Obj<T> = Retained<ProtocolObject<T>>;

pub struct MetalCube {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    pipeline: Obj<dyn MTLRenderPipelineState>,
    depth_state: Obj<dyn MTLDepthStencilState>,
    vertices: Obj<dyn MTLBuffer>,
    indices: Obj<dyn MTLBuffer>,
    depth: Option<(Obj<dyn MTLTexture>, (u32, u32))>,
}

impl Renderer for MetalCube {
    fn new(surface: &Surface) -> Self {
        let Some(NativeDevice::Metal { device, queue }) = surface.native_device() else {
            panic!("GPUI ne tourne pas sur Metal");
        };
        let device: Obj<dyn MTLDevice> = unsafe { Retained::retain(device.cast()) }.expect("MTLDevice");
        let queue: Obj<dyn MTLCommandQueue> = unsafe { Retained::retain(queue.cast()) }.expect("MTLCommandQueue");

        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(include_str!("cube.metal")), None)
            .expect("compilation MSL");
        let function = |name: &str| library.newFunctionWithName(&NSString::from_str(name)).expect(name);
        let desc = MTLRenderPipelineDescriptor::new();
        desc.setVertexFunction(Some(&function("vs_main")));
        desc.setFragmentFunction(Some(&function("fs_main")));
        unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) }.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
        desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
        let pipeline = device.newRenderPipelineStateWithDescriptor_error(&desc).expect("pipeline Metal");

        let ds = MTLDepthStencilDescriptor::new();
        ds.setDepthCompareFunction(MTLCompareFunction::Less);
        ds.setDepthWriteEnabled(true);
        let depth_state = device.newDepthStencilStateWithDescriptor(&ds).expect("depth state");

        let vertices = new_buffer(&device, &CUBE_VERTICES);
        let indices = new_buffer(&device, &CUBE_INDICES);
        Self { device, queue, pipeline, depth_state, vertices, indices, depth: None }
    }

    fn render(&mut self, surface: &Surface, scene: &Scene) -> bool {
        let Some(back) = surface.native_back_buffer() else { return false };
        let NativeTexture::Metal(ptr) = back.texture else { return false };
        let target: Obj<dyn MTLTexture> = unsafe { Retained::retain(ptr.cast()) }.expect("MTLTexture");
        let (w, h) = back.size;
        if self.depth.as_ref().map(|d| d.1) != Some((w, h)) {
            self.depth = Some((new_depth(&self.device, w, h), (w, h)));
        }
        let depth = self.depth.as_ref().expect("profondeur").0.clone();

        let pass = MTLRenderPassDescriptor::new();
        let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        let [red, green, blue, alpha] = CLEAR_COLOR;
        color.setTexture(Some(&target));
        color.setLoadAction(MTLLoadAction::Clear);
        color.setClearColor(MTLClearColor { red, green, blue, alpha });
        color.setStoreAction(MTLStoreAction::Store);
        let da = pass.depthAttachment();
        da.setTexture(Some(&depth));
        da.setLoadAction(MTLLoadAction::Clear);
        da.setClearDepth(1.0);
        da.setStoreAction(MTLStoreAction::DontCare);

        let Some(cb) = self.queue.commandBuffer() else { return false };
        let Some(enc) = cb.renderCommandEncoderWithDescriptor(&pass) else { return false };
        enc.setRenderPipelineState(&self.pipeline);
        enc.setDepthStencilState(Some(&self.depth_state));
        enc.setCullMode(MTLCullMode::Back);
        enc.setFrontFacingWinding(MTLWinding::CounterClockwise);
        let mvp = scene.mvp(w, h);
        unsafe {
            enc.setVertexBuffer_offset_atIndex(Some(&self.vertices), 0, 0);
            enc.setVertexBytes_length_atIndex(NonNull::from(&mvp).cast(), size_of_val(&mvp), 1);
            enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                MTLPrimitiveType::Triangle,
                CUBE_INDICES.len(),
                MTLIndexType::UInt16,
                &self.indices,
                0,
            );
        }
        enc.endEncoding();
        cb.commit();
        // Commit avant la publication : le compositeur soumettra après nous sur la même queue.
        surface.swap_buffers();
        true
    }
}

fn new_buffer<T: Copy>(device: &ProtocolObject<dyn MTLDevice>, data: &[T]) -> Obj<dyn MTLBuffer> {
    let ptr = NonNull::new(data.as_ptr() as *mut c_void).expect("données");
    unsafe { device.newBufferWithBytes_length_options(ptr, size_of_val(data), MTLResourceOptions::StorageModeShared) }
        .expect("MTLBuffer")
}

fn new_depth(device: &ProtocolObject<dyn MTLDevice>, w: u32, h: u32) -> Obj<dyn MTLTexture> {
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::Depth32Float,
            w as usize,
            h as usize,
            false,
        )
    };
    desc.setUsage(MTLTextureUsage::RenderTarget);
    // Memoryless : la profondeur vit en mémoire tuile, jamais en VRAM (GPU Apple uniquement).
    desc.setStorageMode(if device.supportsFamily(MTLGPUFamily::Apple1) {
        MTLStorageMode::Memoryless
    } else {
        MTLStorageMode::Private
    });
    device.newTextureWithDescriptor(&desc).expect("texture profondeur")
}
