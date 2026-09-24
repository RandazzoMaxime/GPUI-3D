//! Moteur 3D en D3D12 natif, sans wgpu : GPUI fournit son `ID3D12Device`, son
//! `ID3D12CommandQueue` et la `ID3D12Resource` du tampon arrière ; allocateurs, listes,
//! fence, tas de descripteurs, root signature et PSO sont à nous. Même queue que le
//! compositeur ⇒ l'ordre de soumission porte la synchronisation ; une queue D3D12 est
//! thread-safe, donc pas de `native_queue_lock`.

use std::mem::ManuallyDrop;

use gpui3d_shell::{CLEAR_COLOR, CUBE_INDICES, CUBE_VERTICES, NativeBackBuffer, NativeDevice, NativeTexture, Renderer, Scene, Surface};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Interface, PCSTR, s};

/// = `gpui3d_shell::SURFACE_FORMAT`.
const COLOR_FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
const DEPTH_FORMAT: DXGI_FORMAT = DXGI_FORMAT_D32_FLOAT;
const FRAMES_IN_FLIGHT: usize = 2;
/// Contrat du fork : le tampon arrive et repart dans l'état `RESOURCE` de wgpu.
const SAMPLED: D3D12_RESOURCE_STATES =
    D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0);

pub struct Dx12Cube {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    fence_value: u64,
    frames: Vec<Frame>,
    frame: usize,
    list: ID3D12GraphicsCommandList,
    root: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
    rtv_heap: ID3D12DescriptorHeap,
    dsv_heap: ID3D12DescriptorHeap,
    vertices: (ID3D12Resource, D3D12_VERTEX_BUFFER_VIEW),
    indices: (ID3D12Resource, D3D12_INDEX_BUFFER_VIEW),
    depth: Option<(ID3D12Resource, (u32, u32))>,
}

struct Frame {
    allocator: ID3D12CommandAllocator,
    /// Valeur de fence signalée à la fin GPU de la trame.
    done: u64,
    /// Retient le tampon arrière tant que le GPU l'écrit.
    back: Option<NativeBackBuffer>,
}

impl Renderer for Dx12Cube {
    fn new(surface: &Surface) -> Self {
        let Some(NativeDevice::Dx12 { device, queue }) = surface.native_device() else {
            panic!("GPUI ne tourne pas sur D3D12");
        };
        // Pointeurs empruntés à wgpu : `clone` = AddRef.
        let device = unsafe { ID3D12Device::from_raw_borrowed(&device) }.expect("ID3D12Device").clone();
        let queue = unsafe { ID3D12CommandQueue::from_raw_borrowed(&queue) }.expect("ID3D12CommandQueue").clone();
        unsafe {
            let fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let frames = (0..FRAMES_IN_FLIGHT)
                .map(|_| Frame {
                    allocator: device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("allocateur"),
                    done: 0,
                    back: None,
                })
                .collect::<Vec<_>>();
            let root = create_root_signature(&device);
            let pipeline = create_pipeline(&device, &root);
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &frames[0].allocator, &pipeline)
                .expect("command list");
            list.Close().expect("close");
            let heap = |kind| -> ID3D12DescriptorHeap {
                // Un seul descripteur : OMSetRenderTargets le lit à l'enregistrement.
                let desc = D3D12_DESCRIPTOR_HEAP_DESC { Type: kind, NumDescriptors: 1, ..Default::default() };
                device.CreateDescriptorHeap(&desc).expect("tas de descripteurs")
            };
            let (rtv_heap, dsv_heap) = (heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV), heap(D3D12_DESCRIPTOR_HEAP_TYPE_DSV));
            let vb = upload_buffer(&device, bytemuck::cast_slice(&CUBE_VERTICES));
            let vb_view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: vb.GetGPUVirtualAddress(),
                SizeInBytes: size_of_val(&CUBE_VERTICES) as u32,
                StrideInBytes: 24,
            };
            let ib = upload_buffer(&device, bytemuck::cast_slice(&CUBE_INDICES));
            let ib_view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: ib.GetGPUVirtualAddress(),
                SizeInBytes: size_of_val(&CUBE_INDICES) as u32,
                Format: DXGI_FORMAT_R16_UINT,
            };
            Self {
                device,
                queue,
                fence,
                fence_value: 0,
                frames,
                frame: 0,
                list,
                root,
                pipeline,
                rtv_heap,
                dsv_heap,
                vertices: (vb, vb_view),
                indices: (ib, ib_view),
                depth: None,
            }
        }
    }

    fn render(&mut self, surface: &Surface, scene: &Scene) -> bool {
        self.wait(self.frames[self.frame].done);
        self.frames[self.frame].back = None;
        let Some(back) = surface.native_back_buffer() else { return false };
        let NativeTexture::Dx12(ptr) = back.texture else { return false };
        let (w, h) = back.size;
        if self.depth.as_ref().map(|d| d.1) != Some((w, h)) {
            self.wait(self.fence_value);
            self.depth = Some((create_depth(&self.device, &self.dsv_heap, w, h), (w, h)));
        }
        let target = unsafe { ID3D12Resource::from_raw_borrowed(&ptr) }.expect("ID3D12Resource");
        let frame = &mut self.frames[self.frame];
        let list = &self.list;
        unsafe {
            let rtv = self.rtv_heap.GetCPUDescriptorHandleForHeapStart();
            let dsv = self.dsv_heap.GetCPUDescriptorHandleForHeapStart();
            let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
                Format: COLOR_FORMAT,
                ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_RTV::default() },
            };
            self.device.CreateRenderTargetView(target, Some(&rtv_desc), rtv);

            frame.allocator.Reset().expect("reset allocateur");
            list.Reset(&frame.allocator, &self.pipeline).expect("reset liste");
            list.ResourceBarrier(&[transition(target, SAMPLED, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
            let [r, g, b, a] = CLEAR_COLOR.map(|c| c as f32);
            list.ClearRenderTargetView(rtv, &[r, g, b, a], None);
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
            list.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            list.RSSetViewports(&[D3D12_VIEWPORT { Width: w as f32, Height: h as f32, MaxDepth: 1.0, ..Default::default() }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { right: w as i32, bottom: h as i32, ..Default::default() }]);
            list.SetGraphicsRootSignature(&self.root);
            let mvp = scene.mvp(w, h);
            list.SetGraphicsRoot32BitConstants(0, 16, mvp.as_ptr().cast(), 0);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.IASetVertexBuffers(0, Some(&[self.vertices.1]));
            list.IASetIndexBuffer(Some(&self.indices.1));
            list.DrawIndexedInstanced(CUBE_INDICES.len() as u32, 1, 0, 0, 0);
            list.ResourceBarrier(&[transition(target, D3D12_RESOURCE_STATE_RENDER_TARGET, SAMPLED)]);
            list.Close().expect("close");

            self.queue.ExecuteCommandLists(&[Some(list.cast().expect("ID3D12CommandList"))]);
            self.fence_value += 1;
            self.queue.Signal(&self.fence, self.fence_value).expect("Signal");
        }
        frame.done = self.fence_value;
        frame.back = Some(back);
        self.frame = (self.frame + 1) % FRAMES_IN_FLIGHT;
        // Soumis avant la publication : le compositeur échantillonnera après nous.
        surface.swap_buffers();
        true
    }
}

impl Dx12Cube {
    /// Bloque jusqu'à ce que la fence atteigne `value` (événement nul = attente synchrone).
    fn wait(&self, value: u64) {
        if unsafe { self.fence.GetCompletedValue() } < value {
            unsafe { self.fence.SetEventOnCompletion(value, HANDLE::default()) }.expect("attente fence");
        }
    }
}

impl Drop for Dx12Cube {
    fn drop(&mut self) {
        // Le reste se libère par comptage de références, une fois le GPU au repos.
        self.wait(self.fence_value);
    }
}

/// Barrière de transition sur une ressource empruntée (pas d'AddRef, pas de Release).
fn transition(resource: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn upload_buffer(device: &ID3D12Device, bytes: &[u8]) -> ID3D12Resource {
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes.len() as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    unsafe {
        device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut buffer)
            .expect("tampon upload");
        let buffer = buffer.expect("tampon upload");
        let mut ptr = std::ptr::null_mut();
        buffer.Map(0, None, Some(&mut ptr)).expect("map");
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len());
        buffer.Unmap(0, None);
        buffer
    }
}

fn create_depth(device: &ID3D12Device, heap: &ID3D12DescriptorHeap, w: u32, h: u32) -> ID3D12Resource {
    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: w as u64,
        Height: h,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DEPTH_FORMAT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE,
        ..Default::default()
    };
    let clear = D3D12_CLEAR_VALUE {
        Format: DEPTH_FORMAT,
        Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 } },
    };
    let mut depth: Option<ID3D12Resource> = None;
    unsafe {
        device
            .CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, Some(&clear), &mut depth)
            .expect("profondeur");
        let depth = depth.expect("profondeur");
        device.CreateDepthStencilView(&depth, None, heap.GetCPUDescriptorHandleForHeapStart());
        depth
    }
}

fn create_root_signature(device: &ID3D12Device) -> ID3D12RootSignature {
    let params = [D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 16 },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
    }];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        ..Default::default()
    };
    let mut blob: Option<ID3DBlob> = None;
    let mut error: Option<ID3DBlob> = None;
    unsafe {
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, Some(&mut error))
            .unwrap_or_else(|e| panic!("root signature : {e} {}", blob_text(error.as_ref())));
        device.CreateRootSignature(0, blob_bytes(blob.as_ref().expect("blob"))).expect("root signature")
    }
}

fn create_pipeline(device: &ID3D12Device, root: &ID3D12RootSignature) -> ID3D12PipelineState {
    let vs = compile(s!("vs_main"), s!("vs_5_0"));
    let ps = compile(s!("ps_main"), s!("ps_5_0"));
    let bytecode = |blob: &ID3DBlob| {
        let bytes = unsafe { blob_bytes(blob) };
        D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() }
    };
    let element = |name, offset| D3D12_INPUT_ELEMENT_DESC {
        SemanticName: name,
        Format: DXGI_FORMAT_R32G32B32_FLOAT,
        AlignedByteOffset: offset,
        InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
        ..Default::default()
    };
    let elements = [element(s!("POSITION"), 0), element(s!("COLOR"), 12)];
    let mut blend = D3D12_BLEND_DESC::default();
    blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
    rtv_formats[0] = COLOR_FORMAT;
    let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: unsafe { std::mem::transmute_copy(root) },
        VS: bytecode(&vs),
        PS: bytecode(&ps),
        BlendState: blend,
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_BACK,
            FrontCounterClockwise: true.into(),
            DepthClipEnable: true.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            ..Default::default()
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elements.as_ptr(), NumElements: elements.len() as u32 },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: rtv_formats,
        DSVFormat: DEPTH_FORMAT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    unsafe { device.CreateGraphicsPipelineState(&desc) }.expect("PSO D3D12")
}

/// HLSL → DXBC par le compilateur système (`d3dcompiler_47.dll`, livré avec Windows).
fn compile(entry: PCSTR, target: PCSTR) -> ID3DBlob {
    let source = include_str!("cube.hlsl");
    let mut code: Option<ID3DBlob> = None;
    let mut error: Option<ID3DBlob> = None;
    unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("cube.hlsl"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut error),
        )
    }
    .unwrap_or_else(|e| panic!("HLSL : {e} {}", blob_text(error.as_ref())));
    code.expect("bytecode")
}

unsafe fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) }
}

fn blob_text(blob: Option<&ID3DBlob>) -> String {
    blob.map(|b| String::from_utf8_lossy(unsafe { blob_bytes(b) }).into_owned()).unwrap_or_default()
}
