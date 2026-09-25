//! Moteur 3D wgpu sur le `wgpu::Device` de GPUI : rendu direct dans le back buffer
//! de la surface, zéro copie.

use gpui3d_shell::{Backends, CLEAR_COLOR, CUBE_INDICES, CUBE_VERTICES, Renderer, Scene, Surface, Ui, WgpuSurfaceHandle};
use wgpu::util::DeviceExt;

struct WgpuCube {
    surface: WgpuSurfaceHandle,
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth: Option<(wgpu::TextureView, (u32, u32))>,
}

impl Renderer for WgpuCube {
    fn new(surface: &Surface) -> Self {
        let surface = surface.as_wgpu().expect("l'UI de GPUI ne tourne pas sur wgpu");
        let device = surface.device();
        let shader = device.create_shader_module(wgpu::include_wgsl!("cube.wgsl"));
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cube_uniforms"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cube"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 24,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(surface.format().into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube_vb"),
            contents: bytemuck::cast_slice(&CUBE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube_ib"),
            contents: bytemuck::cast_slice(&CUBE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self { surface, pipeline, vertices, indices, uniforms, bind_group, depth: None }
    }

    fn render(&mut self, _surface: &Surface, scene: &Scene) -> bool {
        let surface = self.surface.clone();
        let Some((view, (w, h))) = surface.back_view_with_size() else { return false };
        let device = surface.device();
        if self.depth.as_ref().map(|d| d.1) != Some((w, h)) {
            // Transient : mémoire tuile seulement sur Apple Silicon, jamais écrite en VRAM.
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("cube_depth"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TRANSIENT_ATTACHMENT,
                view_formats: &[],
            });
            self.depth = Some((depth.create_view(&Default::default()), (w, h)));
        }
        let depth = &self.depth.as_ref().expect("profondeur").0;
        surface.queue().write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&scene.mvp(w, h)));

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let [r, g, b, a] = CLEAR_COLOR;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cube"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..CUBE_INDICES.len() as u32, 0, 0..1);
        }
        surface.present_synced_silent(surface.queue().submit([encoder.finish()]));
        true
    }
}

fn main() {
    // Tous les backends wgpu ; `WGPU_BACKEND=vulkan|metal|dx12|gl` en force un (vide = tous).
    let backends = Backends::from_env().filter(|b| !b.is_empty()).unwrap_or(Backends::all());
    gpui3d_shell::run::<WgpuCube>("wgpu", Ui::Wgpu(backends));
}
