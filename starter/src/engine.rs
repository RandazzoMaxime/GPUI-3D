//! Moteur 3D du starter : wgpu sur le device de GPUI, son propre fil, une `WgpuSurface`.
//! Il ne connaît pas l'UI : il lit un instantané (`Shared::scene`) que l'UI publie
//! seulement quand la scène change, et une caméra que les gestes écrivent.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use glam::{Mat4, Vec3};
use gpui::WgpuSurfaceHandle;
use wgpu::util::DeviceExt;

pub const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
const CLEAR: wgpu::Color = wgpu::Color { r: 0.80, g: 0.84, b: 0.90, a: 1.0 };

/// Un objet tel que le GPU le dessine (instance du cube).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub position: [f32; 3],
    pub scale: [f32; 3],
    pub color: [f32; 3],
    /// bit 0 : tourne sur lui-même, bit 1 : sélectionné.
    pub flags: u32,
}

pub const SPIN: u32 = 1;
pub const SELECTED: u32 = 2;

#[derive(Clone, Copy)]
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { yaw: 0.7, pitch: 0.45, distance: 11.0 }
    }
}

impl OrbitCamera {
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.01;
        self.pitch = (self.pitch + dy * 0.01).clamp(0.05, 1.5);
    }

    pub fn zoom(&mut self, dy: f32) {
        self.distance = (self.distance * (-dy * 0.002).exp()).clamp(3.0, 40.0);
    }

    fn view_proj(&self, aspect: f32) -> Mat4 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let eye = Vec3::new(sy * cp, sp, cy * cp) * self.distance;
        Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 200.0)
            * Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y)
    }
}

/// Ce que l'UI et le fil de rendu partagent. Aucun des deux n'attend l'autre plus que
/// le temps d'une copie.
#[derive(Default)]
pub struct Shared {
    pub scene: Mutex<Vec<Instance>>,
    pub camera: Mutex<OrbitCamera>,
    /// Incrémenté à chaque publication : le moteur ne renvoie les instances au GPU
    /// que si la scène a changé.
    pub version: AtomicU64,
    pub frames: AtomicU64,
}

impl Shared {
    pub fn publish(&self, instances: Vec<Instance>) {
        *self.scene.lock().unwrap() = instances;
        self.version.fetch_add(1, Ordering::Release);
    }
}

/// Démarre le moteur sur son fil, cadencé à `hz` (le rafraîchissement de l'écran).
pub fn spawn(surface: WgpuSurfaceHandle, shared: Arc<Shared>, hz: f64) {
    let period = Duration::from_secs_f64(1.0 / hz.max(1.0));
    std::thread::Builder::new()
        .name("engine-3d".into())
        .spawn(move || {
            let mut engine = Engine::new(&surface);
            let start = Instant::now();
            let mut deadline = start;
            loop {
                let guard = surface.submit_guard();
                if engine.render(&surface, &shared, start.elapsed().as_secs_f32()) {
                    shared.frames.fetch_add(1, Ordering::Relaxed);
                }
                drop(guard);
                // Réveille la fenêtre pour composer la trame, sans redessiner l'UI.
                surface.request_window_redraw();
                deadline = (deadline + period).max(Instant::now());
                std::thread::sleep(deadline - Instant::now());
            }
        })
        .expect("fil engine-3d");
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Frame {
    view_proj: [[f32; 4]; 4],
    light_dir: [f32; 4],
    time: f32,
    _pad: [f32; 3],
}

/// Le sol, toujours dessiné en premier.
const GROUND: Instance = Instance {
    position: [0.0, -0.52, 0.0],
    scale: [14.0, 0.04, 14.0],
    color: [0.93, 0.94, 0.96],
    flags: 0,
};

struct Engine {
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    instances: wgpu::Buffer,
    instance_count: u32,
    uploaded_version: u64,
    frame: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth: Option<(wgpu::TextureView, (u32, u32))>,
}

impl Engine {
    fn new(surface: &WgpuSurfaceHandle) -> Self {
        let device = surface.device();
        let shader = device.create_shader_module(wgpu::include_wgsl!("scene.wgsl"));
        let frame = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame"),
            size: std::mem::size_of::<Frame>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: frame.as_entire_binding() }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 24,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Instance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            2 => Float32x3, 3 => Float32x3, 4 => Float32x3, 5 => Uint32
                        ],
                    }),
                ],
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
        let (vertices, indices) = cube_mesh();
        let index_count = indices.len() as u32;
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube_vb"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube_ib"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            pipeline,
            vertices,
            indices,
            index_count,
            instances: instance_buffer(device, 64),
            instance_count: 0,
            uploaded_version: u64::MAX,
            frame,
            bind_group,
            depth: None,
        }
    }

    fn render(&mut self, surface: &WgpuSurfaceHandle, shared: &Shared, time: f32) -> bool {
        let Some((view, (w, h))) = surface.back_view_with_size() else { return false };
        let (device, queue) = (surface.device(), surface.queue());

        let version = shared.version.load(Ordering::Acquire);
        if version != self.uploaded_version {
            let mut instances = vec![GROUND];
            instances.extend_from_slice(&shared.scene.lock().unwrap());
            let bytes: &[u8] = bytemuck::cast_slice(&instances);
            if self.instances.size() < bytes.len() as u64 {
                self.instances = instance_buffer(device, instances.len().next_power_of_two());
            }
            queue.write_buffer(&self.instances, 0, bytes);
            self.instance_count = instances.len() as u32;
            self.uploaded_version = version;
        }

        if self.depth.as_ref().map(|d| d.1) != Some((w, h)) {
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("depth"),
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
        let camera = *shared.camera.lock().unwrap();
        let frame = Frame {
            view_proj: camera.view_proj(w as f32 / h.max(1) as f32).to_cols_array_2d(),
            light_dir: Vec3::new(-0.4, -1.0, -0.3).normalize().extend(0.0).to_array(),
            time,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&frame));

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(CLEAR), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.as_ref().expect("profondeur").0,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.set_vertex_buffer(1, self.instances.slice(..));
            pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..self.index_count, 0, 0..self.instance_count);
        }
        surface.present_synced_silent(queue.submit([encoder.finish()]));
        true
    }
}

fn instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("instances"),
        size: (capacity * std::mem::size_of::<Instance>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Cube unité (côté 1) : 4 sommets par face pour des normales nettes.
fn cube_mesh() -> (Vec<[f32; 6]>, Vec<u16>) {
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([0., 0., 1.], [1., 0., 0.], [0., 1., 0.]),
        ([0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]),
        ([1., 0., 0.], [0., 0., -1.], [0., 1., 0.]),
        ([-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]),
        ([0., 1., 0.], [1., 0., 0.], [0., 0., -1.]),
        ([0., -1., 0.], [1., 0., 0.], [0., 0., 1.]),
    ];
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    for (normal, u, v) in faces {
        let base = vertices.len() as u16;
        for (su, sv) in [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)] {
            let p = |i: usize| 0.5 * (normal[i] + su * u[i] + sv * v[i]);
            vertices.push([p(0), p(1), p(2), normal[0], normal[1], normal[2]]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
}
