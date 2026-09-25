//! Implémentation OpenGL 4.5 core native (WGL, glow) de [`Gpu`], sans wgpu. Windows.
//!
//! Un contexte GL n'est courant que sur un fil à la fois, alors que le HAL est appelé de
//! partout. Tout accès GL passe donc par une *section* : verrou, contexte rendu courant
//! sur le fil appelant, travail, contexte relâché. Pour tenir le nombre de sections bas
//! (deux par trame), passes et copies sont enregistrées en mémoire puis rejouées au
//! `submit`, précédées des écritures en attente (contrat `write_*` de [`super`]).
//!
//! Convention de wgpu-GL : naga retourne Y dans les vertex shaders, la ligne 0 des
//! textures est donc en haut comme en WebGPU (viewport, scissor et copies inchangés) ; la
//! présentation retourne l'image par un blit vers la fenêtre.
//!
//! Moteurs 3D natifs : ils créent leur propre contexte partageant les objets de celui-ci
//! (sous [`Gpu::queue_lock`]), rendent dans la texture de surface et terminent par
//! `glFinish` avant de la publier. Côté UI, chaque présentation se termine aussi par
//! `glFinish`, de sorte qu'un tampon rendu au moteur n'est plus lu par le compositeur.

use std::ffi::CStr;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use anyhow::{Context as _, Result, anyhow};
use glow::HasContext as _;
use parking_lot::Mutex;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::Graphics::Gdi::{GetDC, HDC, ReleaseDC};
use windows::Win32::Graphics::OpenGL::{
    ChoosePixelFormat, DescribePixelFormat, HGLRC, PFD_DOUBLEBUFFER, PFD_DRAW_TO_WINDOW, PFD_SUPPORT_OPENGL,
    PFD_TYPE_RGBA, PIXELFORMATDESCRIPTOR, SetPixelFormat, SwapBuffers, wglCreateContext, wglDeleteContext,
    wglGetProcAddress, wglMakeCurrent,
};
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryA};
use windows::Win32::UI::WindowsAndMessaging::{
    CS_OWNDC, CreateWindowExW, DefWindowProcW, DestroyWindow, RegisterClassW, WINDOW_EX_STYLE, WNDCLASSW,
    WS_POPUP,
};
use windows::core::{PCSTR, s, w};

use super::{
    Acquire, BindEntry, BindResource, BindingKind, Blend, BufferUsage, Gpu, LayoutEntry, LoadOp, PassDesc,
    PipelineDesc, TextureUsage, Topology,
};
use crate::{GpuSpecs, NativeDevice, NativeTexture, SurfaceFormat, WindowPresentMode};

mod sources {
    include!(concat!(env!("OUT_DIR"), "/glsl.rs"));
}

/// Taille minimale garantie de `GL_MAX_UNIFORM_BLOCK_SIZE`.
const MAX_UNIFORM_BLOCK: u64 = 16384;

type CreateContextAttribs = unsafe extern "system" fn(HDC, HGLRC, *const i32) -> HGLRC;
type SwapInterval = unsafe extern "system" fn(i32) -> i32;

/// Device OpenGL partagé par l'UI et les moteurs 3D d'une application.
#[derive(Clone)]
pub(crate) struct GlGpu(Arc<Shared>);

struct Shared {
    gl: glow::Context,
    context: HGLRC,
    window: HWND,
    window_dc: HDC,
    pixel_format: i32,
    swap_interval: Option<SwapInterval>,
    /// Exclusivité du contexte : tenu pendant toute section.
    section: Mutex<()>,
    pending: Mutex<Pending>,
    sampler: glow::Sampler,
    vertex_array: glow::VertexArray,
    uniform_alignment: u32,
    renderer: String,
    version: String,
}

// SAFETY: le contexte n'est utilisé que sous `section`, courant sur le fil appelant ;
// les poignées Win32 restent valides pour toute la vie de `Shared`.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

#[derive(Default)]
struct Pending {
    uploads: Vec<Upload>,
    garbage: Vec<Garbage>,
}

enum Upload {
    Buffer { buffer: GlBuffer, offset: u64, data: Vec<u8> },
    Texture { texture: GlTexture, origin: (u32, u32), size: (u32, u32), data: Vec<u8> },
}

/// Objets relâchés hors section, supprimés à la section suivante.
enum Garbage {
    Buffer(glow::Buffer),
    Texture(glow::Texture),
    Framebuffer(glow::Framebuffer),
    Program(glow::Program),
}

/// Contexte courant sur le fil appelant tant qu'elle vit.
struct Section<'a> {
    shared: &'a Shared,
    _lock: parking_lot::MutexGuard<'a, ()>,
}

impl Drop for Section<'_> {
    fn drop(&mut self) {
        // SAFETY: relâche le contexte rendu courant par `Shared::section`.
        if let Err(error) = unsafe { wglMakeCurrent(HDC::default(), HGLRC::default()) } {
            log::error!("relâche du contexte OpenGL : {error}");
        }
    }
}

impl Shared {
    fn section(&self, dc: HDC) -> Section<'_> {
        let lock = self.section.lock();
        // SAFETY: contexte libre (verrou tenu), `dc` au format de pixel du contexte.
        unsafe { wglMakeCurrent(dc, self.context) }.expect("contexte OpenGL courant");
        let garbage = std::mem::take(&mut self.pending.lock().garbage);
        for object in garbage {
            // SAFETY: contexte courant ; plus aucune référence Rust à ces objets.
            unsafe {
                match object {
                    Garbage::Buffer(buffer) => self.gl.delete_buffer(buffer),
                    Garbage::Texture(texture) => self.gl.delete_texture(texture),
                    Garbage::Framebuffer(framebuffer) => self.gl.delete_framebuffer(framebuffer),
                    Garbage::Program(program) => self.gl.delete_program(program),
                }
            }
        }
        Section { shared: self, _lock: lock }
    }

    /// Exécute les écritures en attente (contexte courant).
    fn run_uploads(&self, _section: &Section<'_>) {
        let uploads = std::mem::take(&mut self.pending.lock().uploads);
        let gl = &self.gl;
        for upload in uploads {
            // SAFETY: contexte courant ; objets vivants (retenus par l'entrée).
            unsafe {
                match upload {
                    Upload::Buffer { buffer, offset, data } => {
                        gl.named_buffer_sub_data_u8_slice(buffer.0.name, offset as i32, &data);
                    }
                    Upload::Texture { texture, origin, size, data } => {
                        let (format, kind) = pixel_transfer(texture.0.format);
                        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
                        gl.texture_sub_image_2d(
                            texture.0.name,
                            0,
                            origin.0 as i32,
                            origin.1 as i32,
                            size.0 as i32,
                            size.1 as i32,
                            format,
                            kind,
                            glow::PixelUnpackData::Slice(Some(&data)),
                        );
                    }
                }
            }
        }
    }

    fn framebuffer(&self, texture: &TextureInner) -> glow::Framebuffer {
        *texture.framebuffer.get_or_init(|| {
            // SAFETY: appelé en section ; texture vivante.
            unsafe {
                let framebuffer = self.gl.create_named_framebuffer().expect("framebuffer OpenGL");
                self.gl.named_framebuffer_texture(Some(framebuffer), glow::COLOR_ATTACHMENT0, Some(texture.name), 0);
                framebuffer
            }
        })
    }

    fn execute(&self, section: &Section<'_>, commands: Vec<Command>) {
        self.run_uploads(section);
        let gl = &self.gl;
        let mut pipeline: Option<Arc<PipelineInner>> = None;
        let mut bound: [Option<(GlBindGroup, Vec<u32>)>; 8] = Default::default();
        let mut dirty = [false; 8];
        for command in commands {
            // SAFETY: contexte courant (section) ; objets retenus par la commande.
            unsafe {
                match command {
                    Command::BeginPass { target, load } => {
                        let framebuffer = self.framebuffer(&target.0);
                        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
                        gl.disable(glow::SCISSOR_TEST);
                        if let LoadOp::Clear(color) = load {
                            let color = color.map(|channel| channel as f32);
                            gl.clear_named_framebuffer_f32_slice(Some(framebuffer), glow::COLOR, 0, &color);
                        }
                        let (width, height) = (target.0.width as i32, target.0.height as i32);
                        gl.viewport(0, 0, width, height);
                        gl.enable(glow::SCISSOR_TEST);
                        gl.scissor(0, 0, width, height);
                        pipeline = None;
                        bound = Default::default();
                    }
                    Command::SetPipeline(next) => {
                        gl.use_program(Some(next.program));
                        gl.enable(glow::BLEND);
                        gl.blend_equation(glow::FUNC_ADD);
                        gl.blend_func_separate(
                            next.color_source,
                            glow::ONE_MINUS_SRC_ALPHA,
                            glow::ONE,
                            glow::ONE_MINUS_SRC_ALPHA,
                        );
                        pipeline = Some(next);
                        dirty = [true; 8];
                    }
                    Command::SetBindGroup(index, group, offsets) => {
                        if let Some(slot) = bound.get_mut(index as usize) {
                            *slot = Some((group, offsets));
                            dirty[index as usize] = true;
                        }
                    }
                    Command::Viewport([x, y, width, height]) => gl.viewport_f32_slice(0, 1, &[[x, y, width, height]]),
                    Command::Scissor([x, y, width, height]) => gl.scissor(x, y, width, height),
                    Command::Draw(vertices, instances) => {
                        let Some(pipeline) = pipeline.as_ref() else {
                            log::error!("draw OpenGL sans pipeline");
                            continue;
                        };
                        for (index, entry) in bound.iter().enumerate() {
                            if !dirty[index] {
                                continue;
                            }
                            if let Some((group, offsets)) = entry {
                                self.bind_group(index as u32, group, offsets);
                            }
                            dirty[index] = false;
                        }
                        if let Some(location) = &pipeline.first_instance {
                            gl.uniform_1_u32(Some(location), instances.start);
                        }
                        gl.draw_arrays_instanced(
                            pipeline.mode,
                            vertices.start as i32,
                            (vertices.end - vertices.start) as i32,
                            (instances.end - instances.start) as i32,
                        );
                    }
                    Command::CopyBuffer { source, source_offset, destination, destination_offset, size } => {
                        gl.bind_buffer(glow::COPY_READ_BUFFER, Some(source.0.name));
                        gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(destination.0.name));
                        gl.copy_buffer_sub_data(
                            glow::COPY_READ_BUFFER,
                            glow::COPY_WRITE_BUFFER,
                            source_offset as i32,
                            destination_offset as i32,
                            size as i32,
                        );
                    }
                    Command::CopyTexture { source, destination, width, height } => {
                        gl.copy_image_sub_data(
                            source.0.name,
                            glow::TEXTURE_2D,
                            0,
                            0,
                            0,
                            0,
                            destination.0.name,
                            glow::TEXTURE_2D,
                            0,
                            0,
                            0,
                            0,
                            width as i32,
                            height as i32,
                            1,
                        );
                    }
                    Command::Present { source, width, height } => {
                        let framebuffer = self.framebuffer(&source.0);
                        gl.disable(glow::SCISSOR_TEST);
                        // Ligne 0 des textures en haut, de la fenêtre en bas : blit retourné.
                        gl.blit_named_framebuffer(
                            Some(framebuffer),
                            None,
                            0,
                            0,
                            width as i32,
                            height as i32,
                            0,
                            height as i32,
                            width as i32,
                            0,
                            glow::COLOR_BUFFER_BIT,
                            glow::NEAREST,
                        );
                    }
                }
            }
        }
    }

    /// Lie un groupe au point `groupe * BINDING_STRIDE + binding` de chaque classe.
    unsafe fn bind_group(&self, index: u32, group: &GlBindGroup, offsets: &[u32]) {
        let gl = &self.gl;
        let mut dynamic = offsets.iter();
        for entry in &group.0.entries {
            let slot = index * sources::BINDING_STRIDE + entry.binding;
            // SAFETY: contexte courant ; ressources retenues par le groupe.
            unsafe {
                match &entry.resource {
                    GroupResource::Uniform { buffer, offset, size, dynamic: has_dynamic } => {
                        let extra = if *has_dynamic { dynamic.next().copied().unwrap_or(0) } else { 0 };
                        let size = size.unwrap_or(buffer.0.size - offset).min(MAX_UNIFORM_BLOCK);
                        gl.bind_buffer_range(
                            glow::UNIFORM_BUFFER,
                            slot,
                            Some(buffer.0.name),
                            (offset + u64::from(extra)) as i32,
                            size as i32,
                        );
                    }
                    GroupResource::Storage { buffer, offset, size } => {
                        let size = size.unwrap_or(buffer.0.size - offset);
                        gl.bind_buffer_range(
                            glow::SHADER_STORAGE_BUFFER,
                            slot,
                            Some(buffer.0.name),
                            *offset as i32,
                            size as i32,
                        );
                    }
                    GroupResource::Texture(texture) => {
                        gl.bind_texture_unit(slot, Some(texture.0.name));
                        gl.bind_sampler(slot, Some(self.sampler));
                    }
                }
            }
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        {
            let section = self.section(self.window_dc);
            // SAFETY: contexte courant.
            unsafe {
                section.shared.gl.finish();
                section.shared.gl.delete_sampler(self.sampler);
                section.shared.gl.delete_vertex_array(self.vertex_array);
            }
        }
        // SAFETY: plus aucune section ; contexte, DC et fenêtre à nous.
        unsafe {
            if let Err(error) = wglDeleteContext(self.context) {
                log::error!("suppression du contexte OpenGL : {error}");
            }
            ReleaseDC(Some(self.window), self.window_dc);
            if let Err(error) = DestroyWindow(self.window) {
                log::debug!("fenêtre cachée OpenGL : {error}");
            }
        }
    }
}

/// Format et type de transfert d'un format interne (lignes jointives).
fn pixel_transfer(format: u32) -> (u32, u32) {
    match format {
        glow::R8 => (glow::RED, glow::UNSIGNED_BYTE),
        glow::RGBA16F => (glow::RGBA, glow::HALF_FLOAT),
        _ => (glow::RGBA, glow::UNSIGNED_BYTE),
    }
}

// --- Ressources -----------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct GlBuffer(Arc<BufferInner>);

struct BufferInner {
    shared: Arc<Shared>,
    name: glow::Buffer,
    size: u64,
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        self.shared.pending.lock().garbage.push(Garbage::Buffer(self.name));
    }
}

#[derive(Clone)]
pub(crate) struct GlTexture(Arc<TextureInner>);

struct TextureInner {
    shared: Arc<Shared>,
    name: glow::Texture,
    format: u32,
    width: u32,
    height: u32,
    /// Framebuffer de la texture comme cible ou source de blit, créé à la première passe.
    framebuffer: OnceLock<glow::Framebuffer>,
}

impl Drop for TextureInner {
    fn drop(&mut self) {
        let mut pending = self.shared.pending.lock();
        pending.garbage.push(Garbage::Texture(self.name));
        if let Some(framebuffer) = self.framebuffer.get() {
            pending.garbage.push(Garbage::Framebuffer(*framebuffer));
        }
    }
}

#[derive(Clone)]
pub(crate) struct GlTextureView(GlTexture);

/// L'unique sampler du HAL (linéaire, bords clampés) vit dans `Shared`.
pub(crate) struct GlSampler;

pub(crate) struct GlBindGroupLayout(Arc<Vec<LayoutEntry>>);

#[derive(Clone)]
pub(crate) struct GlBindGroup(Arc<GroupInner>);

struct GroupInner {
    entries: Vec<GroupEntry>,
}

struct GroupEntry {
    binding: u32,
    resource: GroupResource,
}

enum GroupResource {
    Uniform { buffer: GlBuffer, offset: u64, size: Option<u64>, dynamic: bool },
    Storage { buffer: GlBuffer, offset: u64, size: Option<u64> },
    Texture(GlTexture),
}

pub(crate) struct GlPipeline(Arc<PipelineInner>);

struct PipelineInner {
    shared: Arc<Shared>,
    program: glow::Program,
    mode: u32,
    color_source: u32,
    /// `naga_vs_first_instance` : `gl_InstanceID` n'inclut pas la première instance.
    first_instance: Option<glow::UniformLocation>,
}

impl Drop for PipelineInner {
    fn drop(&mut self) {
        self.shared.pending.lock().garbage.push(Garbage::Program(self.program));
    }
}

enum Command {
    BeginPass { target: GlTexture, load: LoadOp },
    SetPipeline(Arc<PipelineInner>),
    SetBindGroup(u32, GlBindGroup, Vec<u32>),
    Viewport([f32; 4]),
    Scissor([i32; 4]),
    Draw(Range<u32>, Range<u32>),
    CopyBuffer { source: GlBuffer, source_offset: u64, destination: GlBuffer, destination_offset: u64, size: u64 },
    CopyTexture { source: GlTexture, destination: GlTexture, width: u32, height: u32 },
    Present { source: GlTexture, width: u32, height: u32 },
}

pub(crate) struct GlEncoder {
    commands: Vec<Command>,
    /// DC de la fenêtre présentée par cette trame, s'il y en a une.
    frame_dc: Option<HDC>,
}

pub(crate) struct GlPass<'a> {
    encoder: &'a mut GlEncoder,
}

pub(crate) struct GlSwapchain {
    window: HWND,
    dc: HDC,
    width: u32,
    height: u32,
    present_mode: WindowPresentMode,
    configured: bool,
}

impl Drop for GlSwapchain {
    fn drop(&mut self) {
        // SAFETY: DC obtenu par `GetDC` sur cette fenêtre, rendu une seule fois ; les
        // sections relâchent toujours le contexte, il n'y est donc plus courant.
        unsafe { ReleaseDC(Some(self.window), self.dc) };
    }
}

pub(crate) struct GlFrame {
    dc: HDC,
    interval: i32,
}

// --- Création du device ---------------------------------------------------------------

unsafe extern "system" fn window_proc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    // SAFETY: fenêtre cachée sans comportement propre.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn pixel_format_descriptor() -> PIXELFORMATDESCRIPTOR {
    PIXELFORMATDESCRIPTOR {
        nSize: size_of::<PIXELFORMATDESCRIPTOR>() as u16,
        nVersion: 1,
        dwFlags: PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
        iPixelType: PFD_TYPE_RGBA,
        cColorBits: 32,
        cAlphaBits: 8,
        ..Default::default()
    }
}

impl GlGpu {
    /// Contexte OpenGL 4.5 core (WGL) sur une fenêtre cachée ; les fenêtres de l'UI
    /// prennent le même format de pixel. `GPUI_GL_DEBUG=1` : contexte de debug, erreurs
    /// du pilote relayées sur stderr.
    pub(crate) fn new() -> Result<Self> {
        let debug = std::env::var("GPUI_GL_DEBUG").is_ok_and(|value| value == "1");
        // SAFETY: création Win32/WGL classique ; chaque objet est libéré par `Shared::drop`.
        unsafe {
            let instance = GetModuleHandleW(None)?;
            let class = WNDCLASSW {
                style: CS_OWNDC,
                lpfnWndProc: Some(window_proc),
                hInstance: instance.into(),
                lpszClassName: w!("gpui-opengl-device"),
                ..Default::default()
            };
            RegisterClassW(&class);
            let window = CreateWindowExW(
                WINDOW_EX_STYLE(0),
                w!("gpui-opengl-device"),
                w!("gpui-opengl-device"),
                WS_POPUP,
                0,
                0,
                1,
                1,
                None,
                None,
                Some(instance.into()),
                None,
            )
            .context("fenêtre cachée OpenGL")?;
            let window_dc = GetDC(Some(window));
            let descriptor = pixel_format_descriptor();
            let pixel_format = ChoosePixelFormat(window_dc, &descriptor);
            if pixel_format == 0 {
                anyhow::bail!("aucun format de pixel OpenGL RGBA8 double tampon");
            }
            SetPixelFormat(window_dc, pixel_format, &descriptor).context("format de pixel OpenGL")?;

            // Contexte hérité le temps d'obtenir wglCreateContextAttribsARB.
            let legacy = wglCreateContext(window_dc).context("contexte OpenGL")?;
            wglMakeCurrent(window_dc, legacy).context("contexte OpenGL courant")?;
            let create: Option<CreateContextAttribs> =
                wglGetProcAddress(s!("wglCreateContextAttribsARB")).map(|function| std::mem::transmute(function));
            let swap_interval: Option<SwapInterval> =
                wglGetProcAddress(s!("wglSwapIntervalEXT")).map(|function| std::mem::transmute(function));
            let create = create.ok_or_else(|| anyhow!("WGL_ARB_create_context absent"))?;
            let flags = if debug { 0x0001 } else { 0 };
            // MAJOR, MINOR, PROFILE_MASK = CORE, FLAGS.
            let attributes = [0x2091, 4, 0x2092, 5, 0x9126, 0x0001, 0x2094, flags, 0];
            let context = create(window_dc, HGLRC::default(), attributes.as_ptr());
            wglMakeCurrent(HDC::default(), HGLRC::default())?;
            wglDeleteContext(legacy)?;
            if context.is_invalid() {
                anyhow::bail!("contexte OpenGL 4.5 core refusé par le pilote");
            }
            wglMakeCurrent(window_dc, context).context("contexte OpenGL 4.5 courant")?;

            let opengl32 = LoadLibraryA(s!("opengl32.dll"))?;
            let mut gl = glow::Context::from_loader_function_cstr(|name: &CStr| {
                let name = PCSTR(name.as_ptr().cast());
                match wglGetProcAddress(name).map(|function| function as usize) {
                    // 1, 2, 3 et -1 sont des échecs documentés de wglGetProcAddress.
                    Some(address) if address > 3 && address != usize::MAX => address as *const std::ffi::c_void,
                    _ => GetProcAddress(opengl32, name).map_or(std::ptr::null(), |function| function as *const _),
                }
            });
            if debug {
                gl.enable(glow::DEBUG_OUTPUT_SYNCHRONOUS);
                gl.debug_message_callback(|_source, kind, _id, severity, message| {
                    if kind == glow::DEBUG_TYPE_ERROR || severity == glow::DEBUG_SEVERITY_HIGH {
                        eprintln!("[opengl] {message}");
                    }
                });
            }
            // Le profil core refuse tout draw sans VAO, même sans attribut de sommet ; l'état
            // du contexte persiste d'une section à l'autre.
            let vertex_array = gl.create_vertex_array().map_err(|error| anyhow!("VAO OpenGL : {error}"))?;
            gl.bind_vertex_array(Some(vertex_array));
            let sampler = gl.create_sampler().map_err(|error| anyhow!("sampler OpenGL : {error}"))?;
            for (name, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
            ] {
                gl.sampler_parameter_i32(sampler, name, value as i32);
            }
            let uniform_alignment = gl.get_parameter_i32(glow::UNIFORM_BUFFER_OFFSET_ALIGNMENT).max(1) as u32;
            let renderer = gl.get_parameter_string(glow::RENDERER);
            let version = gl.get_parameter_string(glow::VERSION);
            wglMakeCurrent(HDC::default(), HGLRC::default())?;
            Ok(Self(Arc::new(Shared {
                gl,
                context,
                window,
                window_dc,
                pixel_format,
                swap_interval,
                section: Mutex::new(()),
                pending: Mutex::new(Pending::default()),
                sampler,
                vertex_array,
                uniform_alignment,
                renderer,
                version,
            })))
        }
    }

    fn compile(&self, section: &Section<'_>, kind: u32, source: &str, label: &str) -> glow::Shader {
        let gl = &section.shared.gl;
        // SAFETY: contexte courant (section).
        unsafe {
            let shader = gl.create_shader(kind).unwrap_or_else(|error| panic!("shader OpenGL « {label} » : {error}"));
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                panic!("GLSL « {label} » : {}", gl.get_shader_info_log(shader));
            }
            shader
        }
    }
}

impl Gpu for GlGpu {
    type Format = u32;
    type Buffer = GlBuffer;
    type Texture = GlTexture;
    type TextureView = GlTextureView;
    type Sampler = GlSampler;
    type BindGroupLayout = GlBindGroupLayout;
    type BindGroup = GlBindGroup;
    type Pipeline = GlPipeline;
    type Encoder = GlEncoder;
    type Pass<'a> = GlPass<'a>;
    type Swapchain = GlSwapchain;
    type Frame = GlFrame;
    #[cfg(feature = "flamegraph")]
    type Profiler = super::NoProfiler;

    const ATLAS_MONOCHROME: u32 = glow::R8;
    const ATLAS_POLYCHROME: u32 = glow::RGBA8;

    fn bytes_per_pixel(format: u32) -> u32 {
        match format {
            glow::R8 => 1,
            glow::RGBA16F => 8,
            _ => 4,
        }
    }

    fn gpu_specs(&self) -> GpuSpecs {
        GpuSpecs {
            is_software_emulated: false,
            device_name: self.0.renderer.clone(),
            driver_name: "OpenGL".into(),
            driver_info: self.0.version.clone(),
        }
    }

    fn min_uniform_offset_alignment(&self) -> u32 {
        self.0.uniform_alignment
    }

    fn create_buffer(&self, label: &str, size: u64, _usage: BufferUsage) -> GlBuffer {
        let section = self.0.section(self.0.window_dc);
        let size = size.max(4);
        // SAFETY: contexte courant (section).
        let name = unsafe {
            let gl = &section.shared.gl;
            let name = gl.create_named_buffer().unwrap_or_else(|error| panic!("buffer OpenGL « {label} » : {error}"));
            gl.named_buffer_storage(name, size as i32, None, glow::DYNAMIC_STORAGE_BIT);
            name
        };
        GlBuffer(Arc::new(BufferInner { shared: self.0.clone(), name, size }))
    }

    fn create_buffer_init(&self, label: &str, contents: &[u8], usage: BufferUsage) -> GlBuffer {
        let buffer = self.create_buffer(label, contents.len() as u64, usage);
        self.write_buffer(&buffer, 0, contents);
        buffer
    }

    fn buffer_size(buffer: &GlBuffer) -> u64 {
        buffer.0.size
    }

    fn write_buffer(&self, buffer: &GlBuffer, offset: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        self.0.pending.lock().uploads.push(Upload::Buffer { buffer: buffer.clone(), offset, data: data.to_vec() });
    }

    fn create_texture(&self, label: &str, width: u32, height: u32, format: u32, _usage: TextureUsage) -> GlTexture {
        let (width, height) = (width.max(1), height.max(1));
        let section = self.0.section(self.0.window_dc);
        // SAFETY: contexte courant (section).
        let name = unsafe {
            let gl = &section.shared.gl;
            let name =
                gl.create_named_texture(glow::TEXTURE_2D).unwrap_or_else(|error| panic!("texture OpenGL « {label} » : {error}"));
            gl.texture_storage_2d(name, 1, format, width as i32, height as i32);
            name
        };
        GlTexture(Arc::new(TextureInner {
            shared: self.0.clone(),
            name,
            format,
            width,
            height,
            framebuffer: OnceLock::new(),
        }))
    }

    fn texture_size(texture: &GlTexture) -> (u32, u32) {
        (texture.0.width, texture.0.height)
    }

    fn create_view(texture: &GlTexture) -> GlTextureView {
        GlTextureView(texture.clone())
    }

    fn write_texture(&self, texture: &GlTexture, origin: (u32, u32), size: (u32, u32), _bytes_per_pixel: u32, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        self.0.pending.lock().uploads.push(Upload::Texture {
            texture: texture.clone(),
            origin,
            size,
            data: data.to_vec(),
        });
    }

    fn create_linear_sampler(&self, _label: &str) -> GlSampler {
        GlSampler
    }

    fn create_bind_group_layout(&self, _label: &str, entries: &[LayoutEntry]) -> GlBindGroupLayout {
        GlBindGroupLayout(Arc::new(entries.to_vec()))
    }

    fn create_bind_group(&self, _label: &str, layout: &GlBindGroupLayout, entries: &[BindEntry<'_, Self>]) -> GlBindGroup {
        let resource_of = |binding: u32| entries.iter().find(|entry| entry.binding == binding).map(|entry| &entry.resource);
        let entries = layout
            .0
            .iter()
            .filter_map(|entry| {
                let resource = match (entry.kind, resource_of(entry.binding)) {
                    (BindingKind::Uniform { dynamic_offset, .. }, Some(BindResource::Buffer { buffer, offset, size })) => {
                        GroupResource::Uniform {
                            buffer: (*buffer).clone(),
                            offset: *offset,
                            size: *size,
                            dynamic: dynamic_offset,
                        }
                    }
                    (BindingKind::Storage, Some(BindResource::Buffer { buffer, offset, size })) => {
                        GroupResource::Storage { buffer: (*buffer).clone(), offset: *offset, size: *size }
                    }
                    (BindingKind::Texture, Some(BindResource::Texture(view))) => GroupResource::Texture(view.0.clone()),
                    // Lié avec sa texture (sampler combiné GLSL).
                    (BindingKind::Sampler, _) => return None,
                    (kind, _) => panic!("groupe OpenGL : ressource absente ou incompatible pour {kind:?} @binding({})", entry.binding),
                };
                Some(GroupEntry { binding: entry.binding, resource })
            })
            .collect();
        GlBindGroup(Arc::new(GroupInner { entries }))
    }

    fn create_pipeline(&self, desc: &PipelineDesc<'_, Self>) -> GlPipeline {
        let source = |entry: &str| {
            sources::glsl(desc.shader.name(), entry)
                .unwrap_or_else(|| panic!("GLSL absent : {}::{entry}", desc.shader.name()))
        };
        let section = self.0.section(self.0.window_dc);
        let vertex = self.compile(&section, glow::VERTEX_SHADER, source(desc.vertex_entry), desc.label);
        let fragment = self.compile(&section, glow::FRAGMENT_SHADER, source(desc.fragment_entry), desc.label);
        let gl = &section.shared.gl;
        // SAFETY: contexte courant (section).
        let (program, first_instance) = unsafe {
            let program = gl.create_program().unwrap_or_else(|error| panic!("programme OpenGL « {} » : {error}", desc.label));
            gl.attach_shader(program, vertex);
            gl.attach_shader(program, fragment);
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                panic!("édition de liens GLSL « {} » : {}", desc.label, gl.get_program_info_log(program));
            }
            gl.detach_shader(program, vertex);
            gl.detach_shader(program, fragment);
            gl.delete_shader(vertex);
            gl.delete_shader(fragment);
            (program, gl.get_uniform_location(program, "naga_vs_first_instance"))
        };
        GlPipeline(Arc::new(PipelineInner {
            shared: self.0.clone(),
            program,
            mode: match desc.topology {
                Topology::TriangleList => glow::TRIANGLES,
                Topology::TriangleStrip => glow::TRIANGLE_STRIP,
            },
            // Mêmes équations que `wgpu::BlendState::{ALPHA_BLENDING, PREMULTIPLIED_ALPHA_BLENDING}`.
            color_source: match desc.blend {
                Blend::Alpha => glow::SRC_ALPHA,
                Blend::PremultipliedAlpha => glow::ONE,
            },
            first_instance,
        }))
    }

    fn create_encoder(&self, _label: &str) -> GlEncoder {
        GlEncoder { commands: Vec::new(), frame_dc: None }
    }

    fn begin_pass<'a>(encoder: &'a mut GlEncoder, desc: &PassDesc<'_, Self>) -> GlPass<'a> {
        encoder.commands.push(Command::BeginPass { target: desc.target.0.clone(), load: desc.load });
        GlPass { encoder }
    }

    fn set_pipeline(pass: &mut GlPass<'_>, pipeline: &GlPipeline) {
        pass.encoder.commands.push(Command::SetPipeline(pipeline.0.clone()));
    }

    fn set_bind_group(pass: &mut GlPass<'_>, index: u32, group: &GlBindGroup, dynamic_offsets: &[u32]) {
        pass.encoder.commands.push(Command::SetBindGroup(index, group.clone(), dynamic_offsets.to_vec()));
    }

    fn set_viewport(pass: &mut GlPass<'_>, x: f32, y: f32, width: f32, height: f32) {
        pass.encoder.commands.push(Command::Viewport([x, y, width, height]));
    }

    fn set_scissor_rect(pass: &mut GlPass<'_>, x: u32, y: u32, width: u32, height: u32) {
        pass.encoder.commands.push(Command::Scissor([x as i32, y as i32, width as i32, height as i32]));
    }

    fn draw(pass: &mut GlPass<'_>, vertices: Range<u32>, instances: Range<u32>) {
        pass.encoder.commands.push(Command::Draw(vertices, instances));
    }

    fn copy_buffer_to_buffer(
        encoder: &mut GlEncoder,
        source: &GlBuffer,
        source_offset: u64,
        destination: &GlBuffer,
        destination_offset: u64,
        size: u64,
    ) {
        encoder.commands.push(Command::CopyBuffer {
            source: source.clone(),
            source_offset,
            destination: destination.clone(),
            destination_offset,
            size,
        });
    }

    fn copy_texture_to_texture(encoder: &mut GlEncoder, source: &GlTexture, destination: &GlTexture, width: u32, height: u32) {
        encoder.commands.push(Command::CopyTexture { source: source.clone(), destination: destination.clone(), width, height });
    }

    fn create_swapchain(
        &self,
        window: raw_window_handle::RawWindowHandle,
        _display: raw_window_handle::RawDisplayHandle,
        width: u32,
        height: u32,
    ) -> Result<GlSwapchain> {
        let raw_window_handle::RawWindowHandle::Win32(handle) = window else {
            anyhow::bail!("OpenGL (WGL) exige une fenêtre Win32");
        };
        let window = HWND(handle.hwnd.get() as *mut std::ffi::c_void);
        // SAFETY: fenêtre vivante (elle survit au renderer) ; format de pixel du contexte.
        let dc = unsafe {
            let dc = GetDC(Some(window));
            let mut descriptor = PIXELFORMATDESCRIPTOR::default();
            DescribePixelFormat(dc, self.0.pixel_format, size_of::<PIXELFORMATDESCRIPTOR>() as u32, Some(&mut descriptor));
            SetPixelFormat(dc, self.0.pixel_format, &descriptor).context("format de pixel OpenGL de la fenêtre")?;
            dc
        };
        Ok(GlSwapchain { window, dc, width, height, present_mode: WindowPresentMode::Fifo, configured: false })
    }

    fn swapchain_format(_swapchain: &GlSwapchain) -> u32 {
        glow::RGBA8
    }

    fn swapchain_premultiplied(_swapchain: &GlSwapchain) -> bool {
        false
    }

    fn swapchain_size(swapchain: &GlSwapchain) -> (u32, u32) {
        (swapchain.width, swapchain.height)
    }

    fn swapchain_present_mode(swapchain: &GlSwapchain) -> WindowPresentMode {
        swapchain.present_mode
    }

    fn supported_present_modes(&self, _swapchain: &GlSwapchain) -> Vec<WindowPresentMode> {
        match self.0.swap_interval {
            Some(_) => vec![WindowPresentMode::Fifo, WindowPresentMode::Immediate],
            None => vec![WindowPresentMode::Fifo],
        }
    }

    fn swapchain_frame_latency(_swapchain: &GlSwapchain) -> u32 {
        1
    }

    fn configure_swapchain(&self, swapchain: &mut GlSwapchain, width: u32, height: u32, mode: WindowPresentMode) {
        // Le framebuffer par défaut suit la taille de la fenêtre.
        swapchain.width = width.max(1);
        swapchain.height = height.max(1);
        swapchain.present_mode = mode;
        swapchain.configured = true;
    }

    fn acquire(&self, swapchain: &mut GlSwapchain) -> Acquire<GlFrame> {
        if !swapchain.configured {
            return Acquire::Outdated;
        }
        let interval = if swapchain.present_mode == WindowPresentMode::Fifo { 1 } else { 0 };
        Acquire::Frame(GlFrame { dc: swapchain.dc, interval })
    }

    fn copy_texture_to_frame(encoder: &mut GlEncoder, source: &GlTexture, frame: &GlFrame) {
        encoder.frame_dc = Some(frame.dc);
        encoder.commands.push(Command::Present { source: source.clone(), width: source.0.width, height: source.0.height });
    }

    fn submit(&self, encoder: GlEncoder) {
        let section = self.0.section(encoder.frame_dc.unwrap_or(self.0.window_dc));
        self.0.execute(&section, encoder.commands);
    }

    fn present(&self, frame: GlFrame) {
        let section = self.0.section(frame.dc);
        // SAFETY: contexte courant sur le DC de la fenêtre.
        unsafe {
            if let Some(swap_interval) = self.0.swap_interval {
                swap_interval(frame.interval);
            }
            if let Err(error) = SwapBuffers(frame.dc) {
                log::error!("présentation OpenGL : {error}");
            }
            // ponytail: attente CPU de la trame UI entière ; assure qu'un tampon de surface
            // rendu au moteur 3D n'est plus lu. Fences par tampon si le coût se mesure.
            section.shared.gl.finish();
        }
    }

    fn init_external_textures(&self, textures: [&GlTexture; 3]) {
        let section = self.0.section(self.0.window_dc);
        self.0.run_uploads(&section);
        // Contrat des surfaces 3D : effacées avant le premier échantillonnage.
        for texture in textures {
            let framebuffer = self.0.framebuffer(&texture.0);
            // SAFETY: contexte courant (section).
            unsafe {
                section.shared.gl.disable(glow::SCISSOR_TEST);
                section.shared.gl.clear_named_framebuffer_f32_slice(Some(framebuffer), glow::COLOR, 0, &[0.0; 4]);
            }
        }
        // SAFETY: contexte courant ; le moteur lit ces textures depuis son propre contexte.
        unsafe { section.shared.gl.finish() };
    }

    fn surface_format(_format: SurfaceFormat) -> u32 {
        // L'ordre des canaux ne compte que pour les transferts, que les surfaces n'ont pas.
        glow::SRGB8_ALPHA8
    }

    fn queue_lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.0.section.lock()
    }

    fn native_device(&self) -> Option<NativeDevice> {
        Some(NativeDevice::OpenGl { context: self.0.context.0, pixel_format: self.0.pixel_format })
    }

    fn native_texture(texture: &GlTexture, _view: &GlTextureView) -> Option<NativeTexture> {
        Some(NativeTexture::OpenGl(texture.0.name.0.get()))
    }
}
