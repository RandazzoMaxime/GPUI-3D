//! Full natif : l'UI de GPUI et ce moteur 3D rendent tous deux en OpenGL 4.5 core (WGL),
//! sans wgpu dans le binaire. GPUI fournit son contexte ; le moteur crée le sien en
//! partage d'objets avec lui, sur son propre fil, et rend directement dans la texture de
//! surface (zéro copie). Deux contextes n'ordonnent pas leurs commandes entre eux : le
//! moteur termine chaque trame par `glFinish` avant de la publier, le compositeur chaque
//! présentation par `glFinish` avant de rendre un tampon au moteur.

use std::collections::HashMap;
use std::ffi::{CString, c_void};

use gpui3d_shell::{CLEAR_COLOR, CUBE_INDICES, CUBE_VERTICES, NativeBackBuffer, NativeDevice, NativeTexture, Renderer, Scene, Surface};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::Graphics::Gdi::{GetDC, HDC, ReleaseDC};
use windows::Win32::Graphics::OpenGL::*;
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryA};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{PCSTR, s, w};

#[allow(clippy::all, non_upper_case_globals, unsafe_op_in_unsafe_fn)]
mod gl {
    include!(concat!(env!("OUT_DIR"), "/gl.rs"));
}

pub struct OpenGlCube {
    wgl: Wgl,
    gl: gl::Gl,
    program: u32,
    vao: u32,
    target: Option<Target>,
}

/// Fenêtre cachée + contexte WGL partageant les objets de l'UI, courant sur le fil de rendu.
struct Wgl {
    hwnd: HWND,
    hdc: HDC,
    context: HGLRC,
}

/// Profondeur à la taille de la surface et un FBO par tampon de surface rencontré.
struct Target {
    size: (u32, u32),
    depth: u32,
    framebuffers: HashMap<u32, u32>,
}

impl Renderer for OpenGlCube {
    fn new(surface: &Surface) -> Self {
        let Some(NativeDevice::OpenGl { context, pixel_format }) = surface.native_device() else {
            panic!("GPUI ne tourne pas sur OpenGL");
        };
        let wgl = {
            // Le contexte de l'UI ne doit être courant nulle part pendant le partage.
            let _ui = surface.native_queue_lock();
            Wgl::new(HGLRC(context), pixel_format)
        };
        let gl = load_gl();
        unsafe {
            gl.Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl.DebugMessageCallback(Some(debug_message), std::ptr::null());
            // Convention D3D/wgpu : Y clip vers le haut, ligne 0 en haut, profondeur 0..1.
            gl.ClipControl(gl::UPPER_LEFT, gl::ZERO_TO_ONE);
        }
        let program = create_program(&gl);
        let vao = create_mesh(&gl);
        check(&gl, "initialisation");
        Self { wgl, gl, program, vao, target: None }
    }

    fn render(&mut self, surface: &Surface, scene: &Scene) -> bool {
        let Some(back) = surface.native_back_buffer() else { return false };
        let NativeTexture::OpenGl(texture) = back.texture else { return false };
        let (w, h) = back.size;
        let framebuffer = self.framebuffer(&back, texture);
        let gl = &self.gl;
        unsafe {
            gl.BindFramebuffer(gl::FRAMEBUFFER, framebuffer);
            gl.Viewport(0, 0, w as i32, h as i32);
            gl.Enable(gl::FRAMEBUFFER_SRGB);
            gl.Enable(gl::DEPTH_TEST);
            gl.DepthFunc(gl::LESS);
            gl.Enable(gl::CULL_FACE);
            gl.CullFace(gl::BACK);
            gl.FrontFace(gl::CCW);
            let clear = CLEAR_COLOR.map(|c| c as f32);
            gl.ClearNamedFramebufferfv(framebuffer, gl::COLOR, 0, clear.as_ptr());
            gl.ClearNamedFramebufferfv(framebuffer, gl::DEPTH, 0, &1.0);
            gl.UseProgram(self.program);
            let mvp = scene.mvp(w, h);
            gl.UniformMatrix4fv(0, 1, gl::FALSE, mvp.as_ptr().cast());
            gl.BindVertexArray(self.vao);
            gl.DrawElements(gl::TRIANGLES, CUBE_INDICES.len() as i32, gl::UNSIGNED_SHORT, std::ptr::null());
            // Contrat du fork : la trame est finie sur GPU avant d'être publiée.
            gl.Finish();
        }
        drop(back);
        surface.swap_buffers();
        true
    }
}

impl OpenGlCube {
    /// FBO du tampon `texture` : les trois tampons de la surface tournent, et une nouvelle
    /// taille veut de nouvelles textures (dont les noms peuvent être réutilisés).
    fn framebuffer(&mut self, back: &NativeBackBuffer, texture: u32) -> u32 {
        let gl = &self.gl;
        if self.target.as_ref().map(|target| target.size) != Some(back.size) {
            if let Some(old) = self.target.take() {
                old.release(gl);
            }
            let mut depth = 0;
            unsafe {
                gl.CreateRenderbuffers(1, &mut depth);
                gl.NamedRenderbufferStorage(depth, gl::DEPTH_COMPONENT32F, back.size.0 as i32, back.size.1 as i32);
            }
            self.target = Some(Target { size: back.size, depth, framebuffers: HashMap::new() });
        }
        let target = self.target.as_mut().expect("cible");
        *target.framebuffers.entry(texture).or_insert_with(|| unsafe {
            let mut fbo = 0;
            gl.CreateFramebuffers(1, &mut fbo);
            gl.NamedFramebufferTexture(fbo, gl::COLOR_ATTACHMENT0, texture, 0);
            gl.NamedFramebufferRenderbuffer(fbo, gl::DEPTH_ATTACHMENT, gl::RENDERBUFFER, target.depth);
            assert_eq!(gl.CheckNamedFramebufferStatus(fbo, gl::FRAMEBUFFER), gl::FRAMEBUFFER_COMPLETE, "FBO incomplet");
            fbo
        })
    }
}

impl Target {
    fn release(self, gl: &gl::Gl) {
        unsafe {
            for fbo in self.framebuffers.into_values() {
                gl.DeleteFramebuffers(1, &fbo);
            }
            gl.DeleteRenderbuffers(1, &self.depth);
        }
    }
}

impl Drop for OpenGlCube {
    fn drop(&mut self) {
        // Les objets GL propres au moteur meurent avec son contexte.
        unsafe {
            let _ = wglMakeCurrent(HDC::default(), HGLRC::default());
            let _ = wglDeleteContext(self.wgl.context);
            ReleaseDC(Some(self.wgl.hwnd), self.wgl.hdc);
            let _ = DestroyWindow(self.wgl.hwnd);
        }
    }
}

impl Wgl {
    /// Contexte 4.5 core partageant les objets de `share`, sur une fenêtre cachée au format
    /// de pixel de l'UI (le partage l'exige compatible).
    fn new(share: HGLRC, pixel_format: i32) -> Self {
        unsafe {
            let instance = GetModuleHandleW(None).expect("module");
            let class = WNDCLASSW {
                style: CS_OWNDC,
                lpfnWndProc: Some(window_proc),
                hInstance: instance.into(),
                lpszClassName: w!("gpui3d-gl"),
                ..Default::default()
            };
            RegisterClassW(&class);
            let hwnd = CreateWindowExW(WINDOW_EX_STYLE(0), w!("gpui3d-gl"), w!("gpui3d-gl"), WS_POPUP, 0, 0, 1, 1, None, None, Some(instance.into()), None)
                .expect("fenêtre cachée WGL");
            let hdc = GetDC(Some(hwnd));
            let mut pfd = PIXELFORMATDESCRIPTOR::default();
            DescribePixelFormat(hdc, pixel_format, size_of::<PIXELFORMATDESCRIPTOR>() as u32, Some(&mut pfd));
            SetPixelFormat(hdc, pixel_format, &pfd).expect("format de pixel de l'UI");
            // Contexte hérité le temps d'obtenir wglCreateContextAttribsARB, puis 4.5 core
            // partagé (debug en build debug : le pilote explique ses refus).
            let legacy = wglCreateContext(hdc).expect("contexte OpenGL");
            wglMakeCurrent(hdc, legacy).expect("contexte OpenGL courant");
            type CreateContextAttribs = unsafe extern "system" fn(HDC, HGLRC, *const i32) -> HGLRC;
            let create: CreateContextAttribs =
                std::mem::transmute(wglGetProcAddress(s!("wglCreateContextAttribsARB")).expect("WGL_ARB_create_context"));
            const DEBUG_BIT: i32 = if cfg!(debug_assertions) { 0x0001 } else { 0 };
            // MAJOR, MINOR, PROFILE_MASK = CORE, FLAGS.
            let attribs = [0x2091, 4, 0x2092, 5, 0x9126, 0x0001, 0x2094, DEBUG_BIT, 0];
            let context = create(hdc, share, attribs.as_ptr());
            assert!(!context.is_invalid(), "contexte OpenGL 4.5 core partagé refusé");
            wglMakeCurrent(hdc, context).expect("contexte OpenGL courant");
            let _ = wglDeleteContext(legacy);
            Self { hwnd, hdc, context }
        }
    }
}

extern "system" fn debug_message(
    _source: u32,
    kind: u32,
    _id: u32,
    _severity: u32,
    length: i32,
    message: *const i8,
    _user: *mut c_void,
) {
    if kind == gl::DEBUG_TYPE_ERROR {
        let text = unsafe { std::slice::from_raw_parts(message.cast::<u8>(), length as usize) };
        eprintln!("[opengl] {}", String::from_utf8_lossy(text));
    }
}

unsafe extern "system" fn window_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

/// Fonctions ≥ 1.2 par `wglGetProcAddress`, fonctions 1.1 dans `opengl32.dll`.
fn load_gl() -> gl::Gl {
    let opengl32 = unsafe { LoadLibraryA(s!("opengl32.dll")) }.expect("opengl32.dll");
    gl::Gl::load_with(|name| {
        let name = CString::new(name).unwrap();
        let name = PCSTR(name.as_ptr().cast());
        match unsafe { wglGetProcAddress(name) }.map(|f| f as usize) {
            // 1, 2, 3 et -1 sont des échecs documentés de wglGetProcAddress.
            Some(address) if address > 3 && address != usize::MAX => address as *const c_void,
            _ => unsafe { GetProcAddress(opengl32, name) }.map_or(std::ptr::null(), |f| f as *const c_void),
        }
    })
}

fn create_program(gl: &gl::Gl) -> u32 {
    unsafe {
        let shader = |kind, source: &str| {
            let shader = gl.CreateShader(kind);
            let source = CString::new(source).unwrap();
            gl.ShaderSource(shader, 1, &source.as_ptr(), std::ptr::null());
            gl.CompileShader(shader);
            let mut ok = 0;
            gl.GetShaderiv(shader, gl::COMPILE_STATUS, &mut ok);
            assert!(ok != 0, "GLSL : {}", info_log(|len, buf| gl.GetShaderInfoLog(shader, len, std::ptr::null_mut(), buf)));
            shader
        };
        let (vs, fs) = (shader(gl::VERTEX_SHADER, include_str!("cube.vert")), shader(gl::FRAGMENT_SHADER, include_str!("cube.frag")));
        let program = gl.CreateProgram();
        gl.AttachShader(program, vs);
        gl.AttachShader(program, fs);
        gl.LinkProgram(program);
        let mut ok = 0;
        gl.GetProgramiv(program, gl::LINK_STATUS, &mut ok);
        assert!(ok != 0, "édition de liens GLSL : {}", info_log(|len, buf| gl.GetProgramInfoLog(program, len, std::ptr::null_mut(), buf)));
        gl.DeleteShader(vs);
        gl.DeleteShader(fs);
        program
    }
}

fn info_log(read: impl FnOnce(i32, *mut i8)) -> String {
    let mut buf = vec![0u8; 4096];
    read(buf.len() as i32, buf.as_mut_ptr().cast());
    String::from_utf8_lossy(&buf).trim_end_matches('\0').to_owned()
}

fn create_mesh(gl: &gl::Gl) -> u32 {
    unsafe {
        let buffer = |bytes: &[u8]| {
            let mut buffer = 0;
            gl.CreateBuffers(1, &mut buffer);
            gl.NamedBufferStorage(buffer, bytes.len() as isize, bytes.as_ptr().cast(), 0);
            buffer
        };
        let vb = buffer(bytemuck::cast_slice(&CUBE_VERTICES));
        let ib = buffer(bytemuck::cast_slice(&CUBE_INDICES));
        let mut vao = 0;
        gl.CreateVertexArrays(1, &mut vao);
        gl.VertexArrayVertexBuffer(vao, 0, vb, 0, 24);
        gl.VertexArrayElementBuffer(vao, ib);
        for (location, offset) in [(0, 0), (1, 12)] {
            gl.EnableVertexArrayAttrib(vao, location);
            gl.VertexArrayAttribFormat(vao, location, 3, gl::FLOAT, gl::FALSE, offset);
            gl.VertexArrayAttribBinding(vao, location, 0);
        }
        vao
    }
}

fn check(gl: &gl::Gl, stage: &str) {
    let error = unsafe { gl.GetError() };
    assert_eq!(error, gl::NO_ERROR, "erreur OpenGL 0x{error:x} ({stage})");
}
