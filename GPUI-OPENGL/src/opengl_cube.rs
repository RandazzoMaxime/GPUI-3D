//! Moteur 3D en OpenGL natif (WGL), sans wgpu. GPUI ne sait pas tourner sur un device
//! GL ; il fournit donc son `ID3D12Device`, son `ID3D12CommandQueue` et la
//! `ID3D12Resource` du tampon arrière, et l'image passe par l'interop GL ↔ D3D12
//! (`GL_EXT_memory_object_win32`, `GL_EXT_semaphore_win32`) :
//!
//! GL rend dans son FBO, puis relit en BGRA dans un tampon D3D12 partagé ; la queue
//! D3D12 le copie dans le tampon arrière. Une fence D3D12 partagée, importée en
//! sémaphore GL, ordonne les deux côtés sur GPU : GL attend la copie précédente, la
//! queue attend GL. Aucune attente CPU sur le GPU.

use std::ffi::{CString, c_void};
use std::mem::ManuallyDrop;

use gpui3d_shell::{CLEAR_COLOR, CUBE_INDICES, CUBE_VERTICES, NativeBackBuffer, NativeDevice, NativeTexture, Renderer, Scene, Surface};
use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE, HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::Gdi::{GetDC, HDC, ReleaseDC};
use windows::Win32::Graphics::OpenGL::*;
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryA};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCSTR, PCWSTR, s, w};

#[allow(clippy::all, non_upper_case_globals, unsafe_op_in_unsafe_fn)]
mod gl {
    include!(concat!(env!("OUT_DIR"), "/gl.rs"));
}

const FRAMES_IN_FLIGHT: usize = 2;
/// Contrat du fork : le tampon arrive et repart dans l'état `RESOURCE` de wgpu.
const SAMPLED: D3D12_RESOURCE_STATES =
    D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0);

pub struct OpenGlCube {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    /// Partagée D3D12 ↔ GL : valeurs impaires = rendu GL fini, paires = copie finie.
    fence: ID3D12Fence,
    fence_value: u64,
    frames: Vec<Frame>,
    frame: usize,
    list: ID3D12GraphicsCommandList,
    wgl: Wgl,
    gl: gl::Gl,
    semaphore: u32,
    program: u32,
    vao: u32,
    target: Option<Target>,
}

struct Frame {
    allocator: ID3D12CommandAllocator,
    done: u64,
    /// Retient le tampon arrière tant que le GPU l'écrit.
    back: Option<NativeBackBuffer>,
}

/// Fenêtre cachée + contexte WGL, courant sur le fil de rendu.
struct Wgl {
    hwnd: HWND,
    hdc: HDC,
    context: HGLRC,
}

/// Ressources par taille : FBO GL et tampon D3D12 partagé, importé dans GL.
struct Target {
    size: (u32, u32),
    fbo: u32,
    color: u32,
    depth: u32,
    buffer: ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    memory: u32,
    gl_buffer: u32,
}

impl Renderer for OpenGlCube {
    fn new(surface: &Surface) -> Self {
        let Some(NativeDevice::Dx12 { device, queue }) = surface.native_device() else {
            panic!("GPUI ne tourne pas sur D3D12");
        };
        let device = unsafe { ID3D12Device::from_raw_borrowed(&device) }.expect("ID3D12Device").clone();
        let queue = unsafe { ID3D12CommandQueue::from_raw_borrowed(&queue) }.expect("ID3D12CommandQueue").clone();
        let wgl = Wgl::new();
        let gl = load_gl();
        unsafe {
            // Le pilote explique ses refus sur stderr (erreurs seulement : la relecture
            // synchrone après rendu déclenche un avertissement de performance par trame).
            gl.Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl.DebugMessageCallback(Some(debug_message), std::ptr::null());
            let mut count = 0;
            gl.GetIntegerv(gl::NUM_EXTENSIONS, &mut count);
            let has = |name: &str| {
                (0..count as u32).any(|i| std::ffi::CStr::from_ptr(gl.GetStringi(gl::EXTENSIONS, i).cast()).to_bytes() == name.as_bytes())
            };
            for name in ["GL_EXT_memory_object_win32", "GL_EXT_semaphore_win32"] {
                assert!(has(name), "{name} absent : pilote OpenGL sans interop D3D12");
            }
        }
        unsafe {
            // Même GPU que le device D3D12, sinon aucun partage possible.
            let mut luid = [0u8; 8];
            gl.GetUnsignedBytevEXT(gl::DEVICE_LUID_EXT, luid.as_mut_ptr());
            let want = device.GetAdapterLuid();
            let mut expected = [0u8; 8];
            expected[..4].copy_from_slice(&want.LowPart.to_le_bytes());
            expected[4..].copy_from_slice(&want.HighPart.to_le_bytes());
            assert_eq!(luid, expected, "le contexte OpenGL n'est pas sur l'adaptateur D3D12 de GPUI");

            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_SHARED).expect("fence partagée");
            let handle = shared_handle(&device, &fence);
            let mut semaphore = 0;
            gl.GenSemaphoresEXT(1, &mut semaphore);
            gl.ImportSemaphoreWin32HandleEXT(semaphore, gl::HANDLE_TYPE_D3D12_FENCE_EXT, handle.0);
            let _ = CloseHandle(handle);

            let frames = (0..FRAMES_IN_FLIGHT)
                .map(|_| Frame {
                    allocator: device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("allocateur"),
                    done: 0,
                    back: None,
                })
                .collect::<Vec<_>>();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &frames[0].allocator, None)
                .expect("command list");
            list.Close().expect("close");

            // Convention D3D/wgpu : Y clip vers le haut, ligne 0 en haut, profondeur 0..1.
            gl.ClipControl(gl::UPPER_LEFT, gl::ZERO_TO_ONE);
            let program = create_program(&gl);
            let vao = create_mesh(&gl);
            check(&gl, "initialisation");
            Self {
                device,
                queue,
                fence,
                fence_value: 0,
                frames,
                frame: 0,
                list,
                wgl,
                gl,
                semaphore,
                program,
                vao,
                target: None,
            }
        }
    }

    fn render(&mut self, surface: &Surface, scene: &Scene) -> bool {
        self.wait(self.frames[self.frame].done);
        self.frames[self.frame].back = None;
        let Some(back) = surface.native_back_buffer() else { return false };
        let NativeTexture::Dx12(ptr) = back.texture else { return false };
        let texture = unsafe { ID3D12Resource::from_raw_borrowed(&ptr) }.expect("ID3D12Resource");
        let (w, h) = back.size;
        if self.target.as_ref().map(|t| t.size) != Some((w, h)) {
            self.wait(self.fence_value);
            if let Some(old) = self.target.take() {
                old.release(&self.gl);
            }
            self.target = Some(Target::new(&self.device, &self.gl, texture, (w, h)));
        }
        let target = self.target.as_ref().expect("cible");
        let gl = &self.gl;

        // GL : attend que la copie précédente ait lu le tampon, rend, relit, signale.
        unsafe {
            gl.SemaphoreParameterui64vEXT(self.semaphore, gl::D3D12_FENCE_VALUE_EXT, &self.fence_value);
            gl.WaitSemaphoreEXT(self.semaphore, 1, &target.gl_buffer, 0, std::ptr::null(), std::ptr::null());
            gl.BindFramebuffer(gl::FRAMEBUFFER, target.fbo);
            gl.Viewport(0, 0, w as i32, h as i32);
            gl.Enable(gl::FRAMEBUFFER_SRGB);
            gl.Enable(gl::DEPTH_TEST);
            gl.DepthFunc(gl::LESS);
            gl.Enable(gl::CULL_FACE);
            gl.CullFace(gl::BACK);
            gl.FrontFace(gl::CCW);
            let clear = CLEAR_COLOR.map(|c| c as f32);
            gl.ClearNamedFramebufferfv(target.fbo, gl::COLOR, 0, clear.as_ptr());
            gl.ClearNamedFramebufferfv(target.fbo, gl::DEPTH, 0, &1.0);
            gl.UseProgram(self.program);
            let mvp = scene.mvp(w, h);
            gl.UniformMatrix4fv(0, 1, gl::FALSE, mvp.as_ptr().cast());
            gl.BindVertexArray(self.vao);
            gl.DrawElements(gl::TRIANGLES, CUBE_INDICES.len() as i32, gl::UNSIGNED_SHORT, std::ptr::null());
            // Octets sRGB tels quels, en BGRA, au pas de l'empreinte D3D12.
            gl.Disable(gl::FRAMEBUFFER_SRGB);
            gl.BindBuffer(gl::PIXEL_PACK_BUFFER, target.gl_buffer);
            gl.PixelStorei(gl::PACK_ROW_LENGTH, (target.footprint.Footprint.RowPitch / 4) as i32);
            gl.ReadPixels(0, 0, w as i32, h as i32, gl::BGRA, gl::UNSIGNED_BYTE, std::ptr::null_mut());
            gl.BindBuffer(gl::PIXEL_PACK_BUFFER, 0);
            self.fence_value += 1;
            gl.SemaphoreParameterui64vEXT(self.semaphore, gl::D3D12_FENCE_VALUE_EXT, &self.fence_value);
            gl.SignalSemaphoreEXT(self.semaphore, 1, &target.gl_buffer, 0, std::ptr::null(), std::ptr::null());
            gl.Flush();
        }

        // D3D12 : attend GL, copie dans le tampon arrière.
        let frame = &mut self.frames[self.frame];
        let list = &self.list;
        unsafe {
            self.queue.Wait(&self.fence, self.fence_value).expect("Wait");
            frame.allocator.Reset().expect("reset allocateur");
            list.Reset(&frame.allocator, None).expect("reset liste");
            list.ResourceBarrier(&[transition(texture, SAMPLED, D3D12_RESOURCE_STATE_COPY_DEST)]);
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            // Tampon en COMMON : promu implicitement en COPY_SOURCE, redescend après exécution.
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&target.buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: target.footprint },
            };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.ResourceBarrier(&[transition(texture, D3D12_RESOURCE_STATE_COPY_DEST, SAMPLED)]);
            list.Close().expect("close");
            self.queue.ExecuteCommandLists(&[Some(list.cast().expect("ID3D12CommandList"))]);
            self.fence_value += 1;
            self.queue.Signal(&self.fence, self.fence_value).expect("Signal");
        }
        frame.done = self.fence_value;
        frame.back = Some(back);
        self.frame = (self.frame + 1) % FRAMES_IN_FLIGHT;
        surface.swap_buffers();
        true
    }
}

impl OpenGlCube {
    fn wait(&self, value: u64) {
        if unsafe { self.fence.GetCompletedValue() } < value {
            unsafe { self.fence.SetEventOnCompletion(value, HANDLE::default()) }.expect("attente fence");
        }
    }
}

impl Drop for OpenGlCube {
    fn drop(&mut self) {
        // Les objets GL meurent avec le contexte, les objets D3D12 par comptage de références.
        self.wait(self.fence_value);
        unsafe {
            let _ = wglMakeCurrent(HDC::default(), HGLRC::default());
            let _ = wglDeleteContext(self.wgl.context);
            ReleaseDC(Some(self.wgl.hwnd), self.wgl.hdc);
            let _ = DestroyWindow(self.wgl.hwnd);
        }
    }
}

impl Wgl {
    fn new() -> Self {
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
            let pfd = PIXELFORMATDESCRIPTOR {
                nSize: size_of::<PIXELFORMATDESCRIPTOR>() as u16,
                nVersion: 1,
                dwFlags: PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL,
                iPixelType: PFD_TYPE_RGBA,
                cColorBits: 32,
                ..Default::default()
            };
            SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd).expect("format de pixel");
            // Contexte hérité le temps d'obtenir wglCreateContextAttribsARB, puis 4.5 core
            // (debug en build debug : le pilote explique ses refus).
            let legacy = wglCreateContext(hdc).expect("contexte OpenGL");
            wglMakeCurrent(hdc, legacy).expect("contexte OpenGL courant");
            type CreateContextAttribs = unsafe extern "system" fn(HDC, HGLRC, *const i32) -> HGLRC;
            let create: CreateContextAttribs =
                std::mem::transmute(wglGetProcAddress(s!("wglCreateContextAttribsARB")).expect("WGL_ARB_create_context"));
            const DEBUG_BIT: i32 = if cfg!(debug_assertions) { 0x0001 } else { 0 };
            // MAJOR, MINOR, PROFILE_MASK = CORE, FLAGS.
            let attribs = [0x2091, 4, 0x2092, 5, 0x9126, 0x0001, 0x2094, DEBUG_BIT, 0];
            let context = create(hdc, HGLRC::default(), attribs.as_ptr());
            assert!(!context.is_invalid(), "contexte OpenGL 4.5 core refusé");
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

impl Target {
    fn new(device: &ID3D12Device, gl: &gl::Gl, texture: &ID3D12Resource, size: (u32, u32)) -> Self {
        let (w, h) = (size.0 as i32, size.1 as i32);
        unsafe {
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0u64;
            device.GetCopyableFootprints(&texture.GetDesc(), 0, 1, 0, Some(&mut footprint), None, None, Some(&mut total));
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: total,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let mut buffer: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
                .expect("tampon partagé");
            let buffer = buffer.expect("tampon partagé");

            let handle = shared_handle(device, &buffer);
            let mut memory = 0;
            gl.CreateMemoryObjectsEXT(1, &mut memory);
            gl.MemoryObjectParameterivEXT(memory, gl::DEDICATED_MEMORY_OBJECT_EXT, &(gl::TRUE as i32));
            gl.ImportMemoryWin32HandleEXT(memory, total, gl::HANDLE_TYPE_D3D12_RESOURCE_EXT, handle.0);
            let _ = CloseHandle(handle);
            check(gl, "import du tampon D3D12");
            let mut gl_buffer = 0;
            gl.CreateBuffers(1, &mut gl_buffer);
            gl.NamedBufferStorageMemEXT(gl_buffer, total as isize, memory, 0);
            check(gl, "stockage du tampon importé");

            let (mut fbo, mut color, mut depth) = (0, 0, 0);
            gl.CreateTextures(gl::TEXTURE_2D, 1, &mut color);
            gl.TextureStorage2D(color, 1, gl::SRGB8_ALPHA8, w, h);
            gl.CreateRenderbuffers(1, &mut depth);
            gl.NamedRenderbufferStorage(depth, gl::DEPTH_COMPONENT32F, w, h);
            gl.CreateFramebuffers(1, &mut fbo);
            gl.NamedFramebufferTexture(fbo, gl::COLOR_ATTACHMENT0, color, 0);
            gl.NamedFramebufferRenderbuffer(fbo, gl::DEPTH_ATTACHMENT, gl::RENDERBUFFER, depth);
            assert_eq!(gl.CheckNamedFramebufferStatus(fbo, gl::FRAMEBUFFER), gl::FRAMEBUFFER_COMPLETE, "FBO incomplet");
            check(gl, "cible");
            Self { size, fbo, color, depth, buffer, footprint, memory, gl_buffer }
        }
    }

    /// Appelé GPU au repos.
    fn release(self, gl: &gl::Gl) {
        unsafe {
            gl.DeleteFramebuffers(1, &self.fbo);
            gl.DeleteTextures(1, &self.color);
            gl.DeleteRenderbuffers(1, &self.depth);
            gl.DeleteBuffers(1, &self.gl_buffer);
            gl.DeleteMemoryObjectsEXT(1, &self.memory);
        }
    }
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

fn shared_handle<T: Interface>(device: &ID3D12Device, object: &T) -> HANDLE {
    let child: ID3D12DeviceChild = object.cast().expect("ID3D12DeviceChild");
    unsafe { device.CreateSharedHandle(&child, None, GENERIC_ALL.0, PCWSTR::null()) }.expect("handle partagé")
}

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
