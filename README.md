# GPUI-3D

Kit de démarrage : une app **GPUI** (chrome blanc) avec un **moteur 3D** derrière,
un seul device GPU partagé entre l'UI
et la 3D, zéro copie, et le chrome n'est **jamais** redessiné pour une trame 3D.

Démo : un cube qui tourne, caméra orbitale (glisser = orbite, molette = zoom),
compteur FPS.

| Dossier | Moteur 3D | Renderer de l'UI | Backends |
|---|---|---|---|
| [`GPUI-WGPU`](GPUI-WGPU/src/main.rs) | wgpu | wgpu | tous ceux de wgpu : Metal, Vulkan, DX12, GL (`WGPU_BACKEND=…`) |
| [`GPUI-METAL`](GPUI-METAL/src/metal_cube.rs) | Metal natif (objc2-metal, MSL) | wgpu (Metal) | Metal — macOS |
| [`GPUI-VULKAN`](GPUI-VULKAN/src/main.rs) | Vulkan natif (ash, GLSL → SPIR-V) | **Vulkan natif** — full natif, sans wgpu | Vulkan 1.3 — Windows, Linux ; macOS si MoltenVK expose Vulkan 1.3 (non vérifié) |
| [`GPUI-DX12`](GPUI-DX12/src/dx12_cube.rs) | D3D12 natif (windows-rs, HLSL → DXBC) | **D3D12 natif** — full natif, sans wgpu | D3D12 — Windows |
| [`GPUI-OPENGL`](GPUI-OPENGL/src/opengl_cube.rs) | OpenGL 4.5 natif (WGL, GLSL) | **OpenGL natif** — full natif, sans wgpu | OpenGL 4.5 core — Windows |

Deux familles de variantes :

- **wgpu** : l'UI rend par wgpu, qui gère l'abstraction des API ; le moteur 3D rend
  en wgpu ou dans l'API native du device choisi.
- **full natif** : l'UI de GPUI **et** le moteur 3D rendent tous deux dans l'API
  native, wgpu absent du binaire (`cargo tree -p gpui-vulkan` n'en contient pas).
  Disponible pour Vulkan, D3D12 et OpenGL ; Metal suit par la même couche.

Les moteurs natifs **ne dépendent pas de wgpu** : ils reçoivent de GPUI des
poignées natives brutes (`MTLDevice`/`MTLCommandQueue`/`MTLTexture`,
`VkInstance`/`VkDevice`/`VkQueue`/`VkImage`, `ID3D12Device`/`ID3D12CommandQueue`/`ID3D12Resource`)
et font tout le reste avec l'API — que l'UI tourne sur wgpu ou en natif.

## Lancer

```bash
cargo run -p gpui-wgpu
cargo run -p gpui-metal
cargo run -p gpui-vulkan
cargo run -p gpui-dx12
cargo run -p gpui-opengl
```

Lancer une variante à la fois (`-p`) : un build de tout le workspace unifie les
features et compile aussi wgpu dans les variantes full natif (il y reste inutilisé).

`GPUI_D3D12_DEBUG=1` active la couche de debug D3D12 (« Outils graphiques » de Windows)
et relaie ses avertissements et erreurs sur stderr ; `GPUI_GL_DEBUG=1` fait de même avec
un contexte OpenGL de debug.

`GPUI-OPENGL` exige un pilote OpenGL 4.5 core (WGL) : sinon, échec explicite.

`GPUI3D_TIME=1.3` fige la scène (même image pour tous les moteurs, pour les comparer
pixel à pixel). Sur un poste iGPU + dGPU, le GPU discret est choisi à backend égal ;
l'adaptateur retenu s'affiche au lancement.

`WGPU_BACKEND=vulkan cargo run -p gpui-wgpu` force un backend wgpu (par défaut :
Metal sur macOS). Le bandeau affiche le moteur et l'API réelle du device.

Vulkan sur macOS exige MoltenVK et le chargeur, que `dlopen` ne trouve pas seul
dans Homebrew :

```bash
brew install molten-vk vulkan-loader
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib cargo run -p gpui-vulkan
```

Le premier lancement après l'installation peut prendre ~30 s (vérification de
signature de `libMoltenVK.dylib` par macOS), les suivants sont immédiats.

## Architecture

```
shell/            chrome GPUI + caméra + fil de rendu, commun aux moteurs ; feature `wgpu` ou `vulkan` = renderer de l'UI
GPUI-WGPU/        moteur wgpu (WGSL)
GPUI-METAL/       moteur Metal natif (MSL)
GPUI-VULKAN/      moteur Vulkan natif (GLSL compilé en SPIR-V par build.rs)
GPUI-DX12/        moteur D3D12 natif (HLSL compilé par d3dcompiler_47 au démarrage)
GPUI-OPENGL/      moteur OpenGL natif (WGL, bindings générés par build.rs) + publication D3D12
vendor/wgpui      fork GPUI (gpui-ce / WGPUI) + patchs « GPUI-3D »
vendor/priority-threadpool   copie corrigée (timers GPUI)
```

La recette :

1. **Un device.** WGPUI crée le device de l'UI ; `run::<Moteur>(label, ui)` choisit
   son renderer (`Ui::Wgpu(backends)` ou `Ui::Vulkan`). Le moteur rend sur ce même device.
2. **Une surface triple-buffer** (`window.create_surface`, affichée par
   `gpu_surface(handle)`) : le moteur rend
   dans le tampon arrière, le compositeur échantillonne le tampon affiché. Pas de
   readback, pas de copie.
3. **Un fil de rendu dédié** (`shell::spawn_render_thread`), cadencé sur le
   rafraîchissement de l'écran par échéances absolues, `submit_guard` tenu pendant
   l'encodage et la soumission.
4. **Publication sans redessin du chrome** : `present_synced_silent` (wgpu) ou
   `swap_buffers` (natif), puis `request_window_redraw` ⇒ la fenêtre recompose la
   scène en cache. Jamais `present_synced` ni `cx.notify()` par trame : les deux
   forcent un dessin complet de l'UI.
5. **Les gestes ne notifient pas** : la caméra est un `Arc<Mutex<_>>` écrit par les
   handlers souris et relu par le fil de rendu.

### Contrat natif (patchs `GPUI-3D` du fork)

`SurfaceHandle`, commun à tous les renderers de l'UI, expose `native_device()`,
`native_back_buffer()` et `native_queue_lock()`
([`vendor/wgpui/src/elements/gpu_surface.rs`](vendor/wgpui/src/elements/gpu_surface.rs)).
Sur une UI wgpu, `handle.as_wgpu()` donne en plus l'accès wgpu (`WgpuSurfaceHandle`).
Le renderer de l'UI passe par une couche interne (`platform/cross/hal.rs`) dont wgpu
et Vulkan natif sont deux implémentations, choisies par `RendererBackend`
(ou `GPUI_RENDERER=wgpu|vulkan|dx12|opengl` pour les exemples du fork). Les shaders WGSL
de l'UI sont traduits au build par naga (SPIR-V pour Vulkan, HLSL compilé en DXBC par FXC
pour D3D12, GLSL 4.50 pour OpenGL). Les fenêtres D3D12 et OpenGL sont opaques, là où Vulkan
prend l'alpha prémultiplié : les flous d'arrière-plan diffèrent donc légèrement entre ces
API, exactement comme avec wgpu sur chacune d'elles.

- **Metal** : même `MTLCommandQueue` que le compositeur ⇒ ordre garanti, textures
  « tracked » ⇒ aucun fence. On commit puis `swap_buffers()`.
- **Vulkan** : `VkQueue` exige une synchronisation externe ⇒ tout `vkQueueSubmit`
  se fait sous `native_queue_lock()` (le compositeur le prend aussi). Le tampon
  arrive et repart en `SHADER_READ_ONLY_OPTIMAL` ; les dépendances de subpass
  d'entrée/sortie portent la synchronisation avec le compositeur.
- **D3D12** : même `ID3D12CommandQueue` que le compositeur ⇒ ordre garanti ; une queue
  D3D12 est thread-safe, pas de verrou. Le tampon arrive et repart en
  `PIXEL_SHADER_RESOURCE | NON_PIXEL_SHADER_RESOURCE` (l'état `RESOURCE` de wgpu).
- **OpenGL** : l'UI tourne en OpenGL natif. Le moteur crée son contexte en partage
  d'objets avec celui de l'UI (`NativeDevice::OpenGl`, sous `native_queue_lock()`), rend
  directement dans la texture de surface (ligne 0 en haut :
  `glClipControl(UPPER_LEFT, ZERO_TO_ONE)`) et termine par `glFinish` avant
  `swap_buffers()` ; deux contextes n'ordonnent pas leurs commandes entre eux.
- Les tampons sont initialisés à leur création (sinon wgpu les jugerait vierges et
  les effacerait avant de les échantillonner).
- `adapter_selector` est honoré aussi sur macOS (choisir Metal ou Vulkan/MoltenVK) ;
  sans sélecteur, Metal est préféré à MoltenVK.
- `PRIMITIVE_INDEX` n'est plus exigé : aucun shader WGPUI ne l'utilise et MoltenVK
  ne l'expose pas.

Chercher `GPUI-3D` dans `vendor/wgpui` pour rebaser ces patchs.

## Démarrer une app

Copier le dossier du moteur voulu, remplacer `CUBE_*` et le shader, garder le
`Renderer` : `new(surface)` crée les ressources, `render(surface, scene)` encode une
trame et la publie. L'UI se compose dans `shell/src/lib.rs` (`Shell::render`).
