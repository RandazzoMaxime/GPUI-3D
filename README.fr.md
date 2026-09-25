# GPUI-3D

**Kit de démarrage pour des applications de bureau qui associent une interface
[GPUI](https://www.gpui.rs/) à un moteur 3D temps réel — sur un seul device GPU, sans aucune
copie, et sans jamais redessiner le chrome de l'interface pour une trame 3D.**

Deux façons de rendre l'interface :

- **wgpu** — l'interface passe par wgpu, qui abstrait l'API graphique ; le moteur 3D rend en
  wgpu ou avec l'API native sous-jacente.
- **Full natif** — le renderer de l'interface *et* le moteur 3D parlent tous deux directement à
  l'API native : Vulkan, Direct3D 12, OpenGL ou Metal. wgpu est absent du binaire.

*English version: [README.md](README.md).*

| Vulkan (full natif) | Direct3D 12 (full natif) | OpenGL (full natif) |
|---|---|---|
| ![Vulkan](docs/screenshots/vulkan-native.png) | ![D3D12](docs/screenshots/d3d12-native.png) | ![OpenGL](docs/screenshots/opengl-native.png) |
| **Metal (full natif)** | **wgpu** | **UI : flou d'arrière-plan, OpenGL natif** |
| ![Metal](docs/screenshots/metal-native.png) | ![wgpu](docs/screenshots/wgpu.png) | ![Flou](docs/screenshots/ui-blur-showcase.png) |

La démo : un cube qui tourne derrière un chrome GPUI blanc, caméra orbitale (glisser = orbite,
molette = zoom), compteur FPS.

## Variantes

Chaque dossier est une application complète, à copier. Elles partagent le shell d'interface de
[`shell/`](shell/src/lib.rs).

| Dossier | Moteur 3D | Renderer de l'UI | Plateformes | Vérifié sur |
|---|---|---|---|---|
| [`GPUI-WGPU`](GPUI-WGPU/src/main.rs) | wgpu (WGSL) | wgpu | backends de wgpu : Vulkan, Metal, D3D12, GL | Windows 11 (D3D12, Vulkan), macOS 26 (Metal) |
| [`GPUI-VULKAN`](GPUI-VULKAN/src/main.rs) | Vulkan (ash, GLSL → SPIR-V) | **Vulkan** | Vulkan 1.3 | Windows 11 |
| [`GPUI-DX12`](GPUI-DX12/src/dx12_cube.rs) | Direct3D 12 (windows-rs, HLSL) | **Direct3D 12** | Windows 10+ | Windows 11 |
| [`GPUI-OPENGL`](GPUI-OPENGL/src/opengl_cube.rs) | OpenGL 4.5 core (WGL, GLSL) | **OpenGL 4.5** | Windows (WGL) | Windows 11 |
| [`GPUI-METAL`](GPUI-METAL/src/metal_cube.rs) | Metal (objc2-metal, MSL) | **Metal** | macOS | macOS 26.5, Apple M1 Max |

Les moteurs natifs ne dépendent jamais de wgpu : GPUI leur remet des poignées brutes
(`VkDevice`/`VkImage`, `ID3D12Device`/`ID3D12Resource`, un `HGLRC` partagé et un nom de
texture, `MTLDevice`/`MTLTexture`) et ils font tout le reste avec l'API elle-même.

## Démarrage rapide

Il faut une chaîne Rust stable (édition 2024) et un pilote GPU pour l'API choisie. macOS
demande les outils en ligne de commande de Xcode.

```bash
git clone <ce dépôt> && cd GPUI-3D
cargo run -p gpui-vulkan     # Windows (Linux : non testé)
cargo run -p gpui-dx12       # Windows
cargo run -p gpui-opengl     # Windows
cargo run -p gpui-metal      # macOS
cargo run -p gpui-wgpu       # partout où wgpu tourne
```

Construire une variante à la fois (`-p`) : un build de tout le workspace unifie les features
Cargo et compile aussi wgpu dans les variantes full natif, où il reste inutilisé.

Pour démarrer une app : copier le dossier de la variante voulue, remplacer `CUBE_*` et le
shader, garder la forme du `Renderer` — `new(surface)` crée les ressources GPU,
`render(surface, scene)` encode une trame et la publie. L'interface se compose dans
`shell/src/lib.rs` (`Shell::render`).

## Fonctionnement

1. **Un device.** GPUI crée le device GPU de l'interface ; le moteur 3D rend sur ce même device
   (et cette même queue) par les poignées natives.
2. **Une surface triple tampon** (`window.create_surface(w, h, SurfaceFormat)`, affichée par
   `gpu_surface(handle)`) : le moteur rend dans le tampon arrière pendant que le compositeur
   échantillonne le tampon affiché. Ni relecture ni copie.
3. **Un fil de rendu dédié**, cadencé sur le rafraîchissement de l'écran par échéances
   absolues, `submit_guard()` tenu pendant l'encodage et la soumission.
4. **Publication sans redessin du chrome** : `swap_buffers()` puis `request_window_redraw()` —
   la fenêtre recompose sa scène en cache. Jamais `cx.notify()` par trame : il force un dessin
   complet de l'interface.
5. **Les gestes ne notifient pas** : la caméra est un `Arc<Mutex<_>>` écrit par les handlers
   souris et relu par le fil de rendu.

Sous le capot, le renderer de l'interface est un renderer générique unique, posé sur une petite
couche GPU interne (`vendor/wgpui/src/platform/cross/hal.rs`) qui a cinq implémentations :
wgpu, Vulkan, Direct3D 12, OpenGL, Metal. Les shaders WGSL de l'interface sont traduits au build
par [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) (SPIR-V, HLSL → DXBC, GLSL 4.50,
MSL). Voir [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) pour la couche, les contrats de
synchronisation par API et la vérification des backends.

## Variables d'environnement

| Variable | Effet |
|---|---|
| `GPUI3D_TIME=1.3` | Fige la scène, pour comparer les variantes pixel à pixel. |
| `WGPU_BACKEND=vulkan\|dx12\|metal\|gl` | Force un backend wgpu dans `GPUI-WGPU`. |
| `GPUI_RENDERER=wgpu\|vulkan\|dx12\|opengl\|metal` | Choisit le backend de l'interface dans les exemples du fork (`vendor/wgpui/examples`). |
| `GPUI_D3D12_DEBUG=1` | Active la couche de debug D3D12 (« Outils graphiques » de Windows) et relaie ses messages sur stderr. |
| `GPUI_GL_DEBUG=1` | Crée un contexte OpenGL de debug et relaie les erreurs du pilote sur stderr. |
| `MTL_DEBUG_LAYER=1` | Active la validation de l'API Metal (macOS). |
| `GPUI_FRAME_DUMP=trame.bmp` | Écrit en BMP chaque trame présentée, relue par le GPU (wgpu et Metal) : une capture sans accès à l'écran, en SSH par exemple. |

## Limites connues

- **OpenGL** passe par WGL : Windows seulement (pas encore de GLX/EGL).
- **Vulkan full natif** exige Vulkan 1.3 (dynamic rendering). MoltenVK sur macOS n'est pas testé.
- **Linux** n'a été testé avec aucune variante.
- **Metal** : le redimensionnement interactif de la fenêtre n'a pas été testé (Mac piloté en SSH).
- **Flou d'arrière-plan** : les fenêtres D3D12, OpenGL et Metal sont opaques, là où Vulkan prend
  l'alpha prémultiplié ; les flous diffèrent donc légèrement entre ces API, exactement comme avec
  wgpu.
- La suite de tests du fork a deux comportements intermittents préexistants (un test du profileur
  dépendant de l'ordre, des tests GPU qui se figent parfois quand beaucoup de devices headless
  sont créés en parallèle).

## Organisation du dépôt

```
shell/              chrome GPUI, caméra et fil de rendu, communs à toutes les variantes
GPUI-WGPU/          moteur wgpu (WGSL)
GPUI-VULKAN/        moteur Vulkan natif (GLSL compilé en SPIR-V par build.rs)
GPUI-DX12/          moteur Direct3D 12 natif (HLSL compilé au démarrage)
GPUI-OPENGL/        moteur OpenGL natif (WGL, bindings générés par build.rs)
GPUI-METAL/         moteur Metal natif (MSL)
vendor/wgpui/       fork GPUI (WGPUI / gpui-ce) avec les patchs « GPUI-3D »
vendor/priority-threadpool/   pool de threads corrigé, utilisé par les timers de GPUI
tools/bench/        scripts de comparaison pixel à pixel ayant servi à vérifier les backends
docs/               notes d'architecture et captures d'écran
```

## Licence

Le code propre à GPUI-3D est sous double licence [MIT](LICENSE-MIT) ou
[Apache-2.0](LICENSE-APACHE), au choix. Le code tiers embarqué garde sa licence — voir
[NOTICE](NOTICE) : `vendor/wgpui` est sous Apache-2.0 (dérivé de GPUI, de Zed),
`vendor/priority-threadpool` sous MIT, et les polices embarquées sous SIL Open Font License.
