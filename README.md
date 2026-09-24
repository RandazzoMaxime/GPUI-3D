# GPUI-3D

Kit de démarrage : une app **GPUI** (chrome blanc) avec un **moteur 3D** derrière,
un seul device GPU partagé entre l'UI
et la 3D, zéro copie, et le chrome n'est **jamais** redessiné pour une trame 3D.

Démo : un cube qui tourne, caméra orbitale (glisser = orbite, molette = zoom),
compteur FPS.

| Dossier | Moteur 3D | Backends |
|---|---|---|
| [`GPUI-WGPU`](GPUI-WGPU/src/main.rs) | wgpu | tous ceux de wgpu : Metal, Vulkan, DX12, GL (`WGPU_BACKEND=…`) |
| [`GPUI-METAL`](GPUI-METAL/src/metal_cube.rs) | Metal natif (objc2-metal, MSL) | Metal — macOS |
| [`GPUI-VULKAN`](GPUI-VULKAN/src/main.rs) | Vulkan natif (ash, GLSL → SPIR-V) | Vulkan — Windows, Linux, macOS via MoltenVK |

Les moteurs Metal et Vulkan **ne dépendent pas de wgpu** : ils reçoivent de GPUI des
poignées natives brutes (`MTLDevice`/`MTLCommandQueue`/`MTLTexture`,
`VkInstance`/`VkDevice`/`VkQueue`/`VkImage`) et font tout le reste avec l'API.

## Lancer

```bash
cargo run -p gpui-wgpu
cargo run -p gpui-metal
cargo run -p gpui-vulkan
```

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
shell/            chrome GPUI + caméra + fil de rendu, commun aux trois moteurs
GPUI-WGPU/        moteur wgpu (WGSL)
GPUI-METAL/       moteur Metal natif (MSL)
GPUI-VULKAN/      moteur Vulkan natif (GLSL compilé en SPIR-V par build.rs)
vendor/wgpui      fork GPUI (gpui-ce / WGPUI) + patchs « GPUI-3D »
vendor/priority-threadpool   copie corrigée (timers GPUI)
```

La recette :

1. **Un device.** WGPUI crée le device de l'UI ; `run::<Moteur>(label, backends)`
   choisit son adaptateur. Le moteur rend sur ce même device.
2. **Une surface triple-buffer** (`window.create_wgpu_surface`) : le moteur rend
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

`WgpuSurfaceHandle` gagne `native_device()`, `native_back_buffer()` et
`native_queue_lock()` ([`vendor/wgpui/src/elements/wgpu_surface.rs`](vendor/wgpui/src/elements/wgpu_surface.rs)) :

- **Metal** : même `MTLCommandQueue` que le compositeur ⇒ ordre garanti, textures
  « tracked » ⇒ aucun fence. On commit puis `swap_buffers()`.
- **Vulkan** : `VkQueue` exige une synchronisation externe ⇒ tout `vkQueueSubmit`
  se fait sous `native_queue_lock()` (le compositeur le prend aussi). Le tampon
  arrive et repart en `SHADER_READ_ONLY_OPTIMAL` ; les dépendances de subpass
  d'entrée/sortie portent la synchronisation avec le compositeur.
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

## Performance : défilement continu (`bench/`)

`scroll-bench` : client mail (10 000 messages, barre latérale, panneau de lecture) qui
défile en continu ; mesure CPU %, instructions et cycles par trame (indépendants de la
fréquence CPU).

```bash
cargo build -p scroll-bench --profile profiling
BENCH_MODE=cached BENCH_ROWS=1 ./target/profiling/scroll-bench
```

`BENCH_MODE=root|cached` (tout dans une vue / panneaux `.cached()`), `BENCH_ROWS=1`
(chaque ligne est une vue `.cached()`), `BENCH_SPEED` (px/trame), `WGPUI_RENDER_STATS=1`
(détail par phase et raisons des rejets de cache).

Patchs `GPUI-3D` du fork (chacun désactivable pour comparer) :

| Variable | Effet |
|---|---|
| `GPUI_VIEW_TRANSLATE` | une vue `.cached()` déplacée ou recoupée est rejouée translatée au lieu d'être reconstruite (`0` : amont, `rebuild` : référence de parité) |
| `GPUI_BLOCK_REPLAY` | une plage rejouée réserve un seul créneau dans le `BoundsTree` au lieu d'une insertion par primitive |
| `GPUI_SORT_CACHED` | tri des primitives par paires (clé, indice) au lieu de déplacer chaque struct O(log n) fois |

Mesuré (M1 Max, 60 Hz, médianes, lignes en vues) : 16,2 → 7,8 M instructions par trame,
CPU 27,7 % → 17,9 %, rendu identique au pixel près. Tests :
`cd vendor/wgpui && cargo test --lib --features test-support translated_reuse_tests`.
