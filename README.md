# GPUI-3D

## A repo to quickly start a GPUI project that needs 3D!

[GPUI](https://www.gpui.rs/) UI on top, your real-time 3D engine underneath — one GPU device,
zero copies, and the UI is never redrawn for a 3D frame. Pick a variant, copy its folder, replace
the cube.

**Pour démarrer une app : [`starter/`](starter/src/main.rs)**, un petit éditeur de scène
(cubes 3D, liste, inspecteur, UI translucide sur la 3D) qui applique toutes les règles
de perf du kit. Les dossiers ci-dessous sont les démos d'intégration par backend (un cube
qui tourne, caméra orbitale).

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
cargo run -p gpui3d-starter
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

## Starter

`starter/` est le squelette à copier : `engine.rs` (moteur wgpu sur son fil, cubes
instanciés éclairés, caméra) et `main.rs` (état et vues). Ce qu'il montre :

- **État dans des entités** : un `Object` par cube, un `Model` qui les liste. L'UI publie
  un instantané au moteur (`Shared::publish`) quand la scène change ; le moteur ne
  renvoie ses instances au GPU que sur une nouvelle version.
- **Une vue `.cached()` par panneau et par ligne de liste** (`uniform_list` de vues
  `Row`). Sélectionner ou modifier un objet ne reconstruit que ses lignes et
  l'inspecteur ; défiler rejoue les lignes translatées.
- **UI translucide au-dessus de la 3D** : une image 3D ne redessine jamais l'UI, et
  l'UI immobile n'est pas renvoyée au GPU. Mesuré : au repos, ~59 compositions/s pour
  2 vues reconstruites/s (le compteur fps).
- **Gestes caméra sans `notify`** : glisser dans la vue = orbite (le glisser continue
  au-dessus des panneaux), molette = zoom.

Pour aller plus loin : sélection par clic dans la 3D (lancer de rayon), champs de
saisie (gpui-component `Input`), sauvegarde de la scène.

## gpui-component

[`vendor/gpui-component`](vendor/gpui-component) (fork longbridge, adapté à ce GPUI) est
disponible mais **pas utilisé par le starter**. Il apporte ce qu'on ne réécrit pas en
une après-midi : éditeur de texte (`input` : IME, sélection, presse-papiers, annuler,
coloration), `table` et listes virtualisées, `dock` / panneaux redimensionnables, `tree`,
`select`, menus et menus contextuels, `popover` / `tooltip` / `dialog` / `notification` /
`sheet`, sélecteurs de date et de couleur, graphiques, et un système de thèmes.

Coût mesuré (`BENCH_WIDGETS=component`, mêmes lignes de mail qu'en GPUI pur, sans vsync) :
**+13 % d'instructions (−8 % de fps) quand tout se reconstruit à chaque image, +5 % (−4 %)
avec des lignes en vues `.cached()`**, aucun pic. L'écart vient de ce que les composants
dessinent en plus (+38 % d'éléments, +65 % de quads : bordures, fonds, conteneurs), pas
d'un coût par image caché. Points d'attention : racine `gpui_component::Root` obligatoire
(calques), `gpui_component::init(cx)`, un `Input` focalisé re-rend sa vue toutes les
~500 ms (curseur), dépendances lourdes à compiler (tree-sitter, syntect, ropey…).

Règle : composants pour ce qui est complexe (saisie, tables, dock, menus, dialogues),
GPUI pur ou vues `.cached()` pour ce qui se répète (lignes de liste, grilles). Pour
l'ajouter à une crate :

```toml
gpui-component = { path = "../vendor/gpui-component/crates/ui" }
```

puis `gpui_component::init(cx)` au démarrage et la vue racine enveloppée dans
`gpui_component::Root::new(vue, window, cx)`.

## Architecture

```
shell/            chrome GPUI + caméra + fil de rendu, commun aux trois moteurs
GPUI-WGPU/        moteur wgpu (WGSL)
GPUI-METAL/       moteur Metal natif (MSL)
GPUI-VULKAN/      moteur Vulkan natif (GLSL compilé en SPIR-V par build.rs)
starter/          app de démarrage (éditeur de scène)
bench/            banc de performance (défilement, overlay 3D, composants)
vendor/           forks — provenance : vendor/FORKS.md
  wgpui/          GPUI (Zed → gpui-ce → WGPUI) + patchs GPUI-3D
  gpui-component/ composants (longbridge) adaptés à ce GPUI
  priority-threadpool/  pool de fils (timers GPUI), course corrigée
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

Provenance des forks, révisions, licences et liste complète des patchs :
[`vendor/FORKS.md`](vendor/FORKS.md). Ces forks sont entretenus ici ; chercher `GPUI-3D`
et `Compat` dans `vendor/wgpui` avant d'y reprendre un correctif amont.

## Démarrer une app

Copier `starter/` et l'adapter : le moteur (`engine.rs`) s'échange contre le vôtre tant
qu'il garde le contrat (rendre sur son fil dans la `WgpuSurface`, lire l'instantané
publié par l'UI). Pour un moteur Metal ou Vulkan natif, reprendre `GPUI-METAL` ou
`GPUI-VULKAN` : `Renderer::new(surface)` crée les ressources, `render(surface, scene)`
encode une trame et la publie.

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
| `GPUI_SKIP_SAME_SCENE_UPLOAD` | quand seule une surface 3D a changé, la scène UI (inchangée) n'est pas renvoyée au GPU à chaque image |
| `GPUI_BIND_GROUP_CACHE` | bind group de page d'atlas gardé d'une trame à l'autre au lieu d'un par lot de sprites |

Sans interrupteur : glyphes au format GPU compact (88 octets au lieu de 168, dégradés et
transformations dans une table annexe), `DecorationRuns` à 2 runs en place au lieu de 32
(une ligne de texte pesait ~5 Ko), `LineWrapper` en `Box` et pris seulement pour tronquer.
`BENCH_MODE=overlay` : cube 3D plein écran derrière l'UI (`BENCH_UI=opaque|alpha|opacity|glass|none`,
`BENCH_3D_HZ`). L'UI n'est jamais reconstruite pour une image 3D ; immobile, elle ne coûte plus
que ~0,07 ms par image (renvoi évité), la transparence ne coûte rien au CPU, le verre flouté ~+20 %.
`GPUI_PRESENT_MODE=immediate` : fps débloqués, sans le plafond de 250 Hz de la boucle.

Le banc rapporte aussi la régularité : intervalles entre trames (`p50/p95/p99/max_ms`,
`hitches` > 2× médiane, `late` > 1,2× médiane) et temps CPU du fil principal par trame
(`work_*`), qui sépare notre travail des attentes du compositeur.

Mesuré (M1 Max, 60 Hz, médianes, lignes en vues) : 16,2 → 7,8 M instructions par trame,
CPU 27,7 % → 17,9 %, rendu identique au pixel près. Tests :
`cd vendor/wgpui && cargo test --lib --features test-support translated_reuse_tests`.
