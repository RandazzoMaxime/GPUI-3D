# Forks vendorés

Ces forks ont divergé de leurs amonts (patchs GPUI-3D) et ne sont plus
synchronisés automatiquement. Un correctif amont se reprend à la main
(voir « Reprendre un correctif amont »).

## `vendor/wgpui` : GPUI (paquet `gpui-ce`, crate `gpui`)

| Étape | Source | Révision |
|---|---|---|
| Origine | [Zed GPUI](https://github.com/zed-industries/zed) (`crates/gpui`) | — |
| Fork communautaire | [gpui-ce/gpui-ce](https://github.com/gpui-ce/gpui-ce) | — |
| Backend unique wgpu + winit | [Far-Beyond-Pulsar/WGPUI](https://github.com/Far-Beyond-Pulsar/wgpui) | `f9c3abb4a` (main) |
| Patchs GPUI-3D | ce dépôt | branche `optim` |

Licence : Apache-2.0 (`vendor/wgpui/LICENSE-APACHE`).

API rétablies pour gpui-component (`src/compat.rs`, blocs « Compat »), device
hôte partagé et sélecteur d'adaptateur, `WgpuSurfaceHandle` (surfaces zéro copie,
`present_synced_silent`, `request_window_redraw`), boucle réveillée seulement quand une
trame est voulue, couches retenues désactivées par défaut (`WGPUI_LAYERS`), corrections
de saisie macOS / Windows, double `MouseUp`, verre figé sous les modales.

**Patchs GPUI-3D** (marqués `GPUI-3D` dans le code) :

| Patch | Fichiers | Interrupteur |
|---|---|---|
| Interop native Metal / Vulkan (poignées brutes, verrou de queue, init des tampons, contrat de layout) | `elements/wgpu_surface.rs`, `platform/cross/surface_registry.rs`, `renderer.rs` | — |
| Sélecteur d'adaptateur honoré sur macOS, Metal préféré à MoltenVK, `PRIMITIVE_INDEX` non exigé | `platform/cross/render_context.rs` | — |
| Vues `.cached()` rejouées translatées (défilement), origine calée au pixel physique | `view.rs`, `window.rs`, `scene.rs` | `GPUI_VIEW_TRANSLATE` |
| Rejeu d'une plage en un bloc (un créneau `BoundsTree`) | `scene.rs` | `GPUI_BLOCK_REPLAY` |
| Tri des primitives par paires (clé, indice) | `scene.rs` | `GPUI_SORT_CACHED` |
| Glyphes au format GPU compact (88 o) + `mono_sprites_compact.wgsl` | `scene.rs`, `scene_pack.rs`, `renderer.rs`, `shaders/` | — |
| `DecorationRuns` à 2 runs en place, `LineWrapper` en `Box` et pris seulement pour tronquer | `text_system.rs`, `text_system/line.rs`, `elements/text.rs` | — |
| Bind group de page d'atlas gardé d'une trame à l'autre | `renderer.rs` | `GPUI_BIND_GROUP_CACHE` |
| Scène inchangée : instances pas renvoyées au GPU (3D qui anime derrière l'UI) | `scene.rs`, `renderer.rs`, `render_context.rs` | `GPUI_SKIP_SAME_SCENE_UPLOAD` |
| Sans vsync (`Immediate`) : plus de plafond de 250 Hz sur la boucle | `platform/cross/platform.rs` | `GPUI_PRESENT_MODE` |

Tests : `cd vendor/wgpui && cargo test --lib --features test-support`
(dont `view::translated_reuse_tests`).

## `vendor/gpui-component`

| Étape | Source | Révision |
|---|---|---|
| Origine | [longbridge/gpui-component](https://github.com/longbridge/gpui-component) | 0.5.2 |
| Adaptation WGPUI | ce dépôt, `vendor/gpui-component` | fork local `ebef0e1c`, vendoré en `a98026205`, lock `19e1cdf75` |
| GPUI-3D | ce dépôt | seules `crates/{ui,base,macros,assets}` ; feature `flamegraph` de gpui retirée du workspace |

Licence : Apache-2.0 (`vendor/gpui-component/LICENSE-APACHE`).

## `vendor/priority-threadpool`

| Étape | Source | Révision |
|---|---|---|
| Origine | [tristanpoland/priority-threadpool](https://github.com/tristanpoland/priority-threadpool) | `bb1ab91` |
| Patch GPUI-3D | course `fetch_add`/`push` corrigée (jobs orphelins ⇒ timers GPUI perdus) | — |

Licence : MIT (`vendor/priority-threadpool/LICENSE`). Branché par `[patch]` dans le
`Cargo.toml` racine.

## Reprendre un correctif amont

1. Repérer le commit amont (WGPUI, gpui-ce ou Zed pour `vendor/wgpui`).
2. L'appliquer à la main dans `vendor/…` : les chemins ne correspondent plus un pour
   un (Zed : `crates/gpui/src/…` ; ici : `vendor/wgpui/src/…`).
3. Rechercher `GPUI-3D` et `Compat` dans les fichiers touchés pour ne pas écraser
   un patch.
4. Relancer la suite du fork et le banc (`README.md`, section Performance) : parité et
   mesures.
5. Commiter en citant le commit amont.

Un correctif amont utile ici se reporte dans ce dépôt.
