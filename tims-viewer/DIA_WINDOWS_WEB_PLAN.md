# Plan — DIA isolation-window overlay on the web (precursor selection in the scan×m/z plane)

Status: DESIGN (for codex review). Port the native viewer's DIA/MIDIA window overlay to the WASM
viewer's 3D scene: group-colored wireframe rectangles showing where the quadrupole selects precursors,
in the m/z × 1/K₀ (scan) plane. Reuses the native window-read logic + the render-safe
`AnnotationRenderer` + `group_color`.

## What exists (recon)

- **Native** (`src/data/loader.rs::build_annotations`, DIA branch): reads `read_dia_ms_ms_windows` +
  `read_dia_ms_ms_info`, maps each window's `scan_num_begin/end → 1/K₀` (per-group representative
  frame, `scan_to_inverse_mobility`), and draws the `(m/z × 1/K₀)` footprint
  `[isolation_mz ± isolation_width/2] × [im0,im1]` as a rectangle at `N_SLICES=6` RT slices, colored
  by `group_color(window_group, n_groups)`. Subsamples a fine MIDIA diagonal to `TARGET_WINDOWS≈800`.
- **Renderer**: `AnnotationRenderer` (render-safe, wasm-reachable) draws `LineVertex` line-lists with
  a depth-Always overlay pass + a per-vertex filter cull (`update_filter`). The web already uses ONE
  instance for the axes cube.
- **`group_color`** is in render-safe `render/colormap.rs` — callable from the web.
- **Server** (`serve.rs`): `MetaIndex.mode = LoaderMode::Real { path }` carries the run path, so the
  server can read windows at init; the dataset opens via `TimsDataset::new("NO_SDK", path, …)`.
- **Web** has NO window data today (`/meta` doesn't send it; the web never parses it).

## Architecture: run-level `/windows`, re-normalized client-side

Windows are a property of the **run**, not the region — so fetch them **once**, re-normalize locally
on each focus (no re-fetch). Cleaner than embedding per-region in `/meta`.

### Server
1. **Helper** (`loader.rs`): `dia_window_rects(mode: &LoaderMode) -> (Vec<WindowRect>, u32)` where
   `WindowRect { group: u32, mz0,mz1,im0,im1: f64 }` in **real units**. DIA only (open the dataset,
   read windows+info, scan→im per group, subsample to `TARGET_WINDOWS`); Demo/DDA → empty. Factor the
   real-unit math out of `build_annotations`' DIA branch so both share it (the native then normalizes
   the same rects to `LineVertex`).
2. **State**: compute once in `serve()` init from `plan.meta.mode`; store `windows: Vec<WindowRect>`,
   `n_window_groups: u32`.
3. **Endpoint** `GET /windows` → JSON `{ "n_window_groups": N, "windows": [[g,mz0,mz1,im0,im1], …] }`.
   Run-level, tiny, served from State (no per-region build).

### Web
1. **Fetch `/windows` once** at startup → `Vec<WindowRect>` (real units) + `n_window_groups` on `Gfx`.
2. **Build the overlay** (`rebuild_window_overlay`): for each window × `N_SLICES` RT slices, normalize
   the 4 `(m/z,1/K₀)` corners via `axis_bounds`, push the 4 edges as `LineVertex` colored by
   `group_color(g, n_window_groups)`. Port `push_rect_mz_im`. Upload to a **second
   `AnnotationRenderer`** (`windows`).
3. **Cull off-region** windows with `update_filter([-1,1]³)` so windows outside the focused cube clip
   (mirrors the native per-vertex cull) — windows normalize outside `[-1,1]` when off-region.
4. **frame()**: `windows.update_camera` each frame; when `show_windows`, `windows.render(rpass)` after
   the axes (depth Always). Re-`update_filter` from the crop sliders if we want them to honor crops
   (decide: cull to cube only, or to the crop box too).
5. **Rebuild triggers**: on `apply_load` (new `axis_bounds` → re-normalize) and on the toggle.
6. **UI**: a **"Show DIA windows"** toggle (Focus tab), disabled when `n_window_groups == 0` (demo/
   DDA). Per-group visibility bitmask is a follow-up (native has it; V1 shows all groups).

## Phasing

- **W1** — server `/windows` (helper + State + endpoint) and the web overlay (fetch, build, second
  renderer, toggle, rebuild on load). DIA only, all groups.
- **W2** — per-group visibility (checklist / legend, like the native bitmask), DDA precursor crosses,
  honor crop sliders.

## Open questions / risks (for codex)

- **Helper refactor / reuse.** Factor `dia_window_rects` out of `build_annotations` cleanly (the scan→
  im per-group + subsample + symmetric scan-widening for MIDIA) so the native overlay and the server
  endpoint don't drift. Keep the native rendering identical.
- **Server init cost / availability.** Reading windows + opening the dataset once at `serve()` init —
  acceptable? Demo/DDA must no-op (empty). Confirm `read_dia_ms_ms_windows`/`scan_to_inverse_mobility`
  are reachable from `serve.rs` (they're rustdf/native; the server is native).
- **Normalization consistency.** Web normalizes windows to `axis_bounds` (= region bounds from
  `/meta`), exactly like the cloud, so they align; on refocus, re-normalize the same real-unit list to
  the new bounds (no re-fetch). Off-region windows map outside `[-1,1]` → `update_filter` clips them.
- **RT slicing.** The native draws the footprint at `N_SLICES` RT slices so the recurring selection
  sits on the data through the run. Port the same; confirm it reads well in the web (a focused RT
  region still gets slices within `[rt0,rt1]`).
- **Payload / subsample.** `TARGET_WINDOWS≈800` real-unit rects = a few KB JSON. A fine MIDIA scheme
  (~15k windows) must subsample; surface the stride so we don't silently drop.
- **Second AnnotationRenderer.** Confirm a second instance composes (its own buffer/camera/filter) and
  the extra depth-Always overlay pass after axes is correct; per-frame `update_camera` for both.
- **Toggle/UI + lifecycle.** Disabled when no windows; rebuild on load; does it interact with cluster
  coloring / volume mode (windows are a 3D-scene overlay — show in Points and Volume? native shows
  over the scene; decide for Volume).
- **Crop interaction.** Should windows honor the crop sliders (cull to the crop box) or only the cube?
  The native culls to the cube; the crop box is a separate decision.
