# Feature idea — volume rendering on the web

Status: IDEA (pre-design, for codex feasibility review). Brings the native viewer's volume mode to
the WASM shell, and exploits the focus-lens for high-resolution local volumes.

## What the native volume is (recon)

A self-contained **fragment-shader raymarcher** (`src/render/volume.rs`, `src/shaders/volume.wgsl`):

- A **3D density grid** `VOLUME_DIMS = [256, 128, 192]` (m/z · 1/K₀ · RT), `R16Float`.
- Built **CPU-side**: each point is trilinearly deposited (cloud-in-cell, mass-conserving) into one of
  two grids by MS level (`grid_ms1`, `grid_ms2`); a toggle folds them into the display grid. Peak
  chunks are not deposited. Normalized by `density_scale` (max voxel → ~60000 in f16).
- Uploaded via `write_texture` (3D extent); a fullscreen triangle raymarches it in the fragment
  shader (ray↔box intersect, per-sample transfer lookup, front-to-back **composite** or **MIP**).
- `VolumeUniform` (128 B): `inv_view_proj`, `box_min/max`, `transfer[mode,i_min,i_max,exposure]`,
  `steps`, `style` (composite/MIP), `colormap_id`, `density_scale`, `focus`.
- **Compiles for wasm already** (no native deps: wgpu/bytemuck/half), not feature-gated.

## Why it fits the web — and the lens synergy

- The renderer + grid math are wasm-ready; the web client **already retains the points**
  (`cpu_points`), so the CPU trilinear deposit ports directly (build MS1/MS2 grids from the resident
  set, optionally skipping points below the intensity floor during deposit).
- **Focus → high-res volume.** The grid is fixed-resolution; over a *focused region* (a small box)
  the same 256×128×192 voxels resolve far finer detail than over the whole run. So "Focus, then
  Volume" is the fine-grain mode, and a re-focus rebuilds the local volume — the lens and the volume
  reinforce each other.

## The backend decision (headline — corrected by codex)

**Earlier premise ("WebGL2 has no 3D textures") was WRONG.** WebGL2 (the API) supports 3D textures
(`texImage3D`), and wgpu-22's WebGL2 backend maps `TextureDimension::D3` → `TEXTURE_3D` and uploads
via `tex_sub_image_3d`, with a default **max 3D dimension of 256** — which exactly fits the native
`VOLUME_DIMS = [256, 128, 192]`. `R16Float` is reported sampleable + **filterable** on that path. So
the **native 3D-texture raymarcher may run directly on WebGL2** — the current backend — with no atlas.

That collapses the design: V1 is a near-direct reuse of `volume.rs` + `VolumeGrid`, not a rewrite.

1. **Native 3D-texture path (recommended).** Reuse `VolumeRenderer` + `VolumeGrid` as-is. First add a
   **forced-WebGL2 capability smoke probe** (`R16Float` 3D texture + linear filtering + `write_texture`)
   so we *confirm* it on the target browser; gate volume off (or fall back) if it ever fails.
2. **2D-atlas emulation (fallback only).** Flatten the 192 slices into a tiled 2D R16F atlas with
   manual trilinear — *only* if the 3D path fails a probe. More shader work, edge-bleed care, bounded
   by `max_texture_dimension_2d` (WebGL2 guarantees only 2048; pick the layout from the real limit).

**Recommendation:** smoke-test the native 3D path on forced WebGL2 first; if green (expected), ship
that. Keep the atlas as a documented fallback, not the default.

## Sketch of the web implementation (native 3D path)

- **Reuse, don't reimplement.** `VolumeRenderer`, `VolumeGrid` (`deposit`, `combine`, `density_scale`,
  `to_f16_scaled`) and `volume.wgsl` are public + render-safe (wasm-reachable). The web shell builds
  the grid and drives the renderer; it does NOT mirror the deposit math.
- **Grid build** (`web/src/lib.rs`, from the retained `cpu_points`): `VolumeGrid::deposit` each point
  into `grid_ms1` / `grid_ms2`, `combine` by the current MS mask. Track `max_density`. Built **once
  per load/focus** (and on MS-toggle), with a "building volume…" status; for big loads, chunk across
  RAF / yield (or a worker) later — millions × 8 deposits + f16 pack blocks the main thread otherwise.
- **Upload + draw**: `VolumeRenderer::upload(to_f16_scaled())` (3D `write_texture`) + a fullscreen
  raymarch pass. Reuse the existing camera `inv_view_proj`, depth attachment, and colormap LUT.
- **View mode**: add an explicit `ViewMode { Points, Volume }`; in `frame()`, Volume mode draws the
  **volume + axes** (not the points). Don't make it "just a second pass" — branch deliberately.
- **What the volume reflects (product decision, per codex):** *all resident points* + *current MS
  mask* + the *current crop box* (passed as `box_min/max` into the volume shader, which already has a
  window box). Do **not** tie it to the Display % (that's a point draw-count, not a density). The
  intensity floor: either ignore it in the grid, or bake it during deposit and accept a rebuild when
  the floor changes — decide explicitly.
- **UI**: a **View** segment "Points / Volume" + volume-only controls (composite/MIP style, steps,
  density transfer mode/exposure), reusing the colormap select.

## Phasing

- **V0 — capability probe.** Forced-WebGL2 smoke test: create an `R16Float` 3D texture, `write_texture`
  a tiny grid, sample with linear filtering in an offscreen pass. Decides 3D-path vs atlas fallback.
- **V1 — native 3D volume.** Wire `VolumeRenderer` + `VolumeGrid` into the web `Gfx`/`frame()`, build
  the grid from `cpu_points` on load/focus, Points/Volume toggle, density transfer/steps/MIP controls.
  Whole-run and focused volumes (the lens gives the fine grain). Capability-gate: disable with a
  message if the probe fails.
- **V2 — atlas fallback (only if V0 fails on a target).** 2D R16F atlas + manual trilinear.
- **V3 — polish.** Gradient/lighting shading, isosurface, per-MS coloring, density auto-range on
  enter (mirror native), coarser-while-moving, off-main-thread grid build.

## Open questions / risks (for codex)

- **WebGL2 R16F atlas support.** Confirm `R16F` sample + `OES_texture_half_float_linear` filtering on
  the target (Apple M-series via ANGLE/Metal). Fallback to `RGBA8` packing of f16 if linear R16F is
  unavailable. Max 2D texture size vs the atlas (4096×1536 fits 8192). Upload row alignment.
- **Grid build cost.** Trilinear deposit of up to ~N_cap points (millions) CPU-side per load — tens to
  hundreds of ms, blocking the frame. Acceptable one-time on load? Or chunk / web-worker (later).
- **Manual trilinear correctness.** Slice bracketing + tile UV math (clamp at slice/tile edges; avoid
  bleeding across tile borders — pad tiles or clamp UVs). Compare visually to native 3D trilinear.
- **Memory.** `grid f32` (256·128·192·4 ≈ 25 MB ×2 for MS) + atlas (~12 MB f16). Bounded, fine.
- **Filter composition.** Should the volume reflect the window crops + intensity floor + MS like the
  points do? Native builds raw MS grids + folds by toggle (no window/intensity in the grid). For the
  web, the resident points are already the focused set; applying the floor during deposit is a cheap
  win, but window crops would need either re-deposit or a box test in the shader (`box_min/max`).
- **Reuse vs reimplement.** Is the native `VolumeGrid::deposit` / `to_f16_scaled` reachable from the
  web crate (render-safe), so V1 reuses the deposit math instead of mirroring it?
- **Scope.** With the 3D path, V1 is mostly *wiring* (reuse the renderer + grid; new code = grid build
  from `cpu_points`, the view-mode branch, the controls, the probe). Much smaller than the atlas.

## Codex feasibility review (incorporated)

Verdict: feasible, and **simpler than first written** — the atlas premise was wrong.

- **[blocker→fixed] "WebGL2 has no 3D textures" is false.** wgpu-22's WebGL2 maps `D3` →
  `TEXTURE_3D` (max dim 256, fits `256×128×192`) and `R16Float` is filterable. → Recommendation
  flipped to the **native 3D path**, smoke-probed on forced WebGL2 first; atlas demoted to fallback.
- **[risk] grid build blocks the main thread** (millions × 8 deposits + f16). → build once per
  load/focus with status; chunk/worker later.
- **[risk] `VolumeGrid` reuse is viable** — `deposit`/`combine`/`density_scale`/`to_f16_scaled` are
  public + render-safe; only raw `data` is private (we don't need it). → reuse, don't mirror.
- **[risk] filter semantics need a decision** — volume = all resident points + current MS mask +
  current crop box (in-shader `box_min/max`); not tied to Display %; bake the floor only if accepting
  rebuild-on-floor-change.
- **[risk] frame integration** — add an explicit `ViewMode`; Volume draws volume + axes, not points.
- **[risk] atlas (fallback only)** — `R16F` 2D sample/filter looks supported; `4096×1536` is not
  WebGL2-guaranteed (only 2048) — size from `max_texture_dimension_2d`; clamp tile UVs / add gutters
  to avoid bilinear bleed; cell-centered sampling must match `dim*norm - 0.5`.
- **[nit]** the native `volume.rs` header comment is stale (says MIP/nearest MVP; code is trilinear
  additive) — update before treating it as authority.

**Bottom line (codex):** start with **V0 — a forced-WebGL2 capability probe** of the existing 3D
`R16Float` renderer. If green (expected), V1 is a direct reuse; the atlas is only needed if a target
browser fails the probe.
