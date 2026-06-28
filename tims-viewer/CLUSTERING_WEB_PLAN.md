# Plan — clustering on the web (DBSCAN + MIDIA-style stats)

Status: DESIGN (for codex review). The capstone: cluster the (focused) cloud, color it by cluster,
and show a MIDIA-style stats panel. Builds on the focus-lens (FOCUS_LENS_PLAN.md) and reuses the
native clustering + cluster-color path.

## What already exists (recon)

- **DBSCAN, render-safe.** `tims-viewer/src/cluster.rs` — hand-rolled, grid-accelerated DBSCAN in
  pure `std` (no rayon/native deps), `dbscan(&[[f32;3]], eps, min_pts) -> (Vec<i32>, k)`,
  noise = `-1`. Scales to ~2M points. **Compiles for wasm** — IF the `cluster` module is `pub` and
  not behind `#[cfg(feature="native")]` (must verify/ungate so the web crate can call it).
- **Cluster-color path, wired.** `GpuPoint._pad[0]` is the cluster id (sentinel `NO_CLUSTER =
  u32::MAX`); `ParamsUniform.color_mode == 1` makes `point_cloud.wgsl` color by cluster
  (`hsv2rgb(fract(id*0.618…), 0.65, 1.0)`, grey for noise) and **forces opaque** blending.
- **Native reference.** `app.rs::run_clustering` clusters the *filtered* resident subset (window +
  intensity + MS), capped at `CLUSTER_CAP = 2_000_000`; defaults `eps = 0.012` (normalized cube
  units), `min_pts = 8`; writes labels into `_pad[0]`, re-uploads, records `cluster_count` /
  `cluster_noise` / `cluster_input_count`.
- **MIDIA stats (Python, to reproduce):** a summary table (Points / Intensity / Clusters × Total /
  Cluster / Noise / Ratio) + four histograms across clusters (RT-extent, mobility-extent, m/z-extent,
  and cluster size). HDBSCAN adds per-point membership probability — **no Rust HDBSCAN exists**, so
  the web is **DBSCAN-only** for now.

## Architecture: client-side, reuse everything

Clustering runs **in wasm** on the retained `cpu_points`. No server. The lens makes it tractable:
**Focus a region → cluster that** (fewer points). The flow:

1. **Select the input:** ALL resident points passing the current filter (spatial crop + intensity
   floor + MS mask) — *not* the Display draw-count prefix (Display is a render perf knob; native
   clusters the full filtered resident). Build `idx: Vec<usize>` of those `cpu_points` indices for an
   exact writeback. **V1: require `idx.len() <= CLUSTER_CAP` (2M)** — if more, prompt "Focus a region
   first" (matches native, which refuses oversize input). Lowering the web cap further is fine since
   it blocks the main thread.
2. **Run** `tims_viewer::cluster::dbscan(&positions, eps, min_pts)` (positions = normalized cube
   coords of the survivors). Status "clustering…" (it blocks the main thread for seconds — like the
   volume build; worker later).
3. **Write back + recolor:** set `cpu_points[idx[j]]._pad[0] = label as u32` (or `NO_CLUSTER` for
   noise; non-survivors stay `NO_CLUSTER`), re-upload via `renderer.reset()` + `append(&cpu_points)`
   (full re-upload for V1 — no partial `_pad[0]`-only API exists), set `params.color_mode = 1`, and
   **switch the view to Points + force `PointMode::StructuralOpaque`** while clustered. ⚠ The web
   `frame()` always renders with `self.point_mode`, so additive mode would mush the cluster hues —
   render opaque when `color_mode == 1` (native does exactly this).
4. **Stats:** compute the MIDIA metrics from the labels + survivor points (below), render into the
   right-hand panel.

## Cluster lifecycle (state)

A clustering result is tied to the current loaded set + filter. Add `invalidate_clusters()`:
`color_mode = 0`, reset all `_pad[0]` to `NO_CLUSTER`, clear the stats/right-panel, restore the
user's point_mode. Call it on **Load / Focus / Back** (apply_load) and on **filter changes**
(crop/floor/MS — `filter_dirty`): either invalidate, or mark the result **stale** and disable
"Color by cluster" until re-run (so stale labels never colour a changed cloud). Track `clustered:
bool` + the result (`k`, noise count, per-cluster aggregates) on `Gfx`.

**Volume mode:** cluster coloring is a point-render concept and is invisible in Volume mode. On Run,
switch to Points (and grey the Cluster tab while in Volume), matching native.

## Stats to compute (web, from labels + survivors)

- **Summary:** `k` clusters; signal vs noise point counts; signal vs noise intensity sums; the two
  ratios (signal/noise points, signal/noise intensity) — exactly the MIDIA table.
- **Per-cluster aggregates** (one pass over survivors, bucket by label): count, intensity sum, and
  the bounding extent on each axis (mz/im/rt, in real units via `axis_bounds`). The MIDIA "unique
  cycle/scan/mz per cluster" → web equivalent = per-cluster **extent** (or distinct-bin count) on
  RT / 1/K₀ / m/z.
- **Four histograms across clusters:** distributions of {RT-extent, 1/K₀-extent, m/z-extent, size},
  drawn as SVG bars (same technique as the crop strips / projection maps).

## UI

- **Left rail — a new "Cluster" tab** (alongside View / Focus): ε slider, min-pts slider, a **Run**
  button, a **Color by cluster** toggle (sets `color_mode`), and a readout (`k` clusters, noise %).
- **Right panel — results** (a second `.rail` on `right: 16px`), shown only when the Cluster tab is
  active *and* a result exists: the summary table + the four histograms. Hidden otherwise so the
  canvas keeps the full width. Responsive: collapse/overlay on narrow viewports.

## Phasing

- **C1 — cluster + color.** Verify/ungate the `cluster` module for wasm; Cluster tab (ε / min-pts /
  Run / Color-by-cluster); run DBSCAN on the filtered survivors (capped), write `_pad[0]`, re-upload,
  `color_mode = 1`; HUD/readout `k` + noise %. Invalidate on reload.
- **C2 — stats panel.** Right-hand panel with the summary table + four cluster histograms.
- **C3 — interactions.** Click a cluster (in a list or on the cloud) → isolate/highlight it; optional
  per-cluster intensity/RT profile; export labels.

## Open questions / risks (for codex)

- **Module reachability.** Is `tims-viewer/src/cluster.rs` `pub` and free of `#[cfg(feature=
  "native")]` so `tims-viewer-web` (built `--no-default-features`) can call `dbscan`? If gated,
  ungate it (it's pure std).
- **Input selection + cap.** Cluster the filtered survivors capped at 2M — for a full-run resident
  set (>2M) we cluster a prefix; is that acceptable, or require a Focus first? Keep the survivor↔
  `cpu_points` index map straight so writeback targets the right points.
- **eps units / scaling.** ε is in normalized cube units (uniform across m/z·1/K₀·RT), so the metric
  is isotropic in normalized space but anisotropic in real units. MIDIA had an `mz_scaling`; do we
  need per-axis weights for the web (defer to C3)? After a Focus, the cube is re-normalized to the
  region, so the same ε clusters finer in real units — intended, but document it.
- **Re-upload cost.** Writeback re-uploads the resident `cpu_points` (`reset` + `append`) — up to
  N_cap × 32 B. Acceptable one-time per Run? Or update only the cluster-id attribute (partial
  `write_buffer`) to avoid re-sending positions.
- **color_mode + blend.** Cluster mode forces opaque in the shader — confirm it composes with the
  Density/Solid `point_mode` and the Display draw-count; the recount HUD is intensity/MS based, not
  cluster.
- **Main-thread block.** 2M-point DBSCAN in wasm is seconds and freezes the tab — status + (later)
  a worker; the grid-accelerated O(n) helps but verify.
- **Lifecycle correctness.** Clearing ids/color on reload/Focus; not leaving a stale colored cloud or
  a stale stats panel; what Volume mode shows (clusters are a point-render concept — disable or grey
  the Cluster tab in Volume mode, or build a per-cluster density?).
- **Right panel layout.** A second fixed rail — z-index, the canvas width, axis-label overlay, and
  small/again-narrow viewports.
- **HDBSCAN.** Deferred (no Rust impl); if needed it's a server-side (Python imspy) path — out of
  scope for the wasm-only V1.

## Codex review (incorporated)

**Feasibility gate: PASS** — `cluster` is `pub mod cluster;` with no native cfg, so
`tims_viewer::cluster::dbscan` is callable from the wasm web crate as-is (no ungate needed).

- **[blocker] opaque blend** — the web `frame()` always renders `self.point_mode`, so cluster colour
  in additive mode mushes. Render `StructuralOpaque` when `color_mode == 1` (folded into step 3).
- **[risk] input set** — cluster the *filtered resident* set, not the drawn prefix; build the index
  map from exactly that set for writeback (folded into step 1).
- **[risk] cap** — V1 requires `idx.len() <= CLUSTER_CAP` (Focus first if larger); a lower web cap is
  fine.
- **[risk] re-upload** — only `reset()+append()` exists; V1 does a full re-upload (a partial
  `_pad[0]` update would need a new renderer API + is awkward at the 32-byte stride).
- **[risk] invalidation** — `apply_load` + filter changes must clear cluster state / mark stale
  (folded into Lifecycle).
- **[risk] Volume** — switch to Points on Run / grey the tab in Volume (folded in).
- **[risk] main-thread block** — 2M DBSCAN is seconds–tens of seconds in wasm; status + a lower V1
  cap, worker later.
- **[nit] stats** — `GpuPoint` has no cycle/scan/original-m/z metadata, so the web histograms are
  per-cluster **extents** (from `axis_bounds`), not exact unique counts; intensity sums use
  `intensity * weight`.
- **[nit] layout** — the right stats rail competes with the colorbar; hide the colorbar in cluster
  mode / reserve the right-rail layout.

**Codex's simpler first cut (= our C1):** Cluster tab (eps / min_pts / Run), require filtered
resident ≤ CLUSTER_CAP, full `reset()+append()` re-upload, force opaque, basic `k / noise / input`
stats. Defer partial GPU updates, the worker, and the full MIDIA histograms to C2/C3.
