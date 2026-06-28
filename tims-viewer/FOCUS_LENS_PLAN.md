# Focus-Lens LOD — plan (tims-viewer web)

Status: DESIGN, codex-reviewed (findings folded in — see "Codex review" at end).
Builds on NATIVE_WEB.md Phase 2 (point server + WASM shell).

## Problem

A real run is ~1.1 billion points (~33 GB at 32 B/point). The GPU can hold only
**N_cap = max_buffer_size / 32** points (already the hard cap in the web shell). We need both an
overview of the whole run and the ability to drill into a region at high detail, never exceeding
N_cap. The current 12M `--budget` is an arbitrary pre-cap; the real ceiling is N_cap.

## Mental model: a fixed-capacity 4D lens

The GPU is a lens of fixed point-capacity (N_cap) that we move and resize over the run. Resizing the
lens **smaller** concentrates the same point budget into a smaller volume → higher local resolution
at constant GPU cost. The lens is **4D**: m/z × 1/K₀ × RT × **intensity**. Narrowing *any* axis
(including intensity) shrinks the region, so the same budget resolves it more finely; once the
survivors in the source fall below N_cap, we load **all** of them (stride 1, full detail) and the
displayed count simply drops.

## The single primitive — and its contract

```
load(region: Region4D, budget: u32) -> LoadResult     // budget ≤ N_cap, enforced
```

`load()` must return ONE coherent object (codex: the contract was too thin):

```
LoadResult {
  points:        Vec<GpuPoint>,   // shuffled, normalized to display_bounds
  returned:      u32,             // == points.len()
  survivors_est: u64,             // est. total points matching region in the source (pre-sample)
  sampling:      { stride|reservoir, seed },   // how `points` was drawn from survivors
  source_region: Region4D,        // the committed filter (for cache key + intensity semantics)
  display_bounds:[(f64,f64);3],   // the [-1,1] re-normalization basis (mz,im,rt)
  meta:          { hist[mz,im,rt], i_hist, percentiles, sample_based: true },
}
```

`Region4D = { mz:(f64,f64), im:(f64,f64), rt:(f64,f64), intensity_min:f32 }`. Server returns
`min(survivors_est, budget)` points. Client rebuilds the GPU buffer, re-normalizes to
`display_bounds`, and rebinds axes / crop ranges / strips from `meta` — **atomically**.

| Operation     | call                                  |
|---------------|---------------------------------------|
| coarse view   | `load(full_bounds, N_cap)`            |
| pre-load      | the first `load(full, N_cap)` at start|
| focus / drill | `load(crop_region_4d, N_cap)`         |
| re-load       | any subsequent `load(...)`            |
| reset / back  | replay a cached `LoadResult` (stack)  |
| display       | runtime `draw_count` (no reload)      |

## ⚠ Budget must be region-relative (codex blocker 1)

The loader computes `stride_for(total_estimate, budget)` from the **full-run** estimate *before*
filtering ([loader.rs:203], filter applied later at [loader.rs:388]). So a naive region query keeps
the coarse full-run stride and **focus would not densify** — the core premise. Fix, required in
Phase A:

- Estimate **survivors in the region** first: frames in `[rt0,rt1]` give the frame set; scale the
  per-frame point estimate by the m/z·1/K₀·intensity selectivity (cheap heuristic) OR do a real
  **count pass** over the filtered stream.
- Drive the sampler from `survivors_est`, not the full-run total: stride = `survivors_est / budget`
  (≥1), or switch to a **bounded reservoir** over the filtered stream (size = budget) so we don't
  need an exact pre-count. Reservoir is the robust default; it self-limits to `budget` and yields
  stride 1 automatically when survivors ≤ budget.
- Return `survivors_est` separately from `returned` so the client/HUD can say "showing all" vs
  "sampled X of ~Y".

## Two-tier filtering (the crux) + intensity's dual role

The crop sliders and the intensity floor do double duty:

1. **Preview (instant, in-GPU cull):** the existing window + intensity-floor cull runs in the
   vertex/compute shader on the *currently loaded* sample. Free and live; the displayed count drops
   as you filter (see Displayed count).
2. **Commit (Focus → re-load):** the *same* 4D crop+intensity box becomes the `Region4D`; the
   re-load spends the full budget on survivors and re-normalizes the cube to fill the view.

**Intensity has two distinct roles (codex risk).** The floor is BOTH a live display cull (on the
resident set) AND, when committed via Focus, a *source* filter. After a focus with `intensity_min`,
low-intensity points are no longer resident — lowering the live floor below the **committed source
floor** cannot reveal them. So state splits explicitly:

- `display_floor` — live cull on resident points (instant, reversible).
- `committed_floor` — part of `source_region` of the current load.
- If the user drags `display_floor` **below** `committed_floor`, the UI marks it "needs reload"
  (greys the region beyond what's loaded) and offers Re-load; it never silently shows nothing.

## Displayed count (must reflect the 4D filter)

"Displayed" = points surviving the active cull, not resident / `draw_count`. Use **CPU recount on
both backends** (simpler and correct; avoids the compaction pitfalls below):

- Keep the resident `Vec<GpuPoint>` in scene state (codex risk: `Gfx` doesn't retain it today — add
  it; update on every reload). Recount survivors **on filter change** (not per frame) over the same
  `draw_count` prefix Display uses. O(resident) occasionally — fine at ≤ N_cap.
- HUD shows `displayed / resident` (+ "of ~survivors_est in source" when sampled).

Why not the WebGPU indirect count (codex blocker 2): the compaction shader culls window+MS+frustum
but **not intensity** ([compact.wgsl:67] vs [point_cloud.wgsl:86]), so `instance_count` overcounts;
and `draw_args` lacks `COPY_SRC` for readback ([point_cloud.rs:377]). CPU recount sidesteps both. (If
we later want GPU truth, add intensity to compaction + `COPY_SRC` + a staging readback, and define
whether the count includes frustum culling.)

## Server: query-capable with a job + cache layer (codex risk)

`serve()` today precomputes one `Arc<[u8]>`, ignores query strings, and handles requests serially in
the `tiny_http` loop ([serve.rs:36], [serve.rs:85]). A blocking region load would stall everything.
Redesign:

- **Job layer:** region loads run off the accept loop (worker thread/pool); the HTTP handler returns
  job status / the finished `LoadResult` bytes. Support cancellation (a newer focus supersedes an
  in-flight one).
- **Shared result:** `/meta` and `/points` for a region serve from ONE computed `LoadResult` keyed
  by `(Region4D, budget, display_bounds)` — never recompute independently.
- **Cache:** an LRU of `LoadResult`s keyed as above. Each entry is an inseparable bundle
  (points + meta + display_bounds + source_region); cached bytes are normalized to *their* bounds,
  so they cannot be mixed. **Only cached stack entries are instant** — nested Back is instant only if
  each focused `LoadResult` is still cached; otherwise it re-queries.
- Bind localhost-only; keep CORS. Query: `GET /points?n=&mz0=&mz1=&im0=&im1=&rt0=&rt1=&imin=` and
  matching `/meta?…`.

### RegionFilter extension (loader)

`RegionFilter` is `{ mz, im }` today (RT via frame selection). Extend with `intensity_min: f32` so
intensity is a source-side cull. RT stays frame-selection-based; the web query maps `rt0/rt1` →
frame_ids, `mz/im/imin` → `RegionFilter`. **Note (codex):** apply `intensity_min` *before* peak
selection so peaks are chosen from survivors; and label region `meta` as **sample-based** (hists /
percentiles are over the kept systematic sample + peak tail, not the true region distribution) —
optionally add a true stats pass later.

## Re-normalization must be atomic (codex risk)

Points are normalized before upload, and the shader also has a `params.focus` re-fit path
([point_cloud.wgsl:109]). A focused reload that normalizes the region to `[-1,1]` while leaving old
crop slider values or shader focus active → double-crop/remap. On every load, **atomically**:
reset spatial crop uniforms + sliders to full `[-1,1]`, turn shader `focus` off, and set
`axis_bounds`, tick labels, distribution strips, and crop readouts from `LoadResult.display_bounds`.

## Client: reload path

- **N_cap detect:** `device.limits().max_buffer_size / size_of::<GpuPoint>()` (clamp by
  storage-bind limit on the compaction path). Already computed for the capacity cap; surface it.
- **`load(region, budget)`** (async): fetch `/meta?…` + `/points?…&n=budget`, rebuild the
  `PointCloudRenderer` at the new capacity, upload, retain CPU points, re-normalize + rebind
  atomically, reset Display to 100%, show a loading state during the query.
- **Focus stack:** `Vec<Region4D + LoadResult-ref>`; Focus pushes the current crop+intensity box,
  Back pops (replays cache or re-queries). Full-run region is the base.
- **UI:** Focus button, Back/Reset, Load field (budget within `[0, N_cap]`), the existing Display
  slider (runtime draw_count). Loading spinner during queries; "needs reload" hint when the live
  floor drops below the committed floor.

## Decisions (confirmed with user)

1. Focus **re-normalizes** the region to `[-1,1]`; full bounds cached for Back. ✔
2. **Nested drill-down via a focus stack**. ✔
3. Default pre-load budget = **N_cap**. ✔
4. Focus region source = **crop box + intensity floor + a Focus button**; 2D box-select feeds the
   same primitive later. ✔
5. **Drop the server "pool" concept** — every (uncached) load re-queries the source; `--budget` is
   only a fallback default for N_cap when the GPU can't be probed. ✔
6. **Caveat:** the lens is 4D — intensity shrinks the region and lowers the displayed count, both as
   a live cull and as part of the Focus re-load. ✔

## Phasing (reordered per codex — de-risk the hard parts first)

- **A1 — server job/cache/query model.** Query parser, off-loop job runner, `LoadResult` cache,
  shared `/meta`+`/points`. No new sampling yet.
- **A2 — region + intensity sampling on DEMO (known counts).** Extend `RegionFilter` with
  `intensity_min`; make sampling **region-relative** (reservoir or count-then-stride); validate that
  a focused region actually spends the budget and `survivors_est` is right, using the demo source
  where counts are known. *(This is the blocker-1 proof.)*
- **A3 — client reload/rebind.** `load()` end to end on full bounds: rebuild renderer, retain CPU
  points, atomic re-normalize + rebind, Load field, N_cap surfaced.
- **B — displayed-count parity.** CPU recount on both backends; HUD `displayed / resident`
  (+survivors). Split `display_floor` vs `committed_floor`.
- **C — Focus / Back.** Crop+intensity box → Focus re-load → focus stack + cache replay + "needs
  reload" affordance.
- **D — 2D projection tab + (optional) progressive streaming.** Heatmaps of
  `proj_mz_im/proj_mz_rt/proj_im_rt`; box-select writes a `Region4D` and calls the same `load()`.

## Risks / open questions

- **Region-relative sampling is the crux** — without it, focus doesn't densify (blocker 1). Reservoir
  sampling over the filtered stream is the recommended default (no exact pre-count needed).
- **Region query latency:** re-reading frames per Focus takes seconds → non-blocking job + loading
  state; progressive first paint is a later option.
- **Sample-based meta:** region hists/percentiles approximate the region (kept sample + peak tail);
  label as such, add a true stats pass only if needed.
- **Empty regions:** a focus yielding 0 survivors must no-op (keep prior view + warn).
- **Cache memory:** each cached stack entry is ~N_cap × 32 B; bound the LRU.

## Codex review (incorporated)

- [blocker] region budget was driven by the full-run estimate → focus wouldn't densify → added the
  "Budget must be region-relative" section (reservoir / count-then-sample) and `survivors_est`.
- [blocker] WebGPU displayed-count via indirect `instance_count` overcounts (no intensity cull in
  compaction) and `draw_args` isn't `COPY_SRC` → switched Displayed count to **CPU recount on both
  backends** + retain CPU points in scene state.
- [risk] intensity's dual role (live cull vs committed source floor) → split `display_floor` /
  `committed_floor` + "needs reload" affordance.
- [risk] server can't be lightly extended → job/cache layer, off-loop loads, shared `LoadResult`.
- [risk] cache correctness → keyed by `Region4D+budget+display_bounds`, inseparable bundle, only
  cached stack entries instant.
- [risk] intensity-at-source vs peak preservation / histograms → apply floor before peak selection;
  label meta sample-based.
- [risk] re-normalization double-apply → atomic reset of crops + shader focus + bounds rebind.
- [risk] CPU recount needs retained CPU points → add resident `Vec<GpuPoint>` to scene state.
- [nit] phase order → reordered to A1/A2/A3 → B → C → D so sampling + displayed-count land before
  Focus UX.
