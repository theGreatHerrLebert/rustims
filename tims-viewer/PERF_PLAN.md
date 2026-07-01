# tims-viewer performance plan (pre-merge)

Source: a 5-agent efficiency deep-dive (data pipeline, server transport/memory, GPU/render,
clustering, Python) over the `feature/tims-viewer-web` branch. The core algorithms are already
well-tuned; the wins are three themes — **stop copying full buffers, stop working when idle, cache
recomputes.**

## Execution model
Tier-1 is landed via a multi-agent workflow: one agent per **subsystem group** (disjoint file sets)
implements its items in an isolated git worktree, builds + tests + commits; a second adversarial
agent verifies behavior-preservation per group; only verified commits are cherry-picked onto the
feature branch. Tier-2/3 are follow-ups.

---

## Tier 1 — this pass (surgical, low-risk, test-covered)

### Group A — `serve.rs`
- **A1** body build: replace `bytemuck::cast_slice(&points).to_vec()` with an in-place
  `bytemuck::allocation::cast_vec::<GpuPoint,u8>(points)` → **−356 MB transient + one ~50–100 ms
  memcpy** per build. (Also flagged by the loader agent.)

### Group B — `data/loader.rs`
- **B1** fold hot loop: replace `global_i % stride_u64` with a countdown counter (preserve the
  "index 0 kept" phase the tests assert) → removes an integer division over **100 M+ points**.
- **B2** peak-tail: `select_nth_unstable_by(peak_room, …)` instead of a full `sort_by` of ~1 M peaks
  (order irrelevant — shuffled downstream).
- **B3** reserve capacity where survivor Vecs grow in a loop (if any remain after B1/B2).

### Group C — `web/src/lib.rs` + `render/volume.rs` + `shaders/volume.wgsl`
- **C1** clustering boundary: drop the `flat`↔`[f32;3]` rebuilds — `Float32Array::from(cast_slice::
  <[f32;3],f32>(&positions))` on send and `cast_slice(flat)` in the worker → **−144 MB memcpy per
  Run** (6 M pts). Reserve `idx`/`positions` gather Vecs (`with_capacity`).
- **C2** `VolumeGrid::density_percentiles`: `select_nth_unstable_by` (quickselect) instead of a full
  sort of up to 6.3 M voxels on every re-voxel.
- **C3** raymarch default `vol_steps` 256 → 128 (+ add the composite early-out to the MIP branch) →
  ~2× volume fragment throughput on the fill-bound WebGL2 path.

### Group D — Python (`cluster_service.py`, `midia/*.py`)
- **D1** `cluster_service.py`: keep **float32** (drop `.astype(float64)` — sklearn preserves f32 and
  eps≈0.012 has huge headroom) → working set 144 → 72 MB; add `n_jobs=-1` to the `HDBSCAN(...)` call.
- **D2** `midia/data.py`: build dense numpy lookup arrays for `cycle_by_frame`/`step_by_frame` once in
  `MidiaExperiment.__init__`, gather with `frame-1` indexing instead of `Series.map(pydict)` over
  millions of rows → **~0.3 s → ~10 ms** per slice-load.
- **D3** midia cleanups: `groupby(...).sum().sum()` → `.sum()`; four `nunique` groupbys → one `.agg`;
  `np.vstack([...]).T` → `np.stack(..., axis=1)`; drop the duplicate `.copy()` in
  `cluster_precursors_hdbscan` (introduced by the earlier empty-guard fix).

---

## Tier 2 — fast-follow (bigger UX / latency wins)
- **Idle render loop** (`web/lib.rs frame()`): a `needs_redraw` dirty flag so a static cloud isn't
  re-splat at 60 fps → idle GPU/CPU → ~0 (subsumes per-frame uniform + DOM-label churn). Medium —
  must wake on every state change.
- **Cache the zstd body** per (region,budget) in `LoadResult` → saves 0.3–0.9 s re-encode per repeat
  `/points`. Medium.
- **Two persistent volume grids + the (dead, already-tested) `VolumeGrid::combine`** for MS-mask
  toggles → O(8N pts) scatter becomes O(voxels). Medium.
- **Peak-grid SoA** in `loader.rs` (48 MB RMW → 4 MB read) + double-buffer decode/fold → fold
  ~1.3–1.8×, load ~1.3×. Low-med / med.

## Tier 3 — post-merge / blocked
- Lighter `rustdf` `get_frame` decode (skip unused `tof_i32`/`scan as i32`, narrow intensity) →
  −~1.2 GB alloc for a 100 M-pt run. Touches shared API.
- `mmap` the `.tdf_bin` (remove per-frame `File::open` + scratch allocs). `unsafe`, shared crate.
- `Arc<Vec<u8>>` to eliminate the *second* 356 MB copy. Broad ripple.
- WebGPU compute-compaction (draw visible, not loaded) — **blocked on the deferred wgpu 23+ upgrade.**

## Confirmed already-optimal (do not touch)
Per-request `/points` serving (zero-copy Arc), the Fisher–Yates shuffle (required for the
prefix-is-uniform contract), the DBSCAN core (squared distances, no re-scans, BFS dedup, hoisted
scales), `fetch_points` bulk cast, incremental `append`, gated `recount_displayed`, FPS-HUD throttle,
Rayon batch endpoint, histogram `batch_update`.
