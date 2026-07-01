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

## Tier 1 — this pass (behavior-preserving, low-risk)

> **Claudex (codex second-opinion) corrections folded in.** Every item verified against the code;
> `[risk]`/`[gap]` annotations are codex's. Two items (C3, D1) were found **not** behavior-preserving
> and moved to their own "behavior-changing" note below. Two Tier-2 items (zstd cache, idle render)
> were promoted — they outrank the micro-cleanups.

### Group A — `serve.rs`
- **A1** body build (`serve.rs:645`): replace `bytemuck::cast_slice(&points).to_vec().into_boxed_slice()`
  → `Arc::<[u8]>::from(bytemuck::cast_slice::<GpuPoint,u8>(&points))` (then `drop(points)`).
  `[risk]` **Do NOT use `cast_vec`** — `bytemuck::allocation` isn't enabled and `cast_vec` needs equal
  alignment (GpuPoint is align-4, not 1). This is **one copy instead of two** (not "in-place"):
  **−356 MB transient + one ~50–100 ms memcpy** per build.

### Group B — `data/loader.rs`
- **B1** fold hot loop: replace the per-point gate `if global_i % stride_u64 == 0` with a countdown
  (init 0 → keep survivor index 0, reset to `stride-1`; `stride==1` keeps all) → removes a division
  over **100 M+ points**. `[risk]` **`global_i` must still increment** — the peak-tail dedup at
  `loader.rs:441` reads `gi % stride`, so that modulo stays (it runs ~1 M×, not per point). Regression
  test `region_estimate_concentrates_budget` (`loader.rs:861`) asserts exact `ceil(survivors/stride)`.
  Gain plausible but decode may dominate.
- **B2** peak-tail (`loader.rs:437`): `select_nth_unstable_by(peak_room, …)` + `truncate` instead of a
  full sort. `[risk]` **guard `if peak_room < peak_pts.len()`** — `select_nth` at `index == len` is OOB.
- ~~**B3** reserve capacity~~ — *demoted*: `collect` and loader chunks already reserve; only `peak_pts`
  and the web `idx/positions` gathers are candidates (fold the latter into C1).

### Group C — `web/src/lib.rs` + `render/volume.rs`
- **C1** clustering boundary: drop the `flat`↔`[f32;3]` rebuilds — `Float32Array::from(cast_slice::
  <[f32;3],f32>(&positions))` on send (`lib.rs:2882,2935`) and `cast_slice(flat)` in the worker
  (`lib.rs:277`) + `with_capacity` on the `idx/positions` gathers. `[risk]` **add `flat.len() % 3 == 0`
  guard** before `cast_slice(flat)` in the exported `cluster_dbscan_flat` — today `chunks_exact`
  truncates malformed input, `cast_slice` would panic. Gain **smaller than −144 MB**: only one backend
  path runs per Run and `Float32Array::from` still copies wasm→JS. Saves the two intermediate rebuilds.
- **C2** `VolumeGrid::density_percentiles` (`volume.rs:125`): `select_nth_unstable_by` instead of a full
  sort of ≤6.3 M voxels. `[risk]` **handle empty first**; use `lo=(len-1)*0.50`, `hi=(len-1)*0.999` to
  reproduce the exact percentile indices; keep the `hi>lo` enforcement.

### Group D — Python (`cluster_service.py`, `midia/*.py`)
- **D2** `midia/data.py`: dense numpy `cycle_by_frame`/`step_by_frame` lookup arrays built once in
  `__init__`, gathered via `frame-1` instead of `Series.map(pydict)` over millions of rows →
  **~0.3 s → ~10 ms** per slice-load. `[risk]` **assert Id contiguity (1..N) before `frame-1`** (Python
  doesn't prove it like `meta.rs:79` does); keep the `int64` dtypes and `fillna(0)` step semantics.
- **D3** midia cleanups (safe, small): `groupby(...).sum().sum()` → `.sum()` (`clustering.py:177`);
  `np.vstack([...]).T` → `np.stack(..., axis=1)` (`clustering.py:51,86,158`); four `nunique` groupbys →
  one `.agg` (`widgets.py:753`); drop the duplicate `.copy()` in `cluster_precursors_hdbscan`
  (`clustering.py:71` vs `:88`, from the earlier empty-guard fix).

### Promoted from Tier 2 by claudex (do alongside Tier 1)
- **P1 — cache the zstd body** per (region,budget) in `LoadResult` → the compressed `/points` is
  re-encoded (0.3–0.9 s) on **every** request today; the raw path is already cached. Medium; biggest
  remote-latency win. **[gap] outranks the micro-cleanups.**
- **P2 — idle render dirty flag** (`web/lib.rs frame()`): a `needs_redraw` so a static cloud isn't
  re-splat at 60 fps → idle GPU/CPU → ~0 (subsumes per-frame uniform + DOM-label churn). Medium; must
  wake on every state change. **[gap] likely a larger user-visible win than B/D micro-items.**

### Behavior-CHANGING (not silent low-risk — decide explicitly, not folded into "preserving")
- **C3** raymarch default `vol_steps` 256 → 128 (`lib.rs:883`, `uniforms.rs:85`): a **quality/perf
  trade**, not test-covered-safe. ~2× volume fragment throughput but visibly coarser. MIP has no clean
  early-out (transfer clamps at `volume.wgsl:51`; only valid when `maxt>=1.0`). Ship only if we accept
  the visual change (or make it a UI default we can bump).
- **D1** `cluster_service.py` **float32** (drop `.astype(float64)`): 144 → 72 MB + faster kNN, **but f32
  distance math can flip labels for points near `eps`** — "numerically acceptable, not exact." Also
  `DBSCAN` already has `n_jobs=-1`; only `HDBSCAN` would gain it — **verify the installed `HDBSCAN`
  accepts `n_jobs`** first. `[gap]` separately, the handler does `bytearray` + `bytes(buf)` before
  numpy (`:74,:87`) — using the mutable buffer directly avoids one full body copy regardless of dtype.

---

## Tier 2 — fast-follow (remaining, after Tier 1 + P1/P2)
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
