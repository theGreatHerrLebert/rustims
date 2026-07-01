# tims-viewer performance — Tier-2 (post-merge round)

Follow-up to Tier-1 (merged in #429). Source: the 5-agent deep-dive. Same loop: plan → claudex →
implement. Behavior-preserving unless flagged. Scope this round: the loader peak-grid SoA (primary)
and the volume MS-toggle `combine` (safe, reuses tested code). Double-buffer is a stretch.

> **Claudex (codex) corrections folded in.** My original occupancy premise was wrong (sentinel is
> `-1.0`, not `0.0`). T2.1 revised to the *safer* SoA (split hot `best_i` only; keep the cold
> `GpuPoint` — no reconstruction). T2.2 **deferred** this round (intensity-floor rebuild + web-deposits-
> peaks ambiguity make "behavior-preserving" non-trivial; it's a niche MS-toggle path).

## T2.1 — Peak-grid Struct-of-Arrays (loader.rs) — PRIMARY, biggest remaining win
**Now:** the peak grid is `peaks: Vec<(f32, u64, GpuPoint)>` — ~1M cells × ~48 B ≈ **48 MB**. Every
one of the 100M+ surviving points does `let slot = &mut peaks[peak_cell(pos)]` — a *random* index
into 48 MB (near-guaranteed L2/L3 miss) + a read of `slot.0` (needs only 4 B) and, on a new max, a
44-byte write. This random RMW over a 48 MB array is the dominant fold cost (memory-bound).

**Change (SAFE version — no GpuPoint reconstruction):** split only the *hot* intensity out of the
current `peaks: Vec<(f32, u64, GpuPoint)>` (loader.rs:224):
- `best_i: Vec<f32>` init **`-1.0`** (exactly today's tuple `.0` default) — the hot per-point read for
  `intensity > best_i[cell]`. 1M × 4 B = **4 MB**, so the compare stops pulling a 48 B cache line of
  cold data.
- `best_gi: Vec<u64>` + `best_gp: Vec<GpuPoint>` — the cold record, written **only** on the new-max
  branch (same fields stored as today; NO reconstruction).

**Invariants (codex-flagged — verify all):**
- `best_i` init `-1.0` (NOT 0.0) — with strict `intensity > best_i[cell]`, a real 0-intensity point
  still occupies an empty cell (`0.0 > -1.0`), matching today (loader.rs:351).
- Keep strict `>` (first-wins on ties); `>=` would flip to last-wins (loader.rs:352).
- Peak-tail (loader.rs:445): gate **occupancy first** — `best_i[cell] >= 0.0` (== today's `i >= 0.0`)
  **then** `best_gi[cell] % stride != 0`. Never run the modulo on an empty cell.
- `global_i` still assigned-before-increment, survivor-relative (loader.rs:337-380); emit order + the
  `region_estimate_concentrates_budget` test unchanged. (Note: that test only guards the systematic
  count — peak occupancy/flags/zero-intensity/dedup are NOT covered, so review carefully.)

**Gain:** a fold speedup — treat "~1.3–1.8×" as a **hypothesis to benchmark**, not a given: access is
slab-local (RT frame order), and the fold also does normalization/binning/histograms/systematic pushes.
**Risk:** low–med, local to loader.rs.

## T2.2 — Volume MS-toggle via `VolumeGrid::combine` (volume.rs + web/lib.rs) — SAFE, niche
**Now:** an MS1/MS2/both toggle sets `vol_needs_grid`, and `ensure_volume` re-deposits **all**
`cpu_points` (trilinear scatter, ~8 voxel adds/point) — O(8N points) on every toggle.

**Change:** deposit two persistent grids (MS1, MS2) once at load; an MS-toggle calls the
already-written-and-tested `VolumeGrid::combine(a, b, wa, wb)` (dead in the web path today) — one
O(voxels) pass with 0/1 weights. Extra cost: ~2× grid RAM (~50 MB f32) + one extra deposit at load.

**Gain:** MS-toggle-in-volume-mode drops from O(8N) scatter to O(voxels). **Niche** (only that path).
**Risk:** medium; `combine` is tested, but wiring two grids + the toggle path needs care. Skippable if
the effort/benefit doesn't hold up under claudex.

## Stretch — double-buffer decode/fold (loader.rs)
Overlap batch N+1 decode (rayon) with batch N fold. ~1.3× load. **Higher risk** (concurrency +
`global_i` prefix ordering the test pins exactly). Defer unless T2.1 lands cleanly and there's appetite.

## Out of scope this round
Behavior-changing C3 (raymarch 256→128) / D1 (f32 clustering); rustdf decode slimming + mmap (shared
crate); `Arc<Vec<u8>>`; WebGPU compaction (blocked on wgpu 23+).
