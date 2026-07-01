# tims-viewer performance — Tier-2 (post-merge round)

Follow-up to Tier-1 (merged in #429). Source: the 5-agent deep-dive. Same loop: plan → claudex →
implement. Behavior-preserving unless flagged. Scope this round: the loader peak-grid SoA (primary)
and the volume MS-toggle `combine` (safe, reuses tested code). Double-buffer is a stretch.

## T2.1 — Peak-grid Struct-of-Arrays (loader.rs) — PRIMARY, biggest remaining win
**Now:** the peak grid is `peaks: Vec<(f32, u64, GpuPoint)>` — ~1M cells × ~48 B ≈ **48 MB**. Every
one of the 100M+ surviving points does `let slot = &mut peaks[peak_cell(pos)]` — a *random* index
into 48 MB (near-guaranteed L2/L3 miss) + a read of `slot.0` (needs only 4 B) and, on a new max, a
44-byte write. This random RMW over a 48 MB array is the dominant fold cost (memory-bound).

**Change:** split hot/cold (SoA):
- `best_i: Vec<f32>` — 1M × 4 B = **4 MB**, cache-resident for the per-point `intensity > best_i[cell]`
  compare (the hot read).
- `best_gi: Vec<u64>` and a compact best-point record `best_pt: Vec<PeakPt>` where
  `PeakPt { pos: [f32;3], flags: u32 }` (16 B) — written **only** on the rare new-max branch.
- Reconstruct the `GpuPoint` at emit time (the peak-tail loop) from `best_pt` (`weight: 1.0`,
  `_pad: [NO_CLUSTER, 0]`) — the grid never needs `weight`/`_pad`.

**Invariants to preserve (claudex will check):**
- Init `best_i` to a sentinel below any real intensity so cell-0 semantics match today's `> slot.0`
  with `slot.0` default 0.0 (use `0.0` init and keep the `intensity > best_i[cell]` compare — a point
  with intensity 0 was never a peak anyway). Confirm against the current default.
- The peak-tail filter uses `gi % stride != 0` and `i >= 0.0` (occupied) — replace `i >= 0.0` occupancy
  with an explicit "was this cell ever written" test (e.g. `best_gi` sentinel `u64::MAX` = empty), since
  `best_i` default 0.0 no longer distinguishes empty from a real 0-intensity peak.
- `select_nth`/emit order unchanged; `region_estimate_concentrates_budget` still passes.

**Gain:** ~1.3–1.8× on the fold phase (48 B RMW → 4 B read per point). **Risk:** medium, local to
loader.rs, test-covered.

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
