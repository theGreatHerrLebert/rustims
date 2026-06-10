# tims-viewer — next steps

Future ideas for the native GPU timsTOF RAW viewer, captured at the end of the
initial build (Phases 1–3 complete on branch `feat/tims-viewer`, PR #406).

## Where it stands today

- **Point cloud**: streaming load, device-budget downsampling with stratified
  peak-preservation, GPU compaction + `draw_indirect` visible-set reduction,
  additive-density and structural-opaque modes.
- **Volume**: trilinear cloud-in-cell density voxelization → R16F 3D texture,
  fullscreen raycaster (composite + MIP), density-domain auto transfer range.
- **Shared**: live RT / m-z / 1-K0 window filtering, MS1/MS2 toggle, transfer
  function (lin/sqrt/log) + colormaps, orbit camera.
- **Annotations**: DDA precursor crosses / DIA isolation-window boxes, clipped
  to the active window.
- **Export**: headless `--render-png` (volume/points, `--ms`, `--annotations`,
  transfer overrides) — works over SSH with no display.
- Verified on a real 1.9 GB HeLa DDA run; 12 tests incl. offscreen-GPU renders;
  6 codex review rounds.

## Candidate next steps

Roughly ordered by value-to-effort.

1. **Live region refinement (region LOD).** When the user narrows the window,
   reload just the in-window frames at higher density (the `RefineRegion`
   command + generation plumbing is already shaped for this in the loader). Turns
   "coarse globally" into "crisp where you're looking."
2. **In-app screenshot button.** The headless `offscreen` path already does the
   render-to-PNG; wire a UI button / `S` key that renders the live view offscreen
   and saves a PNG (and optionally save/restore camera pose).
3. **Axis gizmo + tick labels + colorbar.** Right now it's a bare cube; drawing
   the m/z / 1-K0 / RT axes with real-unit ticks (we keep the `AxisTransform`s)
   and an intensity colorbar would make renders self-explanatory.
4. **Point picking / hover readout.** Click or hover to report the real
   (m/z, 1/K0, RT, intensity) under the cursor — needs a pick pass or CPU
   nearest-search against a spatial index.
5. **DIA showcase.** Render a DIA-PASEF run to exercise the isolation-window box
   overlay end-to-end on real data (only DDA was validated during the build).
6. **GPU compute voxelization + empty-space skipping.** Move trilinear deposition
   to a compute scatter (atomics on a u32 grid) so the volume revoxelizes live as
   filters change, and add an occupancy mip to skip empty bricks in the raycast.
7. **Multi-run overlay / comparison.** Load two runs (e.g. condition A vs B) and
   render them in distinct colormaps in the same cube.
8. **Quality / polish.** Weighted-blended OIT for translucent structural points;
   `compact` f16 point format to double on-GPU capacity; perceptual exposure auto
   from a histogram; configurable volume resolution; save/load view presets.

## Notes / gotchas for whoever picks this up

- Frame ids must be contiguous `1..=N` (rustdf indexes frames by vector
  position; the loader validates this at load).
- timsTOF intensities span ~6 orders of magnitude with a low noise floor — raise
  `i_min` (transfer) to threshold noise out, or renders look like static.
- The dev box is headless; verify with `cargo test -p tims-viewer` (includes real
  offscreen-GPU smokes) and `--render-png ... DEMO` / a `.d` path. Inspect PNGs
  directly. The windowed app needs a display, run it on a workstation.
- Pinned dependency triple: wgpu `=22.1` / winit `=0.30.5` / egui `=0.29.1`
  (+ egui-wgpu/egui-winit). Keep them locked together when bumping.
