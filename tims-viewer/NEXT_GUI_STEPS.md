# tims-viewer GUI upgrade — Plan A+C

Goal: turn the prototype from "a glowing cloud in an unlabeled box with every control
thrown at you" into a readable scientific instrument that looks right the moment data lands.
Two tracks: **A — make it a real plot** (readable axes + legends), **C — smart defaults**
(it just works). Plus the chosen panel reorg (collapsible, filters unified).

## A — Instrument the cube

### A1. Axis ticks + numeric labels  *(highest clarity-per-effort)*
- For each axis (x=m/z, y=1/K0, z=RT) place ~5 ticks at **"nice" round steps** spanning the
  visible range (full bounds, or the window range when focus is on — mirror `draw_axis_labels`).
- Nice-step algorithm: pick step ∈ {1,2,5}·10^k so 4–6 ticks cover the span; first tick at
  `ceil(lo/step)*step`.
- Per tick: a short 3D tick segment just outside the cube edge (reuse the line renderer) + a
  numeric label projected to screen (reuse `draw_axis_labels`'s world→screen projection).
- Formatting per axis: m/z integer, 1/K0 2 decimals, RT integer seconds (or min if range large).
- Must re-fit under focus/zoom (use window range when `state.focus`).

### A2. Faint back-face gridlines  *(optional, lower priority)*
- Light lines connecting opposite ticks across the two far cube faces for depth reference.
  Subtle (low alpha / dim color); skip on the near faces to avoid clutter over the cloud.

### A3. Intensity colorbar
- Vertical colorbar showing the active colormap gradient with `i_min`/`i_max` end labels and
  the transfer mode (lin/sqrt/log). egui-drawn, in the panel or a screen corner.

### A4. Group-color legend
- When selection windows are on, a compact legend: group number → swatch (same `group_color`
  as the boxes and the checkboxes). Lets the colored diagonals be read.

## C — Smart defaults (it just works)

### C1. Auto-exposure / transfer on load
- On load-complete (and on MS toggle), set `i_min`/`i_max`/`exposure` from the data so the
  cloud is **never blown out** on first paint (kills the white-out on noiseless data).
  - Points: robust p1/p99 of retained intensity (already computed) + choose exposure so the
    median maps to mid-range under the active transfer.
  - Volume: density percentiles (already computed).
- Acceptance: opening MIDIA-250K-NOISELESS-FRAG renders a legible viridis/inferno cloud with
  no manual slider tweaking.

### C2. Good default camera + point size
- Verify the default orbit angle frames the cube nicely; keep the existing "Reset".
- Default point size scaled to resident count (denser → smaller) so it reads at any budget.

### C3. Default render mode
- Decide: keep Points as default, or open in Volume/MIP as the "hero" view. (Lean: Points for
  responsiveness, but auto-exposed so it's not white.)

## Panel reorg (chosen: collapsible, filters unified)
egui `CollapsingHeader` sections; readouts always visible at top:
- **Filters** (default open): MS1/MS2 · RT/m-z/1-K0 range sliders · Focus to window · Reset.
- **Selection windows**: show toggle · per-group colored checkboxes · all/none.
- **Rendering**: Points/Volume · Additive/Opaque · Composite/MIP · transfer · colormap ·
  point size/opacity.
- **View / Camera**: axis frame · ortho · m/z|mob|RT snaps.

## Out of scope here (later: Plan B — explorable)
Hover/click readout (m/z,1/K0,RT,intensity), linked 2D projections, crosshair probe.

## Acceptance checklist
- [ ] Ticks with values on all 3 axes, correct under focus/zoom.
- [ ] Intensity colorbar with end values + transfer mode.
- [ ] Group legend when windows on.
- [ ] Default load is legible (not white) with zero slider tweaks.
- [ ] Panel is collapsible with all data filters under one "Filters" section.
- [ ] `cargo build --release` clean, `cargo test` green, headless render still works.
