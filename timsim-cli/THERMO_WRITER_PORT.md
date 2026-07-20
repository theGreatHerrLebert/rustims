# THERMO_WRITER_PORT — robust MS1 FTMS-profile authoring in `thermorawfile`

Status: DRAFT for claudex. Prerequisite for `THERMO_PLAN.md` M2 (the DIA render needs to author MS1
precursor-survey profile scans). Target crate: `theGreatHerrLebert/thermorawfile` (rev 9e908ba here;
the writer is self-described WIP — "reader and, soon, writer").

## What we actually found (grounded, not assumed)

Earlier I reported "`author_profile` fails on Astral templates." That was imprecise. Direct diagnostics
on two real Astral templates (PXD049028 8-min HeLa nDIA 2-Th; PXD061065 SW14 single-cell):

1. **`author_profile` WORKS** for in-range, on-grid peaks — authoring 3 grid-derived peaks into scan 1
   with the template's calibration returned `Ok` and round-tripped.
2. The crate's `p6_0_fullrun` example "failed" (`budget_fail = all MS1`) only because its **hardcoded**
   payload includes **m/z 880.5, above this template's grid top (~808)**. `author_profile` rejects the
   ENTIRE call if ANY peak's bin falls outside `[0, nbins)` ("peak m/z falls outside the scan's
   frequency grid"). So it was an out-of-range payload, not a broken writer.
3. **MS1 grids are uniform across a run** (scan 1 / mid / last: identical `first_value`, `step`,
   `nbins`, m/z span). So on these templates the per-scan *grid* does not drift.
4. The crate reads ONE **global** calibration (`calibration_at_event(scantrailer_addr + 4)` = scan 1's)
   and applies it everywhere; the per-scan event-offset table (`scan_event_offsets`) already exists but
   is not used to fetch each scan's own calibration. This is the "per-scan calibration not yet ported"
   note in the source.

So there are **two distinct changes**, and the *practical M2 blocker is #A*, while #B is the accuracy
item flagged:

## Change A — out-of-range peaks must not fail the whole scan (the M2 blocker)

A simulated MS1 survey deposits a broad isotope-envelope spectrum; some peaks inevitably fall outside a
given scan's fixed mass range (e.g. 396–808 or 380–980). Today one such peak errors the whole
`author_profile` call, so the scan can't be authored at all.

**Proposal:** `author_profile` (and `overlay_profile`) **drop** peaks whose bin is outside `[0, nbins)`
(do NOT clip to the edge — clipping fabricates boundary signal / false isotope evidence), author the
in-range remainder, and return an accounting struct:

```rust
pub struct ProfileWriteResult {
    pub written_peaks: usize,
    pub dropped_below_range: usize,
    pub dropped_above_range: usize,
    pub dropped_intensity: f64,   // ion current lost — matters more than peak COUNT
    pub merged_bins: usize,       // collisions summed
    pub saturated_bins: usize,    // clamped to the intensity ceiling
}
```

**Error vs drop (claudex):** a *finite* frequency outside the grid is a normal drop; **hard errors** stay
for degenerate inputs only — non-finite m/z, invalid calibration coefficients, non-finite computed
frequency, invalid grid metadata. **Collisions:** two peaks landing on one bin **sum** in a wider
accumulator, then saturate/clamp to the file's representable intensity type — never wrap (confirm the
current behaviour is sum, not overwrite). **Driver duty:** log per-scan and run-level *dropped-ion-current
fraction* and warn/fail the simulation above a configured threshold — count alone can hide the loss of a
dominant precursor envelope. Caller-side pre-clipping is fine for accounting but must NOT be required for
correctness; the writer's drop-and-account is the safety boundary.

## Change B — per-scan frequency→m/z calibration (the accuracy port)

Even for in-range peaks, using scan 1's calibration for a later scan places the peak at a slightly
wrong m/z if that scan's coefficients drifted (Orbitrap lock-mass recalibration over a run). For DIA at
~5–10 ppm this can matter. The per-scan calibration lives in each scan's event record; the offset table
is already parsed.

**Proposal:**
1. Add `RawFile::calibration_for_scan(scan) -> Option<Calibration>`, reading from
   `scan_event_offsets[scan - first_scan]` (same decoder as `calibration_at_event`, just per-scan).
2. `author_profile` / `overlay_profile` / `repack_profile` use the **scan's own** calibration rather
   than a passed-in global one — either drop the `calib` param and look it up internally (cleanest, but
   a breaking API change), or keep the param and default to per-scan when `None`.
3. Reader parity: profile→m/z conversions (`Profile::mz_of_bin` callers) should use the per-scan
   calibration too, or the crate should offer a convenience that does. Otherwise read-back m/z is
   globally-calibrated while authored m/z is per-scan — an internal inconsistency.

**First: MEASURE the magnitude rigorously** (claudex — coefficient deltas alone are not the metric).
Add `calibration_for_scan`, then for EVERY MS1 scan compute the ppm difference between global- and
per-scan-calibrated m/z at representative points (400/600/800/1000 + each template's grid edges):
`1e6·(mz_global(f) − mz_scan(f)) / mz_scan(f)`. Report max / median / 95th / 99th percentile by scan and
m/z, on multiple Astral methods. "Global calibration is fine" must mean comfortably below the downstream
tolerance across the whole range — not below a single arbitrary max. If it is (as the uniform grids
hint), Change B is low-priority and Change A alone unblocks M2; if it exceeds ~2–3 ppm anywhere
material, do the full port. Don't port blind.

## API shape (claudex-revised)

Do NOT make the primary API `Option<Calibration>` — that makes an important correctness decision implicit
and keeps a confusing external-calibration seam. Prefer a scan-aware primary that looks up the per-scan
calibration internally, plus an explicitly-named escape hatch for expert/testing use:

```rust
pub fn author_profile(&mut self, scan: u32, peaks: &[(f64, f32)]) -> io::Result<ProfileWriteResult>;
pub fn author_profile_with_calibration(&mut self, scan, peaks, calib, /* policy */) -> io::Result<ProfileWriteResult>;
```

This is a breaking change (prevents ordinary callers — including `ThermoRawWriter` — from accidentally
using scan 1's calibration forever). Migration: add the scan-aware method now, keep the old one as a
deprecated shim, and migrate `rustdf::sim::acquisition::ThermoRawWriter::write_scan` (currently passes
`self.calib`) + our `thermo_m0/m1` probes immediately. The seam then stops precomputing one global
`self.calib`. **Reader parity is mandatory if B lands** — else write-side (per-scan) and read-side
(global) m/z disagree and validation is meaningless.

## Test plan

- **A (out-of-range):** author a spectrum mixing in-range + above-top + below-bottom peaks → in-range
  land at correct bins; out-of-range dropped with accurate below/above counts + dropped-intensity; call
  succeeds; read-back shows ONLY the in-range peaks. Full-run authoring with GRID-DERIVED in-range
  payloads → zero drops, zero residual (the corrected `p6_0_fullrun`). Edge cases: empty input and
  all-out-of-range input (explicit: a zero-write result that CLEARS the slot, not one that leaves stale
  template signal); duplicate bins (summed); zero/negative/NaN/inf intensity; saturation clamp;
  scan-type misuse (profile call on a centroid slot).
- **Bin semantics:** nearest-bin vs floor rounding, the negative-`step` frequency direction, endpoint
  inclusion, and m/z within epsilon of either grid edge.
- **Replace vs overlay:** confirm Replace authoring CLEARS the template's real intensities (stale real
  precursor signal in an authored scan is a major realism/correctness bug), and Overlay adds on top.
- **B (per-scan calib):** `calibration_for_scan` returns per-scan coefficients; the ppm-distribution
  measurement above; author one on-grid peak into early/mid/late MS1, read back within tolerance using
  the per-scan calib.
- **Regression:** MS2 centroid authoring unchanged (already validated: M0/M1 all MS2 round-trip). Assert
  **non-payload** metadata unchanged + valid recomputed checksum (NOT whole-file bit-identical — authored
  payload bytes legitimately change; the M0/M1 probes already snapshot only schedule fields, so they hold).
- **Independent reader (M2 gate):** the corrected full-run `.raw` opens in ThermoRawFileParser/msconvert
  with matching scan counts + windows AND correct authored **m/z positions and intensities** (not just
  counts/checksum), through the exact intended TRFP→centroid→DiaNN route.

## Risks / open questions (for claudex)

1. Is skip-and-count the right default, or does silently dropping ion current bias downstream quant?
   (We log the dropped fraction; is that enough?)
2. Does the per-scan calibration actually drift on these Astral files, or is the global one adequate?
   (Change B is gated on measuring this.)
3. Profile peak SHAPE (separate from THERMO_PLAN risk #2): `author_profile` places peaks on grid bins as
   single-bin spikes — structurally fine, but not realistic FTMS peaks (some centroiding/feature
   pipelines mischaracterize spikes: no width, broken isotope-envelope continuity, odd S/N). Keep shape
   OUT of this low-level writer — the RENDER driver deposits a calibrated instrument line shape over
   multiple bins (intensity-conserving, configurable resolution). But do NOT defer VALIDATION: a required
   M2 integration test authors a realistic shape and runs it through the real TRFP/msconvert→centroid
   →DiaNN route before M2 is called complete.
4. Saturation/summing when two authored peaks hit the same bin.
5. The crate is a git dependency of `rustdf`; landing the port means a `rustdf/Cargo.toml` rev bump. Fork
   / branch strategy for `thermorawfile` (it's the user's repo).

## Non-goals

Profile peak-shape modelling (belongs in the driver / THERMO_PLAN risk #2); SCIEX/other vendors; any MS2
centroid changes (that path already works); reader changes beyond the per-scan calibration parity needed
for authoring consistency.
