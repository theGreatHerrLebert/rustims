# THERMO_PLAN — non-IMS Thermo Astral `.raw` generation from the v2 feature space

Status: DRAFT for claudex review. Companion to `DDA_PLAN.md` (timsTOF DDA, done).

## Goal & why

Prove the v2 architecture's core bet: an **instrument-agnostic feature space** (peptides, RT,
charge, native-m/z spectra, abundance) feeds an **instrument-specific projector/writer**. timsTOF
exercises the 4-D case `(frame, scan, tof)`. Thermo Astral is the clean falsifier — **no ion
mobility**, so it collapses to `(retention time → 1-D m/z spectrum)`. If the feature space is truly
decoupled, a non-IMS render reuses everything except the scan/mobility/TOF-index machinery and
*drops* it (not works around it). Where it can't cleanly drop out is where our abstraction leaks —
a useful finding either way. Target instrument: **Thermo Astral, DIA**.

## Decisions locked (with the user)

1. **Template = a real Astral DIA `.raw`.** The writer is template-based (below); the template fixes
   the scan schedule and the DIA isolation-window scheme. We render the biology onto its slots.
2. **Fresh v2 bin, reuse the writer seam.** New `timsim-cli/src/bin/render_thermo.rs`
   (`timsim-render-thermo`, behind a `thermo` cargo feature) reads the v2 parquet and authors via the
   already-built, tested `rustdf::sim::acquisition::{ScanDescriptor, ThermoRawWriter}`. We do NOT drag
   in the SQLite container model (`rustdf::sim::{handle,containers,dda,dia}`).

## Prerequisite (GATING) — RESOLVED

A real **Astral DIA `.raw`** is the analog of the reference `.d`; the writer overwrites a real file's
scan slots (closed format, no from-scratch). No Astral existed locally (the box has only Orbitrap **DDA**
`.raw` — TUM pool, HELA, PXD001091 — plus Bruker diaPASEF `.d`). Obtained from PRIDE:
- **PXD049028** "The one hour human proteome: Orbitrap Astral" (instrument CV: Orbitrap Astral; DIA).
- Template file: `20230321_OLEP07_JC_8min_DIA_2Th_3p5ms_240k_HAP1_200ng_1.raw` (1.64 GB) — 8-min
  gradient, 2-Th narrow windows, 3.5 ms inject, 240k MS1 res, HAP1 @ 200 ng. →
  `/media/hd02/data/raw/thermo/astral/`.
- Fallback for pure writer de-risk (M0/M1, instrument-agnostic): the local Orbitrap DDA
  `/media/hd02/data/raw/thermo/01625b_GA1-TUM_first_pool_1_01_01-DDA-1h-R2.raw`, and the crate's
  `small2.RAW` (rev-66 Orbitrap, 95 scans).

## Load-bearing risk (claudex): the template must survive rewrite intact

The single highest-risk assumption is NOT CRC/slot-writing — it is that the template's retained scan
metadata, frequency encoding, and DIA schedule survive `save()`/`repack_many` in a form Thermo readers
AND DiaNN interpret **identically to the original method**. `from_template` reads `ScanEvent` before
writing; that is insufficient. M0 (below) reads every schedule field back AFTER writing and diffs it.
De-risk this before any feature rendering.

## The writer seam (already built — `rustdf/src/sim/acquisition.rs` + `thermorawfile` crate)

- `ScanDescriptor { ms_level: u8, retention_time: f64, isolation: Option<IsolationWindow>, peaks: Vec<(f64,f32)> }`
  (`acquisition.rs:24`). `IsolationWindow { center_mz, width_mz, collision_energy }` (`:12`).
  Vendor-neutral, **no IMS axis** by construction.
- `ThermoRawWriter::from_template(template, out)` (`:216`) opens the `.raw` and builds an ordered slot
  manifest `(scan, ms_level, is_profile)` (`:314`). `capacity() -> (ms1, ms2)` (`:289`),
  `slot_count()` (`:296`), `manifest()` (`:314`).
- `write_scan(&ScanDescriptor)` (`:320`): a **single cursor** walks slots in acquisition order —
  ms-level must match the slot (`slot_level_matches`, `:203`); MS1→`author_profile` (binned onto the
  template scan's frequency grid via `Calibration`), MS2→`author_centroids` (m/z-native, no calib).
  Over-budget payloads are deferred and applied in one `repack_many` at finalize.
- `finalize()` saves (recomputes CRC) and enforces the **zero-residual contract**: every slot must be
  authored, else real template signal leaks into the gaps (opt out with `with_allow_partial(true)` for
  smoke tests only). `WriteMode::Replace` (default) vs `Overlay` (real⊕sim, for a future noise model).

**Reference driver:** `rustdf::sim::astral_dispatch::write_astral_raw` (`astral_dispatch.rs:118`) is
exactly this loop — but reads pre-rendered frames from `synthetic_data.db` 1:1 with slots (no RT
mapping). Our driver renders per-slot from the parquet at the slot's own retention time instead.

## What the template provides vs what we render

The template is the acquisition **schedule**; we supply the **signal**. Per slot we need its retention
time and (MS2) its isolation window. `manifest()` gives `(scan, ms_level, is_profile)` but NOT rt /
isolation, so the driver also opens the template via `thermorawfile::RawFile` (read side) to pull, per
slot: `ScanIndexEntry.time` (rt) and, for MS2, `ScanEvent { isolation_center, isolation_width,
collision_energy }` (`scan_event(scan)`). (Alternative: add a `schedule()` accessor to
`ThermoRawWriter` exposing `(rt, Option<IsolationWindow>)` per slot — cleaner; decide in M2.)

## Feature-space reuse vs replace (the flexibility test, made concrete)

REUSE (same loaders as `bin/render.rs`) — with the honest caveat that "instrument-agnostic" means the
m/z *coordinates* are portable, NOT that the intensity/fragmentation model is Thermo-*calibrated*:
- `peptide_rt` → `rt_index`; map onto the template's **effective analytical gradient** via a monotone
  quantile transform (see RT mapping below), not a naive endpoint stretch.
- `ion_spectra` (**both** ms_levels — MS1 precursor isotopes AND MS2 fragments; the timsTOF MS1 bin
  currently loads only level 1) as pure `(m/z, intensity)`. m/z values are native/portable. **BUT the
  MS2 fragment INTENSITIES must be re-predicted (see "Astral fragment predictor" below) — Astral is
  Orbitrap-HCD, not timsTOF-CE; reusing the timsTOF fragment model is physically wrong and would
  mismatch DiaNN's internally-predicted Orbitrap-HCD library in M3.** m/z coordinates port; the
  intensity *model* does not.
- `precursors` (native `mz`, charge, `charge_fraction`, `ionization_propensity`, `modform_fraction`),
  `peptide_quantities` → `abundance = amount · ionz · mff · frac` (identical to timsTOF). Note
  charge-fraction/ionization are sample/source-dependent (not IMS-dependent) — portable first-pass.

REPLACE / DROP (IMS-/Bruker-specific):
- **Drop** `precursor_ccs` entirely (no mobility), the CCS→1/K₀→scan placement (`place_scan`), and the
  mobility Gaussian / scan-window inner loop.
- **Drop** m/z→TOF-index conversion; carry **native m/z** into `ScanDescriptor.peaks`.
- **Replace** `TdfWriter`/`.d` with `ThermoRawWriter`/`.raw`.

## Driver design (`render_thermo.rs`)

1. Open template → `ThermoRawWriter::from_template` + read-side schedule (rt + MS2 isolation per slot).
   Preflight: MS-level sequence of the manifest is the schedule we must satisfy.
2. Load parquet feature space (as above). Build per-precursor: apex rt (mapped, below), abundance, MS1
   peaks, MS2 peaks (native m/z). No mobility.
3. Walk the manifest **in slot order**; for each slot at its own template retention time `t`:
   - **MS1 (profile):** deposit every precursor's MS1 isotope peaks scaled by `abundance · elution(t)`;
     sum co-incident m/z. **MS1 peaks must be authored as instrument-consistent profile PEAK SHAPES
     (not delta sticks)** — see risk #2; M0 determines whether `author_profile` spreads a stick or we
     must pre-convolve to the template's resolution. Emit `ScanDescriptor{ms_level:1, rt:t,
     isolation:None, peaks}`.
   - **MS2 (DIA centroid):** for the slot's isolation window `[center ± width/2]`, deposit the MS2
     fragments of every precursor whose **isotope m/z** falls in the window (a quadrupole transmits by
     actual isotope m/z; first pass may approximate by monoisotopic-in-window — decide in M2), scaled by
     `abundance · elution(t)`; **sum identical/near-identical fragment m/z across co-isolated
     precursors**. Co-isolation of multiple precursors per window is realistic and KEPT (DIA windows are
     wide — that's the point). Emit `ScanDescriptor{ms_level:2, rt:t, isolation:Some(window), peaks}`.
     Empty window ⇒ empty spectrum (never residual template signal). Quadrupole edge transmission and
     fragment-elution correlation deferred.
   - `write_scan`. **Only data-bearing MS1/MS2 acquisition slots are authored** — preflight FAILS on any
     unsupported scan-event form (MS3, lock-mass/calibration, MSX/staggered/overlapping windows,
     polarity change), rather than manufacturing spectra for them.
4. `finalize` (Replace, full coverage). Emit a **sidecar answer key** parquet: per precursor present —
   `peptide_id`, `sequence`/proforma, `charge`, precursor `mz`, apex `rt_seconds`, `abundance` — a
   richer join key than seq+charge+mass (add proforma + precursor m/z±tol + apex RT to disambiguate
   modified/isobaric peptides).

### Astral fragment predictor (REQUIRED feature-space swap for M2/M3 — user-flagged, load-bearing)

The MS2 fragment intensities in `ion_spectra` come from the timsTOF Prosit model (`Prosit2023TimsTof`).
Astral is an **Orbitrap-HCD analyzer** — different collision-energy regime and fragmentation, so the
fragment intensity *pattern* differs. The Astral track must **regenerate `ion_spectra` (level 2) with an
Orbitrap/HCD fragment intensity predictor** (candidate: Prosit 2020 HCD via Koina, or an AlphaPeptDeep
Orbitrap model) at the template's CE, *before* M2's realism render. Precursor m/z, RT, charge, abundance
are unaffected — only the fragment intensity model swaps. This does NOT block M0/M1 (writer de-risk is
instrument-agnostic); an M2 render on timsTOF intensities is a **mechanics** test only — the **realism**
render (and M3 scoring) requires the swapped predictor. Open: confirm the Koina/Prosit-HCD path in
`imspy-predictors` and the CE mapping (template CE 14.9–32.3 → the predictor's NCE input).

### RT mapping (biggest biological risk — claudex)

Do NOT map `rt_index` linearly onto the template's full duration (piles peptides into loading/wash
regions, distorts density across differing gradients). Instead:
- Extract the template's **effective analytical gradient** `[g_start, g_end]` (exclude pre-gradient
  equilibration and wash) from its metadata or a supplied config.
- Map `rt_index` through its **empirical quantiles** (monotone) into `[g_start, g_end]`.
- Scale chromatographic peak width in **physical seconds** with the gradient duration; enforce a minimum
  width / cycle-sampling floor (a peak must span enough MS1 cycles to be found).
- **Clip or resample** peptides whose apex can't fit a full peak inside the analytical region — never
  silently pile at edges.
- Better (later): fit a monotone map from `rt_index` to observed RTs of a real Astral run/library.

## Validation (format-semantics FIRST, then the DIA harness)

A successful DiaNN search does NOT alone prove format/mass-axis correctness (claudex). Two layers:

**(a) Format semantics, independent of any search engine** — the primary correctness gate:
- Post-write reader round-trip vs the template: scan count/order, per-slot RT, MS levels, profile/
  centroid flags, filter strings, isolation center/width/CE, polarity/analyzer — all unchanged except
  the authored payloads.
- Independent conversion (ThermoRawFileParser or msconvert) → compare spectra/windows/RTs.
- **Synthetic sentinels** in several early/mid/late cycles and windows to detect any cursor shift /
  off-by-one in slot authoring.
- MS1 re-centroid round-trip: median/95th-pct ppm error, FWHM/resolution, isotope-ratio preservation.
- Negative controls: absent peptides + a decoy library + window-empty scans, to establish false
  positives independent of the search.

**(b) Biological validity — reuse the existing DIA harness.** The flexibility payoff: the **same**
feature space and the **same** eval harness, a different instrument.
- **DiaNN** reads `.raw` natively → score with `validate/v2_eval.py` (recall-by-abundance, RT/quant
  correlation, FDP) as for timsTOF DIA. Confirm nothing timsTOF-specific (mobility columns) leaks into
  its report↔answer-key join.
- Or convert `.raw`→mzML and use Sage/DiaNN. Decide by tool availability in M3.
Success = DiaNN identifies our peptides with a well-calibrated FDP and sane recall-by-abundance — the
non-IMS path as trustworthy as the timsTOF path.

## Milestones

- **M0 — template characterization & rewrite-survival (load-bearing, FIRST).** With a real Astral DIA
  template: enumerate and classify EVERY scan-event form (MS1/MS2/other); extract the effective gradient
  bounds; measure the MS1 frequency-grid spacing and profile/centroid packet budgets; determine what
  `author_profile` actually does to a peak (spread-to-shape vs stick). Then author a trivial payload,
  `save`, RE-OPEN, and diff every retained schedule field (RTs, filters, windows, CE, flags) against the
  original + `checksum_valid`. Gate: the template survives rewrite intact and unsupported forms are
  detectable. Deliverable also answers risks #2/#4 empirically.
- **M1 — writer smoke + sentinels.** Author distinctive known payloads into a handful of slots
  (`with_allow_partial`), `finalize`, re-open, verify checksum + authored peaks round-trip AND the
  sentinel positions land in the right slots (no cursor drift). Independent-reader (TRFP/msconvert)
  round-trip on the smoke file.
- **M2 — parquet → `.raw` DIA render.** The full driver: quantile RT map onto the analytical gradient,
  render MS1(profile-shaped)+MS2(centroid) from the parquet, author every data-bearing slot, `finalize`
  a complete valid `.raw` + the answer key. Verify capacity holds AFTER co-isolation (repack only if the
  measured budget is exceeded) and windows/RTs match the template.
- **M3 — search + score.** DiaNN (or Sage-via-mzML) on the `.raw`; wire the answer key into
  `v2_eval.py`; report recall-by-abundance + FDP + RT/quant correlation, plus the negative-control false
  positive rate. The flexibility result.

## Risks & open questions (for claudex)

1. **RT mapping** (biggest biological risk — approach now specified above: quantile map onto the
   effective analytical gradient, widths in seconds, clip non-fitting). VERIFY in M2 that peptide apex
   density is realistic across the gradient, not piled at edges. Template slot count also bounds run size.
2. **MS1 profile SHAPE, not just binning.** m/z accuracy from grid-binning is fine (and Astral ID is
   MS2-driven); the risk is authoring delta sticks into a profile array (unnatural centroiding/S:N). M0
   determines whether `author_profile` spreads to an instrument peak shape or we must pre-convolve to the
   template resolution + sample the isotope envelope. Test via re-centroid ppm/FWHM/isotope-ratio.
   `repack_profile` addresses capacity only, not shape — use only if the measured budget overflows.
3. **Intensity calibration.** Same open thread as timsTOF — the `.raw`'s real per-peak intensity scale
   is unknown until we characterize a real Astral run. First pass: an `--intensity-scale` knob; defer
   true calibration (parked `CALIBRATION_PLAN.md`).
4. **DIA window scheme fidelity.** We inherit the template's exact isolation windows. Do our precursor
   m/z values land sensibly across them? Any window with zero eligible precursors is authored empty
   (fine) — but is the window map what DiaNN expects (it reads it from the file)?
5. **Answer-key / locator join for DIA.** DIA is precursor-centric (no per-spectrum locator like DDA);
   `v2_eval` already joins DiaNN report ↔ answer key by sequence+charge+mass. Confirm nothing timsTOF-
   specific (mobility columns) leaks into that join.
6. **`thermo` feature build surface.** Enabling `rustdf/thermo` in `timsim-cli` pulls the
   `thermorawfile` git dep; keep it opt-in so default builds are unaffected (like `tdf`).

## Non-goals (deferred — named, not silently dropped)

Quadrupole edge-transmission weighting; isotope-**resolved** window transmission (first pass approximates
by isotope/monoisotopic-in-window); fragment-elution correlation across co-isolated precursors; realistic
Astral noise (Overlay mode + a noise model); SCIEX `.wiff`; DDA-on-Thermo; charge-state
authoring (the template carries it read-only — same "read from file" limitation accepted for timsTOF DDA);
support for MS3/MSX/staggered/overlapping-window templates (preflight rejects them for now).
