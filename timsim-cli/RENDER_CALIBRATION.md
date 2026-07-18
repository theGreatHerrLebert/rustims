# Render intensity calibration target

The v2 render must produce per-peak intensity distributions that look like real timsTOF data.
This spec defines that target — measured from real data, **per frame type** — and is the acceptance
criterion for two separate pieces of work: the **signal-level calibration** (intensity shape) and the
**noise model** (peak density). Do not conflate them; the measurements below show they are orthogonal.

## Provenance & method

- **Source:** `K240723` DIA-PASEF reference runs — 24-window, 2640 s gradient, MS2 m/z 400–1000, ~HeLa
  complexity. Three technical replicates (`_002`, `_012`, `_022`).
- **Reading:** raw peaks decoded via `imspy` `get_tims_frame().intensity`, **validated exact** against
  the acquisition's own stored `Frames.NumPeaks / MaxIntensity / SummedIntensities` (peak counts and
  maxima matched to the integer; sums to rounding). So these are the true stored detector counts.
- **Sampling:** ~100–120 frames per type, spread across the gradient; separated by `MsMsType`
  (0 = MS1 precursor, 9 = MS2 fragment). Reproduce with
  `python -m imspy_simulation.timsim.validate.peak_distribution <.d> <n_frames>`.
- **Stability:** floor, median, and peaks/scan are within **±4 % across the 3 replicates** — not a
  one-run quirk.
- **Caveats:** (1) intensities are **signal + noise combined** — we have no method-matched blank, so we
  cannot yet decompose the two. (2) These numbers are **specific to this instrument + acquisition
  method**; other instruments/methods have different floors and densities. (3) The bright tail
  (`p99.9`, `max`) is undersampled relative to the bulk — treat `max` as approximate.

## Target distribution — real timsTOF (per peak)

Raw integer counts. `dyn` = p99.9 / p1. `peaks/scan` = peaks per mobility scan (density).

| frame type | floor | p1 | p25 | p50 (median) | p75 | p90 | p99 | p99.9 | max | dyn | peaks/scan |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **MS1 (precursor)** | **21** | 25 | 39 | **53** | 75 | 104 | 246 | 1,375 | ~60,000 | **55×** | **~335** |
| **MS2 (fragment)**  | **21** | 25 | 50 | **70** | 96 | 128 | 266 | 1,161 | ~8,600  | **46×** | **~24**  |

Key shape facts: a **hard floor at 21** (detector/threshold count), a compact unimodal bump centred at
~50–70 that has **lifted well off the floor** (the mode is above the floor, not at it), and a **narrow
dynamic range** (~50×, not thousands). MS1 and MS2 share nearly the same intensity *shape*; they differ
massively in *density* (MS1 is ~14× denser than MS2).

## Current v2 render (same measurement, `out/250k/v2_250k.d`)

| frame type | floor | p50 | p99 | p99.9 | max | dyn | peaks/scan |
|---|---|---|---|---|---|---|---|
| MS1 | 3 | 10 | 1,394 | 11,559 | ~278,000 | 3,853× | ~11 |
| MS2 | 3 | 6  | 211   | 690    | ~5,200   | 230×   | ~1.9 |

## The gap = two orthogonal levers

**Lever A — intensity SHAPE (signal-level calibration; render-side).**
The rendered per-peak bump sits ~5× too dim, floored ~7× too low, and is **far too wide** (MS1 dynamic
range 3,853× vs 55×). Fixing it is a render + abundance change, not a single scale factor:
- Raise the quantisation floor: `--min-peak-intensity 3 → ~21` (match the detector floor).
- Lift the bulk so the median lands at ~53 (MS1) / ~70 (MS2) instead of ~10 (re-anchor the intensity
  calibration on a robust central statistic, not the single brightest peak).
- **Compress the dynamic range** so the bright tail lands near real's max (~60k MS1 / ~8.6k MS2) instead
  of ~278k — i.e. narrow the upstream abundance spread (the `amol × flyability` log-normals) toward
  real's ~50× per-peak range. A shift alone overshoots the top; it must shift **and** compress.

**Lever B — peak DENSITY (noise model; separate feature).**
Real is **~30× denser in MS1 (335 vs 11 peaks/scan)** and **~13× denser in MS2 (24 vs 1.9)**. That
missing population is the dense low-level floor (real has ~31 % of MS1 peaks within 2× of the floor)
— chemical/background noise plus spectral richness the deterministic render doesn't produce. This is
the **noise model's** job, not a scale factor, and it is deferred to that feature.

## Acceptance criteria

Render a run, probe it with `probe2.py`, and compare **per frame type** to the table above:

**Lever A (calibration) — required now:**
- Floor exactly **21** on both MS1 and MS2.
- Median within **±20 %** of target (MS1 ~53, MS2 ~70).
- `max` / `p99.9` within **~2×** of target (no runaway bright tail).
- Dynamic range (p99.9/p1) within **~2×** of target (MS1 ~55×, MS2 ~46×).

**Lever B (noise) — checked when that feature lands:**
- peaks/scan within **~2×** of target (MS1 ~335, MS2 ~24).

Downstream, the eval harness should show recall-vs-abundance move toward a realistic, **noise-limited**
floor once both levers are in — not the quantisation-limited floor it sees today.

## Order of work

Lever A (shape) and Lever B (noise) interact — a noise floor under too-dim signal buries everything;
under too-bright signal it does nothing. So they are calibrated **together**, but the render-side shape
(floor + bulk + compression) is the concrete first move, with the eval harness and `probe2.py`
distribution check measuring convergence at each step.
