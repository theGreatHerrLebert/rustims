# Render intensity calibration — target, model, and what NOT to touch

The v2 render must produce data that *looks* like real timsTOF. The naive reading of that — "reshape
the rendered intensities to match the real per-peak distribution" — is a **category error** that would
damage the simulator. This document states the corrected framing, the observation model to build, and
the acceptance criteria, after a domain review (Codex, 2026-07-19) caught the error below.

> **The error to avoid:** the real per-peak intensity distribution is *narrow* (~55× dynamic range),
> but that is **not** because real peptide abundances are narrow. It is a property of *measurement* —
> one peptide's ions spread across RT / mobility / isotope / fragment bins, and a hard count floor
> censors the low end. Narrowing the render's **biological abundance** distribution to hit ~55× would
> erase the ~6-order ground-truth abundance axis the whole eval harness exists to test
> (recall-vs-abundance). **Never compress the abundance axis to match a per-peak statistic.**

## Two axes — keep them separate

| axis | what it is | for the render |
|---|---|---|
| **Biological abundance** | `amount × ionization × modform × charge` — the ground truth, ~6 orders | **Held FIXED. Wide. Never calibrated to a per-peak target.** |
| **Measurement / observation** | how an ion's abundance becomes stored per-peak counts: spreading, floor/censoring, count noise, background | **This is what we calibrate.** The narrow per-peak shape is an *emergent output* of this layer, not an input target. |

So "signal calibration" and "noise model" are **not two sequential features** — they are **one instrument
observation model**, built together, because their effects are statistically coupled (adding the missing
near-floor population lowers the pooled median and changes the apparent dynamic range — see below).

## What real data shows — as *combined observables* (not decompositions)

Measured on real `K240723` DIA-PASEF (24-window, 2640 s, m/z 400–1000; 3 replicates, ±4% stable; raw
peaks validated exact vs stored `NumPeaks/MaxIntensity/SummedIntensities`). Reproduce:
`python -m imspy_simulation.timsim.validate.peak_distribution <.d> <n_frames>`.

| per-peak (real) | floor | p50 | p99 | p99.9 | max | dyn (p99.9/p1) | peaks/scan |
|---|---|---|---|---|---|---|---|
| **MS1 precursor** | **21** | 53 | 246 | 1,375 | ~60,000 | ~55× | **~335** |
| **MS2 fragment**  | **21** | 70 | 266 | 1,161 | ~8,600  | ~46× | **~24**  |

**These are combined signal + isotopes + co-elution + background + thresholded noise, pooled over all
retained bins.** They are therefore **not** estimates of peptide abundance and **not** a detector
transfer function. Read them only as: "the observation model, run on a HeLa-like load at this method,
must produce roughly this combined distribution per frame type." Sample- and method-specific.

Current render (same probe, `out/250k/v2_250k.d`): MS1 floor 3 / median 10 / ~11 peaks/scan;
MS2 floor 3 / median 6 / ~1.9 peaks/scan — i.e. too dim in the bulk, floored too low, and **~30× (MS1)
/ ~13× (MS2) too sparse**. The sparsity is the missing measurement layer, not missing abundance.

## The observation model to build (abundance held fixed)

Applied *after* ion generation, per frame type, ideally conditioned on m/z / mobility / gradient position:

1. **Signal spreading** — an ion's current over RT (frames) and mobility (scans) Gaussians, and its
   isotope/fragment structure. Partly present already; it is a large part of why per-peak intensity is
   far below total ion abundance.
2. **Count floor / censoring** at the detector threshold (~21). Implement as a *real* floor/censor, not
   only the current post-quantisation drop cutoff (`--min-peak-intensity`).
3. **Ion-count noise** — shot/counting statistics on the (small) per-bin counts.
4. **Background process** — the dense low-level population that fills the ~30× density gap, conditioned
   on frame type / m/z / mobility / gradient region. **This needs a method-matched blank to measure**
   (see the sample request); do not assume it is "just noise" — it may include real low-level analyte.
5. **Signal→response transfer** (nonlinearity / saturation) — **only if a dilution series shows it.**
   The current evidence does *not* demonstrate saturation (real maxima ~60k/8.6k are not obviously
   clipped; the `u32` ceiling is irrelevant to instrument saturation). Default to linear response until
   data says otherwise.

## Acceptance criteria (corrected)

**Primary — truth preservation (must hold):**
- Recall-vs-**unchanged** abundance still spans the full range (the abundance axis was not compressed).
- Response curve for identified / spiked precursors is monotonic and linear where the real data is.
- Feature-level isotope/envelope intensities keep their true ratios.

**Hard compatibility check:**
- Per-peak floor is exactly **21** on MS1 and MS2 (an instrument/method threshold, verified in blanks).

**Secondary — emergent-shape diagnostics (regression checks, NOT primary objectives):**
- After the full observation model, the pooled per-peak median / density / dynamic range land within
  tolerance of real, **stratified by frame type and gradient region**, with **analyte and background
  peaks compared separately** (using the blank).
- These are explicitly *joint post-model* diagnostics. They can "pass for the wrong reasons" (a high
  cutoff + injected floor noise matches floor/median/density while destroying abundance-response
  fidelity), so they gate nothing on their own — truth preservation does.

## What we can and cannot do before the new calibration samples

Without a **method-matched blank** we cannot separate background from signal — only match the *combined*
distribution, which risks looking-right-for-wrong-reasons. Without a **dilution series** we cannot fit
or verify the response curve. Until those exist, the honest scope is: keep abundance fixed; add
spreading + a real floor/censor + count noise + a *provisional* background fit to the combined real
distribution; label the response linear; and treat every shape diagnostic as provisional. The clean fit
comes from the samples specified in `CALIBRATION_SAMPLE_REQUEST.md`.
