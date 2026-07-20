# THERMO M3 — "only 251 DiaNN IDs" investigation

Status: findings for claudex. Astral DIA M3 first run looked alarmingly low; this records what we
measured and the leading explanation, for an independent pressure-test.

## Symptom

First full HCD→Astral→DiaNN run (gen10k feature space → Koina Prosit_2020_intensity_HCD fragments →
timsim-spectra → timsim-render-thermo → PXD061065 single-cell Astral template → TRFP mzML → DiaNN 2.5
library-free):
- DiaNN: **255 precursors at 1% FDR**, of which **251 correct** vs the answer key (FDP 1.57% — well
  calibrated). All 251 correct IDs are "eligible".
- Apparent **conditional recall 251 / 12,542 eligible = 2.0%** — this is what looked wrong.

## Diagnostics measured

1. **The abundance denominator is dominated by ZERO-abundance precursors.** Eligible-precursor
   abundance: min 0, **median 0.0**, max 901 (dynamic range ~9e11). Found precursors sit at the **99th
   percentile** of abundance (found median abundance 0.035 vs missed median 0.0).
2. **Root cause is the sample amounts.** `peptide_quantities` sample `A_R1`: `amount_amol` is **exactly
   0 for 96.0%** of 11,999 peptides (10th/50th/90th pct all 0; 99th = 36.5; max 69,250). Only ~480
   peptides are "present" in this sample. The other abundance factors are all healthy and non-zero:
   `charge_fraction` (med 0.16), `ionization_propensity` (med 0.010), `modform_fraction` (all 1.0).
3. **Detectable set is small by construction.** Eligible precursors (has_ms2 & in_any_window) with
   abundance above a floor: **>1e-3 → 330; >1e-2 → 216; >1e-1 → 132; >1.0 → 65.** DiaNN found 251.
4. **Not a fragment problem.** HCD MS2 fragments/precursor: median 84, min 12, 0% below 6.
5. **Not a modification problem.** Peptides are unmodified (0% carry a `[UNIMOD]`), so the bare-FASTA
   DiaNN search is not mismatched on mods.
6. **DiaNN logged repeated** "Cannot perform mass calibration, too few confidently identified
   precursors" and set a narrow RT window (1.145) — consistent with a very sparse run where DiaNN never
   gets enough anchors to calibrate.

## Leading explanation (to pressure-test)

The "2%" is a **denominator artifact**: the eligibility flags (`has_ms2 & in_any_window`) do NOT encode
whether the precursor is actually **present** (abundance > 0). 96% of the sample is zero-amount, so
12,542 "eligible" includes ~12,000 precursors that can never be detected because they are not in the
sample. Against the **present-and-detectable** set (~330 with abundance > 1e-3), DiaNN's 251 is **~76%
recall** — which would be a healthy number, and the FDP (1.57%) is well-calibrated.

## Open questions for claudex

1. **Is 96%-zero-amount the intended sample, or a quantity-generation issue?** A prior timsTOF DIA run
   (memory: 250K peptides → 18,477 identified precursors) implies FAR more present peptides than 4%.
   Either gen10k's `A_R1` is a deliberately sparse single-sample replicate, or the quantity model /
   sample selection is producing an unintentionally near-empty sample. Which is more likely, and how
   would we tell (e.g. does the multi-sample design concentrate presence into few peptides per
   replicate)?
2. **Is ~76%-of-detectable actually good, or is there a residual limiter** beyond sparsity? Candidates:
   (a) DiaNN can't mass-calibrate on so few anchors, capping recall; (b) the single-cell PXD061065
   template (narrow 396–808 MS1, SW14 wide windows, low-signal SCP method) is a poor match; (c) the
   amol→counts intensity scale (parked calibration) still clips the mid-abundance tail; (d) RT: our
   rt_index→gradient map may not correlate with DiaNN's predicted RT, weakening scoring.
3. **How should the answer-key define the recall denominator for DIA?** Add an `abundance > 0` (present)
   flag? A detectability model (abundance × best-fragment × scale > floor)? The current has_ms2 &
   in_any_window is necessary but not sufficient.
4. **Would a bulk template + a non-sparse sample** (e.g. PXD049028 8-min HeLa nDIA, and a sample with
   realistic presence) be the right way to get a meaningful absolute-ID number, or is there a bug to fix
   first?

## RESOLUTION (post-claudex decisive check)

Codex's #1 test — tabulate presence across all samples — settled it:
- `peptide_quantities.parquet` (the file the render used): BOTH samples `A_R1` and `B_R1` have the
  **exact same 482/11,999 present peptides (4.0%)**; 11,517 peptides are present in NEITHER. So it is not
  a random per-sample dropout — it's a **fixed 482-peptide present subset**.
- **`peptide_quantities_5k.parquet` (sibling file) is 100% present.** A dense quantity file exists.

So the low absolute count is dominated by **using the sparse quantity file** — a 482-peptide sample —
where a dense one was available. Against that sample's ~330 present-and-detectable precursors, DiaNN's
251 correct IDs is ~76% recall, which is healthy. Next experiment: re-render with
`peptide_quantities_5k` (dense) and confirm the absolute ID count rises proportionally.

Open (still worth resolving): WHY does `peptide_quantities.parquet` carry only a fixed 482-peptide
subset — is that an intended small "present proteome" for this generation, or a dropout/selection step
applied where it shouldn't be? And the answer-key denominator should become hierarchical (present +
in-coverage + adequately-sampled + ≥3–4 fragments above SNR), reported as a recall-vs-signal curve, per
codex.

## CONFIRMED (decisive experiment)

Re-rendered the identical pipeline with the DENSE `peptide_quantities_5k.parquet` (all present),
everything else fixed:
- DiaNN IDs 251 → **6,653** (26×); correct 251 → **6,513**; FDP 1.57% → **2.10%** (still well-calibrated);
  **5,137 protein groups** at 1% FDR.
- Recall of the present+detectable set (abundance>1e-3 & has_ms2 & in_window, ~9,444): **56.4%** — a
  realistic DIA recall.
- DiaNN now **mass-calibrated successfully** — confirming the earlier "too few to calibrate" warnings
  were a SYMPTOM of sparsity, not a rendering m/z bug (codex's ranking #4 was right).

Verdict: the low absolute count was entirely the sparse quantity file. The pipeline produces a realistic,
well-calibrated Astral DIA dataset. Residual recall (~56% of detectable) is the parked intensity-scale
+ single-cell-template refinement, per codex's ranked residual factors — a normal tuning axis, not a bug.

## Parked changes — added

- **Hierarchical recall denominator + harness** — `validate/v2_thermo_eval.py` scores DiaNN vs the answer
  key with a nested denominator (all → present(abundance>0) → +in-window → +has-fragments → +above floor)
  and a recall-vs-abundance-decile curve. On the dense `.raw`-direct run it reports FDP 2.11% and, tellingly,
  **recall is FLAT ~53–60% across all abundance deciles** — so intensity is NOT the current limiter. The
  levers are **template m/z coverage** (only 42% of present precursors fall in a DIA window on the
  single-cell PXD061065 template) and **in-window recall ~52%** (RT/sampling/window factors), not signal
  scale. The recall-vs-abundance curve was built precisely to reveal this, and it did.
- **482-subset origin — RESOLVED as a sample-design artifact.** The 482 present peptides come from 411
  proteins of 9,046 (**4.5% expressed**) — a sparse sample by protein expression, not a bug. The dense
  sample is `peptide_quantities_5k` (all expressed). Use the dense one for a bulk benchmark.
- **Intensity calibration — deferred (lab samples), confirmed by the user.** And the flat recall-vs-
  abundance curve independently shows it is not the bottleneck right now, so deferring it costs nothing
  for the current template; a bulk template with wider m/z coverage is the higher-value next lever.

## What is NOT in doubt

The pipeline is correct end-to-end: DiaNN reads the synthetic Astral file, finds the DIA windows, and
its IDs are 98.4% correct against the answer key with well-calibrated FDR. The question is purely why the
ABSOLUTE count is small, and the evidence points hard at sample sparsity + a denominator that counts
absent precursors.
