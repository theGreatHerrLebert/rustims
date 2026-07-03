# LSH Spectral-Match Index — Hand-off

Pick-up doc for continuing on a bigger machine. Full design rationale is in
`TO_LSH_SCAN_INDEX.md` (same branch). This file is the operational summary.

## TL;DR
- Branch: **`feat/lsh-spectral-index`** (pushed to origin). Base `main`.
- A sparse LSH index over timsTOF spectra. Two use cases, and they want
  **different tools** (established empirically):
  - **Peptide-ID / membership** (clean query bag vs chimeric data unit): the
    **inverted feature index** wins (containment-native); cosine-LSH is weak.
  - **Spectrum recurrence / MBR / clustering** (data unit ↔ data unit): **cosine
    SimHash** is the right tool — this is the current focus.
- Core is done and tested: `LshScheme` trait + `CosineSimHash` + `MinHash`
  (`mscore/src/algorithm/lsh/`), mobility-window driver `mscore/src/timstof/lsh.rs`,
  three `rustdf/examples/` harnesses. All Rust tests green; Codex-reviewed.

## Build / environment
- Rust 1.84+. `cargo build --release -p rustdf` (pulls mscore).
- **Calibration:** the default reader now falls back SDK→Calibrated→**Bruker-formula
  (SDK-free, accurate)**→Simple. So macOS gets accurate m/z without the SDK; a
  Linux box with the Bruker SDK installed uses it directly. No code change needed.

## Data (copy to the box; rsync `~/Promotion/midia/timsim/`)
- `DIA-250K-NOISY-CHRONO/` — sim `.d` + `synthetic_data.db` (exact ground truth).
  32,020 MS2 frames. **Primary dataset.**
- `MIDIA-250K-NOISY-CHRONO-FRAG/` — DIFFERENT sample + acquisition (MIDIA). NOT an
  A/B pair with DIA-250K (different peptides, cross-scheme).
- `O240206_003_S1-B2_1_15479.d` — REAL run + DiaNN results (`_dia-NN.tsv`) for
  future S2/S3 orthogonal labels.

## Run
```bash
D=/path/to/DIA-250K-NOISY-CHRONO
# self-test + scale (parallel, timed, extrapolated). "mem" loads .d into RAM.
cargo run --release --example recurrence_selftest -p rustdf -- $D/DIA-250K-NOISY-CHRONO.d 40
cargo run --release --example recurrence_selftest -p rustdf -- $D/DIA-250K-NOISY-CHRONO.d 32020 mem   # FULL build

# ground-truth recall (unit->ion labels from the DB): the real recall number.
cargo run --release --example recurrence_recall -p rustdf -- \
    $D/DIA-250K-NOISY-CHRONO.d $D/synthetic_data.db 2000
```

## Findings so far
- **Full-dataset build is feasible:** ~3.1M units, ~4.2 GB, ~27 min single-machine
  at (64,32) — **hashing (Gaussian probit) is 99%** of it → ~2–3 min on 128 cores,
  or use a probit table/ziggurat/Rademacher for a 10–20× cut.
- **Bands (from the paper, PMC9301846):** AND-of-OR `(m,n)` = m OR-bands × n AND-bits,
  candidate = ≥1 band matches. Larger n = sharper cosine threshold, collapses
  candidate volume (n=10→96/q, (64,32)→2.4/q). Random collisions ~ m·N/2ⁿ → ~0 once
  n>log₂N≈22, so volume is scale-invariant. **The `(m,n)` is use-case-specific.**
- **Mobility gate is load-bearing** (`|Δcenter_scan|≤5`): ~24× candidate cut on real
  data at any banding. "LSH proposes, physics disposes."
- **Ground-truth recall (200-frame slice):** 72.9% of units labeled; ~7.9 ions/unit
  (chimeric). True same-ion cross-frame pairs: **median cosine 0.0**, ~31% ≥0.7 —
  "shares an ion" ≠ "similar spectrum" (shared ion often a minor component). LSH
  recall ~0.30 (32,16) / 0.20 (64,32) at **80–91% precision** → it recovers the
  genuinely-similar pairs. **(32,16) fits recurrence; (64,32) is too strict** (it's
  for near-duplicate denoising).
- Representation: mobility-window units are dense (~150 peaks) → **top-N=25 peak
  picking** makes them spectrum-like. m/z map = log-ppm bins (bin 2 ppm) + tol-width
  triangular splat (tol 6 ppm). Intensity `sqrt` + L2 norm.

## Next steps (prioritized)
1. **Bigger recall run:** `recurrence_recall … 2000+` on the box — the 200-frame
   histogram had only n=177 pairs. Get a robust cosine distribution + recall.
2. **Stricter labels:** define a unit's label by its **dominant** ion (by
   `ions.scan_abundance` / `peptides.frame_abundance`), not "shares any ion". This
   should raise true-pair cosines and give a cleaner recall/(m,n) answer.
3. **Full-dataset build** (`… 32020 mem`) — confirm the ~3 min / ~4 GB projection on
   128 threads; watch memory.
4. **Cross-run A/B (true MBR):** SIM a *small, same-scheme* TimSim replicate pair
   (same peptides+abundances, 2 seeds, short gradient — NOT another 250K monster),
   then index A / query B in `recurrence_recall` (generalize it to two datasets).
5. **Verify the calibration change:** run the Python/imspy test suite — the
   default-fallback change shifts m/z/mobility for every non-SDK caller (accurate,
   but a behavior change). This is the one thing not verifiable in pure Rust here.
6. **Perf (if needed):** swap the Gaussian probit for a lookup table/ziggurat.
7. **Inverted index at scale:** for the *membership* use case, re-check the inverted
   index's candidate volume on real clustered m/z + the mobility gate.

## Gotchas
- **Calibration blast radius:** non-SDK callers (imspy on macOS, etc.) now get
  different (accurate) m/z. Fine, but golden-value tests may move. See step 5.
- **Band packing:** `CosineSimHash` packs n bits into one u64 → **n≤64**. n=128
  needs the band key to be a *hash* of the bits (small change).
- **Chimericity:** same-ion ≠ similar spectrum (median true-pair cosine 0.0). The
  recurrence use case only recovers the ~30% dominant-shared pairs.
- **Lazy `get_slice` is serial;** pass `mem` (in-memory loader) for parallel reads.
- **`MzFeatureMap` params & top-N are tuned by eye,** not swept — Phase 1.5 knobs.

## Commit trail (branch, newest last)
Phase 0 SimHash → Phase 1 driver → MinHash+spike → top-N+self-test → band sweep →
Codex fixes → default Bruker-formula calibration → ground-truth recall harness.

---

## Scale-run update (monster3, 120 threads — 2026-07-03)

Runs on the beefier box. **Two handoff expectations were corrected — see below.**

### Full-dataset build IS feasible (task 3, `... 32020 mem`)
- **3,061,009 units** (96/frame, 117 feat/unit after top-25).
- `mem` (in-memory, SDK-free BrukerFormula) reads all 32020 frames in **1.03 s**
  (parallel; no SDK lock) — confirms the "go SDK-free for PAR reads" call.
- Build (hash, (64,32)) **232.75 s** (~3.9 min), the 99%-dominant cost, as projected.
- **Peak RSS ~35 GB** (not the ~4 GB "feature store" figure — that was the store
  only; peak holds raw .d + decoded frames + units + signatures at once). Fine on a
  big box; size the machine for ~35 GB, not 4.

### Bigger recall run (task 2, n=2000 frames, 194,071 units)
Robust now (n=2382-4513 true pairs vs 177 before). **Two labels A/B (task 4):**

TRUE cross-frame same-ion pair cosine — **median 0.000 under BOTH labels**:
| label | n | p75 | p90 | frac>=0.7 | frac>=0.9 |
|-------|---|-----|-----|-----------|-----------|
| shares-any-ion | 2382 | 0.624 | 0.900 | 0.22 | 0.10 |
| dominant-ion   | 4513 | 0.612 | 0.921 | 0.23 | 0.12 |

**CORRECTION 1 — the dominant-ion relabel (handoff step 2) did NOT help.** It barely
moved the true-pair cosine tail (frac>=0.7 0.23 vs 0.22) and *collapsed* the ion-label
precision (below). Reason: in ~8-ion chimeric windows the abundance-argmax ion is
frame-unstable, so genuinely-similar recurrences often get *different* dominant labels.
The bottleneck is chimericity, not the label — a better label cant fix it.

---

## Scale-run update (monster3, 120 threads — 2026-07-03)

Runs on the beefier box. **Two handoff expectations were corrected — see below.**

### Full-dataset build IS feasible (task 3, `... 32020 mem`)
- **3,061,009 units** (96/frame, 117 feat/unit after top-25).
- `mem` (in-memory, SDK-free BrukerFormula) reads all 32020 frames in **1.03 s**
  (parallel; no SDK lock) — confirms the "go SDK-free for PAR reads" call.
- Build (hash, (64,32)) **232.75 s** (~3.9 min), the 99%-dominant cost, as projected.
- **Peak RSS ~35 GB** (not the ~4 GB "feature store" figure — that was the store
  only; peak holds raw .d + decoded frames + units + signatures at once). Fine on a
  big box; size the machine for ~35 GB, not 4.

### Bigger recall run (task 2, n=2000 frames, 194,071 units)
Robust now (n=2382-4513 true pairs vs 177 before). **Two labels A/B (task 4):**

TRUE cross-frame same-ion pair cosine — **median 0.000 under BOTH labels**:

| label | n | p75 | p90 | frac>=0.7 | frac>=0.9 |
|-------|---|-----|-----|-----------|-----------|
| shares-any-ion | 2382 | 0.624 | 0.900 | 0.22 | 0.10 |
| dominant-ion   | 4513 | 0.612 | 0.921 | 0.23 | 0.12 |

**CORRECTION 1 — the dominant-ion relabel (handoff step 2) did NOT help.** It barely
moved the true-pair cosine tail (frac>=0.7 0.23 vs 0.22) and *collapsed* the ion-label
precision (below). Reason: in ~8-ion chimeric windows the abundance-argmax ion is
frame-unstable, so genuinely-similar recurrences often get *different* dominant labels.
The bottleneck is chimericity, not the label — a better label can't fix it.

LSH recall / ion-label precision / cand/q, plus the **label-independent TRUE cosine of
the pairs LSH returns** (this is the honest recurrence metric):

| m x n | recall (any/dom) | ion-prec any | ion-prec dom | cand/q | cand-cos p50 | p90 | frac>=0.7 |
|-------|------------------|--------------|--------------|--------|--------------|-----|-----------|
| 32x16 | 0.30 / 0.19 | 0.409 | 0.095 | 3.53 | **0.555** | 0.960 | **0.48** |
| 64x32 | 0.20 / 0.12 | 0.813 | 0.185 | 0.94 | **0.950** | 0.982 | **1.00** |

**CORRECTION 2 — ion-label precision is the wrong yardstick.** It measures ion-label
agreement (confounded by chimericity), and at scale it is far below the small-n
80-91% (any-label: 0.41/0.81). Judged against *actual spectral similarity* (cand-cos),
the index is excellent: **(64,32) returns essentially only near-duplicates (median
cosine 0.95, 100% >=0.7); (32,16) is a wider net (median 0.55, 48% >=0.7).**

### Verdict for recurrence / MBR: VIABLE.
- **(64,32)** = high-purity near-duplicate / recurrence finder (median cand cosine
  0.95, 100% >=0.7, 0.94 cand/q). Use when you want clean recurring-spectrum pairs.
- **(32,16)** = broader recall net (median 0.55, 48% >=0.7, 3.5 cand/q). Use when you
  also want moderate-similarity recurrences and will re-score candidates by exact
  cosine (cheap: cand/q is small). The re-score step makes its 48% purity a non-issue.
- The mobility gate remains load-bearing; SDK-free + `mem` gives parallel reads for
  the full build. Ion-sharing labels should be retired as a similarity proxy — score
  candidates by cosine directly.

### dRT stratification — the recurrence signal is intra-elution-peak (2026-07-03)

Bucketed the (m,n) candidate-pair cosines by |ΔRT| (label-independent). This tests
whether the "recurrence" the hash finds is trivial elution-neighbour redundancy
(small ΔRT) or genuine cross-time recurrence (large ΔRT).

(64,32) candidate pairs — pure at every ΔRT, but ~97% are within 15 s:

| ΔRT | % of cand | median cos | frac>=0.7 |
|-----|-----------|-----------|-----------|
| 0-2 s   | 40.0% | 0.950 | 1.00 |
| 2-5 s   | 25.2% | 0.950 | 1.00 |
| 5-15 s  | 31.9% | 0.949 | 1.00 |
| 15-45 s |  2.9% | 0.952 | 1.00 |
| 45-120 s| 0.03% (32 pairs) | 0.79 | 0.84 |
| 120 s+  | 0.02% (18 pairs) | 0.80 | 0.72 |

(32,16) — high cosine only survives to ~15 s, then a cliff to cosine≈0 (the
long-ΔRT candidates are spurious band collisions, not similar spectra):

| ΔRT | median cos | frac>=0.7 |
|-----|-----------|-----------|
| 0-2 s   | 0.913 | 0.92 |
| 5-15 s  | 0.883 | 0.75 |
| 15-45 s | 0.000 | 0.10 |
| 45-120 s| 0.000 | 0.01 |

**Conclusion — reframes the use case.** ~97% of genuinely-similar pairs sit within
15 s ΔRT = inside one elution peak (timsTOF FWHM ~10-30 s). The hash finds
**elution-neighbour redundancy, not cross-time recurrence.** This is physically
expected: in a single DIA run each analyte elutes once, so there is no long-range
recurrence to find — hence the high-cosine mass vanishes past ~15-45 s.

Therefore:
- **What the within-run hash is genuinely good for:** unsupervised
  **denoising / redundancy collapse / 4D-feature construction** — group the
  co-eluting, co-mobile, cosine-similar mobility-window units of one elution peak
  into a single representative. Proven, valuable, but it is *peak/feature detection*,
  not MBR.
- **"Recurrence/MBR within one run" was chasing a signal that does not exist** in a
  single DIA acquisition. MBR is inherently cross-run and remains **untested**: it
  needs index-A / query-B on a same-scheme TimSim replicate pair (handoff step 4).
  The within-run experiments cannot validate or refute MBR.
- Next decisive experiment: simulate a small same-scheme replicate pair (same
  peptides+abundances, 2 seeds, short gradient) and measure cosine survival of the
  SAME analyte across the two runs. That is the real MBR test.

### Cross-run MBR test — the hash DOES match the same analyte across runs (2026-07-03)

Built the replicate pair the single-run experiments needed. TimSim `from_existing`
template mode: **MBR-PAIR-A** (fresh, 20k sampled peptides, 700 s DIA, diaPASEF
reference layout, real-data-noise OFF) → **MBR-PAIR-B** = `from_existing=A` with the
ONLY differences being per-analyte Gaussian **RT drift std=3.0 s** and **IM drift
std=0.003** (1/K0) — same peptides, ions, abundances, and crucially the **same
`ion_id` space** (verified identical), so "same analyte in A and B" is exact ground
truth. Configs: `rustdf/examples/mbr/mbr_pair_{a,b}.toml`. Harness:
`rustdf/examples/mbr_crossrun.rs` (index A, query B). Runtime ~20 s for the full
pair (~114k units each) on monster3.

A units 113,493 (110,812 labeled) | B units 114,878 (112,218 labeled).
Distinct dominant ions: A 11,466 / B 11,487 / **shared 10,379**.

**TRUE cross-run same-dominant-ion pair cosine** (best A twin per B unit, n=98,485):

| p10 | p25 | p50 | p75 | p90 | frac>=0.7 | frac>=0.9 |
|-----|-----|-----|-----|-----|-----------|-----------|
| 0.000 | 0.711 | **0.919** | 0.961 | 0.980 | **0.75** | 0.57 |

Contrast the WITHIN-run number: median **0.000**. Across runs the *same analyte*
recurs at median cosine **0.92** — because we are now comparing the same analyte's
window to its replica (same co-isolated neighbourhood + tiny drift + independent
noise), not two different chimeric windows that merely share a minor ion.

**LSH cross-run retrieval (B query -> A index):**

| m x n | recall | ion-prec | cand/q | cand-cos p50 | p90 | frac>=0.7 |
|-------|--------|----------|--------|--------------|-----|-----------|
| 32x16 | **0.728** | 0.314 | 9.52 | 0.906 | 0.969 | 0.84 |
| 64x32 | **0.565** | 0.390 | 4.48 | 0.946 | 0.979 | **1.00** |

(ion-prec is the same confounded metric as before — dominant-ion instability, not a
retrieval failure; cand-cos is the honest yardstick and it is excellent.)

**Verdict: MBR is viable.** Cross-run, the SimHash index matches the same analyte at
high cosine — the capability that provably does NOT exist within one run. (32,16)
recovers ~73% of matchable analytes; (64,32) is a purity-1.0 near-duplicate matcher
at ~57% recall. This is the intended MBR use case, demonstrated.

**Caveats / next (honest):** this is an optimistic pair — real-data noise OFF, small
drift, identical abundances ("IM+RT shift only", as scoped). Before claiming
production MBR: (1) re-run with `add_real_data_noise=true`; (2) sweep drift
(rt_variation_std, ion_mobility_variation_std) incl. IM drift past the |Δscan|<=5
gate; (3) add abundance variation (intensity_variation_std) for biological-replicate
realism; (4) widen/relax the mobility gate as IM drift grows. Each will lower recall —
the question is the operating curve, not existence, which is now established.

### Cross-run MBR with real-data noise ON (2026-07-03)

Regenerated the pair with `add_real_data_noise=true` (real 0.1% FA blank overlay,
`reference_noise_intensity_max=150000`), same seed-7 peptides + same RT/IM drift.
Configs `rustdf/examples/mbr/mbr_pair_{a,b}_noisy.toml`.

| metric (cross-run, same analyte) | clean | noise-on |
|----------------------------------|-------|----------|
| units / run | 113k | **586k (5x)** |
| fraction labeled | 98% | 38% |
| TRUE same-analyte pair cosine, median | 0.919 | **0.007** |
| ... frac >=0.7 | 0.75 | 0.13 |
| LSH (64,32) recall | 0.565 | **0.073** |
| LSH (64,32) returned-pair cosine frac>=0.7 | 1.00 | **1.00** |
| LSH (32,16) cand-cos median / recall | 0.906 / 0.73 | **0.000 / 0.13** |

**What happened.** The blank overlay adds ~70 background windows/frame (unit count
5x, 62% pure-noise units). After top-25 peak-picking, most analyte windows are
polluted by run-specific noise peaks, so the same analyte's A vs B windows stop
sharing their top peaks -> median true-pair cosine collapses 0.92 -> 0.007.

**But MBR survives at (64,32):** it still returns ONLY genuine matches (median 0.944,
100% >=0.7), just far fewer (recall 0.073). Precision is intact; yield drops. (32,16)
drowns in noise-noise collisions (3.3M candidates, median cand-cos 0).

**Two caveats — do not over-conclude from recall 0.07:**
1. **No denoising before unitization.** The harness hashes raw mobility-window top-25
   with no min-intensity / min-signal filter. A pre-hash threshold (drop noise-only
   windows; require a peak above the local floor) should recover most of the lost
   recall. This is a REPRESENTATION gap, not an LSH failure — the (64,32) 100%-purity
   result proves the hash still works on windows that survive.
2. **Thin-run SNR.** The blank noise floor is absolute (150000) but this run has only
   20k peptides -> sparse analyte layer -> worse SNR than a full-density (250k) run
   using the same floor. Recall here is likely pessimistic vs a realistic dense run.

**Net.** Existence of cross-run matching is robust (clean AND noisy, (64,32) returns
100% true-cosine pairs). The operating recall is noise-limited and hinges on a
denoising/peak-filtering step that the current representation lacks. Next, in order:
(a) add a min-intensity/min-peak filter to unit building and re-run noisy;
(b) SNR-matched noise or a dense run; (c) drift sweep. The headline "does it work"
is yes; the recall curve is a representation-robustness problem to solve next.

### Denoising recovers noisy MBR — top-N sub-sampling is the wrong knob (2026-07-03)

Follow-up to the noise-on collapse. Two representation levers, tested on the noisy
pair via `mbr_crossrun` args `[min_peak_intensity] [top_n|all]`:

Raw peak-intensity distribution (noisy .d): **p50=5, p90=68, p99=483, max ~200k**.
=> each mobility window holds a few strong real fragments buried in ~100+ intensity-5
blank-overlay noise peaks. `top_n=25` by count then spends ~20/25 slots on noise.

(1) **Drop top-N, keep ALL peaks** (let squared intensity weighting suppress noise):
true-pair cosine median 0.007->0.050, (32,16) recall 0.13->0.21. Helps, but under the
`sqrt` transform ||v||^2 = total window intensity, so the ~2k injected noise-intensity
still swamps WEAK analytes. Necessary, not sufficient.

(2) **Intensity floor (denoise) — the real lever.** all-peaks + floor sweep:

| floor | units (labeled%) | true-pair cos p50 | frac>=0.7 | (32,16) recall | (64,32) recall |
|-------|------------------|-------------------|-----------|----------------|----------------|
| 0     | 586k (38%) | 0.050 | 0.22 | 0.208 | 0.046 |
| 30    | 495k (39%) | 0.004 | 0.14 | 0.128 | 0.045 |
| 100   | 121k (51%) | ~0 (p75 0.84) | 0.32 | 0.294 | 0.153 |
| 300   | 16.6k (95%) | **0.884** | **0.81** | **0.748** | 0.438 |
| 1000  | 6.9k (98%) | 0.899 | 0.86 | **0.799** | 0.494 |

At floor 300-1000 the NOISY pair matches the CLEAN pair (clean: recall 0.73, cosine
0.92). **The noise collapse was entirely a denoising problem.** Removing the sub-300
intensity-5 blank background restores full cross-run MBR performance; recall at floor
1000 (0.80) even exceeds clean because denoising also sharpens the signal windows.

**Recommended representation for noisy data:** all-peaks + a signal floor (drop top-N).
The flat intensity floor works because TimSim blank noise is LOW-intensity; on real
data prefer a coherence filter (peak must persist across >=k adjacent scans/frames —
analytes form mobility/RT traces, noise does not). Floor is a coverage/precision dial:
higher floor = fewer analytes but each matched better.

Harness: `mbr_crossrun` now takes `[min_peak_intensity=0] [top_n=25|all|N]` and
prints the raw-intensity percentiles (floor=0). `filter_frame` does the pre-hash
denoise by rebuilding a parallel-array TimsFrame.
