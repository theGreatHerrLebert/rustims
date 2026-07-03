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
