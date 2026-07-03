# Plan: Sparse LSH Spectral-Match Index for timsTOF Datasets

**Status:** Draft for review (pre-implementation)
**Author:** David Teschner (with Claude)
**Date:** 2026-07-03

---

## 0. What this is (and is not)

**Product:** a locality-sensitive-hashing index over a timsTOF dataset's
**fragment (and precursor) spectra**, so that a batch of **query spectra** —
theoretical/predicted peptide fragment bags, or spectra from another run —
can be answered fast: *"is a spectrum carrying this fragment group present in
the data, and where (RT / mobility)?"* The intended role is a **cheap
approximate candidate-generation / spectral-match pre-filter** in front of
exact or co-elution-aware scoring — not a replacement for it.

**Why cosine-between-spectra is the right signal.** A spectrum is the *bag of
co-occurring fragment peaks*; that bag is the peptide's fingerprint. The same
peptide fragments into the same set of m/z at characteristic relative
intensities → **high cosine**. This is the established basis of spectral
library search and of large-scale spectral clustering (e.g. **falcon** uses
random-projection LSH; **GLEAMS** uses a learned embedding + ANN). We are
**not inventing the technique** — the contribution is a timsTOF-DIA-native,
per-mobility-window, Rust implementation aimed at search-time candidate
generation.

**Honest positioning / risks (see §2.1):**
- The technique is prior art; novelty is the application + granularity.
- A single raw DIA scan is a *multiplexed, partial* slice — this is why the
  **unit is a mobility moving-average window**, not a raw scan (§4.1).
- The query is a *clean* bag; a data spectrum is a *chimeric mixture* — so
  **asymmetric containment may beat symmetric cosine** (§4.1.2). First-class
  design fork.
- Whether this beats existing candidate generation (fragment-ion indices,
  library/XIC extraction) is **unproven** — Phase 1.5 must show it.

## 1. Motivation

The repo already ships a SimHash / random-projection LSH implementation
(`packages/imspy-predictors/src/imspy_predictors/hashing.py`: `CosimHasher`,
`TimsHasher`, `SpectralHasher`). It is **wired to nothing** outside its own
tests, it is **torch-based**, and — most importantly — its `TimsHasher`
representation is structurally wrong for real data:

```python
# hashing.py — TimsHasher
target_vector_length = num_dalton * 10**resolution + 1
```

This forces a **fixed absolute m/z grid** committed *before* hashing. Every
choice it offers is bad:

- **Resolution trap:** defaults give 10 Da @ `resolution=1` → 101 bins =
  0.1 Da/bin, far too coarse for timsTOF. Real resolution (`resolution=3`)
  makes a single 10 Da window 10,001 dims, and you'd need ~160 windows to
  cover 100–1700 m/z. Dense, huge, ~99.9% zeros.
- **Sparsity wasted:** real spectra are a handful of peaks; the dense
  `W @ hash_tensor` matmul pays full price to learn which bins are occupied.
- **No mass invariance:** an absolute grid means calibration drift / mass
  shift walks peaks across bin boundaries; window edges split peaks.
- **Arbitrary window:** 10 Da is a demo number with no principled basis.

**Verdict:** nice notebook demo, structurally unusable at scale.

## 2. Idea

Keep the SimHash *machinery* (sign-of-random-projection → bit-pack), throw
away the dense-window *representation*. Hash **sparse spectra** (bags of
co-occurring peaks) via **feature hashing** (projection weights generated on
the fly from a hash of the m/z bin — no stored matrix, no vocabulary bound),
build a banded LSH index over the dataset, and query it with a batch of
spectra.

**The unit is a mobility moving-average window — and that is load-bearing,
not a denoising afterthought.** In diaPASEF, mobility separation happens
*before* fragmentation: ions elute out of the TIMS device by mobility, then
are isolated and fragmented, so **a fragment inherits its precursor's
mobility.** All fragments of one precursor therefore land at essentially the
**same scan index** — the fragment group is *concentrated in a narrow scan
band*, not scattered. Consequences:

- A single raw scan catches only a thin slice of that band (partial bag, low
  SNR).
- The precursor population spans an **IM peak**, and each scan samples the
  **same fragment bag** at different abundance/noise. A **moving average over
  adjacent scans integrates that IM peak profile** — improving SNR and
  completeness of the bag. It is *not* true deconvolution (co-isolated
  precursors remain mixed, §4.1.2); it is a cheap completeness/SNR device.
  This is *why* the average is over the **scan/mobility axis**: that is the
  axis along which one precursor's fragments coincide.
- Keep the **native mobility resolution** (~hundreds–1000 scans/frame); slide
  the average along it (overlapping windows, ~one unit per scan), **clamped to
  DIA isolation-tile scan boundaries** (§4.1). Mobility is preserved as a
  discriminative coordinate of each unit.

Window width `w` ≈ the **IM peak FWHM** (physically-anchored default); it is
the primary granularity knob (§4.1). RT (frame) is the second co-occurrence
axis — averaging across frames for SNR, or intersection/persistence across
frames for sharpening — kept as a secondary knob, not v1.

### 2.1 Where this sits vs. existing methods (be honest)

- **Not** peptide identification on its own, and **not** a replacement for
  library/XIC extraction (which scores the full co-elution profile and is
  more sensitive). This is a *coarse, cheap first pass*.
- **Not** a fragment-grouping / deconvolution replacement — grouping
  co-eluting fragments into pseudo-spectra is a co-elution problem the
  existing geometric path (`candidates.rs` interval search) already handles.
- **Prior art exists** (falcon, GLEAMS, MASST-style repository search): LSH/
  ANN over spectra is established. Novelty = timsTOF-DIA, per-mobility-window,
  Rust, as a search pre-filter.
- **Legitimate targets:** search-time candidate generation (pre-filter before
  exact scoring), cross-run / match-between-runs recurrence, dedup /
  non-redundant library building, and QC/exploration ("is this pattern
  anywhere in my run").
- **Must be proven, not assumed:** that (i) the mobility-window bag carries
  enough of the fragment group, and (ii) LSH candidate generation is actually
  faster/complementary to what exists. Phase 1.5 is the go/no-go.

**Non-goal:** does **not** replace the DIA clustering / pseudo-spectrum
pipeline; it is a **parallel approach** built to *compare* against it.

## 3. Cost hypothesis (to be proven, not assumed)

Order-of-magnitude for one DIA run (constants are guesses, to be measured):

| Quantity                              | Rough      |
|---------------------------------------|------------|
| Mobility-window units / run (≈1/scan) | ~2–5 M     |
| Peaks per unit `k` (after windowing)  | ~10²–10³   |
| Projections `L = b·r`                 | ~640       |
| MACs to hash everything               | ~6e11+     |

(Windowing unions peaks across `2w+1` scans, so `k` per unit is higher than a
raw scan's — another reason the number below is a loose lower bound.)

**This table is a HYPOTHESIS, not a design input — do not let it steer
decisions.** The `~6e11 MACs → ~4 s` framing is misleading: this is *not* a
dense GEMM workload. It is hundreds of billions of hash/mix/sign ops plus
sparse float accumulations, which do not vectorize like a matmul and have
irregular memory access. The real number is unknown until measured.

**Headline hypothesis (to be confirmed or killed by the Phase 2 benchmark):**
*hashing the whole dataset is cheaper than reading it* — wall-clock is
dominated by frame I/O + zstd/LZF decompression that `rustdf` already pays.
The benchmark must **time hashing separately from I/O**. If false, this
whole cost argument collapses and the approach must be re-justified on
retrieval value alone.

## 4. Design

### 4.1 Representation

- **Unit:** a sparse spectrum = bag of `(mz, intensity)` peaks, formed by a
  **mobility moving-average window** over the native scan axis (§2). Each unit
  is centered at a scan `s` and aggregates scans `[s−w, s+w]` (sum or mean of
  the sparse peaks), producing overlapping windows at native mobility
  resolution.
- **Window width `w` (primary granularity knob):** anchored to the **IM peak
  FWHM** (a few → tens of scans). This is a **signal-vs-chimericity dial**:
  - too narrow → captures only part of the fragment group (partial bag),
  - too wide → sweeps in fragments of *other* precursors at neighboring
    mobility (contamination).
  Swept in Phase 1.5; the physically-motivated default is ≈ one IM FWHM.
  `w = 0` (raw scan) and cross-frame RT aggregation are endpoints of the same
  sweep, not separate code paths.
- **HARD RULE — windows must not cross DIA isolation-tile scan boundaries
  (from review; would otherwise be silent corruption).** In diaPASEF the
  quadrupole steps through different isolation windows across scan ranges
  *within a frame*; a window straddling a tile edge aggregates fragments from
  two different quad selections → ambiguous isolation context. **Clamp each
  moving-average window to the enclosing `ProgramSlice` scan range** (v1), so a
  unit belongs to exactly one isolation window. Boundaries come from
  `DiaIndex.group_to_scan_ranges` / `ProgramSlice` (`rustdf/src/data/dia.rs`,
  scan ranges ~`:237`). (MS1 has no tiles → no clamp; the driver takes an
  optional segment-boundary list, empty for precursor.)
- **m/z → feature index (THE CRUX — see §4.1.1).** SimHash hashes a vector
  in ℝ^D; the *feature id is the coordinate index*, so this mapping defines
  the vector space itself. Feature hashing (`hash(seed, feature_id, k)`)
  only saves us from *materializing* ℝ^D — it does nothing to fix a bad
  coordinate assignment.

### 4.1.1 m/z → feature index (defines the space)

The mapping must satisfy three properties at once:

1. **Deterministic & stable** — same physical m/z → same id, across build
   *and* query, across machines.
2. **Tolerance-aware** — peaks within instrument tolerance must share a
   coordinate (or share weight), or real matches never collide.
3. **Discriminative** — peaks farther apart than tolerance must get
   different coordinates, or everything collides with everything.

Naive fixed binning satisfies (1) and (3) but **fails (2)** — it
reintroduces the boundary problem we are fleeing (two identical peaks
straddling an edge get different ids).

**Key correction (from review):** a single splat kernel whose width equals
the bin width does *not* guarantee tolerance. A peak shifted by ~one bin can
have near-zero overlap depending on its phase within the bin. So we must
**decouple bin granularity from matching tolerance**: bins finer than the
tolerance, and a splat kernel spanning ≈ the tolerance.

- **(a) Log-ppm bins (not linear mDa)** — constant-ppm width matches MS
  resolution behaviour. **`bin_ppm` is the bin *width*** (define once, not
  half-width, not σ), and it must be set *finer* than the desired matching
  tolerance `tol_ppm` (e.g. `bin_ppm = tol_ppm / 3`):
  ```
  scale = 1 / ln(1 + bin_ppm·1e-6)      // fine bins, e.g. bin_ppm ≈ 5
  pos   = ln(mz) · scale                // continuous bin coordinate
  center= round(pos)                    // nearest bin center
  ```
- **(b) Tolerance-width splat kernel** — spread each peak over the bins
  within ±`tol_ppm` using a triangular (or Gaussian) kernel, weights summing
  to 1, so a sub-tolerance mass drift moves weight smoothly and two peaks
  within `tol_ppm` retain high overlap (graceful cosine degradation under
  mass shift). With `bin_ppm = tol_ppm/3` this is a ~3–7-bin splat, not
  2-bin. Each emitted feature is `(bin_id, intensity·kernel_weight)`; then
  `feature_id = bin_id` (any i64) → `hash(seed, bin_id, k)`; no compaction
  into [0,D) needed. **L2-normalize AFTER splatting.**
  - *Open:* exact kernel shape + width (`tol_ppm`) and cost vs benefit of
    Gaussian over triangular — settle empirically in Phase 1.5.

**tof-index shortcut — RESTRICTED (coupled to probe source, §7 Q8):** the
tof index in `IndexedMzSpectrum.index` (`mscore/src/data/spectrum.rs:500`)
is discrete and calibration-consistent *within a dataset*, but the converter
API is frame-specific (`tof_to_mz`/`mz_to_tof` take a `frame_id`,
`handle.rs:229,271`) — so raw tof is **not** a universal m/z coordinate.
  - **Only** valid for probes that literally preserve raw tof indices
    (same-run self-probes, pre-calibration). The moment a probe passes
    through m/z values / Python wrappers / predictions, the raw index is
    gone.
  - Even then, **exact tof equality has the same hard-bin boundary problem**
    (a one-index centroid shift ⇒ no shared feature), so tof-direct must
    **still splat over neighboring tof indices** — it is not "no binning."

The chosen mapping (tof-direct+splat vs log-ppm+splat, `bin_ppm`, `tol_ppm`,
kernel) is part of the index config header — a query only aligns against a
matching mapping.
- **Intensity transform (parameter):** `none | sqrt | log1p`, then **L2
  normalize** before signing (cosine is intensity-dominated; the top peak
  must not decide every bit).

### 4.1.2 Similarity semantics — containment vs cosine (first-class fork)

The query is a **clean** fragment bag; a data-side unit is a **chimeric
mixture** (co-isolated precursors at the same mobility+RT), where the query
peptide's fragments are a **subset** plus contaminating peaks. Symmetric
cosine **penalizes the extra peaks in the data unit**, so a fully-present
query can score low purely from mixture contamination. What we often actually
want is **asymmetric containment**: *how much of the query is explained by the
data unit*, not *how similar the two bags are*.

This matters because **plain SimHash approximates symmetric cosine.** If
containment is the true target, the design changes:

- **(A) Symmetric cosine (SimHash, default to start):** simplest; degrades
  with chimericity. Good when the mobility window is clean enough.
- **(B) Asymmetric containment:** index **data** peaks, query with the
  **fragment** bag, and score/verify by *matched query intensity fraction*.
  Candidate generation can still be SimHash (a contained query still shares
  many sign bits), but **verification uses a containment score**, not cosine.
- **(C) Set-containment LSH (MinHash/​b-bit) on peak *sets*** for the
  candidate stage if intensity-weighting proves unnecessary — MinHash
  estimates Jaccard/containment, which is the *right* family for "is this set
  inside that set." (Contrast §4.5, where MinHash is wrong for *weighted
  cosine verification*.)

**Precise containment score (define it, don't hand-wave):** with query bag
`q` and data unit `d` both over splatted, duplicate-summed, sorted feature
ids, and `q` L2-normalized,

```
containment(q, d) = Σ_i  q_i · min(q_i, d̂_i) / Σ_i q_i²
```

where `d̂` is `d` restricted to `q`'s features and per-feature scaled so a
fully-present, correctly-proportioned fragment set → 1.0, and the denominator
is the **query** norm (asymmetric: extra peaks in `d` are *not* penalized).
Exact weighting (`min`, dot-on-matched, or spectral-angle-on-matched) is a
Phase-1.5 choice, but the contract is fixed: **denominator = query mass,
duplicates summed pre-score, splat applied identically to query and unit.**

Decision: **start with (A) symmetric cosine for the candidate-generation
primitive**, verification metric **pluggable (cosine vs containment)**, settle
in Phase 1.5. **KILL-SWITCH (from review):** cosine-SimHash candidate
generation can *fail to surface* a candidate that containment would accept
once contamination dominates the sign bits — so Phase 1.5 must measure
**recall as a function of mixture load**, not assume "a contained query still
shares many bits." If candidate recall collapses under realistic chimericity,
the candidate stage itself (not just the verifier) must move toward a
containment/set-LSH family (B)/(C).

### 4.2 LSH scheme seam + SimHash (variant #1) — `mscore/src/algorithm/lsh/` (new)

**Do not commit to cosine only.** Cosine SimHash is the *first* LSH family we
ship, behind a thin trait so a set/containment family (MinHash / weighted
MinHash — the §4.1.2 (B)/(C) escape hatch) can drop in later without touching
the index, store, or driver. This is a seam along a **known** axis of variation
(§4.1.2), not speculative generality.

```rust
/// One LSH family. Turns a sparse feature vector into band keys whose
/// collision probability is monotone in the family's target similarity.
trait LshScheme {
    /// Uniform output: `b` band keys. The banded index is NOT generic over
    /// this — it is always Vec<HashMap<u64, Vec<UnitId>>>, any family.
    fn signature(&self, features: &[(i64, f32)]) -> Vec<u64>;
    /// Verification metric this family implies (cosine / containment / Jaccard).
    fn verify(&self, query: SparseView, unit: SparseView) -> f32;
    /// Analytic collision law P(collision | similarity) — drives the sweep.
    fn collision_law(&self, similarity: f32) -> f32;
}
```

Why the seam stays thin (nothing else becomes generic):
- **Driver** already emits `&[(i64, f32)]` — SimHash uses weights, MinHash
  ignores them, weighted MinHash uses them; same input feeds all three.
- **CSR store** (§4.5) holds normalized sparse vectors — cosine, containment,
  and Jaccard are all computable from it at verify time.
- **Index** is monomorphic because `signature()` is always `Vec<u64>`.
- **Per-family collision law:** the `(b, r)` sweep (§4.4.1) is re-run per
  scheme — hence `collision_law` on the trait; the config header records the
  scheme id so a query only aligns against a matching scheme+params.

Ship exactly **one** impl now (`CosineSimHash` in `lsh/simhash.rs`); add a
second only if Phase 1.5 demands it. Trait + first impl live under
`mscore/src/algorithm/lsh/`, registered via `pub mod lsh;` in
`algorithm/mod.rs`.

**`CosineSimHash` (variant #1):**

- Input: a sparse vector as `&[(i64 feature_id, f32 value)]`. **`feature_id`
  is `i64` everywhere** (log-ppm bin ids can be negative for m/z < 1; CSR
  stores `i64`; the primitive takes `i64`) — one signed type end to end, no
  `u64`/`i64` boundary casts.
- **Projection distribution (default Gaussian, from review):** for each of
  the `L = b·r` projections `k ∈ [0, L)`, weight `= proj(hash(seed,
  feature_id, k))`, accumulate `Σ value · weight`, take sign → **one bit**.
  - **Gaussian (default)** — `proj` maps the hash to `N(0,1)` via cheap
    Box-Muller (or a ziggurat). This is what the `1 − θ/π` collision
    guarantee actually requires (rotationally invariant hyperplanes) and
    matches the existing reference (`hashing.py:74` uses `np.random.normal`).
  - **Rademacher ±1 (opt-in)** — cheaper, but only *asymptotically* satisfies
    `1 − θ/π` (CLT) and can deviate for sparse / low-effective-dimension
    vectors with repeated weights. Treat as an **empirical** variant to be
    benchmarked against Gaussian, not a proven gate.
  - **Zero-dot tie handling (define explicitly):** when `Σ value·weight ==
    0`, map to a fixed bit (e.g. `sign(0) := +1`) deterministically — do not
    leave it platform-dependent.
- Output: signature = `L = b·r` bits, later split into `b` bands of `r` bits
  (§4.4.1). Pack each band into an integer (`u32` if `r ≤ 32`, else `u64`).
- Hash: fast, stable (wyhash/xxhash/splitmix64) — **must be reproducible
  across machines** (no nondeterminism).
- **Correctness gate — measure the three probabilities SEPARATELY (from
  review; the earlier draft conflated them):**
  - **per-bit** collision probability must track `p = 1 − θ/π` (this is the
    quantity the SimHash guarantee is about — Gaussian);
  - **per-band** collision = `p^r` (all `r` bits agree);
  - **candidate** probability = `1 − (1 − p^r)^b` (≥1 band agrees, §4.4.1).
  Phase 0 tests the **per-bit** law directly (Hamming distance vs cosine);
  band/candidate behavior is a downstream consequence tested in §4.4.1 /
  Phase 1.5. For Rademacher the per-bit gate is "close enough to Gaussian on
  *MS-realistic* sparse vectors," measured, not assumed.

### 4.3 timsTOF driver — `mscore/src/timstof/lsh.rs` (new)

- Takes a `TimsFrame`, forms **mobility moving-average windows** (§4.1),
  applies transform + L2 norm, calls the `simhash` primitive.
- **Do NOT use `TimsFrame::to_tims_spectra` on the hot path (from review)** —
  it builds a `BTreeMap` and allocates per scan (`frame.rs:239`). Walk the
  frame's flat SoA arrays (`scan[]`, `tof[]`, `mz[]`, `intensity[]`) by scan
  range instead.
- **Ownership (from review): `mscore` cannot reuse `build_scan_slices` —**
  that helper lives in `rustdf` (`cluster/cluster.rs:46`) and `mscore` has no
  `rustdf` dependency (the arrow points the other way). So implement a
  scan-range iterator **in `mscore`** over `TimsFrame`'s flat arrays; `rustdf`'s
  version is the *model*, and can later be refactored to call the `mscore` one
  to avoid duplication. Note the helper assumes scans are **contiguously
  grouped** in the frame arrays — verify/enforce that ordering (sort by scan
  if needed) before windowing.
- The window is a widened, sliding scan range over the same arrays; sum peaks
  across the range, then (per §4.1.2) either merge duplicate feature ids by
  summing values (for cosine) or keep set semantics (for containment).
- **Takes an optional `segment_boundaries: &[(scan_lo, scan_hi)]`** (the
  frame's `ProgramSlice` scan ranges); windows are **clamped within a
  segment** and never span two (§4.1 hard rule). Empty ⇒ no clamp (MS1). This
  keeps `mscore` DIA-agnostic — the caller (`rustdf`, which owns `DiaIndex`)
  supplies the boundaries — while making it impossible for a DIA build to
  produce cross-tile units.
- Emits `(unit_id, signature, metadata)` where `unit_id` identifies
  `(frame_id, center_scan)` and metadata carries `rt, mobility, ms_type` **and
  (for MS2) the DIA isolation window / quadrupole context** (§4.4).
- Note the two conventions: assembled `TimsFrame` uses `scan: Vec<i32>` while
  the raw reader path uses `scan: Vec<u32>` — pick one at the boundary
  (§4.2 fixes `feature_id` to `i64`; keep scan handling consistent too).

### 4.4 Band index + dataset build — `rustdf/src/cluster/lsh_index.rs` (new)

Lives in `rustdf` because it owns frame I/O (`mscore` has no reader
dependency). Sits beside the existing exact-key grouping in
`rustdf/src/cluster/`.

- **Index structure:** classic banded LSH, **scheme-agnostic** (§4.2) — `b`
  band tables `Vec<HashMap<u64, Vec<unit_id>>>`, fed by any `LshScheme`'s
  `signature()`; the index code never changes when the family does. Two units
  are candidates if they collide in ≥1 band. Band count `b` and band width `r`
  (bits/band) are **swept parameters set by Phase 1.5, not the inherited
  `32×20` defaults** (§4.4.1), and the sweep is **per-scheme** (each family has
  its own collision law).
  Separate indices for **precursor (MS1)** and **fragment (MS2)** spaces
  (filter on `ms_type`: `Precursor=0`, `FragmentDda=8`, `FragmentDia=9`).
- **Build:** iterate `frame_id ∈ 1..=get_frame_count()` (there is **no**
  built-in frame iterator — 1-based ids, `frame_index = frame_id - 1`),
  filter by `ms_type`, drive the `mscore` per-scan hasher, insert into band
  tables.
- **Parallelism + THREADING CONSTRAINT (more nuanced than first stated):**
  the lazy loader's `get_slice` is **serial** and the Bruker SDK converter is
  **not thread-safe** (`handle.rs:817`, `uses_bruker_sdk()` `:1283`). But the
  in-memory loader is **not automatically safe for every dataset either** —
  its raw-frame path assumes type-2 (zstd) decompression (`handle.rs:880`)
  while the lazy path handles type-1 (LZF) *and* type-2 separately
  (`handle.rs:598`). So the build must: (i) gate on `uses_bruker_sdk()`, (ii)
  use a thread-safe converter (`Simple`/`Calibrated`/`Lookup`/
  `BrukerFormula`), **and** (iii) confirm the loader path handles the
  dataset's actual compression type. Pattern: rayon over frames → per-thread
  partial band maps → merge (or sharded-lock concurrent insert).
- **Converter ACCURACY must be compatible with `tol_ppm` (from review):**
  thread-safe is necessary but not sufficient. The `Simple` fallback can be
  ~5 Da off (`handle.rs:987` warning) and even `BrukerFormula` model-2 shows
  max residuals >10 ppm in the docs (`rustdf/BRUKER_TRANSLATORS.md:122`). If
  the m/z error of the chosen converter exceeds `tol_ppm`, the splat can't
  save it — build and query must use the **same** converter, and `tol_ppm`
  must be set ≥ the converter's residual (or use the Bruker SDK single-threaded
  for accuracy-critical builds).
- **Metadata per unit (needed for correct DIA queries + physics gate §4.6):**
  store `frame_id, center_scan, rt, mobility, ms_type`, **the DIA isolation
  window** for MS2 (`window_group` / quad isolation m/z range), and an optional
  **normalized/indexed RT** (`rt_norm`, for cross-run relative-RT gating —
  populated when an alignment/iRT mapping is available, else absent). Mobility
  and `window_group` are the always-on gate coordinates; `rt_norm` is optional.
  Fragment "presence" in DIA is window- *and* mobility-scoped — the same bag at
  a different isolation window or mobility is a different context. Source:
  `DiaIndex.frame_to_group` / `group_to_isolation` (`rustdf/src/data/dia.rs:131-147`).
- **Persistence:** serialize with bincode (build-once / query-many). Record
  the full param set (**scheme id**, seed, `b`, `r`, projection distribution,
  `bin_ppm`, `tol_ppm`, kernel, transform, unit granularity, multi-probe
  radius) in a header — a query only makes sense against a byte-for-byte
  matching config, **including the same `LshScheme`**.

### 4.4.1 Banding: recall analysis (do the math BEFORE building)

The band structure — not the raw signature — decides recall. For band count
`b` and band width `r` bits, a pair at cosine `c` becomes a candidate with

```
P(candidate) = 1 − (1 − p^r)^b ,   p = 1 − arccos(c)/π
```

The inherited demo params (`b=32, r=20`) are a **near-duplicate detector**:

| true cosine `c` | P(candidate) at b=32, r=20 |
|-----------------|----------------------------|
| 0.95            | ~0.98                      |
| 0.90            | ~0.77                      |
| 0.85            | ~0.49                      |
| 0.80            | ~0.28                      |
| 0.70            | ~0.09                      |

Noisy per-scan DIA matches may sit at `c ≈ 0.75–0.85`, where recall collapses
to ~30%. So `32×20` is **presumed wrong** until a sweep says otherwise.
Levers to raise recall at the target cosine band:

- **Shorter bands** (`r ≈ 8–14`) and/or **more tables** (`b`) — raises recall
  but also the raw-collision rate: unrelated pairs (`p=0.5`) yield
  ≈ `b · 2^-r · N` candidates/query, which sets verification cost. There is a
  real recall ⇄ candidate-volume ⇄ verify-cost trade-off; the sweep finds the
  knee.
- **Multi-probe LSH** — also probe band keys within Hamming radius 1–2, to
  recover near-misses without shrinking `r`.
- **Physics gate as a candidate-volume lever (§4.6):** mobility (+ isolation
  window + relative RT) filtering removes most spurious collisions *before*
  verification, so we can afford **looser bands** (higher recall) while holding
  verify-cost down — *loosen the hash, tighten with physics*. The `(b, r)`
  sweep should therefore be run **with the gate on**, since the gate changes
  the usable operating point.

Deliverable of §Phase 1.5: the `(b, r, multi-probe radius)` operating point
for the target cosine range **with the physics gate enabled**, reporting
recall *and* candidates/query *before and after* the gate.

### 4.5 Verification storage

To answer "present or not" with real similarity (not just collision count)
we need each candidate's normalized sparse vector at query time.

**Decision (was open, now settled per review): store the L2-normalized
features in CSR layout.** Not `Vec`-per-scan — for 2–5M units × ~100 peaks,
that is millions of tiny allocations, and even `(u32 feature, f32 value)`
pairs are ~1.6–4 GB raw before offsets/metadata/allocator overhead (splatting
inflates this further). Use three flat arrays:

```
unit_offsets: Vec<u64>     // len = n_units + 1  (CSR row pointers)
feature_ids:  Vec<i64>     // len = nnz total
values:       Vec<f32>     // len = nnz total, already L2-normalized
```

plus parallel metadata arrays (`frame_id`, `center_scan`, `rt`, `mobility`,
`ms_type`, `window_group`). Verify = sparse dot (cosine) or matched-query
fraction (containment, §4.1.2) over sorted feature ids; cache-friendly, no
per-unit allocation.

**Memory estimate — compute it in Phase 1.5, do NOT defer to Phase 3 (from
review).** The moving-average window *increases* nnz (it unions peaks across
`2w+1` scans), and the splat multiplies each peak by the kernel width. Rough:
`n_units · avg_nnz · splat_factor · (8 B id + 4 B val)` — for 2–5M units,
avg_nnz ~10²–10³ after windowing, splat ×3–7, this is **plausibly tens of GB
before band postings and hashmap overhead**. This can be the real ceiling, so
Phase 1.5 must report measured nnz-after-windowing and projected index size —
it feeds the `w` / storage-mode decision, and the option-(C) fallback (longer
SimHash instead of stored CSR).

Rejected alternatives:
- **Collision-count-only** (store nothing) — too imprecise to gate
  membership; keep only as a fast pre-rank.
- **Plain MinHash refinement** — MinHash estimates **Jaccard, not weighted
  cosine**; wrong tool here. If CSR memory ever becomes the ceiling, the
  right escape hatch is a **longer SimHash signature** (Hamming-based cosine
  estimate), not MinHash.

### 4.6 Query path — `rustdf/src/cluster/lsh_index.rs`

- Input: a **batch of query spectra** (clean fragment bags — predicted /
  theoretical, or units from another run), each optionally carrying an
  **expected mobility** and **normalized RT** (see gate below).
- Per query, a **"LSH proposes, physics disposes"** pipeline: hash → per-band
  candidate lookup → union → **physics gate** (below) → **verify** by cosine or
  containment (§4.1.2, §4.5) → threshold → result: `present: bool`, matching
  unit locations `(frame_id, center_scan, rt, mobility)`, and scores.
- **Physics gate (all optional, cheap pre-verify sieve on §4.4 metadata):**
  - **Isolation window** — drop candidates whose `window_group` can't contain
    the query's precursor m/z (MS2 only).
  - **Mobility (strong, default on)** — `|mobility_q − mobility_unit| ≤ Δm`.
    Two distinct tolerances (from review — do not use one):
    - **Measured query** (unit-vs-unit, S2): `Δm ~ a few scans` (CCS
      reproducibility ≪ 1 %). Hard gate.
    - **Predicted query** (peptide bag): mobility comes from
      `DeepPeptideIonMobilityApex`, which needs **explicit `(sequence, charge,
      precursor_mz)`** and predicts CCS → `1/K0` (the CCS→`1/K0` conversion
      itself depends on m/z + charge). **Require charge + m/z explicitly — no
      charge guessing; unsupported charge → `NaN` → skip the gate for that
      query, don't fabricate one.** Prediction MAE (~7 CCS units bundled) ≫
      instrument reproducibility, so `Δm` here must be **wider and
      run-calibrated**, or the predicted-mobility gate stays **soft** (rank
      term, not hard filter) until Phase 1.5 calibrates it.
  - **Relative RT (optional, alignment-dependent)** — gate in a **normalized /
    aligned RT space** (iRT / indexed-RT / gradient-fraction), *not* absolute
    RT, so it works cross-run. **Off by default cross-run** until an alignment
    / iRT mapping exists; on and strict for same-run / dedup.
  - Gate can act as a **hard filter** (drop) and/or a **soft term** folded into
    the final rank (closer mobility/RT ⇒ higher score).
- **Decision rule (define explicitly):** similarity/containment threshold ⇒
  "present"; batch-level false-positive control (e.g. decoy queries →
  empirical FDR). "Present" for MS2 is **isolation-window- and
  mobility-scoped** — the same bag at the wrong mobility/window is not a hit.
  (See §6.)

### 4.7 PyO3 binding + Python wrapper

- `imspy_connector/src/py_*.rs`: expose `build_lsh_index(data_path, params)
  -> PyLshIndex`, `PyLshIndex.query(probe_spectra) -> results`,
  `save`/`load`.
- Python wrapper following the `RustWrapperObject` pattern (`get_py_ptr` /
  `from_py_ptr`), likely in `imspy-core` (timstof) or a small new module.
- **Deprecate** the dense torch `TimsHasher`; keep `CosimHasher` /
  `SpectralHasher` as the Python reference impl.

## 5. Phasing (staged with kill-switches)

- **Phase 0 — `LshScheme` trait + `CosineSimHash` variant #1 (`mscore`).**
  `algorithm/lsh/` (trait in `mod.rs`, impl in `simhash.rs`) + unit tests.
  Ship one impl; the trait exists so Phase 1.5 can add MinHash without
  touching the scaffolding (§4.2). Port the *skeleton* of `test_hashing.py`
  (§6.1) but build the real statistical gate here. *Gate (Gaussian):* per-bit
  collision tracks `1 − θ/π`; Rademacher measured against it, not assumed.
- **Phase 1 — timsTOF driver + m/z→index (`mscore`).** `lsh.rs`: mobility
  moving-average windows over flat frame arrays (not `to_tims_spectra`;
  own scan-slice iterator in `mscore`), transforms, L2 norm, log-ppm+splat
  mapping. **Must include the optional `segment_boundaries` clamp from day one**
  (§4.1 hard rule) so DIA callers can't produce cross-isolation-tile units —
  don't bake in the wrong unit semantics. Tests on synthetic frames.
- **Phase 1.5 — Recall/param spike (NEW — the real go/no-go).** *Before* any
  persistence or PyO3, and against **real predicted/theoretical query spectra
  where possible** (not just self-probes): sweep and measure. Axes:
  - **unit granularity** = mobility window width `w` (raw scan → ≈IM FWHM →
    wider) — the signal-vs-chimericity dial;
  - **similarity semantics** = symmetric cosine vs asymmetric containment
    (§4.1.2);
  - `(b, r, multi-probe radius)`, splat `tol_ppm` + kernel, transform,
    Gaussian-vs-Rademacher.
  Report recall *and* candidates/query *and* nnz-after-windowing / projected
  index memory (§4.5), **and candidate recall as a function of mixture load /
  chimericity** (§4.1.2 kill-switch — proves cosine-SimHash actually surfaces
  contained queries, or forces the candidate stage to a set/containment LSH).
  **Kill-switch:** if no operating point gives useful recall at acceptable
  candidates/query *and* memory for the target similarity range, stop — later
  engineering can't fix a bad representation. Also sanity vs. a baseline
  (exact cosine / simple fragment-index) so we know it's actually
  *complementary or faster*, not just "works."
- **Phase 2 — Dataset build + band index + cost benchmark (`rustdf`).** Build
  over one real run using the Phase-1.5 operating point; **time hashing
  separately from I/O**; report units, peaks/unit distribution, build
  throughput, index memory. *Kill-switch:* if hashing is not
  cheap-relative-to-I/O, the cost story (§3) collapses.
- **Phase 3 — Query path + CSR verification + retrieval benchmark.** Batch
  query at scale; recall@candidate vs brute-force ground truth on the full
  run; measure probes/s and false-positive rate on decoys.
- **Phase 4 — PyO3 + Python wrapper + example.** Callable end-to-end;
  example notebook; deprecate dense `TimsHasher`.
- **Phase 5 (stretch) — Comparison harness vs DIA.** Compare LSH candidate
  generation against the geometric path (`candidates.rs` interval search) on
  the same data — quality + speed. **No integration**, just a bake-off.

## 6. Evaluation protocol (how we know it works)

Cost is settled by Phase 2. Quality is the real risk:

- **Recall (self-probe + predicted):** probes extracted from the dataset, and
  ideally *predicted* fragment bags for peptides known to be present, must be
  found. Report recall@candidate vs brute-force ground truth.
- **Precision (decoy probe):** random / shuffled / foreign-run / wrong-window
  probes must be reported absent. Report false-positive rate / empirical FDR.
- **Sweep:** `b`, `r`, multi-probe radius, `bin_ppm`/`tol_ppm`/kernel,
  transform, **mobility window width `w`**, **cosine vs containment**,
  threshold. Produce ROC-like curves.
- **Chimericity check:** does widening `w` help (fuller bag) or hurt (more
  contamination)? Where does containment beat cosine? This locates the
  operating point.
- **Baseline comparison:** vs exact cosine/containment and (if available) a
  simple fragment-ion index — LSH must be *faster or more complete*, not just
  functional.

### 6.1 Test suite — port skeleton, build the real gate

The existing `packages/imspy-predictors/tests/test_hashing.py` is **mostly
smoke tests** (shapes, dtype, `repr`, device); the *only* semantic assertion
is `test_similar_vectors_similar_keys` (`:52`), and it is a weak
"similar > different matches" check — not a collision-vs-cosine curve. So:

- **Port** the structural skeleton to Rust (`#[cfg(test)]` in `simhash.rs`) —
  construction, shapes, determinism, tie handling.
- **Build new** (the actual correctness gate, absent today): (i) measured
  band-collision probability vs true cosine tracking `1 − θ/π` (Gaussian);
  (ii) Gaussian-vs-Rademacher divergence on MS-realistic sparse vectors;
  (iii) splat robustness — cosine stability under sub-`tol_ppm` mass shift;
  (iv) recall/candidates-per-query on synthetic data with known ground truth.

Synthetic-data unit tests are the right home for all of this and are cheap.

### 6.2 Real-world utility scenarios (is it *useful*, not just *correct*)

Distinct from the correctness gate above: does the index earn its keep? Two
standing methodological rules for **all** scenarios:

- **Always vs. a matched null.** A raw collision count is meaningless — buckets
  collide by birthday paradox. Report collision rate **above a null that
  preserves** RT, isolation window, intensity/nnz, and candidate volume (not
  just shuffled ids) — otherwise the "enrichment" is an artifact of those
  confounders.
- **CCS agreement = orthogonal signal, NOT identity ground truth (from
  review).** Mobility consistency is a strong *enrichment/precision* signal,
  but two peptides can share mobility — it doesn't prove identity. **Circularity
  warning:** if mobility is also a *hard gate* (§4.6), grading precision by
  mobility agreement is partly circular. So report mobility agreement **both
  before and after** the gate, and let **S1 (real labels)** be the actual
  ground truth.

Scenarios, cheap → real (★ = first cut):

- **★ S1 — Simulated DIA, exact ground truth (run FIRST — the decider).**
  TimSim a run from a known peptide list; query with Prosit-predicted bags for
  present vs. absent peptides → ROC / recall@FDR with **exact** labels. If it
  fails here, real data is hopeless. This is *the* ground-truth test; S2's
  mobility agreement is not a substitute.
- **★ S4 — Pre-filter vs. brute force (the value claim).** LSH candidates+verify
  vs. exhaustive exact cosine/containment: plot **recall-of-exact-top-k vs.
  wall-clock speedup** over `(b, r)`. If "same speed, worse recall," it dies
  here — honestly. Rides along with S1 (shared ground truth).
- **★ S2 — Two-DIA-dataset collisions (corroborating real-data evidence, not a
  label substitute).** Index run A, query with run B's units, count *verified*
  collisions vs. a matched null (above). Variants: technical replicates (recall
  ceiling), different samples (specificity floor), dilution series (collisions
  should track shared content monotonically). Grade with mobility agreement
  **before and after** the gate (§6.2 circularity note).
- **S3 — DDA→DIA transfer.** Confident peptide IDs from a DDA run (existing
  `imspy-search`/sage) → query the DIA index → recall = fraction found.
  Orthogonal real labels, no simulator.
- **S5 — Self-consistency / feature recovery.** Index one run, query with its
  own units; do buckets collapse into coherent RT–IM–m/z features? Cross-check
  vs. existing DIA clustering (`candidates.rs`) as reference labels (dedup /
  non-redundant-library use case).
- **S6 — Chimericity stress.** Inject contaminating peaks / mass shifts /
  dropout into real units; watch match degrade. Grades cosine-vs-containment
  and `tol_ppm` on real data; feeds the Phase-1.5 kill-switch.

Most of this is **wiring existing rustims pieces** — TimSim (ground truth),
Prosit + CCS predictors (query bags + expected mobility), sage DDA IDs
(orthogonal labels), existing DIA clustering (reference features), CCS
reproducibility (label-free grading) — not new science.

## 7. Open questions for review

*Settled in this revision (kept for the reviewer's trace):* product framing →
spectral-match candidate-generation pre-filter, not search replacement
(§0, §2.1); unit → mobility moving-average window, width ≈ IM FWHM (§4.1);
containment-vs-cosine → pluggable verify metric, decided in Phase 1.5
(§4.1.2); `feature_id` → `i64` end-to-end (§4.2); per-bit/band/candidate
probabilities → measured separately (§4.2); verification storage → CSR (§4.5);
projection → Gaussian default (§4.2); banding `32×20` → swept (§4.4.1); build
path → own `mscore` scan-slice iterator, not `to_tims_spectra`/`rustdf` helper
(§4.3); converter accuracy must meet `tol_ppm` (§4.4); DIA isolation-window
metadata + query filter (§4.4, §4.6).

Still open (for Phase 1.5 to resolve empirically):

1. **Window width `w`** — the IM-FWHM anchor and the signal-vs-chimericity
   knee (§4.1).
2. **Cosine vs containment** — which verify metric wins at the chosen `w`
   (§4.1.2).
3. **`tol_ppm` / kernel** — tolerance width, kernel shape, `bin_ppm=tol_ppm/3`
   ratio (§4.1.1).
4. **Transform** — sqrt vs log1p vs none.
5. **Index size at scale** — CSR + band postings memory after windowing+splat;
   is it affordable, or do we need the longer-SimHash fallback (§4.5)?
6. **Decision rule** — threshold + empirical FDR, isolation-window-scoped
   (§4.6).
7. **Query source** — predicted fragment bags, cross-run units, or same-run
   self-probes (decides tof-direct admissibility, §4.1.1).
8. **Multi-probe** — Hamming-radius expansion vs shorter bands / more tables.
9. **RT axis** — is cross-frame averaging (SNR) or persistence/intersection
   (sharpening) worth it beyond v1 (§2)?
10. **Physics gate params (§4.6)** — mobility tolerance `Δm`; query-mobility
    source + charge assumption for peptide queries; whether the gate is hard
    filter, soft rank term, or both.
11. **Relative-RT gating** — which normalized-RT space (iRT / indexed-RT /
    gradient-fraction) and what alignment step; keep cross-run RT gating off
    until it exists.

## 8. Grounding references (verified in code)

- Sparse frame layout (SoA, one entry/peak):
  `mscore/src/timstof/frame.rs:143-150` (`TimsFrame`), `:62-68` (`ImsFrame`).
- Per-scan explode: `TimsFrame::to_tims_spectra` `frame.rs:239-265`;
  `TimsSpectrum` `mscore/src/timstof/spectrum.rs:16-24`; sparse peaks in
  `IndexedMzSpectrum`/`MzSpectrum` `mscore/src/data/spectrum.rs:500-504,
  86-90`.
- Reader API: `TimsData` trait `rustdf/src/data/handle.rs:220-227`
  (`get_frame`, `get_slice`, `get_frame_count`); no frame iterator (1-based
  ids). MS type mapping `handle.rs:674-680`; `MsType`
  `mscore/src/data/spectrum.rs:31-64`.
- Threading constraint: lazy `get_slice` serial `handle.rs:817`;
  `uses_bruker_sdk()` `handle.rs:1283`. Rayon idiom `slice.rs:64-77`.
- Module placement: `mscore/src/algorithm/mod.rs:1-3`,
  `mscore/src/timstof/mod.rs:1-7`, `rustdf/src/cluster/`.
- Existing sparse unit in cluster path: `ScanSlice`
  `rustdf/src/cluster/cluster.rs:40-64`.
- Existing (competing) geometric candidate gen: `candidates.rs:1185-1515`
  (`query_precursor`), exact-key grouping `pseudo.rs:86-96, 189-325`,
  `candidates.rs:1029-1061`.
- Dense-window impl being superseded:
  `packages/imspy-predictors/src/imspy_predictors/hashing.py:22-256`
  (unused outside `tests/test_hashing.py`).
