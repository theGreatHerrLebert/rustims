# NEXT_STEPS — LSH / dataset-similarity work (resume doc)

Written 2026-07-03. Everything is committed + pushed to branch
**`feat/lsh-spectral-index`** (HEAD around `b1be304e` + the commit adding this file).
Full running log of findings is in **`LSH_HANDOFF.md`** (same branch, root) — this file
is the operational "where things are + what to do next" summary. Read both.

---

## 0. Where / how to work

- **Machine:** `monster3` (120 cores, RTX 5090 32 GB). Reached via `ssh monster3`
  (user `dateschn`). No direct shell — wrap commands in `ssh monster3 '...'`.
- **Repo:** `/globalscratch/dateschn/rustims`  (branch `feat/lsh-spectral-index`,
  all pushed). `git fetch && git checkout feat/lsh-spectral-index && git pull`.
- **Rust build:** `source ~/.cargo/env; cargo build --release -p rustdf` (rustc 1.96).
- **Reader is SDK-free by default here:** harnesses open with
  `TimsDatasetDIA::new("NO_SDK", path, in_memory, false)` → `BrukerFormulaConverter`
  (accurate m/z, thread-safe → parallel reads). `uses_bruker_sdk()==false`. Pass
  `mem`/in_memory=true for fully parallel I/O on big builds.
- **Git hygiene:** commit incremental findings and `git push origin
  feat/lsh-spectral-index`. Commit with `git -c user.email=davidteschner@googlemail.com`.
  NEVER add Claude/Anthropic attribution or Co-Authored-By trailers (user rule).

## 1. Data locations (all under `/globalscratch/dateschn/sim_output/`)

Each dataset dir has `<NAME>.d/` (Bruker: analysis.tdf + analysis.tdf_bin) and
`synthetic_data.db` (TimSim exact ground truth: peptides, ions, frame/scan occurrence
+ abundance).

- **Primary (HeLa, 250k peptides, 1h gradient, 32,020 MS2 frames):**
  `DIA-250K-NOISY-CHRONO/`
- **MBR replicate pairs (I made these):**
  - `MBR-PAIR-A/` , `MBR-PAIR-B/` — clean replicates. B = `from_existing` A with ONLY
    RT drift std 3 s + IM drift std 0.003 (same peptides/ions/abundances, **shared
    `ion_id` space** = exact cross-run ground truth). ~20k peptides, 700 s gradient.
  - `MBR-PAIR-A-NOISY/`, `MBR-PAIR-B-NOISY/` — same but `add_real_data_noise=true`.
  - Configs: `sim_output/mbr_pair_{a,b}{,_noisy}.toml`; copies committed in repo at
    `rustdf/examples/mbr/`.
- **Full zoo (15 sets, for dataset-similarity validation):** CLEAN-DIA-YEAST-100K,
  CLEAN-MIDIA-YEAST-100K, DIA-250K-NOISY-CHRONO, MBR-PAIR-{A,B}{,-NOISY}, MIDIA-250K-
  {NOISELESS-FRAG, NOISELESS-NOFRAG, NOISY-CHRONO-FRAG, NOISY-FRAG}, MIDIA-PANICTEST-5K,
  SUPERIMPOSE-{DIA,MIDIA}-YEAST-100K, TIMSIM-MIDIA-150K-24-12.
- Assets: reference blank `/globalscratch/dateschn/raw/dia/blanks/G241217_011_Slot2-2_1_16312.d`;
  HeLa fasta `/globalscratch/dateschn/raw/fasta/hela/uniprotkb_proteome_UP000005640_AND_revi_2024_05_21.fasta`;
  chronologer `/globalscratch/dateschn/primitives/chronologer/models/Chronologer_20220601193755.pt`.

## 2. TimSim (to make more datasets / replicates)

- CLI: `/home/dateschn/micromamba/envs/imspy312/bin/timsim <config.toml>`
  (v0.4.1, GPU). Config sections are FLATTENED at load — only KEYS matter, not section
  names. Run detached: `setsid ... > log 2>&1 < /dev/null & disown`. **Do NOT `pkill`
  in the same ssh command** (kills the session). Wait via `kill -0 $PID`.
- **Replicate pair recipe (template mode):** run A normally; run B with
  `from_existing=true`, `existing_path=<A dir>`, and under `[variation]`:
  `re_scale_rt=false`, `rt_variation_std=<sec>`, `ion_mobility_variation_std=<1/K0>`
  (per-analyte Gaussian drift; NO `intensity_variation_std` = "IM+RT shift only").
  `add_real_data_noise=true` for the realistic-noise version. Each run ~2-4 min.

## 3. Harnesses (`rustdf/examples/`) + exact run commands

```bash
D=/globalscratch/dateschn/sim_output
# self-test / scale (add `mem` for full parallel build)
cargo run --release --example recurrence_selftest -p rustdf -- $D/DIA-250K-NOISY-CHRONO/DIA-250K-NOISY-CHRONO.d 40
# within-run ground-truth recall; label mode dominant|any
cargo run --release --example recurrence_recall -p rustdf -- <path.d> <synthetic_data.db> 2000 dominant
# cross-run MBR: index A, query B.  args: <A.d> <A.db> <B.d> <B.db> [n_frames] [min_int] [top_n|all] [apex]
cargo run --release --example mbr_crossrun -p rustdf -- \
    $D/MBR-PAIR-A-NOISY/MBR-PAIR-A-NOISY.d $D/MBR-PAIR-A-NOISY/synthetic_data.db \
    $D/MBR-PAIR-B-NOISY/MBR-PAIR-B-NOISY.d $D/MBR-PAIR-B-NOISY/synthetic_data.db 100000 100 all
# dataset similarity over a directory of runs.  args: <base_dir> [n_frames] [min_int] [ms_level]
cargo run --release --example dataset_similarity -p rustdf -- $D 800 100 2
```

## 4. Findings so far (condensed — full detail in LSH_HANDOFF.md)

- **Two use cases, different tools:** peptide-ID/membership → inverted feature index
  (NOT this work). Spectrum recurrence/MBR → cosine SimHash (this work).
- **Judge by cosine, not ion labels.** Units are ~8-ion chimeric; "shares an ion" !=
  "similar spectrum" (median true-pair cosine 0.0). Dominant-ion relabel did NOT help.
- **Full build feasible:** 3.06M units, ~4 min, ~35 GB peak RSS (mem mode) on 120 cores.
- **Within-run "recurrence" = elution-neighbour redundancy** (97% of high-cosine pairs
  within 15 s RT), not cross-time recurrence. Good for denoising/feature-building, NOT MBR.
- **Cross-run MBR WORKS (clean pair):** same analyte recurs at median cosine 0.92; LSH
  B->A recall 0.73 (32,16) / 0.57 (64,32), 100% high-cosine at (64,32).
- **Noise collapses recall — but it's a DENOISING problem.** SIM blank noise is low-
  intensity (peak p50=5, p90=68). Fix: drop top-N (keep ALL peaks; squared intensity
  weighting suppresses noise) + a signal floor. Floor 300-1000 fully recovers MBR to
  clean levels BUT over-fits SIM noise → **use a conservative floor ~100** (honest recall
  ~0.23-0.29); on real data use a coherence filter, not an absolute cutoff.
- **Oracle apex-aggregation does NOT raise MBR recall** — the wall is CHIMERICITY, not RT
  redundancy. Apex is still the right index shape (11x fewer units, cleaner candidates).
  If continuing MBR, the only real lever is **dechimerization** (single-analyte spectra) —
  and that will still lose to a peptide-centric search. Hence the pivot below.

## 5. CURRENT DIRECTION: dataset-level similarity search  (the live task)

Goal: quickly say **how close dataset A is to dataset B** (QC, replicate/batch checks,
run clustering). Aggregating a whole run into one fingerprint makes chimericity
irrelevant. **Endgame = keyspace similarity** (compare datasets by SimHash band-key
occupancy — compact, streaming, hash-native).

- **v1 feature-space reference — DONE & VALIDATED** (`dataset_similarity.rs`): per run ->
  L2-normed fingerprint over binned (m/z 0.01 x mobility 0.01), sqrt-intensity, floor 100,
  800 sampled frames; compare by exact cosine + Jaccard. **On the zoo (MS2), every
  designed pair is the nearest neighbour** (replicates 0.55-0.64, templates 0.52-0.64),
  ordering correct: replicate >> same-sample-diff-scheme (~0.18) > diff-sample (0.08-0.14).
  Caveats: (1) HUB BIAS — densest run (SUPERIMPOSE-MIDIA, 2.1M tokens) is spurious NN of
  unrelated sparse sets; (2) MS2 mixes sample+scheme+blank (clean-A<->noisy-A only 0.06).

### DO NEXT (in order; a decision is pending on the last)
1. **MS1 fingerprint** — rerun `dataset_similarity ... <base> 800 100 1` (ms_level=1).
   MS1 precursor landscape should isolate SAMPLE identity (scheme-independent); expect
   yeast-DIA <-> yeast-MIDIA to rise. Quick.
2. **Hub-bias fix** — IDF-style down-weight of ubiquitous tokens (tokens present in many
   datasets get lower weight), or prefer Jaccard. Recompute matrix; SUPERIMPOSE should
   stop being everyone's NN.
3. **Keyspace sketch (the deliverable) — DECISION PENDING.** Represent a dataset by its
   occupancy over the SimHash keyspace. Three granularities:
   - (c) **MinHash over the (m/z,mobility) token set** — compact/streaming version of the
     exact-Jaccard already computed; O(K) compare. `mscore` has MinHash
     (`min_hashes(&[(i64,f32)])`). RECOMMENDED FIRST (smallest step from validated v1).
   - (a) **Per mobility-window-unit SimHash band keys** -> per-dataset key histogram;
     richest / similarity-aware. Do after (c) if we want sharper tiers.
   - (b) one SimHash signature of the whole fingerprint vector — too coarse, skip.
   Show whichever recovers the v1 reference ordering at a fraction of the size.

### Deferred threads (only if we revisit per-spectrum MBR)
- Drift sweep (`rt_variation_std`/`ion_mobility_variation_std`, incl. IM past |Δscan|<=5).
- Dechimerization (feature deconvolution to single-analyte spectra) — the only MBR recall
  lever, but expected to lose to a real peptide-centric search anyway.

## 6. Gotchas
- Floor value is SIM-specific; don't over-tune it on simulated noise. Coherence filter on
  real data.
- Dense datasets create hub bias in cosine similarity.
- `get_slice` on the lazy loader reads serially; use `mem` for parallel I/O.
- CosineSimHash packs n bits into one u64 → n<=64.
- Detached TimSim: `setsid ... & disown`; never `pkill` in the same ssh invocation.
