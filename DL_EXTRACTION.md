# Plan — lift the deep-learning predictors out of imspy into their own repo (train/inference split)

**Goal.** Cut the timsim v2 pipeline's last cord to the imspy monorepo. Today the DAG's three predictor
steps (`timsim-ccs/rt/fragments`) are entry points in **imspy-simulation**, so installing the pipeline
drags in the whole imspy stack (v1 simulator + EVAL + GUI, imspy-dia, imspy-search, imspy-vis). Extract
the deep-learning code (`imspy-predictors`, ~11.7k LOC) into its **own repo**, split **inference** (lean:
load weights + predict, or Koina remote) from **training** (heavy: datasets + loops + sagepy), and have a
thin `timsim-predict` package expose the 3 CLI jobs against the inference package.

North star: `pip install timsim-predict` pulls only the inference stack (torch *optional*), never the
imspy dead weight.

## The governing invariant (codex correction — read first)
The boundary is a **one-way runtime dependency graph, not a method-name split**: **inference must never
import training**; training may import inference's shared internals. Classify every symbol by *runtime
capability + call sites* (does it load a model / build inference features / tokenize / post-process / serve
a public predict API? → inference — *even if named `predict_*` or `fine_tune_*`*), not by its name. The
proof is not a static grep: it is (a) a clean-venv **closure** test and (b) **every public inference method
executed with the training packages deliberately absent**. "imspy-independent" means *independent of the
imspy application stack* (sim/dia/search/vis) — keeping the published `imspy_connector` pyo3 wheel and
possibly a thin `imspy-core` is compatible with that goal only if their *resolved closures* pull no
monorepo application packages (verify in Stage 0).

## What the mapping pass established (facts, not guesses)

**Train/inference boundary is per-method in the predictor files.** Clean training-only files:
`training.py` (1271), `data_utils.py` (589, HuggingFace `datasets`), `losses.py` (419, unconditional
torch), `utilities/simple_tokenizer.py`, `utilities/hf_tokenizers.py`. But `ccs/rt/intensity/ionization/
predictors.py` each interleave `predict/simulate_*` (inference) with `fine_tune_model` + sagepy
`predict_*` PSM helpers (training) — the cut is per-method there, the trickiest mechanical work.

**Inference deps (lean):** torch (**optional** `[local]`, already guarded by `require_torch`), koinapy
(**optional** `[koina]`, lazy), numpy/pandas/scipy/numba/tqdm, upstream `chronologer` (optional, default
RT backend). Plus three coupling knots (below).

**Three coupling knots to cut for a clean inference package:**
1. **imspy-simulation** — `flatten_prosit_array` is on the fragment **inference** path (called
   unconditionally in `Prosit2023TimsTofWrapper.simulate_ion_intensities`, +6 sites). → **copy the
   function in** (it's a pure array reshape). Severs the only inference→imspy-simulation edge.
2. **imspy_connector** (the Rust wheel) — `utilities/tokenizers.py::ProformaTokenizer` uses
   `imspy_connector.py_ml_utility`; every **local torch** predictor tokenizes through it. The **Koina**
   path is connector-free. → Keep `imspy_connector` as a dep of the `[local]` extra (it's the published
   pyo3 wheel — a real dependency, not dead weight). Not droppable without reimplementing the tokenizer.
3. **imspy-core** — used in only 4 spots on the inference path: `ccs_to_one_over_k0`,
   `one_over_k0_to_ccs`, `calculate_mz` (chemistry); `linear_map`, `remove_unimod_annotation`,
   `tokenize_unimod_sequence` (utils); `PeptideSequence`, `PeptideProductIonSeriesCollection` (data,
   used by intensity). Shallow + localized → **vendorable** (see D1).

**Pretrained weights:** GitHub-Releases based (NOT HuggingFace) — `github.com/theGreatHerrLebert/rustims/
releases/download/models-v0.5.0/`, SHA-256 verified in `pretrained/hub.py`, with bundled `.pt` fallbacks
in `pretrained/` (`ccs|rt|charge|intensity/best_model.pt`, `pretrained_encoder.pt`, `unimod-vocab.json`).
Chronologer base weights fetch from upstream `searlelab/chronologer`.

**Dead deps to drop on the way:** `scikit-learn` (a *core* dep, never imported) and `wandb` (a
`[training]` extra, never imported — training logs via TensorBoard).

## Stage 0 — RESULTS (audit complete; decisions resolved)
- **S0a closures CLEAN ✅** — `imspy-core` deps only `imspy-connector`; `imspy_connector` is a pure pyo3
  wheel (no imspy Python deps). Neither pulls sim/dia/search/vis. So a thin imspy-core dep re-introduces
  **no** application dead weight.
- **S0b data classes are CONNECTOR-BACKED** — `PeptideSequence` + `PeptideProductIonSeriesCollection` are
  `RustWrapperObject`s wrapping Rust types (not value objects) → **not vendorable cheaply**. They're used on
  the **intensity/fragment** path (both local AND Koina). Since that path already needs imspy_connector
  (local: tokenizer; both: these classes), keeping imspy-core there is free of new weight. → **D1 RESOLVED
  = (a):** vendor the 6 pure leaf funcs (CCS + RT paths become imspy-core-free); **fragments/intensity
  keeps imspy-core + imspy_connector** (essentials, clean closure — not dead weight). Full decouple of the
  fragment path (reimplement the ion-series builder in pure Python) is deferred D1(b), not now.
- **S0c boundary CONFIRMED (by capability, not name)** — INFERENCE = `simulate_*`, `predict` (DL),
  `predict_*_with_koina`, `ModelFromKoina.predict`, `predict_intensities`. TRAINING/SEARCH = `fine_tune_*`
  and the sagepy PSM helpers `predict_inverse_ion_mobility` / `predict_retention_time` (consume PSM
  collections via `get_sagepy_*`). The 3 jobs call only the inference set. `*_with_koina` is inference
  despite the `predict_` name.
- **S0d weights = 146M** (charge 62M, intensity/encoder 24M each, ccs/rt 21M each, vocab 27K) → **D5
  RESOLVED:** do NOT bundle in the wheel; download-cache from `pepdl` GitHub releases (SHA-256), source-tree
  `.pt` as dev fallback only. The pyproject already excludes `.pt` from wheels.
- **Independence-gate refinement:** the leanest install still legitimately pulls `imspy-core` +
  `imspy_connector` (the fragment path's connector-backed classes). So gate ① asserts **"no imspy
  APPLICATION packages (sim/dia/search/vis)"**, NOT "no imspy distributions at all." imspy-core/connector
  are allowed essentials with a clean closure; the dead weight is what must be absent.

## DURABLE ARCHITECTURE (v2 — supersedes the imspy-core-dep design below)
Design guidance (David): *build it to survive the test of time; duplicating a few functions or standing up
a small shared Rust/Python primitives lib (à la `spectrum_utils` / Koina) is an acceptable cost of the
split; decouple from pure timsTOF logic.* So `pepdl` does **not** sit on `imspy_connector`/`imspy-core`
(which drag TDF/timsTOF/sim); it sits on a NEW lean wheel:

**`«mscore-py»` — a pyo3 wheel over `mscore` + `ms-chem` ONLY** (the R4-published crates). Exposes exactly
the MS-general primitives the DL path needs — the ProformaTokenizer (`py_ml_utility`), peptide + product-ion
series (`py_peptide` → `mscore::data::peptide`), mass/mz + isotopes (`py_chemistry`), amino-acids/elements/
sumformula/unimod/constants/mz_spectrum. **Closure carries NO `ms-io`/`timsim-*`** — verified: the
connector's peptide/chem/tokenizer modules are backed solely by `mscore`/`ms-chem`; the timsTOF/TDF/sim
modules (`py_dataset/dda/dia/feature/pseudo/quadrupole/tims_frame/tims_slice/acquisition/simulation`) are
the ms-io/timsim-backed set that `mscore-py` simply omits.

**`mscore-py` is a real, versioned public API — not a copy-paste convenience layer (codex).** To avoid
semantics drifting between two wheels: factor the shared pyo3 binding *source* (`py_peptide`, `py_chemistry`,
`py_ml_utility`, …) into a **shared Rust module/crate that BOTH `mscore-py` and `imspy_connector` compile**
(one source of truth; a bug fix lands once). Give `mscore-py` a **distinct Python namespace** (not
`imspy_connector`'s module paths) so nothing mixes the two by accident.

**Two-types coexistence rule (codex — enforced, not assumed).** Even sharing Rust source, a process importing
BOTH wheels gets two *distinct* Python `PeptideSequence` classes (isinstance / pyo3-extract / pickle / dict-
by-type won't interoperate). Safe only if enforced: `pepdl` never accepts or returns `imspy_connector`
peptide/ion objects; all cross-boundary interchange is **strings / ProForma / primitive arrays**, never
duck-typed objects; a **coexistence test** imports both wheels and asserts `pepdl` runs without consuming
connector objects (clear `TypeError` if mixed). Do not promise future interchangeability from Rust-type
similarity — a later `imspy_connector`-on-`mscore-py` re-layer would be an API migration, not a dedupe.

**This SUPERSEDES D1** (the "keep a thin imspy-core dep for the 2 connector-backed classes" call): those
classes come from `mscore-py` now, so **`pepdl` needs neither `imspy-core` nor `imspy_connector`** — a clean
cut, not a transitional dep. The 6 leaf funcs: `calculate_mz`/isotopes come from `mscore-py`; the pure
physics (`ccs_to_one_over_k0`/`one_over_k0_to_ccs`, `linear_map`, `remove_unimod_annotation`) are vendored
into `pepdl` with provenance + regression vectors (or added to `mscore-py` if they belong in the primitives).

`sagepy-rescore` (existing repo) is now **in scope** (Stage 6): it already isolates its sage↔predictor
adapters (`sage_loader.py`/`features.py`) and only needs *inference*, so adoption = swap its
`imspy-predictors @ git+…rustims` dep for `pepdl` and drop `imspy-search`. The imspy-predictors sagepy PSM
functions (`predict_inverse_ion_mobility` etc.) belong to the *old imspy-search rescoring* path — they stay
there, out of both `pepdl` and `sagepy-rescore`.

### Target architecture

```
<DLREPO> repo  (new, own versioning; name = D4)
├─ <dl>              INFERENCE package — load weights + predict / Koina
│    deps: numpy, pandas, scipy, numba, tqdm
│    [local]  = torch, imspy_connector   (on-device predictors + Rust tokenizer)
│    [koina]  = koinapy                  (remote, torch-free, connector-free)
│    [rt]     = chronologer              (default RT backend)
│    (imspy-core: vendored OR dep — D1)
└─ <dl>-train       TRAINING package (own extra or sibling)
     deps: torch, datasets, tensorboard, sagepy, imspy-search

timsim-predict  (rustims-adjacent or in <DLREPO>? — D3)
     the 3 CLI jobs (ccs/rt/fragments) + the 63-line `resolve` helper
     deps: <dl> (inference), numpy/pandas/pyarrow/scipy
     exposes: timsim-ccs, timsim-rt, timsim-fragments

necroflow DAG → timsim-ccs/rt/fragments  (unchanged command names; new provider)

## THREE consumers of pepdl (not two) — the rescoring facet
The DL predictions serve **simulation, rescoring, AND training** — three consumers of one inference core:
- **timsim-predict** (simulation) — parquet I/O → the necroflow pipeline.
- **sagepy-rescore** (rescoring, EXISTING repo `theGreatHerrLebert/sagepy-rescore`) — PSM I/O + mokapot.
- **pepdl-train** (training) — datasets + loops.

**Consequence for the boundary:** the `sagepy predict_*` adapters (`predict_inverse_ion_mobility`,
`predict_retention_time`, `associate_fragment_ions_with_prosit_predicted_intensities`, the
`get_sagepy_*` bridges) are **rescoring** code, NOT training and NOT core inference. They **leave `pepdl`
entirely** — the inference core stays 100% sagepy-free (no `get_sagepy_*`, no sagepy in any closure). Their
home is the **rescoring layer** (`sagepy-rescore`, which then consumes `pepdl` for the raw predictions,
exactly like `timsim-predict` does). This is cleaner than my earlier plan of folding them into `pepdl-train`.

**Scope call (D6):** the *initial* extraction ships `pepdl` + `pepdl-train` + `timsim-predict` and simply
*excludes* the sagepy adapters from `pepdl` (they stay put in imspy-search/-predictors for now, untouched).
`sagepy-rescore` adopting `pepdl` (and absorbing those adapters) is a **clean follow-on** — same pattern,
separate PR — so it doesn't expand the critical path. Design `pepdl`'s inference API to be rescoring-friendly
(peptide→prediction primitives that a PSM adapter can wrap) so that adoption is a thin layer later.
Open item: confirm `sagepy-rescore`'s current deps (does it already pull imspy-predictors?) when we get to it.
```

## Decisions to make (with recommendations)

**D1 — vendor imspy-core, or keep it as an inference dep? ⟵ the clean-cut decision.**
- The 6 leaf *functions* are trivially vendorable (chemistry constants + string utils). ✅ vendor.
- The 2 *data classes* (`PeptideSequence`, `PeptideProductIonSeriesCollection`) are the real question —
  they are richer imspy-core types (methods, possibly connector-backed). Vendoring them may pull more.
- **(a) RECOMMENDED — vendor the 6 functions; keep `imspy-core` as an explicit *transitional* dep ONLY
  for the 2 data classes on the local torch intensity path, with a stated removal milestone.** Result: the
  Koina inference path is 100% imspy-free; the local path keeps one narrow, time-boxed imspy-core dep.
  Valid ONLY if Stage 0 confirms imspy-core's resolved closure pulls no sim/dia/search/vis. Not a "clean
  cut" — the package isn't called imspy-independent until local inference installs with no imspy-core.
- (b) vendor everything incl. the 2 classes → zero imspy-core, but a deeper copy (assess the classes
  first; risk of dragging connector-backed methods).
- (c) keep imspy-core wholesale → simplest, but the inference package still names imspy — weakest cut.

**D2 — Koina-first or local-first inference?** Both ship; question is the *default* the pipeline uses.
The Koina path is dramatically leaner (no torch/connector/weights). Recommend: inference package supports
both; **timsim-predict defaults to whatever the necroflow config already selects** (it already picks
Koina HCD for fragments in some configs) — no behavior change, just a leaner install when Koina is used.

**D3 — where does `timsim-predict` (the 3 jobs) live?** (a) in `<DLREPO>` as a second package, or (b) its
own small repo, or (c) a `flow/`-adjacent package. Recommend **(a)** — keep jobs next to the inference lib
they wrap; one repo to version. (The DAG only cares about the entry-point names.)

**D4 — repo + package name. ✅ DECIDED: `pepdl`** (peptide deep learning). Names verified free on PyPI +
GitHub. So throughout this doc: `<DLREPO>` = `pepdl` repo (github.com/theGreatHerrLebert/pepdl); `<dl>` =
`pepdl` (inference package); training = `pepdl-train`; the 3 CLI jobs = `timsim-predict`.

**D5 — pretrained-weight hosting.** Weights are GitHub-Release assets on the *rustims* repo. On extraction,
re-host them as releases on `<DLREPO>` (and update `hub.py`'s base URL + SHA registry), OR keep pointing at
the rustims release (works, but leaves a cord). Recommend re-host on `<DLREPO>` for a clean break; the
bundled `.pt` fallbacks keep it working in the interim.

**D6 — sagepy-rescore scope. ✅ DECIDED: in scope (Stage 6).** It installs `imspy-predictors @ git+…rustims`
+ `imspy-search` today (the dead weight). Key fact (resolves the codex contradiction): **`sagepy-rescore`
already OWNS every adapter it executes** — its own `sage_loader.py`/`features.py` do the sage-PSM↔predictor
bridging, and it uses `sagepy.core.ml.mobility_model.predict_sage_im` (from `sagepy`, not imspy). It does NOT
use imspy-predictors' sagepy PSM functions (`predict_inverse_ion_mobility` etc.) — those belong to the
*separate* imspy-search rescoring path and stay there, untouched. So adoption = swap `imspy-predictors(git)`
→ `pepdl` (inference), and drop `imspy-search`. **Gated (codex):** an import/dependency audit of the whole
repo (incl. optional paths + CLI entry points) confirming **no direct runtime import of `imspy-search`**
(its declaration is only there to satisfy imspy-predictors' *fine-tune*, which rescoring never calls), then a
clean-venv test running the rescoring path with both `imspy-search` and `imspy-predictors` **absent**.

**D7 — the shared wheel's name + home.** Home = **mscore foundation repo** (it binds `mscore` + `ms-chem`).
Name candidates (verify PyPI/GitHub-topic free): `mscore` (clean PyPI name for the crate's bindings, mirrors
`sagepy`↔`sage`), `pymscore`, `mscore-py`, `pepprim`. **Open — David.** Placeholder `«mscore-py»` used above.
Note: some pyclass duplication with the full `imspy_connector` is accepted per guidance; a later refactor
could re-layer `imspy_connector` on top of this wheel to dedupe, but that is out of scope here.

## Staged execution plan (v2 — with the shared wheel + sagepy-rescore)

- **Stage 0 — audit. ✅ DONE** (see Stage 0 RESULTS). Closures clean; data classes connector-backed;
  boundary = capability; weights 146M. Superseded conclusion: with `mscore-py`, `pepdl` needs no imspy-core.
- **Stage 1 — build the `«mscore-py»` lean primitives wheel (D7).** New pyo3 crate in the **mscore
  foundation repo** (it's the Python binding of `mscore` + `ms-chem`). Include only the mscore/ms-chem-backed
  pyo3 modules (`py_peptide`, `py_ml_utility`, `py_chemistry`, `py_amino_acids`, `py_elements`,
  `py_sumformula`, `py_unimod`, `py_constants`, `py_mz_spectrum` — reused from `imspy_connector`); Cargo deps
  = `mscore` + `ms-chem` ONLY. Add the pure physics helpers (`ccs_to_one_over_k0`/`one_over_k0_to_ccs`) if
  they belong in primitives. **Closure gate (codex — Cargo deps alone are insufficient):** `cargo tree -e
  features -i ms-io` and `-i timsim-*` return nothing across *every shippable feature profile* (inspect
  `mscore`'s default features — ensure no convenience feature pulls I/O); a **CI policy test fails on any
  forbidden package** in the resolved graph; and `pipdeptree` on the **installed wheel from a clean venv**
  shows no ms-io/timsim. Then: wheel builds; Python smoke-test tokenizes + builds a product-ion series +
  `calculate_mz`. Publish to PyPI (D7 name) BEFORE `pepdl` depends on it.
- **Stage 2 — `pepdl` inference package (seam+copy+REPOINT, one pass).** The files carry *unconditional*
  training imports, so a naive copy is unimportable — do the seam pass WITH the copy: push heavyweight/
  training imports behind training-only methods or into dependency-neutral `_model_common.py`/`_features.py`;
  **repoint the tokenizer + `PeptideSequence`/product-ion classes from `imspy_core`/`imspy_connector` →
  `mscore-py`** (the one real redesign); vendor the pure funcs (`linear_map`, `remove_unimod_annotation`)
  + copy `flatten_prosit_array` (each with provenance + regression vectors); drop sklearn/wandb.
  **Intensity-repoint contract tests FIRST (codex):** before swapping the ion-series class, characterize the
  current `imspy_core` object's behavior on representative peptides (modified residues, terminal mods, charge
  states, neutral losses, empty/no-product cases) — series membership + ordering, labels, m/z, charge, and the
  exact arrays/dtype/None-handling passed to the model/postprocessor — and assert `mscore-py`'s equivalent
  matches. **Gate:** those contract tests pass; `import pepdl` + a Koina predict smoke-test succeed in a venv
  with **NO imspy package at all** (mscore-py replaces imspy-core/connector).
- **Stage 3 — enforce the one-way graph.** Move training-only public methods (`fine_tune_*`) into
  `pepdl-train`; they import inference's shared internals, never vice-versa; inference classes stay intact
  (no cross-package split, no edge-reversing base class). **Gate:** every public inference method runs with
  `sagepy`/`datasets`/`tensorboard`/imspy-search **absent** — no `ImportError`.
- **Stage 4 — `timsim-predict` jobs + parity.** Move `ccs/rt/fragments.py` + `resolve` in; repoint imports
  at `pepdl`; expose `timsim-ccs/rt/fragments`; emit **model provenance** into outputs. **Parity gate
  (canonicalized, not byte-identical):** schema, row keys+order, null behavior, exact categorical/id values,
  per-column tolerances vs the imspy-simulation entry points — local pinned to one env, remote to a named
  Koina model revision.
- **Stage 5 — flip the DAG + prove independence, from a WHEEL.** Point `flow/timsim_flow.py`'s three
  commands at the new entry points. **Two gates, each from a built wheel:** ① `pip install
  timsim-predict[koina]` → **zero imspy distributions** (mscore-py is not imspy), no torch; all remote nodes
  run. ② `pip install timsim-predict[local,rt]` → local nodes run; **zero imspy distributions**; torch
  present. A missing `[local]` at local-mode selection raises an **actionable install error**; Koina stays
  importable without it.
- **Stage 6 — publish, sagepy-rescore adoption, weights re-host.** Publish `mscore-py` (done Stage 1),
  `pepdl` (+ `-train`), `timsim-predict` to PyPI. **sagepy-rescore:** swap its
  `imspy-predictors @ git+…rustims` dep → `pepdl`, drop `imspy-search`, keep its own `sage_loader`/`features`
  adapters; **rescoring parity in TWO tiers (codex):** first pin **raw prediction + feature parity** (the RT/
  IM/intensity values and derived rescoring columns), THEN mokapot q-value parity — q-values drift from tiny
  score/dep changes, so the prediction/feature contract is the real gate, pinned to a named model version.
  **Weights (D5):** re-host as `pepdl` releases (SHA-256 + version), rustims URL a time-limited
  tracked mirror; **no Git LFS in wheels**, prefer download-cache over fat bundled `.pt`, no silent-old-model
  substitution. Leave a deprecation **shim** for the old imspy-simulation predictor entry points, then remove.

## Must-specify before Stage 1 (codex — underspecified items)
- **Distribution/namespace:** is `<dl>-train` a separate wheel or a `[training]` extra of `<dl>`? its
  version-compat constraint with `<dl>`; which package owns the shared `_model_common`/`_features` code.
- **Package data:** `.pt`, `unimod-vocab.json`, any templates declared in `pyproject` and loaded via
  `importlib.resources` (not path imports) — verified against a built wheel.
- **Model cache contract:** location (`$IMSPY_CACHE_DIR`→rename?), offline behavior, concurrent-download
  locking, checksum-failure behavior, and migration of the existing `~/.cache/imspy/models/*` cache.
- **CLI extra UX:** a config selecting local inference must pull `[local]` or fail with the exact install
  command; document per-CLI which extras each mode needs.
- **License audit:** redistribution rights for the model weights, the copied `flatten_prosit_array`,
  vendored imspy-core funcs, Unimod vocab, Chronologer assets, and the Rust tokenizer.
- **Migration policy:** shim for existing `imspy_predictors`/`imspy_simulation` imports + the CLI names
  during the DAG cutover.
- **`mscore-py`↔`pepdl` API-compat policy (v2):** semver + min/max version bounds; how tokenizer/model
  feature semantics are versioned so a wheel bump can't silently change predictions.
- **Canonical ProForma/Unimod version (v2):** one vocabulary version + shared regression fixtures across
  `mscore-py` and `imspy_connector` — tokenizer vocab *ordering* is model-compatibility, not a detail.
- **`mscore-py` namespace (v2):** distinct Python module paths from `imspy_connector` (safer than matching
  names, which invite accidental mixed-type imports).

## Pre-existing bugs in copied primitives (backlog — fix at the shared-source-crate refactor)
Surfaced by the Stage-1 codex review; verified byte-identical to the `imspy_connector` originals (faithful
copies, NOT introduced here) and **off the pepdl DL path**. Fixing them in `mscorepy` only would diverge it
from `imspy_connector` — so fix both at once when the pyo3 binding source is factored into a shared crate:
- **[P2] `py_chemistry::calculate_transmission_dependent_fragment_ion_isotope_distribution`** — builds the
  transmitted-isotope set from `transmitted_isotopes`'s *length* only, ignoring its m/z/intensity, so any
  same-length spectrum yields indices `0..n` (can't express a non-contiguous selection). Quadrupole fn, not
  a DL primitive.
- **[P2] `py_spectrum_processing` (all three fns)** — `as_slice().unwrap()` panics (`PanicException`) on
  strided/reversed NumPy inputs (`mz[::2]`); should propagate the error or copy. Not a DL primitive.

## pepdl Stage-2 codex findings (all pre-existing copies, verified byte-identical to imspy-predictors)
- **[P2] fine_tune_model imports `pepdl.losses`** (intensity/predictors.py:972,1273) — RESOLVED BY STAGE 3
  (fine_tune → pepdl-train, which owns losses). Lazy import, so the inference gate is unaffected.
- **[P1] koina-intensity column rename is backwards** (intensity/predictors.py:~601) — renames
  `peptide_sequences`→`seq_col` so Koina's required columns go missing → `ModelFromKoina.predict()` raises.
  Pre-existing on the koina INTENSITY path (my Stage-2 live test covered koina RT, not intensity). **Verify +
  fix in Stage 4** with a live koina-intensity test against the real fragments-job usage.
- **[P1] single-peptide intensity squeezes the batch axis** (intensity/predictors.py:~512) — `reshape_dims`
  →(1,29,6), `squeeze`→(29,6) so the flatten treats 29 rows as separate predictions. Pre-existing; **verify +
  fix in Stage 4** (same live-koina-intensity test; guard: don't squeeze the batch dim).

## Open risks
- **Stage 2 surgery** — predict and fine_tune share intra-class helpers; the fix is `_model_common`
  extraction, not a shared base class (which could reverse the dependency edge). The public-method-with-
  training-absent test is the backstop.
- **D1 closure** — if imspy-core's resolved closure isn't clean, "thin dep" becomes a re-entry of the
  monorepo; Stage 0 decides vendor-more vs keep-transitional (with a removal milestone).
- **Vendored-function drift** — Unimod parsing / unit conventions are domain contracts; provenance markers
  + regression vectors are mandatory, not optional.
- **imspy_connector platform risk** — the `[local]` wheel must be validated for every supported
  Python/OS/arch (incl. the pipeline's deployment target) in CI.
- **Parity semantics** — canonicalized comparison with pinned model versions; never assert byte-identity.
