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

## Target architecture

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

## Staged execution plan

- **Stage 0 — closures + capability audit (gates the whole design).**
  (a) Run a clean dependency resolution of `imspy-core` and `imspy_connector`: confirm neither's *resolved
  closure* pulls imspy-simulation/dia/search/vis (if it does, D1 must vendor more). (b) Read
  `PeptideSequence` + `PeptideProductIonSeriesCollection`: value objects, or connector-backed / method-
  heavy? Decides D1(a) vs (b). (c) Verify the `sagepy predict_*` helpers are *training/PSM-scoring*, not
  an inference API a predictor needs. (d) Measure the bundled `.pt` sizes vs PyPI wheel limits (weights).
- **Stage 1 — import-seam pass + copy TOGETHER (no copy-then-split).** The files carry *unconditional*
  training imports (`losses.py`, `training.py`, torch in `data_utils.py`), so a naive copy is unimportable.
  In one pass: copy the inference modules; push every heavyweight/training import behind training-only
  methods or into `_model_common.py` / `_features.py` (dependency-neutral private modules — importable by
  both sides, importing neither); copy in `flatten_prosit_array` (with shape/dtype/order edge tests +
  provenance); vendor the 6 imspy-core funcs (each with a `# vendored from imspy-core@<ver>` marker,
  license note, and a regression vector); drop sklearn/wandb. **Gate:** `import <dl>` and a Koina predict
  smoke-test succeed in a venv with **no imspy application packages installed**.
- **Stage 2 — enforce the one-way graph.** Move training-only public methods (`fine_tune_*`, and any
  Stage-0-confirmed sagepy training helpers) into the training package; they import inference's shared
  internals, never vice-versa. Keep inference classes intact (no cross-package class split; no shared base
  that reverses the edge). **Gate:** a test enumerates *every public inference method* and runs it with
  `sagepy`/`datasets`/`tensorboard`/imspy-search **absent** — no `ImportError`.
- **Stage 3 — `timsim-predict` jobs + parity.** Move `ccs.py/rt.py/fragments.py` + `resolve` in; repoint
  imports at `<dl>`; expose `timsim-ccs/rt/fragments`; emit **model provenance** (local weight sha / exact
  Koina model+version) into each output's metadata. **Parity gate (canonicalized, not byte-identical):**
  same schema, row keys + order, null behavior, exact categorical/id values, and per-column numerical
  tolerances vs the imspy-simulation entry points — local parity pinned to one controlled env, remote
  parity pinned to a named Koina model revision.
- **Stage 4 — flip the DAG + prove independence, from a WHEEL.** Point `flow/timsim_flow.py`'s three
  commands at the new entry points. **Two independence gates, each from a built wheel (not the source
  checkout):** ① `pip install timsim-predict[koina]` → no torch, no connector, no imspy distributions;
  remote nodes run. ② `pip install timsim-predict[local,rt]` → local nodes run; resolved env contains no
  simulation/dia/search/vis. A missing `[local]` at local-mode selection must raise an **actionable
  install error**, while Koina stays importable without it.
- **Stage 5 — weights re-host (D5) + publish.** Re-host weight assets as `<DLREPO>` releases (SHA-256 +
  model-version metadata); keep the rustims URL only as a **time-limited, tracked** mirror. **No Git LFS in
  wheel contents** (ships pointers). Prefer a small default wheel + first-use download cache (or a
  separately versioned assets wheel) over fat bundled `.pt`; bundled fallbacks must be architecture/vocab-
  compatible and must **not silently substitute an older model** on download failure. Publish `<dl>` (+
  `-train`) + `timsim-predict` to PyPI; leave a deprecation **shim** for the old imspy-simulation entry
  points during the transition, then remove.

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
