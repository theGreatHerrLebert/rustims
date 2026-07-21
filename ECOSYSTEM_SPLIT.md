# ECOSYSTEM_SPLIT — from monorepo to a federation of crates + one ecosystem repo

**Status:** sketch (2026-07). Not yet planned to execution. Run through `claudex` before moving files.

## Thesis

`rustims` today is a single Cargo workspace holding the Rust core, the PyO3 binding, and
(historically) everything else. The end state we want:

- A **federation of independently-published, PyO3-free Rust crates** (the primitives).
- One **ecosystem repo** that integrates them into a coherent Python-facing whole and
  orchestrates the DAG. `rustims` becomes *that* — the integration layer, not the
  code-of-record for the primitives.
- **timsim v2 runs Python-free at its core**, shedding v1 dead weight.
- **No forced PyTorch.** Installing the simulator (or the core) must not drag a ~GB torch/CUDA
  stack. torch becomes an opt-in extra for the *local-ML* path only — this is the single most
  painful coupling today, and it is a **Python-packaging** fix, orthogonal to the Rust split.

## Why this started, and what it became

This began as a thin layer to make timsTOF `.d` raw data accessible from Rust. It has grown into a
real ecosystem — core MS types, file I/O, vendor writers, a streaming simulator, ML predictors,
search, viz, a DAG orchestrator. The problem is not the growth; it's that **we ship it all at once,
every time.** The worst symptom: a user who wants to read a `.d` or run a Koina-backed simulation
still installs PyTorch. The goal of this document is the modularity that lets each consumer take
*only what it needs*.

## Capabilities → homes (the organizing spine)

The hard decision is never "how many repos" — it's **along which seams**. Split by *capability*, and
let three forces decide where each seam actually falls:

- **Dependency weight** — does it drag a heavy/awkward dep (torch/CUDA, a vendor SDK, a native
  runtime)? Heavy deps want to live behind a boundary so they're opt-in.
- **Release cadence** — does it churn on its own schedule, independent of the core? Independent
  churn wants an independent version.
- **Audience** — who needs it *alone*? A capability that a consumer wants without the rest earns
  its own installable.

### Precedent: sagepy already does this, in your own codebase

`sagepy` is the worked example. Its core Python package depends on only `sagepy-connector` + numpy +
pandas + tqdm — **no torch**. All the deep-learning weight (torch, xgboost, scikit-learn, mokapot)
lives in the **separate `sagepy-rescore` satellite**. On the Rust side: `unimod` is a zero-dep leaf
crate (modifications), `qfdrust` is a pure-Rust algorithm crate (FDR/rescore, no pyo3), `sage-core`
is *reused* as the engine, and `sagepy-connector` is the single PyO3 wheel binding all three. Every
principle below is already live there — this plan is, in one line, *"make rustims look like sagepy."*

### The map

| Capability | sagepy does it as | rustims target | Seam driver |
|---|---|---|---|
| **Chemistry / modifications** | `unimod` (leaf) | `ms-chem` (leaf) | reuse; today *triplicated* → R1 |
| **Data structures** (spectra, peptides, m/z, mobility) | `sage-core` types | `mscore` (frozen anchor) | stable → publish early (R2) |
| **Raw data access** (.d / .raw / .wiff / mzML) | `mzdata` (reused) | `ms-io`/`rustdf` + `thermo-io`/`sciex-io` | vendor = weight + license → satellites |
| **Algorithms** (FDR, rescore, simulation) | `qfdrust` (pure Rust) | `qfdrust`(!) + `timsim-core` | independent cadence |
| **Database search** | `sage-core` (reused) | reuse `sage` + `imspy-search` (Py) | don't reinvent |
| **Deep learning / prediction** | `sagepy-rescore` (owns torch) | `imspy-predictors` (torch = *extra*) | weight → the P0 template |
| **Visualization** | *(none)* | `imspy-vis` / `tims-viewer` | audience → satellite |
| **Bindings** | `sagepy-connector` | `imspy_connector` | one wheel (pyclass ownership) |
| **Orchestration** | *(CLI scripts)* | `flow/` (necroflow) | ecosystem repo |

### Convergence (forward-looking, not for now)

`qfdrust` **already depends on `rustims`**, and both ecosystems carry their own chemistry/unimod. So
sagepy and imspy aren't separate worlds — they're two consumers that could eventually share a common
foundation (`ms-chem`, `mscore`), so a fix to isotope tables or a spectrum type benefits both. Not a
decision for this plan, but the federation makes it *possible* in a way the monorepo never did.

## The good news: we're ~70% there already

Reconnaissance on the current workspace (2026-07):

- **The core crates carry no PyO3.** `mscore`, `rustdf`, `rustms` have zero `pyo3` dep — every
  `#[pyclass]` already lives in `imspy_connector`. The "keep binding glue in one place, keep the
  crates pure" discipline is *already enforced*. This is the single biggest risk of the whole
  plan, and it's already retired.
- **v2 is already carved into crates.** `timsim-chem`, `timsim-schema`, and a `timsim-cli`
  workspace of ~15 single-purpose binaries (`timsim-proteome`, `-digest`, `-design`,
  `-precursors`, `-yield`, `-modify`, `-frag-input`, `-localization`, `-render`,
  `-render-thermo`, `-spectra`, …). These are the necroflow DAG steps as standalone bins.
- **Vendor weight is already feature-gated.** In `timsim-cli`, `rustdf` and `mscore` are
  `optional = true`, lit by `thermo = ["dep:rustdf", "rustdf/thermo", "dep:mscore"]`. The lean
  default build is already the design intent.
- **The decoupling seam already exists as data.** Render consumes precomputed Parquet
  (`precursors`, `precursor_ccs`, `peptide_rt`, `fragment_intensities`) via `timsim-schema`. The
  sim engine never imports a predictor — property *prediction* and property *consumption* are
  already separated by the artifact contract (the necroflow node boundary).

So this is **topology + publishing + cleanup**, not a ground-up re-architecture. The seams are
cut in the code; they just aren't repo boundaries or published versions yet.

## Load-bearing principles (mostly already true — keep them true)

1. **Split crates stay pure Rust (no PyO3).** Publishable to crates.io; usable by pure-Rust MS
   consumers. *(Already true for mscore/rustdf/rustms.)*
2. **All `#[pyclass]` glue lives in one binding: `imspy_connector`, in the ecosystem repo.**
   *(Already true.)*
3. **One wheel, not N.** Because the bindings mix-and-match Rust types across crate lines
   (rustdf hands back mscore-defined objects, chemistry operates on them), separate per-crate
   wheels would hit PyO3's per-module pyclass-ownership wall. Precedent: **py-polars**
   (`polars-core`/`-lazy`/`-io` are crates; one `py-polars` binds them into a single wheel).
4. **Feature-gate the optional/vendor layers** so the shipped artifact is lean. *(Already the
   pattern in timsim-cli.)*
5. **Property simulation is decoupled from property prediction via the Parquet feature-space
   contract.** The engine consumes artifacts; it never imports the predictor. *(Already true.)*

## Target crate DAG (bottom-up)

```
L0  ms-chem*          elements, amino acids, UNIMOD, sum formulas, isotopes, mass, modify
                      -> DEDUP TARGET: collapses mscore/chemistry + rustms/chemistry + timsim-chem
L1  mscore            spectra, peptides, m/z + mobility, core algorithms          dep: ms-chem
    ms-io (rustdf)    Bruker TDF read/write                                       dep: mscore
L2  thermo-io*        Thermo .raw (thermorawfile)   [optional, leaf, licensed]    dep: mscore, mzdata
    sciex-io*         SCIEX .wiff read + native write (sciexwiff)                 dep: mscore, mzdata
    timsim-schema     the Parquet feature-space contract (the decoupling seam)
    timsim-core*      v2 sim engine: feature-space assembly + streaming render    dep: mscore, timsim-schema
                      (sweep-line; Python-free; vendor-io behind features: tdf default, thermo/sciex/mzml opt-in)
    timsim-properties?  OPTIONAL Rust analytical property models (see open Q)      dep: ms-chem
```
`*` = candidate to become its own repo. `mscore`/`rustdf` may stay in the ecosystem repo longer
(see R4).

## Ecosystem repo (`rustims`) — what stays / integrates

- **`imspy_connector`** — the single PyO3 extension; depends on all L0–L2 crates; defines every
  pyclass; builds the one wheel. This is where "mix and match" happens.
- **Python packages** (`packages/`) — imspy-core, imspy-predictors (the ML/torch property
  predictors), imspy-dia, imspy-search, imspy-simulation (timsim CLI/GUI wrappers), imspy-vis.
- **`flow/`** — the necroflow DAG factory (`timsim_flow.py`) + configs.
- Job: assemble the crates into a coherent Python-facing whole + orchestrate. NOT the
  code-of-record for the primitives.

## The "no Python in the core sim" axis (your idea, made concrete)

The core sim is *already* PyO3-free at the crate level. To make it a hard, shippable guarantee:

- Carve **`timsim-core`** (engine: feature-space assembly + streaming render) out of `rustdf/sim/`
  + the `timsim-render*` bins, as a pure-Rust crate depending only on `mscore`, `timsim-schema`,
  and (optionally, feature-gated) the vendor-io crates.
- Property generation then has **two interchangeable providers, both speaking the `timsim-schema`
  Parquet contract**:
  - **(a) Rust analytical** — a `timsim-properties` crate (fast, approximate, zero Python).
    **DEFER (claudex):** this is speculative product scope, not a prerequisite for federation or the
    torch decoupling. Keep `timsim-schema` *capable* of supporting it, but defer the implementation +
    repo-home decision until there's a validated zero-Python use case demanding it.
  - **(b) Python ML** — `imspy-predictors` (Prosit/torch; high fidelity).
- Because the engine only ever reads Parquet, it imports neither. A user can run a *fully
  Python-free* simulation with provider (a); reach for (b) when fidelity matters. This is just
  *realizing* a boundary that already exists in artifact form.

## The PyTorch axis — Python-side modularity (the most painful coupling)

This is orthogonal to the Rust crate split and can be done **first, independently** — it's a
`pyproject.toml` refactor, no Rust, no repo moves. Current state (2026-07):

- `imspy-core` — torch-free ✅ (numpy + connector only).
- `imspy-predictors` — **`torch>=2.0.0` in hard `dependencies`** (❌), plus a `koina` extra (remote,
  torch-free) and a `cuda` extra.
- `imspy-search` → depends on `imspy-predictors` → inherits torch.
- `imspy-simulation` (timsim) → `imspy-predictors[koina]` + `imspy-search` → **inherits torch
  transitively even on the Koina-only path that never imports it.**

**Fix:** make torch *optional* in `imspy-predictors` — move `torch` out of `dependencies` into an
extra (e.g. `local = ["torch>=2.0.0"]`); the core package keeps only the predictor *interface* + the
Koina (remote) client. Guard every local-model `import torch` behind a lazy import inside the local
runner, so importing `imspy_predictors` never imports torch at module load. Then:

- `pip install imspy-predictors` → torch-free (interface + Koina).
- `pip install imspy-predictors[local]` → pulls torch + the local model runners.
- timsim's `imspy-predictors[koina]` install becomes genuinely torch-free.
- `imspy-search` depends on the torch-free core unless it needs local scoring (then `[local]`).

**Landmines to check** (flag for claudex): any top-level `import torch` in `imspy_predictors/__init__.py`
or in modules imported at package load; entry-point scripts that import a local runner eagerly;
`imspy-dia`'s own hard torch dep (same treatment). The Rust split does **not** address any of this —
worth doing on its own even if the crate split slips.

This axis + provider (a) (Rust-analytical properties) together give a **fully torch-free simulation**
for the common case; torch is pulled only when a user opts into local ML prediction.

## v1 dead weight to shed (needs an inventory pass)

- v1 SQLite-backed, batch frame assembly (superseded by the v2 streaming render).
- `translate_legacy_config` cruft (keep a thin compat shim only if external configs still need it).
- v1-only code paths inside `imspy-simulation` (Python side).
- The triplicated chemistry (folds into `ms-chem`).

## Migration order (revised after claudex, 2026-07)

Two independent workstreams: **(P) Python-package decoupling** and **(R) Rust crate federation**.
P can run fully in parallel with R.

### Workstream P — Python decoupling (start now; no Rust, no repo moves)

- **P0. Torch-optional + import-surface audit.** Move `torch` out of `imspy-predictors`
  `dependencies` into a `local` extra; keep only the interface + Koina (remote) client in the core.
  **Not sufficient by itself** (claudex): audit the whole import graph —
  `imspy_predictors/__init__.py` eagerly imports the CCS/RT/intensity/ionization subpackages;
  legacy sim job modules (`simulate_peptides`, `simulate_retention_time`,
  `simulate_ion_mobilities`, `simulate_fragment_intensities`, …) import local predictor classes at
  top level; entry-point targets import their module before `main`. Make the `rt.py`/`ccs.py`/
  `fragments.py` *dispatch-then-import* pattern the only reachable path; legacy eager paths must
  become unreachable via CLI, GUI, flow, and `__init__`. Define a precise "install
  `imspy-predictors[local]`" error when a local model is requested without torch.
  **Gate (fresh env, no torch installed):** install + `import` every public package; load every
  console entry point; run each Koina path with a mocked client; assert local dispatch fails with
  the install hint; verify `imspy-dia` (own hard torch dep) and `imspy-search` separately.
- **P1. Python dependency-graph decoupling.** `imspy-search → imspy-predictors` and
  `imspy-simulation → imspy-search + imspy-predictors[koina]` need auditing so torch-free installs
  are real end-to-end, not just at the predictors layer.

### Workstream R — Rust federation (gated on R0 first)

- **R0. Release/CI infrastructure (PREREQUISITE — claudex).** Before splitting *anything*: crate
  ownership + semver policy, crates.io/private-registry credentials, MSRV policy, automated
  API/semver checks, and a **dev meta-workspace** using root `[patch.crates-io]` overrides for local
  cross-crate work (path deps in *published* manifests are not viable; patches only apply from the
  top-level invocation). CI matrix: locked integration build, minimal-supported-versions,
  latest-compatible-versions, vendor feature combinations. **Start lightweight (claudex):** ownership,
  version policy, lockfile, patch meta-workspace, CI matrix are the hard prerequisites; elaborate
  coordinated-release ceremony + changelog machinery can mature *after* the first two releases.
- **R1. `ms-chem` — parity project, THEN extract (NOT mechanical — claudex).** Three impls
  (`mscore/chemistry`, `rustms/chemistry`, `timsim-chem`) may differ *silently* (mono vs average
  mass, isotope/element tables + rounding, residue/terminal conventions, UniMod snapshot + id
  mapping, formula-parser acceptance, mod ordering/canonicalization). A unified crate can change
  sim output while all unit tests pass. **Gate parity first:** inventory public APIs + embedded
  tables; differential-test all three against a corpus (formulas, modified peptides, isotope
  envelopes, known spectra); preserve golden outputs + tolerances; choose + *document* canonical
  semantics; add connector/Python regression fixtures; **plus a deterministic property-based
  differential test** (generate formulas + modified peptides, run all three legacy impls + `ms-chem`,
  assert identical canonical results OR an explicitly-approved semantic-difference record — this
  catches parser/ordering/terminal-mod/isotope/rounding edges the corpus won't). **Then publish a
  versioned `ms-chem` release** — that release point is what R2 pins.
- **R2. Migrate `mscore` onto versioned `ms-chem`, then publish it early as the frozen anchor
  (REVERSED from draft — claudex).** Cut a deliberately boring baseline `mscore` release. Its
  stability (17% co-change, 0 of last 150 commits) makes it the *compatibility anchor* everything
  downstream pins — publishing it *first* avoids the "everyone re-pins together at the end"
  intermediate state. Keep temporary re-export shims if the chem move changed public paths.
- **R3. Vendor-io + `rustdf` onto versioned `mscore`.** Publish `sciex-io` + `thermo-io`, drop the
  local-path/pinned-rev deps (`thermorawfile` = a *local patched dir*, not even git; `sciexwiff`
  pinned to a rev predating this session's writer push — live checkout-drift). Split `rustdf`/`ms-io`
  to pin versioned `mscore`. Zero binding risk (vendor crates carry no PyO3; glue stays in the
  connector).
- **Parallel exception (not an R3 action):** a vendor crate whose *released manifest has no `mscore`
  dependency* may publish immediately after R0 (before R2) to kill the local-path fragility now.
  Any vendor crate that *does* depend on `mscore` waits for R2.
- **R3.5. Version + fixture `timsim-schema` (R7) — prerequisite to R4.** The Parquet contract becomes
  a cross-repo boundary the moment producer and consumer split; version it and add producer/consumer
  conformance fixtures *before* extracting `timsim-core`.
- **R4. Extract `timsim-core`** out of `rustdf/sim` + `timsim-render*`; enforce Python-free; pins the
  versioned `timsim-schema` (R3.5) + `mscore` (R2).

Each R step's gate: crate builds standalone + `cargo test`; `imspy_connector` rebuilds to one wheel
(with a **committed lockfile**); the render/DiaNN parity check (SCIEX 449 PG, Thermo, timsTOF) stays
green.

## Risks & open questions

- **R1 — pyclass ownership.** Enforced by principle #2; already true. Verify no split crate ever
  re-exports a pyclass (grep for `pyo3` in every split crate's manifest as a CI guard).
- **R2 — version coordination centralizes in `imspy_connector`'s `Cargo.toml`, but it's one *edit
  location*, not one *compatibility problem* (claudex).** Because the connector compiles Rust types
  that cross crate boundaries, semver-compatible ranges can still resolve to an *incompatible type
  graph* or changed behavior. Mitigation: version deps for published releases + a **committed
  lockfile** for the wheel; the CI matrix (locked / min-versions / latest-compatible / vendor
  feature combos) from R0 is what actually catches skew.
- **R3 — crates.io vs private git.** The vendor-io crates are license-sensitive; they may want a
  private/tagged-git home rather than crates.io. Decide per crate.
- **R4 — `mscore` API stability. RESOLVED (2026-07): safe to split.** Co-change with `rustdf` is
  56/321 = **17%** all-history, and **0 of the last ~150 commits** touch mscore (last change was
  June 25, before the whole SCIEX/thermo/flow burst). mscore is loosely coupled *and* currently
  frozen → versionable independently today. Split it *last* only because it's the highest-fan-in
  crate (most consumers to re-pin), not for churn risk.
- **R5 — `timsim-core` repo home.** Own repo (max independence, its own release) vs ecosystem repo
  (simpler). Decide with the property-provider choice.
- **R6 — dev ergonomics.** Across repos: a dev `[patch]`/path-dep meta-workspace for local
  cross-crate work, vs strict version deps. Pick a convention so day-to-day hacking stays fast.
- **R7 — `timsim-schema` becomes a cross-repo *data* contract (claudex).** Once producers (predictors,
  Rust-analytical) and consumers (`timsim-core` render) live in different repos, the Parquet
  feature-space is an unversioned coupling waiting to break. It needs: a schema **version**,
  required/optional columns, units, nullability, metadata/provenance, a backward-read policy, and
  **producer/consumer conformance fixtures** owned in the schema crate. Version it *before* R4
  extracts `timsim-core`.
- **R8 — vendor distribution + legal (claudex).** `sciex-io`/`thermo-io` need a distribution decision
  beyond source licensing: their native SDK/runtime availability, wheel CI per platform, and whether
  they publish to crates.io or a private registry. Decide per crate (see R3).
- **Open Q — vendor-io Python surface. RESOLVED (2026-07): zero binding risk, but not render-only.**
  The vendor crates carry **no PyO3** (pure Rust) — but they *are* reachable from Python via
  `imspy_connector/src/py_acquisition.rs` (`from_sciex_wiff`/`from_thermo_raw`, feature-gated pyclass
  methods), chain `connector[sciex] → rustdf/sciex → dep:sciexwiff`. The glue lives in the connector,
  not the vendor crate, so publishing + pinning them changes nothing about the Python surface. Step 2
  is safe.

## Non-goals

- **Not** splitting the compiled binding (`imspy_connector`) into per-crate wheels — mix-and-match
  Rust-type flow across pyclasses kills that. *(The pure-Python feature packages `imspy-*` stay
  separate installables with their own extras — that's exactly how the torch decoupling works; only
  the one compiled connector wheel must stay unified.)*
- **Note:** "leave `timsim-schema` alone" is explicitly **rejected** — it must be versioned and
  validated (see R7). Leaving it implicit trades source coupling for unversioned *data* coupling.
- **Not** a big-bang cutover — the order above is incremental; each step ships and stays green.
