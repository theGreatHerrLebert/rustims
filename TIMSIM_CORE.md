# TODO — R4: extract `timsim-core` (the v2 sim engine)  ⟵ resume marker

**Status when written (2026-07):** R3 done. Four foundation crates published — `ms-chem 0.1.0`,
`mscore 0.5.0`, `ms-io 0.1.0`, `thermorawfile 0.1.0`. `rustdf`/`rustms` retired. SCIEX parked
(`sciex-io`, `publish = false`, private `sciexwiff`). rustims consumes the foundation via
`[patch.crates-io]`. **This doc scopes R4 — the last major Rust federation step.**

Related: `ECOSYSTEM_SPLIT.md` (federation plan), `MS_IO_CUTOVER.md` (R3b — the exact template for the
rewire mechanics), `CHEM_PARITY.md` (R1). The `rustdf → ms-io` cutover (commits `62b08fc7` + `371d14ac`,
codex-reviewed clean) is the mechanical template for stage 3 here.

## Goal
`ms-io` today is **three** modules: `data/` (7.0k lines, pure Bruker TDF I/O), `sim/` (11.4k lines, the
v2 sim engine), `cluster/` (DIA feature extraction). R3 folded the whole of old `rustdf` — engine
included — into `ms-io`, so `ms-io` is currently an I/O crate carrying an entire simulator. **R4 lifts
`sim/` out into a published `timsim-core` crate**, leaving `ms-io` as pure I/O. That makes the engine a
first-class, `cargo add`-able dependency (the "compose freely, don't ship everything" north star) and
stops the I/O crate from dragging the engine + `mzdata`/`thermorawfile` into every I/O consumer.

## The verified layering (why this is safe)
Dependency directions, all checked against the real tree:
- `data/` → `sim/`: **NONE.** `data/` is the lower layer; it never reaches up. ✅ clean cut line.
- `sim/` → `data/`: only `crate::data::meta::{DiaMsMsWindow, DiaMsMisInfo, read_meta_data_sql,
  read_dia_ms_ms_windows, read_dia_ms_ms_info}` (7 refs). → **`timsim-core` depends on `ms-io`** for
  these. Expected: the engine reads a real `.d`'s metadata and writes frames via the TDF writer.
- `sim/` ↔ `cluster/`: **fully disjoint** (zero refs either way). `cluster/` stays in `ms-io`.
- `sim/scheme.rs` → `crate::data::meta::` at line 836 only (inside the *extraction-from-real-.d* fn).
  The `AcquisitionScheme` **type + `validate()` + `from_window_table`** are pure (serde only). ← the
  key seam, see D1.

Target DAG after R4 (D1 = new `timsim-types` serde-only leaf):
```
        mscore ── ms-chem                 timsim-types  (serde-only acquisition contracts)
          │                                 │    │    │
   ms-io (data/ + cluster/)   timsim-schema ┘    │    └─ sciex-io  (Arrow-free R9 seam)
          │            \        (Arrow/Parquet)  │
          └─────────────┴──── timsim-core ───────┘   (= today's ms-io/sim/; deps ms-io + types [+schema?])
                                       │
                          timsim-cli · imspy_connector · tims-viewer
```

## Stage 0 — RESULTS (discovery complete, 2026-07)
Compile-truth recon done. Two findings *simplify* the plan vs its assumptions:

**F1 — `timsim-types` needs ZERO external deps (not even serde).** `scheme.rs` has no `serde` derives, and
**nothing anywhere serializes** an `AcquisitionScheme` / `DiaWindow` / etc. So the leaf is pure `std`.
(The earlier "add serde, cheap" is unnecessary — don't add it unless a future consumer needs it.)

**F2 — `timsim-core` does NOT need `timsim-schema` (nor `timsim-chem`).** `sim/` has **zero**
`polars`/`parquet`/`arrow` references — feature-space Parquet is read by **`timsim-cli`** (its `lib.rs` +
render bins), which then calls the engine with in-memory data. The engine never touches the column schema.
Codex's "don't add speculatively" confirmed by compile.

**`timsim-core` dependency set** (exact, from per-module crate-usage counts):
```toml
[dependencies]
ms-io      = { … }                                  # data::meta (7), tdf_writer, handle, dataset
timsim-types = { … }                                # the scheme leaf
mscore     = "0.5.0"                                 # 49 refs
rusqlite   = { version = "0.32", features=["bundled"] }  # 77 refs — synthetics DB + analysis.tdf
rayon      = "1.10"                                  # 21
serde      = { version = "1", features=["derive"] }  # 1  (a container derive; keep)
serde_json = "1"                                     # 9
rand       = "0.8"                                   # 8
regex      = "1.11"                                  # 1
[features]
thermo = ["dep:thermorawfile"]   # 20 refs — VERIFY in stage 2 whether it must also enable "ms-io/thermo"
mzml   = ["dep:mzdata"]          # 10 refs
```
NOT needed by core: `polars`, `parquet`, `arrow`, `timsim-schema`, `timsim-chem`, `zstd`, `byteorder`,
`lzf`, `libloading`, `bincode`, `clap`, `rustc_hash` (all belong to `data/`/`cluster/`, which stay in ms-io).

**The `scheme.rs` split (exact) — and the one non-mechanical wrinkle: the ORPHAN RULE.**
The leaf owns every *type* + every *std-only inherent method*. But 6 methods are coupled to `ms-io`/
`thermorawfile` and **cannot stay inherent methods on `AcquisitionScheme`** once the type lives in the leaf
(you can't `impl` a foreign type's inherent methods from another crate). They move to `timsim-core` as
**free functions** (exactly the pattern `sciex-io::from_sciex_wiff` already uses) or a `BrukerSchemeExt`
extension trait:

| goes to **`timsim-types`** (pure std) | moves to **`timsim-core`** (free fn / ext-trait) |
|---|---|
| `SCHEME_VERSION`; enums `InstrumentKind, Analyzer, DataMode, ActivationMethod, EnergyUnit, CollisionEnergyModel, DiaGeometry, AcquisitionEvent, RepeatPolicy, SchemeSource, CollisionEnergyPolicy`; structs `IsolationWindow, ActivationCondition, InstrumentCapabilities, ActivationPolicy, DiaWindow, Ms1Event, DiaMs2Frame, Provenance, AcquisitionScheme` | `to_bruker_windows`, `to_bruker_info`, `to_bruker_tables` (→ `data::meta` types) |
| inherent methods: `IsolationWindow::{lower,upper}`, `CollisionEnergyPolicy::at`, `ActivationCondition::{collisional_ev,legacy_bruker}`, `InstrumentCapabilities::{bruker_timstof,astral}`, `ActivationPolicy::{bruker_pasef,thermo_nce,collision_energy_for_scan,condition_for_scan,condition_for_window}`, `AcquisitionScheme::{windows,ms1_count,num_cycles,dia_frame_schedule,from_window_table,validate}` | `from_bruker_d` (reads a real `.d` via `data::meta`); `from_thermo_raw`, `thermo_frame_schedule` + the `TemplateScan` struct (thermo feature, `thermorawfile`) |

**Split refinements (codex-verified against the source):**
- **`bruker_group_layout`** (private inherent helper, std-only) also moves to **core** — the Bruker
  adapter free fns can't call a private inherent method on the leaf's type. Make it a private core fn.
- **`TemplateScan` → leaf** (pure: primitives + `Option<IsolationWindow>`), but **drop its
  `#[cfg(feature="thermo")]` gate** there (leaf has no thermo feature); core's `thermo_frame_schedule`
  returns `Vec<timsim_types::TemplateScan>`.
- **Tests split**: pure unit tests → leaf; the Bruker/thermo round-trip tests (use `crate::data::meta` /
  `thermorawfile`) stay in **core**. Fix `crate::sim`/`crate::data` intra-doc links when moving.
- The whole `#[cfg(feature="thermo")] impl AcquisitionScheme` block moves to core with its imports.

**Consumer call-site changes this forces** (stage 4): connector's `AcquisitionScheme::from_thermo_raw(…)`
→ `timsim_core::from_thermo_raw(…)`; `sciex-io` imports its 15 scheme types from `timsim_types::` (all in
the leaf column above — clean). Connector's other sim surface (`projector` ×4, `utility` ×3, `precursor`,
`mzml`, `library`, `handle`, `dia`, `dda`, `acquisition` writers) → `timsim_core::…`.

## Decisions to make (with recommendations) — David decides D1, D2

### D1 — where does `AcquisitionScheme` (the R9 SCIEX seam) live? ✅ **DECIDED: (d) new `timsim-types` leaf**
`sciex-io` builds an `AcquisitionScheme` through only its public API (the R9 "explicit API gate"), so
whatever crate owns the type is what `sciex-io` depends on. **Split by SEMANTICS, not a hand-picked list:**
the *immutable, serializable acquisition description + `validate()`* is the leaf; the *behavior*
(`.d`/Thermo extraction, conversion to Bruker metadata rows, template rescan, rendering
policy/capability logic) is core. The full contract surface is larger than first listed — `sciex-io`
alone already uses `AcquisitionEvent, Analyzer, DataMode, DiaGeometry, DiaMs2Frame, Ms1Event,
IsolationWindow, DiaWindow, Provenance, RepeatPolicy, SchemeSource, InstrumentKind,
CollisionEnergyPolicy, SCHEME_VERSION, AcquisitionScheme`, and `scheme.rs` also holds `Activation*`,
`EnergyUnit`, `InstrumentCapabilities`, `CollisionEnergyModel`, `ActivationPolicy`. **Stage 0 must
enumerate the exact type set by compiling, not by eyeballing.** Options for the leaf's home:
- **(a) into `timsim-core`** with the rest of `sim/`. → `sciex-io` deps `timsim-core`, i.e. the whole
  engine + `ms-io` + `mzdata`, just to name a scheme struct. Heavy; loses the tiny-leaf seam.
- **(b) into `timsim-schema`.** It *is* "the schema contract for timsim v2," so an acquisition contract
  fits semantically — but note it is **not lightweight**: it already pulls `arrow` + `parquet`. So
  `sciex-io` (and anyone naming a scheme) drags Arrow/Parquet in. Cost: add `serde` to it (cheap).
  Fewest crates; heaviest seam.
- **(d) NEW serde-only leaf `timsim-types` — RECOMMENDED.** A tiny `serde`-only crate holding the
  acquisition contract types + `validate()`, sitting *below* both `timsim-schema` and `timsim-core`:
  ```
  timsim-types (serde-only acquisition contracts)
    ├─ timsim-schema (Arrow/Parquet column contracts)
    ├─ timsim-core   (extraction + engine)
    └─ sciex-io      (Arrow-free — the smallest possible R9 seam)
  ```
  `sciex-io` stays Arrow-free; the extraction fns live in `timsim-core`. Cost: +1 crate. This is the
  genuine trade — **crate count (b) vs dependency weight (d)** — and it's David's call.
- **(c) keep in `ms-io`.** Leaves a sim-flavored module in the "pure I/O" crate — undoes the point.
- No cycle risk in any option: the leaf never imports core; `timsim-core → leaf` is one-way. Enforce
  with a standalone leaf build/test.
- `from_thermo_raw` / `rewindow_thermo_template` (`sim/acquisition.rs`, thermo-gated) are the Thermo
  analogue: their scheme *output type* is the leaf; the extraction *body* is `timsim-core` (thermo
  feature — which may in turn have to enable `ms-io/thermo`; see D4).

### D2 — where does `timsim-core` live? ⟵ **secondary decision, DEFERRABLE**
**Not load-bearing for R4 correctness** (codex): Cargo doesn't care whether the crates share a repo, and
the whole extraction develops against local `path` deps regardless. We can bootstrap + validate first and
decide the repo home only at publish time (stage 3). Choose the family repo for *product boundaries*, not
to fix any Cargo/staging issue. Options:
- **(a) own "timsim" family repo — RECOMMENDED.** A repo holding the three `timsim-*` crates
  (`timsim-schema`, `timsim-chem`, `timsim-core`), mirroring how the `mscore` repo holds
  `ms-chem`/`mscore`/`ms-io`. Clean split: **foundation** (low-level MS) vs **timsim** (the simulator).
  Implies R4 also *publishes* `timsim-schema` + `timsim-chem` (currently rustims-local, unpublished).
- **(b) into the `mscore` foundation repo.** Fewer repos, but conflates the low-level foundation with an
  application engine — the engine is not "foundation."
- **(c) stays in rustims, unpublished.** Defers the decision but blocks external composition (others
  can't `cargo add timsim-core`) — contradicts the north star. Acceptable only as an interim.

### D3 — `ms-io 0.2.0` (breaking: `sim/` removed)
`ms-io 0.1.0` shipped *with* `sim/`. Removing it is a breaking change → **`ms-io 0.2.0`** (`^0.1.0`
requirements will *not* silently pick it up — correct). Blast radius is tiny in practice: `ms-io 0.1.0`
is days old with no known external consumers; the only in-graph users are rustims (patched) + `sciex-io`.
Recommend the clean removal, **not** a deprecation shim (a shim keeps `sim` in the "pure I/O" crate and
defeats the point). Ship a one-line `ms_io::sim::* → timsim_core::*` migration note in the 0.2.0 release
text; skip the `0.1.1`-announcement ceremony (no downstream to announce to).

### D4 — render output writers (`mzml.rs`, `acquisition.rs`/`ThermoRawWriter`, `astral_dispatch.rs`)
These are render *outputs*, part of the engine → move to `timsim-core` under its own `mzml` / `thermo`
features (pulling `mzdata` / `thermorawfile`). `ms-io` keeps its *reader* thermo/mzml features. Recommend
as stated; no decision needed unless we want writers to stay I/O-side.

### D5 — the `ms-io` binary (`main.rs`, uses `sim::dia`)
It's a render tool → move to `timsim-core` as a bin, **or** drop it (timsim-cli already has richer render
bins: `render.rs`, `render_bench.rs`, `render_thermo.rs`). Recommend **drop** unless it has unique use.

## Staged execution plan

> **Bootstrap reality (codex catch — the buildability gap).** `ms-io` lives in the *separate* `mscore`
> repo. The moment `ms-io/sim` (or its successor) references a `timsim-*` type, that repo's `ms-io` needs
> a dependency on an **unpublished** crate — so the split is NOT a rustims-local move. Everything below
> develops against a **coordinated local topology first** (sibling `path` deps across the `mscore`
> checkout and the timsim crates), compiles green end-to-end, and only *then* publishes in strict
> dependency order. Direct `path` deps — not `= "x.y"` + `[patch]` — are the safe cross-repo bootstrap
> for a crate that has never been published (see corrected gotcha). Switch to versioned deps + patch
> only after each crate is on crates.io.

**Stage 0 — discovery (compile-time truth, not greps).** Pin `timsim-core`'s real dep set by making it
compile, not by eyeballing. Known couplings to confirm: `mscore`, `serde`/`serde_json`, `rayon`, `rand`,
`polars`, SQLite-facing `data::meta`, feature-gated `thermorawfile`/`mzdata`. **Does NOT appear to touch
`timsim-chem` — do not add it speculatively; only add if the compiler demands it.** Also enumerate the
exact D1 leaf type-set (the full acquisition-contract surface). Output: exact `[dependencies]` for
`timsim-core` and the exact type list for the leaf.

**Stage 1 — establish local topology + D1 split.** Wire sibling `path` deps: the timsim leaf (per D1) +
`timsim-core` as local paths, and add the leaf as a `path` dep of the `mscore`-checkout `ms-io`. Move the
acquisition-contract *types* (+ `validate()`) into the leaf (add `serde`); keep extraction fns behind in
`sim/`. Verify the whole coordinated graph builds locally with `sciex-io`/connector pointing at the leaf
for the type. This isolates the trickiest ref change while everything is still path-linked.

**Stage 2 — create `timsim-core`.** Move `ms-io/src/sim/*` → `timsim-core/src/` (minus leaf types).
Rewrite `crate::data::` → `ms_io::data::`, `crate::sim::` → `crate::`. Features `thermo`, `mzml` (note:
`timsim-core/thermo` may have to `= ["ms-io/thermo"]` if its Thermo extraction calls ms-io's reader —
document the coupling). Compiles standalone against local-path `ms-io` + leaf.

**Stage 3 — cut `ms-io 0.2.0` + publish in dependency order.** Remove `pub mod sim;`; delete `sim/`;
drop `main.rs` (D5) + sim-side example entries; bump `ms-io` → `0.2.0`; it builds as pure I/O. Then
publish **in order** (each is David's explicit, irreversible authorization): (1) leaf `timsim-types`/
schema, (2) `timsim-chem` *only if* stage 0 proved a dep, (3) `timsim-core`, (4) `ms-io 0.2.0`. Order
matters — a crate can't publish before its deps are on crates.io.

**Stage 4 — rewire rustims + flip paths to versions** (mechanically = R3b stage 3). `ms_io::sim::` →
`timsim_core::` across the connector (~20 paths), `timsim-cli`, `tims-viewer`. Consumers now dep **both**
`ms-io` (data/cluster) **and** `timsim-core` — make that an intentional contract; forward *writer*
features to `timsim-core`, keep *reader* features on `ms-io`. `sciex-io` → the leaf for the scheme type.
Flip the now-published crates from `path` → `= "x.y"` and add them to `[patch.crates-io]`. Verify:
`cargo build --workspace`; **the feature MATRIX, not just all-on** (`--no-default-features`, each feature
alone, then `thermo,sciex,mzml` together — unification hides accidental deps); **`maturin build -m
imspy_connector/Cargo.toml`** (load-bearing — Python survives); parity gate `cargo test -p timsim-chem`;
and a final `cargo build` with patches removed before any release tag.

## Gotchas (carried from R3b — all still apply)
- **NEVER `git add -A` in a foundation/family repo** — it once swept 1.1 GB of `target/*.rlib` into a
  commit. Add specific paths; check `git status` for `target`.
- **Cross-repo bootstrap of an UNPUBLISHED crate → use a direct `path` dep, not `= "x.y"` + `[patch]`.**
  (The blanket "patch only works for published crates" is too strong — Cargo *can* patch an unpublished
  version in some setups — but a direct `path` dep is the always-works, unambiguous route for a crate
  that has never hit crates.io, and it bit us in R3b when we tried otherwise. Switch each dep to
  `= "x.y"` + `[patch]` only *after* that crate is published.) Path levels: from `rustims/Cargo.toml` a
  sibling repo is `../../<repo>/<crate>`; from a nested crate (`rustims/sciex-io/`) it's
  `../../../<repo>/<crate>`.
- **`-p X --features` fails for out-of-workspace crates** ("cannot specify features for packages outside
  of workspace") — build the connector from its own directory when forwarding ms-io/timsim-core features.
- **`/scratch` runs near-full** — `cargo clean` frees regenerable `target/`; **never** delete data
  (`SUBMISSION/*`, benchmarks, `*.d` renders).
- `sciexwiff` stays **legal-held + private**; `sciex-io`/`publish = false` keeps it off crates.io. R4
  only *narrows* sciex-io's dep (sim→schema); it does not publish it.

## Definition of done
`sim/` gone from `ms-io`; `ms-io 0.2.0` is pure I/O (`data/` + `cluster/`); `timsim-core` published and
depended on by rustims; `sciex-io` deps only `timsim-schema` for the scheme type; connector wheel builds;
parity gate green. Then the Rust federation is complete — foundation (`ms-chem`/`mscore`/`ms-io`) +
timsim family (`timsim-schema`/`timsim-chem`/`timsim-core`) + rustims-local `sciex-io`.
