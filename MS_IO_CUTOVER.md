# TODO — ms-io cutover (R3b stages 3 & 4)  ⟵ resume marker

**Status when written:** R3b stages 1–2 DONE. `ms-io` lives in the foundation repo
(`github.com/theGreatHerrLebert/mscore`, at `/scratch/timsim-demo/mscore/ms-io`), builds SCIEX-free
(Bruker + optional `thermo`/`mzml`), packages clean, **not yet published**. `sciex-io` (the SCIEX
`.wiff` satellite) exists at `rustims/sciex-io/` (`publish = false`, deps ms-io + private `sciexwiff`
git). **rustims still uses the old `rustdf`** — this doc is the remaining cutover.

Related: `ECOSYSTEM_SPLIT.md` (federation plan), `CHEM_PARITY.md` (R1), `RELEASING.md` (R0 infra).
The mscore/ms-chem cutover (already done, commit `d4824a02`) is the exact template for this.

## Goal
Cut rustims off `rustdf` and onto **`ms-io` (Bruker/thermo/mzml) + `sciex-io` (sciex)**; delete
`rustdf` from the monorepo; publish `ms-io 0.1.0`. Verify the connector wheel + parity gate stay green.

## Stage 3 — rewire rustims (rename `rustdf` → `ms_io`, sciex → sciex-io)

**3a. Consumer manifests** — switch the `rustdf` dep to `ms-io` (+ `sciex-io` for the sciex feature):
- `imspy_connector/Cargo.toml:15` `rustdf = { path="../rustdf", version="0.4.1" }` → `ms-io = { path = "../../mscore/ms-io" }` (path until ms-io is published; then `= "0.1.0"` + `[patch]`).
- `imsjl_connector/Cargo.toml:9`, `tims-viewer/Cargo.toml:40`, `timsim-cli/Cargo.toml:44` — same (keep `optional`/`version` shape).
- **Feature forwarding** (`imspy_connector/Cargo.toml:23-28`): `thermo = ["rustdf/thermo"]` → `["ms-io/thermo"]`; `mzml = ["rustdf/mzml"]` → `["ms-io/mzml"]`; **`sciex = ["rustdf/sciex"]` → `["dep:sciex-io"]`** (ms-io has NO sciex feature — sciex comes from the sciex-io crate). Add `sciex-io = { path = "../sciex-io", optional = true }`.

**3b. Rust code rename** — `rustdf::` → `ms_io::` (85 refs / 17 files):
`imspy_connector` (62), `timsim-cli` (15), `tims-viewer` (6), `imsjl_connector` (2). Mechanical:
`grep -rl 'rustdf::' <crate>/src | xargs sed -i 's/rustdf::/ms_io::/g'`.

**3c. The SCIEX glue** (`imspy_connector/src/py_acquisition.rs`): `from_thermo_raw` (line ~44) keeps
calling `AcquisitionScheme::from_thermo_raw` (ms-io has it under `thermo`). BUT `from_sciex_wiff`
(line ~88, `#[cfg(feature="sciex")]`) currently calls `AcquisitionScheme::from_sciex_wiff(...)` —
that method **moved to sciex-io**. Change it to `sciex_io::from_sciex_wiff(path, cycle_time_s,
gradient_length_s, ce)` (a free fn returning `io::Result<AcquisitionScheme>`). Same signature/args.

**3d. Remove rustdf from the monorepo**: drop `"rustdf"` from `Cargo.toml` workspace members;
`git rm -r rustdf`. (rustdf 0.4.1 stays on crates.io as legacy; ms-io is its successor.)

**3e. Verify** (all must pass):
- `cargo build --workspace` (ms-io resolves from foundation via path/patch).
- `maturin build -i python3.11 -m imspy_connector/Cargo.toml` — **the load-bearing check** (Python-facing build survives; build with `--features` matching current: thermo + sciex + mzml).
- The relocated parity gate: `cargo test -p timsim-chem --test chem_parity --test ms_chem_parity`.

## Stage 4 — publish `ms-io 0.1.0` (needs David's authorization)
From `/scratch/timsim-demo/mscore`: `cargo publish -p ms-io` (permanent, public). Then in rustims
switch consumer `ms-io` path deps → `= "0.1.0"` + add `ms-io = { path = "../../mscore/ms-io" }` to the
existing `[patch.crates-io]` block (alongside ms-chem/mscore) for local dev.

## Gotchas (learned this session)
- **NEVER `git add -A` in the foundation repo** — it swept `target/` (1.1 GB of `.rlib`s) into commits
  once; `.gitignore` now has `target/`, but add specific paths and check `git status` for `target`.
- **Path levels:** from `rustims/Cargo.toml` the foundation is `../../mscore`; from a nested crate
  (`rustims/sciex-io/`) it's `../../../mscore`. sciex-io already uses `../../../mscore/ms-io`.
- **`[patch.crates-io]` only works for PUBLISHED crates** — ms-io isn't published until stage 4, so
  stage 3 must use a **path dep** to the foundation checkout, not `= "0.1.0"` + patch. Switch after 4.
- **thermo examples** need `required-features = ["thermo"]` (already set in foundation ms-io).
- `/scratch` runs near-full — `cargo clean` frees ~50–80 GB of regenerable target if a build hits
  `No space left on device`. Never delete data (SUBMISSION/*, benchmarks, *.d renders).
- sciexwiff is **legal-held + private**; sciex-io/`publish=false` keeps the private dep off crates.io.

## Definition of done
rustims builds + connector wheel builds against foundation ms-io + sciex-io; rustdf gone from the
monorepo; ms-io 0.1.0 on crates.io; parity gate green. Then R3c (connector adapters polish) / R4
(timsim-core).
