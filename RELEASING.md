# RELEASING — R0 release/CI policy for the crate federation

**Status:** R0 kickoff (2026-07). This is the *lightweight start* the split plan calls for: the
release discipline that must exist **before** any crate is extracted or published. It is policy +
CI, not ceremony — coordinated-release machinery and changelog automation can mature after the first
two crates ship. See [`ECOSYSTEM_SPLIT.md`](./ECOSYSTEM_SPLIT.md) for the *why* and the target DAG.

R0 is a **prerequisite** for the Rust workstream (R1 `ms-chem` parity → R2 `mscore` anchor → R3
vendor-io/`ms-io` → R4 `timsim-core`). Nothing splits until the items below are green.

## 1. Ownership & the federation target

- Code-of-record for each primitive moves to its own repo per the ECOSYSTEM_SPLIT repo map
  (`mscore` foundation, `thermo-io`, `sciex-io`, `timsim`, `rustims` ecosystem). Until a crate is
  extracted, this monorepo remains its home.
- The single compiled binding (`imspy_connector`) stays in the ecosystem repo and is the **only**
  crate that may carry `pyo3`. This is enforced (see CI guard §7).

## 2. Versioning & semver

- Every published crate follows semver. Breaking API change → major (pre-1.0: minor) bump.
- **Internal deps carry both `path` and `version`**: `mscore = { path = "../mscore", version = "0.4.1" }`.
  Cargo uses `path` for local builds and `version` when published — so a manifest is publish-ready
  without edits. *(Already the pattern; the few path-only deps in `timsim-cli` are bins, not
  published — fix them if any becomes a library.)*
- The connector pins **exact** compatible versions of the crates it binds; a semver-compatible range
  can still resolve to an incompatible Rust type-graph across crate lines (see ECOSYSTEM_SPLIT R2).
  The committed lockfile (§5) is what actually guarantees the wheel's reproducibility.

## 3. MSRV

- **MSRV = Rust 1.84**, declared as `rust-version = "1.84"` in every publishable crate
  (mscore, rustdf, rustms, timsim-chem, timsim-schema, imspy_connector — now consistent).
- Raising the MSRV is a minor-version event and must be reflected in the CI matrix (§7).

## 4. Lockfile

- This repo **produces a wheel** (`imspy_connector`), so it is an application for lockfile purposes:
  **commit `Cargo.lock`**. *(Decision pending — see §8; currently gitignored.)* A committed lockfile
  is the reproducibility anchor for the wheel and the `locked` CI leg.

## 5. Dev ergonomics — the `[patch.crates-io]` meta-workspace

Once crates live in separate repos and depend on **published** versions, you can't use `path` deps in
released manifests. For local cross-crate hacking, use a top-level dev workspace that patches the
published crates back to local checkouts:

```toml
# dev-workspace/Cargo.toml (NOT published)
[patch.crates-io]
mscore      = { path = "../mscore/mscore" }
ms-chem     = { path = "../mscore/ms-chem" }
timsim-core = { path = "../timsim/timsim-core" }
```

`[patch]` only applies from the top-level invocation, so this is the sanctioned way to edit several
crates at once without version churn. Published manifests stay version-only.

## 6. Publishing flow (per crate)

1. Green CI matrix (§7) on the crate + a locked `imspy_connector` rebuild to one wheel.
2. Registry target decided **per crate** (crates.io vs private/tagged-git) — vendor readers may
   differ (ECOSYSTEM_SPLIT R3/R8; both are now known-permissive pure-Rust, so neither is *blocked*
   from crates.io — `sciexwiff` needs a license *added* first).
3. `thermorawfile`'s Apache **NOTICE** travels with the crate.

## 7. CI matrix (the hard prerequisite)

Extend `.github/workflows/rust.yml` from a single build into a matrix that catches version skew and
boundary violations *before* a split can hide them:

- **locked** — `cargo build --locked` (the committed lockfile builds).
- **min-versions** — `-Z minimal-versions` resolve + build (lower bounds are real).
- **latest-compatible** — fresh resolve, newest semver-compatible deps.
- **vendor feature combos** — `--no-default-features`, `--features tdf`, `thermo`, `sciex`, `mzml`,
  and `--all-features` (the lean default and each vendor path build).
- **MSRV leg** — build on Rust 1.84 (the declared floor).
- **pyclass guard** — CI fails if any crate other than `imspy_connector` gains a `pyo3` dependency
  (`grep` its manifests). Enforces principle #2 as the crates split.
- **semver-check** — `cargo-semver-checks` on published crates to catch accidental API breaks.

## 8. Decisions still needed (blockers to finishing R0)

- [ ] **Commit `Cargo.lock`?** Recommended yes (wheel-producing repo). Removes it from `.gitignore`.
- [ ] **Default registry** — crates.io vs a private registry, decided per crate (§6). Gates the
      publish workflow and credentials.
- [ ] **`cargo-semver-checks` + minimal-versions** — adopt in CI (adds ~1 job each).

## R0 status

- [x] MSRV consistent at 1.84 across publishable crates.
- [x] Internal deps carry `path` + `version` (publish-ready), verified.
- [x] This policy documented.
- [ ] Cargo.lock committed (pending §8 decision).
- [ ] CI matrix (§7) implemented.
- [ ] Registry + credentials decided.

Once the boxes above are checked, R1 (`ms-chem` parity project) may begin.
