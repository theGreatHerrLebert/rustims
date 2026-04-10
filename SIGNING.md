# SIGNING — moved

The cryptographic provenance work that used to live in this file has
been factored out into a standalone, multi-implementation specification
repository named **mzprov**.

The factoring-out was triggered by the appearance of a second
implementation (a C# port being written by a collaborator). With more
than one implementation in flight, the right home for the design
documents and the normative spec is a project-neutral repository, not
a subdirectory of TimSim.

## Where the work lives now

The mzprov repository contains:

| Directory | Contents |
|---|---|
| `README.md` | Elevator pitch, status, quick-start CLI examples, repo layout, license summary |
| `docs/` | Non-normative design documentation: problem statement, threat model, architecture, canonicalization design, roadmap, related work, FAQ |
| `spec/` | Normative specification (RFC 2119): sidecar format, `.d` and mzML canonicalization, signature scheme, key id derivation, trust model, security considerations |
| `test-vectors/` | Cross-implementation test vectors (CC0): valid sidecars, invalid sidecars covering each rejection path, canonicalization fixtures with expected hashes, test-only signing key |
| `implementations/python/` | Python reference implementation, lifted from `imspy_simulation.provenance`. Imports as `mzprov`. Apache-2.0. |
| `implementations/csharp/` | Independent C# implementation (placeholder; in development) |
| `CONTRIBUTING.md` | Governance, the `v1-draft/` convention for proposing new features, what is frozen at v0 |

For the rationale and the staged migration, see [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md)
in this branch.

## TimSim integration after the lift

The Python implementation under `mzprov/implementations/python/` is the
canonical home for the signing and verification code. TimSim's
`imspy_simulation.provenance` module is now a **thin re-export shim**
that re-exports the public API from `mzprov`. Existing TimSim users
see no change:

- `from imspy_simulation.provenance import sign_simulation_output, verify_sidecar`
  continues to work and resolves to the symbols in `mzprov`.
- The `timsim-verify` and `timsim-keys` console scripts continue to
  exist and dispatch to `mzprov.cli:main` and `mzprov.keys_cli:main`.
- Existing sidecars produced by TimSim before the lift verify
  identically after the lift. The on-disk format did not change.
- Exit codes `0`–`7` are preserved exactly.
- The signing key location (`~/.config/timsim/keys/signing_key.pem`)
  and the trusted-keys registry location
  (`~/.config/timsim/trusted_keys.json`) are preserved. A v1 rename to
  `mzprov`-prefixed paths is candidate work, with a one-time migration
  helper, and is recorded in the mzprov FAQ.

The cutover deleted twelve `.py` files from
`packages/imspy-simulation/src/imspy_simulation/provenance/` (every
file except `__init__.py`, which became the shim) and removed the
`tests/test_provenance/` directory entirely (the test suite now lives
in `mzprov/implementations/python/tests/`). 154 tests pass against
the lifted implementation. The cutover commit history is in this
branch.

## What this rustims branch still owns

After the lift, this branch only owns:

- The shim at `packages/imspy-simulation/src/imspy_simulation/provenance/__init__.py`
- The `mzprov>=0.1` dependency declaration in
  `packages/imspy-simulation/pyproject.toml`
- The `timsim-verify` / `timsim-keys` entry point declarations in the
  same `pyproject.toml`, now pointing at `mzprov.cli:main` /
  `mzprov.keys_cli:main`
- The `imspy_simulation.timsim.simulator` integration that imports
  from `imspy_simulation.provenance` (which goes through the shim)
- This `SIGNING.md` pointer file
- [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) documenting the rationale
  and the order of operations

Everything else lives in mzprov.

## Why this matters for reviewers

The factoring-out is what makes the cross-implementation contract
real. As long as the canonical home for the signing code was
`imspy_simulation.provenance`, any other implementation had to read
Python source to discover what was normative and what was incidental,
and the test suite was binding only on `imspy-simulation`. With the
spec and the test vectors in their own repository, R's C# code (and
any future third-language implementation) targets a written contract,
and the reference test vector run is the conformance check.

For the security considerations and the threat model in the format a
reviewer will recognize, see the spec's
[`security-considerations.md`](#) (in the mzprov repository) and the
non-normative companion in [`docs/03-threat-model.md`](#).
