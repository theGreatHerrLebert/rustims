# mzprov — Migration Plan

This document captures the decision, on 2026-04-10, to factor the Phase 0 provenance work out of `imspy_simulation` into a standalone, multi-implementation specification repository named **mzprov**. It is the bridge document that explains *why* the work is moving and *how* the move will be staged.

It lives in the rustims branch (not in the new repo) so that anyone reading the SIGNING.md history can see the migration was deliberate and follow it forward.

---

## 1. Why factor this out

The Phase 0 provenance work has reached the point where its growth is no longer a feature of TimSim. Specifically:

- **A second implementation is on the way.** A collaborator (R) is writing a C# implementation. Without a written specification, the two implementations will reconcile differences in prose forever. With a written specification *and* test vectors, the reconciliation is mechanical: a conforming implementation accepts every valid vector and rejects every invalid one.
- **The spec is the contract, not the code.** As long as the canonical home is `imspy_simulation.provenance`, any other implementation has to read Python source to discover what is normative and what is incidental. That is the wrong default.
- **Positioning matters.** §17 of SIGNING.md argues that simulator authors are well placed to ship the first reference implementation. That argument is much stronger when the artifact is a portable protocol with two language implementations than when it is a TimSim subdirectory.
- **SIGNING.md has outgrown its file.** It currently mixes problem statement, threat model, security claims, design rationale, and implementation status in 889 lines. Splitting them surfaces which sentences are MUST/SHOULD and which are commentary.

The cost of the move is bounded — most of it is rearranging text we have already written and lifting Python code that already has a clean module boundary. The benefit is that every downstream conversation (with R, with PSI, with vendors, with reviewers) starts one notch closer to "this belongs in the standards conversation."

---

## 2. Decisions made on 2026-04-10

| Decision | Choice | Rationale |
|---|---|---|
| **Name** | `mzprov` | Slots into the HUPO-PSI **mz\*** family (mzML, mzTab, mzIdentML, mzQuantML), signaling the long-term intent to be a community format. Pronounceable, short, available on PyPI, crates.io, and as a GitHub user/org. |
| **Fallback name** | `msprov` | Held in reserve. Also clean across PyPI, crates.io, GitHub. Used only if the PSI-family alignment of `mzprov` turns out to be politically problematic. |
| **Home (development)** | `/scratch/timsim-demo/mzprov/` | Sibling of `rustims`, own git repo from day one, decoupled from the SUBMISSION paper artifact. |
| **Home (publication)** | TBD GitHub home | Deferred decision. Either a new GitHub user/org named `mzprov` or under the existing user namespace. To be resolved before the first push. |
| **License — `docs/` and `spec/`** | CC-BY-4.0 | Specifications must be quotable in papers and other specifications without legal friction. |
| **License — `implementations/python/`** | Apache-2.0 | The patent grant matters for security-adjacent code. Gives downstream commercial implementers more comfort than MIT. |
| **License — `test-vectors/`** | CC0 | Test fixtures should have zero attribution burden so any implementation in any language can ship them as part of its test suite. |
| **Governance** | Solo merge to `main`. New feature proposals land as drafts under `spec/v1-draft/` with a proposal author and discussion notes, and only graduate into the normative spec when there is an implementation, test vectors, and approval. | Appropriate for a project that is currently two people and one design. The `v1-draft/` convention gives R (and any future contributor) a real path to propose features without bypassing review. |
| **Refactor strategy** | **Lift-and-redirect.** `mzprov` becomes the canonical home for the Python code. `imspy_simulation.provenance` becomes a thin wrapper that re-exports from `mzprov`. | TimSim's existing `timsim-verify` and `timsim-keys` CLIs keep working unchanged. The test suite stays in one place (under `mzprov`). No "which copy is canonical" drift problem. |
| **v0 scope** | **Frozen** at exactly what is shipping today: `.d` canonicalization v0, mzML canonicalization v0, the sidecar envelope, Ed25519, the trust model, exit codes 0–7. | Anything new — sampling extensions, run-level mzML metadata, vendor RAW canonicalization — lands in `spec/v1-draft/` and waits for test vectors and implementer review. |

---

## 3. Target repository layout

```
mzprov/
├── README.md                         # elevator pitch + worked example + navigation
├── CHANGELOG.md                      # version history (v0 in development)
├── CONTRIBUTING.md                   # short, points to governance
├── LICENSE                           # explains the three-license split
├── LICENSE-APACHE                    # full Apache 2.0 text
├── LICENSE-CC-BY                     # full CC-BY-4.0 text
├── LICENSE-CC0                       # full CC0 text
├── .gitignore
│
├── docs/                             # NON-NORMATIVE. Problem, design, rationale.
│   ├── 01-problem.md                 # SIGNING.md §1–§6 condensed
│   ├── 02-architecture.md            # SIGNING.md §7–§8
│   ├── 03-threat-model.md            # SIGNING.md §3, §5 unified
│   ├── 04-canonicalization.md        # SIGNING.md §9 + sampling discussion
│   ├── 05-roadmap.md                 # SIGNING.md §15
│   ├── 06-related-work.md            # SIGNING.md §10
│   └── faq.md                        # mzML SHA-1 misread, "why not blockchain", etc.
│
├── spec/                             # NORMATIVE. RFC 2119 keywords. Implementations conform to this.
│   ├── README.md                     # how to read the spec
│   ├── sidecar-format.md             # JSON schema, field semantics
│   ├── canonicalization-d-v0.md      # SQLite content canonical form
│   ├── canonicalization-mzml-v0.md   # mzML spectrum content canonical form
│   ├── signature-scheme.md           # Ed25519 + algorithm-agility envelope
│   ├── key-id-derivation.md          # blake2b-80 + base32-lowercase
│   ├── trust-model.md                # verifier behavior, exit codes
│   ├── security-considerations.md    # SIGNING.md §13 (already written)
│   └── v1-draft/                     # proposed features awaiting implementation + vectors
│
├── test-vectors/                     # THE INTEROP CONTRACT. Any conforming impl runs these.
│   ├── README.md                     # consumption instructions, language-agnostic
│   ├── sidecar/
│   │   ├── valid/                    # MUST verify
│   │   └── invalid/                  # MUST be rejected with the named reason
│   ├── canonicalization/
│   │   ├── mzml/                     # input file + expected canonical hash
│   │   └── d/                        # input file + expected canonical hash
│   └── keys/                         # test-only keypairs, clearly marked
│
└── implementations/
    ├── python/                       # reference implementation (lifted from imspy-simulation)
    │   ├── pyproject.toml
    │   ├── src/mzprov/
    │   └── tests/
    │       └── conformance/          # runs test-vectors/ against this implementation
    └── csharp/                       # independent implementation, R is the author
        └── README.md                 # "implementation goal: pass test-vectors/conformance"
```

The split that matters most:

- **`spec/` is normative.** A claim in `spec/` is binding on every conforming implementation.
- **`docs/` is not normative.** It explains the *why*. Implementations are not required to agree with `docs/` interpretations of design choices, only with `spec/` requirements.
- **`test-vectors/` is the enforcement layer.** It is the only mechanism by which two independent implementations can mechanically verify they agree.

This is the IETF / in-toto / Sigstore / SLSA layout pattern. It is not novel. It is well-trodden because it works.

---

## 4. Order of operations

Five phases, each producing a reviewable artifact. None of phases 1–3 require committing to anything irreversible.

### Phase A — Migration plan (this document)

**Status:** done as of 2026-04-10. Captures the decisions and the target shape so phases B–E have a single source of truth.

### Phase B — Stand up the skeleton

Create the directory tree at `/scratch/timsim-demo/mzprov/`, populate the license files, write the top-level README, and write stub README files in each subdirectory. **No code is moved in this phase.** The deliverable is the artifact we share with R for first reactions.

### Phase C — Break up SIGNING.md into `docs/`

Cut SIGNING.md into the seven `docs/` files listed in §3 above. Mostly mechanical. The §13 security considerations migrate to `spec/security-considerations.md` (it is the most normative section already). Add `docs/faq.md` capturing:

- The mzML `<fileChecksum>` SHA-1 misread (a self-checksum, not a signature, trivially recomputable, broken hash function)
- The full-canonical-hash vs. sampling-based-attestation discussion (full canonical hash strictly dominates n-spectrum sampling for our threat model; sampling is worth considering as a v1-draft defense-in-depth extension)
- The "why not blockchain" question

### Phase D — Lift the Python implementation

Move `imspy_simulation.provenance` into `implementations/python/src/mzprov/`. The PyPI package name is `mzprov`. Keep `imspy_simulation.provenance` as a thin re-export shim:

```python
# imspy_simulation/provenance/__init__.py
from mzprov import (
    sign_simulation_output,
    sign_mzml_output,
    verify,
    # ... full public API ...
)
```

TimSim's `timsim-verify` and `timsim-keys` CLIs become aliases over `mzprov verify` and `mzprov keys`. The existing TimSim integration tests must continue to pass against the shim layer.

Generate the first batch of test vectors from real signing runs against the lifted implementation: valid sidecars (`.d` v0, mzML v0); invalid sidecars covering each rejection path (tampered content hash, tampered payload, wrong key id label, unknown signature algo, cross-format reuse, encoding-tag swap); canonicalization fixtures with input file + expected hash for each invariance the spec claims (whitespace, attribute ordering, binary-array ordering, indexed-vs-non-indexed wrapper).

### Phase E — Write the normative spec

Extract MUST/SHOULD/MAY claims from the existing prose and rewrite them in RFC 2119 style across the seven `spec/` files. Each spec document references the test vectors that enforce its claims. This is the highest-effort writing job in the migration but is informed by every edge case the lifted Python implementation already handles.

### Phase F — Cut over and shorten SIGNING.md

Once `mzprov` has docs/, spec/, and a working Python implementation, replace `SIGNING.md` in this rustims branch with a short pointer:

```markdown
# SIGNING — moved

The provenance and signing work that lived here has moved to a standalone repository:

  → github.com/.../mzprov  (TBD)

That repository contains the full design rationale (`docs/`), the normative
specification (`spec/`), language-agnostic test vectors (`test-vectors/`), and
the Python reference implementation (`implementations/python/`).

TimSim continues to integrate with the Python implementation via
`imspy_simulation.provenance`, which is now a thin wrapper around `mzprov`.
```

Phase F is the only phase with a reverse dependency on rustims. Phases B–E happen entirely in the new repo and can be paused or restarted without disturbing rustims.

---

## 5. Backward compatibility commitments

These commitments hold across the migration so that the move is invisible to existing TimSim users:

- `imspy_simulation.provenance` keeps its current public API. Existing imports continue to work.
- `timsim-verify` and `timsim-keys` continue to exist and accept the same flags.
- Existing sidecars produced by TimSim before the migration verify identically after the migration. There is no sidecar format change in the lift.
- Exit codes `0`–`7` are preserved exactly.
- The signing key location (`~/.config/timsim/keys/signing_key.pem`) and the trusted-keys registry location (`~/.config/timsim/trusted_keys.json`) are preserved. The mzprov CLI may add an alternate config root in addition, but the existing one continues to be honored.

These are not nice-to-haves. They are the commitments that make the lift safe to ship.

---

## 6. Open questions deferred past Phase B

| Question | Earliest decision needed by |
|---|---|
| GitHub home: own org `mzprov`, own org `mzprov-spec`, or under existing user namespace? | Phase F (first push) |
| PyPI publish timing: placeholder package now to claim the name, or wait until Phase D is complete? | Phase D end |
| C# implementation owner: R is the obvious choice but there is no formal commitment yet. | Phase E |
| `v1-draft/` first inhabitant: R's sampling proposal is the natural first candidate. Who writes it? | After R's reply to Phase B skeleton |
| Cross-implementation conformance CI: how do we run test vectors against both Python and C# in one workflow? | Phase E end |

None of these block phases B–E. They become real once there is something to push to GitHub.

---

## 7. What this document does not do

This is a migration plan, not a specification. It does not define the sidecar format, the canonicalization algorithm, the trust model, the verifier behavior, or the security claims. Those live in `spec/` once Phase E is done. Until then the authoritative source is `SIGNING.md` in this branch and the existing Python code under `imspy_simulation.provenance`.
