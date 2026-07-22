# CHEM_PARITY — R1 findings & canonical semantics for `ms-chem`

**Status:** R1 in progress (2026-07). This records the differential-parity evidence that lets us
collapse three chemistry implementations into one **`ms-chem`** crate *without silently changing
simulation output* — the landmine the split plan (`ECOSYSTEM_SPLIT.md` R1) warned about. The harness
lives in `mscore/tests/chem_parity.rs` (timsim-chem is a dev-dependency of mscore; no cycle).

## Inventory — "3 impls" is really 1 + 1 copy + 1 fresh

| impl | role | disposition |
|---|---|---|
| `mscore/chem` | canonical (full tables, 1028-entry UNIMOD mass table) | **becomes `ms-chem`** |
| `rustms/chem` | **strict reduced copy** — elements/formula/utility byte-identical to mscore (only doc-path comments differ); carries a 132-line UNIMOD subset | folds in, ~zero risk |
| `timsim-chem` | independent v2 (function-organized; own isotope/mass algorithms) | the real parity oracle; its *design patterns* inform ms-chem |

Because `rustms/chem` is a byte-identical copy, the entire parity question reduces to
**`mscore/chem` vs the independently-written `timsim-chem`** (anti-self-consistency: an oracle that
did not inherit mscore's choices).

## Gate results

| Gate | What | Corpus | Result |
|---|---|---|---|
| **1 — monoisotopic mass** | neutral peptide mass | 18,420 seqs | **0 divergences**; max rel 8.4e-9 (pure float rounding) |
| **2 — isotope envelope** | first 6 peaks, normalized | 18,420 seqs | max **1.26e-3** abundance; driven by N/S abundance tables; worst = high-sulfur peptides |
| **3 — modification mass** | 6 sim-critical mods, 3 routes | curated set | all present routes agree ≤ **5.3e-6**; mscore's mass & composition tables are **disjoint** |
| **4 — fragment ions (b/y)** | full b∪y m/z ladders | 8,700 peptides / 38k ions | **0 mismatches** (after isobaric-peak merge); max Δ m/z **2.5e-6** (float + proton) |

## Canonical decisions for `ms-chem`

1. **Element monoisotopic masses** — mscore and timsim are *identical*. Adopt as-is.
2. **Residue masses** — agree to ~1e-8 (mscore *computes* from element masses; timsim *hardcodes*
   residue literals). **Canonical: compute from elements** (single source of truth — a residue mass
   can never drift from the element table it's built on). This is the mscore representation.
3. **Isotopic abundances** — N and S differ (older IUPAC vs newer CIAAW); C/H/O identical.
   **DECISION (locked): adopt timsim-chem's newer CIAAW values** — ¹⁵N=0.00364, ³⁴S=0.0425 (+³⁶S).
   Cost: shifts legacy mscore/v1 isotope envelopes by ~1e-3/peak — accepted, and deliberate.
4. **Modifications** — mscore's mass table (1028) and composition table (17) are *disjoint* (GG/121
   has composition but no mass entry; Trimethyl/37 the reverse). **Canonical: one coverage-consistent
   table pairing composition + mass, cross-checked at load** (timsim's `Modification` pattern — a
   modification is not a scalar; its composition reshapes the isotope envelope, and the two routes to
   its mass validate each other). Seed it from mscore's 1028 mass entries, backfill compositions,
   and require every sim-used id to have both.
5. **Proton constant** — mscore `1.007276466621` (full CODATA) vs timsim `1.007276466` (truncated),
   ~6e-10 Da apart. **Canonical: the full CODATA value `1.007276466621`.** It enters every m/z; the
   difference is sub-nano-Dalton, but there is no reason to carry the truncated one.
6. **Fragment ions** — b/y ladders are identical to 2.5e-6 (Gate 4). Note the *representation*
   difference to preserve a choice on: mscore returns a merged `MzSpectrum` (isobaric b/y collapse
   to one peak); timsim returns typed `Fragment`s (b and y kept distinct). ms-chem should expose the
   **typed** form — a peak's ion type is information the merged spectrum destroys, and the sim's
   modification-localization needs it.

## Remaining R1 work before extraction (R2)

- [ ] **Property-based differential test** (plan requirement): generate random formulas + modified
      peptides, assert mscore ≡ timsim within the documented tolerances *or* an approved
      semantic-difference record — catches parser/ordering/terminal-mod edges the fixed corpus won't.
- [x] **Fragment ions** (Gate 4): b/y m/z ladders identical to 2.5e-6 across 8,700 peptides.
- [ ] **Build `ms-chem`**: element/residue tables (compute-from-elements) + CIAAW abundances +
      unified modification table + the composition↔mass cross-check; port mscore's formula parser.
- [ ] **Migrate + fold**: point mscore and rustms at `ms-chem`, delete the two copies, keep the
      parity suite green (it becomes ms-chem's regression gate).

Only after this does R2 (publish `mscore` on `ms-chem`) proceed.
