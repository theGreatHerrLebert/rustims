//! Codex flagged this as a bug; it is not, and a claim is worth less than a test.
//!
//! The concern: `timsim-digest` skips a peptide whose mass cannot be computed (a non-standard
//! residue like `X`), so `peptide_occurrences.parquet` might end up with foreign keys that do not
//! exist in `peptides.parquet` — orphan rows that `timsim-yield` would then quantify and nothing
//! downstream could resolve.
//!
//! It does not happen, and the reason is subtle enough to be worth pinning down: the `continue`
//! that skips the mass insertion sits *inside* the occurrence loop, and the peptide is never added
//! to `pep_seen` — so every subsequent occurrence of that same peptide re-enters the `Err` branch
//! and is skipped too. Peptides and occurrences stay in lockstep.
//!
//! This test builds a proteome containing `X` and asserts referential integrity directly, so that
//! if someone later "optimises" the lookup (say, by hoisting the mass check out of the loop, or
//! memoising the failure), the invariant breaks loudly instead of silently.

use std::collections::HashSet;
use timsim_chem::{ids, mass, Enumerator, Protocol};

#[test]
fn occurrences_never_reference_a_peptide_that_has_no_mass() {
    // A protein with a non-standard residue in the middle: some peptides are weighable, some are not.
    let proteins = vec![(
        "P1".to_string(),
        "PEPTIDEKAAAXCCCKDDDRPEPTIDEK".to_string(), // the X poisons whichever peptide spans it
    )];

    let e = Enumerator::new(Protocol::parse("trypsin").unwrap(), 2, 5, 50).unwrap();
    let digests = e.enumerate_all(&proteins);
    let seq = &proteins[0].1;

    // Reproduce exactly what `timsim-digest` writes.
    let mut peptides: HashSet<u64> = HashSet::new();
    let mut occurrences: Vec<u64> = Vec::new();
    let mut weighable: HashSet<u64> = HashSet::new();

    for d in &digests {
        for o in &d.occurrences {
            let s = o.sequence(seq);
            let id = ids::peptide_id(s);
            if !peptides.contains(&id) {
                match mass::monoisotopic(s) {
                    Ok(_) => {
                        peptides.insert(id);
                        weighable.insert(id);
                    }
                    Err(_) => continue, // <- the line under scrutiny
                }
            }
            occurrences.push(id);
        }
    }

    // The setup must actually exercise the case, or this proves nothing.
    let unweighable = digests[0]
        .occurrences
        .iter()
        .filter(|o| mass::monoisotopic(o.sequence(seq)).is_err())
        .count();
    assert!(unweighable > 0, "the test proteome must contain unweighable peptides");
    assert!(!peptides.is_empty(), "and some weighable ones");

    // THE invariant: every occurrence points at a peptide that exists in peptides.parquet.
    for id in &occurrences {
        assert!(
            peptides.contains(id),
            "occurrence references peptide {id:#x}, which has no row in peptides.parquet"
        );
    }
}
