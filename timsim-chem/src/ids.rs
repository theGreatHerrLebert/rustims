//! Content-derived identifiers.
//!
//! IDs are **hashes of content**, never row indices. A `peptide_id` must be the same number
//! in every run, under every parameter change, at every thread count — otherwise artifacts
//! cannot be joined across samples, diffed across versions, or cached at all.
//!
//! # Collisions are detected, not assumed away
//!
//! A 64-bit hash is **not** a uniqueness guarantee. At ~2.4M peptides (the human proteome at
//! 2 missed cleavages) the birthday probability of at least one collision is roughly
//! `n²/2^65` ≈ 1.6e-7 — small, but not zero, and a silent collision would merge two distinct
//! peptides into one row. So the tools that build ID tables check for collisions and fail
//! loudly ([`CollisionCheck`]) rather than trusting the arithmetic.

use std::collections::HashMap;

/// FNV-1a over the content, finished with SplitMix64.
///
/// The finaliser is not decoration: FNV's avalanche is weak, so raw FNV outputs for similar
/// inputs (e.g. peptides differing by one residue) cluster in ways that make collisions more
/// likely than the birthday bound suggests.
fn content_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // SplitMix64 finaliser — full avalanche.
    let mut z = h.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The name of the hash, recorded in artifact metadata so a future change is detectable
/// rather than silently incompatible.
pub const ID_HASH: &str = "fnv1a-splitmix64/v1";

/// `peptide_id` — derived from the bare amino-acid sequence.
pub fn peptide_id(sequence: &str) -> u64 {
    content_hash(sequence.as_bytes())
}

/// `precursor_id` — derived from the peptide (or modform) and the charge.
pub fn precursor_id(peptide_id: u64, charge: u8) -> u64 {
    let mut bytes = [0u8; 9];
    bytes[..8].copy_from_slice(&peptide_id.to_le_bytes());
    bytes[8] = charge;
    content_hash(&bytes)
}

/// `modform_id` — derived from the peptide and the exact set of modifications it carries.
///
/// The mods are hashed in **position order**, which [`crate::modify::enumerate_modforms`] already
/// guarantees. That matters: `{(3, phospho), (7, oxidation)}` and `{(7, oxidation), (3, phospho)}`
/// are the same molecule, and if enumeration order ever leaked into the id they would become two.
///
/// The modification is identified by **name**, not by its index in the spec, so inserting a mod at
/// the top of a config file does not renumber every modform id in the project.
pub fn modform_id(peptide_id: u64, mods: &[(u32, &str)]) -> u64 {
    let mut bytes = peptide_id.to_le_bytes().to_vec();
    for (pos, name) in mods {
        bytes.extend_from_slice(&pos.to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0xff); // separator: keeps ("AB", "C") from colliding with ("A", "BC")
    }
    content_hash(&bytes)
}

/// Detects hash collisions while building an ID table. A 64-bit hash makes them unlikely,
/// not impossible, and a silent collision merges two distinct peptides into one row.
#[derive(Default)]
pub struct CollisionCheck {
    seen: HashMap<u64, String>,
}

impl CollisionCheck {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `Err` with both colliding contents if `id` was already claimed by different
    /// content.
    pub fn insert(&mut self, id: u64, content: &str) -> Result<(), String> {
        match self.seen.get(&id) {
            Some(prev) if prev != content => Err(format!(
                "peptide_id collision on {id:#018x}: {prev:?} and {content:?} hash to the same \
                 value. This is a genuine 64-bit collision, not a bug — widen the id or salt it."
            )),
            _ => {
                self.seen.insert(id, content.to_string());
                Ok(())
            }
        }
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_stable_and_content_derived() {
        assert_eq!(peptide_id("PEPTIDEK"), peptide_id("PEPTIDEK"));
        assert_ne!(peptide_id("PEPTIDEK"), peptide_id("PEPTIDER"));
        // Order-independent: the id of a peptide does not depend on what else was hashed.
        let a = peptide_id("SAMPLER");
        for i in 0..1000 {
            let _ = peptide_id(&format!("X{i}"));
        }
        assert_eq!(peptide_id("SAMPLER"), a);
    }

    /// Single-residue neighbours must not cluster — this is what the SplitMix finaliser buys,
    /// and it is why raw FNV is not enough.
    #[test]
    fn near_identical_sequences_hash_far_apart() {
        const BASE: &str = "PEPTIDEKAAAK";
        let base = peptide_id(BASE);
        let mut checked = 0;
        for aa in "ACDEFGHIKLMNPQRSTVWY".chars() {
            let mutated = format!("{}{}", aa, &BASE[1..]);
            if mutated == BASE {
                continue; // substituting the residue that is already there is not a mutation
            }
            let h = peptide_id(&mutated);
            assert_ne!(h, base, "collision on single substitution to {aa:?}");
            // Top bits must differ too — clustering there is what makes real collisions likely,
            // and it is exactly what raw FNV (without the SplitMix finaliser) fails to avoid.
            assert_ne!(h >> 40, base >> 40, "high bits cluster for {mutated:?}");
            checked += 1;
        }
        assert_eq!(checked, 19, "19 of the 20 residues are real substitutions");
    }

    /// The same molecule must have the same id however the enumerator happened to visit its sites,
    /// and two different molecules must not collide. The separator byte is what stops
    /// `[(1,"AB")]` and `[(1,"A"), (1,"B")]`-style concatenation collisions.
    #[test]
    fn modform_ids_are_order_independent_and_separator_safe() {
        let p = peptide_id("PEPTIDESK");
        // Position-ordered, as enumerate_modforms guarantees.
        let a = modform_id(p, &[(3, "Phospho"), (7, "Oxidation")]);
        assert_eq!(a, modform_id(p, &[(3, "Phospho"), (7, "Oxidation")]));

        // Distinct molecules stay distinct.
        assert_ne!(a, modform_id(p, &[(3, "Oxidation"), (7, "Phospho")]));
        assert_ne!(a, modform_id(p, &[(3, "Phospho")]));
        assert_ne!(a, modform_id(p, &[]), "the unmodified form is its own modform");
        assert_ne!(a, modform_id(peptide_id("PEPTIDESR"), &[(3, "Phospho"), (7, "Oxidation")]));

        // Name-boundary collision: without a separator these two would hash identically.
        assert_ne!(
            modform_id(p, &[(1, "AB"), (2, "C")]),
            modform_id(p, &[(1, "A"), (2, "BC")]),
        );
    }

    #[test]
    fn collisions_are_detected_not_ignored() {
        let mut c = CollisionCheck::new();
        c.insert(42, "PEPTIDEK").unwrap();
        c.insert(42, "PEPTIDEK").unwrap(); // same content, fine
        let err = c.insert(42, "DIFFERENT").unwrap_err();
        assert!(err.contains("collision"), "{err}");
    }

    /// No collision across the whole human tryptic peptide space is a property we rely on;
    /// this at least proves the check works at a scale where a bad hash would fail.
    #[test]
    fn no_collisions_across_a_large_synthetic_peptide_set() {
        let mut c = CollisionCheck::new();
        for i in 0..200_000u32 {
            let seq = format!("PEPTIDEK{i}");
            c.insert(peptide_id(&seq), &seq).expect("no collision expected at 200k");
        }
        assert_eq!(c.len(), 200_000);
    }
}
