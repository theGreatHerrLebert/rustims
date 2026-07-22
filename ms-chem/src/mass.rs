//! Peptide mass and m/z.
//!
//! Canonical per `CHEM_PARITY.md`: the proton is the full CODATA value (mscore carried it, timsim
//! truncated), and water is computed from the element table rather than hard-coded, so it tracks
//! `elements` like everything else.

use crate::elements;
use crate::residue::residue_monoisotopic_mass;

/// Mass of a proton (Da), CODATA — what a single charge adds to an ion's mass.
pub const PROTON: f64 = 1.007276466621;

/// Neutral monoisotopic mass of water (Da) = 2·H + O, computed from the element table.
pub const WATER: f64 = 2.0 * elements::H + elements::O;

/// A residue byte not in the residue table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownResidue(pub char);

impl std::fmt::Display for UnknownResidue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown residue {:?}", self.0)
    }
}

impl std::error::Error for UnknownResidue {}

/// Neutral monoisotopic mass of an intact peptide chain (Da): water + the residue sum.
pub fn monoisotopic(sequence: &str) -> Result<f64, UnknownResidue> {
    let mut m = WATER;
    for &b in sequence.as_bytes() {
        m += residue_monoisotopic_mass(b).ok_or(UnknownResidue(b as char))?;
    }
    Ok(m)
}

/// m/z of a neutral species at the given charge (number of added protons). Panics on charge 0.
pub fn mz(neutral_mass: f64, charge: u32) -> f64 {
    assert!(charge >= 1, "charge must be >= 1");
    (neutral_mass + charge as f64 * PROTON) / charge as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_from_elements() {
        assert!((WATER - 18.010565).abs() < 1e-5, "WATER={WATER}");
    }

    #[test]
    fn known_peptide_mass() {
        // PEPTIDE neutral monoisotopic mass ~ 799.35997 Da
        let m = monoisotopic("PEPTIDE").unwrap();
        assert!((m - 799.35997).abs() < 1e-3, "PEPTIDE={m}");
        assert_eq!(monoisotopic("PEPTIDEZ"), Err(UnknownResidue('Z')));
    }

    #[test]
    fn mz_singly_charged() {
        let m = monoisotopic("PEPTIDE").unwrap();
        assert!((mz(m, 1) - (m + PROTON)).abs() < 1e-9);
    }
}
