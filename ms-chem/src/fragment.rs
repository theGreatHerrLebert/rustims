//! Backbone fragment ions (b / y) — the m/z ladders of a peptide broken at the amide bond.
//!
//! Pure chemistry: a running residue sum plus a terminal group and the proton. b ions are the
//! N-terminal piece (neutral mass = sum of the first *i* residues); y ions are the C-terminal piece
//! (sum of the last *j* residues + water). Their intensities depend on collision energy and are a
//! separate (predicted) concern — this module is only the m/z.
//!
//! ms-chem returns **typed** fragments (ion type + ordinal), not a merged spectrum: a peak's series
//! is information downstream localization needs, and R1 Gate 4 showed mscore's merged `MzSpectrum`
//! destroys it. The complementarity identity `b_i + y_{n-i} = M + 2·proton` (R1 Gate 5, exact to
//! 1e-11) holds by construction here.

use crate::mass::{UnknownResidue, PROTON, WATER};
use crate::residue::residue_monoisotopic_mass;

/// Backbone ion series.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IonType {
    B,
    Y,
}

impl IonType {
    pub fn as_str(&self) -> &'static str {
        match self {
            IonType::B => "b",
            IonType::Y => "y",
        }
    }
}

/// One fragment ion: its series, how many residues it spans (1-based), its charge, and its m/z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fragment {
    pub ion_type: IonType,
    pub ordinal: u16,
    pub charge: u8,
    pub mz: f64,
}

/// All b and y fragment ions of `sequence` at charges `1..=max_charge`. A peptide of length `n`
/// yields ordinals `1..=n-1` per series per charge. Empty for `n < 2`.
pub fn fragment_ions(sequence: &str, max_charge: u8) -> Result<Vec<Fragment>, UnknownResidue> {
    assert!(max_charge >= 1, "max_charge must be >= 1");
    let bytes = sequence.as_bytes();
    let n = bytes.len();
    let res: Vec<f64> = bytes
        .iter()
        .map(|&b| residue_monoisotopic_mass(b).ok_or(UnknownResidue(b as char)))
        .collect::<Result<_, _>>()?;

    let mut out = Vec::new();
    if n < 2 {
        return Ok(out);
    }
    for z in 1..=max_charge {
        let zf = z as f64;
        // b ions: neutral = sum of the first i residues
        let mut acc = 0.0;
        for i in 0..n - 1 {
            acc += res[i];
            out.push(Fragment {
                ion_type: IonType::B,
                ordinal: (i + 1) as u16,
                charge: z,
                mz: (acc + zf * PROTON) / zf,
            });
        }
        // y ions: neutral = sum of the last j residues + water
        let mut acc = 0.0;
        for j in 0..n - 1 {
            acc += res[n - 1 - j];
            out.push(Fragment {
                ion_type: IonType::Y,
                ordinal: (j + 1) as u16,
                charge: z,
                mz: (acc + WATER + zf * PROTON) / zf,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mass::monoisotopic;

    #[test]
    fn b1_and_y1_of_a_known_peptide() {
        let f = fragment_ions("PEPTIDER", 1).unwrap();
        let b1 = f.iter().find(|x| x.ion_type == IonType::B && x.ordinal == 1).unwrap();
        let y1 = f.iter().find(|x| x.ion_type == IonType::Y && x.ordinal == 1).unwrap();
        assert!((b1.mz - 98.06004).abs() < 1e-4, "b1={}", b1.mz);
        assert!((y1.mz - 175.11895).abs() < 1e-4, "y1={}", y1.mz);
    }

    #[test]
    fn complementarity_holds_by_construction() {
        let seq = "PEPTIDER";
        let n = seq.len();
        let m = monoisotopic(seq).unwrap();
        let f = fragment_ions(seq, 1).unwrap();
        for i in 1..n {
            let bi = f.iter().find(|x| x.ion_type == IonType::B && x.ordinal == i as u16);
            let yni = f.iter().find(|x| x.ion_type == IonType::Y && x.ordinal == (n - i) as u16);
            if let (Some(b), Some(y)) = (bi, yni) {
                assert!((b.mz + y.mz - (m + 2.0 * PROTON)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn short_peptides_and_errors() {
        assert!(fragment_ions("A", 1).unwrap().is_empty());
        assert_eq!(fragment_ions("AZ", 1), Err(UnknownResidue('Z')));
    }
}
