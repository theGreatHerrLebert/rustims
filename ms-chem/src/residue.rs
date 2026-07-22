//! Amino-acid residues — elemental composition and monoisotopic mass.
//!
//! A residue is the amino acid *minus* the water lost forming the peptide bond. Compositions are
//! adopted from the parity-proven timsim-chem table; ms-chem is a **superset** — it adds
//! selenocysteine (`U`), which mscore accepts but timsim's residue table lacked (a coverage item
//! noted in the R1 inventory). Residue masses are **computed from the element consts**, never
//! hard-coded, so they cannot drift from `elements`.

use crate::elements;

/// Elemental composition (atom counts). Covers the atoms that appear in residues and common
/// modifications; `p`/`se` are 0 for the standard amino acids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Composition {
    pub c: u32,
    pub h: u32,
    pub n: u32,
    pub o: u32,
    pub s: u32,
    pub p: u32,
    pub se: u32,
}

impl Composition {
    /// Monoisotopic mass of this composition (Da), summed from the element table.
    pub fn monoisotopic_mass(&self) -> f64 {
        self.c as f64 * elements::C
            + self.h as f64 * elements::H
            + self.n as f64 * elements::N
            + self.o as f64 * elements::O
            + self.s as f64 * elements::S
            + self.p as f64 * elements::P
            + self.se as f64 * elements::SE
    }

    /// Add `times` copies of `other` in place.
    pub fn add(&mut self, other: Composition, times: u32) {
        self.c += other.c * times;
        self.h += other.h * times;
        self.n += other.n * times;
        self.o += other.o * times;
        self.s += other.s * times;
        self.p += other.p * times;
        self.se += other.se * times;
    }
}

/// Elemental composition of an intact peptide chain (one water + every residue), or `None` if any
/// residue byte is unknown.
pub fn peptide_composition(sequence: &str) -> Option<Composition> {
    let mut total = Composition { c: 0, h: 2, n: 0, o: 1, s: 0, p: 0, se: 0 }; // H2O
    for &b in sequence.as_bytes() {
        total.add(residue_composition(b)?, 1);
    }
    Some(total)
}

const fn comp(c: u32, h: u32, n: u32, o: u32, s: u32, se: u32) -> Composition {
    Composition { c, h, n, o, s, p: 0, se }
}

/// Elemental composition of a single residue (amino acid minus one water), or `None` for an unknown
/// residue byte. `L` and `I` are isobaric (same composition).
pub fn residue_composition(aa: u8) -> Option<Composition> {
    Some(match aa {
        b'G' => comp(2, 3, 1, 1, 0, 0),
        b'A' => comp(3, 5, 1, 1, 0, 0),
        b'S' => comp(3, 5, 1, 2, 0, 0),
        b'P' => comp(5, 7, 1, 1, 0, 0),
        b'V' => comp(5, 9, 1, 1, 0, 0),
        b'T' => comp(4, 7, 1, 2, 0, 0),
        b'C' => comp(3, 5, 1, 1, 1, 0),
        b'L' | b'I' => comp(6, 11, 1, 1, 0, 0),
        b'N' => comp(4, 6, 2, 2, 0, 0),
        b'D' => comp(4, 5, 1, 3, 0, 0),
        b'Q' => comp(5, 8, 2, 2, 0, 0),
        b'K' => comp(6, 12, 2, 1, 0, 0),
        b'E' => comp(5, 7, 1, 3, 0, 0),
        b'M' => comp(5, 9, 1, 1, 1, 0),
        b'H' => comp(6, 7, 3, 1, 0, 0),
        b'F' => comp(9, 9, 1, 1, 0, 0),
        b'R' => comp(6, 12, 4, 1, 0, 0),
        b'Y' => comp(9, 9, 1, 2, 0, 0),
        b'W' => comp(11, 10, 2, 1, 0, 0),
        // superset over timsim: selenocysteine = cysteine with Se in place of S. Its MASS is
        // correct under the ⁸⁰Se (most-abundant) convention (see elements::SE). NOTE: Se-containing
        // isotope ENVELOPES need the isotope module to base Se on ⁸⁰Se and handle its sub-base
        // peaks — deferred until that module lands (timsim didn't support U at all).
        b'U' => Composition { c: 3, h: 5, n: 1, o: 1, s: 0, p: 0, se: 1 },
        _ => return None,
    })
}

/// Monoisotopic mass of a residue (Da), computed from its composition and the element table.
pub fn residue_monoisotopic_mass(aa: u8) -> Option<f64> {
    Some(residue_composition(aa)?.monoisotopic_mass())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glycine_and_leucine_isoleucine() {
        // Glycine residue C2H3NO = 57.02146
        let g = residue_monoisotopic_mass(b'G').unwrap();
        assert!((g - 57.02146).abs() < 1e-4, "G={g}");
        // L and I are isobaric
        assert_eq!(residue_composition(b'L'), residue_composition(b'I'));
    }

    #[test]
    fn selenocysteine_is_cysteine_with_selenium() {
        let c = residue_composition(b'C').unwrap();
        let u = residue_composition(b'U').unwrap();
        assert_eq!(u.se, 1);
        assert_eq!(u.s, 0);
        assert_eq!((c.c, c.h, c.n, c.o), (u.c, u.h, u.n, u.o));
    }
}
