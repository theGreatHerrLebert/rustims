//! Elements — monoisotopic masses of the neutral atoms.
//!
//! Canonical per `CHEM_PARITY.md`: element masses are identical across mscore and timsim-chem
//! (Gate 1), so they are adopted verbatim. The biologically-relevant CHNOPS + Se are exposed as
//! named consts so residue and formula masses can be *computed from elements* (a single source of
//! truth — a residue mass can never drift from the element table it is built on). Isotopic
//! abundances (which DID differ — N/S — and were resolved to CIAAW) live in the `isotope` module.

// --- The peptide elements, as named consts (the compute-from-elements basis) ---
/// ¹H monoisotopic mass (Da).
pub const H: f64 = 1.00782503223;
/// ¹²C — the mass scale's anchor, exactly 12.
pub const C: f64 = 12.0;
/// ¹⁴N monoisotopic mass (Da).
pub const N: f64 = 14.00307400443;
/// ¹⁶O monoisotopic mass (Da).
pub const O: f64 = 15.99491461957;
/// ³²S monoisotopic mass (Da).
pub const S: f64 = 31.9720711744;
/// ³¹P monoisotopic mass (Da) — mononuclidic.
pub const P: f64 = 30.97376199842;
/// Selenium base mass (Da) — **⁸⁰Se, the most-abundant isotope** (matches mscore's element table).
///
/// CONVENTION (load-bearing): this ecosystem's "monoisotopic" base for each element is the
/// *most-abundant* isotope, not the lightest. For CHNOPS the two coincide, but selenium is the
/// exception — ⁸⁰Se is most abundant while ⁷⁴Se is lightest, and Se has abundant isotopes *lighter*
/// than the base. So when the `isotope` module lands it MUST base Se's envelope on ⁸⁰Se too (and
/// handle its sub-base peaks), or a Se-containing mass and its envelope would disagree by ~6 Da.
/// The `se_convention_matches_mscore` test pins this.
pub const SE: f64 = 79.9165218;

/// Monoisotopic mass of the most abundant isotope of `symbol` (Da), or `None` if the symbol is not
/// a known element. Backs the sum-formula parser; the CHNOPSSe arms reuse the consts above so the
/// two never diverge.
pub fn monoisotopic_mass(symbol: &str) -> Option<f64> {
    Some(match symbol {
        "H" => H,
        "C" => C,
        "N" => N,
        "O" => O,
        "P" => P,
        "S" => S,
        "Se" => SE,
        "He" => 4.00260325415,
        "Li" => 7.0160034366,
        "Be" => 9.012183065,
        "B" => 11.00930536,
        "F" => 18.99840316273,
        "Ne" => 19.9924401762,
        "Na" => 22.9897692820,
        "Mg" => 23.985041697,
        "Al" => 26.98153853,
        "Si" => 27.97692653465,
        "Cl" => 34.968852682,
        "Ar" => 39.9623831237,
        "K" => 38.963706679,
        "Ca" => 39.96259098,
        "Sc" => 44.95590828,
        "Ti" => 47.9479463,
        "V" => 50.9439595,
        "Cr" => 51.9405075,
        "Mn" => 54.9380455,
        "Fe" => 55.9349375,
        "Co" => 58.9331955,
        "Ni" => 57.9353429,
        "Cu" => 62.9295975,
        "Zn" => 63.9291422,
        "Ga" => 68.9255735,
        "Ge" => 73.9211778,
        "As" => 74.9215965,
        "Br" => 78.9183376,
        "Kr" => 83.911507,
        "Rb" => 84.9117893,
        "Sr" => 87.9056125,
        "Y" => 88.905842,
        "Zr" => 89.9046977,
        "Nb" => 92.906373,
        "Mo" => 97.905404,
        "Tc" => 98.0,
        "Ru" => 101.904349,
        "Rh" => 102.905504,
        "Pd" => 105.903485,
        "Ag" => 106.905093,
        "Cd" => 113.903358,
        "In" => 114.903878,
        "Sn" => 119.902199,
        "Sb" => 120.903818,
        "Te" => 129.906224,
        "I" => 126.904473,
        "Xe" => 131.904155,
        "Cs" => 132.905447,
        "Ba" => 137.905247,
        "La" => 138.906355,
        "Ce" => 139.905442,
        "Pr" => 140.907662,
        "Nd" => 141.907732,
        "Pm" => 145.0,
        "Sm" => 151.919728,
        "Eu" => 152.921225,
        "Gd" => 157.924103,
        "Tb" => 158.925346,
        "Dy" => 163.929171,
        "Ho" => 164.930319,
        "Er" => 165.930290,
        "Tm" => 168.934211,
        "Yb" => 173.938859,
        "Lu" => 174.940770,
        "Hf" => 179.946550,
        "Ta" => 180.947992,
        "W" => 183.950932,
        "Re" => 186.955744,
        "Os" => 191.961467,
        "Ir" => 192.962917,
        "Pt" => 194.964766,
        "Au" => 196.966543,
        "Hg" => 201.970617,
        "Tl" => 204.974427,
        "Pb" => 207.976627,
        "Bi" => 208.980384,
        "Po" => 209.0,
        "At" => 210.0,
        "Rn" => 222.0,
        "Fr" => 223.0,
        "Ra" => 226.0,
        "Ac" => 227.0,
        "Th" => 232.038054,
        "Pa" => 231.035882,
        "U" => 238.050786,
        "Np" => 237.0,
        "Pu" => 244.0,
        "Am" => 243.0,
        "Cm" => 247.0,
        "Bk" => 247.0,
        "Cf" => 251.0,
        "Es" => 252.0,
        "Fm" => 257.0,
        "Md" => 258.0,
        "No" => 259.0,
        "Lr" => 262.0,
        "Rf" => 267.0,
        "Db" => 270.0,
        "Sg" => 271.0,
        "Bh" => 270.0,
        "Hs" => 277.0,
        "Mt" => 276.0,
        "Ds" => 281.0,
        "Rg" => 280.0,
        "Cn" => 285.0,
        "Nh" => 284.0,
        "Fl" => 289.0,
        "Mc" => 288.0,
        "Lv" => 293.0,
        "Ts" => 294.0,
        "Og" => 294.0,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consts_match_the_table() {
        // the named consts and the lookup table must never diverge
        for (sym, val) in [("H", H), ("C", C), ("N", N), ("O", O), ("P", P), ("S", S), ("Se", SE)] {
            assert_eq!(monoisotopic_mass(sym), Some(val), "{sym}");
        }
        assert_eq!(monoisotopic_mass("Xx"), None);
    }

    #[test]
    fn full_table_reaches_oganesson() {
        // the port is faithful through element 118 (superheavies included for completeness)
        assert_eq!(monoisotopic_mass("Og"), Some(294.0));
        assert_eq!(monoisotopic_mass("Rf"), Some(267.0));
    }

    #[test]
    fn se_convention_is_80se() {
        // Pin the most-abundant-isotope base: ⁸⁰Se, not ⁷⁴Se. The isotope module MUST match this
        // base for Se, or a selenocysteine mass and its envelope would disagree by ~6 Da.
        assert!((SE - 79.9165218).abs() < 1e-9, "Se base must be ⁸⁰Se");
    }
}
