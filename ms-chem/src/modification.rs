//! Modifications — a single coverage-consistent table pairing mass **and** composition.
//!
//! Per `CHEM_PARITY.md` finding: mscore split modifications across two disjoint tables (1028 masses,
//! 17 compositions) — GG/121 had a composition but no mass, Trimethyl/37 the reverse. A modification
//! is not a scalar: its composition reshapes the isotope envelope (phospho adds HPO₃, GG adds
//! C₄H₆N₂O₂), so mass and composition must travel together. ms-chem adopts timsim-chem's pattern:
//! each [`Modification`] carries both, and [`Modification::validate`] cross-checks them at load — the
//! composition's monoisotopic mass must equal the declared `mass_delta`, so a typo in either is
//! caught immediately instead of silently shifting a modified precursor's m/z.

use crate::formula;

/// One modification: UNIMOD id, the residues it targets, and the two independent routes to its mass
/// (declared `mass_delta` and elemental `composition`) that must agree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Modification {
    pub unimod_id: u32,
    pub name: &'static str,
    /// Residues it can sit on, e.g. `"STY"` for phospho. Empty means a terminal modification.
    pub targets: &'static str,
    pub mass_delta: f64,
    /// Elemental composition of the delta, e.g. `"HO3P"` for phospho.
    pub composition: &'static str,
}

impl Modification {
    /// Cross-check the two routes to the modification's mass. The composition's monoisotopic mass
    /// (from [`crate::formula`], built on the same element table as everything else) must equal the
    /// declared `mass_delta` within 1e-3 Da.
    pub fn validate(&self) -> Result<(), String> {
        let from_comp = formula::monoisotopic_mass(self.composition)
            .map_err(|e| format!("{} (id {}): composition {:?}: {e}", self.name, self.unimod_id, self.composition))?;
        let d = (from_comp - self.mass_delta).abs();
        if d > 1e-3 {
            return Err(format!(
                "{} (id {}): composition {} weighs {from_comp:.5} but mass_delta = {:.5} (Δ{d:.2e})",
                self.name, self.unimod_id, self.composition, self.mass_delta
            ));
        }
        Ok(())
    }
}

/// The curated catalog of sim-relevant modifications, each with mass AND composition. Unlike either
/// of mscore's tables alone, every entry here is coverage-consistent — GG (121) has both, so does
/// Trimethyl (37). (The long tail of UNIMOD masses for search/annotation is a separate mass-only
/// lookup, ported in a follow-up.)
pub const BUILTIN: &[Modification] = &[
    Modification { unimod_id: 1, name: "Acetyl", targets: "K", mass_delta: 42.01057, composition: "C2H2O" },
    Modification { unimod_id: 4, name: "Carbamidomethyl", targets: "C", mass_delta: 57.02146, composition: "C2H3NO" },
    Modification { unimod_id: 21, name: "Phospho", targets: "STY", mass_delta: 79.96633, composition: "HO3P" },
    Modification { unimod_id: 35, name: "Oxidation", targets: "M", mass_delta: 15.99491, composition: "O" },
    Modification { unimod_id: 37, name: "Trimethyl", targets: "K", mass_delta: 42.04695, composition: "C3H6" },
    Modification { unimod_id: 121, name: "GG", targets: "K", mass_delta: 114.04293, composition: "C4H6N2O2" },
];

/// Look up a builtin modification by UNIMOD id.
pub fn by_id(id: u32) -> Option<Modification> {
    BUILTIN.iter().copied().find(|m| m.unimod_id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_cross_checks() {
        // the coverage-consistency guarantee: mass and composition agree for every entry
        for m in BUILTIN {
            m.validate().unwrap_or_else(|e| panic!("{e}"));
        }
    }

    #[test]
    fn ids_are_unique_and_lookup_works() {
        let mut ids: Vec<u32> = BUILTIN.iter().map(|m| m.unimod_id).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "duplicate unimod ids");
        assert_eq!(by_id(21).unwrap().name, "Phospho");
        assert_eq!(by_id(999999), None);
    }
}
