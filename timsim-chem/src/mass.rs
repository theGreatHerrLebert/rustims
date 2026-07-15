//! Peptide and protein masses.
//!
//! Two different masses, for two different jobs, and conflating them is a real error:
//!
//! - **Monoisotopic** — for m/z. The mass of the isotopologue made only of the most
//!   abundant isotope of each element. This is what an instrument measures.
//! - **Average** — for bulk quantities. The isotope-abundance-weighted mean. This is what
//!   `--load-ng` means, because nanograms on column is a *bulk* mass, not the mass of one
//!   particular isotopologue.

/// Average mass of a water molecule (Da). Added once per intact chain.
pub const WATER_AVERAGE: f64 = 18.015_28;
/// Monoisotopic mass of a water molecule (Da).
pub const WATER_MONOISOTOPIC: f64 = 18.010_565;

/// A residue whose mass we do not know. Real databases contain these (`X` unknown,
/// `B` = D/N, `Z` = E/Q, `J` = I/L), and silently guessing a mass for them would corrupt
/// the mass balance without saying so.
///
/// # Selenocysteine (`U`) is refused, deliberately
///
/// An earlier version gave `U` a mass, and that was worse than refusing it, in two compounding
/// ways.
///
/// **It was inconsistent.** `mass` knew `U`; [`crate::isotope::composition`] did not. So a
/// selenopeptide passed the digest, got a mass, entered `peptides.parquet` — and was then *silently
/// dropped* by `timsim-precursors`, because it could not be given an isotope envelope. It existed in
/// the structure and had no ions. (Found by review.)
///
/// **The mass itself mixed conventions.** The value used, 150.95364, is cysteine with **⁸⁰Se — the
/// most *abundant* selenium isotope**. But *monoisotopic* means the **lightest** isotope, which for
/// selenium is ⁷⁴Se and gives 144.95959. The isotope machinery convolves upward from the lightest,
/// so simply adding selenium to the composition table would have placed the envelope's monoisotopic
/// peak **6 Da below** the m/z computed from this mass — every selenopeptide silently ~6 Da wrong,
/// with a perfectly plausible-looking comb.
///
/// Selenium is genuinely modellable (six stable isotopes: ⁷⁴ 0.89%, ⁷⁶ 9.37%, ⁷⁷ 7.63%, ⁷⁸ 23.77%,
/// ⁸⁰ 49.61%, ⁸² 8.73%) and doing it properly means a 9-wide abundance array **and** a decision about
/// which convention the monoisotopic mass follows. Until that is done, `U` is refused at the first
/// boundary — the digest — where it is counted and reported. Roughly 25 human proteins are
/// selenoproteins; losing them loudly beats simulating them wrongly in silence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnknownResidue(pub char);

impl std::fmt::Display for UnknownResidue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "non-standard residue {:?}", self.0)
    }
}

#[inline]
fn average_residue(aa: u8) -> Option<f64> {
    Some(match aa {
        b'G' => 57.051_9,
        b'A' => 71.078_8,
        b'S' => 87.078_2,
        b'P' => 97.116_7,
        b'V' => 99.132_6,
        b'T' => 101.105_1,
        b'C' => 103.138_8,
        b'L' | b'I' => 113.159_4,
        b'N' => 114.103_8,
        b'D' => 115.088_6,
        b'Q' => 128.130_7,
        b'K' => 128.174_1,
        b'E' => 129.115_5,
        b'M' => 131.192_6,
        b'H' => 137.141_1,
        b'F' => 147.176_6,
        b'R' => 156.187_5,
        b'Y' => 163.176_0,
        b'W' => 186.213_2,
        _ => return None,
    })
}

#[inline]
pub(crate) fn monoisotopic_residue(aa: u8) -> Option<f64> {
    Some(match aa {
        b'G' => 57.021_464,
        b'A' => 71.037_114,
        b'S' => 87.032_028,
        b'P' => 97.052_764,
        b'V' => 99.068_414,
        b'T' => 101.047_678,
        b'C' => 103.009_185,
        b'L' | b'I' => 113.084_064,
        b'N' => 114.042_927,
        b'D' => 115.026_943,
        b'Q' => 128.058_578,
        b'K' => 128.094_963,
        b'E' => 129.042_593,
        b'M' => 131.040_485,
        b'H' => 137.058_912,
        b'F' => 147.068_414,
        b'R' => 156.101_111,
        b'Y' => 163.063_329,
        b'W' => 186.079_313,
        _ => return None,
    })
}

/// Average molecular weight of an intact chain (Da). Use this for `ng` ↔ `amol`.
pub fn average(sequence: &str) -> Result<f64, UnknownResidue> {
    let mut m = WATER_AVERAGE;
    for &b in sequence.as_bytes() {
        m += average_residue(b).ok_or(UnknownResidue(b as char))?;
    }
    Ok(m)
}

/// Monoisotopic mass of an intact chain (Da). Use this for m/z.
pub fn monoisotopic(sequence: &str) -> Result<f64, UnknownResidue> {
    let mut m = WATER_MONOISOTOPIC;
    for &b in sequence.as_bytes() {
        m += monoisotopic_residue(b).ok_or(UnknownResidue(b as char))?;
    }
    Ok(m)
}

/// Attomoles of a species, given its bulk mass in nanograms and its average MW.
///
/// ```text
///     mass_ng = amount_amol · 1e-18 mol · MW g/mol · 1e9 ng/g = amount_amol · MW · 1e-9
///  ⇒  amount_amol = mass_ng · 1e9 / MW
/// ```
#[inline]
pub fn ng_to_amol(mass_ng: f64, average_mw: f64) -> f64 {
    mass_ng * 1e9 / average_mw
}

/// Nanograms of a species, given its molar amount in attomoles and its average MW.
#[inline]
pub fn amol_to_ng(amount_amol: f64, average_mw: f64) -> f64 {
    amount_amol * average_mw * 1e-9
}

/// Mass of a proton (Da). What a charge actually adds to an ion.
pub const PROTON: f64 = 1.007_276_466;

/// Molecules per attomole (Avogadro × 1e-18). A useful sanity anchor: 1 amol ≈ 602,214
/// molecules.
pub const MOLECULES_PER_AMOL: f64 = 602_214.076;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Angiotensin II — a standard whose masses are widely tabulated.
    #[test]
    fn known_peptide_masses() {
        let seq = "DRVYIHPF";
        assert_relative_eq!(monoisotopic(seq).unwrap(), 1045.5345, epsilon = 1e-3);
        assert_relative_eq!(average(seq).unwrap(), 1046.19, epsilon = 2e-2);
    }

    /// Average mass exceeds monoisotopic, and the gap grows with size — heavier isotopes
    /// contribute more the more atoms there are.
    #[test]
    fn average_exceeds_monoisotopic_and_the_gap_grows() {
        let short = "PEPTIDEK";
        let long = "PEPTIDEK".repeat(10);
        let gap_short = average(short).unwrap() - monoisotopic(short).unwrap();
        let gap_long = average(&long).unwrap() - monoisotopic(&long).unwrap();
        assert!(gap_short > 0.0);
        assert!(gap_long > gap_short * 5.0);
    }

    /// ng ↔ amol must round-trip, and land on the scale the spec claims: 200 ng of a
    /// ~1 kDa peptide is ~200 pmol = 2e8 amol.
    #[test]
    fn ng_amol_round_trips_at_the_right_scale() {
        let mw = 1000.0;
        assert_relative_eq!(ng_to_amol(200.0, mw), 2e8, max_relative = 1e-12);
        assert_relative_eq!(amol_to_ng(ng_to_amol(200.0, mw), mw), 200.0, max_relative = 1e-12);
        // 1 amol is about 602,214 molecules.
        assert_relative_eq!(MOLECULES_PER_AMOL, 602_214.076, epsilon = 1e-3);
    }

    /// Non-standard residues are refused, not silently guessed. A quietly wrong mass would
    /// corrupt the mass balance without failing anything.
    #[test]
    fn non_standard_residues_are_refused() {
        assert_eq!(average("PEPTXDEK"), Err(UnknownResidue('X')));
        assert_eq!(monoisotopic("PEPTBDEK"), Err(UnknownResidue('B')));
    }

    /// **Selenocysteine must be refused by mass EXACTLY as it is by composition.**
    ///
    /// It previously had a mass but no elemental composition, so a selenopeptide passed the digest,
    /// entered the artifacts, and was then silently dropped by the ion layer — present in the
    /// structure, absent from the ions. Two tables disagreeing about which residues exist is the
    /// same class of bug as two stages disagreeing about a bound.
    ///
    /// This test asserts they agree, rather than asserting either one in isolation — which is the
    /// only form that can fail against the bug.
    #[test]
    fn mass_and_composition_agree_on_which_residues_exist() {
        for aa in "ACDEFGHIKLMNPQRSTVWY".chars() {
            let seq = format!("PEPT{aa}DEK");
            assert!(monoisotopic(&seq).is_ok(), "{aa} must have a mass");
            assert!(crate::isotope::composition(&seq).is_ok(), "{aa} must have a composition");
        }
        for aa in ['U', 'X', 'B', 'Z', 'J'] {
            let seq = format!("PEPT{aa}DEK");
            assert!(monoisotopic(&seq).is_err(), "{aa} must be refused by mass");
            assert!(
                crate::isotope::composition(&seq).is_err(),
                "{aa} must be refused by composition — a residue with a mass but no composition \
                 gets a peptide into the digest and then silently no ions"
            );
        }
    }
}
