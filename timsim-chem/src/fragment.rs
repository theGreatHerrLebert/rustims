//! Backbone fragment ions (b / y), and the fragments that *localise a modification*.
//!
//! A peptide broken at the amide bond gives two complementary series: **b ions** (the N-terminal
//! piece) and **y ions** (the C-terminal piece). Their m/z is pure chemistry — a running sum of
//! residue masses — so it belongs on the structure axis, computed once. What a mass spectrometer
//! *does* to their intensities depends on the collision energy, and that is a measurement (predicted
//! by the deep model, per run); this module is only the m/z.
//!
//! # Why this is the module the phospho benchmark needs
//!
//! Two positional isomers — phospho on the S vs the Y of the same peptide — have identical precursor
//! m/z (same composition; see [`crate::isotope`]). They are told apart only by the **fragments that
//! contain one site but not the other**: those shift by the modification mass, the rest do not. Those
//! are the *site-determining* fragments, and enumerating them is what lets a simulator score a
//! localisation tool against the truth instead of, as v1 did, writing `site_probability = 1.0`.
//!
//! # The oracle
//!
//! A b ion and its complementary y ion partition the peptide: `b_i` (neutral) + `y_{n−i}` (neutral)
//! equals the peptide's neutral monoisotopic mass, for every `i`. That identity holds modform by
//! modform (each modification's mass lands in exactly one of the two complements) and is the
//! independent check that the running sums are right.

use crate::mass::{self, monoisotopic_residue, UnknownResidue};

/// Which backbone series a fragment belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IonType {
    /// N-terminal fragment (prefix residues).
    B,
    /// C-terminal fragment (suffix residues + water).
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

/// One fragment ion: its series, how many residues it spans, its charge, and its m/z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fragment {
    pub ion_type: IonType,
    /// Residues spanned: `b3` is the first 3 residues, `y2` the last 2. 1-based, never 0.
    pub ordinal: u16,
    pub charge: u8,
    pub mz: f64,
}

/// m/z of a neutral fragment mass at a given charge: `(neutral + z·proton) / z`.
#[inline]
fn mz(neutral: f64, charge: u8) -> f64 {
    (neutral + charge as f64 * mass::PROTON) / charge as f64
}

/// The **modform mass deltas along the backbone**: `delta[i]` is the mass added at 0-based residue
/// position `i`. Length equals the peptide length. This is the general input — an unmodified peptide
/// passes all zeros — and it is what makes a fragment site-aware: a `b_i` carries every delta at
/// positions `0..i`, a `y_j` every delta at positions `n−j..n`.
///
/// Callers build this from a [`crate::modify::Modform`] by scattering each modification's mass onto
/// its position; keeping the fragment chemistry in terms of a per-position delta means this module
/// never needs to know what a "phospho" is.
pub fn fragment_ions_with_deltas(
    sequence: &str,
    position_deltas: &[f64],
    max_charge: u8,
) -> Result<Vec<Fragment>, UnknownResidue> {
    let bytes = sequence.as_bytes();
    let n = bytes.len();
    assert_eq!(
        position_deltas.len(),
        n,
        "position_deltas must have one entry per residue"
    );
    if n < 2 {
        return Ok(Vec::new()); // a single residue has no backbone bond to break
    }

    // Residue masses once, so both series are simple running sums.
    let res: Vec<f64> = bytes
        .iter()
        .map(|&b| monoisotopic_residue(b).ok_or(UnknownResidue(b as char)))
        .collect::<Result<_, _>>()?;

    let max_charge = max_charge.max(1);
    let mut out = Vec::with_capacity((n - 1) * 2 * max_charge as usize);

    // b ions: prefix sums from the N-terminus. b_i spans residues 0..i.
    let mut prefix = 0.0;
    let mut prefix_delta = 0.0;
    for i in 1..n {
        prefix += res[i - 1];
        prefix_delta += position_deltas[i - 1];
        let neutral = prefix + prefix_delta; // b ion neutral = Σ residues (no water)
        for z in 1..=max_charge {
            out.push(Fragment {
                ion_type: IonType::B,
                ordinal: i as u16,
                charge: z,
                mz: mz(neutral, z),
            });
        }
    }

    // y ions: suffix sums from the C-terminus. y_j spans the last j residues, plus water.
    let mut suffix = 0.0;
    let mut suffix_delta = 0.0;
    for j in 1..n {
        suffix += res[n - j];
        suffix_delta += position_deltas[n - j];
        let neutral = suffix + suffix_delta + mass::WATER_MONOISOTOPIC;
        for z in 1..=max_charge {
            out.push(Fragment {
                ion_type: IonType::Y,
                ordinal: j as u16,
                charge: z,
                mz: mz(neutral, z),
            });
        }
    }

    Ok(out)
}

/// Backbone fragments of an **unmodified** peptide.
pub fn fragment_ions(sequence: &str, max_charge: u8) -> Result<Vec<Fragment>, UnknownResidue> {
    fragment_ions_with_deltas(sequence, &vec![0.0; sequence.len()], max_charge)
}

/// The **site-determining fragments** distinguishing two positional isomers of the same peptide:
/// the fragments whose m/z differs when a modification of mass `delta` sits at `pos_a` versus
/// `pos_b` (0-based). These are exactly the fragments that span one site but not the other, and they
/// are the evidence a localisation search must find.
///
/// Returns `(ion_type, ordinal)` pairs — charge-independent, since a shift at charge 1 is a shift at
/// every charge.
pub fn site_determining(sequence: &str, pos_a: usize, pos_b: usize) -> Vec<(IonType, u16)> {
    let n = sequence.len();
    let (lo, hi) = if pos_a <= pos_b {
        (pos_a, pos_b)
    } else {
        (pos_b, pos_a)
    };
    let mut out = Vec::new();
    // A b_i spans positions 0..i, so it separates the two sites iff exactly one of them is < i,
    // i.e. lo < i <= hi.
    for i in (lo + 1)..=hi.min(n.saturating_sub(1)) {
        out.push((IonType::B, i as u16));
    }
    // A y_j spans the last j positions (n-j..n), so it separates them iff n-j is in (lo, hi],
    // i.e. j in [n-hi, n-lo).
    for j in (n.saturating_sub(hi))..(n - lo) {
        if j >= 1 && j < n {
            out.push((IonType::Y, j as u16));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn find(frags: &[Fragment], t: IonType, ord: u16, z: u8) -> f64 {
        frags
            .iter()
            .find(|f| f.ion_type == t && f.ordinal == ord && f.charge == z)
            .unwrap_or_else(|| panic!("no {}{ord} at z={z}", t.as_str()))
            .mz
    }

    /// Known singly-charged b/y ions for PEPTIDE — a peptide whose fragment masses are widely
    /// tabulated. b2 (PE) and y2 (DE) checked against reference values.
    #[test]
    fn known_fragment_masses() {
        let f = fragment_ions("PEPTIDE", 1).unwrap();
        // b2 = P+E residues + proton = 97.052764 + 129.042593 + 1.007276 = 227.1026
        assert_relative_eq!(find(&f, IonType::B, 2, 1), 227.1026, epsilon = 1e-3);
        // y2 = D+E residues + water + proton = 115.026943 + 129.042593 + 18.010565 + 1.007276
        assert_relative_eq!(find(&f, IonType::Y, 2, 1), 263.0874, epsilon = 1e-3);
    }

    /// THE oracle: b_i and its complement y_{n−i} partition the peptide, so their neutral masses sum
    /// to the peptide's neutral monoisotopic mass. Checked at every split, and with a modification,
    /// whose mass must land in exactly one of the two complements.
    #[test]
    fn complementary_fragments_reconstruct_the_peptide() {
        let seq = "SAMPLERK";
        let n = seq.len();
        let m = mass::monoisotopic(seq).unwrap(); // neutral peptide (incl. water)

        for deltas in [vec![0.0; n], {
            let mut d = vec![0.0; n];
            d[3] = 79.966331; // a phospho somewhere in the middle
            d
        }] {
            let total = m + deltas.iter().sum::<f64>();
            let f = fragment_ions_with_deltas(seq, &deltas, 1).unwrap();
            for i in 1..n {
                // neutral = singly-charged m/z − proton.
                let b = find(&f, IonType::B, i as u16, 1) - mass::PROTON;
                let y = find(&f, IonType::Y, (n - i) as u16, 1) - mass::PROTON;
                assert_relative_eq!(b + y, total, epsilon = 1e-6);
            }
        }
    }

    /// A modification shifts exactly the fragments that span its site, by exactly its mass, and
    /// leaves the others untouched — the fact the whole localisation story rests on.
    #[test]
    fn a_modification_shifts_only_the_fragments_that_span_it() {
        let seq = "PEPTIDEK"; // positions 0..7
        let bare = fragment_ions(seq, 1).unwrap();
        let mut d = vec![0.0; seq.len()];
        d[3] = 79.966331; // phospho on residue index 3 (the T)
        let modded = fragment_ions_with_deltas(seq, &d, 1).unwrap();

        // b3 spans residues 0..3 (does NOT include index 3): unchanged.
        assert_relative_eq!(find(&bare, IonType::B, 3, 1), find(&modded, IonType::B, 3, 1), epsilon = 1e-9);
        // b4 spans 0..4 (includes index 3): shifted by the mod mass.
        assert_relative_eq!(
            find(&modded, IonType::B, 4, 1) - find(&bare, IonType::B, 4, 1),
            79.966331,
            epsilon = 1e-6
        );
        // y4 spans the last 4 (indices 4..8, excludes 3): unchanged.
        assert_relative_eq!(find(&bare, IonType::Y, 4, 1), find(&modded, IonType::Y, 4, 1), epsilon = 1e-9);
        // y5 spans the last 5 (indices 3..8, includes 3): shifted.
        assert_relative_eq!(
            find(&modded, IonType::Y, 5, 1) - find(&bare, IonType::Y, 5, 1),
            79.966331,
            epsilon = 1e-6
        );
    }

    /// Multiply-charged fragments follow (neutral + z·proton)/z, so a 2+ fragment sits at roughly
    /// half its 1+ neutral, and the isotope-style spacing is the proton over the charge.
    #[test]
    fn charge_scaling_is_correct() {
        let f = fragment_ions("SAMPLERK", 2).unwrap();
        let b3_1 = find(&f, IonType::B, 3, 1);
        let b3_2 = find(&f, IonType::B, 3, 2);
        let neutral = b3_1 - mass::PROTON;
        assert_relative_eq!(b3_2, (neutral + 2.0 * mass::PROTON) / 2.0, epsilon = 1e-9);
    }

    /// The site-determining fragments for phospho-on-S (index 2) vs phospho-on-Y (index 5) are the
    /// b/y ions that span one site but not the other — and each of them actually differs in m/z
    /// between the two isomers, while the ones NOT in the set do not. This is the localisation
    /// ground truth, cross-checked against the m/z it predicts.
    #[test]
    fn site_determining_fragments_are_exactly_the_ones_that_differ() {
        let seq = "AASPEYK"; // S at index 2, Y at index 5
        let (a, b) = (2usize, 5usize);
        let determining = site_determining(seq, a, b);

        let mut da = vec![0.0; seq.len()];
        da[a] = 79.966331;
        let mut db = vec![0.0; seq.len()];
        db[b] = 79.966331;
        let fa = fragment_ions_with_deltas(seq, &da, 1).unwrap();
        let fb = fragment_ions_with_deltas(seq, &db, 1).unwrap();

        for f in &fa {
            let mz_b = find(&fb, f.ion_type, f.ordinal, f.charge);
            let differs = (f.mz - mz_b).abs() > 1e-6;
            let listed = determining.contains(&(f.ion_type, f.ordinal));
            assert_eq!(
                differs, listed,
                "{}{} differs={differs} but listed={listed}",
                f.ion_type.as_str(),
                f.ordinal
            );
        }
        assert!(!determining.is_empty(), "there must be fragments that localise the site");
    }

    #[test]
    fn unknown_residues_are_refused() {
        assert!(fragment_ions("PEPTXDE", 1).is_err());
    }

    #[test]
    fn a_single_residue_has_no_fragments() {
        assert!(fragment_ions("K", 1).unwrap().is_empty());
    }
}
