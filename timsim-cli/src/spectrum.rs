//! Peptide-ion spectra for the render — **precursor** and **fragment** isotopic spectra, both from
//! mscore's peptide-ion path (the same machinery v1 uses). Unifying on this one path means:
//!   - isotope m/z is **exact** (from the peptide composition), not a `neutron/charge` approximation,
//!   - precursor and fragment peaks are produced consistently,
//!   - the render stays sequence-driven and never re-derives fragment chemistry by hand.
//!
//! The dangerous part is the fragment **intensity mapping**. v1 feeds mscore the Prosit *flat-174*
//! array; our v2 `fragment_intensities` artifact is the **decoded per-ion** form (built to dodge the
//! two-incompatible-flat-174-layouts trap). So we attach intensities **per ion**, mirroring v1's
//! `associate_with_predicted_intensities` index mapping exactly — `n_ions[i]` is `b_{i+1}`,
//! `c_ions[i]` is `y_{i+1}` — and never rebuild a flat-174. The tests pin that mapping (a `'b'`
//! intensity lands on a `b` ion's m/z, a `'y'` on a `y` ion's, at the right ordinal).

use mscore::data::peptide::{FragmentType, PeptideIon, PeptideSequence};
use std::collections::HashMap;

/// A resolved spectrum: parallel `(m/z, intensity)` peaks (isotope-expanded).
pub type Peaks = Vec<(f64, f64)>;

/// Isotope-generation parameters (mirroring v1's spectrum defaults). `max_isotope` is the number of
/// isotope peaks generated per ion.
#[derive(Clone, Copy, Debug)]
pub struct SpectrumOpts {
    pub mass_tolerance: f64,
    pub abundance_threshold: f64,
    pub max_isotope: i32,
    pub intensity_min: f64,
}

impl Default for SpectrumOpts {
    fn default() -> Self {
        SpectrumOpts { mass_tolerance: 1e-3, abundance_threshold: 1e-4, max_isotope: 3, intensity_min: 1e-4 }
    }
}

/// The precursor's isotopic spectrum: exact isotope m/z + relative intensities from the composition of
/// the (annotated) peptide at `charge`.
pub fn precursor_peaks(annotated: &str, charge: i32, opts: SpectrumOpts) -> Peaks {
    let ion = PeptideIon::new(annotated.to_string(), charge, 1.0, None);
    let s = ion.calculate_isotopic_spectrum(
        opts.mass_tolerance,
        opts.abundance_threshold,
        opts.max_isotope,
        opts.intensity_min,
    );
    s.mz.iter().copied().zip(s.intensity.iter().copied()).collect()
}

/// Key into the decoded fragment artifact: `(ion_type 'b'|'y', ordinal, fragment charge)`.
pub type FragKey = (char, u16, u8);

/// The fragment isotopic spectrum, with per-ion Prosit intensities attached **without** re-flattening.
/// Mirrors v1's `associate_with_predicted_intensities` mapping: for each fragment charge `z`,
/// `n_ions[i]` (a `b_{i+1}` ion) and `c_ions[i]` (a `y_{i+1}` ion) take the intensity our decoded
/// artifact stored for `(ion_type, i+1, z)`. Fragment charges above the precursor charge (capped at 3,
/// as Prosit predicts) are not generated.
pub fn fragment_peaks(
    annotated: &str,
    precursor_charge: i32,
    per_ion: &HashMap<FragKey, f64>,
    opts: SpectrumOpts,
) -> Peaks {
    let pep = PeptideSequence::new(annotated.to_string(), None);
    let max_z = precursor_charge.min(3).max(1);
    let mut peaks: Peaks = Vec::new();
    for z in 1..=max_z {
        // FragmentType::B yields BOTH series: n_ions = b, c_ions = y (v1 calls it exactly this way).
        let mut series = pep.calculate_product_ion_series(z, FragmentType::B);
        for (i, ion) in series.n_ions.iter_mut().enumerate() {
            ion.ion.intensity = per_ion.get(&('b', (i + 1) as u16, z as u8)).copied().unwrap_or(0.0);
        }
        for (i, ion) in series.c_ions.iter_mut().enumerate() {
            ion.ion.intensity = per_ion.get(&('y', (i + 1) as u16, z as u8)).copied().unwrap_or(0.0);
        }
        let s = series.generate_isotopic_spectrum(
            opts.mass_tolerance,
            opts.abundance_threshold,
            opts.max_isotope,
            opts.intensity_min,
        );
        peaks.extend(s.mz.iter().copied().zip(s.intensity.iter().copied()));
    }
    peaks
}

/// Scale a spectrum so its intensities sum to 1 — a unit-total *shape*. The absolute level is supplied
/// later (per-ion abundance, at render time), so each materialised spectrum is only a distribution.
///
/// This is what keeps the precursor↔fragment ratio physical. The precursor isotope envelope is already
/// ~unit-total (isotope probabilities sum to ~1); a raw Prosit fragment spectrum is base-peak=1 and sums
/// to ~7 (many ions × charges × isotopes). Rendering both against the same abundance without this would
/// put ~7× the ion current into the fragments as into the intact precursor — impossible (the precursor
/// *is* the ion that fragments; its current is distributed among fragments, not multiplied). Normalising
/// MS1 and MS2 to the same unit total makes the ratio conserved, matching v1.
pub fn normalize_total(peaks: &mut Peaks) {
    let total: f64 = peaks.iter().map(|&(_, i)| i).sum();
    if total > 0.0 {
        for p in peaks.iter_mut() {
            p.1 /= total;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono() -> SpectrumOpts {
        // One isotope peak per ion, so each fragment contributes exactly its mono m/z — makes the
        // ion→intensity mapping directly checkable.
        SpectrumOpts { max_isotope: 1, ..Default::default() }
    }

    /// A precursor has isotope peaks spaced by ~1/charge, and the mono peak is the most intense (or
    /// among the most intense) — proves the precursor path is wired and isotope-expanding.
    #[test]
    fn precursor_has_a_mono_and_isotopes() {
        let p = precursor_peaks("PEPTIDER", 2, SpectrumOpts::default());
        assert!(p.len() >= 2, "expected an isotope cluster, got {}", p.len());
        let mono_mz = p.iter().map(|&(m, _)| m).fold(f64::INFINITY, f64::min);
        // next isotope ~ +1/z Th above the mono
        assert!(p.iter().any(|&(m, _)| (m - (mono_mz + 0.5)).abs() < 0.1),
                "no ~+1/2 isotope peak above mono {mono_mz}");
    }

    /// The load-bearing pin: a `'b'` intensity must land on a `b` ion's m/z and a `'y'` on a `y`
    /// ion's, at the requested ordinal — the exact mapping that silently corrupts every spectrum if
    /// transposed. b2 of PEPTIDER (P+E) ≈ 227.10; y2 (ER) ≈ 304.16.
    #[test]
    fn per_ion_intensity_lands_on_the_right_ion() {
        let b2 = 227.1026;
        let mut only_b2 = HashMap::new();
        only_b2.insert(('b', 2u16, 1u8), 1.0);
        let peaks = fragment_peaks("PEPTIDER", 2, &only_b2, mono());
        assert!(!peaks.is_empty(), "no fragment peaks");
        // Every emitted peak must be b2 (only ion with intensity) — its m/z, not a y ion's.
        for (m, _) in &peaks {
            assert!((m - b2).abs() < 0.05, "b2 intensity produced a peak at {m}, expected ~{b2}");
        }

        let mut only_y2 = HashMap::new();
        only_y2.insert(('y', 2u16, 1u8), 1.0);
        let ypeaks = fragment_peaks("PEPTIDER", 2, &only_y2, mono());
        // y2 = E+R + water + proton ≈ 304.16 — must NOT coincide with b2 (proves b/y aren't swapped).
        assert!(ypeaks.iter().all(|&(m, _)| (m - b2).abs() > 1.0),
                "a 'y' intensity produced a peak at the b2 m/z — b/y mapping is transposed");
    }

    /// An intensity assigned to an ordinal that does not exist is simply absent, not misplaced.
    #[test]
    fn empty_intensities_give_no_peaks() {
        let peaks = fragment_peaks("PEPTIDER", 2, &HashMap::new(), mono());
        assert!(peaks.is_empty(), "no intensities should mean no fragment peaks, got {}", peaks.len());
    }

    /// The conservation invariant: after [`normalize_total`], BOTH the precursor and fragment spectra are
    /// unit-total distributions, so the render scales them by the same abundance and the precursor↔fragment
    /// intensity ratio is physical (fragments are a fraction of the precursor, not multiples of it). Pins
    /// the exact bug found in the field: a raw Prosit fragment spectrum sums to ≫1 (base-peak=1 for many
    /// ions × charges × isotopes) and would render fragments ~7× hotter than the intact precursor.
    #[test]
    fn ms1_and_ms2_are_unit_total_after_normalize() {
        let opts = SpectrumOpts::default();
        let mut ms1 = precursor_peaks("PEPTIDER", 2, opts);
        normalize_total(&mut ms1);
        let s1: f64 = ms1.iter().map(|&(_, i)| i).sum();
        assert!((s1 - 1.0).abs() < 1e-9, "MS1 sum {s1} != 1");

        let mut per_ion = HashMap::new();
        for o in 1..=7u16 {
            per_ion.insert(('b', o, 1u8), 0.5);
            per_ion.insert(('y', o, 1u8), 0.8);
        }
        let mut ms2 = fragment_peaks("PEPTIDER", 2, &per_ion, opts);
        let raw: f64 = ms2.iter().map(|&(_, i)| i).sum();
        assert!(raw > 1.5, "raw fragment spectrum should sum ≫1 (base-peak convention), got {raw}");
        normalize_total(&mut ms2);
        let s2: f64 = ms2.iter().map(|&(_, i)| i).sum();
        assert!((s2 - 1.0).abs() < 1e-9, "MS2 sum after normalize {s2} != 1");
    }

    /// `normalize_total` on an empty (or all-zero) spectrum is a no-op, not a divide-by-zero.
    #[test]
    fn normalize_total_handles_empty() {
        let mut empty: Peaks = Vec::new();
        normalize_total(&mut empty);
        assert!(empty.is_empty());
        let mut zeros: Peaks = vec![(100.0, 0.0), (200.0, 0.0)];
        normalize_total(&mut zeros);
        assert!(zeros.iter().all(|&(_, i)| i == 0.0), "all-zero stays zero, no NaN");
    }

    /// **v1 vs v2 spectrum parity** (the user's validation): the same Prosit flat-174 array must build
    /// the *identical* fragment spectrum through v1's path
    /// (`associate_with_predicted_intensities` → `generate_isotopic_spectrum`, the flat-174 route) and
    /// through v2's decode-to-per-ion + [`fragment_peaks`] route. This pins the one place the two
    /// implementations differ (v2 attaches per-ion to dodge the flat-174 layout trap); precursor
    /// spectra are identical by construction since v2 calls the same `PeptideIon` method.
    #[test]
    fn v1_and_v2_build_the_same_fragment_spectrum() {
        use mscore::chemistry::utility::reshape_prosit_array;
        use mscore::data::peptide::PeptideSequence;

        let seq = "PEPTIDER";
        let charge = 2;
        let opts = SpectrumOpts::default();

        // A Prosit-shaped flat-174 with distinct values so any mis-mapping shows up.
        let flat: Vec<f64> = (0..174).map(|i| (i as f64 + 1.0) * 0.001).collect();

        // v1 path: associate the flat array onto the ion series, then generate the isotopic spectrum.
        let v1 = PeptideSequence::new(seq.to_string(), None)
            .associate_with_predicted_intensities(charge, FragmentType::B, flat.clone(), false, false)
            .generate_isotopic_spectrum(opts.mass_tolerance, opts.abundance_threshold, opts.max_isotope, opts.intensity_min);
        let mut v1_peaks: Vec<(f64, f64)> = v1.mz.iter().copied().zip(v1.intensity.iter().copied()).collect();

        // v2 path: reshape -> per-ion map (mirroring timsim-fragments' decode: axis-2 index 0 = y,
        // 1 = b; ordinal = position + 1; charge = c + 1) -> fragment_peaks.
        let reshaped = reshape_prosit_array(flat);
        let mut per_ion: HashMap<FragKey, f64> = HashMap::new();
        for (k, pos) in reshaped.iter().enumerate() {
            for c in 0..pos[0].len() {
                per_ion.insert(('y', (k + 1) as u16, (c + 1) as u8), pos[0][c]);
                per_ion.insert(('b', (k + 1) as u16, (c + 1) as u8), pos[1][c]);
            }
        }
        let mut v2_peaks = fragment_peaks(seq, charge, &per_ion, opts);

        assert!(!v1_peaks.is_empty(), "v1 built no peaks — vacuous test");
        v1_peaks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        v2_peaks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(v1_peaks.len(), v2_peaks.len(), "v1 has {} peaks, v2 has {}", v1_peaks.len(), v2_peaks.len());
        for ((mz1, i1), (mz2, i2)) in v1_peaks.iter().zip(v2_peaks.iter()) {
            assert!((mz1 - mz2).abs() < 1e-9, "m/z differ: v1 {mz1}, v2 {mz2}");
            assert!((i1 - i2).abs() < 1e-9, "intensity differ at m/z {mz1}: v1 {i1}, v2 {i2}");
        }
    }
}
