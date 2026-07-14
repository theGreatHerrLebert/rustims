//! Elemental composition and isotope envelopes.
//!
//! The isotope pattern is not a decoration — it is what a mass spectrometer actually sees. A
//! peptide does not produce one peak; it produces a comb whose shape is fixed by how many carbons,
//! nitrogens, oxygens and sulfurs it contains, and whose spacing is the neutron mass.
//!
//! # How it is computed
//!
//! Each element's natural isotope abundances form a small probability distribution over "how many
//! extra neutrons". A molecule with `n` atoms of that element has the `n`-fold **convolution** of
//! that distribution. Convolving across all elements gives the molecule's envelope. This is exact
//! (to the truncation depth), not an averagine approximation — we know the composition, so there is
//! no reason to guess it.

use crate::mass;

/// Atom counts for a peptide: C, H, N, O, S. Everything a proteomic isotope envelope needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Composition {
    pub c: u32,
    pub h: u32,
    pub n: u32,
    pub o: u32,
    pub s: u32,
    /// Phosphorus. No amino acid contains it, but **phospho does** — and it is monoisotopic
    /// (³¹P is 100% abundant), so it shifts the mass without touching the envelope's shape.
    pub p: u32,
}

impl Composition {
    fn add(&mut self, o: Composition, times: u32) {
        self.c += o.c * times;
        self.h += o.h * times;
        self.n += o.n * times;
        self.o += o.o * times;
        self.s += o.s * times;
        self.p += o.p * times;
    }
}

const fn comp(c: u32, h: u32, n: u32, o: u32, s: u32) -> Composition {
    Composition { c, h, n, o, s, p: 0 }
}

/// Residue composition (the amino acid *minus* the water lost in the peptide bond).
fn residue(aa: u8) -> Option<Composition> {
    Some(match aa {
        b'G' => comp(2, 3, 1, 1, 0),
        b'A' => comp(3, 5, 1, 1, 0),
        b'S' => comp(3, 5, 1, 2, 0),
        b'P' => comp(5, 7, 1, 1, 0),
        b'V' => comp(5, 9, 1, 1, 0),
        b'T' => comp(4, 7, 1, 2, 0),
        b'C' => comp(3, 5, 1, 1, 1),
        b'L' | b'I' => comp(6, 11, 1, 1, 0),
        b'N' => comp(4, 6, 2, 2, 0),
        b'D' => comp(4, 5, 1, 3, 0),
        b'Q' => comp(5, 8, 2, 2, 0),
        b'K' => comp(6, 12, 2, 1, 0),
        b'E' => comp(5, 7, 1, 3, 0),
        b'M' => comp(5, 9, 1, 1, 1),
        b'H' => comp(6, 7, 3, 1, 0),
        b'F' => comp(9, 9, 1, 1, 0),
        b'R' => comp(6, 12, 4, 1, 0),
        b'Y' => comp(9, 9, 1, 2, 0),
        b'W' => comp(11, 10, 2, 1, 0),
        _ => return None,
    })
}

/// Elemental composition of an intact peptide chain (residues + one water).
pub fn composition(sequence: &str) -> Result<Composition, mass::UnknownResidue> {
    let mut total = comp(0, 2, 0, 1, 0); // H2O
    for &b in sequence.as_bytes() {
        let r = residue(b).ok_or(mass::UnknownResidue(b as char))?;
        total.add(r, 1);
    }
    Ok(total)
}

/// Natural abundances, indexed by *extra neutrons* (0 = the lightest isotope).
/// IUPAC representative values.
const AB_C: [f64; 2] = [0.9893, 0.0107]; //  12C, 13C
const AB_H: [f64; 2] = [0.999_885, 0.000_115]; //   1H,  2H
const AB_N: [f64; 2] = [0.996_36, 0.003_64]; //  14N, 15N
const AB_O: [f64; 3] = [0.997_57, 0.000_38, 0.002_05]; //  16O, 17O, 18O
const AB_S: [f64; 5] = [0.9499, 0.0075, 0.0425, 0.0, 0.0001]; //  32S, 33S, 34S, --, 36S
/// Phosphorus is **mononuclidic** — ³¹P is 100% abundant. It moves the mass and leaves the comb's
/// shape untouched, which is why a phosphopeptide's envelope looks like its unmodified self's.
const AB_P: [f64; 1] = [1.0];

/// Mass difference per additional neutron (Da). The ¹³C−¹²C spacing dominates a peptide's comb.
pub const NEUTRON: f64 = 1.003_355;

/// Convolve `dist` with itself `n` times, keeping `depth` peaks. Uses binary exponentiation, so a
/// 300-carbon peptide costs ~8 convolutions, not 300.
fn self_convolve(dist: &[f64], n: u32, depth: usize) -> Vec<f64> {
    let mut result = vec![0.0; depth];
    result[0] = 1.0;
    if n == 0 {
        return result;
    }
    let mut base: Vec<f64> = {
        let mut b = vec![0.0; depth];
        for (i, &v) in dist.iter().take(depth).enumerate() {
            b[i] = v;
        }
        b
    };
    let mut e = n;
    while e > 0 {
        if e & 1 == 1 {
            result = convolve(&result, &base, depth);
        }
        e >>= 1;
        if e > 0 {
            base = convolve(&base, &base, depth);
        }
    }
    result
}

fn convolve(a: &[f64], b: &[f64], depth: usize) -> Vec<f64> {
    let mut out = vec![0.0; depth];
    for (i, &x) in a.iter().enumerate() {
        if x == 0.0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            if i + j >= depth {
                break;
            }
            out[i + j] += x * y;
        }
    }
    out
}

/// A modification's elemental composition, which may **remove** atoms as well as add them
/// (pyroglutamate loses NH₃; a dehydration loses H₂O). Hence signed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompositionDelta {
    pub c: i32,
    pub h: i32,
    pub n: i32,
    pub o: i32,
    pub s: i32,
    pub p: i32,
}

/// Monoisotopic mass of a composition delta (Da). Used to **cross-check** the `mass_delta` a
/// modification spec declares: two independent routes to one number, so a typo in either is caught
/// rather than silently shifting every modified precursor's m/z.
pub fn delta_mass(d: CompositionDelta) -> f64 {
    const M_C: f64 = 12.0;
    const M_H: f64 = 1.007_825_032;
    const M_N: f64 = 14.003_074_004;
    const M_O: f64 = 15.994_914_620;
    const M_S: f64 = 31.972_071_174;
    const M_P: f64 = 30.973_761_998;
    d.c as f64 * M_C
        + d.h as f64 * M_H
        + d.n as f64 * M_N
        + d.o as f64 * M_O
        + d.s as f64 * M_S
        + d.p as f64 * M_P
}

/// Parse a chemical formula such as `"HO3P"`, `"C2H3NO"`, or `"H-1N-1"` (a loss).
///
/// Only C, H, N, O, S are modelled — the elements that shape a proteomic isotope envelope. A
/// formula naming anything else is **refused**, not silently ignored: quietly dropping the P from
/// phospho would leave the mass right and the envelope wrong, which is exactly the kind of error
/// that survives every summary statistic.
pub fn parse_formula(formula: &str) -> Result<CompositionDelta, String> {
    let mut d = CompositionDelta::default();
    let bytes = formula.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if !ch.is_ascii_alphabetic() {
            return Err(format!("formula {formula:?}: unexpected {ch:?} at position {i}"));
        }
        i += 1;
        // Signed count; absent means 1.
        let start = i;
        if i < bytes.len() && bytes[i] == b'-' {
            i += 1;
        }
        while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
            i += 1;
        }
        let count: i32 = if i == start {
            1
        } else {
            formula[start..i]
                .parse()
                .map_err(|_| format!("formula {formula:?}: bad count {:?}", &formula[start..i]))?
        };
        match ch {
            'C' => d.c += count,
            'H' => d.h += count,
            'N' => d.n += count,
            'O' => d.o += count,
            'S' => d.s += count,
            'P' => d.p += count,
            // Selenium, metals, and the heavy isotopes of an isobaric label are all real, and none
            // of them is modelled here.
            other => {
                return Err(format!(
                    "formula {formula:?}: element {other:?} is not modelled. Only C, H, N, O, S and \
                     P are; dropping an element silently would leave the mass right and the \
                     envelope wrong."
                ))
            }
        }
    }
    if d == CompositionDelta::default() {
        return Err(format!("formula {formula:?} is empty"));
    }
    Ok(d)
}

impl Composition {
    /// Apply a modification's composition delta. Fails rather than wrapping if the delta would take
    /// an atom count below zero — a `u32` underflow here would produce a ~4-billion-atom peptide and
    /// an isotope envelope that is not merely wrong but nonsensical.
    pub fn apply(self, d: CompositionDelta) -> Result<Composition, String> {
        let f = |name: &str, base: u32, delta: i32| -> Result<u32, String> {
            let v = base as i64 + delta as i64;
            if v < 0 {
                return Err(format!(
                    "modification removes more {name} than the peptide has ({base} + {delta})"
                ));
            }
            Ok(v as u32)
        };
        Ok(Composition {
            c: f("C", self.c, d.c)?,
            h: f("H", self.h, d.h)?,
            n: f("N", self.n, d.n)?,
            o: f("O", self.o, d.o)?,
            s: f("S", self.s, d.s)?,
            p: f("P", self.p, d.p)?,
        })
    }
}

/// The isotope envelope of a peptide: relative intensities of the monoisotopic peak and its
/// `depth - 1` heavier neighbours, normalised to sum to 1.
///
/// Exact for the given composition — no averagine guess, because we know the formula.
pub fn envelope(sequence: &str, depth: usize) -> Result<Vec<f32>, mass::UnknownResidue> {
    Ok(envelope_of(composition(sequence)?, depth))
}

/// As [`envelope`], for a composition already in hand — a **modform's**, say, which is the peptide's
/// composition plus its modifications'. A modified peptide does not merely weigh more: phospho adds
/// HPO₃ and GG adds C₄H₆N₂O₂, and each reshapes the comb. Shifting the mass while keeping the
/// unmodified peptide's envelope would put the peaks in the right place with the wrong heights.
pub fn envelope_of(c: Composition, depth: usize) -> Vec<f32> {
    let depth = depth.max(1);

    let mut dist = self_convolve(&AB_C, c.c, depth);
    for (ab, n) in [
        (&AB_H[..], c.h),
        (&AB_N[..], c.n),
        (&AB_O[..], c.o),
        (&AB_S[..], c.s),
        (&AB_P[..], c.p),
    ] {
        if n > 0 {
            dist = convolve(&dist, &self_convolve(ab, n, depth), depth);
        }
    }

    let total: f64 = dist.iter().sum();
    dist.iter().map(|v| (v / total) as f32).collect()
}

/// m/z of isotope peak `k` for a peptide of monoisotopic mass `mono` at charge `z`.
#[inline]
pub fn mz(mono_mass: f64, charge: u8, k: usize) -> f64 {
    debug_assert!(charge >= 1);
    (mono_mass + k as f64 * NEUTRON + charge as f64 * mass::PROTON) / charge as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The composition must reproduce the monoisotopic mass computed independently from residue
    /// masses. Two different routes to the same number — a real cross-check, not a restatement.
    #[test]
    fn composition_reproduces_the_monoisotopic_mass() {
        // Element monoisotopic masses.
        const M_C: f64 = 12.0;
        const M_H: f64 = 1.007_825_032;
        const M_N: f64 = 14.003_074_004;
        const M_O: f64 = 15.994_914_620;
        const M_S: f64 = 31.972_071_174;

        for seq in ["PEPTIDEK", "DRVYIHPF", "MCSSSKWR", "AAAAAAAAAA"] {
            let c = composition(seq).unwrap();
            let from_atoms = c.c as f64 * M_C
                + c.h as f64 * M_H
                + c.n as f64 * M_N
                + c.o as f64 * M_O
                + c.s as f64 * M_S;
            let from_residues = mass::monoisotopic(seq).unwrap();
            assert_relative_eq!(from_atoms, from_residues, epsilon = 1e-4);
        }
    }

    /// The envelope is a probability distribution.
    #[test]
    fn the_envelope_sums_to_one() {
        for seq in ["PEPTIDEK", "MCSSSKWR", "PEPTIDEK".repeat(6).as_str()] {
            let e = envelope(seq, 8).unwrap();
            let total: f32 = e.iter().sum();
            assert_relative_eq!(total, 1.0, epsilon = 1e-5);
            assert!(e.iter().all(|&x| x >= 0.0));
        }
    }

    /// Bigger peptides have relatively less monoisotopic peak — more carbons, more chances that at
    /// least one is ¹³C. This is why large peptides show a shifted, broadened comb.
    #[test]
    fn the_monoisotopic_fraction_falls_with_size() {
        let small = envelope("PEPTIDEK", 10).unwrap()[0];
        let big = envelope(&"PEPTIDEK".repeat(6), 10).unwrap()[0];
        assert!(small > big, "small {small} should exceed big {big}");
        // Angiotensin-scale peptide: monoisotopic peak is roughly half the envelope.
        assert!((0.4..0.7).contains(&small), "got {small}");
    }

    /// Sulfur is the giveaway: ³⁴S is 4.25% abundant and sits **two** neutrons up, so a
    /// cysteine/methionine-rich peptide has an anomalously tall M+2. Any isotope model that ignores
    /// sulfur gets this wrong.
    #[test]
    fn sulfur_lifts_the_m_plus_2_peak() {
        let with_s = envelope("MCMCMCMK", 6).unwrap();
        let without = envelope("AAAAAAAK", 6).unwrap();
        let ratio_s = with_s[2] / with_s[0];
        let ratio_n = without[2] / without[0];
        assert!(
            ratio_s > 3.0 * ratio_n,
            "sulfur must lift M+2: {ratio_s:.4} vs {ratio_n:.4}"
        );
    }

    /// m/z spacing between isotope peaks is the neutron mass divided by charge — the thing a
    /// deconvolution algorithm keys on.
    #[test]
    fn isotope_spacing_is_the_neutron_mass_over_charge() {
        let m = mass::monoisotopic("PEPTIDEK").unwrap();
        for z in 1..=4u8 {
            let d = mz(m, z, 1) - mz(m, z, 0);
            assert_relative_eq!(d, NEUTRON / z as f64, epsilon = 1e-9);
        }
        // And the monoisotopic m/z is the textbook (M + z·proton)/z.
        assert_relative_eq!(
            mz(m, 2, 0),
            (m + 2.0 * mass::PROTON) / 2.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn unknown_residues_are_refused() {
        assert!(composition("PEPTXDEK").is_err());
        assert!(envelope("PEPTXDEK", 4).is_err());
    }

    /// **The spec cross-check.** A modification declares both a formula and a `mass_delta`. Those
    /// are two independent routes to one number, and they must agree — a typo in either would shift
    /// every modified precursor's m/z while every other number in the run stayed plausible.
    ///
    /// The values here are the real UNIMOD deltas.
    #[test]
    fn formula_mass_reproduces_the_declared_unimod_delta() {
        for (formula, declared, name) in [
            ("C2H3NO", 57.021464, "Carbamidomethyl"),
            ("O", 15.994915, "Oxidation"),
            ("HO3P", 79.966331, "Phospho"),
            ("C4H6N2O2", 114.042927, "GG"),
            ("C2H2O", 42.010565, "Acetyl"),
            ("C2H2O", 42.010565, "Trimethyl-ish"),
        ] {
            let d = parse_formula(formula)
                .unwrap_or_else(|e| panic!("{name} ({formula}) must parse: {e}"));
            assert_relative_eq!(delta_mass(d), declared, epsilon = 1e-5);
        }
    }

    /// Losses are real — pyroglutamate from Q loses NH₃ — so the formula must accept them.
    #[test]
    fn formulas_may_remove_atoms() {
        let d = parse_formula("H-3N-1").unwrap();
        assert_eq!((d.h, d.n), (-3, -1));
        assert_relative_eq!(delta_mass(d), -17.026549, epsilon = 1e-5);
    }

    /// An element we do not model must be REFUSED, not ignored. Silently dropping an element leaves
    /// the mass delta correct (it is declared separately) and the envelope wrong — an error that
    /// survives every summary statistic.
    #[test]
    fn unmodelled_elements_are_refused_not_dropped() {
        // Selenium is real (selenocysteine) and not modelled.
        let e = parse_formula("Se").unwrap_err();
        assert!(e.contains("not modelled"), "{e}");
        assert!(parse_formula("").is_err(), "an empty formula is not a modification");
        assert!(parse_formula("C2x").is_err(), "junk must not parse");
    }

    /// Phosphorus is mononuclidic, so phospho shifts the mass and leaves the comb's SHAPE alone.
    /// This is the fact that makes a phosphopeptide's envelope look like its unmodified self's, and
    /// it is worth pinning: an isotope table that gave P a spurious heavy isotope would broaden
    /// every phosphopeptide by a hair, which no summary statistic would ever show.
    #[test]
    fn phospho_shifts_the_mass_but_not_the_envelope_shape() {
        let bare = composition("PEPTIDESK").unwrap();
        let phos = bare.apply(parse_formula("HO3P").unwrap()).unwrap();
        assert_eq!(phos.p, 1);
        let e_bare = envelope_of(bare, 8);
        let e_phos = envelope_of(phos, 8);
        // HO3 does perturb it slightly (oxygen has ¹⁸O), but only slightly — and P adds nothing.
        for k in 0..4 {
            assert!(
                (e_bare[k] - e_phos[k]).abs() < 0.01,
                "peak {k}: {:.4} vs {:.4} — phospho must not reshape the comb",
                e_bare[k],
                e_phos[k]
            );
        }
        assert_relative_eq!(delta_mass(parse_formula("HO3P").unwrap()), 79.966331, epsilon = 1e-5);
    }

    /// A modification really does reshape the comb, not merely shift it. Four glycine-ish atoms of
    /// GG add carbons, and more carbons means relatively less monoisotopic peak.
    #[test]
    fn a_modification_reshapes_the_envelope_not_just_the_mass() {
        let bare = composition("PEPTIDEK").unwrap();
        let gg = bare.apply(parse_formula("C4H6N2O2").unwrap()).unwrap();
        let e_bare = envelope_of(bare, 8);
        let e_gg = envelope_of(gg, 8);
        assert!(
            e_gg[0] < e_bare[0],
            "adding 4 carbons must LOWER the monoisotopic fraction: {:.4} vs {:.4}",
            e_gg[0],
            e_bare[0]
        );
        // And it is not a rounding wobble — the shift is real.
        assert!(e_bare[0] - e_gg[0] > 0.01);
    }

    /// A delta that removes more atoms than the peptide has must fail, not wrap. `u32` underflow
    /// here would yield a four-billion-atom peptide and a nonsensical envelope.
    #[test]
    fn a_delta_that_underflows_the_composition_is_refused() {
        let glycine = composition("G").unwrap();
        let err = glycine.apply(parse_formula("S-5").unwrap()).unwrap_err();
        assert!(err.contains('S'), "{err}");
        // The sane case still works.
        assert!(glycine.apply(parse_formula("O").unwrap()).is_ok());
    }
}
