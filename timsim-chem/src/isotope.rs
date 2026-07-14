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
}

impl Composition {
    fn add(&mut self, o: Composition, times: u32) {
        self.c += o.c * times;
        self.h += o.h * times;
        self.n += o.n * times;
        self.o += o.o * times;
        self.s += o.s * times;
    }
}

const fn comp(c: u32, h: u32, n: u32, o: u32, s: u32) -> Composition {
    Composition { c, h, n, o, s }
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

/// The isotope envelope of a peptide: relative intensities of the monoisotopic peak and its
/// `depth - 1` heavier neighbours, normalised to sum to 1.
///
/// Exact for the given composition — no averagine guess, because we know the formula.
pub fn envelope(sequence: &str, depth: usize) -> Result<Vec<f32>, mass::UnknownResidue> {
    let depth = depth.max(1);
    let c = composition(sequence)?;

    let mut dist = self_convolve(&AB_C, c.c, depth);
    for (ab, n) in [
        (&AB_H[..], c.h),
        (&AB_N[..], c.n),
        (&AB_O[..], c.o),
        (&AB_S[..], c.s),
    ] {
        if n > 0 {
            dist = convolve(&dist, &self_convolve(ab, n, depth), depth);
        }
    }

    let total: f64 = dist.iter().sum();
    Ok(dist.iter().map(|v| (v / total) as f32).collect())
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
}
