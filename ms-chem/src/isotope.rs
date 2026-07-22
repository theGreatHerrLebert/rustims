//! Isotope envelopes — the relative intensities of a peptide's monoisotopic peak and its heavier
//! neighbours.
//!
//! Exact for a known composition (no averagine): convolve each element's natural-abundance comb,
//! keeping `depth` peaks. The algorithm is timsim-chem's (binary-exponentiation self-convolution,
//! proven in R1 Gates 2/5); the abundances are the **canonical CIAAW values** (`CHEM_PARITY.md` —
//! N and S differed between the source impls; CIAAW won).
//!
//! Selenium is deliberately unsupported here: its most-abundant isotope (⁸⁰Se, the mass base — see
//! `elements::SE`) is *not* its lightest, so its comb has peaks *below* the base, which this
//! nominal-neutron-count convolution cannot represent. Se-containing envelopes (selenocysteine) are
//! deferred until that case is handled, rather than silently producing a wrong shape.

use crate::mass::PROTON;
use crate::residue::{peptide_composition, residue_composition, Composition};

// Natural abundances (CIAAW), indexed by *extra neutrons* (0 = the lightest = most abundant, for
// every element here — the property Se lacks). C/H/O match both source impls; N/S are the CIAAW
// values ms-chem adopted.
const AB_C: [f64; 2] = [0.9893, 0.0107];
const AB_H: [f64; 2] = [0.999_885, 0.000_115];
const AB_N: [f64; 2] = [0.996_36, 0.003_64];
const AB_O: [f64; 3] = [0.997_57, 0.000_38, 0.002_05];
const AB_S: [f64; 5] = [0.9499, 0.0075, 0.0425, 0.0, 0.0001];
const AB_P: [f64; 1] = [1.0];

/// Mass difference per additional neutron (Da). The ¹³C−¹²C spacing dominates a peptide's comb.
pub const NEUTRON: f64 = 1.003_355;

/// Why an isotope envelope could not be produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeError {
    /// A residue byte not in the residue table.
    UnknownResidue(char),
    /// Selenium (selenocysteine) — its ⁸⁰Se base has abundant lighter isotopes; see module docs.
    SeleniumUnsupported,
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnvelopeError::UnknownResidue(c) => write!(f, "unknown residue {c:?}"),
            EnvelopeError::SeleniumUnsupported => {
                write!(f, "selenium isotope envelope not supported (see ms_chem::isotope docs)")
            }
        }
    }
}

impl std::error::Error for EnvelopeError {}

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

/// Convolve `dist` with itself `n` times, keeping `depth` peaks — binary exponentiation, so a
/// 300-carbon peptide costs ~8 convolutions, not 300.
fn self_convolve(dist: &[f64], n: u32, depth: usize) -> Vec<f64> {
    let mut result = vec![0.0; depth];
    result[0] = 1.0;
    if n == 0 {
        return result;
    }
    let mut base = {
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

/// Isotope envelope of a composition: relative intensities of the first `depth` peaks, normalized to
/// sum to 1. Errors on selenium (see module docs).
pub fn envelope_of(c: Composition, depth: usize) -> Result<Vec<f64>, EnvelopeError> {
    if c.se > 0 {
        return Err(EnvelopeError::SeleniumUnsupported);
    }
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
    Ok(dist.iter().map(|v| v / total).collect())
}

/// Isotope envelope of an intact peptide (residues + water).
pub fn envelope(sequence: &str, depth: usize) -> Result<Vec<f64>, EnvelopeError> {
    let c = peptide_composition(sequence).ok_or_else(|| {
        let bad = sequence
            .bytes()
            .find(|&b| residue_composition(b).is_none())
            .unwrap_or(b'?');
        EnvelopeError::UnknownResidue(bad as char)
    })?;
    envelope_of(c, depth)
}

/// m/z of isotope peak `k` for a species of monoisotopic mass `mono` at the given charge.
pub fn peak_mz(mono_mass: f64, charge: u32, k: usize) -> f64 {
    assert!(charge >= 1, "charge must be >= 1");
    (mono_mass + k as f64 * NEUTRON + charge as f64 * PROTON) / charge as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_sums_to_one_and_starts_positive() {
        let e = envelope("PEPTIDER", 6).unwrap();
        let sum: f64 = e.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert!(e[0] > 0.5); // monoisotopic peak dominates a small peptide
    }

    #[test]
    fn selenium_is_deferred_cleanly() {
        assert_eq!(envelope("UGG", 6), Err(EnvelopeError::SeleniumUnsupported));
        assert!(matches!(envelope("PZP", 6), Err(EnvelopeError::UnknownResidue('Z'))));
    }
}
