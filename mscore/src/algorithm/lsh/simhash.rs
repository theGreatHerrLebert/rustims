//! Cosine SimHash — the first [`LshScheme`](super::LshScheme) family.
//!
//! Signed random-projection LSH for cosine similarity. Each of the `L = b·r`
//! projections hashes every active feature id to a random weight, accumulates
//! `Σ value · weight`, and takes the sign as one bit. Bits are packed into `b`
//! bands of `r` bits. Two vectors at cosine `c` agree on any single bit with
//! probability `1 − θ/π` (`θ = arccos c`) — the property the tests verify.
//!
//! Projection weights are generated on the fly from a `splitmix64` hash of
//! `(seed, feature_id, k)` — feature hashing, so there is no stored projection
//! matrix and no vocabulary bound. The default is Gaussian weights (isotropic
//! hyperplanes, for which the `1 − θ/π` law is exact); Rademacher ±1 is a
//! cheaper opt-in that only approaches the law via the CLT.

use std::f64::consts::PI;

use statrs::distribution::{ContinuousCDF, Normal};

use super::{feature_base, projection_hash, sparse_cosine, LshScheme};

/// Random-projection weight distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    /// `N(0,1)` weights — isotropic hyperplanes; the `1 − θ/π` law is exact.
    Gaussian,
    /// `±1` weights — cheaper, only asymptotically satisfies the law.
    Rademacher,
}

/// Cosine SimHash hasher.
#[derive(Clone, Debug)]
pub struct CosineSimHash {
    seed: u64,
    bands: usize,
    band_bits: usize,
    projection: Projection,
}

impl CosineSimHash {
    /// Create a hasher with `bands` bands (`b`) of `band_bits` bits (`r`).
    ///
    /// Returns `Err` if `bands == 0`, or `band_bits` is not in `1..=64`
    /// (a band key must fit in a `u64`).
    pub fn new(
        seed: u64,
        bands: usize,
        band_bits: usize,
        projection: Projection,
    ) -> Result<Self, String> {
        if bands == 0 {
            return Err("bands must be > 0".to_string());
        }
        if band_bits == 0 || band_bits > 64 {
            return Err(format!("band_bits must be in 1..=64, got {band_bits}"));
        }
        Ok(Self { seed, bands, band_bits, projection })
    }

    /// Total number of bits in the signature, `L = b·r`.
    #[inline]
    pub fn num_bits(&self) -> usize {
        self.bands * self.band_bits
    }

    /// Raw per-projection sign bits (length `L`). `signature` packs these into
    /// bands; tests use them to measure the per-bit collision law directly.
    ///
    /// Zero-dot tie handling: an accumulator of exactly `0.0` maps to `true`
    /// (`sign(0) := +1`), deterministically — `acc >= 0.0` includes `-0.0`.
    pub fn bits(&self, features: &[(i64, f32)]) -> Vec<bool> {
        let l = self.num_bits();
        let mut acc = vec![0.0f64; l];
        // Standard normal, constructed once; used only for Gaussian weights.
        let normal = Normal::new(0.0, 1.0).expect("standard normal is valid");
        for &(fid, val) in features {
            if val == 0.0 {
                continue;
            }
            let v = val as f64;
            let base = feature_base(self.seed, fid);
            for (k, a) in acc.iter_mut().enumerate() {
                let h = projection_hash(base, k);
                let w = match self.projection {
                    Projection::Rademacher => {
                        if h >> 63 == 0 {
                            -1.0
                        } else {
                            1.0
                        }
                    }
                    Projection::Gaussian => {
                        // 53-bit uniform in the open interval (0,1), then probit.
                        let u = ((h >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64);
                        normal.inverse_cdf(u)
                    }
                };
                *a += v * w;
            }
        }
        acc.iter().map(|&a| a >= 0.0).collect()
    }
}

impl LshScheme for CosineSimHash {
    fn num_bands(&self) -> usize {
        self.bands
    }

    fn band_bits(&self) -> usize {
        self.band_bits
    }

    fn signature(&self, features: &[(i64, f32)]) -> Vec<u64> {
        let bits = self.bits(features);
        let mut out = vec![0u64; self.bands];
        for (j, key) in out.iter_mut().enumerate() {
            let mut k = 0u64;
            for bit in 0..self.band_bits {
                if bits[j * self.band_bits + bit] {
                    k |= 1u64 << bit;
                }
            }
            *key = k;
        }
        out
    }

    fn verify(&self, a: &[(i64, f32)], b: &[(i64, f32)]) -> f32 {
        sparse_cosine(a, b)
    }

    fn collision_law(&self, similarity: f32) -> f32 {
        let c = (similarity as f64).clamp(-1.0, 1.0);
        (1.0 - c.acos() / PI) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn hasher(bands: usize, band_bits: usize, proj: Projection) -> CosineSimHash {
        CosineSimHash::new(0xC0FFEE, bands, band_bits, proj).unwrap()
    }

    /// Random vector over feature ids `0..dim`, components uniform in [-1,1].
    fn rand_vec(rng: &mut StdRng, dim: i64) -> Vec<(i64, f32)> {
        (0..dim).map(|i| (i, (rng.gen::<f32>() * 2.0 - 1.0))).collect()
    }

    /// `b = t·a + sqrt(1-t²)·u` has cosine ≈ `t` with `a` for large dim
    /// (a, u independent, mean-zero components).
    fn correlated_vec(a: &[(i64, f32)], u: &[(i64, f32)], t: f32) -> Vec<(i64, f32)> {
        let s = (1.0 - t * t).sqrt();
        a.iter()
            .zip(u.iter())
            .map(|(&(id, av), &(_, uv))| (id, t * av + s * uv))
            .collect()
    }

    fn bit_agreement(x: &[bool], y: &[bool]) -> f64 {
        let same = x.iter().zip(y).filter(|(a, b)| a == b).count();
        same as f64 / x.len() as f64
    }

    #[test]
    fn construction_validates_params() {
        assert!(CosineSimHash::new(1, 0, 8, Projection::Gaussian).is_err());
        assert!(CosineSimHash::new(1, 4, 0, Projection::Gaussian).is_err());
        assert!(CosineSimHash::new(1, 4, 65, Projection::Gaussian).is_err());
        let h = CosineSimHash::new(1, 4, 20, Projection::Gaussian).unwrap();
        assert_eq!(h.num_bands(), 4);
        assert_eq!(h.band_bits(), 20);
        assert_eq!(h.num_bits(), 80);
    }

    #[test]
    fn signature_is_deterministic() {
        let h = hasher(32, 16, Projection::Gaussian);
        let mut rng = StdRng::seed_from_u64(1);
        let v = rand_vec(&mut rng, 100);
        assert_eq!(h.signature(&v), h.signature(&v));
    }

    #[test]
    fn identical_vectors_agree_on_every_bit() {
        let h = hasher(64, 8, Projection::Gaussian);
        let mut rng = StdRng::seed_from_u64(2);
        let v = rand_vec(&mut rng, 100);
        assert_eq!(bit_agreement(&h.bits(&v), &h.bits(&v)), 1.0);
        assert_eq!(h.signature(&v), h.signature(&v));
    }

    #[test]
    fn empty_vector_ties_to_plus_one() {
        // Zero accumulators → all bits set (sign(0) := +1) → each band key is
        // all-ones over its r bits.
        let h = hasher(4, 8, Projection::Gaussian);
        let bits = h.bits(&[]);
        assert!(bits.iter().all(|&b| b));
        let full = (1u64 << 8) - 1;
        assert_eq!(h.signature(&[]), vec![full; 4]);
    }

    /// The correctness gate: measured per-bit collision rate tracks `1 − θ/π`.
    #[test]
    fn per_bit_collision_law_gaussian() {
        const DIM: i64 = 128;
        const NBITS: usize = 512;
        const PAIRS: usize = 15;
        // One bit per band → bits() yields NBITS independent projections.
        let h = hasher(NBITS, 1, Projection::Gaussian);
        let mut rng = StdRng::seed_from_u64(42);

        for &t in &[0.0f32, 0.3, 0.6, 0.9] {
            let mut sum_measured = 0.0;
            let mut sum_predicted = 0.0;
            for _ in 0..PAIRS {
                let a = rand_vec(&mut rng, DIM);
                let u = rand_vec(&mut rng, DIM);
                let b = correlated_vec(&a, &u, t);
                let cos = h.verify(&a, &b);
                sum_predicted += h.collision_law(cos) as f64;
                sum_measured += bit_agreement(&h.bits(&a), &h.bits(&b));
            }
            let measured = sum_measured / PAIRS as f64;
            let predicted = sum_predicted / PAIRS as f64;
            assert!(
                (measured - predicted).abs() < 0.03,
                "t={t}: measured per-bit agreement {measured:.4} vs law {predicted:.4}"
            );
        }
    }

    /// Rademacher approaches the same law (looser tolerance, no exactness claim).
    #[test]
    fn per_bit_collision_law_rademacher() {
        const DIM: i64 = 128;
        const NBITS: usize = 512;
        const PAIRS: usize = 15;
        let h = hasher(NBITS, 1, Projection::Rademacher);
        let mut rng = StdRng::seed_from_u64(7);

        for &t in &[0.0f32, 0.5, 0.9] {
            let mut sum_measured = 0.0;
            let mut sum_predicted = 0.0;
            for _ in 0..PAIRS {
                let a = rand_vec(&mut rng, DIM);
                let u = rand_vec(&mut rng, DIM);
                let b = correlated_vec(&a, &u, t);
                sum_predicted += h.collision_law(h.verify(&a, &b)) as f64;
                sum_measured += bit_agreement(&h.bits(&a), &h.bits(&b));
            }
            let measured = sum_measured / PAIRS as f64;
            let predicted = sum_predicted / PAIRS as f64;
            assert!(
                (measured - predicted).abs() < 0.05,
                "t={t}: rademacher agreement {measured:.4} vs law {predicted:.4}"
            );
        }
    }

    #[test]
    fn verify_matches_reference_cosine() {
        let h = hasher(8, 8, Projection::Gaussian);
        // Proportional vectors → cosine 1.
        assert!((h.verify(&[(1, 1.0), (2, 2.0)], &[(1, 2.0), (2, 4.0)]) - 1.0).abs() < 1e-6);
        // Disjoint features → cosine 0.
        assert_eq!(h.verify(&[(1, 1.0)], &[(2, 1.0)]), 0.0);
        // Empty → 0.
        assert_eq!(h.verify(&[], &[(1, 1.0)]), 0.0);
    }

    #[test]
    fn collision_law_endpoints_and_monotone() {
        let h = hasher(1, 1, Projection::Gaussian);
        assert!((h.collision_law(1.0) - 1.0).abs() < 1e-6);
        assert!((h.collision_law(0.0) - 0.5).abs() < 1e-6);
        assert!((h.collision_law(-1.0) - 0.0).abs() < 1e-6);
        assert!(h.collision_law(0.9) > h.collision_law(0.5));
        assert!(h.collision_law(0.5) > h.collision_law(0.0));
    }
}
