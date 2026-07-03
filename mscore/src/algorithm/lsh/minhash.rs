//! MinHash — the second [`LshScheme`](super::LshScheme) family (set/Jaccard).
//!
//! Where [`super::simhash::CosineSimHash`] hashes a *weighted* vector for
//! cosine, MinHash hashes the *set* of feature ids for Jaccard similarity. Each
//! of `k = b·r` hash functions maps every feature id to a hash value; the
//! signature keeps the per-function minimum. Two sets agree on any single
//! min-hash with probability equal to their Jaccard similarity — the property
//! the tests verify. `r` min-hashes are packed per band, `b` bands, so the
//! output is the same uniform `Vec<u64>` of band keys the index expects.
//!
//! Intensities are ignored (pure set semantics). This is the §4.1.2(C) family:
//! candidate generation for containment/near-duplicate spectra where the peak
//! *set* matters more than its weighting. NOTE: MinHash estimates Jaccard, not
//! containment — a small set inside a much larger one has low Jaccard, so plain
//! MinHash is still sensitive to heavy dilution.

use super::{feature_base, projection_hash, splitmix64, LshScheme};

/// MinHash hasher: `bands` bands of `rows` min-hashes each.
#[derive(Clone, Debug)]
pub struct MinHash {
    seed: u64,
    bands: usize,
    rows: usize,
}

impl MinHash {
    /// Create a hasher with `bands` bands (`b`) of `rows` min-hashes (`r`);
    /// `k = bands·rows` hash functions total. Both must be `> 0`.
    pub fn new(seed: u64, bands: usize, rows: usize) -> Result<Self, String> {
        if bands == 0 || rows == 0 {
            return Err(format!("bands and rows must be > 0, got {bands}, {rows}"));
        }
        Ok(Self { seed, bands, rows })
    }

    /// Total number of hash functions, `k = b·r`.
    #[inline]
    pub fn num_hashes(&self) -> usize {
        self.bands * self.rows
    }

    /// Raw per-function minima (length `k`). An empty set yields all
    /// `u64::MAX` (a deterministic, shared signature). Tests use this to
    /// measure the per-min-hash collision law directly.
    pub fn min_hashes(&self, features: &[(i64, f32)]) -> Vec<u64> {
        let k = self.num_hashes();
        let mut mins = vec![u64::MAX; k];
        for &(fid, _val) in features {
            let base = feature_base(self.seed, fid);
            for (slot, m) in mins.iter_mut().enumerate() {
                let h = projection_hash(base, slot);
                if h < *m {
                    *m = h;
                }
            }
        }
        mins
    }
}

impl LshScheme for MinHash {
    fn num_bands(&self) -> usize {
        self.bands
    }

    fn band_bits(&self) -> usize {
        self.rows // "rows per band" for MinHash, not bits
    }

    fn signature(&self, features: &[(i64, f32)]) -> Vec<u64> {
        let mins = self.min_hashes(features);
        let mut out = vec![0u64; self.bands];
        for (j, key) in out.iter_mut().enumerate() {
            // Fold the band's r min-hashes into one key.
            let mut acc = 0xcbf2_9ce4_8422_2325u64; // FNV offset basis
            for row in 0..self.rows {
                acc = splitmix64(acc ^ mins[j * self.rows + row]);
            }
            *key = acc;
        }
        out
    }

    fn verify(&self, a: &[(i64, f32)], b: &[(i64, f32)]) -> f32 {
        jaccard(a, b)
    }

    fn collision_law(&self, similarity: f32) -> f32 {
        // Per-min-hash collision probability equals the Jaccard similarity.
        similarity.clamp(0.0, 1.0)
    }
}

/// Jaccard similarity of two feature-id sets. Order- and duplicate-independent
/// (built from sets), matching the set semantics of `signature`/`min_hashes` —
/// so `verify` never disagrees with the hasher on unsorted or repeated ids.
fn jaccard(a: &[(i64, f32)], b: &[(i64, f32)]) -> f32 {
    use std::collections::HashSet;
    let sa: HashSet<i64> = a.iter().map(|&(id, _)| id).collect();
    let sb: HashSet<i64> = b.iter().map(|&(id, _)| id).collect();
    if sa.is_empty() && sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count();
    let uni = sa.len() + sb.len() - inter;
    inter as f32 / uni as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(ids: &[i64]) -> Vec<(i64, f32)> {
        ids.iter().map(|&id| (id, 1.0)).collect()
    }

    fn agreement(a: &[u64], b: &[u64]) -> f64 {
        a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len() as f64
    }

    #[test]
    fn construction_validates() {
        assert!(MinHash::new(1, 0, 4).is_err());
        assert!(MinHash::new(1, 4, 0).is_err());
        let h = MinHash::new(1, 8, 4).unwrap();
        assert_eq!(h.num_bands(), 8);
        assert_eq!(h.num_hashes(), 32);
    }

    #[test]
    fn identical_sets_agree_everywhere() {
        let h = MinHash::new(9, 16, 4).unwrap();
        let s = set(&[3, 1, 4, 1, 5, 9, 2, 6]);
        assert_eq!(agreement(&h.min_hashes(&s), &h.min_hashes(&s)), 1.0);
        assert_eq!(h.signature(&s), h.signature(&s));
        assert!((h.verify(&set(&[1, 2, 3]), &set(&[1, 2, 3])) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn disjoint_sets_have_zero_jaccard() {
        let h = MinHash::new(1, 4, 2).unwrap();
        assert_eq!(h.verify(&set(&[1, 2, 3]), &set(&[4, 5, 6])), 0.0);
    }

    #[test]
    fn verify_is_order_and_duplicate_independent() {
        // Set semantics: unsorted + repeated ids must not change Jaccard, so
        // verify agrees with the (order-independent) signature.
        let h = MinHash::new(1, 4, 2).unwrap();
        let a = vec![(2i64, 1.0f32), (1, 1.0), (2, 1.0)]; // unsorted, dup
        let b = vec![(1i64, 1.0f32), (2, 1.0)];
        assert!((h.verify(&a, &b) - 1.0).abs() < 1e-6);
    }

    /// Correctness gate: measured per-min-hash agreement tracks Jaccard.
    #[test]
    fn per_minhash_collision_tracks_jaccard() {
        let h = MinHash::new(123, 4096, 1).unwrap(); // 4096 independent hashes
        // A = {0..99}, B = {50..149}: |∩|=50, |∪|=150, J = 1/3.
        let a: Vec<i64> = (0..100).collect();
        let b: Vec<i64> = (50..150).collect();
        let measured = agreement(&h.min_hashes(&set(&a)), &h.min_hashes(&set(&b)));
        let jac = h.verify(&set(&a), &set(&b)) as f64;
        assert!((jac - 1.0 / 3.0).abs() < 1e-6, "jaccard {jac}");
        assert!((measured - jac).abs() < 0.03, "measured {measured} vs jaccard {jac}");
    }

    #[test]
    fn containment_is_not_jaccard_under_dilution() {
        // A small set fully inside a large one has containment 1 but low
        // Jaccard — the reason plain MinHash still struggles with dilution.
        let h = MinHash::new(1, 4, 2).unwrap();
        let small: Vec<i64> = (0..10).collect();
        let large: Vec<i64> = (0..100).collect(); // small ⊆ large
        let j = h.verify(&set(&small), &set(&large));
        assert!((j - 0.1).abs() < 1e-6, "jaccard under 10x dilution = {j}");
    }
}
