//! Locality-sensitive hashing (LSH) for sparse spectra.
//!
//! This module provides a small, pluggable LSH abstraction. A concrete LSH
//! *family* implements [`LshScheme`]: it turns a sparse feature vector into a
//! set of band keys whose collision probability is monotone in the family's
//! target similarity. Cosine [`simhash::CosineSimHash`] is the first (and,
//! for now, only) family; a set/containment family (MinHash) can be added
//! later behind the same trait without touching the index, store, or driver.
//!
//! The seam is deliberately thin: every family emits a uniform
//! `Vec<u64>` of band keys, so a banded index stays a concrete
//! `Vec<HashMap<u64, _>>` — only the hashing is polymorphic.
//!
//! Feature ids are `i64` end to end (log-ppm bin ids can be negative for
//! m/z < 1). Hashing is a deterministic, dependency-free `splitmix64` mix so
//! signatures are reproducible across machines.

pub mod minhash;
pub mod simhash;

/// A single splitmix64 finalizer step — a fast, well-distributed 64-bit mix.
///
/// Deterministic and platform-independent: the same input always yields the
/// same output, which LSH reproducibility requires.
#[inline]
pub(crate) fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Golden-ratio odd constant used to decorrelate successive projections.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// Mix `seed` with a feature id into a per-feature base hash (computed once
/// per feature, then combined with the projection index).
#[inline]
pub(crate) fn feature_base(seed: u64, feature_id: i64) -> u64 {
    splitmix64(seed ^ (feature_id as u64).wrapping_mul(GOLDEN))
}

/// Derive the hash for projection `k` of a feature from its base hash.
#[inline]
pub(crate) fn projection_hash(base: u64, k: usize) -> u64 {
    splitmix64(base.wrapping_add((k as u64).wrapping_mul(GOLDEN)))
}

/// An LSH family over sparse `(feature_id, value)` vectors.
///
/// Implementors turn a sparse vector into `num_bands()` band keys. Two
/// vectors are candidates iff they collide in ≥1 band; the per-family
/// [`collision_law`](LshScheme::collision_law) describes how that probability
/// relates to the family's similarity metric, so the banding sweep can be run
/// per scheme.
pub trait LshScheme {
    /// Number of bands `b` (independent hash tables).
    fn num_bands(&self) -> usize;

    /// Bits per band `r`.
    fn band_bits(&self) -> usize;

    /// Produce `num_bands()` band keys for a sparse feature vector.
    ///
    /// The output is always `Vec<u64>` so the banded index is never generic
    /// over the scheme. Each key is looked up in that band's own table.
    fn signature(&self, features: &[(i64, f32)]) -> Vec<u64>;

    /// The family's verification similarity between two sparse vectors,
    /// used to confirm a candidate after bucket collision.
    fn verify(&self, a: &[(i64, f32)], b: &[(i64, f32)]) -> f32;

    /// Analytic collision law: probability that a single hash bit/component
    /// agrees between two vectors of the given similarity. For cosine SimHash
    /// this is the per-bit law `1 − θ/π`.
    fn collision_law(&self, similarity: f32) -> f32;
}

/// Cosine similarity between two sparse vectors given as `(feature_id, value)`
/// slices. Order-independent; returns 0.0 if either side is empty or
/// zero-norm.
pub(crate) fn sparse_cosine(a: &[(i64, f32)], b: &[(i64, f32)]) -> f32 {
    use std::collections::HashMap;
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    // Sum duplicate feature ids on the smaller side into a map, then walk the
    // other side. Using f64 accumulators to limit rounding error.
    let (map_src, other) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut map: HashMap<i64, f64> = HashMap::with_capacity(map_src.len());
    let mut norm_src = 0.0f64;
    for &(id, v) in map_src {
        let v = v as f64;
        *map.entry(id).or_insert(0.0) += v;
    }
    for v in map.values() {
        norm_src += v * v;
    }
    let mut dot = 0.0f64;
    let mut norm_other_map: HashMap<i64, f64> = HashMap::with_capacity(other.len());
    for &(id, v) in other {
        let v = v as f64;
        *norm_other_map.entry(id).or_insert(0.0) += v;
    }
    let mut norm_other = 0.0f64;
    for (id, v) in &norm_other_map {
        norm_other += v * v;
        if let Some(sv) = map.get(id) {
            dot += sv * v;
        }
    }
    let denom = (norm_src * norm_other).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (dot / denom) as f32
    }
}
