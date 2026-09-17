//! Deterministic RNG streams for the simulation's noise, keyed by identity rather than by
//! scheduling order.
//!
//! The m/z jitter and the reference-noise overlay used `rand::thread_rng`, so which thread picked
//! up a spectrum decided its noise. That makes a run irreproducible, and it is not a cosmetic
//! problem: a jittered m/z can cross a TOF bin boundary, which changes the `(scan, tof)` dedup and
//! so the peaks written to the file.
//!
//! Deriving the stream from a master seed plus the stable identifiers already in scope — frame id,
//! peptide id, ion index, scan — makes the draw independent of thread count and scheduling.

use rand::rngs::StdRng;
use rand::SeedableRng;

/// Golden-ratio odd constant, the usual choice for cheap integer mixing.
const MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Mix a master seed and a list of identifiers into one 64-bit stream key.
///
/// Order matters: `[frame, peptide]` and `[peptide, frame]` give different streams, which is what
/// you want, since the components are not interchangeable.
#[inline]
pub fn stream_key(seed: u64, keys: &[u64]) -> u64 {
    let mut h = seed ^ MIX;
    for (i, &k) in keys.iter().enumerate() {
        // SplitMix64-style finaliser per component, position-dependent so permutations differ.
        let mut x = k.wrapping_add(MIX.wrapping_mul(i as u64 + 1));
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        h = (h ^ x).wrapping_mul(MIX);
    }
    h
}

/// A reproducible RNG for one unit of noise, keyed by `seed` and `keys`.
#[inline]
pub fn noise_rng(seed: u64, keys: &[u64]) -> StdRng {
    StdRng::seed_from_u64(stream_key(seed, keys))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn draw(seed: u64, keys: &[u64]) -> u64 {
        noise_rng(seed, keys).gen()
    }

    #[test]
    fn same_inputs_give_the_same_stream() {
        assert_eq!(draw(41, &[7, 3, 1]), draw(41, &[7, 3, 1]));
    }

    #[test]
    fn the_seed_matters() {
        assert_ne!(draw(41, &[7, 3, 1]), draw(42, &[7, 3, 1]));
    }

    #[test]
    fn each_key_component_matters() {
        let base = draw(41, &[7, 3, 1]);
        assert_ne!(base, draw(41, &[8, 3, 1]));
        assert_ne!(base, draw(41, &[7, 4, 1]));
        assert_ne!(base, draw(41, &[7, 3, 2]));
    }

    #[test]
    fn key_order_matters() {
        assert_ne!(draw(41, &[1, 2]), draw(41, &[2, 1]));
    }

    #[test]
    fn neighbouring_keys_do_not_collide() {
        // Frame/scan ids are dense and small; adjacent ones must not share a stream.
        let mut seen = std::collections::HashSet::new();
        for frame in 0..200u64 {
            for scan in 0..50u64 {
                assert!(seen.insert(stream_key(41, &[frame, scan])), "collision at {frame}/{scan}");
            }
        }
    }
}
