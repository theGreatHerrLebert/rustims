//! Phase 1.5 recall/param spike for the sparse LSH spectral-match index.
//!
//! MECHANISM test on synthetic data with EXACT ground truth (not a realism
//! test — TimSim S1 / real-.d S2 are the realism confirmation). We generate
//! random "peptide" fragment bags, embed each into a chimeric data "unit"
//! (its own fragments + `mixture_load` other peptides' fragments + mass
//! jitter), build a minimal banded LSH index over the units, and query with
//! each peptide's clean bag. We measure, per (b, r) / mixture load / verify
//! metric:
//!   - recall@candidate  — did the bucket lookup surface the right unit at all?
//!   - recall@1          — is the right unit the top-ranked candidate?
//!   - candidates/query  — verification load (→ rough speedup vs brute force)
//! and compare cosine vs asymmetric containment as chimericity rises.
//!
//! Run:  cargo run --release --example lsh_recall_spike -p mscore
//!
//! This deliberately builds a throwaway in-memory index (the real one is
//! Phase 2 in rustdf); the point is the go/no-go signal before we build it.

use std::collections::{HashMap, HashSet};

use mscore::algorithm::lsh::minhash::MinHash;
use mscore::algorithm::lsh::simhash::{CosineSimHash, Projection};
use mscore::algorithm::lsh::LshScheme;
use mscore::timstof::lsh::{IntensityTransform, MzFeatureMap};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---- knobs (modest so it runs in a few seconds in --release) ----
const N_PEPTIDES: usize = 250;
/// Contaminants are drawn from a DISJOINT background pool so a query peptide
/// appears in exactly one unit — otherwise recall@1 of a specific unit is
/// ill-posed (a peptide reused as a contaminant matches many units).
const N_BACKGROUND: usize = 500;
const FRAGS_PER_PEPTIDE: usize = 10;
const JITTER_PPM: f64 = 3.0; // per-peak mass jitter in the data units
const MIXTURE_LOADS: [usize; 3] = [0, 3, 8]; // # of contaminating peptides / unit
const SEED: u64 = 20260703;

/// A clean fragment bag: (m/z, intensity).
type Bag = Vec<(f64, f64)>;

/// Standard normal via Box-Muller (rand_distr is not a dependency).
fn randn(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn gen_peptides(rng: &mut StdRng, n: usize, nfrag: usize) -> Vec<Bag> {
    (0..n)
        .map(|_| {
            (0..nfrag)
                .map(|_| {
                    let mz = 200.0 + rng.gen::<f64>() * 1600.0; // 200..1800
                    let intensity = -rng.gen::<f64>().max(1e-12).ln(); // ~Exp(1)
                    (mz, intensity)
                })
                .collect()
        })
        .collect()
}

/// Build a chimeric data unit around `primary`: its fragments plus those of
/// `contaminants`, every peak mass-jittered by ~`JITTER_PPM`.
fn build_unit(rng: &mut StdRng, primary: &Bag, contaminants: &[&Bag]) -> Bag {
    let mut out = Vec::new();
    let mut push_jittered = |rng: &mut StdRng, bag: &Bag| {
        for &(mz, i) in bag {
            out.push((mz * (1.0 + randn(rng) * JITTER_PPM * 1e-6), i));
        }
    };
    push_jittered(rng, primary);
    for c in contaminants {
        push_jittered(rng, c);
    }
    out
}

/// Build and featurize all units for a mixture load: unit `i` = primary
/// peptide `i` + `mix` contaminants drawn from the disjoint `background` pool.
fn build_units(
    seed: u64,
    primaries: &[Bag],
    background: &[Bag],
    mix: usize,
    map: &MzFeatureMap,
    transform: IntensityTransform,
) -> Vec<Vec<(i64, f32)>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..primaries.len())
        .map(|i| {
            let contaminants: Vec<&Bag> =
                (0..mix).map(|_| &background[rng.gen_range(0..background.len())]).collect();
            map.features(&build_unit(&mut rng, &primaries[i], &contaminants), transform)
        })
        .collect()
}

/// Overlap (= cosine, both L2-normalized) and Σ_{i∈q} d_i² over shared
/// features, in one merge walk of two feature-id-sorted sparse vectors.
fn match_stats(q: &[(i64, f32)], d: &[(i64, f32)]) -> (f64, f64) {
    let (mut i, mut j) = (0usize, 0usize);
    let (mut overlap, mut d_norm_sq) = (0.0f64, 0.0f64);
    while i < q.len() && j < d.len() {
        match q[i].0.cmp(&d[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                overlap += q[i].1 as f64 * d[j].1 as f64;
                d_norm_sq += (d[j].1 as f64).powi(2);
                i += 1;
                j += 1;
            }
        }
    }
    (overlap, d_norm_sq)
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // Cosine kept for reference; the sweep verifies with Containment
enum Metric {
    Cosine,
    Containment,
}

impl Metric {
    fn score(&self, q: &[(i64, f32)], d: &[(i64, f32)]) -> f64 {
        let (overlap, d_norm_sq) = match_stats(q, d);
        match self {
            // Both L2-normalized → dot over shared features is the cosine.
            Metric::Cosine => overlap,
            // Asymmetric: divide out the contaminant mass in d (renormalize d
            // over q's support), so contamination does not penalize a present
            // query. Invariant to whether d was pre-normalized.
            Metric::Containment => {
                if d_norm_sq > 0.0 {
                    overlap / d_norm_sq.sqrt()
                } else {
                    0.0
                }
            }
        }
    }
}

/// Minimal banded LSH index: one hash table per band. Scheme-agnostic — takes
/// any `LshScheme` (cosine SimHash or MinHash) via its `Vec<u64>` signature.
struct BandIndex {
    tables: Vec<HashMap<u64, Vec<u32>>>,
}

impl BandIndex {
    fn build(scheme: &dyn LshScheme, units: &[Vec<(i64, f32)>]) -> Self {
        let mut tables = vec![HashMap::new(); scheme.num_bands()];
        for (u, feats) in units.iter().enumerate() {
            for (band, key) in scheme.signature(feats).into_iter().enumerate() {
                tables[band].entry(key).or_insert_with(Vec::new).push(u as u32);
            }
        }
        BandIndex { tables }
    }

    /// Union of units colliding in ≥1 band with `sig`.
    fn candidates(&self, sig: &[u64]) -> HashSet<u32> {
        let mut set = HashSet::new();
        for (band, key) in sig.iter().enumerate() {
            if let Some(v) = self.tables[band].get(key) {
                set.extend(v.iter().copied());
            }
        }
        set
    }
}

/// Containment-native baseline: an inverted feature index (Sage-style). A unit
/// is a candidate if it shares ≥ `min_shared` feature bins with the query.
/// Robust to dilution — absolute overlap, not normalized similarity.
struct InvertedIndex {
    postings: HashMap<i64, Vec<u32>>,
}

impl InvertedIndex {
    fn build(units: &[Vec<(i64, f32)>]) -> Self {
        let mut postings: HashMap<i64, Vec<u32>> = HashMap::new();
        for (u, feats) in units.iter().enumerate() {
            for &(id, _) in feats {
                postings.entry(id).or_default().push(u as u32);
            }
        }
        InvertedIndex { postings }
    }

    fn candidates(&self, query: &[(i64, f32)], min_shared: usize) -> HashSet<u32> {
        let mut counts: HashMap<u32, usize> = HashMap::new();
        for &(id, _) in query {
            if let Some(v) = self.postings.get(&id) {
                for &u in v {
                    *counts.entry(u).or_insert(0) += 1;
                }
            }
        }
        counts.into_iter().filter(|&(_, c)| c >= min_shared).map(|(u, _)| u).collect()
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);
    let map = MzFeatureMap::new(2.0, 10.0).unwrap();
    let transform = IntensityTransform::Sqrt;

    // Clean peptide bags: these are the queries; unit `i` is built around peptide `i`.
    let peptides = gen_peptides(&mut rng, N_PEPTIDES, FRAGS_PER_PEPTIDE);
    // Disjoint background pool used only as contaminants.
    let background = gen_peptides(&mut rng, N_BACKGROUND, FRAGS_PER_PEPTIDE);
    let queries: Vec<Vec<(i64, f32)>> =
        peptides.iter().map(|p| map.features(p, transform)).collect();

    println!(
        "Phase 1.5 synthetic recall spike — {} peptides, {} frags each, jitter {:.0} ppm\n\
         comparing CANDIDATE GENERATORS (verify = containment; brute-force rec@1 = 1.0 always)\n\
         (mechanism test, exact labels; realism = TimSim S1 / real .d S2)\n",
        N_PEPTIDES, FRAGS_PER_PEPTIDE, JITTER_PPM
    );
    println!(
        "{:>4}  {:<18} {:>10} {:>9} {:>10}",
        "mix", "generator", "rec@cand", "rec@1", "cand/q"
    );
    println!("{}", "-".repeat(58));

    // Evaluate one candidate generator: recall@candidate, recall@1 (ranked by
    // containment), and mean candidates/query.
    let eval = |cand_fn: &dyn Fn(usize) -> HashSet<u32>, units: &[Vec<(i64, f32)>]| {
        let (mut hc, mut h1, mut ct) = (0usize, 0usize, 0usize);
        for i in 0..N_PEPTIDES {
            let cands = cand_fn(i);
            ct += cands.len();
            if cands.contains(&(i as u32)) {
                hc += 1;
            }
            let best = cands
                .iter()
                .map(|&u| (u, Metric::Containment.score(&queries[i], &units[u as usize])))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            if best.map(|(u, _)| u) == Some(i as u32) {
                h1 += 1;
            }
        }
        let n = N_PEPTIDES as f64;
        (hc as f64 / n, h1 as f64 / n, ct as f64 / n)
    };

    let row = |mix: usize, name: String, r: (f64, f64, f64)| {
        println!("{:>4}  {:<18} {:>10.3} {:>9.3} {:>10.1}", mix, name, r.0, r.1, r.2);
    };

    for &mix in &MIXTURE_LOADS {
        let units = build_units(
            SEED ^ (mix as u64).wrapping_mul(0x9E3779B9),
            &peptides,
            &background,
            mix,
            &map,
            transform,
        );

        // Cosine SimHash (variant #1).
        for &(b, r) in &[(32usize, 12usize), (48, 8)] {
            let h = CosineSimHash::new(0xC0FFEE, b, r, Projection::Gaussian).unwrap();
            let idx = BandIndex::build(&h, &units);
            let f = |i: usize| idx.candidates(&h.signature(&queries[i]));
            row(mix, format!("simhash {b}x{r}"), eval(&f, &units));
        }
        // MinHash (variant #2) — Jaccard set LSH.
        for &(b, r) in &[(32usize, 4usize), (48, 2)] {
            let h = MinHash::new(0xC0FFEE, b, r).unwrap();
            let idx = BandIndex::build(&h, &units);
            let f = |i: usize| idx.candidates(&h.signature(&queries[i]));
            row(mix, format!("minhash {b}x{r}"), eval(&f, &units));
        }
        // Inverted feature index (containment-native baseline).
        let inv = InvertedIndex::build(&units);
        for &m in &[8usize, 25] {
            let f = |i: usize| inv.candidates(&queries[i], m);
            row(mix, format!("invidx >={m}"), eval(&f, &units));
        }
        println!();
    }
}
