//! Dataset-level similarity search. Instead of matching individual (chimeric)
//! spectra, summarize each run as ONE compact fingerprint over a binned
//! (m/z x ion-mobility) grid — intensity-weighted, denoised — and compare runs
//! by cosine (quantitative profile) and Jaccard (compositional overlap). This
//! is what MinHash/SimHash sketches are built for: "how close is dataset A to
//! B" in one cheap comparison. Aggregating to a whole-run fingerprint makes
//! chimericity irrelevant (we summarize composition, we don't resolve analytes).
//!
//! MS1 fingerprint ~ "same sample" (scheme-independent precursor landscape);
//! MS2 fingerprint ~ "same sample AND acquisition scheme" (fragment content).
//!
//! Validation: run over the TimSim zoo and check the similarity ordering
//! recovers known relationships (replicates ~1, cross-sample low).
//!
//! Run: cargo run --release --example dataset_similarity -p rustdf -- \
//!        <base_dir> [n_frames=800] [min_int=100] [ms_level=2]

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;

use rayon::prelude::*;
use rustdf::data::dia::TimsDatasetDIA;
use rustdf::data::handle::TimsData;

// Fingerprint bin widths. m/z 0.01 Da tolerates 5 ppm noise (~5 mDa @ 1000);
// mobility 0.01 in 1/K0. Coarse enough to be calibration-robust, fine enough
// to discriminate samples.
const MZ_W: f64 = 0.01;
const MOB_W: f64 = 0.01;

fn token(mz: f64, mob: f64) -> i64 {
    let mzb = (mz / MZ_W).round() as i64;
    let mob_b = (mob / MOB_W).round() as i64;
    mzb * 1_000_000 + mob_b
}

/// One run -> (L2-normalized sparse fingerprint sorted by token, token set).
fn fingerprint(path: &str, n_frames: usize, min_int: f64, ms_level: i32) -> (Vec<(i64, f32)>, HashSet<i64>) {
    let ds = TimsDatasetDIA::new("NO_SDK", path, false, false);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);

    // sampled frame ids of the requested MS level, evenly spread across the run
    let mut ids: Vec<u32> = ds.meta_data.iter()
        .filter(|m| if ms_level == 1 { m.ms_ms_type == 0 } else { m.ms_ms_type != 0 })
        .map(|m| m.id as u32).collect();
    ids.sort_unstable();
    let step = (ids.len() / n_frames.max(1)).max(1);
    let slice: Vec<u32> = ids.iter().step_by(step).copied().collect();

    let frames = ds.get_slice(slice, threads).frames;
    // per-frame partial fingerprints, merged (parallel over frames)
    let acc: HashMap<i64, f64> = frames.par_iter().fold(HashMap::new, |mut m, f| {
        let mz = &f.ims_frame.mz;
        let mob = &f.ims_frame.mobility;
        let inten = &f.ims_frame.intensity;
        for i in 0..inten.len() {
            if inten[i] >= min_int && mz[i] > 0.0 {
                *m.entry(token(mz[i], mob[i])).or_insert(0.0) += inten[i].sqrt();
            }
        }
        m
    }).reduce(HashMap::new, |mut a, b| {
        for (k, v) in b { *a.entry(k).or_insert(0.0) += v; }
        a
    });

    let norm: f64 = acc.values().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    let mut fp: Vec<(i64, f32)> = acc.iter().map(|(&k, &v)| (k, (v / norm) as f32)).collect();
    fp.sort_unstable_by_key(|&(k, _)| k);
    let set: HashSet<i64> = acc.keys().copied().collect();
    (fp, set)
}

fn cosine(a: &[(i64, f32)], b: &[(i64, f32)]) -> f64 {
    let (mut i, mut j, mut dot) = (0usize, 0usize, 0.0f64);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => { dot += a[i].1 as f64 * b[j].1 as f64; i += 1; j += 1; }
        }
    }
    dot
}

fn jaccard(a: &HashSet<i64>, b: &HashSet<i64>) -> f64 {
    let inter = a.intersection(b).count();
    let union = a.len() + b.len() - inter;
    if union == 0 { 0.0 } else { inter as f64 / union as f64 }
}

fn main() {
    let base = env::args().nth(1).expect("usage: dataset_similarity <base_dir> [n_frames] [min_int] [ms_level]");
    let n_frames: usize = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(800);
    let min_int: f64 = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(100.0);
    let ms_level: i32 = env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(2);
    println!("base={base}  n_frames={n_frames}  min_int={min_int}  ms_level=MS{ms_level}  bins=(mz {MZ_W}, mob {MOB_W})");

    // discover datasets: <base>/<name>/<something>.d
    let mut datasets: Vec<(String, String)> = Vec::new();
    for entry in fs::read_dir(&base).unwrap().flatten() {
        if entry.path().is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(dpath) = fs::read_dir(entry.path()).ok().and_then(|rd| {
                rd.flatten().map(|e| e.path()).find(|p| p.extension().map_or(false, |x| x == "d"))
            }) {
                datasets.push((name, dpath.to_string_lossy().to_string()));
            }
        }
    }
    datasets.sort();
    println!("discovered {} datasets", datasets.len());

    // fingerprint each (sequential: open, sketch, drop)
    let mut names = Vec::new();
    let mut fps = Vec::new();
    let mut sets = Vec::new();
    for (name, dpath) in &datasets {
        let (fp, set) = fingerprint(dpath, n_frames, min_int, ms_level);
        println!("  {:32} tokens={}", name, set.len());
        names.push(name.clone());
        fps.push(fp);
        sets.push(set);
    }

    let n = names.len();
    // pairwise cosine + jaccard
    println!("\n=== cosine (upper) / jaccard (lower) ===");
    print!("{:>26}", "");
    for j in 0..n { print!(" {:>6}", j); }
    println!();
    for i in 0..n {
        print!("{:>2} {:>23}", i, &names[i][..names[i].len().min(23)]);
        for j in 0..n {
            let v = if j >= i { cosine(&fps[i], &fps[j]) } else { jaccard(&sets[i], &sets[j]) };
            print!(" {:>6.3}", v);
        }
        println!();
    }

    // ranked nearest neighbour per dataset (by cosine, excluding self)
    println!("\n=== nearest neighbour by cosine ===");
    for i in 0..n {
        let mut best = (usize::MAX, -1.0f64);
        for j in 0..n {
            if j != i {
                let c = cosine(&fps[i], &fps[j]);
                if c > best.1 { best = (j, c); }
            }
        }
        println!("  {:32} -> {:32} cos {:.3}", names[i], names[best.0], best.1);
    }
}
