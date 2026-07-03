//! S1 self-test + scale test on a TimSim `.d`.
//!
//! Builds mobility-window units from real MS2 frames (top-N peak-picked into
//! spectrum-like bags), times the build in parallel, extrapolates to the full
//! dataset, and — for a small enough slice — runs the self-query analysis:
//! self-retrieval sanity, candidate volume on REAL clustered m/z, the effect of
//! the mobility gate, and a first same-mobility/other-frame recurrence signal.
//!
//! Run: cargo run --release --example recurrence_selftest -p rustdf -- <path.d> [n_frames]

use std::collections::{HashMap, HashSet};
use std::env;
use std::time::Instant;

use mscore::algorithm::lsh::simhash::{CosineSimHash, Projection};
use mscore::algorithm::lsh::LshScheme;
use mscore::timstof::lsh::{
    frame_to_units, FrameUnit, IntensityTransform, MzFeatureMap, WindowConfig,
};
use rayon::prelude::*;
use rustdf::data::dia::TimsDatasetDIA;
use rustdf::data::handle::TimsData;

const HALF_WIDTH: usize = 5;
const STRIDE: usize = 5;
const TOP_N: usize = 25; // peaks kept per window (spectrum-like)
const MOB_GATE_SCANS: i32 = 5;
const RECUR_COS: f64 = 0.5;
const ANALYSIS_CAP: usize = 80_000; // run the O(n·cand) analysis only below this

/// Inclusive DIA isolation-tile scan ranges for a frame (empty = no clamp).
fn segments(ds: &TimsDatasetDIA, frame_id: i32) -> Vec<(i32, i32)> {
    ds.dia_index
        .frame_to_group
        .get(&(frame_id as u32))
        .and_then(|g| ds.dia_index.group_to_scan_unions.get(g))
        .map(|v| v.iter().map(|&(lo, hi)| (lo as i32, hi as i32)).collect())
        .unwrap_or_default()
}

fn cosine(a: &[(i64, f32)], b: &[(i64, f32)]) -> f64 {
    let (mut i, mut j, mut dot) = (0usize, 0usize, 0.0f64);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                dot += a[i].1 as f64 * b[j].1 as f64;
                i += 1;
                j += 1;
            }
        }
    }
    dot
}

fn main() {
    let path = env::args().nth(1).expect("usage: recurrence_selftest <path.d> [n_frames]");
    let n_frames: usize = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(40);
    println!("opening {path} ...");
    let ds = TimsDatasetDIA::new("NO_SDK", &path, false, false);

    let mut ms2: Vec<(u32, f64)> = ds
        .meta_data
        .iter()
        .filter(|m| m.ms_ms_type != 0)
        .map(|m| (m.id as u32, m.time))
        .collect();
    ms2.sort_by_key(|&(id, _)| id);
    let total_ms2 = ms2.len();
    let start = (ms2.len().saturating_sub(n_frames)) / 2;
    let slice: Vec<u32> = ms2.into_iter().skip(start).take(n_frames).map(|(id, _)| id).collect();
    println!("MS2 frames: {total_ms2} total; processing {} (middle slice)", slice.len());

    let map = MzFeatureMap::new(2.0, 6.0).unwrap();
    let cfg = WindowConfig {
        half_width: HALF_WIDTH,
        stride: STRIDE,
        transform: IntensityTransform::Sqrt,
        feature_map: map,
        top_n: Some(TOP_N),
    };

    // --- build (timed) ---
    let t = Instant::now();
    let frames = ds.get_slice(slice.clone(), 8).frames;
    let read_t = t.elapsed();

    let dia = &ds; // borrow for the parallel closure
    let t = Instant::now();
    let units: Vec<FrameUnit> = frames
        .par_iter()
        .flat_map_iter(|f| frame_to_units(f, &cfg, &segments(dia, f.frame_id)))
        .collect();
    let unit_t = t.elapsed();

    if units.is_empty() {
        println!("no units produced");
        return;
    }

    let hasher = CosineSimHash::new(0xC0FFEE, 24, 10, Projection::Gaussian).unwrap();
    let t = Instant::now();
    let sigs: Vec<Vec<u64>> = units.par_iter().map(|u| hasher.signature(&u.features)).collect();
    let hash_t = t.elapsed();

    // --- scale report ---
    let mean_feat = units.iter().map(|u| u.features.len()).sum::<usize>() as f64 / units.len() as f64;
    let upf = units.len() as f64 / slice.len() as f64;
    let full_units = upf * total_ms2 as f64;
    let scale = total_ms2 as f64 / slice.len() as f64;
    let full_build = (read_t + unit_t + hash_t).as_secs_f64() * scale;
    let mem_gb = full_units * mean_feat * 12.0 / 1e9; // (i64 id + f32 val) CSR
    println!(
        "\n--- build ({} frames) ---\n\
         units {}  ({:.0}/frame, mean {:.0} feat/unit after top-{})\n\
         read {:.2?}  units {:.2?}  hash {:.2?}\n\
         --- extrapolated to full {total_ms2} MS2 frames ---\n\
         ~{:.1}M units,  ~{:.0}s build,  ~{:.1} GB feature store",
        slice.len(), units.len(), upf, mean_feat, TOP_N,
        read_t, unit_t, hash_t,
        full_units / 1e6, full_build, mem_gb,
    );

    if units.len() > ANALYSIS_CAP {
        println!("\n(slice too large for the O(n) self-query analysis; scale numbers above)");
        return;
    }

    // --- self-query analysis (uses precomputed sigs) ---
    let mut tables = vec![HashMap::<u64, Vec<u32>>::new(); hasher.num_bands()];
    for (u, sig) in sigs.iter().enumerate() {
        for (band, &key) in sig.iter().enumerate() {
            tables[band].entry(key).or_default().push(u as u32);
        }
    }
    let candidates = |sig: &[u64]| -> HashSet<u32> {
        let mut s = HashSet::new();
        for (band, key) in sig.iter().enumerate() {
            if let Some(v) = tables[band].get(key) {
                s.extend(v.iter().copied());
            }
        }
        s
    };

    let n = units.len();
    let (mut self_hit, mut cand_sum, mut gated_sum, mut recur_hit) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..n {
        let (q, qm) = (&units[i].features, &units[i].meta);
        let cands = candidates(&sigs[i]);
        cand_sum += cands.len();
        if cands.contains(&(i as u32)) {
            self_hit += 1;
        }
        let mut gated = 0usize;
        let mut sibling = false;
        for &u in &cands {
            let um = &units[u as usize].meta;
            if (um.center_scan - qm.center_scan).abs() <= MOB_GATE_SCANS {
                gated += 1;
                if u != i as u32 && um.frame_id != qm.frame_id && cosine(q, &units[u as usize].features) >= RECUR_COS {
                    sibling = true;
                }
            }
        }
        gated_sum += gated;
        if sibling {
            recur_hit += 1;
        }
    }
    let nf = n as f64;
    println!(
        "\n--- self-query (real m/z, top-{TOP_N}) ---\n\
         self-retrieval {:.3}   cand/q {:.1}   after mobility gate {:.1}\n\
         sibling recurrence (same mobility, other frame, cos>={RECUR_COS}): {:.3}",
        self_hit as f64 / nf, cand_sum as f64 / nf, gated_sum as f64 / nf, recur_hit as f64 / nf,
    );
}
