//! Cross-run MBR test: index run A, query run B, where B is a TimSim replicate of
//! A (same peptides/ions/abundances; only per-analyte RT+IM Gaussian drift + an
//! independent acquisition-noise realization). Because B is `from_existing` A, the
//! two runs share the SAME `ion_id` space — so "the same analyte in A and B" is
//! exact ground truth, and we can finally ask the MBR question the single-run
//! experiments could not: does the SimHash index match the same analyte ACROSS
//! runs despite drift?
//!
//! Metrics (per (m,n) band setting):
//!   - TRUE same-dominant-ion cross-run pair cosine: do A-unit and B-unit of the
//!     same analyte actually look alike across two independent acquisitions?
//!   - LSH recall  = of B units with a true A match (same dominant ion, |Δscan|<=gate),
//!                   fraction for which LSH returns >=1 true A match.
//!   - LSH ion-precision = of LSH B->A candidate pairs, fraction that are true.
//!   - cand-cos (LABEL-INDEPENDENT, the headline) = true cosine of the B->A pairs
//!     LSH actually returns. High cand-cos across runs == MBR works.
//!
//! Run: cargo run --release --example mbr_crossrun -p rustdf -- <A.d> <A.db> <B.d> <B.db> [n_frames]

use std::collections::{HashMap, HashSet};
use std::env;

use mscore::algorithm::lsh::simhash::{CosineSimHash, Projection};
use mscore::algorithm::lsh::LshScheme;
use mscore::timstof::frame::TimsFrame;
use mscore::timstof::lsh::{frame_to_units, FrameUnit, IntensityTransform, MzFeatureMap, WindowConfig};
use rayon::prelude::*;
use rusqlite::Connection;
use rustdf::data::dia::TimsDatasetDIA;
use rustdf::data::handle::TimsData;

const HALF_WIDTH: usize = 5;
const STRIDE: usize = 5;
const TOP_N: usize = 25;
const MOB_GATE: i32 = 5;

fn parse_int_list(s: &str) -> Vec<i32> {
    s.trim_matches(|c| c == '[' || c == ']').split(',').filter_map(|t| t.trim().parse::<i32>().ok()).collect()
}
fn parse_float_list(s: &str) -> Vec<f32> {
    s.trim_matches(|c| c == '[' || c == ']').split(',').filter_map(|t| t.trim().parse::<f32>().ok()).collect()
}

fn segments(ds: &TimsDatasetDIA, frame_id: i32) -> Vec<(i32, i32)> {
    ds.dia_index.frame_to_group.get(&(frame_id as u32))
        .and_then(|g| ds.dia_index.group_to_scan_unions.get(g))
        .map(|v| v.iter().map(|&(lo, hi)| (lo as i32, hi as i32)).collect())
        .unwrap_or_default()
}
fn isolation_window(ds: &TimsDatasetDIA, frame_id: i32, center_scan: i32) -> Option<(f64, f64)> {
    let g = *ds.dia_index.frame_to_group.get(&(frame_id as u32))?;
    ds.program_slices_for_group(g).into_iter()
        .find(|s| center_scan >= s.scan_lo as i32 && center_scan <= s.scan_hi as i32)
        .map(|s| (s.mz_lo, s.mz_hi))
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

struct Ion { ion_id: i64, peptide_id: i64, mz: f64, rel_ab: f64, scans: Vec<i32>, scan_ab: Vec<f32> }

/// Drop peaks below a raw-intensity floor, rebuilding a parallel-array TimsFrame.
/// This is the pre-hash denoiser: remove the blank-overlay noise peaks BEFORE
/// top-N picking so they can't pollute an analyte window's signature.
fn filter_frame(f: &TimsFrame, floor: f64) -> TimsFrame {
    let keep: Vec<usize> = (0..f.ims_frame.intensity.len()).filter(|&i| f.ims_frame.intensity[i] >= floor).collect();
    TimsFrame::new(
        f.frame_id,
        f.ms_type.clone(),
        f.ims_frame.retention_time,
        keep.iter().map(|&i| f.scan[i]).collect(),
        keep.iter().map(|&i| f.ims_frame.mobility[i]).collect(),
        keep.iter().map(|&i| f.tof[i]).collect(),
        keep.iter().map(|&i| f.ims_frame.mz[i]).collect(),
        keep.iter().map(|&i| f.ims_frame.intensity[i]).collect(),
    )
}

/// Build mobility-window units for the middle `n_frames` MS2 frames of one run and
/// label each by its DOMINANT ion (argmax local abundance), using that run's own DB
/// (frame_occurrence reflects this run's drifted RT). `min_int` drops peaks below a
/// raw-intensity floor before unitization (0 = no denoising). Returns (units, label).
fn build_labeled(path: &str, db_path: &str, n_frames: usize, min_int: f64, top_n: Option<usize>) -> (Vec<FrameUnit>, Vec<Option<i64>>) {
    let ds = TimsDatasetDIA::new("NO_SDK", path, false, false);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let mut ms2: Vec<(u32, f64)> = ds.meta_data.iter().filter(|m| m.ms_ms_type != 0).map(|m| (m.id as u32, m.time)).collect();
    ms2.sort_by_key(|&(id, _)| id);
    let start = (ms2.len().saturating_sub(n_frames)) / 2;
    let slice: Vec<u32> = ms2[start..(start + n_frames).min(ms2.len())].iter().map(|&(id, _)| id).collect();
    let slice_set: HashSet<i32> = slice.iter().map(|&f| f as i32).collect();

    let map = MzFeatureMap::new(2.0, 6.0).unwrap();
    let cfg = WindowConfig { half_width: HALF_WIDTH, stride: STRIDE, transform: IntensityTransform::Sqrt, feature_map: map, top_n };
    let frames0 = ds.get_slice(slice.clone(), threads).frames;
    if min_int <= 0.0 {
        let mut ints: Vec<f64> = frames0.iter().flat_map(|f| f.ims_frame.intensity.iter().copied()).collect();
        ints.sort_by(|a, b| a.total_cmp(b));
        let q = |p: f64| if ints.is_empty() { 0.0 } else { ints[((p * (ints.len() - 1) as f64) as usize).min(ints.len() - 1)] };
        println!("  [{}] raw peak intensity: n={} p50 {:.0} p90 {:.0} p99 {:.0} max {:.0}", path.rsplit('/').next().unwrap_or(""), ints.len(), q(0.5), q(0.9), q(0.99), q(1.0));
    }
    let frames: Vec<TimsFrame> = if min_int > 0.0 {
        frames0.par_iter().map(|f| filter_frame(f, min_int)).collect()
    } else {
        frames0
    };
    let dia = &ds;
    let units: Vec<FrameUnit> = frames.par_iter().flat_map_iter(|f| frame_to_units(f, &cfg, &segments(dia, f.frame_id))).collect();

    // ground truth from this run's DB
    let con = Connection::open(db_path).unwrap();
    let mut pep_frames: HashMap<i64, HashSet<i32>> = HashMap::new();
    let mut pep_frame_ab: HashMap<i64, HashMap<i32, f32>> = HashMap::new();
    {
        let mut stmt = con.prepare("SELECT peptide_id, frame_occurrence, frame_abundance FROM peptides").unwrap();
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))).unwrap();
        for row in rows.flatten() {
            let (fo, fa) = (parse_int_list(&row.1), parse_float_list(&row.2));
            let (mut frames, mut abmap) = (HashSet::new(), HashMap::new());
            for (i, &f) in fo.iter().enumerate() {
                if slice_set.contains(&f) { frames.insert(f); abmap.insert(f, *fa.get(i).unwrap_or(&0.0)); }
            }
            if !frames.is_empty() { pep_frames.insert(row.0, frames); pep_frame_ab.insert(row.0, abmap); }
        }
    }
    let mut ions: Vec<Ion> = Vec::new();
    {
        let mut stmt = con.prepare("SELECT ion_id, peptide_id, mz, relative_abundance, scan_occurrence, scan_abundance FROM ions").unwrap();
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, f64>(2)?, r.get::<_, f64>(3)?, r.get::<_, String>(4)?, r.get::<_, String>(5)?))).unwrap();
        for row in rows.flatten() {
            if pep_frames.contains_key(&row.1) {
                ions.push(Ion { ion_id: row.0, peptide_id: row.1, mz: row.2, rel_ab: row.3, scans: parse_int_list(&row.4), scan_ab: parse_float_list(&row.5) });
            }
        }
    }
    let mut frame_ions: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, ion) in ions.iter().enumerate() {
        for &f in &pep_frames[&ion.peptide_id] { frame_ions.entry(f).or_default().push(idx); }
    }
    let labels: Vec<Option<i64>> = units.par_iter().map(|u| {
        let (f, cs) = (u.meta.frame_id, u.meta.center_scan);
        let mzwin = isolation_window(dia, f, cs);
        let mut best: Option<(i64, f64)> = None;
        if let Some(cand) = frame_ions.get(&f) {
            for &ii in cand {
                let ion = &ions[ii];
                if let Some((lo, hi)) = mzwin { if ion.mz < lo || ion.mz > hi { continue; } }
                let (mut sab, mut matched) = (0.0f64, false);
                for (k, &s) in ion.scans.iter().enumerate() {
                    if (s - cs).abs() <= HALF_WIDTH as i32 { matched = true; sab += *ion.scan_ab.get(k).unwrap_or(&0.0) as f64; }
                }
                if !matched { continue; }
                let fab = pep_frame_ab.get(&ion.peptide_id).and_then(|m| m.get(&f)).copied().unwrap_or(0.0) as f64;
                let local = ion.rel_ab * fab * sab;
                if best.map_or(true, |(_, b)| local > b) { best = Some((ion.ion_id, local)); }
            }
        }
        best.map(|(iid, _)| iid)
    }).collect();
    (units, labels)
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return f64::NAN; }
    sorted[((p * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1)]
}

fn main() {
    let a_d = env::args().nth(1).expect("usage: mbr_crossrun <A.d> <A.db> <B.d> <B.db> [n_frames]");
    let a_db = env::args().nth(2).unwrap();
    let b_d = env::args().nth(3).unwrap();
    let b_db = env::args().nth(4).unwrap();
    let n_frames: usize = env::args().nth(5).and_then(|s| s.parse().ok()).unwrap_or(100000);
    let min_int: f64 = env::args().nth(6).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    // 7th arg: top-N peak cap. "all"/"0" = keep ALL peaks (let intensity weighting
    // suppress noise); a number = old top-N sub-sampling.
    let top_n: Option<usize> = match env::args().nth(7).as_deref() {
        None => Some(TOP_N),
        Some("all") | Some("0") => None,
        Some(s) => s.parse().ok(),
    };
    println!("min_peak_intensity floor = {min_int}   top_n = {:?}", top_n);

    let (units_a, lab_a) = build_labeled(&a_d, &a_db, n_frames, min_int, top_n);
    let (units_b, lab_b) = build_labeled(&b_d, &b_db, n_frames, min_int, top_n);
    let la = lab_a.iter().filter(|l| l.is_some()).count();
    let lb = lab_b.iter().filter(|l| l.is_some()).count();
    println!("A units {} ({} labeled)  |  B units {} ({} labeled)", units_a.len(), la, units_b.len(), lb);

    // shared-ion_id sanity: how many dominant ions overlap between the two runs?
    let ions_a: HashSet<i64> = lab_a.iter().flatten().copied().collect();
    let ions_b: HashSet<i64> = lab_b.iter().flatten().copied().collect();
    println!("distinct dominant ions: A {}  B {}  shared {}", ions_a.len(), ions_b.len(), ions_a.intersection(&ions_b).count());

    // A index: dominant ion -> A unit indices (for ground-truth cross-run pairing)
    let mut a_by_ion: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, l) in lab_a.iter().enumerate() { if let Some(iid) = l { a_by_ion.entry(*iid).or_default().push(i); } }

    // TRUE cross-run same-dominant-ion pair cosine (mobility-gated), B-unit vs its A twins
    let mut true_cos: Vec<f64> = Vec::new();
    for (bi, l) in lab_b.iter().enumerate() {
        if let Some(iid) = l {
            if let Some(ais) = a_by_ion.get(iid) {
                let bm = &units_b[bi].meta;
                // best (max-cosine) true A twin within the mobility gate
                let mut best = f64::NAN;
                for &ai in ais {
                    if (units_a[ai].meta.center_scan - bm.center_scan).abs() <= MOB_GATE {
                        let c = cosine(&units_b[bi].features, &units_a[ai].features);
                        if !(best >= c) { best = c; }
                    }
                }
                if !best.is_nan() { true_cos.push(best); }
            }
        }
    }
    true_cos.sort_by(|a, b| a.total_cmp(b));
    let tf = |t: f64| true_cos.iter().filter(|&&c| c >= t).count() as f64 / true_cos.len().max(1) as f64;
    println!("\n--- TRUE cross-run same-dominant-ion pair cosine (best A twin per B unit, n={}) ---", true_cos.len());
    println!("  p10 {:.3}  p25 {:.3}  p50 {:.3}  p75 {:.3}  p90 {:.3}", pct(&true_cos,0.10), pct(&true_cos,0.25), pct(&true_cos,0.50), pct(&true_cos,0.75), pct(&true_cos,0.90));
    println!("  frac>=0.5 {:.2}  >=0.7 {:.2}  >=0.9 {:.2}", tf(0.5), tf(0.7), tf(0.9));

    // has a true cross-run match?
    let b_has_match: Vec<bool> = lab_b.iter().enumerate().map(|(bi, l)| {
        l.and_then(|iid| a_by_ion.get(&iid)).map_or(false, |ais| {
            let bm = &units_b[bi].meta;
            ais.iter().any(|&ai| (units_a[ai].meta.center_scan - bm.center_scan).abs() <= MOB_GATE)
        })
    }).collect();
    let n_matchable = b_has_match.iter().filter(|&&x| x).count();

    println!("\n--- LSH cross-run recall / ion-precision / candidate cosine (B query -> A index) ---");
    println!("{:>9} {:>9} {:>10} {:>8}   headline: TRUE cosine of returned B->A pairs", "m x n", "recall", "ion-prec", "cand/q");
    for &(m, nbits) in &[(32usize, 16usize), (64, 32)] {
        let h = CosineSimHash::new(0xC0FFEE, m, nbits, Projection::Gaussian).unwrap();
        let sig_a: Vec<Vec<u64>> = units_a.par_iter().map(|u| h.signature(&u.features)).collect();
        let sig_b: Vec<Vec<u64>> = units_b.par_iter().map(|u| h.signature(&u.features)).collect();
        // band tables over A
        let mut tbl = vec![HashMap::<u64, Vec<u32>>::new(); m];
        for (u, sig) in sig_a.iter().enumerate() {
            for (band, &key) in sig.iter().enumerate() { tbl[band].entry(key).or_default().push(u as u32); }
        }
        let (mut recall_hit, mut pair_seen, mut pair_true) = (0usize, 0usize, 0usize);
        let mut cand_cos: Vec<f64> = Vec::new();
        for bi in 0..units_b.len() {
            let bm = &units_b[bi].meta;
            let mut cands = HashSet::new();
            for (band, &key) in sig_b[bi].iter().enumerate() {
                if let Some(v) = tbl[band].get(&key) { cands.extend(v.iter().copied()); }
            }
            let gated: Vec<usize> = cands.into_iter().map(|c| c as usize)
                .filter(|&ai| (units_a[ai].meta.center_scan - bm.center_scan).abs() <= MOB_GATE).collect();
            let mut hit = false;
            for &ai in &gated {
                pair_seen += 1;
                cand_cos.push(cosine(&units_b[bi].features, &units_a[ai].features));
                if lab_b[bi].is_some() && lab_b[bi] == lab_a[ai] { pair_true += 1; hit = true; }
            }
            if b_has_match[bi] && hit { recall_hit += 1; }
        }
        cand_cos.sort_by(|a, b| a.total_cmp(b));
        let cf = |t: f64| cand_cos.iter().filter(|&&c| c >= t).count() as f64 / cand_cos.len().max(1) as f64;
        println!("{:>4} x {:<3} {:>9.3} {:>10.3} {:>8.2}   cand-cos p50 {:.3} p90 {:.3}  frac>=0.7 {:.2}  (n_cand {})",
            m, nbits,
            if n_matchable > 0 { recall_hit as f64 / n_matchable as f64 } else { 0.0 },
            if pair_seen > 0 { pair_true as f64 / pair_seen as f64 } else { 0.0 },
            pair_seen as f64 / units_b.len().max(1) as f64,
            pct(&cand_cos, 0.50), pct(&cand_cos, 0.90), cf(0.7), cand_cos.len());
    }
}
