//! S1 ground-truth recall: label each mobility-window unit with the true ion(s)
//! present (from TimSim's synthetic_data.db), then measure what the `cos>=0.5`
//! proxy couldn't — the cosine distribution of TRUE same-ion units recurring
//! across frames, and the real recall/precision of the LSH index at the paper's
//! (m,n) band settings. The cosine histogram is the number that picks (m,n).
//!
//! Ground truth: an ion's fragments sit at MS2 frame f, scan s iff
//!   f in frame_occurrence(peptide)  AND  s in scan_occurrence(ion) (± window)
//!   AND ion.mz in the isolation window active at scan s.
//! (diaPASEF: fragments inherit the precursor's mobility, so scan_occurrence —
//! an MS1 property — also places the MS2 fragments.)
//!
//! Run: cargo run --release --example recurrence_recall -p rustdf -- <path.d> <synthetic_data.db> [n_frames]

use std::collections::{HashMap, HashSet};
use std::env;

use mscore::algorithm::lsh::simhash::{CosineSimHash, Projection};
use mscore::algorithm::lsh::LshScheme;
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
    s.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .filter_map(|t| t.trim().parse::<i32>().ok())
        .collect()
}

fn segments(ds: &TimsDatasetDIA, frame_id: i32) -> Vec<(i32, i32)> {
    ds.dia_index
        .frame_to_group
        .get(&(frame_id as u32))
        .and_then(|g| ds.dia_index.group_to_scan_unions.get(g))
        .map(|v| v.iter().map(|&(lo, hi)| (lo as i32, hi as i32)).collect())
        .unwrap_or_default()
}

/// Isolation m/z window active at `center_scan` in `frame_id` (the ProgramSlice
/// whose scan range contains it), used to reject ions not co-isolated there.
fn isolation_window(ds: &TimsDatasetDIA, frame_id: i32, center_scan: i32) -> Option<(f64, f64)> {
    let g = *ds.dia_index.frame_to_group.get(&(frame_id as u32))?;
    ds.program_slices_for_group(g)
        .into_iter()
        .find(|s| center_scan >= s.scan_lo as i32 && center_scan <= s.scan_hi as i32)
        .map(|s| (s.mz_lo, s.mz_hi))
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

struct Ion {
    ion_id: i64,
    peptide_id: i64,
    mz: f64,
    scans: Vec<i32>,
}

fn main() {
    let path = env::args().nth(1).expect("usage: recurrence_recall <path.d> <synthetic_data.db> [n_frames]");
    let db_path = env::args().nth(2).expect("need the synthetic_data.db path");
    let n_frames: usize = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(200);

    let ds = TimsDatasetDIA::new("NO_SDK", &path, false, false);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);

    let mut ms2: Vec<(u32, f64)> = ds
        .meta_data
        .iter()
        .filter(|m| m.ms_ms_type != 0)
        .map(|m| (m.id as u32, m.time))
        .collect();
    ms2.sort_by_key(|&(id, _)| id);
    let start = (ms2.len().saturating_sub(n_frames)) / 2;
    let slice: Vec<u32> = ms2[start..(start + n_frames).min(ms2.len())].iter().map(|&(id, _)| id).collect();
    let slice_set: HashSet<i32> = slice.iter().map(|&f| f as i32).collect();
    println!("MS2 frames {} total; slice {} (ids {}..={})", ms2.len(), slice.len(), slice[0], slice[slice.len() - 1]);

    // --- build units ---
    let map = MzFeatureMap::new(2.0, 6.0).unwrap();
    let cfg = WindowConfig { half_width: HALF_WIDTH, stride: STRIDE, transform: IntensityTransform::Sqrt, feature_map: map, top_n: Some(TOP_N) };
    let frames = ds.get_slice(slice.clone(), threads).frames;
    let dia = &ds;
    let units: Vec<FrameUnit> = frames
        .par_iter()
        .flat_map_iter(|f| frame_to_units(f, &cfg, &segments(dia, f.frame_id)))
        .collect();
    println!("units: {}", units.len());

    // --- ground truth from synthetic_data.db ---
    let con = Connection::open(&db_path).unwrap();
    let mut pep_frames: HashMap<i64, HashSet<i32>> = HashMap::new();
    {
        let mut stmt = con.prepare("SELECT peptide_id, frame_occurrence FROM peptides").unwrap();
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))).unwrap();
        for row in rows.flatten() {
            let frames: HashSet<i32> = parse_int_list(&row.1).into_iter().filter(|f| slice_set.contains(f)).collect();
            if !frames.is_empty() {
                pep_frames.insert(row.0, frames);
            }
        }
    }
    let mut ions: Vec<Ion> = Vec::new();
    {
        let mut stmt = con.prepare("SELECT ion_id, peptide_id, mz, scan_occurrence FROM ions").unwrap();
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, f64>(2)?, r.get::<_, String>(3)?)))
            .unwrap();
        for row in rows.flatten() {
            if pep_frames.contains_key(&row.1) {
                ions.push(Ion { ion_id: row.0, peptide_id: row.1, mz: row.2, scans: parse_int_list(&row.3) });
            }
        }
    }
    println!("ground truth: {} peptides / {} ions touch this slice", pep_frames.len(), ions.len());

    // frame -> ion indices whose peptide elutes in that frame
    let mut frame_ions: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, ion) in ions.iter().enumerate() {
        for &f in &pep_frames[&ion.peptide_id] {
            frame_ions.entry(f).or_default().push(idx);
        }
    }

    // --- label each unit with the true ion_ids present ---
    let labels: Vec<HashSet<i64>> = units
        .par_iter()
        .map(|u| {
            let (f, cs) = (u.meta.frame_id, u.meta.center_scan);
            let mzwin = isolation_window(dia, f, cs);
            let mut lbl = HashSet::new();
            if let Some(cand) = frame_ions.get(&f) {
                for &ii in cand {
                    let ion = &ions[ii];
                    if let Some((lo, hi)) = mzwin {
                        if ion.mz < lo || ion.mz > hi {
                            continue;
                        }
                    }
                    if ion.scans.iter().any(|&s| (s - cs).abs() <= HALF_WIDTH as i32) {
                        lbl.insert(ion.ion_id);
                    }
                }
            }
            lbl
        })
        .collect();
    let labeled = labels.iter().filter(|l| !l.is_empty()).count();
    let mean_ions = labels.iter().map(|l| l.len()).sum::<usize>() as f64 / units.len().max(1) as f64;
    println!("labeled units: {labeled}/{} ({:.1}% ) mean ions/unit {:.2}", units.len(), 100.0 * labeled as f64 / units.len() as f64, mean_ions);
    if labeled == 0 {
        println!("\n!! zero labels — frame_occurrence numbering likely != raw tdf frame ids. Fix the mapping before trusting recall.");
        return;
    }

    // ion -> units containing it
    let mut ion_units: HashMap<i64, Vec<usize>> = HashMap::new();
    for (ui, lbl) in labels.iter().enumerate() {
        for &iid in lbl {
            ion_units.entry(iid).or_default().push(ui);
        }
    }

    // --- cosine histogram of TRUE recurring pairs (same ion, diff frame, mobility-close) ---
    let mut cosines: Vec<f64> = Vec::new();
    for us in ion_units.values() {
        let mut v = us.clone();
        v.sort_by_key(|&ui| (units[ui].meta.frame_id, units[ui].meta.center_scan));
        for w in v.windows(2) {
            let (a, b) = (&units[w[0]].meta, &units[w[1]].meta);
            if a.frame_id != b.frame_id && (a.center_scan - b.center_scan).abs() <= MOB_GATE {
                cosines.push(cosine(&units[w[0]].features, &units[w[1]].features));
            }
        }
    }
    cosines.sort_by(|a, b| a.total_cmp(b));
    if cosines.is_empty() {
        println!("\nno true cross-frame recurring pairs in this slice (widen n_frames)");
    } else {
        let q = |p: f64| cosines[((p * (cosines.len() - 1) as f64).round() as usize).min(cosines.len() - 1)];
        let frac = |t: f64| cosines.iter().filter(|&&c| c >= t).count() as f64 / cosines.len() as f64;
        println!("\n--- TRUE same-ion cross-frame pair cosine (n={}) ---", cosines.len());
        println!("  p10 {:.3}  p25 {:.3}  p50 {:.3}  p75 {:.3}  p90 {:.3}", q(0.10), q(0.25), q(0.50), q(0.75), q(0.90));
        println!("  frac>=0.7 {:.2}  >=0.8 {:.2}  >=0.9 {:.2}  >=0.95 {:.2}", frac(0.7), frac(0.8), frac(0.9), frac(0.95));
    }

    // precompute which units have >=1 true sibling
    let has_sibling = |u: usize| -> bool {
        let um = &units[u].meta;
        labels[u].iter().any(|iid| {
            ion_units[iid].iter().any(|&v| v != u && units[v].meta.frame_id != um.frame_id && (units[v].meta.center_scan - um.center_scan).abs() <= MOB_GATE)
        })
    };
    let is_true_pair = |u: usize, v: usize| -> bool {
        let (um, vm) = (&units[u].meta, &units[v].meta);
        um.frame_id != vm.frame_id
            && (um.center_scan - vm.center_scan).abs() <= MOB_GATE
            && labels[u].intersection(&labels[v]).next().is_some()
    };

    // --- LSH recall / precision at the paper's (m,n) ---
    println!("\n--- LSH recall/precision (mobility-gated candidates) ---");
    println!("{:>9} {:>9} {:>10} {:>9}", "m x n", "recall", "precision", "cand/q");
    for &(m, nbits) in &[(32usize, 16usize), (64, 32)] {
        let h = CosineSimHash::new(0xC0FFEE, m, nbits, Projection::Gaussian).unwrap();
        let sgs: Vec<Vec<u64>> = units.par_iter().map(|u| h.signature(&u.features)).collect();
        let mut tbl = vec![HashMap::<u64, Vec<u32>>::new(); m];
        for (u, sig) in sgs.iter().enumerate() {
            for (band, &key) in sig.iter().enumerate() {
                tbl[band].entry(key).or_default().push(u as u32);
            }
        }
        let (mut denom, mut recall_hit, mut pair_seen, mut pair_true) = (0usize, 0usize, 0usize, 0usize);
        for u in 0..units.len() {
            let um = &units[u].meta;
            let mut cands = HashSet::new();
            for (band, &key) in sgs[u].iter().enumerate() {
                if let Some(v) = tbl[band].get(&key) {
                    cands.extend(v.iter().copied());
                }
            }
            // mobility-gated, excluding self
            let gated: Vec<usize> = cands
                .into_iter()
                .map(|c| c as usize)
                .filter(|&c| c != u && (units[c].meta.center_scan - um.center_scan).abs() <= MOB_GATE && units[c].meta.frame_id != um.frame_id)
                .collect();
            for &c in &gated {
                pair_seen += 1;
                if is_true_pair(u, c) {
                    pair_true += 1;
                }
            }
            if has_sibling(u) {
                denom += 1;
                if gated.iter().any(|&c| is_true_pair(u, c)) {
                    recall_hit += 1;
                }
            }
        }
        println!(
            "{:>4} x {:<3} {:>9.3} {:>10.3} {:>9.2}",
            m, nbits,
            if denom > 0 { recall_hit as f64 / denom as f64 } else { 0.0 },
            if pair_seen > 0 { pair_true as f64 / pair_seen as f64 } else { 0.0 },
            pair_seen as f64 / units.len() as f64,
        );
    }
}
