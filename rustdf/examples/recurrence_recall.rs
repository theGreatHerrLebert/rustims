//! S1 ground-truth recall: label each mobility-window unit with the true ion(s)
//! present (from TimSim's synthetic_data.db), then measure what the `cos>=0.5`
//! proxy couldn't — the cosine distribution of TRUE same-ion units recurring
//! across frames, and the real recall/precision of the LSH index at the paper's
//! (m,n) band settings. The cosine histogram is the number that picks (m,n).
//!
//! Two labelling modes (4th CLI arg):
//!   `dominant` (default) — a unit is labelled by its single DOMINANT ion, the
//!       co-isolated ion with the largest local abundance
//!       `relative_abundance * frame_abundance(pep,frame) * Σ scan_abundance(in window)`.
//!       A "true pair" = two units whose dominant ion is the same. This is the
//!       stricter, spectrum-honest label: units that actually look alike.
//!   `any` — a unit is labelled by the SET of all co-isolated ions; a "true pair"
//!       shares >=1 ion (the old, weak proxy; kept for A/B comparison).
//!
//! Ground truth: an ion's fragments sit at MS2 frame f, scan s iff
//!   f in frame_occurrence(peptide)  AND  s in scan_occurrence(ion) (± window)
//!   AND ion.mz in the isolation window active at scan s.
//! (diaPASEF: fragments inherit the precursor's mobility, so scan_occurrence —
//! an MS1 property — also places the MS2 fragments.)
//!
//! Run: cargo run --release --example recurrence_recall -p rustdf -- <path.d> <synthetic_data.db> [n_frames] [dominant|any]

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

fn parse_float_list(s: &str) -> Vec<f32> {
    s.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .filter_map(|t| t.trim().parse::<f32>().ok())
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
    rel_ab: f64,
    scans: Vec<i32>,
    scan_ab: Vec<f32>,
}

fn main() {
    let path = env::args().nth(1).expect("usage: recurrence_recall <path.d> <synthetic_data.db> [n_frames] [dominant|any]");
    let db_path = env::args().nth(2).expect("need the synthetic_data.db path");
    let n_frames: usize = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(200);
    let dominant = !matches!(env::args().nth(4).as_deref(), Some("any"));
    println!("label mode: {}", if dominant { "dominant-ion" } else { "shares-any-ion" });

    let ds = TimsDatasetDIA::new("NO_SDK", &path, false, false);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);

    // frame_id -> retention time (s), to stratify candidate pairs by dRT:
    // is the recurrence signal trivial elution-neighbour redundancy (small dRT)
    // or genuine cross-time recurrence (large dRT)?
    let frame_time: HashMap<i32, f64> = ds.meta_data.iter().map(|m| (m.id as i32, m.time)).collect();

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
    // peptide -> (frame -> frame_abundance), restricted to the slice
    let mut pep_frame_ab: HashMap<i64, HashMap<i32, f32>> = HashMap::new();
    {
        let mut stmt = con.prepare("SELECT peptide_id, frame_occurrence, frame_abundance FROM peptides").unwrap();
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)))
            .unwrap();
        for row in rows.flatten() {
            let fo = parse_int_list(&row.1);
            let fa = parse_float_list(&row.2);
            let mut frames = HashSet::new();
            let mut abmap = HashMap::new();
            for (i, &f) in fo.iter().enumerate() {
                if slice_set.contains(&f) {
                    frames.insert(f);
                    abmap.insert(f, *fa.get(i).unwrap_or(&0.0));
                }
            }
            if !frames.is_empty() {
                pep_frames.insert(row.0, frames);
                pep_frame_ab.insert(row.0, abmap);
            }
        }
    }
    let mut ions: Vec<Ion> = Vec::new();
    {
        let mut stmt = con
            .prepare("SELECT ion_id, peptide_id, mz, relative_abundance, scan_occurrence, scan_abundance FROM ions")
            .unwrap();
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, f64>(2)?,
                    r.get::<_, f64>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                ))
            })
            .unwrap();
        for row in rows.flatten() {
            if pep_frames.contains_key(&row.1) {
                ions.push(Ion {
                    ion_id: row.0,
                    peptide_id: row.1,
                    mz: row.2,
                    rel_ab: row.3,
                    scans: parse_int_list(&row.4),
                    scan_ab: parse_float_list(&row.5),
                });
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

    // --- label each unit ---
    // `dominant`: singleton {dominant ion}; `any`: set of all co-isolated ions.
    // Keeping the label a HashSet in both modes lets every downstream stage
    // (ion_units, is_true_pair via set-intersection) stay identical.
    let labels: Vec<HashSet<i64>> = units
        .par_iter()
        .map(|u| {
            let (f, cs) = (u.meta.frame_id, u.meta.center_scan);
            let mzwin = isolation_window(dia, f, cs);
            let mut lbl = HashSet::new();
            let mut best: Option<(i64, f64)> = None;
            if let Some(cand) = frame_ions.get(&f) {
                for &ii in cand {
                    let ion = &ions[ii];
                    if let Some((lo, hi)) = mzwin {
                        if ion.mz < lo || ion.mz > hi {
                            continue;
                        }
                    }
                    // sum this ion's scan abundance within the unit's mobility window
                    let mut sab = 0.0f64;
                    let mut matched = false;
                    for (k, &s) in ion.scans.iter().enumerate() {
                        if (s - cs).abs() <= HALF_WIDTH as i32 {
                            matched = true;
                            sab += *ion.scan_ab.get(k).unwrap_or(&0.0) as f64;
                        }
                    }
                    if !matched {
                        continue;
                    }
                    if dominant {
                        let fab = pep_frame_ab.get(&ion.peptide_id).and_then(|m| m.get(&f)).copied().unwrap_or(0.0) as f64;
                        let local = ion.rel_ab * fab * sab;
                        if best.map_or(true, |(_, b)| local > b) {
                            best = Some((ion.ion_id, local));
                        }
                    } else {
                        lbl.insert(ion.ion_id);
                    }
                }
            }
            if dominant {
                if let Some((iid, _)) = best {
                    lbl.insert(iid);
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
        println!("\n--- TRUE same-{}-ion cross-frame pair cosine (n={}) ---", if dominant { "dominant" } else { "any" }, cosines.len());
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
        // label-independent quality of what LSH actually returns: the TRUE cosine
        // of each unique mobility-gated candidate pair (u<c). This sidesteps the
        // confounded ion-label proxy — for recurrence, high candidate cosine IS
        // the win, whatever ion story produced it.
        let mut cand_cos: Vec<f64> = Vec::new();
        // dRT stratification: bucket each candidate pair's cosine by |t_u - t_c|.
        let rt_edges = [0.0f64, 2.0, 5.0, 15.0, 45.0, 120.0, f64::INFINITY];
        let mut rt_buckets: Vec<Vec<f64>> = vec![Vec::new(); rt_edges.len() - 1];
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
                if c > u {
                    let cos = cosine(&units[u].features, &units[c].features);
                    cand_cos.push(cos);
                    let drt = (frame_time.get(&units[u].meta.frame_id).copied().unwrap_or(0.0)
                        - frame_time.get(&units[c].meta.frame_id).copied().unwrap_or(0.0))
                        .abs();
                    let b = rt_edges.iter().rposition(|&e| drt >= e).unwrap_or(0).min(rt_buckets.len() - 1);
                    rt_buckets[b].push(cos);
                }
            }
            if has_sibling(u) {
                denom += 1;
                if gated.iter().any(|&c| is_true_pair(u, c)) {
                    recall_hit += 1;
                }
            }
        }
        cand_cos.sort_by(|a, b| a.total_cmp(b));
        let (cq, cfrac) = if cand_cos.is_empty() {
            ("n/a".to_string(), 0.0)
        } else {
            let q = |p: f64| cand_cos[((p * (cand_cos.len() - 1) as f64).round() as usize).min(cand_cos.len() - 1)];
            (
                format!("p50 {:.3} p90 {:.3}", q(0.50), q(0.90)),
                cand_cos.iter().filter(|&&c| c >= 0.7).count() as f64 / cand_cos.len() as f64,
            )
        };
        println!(
            "{:>4} x {:<3} {:>9.3} {:>10.3} {:>9.2}   | cand-cos {}  frac>=0.7 {:.2}",
            m, nbits,
            if denom > 0 { recall_hit as f64 / denom as f64 } else { 0.0 },
            if pair_seen > 0 { pair_true as f64 / pair_seen as f64 } else { 0.0 },
            pair_seen as f64 / units.len() as f64,
            cq, cfrac,
        );
        // dRT stratification of candidate-pair cosine (label-independent):
        // does high cosine survive beyond the immediate elution neighbourhood?
        println!("        dRT bucket   pairs   %-of-cand   median-cos   frac>=0.7");
        let total_cand = cand_cos.len().max(1) as f64;
        for (bi, bucket) in rt_buckets.iter().enumerate() {
            let (lo, hi) = (rt_edges[bi], rt_edges[bi + 1]);
            let name = if hi.is_infinite() { format!("{:>4.0}s+   ", lo) } else { format!("{:>4.0}-{:<4.0}s", lo, hi) };
            let (med, f07) = if bucket.is_empty() {
                (f64::NAN, 0.0)
            } else {
                let mut v = bucket.clone();
                v.sort_by(|a, b| a.total_cmp(b));
                (v[v.len() / 2], v.iter().filter(|&&c| c >= 0.7).count() as f64 / v.len() as f64)
            };
            println!(
                "        {:>10}  {:>6}   {:>8.1}%   {:>10.3}   {:>9.2}",
                name, bucket.len(), 100.0 * bucket.len() as f64 / total_cand, med, f07,
            );
        }
    }
}
