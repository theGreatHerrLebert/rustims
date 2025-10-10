// rustdf/src/cluster/matching.rs
use std::collections::{HashMap, HashSet};
use ndarray::Array2;
use crate::cluster::cluster_eval::{ClusterResult, LinkCandidate};

/// One-to-one MS2→MS1 assignment within each DIA group using LAPJV.
/// - Pads rectangular problems to square with dummy rows/cols.
/// - Returns only real↔real matches with score >= min_score.
/// - Never panics.
pub fn assign_ms2_to_ms1_lapjv(
    ms1: &[ClusterResult],
    ms2: &[ClusterResult],
    candidates: &[LinkCandidate],
    min_score: f32,
) -> Result<Vec<LinkCandidate>, String> {
    // group ms2 by window_group
    let mut group_to_ms2: HashMap<u32, Vec<usize>> = HashMap::new();
    for (j, c2) in ms2.iter().enumerate() {
        if let Some(g) = c2.window_group {
            group_to_ms2.entry(g).or_default().push(j);
        }
    }

    // index ms1 by groups that cover their precursor m/z
    let mut group_to_ms1: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, c1) in ms1.iter().enumerate() {
        if let Some(ref gs) = c1.window_groups_covering_mz {
            for &g in gs {
                group_to_ms1.entry(g).or_default().push(i);
            }
        }
    }

    // bucket the candidate edges by group (keep only compatible ones)
    let mut group_edges: HashMap<u32, Vec<LinkCandidate>> = HashMap::new();
    for e in candidates {
        let g = match ms2.get(e.ms2_idx).and_then(|c| c.window_group) {
            Some(x) => x,
            None => continue,
        };
        // ensure the ms1 side could be in g
        if let Some(ms1s) = group_to_ms1.get(&g) {
            if ms1s.contains(&e.ms1_idx) {
                group_edges.entry(g).or_default().push(e.clone());
            }
        }
    }

    let mut selected: Vec<LinkCandidate> = Vec::new();

    for (g, ms2_list) in group_to_ms2.into_iter() {
        // compatible ms1 list for this group
        let Some(ms1_list) = group_to_ms1.get(&g) else { continue };

        if ms2_list.is_empty() || ms1_list.is_empty() {
            continue;
        }

        // local index maps
        let mut ms1_loc: HashMap<usize, usize> = HashMap::new();
        for (r, &i_ms1) in ms1_list.iter().enumerate() { ms1_loc.insert(i_ms1, r); }
        let mut ms2_loc: HashMap<usize, usize> = HashMap::new();
        for (c, &j_ms2) in ms2_list.iter().enumerate() { ms2_loc.insert(j_ms2, c); }

        // collect edges for this group
        let edges = group_edges.get(&g).cloned().unwrap_or_default();

        // dimensions and padding
        let r = ms1_list.len();
        let c = ms2_list.len();
        let n = r.max(c);
        let big: f64 = 1e6;

        // build square cost matrix
        let mut mat = Array2::<f64>::from_elem((n, n), big);

        // track which pairs are real
        let mut real_pairs: HashSet<(usize, usize)> = HashSet::new();

        for e in edges {
            if e.score < min_score { continue; }
            if let (Some(&rr), Some(&cc)) = (ms1_loc.get(&e.ms1_idx), ms2_loc.get(&e.ms2_idx)) {
                // cost = 1 - score (score in [0,1])
                let cost = (1.0f32 - e.score) as f64;
                // Keep the best (lowest) cost if multiple edges hit same cell
                let cell = mat.get_mut((rr, cc)).unwrap();
                if cost < *cell {
                    *cell = cost;
                }
                real_pairs.insert((rr, cc));
            }
        }

        // If no viable edges, skip this group
        if real_pairs.is_empty() {
            continue;
        }

        // Solve LAP
        let res = lapjv::lapjv(&mat);
        let (rowsol, _colsol) = match res {
            Ok(rc) => rc,
            Err(e) => {
                // Do not panic: skip this group with a message
                eprintln!("lapjv failed for group {}: {:?}", g, e);
                continue;
            }
        };

        // Translate back to global indices; only keep real↔real, not dummy
        for (rr, &cc) in rowsol.iter().enumerate() {
            if rr >= r || cc >= c { continue; } // dummy row/col
            if !real_pairs.contains(&(rr, cc)) { continue; } // not a filled real edge

            let i_ms1 = ms1_list[rr];
            let j_ms2 = ms2_list[cc];

            // derive score back from the cost in the matrix
            let cost = mat[(rr, cc)] as f32;
            let score = (1.0f32 - cost).max(0.0).min(1.0);
            if score < min_score { continue; }

            selected.push(LinkCandidate { ms1_idx: i_ms1, ms2_idx: j_ms2, score, group: g } );
        }
    }

    Ok(selected)
}

pub fn build_precursor_fragment_annotation(
    ms1: &[ClusterResult],
    ms2: &[ClusterResult],
    candidates: &[LinkCandidate],
    min_score: f32,
) -> Result<Vec<(ClusterResult, Vec<ClusterResult>)>, String> {
    let selected = assign_ms2_to_ms1_lapjv(ms1, ms2, candidates, min_score)?;

    use std::cmp::Ordering;
    use std::collections::HashMap;

    let mut score_map: HashMap<(usize, usize), f32> = HashMap::new();
    for s in &selected {
        score_map.insert((s.ms1_idx, s.ms2_idx), s.score);
    }

    let mut by_ms1: HashMap<usize, Vec<usize>> = HashMap::new();
    for s in selected {
        by_ms1.entry(s.ms1_idx).or_default().push(s.ms2_idx);
    }

    let mut out: Vec<(ClusterResult, Vec<ClusterResult>)> = Vec::new();
    for (i_ms1, mut ms2_idxs) in by_ms1 {
        ms2_idxs.sort_by(|&a, &b| {
            let sa = *score_map.get(&(i_ms1, a)).unwrap_or(&0.0);
            let sb = *score_map.get(&(i_ms1, b)).unwrap_or(&0.0);
            sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
        });
        let frags: Vec<ClusterResult> = ms2_idxs.into_iter().map(|j| ms2[j].clone()).collect();
        out.push((ms1[i_ms1].clone(), frags));
    }

    out.sort_by(|a, b| {
        b.0.raw_sum
            .partial_cmp(&a.0.raw_sum)
            .unwrap_or(Ordering::Equal)
    });

    Ok(out)
}