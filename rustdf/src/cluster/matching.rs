use std::collections::HashMap;
use ndarray::Array2;
use crate::cluster::cluster_eval::{ClusterResult, LinkCandidate};
use std::cmp::Ordering;

fn clamp01_f64(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x }
}

/// Solve per-group 1:1 MS2→MS1 assignments with Jonker–Volgenant (via `lapjv`).
/// - `candidates`: many-to-many edges with scores (higher is better).
/// - `min_score`: drop weak assignments after solving (range 0..1).
/// Returns a de-duplicated set of `LinkCandidate`s (one-to-one within each group).
pub fn assign_ms2_to_ms1_lapjv(
    _ms1: &[ClusterResult],
    _ms2: &[ClusterResult],
    candidates: &[LinkCandidate],
    min_score: f32,
) -> Vec<LinkCandidate> {
    // Group edges by DIA group
    let mut by_group: HashMap<u32, Vec<&LinkCandidate>> = HashMap::new();
    for c in candidates {
        by_group.entry(c.group).or_default().push(c);
    }

    let mut selected: Vec<LinkCandidate> = Vec::new();

    for (g, edges) in by_group {
        // Build local index maps: rows = MS2, cols = MS1
        let mut ms2_list: Vec<usize> = Vec::new();
        let mut ms1_list: Vec<usize> = Vec::new();
        let mut ms2_loc: HashMap<usize, usize> = HashMap::new();
        let mut ms1_loc: HashMap<usize, usize> = HashMap::new();

        for e in &edges {
            if !ms2_loc.contains_key(&e.ms2_idx) {
                ms2_loc.insert(e.ms2_idx, ms2_list.len());
                ms2_list.push(e.ms2_idx);
            }
            if !ms1_loc.contains_key(&e.ms1_idx) {
                ms1_loc.insert(e.ms1_idx, ms1_list.len());
                ms1_list.push(e.ms1_idx);
            }
        }

        let n_rows = ms2_list.len();
        let n_cols = ms1_list.len();
        if n_rows == 0 || n_cols == 0 {
            continue;
        }

        // Rectangular cost matrix (Array2<f64>), default = 1.0 (discourage).
        let mut mat: Array2<f64> = Array2::from_elem((n_rows, n_cols), 1.0);

        // Fill with best (lowest) cost for each edge: cost = 1 - score
        for e in &edges {
            let i = ms2_loc[&e.ms2_idx];
            let j = ms1_loc[&e.ms1_idx];
            let s = clamp01_f64(e.score as f64);
            let c = 1.0 - s;
            // keep the smallest cost if multiple edges map to same (i,j)
            if c < mat[(i, j)] {
                mat[(i, j)] = c;
            }
        }

        // Solve LAP; returns (rowsol, colsol)
        let (rowsol, _colsol) = lapjv::lapjv(&mat).expect("lapjv assignment failed");

        // rowsol[i] = chosen column j for row i, or an out-of-range value for 'none'
        for (i, &j) in rowsol.iter().enumerate() {
            if j >= n_cols { continue; } // treat out-of-range as unassigned

            let cost = mat[(i, j)] as f32;
            // safer than `.clamp()` with ambiguous floats:
            let mut score = 1.0_f32 - cost;
            if score < 0.0 { score = 0.0; }
            if score > 1.0 { score = 1.0; }
            if score < min_score { continue; }

            selected.push(LinkCandidate {
                ms1_idx: ms1_list[j],
                ms2_idx: ms2_list[i],
                group: g,
                score,
            });
        }
    }

    // sort best-first (optional)
    selected.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    selected
}

pub fn build_precursor_fragment_annotation(
    ms1: &[ClusterResult],
    ms2: &[ClusterResult],
    candidates: &[LinkCandidate],
    min_score: f32,
) -> Vec<(ClusterResult, Vec<ClusterResult>)> {
    // 1) One-to-one per group (many groups can still map to same MS1)
    let selected = assign_ms2_to_ms1_lapjv(ms1, ms2, candidates, min_score);

    // Keep scores so we can sort fragments per precursor by quality
    let mut score_map: HashMap<(usize, usize), f32> = HashMap::new();
    for s in &selected {
        score_map.insert((s.ms1_idx, s.ms2_idx), s.score);
    }

    // 2) Group MS2 by MS1
    let mut by_ms1: HashMap<usize, Vec<usize>> = HashMap::new();
    for s in selected {
        by_ms1.entry(s.ms1_idx).or_default().push(s.ms2_idx);
    }

    // 3) Materialize (ClusterResult(ms1), Vec<ClusterResult(ms2)>) pairs
    let mut out: Vec<(ClusterResult, Vec<ClusterResult>)> = Vec::new();
    for (i_ms1, ms2_idxs) in by_ms1 {
        // sort fragments by score desc (stable for reproducibility)
        let mut ms2_sorted = ms2_idxs;
        ms2_sorted.sort_by(|&a, &b| {
            let sa = *score_map.get(&(i_ms1, a)).unwrap_or(&0.0);
            let sb = *score_map.get(&(i_ms1, b)).unwrap_or(&0.0);
            sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
        });

        let frags: Vec<ClusterResult> = ms2_sorted.into_iter().map(|j| ms2[j].clone()).collect();
        out.push((ms1[i_ms1].clone(), frags));
    }

    // optional: sort MS1 entries by some priority (e.g., precursor intensity)
    out.sort_by(|a, b| {
        b.0.raw_sum
            .partial_cmp(&a.0.raw_sum)
            .unwrap_or(Ordering::Equal)
    });

    out
}