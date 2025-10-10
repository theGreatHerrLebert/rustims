// rustdf/src/cluster/matching.rs
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use crate::cluster::cluster_eval::{ClusterResult, LinkCandidate};

/// Strategy toggle
#[derive(Clone, Copy, Debug)]
pub enum MatchCardinality {
    /// Each MS2 assigned to at most one MS1, MS1 can take many MS2 (typical precursor→fragments)
    OneToMany,
    /// Classical 1↔1 matching
    OneToOne,
}

/// Fast, non-optimal greedy matcher. Parallel across window groups.
/// - culls to top_k_per_ms2 (and optionally per MS1) to keep it snappy
pub fn assign_ms2_to_ms1_greedy_parallel(
    ms1: &[ClusterResult],
    ms2: &[ClusterResult],
    candidates: &[LinkCandidate],
    min_score: f32,
    cardinality: MatchCardinality,
    top_k_per_ms2: usize,          // e.g. 8
    top_k_per_ms1: Option<usize>,  // e.g. Some(32) or None to skip
) -> Vec<LinkCandidate> {
    // group -> MS2 list
    let mut group_to_ms2: HashMap<u32, Vec<usize>> = HashMap::new();
    for (j, c2) in ms2.iter().enumerate() {
        if let Some(g) = c2.window_group {
            group_to_ms2.entry(g).or_default().push(j);
        }
    }

    // group -> MS1 list (covering_mz includes g)
    let mut group_to_ms1: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, c1) in ms1.iter().enumerate() {
        if let Some(gs) = &c1.window_groups_covering_mz {
            for &g in gs {
                group_to_ms1.entry(g).or_default().push(i);
            }
        }
    }

    // group -> edges (already filtered by group compatibility)
    let mut grouped_edges: HashMap<u32, Vec<LinkCandidate>> = HashMap::new();
    for e in candidates {
        if e.score < min_score { continue; }
        let Some(g) = ms2.get(e.ms2_idx).and_then(|c| c.window_group) else { continue; };
        // Keep only edges whose MS1 is actually compatible with this group
        if let Some(ms1s) = group_to_ms1.get(&g) {
            if ms1s.contains(&e.ms1_idx) {
                grouped_edges.entry(g).or_default().push(e.clone());
            }
        }
    }

    // Build per-group work items
    let mut work: Vec<(u32, Vec<usize>, Vec<usize>, Vec<LinkCandidate>)> = Vec::new();
    for (g, ms2_list) in group_to_ms2 {
        if let Some(ms1_list) = group_to_ms1.get(&g) {
            if !ms1_list.is_empty() && !ms2_list.is_empty() {
                let edges = grouped_edges.remove(&g).unwrap_or_default();
                if !edges.is_empty() {
                    work.push((g, ms1_list.clone(), ms2_list, edges));
                }
            }
        }
    }

    // Solve per group in parallel
    work.par_iter()
        .map(|(_g, ms1_list, ms2_list, edges)| {
            // 1) CULL: top-K per MS2 (and optional per MS1)
            let mut by_ms2: HashMap<usize, Vec<LinkCandidate>> = HashMap::new();
            for e in edges {
                by_ms2.entry(e.ms2_idx).or_default().push(e.clone());
            }
            for v in by_ms2.values_mut() {
                v.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
                if v.len() > top_k_per_ms2 { v.truncate(top_k_per_ms2); }
            }
            let mut culled: Vec<LinkCandidate> = by_ms2.into_values().flatten().collect();

            if let Some(k) = top_k_per_ms1 {
                let mut by_ms1: HashMap<usize, Vec<LinkCandidate>> = HashMap::new();
                for e in &culled { by_ms1.entry(e.ms1_idx).or_default().push(e.clone()); }
                let mut culled2 = Vec::with_capacity(culled.len());
                for mut v in by_ms1.into_values() {
                    v.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
                    if v.len() > k { v.truncate(k); }
                    culled2.extend(v);
                }
                culled = culled2;
            }

            // 2) GREEDY: sort by score desc, accept if available
            culled.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

            let mut used_ms2: HashSet<usize> = HashSet::new();
            let mut used_ms1: HashSet<usize> = HashSet::new();
            let mut out: Vec<LinkCandidate> = Vec::new();

            for e in culled {
                if e.score < min_score { continue; }
                if !ms2_list.contains(&e.ms2_idx) { continue; }
                if !ms1_list.contains(&e.ms1_idx) { continue; }

                // enforce cardinality
                if used_ms2.contains(&e.ms2_idx) { continue; }
                if matches!(cardinality, MatchCardinality::OneToOne) && used_ms1.contains(&e.ms1_idx) {
                    continue;
                }

                out.push(e.clone());
                used_ms2.insert(e.ms2_idx);
                if matches!(cardinality, MatchCardinality::OneToOne) {
                    used_ms1.insert(e.ms1_idx);
                }
            }
            out
        })
        .flatten()
        .collect()
}

pub fn build_precursor_fragment_annotation(
    ms1: &[ClusterResult],
    ms2: &[ClusterResult],
    candidates: &[LinkCandidate],
    min_score: f32,
    cardinality: MatchCardinality,   // OneToMany is typical for DIA
    top_k_per_ms2: usize,            // e.g. 8
    top_k_per_ms1: Option<usize>,    // e.g. Some(32)
) -> Vec<(ClusterResult, Vec<ClusterResult>)> {
    let selected = assign_ms2_to_ms1_greedy_parallel(
        ms1, ms2, candidates, min_score, cardinality, top_k_per_ms2, top_k_per_ms1
    );

    // Group selected links by MS1, order fragments by score desc
    let mut score_map: HashMap<(usize, usize), f32> = HashMap::new();
    for s in &selected { score_map.insert((s.ms1_idx, s.ms2_idx), s.score); }

    let mut by_ms1: HashMap<usize, Vec<usize>> = HashMap::new();
    for s in selected { by_ms1.entry(s.ms1_idx).or_default().push(s.ms2_idx); }

    let mut out: Vec<(ClusterResult, Vec<ClusterResult>)> = Vec::new();
    for (i_ms1, mut idxs) in by_ms1 {
        idxs.sort_by(|&a, &b| {
            let sa = *score_map.get(&(i_ms1, a)).unwrap_or(&0.0);
            let sb = *score_map.get(&(i_ms1, b)).unwrap_or(&0.0);
            sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
        });
        let frags = idxs.into_iter().map(|j| ms2[j].clone()).collect();
        out.push((ms1[i_ms1].clone(), frags));
    }

    // sort precursor blocks by precursor strength
    out.sort_by(|a, b| b.0.raw_sum.partial_cmp(&a.0.raw_sum).unwrap_or(Ordering::Equal));
    out
}