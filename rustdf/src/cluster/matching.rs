use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use crate::cluster::cluster_eval::{ClusterResult, LinkCandidate};

// Internal knobs (kept here to avoid binding changes)
const CARDINALITY_ONE_TO_ONE: bool = false; // false => one-to-many
const TOP_K_PER_MS2: usize = 32;             // speed guard
const TOP_K_PER_MS1: Option<usize> = Some(512);

pub fn build_precursor_fragment_annotation(
    ms1: &[ClusterResult],
    ms2: &[ClusterResult],
    candidates: &[LinkCandidate],
    min_score: f32,
) -> Vec<(ClusterResult, Vec<ClusterResult>)> {

    // --- Group MS2 + MS1 by DIA window group for parallelism ----------------
    let mut group_to_ms2: HashMap<u32, Vec<usize>> = HashMap::new();
    for (j, c2) in ms2.iter().enumerate() {
        if let Some(g) = c2.window_group {
            group_to_ms2.entry(g).or_default().push(j);
        }
    }

    let mut group_to_ms1: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, c1) in ms1.iter().enumerate() {
        if let Some(gs) = &c1.window_groups_covering_mz {
            for &g in gs {
                group_to_ms1.entry(g).or_default().push(i);
            }
        }
    }

    // Pre-filter edges to compatible groups
    let mut grouped_edges: HashMap<u32, Vec<LinkCandidate>> = HashMap::new();
    for e in candidates {
        if e.score < min_score { continue; }
        // Need the group from the MS2 cluster (safer than trusting e.group blindly)
        if let Some(g) = ms2.get(e.ms2_idx).and_then(|c| c.window_group) {
            // Also ensure the MS1 is actually compatible with that group
            if let Some(ms1s) = group_to_ms1.get(&g) {
                if ms1s.iter().any(|&idx| idx == e.ms1_idx) {
                    grouped_edges.entry(g).or_default().push(e.clone());
                }
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

    // Solve each group in parallel
    let selected: Vec<LinkCandidate> = work.par_iter()
        .map(|(_g, ms1_list, ms2_list, edges)| {
            // Convert lists to sets for O(1) membership checks
            let ms1_set: HashSet<usize> = ms1_list.iter().copied().collect();
            let ms2_set: HashSet<usize> = ms2_list.iter().copied().collect();

            // 1) Top-K cull per MS2 (and optional per MS1) to keep things fast
            let mut by_ms2: HashMap<usize, Vec<LinkCandidate>> = HashMap::new();
            for e in edges {
                if !ms1_set.contains(&e.ms1_idx) || !ms2_set.contains(&e.ms2_idx) { continue; }
                by_ms2.entry(e.ms2_idx).or_default().push(e.clone());
            }
            for v in by_ms2.values_mut() {
                v.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
                if v.len() > TOP_K_PER_MS2 { v.truncate(TOP_K_PER_MS2); }
            }
            let mut culled: Vec<LinkCandidate> = by_ms2.into_values().flatten().collect();

            if let Some(k) = TOP_K_PER_MS1 {
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

            // 2) Greedy accept best-first
            culled.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

            let mut used_ms2: HashSet<usize> = HashSet::new();
            let mut used_ms1: HashSet<usize> = HashSet::new();
            let mut out: Vec<LinkCandidate> = Vec::new();

            for e in culled {
                if e.score < min_score { continue; }
                if !ms2_set.contains(&e.ms2_idx) { continue; }
                if !ms1_set.contains(&e.ms1_idx) { continue; }

                if used_ms2.contains(&e.ms2_idx) { continue; }
                if CARDINALITY_ONE_TO_ONE && used_ms1.contains(&e.ms1_idx) { continue; }

                out.push(e.clone());
                used_ms2.insert(e.ms2_idx);
                if CARDINALITY_ONE_TO_ONE { used_ms1.insert(e.ms1_idx); }
            }
            out
        })
        .flatten()
        .collect();

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
        let frags = idxs.into_iter().map(|j| ms2[j].clone()).collect::<Vec<_>>();
        out.push((ms1[i_ms1].clone(), frags));
    }

    // Sort precursor blocks by precursor strength (raw_sum)
    out.sort_by(|a, b| b.0.raw_sum.partial_cmp(&a.0.raw_sum).unwrap_or(Ordering::Equal));
    out
}