// rustdf/src/cluster/cluster_link.rs

use crate::cluster::cluster::ClusterResult1D;
use crate::data::dia::TimsDatasetDIA;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct Ms2ToMs1LinkerOpts {
    /// Minimum RT overlap (Jaccard in **time space**, seconds)
    pub min_rt_jaccard: f32,            // e.g. 0.1
    /// Max |Δ apex RT| in seconds (None disables)
    pub max_rt_apex_sec: Option<f32>,   // e.g. Some(8.0)
    /// Max |Δ apex IM| in scans (None disables)
    pub max_im_apex_scans: Option<f32>, // e.g. Some(5.0)
    /// Require IM overlap of (MS1 cluster vs MS2 cluster) AND program scan ranges
    pub require_im_overlap: bool,
    /// Soft pad on absolute time bounds (seconds) for co-elution check
    pub rt_guard_sec: f64,
    /// PPM pad applied to the MS1 cluster m/z window when gating against DIA isolation windows
    pub mz_ppm: f32,
}

impl Default for Ms2ToMs1LinkerOpts {
    fn default() -> Self {
        Self {
            min_rt_jaccard: 0.1,
            max_rt_apex_sec: Some(8.0),
            max_im_apex_scans: Some(5.0),
            require_im_overlap: true,
            rt_guard_sec: 0.0,
            mz_ppm: 20.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinkCandidate {
    pub ms1_idx: usize,
    pub ms2_idx: usize,
    pub score: f32,
    pub group: u32,
}

#[derive(Clone, Debug)]
pub struct SelectionCfg {
    /// Discard candidates with score < min_score
    pub min_score: f32,
    /// Keep at most this many candidates per MS2 before greedy selection
    pub top_k_per_ms2: usize,
    /// Optionally keep at most this many candidates per MS1 (after per-MS2 cull)
    pub top_k_per_ms1: Option<usize>,
    /// If true, enforce 1:1 (each MS1 used at most once) across accepted links
    pub cardinality_one_to_one: bool,
}

impl Default for SelectionCfg {
    fn default() -> Self {
        Self {
            min_score: 0.0,
            top_k_per_ms2: 32,
            top_k_per_ms1: Some(512),
            cardinality_one_to_one: false,
        }
    }
}

#[inline]
fn jaccard_time(a_lo: f64, a_hi: f64, b_lo: f64, b_hi: f64) -> f32 {
    if !a_lo.is_finite() || !a_hi.is_finite() || !b_lo.is_finite() || !b_hi.is_finite() {
        return 0.0;
    }
    if a_hi < b_lo || b_hi < a_lo {
        return 0.0;
    }
    let inter = (a_hi.min(b_hi) - a_lo.max(b_lo)).max(0.0);
    let union = (a_hi.max(b_hi) - a_lo.min(b_lo)).max(0.0);
    if union <= 0.0 { 0.0 } else { (inter / union) as f32 }
}

#[inline]
fn overlaps_scan_usize(a: (usize, usize), b: (u32, u32)) -> bool {
    let (al, ah) = (a.0 as u32, a.1 as u32);
    !(ah < b.0 || b.1 < al)
}

#[inline]
fn any_prog_im_overlap(want_im: (usize, usize), ranges: &[(u32, u32)]) -> bool {
    ranges.iter().any(|&r| overlaps_scan_usize(want_im, r))
}

/// Build raw link candidates **without** a precursor m/z index.
/// m/z gating uses MS1 cluster window (padded by `mz_ppm`) vs DIA isolation windows.
pub fn link_ms2_to_ms1_candidates_noindex(
    ds: &TimsDatasetDIA,
    ms1: &[ClusterResult1D],   // precursors (ms_level==1)
    ms2: &[ClusterResult1D],   // fragments  (ms_level==2, window_group=Some)
    opts: &Ms2ToMs1LinkerOpts,
) -> Vec<LinkCandidate> {
    // frame_id -> time (seconds)
    let frame_time = &ds.dia_index.frame_time;

    // Precompute per-MS1 absolute time bounds (seconds) for soft gate & Jaccard
    let mut ms1_time_bounds: Vec<(f64, f64)> = Vec::with_capacity(ms1.len());
    for c in ms1 {
        let mut t_lo = f64::INFINITY;
        let mut t_hi = f64::NEG_INFINITY;
        for &fid in &c.frame_ids_used {
            if let Some(&t) = frame_time.get(&fid) {
                if t < t_lo { t_lo = t; }
                if t > t_hi { t_hi = t; }
            }
        }
        ms1_time_bounds.push((t_lo, t_hi));
    }

    // Group MS2 indices by DIA window group
    let mut by_group: HashMap<u32, Vec<usize>> = HashMap::new();
    for (j, c2) in ms2.iter().enumerate() {
        if let Some(g) = c2.window_group {
            by_group.entry(g).or_default().push(j);
        }
    }

    // Parallel per-group; flatten to Vec<LinkCandidate>
    by_group
        .into_par_iter()
        .flat_map(|(g, js)| {
            let prog = ds.program_for_group(g);
            if prog.mz_windows.is_empty() {
                return Vec::<LinkCandidate>::new();
            }

            // Precompute MS1 candidates passing m/z isolation test (window-vs-window with ppm pad)
            let ms1_cand: Vec<usize> = (0..ms1.len())
                .filter(|&i| {
                    let c = &ms1[i];
                    // Skip non-finite fits early
                    if !(c.mz_fit.mu.is_finite() && c.rt_fit.mu.is_finite() && c.im_fit.mu.is_finite()) {
                        return false;
                    }
                    // Pad the MS1 *window* (not just μ) by ppm
                    let mut lo = c.mz_window.0 as f64;
                    let mut hi = c.mz_window.1 as f64;
                    if opts.mz_ppm > 0.0 && lo.is_finite() && hi.is_finite() && hi > lo {
                        // proportional pad on both edges
                        lo -= lo.abs() * (opts.mz_ppm as f64) * 1e-6;
                        hi += hi.abs() * (opts.mz_ppm as f64) * 1e-6;
                    }
                    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
                        return false;
                    }
                    prog.mz_windows.iter().any(|&(wlo, whi)| {
                        let (wlo, whi) = (wlo as f64, whi as f64);
                        !(hi < wlo || whi < lo)
                    })
                })
                .collect();

            let mut local: Vec<LinkCandidate> = Vec::new();

            for &j in &js {
                let c2 = &ms2[j];

                // Guard: finite fits
                if !(c2.rt_fit.mu.is_finite() && c2.im_fit.mu.is_finite()) {
                    continue;
                }

                // Absolute time bounds for this MS2 (padded)
                let mut t2_lo = f64::INFINITY;
                let mut t2_hi = f64::NEG_INFINITY;
                for &fid in &c2.frame_ids_used {
                    if let Some(&t) = frame_time.get(&fid) {
                        if t < t2_lo { t2_lo = t; }
                        if t > t2_hi { t2_hi = t; }
                    }
                }
                if t2_lo.is_finite() { t2_lo -= opts.rt_guard_sec; }
                if t2_hi.is_finite() { t2_hi += opts.rt_guard_sec; }

                for &i in &ms1_cand {
                    let c1 = &ms1[i];

                    // IM strictness: cluster overlap AND program scan ranges
                    if opts.require_im_overlap {
                        let im_ok_cluster = !(c1.im_window.1 < c2.im_window.0 || c2.im_window.1 < c1.im_window.0);
                        let im_ok_program = prog.scan_ranges.is_empty()
                            || any_prog_im_overlap(c1.im_window, &prog.scan_ranges);
                        if !(im_ok_cluster && im_ok_program) {
                            continue;
                        }
                    }

                    // Optional absolute-time co-elution gate before Jaccard
                    let (t1_lo, t1_hi) = ms1_time_bounds[i];
                    if t2_lo.is_finite() && t2_hi.is_finite() && t1_lo.is_finite() && t1_hi.is_finite() {
                        if t1_hi < t2_lo || t2_hi < t1_lo {
                            continue;
                        }
                    }

                    // RT Jaccard in time space
                    let jacc = jaccard_time(t1_lo, t1_hi, t2_lo, t2_hi);
                    if jacc < opts.min_rt_jaccard {
                        continue;
                    }

                    // Apex deltas (seconds, scans) with guards
                    let d_rt = (c1.rt_fit.mu - c2.rt_fit.mu).abs();
                    if let Some(max) = opts.max_rt_apex_sec {
                        if !d_rt.is_finite() || d_rt > max {
                            continue;
                        }
                    }

                    let d_im = (c1.im_fit.mu - c2.im_fit.mu).abs();
                    if let Some(max) = opts.max_im_apex_scans {
                        if !d_im.is_finite() || d_im > max {
                            continue;
                        }
                    }

                    // Score: Jaccard * decreasing functions of apex deltas
                    let score = jacc
                        * (1.0 / (1.0 + d_rt))
                        * (if let Some(max) = opts.max_im_apex_scans {
                        1.0 / (1.0 + d_im / (max + 1e-3))
                    } else {
                        1.0
                    });

                    if score.is_finite() {
                        local.push(LinkCandidate { ms1_idx: i, ms2_idx: j, score, group: g });
                    }
                }
            }

            // best-first (makes later truncations cheaper)
            local.sort_by(|a, b| b.score.total_cmp(&a.score));
            local
        })
        .collect()
}

/// Greedy selection & grouping (MS2-centric), returning (precursor, [fragments]) blocks.
/// Assumes `candidates` are already group-compatible and pre-gated (m/z/IM/RT).
pub fn build_precursor_fragment_annotation_ms2centric(
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    candidates: &[LinkCandidate],
    cfg: &SelectionCfg,
) -> Vec<(ClusterResult1D, Vec<ClusterResult1D>)> {
    if candidates.is_empty() || ms1.is_empty() || ms2.is_empty() {
        return Vec::new();
    }

    // Group candidates by DIA window group
    let mut by_group: HashMap<u32, Vec<LinkCandidate>> = HashMap::new();
    for c in candidates {
        if c.score >= cfg.min_score {
            by_group.entry(c.group).or_default().push(c.clone());
        }
    }

    // Per-group selection in parallel
    let selected: Vec<LinkCandidate> = by_group
        .into_par_iter()
        .map(|(_g, mut edges)| {
            if edges.is_empty() {
                return Vec::<LinkCandidate>::new();
            }

            // 1) Per-MS2 top-K cull
            let mut by_ms2: HashMap<usize, Vec<LinkCandidate>> = HashMap::new();
            for e in edges.drain(..) {
                if e.ms2_idx < ms2.len() && e.ms1_idx < ms1.len() {
                    by_ms2.entry(e.ms2_idx).or_default().push(e);
                }
            }
            for v in by_ms2.values_mut() {
                v.sort_by(|a, b| b.score.total_cmp(&a.score));
                if v.len() > cfg.top_k_per_ms2 {
                    v.truncate(cfg.top_k_per_ms2);
                }
            }
            let mut culled: Vec<LinkCandidate> = by_ms2.into_values().flatten().collect();

            // 2) Optional per-MS1 top-K cull
            if let Some(k) = cfg.top_k_per_ms1 {
                let mut by_ms1: HashMap<usize, Vec<LinkCandidate>> = HashMap::new();
                for e in culled.drain(..) {
                    by_ms1.entry(e.ms1_idx).or_default().push(e);
                }
                for v in by_ms1.values_mut() {
                    v.sort_by(|a, b| b.score.total_cmp(&a.score));
                    if v.len() > k {
                        v.truncate(k);
                    }
                }
                culled = by_ms1.into_values().flatten().collect();
            }

            // 3) Greedy accept best-first with cardinality constraints
            culled.sort_by(|a, b| b.score.total_cmp(&a.score));
            let mut used_ms2: HashSet<usize> = HashSet::new();
            let mut used_ms1: HashSet<usize> = HashSet::new();
            let mut out = Vec::new();

            for e in culled.into_iter() {
                if e.ms2_idx >= ms2.len() || e.ms1_idx >= ms1.len() {
                    continue;
                }
                if used_ms2.contains(&e.ms2_idx) {
                    continue;
                }
                if cfg.cardinality_one_to_one && used_ms1.contains(&e.ms1_idx) {
                    continue;
                }
                out.push(e.clone());
                used_ms2.insert(e.ms2_idx);
                if cfg.cardinality_one_to_one {
                    used_ms1.insert(e.ms1_idx);
                }
            }
            out
        })
        .flatten()
        .collect();

    if selected.is_empty() {
        return Vec::new();
    }

    // Group accepted links by MS1; sort fragment lists by score desc
    let mut score_lookup: HashMap<(usize, usize), f32> = HashMap::new();
    for s in &selected {
        score_lookup.insert((s.ms1_idx, s.ms2_idx), s.score);
    }

    let mut by_ms1: HashMap<usize, Vec<usize>> = HashMap::new();
    for s in selected {
        by_ms1.entry(s.ms1_idx).or_default().push(s.ms2_idx);
    }

    let mut out: Vec<(ClusterResult1D, Vec<ClusterResult1D>)> = Vec::with_capacity(by_ms1.len());
    for (i_ms1, mut frag_idxs) in by_ms1.into_iter() {
        frag_idxs.sort_by(|&a, &b| {
            let sa = *score_lookup.get(&(i_ms1, a)).unwrap_or(&0.0);
            let sb = *score_lookup.get(&(i_ms1, b)).unwrap_or(&0.0);
            sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
        });
        let frags = frag_idxs
            .into_iter()
            .filter_map(|j| ms2.get(j).cloned())
            .collect::<Vec<_>>();
        if let Some(prec) = ms1.get(i_ms1).cloned() {
            out.push((prec, frags));
        }
    }

    // Order precursor blocks by precursor strength (raw_sum desc)
    out.sort_by(|a, b| b.0.raw_sum.total_cmp(&a.0.raw_sum));
    out
}

/// Convenience wrapper: full MS2→MS1 linking without an m/z index.
pub fn link_ms2_to_ms1_noindex(
    ds: &TimsDatasetDIA,
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    linker_opts: Ms2ToMs1LinkerOpts,
    sel_cfg: SelectionCfg,
) -> Vec<(ClusterResult1D, Vec<ClusterResult1D>)> {
    let cands = link_ms2_to_ms1_candidates_noindex(ds, ms1, ms2, &linker_opts);
    build_precursor_fragment_annotation_ms2centric(ms1, ms2, &cands, &sel_cfg)
}