// rustdf/src/cluster/cluster_link.rs

use crate::cluster::cluster::ClusterResult1D;
use crate::data::dia::TimsDatasetDIA;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

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

/// Coarse Da-binned index for MS1 m/z windows (built on the fly per call)
struct MzBins {
    bins: Vec<Vec<usize>>,
    lo: f64,
    hi: f64,
    w: f64,
}

impl MzBins {
    fn new(lo: f64, hi: f64, w: f64) -> Self {
        let span = (hi - lo).max(1.0);
        let n = ((span / w).ceil() as usize).max(1);
        Self { bins: vec![Vec::new(); n], lo, hi, w }
    }

    #[inline]
    fn clamp_bin(&self, x: f64) -> usize {
        if x <= self.lo { return 0; }
        if x >= self.hi { return self.bins.len().saturating_sub(1); }
        let idx = ((x - self.lo) / self.w).floor() as isize;
        idx.clamp(0, (self.bins.len() as isize) - 1) as usize
    }

    #[inline]
    fn bin_range(&self, a: f64, b: f64) -> (usize, usize) {
        let i0 = self.clamp_bin(a.min(b));
        let i1 = self.clamp_bin(a.max(b));
        (i0.min(i1), i0.max(i1))
    }

    fn push_window(&mut self, lo: f64, hi: f64, idx: usize) {
        if !(lo.is_finite() && hi.is_finite() && hi > lo) { return; }
        let (i0, i1) = self.bin_range(lo, hi);
        for i in i0..=i1 {
            self.bins[i].push(idx);
        }
    }

    fn union_for_window(&self, lo: f64, hi: f64, out: &mut Vec<usize>) {
        if !(lo.is_finite() && hi.is_finite() && hi > lo) { return; }
        let (i0, i1) = self.bin_range(lo, hi);
        for i in i0..=i1 {
            out.extend_from_slice(&self.bins[i]);
        }
    }
}

/// Each MS1 with [t_lo, t_hi] is inserted into every bucket touched by the interval.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct RtBuckets {
    lo: f64,
    bw: f64,
    inv_bw: f64,
    buckets: Vec<Vec<usize>>,
}

impl RtBuckets {
    fn new(global_lo: f64, global_hi: f64, bucket_width: f64, ms1_time_bounds: &[(f64,f64)]) -> Self {
        let lo = global_lo.floor();
        let hi = global_hi.ceil().max(lo + bucket_width);
        let bw = bucket_width.max(0.5); // don’t allow ultra tiny buckets
        let n = (((hi - lo) / bw).ceil() as usize).max(1);
        let mut buckets = vec![Vec::<usize>::new(); n];

        let inv_bw = 1.0 / bw;
        let clamp = |x: f64| -> usize {
            if x <= lo { 0 } else if x >= hi { n.saturating_sub(1) } else { ((x - lo) * inv_bw).floor() as usize }
        };

        for (i, &(t0, t1)) in ms1_time_bounds.iter().enumerate() {
            if !(t0.is_finite() && t1.is_finite()) { continue; }
            if t1 <= t0 { continue; }
            let b0 = clamp(t0);
            let b1 = clamp(t1);
            for b in b0..=b1 {
                buckets[b].push(i);
            }
        }

        Self { lo, bw, inv_bw, buckets }
    }

    #[inline]
    fn bucket_range(&self, t0: f64, t1: f64) -> (usize, usize) {
        let n = self.buckets.len();
        let clamp = |x: f64| -> usize {
            if x <= self.lo { 0 }
            else {
                let idx = ((x - self.lo) * self.inv_bw).floor() as isize;
                idx.clamp(0, (n as isize) - 1) as usize
            }
        };
        let a = clamp(t0.min(t1));
        let b = clamp(t0.max(t1));
        (a.min(b), a.max(b))
    }

    /// Collect MS1 indices that touch [t0, t1] (not deduped).
    #[inline]
    fn gather(&self, t0: f64, t1: f64, out: &mut Vec<usize>) {
        let (b0, b1) = self.bucket_range(t0, t1);
        for b in b0..=b1 {
            out.extend_from_slice(&self.buckets[b]);
        }
    }
}

pub fn link_ms2_to_ms1_candidates_noindex(
    ds: &TimsDatasetDIA,
    ms1: &[ClusterResult1D],   // precursors (ms_level==1)
    ms2: &[ClusterResult1D],   // fragments  (ms_level==2, window_group=Some)
    opts: &Ms2ToMs1LinkerOpts,
) -> Vec<LinkCandidate> {
    let frame_time = &ds.dia_index.frame_time;

    // ---- group MS2 by DIA window group ----
    let mut by_group: HashMap<u32, Vec<usize>> = HashMap::new();
    for (j, c2) in ms2.iter().enumerate() {
        if let Some(g) = c2.window_group {
            by_group.entry(g).or_default().push(j);
        }
    }

    // ---- PRECOMPUTES (parallel where useful) --------------------------------

    // MS1 and MS2 absolute time bounds (seconds)
    let ms1_time_bounds: Vec<(f64, f64)> = ms1.par_iter().map(|c| {
        let mut t_lo = f64::INFINITY;
        let mut t_hi = f64::NEG_INFINITY;
        for &fid in &c.frame_ids_used {
            if let Some(&t) = frame_time.get(&fid) {
                if t < t_lo { t_lo = t; }
                if t > t_hi { t_hi = t; }
            }
        }
        (t_lo, t_hi)
    }).collect();

    let ms2_time_bounds: Vec<(f64, f64)> = ms2.par_iter().map(|c2| {
        let mut t_lo = f64::INFINITY;
        let mut t_hi = f64::NEG_INFINITY;
        for &fid in &c2.frame_ids_used {
            if let Some(&t) = frame_time.get(&fid) {
                if t < t_lo { t_lo = t; }
                if t > t_hi { t_hi = t; }
            }
        }
        (t_lo, t_hi)
    }).collect();

    // Global RT span (for buckets)
    let (mut rt_min, mut rt_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(a,b) in &ms1_time_bounds {
        if a.is_finite() { rt_min = rt_min.min(a); }
        if b.is_finite() { rt_max = rt_max.max(b); }
    }
    for &(a,b) in &ms2_time_bounds {
        if a.is_finite() { rt_min = rt_min.min(a); }
        if b.is_finite() { rt_max = rt_max.max(b); }
    }
    if !rt_min.is_finite() || !rt_max.is_finite() || rt_max <= rt_min {
        return Vec::new();
    }

    // Build RT time buckets over MS1 once (choose 1.0 s buckets; adjust as needed)
    let rt_buckets = Arc::new(RtBuckets::new(rt_min, rt_max, 1.0, &ms1_time_bounds));

    // MS1 padded m/z windows (ppm on edges)
    let ppm = (opts.mz_ppm as f64) * 1e-6;
    let ms1_padded_mz: Vec<(f64, f64)> = ms1.par_iter().map(|c| {
        let (mut lo, mut hi) = (c.mz_window.0 as f64, c.mz_window.1 as f64);
        if lo.is_finite() && hi.is_finite() && hi > lo && ppm > 0.0 {
            lo -= lo.abs() * ppm;
            hi += hi.abs() * ppm;
        }
        (lo, hi)
    }).collect();

    // Cache group programs (mz windows + merged IM scan ranges)
    let mut prog_cache: HashMap<u32, (Vec<(f32,f32)>, Vec<(u32,u32)>)> = HashMap::new();
    for &g in by_group.keys() {
        let prog = ds.program_for_group(g);
        let scans = if let Some(u) = ds.scan_unions_for_window_group_core(g) {
            u.into_iter().map(|(a,b)| (a as u32, b as u32)).collect()
        } else {
            prog.scan_ranges.clone()
        };
        prog_cache.insert(g, (prog.mz_windows, scans));
    }

    // Coarse Da-binned index for MS1 windows (global)
    let global_lo = ds.global_meta_data.mz_acquisition_range_lower as f64;
    let global_hi = ds.global_meta_data.mz_acquisition_range_upper as f64;
    let mut bins_build = MzBins::new(global_lo.max(1.0), global_hi, 1.0);
    for (i, &(lo, hi)) in ms1_padded_mz.iter().enumerate() {
        bins_build.push_window(lo, hi, i);
    }
    let bins = Arc::new(bins_build);

    // Per-group MS1 candidates by m/z (via bin unions) + per-group boolean mask for O(1) membership
    let (_ms1_cand_by_group, is_ms1_cand_mask): (HashMap<u32, Vec<usize>>, HashMap<u32, Vec<bool>>) =
        by_group.par_iter().map(|(&g, _)| {
            let (mzwins, _scans) = &prog_cache[&g];
            let mut tmp = Vec::<usize>::new();
            tmp.reserve(ms1.len().min(8192));
            for &(wlo, whi) in mzwins {
                bins.union_for_window(wlo as f64, whi as f64, &mut tmp);
            }
            tmp.sort_unstable();
            tmp.dedup();

            let mut mask = vec![false; ms1.len()];
            for &i in &tmp { mask[i] = true; }
            ( (g, tmp), (g, mask) )
        }).unzip();

    // IM program feasibility mask per group (O(1) later)
    let im_prog_ok_by_group: HashMap<u32, Vec<bool>> = if !opts.require_im_overlap {
        HashMap::new()
    } else {
        by_group.par_iter().map(|(&g, _)| {
            let scans = &prog_cache[&g].1;
            if scans.is_empty() {
                return (g, vec![true; ms1.len()]);
            }
            let mask: Vec<bool> = ms1.par_iter().map(|c1| {
                scans.iter().any(|&(sl, sh)| !( (c1.im_window.1 as u32) < sl || sh < (c1.im_window.0 as u32) ))
            }).collect();
            (g, mask)
        }).collect()
    };

    // ---- PER-GROUP, THEN PER-FRAGMENT PARALLEL ------------------------------
    by_group.into_par_iter().flat_map(|(g, js)| {
        let _scans = &prog_cache[&g].1;
        let is_ms1_cand = &is_ms1_cand_mask[&g];
        let im_prog_mask_opt = im_prog_ok_by_group.get(&g);

        // parallel over fragment clusters in this group
        js.par_iter().flat_map(|&j| {
            let c2 = &ms2[j];
            if !(c2.rt_fit.mu.is_finite() && c2.im_fit.mu.is_finite()) {
                return Vec::<LinkCandidate>::new();
            }

            // MS2 absolute time with guard
            let (mut t2_lo, mut t2_hi) = ms2_time_bounds[j];
            if t2_lo.is_finite() { t2_lo -= opts.rt_guard_sec; }
            if t2_hi.is_finite() { t2_hi += opts.rt_guard_sec; }
            if !(t2_lo.is_finite() && t2_hi.is_finite() && t2_hi > t2_lo) {
                return Vec::<LinkCandidate>::new();
            }

            // Pull MS1 candidates that *time-overlap* the MS2 (via buckets)
            let mut ms1_time_hits = Vec::<usize>::new();
            rt_buckets.gather(t2_lo, t2_hi, &mut ms1_time_hits);
            ms1_time_hits.sort_unstable(); // dedup after gather
            ms1_time_hits.dedup();

            let mut local = Vec::<LinkCandidate>::new();
            local.reserve(16);

            for i in ms1_time_hits {
                // Intersect with group’s m/z-gated MS1 set in O(1)
                if !is_ms1_cand[i] { continue; }

                let c1 = &ms1[i];

                // Absolute co-elution was ensured by buckets, but guard against degenerate bounds:
                let (t1_lo, t1_hi) = ms1_time_bounds[i];
                if !(t1_lo.is_finite() && t1_hi.is_finite() && t1_hi > t1_lo) { continue; }

                // Jaccard in time space
                let jacc = jaccard_time(t1_lo, t1_hi, t2_lo, t2_hi);
                if jacc < opts.min_rt_jaccard { continue; }

                // IM checks
                if opts.require_im_overlap {
                    if let Some(mask) = im_prog_mask_opt {
                        if !mask[i] { continue; }
                    }
                    let im_ok_cluster = !(c1.im_window.1 < c2.im_window.0 || c2.im_window.1 < c1.im_window.0);
                    if !im_ok_cluster { continue; }
                }

                // Apex deltas + simple score
                let d_rt = (c1.rt_fit.mu - c2.rt_fit.mu).abs();
                if let Some(max) = opts.max_rt_apex_sec {
                    if !d_rt.is_finite() || d_rt > max { continue; }
                }

                let d_im = (c1.im_fit.mu - c2.im_fit.mu).abs();
                if let Some(max) = opts.max_im_apex_scans {
                    if !d_im.is_finite() || d_im > max { continue; }
                }

                let score = jacc
                    * (1.0 / (1.0 + d_rt))
                    * (if let Some(max) = opts.max_im_apex_scans {
                    1.0 / (1.0 + d_im / (max + 1e-3))
                } else { 1.0 });

                if score.is_finite() {
                    local.push(LinkCandidate { ms1_idx: i, ms2_idx: j, score, group: g });
                }
            }

            // Sort best-first (cheap truncation later)
            local.sort_by(|a,b| b.score.total_cmp(&a.score));
            local
        }).collect::<Vec<_>>() // end per-fragment parallel
    }).collect() // end per-group parallel
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