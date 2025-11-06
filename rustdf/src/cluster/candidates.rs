use crate::cluster::cluster::ClusterResult1D;
use crate::data::dia::TimsDatasetDIA;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Time-overlap Jaccard in absolute seconds.
#[inline]
fn jaccard_time(a_lo: f64, a_hi: f64, b_lo: f64, b_hi: f64) -> f32 {
    if !a_lo.is_finite() || !a_hi.is_finite() || !b_lo.is_finite() || !b_hi.is_finite() {
        return 0.0;
    }
    if a_hi < b_lo || b_hi < a_lo { return 0.0; }
    let inter = (a_hi.min(b_hi) - a_lo.max(b_lo)).max(0.0);
    let union = (a_hi.max(b_hi) - a_lo.min(b_lo)).max(0.0);
    if union <= 0.0 { 0.0 } else { (inter / union) as f32 }
}

/// Coarse RT bucketing over absolute time in seconds.
#[derive(Clone, Debug)]
pub struct RtBuckets {
    lo: f64,
    inv_bw: f64,
    buckets: Vec<Vec<usize>>,
}
impl RtBuckets {
    pub fn build(
        global_lo: f64,
        global_hi: f64,
        bucket_width: f64,
        ms1_time_bounds: &[(f64, f64)],
        ms1_keep: Option<&[bool]>,
    ) -> Self {
        let bw = bucket_width.max(0.5);
        let lo = global_lo.floor();
        let hi = global_hi.ceil().max(lo + bw);
        let n = (((hi - lo) / bw).ceil() as usize).max(1);
        let inv_bw = 1.0 / bw;
        let mut buckets = vec![Vec::<usize>::new(); n];

        let clamp = |x: f64| -> usize {
            if x <= lo { 0 } else {
                (((x - lo) * inv_bw).floor() as isize)
                    .clamp(0, (n as isize) - 1) as usize
            }
        };

        for (i, &(t0, t1)) in ms1_time_bounds.iter().enumerate() {
            if let Some(keep) = ms1_keep {
                if !keep[i] { continue; }
            }
            if !(t0.is_finite() && t1.is_finite()) || t1 <= t0 { continue; }
            let b0 = clamp(t0);
            let b1 = clamp(t1);
            for b in b0..=b1 { buckets[b].push(i); }
        }
        Self { lo, inv_bw, buckets }
    }

    #[inline]
    fn range(&self, t0: f64, t1: f64) -> (usize, usize) {
        let n = self.buckets.len();
        let clamp = |x: f64| -> usize {
            if x <= self.lo { 0 } else {
                (((x - self.lo) * self.inv_bw).floor() as isize)
                    .clamp(0, (n as isize) - 1) as usize
            }
        };
        let a = clamp(t0.min(t1));
        let b = clamp(t0.max(t1));
        (a.min(b), a.max(b))
    }

    /// Append MS1 indices that touch [t0, t1] (not deduped).
    #[inline]
    pub fn gather(&self, t0: f64, t1: f64, out: &mut Vec<usize>) {
        let (b0, b1) = self.range(t0, t1);
        for b in b0..=b1 { out.extend_from_slice(&self.buckets[b]); }
    }
}

/// Light cache of DIA program constraints we care about for gating.
#[derive(Clone, Debug)]
struct ProgramCache {
    /// group -> merged active IM scan ranges (inclusive)
    group_to_scans: HashMap<u32, Vec<(u32, u32)>>,
}
impl ProgramCache {
    fn build(ds: &TimsDatasetDIA) -> Self {
        let mut m = HashMap::new();
        for g in ds.dia_window_groups() {
            let scans = if let Some(u) = ds.scan_unions_for_window_group_core(g) {
                u.into_iter().map(|(l, r)| (l as u32, r as u32)).collect()
            } else {
                ds.program_for_group(g).scan_ranges
            };
            m.insert(g, scans);
        }
        Self { group_to_scans: m }
    }

    #[inline]
    fn im_ok_for_group(&self, g: u32, im_win: (usize, usize)) -> bool {
        let scans = match self.group_to_scans.get(&g) {
            Some(v) => v,
            None => return true, // no info → don't block
        };
        // overlap (inclusive) between cluster IM window and any active DIA scan range
        let (l0, r0) = (im_win.0 as u32, im_win.1 as u32);
        for &(l, r) in scans {
            if !(r0 < l || r < l0) { return true; }
        }
        false
    }
}

/// Options controlling candidate enumeration.
#[derive(Clone, Debug)]
pub struct CandidateOpts {
    /// Minimum RT Jaccard overlap to accept candidate
    pub min_rt_jaccard: f32,     // e.g., 0.05–0.2
    /// Pad on absolute RT bounds, seconds (applied to MS2 interval)
    pub rt_guard_sec: f64,       // e.g., 0.0–2.0
    /// Optional absolute max |Δ apex RT|, seconds
    pub max_rt_apex_sec: Option<f32>,
    /// Whether to enforce IM scan overlap with DIA program ranges
    pub require_im_overlap: bool,
    /// Width of RT buckets (seconds)
    pub rt_bucket_width: f64,    // e.g., 1.0
    /// Drop very long or very wide clusters up front
    pub max_ms1_rt_span_sec: Option<f64>,
    pub max_ms2_rt_span_sec: Option<f64>,
    pub max_im_span_scans: Option<usize>,
    pub min_raw_sum: f32,
}
impl Default for CandidateOpts {
    fn default() -> Self {
        Self {
            min_rt_jaccard: 0.10,
            rt_guard_sec: 0.0,
            max_rt_apex_sec: Some(8.0),
            require_im_overlap: true,
            rt_bucket_width: 1.0,
            max_ms1_rt_span_sec: Some(60.0),
            max_ms2_rt_span_sec: Some(60.0),
            max_im_span_scans: Some(80),
            min_raw_sum: 1.0,
        }
    }
}

/// A built search index over MS1 to accelerate MS2→MS1 candidate enumeration.
#[derive(Clone, Debug)]
pub struct PrecursorSearchIndex {
    ms1_time_bounds: Vec<(f64, f64)>,
    ms1_keep: Vec<bool>,
    rt_buckets: RtBuckets,
    prog: ProgramCache,
    frame_time: Arc<HashMap<u32, f64>>,
}
impl PrecursorSearchIndex {
    /// Build once per file / MS1 set.
    pub fn build(ds: &TimsDatasetDIA, ms1: &[ClusterResult1D], opts: &CandidateOpts) -> Self {
        let frame_time = Arc::new(ds.dia_index.frame_time.clone());

        // Absolute time bounds for MS1
        let ms1_time_bounds: Vec<(f64, f64)> = ms1
            .par_iter()
            .map(|c| {
                let mut t_lo = f64::INFINITY;
                let mut t_hi = f64::NEG_INFINITY;
                for &fid in &c.frame_ids_used {
                    if let Some(&t) = frame_time.get(&fid) {
                        if t < t_lo { t_lo = t; }
                        if t > t_hi { t_hi = t; }
                    }
                }
                (t_lo, t_hi)
            })
            .collect();

        // Keep mask for MS1 (span + intensity)
        let ms1_keep: Vec<bool> = ms1
            .par_iter()
            .enumerate()
            .map(|(i, c)| {
                if c.ms_level != 1 { return false; }
                if c.raw_sum < opts.min_raw_sum { return false; }
                let im_span = if c.im_window.1 >= c.im_window.0 {
                    (c.im_window.1 as isize - c.im_window.0 as isize + 1).max(0) as usize
                } else { 0 };
                if let Some(max_im) = opts.max_im_span_scans { if im_span > max_im { return false; } }
                let (t_lo, t_hi) = ms1_time_bounds[i];
                if !(t_lo.is_finite() && t_hi.is_finite()) { return false; }
                let rt_span = (t_hi - t_lo).max(0.0);
                if let Some(max_rt) = opts.max_ms1_rt_span_sec { if rt_span > max_rt { return false; } }
                true
            })
            .collect();

        // Global RT span across MS1 (fallback to some sane window if empty)
        let mut rt_min = f64::INFINITY;
        let mut rt_max = f64::NEG_INFINITY;
        for &(a, b) in &ms1_time_bounds {
            if a.is_finite() { rt_min = rt_min.min(a); }
            if b.is_finite() { rt_max = rt_max.max(b); }
        }
        if !rt_min.is_finite() || !rt_max.is_finite() || rt_max <= rt_min {
            // Fallback to [0,1] to avoid panics; buckets will just be trivial.
            rt_min = 0.0; rt_max = 1.0;
        }

        // RT buckets (respect keep mask)
        let rt_buckets = RtBuckets::build(rt_min, rt_max, opts.rt_bucket_width, &ms1_time_bounds, Some(&ms1_keep));

        // DIA program cache
        let prog = ProgramCache::build(ds);

        Self {
            ms1_time_bounds,
            ms1_keep,
            rt_buckets,
            prog,
            frame_time,
        }
    }

    /// Enumerate (ms2_idx, ms1_idx) candidates:
    /// 1) time-overlap prefilter (RT buckets) + Jaccard threshold
    /// 2) optional apex ΔRT cut
    /// 3) optional IM-scan overlap with DIA program of this MS2's group
    pub fn enumerate_pairs(
        &self,
        ms1: &[ClusterResult1D],
        ms2: &[ClusterResult1D],
        opts: &CandidateOpts,
    ) -> Vec<(usize, usize)> {
        // Precompute absolute MS2 time bounds (+ guard) and basic keep
        let ms2_time_bounds: Vec<(f64, f64)> = ms2
            .par_iter()
            .map(|c| {
                let mut t_lo = f64::INFINITY;
                let mut t_hi = f64::NEG_INFINITY;
                for &fid in &c.frame_ids_used {
                    if let Some(&t) = self.frame_time.get(&fid) {
                        if t < t_lo { t_lo = t; }
                        if t > t_hi { t_hi = t; }
                    }
                }
                (t_lo, t_hi)
            })
            .collect();

        let ms2_time_bounds = Arc::new(ms2_time_bounds);

        let ms2_keep: Vec<bool> = ms2
            .par_iter()
            .enumerate()
            .map(|(i, c)| {
                if c.ms_level != 2 { return false; }
                if c.window_group.is_none() { return false; }
                if c.raw_sum < opts.min_raw_sum { return false; }
                let im_span = if c.im_window.1 >= c.im_window.0 {
                    (c.im_window.1 as isize - c.im_window.0 as isize + 1).max(0) as usize
                } else { 0 };
                if let Some(max_im) = opts.max_im_span_scans { if im_span > max_im { return false; } }
                let (mut t_lo, mut t_hi) = ms2_time_bounds[i];
                if t_lo.is_finite() { t_lo -= opts.rt_guard_sec; }
                if t_hi.is_finite() { t_hi += opts.rt_guard_sec; }
                if !(t_lo.is_finite() && t_hi.is_finite() && t_hi > t_lo) { return false; }
                let rt_span = (t_hi - t_lo).max(0.0);
                if let Some(max_rt) = opts.max_ms2_rt_span_sec { if rt_span > max_rt { return false; } }
                true
            })
            .collect();

        // Group MS2 indices by DIA group to apply program constraints cheaply
        let mut by_group: HashMap<u32, Vec<usize>> = HashMap::new();
        for (j, c2) in ms2.iter().enumerate() {
            if !ms2_keep[j] { continue; }
            if let Some(g) = c2.window_group { by_group.entry(g).or_default().push(j); }
        }

        let idx_arc = Arc::new(self.clone());

        by_group
            .into_par_iter()
            .flat_map(|(g, js)| {
                let idx = idx_arc.clone();
                let ms2_tb = ms2_time_bounds.clone(); // <-- clone Arc here
                js.into_par_iter().flat_map(move |j| {
                    let c2 = &ms2[j];
                    let (mut t2_lo, mut t2_hi) = ms2_tb[j]; // <-- use Arc-backed vec
                    if t2_lo.is_finite() { t2_lo -= opts.rt_guard_sec; }
                    if t2_hi.is_finite() { t2_hi += opts.rt_guard_sec; }

                    let mut hits = Vec::<usize>::new();
                    idx.rt_buckets.gather(t2_lo, t2_hi, &mut hits);
                    hits.sort_unstable();
                    hits.dedup();

                    let mut local = Vec::<(usize, usize)>::with_capacity(16);
                    for i in hits {
                        if !idx.ms1_keep[i] { continue; }
                        let (t1_lo, t1_hi) = idx.ms1_time_bounds[i];
                        if !(t1_lo.is_finite() && t1_hi.is_finite() && t1_hi > t1_lo) { continue; }

                        let jacc = jaccard_time(t1_lo, t1_hi, t2_lo, t2_hi);
                        if jacc < opts.min_rt_jaccard { continue; }

                        if let Some(max) = opts.max_rt_apex_sec {
                            let d_rt = (ms1[i].rt_fit.mu - c2.rt_fit.mu).abs();
                            if !d_rt.is_finite() || d_rt > max { continue; }
                        }

                        if opts.require_im_overlap {
                            if !idx.prog.im_ok_for_group(g, ms1[i].im_window) { continue; }
                            let im_ok_cluster = !(ms1[i].im_window.1 < c2.im_window.0
                                || c2.im_window.1 < ms1[i].im_window.0);
                            if !im_ok_cluster { continue; }
                        }

                        local.push((j, i));
                    }

                    local.into_par_iter()
                })
            })
            .collect::<Vec<(usize, usize)>>()
    }
}