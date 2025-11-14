use std::collections::HashMap;
use std::sync::Arc;
use rayon::prelude::*;
use crate::cluster::cluster::ClusterResult1D;
use crate::data::dia::TimsDatasetDIA;

/// Jaccard overlap in **absolute seconds** for two closed intervals.
#[inline]
pub fn jaccard_time(a_lo: f64, a_hi: f64, b_lo: f64, b_hi: f64) -> f32 {
    if !a_lo.is_finite() || !a_hi.is_finite() || !b_lo.is_finite() || !b_hi.is_finite() {
        return 0.0;
    }
    if a_hi < b_lo || b_hi < a_lo { return 0.0; }
    let inter = (a_hi.min(b_hi) - a_lo.max(b_lo)).max(0.0);
    let union = (a_hi.max(b_hi) - a_lo.min(b_lo)).max(0.0);
    if union <= 0.0 { 0.0 } else { (inter / union) as f32 }
}

#[inline]
fn overlap_f32(a: (f32, f32), b: (f32, f32)) -> bool {
    let (a0, a1) = a; let (b0, b1) = b;
    a0.is_finite() && a1.is_finite() && b0.is_finite() && b1.is_finite() && a1 >= b0 && b1 >= a0
}

#[inline]
fn overlap_u32(a: (u32, u32), b: (u32, u32)) -> bool {
    a.1 >= b.0 && b.1 >= a.0
}

/// Coarse RT bucketing over absolute time in seconds (closed intervals).
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
                (((x - lo) * inv_bw).floor() as isize).clamp(0, (n as isize) - 1) as usize
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
                (((x - self.lo) * self.inv_bw).floor() as isize).clamp(0, (n as isize) - 1) as usize
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

/// Options for the simple candidate enumeration.
/// Rule = RT overlap (seconds) AND group eligibility (mz ∩ isolation AND scans ∩ program).
// --- add to CandidateOpts ---
#[derive(Clone, Debug)]
pub struct CandidateOpts {
    /// Require at least this Jaccard in RT (set 0.0 for “any overlap”).
    pub min_rt_jaccard: f32,
    /// Guard pad on MS2 time bounds (applied symmetrically), in seconds.
    pub ms2_rt_guard_sec: f64,
    /// RT bucket width (seconds).
    pub rt_bucket_width: f64,
    /// Pre-filters to drop weird clusters.
    pub max_ms1_rt_span_sec: Option<f64>,
    pub max_ms2_rt_span_sec: Option<f64>,
    pub min_raw_sum: f32,

    // ---- NEW tight guards ----
    /// Maximum allowed |rt_apex_MS1 - rt_apex_MS2| in seconds (None disables).
    pub max_rt_apex_delta_sec: Option<f32>,
    /// Maximum allowed |im_apex_MS1 - im_apex_MS2| in global scans (None disables).
    pub max_scan_apex_delta: Option<usize>,
    /// Require at least this many scan-overlap between MS1 and MS2 IM windows.
    pub min_im_overlap_scans: usize,
}

impl Default for CandidateOpts {
    fn default() -> Self {
        Self {
            min_rt_jaccard: 0.0,
            ms2_rt_guard_sec: 0.0,
            rt_bucket_width: 1.0,
            max_ms1_rt_span_sec: Some(60.0),
            max_ms2_rt_span_sec: Some(60.0),
            min_raw_sum: 1.0,

            // sensible, but conservative defaults
            max_rt_apex_delta_sec: Some(2.0),   // tighten to taste
            max_scan_apex_delta:   Some(6),     // ~ few IM scans
            min_im_overlap_scans:  1,           // at least touch
        }
    }
}

/// A built search index over MS1 with per-group eligibility masks.
#[derive(Clone, Debug)]
pub struct PrecursorSearchIndex {
    ms1_time_bounds: Vec<(f64, f64)>,
    ms1_keep: Vec<bool>,
    rt_buckets: RtBuckets,
    /// group -> mask[i] (true if MS1[i] is eligible for that group by (mz ∩ isolation) AND (scan ∩ ranges))
    ms1_group_ok: HashMap<u32, Vec<bool>>,
    /// frame_id -> time (seconds)
    frame_time: Arc<HashMap<u32, f64>>,
}

impl PrecursorSearchIndex {
    /// Build once per dataset / MS1 set.
    pub fn build(ds: &TimsDatasetDIA, ms1: &[ClusterResult1D], opts: &CandidateOpts) -> Self {
        let frame_time = Arc::new(ds.dia_index.frame_time.clone());

        // 1) Absolute MS1 time bounds
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

        // 2) Keep mask for MS1
        let ms1_keep: Vec<bool> = ms1.par_iter().enumerate().map(|(i, c)| {
            if c.ms_level != 1 { return false; }
            if c.raw_sum < opts.min_raw_sum { return false; }
            let (t0, t1) = ms1_time_bounds[i];
            if !(t0.is_finite() && t1.is_finite()) || t1 <= t0 { return false; }
            if let Some(max_rt) = opts.max_ms1_rt_span_sec { if (t1 - t0) > max_rt { return false; } }
            true
        }).collect();

        // 3) RT buckets across MS1
        let (mut rt_min, mut rt_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for &(a, b) in &ms1_time_bounds {
            if a.is_finite() { rt_min = rt_min.min(a); }
            if b.is_finite() { rt_max = rt_max.max(b); }
        }
        if !rt_min.is_finite() || !rt_max.is_finite() || rt_max <= rt_min {
            rt_min = 0.0; rt_max = 1.0;
        }
        let rt_buckets = RtBuckets::build(rt_min, rt_max, opts.rt_bucket_width, &ms1_time_bounds, Some(&ms1_keep));

        // 4) Per-group eligibility masks (mz ∩ isolation AND scans ∩ program), independent of RT.
        let ms1_group_ok: HashMap<u32, Vec<bool>> = ds.dia_index
            .group_to_isolation
            .par_iter()
            .map(|(&g, mz_rows)| {
                let scans = ds.dia_index.group_to_scan_ranges.get(&g).cloned().unwrap_or_default();
                let mz_rows_f32: Vec<(f32,f32)> = mz_rows.iter().map(|&(a,b)| (a as f32, b as f32)).collect();
                let scan_rows_u32: Vec<(u32,u32)> = scans.iter().copied().collect();

                if mz_rows_f32.is_empty() || scan_rows_u32.is_empty() {
                    return (g, vec![false; ms1.len()]);
                }

                let mask: Vec<bool> = ms1.par_iter().map(|c| {
                    if c.ms_level != 1 { return false; }
                    let mz_ok = mz_rows_f32.iter().any(|&w| overlap_f32(c.mz_window.unwrap(), w));
                    if !mz_ok { return false; }
                    let im_u32 = (c.im_window.0 as u32, c.im_window.1 as u32);
                    scan_rows_u32.iter().any(|&s| overlap_u32(im_u32, s))
                }).collect();

                (g, mask)
            })
            .collect();

        Self { ms1_time_bounds, ms1_keep, rt_buckets, ms1_group_ok, frame_time }
    }

    pub fn enumerate_pairs(
        &self,
        ms1: &[ClusterResult1D],
        ms2: &[ClusterResult1D],
        opts: &CandidateOpts,
    ) -> Vec<(usize, usize)> {
        // Precompute MS2 absolute time bounds
        let ms2_time_bounds: Vec<(f64, f64)> = ms2.par_iter().map(|c| {
            let mut t_lo = f64::INFINITY;
            let mut t_hi = f64::NEG_INFINITY;
            for &fid in &c.frame_ids_used {
                if let Some(&t) = self.frame_time.get(&fid) {
                    if t < t_lo { t_lo = t; }
                    if t > t_hi { t_hi = t; }
                }
            }
            (t_lo, t_hi)
        }).collect();
        // Share across threads safely
        let ms2_time_bounds = Arc::new(ms2_time_bounds);

        // Keep MS2s
        let ms2_keep: Vec<bool> = ms2.par_iter().enumerate().map(|(i, c)| {
            if c.ms_level != 2 { return false; }
            if c.window_group.is_none() { return false; }
            if c.raw_sum < opts.min_raw_sum { return false; }
            let (mut t0, mut t1) = ms2_time_bounds[i];
            if t0.is_finite() { t0 -= opts.ms2_rt_guard_sec; }
            if t1.is_finite() { t1 += opts.ms2_rt_guard_sec; }
            if !(t0.is_finite() && t1.is_finite() && t1 > t0) { return false; }
            if let Some(max_rt) = opts.max_ms2_rt_span_sec {
                if (t1 - t0) > max_rt { return false; }
            }
            true
        }).collect();

        // Group MS2 by window_group
        let mut by_group: HashMap<u32, Vec<usize>> = HashMap::new();
        for (j, c2) in ms2.iter().enumerate() {
            if !ms2_keep[j] { continue; }
            if let Some(g) = c2.window_group {
                by_group.entry(g).or_default().push(j);
            }
        }

        // Share index across groups/threads
        let idx_arc = Arc::new(self.clone());

        let mut out = by_group
            .into_par_iter()
            .flat_map(|(g, js)| {
                // Resolve the eligibility mask once per group, materialize it so we
                // don't return a reference to a temporary.
                let mask_vec: Vec<bool> = idx_arc
                    .ms1_group_ok
                    .get(&g)
                    .cloned()
                    .unwrap_or_else(|| vec![false; ms1.len()]);
                let mask_arc = Arc::new(mask_vec);

                // Clone shared state for the per-MS2 loop
                let idx = idx_arc.clone();
                let tb = ms2_time_bounds.clone();

                js.into_par_iter().flat_map(move |j| {
                    let (mut t2_lo, mut t2_hi) = tb[j];
                    if t2_lo.is_finite() { t2_lo -= opts.ms2_rt_guard_sec; }
                    if t2_hi.is_finite() { t2_hi += opts.ms2_rt_guard_sec; }

                    // Time-prefilter MS1 via buckets
                    let mut hits = Vec::<usize>::new();
                    idx.rt_buckets.gather(t2_lo, t2_hi, &mut hits);
                    hits.sort_unstable();
                    hits.dedup();

                    // Use the materialized mask
                    let mask = mask_arc.clone();

                    let mut local = Vec::<(usize, usize)>::with_capacity(16);
                    for i in hits {
                        if !idx.ms1_keep[i] { continue; }
                        if !mask[i] { continue; } // eligible by group (mz & program scans)

                        let (t1_lo, t1_hi) = idx.ms1_time_bounds[i];
                        if !(t1_lo.is_finite() && t1_hi.is_finite()) { continue; }

                        // Any RT overlap; optionally enforce Jaccard threshold
                        if t1_hi < t2_lo || t2_hi < t1_lo { continue; }
                        if opts.min_rt_jaccard > 0.0 {
                            let jacc = jaccard_time(t1_lo, t1_hi, t2_lo, t2_hi);
                            if jacc < opts.min_rt_jaccard { continue; }
                        }

                        // ---- NEW: IM window overlap in global scan axis ----
                        let im1 = ms1[i].im_window;
                        let im2 = ms2[j].im_window;
                        let im_overlap = {
                            let lo = im1.0.max(im2.0);
                            let hi = im1.1.min(im2.1);
                            hi.saturating_sub(lo).saturating_add(1)
                        };
                        if im_overlap < opts.min_im_overlap_scans { continue; }

                        // ---- NEW: IM apex delta in scans ----
                        if let Some(max_d) = opts.max_scan_apex_delta {
                            // Fit μ is in scan units for IM (you stamp attach_axes=true in clustering)
                            let s1 = ms1[i].im_fit.mu;
                            let s2 = ms2[j].im_fit.mu;
                            if s1.is_finite() && s2.is_finite() {
                                let d = (s1 - s2).abs() as f32;
                                if d > (max_d as f32) { continue; }
                            } else {
                                // if either apex missing/degenerate, be strict and drop
                                continue;
                            }
                        }

                        // ---- NEW: RT apex delta in seconds ----
                        if let Some(max_dt) = opts.max_rt_apex_delta_sec {
                            let r1 = ms1[i].rt_fit.mu; // seconds when attach_axes=true
                            let r2 = ms2[j].rt_fit.mu;
                            if r1.is_finite() && r2.is_finite() {
                                if (r1 - r2).abs() > max_dt { continue; }
                            } else {
                                continue;
                            }
                        }

                        local.push((j, i));
                    }

                    // Return as a parallel iterator to Rayon
                    local.into_par_iter()
                })
            })
            .collect::<Vec<(usize, usize)>>();

        // --- HARD DE-DUP step ---
        out.sort_unstable();
        out.dedup();

        out
    }
}

/// Convenience: build, enumerate, done.
pub fn enumerate_ms2_ms1_pairs_simple(
    ds: &TimsDatasetDIA,
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    opts: &CandidateOpts,
) -> Vec<(usize, usize)> {
    let idx = PrecursorSearchIndex::build(ds, ms1, opts);
    idx.enumerate_pairs(ms1, ms2, opts)
}