use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;
pub(crate) use crate::cluster::candidates::{CandidateOpts, PairFeatures, ScoreOpts};
use crate::cluster::cluster::ClusterResult1D;
use crate::cluster::feature::SimpleFeature;
use crate::cluster::pseudo::cluster_mz_mu;
use crate::data::dia::{DiaIndex, TimsDatasetDIA};

#[derive(Clone, Copy, Debug)]
pub enum MatchScoreMode {
    /// Geometric / “geom” style: uses PairFeatures + ScoreOpts
    Geom,
    /// XIC correlation style: uses XicScoreOpts
    Xic,
}

#[derive(Clone, Debug)]
pub struct XicDetails {
    /// RT XIC similarity in [0,1], if used.
    pub s_rt: Option<f32>,
    /// IM XIC similarity in [0,1], if used.
    pub s_im: Option<f32>,
    /// Intensity-ratio consistency term in (0,1], if used.
    pub s_intensity: Option<f32>,

    /// Raw RT Pearson correlation in [-1,1], if available.
    pub r_rt: Option<f32>,
    /// Raw IM Pearson correlation in [-1,1], if available.
    pub r_im: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct ScoredHit {
    /// Index into the MS2 slice (typically `self.ms2` in FragmentIndex).
    pub frag_idx: usize,
    /// Final scalar score (geom or XIC, depending on mode).
    pub score: f32,
    /// Geometric feature bundle (only set in `Geom` mode).
    pub geom: Option<PairFeatures>,
    /// XIC scoring details (only set in `Xic` mode).
    pub xic: Option<XicDetails>,
}

#[derive(Clone, Copy)]
pub enum PrecursorLike<'a> {
    Cluster(&'a ClusterResult1D),
    Feature(&'a SimpleFeature),
}

#[derive(Clone, Debug)]
pub struct XicScoreOpts {
    /// Weight for RT-XIC correlation term.
    pub w_rt: f32,
    /// Weight for IM-XIC correlation term.
    pub w_im: f32,
    /// Weight for intensity-ratio consistency term.
    pub w_intensity: f32,

    /// Pseudo-temperature for the log-intensity ratio penalty.
    /// Larger = more tolerant to MS1/MS2 intensity mismatch.
    pub intensity_tau: f32,

    /// Minimum final score in [0,1] to accept a pair.
    pub min_total_score: f32,

    /// Whether to use RT/IM/intensity terms at all.
    pub use_rt: bool,
    pub use_im: bool,
    pub use_intensity: bool,
}

impl Default for XicScoreOpts {
    fn default() -> Self {
        Self {
            // shape dominates, intensity is a weak prior
            w_rt: 0.45,
            w_im: 0.45,
            w_intensity: 0.10,
            intensity_tau: 1.5,
            min_total_score: 0.0, // no cutoff by default

            use_rt: true,
            use_im: true,
            use_intensity: true,
        }
    }
}

#[inline]
fn zscore(v: &[f32]) -> Option<Vec<f32>> {
    let n = v.len();
    if n < 3 {
        return None;
    }
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    for &x in v {
        let xf = x as f64;
        sum += xf;
        sum2 += xf * xf;
    }
    let n_f = n as f64;
    let mean = sum / n_f;
    let var = (sum2 / n_f) - mean * mean;
    if !var.is_finite() || var <= 0.0 {
        return None;
    }
    let std = var.sqrt();
    let mut out = Vec::with_capacity(n);
    for &x in v {
        out.push(((x as f64 - mean) / std) as f32);
    }
    Some(out)
}

/// Pearson correlation on two traces, cropped to the common length and z-scored.
fn pearson_corr_z(a: &[f32], b: &[f32]) -> Option<f32> {
    let n = a.len().min(b.len());
    if n < 3 {
        return None;
    }
    let az = zscore(&a[..n])?;
    let bz = zscore(&b[..n])?;

    let mut num = 0.0f64;
    for i in 0..n {
        num += az[i] as f64 * bz[i] as f64;
    }
    let den = (n as f64).max(1.0);
    let r = (num / den)
        .max(-1.0)
        .min(1.0);
    Some(r as f32)
}

pub fn xic_match_score(
    ms1: &ClusterResult1D,
    ms2: &ClusterResult1D,
    opts: &XicScoreOpts,
) -> Option<(XicDetails, f32)> {
    let mut score   = 0.0f32;
    let mut w_sum   = 0.0f32;

    let mut s_rt: Option<f32>         = None;
    let mut s_im: Option<f32>         = None;
    let mut s_intensity: Option<f32>  = None;
    let mut r_rt: Option<f32>         = None;
    let mut r_im: Option<f32>         = None;

    // ---- RT XIC: Pearson, mapped from [-1,1] to [0,1] ----
    if opts.use_rt {
        if let (Some(ref rt1), Some(ref rt2)) = (&ms1.rt_trace, &ms2.rt_trace) {
            if let Some(r) = pearson_corr_z(rt1, rt2) {
                if r.is_finite() {
                    let s = 0.5 * (r + 1.0); // [-1,1] -> [0,1]
                    r_rt  = Some(r);
                    s_rt  = Some(s);
                    score += opts.w_rt * s;
                    w_sum += opts.w_rt;
                }
            }
        }
    }

    // ---- IM XIC: Pearson, same mapping ----
    if opts.use_im {
        if let (Some(ref im1), Some(ref im2)) = (&ms1.im_trace, &ms2.im_trace) {
            if let Some(r) = pearson_corr_z(im1, im2) {
                if r.is_finite() {
                    let s = 0.5 * (r + 1.0);
                    r_im  = Some(r);
                    s_im  = Some(s);
                    score += opts.w_im * s;
                    w_sum += opts.w_im;
                }
            }
        }
    }

    // ---- Intensity ratio: weak, symmetric penalty on log ratio ----
    if opts.use_intensity && opts.w_intensity > 0.0 && opts.intensity_tau > 0.0 {
        // Use RT trace integrals as a proxy; fall back to raw_sum if needed.
        let i1 = if let Some(ref rt1) = ms1.rt_trace {
            rt1.iter().fold(0.0f32, |acc, x| acc + x.max(0.0))
        } else {
            ms1.raw_sum.max(0.0)
        };

        let i2 = if let Some(ref rt2) = ms2.rt_trace {
            rt2.iter().fold(0.0f32, |acc, x| acc + x.max(0.0))
        } else {
            ms2.raw_sum.max(0.0)
        };

        if i1 > 0.0 && i2 > 0.0 {
            let ratio = (i2 / i1).max(1e-6);
            let d     = ratio.ln().abs(); // |log(I2/I1)|
            let s     = (-d / opts.intensity_tau).exp(); // in (0,1]
            if s.is_finite() {
                s_intensity = Some(s);
                score      += opts.w_intensity * s;
                w_sum      += opts.w_intensity;
            }
        }
    }

    if w_sum <= 0.0 {
        return None;
    }

    let final_score = score / w_sum;
    if !final_score.is_finite() {
        None
    } else {
        let details = XicDetails {
            s_rt,
            s_im,
            s_intensity,
            r_rt,
            r_im,
        };
        Some((details, final_score.clamp(0.0, 1.0)))
    }
}

pub fn assign_ms2_to_best_ms1_by_xic(
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    pairs: &[(usize, usize)],
    opts: &XicScoreOpts,
) -> Vec<(usize, usize, f32)> {
    if ms1.is_empty() || ms2.is_empty() || pairs.is_empty() {
        return Vec::new();
    }

    // Precompute RT-integrated intensities (or raw_sum fallback) once per cluster.
    let ms1_int = precompute_intensities(ms1);
    let ms2_int = precompute_intensities(ms2);

    // Dense best-hit buffer: one slot per MS2, no hashing.
    let mut best: Vec<Option<(usize, f32)>> = vec![None; ms2.len()];

    for &(ms2_idx, ms1_idx) in pairs {
        if ms1_idx >= ms1.len() || ms2_idx >= ms2.len() {
            continue;
        }

        // Use the precomputed-intensity version of the scorer.
        let s = match xic_match_score_precomputed(
            ms1_idx,
            ms2_idx,
            ms1,
            ms2,
            &ms1_int,
            &ms2_int,
            opts,
        ) {
            Some(v) => v,
            None => continue,
        };

        // You can drop this if you already enforce the cutoff in xic_match_score_precomputed.
        if s < opts.min_total_score {
            continue;
        }

        match &mut best[ms2_idx] {
            Some((_best_i, best_s)) if s <= *best_s => {
                // keep existing winner
            }
            slot => {
                // new best (or first) hit for this MS2
                *slot = Some((ms1_idx, s));
            }
        }
    }

    // Build result already sorted by ms2_idx.
    let mut out = Vec::new();
    out.reserve(pairs.len().min(ms2.len()));
    for (ms2_idx, maybe) in best.into_iter().enumerate() {
        if let Some((ms1_idx, s)) = maybe {
            out.push((ms2_idx, ms1_idx, s));
        }
    }
    out
}

/// Jaccard overlap in **absolute seconds** for two closed intervals.
#[inline]
pub fn jaccard_time(a_lo: f64, a_hi: f64, b_lo: f64, b_hi: f64) -> f32 {
    if !a_lo.is_finite() || !a_hi.is_finite() || !b_lo.is_finite() || !b_hi.is_finite() {
        return 0.0;
    }
    if a_hi < b_lo || b_hi < a_lo {
        return 0.0;
    }
    let inter = (a_hi.min(b_hi) - a_lo.max(b_lo)).max(0.0);
    let union = (a_hi.max(b_hi) - a_lo.min(b_lo)).max(0.0);
    if union <= 0.0 {
        0.0
    } else {
        (inter / union) as f32
    }
}

#[inline]
fn overlap_f32(a: (f32, f32), b: (f32, f32)) -> bool {
    let (a0, a1) = a;
    let (b0, b1) = b;
    a0.is_finite()
        && a1.is_finite()
        && b0.is_finite()
        && b1.is_finite()
        && a1 >= b0
        && b1 >= a0
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
            if x <= lo {
                0
            } else {
                (((x - lo) * inv_bw).floor() as isize).clamp(0, (n as isize) - 1) as usize
            }
        };

        for (i, &(t0, t1)) in ms1_time_bounds.iter().enumerate() {
            if let Some(keep) = ms1_keep {
                if !keep[i] {
                    continue;
                }
            }
            if !(t0.is_finite() && t1.is_finite()) || t1 <= t0 {
                continue;
            }
            let b0 = clamp(t0);
            let b1 = clamp(t1);
            for b in b0..=b1 {
                buckets[b].push(i);
            }
        }
        Self { lo, inv_bw, buckets }
    }

    #[inline]
    fn range(&self, t0: f64, t1: f64) -> (usize, usize) {
        let n = self.buckets.len();
        let clamp = |x: f64| -> usize {
            if x <= self.lo {
                0
            } else {
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
        for b in b0..=b1 {
            out.extend_from_slice(&self.buckets[b]);
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
    /// DIA program index (for tile-level checks).
    dia_index: Arc<DiaIndex>,
}

impl PrecursorSearchIndex {
    /// Build once per dataset / MS1 set.
    pub fn build(ds: &TimsDatasetDIA, ms1: &[ClusterResult1D], opts: &CandidateOpts) -> Self {
        let frame_time = Arc::new(ds.dia_index.frame_time.clone());
        let dia_index = Arc::new(ds.dia_index.clone());

        // 1) Absolute MS1 time bounds
        let ms1_time_bounds: Vec<(f64, f64)> = ms1
            .par_iter()
            .map(|c| {
                let mut t_lo = f64::INFINITY;
                let mut t_hi = f64::NEG_INFINITY;

                // Preferred: use stored frame_ids_used if available
                if !c.frame_ids_used.is_empty() {
                    for &fid in &c.frame_ids_used {
                        if let Some(&t) = frame_time.get(&fid) {
                            if t < t_lo { t_lo = t; }
                            if t > t_hi { t_hi = t; }
                        }
                    }
                }

                // Fallback: infer from rt_window if frame_ids_used is empty
                if !t_lo.is_finite() || !t_hi.is_finite() {
                    let (rt_lo, rt_hi) = c.rt_window;
                    if rt_hi >= rt_lo {
                        for fid in rt_lo as u32..=rt_hi as u32 {
                            if let Some(&t) = frame_time.get(&fid) {
                                if t < t_lo { t_lo = t; }
                                if t > t_hi { t_hi = t; }
                            }
                        }
                    }
                }

                // If still invalid, caller will filter this out via ms1_keep.
                (t_lo, t_hi)
            })
            .collect();

        // 2) Keep mask for MS1
        let ms1_keep: Vec<bool> = ms1
            .par_iter()
            .enumerate()
            .map(|(i, c)| {
                if c.ms_level != 1 {
                    return false;
                }
                if c.raw_sum < opts.min_raw_sum {
                    return false;
                }
                let (t0, t1) = ms1_time_bounds[i];
                if !(t0.is_finite() && t1.is_finite()) || t1 <= t0 {
                    return false;
                }
                if let Some(max_rt) = opts.max_ms1_rt_span_sec {
                    if (t1 - t0) > max_rt {
                        return false;
                    }
                }
                true
            })
            .collect();

        // 3) RT buckets across MS1
        let (mut rt_min, mut rt_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for &(a, b) in &ms1_time_bounds {
            if a.is_finite() {
                rt_min = rt_min.min(a);
            }
            if b.is_finite() {
                rt_max = rt_max.max(b);
            }
        }
        if !rt_min.is_finite() || !rt_max.is_finite() || rt_max <= rt_min {
            rt_min = 0.0;
            rt_max = 1.0;
        }
        let rt_buckets = RtBuckets::build(
            rt_min,
            rt_max,
            opts.rt_bucket_width,
            &ms1_time_bounds,
            Some(&ms1_keep),
        );

        // 4) Per-group eligibility masks (mz ∩ isolation AND scans ∩ program), independent of RT.
        let ms1_group_ok: HashMap<u32, Vec<bool>> = ds
            .dia_index
            .group_to_isolation
            .par_iter()
            .map(|(&g, mz_rows)| {
                let scans = ds
                    .dia_index
                    .group_to_scan_ranges
                    .get(&g)
                    .cloned()
                    .unwrap_or_default();
                let mz_rows_f32: Vec<(f32, f32)> =
                    mz_rows.iter().map(|&(a, b)| (a as f32, b as f32)).collect();
                let scan_rows_u32: Vec<(u32, u32)> = scans.iter().copied().collect();

                if mz_rows_f32.is_empty() || scan_rows_u32.is_empty() {
                    return (g, vec![false; ms1.len()]);
                }

                let mask: Vec<bool> = ms1
                    .par_iter()
                    .map(|c| {
                        if c.ms_level != 1 {
                            return false;
                        }
                        let mz_ok = mz_rows_f32
                            .iter()
                            .any(|&w| overlap_f32(c.mz_window.unwrap(), w));
                        if !mz_ok {
                            return false;
                        }
                        let im_u32 = (c.im_window.0 as u32, c.im_window.1 as u32);
                        scan_rows_u32.iter().any(|&s| overlap_u32(im_u32, s))
                    })
                    .collect();

                (g, mask)
            })
            .collect();

        Self {
            ms1_time_bounds,
            ms1_keep,
            rt_buckets,
            ms1_group_ok,
            frame_time,
            dia_index,
        }
    }

    /// Enumerate physically plausible MS1–MS2 pairs.
    ///
    /// Conditions:
    ///   - Same window group.
    ///   - RT overlap (with optional Jaccard threshold).
    ///   - IM window overlap (min_im_overlap_scans).
    ///   - Apex deltas in RT/IM within user bounds.
    ///   - NEW: there exists at least one tile (ProgramSlice) where:
    ///       * precursor IM window overlaps tile scans AND precursor m/z is in tile isolation
    ///       * fragment IM window overlaps tile scans (no m/z restriction).
    pub fn enumerate_pairs(
        &self,
        ms1: &[ClusterResult1D],
        ms2: &[ClusterResult1D],
        opts: &CandidateOpts,
    ) -> Vec<(usize, usize)> {
        // Precompute MS2 absolute time bounds
        let ms2_time_bounds: Vec<(f64, f64)> = ms2
            .par_iter()
            .map(|c| {
                let mut t_lo = f64::INFINITY;
                let mut t_hi = f64::NEG_INFINITY;

                if !c.frame_ids_used.is_empty() {
                    for &fid in &c.frame_ids_used {
                        if let Some(&t) = self.frame_time.get(&fid) {
                            if t < t_lo { t_lo = t; }
                            if t > t_hi { t_hi = t; }
                        }
                    }
                }

                // Fallback: use rt_window if no frame_ids_used
                if !t_lo.is_finite() || !t_hi.is_finite() {
                    let (rt_lo, rt_hi) = c.rt_window;
                    if rt_hi >= rt_lo {
                        for fid in rt_lo as u32..=rt_hi as u32 {
                            if let Some(&t) = self.frame_time.get(&fid) {
                                if t < t_lo { t_lo = t; }
                                if t > t_hi { t_hi = t; }
                            }
                        }
                    }
                }

                (t_lo, t_hi)
            })
            .collect();

        let ms2_time_bounds = Arc::new(ms2_time_bounds);

        // Keep MS2s
        let ms2_keep: Vec<bool> = ms2
            .par_iter()
            .enumerate()
            .map(|(i, c)| {
                if c.ms_level != 2 {
                    return false;
                }
                if c.window_group.is_none() {
                    return false;
                }
                if c.raw_sum < opts.min_raw_sum {
                    return false;
                }
                let (mut t0, mut t1) = ms2_time_bounds[i];
                if t0.is_finite() {
                    t0 -= opts.ms2_rt_guard_sec;
                }
                if t1.is_finite() {
                    t1 += opts.ms2_rt_guard_sec;
                }
                if !(t0.is_finite() && t1.is_finite() && t1 > t0) {
                    return false;
                }
                if let Some(max_rt) = opts.max_ms2_rt_span_sec {
                    if (t1 - t0) > max_rt {
                        return false;
                    }
                }
                true
            })
            .collect();

        // Group MS2 by window_group
        let mut by_group: HashMap<u32, Vec<usize>> = HashMap::new();
        for (j, c2) in ms2.iter().enumerate() {
            if !ms2_keep[j] {
                continue;
            }
            if let Some(g) = c2.window_group {
                by_group.entry(g).or_default().push(j);
            }
        }

        let idx_arc = Arc::new(self.clone());
        let ms2_tb = ms2_time_bounds.clone();

        let mut out = by_group
            .into_par_iter()
            .flat_map(|(g, js)| {
                // Eligibility mask for this group
                let mask_vec: Vec<bool> = idx_arc
                    .ms1_group_ok
                    .get(&g)
                    .cloned()
                    .unwrap_or_else(|| vec![false; ms1.len()]);
                let mask_arc = Arc::new(mask_vec);

                let idx = idx_arc.clone();
                let tb = ms2_tb.clone();
                let dia_index = idx.dia_index.clone();

                // NEW: program slices for this group, shared across MS2 in this group
                let slices_vec = dia_index.program_slices_for_group(g);
                let slices = Arc::new(slices_vec);

                js.into_par_iter().flat_map(move |j| {
                    let (mut t2_lo, mut t2_hi) = tb[j];
                    if t2_lo.is_finite() {
                        t2_lo -= opts.ms2_rt_guard_sec;
                    }
                    if t2_hi.is_finite() {
                        t2_hi += opts.ms2_rt_guard_sec;
                    }

                    // RT prefilter via buckets
                    let mut hits = Vec::<usize>::new();
                    idx.rt_buckets.gather(t2_lo, t2_hi, &mut hits);
                    hits.sort_unstable();
                    hits.dedup();

                    let mask = mask_arc.clone();
                    let slices = slices.clone();

                    let mut local = Vec::<(usize, usize)>::with_capacity(16);
                    for i in hits {
                        if !idx.ms1_keep[i] {
                            continue;
                        }
                        if !mask[i] {
                            continue;
                        }

                        let (t1_lo, t1_hi) = idx.ms1_time_bounds[i];
                        if !(t1_lo.is_finite() && t1_hi.is_finite()) {
                            continue;
                        }

                        // RT overlap + optional Jaccard
                        if t1_hi < t2_lo || t2_hi < t1_lo {
                            continue;
                        }
                        if opts.min_rt_jaccard > 0.0 {
                            let jacc = jaccard_time(t1_lo, t1_hi, t2_lo, t2_hi);
                            if jacc < opts.min_rt_jaccard {
                                continue;
                            }
                        }

                        let im1 = ms1[i].im_window;
                        let im2 = ms2[j].im_window;

                        // Basic IM overlap (before tile check)
                        let im_overlap = {
                            let lo = im1.0.max(im2.0);
                            let hi = im1.1.min(im2.1);
                            hi.saturating_sub(lo).saturating_add(1)
                        };
                        if im_overlap < opts.min_im_overlap_scans {
                            continue;
                        }

                        // Apex deltas in IM
                        if let Some(max_d) = opts.max_scan_apex_delta {
                            let s1 = ms1[i].im_fit.mu;
                            let s2 = ms2[j].im_fit.mu;
                            if s1.is_finite() && s2.is_finite() {
                                let d = (s1 - s2).abs() as f32;
                                if d > max_d as f32 {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        // Apex deltas in RT
                        if let Some(max_dt) = opts.max_rt_apex_delta_sec {
                            let r1 = ms1[i].rt_fit.mu;
                            let r2 = ms2[j].rt_fit.mu;
                            if r1.is_finite() && r2.is_finite() {
                                if (r1 - r2).abs() > max_dt {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        // ---- Tile-level physical check ----
                        let prec_mz = match cluster_mz_mu(&ms1[i]) {
                            Some(m) if m.is_finite() && m > 0.0 => m,
                            _ => continue,
                        };

                        // Require the precursor IM apex to lie inside some tile
                        let prec_im_apex = ms1[i].im_fit.mu;
                        if !prec_im_apex.is_finite() {
                            continue;
                        }

                        // Tiles where this precursor could have been selected in g (apex-based)
                        let prec_tiles = dia_index.tiles_for_precursor_in_group(
                            g,
                            prec_mz,
                            prec_im_apex,
                        );
                        if prec_tiles.is_empty() {
                            continue;
                        }

                        // Tiles where this fragment cluster could appear in g (window-based)
                        let frag_tiles = dia_index.tiles_for_fragment_in_group(g, im2);
                        if frag_tiles.is_empty() {
                            continue;
                        }

                        // At least one shared tile index (physical co-occurrence)
                        let mut ok = false;
                        for t in &prec_tiles {
                            if frag_tiles.contains(t) {
                                ok = true;
                                break;
                            }
                        }
                        if !ok {
                            continue;
                        }

                        // NEW: reject fragments whose own selection lies in the same isolation tile
                        // as the precursor (to avoid unfragmented precursor intensity in MS2).
                        if opts.reject_frag_inside_precursor_tile {
                            if let Some(frag_mz) = cluster_mz_mu(&ms2[j]) {
                                if frag_mz.is_finite() && frag_mz > 0.0 {
                                    let mut reject = false;

                                    // only tiles in intersection prec_tiles ∩ frag_tiles
                                    for &tile_idx in &prec_tiles {
                                        if !frag_tiles.contains(&tile_idx) {
                                            continue;
                                        }
                                        if tile_idx >= slices.len() {
                                            continue;
                                        }
                                        let s = &slices[tile_idx]; // ProgramSlice { mz_lo, mz_hi, scan_lo, scan_hi }

                                        if (frag_mz as f64) >= s.mz_lo
                                            && (frag_mz as f64) <= s.mz_hi
                                        {
                                            // fragment looks like it's still inside the
                                            // precursor's isolation tile -> drop this pair
                                            reject = true;
                                            break;
                                        }
                                    }

                                    if reject {
                                        continue;
                                    }
                                }
                            }
                        }

                        // Survives all guards
                        local.push((j, i));
                    }

                    local.into_par_iter()
                })
            })
            .collect::<Vec<(usize, usize)>>();

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

// ---------------------------------------------------------------------------
// Scoring (unchanged; left here for completeness)
// ---------------------------------------------------------------------------

/// Single scalar score in [0, ∞), larger is better.
/// Robust to missing fits (uses `shape_neutral` if shape is unavailable).
#[inline]
fn score_from_features(f: &PairFeatures, opts: &ScoreOpts) -> f32 {
    let shape_term = if f.shape_ok { f.s_shape } else { opts.shape_neutral };

    let rt_close = crate::cluster::candidates::exp_decay(f.rt_apex_delta_s, opts.rt_apex_scale_s);
    let im_close = crate::cluster::candidates::exp_decay(f.im_apex_delta_scans, opts.im_apex_scale_scans);

    let im_ratio = (f.im_overlap_scans as f32) / (f.im_union_scans as f32);

    let ms1_int = crate::cluster::candidates::safe_log1p(f.ms1_raw_sum);

    opts.w_jacc_rt * f.jacc_rt
        + opts.w_shape * shape_term
        + opts.w_rt_apex * rt_close
        + opts.w_im_apex * im_close
        + opts.w_im_overlap * im_ratio
        + opts.w_ms1_intensity * ms1_int
}

/// Score all pairs (ms2_idx, ms1_idx).
pub fn score_pairs(
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    pairs: &[(usize, usize)],
    opts: &ScoreOpts,
) -> Vec<(usize, usize, PairFeatures, f32)> {
    pairs.par_iter().map(|&(j, i)| {
        let f = crate::cluster::candidates::build_features(&ms1[i], &ms2[j], opts);
        let s = score_from_features(&f, opts);
        (j, i, f, s)
    }).collect()
}

/// For each MS2, choose the best MS1 index (by score, then deterministic tie-breaks).
/// Returns a Vec<Option<usize>> indexed by ms2_idx.
pub fn best_ms1_for_each_ms2(
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    pairs: &[(usize, usize)],
    opts: &ScoreOpts,
) -> Vec<Option<usize>> {
    let scored = score_pairs(ms1, ms2, pairs, opts);

    // group by ms2_idx
    let mut by_ms2: Vec<Vec<(usize, PairFeatures, f32)>> = vec![Vec::new(); ms2.len()];
    for (j, i, f, s) in scored {
        by_ms2[j].push((i, f, s));
    }

    by_ms2
        .into_par_iter()
        .map(|mut vec_i| {
            if vec_i.is_empty() { return None; }
            vec_i.sort_unstable_by(|a, b| {
                // primary: score desc
                match b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal) {
                    Ordering::Equal => {
                        // tie-breaks (deterministic):
                        // 1) higher jaccard
                        let ja = a.1.jacc_rt;
                        let jb = b.1.jacc_rt;
                        if (ja - jb).abs() > 1e-6 {
                            return jb.partial_cmp(&ja).unwrap_or(Ordering::Equal);
                        }
                        // 2) if both have shape, higher s_shape
                        let sa = if a.1.shape_ok { a.1.s_shape } else { 0.0 };
                        let sb = if b.1.shape_ok { b.1.s_shape } else { 0.0 };
                        if (sa - sb).abs() > 1e-6 {
                            return sb.partial_cmp(&sa).unwrap_or(Ordering::Equal);
                        }
                        // 3) smaller RT apex delta
                        let dra = a.1.rt_apex_delta_s;
                        let drb = b.1.rt_apex_delta_s;
                        if (dra - drb).abs() > 1e-6 {
                            return dra.partial_cmp(&drb).unwrap_or(Ordering::Equal);
                        }
                        // 4) smaller IM apex delta
                        let dia = a.1.im_apex_delta_scans;
                        let dib = b.1.im_apex_delta_scans;
                        if (dia - dib).abs() > 1e-6 {
                            return dia.partial_cmp(&dib).unwrap_or(Ordering::Equal);
                        }
                        // 5) larger IM overlap
                        let oa = a.1.im_overlap_scans;
                        let ob = b.1.im_overlap_scans;
                        if oa != ob {
                            return ob.cmp(&oa);
                        }
                        // 6) higher MS1 intensity
                        let ia = a.1.ms1_raw_sum;
                        let ib = b.1.ms1_raw_sum;
                        ib.partial_cmp(&ia).unwrap_or(Ordering::Equal)
                    }
                    ord => ord,
                }
            });
            Some(vec_i[0].0)
        })
        .collect()
}

/// Build an MS1 → Vec<MS2> map from a winner list (ms2→best ms1).
/// Returns a Vec<Vec<usize>> with length ms1.len(), where each entry lists MS2 indices.
pub fn ms1_to_ms2_map(
    ms1_len: usize,
    ms2_to_best_ms1: &[Option<usize>],
) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::<usize>::new(); ms1_len];
    for (ms2_idx, maybe_ms1) in ms2_to_best_ms1.iter().enumerate() {
        if let Some(i) = maybe_ms1 {
            if *i < ms1_len {
                out[*i].push(ms2_idx);
            }
        }
    }
    out
}

fn precompute_intensities(clusters: &[ClusterResult1D]) -> Vec<f32> {
    clusters
        .par_iter()
        .map(|c| {
            if let Some(ref rt) = c.rt_trace {
                rt.iter().fold(0.0f32, |acc, x| acc + x.max(0.0))
            } else {
                c.raw_sum.max(0.0)
            }
        })
        .collect()
}

#[inline]
fn top_intensity_member(feat: &SimpleFeature) -> Option<&ClusterResult1D> {
    if feat.member_clusters.is_empty() {
        return None;
    }
    feat.member_clusters
        .iter()
        .max_by(|a, b| a.raw_sum.partial_cmp(&b.raw_sum).unwrap_or(Ordering::Equal))
}

/// Geometric score for a *single* precursor cluster vs a fragment.
///
/// Returns (PairFeatures, score).
#[inline]
pub fn geom_match_score_single_cluster(
    ms1: &ClusterResult1D,
    ms2: &ClusterResult1D,
    opts: &ScoreOpts,
) -> (PairFeatures, f32) {
    let f = crate::cluster::candidates::build_features(ms1, ms2, opts);
    let s = score_from_features(&f, opts);
    (f, s)
}

pub fn xic_match_score_precomputed(
    ms1_idx: usize,
    ms2_idx: usize,
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    ms1_int: &[f32],
    ms2_int: &[f32],
    opts: &XicScoreOpts,
) -> Option<f32> {
    let mut score = 0.0f32;
    let mut w_sum = 0.0f32;

    let c1 = &ms1[ms1_idx];
    let c2 = &ms2[ms2_idx];

    // RT XIC
    if opts.use_rt {
        if let (Some(ref rt1), Some(ref rt2)) = (&c1.rt_trace, &c2.rt_trace) {
            if let Some(r) = pearson_corr_z(rt1, rt2) {
                if r.is_finite() {
                    let s = 0.5 * (r + 1.0);
                    score += opts.w_rt * s;
                    w_sum += opts.w_rt;
                }
            }
        }
    }

    // IM XIC
    if opts.use_im {
        if let (Some(ref im1), Some(ref im2)) = (&c1.im_trace, &c2.im_trace) {
            if let Some(r) = pearson_corr_z(im1, im2) {
                if r.is_finite() {
                    let s = 0.5 * (r + 1.0);
                    score += opts.w_im * s;
                    w_sum += opts.w_im;
                }
            }
        }
    }

    // Intensity ratio using precomputed integrals
    if opts.use_intensity && opts.w_intensity > 0.0 && opts.intensity_tau > 0.0 {
        let i1 = ms1_int[ms1_idx];
        let i2 = ms2_int[ms2_idx];

        if i1 > 0.0 && i2 > 0.0 {
            let ratio = (i2 / i1).max(1e-6);
            let d = ratio.ln().abs();
            let s = (-d / opts.intensity_tau).exp();
            if s.is_finite() {
                score += opts.w_intensity * s;
                w_sum += opts.w_intensity;
            }
        }
    }

    if w_sum <= 0.0 {
        return None;
    }
    let final_score = score / w_sum;
    if !final_score.is_finite() {
        None
    } else {
        Some(final_score.clamp(0.0, 1.0))
    }
}

pub fn xic_match_score_precursor(
    prec: PrecursorLike<'_>,
    frag: &ClusterResult1D,
    opts: &XicScoreOpts,
) -> Option<(XicDetails, f32)> {
    match prec {
        PrecursorLike::Cluster(c) => xic_match_score(c, frag, opts),

        PrecursorLike::Feature(f) => {
            let top = top_intensity_member(f)?;
            xic_match_score(top, frag, opts)
        }
    }
}


pub fn geom_match_score_precursor(
    prec: PrecursorLike<'_>,
    frag: &ClusterResult1D,
    opts: &ScoreOpts,
) -> Option<(PairFeatures, f32)> {
    match prec {
        PrecursorLike::Cluster(c) => {
            let (f, s) = geom_match_score_single_cluster(c, frag, opts);
            Some((f, s))
        }

        PrecursorLike::Feature(feat) => {
            let top = top_intensity_member(feat)?;
            let (f, s) = geom_match_score_single_cluster(top, frag, opts);
            Some((f, s))
        }
    }
}

/// Score a single precursor (cluster or feature) against a set of MS2 candidates.
///
/// - `prec`          : the precursor-like object (ClusterResult1D or SimpleFeature)
/// - `ms2`           : full fragment cluster array (e.g. `self.ms2`)
/// - `candidate_ids` : indices into `ms2` that passed the physical filters
/// - `mode`          : which scoring to use (Geom vs XIC)
/// - `geom_opts`     : required for Geom mode
/// - `xic_opts`      : required for XIC mode
/// - `min_score`     : keep only hits with `score >= min_score`
///
/// Returns a Vec of (frag_idx, score) sorted by descending score.
pub fn query_precursor_scored(
    prec: PrecursorLike<'_>,
    ms2: &[ClusterResult1D],
    candidate_ids: &[usize],
    mode: MatchScoreMode,
    geom_opts: &ScoreOpts,
    xic_opts: &XicScoreOpts,
    min_score: f32,
) -> Vec<ScoredHit> {
    let mut out: Vec<ScoredHit> = Vec::with_capacity(candidate_ids.len());

    for &j in candidate_ids {
        if j >= ms2.len() {
            continue;
        }
        let frag = &ms2[j];

        match mode {
            MatchScoreMode::Geom => {
                if let Some((f_geom, s)) = geom_match_score_precursor(prec, frag, geom_opts) {
                    if s.is_finite() && s >= min_score {
                        out.push(ScoredHit {
                            frag_idx: j,
                            score:   s,
                            geom:    Some(f_geom),
                            xic:     None,
                        });
                    }
                }
            }
            MatchScoreMode::Xic => {
                if let Some((xic_det, s)) = xic_match_score_precursor(prec, frag, xic_opts) {
                    if s.is_finite() && s >= min_score {
                        out.push(ScoredHit {
                            frag_idx: j,
                            score:   s,
                            geom:    None,
                            xic:     Some(xic_det),
                        });
                    }
                }
            }
        }
    }

    out.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
    });

    out
}

/// Score many precursors in parallel.
///
/// - `precs`              : slice of precursor-like objects (Cluster or Feature)
/// - `ms2`                : full MS2 slice
/// - `candidates_per_prec`: for each precursor, the Vec<usize> of candidate MS2 indices
///
/// Returns a Vec of length `precs.len()`, where each entry is the
/// scored hits for that precursor (sorted by descending score).
pub fn query_precursors_scored_par(
    precs: &[PrecursorLike<'_>],
    ms2: &[ClusterResult1D],
    candidates_per_prec: &[Vec<usize>],
    mode: MatchScoreMode,
    geom_opts: &ScoreOpts,
    xic_opts: &XicScoreOpts,
    min_score: f32,
) -> Vec<Vec<ScoredHit>> {
    assert_eq!(
        precs.len(),
        candidates_per_prec.len(),
        "precs and candidates_per_prec must have same length"
    );

    precs
        .par_iter()
        .zip(candidates_per_prec.par_iter())
        .map(|(&prec, cands)| {
            query_precursor_scored(
                prec,
                ms2,
                cands,
                mode,
                geom_opts,
                xic_opts,
                min_score,
            )
        })
        .collect()
}