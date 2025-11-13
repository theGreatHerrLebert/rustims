

/*
use crate::cluster::cluster::ClusterResult1D;

/// Compact feature bundle per pair for traceability.
#[derive(Clone, Copy, Debug)]
pub struct PairFeatures {
    pub jacc_rt: f32,            // Jaccard in RT (0..1)
    pub rt_apex_delta_s: f32,    // |μ_rt(MS1)-μ_rt(MS2)| in seconds
    pub im_apex_delta_scans: f32,// |μ_im(MS1)-μ_im(MS2)| in scans
    pub im_overlap_scans: u32,   // intersection size of IM windows in scans
    pub im_union_scans: u32,     // union size of IM windows in scans
    pub ms1_raw_sum: f32,        // intensity proxy for MS1
    pub shape_ok: bool,          // both σ present & finite
    pub z_rt: f32,               // pooled-σ z for RT apex delta
    pub z_im: f32,               // pooled-σ z for IM apex delta
    pub s_shape: f32,            // exp(-0.5 (w_rt z_rt^2 + w_im z_im^2)) in [0,1]
}

/// Scoring knobs. Defaults are conservative and width-aware but won’t punish
/// pairs that lack good fits (we use `shape_neutral` when shape data is missing).
#[derive(Clone, Debug)]
pub struct ScoreOpts {
    /// Weight for RT Jaccard.
    pub w_jacc_rt: f32,
    /// Weight for shape similarity S_shape.
    pub w_shape: f32,
    /// Weight for RT apex proximity term (smaller delta = better).
    pub w_rt_apex: f32,
    /// Weight for IM apex proximity term (smaller delta = better).
    pub w_im_apex: f32,
    /// Weight for IM overlap ratio.
    pub w_im_overlap: f32,
    /// Weight for MS1 raw_sum (log-compressed).
    pub w_ms1_intensity: f32,

    /// Scales to normalize apex deltas into ~0..1 decays (exp(-delta/scale)).
    pub rt_apex_scale_s: f32,
    pub im_apex_scale_scans: f32,

    /// If shape is unavailable, use this neutral value instead of 0.
    pub shape_neutral: f32,

    /// Floors for σ to avoid division by ~0.
    pub min_sigma_rt: f32,
    pub min_sigma_im: f32,

    /// Shape component internal weights (multiply z^2 inside exp).
    pub w_shape_rt_inner: f32,
    pub w_shape_im_inner: f32,
}

impl Default for ScoreOpts {
    fn default() -> Self {
        Self {
            w_jacc_rt: 1.0,
            w_shape: 1.0,
            w_rt_apex: 0.75,
            w_im_apex: 0.75,
            w_im_overlap: 0.5,
            w_ms1_intensity: 0.25,
            rt_apex_scale_s: 0.75,       // ~sub-second deltas favored
            im_apex_scale_scans: 3.0,    // a few scans favored
            shape_neutral: 0.6,          // don’t punish missing shape harshly
            min_sigma_rt: 0.05,
            min_sigma_im: 0.5,
            w_shape_rt_inner: 1.0,
            w_shape_im_inner: 1.0,
        }
    }
}

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

#[inline]
fn im_overlap_and_union(a: (usize, usize), b: (usize, usize)) -> (u32, u32) {
    let lo = a.0.max(b.0);
    let hi = a.1.min(b.1);
    let inter = if hi >= lo { (hi - lo + 1) as u32 } else { 0 };
    let a_len = if a.1 >= a.0 { (a.1 - a.0 + 1) as u32 } else { 0 };
    let b_len = if b.1 >= b.0 { (b.1 - b.0 + 1) as u32 } else { 0 };
    let union = a_len + b_len - inter;
    (inter, union.max(1))
}

#[inline]
fn pooled_sigma(s1: f32, s2: f32) -> Option<f32> {
    let v = s1 * s1 + s2 * s2;
    if v.is_finite() && v > 0.0 { Some(v.sqrt()) } else { None }
}

#[inline]
fn exp_decay(delta: f32, scale: f32) -> f32 {
    // Monotone in [0,∞): 1 at 0, then decays smoothly
    if !delta.is_finite() || !scale.is_finite() || scale <= 0.0 { return 0.0; }
    (-delta / scale).exp()
}

#[inline]
fn safe_log1p(x: f32) -> f32 {
    if x.is_finite() && x >= 0.0 { (1.0 + x as f64).ln() as f32 } else { 0.0 }
}

#[inline]
fn build_features(ms1: &ClusterResult1D, ms2: &ClusterResult1D, opts: &ScoreOpts) -> PairFeatures {
    // RT Jaccard over absolute time bounds derived from frame_ids_used + rt_fit.mu as fallback
    let (rt1_lo, rt1_hi) = (ms1.rt_fit.mu as f64 - (ms1.rt_fit.sigma as f64)*3.0,
                            ms1.rt_fit.mu as f64 + (ms1.rt_fit.sigma as f64)*3.0);
    let (rt2_lo, rt2_hi) = (ms2.rt_fit.mu as f64 - (ms2.rt_fit.sigma as f64)*3.0,
                            ms2.rt_fit.mu as f64 + (ms2.rt_fit.sigma as f64)*3.0);
    let jacc_rt = jaccard_time(rt1_lo, rt1_hi, rt2_lo, rt2_hi).clamp(0.0, 1.0);

    // Apex deltas
    let rt_apex_delta_s = (ms1.rt_fit.mu - ms2.rt_fit.mu).abs();
    let im_apex_delta_scans = (ms1.im_fit.mu - ms2.im_fit.mu).abs();

    // IM overlap ratio
    let (im_inter, im_union) = im_overlap_and_union(ms1.im_window, ms2.im_window);

    // Shape similarity using pooled σ in each dimension (only if both finite)
    let s1_rt = ms1.rt_fit.sigma.max(opts.min_sigma_rt);
    let s2_rt = ms2.rt_fit.sigma.max(opts.min_sigma_rt);
    let s1_im = ms1.im_fit.sigma.max(opts.min_sigma_im);
    let s2_im = ms2.im_fit.sigma.max(opts.min_sigma_im);

    let (mut shape_ok, mut z_rt, mut z_im, mut s_shape) = (false, 0.0, 0.0, 0.0);
    if let (Some(sig_rt), Some(sig_im)) = (pooled_sigma(s1_rt, s2_rt), pooled_sigma(s1_im, s2_im)) {
        if sig_rt.is_finite() && sig_rt > 0.0 && sig_im.is_finite() && sig_im > 0.0 {
            z_rt = rt_apex_delta_s / sig_rt;
            z_im = im_apex_delta_scans / sig_im;
            let q = -0.5_f32 * (opts.w_shape_rt_inner * z_rt * z_rt + opts.w_shape_im_inner * z_im * z_im);
            s_shape = q.exp();         // ∈ (0,1]
            shape_ok = s_shape.is_finite();
        }
    }

    PairFeatures {
        jacc_rt,
        rt_apex_delta_s,
        im_apex_delta_scans,
        im_overlap_scans: im_inter,
        im_union_scans: im_union,
        ms1_raw_sum: ms1.raw_sum,
        shape_ok,
        z_rt,
        z_im,
        s_shape,
    }
}

/// Single scalar score in [0, ∞), larger is better.
/// Robust to missing fits (uses `shape_neutral` if shape is unavailable).
#[inline]
fn score_from_features(f: &PairFeatures, opts: &ScoreOpts) -> f32 {
    let shape_term = if f.shape_ok { f.s_shape } else { opts.shape_neutral };

    let rt_close = exp_decay(f.rt_apex_delta_s, opts.rt_apex_scale_s);
    let im_close = exp_decay(f.im_apex_delta_scans, opts.im_apex_scale_scans);

    let im_ratio = (f.im_overlap_scans as f32) / (f.im_union_scans as f32);

    let ms1_int = safe_log1p(f.ms1_raw_sum);

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
        let f = build_features(&ms1[i], &ms2[j], opts);
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

#[derive(Clone, Debug)]
pub struct AssignmentResult {
    /// All enumerated pairs (ms2_idx, ms1_idx) after your hard guards.
    pub pairs: Vec<(usize, usize)>,
    /// For each MS2 j, the chosen MS1 index (or None if no candidate).
    pub ms2_best_ms1: Vec<Option<usize>>,
    /// For each MS1 i, the list of MS2 indices assigned to it.
    pub ms1_to_ms2: Vec<Vec<usize>>,
}
 */