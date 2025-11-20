use crate::cluster::cluster::ClusterResult1D;
use crate::cluster::feature::SimpleFeature;
use crate::cluster::pseudo::{PseudoSpecOpts, PseudoSpectrum, build_pseudo_spectra_from_pairs};
use crate::cluster::scoring::{
    assign_ms2_to_best_ms1_by_xic,
    jaccard_time,
    ms1_to_ms2_map,
    PrecursorSearchIndex,
    XicScoreOpts,
};
use crate::data::dia::TimsDatasetDIA;

// ---------------------------------------------------------------------------
// Assignment mode abstraction (geom vs XIC)
// ---------------------------------------------------------------------------

pub enum AssignmentMode<'a> {
    Geometric(&'a ScoreOpts),
    Xic(&'a XicScoreOpts),
}

/// Geometric scoring (width / Jaccard / apex / overlap etc.).
pub fn best_ms1_for_each_ms2_geom(
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    pairs: &[(usize, usize)],
    opts: &ScoreOpts,
) -> Vec<Option<usize>> {
    crate::cluster::scoring::best_ms1_for_each_ms2(ms1, ms2, pairs, opts)
}

/// XIC-based assignment: score via RT/IM traces + intensity ratio,
/// choose best MS1 per MS2, return Vec<Option<ms1_idx>> indexed by ms2_idx.
pub fn best_ms1_for_each_ms2_xic(
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    pairs: &[(usize, usize)],
    opts: &XicScoreOpts,
) -> Vec<Option<usize>> {
    let triples = assign_ms2_to_best_ms1_by_xic(ms1, ms2, pairs, opts);
    let mut best: Vec<Option<usize>> = vec![None; ms2.len()];

    for (ms2_idx, ms1_idx, _s) in triples {
        if ms2_idx < best.len() {
            best[ms2_idx] = Some(ms1_idx);
        }
    }
    best
}

/// Switch between geometric and XIC assignment.
pub fn best_ms1_for_each_ms2_any(
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    pairs: &[(usize, usize)],
    mode: AssignmentMode<'_>,
) -> Vec<Option<usize>> {
    match mode {
        AssignmentMode::Geometric(opts) => {
            best_ms1_for_each_ms2_geom(ms1, ms2, pairs, opts)
        }
        AssignmentMode::Xic(opts) => {
            best_ms1_for_each_ms2_xic(ms1, ms2, pairs, opts)
        }
    }
}

// ---------------------------------------------------------------------------
// Candidate enumeration knobs
// ---------------------------------------------------------------------------

/// Options for the simple candidate enumeration.
/// Rule = RT overlap (seconds) AND group eligibility (mz ∩ isolation AND scans ∩ program).
/// Options for the simple candidate enumeration.
/// Rule = RT overlap (seconds) AND group eligibility (mz ∩ isolation AND scans ∩ program).
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

    // ---- tight guards ----
    /// Maximum allowed |rt_apex_MS1 - rt_apex_MS2| in seconds (None disables).
    pub max_rt_apex_delta_sec: Option<f32>,
    /// Maximum allowed |im_apex_MS1 - im_apex_MS2| in global scans (None disables).
    pub max_scan_apex_delta: Option<usize>,
    /// Require at least this many scan-overlap between MS1 and MS2 IM windows.
    pub min_im_overlap_scans: usize,

    /// If true, drop pairs where the fragment cluster's **own** selection
    /// (mz, scan) lies inside the same DIA program slice that could select
    /// the precursor. This suppresses residual, unfragmented precursor
    /// intensity being treated as a fragment.
    pub reject_frag_inside_precursor_tile: bool,
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

            max_rt_apex_delta_sec: Some(2.0),
            max_scan_apex_delta: Some(6),
            min_im_overlap_scans: 1,

            reject_frag_inside_precursor_tile: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Scoring model for geometric assignment
// ---------------------------------------------------------------------------

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
pub(crate) fn build_features(ms1: &ClusterResult1D, ms2: &ClusterResult1D, opts: &ScoreOpts) -> PairFeatures {
    // RT Jaccard over absolute time bounds derived from frame_ids_used + rt_fit.mu as fallback
    let (rt1_lo, rt1_hi) = (
        ms1.rt_fit.mu as f64 - (ms1.rt_fit.sigma as f64) * 3.0,
        ms1.rt_fit.mu as f64 + (ms1.rt_fit.sigma as f64) * 3.0,
    );
    let (rt2_lo, rt2_hi) = (
        ms2.rt_fit.mu as f64 - (ms2.rt_fit.sigma as f64) * 3.0,
        ms2.rt_fit.mu as f64 + (ms2.rt_fit.sigma as f64) * 3.0,
    );
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
            let q = -0.5_f32 * (opts.w_shape_rt_inner * z_rt * z_rt
                + opts.w_shape_im_inner * z_im * z_im);
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
pub(crate) fn exp_decay(delta: f32, scale: f32) -> f32 {
    // Monotone in [0,∞): 1 at 0, then decays smoothly
    if !delta.is_finite() || !scale.is_finite() || scale <= 0.0 { return 0.0; }
    (-delta / scale).exp()
}

#[inline]
pub(crate) fn safe_log1p(x: f32) -> f32 {
    if x.is_finite() && x >= 0.0 { (1.0 + x as f64).ln() as f32 } else { 0.0 }
}

// ---------------------------------------------------------------------------
// End-to-end pseudo-spectra builders
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// End-to-end pseudo-spectra builders
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AssignmentResult {
    /// All enumerated pairs (ms2_idx, ms1_idx) after your hard guards.
    pub pairs: Vec<(usize, usize)>,
    /// For each MS2 j, the chosen MS1 index (or None if no candidate).
    ///
    /// NOTE: in non-competitive "all pairs" mode, this is intentionally
    /// left as `None` for all entries, because there is no unique best.
    pub ms2_best_ms1: Vec<Option<usize>>,
    /// For each MS1 i, the list of MS2 indices assigned to it.
    ///
    /// In "all pairs" mode this is simply the inverted candidate list, so
    /// each MS2 may appear in multiple MS1 buckets.
    pub ms1_to_ms2: Vec<Vec<usize>>,
}

/// Full result of the DIA → pseudo-DDA pipeline.
#[derive(Clone, Debug)]
pub struct PseudoBuildResult {
    /// Assignment information (pairs, best MS1 per MS2, inverted map).
    pub assignment: AssignmentResult,
    /// Final pseudo-MS/MS spectra, one per assigned precursor.
    pub pseudo_spectra: Vec<PseudoSpectrum>,
}

/// NON-COMPETITIVE builder: mainly for debugging / exploration.
///
/// Links *all* MS1–MS2 pairs that
///   - are program-legal (same window group, isolation & scans),
///   - satisfy candidate guards (RT/IM overlap etc.),
/// and then groups them into pseudo-spectra without any competition.
///
/// CONSEQUENCES:
///   - An MS2 cluster may contribute to **multiple** precursors.
///   - `assignment.ms2_best_ms1` is left as `None` for all MS2 indices.
///   - `assignment.ms1_to_ms2[i]` contains **all** MS2 indices linked to MS1 i.
///   - Use `build_pseudo_spectra_end_to_end{,_xic}` for competitive, 1:1 assignment.
pub fn build_pseudo_spectra_all_pairs(
    ds: &TimsDatasetDIA,
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    features: Option<&[SimpleFeature]>,
    pseudo_opts: &PseudoSpecOpts,
) -> PseudoBuildResult {
    // Hard-coded "wide" candidate guards: any reasonable RT/IM overlap is allowed.
    let cand_opts = CandidateOpts {
        min_rt_jaccard: 0.0,
        ms2_rt_guard_sec: 0.0,
        rt_bucket_width: 1.0,
        max_ms1_rt_span_sec: None,
        max_ms2_rt_span_sec: None,
        min_raw_sum: 1.0,

        max_rt_apex_delta_sec: None,
        max_scan_apex_delta:   None,
        min_im_overlap_scans:  1,
        reject_frag_inside_precursor_tile: true,
    };

    // 1) Enumerate all program-legal candidates with hard guards.
    let idx = PrecursorSearchIndex::build(ds, ms1, &cand_opts);
    let pairs = idx.enumerate_pairs(ms1, ms2, &cand_opts); // Vec<(ms2_idx, ms1_idx)>

    // trivial fast-path: nothing links to anything
    if pairs.is_empty() {
        return PseudoBuildResult {
            assignment: AssignmentResult {
                pairs,
                ms2_best_ms1: vec![None; ms2.len()],
                ms1_to_ms2: vec![Vec::new(); ms1.len()],
            },
            pseudo_spectra: Vec::new(),
        };
    }

    // 2) Build pseudo spectra from *all* pairs (no competition).
    let empty_feats: &[SimpleFeature] = &[];
    let feat_slice = features.unwrap_or(empty_feats);

    let pseudo_spectra = build_pseudo_spectra_from_pairs(
        ms1,
        ms2,
        feat_slice,
        &pairs,
        pseudo_opts,
    );

    // 3) Build a non-competitive assignment view.
    //
    // - ms2_best_ms1: no unique best in all-pairs mode → all None.
    // - ms1_to_ms2: for each MS1 i, list all MS2 indices linked to it.
    let mut ms1_to_ms2: Vec<Vec<usize>> = vec![Vec::new(); ms1.len()];
    for (ms2_idx, ms1_idx) in &pairs {
        if *ms1_idx < ms1_to_ms2.len() {
            ms1_to_ms2[*ms1_idx].push(*ms2_idx);
        }
    }

    let ms2_best_ms1 = vec![None; ms2.len()];

    let assignment = AssignmentResult {
        pairs,
        ms2_best_ms1,
        ms1_to_ms2,
    };

    PseudoBuildResult {
        assignment,
        pseudo_spectra,
    }
}

/// End-to-end, **geometric** competitive builder:
///   DIA index + MS1 clusters + MS2 clusters (+ optional features)
///   → candidates → geometric scoring/assignment → pseudo-MS/MS spectra.
///
/// Properties:
///   - Each MS2 cluster participates in **at most one** precursor.
///   - Physical program legality (group, tiles, apex-in-tile) is enforced
///     inside `PrecursorSearchIndex::enumerate_pairs(...)`.
pub fn build_pseudo_spectra_end_to_end(
    ds: &TimsDatasetDIA,
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    features: Option<&[SimpleFeature]>,
    cand_opts: &CandidateOpts,
    score_opts: &ScoreOpts,
    pseudo_opts: &PseudoSpecOpts,
) -> PseudoBuildResult {
    // 1) Enumerate all program-legal candidates with hard guards.
    let idx = PrecursorSearchIndex::build(ds, ms1, cand_opts);
    let pairs = idx.enumerate_pairs(ms1, ms2, cand_opts);  // Vec<(ms2_idx, ms1_idx)>

    // trivial fast-path
    if pairs.is_empty() {
        return PseudoBuildResult {
            assignment: AssignmentResult {
                pairs,
                ms2_best_ms1: vec![None; ms2.len()],
                ms1_to_ms2: vec![Vec::new(); ms1.len()],
            },
            pseudo_spectra: Vec::new(),
        };
    }

    // 2) Score & choose best MS1 per MS2 (this is the **competition** step).
    let ms2_best_ms1 = best_ms1_for_each_ms2_geom(
        ms1,
        ms2,
        &pairs,
        score_opts,
    );
    let ms1_to_ms2 = ms1_to_ms2_map(
        ms1.len(),
        &ms2_best_ms1,
    );

    let assignment = AssignmentResult {
        pairs,
        ms2_best_ms1: ms2_best_ms1.clone(),
        ms1_to_ms2: ms1_to_ms2.clone(),
    };

    // 3) Turn the winner map into "winning pairs" (one MS2 → 0 or 1 MS1).
    let mut winning_pairs: Vec<(usize, usize)> = Vec::new();
    for (ms1_idx, js) in ms1_to_ms2.iter().enumerate() {
        for &ms2_idx in js {
            winning_pairs.push((ms2_idx, ms1_idx));
        }
    }

    // Optional sanity check in debug builds: each MS2 appears at most once.
    debug_assert!({
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for (ms2_idx, _) in &winning_pairs {
            if !seen.insert(ms2_idx) {
                // duplicate MS2 assignment = bug in best_ms1_for_each_ms2_geom
                panic!("Duplicated ms2");
            }
        }
        true
    });

    // 4) Build pseudo spectra from those winning links; grouping by
    //    (precursor, window_group) happens in build_pseudo_spectra_from_pairs.
    let empty_feats: &[SimpleFeature] = &[];
    let feat_slice = features.unwrap_or(empty_feats);

    let pseudo_spectra = build_pseudo_spectra_from_pairs(
        ms1,
        ms2,
        feat_slice,
        &winning_pairs,
        pseudo_opts,
    );

    PseudoBuildResult {
        assignment,
        pseudo_spectra,
    }
}

/// End-to-end, **XIC-based** competitive builder:
///   DIA index + MS1 clusters + MS2 clusters (+ optional features)
///   → candidates → XIC scoring/assignment → pseudo-MS/MS spectra.
///
/// Properties:
///   - Each MS2 cluster participates in **at most one** precursor.
///   - Physical program legality (group, tiles, apex-in-tile) is enforced
///     inside `PrecursorSearchIndex::enumerate_pairs(...)`.
pub fn build_pseudo_spectra_end_to_end_xic(
    ds: &TimsDatasetDIA,
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    features: Option<&[SimpleFeature]>,
    cand_opts: &CandidateOpts,
    xic_opts: &XicScoreOpts,
    pseudo_opts: &PseudoSpecOpts,
) -> PseudoBuildResult {
    // 1) Enumerate all program-legal candidates with hard guards.
    let idx = PrecursorSearchIndex::build(ds, ms1, cand_opts);
    let pairs = idx.enumerate_pairs(ms1, ms2, cand_opts);  // Vec<(ms2_idx, ms1_idx)>

    // trivial fast-path
    if pairs.is_empty() {
        return PseudoBuildResult {
            assignment: AssignmentResult {
                pairs,
                ms2_best_ms1: vec![None; ms2.len()],
                ms1_to_ms2: vec![Vec::new(); ms1.len()],
            },
            pseudo_spectra: Vec::new(),
        };
    }

    // 2) XIC scoring & choose best MS1 per MS2 (this is the **competition** step).
    let ms2_best_ms1 = best_ms1_for_each_ms2_xic(
        ms1,
        ms2,
        &pairs,
        xic_opts,
    );
    let ms1_to_ms2 = ms1_to_ms2_map(
        ms1.len(),
        &ms2_best_ms1,
    );

    let assignment = AssignmentResult {
        pairs,
        ms2_best_ms1: ms2_best_ms1.clone(),
        ms1_to_ms2: ms1_to_ms2.clone(),
    };

    // 3) Turn the winner map into "winning pairs" (one MS2 → 0 or 1 MS1).
    let mut winning_pairs: Vec<(usize, usize)> = Vec::new();
    for (ms1_idx, js) in ms1_to_ms2.iter().enumerate() {
        for &ms2_idx in js {
            winning_pairs.push((ms2_idx, ms1_idx));
        }
    }

    // Optional sanity check in debug builds: each MS2 appears at most once.
    debug_assert!({
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for (ms2_idx, _) in &winning_pairs {
            if !seen.insert(ms2_idx) {
                // duplicate MS2 assignment = bug in best_ms1_for_each_ms2_xic
                panic!("Duplicated ms2");
            }
        }
        true
    });

    // 4) Build pseudo spectra from those winning links; grouping by
    //    (precursor, window_group) happens in build_pseudo_spectra_from_pairs.
    let empty_feats: &[SimpleFeature] = &[];
    let feat_slice = features.unwrap_or(empty_feats);

    let pseudo_spectra = build_pseudo_spectra_from_pairs(
        ms1,
        ms2,
        feat_slice,
        &winning_pairs,
        pseudo_opts,
    );

    PseudoBuildResult {
        assignment,
        pseudo_spectra,
    }
}