use std::collections::HashMap;
use std::sync::Arc;
use rayon::prelude::*;
use crate::cluster::cluster::ClusterResult1D;
use crate::cluster::feature::SimpleFeature;
use crate::cluster::pseudo::{PseudoSpecOpts, PseudoSpectrum, build_pseudo_spectra_from_pairs, cluster_mz_mu, PseudoFragment};
use crate::cluster::scoring::{assign_ms2_to_best_ms1_by_xic, jaccard_time, ms1_to_ms2_map, query_precursor_scored, query_precursors_scored_par, MatchScoreMode, PrecursorLike, PrecursorSearchIndex, ScoredHit, XicScoreOpts};
use crate::data::dia::{DiaIndex, TimsDatasetDIA};
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

#[derive(Clone, Debug)]
pub struct FragmentGroupIndex {
    /// MS2 indices belonging to this WG, sorted by rt_apex.
    pub ms2_indices: Vec<usize>,
    /// rt_fit.mu (or RT midpoint fallback) for those indices, same order as `ms2_indices`.
    pub rt_apex: Vec<f32>,
}

/// Axis-aligned query knobs for fragment candidates.
#[derive(Clone, Debug)]
pub struct FragmentQueryOpts {
    /// Maximum allowed |rt_apex(prec) - rt_apex(ms2)| in seconds (None = no RT slice).
    pub max_rt_apex_delta_sec: Option<f32>,
    /// Maximum allowed |im_apex(prec) - im_apex(ms2)| in global scans.
    pub max_scan_apex_delta: Option<usize>,
    /// Minimum IM overlap in scans between precursor and fragment.
    pub min_im_overlap_scans: usize,
    /// Require at least one shared DIA tile (ProgramSlice) between precursor and fragment.
    pub require_tile_compat: bool,
    /// If true, drop fragments whose own selection (mz, scan)
    pub reject_frag_inside_precursor_tile: bool,
}

impl Default for FragmentQueryOpts {
    fn default() -> Self {
        Self {
            max_rt_apex_delta_sec: Some(2.0),
            max_scan_apex_delta: Some(6),
            min_im_overlap_scans: 1,
            require_tile_compat: true,
            reject_frag_inside_precursor_tile: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FragmentIndex<'a> {
    dia_index: Arc<DiaIndex>,

    /// Borrowed MS2 clusters – single owner elsewhere.
    ms2: &'a [ClusterResult1D],

    // per-MS2 metadata (global index over all groups)
    ms2_im_mu: Vec<f32>,
    ms2_im_windows: Vec<(usize, usize)>,
    ms2_keep: Vec<bool>,
    /// Stable cluster IDs, same indexing as ms2_* vectors.
    ms2_cluster_ids: Vec<u64>,

    /// Representative m/z per MS2 cluster (μ or window midpoint).
    ms2_mz_mu: Vec<f32>,

    /// group -> RT-sorted MS2
    by_group: HashMap<u32, FragmentGroupIndex>,
}

impl<'a> FragmentIndex<'a> {
    /// Build RT-sorted fragment index per window group.
    ///
    /// Only uses the MS2-relevant fields of CandidateOpts:
    ///   - min_raw_sum
    ///   - ms2_rt_guard_sec
    ///   - max_ms2_rt_span_sec
    pub fn build(
        dia_index: Arc<DiaIndex>,
        ms2: &'a [ClusterResult1D],
        opts: &CandidateOpts,
    ) -> Self {
        let frame_time = &dia_index.frame_time;

        // 1) Absolute MS2 time bounds (seconds) with fallback for missing frame_ids_used
        let ms2_time_bounds: Vec<(f64, f64)> = ms2
            .par_iter()
            .map(|c| {
                let mut t_lo = f64::INFINITY;
                let mut t_hi = f64::NEG_INFINITY;

                // Preferred: use explicit frame_ids_used if present
                if !c.frame_ids_used.is_empty() {
                    for &fid in &c.frame_ids_used {
                        if let Some(&t) = frame_time.get(&fid) {
                            if t < t_lo { t_lo = t; }
                            if t > t_hi { t_hi = t; }
                        }
                    }
                }

                // Fallback: infer from rt_window if we have no usable times yet
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

                (t_lo, t_hi)
            })
            .collect();

        // 2) Keep mask – unchanged, but now ms2_time_bounds is actually finite
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

        // 3) IM window + IM apex, with fallback to midpoint – unchanged
        let ms2_im_windows: Vec<(usize, usize)> =
            ms2.iter().map(|c| c.im_window).collect();

        let ms2_im_mu: Vec<f32> = ms2
            .iter()
            .map(|c| {
                let mu = c.im_fit.mu;
                if mu.is_finite() && mu > 0.0 {
                    mu
                } else {
                    let (lo, hi) = c.im_window;
                    if hi > lo {
                        ((lo + hi) as f32) * 0.5
                    } else {
                        f32::NAN
                    }
                }
            })
            .collect();

        let ms2_cluster_ids: Vec<u64> =
            ms2.iter().map(|c| c.cluster_id).collect();

        // 4) RT apex (seconds) with fallback – unchanged
        let ms2_rt_apex: Vec<f32> = ms2
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mu = c.rt_fit.mu;
                if mu.is_finite() && mu > 0.0 {
                    mu
                } else {
                    let (t0, t1) = ms2_time_bounds[i];
                    if t0.is_finite() && t1.is_finite() && t1 > t0 {
                        ((t0 + t1) * 0.5) as f32
                    } else {
                        f32::NAN
                    }
                }
            })
            .collect();

        // 4.5) Representative m/z – unchanged
        let ms2_mz_mu: Vec<f32> = ms2
            .iter()
            .map(|c| {
                if let Some(m) = cluster_mz_mu(c) {
                    if m.is_finite() && m > 0.0 {
                        m
                    } else {
                        match c.mz_window {
                            Some((lo, hi)) if lo.is_finite() && hi.is_finite() && hi > lo => {
                                0.5 * (lo + hi) as f32
                            }
                            _ => f32::NAN,
                        }
                    }
                } else {
                    match c.mz_window {
                        Some((lo, hi)) if lo.is_finite() && hi.is_finite() && hi > lo => {
                            0.5 * (lo + hi) as f32
                        }
                        _ => f32::NAN,
                    }
                }
            })
            .collect();

        // 5) Group and sort by RT apex – unchanged
        let mut by_group_raw: HashMap<u32, Vec<(f32, usize)>> = HashMap::new();

        for (j, c2) in ms2.iter().enumerate() {
            if !ms2_keep[j] {
                continue;
            }
            let g = match c2.window_group {
                Some(g) => g,
                None => continue,
            };

            let rt = ms2_rt_apex[j];
            if !rt.is_finite() {
                continue;
            }

            by_group_raw.entry(g).or_default().push((rt, j));
        }

        let mut by_group: HashMap<u32, FragmentGroupIndex> = HashMap::new();
        for (g, mut v) in by_group_raw {
            v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let (rt_apex, ms2_indices): (Vec<f32>, Vec<usize>) =
                v.into_iter().map(|(rt, j)| (rt, j)).unzip();
            by_group.insert(g, FragmentGroupIndex { ms2_indices, rt_apex });
        }

        FragmentIndex {
            dia_index,
            ms2, // <-- no clone
            ms2_im_mu,
            ms2_im_windows,
            ms2_keep,
            ms2_cluster_ids,
            ms2_mz_mu,
            by_group,
        }
    }

    fn precursor_rt_apex_sec(&self, prec: &ClusterResult1D) -> Option<f32> {
        // 1) use fit if it looks sane
        let mu = prec.rt_fit.mu;
        if mu.is_finite() && mu > 0.0 {
            return Some(mu);
        }

        let ft = &self.dia_index.frame_time;

        // 2) fallback: derive from frame_ids_used
        let mut sum = 0.0f64;
        let mut n = 0usize;
        for fid in &prec.frame_ids_used {
            if let Some(&t) = ft.get(fid) {
                if t.is_finite() {
                    sum += t;
                    n += 1;
                }
            }
        }
        if n > 0 {
            return Some((sum / n as f64) as f32);
        }

        // 3) final fallback: infer from rt_window range using frame IDs
        let (rt_lo, rt_hi) = prec.rt_window;
        if rt_hi >= rt_lo {
            let mut sum = 0.0f64;
            let mut n = 0usize;
            for fid in rt_lo as u32..=rt_hi as u32 {
                if let Some(&t) = ft.get(&fid) {
                    if t.is_finite() {
                        sum += t;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                return Some((sum / n as f64) as f32);
            }
        }

        None
    }

    pub fn query_precursor(
        &self,
        prec: &ClusterResult1D,
        groups: Option<&[u32]>,
        opts: &FragmentQueryOpts,
    ) -> Vec<u64> {
        let mut out = Vec::new();

        // --- precursor m/z (for group selection + "inside selection" test) ---
        let prec_mz = if let Some(m) = cluster_mz_mu(prec) {
            if m.is_finite() && m > 0.0 {
                m
            } else {
                match prec.mz_window {
                    Some((lo, hi)) if lo.is_finite() && hi.is_finite() && hi > lo => {
                        0.5 * (lo + hi)
                    }
                    _ => return out, // no usable m/z
                }
            }
        } else {
            match prec.mz_window {
                Some((lo, hi)) if lo.is_finite() && hi.is_finite() && hi > lo => {
                    0.5 * (lo + hi)
                }
                _ => return out,
            }
        };

        // --- precursor RT apex (SECONDS!) ---
        let prec_rt = match self.precursor_rt_apex_sec(prec) {
            Some(t) => t,
            None => return out,
        };

        // --- precursor IM apex (scan space) ---
        let prec_im = if prec.im_fit.mu.is_finite() {
            prec.im_fit.mu
        } else {
            let (lo, hi) = prec.im_window;
            if hi > lo {
                ((lo + hi) as f32) * 0.5
            } else {
                return out;
            }
        };

        let prec_im_win = prec.im_window;
        let prec_scan_win = (prec_im_win.0 as u32, prec_im_win.1 as u32);

        // Decide which groups to query
        let groups: Vec<u32> = match groups {
            Some(gs) if !gs.is_empty() => gs.to_vec(),
            _ => self.dia_index.groups_for_precursor(prec_mz, prec_im),
        };
        if groups.is_empty() {
            return Vec::new();
        }

        let require_tile  = opts.require_tile_compat;
        let reject_inside = opts.reject_frag_inside_precursor_tile;

        for g in groups {
            let fg = match self.by_group.get(&g) {
                Some(fg) => fg,
                None => continue,
            };

            // RT slice via binary search (fg.rt_apex is already in seconds)
            let (lo_idx, hi_idx) = if let Some(dt) = opts.max_rt_apex_delta_sec {
                if dt > 0.0 {
                    let lo_t = prec_rt - dt;
                    let hi_t = prec_rt + dt;
                    (
                        lower_bound(&fg.rt_apex, lo_t),
                        upper_bound(&fg.rt_apex, hi_t),
                    )
                } else {
                    (0, fg.ms2_indices.len())
                }
            } else {
                (0, fg.ms2_indices.len())
            };

            // Program slices (tiles) for this group; we'll inspect them directly.
            let slices = self.dia_index.program_slices_for_group(g);

            'ms2_loop: for k in lo_idx..hi_idx {
                let j = fg.ms2_indices[k];
                if !self.ms2_keep[j] {
                    continue;
                }

                // IM window overlap
                let im2 = self.ms2_im_windows[j];
                let frag_scan_win = (im2.0 as u32, im2.1 as u32);

                let im_overlap = {
                    let lo = prec_im_win.0.max(im2.0);
                    let hi = prec_im_win.1.min(im2.1);
                    hi.saturating_sub(lo).saturating_add(1)
                };
                if im_overlap < opts.min_im_overlap_scans {
                    continue;
                }

                // IM apex delta
                if let Some(max_d) = opts.max_scan_apex_delta {
                    let s2 = self.ms2_im_mu[j];
                    if s2.is_finite() {
                        let d = (prec_im - s2).abs() as f32;
                        if d > max_d as f32 {
                            continue 'ms2_loop;
                        }
                    } else {
                        continue 'ms2_loop;
                    }
                }

                // --- Tile compatibility + "reject inside precursor-selection" guard ---
                if require_tile || reject_inside {
                    let mut tile_compatible   = false;
                    let mut inside_prec_tile  = false;
                    let frag_mz = self.ms2_mz_mu[j];

                    for s in &slices {
                        let tile_scans = (s.scan_lo, s.scan_hi);

                        // Do both precursor and fragment overlap this tile in scan space?
                        let prec_hits_tile = ranges_overlap_u32(prec_scan_win, tile_scans);
                        let frag_hits_tile = ranges_overlap_u32(frag_scan_win, tile_scans);

                        if !(prec_hits_tile && frag_hits_tile) {
                            continue;
                        }

                        tile_compatible = true;

                        if reject_inside && frag_mz.is_finite() {
                            // Precursor m/z inside this tile's isolation window?
                            let prec_in_tile =
                                prec_mz >= s.mz_lo as f32 && prec_mz <= s.mz_hi as f32;
                            // Fragment m/z inside the same isolation window?
                            let frag_in_tile =
                                frag_mz >= s.mz_lo as f32 && frag_mz <= s.mz_hi as f32;

                            if prec_in_tile && frag_in_tile {
                                inside_prec_tile = true;
                                // No need to inspect more tiles for this fragment;
                                // we already know it's "inside precursor selection".
                                break;
                            }
                        }
                    }

                    // 1) If we require tile compatibility: must have *some* shared scan tile
                    if require_tile && !tile_compatible {
                        continue 'ms2_loop;
                    }

                    // 2) If we reject fragments inside precursor tile: drop them.
                    if reject_inside && inside_prec_tile {
                        continue 'ms2_loop;
                    }
                }

                // Survived all guards: keep this fragment for the precursor.
                out.push(self.ms2_cluster_ids[j]);
            }
        }

        out.sort_unstable();
        out.dedup();
        out
    }

    /// Parallel batch query: for each precursor, return a Vec of MS2 cluster_ids.
    pub fn query_precursors_par(
        &self,
        precursors: &[ClusterResult1D],
        opts: &FragmentQueryOpts,
        num_threads: usize,
    ) -> Vec<Vec<u64>> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        pool.install(|| {
            precursors
                .par_iter()
                .map(|prec| self.query_precursor(prec, None, opts))
                .collect()
        })
    }

    /// Enumerate candidate MS2 indices for a thin precursor.
    ///
    /// Returns indices into `self.ms2`.
    pub fn enumerate_candidates_for_precursor_thin(
        &self,
        prec: &ThinPrecursor,
        window_groups: Option<&[u32]>,
        opts: &FragmentQueryOpts,
    ) -> Vec<usize> {
        let mut out = Vec::<usize>::new();

        let prec_mz      = prec.mz_mu;
        let prec_rt      = prec.rt_mu;
        let prec_im      = prec.im_mu;
        let prec_im_win  = prec.im_window;
        let prec_scan_win = (prec_im_win.0 as u32, prec_im_win.1 as u32);

        // Decide which groups to query
        let groups: Vec<u32> = match window_groups {
            Some(gs) if !gs.is_empty() => gs.to_vec(),
            _ => self.dia_index.groups_for_precursor(prec_mz, prec_im),
        };
        if groups.is_empty() {
            return out;
        }

        let require_tile  = opts.require_tile_compat;
        let reject_inside = opts.reject_frag_inside_precursor_tile;

        for g in groups {
            let fg = match self.by_group.get(&g) {
                Some(fg) => fg,
                None => continue,
            };

            // RT slice via binary search (fg.rt_apex is already in seconds)
            let (lo_idx, hi_idx) = if let Some(dt) = opts.max_rt_apex_delta_sec {
                if dt > 0.0 {
                    let lo_t = prec_rt - dt;
                    let hi_t = prec_rt + dt;
                    (
                        lower_bound(&fg.rt_apex, lo_t),
                        upper_bound(&fg.rt_apex, hi_t),
                    )
                } else {
                    (0, fg.ms2_indices.len())
                }
            } else {
                (0, fg.ms2_indices.len())
            };

            // Program slices (tiles) for this group; we'll inspect them directly.
            let slices = self.dia_index.program_slices_for_group(g);

            'ms2_loop: for k in lo_idx..hi_idx {
                let j = fg.ms2_indices[k];
                if !self.ms2_keep[j] {
                    continue;
                }

                // IM window overlap
                let im2 = self.ms2_im_windows[j];
                let frag_scan_win = (im2.0 as u32, im2.1 as u32);

                let im_overlap = {
                    let lo = prec_im_win.0.max(im2.0);
                    let hi = prec_im_win.1.min(im2.1);
                    hi.saturating_sub(lo).saturating_add(1)
                };
                if im_overlap < opts.min_im_overlap_scans {
                    continue;
                }

                // IM apex delta
                if let Some(max_d) = opts.max_scan_apex_delta {
                    let s2 = self.ms2_im_mu[j];
                    if s2.is_finite() {
                        let d = (prec_im - s2).abs() as f32;
                        if d > max_d as f32 {
                            continue 'ms2_loop;
                        }
                    } else {
                        continue 'ms2_loop;
                    }
                }

                // --- Tile compatibility + "reject inside precursor-selection" guard ---
                if require_tile || reject_inside {
                    let mut tile_compatible   = false;
                    let mut inside_prec_tile  = false;
                    let frag_mz = self.ms2_mz_mu[j];

                    for s in &slices {
                        let tile_scans = (s.scan_lo, s.scan_hi);

                        // Do both precursor and fragment overlap this tile in scan space?
                        let prec_hits_tile = ranges_overlap_u32(prec_scan_win, tile_scans);
                        let frag_hits_tile = ranges_overlap_u32(frag_scan_win, tile_scans);

                        if !(prec_hits_tile && frag_hits_tile) {
                            continue;
                        }

                        tile_compatible = true;

                        if reject_inside && frag_mz.is_finite() {
                            // Precursor m/z inside this tile's isolation window?
                            let prec_in_tile =
                                prec_mz >= s.mz_lo as f32 && prec_mz <= s.mz_hi as f32;
                            // Fragment m/z inside the same isolation window?
                            let frag_in_tile =
                                frag_mz >= s.mz_lo as f32 && frag_mz <= s.mz_hi as f32;

                            if prec_in_tile && frag_in_tile {
                                inside_prec_tile = true;
                                break;
                            }
                        }
                    }

                    if require_tile && !tile_compatible {
                        continue 'ms2_loop;
                    }
                    if reject_inside && inside_prec_tile {
                        continue 'ms2_loop;
                    }
                }

                out.push(j);
            }
        }

        out.sort_unstable();
        out.dedup();
        out
    }

    /// Enumerate candidates for many thin precursors.
    ///
    /// Keeps the same length as `precs`: `None` ⇒ empty candidate list.
    fn enumerate_candidates_for_precursors_thin(
        &self,
        precs: &[Option<ThinPrecursor>],
        opts: &FragmentQueryOpts,
    ) -> Vec<Vec<usize>> {
        precs
            .iter()
            .map(|maybe_prec| {
                if let Some(ref p) = maybe_prec {
                    self.enumerate_candidates_for_precursor_thin(p, None, opts)
                } else {
                    Vec::new()
                }
            })
            .collect()
    }

    /// Enumerate candidate MS2 indices for a given SimpleFeature.
    ///
    /// Uses the representative member cluster (highest raw_sum) of the feature
    /// and reuses the standard precursor enumeration logic.
    ///
    /// Returns indices into `self.ms2`.
    pub fn enumerate_candidates_for_feature(
        &self,
        feat: &SimpleFeature,
        opts: &FragmentQueryOpts,
    ) -> Vec<usize> {
        let prec_cluster = match feature_representative_cluster(feat) {
            Some(c) => c,
            None => return Vec::new(),
        };
        let thin = match self.make_thin_precursor(prec_cluster) {
            Some(t) => t,
            None => return Vec::new(),
        };
        // NOTE: groups always decided by the index.
        self.enumerate_candidates_for_precursor_thin(&thin, None, opts)
    }

    pub fn query_precursor_scored(
        &self,
        prec: &ClusterResult1D,
        window_groups: Option<&[u32]>,
        opts: &FragmentQueryOpts,
        mode: MatchScoreMode,
        geom_opts: &ScoreOpts,
        xic_opts: &XicScoreOpts,
        min_score: f32,
    ) -> Vec<ScoredHit> {
        // Build thin precursor for enumeration only.
        let thin = match self.make_thin_precursor(prec) {
            Some(t) => t,
            None => return Vec::new(),
        };

        let candidate_ids =
            self.enumerate_candidates_for_precursor_thin(&thin, window_groups, opts);

        // Scoring still uses the full cluster as PrecursorLike::Cluster.
        query_precursor_scored(
            PrecursorLike::Cluster(prec),
            self.ms2,
            &candidate_ids,
            mode,
            geom_opts,
            xic_opts,
            min_score,
        )
    }

    pub fn query_precursors_scored_par(
        &self,
        precs: &[ClusterResult1D],
        opts: &FragmentQueryOpts,
        mode: MatchScoreMode,
        geom_opts: &ScoreOpts,
        xic_opts: &XicScoreOpts,
        min_score: f32,
    ) -> Vec<Vec<ScoredHit>> {
        // 1) Build thin precursors (Option → keep alignment with `precs`).
        let thin_precs: Vec<Option<ThinPrecursor>> = precs
            .iter()
            .map(|c| self.make_thin_precursor(c))
            .collect();

        // 2) Enumerate candidates using only the thin data.
        let all_candidates = self.enumerate_candidates_for_precursors_thin(&thin_precs, opts);

        // 3) Build PrecursorLike slice referencing the full clusters for scoring.
        let prec_like: Vec<PrecursorLike<'_>> =
            precs.iter().map(|c| PrecursorLike::Cluster(c)).collect();

        // 4) Score in parallel (unchanged).
        query_precursors_scored_par(
            &prec_like,
            self.ms2,
            &all_candidates,
            mode,
            geom_opts,
            xic_opts,
            min_score,
        )
    }

    fn enumerate_candidates_for_features(
        &self,
        feats: &[SimpleFeature],
        opts: &FragmentQueryOpts,
    ) -> Vec<Vec<usize>> {
        feats
            .iter()
            .map(|feat| {
                let prec_cluster = match feature_representative_cluster(feat) {
                    Some(c) => c,
                    None => return Vec::new(),
                };
                let thin = match self.make_thin_precursor(prec_cluster) {
                    Some(t) => t,
                    None => return Vec::new(),
                };
                // NOTE: groups always decided by the index.
                self.enumerate_candidates_for_precursor_thin(&thin, None, opts)
            })
            .collect()
    }

    /// Score a single SimpleFeature precursor against all physically plausible
    /// fragment candidates, as determined internally by the FragmentIndex.
    ///
    /// This is the feature twin of `query_precursor_scored`, but it does *not*
    /// require the caller to pass window groups or candidate indices.
    pub fn query_feature_scored(
        &self,
        feat: &SimpleFeature,
        opts: &FragmentQueryOpts,
        mode: MatchScoreMode,
        geom_opts: &ScoreOpts,
        xic_opts: &XicScoreOpts,
        min_score: f32,
    ) -> Vec<ScoredHit> {
        let candidate_ids = self.enumerate_candidates_for_feature(feat, opts);

        query_precursor_scored(
            PrecursorLike::Feature(feat),
            self.ms2,
            &candidate_ids,
            mode,
            geom_opts,
            xic_opts,
            min_score,
        )
    }

    /// Parallel scoring for many SimpleFeatures.
    ///
    /// - Candidate enumeration is done *inside* the index.
    /// - `feats[i]` is scored against the MS2 candidates inferred for feat i.
    pub fn query_features_scored_par(
        &self,
        feats: &[SimpleFeature],
        opts: &FragmentQueryOpts,
        mode: MatchScoreMode,
        geom_opts: &ScoreOpts,
        xic_opts: &XicScoreOpts,
        min_score: f32,
    ) -> Vec<Vec<ScoredHit>> {
        let all_candidates = self.enumerate_candidates_for_features(feats, opts);

        let prec_like: Vec<PrecursorLike<'_>> =
            feats.iter().map(|f| PrecursorLike::Feature(f)).collect();

        query_precursors_scored_par(
            &prec_like,
            self.ms2,
            &all_candidates,
            mode,
            geom_opts,
            xic_opts,
            min_score,
        )
    }

    /// Build a thin precursor representation used for candidate enumeration.
    ///
    /// This folds all the "how do we get RT apex / m/z apex / IM apex?" logic
    /// into one place and drops everything else (raw points, axes, …).
    pub fn make_thin_precursor(&self, prec: &ClusterResult1D) -> Option<ThinPrecursor> {
        // --- precursor m/z (same rules as before) ---
        let prec_mz = if let Some(m) = cluster_mz_mu(prec) {
            if m.is_finite() && m > 0.0 {
                m
            } else {
                match prec.mz_window {
                    Some((lo, hi)) if lo.is_finite() && hi.is_finite() && hi > lo => {
                        0.5 * (lo + hi)
                    }
                    _ => return None, // no usable m/z
                }
            }
        } else {
            match prec.mz_window {
                Some((lo, hi)) if lo.is_finite() && hi.is_finite() && hi > lo => {
                    0.5 * (lo + hi)
                }
                _ => return None,
            }
        };

        // --- RT apex in seconds (uses existing helper for fallback) ---
        let rt_mu = match self.precursor_rt_apex_sec(prec) {
            Some(t) => t,
            None => return None,
        };

        // --- IM apex + window (scan space) ---
        let im_window = prec.im_window;
        let im_mu = if prec.im_fit.mu.is_finite() {
            prec.im_fit.mu
        } else {
            let (lo, hi) = im_window;
            if hi > lo {
                ((lo + hi) as f32) * 0.5
            } else {
                return None;
            }
        };

        Some(ThinPrecursor {
            mz_mu: prec_mz,
            rt_mu,
            im_mu,
            im_window,
        })
    }
    #[inline]
    pub fn ms2_slice(&self) -> &[ClusterResult1D] {
        self.ms2
    }
}

#[derive(Clone, Debug)]
pub struct ThinPrecursor {
    /// Precursor m/z apex (Da), already cleaned / representative.
    pub mz_mu: f32,

    /// RT apex in *seconds*.
    pub rt_mu: f32,

    /// IM apex in scan space.
    pub im_mu: f32,

    /// IM window (scan indices) used for overlap / tile tests.
    pub im_window: (usize, usize),
}



// Simple helpers (local)

fn lower_bound(xs: &[f32], x: f32) -> usize {
    let mut lo = 0;
    let mut hi = xs.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if xs[mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline]
fn ranges_overlap_u32(a: (u32, u32), b: (u32, u32)) -> bool {
    let lo = a.0.max(b.0);
    let hi = a.1.min(b.1);
    hi >= lo
}

fn upper_bound(xs: &[f32], x: f32) -> usize {
    let mut lo = 0;
    let mut hi = xs.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Convert an MS2 ClusterResult1D into a PseudoFragment.
/// Return None if the cluster is unusable (no m/z).
pub fn fragment_from_cluster(c: &ClusterResult1D) -> Option<PseudoFragment> {
    // mz: prefer fitted μ, fallback to window mid
    let mz = if let Some(fit) = &c.mz_fit {
        if fit.mu.is_finite() && fit.mu > 0.0 {
            fit.mu
        } else if let Some((lo, hi)) = c.mz_window {
            0.5 * (lo + hi)
        } else {
            return None;
        }
    } else if let Some((lo, hi)) = c.mz_window {
        0.5 * (lo + hi)
    } else {
        return None;
    };

    if !mz.is_finite() {
        return None;
    }

    // intensity: choose raw_sum as best available proxy
    let intensity = c.raw_sum;

    Some(PseudoFragment {
        mz,
        intensity,
        ms2_cluster_index: 0,
        ms2_cluster_id: c.cluster_id,
        window_group: c.window_group.unwrap_or(0),
    })
}

fn feature_representative_cluster<'a>(feat: &'a SimpleFeature) -> Option<&'a ClusterResult1D> {
    if feat.member_clusters.is_empty() {
        return None;
    }

    feat.member_clusters
        .iter()
        .max_by(|a, b| a.raw_sum.partial_cmp(&b.raw_sum).unwrap_or(std::cmp::Ordering::Equal))
}