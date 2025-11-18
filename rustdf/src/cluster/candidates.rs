use std::cmp::Ordering;
use rayon::prelude::*;

use crate::cluster::cluster::ClusterResult1D;
use crate::cluster::feature::SimpleFeature;
use crate::cluster::pseudo::{
    cluster_mz_mu,
    PrecursorKey,
    PseudoFragment,
    PseudoSpecOpts,
    PseudoSpectrum,
};
use crate::cluster::scoring::{
    jaccard_time,
    CandidateOpts,
    PrecursorSearchIndex,
};
use crate::data::dia::TimsDatasetDIA;

// ---------------------------------------------------------------------------
// Helper: MS1 → feature map
// ---------------------------------------------------------------------------

/// Build a mapping from MS1 cluster index → Option<feature_id>.
///
/// Assumes that each MS1 cluster belongs to at most one SimpleFeature
/// (which should be true for your greedy builder).
fn build_cluster_to_feature_map(
    n_ms1: usize,
    features: &[SimpleFeature],
) -> Vec<Option<usize>> {
    let mut map = vec![None; n_ms1];
    for (f_idx, feat) in features.iter().enumerate() {
        for &ci in &feat.member_cluster_indices {
            if ci < n_ms1 {
                map[ci] = Some(f_idx);
            }
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Helper: precursor apex from feature / orphan
// ---------------------------------------------------------------------------

/// Derive a precursor apex (RT, IM) from either a feature or an orphan MS1 cluster.
///
/// Feature: weighted average of member cluster apex positions (weights = raw_sum).
/// Orphan: use the single cluster’s rt_fit.mu / im_fit.mu.
fn precursor_apex_from_feature_or_cluster(
    key: PrecursorKey,
    ms1: &[ClusterResult1D],
    features: &[SimpleFeature],
) -> (f32, f32) {
    match key {
        PrecursorKey::OrphanCluster { cluster_idx, .. } => {
            let c = &ms1[cluster_idx];
            (c.rt_fit.mu, c.im_fit.mu)
        }
        PrecursorKey::Feature { feature_id, .. } => {
            let feat = &features[feature_id];
            let mut rt_w = 0.0_f64;
            let mut im_w = 0.0_f64;
            let mut wsum = 0.0_f64;

            for &ci in &feat.member_cluster_indices {
                if ci >= ms1.len() {
                    continue;
                }
                let c = &ms1[ci];
                let w = c.raw_sum.max(1.0) as f64;

                if c.rt_fit.mu.is_finite() {
                    rt_w += (c.rt_fit.mu as f64) * w;
                }
                if c.im_fit.mu.is_finite() {
                    im_w += (c.im_fit.mu as f64) * w;
                }
                wsum += w;
            }

            if wsum > 0.0 {
                ((rt_w / wsum) as f32, (im_w / wsum) as f32)
            } else {
                // Fallback: just use bounds midpoints
                let (rt_lo, rt_hi) = feat.rt_bounds;
                let (im_lo, im_hi) = feat.im_bounds;
                (
                    0.5 * ((rt_lo + rt_hi) as f32),
                    0.5 * ((im_lo + im_hi) as f32),
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Core: build pseudo spectra from pairs
// ---------------------------------------------------------------------------

/// Main builder: turn candidate MS1–MS2 pairs into pseudo-DDA spectra.
///
/// `pairs` are (ms2_index, ms1_index) in the same index space as `ms1` / `ms2`.
/// The **window_group** is taken from the MS2 cluster, and becomes part of
/// the `PrecursorKey`, so you never mix fragments from multiple groups for
/// the same feature in one pseudo spectrum.
pub fn build_pseudo_spectra_from_pairs(
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    features: &[SimpleFeature],
    pairs: &[(usize, usize)],     // (ms2_idx, ms1_idx)
    opts: &PseudoSpecOpts,
) -> Vec<PseudoSpectrum> {
    if ms1.is_empty() || ms2.is_empty() || pairs.is_empty() {
        return Vec::new();
    }

    // 1) Map MS1 cluster index -> feature_id (if any)
    let cluster_to_feature = build_cluster_to_feature_map(ms1.len(), features);

    // 2) Group all MS2 indices by precursor key (feature/orphan + window_group)
    use std::collections::HashMap;
    let mut grouped: HashMap<PrecursorKey, Vec<usize>> = HashMap::new();

    for &(ms2_idx, ms1_idx) in pairs {
        if ms1_idx >= ms1.len() || ms2_idx >= ms2.len() {
            continue;
        }

        let c2 = &ms2[ms2_idx];
        let g = match c2.window_group {
            Some(g) => g,
            None => continue, // should not happen if candidates were built sensibly
        };

        let key = match cluster_to_feature[ms1_idx] {
            Some(fid) => PrecursorKey::Feature {
                feature_id: fid,
                window_group: g,
            },
            None => PrecursorKey::OrphanCluster {
                cluster_idx: ms1_idx,
                window_group: g,
            },
        };
        grouped.entry(key).or_default().push(ms2_idx);
    }

    if grouped.is_empty() {
        return Vec::new();
    }

    // 3) Build pseudo spectra in parallel over precursors
    let top_n = opts.top_n_fragments;

    let mut spectra: Vec<PseudoSpectrum> = grouped
        .into_par_iter()
        .map(|(key, mut frag_indices)| {
            // Dedup fragment cluster indices per precursor
            frag_indices.sort_unstable();
            frag_indices.dedup();

            // ---- Precursor summary ----
            let (prec_mz, charge, prec_cluster_indices, prec_cluster_ids) = match key {
                PrecursorKey::OrphanCluster { cluster_idx, .. } => {
                    let c = &ms1[cluster_idx];
                    let mz = cluster_mz_mu(c).unwrap_or(0.0);
                    let z  = 0u8; // unknown; you can change to 1 if you prefer
                    let ids = vec![c.cluster_id];
                    let idxs = vec![cluster_idx];
                    (mz, z, idxs, ids)
                }
                PrecursorKey::Feature { feature_id, .. } => {
                    let feat = &features[feature_id];
                    let mz = feat.mz_mono;      // monoisotopic m/z from feature builder
                    let z  = feat.charge;
                    let idxs = feat.member_cluster_indices.clone();
                    let ids  = feat.member_cluster_ids.clone();
                    (mz, z, idxs, ids)
                }
            };

            // Apex RT/IM
            let (rt_apex, im_apex) =
                precursor_apex_from_feature_or_cluster(key, ms1, features);

            // ---- Fragment list ----
            let mut frags: Vec<PseudoFragment> = frag_indices
                .into_iter()
                .filter_map(|j| {
                    let c2 = &ms2[j];
                    let mz = match cluster_mz_mu(c2) {
                        Some(m) if m.is_finite() && m > 0.0 => m,
                        _ => return None,
                    };
                    let intensity = c2.raw_sum.max(0.0);
                    Some(PseudoFragment {
                        mz,
                        intensity,
                        ms2_cluster_index: j,
                        ms2_cluster_id: c2.cluster_id,
                    })
                })
                .collect();

            // Sort by fragment intensity (desc) and cap at top N if requested
            frags.sort_unstable_by(|a, b| {
                b.intensity
                    .partial_cmp(&a.intensity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if top_n > 0 && frags.len() > top_n {
                frags.truncate(top_n);
            }

            PseudoSpectrum {
                precursor_mz: prec_mz,
                precursor_charge: charge,
                rt_apex,
                im_apex,
                window_group: match key {
                    PrecursorKey::Feature { window_group, .. } => window_group,
                    PrecursorKey::OrphanCluster { window_group, .. } => window_group,
                },
                feature_id: match key {
                    PrecursorKey::Feature { feature_id, .. } => Some(feature_id),
                    PrecursorKey::OrphanCluster { .. } => None,
                },
                precursor_cluster_indices: prec_cluster_indices,
                precursor_cluster_ids: prec_cluster_ids,
                fragments: frags,
            }
        })
        .collect();

    // (optional) sort spectra by RT apex, then m/z
    spectra.sort_unstable_by(|a, b| {
        a.rt_apex
            .partial_cmp(&b.rt_apex)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                a.precursor_mz
                    .partial_cmp(&b.precursor_mz)
                    .unwrap_or(Ordering::Equal)
            })
    });

    spectra
}

// ---------------------------------------------------------------------------
// Assignment + end-to-end builder (unchanged API, but now uses group-aware keys)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AssignmentResult {
    /// All enumerated pairs (ms2_idx, ms1_idx) after your hard guards.
    pub pairs: Vec<(usize, usize)>,
    /// For each MS2 j, the chosen MS1 index (or None if no candidate).
    pub ms2_best_ms1: Vec<Option<usize>>,
    /// For each MS1 i, the list of MS2 indices assigned to it.
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

/// NON-COMPETITIVE builder: mainly for debugging.
///
/// Links *all* MS1–MS2 pairs that
///   - are program-legal (same window group, isolation & scans),
///   - satisfy candidate guards (RT/IM overlap etc.),
/// and then groups them into pseudo-spectra without any competition.
///
/// CONSEQUENCE:
///   - An MS2 cluster may contribute to **multiple** precursors.
///   - Use `build_pseudo_spectra_end_to_end` for a competitive, 1:1 assignment.
pub fn build_pseudo_spectra_all_pairs(
    ds: &TimsDatasetDIA,
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    features: Option<&[SimpleFeature]>,
    pseudo_opts: &PseudoSpecOpts,
) -> Vec<PseudoSpectrum> {
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
    };

    let idx = PrecursorSearchIndex::build(ds, ms1, &cand_opts);
    let pairs = idx.enumerate_pairs(ms1, ms2, &cand_opts);

    let empty_feats: &[SimpleFeature] = &[];
    let feat_slice = features.unwrap_or(empty_feats);

    build_pseudo_spectra_from_pairs(
        ms1,
        ms2,
        feat_slice,
        &pairs,
        pseudo_opts,
    )
}

/// End-to-end, **competitive** builder:
///   DIA index + MS1 clusters + MS2 clusters (+ optional features)
///   → candidates → scoring/assignment → pseudo-MS/MS spectra.
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
    let ms2_best_ms1 = best_ms1_for_each_ms2(
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
                // duplicate MS2 assignment = bug in best_ms1_for_each_ms2
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

// ---------------------------------------------------------------------------
// Scoring (unchanged; left here for completeness)
// ---------------------------------------------------------------------------

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
fn build_features(ms1: &ClusterResult1D, ms2: &ClusterResult1D, opts: &ScoreOpts) -> PairFeatures {
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
fn exp_decay(delta: f32, scale: f32) -> f32 {
    // Monotone in [0,∞): 1 at 0, then decays smoothly
    if !delta.is_finite() || !scale.is_finite() || scale <= 0.0 { return 0.0; }
    (-delta / scale).exp()
}

#[inline]
fn safe_log1p(x: f32) -> f32 {
    if x.is_finite() && x >= 0.0 { (1.0 + x as f64).ln() as f32 } else { 0.0 }
}