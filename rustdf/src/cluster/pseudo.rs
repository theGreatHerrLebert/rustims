//! Build pseudo-DDA spectra from MS1/MS2 clusters and simple isotopic features.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;

use rayon::prelude::*;
use crate::cluster::candidates::CandidateOpts;
use crate::cluster::cluster::ClusterResult1D;
use crate::cluster::feature::SimpleFeature;
use crate::cluster::scoring::{assign_ms2_to_best_ms1_by_xic, enumerate_ms2_ms1_pairs_simple, XicScoreOpts};
use crate::data::dia::TimsDatasetDIA;
// ---------------------------------------------------------------------------
// Basic helpers
// ---------------------------------------------------------------------------

/// Best-effort m/z "center" for a cluster.
///
/// Prefer `mz_fit.mu` when present and sane; otherwise fall back to window mid.
pub fn cluster_mz_mu(c: &ClusterResult1D) -> Option<f32> {
    if let Some(ref f) = c.mz_fit {
        if f.mu.is_finite() && f.mu > 0.0 {
            return Some(f.mu);
        }
    }
    if let Some((lo, hi)) = c.mz_window {
        let mu = 0.5 * (lo + hi);
        if mu.is_finite() && mu > 0.0 {
            return Some(mu);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// One fragment peak in a pseudo-MS/MS spectrum.
#[derive(Clone, Debug)]
pub struct PseudoFragment {
    /// Fragment m/z (cluster center).
    pub mz: f32,
    /// Fragment intensity proxy (currently raw_sum, can be refined later).
    pub intensity: f32,
    /// Index into the `ms2` slice that produced this fragment.
    pub ms2_cluster_index: usize,
    /// Stable cluster ID copied from `ClusterResult1D::cluster_id`.
    pub ms2_cluster_id: u64,
    pub window_group: u32,
}

/// One pseudo-DDA spectrum: precursor + set of fragment peaks.
#[derive(Clone, Debug)]
pub struct PseudoSpectrum {
    pub precursor_mz: f32,
    pub precursor_charge: u8,
    pub rt_apex: f32,
    pub im_apex: f32,

    pub window_group: u32,

    pub feature_id: Option<usize>,
    pub precursor_cluster_indices: Vec<usize>,
    pub precursor_cluster_ids: Vec<u64>,
    pub fragments: Vec<PseudoFragment>,
}

/// Options for building pseudo spectra, diaTracer-style.
#[derive(Clone, Debug)]
pub struct PseudoSpecOpts {
    /// Keep at most this many fragments per spectrum (0 = no limit).
    pub top_n_fragments: usize,
}

impl Default for PseudoSpecOpts {
    fn default() -> Self {
        Self {
            top_n_fragments: 500, // diaTracer-like RF max default
        }
    }
}

/// Precursor identity: either an isotopic SimpleFeature or a single orphan MS1 cluster.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PrecursorKey {
    Feature {
        feature_id: usize,
        window_group: u32,
    },
    OrphanCluster {
        cluster_idx: usize,
        window_group: u32,
    },
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Map MS1 cluster index -> Option<feature_id>.
///
/// Assumes a cluster belongs to at most one SimpleFeature (true for the greedy builder).
fn build_cluster_to_feature_map(n_ms1: usize, features: &[SimpleFeature]) -> Vec<Option<usize>> {
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

/// Derive a precursor apex (RT, IM) from either a feature or an orphan MS1 cluster.
///
/// Feature: weighted average of member cluster apex positions (weights = raw_sum).
/// Orphan: use the single cluster’s rt_fit.mu / im_fit.mu.
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
                // Fallback: just midpoints of feature bounds if everything is degenerate.
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
// Public: build pseudo spectra from MS1/MS2 clusters + features + pairs
// ---------------------------------------------------------------------------

/// Build pseudo-DDA spectra from:
/// - `ms1`: MS1 clusters
/// - `ms2`: MS2 clusters
/// - `features`: isotopic SimpleFeatures built on MS1
/// - `pairs`: candidate links (ms2_idx, ms1_idx) from your candidate search
///
/// Behaviour:
/// - If `ms1_idx` belongs to a feature, use the **feature** as the precursor.
/// - Otherwise, use that MS1 cluster as a degenerate “orphan” precursor.
/// - All MS2 clusters linked to any member (for features) are grouped into one spectrum.
/// - Fragments are sorted by intensity and capped to `top_n_fragments` if >0.
///
/// This is the DIA→pseudo-DDA “glue” that you can feed into an mzML writer.
pub fn build_pseudo_spectra_from_pairs(
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    features: &[SimpleFeature],
    pairs: &[(usize, usize)],
    opts: &PseudoSpecOpts,
) -> Vec<PseudoSpectrum> {
    if ms1.is_empty() || ms2.is_empty() || pairs.is_empty() {
        return Vec::new();
    }

    let cluster_to_feature = build_cluster_to_feature_map(ms1.len(), features);

    let mut grouped: HashMap<PrecursorKey, Vec<usize>> = HashMap::new();

    for &(ms2_idx, ms1_idx) in pairs {
        if ms1_idx >= ms1.len() || ms2_idx >= ms2.len() {
            continue;
        }

        // If an MS2 has no window_group, we should never have seen it as a candidate.
        let g = ms2[ms2_idx]
            .window_group
            .unwrap_or(0); // if 0 ever appears, that’s a bug upstream

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

    let top_n = opts.top_n_fragments;

    let mut spectra: Vec<PseudoSpectrum> = grouped
        .into_par_iter()
        .map(|(key, mut frag_indices)| {
            frag_indices.sort_unstable();
            frag_indices.dedup();

            let (prec_mz, charge, prec_cluster_indices, prec_cluster_ids, wg) = match key {
                PrecursorKey::OrphanCluster { cluster_idx, window_group } => {
                    let c = &ms1[cluster_idx];
                    let mz = cluster_mz_mu(c).unwrap_or(0.0);
                    let z  = 0u8;
                    let idxs = vec![cluster_idx];
                    let ids  = vec![c.cluster_id];
                    (mz, z, idxs, ids, window_group)
                }
                PrecursorKey::Feature { feature_id, window_group } => {
                    let feat = &features[feature_id];
                    let mz   = feat.mz_mono;
                    let z    = feat.charge;
                    let idxs = feat.member_cluster_indices.clone();
                    let ids  = feat.member_cluster_ids.clone();
                    (mz, z, idxs, ids, window_group)
                }
            };

            let (rt_apex, im_apex) =
                precursor_apex_from_feature_or_cluster(key, ms1, features);

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
                        window_group: c2.window_group.unwrap_or(0),
                    })
                })
                .collect();

            frags.sort_unstable_by(|a, b| {
                b.intensity
                    .partial_cmp(&a.intensity)
                    .unwrap_or(Ordering::Equal)
            });
            if top_n > 0 && frags.len() > top_n {
                frags.truncate(top_n);
            }

            PseudoSpectrum {
                precursor_mz: prec_mz,
                precursor_charge: charge,
                rt_apex,
                im_apex,
                window_group: wg,
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

/// High-level entrypoint:
/// 1) enumerate geometric/tile-based candidates
/// 2) refine by XIC score (best precursor per fragment + cutoff)
/// 3) build pseudo-spectra from the surviving pairs
pub fn build_pseudo_spectra_with_xic(
    ds: &TimsDatasetDIA,
    ms1: &[ClusterResult1D],
    ms2: &[ClusterResult1D],
    features: &[SimpleFeature],
    cand_opts: &CandidateOpts,
    xic_opts: &XicScoreOpts,
    pseudo_opts: &PseudoSpecOpts,
) -> (Vec<PseudoSpectrum>, Vec<(usize, usize, f32)>) {
    // 1) geometric / program-based candidates
    let pairs_geom: Vec<(usize, usize)> =
        enumerate_ms2_ms1_pairs_simple(ds, ms1, ms2, cand_opts);

    if pairs_geom.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // 2) XIC scoring + best-precursor-per-fragment + cutoff
    let pairs_scored: Vec<(usize, usize, f32)> =
        assign_ms2_to_best_ms1_by_xic(ms1, ms2, &pairs_geom, xic_opts);

    if pairs_scored.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Drop scores for the pseudo-spectra builder (it just wants indices)
    let final_pairs: Vec<(usize, usize)> = pairs_scored
        .iter()
        .map(|(ms2_idx, ms1_idx, _s)| (*ms2_idx, *ms1_idx))
        .collect();

    // 3) existing glue
    let spectra = build_pseudo_spectra_from_pairs(
        ms1,
        ms2,
        features,
        &final_pairs,
        pseudo_opts,
    );

    (spectra, pairs_scored)
}