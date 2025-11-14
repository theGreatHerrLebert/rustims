//! Build pseudo-DDA spectra from MS1/MS2 clusters and simple isotopic features.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;

use rayon::prelude::*;
use crate::cluster::cluster::ClusterResult1D;
use crate::cluster::feature::SimpleFeature;

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
}

/// One pseudo-DDA spectrum: precursor + set of fragment peaks.
#[derive(Clone, Debug)]
pub struct PseudoSpectrum {
    /// Precursor m/z (feature mono if available, otherwise MS1 cluster center).
    pub precursor_mz: f32,
    /// Precursor charge (0 = unknown for orphan clusters).
    pub precursor_charge: u8,
    /// RT apex in seconds.
    pub rt_apex: f32,
    /// IM apex in global scan units.
    pub im_apex: f32,

    /// If this precursor came from a SimpleFeature, which one; otherwise None.
    pub feature_id: Option<usize>,

    /// MS1 clusters that define this precursor (feature members or single orphan MS1).
    pub precursor_cluster_indices: Vec<usize>,
    pub precursor_cluster_ids: Vec<u64>,

    /// Fragment peaks (after filtering / top-N selection).
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
    Feature(usize),
    OrphanCluster(usize),
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
fn precursor_apex_from_feature_or_cluster(
    key: PrecursorKey,
    ms1: &[ClusterResult1D],
    features: &[SimpleFeature],
) -> (f32, f32) {
    match key {
        PrecursorKey::OrphanCluster(i) => {
            let c = &ms1[i];
            (c.rt_fit.mu, c.im_fit.mu)
        }
        PrecursorKey::Feature(fid) => {
            let feat = &features[fid];
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
    pairs: &[(usize, usize)], // (ms2_idx, ms1_idx)
    opts: &PseudoSpecOpts,
) -> Vec<PseudoSpectrum> {
    if ms1.is_empty() || ms2.is_empty() || pairs.is_empty() {
        return Vec::new();
    }

    // 1) Map MS1 cluster index -> feature_id (if any).
    let cluster_to_feature = build_cluster_to_feature_map(ms1.len(), features);

    // 2) Group all MS2 indices by precursor key (feature or orphan MS1 cluster).
    let mut grouped: HashMap<PrecursorKey, Vec<usize>> = HashMap::new();
    for &(ms2_idx, ms1_idx) in pairs {
        if ms1_idx >= ms1.len() || ms2_idx >= ms2.len() {
            continue;
        }
        let key = match cluster_to_feature[ms1_idx] {
            Some(fid) => PrecursorKey::Feature(fid),
            None => PrecursorKey::OrphanCluster(ms1_idx),
        };
        grouped.entry(key).or_default().push(ms2_idx);
    }

    if grouped.is_empty() {
        return Vec::new();
    }

    let top_n = opts.top_n_fragments;

    // 3) Build pseudo spectra in parallel over precursors.
    let mut spectra: Vec<PseudoSpectrum> = grouped
        .into_par_iter()
        .map(|(key, mut frag_indices)| {
            // Dedup MS2 cluster indices per precursor.
            frag_indices.sort_unstable();
            frag_indices.dedup();

            // ---- Precursor summary ----
            let (prec_mz, charge, prec_cluster_indices, prec_cluster_ids) = match key {
                PrecursorKey::OrphanCluster(ci) => {
                    let c = &ms1[ci];
                    let mz = cluster_mz_mu(c).unwrap_or(0.0);
                    let z = 0u8; // unknown; you can flip to 1 if you want “assume +1”
                    let idxs = vec![ci];
                    let ids = vec![c.cluster_id];
                    (mz, z, idxs, ids)
                }
                PrecursorKey::Feature(fid) => {
                    let feat = &features[fid];
                    let mz = feat.mz_mono;
                    let z = feat.charge;
                    let idxs = feat.member_cluster_indices.clone();
                    let ids = feat.member_cluster_ids.clone();
                    (mz, z, idxs, ids)
                }
            };

            // RT / IM apex of precursor.
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

            // Sort by fragment intensity (desc) and cap at top N if requested.
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
                feature_id: match key {
                    PrecursorKey::Feature(fid) => Some(fid),
                    PrecursorKey::OrphanCluster(_) => None,
                },
                precursor_cluster_indices: prec_cluster_indices,
                precursor_cluster_ids: prec_cluster_ids,
                fragments: frags,
            }
        })
        .collect();

    // Optional but nice: sort spectra by RT apex, then m/z.
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