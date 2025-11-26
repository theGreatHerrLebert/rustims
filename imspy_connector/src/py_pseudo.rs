use std::cmp::Ordering;
use pyo3::prelude::*;
use rustdf::cluster::cluster::ClusterResult1D;
use rustdf::cluster::pseudo::{PseudoFragment, PseudoSpectrum};
use crate::py_dia::PyClusterResult1D;
use crate::py_feature::PySimpleFeature;

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyPseudoFragment {
    mz: f32,
    intensity: f32,
    ms2_cluster_id: u64,
    window_group: u32,
}

#[pymethods]
impl PyPseudoFragment {
    #[getter]
    pub fn mz(&self) -> f32 { self.mz }

    #[getter]
    pub fn intensity(&self) -> f32 { self.intensity }

    #[getter]
    pub fn ms2_cluster_id(&self) -> u64 { self.ms2_cluster_id }

    #[getter]
    pub fn window_group(&self) -> u32 {
        self.window_group
    }

    fn __repr__(&self) -> String {
        format!(
            "PseudoFragment(mz={:.4}, intensity={:.1}, ms2_id={})",
            self.mz, self.intensity, self.ms2_cluster_id
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyPseudoSpectrum {
    pub inner: PseudoSpectrum,
}

#[pymethods]
impl PyPseudoSpectrum {
    #[new]
    #[pyo3(signature = (fragments, precursor=None, feature=None))]
    pub fn new(
        py: Python<'_>,
        fragments: Vec<Py<PyClusterResult1D>>,
        precursor: Option<Py<PyClusterResult1D>>,
        feature:   Option<Py<PySimpleFeature>>,
    ) -> PyResult<Self> {
        use pyo3::exceptions::PyValueError;

        let have_prec = precursor.is_some();
        let have_feat = feature.is_some();
        if have_prec == have_feat {
            return Err(PyValueError::new_err(
                "PseudoSpectrum: exactly one of `precursor` or `feature` must be provided",
            ));
        }

        // ---- Build fragments (shared for both modes) --------------------
        let mut window_group: u32 = 0;
        let mut frags: Vec<PseudoFragment> = Vec::new();

        for f_py in fragments {
            let f_ref = f_py.borrow(py);
            let c = f_ref.inner.clone();

            // take first non-zero window_group we see
            if window_group == 0 {
                window_group = c.window_group.unwrap_or(0);
            }

            // only accept MS2 clusters as fragments
            if c.ms_level != 2 {
                continue;
            }

            if let Some(pf) = fragment_from_cluster(&c) {
                frags.push(pf);
            }
        }

        // sort by m/z ascending
        frags.sort_by(|a, b| {
            a.mz.partial_cmp(&b.mz).unwrap_or(Ordering::Equal)
        });

        if frags.is_empty() {
            return Err(PyValueError::new_err(
                "PseudoSpectrum: no usable fragment clusters provided",
            ));
        }

        // ---- Mode 1: from precursor cluster -----------------------------
        let inner = if let Some(p) = precursor {
            let prec_ref = p.borrow(py);
            let prec = prec_ref.inner.clone();

            if prec.ms_level != 1 {
                return Err(PyValueError::new_err(
                    "PseudoSpectrum: precursor must be MS1 (ms_level=1)",
                ));
            }

            // Precursor m/z: prefer mz_fit, fallback to window midpoint
            let precursor_mz = if let Some(fit) = &prec.mz_fit {
                fit.mu
            } else if let Some((lo, hi)) = prec.mz_window {
                0.5 * (lo + hi)
            } else {
                return Err(PyValueError::new_err(
                    "Precursor cluster has no m/z fit and no m/z window.",
                ));
            };

            PseudoSpectrum {
                precursor_mz,
                precursor_charge: 0, // we don't know charge from the cluster
                rt_apex: prec.rt_fit.mu,
                im_apex: prec.im_fit.mu,
                feature_id: None,
                window_group,
                precursor_cluster_ids: vec![prec.cluster_id],
                fragments: frags,
                precursor_cluster_indices: vec![],
            }
        } else {
            // ---- Mode 2: from SimpleFeature --------------------------------
            let feat = feature.unwrap().borrow(py).inner.clone();

            // Use feature’s own mono m/z and charge
            let precursor_mz = feat.mz_mono;
            let precursor_charge = feat.charge;
            let feature_id = Some(feat.feature_id);

            // Derive RT / IM apex + cluster IDs from the top-intensity member
            let mut rt_apex = 0.0f32;
            let mut im_apex = 0.0f32;
            let mut precursor_cluster_ids: Vec<u64> = Vec::new();

            if let Some(top) = feat
                .member_clusters
                .iter()
                .max_by(|a, b| {
                    a.raw_sum
                        .partial_cmp(&b.raw_sum)
                        .unwrap_or(Ordering::Equal)
                })
            {
                rt_apex = top.rt_fit.mu;
                im_apex = top.im_fit.mu;
                if window_group == 0 {
                    // if not set from fragments, fall back to precursor tile
                    window_group = top.window_group.unwrap_or(0);
                }
                precursor_cluster_ids.push(top.cluster_id);
            } else {
                // Worst case: feature has no backing clusters (should not happen)
                // fall back to rough midpoints of RT / IM bounds
                let (rt_lo, rt_hi) = feat.rt_bounds;
                let (im_lo, im_hi) = feat.im_bounds;
                rt_apex = 0.5 * (rt_lo as f32 + rt_hi as f32);
                im_apex = 0.5 * (im_lo as f32 + im_hi as f32);
            }

            PseudoSpectrum {
                precursor_mz,
                precursor_charge,
                rt_apex,
                im_apex,
                feature_id,
                window_group,
                precursor_cluster_ids,
                fragments: frags,
                // this is actually meaningful for features
                precursor_cluster_indices: feat.member_cluster_indices.clone(),
            }
        };

        Ok(PyPseudoSpectrum { inner })
    }

    #[getter]
    pub fn precursor_mz(&self) -> f32 {
        self.inner.precursor_mz
    }

    #[getter]
    pub fn precursor_charge(&self) -> u8 {
        self.inner.precursor_charge
    }

    #[getter]
    pub fn rt_apex(&self) -> f32 {
        self.inner.rt_apex
    }

    #[getter]
    pub fn im_apex(&self) -> f32 {
        self.inner.im_apex
    }

    #[getter]
    pub fn feature_id(&self) -> Option<usize> {
        self.inner.feature_id
    }

    #[getter]
    pub fn window_group_id(&self) -> u32 {
        self.inner.window_group
    }

    #[getter]
    pub fn precursor_cluster_ids(&self) -> Vec<u64> {
        self.inner.precursor_cluster_ids.clone()
    }

    #[getter]
    pub fn fragments(&self) -> Vec<PyPseudoFragment> {
        self.inner.fragments
            .iter()
            .map(|f| PyPseudoFragment {
                mz: f.mz,
                intensity: f.intensity,
                ms2_cluster_id: f.ms2_cluster_id,
                window_group: f.window_group,
            })
            .collect()
    }
    /// Vector of fragment m/z values
    #[getter]
    pub fn fragment_mz_array(&self) -> Vec<f32> {
        self.inner.fragments.iter().map(|f| f.mz).collect()
    }

    /// Vector of fragment intensities
    #[getter]
    pub fn fragment_intensity_array(&self) -> Vec<f32> {
        self.inner.fragments.iter().map(|f| f.intensity).collect()
    }

    /// Number of fragments
    #[getter]
    pub fn n_fragments(&self) -> usize {
        self.inner.fragments.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PseudoSpectrum(mz={:.4}, z={}, n_frags={})",
            self.inner.precursor_mz,
            self.inner.precursor_charge,
            self.inner.fragments.len(),
        )
    }
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

/// Module init for this submodule.
#[pymodule]
pub fn py_pseudo(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPseudoFragment>()?;
    m.add_class::<PyPseudoSpectrum>()?;
    Ok(())
}