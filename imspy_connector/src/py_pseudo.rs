use pyo3::prelude::*;
use rustdf::cluster::cluster::ClusterResult1D;
use rustdf::cluster::pseudo::{PseudoFragment, PseudoSpectrum};
use crate::py_dia::PyClusterResult1D;

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyPseudoFragment {
    mz: f32,
    intensity: f32,
    ms2_cluster_id: u64,
}

#[pymethods]
impl PyPseudoFragment {
    #[getter]
    pub fn mz(&self) -> f32 { self.mz }

    #[getter]
    pub fn intensity(&self) -> f32 { self.intensity }

    #[getter]
    pub fn ms2_cluster_id(&self) -> u64 { self.ms2_cluster_id }

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
    pub fn new(
        py: Python<'_>,
        precursor: Py<PyClusterResult1D>,
        fragments: Vec<Py<PyClusterResult1D>>,
    ) -> PyResult<Self> {

        // ---- Extract precursor ----
        let prec_ref = precursor.borrow(py);
        let prec = prec_ref.inner.clone();

        // sanity
        if prec.ms_level != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "PseudoSpectrum: precursor must be MS1 (ms_level=1)",
            ));
        }

        // Precursor m/z: prefer mz_fit, fallback to window midpoint
        let precursor_mz = if let Some(fit) = &prec.mz_fit {
            fit.mu
        } else if let Some((lo, hi)) = prec.mz_window {
            0.5 * (lo + hi)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Precursor cluster has no m/z fit and no m/z window.",
            ));
        };

        // ---- Build fragments ----
        let mut frags: Vec<PseudoFragment> = Vec::new();
        for f_py in fragments {
            let f_ref = f_py.borrow(py);
            let c = f_ref.inner.clone();

            if c.ms_level != 2 {
                continue; // ignore accidental precursors
            }
            if let Some(pf) = fragment_from_cluster(&c) {
                frags.push(pf);
            }
        }

        // ---- Build the Rust PseudoSpectrum ----
        let ps = PseudoSpectrum {
            precursor_mz,
            precursor_charge: 0,
            rt_apex: prec.rt_fit.mu,
            im_apex: prec.im_fit.mu,
            feature_id: None,
            window_group: prec.window_group.unwrap_or(0),
            precursor_cluster_ids: vec![prec.cluster_id],
            fragments: frags,
            precursor_cluster_indices: vec![],
        };

        Ok(PyPseudoSpectrum { inner: ps })
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
    })
}

/// Module init for this submodule.
#[pymodule]
pub fn py_pseudo(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPseudoFragment>()?;
    m.add_class::<PyPseudoSpectrum>()?;
    Ok(())
}