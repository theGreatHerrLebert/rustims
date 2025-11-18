use pyo3::prelude::*;
use rustdf::cluster::pseudo::PseudoSpectrum;

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

    fn __repr__(&self) -> String {
        format!(
            "PseudoSpectrum(mz={:.4}, z={}, n_frags={})",
            self.inner.precursor_mz,
            self.inner.precursor_charge,
            self.inner.fragments.len(),
        )
    }
}

/// Module init for this submodule.
#[pymodule]
pub fn py_pseudo(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPseudoFragment>()?;
    m.add_class::<PyPseudoSpectrum>()?;
    Ok(())
}