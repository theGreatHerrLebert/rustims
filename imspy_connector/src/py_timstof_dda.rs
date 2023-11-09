use pyo3::prelude::*;
use mscore::{ TimsDDAPrecursor };

#[pyclass]
#[derive(Clone)]
pub struct PyTimsDDAPrecursor {
    pub inner: TimsDDAPrecursor,
}

#[pymethods]
impl PyTimsDDAPrecursor {
    #[new]
    pub fn new(
        _py: Python,
        id: u32,
        parent_id: u32,
        scan_num: u32,
        mz_average: f32,
        mz_most_intense: f32,
        intensity: f32,
        mz_mono_isotopic: Option<f32>,
        charge: Option<u32>,
    ) -> PyResult<Self> {
        let tims_dda_precursor = TimsDDAPrecursor {
            id,
            parent_id,
            scan_num,
            mz_average,
            mz_most_intense,
            intensity,
            mz_mono_isotopic,
            charge,
        };

        Ok(PyTimsDDAPrecursor { inner: tims_dda_precursor })
    }

    #[getter]
    pub fn id(&self) -> u32 { self.inner.id }

    #[getter]
    pub fn parent_id(&self) -> u32 { self.inner.parent_id }

    #[getter]
    pub fn scan_num(&self) -> u32 { self.inner.scan_num }

    #[getter]
    pub fn mz_average(&self) -> f32 { self.inner.mz_average }

    #[getter]
    pub fn mz_most_intense(&self) -> f32 { self.inner.mz_most_intense }

    #[getter]
    pub fn intensity(&self) -> f32 { self.inner.intensity }

    #[getter]
    pub fn mz_mono_isotopic(&self) -> Option<f32> { self.inner.mz_mono_isotopic }

    #[getter]
    pub fn charge(&self) -> Option<u32> { self.inner.charge }
}