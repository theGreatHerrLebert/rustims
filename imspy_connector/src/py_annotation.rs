use std::collections::HashSet;
use pyo3::prelude::*;
use mscore::simulation::annotation::{SourceType, SignalAttributes};

#[pyclass]
pub struct PySourceType {
    inner: SourceType,
}
#[pymethods]
impl PySourceType {
    #[new]
    pub fn new(source_type: i32) -> PyResult<Self> {
        Ok(PySourceType {
            inner: SourceType::new(source_type),
        })
    }
    #[getter]
    pub fn source_type(&self) -> String { self.inner.to_string() }
}

#[pyclass]
pub struct PySignalAttributes {
    inner: SignalAttributes,
}
#[pymethods]
impl PySignalAttributes {
    #[new]
    pub fn new(charge_state: i32, peptide_id: i32, isotope_peak: i32) -> PyResult<Self> {
        Ok(PySignalAttributes {
            inner: SignalAttributes {
                charge_state,
                peptide_id,
                isotope_peak,
            },
        })
    }
    #[getter]
    pub fn charge_state(&self) -> i32 { self.inner.charge_state }
    #[getter]
    pub fn peptide_id(&self) -> i32 { self.inner.peptide_id }
    #[getter]
    pub fn isotope_peak(&self) -> i32 { self.inner.isotope_peak }
}

#[pymodule]
pub fn annotation(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySourceType>()?;
    m.add_class::<PySignalAttributes>()?;
    Ok(())
}