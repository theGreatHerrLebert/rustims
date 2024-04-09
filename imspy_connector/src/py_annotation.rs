use std::collections::HashSet;
use pyo3::prelude::*;
use mscore::simulation::annotation::{PeakAnnotation, MzSpectrumAnnotated, SourceType, ContributionSource, SignalAttributes};
use rustdf::data::dataset::TimsDataset;

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

#[pymodule]
pub fn annotation(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySourceType>()?;
    Ok(())
}