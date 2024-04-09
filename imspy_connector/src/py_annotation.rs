use std::collections::HashSet;
use pyo3::prelude::*;
use mscore::simulation::annotation::{SourceType, SignalAttributes};

#[pyclass]
#[derive(Clone)]
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
#[derive(Clone)]
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

#[pyclass]
pub struct PyContributionSource {
    pub intensity_contribution: f64,
    pub source_type: PySourceType,
    pub signal_attributes: Option<PySignalAttributes>,
}

#[pymethods]
impl PyContributionSource {
    #[new]
    pub fn new(intensity_contribution: f64, source_type: i32, signal_attributes: Option<PySignalAttributes>) -> Self {
        PyContributionSource {
            intensity_contribution,
            source_type: PySourceType::new(source_type).unwrap(),
            signal_attributes,
        }
    }
    #[getter]
    pub fn intensity_contribution(&self) -> f64 { self.intensity_contribution }

    #[getter]
    pub fn source_type(&self) -> PySourceType { self.source_type.clone() }

    #[getter]
    pub fn signal_attributes(&self) -> Option<PySignalAttributes> { self.signal_attributes.clone() }
}

#[pymodule]
pub fn annotation(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySourceType>()?;
    m.add_class::<PySignalAttributes>()?;
    m.add_class::<PyContributionSource>()?;
    Ok(())
}