use pyo3::prelude::*;
use mscore::simulation::annotation::{SourceType, SignalAttributes, ContributionSource, MzSpectrumAnnotated, PeakAnnotation};

#[pyclass]
#[derive(Clone)]
pub struct PyMzSpectrumAnnotated {
    pub inner: MzSpectrumAnnotated,
}

#[pymethods]
impl PyMzSpectrumAnnotated {
    #[new]
    pub fn new(mz: Vec<f64>, intensity: Vec<f64>, annotations: Vec<PyPeakAnnotation>) -> Self {
        assert!(mz.len() == intensity.len() && intensity.len() == annotations.len());
        let annotations = annotations.iter().map(|x| PeakAnnotation {
            contributions: x.inner.clone()
        }).collect();
        PyMzSpectrumAnnotated {
            inner: MzSpectrumAnnotated {
                mz,
                intensity,
                annotations,
            },
        }
    }
    #[getter]
    pub fn mz(&self) -> Vec<f64> { self.inner.mz.clone() }

    #[getter]
    pub fn intensity(&self) -> Vec<f64> { self.inner.intensity.clone() }

    #[getter]
    pub fn annotations(&self) -> Vec<PyPeakAnnotation> {
        self.inner.annotations.iter().map(|x| PyPeakAnnotation { inner: x.contributions.clone() }).collect()
    }

    pub fn __add__(&self, other: PyMzSpectrumAnnotated) -> PyResult<PyMzSpectrumAnnotated> {
        Ok(PyMzSpectrumAnnotated { inner: self.inner.clone() + other.inner })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyPeakAnnotation {
    pub inner: Vec<ContributionSource>,
}

#[pymethods]
impl PyPeakAnnotation {
    #[new]
    pub fn new(contributions: Vec<PyContributionSource>) -> Self {
        PyPeakAnnotation {
            inner: contributions.iter().map(|x| x.inner.clone()).collect(),
        }
    }
    #[getter]
    pub fn contributions(&self) -> Vec<PyContributionSource> {
        self.inner.iter().map(|x| PyContributionSource { inner: x.clone() }).collect()
    }
}

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
    pub fn new(charge_state: i32, peptide_id: i32, isotope_peak: i32, description: Option<String>) -> PyResult<Self> {
        Ok(PySignalAttributes {
            inner: SignalAttributes {
                charge_state,
                peptide_id,
                isotope_peak,
                description,
            },
        })
    }
    #[getter]
    pub fn charge_state(&self) -> i32 { self.inner.charge_state }
    #[getter]
    pub fn peptide_id(&self) -> i32 { self.inner.peptide_id }
    #[getter]
    pub fn isotope_peak(&self) -> i32 { self.inner.isotope_peak }
    #[getter]
    pub fn description(&self) -> Option<String> { self.inner.description.clone() }
}

#[pyclass]
#[derive(Clone)]
pub struct PyContributionSource {
    pub inner: ContributionSource,
}

#[pymethods]
impl PyContributionSource {
    #[new]
    pub fn new(intensity_contribution: f64, source_type: PySourceType, signal_attributes: Option<PySignalAttributes>) -> Self {
        PyContributionSource {
            inner: ContributionSource {
                intensity_contribution,
                source_type: source_type.inner.clone(),
                signal_attributes: signal_attributes.map(|x| x.inner.clone()),
            },
        }
    }

    #[getter]
    pub fn intensity_contribution(&self) -> f64 { self.inner.intensity_contribution }

    #[getter]
    pub fn source_type(&self) -> PySourceType { PySourceType { inner: self.inner.source_type.clone() } }

    #[getter]
    pub fn signal_attributes(&self) -> Option<PySignalAttributes> {
        self.inner.signal_attributes.as_ref().map(|x| PySignalAttributes { inner: x.clone() })
    }
}

#[pymodule]
pub fn annotation(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySourceType>()?;
    m.add_class::<PySignalAttributes>()?;
    m.add_class::<PyContributionSource>()?;
    m.add_class::<PyPeakAnnotation>()?;
    m.add_class::<PyMzSpectrumAnnotated>()?;
    Ok(())
}