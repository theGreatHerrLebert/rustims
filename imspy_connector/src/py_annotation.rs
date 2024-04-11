use mscore::data::spectrum::MsType;
use pyo3::prelude::*;
use mscore::simulation::annotation::{SourceType, SignalAttributes, ContributionSource, MzSpectrumAnnotated, PeakAnnotation, TimsFrameAnnotated};
use numpy::{IntoPyArray, PyArray1};

#[pyclass]
#[derive(Clone)]
pub struct PyTimsFrameAnnotated {
    pub inner: TimsFrameAnnotated,
}

#[pymethods]
impl PyTimsFrameAnnotated {
    #[new]
    pub unsafe fn new(frame_id: i32,
                      retention_time: f64,
                      ms_type: i32,
                      tof: &PyArray1<u32>,
                      mz: &PyArray1<f64>,
                      scan: &PyArray1<u32>,
                      inv_mobility: &PyArray1<f64>,
                      intensity: &PyArray1<f64>,
                      annotations: Vec<PyPeakAnnotation>) -> PyResult<Self> {

        assert!(tof.len() == mz.len() && mz.len() == scan.len() && scan.len() == inv_mobility.len() && inv_mobility.len() == intensity.len() && intensity.len() == annotations.len());

        let ms_type = MsType::new(ms_type);
        let annotations = annotations.iter().map(|x| PeakAnnotation {
            contributions: x.inner.clone()
        }).collect();

        Ok(PyTimsFrameAnnotated {
            inner: TimsFrameAnnotated {
                frame_id,
                retention_time,
                ms_type,
                tof: tof.as_slice()?.to_vec(),
                mz: mz.as_slice()?.to_vec(),
                scan: scan.as_slice()?.to_vec(),
                inv_mobility: inv_mobility.as_slice()?.to_vec(),
                intensity: intensity.as_slice()?.to_vec(),
                annotations,
            },
        })
    }

    #[getter]
    pub fn frame_id(&self) -> i32 { self.inner.frame_id }

    #[getter]
    pub fn retention_time(&self) -> f64 { self.inner.retention_time }
    #[getter]
    pub fn tof(&self, py: Python) -> Py<PyArray1<u32>> { self.inner.tof.clone().into_pyarray_bound(py).unbind() }

    #[getter]
    pub fn mz(&self, py: Python) -> Py<PyArray1<f64>> { self.inner.mz.clone().into_pyarray_bound(py).unbind() }

    #[getter]
    pub fn scan(&self, py: Python) -> Py<PyArray1<u32>> { self.inner.scan.clone().into_pyarray_bound(py).unbind() }

    #[getter]
    pub fn inv_mobility(&self, py: Python) -> Py<PyArray1<f64>> { self.inner.inv_mobility.clone().into_pyarray_bound(py).unbind() }

    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> { self.inner.intensity.clone().into_pyarray_bound(py).unbind() }

    #[getter]
    pub fn annotations(&self) -> Vec<PyPeakAnnotation> {
        self.inner.annotations.iter().map(|x| PyPeakAnnotation { inner: x.contributions.clone() }).collect()
    }

    #[getter]
    pub fn peptide_ids_first_only(&self, py: Python) -> Py<PyArray1<i32>> {
        let data: Vec<_> = self.inner.annotations.iter().map(|x| {
            x.contributions.first().map_or(-1, |contribution| {
                contribution.signal_attributes.as_ref().map_or(-1, |signal_attributes| signal_attributes.peptide_id)
            })
        }).collect();
        data.into_pyarray_bound(py).unbind()
    }

    #[getter]
    pub fn charge_states_first_only(&self, py: Python) ->  Py<PyArray1<i32>> {
        let data: Vec<_> =self.inner.annotations.iter().map(|x| x.contributions.first().unwrap().signal_attributes.as_ref().unwrap().charge_state).collect();
        data.into_pyarray_bound(py).unbind()
    }

    #[getter]
    pub fn isotope_peaks_first_only(&self, py: Python) ->  Py<PyArray1<i32>> {
        let data: Vec<_> =self.inner.annotations.iter().map(|x| x.contributions.first().unwrap().signal_attributes.as_ref().unwrap().isotope_peak).collect();
        data.into_pyarray_bound(py).unbind()
    }

    #[getter]
    pub fn ms_type_numeric(&self) -> i32 {
        self.inner.ms_type.ms_type_numeric()
    }

    #[setter]
    pub unsafe fn set_tof(&mut self, tof: &PyArray1<u32>) {
        self.inner.tof = tof.as_slice().unwrap().to_vec();
    }

    pub fn __add__(&self, other: PyTimsFrameAnnotated) -> PyResult<PyTimsFrameAnnotated> {
        Ok(PyTimsFrameAnnotated { inner: self.inner.clone() + other.inner })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyMzSpectrumAnnotated {
    pub inner: MzSpectrumAnnotated,
}

#[pymethods]
impl PyMzSpectrumAnnotated {
    #[new]
    pub unsafe fn new(mz: &PyArray1<f64>, intensity: &PyArray1<f64>, annotations: Vec<PyPeakAnnotation>) -> PyResult<Self> {
        assert!(mz.len() == intensity.len() && intensity.len() == annotations.len());
        let annotations = annotations.iter().map(|x| PeakAnnotation {
            contributions: x.inner.clone()
        }).collect();
        Ok(PyMzSpectrumAnnotated {
            inner: MzSpectrumAnnotated {
                mz: mz.as_slice()?.to_vec(),
                intensity: intensity.as_slice()?.to_vec(),
                annotations,
            },
        })
    }
    #[getter]
    pub fn mz(&self, py: Python) -> Py<PyArray1<f64>> { self.inner.mz.clone().into_pyarray_bound(py).unbind() }

    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> { self.inner.intensity.clone().into_pyarray_bound(py).unbind() }

    #[getter]
    pub fn annotations(&self) -> Vec<PyPeakAnnotation> {
        self.inner.annotations.iter().map(|x| PyPeakAnnotation { inner: x.contributions.clone() }).collect()
    }

    pub fn __add__(&self, other: PyMzSpectrumAnnotated) -> PyResult<PyMzSpectrumAnnotated> {
        Ok(PyMzSpectrumAnnotated { inner: self.inner.clone() + other.inner })
    }

    pub fn __mul__(&self, other: f64) -> PyResult<PyMzSpectrumAnnotated> {
        Ok(PyMzSpectrumAnnotated { inner: self.inner.clone() * other })
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
    m.add_class::<PyTimsFrameAnnotated>()?;
    Ok(())
}