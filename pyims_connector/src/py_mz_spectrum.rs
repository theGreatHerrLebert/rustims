use pyo3::prelude::*;
use numpy::{PyArray1, IntoPyArray};
use mscore::{MzSpectrum, IndexedMzSpectrum, ImsSpectrum, TimsSpectrum, MsType};

#[pyclass]
pub struct PyMsType {
    pub inner: MsType,
}

#[pymethods]
impl PyMsType {
    #[new]
    pub fn new(ms_type: i32) -> PyResult<Self> {
        Ok(PyMsType {
            inner: MsType::new(ms_type),
        })
    }
    #[getter]
    pub fn ms_type(&self) -> String { self.inner.to_string() }

    #[getter]
    pub fn ms_type_numeric(&self) -> i32 { self.inner.ms_type_numeric() }
}

#[pyclass]
pub struct PyMzSpectrum {
    pub inner: MzSpectrum,
}

#[pymethods]
impl PyMzSpectrum {
    #[new]
    pub unsafe fn new(mz: &PyArray1<f64>, intensity: &PyArray1<f64>) -> PyResult<Self> {
        Ok(PyMzSpectrum {
            inner: MzSpectrum {
                mz: mz.as_slice()?.to_vec(),
                intensity: intensity.as_slice()?.to_vec(),
            },
        })
    }

    #[getter]
    pub fn mz(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.mz.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.intensity.clone().into_pyarray(py).to_owned()
    }
}

#[pyclass]
pub struct PyIndexedMzSpectrum {
    pub inner: IndexedMzSpectrum,
}

#[pymethods]
impl PyIndexedMzSpectrum {
    #[new]
    pub unsafe fn new(index: &PyArray1<i32>, mz: &PyArray1<f64>, intensity: &PyArray1<f64>) -> PyResult<Self> {
        Ok(PyIndexedMzSpectrum {
            inner: IndexedMzSpectrum {
                index: index.as_slice()?.to_vec(),
                mz: mz.as_slice()?.to_vec(),
                intensity: intensity.as_slice()?.to_vec(),
            },
        })
    }

    #[getter]
    pub fn index(&self, py: Python) -> Py<PyArray1<i32>> {
        self.inner.index.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn mz(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.mz.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.intensity.clone().into_pyarray(py).to_owned()
    }
}

#[pyclass]
pub struct PyImsSpectrum {
    pub inner: ImsSpectrum,
}

#[pymethods]
impl PyImsSpectrum {
    #[new]
    pub unsafe fn new(retention_time: f64, inv_mobility: f64, mz: &PyArray1<f64>, intensity: &PyArray1<f64>) -> PyResult<Self> {
        Ok(PyImsSpectrum {
            inner: ImsSpectrum {
                retention_time,
                inv_mobility,
                spectrum: MzSpectrum {
                    mz: mz.as_slice()?.to_vec(),
                    intensity: intensity.as_slice()?.to_vec(),
                },
            },
        })
    }

    #[getter]
    pub fn retention_time(&self) -> f64 {
        self.inner.retention_time
    }

    #[getter]
    pub fn inv_mobility(&self) -> f64 {
        self.inner.inv_mobility
    }

    #[getter]
    pub fn mz(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.spectrum.mz.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.spectrum.intensity.clone().into_pyarray(py).to_owned()
    }
}

#[pyclass]
pub struct PyTimsSpectrum {
    pub inner: TimsSpectrum,
}

#[pymethods]
impl PyTimsSpectrum {
    #[new]
    pub unsafe fn new(frame_id: i32, scan: i32, retention_time: f64, inv_mobility: f64, ms_type: i32, index: &PyArray1<i32>, mz: &PyArray1<f64>, intensity: &PyArray1<f64>) -> PyResult<Self> {
        Ok(PyTimsSpectrum {
            inner: TimsSpectrum {
                frame_id,
                scan,
                retention_time,
                inv_mobility,
                ms_type: MsType::new(ms_type),
                spectrum: IndexedMzSpectrum {
                    index: index.as_slice()?.to_vec(),
                    mz: mz.as_slice()?.to_vec(),
                    intensity: intensity.as_slice()?.to_vec(),
                },
            },
        })
    }

    #[getter]
    pub fn frame_id(&self) -> i32 {
        self.inner.frame_id
    }

    #[getter]
    pub fn scan(&self) -> i32 {
        self.inner.scan
    }

    #[getter]
    pub fn retention_time(&self) -> f64 {
        self.inner.retention_time
    }

    #[getter]
    pub fn inv_mobility(&self) -> f64 {
        self.inner.inv_mobility
    }

    #[getter]
    pub fn mz(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.spectrum.mz.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.spectrum.intensity.clone().into_pyarray(py).to_owned()
    }
}
