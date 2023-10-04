use pyo3::prelude::*;
use numpy::{PyArray1, IntoPyArray};
use mscore::{MzSpectrum, IndexedMzSpectrum, ImsSpectrum, TimsSpectrum, MsType};
use pyo3::types::{PyList, PyTuple};

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
    pub fn to_windows(&self, py: Python, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64) -> PyResult<PyObject> {
        let spectra = self.inner.to_windows(window_length, overlapping, min_peaks, min_intensity);

        let mut indices: Vec<i32> = Vec::new();
        let py_list: Py<PyList> = PyList::empty(py).into();

        for (index, spec) in spectra {
            indices.push(index);
            let py_spec = Py::new(py, PyMzSpectrum { inner: spec })?;
            py_list.as_ref(py).append(py_spec)?;
        }

        let numpy_indices = indices.into_pyarray(py);

        Ok(PyTuple::new(py, &[numpy_indices.to_object(py), py_list.into()]).to_object(py))
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
                mz_spectrum: MzSpectrum { mz: mz.as_slice()?.to_vec(), intensity:  intensity.as_slice()?.to_vec() },
            },
        })
    }

    #[getter]
    pub fn index(&self, py: Python) -> Py<PyArray1<i32>> {
        self.inner.index.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn mz(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.mz_spectrum.mz.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.mz_spectrum.intensity.clone().into_pyarray(py).to_owned()
    }
}

#[pyclass]
pub struct PyImsSpectrum {
    pub inner: ImsSpectrum,
}

#[pymethods]
impl PyImsSpectrum {
    #[new]
    pub unsafe fn new(retention_time: f64, mobility: f64, mz: &PyArray1<f64>, intensity: &PyArray1<f64>) -> PyResult<Self> {
        Ok(PyImsSpectrum {
            inner: ImsSpectrum {
                retention_time,
                mobility: mobility,
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
    pub fn mobility(&self) -> f64 {
        self.inner.mobility
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
    pub unsafe fn new(frame_id: i32, scan: i32, retention_time: f64, mobility: f64,
                      ms_type: i32, index: &PyArray1<i32>, mz: &PyArray1<f64>, intensity: &PyArray1<f64>) -> PyResult<Self> {
        Ok(PyTimsSpectrum {
            inner: TimsSpectrum {
                frame_id,
                scan,
                retention_time,
                mobility,
                ms_type: MsType::new(ms_type),
                spectrum: IndexedMzSpectrum {
                    index: index.as_slice()?.to_vec(),
                    mz_spectrum: MzSpectrum {
                        mz: mz.as_slice()?.to_vec(),
                        intensity: intensity.as_slice()?.to_vec(),
                    },
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
    pub fn mobility(&self) -> f64 {
        self.inner.mobility
    }

    #[getter]
    pub fn index(&self, py: Python) -> Py<PyArray1<i32>> {
        self.inner.spectrum.index.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn mz(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.spectrum.mz_spectrum.mz.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.spectrum.mz_spectrum.intensity.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn ms_type(&self) -> String { self.inner.ms_type.to_string() }

    #[getter]
    pub fn ms_type_numeric(&self) -> i32 { self.inner.ms_type.ms_type_numeric() }

    #[getter]
    pub fn indexed_mz_spectrum(&self) -> PyIndexedMzSpectrum {
        PyIndexedMzSpectrum { inner: self.inner.spectrum.clone() }
    }

    #[getter]
    pub fn mz_spectrum(&self) -> PyMzSpectrum {
        PyMzSpectrum { inner: self.inner.spectrum.mz_spectrum.clone() }
    }
}
