use pyo3::prelude::*;
use numpy::{PyArray1, IntoPyArray};
use mscore::{MzSpectrum, IndexedMzSpectrum, TimsSpectrum, MsType, MzSpectrumVectorized, ToResolution, Vectorized};
use pyo3::types::{PyList, PyTuple};

#[pyclass]
#[derive(Clone)]
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

    pub fn to_resolution(&self, resolution: i32) -> PyMzSpectrum {
        PyMzSpectrum { inner: self.inner.to_resolution(resolution) }
    }

    pub fn vectorized(&self, _py: Python, resolution: i32) -> PyResult<PyMzSpectrumVectorized> {
        let vectorized = self.inner.vectorized(resolution);
        let py_vectorized = PyMzSpectrumVectorized {
            inner: vectorized,
        };
        Ok(py_vectorized)
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min: f64, intensity_max: f64) -> PyResult<PyMzSpectrum> {
        let filtered = self.inner.filter_ranged(mz_min, mz_max, intensity_min, intensity_max);
        let py_filtered = PyMzSpectrum {
            inner: filtered,
        };
        Ok(py_filtered)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyMzSpectrumVectorized {
    pub inner: MzSpectrumVectorized,
}

#[pymethods]
impl PyMzSpectrumVectorized {
    #[new]
    pub unsafe fn new(indices: &PyArray1<i32>, values: &PyArray1<f64>, resolution: i32) -> PyResult<Self> {
        Ok(PyMzSpectrumVectorized {
            inner: MzSpectrumVectorized {
                resolution,
                indices: indices.as_slice()?.to_vec(),
                values: values.as_slice()?.to_vec(),
            },
        })
    }

    #[getter]
    pub fn resolution(&self) -> i32 {
        self.inner.resolution
    }

    #[getter]
    pub fn indices(&self, py: Python) -> Py<PyArray1<i32>> {
        self.inner.indices.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn values(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.values.clone().into_pyarray(py).to_owned()
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

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min: f64, intensity_max: f64) -> PyResult<PyIndexedMzSpectrum> {
        let filtered = self.inner.filter_ranged(mz_min, mz_max, intensity_min, intensity_max);
        let py_filtered = PyIndexedMzSpectrum {
            inner: filtered,
        };
        Ok(py_filtered)
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
