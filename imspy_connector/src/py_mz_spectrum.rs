use pyo3::prelude::*;
use numpy::{PyArray1, IntoPyArray, PyArrayMethods};
use mscore::data::spectrum::{ToResolution, Vectorized};
use mscore::data::spectrum::{MzSpectrum, IndexedMzSpectrum, MsType, MzSpectrumVectorized};
use mscore::timstof::spectrum::{TimsSpectrum};
use pyo3::types::{PyList, PyTuple};
use std::sync::OnceLock;

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
    // Cached numpy arrays
    mz_cache: OnceLock<PyObject>,
    intensity_cache: OnceLock<PyObject>,
}

impl Clone for PyMzSpectrum {
    fn clone(&self) -> Self {
        PyMzSpectrum {
            inner: self.inner.clone(),
            mz_cache: OnceLock::new(),
            intensity_cache: OnceLock::new(),
        }
    }
}

impl PyMzSpectrum {
    pub fn from_inner(inner: MzSpectrum) -> Self {
        PyMzSpectrum {
            inner,
            mz_cache: OnceLock::new(),
            intensity_cache: OnceLock::new(),
        }
    }
}

#[pymethods]

impl PyMzSpectrum {
    #[new]
    pub unsafe fn new(mz: &Bound<'_, PyArray1<f64>>, intensity: &Bound<'_, PyArray1<f64>>) -> PyResult<Self> {
        Ok(PyMzSpectrum {
            inner: MzSpectrum::new(mz.as_slice()?.to_vec(), intensity.as_slice()?.to_vec()),
            mz_cache: OnceLock::new(),
            intensity_cache: OnceLock::new(),
        })
    }
    #[staticmethod]
    pub unsafe fn from_mzspectra_list(list: Vec<PyMzSpectrum>, resolution: i32) -> PyResult<Self> {
        if list.is_empty(){
            Ok(PyMzSpectrum::from_inner(MzSpectrum::new(Vec::new(), Vec::new())))
        }
        else {
            let mut convoluted: MzSpectrum = MzSpectrum::new(vec![], vec![]);
            for spectrum in list {
                convoluted = convoluted + spectrum.inner;
            }
            Ok(PyMzSpectrum::from_inner(convoluted.to_resolution(resolution)))
        }
    }

    #[getter]
    pub fn mz(&self, py: Python) -> PyObject {
        self.mz_cache.get_or_init(|| {
            (*self.inner.mz).clone().into_pyarray_bound(py).unbind().into_any()
        }).clone_ref(py)
    }
    #[getter]
    pub fn intensity(&self, py: Python) -> PyObject {
        self.intensity_cache.get_or_init(|| {
            (*self.inner.intensity).clone().into_pyarray_bound(py).unbind().into_any()
        }).clone_ref(py)
    }
    pub fn to_windows(&self, py: Python, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64) -> PyResult<PyObject> {
        let spectra = self.inner.to_windows(window_length, overlapping, min_peaks, min_intensity);

        let mut indices: Vec<i32> = Vec::new();
        let py_list: Py<PyList> = PyList::empty_bound(py).into();

        for (index, spec) in spectra {
            indices.push(index);
            let py_spec = Py::new(py, PyMzSpectrum::from_inner(spec))?;
            py_list.bind(py).append(py_spec)?;
        }

        let numpy_indices = indices.into_pyarray_bound(py).unbind();

        Ok(PyTuple::new_bound(py, &[numpy_indices.to_object(py), py_list.into()]).to_object(py))
    }

    pub fn to_resolution(&self, resolution: i32) -> PyMzSpectrum {
        PyMzSpectrum::from_inner(self.inner.to_resolution(resolution))
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
        Ok(PyMzSpectrum::from_inner(filtered))
    }
    pub fn __add__(&self, other: PyMzSpectrum) -> PyResult<PyMzSpectrum> {
        Ok(PyMzSpectrum::from_inner(self.inner.clone() + other.inner))
    }
    pub fn __mul__(&self, scale: f64) -> PyResult<PyMzSpectrum> {
        Ok(PyMzSpectrum::from_inner(self.inner.clone() * scale))
    }

    pub fn to_centroided(&self, baseline_noise_level: i32, sigma: f64, normalize: bool) -> PyMzSpectrum {
        PyMzSpectrum::from_inner(self.inner.to_centroid(baseline_noise_level, sigma, normalize))
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.inner).unwrap()
    }

    pub fn add_mz_noise_uniform(&self, noise_ppm: f64, right_drag: bool) -> PyMzSpectrum {
        PyMzSpectrum::from_inner(self.inner.add_mz_noise_uniform(noise_ppm, right_drag))
    }

    pub fn add_mz_noise_normal(&self, noise_ppm: f64) -> PyMzSpectrum {
        PyMzSpectrum::from_inner(self.inner.add_mz_noise_normal(noise_ppm))
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
    pub unsafe fn new(indices: &Bound<'_, PyArray1<i32>>, values: &Bound<'_, PyArray1<f64>>, resolution: i32) -> PyResult<Self> {
        Ok(PyMzSpectrumVectorized {
            inner: MzSpectrumVectorized {
                resolution,
                indices: indices.as_slice()?.to_vec(),
                values: values.as_slice()?.to_vec(),
            },
        })
    }

    #[pyo3(signature = (max_index=None))]
    pub fn to_dense_spectrum(&self, max_index: Option<usize>) -> PyMzSpectrumVectorized {
        PyMzSpectrumVectorized { inner: self.inner.to_dense_spectrum(max_index) }
    }
    
    #[getter]
    pub fn resolution(&self) -> i32 {
        self.inner.resolution
    }

    #[getter]
    pub fn indices(&self, py: Python) -> Py<PyArray1<i32>> {
        self.inner.indices.clone().into_pyarray_bound(py).unbind()
    }

    #[getter]
    pub fn values(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.values.clone().into_pyarray_bound(py).unbind()
    }
}

#[pyclass]
pub struct PyIndexedMzSpectrum {
    pub inner: IndexedMzSpectrum,
    // Cached numpy arrays
    index_cache: OnceLock<PyObject>,
    mz_cache: OnceLock<PyObject>,
    intensity_cache: OnceLock<PyObject>,
}

impl Clone for PyIndexedMzSpectrum {
    fn clone(&self) -> Self {
        PyIndexedMzSpectrum {
            inner: self.inner.clone(),
            index_cache: OnceLock::new(),
            mz_cache: OnceLock::new(),
            intensity_cache: OnceLock::new(),
        }
    }
}

impl PyIndexedMzSpectrum {
    pub fn from_inner(inner: IndexedMzSpectrum) -> Self {
        PyIndexedMzSpectrum {
            inner,
            index_cache: OnceLock::new(),
            mz_cache: OnceLock::new(),
            intensity_cache: OnceLock::new(),
        }
    }
}

#[pymethods]
impl PyIndexedMzSpectrum {
    #[new]
    pub unsafe fn new(index:&Bound<'_, PyArray1<i32>>, mz: &Bound<'_, PyArray1<f64>>, intensity: &Bound<'_, PyArray1<f64>>) -> PyResult<Self> {
        Ok(PyIndexedMzSpectrum {
            inner: IndexedMzSpectrum::new(
                index.as_slice()?.to_vec(),
                mz.as_slice()?.to_vec(),
                intensity.as_slice()?.to_vec(),
            ),
            index_cache: OnceLock::new(),
            mz_cache: OnceLock::new(),
            intensity_cache: OnceLock::new(),
        })
    }

    #[getter]
    pub fn index(&self, py: Python) -> PyObject {
        self.index_cache.get_or_init(|| {
            self.inner.index.clone().into_pyarray_bound(py).unbind().into_any()
        }).clone_ref(py)
    }

    #[getter]
    pub fn mz(&self, py: Python) -> PyObject {
        self.mz_cache.get_or_init(|| {
            (*self.inner.mz_spectrum.mz).clone().into_pyarray_bound(py).unbind().into_any()
        }).clone_ref(py)
    }

    #[getter]
    pub fn intensity(&self, py: Python) -> PyObject {
        self.intensity_cache.get_or_init(|| {
            (*self.inner.mz_spectrum.intensity).clone().into_pyarray_bound(py).unbind().into_any()
        }).clone_ref(py)
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min: f64, intensity_max: f64) -> PyResult<PyIndexedMzSpectrum> {
        let filtered = self.inner.filter_ranged(mz_min, mz_max, intensity_min, intensity_max);
        Ok(PyIndexedMzSpectrum::from_inner(filtered))
    }
}

#[pyclass]
pub struct PyTimsSpectrum {
    pub inner: TimsSpectrum,
    // Cached numpy arrays
    index_cache: OnceLock<PyObject>,
    mz_cache: OnceLock<PyObject>,
    intensity_cache: OnceLock<PyObject>,
}

impl Clone for PyTimsSpectrum {
    fn clone(&self) -> Self {
        PyTimsSpectrum {
            inner: self.inner.clone(),
            index_cache: OnceLock::new(),
            mz_cache: OnceLock::new(),
            intensity_cache: OnceLock::new(),
        }
    }
}

impl PyTimsSpectrum {
    pub fn from_inner(inner: TimsSpectrum) -> Self {
        PyTimsSpectrum {
            inner,
            index_cache: OnceLock::new(),
            mz_cache: OnceLock::new(),
            intensity_cache: OnceLock::new(),
        }
    }
}

#[pymethods]
impl PyTimsSpectrum {
    #[new]
    pub unsafe fn new(frame_id: i32, scan: i32, retention_time: f64, mobility: f64,
                      ms_type: i32, index: &Bound<'_, PyArray1<i32>>, mz: &Bound<'_, PyArray1<f64>>, intensity: &Bound<'_, PyArray1<f64>>) -> PyResult<Self> {
        Ok(PyTimsSpectrum {
            inner: TimsSpectrum {
                frame_id,
                scan,
                retention_time,
                mobility,
                ms_type: MsType::new(ms_type),
                spectrum: IndexedMzSpectrum::new(
                    index.as_slice()?.to_vec(),
                    mz.as_slice()?.to_vec(),
                    intensity.as_slice()?.to_vec(),
                ),
            },
            index_cache: OnceLock::new(),
            mz_cache: OnceLock::new(),
            intensity_cache: OnceLock::new(),
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
    pub fn index(&self, py: Python) -> PyObject {
        self.index_cache.get_or_init(|| {
            self.inner.spectrum.index.clone().into_pyarray_bound(py).unbind().into_any()
        }).clone_ref(py)
    }

    #[getter]
    pub fn mz(&self, py: Python) -> PyObject {
        self.mz_cache.get_or_init(|| {
            (*self.inner.spectrum.mz_spectrum.mz).clone().into_pyarray_bound(py).unbind().into_any()
        }).clone_ref(py)
    }

    #[getter]
    pub fn intensity(&self, py: Python) -> PyObject {
        self.intensity_cache.get_or_init(|| {
            (*self.inner.spectrum.mz_spectrum.intensity).clone().into_pyarray_bound(py).unbind().into_any()
        }).clone_ref(py)
    }

    #[getter]
    pub fn ms_type(&self) -> String { self.inner.ms_type.to_string() }

    #[getter]
    pub fn ms_type_numeric(&self) -> i32 { self.inner.ms_type.ms_type_numeric() }

    #[getter]
    pub fn indexed_mz_spectrum(&self) -> PyIndexedMzSpectrum {
        PyIndexedMzSpectrum::from_inner(self.inner.spectrum.clone())
    }

    #[getter]
    pub fn mz_spectrum(&self) -> PyMzSpectrum {
        PyMzSpectrum::from_inner(self.inner.spectrum.mz_spectrum.clone())
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min: f64, intensity_max: f64) -> PyResult<PyTimsSpectrum> {
        let filtered = self.inner.filter_ranged(mz_min, mz_max, intensity_min, intensity_max);
        Ok(PyTimsSpectrum::from_inner(filtered))
    }

    pub fn to_resolution(&self, resolution: i32) -> Self {
        PyTimsSpectrum::from_inner(self.inner.to_resolution(resolution))
    }

    pub fn __add__(&self, other: PyTimsSpectrum) -> PyResult<Self> {
        Ok(PyTimsSpectrum::from_inner(self.inner.clone() + other.inner))
    }
}

#[pymodule]
pub fn py_spectrum(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMsType>()?;
    m.add_class::<PyMzSpectrum>()?;
    m.add_class::<PyMzSpectrumVectorized>()?;
    m.add_class::<PyIndexedMzSpectrum>()?;
    m.add_class::<PyTimsSpectrum>()?;
    Ok(())
}
