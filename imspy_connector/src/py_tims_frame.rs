use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::PyTuple;
use numpy::{PyArray1, IntoPyArray, PyArrayMethods};
use mscore::timstof::spectrum::{TimsSpectrum};
use mscore::data::spectrum::{MsType, ToResolution, Vectorized, };
use mscore::timstof::frame::{TimsFrame, TimsFrameVectorized, ImsFrameVectorized, RawTimsFrame};
use crate::py_annotation::PyTimsFrameAnnotated;
use std::sync::OnceLock;

use crate::py_mz_spectrum::{PyIndexedMzSpectrum, PyTimsSpectrum};

#[pyclass]
#[derive(Clone)]
pub struct PyRawTimsFrame {
    pub inner: RawTimsFrame,
}

#[pymethods]
impl PyRawTimsFrame {
    #[new]
    pub unsafe fn new(frame_id: i32,
                      ms_type: i32,
                      retention_time: f64,
                      scan: &Bound<'_, PyArray1<u32>>,
                      tof: &Bound<'_, PyArray1<u32>>,
                      intensity: &Bound<'_, PyArray1<f64>>) -> PyResult<Self> {
        Ok(PyRawTimsFrame {
            inner: RawTimsFrame {
                frame_id,
                retention_time,
                ms_type: MsType::new(ms_type),
                scan: scan.as_slice()?.to_vec(),
                tof: tof.as_slice()?.to_vec(),
                intensity: intensity.as_slice()?.to_vec(),
            },
        })
    }

    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.intensity.clone().into_pyarray_bound(py).unbind()
    }
    #[getter]
    pub fn scan(&self, py: Python) -> Py<PyArray1<u32>> {
        self.inner.scan.clone().into_pyarray_bound(py).unbind()
    }

    #[getter]
    pub fn tof(&self, py: Python) -> Py<PyArray1<u32>> {
        self.inner.tof.clone().into_pyarray_bound(py).unbind()
    }

    #[getter]
    pub fn frame_id(&self) -> i32 {
        self.inner.frame_id
    }

    #[getter]
    pub fn ms_type_numeric(&self) -> i32 {
        self.inner.ms_type.ms_type_numeric()
    }

    #[getter]
    pub fn ms_type(&self) -> String {
        self.inner.ms_type.to_string()
    }

    #[getter]
    pub fn retention_time(&self) -> f64 {
        self.inner.retention_time
    }
}

#[pyclass]
pub struct PyTimsFrame {
    pub inner: TimsFrame,
    // Cached numpy arrays - lazily initialized on first access (stored as PyObject for Clone)
    mz_cache: OnceLock<PyObject>,
    intensity_cache: OnceLock<PyObject>,
    mobility_cache: OnceLock<PyObject>,
    scan_cache: OnceLock<PyObject>,
    tof_cache: OnceLock<PyObject>,
}

impl Clone for PyTimsFrame {
    fn clone(&self) -> Self {
        // Clone resets caches - new instance lazily creates its own numpy arrays
        PyTimsFrame {
            inner: self.inner.clone(),
            mz_cache: OnceLock::new(),
            intensity_cache: OnceLock::new(),
            mobility_cache: OnceLock::new(),
            scan_cache: OnceLock::new(),
            tof_cache: OnceLock::new(),
        }
    }
}

impl PyTimsFrame {
    /// Create a new PyTimsFrame from a TimsFrame (internal use)
    pub fn from_inner(inner: TimsFrame) -> Self {
        PyTimsFrame {
            inner,
            mz_cache: OnceLock::new(),
            intensity_cache: OnceLock::new(),
            mobility_cache: OnceLock::new(),
            scan_cache: OnceLock::new(),
            tof_cache: OnceLock::new(),
        }
    }
}

#[pymethods]
impl PyTimsFrame {

    #[new]
    pub unsafe fn new(frame_id: i32,
                      ms_type: i32,
                      retention_time: f64,
                      scan:&Bound<'_, PyArray1<i32>>,
                      mobility: &Bound<'_, PyArray1<f64>>,
                      tof: &Bound<'_, PyArray1<i32>>,
                      mz: &Bound<'_, PyArray1<f64>>,
                      intensity: &Bound<'_, PyArray1<f64>>) -> PyResult<Self> {
        Ok(PyTimsFrame {
            inner: TimsFrame::new(
                frame_id,
                MsType::new(ms_type),
                retention_time,
                scan.as_slice()?.to_vec(),
                mobility.as_slice()?.to_vec(),
                tof.as_slice()?.to_vec(),
                mz.as_slice()?.to_vec(),
                intensity.as_slice()?.to_vec(),
            ),
            mz_cache: OnceLock::new(),
            intensity_cache: OnceLock::new(),
            mobility_cache: OnceLock::new(),
            scan_cache: OnceLock::new(),
            tof_cache: OnceLock::new(),
        })
    }
    #[getter]
    pub fn mz(&self, py: Python) -> PyObject {
        self.mz_cache.get_or_init(|| {
            (*self.inner.ims_frame.mz).clone().into_pyarray_bound(py).unbind().into_any()
        }).clone_ref(py)
    }
    #[getter]
    pub fn intensity(&self, py: Python) -> PyObject {
        self.intensity_cache.get_or_init(|| {
            (*self.inner.ims_frame.intensity).clone().into_pyarray_bound(py).unbind().into_any()
        }).clone_ref(py)
    }
    #[getter]
    pub fn scan(&self, py: Python) -> PyObject {
        self.scan_cache.get_or_init(|| {
            self.inner.scan.clone().into_pyarray_bound(py).unbind().into_any()
        }).clone_ref(py)
    }
    #[getter]
    pub fn mobility(&self, py: Python) -> PyObject {
        self.mobility_cache.get_or_init(|| {
            (*self.inner.ims_frame.mobility).clone().into_pyarray_bound(py).unbind().into_any()
        }).clone_ref(py)
    }
    #[getter]
    pub fn tof(&self, py: Python) -> PyObject {
        self.tof_cache.get_or_init(|| {
            self.inner.tof.clone().into_pyarray_bound(py).unbind().into_any()
        }).clone_ref(py)
    }
    #[setter]
    pub unsafe fn set_tof(&mut self, tof: &Bound<'_, PyArray1<i32>>) {
        self.inner.tof = tof.as_slice().unwrap().to_vec();
        // Clear the cache since data changed
        self.tof_cache = OnceLock::new();
    }

    #[getter]
    pub fn frame_id(&self) -> i32 {
        self.inner.frame_id
    }
    #[getter]
    pub fn ms_type_numeric(&self) -> i32 {
        self.inner.ms_type.ms_type_numeric()
    }
    #[getter]
    pub fn ms_type(&self) -> String {
        self.inner.ms_type.to_string()
    }
    #[getter]
    pub fn retention_time(&self) -> f64 {
        self.inner.ims_frame.retention_time
    }

    pub fn to_resolution(&self, resolution: i32) -> PyTimsFrame {
        PyTimsFrame::from_inner(self.inner.to_resolution(resolution))
    }

    pub fn get_tims_spectrum(&self, scan_index: i32) -> Option<PyTimsSpectrum> {
        self.inner.get_tims_spectrum(scan_index).map(|spectrum| PyTimsSpectrum::from_inner(spectrum))
    }

    pub fn to_tims_spectra(&self, py: Python) -> PyResult<Py<PyList>> {
        let spectra = self.inner.to_tims_spectra();
        let list: Py<PyList> = PyList::empty_bound(py).into();

        for spec in spectra {
            let py_tims_spectrum = Py::new(py, PyTimsSpectrum::from_inner(spec))?;
            list.bind(py).append(py_tims_spectrum)?;
        }

        Ok(list.into())
    }

    pub fn to_windows(&self, py: Python, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64) -> PyResult<Py<PyList>> {

        let windows = self.inner.to_windows(window_length, overlapping, min_peaks, min_intensity);
        let list: Py<PyList> = PyList::empty_bound(py).into();

        for window in windows {
            let py_mz_spectrum = Py::new(py, PyTimsSpectrum::from_inner(window))?;
            list.bind(py).append(py_mz_spectrum)?;
        }

        Ok(list.into())
    }

    pub fn to_indexed_mz_spectrum(&self) -> PyIndexedMzSpectrum {
        PyIndexedMzSpectrum::from_inner(self.inner.to_indexed_mz_spectrum())
    }

    pub fn vectorized(&self, resolution: i32) -> PyTimsFrameVectorized {
        let vectorized = self.inner.vectorized(resolution);
        let py_vectorized = PyTimsFrameVectorized {
            inner: vectorized,
        };
        py_vectorized
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, inv_mob_min: f64, inv_mob_max: f64, intensity_min: f64, intensity_max: f64) -> PyTimsFrame {
        PyTimsFrame::from_inner(self.inner.filter_ranged(mz_min, mz_max, scan_min, scan_max, inv_mob_min, inv_mob_max, intensity_min, intensity_max))
    }

    pub fn get_inverse_mobility_along_scan_marginal(&self) -> f64 {
        self.inner.get_inverse_mobility_along_scan_marginal()
    }

    pub fn get_mobility_mean_and_variance(&self) -> (f64, f64) {
        self.inner.get_mobility_mean_and_variance()
    }

    #[staticmethod]
    pub fn from_windows(_py: Python, windows: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut spectra: Vec<TimsSpectrum> = Vec::new();
        for window in windows.iter() {
            let window: PyRef<PyTimsSpectrum> = window.extract()?;
            spectra.push(window.inner.clone());
        }

        Ok(PyTimsFrame::from_inner(TimsFrame::from_windows(spectra)))
    }

    #[staticmethod]
    pub fn from_tims_spectra(_py: Python, spectra: Vec<PyTimsSpectrum>) -> PyResult<Self> {
        // Use into_iter() to move ownership instead of cloning each spectrum
        Ok(PyTimsFrame::from_inner(TimsFrame::from_tims_spectra(spectra.into_iter().map(|s| s.inner).collect())))
    }

    pub fn to_dense_windows(&self, py: Python, window_length: f64, resolution: i32, overlapping: bool, min_peaks: usize, min_intensity: f64) -> PyResult<PyObject> {

        let (data, mobilities, mzs, scans, window_indices, rows, cols) = self.inner.to_dense_windows(window_length, overlapping, min_peaks, min_intensity, resolution);
        let py_array: Bound<'_, PyArray1<f64>> = data.into_pyarray_bound(py);
        let mobilities: Bound<'_, PyArray1<f64>> = mobilities.into_pyarray_bound(py);
        let mzs: Bound<'_, PyArray1<f64>> = mzs.into_pyarray_bound(py);
        let py_scans: Bound<'_, PyArray1<i32>> = scans.into_pyarray_bound(py);
        let py_window_indices: Bound<'_, PyArray1<i32>> = window_indices.into_pyarray_bound(py);

        // If you need them outside the GIL context, unbind them:
        let py_array = py_array.unbind();
        let py_scans = py_scans.unbind();
        let py_window_indices = py_window_indices.unbind();
        let tuple = PyTuple::new_bound(py, &[rows.to_owned().into_py(py), cols.to_owned().into_py(py),
            py_array.into_py(py),
            mobilities.into_py(py),
            mzs.into_py(py),
            py_scans.into_py(py),
            py_window_indices.into_py(py)]);

        Ok(tuple.into())
    }

    pub fn to_noise_annotated_tims_frame(&self) -> PyTimsFrameAnnotated {
        let result = self.inner.to_noise_annotated_tims_frame();
        PyTimsFrameAnnotated { inner: result }
    }

    pub fn __add__(&self, other: PyTimsFrame) -> PyTimsFrame {
        // Only clone self.inner (borrowed), other.inner can be moved
        let result = self.inner.clone() + other.inner;
        PyTimsFrame::from_inner(result)
    }

    pub fn random_subsample_frame(&self, take_probability: f64) -> PyTimsFrame {
        let result = self.inner.generate_random_sample(take_probability);
        PyTimsFrame::from_inner(result)
    }

    pub fn fold_along_scan_axis(&self, fold_width: usize) -> PyTimsFrame {
        let folded_frame = self.inner.clone().fold_along_scan_axis(fold_width);
        PyTimsFrame::from_inner(folded_frame)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyTimsFrameVectorized {
    pub inner: TimsFrameVectorized,
}

#[pymethods]
impl  PyTimsFrameVectorized {
    #[new]
   pub unsafe fn new(frame_id: i32,
                     ms_type: i32,
                     retention_time: f64,
                     scan: &Bound<'_, PyArray1<i32>>,
                     mobility: &Bound<'_, PyArray1<f64>>,
                     tof: &Bound<'_, PyArray1<i32>>,
                     indices: &Bound<'_, PyArray1<i32>>,
                     intensity: &Bound<'_, PyArray1<f64>>,
                    resolution: i32,
                    ) -> PyResult<Self> {
       Ok(PyTimsFrameVectorized {
           inner: TimsFrameVectorized {
               frame_id,
               ms_type: MsType::new(ms_type),
               scan: scan.as_slice()?.to_vec(),
               tof: tof.as_slice()?.to_vec(),
               ims_frame: ImsFrameVectorized {
                   retention_time,
                   mobility: mobility.as_slice()?.to_vec(),
                   indices: indices.as_slice()?.to_vec(),
                   values: intensity.as_slice()?.to_vec(),
                   resolution,
               },
           },
       })
   }
   #[getter]
    pub fn indices(&self, py: Python) -> Py<PyArray1<i32>> {
         self.inner.ims_frame.indices.clone().into_pyarray_bound(py).unbind()
    }

    #[getter]
    pub fn values(&self, py: Python) -> Py<PyArray1<f64>> {
         self.inner.ims_frame.values.clone().into_pyarray_bound(py).unbind()
    }

    #[getter]
    pub fn resolution(&self) -> i32 {
         self.inner.ims_frame.resolution
    }

    #[getter]
    pub fn scan(&self, py: Python) -> Py<PyArray1<i32>> {
         self.inner.scan.clone().into_pyarray_bound(py).unbind()
    }

    #[getter]
    pub fn mobility(&self, py: Python) -> Py<PyArray1<f64>> {
         self.inner.ims_frame.mobility.clone().into_pyarray_bound(py).unbind()
    }

    #[getter]
    pub fn tof(&self, py: Python) -> Py<PyArray1<i32>> {
         self.inner.tof.clone().into_pyarray_bound(py).unbind()
    }

    #[getter]
    pub fn frame_id(&self) -> i32 {
         self.inner.frame_id
    }

    #[getter]
    pub fn ms_type_numeric(&self) -> i32 {
         self.inner.ms_type.ms_type_numeric()
    }

    #[getter]
    pub fn ms_type(&self) -> String {
         self.inner.ms_type.to_string()
    }

    #[getter]
    pub fn retention_time(&self) -> f64 {
         self.inner.ims_frame.retention_time
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, inv_mob_min: f64, inv_mob_max: f64, intensity_min: f64, intensity_max: f64) -> PyTimsFrameVectorized {
        return PyTimsFrameVectorized { inner: self.inner.filter_ranged(mz_min, mz_max, scan_min, scan_max, inv_mob_min, inv_mob_max, intensity_min, intensity_max) }
    }

}

#[pymodule]
pub fn py_tims_frame(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimsFrame>()?;
    m.add_class::<PyTimsFrameVectorized>()?;
    m.add_class::<PyRawTimsFrame>()?;
    Ok(())
}

