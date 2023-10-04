use pyo3::prelude::*;
use mscore::TimsSlice;
use pyo3::types::PyList;
use crate::py_mz_spectrum::PyMzSpectrum;

use crate::py_tims_frame::{PyTimsFrame};

#[pyclass]
#[derive(Clone)]
pub struct PyTimsSlice {
    pub inner: TimsSlice,
}


#[pymethods]
impl PyTimsSlice {
    #[getter]
    pub fn first_frame_id(&self) -> i32 { self.inner.frames.first().unwrap().frame_id }

    #[getter]
    pub fn last_frame_id(&self) -> i32 { self.inner.frames.last().unwrap().frame_id }

    #[getter]
    pub fn frame_count(&self) -> i32 { self.inner.frames.len() as i32 }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, intensity_min: f64, num_threads: usize) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.filter_ranged(mz_min, mz_max, scan_min, scan_max, intensity_min, num_threads) }
    }
    pub fn get_frames(&self, py: Python) -> PyResult<Py<PyList>> {
        let frames = &self.inner.frames;
        let list: Py<PyList> = PyList::empty(py).into();

        for frame in frames {
            let py_tims_frame = Py::new(py, PyTimsFrame { inner: frame.clone() })?;
            list.as_ref(py).append(py_tims_frame)?;
        }

        Ok(list.into())
    }

    pub fn to_windows(&self, py: Python, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64, num_threads: usize) -> PyResult<Py<PyList>> {

        let windows = self.inner.to_windows(window_length, overlapping, min_peaks, min_intensity, num_threads);
        let list: Py<PyList> = PyList::empty(py).into();

        for window in windows {
            let py_mz_spectrum = Py::new(py, PyMzSpectrum { inner: window })?;
            list.as_ref(py).append(py_mz_spectrum)?;
        }

        Ok(list.into())
    }

    pub fn get_frame_at_index(&self, index: i32) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.frames[index as usize].clone() }
    }
}
