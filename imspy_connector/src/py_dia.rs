use pyo3::prelude::*;

use rustdf::data::dia::TimsDatasetDIA;
use rustdf::data::handle::TimsData;
use crate::py_tims_frame::PyTimsFrame;
use crate::py_tims_slice::PyTimsSlice;
use numpy::{PyArray1, PyArray2};
use numpy::ndarray::{Array2, ShapeBuilder};

#[pyclass]
pub struct PyTimsDatasetDIA {
    inner: TimsDatasetDIA,
}

#[pymethods]
impl PyTimsDatasetDIA {
    #[new]
    pub fn new(data_path: &str, bruker_lib_path: &str, in_memory: bool, use_bruker_sdk: bool) -> Self {
        let dataset = TimsDatasetDIA::new(bruker_lib_path, data_path, in_memory, use_bruker_sdk);
        PyTimsDatasetDIA { inner: dataset }
    }
    pub fn get_frame(&self, frame_id: u32) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.get_frame(frame_id) }
    }

    pub fn get_slice(&self, frame_ids: Vec<u32>, num_threads: usize) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.get_slice(frame_ids, num_threads) }
    }

    pub fn get_acquisition_mode(&self) -> String {
        self.inner.get_acquisition_mode().to_string()
    }

    pub fn get_frame_count(&self) -> i32 {
        self.inner.get_frame_count()
    }

    pub fn get_data_path(&self) -> &str {
        self.inner.get_data_path()
    }
    
    pub fn sample_precursor_signal(&self, num_frames: usize, max_intensity: f64, take_probability: f64) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.sample_precursor_signal(num_frames, max_intensity, take_probability) }
    }
    
    pub fn sample_fragment_signal(&self, num_frames: usize, window_group: u32, max_intensity: f64, take_probability: f64) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.sample_fragment_signal(num_frames, window_group, max_intensity, take_probability) }
    }

    #[pyo3(signature = (resolution, num_threads, truncate, maybe_sigma_frames=None))]
    pub fn build_dense_rt_by_mz(
        &self,
        resolution: usize,
        num_threads: usize,
        truncate: f32,
        maybe_sigma_frames: Option<f32>,
        py: Python<'_>,
    ) -> PyResult<(
        Py<PyArray1<u32>>,
        Py<PyArray1<u32>>,
        Py<PyArray2<f32>>,
    )> {
        let rt = self.inner.get_dense_rt_by_mz(maybe_sigma_frames, truncate, resolution, num_threads);

        let bins_py   = PyArray1::from_vec_bound(py, rt.bins).unbind();
        let frames_py = PyArray1::from_vec_bound(py, rt.frames).unbind();

        let arr_f: Array2<f32> = Array2::from_shape_vec((rt.rows, rt.cols).f(), rt.data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("shape error: {e}")))?;
        let data_py = PyArray2::from_owned_array_bound(py, arr_f).unbind();

        Ok((bins_py, frames_py, data_py))
    }
}

#[pymodule]
pub fn py_dia(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimsDatasetDIA>()?;
    Ok(())
}