use pyo3::prelude::*;
use numpy::{PyArray1};

use rustdf::data::handle::{TimsDataHandle};
use crate::py_tims_frame::{PyTimsFrame};
use crate::py_tims_slice::PyTimsSlice;

#[pyclass(subclass)]
pub struct PyTimsDataset {
    inner: TimsDataHandle,
}

#[pymethods]
impl PyTimsDataset {
    #[new]
    pub fn new(data_path: &str, bruker_lib_path: &str) -> Self {
        let handle = TimsDataHandle::new(bruker_lib_path, data_path).unwrap();
        PyTimsDataset { inner: handle }
    }

    #[getter]
    pub fn get_data_path(&self) -> &str {
        &self.inner.data_path
    }

    #[getter]
    pub fn get_bruker_lib_path(&self) -> &str {
        &self.inner.bruker_lib_path
    }
    #[getter]
    pub fn frame_count(&self) -> i32 {
        self.inner.get_frame_count()
    }

    pub fn get_tims_frame(&self, frame_id: u32) -> PyResult<PyTimsFrame> {
        let frame = self.inner.get_frame(frame_id).unwrap();
        Ok(PyTimsFrame { inner: frame })
    }

    pub fn get_acquisition_mode(&self) -> i32 {
       self.inner.acquisition_mode.to_i32()
    }

    pub fn get_acquisition_mode_as_string(&self) -> String {
       self.inner.acquisition_mode.to_string()
    }

    pub fn get_tims_slice(&self, frame_ids: &PyArray1<i32>) -> PyTimsSlice {
        let frames = self.inner.get_tims_slice(frame_ids.to_vec().unwrap().iter().map(|f| *f as u32).collect());
        PyTimsSlice { inner: frames }
    }
}