use pyo3::prelude::*;

use rustdf::data::handle::{TimsDataHandle};
use crate::py_tims_frame::{PyTimsFrame, PyImsFrame};
#[pyclass]
pub struct PyTimsDataHandle {
    inner: TimsDataHandle,
}

#[pymethods]
impl PyTimsDataHandle {
    #[new]
    pub fn new(data_path: &str, bruker_lib_path: &str) -> Self {
        let dataset = TimsDataHandle::new(bruker_lib_path, data_path).unwrap();
        PyTimsDataHandle { inner: dataset }
    }

    #[getter]
    pub fn get_data_path(&self) -> &str {
        &self.inner.data_path
    }

    #[getter]
    pub fn get_bruker_lib_path(&self) -> &str {
        &self.inner.bruker_lib_path
    }
    pub fn get_tims_frame(&self, frame_id: u32) -> PyResult<PyTimsFrame> {
        let frame = self.inner.get_frame(frame_id).unwrap();
        Ok(PyTimsFrame { inner: frame })
    }

    pub fn get_ims_frame(&self, frame_id: u32) -> PyResult<PyImsFrame> {
        let frame = self.inner.get_frame(frame_id).unwrap();
        Ok(PyImsFrame { inner: frame.get_ims_frame() })
    }

    pub fn get_acquisition_mode(&self) -> i32 {
       self.inner.acquisition_mode.to_i32()
    }

    pub fn get_acquisition_mode_as_string(&self) -> String {
       self.inner.acquisition_mode.to_string()
    }
}