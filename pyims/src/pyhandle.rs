use pyo3::prelude::*;

use rustdf::data::handle::{TimsDataset};
use crate::py_mz_spectrum::PyTimsFrame;

#[pyclass]
pub struct PyTimsDataset {
    inner: TimsDataset,
}

#[pymethods]
impl PyTimsDataset {
    #[new]
    pub fn new(data_path: &str, bruker_lib_path: &str) -> Self {
        let dataset = TimsDataset::new(bruker_lib_path, data_path).unwrap();
        PyTimsDataset { inner: dataset }
    }
    #[getter]
    pub fn get_data_path(&self) -> &str {
        &self.inner.data_path
    }
    #[getter]
    pub fn get_bruker_lib_path(&self) -> &str {
        &self.inner.bruker_lib_path
    }


    pub fn get_frame(&self, frame_id: u32) -> PyResult<PyTimsFrame> {
        let frame = self.inner.get_frame(frame_id).unwrap();
        Ok(PyTimsFrame { inner: frame })
    }
}