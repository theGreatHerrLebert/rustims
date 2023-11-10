use pyo3::prelude::*;

use rustdf::data::dia::TimsDatasetDIA;
use rustdf::data::handle::TimsData;
use crate::py_tims_frame::PyTimsFrame;
use crate::py_tims_slice::PyTimsSlice;

#[pyclass]
pub struct PyTimsDatasetDIA {
    inner: TimsDatasetDIA,
}

#[pymethods]
impl PyTimsDatasetDIA {
    #[new]
    pub fn new(data_path: &str, bruker_lib_path: &str) -> Self {
        let dataset = TimsDatasetDIA::new(bruker_lib_path, data_path);
        PyTimsDatasetDIA { inner: dataset }
    }
    pub fn get_frame(&self, frame_id: u32) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.get_frame(frame_id) }
    }

    pub fn get_slice(&self, frame_ids: Vec<u32>) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.get_slice(frame_ids) }
    }

    pub fn get_aquisition_mode(&self) -> String {
        self.inner.get_aquisition_mode().to_string()
    }

    pub fn get_frame_count(&self) -> i32 {
        self.inner.get_frame_count()
    }

    pub fn get_data_path(&self) -> &str {
        self.inner.get_data_path()
    }

    pub fn get_bruker_lib_path(&self) -> &str {
        self.inner.get_bruker_lib_path()
    }
}