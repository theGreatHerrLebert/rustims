use pyo3::{pyclass, pymethods, pymodule, PyResult, Python};
use pyo3::prelude::PyModule;
use rustdf::data::in_memory::{TimsData, TimsDataLoader};
use crate::py_tims_frame::PyTimsFrame;

#[pyclass]
pub struct PyTimsDatasetInMemory {
    inner: TimsDataLoader,
}

#[pymethods]
impl PyTimsDatasetInMemory {
    #[new]
    pub fn new(data_path: &str, bruker_lib_path: &str) -> Self {
        let dataset = TimsDataLoader::new_in_memory(bruker_lib_path, data_path);
        PyTimsDatasetInMemory { inner: dataset }
    }
    pub fn get_frame(&self, frame_id: u32) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.get_frame(frame_id) }
    }
}

#[pymodule]
pub fn in_memory(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTimsDatasetInMemory>()?;
    Ok(())
}