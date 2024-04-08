use pyo3::{pyclass, pymethods, pymodule, PyResult, Python};
use pyo3::prelude::PyModule;
use rustdf::data::in_memory::TimsDatasetInMemory;

#[pyclass]
pub struct PyTimsDatasetInMemory {
    inner: TimsDatasetInMemory,
}

#[pymethods]
impl PyTimsDatasetInMemory {
    #[new]
    pub fn new(data_path: &str, bruker_lib_path: &str) -> Self {
        let dataset = TimsDatasetInMemory::new(bruker_lib_path, data_path);
        PyTimsDatasetInMemory { inner: dataset }
    }
}

#[pymodule]
pub fn in_memory(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTimsDatasetInMemory>()?;
    Ok(())
}