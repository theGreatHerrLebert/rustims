use pyo3::prelude::*;
use rustdf::data::dataset::TimsDataset;
use rustdf::data::handle::{TimsData, AcquisitionMode};

use crate::py_tims_frame::{PyTimsFrame};
use crate::py_tims_slice::PyTimsSlice;

#[pyclass]
pub struct PyTimsDataset {
    inner: TimsDataset,
}

#[pymethods]
impl PyTimsDataset {
    #[new]
    pub fn new(data_path: &str, bruker_lib_path: &str) -> Self {
        let dataset = TimsDataset::new(bruker_lib_path, data_path);
        PyTimsDataset { inner: dataset }
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

    pub fn frame_count(&self) -> i32 {
        self.inner.get_frame_count()
    }
}

#[pyclass]
pub struct PyAquisitionMode {
    inner: AcquisitionMode,
}

#[pymethods]
impl PyAquisitionMode {
    #[new]
    pub fn new(acquisition_mode: &str) -> Self {
        PyAquisitionMode { inner: AcquisitionMode::from(acquisition_mode) }
    }

    #[getter]
    pub fn acquisition_mode(&self) -> String {
        self.inner.to_str().to_string()
    }

    #[getter]
    pub fn acquisition_mode_numeric(&self) -> i32 {
        self.inner.to_i32()
    }
}