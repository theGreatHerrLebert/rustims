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

    pub fn get_acquisition_mode(&self) -> String {
        self.inner.get_aquisition_mode().to_string()
    }

    pub fn get_acquisition_mode_numeric(&self) -> i32 {
        self.inner.get_aquisition_mode().to_i32()
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

    // TODO: make this more efficient
    pub fn mz_to_tof(&self, frame_id: u32, mz_values: Vec<f64>) -> Vec<u32> {
        self.inner.mz_to_tof(frame_id, &mz_values.clone())
    }

    // TODO: make this more efficient
    pub fn tof_to_mz(&self, frame_id: u32, tof_values: Vec<u32>) -> Vec<f64> {
        self.inner.tof_to_mz(frame_id, &tof_values.clone())
    }
}

#[pyclass]
pub struct PyAcquisitionMode {
    inner: AcquisitionMode,
}

#[pymethods]
impl PyAcquisitionMode {
    #[new]
    pub fn new(acquisition_mode: &str) -> Self {
        PyAcquisitionMode { inner: AcquisitionMode::from(acquisition_mode) }
    }

    #[staticmethod]
    pub fn from_numeric(acquisition_mode: i32) -> Self {
        PyAcquisitionMode { inner: AcquisitionMode::from(acquisition_mode) }
    }

    #[staticmethod]
    pub fn from_string(acquisition_mode: &str) -> Self {
        PyAcquisitionMode { inner: AcquisitionMode::from(acquisition_mode) }
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