use pyo3::prelude::*;

use rustdf::data::dda::{PASEFDDAFragment, TimsDatasetDDA};
use rustdf::data::handle::TimsData;
use crate::py_tims_frame::PyTimsFrame;
use crate::py_tims_slice::PyTimsSlice;

#[pyclass]
pub struct PyTimsDatasetDDA {
    inner: TimsDatasetDDA,
}

#[pymethods]
impl PyTimsDatasetDDA {
    #[new]
    pub fn new(data_path: &str, bruker_lib_path: &str, in_memory: bool, use_bruker_sdk: bool) -> Self {
        let dataset = TimsDatasetDDA::new(bruker_lib_path, data_path, in_memory, use_bruker_sdk);
        PyTimsDatasetDDA { inner: dataset }
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

    pub fn get_pasef_fragments(&self, num_threads: usize) -> Vec<PyTimsFragmentDDA> {
        let pasef_fragments = self.inner.get_pasef_fragments(num_threads);
        pasef_fragments.iter().map(|pasef_fragment| PyTimsFragmentDDA { inner: pasef_fragment.clone() }).collect()
    }
}

#[pyclass]
pub struct PyTimsFragmentDDA {
    inner: PASEFDDAFragment,
}

#[pymethods]
impl PyTimsFragmentDDA {
    #[new]
    pub fn new(frame_id: u32, precursor_id: u32, collision_energy: f64, selected_fragment: &PyTimsFrame) -> PyResult<Self> {

        let pasef_fragment = PASEFDDAFragment {
            frame_id,
            precursor_id,
            collision_energy,
            selected_fragment: selected_fragment.inner.clone(),
        };

        Ok(PyTimsFragmentDDA { inner: pasef_fragment })
    }

    #[getter]
    pub fn frame_id(&self) -> u32 { self.inner.frame_id }

    #[getter]
    pub fn precursor_id(&self) -> u32 { self.inner.precursor_id }

    #[getter]
    pub fn selected_fragment(&self) -> PyTimsFrame { PyTimsFrame { inner: self.inner.selected_fragment.clone() } }

    #[getter]
    pub fn collision_energy(&self) -> f64 { self.inner.collision_energy }
}

#[pymodule]
pub fn dda(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTimsDatasetDDA>()?;
    m.add_class::<PyTimsFragmentDDA>()?;
    Ok(())
}