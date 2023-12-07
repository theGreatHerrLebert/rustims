use pyo3::prelude::*;
use rustdf::data::simulation::{TimsTofSynthetics};
use crate::py_tims_frame::PyTimsFrame;

#[pyclass]
pub struct PyTimsTofSynthetics {
    pub inner: TimsTofSynthetics,
}

#[pymethods]
impl PyTimsTofSynthetics {
    #[new]
    pub fn new(db_path: &str) -> Self {
        let path = std::path::Path::new(db_path);
        PyTimsTofSynthetics { inner: TimsTofSynthetics::new(path).unwrap() }
    }

    pub fn build_frame(&self, frame_id: u32) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.build_frame(frame_id) }
    }

    pub fn build_frames(&self, frame_ids: Vec<u32>, num_threads: usize) -> Vec<PyTimsFrame> {
        let frames = self.inner.build_frames(frame_ids, num_threads);
        frames.iter().map(|x| PyTimsFrame { inner: x.clone() }).collect::<Vec<_>>()
    }
}