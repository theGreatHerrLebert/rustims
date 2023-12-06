use pyo3::prelude::*;
use rustdf::data::simulation::{TimsTofSyntheticsDDA};
use crate::py_tims_frame::PyTimsFrame;

#[pyclass]
pub struct PyTimsTofSyntheticsDDA {
    pub inner: TimsTofSyntheticsDDA,
}

#[pymethods]
impl PyTimsTofSyntheticsDDA {
    #[new]
    pub fn new(db_path: &str) -> Self {
        let path = std::path::Path::new(db_path);
        PyTimsTofSyntheticsDDA { inner: TimsTofSyntheticsDDA::new(path).unwrap() }
    }

    pub fn build_frame(&self, frame_id: i64) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.build_frame(frame_id) }
    }

    pub fn build_frames(&self, frame_ids: Vec<i64>, num_threads: usize) -> Vec<PyTimsFrame> {
        let frames = self.inner.build_frames(frame_ids, num_threads);
        frames.iter().map(|x| PyTimsFrame { inner: x.clone() }).collect::<Vec<_>>()
    }
}