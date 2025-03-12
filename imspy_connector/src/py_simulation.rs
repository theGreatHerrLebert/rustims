use std::collections::BTreeMap;
use mscore::timstof::collision::TimsTofCollisionEnergy;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rustdf::sim::dda::TimsTofSyntheticsFrameBuilderDDA;
use rustdf::sim::dia::{TimsTofSyntheticsFrameBuilderDIA};
use rustdf::sim::precursor::{TimsTofSyntheticsPrecursorFrameBuilder};
use rustdf::sim::handle::TimsTofSyntheticsDataHandle;
use crate::py_annotation::PyTimsFrameAnnotated;
use crate::py_quadrupole::PyPasefMeta;
use crate::py_tims_frame::PyTimsFrame;

#[pyclass]
pub struct PyTimsTofSyntheticsDataHandle {
    pub inner: TimsTofSyntheticsDataHandle,
}

#[pymethods]
impl PyTimsTofSyntheticsDataHandle {
    #[new]
    pub fn new(db_path: &str) -> Self {
        let path = std::path::Path::new(db_path);
        PyTimsTofSyntheticsDataHandle { inner: TimsTofSyntheticsDataHandle::new(path).unwrap() }
    }

    #[pyo3(signature = (num_threads=None, dda=None))]
    pub fn get_transmitted_ions(&self, num_threads: Option<usize>, dda: Option<bool>) -> (Vec<i32>, Vec<i32>, Vec<String>, Vec<i8>, Vec<f32>) {
        let threads = num_threads.unwrap_or(4);
        self.inner.get_transmitted_ions(threads, dda.unwrap_or(false))
    }
}

#[pyclass]
pub struct PyTimsTofSyntheticsPrecursorFrameBuilder {
    pub inner: TimsTofSyntheticsPrecursorFrameBuilder,
}

#[pymethods]
impl PyTimsTofSyntheticsPrecursorFrameBuilder {
    #[new]
    pub fn new(db_path: &str) -> Self {
        let path = std::path::Path::new(db_path);
        PyTimsTofSyntheticsPrecursorFrameBuilder { inner: TimsTofSyntheticsPrecursorFrameBuilder::new(path).unwrap() }
    }

    pub fn build_precursor_frame(&self, frame_id: u32, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, right_drag: bool) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.build_precursor_frame(frame_id, mz_noise_precursor, uniform, precursor_noise_ppm, right_drag) }
    }

    pub fn build_precursor_frames(&self, frame_ids: Vec<u32>, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, right_drag: bool, num_threads: usize) -> Vec<PyTimsFrame> {
        let frames = self.inner.build_precursor_frames(frame_ids, mz_noise_precursor, uniform, precursor_noise_ppm, right_drag, num_threads);
        frames.iter().map(|x| PyTimsFrame { inner: x.clone() }).collect::<Vec<_>>()
    }

    pub fn build_precursor_frame_annotated(&self, frame_id: u32, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, right_drag: bool) -> PyTimsFrameAnnotated {
        PyTimsFrameAnnotated { inner: self.inner.build_precursor_frame_annotated(frame_id, mz_noise_precursor, uniform, precursor_noise_ppm, right_drag) }
    }

    pub fn build_precursor_frames_annotated(&self, frame_ids: Vec<u32>, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, right_drag: bool, num_threads: usize) -> Vec<PyTimsFrameAnnotated> {
        let frames = self.inner.build_precursor_frames_annotated(frame_ids, mz_noise_precursor, uniform, precursor_noise_ppm, right_drag, num_threads);
        frames.iter().map(|x| PyTimsFrameAnnotated { inner: x.clone() }).collect::<Vec<_>>()
    }
    pub fn frame_to_abundances(&self) -> BTreeMap<u32, (Vec<u32>, Vec<f32>)> {
        self.inner.frame_to_abundances.clone()
    }
}

#[pyclass(unsendable)]
pub struct PyTimsTofSyntheticsFrameBuilderDIA {
    pub inner: TimsTofSyntheticsFrameBuilderDIA,
}

#[pymethods]
impl PyTimsTofSyntheticsFrameBuilderDIA {
    #[new]
    pub fn new(db_path: &str, with_annotations: bool, num_threads: usize) -> Self {
        let path = std::path::Path::new(db_path);
        PyTimsTofSyntheticsFrameBuilderDIA { inner: TimsTofSyntheticsFrameBuilderDIA::new(path, with_annotations, num_threads).unwrap() }
    }

    pub fn build_frame(&self, frame_id: u32, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool) -> PyTimsFrame {
        let frames = self.inner.build_frames(vec![frame_id], fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, 1);
        PyTimsFrame { inner: frames[0].clone() }
    }

    pub fn build_frame_annotated(&self, frame_id: u32, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool) -> PyTimsFrameAnnotated {
        let frames = self.inner.build_frames_annotated(vec![frame_id], fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, 1);
        PyTimsFrameAnnotated { inner: frames[0].clone() }
    }

    pub fn build_frames(&self, frame_ids: Vec<u32>, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool, num_threads: usize) -> Vec<PyTimsFrame> {
        let frames = self.inner.build_frames(frame_ids, fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, num_threads);
        frames.iter().map(|x| PyTimsFrame { inner: x.clone() }).collect::<Vec<_>>()
    }

    pub fn build_frames_annotated(&self, frame_ids: Vec<u32>, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool, num_threads: usize) -> Vec<PyTimsFrameAnnotated> {
        let frames = self.inner.build_frames_annotated(frame_ids, fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, num_threads);
        frames.iter().map(|x| PyTimsFrameAnnotated { inner: x.clone() }).collect::<Vec<_>>()
    }

    pub fn get_collision_energy(&self, frame_id: i32, scan_id: i32) -> f64 {
        self.inner.get_collision_energy(frame_id, scan_id)
    }

    pub fn get_collision_energies(&self, frame_ids: Vec<i32>, scan_ids: Vec<i32>) -> Vec<f64> {
        let mut result = Vec::with_capacity(frame_ids.len());
        for (frame_id, scan_id) in frame_ids.iter().zip(scan_ids.iter()) {
            result.push(self.inner.get_collision_energy(*frame_id, *scan_id));
        }
        result
    }

    pub fn get_ion_transmission_matrix(&self, peptide_id: u32, charge: i8, include_precursor_frames: bool) -> Vec<Vec<f32>> {
        self.inner.get_ion_transmission_matrix(peptide_id, charge, include_precursor_frames)
    }

    pub fn count_number_transmissions(&self, py: Python, peptide_id: u32, charge: i8) -> PyResult<PyObject> {
        let (frame_count, scan_count) = self.inner.count_number_transmissions(peptide_id, charge);
        let tuple = PyTuple::new_bound(py, &[frame_count.to_owned().into_py(py), scan_count.to_owned().into_py(py)]);
        Ok(tuple.into())
    }

    pub fn count_number_transmissions_parallel(&self, peptide_ids: Vec<u32>, charge: Vec<i8>, num_threads: usize) -> Vec<(usize, usize)> {
        self.inner.count_number_transmissions_parallel(peptide_ids, charge, num_threads)
    }
}

#[pyclass(unsendable)]
pub struct PyTimsTofSyntheticsFrameBuilderDDA {
    pub inner: TimsTofSyntheticsFrameBuilderDDA,
}

#[pymethods]
impl PyTimsTofSyntheticsFrameBuilderDDA {
    #[new]
    pub fn new(db_path: &str, with_annotations: bool, num_threads: usize) -> Self {
        let path = std::path::Path::new(db_path);
        PyTimsTofSyntheticsFrameBuilderDDA { inner: TimsTofSyntheticsFrameBuilderDDA::new(path, with_annotations, num_threads) }
    }

    pub fn build_frame(&self, frame_id: u32, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool) -> PyTimsFrame {
        let frames = self.inner.build_frames(vec![frame_id], fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, 1);
        PyTimsFrame { inner: frames[0].clone() }
    }

    pub fn build_frame_annotated(&self, frame_id: u32, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool) -> PyTimsFrameAnnotated {
        let frames = self.inner.build_frames_annotated(vec![frame_id], fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, 1);
        PyTimsFrameAnnotated { inner: frames[0].clone() }
    }

    pub fn build_frames(&self, frame_ids: Vec<u32>, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool, num_threads: usize) -> Vec<PyTimsFrame> {
        let frames = self.inner.build_frames(frame_ids, fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, num_threads);
        frames.iter().map(|x| PyTimsFrame { inner: x.clone() }).collect::<Vec<_>>()
    }

    pub fn build_frames_annotated(&self, frame_ids: Vec<u32>, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool, num_threads: usize) -> Vec<PyTimsFrameAnnotated> {
        let frames = self.inner.build_frames_annotated(frame_ids, fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, num_threads);
        frames.iter().map(|x| PyTimsFrameAnnotated { inner: x.clone() }).collect::<Vec<_>>()
    }

    pub fn get_pasef_meta(&self) -> Vec<PyPasefMeta> {
        let pasef_meta = self.inner.transmission_settings.pasef_meta.clone();
        // go over all key, list<value> pairs, extract the values, flatten
        let mut result = Vec::new();
        for (_, values) in pasef_meta.iter() {
            for value in values.iter() {
                result.push(PyPasefMeta { inner: value.clone() });
            }
        }

        result
    }

    pub fn get_fragment_frames(&self) -> Vec<i32> {
        // extract the keys from the pasef_meta to get all fragment frames sorted by frame_id ascending
        let pasef_meta = self.inner.transmission_settings.pasef_meta.clone();
        let mut result = Vec::new();
        for (key, _) in pasef_meta.iter() {
            result.push(key.clone());
        }
        result
    }
}

#[pymodule]
pub fn py_simulation(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimsTofSyntheticsDataHandle>()?;
    m.add_class::<PyTimsTofSyntheticsPrecursorFrameBuilder>()?;
    m.add_class::<PyTimsTofSyntheticsFrameBuilderDIA>()?;
    m.add_class::<PyTimsTofSyntheticsFrameBuilderDDA>()?;
    Ok(())
}