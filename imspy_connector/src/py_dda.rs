use pyo3::prelude::*;

use rustdf::data::dda::{PASEFDDAFragment, TimsDatasetDDA};
use rustdf::data::handle::TimsData;
use rustdf::data::meta::{DDAPrecursor};
use crate::py_tims_frame::PyTimsFrame;
use crate::py_tims_slice::PyTimsSlice;

#[pyclass]
pub struct PyDDAPrecursor {
    inner: DDAPrecursor,
}

#[pymethods]
impl PyDDAPrecursor {
    #[new]
    #[pyo3(signature = (frame_id, precursor_id, highest_intensity_mz, average_mz, inverse_ion_mobility, collision_energy, precuror_total_intensity, isolation_mz, isolation_width, mono_mz=None, charge=None))]
    pub fn new(
        frame_id: i64,
        precursor_id: i64,
        highest_intensity_mz: f64,
        average_mz: f64,
        inverse_ion_mobility: f64,
        collision_energy: f64,
        precuror_total_intensity: f64,
        isolation_mz: f64,
        isolation_width: f64,
        mono_mz: Option<f64>,
        charge: Option<i64>,
    ) -> Self {
        let precursor = DDAPrecursor {
            frame_id,
            precursor_id,
            mono_mz,
            highest_intensity_mz,
            average_mz,
            charge,
            inverse_ion_mobility,
            collision_energy,
            precuror_total_intensity,
            isolation_mz,
            isolation_width,
        };
        PyDDAPrecursor { inner: precursor }
    }

    #[getter]
    pub fn frame_id(&self) -> i64 { self.inner.frame_id }

    #[getter]
    pub fn precursor_id(&self) -> i64 { self.inner.precursor_id }

    #[getter]
    pub fn mono_mz(&self) -> Option<f64> { self.inner.mono_mz }

    #[getter]
    pub fn highest_intensity_mz(&self) -> f64 { self.inner.highest_intensity_mz }

    #[getter]
    pub fn average_mz(&self) -> f64 { self.inner.average_mz }

    #[getter]
    pub fn charge(&self) -> Option<i64> { self.inner.charge }

    #[getter]
    pub fn inverse_ion_mobility(&self) -> f64 { self.inner.inverse_ion_mobility }

    #[getter]
    pub fn collision_energy(&self) -> f64 { self.inner.collision_energy }

    #[getter]
    pub fn precuror_total_intensity(&self) -> f64 { self.inner.precuror_total_intensity }

    #[getter]
    pub fn isolation_mz(&self) -> f64 { self.inner.isolation_mz }

    #[getter]
    pub fn isolation_width(&self) -> f64 { self.inner.isolation_width }
}

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

    pub fn get_selected_precursors(&self) -> Vec<PyDDAPrecursor> {
        let pasef_precursor_meta = self.inner.get_selected_precursors();
        pasef_precursor_meta.iter().map(|precursor_meta| PyDDAPrecursor { inner: precursor_meta.clone() }).collect()
    }

    pub fn get_precursor_frames(&self, min_intensity: f64, max_peaks: usize, num_threads: usize) -> Vec<PyTimsFrame> {
        let precursor_frames = self.inner.get_precursor_frames(min_intensity, max_peaks, num_threads);
        precursor_frames.iter().map(|frame| PyTimsFrame { inner: frame.clone() }).collect()
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
pub fn py_dda(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimsDatasetDDA>()?;
    m.add_class::<PyTimsFragmentDDA>()?;
    m.add_class::<PyDDAPrecursor>()?;
    Ok(())
}