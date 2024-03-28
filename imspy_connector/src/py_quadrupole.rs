use std::collections::HashSet;
use mscore::timstof::collision::{TimsTofCollisionEnergy, TimsTofCollisionEnergyDIA};
use pyo3::prelude::*;

use mscore::timstof::quadrupole::{IonTransmission, TimsTransmissionDIA};
use crate::py_mz_spectrum::PyMzSpectrum;
use crate::py_tims_frame::PyTimsFrame;

#[pyclass(unsendable)]
pub struct PyTimsTransmissionDIA {
    pub inner: TimsTransmissionDIA,
}

#[pymethods]
impl PyTimsTransmissionDIA {
    #[new]
    pub fn new(frame: Vec<i32>,
               frame_window_group: Vec<i32>,
               window_group: Vec<i32>,
               scan_start: Vec<i32>,
               scan_end: Vec<i32>,
               isolation_mz: Vec<f64>,
               isolation_width: Vec<f64>,
               k: Option<f64>) -> Self {
        PyTimsTransmissionDIA {
            inner: TimsTransmissionDIA::new(
            frame,
            frame_window_group,
            window_group,
            scan_start,
            scan_end,
            isolation_mz,
            isolation_width,
            k)
        }
    }

    pub fn apply_transmission(&self, frame_id: i32, scan_id: i32, mz: Vec<f64>) -> Vec<f64> {
        self.inner.apply_transmission(frame_id, scan_id, &mz)
    }

    pub fn transmit_spectrum(&self, frame_id: i32, scan_id: i32, spectrum: PyMzSpectrum, min_probability: Option<f64>) -> PyMzSpectrum {
        PyMzSpectrum { inner: self.inner.transmit_spectrum(frame_id, scan_id, spectrum.inner, min_probability) }
    }

    pub fn transmit_tims_frame(&self, frame: PyTimsFrame, min_probability: Option<f64>) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.transmit_tims_frame(&frame.inner, min_probability) }
    }

    pub fn transmit_ion(&self, frames: Vec<i32>, scans: Vec<i32>, spectrum: PyMzSpectrum, min_proba: Option<f64>) -> Vec<Vec<PyMzSpectrum>> {
        let transmission_profile = self.inner.transmit_ion(frames, scans, spectrum.inner, min_proba);
        transmission_profile.iter().map(|x| x.iter().map(|y| PyMzSpectrum { inner: y.clone() }).collect::<Vec<_>>()).collect::<Vec<_>>()
    }

    pub fn get_setting(&self, window_group: i32, scan_id: i32) -> (f64, f64) {
        let result = self.inner.get_setting(window_group, scan_id);
        match result {
            Some((a, b)) => (*a, *b),
            None => (-1.0, -1.0)
        }
    }

    pub fn frame_to_window_group(&self, frame_id: i32) -> i32 {
        self.inner.frame_to_window_group(frame_id)
    }

    pub fn is_transmitted(&self, frame_id: i32, scan_id: i32, mz: f64, min_proba: Option<f64>) -> bool {
        self.inner.is_transmitted(frame_id, scan_id, mz, min_proba)
    }

    pub fn any_transmitted(&self, frame_id: i32, scan_id: i32, mz: Vec<f64>, min_proba: Option<f64>) -> bool {
        self.inner.any_transmitted(frame_id, scan_id, &mz, min_proba)
    }

    pub fn all_transmitted(&self, frame_id: i32, scan_id: i32, mz: Vec<f64>, min_proba: Option<f64>) -> bool {
        self.inner.all_transmitted(frame_id, scan_id, &mz, min_proba)
    }

    pub fn get_transmission_set(&self, frame_id: i32, scan_id: i32, mz: Vec<f64>, min_proba: Option<f64>) -> HashSet<usize> {
        self.inner.get_transmission_set(frame_id, scan_id, &mz, min_proba)
    }

    pub fn is_precursor(&self, frame_id: i32) -> bool {
        self.inner.is_precursor(frame_id)
    }

    pub fn isotopes_transmitted(&self, frame_id: i32, scan_id: i32, mz_mono: f64, mz: Vec<f64>, min_proba: Option<f64>) -> (f64, Vec<(f64, f64)>) {
        self.inner.isotopes_transmitted(frame_id, scan_id, mz_mono, &mz, min_proba)
    }
}

#[pyclass]
pub struct PyTimsTofCollisionEnergyDIA {
    pub inner: TimsTofCollisionEnergyDIA,
}

#[pymethods]
impl PyTimsTofCollisionEnergyDIA {
    #[new]
    pub fn new(frame: Vec<i32>,
               frame_window_group: Vec<i32>,
               window_group: Vec<i32>,
               scan_start: Vec<i32>,
               scan_end: Vec<i32>,
               collision_energy: Vec<f64>) -> Self {
        PyTimsTofCollisionEnergyDIA {
            inner: TimsTofCollisionEnergyDIA::new(
            frame,
            frame_window_group,
            window_group,
            scan_start,
            scan_end,
            collision_energy)
        }
    }

    pub fn get_collision_energy(&self, frame_id: i32, scan_id: i32) -> f64 {
        self.inner.get_collision_energy(frame_id, scan_id)
    }

    pub fn get_collision_energies(&self, frame_ids: Vec<i32>, scan_ids: Vec<i32>) -> Vec<f64> {
        let mut collision_energies = Vec::with_capacity(frame_ids.len());
        for (frame_id, scan_id) in frame_ids.iter().zip(scan_ids.iter()) {
            collision_energies.push(self.inner.get_collision_energy(*frame_id, *scan_id));
        }
        collision_energies
    }
}

#[pyfunction]
pub fn apply_transmission(midpoint: f64, window_length: f64, mz: Vec<f64>, k: Option<f64>) -> Vec<f64> {
    mscore::timstof::quadrupole::apply_transmission(midpoint, window_length, k.unwrap_or(15.0), mz)
}

#[pymodule]
pub fn quadrupole(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTimsTransmissionDIA>()?;
    m.add_class::<PyTimsTofCollisionEnergyDIA>()?;
    m.add_function(wrap_pyfunction!(apply_transmission, m)?)?;
    Ok(())
}
