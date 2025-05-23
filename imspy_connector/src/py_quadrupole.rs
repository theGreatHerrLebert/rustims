use std::collections::HashSet;
use mscore::timstof::collision::{TimsTofCollisionEnergy, TimsTofCollisionEnergyDIA};
use pyo3::prelude::*;

use mscore::timstof::quadrupole::{IonTransmission, PASEFMeta, TimsTransmissionDDA, TimsTransmissionDIA};
use crate::py_mz_spectrum::PyMzSpectrum;
use crate::py_tims_frame::PyTimsFrame;

#[pyclass]
#[derive(Clone)]
pub struct PyPasefMeta {
    pub inner : PASEFMeta,
}

#[pymethods]
impl PyPasefMeta {
    #[new]
    pub fn new(frame: i32,
               scan_start: i32,
               scan_end: i32,
               isolation_mz: f64,
               isolation_width: f64,
               collision_energy: f64,
               precursor: i32) -> Self {
        PyPasefMeta {
            inner: PASEFMeta {
                frame,
                scan_start,
                scan_end,
                isolation_mz,
                isolation_width,
                collision_energy,
                precursor,
            }
        }
    }
    #[getter]
    pub fn frame(&self) -> i32 {
        self.inner.frame
    }
    #[getter]
    pub fn scan_start(&self) -> i32 {
        self.inner.scan_start
    }
    #[getter]
    pub fn scan_end(&self) -> i32 {
        self.inner.scan_end
    }
    #[getter]
    pub fn isolation_mz(&self) -> f64 {
        self.inner.isolation_mz
    }
    #[getter]
    pub fn isolation_width(&self) -> f64 {
        self.inner.isolation_width
    }
    #[getter]
    pub fn collision_energy(&self) -> f64 {
        self.inner.collision_energy
    }
    #[getter]
    pub fn precursor(&self) -> i32 {
        self.inner.precursor
    }
}

#[pyclass(unsendable)]
pub struct PyTimsTransmissionDDA {
    pub inner: TimsTransmissionDDA,
}

#[pymethods]
impl PyTimsTransmissionDDA {
    #[new]
    #[pyo3(signature = (pasef_meta, k=None))]
    pub fn new(pasef_meta: Vec<PyPasefMeta>, k: Option<f64>) -> Self {
        let inner_meta_vec = pasef_meta.iter().map(|x| x.inner.clone()).collect::<Vec<_>>();
        PyTimsTransmissionDDA {
            inner: TimsTransmissionDDA::new(inner_meta_vec, k)
        }
    }

    #[pyo3(signature = (frame_id, scan_id, mz))]
    pub fn apply_transmission(&self, frame_id: i32, scan_id: i32, mz: Vec<f64>) -> Vec<f64> {
        self.inner.apply_transmission(frame_id, scan_id, &mz)
    }

    #[pyo3(signature = (frame_id, scan_id, spectrum, min_probability=None))]
    pub fn transmit_spectrum(&self, frame_id: i32, scan_id: i32, spectrum: PyMzSpectrum, min_probability: Option<f64>) -> PyMzSpectrum {
        PyMzSpectrum { inner: self.inner.transmit_spectrum(frame_id, scan_id, spectrum.inner, min_probability) }
    }

    #[pyo3(signature = (frame, min_probability=None))]
    pub fn transmit_tims_frame(&self, frame: PyTimsFrame, min_probability: Option<f64>) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.transmit_tims_frame(&frame.inner, min_probability) }
    }

    #[pyo3(signature = (frames, scans, spectrum, min_proba=None))]
    pub fn transmit_ion(&self, frames: Vec<i32>, scans: Vec<i32>, spectrum: PyMzSpectrum, min_proba: Option<f64>) -> Vec<Vec<PyMzSpectrum>> {
        let transmission_profile = self.inner.transmit_ion(frames, scans, spectrum.inner, min_proba);
        transmission_profile.iter().map(|x| x.iter().map(|y| PyMzSpectrum { inner: y.clone() }).collect::<Vec<_>>()).collect::<Vec<_>>()
    }

    #[pyo3(signature = (frame_id, scan_id, mz, min_proba=None))]
    pub fn is_transmitted(&self, frame_id: i32, scan_id: i32, mz: f64, min_proba: Option<f64>) -> bool {
        self.inner.is_transmitted(frame_id, scan_id, mz, min_proba)
    }

    #[pyo3(signature = (frame_id, scan_id, mz, min_proba=None))]
    pub fn any_transmitted(&self, frame_id: i32, scan_id: i32, mz: Vec<f64>, min_proba: Option<f64>) -> bool {
        self.inner.any_transmitted(frame_id, scan_id, &mz, min_proba)
    }

    #[pyo3(signature = (frame_id, scan_id, mz, min_proba=None))]
    pub fn all_transmitted(&self, frame_id: i32, scan_id: i32, mz: Vec<f64>, min_proba: Option<f64>) -> bool {
        self.inner.all_transmitted(frame_id, scan_id, &mz, min_proba)
    }

    #[pyo3(signature = (frame_id, scan_id, mz, min_proba=None))]
    pub fn get_transmission_set(&self, frame_id: i32, scan_id: i32, mz: Vec<f64>, min_proba: Option<f64>) -> HashSet<usize> {
        self.inner.get_transmission_set(frame_id, scan_id, &mz, min_proba)
    }
    #[pyo3(signature = (frame_id, scan_id, mz_mono, mz, min_proba=None))]
    pub fn isotopes_transmitted(&self, frame_id: i32, scan_id: i32, mz_mono: f64, mz: Vec<f64>, min_proba: Option<f64>) -> (f64, Vec<(f64, f64)>) {
        self.inner.isotopes_transmitted(frame_id, scan_id, mz_mono, &mz, min_proba)
    }
}


#[pyclass(unsendable)]
pub struct PyTimsTransmissionDIA {
    pub inner: TimsTransmissionDIA,
}

#[pymethods]
impl PyTimsTransmissionDIA {
    #[new]
    #[pyo3(signature = (frame, frame_window_group, window_group, scan_start, scan_end, isolation_mz, isolation_width, k=None))]
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

    #[pyo3(signature = (frame_id, scan_id, spectrum, min_probability=None))]
    pub fn transmit_spectrum(&self, frame_id: i32, scan_id: i32, spectrum: PyMzSpectrum, min_probability: Option<f64>) -> PyMzSpectrum {
        PyMzSpectrum { inner: self.inner.transmit_spectrum(frame_id, scan_id, spectrum.inner, min_probability) }
    }

    #[pyo3(signature = (frame, min_probability=None))]
    pub fn transmit_tims_frame(&self, frame: PyTimsFrame, min_probability: Option<f64>) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.transmit_tims_frame(&frame.inner, min_probability) }
    }

    #[pyo3(signature = (frames, scans, spectrum, min_proba=None))]
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

    #[pyo3(signature = (frame_id, scan_id, mz, min_proba=None))]
    pub fn is_transmitted(&self, frame_id: i32, scan_id: i32, mz: f64, min_proba: Option<f64>) -> bool {
        self.inner.is_transmitted(frame_id, scan_id, mz, min_proba)
    }

    #[pyo3(signature = (frame_id, scan_id, mz, min_proba=None))]
    pub fn any_transmitted(&self, frame_id: i32, scan_id: i32, mz: Vec<f64>, min_proba: Option<f64>) -> bool {
        self.inner.any_transmitted(frame_id, scan_id, &mz, min_proba)
    }

    #[pyo3(signature = (frame_id, scan_id, mz, min_proba=None))]
    pub fn all_transmitted(&self, frame_id: i32, scan_id: i32, mz: Vec<f64>, min_proba: Option<f64>) -> bool {
        self.inner.all_transmitted(frame_id, scan_id, &mz, min_proba)
    }

    #[pyo3(signature = (frame_id, scan_id, mz, min_proba=None))]
    pub fn get_transmission_set(&self, frame_id: i32, scan_id: i32, mz: Vec<f64>, min_proba: Option<f64>) -> HashSet<usize> {
        self.inner.get_transmission_set(frame_id, scan_id, &mz, min_proba)
    }

    pub fn is_precursor(&self, frame_id: i32) -> bool {
        self.inner.is_precursor(frame_id)
    }

    #[pyo3(signature = (frame_id, scan_id, mz_mono, mz, min_proba=None))]
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
#[pyo3(signature = (midpoint, window_length, mz, k=None))]
pub fn apply_transmission(midpoint: f64, window_length: f64, mz: Vec<f64>, k: Option<f64>) -> Vec<f64> {
    mscore::timstof::quadrupole::apply_transmission(midpoint, window_length, k.unwrap_or(15.0), mz)
}

#[pymodule]
pub fn py_quadrupole(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPasefMeta>()?;
    m.add_class::<PyTimsTransmissionDDA>()?;
    m.add_class::<PyTimsTransmissionDIA>()?;
    m.add_class::<PyTimsTofCollisionEnergyDIA>()?;
    m.add_function(wrap_pyfunction!(apply_transmission, m)?)?;
    Ok(())
}
