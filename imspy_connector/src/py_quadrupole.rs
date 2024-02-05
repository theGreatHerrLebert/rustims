use pyo3::prelude::*;

use mscore::{IonTransmission, TimsTransmissionDIA};
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

    pub fn is_transmitted(&self, frame_id: i32, scan_id: i32, mz: f64) -> bool {
        self.inner.is_transmitted(frame_id, scan_id, mz)
    }
}

#[pyfunction]
pub fn apply_transmission(midpoint: f64, window_length: f64, mz: Vec<f64>, k: Option<f64>) -> Vec<f64> {
    mscore::apply_transmission(midpoint, window_length, k, mz)
}