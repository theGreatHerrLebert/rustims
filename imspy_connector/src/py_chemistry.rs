use pyo3::prelude::*;

use mscore::{generate_averagine_spectrum};
use crate::py_mz_spectrum::PyMzSpectrum;

#[pyfunction]
pub fn generate_precursor_spectrum(mass: f64, charge: i32, min_intensity: i32, k: i32, resolution: i32, centroid: bool) -> PyMzSpectrum {
    PyMzSpectrum { inner: generate_averagine_spectrum(mass, charge, min_intensity, k, resolution, centroid, None) }
}