use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use mscore::{MzSpectrum, generate_spectrum};

#[pyfunction]
pub fn generate_precursor_spectrum(mass: f64, charge: i32, min_intensity: i32, k: i32, resolution: i32, centroid: bool) -> PyResult<MzSpectrum> {
    Ok(generate_spectrum(mass, charge, min_intensity, k, resolution, centroid, None))
}