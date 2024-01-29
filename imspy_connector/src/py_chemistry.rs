use pyo3::prelude::*;

use mscore::{generate_averagine_spectra, generate_averagine_spectrum};
use crate::py_mz_spectrum::PyMzSpectrum;

#[pyfunction]
pub fn generate_precursor_spectrum(mass: f64, charge: i32, min_intensity: i32, k: i32, resolution: i32, centroid: bool) -> PyMzSpectrum {
    PyMzSpectrum { inner: generate_averagine_spectrum(mass, charge, min_intensity, k, resolution, centroid, None) }
}

#[pyfunction]
pub fn generate_precursor_spectra(
    masses: Vec<f64>,
    charges: Vec<i32>,
    min_intensity: i32,
    k: i32,
    resolution: i32,
    centroid: bool,
    num_threads: usize
) -> Vec<PyMzSpectrum> {
    let result = generate_averagine_spectra(masses, charges, min_intensity, k, resolution, centroid, num_threads, None);
    result.into_iter().map(|spectrum| PyMzSpectrum { inner: spectrum }).collect()
}

#[pyfunction]
pub fn calculate_monoisotopic_mass(sequence: &str) -> f64 {
    mscore::calculate_monoisotopic_mass(sequence)
}

#[pyfunction]
pub fn calculate_b_y_ion_series(sequence: &str, modifications: Vec<f64>, charge: Option<i32>) -> (Vec<(f64, String, String)>, Vec<(f64, String, String)>) {
    mscore::calculate_b_y_ion_series(sequence, modifications, charge)
}