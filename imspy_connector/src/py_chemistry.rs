use pyo3::prelude::*;

use mscore::algorithm::isotope::{generate_averagine_spectra, generate_averagine_spectrum};
use crate::py_mz_spectrum::PyMzSpectrum;
use crate::py_peptide::PyPeptideSequence;

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
pub fn calculate_monoisotopic_mass(peptide_sequence: PyPeptideSequence) -> f64 {
    mscore::algorithm::peptide::calculate_peptide_mono_isotopic_mass(&peptide_sequence.inner)
}

#[pyfunction]
pub fn simulate_charge_state_for_sequence(sequence: &str, max_charge: Option<usize>, charge_probability: Option<f64>) -> Vec<f64> {
    mscore::algorithm::peptide::simulate_charge_state_for_sequence(sequence, max_charge, charge_probability)
}

#[pyfunction]
pub fn simulate_charge_states_for_sequences(sequences: Vec<&str>, num_threads: usize, max_charge: Option<usize>, charge_probability: Option<f64>) -> Vec<Vec<f64>> {
    mscore::algorithm::peptide::simulate_charge_states_for_sequences(sequences, num_threads, max_charge, charge_probability)
}

#[pyfunction]
pub fn find_unimod_annotations(sequence: &str) -> (String, Vec<f64>) {
    mscore::chemistry::utility::find_unimod_patterns(sequence)
}
#[pyfunction]
pub fn sequence_to_all_ions_ims(sequence: &str, charge: i32, intensities: Vec<f64>, normalize: bool, half_charge_one: bool) -> String {
    rustdf::sim::utility::sequence_to_all_ions(sequence, charge, &intensities, normalize, half_charge_one)
}

#[pyfunction]
pub fn reshape_prosit_array(flat_array: Vec<f64>) -> Vec<Vec<Vec<f64>>> {
    rustdf::sim::utility::reshape_prosit_array(flat_array)
}

#[pyfunction]
pub fn sequence_to_all_ions_par(sequences: Vec<&str>, charges: Vec<i32>, intensities: Vec<Vec<f64>>, normalize: bool, half_charge_one: bool, num_threads: usize, ) -> Vec<String> {
    rustdf::sim::utility::sequence_to_all_ions_par(sequences, charges, intensities, normalize, half_charge_one, num_threads)
}

#[pyfunction]
pub fn unimod_sequence_to_tokens(sequence: &str) -> Vec<String> {
    mscore::chemistry::utility::unimod_sequence_to_tokens(sequence, false)
}

#[pyfunction]
pub fn generate_isotope_distribution(atomic_composition: Vec<(String, f64)>, mass_tolerance: f64, abundance_threshold: f64, max_result: i32) -> Vec<(f64, f64)> {
    mscore::algorithm::isotope::generate_isotope_distribution(&atomic_composition.iter().map(|(k, v)| (k.to_string(), *v as i32)).collect(),
        mass_tolerance, abundance_threshold, max_result)
}

#[pymodule]
pub fn chemistry(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_precursor_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(generate_precursor_spectra, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_monoisotopic_mass, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_charge_state_for_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_charge_states_for_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(find_unimod_annotations, m)?)?;
    m.add_function(wrap_pyfunction!(sequence_to_all_ions_ims, m)?)?;
    m.add_function(wrap_pyfunction!(reshape_prosit_array, m)?)?;
    m.add_function(wrap_pyfunction!(sequence_to_all_ions_par, m)?)?;
    m.add_function(wrap_pyfunction!(unimod_sequence_to_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(generate_isotope_distribution, m)?)?;
    Ok(())
}

