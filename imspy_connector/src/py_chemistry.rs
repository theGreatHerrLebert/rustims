use pyo3::prelude::*;

use mscore::algorithm::isotope_distributions::{generate_averagine_spectra, generate_averagine_spectrum};
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
    mscore::chemistry::aa_sequence::calculate_monoisotopic_mass(sequence)
}

#[pyfunction]
pub fn calculate_b_y_ion_series(sequence: &str, modifications: Vec<f64>, charge: Option<i32>) -> (Vec<(f64, String, String)>, Vec<(f64, String, String)>) {
    mscore::chemistry::aa_sequence::calculate_b_y_ion_series(sequence, modifications, charge)
}

#[pyfunction]
pub fn simulate_charge_state_for_sequence(sequence: &str, max_charge: Option<usize>, charge_probability: Option<f64>) -> Vec<f64> {
    mscore::algorithm::aa_sequence::simulate_charge_state_for_sequence(sequence, max_charge, charge_probability)
}

#[pyfunction]
pub fn simulate_charge_states_for_sequences(sequences: Vec<&str>, num_threads: usize, max_charge: Option<usize>, charge_probability: Option<f64>) -> Vec<Vec<f64>> {
    mscore::algorithm::aa_sequence::simulate_charge_states_for_sequences(sequences, num_threads, max_charge, charge_probability)
}

#[pyfunction]
pub fn find_unimod_annotations(sequence: &str) -> (String, Vec<f64>) {
    rustdf::sim::utility::find_unimod_patterns(sequence)
}

#[pyfunction]
pub fn find_unimod_annotations_par(sequences: Vec<&str>, num_threads: usize) -> Vec<(String, Vec<f64>)> {
    rustdf::sim::utility::find_unimod_patterns_par(sequences, num_threads)
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
    mscore::chemistry::aa_sequence::unimod_sequence_to_tokens(sequence)
}

#[pyfunction]
pub fn unimod_sequence_to_atomic_composition(sequence: &str) -> Vec<(&str, i32)> {
    mscore::chemistry::aa_sequence::unimod_sequence_to_atomic_composition(sequence).iter().map(|(k, v)| (*k, *v)).collect()
}

#[pyfunction]
pub fn mono_isotopic_b_y_fragment_composition(sequence: &str, is_y: Option<bool>) -> Vec<(&str, i32)> {
    mscore::chemistry::aa_sequence::mono_isotopic_b_y_fragment_composition(sequence, is_y)
}

#[pyfunction]
pub fn atomic_composition_to_monoisotopic_mass(composition: Vec<(&str, i32)>) -> f64 {
    mscore::chemistry::aa_sequence::atomic_composition_to_monoisotopic_mass(&composition)
}

#[pyfunction]
pub fn mono_isotopic_b_fragments_to_composition(sequences: Vec<&str>, num_threads: usize) -> Vec<Vec<(&str, i32)>> {
    mscore::chemistry::aa_sequence::mono_isotopic_b_fragments_to_composition(sequences, num_threads)
}

#[pyfunction]
pub fn mono_isotopic_y_fragments_to_composition(sequences: Vec<&str>, num_threads: usize) -> Vec<Vec<(&str, i32)>> {
    mscore::chemistry::aa_sequence::mono_isotopic_y_fragments_to_composition(sequences, num_threads)
}

#[pyfunction]
pub fn mono_isotopic_mass_from_unimod_sequence(sequence: &str) -> f64 {
    mscore::chemistry::aa_sequence::mono_isotopic_mass_from_unimod_sequence(sequence)
}