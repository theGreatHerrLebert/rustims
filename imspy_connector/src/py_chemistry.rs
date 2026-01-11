use rayon::prelude::*;
use std::collections::HashSet;
use pyo3::prelude::*;

use mscore::algorithm::isotope::{generate_averagine_spectra, generate_averagine_spectrum};
use mscore::data::spectrum::MzSpectrum;
use crate::py_mz_spectrum::PyMzSpectrum;
use crate::py_peptide::{PyPeptideSequence};

#[pyfunction]
pub fn generate_precursor_spectrum(mass: f64, charge: i32, min_intensity: i32, k: i32, resolution: i32, centroid: bool) -> PyMzSpectrum {
    PyMzSpectrum::from_inner(generate_averagine_spectrum(mass, charge, min_intensity, k, resolution, centroid, None))
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
    result.into_iter().map(|spectrum| PyMzSpectrum::from_inner(spectrum)).collect()
}

#[pyfunction]
pub fn calculate_monoisotopic_mass(peptide_sequence: PyPeptideSequence) -> f64 {
    mscore::algorithm::peptide::calculate_peptide_mono_isotopic_mass(&peptide_sequence.inner)
}

#[pyfunction]
#[pyo3(signature = (sequence, max_charge=None, charge_probability=None))]
pub fn simulate_charge_state_for_sequence(sequence: &str, max_charge: Option<usize>, charge_probability: Option<f64>) -> Vec<f64> {
    mscore::algorithm::peptide::simulate_charge_state_for_sequence(sequence, max_charge, charge_probability)
}

#[pyfunction]
#[pyo3(signature = (sequences, num_threads, max_charge=None, charge_probability=None))]
pub fn simulate_charge_states_for_sequences(sequences: Vec<&str>, num_threads: usize, max_charge: Option<usize>, charge_probability: Option<f64>) -> Vec<Vec<f64>> {
    mscore::algorithm::peptide::simulate_charge_states_for_sequences(sequences, num_threads, max_charge, charge_probability)
}

#[pyfunction]
pub fn find_unimod_annotations(sequence: &str) -> (String, Vec<f64>) {
    mscore::chemistry::utility::find_unimod_patterns(sequence)
}
#[pyfunction]
#[pyo3(signature = (sequence, charge, intensities, normalize, half_charge_one, peptide_id=None))]
pub fn sequence_to_all_ions_ims(sequence: &str, charge: i32, intensities: Vec<f64>, normalize: bool, half_charge_one: bool, peptide_id: Option<i32>) -> String {
    rustdf::sim::utility::sequence_to_all_ions(sequence, charge, &intensities, normalize, half_charge_one, peptide_id)
}

#[pyfunction]
pub fn reshape_prosit_array(flat_array: Vec<f64>) -> Vec<Vec<Vec<f64>>> {
    rustdf::sim::utility::reshape_prosit_array(flat_array)
}

#[pyfunction]
pub fn sequence_to_all_ions_par(sequences: Vec<&str>, charges: Vec<i32>, intensities: Vec<Vec<f64>>, normalize: bool, half_charge_one: bool, num_threads: usize, peptide_ids: Vec<Option<i32>>) -> Vec<String> {
    rustdf::sim::utility::sequence_to_all_ions_par(sequences, charges, intensities, normalize, half_charge_one, num_threads, peptide_ids)
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

#[pyfunction]
pub fn one_over_reduced_mobility_to_ccs(one_over_k0: f64, mz: f64, charge: u32, mass_gas: f64, temp: f64, t_diff: f64) -> f64 {
    mscore::chemistry::formulas::one_over_reduced_mobility_to_ccs(one_over_k0, mz, charge, mass_gas, temp, t_diff)
}

#[pyfunction]
pub fn one_over_reduced_mobility_to_ccs_par(one_over_k0: Vec<f64>, mz: Vec<f64>, charge: Vec<u32>, mass_gas: f64, temp: f64, t_diff: f64, num_threads: usize) -> Vec<f64> {
    let thread_pool = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    thread_pool.install(|| {
        one_over_k0.par_iter().zip(mz.par_iter()).zip(charge.par_iter()).map(|((k0, mz), charge)| {
            mscore::chemistry::formulas::one_over_reduced_mobility_to_ccs(*k0, *mz, *charge, mass_gas, temp, t_diff)
        }).collect()
    })
}

#[pyfunction]
pub fn ccs_to_one_over_reduced_mobility(ccs: f64, mz: f64, charge: u32, mass_gas: f64, temp: f64, t_diff: f64) -> f64 {
    mscore::chemistry::formulas::ccs_to_one_over_reduced_mobility(ccs, mz, charge, mass_gas, temp, t_diff)
}

#[pyfunction]
pub fn ccs_to_one_over_reduced_mobility_par(ccs: Vec<f64>, mz: Vec<f64>, charge: Vec<u32>, mass_gas: f64, temp: f64, t_diff: f64, num_threads: usize) -> Vec<f64> {
    let thread_pool = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    thread_pool.install(|| {
        ccs.par_iter().zip(mz.par_iter()).zip(charge.par_iter()).map(|((ccs, mz), charge)| {
            mscore::chemistry::formulas::ccs_to_one_over_reduced_mobility(*ccs, *mz, *charge, mass_gas, temp, t_diff)
        }).collect()
    })
}

#[pyfunction]
pub fn calculate_mz(mono_isotopic_mass: f64, charge: i32) -> f64 {
    mscore::chemistry::formulas::calculate_mz(mono_isotopic_mass, charge)
}

#[pyfunction]
#[pyo3(signature = (sequence, charge, peptide_id=None))]
pub fn simulate_precursor_spectrum(sequence: &str, charge: i32, peptide_id: Option<i32>) -> PyMzSpectrum {
    PyMzSpectrum::from_inner(mscore::algorithm::isotope::generate_precursor_spectrum(&sequence, charge, peptide_id))
}

#[pyfunction]
pub fn simulate_precursor_spectra(sequences: Vec<&str>, charges: Vec<i32>, num_threads: usize, peptide_ids: Vec<Option<i32>>) -> Vec<PyMzSpectrum> {
    let spectra = mscore::algorithm::isotope::generate_precursor_spectra(&sequences, &charges, num_threads, peptide_ids);
    spectra.into_iter().map(|spectrum| PyMzSpectrum::from_inner(spectrum)).collect()
}

#[pyfunction]
pub fn calculate_transmission_dependent_fragment_ion_isotope_distribution(target_spec: PyMzSpectrum, complement_spec: PyMzSpectrum, transmitted_isotopes: PyMzSpectrum, max_isotope: usize) -> PyMzSpectrum {

    let transmitted_map: HashSet<usize> = transmitted_isotopes.inner.mz.iter().enumerate().map(|(i, _)| i).collect();

    let target: Vec<(f64, f64)> = target_spec.inner.mz.iter().zip(target_spec.inner.intensity.iter()).map(|(mz, intensity)| (*mz, *intensity)).collect();
    let complement: Vec<(f64, f64)> = complement_spec.inner.mz.iter().zip(complement_spec.inner.intensity.iter()).map(|(mz, intensity)| (*mz, *intensity)).collect();

    let result = mscore::algorithm::isotope::calculate_transmission_dependent_fragment_ion_isotope_distribution(
        &target, &complement, &transmitted_map, max_isotope
    );

    let mz_vec: Vec<f64> = result.iter().map(|(mz, _)| *mz).collect();
    let intensity_vec: Vec<f64> = result.iter().map(|(_, intensity)| *intensity).collect();
    PyMzSpectrum::from_inner(MzSpectrum::new(mz_vec, intensity_vec))
}

#[pymodule]
pub fn py_chemistry(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(one_over_reduced_mobility_to_ccs, m)?)?;
    m.add_function(wrap_pyfunction!(one_over_reduced_mobility_to_ccs_par, m)?)?;
    m.add_function(wrap_pyfunction!(ccs_to_one_over_reduced_mobility, m)?)?;
    m.add_function(wrap_pyfunction!(ccs_to_one_over_reduced_mobility_par, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_mz, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_precursor_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_precursor_spectra, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_transmission_dependent_fragment_ion_isotope_distribution, m)?)?;
    Ok(())
}

