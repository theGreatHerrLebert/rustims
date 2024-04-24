use pyo3::prelude::*;
use crate::py_peptide::{PyPeptideSequence};
#[pyfunction]
pub fn calculate_monoisotopic_mass(peptide_sequence: PyPeptideSequence) -> f64 {
    rustms::algorithm::peptide::calculate_peptide_mono_isotopic_mass(&peptide_sequence.inner)
}

#[pyfunction]
pub fn find_unimod_annotations(sequence: &str) -> (String, Vec<f64>) {
    rustms::chemistry::utility::find_unimod_patterns(sequence)
}

#[pyfunction]
pub fn unimod_sequence_to_tokens(sequence: &str) -> Vec<String> {
    rustms::chemistry::utility::unimod_sequence_to_tokens(sequence, false)
}

#[pyfunction]
pub fn generate_isotope_distribution(atomic_composition: Vec<(String, f64)>, mass_tolerance: f64, abundance_threshold: f64, max_result: i32) -> Vec<(f64, f64)> {
    rustms::algorithm::isotope::generate_isotope_distribution(&atomic_composition.iter().map(|(k, v)| (k.to_string(), *v as i32)).collect(),
                                                              mass_tolerance, abundance_threshold, max_result)
}

#[pyfunction]
pub fn one_over_reduced_mobility_to_ccs(one_over_k0: f64, mz: f64, charge: u32, mass_gas: f64, temp: f64, t_diff: f64) -> f64 {
    rustms::chemistry::formula::one_over_reduced_mobility_to_ccs(one_over_k0, mz, charge, mass_gas, temp, t_diff)
}

#[pyfunction]
pub fn ccs_to_one_over_reduced_mobility(ccs: f64, mz: f64, charge: u32, mass_gas: f64, temp: f64, t_diff: f64) -> f64 {
    rustms::chemistry::formula::ccs_to_one_over_reduced_mobility(ccs, mz, charge, mass_gas, temp, t_diff)
}

#[pyfunction]
pub fn calculate_mz(mono_isotopic_mass: f64, charge: i32) -> f64 {
    rustms::chemistry::formula::calculate_mz(mono_isotopic_mass, charge)
}

#[pymodule]
pub fn chemistry(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_monoisotopic_mass, m)?)?;
    m.add_function(wrap_pyfunction!(find_unimod_annotations, m)?)?;
    m.add_function(wrap_pyfunction!(unimod_sequence_to_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(generate_isotope_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(one_over_reduced_mobility_to_ccs, m)?)?;
    m.add_function(wrap_pyfunction!(ccs_to_one_over_reduced_mobility, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_mz, m)?)?;
    Ok(())
}

