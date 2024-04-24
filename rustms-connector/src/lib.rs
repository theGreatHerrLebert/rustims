use pyo3::prelude::*;
pub mod py_amino_acids;
pub mod py_mz_spectrum;
pub mod py_chemistry;
pub mod py_peptide;
pub mod py_elements;
pub mod py_constants;
pub mod py_sumformula;
mod py_unimod;


use py_amino_acids::amino_acids;
use py_chemistry::chemistry;
use py_peptide::peptides;
use py_mz_spectrum::mz_spectrum;
use py_elements::elements;


#[pymodule]
fn rustms_connector(py: Python, m: &PyModule) -> PyResult<()> {
    // py_amino_acid submodule //
    let py_amino_acids_submodule = PyModule::new(py, "py_amino_acids")?;
    amino_acids(py, &py_amino_acids_submodule)?;
    m.add_submodule(py_amino_acids_submodule)?;

    // py_chemistry submodule //
    let py_chemistry_submodule = PyModule::new(py, "py_chemistry")?;
    chemistry(py, &py_chemistry_submodule)?;
    m.add_submodule(py_chemistry_submodule)?;

    // py_peptide submodule //
    let py_peptide_submodule = PyModule::new(py, "py_peptide")?;
    peptides(py, &py_peptide_submodule)?;
    m.add_submodule(py_peptide_submodule)?;

    // py_mz_spectrum submodule //
    let py_mz_spectrum_submodule = PyModule::new(py, "py_spectrum")?;
    mz_spectrum(py, &py_mz_spectrum_submodule)?;
    m.add_submodule(py_mz_spectrum_submodule)?;

    // py_elements submodule //
    let py_elements_submodule = PyModule::new(py, "py_elements")?;
    elements(py, &py_elements_submodule)?;
    m.add_submodule(py_elements_submodule)?;

    // py_constants submodule //
    let py_constants_submodule = PyModule::new(py, "py_constants")?;
    py_constants::constants(py, &py_constants_submodule)?;
    m.add_submodule(py_constants_submodule)?;

    // py_sumformula submodule //
    let py_sumformula_submodule = PyModule::new(py, "py_sumformula")?;
    py_sumformula::sum_formula(py, &py_sumformula_submodule)?;
    m.add_submodule(py_sumformula_submodule)?;

    Ok(())
}