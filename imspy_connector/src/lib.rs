use pyo3::prelude::*;

pub mod py_constants;
use py_constants::constants;

pub mod py_chemistry;
use py_chemistry::chemistry;

pub mod py_dataset;
use py_dataset::dataset;

pub mod py_dda;
use py_dda::dda;

pub mod py_dia;
use py_dia::dia;

pub mod py_elements;
use py_elements::elements;

pub mod py_mz_spectrum;
use py_mz_spectrum::mz_spectrum;

pub mod py_quadrupole;
use py_quadrupole::quadrupole;

pub mod py_peptide;
use py_peptide::peptides;

pub mod py_simulation;
use py_simulation::simulation;

pub mod py_tims_frame;
use py_tims_frame::tims_frame;

pub mod py_tims_slice;
use py_tims_slice::tims_slice;

mod py_amino_acids;
use py_amino_acids::amino_acids;

mod py_unimod;
mod py_utility;
mod py_sumformula;
mod py_annotation;
use py_annotation::annotation;

use py_sumformula::sum_formula;
use py_unimod::unimod;

#[pymodule]
fn imspy_connector(py: Python, m: &PyModule) -> PyResult<()> {
    // py_constants submodule //
    let py_constants_submodule = PyModule::new(py, "py_constants")?;
    constants(py, &py_constants_submodule)?;
    m.add_submodule(py_constants_submodule)?;

    // py_chemistry submodule //
    let py_chemistry_submodule = PyModule::new(py, "py_chemistry")?;
    chemistry(py, &py_chemistry_submodule)?;
    m.add_submodule(py_chemistry_submodule)?;

    // py_dataset submodule //
    let py_dataset_submodule = PyModule::new(py, "py_dataset")?;
    dataset(py, &py_dataset_submodule)?;
    m.add_submodule(py_dataset_submodule)?;

    // py_dda submodule //
    let py_dda_submodule = PyModule::new(py, "py_dda")?;
    dda(py, &py_dda_submodule)?;
    m.add_submodule(py_dda_submodule)?;

    // py_dia submodule //
    let py_dia_submodule = PyModule::new(py, "py_dia")?;
    dia(py, &py_dia_submodule)?;
    m.add_submodule(py_dia_submodule)?;

    // py_elements submodule //
    let py_elements_submodule = PyModule::new(py, "py_elements")?;
    elements(py, &py_elements_submodule)?;
    m.add_submodule(py_elements_submodule)?;

    // py_mz_spectrum submodule //
    let py_mz_spectrum_submodule = PyModule::new(py, "py_spectrum")?;
    mz_spectrum(py, &py_mz_spectrum_submodule)?;
    m.add_submodule(py_mz_spectrum_submodule)?;

    // py_quadrupole submodule //
    let py_quadrupole_submodule = PyModule::new(py, "py_quadrupole")?;
    quadrupole(py, &py_quadrupole_submodule)?;
    m.add_submodule(py_quadrupole_submodule)?;

    // py_sequence submodule //
    let py_peptide_submodule = PyModule::new(py, "py_peptide")?;
    peptides(py, &py_peptide_submodule)?;
    m.add_submodule(py_peptide_submodule)?;

    // py_simulation submodule //
    let py_simulation_submodule = PyModule::new(py, "py_simulation")?;
    simulation(py, &py_simulation_submodule)?;
    m.add_submodule(py_simulation_submodule)?;

    // py_tims_frame submodule //
    let py_tims_frame_submodule = PyModule::new(py, "py_tims_frame")?;
    tims_frame(py, &py_tims_frame_submodule)?;
    m.add_submodule(py_tims_frame_submodule)?;

    // py_tims_slice submodule //
    let py_tims_slice_submodule = PyModule::new(py, "py_tims_slice")?;
    tims_slice(py, &py_tims_slice_submodule)?;
    m.add_submodule(py_tims_slice_submodule)?;

    // py_amino_acids submodule //
    let py_amino_acids_submodule = PyModule::new(py, "py_amino_acids")?;
    amino_acids(py, &py_amino_acids_submodule)?;
    m.add_submodule(py_amino_acids_submodule)?;

    // py_unimod submodule //
    let py_unimod_submodule = PyModule::new(py, "py_unimod")?;
    unimod(py, &py_unimod_submodule)?;
    m.add_submodule(py_unimod_submodule)?;

    // py_utility submodule //
    let py_utility_submodule = PyModule::new(py, "py_utility")?;
    py_utility::utility(py, &py_utility_submodule)?;
    m.add_submodule(py_utility_submodule)?;

    // py_sumformula submodule //
    let py_sumformula_submodule = PyModule::new(py, "py_sumformula")?;
    sum_formula(py, &py_sumformula_submodule)?;
    m.add_submodule(py_sumformula_submodule)?;

    // py_annotation submodule //
    let py_annotation_submodule = PyModule::new(py, "py_annotation")?;
    annotation(py, &py_annotation_submodule)?;
    m.add_submodule(py_annotation_submodule)?;

    Ok(())
}