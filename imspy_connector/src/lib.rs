use pyo3::prelude::*;
use pyo3::wrap_pymodule;

pub mod py_amino_acids;
pub mod py_annotation;
pub mod py_constants;
pub mod py_chemistry;
pub mod py_dataset;
pub mod py_dda;
pub mod py_dia;
pub mod py_elements;
pub mod py_mz_spectrum;
pub mod py_quadrupole;
pub mod py_peptide;
pub mod py_simulation;
pub mod py_tims_frame;
pub mod py_tims_slice;
pub mod py_unimod;
pub mod py_utility;
pub mod py_sumformula;

#[pymodule]
fn imspy_connector(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(py_annotation::py_annotation))?;
    m.add_wrapped(wrap_pymodule!(py_amino_acids::py_amino_acids))?;
    m.add_wrapped(wrap_pymodule!(py_chemistry::py_chemistry))?;
    m.add_wrapped(wrap_pymodule!(py_constants::py_constants))?;
    m.add_wrapped(wrap_pymodule!(py_dataset::py_dataset))?;
    m.add_wrapped(wrap_pymodule!(py_dda::py_dda))?;
    m.add_wrapped(wrap_pymodule!(py_dia::py_dia))?;
    m.add_wrapped(wrap_pymodule!(py_elements::py_elements))?;
    m.add_wrapped(wrap_pymodule!(py_mz_spectrum::py_mz_spectrum))?;
    m.add_wrapped(wrap_pymodule!(py_quadrupole::py_quadrupole))?;
    m.add_wrapped(wrap_pymodule!(py_peptide::py_peptide))?;
    m.add_wrapped(wrap_pymodule!(py_simulation::py_simulation))?;
    m.add_wrapped(wrap_pymodule!(py_tims_frame::py_tims_frame))?;
    m.add_wrapped(wrap_pymodule!(py_tims_slice::py_tims_slice))?;
    m.add_wrapped(wrap_pymodule!(py_unimod::py_unimod))?;
    m.add_wrapped(wrap_pymodule!(py_utility::py_utility))?;
    m.add_wrapped(wrap_pymodule!(py_sumformula::py_sum_formula))?;

    Ok(())
}