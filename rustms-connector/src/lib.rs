use pyo3::prelude::*;
pub mod py_amino_acids;
use py_amino_acids::amino_acids;

#[pymodule]
fn rustms_connector(py: Python, m: &PyModule) -> PyResult<()> {
    // py_amino_acid submodule //
    let py_amino_acids_submodule = PyModule::new(py, "py_amino_acids")?;
    amino_acids(py, &py_amino_acids_submodule)?;
    m.add_submodule(py_amino_acids_submodule)?;

    Ok(())
}