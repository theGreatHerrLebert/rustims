use pyo3::prelude::*;

mod py_amino_acid;
use py_amino_acid::amino_acid;

#[pymodule]
fn rustms_connector(py: Python, m: &PyModule) -> PyResult<()> {
    // py_amino_acids submodule //
    let py_amino_acid_submodule = PyModule::new(py, "py_amino_acid")?;
    amino_acid(py, &py_amino_acid_submodule)?;
    m.add_submodule(py_amino_acid_submodule)?;

    Ok(())
}