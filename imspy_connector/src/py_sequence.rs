use std::collections::HashMap;
use pyo3::prelude::*;

use mscore::chemistry::aa_sequence::PeptideSequence;

#[pyclass]
pub struct PyPeptideSequence {
    pub inner: PeptideSequence,
}

#[pymethods]
impl PyPeptideSequence {
    #[new]
    pub fn new(sequence: String) -> Self {
        PyPeptideSequence { inner: PeptideSequence::new(sequence) }
    }

    #[getter]
    pub fn sequence(&self) -> String {
        self.inner.sequence.clone()
    }

    #[getter]
    pub fn mono_isotopic_mass(&self) -> f64 {
        self.inner.mono_isotopic_mass()
    }

    pub fn atomic_composition(&self) -> HashMap<&str, i32> {
        self.inner.atomic_composition()
    }
}

#[pymodule]
pub fn py_sequence(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPeptideSequence>()?;
    Ok(())
}