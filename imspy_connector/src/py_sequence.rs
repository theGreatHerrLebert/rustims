use std::collections::HashMap;
use pyo3::prelude::*;

use mscore::chemistry::aa_sequence::{FragmentType, PeptideSequence, ProductIon};

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

    pub fn to_tokens(&self, group_modifications: bool) -> Vec<String> {
        self.inner.to_tokens(group_modifications)
    }

    pub fn to_sage_representation(&self) -> (String, Vec<f64>) {
        self.inner.to_sage_representation()
    }

    pub fn calculate_b_y_product_ion_series(&self, charge: i32) -> (Vec<PyProductIon>, Vec<PyProductIon>) {
        let (b, y) = self.inner.calculate_b_y_product_ion_series(charge);
        let b_ions: Vec<PyProductIon> = b.iter().map(|ion| PyProductIon { inner: ion.clone() }).collect();
        let y_ions: Vec<PyProductIon> = y.iter().map(|ion| PyProductIon { inner: ion.clone() }).collect();
        (b_ions, y_ions)
    }
}

#[pyclass]
pub struct PyProductIon {
    pub inner: ProductIon,
}

#[pymethods]
impl PyProductIon {
    #[new]
    pub fn new(kind: &str, sequence: String, charge: i32, intensity: f64) -> Self {

        let kind = match kind {
            "a" => FragmentType::A,
            "b" => FragmentType::B,
            "c" => FragmentType::C,
            "x" => FragmentType::X,
            "y" => FragmentType::Y,
            "z" => FragmentType::Z,
            _ => panic!("Invalid product ion kind"),
        };

        PyProductIon { inner: ProductIon::new(kind, sequence, charge, intensity) }
    }
    #[getter]
    pub fn kind(&self) -> String {
        match self.inner.kind {
            FragmentType::A => "a".to_string(),
            FragmentType::B => "b".to_string(),
            FragmentType::C => "c".to_string(),
            FragmentType::X => "x".to_string(),
            FragmentType::Y => "y".to_string(),
            FragmentType::Z => "z".to_string(),
        }
    }
    #[getter]
    pub fn sequence(&self) -> String {
        self.inner.sequence.sequence.clone()
    }
    #[getter]
    pub fn charge(&self) -> i32 {
        self.inner.charge
    }
    #[getter]
    pub fn mz(&self) -> f64 {
        self.inner.mz()
    }
    #[getter]
    pub fn intensity(&self) -> f64 {
        self.inner.intensity
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
    m.add_class::<PyProductIon>()?;
    Ok(())
}