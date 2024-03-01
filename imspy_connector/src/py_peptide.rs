use std::collections::{BTreeMap, HashMap};
use pyo3::prelude::*;

use mscore::data::peptide::{FragmentType, PeptideSequence, PeptideProductIon};

#[pyclass]
#[derive(Clone)]
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

    pub fn calculate_product_ion_series(&self, charge: i32, fragment_type: String) -> (Vec<PyPeptideProductIon>, Vec<PyPeptideProductIon>) {

        let f_type = match fragment_type.as_str() {
            "a" => FragmentType::A,
            "b" => FragmentType::B,
            "c" => FragmentType::C,
            "x" => FragmentType::X,
            "y" => FragmentType::Y,
            "z" => FragmentType::Z,
            _ => panic!("Invalid fragment type"),
        };

        let (n, c) = self.inner.calculate_product_ion_series(charge, f_type);
        let n_ions: Vec<PyPeptideProductIon> = n.iter().map(|ion| PyPeptideProductIon { inner: ion.clone() }).collect();
        let c_ions: Vec<PyPeptideProductIon> = c.iter().map(|ion| PyPeptideProductIon { inner: ion.clone() }).collect();
        (n_ions, c_ions)
    }

    pub fn associate_with_predicted_intensities(
        &self,
        flat_intensities: Vec<f64>,
        charge: i32,
        fragment_type: &str,
        normalize: bool,
        half_charge_one: bool,
    ) -> BTreeMap<i32, (Vec<PyPeptideProductIon>, Vec<PyPeptideProductIon>)> {

        let fragment_type = match fragment_type {
            "a" => FragmentType::A,
            "b" => FragmentType::B,
            "c" => FragmentType::C,
            "x" => FragmentType::X,
            "y" => FragmentType::Y,
            "z" => FragmentType::Z,
            _ => panic!("Invalid fragment type"),
        };

        let result = self.inner.associate_with_predicted_intensities(charge, fragment_type, flat_intensities, normalize, half_charge_one);
        let mut map: BTreeMap<i32, (Vec<PyPeptideProductIon>, Vec<PyPeptideProductIon>)> = BTreeMap::new();
        for (k, (n, c)) in result {
            let n_ions: Vec<PyPeptideProductIon> = n.iter().map(|ion| PyPeptideProductIon { inner: ion.clone() }).collect();
            let c_ions: Vec<PyPeptideProductIon> = c.iter().map(|ion| PyPeptideProductIon { inner: ion.clone() }).collect();
            map.insert(k, (n_ions, c_ions));
        }
        map
    }
}

#[pyclass]
pub struct PyPeptideProductIon {
    pub inner: PeptideProductIon,
}

#[pymethods]
impl PyPeptideProductIon {
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

        PyPeptideProductIon { inner: PeptideProductIon::new(kind, sequence, charge, intensity) }
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
        self.inner.ion.sequence.sequence.clone()
    }
    #[getter]
    pub fn charge(&self) -> i32 {
        self.inner.ion.charge
    }
    #[getter]
    pub fn mz(&self) -> f64 {
        self.inner.mz()
    }
    #[getter]
    pub fn intensity(&self) -> f64 {
        self.inner.ion.intensity
    }
    #[getter]
    pub fn mono_isotopic_mass(&self) -> f64 {
        self.inner.mono_isotopic_mass()
    }

    pub fn atomic_composition(&self) -> HashMap<&str, i32> {
        self.inner.atomic_composition()
    }

    pub fn isotope_distribution(&self, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> Vec<(f64, f64)> {
        self.inner.isotope_distribution(mass_tolerance, abundance_threshold, max_result, intensity_min)
    }
}

#[pymodule]
pub fn peptides(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPeptideSequence>()?;
    m.add_class::<PyPeptideProductIon>()?;
    Ok(())
}