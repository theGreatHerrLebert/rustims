use std::collections::{HashMap};
use pyo3::prelude::*;

use mscore::data::peptide::{FragmentType, PeptideSequence, PeptideProductIon,
                            PeptideProductIonSeries, PeptideProductIonSeriesCollection, PeptideIon};
use crate::py_annotation::PyMzSpectrumAnnotated;

use crate::py_mz_spectrum::PyMzSpectrum;

#[pyclass]
pub struct PyPeptideIon {
    pub inner: PeptideIon,
}

#[pymethods]
impl PyPeptideIon {
    #[new]
    #[pyo3(signature = (sequence, charge, intensity, peptide_id=None))]
    pub fn new(sequence: String, charge: i32, intensity: f64, peptide_id: Option<i32>) -> Self {
        PyPeptideIon { inner: PeptideIon::new(sequence, charge, intensity, peptide_id) }
    }

    #[getter]
    pub fn sequence(&self) -> PyPeptideSequence {
        PyPeptideSequence { inner: self.inner.sequence.clone() }
    }

    #[getter]
    pub fn charge(&self) -> i32 {
        self.inner.charge
    }

    #[getter]
    pub fn intensity(&self) -> f64 {
        self.inner.intensity
    }

    #[getter]
    pub fn mz(&self) -> f64 {
        self.inner.mz()
    }

    #[getter]
    pub fn peptide_id(&self) -> Option<i32> {
        self.inner.sequence.peptide_id
    }

    pub fn calculate_isotopic_spectrum(&self, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> PyMzSpectrum {
        PyMzSpectrum { inner: self.inner.calculate_isotopic_spectrum(mass_tolerance, abundance_threshold, max_result, intensity_min) }
    }

    pub fn calculate_isotopic_spectrum_annotated(&self, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> PyMzSpectrumAnnotated {
        let annotated_spectrum = self.inner.calculate_isotopic_spectrum_annotated(mass_tolerance, abundance_threshold, max_result, intensity_min);
        PyMzSpectrumAnnotated { inner: annotated_spectrum }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyPeptideProductIonSeries {
    pub inner: PeptideProductIonSeries,
}

#[pymethods]
impl PyPeptideProductIonSeries {
    #[new]
    pub fn new(charge: i32, n_ions: Vec<PyPeptideProductIon>, c_ions: Vec<PyPeptideProductIon>) -> Self {
        let n_ions: Vec<PeptideProductIon> = n_ions.iter().map(|ion| ion.inner.clone()).collect();
        let c_ions: Vec<PeptideProductIon> = c_ions.iter().map(|ion| ion.inner.clone()).collect();
        PyPeptideProductIonSeries { inner: PeptideProductIonSeries::new(charge, n_ions, c_ions) }
    }

    #[getter]
    pub fn charge(&self) -> i32 {
        self.inner.charge
    }

    #[getter]
    pub fn n_ions(&self) -> Vec<PyPeptideProductIon> {
        self.inner.n_ions.iter().map(|ion| PyPeptideProductIon { inner: ion.clone() }).collect()
    }

    #[getter]
    pub fn c_ions(&self) -> Vec<PyPeptideProductIon> {
        self.inner.c_ions.iter().map(|ion| PyPeptideProductIon { inner: ion.clone() }).collect()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.inner).unwrap()
    }
}

#[pyclass]
pub struct PyPeptideProductIonSeriesCollection {
    pub inner: PeptideProductIonSeriesCollection,
}

#[pymethods]
impl PyPeptideProductIonSeriesCollection {
    #[new]
    pub fn new(peptide_product_ion_series: Vec<PyPeptideProductIonSeries>) -> Self {
        let inner: Vec<PeptideProductIonSeries> = peptide_product_ion_series.iter().map(|series| series.inner.clone()).collect();
        PyPeptideProductIonSeriesCollection { inner: PeptideProductIonSeriesCollection::new(inner) }
    }

    #[getter]
    pub fn series(&self) -> Vec<PyPeptideProductIonSeries> {
        self.inner.peptide_ions.iter().map(|series| PyPeptideProductIonSeries { inner: series.clone() }).collect()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.inner).unwrap()
    }

    pub fn find_ion_series(&self, charge: i32) -> Option<PyPeptideProductIonSeries> {
        let maybe_ion_series = self.inner.find_ion_series(charge);
        match maybe_ion_series {
            Some(ion_series) => Some(PyPeptideProductIonSeries { inner: ion_series.clone() }),
            None => None,
        }
    }

    pub fn generate_isotopic_spectrum(&self, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> PyMzSpectrum {
        let spectrum = self.inner.generate_isotopic_spectrum(mass_tolerance, abundance_threshold, max_result, intensity_min);
        PyMzSpectrum { inner: spectrum }
    }

    pub fn generate_isotopic_spectrum_annotated(&self, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> PyMzSpectrumAnnotated {
        let spectrum = self.inner.generate_isotopic_spectrum_annotated(mass_tolerance, abundance_threshold, max_result, intensity_min);
        PyMzSpectrumAnnotated { inner: spectrum }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyPeptideSequence {
    pub inner: PeptideSequence,
}

#[pymethods]
impl PyPeptideSequence {
    #[new]
    #[pyo3(signature = (sequence, peptide_id=None))]
    pub fn new(sequence: String, peptide_id: Option<i32>) -> Self {
        PyPeptideSequence { inner: PeptideSequence::new(sequence, peptide_id) }
    }

    #[getter]
    pub fn sequence(&self) -> String {
        self.inner.sequence.clone()
    }

    #[getter]
    pub fn peptide_id(&self) -> Option<i32> {
        self.inner.peptide_id
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

    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.inner).unwrap()
    }

    pub fn to_sage_representation(&self) -> (String, Vec<f64>) {
        self.inner.to_sage_representation()
    }

    pub fn amino_acid_count(&self) -> usize {
        self.inner.amino_acid_count()
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

        let series = self.inner.calculate_product_ion_series(charge, f_type);
        let n_ions: Vec<PyPeptideProductIon> = series.n_ions.iter().map(|ion| PyPeptideProductIon { inner: ion.clone() }).collect();
        let c_ions: Vec<PyPeptideProductIon> = series.c_ions.iter().map(|ion| PyPeptideProductIon { inner: ion.clone() }).collect();
        (n_ions, c_ions)
    }

    pub fn calculate_mono_isotopic_product_ion_spectrum(&self, charge: i32, fragment_type: String) -> PyMzSpectrum {
        let f_type = match fragment_type.as_str() {
            "a" => FragmentType::A,
            "b" => FragmentType::B,
            "c" => FragmentType::C,
            "x" => FragmentType::X,
            "y" => FragmentType::Y,
            "z" => FragmentType::Z,
            _ => panic!("Invalid fragment type"),
        };

        let spectrum = self.inner.calculate_mono_isotopic_product_ion_spectrum(charge, f_type);
        PyMzSpectrum { inner: spectrum }
    }

    pub fn calculate_mono_isotopic_product_ion_spectrum_annotated(&self, charge: i32, fragment_type: String) -> PyMzSpectrumAnnotated {
        let f_type = match fragment_type.as_str() {
            "a" => FragmentType::A,
            "b" => FragmentType::B,
            "c" => FragmentType::C,
            "x" => FragmentType::X,
            "y" => FragmentType::Y,
            "z" => FragmentType::Z,
            _ => panic!("Invalid fragment type"),
        };

        let spectrum = self.inner.calculate_mono_isotopic_product_ion_spectrum_annotated(charge, f_type);
        PyMzSpectrumAnnotated { inner: spectrum }
    }

    pub fn calculate_isotopic_product_ion_spectrum_annotated(&self, charge: i32, fragment_type: String, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> PyMzSpectrumAnnotated {
        let f_type = match fragment_type.as_str() {
            "a" => FragmentType::A,
            "b" => FragmentType::B,
            "c" => FragmentType::C,
            "x" => FragmentType::X,
            "y" => FragmentType::Y,
            "z" => FragmentType::Z,
            _ => panic!("Invalid fragment type"),
        };

        let spectrum = self.inner.calculate_isotopic_product_ion_spectrum_annotated(charge, f_type, mass_tolerance, abundance_threshold, max_result, intensity_min);
        PyMzSpectrumAnnotated { inner: spectrum }
    }

    pub fn associate_with_predicted_intensities(
        &self,
        flat_intensities: Vec<f64>,
        charge: i32,
        fragment_type: &str,
        normalize: bool,
        half_charge_one: bool,
    ) -> PyPeptideProductIonSeriesCollection {

        let fragment_type = match fragment_type {
            "a" => FragmentType::A,
            "b" => FragmentType::B,
            "c" => FragmentType::C,
            "x" => FragmentType::X,
            "y" => FragmentType::Y,
            "z" => FragmentType::Z,
            _ => panic!("Invalid fragment type"),
        };

        let result = self.inner.associate_with_predicted_intensities(
            charge,
            fragment_type,
            flat_intensities,
            normalize,
            half_charge_one
        );

        PyPeptideProductIonSeriesCollection { inner: result }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyPeptideProductIon {
    pub inner: PeptideProductIon,
}

#[pymethods]
impl PyPeptideProductIon {
    #[new]
    #[pyo3(signature = (kind, sequence, charge, intensity, peptide_id=None))]
    pub fn new(kind: &str, sequence: String, charge: i32, intensity: f64, peptide_id: Option<i32>) -> Self {

        let kind = match kind {
            "a" => FragmentType::A,
            "b" => FragmentType::B,
            "c" => FragmentType::C,
            "x" => FragmentType::X,
            "y" => FragmentType::Y,
            "z" => FragmentType::Z,
            _ => panic!("Invalid product ion kind"),
        };

        PyPeptideProductIon { inner: PeptideProductIon::new(kind, sequence, charge, intensity, peptide_id) }
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

    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.inner).unwrap()
    }

    pub fn isotope_distribution(&self, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> Vec<(f64, f64)> {
        self.inner.isotope_distribution(mass_tolerance, abundance_threshold, max_result, intensity_min)
    }
}

#[pymodule]
pub fn py_peptide(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPeptideSequence>()?;
    m.add_class::<PyPeptideIon>()?;
    m.add_class::<PyPeptideProductIon>()?;
    m.add_class::<PyPeptideProductIonSeries>()?;
    m.add_class::<PyPeptideProductIonSeriesCollection>()?;
    Ok(())
}