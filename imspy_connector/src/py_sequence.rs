use std::collections::HashMap;
use pyo3::prelude::*;

use mscore::chemistry::aa_sequence::AminoAcidSequence;
use crate::py_mz_spectrum::PyMzSpectrum;

#[pyclass]
pub struct  PyAminoAcidSequence {
    pub inner: AminoAcidSequence,
}

#[pymethods]
impl PyAminoAcidSequence {
    #[new]
    pub fn new(sequence: String) -> Self {
        PyAminoAcidSequence { inner: AminoAcidSequence::new(sequence) }
    }

    #[getter]
    pub fn sequence(&self) -> String {
        self.inner.sequence.clone()
    }

    #[getter]
    pub fn monoisotopic_mass(&self) -> f64 {
        self.inner.calculate_monoisotopic_mass()
    }

    pub fn monoisotopic_mass_from_atomic_composition(&self) -> f64 {
        self.inner.calculate_monoisotopic_mass_from_atomic_composition()
    }

    pub fn get_mz(&self, charge: i32) -> f64 {
        self.inner.calculate_mz(charge)
    }

    pub fn get_atomic_composition(&self) -> HashMap<&str, i32> {
        self.inner.calculate_atomic_composition()
    }

    pub fn precursor_spectrum_averagine(&self, charge: i32, min_intensity: i32, k: i32, resolution: i32, centroid: bool) -> PyMzSpectrum {
        PyMzSpectrum { inner: self.inner.precursor_spectrum_averagine(charge, min_intensity, k, resolution, centroid) }
    }

    pub fn precursor_spectrum_from_atomic_composition(&self, charge: i32, mass_tolerance: f64, abundance_threshold: f64, max_result: i32) -> PyMzSpectrum {
        PyMzSpectrum { inner: self.inner.precursor_spectrum_from_atomic_composition(charge, mass_tolerance, abundance_threshold, max_result) }
    }
}

#[pymodule]
pub fn py_sequence(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyAminoAcidSequence>()?;
    Ok(())
}