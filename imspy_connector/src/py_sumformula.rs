use std::collections::HashMap;
use pyo3::prelude::*;
use mscore::chemistry::sum_formula::SumFormula;
use crate::py_mz_spectrum::PyMzSpectrum;

#[pyclass]
pub struct PySumFormula {
    inner: SumFormula,
}

#[pymethods]
impl PySumFormula {
    #[new]
    pub fn new(formula: &str) -> Self {
        PySumFormula { inner: SumFormula::new(formula) }
    }

    #[getter]
    pub fn formula(&self) -> String {
        self.inner.formula.clone()
    }

    #[getter]
    pub fn elements(&self) -> HashMap<String, i32> {
        self.inner.elements.clone()
    }

    #[getter]
    pub fn monoisotopic_mass(&self) -> f64 {
        self.inner.monoisotopic_weight()
    }

    pub fn generate_isotope_distribution(&self, charge: i32) -> PyMzSpectrum {
        PyMzSpectrum::from_inner(self.inner.isotope_distribution(charge))
    }
}


#[pymodule]
pub fn py_sum_formula(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySumFormula>()?;
    Ok(())
}