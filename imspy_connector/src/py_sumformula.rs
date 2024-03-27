use std::collections::HashMap;
use pyo3::prelude::*;
use mscore::chemistry::sum_formula::SumFormula;

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
}


#[pymodule]
pub fn sum_formula(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySumFormula>()?;
    Ok(())
}