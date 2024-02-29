
use std::collections::HashMap;
use pyo3::prelude::*;
use mscore::chemistry::elements::{isotopic_abundance};
#[pyfunction]
pub fn t() -> HashMap<String, Vec<f64>> {
    let map = isotopic_abundance();
    map.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

#[pymodule]
pub fn unimod(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(t, m)?)?;
    Ok(())
}

