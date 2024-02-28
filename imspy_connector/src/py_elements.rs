use std::collections::HashMap;
use pyo3::prelude::*;
use mscore::chemistry::elements::{atoms_isotopic_weights, atomic_weights_mono_isotopic, isotopic_abundance};
#[pyfunction]
pub fn get_elemental_isotope_abundance_map() -> HashMap<String, Vec<f64>> {
    let map = isotopic_abundance();
    map.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

#[pyfunction]
pub fn get_elemental_isotope_weight_map() -> HashMap<String, Vec<f64>> {
    let map = atoms_isotopic_weights();
    map.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

#[pyfunction]
pub fn get_elemental_mono_isotopic_weight_map() -> HashMap<String, f64> {
    let map = atomic_weights_mono_isotopic();
    map.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

#[pymodule]
pub fn py_elements(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_elemental_isotope_abundance_map, m)?)?;
    m.add_function(wrap_pyfunction!(get_elemental_isotope_weight_map, m)?)?;
    m.add_function(wrap_pyfunction!(get_elemental_mono_isotopic_weight_map, m)?)?;
    Ok(())
}

