
use std::collections::HashMap;
use pyo3::prelude::*;
use rustms::chemistry::unimod::{modification_atomic_composition, unimod_modifications_mass};
#[pyfunction]
pub fn get_unimod_masses() -> HashMap<String, f64> {
    let map = unimod_modifications_mass();
    map.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

#[pyfunction]
pub fn get_unimod_atomic_compositions() -> HashMap<String, HashMap<String, i32>> {
    let map = modification_atomic_composition();
    let mut result = HashMap::new();
    for (k, v) in map.iter() {
        let parsed_map = v.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        result.insert(k.to_string(), parsed_map);
    }

    result
}

#[pymodule]
pub fn unimod(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_unimod_masses, m)?)?;
    m.add_function(wrap_pyfunction!(get_unimod_atomic_compositions, m)?)?;
    Ok(())
}

