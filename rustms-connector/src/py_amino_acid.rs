use std::collections::HashMap;
use pyo3::prelude::*;
use rustms::proteomics::amino_acid;
#[pyfunction]
pub fn get_amino_acids() -> HashMap<String, String> {
    let map = amino_acid::amino_acids();
    map.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

#[pyfunction]
pub fn get_amino_acid_mono_isotopic_masses() -> HashMap<String, f64> {
    let map = rustms::chemistry::amino_acid::amino_acid_masses();
    map.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

#[pyfunction]
pub fn get_amino_acid_atomic_compositions() -> HashMap<String, HashMap<String, i32>> {
    let map = rustms::chemistry::amino_acid::amino_acid_composition();

    let mut result = HashMap::new();

    for (k, v) in map.iter() {
        let parsed_map = v.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        result.insert(k.to_string(), parsed_map);
    }

    result
}

#[pymodule]
pub fn amino_acid(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_amino_acids, m)?)?;
    m.add_function(wrap_pyfunction!(get_amino_acid_mono_isotopic_masses, m)?)?;
    m.add_function(wrap_pyfunction!(get_amino_acid_atomic_compositions, m)?)?;
    Ok(())
}