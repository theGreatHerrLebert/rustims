use pyo3::prelude::*;
use mscore::chemistry::constants::*;

#[pyfunction]
pub fn mass_proton() -> f64 {
    MASS_PROTON
}
#[pyfunction]
pub fn mass_neutron() -> f64 {
    MASS_NEUTRON
}
#[pyfunction]
pub fn mass_electron() -> f64 {
    MASS_ELECTRON
}
#[pyfunction]
pub fn mass_water() -> f64 {
    MASS_WATER
}
#[pyfunction]
pub fn standard_temperature() -> f64 {
    STANDARD_TEMPERATURE
}
#[pyfunction]
pub fn standard_pressure() -> f64 {
    STANDARD_PRESSURE
}
#[pyfunction]
pub fn elementary_charge() -> f64 {
    ELEMENTARY_CHARGE
}
#[pyfunction]
pub fn k_boltzmann() -> f64 {
    K_BOLTZMANN
}

#[pyfunction]
pub fn avogadro() -> f64 {
    AVOGADRO
}

#[pymodule]
fn py_constants(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mass_proton, m)?)?;
    m.add_function(wrap_pyfunction!(mass_neutron, m)?)?;
    m.add_function(wrap_pyfunction!(mass_electron, m)?)?;
    m.add_function(wrap_pyfunction!(mass_water, m)?)?;
    m.add_function(wrap_pyfunction!(standard_temperature, m)?)?;
    m.add_function(wrap_pyfunction!(standard_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(elementary_charge, m)?)?;
    m.add_function(wrap_pyfunction!(k_boltzmann, m)?)?;
    m.add_function(wrap_pyfunction!(avogadro, m)?)?;
    Ok(())
}


