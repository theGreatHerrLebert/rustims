use pyo3::prelude::*;
#[pyfunction]
pub fn emg_cdf_range(lower_limit: f64, upper_limit: f64, mu: f64, sigma: f64, lambda: f64) -> f64 {
    mscore::algorithm::utility::emg_cdf_range(lower_limit, upper_limit, mu, sigma, lambda)
}

#[pyfunction]
pub fn emg_function(x: f64, mu: f64, sigma: f64, lambda: f64) -> f64 {
    mscore::algorithm::utility::emg_function(x, mu, sigma, lambda)
}

#[pymodule]
pub fn utility(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(emg_function, m)?)?;
    m.add_function(wrap_pyfunction!(emg_cdf_range, m)?)?;
    Ok(())
}
