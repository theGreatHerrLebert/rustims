use pyo3::prelude::*;

#[pyfunction]
pub fn emg_cdf(x: f64, mu: f64, sigma: f64, lambda: f64) -> f64 {
    mscore::algorithm::utility::emg_function(x, mu, sigma, lambda)
}

#[pyfunction]
pub fn accumulated_cdf_emg(lower_limit: f64, upper_limit: f64, mu: f64, sigma: f64, lambda: f64) -> f64 {
    mscore::algorithm::utility::emg_cdf_range(lower_limit, upper_limit, mu, sigma, lambda)
}

#[pyfunction]
pub fn calculate_bounds_emg(mu: f64, sigma: f64, lambda: f64, step_size: f64, target: f64, upper_start: f64, lower_start: f64) -> (f64, f64) {
    mscore::algorithm::utility::calculate_bounds_emg(mu, sigma, lambda, step_size, target, upper_start, lower_start)
}

#[pyfunction]
pub fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    mscore::algorithm::utility::custom_cdf_normal(x, mean, std_dev)
}

#[pyfunction]
pub fn accumulated_cdf_normal(sample_start: f64, sample_end: f64, mean: f64, std_dev: f64) -> f64 {
    mscore::algorithm::utility::accumulated_intensity_cdf_normal(sample_start, sample_end, mean, std_dev)
}

#[pyfunction]
pub fn calculate_bounds_normal(mean: f64, std: f64, z_score: f64) -> (f64, f64) {
    mscore::algorithm::utility::calculate_bounds_normal(mean, std, z_score)
}

#[pymodule]
pub fn utility(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(emg_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(accumulated_cdf_emg, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_bounds_emg, m)?)?;
    m.add_function(wrap_pyfunction!(normal_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(accumulated_cdf_normal, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_bounds_normal, m)?)?;
    Ok(())
}
