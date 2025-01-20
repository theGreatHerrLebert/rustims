use std::collections::HashMap;
use pyo3::prelude::*;

#[pyfunction]
pub fn emg_cdf(x: f64, mu: f64, sigma: f64, lambda: f64) -> f64 {
    mscore::algorithm::utility::emg_function(x, mu, sigma, lambda)
}

#[pyfunction]
#[pyo3(signature = (lower_limit, upper_limit, mu, sigma, lambda, n_steps=None))]
pub fn accumulated_cdf_emg(lower_limit: f64, upper_limit: f64, mu: f64, sigma: f64, lambda: f64, n_steps: Option<usize>) -> f64 {
    mscore::algorithm::utility::emg_cdf_range(lower_limit, upper_limit, mu, sigma, lambda, n_steps)
}

#[pyfunction]
#[pyo3(signature = (mu, sigma, lambda, step_size, target, lower_start, upper_start, n_steps=None))]
pub fn calculate_bounds_emg(mu: f64, sigma: f64, lambda: f64, step_size: f64, target: f64, lower_start: f64, upper_start: f64, n_steps: Option<usize>) -> (f64, f64) {
    mscore::algorithm::utility::calculate_bounds_emg(mu, sigma, lambda, step_size, target, lower_start, upper_start, n_steps)
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

#[pyfunction]
#[pyo3(signature = (retention_times, rt, sigma, lambda_, target_p, step_size, n_steps=None))]
pub fn calculate_frame_occurrence_emg(retention_times: Vec<f64>, rt: f64, sigma: f64, lambda_: f64, target_p: f64, step_size: f64, n_steps: Option<usize>) -> Vec<i32> {
    mscore::algorithm::utility::calculate_frame_occurrence_emg(&retention_times, rt, sigma, lambda_, target_p, step_size, n_steps)
}

#[pyfunction]
#[pyo3(signature = (frame_ids, retention_times, frame_occurrences, rt, sigma, lambda_, rt_cycle_length, n_steps=None))]
pub fn calculate_frame_abundance_emg(frame_ids: Vec<i32>, retention_times: Vec<f64>, frame_occurrences: Vec<i32>, rt: f64, sigma: f64, lambda_: f64, rt_cycle_length: f64, n_steps: Option<usize>) -> Vec<f64> {
    let time_map: HashMap<i32, f64> = frame_ids.iter().zip(retention_times.iter()).map(|(id, rt)| (*id, *rt)).collect();
    mscore::algorithm::utility::calculate_frame_abundance_emg(&time_map, &frame_occurrences, rt, sigma, lambda_, rt_cycle_length, n_steps)
}

#[pyfunction]
#[pyo3(signature = (retention_times, rts, sigmas, lambdas, target_p, step_size, num_threads, n_steps=None))]
pub fn calculate_frame_occurrences_emg_par(retention_times: Vec<f64>, rts: Vec<f64>, sigmas: Vec<f64>, lambdas: Vec<f64>, target_p: f64, step_size: f64, num_threads: usize, n_steps: Option<usize>) -> Vec<Vec<i32>> {
    mscore::algorithm::utility::calculate_frame_occurrences_emg_par(&retention_times, rts, sigmas, lambdas, target_p, step_size, num_threads, n_steps)
}

#[pyfunction]
#[pyo3(signature = (frame_ids, retention_times, frame_occurrences, rts, sigmas, lambdas, rt_cycle_length, num_threads, n_steps=None))]
pub fn calculate_frame_abundances_emg_par(frame_ids: Vec<i32>, retention_times: Vec<f64>, frame_occurrences: Vec<Vec<i32>>, rts: Vec<f64>, sigmas: Vec<f64>, lambdas: Vec<f64>, rt_cycle_length: f64, num_threads: usize, n_steps: Option<usize>) -> Vec<Vec<f64>> {
    let time_map: HashMap<i32, f64> = frame_ids.iter().zip(retention_times.iter()).map(|(id, rt)| (*id, *rt)).collect();
    mscore::algorithm::utility::calculate_frame_abundances_emg_par(&time_map, frame_occurrences, rts, sigmas, lambdas, rt_cycle_length, num_threads, n_steps)
}

#[pyfunction]
pub fn calculate_occurence_gaussian(times: Vec<f64>, mean: f64, sigma: f64, target_p: f64, step_size: f64, n_lower_start: f64, n_upper_start: f64) -> Vec<i32> {
    mscore::algorithm::utility::calculate_occurrence_gaussian(&times, mean, sigma, target_p, step_size, n_lower_start, n_upper_start)
}

#[pyfunction]
pub fn calculate_abundance_gaussian(indices: Vec<i32>, times: Vec<f64>, occurrences: Vec<i32>, mean: f64, sigma: f64, cycle_length: f64) -> Vec<f64> {
    let time_map: HashMap<i32, f64> = indices.iter().zip(times.iter()).map(|(id, t)| (*id, *t)).collect();
    mscore::algorithm::utility::calculate_abundance_gaussian(&time_map, &occurrences, mean, sigma, cycle_length)
}

#[pyfunction]
pub fn calculate_occurrences_gaussian_par(times: Vec<f64>, means: Vec<f64>, sigmas: Vec<f64>, target_p: f64, step_size: f64, n_lower_start: f64, n_upper_start: f64, num_threads: usize) -> Vec<Vec<i32>> {
    mscore::algorithm::utility::calculate_occurrences_gaussian_par(&times, means, sigmas, target_p, step_size, n_lower_start, n_upper_start, num_threads)
}

#[pyfunction]
pub fn calculate_abundances_gaussian_par(indices: Vec<i32>, times: Vec<f64>, occurrences: Vec<Vec<i32>>, means: Vec<f64>, sigmas: Vec<f64>, cycle_length: f64, num_threads: usize) -> Vec<Vec<f64>> {
    let time_map: HashMap<i32, f64> = indices.iter().zip(times.iter()).map(|(id, t)| (*id, *t)).collect();
    mscore::algorithm::utility::calculate_abundances_gaussian_par(&time_map, occurrences, means, sigmas, cycle_length, num_threads)
}

#[pymodule]
pub fn py_utility(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(emg_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(accumulated_cdf_emg, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_bounds_emg, m)?)?;
    m.add_function(wrap_pyfunction!(normal_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(accumulated_cdf_normal, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_bounds_normal, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_frame_occurrence_emg, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_frame_abundance_emg, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_frame_occurrences_emg_par, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_frame_abundances_emg_par, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_occurence_gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_abundance_gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_occurrences_gaussian_par, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_abundances_gaussian_par, m)?)?;
    Ok(())
}
