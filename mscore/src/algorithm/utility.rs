extern crate rgsl;

use rgsl::{IntegrationWorkspace, error::erfc, error::erf};
use std::f64::consts::SQRT_2;

pub fn custom_cdf_normal(x: f64, mean: f64, std_dev: f64) -> f64 {
    let z = (x - mean) / std_dev;
    0.5 * (1.0 + erf(z / SQRT_2))
}

pub fn accumulated_intensity_cdf_normal(sample_start: f64, sample_end: f64, mean: f64, std_dev: f64) -> f64 {
    let cdf_start = custom_cdf_normal(sample_start, mean, std_dev);
    let cdf_end = custom_cdf_normal(sample_end, mean, std_dev);
    cdf_end - cdf_start
}

pub fn calculate_bounds_normal(mean: f64, std: f64, z_score: f64) -> (f64, f64) {
    (mean - z_score * std, mean + z_score * std)
}

pub fn emg_function(x: f64, mu: f64, sigma: f64, lambda: f64) -> f64 {
    let prefactor = lambda / 2.0 * ((lambda / 2.0) * (2.0 * mu + lambda * sigma.powi(2) - 2.0 * x)).exp();
    let erfc_part = erfc((mu + lambda * sigma.powi(2) - x) / (SQRT_2 * sigma));
    prefactor * erfc_part
}

pub fn emg_cdf_range(lower_limit: f64, upper_limit: f64, mu: f64, sigma: f64, lambda: f64) -> f64 {
    let mut workspace = IntegrationWorkspace::new(1000).expect("IntegrationWorkspace::new failed");

    let (result, _) = workspace.qags(
        |x| emg_function(x, mu, sigma, lambda),
        lower_limit,
        upper_limit,
        0.0,
        1e-7,
        1000,
    )
        .unwrap();

    result
}

pub fn find_bounds_for_target_probability_emg(mu: f64, sigma: f64, lambda: f64, target_prob: f64, step_size: f64) -> (f64, f64) {
    let range_start = mu - 10.0 * sigma;
    let range_end = mu + 10.0 * sigma;
    let steps = ((range_end - range_start) / step_size).round() as usize;
    let search_space: Vec<f64> = (0..=steps).map(|i| range_start + i as f64 * step_size).collect();

    // Helper function to find the cumulative probability up to a certain index in the search space
    let cumulative_prob_at_index = |index: usize| -> f64 {
        emg_cdf_range(mu, search_space[index], mu, sigma, lambda)
    };

    // Binary search for the left boundary
    let mut low = 0;
    let mut high = steps / 2; // Assuming the distribution is centered around mu, adjust if not
    while low < high {
        let mid = (low + high) / 2;
        if cumulative_prob_at_index(mid) < (1.0 - target_prob) / 2.0 {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let left_boundary = search_space[low];

    // Reset for right boundary search
    low = steps / 2;
    high = steps;

    // Binary search for the right boundary
    while low < high {
        let mid = (low + high) / 2;
        if cumulative_prob_at_index(mid) - cumulative_prob_at_index(steps / 2) < target_prob / 2.0 {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let right_boundary = search_space[low];

    (left_boundary, right_boundary)
}