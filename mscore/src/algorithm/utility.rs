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

pub fn find_bounds_for_target_probability_emg(mu: f64, sigma: f64, lambda: f64, target_prob: f64) -> (f64, f64) {
    // Initial bounds based on the distribution's skewness
    let mut lower_bound = mu - 10.0 * sigma; // Consider less extension if skew is to the right
    let mut upper_bound = mu + 10.0 * sigma + lambda; // Extend more if skew is to the right

    let mut prob = emg_cdf_range(lower_bound, upper_bound, mu, sigma, lambda);

    while (prob - target_prob).abs() > 1e-5 { // Arbitrary tolerance
        if prob < target_prob {
            // Expand the range more on the skewed side
            lower_bound -= 0.5 * sigma; // Smaller adjustment if skew is to the right
            upper_bound += sigma + 0.1 * lambda; // Larger adjustment to accommodate skew
        } else {
            // Narrow the range, more nuanced adjustment
            let mid_point = (lower_bound + upper_bound) / 2.0;
            let mid_prob = emg_cdf_range(lower_bound, mid_point, mu, sigma, lambda);
            // Decide which half to keep, considering skew
            if mid_prob < target_prob * 0.5 {
                lower_bound = mid_point;
            } else {
                upper_bound = mid_point;
            }
        }
        prob = emg_cdf_range(lower_bound, upper_bound, mu, sigma, lambda);
    }

    (lower_bound, upper_bound)
}