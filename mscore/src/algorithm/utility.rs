extern crate rgsl;

use std::collections::HashMap;
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

pub fn calculate_bounds_emg(mu: f64, sigma: f64, lambda: f64, step_size: f64, target: f64, lower_start: f64, upper_start: f64) -> (f64, f64) {
    assert!(0.0 <= target && target <= 1.0, "target must be in [0, 1]");

    let lower_initial = mu - lower_start * sigma;
    let upper_initial = mu + upper_start * sigma;

    let steps = ((upper_initial - lower_initial) / step_size).round() as usize;
    let search_space: Vec<f64> = (0..=steps).map(|i| lower_initial + i as f64 * step_size).collect();

    let calc_cdf = |low: usize, high: usize| -> f64 {
        emg_cdf_range(search_space[low], search_space[high], mu, sigma, lambda)
    };

    // Binary search for cutoff values
    let (mut low, mut high) = (0, steps);
    while low < high {
        let mid = low + (high - low) / 2;
        if calc_cdf(0, mid) < target {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let upper_cutoff_index = low;

    low = 0;
    high = upper_cutoff_index;
    while low < high {
        let mid = high - (high - low) / 2;
        let prob_mid_to_upper = calc_cdf(mid, upper_cutoff_index);

        if prob_mid_to_upper < target {
            high = mid - 1;
        } else {
            low = mid;
        }
    }
    let lower_cutoff_index = high;

    (search_space[lower_cutoff_index], search_space[upper_cutoff_index])
}

pub fn calculate_frame_occurrence_emg(retention_times: &[f64], rt: f64, sigma: f64, lambda_: f64) -> Vec<i32> {
    let step_size = 0.001;
    let target = 0.99;
    let (rt_min, rt_max) = calculate_bounds_emg(rt, sigma, lambda_, step_size, target, 20.0, 60.0);

    // Finding the frame closest to rt_min
    let first_frame = retention_times.iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| (a - rt_min).abs().partial_cmp(&(b - rt_min).abs()).unwrap())
        .map(|(idx, _)| idx + 1) // Rust is zero-indexed, so +1 to match Python's 1-indexing
        .unwrap_or(0); // Fallback in case of an empty slice

    // Finding the frame closest to rt_max
    let last_frame = retention_times.iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| (a - rt_max).abs().partial_cmp(&(b - rt_max).abs()).unwrap())
        .map(|(idx, _)| idx + 1) // Same adjustment for 1-indexing
        .unwrap_or(0); // Fallback

    // Generating the range of frames
    (first_frame..=last_frame).map(|x| x as i32).collect()
}

pub fn calculate_frame_abundance_emg(time_map: &HashMap<i32, f64>, occurrences: &[i32], rt: f64, sigma: f64, lambda_: f64, rt_cycle_length: f64) -> Vec<f64> {
    let mut frame_abundance = Vec::new();

    for &occurrence in occurrences {
        if let Some(&time) = time_map.get(&occurrence) {
            let start = time - rt_cycle_length;
            let i = emg_cdf_range(start, time, rt, sigma, lambda_);
            frame_abundance.push(i);
        }
    }

    frame_abundance
}
