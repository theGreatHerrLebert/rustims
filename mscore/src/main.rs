use mscore::algorithm::utility::{calculate_bounds_emg, emg_cdf_range};

fn main() {
    let mu = 0.0;
    let sigma = 1.0;
    let lambda = 1.0;
    let n_steps: usize = 1000;
    let step_size = 0.0001;
    let l_s = 25.0;
    let u_s = 25.0;

    let (lb, ub) = calculate_bounds_emg(mu, sigma, lambda, step_size, 0.99, l_s, u_s, Some(n_steps));

    let cdf_range = emg_cdf_range(lb, ub, mu, sigma, lambda, Some(n_steps));

    println!("EMG CDF from {} to {} = {}", lb, ub, cdf_range);
}
