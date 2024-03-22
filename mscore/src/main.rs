use mscore::algorithm::utility::emg_cdf_range;

fn main() {
    let mu = 0.0;
    let sigma = 1.0;
    let lambda = 1.0;
    let lower_limit = 0.0;
    let upper_limit = 2.0;

    let cdf_range = emg_cdf_range(lower_limit, upper_limit, mu, sigma, lambda);

    println!("EMG CDF from {} to {} = {}", lower_limit, upper_limit, cdf_range);
}
