extern crate statrs;
use statrs::distribution::{Continuous, Normal};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::{MzSpectrum, ToResolution};

fn normal_pdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    let normal = Normal::new(mean, std_dev).unwrap();
    normal.pdf(x)
}

const MASS_PROTON: f64 = 1.007276466621;
const MASS_NEUTRON: f64 = 1.00866491595;

/// convert 1 over reduced ion mobility (1/k0) to CCS
///
/// Arguments:
///
/// * `one_over_k0` - 1 over reduced ion mobility (1/k0)
/// * `charge` - charge state of the ion
/// * `mz` - mass-over-charge of the ion
/// * `mass_gas` - mass of drift gas (N2)
/// * `temp` - temperature of the drift gas in C°
/// * `t_diff` - factor to translate from C° to K
///
/// Returns:
///
/// * `ccs` - collision cross-section
///
/// # Examples
///
/// ```
/// use mscore::one_over_reduced_mobility_to_ccs;
///
/// let ccs = one_over_reduced_mobility_to_ccs(0.5, 1000.0, 2, 28.013, 31.85, 273.15);
/// assert_eq!(ccs, 806.5918693771381);
/// ```
pub fn one_over_reduced_mobility_to_ccs(
    one_over_k0: f64,
    mz: f64,
    charge: u32,
    mass_gas: f64,
    temp: f64,
    t_diff: f64,
) -> f64 {
    let summary_constant = 18509.8632163405;
    let reduced_mass = (mz * charge as f64 * mass_gas) / (mz * charge as f64 + mass_gas);
    summary_constant * charge as f64 / (reduced_mass * (temp + t_diff)).sqrt() / one_over_k0
}


/// convert CCS to 1 over reduced ion mobility (1/k0)
///
/// Arguments:
///
/// * `ccs` - collision cross-section
/// * `charge` - charge state of the ion
/// * `mz` - mass-over-charge of the ion
/// * `mass_gas` - mass of drift gas (N2)
/// * `temp` - temperature of the drift gas in C°
/// * `t_diff` - factor to translate from C° to K
///
/// Returns:
///
/// * `one_over_k0` - 1 over reduced ion mobility (1/k0)
///
/// # Examples
///
/// ```
/// use mscore::ccs_to_reduced_mobility;
///
/// let k0 = ccs_to_reduced_mobility(806.5918693771381, 1000.0, 2, 28.013, 31.85, 273.15);
/// assert_eq!(1.0 / k0, 0.5);
/// ```
pub fn ccs_to_reduced_mobility(
    ccs: f64,
    mz: f64,
    charge: u32,
    mass_gas: f64,
    temp: f64,
    t_diff: f64,
) -> f64 {
    let summary_constant = 18509.8632163405;
    let reduced_mass = (mz * charge as f64 * mass_gas) / (mz * charge as f64 + mass_gas);
    ((reduced_mass * (temp + t_diff)).sqrt() * ccs) / (summary_constant * charge as f64)
}


/// calculate the factorial of a number
///
/// Arguments:
///
/// * `n` - number to calculate factorial of
///
/// Returns:
///
/// * `f64` - factorial of `n`
///
/// # Examples
///
/// ```
/// use mscore::factorial;
///
/// let fact = factorial(5);
/// assert_eq!(fact, 120.0);
/// ```
pub fn factorial(n: i32) -> f64 {
    (1..=n).fold(1.0, |acc, x| acc * x as f64)
}

fn weight(mass: f64, peak_nums: Vec<i32>, normalize: bool) -> Vec<f64> {
    let lam_val = lam(mass, 0.000594, -0.03091);
    let factorials: Vec<f64> = peak_nums.iter().map(|&k| factorial(k)).collect();
    let mut weights: Vec<f64> = peak_nums.iter().map(|&k| {
        let pow = lam_val.powi(k);
        let exp = (-lam_val).exp();
        exp * pow / factorials[k as usize]
    }).collect();

    if normalize {
        let sum: f64 = weights.iter().sum();
        weights = weights.iter().map(|&w| w / sum).collect();
    }

    weights
}

fn lam(mass: f64, slope: f64, intercept: f64) -> f64 {
    slope * mass + intercept
}

fn iso(x: &Vec<f64>, mass: f64, charge: f64, sigma: f64, amp: f64, k: usize, step_size: f64, mass_neutron: f64) -> Vec<f64> {
    let k_range: Vec<usize> = (0..k).collect();
    let means: Vec<f64> = k_range.iter().map(|&k_val| (mass + mass_neutron * k_val as f64) / charge).collect();
    let weights = weight(mass, k_range.iter().map(|&k_val| k_val as i32).collect::<Vec<i32>>(), true);

    let mut intensities = vec![0.0; x.len()];
    for (i, x_val) in x.iter().enumerate() {
        for (j, &mean) in means.iter().enumerate() {
            intensities[i] += weights[j] * normal_pdf(*x_val, mean, sigma);
        }
        intensities[i] *= step_size;
    }
    intensities.iter().map(|&intensity| intensity * amp).collect()
}

fn generate_isotope_pattern(lower_bound: f64, upper_bound: f64, mass: f64, charge: f64, amp: f64, k: usize, sigma: f64, resolution: i32) -> (Vec<f64>, Vec<f64>) {
    let step_size = f64::min(sigma / 10.0, 1.0 / 10f64.powi(resolution));
    let size = ((upper_bound - lower_bound) / step_size).ceil() as usize;
    let mzs: Vec<f64> = (0..size).map(|i| lower_bound + step_size * i as f64).collect();
    let intensities = iso(&mzs, mass, charge, sigma, amp, k, step_size, MASS_NEUTRON);

    (mzs.iter().map(|&mz| mz + MASS_PROTON).collect(), intensities)
}

pub fn generate_averagine_spectrum(
    mass: f64,
    charge: i32,
    min_intensity: i32,
    k: i32,
    resolution: i32,
    centroid: bool,
    amp: Option<f64>
) -> MzSpectrum {
    let amp = amp.unwrap_or(1e4);
    let lb = mass / charge as f64 - 0.2;
    let ub = mass / charge as f64 + k as f64 + 0.2;

    let (mz, intensities) = generate_isotope_pattern(
        lb,
        ub,
        mass,
        charge as f64,
        amp,
        k as usize,
        0.008492569002123142,
        resolution,
    );

    let spectrum = MzSpectrum::new(mz, intensities).to_resolution(resolution).filter_ranged(lb, ub, min_intensity as f64, 1e9);

    if centroid {
        spectrum.to_centroided(std::cmp::max(min_intensity, 1), 1.0 / 10f64.powi(resolution - 1), true)
    } else {
        spectrum
    }
}

pub fn generate_averagine_spectra(
    masses: Vec<f64>,
    charges: Vec<i32>,
    min_intensity: i32,
    k: i32,
    resolution: i32,
    centroid: bool,
    num_threads: usize,
    amp: Option<f64>
) -> Vec<MzSpectrum> {
    let amp = amp.unwrap_or(1e5);
    let mut spectra: Vec<MzSpectrum> = Vec::new();
    let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();

    thread_pool.install(|| {
        spectra = masses.par_iter().zip(charges.par_iter()).map(|(&mass, &charge)| {
            generate_averagine_spectrum(mass, charge, min_intensity, k, resolution, centroid, Some(amp))
        }).collect();
    });

    spectra
}

