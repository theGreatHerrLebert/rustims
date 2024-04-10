extern crate statrs;

use std::collections::{BTreeMap, HashMap, HashSet};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use statrs::distribution::{Continuous, Normal};
use crate::chemistry::constants::{MASS_NEUTRON, MASS_PROTON};
use crate::chemistry::elements::{atoms_isotopic_weights, isotopic_abundance};
use crate::data::peptide::PeptideIon;
use crate::data::spectrum::MzSpectrum;
use crate::data::spectrum::ToResolution;

/// convolve two distributions of masses and abundances
///
/// Arguments:
///
/// * `dist_a` - first distribution of masses and abundances
/// * `dist_b` - second distribution of masses and abundances
/// * `mass_tolerance` - mass tolerance for combining peaks
/// * `abundance_threshold` - minimum abundance for a peak to be included in the result
/// * `max_results` - maximum number of peaks to include in the result
///
/// Returns:
///
/// * `Vec<(f64, f64)>` - combined distribution of masses and abundances
///
/// # Examples
///
/// ```
/// use mscore::algorithm::isotope::convolve;
///
/// let dist_a = vec![(100.0, 0.5), (101.0, 0.5)];
/// let dist_b = vec![(100.0, 0.5), (101.0, 0.5)];
/// let result = convolve(&dist_a, &dist_b, 1e-6, 1e-12, 200);
/// assert_eq!(result, vec![(200.0, 0.25), (201.0, 0.5), (202.0, 0.25)]);
/// ```
pub fn convolve(dist_a: &Vec<(f64, f64)>, dist_b: &Vec<(f64, f64)>, mass_tolerance: f64, abundance_threshold: f64, max_results: usize) -> Vec<(f64, f64)> {

    let mut result: Vec<(f64, f64)> = Vec::new();

    for (mass_a, abundance_a) in dist_a {
        for (mass_b, abundance_b) in dist_b {
            let combined_mass = mass_a + mass_b;
            let combined_abundance = abundance_a * abundance_b;

            // Skip entries with combined abundance below the threshold
            if combined_abundance < abundance_threshold {
                continue;
            }

            // Insert or update the combined mass in the result distribution
            if let Some(entry) = result.iter_mut().find(|(m, _)| (*m - combined_mass).abs() < mass_tolerance) {
                entry.1 += combined_abundance;
            } else {
                result.push((combined_mass, combined_abundance));
            }
        }
    }

    // Sort by abundance (descending) to prepare for trimming
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Trim the vector if it exceeds max_results
    if result.len() > max_results {
        result.truncate(max_results);
    }

    // Optionally, sort by mass if needed for further processing
    result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    result
}

/// convolve a distribution with itself n times
///
/// Arguments:
///
/// * `dist` - distribution of masses and abundances
/// * `n` - number of times to convolve the distribution with itself
///
/// Returns:
///
/// * `Vec<(f64, f64)>` - distribution of masses and abundances
///
/// # Examples
///
/// ```
/// use mscore::algorithm::isotope::convolve_pow;
///
/// let dist = vec![(100.0, 0.5), (101.0, 0.5)];
/// let result = convolve_pow(&dist, 2);
/// assert_eq!(result, vec![(200.0, 0.25), (201.0, 0.5), (202.0, 0.25)]);
/// ```
pub fn convolve_pow(dist: &Vec<(f64, f64)>, n: i32) -> Vec<(f64, f64)> {
    if n == 0 {
        return vec![(0.0, 1.0)]; // Return the delta distribution
    }
    if n == 1 {
        return dist.clone();
    }

    let mut result = dist.clone();
    let mut power = 2;

    while power <= n {
        result = convolve(&result, &result, 1e-6, 1e-12, 200); // Square the result to get the next power of 2
        power *= 2;
    }

    // If n is not a power of 2, recursively fill in the remainder
    if power / 2 < n {
        result = convolve(&result, &convolve_pow(dist, n - power / 2, ), 1e-6, 1e-12, 200);
    }

    result
}

/// generate the isotope distribution for a given atomic composition
///
/// Arguments:
///
/// * `atomic_composition` - atomic composition of the peptide
/// * `mass_tolerance` - mass tolerance for combining peaks
/// * `abundance_threshold` - minimum abundance for a peak to be included in the result
/// * `max_result` - maximum number of peaks to include in the result
///
/// Returns:
///
/// * `Vec<(f64, f64)>` - distribution of masses and abundances
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use mscore::algorithm::isotope::generate_isotope_distribution;
///
/// let mut atomic_composition = HashMap::new();
/// atomic_composition.insert("C".to_string(), 5);
/// atomic_composition.insert("H".to_string(), 9);
/// atomic_composition.insert("N".to_string(), 1);
/// atomic_composition.insert("O".to_string(), 1);
/// let result = generate_isotope_distribution(&atomic_composition, 1e-6, 1e-12, 200);
/// ```
pub fn generate_isotope_distribution(
    atomic_composition: &HashMap<String, i32>,
    mass_tolerance: f64,
    abundance_threshold: f64,
    max_result: i32
) -> Vec<(f64, f64)> {

    let mut cumulative_distribution: Option<Vec<(f64, f64)>> = None;
    let atoms_isotopic_weights: HashMap<String, Vec<f64>> = atoms_isotopic_weights().iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
    let atomic_isotope_abundance: HashMap<String, Vec<f64>> = isotopic_abundance().iter().map(|(k, v)| (k.to_string(), v.clone())).collect();

    for (element, &count) in atomic_composition.iter() {
        let elemental_isotope_weights = atoms_isotopic_weights.get(element).expect("Element not found in isotopic weights table").clone();
        let elemental_isotope_abundance = atomic_isotope_abundance.get(element).expect("Element not found in isotopic abundance table").clone();

        let element_distribution: Vec<(f64, f64)> = elemental_isotope_weights.iter().zip(elemental_isotope_abundance.iter()).map(|(&mass, &abundance
                                                                                                                                  )| (mass, abundance)).collect();

        let element_power_distribution = if count > 1 {
            convolve_pow(&element_distribution, count)
        } else {
            element_distribution
        };

        cumulative_distribution = match cumulative_distribution {
            Some(cum_dist) => Some(convolve(&cum_dist, &element_power_distribution, mass_tolerance, abundance_threshold, max_result as usize)),
            None => Some(element_power_distribution),
        };
    }

    let final_distribution = cumulative_distribution.expect("Peptide has no elements");
    // Normalize the distribution
    let total_abundance: f64 = final_distribution.iter().map(|&(_, abundance)| abundance).sum();
    let result: Vec<_> = final_distribution.into_iter().map(|(mass, abundance)| (mass, abundance / total_abundance)).collect();

    let mut sort_map: BTreeMap<i64, f64> = BTreeMap::new();
    let quantize = |mz: f64| -> i64 { (mz * 1_000_000.0).round() as i64 };

    for (mz, intensity) in result {
        let key = quantize(mz);
        sort_map.entry(key).and_modify(|e| *e += intensity).or_insert(intensity);
    }

    let mz: Vec<f64> = sort_map.keys().map(|&key| key as f64 / 1_000_000.0).collect();
    let intensity: Vec<f64> = sort_map.values().map(|&intensity| intensity).collect();
    mz.iter().zip(intensity.iter()).map(|(&mz, &intensity)| (mz, intensity)).collect()
}


/// calculate the normal probability density function
///
/// Arguments:
///
/// * `x` - value to calculate the probability density function of
/// * `mean` - mean of the normal distribution
/// * `std_dev` - standard deviation of the normal distribution
///
/// Returns:
///
/// * `f64` - probability density function of `x`
///
/// # Examples
///
/// ```
/// use mscore::algorithm::isotope::normal_pdf;
///
/// let pdf = normal_pdf(0.0, 0.0, 1.0);
/// assert_eq!(pdf, 0.39894228040143265);
/// ```
pub fn normal_pdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    let normal = Normal::new(mean, std_dev).unwrap();
    normal.pdf(x)
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
/// use mscore::algorithm::isotope::factorial;
///
/// let fact = factorial(5);
/// assert_eq!(fact, 120.0);
/// ```
pub fn factorial(n: i32) -> f64 {
    (1..=n).fold(1.0, |acc, x| acc * x as f64)
}

pub fn weight(mass: f64, peak_nums: Vec<i32>, normalize: bool) -> Vec<f64> {
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

/// calculate the lambda value for a given mass
///
/// Arguments:
///
/// * `mass` - mass of the peptide
/// * `slope` - slope of the linear regression
/// * `intercept` - intercept of the linear regression
///
/// Returns:
///
/// * `f64` - lambda value
///
/// # Examples
///
/// ```
/// use mscore::algorithm::isotope::lam;
///
/// let lambda = lam(1000.0, 0.000594, -0.03091);
/// assert_eq!(lambda, 0.56309);
pub fn lam(mass: f64, slope: f64, intercept: f64) -> f64 {
    slope * mass + intercept
}

/// calculate the isotope pattern for a given mass and charge based on the averagine model
/// using the normal distribution for peak shapes
///
/// Arguments:
///
/// * `x` - list of m/z values to probe
/// * `mass` - mass of the peptide
/// * `charge` - charge of the peptide
/// * `sigma` - standard deviation of the normal distribution
/// * `amp` - amplitude of the isotope pattern
/// * `k` - number of isotopes to consider
/// * `step_size` - step size for the m/z values to probe
///
/// Returns:
///
/// * `Vec<f64>` - isotope pattern
///
pub fn iso(x: &Vec<f64>, mass: f64, charge: f64, sigma: f64, amp: f64, k: usize, step_size: f64) -> Vec<f64> {
    let k_range: Vec<usize> = (0..k).collect();
    let means: Vec<f64> = k_range.iter().map(|&k_val| (mass + MASS_NEUTRON * k_val as f64) / charge).collect();
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

/// generate the isotope pattern for a given mass and charge
///
/// Arguments:
///
/// * `lower_bound` - lower bound of the isotope pattern
/// * `upper_bound` - upper bound of the isotope pattern
/// * `mass` - mass of the peptide
/// * `charge` - charge of the peptide
/// * `amp` - amplitude of the isotope pattern
/// * `k` - number of isotopes to consider
/// * `sigma` - standard deviation of the normal distribution
/// * `resolution` - resolution of the isotope pattern
///
/// Returns:
///
/// * `(Vec<f64>, Vec<f64>)` - isotope pattern
///
/// # Examples
///
/// ```
/// use mscore::algorithm::isotope::generate_isotope_pattern;
///
/// let (mzs, intensities) = generate_isotope_pattern(1500.0, 1510.0, 3000.0, 2.0, 1e4, 10, 1.0, 3);
/// ```
pub fn generate_isotope_pattern(lower_bound: f64, upper_bound: f64, mass: f64, charge: f64, amp: f64, k: usize, sigma: f64, resolution: i32) -> (Vec<f64>, Vec<f64>) {
    let step_size = f64::min(sigma / 10.0, 1.0 / 10f64.powi(resolution));
    let size = ((upper_bound - lower_bound) / step_size).ceil() as usize;
    let mzs: Vec<f64> = (0..size).map(|i| lower_bound + step_size * i as f64).collect();
    let intensities = iso(&mzs, mass, charge, sigma, amp, k, step_size);

    (mzs.iter().map(|&mz| mz + MASS_PROTON).collect(), intensities)
}

/// generate the averagine spectrum for a given mass and charge
///
/// Arguments:
///
/// * `mass` - mass of the peptide
/// * `charge` - charge of the peptide
/// * `min_intensity` - minimum intensity for a peak to be included in the result
/// * `k` - number of isotopes to consider
/// * `resolution` - resolution of the isotope pattern
/// * `centroid` - whether to centroid the spectrum
/// * `amp` - amplitude of the isotope pattern
///
/// Returns:
///
/// * `MzSpectrum` - averagine spectrum
///
/// # Examples
///
/// ```
/// use mscore::algorithm::isotope::generate_averagine_spectrum;
///
/// let spectrum = generate_averagine_spectrum(3000.0, 2, 1, 10, 3, true, None);
/// ```
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
        spectrum.to_centroid(std::cmp::max(min_intensity, 1), 1.0 / 10f64.powi(resolution - 1), true)
    } else {
        spectrum
    }
}

/// generate the averagine spectra for a given list of masses and charges
/// using multiple threads
///
/// Arguments:
///
/// * `masses` - list of masses of the peptides
/// * `charges` - list of charges of the peptides
/// * `min_intensity` - minimum intensity for a peak to be included in the result
/// * `k` - number of isotopes to consider
/// * `resolution` - resolution of the isotope pattern
/// * `centroid` - whether to centroid the spectrum
/// * `num_threads` - number of threads to use
/// * `amp` - amplitude of the isotope pattern
///
/// Returns:
///
/// * `Vec<MzSpectrum>` - list of averagine spectra
///
/// # Examples
///
/// ```
/// use mscore::algorithm::isotope::generate_averagine_spectra;
///
/// let masses = vec![3000.0, 3000.0];
/// let charges = vec![2, 3];
/// let spectra = generate_averagine_spectra(masses, charges, 1, 10, 3, true, 4, None);
/// ```
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

/// generate the precursor spectrum for a given peptide sequence and charge
/// using isotope convolutions
///
/// Arguments:
///
/// * `sequence` - peptide sequence
/// * `charge` - charge of the peptide
///
/// Returns:
///
/// * `MzSpectrum` - precursor spectrum
///
pub fn generate_precursor_spectrum(sequence: &str, charge: i32, peptide_id: Option<i32>) -> MzSpectrum {
    let peptide_ion = PeptideIon::new(sequence.to_string(), charge, 1.0, peptide_id);
    peptide_ion.calculate_isotopic_spectrum(1e-3, 1e-9, 200, 1e-6)
}

/// parallel version of `generate_precursor_spectrum`
///
/// Arguments:
///
/// * `sequences` - list of peptide sequences
/// * `charges` - list of charges of the peptides
/// * `num_threads` - number of threads to use
///
/// Returns:
///
/// * `Vec<MzSpectrum>` - list of precursor spectra
///
pub fn generate_precursor_spectra(sequences: &Vec<&str>, charges: &Vec<i32>, num_threads: usize, peptide_ids: Vec<Option<i32>>) -> Vec<MzSpectrum> {
    let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    // need to zip sequences and charges and peptide_ids
    let result = thread_pool.install(|| {
        sequences.par_iter().zip(charges.par_iter()).zip(peptide_ids.par_iter()).map(|((&sequence, &charge), &peptide_id)| {
            generate_precursor_spectrum(sequence, charge, peptide_id)
        }).collect()
    });
    result
}

// Calculates the isotope distribution for a fragment given the isotope distribution of the fragment, the isotope distribution of the complementary fragment, and the transmitted precursor isotopes
// implemented based on OpenMS: "https://github.com/OpenMS/OpenMS/blob/079143800f7ed036a7c68ea6e124fe4f5cfc9569/src/openms/source/CHEMISTRY/ISOTOPEDISTRIBUTION/CoarseIsotopePatternGenerator.cpp#L415"
pub fn calculate_transmission_dependent_fragment_ion_isotope_distribution(fragment_isotope_dist: &Vec<(f64, f64)>, comp_fragment_isotope_dist: &Vec<(f64, f64)>, precursor_isotopes: &HashSet<usize>, max_isotope: usize) -> Vec<(f64, f64)> {

    if fragment_isotope_dist.is_empty() || comp_fragment_isotope_dist.is_empty() {
        return Vec::new();
    }

    let mut r_max = fragment_isotope_dist.len();
    if max_isotope != 0 && r_max > max_isotope {
        r_max = max_isotope;
    }

    let mut result = (0..r_max).map(|i| (fragment_isotope_dist[0].0 + i as f64, 0.0)).collect::<Vec<(f64, f64)>>();

    // Calculation of dependent isotope distribution
    for (i, &(_mz, intensity)) in fragment_isotope_dist.iter().enumerate().take(r_max) {
        for &precursor in precursor_isotopes {
            if precursor >= i && (precursor - i) < comp_fragment_isotope_dist.len() {
                let comp_intensity = comp_fragment_isotope_dist[precursor - i].1;
                result[i].1 += comp_intensity;
            }
        }
        result[i].1 *= intensity;
    }

    result
}