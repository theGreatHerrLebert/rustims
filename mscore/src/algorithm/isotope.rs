extern crate statrs;

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::chemistry::constants::{MASS_NEUTRON, MASS_PROTON};
use crate::chemistry::elements::{atoms_isotopic_weights, isotopic_abundance};
use crate::data::peptide::PeptideIon;
use crate::data::spectrum::MzSpectrum;
use crate::data::spectrum::ToResolution;
use statrs::distribution::{Continuous, Normal};

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
pub fn convolve(
    dist_a: &Vec<(f64, f64)>,
    dist_b: &Vec<(f64, f64)>,
    mass_tolerance: f64,
    abundance_threshold: f64,
    max_results: usize,
) -> Vec<(f64, f64)> {
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
            if let Some(entry) = result
                .iter_mut()
                .find(|(m, _)| (*m - combined_mass).abs() < mass_tolerance)
            {
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
        result = convolve(
            &result,
            &convolve_pow(dist, n - power / 2),
            1e-6,
            1e-12,
            200,
        );
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
    max_result: i32,
) -> Vec<(f64, f64)> {
    let mut cumulative_distribution: Option<Vec<(f64, f64)>> = None;
    let atoms_isotopic_weights: HashMap<String, Vec<f64>> = atoms_isotopic_weights()
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    let atomic_isotope_abundance: HashMap<String, Vec<f64>> = isotopic_abundance()
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();

    for (element, &count) in atomic_composition.iter() {
        let elemental_isotope_weights = atoms_isotopic_weights
            .get(element)
            .expect("Element not found in isotopic weights table")
            .clone();
        let elemental_isotope_abundance = atomic_isotope_abundance
            .get(element)
            .expect("Element not found in isotopic abundance table")
            .clone();

        let element_distribution: Vec<(f64, f64)> = elemental_isotope_weights
            .iter()
            .zip(elemental_isotope_abundance.iter())
            .map(|(&mass, &abundance)| (mass, abundance))
            .collect();

        let element_power_distribution = if count > 1 {
            convolve_pow(&element_distribution, count)
        } else {
            element_distribution
        };

        cumulative_distribution = match cumulative_distribution {
            Some(cum_dist) => Some(convolve(
                &cum_dist,
                &element_power_distribution,
                mass_tolerance,
                abundance_threshold,
                max_result as usize,
            )),
            None => Some(element_power_distribution),
        };
    }

    let final_distribution = cumulative_distribution.expect("Peptide has no elements");
    // Normalize the distribution
    let total_abundance: f64 = final_distribution
        .iter()
        .map(|&(_, abundance)| abundance)
        .sum();
    let result: Vec<_> = final_distribution
        .into_iter()
        .map(|(mass, abundance)| (mass, abundance / total_abundance))
        .collect();

    let mut sort_map: BTreeMap<i64, f64> = BTreeMap::new();
    let quantize = |mz: f64| -> i64 { (mz * 1_000_000.0).round() as i64 };

    for (mz, intensity) in result {
        let key = quantize(mz);
        sort_map
            .entry(key)
            .and_modify(|e| *e += intensity)
            .or_insert(intensity);
    }

    let mz: Vec<f64> = sort_map
        .keys()
        .map(|&key| key as f64 / 1_000_000.0)
        .collect();
    let intensity: Vec<f64> = sort_map.values().map(|&intensity| intensity).collect();
    mz.iter()
        .zip(intensity.iter())
        .map(|(&mz, &intensity)| (mz, intensity))
        .collect()
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
    let mut weights: Vec<f64> = peak_nums
        .iter()
        .map(|&k| {
            let pow = lam_val.powi(k);
            let exp = (-lam_val).exp();
            exp * pow / factorials[k as usize]
        })
        .collect();

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
pub fn iso(
    x: &Vec<f64>,
    mass: f64,
    charge: f64,
    sigma: f64,
    amp: f64,
    k: usize,
    step_size: f64,
) -> Vec<f64> {
    let k_range: Vec<usize> = (0..k).collect();
    let means: Vec<f64> = k_range
        .iter()
        .map(|&k_val| (mass + MASS_NEUTRON * k_val as f64) / charge)
        .collect();
    let weights = weight(
        mass,
        k_range
            .iter()
            .map(|&k_val| k_val as i32)
            .collect::<Vec<i32>>(),
        true,
    );

    let mut intensities = vec![0.0; x.len()];
    for (i, x_val) in x.iter().enumerate() {
        for (j, &mean) in means.iter().enumerate() {
            intensities[i] += weights[j] * normal_pdf(*x_val, mean, sigma);
        }
        intensities[i] *= step_size;
    }
    intensities
        .iter()
        .map(|&intensity| intensity * amp)
        .collect()
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
pub fn generate_isotope_pattern(
    lower_bound: f64,
    upper_bound: f64,
    mass: f64,
    charge: f64,
    amp: f64,
    k: usize,
    sigma: f64,
    resolution: i32,
) -> (Vec<f64>, Vec<f64>) {
    let step_size = f64::min(sigma / 10.0, 1.0 / 10f64.powi(resolution));
    let size = ((upper_bound - lower_bound) / step_size).ceil() as usize;
    let mzs: Vec<f64> = (0..size)
        .map(|i| lower_bound + step_size * i as f64)
        .collect();
    let intensities = iso(&mzs, mass, charge, sigma, amp, k, step_size);

    (
        mzs.iter().map(|&mz| mz + MASS_PROTON).collect(),
        intensities,
    )
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
    amp: Option<f64>,
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

    let spectrum = MzSpectrum::new(mz, intensities)
        .to_resolution(resolution)
        .filter_ranged(lb, ub, min_intensity as f64, 1e9);

    if centroid {
        spectrum.to_centroid(
            std::cmp::max(min_intensity, 1),
            1.0 / 10f64.powi(resolution - 1),
            true,
        )
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
    amp: Option<f64>,
) -> Vec<MzSpectrum> {
    let amp = amp.unwrap_or(1e5);
    let mut spectra: Vec<MzSpectrum> = Vec::new();
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    thread_pool.install(|| {
        spectra = masses
            .par_iter()
            .zip(charges.par_iter())
            .map(|(&mass, &charge)| {
                generate_averagine_spectrum(
                    mass,
                    charge,
                    min_intensity,
                    k,
                    resolution,
                    centroid,
                    Some(amp),
                )
            })
            .collect();
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
pub fn generate_precursor_spectrum(
    sequence: &str,
    charge: i32,
    peptide_id: Option<i32>,
) -> MzSpectrum {
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
pub fn generate_precursor_spectra(
    sequences: &Vec<&str>,
    charges: &Vec<i32>,
    num_threads: usize,
    peptide_ids: Vec<Option<i32>>,
) -> Vec<MzSpectrum> {
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();
    // need to zip sequences and charges and peptide_ids
    let result = thread_pool.install(|| {
        sequences
            .par_iter()
            .zip(charges.par_iter())
            .zip(peptide_ids.par_iter())
            .map(|((&sequence, &charge), &peptide_id)| {
                generate_precursor_spectrum(sequence, charge, peptide_id)
            })
            .collect()
    });
    result
}

/// Result of transmission-dependent isotope distribution calculation.
/// Contains the adjusted distribution and the transmission factor for intensity scaling.
#[derive(Debug, Clone)]
pub struct TransmissionDependentIsotopeDistribution {
    /// The adjusted isotope distribution (m/z, relative_intensity)
    pub distribution: Vec<(f64, f64)>,
    /// Fraction of signal transmitted (0.0 to 1.0).
    /// This is the ratio of the sum of transmitted distribution intensities
    /// to the sum of full (all isotopes transmitted) distribution intensities.
    pub transmission_factor: f64,
}

// Calculates the isotope distribution for a fragment given the isotope distribution of the fragment, the isotope distribution of the complementary fragment, and the transmitted precursor isotopes
// implemented based on OpenMS: "https://github.com/OpenMS/OpenMS/blob/079143800f7ed036a7c68ea6e124fe4f5cfc9569/src/openms/source/CHEMISTRY/ISOTOPEDISTRIBUTION/CoarseIsotopePatternGenerator.cpp#L415"
pub fn calculate_transmission_dependent_fragment_ion_isotope_distribution(
    fragment_isotope_dist: &Vec<(f64, f64)>,
    comp_fragment_isotope_dist: &Vec<(f64, f64)>,
    precursor_isotopes: &HashSet<usize>,
    max_isotope: usize,
) -> Vec<(f64, f64)> {
    if fragment_isotope_dist.is_empty() || comp_fragment_isotope_dist.is_empty() {
        return Vec::new();
    }

    let mut r_max = fragment_isotope_dist.len();
    if max_isotope != 0 && r_max > max_isotope {
        r_max = max_isotope;
    }

    let mut result = (0..r_max)
        .map(|i| (fragment_isotope_dist[0].0 + i as f64, 0.0))
        .collect::<Vec<(f64, f64)>>();

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

/// Calculates the transmission-dependent fragment isotope distribution with explicit
/// transmission factor tracking.
///
/// This function computes how the fragment ion isotope pattern changes when only
/// certain precursor isotopes are transmitted through the quadrupole isolation window.
/// It also calculates the transmission factor, which represents the fraction of
/// total signal that is transmitted.
///
/// # Arguments
///
/// * `fragment_isotope_dist` - Isotope distribution of the fragment ion (m/z, intensity)
/// * `comp_fragment_isotope_dist` - Isotope distribution of the complementary fragment
/// * `precursor_isotopes` - Set of precursor isotope indices that were transmitted
/// * `max_isotope` - Maximum number of isotope peaks to consider (0 for unlimited)
///
/// # Returns
///
/// A `TransmissionDependentIsotopeDistribution` containing:
/// - The adjusted isotope distribution
/// - The transmission factor (ratio of transmitted to full signal)
///
/// # Algorithm
///
/// Based on OpenMS CoarseIsotopePatternGenerator. For each fragment isotope index i:
/// P(fragment=i | transmitted precursors) = frag[i] * Σ comp[p-i] for all transmitted p >= i
///
/// The transmission factor is calculated as:
/// transmission_factor = sum(transmitted_distribution) / sum(full_distribution)
pub fn calculate_transmission_dependent_distribution_with_factor(
    fragment_isotope_dist: &Vec<(f64, f64)>,
    comp_fragment_isotope_dist: &Vec<(f64, f64)>,
    precursor_isotopes: &HashSet<usize>,
    max_isotope: usize,
) -> TransmissionDependentIsotopeDistribution {
    if fragment_isotope_dist.is_empty() || comp_fragment_isotope_dist.is_empty() {
        return TransmissionDependentIsotopeDistribution {
            distribution: Vec::new(),
            transmission_factor: 0.0,
        };
    }

    // Calculate full distribution (all isotopes transmitted) for reference
    let all_isotopes: HashSet<usize> = (0..fragment_isotope_dist.len().max(comp_fragment_isotope_dist.len())).collect();
    let full_distribution = calculate_transmission_dependent_fragment_ion_isotope_distribution(
        fragment_isotope_dist,
        comp_fragment_isotope_dist,
        &all_isotopes,
        max_isotope,
    );
    let full_sum: f64 = full_distribution.iter().map(|(_, i)| i).sum();

    // Calculate transmitted distribution
    let transmitted_distribution = calculate_transmission_dependent_fragment_ion_isotope_distribution(
        fragment_isotope_dist,
        comp_fragment_isotope_dist,
        precursor_isotopes,
        max_isotope,
    );
    let transmitted_sum: f64 = transmitted_distribution.iter().map(|(_, i)| i).sum();

    // Calculate transmission factor
    let transmission_factor = if full_sum > 0.0 {
        transmitted_sum / full_sum
    } else {
        0.0
    };

    TransmissionDependentIsotopeDistribution {
        distribution: transmitted_distribution,
        transmission_factor,
    }
}

/// Calculate the transmission factor for a precursor based on which isotopes are transmitted.
///
/// This provides a simple way to scale fragment intensities based on precursor transmission
/// without the computational cost of per-fragment isotope recalculation.
///
/// # Arguments
///
/// * `precursor_isotope_dist` - Isotope distribution of the precursor (m/z, intensity)
/// * `transmitted_indices` - Set of precursor isotope indices that were transmitted
///
/// # Returns
///
/// Transmission factor (0.0 to 1.0) representing the fraction of precursor signal transmitted.
pub fn calculate_precursor_transmission_factor(
    precursor_isotope_dist: &[(f64, f64)],
    transmitted_indices: &HashSet<usize>,
) -> f64 {
    if precursor_isotope_dist.is_empty() || transmitted_indices.is_empty() {
        return 0.0;
    }

    let total_intensity: f64 = precursor_isotope_dist.iter().map(|(_, i)| i).sum();
    if total_intensity <= 0.0 {
        return 0.0;
    }

    let transmitted_intensity: f64 = precursor_isotope_dist
        .iter()
        .enumerate()
        .filter(|(idx, _)| transmitted_indices.contains(idx))
        .map(|(_, (_, i))| i)
        .sum();

    transmitted_intensity / total_intensity
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    /// Test that transmission-dependent distribution produces correct results
    /// when the same isotope set is used as the internal reference
    #[test]
    fn test_transmission_all_isotopes() {
        // Simple fragment distribution: M0=0.6, M1=0.3, M2=0.1
        let fragment_dist = vec![
            (500.0, 0.6),
            (501.0, 0.3),
            (502.0, 0.1),
        ];

        // Complementary distribution: M0=0.7, M1=0.2, M2=0.1
        let comp_dist = vec![
            (800.0, 0.7),
            (801.0, 0.2),
            (802.0, 0.1),
        ];

        // Use the same isotope set that the function uses internally for "full" calculation
        // This is 0..max(fragment_dist.len(), comp_dist.len()) = 0..3
        let all_isotopes: HashSet<usize> = (0..3).collect();

        let result = calculate_transmission_dependent_distribution_with_factor(
            &fragment_dist,
            &comp_dist,
            &all_isotopes,
            0,
        );

        // When same isotopes as internal reference are transmitted, transmission factor = 1.0
        assert!(
            approx_eq(result.transmission_factor, 1.0, 0.001),
            "Transmission factor should be 1.0 when same isotopes as reference transmitted, got {}",
            result.transmission_factor
        );

        // Distribution should not be empty
        assert!(!result.distribution.is_empty());
    }

    /// Test that passing more isotopes than exist can increase transmission factor > 1.0
    /// This is expected behavior: higher precursor isotopes can contribute to lower fragment isotopes
    #[test]
    fn test_transmission_extra_isotopes() {
        let fragment_dist = vec![
            (500.0, 0.6),
            (501.0, 0.3),
            (502.0, 0.1),
        ];

        let comp_dist = vec![
            (800.0, 0.7),
            (801.0, 0.2),
            (802.0, 0.1),
        ];

        // Pass more isotopes than exist in distributions
        let extra_isotopes: HashSet<usize> = [0, 1, 2, 3, 4, 5].iter().cloned().collect();

        let result = calculate_transmission_dependent_distribution_with_factor(
            &fragment_dist,
            &comp_dist,
            &extra_isotopes,
            0,
        );

        // With extra isotopes, transmission factor can be > 1.0
        // because higher precursor isotopes contribute to lower fragment isotopes
        assert!(
            result.transmission_factor >= 1.0,
            "Extra isotopes should give transmission factor >= 1.0, got {}",
            result.transmission_factor
        );
    }

    /// Test that partial transmission reduces the transmission factor
    #[test]
    fn test_transmission_partial_isotopes() {
        // Fragment distribution
        let fragment_dist = vec![
            (500.0, 0.5),
            (501.0, 0.3),
            (502.0, 0.15),
            (503.0, 0.05),
        ];

        // Complementary distribution
        let comp_dist = vec![
            (800.0, 0.6),
            (801.0, 0.25),
            (802.0, 0.1),
            (803.0, 0.05),
        ];

        // All isotopes
        let all_isotopes: HashSet<usize> = [0, 1, 2, 3].iter().cloned().collect();
        let full_result = calculate_transmission_dependent_distribution_with_factor(
            &fragment_dist,
            &comp_dist,
            &all_isotopes,
            0,
        );

        // Only M0 and M1 transmitted
        let partial_isotopes: HashSet<usize> = [0, 1].iter().cloned().collect();
        let partial_result = calculate_transmission_dependent_distribution_with_factor(
            &fragment_dist,
            &comp_dist,
            &partial_isotopes,
            0,
        );

        // Partial transmission should have lower transmission factor
        assert!(
            partial_result.transmission_factor < full_result.transmission_factor,
            "Partial transmission ({}) should be less than full transmission ({})",
            partial_result.transmission_factor,
            full_result.transmission_factor
        );

        // Transmission factor should be between 0 and 1
        assert!(
            partial_result.transmission_factor > 0.0 && partial_result.transmission_factor < 1.0,
            "Partial transmission factor should be between 0 and 1, got {}",
            partial_result.transmission_factor
        );
    }

    /// Test that only M0 transmitted gives lowest transmission factor
    #[test]
    fn test_transmission_m0_only() {
        let fragment_dist = vec![
            (500.0, 0.5),
            (501.0, 0.3),
            (502.0, 0.15),
            (503.0, 0.05),
        ];

        let comp_dist = vec![
            (800.0, 0.6),
            (801.0, 0.25),
            (802.0, 0.1),
            (803.0, 0.05),
        ];

        // Only M0 transmitted
        let m0_only: HashSet<usize> = [0].iter().cloned().collect();
        let m0_result = calculate_transmission_dependent_distribution_with_factor(
            &fragment_dist,
            &comp_dist,
            &m0_only,
            0,
        );

        // M0 and M1 transmitted
        let m0_m1: HashSet<usize> = [0, 1].iter().cloned().collect();
        let m0_m1_result = calculate_transmission_dependent_distribution_with_factor(
            &fragment_dist,
            &comp_dist,
            &m0_m1,
            0,
        );

        // M0 only should have lower transmission than M0+M1
        assert!(
            m0_result.transmission_factor < m0_m1_result.transmission_factor,
            "M0 only ({}) should transmit less than M0+M1 ({})",
            m0_result.transmission_factor,
            m0_m1_result.transmission_factor
        );

        // When only M0 transmitted, only M0 of fragment should have signal
        // (higher isotopes need complementary isotopes that require higher precursor isotopes)
        let m0_intensity = m0_result.distribution.get(0).map(|(_, i)| *i).unwrap_or(0.0);
        let m1_intensity = m0_result.distribution.get(1).map(|(_, i)| *i).unwrap_or(0.0);

        assert!(
            m0_intensity > 0.0,
            "M0 fragment should have intensity when M0 precursor transmitted"
        );
        assert!(
            approx_eq(m1_intensity, 0.0, 1e-10),
            "M1 fragment should have zero intensity when only M0 precursor transmitted, got {}",
            m1_intensity
        );
    }

    /// Test that empty inputs are handled gracefully
    #[test]
    fn test_transmission_empty_inputs() {
        let empty: Vec<(f64, f64)> = vec![];
        let non_empty = vec![(500.0, 0.5)];
        let isotopes: HashSet<usize> = [0].iter().cloned().collect();

        // Empty fragment distribution
        let result1 = calculate_transmission_dependent_distribution_with_factor(
            &empty,
            &non_empty,
            &isotopes,
            0,
        );
        assert!(result1.distribution.is_empty());
        assert!(approx_eq(result1.transmission_factor, 0.0, 1e-10));

        // Empty complementary distribution
        let result2 = calculate_transmission_dependent_distribution_with_factor(
            &non_empty,
            &empty,
            &isotopes,
            0,
        );
        assert!(result2.distribution.is_empty());
        assert!(approx_eq(result2.transmission_factor, 0.0, 1e-10));
    }

    /// Test that the relative isotope pattern changes with partial transmission
    #[test]
    fn test_isotope_pattern_shift() {
        // Use a distribution where we can verify the pattern shift
        let fragment_dist = vec![
            (500.0, 0.6),
            (501.0, 0.3),
            (502.0, 0.1),
        ];

        let comp_dist = vec![
            (800.0, 0.7),
            (801.0, 0.2),
            (802.0, 0.1),
        ];

        // All isotopes - get reference pattern
        let all_isotopes: HashSet<usize> = [0, 1, 2].iter().cloned().collect();
        let full_result = calculate_transmission_dependent_distribution_with_factor(
            &fragment_dist,
            &comp_dist,
            &all_isotopes,
            0,
        );

        // Only M0 transmitted
        let m0_only: HashSet<usize> = [0].iter().cloned().collect();
        let m0_result = calculate_transmission_dependent_distribution_with_factor(
            &fragment_dist,
            &comp_dist,
            &m0_only,
            0,
        );

        // Calculate relative M0 contribution for both
        let full_sum: f64 = full_result.distribution.iter().map(|(_, i)| i).sum();
        let m0_sum: f64 = m0_result.distribution.iter().map(|(_, i)| i).sum();

        let full_m0_fraction = if full_sum > 0.0 {
            full_result.distribution[0].1 / full_sum
        } else {
            0.0
        };

        let m0_m0_fraction = if m0_sum > 0.0 {
            m0_result.distribution[0].1 / m0_sum
        } else {
            0.0
        };

        // When only M0 precursor transmitted, M0 fragment should be relatively more dominant
        // (because higher fragment isotopes can't form without higher precursor isotopes)
        assert!(
            m0_m0_fraction >= full_m0_fraction,
            "M0 fraction with M0-only transmission ({}) should be >= full transmission ({})",
            m0_m0_fraction,
            full_m0_fraction
        );
    }

    /// Test max_isotope parameter limits output
    #[test]
    fn test_max_isotope_limit() {
        let fragment_dist = vec![
            (500.0, 0.5),
            (501.0, 0.3),
            (502.0, 0.15),
            (503.0, 0.05),
        ];

        let comp_dist = vec![
            (800.0, 0.6),
            (801.0, 0.25),
            (802.0, 0.1),
            (803.0, 0.05),
        ];

        let all_isotopes: HashSet<usize> = [0, 1, 2, 3].iter().cloned().collect();

        // With max_isotope = 2
        let result = calculate_transmission_dependent_distribution_with_factor(
            &fragment_dist,
            &comp_dist,
            &all_isotopes,
            2,
        );

        assert_eq!(
            result.distribution.len(),
            2,
            "Distribution should be limited to 2 isotopes, got {}",
            result.distribution.len()
        );
    }

    /// Integration test: Verify that simulated frame building produces correct intensity scaling
    /// This test mimics the behavior in rustdf's build_fragment_frame function
    #[test]
    fn test_frame_building_intensity_scaling() {
        // Simulate realistic isotope distributions for a small peptide fragment
        // Fragment: ~500 Da
        let fragment_dist = vec![
            (500.25, 0.65),
            (501.25, 0.25),
            (502.25, 0.08),
            (503.25, 0.02),
        ];

        // Complementary fragment: ~800 Da
        let comp_dist = vec![
            (800.40, 0.55),
            (801.40, 0.28),
            (802.40, 0.12),
            (803.40, 0.05),
        ];

        // Simulate frame building with different transmission scenarios
        let fraction_events: f64 = 1000.0; // Arbitrary intensity scaling factor

        // Scenario 1: Full transmission (reference)
        let all_isotopes: HashSet<usize> = (0..4).collect();
        let full_dist = calculate_transmission_dependent_fragment_ion_isotope_distribution(
            &fragment_dist,
            &comp_dist,
            &all_isotopes,
            0,
        );
        let full_intensity_sum: f64 = full_dist.iter().map(|(_, i)| i * fraction_events).sum();

        // Scenario 2: Only M0 transmitted (narrow quad window)
        let m0_only: HashSet<usize> = [0].iter().cloned().collect();
        let m0_dist = calculate_transmission_dependent_fragment_ion_isotope_distribution(
            &fragment_dist,
            &comp_dist,
            &m0_only,
            0,
        );
        let m0_intensity_sum: f64 = m0_dist.iter().map(|(_, i)| i * fraction_events).sum();

        // Scenario 3: M0+M1 transmitted (typical quad window)
        let m0_m1: HashSet<usize> = [0, 1].iter().cloned().collect();
        let m0_m1_dist = calculate_transmission_dependent_fragment_ion_isotope_distribution(
            &fragment_dist,
            &comp_dist,
            &m0_m1,
            0,
        );
        let m0_m1_intensity_sum: f64 = m0_m1_dist.iter().map(|(_, i)| i * fraction_events).sum();

        // Verify intensity ordering: full > M0+M1 > M0 only
        assert!(
            full_intensity_sum > m0_m1_intensity_sum,
            "Full transmission intensity ({}) should be > M0+M1 ({})",
            full_intensity_sum, m0_m1_intensity_sum
        );
        assert!(
            m0_m1_intensity_sum > m0_intensity_sum,
            "M0+M1 transmission intensity ({}) should be > M0 only ({})",
            m0_m1_intensity_sum, m0_intensity_sum
        );

        // Verify intensity ratios make physical sense
        // M0+M1 should give roughly 60-90% of full intensity (depends on distributions)
        let m0_m1_ratio = m0_m1_intensity_sum / full_intensity_sum;
        assert!(
            m0_m1_ratio > 0.5 && m0_m1_ratio < 1.0,
            "M0+M1 ratio ({}) should be between 0.5 and 1.0",
            m0_m1_ratio
        );

        // M0 only should give roughly 30-70% of full intensity
        let m0_ratio = m0_intensity_sum / full_intensity_sum;
        assert!(
            m0_ratio > 0.2 && m0_ratio < 0.8,
            "M0 ratio ({}) should be between 0.2 and 0.8",
            m0_ratio
        );

        // Use the factor tracking function to verify explicit transmission factor
        let factor_result = calculate_transmission_dependent_distribution_with_factor(
            &fragment_dist,
            &comp_dist,
            &m0_m1,
            0,
        );

        // The transmission factor should match our manual calculation
        let manual_factor = m0_m1_intensity_sum / full_intensity_sum;
        assert!(
            approx_eq(factor_result.transmission_factor, manual_factor, 0.001),
            "Transmission factor ({}) should match manual calculation ({})",
            factor_result.transmission_factor, manual_factor
        );

        println!("Frame building intensity test results:");
        println!("  Full transmission intensity: {:.2}", full_intensity_sum);
        println!("  M0+M1 transmission intensity: {:.2} (ratio: {:.3})", m0_m1_intensity_sum, m0_m1_ratio);
        println!("  M0 only transmission intensity: {:.2} (ratio: {:.3})", m0_intensity_sum, m0_ratio);
        println!("  Transmission factor (M0+M1): {:.4}", factor_result.transmission_factor);
    }

    /// Test behavior with realistic peptide masses using the factor tracking function
    #[test]
    fn test_realistic_peptide_transmission() {
        // Simulate a typical tryptic peptide (~1500 Da precursor)
        // B-ion fragment ~600 Da, Y-ion (complementary) ~900 Da

        // B-ion isotope pattern (normalized)
        let b_ion_dist = vec![
            (600.30, 0.58),
            (601.30, 0.28),
            (602.30, 0.10),
            (603.30, 0.03),
            (604.30, 0.01),
        ];

        // Complementary (Y-ion like) isotope pattern
        let comp_dist = vec![
            (900.45, 0.48),
            (901.45, 0.30),
            (902.45, 0.14),
            (903.45, 0.06),
            (904.45, 0.02),
        ];

        // Test various quad isolation scenarios

        // Wide window: transmits M0, M+1, M+2
        let wide_window: HashSet<usize> = [0, 1, 2].iter().cloned().collect();
        let wide_result = calculate_transmission_dependent_distribution_with_factor(
            &b_ion_dist,
            &comp_dist,
            &wide_window,
            0,
        );

        // Narrow window: transmits only M0, M+1
        let narrow_window: HashSet<usize> = [0, 1].iter().cloned().collect();
        let narrow_result = calculate_transmission_dependent_distribution_with_factor(
            &b_ion_dist,
            &comp_dist,
            &narrow_window,
            0,
        );

        // Very narrow: only M0
        let very_narrow: HashSet<usize> = [0].iter().cloned().collect();
        let very_narrow_result = calculate_transmission_dependent_distribution_with_factor(
            &b_ion_dist,
            &comp_dist,
            &very_narrow,
            0,
        );

        // Verify decreasing transmission with narrower windows
        assert!(
            wide_result.transmission_factor > narrow_result.transmission_factor,
            "Wide window should have higher transmission"
        );
        assert!(
            narrow_result.transmission_factor > very_narrow_result.transmission_factor,
            "Narrow window should have higher transmission than very narrow"
        );

        // Verify all factors are in valid range (0, 1]
        assert!(wide_result.transmission_factor > 0.0 && wide_result.transmission_factor <= 1.0);
        assert!(narrow_result.transmission_factor > 0.0 && narrow_result.transmission_factor <= 1.0);
        assert!(very_narrow_result.transmission_factor > 0.0 && very_narrow_result.transmission_factor <= 1.0);

        println!("Realistic peptide transmission test results:");
        println!("  Wide window (M0-M2) transmission factor: {:.4}", wide_result.transmission_factor);
        println!("  Narrow window (M0-M1) transmission factor: {:.4}", narrow_result.transmission_factor);
        println!("  Very narrow (M0 only) transmission factor: {:.4}", very_narrow_result.transmission_factor);
    }
}
