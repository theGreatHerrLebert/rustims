use std::collections::{BTreeMap, HashMap};
use crate::chemistry::element::{atoms_isotopic_weights, isotopic_abundance};

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
/// use rustms::algorithm::isotope::convolve;
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
/// use rustms::algorithm::isotope::convolve_pow;
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
/// use rustms::algorithm::isotope::generate_isotope_distribution;
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