use std::collections::HashMap;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use statrs::distribution::{Binomial, Discrete};
use crate::chemistry::elements::{atoms_isotopic_weights, isotopic_abundance};

fn convolve(dist_a: &Vec<(f64, f64)>, dist_b: &Vec<(f64, f64)>, mass_tolerance: f64, abundance_threshold: f64, max_results: usize) -> Vec<(f64, f64)> {
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

fn convolve_pow(dist: &Vec<(f64, f64)>, n: i32) -> Vec<(f64, f64)> {
    // Base cases
    if n == 0 {
        return vec![(0.0, 1.0)]; // Return the delta distribution
    }
    if n == 1 {
        return dist.clone();
    }

    // Recursive case
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
    final_distribution.into_iter().map(|(mass, abundance)| (mass, abundance / total_abundance)).collect()
}



pub fn get_num_protonizable_sites(sequence: &str) -> usize {
    let mut sites = 1; // n-terminus
    for s in sequence.chars() {
        match s {
            'H' | 'R' | 'K' => sites += 1,
            _ => {}
        }
    }
    sites
}

pub fn simulate_charge_state_for_sequence(sequence: &str, max_charge: Option<usize>, charged_probability: Option<f64>) -> Vec<f64> {
    let charged_prob = charged_probability.unwrap_or(0.5);
    let max_charge = max_charge.unwrap_or(5);
    let num_protonizable_sites = get_num_protonizable_sites(sequence);
    let mut charge_state_probs = vec![0.0; max_charge];

    for charge in 0..max_charge {
        let binom = Binomial::new(charged_prob, num_protonizable_sites as u64).unwrap();
        let prob = binom.pmf(charge as u64);

        charge_state_probs[charge] = prob;
    }

    charge_state_probs
}

pub fn simulate_charge_states_for_sequences(sequences: Vec<&str>,  num_threads: usize, max_charge: Option<usize>, charged_probability: Option<f64>) -> Vec<Vec<f64>> {
    let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    pool.install(|| {
        sequences.par_iter()
            .map(|sequence| simulate_charge_state_for_sequence(sequence, max_charge, charged_probability))
            .collect()
    })
}