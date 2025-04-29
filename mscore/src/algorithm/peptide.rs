use crate::chemistry::amino_acid::{amino_acid_composition, amino_acid_masses};
use crate::chemistry::constants::{MASS_CO, MASS_NH3, MASS_PROTON, MASS_WATER};
use crate::chemistry::formulas::calculate_mz;
use crate::chemistry::unimod::{
    modification_atomic_composition, unimod_modifications_mass_numerical,
};
use crate::chemistry::utility::{find_unimod_patterns, unimod_sequence_to_tokens};
use crate::data::peptide::{FragmentType, PeptideProductIon, PeptideSequence};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use regex::Regex;
use statrs::distribution::{Binomial, Discrete};
use std::collections::HashMap;

/// calculate the monoisotopic mass of a peptide sequence
///
/// Arguments:
///
/// * `sequence` - peptide sequence
///
/// Returns:
///
/// * `mass` - monoisotopic mass of the peptide
///
/// # Examples
///
/// ```
/// use mscore::algorithm::peptide::calculate_peptide_mono_isotopic_mass;
/// use mscore::data::peptide::PeptideSequence;
///
/// let peptide_sequence = PeptideSequence::new("PEPTIDEH".to_string(), Some(1));
/// let mass = calculate_peptide_mono_isotopic_mass(&peptide_sequence);
/// let mass_quantized = (mass * 1e6).round() as i32;
/// assert_eq!(mass_quantized, 936418877);
/// ```
pub fn calculate_peptide_mono_isotopic_mass(peptide_sequence: &PeptideSequence) -> f64 {
    let amino_acid_masses = amino_acid_masses();
    let modifications_mz_numerical = unimod_modifications_mass_numerical();
    let pattern = Regex::new(r"\[UNIMOD:(\d+)]").unwrap();

    let sequence = peptide_sequence.sequence.as_str();

    // Find all occurrences of the pattern
    let modifications: Vec<u32> = pattern
        .find_iter(sequence)
        .filter_map(|mat| mat.as_str()[8..mat.as_str().len() - 1].parse().ok())
        .collect();

    // Remove the modifications from the sequence
    let sequence = pattern.replace_all(sequence, "");

    // Count occurrences of each amino acid
    let mut aa_counts = HashMap::new();
    for char in sequence.chars() {
        *aa_counts.entry(char).or_insert(0) += 1;
    }

    // Mass of amino acids and modifications
    let mass_sequence: f64 = aa_counts
        .iter()
        .map(|(aa, &count)| {
            amino_acid_masses.get(&aa.to_string()[..]).unwrap_or(&0.0) * count as f64
        })
        .sum();
    let mass_modifications: f64 = modifications
        .iter()
        .map(|&mod_id| modifications_mz_numerical.get(&mod_id).unwrap_or(&0.0))
        .sum();

    mass_sequence + mass_modifications + MASS_WATER
}

/// calculate the monoisotopic mass of a peptide product ion for a given fragment type
///
/// Arguments:
///
/// * `sequence` - peptide sequence
/// * `kind` - fragment type
///
/// Returns:
///
/// * `mass` - monoisotopic mass of the peptide
///
/// # Examples
/// ```
/// use mscore::algorithm::peptide::calculate_peptide_product_ion_mono_isotopic_mass;
/// use mscore::data::peptide::FragmentType;
/// let sequence = "PEPTIDEH";
/// let mass = calculate_peptide_product_ion_mono_isotopic_mass(sequence, FragmentType::Y);
/// assert_eq!(mass, 936.4188766862999);
/// ```
pub fn calculate_peptide_product_ion_mono_isotopic_mass(sequence: &str, kind: FragmentType) -> f64 {
    let (sequence, modifications) = find_unimod_patterns(sequence);

    // Return mz of empty sequence
    if sequence.is_empty() {
        return 0.0;
    }

    let amino_acid_masses = amino_acid_masses();

    // Add up raw amino acid masses and potential modifications
    let mass_sequence: f64 = sequence
        .chars()
        .map(|aa| amino_acid_masses.get(&aa.to_string()[..]).unwrap_or(&0.0))
        .sum();

    let mass_modifications: f64 = modifications.iter().sum();

    // Calculate total mass
    let mass = mass_sequence + mass_modifications + MASS_WATER;

    let mass = match kind {
        FragmentType::A => mass - MASS_CO - MASS_WATER,
        FragmentType::B => mass - MASS_WATER,
        FragmentType::C => mass + MASS_NH3 - MASS_WATER,
        FragmentType::X => mass + MASS_CO - 2.0 * MASS_PROTON,
        FragmentType::Y => mass,
        FragmentType::Z => mass - MASS_NH3,
    };

    mass
}

/// calculate the monoisotopic m/z of a peptide product ion for a given fragment type and charge
///
/// Arguments:
///
/// * `sequence` - peptide sequence
/// * `kind` - fragment type
/// * `charge` - charge
///
/// Returns:
///
/// * `mz` - monoisotopic mass of the peptide
///
/// # Examples
/// ```
/// use mscore::algorithm::peptide::calculate_product_ion_mz;
/// use mscore::chemistry::constants::MASS_PROTON;
/// use mscore::data::peptide::FragmentType;
/// let sequence = "PEPTIDEH";
/// let mz = calculate_product_ion_mz(sequence, FragmentType::Y, Some(1));
/// assert_eq!(mz, 936.4188766862999 + MASS_PROTON);
/// ```
pub fn calculate_product_ion_mz(sequence: &str, kind: FragmentType, charge: Option<i32>) -> f64 {
    let mass = calculate_peptide_product_ion_mono_isotopic_mass(sequence, kind);
    calculate_mz(mass, charge.unwrap_or(1))
}

/// get a count dictionary of the amino acid composition of a peptide sequence
///
/// Arguments:
///
/// * `sequence` - peptide sequence
///
/// Returns:
///
/// * `composition` - a dictionary of amino acid composition
///
/// # Examples
///
/// ```
/// use mscore::algorithm::peptide::calculate_amino_acid_composition;
///
/// let sequence = "PEPTIDEH";
/// let composition = calculate_amino_acid_composition(sequence);
/// assert_eq!(composition.get("P"), Some(&2));
/// assert_eq!(composition.get("E"), Some(&2));
/// assert_eq!(composition.get("T"), Some(&1));
/// assert_eq!(composition.get("I"), Some(&1));
/// assert_eq!(composition.get("D"), Some(&1));
/// assert_eq!(composition.get("H"), Some(&1));
/// ```
pub fn calculate_amino_acid_composition(sequence: &str) -> HashMap<String, i32> {
    let mut composition = HashMap::new();
    for char in sequence.chars() {
        *composition.entry(char.to_string()).or_insert(0) += 1;
    }
    composition
}

/// calculate the atomic composition of a peptide sequence
pub fn peptide_sequence_to_atomic_composition(
    peptide_sequence: &PeptideSequence,
) -> HashMap<&'static str, i32> {
    let token_sequence = unimod_sequence_to_tokens(peptide_sequence.sequence.as_str(), false);
    let mut collection: HashMap<&'static str, i32> = HashMap::new();

    // Assuming amino_acid_composition and modification_composition return appropriate mappings...
    let aa_compositions = amino_acid_composition();
    let mod_compositions = modification_atomic_composition();

    // No need for conversion to HashMap<String, ...> as long as you're directly accessing
    // the HashMap provided by modification_composition() if it uses String keys.
    for token in token_sequence {
        if token.len() == 1 {
            let char = token.chars().next().unwrap();
            if let Some(composition) = aa_compositions.get(&char) {
                for (key, value) in composition.iter() {
                    *collection.entry(key).or_insert(0) += *value;
                }
            }
        } else {
            // Directly use &token without .as_str() conversion
            if let Some(composition) = mod_compositions.get(&token) {
                for (key, value) in composition.iter() {
                    *collection.entry(key).or_insert(0) += *value;
                }
            }
        }
    }

    // Add water
    *collection.entry("H").or_insert(0) += 2; //
    *collection.entry("O").or_insert(0) += 1; //

    collection
}

/// calculate the atomic composition of a product ion
///
/// Arguments:
///
/// * `product_ion` - a PeptideProductIon instance
///
/// Returns:
///
/// * `Vec<(&str, i32)>` - a vector of tuples representing the atomic composition of the product ion
pub fn atomic_product_ion_composition(product_ion: &PeptideProductIon) -> Vec<(&str, i32)> {
    let mut composition = peptide_sequence_to_atomic_composition(&product_ion.ion.sequence);

    match product_ion.kind {
        FragmentType::A => {
            // A: peptide_mass - CO - Water
            *composition.entry("H").or_insert(0) -= 2;
            *composition.entry("O").or_insert(0) -= 2;
            *composition.entry("C").or_insert(0) -= 1;
        }
        FragmentType::B => {
            // B: peptide_mass - Water
            *composition.entry("H").or_insert(0) -= 2;
            *composition.entry("O").or_insert(0) -= 1;
        }
        FragmentType::C => {
            // C: peptide_mass + NH3 - Water
            *composition.entry("H").or_insert(0) += 1;
            *composition.entry("N").or_insert(0) += 1;
            *composition.entry("O").or_insert(0) -= 1;
        }
        FragmentType::X => {
            // X: peptide_mass + CO
            *composition.entry("C").or_insert(0) += 1; // Add 1 for CO
            *composition.entry("O").or_insert(0) += 1; // Add 1 for CO
            *composition.entry("H").or_insert(0) -= 2; // Subtract 2 for 2 protons
        }
        FragmentType::Y => (),
        FragmentType::Z => {
            *composition.entry("H").or_insert(0) -= 3;
            *composition.entry("N").or_insert(0) -= 1;
        }
    }

    composition.iter().map(|(k, v)| (*k, *v)).collect()
}

/// calculate the atomic composition of a peptide product ion series
/// Arguments:
///
/// * `product_ions` - a vector of PeptideProductIon instances
/// * `num_threads` - an usize representing the number of threads to use
/// Returns:
///
/// * `Vec<Vec<(String, i32)>>` - a vector of vectors of tuples representing the atomic composition of each product ion
///
pub fn fragments_to_composition(
    product_ions: Vec<PeptideProductIon>,
    num_threads: usize,
) -> Vec<Vec<(String, i32)>> {
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();
    let result = thread_pool.install(|| {
        product_ions
            .par_iter()
            .map(|ion| atomic_product_ion_composition(ion))
            .map(|composition| {
                composition
                    .iter()
                    .map(|(k, v)| (k.to_string(), *v))
                    .collect()
            })
            .collect()
    });
    result
}

/// count the number of protonizable sites in a peptide sequence
///
/// # Arguments
///
/// * `sequence` - a string representing the peptide sequence
///
/// # Returns
///
/// * `usize` - the number of protonizable sites in the peptide sequence
///
/// # Example
///
/// ```
/// use mscore::algorithm::peptide::get_num_protonizable_sites;
///
/// let sequence = "PEPTIDEH";
/// let num_protonizable_sites = get_num_protonizable_sites(sequence);
/// assert_eq!(num_protonizable_sites, 2);
/// ```
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

/// simulate the charge state distribution for a peptide sequence
///
/// # Arguments
///
/// * `sequence` - a string representing the peptide sequence
/// * `max_charge` - an optional usize representing the maximum charge state to simulate
/// * `charged_probability` - an optional f64 representing the probability of a site being charged
///
/// # Returns
///
/// * `Vec<f64>` - a vector of f64 representing the probability of each charge state
///
/// # Example
///
/// ```
/// use mscore::algorithm::peptide::simulate_charge_state_for_sequence;
///
/// let sequence = "PEPTIDEH";
/// let charge_state_probs = simulate_charge_state_for_sequence(sequence, None, None);
/// assert_eq!(charge_state_probs, vec![0.03999999999999999, 0.32, 0.64, 0.0, 0.0]);
pub fn simulate_charge_state_for_sequence(
    sequence: &str,
    max_charge: Option<usize>,
    charged_probability: Option<f64>,
) -> Vec<f64> {
    let charged_prob = charged_probability.unwrap_or(0.8);
    let max_charge = max_charge.unwrap_or(4)+1;
    let num_protonizable_sites = get_num_protonizable_sites(sequence);
    let mut charge_state_probs = vec![0.0; max_charge];
    let binom = Binomial::new(charged_prob, num_protonizable_sites as u64).unwrap();
    
    for charge in 0..max_charge {
        charge_state_probs[charge] = binom.pmf(charge as u64);
    }
    charge_state_probs
}

/// simulate the charge state distribution for a list of peptide sequences
///
/// # Arguments
///
/// * `sequences` - a vector of strings representing the peptide sequences
/// * `num_threads` - an usize representing the number of threads to use
/// * `max_charge` - an optional usize representing the maximum charge state to simulate
/// * `charged_probability` - an optional f64 representing the probability of a site being charged
///
/// # Returns
///
/// * `Vec<Vec<f64>>` - a vector of vectors of f64 representing the probability of each charge state for each sequence
///
/// # Example
///
/// ```
/// use mscore::algorithm::peptide::simulate_charge_states_for_sequences;
///
/// let sequences = vec!["PEPTIDEH", "PEPTIDEH", "PEPTIDEH"];
/// let charge_state_probs = simulate_charge_states_for_sequences(sequences, 4, None, None);
/// assert_eq!(charge_state_probs, vec![vec![0.03999999999999999, 0.32, 0.64, 0.0, 0.0], vec![0.03999999999999999, 0.32, 0.64, 0.0, 0.0], vec![0.03999999999999999, 0.32, 0.64, 0.0, 0.0]]);
/// ```
pub fn simulate_charge_states_for_sequences(
    sequences: Vec<&str>,
    num_threads: usize,
    max_charge: Option<usize>,
    charged_probability: Option<f64>,
) -> Vec<Vec<f64>> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();
    pool.install(|| {
        sequences
            .par_iter()
            .map(|sequence| {
                simulate_charge_state_for_sequence(sequence, max_charge, charged_probability)
            })
            .collect()
    })
}
