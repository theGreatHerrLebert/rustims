use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use statrs::distribution::{Binomial, Discrete};

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
/// assert_eq!(charge_state_probs, vec![0.25, 0.5, 0.25, 0.0, 0.0]);
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
/// assert_eq!(charge_state_probs, vec![vec![0.25, 0.5, 0.25, 0.0, 0.0], vec![0.25, 0.5, 0.25, 0.0, 0.0], vec![0.25, 0.5, 0.25, 0.0, 0.0]]);
/// ```
pub fn simulate_charge_states_for_sequences(sequences: Vec<&str>,  num_threads: usize, max_charge: Option<usize>, charged_probability: Option<f64>) -> Vec<Vec<f64>> {
    let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    pool.install(|| {
        sequences.par_iter()
            .map(|sequence| simulate_charge_state_for_sequence(sequence, max_charge, charged_probability))
            .collect()
    })
}