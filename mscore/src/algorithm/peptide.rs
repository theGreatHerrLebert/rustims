use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use statrs::distribution::{Binomial, Discrete};



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