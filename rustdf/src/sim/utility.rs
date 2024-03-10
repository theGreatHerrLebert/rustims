use mscore::data::peptide::{FragmentType, PeptideSequence};

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde_json::to_string;

trait RoundDecimals {
    fn round_decimals(&self, num_decimals: u32) -> f64;
}

impl RoundDecimals for f64 {
    fn round_decimals(&self, num_decimals: u32) -> f64 {
        let multiplier = 10f64.powi(num_decimals as i32);
        (self * multiplier).round() / multiplier
    }
}

pub fn reshape_prosit_array(array: Vec<f64>) -> Vec<Vec<Vec<f64>>> {
    let mut array_return: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; 3]; 2]; 29];
    let mut ptr = 0;

    for c in 0..3 {
        for row in 0..29 {
            // Fill in the Y ion values
            array_return[row][0][c] = array[ptr];
            ptr += 1;
        }
        for row in 0..29 {
            // Fill in the B ion values
            array_return[row][1][c] = array[ptr];
            ptr += 1;
        }
    }

    array_return
}

pub fn sequence_to_all_ions(
    sequence: &str,
    charge: i32,
    intensity_pred_flat: &Vec<f64>, // Assuming this is the reshaped intensity predictions array
    normalize: bool,
    half_charge_one: bool,
) -> String {

    let peptide_sequence = PeptideSequence::new(sequence.to_string());
    let fragments = peptide_sequence.associate_with_predicted_intensities(
        charge,
        FragmentType::B,
        intensity_pred_flat.clone(),
        normalize,
        half_charge_one
    );
    to_string(&fragments).unwrap()
}

pub fn sequence_to_all_ions_par(
    sequences: Vec<&str>,
    charges: Vec<i32>,
    intensities_pred_flat: Vec<Vec<f64>>,
    normalize: bool,
    half_charge_one: bool,
    num_threads: usize,
) -> Vec<String> {
    let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    let result = thread_pool.install(|| {
        sequences.par_iter().zip(charges.par_iter()).zip(intensities_pred_flat.par_iter())
            .map(|((seq, charge), intensities)| sequence_to_all_ions(seq, *charge, intensities, normalize, half_charge_one))
            .collect()
    });
    result
}
