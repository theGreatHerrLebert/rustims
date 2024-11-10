use mscore::data::peptide::{FragmentType, PeptideSequence};

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde_json::to_string;

/// helper function to reshape the flat prosit predicted intensity array into a 3D array where:
/// 1st dimension: 29 rows for every potential ion since prosit allows precursor sequences up to 30 amino acids
/// 2nd dimension: 2 columns for B and Y ions
/// 3rd dimension: 3 channels for charge 1, 2, and 3
///
/// # Arguments
///
/// * `array` - A vector of f64 representing the flat prosit array
///
/// # Returns
///
/// * A 3D vector of f64 representing the reshaped prosit array
///
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

/// helper function to convert a peptide ion to all possible ions and serialize the result to a json string
///
/// # Arguments
///
/// * `sequence` - A string representing the peptide sequence
/// * `charge` - An i32 representing the charge
/// * `intensity_pred_flat` - A vector of f64 representing the flat prosit predicted intensity array
/// * `normalize` - A bool indicating whether to normalize the intensity values
/// * `half_charge_one` - A bool indicating whether to use half charge one
///
/// # Returns
///
/// * A json string representing the peptide ions ready to pe put into a database
///
pub fn sequence_to_all_ions(
    sequence: &str,
    charge: i32,
    intensity_pred_flat: &Vec<f64>, // Assuming this is the reshaped intensity predictions array
    normalize: bool,
    half_charge_one: bool,
    peptide_id: Option<i32>,
) -> String {

    let peptide_sequence = PeptideSequence::new(sequence.to_string(), peptide_id);
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
    peptide_ids: Vec<Option<i32>>
) -> Vec<String> {
    let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();

    let result = thread_pool.install(|| {
        sequences.par_iter().zip(charges.par_iter()).zip(intensities_pred_flat.par_iter()).zip(peptide_ids.par_iter())
            .map(|(((seq, charge), intensities), peptide_id)| sequence_to_all_ions(seq, *charge, intensities, normalize, half_charge_one, *peptide_id))
            .collect()
    });

    result
}
