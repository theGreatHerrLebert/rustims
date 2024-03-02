use mscore::chemistry::utility::find_unimod_patterns;
use mscore::data::peptide::{FragmentType, PeptideSequence};

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::sim::containers::{FragmentIon, FragmentIonSeries};
use serde_json::to_string;

pub fn generate_fragments(
    charge: i32,
    b_ions: Vec<(f64, String, String)>, // Assuming a tuple of (mz, ion_type)
    y_ions: Vec<(f64, String, String)>,
    intensity_b:Vec<f64>, // Optional intensity vectors
    intensity_y: Vec<f64>,
    num_decimals: u32,
) -> FragmentIonSeries {
    let mut peptide_ion_data = FragmentIonSeries {
        charge,
        b_ions: Vec::new(),
        y_ions: Vec::new(),
    };

    for (i, (mz, ion_type, sequence)) in b_ions.into_iter().enumerate() {
        let intensity = intensity_b[i].round_decimals(num_decimals);
        peptide_ion_data.b_ions.push(FragmentIon {
            mz: mz.round_decimals(num_decimals),
            kind: ion_type.trim_end_matches(char::is_numeric).to_string(),
            sequence: sequence.to_string(),
            intensity,
        });
    }

    for (i, (mz, ion_type, sequence)) in y_ions.into_iter().enumerate() {
        let intensity = intensity_y[i].round_decimals(num_decimals);

        peptide_ion_data.y_ions.push(FragmentIon {
            mz: mz.round_decimals(num_decimals),
            kind: ion_type.trim_end_matches(char::is_numeric).to_string(),
            sequence: sequence.to_string(),
            intensity,
        });
    }

    peptide_ion_data
}

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
    let (stripped_sequence, _) = find_unimod_patterns(sequence);
    let seq_len = stripped_sequence.len() - 1; // Adjust for indexing

    let max_charge = std::cmp::min(charge, 3).max(2); // Ensure at least 2 for loop range
    let mut sum_intensity = if normalize { 0.0 } else { 1.0 };
    let intensity_pred = reshape_prosit_array(intensity_pred_flat.clone());

    if normalize {
        for z in 1..=max_charge {
            let intensity_y: Vec<f64> = intensity_pred[..seq_len].iter().map(|x| x[0][z as usize - 1]).filter(|&x| x > 0.0).collect();
            let intensity_b: Vec<f64> = intensity_pred[..seq_len].iter().map(|x| x[1][z as usize - 1]).filter(|&x| x > 0.0).collect();

            sum_intensity += intensity_b.iter().sum::<f64>() + intensity_y.iter().sum::<f64>();
        }
    }

    let mut r_list = Vec::new();

    for z in 1..=max_charge {

        let (b, y) = peptide_sequence.calculate_product_ion_series(charge, FragmentType::B);

        let b_mz_values = b.iter().map(|prod_ion| prod_ion.mz()).collect::<Vec<f64>>();
        let mut b_ion_types: Vec<String> = Vec::with_capacity(b.len());

        for (i, prod_ion) in b.iter().enumerate() {
            let ion_type = match prod_ion.kind {
                FragmentType::A => "a".to_string(),
                FragmentType::B => "b".to_string(),
                FragmentType::C => "c".to_string(),
                FragmentType::X => "x".to_string(),
                FragmentType::Y => "y".to_string(),
                FragmentType::Z => "z".to_string(),
            };
            b_ion_types.push(format!("{}{}", ion_type, i + 1));
        }

        let b_sequences: Vec<String> = b.iter().map(|prod_ion| prod_ion.ion.sequence.sequence.clone()).collect();

        let y_mz_values = y.iter().map(|prod_ion| prod_ion.mz()).collect::<Vec<f64>>();
        let mut y_ion_types: Vec<String> = Vec::with_capacity(y.len());

        for (i, prod_ion) in y.iter().enumerate() {
            let ion_type = match prod_ion.kind {
                FragmentType::A => "a".to_string(),
                FragmentType::B => "b".to_string(),
                FragmentType::C => "c".to_string(),
                FragmentType::X => "x".to_string(),
                FragmentType::Y => "y".to_string(),
                FragmentType::Z => "z".to_string(),
            };
            y_ion_types.push(format!("{}{}", ion_type, i + 1));
        }

        let y_sequences: Vec<String> = y.iter().map(|prod_ion| prod_ion.ion.sequence.sequence.clone()).collect();


        let intensity_b: Vec<f64> = intensity_pred[..seq_len].iter().map(|x| x[1][z as usize - 1]).collect();
        // TODO: check if this still needs to be reversed
        let intensity_y: Vec<f64> = intensity_pred[..seq_len].iter().map(|x| x[0][z as usize - 1]).rev().collect(); // Reverse for y

        let adjusted_sum_intensity = if max_charge == 1 && half_charge_one { sum_intensity * 2.0 } else { sum_intensity };

        let fragments = generate_fragments(
            z,
               b_mz_values.iter().zip(b_ion_types.iter()).zip(b_sequences.iter()).map(|((mz, ion_type), sequence)| (*mz, ion_type.clone(), sequence.clone())).collect::<Vec<(f64, String, String)>>(),
                y_mz_values.iter().zip(y_ion_types.iter()).zip(y_sequences.iter()).map(|((mz, ion_type), sequence)| (*mz, ion_type.clone(), sequence.clone())).collect::<Vec<(f64, String, String)>>(),
            intensity_b.iter().map(|&i| i / adjusted_sum_intensity).collect::<Vec<f64>>(),
            intensity_y.iter().map(|&i| i / adjusted_sum_intensity).collect::<Vec<f64>>(),
            4,
        );

        r_list.push(fragments);
    }

    to_string(&r_list).unwrap_or_else(|_| "[]".to_string())
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
