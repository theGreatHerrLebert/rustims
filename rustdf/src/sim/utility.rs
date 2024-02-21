use mscore::chemistry::aa_sequence::calculate_b_y_ion_series;
use regex::Regex;
use mscore::chemistry::unimod::unimod_modifications_mz;

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::sim::containers::{FragmentIon, FragmentIonSeries};
use serde_json::to_string;

pub fn find_unimod_patterns(input_string: &str) -> (String, Vec<f64>) {
    let results = extract_unimod_patterns(input_string);
    let stripped_sequence = remove_unimod_annotation(input_string);
    let index_list = generate_index_list(&results, input_string);
    let mods = calculate_modifications(&index_list, &stripped_sequence);
    (stripped_sequence, mods)
}

fn remove_unimod_annotation(sequence: &str) -> String {
    let pattern = Regex::new(r"\[UNIMOD:\d+\]").unwrap();
    pattern.replace_all(sequence, "").to_string()
}

fn extract_unimod_patterns(input_string: &str) -> Vec<(usize, usize, String)> {
    let pattern = Regex::new(r"\[UNIMOD:\d+\]").unwrap();
    pattern.find_iter(input_string)
        .map(|mat| (mat.start(), mat.end(), mat.as_str().to_string()))
        .collect()
}

fn generate_index_list(results: &[(usize, usize, String)], sequence: &str) -> Vec<(usize, String)> {
    let mut index_list = Vec::new();
    let mut chars_removed_counter = 0;

    for (start, end, _) in results {
        let num_chars_removed = end - start;
        let mod_str = &sequence[*start..*end];

        let later_aa_index = if *start != 0 {
            start - 1 - chars_removed_counter
        } else {
            0
        };

        index_list.push((later_aa_index, mod_str.to_string()));
        chars_removed_counter += num_chars_removed;
    }

    index_list
}

fn calculate_modifications(index_list: &[(usize, String)], stripped_sequence: &str) -> Vec<f64> {
    let mut mods = vec![0.0; stripped_sequence.len()];
    for (index, mod_str) in index_list {
        if let Some(mass) = unimod_modifications_mz().get(mod_str.as_str()) {
            mods[*index] += mass;
        }
    }
    mods
}

pub fn find_unimod_patterns_par(sequences: Vec<&str>, num_threads: usize) -> Vec<(String, Vec<f64>)> {
    let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    let result = thread_pool.install(|| {
        sequences.par_iter().map(|seq| find_unimod_patterns(seq)).collect()
    });
    result
}

pub fn generate_fragments_json(
    charge: i32,
    b_ions: Vec<(f64, String)>, // Assuming a tuple of (mz, ion_type)
    y_ions: Vec<(f64, String)>,
    intensity_b:Vec<f64>, // Optional intensity vectors
    intensity_y: Vec<f64>,
    num_decimals: u32,
) -> String {
    let mut peptide_ion_data = FragmentIonSeries {
        charge,
        b_ions: Vec::new(),
        y_ions: Vec::new(),
    };

    for (i, (mz, ion_type)) in b_ions.into_iter().enumerate() {
        let intensity = intensity_b[i].round_decimals(num_decimals);
        peptide_ion_data.b_ions.push(FragmentIon {
            mz: mz.round_decimals(num_decimals),
            kind: ion_type.trim_end_matches(char::is_numeric).to_string(),
            intensity,
        });
    }

    for (i, (mz, ion_type)) in y_ions.into_iter().enumerate() {
        let intensity = intensity_y[i].round_decimals(num_decimals);

        peptide_ion_data.y_ions.push(FragmentIon {
            mz: mz.round_decimals(num_decimals),
            kind: ion_type.trim_end_matches(char::is_numeric).to_string(),
            intensity,
        });
    }

    to_string(&peptide_ion_data).unwrap_or_default()
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
    charge: usize,
    intensity_pred_flat: &Vec<f64>, // Assuming this is the reshaped intensity predictions array
    normalize: bool,
    half_charge_one: bool,
) -> String {
    let (stripped_sequence, mods) = find_unimod_patterns(sequence);
    let seq_len = stripped_sequence.len() - 1; // Adjust for indexing

    let max_charge = std::cmp::min(charge, 3).max(2); // Ensure at least 2 for loop range
    let mut sum_intensity = if normalize { 0.0 } else { 1.0 };
    let intensity_pred = reshape_prosit_array(intensity_pred_flat.clone());

    if normalize {
        for z in 1..=max_charge {
            let c = z as i32;
            let intensity_y: Vec<f64> = intensity_pred[..seq_len].iter().map(|x| x[0][c - 1]).collect();
            let intensity_b: Vec<f64> = intensity_pred[..seq_len].iter().map(|x| x[1][c - 1]).collect();

            sum_intensity += intensity_b.iter().sum::<f64>() + intensity_y.iter().sum::<f64>();
        }
    }

    let mut r_list = Vec::new();

    for z in 1..=max_charge {
        let c = z as i32;
        let (b, y) = calculate_b_y_ion_series(&stripped_sequence, mods.clone(), Some(c));

        let intensity_b: Vec<f64> = intensity_pred[..seq_len].iter().map(|x| x[1][c - 1]).collect();
        let intensity_y: Vec<f64> = intensity_pred[..seq_len].iter().map(|x| x[0][c - 1]).rev().collect(); // Reverse for y

        let adjusted_sum_intensity = if max_charge == 1 && half_charge_one { sum_intensity * 2.0 } else { sum_intensity };

        let json_str = generate_fragments_json(
            c,
            b.iter().map(|t| (t.0, t.1.clone())).collect::<Vec<(f64, String)>>(),
            y.iter().map(|t| (t.0, t.1.clone())).collect::<Vec<(f64, String)>>(),
            intensity_b.iter().map(|&i| i / adjusted_sum_intensity).collect::<Vec<f64>>(),
            intensity_y.iter().map(|&i| i / adjusted_sum_intensity).collect::<Vec<f64>>(),
            4,
        );

        r_list.push(json_str);
    }

    to_string(&r_list).unwrap_or_else(|_| "[]".to_string())
}
