use std::collections::HashMap;
use regex::Regex;
use crate::chemistry::constants::{MASS_WATER, MASS_PROTON};
use crate::chemistry::amino_acids::amino_acid_masses;
use crate::chemistry::unimod::unimod_modifications_mz_numerical;

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
/// use mscore::chemistry::aa_sequence::calculate_monoisotopic_mass;
///
/// let mass = calculate_monoisotopic_mass("PEPTIDEC[UNIMOD:4]R");
/// // assert_eq!(mass, 1115.4917246863);
/// ```
pub fn calculate_monoisotopic_mass(sequence: &str) -> f64 {
    let amino_acid_masses = amino_acid_masses();
    let modifications_mz_numerical = unimod_modifications_mz_numerical();
    let pattern = Regex::new(r"\[UNIMOD:(\d+)\]").unwrap();

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
    let mass_sequence: f64 = aa_counts.iter().map(|(aa, &count)| amino_acid_masses.get(&aa.to_string()[..]).unwrap_or(&0.0) * count as f64).sum();
    let mass_modifics: f64 = modifications.iter().map(|&mod_id| modifications_mz_numerical.get(&mod_id).unwrap_or(&0.0)).sum();

    mass_sequence + mass_modifics + MASS_WATER
}

pub fn calculate_b_y_fragment_mz(sequence: &str, modifications: Vec<f64>, is_y: Option<bool>, charge: Option<i32>) -> f64 {
    // Return mz of empty sequence
    if sequence.is_empty() {
        return 0.0;
    }

    let amino_acid_masses = amino_acid_masses();

    // Add up raw amino acid masses and potential modifications
    let mass_sequence: f64 = sequence.chars()
        .map(|aa| amino_acid_masses.get(&aa.to_string()[..]).unwrap_or(&0.0))
        .sum();

    let mass_modifications: f64 = modifications.iter().sum();

    // Calculate total mass
    let mass = mass_sequence + mass_modifications;

    // Set default values if None
    let is_y = is_y.unwrap_or(false);
    let charge = charge.unwrap_or(1);

    // If sequence is n-terminal (is_y is true), add water mass and calculate mz
    if is_y {
        calculate_mz(mass + MASS_WATER, charge)
    } else {
        // Otherwise, calculate mz
        calculate_mz(mass, charge)
    }
}

pub fn calculate_b_y_ion_series(sequence: &str, modifications: Vec<f64>, charge: Option<i32>) -> (Vec<(f64, String, String)>, Vec<(f64, String, String)>) {
    let mut b_ions = Vec::new();
    let mut y_ions = Vec::new();

    let char_indices: Vec<usize> = sequence.char_indices().map(|(i, _)| i).collect();
    let sequence_length = char_indices.len();

    // Iterate over all possible cleavage sites
    for i in 0..=sequence_length {
        let b_index = *char_indices.get(i).unwrap_or(&sequence.len());
        let y_index = *char_indices.get(i).unwrap_or(&0);

        let y = &sequence[y_index..];
        let b = &sequence[..b_index];
        let m_y = &modifications[i..];
        let m_b = &modifications[..i];

        // Calculate mz of b ions
        if !b.is_empty() && i != sequence_length {
            let b_mass = calculate_b_y_fragment_mz(b, m_b.to_vec(), Some(false), charge);
            b_ions.push((b_mass, format!("b{}+{}", i, charge.unwrap_or(1)), b.to_string()));
        }

        // Calculate mz of y ions
        if !y.is_empty() && i != 0 && i != sequence_length {
            let y_mass = calculate_b_y_fragment_mz(y, m_y.to_vec(), Some(true), charge);
            y_ions.push((y_mass, format!("y{}+{}", sequence_length - i, charge.unwrap_or(1)), y.to_string()));
        }
    }

    (b_ions, y_ions)
}


/// calculate the m/z of an ion
///
/// Arguments:
///
/// * `mono_mass` - monoisotopic mass of the ion
/// * `charge` - charge state of the ion
///
/// Returns:
///
/// * `mz` - mass-over-charge of the ion
///
/// # Examples
///
/// ```
/// use mscore::chemistry::aa_sequence::calculate_mz;
///
/// let mz = calculate_mz(1000.0, 2);
/// assert_eq!(mz, 501.007276466621);
/// ```
pub fn calculate_mz(monoisotopic_mass: f64, charge: i32) -> f64 {
    (monoisotopic_mass + charge as f64 * MASS_PROTON) / charge as f64
}