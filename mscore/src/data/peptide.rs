use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use std::collections::HashMap;
use regex::Regex;
use crate::chemistry::constants::{MASS_WATER, MASS_PROTON, MASS_CO, MASS_NH3};
use crate::chemistry::amino_acid::{amino_acid_composition, amino_acid_masses};
use crate::chemistry::unimod::{modification_atomic_composition, unimod_modifications_mass_numerical};
use crate::chemistry::utility::{find_unimod_patterns, unimod_sequence_to_tokens};

// helper types for easier reading
type Mass = f64;
type Abundance = f64;
type IsotopeDistribution = Vec<(Mass, Abundance)>;

#[derive(Debug, Clone)]
pub struct PeptideIon {
    pub sequence: PeptideSequence,
    pub charge: i32,
    pub intensity: f64,
}

impl PeptideIon {
    pub fn new(sequence: String, charge: i32, intensity: f64) -> Self {
        PeptideIon {
            sequence: PeptideSequence::new(sequence),
            charge,
            intensity,
        }
    }
    pub fn mz(&self) -> f64 {
        calculate_mz(self.sequence.mono_isotopic_mass(), self.charge)
    }

    pub fn isotope_distribution(
        &self,
        mass_tolerance: f64,
        abundance_threshold: f64,
        max_result: i32,
        intensity_min: f64,
    ) -> IsotopeDistribution {

        let atomic_composition: HashMap<String, i32> = self.sequence.atomic_composition().iter().map(|(k, v)| (k.to_string(), *v)).collect();

        let distribution: IsotopeDistribution = crate::algorithm::isotope::generate_isotope_distribution(&atomic_composition, mass_tolerance, abundance_threshold, max_result)
            .into_iter().filter(|&(_, abundance)| abundance > intensity_min).collect();

        let mz_distribution = distribution.iter().map(|(mass, _)| calculate_mz(*mass, self.charge))
            .zip(distribution.iter().map(|&(_, abundance)| abundance)).collect();

        mz_distribution
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FragmentType { A, B, C, X, Y, Z, }

#[derive(Debug, Clone)]
pub struct PeptideProductIon {
    pub kind: FragmentType,
    pub ion: PeptideIon,
}

impl PeptideProductIon {
    pub fn new(kind: FragmentType, sequence: String, charge: i32, intensity: f64) -> Self {
        PeptideProductIon {
            kind,
            ion: PeptideIon {
                sequence: PeptideSequence::new(sequence),
                charge,
                intensity,
            },
        }
    }

    pub fn mono_isotopic_mass(&self) -> f64 {
        let (sequence, modifications) = self.ion.sequence.to_sage_representation();
        calculate_product_ion_mono_isotopic_mass(&sequence, modifications, self.kind)
    }

    pub fn atomic_composition(&self) -> HashMap<&str, i32> {

        let mut composition = peptide_sequence_to_atomic_composition(&self.ion.sequence);

        match self.kind {
            FragmentType::A => {
                *composition.entry("H").or_insert(0) -= 2;
                *composition.entry("O").or_insert(0) -= 2;
                *composition.entry("C").or_insert(0) -= 1;
            },

            FragmentType::B => {
                // B: peptide_mass - Water
                *composition.entry("H").or_insert(0) -= 2;
                *composition.entry("O").or_insert(0) -= 1;
            },

            FragmentType::C => {
                // C: peptide_mass + NH3 - Water
                *composition.entry("H").or_insert(0) += 1;
                *composition.entry("N").or_insert(0) += 1;
                *composition.entry("O").or_insert(0) -= 1;
            },

            FragmentType::X => {
                // X: peptide_mass + CO + 2*H - Water
                *composition.entry("C").or_insert(0) += 1;
                *composition.entry("O").or_insert(0) += 1;
            },

            FragmentType::Y => {
                ()
            },

            FragmentType::Z => {
                *composition.entry("H").or_insert(0) -= 1;
                *composition.entry("N").or_insert(0) -= 3;
            },
        }
        composition
    }

    pub fn mz(&self) -> f64 {
        calculate_mz(self.mono_isotopic_mass(), self.ion.charge)
    }

    pub fn isotope_distribution(&self,
                                mass_tolerance: f64,
                                abundance_threshold: f64,
                                max_result: i32,
                                intensity_min: f64,
    ) -> IsotopeDistribution {

        let atomic_composition: HashMap<String, i32> = self.atomic_composition().iter().map(|(k, v)| (k.to_string(), *v)).collect();

        let distribution: IsotopeDistribution = crate::algorithm::isotope::generate_isotope_distribution(&atomic_composition, mass_tolerance, abundance_threshold, max_result)
            .into_iter().filter(|&(_, abundance)| abundance > intensity_min).collect();

        let mz_distribution = distribution.iter().map(|(mass, _)| calculate_mz(*mass, self.ion.charge)).zip(distribution.iter().map(|&(_, abundance)| abundance)).collect();

        mz_distribution
    }
}

#[derive(Debug, Clone)]
pub struct PeptideSequence {
    pub sequence: String,
}

impl PeptideSequence {
    pub fn new(raw_sequence: String) -> Self {

        // constructor will parse the sequence and check if it is valid
        let pattern = Regex::new(r"\[UNIMOD:(\d+)]").unwrap();

        // remove the modifications from the sequence
        let sequence = pattern.replace_all(&raw_sequence, "").to_string();

        // check if all remaining characters are valid amino acids
        let valid_amino_acids = sequence.chars().all(|c| amino_acid_masses().contains_key(&c.to_string()[..]));
        if !valid_amino_acids {
            panic!("Invalid amino acid sequence, use only valid amino acids: ARNDCQEGHILKMFPSTWYVU, and modifications in the format [UNIMOD:ID]");
        }
        PeptideSequence { sequence: raw_sequence }
    }

    pub fn mono_isotopic_mass(&self) -> f64 {
        calculate_peptide_mono_isotopic_mass(&*self.sequence)
    }

    pub fn atomic_composition(&self) -> HashMap<&str, i32> {
        peptide_sequence_to_atomic_composition(self)
    }

    pub fn to_tokens(&self, group_modifications: bool) -> Vec<String> {
        unimod_sequence_to_tokens(&*self.sequence, group_modifications)
    }

    pub fn to_sage_representation(&self) -> (String, Vec<f64>) {
        find_unimod_patterns(&*self.sequence)
    }

    pub fn calculate_product_ion_series(&self, charge: i32, fragment_type: FragmentType) -> (Vec<PeptideProductIon>, Vec<PeptideProductIon>){

        // TODO: check for n-terminal modifications
        let tokens = unimod_sequence_to_tokens(self.sequence.as_str(), true);
        let mut n_terminal_ions = Vec::new();
        let mut c_terminal_ions = Vec::new();

        // Generate b ions
        for i in 1..tokens.len() {
            let n_ion_seq = tokens[..i].join("");
            n_terminal_ions.push(PeptideProductIon {
                kind: match fragment_type {
                    FragmentType::A => FragmentType::A,
                    FragmentType::B => FragmentType::B,
                    FragmentType::C => FragmentType::C,
                    FragmentType::X => FragmentType::A,
                    FragmentType::Y => FragmentType::B,
                    FragmentType::Z => FragmentType::C,
                },
                ion: PeptideIon {
                    sequence: PeptideSequence {
                        sequence: n_ion_seq,
                    },
                    charge, // Assuming charge 1 for simplicity
                    intensity: 1.0, // Placeholder intensity
                },
            });
        }

        // Generate y ions
        for i in 1..tokens.len() {
            let c_ion_seq = tokens[tokens.len() - i..].join("");
            c_terminal_ions.push(PeptideProductIon {
                kind: match fragment_type {
                    FragmentType::A => FragmentType::X,
                    FragmentType::B => FragmentType::Y,
                    FragmentType::C => FragmentType::Z,
                    FragmentType::X => FragmentType::X,
                    FragmentType::Y => FragmentType::Y,
                    FragmentType::Z => FragmentType::Z,
                },
                ion: PeptideIon {
                    sequence: PeptideSequence {
                        sequence: c_ion_seq,
                    },
                    charge, // Assuming charge 1 for simplicity
                    intensity: 1.0, // Placeholder intensity
                },
            });
        }

        (n_terminal_ions, c_terminal_ions)
    }
}

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
/// use mscore::data::peptide::calculate_peptide_mono_isotopic_mass;
///
/// let mass = calculate_peptide_mono_isotopic_mass("PEPTIDEC[UNIMOD:4]R");
/// // assert_eq!(mass, 1115.4917246863);
/// ```
pub fn calculate_peptide_mono_isotopic_mass(sequence: &str) -> f64 {
    let amino_acid_masses = amino_acid_masses();
    let modifications_mz_numerical = unimod_modifications_mass_numerical();
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

pub fn calculate_product_ion_mono_isotopic_mass(sequence: &str, modifications: Vec<f64>, kind: FragmentType) -> f64 {

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

pub fn calculate_product_ion_mz(sequence: &str, modifications: Vec<f64>, kind: FragmentType, charge: Option<i32>) -> f64 {
    let mass = calculate_product_ion_mono_isotopic_mass(sequence, modifications, kind);
    calculate_mz(mass, charge.unwrap_or(1))
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
            let b_mass = calculate_product_ion_mz(b, m_b.to_vec(), FragmentType::B, charge);
            b_ions.push((b_mass, format!("b{}+{}", i, charge.unwrap_or(1)), b.to_string()));
        }

        // Calculate mz of y ions
        if !y.is_empty() && i != 0 && i != sequence_length {
            let y_mass = calculate_product_ion_mz(y, m_y.to_vec(), FragmentType::Y, charge);
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
/// use mscore::data::peptide::calculate_mz;
///
/// let mz = calculate_mz(1000.0, 2);
/// assert_eq!(mz, 501.007276466621);
/// ```
pub fn calculate_mz(monoisotopic_mass: f64, charge: i32) -> f64 {
    (monoisotopic_mass + charge as f64 * MASS_PROTON) / charge as f64
}

pub fn calculate_amino_acid_composition(sequence: &str) -> HashMap<String, i32> {
    let mut composition = HashMap::new();
    for char in sequence.chars() {
        *composition.entry(char.to_string()).or_insert(0) += 1;
    }
    composition
}

pub fn peptide_sequence_to_atomic_composition(peptide_sequence: &PeptideSequence) -> HashMap<&'static str, i32> {

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

pub fn mono_isotopic_product_ion_composition(product_ion: &PeptideProductIon) -> Vec<(&str, i32)> {

    let mut composition = peptide_sequence_to_atomic_composition(&product_ion.ion.sequence);

    match product_ion.kind {
        FragmentType::A => {
            // A: peptide_mass - CO - Water
            *composition.entry("H").or_insert(0) -= 2;
            *composition.entry("O").or_insert(0) -= 2;
            *composition.entry("C").or_insert(0) -= 1;
        },
        FragmentType::B => {
            // B: peptide_mass - Water
            *composition.entry("H").or_insert(0) -= 2;
            *composition.entry("O").or_insert(0) -= 1;
        },
        FragmentType::C => {
            // C: peptide_mass + NH3 - Water
            *composition.entry("H").or_insert(0) += 1;
            *composition.entry("N").or_insert(0) += 1;
            *composition.entry("O").or_insert(0) -= 1;
        },
        FragmentType::X => {
            // X: peptide_mass + CO
            *composition.entry("C").or_insert(0) += 1; // Add 1 for CO
            *composition.entry("O").or_insert(0) += 1; // Add 1 for CO
            *composition.entry("H").or_insert(0) -= 2; // Subtract 2 for 2 protons
        },
        FragmentType::Y => {
            ()
        },
        FragmentType::Z => {
            *composition.entry("H").or_insert(0) -= 3;
            *composition.entry("N").or_insert(0) -= 1;
        },
    }

    composition.iter().map(|(k, v)| (*k, *v)).collect()
}

pub fn fragments_to_composition(product_ions: Vec<PeptideProductIon>, num_threads: usize) -> Vec<Vec<(String, i32)>> {
    let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    let result = thread_pool.install(|| {
        product_ions.par_iter().map(|ion| mono_isotopic_product_ion_composition(ion)).map(|composition| {
            composition.iter().map(|(k, v)| (k.to_string(), *v)).collect()
        }).collect()
    });
    result
}
