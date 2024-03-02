use std::collections::HashMap;
use itertools::Itertools;
use regex::Regex;
use crate::algorithm::peptide::{calculate_peptide_mono_isotopic_mass, calculate_peptide_product_ion_mono_isotopic_mass, peptide_sequence_to_atomic_composition};
use crate::chemistry::amino_acid::{amino_acid_masses};
use crate::chemistry::formulas::calculate_mz;
use crate::chemistry::utility::{find_unimod_patterns, reshape_prosit_array, unimod_sequence_to_tokens};
use crate::data::spectrum::MzSpectrum;

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
        calculate_peptide_product_ion_mono_isotopic_mass(self.ion.sequence.sequence.as_str(), self.kind)
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
        calculate_peptide_mono_isotopic_mass(self)
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

    pub fn amino_acid_count(&self) -> usize {
        self.to_tokens(true).len()
    }

    pub fn calculate_mono_isotopic_product_ion_spectrum(&self, charge: i32, fragment_type: FragmentType) -> MzSpectrum {
        let product_ions = self.calculate_product_ion_series(charge, fragment_type);
        product_ions.generate_mono_isotopic_spectrum()
    }

    pub fn calculate_isotopic_product_ion_spectrum(&self, charge: i32, fragment_type: FragmentType, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> MzSpectrum {
        let product_ions = self.calculate_product_ion_series(charge, fragment_type);
        product_ions.generate_isotopic_spectrum(mass_tolerance, abundance_threshold, max_result, intensity_min)
    }

    pub fn calculate_product_ion_series(&self, target_charge: i32, fragment_type: FragmentType) -> PeptideProductIonSeries {

        // TODO: check for n-terminal modifications
        let tokens = unimod_sequence_to_tokens(self.sequence.as_str(), true);
        let mut n_terminal_ions = Vec::new();
        let mut c_terminal_ions = Vec::new();

        // Generate n ions
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
                    charge: target_charge,
                    intensity: 1.0, // Placeholder intensity
                },
            });
        }

        // Generate c ions
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
                    charge: target_charge,
                    intensity: 1.0, // Placeholder intensity
                },
            });
        }

        PeptideProductIonSeries::new(target_charge, n_terminal_ions, c_terminal_ions)
    }

    pub fn associate_with_predicted_intensities(
        &self,
        // TODO: check docs of prosit if charge is meant as precursor charge or max charge of fragments to generate
        charge: i32,
        fragment_type: FragmentType,
        flat_intensities: Vec<f64>,
        normalize: bool,
        half_charge_one: bool,
    ) -> PeptideProductIonSeriesCollection {

        let reshaped_intensities = reshape_prosit_array(flat_intensities);
        let max_charge = std::cmp::min(charge, 3).max(1); // Ensure at least 1 for loop range
        let mut sum_intensity = if normalize { 0.0 } else { 1.0 };
        let num_tokens = self.amino_acid_count() - 1; // Full sequence length is not counted as fragment, since nothing is cleaved off, therefore -1

        let mut peptide_ion_collection = Vec::new();

        if normalize {
            for z in 1..=max_charge {

                let intensity_c: Vec<f64> = reshaped_intensities[..num_tokens].iter().map(|x| x[0][z as usize - 1]).filter(|&x| x > 0.0).collect();
                let intensity_n: Vec<f64> = reshaped_intensities[..num_tokens].iter().map(|x| x[1][z as usize - 1]).filter(|&x| x > 0.0).collect();

                sum_intensity += intensity_n.iter().sum::<f64>() + intensity_c.iter().sum::<f64>();
            }
        }

        for z in 1..=max_charge {

            let mut product_ions = self.calculate_product_ion_series(z, fragment_type);
            let intensity_n: Vec<f64> = reshaped_intensities[..num_tokens].iter().map(|x| x[1][z as usize - 1]).collect();
            let intensity_c: Vec<f64> = reshaped_intensities[..num_tokens].iter().map(|x| x[0][z as usize - 1]).collect(); // Reverse for y

            let adjusted_sum_intensity = if max_charge == 1 && half_charge_one { sum_intensity * 2.0 } else { sum_intensity };

            for (i, ion) in product_ions.n_ions.iter_mut().enumerate() {
                ion.ion.intensity = intensity_n[i] / adjusted_sum_intensity;
            }
            for (i, ion) in product_ions.c_ions.iter_mut().enumerate() {
                ion.ion.intensity = intensity_c[i] / adjusted_sum_intensity;
            }

            peptide_ion_collection.push(PeptideProductIonSeries::new(z, product_ions.n_ions, product_ions.c_ions));
        }

        PeptideProductIonSeriesCollection::new(peptide_ion_collection)
    }
}

#[derive(Debug, Clone)]
pub struct PeptideProductIonSeries {
    pub charge: i32,
    pub n_ions: Vec<PeptideProductIon>,
    pub c_ions: Vec<PeptideProductIon>,
}

impl PeptideProductIonSeries {
    pub fn new(charge: i32, n_ions: Vec<PeptideProductIon>, c_ions: Vec<PeptideProductIon>) -> Self {
        PeptideProductIonSeries {
            charge,
            n_ions,
            c_ions,
        }
    }

    pub fn generate_mono_isotopic_spectrum(&self) -> MzSpectrum {
        let mz_i_n = self.n_ions.iter().map(|ion| (ion.mz(), ion.ion.intensity)).collect_vec();
        let mz_i_c = self.c_ions.iter().map(|ion| (ion.mz(), ion.ion.intensity)).collect_vec();
        let n_spectrum = MzSpectrum::new(mz_i_n.iter().map(|(mz, _)| *mz).collect(), mz_i_n.iter().map(|(_, abundance)| *abundance).collect());
        let c_spectrum = MzSpectrum::new(mz_i_c.iter().map(|(mz, _)| *mz).collect(), mz_i_c.iter().map(|(_, abundance)| *abundance).collect());
        MzSpectrum::from_collection(vec![n_spectrum, c_spectrum]).filter_ranged(0.0, 5_000.0, 1e-6, 1e6)
    }
    pub fn generate_isotopic_spectrum(&self, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> MzSpectrum {
        let mut spectra: Vec<MzSpectrum> = Vec::new();

        for ion in &self.n_ions {
            let n_isotopes = ion.isotope_distribution(mass_tolerance, abundance_threshold, max_result, intensity_min);
            let spectrum = MzSpectrum::new(n_isotopes.iter().map(|(mz, _)| *mz).collect(), n_isotopes.iter().map(|(_, abundance)| *abundance * ion.ion.intensity).collect());
            spectra.push(spectrum);
        }

        for ion in &self.c_ions {
            let c_isotopes = ion.isotope_distribution(mass_tolerance, abundance_threshold, max_result, intensity_min);
            let spectrum = MzSpectrum::new(c_isotopes.iter().map(|(mz, _)| *mz).collect(), c_isotopes.iter().map(|(_, abundance)| *abundance * ion.ion.intensity).collect());
            spectra.push(spectrum);
        }

        MzSpectrum::from_collection(spectra).filter_ranged(0.0, 5_000.0, 1e-6, 1e6)
    }
}

#[derive(Debug, Clone)]
pub struct PeptideProductIonSeriesCollection {
    pub peptide_ions: Vec<PeptideProductIonSeries>,
}
impl PeptideProductIonSeriesCollection {
    pub fn new(peptide_ions: Vec<PeptideProductIonSeries>) -> Self {
        PeptideProductIonSeriesCollection {
            peptide_ions,
        }
    }

    pub fn find_ion_series(&self, charge: i32) -> Option<&PeptideProductIonSeries> {
        self.peptide_ions.iter().find(|ion_series| ion_series.charge == charge)
    }

    pub fn generate_isotopic_spectrum(&self, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> MzSpectrum {
        let mut spectra: Vec<MzSpectrum> = Vec::new();

        for ion_series in &self.peptide_ions {
            let isotopic_spectrum = ion_series.generate_isotopic_spectrum(mass_tolerance, abundance_threshold, max_result, intensity_min);
            spectra.push(isotopic_spectrum);
        }

        MzSpectrum::from_collection(spectra).filter_ranged(0.0, 5_000.0, 1e-6, 1e6)
    }
}