use std::collections::{HashMap, HashSet};
use bincode::{Decode, Encode};
use itertools::Itertools;
use regex::Regex;
use serde::{Deserialize, Serialize};
use crate::algorithm::peptide::{calculate_peptide_mono_isotopic_mass, calculate_peptide_product_ion_mono_isotopic_mass, peptide_sequence_to_atomic_composition};
use crate::chemistry::amino_acid::{amino_acid_masses};
use crate::chemistry::formulas::calculate_mz;
use crate::chemistry::constants::{MASS_WATER, MASS_NH3, MASS_CO, MASS_PROTON, MASS_ELECTRON};
use crate::chemistry::utility::{find_unimod_patterns, reshape_prosit_array, unimod_sequence_to_tokens};
use crate::data::spectrum::MzSpectrum;
use crate::simulation::annotation::{MzSpectrumAnnotated, ContributionSource, SignalAttributes, SourceType, PeakAnnotation};

// helper types for easier reading
type Mass = f64;
type Abundance = f64;
type IsotopeDistribution = Vec<(Mass, Abundance)>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeptideIon {
    pub sequence: PeptideSequence,
    pub charge: i32,
    pub intensity: f64,
}

impl PeptideIon {
    pub fn new(sequence: String, charge: i32, intensity: f64, peptide_id: Option<i32>) -> Self {
        PeptideIon {
            sequence: PeptideSequence::new(sequence, peptide_id),
            charge,
            intensity,
        }
    }
    pub fn mz(&self) -> f64 {
        calculate_mz(self.sequence.mono_isotopic_mass(), self.charge)
    }

    pub fn calculate_isotope_distribution(
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

    pub fn calculate_isotopic_spectrum(
        &self,
        mass_tolerance: f64,
        abundance_threshold: f64,
        max_result: i32,
        intensity_min: f64,
    ) -> MzSpectrum {
        let isotopic_distribution = self.calculate_isotope_distribution(mass_tolerance, abundance_threshold, max_result, intensity_min);
        MzSpectrum::new(isotopic_distribution.iter().map(|(mz, _)| *mz).collect(), isotopic_distribution.iter().map(|(_, abundance)| *abundance).collect()) * self.intensity
    }

    pub fn calculate_isotopic_spectrum_annotated(
        &self,
        mass_tolerance: f64,
        abundance_threshold: f64,
        max_result: i32,
        intensity_min: f64,
    ) -> MzSpectrumAnnotated {
        let isotopic_distribution = self.calculate_isotope_distribution(mass_tolerance, abundance_threshold, max_result, intensity_min);
        let mut annotations = Vec::new();
        let mut isotope_counter = 0;
        let mut previous_mz = isotopic_distribution[0].0;



        for (mz, abundance) in isotopic_distribution.iter() {

            let ppm_tolerance = (mz / 1e6) * 25.0;

            if (mz - previous_mz).abs() > ppm_tolerance {
                isotope_counter += 1;
                previous_mz = *mz;
            }

            let signal_attributes = SignalAttributes {
                charge_state: self.charge,
                peptide_id: self.sequence.peptide_id.unwrap_or(-1),
                isotope_peak: isotope_counter,
                description: None,
            };

            let contribution_source = ContributionSource {
                intensity_contribution: *abundance,
                source_type: SourceType::Signal,
                signal_attributes: Some(signal_attributes)
            };

            annotations.push(PeakAnnotation {
                contributions: vec![contribution_source]
            });
        }

        MzSpectrumAnnotated::new(isotopic_distribution.iter().map(|(mz, _)| *mz).collect(), isotopic_distribution.iter().map(|(_, abundance)| *abundance).collect(), annotations)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FragmentType { A, B, C, X, Y, Z, }

// implement to string for fragment type
impl std::fmt::Display for FragmentType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            FragmentType::A => write!(f, "a"),
            FragmentType::B => write!(f, "b"),
            FragmentType::C => write!(f, "c"),
            FragmentType::X => write!(f, "x"),
            FragmentType::Y => write!(f, "y"),
            FragmentType::Z => write!(f, "z"),
        }
    }
}

/// A neutral loss from a product ion (e.g. water, ammonia, phospho H3PO4).
/// Carries the element composition (atoms removed, positive counts) so the loss
/// ion's isotope envelope stays chemically correct — a mass-only delta would not.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutralLoss {
    pub name: String,
    pub mono_mass: f64,
    pub composition: HashMap<String, i32>,
}

impl NeutralLoss {
    pub fn new(name: &str, mono_mass: f64, composition: Vec<(&str, i32)>) -> Self {
        NeutralLoss {
            name: name.to_string(),
            mono_mass,
            composition: composition.into_iter().map(|(e, c)| {
                intern_element(e); // validate the element symbol at construction (fail-fast)
                (e.to_string(), c)
            }).collect(),
        }
    }
    /// Water loss, -H2O (Ser/Thr/Glu/Asp side chains, C-term).
    pub fn water() -> Self { NeutralLoss::new("H2O", MASS_WATER, vec![("H", 2), ("O", 1)]) }
    /// Ammonia loss, -NH3 (Lys/Arg/Asn/Gln side chains).
    pub fn ammonia() -> Self { NeutralLoss::new("NH3", MASS_NH3, vec![("N", 1), ("H", 3)]) }
    /// Phosphoric acid loss, -H3PO4 (phospho-Ser/Thr diagnostic, -97.9769).
    pub fn phospho() -> Self { NeutralLoss::new("H3PO4", 97.9768958, vec![("H", 3), ("P", 1), ("O", 4)]) }
}

/// Intern an element symbol to a 'static reference so a loss composition can be folded
/// into the 'static-keyed atomic-composition map. Covers every element that appears in a
/// realistic proteomics neutral loss (organics + common halogens/metals/adduct ions). An
/// unsupported symbol panics — the same contract as `generate_isotope_distribution`, which
/// also rejects unknown elements; `NeutralLoss::new` validates against this set so the
/// failure surfaces at construction rather than deep in a mass calculation.
fn intern_element(symbol: &str) -> &'static str {
    match symbol {
        "H" => "H", "C" => "C", "N" => "N", "O" => "O", "P" => "P", "S" => "S",
        "Se" => "Se", "Cl" => "Cl", "Br" => "Br", "I" => "I", "F" => "F",
        "Na" => "Na", "K" => "K", "Fe" => "Fe", "Mg" => "Mg", "Ca" => "Ca", "Zn" => "Zn",
        other => panic!("Unsupported element symbol '{}' in neutral loss composition", other),
    }
}

/// An immonium ion: a single side-chain cation from an internal a/y double cleavage,
/// `H2N+=CHR`. Charge 1 only. The m/z follows spectrum_utils' convention
/// `residue_mass - CO + H_atom` (H atom = proton + electron, not the bare proton; note
/// the backbone ions use the proton — this split mirrors the oracle exactly).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmoniumIon {
    /// Residue token the ion derives from, including any modification, e.g. "P" or "S[UNIMOD:21]".
    pub residue: String,
    pub mz: f64,
    /// Atomic composition (residue - CO) for isotope generation.
    pub composition: HashMap<String, i32>,
}

impl ImmoniumIon {
    pub fn isotope_distribution(&self, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> IsotopeDistribution {
        if self.composition.values().any(|&c| c < 0) {
            return Vec::new();
        }
        let composition: HashMap<String, i32> = self.composition.iter().filter(|&(_, &v)| v > 0).map(|(k, v)| (k.clone(), *v)).collect();
        let raw = crate::algorithm::isotope::generate_isotope_distribution(&composition, mass_tolerance, abundance_threshold, max_result);
        // Anchor on the true monoisotopic mass (lowest mass) of the UNFILTERED envelope, so
        // the shift onto the charge-1 immonium m/z is correct even if the mono peak is later
        // dropped by intensity_min.
        let mono = match raw.iter().map(|&(m, _)| m).fold(f64::INFINITY, f64::min) {
            m if m.is_finite() => m,
            _ => return Vec::new(),
        };
        raw.into_iter()
            .filter(|&(_, abundance)| abundance > intensity_min)
            .map(|(mass, abundance)| (self.mz + (mass - mono), abundance))
            .collect()
    }
}

/// An internal fragment: a b-type ion over an interior subsequence of the peptide
/// (both terminal residues excluded), produced by a double backbone cleavage. Chemistry
/// is identical to a b ion of that subsequence, so it reuses `PeptideProductIon{kind:B}`;
/// only the label (a residue span) differs. Matches spectrum_utils' "m" ions, whose
/// `m{start}:{end}` spans 1-based residue positions `start..end-1`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalIon {
    /// 1-based start residue position (inclusive).
    pub start: usize,
    /// 1-based end position, exclusive — matching spectrum_utils' `m{start}:{end}` label.
    pub end: usize,
    /// The underlying b-type product ion over the interior subsequence (charge baked in).
    pub ion: PeptideProductIon,
}

impl InternalIon {
    pub fn mz(&self) -> f64 { self.ion.mz() }
    /// spectrum_utils-style label, e.g. "m2:4".
    pub fn label(&self) -> String { format!("m{}:{}", self.start, self.end) }
    pub fn isotope_distribution(&self, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> IsotopeDistribution {
        self.ion.isotope_distribution(mass_tolerance, abundance_threshold, max_result, intensity_min)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeptideProductIon {
    pub kind: FragmentType,
    pub ion: PeptideIon,
    pub neutral_loss: Option<NeutralLoss>,
}

impl PeptideProductIon {
    pub fn new(kind: FragmentType, sequence: String, charge: i32, intensity: f64, peptide_id: Option<i32>) -> Self {
        PeptideProductIon {
            kind,
            ion: PeptideIon {
                sequence: PeptideSequence::new(sequence, peptide_id),
                charge,
                intensity,
            },
            neutral_loss: None,
        }
    }

    /// Attach a neutral loss to this product ion (builder; no post-construction mutation).
    pub fn with_neutral_loss(mut self, loss: NeutralLoss) -> Self {
        self.neutral_loss = Some(loss);
        self
    }

    pub fn mono_isotopic_mass(&self) -> f64 {
        let base = calculate_peptide_product_ion_mono_isotopic_mass(self.ion.sequence.sequence.as_str(), self.kind);
        match &self.neutral_loss {
            Some(nl) => base - nl.mono_mass,
            None => base,
        }
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

        // Fold in any neutral loss so the composition reflects the actual ion and stays
        // consistent with mono_isotopic_mass()/mz()/isotope_distribution(). A loss larger
        // than the fragment leaves negative counts (a chemically impossible ion).
        if let Some(nl) = &self.neutral_loss {
            for (element, count) in &nl.composition {
                *composition.entry(intern_element(element)).or_insert(0) -= count;
            }
        }
        composition
    }

    pub fn mz(&self) -> f64 {
        calculate_mz(self.mono_isotopic_mass(), self.ion.charge)
    }

    pub fn isotope_distribution(
        &self,
        mass_tolerance: f64,
        abundance_threshold: f64,
        max_result: i32,
        intensity_min: f64,
    ) -> IsotopeDistribution {

        // atomic_composition() already folds in any neutral loss. Only the loss path needs
        // the impossible-ion guard; the loss-free path is left byte-identical to before.
        let composition = self.atomic_composition();
        let atomic_composition: HashMap<String, i32> = if self.neutral_loss.is_some() {
            // A loss larger than the fragment (e.g. -H3PO4 on a non-phospho fragment) leaves
            // a negative atom count -> chemically impossible, no isotope envelope.
            if composition.values().any(|&c| c < 0) {
                return Vec::new();
            }
            // Drop zero counts: generate_isotope_distribution treats count <= 1 as a single
            // atom, so a 0 would spuriously add one atom of that element.
            composition.iter().filter(|&(_, &v)| v > 0).map(|(k, v)| (k.to_string(), *v)).collect()
        } else {
            composition.iter().map(|(k, v)| (k.to_string(), *v)).collect()
        };

        let distribution: IsotopeDistribution = crate::algorithm::isotope::generate_isotope_distribution(&atomic_composition, mass_tolerance, abundance_threshold, max_result)
            .into_iter().filter(|&(_, abundance)| abundance > intensity_min).collect();

        let mz_distribution = distribution.iter().map(|(mass, _)| calculate_mz(*mass, self.ion.charge)).zip(distribution.iter().map(|&(_, abundance)| abundance)).collect();

        mz_distribution
    }

    /// Calculate the isotope distribution of the complementary fragment.
    ///
    /// This is used for quad-selection dependent isotope transmission calculations.
    /// The complementary fragment is the portion of the precursor that remains
    /// after the fragment ion is produced.
    ///
    /// # Arguments
    ///
    /// * `precursor_composition` - atomic composition of the full precursor
    /// * `mass_tolerance` - mass tolerance for isotope distribution calculation
    /// * `abundance_threshold` - minimum abundance threshold
    /// * `max_result` - maximum number of isotope peaks
    ///
    /// # Returns
    ///
    /// * `Vec<(f64, f64)>` - complementary fragment isotope distribution as (mass, abundance) pairs
    pub fn complementary_isotope_distribution(
        &self,
        precursor_composition: &HashMap<&str, i32>,
        mass_tolerance: f64,
        abundance_threshold: f64,
        max_result: i32,
    ) -> Vec<(f64, f64)> {
        let fragment_composition = self.atomic_composition();
        let complementary_composition = crate::algorithm::peptide::calculate_complementary_fragment_composition(
            precursor_composition,
            &fragment_composition,
        );

        crate::algorithm::isotope::generate_isotope_distribution(
            &complementary_composition,
            mass_tolerance,
            abundance_threshold,
            max_result,
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct PeptideSequence {
    pub sequence: String,
    pub peptide_id: Option<i32>,
}

impl PeptideSequence {
    pub fn new(raw_sequence: String, peptide_id: Option<i32>) -> Self {

        // constructor will parse the sequence and check if it is valid
        let pattern = Regex::new(r"\[UNIMOD:(\d+)]").unwrap();

        // remove the modifications from the sequence
        let sequence = pattern.replace_all(&raw_sequence, "").to_string();

        // check if all remaining characters are valid amino acids
        let valid_amino_acids = sequence.chars().all(|c| amino_acid_masses().contains_key(&c.to_string()[..]));
        if !valid_amino_acids {
            panic!("Invalid amino acid sequence, use only valid amino acids: ARNDCQEGHILKMFPSTWYVU, and modifications in the format [UNIMOD:ID]");
        }

        PeptideSequence { sequence: raw_sequence, peptide_id }
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

    pub fn calculate_mono_isotopic_product_ion_spectrum_annotated(&self, charge: i32, fragment_type: FragmentType) -> MzSpectrumAnnotated {
        let product_ions = self.calculate_product_ion_series(charge, fragment_type);
        product_ions.generate_mono_isotopic_spectrum_annotated()
    }

    pub fn calculate_isotopic_product_ion_spectrum(&self, charge: i32, fragment_type: FragmentType, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> MzSpectrum {
        let product_ions = self.calculate_product_ion_series(charge, fragment_type);
        product_ions.generate_isotopic_spectrum(mass_tolerance, abundance_threshold, max_result, intensity_min)
    }

    pub fn calculate_isotopic_product_ion_spectrum_annotated(&self, charge: i32, fragment_type: FragmentType, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> MzSpectrumAnnotated {
        let product_ions = self.calculate_product_ion_series(charge, fragment_type);
        product_ions.generate_isotopic_spectrum_annotated(mass_tolerance, abundance_threshold, max_result, intensity_min)
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
                        peptide_id: self.peptide_id,
                    },
                    charge: target_charge,
                    intensity: 1.0, // Placeholder intensity
                },
                neutral_loss: None,
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
                        peptide_id: self.peptide_id,
                    },
                    charge: target_charge,
                    intensity: 1.0, // Placeholder intensity
                },
                neutral_loss: None,
            });
        }

        PeptideProductIonSeries::new(target_charge, n_terminal_ions, c_terminal_ions)
    }

    /// Backbone product-ion series augmented with neutral-loss variants. The base
    /// (loss-free) ions are kept; for every backbone ion and every supplied loss an
    /// extra ion carrying that loss is appended (spectrum_utils-style: each loss is
    /// applied to every backbone ion). Residue/mod-conditional pruning is a follow-up.
    pub fn calculate_product_ion_series_with_losses(
        &self,
        target_charge: i32,
        fragment_type: FragmentType,
        neutral_losses: Vec<NeutralLoss>,
    ) -> PeptideProductIonSeries {
        let base = self.calculate_product_ion_series(target_charge, fragment_type);
        let mut n_ions = base.n_ions.clone();
        let mut c_ions = base.c_ions.clone();
        for ion in &base.n_ions {
            for loss in &neutral_losses {
                n_ions.push(ion.clone().with_neutral_loss(loss.clone()));
            }
        }
        for ion in &base.c_ions {
            for loss in &neutral_losses {
                c_ions.push(ion.clone().with_neutral_loss(loss.clone()));
            }
        }
        PeptideProductIonSeries::new(target_charge, n_ions, c_ions)
    }

    /// Monoisotopic product-ion spectrum including neutral-loss peaks.
    pub fn calculate_mono_isotopic_product_ion_spectrum_with_losses(
        &self,
        charge: i32,
        fragment_type: FragmentType,
        neutral_losses: Vec<NeutralLoss>,
    ) -> MzSpectrum {
        self.calculate_product_ion_series_with_losses(charge, fragment_type, neutral_losses)
            .generate_mono_isotopic_spectrum()
    }

    /// Immonium ions for the distinct residues (including modified residues) present in
    /// the peptide. Charge 1. m/z = residue_mass - CO + H_atom (spectrum_utils convention).
    /// Modified-residue immonium (e.g. phospho-S) are an extension beyond spectrum_utils,
    /// which only enumerates the 20 unmodified residues.
    pub fn calculate_immonium_ions(&self) -> Vec<ImmoniumIon> {
        let tokens = unimod_sequence_to_tokens(self.sequence.as_str(), true);
        let mut seen: HashSet<String> = HashSet::new();
        let mut ions = Vec::new();
        for token in tokens {
            // Skip N/C-terminal modification sentinel tokens (they do not begin with a
            // residue letter, e.g. "\u{0}[UNIMOD:1]"); immonium ions are residue side chains.
            if !token.chars().next().map_or(false, |c| c.is_ascii_uppercase()) {
                continue;
            }
            if !seen.insert(token.clone()) {
                continue;
            }
            let residue = PeptideSequence::new(token.clone(), self.peptide_id);
            // single-residue "peptide" mass = residue + water; immonium = residue - CO + H_atom.
            let residue_mass = calculate_peptide_mono_isotopic_mass(&residue) - MASS_WATER;
            let mz = residue_mass - MASS_CO + MASS_PROTON + MASS_ELECTRON;
            // Immonium cation atoms = residue - CO + H  (e.g. proline C4H8N+, glycine CH4N+).
            // residue.atomic_composition() is (residue + water), so: - water - CO + H, i.e.
            // net H: -2 (water) -0 (CO) +1 (the immonium H) = -1; O: -1 (water) -1 (CO) = -2; C: -1 (CO).
            let mut composition: HashMap<String, i32> = residue.atomic_composition().iter().map(|(k, v)| (k.to_string(), *v)).collect();
            *composition.entry("H".to_string()).or_insert(0) -= 1;
            *composition.entry("O".to_string()).or_insert(0) -= 2;
            *composition.entry("C".to_string()).or_insert(0) -= 1;
            ions.push(ImmoniumIon { residue: token, mz, composition });
        }
        ions
    }

    /// Monoisotopic immonium spectrum (charge-1 peaks, unit intensity).
    pub fn calculate_immonium_spectrum(&self) -> MzSpectrum {
        let ions = self.calculate_immonium_ions();
        let mz: Vec<f64> = ions.iter().map(|i| i.mz).collect();
        let intensity: Vec<f64> = ions.iter().map(|_| 1.0).collect();
        MzSpectrum::new(mz, intensity).filter_ranged(0.0, 5_000.0, 1e-6, 1e6)
    }

    /// Internal fragment ions: b-type ions over interior subsequences (both terminal
    /// residues excluded), length >= 2. Matches spectrum_utils' "m" ions. Single-residue
    /// internals are immonium ions, generated separately by `calculate_immonium_ions`.
    /// `max_length` optionally caps the subsequence length to bound the O(L^2) blow-up;
    /// `None` enumerates all interior subsequences (spectrum_utils parity).
    pub fn calculate_internal_ions(&self, charge: i32, max_length: Option<usize>) -> Vec<InternalIon> {
        // Keep only residue tokens; terminal-modification sentinels (e.g. "\u{0}[UNIMOD:1]")
        // are not interior residues and would otherwise shift the position indexing.
        let tokens: Vec<String> = unimod_sequence_to_tokens(self.sequence.as_str(), true)
            .into_iter()
            .filter(|t| t.chars().next().map_or(false, |c| c.is_ascii_uppercase()))
            .collect();
        let len = tokens.len();
        let mut ions = Vec::new();
        if len < 4 {
            return ions; // need at least two interior residues for a length-2 internal
        }
        let cap = max_length.unwrap_or(usize::MAX);
        // 0-based token indices: start at 1 (exclude N-terminal residue), end exclusive at
        // most len-1 (exclude C-terminal residue), length (end - start) >= 2.
        for start_0 in 1..=(len - 3) {
            for end_excl in (start_0 + 2)..=(len - 1) {
                if end_excl - start_0 > cap {
                    break;
                }
                let subseq = tokens[start_0..end_excl].join("");
                let ion = PeptideProductIon::new(FragmentType::B, subseq, charge, 1.0, self.peptide_id);
                ions.push(InternalIon { start: start_0 + 1, end: end_excl + 1, ion });
            }
        }
        ions
    }

    /// Monoisotopic internal-fragment spectrum (unit intensity).
    pub fn calculate_internal_ion_spectrum(&self, charge: i32, max_length: Option<usize>) -> MzSpectrum {
        let ions = self.calculate_internal_ions(charge, max_length);
        let mz: Vec<f64> = ions.iter().map(|i| i.mz()).collect();
        let intensity: Vec<f64> = ions.iter().map(|_| 1.0).collect();
        MzSpectrum::new(mz, intensity).filter_ranged(0.0, 5_000.0, 1e-6, 1e6)
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    pub fn generate_mono_isotopic_spectrum_annotated(&self) -> MzSpectrumAnnotated {
        let mut annotations: Vec<PeakAnnotation> = Vec::with_capacity(self.n_ions.len() + self.c_ions.len());
        let mut mz_values = Vec::with_capacity(self.n_ions.len() + self.c_ions.len());
        let mut intensity_values = Vec::with_capacity(self.n_ions.len() + self.c_ions.len());

        for (index, n_ion) in self.n_ions.iter().enumerate() {
            let kind = n_ion.kind;
            let charge = n_ion.ion.charge;
            let mz = n_ion.mz();
            let intensity = n_ion.ion.intensity;
            let signal_attributes = SignalAttributes {
                charge_state: charge,
                peptide_id: n_ion.ion.sequence.peptide_id.unwrap_or(-1),
                isotope_peak: 0,
                description: Some(format!("{}_{}_{}", kind, index + 1, 0)),
            };
            let contribution_source = ContributionSource {
                intensity_contribution: intensity,
                source_type: SourceType::Signal,
                signal_attributes: Some(signal_attributes)
            };

            annotations.push(PeakAnnotation {
                contributions: vec![contribution_source]
            });
            mz_values.push(mz);
            intensity_values.push(intensity);
        }

        for (index, c_ion) in self.c_ions.iter().enumerate() {
            let kind = c_ion.kind;
            let charge = c_ion.ion.charge;
            let mz = c_ion.mz();
            let intensity = c_ion.ion.intensity;
            let signal_attributes = SignalAttributes {
                charge_state: charge,
                peptide_id: c_ion.ion.sequence.peptide_id.unwrap_or(-1),
                isotope_peak: 0,
                description: Some(format!("{}_{}_{}", kind, index + 1, 0)),
            };
            let contribution_source = ContributionSource {
                intensity_contribution: intensity,
                source_type: SourceType::Signal,
                signal_attributes: Some(signal_attributes)
            };

            annotations.push(PeakAnnotation {
                contributions: vec![contribution_source]
            });
            mz_values.push(mz);
            intensity_values.push(intensity);
        }

        MzSpectrumAnnotated::new(mz_values, intensity_values, annotations)
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

    pub fn generate_isotopic_spectrum_annotated(&self, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> MzSpectrumAnnotated {
        let mut annotations: Vec<PeakAnnotation> = Vec::new();
        let mut mz_values = Vec::new();
        let mut intensity_values = Vec::new();

        for (index, ion) in self.n_ions.iter().enumerate() {
            let n_isotopes = ion.isotope_distribution(mass_tolerance, abundance_threshold, max_result, intensity_min);
            let mut isotope_counter = 0;
            let mut previous_mz = n_isotopes[0].0;

            for (mz, abundance) in n_isotopes.iter() {
                let ppm_tolerance = (mz / 1e6) * 25.0;

                if (mz - previous_mz).abs() > ppm_tolerance {
                    isotope_counter += 1;
                    previous_mz = *mz;
                }

                let signal_attributes = SignalAttributes {
                    charge_state: ion.ion.charge,
                    peptide_id: ion.ion.sequence.peptide_id.unwrap_or(-1),
                    isotope_peak: isotope_counter,
                    // use convention of 1-based indexing for fragment ion enumeration
                    description: Some(format!("{}_{}_{}", ion.kind, index + 1, isotope_counter)),
                };

                let contribution_source = ContributionSource {
                    intensity_contribution: *abundance * ion.ion.intensity,
                    source_type: SourceType::Signal,
                    signal_attributes: Some(signal_attributes)
                };

                annotations.push(PeakAnnotation {
                    contributions: vec![contribution_source]
                });
                mz_values.push(*mz);
                intensity_values.push(*abundance * ion.ion.intensity);
            }
        }

        for (index, ion) in self.c_ions.iter().enumerate() {
            let c_isotopes = ion.isotope_distribution(mass_tolerance, abundance_threshold, max_result, intensity_min);
            let mut isotope_counter = 0;
            let mut previous_mz = c_isotopes[0].0;

            for (mz, abundance) in c_isotopes.iter() {
                let ppm_tolerance = (mz / 1e6) * 25.0;

                if (mz - previous_mz).abs() > ppm_tolerance {
                    isotope_counter += 1;
                    previous_mz = *mz;
                }

                let signal_attributes = SignalAttributes {
                    charge_state: ion.ion.charge,
                    peptide_id: ion.ion.sequence.peptide_id.unwrap_or(-1),
                    isotope_peak: isotope_counter,
                    description: Some(format!("{}_{}_{}", ion.kind, index + 1, isotope_counter)),
                };

                let contribution_source = ContributionSource {
                    intensity_contribution: *abundance * ion.ion.intensity,
                    source_type: SourceType::Signal,
                    signal_attributes: Some(signal_attributes)
                };

                annotations.push(PeakAnnotation {
                    contributions: vec![contribution_source]
                });

                mz_values.push(*mz);
                intensity_values.push(*abundance * ion.ion.intensity);
            }
        }
        MzSpectrumAnnotated::new(mz_values, intensity_values, annotations)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    pub fn generate_isotopic_spectrum_annotated(&self, mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64) -> MzSpectrumAnnotated {
        let mut annotations: Vec<PeakAnnotation> = Vec::new();
        let mut mz_values = Vec::new();
        let mut intensity_values = Vec::new();

        for ion_series in &self.peptide_ions {
            let isotopic_spectrum = ion_series.generate_isotopic_spectrum_annotated(mass_tolerance, abundance_threshold, max_result, intensity_min);
            for (mz, intensity) in isotopic_spectrum.mz.iter().zip(isotopic_spectrum.intensity.iter()) {
                mz_values.push(*mz);
                intensity_values.push(*intensity);
            }
            annotations.extend(isotopic_spectrum.annotations.iter().cloned());
        }

        MzSpectrumAnnotated::new(mz_values, intensity_values, annotations)
    }
}

#[cfg(test)]
mod neutral_loss_tests {
    use super::*;
    use crate::chemistry::constants::MASS_PROTON;

    // Layer 1 (correctness anchor): hand-computed neutral monoisotopic masses on PEPTIDE.
    // Base b3 (PEP) neutral mass = residues(P+E+P) + H2O - H2O = sum of residue masses.
    // P=97.05276384, E=129.04259308 -> b3 neutral = 2*97.05276384 + 129.04259308 = 323.14812076.
    #[test]
    fn b3_neutral_mass_matches_hand_computed() {
        let ps = PeptideSequence::new("PEPTIDE".to_string(), None);
        let series = ps.calculate_product_ion_series(1, FragmentType::B);
        let b3 = &series.n_ions[2];
        assert!((b3.mono_isotopic_mass() - 323.14812076).abs() < 1e-5,
            "b3 neutral mass {} != 323.14812076", b3.mono_isotopic_mass());
    }

    // Loss subtracts exactly its monoisotopic mass from the neutral fragment mass.
    #[test]
    fn neutral_loss_subtracts_mono_mass() {
        let ps = PeptideSequence::new("PEPTIDESK".to_string(), None);
        let series = ps.calculate_product_ion_series(1, FragmentType::Y);
        let y_ion = &series.c_ions[3];
        let base = y_ion.mono_isotopic_mass();
        for nl in [NeutralLoss::water(), NeutralLoss::ammonia(), NeutralLoss::phospho()] {
            let expected = base - nl.mono_mass;
            let got = y_ion.clone().with_neutral_loss(nl).mono_isotopic_mass();
            assert!((got - expected).abs() < 1e-9, "loss mass {} != {}", got, expected);
        }
    }

    // m/z = (neutral_mass - loss + z*proton)/z : loss applied to the neutral mass before protonation.
    #[test]
    fn loss_mz_at_charge_two() {
        let ps = PeptideSequence::new("PEPTIDE".to_string(), None);
        let series = ps.calculate_product_ion_series(2, FragmentType::B);
        let b3 = &series.n_ions[2];
        let loss = NeutralLoss::water();
        let neutral = b3.mono_isotopic_mass();
        let expected_mz = (neutral - loss.mono_mass + 2.0 * MASS_PROTON) / 2.0;
        let got = b3.clone().with_neutral_loss(loss).mz();
        assert!((got - expected_mz).abs() < 1e-6, "loss mz {} != {}", got, expected_mz);
    }

    // with-losses generation keeps the base ions and adds one extra peak per (ion, loss).
    #[test]
    fn with_losses_appends_variants() {
        let ps = PeptideSequence::new("PEPTIDE".to_string(), None);
        let base = ps.calculate_product_ion_series(1, FragmentType::B);
        let augmented = ps.calculate_product_ion_series_with_losses(
            1, FragmentType::B, vec![NeutralLoss::water(), NeutralLoss::ammonia()]);
        assert_eq!(augmented.n_ions.len(), base.n_ions.len() * 3);
        assert_eq!(augmented.c_ions.len(), base.c_ions.len() * 3);
        assert!(augmented.n_ions[..base.n_ions.len()].iter().all(|i| i.neutral_loss.is_none()));
        assert!(augmented.n_ions[base.n_ions.len()..].iter().all(|i| i.neutral_loss.is_some()));
    }

    // atomic_composition() must reflect the loss so it agrees with mass/mz/isotopes.
    #[test]
    fn atomic_composition_reflects_loss() {
        let ps = PeptideSequence::new("PEPTIDESK".to_string(), None);
        let series = ps.calculate_product_ion_series(1, FragmentType::Y);
        let y = series.c_ions[3].clone();
        let y_loss = y.clone().with_neutral_loss(NeutralLoss::water());
        let base = y.atomic_composition();
        let lost = y_loss.atomic_composition();
        assert_eq!(lost["H"], base["H"] - 2, "H not reduced by water loss");
        assert_eq!(lost["O"], base["O"] - 1, "O not reduced by water loss");
    }

    // Chemically impossible loss (-H3PO4 on a non-phospho proline b1) -> no isotope envelope.
    #[test]
    fn impossible_loss_yields_empty_isotopes() {
        let ps = PeptideSequence::new("PEPTIDEK".to_string(), None);
        let series = ps.calculate_product_ion_series(1, FragmentType::B);
        let b1 = series.n_ions[0].clone().with_neutral_loss(NeutralLoss::phospho());
        assert!(b1.atomic_composition().values().any(|&c| c < 0), "expected a negative atom count");
        let iso = b1.isotope_distribution(1e-3, 1e-4, 10, 1e-4);
        assert!(iso.is_empty(), "impossible loss should yield no isotope peaks, got {}", iso.len());
    }

    // A valid loss (-H2O) still produces a sane isotope envelope.
    #[test]
    fn valid_loss_yields_isotopes() {
        let ps = PeptideSequence::new("PEPTIDESK".to_string(), None);
        let series = ps.calculate_product_ion_series(1, FragmentType::Y);
        let y = series.c_ions[3].clone().with_neutral_loss(NeutralLoss::water());
        let iso = y.isotope_distribution(1e-3, 1e-4, 10, 1e-6);
        assert!(!iso.is_empty(), "valid loss ion should have isotope peaks");
        // monoisotopic peak m/z matches the mono spectrum within tolerance
        assert!((iso[0].0 - y.mz()).abs() < 1e-3, "first isotope mz {} != mono mz {}", iso[0].0, y.mz());
    }

    // Layer 1: proline immonium m/z is the well-known 70.0657 (residue - CO + H_atom).
    #[test]
    fn immonium_proline_mass() {
        let ps = PeptideSequence::new("PEPTIDE".to_string(), None);
        let imm = ps.calculate_immonium_ions();
        let pro = imm.iter().find(|i| i.residue == "P").expect("no proline immonium");
        assert!((pro.mz - 70.0657).abs() < 1e-3, "proline immonium {} != 70.0657", pro.mz);
    }

    // Only distinct residues yield immonium ions (PEPTIDE -> P,E,T,I,D = 5).
    #[test]
    fn immonium_distinct_residues() {
        let ps = PeptideSequence::new("PEPTIDE".to_string(), None);
        assert_eq!(ps.calculate_immonium_ions().len(), 5);
    }

    // Modified-residue immonium: phospho-S immonium = S immonium + phospho mod mass (~79.9663).
    #[test]
    fn immonium_modified_residue() {
        let plain = PeptideSequence::new("SAGE".to_string(), None);
        let s_plain = plain.calculate_immonium_ions().iter().find(|i| i.residue == "S").unwrap().mz;
        let phospho = PeptideSequence::new("S[UNIMOD:21]AGE".to_string(), None);
        let s_phos = phospho.calculate_immonium_ions().iter().find(|i| i.residue == "S[UNIMOD:21]").unwrap().mz;
        assert!((s_phos - s_plain - 79.966331).abs() < 1e-3, "phospho-S immonium delta {} != 79.9663", s_phos - s_plain);
    }

    // Immonium cation composition must be residue - CO + H (proline -> C4H8N+), and its
    // isotope envelope must place the monoisotopic peak exactly on the ion m/z.
    #[test]
    fn immonium_composition_and_isotope_anchor() {
        let ps = PeptideSequence::new("PEPTIDE".to_string(), None);
        let pro = ps.calculate_immonium_ions().into_iter().find(|i| i.residue == "P").unwrap();
        assert_eq!(pro.composition["C"], 4);
        assert_eq!(pro.composition["H"], 8);
        assert_eq!(pro.composition["N"], 1);
        assert_eq!(pro.composition.get("O").copied().unwrap_or(0), 0);
        let iso = pro.isotope_distribution(1e-3, 1e-4, 5, 1e-6);
        assert!(!iso.is_empty(), "immonium should have isotope peaks");
        assert!(iso.iter().any(|&(m, _)| (m - pro.mz).abs() < 1e-6), "no isotope peak on the immonium m/z");
    }

    // PEPTIDE has 10 interior subsequences of length >=2 (spectrum_utils gives the same 10).
    #[test]
    fn internal_ion_count() {
        let ps = PeptideSequence::new("PEPTIDE".to_string(), None);
        assert_eq!(ps.calculate_internal_ions(1, None).len(), 10);
    }

    // First internal of PEPTIDE is m2:4 = "EP", a b-ion: E+P residues + proton = 227.10263.
    #[test]
    fn internal_ion_m2_4_mass() {
        let ps = PeptideSequence::new("PEPTIDE".to_string(), None);
        let ions = ps.calculate_internal_ions(1, None);
        let m24 = ions.iter().find(|i| i.label() == "m2:4").expect("no m2:4");
        assert_eq!(m24.ion.ion.sequence.sequence, "EP");
        assert!((m24.mz() - 227.10263).abs() < 1e-3, "m2:4 mz {} != 227.10263", m24.mz());
    }

    // Length cap limits subsequence length; termini are always excluded.
    #[test]
    fn internal_ion_length_cap() {
        let ps = PeptideSequence::new("PEPTIDE".to_string(), None);
        let capped = ps.calculate_internal_ions(1, Some(2));
        assert!(capped.iter().all(|i| i.end - i.start == 2), "cap=2 should yield only length-2 internals");
        assert!(capped.iter().all(|i| i.start >= 2 && i.end <= 7), "internals must exclude both termini");
    }

    // N-terminal modifications tokenize as a sentinel; immonium/internal must skip it
    // (not panic) and position indexing must stay residue-based.
    #[test]
    fn nterm_mod_handled() {
        let ps = PeptideSequence::new("[UNIMOD:1]EEEDKPK".to_string(), None);
        let imm = ps.calculate_immonium_ions(); // must not panic
        assert!(imm.iter().all(|i| i.residue.chars().next().unwrap().is_ascii_uppercase()));
        let internals = ps.calculate_internal_ions(1, None);
        // residues are EEEDKPK (7); internals exclude the terminal E(1) and K(7).
        assert!(internals.iter().all(|i| i.start >= 2 && i.end <= 7),
            "n-term-mod internals must still exclude both terminal residues");
    }
}