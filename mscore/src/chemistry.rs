extern crate statrs;
use statrs::distribution::{Continuous, Normal};
use regex::Regex;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::{MzSpectrum, ToResolution};
use std::collections::HashMap;

fn normal_pdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    let normal = Normal::new(mean, std_dev).unwrap();
    normal.pdf(x)
}

pub const MASS_PROTON: f64 = 1.007276466621;
pub const MASS_NEUTRON: f64 = 1.00866491595;
pub const MASS_ELECTRON: f64 = 0.00054857990946;
pub const MASS_WATER: f64 = 18.0105646863;

// IUPAC Standards
pub const STANDARD_TEMPERATURE: f64 = 273.15; // Kelvin
pub const STANDARD_PRESSURE: f64 = 1e5; // Pascal
pub const ELEMENTARY_CHARGE: f64 = 1.602176634e-19;
pub const K_BOLTZMANN: f64 = 1.380649e-23; // J/K

// Amino Acids and Their Codes
fn amino_acids() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();
    map.insert("Lysine", "K");
    map.insert("Alanine", "A");
    map.insert("Glycine", "G");
    map.insert("Valine", "V");
    map.insert("Tyrosine", "Y");
    map.insert("Arginine", "R");
    map.insert("Glutamic Acid", "E");
    map.insert("Phenylalanine", "F");
    map.insert("Tryptophan", "W");
    map.insert("Leucine", "L");
    map.insert("Threonine", "T");
    map.insert("Cysteine", "C");
    map.insert("Serine", "S");
    map.insert("Glutamine", "Q");
    map.insert("Methionine", "M");
    map.insert("Isoleucine", "I");
    map.insert("Asparagine", "N");
    map.insert("Proline", "P");
    map.insert("Histidine", "H");
    map.insert("Aspartic Acid", "D");
    map.insert("Selenocysteine", "U");
    map
}

// Amino Acid Masses
fn amino_acid_masses() -> HashMap<&'static str, f64> {
    let mut map = HashMap::new();
    map.insert("A", 71.037114);
    map.insert("R", 156.101111);
    map.insert("N", 114.042927);
    map.insert("D", 115.026943);
    map.insert("C", 103.009185);
    map.insert("E", 129.042593);
    map.insert("Q", 128.058578);
    map.insert("G", 57.021464);
    map.insert("H", 137.058912);
    map.insert("I", 113.084064);
    map.insert("L", 113.084064);
    map.insert("K", 128.094963);
    map.insert("M", 131.040485);
    map.insert("F", 147.068414);
    map.insert("P", 97.052764);
    map.insert("S", 87.032028);
    map.insert("T", 101.047679);
    map.insert("W", 186.079313);
    map.insert("Y", 163.063329);
    map.insert("V", 99.068414);
    map.insert("U", 168.053);
    map
}

// MODIFICATIONS_MZ with string keys and float values
fn modifications_mz() -> HashMap<&'static str, f64> {
    let mut map = HashMap::new();
    map.insert("[UNIMOD:58]", 56.026215);
    map.insert("[UNIMOD:408]", 148.037173);
    map.insert("[UNIMOD:43]", 203.079373);
    map.insert("[UNIMOD:7]", 0.984016);
    map.insert("[UNIMOD:1]", 42.010565);
    map.insert("[UNIMOD:35]", 15.994915);
    map.insert("[UNIMOD:1289]", 70.041865);
    map.insert("[UNIMOD:3]", 226.077598);
    map.insert("[UNIMOD:1363]", 68.026215);
    map.insert("[UNIMOD:36]", 28.031300);
    map.insert("[UNIMOD:122]", 27.994915);
    map.insert("[UNIMOD:1848]", 114.031694);
    map.insert("[UNIMOD:1849]", 86.036779);
    map.insert("[UNIMOD:64]", 100.016044);
    map.insert("[UNIMOD:37]", 42.046950);
    map.insert("[UNIMOD:121]", 114.042927);
    map.insert("[UNIMOD:747]", 86.000394);
    map.insert("[UNIMOD:34]", 14.015650);
    map.insert("[UNIMOD:354]", 44.985078);
    map.insert("[UNIMOD:4]", 57.021464);
    map.insert("[UNIMOD:21]", 79.966331);
    map.insert("[UNIMOD:312]", 119.004099);
    map
}

// MODIFICATIONS_MZ_NUMERICAL with integer keys and float values
fn modifications_mz_numerical() -> HashMap<u32, f64> {
    let mut map = HashMap::new();
    map.insert(58, 56.026215);
    map.insert(408, 148.037173);
    map.insert(43, 203.079373);
    map.insert(7, 0.984016);
    map.insert(1, 42.010565);
    map.insert(35, 15.994915);
    map.insert(1289, 70.041865);
    map.insert(3, 226.077598);
    map.insert(1363, 68.026215);
    map.insert(36, 28.031300);
    map.insert(122, 27.994915);
    map.insert(1848, 114.031694);
    map.insert(1849, 86.036779);
    map.insert(64, 100.016044);
    map.insert(37, 42.046950);
    map.insert(121, 114.042927);
    map.insert(747, 86.000394);
    map.insert(34, 14.015650);
    map.insert(354, 44.985078);
    map.insert(4, 57.021464);
    map.insert(21, 79.966331);
    map.insert(312, 119.004099);
    map
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
/// use mscore::calculate_monoisotopic_mass;
///
/// let mass = calculate_monoisotopic_mass("PEPTIDEC[UNIMOD:4]R");
/// // assert_eq!(mass, 1115.4917246863);
/// ```
pub fn calculate_monoisotopic_mass(sequence: &str) -> f64 {
    let amino_acid_masses = amino_acid_masses();
    let modifications_mz_numerical = modifications_mz_numerical();
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
/// use mscore::calculate_mz;
///
/// let mz = calculate_mz(1000.0, 2);
/// assert_eq!(mz, 501.007276466621);
/// ```
pub fn calculate_mz(monoisotopic_mass: f64, charge: i32) -> f64 {
    (monoisotopic_mass + charge as f64 * MASS_PROTON) / charge as f64
}

/// convert 1 over reduced ion mobility (1/k0) to CCS
///
/// Arguments:
///
/// * `one_over_k0` - 1 over reduced ion mobility (1/k0)
/// * `charge` - charge state of the ion
/// * `mz` - mass-over-charge of the ion
/// * `mass_gas` - mass of drift gas (N2)
/// * `temp` - temperature of the drift gas in C°
/// * `t_diff` - factor to translate from C° to K
///
/// Returns:
///
/// * `ccs` - collision cross-section
///
/// # Examples
///
/// ```
/// use mscore::one_over_reduced_mobility_to_ccs;
///
/// let ccs = one_over_reduced_mobility_to_ccs(0.5, 1000.0, 2, 28.013, 31.85, 273.15);
/// assert_eq!(ccs, 806.5918693771381);
/// ```
pub fn one_over_reduced_mobility_to_ccs(
    one_over_k0: f64,
    mz: f64,
    charge: u32,
    mass_gas: f64,
    temp: f64,
    t_diff: f64,
) -> f64 {
    let summary_constant = 18509.8632163405;
    let reduced_mass = (mz * charge as f64 * mass_gas) / (mz * charge as f64 + mass_gas);
    summary_constant * charge as f64 / (reduced_mass * (temp + t_diff)).sqrt() / one_over_k0
}


/// convert CCS to 1 over reduced ion mobility (1/k0)
///
/// Arguments:
///
/// * `ccs` - collision cross-section
/// * `charge` - charge state of the ion
/// * `mz` - mass-over-charge of the ion
/// * `mass_gas` - mass of drift gas (N2)
/// * `temp` - temperature of the drift gas in C°
/// * `t_diff` - factor to translate from C° to K
///
/// Returns:
///
/// * `one_over_k0` - 1 over reduced ion mobility (1/k0)
///
/// # Examples
///
/// ```
/// use mscore::ccs_to_reduced_mobility;
///
/// let k0 = ccs_to_reduced_mobility(806.5918693771381, 1000.0, 2, 28.013, 31.85, 273.15);
/// assert_eq!(1.0 / k0, 0.5);
/// ```
pub fn ccs_to_reduced_mobility(
    ccs: f64,
    mz: f64,
    charge: u32,
    mass_gas: f64,
    temp: f64,
    t_diff: f64,
) -> f64 {
    let summary_constant = 18509.8632163405;
    let reduced_mass = (mz * charge as f64 * mass_gas) / (mz * charge as f64 + mass_gas);
    ((reduced_mass * (temp + t_diff)).sqrt() * ccs) / (summary_constant * charge as f64)
}


/// calculate the factorial of a number
///
/// Arguments:
///
/// * `n` - number to calculate factorial of
///
/// Returns:
///
/// * `f64` - factorial of `n`
///
/// # Examples
///
/// ```
/// use mscore::factorial;
///
/// let fact = factorial(5);
/// assert_eq!(fact, 120.0);
/// ```
pub fn factorial(n: i32) -> f64 {
    (1..=n).fold(1.0, |acc, x| acc * x as f64)
}

fn weight(mass: f64, peak_nums: Vec<i32>, normalize: bool) -> Vec<f64> {
    let lam_val = lam(mass, 0.000594, -0.03091);
    let factorials: Vec<f64> = peak_nums.iter().map(|&k| factorial(k)).collect();
    let mut weights: Vec<f64> = peak_nums.iter().map(|&k| {
        let pow = lam_val.powi(k);
        let exp = (-lam_val).exp();
        exp * pow / factorials[k as usize]
    }).collect();

    if normalize {
        let sum: f64 = weights.iter().sum();
        weights = weights.iter().map(|&w| w / sum).collect();
    }

    weights
}

fn lam(mass: f64, slope: f64, intercept: f64) -> f64 {
    slope * mass + intercept
}

fn iso(x: &Vec<f64>, mass: f64, charge: f64, sigma: f64, amp: f64, k: usize, step_size: f64, mass_neutron: f64) -> Vec<f64> {
    let k_range: Vec<usize> = (0..k).collect();
    let means: Vec<f64> = k_range.iter().map(|&k_val| (mass + mass_neutron * k_val as f64) / charge).collect();
    let weights = weight(mass, k_range.iter().map(|&k_val| k_val as i32).collect::<Vec<i32>>(), true);

    let mut intensities = vec![0.0; x.len()];
    for (i, x_val) in x.iter().enumerate() {
        for (j, &mean) in means.iter().enumerate() {
            intensities[i] += weights[j] * normal_pdf(*x_val, mean, sigma);
        }
        intensities[i] *= step_size;
    }
    intensities.iter().map(|&intensity| intensity * amp).collect()
}

fn generate_isotope_pattern(lower_bound: f64, upper_bound: f64, mass: f64, charge: f64, amp: f64, k: usize, sigma: f64, resolution: i32) -> (Vec<f64>, Vec<f64>) {
    let step_size = f64::min(sigma / 10.0, 1.0 / 10f64.powi(resolution));
    let size = ((upper_bound - lower_bound) / step_size).ceil() as usize;
    let mzs: Vec<f64> = (0..size).map(|i| lower_bound + step_size * i as f64).collect();
    let intensities = iso(&mzs, mass, charge, sigma, amp, k, step_size, MASS_NEUTRON);

    (mzs.iter().map(|&mz| mz + MASS_PROTON).collect(), intensities)
}

pub fn generate_averagine_spectrum(
    mass: f64,
    charge: i32,
    min_intensity: i32,
    k: i32,
    resolution: i32,
    centroid: bool,
    amp: Option<f64>
) -> MzSpectrum {
    let amp = amp.unwrap_or(1e4);
    let lb = mass / charge as f64 - 0.2;
    let ub = mass / charge as f64 + k as f64 + 0.2;

    let (mz, intensities) = generate_isotope_pattern(
        lb,
        ub,
        mass,
        charge as f64,
        amp,
        k as usize,
        0.008492569002123142,
        resolution,
    );

    let spectrum = MzSpectrum::new(mz, intensities).to_resolution(resolution).filter_ranged(lb, ub, min_intensity as f64, 1e9);

    if centroid {
        spectrum.to_centroided(std::cmp::max(min_intensity, 1), 1.0 / 10f64.powi(resolution - 1), true)
    } else {
        spectrum
    }
}

pub fn generate_averagine_spectra(
    masses: Vec<f64>,
    charges: Vec<i32>,
    min_intensity: i32,
    k: i32,
    resolution: i32,
    centroid: bool,
    num_threads: usize,
    amp: Option<f64>
) -> Vec<MzSpectrum> {
    let amp = amp.unwrap_or(1e5);
    let mut spectra: Vec<MzSpectrum> = Vec::new();
    let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();

    thread_pool.install(|| {
        spectra = masses.par_iter().zip(charges.par_iter()).map(|(&mass, &charge)| {
            generate_averagine_spectrum(mass, charge, min_intensity, k, resolution, centroid, Some(amp))
        }).collect();
    });

    spectra
}

