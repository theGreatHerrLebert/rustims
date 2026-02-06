use std::collections::BTreeMap;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};

/// Configuration for spectrum preprocessing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpectrumProcessingConfig {
    /// Maximum number of peaks to keep after filtering (default: 150)
    pub take_top_n: usize,
    /// Whether to perform deisotoping (default: true)
    pub deisotope: bool,
    /// PPM tolerance for deisotoping (default: 10.0)
    pub deisotope_tolerance_ppm: f64,
    /// Minimum m/z difference for deisotoping (default: 5.0 ppm equivalent at low mass)
    pub deisotope_min_mz_diff: f64,
}

impl Default for SpectrumProcessingConfig {
    fn default() -> Self {
        SpectrumProcessingConfig {
            take_top_n: 150,
            deisotope: true,
            deisotope_tolerance_ppm: 10.0,
            deisotope_min_mz_diff: 0.0005,
        }
    }
}

/// Represents a fully preprocessed spectrum ready for database search
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreprocessedSpectrum {
    /// Unique spectrum identifier (e.g., "frame_id-precursor_id-dataset_name")
    pub spec_id: String,
    /// Precursor m/z value (monoisotopic if available, otherwise highest intensity)
    pub precursor_mz: f64,
    /// Precursor charge (optional)
    pub precursor_charge: Option<i32>,
    /// Precursor intensity
    pub precursor_intensity: f64,
    /// Inverse ion mobility (1/K0)
    pub inverse_ion_mobility: f64,
    /// Collision energy used for fragmentation
    pub collision_energy: f64,
    /// Scan start time / retention time in minutes
    pub scan_start_time: f64,
    /// Total ion current of the spectrum
    pub total_ion_current: f64,
    /// Fragment neutral mass values (as f32 for Sage compatibility)
    /// Note: These are neutral masses (m/z - proton mass), matching Sage's Peak representation
    pub mz: Vec<f32>,
    /// Fragment intensity values (as f32 for Sage compatibility)
    pub intensity: Vec<f32>,
    /// Isolation m/z window center
    pub isolation_mz: f64,
    /// Isolation window width
    pub isolation_width: f64,
}

impl PreprocessedSpectrum {
    pub fn new(
        spec_id: String,
        precursor_mz: f64,
        precursor_charge: Option<i32>,
        precursor_intensity: f64,
        inverse_ion_mobility: f64,
        collision_energy: f64,
        scan_start_time: f64,
        total_ion_current: f64,
        mz: Vec<f32>,
        intensity: Vec<f32>,
        isolation_mz: f64,
        isolation_width: f64,
    ) -> Self {
        PreprocessedSpectrum {
            spec_id,
            precursor_mz,
            precursor_charge,
            precursor_intensity,
            inverse_ion_mobility,
            collision_energy,
            scan_start_time,
            total_ion_current,
            mz,
            intensity,
            isolation_mz,
            isolation_width,
        }
    }
}

/// Calculate the isotope mass difference (~1.003 Da for most elements)
const ISOTOPE_MASS_DIFF: f64 = 1.00335;

/// Proton mass in Daltons - used for converting m/z to neutral mass
/// Sage stores fragment peaks as neutral mass, not m/z
const PROTON_MASS: f64 = 1.007276466812;

/// Deisotope a spectrum by removing peaks that are likely isotopes of more intense peaks.
///
/// # Arguments
/// * `mz` - m/z values (must be sorted in ascending order)
/// * `intensity` - intensity values corresponding to m/z
/// * `tolerance_ppm` - PPM tolerance for matching isotope peaks
///
/// # Returns
/// Tuple of (filtered_mz, filtered_intensity) with isotope peaks removed
pub fn deisotope_spectrum(
    mz: &[f64],
    intensity: &[f64],
    tolerance_ppm: f64,
) -> (Vec<f64>, Vec<f64>) {
    if mz.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let n = mz.len();
    let mut is_isotope = vec![false; n];

    // For each peak, check if there's a more intense peak that could be the monoisotopic
    for i in 0..n {
        if is_isotope[i] {
            continue;
        }

        let current_mz = mz[i];
        let current_intensity = intensity[i];

        // Look for potential isotope peaks (up to +4 Da to cover up to charge 1-4)
        for charge in 1..=4 {
            let isotope_spacing = ISOTOPE_MASS_DIFF / charge as f64;

            // Check for isotope peaks at +1, +2, +3 isotope positions
            for isotope_num in 1..=3 {
                let expected_mz = current_mz + isotope_spacing * isotope_num as f64;
                let tolerance = expected_mz * tolerance_ppm / 1_000_000.0;

                // Binary search for potential matches
                for j in (i + 1)..n {
                    if mz[j] < expected_mz - tolerance {
                        continue;
                    }
                    if mz[j] > expected_mz + tolerance {
                        break;
                    }

                    // Found a potential isotope peak
                    // Mark it as isotope if it's less intense than the monoisotopic
                    if intensity[j] < current_intensity {
                        is_isotope[j] = true;
                    }
                }
            }
        }
    }

    // Collect non-isotope peaks
    let mut filtered_mz = Vec::with_capacity(n);
    let mut filtered_intensity = Vec::with_capacity(n);

    for i in 0..n {
        if !is_isotope[i] {
            filtered_mz.push(mz[i]);
            filtered_intensity.push(intensity[i]);
        }
    }

    (filtered_mz, filtered_intensity)
}

/// Filter spectrum to keep only the top N most intense peaks.
///
/// # Arguments
/// * `mz` - m/z values
/// * `intensity` - intensity values
/// * `top_n` - maximum number of peaks to keep
///
/// # Returns
/// Tuple of (filtered_mz, filtered_intensity) sorted by m/z
pub fn filter_top_n(
    mz: &[f64],
    intensity: &[f64],
    top_n: usize,
) -> (Vec<f64>, Vec<f64>) {
    if mz.len() <= top_n {
        return (mz.to_vec(), intensity.to_vec());
    }

    // Create indices sorted by intensity (descending)
    let mut indices: Vec<usize> = (0..mz.len()).collect();
    indices.sort_by(|&a, &b| {
        intensity[b].partial_cmp(&intensity[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top N indices
    indices.truncate(top_n);

    // Sort by m/z for the final result
    indices.sort_by(|&a, &b| {
        mz[a].partial_cmp(&mz[b]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let filtered_mz: Vec<f64> = indices.iter().map(|&i| mz[i]).collect();
    let filtered_intensity: Vec<f64> = indices.iter().map(|&i| intensity[i]).collect();

    (filtered_mz, filtered_intensity)
}

/// Normalize intensity values to sum to a constant (e.g., 1.0 or 10000.0)
pub fn normalize_intensity(intensity: &[f64], target_sum: f64) -> Vec<f64> {
    let sum: f64 = intensity.iter().sum();
    if sum == 0.0 {
        return intensity.to_vec();
    }
    intensity.iter().map(|&i| i * target_sum / sum).collect()
}

/// Flatten a TimsFrame-like structure (multiple scans) into a single spectrum.
/// Groups peaks by TOF index, sums intensities, and averages m/z values.
///
/// # Arguments
/// * `tof` - TOF indices for each data point
/// * `mz` - m/z values for each data point
/// * `intensity` - intensity values for each data point
///
/// # Returns
/// Tuple of (flattened_mz, flattened_intensity) sorted by m/z
pub fn flatten_frame_to_spectrum(
    tof: &[i32],
    mz: &[f64],
    intensity: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    // Group by TOF index
    let mut grouped_data: BTreeMap<i32, Vec<(f64, f64)>> = BTreeMap::new();

    for i in 0..tof.len() {
        grouped_data
            .entry(tof[i])
            .or_insert_with(Vec::new)
            .push((mz[i], intensity[i]));
    }

    let mut result_mz = Vec::with_capacity(grouped_data.len());
    let mut result_intensity = Vec::with_capacity(grouped_data.len());

    for (_, values) in grouped_data {
        let sum_intensity: f64 = values.iter().map(|&(_, i)| i).sum();
        let avg_mz: f64 = values.iter().map(|&(m, _)| m).sum::<f64>() / values.len() as f64;

        result_mz.push(avg_mz);
        result_intensity.push(sum_intensity);
    }

    (result_mz, result_intensity)
}

/// Calculate the inverse mobility at the scan with highest total intensity.
/// This implements the "marginal" mobility calculation.
///
/// # Arguments
/// * `scan` - scan indices for each data point
/// * `mobility` - inverse mobility values for each data point
/// * `intensity` - intensity values for each data point
///
/// # Returns
/// The inverse mobility value at the scan with highest summed intensity
pub fn get_inverse_mobility_along_scan_marginal(
    scan: &[i32],
    mobility: &[f64],
    intensity: &[f64],
) -> f64 {
    let mut marginal_map: BTreeMap<i32, (f64, f64)> = BTreeMap::new();

    for i in 0..scan.len() {
        let entry = marginal_map.entry(scan[i]).or_insert((0.0, 0.0));
        entry.0 += intensity[i];  // Sum intensity
        entry.1 = mobility[i];     // Store mobility (overwrite is fine, same scan = same mobility)
    }

    marginal_map
        .iter()
        .max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, (_, mob))| *mob)
        .unwrap_or(0.0)
}

/// Process a single spectrum: flatten, optionally deisotope, filter top N peaks.
///
/// # Arguments
/// * `tof` - TOF indices
/// * `mz` - m/z values
/// * `intensity` - intensity values
/// * `config` - processing configuration
///
/// # Returns
/// Tuple of (processed_mass, processed_intensity) as f32 vectors for Sage compatibility.
/// Note: m/z values are converted to neutral mass by subtracting the proton mass,
/// matching Sage's Peak representation which stores neutral mass.
pub fn process_spectrum(
    tof: &[i32],
    mz: &[f64],
    intensity: &[f64],
    config: &SpectrumProcessingConfig,
) -> (Vec<f32>, Vec<f32>) {
    // Step 1: Flatten frame to spectrum (group by TOF)
    let (flat_mz, flat_intensity) = flatten_frame_to_spectrum(tof, mz, intensity);

    // Step 2: Optionally deisotope
    let (deiso_mz, deiso_intensity) = if config.deisotope {
        deisotope_spectrum(&flat_mz, &flat_intensity, config.deisotope_tolerance_ppm)
    } else {
        (flat_mz, flat_intensity)
    };

    // Step 3: Filter to top N peaks
    let (final_mz, final_intensity) = filter_top_n(&deiso_mz, &deiso_intensity, config.take_top_n);

    // Convert m/z to neutral mass by subtracting proton mass (assumes singly charged fragments)
    // This matches Sage's SpectrumProcessor behavior which stores neutral mass in Peak objects
    let mass_f32: Vec<f32> = final_mz.iter().map(|&x| (x - PROTON_MASS) as f32).collect();
    let intensity_f32: Vec<f32> = final_intensity.iter().map(|&x| x as f32).collect();

    (mass_f32, intensity_f32)
}

/// Metadata required for processing a single PASEF fragment
#[derive(Clone, Debug)]
pub struct PASEFFragmentData {
    pub frame_id: u32,
    pub precursor_id: u32,
    pub collision_energy: f64,
    pub scan_start_time: f64,  // retention time in minutes
    pub scan: Vec<i32>,
    pub mobility: Vec<f64>,
    pub tof: Vec<i32>,
    pub mz: Vec<f64>,
    pub intensity: Vec<f64>,
    // Precursor metadata
    pub precursor_mz: f64,
    pub precursor_charge: Option<i32>,
    pub precursor_intensity: f64,
    pub isolation_mz: f64,
    pub isolation_width: f64,
}

/// Process a batch of PASEF fragments in parallel.
///
/// # Arguments
/// * `fragments` - Vector of PASEF fragment data
/// * `dataset_name` - Name of the dataset for generating spec_ids
/// * `config` - Processing configuration
/// * `num_threads` - Number of threads to use for parallel processing
///
/// # Returns
/// Vector of preprocessed spectra ready for database search
pub fn process_pasef_fragments_batch(
    fragments: Vec<PASEFFragmentData>,
    dataset_name: &str,
    config: &SpectrumProcessingConfig,
    num_threads: usize,
) -> Vec<PreprocessedSpectrum> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let ds_name = dataset_name.to_string();

    pool.install(|| {
        fragments
            .par_iter()
            .map(|frag| {
                // Calculate inverse mobility along scan marginal
                let inverse_mobility = get_inverse_mobility_along_scan_marginal(
                    &frag.scan,
                    &frag.mobility,
                    &frag.intensity,
                );

                // Process the spectrum (flatten, deisotope, filter)
                let (processed_mz, processed_intensity) = process_spectrum(
                    &frag.tof,
                    &frag.mz,
                    &frag.intensity,
                    config,
                );

                // Calculate total ion current
                let total_ion_current: f64 = frag.intensity.iter().sum();

                // Generate spec_id: frame_id-precursor_id-dataset_name
                let spec_id = format!("{}-{}-{}", frag.frame_id, frag.precursor_id, ds_name);

                PreprocessedSpectrum::new(
                    spec_id,
                    frag.precursor_mz,
                    frag.precursor_charge,
                    frag.precursor_intensity,
                    inverse_mobility,
                    frag.collision_energy,
                    frag.scan_start_time,
                    total_ion_current,
                    processed_mz,
                    processed_intensity,
                    frag.isolation_mz,
                    frag.isolation_width,
                )
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deisotope_spectrum() {
        // Simple test case: monoisotopic peak at 500.0 with isotopes at ~501.003 and ~502.006
        let mz = vec![500.0, 501.003, 502.006, 600.0];
        let intensity = vec![1000.0, 500.0, 250.0, 800.0];

        let (filtered_mz, _filtered_intensity) = deisotope_spectrum(&mz, &intensity, 10.0);

        // Should keep monoisotopic (500.0) and remove isotopes, keep 600.0
        assert_eq!(filtered_mz.len(), 2);
        assert!((filtered_mz[0] - 500.0).abs() < 0.001);
        assert!((filtered_mz[1] - 600.0).abs() < 0.001);
    }

    #[test]
    fn test_filter_top_n() {
        let mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let intensity = vec![10.0, 50.0, 30.0, 100.0, 20.0];

        let (filtered_mz, _filtered_intensity) = filter_top_n(&mz, &intensity, 3);

        // Should keep top 3 by intensity: 400 (100), 200 (50), 300 (30)
        assert_eq!(filtered_mz.len(), 3);
        // Should be sorted by m/z
        assert!((filtered_mz[0] - 200.0).abs() < 0.001);
        assert!((filtered_mz[1] - 300.0).abs() < 0.001);
        assert!((filtered_mz[2] - 400.0).abs() < 0.001);
    }

    #[test]
    fn test_flatten_frame_to_spectrum() {
        // Two data points with same TOF should be merged
        let tof = vec![1000, 1000, 2000];
        let mz = vec![500.0, 500.1, 600.0];
        let intensity = vec![100.0, 200.0, 300.0];

        let (flat_mz, flat_intensity) = flatten_frame_to_spectrum(&tof, &mz, &intensity);

        assert_eq!(flat_mz.len(), 2);
        // First group: avg mz = (500.0 + 500.1) / 2 = 500.05, sum intensity = 300
        assert!((flat_mz[0] - 500.05).abs() < 0.001);
        assert!((flat_intensity[0] - 300.0).abs() < 0.001);
        // Second group
        assert!((flat_mz[1] - 600.0).abs() < 0.001);
        assert!((flat_intensity[1] - 300.0).abs() < 0.001);
    }

    #[test]
    fn test_get_inverse_mobility_along_scan_marginal() {
        let scan = vec![100, 100, 101, 101];
        let mobility = vec![1.0, 1.0, 1.1, 1.1];
        let intensity = vec![50.0, 100.0, 200.0, 100.0];

        let result = get_inverse_mobility_along_scan_marginal(&scan, &mobility, &intensity);

        // Scan 101 has total intensity 300, scan 100 has 150
        // So should return mobility at scan 101 = 1.1
        assert!((result - 1.1).abs() < 0.001);
    }

    #[test]
    fn test_process_spectrum() {
        let tof = vec![1000, 1001, 1002, 2000, 2001];
        let mz = vec![500.0, 501.003, 502.006, 600.0, 601.003];
        let intensity = vec![1000.0, 500.0, 250.0, 800.0, 400.0];

        let config = SpectrumProcessingConfig {
            take_top_n: 10,
            deisotope: true,
            deisotope_tolerance_ppm: 10.0,
            deisotope_min_mz_diff: 0.0005,
        };

        let (processed_mz, processed_intensity) = process_spectrum(&tof, &mz, &intensity, &config);

        // After deisotoping, should have fewer peaks
        assert!(processed_mz.len() <= mz.len());
        // Intensities should sum to less than original (isotopes removed)
        let processed_sum: f32 = processed_intensity.iter().sum();
        let original_sum: f64 = intensity.iter().sum();
        assert!(processed_sum < original_sum as f32);
    }
}
