use crate::data::acquisition::AcquisitionMode;
use crate::data::handle::{IndexConverter, TimsData, TimsDataLoader};
use crate::data::meta::{read_dda_precursor_meta, read_global_meta_sql, read_meta_data_sql, read_pasef_frame_ms_ms_info, DDAPrecursor, DDAPrecursorMeta, PasefMsMsMeta};
use mscore::timstof::frame::{ImsFrame, RawTimsFrame, TimsFrame};
use mscore::timstof::slice::TimsSlice;
use mscore::timstof::spectrum_processing::{
    PASEFFragmentData, PreprocessedSpectrum, SpectrumProcessingConfig,
    process_pasef_fragments_batch,
};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::collections::BTreeMap;
use rand::prelude::IteratorRandom;
use mscore::data::spectrum::MsType;
use std::collections::HashMap;

#[derive(Clone)]
pub struct PASEFDDAFragment {
    pub frame_id: u32,
    pub precursor_id: u32,
    pub collision_energy: f64,
    pub selected_fragment: TimsFrame,
}

/// Statistical moments of a 1D signal distribution
#[derive(Clone, Debug, Default)]
pub struct SignalMoments {
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub apex: f64,
    pub fwhm: f64,
    pub total_intensity: f64,
}

impl SignalMoments {
    /// Calculate moments from coordinate and intensity arrays
    pub fn from_signal(coords: &[f64], intensities: &[f64]) -> Self {
        if coords.is_empty() || intensities.iter().sum::<f64>() == 0.0 {
            return Self::default();
        }

        let total: f64 = intensities.iter().sum();

        // First moment (weighted mean)
        let mean: f64 = coords.iter()
            .zip(intensities.iter())
            .map(|(c, i)| c * i / total)
            .sum();

        // Second moment (weighted variance)
        let variance: f64 = coords.iter()
            .zip(intensities.iter())
            .map(|(c, i)| i / total * (c - mean).powi(2))
            .sum();

        // Third moment (weighted skewness)
        let std = variance.sqrt().max(1e-10);
        let skewness: f64 = coords.iter()
            .zip(intensities.iter())
            .map(|(c, i)| i / total * ((c - mean) / std).powi(3))
            .sum();

        // Apex (position of maximum)
        let apex_idx = intensities.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let apex = coords.get(apex_idx).copied().unwrap_or(0.0);

        // FWHM estimation
        let half_max = intensities.get(apex_idx).copied().unwrap_or(0.0) / 2.0;
        let above_half: Vec<f64> = coords.iter()
            .zip(intensities.iter())
            .filter(|(_, i)| **i >= half_max)
            .map(|(c, _)| *c)
            .collect();
        let fwhm = if above_half.len() >= 2 {
            above_half.last().unwrap_or(&0.0) - above_half.first().unwrap_or(&0.0)
        } else {
            2.355 * std  // Gaussian approximation
        };

        SignalMoments {
            mean,
            variance,
            skewness,
            apex,
            fwhm,
            total_intensity: total,
        }
    }
}

/// MS1 precursor signal extracted from surrounding frames
#[derive(Clone, Debug)]
pub struct PrecursorMS1Signal {
    pub precursor_id: u32,

    // XIC (chromatographic profile) - 1D projection
    pub rt_coords: Vec<f64>,           // RT in seconds
    pub rt_intensities: Vec<f64>,
    pub rt_moments: SignalMoments,

    // Mobilogram (IM profile) - 1D projection
    pub im_coords: Vec<f64>,           // 1/K0
    pub im_intensities: Vec<f64>,
    pub im_moments: SignalMoments,

    // Isotope envelope - 1D projection
    pub isotope_mz: Vec<f64>,
    pub isotope_intensity: Vec<f64>,
    pub mz_moments: SignalMoments,

    // Raw 2D data (all peaks from filtered MS1 frames in RT window, merged)
    pub raw_rt: Vec<f64>,              // RT per peak (seconds)
    pub raw_mz: Vec<f64>,              // m/z per peak
    pub raw_mobility: Vec<f64>,        // 1/K0 per peak
    pub raw_intensity: Vec<f64>,       // intensity per peak
}

/// Input for MS1 extraction - precursor coordinates
#[derive(Clone, Debug)]
pub struct PrecursorCoord {
    pub precursor_id: u32,
    pub mz: f64,           // m/z for XIC/mobilogram extraction (use largest_peak_mz for best signal)
    pub mono_mz: f64,      // Monoisotopic m/z for isotope envelope extraction (M+0 starting point)
    pub rt_seconds: f64,
    pub mobility: f64,
    pub charge: i32,
}

pub struct TimsDatasetDDA {
    pub loader: TimsDataLoader,
    pub pasef_meta: Vec<PasefMsMsMeta>,
}

impl TimsDatasetDDA {
    pub fn new(
        bruker_lib_path: &str,
        data_path: &str,
        in_memory: bool,
        use_bruker_sdk: bool,
    ) -> Self {
        // TODO: error handling
        let global_meta_data = read_global_meta_sql(data_path).unwrap();
        let meta_data = read_meta_data_sql(data_path).unwrap();

        let scan_max_index = meta_data.iter().map(|x| x.num_scans).max().unwrap() as u32;
        let im_lower = global_meta_data.one_over_k0_range_lower;
        let im_upper = global_meta_data.one_over_k0_range_upper;

        let tof_max_index = global_meta_data.tof_max_index;
        let mz_lower = global_meta_data.mz_acquisition_range_lower;
        let mz_upper = global_meta_data.mz_acquisition_range_upper;

        let loader = match in_memory {
            true => TimsDataLoader::new_in_memory(
                bruker_lib_path,
                data_path,
                use_bruker_sdk,
                scan_max_index,
                im_lower,
                im_upper,
                tof_max_index,
                mz_lower,
                mz_upper,
            ),
            false => TimsDataLoader::new_lazy(
                bruker_lib_path,
                data_path,
                use_bruker_sdk,
                scan_max_index,
                im_lower,
                im_upper,
                tof_max_index,
                mz_lower,
                mz_upper,
            ),
        };
        
        let pasef_meta = read_pasef_frame_ms_ms_info(data_path).unwrap();

        TimsDatasetDDA { loader, pasef_meta }
    }

    /// Create a new DDA dataset with pre-computed ion mobility calibration lookup table.
    ///
    /// This enables accurate ion mobility calibration with fast parallel extraction.
    /// The im_lookup table should be pre-computed using the Bruker SDK.
    ///
    /// # Arguments
    /// * `data_path` - Path to the .d folder
    /// * `in_memory` - Whether to load all data into memory
    /// * `im_lookup` - Pre-computed scan→1/K0 lookup table from Bruker SDK
    ///
    /// # Returns
    /// A new TimsDatasetDDA with LookupIndexConverter (thread-safe, accurate)
    pub fn new_with_calibration(
        data_path: &str,
        in_memory: bool,
        im_lookup: Vec<f64>,
    ) -> Self {
        let global_meta_data = read_global_meta_sql(data_path).unwrap();

        let tof_max_index = global_meta_data.tof_max_index;
        let mz_lower = global_meta_data.mz_acquisition_range_lower;
        let mz_upper = global_meta_data.mz_acquisition_range_upper;

        let loader = match in_memory {
            true => TimsDataLoader::new_in_memory_with_calibration(
                data_path,
                tof_max_index,
                mz_lower,
                mz_upper,
                im_lookup,
            ),
            false => TimsDataLoader::new_lazy_with_calibration(
                data_path,
                tof_max_index,
                mz_lower,
                mz_upper,
                im_lookup,
            ),
        };

        let pasef_meta = read_pasef_frame_ms_ms_info(data_path).unwrap();

        TimsDatasetDDA { loader, pasef_meta }
    }

    /// Check if the Bruker SDK is being used for index conversion.
    /// Returns false for both Simple and Lookup converters (which are thread-safe).
    pub fn uses_bruker_sdk(&self) -> bool {
        self.loader.uses_bruker_sdk()
    }

    pub fn get_selected_precursors(&self) -> Vec<DDAPrecursor> {
        let precursor_meta = read_dda_precursor_meta(&self.loader.get_data_path()).unwrap();
        let pasef_meta = &self.pasef_meta;

        let precursor_id_to_pasef_meta: BTreeMap<i64, &PasefMsMsMeta> = pasef_meta
            .iter()
            .map(|x| (x.precursor_id as i64, x))
            .collect();

        // go over all precursors and get the precursor meta data
        let result: Vec<_> = precursor_meta
            .iter()
            .map(|precursor| {
                let pasef_meta = precursor_id_to_pasef_meta
                    .get(&precursor.precursor_id)
                    .unwrap();
                DDAPrecursor {
                    frame_id: precursor.precursor_frame_id,
                    precursor_id: precursor.precursor_id,
                    mono_mz: precursor.precursor_mz_monoisotopic,
                    highest_intensity_mz: precursor.precursor_mz_highest_intensity,
                    average_mz: precursor.precursor_mz_average,
                    charge: precursor.precursor_charge,
                    inverse_ion_mobility: self.scan_to_inverse_mobility(
                        precursor.precursor_frame_id as u32,
                        &vec![precursor.precursor_average_scan_number as u32],
                    )[0],
                    collision_energy: pasef_meta.collision_energy,
                    precuror_total_intensity: precursor.precursor_total_intensity,
                    isolation_mz: pasef_meta.isolation_mz,
                    isolation_width: pasef_meta.isolation_width,
                }
            })
            .collect();

        result
    }

    pub fn get_precursor_frames(
        &self,
        min_intensity: f64,
        max_num_peaks: usize,
        num_threads: usize,
    ) -> Vec<TimsFrame> {
        // get all precursor frames
        let meta_data = read_meta_data_sql(&self.loader.get_data_path()).unwrap();

        // get the precursor frames
        let precursor_frames = meta_data.iter().filter(|x| x.ms_ms_type == 0);

        let tims_silce =
            self.get_slice(precursor_frames.map(|x| x.id as u32).collect(), num_threads);

        let result: Vec<_> = tims_silce
            .frames
            .par_iter()
            .map(|frame| {
                frame
                    .filter_ranged(0.0, 2000.0, 0, 2000, 0.0, 5.0, min_intensity, 1e9, 0, i32::MAX)
                    .top_n(max_num_peaks)
            })
            .collect();

        result
    }

    /// Extract MS1 precursor signals for a batch of precursors in parallel.
    ///
    /// For each precursor, extracts:
    /// - XIC (chromatographic profile) from MS1 frames in RT window
    /// - Mobilogram (IM profile)
    /// - Isotope envelope
    /// - Statistical moments (mean, variance, skewness, apex, FWHM) for each dimension
    ///
    /// Uses batched processing to avoid loading all MS1 frames at once.
    ///
    /// # Arguments
    /// * `precursor_coords` - Vector of precursor coordinates (id, mz, rt_sec, mobility, charge)
    /// * `rt_window_sec` - RT window in seconds (total width)
    /// * `mz_tol_ppm` - m/z tolerance in ppm
    /// * `im_window` - IM window in 1/K0 units (total width)
    /// * `n_isotopes` - Number of isotope peaks to extract
    /// * `num_threads` - Number of threads for parallel processing
    ///
    /// # Returns
    /// Vector of PrecursorMS1Signal, one per input precursor
    pub fn extract_precursor_ms1_signals(
        &self,
        precursor_coords: Vec<PrecursorCoord>,
        rt_window_sec: f64,
        mz_tol_ppm: f64,
        im_window: f64,
        n_isotopes: usize,
        num_threads: usize,
    ) -> Vec<PrecursorMS1Signal> {
        if precursor_coords.is_empty() {
            return Vec::new();
        }

        // Get frame metadata
        let meta_data = read_meta_data_sql(&self.loader.get_data_path()).unwrap();

        // Get all MS1 frame info sorted by time
        let mut ms1_frame_info: Vec<(u32, f64)> = meta_data
            .iter()
            .filter(|f| f.ms_ms_type == 0)
            .map(|f| (f.id as u32, f.time))
            .collect();
        ms1_frame_info.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let ms1_times: Vec<f64> = ms1_frame_info.iter().map(|(_, t)| *t).collect();

        // Sort precursors by RT for batched processing
        let mut sorted_coords: Vec<(usize, &PrecursorCoord)> = precursor_coords
            .iter()
            .enumerate()
            .collect();
        sorted_coords.sort_by(|a, b| a.1.rt_seconds.partial_cmp(&b.1.rt_seconds).unwrap());

        // Process in RT batches (5 minute chunks = 300 sec)
        let batch_size_sec = 300.0;
        let mut results: Vec<(usize, PrecursorMS1Signal)> = Vec::with_capacity(precursor_coords.len());

        let mut batch_start = 0;
        while batch_start < sorted_coords.len() {
            // Find batch end (all precursors within batch_size_sec of first)
            let batch_rt_start = sorted_coords[batch_start].1.rt_seconds;
            let batch_rt_end = batch_rt_start + batch_size_sec;

            let mut batch_end = batch_start;
            while batch_end < sorted_coords.len() && sorted_coords[batch_end].1.rt_seconds < batch_rt_end {
                batch_end += 1;
            }

            // Determine MS1 frames needed for this batch (with RT window margin)
            let frame_rt_min = batch_rt_start - rt_window_sec;
            let frame_rt_max = batch_rt_end + rt_window_sec;

            let frame_start_idx = ms1_times.partition_point(|t| *t < frame_rt_min);
            let frame_end_idx = ms1_times.partition_point(|t| *t <= frame_rt_max);

            // Load frames for this batch
            let batch_frame_ids: Vec<u32> = ms1_frame_info[frame_start_idx..frame_end_idx]
                .iter()
                .map(|(id, _)| *id)
                .collect();

            let batch_frames = if !batch_frame_ids.is_empty() {
                self.loader.get_slice(batch_frame_ids, num_threads)
            } else {
                TimsSlice { frames: Vec::new() }
            };

            let batch_times: Vec<f64> = ms1_times[frame_start_idx..frame_end_idx].to_vec();

            // Process precursors in this batch in parallel
            let batch_coords = &sorted_coords[batch_start..batch_end];

            let pool = ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap();

            let batch_results: Vec<(usize, PrecursorMS1Signal)> = pool.install(|| {
                batch_coords.par_iter().map(|(orig_idx, coord)| {
                    let signal = Self::extract_single_precursor(
                        coord,
                        &batch_frames.frames,
                        &batch_times,
                        rt_window_sec,
                        mz_tol_ppm,
                        im_window,
                        n_isotopes,
                    );
                    (*orig_idx, signal)
                }).collect()
            });

            results.extend(batch_results);
            batch_start = batch_end;
        }

        // Restore original order
        results.sort_by_key(|(idx, _)| *idx);
        results.into_iter().map(|(_, signal)| signal).collect()
    }

    /// Extract MS1 signal for a single precursor from pre-loaded frames
    fn extract_single_precursor(
        coord: &PrecursorCoord,
        frames: &[TimsFrame],
        frame_times: &[f64],
        rt_window_sec: f64,
        mz_tol_ppm: f64,
        im_window: f64,
        n_isotopes: usize,
    ) -> PrecursorMS1Signal {
        let rt_sec = coord.rt_seconds;
        let mz_tol = coord.mz * mz_tol_ppm / 1e6;

        // Binary search to find RT window bounds within batch frames
        let rt_min = rt_sec - rt_window_sec / 2.0;
        let rt_max = rt_sec + rt_window_sec / 2.0;

        let start_idx = frame_times.partition_point(|t| *t < rt_min);
        let end_idx = frame_times.partition_point(|t| *t <= rt_max);

        // Calculate isotope m/z values starting from MONOISOTOPIC (M+0 through M+3)
        let isotope_spacing = 1.003355 / (coord.charge.max(1) as f64);
        let n_isotopes_to_extract = 4.min(n_isotopes); // M+0 to M+3 only
        let isotope_mz_values: Vec<f64> = (0..n_isotopes_to_extract)
            .map(|i| coord.mono_mz + (i as f64) * isotope_spacing)
            .collect();

        // m/z ranges:
        // - XIC/mobilogram: use coord.mz (largest_peak_mz) ± tolerance (single peak, no isotope summing)
        // - Isotopes: use mono_mz-based isotope positions
        let xic_mz_min = coord.mz - mz_tol;
        let xic_mz_max = coord.mz + mz_tol;
        let iso_mz_min = coord.mono_mz - mz_tol;
        let iso_mz_max = coord.mono_mz + ((n_isotopes_to_extract - 1) as f64) * isotope_spacing + mz_tol;

        // IM range
        let im_min = coord.mobility - im_window / 2.0;
        let im_max = coord.mobility + im_window / 2.0;

        // Accumulators
        let n_frames = end_idx.saturating_sub(start_idx);
        let mut rt_coords = Vec::with_capacity(n_frames);
        let mut rt_intensities = Vec::with_capacity(n_frames);
        let mut im_dict: HashMap<i64, f64> = HashMap::new();
        let mut isotope_intensity = vec![0.0f64; n_isotopes];  // Keep original size for output

        // Accumulators for raw 2D data (merged from all frames)
        let mut raw_rt = Vec::new();
        let mut raw_mz = Vec::new();
        let mut raw_mobility = Vec::new();
        let mut raw_intensity = Vec::new();

        // Extract from each MS1 frame in the RT window
        for idx in start_idx..end_idx.min(frames.len()) {
            let frame_time = frame_times[idx];
            let frame = &frames[idx];

            // === XIC and Mobilogram: Extract from SINGLE m/z (largest_peak_mz) only ===
            let xic_filtered = frame.filter_ranged(
                xic_mz_min, xic_mz_max,
                0, 1000,
                im_min, im_max,
                0.0, 1e9,
                0, i32::MAX,
            );

            // XIC: intensity at this RT from the targeted m/z only
            let xic_intensity: f64 = xic_filtered.ims_frame.intensity.iter().sum();
            rt_coords.push(frame_time);
            rt_intensities.push(xic_intensity);

            // Mobilogram: accumulate from targeted m/z only
            for (mob, inten) in xic_filtered.ims_frame.mobility.iter().zip(xic_filtered.ims_frame.intensity.iter()) {
                let mob_bin = (*mob * 1000.0).round() as i64;
                *im_dict.entry(mob_bin).or_insert(0.0) += *inten;
            }

            // === Isotope envelope: Extract M+0 through M+3 from mono_mz ===
            let iso_filtered = frame.filter_ranged(
                iso_mz_min, iso_mz_max,
                0, 1000,
                im_min, im_max,
                0.0, 1e9,
                0, i32::MAX,
            );

            for (iso_idx, iso_mz) in isotope_mz_values.iter().enumerate() {
                let iso_peak_min = iso_mz - mz_tol;
                let iso_peak_max = iso_mz + mz_tol;
                let iso_intensity_sum: f64 = iso_filtered.ims_frame.mz.iter()
                    .zip(iso_filtered.ims_frame.intensity.iter())
                    .filter(|(mz, _)| **mz >= iso_peak_min && **mz <= iso_peak_max)
                    .map(|(_, i)| *i)
                    .sum();
                isotope_intensity[iso_idx] += iso_intensity_sum;
            }

            // Raw 2D data: store all peaks from isotope range
            let n_peaks = iso_filtered.ims_frame.mz.len();
            for i in 0..n_peaks {
                raw_rt.push(frame_time);
                raw_mz.push(iso_filtered.ims_frame.mz[i]);
                raw_mobility.push(iso_filtered.ims_frame.mobility[i]);
                raw_intensity.push(iso_filtered.ims_frame.intensity[i]);
            }
        }

        // Convert mobilogram accumulator to sorted arrays
        let mut im_entries: Vec<(i64, f64)> = im_dict.into_iter().collect();
        im_entries.sort_by_key(|(k, _)| *k);
        let im_coords: Vec<f64> = im_entries.iter().map(|(k, _)| *k as f64 / 1000.0).collect();
        let im_intensities: Vec<f64> = im_entries.iter().map(|(_, v)| *v).collect();

        // Calculate moments
        let rt_moments = SignalMoments::from_signal(&rt_coords, &rt_intensities);
        let im_moments = SignalMoments::from_signal(&im_coords, &im_intensities);
        let mz_moments = SignalMoments::from_signal(&isotope_mz_values, &isotope_intensity);

        PrecursorMS1Signal {
            precursor_id: coord.precursor_id,
            rt_coords,
            rt_intensities,
            rt_moments,
            im_coords,
            im_intensities,
            im_moments,
            isotope_mz: isotope_mz_values,
            isotope_intensity,
            mz_moments,
            raw_rt,
            raw_mz,
            raw_mobility,
            raw_intensity,
        }
    }

    pub fn get_pasef_frame_ms_ms_info(&self) -> Vec<PasefMsMsMeta> {
        read_pasef_frame_ms_ms_info(&self.loader.get_data_path()).unwrap()
    }

    /// Get the fragment spectra for all PASEF selected precursors
    pub fn get_pasef_fragments(&self, num_threads: usize) -> Vec<PASEFDDAFragment> {
        // Delegate to the filtered version with no filter (all precursors)
        self.get_pasef_fragments_for_precursors(None, num_threads)
    }

    /// Get fragment spectra for specific precursor IDs only.
    /// If precursor_ids is None, returns all fragments (same as get_pasef_fragments).
    /// This is more memory-efficient for batched processing.
    pub fn get_pasef_fragments_for_precursors(
        &self,
        precursor_ids: Option<&[u32]>,
        num_threads: usize,
    ) -> Vec<PASEFDDAFragment> {
        // extract fragment spectra information
        let pasef_info = self.get_pasef_frame_ms_ms_info();

        // Filter to requested precursor IDs if specified
        let filtered_pasef_info: Vec<&PasefMsMsMeta> = match precursor_ids {
            Some(ids) => {
                // Create a HashSet for O(1) lookup
                let id_set: std::collections::HashSet<u32> = ids.iter().copied().collect();
                pasef_info.iter()
                    .filter(|info| id_set.contains(&(info.precursor_id as u32)))
                    .collect()
            }
            None => pasef_info.iter().collect(),
        };

        // Note: The Bruker SDK is NOT thread-safe, so we must use sequential iteration
        // when the SDK is being used for index conversion.
        let uses_bruker_sdk = self.loader.uses_bruker_sdk();

        // Helper closure to process a single PASEF fragment
        let process_fragment = |pasef_info: &PasefMsMsMeta| -> PASEFDDAFragment {
            // get the frame
            let frame = self.loader.get_frame(pasef_info.frame_id as u32);

            // get five percent of the scan range
            let scan_margin = (pasef_info.scan_num_end - pasef_info.scan_num_begin) / 20;

            // get the fragment spectrum by scan range
            let filtered_frame = frame.filter_ranged(
                0.0,
                2000.0,
                (pasef_info.scan_num_begin - scan_margin) as i32,
                (pasef_info.scan_num_end + scan_margin) as i32,
                0.0,
                5.0,
                0.0,
                1e9,
                0,
                i32::MAX,
            );

            PASEFDDAFragment {
                frame_id: pasef_info.frame_id as u32,
                precursor_id: pasef_info.precursor_id as u32,
                collision_energy: pasef_info.collision_energy,
                // flatten the spectrum
                selected_fragment: filtered_frame,
            }
        };

        if uses_bruker_sdk {
            // Sequential processing when using Bruker SDK (not thread-safe)
            filtered_pasef_info.iter().map(|info| process_fragment(info)).collect()
        } else {
            // Parallel processing when using simple index converter (thread-safe)
            let pool = ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap();

            pool.install(|| {
                filtered_pasef_info.par_iter().map(|info| process_fragment(info)).collect()
            })
        }
    }

    /// Get preprocessed PASEF fragments ready for database search.
    /// This method performs parallel processing of all fragment spectra, including:
    /// - Flattening frames across ion mobility dimension
    /// - Deisotoping (optional)
    /// - Filtering to top N peaks
    /// - Computing inverse mobility along scan marginal
    ///
    /// # Arguments
    /// * `dataset_name` - Name of the dataset for generating spec_ids
    /// * `config` - Spectrum processing configuration
    /// * `num_threads` - Number of threads to use for parallel processing
    ///
    /// # Returns
    /// Vector of preprocessed spectra ready for Sage search
    pub fn get_preprocessed_pasef_fragments(
        &self,
        dataset_name: &str,
        config: SpectrumProcessingConfig,
        num_threads: usize,
    ) -> Vec<PreprocessedSpectrum> {
        // Step 1: Get raw PASEF fragments info
        let pasef_info = self.get_pasef_frame_ms_ms_info();

        // Step 2: Get precursor metadata
        let precursor_meta = read_dda_precursor_meta(&self.loader.get_data_path()).unwrap_or_default();
        let frame_meta = read_meta_data_sql(&self.loader.get_data_path()).unwrap_or_default();

        // Create lookup maps
        let precursor_map: BTreeMap<i64, &DDAPrecursorMeta> = precursor_meta
            .iter()
            .map(|p| (p.precursor_id, p))
            .collect();

        let frame_time_map: BTreeMap<i64, f64> = frame_meta
            .iter()
            .map(|f| (f.id, f.time / 60.0))  // Convert to minutes
            .collect();

        // Step 3: Group PASEF info by precursor_id to aggregate re-fragmented precursors
        // This matches Python's groupby('precursor_id').agg({'raw_data': 'sum', ...})
        let mut pasef_by_precursor: BTreeMap<i64, Vec<&PasefMsMsMeta>> = BTreeMap::new();
        for info in &pasef_info {
            pasef_by_precursor
                .entry(info.precursor_id)
                .or_insert_with(Vec::new)
                .push(info);
        }

        // Step 4: Build fragment data with aggregation (merge frames for same precursor)
        // Note: The Bruker SDK is NOT thread-safe, so we must use sequential iteration
        // when the SDK is being used for index conversion.
        let uses_bruker_sdk = self.loader.uses_bruker_sdk();

        // Helper closure to process a single precursor
        let process_precursor = |(precursor_id, pasef_infos): (&i64, &Vec<&PasefMsMsMeta>)| -> Option<PASEFFragmentData> {
            // Get precursor metadata
            let precursor = precursor_map.get(precursor_id)?;

            // Use first PASEF info for metadata (matches Python's 'first')
            let first_pasef = pasef_infos.first()?;

            // Get retention time from first frame
            let scan_start_time = frame_time_map.get(&first_pasef.frame_id).copied().unwrap_or(0.0);

            // Collect and merge all frames for this precursor (matches Python's 'sum')
            let mut combined_scan = Vec::new();
            let mut combined_mobility = Vec::new();
            let mut combined_tof = Vec::new();
            let mut combined_mz = Vec::new();
            let mut combined_intensity = Vec::new();

            for pasef_info in pasef_infos {
                // Get the frame and filter by scan range
                let frame = self.loader.get_frame(pasef_info.frame_id as u32);

                // Get five percent of the scan range for margin
                let scan_margin = (pasef_info.scan_num_end - pasef_info.scan_num_begin) / 20;

                // Filter frame by scan range
                let filtered_frame = frame.filter_ranged(
                    0.0,
                    2000.0,
                    (pasef_info.scan_num_begin - scan_margin) as i32,
                    (pasef_info.scan_num_end + scan_margin) as i32,
                    0.0,
                    5.0,
                    0.0,
                    1e9,
                    0,
                    i32::MAX,
                );

                // Append data from this frame (merge/sum behavior)
                combined_scan.extend(filtered_frame.scan.iter());
                combined_mobility.extend(filtered_frame.ims_frame.mobility.iter());
                combined_tof.extend(filtered_frame.tof.iter());
                combined_mz.extend(filtered_frame.ims_frame.mz.iter());
                combined_intensity.extend(filtered_frame.ims_frame.intensity.iter());
            }

            if combined_mz.is_empty() {
                return None;
            }

            // Determine precursor m/z (prefer monoisotopic, fallback to highest intensity)
            let precursor_mz = precursor.precursor_mz_monoisotopic
                .unwrap_or(precursor.precursor_mz_highest_intensity);

            Some(PASEFFragmentData {
                frame_id: first_pasef.frame_id as u32,
                precursor_id: *precursor_id as u32,
                collision_energy: first_pasef.collision_energy,
                scan_start_time,
                scan: combined_scan,
                mobility: combined_mobility,
                tof: combined_tof,
                mz: combined_mz,
                intensity: combined_intensity,
                precursor_mz,
                precursor_charge: precursor.precursor_charge.map(|c| c as i32),
                precursor_intensity: precursor.precursor_total_intensity,
                isolation_mz: first_pasef.isolation_mz,
                isolation_width: first_pasef.isolation_width,
            })
        };

        let fragment_data: Vec<PASEFFragmentData> = if uses_bruker_sdk {
            // Sequential processing when using Bruker SDK (not thread-safe)
            pasef_by_precursor
                .iter()
                .filter_map(process_precursor)
                .collect()
        } else {
            // Parallel processing when using simple index converter (thread-safe)
            let pool = ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap();

            pool.install(|| {
                pasef_by_precursor
                    .par_iter()
                    .filter_map(|item| process_precursor(item))
                    .collect()
            })
        };

        // Step 5: Process all fragments in parallel using the batch processor
        process_pasef_fragments_batch(fragment_data, dataset_name, &config, num_threads)
    }

    pub fn sample_pasef_fragment_random(
        &self,
        target_scan_apex: i32,
        experiment_max_scan: i32,
    ) -> TimsFrame {
        let pasef_meta = &self.pasef_meta;
        let random_index = rand::random::<usize>() % pasef_meta.len();
        let pasef_info = &pasef_meta[random_index];
        
        // get the frame
        let frame = self.loader.get_frame(pasef_info.frame_id as u32);
        
        // get five percent of the scan range
        let scan_margin = (pasef_info.scan_num_end - pasef_info.scan_num_begin) / 20;
        
        // get the fragment spectrum by scan range
        let mut filtered = frame.filter_ranged(
            0.0,
            2000.0,
            (pasef_info.scan_num_begin - scan_margin) as i32,
            (pasef_info.scan_num_end + scan_margin) as i32,
            0.0,
            5.0,
            0.0,
            1e9,
            0,
            i32::MAX,
        );

        // Safety check
        if filtered.scan.is_empty() {
            return filtered;
        }

        // Compute median scan
        let mut scan_copy = filtered.scan.clone();
        scan_copy.sort_unstable();
        let median_scan = scan_copy[scan_copy.len() / 2];

        // Compute shift
        let scan_shift = target_scan_apex - median_scan;

        // Apply shift to scan values
        for s in filtered.scan.iter_mut() {
            *s += scan_shift;
        }

        // Refilter to clip shifted scans that fall outside valid bounds
        let re_filtered = filtered.filter_ranged(
            0.0,
            2000.0,
            0,
            experiment_max_scan,
            0.0,
            5.0,
            0.0,
            1e9,
            0,
            i32::MAX,
        );

        re_filtered
    }
    
    pub fn sample_pasef_fragments_random(
        &self,
        target_scan_apex_values: Vec<i32>,
        experiment_max_scan: i32,
    ) -> TimsFrame {

        // return empty frame is target_scan_apex_values is empty
        if target_scan_apex_values.is_empty() {
            return TimsFrame {
                frame_id: 0, // Replace with a suitable default value
                ms_type: MsType::FragmentDda,
                scan: Vec::new(),
                tof: Vec::new(),
                ims_frame: ImsFrame::default(), // Uses the default implementation for `ImsFrame`
            }
        }
        
        let mut pasef_frames = Vec::new();
        
        for target_scan_apex in target_scan_apex_values {
            let pasef_frame = self.sample_pasef_fragment_random(target_scan_apex, experiment_max_scan);
            pasef_frames.push(pasef_frame);
        }
        
        // create combined frame by summing the frame structures, they override add
        let mut combined_frame = pasef_frames[0].clone();
        
        for frame in pasef_frames.iter().skip(1) {
            combined_frame = combined_frame + frame.clone();
        }

        // re-calculate ion mobility
        let im_values = self.scan_to_inverse_mobility(
            combined_frame.frame_id as u32,
            &combined_frame.scan.iter().map(|x| *x as u32).collect(),
        );

        // Update the inverse mobility values
        combined_frame.ims_frame.mobility = std::sync::Arc::new(im_values);
        
        combined_frame
    }

    pub fn sample_precursor_signal(
        &self,
        num_frames: usize,
        max_intensity: f64,
        take_probability: f64,
    ) -> TimsFrame {
        // get all precursor frames
        let meta_data = read_meta_data_sql(&self.loader.get_data_path()).unwrap();
        let precursor_frames = meta_data.iter().filter(|x| x.ms_ms_type == 0);

        // randomly sample num_frames
        let mut rng = rand::thread_rng();
        let mut sampled_frames: Vec<TimsFrame> = Vec::new();

        // go through each frame and sample the data
        for frame in precursor_frames.choose_multiple(&mut rng, num_frames) {
            let frame_id = frame.id;
            let frame_data = self
                .loader
                .get_frame(frame_id as u32)
                .filter_ranged(0.0, 2000.0, 0, 1000, 0.0, 5.0, 1.0, max_intensity, 0, i32::MAX)
                .generate_random_sample(take_probability);
            sampled_frames.push(frame_data);
        }

        // get the first frame
        let mut sampled_frame = sampled_frames.remove(0);

        // sum all the other frames to the first frame
        for frame in sampled_frames {
            sampled_frame = sampled_frame + frame;
        }

        sampled_frame
    }
}

impl TimsData for TimsDatasetDDA {
    fn get_frame(&self, frame_id: u32) -> TimsFrame {
        self.loader.get_frame(frame_id)
    }

    fn get_raw_frame(&self, frame_id: u32) -> RawTimsFrame {
        self.loader.get_raw_frame(frame_id)
    }

    fn get_slice(&self, frame_ids: Vec<u32>, num_threads: usize) -> TimsSlice {
        self.loader.get_slice(frame_ids, num_threads)
    }

    fn get_acquisition_mode(&self) -> AcquisitionMode {
        self.loader.get_acquisition_mode().clone()
    }

    fn get_frame_count(&self) -> i32 {
        self.loader.get_frame_count()
    }

    fn get_data_path(&self) -> &str {
        &self.loader.get_data_path()
    }
}

impl IndexConverter for TimsDatasetDDA {
    fn tof_to_mz(&self, frame_id: u32, tof_values: &Vec<u32>) -> Vec<f64> {
        self.loader
            .get_index_converter()
            .tof_to_mz(frame_id, tof_values)
    }

    fn mz_to_tof(&self, frame_id: u32, mz_values: &Vec<f64>) -> Vec<u32> {
        self.loader
            .get_index_converter()
            .mz_to_tof(frame_id, mz_values)
    }

    fn scan_to_inverse_mobility(&self, frame_id: u32, scan_values: &Vec<u32>) -> Vec<f64> {
        self.loader
            .get_index_converter()
            .scan_to_inverse_mobility(frame_id, scan_values)
    }

    fn inverse_mobility_to_scan(
        &self,
        frame_id: u32,
        inverse_mobility_values: &Vec<f64>,
    ) -> Vec<u32> {
        self.loader
            .get_index_converter()
            .inverse_mobility_to_scan(frame_id, inverse_mobility_values)
    }
}
