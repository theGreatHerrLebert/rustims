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

#[derive(Clone)]
pub struct PASEFDDAFragment {
    pub frame_id: u32,
    pub precursor_id: u32,
    pub collision_energy: f64,
    pub selected_fragment: TimsFrame,
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

    pub fn get_pasef_frame_ms_ms_info(&self) -> Vec<PasefMsMsMeta> {
        read_pasef_frame_ms_ms_info(&self.loader.get_data_path()).unwrap()
    }

    /// Get the fragment spectra for all PASEF selected precursors
    pub fn get_pasef_fragments(&self, num_threads: usize) -> Vec<PASEFDDAFragment> {
        // extract fragment spectra information
        let pasef_info = self.get_pasef_frame_ms_ms_info();

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
            pasef_info.iter().map(process_fragment).collect()
        } else {
            // Parallel processing when using simple index converter (thread-safe)
            let pool = ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap();

            pool.install(|| {
                pasef_info.par_iter().map(|item| process_fragment(item)).collect()
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
