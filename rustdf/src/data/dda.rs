use crate::data::acquisition::AcquisitionMode;
use crate::data::handle::{IndexConverter, TimsData, TimsDataLoader};
use crate::data::meta::{read_dda_precursor_meta, read_global_meta_sql, read_meta_data_sql, read_pasef_frame_ms_ms_info, DDAPrecursor, PasefMsMsMeta};
use mscore::timstof::frame::{ImsFrame, RawTimsFrame, TimsFrame};
use mscore::timstof::slice::TimsSlice;
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
                    .filter_ranged(0.0, 2000.0, 0, 2000, 0.0, 5.0, min_intensity, 1e9)
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

        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let filtered_frames = pool.install(|| {
            let result: Vec<_> = pasef_info
                .par_iter()
                .map(|pasef_info| {
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
                    );

                    PASEFDDAFragment {
                        frame_id: pasef_info.frame_id as u32,
                        precursor_id: pasef_info.precursor_id as u32,
                        collision_energy: pasef_info.collision_energy,
                        // flatten the spectrum
                        selected_fragment: filtered_frame,
                    }
                })
                .collect();

            result
        });

        filtered_frames
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
        let mut re_filtered = filtered.filter_ranged(
            0.0,
            2000.0,
            0,
            experiment_max_scan,
            0.0,
            5.0,
            0.0,
            1e9,
        );
        
        // re-calculate ion mobility
        let im_values = self.scan_to_inverse_mobility(
            pasef_info.frame_id as u32,
            &re_filtered.scan.iter().map(|x| *x as u32).collect(),
        );
        
        // Update the inverse mobility values
        re_filtered.ims_frame.mobility = im_values;
        
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
                .filter_ranged(0.0, 2000.0, 0, 1000, 0.0, 5.0, 1.0, max_intensity)
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
