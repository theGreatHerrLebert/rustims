use std::collections::BTreeMap;
use mscore::timstof::frame::{RawTimsFrame, TimsFrame};
use mscore::timstof::slice::TimsSlice;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::data::acquisition::AcquisitionMode;
use crate::data::handle::{IndexConverter, TimsData, TimsDataLoader};
use crate::data::meta::{PasefMsMsMeta, read_dda_precursor_meta, read_pasef_frame_ms_ms_info, read_global_meta_sql, read_meta_data_sql, DDAPrecursor};

#[derive(Clone)]
pub struct PASEFDDAFragment {
    pub frame_id: u32,
    pub precursor_id: u32,
    pub collision_energy: f64,
    pub selected_fragment: TimsFrame,
}

pub struct TimsDatasetDDA {
    pub loader: TimsDataLoader,
}

impl TimsDatasetDDA {

    pub fn new(bruker_lib_path: &str, data_path: &str, in_memory: bool, use_bruker_sdk: bool) -> Self {

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
            true => TimsDataLoader::new_in_memory(bruker_lib_path, data_path, use_bruker_sdk, scan_max_index, im_lower, im_upper, tof_max_index, mz_lower, mz_upper),
            false => TimsDataLoader::new_lazy(bruker_lib_path, data_path, use_bruker_sdk, scan_max_index, im_lower, im_upper, tof_max_index, mz_lower, mz_upper),
        };
        TimsDatasetDDA { loader }
    }

    /*
    #[derive(Debug, Clone)]
pub struct DDAPrecursor {
    pub frame_id: i64,
    pub precursor_id: i64,
    pub mono_mz: f64,
    pub highest_intensity_mz: f64,
    pub average_mz: f64,
    pub charge: Option<i64>,
    pub inverse_ion_mobility: f64,
    pub collision_energy: f64,
    pub precuror_total_intensity: f64,
    pub isolation_mz: f64,
    pub isolation_width: f64,
}
     */



    pub fn get_selected_precursors(&self) -> Vec<DDAPrecursor> {
        let precursor_meta = read_dda_precursor_meta(&self.loader.get_data_path()).unwrap();
        let pasef_meta = self.get_pasef_frame_ms_ms_info();

        let precursor_id_to_pasef_meta: BTreeMap<i64, &PasefMsMsMeta> = pasef_meta.iter().map(|x| (x.precursor_id as i64, x)).collect();

        // go over all precursors and get the precursor meta data
        let result: Vec<_> = precursor_meta.iter().map(|precursor| {
            let pasef_meta = precursor_id_to_pasef_meta.get(&precursor.precursor_id).unwrap();
            DDAPrecursor {
                frame_id: precursor.precursor_frame_id,
                precursor_id: precursor.precursor_id,
                mono_mz: precursor.precursor_mz_monoisotopic,
                highest_intensity_mz: precursor.precursor_mz_highest_intensity,
                average_mz: precursor.precursor_mz_average,
                charge: precursor.precursor_charge,
                inverse_ion_mobility: self.scan_to_inverse_mobility(precursor.precursor_frame_id as u32, &vec![precursor.precursor_average_scan_number as u32])[0],
                collision_energy: pasef_meta.collision_energy,
                precuror_total_intensity: precursor.precursor_total_intensity,
                isolation_mz: pasef_meta.isolation_mz,
                isolation_width: pasef_meta.isolation_width,
            }
        }).collect();

        result
    }

    pub fn get_precursor_frames(&self, min_intensity: f64, max_num_peaks: usize, num_threads: usize) -> Vec<TimsFrame> {
        // get all precursor frames
        let meta_data = read_meta_data_sql(&self.loader.get_data_path()).unwrap();

        // get the precursor frames
        let precursor_frames = meta_data.iter().filter(|x| x.ms_ms_type == 0);

        let tims_silce = self.get_slice(precursor_frames.map(|x| x.id as u32).collect(), num_threads);

        let result: Vec<_> = tims_silce.frames.par_iter().map(|frame| {
            frame.filter_ranged(0.0, 2000.0, 0, 2000, 0.0, 5.0, min_intensity, 1e9).top_n(max_num_peaks)
        }).collect();

        result
    }

    pub fn get_pasef_frame_ms_ms_info(&self) -> Vec<PasefMsMsMeta> {
        read_pasef_frame_ms_ms_info(&self.loader.get_data_path()).unwrap()
    }

    /// Get the fragment spectra for all PASEF selected precursors
    pub fn get_pasef_fragments(&self, num_threads: usize) -> Vec<PASEFDDAFragment> {
        // extract fragment spectra information
        let pasef_info = self.get_pasef_frame_ms_ms_info();

        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();

        let filtered_frames = pool.install(|| {

            let result: Vec<_> = pasef_info.par_iter().map(|pasef_info| {

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
            }).collect();

            result
        });

        filtered_frames
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
        self.loader.get_index_converter().tof_to_mz(frame_id, tof_values)
    }

    fn mz_to_tof(&self, frame_id: u32, mz_values: &Vec<f64>) -> Vec<u32> {
        self.loader.get_index_converter().mz_to_tof(frame_id, mz_values)
    }

    fn scan_to_inverse_mobility(&self, frame_id: u32, scan_values: &Vec<u32>) -> Vec<f64> {
        self.loader.get_index_converter().scan_to_inverse_mobility(frame_id, scan_values)
    }

    fn inverse_mobility_to_scan(&self, frame_id: u32, inverse_mobility_values: &Vec<f64>) -> Vec<u32> {
        self.loader.get_index_converter().inverse_mobility_to_scan(frame_id, inverse_mobility_values)
    }
}