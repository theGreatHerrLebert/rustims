use mscore::timstof::frame::{RawTimsFrame, TimsFrame};
use mscore::timstof::slice::TimsSlice;
use crate::data::acquisition::AcquisitionMode;
use crate::data::handle::{IndexConverter, TimsData, TimsDataLoader};
use crate::data::meta::{read_global_meta_sql, read_meta_data_sql};

pub struct TimsDatasetDIA {
    pub loader: TimsDataLoader,
}

impl TimsDatasetDIA {
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
        TimsDatasetDIA { loader }
    }
}

impl TimsData for TimsDatasetDIA {
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

impl IndexConverter for TimsDatasetDIA {
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