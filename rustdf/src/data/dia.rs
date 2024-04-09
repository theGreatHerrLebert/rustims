use mscore::timstof::frame::TimsFrame;
use mscore::timstof::slice::TimsSlice;
use crate::data::acquisition::AcquisitionMode;
use crate::data::handle::{TimsData, TimsDataHandle};

pub struct TimsDatasetDIA {
    pub handle: TimsDataHandle,
}

impl TimsDatasetDIA {
    pub fn new(bruker_lib_path: &str, data_path: &str) -> Self {
        let handle = TimsDataHandle::new(bruker_lib_path, data_path).unwrap();
        TimsDatasetDIA { handle }
    }
}

impl TimsData for TimsDatasetDIA {
    fn get_frame(&self, frame_id: u32) -> TimsFrame {
        self.handle.get_frame(frame_id).unwrap()
    }
    fn get_slice(&self, frame_ids: Vec<u32>) -> TimsSlice {
        self.handle.get_tims_slice(frame_ids)
    }
    fn get_acquisition_mode(&self) -> AcquisitionMode {
        self.handle.acquisition_mode.clone()
    }

    fn get_frame_count(&self) -> i32 {
        self.handle.get_frame_count()
    }

    fn get_data_path(&self) -> &str {
        &self.handle.data_path
    }

    fn get_bruker_lib_path(&self) -> &str {
        &self.handle.bruker_lib_path
    }

    fn tof_to_mz(&self, frame_id: u32, tof_values: &Vec<u32>) -> Vec<f64> {
        self.handle.tof_to_mz(frame_id, tof_values)
    }

    fn mz_to_tof(&self, frame_id: u32, mz_values: &Vec<f64>) -> Vec<u32> {
        self.handle.mz_to_tof(frame_id, mz_values)
    }

    fn scan_to_inverse_mobility(&self, frame_id: u32, scan_values: &Vec<i32>) -> Vec<f64> {
        self.handle.scan_to_inverse_mobility(frame_id, scan_values)
    }

    fn inverse_mobility_to_scan(&self, frame_id: u32, inverse_mobility_values: &Vec<f64>) -> Vec<i32> {
        self.handle.inverse_mobility_to_scan(frame_id, inverse_mobility_values)
    }
}