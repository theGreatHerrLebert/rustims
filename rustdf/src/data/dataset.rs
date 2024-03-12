use mscore::timstof::frame::TimsFrame;
use mscore::timstof::slice::TimsSlice;
use crate::data::handle::{TimsDataHandle, TimsData, AcquisitionMode};

pub struct TimsDataset {
    pub handle: TimsDataHandle,
}

impl TimsDataset {
    pub fn new(bruker_lib_path: &str, data_path: &str) -> Self {
        let handle = TimsDataHandle::new(bruker_lib_path, data_path).unwrap();
        TimsDataset { handle }
    }
}

impl TimsData for TimsDataset {
    // Get a frame by its id
    fn get_frame(&self, frame_id: u32) -> TimsFrame {
        self.handle.get_frame(frame_id).unwrap()
    }
    // Get a collection of frames by their ids
    fn get_slice(&self, frame_ids: Vec<u32>) -> TimsSlice {
        self.handle.get_tims_slice(frame_ids)
    }
    // Get the acquisition mode, DDA or DIA
    fn get_aquisition_mode(&self) -> AcquisitionMode {
        self.handle.acquisition_mode.clone()
    }
    // Get total number of frames in the dataset
    fn get_frame_count(&self) -> i32 {
        self.handle.get_frame_count()
    }
    // Get the path to the data
    fn get_data_path(&self) -> &str {
        &self.handle.data_path
    }
    // Get the path to the bruker library for raw data reading from tdf files
    fn get_bruker_lib_path(&self) -> &str {
        &self.handle.bruker_lib_path
    }
    // convert TOF values to m/z given a valid data handle and frame id
    fn tof_to_mz(&self, frame_id: u32, tof_values: &Vec<u32>) -> Vec<f64> {
        self.handle.tof_to_mz(frame_id, tof_values)
    }
    // convert m/z values to TOF values given a valid data handle and frame id
    fn mz_to_tof(&self, frame_id: u32, mz_values: &Vec<f64>) -> Vec<u32> {
        self.handle.mz_to_tof(frame_id, mz_values)
    }
    // convert inverse mobility values to scan values given a valid data handle and frame id
    fn scan_to_inverse_mobility(&self, frame_id: u32, scan_values: &Vec<i32>) -> Vec<f64> {
        self.handle.scan_to_inverse_mobility(frame_id, scan_values)
    }
    // convert scan values to inverse mobility values given a valid data handle and frame id
    fn inverse_mobility_to_scan(&self, frame_id: u32, inverse_mobility_values: &Vec<f64>) -> Vec<i32> {
        self.handle.inverse_mobility_to_scan(frame_id, inverse_mobility_values)
    }
}