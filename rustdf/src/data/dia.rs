use mscore::{TimsFrame, TimsSlice};
use crate::data::handle::{AcquisitionMode, TimsData, TimsDataHandle};

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
    fn get_aquisition_mode(&self) -> AcquisitionMode {
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
}