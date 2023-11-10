use mscore::{TimsFrame, TimsSlice};
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
}