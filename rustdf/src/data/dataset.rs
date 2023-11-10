use crate::data::handle::TimsDataHandle;

pub struct TimsDataset {
    pub handle: TimsDataHandle,
}

impl TimsDataset {
    pub fn new(bruker_lib_path: &str, data_path: &str) -> Self {
        let handle = TimsDataHandle::new(bruker_lib_path, data_path).unwrap();
        TimsDataset { handle }
    }
}