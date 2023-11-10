use crate::data::handle::TimsDataHandle;

pub struct TimsDatasetDIA {
    pub handle: TimsDataHandle,
}

impl TimsDatasetDIA {
    pub fn new(bruker_lib_path: &str, data_path: &str) -> Self {
        let handle = TimsDataHandle::new(bruker_lib_path, data_path).unwrap();
        TimsDatasetDIA { handle }
    }
}