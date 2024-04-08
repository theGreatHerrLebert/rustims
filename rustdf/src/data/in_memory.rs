use crate::data::handle::TimsDataHandle;
use crate::data::meta::{FrameMeta, GlobalMetaData, read_global_meta_sql, read_meta_data_sql};

pub struct TimsDatasetInMemory {
    handle: TimsDataHandle,
    pub global_meta_data: GlobalMetaData,
    pub frame_meta_data: Vec<FrameMeta>,
    pub compressed_data: Vec<u8>
}

impl TimsDatasetInMemory {
    pub fn new(bruker_lib_path: &str, data_path: &str) -> Self {
        let handle = TimsDataHandle::new(bruker_lib_path, data_path).unwrap();
        let global_meta_data = read_global_meta_sql(&handle.data_path).unwrap();
        let frame_meta_data = read_meta_data_sql(&handle.data_path).unwrap();
        let compressed_data = handle.read_compressed_data_full();
        TimsDatasetInMemory {
            handle,
            global_meta_data,
            frame_meta_data,
            compressed_data
        }
    }
}