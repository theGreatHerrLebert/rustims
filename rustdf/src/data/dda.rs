use crate::data::handle::TimsDataHandle;
use crate::data::meta::{DDAPrecursorMeta, PasefMsMsMeta, read_dda_precursor_meta, read_pasef_frame_ms_ms_info};

pub struct TimsDataHandleDda {
    handle: TimsDataHandle,
}

impl TimsDataHandleDda {
    pub fn new(bruker_lib_path: &str, data_path: &str) -> Result<TimsDataHandleDda, Box<dyn std::error::Error>> {
        let handle = TimsDataHandle::new(bruker_lib_path, data_path)?;
        Ok(TimsDataHandleDda {
            handle,
        })
    }
    pub fn get_selected_precursors(&self) -> Vec<DDAPrecursorMeta> {
        read_dda_precursor_meta(&self.handle.data_path).unwrap()
    }

    pub fn get_pasef_frame_ms_ms_info(&self) -> Vec<PasefMsMsMeta> {
        read_pasef_frame_ms_ms_info(&self.handle.data_path).unwrap()
    }
}