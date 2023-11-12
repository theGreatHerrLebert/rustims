use mscore::{TimsFrame, TimsSlice};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::data::handle::{TimsDataHandle, TimsData, AcquisitionMode};
use crate::data::meta::{DDAPrecursorMeta, PasefMsMsMeta, read_dda_precursor_meta, read_pasef_frame_ms_ms_info};

#[derive(Clone)]
pub struct PASEFDDAFragment {
    pub frame_id: u32,
    pub precursor_id: u32,
    pub selected_fragment: TimsFrame,
}

pub struct TimsDatasetDDA {
    pub handle: TimsDataHandle,
}

impl TimsDatasetDDA {

    pub fn new(bruker_lib_path: &str, data_path: &str) -> Self {
        let handle = TimsDataHandle::new(bruker_lib_path, data_path).unwrap();
        TimsDatasetDDA { handle }
    }

    pub fn get_selected_precursors(&self) -> Vec<DDAPrecursorMeta> {
        read_dda_precursor_meta(&self.handle.data_path).unwrap()
    }

    pub fn get_pasef_frame_ms_ms_info(&self) -> Vec<PasefMsMsMeta> {
        read_pasef_frame_ms_ms_info(&self.handle.data_path).unwrap()
    }

    /// Get the fragment spectra for all PASEF selected precursors
    pub fn get_pasef_fragments(&self, num_threads: usize) -> Vec<PASEFDDAFragment> {
        // extract fragment spectra information
        let pasef_info = self.get_pasef_frame_ms_ms_info();

        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();



        let filtered_frames = pool.install(|| {

            let result: Vec<_> = pasef_info.par_iter().map(|pasef_info| {

                // get the frame
                let frame = self.handle.get_frame(pasef_info.frame_id as u32).unwrap();

                // get the fragment spectrum by scan range
                let filtered_frame = frame.filter_ranged(
                    0.0,
                    2000.0,
                    pasef_info.scan_num_begin as i32,
                    // TODO: +1?
                    pasef_info.scan_num_end as i32 + 1,
                    0.0,
                    5.0,
                    0.0,
                    1e9,
                );

                PASEFDDAFragment {
                    frame_id: pasef_info.frame_id as u32,
                    precursor_id: pasef_info.precursor_id as u32,
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