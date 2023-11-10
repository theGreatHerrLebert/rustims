use mscore::{IndexedMzSpectrum};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::data::handle::TimsDataHandle;
use crate::data::meta::{DDAPrecursorMeta, PasefMsMsMeta, read_dda_precursor_meta, read_pasef_frame_ms_ms_info};

#[derive(Clone)]
pub struct PASEFDDAFragment {
    pub precursor_id: u32,
    pub selected_fragment: IndexedMzSpectrum,
}

pub struct TimsTofDatasetDDA {
    pub handle: TimsDataHandle,
}

impl TimsTofDatasetDDA {

    pub fn new(bruker_lib_path: &str, data_path: &str) -> Result<TimsTofDatasetDDA, Box<dyn std::error::Error>> {
        let handle = TimsDataHandle::new(bruker_lib_path, data_path)?;
        Ok(TimsTofDatasetDDA {
            handle,
        })
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

        // extract fragment spectra in parallel
        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
        let pasef_fragments: Vec<PASEFDDAFragment> = pool.install(|| {

            pasef_info.par_iter().map(|pasef_info| {
                // get the frame
                let frame = self.handle.get_frame(pasef_info.frame_id as u32).unwrap();

                // get the fragment spectrum by scan range
                let filtered_frame = frame.filter_ranged(
                    0.0,
                    2000.0,
                    pasef_info.scan_num_begin as i32,
                    pasef_info.scan_num_end as i32,
                    0.0,
                    5.0,
                    0.0,
                    1e9,
                );

                PASEFDDAFragment {
                    precursor_id: pasef_info.precursor_id as u32,
                    // flatten the spectrum
                    selected_fragment: filtered_frame.to_indexed_mz_spectrum(),
                }
            }).collect()
        });
        pasef_fragments
    }
}