use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::mz_spectrum::{MzSpectrum};
use crate::tims_frame::{TimsFrame, TimsFrameVectorized};

#[derive(Clone)]
pub struct TimsSlice {
    pub frames: Vec<TimsFrame>,
}

impl TimsSlice {

    pub fn new(frames: Vec<TimsFrame>) -> Self {
        TimsSlice { frames }
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, intensity_min: f64, num_threads: usize) -> TimsSlice {

        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap(); // Set to the desired number of threads

        // Use the thread pool
        let filtered_frames = pool.install(|| {
            let result: Vec<_> =  self.frames.par_iter()
                .map(|f| f.filter_ranged(mz_min, mz_max, scan_min, scan_max, intensity_min))
                .collect();
            result
        });

        TimsSlice { frames: filtered_frames }
    }

    pub fn to_windows(&self, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64, num_threads: usize) -> Vec<MzSpectrum> {
        // Create a thread pool
        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap(); // Set to the desired number of threads

        // Use the thread pool
        let windows = pool.install(|| {
            let windows: Vec<_> = self.frames.par_iter()
                .flat_map( | frame | frame.to_windows(window_length, overlapping, min_peaks, min_intensity))
                .collect();
            windows
        });

        windows
    }
}

#[derive(Clone)]
pub struct TimsSliceVectorized {
    pub frames: Vec<TimsFrameVectorized>,
}