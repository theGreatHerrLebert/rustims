use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::mz_spectrum::{MzSpectrum};
use crate::tims_frame::{TimsFrame, TimsFrameFlat, TimsFrameVectorized};

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

    pub fn flatten(&self) -> TimsFrameFlat {
        let mut frame_ids = Vec::new();
        let mut scans = Vec::new();
        let mut tofs = Vec::new();
        let mut retention_times = Vec::new();
        let mut mobilities = Vec::new();
        let mut mzs = Vec::new();
        let mut intensities = Vec::new();

        for frame in &self.frames {
            let length = frame.scan.len();
            frame_ids.extend(vec![frame.frame_id; length].into_iter());
            scans.extend(frame.scan.clone());
            tofs.extend(frame.tof.clone());
            retention_times.extend(vec![frame.ims_frame.retention_time; length].into_iter());
            mobilities.extend(&frame.ims_frame.mobility);
            mzs.extend(&frame.ims_frame.mz);
            intensities.extend(&frame.ims_frame.intensity);
        }

        TimsFrameFlat {
            frame_ids,
            scans,
            tofs,
            retention_times,
            mobilities,
            mzs,
            intensities,
        }
    }
}

#[derive(Clone)]
pub struct TimsSliceVectorized {
    pub frames: Vec<TimsFrameVectorized>,
}