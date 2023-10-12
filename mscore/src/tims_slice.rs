use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use std::collections::BTreeMap;
use itertools::multizip;
use crate::MsType;

use crate::mz_spectrum::{MzSpectrum};
use crate::tims_frame::{TimsFrame, TimsSliceFlat, TimsFrameVectorized};

#[derive(Clone)]
pub struct TimsSlice {
    pub frames: Vec<TimsFrame>,
}

impl TimsSlice {

    pub fn new(frames: Vec<TimsFrame>) -> Self {
        TimsSlice { frames }
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, intensity_min: f64, intensity_max: f64, num_threads: usize) -> TimsSlice {

        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap(); // Set to the desired number of threads

        // Use the thread pool
        let filtered_frames = pool.install(|| {
            let result: Vec<_> =  self.frames.par_iter()
                .map(|f| f.filter_ranged(mz_min, mz_max, scan_min, scan_max, intensity_min, intensity_max))
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

    pub fn get_slice_by_type(&self, t: MsType) -> TimsSlice {
        let filtered_frames = self.frames.iter()
            .filter(|f| f.ms_type == t)
            .map(|f| f.clone())
            .collect();
        TimsSlice { frames: filtered_frames }
    }

    pub fn flatten(&self) -> TimsSliceFlat {
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

        TimsSliceFlat {
            frame_ids,
            scans,
            tofs,
            retention_times,
            mobilities,
            mzs,
            intensities,
        }
    }

    pub fn to_tims_planes(&self, tof_max_value: i32, num_chunks: i32, num_threads: usize) -> Vec<TimsPlane> {

        let flat_slice = self.flatten();

        let chunk_size = (tof_max_value as f64 / num_chunks as f64) as i32;

        // Calculate range_and_width based on num_chunks and chunk_size
        let range_and_width: Vec<(i32, i32)> = (1..=num_chunks)
            .map(|i| (chunk_size * i, i + 2))
            .collect();

        let mut tof_map: BTreeMap<(i32, i32), (Vec<i32>, Vec<f64>, Vec<i32>, Vec<f64>, Vec<i32>, Vec<f64>, Vec<f64>)> = BTreeMap::new();

        // Iterate over the data points using multizip
        for (id, rt, scan, mobility, tof, mz, intensity)

        in multizip((flat_slice.frame_ids, flat_slice.retention_times, flat_slice.scans, flat_slice.mobilities, flat_slice.tofs, flat_slice.mzs, flat_slice.intensities)) {

            for &(switch_point, width) in &range_and_width {
                if tof < switch_point {

                    let key = (width, (tof as f64 / width as f64).floor() as i32);

                    tof_map.entry(key).or_insert_with(|| (vec![], vec![], vec![], vec![], vec![], vec![], vec![])).0.push(id);
                    tof_map.entry(key).or_insert_with(|| (vec![], vec![], vec![], vec![], vec![], vec![], vec![])).1.push(rt);
                    tof_map.entry(key).or_insert_with(|| (vec![], vec![], vec![], vec![], vec![], vec![], vec![])).2.push(scan);
                    tof_map.entry(key).or_insert_with(|| (vec![], vec![], vec![], vec![], vec![], vec![], vec![])).3.push(mobility);
                    tof_map.entry(key).or_insert_with(|| (vec![], vec![], vec![], vec![], vec![], vec![], vec![])).4.push(tof);
                    tof_map.entry(key).or_insert_with(|| (vec![], vec![], vec![], vec![], vec![], vec![], vec![])).5.push(mz);
                    tof_map.entry(key).or_insert_with(|| (vec![], vec![], vec![], vec![], vec![], vec![], vec![])).6.push(intensity);

                    break
                }
            }
        }

        // Create a thread pool with the desired number of threads
        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();

        let tims_planes: Vec<TimsPlane> = pool.install(|| {
            tof_map.par_iter()
                .map(|(key, values)| collapse_entry(key, values))
                .collect()
        });

        tims_planes
    }
}

#[derive(Clone)]
pub struct TimsSliceVectorized {
    pub frames: Vec<TimsFrameVectorized>,
}

#[derive(Clone)]
pub struct TimsPlane {
    pub tof_mean: f64,
    pub tof_std: f64,
    pub mz_mean: f64,
    pub mz_std: f64,

    pub frame_id: Vec<i32>,
    pub retention_time: Vec<f64>,
    pub scan: Vec<i32>,
    pub mobility: Vec<f64>,
    pub intensity: Vec<f64>,
}

fn collapse_entry(_key: &(i32, i32), values: &(Vec<i32>, Vec<f64>, Vec<i32>, Vec<f64>, Vec<i32>, Vec<f64>, Vec<f64>)) -> TimsPlane {

    let (frame_ids, retention_times, scans, mobilities, tofs, mzs, intensities) = values;

    // 1. Calculate mean and std for tof and mz
    let tof_mean: f64 = tofs.iter().map(|&x| x as f64).sum::<f64>() / tofs.len() as f64;
    let tof_std: f64 = (tofs.iter().map(|&x| (x as f64 - tof_mean).powi(2)).sum::<f64>() / tofs.len() as f64).sqrt();
    let mz_mean: f64 = mzs.iter().map(|&x| x as f64).sum::<f64>() / mzs.len() as f64;
    let mz_std: f64 = (mzs.iter().map(|&x| (x as f64 - mz_mean).powi(2)).sum::<f64>() / mzs.len() as f64).sqrt();

    // 2. Aggregate data by frame_id and scan using a BTreeMap for sorted order
    let mut grouped_data: BTreeMap<(i32, i32), (f64, f64, f64)> = BTreeMap::new();

    for (f, r, s, m, i) in multizip((frame_ids, retention_times, scans, mobilities, intensities)) {
        let key = (*f, *s);
        let entry = grouped_data.entry(key).or_insert((0.0, 0.0, 0.0));  // (intensity_sum, mobility, retention_time)
        entry.0 += *i;
        entry.1 = *m;
        entry.2 = *r;
    }

    // Extract data from the grouped_data
    let mut frame_id = vec![];
    let mut retention_time = vec![];
    let mut scan = vec![];
    let mut mobility = vec![];
    let mut intensity = vec![];

    for ((f, s), (i, m, r)) in grouped_data {
            frame_id.push(f);
            retention_time.push(r);
            scan.push(s);
            mobility.push(m);
            intensity.push(i);
        }

    TimsPlane {
        tof_mean,
        tof_std,
        mz_mean,
        mz_std,
        frame_id,
        retention_time,
        scan,
        mobility,
        intensity,
    }
}
