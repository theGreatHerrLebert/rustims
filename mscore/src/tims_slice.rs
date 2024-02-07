use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use itertools::multizip;
use crate::{MsType, TimsSpectrum};

use crate::tims_frame::{ImsFrame, TimsFrame, TimsFrameVectorized, Vectorized, ToResolution};

#[derive(Clone)]
pub struct TimsSlice {
    pub frames: Vec<TimsFrame>,
}

impl TimsSlice {

    /// Create a new TimsSlice from a vector of TimsFrames
    ///
    /// # Arguments
    ///
    /// * `frames` - A vector of TimsFrames
    ///
    /// # Returns
    ///
    /// * `TimsSlice` - A TimsSlice containing the TimsFrames
    ///
    /// # Example
    ///
    /// ```
    /// use mscore::TimsSlice;
    ///
    /// let slice = TimsSlice::new(vec![]);
    /// ```
    pub fn new(frames: Vec<TimsFrame>) -> Self {
        TimsSlice { frames }
    }

    /// Filter the TimsSlice by m/z, scan, and intensity
    ///
    /// # Arguments
    ///
    /// * `mz_min` - The minimum m/z value
    /// * `mz_max` - The maximum m/z value
    /// * `scan_min` - The minimum scan value
    /// * `scan_max` - The maximum scan value
    /// * `intensity_min` - The minimum intensity value
    /// * `intensity_max` - The maximum intensity value
    /// * `num_threads` - The number of threads to use
    ///
    /// # Returns
    ///
    /// * `TimsSlice` - A TimsSlice containing only the TimsFrames that pass the filter
    ///
    /// # Example
    ///
    /// ```
    /// use mscore::TimsSlice;
    ///
    /// let slice = TimsSlice::new(vec![]);
    /// let filtered_slice = slice.filter_ranged(400.0, 2000.0, 0, 1000, 0.0, 100000.0, 0.0, 1.6, 4);
    /// ```
    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, inv_mob_min: f64, inv_mob_max: f64, intensity_min: f64, intensity_max: f64, num_threads: usize) -> TimsSlice {

        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap(); // Set to the desired number of threads

        // Use the thread pool
        let filtered_frames = pool.install(|| {
            let result: Vec<_> =  self.frames.par_iter()
                .map(|f| f.filter_ranged(mz_min, mz_max, scan_min, scan_max, inv_mob_min, inv_mob_max, intensity_min, intensity_max))
                .collect();
            result
        });

        TimsSlice { frames: filtered_frames }
    }

    pub fn filter_ranged_ms_type_specific(&self,
                                          mz_min_ms1: f64,
                                          mz_max_ms1: f64,
                                          scan_min_ms1: i32,
                                          scan_max_ms1: i32,
                                          inv_mob_min_ms1: f64,
                                          inv_mob_max_ms1: f64,
                                          intensity_min_ms1: f64,
                                          intensity_max_ms1: f64,
                                          mz_min_ms2: f64,
                                          mz_max_ms2: f64,
                                          scan_min_ms2: i32,
                                          scan_max_ms2: i32,
                                          inv_mob_min_ms2: f64,
                                          inv_mob_max_ms2: f64,
                                          intensity_min_ms2: f64,
                                          intensity_max_ms2: f64,
                                          num_threads: usize) -> TimsSlice {

        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap(); // Set to the desired number of threads

        // Use the thread pool
        let filtered_frames = pool.install(|| {
            let result: Vec<_> =  self.frames.par_iter()
                .map(|f| match f.ms_type {
                    MsType::Precursor => f.filter_ranged(mz_min_ms1, mz_max_ms1, scan_min_ms1, scan_max_ms1, inv_mob_min_ms1, inv_mob_max_ms1, intensity_min_ms1, intensity_max_ms1),
                    _ => f.filter_ranged(mz_min_ms2, mz_max_ms2, scan_min_ms2, scan_max_ms2, inv_mob_min_ms2, inv_mob_max_ms2, intensity_min_ms2, intensity_max_ms2),
                })
                .collect();
            result
        });

        TimsSlice { frames: filtered_frames }
    }

    /// Get a vector of TimsFrames by MsType
    ///
    /// # Arguments
    ///
    /// * `t` - The MsType to filter by
    ///
    /// # Returns
    ///
    /// * `TimsSlice` - A TimsSlice containing only the TimsFrames of the specified MsType
    pub fn get_slice_by_type(&self, t: MsType) -> TimsSlice {
        let filtered_frames = self.frames.iter()
            .filter(|f| f.ms_type == t)
            .map(|f| f.clone())
            .collect();
        TimsSlice { frames: filtered_frames }
    }

    pub fn to_resolution(&self, resolution: i32, num_threads: usize) -> TimsSlice {

        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap(); // Set to the desired number of threads

        // Use the thread pool
        let result_frames = pool.install(|| {
            let result: Vec<_> =  self.frames.par_iter()
                .map(|f| f.to_resolution(resolution))
                .collect();
            result
        });

        TimsSlice { frames: result_frames }
    }

    pub fn vectorized(&self, resolution: i32, num_threads: usize) -> TimsSliceVectorized {

        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();

        // Use the thread pool
        let result_frames = pool.install(|| {
            let result: Vec<_> =  self.frames.par_iter()
                .map(|f| f.vectorized(resolution))
                .collect();
            result
        });

        TimsSliceVectorized { frames: result_frames }
    }

    pub fn from_flat_slice(frame_ids: Vec<i32>,
                           scans: Vec<i32>,
                           tofs: Vec<i32>,
                           retention_times: Vec<f64>,
                           mobilities: Vec<f64>,
                           mzs: Vec<f64>,
                           intensities: Vec<f64>) -> Self {

        let mut frames = Vec::new();
        let unique_frame_ids: BTreeSet<_> = frame_ids.iter().cloned().collect();

        for frame_id in unique_frame_ids {
            let indices: Vec<usize> = frame_ids.iter().enumerate().filter(|(_, &x)| x == frame_id).map(|(i, _)| i).collect();
            let mut scan = Vec::new();
            let mut tof = Vec::new();
            let mut retention_time = Vec::new();
            let mut mobility = Vec::new();
            let mut mz = Vec::new();
            let mut intensity = Vec::new();

            for index in indices {
                scan.push(scans[index]);
                tof.push(tofs[index]);
                retention_time.push(retention_times[index]);
                mobility.push(mobilities[index]);
                mz.push(mzs[index]);
                intensity.push(intensities[index]);
            }

            let ims_frame = ImsFrame {
                retention_time: retention_time[0],
                mobility,
                mz,
                intensity,
            };

            let tims_frame = TimsFrame {
                frame_id,
                ms_type: MsType::Unknown,
                scan,
                tof,
                ims_frame,
            };

            frames.push(tims_frame);
        }

        TimsSlice { frames }
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

    pub fn to_windows(&self, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64, num_threads: usize) -> Vec<TimsSpectrum> {
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

    pub fn to_dense_windows(&self, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64, resolution: i32, num_threads: usize) -> Vec<(Vec<f64>, Vec<i32>, Vec<i32>, usize, usize)> {
        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();

        let result = pool.install(|| {
            let t = self.frames.par_iter().map(|f| f.to_dense_windows(window_length, overlapping, min_peaks, min_intensity, resolution)).collect::<Vec<_>>();
            t
        });

        result
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

impl TimsSliceVectorized {
    pub fn flatten(&self) -> TimsSliceVectorizedFlat {
        let mut frame_ids = Vec::new();
        let mut scans = Vec::new();
        let mut tofs = Vec::new();
        let mut retention_times = Vec::new();
        let mut mobilities = Vec::new();
        let mut indices = Vec::new();
        let mut intensities = Vec::new();

        for frame in &self.frames {
            let length = frame.ims_frame.indices.len();
            frame_ids.extend(vec![frame.frame_id; length].into_iter());
            scans.extend(frame.scan.clone());
            tofs.extend(frame.tof.clone());
            retention_times.extend(vec![frame.ims_frame.retention_time; length].into_iter());
            mobilities.extend(&frame.ims_frame.mobility);
            indices.extend(&frame.ims_frame.indices);
            intensities.extend(&frame.ims_frame.values);
        }

        TimsSliceVectorizedFlat {
            frame_ids,
            scans,
            tofs,
            retention_times,
            mobilities,
            indices,
            intensities,
        }
    }
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

#[derive(Clone, Debug)]
pub struct TimsSliceFlat {
    pub frame_ids: Vec<i32>,
    pub scans: Vec<i32>,
    pub tofs: Vec<i32>,
    pub retention_times: Vec<f64>,
    pub mobilities: Vec<f64>,
    pub mzs: Vec<f64>,
    pub intensities: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct TimsSliceVectorizedFlat {
    pub frame_ids: Vec<i32>,
    pub scans: Vec<i32>,
    pub tofs: Vec<i32>,
    pub retention_times: Vec<f64>,
    pub mobilities: Vec<f64>,
    pub indices: Vec<i32>,
    pub intensities: Vec<f64>,
}