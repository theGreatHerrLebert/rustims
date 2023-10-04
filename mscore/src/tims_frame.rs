use std::fmt;
use std::collections::BTreeMap;
use std::fmt::{Formatter};
use itertools;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::mz_spectrum::{MsType, MzSpectrum, IndexedMzSpectrum, TimsSpectrum};

#[derive(Clone)]
pub struct TimsSlice {
    pub frames: Vec<TimsFrame>,
}

impl TimsSlice {
    
    pub fn new(frames: Vec<TimsFrame>) -> Self {
        TimsSlice { frames }
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, intensity_min: f64) -> TimsSlice {

        let result: Vec<TimsFrame> = self.frames.par_iter()
            .map(|f| f.filter_ranged(mz_min, mz_max, scan_min, scan_max, intensity_min))
            .collect();

        TimsSlice { frames: result }
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

#[derive(Clone)]
pub struct ImsFrame {
    pub retention_time: f64,
    pub mobility: Vec<f64>,
    pub mz: Vec<f64>,
    pub intensity: Vec<f64>,
}

impl ImsFrame {
    /// Creates a new `ImsFrame` instance.
    ///
    /// # Arguments
    ///
    /// * `retention_time` - The retention time in seconds.
    /// * `mobility` - A vector of inverse ion mobilities.
    /// * `mz` - A vector of m/z values.
    /// * `intensity` - A vector of intensity values.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::ImsFrame;
    ///
    /// let frame = ImsFrame::new(100.0, vec![0.1, 0.2], vec![100.5, 200.5], vec![50.0, 60.0]);
    /// ```
    pub fn new(retention_time: f64, mobility: Vec<f64>, mz: Vec<f64>, intensity: Vec<f64>) -> Self {
        ImsFrame { retention_time, mobility, mz, intensity }
    }
}

impl fmt::Display for ImsFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "ImsFrame(rt: {}, data points: {})", self.retention_time, self.mobility.len())
    }
}

#[derive(Clone)]
pub struct ImsFrameVectorized {
    pub retention_time: f64,
    pub mobility: Vec<f64>,
    pub indices: Vec<i32>,
    pub values: Vec<f64>,
    pub resolution: i32,
}

#[derive(Clone)]
pub struct TimsFrame {
    pub frame_id: i32,
    pub ms_type: MsType,
    pub scan: Vec<i32>,
    pub tof: Vec<i32>,
    pub ims_frame: ImsFrame,
}

impl TimsFrame {
    /// Creates a new `TimsFrame` instance.
    ///
    /// # Arguments
    ///
    /// * `frame_id` - index of frame in TDF raw file.
    /// * `ms_type` - The type of frame.
    /// * `retention_time` - The retention time in seconds.
    /// * `scan` - A vector of scan IDs.
    /// * `mobility` - A vector of inverse ion mobilities.
    /// * `tof` - A vector of time-of-flight values.
    /// * `mz` - A vector of m/z values.
    /// * `intensity` - A vector of intensity values.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::MsType;
    /// use mscore::{TimsFrame, ImsFrame};
    ///
    /// let frame = TimsFrame::new(1, MsType::Precursor, 100.0, vec![1, 2], vec![0.1, 0.2], vec![1000, 2000], vec![100.5, 200.5], vec![50.0, 60.0]);
    /// ```
    pub fn new(frame_id: i32, ms_type: MsType, retention_time: f64, scan: Vec<i32>, mobility: Vec<f64>, tof: Vec<i32>, mz: Vec<f64>, intensity: Vec<f64>) -> Self {
        TimsFrame { frame_id, ms_type, scan, tof, ims_frame: ImsFrame { retention_time, mobility, mz, intensity } }
    }

    ///
    /// Convert a given TimsFrame to an ImsFrame.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::{TimsSpectrum, TimsFrame, MsType};
    ///
    /// let frame = TimsFrame::new(1, MsType::Precursor, 100.0, vec![1, 2], vec![0.1, 0.2], vec![1000, 2000], vec![100.5, 200.5], vec![50.0, 60.0]);
    /// let ims_spectrum = frame.get_ims_frame();
    /// ```
    pub fn get_ims_frame(&self) -> ImsFrame { self.ims_frame.clone() }

    ///
    /// Convert a given TimsFrame to a vector of TimsSpectrum.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::{TimsSpectrum, TimsFrame, MsType};
    ///
    /// let frame = TimsFrame::new(1, MsType::Precursor, 100.0, vec![1, 2], vec![0.1, 0.2], vec![1000, 2000], vec![100.5, 200.5], vec![50.0, 60.0]);
    /// let tims_spectra = frame.to_tims_spectra();
    /// ```
    pub fn to_tims_spectra(&self) -> Vec<TimsSpectrum> {
        // use a sorted map where scan is used as key
        let mut spectra = BTreeMap::<i32, (f64, Vec<i32>, Vec<f64>, Vec<f64>)>::new();

        // all indices and the intensity values are sorted by scan and stored in the map as a tuple (mobility, tof, mz, intensity)
        for (scan, mobility, tof, mz, intensity) in itertools::multizip((
            &self.scan,
            &self.ims_frame.mobility,
            &self.tof,
            &self.ims_frame.mz,
            &self.ims_frame.intensity)) {
            let entry = spectra.entry(*scan).or_insert_with(|| (*mobility, Vec::new(), Vec::new(), Vec::new()));
            entry.1.push(*tof);
            entry.2.push(*mz);
            entry.3.push(*intensity);
        }

        // convert the map to a vector of TimsSpectrum
        let mut tims_spectra: Vec<TimsSpectrum> = Vec::new();

        for (scan, (mobility, tof, mz, intensity)) in spectra {
            let spectrum = IndexedMzSpectrum::new(tof, mz, intensity);
            tims_spectra.push(TimsSpectrum::new(self.frame_id, scan, self.ims_frame.retention_time, mobility, self.ms_type.clone(), spectrum));
        }

        tims_spectra
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, intensity_min: f64) -> TimsFrame {

        let mut scan_vec = Vec::new();
        let mut mobility_vec = Vec::new();
        let mut tof_vec = Vec::new();
        let mut mz_vec = Vec::new();
        let mut intensity_vec = Vec::new();

        for (mz, intensity, scan, mobility, tof) in itertools::multizip((&self.ims_frame.mz, &self.ims_frame.intensity, &self.scan, &self.ims_frame.mobility, &self.tof)) {
            if mz >= &mz_min && mz <= &mz_max && scan >= &scan_min && scan <= &scan_max && intensity >= &intensity_min {
                scan_vec.push(*scan);
                mobility_vec.push(*mobility);
                tof_vec.push(*tof);
                mz_vec.push(*mz);
                intensity_vec.push(*intensity);
            }
        }

        TimsFrame::new(self.frame_id, self.ms_type.clone(), self.ims_frame.retention_time, scan_vec, mobility_vec, tof_vec, mz_vec, intensity_vec)
    }

    pub fn to_resolution(&self, resolution: i32) -> TimsFrame {

        let factor = (10.0f64).powi(resolution);

        // Using a tuple of (scan, mz_bin) as a key
        // Value will store sum of intensities, sum of tofs, sum of mobilities and their count for averaging
        let mut bin_map: BTreeMap<(i32, i32), (f64, f64, f64, i32)> = BTreeMap::new();

        for i in 0..self.ims_frame.mz.len() {
            let rounded_mz = (self.ims_frame.mz[i] * factor).round() as i32;
            let scan_val = self.scan[i];
            let intensity_val = self.ims_frame.intensity[i] as f64;
            let tof_val = self.tof[i] as f64;
            let mobility_val = self.ims_frame.mobility[i] as f64;

            let entry = bin_map.entry((scan_val, rounded_mz)).or_insert((0.0, 0.0, 0.0, 0));
            entry.0 += intensity_val;
            entry.1 += tof_val;
            entry.2 += mobility_val;
            entry.3 += 1;
        }

        let mut new_mz = Vec::with_capacity(bin_map.len());
        let mut new_scan = Vec::with_capacity(bin_map.len());
        let mut new_intensity = Vec::with_capacity(bin_map.len());
        let mut new_tof = Vec::with_capacity(bin_map.len());
        let mut new_mobility = Vec::with_capacity(bin_map.len());

        for ((scan, mz_bin), (intensity_sum, tof_sum, mobility_sum, count)) in bin_map {
            new_mz.push(mz_bin as f64 / factor);
            new_scan.push(scan);
            new_intensity.push(intensity_sum);
            new_tof.push((tof_sum / count as f64) as i32);
            new_mobility.push(mobility_sum / count as f64);
        }

        TimsFrame {
            frame_id: self.frame_id,
            ms_type: self.ms_type.clone(),
            scan: new_scan,
            tof: new_tof,
            ims_frame: ImsFrame {
                retention_time: self.ims_frame.retention_time,
                mobility: new_mobility,
                mz: new_mz,
                intensity: new_intensity,
            },
        }
    }

    pub fn to_windows(&self, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64) -> Vec<MzSpectrum> {

        let spectra = self.to_tims_spectra();

        let windows: Vec<_> = spectra.iter().map(|spectrum|
            spectrum.spectrum.mz_spectrum.to_windows(window_length, overlapping, min_peaks, min_intensity))
            .collect();

        let mut result: Vec<MzSpectrum> = Vec::new();

        for window in windows {
            for (_, spectrum) in window {
                result.push(spectrum);
            }
        }

        result
    }
}

impl fmt::Display for TimsFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {

        let (mz, i) = self.ims_frame.mz.iter()
            .zip(&self.ims_frame.intensity)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        write!(f, "TimsFrame(id: {}, type: {}, rt: {}, data points: {}, max by intensity: (mz: {}, intensity: {}))",
               self.frame_id, self.ms_type, self.ims_frame.retention_time, self.scan.len(), format!("{:.3}", mz), i)
    }
}

#[derive(Clone)]
pub struct TimsFrameVectorized {
    pub frame_id: i32,
    pub ms_type: MsType,
    pub scan: Vec<i32>,
    pub tof: Vec<i32>,
    pub ims_frame: ImsFrameVectorized,
}
