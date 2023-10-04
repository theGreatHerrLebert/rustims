use std::fmt;
use std::collections::BTreeMap;
use std::fmt::{Formatter};
use itertools;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::mz_spectrum::{MsType, MzSpectrum, IndexedMzSpectrum, ImsSpectrum, TimsSpectrum};

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
}

#[derive(Clone)]
pub struct ImsFrame {
    pub retention_time: f64,
    pub inv_mobility: Vec<f64>,
    pub mz: Vec<f64>,
    pub intensity: Vec<f64>,
}

impl ImsFrame {
    /// Creates a new `ImsFrame` instance.
    ///
    /// # Arguments
    ///
    /// * `retention_time` - The retention time in seconds.
    /// * `inv_mobility` - A vector of inverse ion mobilities.
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
    pub fn new(retention_time: f64, inv_mobility: Vec<f64>, mz: Vec<f64>, intensity: Vec<f64>) -> Self {
        ImsFrame { retention_time, inv_mobility, mz, intensity }
    }
}

impl fmt::Display for ImsFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "ImsFrame(rt: {}, data points: {})", self.retention_time, self.inv_mobility.len())
    }
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
    /// * `inv_mobility` - A vector of inverse ion mobilities.
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
    pub fn new(frame_id: i32, ms_type: MsType, retention_time: f64, scan: Vec<i32>, inv_mobility: Vec<f64>, tof: Vec<i32>, mz: Vec<f64>, intensity: Vec<f64>) -> Self {
        TimsFrame { frame_id, ms_type, scan, tof, ims_frame: ImsFrame { retention_time, inv_mobility, mz, intensity } }
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

        // all indices and the intensity values are sorted by scan and stored in the map as a tuple (inv_mobility, tof, mz, intensity)
        for (scan, inv_mobility, tof, mz, intensity) in itertools::multizip((
            &self.scan,
            &self.ims_frame.inv_mobility,
            &self.tof,
            &self.ims_frame.mz,
            &self.ims_frame.intensity,
        )) {
            let entry = spectra.entry(*scan).or_insert_with(|| (*inv_mobility, Vec::new(), Vec::new(), Vec::new()));
            entry.1.push(*tof);
            entry.2.push(*mz);
            entry.3.push(*intensity);
        }

        // convert the map to a vector of TimsSpectrum
        let mut tims_spectra: Vec<TimsSpectrum> = Vec::new();

        for (scan, (inv_mobility, tof, mz, intensity)) in spectra {
            let spectrum = IndexedMzSpectrum::new(tof, mz, intensity);
            tims_spectra.push(TimsSpectrum::new(self.frame_id, scan, self.ims_frame.retention_time, inv_mobility, self.ms_type.clone(), spectrum));
        }

        tims_spectra
    }

    ///
    /// Convert a given TimsFrame to a vector of ImsSpectrum.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::{TimsSpectrum, TimsFrame, MsType};
    ///
    /// let frame = TimsFrame::new(1, MsType::Precursor, 100.0, vec![1, 2], vec![0.1, 0.2], vec![1000, 2000], vec![100.5, 200.5], vec![50.0, 60.0]);
    /// let ims_spectra = frame.to_ims_spectra();
    /// ```
    pub fn to_ims_spectra(&self) -> Vec<ImsSpectrum> {
        let tims_spectra = self.to_tims_spectra();
        let mut ims_spectra: Vec<ImsSpectrum> = Vec::new();

        for spec in tims_spectra {
            let ims_spec = ImsSpectrum::new(spec.retention_time, spec.inv_mobility, MzSpectrum::new(spec.spectrum.mz_spectrum.mz, spec.spectrum.mz_spectrum.intensity));
            ims_spectra.push(ims_spec);
        }

        ims_spectra
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, intensity_min: f64) -> TimsFrame {

        let mut scan_vec = Vec::new();
        let mut inv_mobility_vec = Vec::new();
        let mut tof_vec = Vec::new();
        let mut mz_vec = Vec::new();
        let mut intensity_vec = Vec::new();

        for (mz, intensity, scan, inv_mobility, tof) in itertools::multizip((&self.ims_frame.mz, &self.ims_frame.intensity, &self.scan, &self.ims_frame.inv_mobility, &self.tof)) {
            if mz >= &mz_min && mz <= &mz_max && scan >= &scan_min && scan <= &scan_max && intensity >= &intensity_min {
                scan_vec.push(*scan);
                inv_mobility_vec.push(*inv_mobility);
                tof_vec.push(*tof);
                mz_vec.push(*mz);
                intensity_vec.push(*intensity);
            }
        }

        TimsFrame::new(self.frame_id, self.ms_type.clone(), self.ims_frame.retention_time, scan_vec, inv_mobility_vec, tof_vec, mz_vec, intensity_vec)
    }

    pub fn to_windows(&self, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64, num_threads: usize) -> Vec<MzSpectrum> {

        let spectra = self.to_tims_spectra();

        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap(); // Set to the desired number of threads

        // Use the thread pool
        let windows = pool.install(|| {

            let windows: Vec<_> = spectra.par_iter()
                .map(|spectrum| spectrum.spectrum.mz_spectrum.to_windows(window_length, overlapping, min_peaks, min_intensity))
                .collect();

            windows

        });

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
