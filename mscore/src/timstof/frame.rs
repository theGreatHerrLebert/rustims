use std::fmt;
use std::collections::BTreeMap;
use std::fmt::{Formatter};
use itertools;
use itertools::izip;
use rand::Rng;

use crate::timstof::spectrum::TimsSpectrum;
use crate::data::spectrum::{MsType, MzSpectrum, IndexedMzSpectrum, Vectorized, ToResolution};

#[derive(Clone)]
pub struct RawTimsFrame {
    pub frame_id: i32,
    pub retention_time: f64,
    pub ms_type: MsType,
    pub scan: Vec<i32>,
    pub tof: Vec<i32>,
    pub intensity: Vec<f64>,
}

#[derive(Clone, Debug)]
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
    /// use mscore::timstof::frame::ImsFrame;
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

#[derive(Clone, Debug)]
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
    /// use mscore::data::spectrum::MsType;
    /// use mscore::timstof::frame::TimsFrame;
    /// use mscore::timstof::frame::ImsFrame;
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
    /// use mscore::data::spectrum::MsType;
    /// use mscore::timstof::frame::TimsFrame;
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
    /// use mscore::data::spectrum::MsType;
    /// use mscore::timstof::frame::TimsFrame;
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

    ///
    /// Filter a given TimsFrame by m/z, scan, and intensity.
    ///
    /// # Arguments
    ///
    /// * `mz_min` - The minimum m/z value.
    /// * `mz_max` - The maximum m/z value.
    /// * `scan_min` - The minimum scan value.
    /// * `scan_max` - The maximum scan value.
    /// *
    /// * `intensity_min` - The minimum intensity value.
    /// * `intensity_max` - The maximum intensity value.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::data::spectrum::MsType;
    /// use mscore::timstof::frame::TimsFrame;
    ///
    /// let frame = TimsFrame::new(1, MsType::Precursor, 100.0, vec![1, 2], vec![0.1, 0.2], vec![1000, 2000], vec![100.5, 200.5], vec![50.0, 60.0]);
    /// let filtered_frame = frame.filter_ranged(100.0, 200.0, 1, 2, 0.0, 1.6, 50.0, 60.0);
    /// ```
    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, inv_mob_min: f64, inv_mob_max: f64, intensity_min: f64, intensity_max: f64) -> TimsFrame {

        let mut scan_vec = Vec::new();
        let mut mobility_vec = Vec::new();
        let mut tof_vec = Vec::new();
        let mut mz_vec = Vec::new();
        let mut intensity_vec = Vec::new();

        for (mz, intensity, scan, mobility, tof) in itertools::multizip((&self.ims_frame.mz, &self.ims_frame.intensity, &self.scan, &self.ims_frame.mobility, &self.tof)) {
            if mz >= &mz_min && mz <= &mz_max && scan >= &scan_min && scan <= &scan_max && mobility >= &inv_mob_min && mobility <= &inv_mob_max && intensity >= &intensity_min && intensity <= &intensity_max {
                scan_vec.push(*scan);
                mobility_vec.push(*mobility);
                tof_vec.push(*tof);
                mz_vec.push(*mz);
                intensity_vec.push(*intensity);
            }
        }

        TimsFrame::new(self.frame_id, self.ms_type.clone(), self.ims_frame.retention_time, scan_vec, mobility_vec, tof_vec, mz_vec, intensity_vec)
    }

    pub fn to_windows_indexed(&self, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64) -> (Vec<i32>, Vec<i32>, Vec<TimsSpectrum>) {
        // split by scan (ion mobility)
        let spectra = self.to_tims_spectra();

        let windows: Vec<_> = spectra.iter().map(|spectrum|
            spectrum.to_windows(window_length, overlapping, min_peaks, min_intensity))
            .collect();

        let mut scan_indices = Vec::new();

        for tree in windows.iter() {
            for (_, window) in tree {
                scan_indices.push(window.scan)
            }
        }

        let mut spectra = Vec::new();
        let mut window_indices = Vec::new();

        for window in windows {
            for (i, spectrum) in window {
                spectra.push(spectrum);
                window_indices.push(i);
            }
        }

        (scan_indices, window_indices, spectra)
    }

    pub fn to_windows(&self, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64) -> Vec<TimsSpectrum> {
        let (_, _, widows) = self.to_windows_indexed(window_length, overlapping, min_peaks, min_intensity);
        widows
    }

    pub fn from_windows(windows: Vec<TimsSpectrum>) -> TimsFrame {

        let first_window = windows.first().unwrap();

        let mut scan = Vec::new();
        let mut tof = Vec::new();
        let mut mzs = Vec::new();
        let mut intensity = Vec::new();
        let mut mobility = Vec::new();

        for window in windows.iter() {
            for (i, mz) in window.spectrum.mz_spectrum.mz.iter().enumerate() {
                scan.push(window.scan);
                mobility.push(window.mobility);
                tof.push(window.spectrum.index[i]);
                mzs.push(*mz);
                intensity.push(window.spectrum.mz_spectrum.intensity[i]);
            }
        }

        TimsFrame::new(first_window.frame_id, first_window.ms_type.clone(), first_window.retention_time, scan, mobility, tof, mzs, intensity)
    }

    pub fn from_tims_spectra(spectra: Vec<TimsSpectrum>) -> TimsFrame {

        // Helper to quantize mz to an integer key
        let quantize = |mz: f64| -> i64 {
            (mz * 1_000_000.0).round() as i64
        };

        // Step 1: Get frame coordinates
        let frame_id = spectra.first().unwrap().frame_id;
        let ms_type = spectra.first().unwrap().ms_type.clone();
        let retention_time = spectra.first().unwrap().retention_time;

        let mut frame_map: BTreeMap<i32, (f64, BTreeMap<i64, (i32, f64)>)> = BTreeMap::new();
        let mut capacity_count = 0;

        // Step 2: Group by scan and unroll all spectra to a single vector per scan
        for spectrum in &spectra {
            let inv_mobility = spectrum.mobility;
            let entry = frame_map.entry(spectrum.scan).or_insert_with(|| (inv_mobility, BTreeMap::new()));
            for (i, mz) in spectrum.spectrum.mz_spectrum.mz.iter().enumerate() {
                let tof = spectrum.spectrum.index[i];
                let intensity = spectrum.spectrum.mz_spectrum.intensity[i];
                entry.1.entry(quantize(*mz)).and_modify(|e| *e = (tof, e.1 + intensity)).or_insert((tof, intensity));
                capacity_count += 1;
            }
        }

        // Step 3: Unroll the map to vectors
        let mut scan = Vec::with_capacity(capacity_count);
        let mut mobility = Vec::with_capacity(capacity_count);
        let mut tof = Vec::with_capacity(capacity_count);
        let mut mzs = Vec::with_capacity(capacity_count);
        let mut intensity = Vec::with_capacity(capacity_count);

        for (scan_val, (mobility_val, mz_map)) in frame_map {
            for (mz_val, (tof_val, intensity_val)) in mz_map {
                scan.push(scan_val);
                mobility.push(mobility_val);
                tof.push(tof_val);
                mzs.push(mz_val as f64 / 1_000_000.0);
                intensity.push(intensity_val);
            }
        }

        TimsFrame::new(frame_id, ms_type, retention_time, scan, mobility, tof, mzs, intensity)
    }

    pub fn to_dense_windows(&self, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64, resolution: i32) -> (Vec<f64>, Vec<i32>, Vec<i32>, usize, usize) {
        let factor = (10.0f64).powi(resolution);
        let num_colums = ((window_length * factor).round() + 1.0) as usize;

        let (scans, window_indices, spectra) = self.to_windows_indexed(window_length, overlapping, min_peaks, min_intensity);
        let vectorized_spectra = spectra.iter().map(|spectrum| spectrum.vectorized(resolution)).collect::<Vec<_>>();

        let mut flat_matrix: Vec<f64> = vec![0.0; spectra.len() * num_colums];

        for (row_index, (window_index, spectrum)) in itertools::multizip((&window_indices, vectorized_spectra)).enumerate() {

            let vectorized_window_index = match *window_index >= 0 {
                true => (*window_index as f64 * window_length * factor).round() as i32,
                false => (((-1.0 * (*window_index as f64)) * window_length - (0.5 * window_length)) * factor).round() as i32,
            };

            for (i, index) in spectrum.vector.mz_vector.indices.iter().enumerate() {
                let zero_based_index = (index - vectorized_window_index) as usize;
                let current_index = row_index * num_colums + zero_based_index;
                flat_matrix[current_index] = spectrum.vector.mz_vector.values[i];
            }

        }
        (flat_matrix, scans, window_indices, spectra.len(), num_colums)
    }


    pub fn to_indexed_mz_spectrum(&self) -> IndexedMzSpectrum {
        let mut grouped_data: BTreeMap<i32, Vec<(f64, f64)>> = BTreeMap::new();

        // Group by 'tof' with 'mz' and 'intensity'
        for (&tof, (&mz, &intensity)) in self.tof.iter().zip(self.ims_frame.mz.iter().zip(self.ims_frame.intensity.iter())) {
            grouped_data.entry(tof).or_insert_with(Vec::new).push((mz, intensity));
        }

        let mut index = Vec::new();
        let mut mz = Vec::new();
        let mut intensity = Vec::new();

        for (&tof_val, values) in &grouped_data {
            let sum_intensity: f64 = values.iter().map(|&(_, i)| i).sum();
            let avg_mz: f64 = values.iter().map(|&(m, _)| m).sum::<f64>() / values.len() as f64;

            index.push(tof_val);
            mz.push(avg_mz);
            intensity.push(sum_intensity);
        }

        IndexedMzSpectrum {
            index,
            mz_spectrum: MzSpectrum { mz, intensity },
        }
    }

    pub fn generate_random_sample(&self, take_probability: f64) -> TimsFrame {
        assert!(take_probability >= 0.0 && take_probability <= 1.0);

        let mut rng = rand::thread_rng();
        let mut scan = Vec::new();
        let mut mobility = Vec::new();
        let mut tof = Vec::new();
        let mut mz = Vec::new();
        let mut intensity = Vec::new();

        for (s, m, t, mz_val, i) in itertools::multizip((&self.scan, &self.ims_frame.mobility, &self.tof, &self.ims_frame.mz, &self.ims_frame.intensity)) {
            if rng.gen::<f64>() <= take_probability {
                scan.push(*s);
                mobility.push(*m);
                tof.push(*t);
                mz.push(*mz_val);
                intensity.push(*i);
            }
        }

        TimsFrame::new(self.frame_id, self.ms_type.clone(), self.ims_frame.retention_time, scan, mobility, tof, mz, intensity)
    }
}

struct AggregateData {
    intensity_sum: f64,
    ion_mobility_sum: f64,
    tof_sum: i64,
    count: i32,
}

impl std::ops::Add for TimsFrame {
    type Output = Self;
    fn add(self, other: Self) -> TimsFrame {
        let mut combined_map: BTreeMap<(i64, i32), AggregateData> = BTreeMap::new();

        let quantize = |mz: f64| -> i64 {
            (mz * 1_000_000.0).round() as i64
        };

        let add_to_map = |map: &mut BTreeMap<(i64, i32), AggregateData>, mz, ion_mobility, tof, scan, intensity| {
            let key = (quantize(mz), scan);
            let entry = map.entry(key).or_insert(AggregateData { intensity_sum: 0.0, ion_mobility_sum: 0.0, tof_sum: 0, count: 0 });
            entry.intensity_sum += intensity;
            entry.ion_mobility_sum += ion_mobility;
            entry.tof_sum += tof as i64;
            entry.count += 1;
        };

        for (mz, tof, ion_mobility, scan, intensity) in izip!(&self.ims_frame.mz, &self.tof, &self.ims_frame.mobility, &self.scan, &self.ims_frame.intensity) {
            add_to_map(&mut combined_map, *mz, *ion_mobility, *tof, *scan, *intensity);
        }

        for (mz, tof, ion_mobility, scan, intensity) in izip!(&other.ims_frame.mz, &other.tof, &other.ims_frame.mobility, &other.scan, &other.ims_frame.intensity) {
            add_to_map(&mut combined_map, *mz, *ion_mobility, *tof, *scan, *intensity);
        }

        let mut mz_combined = Vec::new();
        let mut tof_combined = Vec::new();
        let mut ion_mobility_combined = Vec::new();
        let mut scan_combined = Vec::new();
        let mut intensity_combined = Vec::new();

        for ((quantized_mz, scan), data) in combined_map {
            mz_combined.push(quantized_mz as f64 / 1_000_000.0);
            tof_combined.push(data.tof_sum / data.count as i64);
            ion_mobility_combined.push(data.ion_mobility_sum / data.count as f64);
            scan_combined.push(scan);
            intensity_combined.push(data.intensity_sum);
        }

        TimsFrame {
            frame_id: self.frame_id,
            ms_type: if self.ms_type == other.ms_type { self.ms_type.clone() } else { MsType::Unknown },
            scan: scan_combined,
            tof: tof_combined.iter().map(|&x| x as i32).collect(),
            ims_frame: ImsFrame {
                retention_time: self.ims_frame.retention_time,
                mobility: ion_mobility_combined,
                mz: mz_combined,
                intensity: intensity_combined,
            },
        }
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

impl Vectorized<TimsFrameVectorized> for TimsFrame {
    fn vectorized(&self, resolution: i32) -> TimsFrameVectorized {
        let binned_frame = self.to_resolution(resolution);
        // Translate the m/z values into integer indices
        let indices: Vec<i32> = binned_frame.ims_frame.mz.iter().map(|&mz| (mz * 10f64.powi(resolution)).round() as i32).collect();
        // Create a vector of values
        return TimsFrameVectorized {
            frame_id: self.frame_id,
            ms_type: self.ms_type.clone(),
            scan: binned_frame.scan,
            tof: binned_frame.tof,
            ims_frame: ImsFrameVectorized {
                retention_time: binned_frame.ims_frame.retention_time,
                mobility: binned_frame.ims_frame.mobility,
                indices,
                values: binned_frame.ims_frame.intensity,
                resolution,
            },
        };
    }
}

///
/// Convert a given TimsFrame to a vector of TimsSpectrum.
///
/// # Arguments
///
/// * `resolution` - The resolution to which the m/z values should be rounded.
///
/// # Examples
///
/// ```
/// use mscore::data::spectrum::MsType;
/// use mscore::timstof::frame::TimsFrame;
/// use mscore::timstof::spectrum::TimsSpectrum;
/// use mscore::data::spectrum::IndexedMzSpectrum;
/// use mscore::data::spectrum::ToResolution;
///
/// let frame = TimsFrame::new(1, MsType::Precursor, 100.0, vec![1, 2], vec![0.1, 0.2], vec![1000, 2000], vec![100.5, 200.5], vec![50.0, 60.0]);
/// let low_res_frame = frame.to_resolution(1);
/// ```
impl ToResolution for TimsFrame {
    fn to_resolution(&self, resolution: i32) -> TimsFrame {
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
}

#[derive(Clone)]
pub struct TimsFrameVectorized {
    pub frame_id: i32,
    pub ms_type: MsType,
    pub scan: Vec<i32>,
    pub tof: Vec<i32>,
    pub ims_frame: ImsFrameVectorized,
}

impl TimsFrameVectorized {
    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, inv_mob_min: f64, inv_mob_max: f64, intensity_min: f64, intensity_max: f64) -> TimsFrameVectorized {
        let mut scan_vec = Vec::new();
        let mut mobility_vec = Vec::new();
        let mut tof_vec = Vec::new();
        let mut mz_vec = Vec::new();
        let mut intensity_vec = Vec::new();
        let mut indices_vec = Vec::new();

        for (mz, intensity, scan, mobility, tof, index) in itertools::multizip((&self.ims_frame.values, &self.ims_frame.values, &self.scan, &self.ims_frame.mobility, &self.tof, &self.ims_frame.indices)) {
            if mz >= &mz_min && mz <= &mz_max && scan >= &scan_min && scan <= &scan_max && mobility >= &inv_mob_min && mobility <= &inv_mob_max && intensity >= &intensity_min && intensity <= &intensity_max {
                scan_vec.push(*scan);
                mobility_vec.push(*mobility);
                tof_vec.push(*tof);
                mz_vec.push(*mz);
                intensity_vec.push(*intensity);
                indices_vec.push(*index);
            }
        }

        TimsFrameVectorized {
            frame_id: self.frame_id,
            ms_type: self.ms_type.clone(),
            scan: scan_vec,
            tof: tof_vec,
            ims_frame: ImsFrameVectorized {
                retention_time: self.ims_frame.retention_time,
                mobility: mobility_vec,
                indices: indices_vec,
                values: mz_vec,
                resolution: self.ims_frame.resolution,
            },
        }
    }
}

impl fmt::Display for TimsFrameVectorized {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {

        let (mz, i) = self.ims_frame.values.iter()
            .zip(&self.ims_frame.values)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        write!(f, "TimsFrame(id: {}, type: {}, rt: {}, data points: {}, max by intensity: (mz: {}, intensity: {}))",
               self.frame_id, self.ms_type, self.ims_frame.retention_time, self.scan.len(), format!("{:.3}", mz), i)
    }
}
