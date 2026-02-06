use std::collections::BTreeMap;
use std::fmt;
use std::fmt::{Display, Formatter};
use crate::data::spectrum::{IndexedMzSpectrum, IndexedMzSpectrumVectorized, MsType, MzSpectrum};

#[derive(Clone)]
pub struct TimsSpectrumVectorized {
    pub frame_id: i32,
    pub scan: i32,
    pub retention_time: f64,
    pub mobility: f64,
    pub ms_type: MsType,
    pub vector: IndexedMzSpectrumVectorized,
}

#[derive(Clone, Debug)]
pub struct TimsSpectrum {
    pub frame_id: i32,
    pub scan: i32,
    pub retention_time: f64,
    pub mobility: f64,
    pub ms_type: MsType,
    pub spectrum: IndexedMzSpectrum,
}

impl TimsSpectrum {
    /// Creates a new `TimsSpectrum` instance.
    ///
    /// # Arguments
    ///
    /// * `frame_id` - index of frame in TDF raw file.
    /// * `scan_id` - index of scan in TDF raw file.
    /// * `retention_time` - The retention time in seconds.
    /// * `mobility` - The inverse ion mobility.
    /// * `spectrum` - A `TOFMzSpectrum` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::data::spectrum::{IndexedMzSpectrum, MsType};
    /// use mscore::timstof::spectrum::{TimsSpectrum};
    ///
    /// let spectrum = TimsSpectrum::new(1, 1, 100.0, 0.1, MsType::FragmentDda, IndexedMzSpectrum::new(vec![1000, 2000], vec![100.5, 200.5], vec![50.0, 60.0]));
    /// ```
    pub fn new(frame_id: i32, scan_id: i32, retention_time: f64, mobility: f64, ms_type: MsType, spectrum: IndexedMzSpectrum) -> Self {
        TimsSpectrum { frame_id, scan: scan_id, retention_time, mobility: mobility, ms_type, spectrum }
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min: f64, intensity_max: f64) -> Self {
        let filtered = self.spectrum.filter_ranged(mz_min, mz_max, intensity_min, intensity_max);
        TimsSpectrum { frame_id: self.frame_id, scan: self.scan, retention_time: self.retention_time, mobility: self.mobility, ms_type: self.ms_type.clone(), spectrum: filtered }
    }

    pub fn to_resolution(&self, resolution: i32) -> TimsSpectrum {
        let spectrum = self.spectrum.to_resolution(resolution);
        TimsSpectrum { frame_id: self.frame_id, scan: self.scan, retention_time: self.retention_time, mobility: self.mobility, ms_type: self.ms_type.clone(), spectrum }
    }

    pub fn vectorized(&self, resolution: i32) -> TimsSpectrumVectorized {
        let vector = self.spectrum.vectorized(resolution);
        TimsSpectrumVectorized { frame_id: self.frame_id, scan: self.scan, retention_time: self.retention_time, mobility: self.mobility, ms_type: self.ms_type.clone(), vector }
    }

    pub fn to_windows(&self, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64) -> BTreeMap<i32, TimsSpectrum> {
        // Build intermediate structures with raw Vecs first
        let mut splits_raw: BTreeMap<i32, (Vec<f64>, Vec<f64>, Vec<i32>)> = BTreeMap::new();

        for (i, &mz) in self.spectrum.mz_spectrum.mz.iter().enumerate() {
            let intensity = self.spectrum.mz_spectrum.intensity[i];
            let tof = self.spectrum.index[i];

            let tmp_key = (mz / window_length).floor() as i32;

            let entry = splits_raw.entry(tmp_key).or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
            entry.0.push(mz);
            entry.1.push(intensity);
            entry.2.push(tof);
        }

        if overlapping {
            let mut splits_offset: BTreeMap<i32, (Vec<f64>, Vec<f64>, Vec<i32>)> = BTreeMap::new();

            for (i, &mmz) in self.spectrum.mz_spectrum.mz.iter().enumerate() {
                let intensity = self.spectrum.mz_spectrum.intensity[i];
                let tof = self.spectrum.index[i];

                let tmp_key = -((mmz + window_length / 2.0) / window_length).floor() as i32;

                let entry = splits_offset.entry(tmp_key).or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
                entry.0.push(mmz);
                entry.1.push(intensity);
                entry.2.push(tof);
            }

            for (key, val) in splits_offset {
                let entry = splits_raw.entry(key).or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
                entry.0.extend(val.0);
                entry.1.extend(val.1);
                entry.2.extend(val.2);
            }
        }

        // Convert to TimsSpectrum and filter
        let mut splits: BTreeMap<i32, TimsSpectrum> = BTreeMap::new();
        for (key, (mz_vec, intensity_vec, index_vec)) in splits_raw {
            if mz_vec.len() >= min_peaks {
                let max_intensity = intensity_vec.iter().cloned().max_by(
                    |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                ).unwrap_or(0.0);
                if max_intensity >= min_intensity {
                    let spectrum = IndexedMzSpectrum::new(index_vec, mz_vec, intensity_vec);
                    splits.insert(key, TimsSpectrum::new(
                        self.frame_id, self.scan, self.retention_time, self.mobility,
                        self.ms_type.clone(), spectrum
                    ));
                }
            }
        }

        splits
    }
}

// implement default (empty TimsSpectrum) constructor
impl Default for TimsSpectrum {
    fn default() -> Self {
        TimsSpectrum { frame_id: 0, scan: 0, retention_time: 0.0, mobility: 0.0, ms_type: MsType::Unknown, spectrum: IndexedMzSpectrum::default() }
    }
}

impl std::ops::Add for TimsSpectrum {
    type Output = Self;

    fn add(self, other: Self) -> TimsSpectrum {
        assert_eq!(self.frame_id, other.frame_id);

        let average_mobility = (self.mobility + other.mobility) / 2.0;
        let average_retention_time = (self.retention_time + other.retention_time) / 2.0;
        let average_scan_floor = ((self.scan as f64 + other.scan as f64) / 2.0) as i32;

        let mut combined_map: BTreeMap<i64, (f64, i32, i32)> = BTreeMap::new();
        let quantize = |mz: f64| -> i64 { (mz * 1_000_000.0).round() as i64 };

        for ((mz, intensity), index) in self.spectrum.mz_spectrum.mz.iter().zip(self.spectrum.mz_spectrum.intensity.iter()).zip(self.spectrum.index.iter()) {
            let key = quantize(*mz);
            combined_map.insert(key, (*intensity, *index, 1)); // Initialize count as 1
        }

        for ((mz, intensity), index) in other.spectrum.mz_spectrum.mz.iter().zip(other.spectrum.mz_spectrum.intensity.iter()).zip(other.spectrum.index.iter()) {
            let key = quantize(*mz);
            combined_map.entry(key).and_modify(|e| {
                e.0 += *intensity; // Sum intensity
                e.1 += *index;     // Sum index
                e.2 += 1;          // Increment count
            }).or_insert((*intensity, *index, 1));
        }

        let mz_combined: Vec<f64> = combined_map.keys().map(|&key| key as f64 / 1_000_000.0).collect();
        let intensity_combined: Vec<f64> = combined_map.values().map(|(intensity, _, _)| *intensity).collect();
        let index_combined: Vec<i32> = combined_map.values().map(|(_, index, count)| index / count).collect(); // Average index

        let spectrum = IndexedMzSpectrum { index: index_combined, mz_spectrum: MzSpectrum::new(mz_combined, intensity_combined) };
        TimsSpectrum { frame_id: self.frame_id, scan: average_scan_floor, retention_time: average_retention_time, mobility: average_mobility, ms_type: self.ms_type.clone(), spectrum }
    }
}

impl Display for TimsSpectrum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "TimsSpectrum(frame_id: {}, scan_id: {}, retention_time: {}, mobility: {}, spectrum: {})", self.frame_id, self.scan, self.retention_time, self.mobility, self.spectrum)
    }
}
