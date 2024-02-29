use std::fmt;
use std::collections::BTreeMap;
use nalgebra::DVector;
use std::fmt::{Display, Formatter};
use serde::{Serialize, Deserialize};

/// Represents a vectorized mass spectrum.
pub trait ToResolution {
    fn to_resolution(&self, resolution: i32) -> Self;
}

/// Vectorized representation for Structs holding m/z values and intensities.
pub trait Vectorized<T> {
    fn vectorized(&self, resolution: i32) -> T;
}

/// Represents the type of spectrum.
///
/// # Description
///
/// The `SpecType` enum is used to distinguish between precursor and fragment spectra.
///
#[derive(Clone, PartialEq, Debug)]
pub enum MsType {
    Precursor,
    FragmentDda,
    FragmentDia,
    Unknown,
}

impl MsType {
    /// Returns the `MsType` enum corresponding to the given integer value.
    ///
    /// # Arguments
    ///
    /// * `ms_type` - An integer value corresponding to the `MsType` enum.
    ///
    pub fn new(ms_type: i32) -> MsType {
        match ms_type {
            0 => MsType::Precursor,
            8 => MsType::FragmentDda,
            9 => MsType::FragmentDia,
            _ => MsType::Unknown,
        }
    }

    /// Returns the integer value corresponding to the `MsType` enum.
    pub fn ms_type_numeric(&self) -> i32 {
        match self {
            MsType::Precursor => 0,
            MsType::FragmentDda => 8,
            MsType::FragmentDia => 9,
            MsType::Unknown => -1,
        }
    }
}

impl Display for MsType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            MsType::Precursor => write!(f, "Precursor"),
            MsType::FragmentDda => write!(f, "FragmentDda"),
            MsType::FragmentDia => write!(f, "FragmentDia"),
            MsType::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Represents a mass spectrum with associated m/z values and intensities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MzSpectrum {
    pub mz: Vec<f64>,
    pub intensity: Vec<f64>,
}

impl MzSpectrum {
    /// Constructs a new `MzSpectrum`.
    ///
    /// # Arguments
    ///
    /// * `mz` - A vector of m/z values.
    /// * `intensity` - A vector of intensity values corresponding to the m/z values.
    ///
    /// # Panics
    ///
    /// Panics if the lengths of `mz` and `intensity` are not the same. (actually, it doesn't at the moment, planning on adding this later)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use mscore::data::spectrum::MzSpectrum;
    /// let spectrum = MzSpectrum::new(vec![100.0, 200.0], vec![10.0, 20.0]);
    /// assert_eq!(spectrum.mz, vec![100.0, 200.0]);
    /// assert_eq!(spectrum.intensity, vec![10.0, 20.0]);
    /// ```
    pub fn new(mz: Vec<f64>, intensity: Vec<f64>) -> Self {
        MzSpectrum {mz, intensity}
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min:f64, intensity_max: f64) -> Self {
        let mut mz_vec: Vec<f64> = Vec::new();
        let mut intensity_vec: Vec<f64> = Vec::new();

        for (mz, intensity) in self.mz.iter().zip(self.intensity.iter()) {
            if mz_min <= *mz && *mz <= mz_max && *intensity >= intensity_min && *intensity <= intensity_max {
                mz_vec.push(*mz);
                intensity_vec.push(*intensity);
            }
        }
        MzSpectrum { mz: mz_vec, intensity: intensity_vec }
    }

    /// Splits the spectrum into a collection of windows based on m/z values.
    ///
    /// This function divides the spectrum into smaller spectra (windows) based on a specified window length.
    /// Each window contains peaks from the original spectrum that fall within the m/z range of that window.
    ///
    /// # Arguments
    ///
    /// * `window_length`: The size (in terms of m/z values) of each window.
    ///
    /// * `overlapping`: If `true`, each window will overlap with its neighboring windows by half of the `window_length`.
    ///   This means that a peak may belong to multiple windows. If `false`, windows do not overlap.
    ///
    /// * `min_peaks`: The minimum number of peaks a window must have to be retained in the result.
    ///
    /// * `min_intensity`: The minimum intensity value a window must have (in its highest intensity peak) to be retained in the result.
    ///
    /// # Returns
    ///
    /// A `BTreeMap` where the keys represent the window indices and the values are the spectra (`MzSpectrum`) within those windows.
    /// Windows that do not meet the criteria of having at least `min_peaks` peaks or a highest intensity peak
    /// greater than or equal to `min_intensity` are discarded.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use mscore::data::spectrum::MzSpectrum;
    /// let spectrum = MzSpectrum::new(vec![100.0, 101.0, 102.5, 103.0], vec![10.0, 20.0, 30.0, 40.0]);
    /// let windowed_spectrum = spectrum.to_windows(1.0, false, 1, 10.0);
    /// assert!(windowed_spectrum.contains_key(&100));
    /// assert!(windowed_spectrum.contains_key(&102));
    /// ```
    pub fn to_windows(&self, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64) -> BTreeMap<i32, MzSpectrum> {
        let mut splits = BTreeMap::new();

        for (i, &mz) in self.mz.iter().enumerate() {
            let intensity = self.intensity[i];

            let tmp_key = (mz / window_length).floor() as i32;

            splits.entry(tmp_key).or_insert_with(|| MzSpectrum::new(Vec::new(), Vec::new())).mz.push(mz);
            splits.entry(tmp_key).or_insert_with(|| MzSpectrum::new(Vec::new(), Vec::new())).intensity.push(intensity);
        }

        if overlapping {
            let mut splits_offset = BTreeMap::new();

            for (i, &mmz) in self.mz.iter().enumerate() {
                let intensity = self.intensity[i];

                let tmp_key = -((mmz + window_length / 2.0) / window_length).floor() as i32;

                splits_offset.entry(tmp_key).or_insert_with(|| MzSpectrum::new(Vec::new(), Vec::new())).mz.push(mmz);
                splits_offset.entry(tmp_key).or_insert_with(|| MzSpectrum::new(Vec::new(), Vec::new())).intensity.push(intensity);
            }

            for (key, val) in splits_offset {
                splits.entry(key).or_insert_with(|| MzSpectrum::new(Vec::new(), Vec::new())).mz.extend(val.mz);
                splits.entry(key).or_insert_with(|| MzSpectrum::new(Vec::new(), Vec::new())).intensity.extend(val.intensity);
            }
        }

        splits.retain(|_, spectrum| {
            spectrum.mz.len() >= min_peaks && spectrum.intensity.iter().cloned().max_by(
                |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0.0) >= min_intensity
        });

        splits
    }

    pub fn to_centroid(&self, baseline_noise_level: i32, sigma: f64, normalize: bool) -> MzSpectrum {

        let filtered = self.filter_ranged(0.0, 1e9, baseline_noise_level as f64, 1e9);

        let mut cent_mz = Vec::new();
        let mut cent_i: Vec<f64> = Vec::new();

        let mut last_mz = 0.0;
        let mut mean_mz = 0.0;
        let mut sum_i = 0.0;

        for (i, &current_mz) in filtered.mz.iter().enumerate() {
            let current_intensity = filtered.intensity[i];

            // If peak is too far away from last peak, push centroid
            if current_mz - last_mz > sigma && mean_mz > 0.0 {
                mean_mz /= sum_i;
                cent_mz.push(mean_mz);
                cent_i.push(sum_i);

                // Start new centroid
                sum_i = 0.0;
                mean_mz = 0.0;
            }

            mean_mz += current_mz * current_intensity as f64;
            sum_i += current_intensity;
            last_mz = current_mz;
        }

        // Push back last remaining centroid
        if mean_mz > 0.0 {
            mean_mz /= sum_i;
            cent_mz.push(mean_mz);
            cent_i.push(sum_i);
        }

        if normalize {
            let sum_i: f64 = cent_i.iter().sum();
            cent_i = cent_i.iter().map(|&i| i / sum_i).collect();
        }

        MzSpectrum::new(cent_mz, cent_i)
    }
}

impl ToResolution for MzSpectrum {
    /// Bins the spectrum's m/z values to a given resolution and sums the intensities.
    ///
    /// # Arguments
    ///
    /// * `resolution` - The desired resolution in terms of decimal places. For instance, a resolution of 2
    /// would bin m/z values to two decimal places.
    ///
    /// # Returns
    ///
    /// A new `MzSpectrum` where m/z values are binned according to the given resolution.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use mscore::data::spectrum::MzSpectrum;
    /// # use mscore::data::spectrum::ToResolution;
    /// let spectrum = MzSpectrum::new(vec![100.123, 100.121, 100.131], vec![10.0, 20.0, 30.0]);
    /// let binned_spectrum_1 = spectrum.to_resolution(1);
    /// let binned_spectrum_2 = spectrum.to_resolution(2);
    /// /// assert_eq!(binned_spectrum_2.mz, vec![100.1]);
    /// assert_eq!(binned_spectrum_1.intensity, vec![60.0]);
    /// assert_eq!(binned_spectrum_2.mz, vec![100.12, 100.13]);
    /// assert_eq!(binned_spectrum_2.intensity, vec![30.0, 30.0]);
    /// ```
    fn to_resolution(&self, resolution: i32) -> Self {
        let mut binned: BTreeMap<i64, f64> = BTreeMap::new();
        let factor = 10f64.powi(resolution);

        for (mz, inten) in self.mz.iter().zip(self.intensity.iter()) {

            let key = (mz * factor).round() as i64;
            let entry = binned.entry(key).or_insert(0.0);
            *entry += *inten;
        }

        let mz: Vec<f64> = binned.keys().map(|&key| key as f64 / 10f64.powi(resolution)).collect();
        let intensity: Vec<f64> = binned.values().cloned().collect();

        MzSpectrum { mz, intensity }
    }
}

impl Vectorized<MzSpectrumVectorized> for MzSpectrum {
    /// Convert the `MzSpectrum` to a `MzSpectrumVectorized` using the given resolution for binning.
    ///
    /// After binning to the desired resolution, the binned m/z values are translated into integer indices.
    fn vectorized(&self, resolution: i32) -> MzSpectrumVectorized {

        let binned_spectrum = self.to_resolution(resolution);

        // Translate the m/z values into integer indices
        let indices: Vec<i32> = binned_spectrum.mz.iter().map(|&mz| (mz * 10f64.powi(resolution)).round() as i32).collect();

        MzSpectrumVectorized {
            resolution,
            indices,
            values: binned_spectrum.intensity,
        }
    }
}

/// Formats the `MzSpectrum` for display.
impl Display for MzSpectrum {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {

        let (mz, i) = self.mz.iter()
            .zip(&self.intensity)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        write!(f, "MzSpectrum(data points: {}, max  by intensity:({}, {}))", self.mz.len(), format!("{:.3}", mz), i)
    }
}

impl std::ops::Add for MzSpectrum {
    type Output = Self;
    /// Combines two `MzSpectrum` instances by summing up the intensities of matching m/z values.
    ///
    /// # Description
    /// Each m/z value is quantized to retain at least 6 decimals. If two spectra have m/z values
    /// that quantize to the same integer value, their intensities are summed.
    ///
    /// # Example
    /// ```
    /// # use mscore::data::spectrum::MzSpectrum;
    /// let spectrum1 = MzSpectrum { mz: vec![100.523, 101.923], intensity: vec![10.0, 20.0] };
    /// let spectrum2 = MzSpectrum { mz: vec![101.235, 105.112], intensity: vec![15.0, 30.0] };
    ///
    /// let combined = spectrum1 + spectrum2;
    ///
    /// assert_eq!(combined.mz, vec![100.523, 101.235, 101.923, 105.112]);
    /// assert_eq!(combined.intensity, vec![10.0, 15.0, 20.0, 30.0]);
    /// ```
    fn add(self, other: Self) -> MzSpectrum {
        let mut combined_map: BTreeMap<i64, f64> = BTreeMap::new();

        // Helper to quantize mz to an integer key
        let quantize = |mz: f64| -> i64 {
            (mz * 1_000_000.0).round() as i64
        };

        // Add the m/z and intensities from the first spectrum to the map
        for (mz, intensity) in self.mz.iter().zip(self.intensity.iter()) {
            let key = quantize(*mz);
            combined_map.insert(key, *intensity);
        }

        // Combine the second spectrum into the map
        for (mz, intensity) in other.mz.iter().zip(other.intensity.iter()) {
            let key = quantize(*mz);
            let entry = combined_map.entry(key).or_insert(0.0);
            *entry += *intensity;
        }

        // Convert the combined map back into two Vec<f64>
        let mz_combined: Vec<f64> = combined_map.keys().map(|&key| key as f64 / 1_000_000.0).collect();
        let intensity_combined: Vec<f64> = combined_map.values().cloned().collect();

        MzSpectrum { mz: mz_combined, intensity: intensity_combined }
    }
}

impl std::ops::Mul<f64> for MzSpectrum {
    type Output = Self;
    fn mul(self, scale: f64) -> Self::Output{
        let mut scaled_intensities: Vec<f64> = vec![0.0; self.intensity.len()];
        for (idx,intensity) in self.intensity.iter().enumerate(){
            scaled_intensities[idx] = scale*intensity;
        }
        Self{ mz: self.mz.clone(), intensity: scaled_intensities}

    }
}
/// Represents a mass spectrum with associated m/z indices, m/z values, and intensities
#[derive(Clone, Debug)]
pub struct IndexedMzSpectrum {
    pub index: Vec<i32>,
    pub mz_spectrum: MzSpectrum,
}

impl IndexedMzSpectrum {
    /// Creates a new `TOFMzSpectrum` instance.
    ///
    /// # Arguments
    ///
    /// * `index` - A vector containing the mz index, e.g., time-of-flight values.
    /// * `mz` - A vector containing the m/z values.
    /// * `intensity` - A vector containing the intensity values.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::data::spectrum::IndexedMzSpectrum;
    /// use mscore::data::spectrum::MzSpectrum;
    ///
    /// let spectrum = IndexedMzSpectrum::new(vec![1000, 2000], vec![100.5, 200.5], vec![50.0, 60.0]);
    /// ```
    pub fn new(index: Vec<i32>, mz: Vec<f64>, intensity: Vec<f64>) -> Self {
        IndexedMzSpectrum { index, mz_spectrum: MzSpectrum { mz, intensity } }
    }
    /// Bins the spectrum based on a given m/z resolution, summing intensities and averaging index values
    /// for m/z values that fall into the same bin.
    ///
    /// # Arguments
    ///
    /// * `resolution` - The desired m/z resolution for binning.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::data::spectrum::IndexedMzSpectrum;
    ///
    /// let spectrum = IndexedMzSpectrum::new(vec![1000, 2000], vec![100.42, 100.43], vec![50.0, 60.0]);
    /// let binned_spectrum = spectrum.to_resolution(1);
    ///
    /// assert_eq!(binned_spectrum.mz_spectrum.mz, vec![100.4]);
    /// assert_eq!(binned_spectrum.mz_spectrum.intensity, vec![110.0]);
    /// assert_eq!(binned_spectrum.index, vec![1500]);
    /// ```
    pub fn to_resolution(&self, resolution: i32) -> IndexedMzSpectrum {

        let mut mz_bins: BTreeMap<i64, (f64, Vec<i64>)> = BTreeMap::new();
        let factor = 10f64.powi(resolution);

        for ((mz, intensity), tof_val) in self.mz_spectrum.mz.iter().zip(self.mz_spectrum.intensity.iter()).zip(&self.index) {
            let key = (mz * factor).round() as i64;
            let entry = mz_bins.entry(key).or_insert((0.0, Vec::new()));
            entry.0 += *intensity;
            entry.1.push(*tof_val as i64);
        }

        let mz: Vec<f64> = mz_bins.keys().map(|&key| key as f64 / factor).collect();
        let intensity: Vec<f64> = mz_bins.values().map(|(intensity, _)| *intensity).collect();
        let tof: Vec<i32> = mz_bins.values().map(|(_, tof_vals)| {
            let sum: i64 = tof_vals.iter().sum();
            let count: i32 = tof_vals.len() as i32;
            (sum as f64 / count as f64).round() as i32
        }).collect();

        IndexedMzSpectrum {index: tof, mz_spectrum: MzSpectrum {mz, intensity } }
    }

    /// Convert the `IndexedMzSpectrum` to a `IndexedMzVector` using the given resolution for binning.
    ///
    /// After binning to the desired resolution, the binned m/z values are translated into integer indices.
    ///
    /// # Arguments
    ///
    /// * `resolution` - The desired m/z resolution for binning.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::data::spectrum::IndexedMzSpectrum;
    ///
    /// let spectrum = IndexedMzSpectrum::new(vec![1000, 2000], vec![100.42, 100.43], vec![50.0, 60.0]);
    /// let binned_spectrum = spectrum.to_resolution(1);
    ///
    /// assert_eq!(binned_spectrum.mz_spectrum.mz, vec![100.4]);
    /// assert_eq!(binned_spectrum.mz_spectrum.intensity, vec![110.0]);
    /// assert_eq!(binned_spectrum.index, vec![1500]);
    /// ```
    pub fn vectorized(&self, resolution: i32) -> IndexedMzSpectrumVectorized {

        let binned_spectrum = self.to_resolution(resolution);

        // Translate the m/z values into integer indices
        let indices: Vec<i32> = binned_spectrum.mz_spectrum.mz.iter()
            .map(|&mz| (mz * 10f64.powi(resolution)).round() as i32).collect();

        IndexedMzSpectrumVectorized {
            index: binned_spectrum.index,
            mz_vector: MzSpectrumVectorized {
                resolution,
                indices,
                values: binned_spectrum.mz_spectrum.intensity,
            }
        }
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min:f64, intensity_max: f64) -> Self {
        let mut mz_vec: Vec<f64> = Vec::new();
        let mut intensity_vec: Vec<f64> = Vec::new();
        let mut index_vec: Vec<i32> = Vec::new();

        for ((&mz, &intensity), &index) in self.mz_spectrum.mz.iter().zip(self.mz_spectrum.intensity.iter()).zip(self.index.iter()) {
            if mz_min <= mz && mz <= mz_max && intensity >= intensity_min && intensity <= intensity_max {
                mz_vec.push(mz);
                intensity_vec.push(intensity);
                index_vec.push(index);
            }
        }
        IndexedMzSpectrum { index: index_vec, mz_spectrum: MzSpectrum { mz: mz_vec, intensity: intensity_vec } }
    }
}

impl Display for IndexedMzSpectrum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let (mz, i) = self.mz_spectrum.mz.iter()
            .zip(&self.mz_spectrum.intensity)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        write!(f, "IndexedMzSpectrum(data points: {}, max  by intensity:({}, {}))", self.mz_spectrum.mz.len(), format!("{:.3}", mz), i)
    }
}

#[derive(Clone)]
pub struct MzSpectrumVectorized {
    pub resolution: i32,
    pub indices: Vec<i32>,
    pub values: Vec<f64>,
}

impl MzSpectrumVectorized {
    /// Convert the `MzVector` to a dense vector with a specified maximum index.
    ///
    /// The resulting vector has length equal to `max_index + 1` and its values
    /// are the intensities corresponding to each index. Indices with no associated intensity will have a value of 0.
    ///
    /// # Arguments
    ///
    /// * `max_index` - The maximum index for the dense vector.
    
    fn get_max_index(&self) -> usize {
        let base: i32 = 10;
        let max_mz: i32 = 2000;
        let max_index: usize = (max_mz*base.pow(self.resolution as u32)) as usize;
        max_index
    }

    pub fn to_dense(&self, max_index: Option<usize>) -> DVector<f64> {
        let max_index = match max_index {
            Some(max_index) => max_index,
            None => self.get_max_index(),
        };
        let mut dense_intensities: DVector<f64> = DVector::<f64>::zeros(max_index + 1);
        for (&index, &intensity) in self.indices.iter().zip(self.values.iter()) {
            if (index as usize) <= max_index {
                dense_intensities[index as usize] = intensity;
            }
        }
        dense_intensities
    }
    pub fn to_dense_spectrum(&self, max_index: Option<usize>) -> MzSpectrumVectorized{
        let max_index = match max_index {
            Some(max_index) => max_index,
            None => self.get_max_index(),
        };
        let dense_intensities: Vec<f64> = self.to_dense(Some(max_index)).data.into();
        let dense_indices: Vec<i32> = (0..=max_index).map(|i| i as i32).collect();
        let dense_spectrum: MzSpectrumVectorized = MzSpectrumVectorized { resolution: (self.resolution), indices: (dense_indices), values: (dense_intensities) };
        dense_spectrum
    }
}

#[derive(Clone)]
pub struct IndexedMzSpectrumVectorized {
    pub index: Vec<i32>,
    pub mz_vector: MzSpectrumVectorized,
}



