use std::fmt;
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

/// Represents the type of spectrum.
///
/// # Description
///
/// The `SpecType` enum is used to distinguish between precursor and fragment spectra.
///
#[derive(Clone)]
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
#[derive(Clone)]
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
    /// # use mscore::MzSpectrum;
    /// let spectrum = MzSpectrum::new(vec![100.0, 200.0], vec![10.0, 20.0]);
    /// assert_eq!(spectrum.mz, vec![100.0, 200.0]);
    /// assert_eq!(spectrum.intensity, vec![10.0, 20.0]);
    /// ```
    pub fn new(mz: Vec<f64>, intensity: Vec<f64>) -> Self {
        
        MzSpectrum {mz, intensity}
    }

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
    /// # use mscore::MzSpectrum;
    /// let spectrum = MzSpectrum::new(vec![100.123, 100.121, 100.131], vec![10.0, 20.0, 30.0]);
    /// let binned_spectrum_1 = spectrum.to_resolution(1);
    /// let binned_spectrum_2 = spectrum.to_resolution(2);
    /// /// assert_eq!(binned_spectrum_2.mz, vec![100.1]);
    /// assert_eq!(binned_spectrum_1.intensity, vec![60.0]);
    /// assert_eq!(binned_spectrum_2.mz, vec![100.12, 100.13]);
    /// assert_eq!(binned_spectrum_2.intensity, vec![30.0, 30.0]);
    /// ```
    pub fn to_resolution(&self, resolution: u32) -> MzSpectrum {
        let mut binned: BTreeMap<i64, f64> = BTreeMap::new();
        let factor = 10f64.powi(resolution as i32);

        for (mz, inten) in self.mz.iter().zip(self.intensity.iter()) {
            
            let key = (mz * factor).round() as i64;
            let entry = binned.entry(key).or_insert(0.0);
            *entry += *inten;
        }

        let mz: Vec<f64> = binned.keys().map(|&key| key as f64 / 10f64.powi(resolution as i32)).collect();
        let intensity: Vec<f64> = binned.values().cloned().collect();

        MzSpectrum { mz, intensity }
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min:f64) -> Self {
        let mut mz_vec: Vec<f64> = Vec::new();
        let mut intensity_vec: Vec<f64> = Vec::new();

        for (mz, intensity) in self.mz.iter().zip(self.intensity.iter()) {
            if mz_min <= *mz && *mz <= mz_max && *intensity >= intensity_min {
                mz_vec.push(*mz);
                intensity_vec.push(*intensity);
            }
        }
        MzSpectrum { mz: mz_vec, intensity: intensity_vec }
    }
}

/// Formats the `MzSpectrum` for display.
impl fmt::Display for MzSpectrum {
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
    /// # use mscore::MzSpectrum;
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

/// Represents a mass spectrum with associated m/z indices, m/z values, and intensities
#[derive(Clone)]
pub struct IndexedMzSpectrum {
    pub index: Vec<i32>,
    pub mz: Vec<f64>,
    pub intensity: Vec<f64>,
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
    /// use mscore::IndexedMzSpectrum;
    ///
    /// let spectrum = IndexedMzSpectrum::new(vec![1000, 2000], vec![100.5, 200.5], vec![50.0, 60.0]);
    /// ```
    pub fn new(tof: Vec<i32>, mz: Vec<f64>, intensity: Vec<f64>) -> Self {
        IndexedMzSpectrum { index: tof, mz, intensity}
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
    /// use mscore::IndexedMzSpectrum;
    ///
    /// let spectrum = IndexedMzSpectrum::new(vec![1000, 2000], vec![100.42, 100.43], vec![50.0, 60.0]);
    /// let binned_spectrum = spectrum.to_resolution(1);
    ///
    /// assert_eq!(binned_spectrum.mz, vec![100.4]);
    /// assert_eq!(binned_spectrum.intensity, vec![110.0]);
    /// assert_eq!(binned_spectrum.index, vec![1500]);
    /// ```
    pub fn to_resolution(&self, resolution: u32) -> IndexedMzSpectrum {

        let mut mz_bins: BTreeMap<i64, (f64, Vec<i64>)> = BTreeMap::new();
        let factor = 10f64.powi(resolution as i32);

        for ((mz, intensity), tof_val) in self.mz.iter().zip(self.intensity.iter()).zip(&self.index) {
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

        IndexedMzSpectrum { mz, intensity, index: tof }
    }
}

impl fmt::Display for IndexedMzSpectrum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let (mz, i) = self.mz.iter()
            .zip(&self.intensity)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        write!(f, "IndexedMzSpectrum(data points: {}, max  by intensity:({}, {}))", self.mz.len(), format!("{:.3}", mz), i)
    }
}


#[derive(Clone)]
pub struct ImsSpectrum {
    pub retention_time: f64,
    pub inv_mobility: f64,
    pub spectrum: MzSpectrum,
}

impl ImsSpectrum {
    ///
    /// Creates a new `ImsSpectrum` instance.
    ///
    /// # Arguments
    ///
    /// * `retention_time` - The retention time in seconds.
    /// * `inv_mobility` - The inverse ion mobility.
    /// * `spectrum` - A `MzSpectrum` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::{ImsSpectrum, MzSpectrum};
    ///
    /// let spectrum = ImsSpectrum::new(100.0, 0.1, MzSpectrum::new(vec![100.5, 200.5], vec![50.0, 60.0]));
    /// ```
    pub fn new(retention_time: f64, inv_mobility: f64, spectrum: MzSpectrum) -> Self {
        ImsSpectrum { retention_time, inv_mobility, spectrum }
    }
}

impl fmt::Display for ImsSpectrum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "ImsSpectrum(rt: {}, inv_mobility: {}, spectrum: {})", self.retention_time, self.inv_mobility, self.spectrum)
    }
}

#[derive(Clone)]
pub struct TimsSpectrum {
    pub frame_id: i32,
    pub scan: i32,
    pub retention_time: f64,
    pub inv_mobility: f64,
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
    /// * `inv_mobility` - The inverse ion mobility.
    /// * `spectrum` - A `TOFMzSpectrum` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::{TimsSpectrum, IndexedMzSpectrum};
    ///
    /// let spectrum = TimsSpectrum::new(1, 1, 100.0, 0.1, IndexedMzSpectrum::new(vec![1000, 2000], vec![100.5, 200.5], vec![50.0, 60.0]));
    /// ```
    pub fn new(frame_id: i32, scan_id: i32, retention_time: f64, inv_mobility: f64, spectrum: IndexedMzSpectrum) -> Self {
        TimsSpectrum { frame_id, scan: scan_id, retention_time, inv_mobility, spectrum }
    }
}

impl fmt::Display for TimsSpectrum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "TimsSpectrum(frame_id: {}, scan_id: {}, retention_time: {}, inv_mobility: {}, spectrum: {})", self.frame_id, self.scan, self.retention_time, self.inv_mobility, self.spectrum)
    }
}


