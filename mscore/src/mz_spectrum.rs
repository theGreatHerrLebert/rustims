use std::fmt;
use std::collections::BTreeMap;
use std::fmt::Formatter;
use itertools;

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
}

/// Formats the `MzSpectrum` for display.
impl fmt::Display for MzSpectrum {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        let (mz, i) = self.mz.iter()
            .zip(&self.intensity)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        write!(f, "MzSpectrum(data points: {}, max value:({}, {}))", self.mz.len(), mz, i)
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

/// Represents a Time-Of-Flight (TOF) mass spectrum with corresponding m/z values and intensities.
#[derive(Clone)]
pub struct TOFMzSpectrum {
    pub tof: Vec<i32>,
    pub mz: Vec<f64>,
    pub intensity: Vec<f64>,
}

impl TOFMzSpectrum {
    /// Creates a new `TOFMzSpectrum` instance.
    ///
    /// # Arguments
    ///
    /// * `tof` - A vector containing the time-of-flight values.
    /// * `mz` - A vector containing the m/z values.
    /// * `intensity` - A vector containing the intensity values.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::TOFMzSpectrum;
    /// 
    /// let spectrum = TOFMzSpectrum::new(vec![1000, 2000], vec![100.5, 200.5], vec![50.0, 60.0]);
    /// ```
    pub fn new(tof: Vec<i32>, mz: Vec<f64>, intensity: Vec<f64>) -> Self {
        TOFMzSpectrum {tof, mz, intensity}
    }
    /// Bins the spectrum based on a given m/z resolution, summing intensities and averaging TOF values
    /// for m/z values that fall into the same bin.
    ///
    /// # Arguments
    ///
    /// * `resolution` - The desired m/z resolution for binning.
    ///
    /// # Examples
    ///
    /// ```
    /// use mscore::TOFMzSpectrum;
    /// 
    /// let spectrum = TOFMzSpectrum::new(vec![1000, 2000], vec![100.42, 100.43], vec![50.0, 60.0]);
    /// let binned_spectrum = spectrum.to_resolution(1);
    /// 
    /// assert_eq!(binned_spectrum.mz, vec![100.4]);
    /// assert_eq!(binned_spectrum.intensity, vec![110.0]);
    /// assert_eq!(binned_spectrum.tof, vec![1500]);
    /// ```
    pub fn to_resolution(&self, resolution: u32) -> TOFMzSpectrum {

        let mut mz_bins: BTreeMap<i64, (f64, Vec<i64>)> = BTreeMap::new();
        let factor = 10f64.powi(resolution as i32);

        for ((mz, intensity), tof_val) in self.mz.iter().zip(self.intensity.iter()).zip(&self.tof) {
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

        TOFMzSpectrum { mz, intensity, tof }
    }
}

impl fmt::Display for TOFMzSpectrum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let (mz, i) = self.mz.iter()
            .zip(&self.intensity)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        write!(f, "TOFMzSpectrum(data points: {}, max value:({}, {}))", self.mz.len(), mz, i)
    }
}


#[derive(Clone)]
pub struct ImsSpectrum {
    pub retention_time: f64,
    pub inv_mobility: f64,
    pub spectrum: MzSpectrum,
}

impl fmt::Display for ImsSpectrum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "ImsSpectrum(rt: {}, inv_mobility: {}, spectrum: {})", self.retention_time, self.inv_mobility, self.spectrum)
    }
}

#[derive(Clone)]
pub struct TimsSpectrum {
    pub frame_id: i32,
    pub scan_id: i32,
    pub retention_time: f64,
    pub inv_mobility: f64,
    pub spectrum: TOFMzSpectrum,
}

impl TimsSpectrum {
    pub fn new(frame_id: i32, scan_id: i32, retention_time: f64, inv_mobility: f64, spectrum: TOFMzSpectrum) -> Self {
        TimsSpectrum { frame_id, scan_id, retention_time, inv_mobility, spectrum }
    }
}

impl fmt::Display for TimsSpectrum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "TimsSpectrum(frame_id: {}, scan_id: {}, retention_time: {}, inv_mobility: {}, spectrum: {})", self.frame_id, self.scan_id, self.retention_time, self.inv_mobility, self.spectrum)
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
    pub retention_time: f64,
    pub scan: Vec<i32>,
    pub inv_mobility: Vec<f64>,
    pub tof: Vec<i32>,
    pub mz: Vec<f64>,
    pub intensity: Vec<f64>,
}

impl TimsFrame {
    /// Creates a new `TimsFrame` instance.
    /// 
    /// # Arguments
    /// 
    /// * `frame_id` - index of frame in TDF raw file.
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
    /// use mscore::TimsFrame;
    /// 
    /// let frame = TimsFrame::new(1, 100.0, vec![1, 2], vec![0.1, 0.2], vec![1000, 2000], vec![100.5, 200.5], vec![50.0, 60.0]);
    /// ```
    pub fn new(frame_id: i32, retention_time: f64, scan: Vec<i32>, inv_mobility: Vec<f64>, tof: Vec<i32>, mz: Vec<f64>, intensity: Vec<f64>) -> Self {
        TimsFrame { frame_id, retention_time, scan, inv_mobility, tof, mz, intensity }
    }

    pub fn to_ims_frame(&self) -> ImsFrame {
        ImsFrame { retention_time: self.retention_time, inv_mobility: self.inv_mobility.clone(), mz: self.mz.clone(), intensity: self.intensity.clone() }
    }

    pub fn to_tims_spectra(&self) -> Vec<TimsSpectrum> {
        let mut spectra = BTreeMap::<i32, (f64, Vec<i32>, Vec<f64>, Vec<f64>)>::new();

        for (scan, inv_mobility, tof, mz, intensity) in itertools::multizip((
            &self.scan,
            &self.inv_mobility,
            &self.tof,
            &self.mz,
            &self.intensity,
        )) {
            let entry = spectra.entry(*scan).or_insert_with(|| (*inv_mobility, Vec::new(), Vec::new(), Vec::new()));
            entry.1.push(*tof);
            entry.2.push(*mz);
            entry.3.push(*intensity);
        }

        let mut tims_spectra: Vec<TimsSpectrum> = Vec::new();

        for (scan, (inv_mobility, tof, mz, intensity)) in spectra {
            let spectrum = TOFMzSpectrum::new(tof, mz, intensity);
            tims_spectra.push(TimsSpectrum::new(self.frame_id, scan, self.retention_time, inv_mobility, spectrum));
        }

        tims_spectra
    }
}

impl fmt::Display for TimsFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {

        let (mz, i) = self.mz.iter()
            .zip(&self.intensity)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        write!(f, "TimsFrame(id: {}, rt: {}, data points: {}, max value: (mz: {}, intensity: {}))", self.frame_id, self.retention_time, self.scan.len(), mz, i)
    }
}


