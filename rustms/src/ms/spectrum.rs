use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

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
    /// # use rustms::ms::spectrum::MzSpectrum;
    /// let spectrum = MzSpectrum::new(vec![200.0, 100.0], vec![20.0, 10.0]);
    /// assert_eq!(spectrum.mz, vec![100.0, 200.0]);
    /// assert_eq!(spectrum.intensity, vec![10.0, 20.0]);
    /// ```
    pub fn new(mz: Vec<f64>, intensity: Vec<f64>) -> Self {
        assert_eq!(mz.len(), intensity.len(), "mz and intensity vectors must have the same length");
        // make sure mz and intensity are sorted by mz
        let mut mz_intensity: Vec<(f64, f64)> = mz.iter().zip(intensity.iter()).map(|(m, i)| (*m, *i)).collect();
        mz_intensity.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        MzSpectrum { mz: mz_intensity.iter().map(|(m, _)| *m).collect(), intensity: mz_intensity.iter().map(|(_, i)| *i).collect() }
    }

    /// Filters the m/z values and intensities based on a range of m/z values and intensities.
    ///
    /// # Arguments
    ///
    /// * `mz_min` - The minimum m/z value.
    /// * `mz_max` - The maximum m/z value.
    /// * `intensity_min` - The minimum intensity value.
    /// * `intensity_max` - The maximum intensity value.
    ///
    /// # Returns
    ///
    /// * `MzSpectrum` - A new `MzSpectrum` with m/z values and intensities within the specified ranges.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rustms::ms::spectrum::MzSpectrum;
    /// let spectrum = MzSpectrum::new(vec![100.0, 200.0, 300.0], vec![10.0, 20.0, 30.0]);
    /// let filtered_spectrum = spectrum.filter_ranged(150.0, 250.0, 15.0, 25.0);
    /// assert_eq!(filtered_spectrum.mz, vec![200.0]);
    /// assert_eq!(filtered_spectrum.intensity, vec![20.0]);
    /// ```
    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min: f64, intensity_max: f64) -> Self {
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

    pub fn from_collection(collection: Vec<MzSpectrum>) -> MzSpectrum {

        let quantize = |mz: f64| -> i64 {
            (mz * 1_000_000.0).round() as i64
        };

        let mut combined_map: BTreeMap<i64, f64> = BTreeMap::new();

        for spectrum in collection {
            for (mz, intensity) in spectrum.mz.iter().zip(spectrum.intensity.iter()) {
                let key = quantize(*mz);
                let entry = combined_map.entry(key).or_insert(0.0);
                *entry += *intensity;
            }
        }

        let mz_combined: Vec<f64> = combined_map.keys().map(|&key| key as f64 / 1_000_000.0).collect();
        let intensity_combined: Vec<f64> = combined_map.values().cloned().collect();

        MzSpectrum { mz: mz_combined, intensity: intensity_combined }
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
    /// # use rustms::ms::spectrum::MzSpectrum;
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

impl std::ops::Sub for MzSpectrum {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
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
            *entry -= *intensity;
        }

        // Convert the combined map back into two Vec<f64>
        let mz_combined: Vec<f64> = combined_map.keys().map(|&key| key as f64 / 1_000_000.0).collect();
        let intensity_combined: Vec<f64> = combined_map.values().cloned().collect();

        MzSpectrum { mz: mz_combined, intensity: intensity_combined }
    }
}