use std::collections::BTreeMap;
use std::fmt::Display;
use itertools::izip;
use rand::distributions::{Uniform, Distribution};
use rand::rngs::ThreadRng;
use statrs::distribution::Normal;
use crate::data::spectrum::{MsType, ToResolution};

#[derive(Clone, Debug)]
pub struct PeakAnnotation {
    pub contributions: Vec<ContributionSource>,
}

impl PeakAnnotation {
    pub fn new_random_noise(intensity: f64) -> Self {
        let contribution_source = ContributionSource {
            intensity_contribution: intensity,
            source_type: SourceType::RandomNoise,
            signal_attributes: None,
        };

        PeakAnnotation {
            contributions: vec![contribution_source],
        }
    }
}


#[derive(Clone, Debug)]
pub struct ContributionSource {
    pub intensity_contribution: f64,
    pub source_type: SourceType,
    pub signal_attributes: Option<SignalAttributes>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SourceType {
    Signal,
    ChemicalNoise,
    RandomNoise,
    Unknown,
}

impl SourceType {
    pub fn new(source_type: i32) -> Self {
        match source_type {
            0 => SourceType::Signal,
            1 => SourceType::ChemicalNoise,
            2 => SourceType::RandomNoise,
            3 => SourceType::Unknown,
            _ => panic!("Invalid source type"),
        }
    }
}

impl Display for SourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceType::Signal => write!(f, "Signal"),
            SourceType::ChemicalNoise => write!(f, "ChemicalNoise"),
            SourceType::RandomNoise => write!(f, "RandomNoise"),
            SourceType::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SignalAttributes {
    pub charge_state: i32,
    pub peptide_id: i32,
    pub isotope_peak: i32,
    pub description: Option<String>,
}

#[derive(Clone, Debug)]
pub struct MzSpectrumAnnotated {
    pub mz: Vec<f64>,
    pub intensity: Vec<f64>,
    pub annotations: Vec<PeakAnnotation>,
}

impl MzSpectrumAnnotated {
    pub fn new(mz: Vec<f64>, intensity: Vec<f64>, annotations: Vec<PeakAnnotation>) -> Self {
        assert!(mz.len() == intensity.len() && intensity.len() == annotations.len());
        MzSpectrumAnnotated {
            mz,
            intensity,
            annotations,
        }
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min: f64, intensity_max: f64) -> Self {
        let mut mz_filtered: Vec<f64> = Vec::new();
        let mut intensity_filtered: Vec<f64> = Vec::new();
        let mut annotations_filtered: Vec<PeakAnnotation> = Vec::new();

        for (mz, intensity, annotation) in izip!(self.mz.iter(), self.intensity.iter(), self.annotations.iter()) {
            if *mz >= mz_min && *mz <= mz_max && *intensity >= intensity_min && *intensity <= intensity_max {
                mz_filtered.push(*mz);
                intensity_filtered.push(*intensity);
                annotations_filtered.push(annotation.clone());
            }
        }
        // after filtering, the length of the mz, intensity and annotations vectors should be the same
        assert!(mz_filtered.len() == intensity_filtered.len() && intensity_filtered.len() == annotations_filtered.len());

        MzSpectrumAnnotated {
            mz: mz_filtered,
            intensity: intensity_filtered,
            annotations: annotations_filtered,
        }
    }

    pub fn add_mz_noise_uniform(&self, ppm: f64, right_drag: bool) -> Self {
        let mut rng = rand::thread_rng();
        self.add_mz_noise(ppm, &mut rng, |rng, mz, ppm| {

            let ppm_mz = match right_drag {
                true => mz * ppm / 1e6 / 2.0,
                false => mz * ppm / 1e6,
            };

            let dist = match right_drag {
                true => Uniform::from(mz - (ppm_mz / 3.0)..=mz + ppm_mz),
                false => Uniform::from(mz - ppm_mz..=mz + ppm_mz),
            };

            dist.sample(rng)
        })
    }

    pub fn add_mz_noise_normal(&self, ppm: f64) -> Self {
        let mut rng = rand::thread_rng();
        self.add_mz_noise(ppm, &mut rng, |rng, mz, ppm| {
            let ppm_mz = mz * ppm / 1e6;
            let dist = Normal::new(mz, ppm_mz / 3.0).unwrap(); // 3 sigma ? good enough?
            dist.sample(rng)
        })
    }

    fn add_mz_noise<F>(&self, ppm: f64, rng: &mut ThreadRng, noise_fn: F) -> Self
        where
            F: Fn(&mut ThreadRng, f64, f64) -> f64,
    {
        let mz: Vec<f64> = self.mz.iter().map(|&mz_value| noise_fn(rng, mz_value, ppm)).collect();
        let spectrum = MzSpectrumAnnotated { mz, intensity: self.intensity.clone(), annotations: self.annotations.clone()};

        // Sort the spectrum by m/z values and potentially sum up intensities and extend annotations at the same m/z value
        spectrum.to_resolution(6)
    }
}

impl std::ops::Add for MzSpectrumAnnotated {
    type Output = Self;
    fn add(self, other: Self) -> Self {

        let quantize = |mz: f64| -> i64 { (mz * 1_000_000.0).round() as i64 };
        let mut spec_map: BTreeMap<i64, (f64, PeakAnnotation)> = BTreeMap::new();

        for ((mz, intensity), annotation) in self.mz.iter().zip(self.intensity.iter()).zip(self.annotations.iter()) {
            let key = quantize(*mz);
            spec_map.insert(key, (*intensity, annotation.clone()));
        }

        for ((mz, intensity), annotation) in other.mz.iter().zip(other.intensity.iter()).zip(other.annotations.iter()) {
            let key = quantize(*mz);
            spec_map.entry(key).and_modify(|e| {
                e.0 += *intensity;
                e.1.contributions.extend(annotation.contributions.clone());
            }).or_insert((*intensity, annotation.clone()));
        }

        let mz: Vec<f64> = spec_map.keys().map(|&key| key as f64 / 1_000_000.0).collect();
        let intensity: Vec<f64> = spec_map.values().map(|(intensity, _)| *intensity).collect();
        let annotations: Vec<PeakAnnotation> = spec_map.values().map(|(_, annotation)| annotation.clone()).collect();

        assert!(mz.len() == intensity.len() && intensity.len() == annotations.len());

        MzSpectrumAnnotated {
            mz,
            intensity,
            annotations,
        }
    }
}

impl ToResolution for MzSpectrumAnnotated {
    fn to_resolution(&self, resolution: i32) -> Self {
        let mut spec_map: BTreeMap<i64, (f64, PeakAnnotation)> = BTreeMap::new();
        let quantize = |mz: f64| -> i64 { (mz * 10.0_f64.powi(resolution)).round() as i64 };

        for ((mz, intensity), annotation) in self.mz.iter().zip(self.intensity.iter()).zip(self.annotations.iter()) {
            let key = quantize(*mz);
            spec_map.entry(key).and_modify(|e| {
                e.0 += *intensity;
                e.1.contributions.extend(annotation.contributions.clone());
            }).or_insert((*intensity, annotation.clone()));
        }

        let mz: Vec<f64> = spec_map.keys().map(|&key| key as f64 / 10.0_f64.powi(resolution)).collect();
        let intensity: Vec<f64> = spec_map.values().map(|(intensity, _)| *intensity).collect();
        let annotations: Vec<PeakAnnotation> = spec_map.values().map(|(_, annotation)| annotation.clone()).collect();

        assert!(mz.len() == intensity.len() && intensity.len() == annotations.len());

        MzSpectrumAnnotated {
            mz,
            intensity,
            annotations,
        }
    }
}

impl std::ops::Mul<f64> for MzSpectrumAnnotated {
    type Output = Self;
    fn mul(self, scale: f64) -> Self::Output{

        let mut scaled_intensities: Vec<f64> = vec![0.0; self.intensity.len()];

        for (idx,intensity) in self.intensity.iter().enumerate(){
            scaled_intensities[idx] = scale*intensity;
        }

        MzSpectrumAnnotated { mz: self.mz.clone(), intensity: scaled_intensities, annotations: self.annotations.clone() }
    }
}

#[derive(Clone, Debug)]
pub struct TimsSpectrumAnnotated {
    pub frame_id: i32,
    pub scan: u32,
    pub retention_time: f64,
    pub mobility: f64,
    pub ms_type: MsType,
    pub tof: Vec<u32>,
    pub spectrum: MzSpectrumAnnotated,
}

impl TimsSpectrumAnnotated {
    pub fn new(frame_id: i32, scan: u32, retention_time: f64, mobility: f64, ms_type: MsType, tof: Vec<u32>, spectrum: MzSpectrumAnnotated) -> Self {
        assert!(tof.len() == spectrum.mz.len() && spectrum.mz.len() == spectrum.intensity.len() && spectrum.intensity.len() == spectrum.annotations.len());
        TimsSpectrumAnnotated {
            frame_id,
            scan,
            retention_time,
            mobility,
            ms_type,
            tof,
            spectrum,
        }
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min: f64, intensity_max: f64) -> Self {
        let mut tof_filtered: Vec<u32> = Vec::new();
        let mut mz_filtered: Vec<f64> = Vec::new();
        let mut intensity_filtered: Vec<f64> = Vec::new();
        let mut annotations_filtered: Vec<PeakAnnotation> = Vec::new();

        for (tof, mz, intensity, annotation) in izip!(self.tof.iter(), self.spectrum.mz.iter(), self.spectrum.intensity.iter(), self.spectrum.annotations.iter()) {
            if *mz >= mz_min && *mz <= mz_max && *intensity >= intensity_min && *intensity <= intensity_max {
                tof_filtered.push(*tof);
                mz_filtered.push(*mz);
                intensity_filtered.push(*intensity);
                annotations_filtered.push(annotation.clone());
            }
        }

        assert!(tof_filtered.len() == mz_filtered.len() && mz_filtered.len() == intensity_filtered.len() && intensity_filtered.len() == annotations_filtered.len());

        TimsSpectrumAnnotated {
            frame_id: self.frame_id,
            scan: self.scan,
            retention_time: self.retention_time,
            mobility: self.mobility,
            ms_type: self.ms_type.clone(),
            tof: tof_filtered,
            spectrum: MzSpectrumAnnotated::new(mz_filtered, intensity_filtered, annotations_filtered),
        }
    }

    pub fn add_mz_noise_uniform(&self, ppm: f64, right_drag: bool) -> Self {
        TimsSpectrumAnnotated {
            frame_id: self.frame_id,
            scan: self.scan,
            retention_time: self.retention_time,
            mobility: self.mobility,
            ms_type: self.ms_type.clone(),
            // TODO: adding noise to mz means that TOF values need to be re-calculated
            tof: self.tof.clone(),
            spectrum: self.spectrum.add_mz_noise_uniform(ppm, right_drag),
        }
    }

    pub fn add_mz_noise_normal(&self, ppm: f64) -> Self {
        TimsSpectrumAnnotated {
            frame_id: self.frame_id,
            scan: self.scan,
            retention_time: self.retention_time,
            mobility: self.mobility,
            ms_type: self.ms_type.clone(),
            // TODO: adding noise to mz means that TOF values need to be re-calculated
            tof: self.tof.clone(),
            spectrum: self.spectrum.add_mz_noise_normal(ppm),
        }
    }
}

impl std::ops::Add for TimsSpectrumAnnotated {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        assert_eq!(self.scan, other.scan);

        let quantize = |mz: f64| -> i64 { (mz * 1_000_000.0).round() as i64 };
        let mut spec_map: BTreeMap<i64, (u32, f64, PeakAnnotation, i64)> = BTreeMap::new();

        for (tof, mz, intensity, annotation) in izip!(self.tof.iter(), self.spectrum.mz.iter(), self.spectrum.intensity.iter(), self.spectrum.annotations.iter()) {
            let key = quantize(*mz);
            spec_map.insert(key, (*tof, *intensity, annotation.clone(), 1));
        }

        for (tof, mz, intensity, annotation) in izip!(other.tof.iter(), other.spectrum.mz.iter(), other.spectrum.intensity.iter(), other.spectrum.annotations.iter()) {
            let key = quantize(*mz);
            spec_map.entry(key).and_modify(|e| {
                e.0 += *tof;
                e.1 += *intensity;
                e.2.contributions.extend(annotation.contributions.clone());
                e.3 += 1;
            }).or_insert((*tof, *intensity, annotation.clone(), 1));
        }

        let mut tof_vec: Vec<u32> = Vec::with_capacity(spec_map.len());
        let mut mz_vec: Vec<f64> = Vec::with_capacity(spec_map.len());
        let mut intensity_vec: Vec<f64> = Vec::with_capacity(spec_map.len());
        let mut annotations_vec: Vec<PeakAnnotation> = Vec::with_capacity(spec_map.len());

        for (key, (tof, intensity, annotation, count)) in spec_map.iter() {
            tof_vec.push((*tof as f64 / *count as f64) as u32);
            mz_vec.push(*key as f64 / 1_000_000.0);
            intensity_vec.push(*intensity / *count as f64);
            annotations_vec.push(annotation.clone());
        }

        assert!(tof_vec.len() == mz_vec.len() && mz_vec.len() == intensity_vec.len() && intensity_vec.len() == annotations_vec.len());

        TimsSpectrumAnnotated {
            frame_id: self.frame_id,
            scan: self.scan,
            retention_time: self.retention_time,
            mobility: self.mobility,
            ms_type: if self.ms_type == other.ms_type { self.ms_type.clone() } else { MsType::Unknown },
            tof: tof_vec,
            spectrum: MzSpectrumAnnotated::new(mz_vec, intensity_vec, annotations_vec),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TimsFrameAnnotated {
    pub frame_id: i32,
    pub retention_time: f64,
    pub ms_type: MsType,
    pub tof: Vec<u32>,
    pub mz: Vec<f64>,
    pub scan: Vec<u32>,
    pub inv_mobility: Vec<f64>,
    pub intensity: Vec<f64>,
    pub annotations: Vec<PeakAnnotation>,
}

impl TimsFrameAnnotated {
    pub fn new(frame_id: i32, retention_time: f64, ms_type: MsType, tof: Vec<u32>, mz: Vec<f64>, scan: Vec<u32>, inv_mobility: Vec<f64>, intensity: Vec<f64>, annotations: Vec<PeakAnnotation>) -> Self {
        assert!(tof.len() == mz.len() && mz.len() == scan.len() && scan.len() == inv_mobility.len() && inv_mobility.len() == intensity.len() && intensity.len() == annotations.len());
        TimsFrameAnnotated {
            frame_id,
            retention_time,
            ms_type,
            tof,
            mz,
            scan,
            inv_mobility,
            intensity,
            annotations,
        }
    }
    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, inv_mobility_min: f64, inv_mobility_max: f64, scan_min: u32, scan_max: u32, intensity_min: f64, intensity_max: f64) -> Self {
        let mut tof_filtered: Vec<u32> = Vec::new();
        let mut mz_filtered: Vec<f64> = Vec::new();
        let mut scan_filtered: Vec<u32> = Vec::new();
        let mut inv_mobility_filtered: Vec<f64> = Vec::new();
        let mut intensity_filtered: Vec<f64> = Vec::new();
        let mut annotations_filtered: Vec<PeakAnnotation> = Vec::new();

        for (tof, mz, scan, inv_mobility, intensity, annotation) in izip!(self.tof.iter(), self.mz.iter(), self.scan.iter(), self.inv_mobility.iter(), self.intensity.iter(), self.annotations.iter()) {
            if *mz >= mz_min && *mz <= mz_max && *inv_mobility >= inv_mobility_min && *inv_mobility <= inv_mobility_max && *scan >= scan_min && *scan <= scan_max && *intensity >= intensity_min && *intensity <= intensity_max {
                tof_filtered.push(*tof);
                mz_filtered.push(*mz);
                scan_filtered.push(*scan);
                inv_mobility_filtered.push(*inv_mobility);
                intensity_filtered.push(*intensity);
                annotations_filtered.push(annotation.clone());
            }
        }

        assert!(tof_filtered.len() == mz_filtered.len() && mz_filtered.len() == scan_filtered.len() && scan_filtered.len() == inv_mobility_filtered.len() && inv_mobility_filtered.len() == intensity_filtered.len() && intensity_filtered.len() == annotations_filtered.len());

        TimsFrameAnnotated {
            frame_id: self.frame_id,
            retention_time: self.retention_time,
            ms_type: self.ms_type.clone(),
            tof: tof_filtered,
            mz: mz_filtered,
            scan: scan_filtered,
            inv_mobility: inv_mobility_filtered,
            intensity: intensity_filtered,
            annotations: annotations_filtered,
        }
    }

    pub fn from_tims_spectra_annotated(spectra: Vec<TimsSpectrumAnnotated>) -> TimsFrameAnnotated {
        let quantize = |mz: f64| -> i64 { (mz * 1_000_000.0).round() as i64 };
        let mut spec_map: BTreeMap<(u32, i64), (f64, u32, f64, PeakAnnotation, i64)> = BTreeMap::new();
        let mut capacity_count = 0;

        for spectrum in &spectra {
            let inv_mobility = spectrum.mobility;
            for (i, mz) in spectrum.spectrum.mz.iter().enumerate() {
                let scan = spectrum.scan;
                let tof = spectrum.tof[i];
                let intensity = spectrum.spectrum.intensity[i];
                let annotation = spectrum.spectrum.annotations[i].clone();
                let key = (scan, quantize(*mz));
                spec_map.entry(key).and_modify(|e| {
                    e.0 += intensity;
                    e.1 += tof;
                    e.2 += inv_mobility;
                    e.3.contributions.extend(annotation.contributions.clone());
                    e.4 += 1;
                }).or_insert((intensity, tof, inv_mobility, annotation, 1));
                capacity_count += 1;
            }
        }

        let mut scan_vec: Vec<u32> = Vec::with_capacity(capacity_count);
        let mut inv_mobility_vec: Vec<f64> = Vec::with_capacity(capacity_count);
        let mut tof_vec: Vec<u32> = Vec::with_capacity(capacity_count);
        let mut mz_vec: Vec<f64> = Vec::with_capacity(capacity_count);
        let mut intensity_vec: Vec<f64> = Vec::with_capacity(capacity_count);
        let mut annotations_vec: Vec<PeakAnnotation> = Vec::with_capacity(capacity_count);

        for ((scan, mz), (intensity, tof, inv_mobility, annotation, count)) in spec_map.iter() {
            scan_vec.push(*scan);
            inv_mobility_vec.push(*inv_mobility / *count as f64);
            tof_vec.push((*tof as f64 / *count as f64) as u32);
            mz_vec.push(*mz as f64 / 1_000_000.0);
            intensity_vec.push(*intensity);
            annotations_vec.push(annotation.clone());
        }

        assert!(tof_vec.len() == mz_vec.len() && mz_vec.len() == scan_vec.len() && scan_vec.len() == inv_mobility_vec.len() && inv_mobility_vec.len() == intensity_vec.len() && intensity_vec.len() == annotations_vec.len());

        TimsFrameAnnotated {
            frame_id: spectra.first().unwrap().frame_id,
            retention_time: spectra.first().unwrap().retention_time,
            ms_type: spectra.first().unwrap().ms_type.clone(),
            tof: tof_vec,
            mz: mz_vec,
            scan: scan_vec,
            inv_mobility: inv_mobility_vec,
            intensity: intensity_vec,
            annotations: annotations_vec,
        }
    }
}

impl std::ops::Add for TimsFrameAnnotated {
    type Output = Self;
    fn add(self, other: Self) -> Self {

        let quantize = |mz: f64| -> i64 { (mz * 1_000_000.0).round() as i64 };
        let mut spec_map: BTreeMap<(u32, i64), (f64, u32, f64, PeakAnnotation, i64)> = BTreeMap::new();

        for (scan, mz, tof, inv_mobility, intensity, annotation) in
        izip!(self.scan.iter(), self.mz.iter(), self.tof.iter(), self.inv_mobility.iter(), self.intensity.iter(), self.annotations.iter()) {
            let key = (*scan, quantize(*mz));
            spec_map.insert(key, (*intensity, *tof, *inv_mobility, annotation.clone(), 1));
        }

        for (scan, mz, tof, inv_mobility, intensity, annotation) in
        izip!(other.scan.iter(), other.mz.iter(), other.tof.iter(), other.inv_mobility.iter(), other.intensity.iter(), other.annotations.iter()) {
            let key = (*scan, quantize(*mz));
            spec_map.entry(key).and_modify(|e| {
                e.0 += *intensity;
                e.1 += *tof;
                e.2 += *inv_mobility;
                e.3.contributions.extend(annotation.contributions.clone());
                e.4 += 1;
            }).or_insert((*intensity, *tof, *inv_mobility, annotation.clone(), 1));
        }

        let mut tof_vec: Vec<u32> = Vec::with_capacity(spec_map.len());
        let mut mz_vec: Vec<f64> = Vec::with_capacity(spec_map.len());
        let mut scan_vec: Vec<u32> = Vec::with_capacity(spec_map.len());
        let mut inv_mobility_vec: Vec<f64> = Vec::with_capacity(spec_map.len());
        let mut intensity_vec: Vec<f64> = Vec::with_capacity(spec_map.len());
        let mut annotations_vec: Vec<PeakAnnotation> = Vec::with_capacity(spec_map.len());

        for ((scan, mz), (intensity, tof, inv_mobility, annotation, count)) in spec_map.iter() {
            scan_vec.push(*scan);
            mz_vec.push(*mz as f64 / 1_000_000.0);
            intensity_vec.push(*intensity);
            tof_vec.push((*tof as f64 / *count as f64) as u32);
            inv_mobility_vec.push(*inv_mobility / *count as f64);
            annotations_vec.push(annotation.clone());
        }

        assert!(tof_vec.len() == mz_vec.len() && mz_vec.len() == scan_vec.len() && scan_vec.len() == inv_mobility_vec.len() && inv_mobility_vec.len() == intensity_vec.len() && intensity_vec.len() == annotations_vec.len());

        TimsFrameAnnotated {
            frame_id: self.frame_id,
            retention_time: self.retention_time,
            ms_type: if self.ms_type == other.ms_type { self.ms_type.clone() } else { MsType::Unknown },
            tof: tof_vec,
            mz: mz_vec,
            scan: scan_vec,
            inv_mobility: inv_mobility_vec,
            intensity: intensity_vec,
            annotations: annotations_vec,
        }
    }
}