use std::collections::{BTreeMap, HashMap};
use std::fmt::Display;
use itertools::{izip, multizip};
use rand::distributions::{Uniform, Distribution};
use rand::rngs::ThreadRng;
use statrs::distribution::Normal;
use crate::data::spectrum::{MsType, ToResolution, Vectorized};

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
        // zip and sort by mz
        let mut mz_intensity_annotations: Vec<(f64, f64, PeakAnnotation)> = izip!(mz.iter(), intensity.iter(), annotations.iter()).map(|(mz, intensity, annotation)| (*mz, *intensity, annotation.clone())).collect();
        mz_intensity_annotations.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        MzSpectrumAnnotated {
            mz: mz_intensity_annotations.iter().map(|(mz, _, _)| *mz).collect(),
            intensity: mz_intensity_annotations.iter().map(|(_, intensity, _)| *intensity).collect(),
            annotations: mz_intensity_annotations.iter().map(|(_, _, annotation)| annotation.clone()).collect(),
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

    pub fn to_windows(&self, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64) -> BTreeMap<i32, MzSpectrumAnnotated> {
        let mut splits = BTreeMap::new();

        for (i, &mz) in self.mz.iter().enumerate() {
            let intensity = self.intensity[i];
            let annotation = self.annotations[i].clone();

            let tmp_key = (mz / window_length).floor() as i32;

            splits.entry(tmp_key).or_insert_with(|| MzSpectrumAnnotated::new(Vec::new(), Vec::new(), Vec::new())).mz.push(mz);
            splits.entry(tmp_key).or_insert_with(|| MzSpectrumAnnotated::new(Vec::new(), Vec::new(), Vec::new())).intensity.push(intensity);
            splits.entry(tmp_key).or_insert_with(|| MzSpectrumAnnotated::new(Vec::new(), Vec::new(), Vec::new())).annotations.push(annotation);
        }

        if overlapping {
            let mut splits_offset = BTreeMap::new();

            for (i, &mmz) in self.mz.iter().enumerate() {
                let intensity = self.intensity[i];
                let annotation = self.annotations[i].clone();

                let tmp_key = -((mmz + window_length / 2.0) / window_length).floor() as i32;

                splits_offset.entry(tmp_key).or_insert_with(|| MzSpectrumAnnotated::new(Vec::new(), Vec::new(), Vec::new())).mz.push(mmz);
                splits_offset.entry(tmp_key).or_insert_with(|| MzSpectrumAnnotated::new(Vec::new(), Vec::new(), Vec::new())).intensity.push(intensity);
                splits_offset.entry(tmp_key).or_insert_with(|| MzSpectrumAnnotated::new(Vec::new(), Vec::new(), Vec::new())).annotations.push(annotation);
            }

            for (key, val) in splits_offset {
                splits.entry(key).or_insert_with(|| MzSpectrumAnnotated::new(Vec::new(), Vec::new(), Vec::new())).mz.extend(val.mz);
                splits.entry(key).or_insert_with(|| MzSpectrumAnnotated::new(Vec::new(), Vec::new(), Vec::new())).intensity.extend(val.intensity);
                splits.entry(key).or_insert_with(|| MzSpectrumAnnotated::new(Vec::new(), Vec::new(), Vec::new())).annotations.extend(val.annotations);
            }
        }

        splits.retain(|_, spectrum| {
            spectrum.mz.len() >= min_peaks && spectrum.intensity.iter().cloned().max_by(
                |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0.0) >= min_intensity
        });

        splits
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

        let mut scaled_annotations: Vec<PeakAnnotation> = Vec::new();

        for annotation in self.annotations.iter(){
            let mut scaled_contributions: Vec<ContributionSource> = Vec::new();
            for contribution in annotation.contributions.iter(){
                let scaled_intensity = (contribution.intensity_contribution*scale).round();
                let scaled_contribution = ContributionSource{
                    intensity_contribution: scaled_intensity,
                    source_type: contribution.source_type.clone(),
                    signal_attributes: contribution.signal_attributes.clone(),
                };
                scaled_contributions.push(scaled_contribution);
            }
            let scaled_annotation = PeakAnnotation{
                contributions: scaled_contributions,
            };
            scaled_annotations.push(scaled_annotation);
        }

        MzSpectrumAnnotated { mz: self.mz.clone(), intensity: scaled_intensities, annotations: scaled_annotations }
    }
}

impl Vectorized<MzSpectrumAnnotatedVectorized> for MzSpectrumAnnotated {
    fn vectorized(&self, resolution: i32) -> MzSpectrumAnnotatedVectorized {

        let quantize = |mz: f64| -> i64 { (mz * 10.0_f64.powi(resolution)).round() as i64 };

        let binned_spec = self.to_resolution(resolution);
        let mut indices: Vec<u32> = Vec::with_capacity(binned_spec.mz.len());
        let mut values: Vec<f64> = Vec::with_capacity(binned_spec.mz.len());
        let mut annotations: Vec<PeakAnnotation> = Vec::with_capacity(binned_spec.mz.len());

        for (mz, intensity, annotation) in izip!(binned_spec.mz.iter(), binned_spec.intensity.iter(), binned_spec.annotations.iter()) {
            indices.push(quantize(*mz) as u32);
            values.push(*intensity);
            annotations.push(annotation.clone());
        }

        MzSpectrumAnnotatedVectorized {
            indices,
            values,
            annotations,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MzSpectrumAnnotatedVectorized {
    pub indices: Vec<u32>,
    pub values: Vec<f64>,
    pub annotations: Vec<PeakAnnotation>,
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
        // zip and sort by mz
        let mut mz_intensity_annotations: Vec<(u32, f64, f64, PeakAnnotation)> = izip!(tof.iter(), spectrum.mz.iter(), spectrum.intensity.iter(),
            spectrum.annotations.iter()).map(|(tof, mz, intensity, annotation)| (*tof, *mz, *intensity, annotation.clone())).collect();
        mz_intensity_annotations.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        TimsSpectrumAnnotated {
            frame_id,
            scan,
            retention_time,
            mobility,
            ms_type,
            tof: mz_intensity_annotations.iter().map(|(tof, _, _, _)| *tof).collect(),
            spectrum: MzSpectrumAnnotated {
                mz: mz_intensity_annotations.iter().map(|(_, mz, _, _)| *mz).collect(),
                intensity: mz_intensity_annotations.iter().map(|(_, _, intensity, _)| *intensity).collect(),
                annotations: mz_intensity_annotations.iter().map(|(_, _, _, annotation)| annotation.clone()).collect(),
            },
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

    pub fn to_windows(
        &self,
        window_length: f64,
        overlapping: bool,
        min_peaks: usize,
        min_intensity: f64,
    ) -> BTreeMap<i32, TimsSpectrumAnnotated> {
        // 1) base‐window buckets
        let mut buckets: BTreeMap<i32, Vec<(u32, f64, f64, PeakAnnotation)>> = BTreeMap::new();
        for (&tof, &mz, &intensity, annotation) in
            izip!(
                &self.tof,
                &self.spectrum.mz,
                &self.spectrum.intensity,
                &self.spectrum.annotations
            )
        {
            let idx = (mz / window_length).floor() as i32;
            buckets.entry(idx)
                .or_default()
                .push((tof, mz, intensity, annotation.clone()));
        }

        // 2) overlapping half‐shifted buckets
        if overlapping {
            let mut off: BTreeMap<i32, Vec<(u32, f64, f64, PeakAnnotation)>> = BTreeMap::new();
            let half = window_length / 2.0;
            for (&tof, &mz, &intensity, annotation) in
                izip!(
                    &self.tof,
                    &self.spectrum.mz,
                    &self.spectrum.intensity,
                    &self.spectrum.annotations
                )
            {
                let idx = -(((mz + half) / window_length).floor() as i32);
                off.entry(idx)
                    .or_default()
                    .push((tof, mz, intensity, annotation.clone()));
            }
            for (k, v) in off {
                buckets.entry(k).or_default().extend(v);
            }
        }

        // 3) filter & rebuild
        let mut out = BTreeMap::new();
        for (idx, group) in buckets {
            if group.len() < min_peaks {
                continue;
            }
            let max_i = group.iter().map(|&(_, _, i, _)| i).fold(0.0, f64::max);
            if max_i < min_intensity {
                continue;
            }

            // manual “unzip4”
            let mut tofs   = Vec::with_capacity(group.len());
            let mut mzs    = Vec::with_capacity(group.len());
            let mut ints   = Vec::with_capacity(group.len());
            let mut annots = Vec::with_capacity(group.len());
            for (tof, mz, intensity, annotation) in group {
                tofs.push(tof);
                mzs.push(mz);
                ints.push(intensity);
                annots.push(annotation);
            }

            // sort+rebuild the annotated spectrum
            let window_spec = MzSpectrumAnnotated::new(mzs, ints, annots);

            let sub = TimsSpectrumAnnotated {
                frame_id:       self.frame_id,
                scan:           self.scan,
                retention_time: self.retention_time,
                mobility:       self.mobility,
                ms_type:        self.ms_type.clone(),
                tof:            tofs,
                spectrum:       window_spec,
            };

            out.insert(idx, sub);
        }

        out
    }
}

impl std::ops::Add for TimsSpectrumAnnotated {
    type Output = Self;
    fn add(self, other: Self) -> Self {

        let quantize = |mz: f64| -> i64 { (mz * 1_000_000.0).round() as i64 };
        let mut spec_map: BTreeMap<i64, (u32, f64, PeakAnnotation, i64)> = BTreeMap::new();
        let mean_scan_floor = ((self.scan as f64 + other.scan as f64) / 2.0) as u32;

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
            scan: mean_scan_floor,
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

    pub fn to_tims_spectra_annotated(&self) -> Vec<TimsSpectrumAnnotated> {
        // use a sorted map where scan is used as key
        let mut spectra = BTreeMap::<i32, (f64, Vec<u32>, MzSpectrumAnnotated)>::new();

        // all indices and the intensity values are sorted by scan and stored in the map as a tuple (mobility, tof, mz, intensity)
        for (scan, mobility, tof, mz, intensity, annotations) in izip!(self.scan.iter(), self.inv_mobility.iter(), self.tof.iter(), self.mz.iter(), self.intensity.iter(), self.annotations.iter()) {
            let entry = spectra.entry(*scan as i32).or_insert_with(|| (*mobility, Vec::new(), MzSpectrumAnnotated::new(Vec::new(), Vec::new(), Vec::new())));
            entry.1.push(*tof);
            entry.2.mz.push(*mz);
            entry.2.intensity.push(*intensity);
            entry.2.annotations.push(annotations.clone());
        }

        // convert the map to a vector of TimsSpectrumAnnotated
        let mut tims_spectra: Vec<TimsSpectrumAnnotated> = Vec::new();

        for (scan, (mobility, tof, spectrum)) in spectra {
            tims_spectra.push(TimsSpectrumAnnotated::new(self.frame_id, scan as u32, self.retention_time, mobility, self.ms_type.clone(), tof, spectrum));
        }

        tims_spectra
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
    pub fn to_windows_indexed(
        &self,
        window_length: f64,
        overlapping: bool,
        min_peaks: usize,
        min_intensity: f64
    ) -> (Vec<u32>, Vec<i32>, Vec<TimsSpectrumAnnotated>) {
        // 1) explode into spectra by scan/mobility
        let spectra = self.to_tims_spectra_annotated();

        // 2) window each one
        let windows_per_scan: Vec<_> = spectra
            .iter()
            .map(|s| s.to_windows(window_length, overlapping, min_peaks, min_intensity))
            .collect();

        // 3) flatten out into three parallel vectors
        let mut scan_indices   = Vec::new();
        let mut window_indices = Vec::new();
        let mut out_spectra    = Vec::new();

        for (spec, window_map) in spectra.iter().zip(windows_per_scan.iter()) {
            for (&win_idx, win_spec) in window_map {
                scan_indices.push(spec.scan);
                window_indices.push(win_idx);
                out_spectra.push(win_spec.clone());
            }
        }

        (scan_indices, window_indices, out_spectra)
    }

    pub fn to_windows(
        &self,
        window_length: f64,
        overlapping: bool,
        min_peaks: usize,
        min_intensity: f64
    ) -> Vec<TimsSpectrumAnnotated> {
        // 1) explode into spectra by scan/mobility
        let spectra = self.to_tims_spectra_annotated();

        // 2) window each one
        let windows_per_scan: Vec<_> = spectra
            .iter()
            .map(|s| s.to_windows(window_length, overlapping, min_peaks, min_intensity))
            .collect();

        // 3) flatten out into a single vector of TimsSpectrumAnnotated
        let mut out_spectra = Vec::new();
        for (_, window_map) in spectra.iter().zip(windows_per_scan.iter()) {
            for (_, win_spec) in window_map {
                out_spectra.push(win_spec.clone());
            }
        }

        out_spectra
    }

    pub fn to_dense_windows(
        &self,
        window_length: f64,
        overlapping: bool,
        min_peaks: usize,
        min_intensity: f64,
        resolution: i32
    ) -> (Vec<f64>, Vec<i32>, Vec<i32>, usize, usize) {
        let factor    = 10f64.powi(resolution);
        let n_cols    = ((window_length * factor).round() + 1.0) as usize;

        // 1) get indexed windows
        let (scan_indices, window_indices, spectra) =
            self.to_windows_indexed(window_length, overlapping, min_peaks, min_intensity);

        // 2) vectorize each window’s MzSpectrumAnnotated
        let vec_specs: Vec<_> = spectra
            .iter()
            .map(|ts| ts.spectrum.vectorized(resolution))
            .collect();

        // 3) prepare flat matrix
        let n_rows      = spectra.len();
        let mut matrix = vec![0.0; n_rows * n_cols];

        // 4) fill in each row
        for (row, ( &win_idx, vec_spec)) in
            multizip((&window_indices, &vec_specs))
                .enumerate()
        {
            // compute the "vectorized" start index of this window
            let start_i = if win_idx >= 0 {
                ((win_idx as f64 * window_length) * factor).round() as i32
            } else {
                // negative key → half‐shifted
                ((((-win_idx) as f64 * window_length) - 0.5 * window_length) * factor)
                    .round() as i32
            };

            // now place each nonzero bin
            for (&idx, &val) in vec_spec.indices.iter().zip(&vec_spec.values) {
                let col = (idx as i32 - start_i) as usize;
                let flat_idx = row * n_cols + col;
                matrix[flat_idx] = val;
            }
        }

        // cast scan indices to i32 for consistency
        let scan_indices: Vec<i32> = scan_indices.iter().map(|&scan| scan as i32).collect();

        (matrix, scan_indices, window_indices, n_rows, n_cols)
    }

    /// Returns:
    ///  - intensity_matrix,
    ///  - scan_indices,
    ///  - window_indices,
    ///  - mz_start for each window,
    ///  - ion_mobility_start for each window,
    ///  - n_rows, n_cols,
    ///  - isotope_peak labels,
    ///  - charge_state labels,
    ///  - peptide_id labels (0..5)
    pub fn to_dense_windows_with_labels(
        &self,
        window_length: f64,
        overlapping: bool,
        min_peaks: usize,
        min_intensity: f64,
        resolution: i32,
    ) -> (
        Vec<f64>,    // intensities
        Vec<u32>,    // scan index per row
        Vec<i32>,    // window key per row
        Vec<f64>,    // mz_start per row
        Vec<f64>,    // ion_mobility_start per row
        usize,       // n_rows
        usize,       // n_cols
        Vec<i32>,    // isotope_peak labels
        Vec<i32>,    // charge_state labels
        Vec<i32>,    // peptide_id labels
    ) {
        let factor = 10f64.powi(resolution);
        let n_cols = ((window_length * factor).round() + 1.0) as usize;

        // 1) explode into per-scan windows
        let (scan_indices, window_indices, spectra) =
            self.to_windows_indexed(window_length, overlapping, min_peaks, min_intensity);
        let vec_specs: Vec<_> = spectra
            .iter()
            .map(|ts| ts.spectrum.vectorized(resolution))
            .collect();

        let n_rows     = vec_specs.len();
        let matrix_sz  = n_rows * n_cols;

        // 2) allocate output arrays
        let mut intensities       = vec![0.0_f64; matrix_sz];
        let mut iso_labels        = vec![-1_i32; matrix_sz];
        let mut charge_labels     = vec![-1_i32; matrix_sz];
        let mut peptide_labels    = vec![-1_i32; matrix_sz];
        let mut mz_start          = Vec::with_capacity(n_rows);
        let mut ion_mobility_start = Vec::with_capacity(n_rows);

        // 3) fill each row
        for (row, ((&win_idx, vec_spec), ts)) in
            multizip((&window_indices, &vec_specs))
                .zip(spectra.iter())
                .enumerate()
        {
            // record the first‐peak m/z and mobility
            let first_mz = ts.spectrum.mz.first().cloned().unwrap_or(0.0);
            mz_start.push(first_mz);
            ion_mobility_start.push(ts.mobility);

            // per‐window map for peptide_id → 0..5
            let mut feat_map  = HashMap::<i32,i32>::new();
            let mut next_feat = 0;

            // window start index
            let start_i = if win_idx >= 0 {
                ((win_idx as f64 * window_length) * factor).round() as i32
            } else {
                (((-win_idx) as f64 * window_length - 0.5 * window_length) * factor)
                    .round() as i32
            };

            // fill columns
            for (&bin_idx, &val, annotation) in
                izip!(&vec_spec.indices, &vec_spec.values, &vec_spec.annotations)
            {
                let col  = (bin_idx as i32 - start_i) as usize;
                let flat = row * n_cols + col;
                intensities[flat] = val;

                // choose best contributor
                if let Some(best) = annotation
                    .contributions
                    .iter()
                    .max_by(|a, b| {
                        a.intensity_contribution
                            .partial_cmp(&b.intensity_contribution)
                            .unwrap()
                    })
                {
                    match best.source_type {
                        SourceType::Signal => {
                            if let Some(sa) = &best.signal_attributes {
                                iso_labels[flat]    = sa.isotope_peak;
                                charge_labels[flat] = sa.charge_state;
                                // re-index peptide_id
                                let old = sa.peptide_id;
                                let new = *feat_map.entry(old).or_insert_with(|| {
                                    let i = next_feat; next_feat += 1; i.min(5)
                                });
                                peptide_labels[flat] = new;
                            }
                        }
                        SourceType::RandomNoise => {
                            iso_labels[flat]    = -2;
                            charge_labels[flat] = -2;
                            peptide_labels[flat] = -2;
                        }
                        _ => { /* leave as -1 */ }
                    }
                }
            }
        }

        (
            intensities,
            scan_indices,
            window_indices,
            mz_start,
            ion_mobility_start,
            n_rows,
            n_cols,
            iso_labels,
            charge_labels,
            peptide_labels,
        )
    }

    pub fn fold_along_scan_axis(self, fold_width: usize) -> TimsFrameAnnotated {
        // extract tims spectra from frame
        let spectra = self.to_tims_spectra_annotated();

        // create a new collection of merged spectra,where spectra are first grouped by the key they create when divided by fold_width
        // and then merge them by addition
        let mut merged_spectra: BTreeMap<u32, TimsSpectrumAnnotated> = BTreeMap::new();
        for spectrum in spectra {
            let key = spectrum.scan / fold_width as u32;

            // if the key already exists, merge the spectra
            if let Some(existing_spectrum) = merged_spectra.get_mut(&key) {

                let merged_spectrum = existing_spectrum.clone() + spectrum;
                // update the existing spectrum with the merged one
                *existing_spectrum = merged_spectrum;

            } else {
                // otherwise, insert the new spectrum
                merged_spectra.insert(key, spectrum);
            }
        }
        // convert the merged spectra back to a TimsFrame
        TimsFrameAnnotated::from_tims_spectra_annotated(merged_spectra.into_values().collect())
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