use std::collections::BTreeMap;
use std::fmt::Display;

#[derive(Clone, Debug)]
pub struct PeakAnnotation {
    pub contributions: Vec<ContributionSource>,
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

        MzSpectrumAnnotated {
            mz,
            intensity,
            annotations,
        }
    }
}