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
}

impl SourceType {
    pub fn new(source_type: i32) -> Self {
        match source_type {
            0 => SourceType::Signal,
            1 => SourceType::ChemicalNoise,
            2 => SourceType::RandomNoise,
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
        }
    }
}

#[derive(Clone, Debug)]
pub struct SignalAttributes {
    pub charge_state: i32,
    pub peptide_id: i32,
    pub isotope_peak: i32,
}
pub struct MzSpectrumAnnotated {
    pub mz: Vec<f64>,
    pub intensity: Vec<f64>,
    pub annotations: Vec<PeakAnnotation>,
}