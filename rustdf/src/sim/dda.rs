use mscore::data::peptide::{PeptideIon, PeptideProductIonSeriesCollection};
use mscore::data::spectrum::{IndexedMzSpectrum, MsType, MzSpectrum};
use mscore::simulation::annotation::{
    MzSpectrumAnnotated, TimsFrameAnnotated, TimsSpectrumAnnotated,
};
use mscore::timstof::frame::TimsFrame;
use mscore::timstof::quadrupole::{IonTransmission, TimsTransmissionDDA};
use mscore::timstof::spectrum::TimsSpectrum;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::sim::handle::TimsTofSyntheticsDataHandle;
use crate::sim::precursor::TimsTofSyntheticsPrecursorFrameBuilder;

pub struct TimsTofSyntheticsFrameBuilderDDA {
    pub path: String,
    pub precursor_frame_builder: TimsTofSyntheticsPrecursorFrameBuilder,
    pub transmission_settings: TimsTransmissionDDA,
    pub fragment_ions:
        Option<BTreeMap<(u32, i8, i8), (PeptideProductIonSeriesCollection, Vec<MzSpectrum>)>>,
    pub fragment_ions_annotated: Option<
        BTreeMap<(u32, i8, i8), (PeptideProductIonSeriesCollection, Vec<MzSpectrumAnnotated>)>,
    >,
}

impl TimsTofSyntheticsFrameBuilderDDA {
    pub fn new(path: &Path, with_annotations: bool, num_threads: usize) -> Self {

        let synthetics = TimsTofSyntheticsPrecursorFrameBuilder::new(path).unwrap();
        let handle = TimsTofSyntheticsDataHandle::new(path).unwrap();
        let transmission_settings = handle.read_pasef_meta().unwrap();
        let tims_transmission = TimsTransmissionDDA::new(transmission_settings, None);

        TimsTofSyntheticsFrameBuilderDDA {
            path: path.to_str().unwrap().to_string(),
            precursor_frame_builder: synthetics,
            transmission_settings: tims_transmission,
            fragment_ions: None,
            fragment_ions_annotated: None,
        }
    }
}