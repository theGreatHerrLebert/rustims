use mscore::data::peptide::PeptideIon;
use mscore::data::spectrum::{IndexedMzSpectrum, MsType, MzSpectrum};
use mscore::simulation::annotation::{
    MzSpectrumAnnotated, PeakAnnotation, TimsFrameAnnotated, TimsSpectrumAnnotated,
};
use mscore::timstof::frame::TimsFrame;
use mscore::timstof::spectrum::TimsSpectrum;
use rusqlite::Result;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use crate::sim::containers::{FramesSim, IonSim, PeptidesSim, ScansSim};
use crate::sim::handle::TimsTofSyntheticsDataHandle;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

pub struct TimsTofSyntheticsPrecursorFrameBuilder {
    pub ions: BTreeMap<u32, Vec<IonSim>>,
    pub peptides: BTreeMap<u32, PeptidesSim>,
    pub scans: Vec<ScansSim>,
    pub frames: Vec<FramesSim>,
    pub precursor_frame_id_set: HashSet<u32>,
    pub frame_to_abundances: BTreeMap<u32, (Vec<u32>, Vec<f32>)>,
    pub peptide_to_ions: BTreeMap<
        u32,
        (
            Vec<f32>,
            Vec<Vec<u32>>,
            Vec<Vec<f32>>,
            Vec<i8>,
            Vec<MzSpectrum>,
        ),
    >,
    pub frame_to_rt: BTreeMap<u32, f32>,
    pub scan_to_mobility: BTreeMap<u32, f32>,
    pub peptide_to_events: BTreeMap<u32, f32>,
}

impl TimsTofSyntheticsPrecursorFrameBuilder {
    /// Create a new instance of TimsTofSynthetics
    ///
    /// # Arguments
    ///
    /// * `path` - A reference to a Path
    ///
    /// # Returns
    ///
    /// * A Result containing the TimsTofSynthetics instance
    ///
    pub fn new(path: &Path) -> Result<Self> {
        let handle = TimsTofSyntheticsDataHandle::new(path)?;
        let ions = handle.read_ions()?;
        let peptides = handle.read_peptides()?;
        let scans = handle.read_scans()?;
        let frames = handle.read_frames()?;
        Ok(Self {
            ions: TimsTofSyntheticsDataHandle::build_peptide_to_ion_map(&ions),
            peptides: TimsTofSyntheticsDataHandle::build_peptide_map(&peptides),
            scans: scans.clone(),
            frames: frames.clone(),
            precursor_frame_id_set: TimsTofSyntheticsDataHandle::build_precursor_frame_id_set(
                &frames,
            ),
            frame_to_abundances: TimsTofSyntheticsDataHandle::build_frame_to_abundances(&peptides),
            peptide_to_ions: TimsTofSyntheticsDataHandle::build_peptide_to_ions(&ions),
            frame_to_rt: TimsTofSyntheticsDataHandle::build_frame_to_rt(&frames),
            scan_to_mobility: TimsTofSyntheticsDataHandle::build_scan_to_mobility(&scans),
            peptide_to_events: TimsTofSyntheticsDataHandle::build_peptide_to_events(&peptides),
        })
    }

    /// Build a precursor frame
    ///
    /// # Arguments
    ///
    /// * `frame_id` - A u32 representing the frame id
    ///
    /// # Returns
    ///
    /// * A TimsFrame instance
    pub fn build_precursor_frame(
        &self,
        frame_id: u32,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        right_drag: bool,
    ) -> TimsFrame {
        let ms_type = match self.precursor_frame_id_set.contains(&frame_id) {
            true => MsType::Precursor,
            false => MsType::Unknown,
        };

        let mut tims_spectra: Vec<TimsSpectrum> = Vec::new();

        // Frame might not have any peptides
        if !self.frame_to_abundances.contains_key(&frame_id) {
            return TimsFrame::new(
                frame_id as i32,
                ms_type.clone(),
                *self.frame_to_rt.get(&frame_id).unwrap() as f64,
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            );
        }
        // Get the peptide ids and abundances for the frame, should now save to unwrap since we checked if the frame is in the map
        let (peptide_ids, abundances) = self.frame_to_abundances.get(&frame_id).unwrap();

        // go over all peptides and their abundances in the frame
        for (peptide_id, abundance) in peptide_ids.iter().zip(abundances.iter()) {
            // jump to next peptide if the peptide_id is not in the peptide_to_ions map
            if !self.peptide_to_ions.contains_key(&peptide_id) {
                continue;
            }

            // one peptide can have multiple ions, occurring in multiple scans
            let (ion_abundances, scan_occurrences, scan_abundances, _, spectra) =
                self.peptide_to_ions.get(&peptide_id).unwrap();

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let scan_occurrence = scan_occurrences.get(index).unwrap();
                let scan_abundance = scan_abundances.get(index).unwrap();
                let spectrum = spectra.get(index).unwrap();

                for (scan, scan_abu) in scan_occurrence.iter().zip(scan_abundance.iter()) {
                    let abundance_factor = abundance
                        * ion_abundance
                        * scan_abu
                        * self.peptide_to_events.get(&peptide_id).unwrap();
                    let scan_id = *scan;
                    let scaled_spec: MzSpectrum = spectrum.clone() * abundance_factor as f64;

                    let mz_spectrum = if mz_noise_precursor {
                        match uniform {
                            true => {
                                scaled_spec.add_mz_noise_uniform(precursor_noise_ppm, right_drag)
                            }
                            false => scaled_spec.add_mz_noise_normal(precursor_noise_ppm),
                        }
                    } else {
                        scaled_spec
                    };

                    let tims_spec = TimsSpectrum::new(
                        frame_id as i32,
                        *scan as i32,
                        *self.frame_to_rt.get(&frame_id).unwrap() as f64,
                        *self.scan_to_mobility.get(&scan_id).unwrap() as f64,
                        ms_type.clone(),
                        IndexedMzSpectrum::from_mz_spectrum(
                            vec![0; mz_spectrum.mz.len()],
                            mz_spectrum,
                        ),
                    );
                    tims_spectra.push(tims_spec);
                }
            }
        }

        let tims_frame = TimsFrame::from_tims_spectra(tims_spectra);

        tims_frame.filter_ranged(0.0, 10000.0, 0, 2000, 0.0, 10.0, 1.0, 1e9)
    }

    /// Build a collection of precursor frames in parallel
    ///
    /// # Arguments
    ///
    /// * `frame_ids` - A vector of u32 representing the frame ids
    /// * `num_threads` - A usize representing the number of threads
    ///
    /// # Returns
    ///
    /// * A vector of TimsFrame instances
    ///
    pub fn build_precursor_frames(
        &self,
        frame_ids: Vec<u32>,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        right_drag: bool,
        num_threads: usize,
    ) -> Vec<TimsFrame> {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        let mut tims_frames: Vec<TimsFrame> = Vec::new();

        thread_pool.install(|| {
            tims_frames = frame_ids
                .par_iter()
                .map(|frame_id| {
                    self.build_precursor_frame(
                        *frame_id,
                        mz_noise_precursor,
                        uniform,
                        precursor_noise_ppm,
                        right_drag,
                    )
                })
                .collect();
        });

        tims_frames.sort_by(|a, b| a.frame_id.cmp(&b.frame_id));

        tims_frames
    }

    pub fn build_precursor_frame_annotated(
        &self,
        frame_id: u32,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        right_drag: bool,
    ) -> TimsFrameAnnotated {
        let ms_type = match self.precursor_frame_id_set.contains(&frame_id) {
            true => MsType::Precursor,
            false => MsType::Unknown,
        };

        // no peptides in the frame
        if !self.frame_to_abundances.contains_key(&frame_id) {
            return TimsFrameAnnotated::new(
                frame_id as i32,
                0.0,
                ms_type.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            );
        }

        let (peptide_ids, abundances) = self.frame_to_abundances.get(&frame_id).unwrap();
        let mut tims_spectra: Vec<TimsSpectrumAnnotated> = Vec::new();

        for (peptide_id, abundance) in peptide_ids.iter().zip(abundances.iter()) {
            // jump to next peptide if the peptide_id is not in the peptide_to_ions map
            if !self.peptide_to_ions.contains_key(&peptide_id) {
                continue;
            }

            let (ion_abundances, scan_occurrences, scan_abundances, charges, _) =
                self.peptide_to_ions.get(&peptide_id).unwrap();

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let scan_occurrence = scan_occurrences.get(index).unwrap();
                let scan_abundance = scan_abundances.get(index).unwrap();
                let charge = charges.get(index).unwrap();
                let peptide = self.peptides.get(peptide_id).unwrap();
                let ion = PeptideIon::new(
                    peptide.sequence.sequence.clone(),
                    *charge as i32,
                    *ion_abundance as f64,
                    Some(*peptide_id as i32),
                );
                // TODO: make this configurable
                let spectrum = ion.calculate_isotopic_spectrum_annotated(1e-3, 1e-8, 200, 1e-4);

                for (scan, scan_abu) in scan_occurrence.iter().zip(scan_abundance.iter()) {
                    let abundance_factor = abundance
                        * ion_abundance
                        * scan_abu
                        * self.peptide_to_events.get(&peptide_id).unwrap();
                    let scan_id = *scan;
                    let scaled_spec: MzSpectrumAnnotated =
                        spectrum.clone() * abundance_factor as f64;

                    let mz_spectrum = if mz_noise_precursor {
                        match uniform {
                            true => {
                                scaled_spec.add_mz_noise_uniform(precursor_noise_ppm, right_drag)
                            }
                            false => scaled_spec.add_mz_noise_normal(precursor_noise_ppm),
                        }
                    } else {
                        scaled_spec
                    };

                    let tims_spec = TimsSpectrumAnnotated::new(
                        frame_id as i32,
                        *scan,
                        *self.frame_to_rt.get(&frame_id).unwrap() as f64,
                        *self.scan_to_mobility.get(&scan_id).unwrap() as f64,
                        ms_type.clone(),
                        vec![0; mz_spectrum.mz.len()],
                        mz_spectrum,
                    );
                    tims_spectra.push(tims_spec);
                }
            }
        }

        let tims_frame = TimsFrameAnnotated::from_tims_spectra_annotated(tims_spectra);

        let filtered_frame = tims_frame.filter_ranged(0.0, 2000.0, 0.0, 2.0, 0, 1000, 1.0, 1e9);

        TimsFrameAnnotated {
            frame_id: filtered_frame.frame_id,
            retention_time: filtered_frame.retention_time,
            ms_type: filtered_frame.ms_type,
            tof: filtered_frame.tof,
            mz: filtered_frame.mz,
            scan: filtered_frame.scan,
            inv_mobility: filtered_frame.inv_mobility,
            intensity: filtered_frame.intensity,
            annotations: filtered_frame
                .annotations
                .iter()
                .map(|x| {
                    let mut contributions = x.contributions.clone();
                    contributions.sort_by(|a, b| {
                        a.intensity_contribution
                            .partial_cmp(&b.intensity_contribution)
                            .unwrap()
                    });
                    PeakAnnotation {
                        contributions,
                        ..*x
                    }
                })
                .collect::<Vec<PeakAnnotation>>(),
        }
    }

    pub fn build_precursor_frames_annotated(
        &self,
        frame_ids: Vec<u32>,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        right_drag: bool,
        num_threads: usize,
    ) -> Vec<TimsFrameAnnotated> {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        let mut tims_frames: Vec<TimsFrameAnnotated> = Vec::new();

        thread_pool.install(|| {
            tims_frames = frame_ids
                .par_iter()
                .map(|frame_id| {
                    self.build_precursor_frame_annotated(
                        *frame_id,
                        mz_noise_precursor,
                        uniform,
                        precursor_noise_ppm,
                        right_drag,
                    )
                })
                .collect();
        });

        tims_frames.sort_by(|a, b| a.frame_id.cmp(&b.frame_id));

        tims_frames
    }
}
