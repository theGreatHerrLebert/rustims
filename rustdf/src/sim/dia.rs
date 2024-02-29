use std::collections::BTreeMap;
use std::path::Path;
use mscore::timstof::collision::{TimsTofCollisionEnergy, TimsTofCollisionEnergyDIA};
use mscore::timstof::quadrupole::{IonTransmission, TimsTransmissionDIA};
use mscore::data::spectrum::{IndexedMzSpectrum, MsType};
use mscore::timstof::frame::TimsFrame;
use mscore::timstof::spectrum::TimsSpectrum;

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::sim::containers::{FragmentIonSeries};
use crate::sim::handle::TimsTofSyntheticsDataHandle;
use crate::sim::precursor::{TimsTofSyntheticsPrecursorFrameBuilder};

pub struct TimsTofSyntheticsFrameBuilderDIA {
    pub precursor_frame_builder: TimsTofSyntheticsPrecursorFrameBuilder,
    pub transmission_settings: TimsTransmissionDIA,
    pub fragmentation_settings: TimsTofCollisionEnergyDIA,
    pub fragment_ions: BTreeMap<(u32, i8, i8), Vec<FragmentIonSeries>>,
}

impl TimsTofSyntheticsFrameBuilderDIA {
    pub fn new(path: &Path) -> rusqlite::Result<Self> {

        let synthetics = TimsTofSyntheticsPrecursorFrameBuilder::new(path)?;
        let handle = TimsTofSyntheticsDataHandle::new(path)?;

        let fragment_ions = handle.read_fragment_ions()?;
        let fragment_ions = TimsTofSyntheticsDataHandle::build_fragment_ions(&fragment_ions);

        // get collision energy settings per window group
        let fragmentation_settings = handle.get_collision_energy_dia();
        // get ion transmission settings per window group
        let transmission_settings = handle.get_transmission_dia();

        Ok(Self {
            precursor_frame_builder: synthetics,
            transmission_settings,
            fragmentation_settings,
            fragment_ions,
        })
    }

    /// Build a frame for DIA synthetic experiment
    ///
    /// # Arguments
    ///
    /// * `frame_id` - The frame id
    /// * `fragmentation` - A boolean indicating if fragmentation is enabled, if false, the frame has same mz distribution as the precursor frame but will be quadrupole filtered
    ///
    /// # Returns
    ///
    /// A TimsFrame
    ///
    pub fn build_frame(&self, frame_id: u32, fragmentation: bool) -> TimsFrame {
        // determine if the frame is a precursor frame
        match self.precursor_frame_builder.precursor_frame_id_set.contains(&frame_id) {
            true => self.build_ms1_frame(frame_id),
            false => self.build_ms2_frame(frame_id, fragmentation),
        }
    }

    pub fn build_frames(&self, frame_ids: Vec<u32>, fragmentation: bool, num_threads: usize) -> Vec<TimsFrame> {
        let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
        let mut tims_frames: Vec<TimsFrame> = Vec::new();

        thread_pool.install(|| {
            tims_frames = frame_ids.par_iter().map(|frame_id| self.build_frame(*frame_id, fragmentation)).collect();
        });

        tims_frames.sort_by(|a, b| a.frame_id.cmp(&b.frame_id));

        tims_frames
    }

    fn build_ms1_frame(&self, frame_id: u32) -> TimsFrame {
        let tims_frame = self.precursor_frame_builder.build_precursor_frame(frame_id);
        tims_frame
    }
    fn build_ms2_frame(&self, frame_id: u32, fragmentation: bool) -> TimsFrame {
        match fragmentation {
            false => {
                let mut frame = self.transmission_settings.transmit_tims_frame(&self.build_ms1_frame(frame_id), None);
                frame.ms_type = MsType::FragmentDia;
                frame
            },
            true => self.build_fragment_frame(frame_id, None, None, None),
        }
    }

    /// Build a fragment frame
    ///
    /// # Arguments
    ///
    /// * `frame_id` - The frame id
    /// * `mz_min` - The minimum m/z value in fragment spectrum
    /// * `mz_max` - The maximum m/z value in fragment spectrum
    /// * `intensity_min` - The minimum intensity value in fragment spectrum
    ///
    /// # Returns
    ///
    /// A TimsFrame
    ///
    fn build_fragment_frame(
        &self,
        frame_id: u32,
        mz_min: Option<f64>,
        mz_max: Option<f64>,
        intensity_min: Option<f64>
    ) -> TimsFrame {

        // check frame id
        let ms_type = match self.precursor_frame_builder.precursor_frame_id_set.contains(&frame_id) {
            false => MsType::FragmentDia,
            true => MsType::Unknown,
        };

        let mut tims_spectra: Vec<TimsSpectrum> = Vec::new();

        // Frame might not have any peptides
        if !self.precursor_frame_builder.frame_to_abundances.contains_key(&frame_id) {
            return TimsFrame::new(
                frame_id as i32,
                ms_type.clone(),
                *self.precursor_frame_builder.frame_to_rt.get(&frame_id).unwrap() as f64,
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            );
        }

        // Get the peptide ids and abundances for the frame, should now save to unwrap since we checked if the frame is in the map
        let (peptide_ids, frame_abundances) = self.precursor_frame_builder.frame_to_abundances.get(&frame_id).unwrap();

        // Go over all peptides in the frame with their respective abundances
        for (peptide_id, frame_abundance) in peptide_ids.iter().zip(frame_abundances.iter()) {

            // jump to next peptide if the peptide_id is not in the peptide_to_ions map
            if !self.precursor_frame_builder.peptide_to_ions.contains_key(&peptide_id) {
                continue;
            }

            // get all the ions for the peptide
            let (ion_abundances, scan_occurrences, scan_abundances, charges, spectra) = self.precursor_frame_builder.peptide_to_ions.get(&peptide_id).unwrap();

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                // occurrence and abundance of the ion in the scan
                let all_scan_occurrence = scan_occurrences.get(index).unwrap();
                let all_scan_abundance = scan_abundances.get(index).unwrap();

                // get precursor spectrum for the ion
                let spectrum = spectra.get(index).unwrap();

                // go over occurrence and abundance of the ion in the scan
                for (scan, scan_abundance) in all_scan_occurrence.iter().zip(all_scan_abundance.iter()) {

                    // first, check if precursor is transmitted
                    if !self.transmission_settings.any_transmitted(frame_id as i32, *scan as i32, &spectrum.mz, None) {
                        continue;
                    }

                    // calculate abundance factor
                    let total_events = self.precursor_frame_builder.peptide_to_events.get(&peptide_id).unwrap();
                    let fraction_events = frame_abundance * scan_abundance * ion_abundance * total_events;

                    // get collision energy for the ion
                    let collision_energy = self.fragmentation_settings.get_collision_energy(frame_id as i32, *scan as i32);
                    let collision_energy_quantized = (collision_energy * 1e3).round() as i8;

                    // get charge state for the ion
                    let charge_state = charges.get(index).unwrap();
                    // extract fragment ions for the peptide, charge state and collision energy
                    let fragment_ions = self.fragment_ions.get(&(*peptide_id, *charge_state, collision_energy_quantized));

                    // jump to next peptide if the fragment_ions is None (can this happen?)
                    if fragment_ions.is_none() {
                        continue;
                    }

                    // for each fragment ion series, create a spectrum and add it to the tims_spectra
                    for fragment_ion_series in fragment_ions.unwrap() {
                        // scale the spectrum by the fraction of events
                        let scaled_spec = fragment_ion_series.to_mz_spectrum() * fraction_events as f64;

                        tims_spectra.push(
                            TimsSpectrum::new(
                                frame_id as i32,
                                *scan as i32,
                                *self.precursor_frame_builder.frame_to_rt.get(&frame_id).unwrap() as f64,
                                *self.precursor_frame_builder.scan_to_mobility.get(&scan).unwrap() as f64,
                                ms_type.clone(),
                                IndexedMzSpectrum::new(vec![0; scaled_spec.mz.len()], scaled_spec.mz, scaled_spec.intensity),
                            )
                        );
                    }
                }
            }
        }

        if tims_spectra.is_empty() {
            return TimsFrame::new(
                frame_id as i32,
                ms_type.clone(),
                *self.precursor_frame_builder.frame_to_rt.get(&frame_id).unwrap() as f64,
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            );
        }

        let tims_frame = TimsFrame::from_tims_spectra(tims_spectra);
        tims_frame.filter_ranged(
            mz_min.unwrap_or(100.0),
            mz_max.unwrap_or(1700.0),
            0,
            1000,
            0.0,
            10.0,
            intensity_min.unwrap_or(1.0),
            1e9,
        )
    }
}

impl TimsTofCollisionEnergy for TimsTofSyntheticsFrameBuilderDIA {
    fn get_collision_energy(&self, frame_id: i32, scan_id: i32) -> f64 {
        self.fragmentation_settings.get_collision_energy(frame_id, scan_id)
    }
}