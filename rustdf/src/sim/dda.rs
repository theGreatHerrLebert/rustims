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
        Option<BTreeMap<(u32, i8, i32), (PeptideProductIonSeriesCollection, Vec<MzSpectrum>)>>,
    pub fragment_ions_annotated: Option<
        BTreeMap<(u32, i8, i32), (PeptideProductIonSeriesCollection, Vec<MzSpectrumAnnotated>)>,
    >,
}

impl TimsTofSyntheticsFrameBuilderDDA {
    pub fn new(path: &Path, with_annotations: bool, num_threads: usize) -> Self {

        let handle = TimsTofSyntheticsDataHandle::new(path).unwrap();
        let fragment_ions = handle.read_fragment_ions().unwrap();
        let transmission_settings = handle.get_transmission_dda();

        let synthetics = TimsTofSyntheticsPrecursorFrameBuilder::new(path).unwrap();

        match with_annotations {
            true => {
                let fragment_ions =
                    Some(TimsTofSyntheticsDataHandle::build_fragment_ions_annotated(
                        &synthetics.peptides,
                        &fragment_ions,
                        num_threads,
                    ));
                Self {
                    path: path.to_str().unwrap().to_string(),
                    precursor_frame_builder: synthetics,
                    transmission_settings,
                    fragment_ions: None,
                    fragment_ions_annotated: fragment_ions,
                }
            }
            false => {
                let fragment_ions = Some(TimsTofSyntheticsDataHandle::build_fragment_ions(
                    &synthetics.peptides,
                    &fragment_ions,
                    num_threads,
                ));
                Self {
                    path: path.to_str().unwrap().to_string(),
                    precursor_frame_builder: synthetics,
                    transmission_settings,
                    fragment_ions,
                    fragment_ions_annotated: None,
                }
            }
        }
    }
    /// Build a frame for DDA synthetic experiment
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
    pub fn build_frame(
        &self,
        frame_id: u32,
        fragmentation: bool,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        mz_noise_fragment: bool,
        fragment_noise_ppm: f64,
        right_drag: bool,
    ) -> TimsFrame {
        // determine if the frame is a precursor frame
        match self
            .precursor_frame_builder
            .precursor_frame_id_set
            .contains(&frame_id)
        {
            true => self.build_ms1_frame(
                frame_id,
                mz_noise_precursor,
                uniform,
                precursor_noise_ppm,
                right_drag,
            ),
            false => self.build_ms2_frame(
                frame_id,
                fragmentation,
                mz_noise_fragment,
                uniform,
                fragment_noise_ppm,
                right_drag,
            ),
        }
    }

    pub fn build_frame_annotated(
        &self,
        frame_id: u32,
        fragmentation: bool,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        mz_noise_fragment: bool,
        fragment_noise_ppm: f64,
        right_drag: bool,
    ) -> TimsFrameAnnotated {
        match self
            .precursor_frame_builder
            .precursor_frame_id_set
            .contains(&frame_id)
        {
            true => self.build_ms1_frame_annotated(
                frame_id,
                mz_noise_precursor,
                uniform,
                precursor_noise_ppm,
                right_drag,
            ),
            false => self.build_ms2_frame_annotated(
                frame_id,
                fragmentation,
                mz_noise_fragment,
                uniform,
                fragment_noise_ppm,
                right_drag,
            ),
        }
    }

    pub fn get_fragment_ion_ids(&self, precursor_frame_ids: Vec<u32>) -> Vec<u32> {
        let mut peptide_ids: HashSet<u32> = HashSet::new();
        // get all peptide ids for the precursor frame ids
        for frame_id in precursor_frame_ids {
            for (peptide_id, peptide) in self.precursor_frame_builder.peptides.iter() {
                if peptide.frame_start <= frame_id && peptide.frame_end >= frame_id {
                    peptide_ids.insert(*peptide_id);
                }
            }
        }
        // get all ion ids for the peptide ids
        let mut result: Vec<u32> = Vec::new();
        for item in peptide_ids {
            let ions = self.precursor_frame_builder.ions.get(&item).unwrap();
            for ion in ions.iter() {
                result.push(ion.ion_id);
            }
        }
        result
    }

    pub fn build_frames(
        &self,
        frame_ids: Vec<u32>,
        fragmentation: bool,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        mz_noise_fragment: bool,
        fragment_noise_ppm: f64,
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
                    self.build_frame(
                        *frame_id,
                        fragmentation,
                        mz_noise_precursor,
                        uniform,
                        precursor_noise_ppm,
                        mz_noise_fragment,
                        fragment_noise_ppm,
                        right_drag,
                    )
                })
                .collect();
        });

        tims_frames.sort_by(|a, b| a.frame_id.cmp(&b.frame_id));

        tims_frames
    }
    pub fn build_frames_annotated(
        &self,
        frame_ids: Vec<u32>,
        fragmentation: bool,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        mz_noise_fragment: bool,
        fragment_noise_ppm: f64,
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
                    self.build_frame_annotated(
                        *frame_id,
                        fragmentation,
                        mz_noise_precursor,
                        uniform,
                        precursor_noise_ppm,
                        mz_noise_fragment,
                        fragment_noise_ppm,
                        right_drag,
                    )
                })
                .collect();
        });

        tims_frames.sort_by(|a, b| a.frame_id.cmp(&b.frame_id));

        tims_frames
    }

    fn build_ms1_frame(
        &self,
        frame_id: u32,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_ppm: f64,
        right_drag: bool,
    ) -> TimsFrame {
        let mut tims_frame = self.precursor_frame_builder.build_precursor_frame(
            frame_id,
            mz_noise_precursor,
            uniform,
            precursor_ppm,
            right_drag,
        );
        let intensities_rounded = tims_frame
            .ims_frame
            .intensity
            .iter()
            .map(|x| x.round())
            .collect::<Vec<_>>();
        tims_frame.ims_frame.intensity = intensities_rounded;
        tims_frame
    }

    fn build_ms1_frame_annotated(
        &self,
        frame_id: u32,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_ppm: f64,
        right_drag: bool,
    ) -> TimsFrameAnnotated {
        let mut tims_frame = self
            .precursor_frame_builder
            .build_precursor_frame_annotated(
                frame_id,
                mz_noise_precursor,
                uniform,
                precursor_ppm,
                right_drag,
            );
        let intensities_rounded = tims_frame
            .intensity
            .iter()
            .map(|x| x.round())
            .collect::<Vec<_>>();
        tims_frame.intensity = intensities_rounded;
        tims_frame
    }

    fn build_ms2_frame(
        &self,
        frame_id: u32,
        fragmentation: bool,
        mz_noise_fragment: bool,
        uniform: bool,
        fragment_ppm: f64,
        right_drag: bool,
    ) -> TimsFrame {
        match fragmentation {
            false => {
                let mut frame = self.transmission_settings.transmit_tims_frame(
                    &self.build_ms1_frame(
                        frame_id,
                        mz_noise_fragment,
                        uniform,
                        fragment_ppm,
                        right_drag,
                    ),
                    None,
                );
                let intensities_rounded = frame
                    .ims_frame
                    .intensity
                    .iter()
                    .map(|x| x.round())
                    .collect::<Vec<_>>();
                frame.ims_frame.intensity = intensities_rounded;
                frame.ms_type = MsType::FragmentDia;
                frame
            }
            true => {
                let mut frame = self.build_fragment_frame(
                    frame_id,
                    &self.fragment_ions.as_ref().unwrap(),
                    mz_noise_fragment,
                    uniform,
                    fragment_ppm,
                    None,
                    None,
                    None,
                    Some(right_drag),
                );
                let intensities_rounded = frame
                    .ims_frame
                    .intensity
                    .iter()
                    .map(|x| x.round())
                    .collect::<Vec<_>>();
                frame.ims_frame.intensity = intensities_rounded;
                frame
            }
        }
    }

    fn build_ms2_frame_annotated(
        &self,
        frame_id: u32,
        fragmentation: bool,
        mz_noise_fragment: bool,
        uniform: bool,
        fragment_ppm: f64,
        right_drag: bool,
    ) -> TimsFrameAnnotated {
        match fragmentation {
            false => {
                let mut frame = self.transmission_settings.transmit_tims_frame_annotated(
                    &self.build_ms1_frame_annotated(
                        frame_id,
                        mz_noise_fragment,
                        uniform,
                        fragment_ppm,
                        right_drag,
                    ),
                    None,
                );
                let intensities_rounded = frame
                    .intensity
                    .iter()
                    .map(|x| x.round())
                    .collect::<Vec<_>>();
                frame.intensity = intensities_rounded;
                frame.ms_type = MsType::FragmentDia;
                frame
            }
            true => {
                let mut frame = self.build_fragment_frame_annotated(
                    frame_id,
                    &self.fragment_ions_annotated.as_ref().unwrap(),
                    mz_noise_fragment,
                    uniform,
                    fragment_ppm,
                    None,
                    None,
                    None,
                    Some(right_drag),
                );
                let intensities_rounded = frame
                    .intensity
                    .iter()
                    .map(|x| x.round())
                    .collect::<Vec<_>>();
                frame.intensity = intensities_rounded;
                frame
            }
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
        fragment_ions: &BTreeMap<
            (u32, i8, i32),
            (PeptideProductIonSeriesCollection, Vec<MzSpectrum>),
        >,
        mz_noise_fragment: bool,
        uniform: bool,
        fragment_ppm: f64,
        mz_min: Option<f64>,
        mz_max: Option<f64>,
        intensity_min: Option<f64>,
        right_drag: Option<bool>,
    ) -> TimsFrame {
        // check frame id
        let ms_type = match self
            .precursor_frame_builder
            .precursor_frame_id_set
            .contains(&frame_id)
        {
            false => MsType::FragmentDda,
            true => MsType::Unknown,
        };

        // pull out RT once so we don’t unwrap twice
        let rt = *self
            .precursor_frame_builder
            .frame_to_rt
            .get(&frame_id)
            .expect("frame_to_rt should always have this frame") as f64;

        // if no PASEF meta *or* no abundances, return an empty frame
        if self
            .transmission_settings
            .pasef_meta
            .get(&(frame_id as i32))
            .is_none()
            || !self
            .precursor_frame_builder
            .frame_to_abundances
            .contains_key(&frame_id)
        {
            return TimsFrame::new(
                frame_id as i32,
                ms_type.clone(),
                rt,
                vec![], vec![], vec![], vec![], vec![],
            );
        }

        let mut tims_spectra: Vec<TimsSpectrum> = Vec::new();

        // Get the peptide ids and abundances for the frame, should now save to unwrap since we checked if the frame is in the map
        let (peptide_ids, frame_abundances) = self
            .precursor_frame_builder
            .frame_to_abundances
            .get(&frame_id)
            .unwrap();

        // Go over all peptides in the frame with their respective abundances
        for (peptide_id, frame_abundance) in peptide_ids.iter().zip(frame_abundances.iter()) {
            // jump to next peptide if the peptide_id is not in the peptide_to_ions map
            if !self
                .precursor_frame_builder
                .peptide_to_ions
                .contains_key(&peptide_id)
            {
                continue;
            }

            // get all the ions for the peptide
            let (ion_abundances, scan_occurrences, scan_abundances, charges, spectra) = self
                .precursor_frame_builder
                .peptide_to_ions
                .get(&peptide_id)
                .unwrap();

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                // occurrence and abundance of the ion in the scan
                let all_scan_occurrence = scan_occurrences.get(index).unwrap();
                let all_scan_abundance = scan_abundances.get(index).unwrap();

                // get precursor spectrum for the ion
                let spectrum = spectra.get(index).unwrap();

                // go over occurrence and abundance of the ion in the scan
                for (scan, scan_abundance) in
                    all_scan_occurrence.iter().zip(all_scan_abundance.iter())
                {
                    // first, check if precursor is transmitted
                    if !self.transmission_settings.any_transmitted(
                        frame_id as i32,
                        *scan as i32,
                        &spectrum.mz,
                        None,
                    ) {
                        continue;
                    }

                    // calculate abundance factor
                    let total_events = self
                        .precursor_frame_builder
                        .peptide_to_events
                        .get(&peptide_id)
                        .unwrap();
                    let fraction_events =
                        frame_abundance * scan_abundance * ion_abundance * total_events;

                    // get PASEF settings for the given frame
                    let maybe_pasef_meta = self.transmission_settings.pasef_meta.get(&(frame_id as i32));

                    let collision_energy: f64 = match maybe_pasef_meta {
                        Some(pasef_meta) => {
                            pasef_meta
                                .iter()
                                .find(|scan_meta| scan_meta.scan_start <= *scan as i32 && scan_meta.scan_end >= *scan as i32)
                                .map(|s| s.collision_energy)
                                .unwrap_or(0.0)
                        },
                        None => 0.0
                    };

                    let collision_energy_quantized = (collision_energy * 1e1).round() as i32;

                    // get charge state for the ion
                    let charge_state = charges.get(index).unwrap();
                    // extract fragment ions for the peptide, charge state and collision energy
                    let maybe_value = fragment_ions.get(&(
                        *peptide_id,
                        *charge_state,
                        collision_energy_quantized,
                    ));

                    // jump to next peptide if the fragment_ions is None (can this happen?)
                    if maybe_value.is_none() {
                        continue;
                    }

                    // for each fragment ion series, create a spectrum and add it to the tims_spectra
                    for fragment_ion_series in maybe_value.unwrap().1.iter() {
                        let scaled_spec = fragment_ion_series.clone() * fraction_events as f64;
                        let right_drag = right_drag.unwrap_or(false);

                        let mz_spectrum = if mz_noise_fragment {
                            match uniform {
                                true => scaled_spec.add_mz_noise_uniform(fragment_ppm, right_drag),
                                false => scaled_spec.add_mz_noise_normal(fragment_ppm),
                            }
                        } else {
                            scaled_spec
                        };

                        tims_spectra.push(TimsSpectrum::new(
                            frame_id as i32,
                            *scan as i32,
                            *self
                                .precursor_frame_builder
                                .frame_to_rt
                                .get(&frame_id)
                                .unwrap() as f64,
                            *self
                                .precursor_frame_builder
                                .scan_to_mobility
                                .get(&scan)
                                .unwrap() as f64,
                            ms_type.clone(),
                            IndexedMzSpectrum::new(
                                vec![0; mz_spectrum.mz.len()],
                                mz_spectrum.mz,
                                mz_spectrum.intensity,
                            )
                                .filter_ranged(100.0, 1700.0, 1.0, 1e9),
                        ));
                    }
                }
            }
        }

        if tims_spectra.is_empty() {
            return TimsFrame::new(
                frame_id as i32,
                ms_type.clone(),
                *self
                    .precursor_frame_builder
                    .frame_to_rt
                    .get(&frame_id)
                    .unwrap() as f64,
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

    pub fn build_fragment_frame_annotated(
        &self,
        frame_id: u32,
        fragment_ions: &BTreeMap<
            (u32, i8, i32),
            (PeptideProductIonSeriesCollection, Vec<MzSpectrumAnnotated>),
        >,
        mz_noise_fragment: bool,
        uniform: bool,
        fragment_ppm: f64,
        mz_min: Option<f64>,
        mz_max: Option<f64>,
        intensity_min: Option<f64>,
        right_drag: Option<bool>,
    ) -> TimsFrameAnnotated {
        let ms_type = match self
            .precursor_frame_builder
            .precursor_frame_id_set
            .contains(&frame_id)
        {
            false => MsType::FragmentDia,
            true => MsType::Unknown,
        };

        let mut tims_spectra: Vec<TimsSpectrumAnnotated> = Vec::new();

        if !self
            .precursor_frame_builder
            .frame_to_abundances
            .contains_key(&frame_id)
        {
            return TimsFrameAnnotated::new(
                frame_id as i32,
                *self
                    .precursor_frame_builder
                    .frame_to_rt
                    .get(&frame_id)
                    .unwrap() as f64,
                ms_type.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            );
        }

        let (peptide_ids, frame_abundances) = self
            .precursor_frame_builder
            .frame_to_abundances
            .get(&frame_id)
            .unwrap();

        for (peptide_id, frame_abundance) in peptide_ids.iter().zip(frame_abundances.iter()) {
            if !self
                .precursor_frame_builder
                .peptide_to_ions
                .contains_key(&peptide_id)
            {
                continue;
            }

            let (ion_abundances, scan_occurrences, scan_abundances, charges, _) = self
                .precursor_frame_builder
                .peptide_to_ions
                .get(&peptide_id)
                .unwrap();

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let all_scan_occurrence = scan_occurrences.get(index).unwrap();
                let all_scan_abundance = scan_abundances.get(index).unwrap();

                let peptide = self
                    .precursor_frame_builder
                    .peptides
                    .get(peptide_id)
                    .unwrap();
                let ion = PeptideIon::new(
                    peptide.sequence.sequence.clone(),
                    charges[index] as i32,
                    *ion_abundance as f64,
                    Some(*peptide_id as i32),
                );
                // TODO: make this configurable
                let spectrum = ion.calculate_isotopic_spectrum_annotated(1e-3, 1e-8, 200, 1e-4);

                for (scan, scan_abundance) in
                    all_scan_occurrence.iter().zip(all_scan_abundance.iter())
                {
                    if !self.transmission_settings.any_transmitted(
                        frame_id as i32,
                        *scan as i32,
                        &spectrum.mz,
                        None,
                    ) {
                        continue;
                    }

                    let total_events = self
                        .precursor_frame_builder
                        .peptide_to_events
                        .get(&peptide_id)
                        .unwrap();
                    let fraction_events =
                        frame_abundance * scan_abundance * ion_abundance * total_events;

                    // get PASEF settings for the given frame
                    let maybe_pasef_meta = self.transmission_settings.pasef_meta.get(&(frame_id as i32));

                    let collision_energy: f64 = match maybe_pasef_meta {
                        Some(pasef_meta) => {
                            pasef_meta
                                .iter()
                                .find(|scan_meta| scan_meta.scan_start <= *scan as i32 && scan_meta.scan_end >= *scan as i32)
                                .map(|s| s.collision_energy)
                                .unwrap_or(0.0)
                        },
                        None => 0.0
                    };

                    let collision_energy_quantized = (collision_energy * 1e1).round() as i32;

                    let charge_state = charges.get(index).unwrap();
                    let maybe_value = fragment_ions.get(&(
                        *peptide_id,
                        *charge_state,
                        collision_energy_quantized,
                    ));

                    if maybe_value.is_none() {
                        continue;
                    }

                    for fragment_ion_series in maybe_value.unwrap().1.iter() {
                        let scaled_spec = fragment_ion_series.clone() * fraction_events as f64;
                        let right_drag = right_drag.unwrap_or(false);

                        let mz_spectrum = if mz_noise_fragment {
                            match uniform {
                                true => scaled_spec.add_mz_noise_uniform(fragment_ppm, right_drag),
                                false => scaled_spec.add_mz_noise_normal(fragment_ppm),
                            }
                        } else {
                            scaled_spec
                        };

                        tims_spectra.push(TimsSpectrumAnnotated::new(
                            frame_id as i32,
                            *scan,
                            *self
                                .precursor_frame_builder
                                .frame_to_rt
                                .get(&frame_id)
                                .unwrap() as f64,
                            *self
                                .precursor_frame_builder
                                .scan_to_mobility
                                .get(&scan)
                                .unwrap() as f64,
                            ms_type.clone(),
                            vec![0; mz_spectrum.mz.len()],
                            mz_spectrum,
                        ));
                    }
                }
            }
        }

        if tims_spectra.is_empty() {
            return TimsFrameAnnotated::new(
                frame_id as i32,
                *self
                    .precursor_frame_builder
                    .frame_to_rt
                    .get(&frame_id)
                    .unwrap() as f64,
                ms_type.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            );
        }

        let tims_frame = TimsFrameAnnotated::from_tims_spectra_annotated(tims_spectra);

        tims_frame.filter_ranged(
            mz_min.unwrap_or(100.0),
            mz_max.unwrap_or(1700.0),
            0.0,
            10.0,
            0,
            1000,
            intensity_min.unwrap_or(1.0),
            1e9,
        )
    }

    pub fn get_collision_energy(&self, frame_id: i32, scan_id: i32) -> f64 {
        self.transmission_settings.get_collision_energy(frame_id, scan_id).unwrap_or(0.0)
    }

    pub fn get_collision_energies(&self, frame_ids: Vec<i32>, scan_ids: Vec<i32>) -> Vec<f64> {
        let mut collision_energies: Vec<f64> = Vec::new();
        for frame_id in frame_ids {
            for scan_id in &scan_ids {
                collision_energies.push(self.get_collision_energy(frame_id, *scan_id));
            }
        }
        collision_energies
    }
}