use mscore::algorithm::isotope::{
    calculate_transmission_dependent_fragment_ion_isotope_distribution,
    calculate_precursor_transmission_factor,
};
use crate::sim::containers::IsotopeTransmissionMode;
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
use std::sync::Arc;

use rayon::prelude::*;
use crate::sim::containers::IsotopeTransmissionConfig;
use crate::sim::handle::{FragmentIonsWithComplementary, TimsTofSyntheticsDataHandle};
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
    /// Configuration for quad-selection dependent isotope transmission
    pub isotope_transmission_config: IsotopeTransmissionConfig,
    /// Fragment ions with complementary distribution data (only populated when isotope_transmission_config.enabled)
    pub fragment_ions_with_complementary: Option<BTreeMap<(u32, i8, i32), FragmentIonsWithComplementary>>,
}

impl TimsTofSyntheticsFrameBuilderDDA {
    /// Create a new DDA frame builder.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the simulation database
    /// * `with_annotations` - Whether to include annotations
    /// * `num_threads` - Number of threads for parallel processing
    /// * `isotope_config` - Optional configuration for quad-dependent isotope transmission
    pub fn new(
        path: &Path,
        with_annotations: bool,
        num_threads: usize,
        isotope_config: Option<IsotopeTransmissionConfig>,
    ) -> Self {
        let handle = TimsTofSyntheticsDataHandle::new(path).unwrap();
        let fragment_ions_raw = handle.read_fragment_ions().unwrap();
        let transmission_settings = handle.get_transmission_dda();

        let synthetics = TimsTofSyntheticsPrecursorFrameBuilder::new(path).unwrap();
        let config = isotope_config.unwrap_or_default();

        // Build fragment ions with complementary data if any transmission mode is enabled
        let fragment_ions_with_complementary = if config.is_enabled() {
            Some(TimsTofSyntheticsDataHandle::build_fragment_ions_with_transmission_data(
                &synthetics.peptides,
                &fragment_ions_raw,
                num_threads,
            ))
        } else {
            None
        };

        match with_annotations {
            true => {
                let fragment_ions =
                    Some(TimsTofSyntheticsDataHandle::build_fragment_ions_annotated(
                        &synthetics.peptides,
                        &fragment_ions_raw,
                        num_threads,
                    ));
                Self {
                    path: path.to_str().unwrap().to_string(),
                    precursor_frame_builder: synthetics,
                    transmission_settings,
                    fragment_ions: None,
                    fragment_ions_annotated: fragment_ions,
                    isotope_transmission_config: config,
                    fragment_ions_with_complementary,
                }
            }
            false => {
                let fragment_ions = Some(TimsTofSyntheticsDataHandle::build_fragment_ions(
                    &synthetics.peptides,
                    &fragment_ions_raw,
                    num_threads,
                ));
                Self {
                    path: path.to_str().unwrap().to_string(),
                    precursor_frame_builder: synthetics,
                    transmission_settings,
                    fragment_ions,
                    fragment_ions_annotated: None,
                    isotope_transmission_config: config,
                    fragment_ions_with_complementary,
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
        // Use thread pool with custom parallelism
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            // Use indexed parallel iteration to maintain order, avoiding post-sort
            let mut tims_frames: Vec<TimsFrame> = Vec::with_capacity(frame_ids.len());
            unsafe { tims_frames.set_len(frame_ids.len()); }

            frame_ids.par_iter().enumerate().for_each(|(idx, frame_id)| {
                let frame = self.build_frame(
                    *frame_id,
                    fragmentation,
                    mz_noise_precursor,
                    uniform,
                    precursor_noise_ppm,
                    mz_noise_fragment,
                    fragment_noise_ppm,
                    right_drag,
                );
                unsafe {
                    let ptr = tims_frames.as_ptr() as *mut TimsFrame;
                    std::ptr::write(ptr.add(idx), frame);
                }
            });

            tims_frames
        })
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
        // Use thread pool with custom parallelism
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            // Use indexed parallel iteration to maintain order, avoiding post-sort
            let mut tims_frames: Vec<TimsFrameAnnotated> = Vec::with_capacity(frame_ids.len());
            unsafe { tims_frames.set_len(frame_ids.len()); }

            frame_ids.par_iter().enumerate().for_each(|(idx, frame_id)| {
                let frame = self.build_frame_annotated(
                    *frame_id,
                    fragmentation,
                    mz_noise_precursor,
                    uniform,
                    precursor_noise_ppm,
                    mz_noise_fragment,
                    fragment_noise_ppm,
                    right_drag,
                );
                unsafe {
                    let ptr = tims_frames.as_ptr() as *mut TimsFrameAnnotated;
                    std::ptr::write(ptr.add(idx), frame);
                }
            });

            tims_frames
        })
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
        tims_frame.ims_frame.intensity = Arc::new(intensities_rounded);
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
                frame.ims_frame.intensity = Arc::new(intensities_rounded);
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
                frame.ims_frame.intensity = Arc::new(intensities_rounded);
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
        // Cache frame-level lookups
        let ms_type = if self.precursor_frame_builder.precursor_frame_id_set.contains(&frame_id) {
            MsType::Unknown
        } else {
            MsType::FragmentDda
        };

        let rt = *self.precursor_frame_builder.frame_to_rt.get(&frame_id)
            .expect("frame_to_rt should always have this frame") as f64;
        let right_drag_val = right_drag.unwrap_or(false);
        let mz_min_val = mz_min.unwrap_or(100.0);
        let mz_max_val = mz_max.unwrap_or(1700.0);
        let intensity_min_val = intensity_min.unwrap_or(1.0);

        // Get PASEF meta for this frame - these define which precursors were SELECTED
        let Some(pasef_meta) = self.transmission_settings.pasef_meta.get(&(frame_id as i32)) else {
            return TimsFrame::new(frame_id as i32, ms_type, rt, vec![], vec![], vec![], vec![], vec![]);
        };

        // Preallocate with estimated capacity
        let estimated_capacity = pasef_meta.len() * 4;
        let mut tims_spectra: Vec<TimsSpectrum> = Vec::with_capacity(estimated_capacity);

        // Iterate over PASEF meta entries - each represents a SELECTED precursor
        for meta in pasef_meta.iter() {
            // Get the selected ion_id from the PASEF precursor field
            let ion_id = meta.precursor as u32;

            // Look up peptide_id and charge from the ion_id
            let Some(&(peptide_id, charge_state)) = self.precursor_frame_builder.ion_id_to_peptide_charge.get(&ion_id) else {
                continue;
            };

            // Get peptide info
            let Some(peptide) = self.precursor_frame_builder.peptides.get(&peptide_id) else {
                continue;
            };

            // Check if this peptide is active in this frame
            if frame_id < peptide.frame_start || frame_id > peptide.frame_end {
                continue;
            }

            // Get ion info from peptide_to_ions
            let Some((ion_abundances, scan_occurrences, scan_abundances, charges, spectra)) = self
                .precursor_frame_builder
                .peptide_to_ions
                .get(&peptide_id)
            else {
                continue;
            };

            // Find the index of this specific charge state
            let Some(ion_index) = charges.iter().position(|&c| c == charge_state) else {
                continue;
            };

            let ion_abundance = ion_abundances[ion_index];
            let all_scan_occurrence = &scan_occurrences[ion_index];
            let all_scan_abundance = &scan_abundances[ion_index];
            let spectrum = &spectra[ion_index];
            let total_events = *self.precursor_frame_builder.peptide_to_events.get(&peptide_id).unwrap();

            // Get frame abundance for this peptide
            let frame_abundance = self.precursor_frame_builder.frame_to_abundances
                .get(&frame_id)
                .and_then(|(pep_ids, abundances)| {
                    pep_ids.iter().position(|&p| p == peptide_id)
                        .map(|idx| abundances[idx])
                })
                .unwrap_or(0.0);

            if frame_abundance == 0.0 {
                continue;
            }

            // Collision energy from the PASEF meta
            let collision_energy = meta.collision_energy;
            let collision_energy_quantized = (collision_energy * 1e1).round() as i32;

            // Look up fragment ions for this (peptide_id, charge, CE)
            let Some((_, fragment_series_vec)) = fragment_ions.get(&(peptide_id, charge_state, collision_energy_quantized)) else {
                continue;
            };

            // Process scans within the PASEF selection window
            for (scan, scan_abundance) in all_scan_occurrence.iter().zip(all_scan_abundance.iter()) {
                // Check if scan is within the PASEF selection window
                let scan_i32 = *scan as i32;
                if scan_i32 < meta.scan_start || scan_i32 > meta.scan_end {
                    continue;
                }

                // Get transmitted isotope indices based on config mode
                let transmitted_indices = match self.isotope_transmission_config.mode {
                    IsotopeTransmissionMode::None => HashSet::new(),
                    IsotopeTransmissionMode::PrecursorScaling | IsotopeTransmissionMode::PerFragment => {
                        self.transmission_settings.get_transmission_set(
                            frame_id as i32,
                            scan_i32,
                            &spectrum.mz,
                            Some(self.isotope_transmission_config.min_probability),
                        )
                    },
                };

                // Calculate abundance factor
                let fraction_events = frame_abundance * scan_abundance * ion_abundance * total_events;

                // Cache scan mobility
                let scan_mobility = *self.precursor_frame_builder.scan_to_mobility.get(scan).unwrap() as f64;

                // Calculate transmission factor for PrecursorScaling mode
                let transmission_factor = if self.isotope_transmission_config.mode == IsotopeTransmissionMode::PrecursorScaling {
                    if let Some(comp_data) = self.fragment_ions_with_complementary.as_ref() {
                        if let Some(frag_data) = comp_data.get(&(peptide_id, charge_state, collision_energy_quantized)) {
                            calculate_precursor_transmission_factor(
                                &frag_data.precursor_isotope_distribution,
                                &transmitted_indices,
                            )
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };

                for (series_idx, fragment_ion_series) in fragment_series_vec.iter().enumerate() {
                    let final_spectrum = match self.isotope_transmission_config.mode {
                        IsotopeTransmissionMode::None => {
                            // Standard spectrum scaling
                            fragment_ion_series.clone() * fraction_events as f64
                        },
                        IsotopeTransmissionMode::PrecursorScaling => {
                            // Apply precursor-based transmission factor
                            fragment_ion_series.clone() * (fraction_events as f64 * transmission_factor)
                        },
                        IsotopeTransmissionMode::PerFragment => {
                            // Per-fragment transmission-dependent calculation
                            if let Some(comp_data) = self.fragment_ions_with_complementary.as_ref() {
                                if let Some(frag_data) = comp_data.get(&(peptide_id, charge_state, collision_energy_quantized)) {
                                    if series_idx < frag_data.per_fragment_data.len() {
                                        // Aggregate adjusted spectra from all fragment ions in this series
                                        let series_data = &frag_data.per_fragment_data[series_idx];
                                        let mut aggregated_mz: Vec<f64> = Vec::new();
                                        let mut aggregated_intensity: Vec<f64> = Vec::new();

                                        for frag_ion_data in series_data {
                                            // Apply transmission-dependent calculation for this fragment
                                            let adjusted_dist = calculate_transmission_dependent_fragment_ion_isotope_distribution(
                                                &frag_ion_data.fragment_distribution,
                                                &frag_ion_data.complementary_distribution,
                                                &transmitted_indices,
                                                self.isotope_transmission_config.max_isotopes,
                                            );

                                            // Scale by predicted intensity and fraction_events
                                            for (mz, abundance) in adjusted_dist {
                                                aggregated_mz.push(mz);
                                                aggregated_intensity.push(abundance * frag_ion_data.predicted_intensity * fraction_events as f64);
                                            }
                                        }

                                        if !aggregated_mz.is_empty() {
                                            MzSpectrum::new(aggregated_mz, aggregated_intensity)
                                        } else {
                                            fragment_ion_series.clone() * fraction_events as f64
                                        }
                                    } else {
                                        fragment_ion_series.clone() * fraction_events as f64
                                    }
                                } else {
                                    fragment_ion_series.clone() * fraction_events as f64
                                }
                            } else {
                                fragment_ion_series.clone() * fraction_events as f64
                            }
                        },
                    };

                    let mz_spectrum = if mz_noise_fragment {
                        if uniform {
                            final_spectrum.add_mz_noise_uniform(fragment_ppm, right_drag_val)
                        } else {
                            final_spectrum.add_mz_noise_normal(fragment_ppm)
                        }
                    } else {
                        final_spectrum
                    };

                    let spectrum_len = mz_spectrum.mz.len();
                    tims_spectra.push(TimsSpectrum::new(
                        frame_id as i32,
                        *scan as i32,
                        rt,
                        scan_mobility,
                        ms_type.clone(),
                        IndexedMzSpectrum::from_mz_spectrum(
                            vec![0; spectrum_len],
                            mz_spectrum,
                        ).filter_ranged(100.0, 1700.0, 1.0, 1e9),
                    ));
                }
            }
        }

        if tims_spectra.is_empty() {
            return TimsFrame::new(frame_id as i32, ms_type, rt, vec![], vec![], vec![], vec![], vec![]);
        }

        TimsFrame::from_tims_spectra_filtered(
            tims_spectra, mz_min_val, mz_max_val, 0, 1000, 0.0, 10.0, intensity_min_val, 1e9,
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
        // Cache frame-level lookups
        let ms_type = if self.precursor_frame_builder.precursor_frame_id_set.contains(&frame_id) {
            MsType::Unknown
        } else {
            MsType::FragmentDda
        };

        let rt = *self.precursor_frame_builder.frame_to_rt.get(&frame_id).unwrap() as f64;
        let right_drag_val = right_drag.unwrap_or(false);
        let mz_min_val = mz_min.unwrap_or(100.0);
        let mz_max_val = mz_max.unwrap_or(1700.0);
        let intensity_min_val = intensity_min.unwrap_or(1.0);

        // Get PASEF meta for this frame - these define which precursors were SELECTED
        let Some(pasef_meta) = self.transmission_settings.pasef_meta.get(&(frame_id as i32)) else {
            return TimsFrameAnnotated::new(frame_id as i32, rt, ms_type, vec![], vec![], vec![], vec![], vec![], vec![]);
        };

        // Preallocate with estimated capacity
        let estimated_capacity = pasef_meta.len() * 4;
        let mut tims_spectra: Vec<TimsSpectrumAnnotated> = Vec::with_capacity(estimated_capacity);

        // Iterate over PASEF meta entries - each represents a SELECTED precursor
        for meta in pasef_meta.iter() {
            // Get the selected ion_id from the PASEF precursor field
            let ion_id = meta.precursor as u32;

            // Look up peptide_id and charge from the ion_id
            let Some(&(peptide_id, charge_state)) = self.precursor_frame_builder.ion_id_to_peptide_charge.get(&ion_id) else {
                continue;
            };

            // Get peptide info
            let Some(peptide) = self.precursor_frame_builder.peptides.get(&peptide_id) else {
                continue;
            };

            // Check if this peptide is active in this frame
            if frame_id < peptide.frame_start || frame_id > peptide.frame_end {
                continue;
            }

            // Get ion info from peptide_to_ions
            let Some((ion_abundances, scan_occurrences, scan_abundances, charges, _)) = self
                .precursor_frame_builder
                .peptide_to_ions
                .get(&peptide_id)
            else {
                continue;
            };

            // Find the index of this specific charge state
            let Some(ion_index) = charges.iter().position(|&c| c == charge_state) else {
                continue;
            };

            let ion_abundance = ion_abundances[ion_index];
            let all_scan_occurrence = &scan_occurrences[ion_index];
            let all_scan_abundance = &scan_abundances[ion_index];
            let total_events = *self.precursor_frame_builder.peptide_to_events.get(&peptide_id).unwrap();

            // Get frame abundance for this peptide
            let frame_abundance = self.precursor_frame_builder.frame_to_abundances
                .get(&frame_id)
                .and_then(|(pep_ids, abundances)| {
                    pep_ids.iter().position(|&p| p == peptide_id)
                        .map(|idx| abundances[idx])
                })
                .unwrap_or(0.0);

            if frame_abundance == 0.0 {
                continue;
            }

            // Collision energy from the PASEF meta
            let collision_energy = meta.collision_energy;
            let collision_energy_quantized = (collision_energy * 1e1).round() as i32;

            // Look up fragment ions for this (peptide_id, charge, CE)
            let Some((_, fragment_series_vec)) = fragment_ions.get(&(peptide_id, charge_state, collision_energy_quantized)) else {
                continue;
            };

            // Create ion for annotation
            let ion = PeptideIon::new(
                peptide.sequence.sequence.clone(),
                charge_state as i32,
                ion_abundance as f64,
                Some(peptide_id as i32),
            );

            // Process scans within the PASEF selection window
            for (scan, scan_abundance) in all_scan_occurrence.iter().zip(all_scan_abundance.iter()) {
                // Check if scan is within the PASEF selection window
                let scan_i32 = *scan as i32;
                if scan_i32 < meta.scan_start || scan_i32 > meta.scan_end {
                    continue;
                }

                // Calculate abundance factor
                let fraction_events = frame_abundance * scan_abundance * ion_abundance * total_events;

                // Cache scan mobility
                let scan_mobility = *self.precursor_frame_builder.scan_to_mobility.get(scan).unwrap() as f64;

                for fragment_ion_series in fragment_series_vec.iter() {
                    let scaled_spec = fragment_ion_series.clone() * fraction_events as f64;

                    let mz_spectrum = if mz_noise_fragment {
                        if uniform {
                            scaled_spec.add_mz_noise_uniform(fragment_ppm, right_drag_val)
                        } else {
                            scaled_spec.add_mz_noise_normal(fragment_ppm)
                        }
                    } else {
                        scaled_spec
                    };

                    let spectrum_len = mz_spectrum.mz.len();
                    tims_spectra.push(TimsSpectrumAnnotated::new(
                        frame_id as i32,
                        *scan,
                        rt,
                        scan_mobility,
                        ms_type.clone(),
                        vec![0; spectrum_len],
                        mz_spectrum,
                    ));
                }
            }
        }

        if tims_spectra.is_empty() {
            return TimsFrameAnnotated::new(frame_id as i32, rt, ms_type, vec![], vec![], vec![], vec![], vec![], vec![]);
        }

        TimsFrameAnnotated::from_tims_spectra_annotated(tims_spectra).filter_ranged(
            mz_min_val, mz_max_val, 0.0, 10.0, 0, 1000, intensity_min_val, 1e9,
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