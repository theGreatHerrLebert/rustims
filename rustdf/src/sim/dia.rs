use mscore::algorithm::isotope::{
    calculate_precursor_transmission_factor,
    calculate_transmission_dependent_fragment_ion_isotope_distribution,
};
use mscore::data::peptide::{PeptideIon, PeptideProductIonSeriesCollection};
use mscore::data::spectrum::{IndexedMzSpectrum, MsType, MzSpectrum};
use mscore::simulation::annotation::{
    MzSpectrumAnnotated, PeakAnnotation, TimsFrameAnnotated, TimsSpectrumAnnotated,
};
use mscore::timstof::collision::{TimsTofCollisionEnergy, TimsTofCollisionEnergyDIA};
use mscore::timstof::frame::TimsFrame;
use mscore::timstof::quadrupole::{IonTransmission, TimsTransmissionDIA, WindowTransmission};
use mscore::timstof::spectrum::TimsSpectrum;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use rand::Rng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::sim::containers::{IsotopeTransmissionConfig, IsotopeTransmissionMode};
use crate::sim::projector::{IntensityStage, MzCoordSpace, RenderedEvent, RenderedSpectrum};
use crate::sim::scheme::{DataMode, InstrumentCapabilities, IsolationWindow};
use crate::sim::handle::{FragmentIonsWithComplementary, TimsTofSyntheticsDataHandle};
use crate::sim::precursor::TimsTofSyntheticsPrecursorFrameBuilder;

/// Vendor-neutral per-fragment-series spectrum (P6a MS2 physics kernel): scale one
/// fragment ion series into its `MzSpectrum` under the isotope-transmission mode,
/// PRE m/z-noise. This is the fragment physics shared by every instrument — the
/// transmission GATING (which scans/windows reach here) and the per-scan vs
/// collapsed aggregation stay in the vendor adapter. `transmission_factor` is the
/// precursor-scaling factor (PrecursorScaling), `frag_data` the per-(peptide,
/// charge,CE) complementary data (PerFragment), `transmitted_indices` the
/// quadrupole-transmitted precursor-isotope indices. Extracted verbatim from the
/// inline DIA/DDA computation so rendered output is unchanged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fragment_series_spectrum(
    mode: IsotopeTransmissionMode,
    fragment_ion_series: &MzSpectrum,
    series_idx: usize,
    fraction_events: f32,
    transmission_factor: f64,
    frag_data: Option<&FragmentIonsWithComplementary>,
    transmitted_indices: &HashSet<usize>,
    max_isotopes: usize,
) -> MzSpectrum {
    match mode {
        IsotopeTransmissionMode::None => fragment_ion_series.clone() * fraction_events as f64,
        IsotopeTransmissionMode::PrecursorScaling => {
            fragment_ion_series.clone() * (fraction_events as f64 * transmission_factor)
        }
        IsotopeTransmissionMode::PerFragment => {
            if let Some(frag_data) = frag_data {
                if series_idx < frag_data.per_fragment_data.len() {
                    let series_data = &frag_data.per_fragment_data[series_idx];
                    let mut aggregated_mz: Vec<f64> = Vec::new();
                    let mut aggregated_intensity: Vec<f64> = Vec::new();
                    for frag_ion_data in series_data {
                        let adjusted_dist =
                            calculate_transmission_dependent_fragment_ion_isotope_distribution(
                                &frag_ion_data.fragment_distribution,
                                &frag_ion_data.complementary_distribution,
                                transmitted_indices,
                                max_isotopes,
                            );
                        for (mz, abundance) in adjusted_dist {
                            aggregated_mz.push(mz);
                            aggregated_intensity
                                .push(abundance * frag_ion_data.predicted_intensity * fraction_events as f64);
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
        }
    }
}

pub struct TimsTofSyntheticsFrameBuilderDIA {
    pub path: String,
    pub precursor_frame_builder: TimsTofSyntheticsPrecursorFrameBuilder,
    pub transmission_settings: TimsTransmissionDIA,
    pub fragmentation_settings: TimsTofCollisionEnergyDIA,
    pub fragment_ions:
        Option<BTreeMap<(u32, i8, i32), (PeptideProductIonSeriesCollection, Vec<MzSpectrum>)>>,
    pub fragment_ions_annotated: Option<
        BTreeMap<(u32, i8, i32), (PeptideProductIonSeriesCollection, Vec<MzSpectrumAnnotated>)>,
    >,
    /// Configuration for quad-selection dependent isotope transmission (already
    /// gated by the instrument capabilities at construction — P5e).
    pub isotope_config: IsotopeTransmissionConfig,
    /// Fragment ions with complementary data for transmission-dependent calculations
    pub fragment_ions_with_transmission:
        Option<BTreeMap<(u32, i8, i32), FragmentIonsWithComplementary>>,
    /// Physical instrument capabilities (P5e). Default = Bruker timsTOF.
    pub capabilities: InstrumentCapabilities,
}

impl TimsTofSyntheticsFrameBuilderDIA {
    pub fn new(path: &Path, with_annotations: bool, num_threads: usize) -> rusqlite::Result<Self> {
        Self::new_with_config(path, with_annotations, num_threads, IsotopeTransmissionConfig::default())
    }

    pub fn new_with_config(
        path: &Path,
        with_annotations: bool,
        num_threads: usize,
        isotope_config: IsotopeTransmissionConfig,
    ) -> rusqlite::Result<Self> {
        Self::new_with_config_and_source(
            path,
            with_annotations,
            num_threads,
            isotope_config,
            &crate::sim::projector::DistributionSource::Columns,
        )
    }

    /// As [`new_with_config`], but the precursor builder's occurrence/abundance
    /// distributions come from `source` (P4: `Columns` default, or the projector).
    pub fn new_with_config_and_source(
        path: &Path,
        with_annotations: bool,
        num_threads: usize,
        isotope_config: IsotopeTransmissionConfig,
        source: &crate::sim::projector::DistributionSource,
    ) -> rusqlite::Result<Self> {
        let synthetics = TimsTofSyntheticsPrecursorFrameBuilder::from_source(path, source)?;
        let handle = TimsTofSyntheticsDataHandle::new(path)?;

        // P5b: refuse to render fragments stored under an incompatible prediction
        // set (CE encoding the render keying can't resolve). Bruker/legacy pass.
        handle
            .read_prediction_set()?
            .assert_render_compatible()
            .map_err(|_| rusqlite::Error::InvalidQuery)?;

        let fragment_ions = handle.read_fragment_ions()?;

        // P5e: gate the isotope-transmission config by instrument capabilities.
        // Default = Bruker timsTOF (no-op); P6 threads real Astral capabilities.
        let capabilities = InstrumentCapabilities::default();
        let isotope_config = isotope_config.gated_by(capabilities);

        // get collision energy settings per window group
        let fragmentation_settings = handle.get_collision_energy_dia();
        // get ion transmission settings per window group
        let transmission_settings = handle.get_transmission_dia();

        // Build transmission data if isotope transmission is enabled
        let fragment_ions_with_transmission = if isotope_config.is_enabled() {
            Some(TimsTofSyntheticsDataHandle::build_fragment_ions_with_transmission_data(
                &synthetics.peptides,
                &fragment_ions,
                num_threads,
            ))
        } else {
            None
        };

        match with_annotations {
            true => {
                let fragment_ions_annotated =
                    Some(TimsTofSyntheticsDataHandle::build_fragment_ions_annotated(
                        &synthetics.peptides,
                        &fragment_ions,
                        num_threads,
                    ));
                Ok(Self {
                    path: path.to_str().unwrap().to_string(),
                    precursor_frame_builder: synthetics,
                    transmission_settings,
                    fragmentation_settings,
                    fragment_ions: None,
                    fragment_ions_annotated,
                    isotope_config,
                    fragment_ions_with_transmission,
                    capabilities,
                })
            }

            false => {
                let fragment_ions = Some(TimsTofSyntheticsDataHandle::build_fragment_ions(
                    &synthetics.peptides,
                    &fragment_ions,
                    num_threads,
                ));
                Ok(Self {
                    path: path.to_str().unwrap().to_string(),
                    precursor_frame_builder: synthetics,
                    transmission_settings,
                    fragmentation_settings,
                    fragment_ions,
                    fragment_ions_annotated: None,
                    isotope_config,
                    fragment_ions_with_transmission,
                    capabilities,
                })
            }
        }
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
        // Use global thread pool with custom parallelism instead of creating new pool each call
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            // Use indexed parallel iteration to maintain order, avoiding post-sort
            let mut tims_frames: Vec<TimsFrame> = Vec::with_capacity(frame_ids.len());
            // Safety: we're about to fill all elements
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
                // Safety: each index is unique due to enumerate
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
        // Cache frame-level lookups once
        let ms_type = if self.precursor_frame_builder.precursor_frame_id_set.contains(&frame_id) {
            MsType::Unknown
        } else {
            MsType::FragmentDia
        };

        let rt = *self.precursor_frame_builder.frame_to_rt.get(&frame_id).unwrap() as f64;
        let right_drag_val = right_drag.unwrap_or(false);
        let mz_min_val = mz_min.unwrap_or(100.0);
        let mz_max_val = mz_max.unwrap_or(1700.0);
        let intensity_min_val = intensity_min.unwrap_or(1.0);

        // Use single lookup instead of contains_key + get
        let Some((peptide_ids, frame_abundances)) = self
            .precursor_frame_builder
            .frame_to_abundances
            .get(&frame_id)
        else {
            return TimsFrame::new(frame_id as i32, ms_type, rt, vec![], vec![], vec![], vec![], vec![]);
        };

        // Preallocate with estimated capacity
        let estimated_capacity = peptide_ids.len() * 4;
        let mut tims_spectra: Vec<TimsSpectrum> = Vec::with_capacity(estimated_capacity);

        // Go over all peptides in the frame with their respective abundances
        for (peptide_id, frame_abundance) in peptide_ids.iter().zip(frame_abundances.iter()) {
            // Single lookup instead of contains_key + get
            let Some((ion_abundances, scan_occurrences, scan_abundances, charges, spectra)) = self
                .precursor_frame_builder
                .peptide_to_ions
                .get(peptide_id)
            else {
                continue;
            };

            // Cache peptide-level lookup
            let total_events = *self.precursor_frame_builder.peptide_to_events.get(peptide_id).unwrap();

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let all_scan_occurrence = &scan_occurrences[index];
                let all_scan_abundance = &scan_abundances[index];
                let spectrum = &spectra[index];
                let charge_state = charges[index];

                for (scan, scan_abundance) in all_scan_occurrence.iter().zip(all_scan_abundance.iter()) {
                    // Get transmitted isotope indices based on config mode
                    let (is_transmitted, transmitted_indices) = match self.isotope_config.mode {
                        IsotopeTransmissionMode::None => {
                            // Standard check without indices
                            let any = self.transmission_settings.any_transmitted(
                                frame_id as i32,
                                *scan as i32,
                                &spectrum.mz,
                                None,
                            );
                            (any, HashSet::new())
                        },
                        IsotopeTransmissionMode::PrecursorScaling | IsotopeTransmissionMode::PerFragment => {
                            let indices = self.transmission_settings.get_transmission_set(
                                frame_id as i32,
                                *scan as i32,
                                &spectrum.mz,
                                Some(self.isotope_config.min_probability),
                            );
                            (!indices.is_empty(), indices)
                        },
                    };

                    if !is_transmitted {
                        continue;
                    }

                    // Calculate abundance factor (total_events cached above)
                    let fraction_events = frame_abundance * scan_abundance * ion_abundance * total_events;

                    // Get collision energy for the ion
                    let collision_energy = self.fragmentation_settings.get_collision_energy(frame_id as i32, *scan as i32);
                    // Resolve the fragment CE key, tolerant to ~0.1 eV quantization
                    // noise (no-op for DIA's pre-rounded window CE; shared with DDA).
                    let Some(collision_energy_quantized) = crate::sim::handle::resolve_fragment_ce_key(
                        fragment_ions, *peptide_id, charge_state, collision_energy,
                    ) else {
                        // Fail loud if fragments exist for this ion but none near the
                        // applied CE (prediction set doesn't cover this CE, P5b); a
                        // precursor with no predicted fragments at all is a legit skip.
                        if crate::sim::handle::fragment_prefix_exists(fragment_ions, *peptide_id, charge_state) {
                            panic!(
                                "DIA fragment lookup miss: peptide {} charge {} applied CE {:.4} eV \
                                 has predicted fragments, but none within 0.1 eV — the prediction \
                                 set does not cover this instrument's collision energy",
                                *peptide_id, charge_state, collision_energy,
                            );
                        }
                        continue;
                    };
                    let (_, fragment_series_vec) = fragment_ions
                        .get(&(*peptide_id, charge_state, collision_energy_quantized))
                        .expect("resolve_fragment_ce_key returned a present key");

                    // Cache scan mobility lookup
                    let scan_mobility = *self.precursor_frame_builder.scan_to_mobility.get(scan).unwrap() as f64;

                    // Calculate transmission factor for PrecursorScaling mode
                    let transmission_factor = if self.isotope_config.mode == IsotopeTransmissionMode::PrecursorScaling {
                        if let Some(comp_data) = self.fragment_ions_with_transmission.as_ref() {
                            if let Some(frag_data) = comp_data.get(&(*peptide_id, charge_state, collision_energy_quantized)) {
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

                    // Complementary per-(peptide,charge,CE) data for PerFragment;
                    // looked up once (same for all series) and passed to the kernel.
                    let frag_data = self
                        .fragment_ions_with_transmission
                        .as_ref()
                        .and_then(|c| c.get(&(*peptide_id, charge_state, collision_energy_quantized)));

                    for (series_idx, fragment_ion_series) in fragment_series_vec.iter().enumerate() {
                        let final_spectrum = fragment_series_spectrum(
                            self.isotope_config.mode,
                            fragment_ion_series,
                            series_idx,
                            fraction_events,
                            transmission_factor,
                            frag_data,
                            &transmitted_indices,
                            self.isotope_config.max_isotopes,
                        );

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

                    // Add unfragmented precursor ions (survival) if configured
                    if self.isotope_config.has_precursor_survival() {
                        let mut rng = rand::thread_rng();
                        let survival_fraction = rng.gen_range(
                            self.isotope_config.precursor_survival_min
                            ..=self.isotope_config.precursor_survival_max
                        );

                        if survival_fraction > 0.0 {
                            // Transmit the precursor spectrum through the quadrupole
                            let precursor_transmitted = self.transmission_settings.transmit_spectrum(
                                frame_id as i32,
                                *scan as i32,
                                spectrum.clone(),
                                Some(self.isotope_config.min_probability),
                            );

                            if !precursor_transmitted.mz.is_empty() {
                                // Scale by survival fraction and event count
                                let precursor_scaled = precursor_transmitted * (fraction_events as f64 * survival_fraction);

                                let precursor_mz_spectrum = if mz_noise_fragment {
                                    if uniform {
                                        precursor_scaled.add_mz_noise_uniform(fragment_ppm, right_drag_val)
                                    } else {
                                        precursor_scaled.add_mz_noise_normal(fragment_ppm)
                                    }
                                } else {
                                    precursor_scaled
                                };

                                let precursor_len = precursor_mz_spectrum.mz.len();
                                tims_spectra.push(TimsSpectrum::new(
                                    frame_id as i32,
                                    *scan as i32,
                                    rt,
                                    scan_mobility,
                                    ms_type.clone(),
                                    IndexedMzSpectrum::from_mz_spectrum(
                                        vec![0; precursor_len],
                                        precursor_mz_spectrum,
                                    ).filter_ranged(100.0, 1700.0, 1.0, 1e9),
                                ));
                            }
                        }
                    }
                }
            }
        }

        if tims_spectra.is_empty() {
            return TimsFrame::new(frame_id as i32, ms_type, rt, vec![], vec![], vec![], vec![], vec![]);
        }

        let tims_frame = TimsFrame::from_tims_spectra(tims_spectra);
        tims_frame.filter_ranged(
            mz_min_val,
            mz_max_val,
            0,
            1000,
            0.0,
            10.0,
            intensity_min_val,
            1e9,
            0,
            i32::MAX,
        )
    }

    /// Render a fragment (MS2) frame as a vendor-neutral [`RenderedEvent::Scan`]
    /// for a non-IMS instrument (P6e): collapse ONE isolation window's fragment
    /// signal into the single MS2 spectrum an Astral/Orbitrap records.
    ///
    /// Astral has no ion-mobility axis, so (a) each precursor ion is gated into the
    /// window ONCE via [`WindowTransmission::any_transmitted`] on its isotope
    /// envelope (an m/z window, NOT a scan range), and (b) every transmitted ion
    /// contributes its fragment series at the FULL mobility marginal
    /// `frame_abundance × ion_abundance × total_events` (scan factor folded to 1.0)
    /// — the same marginal contract as the MS1 render
    /// ([`TimsTofSyntheticsPrecursorFrameBuilder::precursor_scan_marginal_spectrum`]),
    /// NOT the Bruker per-scan `scan_abundance` sum. The isotope-transmission mode
    /// is `None` (Astral capabilities gate it off), so the shared
    /// [`fragment_series_spectrum`] kernel does the per-series scaling. `nce` is the
    /// window's normalized collision energy: both the CE key the stored fragments
    /// were predicted at AND the applied CE (the keying is unit-agnostic). A
    /// precursor whose fragments are not stored near `nce` is SKIPPED (predicted at
    /// a different window's CE) — compatibility is enforced once at registration
    /// (P6d), so the render core never panics on a per-precursor miss. Returns an
    /// empty-spectrum MS2 `Scan` when nothing is transmitted / no fragments exist
    /// — the writer must still consume and clear that template slot (zero residual).
    ///
    /// Fragments ONLY: precursor-survival signal is intentionally not modelled here
    /// (it is stochastic, which would break this deterministic render), and an
    /// Astral run that configures `precursor_survival_*` is rejected at config load
    /// rather than silently dropping it.
    pub fn render_fragment_scan(
        &self,
        frame_id: u32,
        window: &WindowTransmission,
        nce: f64,
        data_mode: DataMode,
    ) -> RenderedEvent {
        let rt = *self
            .precursor_frame_builder
            .frame_to_rt
            .get(&frame_id)
            .unwrap_or(&0.0) as f64;
        let isolation = Some(IsolationWindow {
            center_mz: window.center_mz,
            width_mz: window.width_mz,
        });
        let make_scan = |peaks: MzSpectrum| RenderedEvent::Scan {
            ms_level: 2,
            retention_time_s: rt,
            isolation,
            spectrum: RenderedSpectrum {
                mz: (*peaks.mz).clone(),
                intensity: (*peaks.intensity).clone(),
                coords: MzCoordSpace::Physical,
                mode: data_mode,
                detector_applied: false,
                stage: IntensityStage::Transmitted,
            },
        };

        let (Some(fragment_ions), Some((peptide_ids, frame_abundances))) = (
            self.fragment_ions.as_ref(),
            self.precursor_frame_builder.frame_to_abundances.get(&frame_id),
        ) else {
            return make_scan(MzSpectrum::from_collection(vec![]));
        };

        let mut specs: Vec<MzSpectrum> = Vec::new();
        for (peptide_id, frame_abundance) in peptide_ids.iter().zip(frame_abundances.iter()) {
            let Some((ion_abundances, _scan_occ, _scan_abu, charges, spectra)) =
                self.precursor_frame_builder.peptide_to_ions.get(peptide_id)
            else {
                continue;
            };
            let total_events = *self
                .precursor_frame_builder
                .peptide_to_events
                .get(peptide_id)
                .unwrap();
            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let spectrum = &spectra[index];
                let charge_state = charges[index];
                // Quadrupole window gating (m/z), once per ion — no IMS scans.
                if !window.any_transmitted(&spectrum.mz, None) {
                    continue;
                }
                // Resolve the fragment CE key at the window NCE (unit-agnostic).
                // A miss SKIPS this precursor for this window — unlike the Bruker
                // same-instrument frame builder (which panics on a coverage gap),
                // the Astral render gates precursors by m/z ALONE (no mobility), so
                // a precursor whose fragments were predicted at a different window's
                // CE legitimately does not contribute here. Prediction-set/instrument
                // compatibility is enforced once, up front, at registration (P6d).
                let Some(ce_key) = crate::sim::handle::resolve_fragment_ce_key(
                    fragment_ions,
                    *peptide_id,
                    charge_state,
                    nce,
                ) else {
                    continue;
                };
                let (_, fragment_series_vec) = fragment_ions
                    .get(&(*peptide_id, charge_state, ce_key))
                    .expect("resolve_fragment_ce_key returned a present key");

                // Full mobility marginal: scan factor folded to 1.0 (no grid).
                let fraction_events = frame_abundance * ion_abundance * total_events;
                for (series_idx, fragment_ion_series) in fragment_series_vec.iter().enumerate() {
                    specs.push(fragment_series_spectrum(
                        IsotopeTransmissionMode::None,
                        fragment_ion_series,
                        series_idx,
                        fraction_events,
                        1.0,
                        None,
                        &HashSet::new(),
                        self.isotope_config.max_isotopes,
                    ));
                }
            }
        }
        make_scan(MzSpectrum::from_collection(specs))
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
            MsType::FragmentDia
        };

        let rt = *self.precursor_frame_builder.frame_to_rt.get(&frame_id).unwrap() as f64;
        let right_drag_val = right_drag.unwrap_or(false);
        let mz_min_val = mz_min.unwrap_or(100.0);
        let mz_max_val = mz_max.unwrap_or(1700.0);
        let intensity_min_val = intensity_min.unwrap_or(1.0);

        // Single lookup instead of contains_key + get
        let Some((peptide_ids, frame_abundances)) = self
            .precursor_frame_builder
            .frame_to_abundances
            .get(&frame_id)
        else {
            return TimsFrameAnnotated::new(frame_id as i32, rt, ms_type, vec![], vec![], vec![], vec![], vec![], vec![]);
        };

        // Preallocate with estimated capacity
        let estimated_capacity = peptide_ids.len() * 4;
        let mut tims_spectra: Vec<TimsSpectrumAnnotated> = Vec::with_capacity(estimated_capacity);

        for (peptide_id, frame_abundance) in peptide_ids.iter().zip(frame_abundances.iter()) {
            // Single lookup
            let Some((ion_abundances, scan_occurrences, scan_abundances, charges, _)) = self
                .precursor_frame_builder
                .peptide_to_ions
                .get(peptide_id)
            else {
                continue;
            };

            // Cache peptide-level lookups
            let total_events = *self.precursor_frame_builder.peptide_to_events.get(peptide_id).unwrap();
            let peptide = self.precursor_frame_builder.peptides.get(peptide_id).unwrap();

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let all_scan_occurrence = &scan_occurrences[index];
                let all_scan_abundance = &scan_abundances[index];
                let charge_state = charges[index];

                let ion = PeptideIon::new(
                    peptide.sequence.sequence.clone(),
                    charge_state as i32,
                    *ion_abundance as f64,
                    Some(*peptide_id as i32),
                );
                // TODO: make this configurable
                let spectrum = ion.calculate_isotopic_spectrum_annotated(1e-3, 1e-8, 200, 1e-4);

                for (scan, scan_abundance) in all_scan_occurrence.iter().zip(all_scan_abundance.iter()) {
                    if !self.transmission_settings.any_transmitted(
                        frame_id as i32,
                        *scan as i32,
                        &spectrum.mz,
                        None,
                    ) {
                        continue;
                    }

                    let fraction_events = frame_abundance * scan_abundance * ion_abundance * total_events;

                    let collision_energy = self.fragmentation_settings.get_collision_energy(frame_id as i32, *scan as i32);
                    // Resolve the fragment CE key, tolerant to ~0.1 eV quantization noise.
                    let Some(collision_energy_quantized) = crate::sim::handle::resolve_fragment_ce_key(
                        fragment_ions, *peptide_id, charge_state, collision_energy,
                    ) else {
                        if crate::sim::handle::fragment_prefix_exists(fragment_ions, *peptide_id, charge_state) {
                            panic!(
                                "DIA (annotated) fragment lookup miss: peptide {} charge {} applied CE \
                                 {:.4} eV has predicted fragments, but none within 0.1 eV — the \
                                 prediction set does not cover this instrument's collision energy",
                                *peptide_id, charge_state, collision_energy,
                            );
                        }
                        continue;
                    };
                    let (_, fragment_series_vec) = fragment_ions
                        .get(&(*peptide_id, charge_state, collision_energy_quantized))
                        .expect("resolve_fragment_ce_key returned a present key");

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

                    // Add unfragmented precursor ions (survival) if configured
                    if self.isotope_config.has_precursor_survival() {
                        let mut rng = rand::thread_rng();
                        let survival_fraction = rng.gen_range(
                            self.isotope_config.precursor_survival_min
                            ..=self.isotope_config.precursor_survival_max
                        );

                        if survival_fraction > 0.0 {
                            // Create a non-annotated spectrum for transmission
                            let precursor_mz_spectrum = MzSpectrum::new(spectrum.mz.clone(), spectrum.intensity.clone());

                            // Transmit through the quadrupole
                            let precursor_transmitted = self.transmission_settings.transmit_spectrum(
                                frame_id as i32,
                                *scan as i32,
                                precursor_mz_spectrum,
                                Some(self.isotope_config.min_probability),
                            );

                            if !precursor_transmitted.mz.is_empty() {
                                // Scale by survival fraction and event count
                                let precursor_scaled = precursor_transmitted * (fraction_events as f64 * survival_fraction);

                                let precursor_final = if mz_noise_fragment {
                                    if uniform {
                                        precursor_scaled.add_mz_noise_uniform(fragment_ppm, right_drag_val)
                                    } else {
                                        precursor_scaled.add_mz_noise_normal(fragment_ppm)
                                    }
                                } else {
                                    precursor_scaled
                                };

                                // Convert to annotated spectrum (with precursor annotations)
                                let annotations: Vec<PeakAnnotation> = precursor_final.mz.iter()
                                    .map(|_| PeakAnnotation { contributions: vec![] })
                                    .collect();
                                let precursor_annotated = MzSpectrumAnnotated::new(
                                    precursor_final.mz.to_vec(),
                                    precursor_final.intensity.to_vec(),
                                    annotations,
                                );

                                let precursor_len = precursor_annotated.mz.len();
                                tims_spectra.push(TimsSpectrumAnnotated::new(
                                    frame_id as i32,
                                    *scan,
                                    rt,
                                    scan_mobility,
                                    ms_type.clone(),
                                    vec![0; precursor_len],
                                    precursor_annotated,
                                ));
                            }
                        }
                    }
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

    pub fn get_ion_transmission_matrix(
        &self,
        peptide_id: u32,
        charge: i8,
        include_precursor_frames: bool,
    ) -> Vec<Vec<f32>> {
        let maybe_peptide_sim = self.precursor_frame_builder.peptides.get(&peptide_id);

        let mut frame_ids = match maybe_peptide_sim {
            Some(maybe_peptide_sim) => maybe_peptide_sim.frame_distribution.occurrence.clone(),
            _ => vec![],
        };

        if !include_precursor_frames {
            frame_ids = frame_ids
                .iter()
                .filter(|frame_id| {
                    !self
                        .precursor_frame_builder
                        .precursor_frame_id_set
                        .contains(frame_id)
                })
                .cloned()
                .collect();
        }

        let ion = self
            .precursor_frame_builder
            .ions
            .get(&peptide_id)
            .unwrap()
            .iter()
            .find(|ion| ion.charge == charge)
            .unwrap();
        let spectrum = ion.simulated_spectrum.clone();
        let scan_distribution = &ion.scan_distribution;

        let mut transmission_matrix =
            vec![vec![0.0; frame_ids.len()]; scan_distribution.occurrence.len()];

        for (frame_index, frame) in frame_ids.iter().enumerate() {
            for (scan_index, scan) in scan_distribution.occurrence.iter().enumerate() {
                if self.transmission_settings.all_transmitted(
                    *frame as i32,
                    *scan as i32,
                    &spectrum.mz,
                    None,
                ) {
                    transmission_matrix[scan_index][frame_index] = 1.0;
                } else if self.transmission_settings.any_transmitted(
                    *frame as i32,
                    *scan as i32,
                    &spectrum.mz,
                    None,
                ) {
                    let transmitted_spectrum = self.transmission_settings.transmit_spectrum(
                        *frame as i32,
                        *scan as i32,
                        spectrum.clone(),
                        None,
                    );
                    let percentage_transmitted = transmitted_spectrum.intensity.iter().sum::<f64>()
                        / spectrum.intensity.iter().sum::<f64>();
                    transmission_matrix[scan_index][frame_index] = percentage_transmitted as f32;
                }
            }
        }

        transmission_matrix
    }

    pub fn count_number_transmissions(&self, peptide_id: u32, charge: i8) -> (usize, usize) {
        let frame_ids: Vec<_> = self
            .precursor_frame_builder
            .peptides
            .get(&peptide_id)
            .unwrap()
            .frame_distribution
            .occurrence
            .clone()
            .iter()
            .filter(|frame_id| {
                !self
                    .precursor_frame_builder
                    .precursor_frame_id_set
                    .contains(frame_id)
            })
            .cloned()
            .collect();
        let ion = self
            .precursor_frame_builder
            .ions
            .get(&peptide_id)
            .unwrap()
            .iter()
            .find(|ion| ion.charge == charge)
            .unwrap();
        let spectrum = ion.simulated_spectrum.clone();
        let scan_distribution = &ion.scan_distribution;

        let mut frame_count = 0;
        let mut scan_count = 0;

        for frame in frame_ids.iter() {
            let mut frame_transmitted = false;
            for scan in scan_distribution.occurrence.iter() {
                if self.transmission_settings.any_transmitted(
                    *frame as i32,
                    *scan as i32,
                    &spectrum.mz,
                    None,
                ) {
                    frame_transmitted = true;
                    scan_count += 1;
                }
            }
            if frame_transmitted {
                frame_count += 1;
            }
        }

        (frame_count, scan_count)
    }

    pub fn count_number_transmissions_parallel(
        &self,
        peptide_ids: Vec<u32>,
        charge: Vec<i8>,
        num_threads: usize,
    ) -> Vec<(usize, usize)> {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        let result: Vec<(usize, usize)> = thread_pool.install(|| {
            peptide_ids
                .par_iter()
                .zip(charge.par_iter())
                .map(|(peptide_id, charge)| self.count_number_transmissions(*peptide_id, *charge))
                .collect()
        });

        result
    }
}

impl TimsTofCollisionEnergy for TimsTofSyntheticsFrameBuilderDIA {
    fn get_collision_energy(&self, frame_id: i32, scan_id: i32) -> f64 {
        self.fragmentation_settings
            .get_collision_energy(frame_id, scan_id)
    }
}

#[cfg(test)]
mod p6a_fragment_kernel_tests {
    use super::*;
    use std::collections::HashSet;

    // The scaling branches of the shared fragment-series kernel. (PerFragment is a
    // verbatim move of the inline computation; no DB simulated with quad-transmission
    // data is available to integration-test it, so it is covered by the byte-parity
    // gate on the None path + code review.)
    #[test]
    fn fragment_series_spectrum_scaling_branches() {
        let series = MzSpectrum::new(vec![200.0, 300.0], vec![10.0, 20.0]);
        let empty: HashSet<usize> = HashSet::new();

        // None: scale by fraction_events only.
        let none = fragment_series_spectrum(
            IsotopeTransmissionMode::None, &series, 0, 2.0, 99.0 /*ignored*/, None, &empty, 10,
        );
        assert_eq!(*none.mz, vec![200.0, 300.0]);
        assert_eq!(*none.intensity, vec![20.0, 40.0]);

        // PrecursorScaling: scale by fraction_events * transmission_factor.
        let ps = fragment_series_spectrum(
            IsotopeTransmissionMode::PrecursorScaling, &series, 0, 2.0, 0.5, None, &empty, 10,
        );
        assert_eq!(*ps.intensity, vec![10.0, 20.0]); // 10*(2*0.5), 20*(2*0.5)

        // PerFragment with no complementary data falls back to None-style scaling.
        let pf = fragment_series_spectrum(
            IsotopeTransmissionMode::PerFragment, &series, 0, 2.0, 0.5, None, &empty, 10,
        );
        assert_eq!(*pf.intensity, vec![20.0, 40.0]);
    }

    use crate::sim::precursor::TimsTofSyntheticsPrecursorFrameBuilder;

    /// Hand-built single-frame DIA builder: frame 2 holds peptide 10 (charge 2,
    /// precursor m/z 600) at frame_abundance 0.8, total_events 1000, captured on a
    /// scan grid Σ scan_abundance = 0.5. Its fragments (one b/y series, a single
    /// peak at m/z 200) are stored at NCE 27 (map key round(27*10)=270).
    fn one_fragment_frame_dia() -> TimsTofSyntheticsFrameBuilderDIA {
        let mut frame_to_abundances = BTreeMap::new();
        frame_to_abundances.insert(2u32, (vec![10u32], vec![0.8f32]));
        let mut peptide_to_ions = BTreeMap::new();
        peptide_to_ions.insert(
            10u32,
            (
                vec![1.0f32],
                vec![vec![100u32, 101u32]],
                vec![vec![0.3f32, 0.2f32]], // Σ = 0.5 captured (ignored by the marginal)
                vec![2i8],
                vec![MzSpectrum::new(vec![600.0], vec![1.0])], // precursor envelope
            ),
        );
        let mut peptide_to_events = BTreeMap::new();
        peptide_to_events.insert(10u32, 1000.0f32);
        let mut frame_to_rt = BTreeMap::new();
        frame_to_rt.insert(2u32, 30.0f32);

        let precursor = TimsTofSyntheticsPrecursorFrameBuilder {
            ions: BTreeMap::new(),
            peptides: BTreeMap::new(),
            scans: Vec::new(),
            frames: Vec::new(),
            precursor_frame_id_set: HashSet::new(),
            frame_to_abundances,
            peptide_to_ions,
            frame_to_rt,
            scan_to_mobility: BTreeMap::new(),
            peptide_to_events,
            ion_id_to_peptide_charge: BTreeMap::new(),
        };

        let mut fragment_ions = BTreeMap::new();
        fragment_ions.insert(
            (10u32, 2i8, 270i32), // NCE 27 -> round(27*10)
            (
                PeptideProductIonSeriesCollection::new(vec![]),
                vec![MzSpectrum::new(vec![200.0], vec![1.0])],
            ),
        );

        TimsTofSyntheticsFrameBuilderDIA {
            path: String::new(),
            precursor_frame_builder: precursor,
            transmission_settings: TimsTransmissionDIA::new(
                vec![], vec![], vec![], vec![], vec![], vec![], vec![], None,
            ),
            fragmentation_settings: TimsTofCollisionEnergyDIA::new(
                vec![], vec![], vec![], vec![], vec![], vec![],
            ),
            fragment_ions: Some(fragment_ions),
            fragment_ions_annotated: None,
            isotope_config: IsotopeTransmissionConfig::default(),
            fragment_ions_with_transmission: None,
            capabilities: InstrumentCapabilities::astral(),
        }
    }

    #[test]
    fn astral_ms2_render_full_marginal_and_window_gating() {
        let b = one_fragment_frame_dia();

        // Window covering the precursor (m/z 600): the ion is transmitted once and
        // contributes its fragment series at the FULL mobility marginal
        // 0.8 * 1.0 * 1000 * 1.0 = 800 (NOT the captured Σ scan_abundance=0.5 -> 400).
        let win = WindowTransmission::new(600.0, 50.0, 15.0);
        let RenderedEvent::Scan { ms_level, isolation, spectrum, .. } =
            b.render_fragment_scan(2, &win, 27.0, DataMode::Centroid)
        else {
            panic!("MS2 render must be a Scan");
        };
        assert_eq!(ms_level, 2);
        let iso = isolation.expect("MS2 scan carries an isolation window");
        assert!((iso.center_mz - 600.0).abs() < 1e-9 && (iso.width_mz - 50.0).abs() < 1e-9);
        assert_eq!(spectrum.mz, vec![200.0]);
        assert!((spectrum.intensity[0] - 800.0).abs() < 1e-2, "full marginal, got {}", spectrum.intensity[0]);

        // A window that does NOT cover the precursor transmits nothing -> empty scan
        // (the slot is still authored/cleared by the writer).
        let off = WindowTransmission::new(900.0, 50.0, 15.0);
        let RenderedEvent::Scan { spectrum: empty, .. } =
            b.render_fragment_scan(2, &off, 27.0, DataMode::Centroid)
        else {
            panic!("must be a Scan");
        };
        assert!(empty.mz.is_empty(), "no precursor transmitted -> empty MS2 scan");

        // Determinism.
        let RenderedEvent::Scan { spectrum: s2, .. } =
            b.render_fragment_scan(2, &win, 27.0, DataMode::Centroid)
        else { panic!() };
        assert_eq!(s2.mz, spectrum.mz);
        assert_eq!(s2.intensity, spectrum.intensity);
    }
}
