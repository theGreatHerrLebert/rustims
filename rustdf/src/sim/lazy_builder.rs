//! Lazy frame builders for DIA and DDA synthetic experiments.
//!
//! This module provides `TimsTofLazyFrameBuilderDIA` and `TimsTofLazyFrameBuilderDDA`,
//! memory-efficient alternatives to their non-lazy counterparts that only load
//! peptide/ion data for the frames being built rather than loading everything upfront.

use mscore::data::peptide::PeptideProductIonSeriesCollection;
use mscore::data::spectrum::{IndexedMzSpectrum, MsType, MzSpectrum};
use mscore::timstof::collision::{TimsTofCollisionEnergy, TimsTofCollisionEnergyDIA};
use mscore::timstof::frame::TimsFrame;
use mscore::timstof::quadrupole::{IonTransmission, TimsTransmissionDDA, TimsTransmissionDIA};
use mscore::timstof::spectrum::TimsSpectrum;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;

use crate::sim::containers::{FragmentIonSim, FramesSim, IonSim, PeptidesSim, ScansSim};
use crate::sim::handle::TimsTofSyntheticsDataHandle;

/// A lazy frame builder for DIA experiments that only loads data as needed.
///
/// Unlike `TimsTofSyntheticsFrameBuilderDIA`, this struct does not load all peptides,
/// ions, and fragment ions into memory at construction time. Instead, it stores only
/// the static metadata (frame info, scan info, transmission settings) and loads
/// peptide/ion data on-demand for each batch of frames being built.
///
/// This can significantly reduce memory usage for large simulations.
pub struct TimsTofLazyFrameBuilderDIA {
    /// Path to the SQLite database
    pub db_path: String,
    /// Frame metadata (id, time, ms_type)
    pub frames: Vec<FramesSim>,
    /// Scan metadata (scan_id, mobility)
    pub scans: Vec<ScansSim>,
    /// Set of precursor frame IDs for quick lookup
    pub precursor_frame_id_set: HashSet<u32>,
    /// Map from frame_id to retention_time
    pub frame_to_rt: BTreeMap<u32, f32>,
    /// Map from scan_id to mobility
    pub scan_to_mobility: BTreeMap<u32, f32>,
    /// DIA transmission settings
    pub transmission_settings: TimsTransmissionDIA,
    /// DIA fragmentation/collision energy settings
    pub fragmentation_settings: TimsTofCollisionEnergyDIA,
    /// Number of threads for parallel processing
    pub num_threads: usize,
}

impl TimsTofLazyFrameBuilderDIA {
    /// Create a new lazy frame builder.
    ///
    /// Only loads static metadata (frames, scans, transmission settings).
    /// Peptides, ions, and fragment ions are NOT loaded here.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the SQLite database
    /// * `num_threads` - Number of threads for parallel operations
    ///
    /// # Returns
    ///
    /// Result containing the lazy frame builder
    pub fn new(path: &Path, num_threads: usize) -> rusqlite::Result<Self> {
        let handle = TimsTofSyntheticsDataHandle::new(path)?;

        let frames = handle.read_frames()?;
        let scans = handle.read_scans()?;

        let precursor_frame_id_set = TimsTofSyntheticsDataHandle::build_precursor_frame_id_set(&frames);
        let frame_to_rt = TimsTofSyntheticsDataHandle::build_frame_to_rt(&frames);
        let scan_to_mobility = TimsTofSyntheticsDataHandle::build_scan_to_mobility(&scans);

        let transmission_settings = handle.get_transmission_dia();
        let fragmentation_settings = handle.get_collision_energy_dia();

        Ok(Self {
            db_path: path.to_str().unwrap().to_string(),
            frames,
            scans,
            precursor_frame_id_set,
            frame_to_rt,
            scan_to_mobility,
            transmission_settings,
            fragmentation_settings,
            num_threads,
        })
    }

    /// Load data for a specific frame range from the database.
    ///
    /// Returns peptides, ions, and fragment ions that are relevant to the frame range.
    fn load_data_for_frame_range(
        &self,
        frame_min: u32,
        frame_max: u32,
    ) -> rusqlite::Result<(Vec<PeptidesSim>, Vec<IonSim>, Vec<FragmentIonSim>)> {
        let path = Path::new(&self.db_path);
        let handle = TimsTofSyntheticsDataHandle::new(path)?;

        // Load only peptides for this frame range
        let peptides = handle.read_peptides_for_frame_range(frame_min, frame_max)?;

        if peptides.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        // Get peptide IDs for querying related data
        let peptide_ids: Vec<u32> = peptides.iter().map(|p| p.peptide_id).collect();

        // Load ions and fragment ions for these peptides
        let ions = handle.read_ions_for_peptides(&peptide_ids)?;
        let fragment_ions = handle.read_fragment_ions_for_peptides(&peptide_ids)?;

        Ok((peptides, ions, fragment_ions))
    }

    /// Build frames for a range of frame IDs.
    ///
    /// This method loads only the data needed for the specified frames,
    /// builds the frames, and then releases the loaded data.
    ///
    /// # Arguments
    ///
    /// * `frame_ids` - Vector of frame IDs to build
    /// * `fragmentation` - Whether to include fragmentation
    /// * `mz_noise_precursor` - Whether to add m/z noise to precursor ions
    /// * `uniform` - Whether to use uniform noise distribution
    /// * `precursor_noise_ppm` - Precursor noise in ppm
    /// * `mz_noise_fragment` - Whether to add m/z noise to fragment ions
    /// * `fragment_noise_ppm` - Fragment noise in ppm
    /// * `right_drag` - Whether to use right drag for noise
    ///
    /// # Returns
    ///
    /// Vector of built TimsFrame instances
    pub fn build_frames_lazy(
        &self,
        frame_ids: Vec<u32>,
        fragmentation: bool,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        mz_noise_fragment: bool,
        fragment_noise_ppm: f64,
        right_drag: bool,
    ) -> Vec<TimsFrame> {
        if frame_ids.is_empty() {
            return Vec::new();
        }

        // Determine frame range
        let frame_min = *frame_ids.iter().min().unwrap();
        let frame_max = *frame_ids.iter().max().unwrap();

        // Load data for this frame range
        let (peptides, ions, fragment_ions) = match self.load_data_for_frame_range(frame_min, frame_max) {
            Ok(data) => data,
            Err(_) => return Vec::new(),
        };

        // Build lookup maps
        let peptide_map = TimsTofSyntheticsDataHandle::build_peptide_map(&peptides);
        let peptide_to_ions = TimsTofSyntheticsDataHandle::build_peptide_to_ions(&ions);
        let frame_to_abundances = TimsTofSyntheticsDataHandle::build_frame_to_abundances(&peptides);
        let peptide_to_events = TimsTofSyntheticsDataHandle::build_peptide_to_events(&peptides);

        // Build fragment ions map if fragmentation is enabled
        let fragment_ions_map = if fragmentation {
            Some(TimsTofSyntheticsDataHandle::build_fragment_ions(
                &peptide_map,
                &fragment_ions,
                self.num_threads,
            ))
        } else {
            None
        };

        // Build frames in parallel using indexed iteration to maintain order
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            let mut tims_frames: Vec<TimsFrame> = Vec::with_capacity(frame_ids.len());
            unsafe { tims_frames.set_len(frame_ids.len()); }

            frame_ids.par_iter().enumerate().for_each(|(idx, frame_id)| {
                let frame = self.build_single_frame(
                    *frame_id,
                    fragmentation,
                    mz_noise_precursor,
                    uniform,
                    precursor_noise_ppm,
                    mz_noise_fragment,
                    fragment_noise_ppm,
                    right_drag,
                    &peptide_map,
                    &peptide_to_ions,
                    &frame_to_abundances,
                    &peptide_to_events,
                    &fragment_ions_map,
                );
                unsafe {
                    let ptr = tims_frames.as_ptr() as *mut TimsFrame;
                    std::ptr::write(ptr.add(idx), frame);
                }
            });

            tims_frames
        })
    }

    /// Build a single frame with provided data maps.
    #[allow(clippy::too_many_arguments)]
    fn build_single_frame(
        &self,
        frame_id: u32,
        fragmentation: bool,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        mz_noise_fragment: bool,
        fragment_noise_ppm: f64,
        right_drag: bool,
        _peptide_map: &BTreeMap<u32, PeptidesSim>,
        peptide_to_ions: &BTreeMap<u32, (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<i8>, Vec<MzSpectrum>)>,
        frame_to_abundances: &BTreeMap<u32, (Vec<u32>, Vec<f32>)>,
        peptide_to_events: &BTreeMap<u32, f32>,
        fragment_ions_map: &Option<BTreeMap<(u32, i8, i32), (PeptideProductIonSeriesCollection, Vec<MzSpectrum>)>>,
    ) -> TimsFrame {
        // Determine if this is a precursor or fragment frame
        let is_precursor = self.precursor_frame_id_set.contains(&frame_id);

        if is_precursor {
            self.build_precursor_frame(
                frame_id,
                mz_noise_precursor,
                uniform,
                precursor_noise_ppm,
                right_drag,
                peptide_to_ions,
                frame_to_abundances,
                peptide_to_events,
            )
        } else {
            self.build_fragment_frame(
                frame_id,
                fragmentation,
                mz_noise_fragment,
                uniform,
                fragment_noise_ppm,
                right_drag,
                peptide_to_ions,
                frame_to_abundances,
                peptide_to_events,
                fragment_ions_map,
            )
        }
    }

    /// Build a precursor (MS1) frame.
    #[allow(clippy::too_many_arguments)]
    fn build_precursor_frame(
        &self,
        frame_id: u32,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        right_drag: bool,
        peptide_to_ions: &BTreeMap<u32, (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<i8>, Vec<MzSpectrum>)>,
        frame_to_abundances: &BTreeMap<u32, (Vec<u32>, Vec<f32>)>,
        peptide_to_events: &BTreeMap<u32, f32>,
    ) -> TimsFrame {
        let ms_type = MsType::Precursor;
        let rt = *self.frame_to_rt.get(&frame_id).unwrap_or(&0.0) as f64;

        // Single lookup instead of contains_key + get
        let Some((peptide_ids, abundances)) = frame_to_abundances.get(&frame_id) else {
            return TimsFrame::new(frame_id as i32, ms_type, rt, vec![], vec![], vec![], vec![], vec![]);
        };

        // Preallocate with estimated capacity
        let estimated_capacity = peptide_ids.len() * 4;
        let mut tims_spectra: Vec<TimsSpectrum> = Vec::with_capacity(estimated_capacity);

        for (peptide_id, abundance) in peptide_ids.iter().zip(abundances.iter()) {
            let Some((ion_abundances, scan_occurrences, scan_abundances, _, spectra)) =
                peptide_to_ions.get(peptide_id)
            else {
                continue;
            };

            // Cache peptide-level lookup
            let total_events = *peptide_to_events.get(peptide_id).unwrap_or(&1.0);

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let scan_occurrence = &scan_occurrences[index];
                let scan_abundance = &scan_abundances[index];
                let spectrum = &spectra[index];

                for (scan, scan_abu) in scan_occurrence.iter().zip(scan_abundance.iter()) {
                    let abundance_factor = abundance * ion_abundance * scan_abu * total_events;
                    let scaled_spec: MzSpectrum = spectrum.clone() * abundance_factor as f64;

                    let mz_spectrum = if mz_noise_precursor {
                        if uniform {
                            scaled_spec.add_mz_noise_uniform(precursor_noise_ppm, right_drag)
                        } else {
                            scaled_spec.add_mz_noise_normal(precursor_noise_ppm)
                        }
                    } else {
                        scaled_spec
                    };

                    let scan_mobility = *self.scan_to_mobility.get(scan).unwrap_or(&0.0) as f64;
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
                        ),
                    ));
                }
            }
        }

        let mut filtered = TimsFrame::from_tims_spectra_filtered(
            tims_spectra, 0.0, 10000.0, 0, 2000, 0.0, 10.0, 1.0, 1e9,
        );

        // Round intensities
        let intensities_rounded: Vec<f64> = filtered
            .ims_frame
            .intensity
            .iter()
            .map(|x| x.round())
            .collect();
        filtered.ims_frame.intensity = Arc::new(intensities_rounded);

        filtered
    }

    /// Build a fragment (MS2) frame.
    #[allow(clippy::too_many_arguments)]
    fn build_fragment_frame(
        &self,
        frame_id: u32,
        fragmentation: bool,
        mz_noise_fragment: bool,
        uniform: bool,
        fragment_noise_ppm: f64,
        right_drag: bool,
        peptide_to_ions: &BTreeMap<u32, (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<i8>, Vec<MzSpectrum>)>,
        frame_to_abundances: &BTreeMap<u32, (Vec<u32>, Vec<f32>)>,
        peptide_to_events: &BTreeMap<u32, f32>,
        fragment_ions_map: &Option<BTreeMap<(u32, i8, i32), (PeptideProductIonSeriesCollection, Vec<MzSpectrum>)>>,
    ) -> TimsFrame {
        let ms_type = MsType::FragmentDia;
        let rt = *self.frame_to_rt.get(&frame_id).unwrap_or(&0.0) as f64;

        if !fragmentation || fragment_ions_map.is_none() {
            // If no fragmentation, build a quadrupole-filtered precursor frame
            let precursor_frame = self.build_precursor_frame(
                frame_id,
                mz_noise_fragment,
                uniform,
                fragment_noise_ppm,
                right_drag,
                peptide_to_ions,
                frame_to_abundances,
                peptide_to_events,
            );
            let mut frame = self.transmission_settings.transmit_tims_frame(&precursor_frame, None);
            frame.ms_type = MsType::FragmentDia;
            return frame;
        }

        let fragment_ions = fragment_ions_map.as_ref().unwrap();

        // Single lookup instead of contains_key + get
        let Some((peptide_ids, frame_abundances)) = frame_to_abundances.get(&frame_id) else {
            return TimsFrame::new(frame_id as i32, ms_type, rt, vec![], vec![], vec![], vec![], vec![]);
        };

        // Preallocate with estimated capacity
        let estimated_capacity = peptide_ids.len() * 4;
        let mut tims_spectra: Vec<TimsSpectrum> = Vec::with_capacity(estimated_capacity);

        for (peptide_id, frame_abundance) in peptide_ids.iter().zip(frame_abundances.iter()) {
            let Some((ion_abundances, scan_occurrences, scan_abundances, charges, spectra)) =
                peptide_to_ions.get(peptide_id)
            else {
                continue;
            };

            // Cache peptide-level lookup
            let total_events = *peptide_to_events.get(peptide_id).unwrap_or(&1.0);

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let all_scan_occurrence = &scan_occurrences[index];
                let all_scan_abundance = &scan_abundances[index];
                let spectrum = &spectra[index];
                let charge_state = charges[index];

                for (scan, scan_abundance) in all_scan_occurrence.iter().zip(all_scan_abundance.iter()) {
                    // Check if precursor is transmitted
                    if !self.transmission_settings.any_transmitted(
                        frame_id as i32,
                        *scan as i32,
                        &spectrum.mz,
                        None,
                    ) {
                        continue;
                    }

                    // Calculate abundance factor
                    let fraction_events = frame_abundance * scan_abundance * ion_abundance * total_events;

                    // Get collision energy
                    let collision_energy = self.fragmentation_settings.get_collision_energy(
                        frame_id as i32,
                        *scan as i32,
                    );
                    let collision_energy_quantized = (collision_energy * 1e1).round() as i32;

                    // Single lookup with let-else
                    let Some((_, fragment_series_vec)) = fragment_ions.get(&(*peptide_id, charge_state, collision_energy_quantized)) else {
                        continue;
                    };

                    // Cache scan mobility
                    let scan_mobility = *self.scan_to_mobility.get(scan).unwrap_or(&0.0) as f64;

                    for fragment_ion_series in fragment_series_vec.iter() {
                        let scaled_spec = fragment_ion_series.clone() * fraction_events as f64;

                        let mz_spectrum = if mz_noise_fragment {
                            if uniform {
                                scaled_spec.add_mz_noise_uniform(fragment_noise_ppm, right_drag)
                            } else {
                                scaled_spec.add_mz_noise_normal(fragment_noise_ppm)
                            }
                        } else {
                            scaled_spec
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
        }

        if tims_spectra.is_empty() {
            return TimsFrame::new(frame_id as i32, ms_type, rt, vec![], vec![], vec![], vec![], vec![]);
        }

        let mut filtered = TimsFrame::from_tims_spectra_filtered(
            tims_spectra, 100.0, 1700.0, 0, 1000, 0.0, 10.0, 1.0, 1e9,
        );

        // Round intensities
        let intensities_rounded: Vec<f64> = filtered
            .ims_frame
            .intensity
            .iter()
            .map(|x| x.round())
            .collect();
        filtered.ims_frame.intensity = Arc::new(intensities_rounded);

        filtered
    }

    /// Get the total number of frames.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Get all frame IDs.
    pub fn frame_ids(&self) -> Vec<u32> {
        self.frames.iter().map(|f| f.frame_id).collect()
    }

    /// Get precursor frame IDs.
    pub fn precursor_frame_ids(&self) -> Vec<u32> {
        self.precursor_frame_id_set.iter().cloned().collect()
    }

    /// Get fragment frame IDs.
    pub fn fragment_frame_ids(&self) -> Vec<u32> {
        self.frames
            .iter()
            .filter(|f| !self.precursor_frame_id_set.contains(&f.frame_id))
            .map(|f| f.frame_id)
            .collect()
    }
}

impl TimsTofCollisionEnergy for TimsTofLazyFrameBuilderDIA {
    fn get_collision_energy(&self, frame_id: i32, scan_id: i32) -> f64 {
        self.fragmentation_settings.get_collision_energy(frame_id, scan_id)
    }
}

/// A lazy frame builder for DDA experiments that only loads data as needed.
///
/// Unlike `TimsTofSyntheticsFrameBuilderDDA`, this struct does not load all peptides,
/// ions, and fragment ions into memory at construction time. Instead, it stores only
/// the static metadata (frame info, scan info, transmission settings) and loads
/// peptide/ion data on-demand for each batch of frames being built.
///
/// This can significantly reduce memory usage for large simulations.
pub struct TimsTofLazyFrameBuilderDDA {
    /// Path to the SQLite database
    pub db_path: String,
    /// Frame metadata (id, time, ms_type)
    pub frames: Vec<FramesSim>,
    /// Scan metadata (scan_id, mobility)
    pub scans: Vec<ScansSim>,
    /// Set of precursor frame IDs for quick lookup
    pub precursor_frame_id_set: HashSet<u32>,
    /// Map from frame_id to retention_time
    pub frame_to_rt: BTreeMap<u32, f32>,
    /// Map from scan_id to mobility
    pub scan_to_mobility: BTreeMap<u32, f32>,
    /// DDA transmission settings (includes PASEF metadata with collision energies)
    pub transmission_settings: TimsTransmissionDDA,
    /// Number of threads for parallel processing
    pub num_threads: usize,
}

impl TimsTofLazyFrameBuilderDDA {
    /// Create a new lazy frame builder for DDA.
    ///
    /// Only loads static metadata (frames, scans, transmission settings).
    /// Peptides, ions, and fragment ions are NOT loaded here.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the SQLite database
    /// * `num_threads` - Number of threads for parallel operations
    ///
    /// # Returns
    ///
    /// Result containing the lazy frame builder
    pub fn new(path: &Path, num_threads: usize) -> rusqlite::Result<Self> {
        let handle = TimsTofSyntheticsDataHandle::new(path)?;

        let frames = handle.read_frames()?;
        let scans = handle.read_scans()?;

        let precursor_frame_id_set = TimsTofSyntheticsDataHandle::build_precursor_frame_id_set(&frames);
        let frame_to_rt = TimsTofSyntheticsDataHandle::build_frame_to_rt(&frames);
        let scan_to_mobility = TimsTofSyntheticsDataHandle::build_scan_to_mobility(&scans);

        let transmission_settings = handle.get_transmission_dda();

        Ok(Self {
            db_path: path.to_str().unwrap().to_string(),
            frames,
            scans,
            precursor_frame_id_set,
            frame_to_rt,
            scan_to_mobility,
            transmission_settings,
            num_threads,
        })
    }

    /// Load data for a specific frame range from the database.
    ///
    /// Returns peptides, ions, and fragment ions that are relevant to the frame range.
    fn load_data_for_frame_range(
        &self,
        frame_min: u32,
        frame_max: u32,
    ) -> rusqlite::Result<(Vec<PeptidesSim>, Vec<IonSim>, Vec<FragmentIonSim>)> {
        let path = Path::new(&self.db_path);
        let handle = TimsTofSyntheticsDataHandle::new(path)?;

        // Load only peptides for this frame range
        let peptides = handle.read_peptides_for_frame_range(frame_min, frame_max)?;

        if peptides.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        // Get peptide IDs for querying related data
        let peptide_ids: Vec<u32> = peptides.iter().map(|p| p.peptide_id).collect();

        // Load ions and fragment ions for these peptides
        let ions = handle.read_ions_for_peptides(&peptide_ids)?;
        let fragment_ions = handle.read_fragment_ions_for_peptides(&peptide_ids)?;

        Ok((peptides, ions, fragment_ions))
    }

    /// Build frames for a range of frame IDs.
    ///
    /// This method loads only the data needed for the specified frames,
    /// builds the frames, and then releases the loaded data.
    ///
    /// # Arguments
    ///
    /// * `frame_ids` - Vector of frame IDs to build
    /// * `fragmentation` - Whether to include fragmentation
    /// * `mz_noise_precursor` - Whether to add m/z noise to precursor ions
    /// * `uniform` - Whether to use uniform noise distribution
    /// * `precursor_noise_ppm` - Precursor noise in ppm
    /// * `mz_noise_fragment` - Whether to add m/z noise to fragment ions
    /// * `fragment_noise_ppm` - Fragment noise in ppm
    /// * `right_drag` - Whether to use right drag for noise
    ///
    /// # Returns
    ///
    /// Vector of built TimsFrame instances
    pub fn build_frames_lazy(
        &self,
        frame_ids: Vec<u32>,
        fragmentation: bool,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        mz_noise_fragment: bool,
        fragment_noise_ppm: f64,
        right_drag: bool,
    ) -> Vec<TimsFrame> {
        if frame_ids.is_empty() {
            return Vec::new();
        }

        // Determine frame range
        let frame_min = *frame_ids.iter().min().unwrap();
        let frame_max = *frame_ids.iter().max().unwrap();

        // Load data for this frame range
        let (peptides, ions, fragment_ions) = match self.load_data_for_frame_range(frame_min, frame_max) {
            Ok(data) => data,
            Err(_) => return Vec::new(),
        };

        // Build lookup maps
        let peptide_map = TimsTofSyntheticsDataHandle::build_peptide_map(&peptides);
        let peptide_to_ions = TimsTofSyntheticsDataHandle::build_peptide_to_ions(&ions);
        let frame_to_abundances = TimsTofSyntheticsDataHandle::build_frame_to_abundances(&peptides);
        let peptide_to_events = TimsTofSyntheticsDataHandle::build_peptide_to_events(&peptides);

        // Build fragment ions map if fragmentation is enabled
        let fragment_ions_map = if fragmentation {
            Some(TimsTofSyntheticsDataHandle::build_fragment_ions(
                &peptide_map,
                &fragment_ions,
                self.num_threads,
            ))
        } else {
            None
        };

        // Build frames in parallel using indexed iteration to maintain order
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            let mut tims_frames: Vec<TimsFrame> = Vec::with_capacity(frame_ids.len());
            unsafe { tims_frames.set_len(frame_ids.len()); }

            frame_ids.par_iter().enumerate().for_each(|(idx, frame_id)| {
                let frame = self.build_single_frame(
                    *frame_id,
                    fragmentation,
                    mz_noise_precursor,
                    uniform,
                    precursor_noise_ppm,
                    mz_noise_fragment,
                    fragment_noise_ppm,
                    right_drag,
                    &peptide_to_ions,
                    &frame_to_abundances,
                    &peptide_to_events,
                    &fragment_ions_map,
                );
                unsafe {
                    let ptr = tims_frames.as_ptr() as *mut TimsFrame;
                    std::ptr::write(ptr.add(idx), frame);
                }
            });

            tims_frames
        })
    }

    /// Build a single frame with provided data maps.
    #[allow(clippy::too_many_arguments)]
    fn build_single_frame(
        &self,
        frame_id: u32,
        fragmentation: bool,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        mz_noise_fragment: bool,
        fragment_noise_ppm: f64,
        right_drag: bool,
        peptide_to_ions: &BTreeMap<u32, (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<i8>, Vec<MzSpectrum>)>,
        frame_to_abundances: &BTreeMap<u32, (Vec<u32>, Vec<f32>)>,
        peptide_to_events: &BTreeMap<u32, f32>,
        fragment_ions_map: &Option<BTreeMap<(u32, i8, i32), (PeptideProductIonSeriesCollection, Vec<MzSpectrum>)>>,
    ) -> TimsFrame {
        // Determine if this is a precursor or fragment frame
        let is_precursor = self.precursor_frame_id_set.contains(&frame_id);

        if is_precursor {
            self.build_precursor_frame(
                frame_id,
                mz_noise_precursor,
                uniform,
                precursor_noise_ppm,
                right_drag,
                peptide_to_ions,
                frame_to_abundances,
                peptide_to_events,
            )
        } else {
            self.build_fragment_frame(
                frame_id,
                fragmentation,
                mz_noise_fragment,
                uniform,
                fragment_noise_ppm,
                right_drag,
                peptide_to_ions,
                frame_to_abundances,
                peptide_to_events,
                fragment_ions_map,
            )
        }
    }

    /// Build a precursor (MS1) frame.
    #[allow(clippy::too_many_arguments)]
    fn build_precursor_frame(
        &self,
        frame_id: u32,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        right_drag: bool,
        peptide_to_ions: &BTreeMap<u32, (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<i8>, Vec<MzSpectrum>)>,
        frame_to_abundances: &BTreeMap<u32, (Vec<u32>, Vec<f32>)>,
        peptide_to_events: &BTreeMap<u32, f32>,
    ) -> TimsFrame {
        let ms_type = MsType::Precursor;
        let rt = *self.frame_to_rt.get(&frame_id).unwrap_or(&0.0) as f64;

        // Single lookup instead of contains_key + get
        let Some((peptide_ids, abundances)) = frame_to_abundances.get(&frame_id) else {
            return TimsFrame::new(frame_id as i32, ms_type, rt, vec![], vec![], vec![], vec![], vec![]);
        };

        // Preallocate with estimated capacity
        let estimated_capacity = peptide_ids.len() * 4;
        let mut tims_spectra: Vec<TimsSpectrum> = Vec::with_capacity(estimated_capacity);

        for (peptide_id, abundance) in peptide_ids.iter().zip(abundances.iter()) {
            let Some((ion_abundances, scan_occurrences, scan_abundances, _, spectra)) =
                peptide_to_ions.get(peptide_id)
            else {
                continue;
            };

            // Cache peptide-level lookup
            let total_events = *peptide_to_events.get(peptide_id).unwrap_or(&1.0);

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let scan_occurrence = &scan_occurrences[index];
                let scan_abundance = &scan_abundances[index];
                let spectrum = &spectra[index];

                for (scan, scan_abu) in scan_occurrence.iter().zip(scan_abundance.iter()) {
                    let abundance_factor = abundance * ion_abundance * scan_abu * total_events;
                    let scaled_spec: MzSpectrum = spectrum.clone() * abundance_factor as f64;

                    let mz_spectrum = if mz_noise_precursor {
                        if uniform {
                            scaled_spec.add_mz_noise_uniform(precursor_noise_ppm, right_drag)
                        } else {
                            scaled_spec.add_mz_noise_normal(precursor_noise_ppm)
                        }
                    } else {
                        scaled_spec
                    };

                    let scan_mobility = *self.scan_to_mobility.get(scan).unwrap_or(&0.0) as f64;
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
                        ),
                    ));
                }
            }
        }

        let mut filtered = TimsFrame::from_tims_spectra_filtered(
            tims_spectra, 0.0, 10000.0, 0, 2000, 0.0, 10.0, 1.0, 1e9,
        );

        // Round intensities
        let intensities_rounded: Vec<f64> = filtered
            .ims_frame
            .intensity
            .iter()
            .map(|x| x.round())
            .collect();
        filtered.ims_frame.intensity = Arc::new(intensities_rounded);

        filtered
    }

    /// Build a fragment (MS2) frame for DDA.
    #[allow(clippy::too_many_arguments)]
    fn build_fragment_frame(
        &self,
        frame_id: u32,
        fragmentation: bool,
        mz_noise_fragment: bool,
        uniform: bool,
        fragment_noise_ppm: f64,
        right_drag: bool,
        peptide_to_ions: &BTreeMap<u32, (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<i8>, Vec<MzSpectrum>)>,
        frame_to_abundances: &BTreeMap<u32, (Vec<u32>, Vec<f32>)>,
        peptide_to_events: &BTreeMap<u32, f32>,
        fragment_ions_map: &Option<BTreeMap<(u32, i8, i32), (PeptideProductIonSeriesCollection, Vec<MzSpectrum>)>>,
    ) -> TimsFrame {
        let ms_type = MsType::FragmentDda;
        let rt = *self.frame_to_rt.get(&frame_id).unwrap_or(&0.0) as f64;

        // Cache PASEF meta lookup for this frame
        let pasef_meta = self.transmission_settings.pasef_meta.get(&(frame_id as i32));

        // If no PASEF meta for this frame, return empty or filtered precursor
        if pasef_meta.is_none() {
            if !fragmentation || fragment_ions_map.is_none() {
                // Build quadrupole-filtered precursor frame
                let precursor_frame = self.build_precursor_frame(
                    frame_id,
                    mz_noise_fragment,
                    uniform,
                    fragment_noise_ppm,
                    right_drag,
                    peptide_to_ions,
                    frame_to_abundances,
                    peptide_to_events,
                );
                let mut frame = self.transmission_settings.transmit_tims_frame(&precursor_frame, None);
                frame.ms_type = MsType::FragmentDda;
                return frame;
            }
            return TimsFrame::new(frame_id as i32, ms_type, rt, vec![], vec![], vec![], vec![], vec![]);
        }
        let pasef_meta = pasef_meta.unwrap();

        if !fragmentation || fragment_ions_map.is_none() {
            // Build quadrupole-filtered precursor frame
            let precursor_frame = self.build_precursor_frame(
                frame_id,
                mz_noise_fragment,
                uniform,
                fragment_noise_ppm,
                right_drag,
                peptide_to_ions,
                frame_to_abundances,
                peptide_to_events,
            );
            let mut frame = self.transmission_settings.transmit_tims_frame(&precursor_frame, None);
            frame.ms_type = MsType::FragmentDda;
            return frame;
        }

        let fragment_ions = fragment_ions_map.as_ref().unwrap();

        // Single lookup for frame abundances
        let Some((peptide_ids, frame_abundances)) = frame_to_abundances.get(&frame_id) else {
            return TimsFrame::new(frame_id as i32, ms_type, rt, vec![], vec![], vec![], vec![], vec![]);
        };

        // Preallocate with estimated capacity
        let estimated_capacity = peptide_ids.len() * 4;
        let mut tims_spectra: Vec<TimsSpectrum> = Vec::with_capacity(estimated_capacity);

        for (peptide_id, frame_abundance) in peptide_ids.iter().zip(frame_abundances.iter()) {
            let Some((ion_abundances, scan_occurrences, scan_abundances, charges, spectra)) =
                peptide_to_ions.get(peptide_id)
            else {
                continue;
            };

            // Cache peptide-level lookup
            let total_events = *peptide_to_events.get(peptide_id).unwrap_or(&1.0);

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let all_scan_occurrence = &scan_occurrences[index];
                let all_scan_abundance = &scan_abundances[index];
                let spectrum = &spectra[index];
                let charge_state = charges[index];

                for (scan, scan_abundance) in all_scan_occurrence.iter().zip(all_scan_abundance.iter()) {
                    // Check if precursor is transmitted
                    if !self.transmission_settings.any_transmitted(
                        frame_id as i32,
                        *scan as i32,
                        &spectrum.mz,
                        None,
                    ) {
                        continue;
                    }

                    // Calculate abundance factor
                    let fraction_events = frame_abundance * scan_abundance * ion_abundance * total_events;

                    // Get collision energy from PASEF meta
                    let collision_energy: f64 = pasef_meta
                        .iter()
                        .find(|scan_meta| scan_meta.scan_start <= *scan as i32 && scan_meta.scan_end >= *scan as i32)
                        .map(|s| s.collision_energy)
                        .unwrap_or(0.0);
                    let collision_energy_quantized = (collision_energy * 1e1).round() as i32;

                    // Get fragment ions for this peptide/charge/energy combination
                    let Some((_, fragment_series_vec)) = fragment_ions.get(&(*peptide_id, charge_state, collision_energy_quantized)) else {
                        continue;
                    };

                    // Cache scan mobility
                    let scan_mobility = *self.scan_to_mobility.get(scan).unwrap_or(&0.0) as f64;

                    for fragment_ion_series in fragment_series_vec.iter() {
                        let scaled_spec = fragment_ion_series.clone() * fraction_events as f64;

                        let mz_spectrum = if mz_noise_fragment {
                            if uniform {
                                scaled_spec.add_mz_noise_uniform(fragment_noise_ppm, right_drag)
                            } else {
                                scaled_spec.add_mz_noise_normal(fragment_noise_ppm)
                            }
                        } else {
                            scaled_spec
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
        }

        if tims_spectra.is_empty() {
            return TimsFrame::new(frame_id as i32, ms_type, rt, vec![], vec![], vec![], vec![], vec![]);
        }

        let mut filtered = TimsFrame::from_tims_spectra_filtered(
            tims_spectra, 100.0, 1700.0, 0, 1000, 0.0, 10.0, 1.0, 1e9,
        );

        // Round intensities
        let intensities_rounded: Vec<f64> = filtered
            .ims_frame
            .intensity
            .iter()
            .map(|x| x.round())
            .collect();
        filtered.ims_frame.intensity = Arc::new(intensities_rounded);

        filtered
    }

    /// Get collision energy for a frame/scan combination from PASEF metadata.
    pub fn get_collision_energy(&self, frame_id: i32, scan_id: i32) -> f64 {
        self.transmission_settings.get_collision_energy(frame_id, scan_id).unwrap_or(0.0)
    }

    /// Get the total number of frames.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Get all frame IDs.
    pub fn frame_ids(&self) -> Vec<u32> {
        self.frames.iter().map(|f| f.frame_id).collect()
    }

    /// Get precursor frame IDs.
    pub fn precursor_frame_ids(&self) -> Vec<u32> {
        self.precursor_frame_id_set.iter().cloned().collect()
    }

    /// Get fragment frame IDs.
    pub fn fragment_frame_ids(&self) -> Vec<u32> {
        self.frames
            .iter()
            .filter(|f| !self.precursor_frame_id_set.contains(&f.frame_id))
            .map(|f| f.frame_id)
            .collect()
    }
}
