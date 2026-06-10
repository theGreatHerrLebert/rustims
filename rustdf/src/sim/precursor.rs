use mscore::data::peptide::PeptideIon;
use mscore::data::spectrum::{IndexedMzSpectrum, MsType, MzSpectrum};
use mscore::simulation::annotation::{
    MzSpectrumAnnotated, TimsFrameAnnotated, TimsSpectrumAnnotated,
};
use mscore::timstof::frame::TimsFrame;
use mscore::timstof::spectrum::TimsSpectrum;
use rusqlite::Result;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use crate::sim::containers::{FramesSim, IonSim, PeptidesSim, ScansSim};
use crate::sim::handle::TimsTofSyntheticsDataHandle;
use rayon::prelude::*;

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
    /// Mapping from ion_id to (peptide_id, charge) for DDA precursor lookup
    pub ion_id_to_peptide_charge: BTreeMap<u32, (u32, i8)>,
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
        Self::from_source(path, &crate::sim::projector::DistributionSource::Columns)
    }

    /// Construct the builder with occurrence/abundance distributions sourced from
    /// `source` (P4): `Columns` (legacy JSON, the default — byte-unchanged) or the
    /// render-time `Projector`. Identical-shape `PeptidesSim`/`IonSim` either way,
    /// so the maps + all downstream consumers are untouched.
    pub fn from_source(
        path: &Path,
        source: &crate::sim::projector::DistributionSource,
    ) -> Result<Self> {
        let handle = TimsTofSyntheticsDataHandle::new(path)?;
        let ions = handle.read_ions_with_source(source)?;
        let peptides = handle.read_peptides_with_source(source)?;
        let scans = handle.read_scans()?;
        let frames = handle.read_frames()?;

        Ok(Self::from_entities(ions, peptides, scans, frames))
    }

    /// Construct the builder from already-loaded, in-memory entities instead of
    /// reading the database. Used by the lazy builders, which load only a
    /// per-batch slice of peptides/ions (the `scans`/`frames` metadata is full).
    /// All lookup maps are derived identically to `from_source`, so a builder
    /// constructed here behaves exactly like an eager one restricted to the
    /// supplied entities.
    pub fn from_entities(
        ions: Vec<IonSim>,
        peptides: Vec<PeptidesSim>,
        scans: Vec<ScansSim>,
        frames: Vec<FramesSim>,
    ) -> Self {
        // Build ion_id to (peptide_id, charge) mapping for DDA precursor lookup
        let mut ion_id_to_peptide_charge: BTreeMap<u32, (u32, i8)> = BTreeMap::new();
        for ion in &ions {
            ion_id_to_peptide_charge.insert(ion.ion_id, (ion.peptide_id, ion.charge));
        }

        Self {
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
            ion_id_to_peptide_charge,
        }
    }

    /// Vendor-neutral MS1 spectral-contribution kernel (P6a): the ordered
    /// `(scan, scaled_spectrum)` contributions for a precursor frame — for each
    /// (peptide, ion, scan) the isotope spectrum scaled by its abundance factor,
    /// PRE m/z-noise, in builder order (peptide → ion → scan). Bruker aggregates
    /// these per scan into a `TimsFrame` (see `build_precursor_frame`); a non-IMS
    /// instrument sums them into a single MS1 `Scan`. Factoring this out lets both
    /// vendors share the exact same physics without duplicating it.
    pub fn precursor_frame_contributions(&self, frame_id: u32) -> Vec<(i32, MzSpectrum)> {
        let mut out: Vec<(i32, MzSpectrum)> = Vec::new();
        let Some((peptide_ids, abundances)) = self.frame_to_abundances.get(&frame_id) else {
            return out;
        };
        out.reserve(peptide_ids.len() * 4);
        for (peptide_id, abundance) in peptide_ids.iter().zip(abundances.iter()) {
            let Some((ion_abundances, scan_occurrences, scan_abundances, _, spectra)) =
                self.peptide_to_ions.get(peptide_id)
            else {
                continue;
            };
            let total_events = *self.peptide_to_events.get(peptide_id).unwrap();
            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let scan_occurrence = &scan_occurrences[index];
                let scan_abundance = &scan_abundances[index];
                let spectrum = &spectra[index];
                for (scan, scan_abu) in scan_occurrence.iter().zip(scan_abundance.iter()) {
                    let abundance_factor = abundance * ion_abundance * scan_abu * total_events;
                    out.push((*scan as i32, spectrum.clone() * abundance_factor as f64));
                }
            }
        }
        out
    }

    /// Non-IMS (e.g. Astral) MS1 consumer of the contribution kernel (P6a): the
    /// mobility axis collapses, so all `(scan, scaled_spectrum)` contributions for
    /// the frame sum into ONE spectrum — the single MS1 spectrum a no-mobility
    /// instrument records. The scan coordinate is discarded; peaks at the same m/z
    /// merge (`from_collection`'s deterministic m/z binning). This is the second
    /// consumer the Bruker `TimsFrame` adapter shares the kernel with.
    pub fn precursor_scan_spectrum(&self, frame_id: u32) -> MzSpectrum {
        let specs: Vec<MzSpectrum> = self
            .precursor_frame_contributions(frame_id)
            .into_iter()
            .map(|(_scan, spectrum)| spectrum)
            .collect();
        MzSpectrum::from_collection(specs)
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
        // Cache frame-level lookups
        let ms_type = if self.precursor_frame_id_set.contains(&frame_id) {
            MsType::Precursor
        } else {
            MsType::Unknown
        };
        let rt = *self.frame_to_rt.get(&frame_id).unwrap() as f64;

        // Bruker adapter over the vendor-neutral contribution kernel (P6a):
        // aggregate the ordered (scan, scaled_spectrum) contributions per scan
        // into a TimsFrame. Behaviour is identical to the previous inline loop —
        // same order, same per-contribution m/z-noise, same TimsSpectrum build.
        let contributions = self.precursor_frame_contributions(frame_id);
        let mut tims_spectra: Vec<TimsSpectrum> = Vec::with_capacity(contributions.len());
        for (scan, scaled_spec) in contributions {
            let mz_spectrum = if mz_noise_precursor {
                if uniform {
                    scaled_spec.add_mz_noise_uniform(precursor_noise_ppm, right_drag)
                } else {
                    scaled_spec.add_mz_noise_normal(precursor_noise_ppm)
                }
            } else {
                scaled_spec
            };

            let scan_mobility = *self.scan_to_mobility.get(&(scan as u32)).unwrap() as f64;
            let spectrum_len = mz_spectrum.mz.len();

            tims_spectra.push(TimsSpectrum::new(
                frame_id as i32,
                scan,
                rt,
                scan_mobility,
                ms_type.clone(),
                IndexedMzSpectrum::from_mz_spectrum(vec![0; spectrum_len], mz_spectrum),
            ));
        }

        // A precursor frame can have peptides assigned (passed the
        // frame_to_abundances guard above) yet produce NO spectra once every ion
        // is filtered out. `TimsFrame::from_tims_spectra([])` would then fall back
        // to frame_id=1 / MsType::Unknown / rt=0.0, emitting a ghost frame that
        // collides with the real frame 1 (the writer's Frames.Id uniqueness guard
        // rejects it). Preserve this frame's own id + ms_type for the empty case,
        // mirroring build_fragment_frame.
        if tims_spectra.is_empty() {
            return TimsFrame::new(frame_id as i32, ms_type, rt, vec![], vec![], vec![], vec![], vec![]);
        }
        let tims_frame = TimsFrame::from_tims_spectra(tims_spectra);
        tims_frame.filter_ranged(0.0, 10000.0, 0, 2000, 0.0, 10.0, 1.0, 1e9, 0, i32::MAX)
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
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            // Use indexed parallel iteration to maintain order, avoiding post-sort
            let mut tims_frames: Vec<TimsFrame> = Vec::with_capacity(frame_ids.len());
            unsafe { tims_frames.set_len(frame_ids.len()); }

            frame_ids.par_iter().enumerate().for_each(|(idx, frame_id)| {
                let frame = self.build_precursor_frame(
                    *frame_id,
                    mz_noise_precursor,
                    uniform,
                    precursor_noise_ppm,
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

    pub fn build_precursor_frame_annotated(
        &self,
        frame_id: u32,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        right_drag: bool,
    ) -> TimsFrameAnnotated {
        // Cache frame-level lookups
        let ms_type = if self.precursor_frame_id_set.contains(&frame_id) {
            MsType::Precursor
        } else {
            MsType::Unknown
        };
        let rt = *self.frame_to_rt.get(&frame_id).unwrap_or(&0.0) as f64;

        // Single lookup instead of contains_key + get
        let Some((peptide_ids, abundances)) = self.frame_to_abundances.get(&frame_id) else {
            return TimsFrameAnnotated::new(frame_id as i32, rt, ms_type, vec![], vec![], vec![], vec![], vec![], vec![]);
        };

        // Preallocate with estimated capacity
        let estimated_capacity = peptide_ids.len() * 4;
        let mut tims_spectra: Vec<TimsSpectrumAnnotated> = Vec::with_capacity(estimated_capacity);

        for (peptide_id, abundance) in peptide_ids.iter().zip(abundances.iter()) {
            // Single lookup
            let Some((ion_abundances, scan_occurrences, scan_abundances, charges, _)) =
                self.peptide_to_ions.get(peptide_id)
            else {
                continue;
            };

            // Cache peptide-level lookups
            let total_events = *self.peptide_to_events.get(peptide_id).unwrap();
            let peptide = self.peptides.get(peptide_id).unwrap();

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let scan_occurrence = &scan_occurrences[index];
                let scan_abundance = &scan_abundances[index];
                let charge = charges[index];

                let ion = PeptideIon::new(
                    peptide.sequence.sequence.clone(),
                    charge as i32,
                    *ion_abundance as f64,
                    Some(*peptide_id as i32),
                );
                // TODO: make this configurable
                let spectrum = ion.calculate_isotopic_spectrum_annotated(1e-3, 1e-8, 200, 1e-4);

                for (scan, scan_abu) in scan_occurrence.iter().zip(scan_abundance.iter()) {
                    let abundance_factor = abundance * ion_abundance * scan_abu * total_events;
                    let scaled_spec: MzSpectrumAnnotated = spectrum.clone() * abundance_factor as f64;

                    let mz_spectrum = if mz_noise_precursor {
                        if uniform {
                            scaled_spec.add_mz_noise_uniform(precursor_noise_ppm, right_drag)
                        } else {
                            scaled_spec.add_mz_noise_normal(precursor_noise_ppm)
                        }
                    } else {
                        scaled_spec
                    };

                    // Cache scan mobility
                    let scan_mobility = *self.scan_to_mobility.get(scan).unwrap() as f64;
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

        // Same empty-frame guard as build_precursor_frame: preserve this frame's
        // id + ms_type rather than letting from_tims_spectra_annotated([]) emit a
        // frame_id=1 / Unknown ghost.
        if tims_spectra.is_empty() {
            return TimsFrameAnnotated::new(
                frame_id as i32, rt, ms_type, vec![], vec![], vec![], vec![], vec![], vec![],
            );
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
                .into_iter()
                .map(|mut x| {
                    x.contributions.sort_by(|a, b| {
                        a.intensity_contribution
                            .partial_cmp(&b.intensity_contribution)
                            .unwrap()
                    });
                    x
                })
                .collect(),
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
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            // Use indexed parallel iteration to maintain order, avoiding post-sort
            let mut tims_frames: Vec<TimsFrameAnnotated> = Vec::with_capacity(frame_ids.len());
            unsafe { tims_frames.set_len(frame_ids.len()); }

            frame_ids.par_iter().enumerate().for_each(|(idx, frame_id)| {
                let frame = self.build_precursor_frame_annotated(
                    *frame_id,
                    mz_noise_precursor,
                    uniform,
                    precursor_noise_ppm,
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
}
