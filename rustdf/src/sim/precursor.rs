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
use crate::sim::projector::{IntensityStage, MzCoordSpace, RenderedEvent, RenderedSpectrum};
use crate::sim::scheme::DataMode;
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

    /// Faithful sum of the per-scan contributions for a frame into ONE spectrum:
    /// `Σ_scans (scaled_spectrum)`. The scan coordinate is discarded; peaks at the
    /// same m/z merge (`from_collection`'s deterministic m/z binning).
    ///
    /// NOTE — this is **not** the correct non-IMS MS1 physics. It reproduces what
    /// the TIMS scan grid captured, whose total mobility mass is `Σ scan_abundance`
    /// ≲ `target_p` (grid truncation) and can be far less for ions whose mobility
    /// sits at/outside the grid edge (those are under-counted, or dropped entirely
    /// when no scan captured them). A real no-mobility instrument has no such grid,
    /// so it sees the FULL mobility marginal — use
    /// [`Self::precursor_scan_marginal_spectrum`] for the Astral render. This
    /// captured-grid collapse is kept as a diagnostic / exact-equivalent of the
    /// Bruker per-scan contributions.
    pub fn precursor_scan_spectrum(&self, frame_id: u32) -> MzSpectrum {
        let specs: Vec<MzSpectrum> = self
            .precursor_frame_contributions(frame_id)
            .into_iter()
            .map(|(_scan, spectrum)| spectrum)
            .collect();
        MzSpectrum::from_collection(specs)
    }

    /// Non-IMS (e.g. Astral) MS1 spectrum: the **full mobility marginal** (P6c).
    ///
    /// A no-mobility instrument integrates the entire ion-mobility distribution
    /// (marginal = 1.0 by construction) — it cannot lose ions to TIMS-grid
    /// truncation. So each (peptide, ion) in the frame contributes its isotope
    /// spectrum scaled by `frame_abundance × ion_abundance × total_events`, with
    /// the per-scan mobility split (`scan_abundance`, summing to ≲ `target_p`)
    /// replaced by the full marginal 1.0. Crucially, ions are included by frame
    /// membership (`frame_abundance > 0`), NOT by whether the TIMS grid captured
    /// any of their mobility — so grid-edge ions [`precursor_scan_spectrum`] would
    /// under-count or drop are recorded at full abundance. PRE m/z-noise; peaks at
    /// the same m/z merge deterministically.
    pub fn precursor_scan_marginal_spectrum(&self, frame_id: u32) -> MzSpectrum {
        let mut specs: Vec<MzSpectrum> = Vec::new();
        let Some((peptide_ids, abundances)) = self.frame_to_abundances.get(&frame_id) else {
            return MzSpectrum::from_collection(specs);
        };
        specs.reserve(peptide_ids.len() * 4);
        for (peptide_id, frame_abundance) in peptide_ids.iter().zip(abundances.iter()) {
            let Some((ion_abundances, _scan_occurrences, _scan_abundances, _, spectra)) =
                self.peptide_to_ions.get(peptide_id)
            else {
                continue;
            };
            let total_events = *self.peptide_to_events.get(peptide_id).unwrap();
            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                // Full mobility marginal: scan factor folded to 1.0 (no grid).
                let abundance_factor = frame_abundance * ion_abundance * total_events;
                specs.push(spectra[index].clone() * abundance_factor as f64);
            }
        }
        MzSpectrum::from_collection(specs)
    }

    /// Render a precursor (MS1) frame as a vendor-neutral [`RenderedEvent::Scan`]
    /// for a non-IMS instrument (P6c). This is the scan-based render core's MS1
    /// path: it collapses the mobility axis via
    /// [`Self::precursor_scan_marginal_spectrum`] (the full mobility marginal, the
    /// correct non-IMS physics — NOT the captured-grid sum) into the single MS1
    /// spectrum an Astral/Orbitrap records.
    ///
    /// The spectrum is tagged as physical-m/z, NOT detector-applied, at the
    /// [`IntensityStage::Mobility`] stage — the marginal already carries the yield
    /// (events), time (frame abundance) and mobility (fully integrated) factors,
    /// but no quadrupole transmission (MS1 is unisolated) and no detector response;
    /// the writer / detector model applies those downstream without double-
    /// counting. `data_mode` is the instrument's MS1 acquisition mode (Astral MS1 =
    /// `Profile`). Returns an empty-spectrum `Scan` for a frame with no
    /// contributions (the caller decides whether to emit it).
    pub fn render_precursor_scan(&self, frame_id: u32, data_mode: DataMode) -> RenderedEvent {
        let rt = *self.frame_to_rt.get(&frame_id).unwrap() as f64;
        let spectrum = self.precursor_scan_marginal_spectrum(frame_id);
        RenderedEvent::Scan {
            ms_level: 1,
            retention_time_s: rt,
            isolation: None,
            spectrum: RenderedSpectrum {
                mz: (*spectrum.mz).clone(),
                intensity: (*spectrum.intensity).clone(),
                coords: MzCoordSpace::Physical,
                mode: data_mode,
                detector_applied: false,
                stage: IntensityStage::Mobility,
            },
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built single-frame builder: peptide 10 has an ion whose mobility was
    /// captured on the scan grid as `scan_abundance = [0.3, 0.2]` (Σ = 0.5, i.e.
    /// half its mass truncated by the grid); peptide 11 has an ion the grid
    /// dropped entirely (no captured scans). Both peptides are in the frame with
    /// `frame_abundance = 0.8` and `total_events = 1000`.
    fn one_frame_builder() -> TimsTofSyntheticsPrecursorFrameBuilder {
        let mut frame_to_abundances = BTreeMap::new();
        frame_to_abundances.insert(1u32, (vec![10u32, 11u32], vec![0.8f32, 0.8f32]));

        let mut peptide_to_ions = BTreeMap::new();
        peptide_to_ions.insert(
            10u32,
            (
                vec![1.0f32],
                vec![vec![100u32, 101u32]],
                vec![vec![0.3f32, 0.2f32]], // Σ = 0.5 captured on the grid
                vec![1i8],
                vec![MzSpectrum::new(vec![500.0], vec![1.0])],
            ),
        );
        peptide_to_ions.insert(
            11u32,
            (
                vec![1.0f32],
                vec![vec![]], // grid captured no scans for this ion
                vec![vec![]],
                vec![1i8],
                vec![MzSpectrum::new(vec![700.0], vec![1.0])],
            ),
        );

        let mut peptide_to_events = BTreeMap::new();
        peptide_to_events.insert(10u32, 1000.0f32);
        peptide_to_events.insert(11u32, 1000.0f32);

        let mut frame_to_rt = BTreeMap::new();
        frame_to_rt.insert(1u32, 60.0f32);

        TimsTofSyntheticsPrecursorFrameBuilder {
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
        }
    }

    fn total_intensity(s: &MzSpectrum) -> f64 {
        s.intensity.iter().sum()
    }

    #[test]
    fn astral_marginal_uses_full_mobility_and_recovers_grid_edge_ions() {
        let b = one_frame_builder();

        // Captured-grid sum: ion 10 weighted by Σ scan_abundance = 0.5, so
        //   0.8 * 1.0 * 1000 * 0.5 = 400 at m/z 500. Ion 11 dropped (no scans).
        let grid = b.precursor_scan_spectrum(1);
        assert!((total_intensity(&grid) - 400.0).abs() < 1e-2, "grid total {}", total_intensity(&grid));
        assert_eq!(grid.mz.len(), 1, "grid drops the edge ion (no captured scans)");
        assert!((grid.mz[0] - 500.0).abs() < 1e-9);

        // Full mobility marginal (non-IMS physics): scan factor folded to 1.0, and
        // the edge ion 11 is recovered at full abundance.
        //   ion 10: 0.8 * 1.0 * 1000 * 1.0 = 800 at m/z 500
        //   ion 11: 0.8 * 1.0 * 1000 * 1.0 = 800 at m/z 700
        let marginal = b.precursor_scan_marginal_spectrum(1);
        assert!((total_intensity(&marginal) - 1600.0).abs() < 1e-2, "marginal total {}", total_intensity(&marginal));
        assert_eq!(marginal.mz.len(), 2, "marginal keeps both ions");
        assert!(marginal.mz.iter().any(|&m| (m - 700.0).abs() < 1e-9), "edge ion recovered");

        // The marginal is the principled non-IMS value; the grid here under-counts
        // (truncation). It is NOT a theorem that marginal >= grid per frame — grid
        // binning can also over-count when adjacent scan intervals overlap.
        assert!(total_intensity(&marginal) > total_intensity(&grid));

        // The MS1 scan render must carry exactly the marginal spectrum.
        let RenderedEvent::Scan { ms_level, isolation, spectrum, .. } =
            b.render_precursor_scan(1, DataMode::Profile)
        else {
            panic!("MS1 render must be a Scan");
        };
        assert_eq!(ms_level, 1);
        assert!(isolation.is_none());
        assert_eq!(spectrum.intensity, *marginal.intensity);
        assert_eq!(spectrum.mz, *marginal.mz);
    }
}
