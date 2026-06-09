//! Instrument-dispatch P2: the projector.
//!
//! Turns the vendor-neutral trunk (scalar [`PeptideScalar`] / [`IonScalar`]
//! physics) into device-sampled signal, parameterised by the instrument's
//! sampling geometry and an absolute event timeline derived from the
//! [`AcquisitionScheme`]. This replaces the two device-projection jobs that the
//! legacy pipeline baked into the DB (`frame_occurrence/abundance` and
//! `scan_occurrence/abundance`) with a render-time computation:
//!
//! * **time projection** (all instruments): the EMG RT profile integrated over
//!   each event's true `[start, end]` interval — [`mscore`]'s
//!   `project_emg_over_events` (the corrected, non-fixed-cycle kernel).
//! * **mobility projection** (IMS only): the ion mobility Gaussian sampled onto
//!   the scan grid via the existing gaussian kernels; for a non-IMS instrument
//!   (`MobilityModality::None`) the distribution is marginalised onto a single
//!   scan, conserving total signal.
//!
//! P2 builds and verifies these projections in isolation. Wiring `assemble_frames`
//! onto them (with a DB fallback) and the byte-parity proof are P3.

use crate::sim::containers::{IonScalar, MobilityEnv, PeptideScalar, ScansSim};
use crate::sim::scheme::{
    AcquisitionEvent, AcquisitionScheme, DataMode, IsolationWindow, RepeatPolicy,
};
use mscore::algorithm::utility::{
    calculate_abundance_gaussian, calculate_scan_occurrence_gaussian, project_emg_over_events_par,
};
use std::collections::HashMap;

// --------------------------------------------------------------------------- //
// Sampling geometry (instrument branch 1)
// --------------------------------------------------------------------------- //

/// Ion-mobility separation modality. Only the cases the engine handles today are
/// listed; `DriftTube`/`Faims` are reserved for later instruments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobilityModality {
    /// No mobility axis (Orbitrap / Astral): ions collapse onto the time axis.
    None,
    /// timsTOF trapped ion mobility: a discrete scan grid.
    Tims,
}

/// The instrument's mobility sampling geometry. For `Tims`, `inv_mobility` holds
/// the per-scan 1/K0 grid in **ascending** order — the order the gaussian
/// occurrence kernel expects (it reverses internally; see its doctest). The
/// returned scan indices are positions into this ascending vector; callers that
/// need native (descending) Bruker scan numbers map at the device boundary.
#[derive(Debug, Clone)]
pub struct SamplingGeometry {
    pub modality: MobilityModality,
    pub inv_mobility: Vec<f64>,
}

impl SamplingGeometry {
    /// A non-IMS geometry (single virtual scan).
    pub fn none() -> Self {
        SamplingGeometry { modality: MobilityModality::None, inv_mobility: Vec::new() }
    }

    /// A timsTOF geometry from a per-scan 1/K0 grid.
    pub fn tims(inv_mobility: Vec<f64>) -> Self {
        SamplingGeometry { modality: MobilityModality::Tims, inv_mobility }
    }

    /// Build a timsTOF geometry from the reference dataset's `scans` table. The
    /// 1/K0 values are sorted ascending to satisfy the kernel's input contract
    /// regardless of the table's native scan ordering.
    pub fn from_scans(scans: &[ScansSim]) -> Self {
        let mut inv: Vec<f64> = scans.iter().map(|s| s.mobility as f64).collect();
        inv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        SamplingGeometry::tims(inv)
    }

    /// Number of scans in the grid (1 for the marginalised non-IMS case).
    pub fn num_scans(&self) -> usize {
        match self.modality {
            MobilityModality::None => 1,
            MobilityModality::Tims => self.inv_mobility.len(),
        }
    }
}

// --------------------------------------------------------------------------- //
// Event timeline (schedule expansion — branch 2 laid out over the gradient)
// --------------------------------------------------------------------------- //

/// One concrete acquisition event placed on the absolute run timeline, with its
/// `[start_s, end_s]` exposure interval (the input to the time projection).
#[derive(Debug, Clone)]
pub struct EventSlot {
    /// Global event index over the whole run (0-based, acquisition order).
    pub global_index: usize,
    pub cycle_index: u64,
    /// Position within the cycle (0 = the MS1).
    pub event_in_cycle: usize,
    pub ms_level: u8,
    /// Exposure interval (seconds) the analyte signal is integrated over.
    pub interval: (f64, f64),
}

/// The full run as an ordered list of [`EventSlot`]s.
#[derive(Debug, Clone)]
pub struct EventTimeline {
    pub events: Vec<EventSlot>,
}

impl EventTimeline {
    /// Expand an [`AcquisitionScheme`] into absolute event intervals over the
    /// gradient. Each cycle is laid out starting at `start_time_s + k*cycle_time`;
    /// within a cycle, events are placed back-to-back using their `duration_s`
    /// when present, otherwise the cycle time is split uniformly across events.
    pub fn from_scheme(scheme: &AcquisitionScheme) -> Result<Self, String> {
        let RepeatPolicy::FixedCycleTime { cycle_time_s, gradient_length_s, start_time_s } =
            scheme.repeat;
        if !(cycle_time_s.is_finite() && cycle_time_s > 0.0) {
            return Err("cycle_time_s must be finite and > 0".into());
        }
        let n_cycles = scheme.num_cycles().ok_or("scheme has no derivable cycle count")?;
        let events_per_cycle = scheme.cycle.len();
        if events_per_cycle == 0 {
            return Err("empty cycle".into());
        }

        // Per-event durations within one cycle: explicit duration_s, else an
        // equal split of the cycle time across all events.
        let uniform = cycle_time_s / events_per_cycle as f64;
        let durations: Vec<f64> = scheme
            .cycle
            .iter()
            .map(|e| {
                let d = match e {
                    AcquisitionEvent::Ms1(m) => m.duration_s,
                    AcquisitionEvent::DiaMs2Frame(f) => f.duration_s,
                };
                d.filter(|v| v.is_finite() && *v > 0.0).unwrap_or(uniform)
            })
            .collect();

        let mut events = Vec::with_capacity(n_cycles as usize * events_per_cycle);
        let mut global_index = 0usize;
        for k in 0..n_cycles {
            let cycle_start = start_time_s + k as f64 * cycle_time_s;
            if cycle_start > gradient_length_s {
                break;
            }
            let mut t = cycle_start;
            for (j, ev) in scheme.cycle.iter().enumerate() {
                let start = t;
                let end = t + durations[j];
                t = end;
                let ms_level = match ev {
                    AcquisitionEvent::Ms1(_) => 1,
                    AcquisitionEvent::DiaMs2Frame(_) => 2,
                };
                events.push(EventSlot {
                    global_index,
                    cycle_index: k,
                    event_in_cycle: j,
                    ms_level,
                    interval: (start, end),
                });
                global_index += 1;
            }
        }
        Ok(EventTimeline { events })
    }

    /// The MS1 events only, in order (their `[start,end]` intervals drive the
    /// precursor time projection).
    pub fn ms1_intervals(&self) -> Vec<(f64, f64)> {
        self.events
            .iter()
            .filter(|e| e.ms_level == 1)
            .map(|e| e.interval)
            .collect()
    }
}

// --------------------------------------------------------------------------- //
// RenderedEvent — the frozen output/writer-boundary interface (§3.2)
// --------------------------------------------------------------------------- //

/// Which coordinate space a rendered spectrum's m/z values are in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MzCoordSpace {
    /// Exact physical m/z (projector output, pre-detector).
    Physical,
    /// Native TOF index (Bruker).
    NativeTof,
    /// Native frequency (Thermo).
    NativeFreq,
}

/// Point on the intensity-conservation chain (mirrors the Python
/// `dispatch.IntensityStage`; see plan §3.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntensityStage {
    Yield,
    Time,
    Mobility,
    Transmitted,
    Detected,
    Centroided,
}

/// A spectrum tagged with where it sits in the coordinate/intensity pipeline, so
/// writers and the detector model never double-apply a transform.
#[derive(Debug, Clone)]
pub struct RenderedSpectrum {
    pub mz: Vec<f64>,
    pub intensity: Vec<f64>,
    pub coords: MzCoordSpace,
    pub mode: DataMode,
    pub detector_applied: bool,
    pub stage: IntensityStage,
}

/// One acquisition event's rendered signal. `Scan` for non-IMS instruments;
/// `MobilityFrame` for IMS (preserving per-scan simultaneity rather than
/// flattening + rebundling).
#[derive(Debug, Clone)]
pub enum RenderedEvent {
    Scan {
        ms_level: u8,
        retention_time_s: f64,
        isolation: Option<IsolationWindow>,
        spectrum: RenderedSpectrum,
    },
    MobilityFrame {
        ms_level: u8,
        retention_time_s: f64,
        scans: Vec<(u32, RenderedSpectrum)>,
    },
}

// --------------------------------------------------------------------------- //
// Projections
// --------------------------------------------------------------------------- //

/// Time projection: for each peptide, the `(ms1_event_index, abundance)` list
/// over the timeline's MS1 events. `ms1_event_index` indexes into
/// [`EventTimeline::ms1_intervals`] (i.e. cycle order). Replaces the legacy
/// `frame_occurrence`/`frame_abundance`.
pub fn project_time(
    peptides: &[PeptideScalar],
    timeline: &EventTimeline,
    target_p: f64,
    step_size: f64,
    num_threads: usize,
) -> Vec<Vec<(usize, f64)>> {
    let intervals = timeline.ms1_intervals();
    let rts: Vec<f64> = peptides.iter().map(|p| p.retention_time as f64).collect();
    let sigmas: Vec<f64> = peptides.iter().map(|p| p.rt_sigma as f64).collect();
    let lambdas: Vec<f64> = peptides.iter().map(|p| p.rt_lambda as f64).collect();
    project_emg_over_events_par(
        &intervals, rts, sigmas, lambdas, target_p, step_size, num_threads, None,
    )
}

/// Mobility projection for a single ion: `(scan_index, abundance)` over the
/// geometry's scan grid. For `MobilityModality::None` the whole distribution is
/// marginalised onto scan 0 (total signal conserved). Replaces the legacy
/// `scan_occurrence`/`scan_abundance`.
pub fn project_mobility_ion(
    ion: &IonScalar,
    geometry: &SamplingGeometry,
    env: &MobilityEnv,
    target_p: f64,
    step_size: f64,
) -> Vec<(i32, f64)> {
    match geometry.modality {
        MobilityModality::None => vec![(0, 1.0)],
        MobilityModality::Tims => {
            let mean = ion.inv_mobility(env);
            let sigma = ion.inv_mobility_std as f64;
            let occ = calculate_scan_occurrence_gaussian(
                &geometry.inv_mobility,
                mean,
                sigma,
                target_p,
                step_size,
                3.0,
                3.0,
            );
            // Per-scan abundance via the existing gaussian bin integration over
            // [mobility - bin_width, mobility] (bin_width = mean grid spacing).
            // `calculate_scan_occurrence_gaussian` indexes the *reversed* grid,
            // so the abundance time_map must use that same index space (index i
            // -> inv_mobility[n-1-i]) or the integration reads the wrong scan.
            let n = geometry.inv_mobility.len();
            let time_map: HashMap<i32, f64> = (0..n)
                .map(|i| (i as i32, geometry.inv_mobility[n - 1 - i]))
                .collect();
            let bin_width = mean_grid_spacing(&geometry.inv_mobility);
            let abund = calculate_abundance_gaussian(&time_map, &occ, mean, sigma, bin_width);
            occ.into_iter().zip(abund).collect()
        }
    }
}

/// Mean absolute spacing of a (descending) mobility grid; the per-scan bin width
/// used for abundance integration. Returns a small positive fallback for grids
/// with < 2 points.
fn mean_grid_spacing(grid: &[f64]) -> f64 {
    if grid.len() < 2 {
        return 1e-3;
    }
    let total: f64 = grid.windows(2).map(|w| (w[0] - w[1]).abs()).sum();
    let s = total / (grid.len() - 1) as f64;
    if s.is_finite() && s > 0.0 {
        s
    } else {
        1e-3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::scheme::{
        Analyzer, CollisionEnergyPolicy, DiaGeometry, DiaMs2Frame, DiaWindow, InstrumentKind,
        Ms1Event, Provenance, SchemeSource,
    };
    use mscore::data::spectrum::MzSpectrum;

    fn scheme_one_ms1_two_ms2() -> AcquisitionScheme {
        let win = |c: f64| DiaWindow {
            isolation: IsolationWindow { center_mz: c, width_mz: 25.0 },
            collision_energy: CollisionEnergyPolicy::Value(25.0),
            geometry: DiaGeometry::MzOnly,
        };
        let ms2 = |c: f64| {
            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: vec![win(c)],
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: None,
            })
        };
        AcquisitionScheme {
            version: 1,
            instrument: InstrumentKind::TimsTofDia,
            cycle: vec![
                AcquisitionEvent::Ms1(Ms1Event {
                    analyzer: Analyzer::Tof,
                    data_mode: DataMode::Profile,
                    mz_range: Some((100.0, 1700.0)),
                    duration_s: None,
                }),
                ms2(450.0),
                ms2(550.0),
            ],
            repeat: RepeatPolicy::FixedCycleTime {
                cycle_time_s: 1.2,
                gradient_length_s: 12.0,
                start_time_s: 0.0,
            },
            mz_range: (100.0, 1700.0),
            provenance: Provenance { source: SchemeSource::Programmatic, notes: String::new() },
        }
    }

    #[test]
    fn timeline_expands_to_contiguous_intervals() {
        let scheme = scheme_one_ms1_two_ms2();
        let tl = EventTimeline::from_scheme(&scheme).unwrap();
        // 10 cycles * 3 events = 30 events.
        assert_eq!(tl.events.len(), 30);
        // Within a cycle the three events tile [0,1.2] back-to-back (0.4 each).
        let first3 = &tl.events[..3];
        assert!((first3[0].interval.0 - 0.0).abs() < 1e-9);
        assert!((first3[2].interval.1 - 1.2).abs() < 1e-9);
        for w in first3.windows(2) {
            assert!((w[0].interval.1 - w[1].interval.0).abs() < 1e-9, "events must be contiguous");
        }
        // Second cycle starts at 1.2.
        assert!((tl.events[3].interval.0 - 1.2).abs() < 1e-9);
        // One MS1 per cycle.
        assert_eq!(tl.ms1_intervals().len(), 10);
    }

    fn peptide(rt: f32, sigma: f32, lambda: f32) -> PeptideScalar {
        use mscore::data::peptide::PeptideSequence;
        PeptideScalar {
            protein_id: 0,
            peptide_id: 1,
            sequence: PeptideSequence::new("PEPTIDEK".into(), Some(1)),
            proteins: "P".into(),
            decoy: false,
            missed_cleavages: 0,
            mono_isotopic_mass: 930.0,
            retention_time: rt,
            rt_sigma: sigma,
            rt_lambda: lambda,
            events: 1.0,
            condition_id: None,
        }
    }

    #[test]
    fn project_time_picks_events_near_rt() {
        let scheme = scheme_one_ms1_two_ms2();
        let tl = EventTimeline::from_scheme(&scheme).unwrap();
        // Peptide eluting at ~6 s should map to MS1 events around cycle 5.
        let pep = peptide(6.0, 0.3, 0.5);
        let proj = project_time(&[pep], &tl, 0.9999, 0.01, 1);
        assert_eq!(proj.len(), 1);
        let hits = &proj[0];
        let ms1 = tl.ms1_intervals();
        // RT-support truncation: touches some but NOT all MS1 events.
        assert!(!hits.is_empty(), "peptide must touch some MS1 events");
        assert!(hits.len() < ms1.len(), "RT support must be truncated, not all {} cycles", ms1.len());
        // Touched events are a contiguous index range with positive abundance.
        for w in hits.windows(2) {
            assert_eq!(w[1].0, w[0].0 + 1, "touched MS1 events must be contiguous");
        }
        for (_, abund) in hits {
            assert!(*abund > 0.0);
        }
    }

    fn ion(ccs: f64, mz: f64, charge: i8, std: f32) -> IonScalar {
        IonScalar {
            ion_id: 1,
            peptide_id: 1,
            sequence: "PEPTIDEK".into(),
            charge,
            relative_abundance: 1.0,
            mz,
            ccs,
            inv_mobility_std: std,
            simulated_spectrum: MzSpectrum::new(vec![mz], vec![1.0]),
            condition_id: None,
        }
    }

    #[test]
    fn project_mobility_none_marginalises_to_single_scan() {
        let geom = SamplingGeometry::none();
        let env = MobilityEnv::default();
        let i = ion(350.0, 500.0, 2, 0.01);
        let m = project_mobility_ion(&i, &geom, &env, 0.9999, 0.01);
        assert_eq!(m, vec![(0, 1.0)]);
        assert_eq!(geom.num_scans(), 1);
    }

    #[test]
    fn project_mobility_tims_lands_in_grid() {
        // Ascending 1/K0 grid (the kernel's input contract).
        let grid: Vec<f64> = (0..100).map(|i| 1.005 + i as f64 * 0.005).collect();
        let geom = SamplingGeometry::tims(grid);
        let env = MobilityEnv::default();
        // Pick an ion whose derived 1/K0 sits inside the grid.
        let i = ion(450.0, 600.0, 2, 0.02);
        let mean = i.inv_mobility(&env);
        assert!(mean > 0.6 && mean < 1.5, "test ion 1/K0 {mean} should fall in grid");
        let m = project_mobility_ion(&i, &geom, &env, 0.9999, 0.01);
        assert!(!m.is_empty(), "ion must occupy >= 1 scan");
        let total: f64 = m.iter().map(|(_, a)| a).sum();
        assert!(total > 0.0);
    }
}
