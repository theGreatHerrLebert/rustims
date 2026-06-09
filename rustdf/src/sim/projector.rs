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
    calculate_abundance_gaussian, calculate_frame_abundance_emg, calculate_frame_occurrence_emg,
    calculate_scan_occurrence_gaussian, normal_cdf_range, project_emg_over_events_par,
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
/// the per-scan 1/K0 grid in **ascending** order (the order the gaussian
/// occurrence kernel expects internally). [`project_mobility_ion`] returns scan
/// indices as positions into this ascending vector; callers that need native
/// (descending) Bruker scan numbers map at the device boundary.
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
    /// regardless of the table's native scan ordering. Non-finite mobilities are
    /// dropped (rather than panicking the sort).
    pub fn from_scans(scans: &[ScansSim]) -> Self {
        let mut inv: Vec<f64> = scans
            .iter()
            .map(|s| s.mobility as f64)
            .filter(|m| m.is_finite())
            .collect();
        inv.sort_by(f64::total_cmp);
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
        // Rely on the scheme invariants (exactly one MS1, first; >=1 MS2) so the
        // MS1 ordinal == cycle index downstream (see `project_time`).
        scheme.validate()?;
        let RepeatPolicy::FixedCycleTime { cycle_time_s, gradient_length_s, start_time_s } =
            scheme.repeat;
        if !(cycle_time_s.is_finite() && cycle_time_s > 0.0) {
            return Err("cycle_time_s must be finite and > 0".into());
        }
        if !start_time_s.is_finite() || start_time_s < 0.0 {
            return Err("start_time_s must be finite and >= 0".into());
        }
        let n_cycles = scheme.num_cycles().ok_or("scheme has no derivable cycle count")?;
        let events_per_cycle = scheme.cycle.len();
        if events_per_cycle == 0 {
            return Err("empty cycle".into());
        }

        // Per-event durations within one cycle: explicit `duration_s` are honored;
        // the cycle time left over after the explicit ones is split equally among
        // the unspecified events. Explicit durations summing beyond the cycle time
        // (which would overrun into the next cycle) is an error.
        let explicit: Vec<Option<f64>> = scheme
            .cycle
            .iter()
            .map(|e| {
                let d = match e {
                    AcquisitionEvent::Ms1(m) => m.duration_s,
                    AcquisitionEvent::DiaMs2Frame(f) => f.duration_s,
                };
                d.filter(|v| v.is_finite() && *v > 0.0)
            })
            .collect();
        let explicit_sum: f64 = explicit.iter().flatten().sum();
        let n_unspecified = explicit.iter().filter(|d| d.is_none()).count();
        if explicit_sum > cycle_time_s + 1e-9 {
            return Err(format!(
                "explicit event durations ({explicit_sum}) exceed cycle_time_s ({cycle_time_s})"
            ));
        }
        let per_unspecified = if n_unspecified > 0 {
            ((cycle_time_s - explicit_sum) / n_unspecified as f64).max(0.0)
        } else {
            0.0
        };
        let durations: Vec<f64> =
            explicit.iter().map(|d| d.unwrap_or(per_unspecified)).collect();

        let mut events = Vec::with_capacity(n_cycles as usize * events_per_cycle);
        let mut global_index = 0usize;
        for k in 0..n_cycles {
            let cycle_start = start_time_s + k as f64 * cycle_time_s;
            // Half-open run [start, gradient): num_cycles already floors to full
            // cycles, so this only guards float edge cases.
            if cycle_start >= gradient_length_s {
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
// Projection mode
// --------------------------------------------------------------------------- //

/// Which projection math to use.
///
/// `Accurate` is the default for new/non-Bruker dispatch (event-interval time
/// integration + per-scan mobility bins). `LegacyCompat` faithfully mirrors the
/// pre-dispatch pipeline's kernels and parameters — fixed `[t-cycle,t]` frame
/// bins, fixed `im_cycle_length` mobility bins, the kernels' native index
/// conventions — so it byte-reproduces a legacy Bruker `.d` for the parity gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionMode {
    LegacyCompat,
    Accurate,
}

// --------------------------------------------------------------------------- //
// LegacyCompat projections (parity target — mirror the legacy kernels exactly)
// --------------------------------------------------------------------------- //

/// LegacyCompat time projection: reproduce `frame_occurrence`/`frame_abundance`
/// over the **full** frames table, exactly as `simulate_frame_distributions_emg`
/// did. `frame_ids`/`frame_times` are the whole frames table (ascending by id);
/// the EMG is centred on each peptide's `rt_mu`. Returns `(frame_id, abundance)`
/// per peptide. (Mirrors the py flow: occurrence positions are 1-indexed into
/// the times array and equal the frame id for a contiguous 1..N table; abundance
/// integrates the fixed `[time - rt_cycle_length, time]` bin.)
pub fn project_time_legacy(
    peptides: &[PeptideScalar],
    frame_ids: &[u32],
    frame_times: &[f64],
    rt_cycle_length: f64,
    target_p: f64,
    step_size: f64,
    n_steps: Option<usize>,
) -> Vec<Vec<(u32, f64)>> {
    // time_map keyed by frame id (== occurrence position for 1..N tables).
    let time_map: HashMap<i32, f64> =
        frame_ids.iter().zip(frame_times).map(|(&id, &t)| (id as i32, t)).collect();
    peptides
        .iter()
        .map(|p| {
            let (mu, sigma, lambda) = (p.rt_mu as f64, p.rt_sigma as f64, p.rt_lambda as f64);
            let occ = calculate_frame_occurrence_emg(
                frame_times, mu, sigma, lambda, target_p, step_size, n_steps,
            );
            let abund = calculate_frame_abundance_emg(
                &time_map, &occ, mu, sigma, lambda, rt_cycle_length, n_steps,
            );
            occ.into_iter().map(|p| p as u32).zip(abund).collect()
        })
        .collect()
}

/// LegacyCompat mobility projection for one ion: reproduce
/// `scan_occurrence`/`scan_abundance` exactly as
/// `simulate_scan_distributions_with_variance` did — `calculate_scan_occurrence_
/// gaussian` with `n_lower/upper = 5`, and `calculate_abundance_gaussian` over
/// the fixed `im_cycle_length` bin. `scan_mobilities` must be **ascending** with
/// `scan_ids` aligned (exactly as the legacy job feeds `scans` sorted so
/// `im_cycle_length > 0`); a descending grid makes the occurrence kernel return
/// nothing. Returns `(scan, abundance)`.
pub fn project_mobility_ion_legacy(
    ion: &IonScalar,
    scan_ids: &[u32],
    scan_mobilities: &[f64],
    env: &MobilityEnv,
    im_cycle_length: f64,
    target_p: f64,
    step_size: f64,
) -> Vec<(i32, f64)> {
    let mean = ion.inv_mobility(env);
    let sigma = ion.inv_mobility_std as f64;
    let occ =
        calculate_scan_occurrence_gaussian(scan_mobilities, mean, sigma, target_p, step_size, 5.0, 5.0);
    let time_map: HashMap<i32, f64> =
        scan_ids.iter().zip(scan_mobilities).map(|(&id, &m)| (id as i32, m)).collect();
    let abund = calculate_abundance_gaussian(&time_map, &occ, mean, sigma, im_cycle_length);
    occ.into_iter().zip(abund).collect()
}

// --------------------------------------------------------------------------- //
// Projections (Accurate)
// --------------------------------------------------------------------------- //

/// Time projection: for each peptide, the `(cycle_index, abundance)` list over
/// the run's MS1 events. The index is the MS1 ordinal, which equals the cycle
/// index because [`EventTimeline::from_scheme`] validates exactly one MS1 per
/// cycle — it is NOT a global event index (MS2 events are interleaved). Replaces
/// the legacy `frame_occurrence`/`frame_abundance`.
pub fn project_time(
    peptides: &[PeptideScalar],
    timeline: &EventTimeline,
    target_p: f64,
    step_size: f64,
    num_threads: usize,
) -> Vec<Vec<(usize, f64)>> {
    let intervals = timeline.ms1_intervals();
    // The EMG is centred on rt_mu (the location parameter), not the GRU apex.
    let rts: Vec<f64> = peptides.iter().map(|p| p.rt_mu as f64).collect();
    let sigmas: Vec<f64> = peptides.iter().map(|p| p.rt_sigma as f64).collect();
    let lambdas: Vec<f64> = peptides.iter().map(|p| p.rt_lambda as f64).collect();
    project_emg_over_events_par(
        &intervals,
        rts,
        sigmas,
        lambdas,
        target_p,
        step_size,
        num_threads.max(1), // 0 would panic the rayon pool builder
        None,
    )
}

/// Mobility projection for a single ion: `(scan_index, abundance)` over the
/// geometry's scan grid, scan indices being positions into the **ascending**
/// `geometry.inv_mobility` (per the [`SamplingGeometry`] contract).
///
/// Each occupied scan's abundance is the mobility Gaussian's CDF over that
/// scan's own bin — the midpoints to its neighbours on the (possibly
/// non-uniform) calibrated grid — so dense/sparse regions are integrated
/// correctly (a single mean spacing would overlap/gap them).
///
/// Intensity-contract note: for `MobilityModality::None` the result is the full
/// marginal `[(0, 1.0)]` by definition (no mobility axis, complete integration).
/// For `Tims` the per-scan abundances sum to the captured mass, which is `<= 1`
/// (grid clipping + `target_p` truncation) — the two cases are intentionally
/// different and downstream normalisation must account for it.
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
            let n = geometry.inv_mobility.len();
            if n == 0 {
                return Vec::new();
            }
            if n == 1 {
                // Single scan: integrate the whole captured Gaussian onto it.
                let mean = ion.inv_mobility(env);
                let sigma = ion.inv_mobility_std as f64;
                let a = normal_cdf_range(mean - 6.0 * sigma, mean + 6.0 * sigma, mean, sigma);
                return vec![(0, a)];
            }
            let mean = ion.inv_mobility(env);
            let sigma = ion.inv_mobility_std as f64;
            // `calculate_scan_occurrence_gaussian` returns indices into the
            // REVERSED (descending) grid: occ index i -> inv_mobility[n-1-i].
            let occ = calculate_scan_occurrence_gaussian(
                &geometry.inv_mobility,
                mean,
                sigma,
                target_p,
                step_size,
                3.0,
                3.0,
            );
            let rev: Vec<f64> = geometry.inv_mobility.iter().rev().copied().collect();
            let mut out: Vec<(i32, f64)> = occ
                .into_iter()
                .map(|i| {
                    let i = i as usize;
                    let v = rev[i];
                    // Bin edges = midpoints to neighbours on the descending grid;
                    // endpoints extrapolate by their adjacent half-spacing.
                    let hi = if i > 0 {
                        (rev[i - 1] + v) / 2.0
                    } else {
                        v + (v - rev[1]) / 2.0
                    };
                    let lo = if i + 1 < n {
                        (v + rev[i + 1]) / 2.0
                    } else {
                        v - (rev[i - 1] - v) / 2.0
                    };
                    let abundance = normal_cdf_range(lo, hi, mean, sigma);
                    // Convert reversed index back to an ascending-grid position.
                    ((n - 1 - i) as i32, abundance)
                })
                .collect();
            out.sort_by_key(|(scan, _)| *scan);
            out
        }
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
            rt_mu: rt,
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
    fn timeline_mixed_durations_fill_remaining_cycle_time() {
        // MS1 explicit 0.8 s; the two MS2 events split the remaining 0.4 s.
        let mut scheme = scheme_one_ms1_two_ms2();
        if let AcquisitionEvent::Ms1(ref mut m) = scheme.cycle[0] {
            m.duration_s = Some(0.8);
        }
        let tl = EventTimeline::from_scheme(&scheme).unwrap();
        let first3 = &tl.events[..3];
        assert!((first3[0].interval.1 - first3[0].interval.0 - 0.8).abs() < 1e-9);
        assert!((first3[1].interval.1 - first3[1].interval.0 - 0.2).abs() < 1e-9);
        // Still tile to exactly one cycle (1.2 s).
        assert!((first3[2].interval.1 - 1.2).abs() < 1e-9);
    }

    #[test]
    fn timeline_rejects_durations_overrunning_cycle() {
        let mut scheme = scheme_one_ms1_two_ms2();
        if let AcquisitionEvent::Ms1(ref mut m) = scheme.cycle[0] {
            m.duration_s = Some(2.0); // > cycle_time 1.2
        }
        assert!(EventTimeline::from_scheme(&scheme).is_err());
    }

    #[test]
    fn timeline_honours_nonzero_start_time() {
        let mut scheme = scheme_one_ms1_two_ms2();
        scheme.repeat = RepeatPolicy::FixedCycleTime {
            cycle_time_s: 1.2,
            gradient_length_s: 12.0,
            start_time_s: 3.0,
        };
        let tl = EventTimeline::from_scheme(&scheme).unwrap();
        assert!((tl.events[0].interval.0 - 3.0).abs() < 1e-9);
        // (12 - 3)/1.2 = 7 full cycles.
        assert_eq!(tl.ms1_intervals().len(), 7);
    }

    #[test]
    fn project_mobility_nonuniform_grid_uses_local_bins() {
        // Non-uniform ascending grid: dense low, sparse high.
        let mut grid = vec![0.60, 0.61, 0.62, 0.63, 0.64];
        grid.extend([0.9, 1.2, 1.5]);
        let geom = SamplingGeometry::tims(grid);
        let env = MobilityEnv::default();
        let i = ion(450.0, 600.0, 2, 0.05);
        let m = project_mobility_ion(&i, &geom, &env, 0.9999, 0.01);
        // Output indices are ascending-grid positions, sorted, and in range.
        assert!(!m.is_empty());
        for w in m.windows(2) {
            assert!(w[0].0 < w[1].0, "scan indices must be sorted ascending");
        }
        for (scan, a) in &m {
            assert!((*scan as usize) < geom.num_scans());
            assert!(*a >= 0.0);
        }
    }

    #[test]
    fn legacy_time_matches_raw_kernels() {
        use mscore::algorithm::utility::{
            calculate_frame_abundance_emg, calculate_frame_occurrence_emg,
        };
        // Contiguous 1..N frames table.
        let n = 60usize;
        let frame_ids: Vec<u32> = (1..=n as u32).collect();
        let frame_times: Vec<f64> = (1..=n).map(|i| i as f64 * 0.1).collect();
        let rt_cycle = 0.1;
        let mut pep = peptide(3.0, 0.4, 0.6);
        pep.rt_mu = 3.0;
        let out = project_time_legacy(&[pep.clone()], &frame_ids, &frame_times, rt_cycle, 0.9999, 0.01, None);

        // Raw kernel call with identical inputs — use the SAME f32-rounded
        // values the function reads from the struct (literals would differ at
        // the f32->f64 rounding level and fail an exact compare).
        let (mu, sigma, lambda) = (pep.rt_mu as f64, pep.rt_sigma as f64, pep.rt_lambda as f64);
        let occ = calculate_frame_occurrence_emg(&frame_times, mu, sigma, lambda, 0.9999, 0.01, None);
        let tmap: std::collections::HashMap<i32, f64> =
            frame_ids.iter().zip(&frame_times).map(|(&id, &t)| (id as i32, t)).collect();
        let abund = calculate_frame_abundance_emg(&tmap, &occ, mu, sigma, lambda, rt_cycle, None);
        let expected: Vec<(u32, f64)> = occ.iter().map(|&p| p as u32).zip(abund).collect();
        assert_eq!(out[0], expected, "LegacyCompat time must mirror the raw kernels");
    }

    #[test]
    fn legacy_mobility_matches_raw_kernels() {
        use mscore::algorithm::utility::{
            calculate_abundance_gaussian, calculate_scan_occurrence_gaussian,
        };
        // Mobility ASCENDING (as the legacy job feeds it — im_cycle_length>0);
        // scan ids align (high scan id = low mobility, so ids descend here).
        let nn = 200usize;
        let scan_ids: Vec<u32> = (0..nn as u32).rev().collect();
        let scan_mob: Vec<f64> = (0..nn).map(|i| 0.804 + i as f64 * 0.004).collect();
        let im_cycle = 0.004;
        let env = MobilityEnv::default();
        let i = ion(450.0, 600.0, 2, 0.02);
        let out = project_mobility_ion_legacy(&i, &scan_ids, &scan_mob, &env, im_cycle, 0.9999, 0.0001);

        let mean = i.inv_mobility(&env);
        let sigma = i.inv_mobility_std as f64;
        let occ = calculate_scan_occurrence_gaussian(&scan_mob, mean, sigma, 0.9999, 0.0001, 5.0, 5.0);
        let tmap: std::collections::HashMap<i32, f64> =
            scan_ids.iter().zip(&scan_mob).map(|(&id, &m)| (id as i32, m)).collect();
        let abund = calculate_abundance_gaussian(&tmap, &occ, mean, sigma, im_cycle);
        let expected: Vec<(i32, f64)> = occ.into_iter().zip(abund).collect();
        assert_eq!(out, expected, "LegacyCompat mobility must mirror the raw kernels");
        assert!(!out.is_empty());
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
