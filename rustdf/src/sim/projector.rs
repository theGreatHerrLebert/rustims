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
    AcquisitionEvent, AcquisitionScheme, ActivationPolicy, DataMode, InstrumentCapabilities,
    InstrumentKind, IsolationWindow, RepeatPolicy,
};
use mscore::algorithm::utility::{
    calculate_abundance_gaussian, calculate_frame_abundances_emg_par,
    calculate_frame_occurrences_emg_par, calculate_scan_abundances_gaussian_par,
    calculate_scan_occurrence_gaussian, calculate_scan_occurrences_gaussian_par, normal_cdf_range,
    project_emg_over_events_par,
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
// Instrument config (the dispatch bundle threaded through the render core)
// --------------------------------------------------------------------------- //

/// The vendor-specific configuration the render core dispatches on (P6c). Bundles
/// the three orthogonal axes that decide how a vendor-neutral trunk is recorded:
///
/// * [`InstrumentKind`] — vendor identity (selects the writer at the boundary);
/// * [`InstrumentCapabilities`] — what physics applies (gates mobility / quad
///   isotope transmission; an Astral forces isotope mode to `None`);
/// * [`MobilityModality`] — whether events render as `MobilityFrame` (TIMS) or a
///   single collapsed `Scan` (non-IMS);
/// * [`ActivationPolicy`] — how the collision energy + its *unit* are produced
///   (Bruker eV-per-scan vs Thermo NCE-per-window), so a fragment predictor can
///   reject a unit it was not calibrated for.
///
/// The Bruker default reproduces current behaviour exactly; `astral` is the first
/// non-IMS instrument. This is pure configuration — it holds no analyte state.
#[derive(Debug, Clone, Copy)]
pub struct InstrumentConfig {
    pub kind: InstrumentKind,
    pub capabilities: InstrumentCapabilities,
    pub mobility: MobilityModality,
    pub activation: ActivationPolicy,
}

impl InstrumentConfig {
    /// Bruker timsTOF DDA-PASEF: TIMS mobility, full capabilities, eV CE linear in
    /// scan. `ce_bias`/`ce_slope` reproduce the legacy `dda_selection_scheme`
    /// formula (defaults 54.1984 / -0.0345). Matches the pre-dispatch behaviour.
    pub fn bruker_pasef(ce_bias: f64, ce_slope: f64) -> Self {
        InstrumentConfig {
            kind: InstrumentKind::TimsTofDia,
            capabilities: InstrumentCapabilities::bruker_timstof(),
            mobility: MobilityModality::Tims,
            activation: ActivationPolicy::bruker_pasef(ce_bias, ce_slope),
        }
    }

    /// Orbitrap Astral: no mobility axis (events collapse to a single `Scan`),
    /// Astral capabilities (both false — isotope transmission forced to `None`),
    /// and a Thermo NCE activation policy carrying the per-window normalized CE.
    ///
    /// Takes the per-window collision-energy *policy* (not a full
    /// [`ActivationPolicy`]) and constructs the NCE activation internally, so an
    /// Astral config can never be built with a contradictory eV / scan-dependent
    /// (e.g. Bruker-PASEF) activation model. `ce` is in NCE units by construction.
    pub fn astral(ce: crate::sim::scheme::CollisionEnergyPolicy) -> Self {
        InstrumentConfig {
            kind: InstrumentKind::OrbitrapAstral,
            capabilities: InstrumentCapabilities::astral(),
            mobility: MobilityModality::None,
            activation: ActivationPolicy::thermo_nce(ce),
        }
    }

    /// Whether events render as a single collapsed [`RenderedEvent::Scan`] (no
    /// mobility axis) rather than a [`RenderedEvent::MobilityFrame`].
    pub fn is_scan_based(&self) -> bool {
        self.mobility == MobilityModality::None
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
        // Unspecified events split the remaining cycle time; if the explicit
        // durations leave nothing for them, the schedule would emit zero-exposure
        // (no-signal) events — reject it rather than silently drop their signal.
        let per_unspecified = if n_unspecified > 0 {
            let remaining = cycle_time_s - explicit_sum;
            if remaining <= 1e-12 {
                return Err(format!(
                    "explicit durations ({explicit_sum}) leave no time for {n_unspecified} \
                     unspecified event(s) in cycle_time_s {cycle_time_s}"
                ));
            }
            remaining / n_unspecified as f64
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

/// Parameters the projector needs to reproduce / improve the legacy distribution
/// jobs. Defaults mirror the simulator's config defaults (`target_p=0.999`,
/// `sampling_step_size=0.001`, scan step `0.0001`, `n_steps=1000`,
/// `remove_epsilon=1e-4`).
#[derive(Debug, Clone, Copy)]
pub struct ProjectionParams {
    pub target_p: f64,
    pub frame_step_size: f64,
    pub scan_step_size: f64,
    pub n_steps: Option<usize>,
    pub remove_epsilon: f64,
    pub num_threads: usize,
    /// Decimal places the legacy / `project_distributions` writer rounds stored
    /// abundances to (`python_list_to_json_string` num_decimals, default 4).
    /// LegacyCompat rounds to this so projector-fed rendering byte-matches the
    /// columns a DB was written with. Accurate ignores it (full precision).
    pub num_decimals: u32,
}

impl Default for ProjectionParams {
    fn default() -> Self {
        ProjectionParams {
            target_p: 0.999,
            frame_step_size: 0.001,
            scan_step_size: 0.0001,
            n_steps: Some(1000),
            remove_epsilon: 1e-4,
            num_threads: 4,
            num_decimals: 4,
        }
    }
}

/// Where the per-analyte occurrence/abundance distributions come from when
/// constructing the builder entities (P4 canonical-state contract).
///
/// `Columns` parses them from the legacy JSON columns (the default — byte
/// behaviour unchanged). `Projector` computes them at load time from the scalar
/// trunk entities, so the columns become unnecessary (the P4 goal). Both paths
/// produce identical-shape `PeptidesSim`/`IonSim`, so every downstream consumer
/// (maps, transmission, DDA, fragment-intensity) is untouched.
#[derive(Debug, Clone, Copy)]
pub enum DistributionSource {
    Columns,
    Projector { mode: ProjectionMode, env: MobilityEnv, params: ProjectionParams },
}

// --------------------------------------------------------------------------- //
// LegacyCompat projections (parity target — mirror the legacy kernels exactly)
// --------------------------------------------------------------------------- //

/// LegacyCompat time projection: reproduce `frame_occurrence`/`frame_abundance`
/// over the **full** frames table, exactly as `simulate_frame_distributions_emg`
/// did. Inputs are taken at **f64** (the precision the legacy pipeline used) —
/// NOT the f32 trunk entities — so the result is bit-faithful: `rt_mus`,
/// `rt_sigmas`, `rt_lambdas` are per-peptide EMG params (`rt_mu`, not the GRU
/// apex); `frame_ids`/`frame_times` are the whole frames table ascending by id.
/// Occurrence positions (1-indexed into `frame_times`) are translated to frame
/// ids via `frame_ids[pos-1]`; the `remove_epsilon` filter drops frames whose
/// abundance is `<= remove_epsilon` (matching the legacy job). Returns
/// `(frame_id, abundance)` per peptide.
pub fn project_time_legacy(
    rt_mus: &[f64],
    rt_sigmas: &[f64],
    rt_lambdas: &[f64],
    frame_ids: &[u32],
    frame_times: &[f64],
    rt_cycle_length: f64,
    target_p: f64,
    step_size: f64,
    n_steps: Option<usize>,
    remove_epsilon: f64,
    num_threads: usize,
) -> Vec<Vec<(u32, f64)>> {
    assert_eq!(frame_ids.len(), frame_times.len(), "frame_ids/frame_times length mismatch");
    let n = frame_ids.len();
    let nt = num_threads.max(1);
    // Batched + parallel, exactly the kernels the legacy job used.
    let positions = calculate_frame_occurrences_emg_par(
        frame_times,
        rt_mus.to_vec(),
        rt_sigmas.to_vec(),
        rt_lambdas.to_vec(),
        target_p,
        step_size,
        nt,
        n_steps,
    );
    // Translate 1-indexed positions -> frame ids (robust to non-1..N).
    let occ_ids: Vec<Vec<i32>> = positions
        .iter()
        .map(|ps| {
            ps.iter()
                .map(|&p| {
                    assert!(p >= 1 && (p as usize) <= n, "occurrence position {p} out of range");
                    frame_ids[(p - 1) as usize] as i32
                })
                .collect()
        })
        .collect();
    // Abundance keyed by frame id (we pass ids as occurrences).
    let time_map: HashMap<i32, f64> =
        frame_ids.iter().zip(frame_times).map(|(&id, &t)| (id as i32, t)).collect();
    let abund = calculate_frame_abundances_emg_par(
        &time_map,
        occ_ids.clone(),
        rt_mus.to_vec(),
        rt_sigmas.to_vec(),
        rt_lambdas.to_vec(),
        rt_cycle_length,
        nt,
        n_steps,
    );
    occ_ids
        .into_iter()
        .zip(abund)
        .map(|(ids, ab)| {
            ids.into_iter()
                .map(|id| id as u32)
                .zip(ab)
                .filter(|(_, a)| *a > remove_epsilon)
                .collect()
        })
        .collect()
}

/// Batched + parallel LegacyCompat scan projection over many ions (the kernels
/// the legacy job used). `means`/`sigmas` are the original per-ion 1/K0
/// mean+std; `scan_ids`/`scan_mobilities` ascending+aligned (shared by all ions).
/// Returns one `(scan, abundance)` list per ion. Requires positive sigmas (real
/// data); use [`project_mobility_ion_legacy`] for the point-mass-guarded single-
/// ion path.
pub fn project_mobility_legacy_par(
    means: &[f64],
    sigmas: &[f64],
    scan_ids: &[u32],
    scan_mobilities: &[f64],
    im_cycle_length: f64,
    target_p: f64,
    step_size: f64,
    num_threads: usize,
) -> Vec<Vec<(i32, f64)>> {
    assert_eq!(scan_ids.len(), scan_mobilities.len(), "scan_ids/mobilities length mismatch");
    let nt = num_threads.max(1);
    let occ = calculate_scan_occurrences_gaussian_par(
        scan_mobilities,
        means.to_vec(),
        sigmas.to_vec(),
        target_p,
        step_size,
        5.0,
        5.0,
        nt,
    );
    let time_map: HashMap<i32, f64> =
        scan_ids.iter().zip(scan_mobilities).map(|(&id, &m)| (id as i32, m)).collect();
    let abund = calculate_scan_abundances_gaussian_par(
        &time_map,
        occ.clone(),
        means.to_vec(),
        sigmas.to_vec(),
        im_cycle_length,
        nt,
    );
    occ.into_iter()
        .zip(abund)
        .map(|(o, a)| o.into_iter().zip(a).collect())
        .collect()
}

/// LegacyCompat mobility projection for one ion: reproduce
/// `scan_occurrence`/`scan_abundance` exactly as
/// `simulate_scan_distributions_with_variance` did — `calculate_scan_occurrence_
/// gaussian` with `n_lower/upper = 5`, and `calculate_abundance_gaussian` over
/// the fixed `im_cycle_length` bin. `mean`/`sigma` are the **original** stored
/// 1/K0 mean + std at f64 (NOT a CCS round-trip). `scan_mobilities` must be
/// **ascending** with `scan_ids` aligned (exactly as the legacy job feeds `scans`
/// sorted so `im_cycle_length > 0`); a descending grid makes the occurrence
/// kernel return nothing. The occurrence kernel's indices are the values the
/// legacy job stored directly, so they are returned as-is. Returns
/// `(scan, abundance)`.
pub fn project_mobility_ion_legacy(
    mean: f64,
    sigma: f64,
    scan_ids: &[u32],
    scan_mobilities: &[f64],
    im_cycle_length: f64,
    target_p: f64,
    step_size: f64,
) -> Vec<(i32, f64)> {
    assert_eq!(scan_ids.len(), scan_mobilities.len(), "scan_ids/mobilities length mismatch");
    if scan_ids.is_empty() {
        return Vec::new();
    }
    // Point mass (zero/invalid spread): single nearest scan, no CDF div-by-zero.
    if !(sigma > 0.0) {
        let idx = nearest_scan_ascending(scan_mobilities, mean);
        return vec![(scan_ids[idx] as i32, 1.0)];
    }
    let occ =
        calculate_scan_occurrence_gaussian(scan_mobilities, mean, sigma, target_p, step_size, 5.0, 5.0);
    let time_map: HashMap<i32, f64> =
        scan_ids.iter().zip(scan_mobilities).map(|(&id, &m)| (id as i32, m)).collect();
    let abund = calculate_abundance_gaussian(&time_map, &occ, mean, sigma, im_cycle_length);
    occ.into_iter().zip(abund).collect()
}

/// Index of the grid entry nearest to `value` (grid assumed non-empty).
fn nearest_scan_ascending(grid: &[f64], value: f64) -> usize {
    grid.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (**a - value)
                .abs()
                .partial_cmp(&(**b - value).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// --------------------------------------------------------------------------- //
// Projections (Accurate)
// --------------------------------------------------------------------------- //

/// Time projection: for each peptide, the `(global_event_index, abundance)` list
/// over **every** acquisition event in the run (MS1 and MS2 alike), integrating
/// the EMG over each event's true `[start, end]` exposure interval. The index is
/// the position in `timeline.events` (i.e. `EventSlot::global_index`), so MS2
/// events receive their own RT abundance rather than reusing the MS1 value — this
/// matches the legacy `frame_occurrence`/`frame_abundance`, which span all frames.
pub fn project_time(
    peptides: &[PeptideScalar],
    timeline: &EventTimeline,
    target_p: f64,
    step_size: f64,
    num_threads: usize,
) -> Vec<Vec<(usize, f64)>> {
    // Every event's interval, indexed by global event position.
    let intervals: Vec<(f64, f64)> = timeline.events.iter().map(|e| e.interval).collect();
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
        MobilityModality::Tims => project_mobility_gaussian(
            &geometry.inv_mobility,
            ion.inv_mobility(env),
            ion.inv_mobility_std as f64,
            target_p,
            step_size,
        ),
    }
}

/// Accurate mobility projection core: a Gaussian (`mean`, `sigma`) in 1/K0 onto
/// an **ascending** mobility grid, returning `(ascending_scan_index, abundance)`
/// where each occupied scan's abundance is the CDF over that scan's own
/// midpoint-bounded bin (correct on non-uniform grids). σ ≤ 0 is a point mass at
/// the nearest scan; empty/singleton grids are handled.
pub fn project_mobility_gaussian(
    grid_ascending: &[f64],
    mean: f64,
    sigma: f64,
    target_p: f64,
    step_size: f64,
) -> Vec<(i32, f64)> {
    let n = grid_ascending.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        if !(sigma > 0.0) {
            return vec![(0, 1.0)];
        }
        let a = normal_cdf_range(mean - 6.0 * sigma, mean + 6.0 * sigma, mean, sigma);
        return vec![(0, a)];
    }
    // Point mass (zero/invalid spread): single nearest scan, no CDF div-by-zero.
    if !(sigma > 0.0) {
        let scan = nearest_scan_ascending(grid_ascending, mean);
        return vec![(scan as i32, 1.0)];
    }
    // `calculate_scan_occurrence_gaussian` returns indices into the REVERSED
    // (descending) grid: occ index i -> grid_ascending[n-1-i].
    let occ =
        calculate_scan_occurrence_gaussian(grid_ascending, mean, sigma, target_p, step_size, 3.0, 3.0);
    let rev: Vec<f64> = grid_ascending.iter().rev().copied().collect();
    let mut out: Vec<(i32, f64)> = occ
        .into_iter()
        .map(|i| {
            let i = i as usize;
            let v = rev[i];
            // Bin edges = midpoints to neighbours on the descending grid;
            // endpoints extrapolate by their adjacent half-spacing.
            let hi = if i > 0 { (rev[i - 1] + v) / 2.0 } else { v + (v - rev[1]) / 2.0 };
            let lo = if i + 1 < n { (v + rev[i + 1]) / 2.0 } else { v - (rev[i - 1] - v) / 2.0 };
            let abundance = normal_cdf_range(lo, hi, mean, sigma);
            ((n - 1 - i) as i32, abundance) // reversed index -> ascending position
        })
        .collect();
    out.sort_by_key(|(scan, _)| *scan);
    out
}

/// Batched + parallel Accurate mobility projection over many ions sharing one
/// ascending mobility grid. `means`/`sigmas` are per-ion 1/K0 mean+std. Returns
/// one `(ascending_scan_index, abundance)` list per ion.
pub fn project_mobility_accurate_par(
    means: &[f64],
    sigmas: &[f64],
    grid_ascending: &[f64],
    target_p: f64,
    step_size: f64,
    num_threads: usize,
) -> Vec<Vec<(i32, f64)>> {
    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;
    let pool = ThreadPoolBuilder::new().num_threads(num_threads.max(1)).build().unwrap();
    pool.install(|| {
        means
            .par_iter()
            .zip(sigmas.par_iter())
            .map(|(&m, &s)| project_mobility_gaussian(grid_ascending, m, s, target_p, step_size))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::scheme::{
        Analyzer, CollisionEnergyPolicy, DiaGeometry, DiaMs2Frame, DiaWindow,
        EnergyUnit, InstrumentKind, Ms1Event, Provenance, SchemeSource,
    };
    use mscore::data::spectrum::MzSpectrum;

    #[test]
    fn instrument_config_bundles_the_dispatch_axes() {
        // P6c: Bruker config = TIMS mobility, full capabilities, eV CE, frames.
        let bruker = InstrumentConfig::bruker_pasef(54.1984, -0.0345);
        assert_eq!(bruker.kind, InstrumentKind::TimsTofDia);
        assert_eq!(bruker.mobility, MobilityModality::Tims);
        assert!(bruker.capabilities.has_tims_mobility);
        assert!(bruker.capabilities.has_quad_isotope_transmission);
        assert!(!bruker.is_scan_based());
        assert_eq!(bruker.activation.unit, EnergyUnit::ElectronVolt);
        assert_eq!(bruker.activation.collision_energy_for_scan(250), Some(54.1984 - 0.0345 * 250.0));

        // Astral config = no mobility (scan-based), both capabilities off, NCE.
        // The constructor builds the NCE activation internally from the CE policy,
        // so a contradictory eV/scan-dependent activation is unrepresentable.
        let astral = InstrumentConfig::astral(CollisionEnergyPolicy::Value(27.0));
        assert_eq!(astral.kind, InstrumentKind::OrbitrapAstral);
        assert_eq!(astral.mobility, MobilityModality::None);
        assert!(!astral.capabilities.has_tims_mobility);
        assert!(!astral.capabilities.has_quad_isotope_transmission);
        assert!(astral.is_scan_based());
        assert_eq!(astral.activation.unit, EnergyUnit::NormalizedCe);
        // No IMS: the Astral policy has no scan-parameterised CE.
        assert_eq!(astral.activation.collision_energy_for_scan(250), None);
    }

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

    fn peptide(rt: f64, sigma: f64, lambda: f64) -> PeptideScalar {
        use mscore::data::peptide::PeptideSequence;
        PeptideScalar {
            protein_id: 0,
            peptide_id: 1,
            sequence: PeptideSequence::new("PEPTIDEK".into(), Some(1)),
            proteins: "P".into(),
            decoy: false,
            missed_cleavages: 0,
            n_term: None,
            c_term: None,
            mono_isotopic_mass: 930.0,
            retention_time: rt as f32,
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
        // project_time now spans ALL events (MS1+MS2), returning global indices.
        // RT-support truncation: touches some but NOT all events.
        assert!(!hits.is_empty(), "peptide must touch some events");
        assert!(hits.len() < tl.events.len(), "RT support must be truncated, not all events");
        // Touched events are a contiguous global-index range with positive abundance.
        for w in hits.windows(2) {
            assert_eq!(w[1].0, w[0].0 + 1, "touched events must be contiguous in global index");
        }
        for (_, abund) in hits {
            assert!(*abund > 0.0);
        }
    }

    fn ion(ccs: f64, mz: f64, charge: i8, std: f64) -> IonScalar {
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
        // f64 inputs (LegacyCompat takes legacy-precision params directly).
        let (mu, sigma, lambda) = (3.0_f64, 0.4_f64, 0.6_f64);
        let out = project_time_legacy(
            &[mu], &[sigma], &[lambda], &frame_ids, &frame_times, rt_cycle, 0.9999, 0.01, None, -1.0, 1,
        );

        // Raw kernel call with identical inputs + the same position->id mapping.
        let positions = calculate_frame_occurrence_emg(&frame_times, mu, sigma, lambda, 0.9999, 0.01, None);
        let occ_ids: Vec<i32> = positions.iter().map(|&p| frame_ids[(p - 1) as usize] as i32).collect();
        let tmap: std::collections::HashMap<i32, f64> =
            frame_ids.iter().zip(&frame_times).map(|(&id, &t)| (id as i32, t)).collect();
        let abund = calculate_frame_abundance_emg(&tmap, &occ_ids, mu, sigma, lambda, rt_cycle, None);
        let expected: Vec<(u32, f64)> = occ_ids.iter().map(|&id| id as u32).zip(abund).collect();
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
        let mean = i.inv_mobility(&env);
        let sigma = i.inv_mobility_std as f64;
        let out = project_mobility_ion_legacy(mean, sigma, &scan_ids, &scan_mob, im_cycle, 0.9999, 0.0001);

        let occ = calculate_scan_occurrence_gaussian(&scan_mob, mean, sigma, 0.9999, 0.0001, 5.0, 5.0);
        let tmap: std::collections::HashMap<i32, f64> =
            scan_ids.iter().zip(&scan_mob).map(|(&id, &m)| (id as i32, m)).collect();
        let abund = calculate_abundance_gaussian(&tmap, &occ, mean, sigma, im_cycle);
        let expected: Vec<(i32, f64)> = occ.into_iter().zip(abund).collect();
        assert_eq!(out, expected, "LegacyCompat mobility must mirror the raw kernels");
        assert!(!out.is_empty());
    }

    #[test]
    fn timeline_rejects_explicit_durations_starving_unspecified_events() {
        // MS1 duration == cycle_time leaves 0 for the two unspecified MS2 events.
        let mut scheme = scheme_one_ms1_two_ms2();
        if let AcquisitionEvent::Ms1(ref mut m) = scheme.cycle[0] {
            m.duration_s = Some(1.2); // == cycle_time_s
        }
        assert!(EventTimeline::from_scheme(&scheme).is_err());
    }

    #[test]
    fn project_mobility_zero_sigma_is_point_mass_not_nan() {
        let grid: Vec<f64> = (0..100).map(|i| 1.005 + i as f64 * 0.005).collect();
        let geom = SamplingGeometry::tims(grid);
        let env = MobilityEnv::default();
        let i = ion(450.0, 600.0, 2, 0.0); // zero std
        let m = project_mobility_ion(&i, &geom, &env, 0.9999, 0.01);
        assert_eq!(m.len(), 1, "zero-sigma -> single point-mass scan");
        assert!(m[0].1.is_finite() && (m[0].1 - 1.0).abs() < 1e-12);
        // Legacy path likewise.
        let scan_ids: Vec<u32> = (0..100).rev().collect();
        let scan_mob: Vec<f64> = (0..100).map(|i| 1.005 + i as f64 * 0.005).collect();
        let mean = i.inv_mobility(&env);
        let lm = project_mobility_ion_legacy(mean, 0.0, &scan_ids, &scan_mob, 0.005, 0.9999, 0.0001);
        assert_eq!(lm.len(), 1);
        assert!(lm[0].1.is_finite());
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
