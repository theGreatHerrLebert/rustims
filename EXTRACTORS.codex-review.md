Reading additional input from stdin...
OpenAI Codex v0.136.0
--------
workdir: /scratch/timsim-demo/SUBMISSION/rustims
model: gpt-5.5
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: medium
reasoning summaries: none
session id: 019eab08-de8e-7470-b5f1-79100a6009db
--------
user
Review the NEW code added to these two Rust files for a mass-spec simulator (TimSim): in scheme.rs, the extractors AcquisitionScheme::from_bruker_d and AcquisitionScheme::from_sciex_wiff (the data model, validate(), num_cycles, windows(), from_window_table, and from_thermo_raw were ALREADY reviewed — only skim them for interaction bugs). The second file is sciexwiff's new library API (read_method).

Trusted context: rustdf reads Bruker TDF natively — read_dia_ms_ms_windows(folder)->Vec<DiaMsMsWindow{window_group,scan_num_begin,scan_num_end,isolation_mz,isolation_width,collision_energy}> and read_meta_data_sql(folder)->Vec<FrameMeta{id,time:f64,ms_ms_type:i64,...}> (MsMsType==0 = MS1/precursor frame). from_bruker_d groups windows by window_group into DiaMs2Frame{TimsMobility windows} and derives cycle time from precursor-frame spacing; verified to extract 15 frames/36 windows from a real DIA-PASEF .d. from_sciex_wiff maps SWATH windows to single MzOnly frames with CE Unknown (SCIEX rolling CE not in the method) and caller-supplied timing; verified 60 windows on a real ZenoTOF .wiff. validate() (already reviewed) requires exactly one MS1 first then >=1 MS2 frame, finite/positive widths, window edges within mz_range, CE finite>=0 for Value/resolved-Linear, multi-window frames need TimsMobility on every window and no within-frame m/z+mobility overlap, start<=gradient. sciexwiff::read_method opens a .wiff OLE2 (cfb crate) and parses SWATHMethod (20B records from off 40) + TOFCalibrationData.

Focus: 1) from_bruker_d correctness — window-group grouping, cycle-time from precursor spacing (sorted; what if precursor times equal/unsorted/duplicated, or DIA uses a different MsMsType for MS1?), start/gradient via fold over f64 (NaN handling), whether the produced scheme always passes validate() (e.g. within-frame overlap on real PASEF groups; windows-within-mz_range by construction), empty/edge frames. 2) from_sciex_wiff — caller-supplied timing validation, window mapping, the Unknown-CE contract, mz_range. 3) sciexwiff read_method — bounds/stride safety on untrusted .wiff bytes, partial/short streams, the SWATHMethod record count derivation, calibration optionality. 4) any panic on malformed vendor files, and any case where an extractor returns a scheme that then fails validate() (surprising for a caller). Concrete, ranked by severity, cap ~700 words.

<stdin>
===== FILE: rustdf/src/sim/scheme.rs =====
//! Vendor-neutral DIA acquisition **scheme** (input/design side).
//!
//! An [`AcquisitionScheme`] describes one DIA cycle as an *ordered sequence of
//! physical events* ([`AcquisitionEvent`]) and how that cycle tiles the
//! gradient ([`RepeatPolicy`]). It is the design counterpart to the
//! [`crate::sim::acquisition::AcquisitionWriter`] (output side): the simulator
//! generates scans for the scheme, and a writer materializes them.
//!
//! The key modelling choice (per review) is that a *physical acquisition unit*
//! is explicit: [`AcquisitionEvent::DiaMs2Frame`] carries a `Vec<DiaWindow>`, so
//! a timsTOF MS2 frame holding several mobility-partitioned windows and a linear
//! Astral/SCIEX MS2 scan (one window) are both represented without overloading a
//! `window_group` id to imply simultaneity.

use std::io;

/// Current scheme schema version.
pub const SCHEME_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstrumentKind {
    TimsTofDia,
    OrbitrapAstral,
    SciexZenoTof,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Analyzer {
    Ftms,
    Astms,
    Tof,
    Itms,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataMode {
    Profile,
    Centroid,
}

/// An m/z isolation window (no ion mobility — see [`DiaGeometry`]).
#[derive(Clone, Copy, Debug)]
pub struct IsolationWindow {
    pub center_mz: f64,
    pub width_mz: f64,
}

impl IsolationWindow {
    pub fn lower(&self) -> f64 {
        self.center_mz - self.width_mz / 2.0
    }
    pub fn upper(&self) -> f64 {
        self.center_mz + self.width_mz / 2.0
    }
}

/// How a window's collision energy is determined.
///
/// `Value` covers both a per-window CE and a scheme-wide fixed CE (every window
/// carries the same value) — there is intentionally no separate `Fixed` variant
/// (they were structurally identical). `Unknown` means extraction could not
/// recover it (e.g. SCIEX rolling CE) and a model must be supplied downstream —
/// it is never silently invented.
#[derive(Clone, Copy, Debug)]
pub enum CollisionEnergyPolicy {
    Value(f64),
    Linear { intercept: f64, slope_per_mz: f64 },
    Unknown,
}

impl CollisionEnergyPolicy {
    /// Resolve the CE for a window centered at `center_mz`, if determinable.
    pub fn at(&self, center_mz: f64) -> Option<f64> {
        match *self {
            CollisionEnergyPolicy::Value(v) => Some(v),
            CollisionEnergyPolicy::Linear {
                intercept,
                slope_per_mz,
            } => Some(intercept + slope_per_mz * center_mz),
            CollisionEnergyPolicy::Unknown => None,
        }
    }
}

/// Window geometry: m/z only, or (timsTOF) an additional ion-mobility partition.
/// Mobility is kept off [`IsolationWindow`] so "mobility on a non-IMS instrument"
/// is unrepresentable. Scan numbers are Bruker-grid coordinates (they require the
/// reference dataset's calibration to map to physical mobility).
#[derive(Clone, Copy, Debug)]
pub enum DiaGeometry {
    MzOnly,
    TimsMobility { scan_start: u32, scan_end: u32 },
}

/// One DIA isolation window within a frame.
#[derive(Clone, Copy, Debug)]
pub struct DiaWindow {
    pub isolation: IsolationWindow,
    pub collision_energy: CollisionEnergyPolicy,
    pub geometry: DiaGeometry,
}

/// An MS1 (precursor) acquisition.
#[derive(Clone, Debug)]
pub struct Ms1Event {
    pub analyzer: Analyzer,
    pub data_mode: DataMode,
    /// The MS1 acquisition m/z range, if known. `None` when not recoverable
    /// (e.g. Thermo `scan_event` does not carry it — it must not be confused
    /// with the DIA window coverage in [`AcquisitionScheme::mz_range`]).
    pub mz_range: Option<(f64, f64)>,
    pub duration_s: Option<f64>,
}

/// One physical MS2 acquisition unit. Holds one window for Astral/SCIEX; several
/// mobility-partitioned windows (sharing the frame) for timsTOF.
#[derive(Clone, Debug)]
pub struct DiaMs2Frame {
    pub windows: Vec<DiaWindow>,
    pub analyzer: Analyzer,
    pub data_mode: DataMode,
    pub duration_s: Option<f64>,
}

#[derive(Clone, Debug)]
pub enum AcquisitionEvent {
    Ms1(Ms1Event),
    DiaMs2Frame(DiaMs2Frame),
}

/// How the cycle repeats over the gradient.
#[derive(Clone, Copy, Debug)]
pub enum RepeatPolicy {
    /// The cycle repeats every `cycle_time_s`, starting at `start_time_s`, until
    /// `gradient_length_s`. Per-event `duration_s` (when present) takes
    /// precedence for fine RT placement; otherwise events are spread across the
    /// cycle uniformly.
    FixedCycleTime {
        cycle_time_s: f64,
        gradient_length_s: f64,
        start_time_s: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchemeSource {
    ExtractedBruker,
    ExtractedThermo,
    ExtractedSciex,
    UserTable,
    Programmatic,
}

/// Where the scheme came from (scheme-level attribution; not per-field).
#[derive(Clone, Debug)]
pub struct Provenance {
    pub source: SchemeSource,
    pub notes: String,
}

/// A vendor-neutral DIA acquisition design.
#[derive(Clone, Debug)]
pub struct AcquisitionScheme {
    pub version: u16,
    pub instrument: InstrumentKind,
    pub cycle: Vec<AcquisitionEvent>,
    pub repeat: RepeatPolicy,
    pub mz_range: (f64, f64),
    pub provenance: Provenance,
}

impl AcquisitionScheme {
    /// Iterate every DIA window across all MS2 frames in the cycle.
    pub fn windows(&self) -> impl Iterator<Item = &DiaWindow> {
        self.cycle
            .iter()
            .flat_map(|e| match e {
                AcquisitionEvent::DiaMs2Frame(f) => f.windows.as_slice(),
                AcquisitionEvent::Ms1(_) => &[][..],
            }
            .iter())
    }

    /// Number of MS1 events in one cycle.
    pub fn ms1_count(&self) -> usize {
        self.cycle
            .iter()
            .filter(|e| matches!(e, AcquisitionEvent::Ms1(_)))
            .count()
    }

    /// Number of full cycles that fit the gradient (if derivable).
    pub fn num_cycles(&self) -> Option<u64> {
        match self.repeat {
            RepeatPolicy::FixedCycleTime {
                cycle_time_s,
                gradient_length_s,
                start_time_s,
            } if cycle_time_s.is_finite()
                && cycle_time_s > 0.0
                && gradient_length_s.is_finite()
                && start_time_s.is_finite() =>
            {
                Some(((gradient_length_s - start_time_s).max(0.0) / cycle_time_s) as u64)
            }
            _ => None,
        }
    }

    /// Build a scheme from an explicit window list (injected / CSV). One MS1
    /// followed by one single-window MS2 frame per window.
    pub fn from_window_table(
        instrument: InstrumentKind,
        ms1: Ms1Event,
        windows: Vec<DiaWindow>,
        repeat: RepeatPolicy,
        mz_range: (f64, f64),
    ) -> Self {
        let mut cycle = vec![AcquisitionEvent::Ms1(ms1.clone())];
        for w in windows {
            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: vec![w],
                analyzer: ms1.analyzer,
                data_mode: DataMode::Centroid,
                duration_s: None,
            }));
        }
        AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument,
            cycle,
            repeat,
            mz_range,
            provenance: Provenance {
                source: SchemeSource::UserTable,
                notes: "built from explicit window table".to_string(),
            },
        }
    }

    /// Validate internal consistency. Returns a human-readable error on the first
    /// problem found.
    pub fn validate(&self) -> Result<(), String> {
        if self.version != SCHEME_VERSION {
            return Err(format!(
                "unsupported scheme version {} (this build supports {})",
                self.version, SCHEME_VERSION
            ));
        }
        if self.cycle.is_empty() {
            return Err("empty cycle".into());
        }
        let (lo, hi) = self.mz_range;
        if !(lo.is_finite() && hi.is_finite()) || lo >= hi {
            return Err(format!("invalid mz_range ({lo}, {hi})"));
        }
        // Structure: exactly one MS1, as the first event, then >= 1 MS2 frame.
        if !matches!(self.cycle.first(), Some(AcquisitionEvent::Ms1(_))) {
            return Err("cycle must begin with an MS1 event".into());
        }
        if self.ms1_count() != 1 {
            return Err(format!(
                "cycle must contain exactly one MS1 event (has {})",
                self.ms1_count()
            ));
        }
        if !self
            .cycle
            .iter()
            .any(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_)))
        {
            return Err("cycle has no MS2 frames".into());
        }

        let check_dur = |d: Option<f64>, what: &str| -> Result<(), String> {
            match d {
                Some(v) if !(v.is_finite() && v > 0.0) => {
                    Err(format!("{what} duration must be finite and > 0"))
                }
                _ => Ok(()),
            }
        };

        for ev in &self.cycle {
            match ev {
                AcquisitionEvent::Ms1(m) => {
                    check_dur(m.duration_s, "MS1")?;
                    if let Some((a, b)) = m.mz_range {
                        if !(a.is_finite() && b.is_finite()) || a >= b {
                            return Err(format!("invalid MS1 mz_range ({a}, {b})"));
                        }
                    }
                }
                AcquisitionEvent::DiaMs2Frame(frame) => {
                    check_dur(frame.duration_s, "MS2 frame")?;
                    if frame.windows.is_empty() {
                        return Err("MS2 frame has no windows".into());
                    }
                    let multi = frame.windows.len() > 1;
                    if multi && self.instrument != InstrumentKind::TimsTofDia {
                        return Err(
                            "multi-window MS2 frame is only valid for timsTOF (mobility-partitioned)"
                                .into(),
                        );
                    }
                    for (i, w) in frame.windows.iter().enumerate() {
                        if !(w.isolation.width_mz.is_finite() && w.isolation.width_mz > 0.0) {
                            return Err("window width must be finite and > 0".into());
                        }
                        if !w.isolation.center_mz.is_finite()
                            || !w.isolation.lower().is_finite()
                            || !w.isolation.upper().is_finite()
                        {
                            return Err("window m/z is not finite".into());
                        }
                        if w.isolation.lower() < lo - 1e-6 || w.isolation.upper() > hi + 1e-6 {
                            return Err(format!(
                                "window [{:.4}, {:.4}] outside mz_range [{lo:.4}, {hi:.4}]",
                                w.isolation.lower(),
                                w.isolation.upper()
                            ));
                        }
                        match w.collision_energy {
                            CollisionEnergyPolicy::Value(v) => {
                                if !v.is_finite() || v < 0.0 {
                                    return Err("collision energy must be finite and >= 0".into());
                                }
                            }
                            CollisionEnergyPolicy::Linear {
                                intercept,
                                slope_per_mz,
                            } => {
                                if !intercept.is_finite() || !slope_per_mz.is_finite() {
                                    return Err("non-finite linear CE coefficients".into());
                                }
                                match w.collision_energy.at(w.isolation.center_mz) {
                                    Some(ce) if ce.is_finite() && ce >= 0.0 => {}
                                    _ => {
                                        return Err(
                                            "linear CE resolves to non-finite or negative".into()
                                        )
                                    }
                                }
                            }
                            CollisionEnergyPolicy::Unknown => {}
                        }
                        match w.geometry {
                            DiaGeometry::TimsMobility {
                                scan_start,
                                scan_end,
                            } => {
                                if self.instrument != InstrumentKind::TimsTofDia {
                                    return Err("mobility geometry is only valid for timsTOF".into());
                                }
                                if scan_start > scan_end {
                                    return Err("mobility scan_start > scan_end".into());
                                }
                            }
                            DiaGeometry::MzOnly => {
                                if multi {
                                    return Err(
                                        "multi-window timsTOF frame requires mobility geometry on every window"
                                            .into(),
                                    );
                                }
                            }
                        }
                        // Within one frame, windows must not overlap in BOTH m/z
                        // and mobility (overlap across sequential frames is fine).
                        for w2 in &frame.windows[..i] {
                            let mz_overlap = w.isolation.lower() < w2.isolation.upper()
                                && w2.isolation.lower() < w.isolation.upper();
                            let im_overlap = match (w.geometry, w2.geometry) {
                                (
                                    DiaGeometry::TimsMobility {
                                        scan_start: a0,
                                        scan_end: a1,
                                    },
                                    DiaGeometry::TimsMobility {
                                        scan_start: b0,
                                        scan_end: b1,
                                    },
                                ) => a0 <= b1 && b0 <= a1,
                                _ => true, // m/z-only windows share all mobility
                            };
                            if mz_overlap && im_overlap {
                                return Err(
                                    "windows within one frame overlap in both m/z and mobility"
                                        .into(),
                                );
                            }
                        }
                    }
                }
            }
        }
        match self.repeat {
            RepeatPolicy::FixedCycleTime {
                cycle_time_s,
                gradient_length_s,
                start_time_s,
            } => {
                if !(cycle_time_s.is_finite() && cycle_time_s > 0.0) {
                    return Err("cycle_time_s must be finite and > 0".into());
                }
                if !(gradient_length_s.is_finite() && gradient_length_s > 0.0) {
                    return Err("gradient_length_s must be finite and > 0".into());
                }
                if !start_time_s.is_finite() || start_time_s < 0.0 {
                    return Err("start_time_s must be finite and >= 0".into());
                }
                if start_time_s > gradient_length_s {
                    return Err("start_time_s must be <= gradient_length_s".into());
                }
            }
        }
        Ok(())
    }
}

impl AcquisitionScheme {
    /// Extract the acquisition scheme from a real Bruker timsTOF DIA `.d`.
    ///
    /// Reads `DiaFrameMsMsWindows` and the frame table. Each **window group**
    /// becomes one [`DiaMs2Frame`] holding its mobility-partitioned windows
    /// (`TimsMobility` geometry — the scan ranges are Bruker-grid coordinates),
    /// preceded by a single MS1 event. Cycle timing comes from the spacing of
    /// the precursor (MS1) frames.
    pub fn from_bruker_d<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
        use crate::data::meta::{read_dia_ms_ms_windows, read_meta_data_sql};
        use std::collections::BTreeMap;

        let folder = path.as_ref().to_string_lossy().into_owned();
        let to_io =
            |e: Box<dyn std::error::Error>| io::Error::new(io::ErrorKind::InvalidData, e.to_string());
        let windows = read_dia_ms_ms_windows(&folder).map_err(to_io)?;
        let frames = read_meta_data_sql(&folder).map_err(to_io)?;
        if windows.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "no DiaFrameMsMsWindows rows (not a DIA .d?)",
            ));
        }

        // Group windows by window group (ascending); each group is a frame.
        let mut by_group: BTreeMap<u32, Vec<DiaWindow>> = BTreeMap::new();
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for w in &windows {
            let dw = DiaWindow {
                isolation: IsolationWindow {
                    center_mz: w.isolation_mz,
                    width_mz: w.isolation_width,
                },
                collision_energy: CollisionEnergyPolicy::Value(w.collision_energy),
                geometry: DiaGeometry::TimsMobility {
                    scan_start: w.scan_num_begin,
                    scan_end: w.scan_num_end,
                },
            };
            lo = lo.min(dw.isolation.lower());
            hi = hi.max(dw.isolation.upper());
            by_group.entry(w.window_group).or_default().push(dw);
        }

        let mut cycle = vec![AcquisitionEvent::Ms1(Ms1Event {
            analyzer: Analyzer::Tof,
            data_mode: DataMode::Centroid,
            mz_range: None,
            duration_s: None,
        })];
        let n_groups = by_group.len();
        for (_group, ws) in by_group {
            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: ws,
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
            }));
        }

        // Cycle timing from the precursor (MS1, MsMsType == 0) frame spacing.
        let mut prec_times: Vec<f64> = frames
            .iter()
            .filter(|f| f.ms_ms_type == 0)
            .map(|f| f.time)
            .collect();
        prec_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cycle_time_s = if prec_times.len() >= 2 {
            (prec_times[1] - prec_times[0]).max(0.0)
        } else {
            0.0
        };
        if !(cycle_time_s.is_finite() && cycle_time_s > 0.0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "could not determine cycle time (need >= 2 precursor frames)",
            ));
        }
        let start_time_s = frames
            .iter()
            .map(|f| f.time)
            .fold(f64::INFINITY, f64::min);
        let gradient_length_s = frames
            .iter()
            .map(|f| f.time)
            .fold(f64::NEG_INFINITY, f64::max);

        Ok(AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: InstrumentKind::TimsTofDia,
            cycle,
            repeat: RepeatPolicy::FixedCycleTime {
                cycle_time_s,
                gradient_length_s,
                start_time_s: if start_time_s.is_finite() {
                    start_time_s
                } else {
                    0.0
                },
            },
            mz_range: (lo, hi),
            provenance: Provenance {
                source: SchemeSource::ExtractedBruker,
                notes: format!("extracted from Bruker .d ({n_groups} window groups)"),
            },
        })
    }
}

#[cfg(feature = "sciex")]
impl AcquisitionScheme {
    /// Extract the SWATH scheme from a real SCIEX ZenoTOF `.wiff` method.
    ///
    /// Each SWATH window becomes a single-window [`DiaMs2Frame`] (no ion
    /// mobility) preceded by an MS1. SCIEX uses **rolling collision energy**,
    /// which is not stored in the `.wiff` SWATH method, so CE is
    /// [`CollisionEnergyPolicy::Unknown`] (a model must be supplied downstream;
    /// it is never invented). The `.wiff` method also does not carry run timing,
    /// so `cycle_time_s` and `gradient_length_s` are caller-supplied.
    pub fn from_sciex_wiff<P: AsRef<std::path::Path>>(
        path: P,
        cycle_time_s: f64,
        gradient_length_s: f64,
    ) -> io::Result<Self> {
        let method = sciexwiff::read_method(path)?;
        if method.swath_windows.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "no SWATH windows in the .wiff method",
            ));
        }
        let mut cycle = vec![AcquisitionEvent::Ms1(Ms1Event {
            analyzer: Analyzer::Tof,
            data_mode: DataMode::Centroid,
            mz_range: None,
            duration_s: None,
        })];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let n = method.swath_windows.len();
        for w in &method.swath_windows {
            let iso = IsolationWindow {
                center_mz: w.center_mz(),
                width_mz: w.width_mz(),
            };
            lo = lo.min(iso.lower());
            hi = hi.max(iso.upper());
            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: vec![DiaWindow {
                    isolation: iso,
                    collision_energy: CollisionEnergyPolicy::Unknown,
                    geometry: DiaGeometry::MzOnly,
                }],
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
            }));
        }
        Ok(AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: InstrumentKind::SciexZenoTof,
            cycle,
            repeat: RepeatPolicy::FixedCycleTime {
                cycle_time_s,
                gradient_length_s,
                start_time_s: 0.0,
            },
            mz_range: (lo, hi),
            provenance: Provenance {
                source: SchemeSource::ExtractedSciex,
                notes: format!(
                    "extracted from SCIEX .wiff method ({n} SWATH windows; rolling CE unknown, timing caller-supplied)"
                ),
            },
        })
    }
}

#[cfg(feature = "thermo")]
impl AcquisitionScheme {
    /// Extract the acquisition scheme from a real Thermo `.raw` by walking the
    /// first complete cycle (first MS1 up to the next MS1). Each MS2 scan becomes
    /// a single-window [`DiaMs2Frame`] (Thermo has no mobility), with the
    /// observed isolation center/width and CE.
    pub fn from_thermo_raw<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
        use thermorawfile::RawFile;
        let raw = RawFile::open(path)?;

        let analyzer_of = |a: u8| match a {
            4 => Analyzer::Ftms,
            7 => Analyzer::Astms,
            0 => Analyzer::Itms,
            _ => Analyzer::Unknown,
        };
        let data_mode_of = |scan: u32| -> DataMode {
            let i = (scan - raw.first_scan) as usize;
            let pkt = (raw.data_addr + raw.index[i].offset) as usize;
            if pkt + 8 <= raw.bytes.len() {
                let ps = u32::from_le_bytes(raw.bytes[pkt + 4..pkt + 8].try_into().unwrap());
                if ps > 0 {
                    DataMode::Profile
                } else {
                    DataMode::Centroid
                }
            } else {
                DataMode::Centroid
            }
        };
        let rt_of = |scan: u32| raw.index[(scan - raw.first_scan) as usize].time;

        let mut cycle: Vec<AcquisitionEvent> = Vec::new();
        let mut seen_ms1 = false;
        let mut closed = false;
        let mut start_rt = 0.0f64;
        let mut cycle_time_s = 0.0f64;
        let mut win_lo = f64::INFINITY;
        let mut win_hi = f64::NEG_INFINITY;

        for scan in raw.first_scan..=raw.last_scan {
            let ev = match raw.scan_event(scan) {
                Some(e) => e,
                None => {
                    // A missing event *inside* the selected cycle would silently
                    // drop a window — reject it. Before the first MS1 it's ignorable.
                    if seen_ms1 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("missing scan event for scan {scan} inside the first cycle"),
                        ));
                    }
                    continue;
                }
            };
            if ev.ms_order <= 1 {
                if seen_ms1 {
                    // start of the next cycle -> close this one
                    cycle_time_s = (rt_of(scan) - start_rt).max(0.0);
                    closed = true;
                    break;
                }
                seen_ms1 = true;
                start_rt = rt_of(scan);
                cycle.push(AcquisitionEvent::Ms1(Ms1Event {
                    analyzer: analyzer_of(ev.analyzer),
                    data_mode: data_mode_of(scan),
                    mz_range: None, // scan_event does not carry the MS1 range
                    duration_s: None,
                }));
            } else if seen_ms1 {
                let iso = IsolationWindow {
                    center_mz: ev.isolation_center,
                    width_mz: ev.isolation_width,
                };
                win_lo = win_lo.min(iso.lower());
                win_hi = win_hi.max(iso.upper());
                cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                    windows: vec![DiaWindow {
                        isolation: iso,
                        collision_energy: CollisionEnergyPolicy::Value(ev.collision_energy),
                        geometry: DiaGeometry::MzOnly,
                    }],
                    analyzer: analyzer_of(ev.analyzer),
                    data_mode: data_mode_of(scan),
                    duration_s: None,
                }));
            }
        }

        if !seen_ms1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "template has no MS1 scan",
            ));
        }
        if !closed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "template has only one cycle (no second MS1 to bound it); cannot determine cycle time",
            ));
        }
        let mz_range = if win_lo.is_finite() && win_hi > win_lo {
            (win_lo, win_hi)
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "first cycle has no usable MS2 isolation windows",
            ));
        };
        let gradient_length_s = raw.index.last().map(|e| e.time).unwrap_or(0.0);
        let n_ms2 = cycle
            .iter()
            .filter(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_)))
            .count();
        // Instrument from the analyzers of events actually in the cycle (an ASTMS
        // MS2 frame ⇒ Orbitrap Astral), not from incidental scans elsewhere.
        let any_astms = cycle.iter().any(|e| {
            matches!(e, AcquisitionEvent::DiaMs2Frame(f) if f.analyzer == Analyzer::Astms)
        });

        Ok(AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: if any_astms {
                InstrumentKind::OrbitrapAstral
            } else {
                InstrumentKind::Other
            },
            cycle,
            repeat: RepeatPolicy::FixedCycleTime {
                cycle_time_s,
                gradient_length_s,
                start_time_s: start_rt,
            },
            mz_range,
            provenance: Provenance {
                source: SchemeSource::ExtractedThermo,
                notes: format!("extracted from Thermo .raw (first cycle: 1 MS1 + {n_ms2} MS2)"),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_windows(n: usize) -> Vec<DiaWindow> {
        (0..n)
            .map(|i| DiaWindow {
                isolation: IsolationWindow {
                    center_mz: 400.0 + i as f64 * 10.0,
                    width_mz: 10.0,
                },
                collision_energy: CollisionEnergyPolicy::Value(25.0),
                geometry: DiaGeometry::MzOnly,
            })
            .collect()
    }

    #[test]
    fn from_window_table_validates() {
        let s = AcquisitionScheme::from_window_table(
            InstrumentKind::OrbitrapAstral,
            Ms1Event {
                analyzer: Analyzer::Ftms,
                data_mode: DataMode::Profile,
                mz_range: Some((390.0, 900.0)),
                duration_s: None,
            },
            linear_windows(5),
            RepeatPolicy::FixedCycleTime {
                cycle_time_s: 1.0,
                gradient_length_s: 600.0,
                start_time_s: 0.0,
            },
            (390.0, 900.0),
        );
        s.validate().unwrap();
        assert_eq!(s.windows().count(), 5);
        assert_eq!(s.ms1_count(), 1);
        assert_eq!(s.num_cycles(), Some(600));
    }

    #[test]
    fn multi_window_frame_only_tims() {
        let frame = DiaMs2Frame {
            windows: linear_windows(3),
            analyzer: Analyzer::Tof,
            data_mode: DataMode::Centroid,
            duration_s: None,
        };
        let s = AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: InstrumentKind::OrbitrapAstral, // wrong: multi-window on non-tims
            cycle: vec![
                AcquisitionEvent::Ms1(Ms1Event {
                    analyzer: Analyzer::Ftms,
                    data_mode: DataMode::Profile,
                    mz_range: Some((390.0, 900.0)),
                    duration_s: None,
                }),
                AcquisitionEvent::DiaMs2Frame(frame),
            ],
            repeat: RepeatPolicy::FixedCycleTime {
                cycle_time_s: 1.0,
                gradient_length_s: 600.0,
                start_time_s: 0.0,
            },
            mz_range: (390.0, 900.0),
            provenance: Provenance {
                source: SchemeSource::Programmatic,
                notes: String::new(),
            },
        };
        assert!(s.validate().is_err());
    }

    #[test]
    fn validate_rejects_bad_schemes() {
        let repeat = RepeatPolicy::FixedCycleTime {
            cycle_time_s: 1.0,
            gradient_length_s: 600.0,
            start_time_s: 0.0,
        };
        let ms1 = || Ms1Event {
            analyzer: Analyzer::Ftms,
            data_mode: DataMode::Profile,
            mz_range: Some((390.0, 900.0)),
            duration_s: None,
        };
        let win = || DiaWindow {
            isolation: IsolationWindow {
                center_mz: 500.0,
                width_mz: 10.0,
            },
            collision_energy: CollisionEnergyPolicy::Value(25.0),
            geometry: DiaGeometry::MzOnly,
        };
        let frame = |w: DiaWindow| {
            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: vec![w],
                analyzer: Analyzer::Astms,
                data_mode: DataMode::Centroid,
                duration_s: None,
            })
        };
        let mk = |cycle: Vec<AcquisitionEvent>| AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: InstrumentKind::OrbitrapAstral,
            cycle,
            repeat,
            mz_range: (390.0, 900.0),
            provenance: Provenance {
                source: SchemeSource::Programmatic,
                notes: String::new(),
            },
        };

        // valid baseline
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), frame(win())]).validate().is_ok());
        // MS2 first (no leading MS1)
        assert!(mk(vec![frame(win()), AcquisitionEvent::Ms1(ms1())]).validate().is_err());
        // two MS1 in a cycle
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), AcquisitionEvent::Ms1(ms1()), frame(win())]).validate().is_err());
        // MS1 with no MS2 frame
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1())]).validate().is_err());
        // window outside mz_range
        let mut oob = win();
        oob.isolation.center_mz = 2000.0;
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), frame(oob)]).validate().is_err());
        // negative collision energy
        let mut neg = win();
        neg.collision_energy = CollisionEnergyPolicy::Value(-5.0);
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), frame(neg)]).validate().is_err());
        // bad event duration
        let bad_dur = AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
            windows: vec![win()],
            analyzer: Analyzer::Astms,
            data_mode: DataMode::Centroid,
            duration_s: Some(-1.0),
        });
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), bad_dur]).validate().is_err());
    }

    // Gated: set TIMSIM_BRUKER_DIA_D to a real Bruker DIA-PASEF .d folder.
    #[test]
    fn from_bruker_d_extracts_cycle() {
        let d = match std::env::var("TIMSIM_BRUKER_DIA_D") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP from_bruker_d_extracts_cycle: set TIMSIM_BRUKER_DIA_D=<dia .d>");
                return;
            }
        };
        let s = AcquisitionScheme::from_bruker_d(&d).expect("extract");
        s.validate().expect("valid scheme");
        assert_eq!(s.instrument, InstrumentKind::TimsTofDia);
        assert_eq!(s.ms1_count(), 1);
        let n_win = s.windows().count();
        assert!(n_win > 1, "expected several windows, got {n_win}");
        // timsTOF windows carry mobility geometry with a known CE.
        for w in s.windows() {
            assert!(matches!(w.geometry, DiaGeometry::TimsMobility { .. }));
            assert!(w.collision_energy.at(w.isolation.center_mz).is_some());
        }
        // At least one frame should be mobility-partitioned (>1 window).
        let multi = s.cycle.iter().any(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(f) if f.windows.len() > 1));
        let n_frames = s.cycle.iter().filter(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_))).count();
        eprintln!(
            "from_bruker_d OK: TimsTofDia, {} frames, {} windows ({}), mz {:.1}..{:.1}",
            n_frames, n_win, if multi { "mobility-partitioned" } else { "single-window" },
            s.mz_range.0, s.mz_range.1
        );
    }

    // Gated: set TIMSIM_SCIEX_WIFF to a real ZenoTOF .wiff (OLE2 method).
    #[cfg(feature = "sciex")]
    #[test]
    fn from_sciex_wiff_extracts_windows() {
        let wiff = match std::env::var("TIMSIM_SCIEX_WIFF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP from_sciex_wiff_extracts_windows: set TIMSIM_SCIEX_WIFF=<.wiff>");
                return;
            }
        };
        let s = AcquisitionScheme::from_sciex_wiff(&wiff, 3.5, 1800.0).expect("extract");
        s.validate().expect("valid scheme");
        assert_eq!(s.instrument, InstrumentKind::SciexZenoTof);
        assert_eq!(s.ms1_count(), 1);
        let n = s.windows().count();
        assert!(n > 10, "expected many SWATH windows, got {n}");
        // SCIEX: m/z-only windows, rolling CE not recoverable -> Unknown.
        for w in s.windows() {
            assert!(matches!(w.geometry, DiaGeometry::MzOnly));
            assert!(matches!(w.collision_energy, CollisionEnergyPolicy::Unknown));
            assert!(w.collision_energy.at(w.isolation.center_mz).is_none());
        }
        eprintln!(
            "from_sciex_wiff OK: SciexZenoTof, {} SWATH windows, mz {:.1}..{:.1}",
            n, s.mz_range.0, s.mz_range.1
        );
    }

    #[cfg(feature = "thermo")]
    #[test]
    fn from_thermo_raw_extracts_cycle() {
        let template = match std::env::var("TIMSIM_ASTRAL_TEMPLATE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP from_thermo_raw_extracts_cycle: set TIMSIM_ASTRAL_TEMPLATE");
                return;
            }
        };
        let s = AcquisitionScheme::from_thermo_raw(&template).expect("extract");
        s.validate().expect("valid scheme");
        assert_eq!(s.instrument, InstrumentKind::OrbitrapAstral);
        assert_eq!(s.ms1_count(), 1, "one MS1 per cycle");
        let n_win = s.windows().count();
        assert!(n_win > 1, "expected several MS2 windows, got {n_win}");
        // Thermo: every window is m/z-only with a known CE.
        for w in s.windows() {
            assert!(matches!(w.geometry, DiaGeometry::MzOnly));
            assert!(w.collision_energy.at(w.isolation.center_mz).is_some());
            assert!(w.isolation.width_mz > 0.0);
        }
        // centers should be (weakly) increasing across the first cycle is not
        // guaranteed by Astral, but coverage must be a sane m/z span.
        assert!(s.mz_range.0 > 100.0 && s.mz_range.1 < 3000.0 && s.mz_range.0 < s.mz_range.1);
        eprintln!(
            "from_thermo_raw OK: instrument={:?}, {} MS2 windows, mz {:.1}..{:.1}, cycle {:.4}s",
            s.instrument, n_win, s.mz_range.0, s.mz_range.1,
            match s.repeat { RepeatPolicy::FixedCycleTime { cycle_time_s, .. } => cycle_time_s }
        );
    }
}

===== FILE: sciexwiff/src/lib.rs (separate crate, the .wiff method reader) =====
//! Minimal pure-Rust reader for the SCIEX `.wiff` acquisition **method**
//! (SWATH isolation windows + TOF calibration). See `README.md` for the scope
//! and legitimacy notes — this reads the open OLE2 method, not the proprietary
//! `.wiff.scan` spectra.

use std::io;
use std::path::Path;

/// A SWATH isolation window (m/z bounds).
#[derive(Clone, Copy, Debug)]
pub struct SwathWindow {
    pub lower_mz: f64,
    pub upper_mz: f64,
}

impl SwathWindow {
    pub fn center_mz(&self) -> f64 {
        (self.lower_mz + self.upper_mz) / 2.0
    }
    pub fn width_mz(&self) -> f64 {
        self.upper_mz - self.lower_mz
    }
}

/// TOF → m/z calibration, of the form `m/z = (coef1·tof + coef2)²`.
#[derive(Clone, Copy, Debug)]
pub struct TofCalibration {
    pub coef1: f64,
    pub coef2: f64,
}

/// The acquisition method decoded from a `.wiff`.
#[derive(Clone, Debug)]
pub struct WiffMethod {
    pub swath_windows: Vec<SwathWindow>,
    pub tof_calibration: Option<TofCalibration>,
}

fn f64le(b: &[u8], o: usize) -> Option<f64> {
    b.get(o..o + 8)
        .map(|s| f64::from_le_bytes(s.try_into().unwrap()))
}

fn read_stream(comp: &mut cfb::CompoundFile<std::fs::File>, name: &str) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut s = comp.open_stream(name).ok()?;
    let mut v = Vec::new();
    s.read_to_end(&mut v).ok()?;
    Some(v)
}

/// Open a `.wiff` (OLE2 compound file) and decode its SWATH method + TOF
/// calibration. The SWATH windows live in
/// `/MethodSubtree/Method1/DeviceMethod0/SWATHMethod` as 20-byte records
/// `{ f64 lower, f64 upper, u32 }` starting at offset 40; the calibration in
/// `/SampleSubtree/Sample1/TOFCalibrationData` (24-byte records from offset 32).
pub fn read_method<P: AsRef<Path>>(path: P) -> io::Result<WiffMethod> {
    let mut comp = cfb::open(path.as_ref())?;

    let mut swath_windows = Vec::new();
    if let Some(sw) = read_stream(&mut comp, "/MethodSubtree/Method1/DeviceMethod0/SWATHMethod") {
        let (base, stride) = (40usize, 20usize);
        if sw.len() > base {
            let n = (sw.len() - base) / stride;
            for k in 0..n {
                let o = base + k * stride;
                if let (Some(lo), Some(hi)) = (f64le(&sw, o), f64le(&sw, o + 8)) {
                    if lo.is_finite() && hi.is_finite() && hi > lo {
                        swath_windows.push(SwathWindow {
                            lower_mz: lo,
                            upper_mz: hi,
                        });
                    }
                }
            }
        }
    }

    let tof_calibration = read_stream(&mut comp, "/SampleSubtree/Sample1/TOFCalibrationData")
        .and_then(|cal| {
            let c1 = f64le(&cal, 32)?;
            let c2 = f64le(&cal, 40)?;
            (c1.is_finite() && c2.is_finite()).then_some(TofCalibration {
                coef1: c1,
                coef2: c2,
            })
        });

    Ok(WiffMethod {
        swath_windows,
        tof_calibration,
    })
}
</stdin>
codex
I’ll inspect the repository versions and surrounding APIs/tests, then return only ranked review findings with concrete file/line references.
exec
/bin/bash -lc "nl -ba rustdf/src/sim/scheme.rs | sed -n '280,570p'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
   280	                    Err(format!("{what} duration must be finite and > 0"))
   281	                }
   282	                _ => Ok(()),
   283	            }
   284	        };
   285	
   286	        for ev in &self.cycle {
   287	            match ev {
   288	                AcquisitionEvent::Ms1(m) => {
   289	                    check_dur(m.duration_s, "MS1")?;
   290	                    if let Some((a, b)) = m.mz_range {
   291	                        if !(a.is_finite() && b.is_finite()) || a >= b {
   292	                            return Err(format!("invalid MS1 mz_range ({a}, {b})"));
   293	                        }
   294	                    }
   295	                }
   296	                AcquisitionEvent::DiaMs2Frame(frame) => {
   297	                    check_dur(frame.duration_s, "MS2 frame")?;
   298	                    if frame.windows.is_empty() {
   299	                        return Err("MS2 frame has no windows".into());
   300	                    }
   301	                    let multi = frame.windows.len() > 1;
   302	                    if multi && self.instrument != InstrumentKind::TimsTofDia {
   303	                        return Err(
   304	                            "multi-window MS2 frame is only valid for timsTOF (mobility-partitioned)"
   305	                                .into(),
   306	                        );
   307	                    }
   308	                    for (i, w) in frame.windows.iter().enumerate() {
   309	                        if !(w.isolation.width_mz.is_finite() && w.isolation.width_mz > 0.0) {
   310	                            return Err("window width must be finite and > 0".into());
   311	                        }
   312	                        if !w.isolation.center_mz.is_finite()
   313	                            || !w.isolation.lower().is_finite()
   314	                            || !w.isolation.upper().is_finite()
   315	                        {
   316	                            return Err("window m/z is not finite".into());
   317	                        }
   318	                        if w.isolation.lower() < lo - 1e-6 || w.isolation.upper() > hi + 1e-6 {
   319	                            return Err(format!(
   320	                                "window [{:.4}, {:.4}] outside mz_range [{lo:.4}, {hi:.4}]",
   321	                                w.isolation.lower(),
   322	                                w.isolation.upper()
   323	                            ));
   324	                        }
   325	                        match w.collision_energy {
   326	                            CollisionEnergyPolicy::Value(v) => {
   327	                                if !v.is_finite() || v < 0.0 {
   328	                                    return Err("collision energy must be finite and >= 0".into());
   329	                                }
   330	                            }
   331	                            CollisionEnergyPolicy::Linear {
   332	                                intercept,
   333	                                slope_per_mz,
   334	                            } => {
   335	                                if !intercept.is_finite() || !slope_per_mz.is_finite() {
   336	                                    return Err("non-finite linear CE coefficients".into());
   337	                                }
   338	                                match w.collision_energy.at(w.isolation.center_mz) {
   339	                                    Some(ce) if ce.is_finite() && ce >= 0.0 => {}
   340	                                    _ => {
   341	                                        return Err(
   342	                                            "linear CE resolves to non-finite or negative".into()
   343	                                        )
   344	                                    }
   345	                                }
   346	                            }
   347	                            CollisionEnergyPolicy::Unknown => {}
   348	                        }
   349	                        match w.geometry {
   350	                            DiaGeometry::TimsMobility {
   351	                                scan_start,
   352	                                scan_end,
   353	                            } => {
   354	                                if self.instrument != InstrumentKind::TimsTofDia {
   355	                                    return Err("mobility geometry is only valid for timsTOF".into());
   356	                                }
   357	                                if scan_start > scan_end {
   358	                                    return Err("mobility scan_start > scan_end".into());
   359	                                }
   360	                            }
   361	                            DiaGeometry::MzOnly => {
   362	                                if multi {
   363	                                    return Err(
   364	                                        "multi-window timsTOF frame requires mobility geometry on every window"
   365	                                            .into(),
   366	                                    );
   367	                                }
   368	                            }
   369	                        }
   370	                        // Within one frame, windows must not overlap in BOTH m/z
   371	                        // and mobility (overlap across sequential frames is fine).
   372	                        for w2 in &frame.windows[..i] {
   373	                            let mz_overlap = w.isolation.lower() < w2.isolation.upper()
   374	                                && w2.isolation.lower() < w.isolation.upper();
   375	                            let im_overlap = match (w.geometry, w2.geometry) {
   376	                                (
   377	                                    DiaGeometry::TimsMobility {
   378	                                        scan_start: a0,
   379	                                        scan_end: a1,
   380	                                    },
   381	                                    DiaGeometry::TimsMobility {
   382	                                        scan_start: b0,
   383	                                        scan_end: b1,
   384	                                    },
   385	                                ) => a0 <= b1 && b0 <= a1,
   386	                                _ => true, // m/z-only windows share all mobility
   387	                            };
   388	                            if mz_overlap && im_overlap {
   389	                                return Err(
   390	                                    "windows within one frame overlap in both m/z and mobility"
   391	                                        .into(),
   392	                                );
   393	                            }
   394	                        }
   395	                    }
   396	                }
   397	            }
   398	        }
   399	        match self.repeat {
   400	            RepeatPolicy::FixedCycleTime {
   401	                cycle_time_s,
   402	                gradient_length_s,
   403	                start_time_s,
   404	            } => {
   405	                if !(cycle_time_s.is_finite() && cycle_time_s > 0.0) {
   406	                    return Err("cycle_time_s must be finite and > 0".into());
   407	                }
   408	                if !(gradient_length_s.is_finite() && gradient_length_s > 0.0) {
   409	                    return Err("gradient_length_s must be finite and > 0".into());
   410	                }
   411	                if !start_time_s.is_finite() || start_time_s < 0.0 {
   412	                    return Err("start_time_s must be finite and >= 0".into());
   413	                }
   414	                if start_time_s > gradient_length_s {
   415	                    return Err("start_time_s must be <= gradient_length_s".into());
   416	                }
   417	            }
   418	        }
   419	        Ok(())
   420	    }
   421	}
   422	
   423	impl AcquisitionScheme {
   424	    /// Extract the acquisition scheme from a real Bruker timsTOF DIA `.d`.
   425	    ///
   426	    /// Reads `DiaFrameMsMsWindows` and the frame table. Each **window group**
   427	    /// becomes one [`DiaMs2Frame`] holding its mobility-partitioned windows
   428	    /// (`TimsMobility` geometry — the scan ranges are Bruker-grid coordinates),
   429	    /// preceded by a single MS1 event. Cycle timing comes from the spacing of
   430	    /// the precursor (MS1) frames.
   431	    pub fn from_bruker_d<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
   432	        use crate::data::meta::{read_dia_ms_ms_windows, read_meta_data_sql};
   433	        use std::collections::BTreeMap;
   434	
   435	        let folder = path.as_ref().to_string_lossy().into_owned();
   436	        let to_io =
   437	            |e: Box<dyn std::error::Error>| io::Error::new(io::ErrorKind::InvalidData, e.to_string());
   438	        let windows = read_dia_ms_ms_windows(&folder).map_err(to_io)?;
   439	        let frames = read_meta_data_sql(&folder).map_err(to_io)?;
   440	        if windows.is_empty() {
   441	            return Err(io::Error::new(
   442	                io::ErrorKind::InvalidData,
   443	                "no DiaFrameMsMsWindows rows (not a DIA .d?)",
   444	            ));
   445	        }
   446	
   447	        // Group windows by window group (ascending); each group is a frame.
   448	        let mut by_group: BTreeMap<u32, Vec<DiaWindow>> = BTreeMap::new();
   449	        let mut lo = f64::INFINITY;
   450	        let mut hi = f64::NEG_INFINITY;
   451	        for w in &windows {
   452	            let dw = DiaWindow {
   453	                isolation: IsolationWindow {
   454	                    center_mz: w.isolation_mz,
   455	                    width_mz: w.isolation_width,
   456	                },
   457	                collision_energy: CollisionEnergyPolicy::Value(w.collision_energy),
   458	                geometry: DiaGeometry::TimsMobility {
   459	                    scan_start: w.scan_num_begin,
   460	                    scan_end: w.scan_num_end,
   461	                },
   462	            };
   463	            lo = lo.min(dw.isolation.lower());
   464	            hi = hi.max(dw.isolation.upper());
   465	            by_group.entry(w.window_group).or_default().push(dw);
   466	        }
   467	
   468	        let mut cycle = vec![AcquisitionEvent::Ms1(Ms1Event {
   469	            analyzer: Analyzer::Tof,
   470	            data_mode: DataMode::Centroid,
   471	            mz_range: None,
   472	            duration_s: None,
   473	        })];
   474	        let n_groups = by_group.len();
   475	        for (_group, ws) in by_group {
   476	            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
   477	                windows: ws,
   478	                analyzer: Analyzer::Tof,
   479	                data_mode: DataMode::Centroid,
   480	                duration_s: None,
   481	            }));
   482	        }
   483	
   484	        // Cycle timing from the precursor (MS1, MsMsType == 0) frame spacing.
   485	        let mut prec_times: Vec<f64> = frames
   486	            .iter()
   487	            .filter(|f| f.ms_ms_type == 0)
   488	            .map(|f| f.time)
   489	            .collect();
   490	        prec_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
   491	        let cycle_time_s = if prec_times.len() >= 2 {
   492	            (prec_times[1] - prec_times[0]).max(0.0)
   493	        } else {
   494	            0.0
   495	        };
   496	        if !(cycle_time_s.is_finite() && cycle_time_s > 0.0) {
   497	            return Err(io::Error::new(
   498	                io::ErrorKind::InvalidData,
   499	                "could not determine cycle time (need >= 2 precursor frames)",
   500	            ));
   501	        }
   502	        let start_time_s = frames
   503	            .iter()
   504	            .map(|f| f.time)
   505	            .fold(f64::INFINITY, f64::min);
   506	        let gradient_length_s = frames
   507	            .iter()
   508	            .map(|f| f.time)
   509	            .fold(f64::NEG_INFINITY, f64::max);
   510	
   511	        Ok(AcquisitionScheme {
   512	            version: SCHEME_VERSION,
   513	            instrument: InstrumentKind::TimsTofDia,
   514	            cycle,
   515	            repeat: RepeatPolicy::FixedCycleTime {
   516	                cycle_time_s,
   517	                gradient_length_s,
   518	                start_time_s: if start_time_s.is_finite() {
   519	                    start_time_s
   520	                } else {
   521	                    0.0
   522	                },
   523	            },
   524	            mz_range: (lo, hi),
   525	            provenance: Provenance {
   526	                source: SchemeSource::ExtractedBruker,
   527	                notes: format!("extracted from Bruker .d ({n_groups} window groups)"),
   528	            },
   529	        })
   530	    }
   531	}
   532	
   533	#[cfg(feature = "sciex")]
   534	impl AcquisitionScheme {
   535	    /// Extract the SWATH scheme from a real SCIEX ZenoTOF `.wiff` method.
   536	    ///
   537	    /// Each SWATH window becomes a single-window [`DiaMs2Frame`] (no ion
   538	    /// mobility) preceded by an MS1. SCIEX uses **rolling collision energy**,
   539	    /// which is not stored in the `.wiff` SWATH method, so CE is
   540	    /// [`CollisionEnergyPolicy::Unknown`] (a model must be supplied downstream;
   541	    /// it is never invented). The `.wiff` method also does not carry run timing,
   542	    /// so `cycle_time_s` and `gradient_length_s` are caller-supplied.
   543	    pub fn from_sciex_wiff<P: AsRef<std::path::Path>>(
   544	        path: P,
   545	        cycle_time_s: f64,
   546	        gradient_length_s: f64,
   547	    ) -> io::Result<Self> {
   548	        let method = sciexwiff::read_method(path)?;
   549	        if method.swath_windows.is_empty() {
   550	            return Err(io::Error::new(
   551	                io::ErrorKind::InvalidData,
   552	                "no SWATH windows in the .wiff method",
   553	            ));
   554	        }
   555	        let mut cycle = vec![AcquisitionEvent::Ms1(Ms1Event {
   556	            analyzer: Analyzer::Tof,
   557	            data_mode: DataMode::Centroid,
   558	            mz_range: None,
   559	            duration_s: None,
   560	        })];
   561	        let mut lo = f64::INFINITY;
   562	        let mut hi = f64::NEG_INFINITY;
   563	        let n = method.swath_windows.len();
   564	        for w in &method.swath_windows {
   565	            let iso = IsolationWindow {
   566	                center_mz: w.center_mz(),
   567	                width_mz: w.width_mz(),
   568	            };
   569	            lo = lo.min(iso.lower());
   570	            hi = hi.max(iso.upper());

exec
/bin/bash -lc 'git status --short && git diff -- rustdf/src/sim/scheme.rs sciexwiff/src/lib.rs' in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
?? EXTRACTORS.codex-review.md
?? packages/imspy-predictors/scripts/compare_libraries.py
?? packages/imspy-predictors/scripts/generate_calibrated_library.py
?? primitives/
?? rustdf/examples/build_bruker_pseudo_ms2.rs
?? rustdf/examples/build_bruker_pseudo_ms2_v3.rs
?? rustdf/examples/dump_bruker_ms1_centroids.rs
?? scripts/compare_diann_to_truth.py
?? scripts/reread_rustw_smoke.py

exec
/bin/bash -lc 'nl -ba sciexwiff/src/lib.rs' in /scratch/timsim-demo/SUBMISSION/rustims
 exited 1 in 0ms:
nl: sciexwiff/src/lib.rs: No such file or directory

exec
/bin/bash -lc "rg -n \"struct FrameMeta|enum MsMsType|ms_ms_type|read_dia_ms_ms_windows|SWATHMethod|TOFCalibrationData|read_method\" rustdf sciexwiff -g '*.rs' -g '*.md'" in /scratch/timsim-demo/SUBMISSION/rustims
 exited 2 in 0ms:
rg: sciexwiff: No such file or directory (os error 2)
rustdf/src/sim/scheme.rs:432:        use crate::data::meta::{read_dia_ms_ms_windows, read_meta_data_sql};
rustdf/src/sim/scheme.rs:438:        let windows = read_dia_ms_ms_windows(&folder).map_err(to_io)?;
rustdf/src/sim/scheme.rs:487:            .filter(|f| f.ms_ms_type == 0)
rustdf/src/sim/scheme.rs:548:        let method = sciexwiff::read_method(path)?;
rustdf/src/data/meta.rs:110:pub struct FrameMeta {
rustdf/src/data/meta.rs:115:    pub ms_ms_type: i64,
rustdf/src/data/meta.rs:330:                ms_ms_type: row.get(4)?,
rustdf/src/data/meta.rs:377:pub fn read_dia_ms_ms_windows(
rustdf/src/data/dda.rs:345:        let precursor_frames = meta_data.iter().filter(|x| x.ms_ms_type == 0);
rustdf/src/data/dda.rs:402:            .filter(|f| f.ms_ms_type == 0)
rustdf/src/data/dda.rs:978:        let precursor_frames = meta_data.iter().filter(|x| x.ms_ms_type == 0);
rustdf/src/data/handle.rs:435:            let ms_type_raw = self.raw_data_layout.frame_meta_data[frame_index].ms_ms_type;
rustdf/src/data/handle.rs:548:                let ms_type_raw = self.raw_data_layout.frame_meta_data[frame_index].ms_ms_type;
rustdf/src/data/handle.rs:588:                let ms_type_raw = self.raw_data_layout.frame_meta_data[frame_index].ms_ms_type;
rustdf/src/data/handle.rs:662:                let ms_type_raw = self.raw_data_layout.frame_meta_data[frame_index].ms_ms_type;
rustdf/src/data/handle.rs:771:        let ms_type_raw = self.raw_data_layout.frame_meta_data[frame_index].ms_ms_type;
rustdf/src/data/dia.rs:6:    read_dia_ms_ms_info, read_dia_ms_ms_windows, read_global_meta_sql, read_meta_data_sql,
rustdf/src/data/dia.rs:443:        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
rustdf/src/data/dia.rs:503:        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
rustdf/src/data/dia.rs:594:            .filter(|x| x.ms_ms_type == 0)
rustdf/src/data/dia.rs:710:    /// Map frame_id -> (time, ms_ms_type) for quick lookups.
rustdf/src/data/dia.rs:714:            m.insert(fm.id as u32, (fm.time as f32, fm.ms_ms_type));
rustdf/src/data/dia.rs:731:        // Just in case: keep only MS2 frames (ms_ms_type != 0)
rustdf/src/data/dia.rs:813:    /// RT-sorted PRECURSOR frames (ms_ms_type == 0) into FrameBinView rows.
rustdf/src/data/dia.rs:820:            .filter(|m| m.ms_ms_type == 0)

codex
The SCIEX crate is not under the current repository root at the path shown in the prompt, so I’m locating its actual workspace path and checking the Bruker metadata semantics before finalizing findings.
exec
/bin/bash -lc "nl -ba rustdf/src/sim/scheme.rs | sed -n '533,620p'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
   533	#[cfg(feature = "sciex")]
   534	impl AcquisitionScheme {
   535	    /// Extract the SWATH scheme from a real SCIEX ZenoTOF `.wiff` method.
   536	    ///
   537	    /// Each SWATH window becomes a single-window [`DiaMs2Frame`] (no ion
   538	    /// mobility) preceded by an MS1. SCIEX uses **rolling collision energy**,
   539	    /// which is not stored in the `.wiff` SWATH method, so CE is
   540	    /// [`CollisionEnergyPolicy::Unknown`] (a model must be supplied downstream;
   541	    /// it is never invented). The `.wiff` method also does not carry run timing,
   542	    /// so `cycle_time_s` and `gradient_length_s` are caller-supplied.
   543	    pub fn from_sciex_wiff<P: AsRef<std::path::Path>>(
   544	        path: P,
   545	        cycle_time_s: f64,
   546	        gradient_length_s: f64,
   547	    ) -> io::Result<Self> {
   548	        let method = sciexwiff::read_method(path)?;
   549	        if method.swath_windows.is_empty() {
   550	            return Err(io::Error::new(
   551	                io::ErrorKind::InvalidData,
   552	                "no SWATH windows in the .wiff method",
   553	            ));
   554	        }
   555	        let mut cycle = vec![AcquisitionEvent::Ms1(Ms1Event {
   556	            analyzer: Analyzer::Tof,
   557	            data_mode: DataMode::Centroid,
   558	            mz_range: None,
   559	            duration_s: None,
   560	        })];
   561	        let mut lo = f64::INFINITY;
   562	        let mut hi = f64::NEG_INFINITY;
   563	        let n = method.swath_windows.len();
   564	        for w in &method.swath_windows {
   565	            let iso = IsolationWindow {
   566	                center_mz: w.center_mz(),
   567	                width_mz: w.width_mz(),
   568	            };
   569	            lo = lo.min(iso.lower());
   570	            hi = hi.max(iso.upper());
   571	            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
   572	                windows: vec![DiaWindow {
   573	                    isolation: iso,
   574	                    collision_energy: CollisionEnergyPolicy::Unknown,
   575	                    geometry: DiaGeometry::MzOnly,
   576	                }],
   577	                analyzer: Analyzer::Tof,
   578	                data_mode: DataMode::Centroid,
   579	                duration_s: None,
   580	            }));
   581	        }
   582	        Ok(AcquisitionScheme {
   583	            version: SCHEME_VERSION,
   584	            instrument: InstrumentKind::SciexZenoTof,
   585	            cycle,
   586	            repeat: RepeatPolicy::FixedCycleTime {
   587	                cycle_time_s,
   588	                gradient_length_s,
   589	                start_time_s: 0.0,
   590	            },
   591	            mz_range: (lo, hi),
   592	            provenance: Provenance {
   593	                source: SchemeSource::ExtractedSciex,
   594	                notes: format!(
   595	                    "extracted from SCIEX .wiff method ({n} SWATH windows; rolling CE unknown, timing caller-supplied)"
   596	                ),
   597	            },
   598	        })
   599	    }
   600	}
   601	
   602	#[cfg(feature = "thermo")]
   603	impl AcquisitionScheme {
   604	    /// Extract the acquisition scheme from a real Thermo `.raw` by walking the
   605	    /// first complete cycle (first MS1 up to the next MS1). Each MS2 scan becomes
   606	    /// a single-window [`DiaMs2Frame`] (Thermo has no mobility), with the
   607	    /// observed isolation center/width and CE.
   608	    pub fn from_thermo_raw<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
   609	        use thermorawfile::RawFile;
   610	        let raw = RawFile::open(path)?;
   611	
   612	        let analyzer_of = |a: u8| match a {
   613	            4 => Analyzer::Ftms,
   614	            7 => Analyzer::Astms,
   615	            0 => Analyzer::Itms,
   616	            _ => Analyzer::Unknown,
   617	        };
   618	        let data_mode_of = |scan: u32| -> DataMode {
   619	            let i = (scan - raw.first_scan) as usize;
   620	            let pkt = (raw.data_addr + raw.index[i].offset) as usize;

exec
/bin/bash -lc "nl -ba rustdf/src/data/meta.rs | sed -n '90,135p;360,420p'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
    90	}
    91	
    92	#[derive(Debug)]
    93	pub struct GlobalMetaData {
    94	    pub schema_type: String,
    95	    pub schema_version_major: i64,
    96	    pub schema_version_minor: i64,
    97	    pub acquisition_software_vendor: String,
    98	    pub instrument_vendor: String,
    99	    pub closed_property: i64,
   100	    pub tims_compression_type: i64,
   101	    pub max_num_peaks_per_scan: i64,
   102	    pub mz_acquisition_range_lower: f64,
   103	    pub mz_acquisition_range_upper: f64,
   104	    pub one_over_k0_range_lower: f64,
   105	    pub one_over_k0_range_upper: f64,
   106	    pub tof_max_index: u32,
   107	}
   108	
   109	#[derive(Debug)]
   110	pub struct FrameMeta {
   111	    pub id: i64,
   112	    pub time: f64,
   113	    pub polarity: String,
   114	    pub scan_mode: i64,
   115	    pub ms_ms_type: i64,
   116	    pub tims_id: i64,
   117	    pub max_intensity: f64,
   118	    pub sum_intensity: f64,
   119	    pub num_scans: i64,
   120	    pub num_peaks: i64,
   121	    pub mz_calibration: i64,
   122	    pub t_1: f64,
   123	    pub t_2: f64,
   124	    pub tims_calibration: i64,
   125	    pub property_group: i64,
   126	    pub accumulation_time: f64,
   127	    pub ramp_time: f64,
   128	}
   129	
   130	struct GlobalMetaInternal {
   131	    key: String,
   132	    value: String,
   133	}
   134	
   135	pub fn read_dda_precursor_meta(
   360	    let query = format!("SELECT {} FROM DiaFrameMsMsInfo", rows.join(", "));
   361	
   362	    // execute the query
   363	    let frames_rows: Result<Vec<DiaMsMisInfo>, _> = conn
   364	        .prepare(&query)?
   365	        .query_map([], |row| {
   366	            Ok(DiaMsMisInfo {
   367	                frame_id: row.get(0)?,
   368	                window_group: row.get(1)?,
   369	            })
   370	        })?
   371	        .collect();
   372	
   373	    // return the frames
   374	    Ok(frames_rows?)
   375	}
   376	
   377	pub fn read_dia_ms_ms_windows(
   378	    bruker_d_folder_name: &str,
   379	) -> Result<Vec<DiaMsMsWindow>, Box<dyn std::error::Error>> {
   380	    // Connect to the database
   381	    let db_path = Path::new(bruker_d_folder_name).join("analysis.tdf");
   382	    let conn = Connection::open(db_path)?;
   383	
   384	    // prepare the query
   385	    let rows: Vec<&str> = vec![
   386	        "WindowGroup",
   387	        "ScanNumBegin",
   388	        "ScanNumEnd",
   389	        "IsolationMz",
   390	        "IsolationWidth",
   391	        "CollisionEnergy",
   392	    ];
   393	    let query = format!("SELECT {} FROM DiaFrameMsMsWindows", rows.join(", "));
   394	
   395	    // execute the query
   396	    let frames_rows: Result<Vec<DiaMsMsWindow>, _> = conn
   397	        .prepare(&query)?
   398	        .query_map([], |row| {
   399	            Ok(DiaMsMsWindow {
   400	                window_group: row.get(0)?,
   401	                scan_num_begin: row.get(1)?,
   402	                scan_num_end: row.get(2)?,
   403	                isolation_mz: row.get(3)?,
   404	                isolation_width: row.get(4)?,
   405	                collision_energy: row.get(5)?,
   406	            })
   407	        })?
   408	        .collect();
   409	
   410	    // return the frames
   411	    Ok(frames_rows?)
   412	}
   413	
   414	/// Read m/z calibration data from the MzCalibration table.
   415	/// This provides the coefficients needed for accurate TOF to m/z conversion
   416	/// without requiring the Bruker SDK.
   417	///
   418	/// The calibration formula is:
   419	///   tof_time = (tof_index + 0.5) * digitizer_timebase + digitizer_delay
   420	///   sqrt(mz) = c0 + c1*tof_time + c2*tof_time^2 + ...

exec
/bin/bash -lc "rg -n \"sciexwiff\" Cargo.toml '**/Cargo.toml' Cargo.lock 2>/dev/null" in /scratch/timsim-demo/SUBMISSION/rustims
 exited 2 in 0ms:
Cargo.lock:2156: "sciexwiff",
Cargo.lock:2210:name = "sciexwiff"
Cargo.lock:2212:source = "git+https://github.com/theGreatHerrLebert/sciexwiff?rev=4e8ff07842bb52a0bd9fe601cc61fd95a68b9080#4e8ff07842bb52a0bd9fe601cc61fd95a68b9080"

exec
/bin/bash -lc "find .. -path '*/sciexwiff/src/lib.rs' -o -path '*/sciexwiff/Cargo.toml'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 1303ms:
codex
A key interaction risk is emerging around malformed numeric data and extractor postconditions. I’m checking how Bruker scan bounds are interpreted elsewhere, because `validate()` currently treats touching mobility intervals as overlapping.
exec
/bin/bash -lc "git show 4e8ff07842bb52a0bd9fe601cc61fd95a68b9080:src/lib.rs 2>/dev/null | nl -ba | sed -n '1,180p'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
exec
/bin/bash -lc 'rg -n "scan_num_begin|scan_num_end|scan_start|scan_end" rustdf/src | head -120' in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
rustdf/src/sim/dda.rs:572:                if scan_i32 < meta.scan_start || scan_i32 > meta.scan_end {
rustdf/src/sim/dda.rs:870:                if scan_i32 < meta.scan_start || scan_i32 > meta.scan_end {
rustdf/src/sim/containers.rs:128:    pub scan_start: u32,
rustdf/src/sim/containers.rs:129:    pub scan_end: u32,
rustdf/src/sim/containers.rs:138:        scan_start: u32,
rustdf/src/sim/containers.rs:139:        scan_end: u32,
rustdf/src/sim/containers.rs:146:            scan_start,
rustdf/src/sim/containers.rs:147:            scan_end,
rustdf/src/data/raw.rs:216:    //       uint32_t scan_end,
rustdf/src/data/raw.rs:237:        scan_end: u32,
rustdf/src/data/raw.rs:261:                scan_end,
rustdf/src/sim/scheme.rs:94:    TimsMobility { scan_start: u32, scan_end: u32 },
rustdf/src/sim/scheme.rs:351:                                scan_start,
rustdf/src/sim/scheme.rs:352:                                scan_end,
rustdf/src/sim/scheme.rs:357:                                if scan_start > scan_end {
rustdf/src/sim/scheme.rs:358:                                    return Err("mobility scan_start > scan_end".into());
rustdf/src/sim/scheme.rs:378:                                        scan_start: a0,
rustdf/src/sim/scheme.rs:379:                                        scan_end: a1,
rustdf/src/sim/scheme.rs:382:                                        scan_start: b0,
rustdf/src/sim/scheme.rs:383:                                        scan_end: b1,
rustdf/src/sim/scheme.rs:459:                    scan_start: w.scan_num_begin,
rustdf/src/sim/scheme.rs:460:                    scan_end: w.scan_num_end,
rustdf/src/sim/lazy_builder.rs:952:                        .find(|scan_meta| scan_meta.scan_start <= *scan as i32 && scan_meta.scan_end >= *scan as i32)
rustdf/src/data/meta.rs:15:    pub scan_num_begin: u32,
rustdf/src/data/meta.rs:16:    pub scan_num_end: u32,
rustdf/src/data/meta.rs:25:    pub scan_num_begin: i64,
rustdf/src/data/meta.rs:26:    pub scan_num_end: i64,
rustdf/src/data/meta.rs:64:    pub scan_end: i64,
rustdf/src/data/meta.rs:201:                scan_num_begin: row.get(1)?,
rustdf/src/data/meta.rs:202:                scan_num_end: row.get(2)?,
rustdf/src/data/meta.rs:401:                scan_num_begin: row.get(1)?,
rustdf/src/data/meta.rs:402:                scan_num_end: row.get(2)?,
rustdf/src/data/dia.rs:209:            if let Some(p) = norm_u32_pair(w.scan_num_begin, w.scan_num_end) {
rustdf/src/data/dia.rs:555:                    scan_lo: w.scan_num_begin,
rustdf/src/data/dia.rs:556:                    scan_hi: w.scan_num_end,
rustdf/src/data/dia.rs:747:                let l = w.scan_num_begin as usize;
rustdf/src/data/dia.rs:748:                let r = w.scan_num_end as usize;
rustdf/src/data/handle.rs:133:    scan_start: usize,
rustdf/src/data/handle.rs:143:    let mut current_index = scan_start;
rustdf/src/data/handle.rs:162:    let scan_size = current_index - scan_start;
rustdf/src/data/handle.rs:502:                let mut scan_start = 0usize;
rustdf/src/data/handle.rs:517:                    scan_start += parse_decompressed_bruker_binary_type1(
rustdf/src/data/handle.rs:522:                        scan_start,
rustdf/src/data/dda.rs:671:            let scan_margin = (pasef_info.scan_num_end - pasef_info.scan_num_begin) / 20;
rustdf/src/data/dda.rs:677:                (pasef_info.scan_num_begin - scan_margin) as i32,
rustdf/src/data/dda.rs:678:                (pasef_info.scan_num_end + scan_margin) as i32,
rustdf/src/data/dda.rs:774:            let scan_start_time = frame_time_map.get(&first_pasef.frame_id).copied().unwrap_or(0.0);
rustdf/src/data/dda.rs:788:                let scan_margin = (pasef_info.scan_num_end - pasef_info.scan_num_begin) / 20;
rustdf/src/data/dda.rs:794:                    (pasef_info.scan_num_begin - scan_margin) as i32,
rustdf/src/data/dda.rs:795:                    (pasef_info.scan_num_end + scan_margin) as i32,
rustdf/src/data/dda.rs:824:                scan_start_time,
rustdf/src/data/dda.rs:876:        let scan_margin = (pasef_info.scan_num_end - pasef_info.scan_num_begin) / 20;
rustdf/src/data/dda.rs:882:            (pasef_info.scan_num_begin - scan_margin) as i32,
rustdf/src/data/dda.rs:883:            (pasef_info.scan_num_end + scan_margin) as i32,
rustdf/src/sim/handle.rs:186:                row.get("scan_start")?,
rustdf/src/sim/handle.rs:187:                row.get("scan_end")?,
rustdf/src/sim/handle.rs:219:            "SELECT frame, scan_start, scan_end, isolation_mz, isolation_width, collision_energy, precursor FROM pasef_meta"
rustdf/src/sim/handle.rs:224:                row.get("scan_start")?,
rustdf/src/sim/handle.rs:225:                row.get("scan_end")?,
rustdf/src/sim/handle.rs:374:                .map(|x| x.scan_start as i32)
rustdf/src/sim/handle.rs:378:                .map(|x| x.scan_end as i32)
rustdf/src/sim/handle.rs:419:                .map(|x| x.scan_start as i32)
rustdf/src/sim/handle.rs:423:                .map(|x| x.scan_end as i32)

exec
/bin/bash -lc "find ~/.cargo/git/checkouts -path '*/sciexwiff-*/src/lib.rs' -print 2>/dev/null | head -10" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs

exec
/bin/bash -lc "nl -ba rustdf/src/data/dia.rs | sed -n '430,530p;700,750p;805,840p'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
   430	}
   431	
   432	impl TimsDatasetDIA {
   433	    pub fn new(
   434	        bruker_lib_path: &str,
   435	        data_path: &str,
   436	        in_memory: bool,
   437	        use_bruker_sdk: bool,
   438	    ) -> Self {
   439	        // TODO: error handling
   440	        let global_meta_data = read_global_meta_sql(data_path).unwrap();
   441	        let meta_data = read_meta_data_sql(data_path).unwrap();
   442	        let dia_ms_mis_info = read_dia_ms_ms_info(data_path).unwrap();
   443	        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
   444	
   445	        let scan_max_index = meta_data.iter().map(|x| x.num_scans).max().unwrap() as u32;
   446	        let im_lower = global_meta_data.one_over_k0_range_lower;
   447	        let im_upper = global_meta_data.one_over_k0_range_upper;
   448	
   449	        let tof_max_index = global_meta_data.tof_max_index;
   450	        let mz_lower = global_meta_data.mz_acquisition_range_lower;
   451	        let mz_upper = global_meta_data.mz_acquisition_range_upper;
   452	
   453	        let loader = match in_memory {
   454	            true => TimsDataLoader::new_in_memory(
   455	                bruker_lib_path,
   456	                data_path,
   457	                use_bruker_sdk,
   458	                scan_max_index,
   459	                im_lower,
   460	                im_upper,
   461	                tof_max_index,
   462	                mz_lower,
   463	                mz_upper,
   464	            ),
   465	            false => TimsDataLoader::new_lazy(
   466	                bruker_lib_path,
   467	                data_path,
   468	                use_bruker_sdk,
   469	                scan_max_index,
   470	                im_lower,
   471	                im_upper,
   472	                tof_max_index,
   473	                mz_lower,
   474	                mz_upper,
   475	            ),
   476	        };
   477	
   478	        let dia_index = DiaIndex::new(&meta_data, &dia_ms_mis_info, &dia_ms_ms_windows);
   479	
   480	        TimsDatasetDIA {
   481	            loader,
   482	            global_meta_data,
   483	            meta_data,
   484	            dia_ms_ms_info: dia_ms_mis_info,
   485	            dia_ms_ms_windows,
   486	            dia_index,
   487	        }
   488	    }
   489	
   490	    /// Create a DIA dataset with regression-derived m/z calibration.
   491	    ///
   492	    /// This method uses externally-provided m/z calibration coefficients (e.g., from
   493	    /// linear regression on SDK data) instead of the simple boundary model.
   494	    pub fn new_with_mz_calibration(
   495	        data_path: &str,
   496	        in_memory: bool,
   497	        tof_intercept: f64,
   498	        tof_slope: f64,
   499	    ) -> Self {
   500	        let meta_data = read_meta_data_sql(data_path).unwrap();
   501	        let global_meta_data = read_global_meta_sql(data_path).unwrap();
   502	        let dia_ms_mis_info = read_dia_ms_ms_info(data_path).unwrap();
   503	        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
   504	
   505	        let scan_max_index = meta_data.iter().map(|x| x.num_scans).max().unwrap() as u32;
   506	        let im_lower = global_meta_data.one_over_k0_range_lower;
   507	        let im_upper = global_meta_data.one_over_k0_range_upper;
   508	
   509	        let loader = match in_memory {
   510	            true => TimsDataLoader::new_in_memory_with_mz_calibration(
   511	                data_path,
   512	                tof_intercept,
   513	                tof_slope,
   514	                im_lower,
   515	                im_upper,
   516	                scan_max_index,
   517	            ),
   518	            false => TimsDataLoader::new_lazy_with_mz_calibration(
   519	                data_path,
   520	                tof_intercept,
   521	                tof_slope,
   522	                im_lower,
   523	                im_upper,
   524	                scan_max_index,
   525	            ),
   526	        };
   527	
   528	        let dia_index = DiaIndex::new(&meta_data, &dia_ms_mis_info, &dia_ms_ms_windows);
   529	
   530	        TimsDatasetDIA {
   700	            .collect();
   701	        gs.sort_unstable();
   702	        gs.dedup();
   703	        gs
   704	    }
   705	
   706	    pub fn window_groups_for_precursor(&self, prec_mz: f32, im_apex: f32) -> Vec<u32> {
   707	        self.dia_index.groups_for_precursor(prec_mz, im_apex)
   708	    }
   709	
   710	    /// Map frame_id -> (time, ms_ms_type) for quick lookups.
   711	    fn frame_time_map(&self) -> FxHashMap<u32, (f32, i64)> {
   712	        let mut m = FxHashMap::default();
   713	        for fm in &self.meta_data {
   714	            m.insert(fm.id as u32, (fm.time as f32, fm.ms_ms_type));
   715	        }
   716	        m
   717	    }
   718	
   719	    /// RT-sorted **fragment** frames for a given DIA group.
   720	    pub fn fragment_frame_ids_and_times_for_group_core(&self, window_group: u32) -> (Vec<u32>, Vec<f32>) {
   721	        let time_map = self.frame_time_map();
   722	        let mut rows: Vec<(u32, f32)> = self
   723	            .dia_ms_ms_info
   724	            .iter()
   725	            .filter(|x| x.window_group == window_group)
   726	            .filter_map(|x| {
   727	                time_map.get(&(x.frame_id)).map(|(t, _ms2)| (x.frame_id, *t))
   728	            })
   729	            .collect();
   730	
   731	        // Just in case: keep only MS2 frames (ms_ms_type != 0)
   732	        rows.retain(|(fid, _)| time_map.get(fid).map(|(_, ty)| *ty != 0).unwrap_or(false));
   733	
   734	        rows.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
   735	        let (ids, times): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
   736	        (ids, times)
   737	    }
   738	
   739	    /// Merged, sorted **global scan** unions for this DIA group, from DiaFrameMsMsWindows.
   740	    /// Returns None if there are no window rows for this group.
   741	    pub fn scan_unions_for_window_group_core(&self, window_group: u32) -> Option<Vec<(usize, usize)>> {
   742	        let ranges: Vec<(usize, usize)> = self
   743	            .dia_ms_ms_windows
   744	            .iter()
   745	            .filter(|w| w.window_group == window_group)
   746	            .map(|w| {
   747	                let l = w.scan_num_begin as usize;
   748	                let r = w.scan_num_end as usize;
   749	                if l <= r { (l, r) } else { (r, l) }
   750	            })
   805	        RtFrames {
   806	            frames,
   807	            frame_ids: ids,
   808	            rt_times: times,
   809	            scale: Arc::new(scale),
   810	        }
   811	    }
   812	
   813	    /// RT-sorted PRECURSOR frames (ms_ms_type == 0) into FrameBinView rows.
   814	    /// `tof_step` controls TOF granularity of CSR binning.
   815	    pub fn make_rt_frames_for_precursor(&self, tof_step: i32) -> RtFrames {
   816	        assert!(tof_step > 0);
   817	
   818	        let mut rows: Vec<(u32, f32, usize)> = self.meta_data
   819	            .iter()
   820	            .filter(|m| m.ms_ms_type == 0)
   821	            .map(|m| (m.id as u32, m.time as f32, m.num_scans as usize))
   822	            .collect();
   823	
   824	        assert!(!rows.is_empty(), "No precursor (MS1) frames found");
   825	        rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
   826	
   827	        let frame_ids: Vec<u32> = rows.iter().map(|r| r.0).collect();
   828	        let rt_times:  Vec<f32> = rows.iter().map(|r| r.1).collect();
   829	        let global_num_scans = rows.iter().map(|r| r.2).max().unwrap_or(1);
   830	
   831	        // Build TOF scale from precursor frames
   832	        let frames_for_scale: Vec<_> = frame_ids.iter().map(|&fid| self.get_frame(fid)).collect();
   833	        let scale = TofScale::build_from_tof(&frames_for_scale, tof_step)
   834	            .expect("make_rt_frames_for_precursor: failed to build TOF scale");
   835	
   836	        let frames: Vec<FrameBinView> = frame_ids.par_iter()
   837	            .map(|&fid| build_frame_bin_view(self.get_frame(fid), &scale, global_num_scans))
   838	            .collect();
   839	
   840	        RtFrames {

exec
/bin/bash -lc 'nl -ba /home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs' in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
     1	//! Minimal pure-Rust reader for the SCIEX `.wiff` acquisition **method**
     2	//! (SWATH isolation windows + TOF calibration). See `README.md` for the scope
     3	//! and legitimacy notes — this reads the open OLE2 method, not the proprietary
     4	//! `.wiff.scan` spectra.
     5	
     6	use std::io;
     7	use std::path::Path;
     8	
     9	/// A SWATH isolation window (m/z bounds).
    10	#[derive(Clone, Copy, Debug)]
    11	pub struct SwathWindow {
    12	    pub lower_mz: f64,
    13	    pub upper_mz: f64,
    14	}
    15	
    16	impl SwathWindow {
    17	    pub fn center_mz(&self) -> f64 {
    18	        (self.lower_mz + self.upper_mz) / 2.0
    19	    }
    20	    pub fn width_mz(&self) -> f64 {
    21	        self.upper_mz - self.lower_mz
    22	    }
    23	}
    24	
    25	/// TOF → m/z calibration, of the form `m/z = (coef1·tof + coef2)²`.
    26	#[derive(Clone, Copy, Debug)]
    27	pub struct TofCalibration {
    28	    pub coef1: f64,
    29	    pub coef2: f64,
    30	}
    31	
    32	/// The acquisition method decoded from a `.wiff`.
    33	#[derive(Clone, Debug)]
    34	pub struct WiffMethod {
    35	    pub swath_windows: Vec<SwathWindow>,
    36	    pub tof_calibration: Option<TofCalibration>,
    37	}
    38	
    39	fn f64le(b: &[u8], o: usize) -> Option<f64> {
    40	    b.get(o..o + 8)
    41	        .map(|s| f64::from_le_bytes(s.try_into().unwrap()))
    42	}
    43	
    44	fn read_stream(comp: &mut cfb::CompoundFile<std::fs::File>, name: &str) -> Option<Vec<u8>> {
    45	    use std::io::Read;
    46	    let mut s = comp.open_stream(name).ok()?;
    47	    let mut v = Vec::new();
    48	    s.read_to_end(&mut v).ok()?;
    49	    Some(v)
    50	}
    51	
    52	/// Open a `.wiff` (OLE2 compound file) and decode its SWATH method + TOF
    53	/// calibration. The SWATH windows live in
    54	/// `/MethodSubtree/Method1/DeviceMethod0/SWATHMethod` as 20-byte records
    55	/// `{ f64 lower, f64 upper, u32 }` starting at offset 40; the calibration in
    56	/// `/SampleSubtree/Sample1/TOFCalibrationData` (24-byte records from offset 32).
    57	pub fn read_method<P: AsRef<Path>>(path: P) -> io::Result<WiffMethod> {
    58	    let mut comp = cfb::open(path.as_ref())?;
    59	
    60	    let mut swath_windows = Vec::new();
    61	    if let Some(sw) = read_stream(&mut comp, "/MethodSubtree/Method1/DeviceMethod0/SWATHMethod") {
    62	        let (base, stride) = (40usize, 20usize);
    63	        if sw.len() > base {
    64	            let n = (sw.len() - base) / stride;
    65	            for k in 0..n {
    66	                let o = base + k * stride;
    67	                if let (Some(lo), Some(hi)) = (f64le(&sw, o), f64le(&sw, o + 8)) {
    68	                    if lo.is_finite() && hi.is_finite() && hi > lo {
    69	                        swath_windows.push(SwathWindow {
    70	                            lower_mz: lo,
    71	                            upper_mz: hi,
    72	                        });
    73	                    }
    74	                }
    75	            }
    76	        }
    77	    }
    78	
    79	    let tof_calibration = read_stream(&mut comp, "/SampleSubtree/Sample1/TOFCalibrationData")
    80	        .and_then(|cal| {
    81	            let c1 = f64le(&cal, 32)?;
    82	            let c2 = f64le(&cal, 40)?;
    83	            (c1.is_finite() && c2.is_finite()).then_some(TofCalibration {
    84	                coef1: c1,
    85	                coef2: c2,
    86	            })
    87	        });
    88	
    89	    Ok(WiffMethod {
    90	        swath_windows,
    91	        tof_calibration,
    92	    })
    93	}

exec
/bin/bash -lc "nl -ba rustdf/src/data/dia.rs | sed -n '185,225p;739,775p'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
   185	            frame_time.insert(m.id as u32, m.time);
   186	        }
   187	
   188	        // frame_id -> group; group -> frames
   189	        let mut frame_to_group: HashMap<u32, u32> = HashMap::new();
   190	        let mut group_to_frames: HashMap<u32, Vec<u32>> = HashMap::new();
   191	        for r in info {
   192	            let fid = r.frame_id;
   193	            frame_to_group.insert(fid, r.window_group);
   194	            group_to_frames.entry(r.window_group).or_default().push(fid);
   195	        }
   196	
   197	        // raw program rows
   198	        let mut group_to_isolation: HashMap<u32, Vec<(f64, f64)>> = HashMap::new();
   199	        let mut group_to_scan_ranges: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
   200	        for w in wins {
   201	            let half = 0.5 * w.isolation_width;
   202	            if half.is_finite() && half > 0.0 && w.isolation_mz.is_finite() {
   203	                let lo = w.isolation_mz - half;
   204	                let hi = w.isolation_mz + half;
   205	                if let Some(p) = norm_f64_pair(lo, hi) {
   206	                    group_to_isolation.entry(w.window_group).or_default().push(p);
   207	                }
   208	            }
   209	            if let Some(p) = norm_u32_pair(w.scan_num_begin, w.scan_num_end) {
   210	                group_to_scan_ranges.entry(w.window_group).or_default().push(p);
   211	            }
   212	        }
   213	
   214	        // unions + sort frames by time
   215	        let mut group_to_mz_union: HashMap<u32, (f64, f64)> = HashMap::new();
   216	        let mut group_to_scan_unions: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
   217	
   218	        for (g, frames) in group_to_frames.iter_mut() {
   219	            frames.sort_unstable_by(|&a, &b| {
   220	                let ta = frame_time.get(&a).copied().unwrap_or(f64::NAN);
   221	                let tb = frame_time.get(&b).copied().unwrap_or(f64::NAN);
   222	                ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
   223	            });
   224	
   225	            if let Some(v) = group_to_isolation.get(g) {
   739	    /// Merged, sorted **global scan** unions for this DIA group, from DiaFrameMsMsWindows.
   740	    /// Returns None if there are no window rows for this group.
   741	    pub fn scan_unions_for_window_group_core(&self, window_group: u32) -> Option<Vec<(usize, usize)>> {
   742	        let ranges: Vec<(usize, usize)> = self
   743	            .dia_ms_ms_windows
   744	            .iter()
   745	            .filter(|w| w.window_group == window_group)
   746	            .map(|w| {
   747	                let l = w.scan_num_begin as usize;
   748	                let r = w.scan_num_end as usize;
   749	                if l <= r { (l, r) } else { (r, l) }
   750	            })
   751	            .collect();
   752	        if ranges.is_empty() {
   753	            return None;
   754	        }
   755	        Some(merge_ranges(ranges))
   756	    }
   757	
   758	    /// m/z unions (min..max) for the group from DiaFrameMsMsWindows (wide clamp).
   759	    pub fn mz_bounds_for_window_group_core(&self, window_group: u32) -> Option<(f32, f32)> {
   760	        let mut lo = f32::INFINITY;
   761	        let mut hi = f32::NEG_INFINITY;
   762	        let mut hit = false;
   763	        for w in &self.dia_ms_ms_windows {
   764	            if w.window_group == window_group {
   765	                let c = w.isolation_mz as f32;
   766	                let half = 0.5f32 * (w.isolation_width as f32);
   767	                lo = lo.min(c - half);
   768	                hi = hi.max(c + half);
   769	                hit = true;
   770	            }
   771	        }
   772	        if hit && hi > lo && lo.is_finite() && hi.is_finite() {
   773	            Some((lo, hi))
   774	        } else {
   775	            None

exec
/bin/bash -lc "rg -n \"SWATHMethod|offset 40|20-byte|record|calibration\" /home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07 -g '*.md' -g '*.rs'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/README.md:11:  (`/MethodSubtree/Method1/DeviceMethod0/SWATHMethod`).
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/README.md:12:- **TOF → m/z calibration** coefficients
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/README.md:25:  61-window SWATH, Zeno attributes, TOF calibration).
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/README.md:29:the windows + calibration that let the simulator reproduce a real run's
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/README.md:54:SWATH windows (399.5–899.9 m/z) and TOF calibration (4.898e-4 / -12.90).
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/main.rs:1://! `sciexwiff <file.wiff>` — dump the SWATH window scheme + TOF calibration.
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/main.rs:27:    if let Some(c) = m.tof_calibration {
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:2://! (SWATH isolation windows + TOF calibration). See `README.md` for the scope
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:25:/// TOF → m/z calibration, of the form `m/z = (coef1·tof + coef2)²`.
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:36:    pub tof_calibration: Option<TofCalibration>,
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:53:/// calibration. The SWATH windows live in
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:54:/// `/MethodSubtree/Method1/DeviceMethod0/SWATHMethod` as 20-byte records
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:55:/// `{ f64 lower, f64 upper, u32 }` starting at offset 40; the calibration in
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:56:/// `/SampleSubtree/Sample1/TOFCalibrationData` (24-byte records from offset 32).
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:61:    if let Some(sw) = read_stream(&mut comp, "/MethodSubtree/Method1/DeviceMethod0/SWATHMethod") {
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:79:    let tof_calibration = read_stream(&mut comp, "/SampleSubtree/Sample1/TOFCalibrationData")
/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:91:        tof_calibration,

codex
**Findings**

1. **High: `read_method` silently accepts truncated/corrupt SWATH streams and can return a partial acquisition method.**  
   [lib.rs](/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:61) derives record count solely from `(len - 40) / 20`, ignores trailing partial bytes, and silently drops malformed/non-finite records. A truncated 60-window method may therefore become a valid-looking 59-window method. Require `len >= 40`, `(len - 40) % 20 == 0`, validate any encoded header count if available, and return `InvalidData` when any expected record is malformed.

2. **High: both extractors can return `Ok` with a scheme that immediately fails `validate()`.**  
   Bruker fields are copied without checking widths, centers, CE, scan ordering, or within-group overlap at [scheme.rs](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:451). SCIEX caller timing is copied unchecked at [scheme.rs](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:586). Consequently malformed vendor rows, `cycle_time_s <= 0`, non-finite timing, or non-positive gradient produce successful extraction followed by validation failure. Construct, call `validate()`, and convert failure to `io::ErrorKind::InvalidData` before returning.

3. **Medium: Bruker cycle-time inference is fragile in the presence of duplicate, non-finite, or anomalous precursor times.**  
   [scheme.rs](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:485) sorts using a comparator that treats NaN as equal, then uses only the first two entries. Duplicate earliest times cause rejection even when later distinct precursor frames exist; one anomalous first interval determines the entire cycle. Filter finite times, sort with `total_cmp`, deduplicate, calculate all positive consecutive gaps, and choose a robust representative such as the median. `MsMsType == 0` is consistent with the supplied rustdf/Bruker contract, so that predicate itself is not a defect.

4. **Medium: Bruker event order is inferred from numeric window-group IDs rather than observed frame order.**  
   The `BTreeMap` iteration at [scheme.rs](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:475) assumes ascending `WindowGroup` equals physical acquisition order. The scheme model explicitly represents an ordered cycle, but `DiaFrameMsMsInfo`/frame times are not consulted. Files with arbitrary or reused group numbering will produce the wrong event order. Order groups by their first MS2-frame occurrence in the first bounded cycle.

5. **Medium: stream I/O and structural errors are collapsed into “optional/missing.”**  
   [lib.rs](/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:44) converts both `open_stream` and `read_to_end` errors to `None`. This is defensible for a genuinely optional calibration stream, but not for the required SWATH stream, and it hides CFB corruption or short reads. Return `io::Result<Option<Vec<u8>>>`; treat missing calibration as `None`, malformed calibration as `InvalidData`, and missing/unreadable SWATH as an error.

6. **Low: accepted finite SCIEX bounds can overflow during conversion.**  
   `read_method` accepts any finite `lo < hi`, but [center_mz](/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:17) and [width_mz](/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:20) can overflow to infinity for adversarial values, again yielding an invalid successful extraction. Validate finite center/width or retain lower/upper bounds directly.

I found no direct panic from short record slices: `get()` protects those accesses, and the subsequent `try_into().unwrap()` receives exactly eight bytes.
tokens used
57,439
**Findings**

1. **High: `read_method` silently accepts truncated/corrupt SWATH streams and can return a partial acquisition method.**  
   [lib.rs](/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:61) derives record count solely from `(len - 40) / 20`, ignores trailing partial bytes, and silently drops malformed/non-finite records. A truncated 60-window method may therefore become a valid-looking 59-window method. Require `len >= 40`, `(len - 40) % 20 == 0`, validate any encoded header count if available, and return `InvalidData` when any expected record is malformed.

2. **High: both extractors can return `Ok` with a scheme that immediately fails `validate()`.**  
   Bruker fields are copied without checking widths, centers, CE, scan ordering, or within-group overlap at [scheme.rs](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:451). SCIEX caller timing is copied unchecked at [scheme.rs](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:586). Consequently malformed vendor rows, `cycle_time_s <= 0`, non-finite timing, or non-positive gradient produce successful extraction followed by validation failure. Construct, call `validate()`, and convert failure to `io::ErrorKind::InvalidData` before returning.

3. **Medium: Bruker cycle-time inference is fragile in the presence of duplicate, non-finite, or anomalous precursor times.**  
   [scheme.rs](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:485) sorts using a comparator that treats NaN as equal, then uses only the first two entries. Duplicate earliest times cause rejection even when later distinct precursor frames exist; one anomalous first interval determines the entire cycle. Filter finite times, sort with `total_cmp`, deduplicate, calculate all positive consecutive gaps, and choose a robust representative such as the median. `MsMsType == 0` is consistent with the supplied rustdf/Bruker contract, so that predicate itself is not a defect.

4. **Medium: Bruker event order is inferred from numeric window-group IDs rather than observed frame order.**  
   The `BTreeMap` iteration at [scheme.rs](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:475) assumes ascending `WindowGroup` equals physical acquisition order. The scheme model explicitly represents an ordered cycle, but `DiaFrameMsMsInfo`/frame times are not consulted. Files with arbitrary or reused group numbering will produce the wrong event order. Order groups by their first MS2-frame occurrence in the first bounded cycle.

5. **Medium: stream I/O and structural errors are collapsed into “optional/missing.”**  
   [lib.rs](/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:44) converts both `open_stream` and `read_to_end` errors to `None`. This is defensible for a genuinely optional calibration stream, but not for the required SWATH stream, and it hides CFB corruption or short reads. Return `io::Result<Option<Vec<u8>>>`; treat missing calibration as `None`, malformed calibration as `InvalidData`, and missing/unreadable SWATH as an error.

6. **Low: accepted finite SCIEX bounds can overflow during conversion.**  
   `read_method` accepts any finite `lo < hi`, but [center_mz](/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:17) and [width_mz](/home/administrator/.cargo/git/checkouts/sciexwiff-765fdc7711006f90/4e8ff07/src/lib.rs:20) can overflow to infinity for adversarial values, again yielding an invalid successful extraction. Validate finite center/width or retain lower/upper bounds directly.

I found no direct panic from short record slices: `get()` protects those accesses, and the subsequent `try_into().unwrap()` receives exactly eight bytes.
