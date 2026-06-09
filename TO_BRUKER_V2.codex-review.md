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
session id: 019eab32-7179-79e3-af5d-b09afe89d1ca
--------
user
This is a REVISION of code you reviewed last round. Your prior review of to_bruker_windows found (High) that renumbering window groups 1..N lost the source Bruker WindowGroup identity, breaking round-trip fidelity with DiaFrameMsMsInfo, and that the round-trip test masked it via canonical normalization. The fix just applied: DiaMs2Frame now has a vendor_group_id: Option<u32> field; from_bruker_d sets it to the source WindowGroup (BTreeMap key), all other constructors set None; to_bruker_windows uses frame.vendor_group_id.unwrap_or(sequential 1..N) and now calls validate() first; the round-trip golden test now compares EXACT WindowGroup ids (no normalization).

Verify ONLY whether this revision is correct and complete; skim the rest. Focus: 1) does vendor_group_id genuinely fix the identity-loss bug end-to-end — is it set on EVERY from_bruker_d frame, threaded through all 7 DiaMs2Frame construction sites correctly (from_window_table, from_thermo_raw, from_sciex_wiff, from_bruker_d, and the 3 test sites), and used correctly in to_bruker_windows? 2) any NEW bug introduced by the field addition or the to_bruker_windows changes (e.g. validate() now called first - does that change error semantics or reject previously-accepted valid Bruker schemes? could a from_bruker_d scheme fail its own validate() due to within-frame overlap or mz_range containment on real PASEF window groups?). 3) is the unwrap_or fallback sound when SOME frames have vendor_group_id and others don't (mixed)? could that collide a preserved id with a sequential one? 4) does the exact-id round-trip test now actually prove fidelity? 5) the from_window_table builder doesn't expose vendor_group_id (always None) - is that a gap for injected timsTOF schemes that need specific group ids? Concrete, ranked by severity, ~500 words.

<stdin>
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
    /// The vendor's native group id for this frame, preserved for round-trip
    /// fidelity (Bruker `WindowGroup`). `None` for vendors without one. The cycle
    /// order is authoritative for *simultaneity/ordering*; this is *identity* only
    /// — needed so a Bruker adapter regenerates the original `WindowGroup` values
    /// that companion tables (`DiaFrameMsMsInfo`) reference.
    pub vendor_group_id: Option<u32>,
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
                vendor_group_id: None,
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
    /// Render the scheme's MS2 windows as Bruker `DiaFrameMsMsWindows` rows — the
    /// backward-compatibility adapter for the existing timsTOF write path.
    ///
    /// Each MS2 frame is one window group; its windows must carry `TimsMobility`
    /// geometry (the Bruker-grid scan ranges). The `WindowGroup` id is the frame's
    /// preserved [`DiaMs2Frame::vendor_group_id`] when present (so a Bruker
    /// round-trip keeps the ids that `DiaFrameMsMsInfo` references), else a 1..N
    /// fallback in cycle order. A `Linear` CE policy is resolved at the window
    /// center (a lossy materialization, not an inverse); `Unknown` CE and
    /// non-timsTOF schemes are rejected. The scheme is validated first.
    ///
    /// Row order is not meaningful (the SQLite table has none); the companion
    /// frame→group table `DiaFrameMsMsInfo` needs the run frame schedule and is a
    /// separate step.
    pub fn to_bruker_windows(&self) -> Result<Vec<crate::data::meta::DiaMsMsWindow>, String> {
        self.validate()?;
        if self.instrument != InstrumentKind::TimsTofDia {
            return Err("to_bruker_windows requires a timsTOF (TimsTofDia) scheme".into());
        }
        let mut rows = Vec::new();
        let mut seq: u32 = 0;
        for ev in &self.cycle {
            if let AcquisitionEvent::DiaMs2Frame(frame) = ev {
                seq += 1;
                let group = frame.vendor_group_id.unwrap_or(seq);
                for w in &frame.windows {
                    let (scan_num_begin, scan_num_end) = match w.geometry {
                        DiaGeometry::TimsMobility {
                            scan_start,
                            scan_end,
                        } => (scan_start, scan_end),
                        DiaGeometry::MzOnly => {
                            return Err("timsTOF window lacks mobility geometry".into())
                        }
                    };
                    let collision_energy = match w.collision_energy {
                        CollisionEnergyPolicy::Value(v) => v,
                        CollisionEnergyPolicy::Linear { .. } => w
                            .collision_energy
                            .at(w.isolation.center_mz)
                            .ok_or("could not resolve linear CE")?,
                        CollisionEnergyPolicy::Unknown => {
                            return Err("window has unknown collision energy".into())
                        }
                    };
                    rows.push(crate::data::meta::DiaMsMsWindow {
                        window_group: group,
                        scan_num_begin,
                        scan_num_end,
                        isolation_mz: w.isolation.center_mz,
                        isolation_width: w.isolation.width_mz,
                        collision_energy,
                    });
                }
            }
        }
        Ok(rows)
    }

    /// Extract the acquisition scheme from a real Bruker timsTOF DIA `.d`.
    ///
    /// Reads `DiaFrameMsMsWindows` and the frame table. Each **window group**
    /// becomes one [`DiaMs2Frame`] holding its mobility-partitioned windows
    /// (`TimsMobility` geometry — the scan ranges are Bruker-grid coordinates),
    /// preceded by a single MS1 event. Cycle timing comes from the spacing of
    /// the precursor (MS1) frames. The returned scheme is validated.
    ///
    /// Note: window groups are ordered by ascending `WindowGroup` id, which is
    /// the DIA-PASEF acquisition order in practice; a file that reuses or
    /// permutes group numbering would need ordering by first MS2-frame
    /// occurrence (via `DiaFrameMsMsInfo`) instead.
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
        for (group, ws) in by_group {
            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: ws,
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: Some(group), // preserve Bruker WindowGroup identity
            }));
        }

        // Cycle timing from the precursor (MS1, MsMsType == 0) frame spacing.
        // Use the MEDIAN of the positive gaps between distinct, finite precursor
        // times, so one anomalous interval doesn't set the whole cycle.
        let mut prec_times: Vec<f64> = frames
            .iter()
            .filter(|f| f.ms_ms_type == 0 && f.time.is_finite())
            .map(|f| f.time)
            .collect();
        prec_times.sort_by(f64::total_cmp);
        prec_times.dedup();
        let mut gaps: Vec<f64> = prec_times
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|g| *g > 0.0)
            .collect();
        if gaps.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "could not determine cycle time (need >= 2 distinct precursor frames)",
            ));
        }
        gaps.sort_by(f64::total_cmp);
        let cycle_time_s = gaps[gaps.len() / 2];

        let times: Vec<f64> = frames.iter().map(|f| f.time).filter(|t| t.is_finite()).collect();
        let start_time_s = times.iter().copied().fold(f64::INFINITY, f64::min);
        let gradient_length_s = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let scheme = AcquisitionScheme {
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
        };
        scheme
            .validate()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(scheme)
    }
}

#[cfg(feature = "sciex")]
impl AcquisitionScheme {
    /// Extract the SWATH scheme from a real SCIEX ZenoTOF `.wiff` method.
    ///
    /// Each SWATH window becomes a single-window [`DiaMs2Frame`] (no ion
    /// mobility) preceded by an MS1.
    ///
    /// **Collision energy** is caller-supplied (`collision_energy`), because
    /// SCIEX SWATH uses *rolling* CE computed by the instrument from an m/z
    /// formula at acquisition time — it is **not** stored per-window in the
    /// `.wiff` method (verified: the per-window MS2 parameter streams are
    /// byte-identical across windows). Pass [`CollisionEnergyPolicy::Unknown`]
    /// to leave it unset, or a [`CollisionEnergyPolicy::Linear`] rolling model.
    /// The `.wiff` method also lacks run timing, so `cycle_time_s` /
    /// `gradient_length_s` are caller-supplied. The returned scheme is validated.
    pub fn from_sciex_wiff<P: AsRef<std::path::Path>>(
        path: P,
        cycle_time_s: f64,
        gradient_length_s: f64,
        collision_energy: CollisionEnergyPolicy,
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
                    collision_energy,
                    geometry: DiaGeometry::MzOnly,
                }],
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: None,
            }));
        }
        let scheme = AcquisitionScheme {
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
                    "extracted from SCIEX .wiff method ({n} SWATH windows; CE caller-supplied, timing caller-supplied)"
                ),
            },
        };
        scheme
            .validate()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(scheme)
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
                    vendor_group_id: None,
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

        let scheme = AcquisitionScheme {
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
        };
        scheme
            .validate()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(scheme)
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
            vendor_group_id: None,
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
                vendor_group_id: None,
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
            vendor_group_id: None,
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
    // Gated: round-trip a real Bruker DIA .d through the scheme and back to the
    // DiaFrameMsMsWindows rows; the regenerated table must match the source.
    #[test]
    fn bruker_windows_round_trip() {
        let d = match std::env::var("TIMSIM_BRUKER_DIA_D") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP bruker_windows_round_trip: set TIMSIM_BRUKER_DIA_D");
                return;
            }
        };
        let scheme = AcquisitionScheme::from_bruker_d(&d).expect("extract");
        let regenerated = scheme.to_bruker_windows().expect("to_bruker_windows");
        let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
        assert_eq!(regenerated.len(), original.len(), "row count differs");

        // Compare as a sorted multiset of EXACT field tuples, including the
        // WindowGroup id (preserved via vendor_group_id) — no normalization, so a
        // real id mismatch would fail. Row order in the table is not guaranteed.
        fn tuples(
            rows: &[crate::data::meta::DiaMsMsWindow],
        ) -> Vec<(u32, u32, u32, u64, u64, u64)> {
            let mut out: Vec<_> = rows
                .iter()
                .map(|r| {
                    (
                        r.window_group,
                        r.scan_num_begin,
                        r.scan_num_end,
                        r.isolation_mz.to_bits(),
                        r.isolation_width.to_bits(),
                        r.collision_energy.to_bits(),
                    )
                })
                .collect();
            out.sort_unstable();
            out
        }
        assert_eq!(
            tuples(&regenerated),
            tuples(&original),
            "round-trip windows differ from source (incl. WindowGroup id)"
        );
        eprintln!(
            "bruker_windows_round_trip OK: {} rows match across {} groups",
            original.len(),
            scheme
                .cycle
                .iter()
                .filter(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_)))
                .count()
        );
    }

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
        // Default: CE unknown (rolling CE isn't in the .wiff method).
        let s = AcquisitionScheme::from_sciex_wiff(&wiff, 3.5, 1800.0, CollisionEnergyPolicy::Unknown)
            .expect("extract");
        s.validate().expect("valid scheme");
        assert_eq!(s.instrument, InstrumentKind::SciexZenoTof);
        assert_eq!(s.ms1_count(), 1);
        let n = s.windows().count();
        assert!(n > 10, "expected many SWATH windows, got {n}");
        for w in s.windows() {
            assert!(matches!(w.geometry, DiaGeometry::MzOnly));
            assert!(matches!(w.collision_energy, CollisionEnergyPolicy::Unknown));
            assert!(w.collision_energy.at(w.isolation.center_mz).is_none());
        }
        // Supplying a rolling-CE Linear model gives a resolvable, finite CE.
        let rolling = CollisionEnergyPolicy::Linear { intercept: 5.0, slope_per_mz: 0.045 };
        let s2 = AcquisitionScheme::from_sciex_wiff(&wiff, 3.5, 1800.0, rolling).expect("extract");
        s2.validate().expect("valid with rolling CE");
        for w in s2.windows() {
            let ce = w.collision_energy.at(w.isolation.center_mz).expect("resolvable CE");
            assert!(ce.is_finite() && ce > 0.0);
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
</stdin>
codex
I’ll verify the actual repository diff and all `DiaMs2Frame` construction/use sites, then run the focused tests or compile checks available.
exec
/bin/bash -lc "rg -n \"DiaMs2Frame\\s*\\{|vendor_group_id|to_bruker_windows|from_bruker_d|from_window_table|from_thermo_raw|from_sciex_wiff\" ." in /scratch/timsim-demo/SUBMISSION/rustims
exec
/bin/bash -lc 'git diff -- . && git status --short' in /scratch/timsim-demo/SUBMISSION/rustims
exec
/bin/bash -lc 'rg -n "struct DiaMsMsWindow|read_dia_ms_ms_windows|DiaFrameMsMsInfo|WindowGroup|window_group" src tests crates 2>/dev/null' in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 198ms:
./TO_BRUKER.codex-review.md:14:Review ONLY the newly added code in this Rust file (the rest was already reviewed): the method AcquisitionScheme::to_bruker_windows (a backward-compat adapter that renders a vendor-neutral DIA scheme back to Bruker DiaFrameMsMsWindows rows) and its gated unit test bruker_windows_round_trip. Skim the rest only for interaction.
./TO_BRUKER.codex-review.md:16:Trusted context: crate::data::meta::DiaMsMsWindow { window_group:u32, scan_num_begin:u32, scan_num_end:u32, isolation_mz:f64, isolation_width:f64, collision_energy:f64 } and read_dia_ms_ms_windows(folder)->Vec<DiaMsMsWindow> read the Bruker SQLite DiaFrameMsMsWindows table (no ORDER BY). from_bruker_d builds the scheme by grouping those rows by window_group (BTreeMap ascending) into one DiaMs2Frame per group with TimsMobility geometry. The scheme model: cycle is an ordered Vec<AcquisitionEvent> = Ms1 | DiaMs2Frame(windows:Vec<DiaWindow{isolation:{center_mz,width_mz}, collision_energy:CollisionEnergyPolicy{Value|Linear|Unknown}, geometry:DiaGeometry{MzOnly|TimsMobility{scan_start,scan_end}}}>). to_bruker_windows numbers window groups 1..N in cycle (frame) order; the round-trip test normalizes window-group ids to canonical 1..N on both sides and compares sorted exact-bit field tuples.
./TO_BRUKER.codex-review.md:18:Focus: 1) is to_bruker_windows a FAITHFUL inverse of from_bruker_d for real DIA-PASEF data — specifically the window-group renumbering (1..N by frame order vs original ids): when does the normalized round-trip hide a real id mismatch that would break the actual .d the Python builder writes (which uses reference WindowGroup ids)? Is losing the original WindowGroup id in the scheme model a real data-loss bug for backward compat? 2) CE handling — Value passthrough, Linear resolved at center (is center the right point vs Bruker's stored per-window CE?), Unknown rejected; any lossy/incorrect cases. 3) within-frame and cross-frame window ORDER preservation (SQL has no ORDER BY; from_bruker_d preserves insertion order in the Vec; does to_bruker_windows + the test correctly handle when source rows are interleaved across groups vs grouped?). 4) the test's canonical() normalization — does it actually prove faithfulness, or could it pass while masking a real regression? 5) any correctness bug, panic, or float-equality fragility. Concrete, ranked by severity, ~600 words.
./TO_BRUKER.codex-review.md:140:pub struct DiaMs2Frame {
./TO_BRUKER.codex-review.md:234:    pub fn from_window_table(
./TO_BRUKER.codex-review.md:243:            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./TO_BRUKER.codex-review.md:454:    pub fn to_bruker_windows(&self) -> Result<Vec<crate::data::meta::DiaMsMsWindow>, String> {
./TO_BRUKER.codex-review.md:456:            return Err("to_bruker_windows requires a timsTOF (TimsTofDia) scheme".into());
./TO_BRUKER.codex-review.md:509:    pub fn from_bruker_d<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
./TO_BRUKER.codex-review.md:554:            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./TO_BRUKER.codex-review.md:631:    pub fn from_sciex_wiff<P: AsRef<std::path::Path>>(
./TO_BRUKER.codex-review.md:660:            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./TO_BRUKER.codex-review.md:701:    pub fn from_thermo_raw<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
./TO_BRUKER.codex-review.md:772:                cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./TO_BRUKER.codex-review.md:860:    fn from_window_table_validates() {
./TO_BRUKER.codex-review.md:861:        let s = AcquisitionScheme::from_window_table(
./TO_BRUKER.codex-review.md:885:        let frame = DiaMs2Frame {
./TO_BRUKER.codex-review.md:939:            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./TO_BRUKER.codex-review.md:975:        let bad_dur = AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./TO_BRUKER.codex-review.md:986:    fn from_bruker_d_extracts_cycle() {
./TO_BRUKER.codex-review.md:990:                eprintln!("SKIP from_bruker_d_extracts_cycle: set TIMSIM_BRUKER_DIA_D=<dia .d>");
./TO_BRUKER.codex-review.md:994:        let s = AcquisitionScheme::from_bruker_d(&d).expect("extract");
./TO_BRUKER.codex-review.md:1009:            "from_bruker_d OK: TimsTofDia, {} frames, {} windows ({}), mz {:.1}..{:.1}",
./TO_BRUKER.codex-review.md:1027:        let scheme = AcquisitionScheme::from_bruker_d(&d).expect("extract");
./TO_BRUKER.codex-review.md:1028:        let regenerated = scheme.to_bruker_windows().expect("to_bruker_windows");
./TO_BRUKER.codex-review.md:1075:    fn from_sciex_wiff_extracts_windows() {
./TO_BRUKER.codex-review.md:1079:                eprintln!("SKIP from_sciex_wiff_extracts_windows: set TIMSIM_SCIEX_WIFF=<.wiff>");
./TO_BRUKER.codex-review.md:1084:        let s = AcquisitionScheme::from_sciex_wiff(&wiff, 3.5, 1800.0, CollisionEnergyPolicy::Unknown)
./TO_BRUKER.codex-review.md:1098:        let s2 = AcquisitionScheme::from_sciex_wiff(&wiff, 3.5, 1800.0, rolling).expect("extract");
./TO_BRUKER.codex-review.md:1105:            "from_sciex_wiff OK: SciexZenoTof, {} SWATH windows, mz {:.1}..{:.1}",
./TO_BRUKER.codex-review.md:1112:    fn from_thermo_raw_extracts_cycle() {
./TO_BRUKER.codex-review.md:1116:                eprintln!("SKIP from_thermo_raw_extracts_cycle: set TIMSIM_ASTRAL_TEMPLATE");
./TO_BRUKER.codex-review.md:1120:        let s = AcquisitionScheme::from_thermo_raw(&template).expect("extract");
./TO_BRUKER.codex-review.md:1136:            "from_thermo_raw OK: instrument={:?}, {} MS2 windows, mz {:.1}..{:.1}, cycle {:.4}s",
./TO_BRUKER.codex-review.md:1147:`from_bruker_d` discards the original `WindowGroup` value when converting each `BTreeMap` entry into a `DiaMs2Frame`. `to_bruker_windows` then invents IDs `1..N`.
./TO_BRUKER.codex-review.md:1159:SQL row order is unspecified. `from_bruker_d` orders frames by ascending group ID and preserves database-return order only within each group. `to_bruker_windows` emits rows grouped by frame, so an interleaved source sequence such as `G1-A, G2-A, G1-B` becomes `G1-A, G1-B, G2-A`.
./TO_BRUKER.codex-review.md:1163:4. **Medium: `to_bruker_windows` accepts structurally invalid schemes without validation.**  
./TO_BRUKER.codex-review.md:1175:`from_bruker_d` discards the original `WindowGroup` value when converting each `BTreeMap` entry into a `DiaMs2Frame`. `to_bruker_windows` then invents IDs `1..N`.
./TO_BRUKER.codex-review.md:1187:SQL row order is unspecified. `from_bruker_d` orders frames by ascending group ID and preserves database-return order only within each group. `to_bruker_windows` emits rows grouped by frame, so an interleaved source sequence such as `G1-A, G2-A, G1-B` becomes `G1-A, G1-B, G2-A`.
./TO_BRUKER.codex-review.md:1191:4. **Medium: `to_bruker_windows` accepts structurally invalid schemes without validation.**  
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:14:This is v2 of a vendor-neutral AcquisitionScheme design for a DIA mass-spec simulator (TimSim: Bruker timsTOF today, extending to Thermo Orbitrap Astral + SCIEX ZenoTOF). You reviewed v1; v2 applied your feedback: ordered cycle of events (Ms1Event/DiaMs2Event) instead of flat windows; DiaGeometry splits mobility out of the m/z IsolationWindow; CollisionEnergyPolicy enum (incl. Unknown) instead of scalar CE; MS1 promoted to a first-class event; scheme version + Provenance; an explicit to_bruker_tables() adapter + golden tests; and two usage modes (reference-derived layout+noise, and injected/user schemes). Context to trust: the Thermo/SCIEX Rust readers exist and are oracle-verified; thermorawfile::scan_event works per-scan on Astral (uniform 280-byte stride) so from_thermo_raw is NOT blocked; the ThermoRawWriter now authors MS2 isolation/CE via set_isolation (review item #6, fixed).
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:159:    pub fn from_bruker_d(path: &str) -> io::Result<Self>;      // DiaFrameMsMsWindows (+TimsMobility geometry)
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:161:    pub fn from_thermo_raw(path: &str) -> io::Result<Self>;    // walk one MS2 cycle via scan_event()
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:163:    pub fn from_sciex_wiff(path: &str) -> io::Result<Self>;    // SWATHMethod; CE => Unknown (rolling)
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:164:    pub fn from_window_table(rows, instrument) -> Self;        // injected / CSV
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:168:Each sets `provenance.source` accordingly. `from_thermo_raw` reads the first
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:175:  reference round trip: `from_thermo_raw(template)` → generate → `ThermoRawWriter`
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:215:  location works (uniform stride) so `from_thermo_raw` is unblocked.
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:230:- `from_thermo_raw(astral_template)`: N MS2 windows, monotonic centers, CE policy
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:232:- `from_sciex_wiff(zenotof)`: ~60 windows covering 399.5–899.9, CE `Unknown`.
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:233:- `from_bruker_d(ref)` → `to_bruker_tables()` golden-equal to today's Python path.
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:234:- End-to-end: `from_thermo_raw(template)` → generate 1×MS1 + N×MS2 → `ThermoRawWriter`
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:248:AcquisitionEvent::DiaMs2Frame {
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:319:AcquisitionEvent::DiaMs2Frame {
./EXTRACTORS.codex-review.md:14:Review the NEW code added to these two Rust files for a mass-spec simulator (TimSim): in scheme.rs, the extractors AcquisitionScheme::from_bruker_d and AcquisitionScheme::from_sciex_wiff (the data model, validate(), num_cycles, windows(), from_window_table, and from_thermo_raw were ALREADY reviewed — only skim them for interaction bugs). The second file is sciexwiff's new library API (read_method).
./EXTRACTORS.codex-review.md:16:Trusted context: rustdf reads Bruker TDF natively — read_dia_ms_ms_windows(folder)->Vec<DiaMsMsWindow{window_group,scan_num_begin,scan_num_end,isolation_mz,isolation_width,collision_energy}> and read_meta_data_sql(folder)->Vec<FrameMeta{id,time:f64,ms_ms_type:i64,...}> (MsMsType==0 = MS1/precursor frame). from_bruker_d groups windows by window_group into DiaMs2Frame{TimsMobility windows} and derives cycle time from precursor-frame spacing; verified to extract 15 frames/36 windows from a real DIA-PASEF .d. from_sciex_wiff maps SWATH windows to single MzOnly frames with CE Unknown (SCIEX rolling CE not in the method) and caller-supplied timing; verified 60 windows on a real ZenoTOF .wiff. validate() (already reviewed) requires exactly one MS1 first then >=1 MS2 frame, finite/positive widths, window edges within mz_range, CE finite>=0 for Value/resolved-Linear, multi-window frames need TimsMobility on every window and no within-frame m/z+mobility overlap, start<=gradient. sciexwiff::read_method opens a .wiff OLE2 (cfb crate) and parses SWATHMethod (20B records from off 40) + TOFCalibrationData.
./EXTRACTORS.codex-review.md:18:Focus: 1) from_bruker_d correctness — window-group grouping, cycle-time from precursor spacing (sorted; what if precursor times equal/unsorted/duplicated, or DIA uses a different MsMsType for MS1?), start/gradient via fold over f64 (NaN handling), whether the produced scheme always passes validate() (e.g. within-frame overlap on real PASEF groups; windows-within-mz_range by construction), empty/edge frames. 2) from_sciex_wiff — caller-supplied timing validation, window mapping, the Unknown-CE contract, mz_range. 3) sciexwiff read_method — bounds/stride safety on untrusted .wiff bytes, partial/short streams, the SWATHMethod record count derivation, calibration optionality. 4) any panic on malformed vendor files, and any case where an extractor returns a scheme that then fails validate() (surprising for a caller). Concrete, ranked by severity, cap ~700 words.
./EXTRACTORS.codex-review.md:141:pub struct DiaMs2Frame {
./EXTRACTORS.codex-review.md:235:    pub fn from_window_table(
./EXTRACTORS.codex-review.md:244:            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./EXTRACTORS.codex-review.md:452:    pub fn from_bruker_d<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
./EXTRACTORS.codex-review.md:497:            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./EXTRACTORS.codex-review.md:564:    pub fn from_sciex_wiff<P: AsRef<std::path::Path>>(
./EXTRACTORS.codex-review.md:592:            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./EXTRACTORS.codex-review.md:629:    pub fn from_thermo_raw<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
./EXTRACTORS.codex-review.md:700:                cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./EXTRACTORS.codex-review.md:784:    fn from_window_table_validates() {
./EXTRACTORS.codex-review.md:785:        let s = AcquisitionScheme::from_window_table(
./EXTRACTORS.codex-review.md:809:        let frame = DiaMs2Frame {
./EXTRACTORS.codex-review.md:863:            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./EXTRACTORS.codex-review.md:899:        let bad_dur = AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./EXTRACTORS.codex-review.md:910:    fn from_bruker_d_extracts_cycle() {
./EXTRACTORS.codex-review.md:914:                eprintln!("SKIP from_bruker_d_extracts_cycle: set TIMSIM_BRUKER_DIA_D=<dia .d>");
./EXTRACTORS.codex-review.md:918:        let s = AcquisitionScheme::from_bruker_d(&d).expect("extract");
./EXTRACTORS.codex-review.md:933:            "from_bruker_d OK: TimsTofDia, {} frames, {} windows ({}), mz {:.1}..{:.1}",
./EXTRACTORS.codex-review.md:942:    fn from_sciex_wiff_extracts_windows() {
./EXTRACTORS.codex-review.md:946:                eprintln!("SKIP from_sciex_wiff_extracts_windows: set TIMSIM_SCIEX_WIFF=<.wiff>");
./EXTRACTORS.codex-review.md:950:        let s = AcquisitionScheme::from_sciex_wiff(&wiff, 3.5, 1800.0).expect("extract");
./EXTRACTORS.codex-review.md:963:            "from_sciex_wiff OK: SciexZenoTof, {} SWATH windows, mz {:.1}..{:.1}",
./EXTRACTORS.codex-review.md:970:    fn from_thermo_raw_extracts_cycle() {
./EXTRACTORS.codex-review.md:974:                eprintln!("SKIP from_thermo_raw_extracts_cycle: set TIMSIM_ASTRAL_TEMPLATE");
./EXTRACTORS.codex-review.md:978:        let s = AcquisitionScheme::from_thermo_raw(&template).expect("extract");
./EXTRACTORS.codex-review.md:994:            "from_thermo_raw OK: instrument={:?}, {} MS2 windows, mz {:.1}..{:.1}, cycle {:.4}s",
./EXTRACTORS.codex-review.md:1252:   431	    pub fn from_bruker_d<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
./EXTRACTORS.codex-review.md:1297:   476	            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./EXTRACTORS.codex-review.md:1364:   543	    pub fn from_sciex_wiff<P: AsRef<std::path::Path>>(
./EXTRACTORS.codex-review.md:1456:   543	    pub fn from_sciex_wiff<P: AsRef<std::path::Path>>(
./EXTRACTORS.codex-review.md:1484:   571	            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./EXTRACTORS.codex-review.md:1521:   608	    pub fn from_thermo_raw<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
./TO_BRUKER_V2.codex-review.md:14:This is a REVISION of code you reviewed last round. Your prior review of to_bruker_windows found (High) that renumbering window groups 1..N lost the source Bruker WindowGroup identity, breaking round-trip fidelity with DiaFrameMsMsInfo, and that the round-trip test masked it via canonical normalization. The fix just applied: DiaMs2Frame now has a vendor_group_id: Option<u32> field; from_bruker_d sets it to the source WindowGroup (BTreeMap key), all other constructors set None; to_bruker_windows uses frame.vendor_group_id.unwrap_or(sequential 1..N) and now calls validate() first; the round-trip golden test now compares EXACT WindowGroup ids (no normalization).
./TO_BRUKER_V2.codex-review.md:16:Verify ONLY whether this revision is correct and complete; skim the rest. Focus: 1) does vendor_group_id genuinely fix the identity-loss bug end-to-end — is it set on EVERY from_bruker_d frame, threaded through all 7 DiaMs2Frame construction sites correctly (from_window_table, from_thermo_raw, from_sciex_wiff, from_bruker_d, and the 3 test sites), and used correctly in to_bruker_windows? 2) any NEW bug introduced by the field addition or the to_bruker_windows changes (e.g. validate() now called first - does that change error semantics or reject previously-accepted valid Bruker schemes? could a from_bruker_d scheme fail its own validate() due to within-frame overlap or mz_range containment on real PASEF window groups?). 3) is the unwrap_or fallback sound when SOME frames have vendor_group_id and others don't (mixed)? could that collide a preserved id with a sequential one? 4) does the exact-id round-trip test now actually prove fidelity? 5) the from_window_table builder doesn't expose vendor_group_id (always None) - is that a gap for injected timsTOF schemes that need specific group ids? Concrete, ranked by severity, ~500 words.
./TO_BRUKER_V2.codex-review.md:138:pub struct DiaMs2Frame {
./TO_BRUKER_V2.codex-review.md:148:    pub vendor_group_id: Option<u32>,
./TO_BRUKER_V2.codex-review.md:238:    pub fn from_window_table(
./TO_BRUKER_V2.codex-review.md:247:            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./TO_BRUKER_V2.codex-review.md:252:                vendor_group_id: None,
./TO_BRUKER_V2.codex-review.md:454:    /// preserved [`DiaMs2Frame::vendor_group_id`] when present (so a Bruker
./TO_BRUKER_V2.codex-review.md:463:    pub fn to_bruker_windows(&self) -> Result<Vec<crate::data::meta::DiaMsMsWindow>, String> {
./TO_BRUKER_V2.codex-review.md:466:            return Err("to_bruker_windows requires a timsTOF (TimsTofDia) scheme".into());
./TO_BRUKER_V2.codex-review.md:473:                let group = frame.vendor_group_id.unwrap_or(seq);
./TO_BRUKER_V2.codex-review.md:520:    pub fn from_bruker_d<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
./TO_BRUKER_V2.codex-review.md:565:            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./TO_BRUKER_V2.codex-review.md:570:                vendor_group_id: Some(group), // preserve Bruker WindowGroup identity
./TO_BRUKER_V2.codex-review.md:643:    pub fn from_sciex_wiff<P: AsRef<std::path::Path>>(
./TO_BRUKER_V2.codex-review.md:672:            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./TO_BRUKER_V2.codex-review.md:681:                vendor_group_id: None,
./TO_BRUKER_V2.codex-review.md:714:    pub fn from_thermo_raw<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
./TO_BRUKER_V2.codex-review.md:785:                cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./TO_BRUKER_V2.codex-review.md:794:                    vendor_group_id: None,
./TO_BRUKER_V2.codex-review.md:874:    fn from_window_table_validates() {
./TO_BRUKER_V2.codex-review.md:875:        let s = AcquisitionScheme::from_window_table(
./TO_BRUKER_V2.codex-review.md:899:        let frame = DiaMs2Frame {
./TO_BRUKER_V2.codex-review.md:904:            vendor_group_id: None,
./TO_BRUKER_V2.codex-review.md:954:            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./TO_BRUKER_V2.codex-review.md:959:                vendor_group_id: None,
./TO_BRUKER_V2.codex-review.md:991:        let bad_dur = AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./TO_BRUKER_V2.codex-review.md:996:            vendor_group_id: None,
./TO_BRUKER_V2.codex-review.md:1003:    fn from_bruker_d_extracts_cycle() {
./TO_BRUKER_V2.codex-review.md:1007:                eprintln!("SKIP from_bruker_d_extracts_cycle: set TIMSIM_BRUKER_DIA_D=<dia .d>");
./TO_BRUKER_V2.codex-review.md:1011:        let s = AcquisitionScheme::from_bruker_d(&d).expect("extract");
./TO_BRUKER_V2.codex-review.md:1026:            "from_bruker_d OK: TimsTofDia, {} frames, {} windows ({}), mz {:.1}..{:.1}",
./TO_BRUKER_V2.codex-review.md:1044:        let scheme = AcquisitionScheme::from_bruker_d(&d).expect("extract");
./TO_BRUKER_V2.codex-review.md:1045:        let regenerated = scheme.to_bruker_windows().expect("to_bruker_windows");
./TO_BRUKER_V2.codex-review.md:1050:        // WindowGroup id (preserved via vendor_group_id) — no normalization, so a
./TO_BRUKER_V2.codex-review.md:1089:    fn from_sciex_wiff_extracts_windows() {
./TO_BRUKER_V2.codex-review.md:1093:                eprintln!("SKIP from_sciex_wiff_extracts_windows: set TIMSIM_SCIEX_WIFF=<.wiff>");
./TO_BRUKER_V2.codex-review.md:1098:        let s = AcquisitionScheme::from_sciex_wiff(&wiff, 3.5, 1800.0, CollisionEnergyPolicy::Unknown)
./TO_BRUKER_V2.codex-review.md:1112:        let s2 = AcquisitionScheme::from_sciex_wiff(&wiff, 3.5, 1800.0, rolling).expect("extract");
./TO_BRUKER_V2.codex-review.md:1119:            "from_sciex_wiff OK: SciexZenoTof, {} SWATH windows, mz {:.1}..{:.1}",
./TO_BRUKER_V2.codex-review.md:1126:    fn from_thermo_raw_extracts_cycle() {
./TO_BRUKER_V2.codex-review.md:1130:                eprintln!("SKIP from_thermo_raw_extracts_cycle: set TIMSIM_ASTRAL_TEMPLATE");
./TO_BRUKER_V2.codex-review.md:1134:        let s = AcquisitionScheme::from_thermo_raw(&template).expect("extract");
./TO_BRUKER_V2.codex-review.md:1150:            "from_thermo_raw OK: instrument={:?}, {} MS2 windows, mz {:.1}..{:.1}, cycle {:.4}s",
./TO_BRUKER_V2.codex-review.md:1160:/bin/bash -lc "rg -n \"DiaMs2Frame\\s*\\{|vendor_group_id|to_bruker_windows|from_bruker_d|from_window_table|from_thermo_raw|from_sciex_wiff\" ." in /scratch/timsim-demo/SUBMISSION/rustims
./ACQUISITION_SCHEME_PLAN.codex-review.md:113:    pub fn from_bruker_d(path: &str) -> io::Result<Self>;       // DiaFrameMsMsWindows
./ACQUISITION_SCHEME_PLAN.codex-review.md:115:    pub fn from_thermo_raw(path: &str) -> io::Result<Self>;     // walk MS2 scan_event()
./ACQUISITION_SCHEME_PLAN.codex-review.md:117:    pub fn from_sciex_wiff(path: &str) -> io::Result<Self>;     // SWATHMethod
./ACQUISITION_SCHEME_PLAN.codex-review.md:118:    pub fn from_window_table(...) -> Self;                       // hand-authored / CSV
./ACQUISITION_SCHEME_PLAN.codex-review.md:122:- `from_thermo_raw`: iterate the template's first full cycle of MS2 scans, read
./ACQUISITION_SCHEME_PLAN.codex-review.md:126:- `from_sciex_wiff`: `SWATHMethod` windows → `DiaWindow` (center = (lo+hi)/2,
./ACQUISITION_SCHEME_PLAN.codex-review.md:139:  `from_bruker_d(reference)` → identical behavior (the `mobility_scan_range`
./ACQUISITION_SCHEME_PLAN.codex-review.md:143:  two halves. A round trip is: `from_thermo_raw(template)` → generate → author
./ACQUISITION_SCHEME_PLAN.codex-review.md:168:6. **scan_event location for arbitrary scans** — `from_thermo_raw` needs to read
./ACQUISITION_SCHEME_PLAN.codex-review.md:171:   (MS1 events are longer). Does `from_thermo_raw` need the variable-stride walk
./ACQUISITION_SCHEME_PLAN.codex-review.md:176:- `from_thermo_raw(astral_template)` → assert N windows, monotonic centers,
./ACQUISITION_SCHEME_PLAN.codex-review.md:178:- `from_sciex_wiff(zenotof_wiff)` → assert ~60 windows covering 399.5–899.9.
./ACQUISITION_SCHEME_PLAN.codex-review.md:179:- `from_bruker_d(ref)` round-trips the existing `dia_ms_ms_windows` columns
./ACQUISITION_SCHEME_PLAN.codex-review.md:181:- End-to-end: `from_thermo_raw(template)` → generate a 1-MS1 + N-MS2 cycle →
./ACQUISITION_SCHEME_IMPL.codex-review.md:16:Context to trust: thermorawfile::scan_event(scan) returns {ms_order(1=MS1,2=MS2), analyzer(4=FTMS,7=ASTMS,0=ITMS), isolation_center, isolation_width, collision_energy} and works per-scan (uniform stride); RawFile exposes pub first_scan/last_scan/data_addr/bytes/index, and ScanIndexEntry has pub time/offset; packet profile_size is u32 at packet+4. from_thermo_raw is verified to extract 1 MS1 + 15 MS2 windows from a real Astral file.
./ACQUISITION_SCHEME_IMPL.codex-review.md:18:Focus: 1) from_thermo_raw correctness — the first-cycle boundary detection (first MS1 .. next MS1), RT/cycle_time computation, mz_range derivation from window edges, analyzer/data_mode mapping, and edge cases (no MS2 before first MS1, only one cycle in the file so cycle_time stays 0, MS2 before any MS1, scan_event returning None mid-cycle, instrument detection via any_astms). 2) validate() completeness and correctness vs the model's invariants — anything it should reject but doesn't (e.g. overlapping/duplicate windows, MS1-only cycle, non-finite mz centers vs range membership, Linear CE finiteness, num_cycles with start>gradient). 3) the windows() iterator and num_cycles() arithmetic (truncation, negatives). 4) API/model soundness for the two consumers (extraction vs injection) and for the deferred Bruker/SCIEX extractors. 5) any correctness/robustness bug. Concrete, ranked by severity, cap ~750 words.
./ACQUISITION_SCHEME_IMPL.codex-review.md:137:pub struct DiaMs2Frame {
./ACQUISITION_SCHEME_IMPL.codex-review.md:227:    pub fn from_window_table(
./ACQUISITION_SCHEME_IMPL.codex-review.md:236:            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./ACQUISITION_SCHEME_IMPL.codex-review.md:344:    pub fn from_thermo_raw<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
./ACQUISITION_SCHEME_IMPL.codex-review.md:407:                cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./ACQUISITION_SCHEME_IMPL.codex-review.md:482:    fn from_window_table_validates() {
./ACQUISITION_SCHEME_IMPL.codex-review.md:483:        let s = AcquisitionScheme::from_window_table(
./ACQUISITION_SCHEME_IMPL.codex-review.md:507:        let frame = DiaMs2Frame {
./ACQUISITION_SCHEME_IMPL.codex-review.md:541:    fn from_thermo_raw_extracts_cycle() {
./ACQUISITION_SCHEME_IMPL.codex-review.md:545:                eprintln!("SKIP from_thermo_raw_extracts_cycle: set TIMSIM_ASTRAL_TEMPLATE");
./ACQUISITION_SCHEME_IMPL.codex-review.md:549:        let s = AcquisitionScheme::from_thermo_raw(&template).expect("extract");
./ACQUISITION_SCHEME_IMPL.codex-review.md:565:            "from_thermo_raw OK: instrument={:?}, {} MS2 windows, mz {:.1}..{:.1}, cycle {:.4}s",
./ACQUISITION_SCHEME_IMPL.codex-review.md:576:   `from_thermo_raw()` returns `cycle_time_s = 0.0` when no second MS1 exists, while `validate()` rejects it. Extraction should return `InvalidData`, or derive cycle time using a documented method. Returning an object guaranteed to fail validation is misleading.
./ACQUISITION_SCHEME_IMPL.codex-review.md:622:   `from_thermo_raw()` returns `cycle_time_s = 0.0` when no second MS1 exists, while `validate()` rejects it. Extraction should return `InvalidData`, or derive cycle time using a documented method. Returning an object guaranteed to fail validation is misleading.
./ACQUISITION_SCHEME_PLAN.md:141:    pub fn from_bruker_d(path: &str) -> io::Result<Self>;      // DiaFrameMsMsWindows (+TimsMobility geometry)
./ACQUISITION_SCHEME_PLAN.md:143:    pub fn from_thermo_raw(path: &str) -> io::Result<Self>;    // walk one MS2 cycle via scan_event()
./ACQUISITION_SCHEME_PLAN.md:145:    pub fn from_sciex_wiff(path: &str) -> io::Result<Self>;    // SWATHMethod; CE => Unknown (rolling)
./ACQUISITION_SCHEME_PLAN.md:146:    pub fn from_window_table(rows, instrument) -> Self;        // injected / CSV
./ACQUISITION_SCHEME_PLAN.md:150:Each sets `provenance.source` accordingly. `from_thermo_raw` reads the first
./ACQUISITION_SCHEME_PLAN.md:157:  reference round trip: `from_thermo_raw(template)` → generate → `ThermoRawWriter`
./ACQUISITION_SCHEME_PLAN.md:197:  location works (uniform stride) so `from_thermo_raw` is unblocked.
./ACQUISITION_SCHEME_PLAN.md:212:- `from_thermo_raw(astral_template)`: N MS2 windows, monotonic centers, CE policy
./ACQUISITION_SCHEME_PLAN.md:214:- `from_sciex_wiff(zenotof)`: ~60 windows covering 399.5–899.9, CE `Unknown`.
./ACQUISITION_SCHEME_PLAN.md:215:- `from_bruker_d(ref)` → `to_bruker_tables()` golden-equal to today's Python path.
./ACQUISITION_SCHEME_PLAN.md:216:- End-to-end: `from_thermo_raw(template)` → generate 1×MS1 + N×MS2 → `ThermoRawWriter`
./rustdf/src/sim/scheme.rs:120:pub struct DiaMs2Frame {
./rustdf/src/sim/scheme.rs:130:    pub vendor_group_id: Option<u32>,
./rustdf/src/sim/scheme.rs:220:    pub fn from_window_table(
./rustdf/src/sim/scheme.rs:229:            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./rustdf/src/sim/scheme.rs:234:                vendor_group_id: None,
./rustdf/src/sim/scheme.rs:436:    /// preserved [`DiaMs2Frame::vendor_group_id`] when present (so a Bruker
./rustdf/src/sim/scheme.rs:445:    pub fn to_bruker_windows(&self) -> Result<Vec<crate::data::meta::DiaMsMsWindow>, String> {
./rustdf/src/sim/scheme.rs:448:            return Err("to_bruker_windows requires a timsTOF (TimsTofDia) scheme".into());
./rustdf/src/sim/scheme.rs:455:                let group = frame.vendor_group_id.unwrap_or(seq);
./rustdf/src/sim/scheme.rs:502:    pub fn from_bruker_d<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
./rustdf/src/sim/scheme.rs:547:            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./rustdf/src/sim/scheme.rs:552:                vendor_group_id: Some(group), // preserve Bruker WindowGroup identity
./rustdf/src/sim/scheme.rs:625:    pub fn from_sciex_wiff<P: AsRef<std::path::Path>>(
./rustdf/src/sim/scheme.rs:654:            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./rustdf/src/sim/scheme.rs:663:                vendor_group_id: None,
./rustdf/src/sim/scheme.rs:696:    pub fn from_thermo_raw<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
./rustdf/src/sim/scheme.rs:767:                cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./rustdf/src/sim/scheme.rs:776:                    vendor_group_id: None,
./rustdf/src/sim/scheme.rs:856:    fn from_window_table_validates() {
./rustdf/src/sim/scheme.rs:857:        let s = AcquisitionScheme::from_window_table(
./rustdf/src/sim/scheme.rs:881:        let frame = DiaMs2Frame {
./rustdf/src/sim/scheme.rs:886:            vendor_group_id: None,
./rustdf/src/sim/scheme.rs:936:            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./rustdf/src/sim/scheme.rs:941:                vendor_group_id: None,
./rustdf/src/sim/scheme.rs:973:        let bad_dur = AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
./rustdf/src/sim/scheme.rs:978:            vendor_group_id: None,
./rustdf/src/sim/scheme.rs:985:    fn from_bruker_d_extracts_cycle() {
./rustdf/src/sim/scheme.rs:989:                eprintln!("SKIP from_bruker_d_extracts_cycle: set TIMSIM_BRUKER_DIA_D=<dia .d>");
./rustdf/src/sim/scheme.rs:993:        let s = AcquisitionScheme::from_bruker_d(&d).expect("extract");
./rustdf/src/sim/scheme.rs:1008:            "from_bruker_d OK: TimsTofDia, {} frames, {} windows ({}), mz {:.1}..{:.1}",
./rustdf/src/sim/scheme.rs:1026:        let scheme = AcquisitionScheme::from_bruker_d(&d).expect("extract");
./rustdf/src/sim/scheme.rs:1027:        let regenerated = scheme.to_bruker_windows().expect("to_bruker_windows");
./rustdf/src/sim/scheme.rs:1032:        // WindowGroup id (preserved via vendor_group_id) — no normalization, so a
./rustdf/src/sim/scheme.rs:1071:    fn from_sciex_wiff_extracts_windows() {
./rustdf/src/sim/scheme.rs:1075:                eprintln!("SKIP from_sciex_wiff_extracts_windows: set TIMSIM_SCIEX_WIFF=<.wiff>");
./rustdf/src/sim/scheme.rs:1080:        let s = AcquisitionScheme::from_sciex_wiff(&wiff, 3.5, 1800.0, CollisionEnergyPolicy::Unknown)
./rustdf/src/sim/scheme.rs:1094:        let s2 = AcquisitionScheme::from_sciex_wiff(&wiff, 3.5, 1800.0, rolling).expect("extract");
./rustdf/src/sim/scheme.rs:1101:            "from_sciex_wiff OK: SciexZenoTof, {} SWATH windows, mz {:.1}..{:.1}",
./rustdf/src/sim/scheme.rs:1108:    fn from_thermo_raw_extracts_cycle() {
./rustdf/src/sim/scheme.rs:1112:                eprintln!("SKIP from_thermo_raw_extracts_cycle: set TIMSIM_ASTRAL_TEMPLATE");
./rustdf/src/sim/scheme.rs:1116:        let s = AcquisitionScheme::from_thermo_raw(&template).expect("extract");
./rustdf/src/sim/scheme.rs:1132:            "from_thermo_raw OK: instrument={:?}, {} MS2 windows, mz {:.1}..{:.1}, cycle {:.4}s",

 succeeded in 215ms:
?? TO_BRUKER_V2.codex-review.md
?? packages/imspy-predictors/scripts/compare_libraries.py
?? packages/imspy-predictors/scripts/generate_calibrated_library.py
?? primitives/
?? rustdf/examples/build_bruker_pseudo_ms2.rs
?? rustdf/examples/build_bruker_pseudo_ms2_v3.rs
?? rustdf/examples/dump_bruker_ms1_centroids.rs
?? scripts/compare_diann_to_truth.py
?? scripts/reread_rustw_smoke.py

 exited 2 in 238ms:
codex
The field is present at all seven live construction sites. The main remaining concern is mixed preserved/unassigned frames: the fallback is based on frame position, not on the set of reserved IDs, so collisions are possible and currently unvalidated. I’m checking commit history and test coverage to rank that accurately.
exec
/bin/bash -lc "sed -n '110,145p;210,245p;425,570p;1015,1065p' rustdf/src/sim/scheme.rs" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
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
    /// The vendor's native group id for this frame, preserved for round-trip
    /// fidelity (Bruker `WindowGroup`). `None` for vendors without one. The cycle
    /// order is authoritative for *simultaneity/ordering*; this is *identity* only
    /// — needed so a Bruker adapter regenerates the original `WindowGroup` values
    /// that companion tables (`DiaFrameMsMsInfo`) reference.
    pub vendor_group_id: Option<u32>,
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
                vendor_group_id: None,
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
        }
        Ok(())
    }
}

impl AcquisitionScheme {
    /// Render the scheme's MS2 windows as Bruker `DiaFrameMsMsWindows` rows — the
    /// backward-compatibility adapter for the existing timsTOF write path.
    ///
    /// Each MS2 frame is one window group; its windows must carry `TimsMobility`
    /// geometry (the Bruker-grid scan ranges). The `WindowGroup` id is the frame's
    /// preserved [`DiaMs2Frame::vendor_group_id`] when present (so a Bruker
    /// round-trip keeps the ids that `DiaFrameMsMsInfo` references), else a 1..N
    /// fallback in cycle order. A `Linear` CE policy is resolved at the window
    /// center (a lossy materialization, not an inverse); `Unknown` CE and
    /// non-timsTOF schemes are rejected. The scheme is validated first.
    ///
    /// Row order is not meaningful (the SQLite table has none); the companion
    /// frame→group table `DiaFrameMsMsInfo` needs the run frame schedule and is a
    /// separate step.
    pub fn to_bruker_windows(&self) -> Result<Vec<crate::data::meta::DiaMsMsWindow>, String> {
        self.validate()?;
        if self.instrument != InstrumentKind::TimsTofDia {
            return Err("to_bruker_windows requires a timsTOF (TimsTofDia) scheme".into());
        }
        let mut rows = Vec::new();
        let mut seq: u32 = 0;
        for ev in &self.cycle {
            if let AcquisitionEvent::DiaMs2Frame(frame) = ev {
                seq += 1;
                let group = frame.vendor_group_id.unwrap_or(seq);
                for w in &frame.windows {
                    let (scan_num_begin, scan_num_end) = match w.geometry {
                        DiaGeometry::TimsMobility {
                            scan_start,
                            scan_end,
                        } => (scan_start, scan_end),
                        DiaGeometry::MzOnly => {
                            return Err("timsTOF window lacks mobility geometry".into())
                        }
                    };
                    let collision_energy = match w.collision_energy {
                        CollisionEnergyPolicy::Value(v) => v,
                        CollisionEnergyPolicy::Linear { .. } => w
                            .collision_energy
                            .at(w.isolation.center_mz)
                            .ok_or("could not resolve linear CE")?,
                        CollisionEnergyPolicy::Unknown => {
                            return Err("window has unknown collision energy".into())
                        }
                    };
                    rows.push(crate::data::meta::DiaMsMsWindow {
                        window_group: group,
                        scan_num_begin,
                        scan_num_end,
                        isolation_mz: w.isolation.center_mz,
                        isolation_width: w.isolation.width_mz,
                        collision_energy,
                    });
                }
            }
        }
        Ok(rows)
    }

    /// Extract the acquisition scheme from a real Bruker timsTOF DIA `.d`.
    ///
    /// Reads `DiaFrameMsMsWindows` and the frame table. Each **window group**
    /// becomes one [`DiaMs2Frame`] holding its mobility-partitioned windows
    /// (`TimsMobility` geometry — the scan ranges are Bruker-grid coordinates),
    /// preceded by a single MS1 event. Cycle timing comes from the spacing of
    /// the precursor (MS1) frames. The returned scheme is validated.
    ///
    /// Note: window groups are ordered by ascending `WindowGroup` id, which is
    /// the DIA-PASEF acquisition order in practice; a file that reuses or
    /// permutes group numbering would need ordering by first MS2-frame
    /// occurrence (via `DiaFrameMsMsInfo`) instead.
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
        for (group, ws) in by_group {
            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: ws,
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: Some(group), // preserve Bruker WindowGroup identity
            }));
        }

        // Cycle timing from the precursor (MS1, MsMsType == 0) frame spacing.
        // Use the MEDIAN of the positive gaps between distinct, finite precursor
        // times, so one anomalous interval doesn't set the whole cycle.
        let mut prec_times: Vec<f64> = frames
            .iter()
            .filter(|f| f.ms_ms_type == 0 && f.time.is_finite())
            .map(|f| f.time)
            .collect();
        prec_times.sort_by(f64::total_cmp);
        prec_times.dedup();
        let mut gaps: Vec<f64> = prec_times
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|g| *g > 0.0)
            .collect();
    // Gated: round-trip a real Bruker DIA .d through the scheme and back to the
    // DiaFrameMsMsWindows rows; the regenerated table must match the source.
    #[test]
    fn bruker_windows_round_trip() {
        let d = match std::env::var("TIMSIM_BRUKER_DIA_D") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP bruker_windows_round_trip: set TIMSIM_BRUKER_DIA_D");
                return;
            }
        };
        let scheme = AcquisitionScheme::from_bruker_d(&d).expect("extract");
        let regenerated = scheme.to_bruker_windows().expect("to_bruker_windows");
        let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
        assert_eq!(regenerated.len(), original.len(), "row count differs");

        // Compare as a sorted multiset of EXACT field tuples, including the
        // WindowGroup id (preserved via vendor_group_id) — no normalization, so a
        // real id mismatch would fail. Row order in the table is not guaranteed.
        fn tuples(
            rows: &[crate::data::meta::DiaMsMsWindow],
        ) -> Vec<(u32, u32, u32, u64, u64, u64)> {
            let mut out: Vec<_> = rows
                .iter()
                .map(|r| {
                    (
                        r.window_group,
                        r.scan_num_begin,
                        r.scan_num_end,
                        r.isolation_mz.to_bits(),
                        r.isolation_width.to_bits(),
                        r.collision_energy.to_bits(),
                    )
                })
                .collect();
            out.sort_unstable();
            out
        }
        assert_eq!(
            tuples(&regenerated),
            tuples(&original),
            "round-trip windows differ from source (incl. WindowGroup id)"
        );
        eprintln!(
            "bruker_windows_round_trip OK: {} rows match across {} groups",
            original.len(),
            scheme
                .cycle
                .iter()
                .filter(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_)))
                .count()

exec
/bin/bash -lc 'git log -5 --oneline -- rustdf/src/sim/scheme.rs && git blame -L 430,475 rustdf/src/sim/scheme.rs' in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
d3429367 harden: codex review of to_bruker_windows — preserve WindowGroup identity
4d7fdb8e feat(rustdf): AcquisitionScheme::to_bruker_windows (Bruker adapter) + round-trip golden test
5fde3d5f harden: codex review of extractors + caller-supplied SCIEX CE
75a851ca feat(rustdf): AcquisitionScheme::from_sciex_wiff extractor (+ sciex feature)
20503b56 feat(rustdf): AcquisitionScheme::from_bruker_d extractor
20503b562 (theGreatHerrLebert 2026-06-09 07:35:10 +0200 430) impl AcquisitionScheme {
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 431)     /// Render the scheme's MS2 windows as Bruker `DiaFrameMsMsWindows` rows — the
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 432)     /// backward-compatibility adapter for the existing timsTOF write path.
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 433)     ///
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 434)     /// Each MS2 frame is one window group; its windows must carry `TimsMobility`
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 435)     /// geometry (the Bruker-grid scan ranges). The `WindowGroup` id is the frame's
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 436)     /// preserved [`DiaMs2Frame::vendor_group_id`] when present (so a Bruker
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 437)     /// round-trip keeps the ids that `DiaFrameMsMsInfo` references), else a 1..N
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 438)     /// fallback in cycle order. A `Linear` CE policy is resolved at the window
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 439)     /// center (a lossy materialization, not an inverse); `Unknown` CE and
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 440)     /// non-timsTOF schemes are rejected. The scheme is validated first.
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 441)     ///
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 442)     /// Row order is not meaningful (the SQLite table has none); the companion
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 443)     /// frame→group table `DiaFrameMsMsInfo` needs the run frame schedule and is a
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 444)     /// separate step.
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 445)     pub fn to_bruker_windows(&self) -> Result<Vec<crate::data::meta::DiaMsMsWindow>, String> {
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 446)         self.validate()?;
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 447)         if self.instrument != InstrumentKind::TimsTofDia {
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 448)             return Err("to_bruker_windows requires a timsTOF (TimsTofDia) scheme".into());
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 449)         }
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 450)         let mut rows = Vec::new();
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 451)         let mut seq: u32 = 0;
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 452)         for ev in &self.cycle {
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 453)             if let AcquisitionEvent::DiaMs2Frame(frame) = ev {
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 454)                 seq += 1;
d34293676 (theGreatHerrLebert 2026-06-09 09:01:50 +0200 455)                 let group = frame.vendor_group_id.unwrap_or(seq);
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 456)                 for w in &frame.windows {
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 457)                     let (scan_num_begin, scan_num_end) = match w.geometry {
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 458)                         DiaGeometry::TimsMobility {
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 459)                             scan_start,
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 460)                             scan_end,
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 461)                         } => (scan_start, scan_end),
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 462)                         DiaGeometry::MzOnly => {
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 463)                             return Err("timsTOF window lacks mobility geometry".into())
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 464)                         }
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 465)                     };
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 466)                     let collision_energy = match w.collision_energy {
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 467)                         CollisionEnergyPolicy::Value(v) => v,
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 468)                         CollisionEnergyPolicy::Linear { .. } => w
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 469)                             .collision_energy
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 470)                             .at(w.isolation.center_mz)
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 471)                             .ok_or("could not resolve linear CE")?,
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 472)                         CollisionEnergyPolicy::Unknown => {
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 473)                             return Err("window has unknown collision energy".into())
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 474)                         }
4d7fdb8eb (theGreatHerrLebert 2026-06-09 08:53:39 +0200 475)                     };

exec
/bin/bash -lc "ls rustdf && sed -n '1,180p' rustdf/Cargo.toml && rg -n \"scheme|sim\" rustdf/src/lib.rs rustdf/src/sim/mod.rs Cargo.toml" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
Cargo.toml
README.md
examples
src
[package]
name = "rustdf"
version = "0.4.1"
edition = "2021"
authors = ["David Teschner <davidteschner@googlemail.com>"] # Add your name and email
description = "A Rust library for interacting with Bruker TDF formatted Raw Data."
license = "MIT" 
repository = "https://github.com/theGreatHerrLebert/rustims" 
documentation = "https://docs.rs/rustdf" 
readme = "README.md" 
keywords = ["dataframe", "sql", "compression", "parallel"]
categories = ["data-structures", "science"]
rust-version = "1.84"

[lib]
path = "src/lib.rs"

[dependencies]
# Command-line argument parsing
clap = { version = "4.5.32", features = ["derive"] }
# Dynamic library loading
libloading = "0.8.6"
# SQLite with bundled binaries
rusqlite = { version = "0.32.1", features = ["bundled"] }
# Compression libraries
lzf = "1.0.0"
zstd = "0.13.3"
# Byte order utilities
byteorder = "1.5.0"
# Core library for computational proteomics
mscore = { path = "../mscore", version = "0.4.1" }
# Parallelism
rayon = "1.10.0"
# Serialization
serde = { version = "1.0.219", features = ["derive"] }
serde_json = "1.0.140"
# Random number generation
rand = "0.8.5"
rustc-hash = "2.1.0"
bincode = "1.3.3"
polars = { version = "0.43.1", features = ["ndarray", "parquet", "dtype-u8"] }
# Thermo .raw writer (private repo; only built with the `thermo` feature)
thermorawfile = { git = "https://github.com/theGreatHerrLebert/thermorawfile", rev = "9f4c0affe60a639759883fb569d3c9f5f7601856", optional = true }
# SCIEX .wiff method reader (private repo; only built with the `sciex` feature)
sciexwiff = { git = "https://github.com/theGreatHerrLebert/sciexwiff", rev = "ef7c0cd7e59d3b51160e446cbe3af240d230668d", optional = true }

[features]
# Off by default: enable the Thermo .raw / SCIEX .wiff extractors + writers.
# Each requires build-time access to its private repo.
thermo = ["dep:thermorawfile"]
sciex = ["dep:sciexwiff"]

[package.metadata.docs.rs]
features = ["all"]
rustdf/src/sim/mod.rs:8:pub mod scheme;
rustdf/src/lib.rs:2:pub mod sim;

exec
/bin/bash -lc 'git show HEAD''^:rustdf/src/sim/scheme.rs | sed -n '"'110,140p;425,470p;530,560p;1000,1055p'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
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
    /// backward-compatibility adapter for the existing timsTOF write path.
    ///
    /// Each MS2 frame is one window group; its windows must carry `TimsMobility`
    /// geometry (the Bruker-grid scan ranges). Window groups are numbered 1..N in
    /// cycle order (the DIA-PASEF convention); a `Linear` CE policy is resolved at
    /// the window center, and `Unknown` CE is rejected. Errors if the scheme is
    /// not a timsTOF layout. (The companion frame→group table,
    /// `DiaFrameMsMsInfo`, depends on the full run's frame schedule and is a
    /// separate step.)
    pub fn to_bruker_windows(&self) -> Result<Vec<crate::data::meta::DiaMsMsWindow>, String> {
        if self.instrument != InstrumentKind::TimsTofDia {
            return Err("to_bruker_windows requires a timsTOF (TimsTofDia) scheme".into());
        }
        let mut rows = Vec::new();
        let mut group: u32 = 0;
        for ev in &self.cycle {
            if let AcquisitionEvent::DiaMs2Frame(frame) = ev {
                group += 1;
                for w in &frame.windows {
                    let (scan_num_begin, scan_num_end) = match w.geometry {
                        DiaGeometry::TimsMobility {
                            scan_start,
                            scan_end,
                        } => (scan_start, scan_end),
                        DiaGeometry::MzOnly => {
                            return Err("timsTOF window lacks mobility geometry".into())
                        }
                    };
                    let collision_energy = match w.collision_energy {
                        CollisionEnergyPolicy::Value(v) => v,
                        CollisionEnergyPolicy::Linear { .. } => w
                            .collision_energy
                            .at(w.isolation.center_mz)
                            .ok_or("could not resolve linear CE")?,
                        CollisionEnergyPolicy::Unknown => {
                            return Err("window has unknown collision energy".into())
                        }
                    };
                    rows.push(crate::data::meta::DiaMsMsWindow {
                        window_group: group,
                        scan_num_begin,
                        scan_num_end,
                        isolation_mz: w.isolation.center_mz,
                        isolation_width: w.isolation.width_mz,
                        collision_energy,
                    });
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
        // Use the MEDIAN of the positive gaps between distinct, finite precursor
        // times, so one anomalous interval doesn't set the whole cycle.
        let mut prec_times: Vec<f64> = frames
            .iter()
            .filter(|f| f.ms_ms_type == 0 && f.time.is_finite())
            .map(|f| f.time)
            .collect();
        prec_times.sort_by(f64::total_cmp);
        prec_times.dedup();
        let mut gaps: Vec<f64> = prec_times
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|g| *g > 0.0)
            .collect();
        if gaps.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "could not determine cycle time (need >= 2 distinct precursor frames)",
        let d = match std::env::var("TIMSIM_BRUKER_DIA_D") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP bruker_windows_round_trip: set TIMSIM_BRUKER_DIA_D");
                return;
            }
        };
        let scheme = AcquisitionScheme::from_bruker_d(&d).expect("extract");
        let regenerated = scheme.to_bruker_windows().expect("to_bruker_windows");
        let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
        assert_eq!(regenerated.len(), original.len(), "row count differs");

        // Normalize window-group ids to canonical 1..N (ascending) on both sides,
        // then compare as a sorted multiset of exact-bit field tuples.
        fn canonical(
            rows: &[crate::data::meta::DiaMsMsWindow],
        ) -> Vec<(u32, u32, u32, u64, u64, u64)> {
            let mut groups: Vec<u32> = rows.iter().map(|r| r.window_group).collect();
            groups.sort_unstable();
            groups.dedup();
            let mut out: Vec<_> = rows
                .iter()
                .map(|r| {
                    let g = groups.iter().position(|&x| x == r.window_group).unwrap() as u32 + 1;
                    (
                        g,
                        r.scan_num_begin,
                        r.scan_num_end,
                        r.isolation_mz.to_bits(),
                        r.isolation_width.to_bits(),
                        r.collision_energy.to_bits(),
                    )
                })
                .collect();
            out.sort_unstable();
            out
        }
        assert_eq!(
            canonical(&regenerated),
            canonical(&original),
            "round-trip windows differ from source"
        );
        eprintln!(
            "bruker_windows_round_trip OK: {} rows match across {} groups",
            original.len(),
            scheme
                .cycle
                .iter()
                .filter(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_)))
                .count()
        );
    }

    #[cfg(feature = "sciex")]
    #[test]
    fn from_sciex_wiff_extracts_windows() {

exec
/bin/bash -lc 'git show --stat --oneline d3429367 && git show --format=fuller --find-renames d3429367 -- rustdf/src/sim/scheme.rs' in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
d3429367 harden: codex review of to_bruker_windows — preserve WindowGroup identity
 TO_BRUKER.codex-review.md | 1197 +++++++++++++++++++++++++++++++++++++++++++++
 rustdf/src/sim/scheme.rs  |   56 ++-
 2 files changed, 1233 insertions(+), 20 deletions(-)
commit d342936765b23a278165b82efdac39b30f61540e
Author:     theGreatHerrLebert <davidteschner@googlemail.com>
AuthorDate: Tue Jun 9 09:01:50 2026 +0200
Commit:     theGreatHerrLebert <davidteschner@googlemail.com>
CommitDate: Tue Jun 9 09:01:50 2026 +0200

    harden: codex review of to_bruker_windows — preserve WindowGroup identity
    
    Codex #1/#2: renumbering window groups 1..N lost the source WindowGroup ids, so a
    Bruker round-trip could desync from DiaFrameMsMsInfo (which references the original
    ids) — and the round-trip test masked it via canonical normalization.
    
    - DiaMs2Frame gains vendor_group_id: Option<u32> (identity only; cycle order still
      governs ordering/simultaneity). from_bruker_d sets it to the source WindowGroup;
      other vendors None.
    - to_bruker_windows uses vendor_group_id when present (else 1..N fallback) and now
      calls validate() first (#4).
    - round-trip golden test compares EXACT WindowGroup ids (no normalization).
    
    Verified on the real DIA-PASEF .d: 36 rows / 15 groups round-trip bit-exact
    including WindowGroup id. Includes the codex review.

diff --git a/rustdf/src/sim/scheme.rs b/rustdf/src/sim/scheme.rs
index 65337c24..a5a0c106 100644
--- a/rustdf/src/sim/scheme.rs
+++ b/rustdf/src/sim/scheme.rs
@@ -122,6 +122,12 @@ pub struct DiaMs2Frame {
     pub analyzer: Analyzer,
     pub data_mode: DataMode,
     pub duration_s: Option<f64>,
+    /// The vendor's native group id for this frame, preserved for round-trip
+    /// fidelity (Bruker `WindowGroup`). `None` for vendors without one. The cycle
+    /// order is authoritative for *simultaneity/ordering*; this is *identity* only
+    /// — needed so a Bruker adapter regenerates the original `WindowGroup` values
+    /// that companion tables (`DiaFrameMsMsInfo`) reference.
+    pub vendor_group_id: Option<u32>,
 }
 
 #[derive(Clone, Debug)]
@@ -225,6 +231,7 @@ impl AcquisitionScheme {
                 analyzer: ms1.analyzer,
                 data_mode: DataMode::Centroid,
                 duration_s: None,
+                vendor_group_id: None,
             }));
         }
         AcquisitionScheme {
@@ -425,21 +432,27 @@ impl AcquisitionScheme {
     /// backward-compatibility adapter for the existing timsTOF write path.
     ///
     /// Each MS2 frame is one window group; its windows must carry `TimsMobility`
-    /// geometry (the Bruker-grid scan ranges). Window groups are numbered 1..N in
-    /// cycle order (the DIA-PASEF convention); a `Linear` CE policy is resolved at
-    /// the window center, and `Unknown` CE is rejected. Errors if the scheme is
-    /// not a timsTOF layout. (The companion frame→group table,
-    /// `DiaFrameMsMsInfo`, depends on the full run's frame schedule and is a
-    /// separate step.)
+    /// geometry (the Bruker-grid scan ranges). The `WindowGroup` id is the frame's
+    /// preserved [`DiaMs2Frame::vendor_group_id`] when present (so a Bruker
+    /// round-trip keeps the ids that `DiaFrameMsMsInfo` references), else a 1..N
+    /// fallback in cycle order. A `Linear` CE policy is resolved at the window
+    /// center (a lossy materialization, not an inverse); `Unknown` CE and
+    /// non-timsTOF schemes are rejected. The scheme is validated first.
+    ///
+    /// Row order is not meaningful (the SQLite table has none); the companion
+    /// frame→group table `DiaFrameMsMsInfo` needs the run frame schedule and is a
+    /// separate step.
     pub fn to_bruker_windows(&self) -> Result<Vec<crate::data::meta::DiaMsMsWindow>, String> {
+        self.validate()?;
         if self.instrument != InstrumentKind::TimsTofDia {
             return Err("to_bruker_windows requires a timsTOF (TimsTofDia) scheme".into());
         }
         let mut rows = Vec::new();
-        let mut group: u32 = 0;
+        let mut seq: u32 = 0;
         for ev in &self.cycle {
             if let AcquisitionEvent::DiaMs2Frame(frame) = ev {
-                group += 1;
+                seq += 1;
+                let group = frame.vendor_group_id.unwrap_or(seq);
                 for w in &frame.windows {
                     let (scan_num_begin, scan_num_end) = match w.geometry {
                         DiaGeometry::TimsMobility {
@@ -530,12 +543,13 @@ impl AcquisitionScheme {
             duration_s: None,
         })];
         let n_groups = by_group.len();
-        for (_group, ws) in by_group {
+        for (group, ws) in by_group {
             cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                 windows: ws,
                 analyzer: Analyzer::Tof,
                 data_mode: DataMode::Centroid,
                 duration_s: None,
+                vendor_group_id: Some(group), // preserve Bruker WindowGroup identity
             }));
         }
 
@@ -646,6 +660,7 @@ impl AcquisitionScheme {
                 analyzer: Analyzer::Tof,
                 data_mode: DataMode::Centroid,
                 duration_s: None,
+                vendor_group_id: None,
             }));
         }
         let scheme = AcquisitionScheme {
@@ -758,6 +773,7 @@ impl AcquisitionScheme {
                     analyzer: analyzer_of(ev.analyzer),
                     data_mode: data_mode_of(scan),
                     duration_s: None,
+                    vendor_group_id: None,
                 }));
             }
         }
@@ -867,6 +883,7 @@ mod tests {
             analyzer: Analyzer::Tof,
             data_mode: DataMode::Centroid,
             duration_s: None,
+            vendor_group_id: None,
         };
         let s = AcquisitionScheme {
             version: SCHEME_VERSION,
@@ -921,6 +938,7 @@ mod tests {
                 analyzer: Analyzer::Astms,
                 data_mode: DataMode::Centroid,
                 duration_s: None,
+                vendor_group_id: None,
             })
         };
         let mk = |cycle: Vec<AcquisitionEvent>| AcquisitionScheme {
@@ -957,6 +975,7 @@ mod tests {
             analyzer: Analyzer::Astms,
             data_mode: DataMode::Centroid,
             duration_s: Some(-1.0),
+            vendor_group_id: None,
         });
         assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), bad_dur]).validate().is_err());
     }
@@ -1009,20 +1028,17 @@ mod tests {
         let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
         assert_eq!(regenerated.len(), original.len(), "row count differs");
 
-        // Normalize window-group ids to canonical 1..N (ascending) on both sides,
-        // then compare as a sorted multiset of exact-bit field tuples.
-        fn canonical(
+        // Compare as a sorted multiset of EXACT field tuples, including the
+        // WindowGroup id (preserved via vendor_group_id) — no normalization, so a
+        // real id mismatch would fail. Row order in the table is not guaranteed.
+        fn tuples(
             rows: &[crate::data::meta::DiaMsMsWindow],
         ) -> Vec<(u32, u32, u32, u64, u64, u64)> {
-            let mut groups: Vec<u32> = rows.iter().map(|r| r.window_group).collect();
-            groups.sort_unstable();
-            groups.dedup();
             let mut out: Vec<_> = rows
                 .iter()
                 .map(|r| {
-                    let g = groups.iter().position(|&x| x == r.window_group).unwrap() as u32 + 1;
                     (
-                        g,
+                        r.window_group,
                         r.scan_num_begin,
                         r.scan_num_end,
                         r.isolation_mz.to_bits(),
@@ -1035,9 +1051,9 @@ mod tests {
             out
         }
         assert_eq!(
-            canonical(&regenerated),
-            canonical(&original),
-            "round-trip windows differ from source"
+            tuples(&regenerated),
+            tuples(&original),
+            "round-trip windows differ from source (incl. WindowGroup id)"
         );
         eprintln!(
             "bruker_windows_round_trip OK: {} rows match across {} groups",

exec
/bin/bash -lc 'cargo test -p rustdf sim::scheme::tests --lib' in /scratch/timsim-demo/SUBMISSION/rustims
exec
/bin/bash -lc "nl -ba rustdf/src/sim/scheme.rs | sed -n '118,134p;220,238p;300,430p;445,462p;540,556p;648,667p;760,780p;875,982p;1020,1062p'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
   118	/// mobility-partitioned windows (sharing the frame) for timsTOF.
   119	#[derive(Clone, Debug)]
   120	pub struct DiaMs2Frame {
   121	    pub windows: Vec<DiaWindow>,
   122	    pub analyzer: Analyzer,
   123	    pub data_mode: DataMode,
   124	    pub duration_s: Option<f64>,
   125	    /// The vendor's native group id for this frame, preserved for round-trip
   126	    /// fidelity (Bruker `WindowGroup`). `None` for vendors without one. The cycle
   127	    /// order is authoritative for *simultaneity/ordering*; this is *identity* only
   128	    /// — needed so a Bruker adapter regenerates the original `WindowGroup` values
   129	    /// that companion tables (`DiaFrameMsMsInfo`) reference.
   130	    pub vendor_group_id: Option<u32>,
   131	}
   132	
   133	#[derive(Clone, Debug)]
   134	pub enum AcquisitionEvent {
   220	    pub fn from_window_table(
   221	        instrument: InstrumentKind,
   222	        ms1: Ms1Event,
   223	        windows: Vec<DiaWindow>,
   224	        repeat: RepeatPolicy,
   225	        mz_range: (f64, f64),
   226	    ) -> Self {
   227	        let mut cycle = vec![AcquisitionEvent::Ms1(ms1.clone())];
   228	        for w in windows {
   229	            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
   230	                windows: vec![w],
   231	                analyzer: ms1.analyzer,
   232	                data_mode: DataMode::Centroid,
   233	                duration_s: None,
   234	                vendor_group_id: None,
   235	            }));
   236	        }
   237	        AcquisitionScheme {
   238	            version: SCHEME_VERSION,
   300	                        }
   301	                    }
   302	                }
   303	                AcquisitionEvent::DiaMs2Frame(frame) => {
   304	                    check_dur(frame.duration_s, "MS2 frame")?;
   305	                    if frame.windows.is_empty() {
   306	                        return Err("MS2 frame has no windows".into());
   307	                    }
   308	                    let multi = frame.windows.len() > 1;
   309	                    if multi && self.instrument != InstrumentKind::TimsTofDia {
   310	                        return Err(
   311	                            "multi-window MS2 frame is only valid for timsTOF (mobility-partitioned)"
   312	                                .into(),
   313	                        );
   314	                    }
   315	                    for (i, w) in frame.windows.iter().enumerate() {
   316	                        if !(w.isolation.width_mz.is_finite() && w.isolation.width_mz > 0.0) {
   317	                            return Err("window width must be finite and > 0".into());
   318	                        }
   319	                        if !w.isolation.center_mz.is_finite()
   320	                            || !w.isolation.lower().is_finite()
   321	                            || !w.isolation.upper().is_finite()
   322	                        {
   323	                            return Err("window m/z is not finite".into());
   324	                        }
   325	                        if w.isolation.lower() < lo - 1e-6 || w.isolation.upper() > hi + 1e-6 {
   326	                            return Err(format!(
   327	                                "window [{:.4}, {:.4}] outside mz_range [{lo:.4}, {hi:.4}]",
   328	                                w.isolation.lower(),
   329	                                w.isolation.upper()
   330	                            ));
   331	                        }
   332	                        match w.collision_energy {
   333	                            CollisionEnergyPolicy::Value(v) => {
   334	                                if !v.is_finite() || v < 0.0 {
   335	                                    return Err("collision energy must be finite and >= 0".into());
   336	                                }
   337	                            }
   338	                            CollisionEnergyPolicy::Linear {
   339	                                intercept,
   340	                                slope_per_mz,
   341	                            } => {
   342	                                if !intercept.is_finite() || !slope_per_mz.is_finite() {
   343	                                    return Err("non-finite linear CE coefficients".into());
   344	                                }
   345	                                match w.collision_energy.at(w.isolation.center_mz) {
   346	                                    Some(ce) if ce.is_finite() && ce >= 0.0 => {}
   347	                                    _ => {
   348	                                        return Err(
   349	                                            "linear CE resolves to non-finite or negative".into()
   350	                                        )
   351	                                    }
   352	                                }
   353	                            }
   354	                            CollisionEnergyPolicy::Unknown => {}
   355	                        }
   356	                        match w.geometry {
   357	                            DiaGeometry::TimsMobility {
   358	                                scan_start,
   359	                                scan_end,
   360	                            } => {
   361	                                if self.instrument != InstrumentKind::TimsTofDia {
   362	                                    return Err("mobility geometry is only valid for timsTOF".into());
   363	                                }
   364	                                if scan_start > scan_end {
   365	                                    return Err("mobility scan_start > scan_end".into());
   366	                                }
   367	                            }
   368	                            DiaGeometry::MzOnly => {
   369	                                if multi {
   370	                                    return Err(
   371	                                        "multi-window timsTOF frame requires mobility geometry on every window"
   372	                                            .into(),
   373	                                    );
   374	                                }
   375	                            }
   376	                        }
   377	                        // Within one frame, windows must not overlap in BOTH m/z
   378	                        // and mobility (overlap across sequential frames is fine).
   379	                        for w2 in &frame.windows[..i] {
   380	                            let mz_overlap = w.isolation.lower() < w2.isolation.upper()
   381	                                && w2.isolation.lower() < w.isolation.upper();
   382	                            let im_overlap = match (w.geometry, w2.geometry) {
   383	                                (
   384	                                    DiaGeometry::TimsMobility {
   385	                                        scan_start: a0,
   386	                                        scan_end: a1,
   387	                                    },
   388	                                    DiaGeometry::TimsMobility {
   389	                                        scan_start: b0,
   390	                                        scan_end: b1,
   391	                                    },
   392	                                ) => a0 <= b1 && b0 <= a1,
   393	                                _ => true, // m/z-only windows share all mobility
   394	                            };
   395	                            if mz_overlap && im_overlap {
   396	                                return Err(
   397	                                    "windows within one frame overlap in both m/z and mobility"
   398	                                        .into(),
   399	                                );
   400	                            }
   401	                        }
   402	                    }
   403	                }
   404	            }
   405	        }
   406	        match self.repeat {
   407	            RepeatPolicy::FixedCycleTime {
   408	                cycle_time_s,
   409	                gradient_length_s,
   410	                start_time_s,
   411	            } => {
   412	                if !(cycle_time_s.is_finite() && cycle_time_s > 0.0) {
   413	                    return Err("cycle_time_s must be finite and > 0".into());
   414	                }
   415	                if !(gradient_length_s.is_finite() && gradient_length_s > 0.0) {
   416	                    return Err("gradient_length_s must be finite and > 0".into());
   417	                }
   418	                if !start_time_s.is_finite() || start_time_s < 0.0 {
   419	                    return Err("start_time_s must be finite and >= 0".into());
   420	                }
   421	                if start_time_s > gradient_length_s {
   422	                    return Err("start_time_s must be <= gradient_length_s".into());
   423	                }
   424	            }
   425	        }
   426	        Ok(())
   427	    }
   428	}
   429	
   430	impl AcquisitionScheme {
   445	    pub fn to_bruker_windows(&self) -> Result<Vec<crate::data::meta::DiaMsMsWindow>, String> {
   446	        self.validate()?;
   447	        if self.instrument != InstrumentKind::TimsTofDia {
   448	            return Err("to_bruker_windows requires a timsTOF (TimsTofDia) scheme".into());
   449	        }
   450	        let mut rows = Vec::new();
   451	        let mut seq: u32 = 0;
   452	        for ev in &self.cycle {
   453	            if let AcquisitionEvent::DiaMs2Frame(frame) = ev {
   454	                seq += 1;
   455	                let group = frame.vendor_group_id.unwrap_or(seq);
   456	                for w in &frame.windows {
   457	                    let (scan_num_begin, scan_num_end) = match w.geometry {
   458	                        DiaGeometry::TimsMobility {
   459	                            scan_start,
   460	                            scan_end,
   461	                        } => (scan_start, scan_end),
   462	                        DiaGeometry::MzOnly => {
   540	            analyzer: Analyzer::Tof,
   541	            data_mode: DataMode::Centroid,
   542	            mz_range: None,
   543	            duration_s: None,
   544	        })];
   545	        let n_groups = by_group.len();
   546	        for (group, ws) in by_group {
   547	            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
   548	                windows: ws,
   549	                analyzer: Analyzer::Tof,
   550	                data_mode: DataMode::Centroid,
   551	                duration_s: None,
   552	                vendor_group_id: Some(group), // preserve Bruker WindowGroup identity
   553	            }));
   554	        }
   555	
   556	        // Cycle timing from the precursor (MS1, MsMsType == 0) frame spacing.
   648	            let iso = IsolationWindow {
   649	                center_mz: w.center_mz(),
   650	                width_mz: w.width_mz(),
   651	            };
   652	            lo = lo.min(iso.lower());
   653	            hi = hi.max(iso.upper());
   654	            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
   655	                windows: vec![DiaWindow {
   656	                    isolation: iso,
   657	                    collision_energy,
   658	                    geometry: DiaGeometry::MzOnly,
   659	                }],
   660	                analyzer: Analyzer::Tof,
   661	                data_mode: DataMode::Centroid,
   662	                duration_s: None,
   663	                vendor_group_id: None,
   664	            }));
   665	        }
   666	        let scheme = AcquisitionScheme {
   667	            version: SCHEME_VERSION,
   760	            } else if seen_ms1 {
   761	                let iso = IsolationWindow {
   762	                    center_mz: ev.isolation_center,
   763	                    width_mz: ev.isolation_width,
   764	                };
   765	                win_lo = win_lo.min(iso.lower());
   766	                win_hi = win_hi.max(iso.upper());
   767	                cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
   768	                    windows: vec![DiaWindow {
   769	                        isolation: iso,
   770	                        collision_energy: CollisionEnergyPolicy::Value(ev.collision_energy),
   771	                        geometry: DiaGeometry::MzOnly,
   772	                    }],
   773	                    analyzer: analyzer_of(ev.analyzer),
   774	                    data_mode: data_mode_of(scan),
   775	                    duration_s: None,
   776	                    vendor_group_id: None,
   777	                }));
   778	            }
   779	        }
   780	
   875	        assert_eq!(s.ms1_count(), 1);
   876	        assert_eq!(s.num_cycles(), Some(600));
   877	    }
   878	
   879	    #[test]
   880	    fn multi_window_frame_only_tims() {
   881	        let frame = DiaMs2Frame {
   882	            windows: linear_windows(3),
   883	            analyzer: Analyzer::Tof,
   884	            data_mode: DataMode::Centroid,
   885	            duration_s: None,
   886	            vendor_group_id: None,
   887	        };
   888	        let s = AcquisitionScheme {
   889	            version: SCHEME_VERSION,
   890	            instrument: InstrumentKind::OrbitrapAstral, // wrong: multi-window on non-tims
   891	            cycle: vec![
   892	                AcquisitionEvent::Ms1(Ms1Event {
   893	                    analyzer: Analyzer::Ftms,
   894	                    data_mode: DataMode::Profile,
   895	                    mz_range: Some((390.0, 900.0)),
   896	                    duration_s: None,
   897	                }),
   898	                AcquisitionEvent::DiaMs2Frame(frame),
   899	            ],
   900	            repeat: RepeatPolicy::FixedCycleTime {
   901	                cycle_time_s: 1.0,
   902	                gradient_length_s: 600.0,
   903	                start_time_s: 0.0,
   904	            },
   905	            mz_range: (390.0, 900.0),
   906	            provenance: Provenance {
   907	                source: SchemeSource::Programmatic,
   908	                notes: String::new(),
   909	            },
   910	        };
   911	        assert!(s.validate().is_err());
   912	    }
   913	
   914	    #[test]
   915	    fn validate_rejects_bad_schemes() {
   916	        let repeat = RepeatPolicy::FixedCycleTime {
   917	            cycle_time_s: 1.0,
   918	            gradient_length_s: 600.0,
   919	            start_time_s: 0.0,
   920	        };
   921	        let ms1 = || Ms1Event {
   922	            analyzer: Analyzer::Ftms,
   923	            data_mode: DataMode::Profile,
   924	            mz_range: Some((390.0, 900.0)),
   925	            duration_s: None,
   926	        };
   927	        let win = || DiaWindow {
   928	            isolation: IsolationWindow {
   929	                center_mz: 500.0,
   930	                width_mz: 10.0,
   931	            },
   932	            collision_energy: CollisionEnergyPolicy::Value(25.0),
   933	            geometry: DiaGeometry::MzOnly,
   934	        };
   935	        let frame = |w: DiaWindow| {
   936	            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
   937	                windows: vec![w],
   938	                analyzer: Analyzer::Astms,
   939	                data_mode: DataMode::Centroid,
   940	                duration_s: None,
   941	                vendor_group_id: None,
   942	            })
   943	        };
   944	        let mk = |cycle: Vec<AcquisitionEvent>| AcquisitionScheme {
   945	            version: SCHEME_VERSION,
   946	            instrument: InstrumentKind::OrbitrapAstral,
   947	            cycle,
   948	            repeat,
   949	            mz_range: (390.0, 900.0),
   950	            provenance: Provenance {
   951	                source: SchemeSource::Programmatic,
   952	                notes: String::new(),
   953	            },
   954	        };
   955	
   956	        // valid baseline
   957	        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), frame(win())]).validate().is_ok());
   958	        // MS2 first (no leading MS1)
   959	        assert!(mk(vec![frame(win()), AcquisitionEvent::Ms1(ms1())]).validate().is_err());
   960	        // two MS1 in a cycle
   961	        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), AcquisitionEvent::Ms1(ms1()), frame(win())]).validate().is_err());
   962	        // MS1 with no MS2 frame
   963	        assert!(mk(vec![AcquisitionEvent::Ms1(ms1())]).validate().is_err());
   964	        // window outside mz_range
   965	        let mut oob = win();
   966	        oob.isolation.center_mz = 2000.0;
   967	        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), frame(oob)]).validate().is_err());
   968	        // negative collision energy
   969	        let mut neg = win();
   970	        neg.collision_energy = CollisionEnergyPolicy::Value(-5.0);
   971	        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), frame(neg)]).validate().is_err());
   972	        // bad event duration
   973	        let bad_dur = AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
   974	            windows: vec![win()],
   975	            analyzer: Analyzer::Astms,
   976	            data_mode: DataMode::Centroid,
   977	            duration_s: Some(-1.0),
   978	            vendor_group_id: None,
   979	        });
   980	        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), bad_dur]).validate().is_err());
   981	    }
   982	
  1020	            Ok(p) => p,
  1021	            Err(_) => {
  1022	                eprintln!("SKIP bruker_windows_round_trip: set TIMSIM_BRUKER_DIA_D");
  1023	                return;
  1024	            }
  1025	        };
  1026	        let scheme = AcquisitionScheme::from_bruker_d(&d).expect("extract");
  1027	        let regenerated = scheme.to_bruker_windows().expect("to_bruker_windows");
  1028	        let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
  1029	        assert_eq!(regenerated.len(), original.len(), "row count differs");
  1030	
  1031	        // Compare as a sorted multiset of EXACT field tuples, including the
  1032	        // WindowGroup id (preserved via vendor_group_id) — no normalization, so a
  1033	        // real id mismatch would fail. Row order in the table is not guaranteed.
  1034	        fn tuples(
  1035	            rows: &[crate::data::meta::DiaMsMsWindow],
  1036	        ) -> Vec<(u32, u32, u32, u64, u64, u64)> {
  1037	            let mut out: Vec<_> = rows
  1038	                .iter()
  1039	                .map(|r| {
  1040	                    (
  1041	                        r.window_group,
  1042	                        r.scan_num_begin,
  1043	                        r.scan_num_end,
  1044	                        r.isolation_mz.to_bits(),
  1045	                        r.isolation_width.to_bits(),
  1046	                        r.collision_energy.to_bits(),
  1047	                    )
  1048	                })
  1049	                .collect();
  1050	            out.sort_unstable();
  1051	            out
  1052	        }
  1053	        assert_eq!(
  1054	            tuples(&regenerated),
  1055	            tuples(&original),
  1056	            "round-trip windows differ from source (incl. WindowGroup id)"
  1057	        );
  1058	        eprintln!(
  1059	            "bruker_windows_round_trip OK: {} rows match across {} groups",
  1060	            original.len(),
  1061	            scheme
  1062	                .cycle

codex
The exact-ID golden comparison is now meaningful: it sorts only row tuples and retains raw `WindowGroup`, so renumbering can no longer hide. The `validate()` call does not add a new Bruker-extraction failure path because `from_bruker_d` already validates the same scheme before returning; it does change which error is reported first for arbitrary invalid/non-tims schemes.
 succeeded in 40050ms:
   Compiling libc v0.2.186
   Compiling proc-macro2 v1.0.106
   Compiling quote v1.0.45
   Compiling unicode-ident v1.0.24
   Compiling cfg-if v1.0.4
   Compiling find-msvc-tools v0.1.9
   Compiling shlex v2.0.1
   Compiling autocfg v1.5.1
   Compiling version_check v0.9.5
   Compiling libm v0.2.16
   Compiling serde_core v1.0.228
   Compiling zerocopy v0.8.50
   Compiling serde v1.0.228
   Compiling crossbeam-utils v0.8.21
   Compiling object v0.37.3
   Compiling rayon-core v1.13.0
   Compiling memchr v2.8.1
   Compiling getrandom v0.3.4
   Compiling either v1.16.0
   Compiling bitflags v2.13.0
   Compiling regex-syntax v0.8.10
   Compiling rustversion v1.0.22
   Compiling once_cell v1.21.4
   Compiling allocator-api2 v0.2.21
   Compiling ahash v0.8.12
   Compiling array-init-cursor v0.2.1
   Compiling num-traits v0.2.19
   Compiling smallvec v1.15.1
   Compiling parking_lot_core v0.9.12
   Compiling thiserror v1.0.69
   Compiling planus v0.3.1
   Compiling scopeguard v1.2.0
   Compiling aho-corasick v1.1.4
   Compiling itoa v1.0.18
   Compiling target-features v0.1.6
   Compiling crossbeam-epoch v0.9.18
   Compiling syn v1.0.109
   Compiling pkg-config v0.3.33
   Compiling crossbeam-deque v0.8.6
   Compiling syn v2.0.117
   Compiling lock_api v0.4.14
   Compiling polars-utils v0.43.1
   Compiling ryu v1.0.23
   Compiling static_assertions v1.1.0
   Compiling castaway v0.2.4
   Compiling simdutf8 v0.1.5
   Compiling equivalent v1.0.2
   Compiling hashbrown v0.17.1
   Compiling jobserver v0.1.34
   Compiling memmap2 v0.7.1
   Compiling cc v1.2.63
   Compiling parking_lot v0.12.5
   Compiling raw-cpuid v11.6.0
   Compiling matrixmultiply v0.3.10
   Compiling rayon v1.12.0
   Compiling polars-schema v0.43.1
   Compiling bytes v1.11.1
   Compiling iana-time-zone v0.1.65
   Compiling chrono v0.4.45
   Compiling num-integer v0.1.46
   Compiling num-complex v0.4.6
   Compiling polars-arrow v0.43.1
   Compiling rawpointer v0.2.1
   Compiling regex-automata v0.4.14
   Compiling atoi_simd v0.15.6
   Compiling ethnum v1.5.3
   Compiling dyn-clone v1.0.20
   Compiling indexmap v2.14.0
   Compiling rustix v1.1.4
   Compiling strength_reduce v0.2.4
   Compiling fast-float v0.2.0
   Compiling streaming-iterator v0.1.9
   Compiling polars-compute v0.43.1
   Compiling litrs v1.0.0
   Compiling stacker v0.1.24
   Compiling zstd-sys v2.0.16+zstd.1.5.7
   Compiling linux-raw-sys v0.12.1
   Compiling getrandom v0.2.17
   Compiling lz4-sys v1.11.1+lz4-1.10.0
   Compiling polars-core v0.43.1
   Compiling unicode-width v0.2.2
   Compiling rand_core v0.6.4
   Compiling crc32fast v1.5.0
   Compiling document-features v0.2.12
   Compiling paste v1.0.15
   Compiling unicode-segmentation v1.13.3
   Compiling alloc-no-stdlib v2.0.4
   Compiling zstd-safe v7.2.4
   Compiling ar_archive_writer v0.5.1
   Compiling alloc-stdlib v0.2.2
   Compiling ndarray v0.15.6
   Compiling polars-ops v0.43.1
   Compiling xxhash-rust v0.8.15
   Compiling adler2 v2.0.1
   Compiling simd-adler32 v0.3.9
   Compiling snap v1.1.1
   Compiling fallible-streaming-iterator v0.1.9
   Compiling ppv-lite86 v0.2.21
   Compiling miniz_oxide v0.8.9
   Compiling brotli-decompressor v4.0.3
   Compiling regex v1.12.3
   Compiling approx v0.5.1
   Compiling argminmax v0.6.3
   Compiling crossterm v0.29.0
   Compiling vcpkg v0.2.15
   Compiling rand_chacha v0.3.1
   Compiling utf8parse v0.2.2
   Compiling flate2 v1.1.9
   Compiling anstyle-parse v1.0.0
   Compiling psm v0.1.31
   Compiling rand v0.8.6
   Compiling comfy-table v7.2.2
   Compiling serde_derive v1.0.228
   Compiling bytemuck_derive v1.10.2
   Compiling thiserror-impl v1.0.69
   Compiling nalgebra-macros v0.2.2
   Compiling libsqlite3-sys v0.30.1
   Compiling rand_distr v0.4.3
   Compiling brotli v6.0.0
   Compiling streaming-decompression v0.1.2
   Compiling now v0.1.3
   Compiling num-rational v0.4.2
   Compiling atoi v2.0.0
   Compiling anstyle-query v1.1.5
   Compiling is_terminal_polyfill v1.70.2
   Compiling parquet-format-safe v0.2.4
   Compiling zmij v1.0.21
   Compiling anstyle v1.0.14
   Compiling virtue v0.0.18
   Compiling base64 v0.22.1
   Compiling colorchoice v1.0.5
   Compiling typenum v1.20.1
   Compiling anstream v1.0.0
   Compiling bytemuck v1.25.0
   Compiling polars v0.43.1
   Compiling percent-encoding v2.3.2
   Compiling clap_lex v1.1.0
   Compiling multiversion-macros v0.7.4
   Compiling safe_arch v0.7.4
   Compiling bincode_derive v2.0.1
   Compiling serde_json v1.0.150
   Compiling home v0.5.12
   Compiling strsim v0.11.1
   Compiling heck v0.5.0
   Compiling glob v0.3.3
   Compiling unty v0.0.4
   Compiling clap_builder v4.6.0
   Compiling clap_derive v4.6.1
   Compiling wide v0.7.33
   Compiling ordered-float v4.6.0
   Compiling itertools v0.14.0
   Compiling fallible-iterator v0.3.0
   Compiling bincode v2.0.1
   Compiling libloading v0.8.9
   Compiling rustc-hash v2.1.2
   Compiling lzf v1.0.0
   Compiling byteorder v1.5.0
   Compiling multiversion v0.7.4
   Compiling simba v0.9.1
   Compiling hashbrown v0.14.5
   Compiling polars-arrow-format v0.1.0
   Compiling compact_str v0.8.2
   Compiling bincode v1.3.3
   Compiling clap v4.6.1
   Compiling hashlink v0.9.1
   Compiling polars-error v0.43.1
   Compiling zstd v0.13.3
   Compiling nalgebra v0.33.3
   Compiling rusqlite v0.32.1
   Compiling lz4 v1.28.1
   Compiling polars-row v0.43.1
   Compiling statrs v0.18.0
   Compiling mscore v0.4.1 (/scratch/timsim-demo/SUBMISSION/rustims/mscore)
   Compiling polars-parquet v0.43.1
   Compiling polars-time v0.43.1
   Compiling polars-io v0.43.1
   Compiling rustdf v0.4.1 (/scratch/timsim-demo/SUBMISSION/rustims/rustdf)
warning: unused import: `rand::prelude::IteratorRandom`
  --> rustdf/src/data/dia.rs:12:5
   |
12 | use rand::prelude::IteratorRandom;
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default

warning: creating a mutable reference to mutable static
   --> rustdf/src/data/raw.rs:269:46
    |
269 |         if let Some((mz, intens)) = unsafe { EXTRACT_BUF_DATA.take() } {
    |                                              ^^^^^^^^^^^^^^^^^^^^^^^ mutable reference to mutable static
    |
    = note: mutable references to mutable statics are dangerous; it's undefined behavior if any other pointer to the static is used or if any other reference is created for the static while the mutable reference lives
    = note: for more information, see <https://doc.rust-lang.org/edition-guide/rust-2024/static-mut-references.html>
    = note: `#[warn(static_mut_refs)]` (part of `#[warn(rust_2024_compatibility)]`) on by default

warning: creating a mutable reference to mutable static
   --> rustdf/src/data/raw.rs:304:28
    |
304 |         let out = unsafe { PASEF_BUF_DATA.take() }.unwrap_or_default();
    |                            ^^^^^^^^^^^^^^^^^^^^^ mutable reference to mutable static
    |
    = note: mutable references to mutable statics are dangerous; it's undefined behavior if any other pointer to the static is used or if any other reference is created for the static while the mutable reference lives
    = note: for more information, see <https://doc.rust-lang.org/edition-guide/rust-2024/static-mut-references.html>

warning: `rustdf` (lib test) generated 3 warnings
    Finished `test` profile [unoptimized + debuginfo] target(s) in 39.96s
     Running unittests src/lib.rs (target/debug/deps/rustdf-308eae90b4e80287)

running 5 tests
test sim::scheme::tests::from_bruker_d_extracts_cycle ... ok
test sim::scheme::tests::bruker_windows_round_trip ... ok
test sim::scheme::tests::from_window_table_validates ... ok
test sim::scheme::tests::multi_window_frame_only_tims ... ok
test sim::scheme::tests::validate_rejects_bad_schemes ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 4 filtered out; finished in 0.00s


codex
**Findings**

1. **Medium: mixed or duplicate group IDs can silently collide.**  
   [scheme.rs:451](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:451) assigns `None` frames their positional ID without reserving IDs used by `Some`. For example, `[Some(2), None]` emits both frames as group `2`; two `Some(7)` frames also merge into one group. `validate()` does not detect either case. The adapter should either reject mixed/duplicate assignments or allocate fallback IDs from the unused ID set.

2. **Medium: the golden test may still not prove the regression is fixed.**  
   [scheme.rs:1031](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:1031) now correctly compares exact IDs, but if the gated dataset uses canonical IDs `1..N`, the old renumbering implementation would also pass. The test is also skipped when `TIMSIM_BRUKER_DIA_D` is unset. Add an unconditional unit test with noncanonical IDs such as `7` and `42`, asserting exact output IDs. Include a mixed-ID collision case once its semantics are defined.

3. **Low: `from_window_table` cannot preserve caller-specified Bruker IDs.**  
   [scheme.rs:220](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:220) always assigns `None`. This is fine for fresh injected schemes using generated IDs, and callers can construct `DiaMs2Frame` directly. It is a functional gap if CSV/window-table injection must match an existing `DiaFrameMsMsInfo` table with specific IDs.

The core extraction round trip is otherwise fixed correctly. Every `from_bruker_d` frame receives its exact `BTreeMap` key at [scheme.rs:546](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:546), while all six other construction sites explicitly use `None`. `to_bruker_windows` uses the preserved value correctly.

Calling `validate()` first changes error precedence for invalid/non-tims schemes, but does not newly reject schemes returned by `from_bruker_d`, because that extractor already performs the same validation. The overlap and m/z-containment risks therefore predate this revision; `mz_range` is derived from the exact extracted window edges.

Focused tests compile and pass: 5 passed. The environment-gated Bruker tests reported success but may have taken their skip path.
tokens used
86,136
**Findings**

1. **Medium: mixed or duplicate group IDs can silently collide.**  
   [scheme.rs:451](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:451) assigns `None` frames their positional ID without reserving IDs used by `Some`. For example, `[Some(2), None]` emits both frames as group `2`; two `Some(7)` frames also merge into one group. `validate()` does not detect either case. The adapter should either reject mixed/duplicate assignments or allocate fallback IDs from the unused ID set.

2. **Medium: the golden test may still not prove the regression is fixed.**  
   [scheme.rs:1031](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:1031) now correctly compares exact IDs, but if the gated dataset uses canonical IDs `1..N`, the old renumbering implementation would also pass. The test is also skipped when `TIMSIM_BRUKER_DIA_D` is unset. Add an unconditional unit test with noncanonical IDs such as `7` and `42`, asserting exact output IDs. Include a mixed-ID collision case once its semantics are defined.

3. **Low: `from_window_table` cannot preserve caller-specified Bruker IDs.**  
   [scheme.rs:220](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:220) always assigns `None`. This is fine for fresh injected schemes using generated IDs, and callers can construct `DiaMs2Frame` directly. It is a functional gap if CSV/window-table injection must match an existing `DiaFrameMsMsInfo` table with specific IDs.

The core extraction round trip is otherwise fixed correctly. Every `from_bruker_d` frame receives its exact `BTreeMap` key at [scheme.rs:546](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/scheme.rs:546), while all six other construction sites explicitly use `None`. `to_bruker_windows` uses the preserved value correctly.

Calling `validate()` first changes error precedence for invalid/non-tims schemes, but does not newly reject schemes returned by `from_bruker_d`, because that extractor already performs the same validation. The overlap and m/z-containment risks therefore predate this revision; `mz_range` is derived from the exact extracted window edges.

Focused tests compile and pass: 5 passed. The environment-gated Bruker tests reported success but may have taken their skip path.
