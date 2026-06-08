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
    pub mz_range: (f64, f64),
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
            } if cycle_time_s > 0.0 => {
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
        if self.ms1_count() == 0 {
            return Err("cycle has no MS1 event".into());
        }
        let (lo, hi) = self.mz_range;
        if !(lo.is_finite() && hi.is_finite()) || lo >= hi {
            return Err(format!("invalid mz_range ({lo}, {hi})"));
        }
        for ev in &self.cycle {
            if let AcquisitionEvent::DiaMs2Frame(frame) = ev {
                if frame.windows.is_empty() {
                    return Err("MS2 frame has no windows".into());
                }
                if frame.windows.len() > 1 && self.instrument != InstrumentKind::TimsTofDia {
                    return Err(
                        "multi-window MS2 frame is only valid for timsTOF (mobility-partitioned)"
                            .into(),
                    );
                }
                for w in &frame.windows {
                    if !(w.isolation.width_mz.is_finite() && w.isolation.width_mz > 0.0) {
                        return Err("window width must be finite and > 0".into());
                    }
                    if !w.isolation.center_mz.is_finite() {
                        return Err("window center m/z is not finite".into());
                    }
                    if let CollisionEnergyPolicy::Value(v) = w.collision_energy {
                        if !v.is_finite() {
                            return Err("non-finite collision energy".into());
                        }
                    }
                    match w.geometry {
                        DiaGeometry::TimsMobility {
                            scan_start,
                            scan_end,
                        } => {
                            if self.instrument != InstrumentKind::TimsTofDia {
                                return Err(
                                    "mobility geometry is only valid for timsTOF".into()
                                );
                            }
                            if scan_start > scan_end {
                                return Err("mobility scan_start > scan_end".into());
                            }
                        }
                        DiaGeometry::MzOnly => {}
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
            }
        }
        Ok(())
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
        let mut start_rt = 0.0f64;
        let mut cycle_time_s = 0.0f64;
        let mut win_lo = f64::INFINITY;
        let mut win_hi = f64::NEG_INFINITY;
        let mut any_astms = false;

        for scan in raw.first_scan..=raw.last_scan {
            let ev = match raw.scan_event(scan) {
                Some(e) => e,
                None => continue,
            };
            if ev.analyzer == 7 {
                any_astms = true;
            }
            if ev.ms_order <= 1 {
                if seen_ms1 {
                    // start of the next cycle -> close this one
                    cycle_time_s = (rt_of(scan) - start_rt).max(0.0);
                    break;
                }
                seen_ms1 = true;
                start_rt = rt_of(scan);
                cycle.push(AcquisitionEvent::Ms1(Ms1Event {
                    analyzer: analyzer_of(ev.analyzer),
                    data_mode: data_mode_of(scan),
                    mz_range: (0.0, 0.0), // filled below
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

        if cycle.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "no scan events found in template",
            ));
        }
        let mz_range = if win_lo.is_finite() && win_hi > win_lo {
            (win_lo, win_hi)
        } else {
            (0.0, 0.0)
        };
        for ev in &mut cycle {
            if let AcquisitionEvent::Ms1(m) = ev {
                m.mz_range = mz_range;
            }
        }
        let gradient_length_s = raw.index.last().map(|e| e.time).unwrap_or(0.0);
        let n_ms2 = cycle
            .iter()
            .filter(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_)))
            .count();

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
                mz_range: (390.0, 900.0),
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
                    mz_range: (390.0, 900.0),
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
