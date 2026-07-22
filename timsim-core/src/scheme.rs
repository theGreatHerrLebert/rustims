//! Bruker/Thermo I/O adapters for the acquisition scheme (output/extraction side).
//!
//! The scheme *types* live in the dependency-free `timsim-types` leaf and are re-exported
//! here. The adapters that read/write a real `.d`/`.raw` (via `ms-io` / `thermorawfile`)
//! cannot be inherent methods on a foreign type, so they are extension traits — [`SchemeIo`]
//! (Bruker, always) and [`SchemeThermoIo`] (Thermo, feature `thermo`). Bring them into scope.

use std::io;

pub use timsim_types::*;

// Private: the single source of truth for window-group ids, shared by to_bruker_windows +
// to_bruker_info (was an inherent helper on AcquisitionScheme before the leaf split).
/// Per-cycle-position window-group id: `None` for an MS1 event, `Some(group)`
/// for an MS2 frame. The group id is the frame's preserved `vendor_group_id`
/// (Bruker `WindowGroup`) when set, else a collision-safe positional id drawn
/// from the unused set (so preserved and fallback ids never collide).
/// Duplicate explicit ids are rejected. Validates first; requires a timsTOF
/// scheme. This is the single source of truth shared by `to_bruker_windows`
/// and `to_bruker_info`, so the two tables always agree.
fn bruker_group_layout(scheme: &AcquisitionScheme) -> Result<Vec<Option<u32>>, String> {
    scheme.validate()?;
    if scheme.instrument != InstrumentKind::TimsTofDia {
        return Err("Bruker tables require a timsTOF (TimsTofDia) scheme".into());
    }
    use std::collections::HashSet;
    let mut reserved: HashSet<u32> = HashSet::new();
    for ev in &scheme.cycle {
        if let AcquisitionEvent::DiaMs2Frame(f) = ev {
            if let Some(g) = f.vendor_group_id {
                if !reserved.insert(g) {
                    return Err(format!("duplicate vendor window-group id {g}"));
                }
            }
        }
    }
    let mut layout = Vec::with_capacity(scheme.cycle.len());
    let mut assigned: HashSet<u32> = HashSet::new();
    let mut next_seq: u32 = 1;
    for ev in &scheme.cycle {
        match ev {
            AcquisitionEvent::Ms1(_) => layout.push(None),
            AcquisitionEvent::DiaMs2Frame(frame) => {
                let group = match frame.vendor_group_id {
                    Some(g) => g,
                    None => {
                        while reserved.contains(&next_seq) || assigned.contains(&next_seq) {
                            next_seq += 1;
                        }
                        next_seq
                    }
                };
                if !assigned.insert(group) {
                    return Err(format!("window-group id {group} collides"));
                }
                layout.push(Some(group));
            }
        }
    }
    Ok(layout)
}

/// Bruker DIA-table adapters + `.d` extraction for [`AcquisitionScheme`]. Need `ms-io`, so they are
/// an extension trait in `timsim-core` rather than inherent methods on the leaf type.
pub trait SchemeIo: Sized {
    /// Render the scheme's MS2 windows as Bruker `DiaFrameMsMsWindows` rows.
    fn to_bruker_windows(&self) -> Result<Vec<ms_io::data::meta::DiaMsMsWindow>, String>;
    /// Bruker `DiaFrameMsMsInfo` (frame -> window group) rows for a `num_frames`-frame run.
    fn to_bruker_info(
        &self,
        num_frames: u32,
    ) -> Result<Vec<ms_io::data::meta::DiaMsMisInfo>, String>;
    /// Both Bruker DIA tables for a `num_frames`-frame run, with consistent window-group ids.
    fn to_bruker_tables(
        &self,
        num_frames: u32,
    ) -> Result<
        (
            Vec<ms_io::data::meta::DiaMsMsWindow>,
            Vec<ms_io::data::meta::DiaMsMisInfo>,
        ),
        String,
    >;
    /// Extract the acquisition scheme from a real Bruker timsTOF DIA `.d`.
    fn from_bruker_d<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self>;
}

impl SchemeIo for AcquisitionScheme {
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
    fn to_bruker_windows(&self) -> Result<Vec<ms_io::data::meta::DiaMsMsWindow>, String> {
        let layout = bruker_group_layout(self)?;
        let mut rows = Vec::new();
        for (ev, slot) in self.cycle.iter().zip(&layout) {
            if let (AcquisitionEvent::DiaMs2Frame(frame), Some(group)) = (ev, slot) {
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
                    rows.push(ms_io::data::meta::DiaMsMsWindow {
                        window_group: *group,
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

    /// Generate the Bruker `DiaFrameMsMsInfo` (frame → window group) rows for a run
    /// of `num_frames` total frames, tiling the scheme's cycle. Cycle position
    /// `(frame_id - 1) % cycle_len` selects the event (1-based frame ids), MS1
    /// frames produce no row, and each MS2 frame maps to its window-group id (the
    /// same ids `to_bruker_windows` emits, so the two tables reference the same
    /// groups). The final cycle may be partial.
    ///
    /// Precondition: this models a **clean generated run** — frame 1 is the
    /// cycle's leading MS1 and frame ids are contiguous (as TimSim produces). It
    /// is not a reproducer for arbitrary real files with prefix/calibration
    /// frames, gaps, or acquisition starting mid-cycle.
    fn to_bruker_info(
        &self,
        num_frames: u32,
    ) -> Result<Vec<ms_io::data::meta::DiaMsMisInfo>, String> {
        // Bound the work so an absurd frame count errors cleanly instead of OOM.
        const MAX_FRAMES: u32 = 100_000_000;
        if num_frames > MAX_FRAMES {
            return Err(format!(
                "num_frames {num_frames} exceeds the {MAX_FRAMES} limit"
            ));
        }
        let layout = bruker_group_layout(self)?;
        let fpc = layout.len() as u32;
        if fpc == 0 {
            return Err("empty cycle".into());
        }
        let mut rows = Vec::new();
        for frame_id in 1..=num_frames {
            let pos = ((frame_id - 1) % fpc) as usize;
            if let Some(group) = layout[pos] {
                rows.push(ms_io::data::meta::DiaMsMisInfo {
                    frame_id,
                    window_group: group,
                });
            }
        }
        Ok(rows)
    }

    /// Both Bruker DIA tables for a `num_frames`-frame run: the
    /// `DiaFrameMsMsWindows` rows and the `DiaFrameMsMsInfo` (frame→group) rows,
    /// with consistent window-group ids.
    fn to_bruker_tables(
        &self,
        num_frames: u32,
    ) -> Result<
        (
            Vec<ms_io::data::meta::DiaMsMsWindow>,
            Vec<ms_io::data::meta::DiaMsMisInfo>,
        ),
        String,
    > {
        Ok((self.to_bruker_windows()?, self.to_bruker_info(num_frames)?))
    }

    /// Extract the acquisition scheme from a real Bruker timsTOF DIA `.d`.
    ///
    /// Reads `DiaFrameMsMsWindows` and the frame table. Each **window group**
    /// becomes one [`DiaMs2Frame`] holding its mobility-partitioned windows
    /// (`TimsMobility` geometry — the scan ranges are Bruker-grid coordinates),
    /// preceded by a single MS1 event. Cycle timing comes from the spacing of
    /// the precursor (MS1) frames. The returned scheme is validated.
    ///
    /// Window groups (frames) are ordered by their **first occurrence in
    /// `DiaFrameMsMsInfo`** (ascending frame id), i.e. the real acquisition order
    /// within a cycle — not merely ascending `WindowGroup` id — so the cycle
    /// faithfully represents permuted/reused group numbering. (If the info table
    /// is absent, falls back to ascending group id.)
    fn from_bruker_d<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
        use ms_io::data::meta::{read_dia_ms_ms_info, read_dia_ms_ms_windows, read_meta_data_sql};
        use std::collections::BTreeMap;

        let folder = path.as_ref().to_string_lossy().into_owned();
        let to_io = |e: Box<dyn std::error::Error>| {
            io::Error::new(io::ErrorKind::InvalidData, e.to_string())
        };
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

        // Order groups by first occurrence (ascending frame id) in DiaFrameMsMsInfo
        // = real intra-cycle acquisition order. Missing info -> ascending group id.
        let info = read_dia_ms_ms_info(&folder).unwrap_or_default();
        let mut first_frame: BTreeMap<u32, u32> = BTreeMap::new();
        for r in &info {
            first_frame
                .entry(r.window_group)
                .and_modify(|f| {
                    if r.frame_id < *f {
                        *f = r.frame_id
                    }
                })
                .or_insert(r.frame_id);
        }
        let mut ordered_groups: Vec<u32> = by_group.keys().copied().collect();
        // Stable sort by first-occurrence frame; groups absent from the info table
        // (key u32::MAX) keep their ascending-id relative order.
        ordered_groups.sort_by_key(|g| first_frame.get(g).copied().unwrap_or(u32::MAX));

        let mut cycle = vec![AcquisitionEvent::Ms1(Ms1Event {
            analyzer: Analyzer::Tof,
            data_mode: DataMode::Centroid,
            mz_range: None,
            duration_s: None,
        })];
        let n_groups = ordered_groups.len();
        for group in ordered_groups {
            let ws = by_group.remove(&group).expect("group present");
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

        let times: Vec<f64> = frames
            .iter()
            .map(|f| f.time)
            .filter(|t| t.is_finite())
            .collect();
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

/// Thermo `.raw` adapters for [`AcquisitionScheme`] (feature `thermo`).
#[cfg(feature = "thermo")]
pub trait SchemeThermoIo: Sized {
    /// Extract the acquisition scheme from a real Thermo `.raw`.
    fn from_thermo_raw<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self>;
    /// The build-from-template frame schedule (one entry per template scan).
    fn thermo_frame_schedule<P: AsRef<std::path::Path>>(path: P) -> io::Result<Vec<TemplateScan>>;
}

#[cfg(feature = "thermo")]
impl SchemeThermoIo for AcquisitionScheme {
    /// Extract the acquisition scheme from a real Thermo `.raw` by walking the
    /// first complete cycle (first MS1 up to the next MS1). Each MS2 scan becomes
    /// a single-window [`DiaMs2Frame`] (Thermo has no mobility), with the
    /// observed isolation center/width and CE.
    fn from_thermo_raw<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
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
        let any_astms = cycle.iter().any(
            |e| matches!(e, AcquisitionEvent::DiaMs2Frame(f) if f.analyzer == Analyzer::Astms),
        );

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

    /// Extract the template's FULL per-scan schedule (every scan, not just the
    /// first cycle): scan number, its actual retention time, ms-level, and (for
    /// MS2) the isolation window + collision energy.
    ///
    /// This is the build-from-template timeline source (P6e): an Astral run mirrors
    /// these frames 1:1, so the trunk's per-frame abundances are keyed to the
    /// template's real scan RTs and the authored output's RT/schedule are correct
    /// by construction (rather than relabelling a Bruker-timed frame with a template
    /// RT). The RT is the template's recorded time; ms-level comes from the scan
    /// event (falling back to the profile/centroid packet type when absent).
    fn thermo_frame_schedule<P: AsRef<std::path::Path>>(path: P) -> io::Result<Vec<TemplateScan>> {
        use thermorawfile::RawFile;
        let raw = RawFile::open(path)?;
        // Reject a template whose per-scan scan events can't be decoded at all — without
        // them the ms-level/isolation are unknown and the schedule would be unreliable.
        // (Both fixed-stride (Astral/Velos) and variable-length (Fusion-class) layouts
        // decode; this guards a layout whose event grammar we can't walk.)
        if !raw.has_scan_events() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "template scan events could not be decoded (unsupported scan-event layout); \
                 build-from-template needs per-scan ms-level/isolation.",
            ));
        }
        let mut out = Vec::with_capacity(raw.index.len());
        for scan in raw.first_scan..=raw.last_scan {
            let i = (scan - raw.first_scan) as usize;
            // Thermo stores scan time in MINUTES; expose SECONDS so the `_s` field is
            // honest and downstream (seconds-based) consumers need no further scaling.
            let rt = raw.index[i].time * 60.0;
            let ev = raw.scan_event(scan);
            // ms-level: prefer the scan event; else classify by packet type
            // (a non-zero profile section = MS1 FTMS).
            let ms_level = match ev.as_ref() {
                Some(e) if e.ms_order >= 1 => e.ms_order,
                _ => {
                    let pkt = (raw.data_addr + raw.index[i].offset) as usize;
                    if pkt + 8 <= raw.bytes.len()
                        && u32::from_le_bytes(raw.bytes[pkt + 4..pkt + 8].try_into().unwrap()) > 0
                    {
                        1
                    } else {
                        2
                    }
                }
            };
            let (isolation, collision_energy) = match ev.as_ref() {
                Some(e) if ms_level > 1 => (
                    Some(IsolationWindow {
                        center_mz: e.isolation_center,
                        width_mz: e.isolation_width,
                    }),
                    Some(e.collision_energy),
                ),
                _ => (None, None),
            };
            out.push(TemplateScan {
                scan,
                retention_time_s: rt,
                ms_level,
                isolation,
                collision_energy,
            });
        }
        if out.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "template has no scans",
            ));
        }
        // DIA-suitability: a DIA template tiles the m/z range with a SMALL set of windows
        // that REPEAT across cycles; a DDA acquisition has a (near-)unique precursor per
        // MS2 scan — no tiling. Building a DIA simulation from DDA precursors is degenerate
        // (simulated peptides almost never fall in an arbitrary one-off window), so reject
        // a DDA-shaped schedule with a clear message instead of producing an empty run.
        let n_ms2 = out.iter().filter(|s| s.ms_level > 1).count();
        if n_ms2 > 0 {
            let distinct: std::collections::HashSet<i64> = out
                .iter()
                .filter_map(|s| s.isolation.map(|w| (w.center_mz * 100.0).round() as i64))
                .collect();
            // DIA tiles with a small, fixed window set that repeats every cycle — even
            // ultra-narrow DIA stays in the low hundreds and repeats heavily. DDA has
            // thousands of distinct precursors, each seen only a handful of times.
            // Require BOTH a large absolute window count (run-length-independent) AND a
            // high distinct/MS2 ratio, so neither a short DIA run nor an extreme-narrow
            // DIA scheme is misjudged. Measured: DIA ~75 windows @ ratio 0.0005;
            // DDA ~20909 @ ratio 0.40.
            let ratio = distinct.len() as f64 / n_ms2 as f64;
            if distinct.len() > 2000 && ratio > 0.1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "template looks like DDA, not DIA: {} distinct isolation centers \
                         across {} MS2 scans (DIA windows repeat across cycles; DDA has a \
                         unique precursor per scan). Build-from-template needs a DIA scheme.",
                        distinct.len(),
                        n_ms2
                    ),
                ));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn swath_scheme(cycle_time_s: f64, gradient_length_s: f64) -> AcquisitionScheme {
        let ms1 = Ms1Event {
            analyzer: Analyzer::Tof,
            data_mode: DataMode::Centroid,
            mz_range: None,
            duration_s: None,
        };
        let mk = |center: f64, width: f64| DiaWindow {
            isolation: IsolationWindow {
                center_mz: center,
                width_mz: width,
            },
            collision_energy: CollisionEnergyPolicy::Linear {
                intercept: 5.0,
                slope_per_mz: 0.045,
            },
            geometry: DiaGeometry::MzOnly,
        };
        AcquisitionScheme::from_window_table(
            InstrumentKind::SciexZenoTof,
            ms1,
            vec![mk(420.0, 40.0), mk(480.0, 20.0)],
            RepeatPolicy::FixedCycleTime {
                cycle_time_s,
                gradient_length_s,
                start_time_s: 0.0,
            },
            (400.0, 500.0),
        )
    }

    #[test]
    fn dia_frame_schedule_expands_cycles() {
        // 3 cycles of [MS1, MS2(420), MS2(480)] over a 9 s gradient at 3 s cycles.
        let sched = swath_scheme(3.0, 9.0).dia_frame_schedule();
        assert_eq!(sched.len(), 9, "3 cycles x 3 events");
        // Contiguous 1-based scan ids.
        assert_eq!(
            sched.iter().map(|r| r.0).collect::<Vec<_>>(),
            (1u32..=9).collect::<Vec<_>>()
        );
        // RT strictly increasing.
        for w in sched.windows(2) {
            assert!(
                w[1].1 > w[0].1,
                "RT must increase: {:?} !> {:?}",
                w[1].1,
                w[0].1
            );
        }
        // Cycle starts: rows 0,3,6 at t = 0,3,6.
        for (k, &i) in [0usize, 3, 6].iter().enumerate() {
            assert!((sched[i].1 - (k as f64) * 3.0).abs() < 1e-9);
            assert_eq!(sched[i].2, 1, "cycle starts with MS1");
            assert!(
                sched[i].3.is_none() && sched[i].5.is_none(),
                "MS1 has no iso/CE"
            );
        }
        // First MS2: center 420, width 40, CE resolved by the linear model (>0).
        assert_eq!(sched[1].2, 2);
        assert!((sched[1].3.unwrap() - 420.0).abs() < 1e-9);
        assert!((sched[1].4.unwrap() - 40.0).abs() < 1e-9);
        assert!(sched[1].5.unwrap() > 0.0, "rolling CE resolved for MS2");
    }

    #[test]
    fn dia_frame_schedule_unknown_ce_is_none() {
        let mut s = swath_scheme(3.0, 6.0);
        // Force all window CE policies to Unknown.
        for e in &mut s.cycle {
            if let AcquisitionEvent::DiaMs2Frame(f) = e {
                for w in &mut f.windows {
                    w.collision_energy = CollisionEnergyPolicy::Unknown;
                }
            }
        }
        let sched = s.dia_frame_schedule();
        for r in &sched {
            if r.2 == 2 {
                assert!(
                    r.5.is_none(),
                    "Unknown CE -> None (caller must supply a model)"
                );
            }
        }
    }

    #[test]
    fn activation_condition_carries_typed_unit() {
        let c = ActivationCondition::collisional_ev(30.8);
        assert_eq!(c.method, ActivationMethod::Hcd);
        assert_eq!(c.unit, EnergyUnit::ElectronVolt);
        assert_eq!(c.value, 30.8);

        // The legacy provenance is collisional eV (timsTOF assumption).
        let legacy = ActivationCondition::legacy_bruker();
        assert_eq!(legacy.unit, EnergyUnit::ElectronVolt);

        // An NCE value is a distinct unit and must not compare equal to eV.
        let nce = ActivationCondition {
            method: ActivationMethod::Hcd,
            value: 30.0,
            unit: EnergyUnit::NormalizedCe,
        };
        assert_ne!(nce.unit, c.unit);
    }

    #[test]
    fn bruker_pasef_activation_policy_reproduces_legacy_ce() {
        // Legacy dda_selection_scheme: collision_energy = ce_bias + ce_slope*scan.
        let (ce_bias, ce_slope) = (54.1984, -0.0345);
        let p = ActivationPolicy::bruker_pasef(ce_bias, ce_slope);
        assert_eq!(p.method, ActivationMethod::Hcd);
        assert_eq!(p.unit, EnergyUnit::ElectronVolt);
        for scan in [0u32, 1, 250, 451, 917] {
            assert_eq!(
                p.collision_energy_for_scan(scan),
                Some(ce_bias + ce_slope * scan as f64),
                "CE must match the legacy formula at scan {scan}"
            );
            assert_eq!(
                p.condition_for_scan(scan).unwrap().value,
                ce_bias + ce_slope * scan as f64
            );
        }
        // A per-window (DIA) model has no scan dependence: scan evaluation is
        // None (not a silently-wrong value), and CE comes from the window m/z.
        let w = ActivationPolicy {
            method: ActivationMethod::Hcd,
            unit: EnergyUnit::ElectronVolt,
            model: CollisionEnergyModel::PerWindow(CollisionEnergyPolicy::Value(25.0)),
        };
        assert_eq!(w.condition_for_window(700.0).unwrap().value, 25.0);
        assert_eq!(w.collision_energy_for_scan(100), None);
        assert!(w.condition_for_scan(100).is_none());
    }

    #[test]
    fn instrument_capabilities_default_is_bruker() {
        // Default = full timsTOF capability so existing behaviour is unchanged;
        // an Astral-like instrument keeps quad isolation but drops mobility +
        // mobility-dependent isotope transmission.
        let bruker = InstrumentCapabilities::default();
        assert!(bruker.has_tims_mobility);
        assert!(bruker.has_quad_isotope_transmission);
        assert_eq!(bruker, InstrumentCapabilities::bruker_timstof());

        // P6c: the named Astral constructor is both-false (isotope mode gated off).
        let astral = InstrumentCapabilities::astral();
        assert!(!astral.has_tims_mobility);
        assert!(!astral.has_quad_isotope_transmission);
        assert_ne!(bruker, astral);
    }

    #[test]
    fn thermo_nce_activation_policy_is_normalized_per_window() {
        // P6c: Astral NCE policy. Per-window normalized CE; no scan dependence.
        let p = ActivationPolicy::thermo_nce(CollisionEnergyPolicy::Value(27.0));
        assert_eq!(p.method, ActivationMethod::Hcd);
        assert_eq!(p.unit, EnergyUnit::NormalizedCe);

        // Resolves a window CE as an NCE-unit condition (NOT eV) — a downstream
        // eV-calibrated predictor must be able to reject it on the unit alone.
        let cond = p
            .condition_for_window(650.0)
            .expect("NCE condition for window");
        assert_eq!(cond.value, 27.0);
        assert_eq!(cond.unit, EnergyUnit::NormalizedCe);
        assert_ne!(cond.unit, EnergyUnit::ElectronVolt);

        // No IMS: there is no scan-parameterised CE.
        assert_eq!(p.collision_energy_for_scan(100), None);
        assert!(p.condition_for_scan(100).is_none());

        // A rolling (linear) NCE model resolves per window center.
        let rolling = ActivationPolicy::thermo_nce(CollisionEnergyPolicy::Linear {
            intercept: 20.0,
            slope_per_mz: 0.01,
        });
        assert_eq!(
            rolling.condition_for_window(700.0).unwrap().value,
            20.0 + 0.01 * 700.0
        );
        assert_eq!(
            rolling.condition_for_window(700.0).unwrap().unit,
            EnergyUnit::NormalizedCe
        );
    }

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
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), frame(win())])
            .validate()
            .is_ok());
        // MS2 first (no leading MS1)
        assert!(mk(vec![frame(win()), AcquisitionEvent::Ms1(ms1())])
            .validate()
            .is_err());
        // two MS1 in a cycle
        assert!(mk(vec![
            AcquisitionEvent::Ms1(ms1()),
            AcquisitionEvent::Ms1(ms1()),
            frame(win())
        ])
        .validate()
        .is_err());
        // MS1 with no MS2 frame
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1())]).validate().is_err());
        // window outside mz_range
        let mut oob = win();
        oob.isolation.center_mz = 2000.0;
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), frame(oob)])
            .validate()
            .is_err());
        // negative collision energy
        let mut neg = win();
        neg.collision_energy = CollisionEnergyPolicy::Value(-5.0);
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), frame(neg)])
            .validate()
            .is_err());
        // bad event duration
        let bad_dur = AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
            windows: vec![win()],
            analyzer: Analyzer::Astms,
            data_mode: DataMode::Centroid,
            duration_s: Some(-1.0),
            vendor_group_id: None,
        });
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), bad_dur])
            .validate()
            .is_err());
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
        let multi = s
            .cycle
            .iter()
            .any(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(f) if f.windows.len() > 1));
        let n_frames = s
            .cycle
            .iter()
            .filter(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_)))
            .count();
        eprintln!(
            "from_bruker_d OK: TimsTofDia, {} frames, {} windows ({}), mz {:.1}..{:.1}",
            n_frames,
            n_win,
            if multi {
                "mobility-partitioned"
            } else {
                "single-window"
            },
            s.mz_range.0,
            s.mz_range.1
        );
    }

    // Gated: set TIMSIM_SCIEX_WIFF to a real ZenoTOF .wiff (OLE2 method).
    // Unconditional: to_bruker_windows must preserve explicit vendor_group_ids
    // (incl. non-canonical), allocate non-colliding ids for None frames, and
    // reject duplicate explicit ids. (The gated round-trip uses a real .d whose
    // ids are already 1..N, so it can't prove non-canonical preservation alone.)
    #[test]
    fn to_bruker_windows_group_ids() {
        let frame = |gid: Option<u32>, center: f64| {
            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: vec![DiaWindow {
                    isolation: IsolationWindow {
                        center_mz: center,
                        width_mz: 10.0,
                    },
                    collision_energy: CollisionEnergyPolicy::Value(25.0),
                    geometry: DiaGeometry::TimsMobility {
                        scan_start: 0,
                        scan_end: 100,
                    },
                }],
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: gid,
            })
        };
        let mk = |frames: Vec<AcquisitionEvent>| {
            let mut cycle = vec![AcquisitionEvent::Ms1(Ms1Event {
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                mz_range: None,
                duration_s: None,
            })];
            cycle.extend(frames);
            AcquisitionScheme {
                version: SCHEME_VERSION,
                instrument: InstrumentKind::TimsTofDia,
                cycle,
                repeat: RepeatPolicy::FixedCycleTime {
                    cycle_time_s: 1.0,
                    gradient_length_s: 600.0,
                    start_time_s: 0.0,
                },
                mz_range: (300.0, 900.0),
                provenance: Provenance {
                    source: SchemeSource::Programmatic,
                    notes: String::new(),
                },
            }
        };
        let ids = |s: &AcquisitionScheme| {
            s.to_bruker_windows()
                .map(|r| r.iter().map(|w| w.window_group).collect::<Vec<_>>())
        };

        // preserved non-canonical ids
        assert_eq!(
            ids(&mk(vec![frame(Some(7), 500.0), frame(Some(42), 600.0)])).unwrap(),
            vec![7, 42]
        );
        // all-None -> sequential 1..N
        assert_eq!(
            ids(&mk(vec![frame(None, 500.0), frame(None, 600.0)])).unwrap(),
            vec![1, 2]
        );
        // mixed: None allocates a free id (1), avoiding the reserved 7
        assert_eq!(
            ids(&mk(vec![frame(Some(7), 500.0), frame(None, 600.0)])).unwrap(),
            vec![7, 1]
        );
        // duplicate explicit ids rejected
        assert!(ids(&mk(vec![frame(Some(5), 500.0), frame(Some(5), 600.0)])).is_err());
    }

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
        let original = ms_io::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
        assert_eq!(regenerated.len(), original.len(), "row count differs");

        // Compare as a sorted multiset of EXACT field tuples, including the
        // WindowGroup id (preserved via vendor_group_id) — no normalization, so a
        // real id mismatch would fail. Row order in the table is not guaranteed.
        fn tuples(
            rows: &[ms_io::data::meta::DiaMsMsWindow],
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

    // Unconditional: DiaFrameMsMsInfo tiling + windows/info group-id consistency.
    #[test]
    fn to_bruker_info_tiling() {
        let frame = |c: f64| {
            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: vec![DiaWindow {
                    isolation: IsolationWindow {
                        center_mz: c,
                        width_mz: 10.0,
                    },
                    collision_energy: CollisionEnergyPolicy::Value(25.0),
                    geometry: DiaGeometry::TimsMobility {
                        scan_start: 0,
                        scan_end: 100,
                    },
                }],
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: None,
            })
        };
        let s = AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: InstrumentKind::TimsTofDia,
            cycle: vec![
                AcquisitionEvent::Ms1(Ms1Event {
                    analyzer: Analyzer::Tof,
                    data_mode: DataMode::Centroid,
                    mz_range: None,
                    duration_s: None,
                }),
                frame(500.0),
                frame(600.0),
            ],
            repeat: RepeatPolicy::FixedCycleTime {
                cycle_time_s: 1.0,
                gradient_length_s: 600.0,
                start_time_s: 0.0,
            },
            mz_range: (300.0, 900.0),
            provenance: Provenance {
                source: SchemeSource::Programmatic,
                notes: String::new(),
            },
        };
        // cycle = [MS1, g1, g2]; num_frames=6: 1->skip, 2->g1, 3->g2, 4->skip, 5->g1, 6->g2
        let info: Vec<(u32, u32)> = s
            .to_bruker_info(6)
            .unwrap()
            .iter()
            .map(|r| (r.frame_id, r.window_group))
            .collect();
        assert_eq!(info, vec![(2, 1), (3, 2), (5, 1), (6, 2)]);
        // The two tables must reference the same set of window-group ids.
        let (windows, info2) = s.to_bruker_tables(6).unwrap();
        let wg: std::collections::BTreeSet<u32> = windows.iter().map(|w| w.window_group).collect();
        let ig: std::collections::BTreeSet<u32> = info2.iter().map(|r| r.window_group).collect();
        assert_eq!(wg, ig, "windows/info group ids disagree");

        // Non-ascending explicit ids + a multi-window (mobility-partitioned) frame:
        // info must follow CYCLE order (not sorted id) and emit ONE row per frame.
        let win = |center: f64, s0: u32, s1: u32| DiaWindow {
            isolation: IsolationWindow {
                center_mz: center,
                width_mz: 10.0,
            },
            collision_energy: CollisionEnergyPolicy::Value(20.0),
            geometry: DiaGeometry::TimsMobility {
                scan_start: s0,
                scan_end: s1,
            },
        };
        let dia = |gid: u32, ws: Vec<DiaWindow>| {
            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: ws,
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: Some(gid),
            })
        };
        let s2 = AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: InstrumentKind::TimsTofDia,
            cycle: vec![
                AcquisitionEvent::Ms1(Ms1Event {
                    analyzer: Analyzer::Tof,
                    data_mode: DataMode::Centroid,
                    mz_range: None,
                    duration_s: None,
                }),
                dia(7, vec![win(500.0, 0, 50), win(500.0, 51, 100)]), // 2 mobility windows
                dia(2, vec![win(600.0, 0, 100)]),
            ],
            repeat: RepeatPolicy::FixedCycleTime {
                cycle_time_s: 1.0,
                gradient_length_s: 600.0,
                start_time_s: 0.0,
            },
            mz_range: (300.0, 900.0),
            provenance: Provenance {
                source: SchemeSource::Programmatic,
                notes: String::new(),
            },
        };
        // cycle len 3; frame2->g7 (first), frame3->g2 — acquisition order, NOT sorted.
        let info3: Vec<(u32, u32)> = s2
            .to_bruker_info(3)
            .unwrap()
            .iter()
            .map(|r| (r.frame_id, r.window_group))
            .collect();
        assert_eq!(
            info3,
            vec![(2, 7), (3, 2)],
            "info must follow cycle order, one row/frame"
        );
        // windows: the 2-window frame emits 2 rows (group 7), the other 1 (group 2).
        let w3 = s2.to_bruker_windows().unwrap();
        assert_eq!(w3.iter().filter(|w| w.window_group == 7).count(), 2);
        assert_eq!(w3.iter().filter(|w| w.window_group == 2).count(), 1);
    }

    // Gated: regenerate DiaFrameMsMsInfo for the .d's full frame count and match.
    #[test]
    fn bruker_info_round_trip() {
        let d = match std::env::var("TIMSIM_BRUKER_DIA_D") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP bruker_info_round_trip: set TIMSIM_BRUKER_DIA_D");
                return;
            }
        };
        let scheme = AcquisitionScheme::from_bruker_d(&d).expect("extract");
        let frames = ms_io::data::meta::read_meta_data_sql(&d).expect("frames");
        let num_frames = frames.iter().map(|f| f.id).max().unwrap_or(0) as u32;
        let regenerated = scheme.to_bruker_info(num_frames).expect("to_bruker_info");
        let original = ms_io::data::meta::read_dia_ms_ms_info(&d).expect("read source");
        let key = |r: &ms_io::data::meta::DiaMsMisInfo| (r.frame_id, r.window_group);
        let mut a: Vec<_> = regenerated.iter().map(key).collect();
        let mut b: Vec<_> = original.iter().map(key).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "DiaFrameMsMsInfo round-trip differs from source");
        eprintln!(
            "bruker_info_round_trip OK: {} (frame, group) rows over {num_frames} frames",
            a.len()
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
            s.instrument,
            n_win,
            s.mz_range.0,
            s.mz_range.1,
            match s.repeat {
                RepeatPolicy::FixedCycleTime { cycle_time_s, .. } => cycle_time_s,
            }
        );
    }

    #[cfg(feature = "thermo")]
    #[test]
    fn thermo_frame_schedule_covers_every_scan() {
        let template = match std::env::var("TIMSIM_ASTRAL_TEMPLATE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!(
                    "SKIP thermo_frame_schedule_covers_every_scan: set TIMSIM_ASTRAL_TEMPLATE"
                );
                return;
            }
        };
        let sched = AcquisitionScheme::thermo_frame_schedule(&template).expect("schedule");
        assert!(
            sched.len() > 100,
            "expected the full run, got {}",
            sched.len()
        );
        // Scans are contiguous and ascending; RT is finite and non-decreasing.
        let mut last_rt = f64::NEG_INFINITY;
        let (mut n_ms1, mut n_ms2) = (0usize, 0usize);
        for (k, f) in sched.iter().enumerate() {
            assert_eq!(f.scan, sched[0].scan + k as u32, "scans must be contiguous");
            assert!(f.retention_time_s.is_finite());
            assert!(
                f.retention_time_s >= last_rt - 1e-6,
                "RT must be non-decreasing"
            );
            last_rt = f.retention_time_s;
            if f.ms_level <= 1 {
                n_ms1 += 1;
                assert!(f.isolation.is_none(), "MS1 carries no isolation window");
            } else {
                n_ms2 += 1;
                let iso = f.isolation.expect("MS2 carries an isolation window");
                assert!(iso.width_mz > 0.0 && iso.center_mz > 0.0);
                assert!(
                    f.collision_energy.is_some(),
                    "MS2 carries a collision energy"
                );
            }
        }
        assert!(n_ms1 > 0 && n_ms2 > 0, "expected both MS1 and MS2 scans");
        eprintln!(
            "thermo_frame_schedule OK: {} scans ({} MS1, {} MS2), RT {:.2}..{:.2}s",
            sched.len(),
            n_ms1,
            n_ms2,
            sched.first().unwrap().retention_time_s,
            last_rt
        );
    }

    // A DDA template must be REJECTED — its scan events now decode (variable-length
    // support), but it has a unique precursor per MS2 (no DIA tiling), so the
    // DIA-suitability check rejects it. Gate on a real DDA .raw (e.g. Orbitrap Fusion):
    // `TIMSIM_DDA_TEMPLATE=<dda .raw>`.
    #[cfg(feature = "thermo")]
    #[test]
    fn thermo_frame_schedule_rejects_dda_template() {
        let template = match std::env::var("TIMSIM_DDA_TEMPLATE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP thermo_frame_schedule_rejects_dda_template: set TIMSIM_DDA_TEMPLATE=<dda .raw>");
                return;
            }
        };
        let r = AcquisitionScheme::thermo_frame_schedule(&template);
        let e = r
            .err()
            .expect("a DDA template must be rejected (no DIA tiling)");
        assert!(
            e.to_string().contains("looks like DDA"),
            "unexpected error: {e}"
        );
        eprintln!("thermo_frame_schedule correctly rejected DDA template: {e}");
    }

    // A variable-length-event DIA template (Orbitrap Fusion-class) must be ACCEPTED and
    // decode real ms-level + isolation windows. Gate: `TIMSIM_VARLEN_DIA_TEMPLATE=<fusion dia .raw>`.
    #[cfg(feature = "thermo")]
    #[test]
    fn thermo_frame_schedule_accepts_variable_length_dia() {
        let template = match std::env::var("TIMSIM_VARLEN_DIA_TEMPLATE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP thermo_frame_schedule_accepts_variable_length_dia: set TIMSIM_VARLEN_DIA_TEMPLATE=<fusion dia .raw>");
                return;
            }
        };
        let sched = AcquisitionScheme::thermo_frame_schedule(&template)
            .expect("variable-length DIA template must be accepted");
        let n_ms1 = sched.iter().filter(|s| s.ms_level <= 1).count();
        let n_ms2_iso = sched
            .iter()
            .filter(|s| {
                s.ms_level > 1
                    && s.isolation
                        .is_some_and(|w| w.center_mz > 0.0 && w.width_mz > 0.0)
            })
            .count();
        assert!(n_ms1 > 0, "expected decoded MS1 scans");
        assert!(
            n_ms2_iso > 0,
            "expected MS2 scans with decoded isolation windows"
        );
        eprintln!("variable-length DIA accepted: {n_ms1} MS1, {n_ms2_iso} MS2 with isolation");
    }
}
