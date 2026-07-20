//! Vendor-neutral acquisition-writer abstraction.
//!
//! [`AcquisitionWriter`] turns vendor-neutral [`ScanDescriptor`]s into a vendor
//! raw file. The first implementation is [`ThermoRawWriter`] (Thermo `.raw`, via
//! the `thermo` feature + the `thermorawfile` crate); a Bruker `.d` adapter over
//! the existing write path is a planned follow-up.

use std::io;

/// MS2 quadrupole isolation window.
#[derive(Clone, Copy, Debug)]
pub struct IsolationWindow {
    pub center_mz: f64,
    pub width_mz: f64,
    pub collision_energy: f64,
}

/// One vendor-neutral scan: a peak list plus acquisition metadata.
///
/// Intentionally has **no ion-mobility dimension** — it is the
/// Bruker∩Thermo∩SCIEX common denominator. IMS-bearing Bruker frames are handled
/// by the Bruker writer, which can flatten or carry mobility separately.
#[derive(Clone, Debug)]
pub struct ScanDescriptor {
    pub ms_level: u8,
    pub retention_time: f64,
    /// Present for MS2 (the precursor isolation window).
    pub isolation: Option<IsolationWindow>,
    /// `(m/z, intensity)` pairs; need not be sorted (writers sort).
    pub peaks: Vec<(f64, f32)>,
}

impl ScanDescriptor {
    /// Convert a rendered event into a writer-ready descriptor (P6e-2 glue).
    ///
    /// `collision_energy` is the window's applied CE (eV/NCE) — it is NOT carried
    /// on [`crate::sim::projector::RenderedEvent`]'s isolation (which is m/z-only),
    /// so the driver supplies it. A [`crate::sim::projector::RenderedEvent::MobilityFrame`]
    /// is not writable to a scan-based (Thermo) file. Peak m/z and intensity are
    /// validated finite and the intensity is range-checked into `f32` (claudex):
    /// a non-finite or `f32`-overflowing value is a hard error rather than a NaN/inf
    /// silently authored into the packet.
    pub fn from_rendered_event(
        ev: &crate::sim::projector::RenderedEvent,
        collision_energy: f64,
    ) -> Result<ScanDescriptor, String> {
        use crate::sim::projector::RenderedEvent;
        let (ms_level, retention_time, isolation, spectrum) = match ev {
            RenderedEvent::Scan { ms_level, retention_time_s, isolation, spectrum } => {
                let iso = isolation.map(|w| IsolationWindow {
                    center_mz: w.center_mz,
                    width_mz: w.width_mz,
                    collision_energy,
                });
                (*ms_level, *retention_time_s, iso, spectrum)
            }
            RenderedEvent::MobilityFrame { .. } => {
                return Err("a MobilityFrame cannot be written to a scan-based (Thermo) file; \
                            it is an IMS instrument's output"
                    .to_string());
            }
        };
        if spectrum.mz.len() != spectrum.intensity.len() {
            return Err(format!(
                "rendered spectrum m/z ({}) and intensity ({}) length mismatch",
                spectrum.mz.len(),
                spectrum.intensity.len()
            ));
        }
        let mut peaks = Vec::with_capacity(spectrum.mz.len());
        for (&m, &i) in spectrum.mz.iter().zip(spectrum.intensity.iter()) {
            if !m.is_finite() {
                return Err(format!("non-finite peak m/z {m}"));
            }
            if !i.is_finite() {
                return Err(format!("non-finite peak intensity {i} at m/z {m}"));
            }
            let i32 = i as f32;
            if !i32.is_finite() {
                return Err(format!("peak intensity {i} at m/z {m} overflows f32"));
            }
            peaks.push((m, i32));
        }
        Ok(ScanDescriptor { ms_level, retention_time, isolation, peaks })
    }
}

/// A sink that writes vendor-neutral scans into a vendor raw file. Scans are
/// written in acquisition order; `finalize` flushes the file (and fixes any
/// integrity checksum).
pub trait AcquisitionWriter {
    fn write_scan(&mut self, scan: &ScanDescriptor) -> io::Result<()>;
    fn finalize(&mut self) -> io::Result<()>;
}

#[cfg(feature = "thermo")]
pub use thermo::{rewindow_thermo_template, ThermoRawWriter, WriteMode};

#[cfg(feature = "thermo")]
mod thermo {
    use super::*;
    use std::path::{Path, PathBuf};
    use thermorawfile::{Calibration, ProfileWriteResult, RawFile};

    /// Re-window a Thermo DIA template (Tier-2 3a): copy `src` to `dst` with every MS2
    /// isolation window set to `isolation_width` (centers + CE kept — same cardinality).
    /// Used as a pre-authoring step so a sim can request a custom DIA window width without
    /// a matching real template. Returns the number of MS2 scans re-windowed.
    ///
    /// The new windows must then drive BOTH the schedule read and authoring — so callers
    /// point `template_path` at `dst` for the rest of the run, keeping the acquisition DB
    /// and the authored `.raw` consistent.
    pub fn rewindow_thermo_template<P: AsRef<Path>, Q: AsRef<Path>>(
        src: P,
        dst: Q,
        isolation_width: f64,
    ) -> io::Result<usize> {
        if !(isolation_width.is_finite() && isolation_width > 0.0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("rewindow isolation_width must be finite and > 0, got {isolation_width}"),
            ));
        }
        let mut raw = RawFile::open(src)?;
        if !raw.has_scan_events() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "template scan events could not be decoded; cannot re-window",
            ));
        }
        let n = raw.rewindow_in_place(|_scan, ev| {
            Some((ev.isolation_center, isolation_width, ev.collision_energy))
        })?;
        // Atomic publish: write to a sibling temp then rename, so a partial/failed write
        // can never leave a corrupt file at `dst` that a later run might trust.
        let dst = dst.as_ref();
        let tmp = dst.with_extension("raw.rewindow.tmp");
        raw.save(&tmp)?;
        std::fs::rename(&tmp, dst)?;
        Ok(n)
    }

    /// How `ThermoRawWriter` combines simulated peaks with the template scan.
    #[derive(Clone, Copy, Debug)]
    pub enum WriteMode {
        /// Replace the template scan's signal with the simulated peaks.
        Replace,
        /// Add simulated peaks onto the template's real signal (real⊕sim);
        /// `merge_tol_ppm` merges near-coincident centroids on MS2.
        Overlay { merge_tol_ppm: f64 },
    }

    /// Authors scans into a Thermo `.raw` by **template-mutation**: open a real
    /// `.raw`, overwrite template scans of matching type in acquisition order
    /// (MS1 → profile via `author_profile`, MS2 → centroids via
    /// `author_centroids`), and save on [`AcquisitionWriter::finalize`].
    ///
    /// The template supplies the frequency grid + calibration; synthetic peaks
    /// are placed at their exact m/z within each packet's byte budget. The
    /// calibration is read once from the first scan (it is ≈constant across a
    /// run); each scan's own frequency grid is taken from its profile. For MS2,
    /// a descriptor's [`IsolationWindow`] (center/width/CE) is authored into the
    /// scan event via `set_isolation`, so the output's MS2 metadata reflects the
    /// intended scheme rather than only the template's.
    ///
    /// Current limitation: this *replaces* the template scan's signal. A future
    /// **overlay** mode (keep the real template signal and add simulated peaks)
    /// would support the "reference run = layout + real noise" real⊕sim workflow.
    pub struct ThermoRawWriter {
        raw: RawFile,
        out_path: PathBuf,
        calib: Calibration,
        /// Template slots in ACQUISITION (scan) order: `(scan_number, is_profile)`,
        /// where `is_profile` marks an MS1 FTMS-profile slot (vs an MS2 centroid
        /// slot). A SINGLE cursor walks this in order, so the caller must feed scans
        /// in the template's schedule — a level mismatch is rejected (independent
        /// MS1/MS2 cursors could not catch a reordered stream).
        slots: Vec<(u32, u8, bool)>,   // (scan, ms_level, is_profile): level & encoding decoupled
        cursor: usize,
        mode: WriteMode,
        /// Opt out of the zero-residual completeness check at `finalize` (e.g. a
        /// smoke test authoring only a few of the template's slots). Off by default
        /// so a real Replace run that does not fill every slot fails loudly rather
        /// than saving a file with untouched template signal in the gaps.
        allow_partial: bool,
        /// Over-budget scans deferred during `write_scan` and applied in a SINGLE
        /// `repack_many` rebuild at `finalize` — so growing many scans past the
        /// template budget costs one data-section rebuild, not one splice per scan.
        deferred: Vec<DeferredEdit>,
        /// Run-level tally of MS1 profile authoring drops (peaks outside a scan's mass range), so the
        /// driver can monitor lost ion current — a peak count can look harmless while a dominant
        /// precursor envelope was silently dropped.
        profile_summary: ProfileWriteResult,
    }

    /// A scan whose authored payload overflowed its template packet, deferred for the
    /// batch rebuild at finalize.
    struct DeferredEdit {
        scan: u32,
        peaks: Vec<(f64, f32)>,
        is_profile: bool,
    }

    /// Whether a descriptor's ms-level fits the next template slot type (MS1 ⇒
    /// profile slot, MS2+ ⇒ centroid slot). Free fn so the ordering contract is
    /// unit-testable without a template.
    pub(crate) fn slot_level_matches(slot_ms_level: u8, scan_ms_level: u8) -> bool {
        slot_ms_level == scan_ms_level
    }

    /// True iff `e` is the thermorawfile over-budget overflow — the one error the writer
    /// recovers from (by repacking to grow the packet) rather than propagating. Delegates
    /// to the crate's typed predicate so it can't drift from the error's wording.
    fn is_over_budget(e: &io::Error) -> bool {
        thermorawfile::is_over_budget(e)
    }

    impl ThermoRawWriter {
        /// Open `template` and prepare to author into `out`.
        pub fn from_template<P: AsRef<Path>, Q: AsRef<Path>>(
            template: P,
            out: Q,
        ) -> io::Result<Self> {
            let raw = RawFile::open(template)?;
            let calib = raw
                .calibration_at_event(raw.scantrailer_addr as usize + 4)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "no MS1 calibration in template")
                })?;

            // Build the ordered slot manifest: a non-zero profile section marks an
            // MS1-style (FTMS profile) slot; centroid-only is MS2 (ASTMS). Kept in
            // scan order so a single cursor enforces the template's schedule.
            let mut slots = Vec::with_capacity(raw.index.len());
            for (i, e) in raw.index.iter().enumerate() {
                let pkt = (raw.data_addr + e.offset) as usize;
                if pkt + 8 > raw.bytes.len() {
                    continue;
                }
                let profile_size =
                    u32::from_le_bytes(raw.bytes[pkt + 4..pkt + 8].try_into().unwrap());
                let scan = raw.first_scan + i as u32;
                let is_profile = profile_size > 0;
                // Real ms-level from the scan event, decoupled from profile/centroid (a Q Exactive
                // HF-X DIA template has profile MS2; profile⇒MS1 is only an Astral coincidence).
                let ms_level = raw
                    .scan_event(scan)
                    .map(|e| e.ms_order)
                    .unwrap_or(if is_profile { 1 } else { 2 });
                // GUARD (codex review): decoupling must not silently enable centroid-MS1 overlay,
                // whose isotope-envelope/calibration semantics differ. Keep profile-MS1 explicit.
                if ms_level <= 1 && !is_profile {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "centroid-MS1 template not supported for overlay (scan {}); \
                             profile-MS1 required",
                            scan
                        ),
                    ));
                }
                slots.push((scan, ms_level, is_profile));
            }
            Ok(Self {
                raw,
                out_path: out.as_ref().to_path_buf(),
                calib,
                slots,
                cursor: 0,
                profile_summary: ProfileWriteResult::default(),
                mode: WriteMode::Replace,
                allow_partial: false,
                deferred: Vec::new(),
            })
        }

        /// Set the write mode (default [`WriteMode::Replace`]). Use
        /// [`WriteMode::Overlay`] for the reference-run = layout + real-noise
        /// (real⊕sim) workflow.
        pub fn with_mode(mut self, mode: WriteMode) -> Self {
            self.mode = mode;
            self
        }

        /// Permit `finalize` to save even when not every template slot was authored
        /// (default: off). Only for partial-write smoke tests; a real Replace run
        /// must fill every slot to honour the zero-residual contract.
        pub fn with_allow_partial(mut self, allow: bool) -> Self {
            self.allow_partial = allow;
            self
        }

        /// Template MS1/MS2 capacity, so callers can check a run fits.
        pub fn capacity(&self) -> (usize, usize) {
            let ms1 = self.slots.iter().filter(|s| s.1 <= 1).count();
            (ms1, self.slots.len() - ms1)
        }

        /// Total number of template slots (the exact scan count a complete run
        /// must author, in order).
        pub fn slot_count(&self) -> usize {
            self.slots.len()
        }

        /// Number of slots authored so far.
        pub fn position(&self) -> usize {
            self.cursor
        }

        /// Whether every template slot has been authored (the zero-residual
        /// contract: a `Replace`-mode run that does not fill the template leaves
        /// real template signal in the unconsumed slots).
        pub fn is_complete(&self) -> bool {
            self.cursor == self.slots.len()
        }

        /// The ordered slot manifest `(scan_number, is_profile)` — for preflight
        /// (matching the run's MS-level sequence to the template before writing).
        pub fn manifest(&self) -> &[(u32, u8, bool)] {
            &self.slots
        }

        /// Run-level tally of MS1 profile authoring: bins written and peaks dropped (with their ion
        /// current) because they fell outside a scan's mass range. The driver should surface
        /// `dropped_intensity` as a lost-signal warning.
        pub fn profile_summary(&self) -> ProfileWriteResult {
            self.profile_summary
        }

        /// Per-slot acquisition schedule in manifest order: `(retention_time_s, isolation)`. The
        /// template IS the schedule — a render driver walks this to place elution over the run's real
        /// retention times and to know each MS2 slot's isolation window (the DIA scheme is inherited).
        /// `isolation` is `Some` only for MS2+ slots.
        pub fn schedule(&self) -> Vec<(f64, Option<IsolationWindow>)> {
            self.slots
                .iter()
                .map(|&(scan, ms_level, _)| {
                    let rt = self.raw.index[(scan - self.raw.first_scan) as usize].time;
                    let iso = if ms_level >= 2 {
                        self.raw.scan_event(scan).map(|e| IsolationWindow {
                            center_mz: e.isolation_center,
                            width_mz: e.isolation_width,
                            collision_energy: e.collision_energy,
                        })
                    } else {
                        None
                    };
                    (rt, iso)
                })
                .collect()
        }
    }

    impl AcquisitionWriter for ThermoRawWriter {
        fn write_scan(&mut self, scan: &ScanDescriptor) -> io::Result<()> {
            let (t, slot_ms_level, is_profile) = *self.slots.get(self.cursor).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "template exhausted: {} slots, tried to author slot {}",
                        self.slots.len(),
                        self.cursor + 1
                    ),
                )
            })?;
            // Enforce the template's GLOBAL scan order: the descriptor's level must
            // match the next slot's type. A single cursor catches a reordered stream
            // (MS1,MS2,MS1,MS2 vs MS1,MS1,MS2,MS2) that independent cursors could not.
            if !slot_level_matches(slot_ms_level, scan.ms_level) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "scan-order mismatch at slot {} (template scan {}): template slot \
                         ms_level {} (is_profile {}) but got ms_level {}",
                        self.cursor,
                        t,
                        slot_ms_level,
                        is_profile,
                        scan.ms_level
                    ),
                ));
            }
            // Overlay with no simulated peaks is a true no-op: leave the template
            // slot's REAL signal byte-identical. Don't round-trip it through
            // overlay_profile/overlay_centroids, which re-canonicalize the packet
            // (and could merge near-coincident real centroids within merge_tol_ppm).
            // Replace with no peaks still clears the slot (pure-simulated, zero-
            // residual) — that path falls through below.
            if scan.peaks.is_empty() {
                if let WriteMode::Overlay { .. } = self.mode {
                    self.cursor += 1;
                    return Ok(());
                }
            }
            if is_profile {
                // Advance the cursor only after authoring succeeds, so a failed
                // write doesn't permanently consume the slot.
                match self.mode {
                    WriteMode::Replace => {
                        // Fast path: author in place. On (and only on) an over-budget
                        // overflow, DEFER the scan — it can hold MORE peaks than the
                        // template slot, but growing it relocates the file tail, so we
                        // batch all overflows into one `repack_many` rebuild at finalize
                        // instead of a splice per scan. The common in-budget write stays
                        // in place; nothing is cleared (the old lossy behaviour).
                        match self.raw.author_profile(t, &scan.peaks, &self.calib) {
                            // Out-of-range peaks are dropped-and-accounted (not an error); tally them
                            // run-wide so the driver can watch lost ion current.
                            Ok(res) => self.profile_summary.accumulate(&res),
                            Err(e) if is_over_budget(&e) => self.deferred.push(DeferredEdit {
                                scan: t,
                                peaks: scan.peaks.clone(),
                                is_profile: true,
                            }),
                            Err(e) => return Err(e),
                        }
                    }
                    WriteMode::Overlay { .. } => {
                        self.raw.overlay_profile(t, &scan.peaks, &self.calib)?
                    }
                }
            } else {
                match self.mode {
                    WriteMode::Replace => {
                        match self.raw.author_centroids(t, &scan.peaks) {
                            Ok(()) => {}
                            Err(e) if is_over_budget(&e) => self.deferred.push(DeferredEdit {
                                scan: t,
                                peaks: scan.peaks.clone(),
                                is_profile: false,
                            }),
                            Err(e) => return Err(e),
                        }
                        // Author the descriptor's isolation window + CE into the scan
                        // event. Independent of the data-packet size, and the scan-event
                        // bytes are preserved (relocated) by a later batch rebuild, so it
                        // is correct to set here for both in-place and deferred scans.
                        if let Some(iso) = scan.isolation {
                            self.raw.set_isolation(
                                t,
                                iso.center_mz,
                                iso.width_mz,
                                iso.collision_energy,
                            )?;
                        }
                    }
                    // Overlay keeps the template's real signal + acquisition
                    // metadata (the scheme is derived from this template), so the
                    // isolation/CE are left as the template's.
                    WriteMode::Overlay { merge_tol_ppm } => {
                        self.raw.overlay_centroids(t, &scan.peaks, merge_tol_ppm)?
                    }
                }
            }
            self.cursor += 1;
            Ok(())
        }

        fn finalize(&mut self) -> io::Result<()> {
            // Apply all deferred over-budget scans FIRST, in a SINGLE data-section
            // rebuild (O(file size), vs one splice per scan). The in-place author_*
            // writes have already landed; this grows only the scans that overflowed
            // their slot. Done before the residual check so the file is fully authored
            // by the time any validation runs (the check below is cursor-based today,
            // but applying first keeps it correct even if it ever inspects bytes).
            //
            // Note: each deferred scan's peaks are held cloned until here, and
            // repack_many transiently allocates a second copy of the data section. For a
            // run where MOST scans overflow on a multi-GB template, peak memory is
            // roughly file + rebuilt-data-section + deferred payloads; acceptable for now
            // but the ceiling to watch if this is pushed to huge inputs.
            if !self.deferred.is_empty() {
                let edits: Vec<thermorawfile::ScanEdit> = self
                    .deferred
                    .iter()
                    .map(|d| {
                        if d.is_profile {
                            thermorawfile::ScanEdit::Profile {
                                scan: d.scan,
                                peaks: &d.peaks,
                                calib: &self.calib,
                            }
                        } else {
                            thermorawfile::ScanEdit::Centroids { scan: d.scan, peaks: &d.peaks }
                        }
                    })
                    .collect();
                self.raw.repack_many(&edits)?;
            }

            // Zero-residual contract: in Replace mode every template slot must have
            // been authored (deferred overflow scans counted — their cursor advanced and
            // they were just rebuilt above), else the saved file keeps the template's
            // REAL signal in the unconsumed slots (a partial/cancelled run masquerading
            // as valid output). Overlay mode intentionally retains template signal, so it
            // is exempt; `allow_partial` opts out for partial-write smoke tests.
            if matches!(self.mode, WriteMode::Replace) && !self.allow_partial && !self.is_complete() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "incomplete Replace run: {}/{} template slots authored — the \
                         remaining {} slots still hold the template's real signal. \
                         Author every slot, or use with_allow_partial(true).",
                        self.cursor,
                        self.slots.len(),
                        self.slots.len() - self.cursor
                    ),
                ));
            }
            let path = self.out_path.clone();
            self.raw.save(path)
        }
    }
}

#[cfg(test)]
mod glue_tests {
    use super::*;
    use crate::sim::projector::{IntensityStage, MzCoordSpace, RenderedEvent, RenderedSpectrum};
    use crate::sim::scheme::{DataMode, IsolationWindow as SchemeIso};

    fn ms2_event(mz: Vec<f64>, intensity: Vec<f64>) -> RenderedEvent {
        RenderedEvent::Scan {
            ms_level: 2,
            retention_time_s: 12.5,
            isolation: Some(SchemeIso { center_mz: 600.0, width_mz: 25.0 }),
            spectrum: RenderedSpectrum {
                mz,
                intensity,
                coords: MzCoordSpace::Physical,
                mode: DataMode::Centroid,
                detector_applied: false,
                stage: IntensityStage::Transmitted,
            },
        }
    }

    #[test]
    fn from_rendered_event_maps_and_attaches_ce() {
        let d = ScanDescriptor::from_rendered_event(&ms2_event(vec![200.0, 300.0], vec![10.0, 20.0]), 27.0)
            .expect("valid");
        assert_eq!(d.ms_level, 2);
        assert!((d.retention_time - 12.5).abs() < 1e-9);
        let iso = d.isolation.expect("MS2 isolation");
        assert!((iso.center_mz - 600.0).abs() < 1e-9 && (iso.collision_energy - 27.0).abs() < 1e-9);
        assert_eq!(d.peaks, vec![(200.0, 10.0f32), (300.0, 20.0f32)]);
    }

    #[test]
    fn from_rendered_event_rejects_nonfinite_and_mobility() {
        // Non-finite intensity is a hard error (not a NaN authored into the packet).
        assert!(ScanDescriptor::from_rendered_event(
            &ms2_event(vec![200.0], vec![f64::NAN]), 27.0
        ).is_err());
        // Non-finite m/z too.
        assert!(ScanDescriptor::from_rendered_event(
            &ms2_event(vec![f64::INFINITY], vec![1.0]), 27.0
        ).is_err());
        // A MobilityFrame is not writable to a scan-based file.
        let mob = RenderedEvent::MobilityFrame { ms_level: 1, retention_time_s: 0.0, scans: vec![] };
        assert!(ScanDescriptor::from_rendered_event(&mob, 0.0).is_err());
    }
}

#[cfg(all(test, feature = "thermo"))]
mod tests {
    use super::*;

    #[test]
    fn slot_level_matches_enforces_ordering_contract() {
        use super::thermo::slot_level_matches;
        // slot_level_matches now compares ACTUAL ms-levels (template slot vs simulated scan),
        // decoupled from profile/centroid encoding.
        assert!(slot_level_matches(1, 1));
        assert!(slot_level_matches(2, 2));
        // Mismatches the single-cursor writer rejects (reordered MS1/MS2 stream).
        assert!(!slot_level_matches(1, 2)); // MS2 scan into an MS1 slot
        assert!(!slot_level_matches(2, 1)); // MS1 scan into an MS2 slot
    }

    // Gated: set TIMSIM_ASTRAL_TEMPLATE to a real Orbitrap Astral .raw to run.
    // `cargo test --features thermo -- --nocapture thermo_roundtrip`
    #[test]
    fn thermo_roundtrip() {
        let template = match std::env::var("TIMSIM_ASTRAL_TEMPLATE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP thermo_roundtrip: set TIMSIM_ASTRAL_TEMPLATE=<astral .raw>");
                return;
            }
        };
        let out = std::env::temp_dir().join("rustdf_thermo_roundtrip.raw");

        // Authors only 2 of the template's many slots, so opt into partial finalize.
        let mut w = ThermoRawWriter::from_template(&template, &out)
            .expect("open template")
            .with_allow_partial(true);
        let (n_ms1, n_ms2) = w.capacity();
        assert!(n_ms1 > 0 && n_ms2 > 0, "template has no MS1/MS2 scans");

        let ms1 = ScanDescriptor {
            ms_level: 1,
            retention_time: 0.0,
            isolation: None,
            peaks: vec![(500.0, 1.0e6), (700.0, 5.0e5)],
        };
        let ms2 = ScanDescriptor {
            ms_level: 2,
            retention_time: 0.01,
            isolation: Some(IsolationWindow {
                center_mz: 500.0,
                width_mz: 2.0,
                collision_energy: 25.0,
            }),
            peaks: vec![(150.1, 3.0e4), (420.2, 8.0e4), (610.3, 5.0e4)],
        };
        w.write_scan(&ms1).expect("write MS1");
        w.write_scan(&ms2).expect("write MS2");
        w.finalize().expect("finalize");

        // Read back through thermorawfile and confirm the authored peaks.
        let rf = thermorawfile::RawFile::open(&out).expect("reopen");
        assert!(rf.checksum_valid(), "checksum invalid");

        // The writer used the first profile scan for MS1 and first centroid scan for MS2.
        let cal = rf
            .calibration_at_event(rf.scantrailer_addr as usize + 4)
            .unwrap();
        let mut prof_scan = None;
        let mut cent_scan = None;
        for i in 0..rf.index.len() {
            let scan = rf.first_scan + i as u32;
            let pkt = (rf.data_addr + rf.index[i].offset) as usize;
            let psize = u32::from_le_bytes(rf.bytes[pkt + 4..pkt + 8].try_into().unwrap());
            if psize > 0 && prof_scan.is_none() {
                prof_scan = Some(scan);
            }
            if psize == 0 && cent_scan.is_none() {
                cent_scan = Some(scan);
            }
        }
        let prof = rf.profile(prof_scan.unwrap()).expect("ms1 profile");
        assert_eq!(prof.chunks.len(), 2, "MS1 peak count");
        let ms1_mz: Vec<f64> = prof
            .chunks
            .iter()
            .map(|c| prof.mz_of_bin(c.first_bin, &cal))
            .collect();
        assert!((ms1_mz[0] - 500.0).abs() < 0.01 && (ms1_mz[1] - 700.0).abs() < 0.01);

        let cents = rf.centroid_peaks(cent_scan.unwrap());
        assert_eq!(cents.len(), 3, "MS2 peak count");
        assert!((cents[0].mz - 150.1).abs() < 0.01);
        assert!((cents[2].mz - 610.3).abs() < 0.01);

        // The authored isolation window + CE must be reflected in the scan event.
        let ev = rf.scan_event(cent_scan.unwrap()).expect("ms2 scan event");
        assert!((ev.isolation_center - 500.0).abs() < 0.01, "authored isolation center");
        assert!((ev.isolation_width - 2.0).abs() < 0.01, "authored isolation width");
        assert!((ev.collision_energy - 25.0).abs() < 0.1, "authored CE");

        eprintln!(
            "thermo_roundtrip OK: MS1 {:?}, MS2 {} peaks, MS2 iso {:.2}±{:.2} CE {:.1}",
            ms1_mz, cents.len(), ev.isolation_center, ev.isolation_width, ev.collision_energy
        );
    }

    // Gated: an over-budget MS2 must GROW via the repack fallback, not be cleared.
    #[test]
    fn thermo_overflow_repack() {
        let template = match std::env::var("TIMSIM_ASTRAL_TEMPLATE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP thermo_overflow_repack: set TIMSIM_ASTRAL_TEMPLATE=<astral .raw>");
                return;
            }
        };
        let out = std::env::temp_dir().join("rustdf_thermo_overflow.raw");
        let mut w = ThermoRawWriter::from_template(&template, &out)
            .expect("open template")
            .with_allow_partial(true);

        let ms1 = ScanDescriptor {
            ms_level: 1,
            retention_time: 0.0,
            isolation: None,
            peaks: vec![(500.0, 1.0e6), (700.0, 5.0e5)],
        };
        // Far more peaks than any realistic centroid packet budget → forces the repack.
        let big: Vec<(f64, f32)> = (0..6000)
            .map(|i| (200.0 + i as f64 * 0.1, 100.0 + i as f32))
            .collect();
        let ms2 = ScanDescriptor {
            ms_level: 2,
            retention_time: 0.01,
            isolation: Some(IsolationWindow {
                center_mz: 500.0,
                width_mz: 2.0,
                collision_energy: 25.0,
            }),
            peaks: big.clone(),
        };

        w.write_scan(&ms1).expect("write MS1");
        // The assertion that matters: an over-budget MS2 now SUCCEEDS (grows via repack)
        // instead of erroring / being cleared to empty.
        w.write_scan(&ms2)
            .expect("over-budget MS2 must grow via the repack fallback");
        w.finalize().expect("finalize");

        let rf = thermorawfile::RawFile::open(&out).expect("reopen");
        assert!(rf.checksum_valid(), "checksum after in-sim repack");
        let mut cent = None;
        for i in 0..rf.index.len() {
            let scan = rf.first_scan + i as u32;
            let pkt = (rf.data_addr + rf.index[i].offset) as usize;
            let psize = u32::from_le_bytes(rf.bytes[pkt + 4..pkt + 8].try_into().unwrap());
            if psize == 0 {
                cent = Some(scan);
                break;
            }
        }
        let n = rf.centroid_peaks(cent.unwrap()).len();
        assert_eq!(n, big.len(), "repacked MS2 must keep ALL peaks, not be cleared");
        eprintln!("thermo_overflow_repack OK: MS2 grown to {n} peaks via the repack fallback");
    }

    // Gated: MANY over-budget MS2 scans must all grow via ONE batch rebuild at finalize.
    #[test]
    fn thermo_overflow_batch() {
        let template = match std::env::var("TIMSIM_ASTRAL_TEMPLATE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP thermo_overflow_batch: set TIMSIM_ASTRAL_TEMPLATE=<astral .raw>");
                return;
            }
        };
        let out = std::env::temp_dir().join("rustdf_thermo_overflow_batch.raw");
        let mut w = ThermoRawWriter::from_template(&template, &out)
            .expect("open template")
            .with_allow_partial(true);

        // One MS1, then several over-budget MS2 (the Astral schedule has many MS2 per
        // MS1, so consecutive MS2 slots follow the survey). Each is deferred and applied
        // in a single repack_many at finalize.
        w.write_scan(&ScanDescriptor {
            ms_level: 1,
            retention_time: 0.0,
            isolation: None,
            peaks: vec![(500.0, 1.0e6), (700.0, 5.0e5)],
        })
        .expect("write MS1");

        let sizes = [3000usize, 4000, 5000, 3500, 4500];
        for (k, &sz) in sizes.iter().enumerate() {
            let big: Vec<(f64, f32)> = (0..sz)
                .map(|i| (200.0 + i as f64 * 0.05, 100.0 + i as f32))
                .collect();
            w.write_scan(&ScanDescriptor {
                ms_level: 2,
                retention_time: 0.01 * (k + 1) as f64,
                isolation: Some(IsolationWindow {
                    center_mz: 400.0 + k as f64,
                    width_mz: 2.0,
                    collision_energy: 25.0,
                }),
                peaks: big,
            })
            .unwrap_or_else(|e| panic!("write over-budget MS2 #{k}: {e}"));
        }
        w.finalize().expect("finalize (batch rebuild)");

        let rf = thermorawfile::RawFile::open(&out).expect("reopen");
        assert!(rf.checksum_valid(), "checksum after batch rebuild");
        // The first 5 centroid scans (the authored MS2) must carry the grown counts.
        let cent_scans: Vec<u32> = (rf.first_scan..=rf.last_scan)
            .filter(|&s| {
                let i = (s - rf.first_scan) as usize;
                let pkt = (rf.data_addr + rf.index[i].offset) as usize;
                u32::from_le_bytes(rf.bytes[pkt + 4..pkt + 8].try_into().unwrap()) == 0
            })
            .take(sizes.len())
            .collect();
        for (k, &s) in cent_scans.iter().enumerate() {
            assert_eq!(rf.centroid_peaks(s).len(), sizes[k], "MS2 scan {s} grown count");
        }
        eprintln!("thermo_overflow_batch OK: {} MS2 grown via one repack_many {:?}", sizes.len(), sizes);
    }

    // Gated (Tier-2 3a): rewindow_thermo_template sets every MS2 window to a new width.
    // Gate on a DIA template: `TIMSIM_VARLEN_DIA_TEMPLATE=<dia .raw>`.
    #[test]
    fn rewindow_thermo_template_sets_width() {
        let src = match std::env::var("TIMSIM_VARLEN_DIA_TEMPLATE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP rewindow_thermo_template_sets_width: set TIMSIM_VARLEN_DIA_TEMPLATE=<dia .raw>");
                return;
            }
        };
        let dst = std::env::temp_dir().join("rustdf_rewindow_5th.raw");
        let n = super::rewindow_thermo_template(&src, &dst, 5.0).expect("rewindow");
        assert!(n > 0, "expected MS2 scans re-windowed");
        let rf = thermorawfile::RawFile::open(&dst).expect("reopen");
        assert!(rf.checksum_valid());
        let widths_ok = (rf.first_scan..=rf.last_scan)
            .filter_map(|s| rf.scan_event(s))
            .filter(|e| e.ms_order >= 2)
            .all(|e| (e.isolation_width - 5.0).abs() < 1e-6);
        assert!(widths_ok, "every MS2 window must now be 5.0 Th");
        // bad width rejected
        assert!(super::rewindow_thermo_template(&src, &dst, 0.0).is_err());
        eprintln!("rewindow_thermo_template OK: {n} MS2 windows -> 5.0 Th");
    }

    // Gated: overlay (real⊕sim) — sim peaks added onto the template's real signal.
    #[test]
    fn thermo_overlay() {
        let template = match std::env::var("TIMSIM_ASTRAL_TEMPLATE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP thermo_overlay: set TIMSIM_ASTRAL_TEMPLATE=<astral .raw>");
                return;
            }
        };
        let out = std::env::temp_dir().join("rustdf_thermo_overlay.raw");

        // Baseline counts from the untouched template.
        let base = thermorawfile::RawFile::open(&template).unwrap();
        let (mut prof_scan, mut cent_scan) = (None, None);
        for i in 0..base.index.len() {
            let scan = base.first_scan + i as u32;
            let pkt = (base.data_addr + base.index[i].offset) as usize;
            let psize = u32::from_le_bytes(base.bytes[pkt + 4..pkt + 8].try_into().unwrap());
            if psize > 0 && prof_scan.is_none() { prof_scan = Some(scan); }
            if psize == 0 && cent_scan.is_none() { cent_scan = Some(scan); }
        }
        let base_pts = base.profile(prof_scan.unwrap()).unwrap().point_count();
        let base_cents = base.centroid_peaks(cent_scan.unwrap()).len();

        let mut w = ThermoRawWriter::from_template(&template, &out)
            .expect("open")
            .with_mode(WriteMode::Overlay { merge_tol_ppm: 10.0 });
        w.write_scan(&ScanDescriptor {
            ms_level: 1, retention_time: 0.0, isolation: None,
            peaks: vec![(555.5, 9.9e5), (744.4, 7.7e5)],
        }).unwrap();
        w.write_scan(&ScanDescriptor {
            ms_level: 2, retention_time: 0.01,
            isolation: Some(IsolationWindow { center_mz: 490.0, width_mz: 2.0, collision_energy: 25.0 }),
            peaks: vec![(333.33, 5.0e5), (888.88, 4.0e5)],
        }).unwrap();
        w.finalize().unwrap();

        let rf = thermorawfile::RawFile::open(&out).unwrap();
        assert!(rf.checksum_valid());
        let cal = rf.calibration_at_event(rf.scantrailer_addr as usize + 4).unwrap();
        let prof = rf.profile(prof_scan.unwrap()).unwrap();
        // Real signal retained (point count grew, not replaced) + sim m/z present.
        assert!(prof.point_count() >= base_pts, "real profile points dropped");
        let has = |mz: f64| prof.chunks.iter().any(|c|
            (0..c.signal.len()).any(|j| (prof.mz_of_bin(c.first_bin + j as u32, &cal) - mz).abs() < 0.02));
        assert!(has(555.5) && has(744.4), "sim MS1 peaks missing");
        let cents = rf.centroid_peaks(cent_scan.unwrap());
        assert!(cents.len() >= base_cents + 1, "MS2 real centroids not retained / sim not added");
        eprintln!("thermo_overlay OK: MS1 {}->{} pts (+sim), MS2 {}->{} centroids",
            base_pts, prof.point_count(), base_cents, cents.len());
    }

    // Gated: an EMPTY overlay must be a true no-op — the template's real signal is
    // left untouched. The short-circuit returns before `self.raw` is mutated, so the
    // packet bytes are never round-tripped through overlay/author (which would
    // re-canonicalize and could merge near-coincident real centroids). We assert the
    // DECODED signal is bit-identical (exact equality: nothing is recomputed, so the
    // decode of an untouched packet is deterministic). This is the contract the
    // dispatch's overflow fallback relies on in Overlay mode (an overflowing slot is
    // re-authored empty and must keep its real signal).
    #[test]
    fn overlay_empty_is_noop() {
        let template = match std::env::var("TIMSIM_ASTRAL_TEMPLATE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP overlay_empty_is_noop: set TIMSIM_ASTRAL_TEMPLATE=<astral .raw>");
                return;
            }
        };
        let out = std::env::temp_dir().join("rustdf_overlay_empty_noop.raw");

        // Baseline: decoded MS1 profile + MS2 centroids of the first NON-EMPTY
        // profile/centroid scans (a vacuous empty scan would make the invariant
        // trivially hold), straight from the untouched template.
        let base = thermorawfile::RawFile::open(&template).unwrap();
        let (mut prof_scan, mut cent_scan) = (None, None);
        for i in 0..base.index.len() {
            let scan = base.first_scan + i as u32;
            let pkt = (base.data_addr + base.index[i].offset) as usize;
            let psize = u32::from_le_bytes(base.bytes[pkt + 4..pkt + 8].try_into().unwrap());
            if psize > 0 && prof_scan.is_none()
                && base.profile(scan).map(|p| p.point_count() > 0).unwrap_or(false)
            {
                prof_scan = Some(scan);
            }
            if psize == 0 && cent_scan.is_none() && !base.centroid_peaks(scan).is_empty() {
                cent_scan = Some(scan);
            }
            if prof_scan.is_some() && cent_scan.is_some() { break; }
        }
        let prof_scan = prof_scan.expect("template has no non-empty MS1 profile scan");
        let cent_scan = cent_scan.expect("template has no non-empty MS2 centroid scan");
        let cal = base.calibration_at_event(base.scantrailer_addr as usize + 4).unwrap();
        let profile_points = |rf: &thermorawfile::RawFile, scan: u32| -> Vec<(f64, f32)> {
            let p = rf.profile(scan).unwrap();
            let mut v = Vec::with_capacity(p.point_count());
            for c in &p.chunks {
                for j in 0..c.signal.len() {
                    v.push((p.mz_of_bin(c.first_bin + j as u32, &cal), c.signal[j]));
                }
            }
            v
        };
        let base_prof = profile_points(&base, prof_scan);
        let base_cents: Vec<(f64, f32)> =
            base.centroid_peaks(cent_scan).iter().map(|p| (p.mz, p.intensity)).collect();
        assert!(!base_prof.is_empty(), "picked an empty MS1 profile — test would be vacuous");
        assert!(!base_cents.is_empty(), "picked an empty MS2 centroid — test would be vacuous");

        // Walk every slot in order, authoring an EMPTY descriptor in Overlay mode.
        // A complete pass exercises the finalize path; Overlay is exempt from the
        // zero-residual check, and every empty author short-circuits.
        let mut w = ThermoRawWriter::from_template(&template, &out)
            .expect("open")
            .with_mode(WriteMode::Overlay { merge_tol_ppm: 20.0 });
        for &(_, _, is_profile) in w.manifest().to_vec().iter() {
            w.write_scan(&ScanDescriptor {
                ms_level: if is_profile { 1 } else { 2 },
                retention_time: 0.0,
                isolation: None,
                peaks: Vec::new(),
            })
            .expect("empty overlay author");
        }
        w.finalize().expect("finalize");

        let rf = thermorawfile::RawFile::open(&out).unwrap();
        assert!(rf.checksum_valid(), "checksum invalid");
        let out_prof = profile_points(&rf, prof_scan);
        let out_cents: Vec<(f64, f32)> =
            rf.centroid_peaks(cent_scan).iter().map(|p| (p.mz, p.intensity)).collect();
        // Real signal bit-identical — empty overlay touched nothing (exact equality:
        // the packet is never mutated, so the decode is deterministic).
        assert_eq!(out_prof, base_prof, "MS1 profile changed under empty overlay");
        assert_eq!(out_cents, base_cents, "MS2 centroids changed under empty overlay");
        eprintln!(
            "overlay_empty_is_noop OK: MS1 {} pts + MS2 {} centroids unchanged across full empty pass",
            base_prof.len(), base_cents.len()
        );
    }
}
