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
pub use thermo::{ThermoRawWriter, WriteMode};

#[cfg(feature = "thermo")]
mod thermo {
    use super::*;
    use std::path::{Path, PathBuf};
    use thermorawfile::{Calibration, RawFile};

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
        slots: Vec<(u32, bool)>,
        cursor: usize,
        mode: WriteMode,
        /// Opt out of the zero-residual completeness check at `finalize` (e.g. a
        /// smoke test authoring only a few of the template's slots). Off by default
        /// so a real Replace run that does not fill every slot fails loudly rather
        /// than saving a file with untouched template signal in the gaps.
        allow_partial: bool,
    }

    /// Whether a descriptor's ms-level fits the next template slot type (MS1 ⇒
    /// profile slot, MS2+ ⇒ centroid slot). Free fn so the ordering contract is
    /// unit-testable without a template.
    pub(crate) fn slot_level_matches(is_profile: bool, ms_level: u8) -> bool {
        (ms_level <= 1) == is_profile
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
                slots.push((scan, profile_size > 0));
            }
            Ok(Self {
                raw,
                out_path: out.as_ref().to_path_buf(),
                calib,
                slots,
                cursor: 0,
                mode: WriteMode::Replace,
                allow_partial: false,
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
            let ms1 = self.slots.iter().filter(|s| s.1).count();
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
        pub fn manifest(&self) -> &[(u32, bool)] {
            &self.slots
        }
    }

    impl AcquisitionWriter for ThermoRawWriter {
        fn write_scan(&mut self, scan: &ScanDescriptor) -> io::Result<()> {
            let (t, is_profile) = *self.slots.get(self.cursor).ok_or_else(|| {
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
            if !slot_level_matches(is_profile, scan.ms_level) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "scan-order mismatch at slot {} (template scan {}): template \
                         expects {} but got ms_level {}",
                        self.cursor,
                        t,
                        if is_profile { "MS1/profile" } else { "MS2/centroid" },
                        scan.ms_level
                    ),
                ));
            }
            if is_profile {
                // Advance the cursor only after authoring succeeds, so a failed
                // write doesn't permanently consume the slot.
                match self.mode {
                    WriteMode::Replace => self.raw.author_profile(t, &scan.peaks, &self.calib),
                    WriteMode::Overlay { .. } => {
                        self.raw.overlay_profile(t, &scan.peaks, &self.calib)
                    }
                }?;
            } else {
                match self.mode {
                    WriteMode::Replace => {
                        self.raw.author_centroids(t, &scan.peaks)?;
                        // Author the descriptor's isolation window + CE into the
                        // scan event so the output's MS2 metadata reflects the
                        // intended scheme, not just the template's.
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
            // Zero-residual contract: in Replace mode every template slot must have
            // been authored, else the saved file keeps the template's REAL signal in
            // the unconsumed slots (a partial/cancelled run masquerading as valid
            // output). Overlay mode intentionally retains template signal, so it is
            // exempt; `allow_partial` opts out for partial-write smoke tests.
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
        // MS1 (ms_level 0/1) belongs in a profile slot; MS2+ in a centroid slot.
        assert!(slot_level_matches(true, 1));
        assert!(slot_level_matches(true, 0));
        assert!(slot_level_matches(false, 2));
        assert!(slot_level_matches(false, 3));
        // Mismatches the single-cursor writer rejects.
        assert!(!slot_level_matches(true, 2)); // MS2 into a profile slot
        assert!(!slot_level_matches(false, 1)); // MS1 into a centroid slot
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
}
