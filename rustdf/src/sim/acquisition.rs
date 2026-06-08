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

/// A sink that writes vendor-neutral scans into a vendor raw file. Scans are
/// written in acquisition order; `finalize` flushes the file (and fixes any
/// integrity checksum).
pub trait AcquisitionWriter {
    fn write_scan(&mut self, scan: &ScanDescriptor) -> io::Result<()>;
    fn finalize(&mut self) -> io::Result<()>;
}

#[cfg(feature = "thermo")]
pub use thermo::ThermoRawWriter;

#[cfg(feature = "thermo")]
mod thermo {
    use super::*;
    use std::path::{Path, PathBuf};
    use thermorawfile::{Calibration, RawFile};

    /// Authors scans into a Thermo `.raw` by **template-mutation**: open a real
    /// `.raw`, overwrite template scans of matching type in acquisition order
    /// (MS1 → profile via `author_profile`, MS2 → centroids via
    /// `author_centroids`), and save on [`AcquisitionWriter::finalize`].
    ///
    /// The template supplies the frequency grid + calibration; synthetic peaks
    /// are placed at their exact m/z within each packet's byte budget. The
    /// calibration is read once from the first scan (it is ≈constant across a
    /// run); each scan's own frequency grid is taken from its profile.
    pub struct ThermoRawWriter {
        raw: RawFile,
        out_path: PathBuf,
        calib: Calibration,
        /// Template scans that carry a profile (treated as MS1 targets).
        profile_scans: Vec<u32>,
        /// Centroid-only template scans (treated as MS2 targets).
        centroid_scans: Vec<u32>,
        prof_cur: usize,
        cent_cur: usize,
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

            // Classify template scans by packet type: a non-zero profile section
            // marks an MS1-style (FTMS profile) scan; centroid-only is MS2 (ASTMS).
            let mut profile_scans = Vec::new();
            let mut centroid_scans = Vec::new();
            for (i, e) in raw.index.iter().enumerate() {
                let pkt = (raw.data_addr + e.offset) as usize;
                if pkt + 8 > raw.bytes.len() {
                    continue;
                }
                let profile_size =
                    u32::from_le_bytes(raw.bytes[pkt + 4..pkt + 8].try_into().unwrap());
                let scan = raw.first_scan + i as u32;
                if profile_size > 0 {
                    profile_scans.push(scan);
                } else {
                    centroid_scans.push(scan);
                }
            }
            Ok(Self {
                raw,
                out_path: out.as_ref().to_path_buf(),
                calib,
                profile_scans,
                centroid_scans,
                prof_cur: 0,
                cent_cur: 0,
            })
        }

        /// Template MS1/MS2 capacity, so callers can check a run fits.
        pub fn capacity(&self) -> (usize, usize) {
            (self.profile_scans.len(), self.centroid_scans.len())
        }
    }

    impl AcquisitionWriter for ThermoRawWriter {
        fn write_scan(&mut self, scan: &ScanDescriptor) -> io::Result<()> {
            if scan.ms_level <= 1 {
                let t = *self.profile_scans.get(self.prof_cur).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "template exhausted of MS1 (profile) scans",
                    )
                })?;
                self.prof_cur += 1;
                self.raw.author_profile(t, &scan.peaks, &self.calib)
            } else {
                let t = *self.centroid_scans.get(self.cent_cur).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "template exhausted of MS2 (centroid) scans",
                    )
                })?;
                self.cent_cur += 1;
                self.raw.author_centroids(t, &scan.peaks)
            }
        }

        fn finalize(&mut self) -> io::Result<()> {
            let path = self.out_path.clone();
            self.raw.save(path)
        }
    }
}

#[cfg(all(test, feature = "thermo"))]
mod tests {
    use super::*;

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

        let mut w = ThermoRawWriter::from_template(&template, &out).expect("open template");
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

        eprintln!("thermo_roundtrip OK: MS1 {:?}, MS2 {} peaks", ms1_mz, cents.len());
    }
}
