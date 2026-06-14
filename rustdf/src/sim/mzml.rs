//! Open **mzML** output via Joshua Klein's `mzdata` (`MzMLWriter`).
//!
//! The vendor-neutral output path: a stream of rendered scans (m/z + intensity, MS
//! level, retention time, optional MS2 isolation window + collision energy) is written
//! as standards-compliant mzML — readable by DiaNN/alphaDIA and signable by mzprov.
//! Used for SCIEX, whose proprietary `.wiff.scan` spectra are not authored; reusable as
//! a general open-format output for any instrument.

#![cfg(feature = "mzml")]

use std::fs::File;
use std::io;
use std::path::Path;

use mzdata::io::MzMLWriter;
use mzdata::prelude::*;
use mzdata::spectrum::bindata::{ArrayType, BinaryDataArrayType, DataArray};
use mzdata::spectrum::{
    Activation, BinaryArrayMap, IsolationWindow, Precursor, RawSpectrum, ScanEvent,
    ScanPolarity, SelectedIon, SignalContinuity, SpectrumDescription,
};

/// One MS2 isolation window + applied collision energy.
#[derive(Clone, Debug)]
pub struct MzMlIsolation {
    pub center_mz: f64,
    pub lower_mz: f64,
    pub upper_mz: f64,
    pub collision_energy: f64,
}

/// One vendor-neutral scan to serialise (centroided peak list).
#[derive(Clone, Debug)]
pub struct MzMlScan {
    pub ms_level: u8,
    pub retention_time_s: f64,
    pub mz: Vec<f64>,
    pub intensity: Vec<f32>,
    /// Present for MS2: the precursor isolation window + CE.
    pub isolation: Option<MzMlIsolation>,
}

fn f64_le_bytes(v: &[f64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 8);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// Write a stream of scans to `out_path` as mzML. Scans are written in order;
/// returns the number written. Spectra are centroid peak lists; MS2 spectra carry
/// their isolation window (target + lower/upper bound) and collision energy.
pub fn write_scans_mzml(scans: &[MzMlScan], out_path: &Path) -> io::Result<usize> {
    let file = File::create(out_path)?;
    let mut writer = MzMLWriter::new(file);
    for (i, s) in scans.iter().enumerate() {
        let mut arrays = BinaryArrayMap::new();
        arrays.add(DataArray::wrap(
            &ArrayType::MZArray,
            BinaryDataArrayType::Float64,
            f64_le_bytes(&s.mz),
        ));
        arrays.add(DataArray::wrap(
            &ArrayType::IntensityArray,
            BinaryDataArrayType::Float32,
            f32_le_bytes(&s.intensity),
        ));

        let mut desc = SpectrumDescription::default();
        desc.id = format!("scan={}", i + 1);
        desc.index = i;
        desc.ms_level = s.ms_level;
        desc.polarity = ScanPolarity::Positive;
        desc.signal_continuity = SignalContinuity::Centroid;

        // Scan start time: mzML stores it in minutes.
        let mut scan_event = ScanEvent::default();
        scan_event.start_time = s.retention_time_s / 60.0;
        desc.acquisition.scans.push(scan_event);

        if let Some(iso) = &s.isolation {
            let mut prec = Precursor::default();
            prec.isolation_window = IsolationWindow {
                target: iso.center_mz as f32,
                lower_bound: iso.lower_mz as f32,
                upper_bound: iso.upper_mz as f32,
                ..Default::default()
            };
            let mut ion = SelectedIon::default();
            ion.mz = iso.center_mz;
            prec.ions.push(ion);
            let mut activation = Activation::default();
            activation.energy = iso.collision_energy as f32;
            prec.activation = activation;
            desc.precursor.push(prec);
        }

        let spectrum = RawSpectrum::new(desc, arrays);
        writer.write(&spectrum)?;
    }
    writer.close()?;
    Ok(scans.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mzdata::io::MzMLReader;

    #[test]
    fn write_read_roundtrip() {
        let scans = vec![
            MzMlScan {
                ms_level: 1,
                retention_time_s: 60.0,
                mz: vec![400.1, 500.2, 600.3],
                intensity: vec![1000.0, 2000.0, 1500.0],
                isolation: None,
            },
            MzMlScan {
                ms_level: 2,
                retention_time_s: 61.5,
                mz: vec![150.1, 250.2, 410.3],
                intensity: vec![500.0, 800.0, 300.0],
                isolation: Some(MzMlIsolation {
                    center_mz: 500.0,
                    lower_mz: 495.0,
                    upper_mz: 505.0,
                    collision_energy: 27.0,
                }),
            },
        ];
        let out = std::env::temp_dir().join("rustdf_mzml_roundtrip.mzML");
        let n = write_scans_mzml(&scans, &out).expect("write mzML");
        assert_eq!(n, 2);

        // Read it back with mzdata and confirm the content survived the round-trip.
        let mut reader = MzMLReader::open_path(&out).expect("reopen mzML");
        let read: Vec<_> = (&mut reader).collect();
        assert_eq!(read.len(), 2, "spectrum count");

        let ms1 = &read[0];
        assert_eq!(ms1.ms_level(), 1);
        assert!((ms1.start_time() - 1.0).abs() < 1e-6, "MS1 RT 60s -> 1.0 min");
        let p1 = ms1.peaks();
        assert_eq!(p1.len(), 3, "MS1 peak count");

        let ms2 = &read[1];
        assert_eq!(ms2.ms_level(), 2);
        let prec = ms2.precursor().expect("MS2 precursor");
        assert!((prec.isolation_window.target - 500.0).abs() < 1e-3, "isolation target");
        assert!((prec.isolation_window.lower_bound - 495.0).abs() < 1e-3);
        assert_eq!(ms2.peaks().len(), 3, "MS2 peak count");
    }
}
