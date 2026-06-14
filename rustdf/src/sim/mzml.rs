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
use mzdata::meta::DissociationMethodTerm;
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
        write_one_scan(&mut writer, i, s)?;
    }
    writer.close()?;
    Ok(scans.len())
}

/// Build the mzML spectrum for one scan and write it to an open writer. Shared by the
/// in-memory `write_scans_mzml` and the streaming `render_db_to_mzml` (which writes each
/// frame as it is rendered rather than buffering them all).
fn write_one_scan<W: io::Write>(
    writer: &mut MzMLWriter<W>,
    index: usize,
    s: &MzMlScan,
) -> io::Result<()> {
    if s.mz.len() != s.intensity.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "scan {index}: m/z ({}) and intensity ({}) length mismatch",
                s.mz.len(),
                s.intensity.len()
            ),
        ));
    }
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
    desc.id = format!("scan={}", index + 1);
    desc.index = index;
    desc.ms_level = s.ms_level;
    desc.polarity = ScanPolarity::Positive;
    desc.signal_continuity = SignalContinuity::Centroid;

    // Scan start time: mzML stores it in minutes.
    let mut scan_event = ScanEvent::default();
    scan_event.start_time = s.retention_time_s / 60.0;
    desc.acquisition.scans.push(scan_event);

    if let Some(iso) = &s.isolation {
        let mut prec = Precursor::default();
        // mzdata stores isolation bounds as ABSOLUTE m/z (its writer converts to mzML
        // lower/upper OFFSETS); pass absolute lower/upper.
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
        // Declare the dissociation method (HCD / beam-type CID) so the <activation> is
        // not just a bare energy. NOTE: mzdata serialises `energy` as PSI-MS "collision
        // energy" (eV); our value is the model's NCE/CE as supplied — downstream search
        // engines key on fragments, not this field, but the unit caveat stands.
        activation
            .methods_mut()
            .push(DissociationMethodTerm::BeamTypeCollisionInducedDissociation);
        activation.energy = iso.collision_energy as f32;
        prec.activation = activation;
        desc.precursor.push(prec);
    }

    let spectrum = RawSpectrum::new(desc, arrays);
    writer.write(&spectrum)?;
    Ok(())
}

/// Outcome of a DB → mzML render.
#[derive(Debug, Clone)]
pub struct MzMlWriteSummary {
    pub scans: usize,
    pub ms1: usize,
    pub ms2: usize,
    pub ms2_nonempty: usize,
}

/// Render a no-IM DIA `synthetic_data.db` to mzML — the vendor-neutral output used for
/// SCIEX SWATH (and any no-IM DIA scheme). MS1 frames are rendered via the precursor
/// marginal; MS2 frames via the per-window fragment kernel (gated by the DB window's m/z
/// + CE). This is the open-format analogue of `astral_dispatch::write_astral_raw`: same
/// render kernels, mzML out instead of a vendor `.raw`. Returns counts.
pub fn render_db_to_mzml(
    db_path: &Path,
    out_path: &Path,
    num_threads: usize,
    quad_k: f64,
) -> Result<MzMlWriteSummary, String> {
    use std::collections::HashMap;

    use mscore::timstof::quadrupole::WindowTransmission;

    use crate::sim::acquisition::ScanDescriptor;
    use crate::sim::dia::TimsTofSyntheticsFrameBuilderDIA;
    use crate::sim::handle::TimsTofSyntheticsDataHandle;
    use crate::sim::scheme::DataMode;

    let builder = TimsTofSyntheticsFrameBuilderDIA::new(db_path, false, num_threads)
        .map_err(|e| format!("open DB: {e}"))?;
    let handle =
        TimsTofSyntheticsDataHandle::new(db_path).map_err(|e| format!("open DB handle: {e}"))?;

    // frame -> (center, width, ce) for MS2 frames, via the DB window tables.
    let wg = handle
        .read_window_group_settings()
        .map_err(|e| format!("read windows: {e}"))?;
    let by_group: HashMap<u32, (f64, f64, f64)> = wg
        .iter()
        .map(|w| {
            (
                w.window_group,
                (w.isolation_mz as f64, w.isolation_width as f64, w.collision_energy as f64),
            )
        })
        .collect();
    let f2g = handle
        .read_frame_to_window_group()
        .map_err(|e| format!("read frame->group: {e}"))?;
    let mut frame_window: HashMap<u32, (f64, f64, f64)> = HashMap::new();
    for r in &f2g {
        if let Some(&w) = by_group.get(&r.window_group) {
            frame_window.insert(r.frame_id, w);
        }
    }

    let prec_set = &builder.precursor_frame_builder.precursor_frame_id_set;
    let mut frame_ids: Vec<u32> = builder
        .precursor_frame_builder
        .frames
        .iter()
        .map(|f| f.frame_id)
        .collect();
    frame_ids.sort_unstable();

    // Stream: open the writer and emit each frame as it is rendered, so peak memory is
    // one scan, not the whole run (a 28k-frame run is a ~1.5 GB mzML).
    let file = std::fs::File::create(out_path).map_err(|e| format!("create {out_path:?}: {e}"))?;
    let mut writer = MzMLWriter::new(file);
    let (mut ms1, mut ms2, mut ms2_nonempty, mut total) = (0usize, 0usize, 0usize, 0usize);
    for frame_id in frame_ids {
        let desc = if prec_set.contains(&frame_id) {
            let ev = builder
                .precursor_frame_builder
                .render_precursor_scan(frame_id, DataMode::Centroid);
            ms1 += 1;
            ScanDescriptor::from_rendered_event(&ev, 0.0)?
        } else {
            let (center, width, ce) = *frame_window
                .get(&frame_id)
                .ok_or_else(|| format!("MS2 frame {frame_id} has no window in the DB tables"))?;
            let wt = WindowTransmission::new(center, width, quad_k);
            let ev = builder.render_fragment_scan(frame_id, &wt, ce, DataMode::Centroid);
            ms2 += 1;
            ScanDescriptor::from_rendered_event(&ev, ce)?
        };
        let isolation = desc.isolation.map(|w| MzMlIsolation {
            center_mz: w.center_mz,
            lower_mz: w.center_mz - w.width_mz / 2.0,
            upper_mz: w.center_mz + w.width_mz / 2.0,
            collision_energy: w.collision_energy,
        });
        if isolation.is_some() && !desc.peaks.is_empty() {
            ms2_nonempty += 1;
        }
        let scan = MzMlScan {
            ms_level: desc.ms_level,
            retention_time_s: desc.retention_time,
            mz: desc.peaks.iter().map(|(m, _)| *m).collect(),
            intensity: desc.peaks.iter().map(|(_, i)| *i).collect(),
            isolation,
        };
        write_one_scan(&mut writer, total, &scan).map_err(|e| format!("write scan {total}: {e}"))?;
        total += 1;
    }
    writer.close().map_err(|e| format!("close mzML: {e}"))?;
    Ok(MzMlWriteSummary { scans: total, ms1, ms2, ms2_nonempty })
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

    // Gated: set TIMSIM_DIA_DB to a no-IM DIA synthetic_data.db (e.g. an Orbitrap/SCIEX
    // run) to render it to mzML and confirm it reads back.
    #[test]
    fn render_db_to_mzml_roundtrip() {
        let db = match std::env::var("TIMSIM_DIA_DB") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP render_db_to_mzml_roundtrip: set TIMSIM_DIA_DB=<synthetic_data.db>");
                return;
            }
        };
        let out = std::env::temp_dir().join("rustdf_render_db.mzML");
        let s = render_db_to_mzml(std::path::Path::new(&db), &out, 4, 15.0).expect("render mzML");
        assert!(s.ms1 > 0 && s.ms2 > 0, "expected MS1 + MS2 scans, got {s:?}");
        let mut reader = MzMLReader::open_path(&out).expect("reopen mzML");
        let n_ms1 = (&mut reader).filter(|sp| sp.ms_level() == 1).count();
        let mut reader2 = MzMLReader::open_path(&out).expect("reopen mzML 2");
        let n_total = (&mut reader2).count();
        assert_eq!(n_total, s.scans, "round-trip scan count");
        assert_eq!(n_ms1, s.ms1, "round-trip MS1 count");
        eprintln!(
            "render_db_to_mzml OK: {} scans ({} MS1, {} MS2, {} non-empty MS2) -> mzML reads back {}",
            s.scans, s.ms1, s.ms2, s.ms2_nonempty, n_total
        );
    }
}
