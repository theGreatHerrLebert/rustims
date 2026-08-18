//! MOBILion `.mbi` source: metadata index + m/z lookup.
//!
//! The viewer's cube is `(x = m/z, y = mobility, z = RT)`. MBI supplies all three:
//! m/z from the TOF calibration, mobility as **arrival time in milliseconds** (SLIM
//! has no `1/K0`; see `MetaIndex::im_unit`), and RT from each frame's start time.
//!
//! Fragment frames are identified by collision energy rather than an MS-level field:
//! these acquisitions are quadrupole-free Mobility-Aligned Fragmentation, alternating
//! low-CE precursor frames with high-CE fragment frames.

use anyhow::{Context, Result};

use mobilionmbi::{MbiFile, TofCalibration};

use super::meta::{FrameInfo, MetaIndex};
use super::point::{AxisBounds, AxisTransform};

/// Does this path look like an MBI file?
pub fn is_mbi_path(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .map(|e| e.eq_ignore_ascii_case("mbi"))
        .unwrap_or(false)
}

/// A frame counts as fragment data when its collision energy is above zero.
fn frame_is_ms2(file: &MbiFile, index: usize) -> bool {
    file.collision_energy(index)
        .ok()
        .flatten()
        .map(|v| v > 0.0)
        .unwrap_or(false)
}

/// Precomputed `m/z` for every TOF bin.
///
/// The calibration is stored per frame but is constant within a file in practice, so
/// one table serves the whole run. Computing it once turns a sqrt + polynomial per
/// point into an array index — worth it at ~10^8 points.
pub struct MzLookup {
    pub mz: Vec<f64>,
}

impl MzLookup {
    pub fn build(cal: &TofCalibration, n_tof: usize) -> Self {
        let mz = (0..n_tof as u64).map(|i| cal.index_to_mz(i)).collect();
        MzLookup { mz }
    }

    #[inline]
    pub fn get(&self, tof: u64) -> f64 {
        // Out-of-range cannot happen for data read from the same file, but a corrupt
        // index should not panic the loader thread.
        self.mz.get(tof as usize).copied().unwrap_or(f64::NAN)
    }
}

/// Read a run's metadata without decoding any frame data.
///
/// Frame point counts come from the `data-counts` dataset *shape*, which HDF5 stores
/// in the object header — no decompression, so this stays a metadata-only pass.
pub fn load_meta(path: &str) -> Result<MetaIndex> {
    let file = MbiFile::open(path).with_context(|| format!("opening {path}"))?;
    let n = file.n_frames();
    if n == 0 {
        anyhow::bail!("{path} contains no frames");
    }

    let rts = file
        .retention_times()
        .context("reading per-frame retention times")?;

    let mut frames = Vec::with_capacity(n);
    let mut total: u64 = 0;
    for i in 1..=n {
        let npts = file.frame_nnz(i).unwrap_or(0) as u64;
        total += npts;
        frames.push(FrameInfo {
            id: i as u32,
            retention_time: rts.get(i - 1).copied().unwrap_or(f64::NAN),
            is_ms2: frame_is_ms2(&file, i),
            num_peaks: npts,
        });
    }

    // Mobility axis: the full drift cycle of the first frame. Scan counts can differ by
    // one between frames, so take the longest ramp seen rather than frame 1's.
    let mut n_scans = 0u32;
    let mut period_ms = 0.0f64;
    for i in 1..=n.min(64) {
        if let Ok(axis) = file.drift_axis(i) {
            if axis.n_scans as u32 > n_scans {
                n_scans = axis.n_scans as u32;
                period_ms = axis.period_ms;
            }
        }
    }
    let im_max = (n_scans as f64) * period_ms;

    // m/z axis: the calibrated span of the TOF axis. `adc-record-size` is the full
    // digitiser record; the low bins are pre-injection and calibrate to ~0, so clamp
    // the low end to the first sensible m/z rather than showing a dead decade.
    let cal = file
        .calibration(1)
        .context("reading the mass calibration of frame 1")?;
    let n_tof = file
        .global_metadata()
        .get("adc-record-size")
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let mz_hi = cal.index_to_mz(n_tof.saturating_sub(1) as u64);
    let mz_lo = mz_at_first_useful_bin(&cal, n_tof);

    let rt_lo = frames.first().map(|f| f.retention_time).unwrap_or(0.0);
    let rt_hi = frames.last().map(|f| f.retention_time).unwrap_or(1.0);

    Ok(MetaIndex {
        data_path: path.to_string(),
        frames,
        bounds: AxisBounds {
            mz: AxisTransform::new(mz_lo, mz_hi),
            im: AxisTransform::new(0.0, im_max),
            rt: AxisTransform::new(rt_lo, rt_hi),
        },
        total_points_estimate: total,
        num_scans: n_scans,
        im_unit: "ms",
    })
}

/// The m/z of the first TOF bin worth showing.
///
/// The TOF axis starts at t = 0, which calibrates to ~0 m/z; those bins hold no data
/// (the SDK exposes a valid window starting around bin 30000). Rather than hardcode
/// that offset, walk up to the first bin above a floor of 20 m/z.
fn mz_at_first_useful_bin(cal: &TofCalibration, n_tof: usize) -> f64 {
    const FLOOR_MZ: f64 = 20.0;
    let idx = cal.mz_to_index(FLOOR_MZ);
    if (idx as usize) < n_tof {
        cal.index_to_mz(idx)
    } else {
        cal.index_to_mz(0)
    }
}

/// Open a file and build its m/z lookup, for the loader thread.
pub fn open_for_points(path: &str) -> Result<(MbiFile, MzLookup)> {
    let file = MbiFile::open(path).with_context(|| format!("opening {path}"))?;
    let cal = file.calibration(1).context("reading the mass calibration")?;
    let n_tof = file
        .global_metadata()
        .get("adc-record-size")
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let lookup = MzLookup::build(&cal, n_tof);
    Ok((file, lookup))
}
