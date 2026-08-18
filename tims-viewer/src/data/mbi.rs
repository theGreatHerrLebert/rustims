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
    // SLIM runs spend the first chunk of the drift cycle empty (ions are still in
    // transit), so auto-trim the mobility floor to where signal actually starts.
    let im_lo = detect_im_floor(&file, n, n_scans as usize, period_ms);
    if im_lo > 0.0 {
        log::info!(
            "mobility floor auto-trimmed to {im_lo:.0} ms (<{:.1}% of intensity below)",
            IM_TRIM_QUANTILE * 100.0
        );
    }
    // Scans remaining on the trimmed axis (keeps `span / (num_scans-1)` ≈ the true
    // per-scan spacing, which anchors the client's scans-based clustering reach).
    let scans_trimmed = if period_ms > 0.0 {
        n_scans.saturating_sub((im_lo / period_ms).round() as u32)
    } else {
        n_scans
    };

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
            im: AxisTransform::new(im_lo, im_max),
            rt: AxisTransform::new(rt_lo, rt_hi),
        },
        total_points_estimate: total,
        num_scans: scans_trimmed,
        im_unit: "ms",
    })
}

/// Fraction of total intensity the auto-trimmed mobility floor may exclude. Chosen from a
/// real CE-ramp HeLa run where the cut is insensitive to the exact value: the empty lead-in
/// holds <0.1% of intensity, so anything in [0.1%, 1%] lands within a few ms.
const IM_TRIM_QUANTILE: f64 = 0.005;
/// Snap the detected floor DOWN to a multiple of this (ms), for a tick-friendly axis start.
const IM_TRIM_SNAP_MS: f64 = 10.0;

/// Arrival time (ms) below which the run holds under [`IM_TRIM_QUANTILE`] of its intensity.
///
/// SLIM has no signal in the early drift cycle (ions have not arrived yet), so the raw
/// axis starts with a dead zone. Sample 16 frames spread across the run, accumulate
/// per-scan intensity, and cut at the quantile — snapped down to a clean 10 ms, capped at
/// half the cycle, and 0 for a run whose signal starts immediately.
fn detect_im_floor(file: &MbiFile, n_frames: usize, n_scans: usize, period_ms: f64) -> f64 {
    const SAMPLES: usize = 16;
    if n_scans == 0 || period_ms <= 0.0 {
        return 0.0;
    }
    let mut per_scan = vec![0f64; n_scans];
    // Spread over `samples - 1` intervals so the first AND last frame are hit, and a run
    // shorter than SAMPLES frames samples each frame exactly once (codex review).
    let samples = SAMPLES.min(n_frames).max(1);
    for s in 0..samples {
        let fid = if samples == 1 { 1 } else { 1 + s * (n_frames - 1) / (samples - 1) };
        let Ok(frame) = file.frame(fid) else { continue };
        for row in 0..frame.n_rows.min(n_scans) {
            let (a, b) = (frame.indptr[row] as usize, frame.indptr[row + 1] as usize);
            per_scan[row] += frame.data[a..b].iter().map(|&v| v as f64).sum::<f64>();
        }
    }
    let total: f64 = per_scan.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let threshold = IM_TRIM_QUANTILE * total;
    let mut acc = 0.0;
    let mut cut = 0usize;
    for (row, v) in per_scan.iter().enumerate() {
        acc += v;
        if acc >= threshold {
            cut = row;
            break;
        }
    }
    let floored = ((cut as f64 * period_ms) / IM_TRIM_SNAP_MS).floor() * IM_TRIM_SNAP_MS;
    floored.clamp(0.0, 0.5 * n_scans as f64 * period_ms)
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
