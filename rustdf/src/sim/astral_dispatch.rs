//! P6e dispatch: render an Astral *build-from-template* `synthetic_data.db` to a
//! Thermo `.raw` (the `write_astral_raw` driver).
//!
//! The DB was built by the lean Astral acquisition (frames 1:1 with the template's
//! scans, per-window NCE — see `AstralAcquisitionBuilder`), so the dispatch is
//! clean: walk the template slot manifest in order, render each frame's scan, and
//! author it into its slot. No RT-mapping / modulo-cycling (those were spike
//! artifacts of retrofitting a Bruker trunk). The render core stays panic-free; the
//! whole driver returns `Result` so the PyO3 boundary surfaces structured errors.

#![cfg(feature = "thermo")]

use std::collections::HashMap;
use std::path::Path;

use mscore::timstof::quadrupole::WindowTransmission;

use crate::sim::acquisition::{AcquisitionWriter, ScanDescriptor, ThermoRawWriter};
use crate::sim::dia::TimsTofSyntheticsFrameBuilderDIA;
use crate::sim::handle::TimsTofSyntheticsDataHandle;
use crate::sim::scheme::DataMode;

/// Outcome of a `.raw` dispatch.
#[derive(Debug, Clone)]
pub struct AstralWriteSummary {
    pub scans: usize,
    pub ms1: usize,
    pub ms2: usize,
    pub ms2_nonempty: usize,
    /// Slots whose authored payload overflowed the template packet budget and were
    /// written empty (cleared) to keep the schedule + checksum intact.
    pub overflow_cleared: usize,
    pub checksum_valid: bool,
}

/// Tuning for the dispatch (peak caps keep each authored payload within the
/// template's pre-allocated packet budget; the template-mutation writer cannot
/// grow a packet).
#[derive(Debug, Clone, Copy)]
pub struct AstralWriteOptions {
    pub num_threads: usize,
    /// Quadrupole transmission edge steepness `k` (same as the Bruker path).
    pub quad_k: f64,
    pub max_ms1_peaks: usize,
    pub max_ms2_peaks: usize,
}

impl Default for AstralWriteOptions {
    fn default() -> Self {
        AstralWriteOptions { num_threads: 4, quad_k: 15.0, max_ms1_peaks: 400, max_ms2_peaks: 120 }
    }
}

fn cap_top_intensity(peaks: &mut Vec<(f64, f32)>, max: usize) {
    if peaks.len() > max {
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        peaks.truncate(max);
    }
}

/// Render the Astral build-from-template DB at `db_path` to a Thermo `.raw` at
/// `out_path`, authored onto `template_path`.
///
/// For each template slot (in acquisition order): an MS1 frame is rendered via the
/// full mobility marginal (`precursor_scan_marginal_spectrum`) and authored as a
/// profile; an MS2 frame is rendered via the per-window fragment kernel
/// (`render_fragment_scan`, gated by the DB window's m/z + NCE) and authored as
/// centroids + isolation. MS1 peaks are filtered to the template scan's FTMS
/// frequency-grid extent; peaks are capped to the template packet budget; a payload
/// that still overflows is written empty (cleared) so the slot is consumed and the
/// zero-residual / checksum contract holds.
pub fn write_astral_raw(
    db_path: &Path,
    template_path: &Path,
    out_path: &Path,
    opts: AstralWriteOptions,
) -> Result<AstralWriteSummary, String> {
    use thermorawfile::RawFile;

    let builder = TimsTofSyntheticsFrameBuilderDIA::new(db_path, false, opts.num_threads)
        .map_err(|e| format!("open Astral DB: {e}"))?;
    let handle = TimsTofSyntheticsDataHandle::new(db_path)
        .map_err(|e| format!("open Astral DB handle: {e}"))?;

    // frame -> (center, width, ce) for MS2 frames, via the DB's window tables.
    let wg = handle.read_window_group_settings().map_err(|e| format!("read windows: {e}"))?;
    let by_group: HashMap<u32, (f64, f64, f64)> = wg
        .iter()
        .map(|w| {
            (
                w.window_group,
                (w.isolation_mz as f64, w.isolation_width as f64, w.collision_energy as f64),
            )
        })
        .collect();
    let f2g = handle.read_frame_to_window_group().map_err(|e| format!("read frame->group: {e}"))?;
    let mut frame_window: HashMap<u32, (f64, f64, f64)> = HashMap::new();
    for r in &f2g {
        if let Some(&w) = by_group.get(&r.window_group) {
            frame_window.insert(r.frame_id, w);
        }
    }

    // Template grid (for MS1 m/z filtering) + the ordered slot manifest.
    let raw = RawFile::open(template_path).map_err(|e| format!("open template: {e}"))?;
    let calib = raw
        .calibration_at_event(raw.scantrailer_addr as usize + 4)
        .ok_or("template has no MS1 calibration")?;
    let mut writer = ThermoRawWriter::from_template(template_path, out_path)
        .map_err(|e| format!("open writer: {e}"))?;
    let manifest: Vec<(u32, bool)> = writer.manifest().to_vec();

    // Frames are 1:1 with template slots, in scan order (the Astral builder built
    // them that way). Sort by frame id to align with the manifest order.
    let mut frame_ids: Vec<u32> =
        builder.precursor_frame_builder.frames.iter().map(|f| f.frame_id).collect();
    frame_ids.sort_unstable();
    if frame_ids.len() != manifest.len() {
        return Err(format!(
            "frame/template mismatch: {} DB frames vs {} template slots — the DB \
             was not built from this template",
            frame_ids.len(),
            manifest.len()
        ));
    }

    let prec_set = &builder.precursor_frame_builder.precursor_frame_id_set;
    let (mut ms1, mut ms2, mut ms2_nonempty, mut overflow_cleared) = (0usize, 0usize, 0usize, 0usize);

    for (i, &(slot_scan, is_profile)) in manifest.iter().enumerate() {
        let frame_id = frame_ids[i];
        let is_ms1 = prec_set.contains(&frame_id);
        if is_ms1 != is_profile {
            return Err(format!(
                "level mismatch at slot {i} (template scan {slot_scan}): DB frame {frame_id} \
                 is {} but the template slot is {}",
                if is_ms1 { "MS1" } else { "MS2" },
                if is_profile { "profile/MS1" } else { "centroid/MS2" }
            ));
        }

        let (mut desc, ce_for_empty) = if is_ms1 {
            let ev = builder
                .precursor_frame_builder
                .render_precursor_scan(frame_id, DataMode::Profile);
            let mut d = ScanDescriptor::from_rendered_event(&ev, 0.0)?;
            // Keep only peaks within this scan's FTMS frequency-grid m/z extent.
            if let Some(p) = raw.profile(slot_scan) {
                let a = p.mz_of_bin(0, &calib);
                let b = p.mz_of_bin(p.nbins.saturating_sub(1), &calib);
                let (lo, hi) = (a.min(b), a.max(b));
                d.peaks.retain(|(m, _)| *m >= lo && *m <= hi);
            }
            cap_top_intensity(&mut d.peaks, opts.max_ms1_peaks);
            (d, 0.0)
        } else {
            let (center, width, ce) = *frame_window.get(&frame_id).ok_or_else(|| {
                format!("MS2 frame {frame_id} has no window in the DB tables")
            })?;
            let wt = WindowTransmission::new(center, width, opts.quad_k);
            let ev = builder.render_fragment_scan(frame_id, &wt, ce, DataMode::Centroid);
            let mut d = ScanDescriptor::from_rendered_event(&ev, ce)?;
            cap_top_intensity(&mut d.peaks, opts.max_ms2_peaks);
            if !d.peaks.is_empty() {
                ms2_nonempty += 1;
            }
            (d, ce)
        };

        // Author; on a residual packet-budget overflow, consume the slot with an
        // empty (cleared) scan so the schedule + checksum stay intact.
        match writer.write_scan(&desc) {
            Ok(()) => {}
            Err(_) => {
                desc.peaks = Vec::new();
                writer
                    .write_scan(&ScanDescriptor {
                        ms_level: desc.ms_level,
                        retention_time: desc.retention_time,
                        isolation: desc.isolation,
                        peaks: Vec::new(),
                    })
                    .map_err(|e| format!("author slot {i} (even empty): {e}"))?;
                overflow_cleared += 1;
                let _ = ce_for_empty;
            }
        }
        if is_ms1 {
            ms1 += 1;
        } else {
            ms2 += 1;
        }
    }

    writer.finalize().map_err(|e| format!("finalize: {e}"))?;

    // Independent re-read to confirm the authored file is valid.
    let check = RawFile::open(out_path).map_err(|e| format!("re-open output: {e}"))?;
    Ok(AstralWriteSummary {
        scans: check.scan_count(),
        ms1,
        ms2,
        ms2_nonempty,
        overflow_cleared,
        checksum_valid: check.checksum_valid(),
    })
}
