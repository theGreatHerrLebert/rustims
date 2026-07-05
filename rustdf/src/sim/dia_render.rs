//! Shared no-IM DIA DB render: walk a `synthetic_data.db` and yield one
//! [`ScanDescriptor`] per frame in acquisition (frame-id) order — MS1 via the precursor
//! marginal, MS2 via the per-window fragment kernel (gated by the DB window's m/z + CE).
//!
//! Both the open **mzML** writer (`mzml.rs`) and the native SCIEX **`.wiff.scan`** writer
//! (`sciex_dispatch.rs`) render identically through this; only the encoder downstream
//! differs. Streaming: the sink is called once per scan so peak memory stays at one scan.

#![cfg(any(feature = "mzml", feature = "sciex"))]

use std::collections::HashMap;
use std::path::Path;

use mscore::timstof::quadrupole::WindowTransmission;

use crate::sim::acquisition::ScanDescriptor;
use crate::sim::dia::TimsTofSyntheticsFrameBuilderDIA;
use crate::sim::handle::TimsTofSyntheticsDataHandle;
use crate::sim::scheme::DataMode;

/// Tallies from a DIA render pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderCounts {
    pub scans: usize,
    pub ms1: usize,
    pub ms2: usize,
    pub ms2_nonempty: usize,
}

/// Render every frame of a no-IM DIA `synthetic_data.db` in acquisition (frame-id) order,
/// handing each rendered [`ScanDescriptor`] to `sink`. A `sink` error aborts the walk and
/// is propagated. Returns the counts. The frame order is the acquisition schedule the DB
/// was built with (per cycle: MS1 first, then the MS2 windows in method order), so a
/// positional consumer can rely on that ordering.
pub fn render_dia_scans_each<F>(
    db_path: &Path,
    num_threads: usize,
    quad_k: f64,
    mut sink: F,
) -> Result<RenderCounts, String>
where
    F: FnMut(ScanDescriptor) -> Result<(), String>,
{
    let builder = TimsTofSyntheticsFrameBuilderDIA::new(db_path, false, num_threads)
        .map_err(|e| format!("open DB: {e}"))?;
    let handle =
        TimsTofSyntheticsDataHandle::new(db_path).map_err(|e| format!("open DB handle: {e}"))?;

    // frame -> (center, width, ce) for MS2 frames, via the DB window tables. One window per
    // group is the no-IM DIA invariant; reject conflicting duplicate rows and non-positive
    // widths (silently keeping the last would mis-render).
    let wg = handle
        .read_window_group_settings()
        .map_err(|e| format!("read windows: {e}"))?;
    let mut by_group: HashMap<u32, (f64, f64, f64)> = HashMap::new();
    for w in &wg {
        let v = (w.isolation_mz as f64, w.isolation_width as f64, w.collision_energy as f64);
        if v.1 <= 0.0 {
            return Err(format!(
                "window_group {} has non-positive isolation width {}",
                w.window_group, v.1
            ));
        }
        if let Some(prev) = by_group.insert(w.window_group, v) {
            if prev != v {
                return Err(format!(
                    "conflicting dia_ms_ms_windows rows for window_group {}: {prev:?} vs {v:?} \
                     (expects one window per group, no-IM DIA)",
                    w.window_group
                ));
            }
        }
    }
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

    let mut c = RenderCounts::default();
    for frame_id in frame_ids {
        let desc = if prec_set.contains(&frame_id) {
            let ev = builder
                .precursor_frame_builder
                .render_precursor_scan(frame_id, DataMode::Centroid);
            c.ms1 += 1;
            ScanDescriptor::from_rendered_event(&ev, 0.0)?
        } else {
            let (center, width, ce) = *frame_window
                .get(&frame_id)
                .ok_or_else(|| format!("MS2 frame {frame_id} has no window in the DB tables"))?;
            let wt = WindowTransmission::new(center, width, quad_k);
            let ev = builder.render_fragment_scan(frame_id, &wt, ce, DataMode::Centroid);
            c.ms2 += 1;
            ScanDescriptor::from_rendered_event(&ev, ce)?
        };
        if desc.isolation.is_some() && !desc.peaks.is_empty() {
            c.ms2_nonempty += 1;
        }
        c.scans += 1;
        sink(desc)?;
    }
    Ok(c)
}
