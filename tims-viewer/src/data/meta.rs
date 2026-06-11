//! Run metadata read from the TDF SQLite tables — cheap, no binary frame decoding.
//!
//! This gives us the axis ranges, a frame-id → retention-time map, per-frame MS type,
//! and a total-point estimate before touching `analysis.tdf_bin`, so the camera and
//! axis cube can render instantly while points stream in.

use anyhow::{Context, Result};

use rustdf::data::meta::{read_global_meta_sql, read_meta_data_sql};

use super::point::{AxisBounds, AxisTransform};

/// Per-frame info we keep resident for the whole session.
/// `is_ms2`/`num_peaks` back per-frame LOD and filtering planned for later phases.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct FrameInfo {
    pub id: u32,
    pub retention_time: f64,
    /// True for MS2 / fragment frames (`ms_ms_type != 0`).
    pub is_ms2: bool,
    pub num_peaks: u64,
}

/// Lightweight index over a run's metadata.
pub struct MetaIndex {
    pub data_path: String,
    pub frames: Vec<FrameInfo>,
    pub bounds: AxisBounds,
    pub total_points_estimate: u64,
}

impl MetaIndex {
    /// Load metadata for a real `.d` folder.
    pub fn load(data_path: &str) -> Result<Self> {
        let global = read_global_meta_sql(data_path)
            .map_err(|e| anyhow::anyhow!("read_global_meta_sql failed: {e:?}"))
            .context("reading global TDF metadata")?;
        let meta = read_meta_data_sql(data_path)
            .map_err(|e| anyhow::anyhow!("read_meta_data_sql failed: {e:?}"))
            .context("reading per-frame TDF metadata")?;

        anyhow::ensure!(!meta.is_empty(), "TDF run contains no frames");

        let mut frames = Vec::with_capacity(meta.len());
        let mut rt_min = f64::INFINITY;
        let mut rt_max = f64::NEG_INFINITY;
        let mut total: u64 = 0;
        for m in &meta {
            // Keep FrameMeta.id as the frame id we pass to get_frame. rustdf's get_frame
            // indexes its own frame_meta_data by position (id - 1) from the same
            // unordered SQL query, so we require ids to appear in exact 1..=N positional
            // order (validated below). Checked-convert to avoid an i64->u32 wrap.
            let id = u32::try_from(m.id)
                .map_err(|_| anyhow::anyhow!("frame id {} is out of u32 range", m.id))?;
            let num_peaks = m.num_peaks.max(0) as u64;
            if num_peaks > 0 {
                rt_min = rt_min.min(m.time);
                rt_max = rt_max.max(m.time);
            }
            total = total.saturating_add(num_peaks);
            frames.push(FrameInfo {
                id,
                retention_time: m.time,
                is_ms2: m.ms_ms_type != 0,
                num_peaks,
            });
        }

        // rustdf looks up frames by vector position (id - 1) from the same query, so the
        // metadata must arrive in exact 1..=N order. Checking the id *set* isn't enough:
        // an unordered result like [2,1,3] passes a set check but makes get_frame(2) read
        // position 1 (id 1's data). Require positional order so this can't silently
        // corrupt the view; timsTOF runs satisfy it.
        let n = frames.len();
        let ordered = frames
            .iter()
            .enumerate()
            .all(|(i, f)| f.id == i as u32 + 1);
        anyhow::ensure!(
            ordered,
            "frame metadata is not in contiguous 1..={n} positional order \
             (first id {:?}, last id {:?}); the viewer relies on rustdf's position-based \
             frame indexing",
            frames.first().map(|f| f.id),
            frames.last().map(|f| f.id),
        );

        if !rt_min.is_finite() || !rt_max.is_finite() {
            // No non-empty frames carried RT; fall back to the full frame span.
            rt_min = frames.first().map(|f| f.retention_time).unwrap_or(0.0);
            rt_max = frames.last().map(|f| f.retention_time).unwrap_or(1.0);
        }

        let bounds = AxisBounds {
            mz: AxisTransform::new(
                global.mz_acquisition_range_lower,
                global.mz_acquisition_range_upper,
            ),
            im: AxisTransform::new(
                global.one_over_k0_range_lower,
                global.one_over_k0_range_upper,
            ),
            rt: AxisTransform::new(rt_min, rt_max),
        };

        Ok(MetaIndex {
            data_path: data_path.to_string(),
            frames,
            bounds,
            total_points_estimate: total,
        })
    }

    /// Synthetic metadata for the `DEMO` path (no Bruker data required).
    pub fn demo(num_frames: usize, total_points: u64) -> Self {
        let mut frames = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let rt = i as f64 * 0.5; // 0.5 s spacing
            frames.push(FrameInfo {
                id: (i + 1) as u32,
                retention_time: rt,
                is_ms2: i % 10 == 9, // ~10% MS2
                num_peaks: total_points / num_frames.max(1) as u64,
            });
        }
        let bounds = AxisBounds {
            mz: AxisTransform::new(100.0, 1700.0),
            im: AxisTransform::new(0.6, 1.6),
            rt: AxisTransform::new(0.0, (num_frames.max(1) - 1) as f64 * 0.5),
        };
        MetaIndex {
            data_path: "DEMO".to_string(),
            frames,
            bounds,
            total_points_estimate: total_points,
        }
    }
}
