use crate::cluster::utility::{build_frame_bin_view, smooth_vector_gaussian, FrameBinView, MzScale};
use crate::data::dia::TimsDatasetDIA;
use crate::data::handle::TimsData;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct MzScanGrid {
    pub scans: Vec<usize>,
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub data_raw: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct MzScanWindowGrid {
    pub rt_range_frames: (usize, usize),   // existing: local frame-index range (within the plan)
    pub rt_range_sec:    (f32, f32),       // existing
    pub frame_id_bounds: (u32, u32),       // NEW: actual frame IDs [lo, hi]
    pub window_group:    Option<u32>,      // NEW: DIA group provenance
    pub scans: Vec<usize>,
    pub data: Vec<f32>,     // column-major (rows, cols)
    pub rows: usize,
    pub cols: usize,
    pub data_raw: Option<Vec<f32>>,
}

pub fn build_mz_scan_grid_for_frames(
    data_handle: &TimsDatasetDIA,
    frame_ids_sorted: &[u32],
    scale: &MzScale,
    maybe_sigma_scans: Option<f32>,
    truncate: f32,
    num_threads: usize,
) -> MzScanGrid {
    let frames = data_handle.get_slice(frame_ids_sorted.to_vec(), num_threads).frames;
    if frames.is_empty() {
        return MzScanGrid { scans: vec![], data: vec![], rows: 0, cols: 0, data_raw: None };
    }

    let rows = scale.num_bins();
    let global_num_scans = data_handle.meta_data.iter()
        .map(|fr_meta| (fr_meta.num_scans + 1) as usize)
        .max()
        .unwrap_or(0);
    if rows == 0 || global_num_scans == 0 {
        return MzScanGrid { scans: vec![], data: vec![], rows: 0, cols: 0, data_raw: None };
    }

    let views: Vec<FrameBinView> = (0..frames.len())
        .into_par_iter()
        .map(|i| build_frame_bin_view(frames[i].clone(), scale, global_num_scans))
        .collect();

    let cols = global_num_scans;
    let mut raw = vec![0.0f32; rows * cols];

    for v in &views {
        for b_i in 0..v.unique_bins.len() {
            let row = v.unique_bins[b_i];
            let start = v.offsets[b_i];
            let end   = v.offsets[b_i + 1];
            for i in start..end {
                let s_phys = v.scan_idx[i] as usize;
                if s_phys < cols {
                    raw[s_phys * rows + row] += v.intensity[i];
                }
            }
        }
    }

    let do_smooth = maybe_sigma_scans.unwrap_or(0.0) > 0.0;
    let data = if do_smooth {
        let mut sm = raw.clone();
        for r in 0..rows {
            let mut y: Vec<f32> = (0..cols).map(|c| sm[c * rows + r]).collect();
            smooth_vector_gaussian(&mut y, maybe_sigma_scans.unwrap(), truncate);
            for c in 0..cols { sm[c * rows + r] = y[c]; }
        }
        sm
    } else {
        raw.clone()
    };

    MzScanGrid {
        scans: (0..cols).collect(),
        data,
        rows,
        cols,
        data_raw: if do_smooth { Some(raw) } else { None },
    }
}

use crate::cluster::utility::ImPeak1D;

#[derive(Clone, Copy)]
pub struct StitchParams {
    pub min_overlap_frames: usize,  // e.g. 1–3
    pub max_scan_delta: usize,      // e.g. 1–2
    pub jaccard_min: f32,           // e.g. 0.2 (set 0 to ignore)
}

fn rt_overlap((a0,a1):(usize,usize),(b0,b1):(usize,usize)) -> usize {
    let lo = a0.max(b0);
    let hi = a1.min(b1);
    hi.saturating_sub(lo).saturating_add(1)
}

fn jaccard((a0,a1):(usize,usize),(b0,b1):(usize,usize)) -> f32 {
    let inter = rt_overlap((a0,a1),(b0,b1)) as f32;
    if inter == 0.0 { return 0.0; }
    let len_a = (a1 - a0 + 1) as f32;
    let len_b = (b1 - b0 + 1) as f32;
    inter / (len_a + len_b - inter)
}

fn compatible(p:&ImPeak1D, q:&ImPeak1D, sp:&StitchParams) -> bool {
    if p.mz_row != q.mz_row { return false; }
    if p.window_group != q.window_group { return false; } // keeps groups isolated
    if (p.scan as isize - q.scan as isize).abs() as usize > sp.max_scan_delta { return false; }
    let ov = rt_overlap(p.rt_bounds, q.rt_bounds);
    if ov < sp.min_overlap_frames { return false; }
    if sp.jaccard_min > 0.0 && jaccard(p.rt_bounds, q.rt_bounds) < sp.jaccard_min { return false; }
    true
}

fn merge_two(mut a: ImPeak1D, b: &ImPeak1D) -> ImPeak1D {
    // union of RT/frame bounds
    a.rt_bounds = (a.rt_bounds.0.min(b.rt_bounds.0), a.rt_bounds.1.max(b.rt_bounds.1));
    a.frame_id_bounds = (a.frame_id_bounds.0.min(b.frame_id_bounds.0),
                         a.frame_id_bounds.1.max(b.frame_id_bounds.1));

    // scan / subscan (weighted by apex_smoothed)
    let w0 = a.apex_smoothed.max(1e-6);
    let w1 = b.apex_smoothed.max(1e-6);
    a.subscan = (a.subscan*w0 + b.subscan*w1) / (w0 + w1);
    a.scan = ((a.scan as f32*w0 + b.scan as f32*w1) / (w0+w1)).round() as usize;

    // bounds union
    a.left  = a.left.min(b.left);
    a.right = a.right.max(b.right);
    a.width_scans = a.right.saturating_sub(a.left).saturating_add(1);

    // intensities / stats
    a.apex_raw      = a.apex_raw.max(b.apex_raw);
    a.apex_smoothed = a.apex_smoothed.max(b.apex_smoothed);
    a.prominence    = a.prominence.max(b.prominence);
    a.area_raw     += b.area_raw;

    // keep mobility if any (prefer non-None)
    if a.mobility.is_none() { a.mobility = b.mobility; }

    a
}

/// Stitch duplicates across overlapping RT windows.
/// Input: a flat Vec<ImPeak1D> coming from many windows.
/// Output: deduplicated Vec<ImPeak1D>.
pub fn stitch_im_peaks_across_windows(mut peaks: Vec<ImPeak1D>, sp: StitchParams) -> Vec<ImPeak1D> {
    if peaks.is_empty() { return peaks; }

    // bucket by (window_group, mz_row)
    use std::collections::BTreeMap;
    let mut buckets: BTreeMap<(Option<u32>,usize), Vec<ImPeak1D>> = BTreeMap::new();
    for p in peaks.drain(..) {
        buckets.entry((p.window_group, p.mz_row)).or_default().push(p);
    }

    // parallel stitch per bucket
    buckets.into_par_iter().flat_map(|((_wg, _row), mut v)| {
        // sort by scan, then by rt start (helps the sweep)
        v.sort_unstable_by_key(|p| (p.scan, p.rt_bounds.0));

        let mut out: Vec<ImPeak1D> = Vec::with_capacity(v.len());
        for p in v.into_iter() {
            if let Some(last) = out.last_mut() {
                if compatible(last, &p, &sp) {
                    let merged = merge_two(last.clone(), &p);
                    *last = merged;
                    continue;
                }
            }
            out.push(p);
        }
        out
    }).collect()
}