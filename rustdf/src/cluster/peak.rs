use rayon::prelude::*;
use mscore::timstof::frame::TimsFrame;
use crate::cluster::utility::MzScale;

#[derive(Clone, Debug)]
pub struct ImPeak1D {
    pub mz_row: usize,                 // row in current m/z grid
    pub mz_center: f32,                // center m/z
    pub mz_bounds: (f32, f32),         // min, max m/z
    pub rt_bounds: (usize, usize),     // columns [lo, hi] in current RT grid
    pub frame_id_bounds: (u32, u32),   // materialized for robustness
    pub window_group: Option<u32>,     // DIA group provenance

    pub scan: usize,
    pub mobility: Option<f32>,
    pub apex_smoothed: f32,
    pub apex_raw: f32,
    pub prominence: f32,
    pub left: usize,
    pub right: usize,
    pub left_x: f32,
    pub right_x: f32,
    pub width_scans: usize,
    pub area_raw: f32,
    pub subscan: f32,
}

#[derive(Clone, Debug)]
pub struct FrameBinView {
    pub _frame_id: u32,
    pub unique_bins: Vec<usize>,
    pub offsets: Vec<usize>,
    pub scan_idx: Vec<u32>,
    pub intensity: Vec<f32>,
}

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
    pub rt_range_frames: (usize, usize),
    pub rt_range_sec:    (f32, f32),
    pub frame_id_bounds: (u32, u32),
    pub window_group:    Option<u32>,
    pub scans: Vec<usize>,
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub data_raw: Option<Vec<f32>>,
}

#[derive(Clone, Copy)]
pub struct StitchParams {
    pub min_overlap_frames: usize,
    pub max_scan_delta: usize,
    pub jaccard_min: f32,
    pub max_mz_row_delta: usize,   // NEW, e.g. 0 (current), 1 or 2
    pub allow_cross_groups: bool,  // NEW if you want to stitch across groups
}

fn same_row_or_close_im(p:&ImPeak1D, q:&ImPeak1D, d:usize) -> bool {
    p.mz_row.abs_diff(q.mz_row) <= d
}

/// Allow slight disagreement between mz rows picked by IM vs RT passes
#[inline]
fn _same_row_or_close(a_row: usize, b_row: usize, max_delta: usize) -> bool {
    a_row.abs_diff(b_row) <= max_delta
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
    if !same_row_or_close_im(p, q, sp.max_mz_row_delta) { return false; }
    if !sp.allow_cross_groups && p.window_group != q.window_group { return false; }
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

pub fn build_frame_bin_view(
    fr: TimsFrame,
    scale: &MzScale,
    global_num_scans: usize,
) -> FrameBinView {
    let n = fr.ims_frame.mz.len();
    let mut bins_idx: Vec<usize> = Vec::with_capacity(n);
    let mut scans_u:  Vec<u32>   = Vec::with_capacity(n);
    let mut intens:   Vec<f32>   = Vec::with_capacity(n);

    let scans_vec: &Vec<i32> = &fr.scan;
    for i in 0..n {
        if let Some(idx) = scale.index_of(fr.ims_frame.mz[i] as f32) {
            bins_idx.push(idx);
            let s_val = scans_vec[i];
            debug_assert!(s_val >= 0, "Negative scan index in frame {}", fr.frame_id);
            let s_u32: u32 = u32::try_from(s_val).expect("scan index does not fit u16");
            scans_u.push((s_u32 as usize).min(global_num_scans.saturating_sub(1)) as u32);
            intens.push(fr.ims_frame.intensity[i] as f32);
        }
    }

    // sort by bin index and build CSR
    let mut idx: Vec<usize> = (0..bins_idx.len()).collect();
    idx.sort_unstable_by_key(|&i| bins_idx[i]);

    let mut unique_bins: Vec<usize> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut scan_sorted: Vec<u32> = Vec::with_capacity(idx.len());
    let mut inten_sorted: Vec<f32> = Vec::with_capacity(idx.len());

    let mut cur: Option<usize> = None;
    for &i in &idx {
        let b = bins_idx[i];
        if cur.map_or(true, |c| c != b) {
            unique_bins.push(b);
            counts.push(0);
            cur = Some(b);
        }
        *counts.last_mut().unwrap() += 1;
        scan_sorted.push(scans_u[i]);
        inten_sorted.push(intens[i]);
    }

    let mut offsets = Vec::with_capacity(unique_bins.len() + 1);
    offsets.push(0);
    for c in &counts { offsets.push(offsets.last().unwrap() + *c); }

    FrameBinView {
        _frame_id: fr.frame_id as u32,
        unique_bins,
        offsets,
        scan_idx: scan_sorted,
        intensity: inten_sorted,
    }
}