use std::collections::BTreeMap;
use mscore::timstof::frame::TimsFrame;
use crate::cluster::utility::{bin_range, build_frame_bin_view, smooth_vector_gaussian, FrameBinView, MzScale, RtPeak1D};
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
use crate::data::meta::FrameMeta;

#[derive(Clone, Copy)]
pub struct StitchParams {
    pub min_overlap_frames: usize,
    pub max_scan_delta: usize,
    pub jaccard_min: f32,
    pub max_mz_row_delta: usize,   // NEW, e.g. 0 (current), 1 or 2
    pub allow_cross_groups: bool,  // NEW if you want to stitch across groups
}

fn same_row_or_close(p:&ImPeak1D, q:&ImPeak1D, d:usize) -> bool {
    p.mz_row.abs_diff(q.mz_row) <= d
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
    if !same_row_or_close(p, q, sp.max_mz_row_delta) { return false; }
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

#[inline(always)]
fn frame_rt(meta: &FrameMeta) -> f32 { meta.time as f32 }

/// Build CSR views for the subset of frames in `slice_frames` whose IDs lie in [fid_lo, fid_hi],
/// in *RT order*. Returns (views, frame_ids_sorted, frame_times_sorted, global_num_scans).
pub fn make_views_for_span_from_slice(
    // preloaded frames (e.g. from ds.get_slice called by the caller once per window/group)
    slice_frames: &[TimsFrame],
    // lookup: frame_id -> time (if you don’t want to touch dataset meta, pass a parallel vec instead)
    meta_lookup: Option<&[FrameMeta]>,
    fid_lo: u32,
    fid_hi: u32,
    scale: &MzScale,
) -> (Vec<FrameBinView>, Vec<u32>, Vec<f32>, usize) {
    // filter frames by id bounds
    let mut picked: Vec<&TimsFrame> = slice_frames
        .iter()
        .filter(|fr| (fr.frame_id as u32) >= fid_lo && (fr.frame_id as u32) <= fid_hi)
        .collect();

    // sort by RT if we have metadata; otherwise keep existing order (assume pre-sorted slice)
    if let Some(meta) = meta_lookup {
        picked.sort_by(|a, b| {
            let ta = meta.iter().find(|m| m.id == a.frame_id as i64).map(frame_rt).unwrap_or(0.0);
            let tb = meta.iter().find(|m| m.id == b.frame_id as i64).map(frame_rt).unwrap_or(0.0);
            ta.partial_cmp(&tb).unwrap()
        });
    }

    let frame_ids: Vec<u32> = picked.iter().map(|fr| fr.frame_id as u32).collect();
    let frame_times: Vec<f32> = if let Some(meta) = meta_lookup {
        frame_ids.iter().map(|fid| {
            meta.iter().find(|m| (m.id as u32)==*fid).map(frame_rt).unwrap_or(0.0)
        }).collect()
    } else {
        // If you already carry RT times alongside your slice, pass them instead and replace this path.
        vec![0.0; frame_ids.len()]
    };

    // global max scan across just these frames (caps scan_idx during view build)
    let global_num_scans: usize = picked.iter()
        .flat_map(|fr| fr.scan.iter())
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(0);

    let views: Vec<FrameBinView> = picked
        .into_par_iter()
        .map(|fr| build_frame_bin_view(fr.clone(), scale, global_num_scans))
        .collect();

    (views, frame_ids, frame_times, global_num_scans)
}

#[derive(Clone, Debug)]
pub struct ImFirstRtOptions {
    pub mz_ppm_window: f32,
    pub rt_sigma_frames: Option<f32>,
    pub rt_min_prom: f32,
    pub rt_min_distance_frames: usize,
    pub rt_min_width_frames: usize,
    pub rt_pad_left: usize,
    pub rt_pad_right: usize,
}

impl Default for ImFirstRtOptions {
    fn default() -> Self {
        Self {
            mz_ppm_window: 10.0,
            rt_sigma_frames: Some(1.25),
            rt_min_prom: 100.0,
            rt_min_distance_frames: 2,
            rt_min_width_frames: 2,
            rt_pad_left: 0,
            rt_pad_right: 0,
        }
    }
}

/// Build RT marginal and pick RT sub-peaks **using already-built views**.
/// No disk I/O; no calls to get_slice.
/// - `views`: CSR views for frames in **time order** covering the IM-peak span
/// - `frame_times`: same length as `views`, RT seconds in ascending order
pub fn rt_peaks_from_impeak_on_views(
    scale: &MzScale,
    im_peak: &ImPeak1D,
    views: &[FrameBinView],
    frame_times: &[f32],
    opts: &ImFirstRtOptions,
) -> (Vec<RtPeak1D>, Vec<f32>) {
    if views.is_empty() { return (Vec::new(), Vec::new()); }
    assert_eq!(views.len(), frame_times.len(), "views and frame_times must align");

    // m/z slab around this IM row
    let mz_center = scale.center(im_peak.mz_row);
    let tol_da = mz_center * opts.mz_ppm_window * 1e-6;
    let (bin_lo, bin_hi) = scale.index_range_for_mz_window(mz_center - tol_da, mz_center + tol_da);

    // IM window
    let (im_l, im_r) = (im_peak.left, im_peak.right);

    // RT marginal
    let mut rt_marg_raw = vec![0.0f32; views.len()];
    for (t, v) in views.iter().enumerate() {
        let (start, end) = bin_range(v, bin_lo, bin_hi);
        let mut sum = 0.0f32;
        for i in start..end {
            let s = v.scan_idx[i] as usize;
            if s >= im_l && s <= im_r {
                sum += v.intensity[i];
            }
        }
        rt_marg_raw[t] = sum;
    }

    // optional smoothing
    let mut rt_marg_sm = rt_marg_raw.clone();
    if let Some(sig) = opts.rt_sigma_frames { smooth_vector_gaussian(&mut rt_marg_sm, sig, 3.0); }

    // your peak picker
    let rt_peaks = crate::cluster::utility::find_peaks_row(
        &rt_marg_sm,
        &rt_marg_raw,
        im_peak.mz_row,
        mz_center,
        Some(frame_times),
        opts.rt_min_prom,
        opts.rt_min_distance_frames,
        opts.rt_min_width_frames,
        opts.rt_pad_left,
        opts.rt_pad_right,
    );

    (rt_peaks, rt_marg_raw)
}

/// IM marginal for a chosen RT sub-window, again **on the same views**
pub fn im_marginal_for_rt_window_on_views(
    scale: &MzScale,
    views: &[FrameBinView],
    im_peak: &ImPeak1D,
    rt_left_local: usize,
    rt_right_local: usize,
    mz_ppm_window: f32,
) -> Vec<f32> {
    if views.is_empty() { return Vec::new(); }
    let (im_l, im_r) = (im_peak.left, im_peak.right);
    let scans_len = im_r - im_l + 1;
    let mut im_marg = vec![0.0f32; scans_len];

    let mz_center = scale.center(im_peak.mz_row);
    let tol_da = mz_center * mz_ppm_window * 1e-6;
    let (bin_lo, bin_hi) = scale.index_range_for_mz_window(mz_center - tol_da, mz_center + tol_da);

    let lo = rt_left_local.min(views.len() - 1);
    let hi = rt_right_local.min(views.len() - 1);
    if lo > hi { return im_marg; }

    for v in &views[lo..=hi] {
        let (start, end) = bin_range(v, bin_lo, bin_hi);
        for i in start..end {
            let s = v.scan_idx[i] as usize;
            if s >= im_l && s <= im_r {
                im_marg[s - im_l] += v.intensity[i];
            }
        }
    }
    im_marg
}

pub struct RtImResult {
    pub im_peak: ImPeak1D,
    pub rt_peaks: Vec<RtPeak1D>,
    pub rt_marg: Vec<f32>,
}

pub fn evaluate_impeaks_batched_on_slice(
    slice_frames: &[TimsFrame],      // preloaded once by the caller
    meta_lookup: Option<&[FrameMeta]>,
    scale: &MzScale,
    im_peaks: Vec<ImPeak1D>,
    opts: &ImFirstRtOptions,
) -> Vec<RtImResult> {
    // Bucket by RT frame-id bounds to reuse the same views
    let mut buckets: BTreeMap<(u32,u32), Vec<ImPeak1D>> = BTreeMap::new();
    for p in im_peaks { buckets.entry(p.frame_id_bounds).or_default().push(p); }

    let mut out: Vec<RtImResult> = Vec::new();

    for ((fid_lo, fid_hi), peaks) in buckets.into_iter() {
        // 1) Build views ONCE for this RT span
        let (views, _frame_ids, frame_times, _nscans) =
            make_views_for_span_from_slice(slice_frames, meta_lookup, fid_lo, fid_hi, scale);

        // 2) Evaluate each IM peak on these cached views
        for ip in peaks {
            let (rt_subpeaks, rt_marg) = rt_peaks_from_impeak_on_views(
                scale, &ip, &views, &frame_times, opts
            );

            // If needed, one can compute IM marginals per sub-peak here by calling
            // `im_marginal_for_rt_window_on_views(...)` with pk.left/right.

            out.push(RtImResult { im_peak: ip, rt_peaks: rt_subpeaks, rt_marg });
        }
    }

    out
}