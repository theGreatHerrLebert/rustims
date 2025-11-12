use std::sync::Arc;
use rayon::prelude::*;
use mscore::timstof::frame::TimsFrame;
use crate::cluster::utility::{fallback_rt_peak_from_trace, quad_subsample, rt_peak_id, smooth_vector_gaussian, trapezoid_area_fractional, MzScale};

#[derive(Clone, Debug)]
pub struct RtFrames {
    pub frames: Vec<FrameBinView>,
    pub frame_ids: Vec<u32>,
    pub rt_times: Vec<f32>,
    pub scale: Arc<MzScale>,       // NEW: the CSR scale used for these frames
}
impl RtFrames {
    #[inline]
    pub fn ctx(&self) -> RtTraceCtx<'_> {
        RtTraceCtx {
            frame_ids_sorted: &self.frame_ids,
            rt_times_sec: &self.rt_times,
        }
    }

    #[inline]
    pub fn is_consistent(&self) -> bool {
        self.frames.len() == self.frame_ids.len() && self.frame_ids.len() == self.rt_times.len()
    }
}

#[derive(Clone, Debug)]
pub struct RtLocalPeak {
    pub rt_idx: usize,
    pub rt_sec: Option<f32>,
    pub apex_smoothed: f32,
    pub apex_raw: f32,
    pub prominence: f32,
    pub left_x: f32,
    pub right_x: f32,
    pub width_frames: usize,
    pub area_raw: f32,
    pub subframe: f32,
    pub left_sec: Option<f32>,
    pub right_sec: Option<f32>,
    pub width_sec: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct RtTraceCtx<'a> {
    pub frame_ids_sorted: &'a [u32],
    pub rt_times_sec: &'a [f32],
}

#[derive(Clone, Debug)]
pub struct RtPeak1D {
    // geometry in RT index/time
    pub rt_idx: usize,
    pub rt_sec: Option<f32>,
    pub apex_smoothed: f32,
    pub apex_raw: f32,
    pub prominence: f32,
    pub left_x: f32,
    pub right_x: f32,
    pub width_frames: usize,
    pub area_raw: f32,
    pub subframe: f32,

    // provenance / bounds in frames and frame_ids
    pub rt_bounds_frames: (usize, usize),   // inclusive in local RT trace
    pub frame_id_bounds: (u32, u32),        // materialized
    pub window_group: Option<u32>,

    // m/z context carried from the IM parent (exact row + physical window)
    pub mz_row: usize,
    pub mz_center: f32,
    pub mz_bounds: (f32, f32),

    // linkage
    pub parent_im_id: Option<PeakId>,
    pub id: PeakId,                          // optional: RT-level id
}

// Stable 64-bit id for peaks
pub type PeakId = i64;


#[derive(Clone, Debug)]
pub struct ImPeak1D {
    pub mz_row: usize,                 // row in current m/z grid
    pub mz_center: f32,                // center m/z
    pub mz_bounds: (f32, f32),         // min, max m/z
    pub rt_bounds: (usize, usize),     // columns [lo, hi] in current RT grid
    pub frame_id_bounds: (u32, u32),   // materialized for robustness
    pub window_group: Option<u32>,     // DIA group provenance

    pub scan: usize,
    pub left: usize,
    pub right: usize,

    pub scan_abs: usize,
    pub left_abs: usize,
    pub right_abs: usize,
    
    pub scan_sigma: Option<f32>,
    pub mobility: Option<f32>,
    pub apex_smoothed: f32,
    pub apex_raw: f32,
    pub prominence: f32,
    pub left_x: f32,
    pub right_x: f32,
    pub width_scans: usize,            // interpreted as ABSOLUTE width
    pub area_raw: f32,
    pub subscan: f32,
    pub id: PeakId,
}

#[derive(Clone, Debug)]
pub struct FrameBinView {
    pub _frame_id: u32,
    pub unique_bins: Vec<usize>,
    pub offsets: Vec<usize>,
    pub scan_idx: Vec<u32>,            // ABSOLUTE scan indices
    pub intensity: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct MzScanGrid {
    pub scans: Vec<usize>,
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub data_raw: Option<Vec<f32>>,
    pub scale: MzScale,
}

impl MzScanGrid {
    #[inline]
    pub fn mz_center_for_row(&self, r: usize) -> f32 {
        self.scale.center(r)
    }
    #[inline]
    pub fn mz_bounds_for_row(&self, r: usize) -> (f32, f32) {
        // safe: edges.len() == rows + 1
        (self.scale.edges[r], self.scale.edges[r + 1])
    }
}

#[derive(Clone, Debug)]
pub struct MzScanWindowGrid {
    pub rt_range_frames: (usize, usize),
    pub rt_range_sec:    (f32, f32),
    pub frame_id_bounds: (u32, u32),
    pub window_group:    Option<u32>,

    pub scale: Arc<MzScale>,

    pub scans: Vec<usize>,
    pub data: Option<Vec<f32>>,
    pub rows: usize,
    pub cols: usize,
    pub data_raw: Option<Vec<f32>>,
}

impl MzScanWindowGrid {
    #[inline]
    pub fn mz_center_for_row(&self, r: usize) -> f32 {
        self.scale.center(r)
    }
    #[inline]
    pub fn mz_bounds_for_row(&self, r: usize) -> (f32, f32) {
        // safe: edges.len() == rows + 1
        (self.scale.edges[r], self.scale.edges[r + 1])
    }
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
    // Use ABSOLUTE scan centers for cross-window compatibility
    if (p.scan_abs as isize - q.scan_abs as isize).abs() as usize > sp.max_scan_delta { return false; }
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
    a.subscan  = (a.subscan*w0 + b.subscan*w1) / (w0 + w1);
    a.scan     = ((a.scan as f32*w0 + b.scan as f32*w1) / (w0+w1)).round() as usize;
    a.scan_abs = ((a.scan_abs as f32*w0 + b.scan_abs as f32*w1) / (w0+w1)).round() as usize;

    // bounds union (both local and absolute maintained)
    a.left      = a.left.min(b.left);
    a.right     = a.right.max(b.right);
    a.left_abs  = a.left_abs.min(b.left_abs);
    a.right_abs = a.right_abs.max(b.right_abs);

    // width is defined on the ABSOLUTE axis
    a.width_scans = a.right_abs.saturating_sub(a.left_abs).saturating_add(1);

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
        // sort by ABSOLUTE scan center, then by rt start (helps the sweep)
        v.sort_unstable_by_key(|p| (p.scan_abs, p.rt_bounds.0));

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

#[inline]
fn sum_frame_bins_scans(
    fbv: &FrameBinView,
    bin_lo: usize, bin_hi: usize,
    scan_lo: usize, scan_hi: usize,
) -> f32 {
    // Locate contiguous block of unique_bins in [bin_lo..bin_hi]
    // Since unique_bins is sorted, two binary searches:
    let ub = &fbv.unique_bins;
    if ub.is_empty() || bin_lo > bin_hi { return 0.0; }

    let start = match ub.binary_search(&bin_lo) {
        Ok(i) => i,
        Err(i) => i.min(ub.len()),
    };
    let mut acc = 0.0f32;

    let end_bin = bin_hi;
    let mut i = start;
    while i < ub.len() {
        let b = ub[i];
        if b > end_bin { break; }
        // entries for this bin: offsets[i]..offsets[i+1]
        let lo = fbv.offsets[i];
        let hi = fbv.offsets[i + 1];
        // scan filter (ABSOLUTE axis)
        let scans = &fbv.scan_idx[lo..hi];
        let ints  = &fbv.intensity[lo..hi];
        // linear scan; if this becomes hot, keep scans sorted and do two lower_bounds
        for (s, val) in scans.iter().zip(ints.iter()) {
            let s = *s as usize;
            if s >= scan_lo && s <= scan_hi { acc += *val; }
        }
        i += 1;
    }
    acc
}

/// Returns an RT trace for a single IM peak using bin padding around a row
pub fn rt_trace_for_im_peak(
    frames: &[FrameBinView],        // RT-sorted, same group
    mz_row: usize,
    bin_pad: usize,                 // e.g., 0 or 1
    scan_lo: usize,
    scan_hi: usize,
) -> Vec<f32> {
    let bin_lo = mz_row.saturating_sub(bin_pad);
    let bin_hi = mz_row.saturating_add(bin_pad);
    let mut v = Vec::with_capacity(frames.len());
    for fbv in frames {
        v.push(sum_frame_bins_scans(fbv, bin_lo, bin_hi, scan_lo, scan_hi));
    }
    v
}

#[inline]
pub fn rt_trace_for_im_peak_by_bounds(
    frames: &[FrameBinView],
    rt_scale: &MzScale,            // RtFrames.scale
    mz_bounds: (f32, f32),         // im_peak.mz_bounds
    extra_bins_pad: usize,         // 0–2
    scan_lo: usize,
    scan_hi: usize,
) -> Vec<f32> {
    // Map physical m/z bounds into the RT CSR scale, then pad by bins.
    let (mut bin_l, mut bin_r) = rt_scale.index_range_for_mz_window(mz_bounds.0, mz_bounds.1);
    bin_l = bin_l.saturating_sub(extra_bins_pad);
    bin_r = bin_r.saturating_add(extra_bins_pad);

    let mut v = Vec::with_capacity(frames.len());
    for fbv in frames {
        v.push(sum_frame_bins_scans(fbv, bin_l, bin_r, scan_lo, scan_hi));
    }
    v
}

pub fn find_rt_peaks(
    y_smoothed: &[f32],
    y_raw: &[f32],
    rt_times: &[f32],
    min_prom: f32,
    min_sep_sec: f32,
    min_width_sec: f32,
) -> Vec<RtLocalPeak> {
    let n = y_smoothed.len();
    if n < 3 || y_raw.len() != n || rt_times.len() != n { return Vec::new(); }

    let row_max = y_raw.iter().copied().fold(0.0f32, f32::max);
    if row_max < min_prom { return Vec::new(); }

    // 1) candidates
    let mut cands = Vec::new();
    for i in 1..n-1 {
        let yi = y_smoothed[i];
        if yi > y_smoothed[i-1] && yi >= y_smoothed[i+1] { cands.push(i); }
    }

    let mut peaks: Vec<RtLocalPeak> = Vec::new();
    for &i in &cands {
        let apex = y_smoothed[i];

        // 2) prominence baseline
        let mut l = i; let mut left_min = apex;
        while l > 0 { l -= 1; left_min = left_min.min(y_smoothed[l]); if y_smoothed[l] > apex { break; } }
        let mut r = i; let mut right_min = apex;
        while r + 1 < n { r += 1; right_min = right_min.min(y_smoothed[r]); if y_smoothed[r] > apex { break; } }

        let baseline = left_min.max(right_min);
        let prom = apex - baseline;
        if prom < min_prom { continue; }

        // 3) half-prom fractional crossings
        let half = baseline + 0.5 * prom;

        let mut wl = i;
        while wl > 0 && y_smoothed[wl] > half { wl -= 1; }
        let left_x = if wl < i && wl + 1 < n {
            let y0 = y_smoothed[wl]; let y1 = y_smoothed[wl + 1];
            wl as f32 + if y1 != y0 { (half - y0) / (y1 - y0) } else { 0.0 }
        } else { wl as f32 };

        let mut wr = i;
        while wr + 1 < n && y_smoothed[wr] > half { wr += 1; }
        let right_x = if wr > i && wr < n {
            let y0 = y_smoothed[wr - 1]; let y1 = y_smoothed[wr];
            (wr - 1) as f32 + if y1 != y0 { (half - y0) / (y1 - y0) } else { 0.0 }
        } else { wr as f32 };

        // seconds-based width check
        let left_t  = t_at_index_frac(rt_times, left_x.max(0.0));
        let right_t = t_at_index_frac(rt_times, right_x.min((n - 1) as f32));
        let width_sec = (right_t - left_t).max(0.0);
        if width_sec < min_width_sec { continue; }

        // 4) sub-frame apex offset
        let sub = if i > 0 && i + 1 < n {
            quad_subsample(y_smoothed[i - 1], y_smoothed[i], y_smoothed[i + 1]).clamp(-0.5, 0.5)
        } else { 0.0 };

        // 5) time-based NMS (seconds)
        if let Some(last) = peaks.last() {
            let dt = (rt_times[i] - last.rt_sec.unwrap_or(rt_times[last.rt_idx])).abs();
            if dt < min_sep_sec {
                if apex <= last.apex_smoothed { continue; }
                let _ = peaks.pop();
            }
        }

        // 6) area under raw
        let area = trapezoid_area_fractional(y_raw, left_x.max(0.0), right_x.min((n - 1) as f32));

        // apex time (subsampled)
        let apex_t = t_at_index_frac(rt_times, i as f32 + sub);

        peaks.push(RtLocalPeak {
            rt_idx: i,
            rt_sec: Some(apex_t),
            apex_smoothed: apex,
            apex_raw: y_raw[i],
            prominence: prom,
            left_x,
            right_x,
            width_frames: ((right_x - left_x).max(0.0)).round() as usize, // legacy
            area_raw: area,
            subframe: sub,
            left_sec: Some(left_t),
            right_sec: Some(right_t),
            width_sec: Some(width_sec),
        });
    }
    peaks
}

#[derive(Clone, Copy, Debug)]
pub struct RtExpandParams {
    pub bin_pad: usize,
    pub smooth_sigma_sec: f32,
    pub smooth_trunc_k: f32,
    pub min_prom: f32,
    pub min_sep_sec: f32,
    pub min_width_sec: f32,
    pub fallback_if_frames_lt: usize,
    pub fallback_frac_width: f32,
}

pub fn expand_im_peak_along_rt(
    im_peak: &ImPeak1D,
    frames_sorted: &[FrameBinView],
    rt_ctx: RtTraceCtx<'_>,
    rt_scale: &MzScale,
    p: RtExpandParams,
) -> Vec<RtPeak1D> {

    // ------------------------------------------------------------
    // 1) Map absolute frame IDs → local precursor RT index range
    // ------------------------------------------------------------
    // We binary-search the sorted precursor frame_ids to locate
    // where this IM peak is allowed to exist in RT.
    let (fid_lo_abs, fid_hi_abs) = im_peak.frame_id_bounds;

    let allow_lo = match rt_ctx.frame_ids_sorted.binary_search(&fid_lo_abs) {
        Ok(i) => i,
        Err(_) => return Vec::new(),      // this IM peak does not exist in this precursor trace
    };
    let allow_hi = match rt_ctx.frame_ids_sorted.binary_search(&fid_hi_abs) {
        Ok(i) => i,
        Err(_) => return Vec::new(),
    };
    let (allow_lo, allow_hi) = (allow_lo.min(allow_hi), allow_lo.max(allow_hi));

    // Safety: the peak RT cannot exceed the range of available frames
    if allow_lo >= frames_sorted.len() || allow_hi >= frames_sorted.len() {
        return Vec::new();
    }

    // ------------------------------------------------------------
    // 2) Compute the full raw RT trace (over all precursor frames)
    // ------------------------------------------------------------
    let trace_raw_full = rt_trace_for_im_peak_by_bounds(
        frames_sorted,
        rt_scale,
        im_peak.mz_bounds,
        p.bin_pad,
        im_peak.left_abs,
        im_peak.right_abs,
    );

    if trace_raw_full.is_empty() {
        return Vec::new();
    }

    // ------------------------------------------------------------
    // 3) CLAMP the trace strictly to the IM peak's allowed RT region
    // ------------------------------------------------------------
    let trace_raw = &trace_raw_full[allow_lo ..= allow_hi];
    let rt_times_clamped = &rt_ctx.rt_times_sec[allow_lo ..= allow_hi];
    let n_clamped = trace_raw.len();

    if n_clamped == 0 || !trace_raw.iter().any(|x| *x > 0.0) {
        return Vec::new();
    }

    // ------------------------------------------------------------
    // 4) If very narrow RT region → fallback peak logic
    // ------------------------------------------------------------
    if n_clamped < p.fallback_if_frames_lt {
        if let Some(pk) =
            fallback_rt_peak_from_trace(trace_raw, rt_ctx, im_peak, p.fallback_frac_width)
        {
            // map local clamped bounds back to global indices
            let l = allow_lo + pk.left_x.floor().clamp(0.0, (n_clamped - 1) as f32) as usize;
            let r = allow_lo + pk.right_x.ceil().clamp(0.0, (n_clamped - 1) as f32) as usize;
            let lo_fid = rt_ctx.frame_ids_sorted[l];
            let hi_fid = rt_ctx.frame_ids_sorted[r];

            return vec![
                RtPeak1D {
                    parent_im_id: Some(im_peak.id),
                    mz_row: im_peak.mz_row,
                    mz_center: im_peak.mz_center,
                    mz_bounds: im_peak.mz_bounds,
                    window_group: im_peak.window_group,

                    rt_idx: allow_lo + pk.rt_idx,
                    rt_sec: pk.rt_sec,
                    apex_smoothed: pk.apex_smoothed,
                    apex_raw: pk.apex_raw,
                    prominence: pk.prominence,
                    left_x: pk.left_x + allow_lo as f32,
                    right_x: pk.right_x + allow_lo as f32,
                    width_frames: pk.width_frames,
                    area_raw: pk.area_raw,
                    subframe: pk.subframe,

                    rt_bounds_frames: (l, r),
                    frame_id_bounds: (lo_fid.min(hi_fid), lo_fid.max(hi_fid)),
                    id: rt_peak_id(&pk),
                }
            ];
        }
        return Vec::new();
    }

    // ------------------------------------------------------------
    // 5) Smooth the *clamped* trace
    // ------------------------------------------------------------
    let dt = effective_dt(rt_times_clamped);
    let sigma_frames = (p.smooth_sigma_sec / dt).max(0.75);
    let mut trace_smooth = trace_raw.to_vec();
    smooth_vector_gaussian(&mut trace_smooth[..], sigma_frames, p.smooth_trunc_k);

    // ------------------------------------------------------------
    // 6) Peak finding on the *clamped* region
    // ------------------------------------------------------------
    let base = find_rt_peaks(
        &trace_smooth,
        trace_raw,
        rt_times_clamped,
        p.min_prom,
        p.min_sep_sec,
        p.min_width_sec,
    );

    if base.is_empty() {
        if let Some(pk) =
            fallback_rt_peak_from_trace(trace_raw, rt_ctx, im_peak, p.fallback_frac_width)
        {
            // same handling as fallback above
            let l = allow_lo + pk.left_x.floor().clamp(0.0, (n_clamped - 1) as f32) as usize;
            let r = allow_lo + pk.right_x.ceil().clamp(0.0, (n_clamped - 1) as f32) as usize;
            let lo_fid = rt_ctx.frame_ids_sorted[l];
            let hi_fid = rt_ctx.frame_ids_sorted[r];

            return vec![
                RtPeak1D {
                    parent_im_id: Some(im_peak.id),
                    mz_row: im_peak.mz_row,
                    mz_center: im_peak.mz_center,
                    mz_bounds: im_peak.mz_bounds,
                    window_group: im_peak.window_group,

                    rt_idx: allow_lo + pk.rt_idx,
                    rt_sec: pk.rt_sec,
                    apex_smoothed: pk.apex_smoothed,
                    apex_raw: pk.apex_raw,
                    prominence: pk.prominence,
                    left_x: pk.left_x + allow_lo as f32,
                    right_x: pk.right_x + allow_lo as f32,
                    width_frames: pk.width_frames,
                    area_raw: pk.area_raw,
                    subframe: pk.subframe,

                    rt_bounds_frames: (l, r),
                    frame_id_bounds: (lo_fid.min(hi_fid), lo_fid.max(hi_fid)),
                    id: rt_peak_id(&pk),
                }
            ];
        }
        return Vec::new();
    }

    // ------------------------------------------------------------
    // 7) Normal multi-peak mapping: map clamped indices → global RT indices
    // ------------------------------------------------------------
    let n_frames = frames_sorted.len();

    base.into_iter()
        .map(|r0| {
            let local_left  =
                r0.left_x.floor().clamp(0.0, (n_clamped - 1) as f32) as usize;
            let local_right =
                r0.right_x.ceil().clamp(0.0, (n_clamped - 1) as f32) as usize;

            let global_left = allow_lo + local_left;
            let global_right = allow_hi.min(allow_lo + local_right).min(n_frames - 1);

            let lo_fid = rt_ctx.frame_ids_sorted[global_left];
            let hi_fid = rt_ctx.frame_ids_sorted[global_right];

            let mut r = RtPeak1D {
                parent_im_id: Some(im_peak.id),
                mz_row: im_peak.mz_row,
                mz_center: im_peak.mz_center,
                mz_bounds: im_peak.mz_bounds,
                window_group: im_peak.window_group,

                rt_idx: allow_lo + r0.rt_idx,
                rt_sec: r0.rt_sec,
                apex_smoothed: r0.apex_smoothed,
                apex_raw: r0.apex_raw,
                prominence: r0.prominence,
                left_x: r0.left_x + allow_lo as f32,
                right_x: r0.right_x + allow_lo as f32,
                width_frames: r0.width_frames,
                area_raw: r0.area_raw,
                subframe: r0.subframe,

                rt_bounds_frames: (global_left, global_right),
                frame_id_bounds: (lo_fid.min(hi_fid), lo_fid.max(hi_fid)),
                id: 0,
            };
            r.id = rt_peak_id(&r);
            r
        })
        .collect()
}

// add param
pub fn expand_many_im_peaks_along_rt(
    im_peaks: &[ImPeak1D],
    frames_sorted: &[FrameBinView],
    ctx: RtTraceCtx<'_>,
    rt_scale: &MzScale,              // <-- NEW
    p: RtExpandParams,
) -> Vec<Vec<RtPeak1D>> {
    if im_peaks.is_empty() { return Vec::new(); }

    #[cfg(debug_assertions)]
    {
        let first_g = im_peaks[0].window_group;
        let same = im_peaks.iter().all(|x| x.window_group == first_g);
        debug_assert!(same, "expand_many_im_peaks_along_rt: mixed window_group in batch");
    }

    im_peaks.par_iter()
        .map(|im| expand_im_peak_along_rt(im, frames_sorted, ctx, rt_scale, p))
        .collect()
}

// flat variant
pub fn expand_many_im_peaks_along_rt_flat(
    im_peaks: &[ImPeak1D],
    frames_sorted: &[FrameBinView],
    ctx: RtTraceCtx<'_>,
    rt_scale: &MzScale,              // <-- NEW
    p: RtExpandParams,
) -> Vec<RtPeaksForIm> {
    if im_peaks.is_empty() { return Vec::new(); }

    (0..im_peaks.len()).into_par_iter()
        .map(|i| {
            let im = &im_peaks[i];
            let peaks = expand_im_peak_along_rt(im, frames_sorted, ctx, rt_scale, p);
            RtPeaksForIm { im_index: i, im_id: im.id, peaks }
        })
        .collect()
}
// 3) Variant that returns a flat stream with the input index for downstream use
#[derive(Clone, Debug)]
pub struct RtPeaksForIm {
    pub im_index: usize,       // index into the input im_peaks slice
    pub im_id: PeakId,         // parent id for convenience
    pub peaks: Vec<RtPeak1D>,
}


#[inline]
fn t_at_index_frac(t: &[f32], x: f32) -> f32 {
    if t.is_empty() { return 0.0; }
    if x <= 0.0 { return t[0]; }
    let n1 = (t.len() - 1) as f32;
    if x >= n1 { return t[t.len()-1]; }
    let j0 = x.floor() as usize;
    let j1 = (j0 + 1).min(t.len()-1);
    let frac = x - j0 as f32;
    (1.0 - frac) * t[j0] + frac * t[j1]
}

// robust-ish effective dt for converting σ_sec → σ_frames
#[inline]
fn effective_dt(rt_times: &[f32]) -> f32 {
    if rt_times.len() < 2 { return 1.0; }
    let mut d: Vec<f32> = rt_times.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    d.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    d[d.len()/2].max(1e-3) // median, clamp
}