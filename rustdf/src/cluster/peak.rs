use rayon::prelude::*;
use std::sync::Arc;
use mscore::timstof::frame::TimsFrame;
use crate::cluster::utility::{fallback_rt_peak_from_trace, find_im_peaks_row, quad_subsample, rt_peak_id, smooth_vector_gaussian, trapezoid_area_fractional, MobilityFn, TofScale};

// Stable 64-bit id for peaks
pub type PeakId = i64;

#[derive(Clone, Debug)]
pub struct ImPeak1D {
    pub tof_row: usize,                  // ABSOLUTE TOF bin row (global scale)
    pub tof_center: i32,                 // center TOF index
    pub tof_bounds: (i32, i32),          // min, max TOF index
    pub rt_bounds: (usize, usize),       // columns [lo, hi] in current RT grid
    pub frame_id_bounds: (u32, u32),     // materialized for robustness
    pub window_group: Option<u32>,       // DIA group provenance

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
    pub width_scans: usize,              // interpreted as ABSOLUTE width
    pub area_raw: f32,
    pub subscan: f32,
    pub id: PeakId,
}

#[derive(Clone, Debug)]
pub struct FrameBinView {
    pub unique_bins: Vec<usize>,
    pub offsets: Vec<usize>,
    pub scan_idx: Vec<u32>,              // ABSOLUTE scan indices
    pub intensity: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct TofScanGrid {
    pub scans: Vec<usize>,
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub data_raw: Option<Vec<f32>>,
    pub scale: TofScale,                 // TOF scale internally
    pub row_offset: Option<usize>,
}

impl TofScanGrid {
    #[inline]
    pub fn abs_row(&self, r: usize) -> usize {
        self.row_offset.map(|o| r + o).unwrap_or(r)
    }

    #[inline]
    pub fn tof_center_for_row(&self, r: usize) -> f32 {
        self.scale.center(self.abs_row(r))
    }

    #[inline]
    pub fn tof_bounds_for_row(&self, r: usize) -> (f32, f32) {
        let a = self.abs_row(r);
        (self.scale.edges[a], self.scale.edges[a + 1])
    }
}

#[derive(Clone, Debug)]
pub struct TofScanWindowGrid {
    pub rt_range_frames: (usize, usize),
    pub rt_range_sec:    (f32, f32),
    pub frame_id_bounds: (u32, u32),
    pub window_group:    Option<u32>,

    pub scale: Arc<TofScale>,            // TOF scale

    pub scans: Vec<usize>,
    pub data: Option<Vec<f32>>,
    pub rows: usize,
    pub cols: usize,
    pub data_raw: Option<Vec<f32>>,
}

impl TofScanWindowGrid {
    #[inline]
    pub fn tof_center_for_row(&self, r: usize) -> f32 {
        self.scale.center(r)
    }
    #[inline]
    pub fn tof_bounds_for_row(&self, r: usize) -> (f32, f32) {
        (self.scale.edges[r], self.scale.edges[r + 1])
    }
}

pub fn build_frame_bin_view(
    fr: TimsFrame,
    scale: &TofScale,          // TOF-based CSR scale
    global_num_scans: usize,
) -> FrameBinView {
    let n = fr.tof.len();
    let mut bins_idx: Vec<usize> = Vec::with_capacity(n);
    let mut scans_u:  Vec<u32>   = Vec::with_capacity(n);
    let mut intens:   Vec<f32>   = Vec::with_capacity(n);

    let scans_vec: &Vec<i32> = &fr.scan;
    let tofs: &Vec<i32>      = &fr.tof;

    for i in 0..n {
        let tof = tofs[i];
        if let Some(idx) = scale.index_of_tof(tof) {
            bins_idx.push(idx);

            let s_val = scans_vec[i];
            debug_assert!(s_val >= 0, "Negative scan index in frame {}", fr.frame_id);
            let s_u32: u32 = u32::try_from(s_val).expect("scan index does not fit u32");
            scans_u.push((s_u32 as usize).min(global_num_scans.saturating_sub(1)) as u32);

            intens.push(fr.ims_frame.intensity[i] as f32);
        }
    }

    // sort by bin index and build CSR
    let mut idx: Vec<usize> = (0..bins_idx.len()).collect();
    idx.sort_unstable_by_key(|&i| bins_idx[i]);

    let mut unique_bins: Vec<usize> = Vec::new();
    let mut counts: Vec<usize>      = Vec::new();
    let mut scan_sorted: Vec<u32>   = Vec::with_capacity(idx.len());
    let mut inten_sorted: Vec<f32>  = Vec::with_capacity(idx.len());

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
    for c in &counts {
        offsets.push(offsets.last().unwrap() + *c);
    }

    FrameBinView {
        unique_bins,
        offsets,
        scan_idx: scan_sorted,
        intensity: inten_sorted,
    }
}

#[inline]
fn sum_frame_bins_scans(
    fbv: &FrameBinView,
    bin_lo: usize,
    bin_hi: usize,
    scan_lo: usize,
    scan_hi: usize,
) -> f32 {
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

        let lo = fbv.offsets[i];
        let hi = fbv.offsets[i + 1];
        let scans = &fbv.scan_idx[lo..hi];
        let ints  = &fbv.intensity[lo..hi];

        for (s, val) in scans.iter().zip(ints.iter()) {
            let s = *s as usize;
            if s >= scan_lo && s <= scan_hi {
                acc += *val;
            }
        }
        i += 1;
    }
    acc
}

/// Returns an RT trace for a single IM peak using bin padding around a TOF row.
pub fn rt_trace_for_im_peak(
    frames: &[FrameBinView],        // RT-sorted, same group
    tof_row: usize,
    bin_pad: usize,                 // e.g., 0 or 1
    scan_lo: usize,
    scan_hi: usize,
) -> Vec<f32> {
    let bin_lo = tof_row.saturating_sub(bin_pad);
    let bin_hi = tof_row.saturating_add(bin_pad);
    let mut v = Vec::with_capacity(frames.len());
    for fbv in frames {
        v.push(sum_frame_bins_scans(fbv, bin_lo, bin_hi, scan_lo, scan_hi));
    }
    v
}

#[derive(Clone, Debug)]
pub struct RtFrames {
    pub frames: Vec<FrameBinView>,
    pub frame_ids: Vec<u32>,
    pub rt_times: Vec<f32>,
    pub scale: Arc<TofScale>,
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
        self.frames.len() == self.frame_ids.len()
            && self.frame_ids.len() == self.rt_times.len()
    }
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

    // TOF context carried from the IM parent
    pub tof_row: usize,
    pub tof_center: i32,
    pub tof_bounds: (i32, i32),

    // linkage
    pub parent_im_id: Option<PeakId>,
    pub id: PeakId,
}

#[derive(Clone, Debug)]
pub struct RtPeaksForIm {
    pub im_index: usize,       // index into the input im_peaks slice
    pub im_id: PeakId,         // parent id for convenience
    pub peaks: Vec<RtPeak1D>,
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
pub struct RtExpandParams {
    pub bin_pad: usize,
    pub smooth_sigma_sec: f32,
    pub smooth_trunc_k: f32,
    pub min_prom: f32,
    pub min_sep_sec: f32,
    pub min_width_sec: f32,
    pub fallback_if_frames_lt: usize,
    pub fallback_frac_width: f32,
    /// NEW: symmetric padding in *frame indices* around the IM peak’s frame_id_bounds
    /// when building the RT trace, to avoid cutting off partially overlapping peaks.
    pub rt_pad_frames: usize,
}

#[inline]
pub fn rt_trace_for_im_peak_by_bounds(
    frames: &[FrameBinView],
    rt_scale: &TofScale,            // RtFrames.scale
    tof_bounds: (i32, i32),         // im_peak.tof_bounds
    extra_bins_pad: usize,          // 0–2
    scan_lo: usize,
    scan_hi: usize,
) -> Vec<f32> {
    // Map physical TOF bounds into the RT CSR scale, then pad by bins.
    let (mut bin_l, mut bin_r) = rt_scale.index_range_for_tof_window(tof_bounds.0, tof_bounds.1);
    bin_l = bin_l.saturating_sub(extra_bins_pad);
    bin_r = bin_r.saturating_add(extra_bins_pad);

    let mut v = Vec::with_capacity(frames.len());
    for fbv in frames {
        v.push(sum_frame_bins_scans(fbv, bin_l, bin_r, scan_lo, scan_hi));
    }
    v
}

pub fn expand_im_peak_along_rt(
    im_peak: &ImPeak1D,
    frames_sorted: &[FrameBinView],
    rt_ctx: RtTraceCtx<'_>,
    tof_scale: &TofScale,
    p: RtExpandParams,
) -> Vec<RtPeak1D> {
    // 1) Map absolute frame IDs → local RT index range.
    let (fid_lo_abs, fid_hi_abs) = im_peak.frame_id_bounds;

    let mut allow_lo = match rt_ctx.frame_ids_sorted.binary_search(&fid_lo_abs) {
        Ok(i) => i,
        Err(_) => return Vec::new(),
    };
    let mut allow_hi = match rt_ctx.frame_ids_sorted.binary_search(&fid_hi_abs) {
        Ok(i) => i,
        Err(_) => return Vec::new(),
    };
    if allow_lo > allow_hi {
        std::mem::swap(&mut allow_lo, &mut allow_hi);
    }

    let n_frames = frames_sorted.len();
    if n_frames == 0 {
        return Vec::new();
    }

    // NEW: symmetric padding in frame space to avoid cutting off RT tails
    let pad = p.rt_pad_frames.min(n_frames.saturating_sub(1));
    allow_lo = allow_lo.saturating_sub(pad);
    allow_hi = (allow_hi + pad).min(n_frames.saturating_sub(1));

    if allow_lo >= n_frames || allow_hi >= n_frames {
        return Vec::new();
    }

    // 2) TOF-window RT trace
    let trace_raw_full = rt_trace_for_im_peak_by_bounds(
        frames_sorted,
        tof_scale,
        im_peak.tof_bounds,
        p.bin_pad,
        im_peak.left_abs,
        im_peak.right_abs,
    );
    if trace_raw_full.is_empty() {
        return Vec::new();
    }

    // 3) Clamp to allowed region
    let trace_raw = &trace_raw_full[allow_lo..=allow_hi];
    let rt_times_clamped = &rt_ctx.rt_times_sec[allow_lo..=allow_hi];
    let n_clamped = trace_raw.len();
    if n_clamped == 0 || !trace_raw.iter().any(|x| *x > 0.0) {
        return Vec::new();
    }

    // 4) Fallback branch – unchanged except for TOF fields in RtPeak1D
    if n_clamped < p.fallback_if_frames_lt {
        if let Some(pk) =
            fallback_rt_peak_from_trace(trace_raw, rt_ctx, im_peak, p.fallback_frac_width)
        {
            let l = allow_lo + pk.left_x.floor().clamp(0.0, (n_clamped - 1) as f32) as usize;
            let r = allow_lo + pk.right_x.ceil().clamp(0.0, (n_clamped - 1) as f32) as usize;
            let lo_fid = rt_ctx.frame_ids_sorted[l];
            let hi_fid = rt_ctx.frame_ids_sorted[r];

            return vec![RtPeak1D {
                parent_im_id: Some(im_peak.id),
                window_group: im_peak.window_group,

                tof_row: im_peak.tof_row,
                tof_center: im_peak.tof_center,
                tof_bounds: im_peak.tof_bounds,

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
            }];
        }
        return Vec::new();
    }

    // 5) Smooth clamped trace
    let dt = effective_dt(rt_times_clamped);
    let sigma_frames = (p.smooth_sigma_sec / dt).max(0.75);
    let mut trace_smooth = trace_raw.to_vec();
    smooth_vector_gaussian(&mut trace_smooth[..], sigma_frames, p.smooth_trunc_k);

    // (optional: a heavier smooth for baseline only could be added here)

    // 6) Peak finding (updated)
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
            let l = allow_lo + pk.left_x.floor().clamp(0.0, (n_clamped - 1) as f32) as usize;
            let r = allow_lo + pk.right_x.ceil().clamp(0.0, (n_clamped - 1) as f32) as usize;
            let lo_fid = rt_ctx.frame_ids_sorted[l];
            let hi_fid = rt_ctx.frame_ids_sorted[r];

            return vec![RtPeak1D {
                parent_im_id: Some(im_peak.id),
                window_group: im_peak.window_group,

                tof_row: im_peak.tof_row,
                tof_center: im_peak.tof_center,
                tof_bounds: im_peak.tof_bounds,

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
            }];
        }
        return Vec::new();
    }

    // 7) Normal multi-peak mapping
    let n_frames_total = frames_sorted.len();

    base.into_iter()
        .map(|r0| {
            let local_left =
                r0.left_x.floor().clamp(0.0, (n_clamped - 1) as f32) as usize;
            let local_right =
                r0.right_x.ceil().clamp(0.0, (n_clamped - 1) as f32) as usize;

            let global_left = allow_lo + local_left;
            let global_right = allow_hi
                .min(allow_lo + local_right)
                .min(n_frames_total - 1);

            let lo_fid = rt_ctx.frame_ids_sorted[global_left];
            let hi_fid = rt_ctx.frame_ids_sorted[global_right];

            let mut r = RtPeak1D {
                parent_im_id: Some(im_peak.id),
                window_group: im_peak.window_group,

                tof_row: im_peak.tof_row,
                tof_center: im_peak.tof_center,
                tof_bounds: im_peak.tof_bounds,

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

pub fn expand_many_im_peaks_along_rt(
    im_peaks: &[ImPeak1D],
    frames_sorted: &[FrameBinView],
    ctx: RtTraceCtx<'_>,
    tof_scale: &TofScale,      // <-- was &MzScale
    p: RtExpandParams,
) -> Vec<Vec<RtPeak1D>> {
    if im_peaks.is_empty() { return Vec::new(); }

    #[cfg(debug_assertions)]
    {
        let first_g = im_peaks[0].window_group;
        let same = im_peaks.iter().all(|x| x.window_group == first_g);
        debug_assert!(same, "expand_many_im_peaks_along_rt: mixed window_group in batch");
    }

    im_peaks
        .par_iter()
        .map(|im| expand_im_peak_along_rt(im, frames_sorted, ctx, tof_scale, p))
        .collect()
}

pub fn expand_many_im_peaks_along_rt_flat(
    im_peaks: &[ImPeak1D],
    frames_sorted: &[FrameBinView],
    ctx: RtTraceCtx<'_>,
    tof_scale: &TofScale,      // <-- was &MzScale
    p: RtExpandParams,
) -> Vec<RtPeaksForIm> {
    if im_peaks.is_empty() { return Vec::new(); }

    (0..im_peaks.len())
        .into_par_iter()
        .map(|i| {
            let im = &im_peaks[i];
            let peaks = expand_im_peak_along_rt(im, frames_sorted, ctx, tof_scale, p);
            RtPeaksForIm { im_index: i, im_id: im.id, peaks }
        })
        .collect()
}

// =================== RT peak detection helpers ===================

use std::cmp::Ordering;

fn robust_noise_level(y: &[f32]) -> f32 {
    let n = y.len();
    if n < 3 {
        return 0.0;
    }

    // |Δy| between neighbouring points
    let mut diffs = Vec::with_capacity(n.saturating_sub(1));
    for w in y.windows(2) {
        diffs.push((w[1] - w[0]).abs());
    }
    if diffs.is_empty() {
        return 0.0;
    }

    // median of diffs via selection, not full sort
    let mid = diffs.len() / 2;
    diffs.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    });

    diffs[mid]
}

fn nms_by_time(mut peaks: Vec<RtLocalPeak>, min_sep_sec: f32) -> Vec<RtLocalPeak> {
    if peaks.is_empty() { return peaks; }

    // sort by apex height descending
    peaks.sort_by(|a, b| {
        b.apex_smoothed
            .partial_cmp(&a.apex_smoothed)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected: Vec<RtLocalPeak> = Vec::new();
    'outer: for p in peaks {
        let t_p = p.rt_sec.unwrap_or(0.0);
        for q in &selected {
            let t_q = q.rt_sec.unwrap_or(0.0);
            if (t_p - t_q).abs() < min_sep_sec {
                // too close to a stronger peak
                continue 'outer;
            }
        }
        selected.push(p);
    }

    // re-sort by RT for nicer downstream behaviour
    selected.sort_by(|a, b| {
        a.rt_sec
            .unwrap_or(0.0)
            .partial_cmp(&b.rt_sec.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    selected
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

    // noise-adaptive prominence threshold
    let noise = robust_noise_level(y_smoothed);
    let min_prom_eff = if noise > 0.0 {
        min_prom.max(3.0 * noise)
    } else {
        min_prom
    };

    // 1) candidates
    let mut cands = Vec::with_capacity(n / 4);
    for i in 1..n-1 {
        let yi = y_smoothed[i];
        if yi > y_smoothed[i-1] && yi >= y_smoothed[i+1] { cands.push(i); }
    }

    let mut peaks: Vec<RtLocalPeak> = Vec::with_capacity(n / 4);
    for &i in &cands {
        let apex = y_smoothed[i];

        // 2) prominence baseline
        let mut l = i; let mut left_min = apex;
        while l > 0 { l -= 1; left_min = left_min.min(y_smoothed[l]); if y_smoothed[l] > apex { break; } }
        let mut r = i; let mut right_min = apex;
        while r + 1 < n { r += 1; right_min = right_min.min(y_smoothed[r]); if y_smoothed[r] > apex { break; } }

        let baseline = left_min.max(right_min);
        let prom = apex - baseline;
        if prom < min_prom_eff { continue; }

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

        // 5) area under raw
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

    if peaks.is_empty() {
        return peaks;
    }

    // total area for relative filters (no need for fractional helper)
    let mut total_area = 0.0f32;
    for j in 0..(n - 1) {
        total_area += 0.5 * (y_raw[j] + y_raw[j + 1]);
    }

    // time-based NMS across all peaks
    let mut peaks = nms_by_time(peaks, min_sep_sec);
    if peaks.is_empty() {
        return Vec::new();
    }

    // area-based pruning of tiny shoulders
    let max_area = peaks
        .iter()
        .map(|p| p.area_raw)
        .fold(0.0f32, f32::max)
        .max(1e-6);

    let min_area_frac_trace = 0.02; // 2% of total trace area
    let min_area_frac_best  = 0.05; // 5% of best peak area

    peaks.retain(|p| {
        let frac_trace = p.area_raw / total_area.max(1e-6);
        let frac_best  = p.area_raw / max_area;
        frac_trace >= min_area_frac_trace && frac_best >= min_area_frac_best
    });

    peaks
}

// robust-ish effective dt for converting σ_sec → σ_frames
#[inline]
fn effective_dt(rt_times: &[f32]) -> f32 {
    if rt_times.len() < 2 { return 1.0; }
    let mut d: Vec<f32> = rt_times.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    d.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    d[d.len()/2].max(1e-3) // median, clamp
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

#[derive(Clone, Debug)]
pub struct TofRtGrid {
    /// [lo, hi] in local RT frame indices (0..cols-1)
    pub rt_range_frames: (usize, usize),
    /// [lo, hi] in seconds, from rt_times
    pub rt_range_sec: (f32, f32),
    /// [lo, hi] in absolute frame_ids
    pub frame_id_bounds: (u32, u32),

    /// Optional DIA window group; None for precursor grid
    pub window_group: Option<u32>,

    /// TOF axis definition (shared with RtFrames)
    pub scale: Arc<TofScale>,

    /// RT axis (seconds), len = cols
    pub rt_times: Vec<f32>,
    /// Absolute frame IDs, len = cols
    pub frame_ids: Vec<u32>,

    /// rows = number of TOF bins; cols = number of frames
    pub rows: usize,
    pub cols: usize,

    /// Dense matrix, row-major: data[row * cols + col],
    /// where row = TOF bin index, col = RT frame index.
    pub data: Vec<f32>,
}

/// Build a dense TOF×RT grid (full matrix, summed over *all* scans).
///
/// - rows = rt_frames.scale.num_bins() (TOF bins)
/// - cols = rt_frames.frames.len()     (RT frames)
/// - data[row * cols + col] = sum intensity in that (bin, frame) over all scans.
pub fn build_tof_rt_grid_full(rt_frames: &RtFrames, window_group: Option<u32>) -> TofRtGrid {
    assert!(rt_frames.is_consistent(), "RtFrames inconsistent");

    let cols = rt_frames.frames.len();
    let rows = rt_frames.scale.num_bins();

    let mut data = vec![0.0f32; rows.saturating_mul(cols.max(1))];

    for (col, fbv) in rt_frames.frames.iter().enumerate() {
        let ub = &fbv.unique_bins;
        if ub.is_empty() {
            continue;
        }

        for b_idx in 0..ub.len() {
            let bin = ub[b_idx];
            if bin >= rows {
                continue; // defensive, should not happen if scales match
            }

            let lo = fbv.offsets[b_idx];
            let hi = fbv.offsets[b_idx + 1];

            let mut sum = 0.0f32;
            for k in lo..hi {
                sum += fbv.intensity[k];
            }

            let idx = bin * cols + col; // row-major
            data[idx] += sum;
        }
    }

    let rt_range_frames = if cols > 0 { (0, cols - 1) } else { (0, 0) };
    let rt_range_sec = if cols > 0 {
        (rt_frames.rt_times[0], *rt_frames.rt_times.last().unwrap())
    } else {
        (0.0, 0.0)
    };

    let frame_id_bounds = if rt_frames.frame_ids.is_empty() {
        (0, 0)
    } else {
        (
            *rt_frames.frame_ids.first().unwrap(),
            *rt_frames.frame_ids.last().unwrap(),
        )
    };

    TofRtGrid {
        rt_range_frames,
        rt_range_sec,
        frame_id_bounds,
        window_group,
        scale: rt_frames.scale.clone(),
        rt_times: rt_frames.rt_times.clone(),
        frame_ids: rt_frames.frame_ids.clone(),
        rows,
        cols,
        data,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImDetectParams {
    /// Min prominence in intensity units
    pub min_prom: f32,
    /// NMS distance in scan units (absolute scan index space)
    pub min_distance_scans: usize,
    /// Minimum peak width (in scans)
    pub min_width_scans: usize,
    /// Optional Gaussian smoothing in scan direction
    pub smooth_sigma_scans: f32,
    pub smooth_trunc_k: f32,
}

pub fn detect_im_peaks_from_tof_scan_grid(
    grid: &TofScanGrid,
    rt_bounds_frames: (usize, usize),
    frame_id_bounds: (u32, u32),
    window_group: Option<u32>,
    mobility_of: MobilityFn,
    p: ImDetectParams,
) -> Vec<ImPeak1D> {
    use rayon::prelude::*;

    let rows = grid.rows;
    let cols = grid.cols;
    if rows == 0 || cols == 0 {
        return Vec::new();
    }

    let scans_axis = &grid.scans;
    debug_assert_eq!(scans_axis.len(), cols, "scans axis must match grid.cols");

    let data_raw = grid
        .data_raw
        .as_ref()
        .unwrap_or(&grid.data); // fall back to data if no separate raw

    debug_assert_eq!(
        data_raw.len(),
        rows * cols,
        "TofScanGrid data size mismatch"
    );

    // `mobility_of` is a function pointer (or None), so it's Copy and Send/Sync.
    let mobility_of_copy = mobility_of;

    // Parallel over rows; each row builds its own Vec<ImPeak1D>.
    let per_row: Vec<Vec<ImPeak1D>> = (0..rows)
        .into_par_iter()
        .map(|tof_row| {
            let offset = tof_row * cols;
            let row_raw = &data_raw[offset..offset + cols];

            // quick skip: empty row
            if row_raw.iter().all(|v| *v <= 0.0) {
                return Vec::new();
            }

            // Optional smoothing in scan-space (local buffer)
            let mut row_smooth = row_raw.to_vec();
            if p.smooth_sigma_scans > 0.0 && cols > 2 {
                smooth_vector_gaussian(&mut row_smooth, p.smooth_sigma_scans, p.smooth_trunc_k);
            }

            // TOF metadata
            let (tof_lo_f, tof_hi_f) = grid.tof_bounds_for_row(tof_row);
            let tof_center_f = grid.tof_center_for_row(tof_row);
            let tof_center = tof_center_f.round() as i32;
            let tof_bounds = (tof_lo_f.round() as i32, tof_hi_f.round() as i32);

            let tof_abs_row = grid.abs_row(tof_row);

            // IM detection with existing machinery
            find_im_peaks_row(
                &row_smooth,
                row_raw,
                tof_abs_row,
                tof_center,
                tof_bounds,
                rt_bounds_frames,
                frame_id_bounds,
                window_group,
                mobility_of_copy,
                p.min_prom,
                p.min_distance_scans,
                p.min_width_scans,
                scans_axis,
            )
        })
        .collect();

    // Flatten without extra allocations beyond final Vec
    let total_peaks: usize = per_row.iter().map(|v| v.len()).sum();
    let mut out = Vec::with_capacity(total_peaks);
    for mut v in per_row {
        out.append(&mut v);
    }
    out
}

pub fn detect_im_peaks_from_tof_scan_window(
    win: &TofScanWindowGrid,
    mobility_of: MobilityFn,
    p: ImDetectParams,
) -> Vec<ImPeak1D> {
    let rows = win.rows;
    let cols = win.cols;
    if rows == 0 || cols == 0 {
        return Vec::new();
    }

    // We require that the dense data is materialized
    let data = match &win.data {
        Some(d) => d,
        None => {
            // You can decide whether to panic, return empty, or lazily build here
            // For now, be strict:
            panic!("TofScanWindowGrid.data is None; materialize grid before IM detection");
        }
    };

    let data_raw = win.data_raw.as_ref().unwrap_or(data);
    if data_raw.len() != rows * cols {
        panic!("TofScanWindowGrid data size mismatch");
    }
    debug_assert_eq!(win.scans.len(), cols, "scan axis length mismatch");

    // Build a temporary TofScanGrid view
    let grid = TofScanGrid {
        scans: win.scans.clone(),
        data: data.clone(),
        rows,
        cols,
        data_raw: win.data_raw.clone(),
        scale: (*win.scale).clone(),
        row_offset: None,
    };

    detect_im_peaks_from_tof_scan_grid(
        &grid,
        win.rt_range_frames,
        win.frame_id_bounds,
        win.window_group,
        mobility_of,
        p,
    )
}

/// Dense TOF(bin) × scan(abs) grid summed over frames [rt_lo..=rt_hi].
pub fn build_tof_scan_grid_for_rt_window(
    rt_frames: &RtFrames,
    rt_lo: usize,
    rt_hi: usize,
    global_num_scans: usize,
    // optional: restrict to a TOF bin window to keep memory down:
    bin_lo: Option<usize>,
    bin_hi: Option<usize>,
) -> TofScanGrid {
    let scale = &*rt_frames.scale;
    let n_bins_total = scale.num_bins();

    let (b0, b1) = match (bin_lo, bin_hi) {
        (Some(lo), Some(hi)) => (lo.min(hi).min(n_bins_total.saturating_sub(1)),
                                 hi.max(lo).min(n_bins_total.saturating_sub(1))),
        _ => (0, n_bins_total.saturating_sub(1)),
    };
    let rows = b1 + 1 - b0;
    let cols = global_num_scans.max(1);

    let mut data = vec![0.0f32; rows.saturating_mul(cols)];
    let scans: Vec<usize> = (0..cols).collect(); // absolute scan axis

    let rt_lo = rt_lo.min(rt_frames.frames.len().saturating_sub(1));
    let rt_hi = rt_hi.min(rt_frames.frames.len().saturating_sub(1));
    let (rt_lo, rt_hi) = if rt_lo <= rt_hi { (rt_lo, rt_hi) } else { (rt_hi, rt_lo) };

    for fbv in &rt_frames.frames[rt_lo..=rt_hi] {
        let ub = &fbv.unique_bins;
        if ub.is_empty() { continue; }

        // Iterate bins in the *frame*, but only accumulate if bin ∈ [b0..b1]
        for (i_bin, &bin_abs) in ub.iter().enumerate() {
            if bin_abs < b0 || bin_abs > b1 { continue; }
            let row = bin_abs - b0;

            let lo = fbv.offsets[i_bin];
            let hi = fbv.offsets[i_bin + 1];

            for k in lo..hi {
                let s = fbv.scan_idx[k] as usize;
                if s < cols {
                    data[row * cols + s] += fbv.intensity[k];
                }
            }
        }
    }

    TofScanGrid {
        scans,
        data,
        rows,
        cols,
        data_raw: None,
        scale: (*rt_frames.scale).clone(),
        row_offset: Some(b0),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImExpandFromRtParams {
    pub bin_pad: usize,      // pad around the RT-peak’s TOF context
    pub im: ImDetectParams,  // your existing IM detection params
}

pub fn expand_rt_peak_along_im(
    rt_frames: &RtFrames,
    rt_peak: &RtPeak1D,
    global_num_scans: usize,
    mobility_of: MobilityFn,
    p: ImExpandFromRtParams,
) -> Vec<ImPeak1D> {
    let (rt_lo, rt_hi) = rt_peak.rt_bounds_frames;

    // define a compact TOF-bin window around the rt_peak’s TOF row/bounds
    // You have rt_peak.tof_row and rt_peak.tof_bounds (instrument idx).
    // If you prefer bin space: use rt_peak.tof_row as center bin, pad in bins.
    // If rt_peak.tof_bounds are instrument indices, map to bins via scale.index_range_for_tof_window.
    let scale = &*rt_frames.scale;
    let (mut bin_lo, mut bin_hi) =
        scale.index_range_for_tof_window(rt_peak.tof_bounds.0, rt_peak.tof_bounds.1);

    bin_lo = bin_lo.saturating_sub(p.bin_pad);
    bin_hi = bin_hi.saturating_add(p.bin_pad);

    let grid = build_tof_scan_grid_for_rt_window(
        rt_frames,
        rt_lo,
        rt_hi,
        global_num_scans,
        Some(bin_lo),
        Some(bin_hi),
    );

    // IMPORTANT: you want the produced ImPeak1D to carry *rt bounds* = (rt_lo, rt_hi)
    // and *frame_id_bounds* from rt_peak.frame_id_bounds.
    detect_im_peaks_from_tof_scan_grid(
        &grid,
        (rt_lo, rt_hi),
        rt_peak.frame_id_bounds,
        rt_peak.window_group,
        mobility_of,
        p.im,
    )
}