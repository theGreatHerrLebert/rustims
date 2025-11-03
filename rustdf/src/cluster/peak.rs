use std::sync::Arc;
use rayon::prelude::*;
use mscore::timstof::frame::TimsFrame;
use crate::cluster::utility::{fallback_rt_peak_from_trace, quad_subsample, rt_peak_id, smooth_vector_gaussian, trapezoid_area_fractional, MzScale};

#[derive(Clone, Debug)]
pub struct RtFrames {
    pub frames: Vec<FrameBinView>, // RT-sorted CSR rows for this provenance
    pub frame_ids: Vec<u32>,       // same order as frames
    pub rt_times: Vec<f32>,        // same order as frames
}

impl RtFrames {
    #[inline]
    pub fn ctx(&self) -> RtTraceCtx<'_> {
        RtTraceCtx {
            frame_ids_sorted: &self.frame_ids,
            rt_times_sec: Some(&self.rt_times),
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
}

#[derive(Clone, Copy, Debug)]
pub struct RtTraceCtx<'a> {
    pub frame_ids_sorted: &'a [u32],   // same order as frames_sorted
    pub rt_times_sec: Option<&'a [f32]>,
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
pub type PeakId = u64;


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
    pub id: PeakId,
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
    pub data: Vec<f32>,
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
        // scan filter
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

/// Returns (rt_times_sec, rt_trace_intensity)
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

pub fn find_rt_peaks(
    y_smoothed: &[f32],
    y_raw: &[f32],
    rt_times: Option<&[f32]>,
    min_prom: f32,
    min_sep_frames: usize,
    min_width_frames: usize,
) -> Vec<RtLocalPeak> {
    let n = y_smoothed.len();
    if n < 3 || y_raw.len() != n { return Vec::new(); }

    let row_max = y_raw.iter().copied().fold(0.0f32, f32::max);
    if row_max < min_prom { return Vec::new(); }

    // 1) strict local maxima on smoothed
    let mut cands = Vec::new();
    for i in 1..n - 1 {
        let yi = y_smoothed[i];
        if yi > y_smoothed[i - 1] && yi >= y_smoothed[i + 1] {
            cands.push(i);
        }
    }

    let mut peaks: Vec<RtLocalPeak> = Vec::new();
    for &i in &cands {
        let apex = y_smoothed[i];

        // 2) prominence baseline (bounded by taller peaks)
        let mut l = i; let mut left_min = apex;
        while l > 0 { l -= 1; left_min = left_min.min(y_smoothed[l]); if y_smoothed[l] > apex { break; } }
        let mut r = i; let mut right_min = apex;
        while r + 1 < n { r += 1; right_min = right_min.min(y_smoothed[r]); if y_smoothed[r] > apex { break; } }

        let baseline = left_min.max(right_min);
        let prom = apex - baseline;
        if prom < min_prom { continue; }

        // 3) half-prom crossings (fractional)
        let half = baseline + 0.5 * prom;

        // left crossing
        let mut wl = i;
        while wl > 0 && y_smoothed[wl] > half { wl -= 1; }
        let left_x = if wl < i && wl + 1 < n {
            let y0 = y_smoothed[wl]; let y1 = y_smoothed[wl + 1];
            wl as f32 + if y1 != y0 { (half - y0) / (y1 - y0) } else { 0.0 }
        } else { wl as f32 };

        // right crossing
        let mut wr = i;
        while wr + 1 < n && y_smoothed[wr] > half { wr += 1; }
        let right_x = if wr > i && wr < n {
            let y0 = y_smoothed[wr - 1]; let y1 = y_smoothed[wr];
            (wr - 1) as f32 + if y1 != y0 { (half - y0) / (y1 - y0) } else { 0.0 }
        } else { wr as f32 };

        let width = (right_x - left_x).max(0.0);
        let width_frames = width.round() as usize;
        if width_frames < min_width_frames { continue; }

        // 4) sub-frame apex offset (quadratic interpolation)
        let sub = if i > 0 && i + 1 < n {
            quad_subsample(y_smoothed[i - 1], y_smoothed[i], y_smoothed[i + 1]).clamp(-0.5, 0.5)
        } else { 0.0 };

        // 5) NMS by minimum separation in frames (keep taller on smoothed)
        if let Some(last) = peaks.last() {
            if i.abs_diff(last.rt_idx) < min_sep_frames {
                if apex <= last.apex_smoothed { continue; }
                let _ = peaks.pop();
            }
        }

        // 6) area on raw between fractional bounds
        let area = trapezoid_area_fractional(y_raw, left_x.max(0.0), right_x.min((n - 1) as f32));

        // rt_sec if provided (sub-sample around apex)
        let rt_sec = rt_times.map(|t| {
            if i + 1 < t.len() && i > 0 {
                let base = i as f32 + sub;
                let j0 = base.floor().clamp(0.0, (t.len() - 1) as f32) as usize;
                let j1 = (j0 + 1).min(t.len() - 1);
                let frac = base - j0 as f32;
                (1.0 - frac) * t[j0] + frac * t[j1]
            } else {
                t[i]
            }
        });

        peaks.push(RtLocalPeak {
            rt_idx: i,
            rt_sec,
            apex_smoothed: apex,
            apex_raw: y_raw[i],
            prominence: prom,
            left_x,
            right_x,
            width_frames,
            area_raw: area,
            subframe: sub,
        });
    }
    peaks
}

#[derive(Clone, Copy, Debug)]
pub struct RtExpandParams {
    pub bin_pad: usize,          // include neighboring m/z bins around peak.mz_row (0–2)
    pub smooth_sigma: f32,       // in frames, e.g. 1.0–1.5
    pub smooth_trunc: f32,       // e.g. 3.0
    pub min_prom: f32,           // absolute units after smoothing
    pub min_sep_frames: usize,   // 2–4
    pub min_width_frames: usize, // 2–3
    pub fallback_if_frames_lt: usize, // e.g. 3
    pub fallback_frac_width: f32,     // width at a small fraction of apex (e.g., 0.10)
}

pub fn expand_im_peak_along_rt(
    im_peak: &ImPeak1D,
    frames_sorted: &[FrameBinView], // RT-ordered frames for this DIA group
    ctx: RtTraceCtx<'_>,
    p: RtExpandParams,
) -> Vec<RtPeak1D> {

    let trace_raw = rt_trace_for_im_peak(
        frames_sorted, im_peak.mz_row, p.bin_pad, im_peak.left, im_peak.right,
    );
    if trace_raw.is_empty() || !trace_raw.iter().any(|&x| x.is_finite() && x > 0.0) {
        return Vec::new();
    }

    // --- NEW: if the parent IM peak’s RT support is too short, skip fitting
    let im_rt_span = im_peak.rt_bounds.1.saturating_sub(im_peak.rt_bounds.0) + 1;
    if im_rt_span < p.fallback_if_frames_lt {
        if let Some(pk) = fallback_rt_peak_from_trace(&trace_raw, ctx, im_peak, p.fallback_frac_width) {
            return vec![pk];
        } else {
            return Vec::new();
        }
    }

    // normal path (smooth + half-prom)
    let mut trace_smooth = trace_raw.clone();
    smooth_vector_gaussian(&mut trace_smooth[..], p.smooth_sigma, p.smooth_trunc);

    let base = find_rt_peaks(
        &trace_smooth, &trace_raw, ctx.rt_times_sec,
        p.min_prom, p.min_sep_frames, p.min_width_frames,
    );

    // map into enriched RtPeak1D
    let n_frames = frames_sorted.len();
    let (fid_lo, fid_hi) = im_peak.frame_id_bounds;
    base.into_iter().map(|r0| {
        // compute integer bounds from fractional
        let l = r0.left_x.floor().clamp(0.0, (n_frames.saturating_sub(1)) as f32) as usize;
        let rr = r0.right_x.ceil().clamp(0.0, (n_frames.saturating_sub(1)) as f32) as usize;
        let rt_bounds_frames = (l, rr);

        // map to frame ids using ctx
        let frame_id_bounds = if ctx.frame_ids_sorted.is_empty() {
            (fid_lo, fid_hi) // fallback to IM parent
        } else {
            let lo = ctx.frame_ids_sorted[l.min(ctx.frame_ids_sorted.len()-1)];
            let hi = ctx.frame_ids_sorted[rr.min(ctx.frame_ids_sorted.len()-1)];
            (lo.min(hi), lo.max(hi))
        };

        let mut r = RtPeak1D {
            // geometry carried over
            rt_idx: r0.rt_idx,
            rt_sec: r0.rt_sec,
            apex_smoothed: r0.apex_smoothed,
            apex_raw: r0.apex_raw,
            prominence: r0.prominence,
            left_x: r0.left_x,
            right_x: r0.right_x,
            width_frames: r0.width_frames,
            area_raw: r0.area_raw,
            subframe: r0.subframe,

            // new metadata
            rt_bounds_frames,
            frame_id_bounds,
            window_group: im_peak.window_group,

            mz_row: im_peak.mz_row,
            mz_center: im_peak.mz_center,
            mz_bounds: im_peak.mz_bounds,

            parent_im_id: Some(im_peak.id),
            id: 0,
        };
        r.id = rt_peak_id(&r);
        r
    }).collect()
}

// 2) Parallel expansion: outputs Vec<Vec<RtPeak1D>> aligned to input order
pub fn expand_many_im_peaks_along_rt(
    im_peaks: &[ImPeak1D],             // all from precursor space OR same window_group
    frames_sorted: &[FrameBinView],    // RT-ordered frames for that provenance
    ctx: RtTraceCtx<'_>,
    p: RtExpandParams,
) -> Vec<Vec<RtPeak1D>> {
    if im_peaks.is_empty() {
        return Vec::new();
    }

    // Optional sanity: enforce single-group assumption (None == precursor space)
    #[cfg(debug_assertions)]
    {
        let first_g = im_peaks[0].window_group;
        let same = im_peaks.iter().all(|x| x.window_group == first_g);
        debug_assert!(same, "expand_many_im_peaks_along_rt: mixed window_group in batch");
    }

    // Parallel map, preserving order
    im_peaks.par_iter()
        .map(|im| expand_im_peak_along_rt(im, frames_sorted, ctx, p))
        .collect()
}

// 3) Variant that returns a flat stream with the input index for downstream use
#[derive(Clone, Debug)]
pub struct RtPeaksForIm {
    pub im_index: usize,       // index into the input im_peaks slice
    pub im_id: PeakId,         // parent id for convenience
    pub peaks: Vec<RtPeak1D>,
}

pub fn expand_many_im_peaks_along_rt_flat(
    im_peaks: &[ImPeak1D],
    frames_sorted: &[FrameBinView],
    ctx: RtTraceCtx<'_>,
    p: RtExpandParams,
) -> Vec<RtPeaksForIm> {
    if im_peaks.is_empty() {
        return Vec::new();
    }

    (0..im_peaks.len()).into_par_iter()
        .map(|i| {
            let im = &im_peaks[i];
            let peaks = expand_im_peak_along_rt(im, frames_sorted, ctx, p);
            RtPeaksForIm { im_index: i, im_id: im.id, peaks }
        })
        .collect()
}