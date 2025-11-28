use mscore::timstof::frame::TimsFrame;
use crate::cluster::peak::{FrameBinView, ImPeak1D, PeakId, RtPeak1D, RtTraceCtx};
use std::hash::{Hash, Hasher};
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use rustc_hash::FxHashMap;

/// Optional mobility callback: scan -> 1/K0
pub type MobilityFn = Option<fn(scan: usize) -> f32>;

#[derive(Clone, Debug)]
pub struct TofScale {
    /// Smallest TOF index we include in the grid (inclusive).
    pub tof_min: i32,
    /// Largest TOF index we include in the grid (inclusive).
    pub tof_max: i32,
    /// Step between bins in TOF units (usually 1, 2, 4, …).
    pub tof_step: i32,
    /// Bin edges and centers in TOF space.
    pub edges: Vec<f32>,    // len = num_bins + 1
    pub centers: Vec<f32>,  // len = num_bins
}

impl TofScale {
    /// Build a TOF grid from a set of frames.
    ///
    /// - `tof_step = 1` => 1 bin per raw TOF index (max resolution).
    /// - Larger steps downsample TOF to reduce rows.
    pub fn build_from_frames(frames: &[TimsFrame], tof_step: i32) -> Option<Self> {
        assert!(tof_step > 0);

        let mut tof_min = i32::MAX;
        let mut tof_max = i32::MIN;

        for fr in frames {
            // ADAPT THIS if your TimsFrame layout is different
            let tofs: &[i32] = &fr.tof;
            for &t in tofs {
                if t < tof_min { tof_min = t; }
                if t > tof_max { tof_max = t; }
            }
        }

        if tof_min >= tof_max {
            return None;
        }

        // Snap min/max to step
        let tof_min_aligned = tof_min - (tof_min % tof_step);
        let tof_max_aligned = tof_max - (tof_max % tof_step);

        let num_bins = ((tof_max_aligned - tof_min_aligned) / tof_step + 1) as usize;
        if num_bins == 0 {
            return None;
        }

        let mut edges = Vec::with_capacity(num_bins + 1);
        let mut centers = Vec::with_capacity(num_bins);

        for i in 0..=num_bins {
            let tof_edge = tof_min_aligned + (i as i32) * tof_step;
            edges.push(tof_edge as f32);
        }
        for i in 0..num_bins {
            let a = edges[i];
            let b = edges[i + 1];
            centers.push(0.5 * (a + b)); // arithmetic center in TOF space
        }

        Some(Self {
            tof_min: tof_min_aligned,
            tof_max: tof_max_aligned,
            tof_step,
            edges,
            centers,
        })
    }

    /// Backwards-compatible alias if you already referenced `build_from_tof`.
    #[inline]
    pub fn build_from_tof(frames: &[TimsFrame], tof_step: i32) -> Option<Self> {
        Self::build_from_frames(frames, tof_step)
    }

    #[inline(always)]
    pub fn num_bins(&self) -> usize { self.centers.len() }

    /// Map TOF → bin index (usize). Returns None if outside range.
    #[inline(always)]
    pub fn index_of_tof(&self, tof: i32) -> Option<usize> {
        if tof < self.tof_min || tof > self.tof_max {
            return None;
        }
        let delta = tof - self.tof_min;
        let idx = (delta / self.tof_step) as isize;
        if idx < 0 {
            None
        } else {
            let u = idx as usize;
            if u >= self.num_bins() { None } else { Some(u) }
        }
    }

    /// Return a bin-index range for a TOF window [a, b] (inclusive, clamped).
    #[inline(always)]
    pub fn index_range_for_tof_window(&self, a: i32, b: i32) -> (usize, usize) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        if hi <= self.tof_min {
            return (0, 0);
        }
        if lo >= self.tof_max {
            let last = self.num_bins().saturating_sub(1);
            return (last, last);
        }
        let lo_idx = self
            .index_of_tof(lo.max(self.tof_min))
            .unwrap_or(0);
        let hi_idx = self
            .index_of_tof(hi.min(self.tof_max))
            .unwrap_or(self.num_bins().saturating_sub(1));
        (lo_idx.min(hi_idx), hi_idx.max(lo_idx))
    }

    #[inline(always)]
    pub fn center(&self, idx: usize) -> f32 {
        self.centers[idx]
    }

    /// Continuous center for a *fractional* TOF-bin index mu.
    /// Used by `tof_center_at` on the Py side.
    #[inline]
    pub fn center_at(&self, mu: f32) -> f32 {
        let n = self.num_bins();
        if n == 0 {
            return 0.0;
        }
        if mu <= 0.0 {
            return self.centers[0];
        }
        let last = (n - 1) as f32;
        if mu >= last {
            return self.centers[n - 1];
        }
        let i0 = mu.floor() as usize;
        let i1 = (i0 + 1).min(n - 1);
        let t = mu - (i0 as f32);
        self.centers[i0] * (1.0 - t) + self.centers[i1] * t
    }

    /// Optional helper: get TOF edges (for debug / plotting).
    #[inline(always)]
    pub fn tof_bounds_for_row(&self, r: usize) -> (f32, f32) {
        (self.edges[r], self.edges[r + 1])
    }
}

/// light 1D Gaussian smoothing on a vector (along scans)
pub fn smooth_vector_gaussian(v: &mut [f32], sigma: f32, truncate: f32) {
    if v.is_empty() || sigma <= 0.0 { return; }
    let radius = (truncate * sigma).ceil() as i32;
    let two_sigma2 = 2.0 * sigma * sigma;
    let mut w = Vec::with_capacity((2*radius + 1) as usize);
    for dx in -radius..=radius {
        let x = dx as f32;
        w.push((-x*x / two_sigma2).exp());
    }
    let sum: f32 = w.iter().copied().sum();
    for v_ in &mut w { *v_ /= sum; }

    let n = v.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut acc = 0.0f32;
        let mut norm = 0.0f32;
        for (k,&wk) in w.iter().enumerate() {
            let di = i as isize + (k as isize - (w.len() as isize -1)/2);
            if di >= 0 && (di as usize) < n {
                acc += wk * v[di as usize];
                norm += wk;
            }
        }
        out[i] = if norm > 0.0 { acc / norm } else { 0.0 };
    }
    v.copy_from_slice(&out);
}

/// Build IM marginal (absolute scan axis). We have to touch selected entries:
pub fn build_im_marginal(
    frames: &[FrameBinView],
    rt_lo: usize, rt_hi: usize,
    bin_lo: usize, bin_hi: usize,
    im_lo: usize, im_hi: usize,
) -> Vec<f32> {
    let len = im_hi + 1 - im_lo;
    let mut out = vec![0.0f32; len];

    for fbv in &frames[rt_lo..=rt_hi] {
        let ub = &fbv.unique_bins;
        if ub.is_empty() { continue; }
        let start = match ub.binary_search(&bin_lo) { Ok(i)=>i, Err(i)=>i.min(ub.len()) };
        let mut i = start;
        while i < ub.len() {
            let b = ub[i];
            if b > bin_hi { break; }
            let lo = fbv.offsets[i]; let hi = fbv.offsets[i+1];
            let scans = &fbv.scan_idx[lo..hi];
            let ints  = &fbv.intensity[lo..hi];
            for (s, val) in scans.iter().zip(ints.iter()) {
                let s_abs = *s as usize;
                if s_abs >= im_lo && s_abs <= im_hi {
                    out[s_abs - im_lo] += *val;
                }
            }
            i += 1;
        }
    }
    out
}

// Key for accumulation
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key { bin: usize, scan: u32 }

pub fn gaussian_blur_tof_sparse(
    fbv: &FrameBinView,
    sigma_bins: f32,
    truncate_k: f32,
) -> FrameBinView {
    if fbv.unique_bins.is_empty() { return fbv.clone(); }

    let (deltas, weights) = gaussian_kernel_bins(sigma_bins, truncate_k);
    let nnz = fbv.intensity.len();
    let k = deltas.len();

    // Reserve: heuristic ~ nnz * effective support (not all deltas hit valid bins)
    let mut acc: FxHashMap<Key, f32> = FxHashMap::with_capacity_and_hasher(nnz.saturating_mul(k/2), Default::default());

    // Optional: quick bounds to avoid branching
    let bin_min = *fbv.unique_bins.first().unwrap();
    let bin_max = *fbv.unique_bins.last().unwrap();

    // Iterate bins in order
    for (i, &bin) in fbv.unique_bins.iter().enumerate() {
        let lo = fbv.offsets[i];
        let hi = fbv.offsets[i + 1];
        for j in lo..hi {
            let scan = fbv.scan_idx[j];   // u32 absolute scan
            let val  = fbv.intensity[j];
            if val <= 0.0 { continue; }

            // Spread to neighbors
            for (d, &w) in deltas.iter().zip(weights.iter()) {
                let b2 = if d.is_negative() {
                    bin.saturating_sub((-d) as usize)
                } else {
                    bin.saturating_add(*d as usize)
                };
                if b2 < bin_min || b2 > bin_max { continue; }
                let key = Key { bin: b2, scan };
                *acc.entry(key).or_insert(0.0) += val * w;
            }
        }
    }

    // Compact back into FrameBinView: group by bin, then by insertion order build CSR
    let mut items: Vec<(usize, u32, f32)> = Vec::with_capacity(acc.len());
    for (k, v) in acc.into_iter() {
        if v > 0.0 && k.bin >= bin_min && k.bin <= bin_max {
            items.push((k.bin, k.scan, v));
        }
    }
    items.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut unique_bins = Vec::new();
    let mut offsets = Vec::new();
    let mut scan_idx = Vec::with_capacity(items.len());
    let mut intensity = Vec::with_capacity(items.len());

    offsets.push(0);
    let mut cur_bin: Option<usize> = None;
    for (bin, scan, val) in items.into_iter() {
        if cur_bin.map_or(true, |cb| cb != bin) {
            unique_bins.push(bin);
            offsets.push(offsets.last().copied().unwrap());
            cur_bin = Some(bin);
        }
        scan_idx.push(scan);
        intensity.push(val);
        *offsets.last_mut().unwrap() += 1;
    }
    // Guarantee at least one offset (if frame is empty after filtering)
    if unique_bins.is_empty() {
        return FrameBinView {
            _frame_id: fbv._frame_id,
            unique_bins: Vec::new(),
            offsets: vec![0],
            scan_idx: Vec::new(),
            intensity: Vec::new(),
        };
    }

    FrameBinView {
        _frame_id: fbv._frame_id,
        unique_bins,
        offsets,
        scan_idx,
        intensity,
    }
}

pub fn blur_tof_all_frames(
    frames: &[FrameBinView],
    sigma_bins: f32,
    truncate_k: f32,
) -> Vec<FrameBinView> {
    use rayon::prelude::*;
    frames.par_iter()
        .map(|f| gaussian_blur_tof_sparse(f, sigma_bins, truncate_k))
        .collect()
}

/// Build RT marginal (len = frames in [rt_lo..rt_hi]), using (bin,scan) window
pub fn build_rt_marginal(
    frames: &[FrameBinView],
    rt_lo: usize, rt_hi: usize,
    bin_lo: usize, bin_hi: usize,
    im_lo: usize, im_hi: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rt_hi + 1 - rt_lo];
    for (k, fbv) in frames[rt_lo..=rt_hi].iter().enumerate() {
        out[k] = sum_frame_block(fbv, bin_lo, bin_hi, im_lo, im_hi);
    }
    out
}

pub fn find_im_peaks_row(
    y_smoothed: &[f32],
    y_raw: &[f32],
    tof_row: usize,
    tof_center: i32,
    tof_bounds: (i32, i32),
    rt_bounds: (usize, usize),
    frame_id_bounds: (u32, u32),
    window_group: Option<u32>,
    mobility_of: MobilityFn,
    min_prom: f32,
    min_distance_scans: usize,
    min_width_scans: usize,
    scan_axis: &[usize],
) -> Vec<ImPeak1D> {
    let n = y_smoothed.len();
    if n < 3 { return Vec::new(); }

    let row_max = y_raw.iter().copied().fold(0.0f32, f32::max);
    if row_max < min_prom { return Vec::new(); }

    // 1) strict local maxima on smoothed
    let mut cands = Vec::new();
    for i in 1..n-1 {
        let yi = y_smoothed[i];
        if yi > y_smoothed[i-1] && yi >= y_smoothed[i+1] {
            cands.push(i);
        }
    }

    let mut peaks: Vec<ImPeak1D> = Vec::new();
    for &i in &cands {
        let apex = y_smoothed[i];

        // 2) prominence baseline (walk to mins bounded by taller peaks)
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
        let width_scans = width.round() as usize;
        if width_scans < min_width_scans { continue; }

        // 4) sub-scan apex offset
        let sub = if i > 0 && i + 1 < n {
            quad_subsample(y_smoothed[i - 1], y_smoothed[i], y_smoothed[i + 1]).clamp(-0.5, 0.5)
        } else { 0.0 };

        // 5) NMS by min_distance_scans (keep the taller)
        if let Some(last) = peaks.last() {
            if i.abs_diff(last.scan) < min_distance_scans {
                if apex <= last.apex_smoothed { continue; }
                // replace last
                peaks.pop();
            }
        }

        // 6) area on raw between fractional bounds
        let left_idx  = left_x.floor().clamp(0.0, (n-1) as f32) as usize;
        let right_idx = right_x.ceil().clamp(0.0, (n-1) as f32) as usize;
        let area = trapezoid_area_fractional(y_raw, left_x.max(0.0), right_x.min((n-1) as f32));

        let mobility = mobility_of.map(|f| f(i));

        let scan_abs  = scan_axis[i];
        let left_abs  = scan_axis[left_idx.min(scan_axis.len()-1)];
        let right_abs = scan_axis[right_idx.min(scan_axis.len()-1)];

        let mut peak = ImPeak1D {
            tof_row,
            tof_center,
            tof_bounds,
            rt_bounds,
            frame_id_bounds,
            window_group,
            scan: i,
            scan_sigma: None,
            mobility,
            apex_smoothed: apex,
            apex_raw: y_raw[i],
            prominence: prom,
            left: left_idx,
            right: right_idx,
            left_x,
            right_x,
            width_scans,
            area_raw: area,
            subscan: sub,
            id: 0, // to be filled later
            scan_abs,
            left_abs,
            right_abs,
        };

        peak.id = im_peak_id(&peak);
        peaks.push(peak);
    }
    peaks
}

#[inline(always)]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 { a + t * (b - a) }

/// Safe integration of a piecewise-linear signal y over [x0, x1].
/// x is in sample-index units, segments are [s, s+1] for s=0..n-2.
/// Handles boundaries: no y[n] access, supports n==0/1.
pub fn trapezoid_area_fractional(y: &[f32], mut x0: f32, mut x1: f32) -> f32 {
    let n = y.len();
    if n < 2 {
        // With <2 samples, treat area as 0
        return 0.0;
    }

    // Clamp to [0, n-1]
    let max_x = (n - 1) as f32;
    x0 = x0.clamp(0.0, max_x);
    x1 = x1.clamp(0.0, max_x);
    if x1 <= x0 { return 0.0; }

    let i0 = x0.floor() as usize;
    let i1 = x1.floor() as usize;

    // Both inside same segment [i0, i0+1] (ensure i0 < n-1)
    if i0 == i1 {
        let i = i0.min(n - 2);
        let t0 = x0 - i as f32;
        let t1 = x1 - i as f32;
        let y0 = lerp(y[i], y[i + 1], t0);
        let y1 = lerp(y[i], y[i + 1], t1);
        return 0.5 * (y0 + y1) * (t1 - t0);
    }

    // Work with segment indices s in [0..n-2]
    let s0 = i0.min(n - 2);
    let s1 = i1.min(n - 2);

    let mut area = 0.0f32;

    // Left partial: from x0 within segment s0 up to s0+1
    let t0 = x0 - s0 as f32;              // in [0,1)
    let yl0 = lerp(y[s0], y[s0 + 1], t0); // value at x0
    let yl1 = y[s0 + 1];                  // value at segment end
    area += 0.5 * (yl0 + yl1) * (1.0 - t0);

    // Full interior segments: s in (s0+1 .. s1-1)
    // Only if there is at least one full segment strictly between left/right partials
    if s1 > s0 + 1 {
        for s in (s0 + 1)..s1 {
            area += 0.5 * (y[s] + y[s + 1]); // width = 1
        }
    }

    // Right partial: only if x1 lies *inside* some segment (i1 <= n-2)
    if i1 <= n - 2 {
        let t1 = x1 - s1 as f32;             // in (0,1]
        let yr0 = y[s1];
        let yr1 = lerp(y[s1], y[s1 + 1], t1);
        area += 0.5 * (yr0 + yr1) * t1;
    } else {
        // i1 == n-1 → x1 is exactly at the last node; no right partial to add.
    }

    area
}

#[inline(always)]
pub fn quad_subsample(y0: f32, y1: f32, y2: f32) -> f32 {
    let denom = y0 - 2.0*y1 + y2;
    if denom.abs() < 1e-12 { 0.0 } else { 0.5 * (y0 - y2) / denom }
}


#[inline]
pub fn im_peak_id(p: &ImPeak1D) -> PeakId {
    use rustc_hash::FxHasher;
    use std::hash::{Hash, Hasher};

    // Deterministic over the identity-defining fields
    let mut h = FxHasher::default();
    p.window_group.hash(&mut h);
    p.tof_row.hash(&mut h);
    p.scan.hash(&mut h);
    p.left.hash(&mut h);
    p.right.hash(&mut h);
    p.frame_id_bounds.hash(&mut h);

    let u: u64 = h.finish();
    // force into non-negative i64 space (avoids pandas/NumPy uint64 weirdness)
    (u & 0x7FFF_FFFF_FFFF_FFFF) as i64
}

#[inline]
pub fn rt_peak_id(r: &RtPeak1D) -> PeakId {
    let mut h = FxHasher::default();
    r.parent_im_id.hash(&mut h);
    r.tof_row.hash(&mut h);
    r.rt_idx.hash(&mut h);
    r.frame_id_bounds.hash(&mut h);
    r.window_group.hash(&mut h);
    let u: u64 = h.finish();
    // force into non-negative i64 space (avoids pandas/NumPy uint64 weirdness)
    (u & 0x7FFF_FFFF_FFFF_FFFF) as i64
}

pub fn fallback_rt_peak_from_trace(
    trace_raw: &[f32],
    ctx: RtTraceCtx<'_>,
    im_peak: &ImPeak1D,
    frac: f32, // e.g., 0.10
) -> Option<RtPeak1D> {
    let n = trace_raw.len();
    if n == 0 { return None; }

    // apex on raw
    let (mut i_max, mut y_max) = (0usize, 0.0f32);
    for (i, &y) in trace_raw.iter().enumerate() {
        if y > y_max { y_max = y; i_max = i; }
    }
    if y_max <= 0.0 { return None; }

    // centroid (first moment) for a stable center estimate
    let mut sum_y = 0.0f32;
    let mut sum_ty = 0.0f32;
    for (i, &y) in trace_raw.iter().enumerate() {
        sum_y  += y;
        sum_ty += (i as f32) * y;
    }
    let mu = if sum_y > 0.0 { sum_ty / sum_y } else { i_max as f32 };
    let rt_idx = mu.floor().clamp(0.0, (n - 1) as f32) as usize;
    let subframe = (mu - (rt_idx as f32)).clamp(-0.5, 0.5);

    // fractional width at frac * apex
    let thr = frac * y_max;
    // left crossing
    let mut l = i_max;
    while l > 0 && trace_raw[l] > thr { l -= 1; }
    let left_x = if l < i_max {
        let y0 = trace_raw[l]; let y1 = trace_raw[l + 1];
        l as f32 + if y1 != y0 { (thr - y0) / (y1 - y0) } else { 0.0 }
    } else { l as f32 };

    // right crossing
    let mut r = i_max;
    while r + 1 < n && trace_raw[r] > thr { r += 1; }
    let right_x = if r > i_max && r < n {
        let y0 = trace_raw[r - 1]; let y1 = trace_raw[r];
        (r - 1) as f32 + if y1 != y0 { (thr - y0) / (y1 - y0) } else { 0.0 }
    } else { r as f32 };

    let width_frames = (right_x - left_x).max(0.0).round() as usize;
    let area_raw = trapezoid_area_fractional(trace_raw, left_x.max(0.0), right_x.min((n - 1) as f32));

    // “prominence” proxy: apex – min in support (conservative)
    let base = trace_raw.iter().copied().fold(f32::INFINITY, f32::min);
    let prom = (y_max - base).max(0.0);

    // integer bounds & frame ids
    let l_i = left_x.floor().clamp(0.0, n.saturating_sub(1) as f32) as usize;
    let r_i = right_x.ceil().clamp(0.0, n.saturating_sub(1) as f32) as usize;
    let rt_bounds_frames = (l_i.min(r_i), r_i.max(l_i));

    let frame_id_bounds = if ctx.frame_ids_sorted.is_empty() {
        im_peak.frame_id_bounds
    } else {
        let lo = ctx.frame_ids_sorted[rt_bounds_frames.0.min(ctx.frame_ids_sorted.len()-1)];
        let hi = ctx.frame_ids_sorted[rt_bounds_frames.1.min(ctx.frame_ids_sorted.len()-1)];
        (lo.min(hi), lo.max(hi))
    };

    let t = ctx.rt_times_sec;
    let j0 = rt_idx.min(t.len() - 1);
    let j1 = (j0 + 1).min(t.len() - 1);
    let frac = (subframe + (rt_idx as f32) - j0 as f32).clamp(0.0, 1.0);
    let rt_sec = Some((1.0 - frac) * t[j0] + frac * t[j1]);

    let mut rp = RtPeak1D {
        rt_idx,
        rt_sec,
        apex_smoothed: y_max, // no smoothing in fallback
        apex_raw: y_max,
        prominence: prom,
        left_x,
        right_x,
        width_frames,
        area_raw,
        subframe,

        rt_bounds_frames,
        frame_id_bounds,
        window_group: im_peak.window_group,

        tof_row: im_peak.tof_row,
        tof_center: im_peak.tof_center,
        tof_bounds: im_peak.tof_bounds,

        parent_im_id: Some(im_peak.id),
        id: 0,
    };
    rp.id = rt_peak_id(&rp);
    Some(rp)
}

/// Sum across a single frame in bin [bin_lo..bin_hi], scan [im_lo..im_hi]
#[inline]
fn sum_frame_block(fbv:&FrameBinView, bin_lo:usize, bin_hi:usize, im_lo:usize, im_hi:usize) -> f32 {
    // identical approach as your sum_frame_bins_scans, but for a range
    let ub = &fbv.unique_bins;
    if ub.is_empty() || bin_lo > bin_hi { return 0.0; }
    let start = match ub.binary_search(&bin_lo) { Ok(i)=>i, Err(i)=>i.min(ub.len()) };
    let mut acc = 0.0f32;
    let mut i = start;
    while i < ub.len() {
        let b = ub[i];
        if b > bin_hi { break; }
        let lo = fbv.offsets[i];
        let hi = fbv.offsets[i+1];
        let scans = &fbv.scan_idx[lo..hi];
        let ints  = &fbv.intensity[lo..hi];
        for (s, val) in scans.iter().zip(ints.iter()) {
            let s = *s as usize;
            if s >= im_lo && s <= im_hi { acc += *val; }
        }
        i += 1;
    }
    acc
}

/// Build a histogram over CSR bins (one value per bin in [bin_lo..bin_hi]).
///
/// Returns (hist, centers) where:
///   - `hist[k]` = summed intensity for bin (bin_lo + k)
///   - `centers[k]` = TOF center of that bin (scale.center)
pub fn build_tof_hist(
    frames: &[FrameBinView],
    rt_lo: usize, rt_hi: usize,
    bin_lo: usize, bin_hi: usize,
    im_lo: usize, im_hi: usize,
    scale: &TofScale,
) -> (Vec<f32>, Vec<f32>) {
    let r = bin_hi + 1 - bin_lo;
    let mut hist = vec![0.0f32; r];
    for fbv in &frames[rt_lo..=rt_hi] {
        let ub = &fbv.unique_bins;
        if ub.is_empty() { continue; }
        let start = match ub.binary_search(&bin_lo) { Ok(i)=>i, Err(i)=>i.min(ub.len()) };
        let mut i = start;
        while i < ub.len() {
            let b = ub[i];
            if b > bin_hi { break; }
            let lo = fbv.offsets[i]; let hi = fbv.offsets[i+1];
            let scans = &fbv.scan_idx[lo..hi];
            let ints  = &fbv.intensity[lo..hi];
            let mut sum = 0.0f32;
            for (s, val) in scans.iter().zip(ints.iter()) {
                let s_abs = *s as usize;
                if s_abs >= im_lo && s_abs <= im_hi { sum += *val; }
            }
            hist[b - bin_lo] += sum;
            i += 1;
        }
    }
    let centers = (bin_lo..=bin_hi).map(|i| scale.center(i)).collect::<Vec<_>>();
    (hist, centers)
}

#[inline]
pub fn quantile(values: &[f32], q: f32) -> f32 {
    // Drop non-finite values to avoid NaN poisoning the order
    let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() { return 0.0; }

    let n = v.len();
    let q = q.clamp(0.0, 1.0);
    let idx = (q * (n.saturating_sub(1) as f32)).round() as usize;

    // For floats, use the comparator form + total ordering
    let (_left, nth, _right) = v.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    *nth
}

#[inline]
pub fn quantile_mut(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() { return 0.0; }
    // Replace NaNs with -inf so they sort to the front (or filter them beforehand)
    for x in v.iter_mut() {
        if !x.is_finite() { *x = f32::NEG_INFINITY; }
    }
    let idx = (q.clamp(0.0, 1.0) * (v.len().saturating_sub(1) as f32)).round() as usize;
    let (_l, nth, _r) = v.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    *nth
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Fit1D {
    pub mu: f32,
    pub sigma: f32,
    pub height: f32,
    pub baseline: f32,
    pub area: f32,
    pub r2: f32,
    pub n: usize,
}

pub fn fit1d_moment(y:&[f32], x: Option<&[f32]>) -> Fit1D {
    let n = y.len();
    if n == 0 { return Fit1D::default(); }

    // --- Baseline via 10% quantile (guarded) ---
    let mut ys = y.to_vec();
    let mut b0 = quantile_mut(&mut ys, 0.10);
    if !b0.is_finite() { b0 = 0.0; }

    // --- Domain helpers ---
    let (x_lo, x_hi) = if let Some(xx) = x {
        if !xx.is_empty() {
            let lo = *xx.first().unwrap_or(&0.0);
            let hi = *xx.last().unwrap_or(&lo);
            (if lo.is_finite(){lo}else{0.0}, if hi.is_finite(){hi}else{0.0})
        } else { (0.0, 0.0) }
    } else {
        if n > 0 { (0.0, n.saturating_sub(1) as f32) } else { (0.0, 0.0) }
    };

    let mu_fallback: f32 = 0.5 * (x_lo + x_hi);

    // --- Weighted moments (baseline-subtracted positives) + trapezoid area ---
    let mut wsum=0.0f64;
    let mut xsum=0.0f64;
    let mut x2sum=0.0f64;

    let mut y_max = f32::NEG_INFINITY;
    let mut trap = 0.0f64;

    for i in 0..n {
        let xi_f32 = x.map(|xx| xx[i]).unwrap_or(i as f32);
        let xi = xi_f32 as f64;
        let yi_pos = (y[i] - b0).max(0.0) as f64;

        if yi_pos > 0.0 {
            wsum += yi_pos;
            xsum += yi_pos * xi;
            x2sum += yi_pos * xi * xi;
        }

        if y[i] > y_max { y_max = y[i]; }

        // trapezoid area: handle both irregular and unit spacing
        if i > 0 {
            let y0 = (y[i-1] - b0).max(0.0) as f64;
            let yi = yi_pos;
            let dx = if let Some(xx) = x {
                (xx[i] - xx[i-1]) as f64
            } else {
                1.0
            };
            trap += 0.5 * (y0 + yi) * dx;
        }
    }

    // No positive mass → sane fallback
    if wsum <= 0.0 {
        let mut area = trap as f32;
        if !area.is_finite() || area < 0.0 { area = 0.0; }

        let mut mu = mu_fallback;
        if !mu.is_finite() { mu = 0.0; }
        // clamp mu to domain
        mu = mu.clamp(x_lo, x_hi);

        let mut height = (y_max - b0).max(0.0);
        if !height.is_finite() || height < 0.0 { height = 0.0; }

        let mut baseline = b0.max(0.0);
        if !baseline.is_finite() || baseline < 0.0 { baseline = 0.0; }

        return Fit1D { mu, sigma: 0.0, height, baseline, area, r2: 0.0, n };
    }

    let mut mu = (xsum/wsum) as f32;
    if !mu.is_finite() { mu = mu_fallback; }
    // clamp to domain
    mu = mu.clamp(x_lo, x_hi);

    let var = (x2sum/wsum - (mu as f64)*(mu as f64)).max(0.0) as f32;
    let mut sigma = var.sqrt();
    if !sigma.is_finite() || sigma < 0.0 { sigma = 0.0; }
    // tiny epsilon floor to stabilize Gaussian basis
    if sigma > 0.0 && sigma < 1e-6 { sigma = 1e-6; }

    // --- 2×2 LS for (baseline b, height h) with Gaussian g(mu, sigma) ---
    let mut s_gg=0.0f64; let mut s_g1=0.0f64; let s_11=n as f64;
    let mut s_yg=0.0f64; let mut s_y1=0.0f64;

    if sigma > 0.0 {
        for i in 0..n {
            let xi = x.map(|xx| xx[i]).unwrap_or(i as f32);
            let z = (xi - mu) as f64 / (sigma as f64);
            let g = (-0.5*z*z).exp();
            let yi = y[i] as f64;
            s_gg += g*g; s_g1 += g; s_yg += yi*g; s_y1 += yi;
        }
        let det = s_11*s_gg - s_g1*s_g1;
        if det.abs() > 1e-12 {
            let mut b = ( s_gg*s_y1 - s_g1*s_yg)/det;
            let mut h = (-s_g1*s_y1 + s_11*s_yg)/det;

            // physical clamp
            if !b.is_finite() || b < 0.0 { b = 0.0; }
            if !h.is_finite() || h < 0.0 { h = 0.0; }

            let mut area = (h as f32)*sigma*std::f32::consts::TAU.sqrt();
            if !area.is_finite() || area < 0.0 { area = 0.0; }

            // --- Baseline-aware, clamped R^2 ---
            // Null model is the fitted baseline b (better than raw mean for chromatographic/IM traces)
            let baseline_ref = b as f32;
            let mut ss_res=0.0f64;
            let mut ss_tot=0.0f64;
            for i in 0..n {
                let xi = x.map(|xx| xx[i]).unwrap_or(i as f32);
                let z = (xi - mu)/sigma;
                let yhat = baseline_ref + (h as f32)*(-0.5*z*z).exp();
                let e = (y[i]-yhat) as f64;
                ss_res += e*e;
                let d = (y[i]-baseline_ref) as f64;
                ss_tot += d*d;
            }
            let mut r2 = if ss_tot > 1e-20 { (1.0 - ss_res/ss_tot) as f32 } else { 0.0 };
            if !r2.is_finite() { r2 = 0.0; }
            // clamp to [0,1]
            r2 = r2.clamp(0.0, 1.0);

            return Fit1D { mu, sigma, height: h as f32, baseline: b as f32, area, r2, n };
        }
    }

    // --- Fallback: use moment height & sigma, Gaussian area or trapezoid ---
    let mut height = (y_max - b0).max(0.0);
    if !height.is_finite() || height < 0.0 { height = 0.0; }

    let mut area = if sigma > 0.0 {
        height * sigma * std::f32::consts::TAU.sqrt()
    } else { trap as f32 };
    if !area.is_finite() || area < 0.0 { area = 0.0; }

    let mut baseline = b0.max(0.0);
    if !baseline.is_finite() || baseline < 0.0 { baseline = 0.0; }

    Fit1D { mu, sigma, height, baseline, area, r2: 0.0, n }
}

fn gaussian_kernel_bins(sigma_bins: f32, truncate_k: f32) -> (Vec<i32>, Vec<f32>) {
    let sigma = sigma_bins.max(0.3);
    let radius = (truncate_k * sigma).ceil() as i32;
    let mut offs = Vec::with_capacity((2*radius + 1) as usize);
    let mut w    = Vec::with_capacity(offs.capacity());
    let two_s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;
    for d in -radius..=radius {
        let x = d as f32;
        let wd = (-x*x / two_s2).exp();
        offs.push(d);
        w.push(wd);
        sum += wd;
    }
    for wi in &mut w { *wi /= sum.max(1e-12); }
    (offs, w)
}

fn interval_overlap_usize(a: (usize, usize), b: (usize, usize)) -> usize {
    let lo = a.0.max(b.0);
    let hi = a.1.min(b.1);
    if lo > hi {
        0
    } else {
        hi - lo + 1
    }
}

fn jaccard_1d_usize(a: (usize, usize), b: (usize, usize)) -> f32 {
    let inter = interval_overlap_usize(a, b) as f32;
    if inter <= 0.0 {
        return 0.0;
    }
    let len_a = if a.1 >= a.0 { (a.1 - a.0 + 1) as f32 } else { 0.0 };
    let len_b = if b.1 >= b.0 { (b.1 - b.0 + 1) as f32 } else { 0.0 };
    let union = (len_a + len_b - inter).max(1.0);
    inter / union
}

/// Fast compatibility check for deciding whether two ImPeak1D should be merged.
///
/// All checks are *conservative*: if any condition fails, we do NOT merge.
fn compatible_fast(a: &ImPeak1D, b: &ImPeak1D, params: &StitchParams) -> bool {
    // 1) Window group (unless crossing allowed)
    if !params.allow_cross_groups {
        if a.window_group != b.window_group {
            return false;
        }
    }

    // 2) TOF row proximity (we now use max_mz_row_delta in tof_row space)
    if params.max_tof_row_delta > 0 {
        let d = if a.tof_row > b.tof_row {
            a.tof_row - b.tof_row
        } else {
            b.tof_row - a.tof_row
        };
        if d > params.max_tof_row_delta {
            return false;
        }
    }

    // 3) RT overlap (rt_bounds are indices in a frame grid, inclusive)
    let rt_a = a.rt_bounds;
    let rt_b = b.rt_bounds;
    let rt_overlap = interval_overlap_usize(rt_a, rt_b);
    if rt_overlap < params.min_overlap_frames {
        return false;
    }

    if params.jaccard_min > 0.0 {
        let j_rt = jaccard_1d_usize(rt_a, rt_b);
        if j_rt < params.jaccard_min {
            return false;
        }
    }

    // 4) IM (scan) overlap; left/right are inclusive scan indices on the grid
    let im_a = (a.left, a.right);
    let im_b = (b.left, b.right);
    let im_overlap = interval_overlap_usize(im_a, im_b);
    if im_overlap < params.min_im_overlap_scans {
        return false;
    }

    if params.im_jaccard_min > 0.0 {
        let j_im = jaccard_1d_usize(im_a, im_b);
        if j_im < params.im_jaccard_min {
            return false;
        }
    }

    // 5) Mutual apex inside overlap band (in absolute scan coords)
    // 5) Mutual apex inside overlap band (in absolute scan coords)
    if params.require_mutual_apex_inside {
        let left_abs_overlap  = a.left_abs.max(b.left_abs);
        let right_abs_overlap = a.right_abs.min(b.right_abs);
        if left_abs_overlap > right_abs_overlap {
            return false;
        }

        let lo_f = left_abs_overlap as f32;
        let hi_f = right_abs_overlap as f32;

        // Use absolute scan center as apex; subscan is *fractional* around the local index,
        // not a global coordinate.
        let a_apex = a.scan_abs as f32;
        let b_apex = b.scan_abs as f32;

        if !(a_apex >= lo_f && a_apex <= hi_f && b_apex >= lo_f && b_apex <= hi_f) {
            return false;
        }
    }

    true
}

fn merge_group(peaks: &[&ImPeak1D]) -> ImPeak1D {
    use std::cmp::{min, max};

    assert!(!peaks.is_empty());

    // --- initialize accumulators -----------------------------------------
    let mut rt_lo = usize::MAX;
    let mut rt_hi = 0usize;

    let mut fid_lo = u32::MAX;
    let mut fid_hi = 0u32;

    let mut tof_lo = i32::MAX;
    let mut tof_hi = i32::MIN;

    let mut left = usize::MAX;
    let mut right = 0usize;

    let mut left_abs = usize::MAX;
    let mut right_abs = 0usize;

    let mut area_sum = 0.0f32;

    // weighted centers
    let mut w_sum   = 0.0f32;
    let mut w_tof_row = 0.0f32;
    let mut w_tof_center = 0.0f32;
    let mut w_scan      = 0.0f32;
    let mut w_scan_abs  = 0.0f32;
    let mut w_subscan   = 0.0f32;

    // apex-like
    let mut apex_smoothed_max = f32::MIN;
    let mut apex_raw_max      = f32::MIN;
    let mut prominence_max    = f32::MIN;

    // scan_sigma aggregation (only over Some)
    let mut scan_sigma_w_sum = 0.0f32;
    let mut scan_sigma_w_val = 0.0f32;

    // mobility: keep first non-None
    let mut mobility_agg: Option<f32> = None;

    // window_group: if they differ, collapse to None
    let mut wg_agg = peaks[0].window_group;
    let mut id_min = peaks[0].id;

    for p in peaks {
        let w = p.area_raw.max(1.0);
        w_sum += w;
        area_sum += p.area_raw;

        // bounds
        let (rt0, rt1) = p.rt_bounds;
        rt_lo = min(rt_lo, rt0);
        rt_hi = max(rt_hi, rt1);

        let (fid0, fid1) = p.frame_id_bounds;
        fid_lo = min(fid_lo, fid0);
        fid_hi = max(fid_hi, fid1);

        let (tof0, tof1) = p.tof_bounds;
        tof_lo = min(tof_lo, tof0);
        tof_hi = max(tof_hi, tof1);

        left     = min(left, p.left);
        right    = max(right, p.right);
        left_abs = min(left_abs, p.left_abs);
        right_abs= max(right_abs, p.right_abs);

        // centers
        w_tof_row    += (p.tof_row as f32)    * w;
        w_tof_center += (p.tof_center as f32) * w;
        w_scan       += (p.scan as f32)       * w;
        w_scan_abs   += (p.scan_abs as f32)   * w;
        w_subscan    +=  p.subscan            * w;

        // apex
        apex_smoothed_max = apex_smoothed_max.max(p.apex_smoothed);
        apex_raw_max      = apex_raw_max.max(p.apex_raw);
        prominence_max    = prominence_max.max(p.prominence);

        // scan_sigma
        if let Some(s) = p.scan_sigma {
            scan_sigma_w_sum += w;
            scan_sigma_w_val += s * w;
        }

        // mobility
        if mobility_agg.is_none() {
            mobility_agg = p.mobility;
        }

        // window_group
        if wg_agg != p.window_group {
            wg_agg = None;
        }

        // ID
        if p.id < id_min {
            id_min = p.id;
        }
    }

    // --- finalize weighted centers ----------------------------------------
    let w_sum_safe = if w_sum > 0.0 { w_sum } else { 1.0 };

    let tof_row = (w_tof_row / w_sum_safe).round().max(0.0) as usize;
    let tof_center = (w_tof_center / w_sum_safe).round() as i32;
    let scan = (w_scan / w_sum_safe).round().max(0.0) as usize;
    let scan_abs = (w_scan_abs / w_sum_safe).round().max(0.0) as usize;
    let subscan = w_subscan / w_sum_safe;

    let width_scans = right_abs
        .saturating_sub(left_abs)
        .saturating_add(1);

    let scan_sigma = if scan_sigma_w_sum > 0.0 {
        Some(scan_sigma_w_val / scan_sigma_w_sum)
    } else {
        None
    };

    ImPeak1D {
        tof_row,
        tof_center,
        tof_bounds: (tof_lo, tof_hi),
        rt_bounds: (rt_lo, rt_hi),
        frame_id_bounds: (fid_lo, fid_hi),
        window_group: wg_agg,

        scan,
        left,
        right,

        scan_abs,
        left_abs,
        right_abs,

        scan_sigma,
        mobility: mobility_agg,
        apex_smoothed: apex_smoothed_max,
        apex_raw: apex_raw_max,
        prominence: prominence_max,
        left_x: left_abs as f32,
        right_x: right_abs as f32,
        width_scans,
        area_raw: area_sum,
        subscan,
        id: id_min,
    }
}

pub struct StitchParams {
    pub min_overlap_frames: usize,
    pub max_scan_delta: usize,
    pub jaccard_min: f32,
    pub max_tof_row_delta: usize,  // now acts on tof_row
    pub allow_cross_groups: bool,
    pub min_im_overlap_scans: usize,
    pub im_jaccard_min: f32,
    pub require_mutual_apex_inside: bool,
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct KeyFlat {
    pub wg_norm: Option<u32>,
    pub scan_bin: usize,
    pub tof_bucket: usize,
}

#[inline]
fn norm_wg(wg: Option<u32>, allow_cross: bool) -> Option<u32> {
    if allow_cross { None } else { wg }
}

#[inline]
fn tof_bucket(tof_row: usize, delta: usize) -> usize {
    // Behaves like old mz_bucket() but in TOF-row space.
    // delta == 0 → each row stands alone; otherwise bucket size = delta+1.
    let w = delta.saturating_add(1);
    tof_row / w
}

/// Core stitching logic on ImPeak1D in TOF space.
///
/// - Buckets by (normed window_group, scan_bin, tof_bucket)
/// - Runs an in-bucket fold using `compatible_fast` + `merge_into`
/// - Then a single consolidation pass across bucket boundaries
pub fn stitch_im_peaks_flat_unordered_impl(
    flat: Vec<Arc<ImPeak1D>>,
    params: &StitchParams,
) -> Vec<ImPeak1D> {
    if flat.is_empty() {
        return Vec::new();
    }

    let mut buckets: FxHashMap<KeyFlat, Vec<Arc<ImPeak1D>>> = FxHashMap::default();

    // 1) Collect into buckets
    for arc in flat.into_iter() {
        let r = arc.as_ref();
        let key = KeyFlat {
            wg_norm: norm_wg(r.window_group, params.allow_cross_groups),
            scan_bin: r.scan / params.max_scan_delta.max(1),
            tof_bucket: tof_bucket(r.tof_row, params.max_tof_row_delta),
            // NOTE: we keep the field name max_mz_row_delta in StitchParams
            // for API compatibility, but it now applies to tof_row.
        };
        buckets.entry(key).or_default().push(arc);
    }

    // 2) In-bucket folding with deterministic order + group merge
    let mut stitched: Vec<ImPeak1D> = Vec::new();

    for (_k, mut vec_arc) in buckets.into_iter() {
        // deterministic order within bucket
        vec_arc.sort_unstable_by(|a, b| {
            let aa = a.as_ref();
            let bb = b.as_ref();
            aa.scan
                .cmp(&bb.scan)
                .then_with(|| aa.rt_bounds.0.cmp(&bb.rt_bounds.0))
                .then_with(|| aa.tof_row.cmp(&bb.tof_row))
                .then_with(|| aa.id.cmp(&bb.id))
        });

        let mut group: Vec<Arc<ImPeak1D>> = Vec::new();

        for arc in vec_arc.into_iter() {
            let p = arc.as_ref();
            if let Some(last) = group.last() {
                if compatible_fast(last.as_ref(), p, params) {
                    // still same cluster → extend group
                    group.push(arc);
                } else {
                    // finalize previous group
                    let refs: Vec<&ImPeak1D> =
                        group.iter().map(|a| a.as_ref()).collect();
                    stitched.push(merge_group(&refs));
                    group.clear();
                    group.push(arc);
                }
            } else {
                // first element in this bucket
                group.push(arc);
            }
        }

        // flush final group in this bucket
        if !group.is_empty() {
            let refs: Vec<&ImPeak1D> = group.iter().map(|a| a.as_ref()).collect();
            stitched.push(merge_group(&refs));
        }
    }

    // 3) Optional cross-bucket consolidation pass (again using group merge)
    stitched.sort_unstable_by(|a, b| {
        a.window_group
            .cmp(&b.window_group)
            .then_with(|| a.tof_row.cmp(&b.tof_row))
            .then_with(|| a.scan.cmp(&b.scan))
            .then_with(|| a.rt_bounds.0.cmp(&b.rt_bounds.0))
            .then_with(|| a.id.cmp(&b.id))
    });

    let mut final_out: Vec<ImPeak1D> = Vec::with_capacity(stitched.len());
    let mut group: Vec<ImPeak1D> = Vec::new();

    for p in stitched.into_iter() {
        if let Some(last) = group.last() {
            if compatible_fast(last, &p, params) {
                // still same stitched cluster
                group.push(p);
            } else {
                // finish previous chain
                let refs: Vec<&ImPeak1D> = group.iter().collect();
                final_out.push(merge_group(&refs));
                group.clear();
                group.push(p);
            }
        } else {
            group.push(p);
        }
    }

    if !group.is_empty() {
        let refs: Vec<&ImPeak1D> = group.iter().collect();
        final_out.push(merge_group(&refs));
    }

    final_out
}