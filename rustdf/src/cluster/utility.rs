use mscore::timstof::frame::TimsFrame;
use crate::cluster::peak::{ImPeak1D, PeakId, RtPeak1D, RtTraceCtx};
use std::hash::{Hash, Hasher};
use rustc_hash::FxHasher;

/// Optional mobility callback: scan -> 1/K0
pub type MobilityFn = Option<fn(scan: usize) -> f32>;

#[derive(Clone, Debug)]
pub struct MzScale {
    pub mz_min: f32,
    pub mz_max: f32,
    pub ppm_per_bin: f32,   // e.g., 5.0 => each bin is ~5 ppm wide
    pub ratio: f32,         // 1.0 + ppm_per_bin*1e-6
    pub inv_ln_ratio: f32,  // 1.0 / ln(ratio) for O(1) indexing
    pub edges: Vec<f32>,    // monotonically increasing, len = num_bins + 1
    pub centers: Vec<f32>,  // geometric mean of edges[i..i+1], len = num_bins
}

impl MzScale {
    pub fn build(mz_min: f32, mz_max: f32, ppm_per_bin: f32) -> Self {

        let est_bins = ((mz_max / mz_min).ln() / (1.0 + ppm_per_bin * 1e-6).ln()).ceil() as usize;
        let max_bins = 1_000_000; // tune
        assert!(est_bins <= max_bins, "ppm_per_bin too fine for mz range ({} bins)", est_bins);

        assert!(mz_min > 0.0 && mz_max > mz_min && ppm_per_bin > 0.0);
        let ratio = 1.0 + ppm_per_bin * 1e-6;
        let inv_ln_ratio = 1.0 / ratio.ln();

        // Generate edges multiplicatively: e[k+1] = e[k] * ratio
        let mut edges = Vec::new();
        let mut x = mz_min;
        edges.push(x);
        while x < mz_max {
            x *= ratio;
            edges.push(x);
            // Guard against numerical stickiness
            if edges.len() > 10_000_000 { break; }
        }
        // Ensure last edge ≥ mz_max
        if *edges.last().unwrap() < mz_max {
            edges.push(mz_max);
        }

        let mut centers = Vec::with_capacity(edges.len().saturating_sub(1));
        for w in edges.windows(2) {
            let c = (w[0] * w[1]).sqrt(); // geometric center
            centers.push(c);
        }
        MzScale { mz_min, mz_max, ppm_per_bin, ratio, inv_ln_ratio, edges, centers }
    }

    pub fn from_centers(centers: &[f32]) -> Self {
        assert!(centers.len() >= 2, "need at least 2 centers");
        for w in centers.windows(2) { assert!(w[1] > w[0]); }
        // Estimate ratio as geometric mean of consecutive ratios
        let mut acc = 0.0f64; let mut n = 0usize;
        for w in centers.windows(2) {
            acc += (w[1] as f64 / w[0] as f64).ln();
            n += 1;
        }
        let ln_r = acc / (n as f64);
        let ratio = (ln_r as f32).exp();
        // Sanity: consecutive ratios within ~0.1% (tune as you like)
        let tol = 1e-3;
        for w in centers.windows(2) {
            let r = w[1] / w[0];
            debug_assert!(((r / ratio) - 1.0).abs() < tol, "centers not constant-ppm spaced");
        }

        let inv_ln_ratio = 1.0 / ratio.ln();
        let sqrt_r = ratio.sqrt();
        let mut edges = Vec::with_capacity(centers.len() + 1);
        edges.push(centers[0] / sqrt_r);
        for &c in centers { edges.push(c * sqrt_r); }
        let mz_min = edges[0];
        let mz_max = *edges.last().unwrap();
        let ppm_per_bin = (ratio - 1.0) * 1e6;

        Self { mz_min, mz_max, ppm_per_bin, ratio, inv_ln_ratio, centers: centers.to_vec(), edges }
    }

    #[inline(always)]
    pub fn num_bins(&self) -> usize { self.centers.len() }

    /// O(1) map: m/z -> bin index (usize). Returns None if outside [mz_min, mz_max].
    #[inline(always)]
    pub fn index_of(&self, mz: f32) -> Option<usize> {
        if !(mz.is_finite()) || mz < self.mz_min || mz > self.mz_max { return None; }
        let i = ((mz / self.mz_min).ln() * self.inv_ln_ratio).floor() as isize;
        if i < 0 { return Some(0) }
        let i = i as usize;
        if i >= self.num_bins() { Some(self.num_bins() - 1) } else { Some(i) }
    }

    /// Return a bin‐index range that covers [mz_lo, mz_hi] (inclusive, clamped).
    #[inline(always)]
    pub fn index_range_for_mz_window(&self, a: f32, b: f32) -> (usize, usize) {
        let (mz_lo, mz_hi) = if a <= b { (a, b) } else { (b, a) };
        if mz_hi <= self.mz_min { return (0, 0); }
        if mz_lo >= self.mz_max { return (self.num_bins()-1, self.num_bins()-1); }
        let lo = self.index_of(mz_lo.max(self.mz_min)).unwrap_or(0);
        let hi = self.index_of(mz_hi.min(self.mz_max)).unwrap_or(self.num_bins()-1);
        (lo.min(hi), hi.max(lo))
    }

    #[inline(always)]
    pub fn center(&self, idx: usize) -> f32 { self.centers[idx] }
}

pub fn scan_mz_range(frames: &[TimsFrame]) -> Option<(f32, f32)> {
    let mut minv = f32::INFINITY;
    let mut maxv = f32::NEG_INFINITY;
    for fr in frames {
        for &mz in &fr.ims_frame.mz {
            let mz = mz as f32;
            if mz.is_finite() {
                if mz < minv { minv = mz; }
                if mz > maxv { maxv = mz; }
            }
        }
    }
    if minv.is_finite() && maxv.is_finite() && maxv > minv { Some((minv, maxv)) } else { None }
}

pub fn find_im_peaks_row(
    y_smoothed: &[f32],
    y_raw: &[f32],
    mz_row: usize,
    mz_center: f32,
    mz_bounds: (f32, f32),
    rt_bounds: (usize, usize),
    frame_id_bounds: (u32, u32),
    window_group: Option<u32>,
    mobility_of: MobilityFn,
    min_prom: f32,
    min_distance_scans: usize,
    min_width_scans: usize,
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

        let mut peak = ImPeak1D{
            mz_row,
            mz_center,
            mz_bounds,
            rt_bounds,
            frame_id_bounds,
            window_group,
            scan: i,
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

#[inline]
pub fn im_peak_id(p: &ImPeak1D) -> PeakId {
    // deterministic over the fields that define identity
    let mut h = FxHasher::default();
    p.window_group.hash(&mut h);
    p.mz_row.hash(&mut h);
    p.scan.hash(&mut h);
    p.left.hash(&mut h);
    p.right.hash(&mut h);
    p.frame_id_bounds.hash(&mut h);
    h.finish()
}

#[inline]
pub fn rt_peak_id(r: &RtPeak1D) -> PeakId {
    let mut h = FxHasher::default();
    r.parent_im_id.hash(&mut h);
    r.mz_row.hash(&mut h);
    r.rt_idx.hash(&mut h);
    r.frame_id_bounds.hash(&mut h);
    r.window_group.hash(&mut h);
    h.finish()
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
    let l_i = left_x.floor().clamp(0.0, (n.saturating_sub(1)) as f32) as usize;
    let r_i = right_x.ceil().clamp(0.0, (n.saturating_sub(1)) as f32) as usize;
    let rt_bounds_frames = (l_i.min(r_i), r_i.max(l_i));

    let frame_id_bounds = if ctx.frame_ids_sorted.is_empty() {
        im_peak.frame_id_bounds
    } else {
        let lo = ctx.frame_ids_sorted[rt_bounds_frames.0.min(ctx.frame_ids_sorted.len()-1)];
        let hi = ctx.frame_ids_sorted[rt_bounds_frames.1.min(ctx.frame_ids_sorted.len()-1)];
        (lo.min(hi), lo.max(hi))
    };

    let rt_sec = ctx.rt_times_sec.map(|t| {
        let j0 = rt_idx.min(t.len()-1);
        let j1 = (j0 + 1).min(t.len()-1);
        let frac = (subframe + (rt_idx as f32) - j0 as f32).clamp(0.0, 1.0);
        (1.0 - frac) * t[j0] + frac * t[j1]
    });

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

        mz_row: im_peak.mz_row,
        mz_center: im_peak.mz_center,
        mz_bounds: im_peak.mz_bounds,

        parent_im_id: Some(im_peak.id),
        id: 0,
    };
    rp.id = rt_peak_id(&rp);
    Some(rp)
}