use std::u32;
use mscore::timstof::frame::TimsFrame;
use rustc_hash::{FxHashMap};
use crate::data::dia::TimsDatasetDIA;
use crate::data::handle::TimsData;
use crate::data::meta::FrameMeta;
use rayon::prelude::*;

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

/// find local minima indices between [l, r] (inclusive bounds are fine)
fn find_valleys(y: &[f32], l: usize, r: usize) -> Vec<usize> {
    if r <= l + 2 { return Vec::new(); }
    let mut mins = Vec::new();
    for i in (l+1)..r {
        let yi = y[i];
        if yi <= y[i-1] && yi <= y.get(i+1).copied().unwrap_or(yi) {
            mins.push(i);
        }
    }
    mins
}

/// Rebuild a peak around a chosen apex inside a window [win_l, win_r] using the same
/// prominence/half-prom logic as your main code, but constrained to that window.
/// Returns None if it fails min_prom / min_width_scans.
fn build_peak_in_window(
    y_s: &[f32],
    y_r: &[f32],
    rt_row: usize,
    apex_idx: usize,
    win_l: usize,
    win_r: usize,
    min_prom: f32,
    min_width_scans: usize,
    mobility_of: MobilityFn,
) -> Option<ImPeak1D> {
    let n = y_s.len();
    if n == 0 || win_l >= n || win_r >= n || win_l >= win_r { return None; }

    let apex = y_s[apex_idx];

    // bounded prominence mins (don’t walk beyond the window)
    let mut l = apex_idx; let mut left_min = apex;
    while l > win_l {
        l -= 1;
        left_min = left_min.min(y_s[l]);
        if y_s[l] > apex { break; }
    }
    let mut r = apex_idx; let mut right_min = apex;
    while r + 1 <= win_r {
        r += 1;
        right_min = right_min.min(y_s[r]);
        if y_s[r] > apex { break; }
    }

    let baseline = left_min.max(right_min);
    let prom = apex - baseline;
    if prom < min_prom { return None; }

    // half-prom crossings, constrained
    let half = baseline + 0.5 * prom;

    // left crossing
    let mut wl = apex_idx;
    while wl > win_l && y_s[wl] > half { wl -= 1; }
    let left_x = if wl < apex_idx && wl + 1 < n {
        let y0 = y_s[wl]; let y1 = y_s[wl + 1];
        wl as f32 + if y1 != y0 { (half - y0) / (y1 - y0) } else { 0.0 }
    } else { wl as f32 };

    // right crossing
    let mut wr = apex_idx;
    while wr + 1 <= win_r && y_s[wr] > half { wr += 1; }
    let right_x = if wr > apex_idx && wr < n {
        let y0 = y_s[wr - 1]; let y1 = y_s[wr];
        (wr - 1) as f32 + if y1 != y0 { (half - y0) / (y1 - y0) } else { 0.0 }
    } else { wr as f32 };

    let width = (right_x - left_x).max(0.0);
    let width_scans = width.round() as usize;
    if width_scans < min_width_scans { return None; }

    let left_idx  = left_x.floor().clamp(0.0, (n-1) as f32) as usize;
    let right_idx = right_x.ceil().clamp(0.0, (n-1) as f32) as usize;
    let area = trapezoid_area_fractional(
        y_r,
        left_x.max(0.0),
        right_x.min((n - 1) as f32),
    );

    let mobility = mobility_of.map(|f| f(apex_idx));
    let sub = if apex_idx > 0 && apex_idx + 1 < n {
        quad_subsample(y_s[apex_idx - 1], y_s[apex_idx], y_s[apex_idx + 1]).clamp(-0.5, 0.5)
    } else { 0.0 };

    Some(ImPeak1D {
        rt_row,
        scan: apex_idx,
        mobility,
        apex_smoothed: apex,
        apex_raw: y_r[apex_idx],
        prominence: prom,
        left: left_idx,
        right: right_idx,
        left_x,
        right_x,
        width_scans,
        area_raw: area,
        subscan: sub,
    })
}

/// Try to split a provisional peak at the deepest valley between two lobes.
/// depth_ratio: valley must be <= depth_ratio * min(left_apex_height, right_apex_height)
/// min_sep: minimal scans separating apex from valley/apex to accept a split
fn maybe_split_by_valleys(
    y_s: &[f32],
    y_r: &[f32],
    rt_row: usize,
    peak: &ImPeak1D,
    depth_ratio: f32,          // e.g., 0.6
    min_sep: usize,            // e.g., 2
    min_prom: f32,
    min_width_scans: usize,
    mobility_of: MobilityFn,
) -> Vec<ImPeak1D> {
    let n = y_s.len();
    if n == 0 { return vec![peak.clone()]; }

    let l = peak.left.min(n.saturating_sub(1));
    let r = peak.right.min(n.saturating_sub(1));
    if r <= l + 2*min_sep { return vec![peak.clone()]; }

    // Find a second lobe on the right side of the current apex
    let apex0 = peak.scan;
    let mut apex1 = apex0;
    let mut h1 = y_s[apex0];
    for i in (apex0+min_sep)..=r {
        if y_s[i] > h1 { h1 = y_s[i]; apex1 = i; }
    }
    if apex1 == apex0 {
        // also try a left-side lobe
        for i in l..apex0.saturating_sub(min_sep) {
            if y_s[i] > h1 { h1 = y_s[i]; apex1 = i; }
        }
    }
    if apex1 == apex0 { return vec![peak.clone()]; }

    // Order lobes left→right
    let (left_apex, right_apex) = if apex0 < apex1 { (apex0, apex1) } else { (apex1, apex0) };
    if right_apex <= left_apex + 2*min_sep { return vec![peak.clone()]; }

    // Deepest valley between them
    let valleys = find_valleys(y_s, left_apex + min_sep, right_apex.saturating_sub(min_sep));
    if valleys.is_empty() { return vec![peak.clone()]; }
    let (v, vval) = valleys.into_iter().map(|idx| (idx, y_s[idx]))
        .min_by(|a,b| a.1.total_cmp(&b.1)).unwrap();

    let h_left  = y_s[left_apex];
    let h_right = y_s[right_apex];
    let thr = depth_ratio * h_left.min(h_right);
    if vval > thr { return vec![peak.clone()]; }

    // Build two sub-peaks in [l..v] and [v..r]
    let left_peak = build_peak_in_window(
        y_s, y_r, rt_row, left_apex,
        l, v,
        min_prom, min_width_scans, mobility_of,
    );
    let right_peak = build_peak_in_window(
        y_s, y_r, rt_row, right_apex,
        v, r,
        min_prom, min_width_scans, mobility_of,
    );

    match (left_peak, right_peak) {
        (Some(lp), Some(rp)) => vec![lp, rp],
        _ => vec![peak.clone()],
    }
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

#[derive(Clone, Debug)]
pub struct ImPeak1D {
    pub rt_row: usize,        // which row in ImIndex (i.e., which RtPeak1D)
    pub scan: usize,          // apex scan index
    pub mobility: Option<f32>,// optional 1/K0 (if you pass a converter)
    pub apex_smoothed: f32,   // value on smoothed IM profile
    pub apex_raw: f32,        // raw value at nearest scan
    pub prominence: f32,
    pub left: usize,          // integer bracketing
    pub right: usize,
    pub left_x: f32,          // fractional cross at half-prom (scan units)
    pub right_x: f32,
    pub width_scans: usize,   // ~FWHM estimate
    pub area_raw: f32,        // trapezoid area under raw profile between [left_x,right_x]
    pub subscan: f32,         // sub-scan apex offset in scans (parabolic interp)
}

#[derive(Clone, Debug)]
pub struct RtPeak1D {
    pub mz_row: usize,
    pub mz_center: f32,
    pub rt_col: usize,
    pub rt_time: f32,
    pub apex_smoothed: f32,
    pub apex_raw: f32,
    pub prominence: f32,

    // Half-prominence window (integer indices from fractional crossings)
    pub left: usize,
    pub right: usize,
    pub left_x: f32,    // fractional crossing
    pub right_x: f32,   // fractional crossing

    pub width_frames: usize,
    pub area_raw: f32,          // fractional integral over [left_x, right_x] on RAW trace
    pub subcol: f32,

    // NEW: padded window and its simple area
    pub left_padded: usize,
    pub right_padded: usize,
    pub area_padded: f32,       // integer trapezoid over [left_padded, right_padded] on RAW
}

pub fn find_peaks_row(
    y_smoothed: &[f32],
    y_raw: &[f32],
    mz_row: usize,
    mz_center: f32,
    frame_times: Option<&[f32]>,
    min_prom: f32,
    min_distance: usize,
    min_width: usize,
    pad_left: usize,
    pad_right: usize,
) -> Vec<RtPeak1D> {
    let n = y_smoothed.len();
    if n < 3 { return Vec::new(); }

    // local maxima (ties go to leftmost)
    let mut cands: Vec<usize> = Vec::new();
    for i in 1..n-1 {
        let yi = y_smoothed[i];
        if yi > y_smoothed[i-1] && yi >= y_smoothed[i+1] { cands.push(i); }
    }

    let mut peaks: Vec<RtPeak1D> = Vec::new();
    for &i in &cands {
        let apex = y_smoothed[i];

        // prominence baseline
        let mut l = i; let mut left_min = apex;
        while l > 0 { l -= 1; left_min = left_min.min(y_smoothed[l]); if y_smoothed[l] > apex { break; } }
        let mut r = i; let mut right_min = apex;
        while r+1 < n { r += 1; right_min = right_min.min(y_smoothed[r]); if y_smoothed[r] > apex { break; } }

        let baseline = left_min.max(right_min);
        let prom = apex - baseline;
        if prom < min_prom { continue; }

        // half-prom crossings (flat-top safe)
        let half = baseline + 0.5 * prom;
        let left_cross  = frac_crossing_left(y_smoothed, i, half);
        let right_cross = frac_crossing_right(y_smoothed, i, half);

        let width = (right_cross - left_cross).max(0.0);
        let width_frames = width.round() as usize;
        if width_frames < min_width { continue; }

        let sub = if i > 0 && i+1 < n {
            quad_subsample(y_smoothed[i-1], y_smoothed[i], y_smoothed[i+1]).clamp(-0.5, 0.5)
        } else { 0.0 };

        // fractional area on RAW
        let left_idx  = left_cross.floor().clamp(0.0, (n-1) as f32) as usize;
        let right_idx = right_cross.ceil().clamp(0.0, (n-1) as f32) as usize;
        let area = trapezoid_area_fractional(
            y_raw,
            left_cross.max(0.0),
            right_cross.min((n - 1) as f32),
        );

        // padded integer window
        let left_padded  = left_idx.saturating_sub(pad_left);
        let right_padded = (right_idx + pad_right).min(n - 1);
        let area_padded = _trapezoid_area(y_raw, left_padded, right_padded);

        let pk = RtPeak1D {
            mz_row,
            mz_center,
            rt_col: i,
            rt_time: frame_times.map(|rt| rt[i]).unwrap_or(i as f32),
            apex_smoothed: apex,
            apex_raw: y_raw[i],
            prominence: prom,
            left: left_idx,
            right: right_idx,
            left_x: left_cross,
            right_x: right_cross,
            width_frames,
            area_raw: area,
            subcol: sub,
            left_padded,
            right_padded,
            area_padded,
        };

        // >>> CHANGED: robust NMS against all conflicting kept peaks
        nms_push(
            &mut peaks,
            pk,
            |p: &RtPeak1D| p.rt_col,
            |p: &RtPeak1D| p.apex_smoothed,
            min_distance,
        );
    }

    peaks
}

pub fn pick_peaks_all_rows(
    data_smoothed: &[f32],   // column-major
    data_raw: &[f32],        // column-major
    rows: usize,
    cols: usize,
    frame_times: Option<&[f32]>,
    min_prom: f32,
    min_distance: usize,
    min_width: usize,
    pad_left: usize,         // NEW
    pad_right: usize,
    centers: Option<&[f32]>, // NEW
) -> Vec<RtPeak1D> {
    (0..rows).into_par_iter().flat_map_iter(|r| {
        // gather strided row r
        let mut y_s = Vec::with_capacity(cols);
        let mut y_r = Vec::with_capacity(cols);
        for c in 0..cols {
            y_s.push(data_smoothed[r + c*rows]);
            y_r.push(data_raw     [r + c*rows]);
        }

        let mz_center = centers.map(|cs| cs[r]).unwrap_or(0.0);

        find_peaks_row(
            &y_s, &y_r, r, mz_center,
            frame_times,
            min_prom, min_distance, min_width,
            pad_left, pad_right,
        )
    }).collect()
}

/// Build a normalized 1D Gaussian kernel over frames.
/// `sigma_frames`: stddev in *frames* (e.g., 1.5 means ~1.5-frame sigma)
/// `truncate`: cutoff in sigmas (e.g., 3.0 => radius = ceil(3*sigma))
fn gaussian_kernel_1d(sigma_frames: f32, truncate: f32) -> Vec<f32> {
    assert!(sigma_frames > 0.0 && truncate >= 0.0);
    let radius = (truncate * sigma_frames).ceil() as i32;
    let two_sigma2 = 2.0 * sigma_frames * sigma_frames;
    let mut w = Vec::with_capacity((2 * radius + 1) as usize);
    for dx in -radius..=radius {
        let x = dx as f32;
        w.push((-x * x / two_sigma2).exp());
    }
    // normalize
    let sum: f32 = w.iter().copied().sum();
    for v in &mut w { *v /= sum; }
    w
}

/// Reflect boundary helper: maps t +/- k back into valid [0, cols)
#[inline(always)]
fn reflect_index(idx: isize, len: usize) -> usize {
    // reflect around edges like ... 2 1 | 0 1 2 3 ... 3 2 |
    let len_i = len as isize;
    if len == 0 { return 0; }
    let mut x = idx;
    if x < 0 || x >= len_i {
        // periodic reflection
        // bring x into [-len, 2*len) for few ops
        x = x.rem_euclid(2 * len_i);
        if x < 0 { x += 2 * len_i; }
        if x >= len_i { x = 2 * len_i - 1 - x; }
    }
    x as usize
}

/// In-place time smoothing (along RT) of a column-major matrix.
/// `data`: column-major buffer with shape (rows=num_mz_bins, cols=num_frames).
/// Applies Gaussian along the **time axis** (columns) for each row.
/// Overwrites `data` with the smoothed result.
pub fn smooth_time_gaussian_colmajor(
    data: &mut [f32],
    rows: usize,
    cols: usize,
    sigma_frames: f32,
    truncate: f32,
) {
    if rows == 0 || cols == 0 { return; }
    let w = gaussian_kernel_1d(sigma_frames, truncate);
    let radius = (w.len() as isize - 1) / 2;

    // staging buffer to avoid read-after-write
    let mut out = vec![0.0f32; data.len()];

    // Parallel over output columns (each is a contiguous slice of length `rows`)
    out.par_chunks_mut(rows).enumerate().for_each(|(t, out_col)| {
        // zero the column accumulator
        for v in out_col.iter_mut() { *v = 0.0; }

        // accumulate weighted neighbor columns
        for (k, &wk) in w.iter().enumerate() {
            let dt = k as isize - radius;
            let src_t = reflect_index(t as isize + dt, cols);
            let src_col = &data[src_t * rows .. (src_t + 1) * rows];

            // axpy: out_col += wk * src_col (contiguous, vectorizable)
            for i in 0..rows {
                out_col[i] += wk * src_col[i];
            }
        }
    });

    data.copy_from_slice(&out);
}

#[inline(always)]
fn frame_rt(meta: &FrameMeta) -> f32 { meta.time as f32 }

#[inline(always)]
fn _trapezoid_area(y: &[f32], l: usize, r: usize) -> f32 {
    if r <= l { return 0.0; }
    let mut area = 0.0f32;
    for i in l..r { area += 0.5 * (y[i] + y[i+1]); }
    area
}

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 { a + t * (b - a) }

/// Safe integration of a piecewise-linear signal y over [x0, x1].
/// x is in sample-index units, segments are [s, s+1] for s=0..n-2.
/// Handles boundaries: no y[n] access, supports n==0/1.
fn trapezoid_area_fractional(y: &[f32], mut x0: f32, mut x1: f32) -> f32 {
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
fn quad_subsample(y0: f32, y1: f32, y2: f32) -> f32 {
    let denom = y0 - 2.0*y1 + y2;
    if denom.abs() < 1e-12 { 0.0 } else { 0.5 * (y0 - y2) / denom }
}

#[derive(Clone, Debug)]
pub struct RtIndex {
    pub scale: MzScale,        // <— replace bins: Vec<MzBin>
    pub frames: Vec<u32>,
    pub frame_times: Vec<f32>,
    pub data: Vec<f32>,        // column-major [rows x cols]
    pub rows: usize,
    pub cols: usize,
    pub data_raw: Option<Vec<f32>>,
}

fn scan_mz_range(frames: &[TimsFrame]) -> Option<(f32, f32)> {
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

pub fn build_dense_rt_by_mz_ppm(
    data_handle: &TimsDatasetDIA,
    maybe_sigma_frames: Option<f32>,
    truncate: f32,
    ppm_per_bin: f32,       // <— NEW: constant-ppm bin width
    mz_pad_ppm: f32,        // (optional) pad the min/max by some ppm to avoid edge cutoffs
    num_threads: usize,
) -> RtIndex {
    // 1) RT-sort precursor frames
    let mut rt_sorted_meta: Vec<&FrameMeta> = data_handle.meta_data
        .iter()
        .filter(|m| m.ms_ms_type == 0)
        .collect();
    rt_sorted_meta.sort_by(|a, b| frame_rt(a).partial_cmp(&frame_rt(b)).unwrap());

    let frame_ids_sorted: Vec<u32> = rt_sorted_meta.iter().map(|m| m.id as u32).collect();
    let frame_times: Vec<f32>      = rt_sorted_meta.iter().map(|m| frame_rt(m)).collect();
    let num_frames = frame_ids_sorted.len();

    // 2) materialize at native resolution first (same as before)
    let frames = data_handle
        .get_slice(frame_ids_sorted.clone(), num_threads)
        .frames;

    // 3) discover global mz range and build log-space scale
    let (mut mz_min, mut mz_max) = scan_mz_range(&frames).expect("No m/z values found");
    if mz_pad_ppm > 0.0 {
        let f = 1.0 + mz_pad_ppm * 1e-6;
        mz_min /= f;
        mz_max *= f;
    }
    let scale = MzScale::build(mz_min.max(10.0f32), mz_max, ppm_per_bin);
    let rows = scale.num_bins();

    // 4) fill matrix [rows x cols], col-major
    let mut matrix = vec![0.0f32; rows * num_frames];

    let frames_in_col_order: Vec<&TimsFrame> = frames.iter().collect();
    debug_assert_eq!(frames_in_col_order.len(), num_frames);

    matrix.par_chunks_mut(rows).enumerate().for_each(|(col, col_slice)| {
        let frame = frames_in_col_order[col];
        let mz = &frame.ims_frame.mz;
        let inten = &frame.ims_frame.intensity;

        // local accumulation per column (sparse → dense)
        let mut accum: FxHashMap<usize, f32> = FxHashMap::default();
        for (&m, &i) in mz.iter().zip(inten.iter()) {
            if let Some(row) = scale.index_of(m as f32) {
                *accum.entry(row).or_insert(0.0) += i as f32;
            }
        }
        for (row, val) in accum { col_slice[row] += val; }
    });

    // 5) optional smoothing; keep raw if applied
    let mut data_raw: Option<Vec<f32>> = None;
    if let Some(sigma) = maybe_sigma_frames {
        data_raw = Some(matrix.clone());
        smooth_time_gaussian_colmajor(&mut matrix, rows, num_frames, sigma, truncate);
    }

    RtIndex {
        scale,
        frames: frame_ids_sorted,
        frame_times,
        data: matrix,
        rows,
        cols: num_frames,
        data_raw,
    }
}

/// light 1D Gaussian smoothing on a vector (along scans)
fn smooth_vector_gaussian(v: &mut [f32], sigma: f32, truncate: f32) {
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

struct FrameBinView {
    _frame_id: u32,
    unique_bins: Vec<usize>,
    offsets: Vec<usize>,
    scan_idx: Vec<u32>,
    intensity: Vec<f32>,
}

fn build_frame_bin_view(
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

#[inline(always)]
fn bin_range(view: &FrameBinView, bin_lo: usize, bin_hi: usize) -> (usize, usize) {
    let n = view.unique_bins.len();
    if n == 0 { return (0, 0); }               // empty: nothing to return

    // lower_bound(bin_lo)
    let lb = view.unique_bins.partition_point(|&b| b < bin_lo);
    // upper_bound(bin_hi)
    let ub = view.unique_bins.partition_point(|&b| b <= bin_hi);

    // offsets is CSR with len = unique_bins.len() + 1
    let start = view.offsets[lb];
    let end   = view.offsets[ub];
    (start, end)
}

#[derive(Clone, Debug)]
pub struct ImIndex {
    pub peaks: Vec<RtPeak1D>,  // rows
    pub scans: Vec<usize>,     // 0..num_scans-1 (columns)
    pub data: Vec<f32>,        // column-major: data[scan * rows + row]
    pub rows: usize,
    pub cols: usize,
    pub data_raw: Option<Vec<f32>>,
}

pub fn build_dense_im_by_rtpeaks_ppm(
    data_handle: &TimsDatasetDIA,
    peaks: Vec<RtPeak1D>,      // rows
    rt_index: &RtIndex,        // has .scale and .frames
    num_threads: usize,
    mz_ppm_window: f32,        // ±ppm around the peak center bin
    rt_extra_pad: usize,
    maybe_sigma_scans: Option<f32>,
    truncate: f32,
) -> ImIndex {
    let scale = &rt_index.scale;
    let n_rows = peaks.len();
    if n_rows == 0 {
        return ImIndex { peaks, scans: vec![], data: vec![], rows: 0, cols: 0, data_raw: None };
    }

    let frames = data_handle
        .get_slice(rt_index.frames.clone(), num_threads)
        .frames;

    let global_num_scans: usize = frames.iter()
        .flat_map(|fr| fr.scan.iter())
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(0);


    let ncols = frames.len();
    let views: Vec<FrameBinView> = (0..ncols)
        .into_par_iter()
        .map(|i| build_frame_bin_view(frames[i].clone(), scale, global_num_scans))
        .collect();

    let ncols = views.len();
    if ncols == 0 || global_num_scans == 0 {
        return ImIndex { peaks, scans: vec![], data: vec![], rows: 0, cols: 0, data_raw: None };
    }

    let cols = global_num_scans;
    let rows = n_rows;
    let scans: Vec<usize> = (0..cols).collect();
    let do_smooth = maybe_sigma_scans.unwrap_or(0.0) > 0.0;

    let profiles: Vec<(Vec<f32>, Vec<f32>)> = peaks.par_iter().map(|p| {
        if p.mz_row >= scale.num_bins() { return (vec![0.0; cols], vec![0.0; cols]); }

        // RT window in column space
        let mut lo = p.left_padded.saturating_sub(rt_extra_pad);
        let mut hi = p.right_padded.saturating_add(rt_extra_pad);
        if lo >= ncols { lo = ncols - 1; }
        if hi >= ncols { hi = ncols - 1; }
        if lo > hi { return (vec![0.0; cols], vec![0.0; cols]); }

        // ppm band around bin center
        let mz_center = p.mz_center;
        let tol = mz_center * mz_ppm_window * 1e-6;
        let (bin_lo, bin_hi) = scale.index_range_for_mz_window(mz_center - tol, mz_center + tol);

        let mut raw = vec![0.0f32; cols];
        for t in lo..=hi {
            let v = &views[t];
            let (start, end) = bin_range(v, bin_lo, bin_hi);
            for i in start..end {
                let s = v.scan_idx[i] as usize;
                if s < cols {
                    raw[s] += v.intensity[i];
                }
            }
        }
        let smoothed = if do_smooth {
            let mut tmp = raw.clone();
            smooth_vector_gaussian(&mut tmp, maybe_sigma_scans.unwrap(), truncate);
            tmp
        } else { raw.clone() };

        (smoothed, raw)
    }).collect();

    let mut data = vec![0.0f32; rows * cols];
    let mut data_raw = if do_smooth { Some(vec![0.0f32; rows * cols]) } else { None };

    for (r, (sm, raw)) in profiles.into_iter().enumerate() {
        for s in 0..cols {
            data[s * rows + r] = sm[s];
            if let Some(ref mut dr) = data_raw {
                dr[s * rows + r] = raw[s];
            }
        }
    }

    ImIndex { peaks, scans, data, rows, cols, data_raw }
}

/// Optional mobility callback: scan -> 1/K0
pub type MobilityFn = Option<fn(scan: usize) -> f32>;

#[derive(Clone, Copy)]
pub enum FallbackMode {
    /// No peaks if below low_thresh
    None,
    /// One pseudo-peak spanning the *entire* scan range [0, cols-1]
    FullWindow,
    /// Span only the active scan support where y_raw >= thr,
    /// with optional pad and min width constraints.
    ActiveRange {
        /// absolute floor if rel_thr yields too small
        abs_thr: f32,        // e.g., 5.0
        /// fraction of row_max (0..1), applied to y_raw
        rel_thr: f32,        // e.g., 0.03
        /// expand bounds by this many scans on each side
        pad: usize,          // e.g., 2..4
        /// minimum width to enforce (in scans)
        min_width: usize,    // e.g., 6
    },
    /// Center around the apex with ±half_width scans
    ApexWindow {
        half_width: usize,   // e.g., 15
    },
}

pub fn find_im_peaks_row(
    y_smoothed: &[f32],
    y_raw: &[f32],
    rt_row: usize,
    mobility_of: MobilityFn,   // pass Some(f) if you want mobility values, else None
    min_prom: f32,             // absolute prom threshold in intensity units
    min_distance_scans: usize, // min separation in scans
    min_width_scans: usize,    // min width at half-prom
) -> Vec<ImPeak1D> {
    let n = y_smoothed.len();
    if n < 3 { return Vec::new(); }

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

        // 6) build a provisional peak (same fields you already compute just above)
        let left_idx  = left_x.floor().clamp(0.0, (n-1) as f32) as usize;
        let right_idx = right_x.ceil().clamp(0.0, (n-1) as f32) as usize;
        let area = trapezoid_area_fractional(y_raw, left_x.max(0.0), right_x.min((n-1) as f32));
        let mobility = mobility_of.map(|f| f(i));

        let provisional = ImPeak1D {
            rt_row,
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
        };

        // 7) try to split by valley; then NMS each sub-peak via robust nms_push
        let split_peaks = maybe_split_by_valleys(
            y_smoothed, y_raw, rt_row,
            &provisional,
            /*depth_ratio=*/0.6,
            /*min_sep=*/2,
            min_prom,
            min_width_scans,
            mobility_of,
        );

        for pk in split_peaks {
            // robust NMS against all kept peaks (you already have nms_push in your file)
            nms_push(
                &mut peaks,
                pk,
                |p: &ImPeak1D| p.scan,
                |p: &ImPeak1D| p.apex_smoothed,
                min_distance_scans,
            );
        }
    }
    peaks
}

/// Optional scan->mobility converter from your loader (if you want mobility values).
pub fn pick_im_peaks_on_imindex(
    imx_data_smoothed: &[f32],     // column-major (rows, cols)
    imx_data_raw: Option<&[f32]>,  // None => same as smoothed
    rows: usize,                   // == number of RT peaks
    cols: usize,                   // == num_scans (columns)
    min_prom: f32,
    min_distance_scans: usize,
    min_width_scans: usize,
    mobility_of: MobilityFn,       // e.g., Some(|s| inv_k0_from_scan(s)), else None
) -> Vec<Vec<ImPeak1D>> {
    (0..rows).into_par_iter().map(|r| {
        // gather row r
        let mut y_s = Vec::with_capacity(cols);
        let mut y_r = Vec::with_capacity(cols);
        for s in 0..cols {
            y_s.push(imx_data_smoothed[s * rows + r]);
            let raw_val = imx_data_raw.map(|dr| dr[s * rows + r]).unwrap_or(y_s[s]);
            y_r.push(raw_val);
        }
        find_im_peaks_row(
            &y_s, &y_r, r, mobility_of,
            min_prom, min_distance_scans, min_width_scans
        )
    }).collect()
}

#[inline(always)]
fn frac_crossing_left(y: &[f32], i: usize, half: f32) -> f32 {
    let _n = y.len();
    // walk left until y[k] <= half and y[k+1] > half OR plateau at == half ends
    let mut k = i;
    if y[k] <= half { return k as f32; }
    while k > 0 && y[k - 1] >= half { k -= 1; }
    // Handle plateau at == half by marching further left if needed
    while k > 0 && y[k] == half && y[k - 1] == half { k -= 1; }
    if k == i {
        // normal interpolation between k and k+1
        let y0 = y[k];
        let y1 = y.get(k + 1).copied().unwrap_or(y0);
        if y1 != y0 { k as f32 + (half - y0) / (y1 - y0) } else { k as f32 }
    } else if y[k] == half && k > 0 && y[k - 1] < half {
        // edge of plateau, crossing is at the left boundary of the plateau
        k as f32
    } else {
        // y[k] < half <= y[k+1]
        let y0 = y[k];
        let y1 = y.get(k + 1).copied().unwrap_or(y0);
        if y1 != y0 { k as f32 + (half - y0) / (y1 - y0) } else { k as f32 }
    }
}

#[inline(always)]
fn frac_crossing_right(y: &[f32], i: usize, half: f32) -> f32 {
    let n = y.len();
    // walk right until y[k] <= half and y[k-1] > half OR plateau at == half ends
    let mut k = i;
    if y[k] <= half { return k as f32; }
    while k + 1 < n && y[k + 1] >= half { k += 1; }
    // Handle plateau at == half by marching further right if needed
    while k + 1 < n && y[k] == half && y[k + 1] == half { k += 1; }
    if k == i {
        // normal interpolation between k-1 and k
        if k == 0 { return 0.0; }
        let y0 = y[k - 1];
        let y1 = y[k];
        if y1 != y0 { (k - 1) as f32 + (half - y0) / (y1 - y0) } else { k as f32 }
    } else if y[k] == half && k + 1 < n && y[k + 1] < half {
        // edge of plateau, crossing is at the right boundary of the plateau
        k as f32
    } else {
        // y[k-1] > half >= y[k]
        if k == 0 { return 0.0; }
        let y0 = y[k - 1];
        let y1 = y[k];
        if y1 != y0 { (k - 1) as f32 + (half - y0) / (y1 - y0) } else { k as f32 }
    }
}

#[inline(always)]
fn nms_push<T, FPos, FHeight>(
    store: &mut Vec<T>,
    new_peak: T,
    pos_of: FPos,
    height_of: FHeight,
    min_distance: usize,
) -> bool
where
    FPos: Fn(&T) -> usize,
    FHeight: Fn(&T) -> f32,
{
    let i = pos_of(&new_peak);
    let h = height_of(&new_peak);

    // Pop any existing peaks that conflict but are lower than the new one.
    while let Some(last) = store.last() {
        let j = pos_of(last);
        if i.abs_diff(j) >= min_distance { break; }
        let hj = height_of(last);
        if h > hj {
            store.pop();
            continue; // keep testing against earlier peaks
        } else {
            return false; // new one is lower → drop it
        }
    }

    // Also check earlier (non-adjacent) conflicts if min_distance spans more than one peak.
    // Walk backward while within radius.
    let mut k = store.len();
    while k > 0 {
        k -= 1;
        let j = pos_of(&store[k]);
        if i.abs_diff(j) >= min_distance { break; }
        let hj = height_of(&store[k]);
        if h > hj {
            store.remove(k);
        } else {
            return false;
        }
    }

    store.push(new_peak);
    true
}
