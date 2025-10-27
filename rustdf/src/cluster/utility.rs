use std::u32;
use mscore::timstof::frame::TimsFrame;
use rustc_hash::{FxHashMap};
use crate::data::dia::TimsDatasetDIA;
use crate::data::handle::TimsData;
use crate::data::meta::FrameMeta;
use rayon::prelude::*;

/// How to turn an IM peak into extraction bounds.
#[derive(Clone, Copy, Debug)]
pub enum ImBoundStrategy {
    /// Use the classic half-prominence integer bounds stored on the peak.
    HalfProm,

    /// Center at apex + subscan, span ±k * sigma (sigma estimated from FWHM).
    /// Ensures at least `min_width` scans and clamps to [0, n_scans-1].
    KSigma {
        k: f32,              // e.g., 2.5–4.0
        min_width: usize,    // e.g., 5–9
    },

    /// Simple integer padding around the stored half-prom bounds.
    Padded {
        pad_left: usize,     // extra scans to include on the left
        pad_right: usize,    // extra scans to include on the right
        min_width: usize,    // floor on width after padding
    },

    /// Active range by threshold (like fallback), useful for very broad features.
    ActiveRange {
        abs_thr: f32,        // absolute floor
        rel_thr: f32,        // fraction of row max (0..1) on RAW profile
        pad: usize,          // expand both sides by this many scans
        min_width: usize,    // enforce minimum width
    },
}

#[inline(always)]
fn sigma_from_fwhm(width_scans: usize) -> f32 {
    // FWHM = 2*sqrt(2*ln2) * sigma ≈ 2.354820045* sigma
    // Use max with a small epsilon to avoid zero width corner-cases.
    const INV_2SQRT2LN2: f32 = 1.0 / 2.354820045f32;
    (width_scans.max(1) as f32) * INV_2SQRT2LN2
}

/// Compute IM extraction bounds for a given peak under a chosen strategy.
/// Returns (left, right) as integer scan indices inclusive, clamped to [0, n_scans-1].
pub fn im_bounds_for_peak(
    y_raw_row: &[f32],     // RAW IM profile for this row (for ActiveRange)
    peak: &ImPeak1D,
    n_scans: usize,
    strategy: ImBoundStrategy,
) -> (usize, usize) {
    if n_scans == 0 { return (0, 0); }

    match strategy {
        ImBoundStrategy::HalfProm => {
            let l = peak.left.min(n_scans.saturating_sub(1));
            let r = peak.right.min(n_scans.saturating_sub(1));
            if l <= r { (l, r) } else { (r, l) }
        }

        ImBoundStrategy::KSigma { k, min_width } => {
            // Center at apex + subscan
            let mu = (peak.scan as f32 + peak.subscan)
                .clamp(0.0, (n_scans.saturating_sub(1)) as f32);
            // Estimate σ from FWHM-ish width at half prominence
            let sigma = sigma_from_fwhm(peak.width_scans).max(0.5);
            let half = (k.max(0.5) * sigma).ceil() as isize;

            let mut l = (mu.floor() as isize - half).max(0) as usize;
            let mut r = (mu.ceil()  as isize + half).min((n_scans - 1) as isize) as usize;

            // Enforce minimum width (inclusive)
            if r + 1 - l < min_width && min_width > 1 {
                let need = min_width - (r + 1 - l);
                let grow_left  = need / 2;
                let grow_right = need - grow_left;

                l = l.saturating_sub(grow_left);
                r = (r + grow_right).min(n_scans - 1);

                // If still too short (row is tiny), just take the whole range
                if r + 1 - l < min_width {
                    l = 0; r = n_scans - 1;
                }
            }
            (l.min(r), r.max(l))
        }

        ImBoundStrategy::Padded { pad_left, pad_right, min_width } => {
            let mut l = peak.left.saturating_sub(pad_left).min(n_scans.saturating_sub(1));
            let mut r = (peak.right + pad_right).min(n_scans.saturating_sub(1));
            if r + 1 - l < min_width && min_width > 1 {
                let need = min_width - (r + 1 - l);
                let grow_left  = need / 2;
                let grow_right = need - grow_left;
                l = l.saturating_sub(grow_left);
                r = (r + grow_right).min(n_scans - 1);
                if r + 1 - l < min_width {
                    l = 0; r = n_scans - 1;
                }
            }
            (l.min(r), r.max(l))
        }

        ImBoundStrategy::ActiveRange { abs_thr, rel_thr, pad, min_width } => {
            let n = y_raw_row.len().min(n_scans);
            if n == 0 { return (0, 0); }
            let row_max = y_raw_row.iter().copied().fold(0.0f32, f32::max);
            let thr = abs_thr.max(rel_thr.clamp(0.0, 1.0) * row_max);

            let mut l_opt: Option<usize> = None;
            let mut r_opt: Option<usize> = None;
            for (i, &v) in y_raw_row.iter().take(n).enumerate() {
                if v >= thr {
                    if l_opt.is_none() { l_opt = Some(i); }
                    r_opt = Some(i);
                }
            }
            let (mut l, mut r) = match (l_opt, r_opt) {
                (Some(l0), Some(r0)) => {
                    (l0.saturating_sub(pad), (r0 + pad).min(n - 1))
                }
                _ => {
                    // Fallback: center around apex
                    let apex = peak.scan.min(n.saturating_sub(1));
                    (apex, apex)
                }
            };

            if r + 1 - l < min_width && min_width > 1 {
                let need = min_width - (r + 1 - l);
                let grow_left  = need / 2;
                let grow_right = need - grow_left;
                l = l.saturating_sub(grow_left);
                r = (r + grow_right).min(n - 1);
                if r + 1 - l < min_width {
                    l = 0; r = n - 1;
                }
            }
            (l.min(r), r.max(l))
        }
    }
}

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

#[derive(Clone, Copy, Debug)]
pub struct ImRowContext {
    pub mz_row: usize,
    pub parent_rt_peak_row: usize,
    pub rt_bounds: (usize, usize),
    pub frame_id_bounds: (u32, u32),
    pub window_group: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct ImPeak1D {
    pub mz_row: usize,                 // was rt_row
    pub parent_rt_peak_row: usize,     // row in RtIndex.peaks
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

#[derive(Clone, Debug)]
pub struct FrameBinView {
    pub _frame_id: u32,
    pub unique_bins: Vec<usize>,
    pub offsets: Vec<usize>,
    pub scan_idx: Vec<u32>,
    pub intensity: Vec<f32>,
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

// --- keep your FallbackMode as you posted ---

#[derive(Clone, Copy)]
pub struct ImAdaptivePolicy {
    pub low_thresh: f32,     // e.g., 100.0
    pub mid_thresh: f32,     // e.g., 200.0
    pub sigma_lo: f32,       // e.g., 4.0 scans (extra per-row smoothing)
    pub sigma_hi: f32,       // e.g., 2.0 scans
    pub min_prom_lo: f32,    // e.g., 0.5 * baseline
    pub min_prom_hi: f32,    // e.g., baseline
    pub min_width_lo: usize, // e.g., 6
    pub min_width_hi: usize, // e.g., 3
    pub fallback_mode: FallbackMode,
}

// ---- helpers for fallbacks ----
fn fallback_peak_full(ctx: &ImRowContext, cols: usize, y_raw: &[f32]) -> Vec<ImPeak1D> {
    if cols == 0 { return Vec::new(); }
    let (scan_max, apex_raw) = y_raw.iter().copied().enumerate()
        .max_by(|a,b| a.1.total_cmp(&b.1)).unwrap_or((0,0.0));
    let left_x = 0.0f32;
    let right_x = (cols.saturating_sub(1)) as f32;
    let area = trapezoid_area_fractional(y_raw, left_x, right_x);
    vec![ImPeak1D {
        mz_row: ctx.mz_row,
        parent_rt_peak_row: ctx.parent_rt_peak_row,
        rt_bounds: ctx.rt_bounds,
        frame_id_bounds: ctx.frame_id_bounds,
        window_group: ctx.window_group,

        scan: scan_max,
        mobility: None,
        apex_smoothed: apex_raw,
        apex_raw,
        prominence: 0.0,
        left: 0,
        right: cols.saturating_sub(1),
        left_x,
        right_x,
        width_scans: cols.max(1),
        area_raw: area,
        subscan: 0.0,
    }]
}

fn fallback_peak_active(
    ctx: &ImRowContext,
    y_raw: &[f32],
    pad: usize,
    min_width: usize,
    abs_thr: f32,
    rel_thr: f32,
) -> Vec<ImPeak1D> {
    let n = y_raw.len();
    if n == 0 { return Vec::new(); }
    let row_max = y_raw.iter().copied().fold(0.0f32, f32::max);
    let thr = abs_thr.max(rel_thr * row_max);

    let mut l = None; let mut r = None;
    for (i, &v) in y_raw.iter().enumerate() {
        if v >= thr { l.get_or_insert(i); r = Some(i); }
    }
    let (mut left, mut right) = if let (Some(l0), Some(r0)) = (l, r) {
        (l0.saturating_sub(pad), (r0 + pad).min(n - 1))
    } else {
        let (scan_max, _) = y_raw.iter().copied().enumerate()
            .max_by(|a,b| a.1.total_cmp(&b.1)).unwrap_or((0,0.0));
        (scan_max, scan_max)
    };
    if right + 1 - left < min_width {
        let need = min_width - (right + 1 - left);
        let grow_left = need / 2;
        let grow_right = need - grow_left;
        left = left.saturating_sub(grow_left);
        right = (right + grow_right).min(n - 1);
    }

    let left_x = left as f32; let right_x = right as f32;
    let area = trapezoid_area_fractional(y_raw, left_x, right_x);
    let (scan_max, apex_raw) = y_raw.iter().copied().enumerate()
        .max_by(|a,b| a.1.total_cmp(&b.1)).unwrap_or((left, 0.0));

    vec![ImPeak1D {
        mz_row: ctx.mz_row,
        parent_rt_peak_row: ctx.parent_rt_peak_row,
        rt_bounds: ctx.rt_bounds,
        frame_id_bounds: ctx.frame_id_bounds,
        window_group: ctx.window_group,

        scan: scan_max,
        mobility: None,
        apex_smoothed: apex_raw,
        apex_raw,
        prominence: 0.0,
        left, right, left_x, right_x,
        width_scans: right + 1 - left,
        area_raw: area,
        subscan: 0.0,
    }]
}

/// Safe ApexWindow fallback: center at apex (max raw), use ±half_width scans,
/// clamp to [0, n-1], and enforce a minimum width. Works for n==0/1 as well.
///
/// `min_width`: ensure at least this many scans (inclusive of both ends).
fn fallback_peak_apex_window(
    ctx: &ImRowContext,
    y_raw: &[f32],
    half_width: usize,
    min_width: usize,
) -> Vec<ImPeak1D> {
    let n = y_raw.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        // Degenerate single-scan row
        let apex_raw = y_raw[0];
        let left = 0usize;
        let right = 0usize;
        let left_x = 0.0f32;
        let right_x = 0.0f32;
        let width = 1usize;
        let area = apex_raw; // treat single sample as its own “area”

        return vec![ImPeak1D {
            mz_row: ctx.mz_row,
            parent_rt_peak_row: ctx.parent_rt_peak_row,
            rt_bounds: ctx.rt_bounds,
            frame_id_bounds: ctx.frame_id_bounds,
            window_group: ctx.window_group,

            scan: 0,
            mobility: None,
            apex_smoothed: apex_raw,
            apex_raw,
            prominence: 0.0,
            left,
            right,
            left_x,
            right_x,
            width_scans: width,
            area_raw: area,
            subscan: 0.0,
        }];
    }

    // Find apex (raw)
    let (apex_idx, apex_raw) = y_raw
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();

    // Initial window around apex, clamped
    let mut left  = apex_idx.saturating_sub(half_width);
    let mut right = (apex_idx + half_width).min(n - 1);

    // Enforce minimum width (inclusive). If impossible, expand to whole row.
    let mut width = right + 1 - left;
    if width < min_width {
        let need = min_width - width;
        let grow_left  = need / 2;
        let grow_right = need - grow_left;

        left  = left.saturating_sub(grow_left);
        right = (right + grow_right).min(n - 1);

        width = right + 1 - left;
        if width < min_width {
            left = 0;
            right = n - 1;
            width = n;
        }
    }

    let left_x  = left as f32;
    let right_x = right as f32;

    // Safe area on [left_x, right_x]
    let area = trapezoid_area_fractional(y_raw, left_x, right_x);

    vec![ImPeak1D {
        mz_row: ctx.mz_row,
        parent_rt_peak_row: ctx.parent_rt_peak_row,
        rt_bounds: ctx.rt_bounds,
        frame_id_bounds: ctx.frame_id_bounds,
        window_group: ctx.window_group,

        scan: apex_idx,
        mobility: None,
        apex_smoothed: apex_raw, // no extra smoothing in fallback
        apex_raw,
        prominence: 0.0,
        left,
        right,
        left_x,
        right_x,
        width_scans: width,
        area_raw: area,
        subscan: 0.0,
    }]
}

// ---- adaptive row wrapper ----
pub fn find_im_peaks_row_adaptive(
    y_smoothed_base: &[f32],
    y_raw: &[f32],
    ctx: &ImRowContext,
    mobility_of: MobilityFn,
    min_distance_scans: usize,
    policy: ImAdaptivePolicy,
) -> Vec<ImPeak1D> {
    let n = y_smoothed_base.len();
    if n < 3 { return Vec::new(); }

    let row_max = y_raw.iter().copied().fold(0.0f32, f32::max);
    if row_max < policy.low_thresh {
        return match policy.fallback_mode {
            FallbackMode::None => Vec::new(),
            FallbackMode::FullWindow =>
                fallback_peak_full(ctx, n, y_raw),
            FallbackMode::ActiveRange { abs_thr, rel_thr, pad, min_width } =>
                fallback_peak_active(ctx, y_raw, pad, min_width, abs_thr, rel_thr),
            FallbackMode::ApexWindow { half_width } =>
                fallback_peak_apex_window(ctx, y_raw, half_width, 3),
        };
    }

    let (sigma, min_prom, min_width_scans) = if row_max < policy.mid_thresh {
        (policy.sigma_lo, policy.min_prom_lo, policy.min_width_lo)
    } else {
        (policy.sigma_hi, policy.min_prom_hi, policy.min_width_hi)
    };

    // optional extra per-row smoothing
    let mut y_s = y_smoothed_base.to_vec();
    if sigma > 0.0 {
        let w = gaussian_kernel_1d(sigma, 3.0);
        let rad = (w.len() as isize - 1) / 2;
        let mut out = vec![0.0f32; n];
        for s in 0..n {
            let mut acc = 0.0f32;
            for (k, &wk) in w.iter().enumerate() {
                let ds = k as isize - rad;
                let j = reflect_index(s as isize + ds, n);
                acc += wk * y_s[j];
            }
            out[s] = acc;
        }
        y_s.copy_from_slice(&out);
    }

    find_im_peaks_row(&y_s, y_raw, ctx, mobility_of, min_prom, min_distance_scans, min_width_scans)
}

// ---- adaptive all-rows ----
pub fn pick_im_peaks_on_imindex_adaptive(
    imx_data_smoothed: &[f32],       // column-major
    imx_data_raw: Option<&[f32]>,    // column-major or None
    rows: usize,
    cols: usize,
    min_distance_scans: usize,
    mobility_of: MobilityFn,
    policy: ImAdaptivePolicy,
) -> Vec<Vec<ImPeak1D>> {
    use rayon::prelude::*;
    (0..rows).into_par_iter().map(|r| {
        let mut y_s = Vec::with_capacity(cols);
        let mut y_r = Vec::with_capacity(cols);
        for s in 0..cols {
            y_s.push(imx_data_smoothed[s * rows + r]);
            let raw_val = imx_data_raw.map(|dr| dr[s * rows + r]).unwrap_or(y_s[s]);
            y_r.push(raw_val);
        }

        // Default context when we don’t have RT-peak provenance
        let ctx = ImRowContext {
            mz_row: r,
            parent_rt_peak_row: r,
            rt_bounds: (0, cols.saturating_sub(1)),
            frame_id_bounds: (0, 0),
            window_group: None,
        };

        find_im_peaks_row_adaptive(&y_s, &y_r, &ctx, mobility_of, min_distance_scans, policy)
    }).collect()
}

pub fn find_im_peaks_row(
    y_smoothed: &[f32],
    y_raw: &[f32],
    ctx: &ImRowContext,
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

        // 6) area on raw between fractional bounds
        let left_idx  = left_x.floor().clamp(0.0, (n-1) as f32) as usize;
        let right_idx = right_x.ceil().clamp(0.0, (n-1) as f32) as usize;
        let area = trapezoid_area_fractional(y_raw, left_x.max(0.0), right_x.min((n-1) as f32));

        let mobility = mobility_of.map(|f| f(i));

        peaks.push(ImPeak1D{
            mz_row: ctx.mz_row,
            parent_rt_peak_row: ctx.parent_rt_peak_row,
            rt_bounds: ctx.rt_bounds,
            frame_id_bounds: ctx.frame_id_bounds,
            window_group: ctx.window_group,

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
        });
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

        let ctx = ImRowContext {
            mz_row: r,
            parent_rt_peak_row: r,
            rt_bounds: (0, cols.saturating_sub(1)),
            frame_id_bounds: (0, 0),
            window_group: None,
        };

        find_im_peaks_row(
            &y_s, &y_r, &ctx, mobility_of,
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

/// Build a constant-ppm m/z × RT matrix for a specific **set of frames** (e.g. one DIA window group).
/// - `frame_ids_sorted`: frames in RT order (caller decides the order)
/// - `clamp_mz_to_group`: if Some(lo,hi), clamp mz scale to this range; else infer from frames
pub fn build_dense_rt_by_mz_ppm_for_group(
    data_handle: &TimsDatasetDIA,
    frame_ids_sorted: Vec<u32>,
    ppm_per_bin: f32,
    mz_pad_ppm: f32,
    maybe_sigma_frames: Option<f32>,
    truncate: f32,
    clamp_mz_to_group: Option<(f32, f32)>,
    num_threads: usize,
) -> RtIndex {
    // 1) RT times (in the provided order)
    let frame_times: Vec<f32> = frame_ids_sorted.iter().map(|fid| {
        data_handle
            .meta_data
            .iter()
            .find(|m| m.id as u32 == *fid)
            .map(|m| m.time as f32)
            .unwrap_or(0.0)
    }).collect();
    let num_frames = frame_ids_sorted.len();

    // 2) materialize frames
    let frames = data_handle
        .get_slice(frame_ids_sorted.clone(), num_threads)
        .frames;

    // 3) m/z range: clamp or scan
    let (mut mz_min, mut mz_max) = if let Some((lo, hi)) = clamp_mz_to_group {
        assert!(lo.is_finite() && hi.is_finite() && hi > lo, "invalid group m/z clamp");
        (lo, hi)
    } else {
        // NOTE: now calls the local function, not `super::...`
        scan_mz_range(&frames).expect("No m/z values found for group")
    };

    if mz_pad_ppm > 0.0 {
        let f = 1.0 + mz_pad_ppm * 1e-6;
        mz_min /= f;
        mz_max *= f;
    }

    let scale = MzScale::build(mz_min.max(10.0f32), mz_max, ppm_per_bin);
    let rows = scale.num_bins();

    // 4) fill matrix [rows x num_frames], column-major
    let mut matrix = vec![0.0f32; rows * num_frames];

    matrix.par_chunks_mut(rows).enumerate().for_each(|(col, col_slice)| {
        let frame = &frames[col];
        let mz = &frame.ims_frame.mz;
        let inten = &frame.ims_frame.intensity;

        // sparse → dense per column
        let mut accum: FxHashMap<usize, f32> = FxHashMap::default();
        for (&m, &i) in mz.iter().zip(inten.iter()) {
            if let Some(row) = scale.index_of(m as f32) {
                *accum.entry(row).or_insert(0.0) += i as f32;
            }
        }
        for (row, val) in accum {
            col_slice[row] += val;
        }
    });

    // 5) optional RT smoothing; keep raw if applied
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

/// Like `build_dense_im_by_rtpeaks_ppm`, but limited to a specific set of frames (e.g., a DIA window group)
/// and a local IM axis built from *disjoint scan ranges*. The returned `ImIndex.scans`
/// are **global physical scan numbers** in compact order.
pub fn build_dense_im_by_rtpeaks_ppm_for_group(
    data_handle: &TimsDatasetDIA,
    peaks: Vec<RtPeak1D>,          // rows
    frame_ids_sorted: Vec<u32>,    // columns are drawn from these frames (RT order)
    scale: &MzScale,               // m/z scale from the group’s RT index (use the same!)
    num_threads: usize,
    mz_ppm_window: f32,            // ±ppm around each peak’s m/z center
    rt_extra_pad: usize,           // extra RT columns to include around each peak’s padded RT window
    maybe_sigma_scans: Option<f32>,
    truncate: f32,
    scan_ranges: Option<Vec<(usize, usize)>>, // <-- unions (merged+sorted), None => no clamp
) -> ImIndex {
    let n_rows = peaks.len();
    if n_rows == 0 {
        return ImIndex { peaks, scans: vec![], data: vec![], rows: 0, cols: 0, data_raw: None };
    }

    // Materialize the group’s frames in the given order.
    let slice = data_handle.get_slice(frame_ids_sorted.clone(), num_threads);
    let frames = slice.frames;
    let ncols_rt = frames.len();
    if ncols_rt == 0 {
        return ImIndex { peaks, scans: vec![], data: vec![], rows: 0, cols: 0, data_raw: None };
    }

    // Determine global maximum scan id across these frames (guarding only).
    let global_num_scans: usize = frames.iter()
        .flat_map(|fr| fr.scan.iter())
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(0);

    // --- Build compact IM axis from unions; keep labels as *global* scans ----
    // phys2local maps physical scan -> Some(local index) if included, else None.
    let (scans_compact, phys2local): (Vec<usize>, Vec<Option<usize>>) = if let Some(ranges) = scan_ranges {
        // Expect ranges merged and sorted; if not guaranteed upstream, merge here.
        let mut scans = Vec::new();
        let mut max_phys = 0usize;
        for (lo, hi) in ranges {
            if lo > hi { continue; }
            max_phys = max_phys.max(hi);
            for s in lo..=hi { scans.push(s); }
        }
        if scans.is_empty() {
            return ImIndex { peaks, scans: vec![], data: vec![], rows: 0, cols: 0, data_raw: None };
        }
        let mut phys2local = vec![None; max_phys + 1];
        for (local, &s_phys) in scans.iter().enumerate() {
            if s_phys < phys2local.len() {
                phys2local[s_phys] = Some(local);
            }
        }
        (scans, phys2local)
    } else {
        // No clamp: include all physical scans that appear in the slice
        let n = global_num_scans;
        let scans: Vec<usize> = (0..n).collect();
        let phys2local: Vec<Option<usize>> = (0..n).map(Some).collect();
        (scans, phys2local)
    };

    let cols = scans_compact.len();
    if cols == 0 {
        return ImIndex { peaks, scans: vec![], data: vec![], rows: 0, cols: 0, data_raw: None };
    }

    // Per-frame bin views over the *same* m/z scale.
    let views: Vec<FrameBinView> = (0..ncols_rt)
        .into_par_iter()
        .map(|i| build_frame_bin_view(frames[i].clone(), scale, global_num_scans))
        .collect();

    let do_smooth = maybe_sigma_scans.unwrap_or(0.0) > 0.0;

    // Build per-row IM profiles over the compact axis.
    let profiles: Vec<(Vec<f32>, Vec<f32>)> = peaks
        .par_iter()
        .map(|p| {
            if p.mz_row >= scale.num_bins() {
                return (vec![0.0; cols], vec![0.0; cols]);
            }

            // RT bounds in *group* column space (cap to slice).
            let mut lo_rt = p.left_padded.saturating_sub(rt_extra_pad);
            let mut hi_rt = p.right_padded.saturating_add(rt_extra_pad);
            if lo_rt >= ncols_rt { lo_rt = ncols_rt.saturating_sub(1); }
            if hi_rt >= ncols_rt { hi_rt = ncols_rt.saturating_sub(1); }
            if lo_rt > hi_rt { return (vec![0.0; cols], vec![0.0; cols]); }

            // m/z band (±ppm)
            let mz_center = p.mz_center;
            let tol = mz_center * mz_ppm_window * 1e-6;
            let (bin_lo, bin_hi) = scale.index_range_for_mz_window(mz_center - tol, mz_center + tol);

            let mut raw = vec![0.0f32; cols];
            for t in lo_rt..=hi_rt {
                let v = &views[t];
                let (start, end) = bin_range(v, bin_lo, bin_hi);
                for i in start..end {
                    let s_phys = v.scan_idx[i] as usize;
                    // keep only if in our unions
                    let Some(s_local) = phys2local.get(s_phys).and_then(|&x| x) else { continue };
                    raw[s_local] += v.intensity[i];
                }
            }

            // Optional IM smoothing per row
            let smoothed = if do_smooth {
                let mut tmp = raw.clone();
                smooth_vector_gaussian(&mut tmp, maybe_sigma_scans.unwrap(), truncate);
                tmp
            } else {
                raw.clone()
            };

            (smoothed, raw)
        })
        .collect();

    // Column-major stack
    let rows = n_rows;
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

    // IMPORTANT: `scans` = global physical scan numbers in compact order
    ImIndex { peaks, scans: scans_compact, data, rows, cols, data_raw }
}