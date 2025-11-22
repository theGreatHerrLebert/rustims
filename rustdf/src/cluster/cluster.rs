use std::hash::{DefaultHasher, Hash, Hasher};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use mscore::timstof::frame::TimsFrame;
use mscore::timstof::slice::TimsSlice;
use crate::cluster::peak::{FrameBinView, ImPeak1D, RtFrames, RtPeak1D};
use crate::cluster::utility::{Fit1D, TofScale, fit1d_moment};
use crate::data::handle::IndexConverter;

fn compute_cluster_id_from_spec(spec: &ClusterSpec1D) -> u64 {
    let mut h = DefaultHasher::new();

    // Integer stuff is straightforward
    spec.rt_lo.hash(&mut h);
    spec.rt_hi.hash(&mut h);
    spec.im_lo.hash(&mut h);
    spec.im_hi.hash(&mut h);
    spec.tof_win.0.hash(&mut h);
    spec.tof_win.1.hash(&mut h);
    spec.tof_hist_bins.hash(&mut h);
    spec.ms_level.hash(&mut h);

    // Options: encode presence + value
    spec.window_group.unwrap_or(0).hash(&mut h);
    spec.parent_im_id.unwrap_or(-1).hash(&mut h);
    spec.parent_rt_id.unwrap_or(-1).hash(&mut h);

    // im_prior_sigma is f32 – hash the bit pattern if present
    let sigma_bits: u32 = spec
        .im_prior_sigma
        .map(|s| s.to_bits())
        .unwrap_or(0u32);
    sigma_bits.hash(&mut h);

    h.finish()
}

#[derive(Copy, Clone, Debug)]
pub struct ScanSlice {
    pub scan: usize,
    pub start: usize,
    pub end: usize
}

pub fn build_scan_slices(fr: &TimsFrame) -> Vec<ScanSlice> {
    let scv = &fr.scan;
    let mut out = Vec::new();
    if scv.is_empty() { return out; }
    let mut s_cur = scv[0];
    let mut i_start = 0usize;
    for i in 1..scv.len() {
        if scv[i] != s_cur {
            if s_cur >= 0 {
                out.push(ScanSlice { scan: s_cur as usize, start: i_start, end: i });
            }
            s_cur = scv[i];
            i_start = i;
        }
    }
    if s_cur >= 0 {
        out.push(ScanSlice { scan: s_cur as usize, start: i_start, end: scv.len() });
    }
    out
}

pub struct RawAttachContext {
    pub slice: TimsSlice,
    pub scan_slices: Vec<Vec<ScanSlice>>,
    pub frame_ids_local: Vec<u32>,
    pub rt_axis_sec: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct Attach1DOptions {
    pub attach_points: bool,
    pub attach_axes: bool,
    pub max_points: Option<usize>, // thinning cap
}

impl  Attach1DOptions {
    pub fn default() -> Self {
        Self {
            attach_points: false,
            attach_axes: false,
            max_points: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Eval1DOpts {
    /// Whether to do a 2nd-pass refine of the TOF/axis window around the argmax.
    pub refine_tof_once: bool,
    /// k for "fallback σ" in IM, and generic σ policies.
    pub refine_k_sigma: f32,
    /// Whether to attach RT / IM / axis centers into the result.
    pub attach_axes: bool,
    /// Whether to attach RT and/or IM traces (XICs).
    pub attach_rt_trace: bool,
    pub attach_im_trace: bool,
    /// Raw point attachment options.
    pub attach: Attach1DOptions,
    pub compute_mz_from_tof: bool,

    /// Padding to add around the final windows (in units).
    pub pad_rt_frames: usize,
    pub pad_im_scans: usize,
    pub pad_tof_bins: usize,
}

impl Default for Eval1DOpts {
    fn default() -> Self {
        Self {
            refine_tof_once: true,
            refine_k_sigma: 3.0,
            attach_axes: false,
            attach_rt_trace: false,
            attach_im_trace: false,
            attach: Attach1DOptions::default(),
            compute_mz_from_tof: false,
            pad_rt_frames: 0,
            pad_im_scans: 0,
            pad_tof_bins: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClusterSpec1D {
    // local (RT) indices inside the provided RtFrames slice (inclusive)
    pub rt_lo: usize,
    pub rt_hi: usize,
    // absolute scan bounds (inclusive) in the frame (TIMS axis)
    pub im_lo: usize,
    pub im_hi: usize,
    // bin-index window [lo, hi] on the TofScale axis (encoded as i32)
    pub tof_win: (i32, i32),
    // histogram resolution along the TOF-backed axis (currently a policy knob)
    pub tof_hist_bins: usize,

    // provenance (optional)
    pub window_group: Option<u32>,
    pub parent_im_id: Option<i64>,
    pub parent_rt_id: Option<i64>,
    pub ms_level: u8,

    // prior σ in scan units, from detector/refiner
    pub im_prior_sigma: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawPoints {
    pub mz: Vec<f32>,      // kept for future use; not used in this module
    pub rt: Vec<f32>,
    pub im: Vec<f32>,
    pub scan: Vec<u32>,
    pub intensity: Vec<f32>,
    pub tof: Vec<i32>,
    pub frame: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterResult1D {
    pub cluster_id: u64,
    pub rt_window: (usize, usize),
    pub im_window: (usize, usize),
    pub tof_window: (usize, usize),

    /// Optional m/z window (not set by this module).
    pub mz_window: Option<(f32, f32)>,

    pub rt_fit: Fit1D,
    pub im_fit: Fit1D,
    /// Fit on the third axis (TOF-backed / generic axis).
    pub tof_fit: Fit1D,
    /// Optional m/z-domain fit (not set by this module).
    pub mz_fit: Option<Fit1D>,

    pub raw_sum: f32,
    pub volume_proxy: f32,

    pub frame_ids_used: Vec<u32>,
    pub window_group: Option<u32>,
    pub parent_im_id: Option<i64>,
    pub parent_rt_id: Option<i64>,
    pub ms_level: u8,

    pub rt_axis_sec: Option<Vec<f32>>,
    pub im_axis_scans: Option<Vec<usize>>,
    /// Optional m/z axis (not set by this module).
    pub mz_axis_da: Option<Vec<f32>>,

    /// Optional attachment of raw points (only set if requested elsewhere).
    pub raw_points: Option<RawPoints>,

    /// Optional RT XIC (intensities along rt_window).
    #[serde(default)]
    pub rt_trace: Option<Vec<f32>>,

    /// Optional IM trace (intensities along im_window).
    #[serde(default)]
    pub im_trace: Option<Vec<f32>>,
}

pub fn decorate_with_mz_for_cluster<D: IndexConverter>(
    ds: &D,
    rt_frames: &RtFrames,
    res: &mut ClusterResult1D,
) {
    let n_frames = rt_frames.frame_ids.len();
    if n_frames == 0 {
        return;
    }

    let scale = &*rt_frames.scale;

    // ----------------------------------------------------
    // 1) Pick a representative frame (mid of final rt_window)
    // ----------------------------------------------------
    let (mut rt_lo, mut rt_hi) = res.rt_window;
    if rt_lo > rt_hi {
        std::mem::swap(&mut rt_lo, &mut rt_hi);
    }
    if rt_lo >= n_frames {
        return;
    }
    rt_hi = rt_hi.min(n_frames.saturating_sub(1));
    if rt_lo > rt_hi {
        return;
    }

    let mid_local = (rt_lo + rt_hi) / 2;
    let frame_id = rt_frames.frame_ids[mid_local];

    // ----------------------------------------------------
    // 2) Final TOF-bin window (CSR bins) → TOF indices
    // ----------------------------------------------------
    let (mut bin_lo, mut bin_hi) = res.tof_window;
    if bin_lo > bin_hi {
        std::mem::swap(&mut bin_lo, &mut bin_hi);
    }

    let n_bins = scale.num_bins();
    if n_bins == 0 {
        return;
    }
    bin_lo = bin_lo.min(n_bins.saturating_sub(1));
    bin_hi = bin_hi.min(n_bins.saturating_sub(1));
    if bin_lo > bin_hi {
        return;
    }

    let tof_lo_idx = scale.center(bin_lo).floor().max(0.0) as u32;
    let tof_hi_idx = scale.center(bin_hi).ceil().max(0.0) as u32;

    // ----------------------------------------------------
    // 3) TOF → m/z for window endpoints
    // ----------------------------------------------------
    let tof_pair: Vec<u32> = vec![tof_lo_idx, tof_hi_idx];
    let mz_pair = ds.tof_to_mz(frame_id, &tof_pair);
    if mz_pair.len() != 2 {
        return;
    }

    let mut mz_lo = mz_pair[0] as f32;
    let mut mz_hi = mz_pair[1] as f32;
    if !mz_lo.is_finite() || !mz_hi.is_finite() {
        return;
    }
    if mz_lo > mz_hi {
        std::mem::swap(&mut mz_lo, &mut mz_hi);
    }

    res.mz_window = Some((mz_lo, mz_hi));

    // ----------------------------------------------------
    // 4) Build an m/z axis (one point per TOF bin)
    // ----------------------------------------------------
    let tof_centers: Vec<u32> = (bin_lo..=bin_hi)
        .map(|b| scale.center(b).round().max(0.0) as u32)
        .collect();

    let mz_axis_f64 = ds.tof_to_mz(frame_id, &tof_centers);
    if mz_axis_f64.len() == tof_centers.len() && !mz_axis_f64.is_empty() {
        let mz_axis_da: Vec<f32> = mz_axis_f64.iter().map(|&x| x as f32).collect();
        res.mz_axis_da = Some(mz_axis_da.clone());

        // ------------------------------------------------
        // 5) Derive an m/z-domain Fit1D from the TOF fit
        // ------------------------------------------------
        if res.tof_fit.mu.is_finite() && res.tof_fit.sigma.is_finite() && res.tof_fit.sigma > 0.0 {
            let tof_mu = res.tof_fit.mu.max(0.0).round() as u32;
            let tof_minus = (res.tof_fit.mu - res.tof_fit.sigma).max(0.0).round() as u32;
            let tof_plus = (res.tof_fit.mu + res.tof_fit.sigma).max(0.0).round() as u32;

            let tof_triplet: Vec<u32> = vec![tof_mu, tof_minus, tof_plus];
            let mz_triplet = ds.tof_to_mz(frame_id, &tof_triplet);
            if mz_triplet.len() == 3 {
                let mu_mz = mz_triplet[0] as f32;
                let sigma_mz = ((mz_triplet[2] - mz_triplet[1]).abs() as f32 * 0.5).max(1e-6);

                let mut mz_fit = res.tof_fit.clone();
                mz_fit.mu = mu_mz;
                mz_fit.sigma = sigma_mz;

                res.mz_fit = Some(mz_fit);
            }
        }
    }
}
// ==========================================================
// Core CSR helpers (RT / IM / TOF axis)
// ==========================================================

#[inline]
fn sum_frame_block(
    fbv: &FrameBinView,
    bin_lo: usize,
    bin_hi: usize,
    im_lo: usize,
    im_hi: usize,
) -> f32 {
    let ub = &fbv.unique_bins;
    if ub.is_empty() || bin_lo > bin_hi {
        return 0.0;
    }

    let start = match ub.binary_search(&bin_lo) {
        Ok(i) => i,
        Err(i) => i.min(ub.len()),
    };

    let mut acc = 0.0f32;
    let mut i = start;
    while i < ub.len() {
        let b = ub[i];
        if b > bin_hi {
            break;
        }

        let lo = fbv.offsets[i];
        let hi = fbv.offsets[i + 1];
        let scans = &fbv.scan_idx[lo..hi];
        let ints = &fbv.intensity[lo..hi];

        for (s, val) in scans.iter().zip(ints.iter()) {
            let s = *s as usize;
            if s >= im_lo && s <= im_hi {
                acc += *val;
            }
        }

        i += 1;
    }

    acc
}

/// Build RT marginal (len = frames in [rt_lo..rt_hi]), using (bin, scan) window.
pub fn build_rt_marginal(
    frames: &[FrameBinView],
    rt_lo: usize,
    rt_hi: usize,
    bin_lo: usize,
    bin_hi: usize,
    im_lo: usize,
    im_hi: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rt_hi + 1 - rt_lo];
    for (k, fbv) in frames[rt_lo..=rt_hi].iter().enumerate() {
        out[k] = sum_frame_block(fbv, bin_lo, bin_hi, im_lo, im_hi);
    }
    out
}

/// Build IM marginal (absolute scan axis).
pub fn build_im_marginal(
    frames: &[FrameBinView],
    rt_lo: usize,
    rt_hi: usize,
    bin_lo: usize,
    bin_hi: usize,
    im_lo: usize,
    im_hi: usize,
) -> Vec<f32> {
    let len = im_hi + 1 - im_lo;
    let mut out = vec![0.0f32; len];

    for fbv in &frames[rt_lo..=rt_hi] {
        let ub = &fbv.unique_bins;
        if ub.is_empty() {
            continue;
        }

        let start = match ub.binary_search(&bin_lo) {
            Ok(i) => i,
            Err(i) => i.min(ub.len()),
        };

        let mut i = start;
        while i < ub.len() {
            let b = ub[i];
            if b > bin_hi {
                break;
            }
            let lo = fbv.offsets[i];
            let hi = fbv.offsets[i + 1];
            let scans = &fbv.scan_idx[lo..hi];
            let ints = &fbv.intensity[lo..hi];

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

/// Build TOF/axis histogram over CSR bins [bin_lo..bin_hi].
///
/// The centers returned are whatever `TofScale::center(i)` yields
/// (historically m/z, but here treated as a generic axis).
pub fn build_tof_hist(
    frames: &[FrameBinView],
    rt_lo: usize,
    rt_hi: usize,
    bin_lo: usize,
    bin_hi: usize,
    im_lo: usize,
    im_hi: usize,
    scale: &TofScale,
) -> (Vec<f32>, Vec<f32>) {
    let r = bin_hi + 1 - bin_lo;
    let mut hist = vec![0.0f32; r];

    for fbv in &frames[rt_lo..=rt_hi] {
        let ub = &fbv.unique_bins;
        if ub.is_empty() {
            continue;
        }

        let start = match ub.binary_search(&bin_lo) {
            Ok(i) => i,
            Err(i) => i.min(ub.len()),
        };

        let mut i = start;
        while i < ub.len() {
            let b = ub[i];
            if b > bin_hi {
                break;
            }

            let lo = fbv.offsets[i];
            let hi = fbv.offsets[i + 1];

            let scans = &fbv.scan_idx[lo..hi];
            let ints = &fbv.intensity[lo..hi];

            let mut sum = 0.0f32;
            for (s, val) in scans.iter().zip(ints.iter()) {
                let s_abs = *s as usize;
                if s_abs >= im_lo && s_abs <= im_hi {
                    sum += *val;
                }
            }

            hist[b - bin_lo] += sum;
            i += 1;
        }
    }

    let centers = (bin_lo..=bin_hi).map(|i| scale.center(i)).collect::<Vec<_>>();
    (hist, centers)
}

#[inline]
fn sanitize_fit(
    f: &mut Fit1D,
    mu_bounds: Option<(f32, f32)>,
    min_sigma: f32,
    enforce_nonneg: bool,
) {
    // μ
    if !f.mu.is_finite() {
        if let Some((lo, hi)) = mu_bounds {
            if lo.is_finite() && hi.is_finite() && lo <= hi {
                f.mu = if hi > lo { 0.5 * (lo + hi) } else { lo };
            } else {
                f.mu = 0.0;
            }
        } else {
            f.mu = 0.0;
        }
    }
    // σ, height, baseline, area, r2
    if !f.sigma.is_finite() {
        f.sigma = 0.0;
    }
    if !f.height.is_finite() {
        f.height = 0.0;
    }
    if !f.baseline.is_finite() {
        f.baseline = 0.0;
    }
    if !f.area.is_finite() {
        f.area = 0.0;
    }
    if !f.r2.is_finite() {
        f.r2 = 0.0;
    }

    if let Some((lo, hi)) = mu_bounds {
        if lo.is_finite() && hi.is_finite() && lo < hi {
            if f.mu < lo {
                f.mu = lo;
            }
            if f.mu > hi {
                f.mu = hi;
            }
        }
    }

    if f.sigma < 0.0 {
        f.sigma = 0.0;
    }
    if f.sigma > 0.0 && f.sigma < min_sigma {
        f.sigma = min_sigma;
    }

    if enforce_nonneg {
        if f.baseline < 0.0 {
            f.baseline = 0.0;
        }
        if f.height < 0.0 {
            f.height = 0.0;
        }
        if f.area < 0.0 {
            f.area = 0.0;
        }
    }
}

#[inline]
fn argmax_idx(v: &[f32]) -> usize {
    let mut i_max = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &y) in v.iter().enumerate() {
        if y > best {
            best = y;
            i_max = i;
        }
    }
    i_max
}

/// Sub-bin parabolic apex around the argmax bin.
#[inline]
fn parabolic_refine(centers: &[f32], hist: &[f32], i: usize) -> f32 {
    if i == 0 || i + 1 >= hist.len() {
        return centers[i];
    }
    let y0 = hist[i - 1];
    let y1 = hist[i];
    let y2 = hist[i + 1];
    let x0 = centers[i - 1];
    let x1 = centers[i];
    let x2 = centers[i + 1];
    let denom = y0 - 2.0 * y1 + y2;
    if denom.abs() <= 1e-12 {
        return centers[i];
    }
    let dx = 0.5 * (y0 - y2) / denom * ((x2 - x0) * 0.5);
    let x = x1 + dx;
    let half_bin = ((x2 - x0) * 0.5).abs();
    x.clamp(x1 - half_bin, x1 + half_bin)
}

/// σ from FWHM (linear interpolation across half-height crossings).
/// Returns (sigma, x_left_50, x_right_50).
fn fwhm_sigma_from_hist(centers: &[f32], hist: &[f32], i_max: usize) -> Option<(f32, f32, f32)> {
    let peak = *hist.get(i_max)?;
    if !peak.is_finite() || peak <= 0.0 {
        return None;
    }
    let half = 0.5 * peak;

    // left crossing
    let mut left = i_max;
    while left > 0 && hist[left] >= half {
        left -= 1;
    }
    if left == i_max {
        return None;
    }
    let (xl0, yl0) = (centers[left], hist[left]);
    let (xl1, yl1) = (centers[left + 1], hist[left + 1]);
    if (yl1 - yl0).abs() <= 1e-12 {
        return None;
    }
    let x_left = xl0 + (half - yl0) * (xl1 - xl0) / (yl1 - yl0);

    // right crossing
    let mut right = i_max;
    while right + 1 < hist.len() && hist[right] >= half {
        right += 1;
    }
    if right == i_max {
        return None;
    }
    let (xr0, yr0) = (centers[right - 1], hist[right - 1]);
    let (xr1, yr1) = (centers[right], hist[right]);
    if (yr1 - yr0).abs() <= 1e-12 {
        return None;
    }
    let x_right = xr0 + (half - yr0) * (xr1 - xr0) / (yr1 - yr0);

    let fwhm = (x_right - x_left).abs();
    if !fwhm.is_finite() || fwhm <= 0.0 {
        return None;
    }

    // σ = FWHM / (2*sqrt(2 ln 2))
    let two_sqrt_two_ln2 = 2.0 * 2.0_f32.ln().sqrt();
    let sigma = fwhm / two_sqrt_two_ln2;

    if sigma.is_finite() && sigma > 0.0 {
        Some((sigma, x_left.min(x_right), x_left.max(x_right)))
    } else {
        None
    }
}

// ==========================================================
// Helpers for RT overlap and spec building
// ==========================================================

#[inline]
fn rt_overlap((a0,a1):(usize, usize),(b0,b1):(usize,usize)) -> usize {
    if a0 > a1 || b0 > b1 { return 0; }
    let lo = a0.max(b0);
    let hi = a1.min(b1);
    hi.saturating_sub(lo).saturating_add(1)
}

/// Helper: derive a conservative RT window for an IM peak in local frame indices.
/// Uses the IM peak’s stored `rt_bounds` on the same grid.
fn rt_bounds_from_im(im: &ImPeak1D, n_frames: usize) -> (usize, usize) {
    let (a,b) = im.rt_bounds; // (usize, usize) on the same grid as RtFrames
    let mut lo = a.min(n_frames.saturating_sub(1));
    let mut hi = b.min(n_frames.saturating_sub(1));
    if lo > hi { std::mem::swap(&mut lo, &mut hi); }
    (lo, hi)
}

/// Interpret `tof_win` as a bin-index range [lo..hi] encoded as i32.
#[inline]
pub fn bin_range_for_win(_scale: &TofScale, tof_win: (i32, i32)) -> (usize, usize) {
    let (a, b) = tof_win;
    let mut lo = a.min(b).max(0) as usize;
    let mut hi = a.max(b).max(0) as usize;
    if lo > hi {
        std::mem::swap(&mut lo, &mut hi);
    }
    (lo, hi)
}

// ==========================================================
// Main evaluator (grid-based, CSR/TofScale)
// ==========================================================

pub fn evaluate_spec_1d(
    rt_frames: &RtFrames,
    spec: &ClusterSpec1D,
    opts: &Eval1DOpts,
) -> ClusterResult1D {
    debug_assert!(rt_frames.is_consistent());
    let scale = &*rt_frames.scale;

    let n_frames = rt_frames.frames.len();
    let n_bins   = scale.num_bins();

    // -------------------------------
    // 0) Effective RT / IM / TOF windows with padding
    // -------------------------------

    // Start from spec RT window and clamp + pad in frame space
    let mut rt_lo_eff = spec.rt_lo.min(n_frames.saturating_sub(1));
    let mut rt_hi_eff = spec.rt_hi.min(n_frames.saturating_sub(1));
    if rt_lo_eff > rt_hi_eff {
        std::mem::swap(&mut rt_lo_eff, &mut rt_hi_eff);
    }
    if opts.pad_rt_frames > 0 && n_frames > 0 {
        let pad = opts.pad_rt_frames;
        rt_lo_eff = rt_lo_eff.saturating_sub(pad);
        rt_hi_eff = (rt_hi_eff + pad).min(n_frames.saturating_sub(1));
    }

    // IM window – can pad without clamping to a global max (no bound known here).
    let mut im_lo_eff = spec.im_lo;
    let mut im_hi_eff = spec.im_hi;
    if im_lo_eff > im_hi_eff {
        std::mem::swap(&mut im_lo_eff, &mut im_hi_eff);
    }
    if opts.pad_im_scans > 0 {
        let pad = opts.pad_im_scans;
        im_lo_eff = im_lo_eff.saturating_sub(pad);
        im_hi_eff = im_hi_eff.saturating_add(pad);
    }

    // TOF: start from spec.tof_win → CSR bin range, then pad and clamp to axis.
    let (mut bin_lo0, mut bin_hi0) = bin_range_for_win(scale, spec.tof_win);
    if opts.pad_tof_bins > 0 && n_bins > 0 {
        let pad = opts.pad_tof_bins.min(n_bins.saturating_sub(1));
        bin_lo0 = bin_lo0.saturating_sub(pad);
        bin_hi0 = (bin_hi0 + pad).min(n_bins.saturating_sub(1));
    }

    // --- 1) first pass accumulation ---------------------------------------
    let rt_marg1 = build_rt_marginal(
        &rt_frames.frames,
        rt_lo_eff,
        rt_hi_eff,
        bin_lo0,
        bin_hi0,
        im_lo_eff,
        im_hi_eff,
    );
    let im_marg1 = build_im_marginal(
        &rt_frames.frames,
        rt_lo_eff,
        rt_hi_eff,
        bin_lo0,
        bin_hi0,
        im_lo_eff,
        im_hi_eff,
    );
    let (axis_hist1, axis_centers1) = build_tof_hist(
        &rt_frames.frames,
        rt_lo_eff,
        rt_hi_eff,
        bin_lo0,
        bin_hi0,
        im_lo_eff,
        im_hi_eff,
        scale,
    );

    let rt_sum1: f32 = rt_marg1.iter().copied().sum();
    let im_sum1: f32 = im_marg1.iter().copied().sum();
    let ax_sum1: f32 = axis_hist1.iter().copied().sum();

    let cluster_id = compute_cluster_id_from_spec(spec);

    // Early exit: no signal in any axis
    if rt_sum1 <= 0.0 || im_sum1 <= 0.0 || ax_sum1 <= 0.0 {
        return ClusterResult1D {
            cluster_id,
            rt_window: (rt_lo_eff, rt_hi_eff),
            im_window: (im_lo_eff, im_hi_eff),
            tof_window: (bin_lo0, bin_hi0),

            mz_window: None,
            rt_fit: Fit1D::default(),
            im_fit: Fit1D::default(),
            tof_fit: Fit1D::default(),
            mz_fit: None,

            raw_sum: 0.0,
            volume_proxy: 0.0,
            frame_ids_used: rt_frames.frame_ids[rt_lo_eff..=rt_hi_eff].to_vec(),
            window_group: spec.window_group,
            parent_im_id: spec.parent_im_id,
            parent_rt_id: spec.parent_rt_id,
            ms_level: spec.ms_level,
            rt_axis_sec: opts
                .attach_axes
                .then_some(rt_frames.rt_times[rt_lo_eff..=rt_hi_eff].to_vec()),
            im_axis_scans: opts
                .attach_axes
                .then_some((im_lo_eff..=im_hi_eff).collect()),
            mz_axis_da: None,
            raw_points: None,
            rt_trace: opts
                .attach_rt_trace
                .then_some(thin_f32_vec(&rt_marg1, None)),
            im_trace: opts
                .attach_im_trace
                .then_some(thin_f32_vec(&im_marg1, None)),
        };
    }

    // --- 2) argmax + optional refine of bin window ------------------------
    let i_max1 = argmax_idx(&axis_hist1);

    let (bin_lo, bin_hi, axis_centers2) = {
        if opts.refine_tof_once {
            let center_bin = bin_lo0 + i_max1; // local index into [bin_lo0..bin_hi0]
            let half_span: usize = 4;

            let bl = center_bin.saturating_sub(half_span).max(bin_lo0);
            let bh = (center_bin + half_span).min(bin_hi0);

            (bl, bh, (bl..=bh).map(|i| scale.center(i)).collect::<Vec<_>>())
        } else {
            (bin_lo0, bin_hi0, axis_centers1.clone())
        }
    };

    // --- 3) re-accumulate in refined window -------------------------------
    let rt_marg = build_rt_marginal(
        &rt_frames.frames,
        rt_lo_eff,
        rt_hi_eff,
        bin_lo,
        bin_hi,
        im_lo_eff,
        im_hi_eff,
    );
    let im_marg = build_im_marginal(
        &rt_frames.frames,
        rt_lo_eff,
        rt_hi_eff,
        bin_lo,
        bin_hi,
        im_lo_eff,
        im_hi_eff,
    );
    let (axis_hist, _axis_centers_final) = build_tof_hist(
        &rt_frames.frames,
        rt_lo_eff,
        rt_hi_eff,
        bin_lo,
        bin_hi,
        im_lo_eff,
        im_hi_eff,
        scale,
    );

    let rt_sum: f32 = rt_marg.iter().copied().sum();
    let im_sum: f32 = im_marg.iter().copied().sum();
    let ax_sum: f32 = axis_hist.iter().copied().sum();

    let (use_bin_lo, use_bin_hi, use_rt_marg, use_im_marg, use_axis_hist, use_axis_centers) =
        if rt_sum <= 0.0 || im_sum <= 0.0 || ax_sum <= 0.0 {
            (bin_lo0, bin_hi0, &rt_marg1, &im_marg1, &axis_hist1, &axis_centers1)
        } else {
            (bin_lo, bin_hi, &rt_marg, &im_marg, &axis_hist, &axis_centers2)
        };

    // capture traces (possibly thinned)
    let rt_trace_opt = if opts.attach_rt_trace {
        Some(thin_f32_vec(use_rt_marg, None))
    } else {
        None
    };
    let im_trace_opt = if opts.attach_im_trace {
        Some(thin_f32_vec(use_im_marg, None))
    } else {
        None
    };

    // --- 4) final fits: RT/IM moments, axis argmax + FWHM σ ---------------
    let rt_times: Vec<f32> = rt_frames.rt_times[rt_lo_eff..=rt_hi_eff].to_vec();
    let im_axis: Vec<usize> = (im_lo_eff..=im_hi_eff).collect();
    let im_axis_f32: Vec<f32> = im_axis.iter().map(|&s| s as f32).collect();

    let mut rt_fit = fit1d_moment(use_rt_marg, Some(&rt_times));
    let mut im_fit = fit1d_moment(use_im_marg, Some(&im_axis_f32));

    // third axis (TOF-backed / generic)
    let i_max = argmax_idx(use_axis_hist);
    let mu_raw = parabolic_refine(use_axis_centers, use_axis_hist, i_max);

    let (sigma_est, _fit_lo_50, _fit_hi_50) =
        match fwhm_sigma_from_hist(use_axis_centers, use_axis_hist, i_max) {
            Some((s, xl, xr)) => (s, xl, xr),
            None => {
                // fallback σ from local bin width
                let bw = if i_max + 1 < use_axis_centers.len() {
                    (use_axis_centers[i_max + 1] - use_axis_centers[i_max]).abs()
                } else if i_max > 0 {
                    (use_axis_centers[i_max] - use_axis_centers[i_max - 1]).abs()
                } else {
                    0.0
                };
                let s = (bw / 2.355).max(1e-6);
                let fwhm = 2.355 * s;
                (s, mu_raw - 0.5 * fwhm, mu_raw + 0.5 * fwhm)
            }
        };

    // shape-driven bounds: keep ±k·σ around μ
    let k_bounds = 1.0_f32;
    let min_sigma = 1e-6;
    let sigma_clamped = if sigma_est.is_finite() && sigma_est > 0.0 {
        sigma_est.max(min_sigma)
    } else {
        min_sigma
    };

    let raw_lo = mu_raw - k_bounds * sigma_clamped;
    let raw_hi = mu_raw + k_bounds * sigma_clamped;

    let axis_min = *use_axis_centers.first().unwrap_or(&mu_raw);
    let axis_max = *use_axis_centers.last().unwrap_or(&mu_raw);

    let _fit_lo = raw_lo.max(axis_min);
    let _fit_hi = raw_hi.min(axis_max);

    let mut tof_fit = Fit1D {
        mu: mu_raw,
        sigma: sigma_clamped,
        height: use_axis_hist[i_max],
        baseline: 0.0,
        area: use_axis_hist.iter().copied().sum(),
        r2: 1.0,
        n: use_axis_hist.len(),
    };

    // sanitize + IM fallbacks
    let rt_mu_bounds =
        if rt_lo_eff < rt_frames.rt_times.len() && rt_hi_eff < rt_frames.rt_times.len() {
            Some((rt_frames.rt_times[rt_lo_eff], rt_frames.rt_times[rt_hi_eff]))
        } else {
            None
        };
    let axis_mu_bounds = Some((axis_min, axis_max));
    let im_mu_bounds = Some((im_lo_eff as f32, im_hi_eff as f32));

    sanitize_fit(&mut tof_fit, axis_mu_bounds, 1e-6, true);
    sanitize_fit(&mut rt_fit, rt_mu_bounds, 1e-6, true);
    sanitize_fit(&mut im_fit, im_mu_bounds, 1e-3, true);

    if im_fit.mu == 0.0 {
        let lo = im_lo_eff as f32;
        let hi = im_hi_eff as f32;
        im_fit.mu = if hi > lo { 0.5 * (lo + hi) } else { lo.max(1.0) };
    }
    if !im_fit.sigma.is_finite() || im_fit.sigma <= 0.0 {
        let k = opts.refine_k_sigma.max(1.0);
        let width = (im_hi_eff.saturating_sub(im_lo_eff) as f32) + 1.0;
        let from_window = ((width - 1.0) / (2.0 * k)).max(0.0);
        let prior = spec.im_prior_sigma.unwrap_or(0.0).max(0.0);
        let mut s = prior.max(from_window).max(1e-3);
        if !s.is_finite() {
            s = 1e-3;
        }
        im_fit.sigma = s;
    }

    let raw_sum = use_rt_marg.iter().copied().sum();
    let volume_proxy = (rt_fit.area.max(0.0))
        * (im_fit.area.max(0.0))
        * (tof_fit.area.max(0.0));

    ClusterResult1D {
        cluster_id,
        rt_window: (rt_lo_eff, rt_hi_eff),
        im_window: (im_lo_eff, im_hi_eff),
        tof_window: (use_bin_lo, use_bin_hi),

        mz_window: None,   // not set here
        rt_fit,
        im_fit,
        tof_fit,
        mz_fit: None,      // not set here

        raw_sum,
        volume_proxy,
        frame_ids_used: rt_frames.frame_ids[rt_lo_eff..=rt_hi_eff].to_vec(),
        window_group: spec.window_group,
        parent_im_id: spec.parent_im_id,
        parent_rt_id: spec.parent_rt_id,
        ms_level: spec.ms_level,
        rt_axis_sec: opts.attach_axes.then_some(rt_times),
        im_axis_scans: opts.attach_axes.then_some(im_axis),
        mz_axis_da: None,  // not set here
        raw_points: None,
        rt_trace: rt_trace_opt,
        im_trace: im_trace_opt,
    }
}

// ==========================================================
// Spec building (IM+RT → ClusterSpec1D)
// ==========================================================

#[derive(Clone, Copy, Debug)]
pub struct BuildSpecOpts {
    /// Extra RT padding (in local frame indices) around the RT window
    /// derived from the RT peak / IM peak overlap.
    pub extra_rt_pad: usize,

    /// Extra IM padding (in scan indices) around the IM window derived
    /// from the IM peak [left, right] scan range.
    pub extra_im_pad: usize,

    /// Symmetric padding in *TOF bins* around the minimal TOF window
    /// implied by the IM peak.
    pub tof_bin_pad: usize,

    /// Histogram resolution along the TOF-backed axis.
    pub tof_hist_bins: usize,

    /// MS level that this spec is intended for (1 = MS1, 2 = MS2/DIA).
    pub ms_level: u8,

    /// Minimum IM span in absolute scan units (e.g. 10).
    pub min_im_span: usize,

    /// k for σ logic in IM: when we need a fallback σ_im, we ensure that
    /// the window roughly covers ±k·σ around the IM peak apex.
    pub im_k_sigma: f32,
}

impl BuildSpecOpts {
    #[inline]
    pub fn with_ms_level(self, level: u8) -> Self {
        let mut o = self;
        o.ms_level = level;
        o
    }

    #[inline]
    pub fn ms1_defaults() -> Self {
        BuildSpecOpts {
            extra_rt_pad: 0,
            extra_im_pad: 0,
            tof_bin_pad: 0,
            tof_hist_bins: 16,
            ms_level: 1,
            min_im_span: 10,
            im_k_sigma: 3.0,
        }
    }

    #[inline]
    pub fn ms2_defaults() -> Self {
        BuildSpecOpts {
            extra_rt_pad: 0,
            extra_im_pad: 0,
            tof_bin_pad: 0,
            tof_hist_bins: 16,
            ms_level: 2,
            min_im_span: 10,
            im_k_sigma: 3.0,
        }
    }
}

#[inline]
fn widen_scan_window(lo: usize, hi: usize, want_span: usize, center: usize) -> (usize, usize) {
    let cur_span = hi.saturating_sub(lo).saturating_add(1);
    if cur_span >= want_span { return (lo, hi); }
    let half = want_span / 2;
    let new_lo = center.saturating_sub(half);
    let new_hi = center.saturating_add(want_span - 1 - half);
    (new_lo.min(lo), new_hi.max(hi))  // keep original covered, but ensure ≥ want_span overall
}

#[inline]
fn scans_from_sigma(k: f32, sigma: f32) -> usize {
    // cover ±kσ around the apex, +1 for inclusive range
    let k = k.max(0.0);
    let s = sigma.max(0.0);
    let span = (2.0 * k * s).ceil() as isize + 1;
    span.max(1) as usize
}

/// Map a bin-index window from ImPeak1D (`tof_bounds`) to a clamped CSR bin range.
#[inline]
fn axis_bin_range_for_bounds(scale: &TofScale, bounds: (i32, i32)) -> (usize, usize) {
    let (mut lo, mut hi) = bounds;
    if lo > hi {
        std::mem::swap(&mut lo, &mut hi);
    }
    let n_bins = scale.edges.len().saturating_sub(1);
    if n_bins == 0 {
        return (0, 0);
    }
    let lo_u = lo.max(0) as usize;
    let hi_u = hi.max(0) as usize;
    (lo_u.min(n_bins - 1), hi_u.min(n_bins - 1))
}

pub fn make_spec_from_pair(
    im: &ImPeak1D,
    rt: &RtPeak1D,
    rt_frames: &RtFrames,
    opts: &BuildSpecOpts,
) -> ClusterSpec1D {
    let frames_len = rt_frames.frames.len();
    let scale      = &*rt_frames.scale;

    // 1) RT window (local frame indices) + padding
    let (mut rt_lo, mut rt_hi) = rt.rt_bounds_frames;
    rt_lo = rt_lo
        .saturating_sub(opts.extra_rt_pad)
        .min(frames_len.saturating_sub(1));
    rt_hi = rt_hi
        .saturating_add(opts.extra_rt_pad)
        .min(frames_len.saturating_sub(1));
    if rt_lo > rt_hi {
        std::mem::swap(&mut rt_lo, &mut rt_hi);
    }

    // 2) IM window (scan indices) with min span & prior σ
    let mut im_lo = im.left.saturating_sub(opts.extra_im_pad);
    let mut im_hi = im.right.saturating_add(opts.extra_im_pad);
    if im_lo > im_hi {
        std::mem::swap(&mut im_lo, &mut im_hi);
    }

    let prior_sigma = im.scan_sigma.unwrap_or(0.0).max(0.0);
    let k_im = if opts.im_k_sigma.is_finite() {
        opts.im_k_sigma.max(0.0)
    } else {
        3.0
    };

    let want_from_prior = if prior_sigma > 0.0 {
        scans_from_sigma(k_im, prior_sigma)
    } else {
        0
    };
    let want_from_measured = im.width_scans.max(2);

    let mut want_span = opts.min_im_span.max(want_from_measured);
    if want_from_prior > 0 {
        want_span = want_span.max(want_from_prior);
    }

    let (w_lo, w_hi) = widen_scan_window(im_lo, im_hi, want_span, im.scan);
    im_lo = w_lo;
    im_hi = w_hi;

    // 3) TOF-bin window (no ppm, pad in bin space)
    let axis_bounds = im.tof_bounds; // bin-ish bounds from detector
    let (mut bin_lo, mut bin_hi) = axis_bin_range_for_bounds(scale, axis_bounds);
    if bin_lo > bin_hi {
        std::mem::swap(&mut bin_lo, &mut bin_hi);
    }

    let n_bins = scale.edges.len().saturating_sub(1);
    if n_bins > 0 {
        let pad = opts.tof_bin_pad;
        if pad > 0 {
            bin_lo = bin_lo.saturating_sub(pad);
            bin_hi = (bin_hi + pad).min(n_bins - 1);
        }
    } else {
        bin_lo = 0;
        bin_hi = 0;
    }

    let tof_win = (bin_lo as i32, bin_hi as i32);

    ClusterSpec1D {
        rt_lo,
        rt_hi,
        im_lo,
        im_hi,
        tof_win,
        tof_hist_bins: opts.tof_hist_bins.max(16),
        window_group: im.window_group,
        parent_im_id: Some(im.id),
        parent_rt_id: Some(rt.id),
        ms_level: opts.ms_level,
        im_prior_sigma: if prior_sigma > 0.0 { Some(prior_sigma) } else { None },
    }
}

// ==========================================================
// Parallel spec construction
// ==========================================================

pub fn make_specs_from_im_and_rt_groups_threads(
    im_peaks: &[ImPeak1D],
    rt_groups: &[Vec<RtPeak1D>],
    rt_frames: &RtFrames,
    opts: &BuildSpecOpts,
    require_rt_overlap: bool,
    num_threads: usize,
) -> Vec<ClusterSpec1D> {
    assert_eq!(im_peaks.len(), rt_groups.len(), "rt_groups length mismatch");
    let n_frames = rt_frames.frames.len();
    if n_frames == 0 || im_peaks.is_empty() { return Vec::new(); }

    let build = || {
        (0..im_peaks.len()).into_par_iter()
            .flat_map_iter(|i| {
                let im = &im_peaks[i];
                let rts = &rt_groups[i];

                let mut out: Vec<ClusterSpec1D> = Vec::new();
                if rts.is_empty() { return out.into_iter(); }

                let im_rt = rt_bounds_from_im(im, n_frames);

                for rt in rts {
                    let mut rt_lo = rt.rt_bounds_frames.0.min(n_frames.saturating_sub(1));
                    let mut rt_hi = rt.rt_bounds_frames.1.min(n_frames.saturating_sub(1));
                    if rt_lo > rt_hi { std::mem::swap(&mut rt_lo, &mut rt_hi); }

                    if require_rt_overlap && rt_overlap((rt_lo, rt_hi), im_rt) == 0 {
                        continue;
                    }

                    let mut spec = make_spec_from_pair(im, rt, rt_frames, opts);
                    spec.rt_lo = spec.rt_lo.min(n_frames.saturating_sub(1));
                    spec.rt_hi = spec.rt_hi.min(n_frames.saturating_sub(1));
                    if spec.rt_lo > spec.rt_hi { std::mem::swap(&mut spec.rt_lo, &mut spec.rt_hi); }
                    if spec.im_lo > spec.im_hi { std::mem::swap(&mut spec.im_lo, &mut spec.im_hi); }

                    out.push(spec);
                }

                out.into_iter()
            })
            .collect::<Vec<_>>()
    };

    if num_threads == 0 {
        build()
    } else {
        ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap().install(build)
    }
}

// ==========================================================
// Helpers for axis padding + search
// ==========================================================

#[inline]
fn cushion_hi_edge(scale: &TofScale, hi_edge: f32) -> f32 {
    // Generic: expand the upper edge by a small fraction of a bin width,
    // clamp to the axis max (last edge).
    let axis_max = scale
        .edges
        .last()
        .copied()
        .unwrap_or(hi_edge);

    let n = scale.edges.len();
    let bin_width = if n >= 2 {
        let span = scale.edges[n - 1] - scale.edges[0];
        if span.is_finite() && span != 0.0 {
            (span / (n as f32 - 1.0)).abs()
        } else {
            0.0
        }
    } else {
        0.0
    };

    let eps = 0.25 * bin_width; // small cushion, axis-agnostic
    (hi_edge + eps).min(axis_max)
}

#[inline]
fn lower_bound_in(mz: &[f64], start: usize, end: usize, x: f32) -> usize {
    let mut lo = start;
    let mut hi = end;
    let xf = x as f64;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if mz[mid] < xf {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline]
fn upper_bound_in(mz: &[f64], start: usize, end: usize, x: f32) -> usize {
    let mut lo = start;
    let mut hi = end;
    let xf = x as f64;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if mz[mid] <= xf {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline]
fn thin_stride(total: usize, cap: usize) -> usize {
    if cap == 0 || total <= cap {
        1
    } else {
        (total + cap - 1) / cap
    }
}

// ==========================================================
// Context-based raw attachment (no I/O here)
// ==========================================================

pub fn attach_raw_points_for_spec_1d_in_ctx(
    ctx: &RawAttachContext,
    scale: &TofScale,
    final_bin_lo: usize,
    final_bin_hi: usize,
    final_im_lo: usize,
    final_im_hi: usize,
    final_rt_lo: usize,
    final_rt_hi: usize,
    max_points: Option<usize>,
) -> RawPoints {
    let n_bins = scale.num_bins();
    if n_bins == 0 {
        return RawPoints::default();
    }

    // clamp bin indices
    let mut bin_lo = final_bin_lo.min(n_bins.saturating_sub(1));
    let mut bin_hi = final_bin_hi.min(n_bins.saturating_sub(1));
    if bin_lo > bin_hi {
        std::mem::swap(&mut bin_lo, &mut bin_hi);
    }

    // axis window from bin edges
    let axis_lo = scale.edges[bin_lo];
    let hi_edge_idx = (bin_hi + 1).min(scale.edges.len().saturating_sub(1));
    let axis_hi = cushion_hi_edge(scale, scale.edges[hi_edge_idx]);

    // reuse the pre-built slice and scan_slices
    let slice = &ctx.slice;
    let scan_slices = &ctx.scan_slices;

    // defensive clamp of RT window to slice length
    let n_frames = slice.frames.len();
    if n_frames == 0 {
        return RawPoints::default();
    }
    let rt_lo = final_rt_lo.min(n_frames.saturating_sub(1));
    let rt_hi = final_rt_hi.min(n_frames.saturating_sub(1));
    if rt_lo > rt_hi {
        return RawPoints::default();
    }

    // 1) count
    let mut total = 0usize;
    for fi in rt_lo..=rt_hi {
        let fr = &slice.frames[fi];
        let mz = &fr.ims_frame.mz;
        for sl in &scan_slices[fi] {
            if sl.scan < final_im_lo || sl.scan > final_im_hi {
                continue;
            }
            let l = lower_bound_in(mz, sl.start, sl.end, axis_lo);
            let r = upper_bound_in(mz, sl.start, sl.end, axis_hi);
            total += r.saturating_sub(l);
        }
    }

    // If you still want a "last chance widen", you can re-introduce it here
    // by expanding bin_lo/bin_hi ±1 and recomputing total. For now we keep it
    // simple and just bail out when total == 0.
    if total == 0 {
        return RawPoints::default();
    }

    let stride = max_points
        .map(|cap| thin_stride(total, cap))
        .unwrap_or(1);
    let reserve = total / stride + 8;

    let mut pts = RawPoints {
        mz: Vec::with_capacity(reserve),
        rt: Vec::with_capacity(reserve),
        im: Vec::with_capacity(reserve),
        scan: Vec::with_capacity(reserve),
        intensity: Vec::with_capacity(reserve),
        tof: Vec::with_capacity(reserve),
        frame: Vec::with_capacity(reserve),
    };

    let rt_axis_sec = &ctx.rt_axis_sec[rt_lo..=rt_hi];
    let frame_ids_local = &ctx.frame_ids_local[rt_lo..=rt_hi];

    // 2) extract with thinning
    let mut seen = 0usize;
    for fi in rt_lo..=rt_hi {
        let fr = &slice.frames[fi];
        let mz   = &fr.ims_frame.mz;
        let it   = &fr.ims_frame.intensity;
        let ims  = &fr.ims_frame.mobility;
        let tofs = &fr.tof;

        let len_all = mz.len().min(it.len()).min(ims.len()).min(tofs.len());
        let rt_val = rt_axis_sec[fi - rt_lo];
        let frame_id = frame_ids_local[fi - rt_lo];

        for sl in &scan_slices[fi] {
            let s_abs = sl.scan;
            if s_abs < final_im_lo || s_abs > final_im_hi {
                continue;
            }

            let mut l = lower_bound_in(mz, sl.start, sl.end, axis_lo);
            let mut r = upper_bound_in(mz, sl.start, sl.end, axis_hi);
            if r > len_all {
                r = len_all;
            }
            if l >= r {
                continue;
            }

            while l < r {
                if stride == 1 || (seen % stride == 0) {
                    pts.mz.push(mz[l] as f32);
                    pts.rt.push(rt_val);
                    pts.im.push(ims[l] as f32);
                    pts.scan.push(s_abs as u32);
                    pts.intensity.push(it[l] as f32);
                    pts.frame.push(frame_id);
                    pts.tof.push(tofs[l]);
                }
                seen += 1;
                l += 1;
            }
        }
    }

    pts
}

#[inline]
fn thin_f32_vec(v: &[f32], cap: Option<usize>) -> Vec<f32> {
    let n = v.len();
    let stride = cap.map(|c| thin_stride(n, c)).unwrap_or(1);
    if stride <= 1 {
        return v.to_vec();
    }
    let mut out = Vec::with_capacity((n + stride - 1) / stride);
    let mut i = 0usize;
    while i < n {
        out.push(v[i]);
        i += stride;
    }
    out
}

/// For a set of clusters that are all defined on the same RtFrames/grid
/// (i.e. same slice/context), extract RawPoints per cluster.
///
/// This is the "does the box actually give us the right raw points?" primitive.
pub fn extract_raw_points_for_clusters_in_ctx(
    ctx: &RawAttachContext,
    scale: &TofScale,
    clusters: &[ClusterResult1D],
    max_points: Option<usize>,
) -> Vec<RawPoints> {
    clusters
        .iter()
        .map(|c| {
            let (rt_lo, rt_hi) = c.rt_window;
            let (im_lo, im_hi) = c.im_window;
            let (tof_lo, tof_hi) = c.tof_window;

            attach_raw_points_for_spec_1d_in_ctx(
                ctx,
                scale,
                tof_lo,
                tof_hi,
                im_lo,
                im_hi,
                rt_lo,
                rt_hi,
                max_points,
            )
        })
        .collect()
}