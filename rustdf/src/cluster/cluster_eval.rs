use mscore::timstof::frame::TimsFrame;
use rayon::prelude::*;
use rayon::iter::ParallelIterator;
use crate::cluster::utility::{bin_range, build_frame_bin_view, FrameBinView, ImPeak1D, MzScale, RtIndex, RtPeak1D};
use crate::data::dia::TimsDatasetDIA;
use crate::data::handle::TimsData;
// cluster_eval.rs (or wherever these live)
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawPoints {
    pub mz: Vec<f32>,
    pub rt: Vec<f32>,       // frame time per point
    pub im: Vec<f32>,       // store scan as f32 (or 1/K0 if you have a converter)
    pub scan: Vec<u32>,     // absolute scan index in frame
    pub intensity: Vec<f32>,
    pub tof: Vec<i32>,      // if available in TimsFrame
    pub frame: Vec<u32>,    // frame id per point
}

#[derive(Clone, Debug)]
pub struct ClusterSpec {
    pub rt_left: usize,
    pub rt_right: usize,
    pub im_left: usize,
    pub im_right: usize,
    pub mz_center_hint: f32,
    pub mz_ppm_window: f32,
    pub extra_rt_pad: usize,
    pub extra_im_pad: usize,
    /// number of bins for m/z histogram fit (used for the m/z marginal)
    pub mz_hist_bins: usize,

    /// OPTIONAL: if present, overrides the ppm window during extraction
    /// (used when you do a second pass with μ ± kσ).
    pub mz_window_da_override: Option<(f32, f32)>,
}

impl Default for ClusterSpec {
    fn default() -> Self {
        Self {
            rt_left: 0, rt_right: 0,
            im_left: 0, im_right: 0,
            mz_center_hint: 0.0,
            mz_ppm_window: 10.0,
            extra_rt_pad: 0, extra_im_pad: 0,
            mz_hist_bins: 64,
            mz_window_da_override: None,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ClusterFit1D {
    pub mu: f32,       // center (index for RT/IM; Da for m/z)
    pub sigma: f32,    // stddev (frames/scans/Da)
    pub height: f32,   // peak amplitude above baseline
    pub baseline: f32, // constant background
    pub area: f32,     // height * sigma * sqrt(2π)
    pub r2: f32,       // (optional) goodness (moment fit sets to NaN)
    pub n: usize,
}

#[derive(Clone, Debug, Default)]
pub struct AttachOptions {
    pub attach_frames: bool,
    pub attach_scans: bool,
    pub attach_mz_axis: bool,
    /// NEW: attach raw points bundle (SoA)
    pub attach_points: bool,
    /// Optional cap on attached points (uniform thinning if exceeded)
    pub max_points: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterResult {
    pub id: usize,
    // windows (indices, inclusive)
    pub rt_window: (usize, usize),
    pub im_window: (usize, usize),
    pub mz_window_da: (f32, f32),

    // 1D fits
    pub rt_fit: ClusterFit1D,  // μ in frame time units (we fit on rtIndex.frame_times)
    pub im_fit: ClusterFit1D,  // μ in scan index
    pub mz_fit: ClusterFit1D,  // μ in Da

    // intensities
    pub raw_sum: f32,          // ∑ intensities over final RT×IM×m/z window
    pub fit_volume: f32,       // separable volume proxy (product of 1D areas)

    // provenance
    pub rt_peak_id: usize,
    pub im_peak_id: usize,
    pub mz_center_hint: f32,
    pub frame_ids_used: Vec<u32>, // always useful

    // optional attachments
    pub frames_axis: Option<Vec<u32>>,   // frame IDs (length = frames)
    pub scans_axis: Option<Vec<usize>>,  // scan indices (length = scans)
    pub mz_axis: Option<Vec<f32>>,       // m/z centers of histogram bins (length = mz_hist_bins)

    // raw point payload (optional)
    pub raw_points: Option<RawPoints>,
    pub ms_level: u8,
    pub window_group: Option<u32>,
    // for MS1 only; union of DIA groups whose isolation bands include the precursor m/z
    pub window_groups_covering_mz: Option<Vec<u32>>,
}

#[derive(Copy, Clone, Debug)]
struct ScanSlice { scan: usize, start: usize, end: usize }

#[inline]
fn build_scan_slices(fr: &TimsFrame) -> Vec<ScanSlice> {
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

#[inline]
fn lower_bound_in(mz: &[f64], start: usize, end: usize, x: f32) -> usize {
    let mut lo = start; let mut hi = end; let xf = x as f64;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if mz[mid] < xf { lo = mid + 1; } else { hi = mid; }
    }
    lo
}

#[inline]
fn upper_bound_in(mz: &[f64], start: usize, end: usize, x: f32) -> usize {
    let mut lo = start; let mut hi = end; let xf = x as f64;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if mz[mid] <= xf { lo = mid + 1; } else { hi = mid; }
    }
    lo
}

#[inline]
fn ppm_to_delta_da(mz: f32, ppm: f32) -> f32 { mz * ppm * 1e-6 }

#[inline]
fn mean_and_var(y: &[f32]) -> (f32, f32) {
    if y.is_empty() { return (0.0, 0.0); }
    let n = y.len() as f32;
    let mean = y.iter().copied().sum::<f32>() / n;
    let var = y.iter().map(|v| {
        let d = *v - mean;
        d*d
    }).sum::<f32>() / n.max(1.0);
    (mean, var)
}

#[inline]
fn refine_height_baseline_ls(y: &[f32], x: Option<&[f32]>, mu: f32, sigma: f32, b0: f32, height0: f32)
                             -> (f32, f32) {
    if y.is_empty() || sigma <= 0.0 { return (height0.max(0.0), b0); }
    let n = y.len();
    let mut s_gg = 0.0f64;
    let mut s_g1 = 0.0f64;
    let s_11 = n as f64;
    let mut s_yg = 0.0f64;
    let mut s_y1 = 0.0f64;

    for i in 0..n {
        let xi = if let Some(xx) = x { xx[i] } else { i as f32 };
        let z = (xi - mu) as f64 / (sigma as f64);
        let g = (-0.5f64 * z * z).exp();
        let yi = y[i] as f64;

        s_gg += g * g;
        s_g1 += g;
        s_yg += yi * g;
        s_y1 += yi;
    }

    let det = s_11 * s_gg - s_g1 * s_g1;
    if det.abs() < 1e-12 {
        return (height0.max(0.0), b0);
    }
    let b = ( s_gg * s_y1 - s_g1 * s_yg) / det;
    let h = (-s_g1 * s_y1 + s_11 * s_yg) / det;

    (h as f32, b as f32)
}

#[inline]
fn thin_stride(total: usize, cap: usize) -> usize {
    if cap == 0 || total <= cap { 1 } else { (total + cap - 1) / cap }
}

fn moment_fit_1d(y: &[f32], x: Option<&[f32]>) -> ClusterFit1D {
    let n = y.len();
    if n == 0 {
        return ClusterFit1D { mu: 0.0, sigma: 0.0, height: 0.0, baseline: 0.0, area: 0.0, r2: f32::NAN, n: 0 };
    }

    // robust baseline (10th percentile of y)
    let mut ys: Vec<f32> = y.to_vec();
    ys.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let b0 = ys[((0.10 * (n as f32)).floor() as usize).min(n-1)];

    // moments over positive part
    let mut wsum = 0.0f64;
    let mut xsum = 0.0f64;
    let mut x2sum = 0.0f64;
    let mut peak = 0.0f32;

    let mut pos_integral_trap = 0.0f64;
    let mut pos_integral_unit = 0.0f64;

    for i in 0..n {
        let xi_f64 = if let Some(xx) = x { xx[i] as f64 } else { i as f64 };
        let yi_pos = (y[i] - b0).max(0.0) as f64;

        if yi_pos > 0.0 {
            wsum += yi_pos;
            xsum += yi_pos * xi_f64;
            x2sum += yi_pos * xi_f64 * xi_f64;
            pos_integral_unit += yi_pos;
        }
        if y[i] > peak { peak = y[i]; }

        if let Some(xx) = x {
            if i > 0 {
                let dx_i = (xx[i] as f64 - xx[i-1] as f64).abs();
                let y_prev = (y[i-1] - b0).max(0.0) as f64;
                pos_integral_trap += 0.5f64 * (yi_pos + y_prev) * dx_i;
            }
        }
    }

    if wsum <= 0.0 {
        let area = if let Some(xx) = x {
            let mut raw = 0.0f64;
            for i in 1..n {
                let dx_i = (xx[i] as f64 - xx[i-1] as f64).abs();
                raw += 0.5f64 * ((y[i] as f64) + (y[i-1] as f64)) * dx_i;
            }
            raw.max(0.0) as f32
        } else {
            y.iter().copied().map(|v| v.max(0.0)).sum::<f32>()
        };

        return ClusterFit1D {
            mu: 0.0, sigma: 0.0, height: 0.0, baseline: b0, area,
            r2: f32::NAN, n
        };
    }

    let mu = (xsum / wsum) as f32;
    let var = (x2sum / wsum - (mu as f64)*(mu as f64)).max(0.0) as f32;
    let sigma = var.sqrt();

    if !sigma.is_finite() || sigma <= 0.0 {
        let integ_area = if x.is_some() {
            pos_integral_trap.max(0.0) as f32
        } else {
            pos_integral_unit as f32
        };
        return ClusterFit1D {
            mu, sigma: 0.0, height: 0.0, baseline: b0,
            area: integ_area, r2: f32::NAN, n
        };
    }

    let height0 = (y.iter().copied().fold(f32::NEG_INFINITY, f32::max) - b0).max(0.0);
    let (height_ref, baseline_ref) = refine_height_baseline_ls(y, x, mu, sigma, b0, height0);
    let height = height_ref.max(0.0);
    let baseline = baseline_ref;

    let gauss_area = height * sigma * (std::f32::consts::TAU).sqrt();
    let integ_area = if x.is_some() {
        pos_integral_trap.max(0.0) as f32
    } else {
        pos_integral_unit as f32
    };
    let area = if x.is_some() { integ_area } else { gauss_area.max(integ_area) };

    let (_m, y_var) = mean_and_var(y);
    let mut ss_res = 0.0f64;
    let ss_tot = (y_var as f64) * (n as f64);
    for i in 0..n {
        let xi = if let Some(xx) = x { xx[i] } else { i as f32 };
        let z = (xi - mu) / sigma;
        let y_hat = baseline + height * (-0.5f32 * z * z).exp();
        let e = (y[i] - y_hat) as f64;
        ss_res += e * e;
    }
    let r2 = if sigma > 0.0 && ss_tot > 0.0 { (1.0 - (ss_res / ss_tot)) as f32 } else { f32::NAN };

    ClusterFit1D { mu, sigma, height, baseline, area, r2, n }
}

#[derive(Clone, Debug)]
pub struct EvalOptions {
    pub attach: AttachOptions,
    pub refine_mz_once: bool,
    pub refine_k_sigma: f32,
    /// If Some(k): envelope(requested, μ±kσ) for IM
    pub im_k_sigma: Option<f32>,
    /// Enforce a minimum width in scans for the final IM window (default 1)
    pub im_min_width: usize,

    /// NEW: hard caps (inclusive length) on requested windows
    /// If None, no cap applied.
    pub max_rt_span_frames: Option<usize>,  // e.g., Some(200)
    pub max_im_span_scans: Option<usize>,   // e.g., Some(100)
    /// How to center caps before the first pass (we don't know μ yet).
    pub cap_anchor: CapAnchor,              // see enum below
    pub ms_level: u8,          // if Some, stamp onto results
    pub window_group_hint: Option<u32>,     // if Some, stamp onto results
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            attach: AttachOptions {
                attach_frames: true,
                attach_scans: true,
                attach_mz_axis: true,
                attach_points: false,
                max_points: None,
            },
            refine_mz_once: false,
            refine_k_sigma: 3.0,
            im_k_sigma: None,
            im_min_width: 1,

            max_rt_span_frames: None,
            max_im_span_scans: None,
            cap_anchor: CapAnchor::RequestedMid,
            ms_level: 0,
            window_group_hint: None,
        }
    }
}

/// Where to center the pre-fit cap when we don’t have μ yet.
#[derive(Clone, Copy, Debug)]
pub enum CapAnchor {
    /// Center cap on (l+r)/2 (robust and cheap).
    RequestedMid,
}

/// Product of 1D areas as a separable-volume proxy.
fn separable_volume(rt: &ClusterFit1D, im: &ClusterFit1D, mz: &ClusterFit1D) -> f32 {
    rt.area * im.area * mz.area.max(0.0)
}

/// Collect raw points (SoA) within final windows; uniform thinning if `max_points` set.
fn collect_raw_points_for_cluster(
    slice_frames: &[TimsFrame],
    scan_slices_per_frame: &[Vec<ScanSlice>],
    l_loc: usize, r_loc: usize,              // local frame bounds
    im_l: usize, im_r: usize,                // scan bounds (absolute per frame)
    mz_lo: f32, mz_hi: f32,                  // m/z window in Da
    rt_times_local: &[f32],                  // rt_index.frame_times mapped to l_loc..=r_loc
    frame_ids_local: &[u32],                 // rt_index.frames mapped to l_loc..=r_loc
    max_points: Option<usize>,               // optional cap (thin uniformly)
) -> RawPoints {
    // Count first for capacity + thinning stride
    let mut total = 0usize;
    for (fi, slices) in scan_slices_per_frame[l_loc..=r_loc].iter().enumerate() {
        let fr = &slice_frames[l_loc + fi];
        let mz = &fr.ims_frame.mz;
        for sl in slices {
            if sl.scan < im_l || sl.scan > im_r { continue; }
            let l = lower_bound_in(mz, sl.start, sl.end, mz_lo);
            let r = upper_bound_in(mz, sl.start, sl.end, mz_hi);
            total += r.saturating_sub(l);
        }
    }
    let stride = max_points.map(|cap| thin_stride(total, cap)).unwrap_or(1);

    let mut pts = RawPoints {
        mz: Vec::with_capacity(total / stride + 8),
        rt: Vec::with_capacity(total / stride + 8),
        im: Vec::with_capacity(total / stride + 8),
        scan: Vec::with_capacity(total / stride + 8),
        intensity: Vec::with_capacity(total / stride + 8),
        tof: Vec::with_capacity(total / stride + 8),
        frame: Vec::with_capacity(total / stride + 8),
    };

    if total == 0 { return pts; }

    let mut seen = 0usize;
    for (fi, slices) in scan_slices_per_frame[l_loc..=r_loc].iter().enumerate() {
        let fr = &slice_frames[l_loc + fi];

        let mz   = &fr.ims_frame.mz;         // Vec<f64>
        let it   = &fr.ims_frame.intensity;  // Vec<f32 or f64 depending on your loader
        let ims  = &fr.ims_frame.mobility;   // Vec<f64> (per-point mobility / 1/K0)
        let tofs = &fr.tof;                  // Vec<i32>  (per-point TOF)
        // (scv is not needed here for values; we use it only for scan-slice bounds)

        // Make sure all arrays are aligned; clamp to the shortest length to be safe.
        let len_all = mz.len().min(it.len()).min(ims.len()).min(tofs.len());

        let rt_val = rt_times_local[fi];
        let frame_id = frame_ids_local[fi];

        for sl in slices {
            let s_abs = sl.scan;
            if s_abs < im_l || s_abs > im_r { continue; }

            // Binary search within this scan-run, then clamp r to aligned length.
            let l = lower_bound_in(mz, sl.start, sl.end, mz_lo);
            let mut r = upper_bound_in(mz, sl.start, sl.end, mz_hi);
            if r > len_all { r = len_all; }
            if l >= r { continue; }

            let mut i = l;
            while i < r {
                if stride == 1 || (seen % stride == 0) {
                    pts.mz.push(mz[i] as f32);
                    pts.rt.push(rt_val);
                    pts.im.push(ims[i] as f32);          // use mobility from the frame
                    pts.scan.push(s_abs as u32);         // keep scan index as separate axis
                    // If your intensity is f64, cast; if it's f32, this is a no-op.
                    pts.intensity.push(it[i] as f32);
                    pts.frame.push(frame_id);
                    pts.tof.push(tofs[i]);               // per-point TOF
                }
                seen += 1;
                i += 1;
            }
        }
    }

    pts
}

#[inline]
fn final_im_bounds_from_fit(
    requested: (usize, usize),  // (im_l0, im_r0) after padding
    scan_domain: (usize, usize),// (0, max_scan_abs)
    im_fit: &ClusterFit1D,      // fit from first-pass marginal in requested window
    k_opt: Option<f32>,         // opts.im_k_sigma
    min_width: usize,           // opts.im_min_width
) -> (usize, usize) {
    let (l_req, r_req) = requested;
    let (scan_l, scan_r) = scan_domain;

    let (mut l, mut r) = if let Some(k) = k_opt {
        if im_fit.sigma.is_finite() && im_fit.sigma > 0.0 {
            let lo = (im_fit.mu - k.max(0.5) * im_fit.sigma).floor() as isize;
            let hi = (im_fit.mu + k.max(0.5) * im_fit.sigma).ceil()  as isize;
            let lo = lo.clamp(scan_l as isize, scan_r as isize) as usize;
            let hi = hi.clamp(scan_l as isize, scan_r as isize) as usize;

            // Envelope of requested and μ±kσ (so we never shrink the user's request)
            (lo.min(l_req), hi.max(r_req))
        } else {
            (l_req, r_req)
        }
    } else {
        (l_req, r_req)
    };

    if r < l { std::mem::swap(&mut l, &mut r); }

    // Enforce minimum width
    let width = r + 1 - l;
    if width < min_width {
        let need = min_width - width;
        let grow_l = need / 2;
        let grow_r = need - grow_l;
        l = l.saturating_sub(grow_l);
        r = (r + grow_r).min(scan_r);
        if r + 1 - l < min_width {
            // if still too small because the row is short, expand to the row
            l = scan_l;
            r = scan_r;
        }
    }

    (l, r)
}

#[inline]
fn cap_window_symmetric(
    domain: (usize, usize),   // e.g., (0, cols-1)
    mut l: usize,
    mut r: usize,
    cap_len_inclusive: usize, // inclusive length (e.g., 200 means ≤200 frames)
    anchor: usize,            // center around this index as best as possible
) -> (usize, usize) {
    if l > r { std::mem::swap(&mut l, &mut r); }
    let want = cap_len_inclusive.max(1);
    let len  = r + 1 - l;
    if len <= want { return (l, r); }

    // desired half widths
    let half_left  = want / 2;
    let half_right = want - half_left;

    // place around anchor, but stay inside [l,r] and domain
    let dom_l = domain.0;
    let dom_r = domain.1;

    let mut new_l = anchor.saturating_sub(half_left);
    let mut new_r = anchor.saturating_add(half_right - 1);

    // clamp to initial request
    new_l = new_l.clamp(l, r);
    new_r = new_r.clamp(l, r);

    // ensure inclusive length and clamp to domain
    let mut need = want.saturating_sub(new_r + 1 - new_l);
    while need > 0 && new_l > l { new_l -= 1; need -= 1; }
    while need > 0 && new_r < r { new_r += 1; need -= 1; }

    // still ensure domain
    new_l = new_l.clamp(dom_l, dom_r);
    new_r = new_r.clamp(dom_l, dom_r);
    if new_r < new_l { new_r = new_l; }
    (new_l, new_r)
}

/// Evaluate clusters (RT×IM with m/z marginal) with a single frame preload.
/// Returns ClusterResult objects; optional raw points attached, no dense 2D patch.
pub fn evaluate_clusters_3d(
    ds: &TimsDatasetDIA,
    rt_index: &RtIndex,
    specs: &[ClusterSpec],
    opts: EvalOptions,
    num_threads: usize,
) -> Vec<ClusterResult> {
    let cols = rt_index.cols;
    if cols == 0 || specs.is_empty() { return Vec::new(); }

    // mark used RT columns
    let mut used = vec![false; cols];
    for s in specs {
        let l = s.rt_left.min(cols - 1);
        let r = s.rt_right.min(cols - 1);
        if l <= r { for i in l..=r { used[i] = true; } }
    }

    // RT-ordered frame ids & index mapping
    let mut used_indices = Vec::with_capacity(cols);
    let mut used_frame_ids = Vec::with_capacity(cols);
    for (i, &u) in used.iter().enumerate() {
        if u { used_indices.push(i); used_frame_ids.push(rt_index.frames[i]); }
    }
    if used_frame_ids.is_empty() {
        return specs.iter().enumerate().map(|(cid, spec)| empty_result(cid, spec, (0.0, 0.0))).collect();
    }

    // materialize once
    let slice = ds.get_slice(used_frame_ids.clone(), num_threads);

    // precompute scan slices per frame
    let scan_slices_per_frame: Vec<Vec<ScanSlice>> =
        slice.frames.iter().map(|fr| build_scan_slices(fr)).collect();

    // global->local RT col
    let mut glob2loc = vec![None::<usize>; cols];
    for (loc, &glob) in used_indices.iter().enumerate() { glob2loc[glob] = Some(loc); }

    let frames_total_local = slice.frames.len();

    specs.par_iter().enumerate().map(|(cid, spec)| {
        // RT bounds (local)
        let l_glob = spec.rt_left.min(cols - 1);
        let r_glob = spec.rt_right.min(cols - 1);
        if l_glob > r_glob { return empty_result(cid, spec, (0.0, 0.0)); }

        let l_loc0 = match glob2loc[l_glob] { Some(x) => x, None => return empty_result(cid, spec, (0.0, 0.0)) };
        let r_loc0 = match glob2loc[r_glob] { Some(x) => x, None => return empty_result(cid, spec, (0.0, 0.0)) };
        if l_loc0 > r_loc0 || r_loc0 >= frames_total_local {
            return empty_result(cid, spec, (0.0, 0.0));
        }

        // (optional) cap RT span
        let (l_loc, r_loc) = if let Some(max_rt) = opts.max_rt_span_frames {
            let anchor = match opts.cap_anchor {
                CapAnchor::RequestedMid => (l_loc0 + r_loc0) / 2,
            };
            cap_window_symmetric(
                (0, frames_total_local.saturating_sub(1)),
                l_loc0, r_loc0,
                max_rt,
                anchor,
            )
        } else {
            (l_loc0, r_loc0)
        };

        let frames = r_loc - l_loc + 1;

        // --- IM bounds with padding (requested window) ---
        let mut scan_max_abs: usize = 0;
        for fr in &slice.frames[l_loc..=r_loc] {
            if let Some(&mx) = fr.scan.iter().max() {
                if mx > scan_max_abs as i32 { scan_max_abs = mx as usize; }
            }
        }

        let im_l_req0 = spec.im_left.saturating_sub(spec.extra_im_pad).min(scan_max_abs);
        let im_r_req0 = spec.im_right.saturating_add(spec.extra_im_pad).min(scan_max_abs);

        let (im_l_req, im_r_req) = if let Some(max_im) = opts.max_im_span_scans {
            // center on requested midpoint before we know μ
            let anchor = (im_l_req0 + im_r_req0) / 2;
            cap_window_symmetric((0, scan_max_abs), im_l_req0, im_r_req0, max_im, anchor)
        } else {
            (im_l_req0, im_r_req0)
        };

        if im_l_req > im_r_req || frames == 0 {
            return empty_result_with_axes(
                cid, spec, (0.0, 0.0),
                used_frame_ids[l_loc..=r_loc].to_vec(),
                None, None, &opts
            );
        }

        if im_l_req > im_r_req || frames == 0 {
            return empty_result_with_axes(
                cid, spec, (0.0, 0.0),
                used_frame_ids[l_loc..=r_loc].to_vec(),
                None, None, &opts
            );
        }
        let scans0 = im_r_req - im_l_req + 1;

        // --- m/z window ---
        let (mz_min, mz_max) = if let Some(w) = spec.mz_window_da_override {
            w
        } else {
            let d = ppm_to_delta_da(spec.mz_center_hint, spec.mz_ppm_window);
            (spec.mz_center_hint - d, spec.mz_center_hint + d)
        };
        if !mz_min.is_finite() || !mz_max.is_finite() || mz_max <= mz_min {
            return empty_result_with_axes(
                cid, spec, (mz_min, mz_max),
                used_frame_ids[l_loc..=r_loc].to_vec(),
                Some((im_l_req, im_r_req)), None, &opts
            );
        }

        // --- First pass accumulation: RT/IM marginals + m/z histogram within requested IM window ---
        let bins = spec.mz_hist_bins.max(10);
        let bin_w = (mz_max - mz_min) / bins as f32;
        let inv_win = 1.0f32 / (mz_max - mz_min);
        let mut rt_marg = vec![0.0f32; frames];
        let mut im_marg = vec![0.0f32; scans0];
        let mut mz_hist = vec![0.0f32; bins];

        const MAX_POINTS_PER_SLICE: usize = 10_000;

        for (fi, (fr, slices)) in slice.frames[l_loc..=r_loc]
            .iter()
            .zip(&scan_slices_per_frame[l_loc..=r_loc])
            .enumerate()
        {
            let mz  = &fr.ims_frame.mz;
            let it  = &fr.ims_frame.intensity;

            for sl in slices {
                let s_abs = sl.scan;
                if s_abs < im_l_req || s_abs > im_r_req { continue; }

                let l = lower_bound_in(mz, sl.start, sl.end, mz_min);
                let r = upper_bound_in(mz, sl.start, sl.end, mz_max);
                if l >= r { continue; }

                let len = r - l;
                let stride = if len > MAX_POINTS_PER_SLICE {
                    ((len + MAX_POINTS_PER_SLICE - 1) / MAX_POINTS_PER_SLICE).max(1)
                } else { 1 };
                let weight = if stride > 1 { stride as f32 } else { 1.0 };

                let s_local = s_abs - im_l_req;

                let mut i = l;
                while i < r {
                    let val = (it[i] as f32) * weight;

                    // RT/IM marginals
                    rt_marg[fi] += val;
                    im_marg[s_local] += val;

                    // m/z hist
                    let pos = ((mz[i] as f32 - mz_min) * inv_win * bins as f32).floor();
                    let b = if pos < 0.0 { 0 } else {
                        let pi = pos as usize;
                        if pi >= bins { bins - 1 } else { pi }
                    };
                    mz_hist[b] += val;

                    i += stride;
                }
            }
        }

        // --- Optional m/z refine (μ±kσ) and re-accumulate in restricted m/z; still within requested IM window ---
        let mz_centers = (0..bins).map(|b| mz_min + (b as f32 + 0.5) * bin_w).collect::<Vec<_>>();
        let mz_fit0 = moment_fit_1d(&mz_hist, Some(&mz_centers));
        let want_mz_refine = opts.refine_mz_once && mz_fit0.sigma > 0.0 && mz_fit0.area > 0.0;

        let (_rt_marg2, im_marg2, mz_hist2, mz_centers2, mz_win2) = if want_mz_refine {
            let k = opts.refine_k_sigma.max(1.0);
            let lo = (mz_fit0.mu - k * mz_fit0.sigma).max(mz_min);
            let hi = (mz_fit0.mu + k * mz_fit0.sigma).min(mz_max);
            if hi <= lo {
                (rt_marg, im_marg, mz_hist, mz_centers, (mz_min, mz_max))
            } else {
                let mut rt_acc = vec![0.0f32; frames];
                let mut im_acc = vec![0.0f32; scans0];
                let mut mz_hist_r = vec![0.0f32; bins];

                let inv_win_r = 1.0f32 / (hi - lo);
                let bw = (hi - lo) / bins as f32;
                let mz_centers_r = (0..bins).map(|b| lo + (b as f32 + 0.5) * bw).collect::<Vec<_>>();

                for (fi, (fr, slices)) in slice.frames[l_loc..=r_loc]
                    .iter()
                    .zip(&scan_slices_per_frame[l_loc..=r_loc])
                    .enumerate()
                {
                    let mz  = &fr.ims_frame.mz;
                    let it  = &fr.ims_frame.intensity;

                    for sl in slices {
                        let s_abs = sl.scan;
                        if s_abs < im_l_req || s_abs > im_r_req { continue; }

                        let l = lower_bound_in(mz, sl.start, sl.end, lo);
                        let r = upper_bound_in(mz, sl.start, sl.end, hi);
                        if l >= r { continue; }

                        let len = r - l;
                        let stride = if len > MAX_POINTS_PER_SLICE {
                            ((len + MAX_POINTS_PER_SLICE - 1) / MAX_POINTS_PER_SLICE).max(1)
                        } else { 1 };
                        let weight = if stride > 1 { stride as f32 } else { 1.0 };

                        let s_local = s_abs - im_l_req;

                        let mut i = l;
                        while i < r {
                            let val = (it[i] as f32) * weight;

                            rt_acc[fi] += val;
                            im_acc[s_local] += val;

                            let pos = ((mz[i] as f32 - lo) * inv_win_r * bins as f32).floor();
                            let b = if pos < 0.0 { 0 } else {
                                let pi = pos as usize;
                                if pi >= bins { bins - 1 } else { pi }
                            };
                            mz_hist_r[b] += val;

                            i += stride;
                        }
                    }
                }

                (rt_acc, im_acc, mz_hist_r, mz_centers_r, (lo, hi))
            }
        } else {
            (rt_marg, im_marg, mz_hist, mz_centers, (mz_min, mz_max))
        };

        // --- First-pass IM fit on requested window (after m/z refine stage) ---
        let rt_times: Vec<f32> = used_indices[l_loc..=r_loc]
            .iter()
            .map(|&glob_idx| rt_index.frame_times[glob_idx])
            .collect();
        let im_axis0: Vec<f32> = (im_l_req..=im_r_req).map(|s| s as f32).collect();
        let im_fit0 = moment_fit_1d(&im_marg2, Some(&im_axis0));

        // --- Unified final IM window selection (envelope of requested and μ±kσ) ---
        let (im_l, im_r) = final_im_bounds_from_fit(
            (im_l_req, im_r_req),
            (0usize, scan_max_abs),
            &im_fit0,
            opts.im_k_sigma,
            opts.im_min_width,
        );

        let (im_l, im_r) = if let Some(max_im) = opts.max_im_span_scans {
            // Now we *do* have an estimate of μ; center the cap near μ if finite.
            let anchor = if im_fit0.mu.is_finite() {
                im_fit0.mu.clamp(0.0, scan_max_abs as f32) as usize
            } else {
                (im_l + im_r) / 2
            };
            cap_window_symmetric((0, scan_max_abs), im_l, im_r, max_im, anchor)
        } else {
            (im_l, im_r)
        };

        let scans_final = im_r - im_l + 1;

        // --- Re-accumulate RT/IM marginals inside the FINAL IM window (single path) ---
        let mut rt_marg3 = vec![0.0f32; frames];
        let mut im_marg3 = vec![0.0f32; scans_final];

        for (fi, (fr, slices)) in slice.frames[l_loc..=r_loc]
            .iter()
            .zip(&scan_slices_per_frame[l_loc..=r_loc])
            .enumerate()
        {
            let mz  = &fr.ims_frame.mz;
            let it  = &fr.ims_frame.intensity;

            for sl in slices {
                let s_abs = sl.scan;
                if s_abs < im_l || s_abs > im_r { continue; }

                let l = lower_bound_in(mz, sl.start, sl.end, mz_win2.0);
                let r = upper_bound_in(mz, sl.start, sl.end, mz_win2.1);
                if l >= r { continue; }

                let len = r - l;
                let stride = if len > MAX_POINTS_PER_SLICE {
                    ((len + MAX_POINTS_PER_SLICE - 1) / MAX_POINTS_PER_SLICE).max(1)
                } else { 1 };
                let weight = if stride > 1 { stride as f32 } else { 1.0 };

                let s_local = s_abs - im_l;

                let mut i = l;
                while i < r {
                    let val = (it[i] as f32) * weight;
                    rt_marg3[fi] += val;
                    im_marg3[s_local] += val;
                    i += stride;
                }
            }
        }

        // --- Final fits ---
        let im_axis: Vec<f32> = (im_l..=im_r).map(|s| s as f32).collect();
        let rt_fit = moment_fit_1d(&rt_marg3, Some(&rt_times));
        let im_fit = moment_fit_1d(&im_marg3, Some(&im_axis));
        let mz_fit = moment_fit_1d(&mz_hist2, Some(&mz_centers2));

        // --- Summaries ---
        let raw_sum: f32 = rt_marg3.iter().copied().sum();
        let fit_volume = separable_volume(&rt_fit, &im_fit, &mz_fit);

        // --- Axes & attachments (use FINAL IM window) ---
        let frames_axis_vec = used_frame_ids[l_loc..=r_loc].to_vec();
        let scans_axis_vec: Vec<usize> = (im_l..=im_r).collect();
        let rt_times_local: Vec<f32> = used_indices[l_loc..=r_loc]
            .iter()
            .map(|&glob_idx| rt_index.frame_times[glob_idx])
            .collect();

        let raw_points = if opts.attach.attach_points {
            Some(collect_raw_points_for_cluster(
                &slice.frames,
                &scan_slices_per_frame,
                l_loc, r_loc,
                im_l, im_r,
                mz_win2.0, mz_win2.1,
                &rt_times_local,
                &frames_axis_vec,
                opts.attach.max_points,
            ))
        } else { None };

        ClusterResult {
            id: cid,
            rt_window: (spec.rt_left, spec.rt_right),
            im_window: (im_l, im_r), // final window
            mz_window_da: mz_win2,
            rt_fit, im_fit, mz_fit,
            raw_sum, fit_volume,
            rt_peak_id: cid,
            im_peak_id: cid,
            mz_center_hint: spec.mz_center_hint,
            frame_ids_used: frames_axis_vec.clone(),
            frames_axis: if opts.attach.attach_frames { Some(frames_axis_vec) } else { None },
            scans_axis:  if opts.attach.attach_scans  { Some(scans_axis_vec) } else { None },
            mz_axis:     if opts.attach.attach_mz_axis{ Some(mz_centers2.clone()) } else { None },
            raw_points,
            ms_level: opts.ms_level,
            window_group: opts.window_group_hint,
            window_groups_covering_mz: None,
        }
    }).collect()
}

#[inline]
fn empty_result(cid: usize, spec: &ClusterSpec, mz_win: (f32,f32)) -> ClusterResult {
    ClusterResult {
        id: cid,
        rt_window: (spec.rt_left, spec.rt_right),
        im_window: (spec.im_left, spec.im_right),
        mz_window_da: mz_win,
        rt_fit: ClusterFit1D::default(),
        im_fit: ClusterFit1D::default(),
        mz_fit: ClusterFit1D::default(),
        raw_sum: 0.0, fit_volume: 0.0,
        rt_peak_id: cid, im_peak_id: cid,
        mz_center_hint: spec.mz_center_hint,
        frame_ids_used: Vec::new(),
        frames_axis: None, scans_axis: None, mz_axis: None,
        raw_points: None,
        ms_level: 0,
        window_group: None,
        window_groups_covering_mz: None,
    }
}

#[inline]
fn empty_result_with_axes(
    cid: usize,
    spec: &ClusterSpec,
    mz_win: (f32,f32),
    frames_axis: Vec<u32>,
    scans_bounds: Option<(usize,usize)>,
    mz_axis: Option<Vec<f32>>,
    opts: &EvalOptions,
) -> ClusterResult {
    ClusterResult {
        id: cid,
        rt_window: (spec.rt_left, spec.rt_right),
        im_window: (spec.im_left, spec.im_right),
        mz_window_da: mz_win,
        rt_fit: ClusterFit1D::default(),
        im_fit: ClusterFit1D::default(),
        mz_fit: ClusterFit1D::default(),
        raw_sum: 0.0, fit_volume: 0.0,
        rt_peak_id: cid, im_peak_id: cid,
        mz_center_hint: spec.mz_center_hint,
        frame_ids_used: frames_axis.clone(),
        frames_axis: if opts.attach.attach_frames { Some(frames_axis) } else { None },
        scans_axis:  if opts.attach.attach_scans  {
            scans_bounds.map(|(l,r)| (l..=r).collect())
        } else { None },
        mz_axis: if opts.attach.attach_mz_axis { mz_axis } else { None },
        raw_points: None,
        ms_level: opts.ms_level,
        window_group: opts.window_group_hint,
        window_groups_covering_mz: None,
    }
}

#[derive(Clone, Debug)]
pub struct LinkCandidate {

    pub ms1_idx: usize,
    pub ms2_idx: usize,

    pub ms1_id: usize, // same as ms1[ms1_idx].id
    pub ms2_id: usize, // same as ms2[ms2_idx].id

    pub score: f32,
    pub group:   u32,  // window group of the MS2 cluster
}

/// Jaccard overlap on RT indices (inclusive windows)
#[inline]
fn rt_jaccard(a: (usize,usize), b: (usize,usize)) -> f32 {
    let (al,ar) = a; let (bl,br) = b;
    if ar < bl || br < al { return 0.0; }
    let inter = (ar.min(br) + 1).saturating_sub(al.max(bl));
    let union = (ar.max(br) + 1).saturating_sub(al.min(bl));
    (inter as f32) / (union as f32)
}

/// |Δ apex RT| in seconds (needs frame_times)
#[inline]
fn rt_apex_delta_sec(ms1: &ClusterResult, ms2: &ClusterResult) -> f32 {
    let a = ms1.rt_fit.mu; // already in seconds if you fit with frame_times
    let b = ms2.rt_fit.mu;
    (a - b).abs()
}

/// |Δ IM apex| in scans (TIMS co-mobility cue)
#[inline]
fn im_apex_delta_scans(ms1: &ClusterResult, ms2: &ClusterResult) -> f32 {
    (ms1.im_fit.mu - ms2.im_fit.mu).abs()
}

/// Link MS2→MS1 with group compatibility and co-elution.
/// Returns sorted candidates (best score first).
pub fn link_ms2_to_ms1(
    ms1: &[ClusterResult],         // ms_level==1
    ms2: &[ClusterResult],         // ms_level==2, with window_group = Some(g)
    min_rt_jaccard: f32,           // e.g. 0.1–0.2
    max_rt_apex_sec: f32,          // e.g. 5.0–10.0
    max_im_apex_scans: Option<f32>,// e.g. Some(5.0) or None to ignore
) -> Vec<LinkCandidate> {

    use std::collections::{HashMap};

    // index MS1 by group for quick lookup
    let mut group2ms1: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, c) in ms1.iter().enumerate() {
        if let Some(ref gs) = c.window_groups_covering_mz {
            for &g in gs { group2ms1.entry(g).or_default().push(i); }
        }
    }

    // For each MS2 cluster, check compatible MS1 list
    let mut out: Vec<LinkCandidate> = Vec::new();
    for (j, c2) in ms2.iter().enumerate() {
        let g = match c2.window_group { Some(x) => x, None => continue };
        let Some(cands) = group2ms1.get(&g) else { continue };
        for &i in cands {
            let c1 = &ms1[i];

            // RT overlap and apex proximity
            let jacc = rt_jaccard(c1.rt_window, c2.rt_window);
            if jacc < min_rt_jaccard { continue; }
            let d_rt = rt_apex_delta_sec(c1, c2);
            if d_rt > max_rt_apex_sec { continue; }

            // Optional TIMS co-mobility cue
            if let Some(max_d_im) = max_im_apex_scans {
                if im_apex_delta_scans(c1, c2) > max_d_im { continue; }
            }

            // Simple score: Jaccard boosted and penalized by apex deltas
            let score = jacc * 1.0
                * (1.0 / (1.0 + d_rt))                       // smaller better
                * (if let Some(max_d_im) = max_im_apex_scans {
                1.0 / (1.0 + im_apex_delta_scans(c1, c2) / (max_d_im + 1e-3))
            } else { 1.0 });

            out.push(LinkCandidate {
                ms1_idx: i,
                ms2_idx: j,
                ms1_id: c1.id,
                ms2_id: c2.id,
                score,
                group: g
            } );
        }
    }

    out.sort_by(|a,b| b.score.total_cmp(&a.score));
    out
}

#[derive(Clone, Copy)]
struct RtRowSnap { rt_l: usize, rt_r: usize, mz_center: f32 }

#[derive(Clone, Copy)]
struct ImSnap { im_l: usize, im_r: usize }

/// Build one `ClusterSpec` per (RT peak row, IM peak) pair.
/// - `im_scans_abs`: if provided, maps IM **indices** to **absolute** scans.
/// - Pads RT/IM windows and clamps to the available axes.
/// - Produces `mz_hist_bins` bins for the m/z marginal.
pub fn make_cluster_specs_from_peaks_rs(
    rt_peaks: &[RtPeak1D],
    im_rows: &[Vec<ImPeak1D>],       // same row order/length as rt_peaks
    im_scans_abs: Option<&[usize]>,  // absolute scan axis; None => keep indices
    mz_ppm_window: f32,
    extra_rt_pad: usize,
    extra_im_pad: usize,
    mz_hist_bins: usize,
) -> Vec<ClusterSpec> {
    assert_eq!(
        rt_peaks.len(),
        im_rows.len(),
        "im_rows must have same number of rows as rt_peaks"
    );

    // Precompute per-row RT windows and m/z centers
    let rt_rows: Vec<RtRowSnap> = rt_peaks
        .iter()
        .map(|rt| {
            let rt_l = rt.left_padded.saturating_sub(extra_rt_pad);
            let rt_r = rt.right_padded.saturating_add(extra_rt_pad);
            RtRowSnap { rt_l, rt_r, mz_center: rt.mz_center }
        })
        .collect();

    // Build IM windows per row (optionally mapped to absolute scans)
    let im_rows_snap: Vec<Vec<ImSnap>> = im_rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|im| {
                    // starting from IM **indices** stored on the peak
                    let mut il = im.left.saturating_sub(extra_im_pad);
                    let mut ir = im.right.saturating_add(extra_im_pad);

                    if let Some(axis) = im_scans_abs {
                        // clamp to axis, then map to absolute
                        il = il.min(axis.len().saturating_sub(1));
                        ir = ir.min(axis.len().saturating_sub(1));
                        ImSnap { im_l: axis[il], im_r: axis[ir] }
                    } else {
                        ImSnap { im_l: il, im_r: ir }
                    }
                })
                .collect()
        })
        .collect();

    // One spec per (row, im_peak_j)
    rt_rows
        .par_iter()
        .enumerate()
        .flat_map_iter(|(row_idx, rt)| {
            im_rows_snap[row_idx].iter().map(move |im| ClusterSpec {
                rt_left: rt.rt_l,
                rt_right: rt.rt_r,
                im_left: im.im_l,
                im_right: im.im_r,
                mz_center_hint: rt.mz_center,
                mz_ppm_window,
                extra_rt_pad: 0,    // already applied above
                extra_im_pad: 0,    // already applied above
                mz_hist_bins,
                mz_window_da_override: None,
            })
        })
        .collect()
}

fn rt_marginal_conditioned(
    ds: &TimsDatasetDIA,
    rt_index: &RtIndex,
    im: &ImPeak1D,         // requires im.left/right to be *physical* scans
    mz_center: f32,
    mz_ppm_window: f32,
    num_threads: usize,
) -> (Vec<f32>, Vec<u32>) {

    let frames = ds.get_slice(rt_index.frames.clone(), num_threads).frames;
    let cols = frames.len();
    let mut rt_marg = vec![0.0f32; cols];

    // m/z bin range on the SAME scale as rt_index
    let tol = mz_center * mz_ppm_window * 1e-6;
    let (bin_lo, bin_hi) = rt_index.scale.index_range_for_mz_window(mz_center - tol, mz_center + tol);

    // Precompute FrameBinViews once
    let views: Vec<FrameBinView> = (0..cols)
        .into_par_iter()
        .map(|i| build_frame_bin_view(frames[i].clone(), &rt_index.scale, /*global_num_scans*/ {
            // compute global max scan; cheap to recompute here
            frames[i].scan.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0)
        }))
        .collect();

    // Accumulate per frame within m/z band and IM [left,right] (physical)
    let im_l = im.left;
    let im_r = im.right;

    rt_marg
        .par_iter_mut()
        .enumerate()
        .for_each(|(t, acc)| {
            let v = &views[t];
            let (start, end) = bin_range(v, bin_lo, bin_hi);
            let mut sum = 0.0f32;
            for i in start..end {
                let s_phys = v.scan_idx[i] as usize;
                if s_phys >= im_l && s_phys <= im_r {
                    sum += v.intensity[i];
                }
            }
            *acc = sum;
        });

    (rt_marg, rt_index.frames.clone())
}

fn restrict_rt_to_mu_pm_k_sigma(
    rt_index: &RtIndex,
    mu_sec: f32,
    k: f32,
    prior: (usize, usize),
) -> (usize, usize) {
    let (l0, r0) = prior;
    if l0 > r0 || rt_index.cols == 0 { return prior; }

    // map seconds → nearest frame index by lower_bound on frame_times
    let ft = &rt_index.frame_times;
    let to_idx = |t: f32| -> usize {
        // lower_bound
        let mut lo = 0usize;
        let mut hi = ft.len();
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if ft[mid] < t { lo = mid + 1; } else { hi = mid; }
        }
        lo.min(ft.len().saturating_sub(1))
    };

    // estimate σ in seconds: find nearest index to μ, then derive pixel width ~1 frame
    // NOTE: you already have σ in seconds from fit; we receive μ in seconds and
    //       *use k·σ directly in seconds*. If you prefer: pass σ as an argument instead.
    // Here we assume caller chose k based on the fit's σ (seconds).
    let lo_t = (mu_sec - k * 1.0f32 /* σ will be multiplied by caller before pass if needed */).max(ft.first().copied().unwrap_or(mu_sec));
    let hi_t = (mu_sec + k * 1.0f32).min(ft.last().copied().unwrap_or(mu_sec));

    // If you *do* have σ (seconds), call this with k_sigma_sec = k * sigma_sec:
    // let lo_t = (mu_sec - k_sigma_sec).max(ft[0]);
    // let hi_t = (mu_sec + k_sigma_sec).min(*ft.last().unwrap());

    let mut l = to_idx(lo_t);
    let mut r = to_idx(hi_t);
    if l > r { std::mem::swap(&mut l, &mut r); }

    // intersect with prior
    l = l.max(l0);
    r = r.min(r0);
    (l, r)
}

pub fn make_cluster_specs_from_im_peaks_conditioned(
    im_peaks: &[ImPeak1D],
    scale: &MzScale,
    rt_index: &RtIndex,
    k_rt_sigma: Option<f32>,          // e.g., Some(3.0)
    max_rt_span_frames: Option<usize>,// e.g., Some(200)
    mz_ppm_window: f32,
    mz_hist_bins: usize,
    ds: &TimsDatasetDIA,
    num_threads: usize,
) -> Vec<ClusterSpec> {
    im_peaks.iter().map(|im| {
        let mz_c = scale.center(im.mz_row);
        let (rt_l_relaxed, rt_r_relaxed) = (0, rt_index.cols.saturating_sub(1)); // or a smarter prior

        // Build a quick RT marginal *conditioned* on (IM ∧ m/z)
        let (rt_marg, _frame_ids_local) =
            rt_marginal_conditioned(ds, rt_index, im, mz_c, mz_ppm_window, num_threads);

        let rt_fit0 = moment_fit_1d(&rt_marg, Some(&rt_index.frame_times[rt_l_relaxed..=rt_r_relaxed]));
        let (mut rt_l, mut rt_r) = (rt_l_relaxed, rt_r_relaxed);

        if let Some(k) = k_rt_sigma {
            if rt_fit0.sigma.is_finite() && rt_fit0.sigma > 0.0 {
                // turn seconds → nearest frame index
                let mu_t = rt_fit0.mu;
                let mut best = (rt_l, rt_r);
                // find nearest indices around mu ± kσ
                // (helper omitted for brevity)
                best = restrict_rt_to_mu_pm_k_sigma(rt_index, mu_t, k, best);
                (rt_l, rt_r) = best;
            }
        }
        if let Some(cap) = max_rt_span_frames {
            let anchor = (rt_l + rt_r) / 2;
            (rt_l, rt_r) = cap_window_symmetric((0, rt_index.cols.saturating_sub(1)), rt_l, rt_r, cap, anchor);
        }

        ClusterSpec {
            rt_left: rt_l,
            rt_right: rt_r,
            im_left: im.left,
            im_right: im.right,
            mz_center_hint: mz_c,
            mz_ppm_window,
            extra_rt_pad: 0,
            extra_im_pad: 0,
            mz_hist_bins,
            mz_window_da_override: None,
        }
    }).collect()
}