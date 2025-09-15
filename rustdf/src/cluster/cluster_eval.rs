use rayon::prelude::*;
use rayon::iter::{ ParallelIterator};

use crate::cluster::utility::RtIndex;
use crate::data::dia::TimsDatasetDIA;
use crate::data::handle::TimsData;

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

#[derive(Clone, Debug, Default)]
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
    /// Also attach the 2D dense patch (frames×scans, column-major).
    pub attach_patch_2d: bool,
}

#[derive(Clone, Debug)]
pub struct ClusterResult {
    // windows (indices, inclusive)
    pub rt_window: (usize, usize),
    pub im_window: (usize, usize),
    pub mz_window_da: (f32, f32),

    // 1D fits
    pub rt_fit: ClusterFit1D,  // μ in frame index (convert to time outside if needed)
    pub im_fit: ClusterFit1D,  // μ in scan index
    pub mz_fit: ClusterFit1D,  // μ in Da

    // intensities
    pub raw_sum: f32,          // ∑ over the 2D patch (already m/z-reduced)
    pub fit_volume: f32,       // separable volume proxy (rt*im*mz) if you want (here: product of 1D areas)

    // provenance
    pub rt_peak_id: usize,
    pub im_peak_id: usize,
    pub mz_center_hint: f32,
    pub frame_ids_used: Vec<u32>, // always useful

    // optional attachments
    pub frames_axis: Option<Vec<u32>>,   // frame IDs (length = frames)
    pub scans_axis: Option<Vec<usize>>,  // scan indices (length = scans)
    pub mz_axis: Option<Vec<f32>>,       // m/z centers of histogram bins (length = mz_hist_bins)
    pub patch_2d_colmajor: Option<Vec<f32>>, // (frames×scans) column-major
    pub patch_shape: (usize, usize),     // (frames, scans)
}

#[inline]
fn smooth3_inplace(y: &mut [f32]) {
    if y.len() < 3 { return; }
    let mut prev = y[0];
    for i in 1..y.len()-1 {
        let cur = y[i];
        let nxt = y[i+1];
        y[i] = (prev + cur + nxt) / 3.0;
        prev = cur;
    }
}

fn ppm_to_delta_da(mz: f32, ppm: f32) -> f32 {
    mz * ppm * 1e-6
}

fn moment_fit_1d(y: &[f32], x: Option<&[f32]>) -> ClusterFit1D {
    let n = y.len();
    if n == 0 {
        return ClusterFit1D { mu: 0.0, sigma: 0.0, height: 0.0, baseline: 0.0, area: 0.0, r2: f32::NAN, n: 0 };
    }

    // robust baseline (10th pct)
    let mut ys: Vec<f32> = y.to_vec();
    ys.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let b0 = ys[((0.10 * (n as f32)).floor() as usize).min(n-1)];

    // determine effective bin width for integral (assume uniform if x!=None)
    let dx: f32 = if let Some(xx) = x {
        if xx.len() >= 2 {
            ((xx[xx.len()-1] - xx[0]) / (xx.len().saturating_sub(1) as f32)).abs()
        } else { 1.0 }
    } else { 1.0 };

    // positive part moments + integral
    let mut wsum = 0.0f64;
    let mut xsum = 0.0f64;
    let mut x2sum = 0.0f64;
    let mut peak = 0.0f32;
    let mut pos_integral = 0.0f64;

    for i in 0..n {
        let xi = if let Some(xx) = x { xx[i] as f64 } else { i as f64 };
        let yi = (y[i] - b0).max(0.0) as f64;
        if yi > 0.0 {
            wsum += yi;
            xsum += yi * xi;
            x2sum += yi * xi * xi;
            pos_integral += yi;
        }
        if y[i] > peak { peak = y[i]; }
    }

    if wsum <= 0.0 {
        // flat after baseline → area from raw integral (no baseline), as last resort
        let raw_sum: f32 = y.iter().copied().sum::<f32>() * dx;
        return ClusterFit1D { mu: 0.0, sigma: 0.0, height: 0.0, baseline: b0, area: raw_sum, r2: f32::NAN, n };
    }

    let mu = (xsum / wsum) as f32;
    let var = (x2sum / wsum - (mu as f64)*(mu as f64)).max(0.0) as f32;
    let sigma = var.sqrt();

    // gaussian proxy
    let height = (peak - b0).max(0.0);
    let gauss_area = height * sigma * (std::f32::consts::TAU).sqrt() * 0.5_f32.sqrt();

    // integral fallback (above baseline), scaled by bin width
    let integ_area = (pos_integral as f32) * dx;

    // use the more conservative non-zero estimate
    let area = if gauss_area > 0.0 { gauss_area } else { integ_area };

    ClusterFit1D { mu, sigma, height, baseline: b0, area, r2: f32::NAN, n }
}

#[derive(Clone, Debug)]
pub struct EvalOptions {
    pub attach: AttachOptions,
    pub refine_mz_once: bool,   // do a second pass with μ±kσ
    pub refine_k_sigma: f32,    // 3.0 by default
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            attach: AttachOptions {
                attach_frames: true,
                attach_scans: true,
                attach_mz_axis: true,
                attach_patch_2d: false,
            },
            refine_mz_once: false,
            refine_k_sigma: 3.0,
        }
    }
}

fn extract_patch_and_mz_hist(
    ds: &TimsDatasetDIA,
    rt_index: &RtIndex,
    spec: &ClusterSpec,
) -> (Vec<f32>, usize, usize, Vec<u32>, Vec<usize>, Vec<f32>, Vec<f32>, (f32,f32))
{
    // --- RT window ---
    let cols_total = rt_index.cols;
    if cols_total == 0 {
        return (Vec::new(), 0, 0, Vec::new(), Vec::new(), Vec::new(), Vec::new(), (0.0, 0.0));
    }
    let rt_l0 = spec.rt_left.saturating_sub(spec.extra_rt_pad);
    let rt_r0 = spec.rt_right.saturating_add(spec.extra_rt_pad);
    let rt_l  = rt_l0.min(cols_total - 1);
    let rt_r  = rt_r0.min(cols_total - 1);
    if rt_l > rt_r {
        return (Vec::new(), 0, 0, Vec::new(), Vec::new(), Vec::new(), Vec::new(), (0.0, 0.0));
    }

    let frame_ids: Vec<u32> = rt_index.frames[rt_l..=rt_r].to_vec();
    let frames = frame_ids.len();
    if frames == 0 {
        return (Vec::new(), 0, 0, Vec::new(), Vec::new(), Vec::new(), Vec::new(), (0.0, 0.0));
    }

    // Materialize
    let slice = ds.get_slice(frame_ids.clone(), /*num_threads=*/1);

    // --- IM window (absolute scans) ---
    let global_scan_max = slice.frames.iter()
        .flat_map(|fr| fr.scan.iter().copied())
        .max()
        .unwrap_or(0)
        .max(0) as usize;

    let im_l0 = spec.im_left.saturating_sub(spec.extra_im_pad);
    let im_r0 = spec.im_right.saturating_add(spec.extra_im_pad);
    let im_l  = im_l0.min(global_scan_max);
    let im_r  = im_r0.min(global_scan_max);

    if im_l > im_r {
        // No scans in range → empty, but keep frames for provenance.
        return (Vec::new(), frames, 0, frame_ids, Vec::new(), Vec::new(), Vec::new(), (0.0, 0.0));
    }

    let scans = im_r - im_l + 1;
    if scans == 0 {
        return (Vec::new(), frames, 0, frame_ids, Vec::new(), Vec::new(), Vec::new(), (0.0, 0.0));
    }

    // --- m/z window (Da) ---
    let (mz_min, mz_max) = if let Some(win) = spec.mz_window_da_override {
        win
    } else {
        let mz0 = spec.mz_center_hint;
        let d   = ppm_to_delta_da(mz0, spec.mz_ppm_window);
        (mz0 - d, mz0 + d)
    };
    if !mz_min.is_finite() || !mz_max.is_finite() || mz_max <= mz_min {
        return (Vec::new(), frames, scans, frame_ids, Vec::new(), Vec::new(), Vec::new(), (0.0, 0.0));
    }

    // --- allocate outputs ---
    let mut f_patch = vec![0.0f32; frames * scans]; // [s * frames + f]
    let bins = spec.mz_hist_bins.max(10);
    let bin_w = (mz_max - mz_min) / (bins as f32);
    let mut mz_hist   = vec![0.0f32; bins];

    let mut mz_center = Vec::with_capacity(bins);
    for b in 0..bins {
        mz_center.push(mz_min + (b as f32 + 0.5) * bin_w);
    }

    // --- accumulate ---
    for (fi, fr) in slice.frames.into_iter().enumerate() {
        let mz  = &fr.ims_frame.mz;
        let intensity = &fr.ims_frame.intensity;
        let scv = &fr.scan;

        // Guard: equal length assumptions
        debug_assert_eq!(mz.len(), intensity.len());
        debug_assert_eq!(mz.len(), scv.len());

        for k in 0..mz.len() {
            let da = mz[k] as f32;
            if da < mz_min || da > mz_max { continue; }

            let sc_i = scv[k];
            if sc_i < 0 { continue; }
            let sc = sc_i as usize;
            if sc < im_l || sc > im_r { continue; }

            let val = intensity[k] as f32;

            // 2D patch
            let s_local = sc - im_l;
            // (frames > 0, scans > 0 by construction)
            f_patch[s_local * frames + fi] += val;

            // m/z histogram
            let mut b = ((da - mz_min) / (mz_max - mz_min) * (bins as f32)).floor() as isize;
            if b < 0 { b = 0; }
            if (b as usize) >= bins { b = (bins as isize) - 1; }
            mz_hist[b as usize] += val;
        }
    }

    // scans axis
    let scans_axis: Vec<usize> = (im_l..=im_r).collect();

    smooth3_inplace(&mut mz_hist);

    (f_patch, frames, scans, frame_ids, scans_axis, mz_hist, mz_center, (mz_min, mz_max))
}

/// Product of 1D areas as a separable-volume proxy.
fn separable_volume(rt: &ClusterFit1D, im: &ClusterFit1D, mz: &ClusterFit1D) -> f32 {
    // NOTE: areas here already include height * σ * √(2π) (constant baseline excluded)
    rt.area * im.area * mz.area.max(0.0)
}

pub fn evaluate_clusters_3d(
    ds: &TimsDatasetDIA,
    rt_index: &RtIndex,
    specs: &[ClusterSpec],
    opts: EvalOptions,
) -> Vec<ClusterResult> {
    specs.par_iter().enumerate().map(|(cid, spec)| {
        let (patch, frames, scans, frame_ids, scans_axis, mz_hist_y, mz_centers, mz_da_win) =
            extract_patch_and_mz_hist(ds, rt_index, spec);

        // Early-out: empty extraction → empty result (but keep windows + provenance)
        if frames == 0 || scans == 0 {
            return ClusterResult {
                rt_window: (spec.rt_left, spec.rt_right),
                im_window: (spec.im_left, spec.im_right),
                mz_window_da: mz_da_win,
                rt_fit: ClusterFit1D::default(),
                im_fit: ClusterFit1D::default(),
                mz_fit: ClusterFit1D::default(),
                raw_sum: 0.0,
                fit_volume: 0.0,
                rt_peak_id: cid,
                im_peak_id: cid,
                mz_center_hint: spec.mz_center_hint,
                frame_ids_used: frame_ids.clone(),
                frames_axis: if opts.attach.attach_frames { Some(frame_ids) } else { None },
                scans_axis: if opts.attach.attach_scans { Some(scans_axis) } else { None },
                mz_axis: if opts.attach.attach_mz_axis { Some(mz_centers.clone()) } else { None },
                patch_2d_colmajor: if opts.attach.attach_patch_2d { Some(patch) } else { None },
                patch_shape: (frames, scans),
            };
        }

        // First-pass m/z fit on smoothed histogram
        let mz_fit0 = moment_fit_1d(&mz_hist_y, Some(&mz_centers));

        // Heuristics for whether to tighten μ ± kσ
        let total_mz_mass: f32 = mz_hist_y.iter().copied().sum();
        let mz_area_min: f32 = 0.01 * total_mz_mass;     // e.g., require ≥1% of hist mass
        let raw_min: f32 = 0.0;                          // could use > 0 or a small epsilon

        let want_refine =
            opts.refine_mz_once &&
                mz_fit0.sigma > 0.0 &&
                mz_fit0.area > mz_area_min &&
                patch.iter().any(|&v| v > raw_min);          // some RT×IM signal in the patch

        let (patch2, frames2, scans2, frame_ids2, scans_axis2, mz_hist_y2, mz_centers2, mz_da_win2) =
            if want_refine {
                let k = opts.refine_k_sigma.max(1.0);
                let lo = (mz_fit0.mu - k * mz_fit0.sigma).max(mz_da_win.0);
                let hi = (mz_fit0.mu + k * mz_fit0.sigma).min(mz_da_win.1);
                if hi <= lo {
                    (patch, frames, scans, frame_ids, scans_axis, mz_hist_y, mz_centers, mz_da_win)
                } else {
                    let mut spec2 = spec.clone();
                    spec2.mz_window_da_override = Some((lo, hi));
                    let (p2, fr2, sc2, fids2, scax2, mz_y2, mz_x2, win2) =
                        extract_patch_and_mz_hist(ds, rt_index, &spec2);

                    let mut mz_y2_mut = mz_y2;
                    smooth3_inplace(&mut mz_y2_mut);
                    (p2, fr2, sc2, fids2, scax2, mz_y2_mut, mz_x2, win2)
                }
            } else {
                (patch, frames, scans, frame_ids, scans_axis, mz_hist_y, mz_centers, mz_da_win)
            };

        // Guard again (refined window could be empty)
        if frames2 == 0 || scans2 == 0 {
            return ClusterResult {
                rt_window: (spec.rt_left, spec.rt_right),
                im_window: (spec.im_left, spec.im_right),
                mz_window_da: mz_da_win2,
                rt_fit: ClusterFit1D::default(),
                im_fit: ClusterFit1D::default(),
                mz_fit: ClusterFit1D::default(),
                raw_sum: 0.0,
                fit_volume: 0.0,
                rt_peak_id: cid,
                im_peak_id: cid,
                mz_center_hint: spec.mz_center_hint,
                frame_ids_used: frame_ids2.clone(),
                frames_axis: if opts.attach.attach_frames { Some(frame_ids2) } else { None },
                scans_axis: if opts.attach.attach_scans { Some(scans_axis2) } else { None },
                mz_axis: if opts.attach.attach_mz_axis { Some(mz_centers2.clone()) } else { None },
                patch_2d_colmajor: if opts.attach.attach_patch_2d { Some(patch2) } else { None },
                patch_shape: (frames2, scans2),
            };
        }

        // RT/IM marginals (safe: frames2>0 && scans2>0)
        let mut rt_marg = vec![0.0f32; frames2];
        let mut im_marg = vec![0.0f32; scans2];
        for s in 0..scans2 {
            for f in 0..frames2 {
                let v = patch2[s * frames2 + f];
                rt_marg[f] += v;
                im_marg[s] += v;
            }
        }

        let rt_fit = moment_fit_1d(&rt_marg, None);
        let im_fit = moment_fit_1d(&im_marg, None);
        let mz_fit = moment_fit_1d(&mz_hist_y2, Some(&mz_centers2));

        let raw_sum: f32 = patch2.iter().copied().sum();
        let fit_volume = separable_volume(&rt_fit, &im_fit, &mz_fit);

        ClusterResult {
            rt_window: (spec.rt_left, spec.rt_right),
            im_window: (spec.im_left, spec.im_right),
            mz_window_da: mz_da_win2,
            rt_fit, im_fit, mz_fit,
            raw_sum, fit_volume,
            rt_peak_id: cid,
            im_peak_id: cid,
            mz_center_hint: spec.mz_center_hint,
            frame_ids_used: frame_ids2.clone(),
            frames_axis: if opts.attach.attach_frames { Some(frame_ids2) } else { None },
            scans_axis:  if opts.attach.attach_scans  { Some(scans_axis2) } else { None },
            mz_axis:     if opts.attach.attach_mz_axis{ Some(mz_centers2.clone()) } else { None },
            patch_2d_colmajor: if opts.attach.attach_patch_2d { Some(patch2) } else { None },
            patch_shape: (frames2, scans2),
        }
    }).collect()
}