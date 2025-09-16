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

/// Product of 1D areas as a separable-volume proxy.
fn separable_volume(rt: &ClusterFit1D, im: &ClusterFit1D, mz: &ClusterFit1D) -> f32 {
    // NOTE: areas here already include height * σ * √(2π) (constant baseline excluded)
    rt.area * im.area * mz.area.max(0.0)
}

/// Evaluate clusters (RT×IM with m/z marginal) but **preload all required frames once**
/// using the dataset handle. No per-cluster I/O.
pub fn evaluate_clusters_3d(
    ds: &TimsDatasetDIA,
    rt_index: &RtIndex,
    specs: &[ClusterSpec],
    opts: EvalOptions,
    num_threads: usize,          // will be clamped to 1 if your backend needs it
) -> Vec<ClusterResult> {
    // ---------------------------
    // 0) Figure out which frames we actually need (union of all RT windows)
    // ---------------------------
    let cols = rt_index.cols;
    if cols == 0 || specs.is_empty() {
        return Vec::new();
    }

    // mark used frames
    let mut used = vec![false; cols];
    for s in specs {
        let l = s.rt_left.min(cols - 1);
        let r = s.rt_right.min(cols - 1);
        if l <= r {
            for i in l..=r { used[i] = true; }
        }
    }

    // compress to a single ordered list of frame IDs (RT order)
    let mut used_indices = Vec::with_capacity(cols);
    let mut used_frame_ids = Vec::with_capacity(cols);
    for (i, &u) in used.iter().enumerate() {
        if u {
            used_indices.push(i);
            used_frame_ids.push(rt_index.frames[i]);
        }
    }
    if used_frame_ids.is_empty() {
        // nothing to do
        return specs.iter().enumerate().map(|(cid, spec)| ClusterResult {
            rt_window: (spec.rt_left, spec.rt_right),
            im_window: (spec.im_left, spec.im_right),
            mz_window_da: (0.0, 0.0),
            rt_fit: ClusterFit1D::default(),
            im_fit: ClusterFit1D::default(),
            mz_fit: ClusterFit1D::default(),
            raw_sum: 0.0,
            fit_volume: 0.0,
            rt_peak_id: cid,
            im_peak_id: cid,
            mz_center_hint: spec.mz_center_hint,
            frame_ids_used: Vec::new(),
            frames_axis: None, scans_axis: None, mz_axis: None,
            patch_2d_colmajor: None, patch_shape: (0, 0),
        }).collect();
    }

    // ---------------------------
    // 1) Materialize **once** from disk
    // ---------------------------
    let slice = ds.get_slice(used_frame_ids.clone(), /*num_threads=*/num_threads);

    // Map: global rt_col -> local index in `slice.frames`
    // Since we preserved RT order, local position is the order in used_indices.
    // Build a quick lookup vector of length `cols` with Option<local_pos>.
    let mut glob2loc = vec![None::<usize>; cols];
    for (loc, &glob) in used_indices.iter().enumerate() {
        glob2loc[glob] = Some(loc);
    }

    // For convenience
    let frames_total_local = slice.frames.len(); // == used_frame_ids.len()
    debug_assert_eq!(frames_total_local, used_frame_ids.len());

    // ---------------------------
    // 2) Per-cluster extraction from the preloaded frames (no I/O)
    // ---------------------------
    specs.par_iter().enumerate().map(|(cid, spec)| {
        // RT bounds → local slice bounds
        let l_glob = spec.rt_left.min(cols - 1);
        let r_glob = spec.rt_right.min(cols - 1);
        if l_glob > r_glob {
            return empty_result(cid, spec, (0.0, 0.0));
        }

        let l_loc = match glob2loc[l_glob] { Some(x) => x, None => return empty_result(cid, spec, (0.0,0.0)) };
        let r_loc = match glob2loc[r_glob] { Some(x) => x, None => return empty_result(cid, spec, (0.0,0.0)) };
        if l_loc > r_loc || r_loc >= frames_total_local {
            return empty_result(cid, spec, (0.0, 0.0));
        }

        let frames = r_loc - l_loc + 1;

        // IM bounds (absolute scans); clamp to what exists across this local range
        // Discover max scan in the local range
        let mut scan_max_abs: usize = 0;
        for fr in &slice.frames[l_loc..=r_loc] {
            if let Some(&mx) = fr.scan.iter().max() {
                if mx > scan_max_abs as i32 { scan_max_abs = mx as usize; }
            }
        }
        let im_l = spec.im_left.min(scan_max_abs);
        let im_r = spec.im_right.min(scan_max_abs);
        if im_l > im_r || frames == 0 {
            return empty_result_with_axes(cid, spec, (0.0, 0.0),
                                          used_frame_ids[l_loc..=r_loc].to_vec(),
                                          None, None, (frames, 0), &opts);
        }
        let scans = im_r - im_l + 1;

        // m/z window
        let (mz_min, mz_max) = if let Some(win) = spec.mz_window_da_override {
            win
        } else {
            let d = ppm_to_delta_da(spec.mz_center_hint, spec.mz_ppm_window);
            (spec.mz_center_hint - d, spec.mz_center_hint + d)
        };
        if !mz_min.is_finite() || !mz_max.is_finite() || mz_max <= mz_min {
            return empty_result_with_axes(cid, spec, (mz_min, mz_max),
                                          used_frame_ids[l_loc..=r_loc].to_vec(),
                                          Some((im_l, im_r)), None, (frames, scans), &opts);
        }

        // allocate outputs
        let mut patch = vec![0.0f32; frames * scans]; // [s * frames + f]
        let bins = spec.mz_hist_bins.max(10);
        let bin_w = (mz_max - mz_min) / bins as f32;
        let mut mz_hist = vec![0.0f32; bins];
        let mut mz_centers = Vec::with_capacity(bins);
        for b in 0..bins {
            mz_centers.push(mz_min + (b as f32 + 0.5) * bin_w);
        }

        // accumulate from preloaded frames
        for (fi, fr) in slice.frames[l_loc..=r_loc].iter().enumerate() {
            let mz  = &fr.ims_frame.mz;
            let it  = &fr.ims_frame.intensity;
            let sc  = &fr.scan;
            debug_assert_eq!(mz.len(), it.len());
            debug_assert_eq!(mz.len(), sc.len());

            for k in 0..mz.len() {
                let da = mz[k] as f32;
                if da < mz_min || da > mz_max { continue; }

                let s_abs = sc[k];
                if s_abs < 0 { continue; }
                let s_abs = s_abs as usize;
                if s_abs < im_l || s_abs > im_r { continue; }

                let val = it[k] as f32;
                let s_local = s_abs - im_l;
                patch[s_local * frames + fi] += val;

                // mz hist
                let mut b = ((da - mz_min) / (mz_max - mz_min) * bins as f32).floor() as isize;
                if b < 0 { b = 0; }
                if b as usize >= bins { b = bins as isize - 1; }
                mz_hist[b as usize] += val;
            }
        }

        // optional refine pass
        let mz_fit0 = moment_fit_1d(&mz_hist, Some(&mz_centers));
        let want_refine = opts.refine_mz_once && mz_fit0.sigma > 0.0 && mz_fit0.area > 0.0;

        let (patch2, frames2, scans2, mz_hist2, mz_centers2, mz_win2) = if want_refine {
            let k = opts.refine_k_sigma.max(1.0);
            let lo = (mz_fit0.mu - k * mz_fit0.sigma).max(mz_min);
            let hi = (mz_fit0.mu + k * mz_fit0.sigma).min(mz_max);
            if hi <= lo {
                (patch, frames, scans, mz_hist, mz_centers, (mz_min, mz_max))
            } else {
                // re-accumulate within refined m/z window (cheap pass over same frames)
                let mut patch_r = vec![0.0f32; frames * scans];
                let mut mz_hist_r = vec![0.0f32; bins];
                for (fi, fr) in slice.frames[l_loc..=r_loc].iter().enumerate() {
                    let mz  = &fr.ims_frame.mz;
                    let it  = &fr.ims_frame.intensity;
                    let sc  = &fr.scan;
                    for k in 0..mz.len() {
                        let da = mz[k] as f32;
                        if da < lo || da > hi { continue; }
                        let s_abs = sc[k];
                        if s_abs < 0 { continue; }
                        let s_abs = s_abs as usize;
                        if s_abs < im_l || s_abs > im_r { continue; }
                        let val = it[k] as f32;
                        let s_local = s_abs - im_l;
                        patch_r[s_local * frames + fi] += val;

                        let mut b = ((da - lo) / (hi - lo) * bins as f32).floor() as isize;
                        if b < 0 { b = 0; }
                        if b as usize >= bins { b = bins as isize - 1; }
                        mz_hist_r[b as usize] += val;
                    }
                }
                let mut mz_centers_r = Vec::with_capacity(bins);
                let bw = (hi - lo) / bins as f32;
                for b in 0..bins { mz_centers_r.push(lo + (b as f32 + 0.5) * bw); }
                (patch_r, frames, scans, mz_hist_r, mz_centers_r, (lo, hi))
            }
        } else {
            (patch, frames, scans, mz_hist, mz_centers, (mz_min, mz_max))
        };

        if frames2 == 0 || scans2 == 0 {
            return empty_result_with_axes(
                cid, spec, mz_win2,
                used_frame_ids[l_loc..=r_loc].to_vec(),
                Some((im_l, im_r)), Some(mz_centers2.clone()),
                (frames2, scans2), &opts
            );
        }

        // marginals
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
        let mz_fit = moment_fit_1d(&mz_hist2, Some(&mz_centers2));
        let raw_sum: f32 = patch2.iter().copied().sum();
        let fit_volume = separable_volume(&rt_fit, &im_fit, &mz_fit);

        ClusterResult {
            rt_window: (spec.rt_left, spec.rt_right),
            im_window: (spec.im_left, spec.im_right),
            mz_window_da: mz_win2,
            rt_fit, im_fit, mz_fit,
            raw_sum, fit_volume,
            rt_peak_id: cid,
            im_peak_id: cid,
            mz_center_hint: spec.mz_center_hint,
            frame_ids_used: used_frame_ids[l_loc..=r_loc].to_vec(),
            frames_axis: if opts.attach.attach_frames { Some(used_frame_ids[l_loc..=r_loc].to_vec()) } else { None },
            scans_axis:  if opts.attach.attach_scans  { Some((im_l..=im_r).collect()) } else { None },
            mz_axis:     if opts.attach.attach_mz_axis{ Some(mz_centers2.clone()) } else { None },
            patch_2d_colmajor: if opts.attach.attach_patch_2d { Some(patch2) } else { None },
            patch_shape: (frames2, scans2),
        }
    }).collect()
}

#[inline]
fn empty_result(cid: usize, spec: &ClusterSpec, mz_win: (f32,f32)) -> ClusterResult {
    ClusterResult {
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
        patch_2d_colmajor: None, patch_shape: (0, 0),
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
    shape: (usize,usize),
    opts: &EvalOptions,
) -> ClusterResult {
    ClusterResult {
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
        patch_2d_colmajor: None,
        patch_shape: shape,
    }
}

#[derive(Clone, Debug)]
pub struct Feature {
    pub rt_mu: f32, pub rt_sigma: f32,
    pub im_mu: f32, pub im_sigma: f32,
    pub mz_mono: f32, pub z: u8,
    pub iso_i: [f32;8], pub avg_score: f32, pub z_conf: f32,
    pub raw_sum: f32, pub fit_volume: f32,
    pub source_cluster_id: usize,
}