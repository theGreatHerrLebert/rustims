use serde::{Deserialize, Serialize};
use mscore::timstof::frame::TimsFrame;
use crate::cluster::peak::{ImPeak1D, RtFrames, RtPeak1D};
use crate::cluster::utility::{bin_range_for_win, build_im_marginal, build_mz_hist, build_rt_marginal, fit1d_moment};

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::data::dia::TimsDatasetDIA;
use crate::data::handle::TimsData;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawPoints {
    pub mz: Vec<f32>,
    pub rt: Vec<f32>,
    pub im: Vec<f32>,
    pub scan: Vec<u32>,
    pub intensity: Vec<f32>,
    pub tof: Vec<i32>,
    pub frame: Vec<u32>,
}

#[derive(Copy, Clone, Debug)]
struct ScanSlice { _scan: usize, start: usize, end: usize }

fn build_scan_slices(fr: &TimsFrame) -> Vec<ScanSlice> {
    let scv = &fr.scan;
    let mut out = Vec::new();
    if scv.is_empty() { return out; }
    let mut s_cur = scv[0];
    let mut i_start = 0usize;
    for i in 1..scv.len() {
        if scv[i] != s_cur {
            if s_cur >= 0 {
                out.push(ScanSlice { _scan: s_cur as usize, start: i_start, end: i });
            }
            s_cur = scv[i];
            i_start = i;
        }
    }
    if s_cur >= 0 {
        out.push(ScanSlice { _scan: s_cur as usize, start: i_start, end: scv.len() });
    }
    out
}

#[inline] fn thin_stride(total: usize, cap: usize) -> usize {
    if cap == 0 || total <= cap { 1 } else { (total + cap - 1) / cap }
}

#[derive(Clone, Debug)]
pub struct ClusterSpec1D {
    // local (RT) indices inside the provided RtFrames slice (inclusive)
    pub rt_lo: usize,
    pub rt_hi: usize,
    // absolute scan bounds (inclusive) in the frame (TIMS axis)
    pub im_lo: usize,
    pub im_hi: usize,
    // physical m/z window in Da (inclusive-ish; we clamp to scale on use)
    pub mz_win: (f32, f32),
    // histogram resolution for m/z marginal
    pub mz_hist_bins: usize,

    // provenance (optional)
    pub window_group: Option<u32>,
    pub parent_im_id: Option<u64>,
    pub parent_rt_id: Option<u64>,
    pub ms_level: u8,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterResult1D {
    pub rt_window: (usize, usize),
    pub im_window: (usize, usize),
    pub mz_window: (f32, f32),
    pub rt_fit: Fit1D,
    pub im_fit: Fit1D,
    pub mz_fit: Fit1D,
    pub raw_sum: f32,
    pub volume_proxy: f32,
    pub frame_ids_used: Vec<u32>,
    pub window_group: Option<u32>,
    pub parent_im_id: Option<u64>,
    pub parent_rt_id: Option<u64>,
    pub ms_level: u8,
    pub rt_axis_sec: Option<Vec<f32>>,
    pub im_axis_scans: Option<Vec<usize>>,
    pub mz_axis_da: Option<Vec<f32>>,

    // NEW: optional attachment of raw points (kept None by evaluate_spec_1d)
    pub raw_points: Option<RawPoints>,
}

#[derive(Clone, Copy, Debug)]
pub struct BuildSpecOpts {
    pub extra_rt_pad: usize,
    pub extra_im_pad: usize,
    pub mz_ppm_pad: f32,
    pub mz_hist_bins: usize,
    pub ms_level: u8,
}

impl BuildSpecOpts {
    #[inline]
    pub fn with_ms_level(self, level: u8) -> Self {
        let mut o = self;
        o.ms_level = level;
        o
    }
}

// --- new: options, same spirit as before ---
#[derive(Clone, Debug, Default)]
pub struct Attach1DOptions {
    pub attach_points: bool,
    pub attach_axes: bool,         // you already have this
    pub max_points: Option<usize>, // thinning cap
}

// extend your Eval1DOpts (or keep separate and thread it through)
#[derive(Clone, Debug)]
pub struct Eval1DOpts {
    pub refine_mz_once: bool,
    pub refine_k_sigma: f32,
    pub attach_axes: bool,
    pub attach: Attach1DOptions,   // <-- NEW
}

#[inline]
fn ppm_expand((lo, hi):(f32,f32), ppm: f32, center_hint: f32) -> (f32,f32) {
    if ppm <= 0.0 { return (lo, hi); }
    let d = center_hint.abs() * ppm * 1e-6;
    (lo - d, hi + d)
}

pub fn make_spec_from_pair(
    im: &ImPeak1D,
    rt: &RtPeak1D,
    rt_frames: &RtFrames,          // to clamp RT to local domain
    opts: &BuildSpecOpts,
) -> ClusterSpec1D {
    let frames_len = rt_frames.frames.len();
    let (mut rt_lo, mut rt_hi) = rt.rt_bounds_frames;
    rt_lo = rt_lo.saturating_sub(opts.extra_rt_pad).min(frames_len.saturating_sub(1));
    rt_hi = rt_hi.saturating_add(opts.extra_rt_pad).min(frames_len.saturating_sub(1));
    if rt_lo > rt_hi { std::mem::swap(&mut rt_lo, &mut rt_hi); }

    let mut im_lo = im.left.saturating_sub(opts.extra_im_pad);
    let mut im_hi = im.right.saturating_add(opts.extra_im_pad);
    if im_lo > im_hi { std::mem::swap(&mut im_lo, &mut im_hi); }

    let mz0 = im.mz_bounds; // already physical Da
    let mz_win = ppm_expand(mz0, opts.mz_ppm_pad, im.mz_center);

    ClusterSpec1D {
        rt_lo, rt_hi, im_lo, im_hi,
        mz_win,
        mz_hist_bins: opts.mz_hist_bins.max(16),
        window_group: im.window_group,
        parent_im_id: Some(im.id),
        parent_rt_id: Some(rt.id),
        ms_level: opts.ms_level,
    }
}

pub fn evaluate_spec_1d(
    rt_frames: &RtFrames,        // has frames, frame_ids, rt_times, scale
    spec: &ClusterSpec1D,
    opts: &Eval1DOpts,
) -> ClusterResult1D {
    debug_assert!(rt_frames.is_consistent());
    let scale = &*rt_frames.scale;
    let (bin_lo0, bin_hi0) = bin_range_for_win(scale, spec.mz_win);

    // 1) first pass
    let rt_marg1 = build_rt_marginal(
        &rt_frames.frames, spec.rt_lo, spec.rt_hi, bin_lo0, bin_hi0, spec.im_lo, spec.im_hi);
    let im_marg1 = build_im_marginal(
        &rt_frames.frames, spec.rt_lo, spec.rt_hi, bin_lo0, bin_hi0, spec.im_lo, spec.im_hi);
    let (mz_hist1, mz_centers1) = build_mz_hist(
        &rt_frames.frames, spec.rt_lo, spec.rt_hi, bin_lo0, bin_hi0, spec.im_lo, spec.im_hi, scale);

    let rt_sum1: f32 = rt_marg1.iter().copied().sum();
    let im_sum1: f32 = im_marg1.iter().copied().sum();
    let mz_sum1: f32 = mz_hist1.iter().copied().sum();

    // cheap early exit
    if rt_sum1 <= 0.0 || im_sum1 <= 0.0 || mz_sum1 <= 0.0 {
        return ClusterResult1D {
            rt_window: (spec.rt_lo, spec.rt_hi),
            im_window: (spec.im_lo, spec.im_hi),
            mz_window: spec.mz_win,
            rt_fit: Fit1D::default(),
            im_fit: Fit1D::default(),
            mz_fit: Fit1D::default(),
            raw_sum: 0.0,
            volume_proxy: 0.0,
            frame_ids_used: rt_frames.frame_ids[spec.rt_lo..=spec.rt_hi].to_vec(),
            window_group: spec.window_group,
            parent_im_id: spec.parent_im_id,
            parent_rt_id: spec.parent_rt_id,
            ms_level: spec.ms_level,
            rt_axis_sec: opts.attach_axes.then_some(rt_frames.rt_times[spec.rt_lo..=spec.rt_hi].to_vec()),
            im_axis_scans: opts.attach_axes.then_some((spec.im_lo..=spec.im_hi).collect()),
            mz_axis_da:   opts.attach_axes.then_some(mz_centers1.clone()),
            raw_points: None,
        };
    }

    let mz_fit1 = fit1d_moment(&mz_hist1, Some(&mz_centers1));
    let rt_times: Vec<f32> = rt_frames.rt_times[spec.rt_lo..=spec.rt_hi].to_vec();
    let im_axis: Vec<usize> = (spec.im_lo..=spec.im_hi).collect();

    // 2) optional m/z refine μ±kσ once
    let (bin_lo, bin_hi, mz_centers2) = if opts.refine_mz_once && mz_fit1.sigma>0.0 && mz_fit1.area>0.0 {
        let k = opts.refine_k_sigma.max(1.0);
        let lo_da = (mz_fit1.mu - k*mz_fit1.sigma).max(scale.mz_min);
        let hi_da = (mz_fit1.mu + k*mz_fit1.sigma).min(scale.mz_max);
        let (bl, bh) = bin_range_for_win(scale, (lo_da, hi_da));
        (bl, bh, (bl..=bh).map(|i| scale.center(i)).collect::<Vec<_>>())
    } else {
        (bin_lo0, bin_hi0, mz_centers1.clone())
    };

    // 3) re-accumulate with final m/z window
    let rt_marg = build_rt_marginal(&rt_frames.frames, spec.rt_lo, spec.rt_hi, bin_lo, bin_hi, spec.im_lo, spec.im_hi);
    let im_marg = build_im_marginal(&rt_frames.frames, spec.rt_lo, spec.rt_hi, bin_lo, bin_hi, spec.im_lo, spec.im_hi);
    let (mz_hist, _mz_centers_final) = build_mz_hist(&rt_frames.frames, spec.rt_lo, spec.rt_hi, bin_lo, bin_hi, spec.im_lo, spec.im_hi, scale);

    // 4) final fits
    let rt_fit = fit1d_moment(&rt_marg, Some(&rt_times));
    let im_axis_f32: Vec<f32> = im_axis.iter().map(|&s| s as f32).collect();
    let im_fit = fit1d_moment(&im_marg, Some(&im_axis_f32));
    let mz_fit = fit1d_moment(&mz_hist, Some(&mz_centers2));

    // 5) pack result
    let raw_sum = rt_marg.iter().copied().sum();
    let volume_proxy = rt_fit.area * im_fit.area * mz_fit.area.max(0.0);

    ClusterResult1D {
        rt_window: (spec.rt_lo, spec.rt_hi),
        im_window: (spec.im_lo, spec.im_hi),
        mz_window: (scale.edges[bin_lo], scale.edges[bin_hi+1].min(scale.mz_max)),
        rt_fit, im_fit, mz_fit,
        raw_sum, volume_proxy,
        frame_ids_used: rt_frames.frame_ids[spec.rt_lo..=spec.rt_hi].to_vec(),
        window_group: spec.window_group,
        parent_im_id: spec.parent_im_id,
        parent_rt_id: spec.parent_rt_id,
        ms_level: spec.ms_level,
        rt_axis_sec: opts.attach_axes.then_some(rt_times),
        im_axis_scans: opts.attach_axes.then_some(im_axis),
        mz_axis_da:   opts.attach_axes.then_some(mz_centers2),
        raw_points: None,
    }
}

#[inline]
fn rt_overlap((a0,a1):(usize,usize),(b0,b1):(usize,usize)) -> usize {
    if a0 > a1 || b0 > b1 { return 0; }
    let lo = a0.max(b0);
    let hi = a1.min(b1);
    hi.saturating_sub(lo).saturating_add(1)
}

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

                // collect to a Vec so both branches return IntoIter<_>
                let mut out: Vec<ClusterSpec1D> = Vec::new();
                if rts.is_empty() { return out.into_iter(); }

                // conservative IM-RT support from IM peak on this grid
                let im_rt = rt_bounds_from_im(im, n_frames);

                for rt in rts {
                    // clamp RT to local domain
                    let mut rt_lo = rt.rt_bounds_frames.0.min(n_frames.saturating_sub(1));
                    let mut rt_hi = rt.rt_bounds_frames.1.min(n_frames.saturating_sub(1));
                    if rt_lo > rt_hi { std::mem::swap(&mut rt_lo, &mut rt_hi); }

                    if require_rt_overlap && rt_overlap((rt_lo, rt_hi), im_rt) == 0 {
                        continue;
                    }

                    // Build one spec
                    let mut spec = make_spec_from_pair(im, rt, rt_frames, opts);
                    // final safety clamp
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

/// Helper: derive a conservative RT window for an IM peak in local frame indices.
/// Uses the IM peak’s stored fractional `rt_bounds` if available on the same grid;
/// otherwise falls back to the IM-carried `frame_id_bounds` mapped into this RtFrames slice.
///
/// NOTE: If your `ImPeak1D` already stores `rt_bounds` in the SAME local grid,
///       you can simplify and just return `im.rt_bounds`.
fn rt_bounds_from_im(im: &ImPeak1D, n_frames: usize) -> (usize, usize) {
    // If your ImPeak1D carries local frame-index bounds, prefer them:
    // (Rename `rt_bounds` below to whatever field name you actually use.)
    let (a,b) = im.rt_bounds; // (usize, usize) on the same grid as RtFrames
    let mut lo = a.min(n_frames.saturating_sub(1));
    let mut hi = b.min(n_frames.saturating_sub(1));
    if lo > hi { std::mem::swap(&mut lo, &mut hi); }
    (lo, hi)
}

/// Optional predicate knobs so we can byte-for-byte mirror `filter_ranged`.
#[derive(Clone, Copy, Debug)]
pub struct MirrorFilter {
    pub use_inv_mob: bool,            // set true if you want to constrain inv. mobility
    pub inv_mob_min: f64,
    pub inv_mob_max: f64,
    pub intensity_min: f64,           // usually 0.0
    pub intensity_max: f64,           // usually f64::INFINITY
}

impl Default for MirrorFilter {
    fn default() -> Self {
        Self {
            use_inv_mob: false,
            inv_mob_min: 0.0,
            inv_mob_max: f64::INFINITY,
            intensity_min: 0.0,
            intensity_max: f64::INFINITY,
        }
    }
}

#[inline]
fn mirror_pass_count(
    mz: &[f64], it: &[f64], scan: &[i32], mob: &[f64], tofs: &[i32],
    start: usize, end: usize,
    mz_lo: f64, mz_hi: f64,
    scan_lo: i32, scan_hi: i32,
    mf: MirrorFilter,
) -> usize {
    let end = end.min(mz.len().min(it.len()).min(scan.len()).min(mob.len()).min(tofs.len()));
    let mut n = 0usize;
    if mf.use_inv_mob {
        for i in start..end {
            // Inclusive bounds everywhere — same as filter_ranged
            if mz[i] >= mz_lo && mz[i] <= mz_hi
                && scan[i] >= scan_lo && scan[i] <= scan_hi
                && mob[i]  >= mf.inv_mob_min && mob[i]  <= mf.inv_mob_max
                && it[i]   >= mf.intensity_min && it[i] <= mf.intensity_max
            {
                n += 1;
            }
        }
    } else {
        for i in start..end {
            if mz[i] >= mz_lo && mz[i] <= mz_hi
                && scan[i] >= scan_lo && scan[i] <= scan_hi
                && it[i]   >= mf.intensity_min && it[i] <= mf.intensity_max
            {
                n += 1;
            }
        }
    }
    n
}

#[inline]
fn mirror_pass_push(
    mz: &[f64], it: &[f64], scan: &[i32], mob: &[f64], tofs: &[i32],
    start: usize, end: usize,
    mz_lo: f64, mz_hi: f64,
    scan_lo: i32, scan_hi: i32,
    mf: MirrorFilter,
    rt_val: f32, frame_id: u32,
    seen: &mut usize, stride: usize, out: &mut RawPoints,
) {
    let end = end.min(mz.len().min(it.len()).min(scan.len()).min(mob.len()).min(tofs.len()));
    if mf.use_inv_mob {
        for i in start..end {
            if mz[i] >= mz_lo && mz[i] <= mz_hi
                && scan[i] >= scan_lo && scan[i] <= scan_hi
                && mob[i]  >= mf.inv_mob_min && mob[i]  <= mf.inv_mob_max
                && it[i]   >= mf.intensity_min && it[i] <= mf.intensity_max
            {
                if stride == 1 || (*seen % stride == 0) {
                    out.mz.push(mz[i] as f32);
                    out.rt.push(rt_val);
                    out.im.push(mob[i] as f32);
                    out.scan.push(scan[i] as u32);
                    out.intensity.push(it[i] as f32);
                    out.frame.push(frame_id);
                    out.tof.push(tofs[i]);
                }
                *seen += 1;
            }
        }
    } else {
        for i in start..end {
            if mz[i] >= mz_lo && mz[i] <= mz_hi
                && scan[i] >= scan_lo && scan[i] <= scan_hi
                && it[i]   >= mf.intensity_min && it[i] <= mf.intensity_max
            {
                if stride == 1 || (*seen % stride == 0) {
                    out.mz.push(mz[i] as f32);
                    out.rt.push(rt_val);
                    out.im.push(mob[i] as f32);
                    out.scan.push(scan[i] as u32);
                    out.intensity.push(it[i] as f32);
                    out.frame.push(frame_id);
                    out.tof.push(tofs[i]);
                }
                *seen += 1;
            }
        }
    }
}

/// EXACT mirror of `filter_ranged` semantics: linear scan within each per-scan block.
/// No binary search; all comparisons inclusive (>=, <=).
pub fn attach_raw_points_for_spec_1d(
    ds: &TimsDatasetDIA,
    rt_frames: &RtFrames,
    final_bin_lo: usize, final_bin_hi: usize,     // map to Da via scale.edges
    final_im_lo: usize, final_im_hi: usize,       // inclusive scan bounds
    final_rt_lo: usize, final_rt_hi: usize,       // inclusive RT indices into rt_frames
    max_points: Option<usize>,
) -> RawPoints {
    let scale = rt_frames.scale.as_ref();
    // Use *exactly* the Da interval you previously passed to Python.
    // Both ends are treated inclusive (same as handle.filter_ranged).
    let mz_lo = scale.edges[final_bin_lo] as f64;
    let mz_hi = scale.edges[final_bin_hi + 1].min(scale.mz_max) as f64;

    let scan_lo = final_im_lo as i32;
    let scan_hi = final_im_hi as i32;

    // If you want inv.mob/intensity constraints, tweak here:
    let mf = MirrorFilter::default(); // identical to your handle unless you enable fields

    let frame_ids_local = rt_frames.frame_ids[final_rt_lo..=final_rt_hi].to_vec();
    let slice = ds.get_slice(frame_ids_local.clone(), 1);

    // Pre-compute contiguous [start,end) ranges per scan value for speed
    let scan_slices: Vec<Vec<ScanSlice>> =
        slice.frames.iter().map(|fr| build_scan_slices(fr)).collect();

    // --- counting pass ---
    let mut total = 0usize;
    for (fi, fr) in slice.frames.iter().enumerate() {
        let mz   = &fr.ims_frame.mz;
        let it   = &fr.ims_frame.intensity;
        let sc   = &fr.scan;
        let mob  = &fr.ims_frame.mobility;
        let tofs = &fr.tof;

        for sl in &scan_slices[fi] {
            // Only loop the contiguous block for the scan; still check inclusive bounds
            total += mirror_pass_count(
                mz, it, sc, mob, tofs,
                sl.start, sl.end,
                mz_lo, mz_hi, scan_lo, scan_hi, mf
            );
        }
    }
    let stride = max_points.map(|cap| thin_stride(total, cap)).unwrap_or(1);

    let mut out = RawPoints {
        mz: Vec::with_capacity(total/stride + 8),
        rt: Vec::with_capacity(total/stride + 8),
        im: Vec::with_capacity(total/stride + 8),
        scan: Vec::with_capacity(total/stride + 8),
        intensity: Vec::with_capacity(total/stride + 8),
        tof: Vec::with_capacity(total/stride + 8),
        frame: Vec::with_capacity(total/stride + 8),
    };
    if total == 0 { return out; }

    let rt_axis_sec = rt_frames.rt_times[final_rt_lo..=final_rt_hi].to_vec();

    // --- extraction pass ---
    let mut seen = 0usize;
    for (fi, fr) in slice.frames.iter().enumerate() {
        let mz   = &fr.ims_frame.mz;
        let it   = &fr.ims_frame.intensity;
        let sc   = &fr.scan;
        let mob  = &fr.ims_frame.mobility;
        let tofs = &fr.tof;

        let rt_val   = rt_axis_sec[fi];
        let frame_id = frame_ids_local[fi];

        for sl in &scan_slices[fi] {
            mirror_pass_push(
                mz, it, sc, mob, tofs,
                sl.start, sl.end,
                mz_lo, mz_hi, scan_lo, scan_hi, mf,
                rt_val, frame_id, &mut seen, stride, &mut out
            );
        }
    }
    out
}

pub fn attach_raw_points_for_spec_1d_threads(
    ds: &TimsDatasetDIA,
    rt_frames: &RtFrames,
    final_bin_lo: usize, final_bin_hi: usize,
    final_im_lo: usize, final_im_hi: usize,
    final_rt_lo: usize, final_rt_hi: usize,
    max_points: Option<usize>,
    num_threads: usize,
) -> RawPoints {
    let scale = rt_frames.scale.as_ref();
    let mz_lo = scale.edges[final_bin_lo] as f64;
    let mz_hi = scale.edges[final_bin_hi + 1].min(scale.mz_max) as f64;
    let scan_lo = final_im_lo as i32;
    let scan_hi = final_im_hi as i32;
    let mf = MirrorFilter::default();

    let frame_ids_local = rt_frames.frame_ids[final_rt_lo..=final_rt_hi].to_vec();
    let slice = ds.get_slice(frame_ids_local.clone(), num_threads.max(1));

    let scan_slices: Vec<Vec<ScanSlice>> =
        slice.frames.iter().map(|fr| build_scan_slices(fr)).collect();

    // Count
    let mut total = 0usize;
    for (fi, fr) in slice.frames.iter().enumerate() {
        let mz   = &fr.ims_frame.mz;
        let it   = &fr.ims_frame.intensity;
        let sc   = &fr.scan;
        let mob  = &fr.ims_frame.mobility;
        let tofs = &fr.tof;

        for sl in &scan_slices[fi] {
            total += mirror_pass_count(
                mz, it, sc, mob, tofs,
                sl.start, sl.end,
                mz_lo, mz_hi, scan_lo, scan_hi, mf
            );
        }
    }
    let stride = max_points.map(|cap| thin_stride(total, cap)).unwrap_or(1);

    let mut out = RawPoints {
        mz: Vec::with_capacity(total/stride + 8),
        rt: Vec::with_capacity(total/stride + 8),
        im: Vec::with_capacity(total/stride + 8),
        scan: Vec::with_capacity(total/stride + 8),
        intensity: Vec::with_capacity(total/stride + 8),
        tof: Vec::with_capacity(total/stride + 8),
        frame: Vec::with_capacity(total/stride + 8),
    };
    if total == 0 { return out; }

    let rt_axis_sec = rt_frames.rt_times[final_rt_lo..=final_rt_hi].to_vec();

    // Extract
    let mut seen = 0usize;
    for (fi, fr) in slice.frames.iter().enumerate() {
        let mz   = &fr.ims_frame.mz;
        let it   = &fr.ims_frame.intensity;
        let sc   = &fr.scan;
        let mob  = &fr.ims_frame.mobility;
        let tofs = &fr.tof;

        let rt_val   = rt_axis_sec[fi];
        let frame_id = frame_ids_local[fi];

        for sl in &scan_slices[fi] {
            mirror_pass_push(
                mz, it, sc, mob, tofs,
                sl.start, sl.end,
                mz_lo, mz_hi, scan_lo, scan_hi, mf,
                rt_val, frame_id, &mut seen, stride, &mut out
            );
        }
    }
    out
}