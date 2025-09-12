use mscore::timstof::frame::TimsFrame;
use rustc_hash::{FxHashMap, FxHashSet};
use crate::data::dia::TimsDatasetDIA;
use crate::data::handle::TimsData;
use crate::data::meta::FrameMeta;
use rayon::prelude::*;
use rayon::iter::IntoParallelRefIterator;

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
    frame_times: Option<&[f32]>,
    min_prom: f32,
    min_distance: usize,
    min_width: usize,
    pad_left: usize,          // NEW
    pad_right: usize,         // NEW
) -> Vec<RtPeak1D> {
    let n = y_smoothed.len();
    if n < 3 { return Vec::new(); }

    let mut cands: Vec<usize> = Vec::new();
    for i in 1..n-1 {
        let yi = y_smoothed[i];
        if yi > y_smoothed[i-1] && yi >= y_smoothed[i+1] { cands.push(i); }
    }

    let mut peaks: Vec<RtPeak1D> = Vec::new();
    for &i in &cands {
        let apex = y_smoothed[i];

        let mut l = i; let mut left_min = apex;
        while l > 0 { l -= 1; left_min = left_min.min(y_smoothed[l]); if y_smoothed[l] > apex { break; } }
        let mut r = i; let mut right_min = apex;
        while r+1 < n { r += 1; right_min = right_min.min(y_smoothed[r]); if y_smoothed[r] > apex { break; } }

        let baseline = left_min.max(right_min);
        let prom = apex - baseline;
        if prom < min_prom { continue; }

        let half = baseline + 0.5 * prom;

        let mut wl = i;
        while wl > 0 && y_smoothed[wl] > half { wl -= 1; }
        let left_cross = if wl < i && wl+1 < n {
            let y0 = y_smoothed[wl]; let y1 = y_smoothed[wl+1];
            wl as f32 + if y1 != y0 { (half - y0) / (y1 - y0) } else { 0.0 }
        } else { wl as f32 };

        let mut wr = i;
        while wr+1 < n && y_smoothed[wr] > half { wr += 1; }
        let right_cross = if wr > i && wr < n {
            let y0 = y_smoothed[wr-1]; let y1 = y_smoothed[wr];
            (wr-1) as f32 + if y1 != y0 { (half - y0) / (y1 - y0) } else { 0.0 }
        } else { wr as f32 };

        let width = (right_cross - left_cross).max(0.0);
        let width_frames = width.round() as usize;
        if width_frames < min_width { continue; }

        let sub = if i > 0 && i+1 < n {
            quad_subsample(y_smoothed[i-1], y_smoothed[i], y_smoothed[i+1]).clamp(-0.5, 0.5)
        } else { 0.0 };

        if let Some(last) = peaks.last() {
            if i.abs_diff(last.rt_col) < min_distance {
                if apex <= last.apex_smoothed { continue; }
                peaks.pop();
            }
        }

        // after computing `left_cross`, `right_cross`, `left_idx`, `right_idx`:
        let left_idx  = left_cross.floor().clamp(0.0, (n-1) as f32) as usize;
        let right_idx = right_cross.ceil().clamp(0.0, (n-1) as f32) as usize;

        // exact fractional area on RAW within half-prom window (no padding)
        let area = trapezoid_area_fractional(
            y_raw,
            left_cross.max(0.0),
            right_cross.min((n - 1) as f32),
        );

        // NEW: padded integer window (clamped)
        let left_padded  = left_idx.saturating_sub(pad_left);
        let right_padded = (right_idx + pad_right).min(n - 1);

        // quick integer trapezoid area on RAW for the padded window
        let area_padded = _trapezoid_area(y_raw, left_padded, right_padded);

        // push Peak1D (include both windows + areas)
        peaks.push(RtPeak1D {
            mz_row,
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
        });
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
    pad_right: usize,        // NEW
) -> Vec<RtPeak1D> {
    (0..rows).into_par_iter().flat_map_iter(|r| {
        // gather strided row r
        let mut y_s = Vec::with_capacity(cols);
        let mut y_r = Vec::with_capacity(cols);
        for c in 0..cols {
            y_s.push(data_smoothed[r + c*rows]);
            y_r.push(data_raw     [r + c*rows]);
        }
        find_peaks_row(
            &y_s, &y_r, r, frame_times,
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
fn quantize_mz(mz: f32, resolution: usize) -> u32 {
    let factor = 10f32.powi(resolution as i32);
    (mz * factor + 0.5) as u32
}

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
    // ascending m/z bins (quantized)
    pub bins: Vec<MzBin>,
    // RT-ordered frame ids
    pub frames: Vec<u32>,
    // RT-ordered frame times (seconds or minutes; same unit as FrameMeta.time)
    pub frame_times: Vec<f32>,
    // column-major data (rows = bins, cols = frames), possibly SMOOTHED
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    // if smoothing was applied, keep an unsmoothed copy for quantification
    pub data_raw: Option<Vec<f32>>,
}

type MzBin = u32;

pub fn build_dense_rt_by_mz(
    data_handle: &TimsDatasetDIA,
    maybe_sigma_frames: Option<f32>,
    truncate: f32,
    resolution: usize,
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

    // 2) materialize at chosen resolution
    let frames = data_handle
        .get_slice(frame_ids_sorted.clone(), num_threads)
        .to_resolution(resolution as i32, num_threads)
        .frames;

    let mut frame_by_id: FxHashMap<u32, &TimsFrame> = FxHashMap::default();
    for f in &frames { frame_by_id.insert(f.frame_id as u32, f); }

    // 3) discover bins
    let all_bins: FxHashSet<MzBin> = frames
        .par_iter()
        .map(|frame| {
            let mut local: FxHashSet<MzBin> = FxHashSet::default();
            for &mz in frame.ims_frame.mz.iter() {
                local.insert(quantize_mz(mz as f32, resolution));
            }
            local
        })
        .reduce(FxHashSet::default, |mut a, b| { a.extend(b); a });

    let mut bins_sorted: Vec<MzBin> = all_bins.into_iter().collect();
    bins_sorted.sort_unstable();
    let num_bins = bins_sorted.len();

    let row_index: FxHashMap<MzBin, usize> = bins_sorted
        .iter()
        .enumerate()
        .map(|(r, &bin)| (bin, r))
        .collect();

    // 4) allocate and fill (column-major)
    let mut matrix = vec![0.0f32; num_bins * num_frames];

    let frames_in_col_order: Vec<&TimsFrame> = frame_ids_sorted
        .iter()
        .map(|fid| frame_by_id[fid])
        .collect();

    matrix.par_chunks_mut(num_bins).enumerate().for_each(|(col, col_slice)| {
        let frame = frames_in_col_order[col];
        let mz = &frame.ims_frame.mz;
        let inten = &frame.ims_frame.intensity;

        let mut accum: FxHashMap<usize, f32> = FxHashMap::default();
        for (&m, &i) in mz.iter().zip(inten.iter()) {
            let q = quantize_mz(m as f32, resolution);
            if let Some(&row) = row_index.get(&q) {
                *accum.entry(row).or_insert(0.0) += i as f32;
            }
        }
        for (row, val) in accum { col_slice[row] += val; }
    });

    // 5) optional smoothing; keep raw if applied
    let mut data_raw: Option<Vec<f32>> = None;
    if let Some(sigma) = maybe_sigma_frames {
        data_raw = Some(matrix.clone());
        smooth_time_gaussian_colmajor(&mut matrix, num_bins, num_frames, sigma, truncate);
    }

    RtIndex {
        bins: bins_sorted,
        frames: frame_ids_sorted,
        frame_times,
        data: matrix,
        rows: num_bins,
        cols: num_frames,
        data_raw,
    }
}

#[inline(always)]
fn mz_from_bin(bin: u32, resolution: usize) -> f32 {
    let factor = 10f32.powi(resolution as i32);
    bin as f32 / factor
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
    unique_bins: Vec<u32>,  // sorted unique bins
    offsets: Vec<usize>,    // CSR: len = unique_bins.len() + 1
    scan_idx: Vec<u16>,     // concatenated per-bin
    intensity: Vec<f32>,    // concatenated per-bin
    num_scans: usize,       // GLOBAL scan axis length
}

fn build_frame_bin_view(
    fr: TimsFrame,
    resolution: usize,
    global_num_scans: usize,
) -> FrameBinView {
    let n = fr.ims_frame.mz.len();

    // ---- collect per-point data ----
    let mut bins:    Vec<u32> = Vec::with_capacity(n);
    let mut scans_u: Vec<u16> = Vec::with_capacity(n);
    let mut intens:  Vec<f32> = Vec::with_capacity(n);

    let scans_vec: &Vec<i32> = &fr.scan; // <-- your real field
    for i in 0..n {
        bins.push(quantize_mz(fr.ims_frame.mz[i] as f32, resolution));
        let s_val = scans_vec[i];
        debug_assert!(s_val >= 0, "Negative scan index in frame {}", fr.frame_id);
        let s_u16: u16 = u16::try_from(s_val).expect("scan index does not fit u16");
        scans_u.push(
            (s_u16 as usize).min(global_num_scans.saturating_sub(1)) as u16
        );
        intens.push(fr.ims_frame.intensity[i] as f32);
    }

    // ---- sort by bin and build CSR offsets ----
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by_key(|&i| bins[i]);

    let mut unique_bins: Vec<u32> = Vec::new();
    let mut counts: Vec<usize> = Vec::new(); // temp counts per unique bin
    let mut scan_sorted: Vec<u16> = Vec::with_capacity(n);
    let mut inten_sorted: Vec<f32> = Vec::with_capacity(n);

    let mut cur_bin: Option<u32> = None;
    for &i in &idx {
        let b = bins[i];
        if cur_bin.map_or(true, |cb| cb != b) {
            unique_bins.push(b);
            counts.push(0);
            cur_bin = Some(b);
        }
        *counts.last_mut().unwrap() += 1;
        scan_sorted.push(scans_u[i]);
        inten_sorted.push(intens[i]);
    }

    // prefix-sum counts → offsets
    let mut offsets = Vec::with_capacity(unique_bins.len() + 1);
    offsets.push(0);
    for c in &counts {
        offsets.push(offsets.last().unwrap() + *c);
    }

    FrameBinView {
        _frame_id: fr.frame_id as u32,
        unique_bins,
        offsets,
        scan_idx: scan_sorted,
        intensity: inten_sorted,
        num_scans: global_num_scans,
    }
}

#[inline(always)]
fn bin_range(view: &FrameBinView, bin_lo: u32, bin_hi: u32) -> (usize, usize) {
    use core::cmp::Ordering;
    // lower_bound on unique_bins
    let mut lo = 0usize;
    let mut hi = view.unique_bins.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        match view.unique_bins[mid].cmp(&bin_lo) {
            Ordering::Less => lo = mid + 1,
            _ => hi = mid,
        }
    }
    let start_bin_ix = lo;

    // upper_bound on unique_bins
    lo = 0; hi = view.unique_bins.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        match view.unique_bins[mid].cmp(&bin_hi) {
            Ordering::Greater => hi = mid,
            _ => lo = mid + 1,
        }
    }
    let end_bin_ix = lo;

    // map bin-index range → concatenated slice range
    let start = view.offsets[start_bin_ix];
    let end   = view.offsets[end_bin_ix];
    (start, end) // these index scan_idx/intensity
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

pub fn build_dense_im_by_rtpeaks(
    data_handle: &TimsDatasetDIA,
    peaks: Vec<RtPeak1D>,    // rows
    bins: &[u32],            // from RtIndex (peak.mz_row -> center bin)
    frames_rt: &[u32],       // RT-sorted frame IDs (same order as in RT build)
    resolution: usize,
    num_threads: usize,
    mz_ppm: f32,
    rt_extra_pad: usize,
    maybe_sigma_scans: Option<f32>,
    truncate: f32,
) -> ImIndex {
    let n_rows = peaks.len();
    if n_rows == 0 {
        return ImIndex { peaks, scans: vec![], data: vec![], rows: 0, cols: 0, data_raw: None };
    }

    // 1) materialize precursor frames once
    let frames = data_handle
        .get_slice(frames_rt.to_vec(), num_threads)
        .to_resolution(resolution as i32, num_threads)
        .frames;

    // 1.1) compute GLOBAL scan length
    let global_num_scans: usize = {
        // If you have metadata num_scans, use that (shown above).
        // Or derive from data if you always have fr.scan:
        frames.iter()
            .flat_map(|fr| fr.scan.iter())     // Vec<i32>
            .copied()
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0)
    };

    // 2) build minimal per-frame bin index (grouped by bin) in parallel
    use rayon::prelude::*;
    let views: Vec<FrameBinView> = frames.into_par_iter()
        .map(|fr| build_frame_bin_view(fr, resolution, global_num_scans))
        .collect();

    let num_scans = views.first().map(|v| v.num_scans).unwrap_or(0);
    let scans: Vec<usize> = (0..num_scans).collect();

    // 3) compute per-row IM profiles in parallel
    let do_smooth = maybe_sigma_scans.unwrap_or(0.0) > 0.0;

    let profiles: Vec<(Vec<f32>, Vec<f32>)> = peaks.par_iter().map(|p| {
        // RT window
        let lo = p.left_padded.saturating_sub(rt_extra_pad);
        let hi = (p.right_padded + rt_extra_pad).min(views.len() - 1);

        // ppm band around peak m/z
        let bin_center = bins[p.mz_row];
        let mz_center  = mz_from_bin(bin_center, resolution);
        let tol        = mz_center * mz_ppm * 1e-6;
        let bin_lo     = quantize_mz(mz_center - tol, resolution);
        let bin_hi     = quantize_mz(mz_center + tol, resolution);

        let mut raw = vec![0.0f32; num_scans];

        for t in lo..=hi {
            let v = &views[t];
            let (start, end) = bin_range(v, bin_lo, bin_hi);
            // walk the concatenated slice
            for i in start..end {
                let s = v.scan_idx[i] as usize;
                raw[s] += v.intensity[i];
            }
        }

        let smoothed = if do_smooth {
            let mut tmp = raw.clone();
            smooth_vector_gaussian(&mut tmp, maybe_sigma_scans.unwrap(), truncate);
            tmp
        } else {
            raw.clone()
        };

        (smoothed, raw)
    }).collect();

    // 4) pack into column-major matrix; keep data_raw if smoothed
    let cols = num_scans;
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
fn fallback_peak_full(rt_row: usize, cols: usize, y_raw: &[f32]) -> Vec<ImPeak1D> {
    if cols == 0 { return Vec::new(); }
    let left_x = 0.0f32;
    let right_x = (cols - 1) as f32;
    let area = trapezoid_area_fractional(y_raw, left_x, right_x);
    let (scan_max, apex_raw) = y_raw.iter().copied().enumerate()
        .max_by(|a,b| a.1.total_cmp(&b.1)).unwrap_or((0,0.0));
    vec![ImPeak1D {
        rt_row, scan: scan_max, mobility: None,
        apex_smoothed: apex_raw, apex_raw,
        prominence: 0.0,
        left: 0, right: cols - 1,
        left_x, right_x,
        width_scans: cols, area_raw: area, subscan: 0.0,
    }]
}

fn fallback_peak_active(
    rt_row: usize,
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
        rt_row, scan: scan_max, mobility: None,
        apex_smoothed: apex_raw, apex_raw,
        prominence: 0.0,
        left, right, left_x, right_x,
        width_scans: right + 1 - left, area_raw: area, subscan: 0.0,
    }]
}

/// Safe ApexWindow fallback: center at apex (max raw), use ±half_width scans,
/// clamp to [0, n-1], and enforce a minimum width. Works for n==0/1 as well.
///
/// `min_width`: ensure at least this many scans (inclusive of both ends).
fn fallback_peak_apex_window(
    rt_row: usize,
    y_raw: &[f32],
    half_width: usize,
    min_width: usize,   // e.g., 3 or 5; set to 1 to effectively disable
) -> Vec<ImPeak1D> {
    let n = y_raw.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        // Degenerate single-scan row
        let apex_raw = y_raw[0];
        return vec![ImPeak1D {
            rt_row,
            scan: 0,
            mobility: None,
            apex_smoothed: apex_raw,
            apex_raw,
            prominence: 0.0,
            left: 0,
            right: 0,
            left_x: 0.0,
            right_x: 0.0,
            width_scans: 1,
            area_raw: apex_raw, // treat as area of that single sample
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

        // If still not enough (row too short), just use full range
        width = right + 1 - left;
        if width < min_width {
            left = 0;
            right = n - 1;
            width = n;
        }
    }

    let left_x  = left as f32;
    let right_x = right as f32;

    // Safe area: for n>=2 and left_x <= right_x within [0, n-1]
    let area = trapezoid_area_fractional(y_raw, left_x, right_x);

    vec![ImPeak1D {
        rt_row,
        scan: apex_idx,
        mobility: None,
        apex_smoothed: apex_raw, // in fallback, we don't have extra smoothing
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
    rt_row: usize,
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
                fallback_peak_full(rt_row, n, y_raw),
            FallbackMode::ActiveRange { abs_thr, rel_thr, pad, min_width } =>
                fallback_peak_active(rt_row, y_raw, pad, min_width, abs_thr, rel_thr),
            FallbackMode::ApexWindow { half_width } =>
                fallback_peak_apex_window(rt_row, y_raw, half_width, 3),
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

    find_im_peaks_row(&y_s, y_raw, rt_row, mobility_of, min_prom, min_distance_scans, min_width_scans)
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
        find_im_peaks_row_adaptive(&y_s, &y_r, r, mobility_of, min_distance_scans, policy)
    }).collect()
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

        // 6) area on raw between fractional bounds
        let left_idx  = left_x.floor().clamp(0.0, (n-1) as f32) as usize;
        let right_idx = right_x.ceil().clamp(0.0, (n-1) as f32) as usize;
        let area = trapezoid_area_fractional(y_raw, left_x.max(0.0), right_x.min((n-1) as f32));

        let mobility = mobility_of.map(|f| f(i));

        peaks.push(ImPeak1D{
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
        find_im_peaks_row(
            &y_s, &y_r, r, mobility_of,
            min_prom, min_distance_scans, min_width_scans
        )
    }).collect()
}
