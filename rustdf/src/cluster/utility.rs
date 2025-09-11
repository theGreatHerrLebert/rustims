use mscore::timstof::frame::TimsFrame;
use rustc_hash::{FxHashMap, FxHashSet};
use crate::data::dia::TimsDatasetDIA;
use crate::data::handle::TimsData;
use crate::data::meta::FrameMeta;
use rayon::prelude::*;
use rayon::iter::IntoParallelRefIterator;

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

// tune this to your binning (0.01 m/z here)
#[inline(always)]
fn quantize_mz(mz: f32, resolution: usize) -> u32 {
    // faster than roundf on many CPUs
    let factor = 10f32.powi(resolution as i32);
    (mz * factor + 0.5) as u32
}

pub struct RtIndex {
    // ascending m/z bin list
    pub bins: Vec<MzBin>,
    // RT-ordered frame ids
    pub frames: Vec<u32>,
    // dense matrix column-major
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

type MzBin = u32;

// You likely have an RT/scan time in metadata; adapt `frame_rt` accordingly.
fn frame_rt(meta: &FrameMeta) -> f32 { meta.time  as f32} // <-- replace with real field

// Returns (bins_sorted, frame_ids_sorted_by_rt, dense matrix column-major)
pub fn build_dense_rt_by_mz(
    data_handle: &TimsDatasetDIA,
    maybe_sigma_frames: Option<f32>,
    truncate: f32,
    resolution: usize,
    num_threads: usize,
) -> RtIndex {

    // 1) collect precursor frames and RT sort them
    let precursor_meta: Vec<_> = data_handle.meta_data
        .iter()
        .filter(|m| m.ms_ms_type == 0)
        .collect();

    let mut rt_sorted_meta = precursor_meta.clone();
    rt_sorted_meta.sort_by(|a, b| frame_rt(a).partial_cmp(&frame_rt(b)).unwrap());

    let frame_ids_sorted: Vec<u32> = rt_sorted_meta.iter().map(|m| m.id as u32).collect();
    let num_frames = frame_ids_sorted.len();

    // quick index to go from frame_id -> column
    let _col_index: FxHashMap<u32, usize> = frame_ids_sorted
        .iter()
        .enumerate()
        .map(|(c, &fid)| (fid, c))
        .collect();

    // 2) materialize the frames you need at your chosen resolution
    let frames = data_handle
        .get_slice(frame_ids_sorted.clone(), num_threads)
        .to_resolution(resolution as i32, num_threads)
        .frames;

    // Optional: map from frame_id to its loaded frame record for O(1) access
    let mut frame_by_id: FxHashMap<u32, &TimsFrame> = FxHashMap::default();
    for f in &frames {
        frame_by_id.insert(f.frame_id as u32, f);
    }

    // 3) Pass A: discover all m/z bins in parallel
    let all_bins: FxHashSet<MzBin> = frames
        .par_iter()
        .map(|frame| {
            let mut local: FxHashSet<MzBin> = FxHashSet::default();
            // assuming f32 inputs; cast if f64
            for &mz in frame.ims_frame.mz.iter() {
                local.insert(quantize_mz(mz as f32, resolution));
            }
            local
        })
        .reduce(FxHashSet::default, |mut a, b| { a.extend(b); a });

    // sort bins for deterministic row order
    let mut bins_sorted: Vec<MzBin> = all_bins.into_iter().collect();
    bins_sorted.sort_unstable();
    let num_bins = bins_sorted.len();

    // row index for fast lookups
    let row_index: FxHashMap<MzBin, usize> = bins_sorted
        .iter()
        .enumerate()
        .map(|(r, &bin)| (bin, r))
        .collect();

    // 4) allocate dense matrix (column-major: contiguous per frame/column)
    let mut matrix = vec![0.0f32; num_bins * num_frames];

    // 5) build a column-ordered frame list (aligned with RT order)
    // Each column gets a disjoint mutable slice => safe parallel fill with no locks.
    let frames_in_col_order: Vec<&TimsFrame> = frame_ids_sorted
        .iter()
        .map(|fid| frame_by_id[fid])
        .collect();

    // Parallel fill: one thread per column
    matrix
        .par_chunks_mut(num_bins)
        .enumerate()
        .for_each(|(col, col_slice)| {
            let frame = frames_in_col_order[col];
            let mz = &frame.ims_frame.mz;
            let inten = &frame.ims_frame.intensity;

            // keep an optional small local combiner to collapse duplicates within the column
            // (helps when many points quantize to the same bin)
            // This avoids repeated row_index lookups for identical bins within the same frame.
            let mut accum: FxHashMap<usize, f32> = FxHashMap::default();

            for (&m, &i) in mz.iter().zip(inten.iter()) {
                let q = quantize_mz(m as f32, resolution);
                if let Some(&row) = row_index.get(&q) {
                    *accum.entry(row).or_insert(0.0) += i as f32;
                }
            }

            // write back once per unique row in this column
            for (row, val) in accum {
                col_slice[row] += val;
            }
        });

    if let Some(sigma) = maybe_sigma_frames {
        // IMPORTANT: smoothing along rt (columns)
        smooth_time_gaussian_colmajor(&mut matrix, num_bins, num_frames, sigma, truncate);
    }

    RtIndex {
        bins: bins_sorted,
        frames: frame_ids_sorted,
        data: matrix,
        rows: num_bins,
        cols: num_frames,
    }
}