use crate::cluster::utility::{build_frame_bin_view, smooth_vector_gaussian, FrameBinView, MzScale};
use crate::data::dia::TimsDatasetDIA;
use crate::data::handle::TimsData;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct MzScanGrid {
    pub scans: Vec<usize>,
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub data_raw: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct MzScanWindowGrid {
    pub rt_range_frames: (usize, usize),   // existing: local frame-index range (within the plan)
    pub rt_range_sec:    (f32, f32),       // existing
    pub frame_id_bounds: (u32, u32),       // NEW: actual frame IDs [lo, hi]
    pub window_group:    Option<u32>,      // NEW: DIA group provenance
    pub scans: Vec<usize>,
    pub data: Vec<f32>,     // column-major (rows, cols)
    pub rows: usize,
    pub cols: usize,
    pub data_raw: Option<Vec<f32>>,
}

pub fn build_mz_scan_grid_for_frames(
    data_handle: &TimsDatasetDIA,
    frame_ids_sorted: &[u32],
    scale: &MzScale,
    maybe_sigma_scans: Option<f32>,
    truncate: f32,
    num_threads: usize,
) -> MzScanGrid {
    let frames = data_handle.get_slice(frame_ids_sorted.to_vec(), num_threads).frames;
    if frames.is_empty() {
        return MzScanGrid { scans: vec![], data: vec![], rows: 0, cols: 0, data_raw: None };
    }

    let rows = scale.num_bins();
    let global_num_scans = data_handle.meta_data.iter()
        .map(|fr_meta| (fr_meta.num_scans + 1) as usize)
        .max()
        .unwrap_or(0);
    if rows == 0 || global_num_scans == 0 {
        return MzScanGrid { scans: vec![], data: vec![], rows: 0, cols: 0, data_raw: None };
    }

    let views: Vec<FrameBinView> = (0..frames.len())
        .into_par_iter()
        .map(|i| build_frame_bin_view(frames[i].clone(), scale, global_num_scans))
        .collect();

    let cols = global_num_scans;
    let mut raw = vec![0.0f32; rows * cols];

    for v in &views {
        for b_i in 0..v.unique_bins.len() {
            let row = v.unique_bins[b_i];
            let start = v.offsets[b_i];
            let end   = v.offsets[b_i + 1];
            for i in start..end {
                let s_phys = v.scan_idx[i] as usize;
                if s_phys < cols {
                    raw[s_phys * rows + row] += v.intensity[i];
                }
            }
        }
    }

    let do_smooth = maybe_sigma_scans.unwrap_or(0.0) > 0.0;
    let data = if do_smooth {
        let mut sm = raw.clone();
        for r in 0..rows {
            let mut y: Vec<f32> = (0..cols).map(|c| sm[c * rows + r]).collect();
            smooth_vector_gaussian(&mut y, maybe_sigma_scans.unwrap(), truncate);
            for c in 0..cols { sm[c * rows + r] = y[c]; }
        }
        sm
    } else {
        raw.clone()
    };

    MzScanGrid {
        scans: (0..cols).collect(),
        data,
        rows,
        cols,
        data_raw: if do_smooth { Some(raw) } else { None },
    }
}