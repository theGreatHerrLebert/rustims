use rayon::prelude::*;
use crate::cluster::peak::{ImPeak1D, TofScanWindowGrid};
use crate::cluster::utility::{im_peak_id, quad_subsample, smooth_vector_gaussian, trapezoid_area_fractional, MobilityFn};

fn gaussian_kernel_1d(sigma: f32, truncate: f32) -> Vec<f32> {
    if sigma <= 0.0 { return Vec::new(); }
    let radius = (truncate * sigma).ceil() as i32;
    if radius < 1 { return Vec::new(); }
    let mut w = Vec::with_capacity((2 * radius + 1) as usize);
    let two_s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;
    for dx in -radius..=radius {
        let x = dx as f32;
        let val = (-x * x / two_s2).exp();
        w.push(val);
        sum += val;
    }
    if sum > 0.0 {
        for v in &mut w { *v /= sum; }
    }
    w
}

/// In-place separable Gaussian blur on a dense [rows × cols] image
fn gaussian_blur_dense_2d(
    data: &mut [f32],
    rows: usize,
    cols: usize,
    sigma_scan: f32,
    sigma_tof: f32,
    truncate: f32,
) {
    if rows == 0 || cols == 0 { return; }
    let mut tmp = vec![0.0f32; rows * cols];

    // blur along scan (columns) if needed
    let k_scan = gaussian_kernel_1d(sigma_scan, truncate);
    if !k_scan.is_empty() {
        let r = k_scan.len() as isize;
        let rad = (r - 1) / 2;
        for row in 0..rows {
            for col in 0..cols {
                let mut acc = 0.0f32;
                let mut norm = 0.0f32;
                for (off, &w) in k_scan.iter().enumerate() {
                    let dc = off as isize - rad;
                    let c2 = col as isize + dc;
                    if c2 < 0 || c2 >= cols as isize { continue; }
                    let idx = row * cols + c2 as usize;
                    acc += w * data[idx];
                    norm += w;
                }
                tmp[row * cols + col] =
                    if norm > 0.0 { acc / norm } else { 0.0 };
            }
        }
        data.copy_from_slice(&tmp);
    }

    // blur along TOF (rows) if needed
    let k_tof = gaussian_kernel_1d(sigma_tof, truncate);
    if !k_tof.is_empty() {
        let r = k_tof.len() as isize;
        let rad = (r - 1) / 2;
        for row in 0..rows {
            for col in 0..cols {
                let mut acc = 0.0f32;
                let mut norm = 0.0f32;
                for (off, &w) in k_tof.iter().enumerate() {
                    let dr = off as isize - rad;
                    let r2 = row as isize + dr;
                    if r2 < 0 || r2 >= rows as isize { continue; }
                    let idx = r2 as usize * cols + col;
                    acc += w * data[idx];
                    norm += w;
                }
                tmp[row * cols + col] =
                    if norm > 0.0 { acc / norm } else { 0.0 };
            }
        }
        data.copy_from_slice(&tmp);
    }
}

fn find_seeds_2d(
    img: &[f32],
    rows: usize,
    cols: usize,
    min_val: f32,
    nms_kernel_scan: usize,
    nms_kernel_tof: usize,
) -> Vec<(usize, usize)> {
    let mut seeds = Vec::new();
    if rows == 0 || cols == 0 { return seeds; }

    let k_scan = nms_kernel_scan.max(1) | 1; // force odd
    let k_tof  = nms_kernel_tof.max(1) | 1;
    let r_scan = k_scan / 2;
    let r_tof  = k_tof / 2;

    for r in 0..rows {
        for c in 0..cols {
            let center = img[r * cols + c];
            if center < min_val { continue; }

            let mut is_max = true;
            let r0 = r.saturating_sub(r_tof);
            let r1 = (r + r_tof).min(rows - 1);
            let c0 = c.saturating_sub(r_scan);
            let c1 = (c + r_scan).min(cols - 1);

            'n: for rr in r0..=r1 {
                for cc in c0..=c1 {
                    if rr == r && cc == c { continue; }
                    if img[rr * cols + cc] > center {
                        is_max = false;
                        break 'n;
                    }
                }
            }

            if is_max {
                seeds.push((r, c));
            }
        }
    }
    seeds
}

fn build_im_trace_from_seed(
    data_raw: &[f32], // original (unscaled) dense grid [rows × cols]
    rows: usize,
    cols: usize,
    seed_row: usize,
    pool_tof_rows: usize,
) -> Vec<f32> {
    let mut trace = vec![0.0f32; cols];
    let r0 = seed_row.saturating_sub(pool_tof_rows);
    let r1 = (seed_row + pool_tof_rows).min(rows - 1);

    for r in r0..=r1 {
        let off = r * cols;
        for c in 0..cols {
            trace[c] += data_raw[off + c];
        }
    }
    trace
}

fn im_peak_from_seed(
    y_smoothed: &[f32],
    y_raw: &[f32],
    tof_row: usize,
    tof_center: i32,
    tof_bounds: (i32, i32),
    rt_bounds_frames: (usize, usize),
    frame_id_bounds: (u32, u32),
    window_group: Option<u32>,
    mobility_of: MobilityFn,
    seed_scan_idx: usize,
    min_prom: f32,
    min_width_scans: usize,
    scan_axis: &[usize],
) -> Option<ImPeak1D> {
    let n = y_smoothed.len();
    if n < 3 || seed_scan_idx >= n { return None; }

    let row_max = y_raw.iter().copied().fold(0.0f32, f32::max);
    if row_max <= 0.0 { return None; }

    // 1) choose apex as max in a small neighborhood around seed
    let win = 3_usize;
    let mut i_apex = seed_scan_idx;
    let mut y_apex = y_smoothed[seed_scan_idx];

    let lo = seed_scan_idx.saturating_sub(win);
    let hi = (seed_scan_idx + win).min(n - 1);
    for i in lo..=hi {
        if y_smoothed[i] > y_apex {
            y_apex = y_smoothed[i];
            i_apex = i;
        }
    }
    if y_apex <= 0.0 { return None; }

    // 2) prominence baseline as in find_im_peaks_row
    let mut l = i_apex;
    let mut left_min = y_apex;
    while l > 0 {
        l -= 1;
        left_min = left_min.min(y_smoothed[l]);
        if y_smoothed[l] > y_apex { break; }
    }
    let mut r = i_apex;
    let mut right_min = y_apex;
    while r + 1 < n {
        r += 1;
        right_min = right_min.min(y_smoothed[r]);
        if y_smoothed[r] > y_apex { break; }
    }
    let baseline = left_min.max(right_min);
    let prom = y_apex - baseline;
    if prom < min_prom { return None; }

    // 3) half-height crossings
    let half = baseline + 0.5 * prom;

    let mut wl = i_apex;
    while wl > 0 && y_smoothed[wl] > half { wl -= 1; }
    let left_x = if wl < i_apex && wl + 1 < n {
        let y0 = y_smoothed[wl];
        let y1 = y_smoothed[wl + 1];
        wl as f32 + if y1 != y0 { (half - y0) / (y1 - y0) } else { 0.0 }
    } else {
        wl as f32
    };

    let mut wr = i_apex;
    while wr + 1 < n && y_smoothed[wr] > half { wr += 1; }
    let right_x = if wr > i_apex && wr < n {
        let y0 = y_smoothed[wr - 1];
        let y1 = y_smoothed[wr];
        (wr - 1) as f32 + if y1 != y0 { (half - y0) / (y1 - y0) } else { 0.0 }
    } else {
        wr as f32
    };

    let width = (right_x - left_x).max(0.0);
    let width_scans = width.round() as usize;
    if width_scans < min_width_scans { return None; }

    // 4) sub-scan apex offset
    let sub = if i_apex > 0 && i_apex + 1 < n {
        quad_subsample(
            y_smoothed[i_apex - 1],
            y_smoothed[i_apex],
            y_smoothed[i_apex + 1],
        )
            .clamp(-0.5, 0.5)
    } else {
        0.0
    };

    // 5) area on raw
    let left_idx = left_x.floor().clamp(0.0, (n - 1) as f32) as usize;
    let right_idx = right_x.ceil().clamp(0.0, (n - 1) as f32) as usize;
    let area = trapezoid_area_fractional(
        y_raw,
        left_x.max(0.0),
        right_x.min((n - 1) as f32),
    );

    // mobility & absolute scan coords
    let mobility = mobility_of.map(|f| f(i_apex));
    let scan_abs = scan_axis[i_apex];
    let left_abs = scan_axis[left_idx.min(scan_axis.len() - 1)];
    let right_abs = scan_axis[right_idx.min(scan_axis.len() - 1)];

    let mut peak = ImPeak1D {
        tof_row,
        tof_center,
        tof_bounds,
        rt_bounds: rt_bounds_frames,
        frame_id_bounds,
        window_group,

        scan: i_apex,
        left: left_idx,
        right: right_idx,

        scan_abs,
        left_abs,
        right_abs,

        scan_sigma: None,
        mobility,
        apex_smoothed: y_apex,
        apex_raw: y_raw[i_apex],
        prominence: prom,
        left_x,
        right_x,
        width_scans,
        area_raw: area,
        subscan: scan_abs as f32 + sub, // <- absolute coordinate now!
        id: 0,
    };
    peak.id = im_peak_id(&peak);
    Some(peak)
}

#[derive(Clone, Copy, Debug)]
pub struct ImDetectParams2D {
    pub min_prom_scaled: f32,          // like min_intensity_scaled in Python
    pub smooth_sigma_scan: f32,        // Gaussian blur along scans
    pub smooth_sigma_tof: f32,         // Gaussian blur along TOF rows
    pub smooth_trunc_k: f32,           // kernel truncate
    pub nms_kernel_scan: usize,        // NMS window along scans
    pub nms_kernel_tof: usize,         // NMS window along TOF
    pub pool_tof_rows: usize,          // TOF integration half-width around seed
    pub min_width_scans: usize,
    pub min_distance_scans: usize,
    pub scale_mode: ScaleMode,         // None | Sqrt | Log1p
}

#[derive(Clone, Copy, Debug)]
pub enum ScaleMode {
    None,
    Sqrt,
    Log1p,
}

pub fn detect_im_peaks_from_tof_scan_grid_2d(
    grid: &TofScanWindowGrid,
    mobility_of: MobilityFn,
    params: ImDetectParams2D,
) -> Vec<ImPeak1D> {
    let rows = grid.rows;
    let cols = grid.cols;
    if rows == 0 || cols == 0 {
        return Vec::new();
    }

    // ---- materialize raw & working image (same as before) -----------------
    let data_raw_slice: &[f32] = grid
        .data_raw
        .as_ref()
        .map(|v| v.as_slice())
        .unwrap_or_else(|| {
            grid.data
                .as_ref()
                .expect("TofScanWindowGrid.data is None; materialize grid before IM detection")
                .as_slice()
        });

    let mut img = data_raw_slice.to_vec();
    debug_assert_eq!(img.len(), rows * cols);

    let rt_bounds_frames = grid.rt_range_frames;
    let frame_id_bounds = grid.frame_id_bounds;
    let window_group = grid.window_group;
    let scans_axis: &[usize] = &grid.scans;
    let scale = &grid.scale;

    // --- scale (same as before) -------------------------------------------
    match params.scale_mode {
        ScaleMode::None => {}
        ScaleMode::Sqrt => {
            for v in &mut img {
                *v = v.max(0.0).sqrt();
            }
        }
        ScaleMode::Log1p => {
            for v in &mut img {
                *v = (1.0 + v.max(0.0)).ln();
            }
        }
    }

    // --- 2D blur on scaled image (still sequential, but one big pass) -----
    gaussian_blur_dense_2d(
        &mut img,
        rows,
        cols,
        params.smooth_sigma_scan,
        params.smooth_sigma_tof,
        params.smooth_trunc_k,
    );

    // --- 2D NMS seeding (TOF × scans) -------------------------------------
    let seeds = find_seeds_2d(
        &img,
        rows,
        cols,
        params.min_prom_scaled,
        params.nms_kernel_scan,
        params.nms_kernel_tof,
    );
    if seeds.is_empty() {
        return Vec::new();
    }

    // ---- parallel over seeds ---------------------------------------------
    let mobility_of_copy = mobility_of;
    let pool_tof_rows = params.pool_tof_rows;
    let min_width_scans = params.min_width_scans;

    let peaks_out: Vec<ImPeak1D> = seeds
        .into_par_iter()
        .filter_map(|(tof_row, seed_col)| {
            // build IM trace (scan axis) from a small TOF neighborhood of the seed
            let y_raw = build_im_trace_from_seed(
                data_raw_slice,
                rows,
                cols,
                tof_row,
                pool_tof_rows,
            );
            if y_raw.iter().all(|v| *v <= 0.0) {
                return None;
            }

            let mut y_smoothed = y_raw.clone();
            smooth_vector_gaussian(
                &mut y_smoothed[..],
                params.smooth_sigma_scan.max(0.75),
                params.smooth_trunc_k,
            );

            // TOF metadata (pure reads from shared scale)
            let tof_center_f = scale.centers[tof_row];
            let tof_lo_f = scale.edges[tof_row];
            let tof_hi_f = scale.edges[tof_row + 1];

            let tof_center = tof_center_f.round() as i32;
            let tof_bounds = (tof_lo_f.round() as i32, tof_hi_f.round() as i32);

            // Note: min_prom in raw units; seeding used scaled threshold already
            im_peak_from_seed(
                &y_smoothed,
                &y_raw,
                tof_row,
                tof_center,
                tof_bounds,
                rt_bounds_frames,
                frame_id_bounds,
                window_group,
                mobility_of_copy,
                seed_col,
                0.0,
                min_width_scans,
                scans_axis,
            )
        })
        .collect();

    peaks_out
}