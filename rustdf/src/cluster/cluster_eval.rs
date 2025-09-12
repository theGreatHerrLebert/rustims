use mscore::timstof::frame::TimsFrame;
use rayon::prelude::*;

// ---------- Specs & Results ----------

#[derive(Clone, Debug)]
pub struct ClusterSpec {
    /// Row in the RT index (i.e., mz_row / bin index)
    pub rt_row: usize,
    /// RT frame *column* indices in the RT matrix (inclusive)
    pub rt_left: usize,
    pub rt_right: usize,
    /// IM scan bounds (absolute scan indices, inclusive)
    pub scan_left: usize,
    pub scan_right: usize,
    /// ± ppm around the bin’s m/z center to include
    pub mz_ppm: f32,
    /// RT-bin quantization resolution used in RtIndex (0.1 → 1, 0.01 → 2, etc.)
    pub resolution: usize,
}

#[derive(Clone, Debug)]
pub struct ClusterPatch {
    pub rt_frames: Vec<u32>,     // RT-ordered frame IDs in the patch
    pub scans: Vec<u16>,         // contiguous scan axis [scan_left..scan_right]
    pub rows: usize,             // #frames in patch
    pub cols: usize,             // #scans in patch
    /// Row-major: idx = r * cols + c  (r=frame, c=scan)
    pub patch: Vec<f32>,
    pub rt_trace: Vec<f32>,      // sum over scans → length rows
    pub im_trace: Vec<f32>,      // sum over frames → length cols
    pub total_area: f32,
    pub apex_value: f32,
    pub apex_pos: (usize, usize),
}

#[derive(Clone, Debug)]
pub struct Gaussian1D {
    pub mu: f32,
    pub sigma: f32,
    pub fwhm: f32, // 2.35482 * sigma
}

#[derive(Clone, Debug)]
#[allow(non_snake_case)]
pub struct Separable2DFit {
    /// G(t,s) = A * exp(-(t-μt)^2/(2σt^2)) * exp(-(s-μs)^2/(2σs^2)) + B
    pub rt: Gaussian1D,
    pub im: Gaussian1D,
    pub A: f32,
    pub B: f32,
}

#[derive(Clone, Debug)]
pub struct ClusterQuality {
    pub r2: f32,               // 2D R^2 on the patch
    pub mse: f32,              // mean squared error
    pub snr_local: f32,        // apex / MAD(border)
    pub edge_mass_frac: f32,   // mass on outer ring / total
}

#[derive(Clone, Debug)]
pub struct ClusterResult {
    pub spec: ClusterSpec,
    pub patch: ClusterPatch,
    pub fit: Separable2DFit,
    pub q: ClusterQuality,
}

// ---------- Utils ----------

#[inline(always)]
fn bin_center_from_quantized(bin: u32, resolution: usize) -> f32 {
    let factor = 10f32.powi(resolution as i32);
    (bin as f32) / factor
}

#[inline(always)]
fn ppm_tol(mz: f32, ppm: f32) -> f32 {
    mz * ppm * 1e-6
}

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 { a + t*(b-a) }

/// Safe integration over [x0,x1] on piecewise-linear y, avoids y[n] OOB.
fn trapezoid_area_fractional(y: &[f32], mut x0: f32, mut x1: f32) -> f32 {
    let n = y.len();
    if n < 2 { return 0.0; }
    let max_x = (n-1) as f32;
    x0 = x0.clamp(0.0, max_x);
    x1 = x1.clamp(0.0, max_x);
    if x1 <= x0 { return 0.0; }

    let i0 = x0.floor() as usize;
    let i1 = x1.floor() as usize;

    if i0 == i1 {
        let i = i0.min(n-2);
        let t0 = x0 - i as f32;
        let t1 = x1 - i as f32;
        let y0 = lerp(y[i], y[i+1], t0);
        let y1 = lerp(y[i], y[i+1], t1);
        return 0.5*(y0+y1)*(t1-t0);
    }

    let s0 = i0.min(n-2);
    let s1 = i1.min(n-2);
    let mut area = 0.0f32;

    // left partial (s0)
    let t0 = x0 - s0 as f32;
    let yl0 = lerp(y[s0], y[s0+1], t0);
    let yl1 = y[s0+1];
    area += 0.5*(yl0+yl1)*(1.0 - t0);

    // interiors
    if s1 > s0 + 1 {
        for s in (s0+1)..s1 {
            area += 0.5*(y[s] + y[s+1]);
        }
    }

    // right partial only if i1 <= n-2
    if i1 <= n-2 {
        let t1 = x1 - s1 as f32;
        let yr0 = y[s1];
        let yr1 = lerp(y[s1], y[s1+1], t1);
        area += 0.5*(yr0+yr1)*t1;
    }
    area
}

#[inline(always)]
fn weighted_mean_var(y: &[f32]) -> (f32, f32) {
    let mut wsum = 0.0f32;
    let mut mean = 0.0f32;
    for (i,&v) in y.iter().enumerate() {
        let w = v.max(0.0);
        wsum += w;
        mean += w * (i as f32);
    }
    if wsum <= 0.0 { return ((y.len() as f32 - 1.0)/2.0, (y.len() as f32)/6.0); }
    mean /= wsum;
    let mut var = 0.0f32;
    for (i,&v) in y.iter().enumerate() {
        let w = v.max(0.0);
        let d = (i as f32) - mean;
        var += w * d * d;
    }
    var = if wsum > 0.0 { var/wsum } else { 0.0 };
    (mean, var.max(1e-6))
}

fn median_abs_dev_border(p: &[f32], rows: usize, cols: usize) -> f32 {
    if rows==0 || cols==0 { return 1.0; }
    let mut ring: Vec<f32> = Vec::with_capacity(rows*2 + cols*2);
    let row0 = 0usize; let row1 = rows-1;
    let col0 = 0usize; let col1 = cols-1;

    for c in 0..cols { ring.push(p[row0*cols + c]); }
    for c in 0..cols { ring.push(p[row1*cols + c]); }
    for r in 0..rows { ring.push(p[r*cols + col0]); }
    for r in 0..rows { ring.push(p[r*cols + col1]); }

    if ring.is_empty() { return 1.0; }
    ring.sort_by(|a,b| a.total_cmp(b));
    let med = ring[ring.len()/2];
    let mut dev: Vec<f32> = ring.into_iter().map(|v| (v-med).abs()).collect();
    dev.sort_by(|a,b| a.total_cmp(b));
    let mad = dev[dev.len()/2].max(1e-3);
    mad
}

// Solve [ΣG2  ΣG;  ΣG  N] [A; B] = [ΣPG; ΣP]
fn solve_AB(sum_g2: f64, sum_g: f64, n: usize, sum_pg: f64, sum_p: f64) -> (f32, f32) {
    let n = n as f64;
    let det = sum_g2 * n - sum_g * sum_g;
    if det.abs() < 1e-12 { return (0.0, 0.0); }
    let inv00 =  n / det;
    let inv01 = -sum_g / det;
    let inv10 = -sum_g / det;
    let inv11 =  sum_g2 / det;
    let A = (inv00 * sum_pg + inv01 * sum_p) as f32;
    let B = (inv10 * sum_pg + inv11 * sum_p) as f32;
    (A, B)
}

// ---------- Extraction ----------

/// Build a dense patch for one cluster. Uses frames already in RT order.
/// `frames_in_rt_order[col]` must be the frame for that RT column.
/// `bins` + `spec.rt_row` + `spec.resolution` give the m/z center.
/// We include points with |mz - center| <= ppm_tol and scan in [scan_left, scan_right].
pub fn extract_patch_for_cluster(
    frames_in_rt_order: &[&TimsFrame],
    frame_ids_sorted: &[u32],
    bins: &[u32],
    spec: &ClusterSpec,
) -> ClusterPatch {
    let rt_left = spec.rt_left.min(frame_ids_sorted.len().saturating_sub(1));
    let rt_right = spec.rt_right.min(frame_ids_sorted.len().saturating_sub(1));
    let rows = if rt_right >= rt_left { rt_right - rt_left + 1 } else { 0 };
    let scan_left = spec.scan_left;
    let scan_right = spec.scan_right;
    let cols = if scan_right >= scan_left { scan_right - scan_left + 1 } else { 0 };

    let mut patch = vec![0.0f32; rows * cols];
    let rt_frames: Vec<u32> = frame_ids_sorted[rt_left..=rt_right].to_vec();
    let scans: Vec<u16> = (scan_left..=scan_right).map(|s| s as u16).collect();

    if rows == 0 || cols == 0 {
        return ClusterPatch {
            rt_frames, scans, rows, cols, patch,
            rt_trace: vec![], im_trace: vec![],
            total_area: 0.0, apex_value: 0.0, apex_pos: (0,0),
        };
    }

    let mz_center = bin_center_from_quantized(bins[spec.rt_row] as u32, spec.resolution);
    let tol = ppm_tol(mz_center, spec.mz_ppm);

    for (r, col) in (rt_left..=rt_right).enumerate() {
        let fr = frames_in_rt_order[col];
        let mzs = &fr.ims_frame.mz;
        let ints = &fr.ims_frame.intensity;
        let scans_src: Option<&Vec<i32>> = Some(fr.scan.as_ref());

        // cheap fallback if scan vector absent: round-robin assign
        let fallback_scan = |i: usize| -> i32 {
            let n = fr.ims_frame.mz.len().max(1);
            (i % n) as i32
        };

        for (i, (&mz, &y)) in mzs.iter().zip(ints.iter()).enumerate() {
            if (mz as f32 - mz_center).abs() <= tol {
                let sc = scans_src.map(|v| v[i]).unwrap_or_else(|| fallback_scan(i));
                if sc >= scan_left as i32 && sc <= scan_right as i32 {
                    let c = (sc as usize) - scan_left;
                    patch[r * cols + c] += y.clone() as f32;
                }
            }
        }
    }

    // marginals & stats
    let mut rt_trace = vec![0.0f32; rows];
    let mut im_trace = vec![0.0f32; cols];
    let mut total = 0.0f32;
    let mut apex = 0.0f32;
    let mut apex_pos = (0usize, 0usize);

    for r in 0..rows {
        let row_slice = &patch[r * cols .. (r+1) * cols];
        let sum_r: f32 = row_slice.iter().copied().sum();
        rt_trace[r] = sum_r;
        total += sum_r;
        for c in 0..cols {
            let v = row_slice[c];
            im_trace[c] += v;
            if v > apex {
                apex = v;
                apex_pos = (r, c);
            }
        }
    }

    ClusterPatch {
        rt_frames, scans, rows, cols, patch,
        rt_trace, im_trace, total_area: total,
        apex_value: apex, apex_pos
    }
}

// ---------- Separable Fit (RT×IM) ----------

fn gaussian_1d_from_trace(trace: &[f32]) -> Gaussian1D {
    let (mu, var) = weighted_mean_var(trace);
    let sigma = var.sqrt().max(1e-3);
    let fwhm = 2.354820045f32 * sigma;
    Gaussian1D { mu, sigma, fwhm }
}

fn build_sep_core(rt: &Gaussian1D, im: &Gaussian1D, rows: usize, cols: usize) -> (Vec<f32>, f64, f64) {
    // g_t[r], g_s[c], G[r,c] = g_t[r]*g_s[c]
    let mut g_t = vec![0.0f32; rows];
    let mut g_s = vec![0.0f32; cols];
    let inv2_rt = 0.5f32 / (rt.sigma*rt.sigma);
    let inv2_im = 0.5f32 / (im.sigma*im.sigma);
    for r in 0..rows {
        let d = r as f32 - rt.mu;
        g_t[r] = (-d*d*inv2_rt).exp();
    }
    for c in 0..cols {
        let d = c as f32 - im.mu;
        g_s[c] = (-d*d*inv2_im).exp();
    }
    // ΣG and ΣG2 (for normal equations of A,B)
    let sum_g: f64 = g_t.iter().map(|&gt| gt as f64).sum::<f64>() *
        g_s.iter().map(|&gs| gs as f64).sum::<f64>();
    // ΣG2 = (Σ g_t^2) * (Σ g_s^2)
    let sum_gt2: f64 = g_t.iter().map(|&gt| (gt as f64)*(gt as f64)).sum();
    let sum_gs2: f64 = g_s.iter().map(|&gs| (gs as f64)*(gs as f64)).sum();
    let sum_g2 = sum_gt2 * sum_gs2;

    // Store the outer product as a flat Vec to avoid recomputation:
    let mut G = vec![0.0f32; rows*cols];
    for r in 0..rows {
        for c in 0..cols {
            G[r*cols + c] = g_t[r] * g_s[c];
        }
    }
    (G, sum_g, sum_g2)
}

fn fit_ab_on_patch(patch: &[f32], G: &[f32]) -> (f32, f32, f32, f32) {
    let n = patch.len();
    let mut sum_p = 0.0f64;
    let mut sum_pg = 0.0f64;
    let mut sum_p2 = 0.0f64;
    for i in 0..n {
        let p = patch[i] as f64;
        let g = G[i] as f64;
        sum_p  += p;
        sum_pg += p * g;
        sum_p2 += p * p;
    }
    // ΣG and ΣG2 must be computed alongside G; pass them if you prefer avoiding re-scan
    let mut sum_g = 0.0f64;
    let mut sum_g2 = 0.0f64;
    for &g in G {
        let g = g as f64;
        sum_g  += g;
        sum_g2 += g*g;
    }
    let (A, B) = solve_AB(sum_g2, sum_g, n, sum_pg, sum_p);

    // SSE and SST for R^2
    let mut sse = 0.0f64;
    for i in 0..n {
        let pred = A as f64 * (G[i] as f64) + B as f64;
        let e = (patch[i] as f64) - pred;
        sse += e * e;
    }
    let mean_p = sum_p / (n as f64);
    let mut sst = 0.0f64;
    for i in 0..n {
        let d = (patch[i] as f64) - mean_p;
        sst += d * d;
    }
    let r2 = if sst > 0.0 { 1.0 - (sse / sst) } else { 0.0 };
    (A, B, r2 as f32, (sse/(n as f64)) as f32) // return A,B,R2,MSE
}

fn edge_mass_fraction(p: &[f32], rows: usize, cols: usize) -> f32 {
    if rows==0 || cols==0 { return 0.0; }
    let mut edge = 0.0f32;
    let mut tot = 0.0f32;
    for r in 0..rows {
        for c in 0..cols {
            let v = p[r*cols + c];
            tot += v;
            if r==0 || r==rows-1 || c==0 || c==cols-1 { edge += v; }
        }
    }
    if tot <= 0.0 { 0.0 } else { edge / tot }
}

/// Fit separable 2D Gaussian and compute quality metrics.
pub fn fit_separable_and_score(p: &ClusterPatch) -> (Separable2DFit, ClusterQuality) {
    let rt = gaussian_1d_from_trace(&p.rt_trace);
    let im = gaussian_1d_from_trace(&p.im_trace);

    let (G, _sum_g, _sum_g2) = build_sep_core(&rt, &im, p.rows, p.cols);
    let (A, B, r2, mse) = fit_ab_on_patch(&p.patch, &G);

    let mad = median_abs_dev_border(&p.patch, p.rows, p.cols);
    let snr = if mad > 0.0 { p.apex_value / mad } else { 0.0 };
    let edge_frac = edge_mass_fraction(&p.patch, p.rows, p.cols);

    let fit = Separable2DFit { rt, im, A, B };
    let q = ClusterQuality { r2, mse, snr_local: snr, edge_mass_frac: edge_frac };
    (fit, q)
}

// ---------- Top-level: extract + fit many clusters ----------

/// Evaluate many clusters: extract patches **in RAM**, fit separable 2D Gaussian, score.
pub fn evaluate_clusters_separable(
    frames_in_rt_order: &[&TimsFrame], // same order you used to build RtIndex
    frame_ids_sorted: &[u32],
    bins: &[u32],
    specs: &[ClusterSpec],
) -> Vec<ClusterResult> {
    specs.par_iter().map(|spec| {
        let patch = extract_patch_for_cluster(frames_in_rt_order, frame_ids_sorted, bins, spec);
        let (fit, q) = fit_separable_and_score(&patch);
        ClusterResult { spec: spec.clone(), patch, fit, q }
    }).collect()
}