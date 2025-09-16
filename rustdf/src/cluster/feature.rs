use std::sync::Arc;
use mscore::timstof::frame::TimsFrame;
use mscore::algorithm::isotope::generate_averagine_spectra;

#[derive(Clone, Debug)]
pub struct AveragineLut {
    pub masses: Vec<f32>,               // grid (neutral mass, Da)
    pub z_min: u8,
    pub z_max: u8,
    pub k: usize,                       // peaks per envelope kept
    // flattened: [(z=z_min..z_max) × mass grid] → [k intensities]
    pub envs: Vec<[f32; 8]>,            // k≤8 for speed; pad with zeros
}

impl AveragineLut {
    pub fn build(
        mass_min: f32,
        mass_max: f32,
        step: f32,          // e.g. 25–50 Da
        z_min: u8,
        z_max: u8,
        k: usize,           // keep first k peaks, ≤8
        resolution: i32,    // pass-through to your generator
        num_threads: usize,
    ) -> Self {
        let mut masses = Vec::new();
        let mut m = mass_min.max(200.0);
        while m <= mass_max { masses.push(m); m += step; }

        let mut envs: Vec<[f32; 8]> = Vec::with_capacity(masses.len() * (z_max - z_min + 1) as usize);

        // generate averagine spectra for each mass/charge, then compact to k peaks (intensity-normalized)
        for z in z_min..=z_max {
            let charges: Vec<i32> = vec![z as i32; masses.len()];
            let masses_f64: Vec<f64> = masses.iter().map(|&x| x as f64).collect();
            let specs = generate_averagine_spectra(
                masses_f64, charges, /*min_intensity*/1, /*k*/k as i32,
                resolution, /*centroid*/true, num_threads, /*amp*/None
            );

            for sp in specs {
                // take first k intensities, normalize to unit vector
                let mut v = [0f32; 8];
                for i in 0..k.min(sp.intensity.len()) {
                    v[i] = sp.intensity[i] as f32;
                }
                let norm = (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()).sqrt() as f32;
                if norm > 0.0 { for x in &mut v { *x /= norm; } }
                envs.push(v);
            }
        }

        Self { masses, z_min, z_max, k, envs }
    }

    #[inline]
    pub fn lookup(&self, neutral_mass: f32, z: u8) -> [f32; 8] {
        if z < self.z_min || z > self.z_max || self.masses.is_empty() {
            return [0.0; 8];
        }
        // nearest-neighbor on mass grid
        let zi = (z - self.z_min) as usize;
        let per_z = self.masses.len();
        // clamp index
        let i = match self.masses.binary_search_by(|m| m.partial_cmp(&neutral_mass).unwrap()) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1).min(per_z.saturating_sub(1)),
        };
        self.envs[zi * per_z + i]
    }
}

#[inline]
pub fn cosine(a: &[f32;8], b: &[f32;8]) -> f32 {
    let mut dot = 0f32; let mut na = 0f32; let mut nb = 0f32;
    for i in 0..8 { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt()*nb.sqrt()) }
}

#[inline]
fn mz_ppm_window(mz: f32, ppm: f32) -> (f32, f32) {
    let d = mz * ppm * 1e-6;
    (mz - d, mz + d)
}

/// Integrate k isotope stripes around a mono m/z guess, within [rt_l..rt_r] × [im_l..im_r].
/// Returns up to 8 integrals (unused tail is zero).
pub fn integrate_isotope_series(
    frames: &[Arc<TimsFrame>],      // RT-sorted, already preloaded
    rt_bounds: (usize, usize),      // inclusive indices into `frames`
    im_bounds: (usize, usize),      // inclusive absolute scan indices
    mz_mono: f32,                   // mono m/z (use c.mz_fit.mu)
    z: u8,                          // charge (>=1)
    ppm_narrow: f32,                // e.g. 8..10
    k_max: usize,                   // ≤ 8
) -> [f32; 8] {
    let mut isotopes = [0f32; 8];
    if z == 0 || k_max == 0 || frames.is_empty() { return isotopes; }

    let k_keep = k_max.min(8);

    let (rt_l, rt_r) = {
        let n = frames.len();
        (rt_bounds.0.min(n.saturating_sub(1)), rt_bounds.1.min(n.saturating_sub(1)))
    };
    if rt_l > rt_r { return isotopes; }

    let (im_l, im_r) = im_bounds;
    if im_l > im_r { return isotopes; }

    let dmz = 1.003355f32 / (z as f32);

    for k in 0..k_keep {
        let mz_k = mz_mono + (k as f32) * dmz;
        let (lo, hi) = mz_ppm_window(mz_k, ppm_narrow);

        let mut acc = 0f32;
        for f in rt_l..=rt_r {
            let fr = &frames[f];
            let mz  = &fr.ims_frame.mz;
            let it  = &fr.ims_frame.intensity;
            let scv = &fr.scan;

            // tight in-loop filters (branch-friendly)
            let len = mz.len();
            for i in 0..len {
                let m = mz[i] as f32;
                if m < lo || m > hi { continue; }

                let s = scv[i];
                if s < 0 { continue; }
                let su = s as usize;
                if su < im_l || su > im_r { continue; }

                acc += it[i] as f32;
            }
        }
        isotopes[k] = acc;
    }

    isotopes
}

/// Build a small m/z histogram around `mz_center` within the cluster's RT×IM
/// bounds. Returns (mz_axis_centers, hist_y). `bins` ~ 15–31 is plenty.
///
/// `win_ppm` is the total half-window (+/− ppm) around `mz_center`.
#[inline]
pub fn build_local_mz_histogram(
    frames: &[Arc<TimsFrame>], // RT-sorted, preloaded
    rt_bounds: (usize, usize),                                    // inclusive frame indices
    im_bounds: (usize, usize),                                    // inclusive absolute scan indices
    mz_center: f32,
    win_ppm: f32,                                                 // e.g. 20.0
    bins: usize,                                                  // e.g. 21
) -> (Vec<f32>, Vec<f32>) {
    let bins = bins.max(10);
    let d_da = mz_center * win_ppm * 1e-6;
    let lo = mz_center - d_da;
    let hi = mz_center + d_da;
    if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
        return (Vec::new(), Vec::new());
    }

    // axis centers
    let width = (hi - lo) / (bins as f32);
    let mut axis = Vec::with_capacity(bins);
    for b in 0..bins {
        axis.push(lo + (b as f32 + 0.5) * width);
    }

    let (rt_l, rt_r) = {
        let n = frames.len();
        (rt_bounds.0.min(n.saturating_sub(1)), rt_bounds.1.min(n.saturating_sub(1)))
    };
    if rt_l > rt_r { return (axis, vec![0.0; bins]); }

    let (im_l, im_r) = im_bounds;
    if im_l > im_r { return (axis, vec![0.0; bins]); }

    // accumulate
    let mut y = vec![0.0f32; bins];
    for f in rt_l..=rt_r {
        let fr = &frames[f];
        let mz  = &fr.ims_frame.mz;
        let it  = &fr.ims_frame.intensity;
        let scv = &fr.scan;

        let len = mz.len();
        for i in 0..len {
            let m = mz[i] as f32;
            if m < lo || m > hi { continue; }
            let s = scv[i];
            if s < 0 { continue; }
            let su = s as usize;
            if su < im_l || su > im_r { continue; }

            // bin
            let mut b = ((m - lo) / (hi - lo) * (bins as f32)).floor() as isize;
            if b < 0 { b = 0; }
            if (b as usize) >= bins { b = (bins as isize) - 1; }
            y[b as usize] += it[i] as f32;
        }
    }

    (axis, y)
}

#[inline]
fn arg_peaks(y: &[f32], min_prom: f32, max_peaks: usize) -> Vec<usize> {
    let n = y.len();
    if n < 3 { return Vec::new(); }
    // simple baseline for crude prominence
    let mut ys = y.to_vec();
    ys.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let base = ys[(ys.len() as f32 * 0.1).floor() as usize];

    let mut idxs = Vec::new();
    for i in 1..n-1 {
        if y[i] > y[i-1] && y[i] > y[i+1] && (y[i] - base) >= min_prom {
            idxs.push(i);
            if idxs.len() == max_peaks { break; }
        }
    }
    idxs
}

pub fn estimate_charge_from_hist(mz_axis: &[f32], mz_hist: &[f32]) -> Option<(u8, f32)> {
    if mz_axis.len() < 3 || mz_axis.len() != mz_hist.len() { return None; }
    let peaks = arg_peaks(mz_hist, 0.0, 5);
    if peaks.len() < 2 { return None; }

    let mut deltas = Vec::new();
    for w in peaks.windows(2) {
        let dm = (mz_axis[w[1]] - mz_axis[w[0]]).abs();
        if dm > 0.0 { deltas.push(dm); }
    }
    if deltas.is_empty() { return None; }

    let mut best_z = 0u8;
    let mut best_err = f32::MAX;
    for z in 1..=6u8 {
        let target = 1.003355f32 / (z as f32);
        let err = deltas.iter().map(|&d| (d - target).abs()).sum::<f32>() / (deltas.len() as f32);
        if err < best_err { best_err = err; best_z = z; }
    }
    if best_z == 0 { None } else {
        let conf = (1.0 / (1e-3 + best_err)).min(10.0) / 10.0;
        Some((best_z, conf))
    }
}