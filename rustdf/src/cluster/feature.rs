use std::sync::Arc;
use mscore::timstof::frame::TimsFrame;
use mscore::algorithm::isotope::generate_averagine_spectra;
use crate::cluster::cluster_eval::ClusterResult;

#[derive(Clone, Debug)]
pub struct GroupingParams {
    pub rt_pad_overlap: usize,     // allow N frames on each side when testing overlap
    pub im_pad_overlap: usize,     // allow M scans
    pub mz_ppm_tol: f32,           // e.g. 8.0
    pub iso_ppm_tol: f32,          // e.g. 10.0 (tolerance for Δm ~ 1.003355/z)
    pub z_min: u8,                 // e.g. 1
    pub z_max: u8,                 // e.g. 6
}

#[derive(Clone, Debug)]
pub struct GroupingOutput {
    pub envelopes: Vec<Envelope>,
    /// final assignment: cluster_i -> Some(envelope_id) after resolution
    pub assignment: Vec<Option<usize>>,
    /// optional: provisional groups before resolution
    pub provisional: Vec<Vec<usize>>,
}

pub fn group_clusters_into_envelopes(
    clusters: &[ClusterResult],
    params: &GroupingParams,
) -> GroupingOutput {
    let n = clusters.len();
    if n == 0 {
        return GroupingOutput { envelopes: Vec::new(), assignment: Vec::new(), provisional: Vec::new() };
    }

    // Build adjacency via simple O(n^2) pass (fast enough if you pre-filter “good” clusters;
    // otherwise swap to bucketed by RT and m/z).
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        let ci = &clusters[i];
        let rti = expand(ci.rt_window, params.rt_pad_overlap);
        let imi = expand(ci.im_window, params.im_pad_overlap);
        let mzi = ci.mz_fit.mu;

        for j in (i+1)..n {
            let cj = &clusters[j];
            let rtj = expand(cj.rt_window, params.rt_pad_overlap);
            let imj = expand(cj.im_window, params.im_pad_overlap);
            let mzj = cj.mz_fit.mu;

            // RT/IM overlap tests
            let rt_ov = overlap1d(rti, rtj) > 0;
            let im_ov = overlap1d(imi, imj) > 0;
            if !(rt_ov && im_ov) { continue; }

            // m/z compatible (ppm close OR isotopic offset)
            if !mz_compatible(mzi, mzj, params.mz_ppm_tol, params.iso_ppm_tol, params.z_min, params.z_max) {
                continue;
            }

            neighbors[i].push(j);
            neighbors[j].push(i);
        }
    }

    // Provisional groups = connected components
    let mut seen = vec![false; n];
    let mut provisional: Vec<Vec<usize>> = Vec::new();
    for s in 0..n {
        if seen[s] { continue; }
        let mut stack = vec![s];
        let mut comp = Vec::new();
        seen[s] = true;
        while let Some(u) = stack.pop() {
            comp.push(u);
            for &v in &neighbors[u] {
                if !seen[v] {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        provisional.push(comp);
    }

    // Build envelope summaries for provisional comps
    let mut envs_tmp = Vec::with_capacity(provisional.len());
    for comp in &provisional {
        let (rtb, imb, mzc, mzspan) =
            summarize_envelope(comp, clusters, params.rt_pad_overlap, params.im_pad_overlap);
        envs_tmp.push((rtb, imb, mzc, mzspan));
    }

    // Resolution: score each cluster→provisional envelope and pick the best one.
    // Score = RT overlap fraction * IM overlap fraction * m/z weight (Gaussian-ish).
    let mut best_env_for_cluster: Vec<Option<(usize, f32)>> = vec![None; n];

    for (eid_tmp, comp) in provisional.iter().enumerate() {
        let (rtb, imb, mzc, _span) = envs_tmp[eid_tmp];
        let rtw = (rtb.1 - rtb.0 + 1).max(1) as f32;
        let imw = (imb.1 - imb.0 + 1).max(1) as f32;
        for &cid in comp {
            let c = &clusters[cid];

            let rto = overlap1d(c.rt_window, rtb) as f32 / rtw;
            let imo = overlap1d(c.im_window, imb) as f32 / imw;

            // m/z closeness weight (not ppm, a soft Gaussian on Da)
            let dm = (c.mz_fit.mu - mzc).abs();
            let mz_w = (- (dm * 200.0).powi(2)).exp(); // ~ very sharp around center; tweak if needed

            let score = rto * imo * mz_w;
            if let Some((_, cur)) = best_env_for_cluster[cid] {
                if score > cur { best_env_for_cluster[cid] = Some((eid_tmp, score)); }
            } else {
                best_env_for_cluster[cid] = Some((eid_tmp, score));
            }
        }
    }

    // Final envelopes: collapse provisional IDs that actually got any members after resolution.
    // Map provisional eid_tmp -> final eid
    let mut map_tmp_to_final: Vec<Option<usize>> = vec![None; provisional.len()];
    let mut final_envs: Vec<Envelope> = Vec::new();

    // Collect members
    let mut members_by_tmp: Vec<Vec<usize>> = vec![Vec::new(); provisional.len()];
    for cid in 0..n {
        if let Some((eid_tmp, _)) = best_env_for_cluster[cid] {
            members_by_tmp[eid_tmp].push(cid);
        }
    }

    for (eid_tmp, members) in members_by_tmp.into_iter().enumerate() {
        if members.is_empty() { continue; }
        let final_id = final_envs.len();
        map_tmp_to_final[eid_tmp] = Some(final_id);

        let (rtb, imb, mzc, mzspan) =
            summarize_envelope(&members, clusters, /*store bounds without extra pad*/0, 0);

        final_envs.push(Envelope {
            id: final_id,
            cluster_ids: members,
            rt_bounds: rtb,
            im_bounds: imb,
            mz_center: mzc,
            mz_span_da: mzspan,
        });
    }

    // Final assignment vector
    let mut assignment = vec![None; n];
    for cid in 0..n {
        if let Some((eid_tmp, _)) = best_env_for_cluster[cid] {
            if let Some(eid) = map_tmp_to_final[eid_tmp] {
                assignment[cid] = Some(eid);
            }
        }
    }

    GroupingOutput { envelopes: final_envs, assignment, provisional }
}

/// A logical group (envelope) of clusters that likely belong together.
/// Bounds are inclusive indices (frames/scans); they’re computed from members.
#[derive(Clone, Debug)]
pub struct Envelope {
    pub id: usize,                 // 0..envelopes.len()-1
    pub cluster_ids: Vec<usize>,   // member cluster indices
    pub rt_bounds: (usize, usize), // min..max over members (with optional pad applied)
    pub im_bounds: (usize, usize),
    pub mz_center: f32,            // weighted center (by raw_sum)
    pub mz_span_da: f32,           // span across members (for info)
}

#[inline] fn ppm(delta_da: f32, center_da: f32) -> f32 {
    if center_da <= 0.0 { f32::INFINITY } else { delta_da.abs() * 1.0e6 / center_da }
}

#[inline] fn overlap1d(a: (usize,usize), b: (usize,usize)) -> usize {
    let (al, ar) = a; let (bl, br) = b;
    if ar < bl || br < al { 0 } else { ar.min(br) - al.max(bl) + 1 }
}

#[inline] fn expand(w: (usize,usize), pad: usize) -> (usize,usize) {
    (w.0.saturating_sub(pad), w.1.saturating_add(pad))
}

/// Check if two m/z centers are “compatible”: either within ppm tolerance
/// or near an isotopic spacing 1.003355 / z for z in [z_min..z_max] within iso_ppm_tol.
fn mz_compatible(a_da: f32, b_da: f32, mz_ppm_tol: f32, iso_ppm_tol: f32, z_min: u8, z_max: u8) -> bool {
    let dm = (a_da - b_da).abs();
    if ppm(dm, (a_da + b_da)*0.5) <= mz_ppm_tol { return true; }
    for z in z_min..=z_max {
        let target = 1.003355f32 / (z as f32);
        if ppm((dm - target).abs(), a_da) <= iso_ppm_tol { return true; }
    }
    false
}

/// Compact bounds/center for a set of clusters (optionally with extra padding for the bounds)
fn summarize_envelope(
    cid_list: &[usize],
    clusters: &[ClusterResult],
    rt_pad_overlap: usize,
    im_pad_overlap: usize,
) -> ( (usize,usize), (usize,usize), f32, f32 ) {
    let mut rt_min = usize::MAX; let mut rt_max = 0usize;
    let mut im_min = usize::MAX; let mut im_max = 0usize;
    let mut wsum = 0.0f32; let mut msum = 0.0f32;
    let mut mz_lo = f32::INFINITY; let mut mz_hi = f32::NEG_INFINITY;

    for &cid in cid_list {
        let c = &clusters[cid];
        let (r0, r1) = expand(c.rt_window, rt_pad_overlap);
        let (i0, i1) = expand(c.im_window, im_pad_overlap);
        rt_min = rt_min.min(r0); rt_max = rt_max.max(r1);
        im_min = im_min.min(i0); im_max = im_max.max(i1);
        let w = (c.raw_sum.max(1.0)).ln_1p(); // robust weight
        wsum += w; msum += w * c.mz_fit.mu;
        mz_lo = mz_lo.min(c.mz_fit.mu);
        mz_hi = mz_hi.max(c.mz_fit.mu);
    }
    let mz_center = if wsum > 0.0 { msum / wsum } else { (mz_lo + mz_hi) * 0.5 };
    let mz_span_da = (mz_hi - mz_lo).max(0.0);
    ((rt_min, rt_max), (im_min, im_max), mz_center, mz_span_da)
}

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

#[inline]
pub fn integrate_isotope_series(
    frames: &[Arc<TimsFrame>],
    rt_bounds: (usize, usize),
    im_bounds: (usize, usize),
    mz_mono: f32,
    z: u8,
    ppm_narrow: f32,
    k_max: usize,
) -> [f32; 8] {
    let mut isotopes = [0f32; 8];
    if z == 0 || k_max == 0 || frames.is_empty() { return isotopes; }

    let k_keep = k_max.min(8);
    let n = frames.len();
    let rt_l = rt_bounds.0.min(n.saturating_sub(1));
    let rt_r = rt_bounds.1.min(n.saturating_sub(1));
    if rt_l > rt_r { return isotopes; }

    let (im_l, im_r) = im_bounds;
    if im_l > im_r { return isotopes; }

    let dmz = 1.003355f32 / (z as f32);

    for k in 0..k_keep {
        let mz_k = mz_mono + (k as f32) * dmz;
        let (lo, hi) = mz_ppm_window(mz_k, ppm_narrow);

        // NEW: hard guard – if the window is invalid, skip this isotope.
        if !lo.is_finite() || !hi.is_finite() || hi <= lo {
            continue;
        }

        let mut acc = 0f32;
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