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
    pub iso_abs_da: f32,
}

#[derive(Clone, Debug)]
pub struct GroupingOutput {
    pub envelopes: Vec<Envelope>,
    /// final assignment: cluster_i -> Some(envelope_id) after resolution
    pub assignment: Vec<Option<usize>>,
    /// optional: provisional groups before resolution
    pub provisional: Vec<Vec<usize>>,
}

#[derive(Clone, Debug)]
pub struct Dsu {
    parent: Vec<usize>,
    size:   Vec<usize>,
}

impl Dsu {
    #[inline]
    pub fn new(n: usize) -> Self {
        let mut parent = Vec::with_capacity(n);
        let mut size   = Vec::with_capacity(n);
        for i in 0..n { parent.push(i); size.push(1); }
        Self { parent, size }
    }

    #[inline]
    pub fn find(&mut self, mut x: usize) -> usize {
        // path compression
        let mut p = self.parent[x];
        while p != self.parent[p] {
            p = self.parent[p];
        }
        // compress along the way
        while x != self.parent[x] {
            let next = self.parent[x];
            self.parent[x] = p;
            x = next;
        }
        p
    }

    #[inline]
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb { return false; }
        // union by size
        if self.size[ra] < self.size[rb] { std::mem::swap(&mut ra, &mut rb); }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
        true
    }

    /// Collect components as Vec<Vec<usize>> (each is a set of indices).
    pub fn groups(mut self) -> Vec<Vec<usize>> {
        // canonical root for each index
        let n = self.parent.len();
        let mut root_of = vec![0usize; n];
        for i in 0..n { root_of[i] = self.find(i); }

        // map root -> list
        use std::collections::HashMap;
        let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, r) in root_of.into_iter().enumerate() {
            buckets.entry(r).or_default().push(i);
        }
        let mut out: Vec<Vec<usize>> = buckets.into_values().collect();
        out.sort_by_key(|g| g.len());
        out
    }
}

#[inline]
fn ppm_between(a_da: f32, b_da: f32) -> f32 {
    let dm = (a_da - b_da).abs();
    let center = ((a_da + b_da) * 0.5).max(1e-6);
    1.0e6 * dm / center
}

fn frac_overlap(a:(usize,usize), b:(usize,usize)) -> f32 {
    let l = a.0.max(b.0);
    let r = a.1.min(b.1);
    if r < l { 0.0 } else { (r - l + 1) as f32 / ((a.1-a.0+1).max(b.1-b.0+1) as f32) }
}

fn is_near_duplicate(ci:&ClusterResult, cj:&ClusterResult) -> bool {
    let rt = frac_overlap(ci.rt_window, cj.rt_window) >= 0.6;
    let im = frac_overlap(ci.im_window, cj.im_window) >= 0.6;
    let mz_close = ppm_between(ci.mz_fit.mu, cj.mz_fit.mu) <= 3.0; // tight
    rt && im && mz_close
}

// --- Super nodes --------------------------------------------------------------

struct Super {
    member_cids: Vec<usize>,
    rt_bounds: (usize,usize),
    im_bounds: (usize,usize),
    mz_center: f32,     // raw_sum-weighted
    raw_sum: f32,
}

fn summarize_super(group:&[usize], cl:&[ClusterResult]) -> Super {
    let mut rt_l = usize::MAX; let mut rt_r = 0usize;
    let mut im_l = usize::MAX; let mut im_r = 0usize;
    let mut wsum = 0f32; let mut mz_w = 0f32;
    let mut mz_min = f32::INFINITY; let mut mz_max = f32::NEG_INFINITY;
    let mut raw_sum = 0f32;

    for &cid in group {
        let c = &cl[cid];
        rt_l = rt_l.min(c.rt_window.0);
        rt_r = rt_r.max(c.rt_window.1);
        im_l = im_l.min(c.im_window.0);
        im_r = im_r.max(c.im_window.1);
        let w = c.raw_sum.max(1.0);
        wsum += w; mz_w += w * c.mz_fit.mu;
        mz_min = mz_min.min(c.mz_fit.mu);
        mz_max = mz_max.max(c.mz_fit.mu);
        raw_sum += c.raw_sum;
    }
    Super {
        member_cids: group.to_vec(),
        rt_bounds: (rt_l, rt_r),
        im_bounds: (im_l, im_r),
        mz_center: if wsum>0.0 { mz_w/wsum } else { (mz_min+mz_max)*0.5 },
        raw_sum,
    }
}

fn is_isotopic_link(u:&Super, v:&Super, p:&GroupingParams) -> bool {
    // (rt/im as above)
    let dm = (u.mz_center - v.mz_center).abs();

    // same-peak duplicate path:
    let center = ((u.mz_center + v.mz_center) * 0.5).max(1e-6);
    let ppm_close = 1.0e6 * dm / center <= p.mz_ppm_tol;

    // isotopic path (allow one skipped isotope too)
    let mut iso_ok = false;
    for z in p.z_min..=p.z_max {
        let d1 = 1.003355f32 / (z as f32);
        let tol = (d1 * (p.iso_ppm_tol * 1e-6)).max(p.iso_abs_da);
        if (dm - d1).abs() <= tol || (dm - 2.0*d1).abs() <= tol {
            iso_ok = true; break;
        }
    }
    ppm_close || iso_ok
}
// --- Envelope construction ----------------------------------------------------

fn envelope_from_supers(eid: usize, supers: &[usize], super_nodes: &[Super], charge_hint: Option<u8>) -> Envelope {
    let mut rt_min = usize::MAX; let mut rt_max = 0;
    let mut im_min = usize::MAX; let mut im_max = 0;
    let mut mz_weighted = 0.0f64;
    let mut wsum = 0.0f64;
    let mut mz_lo = f32::MAX;
    let mut mz_hi = f32::MIN;
    let mut member_cids = Vec::new();

    for &sid in supers {
        let s = &super_nodes[sid];
        rt_min = rt_min.min(s.rt_bounds.0);
        rt_max = rt_max.max(s.rt_bounds.1);
        im_min = im_min.min(s.im_bounds.0);
        im_max = im_max.max(s.im_bounds.1);
        mz_lo = mz_lo.min(s.mz_center);
        mz_hi = mz_hi.max(s.mz_center);

        mz_weighted += (s.mz_center as f64) * (s.raw_sum as f64);
        wsum += s.raw_sum as f64;

        member_cids.extend_from_slice(&s.member_cids);
    }

    let mz_center = if wsum > 0.0 { (mz_weighted / wsum) as f32 } else { (mz_lo + mz_hi) * 0.5 };

    Envelope {
        id: eid,
        cluster_ids: member_cids,
        rt_bounds: (rt_min, rt_max),
        im_bounds: (im_min, im_max),
        mz_center,
        mz_span_da: mz_hi - mz_lo,
        charge_hint,
    }
}

// Greedy grower; note `p` (not `params`)
struct Grown { members: Vec<usize>, z: u8 }

fn grow_best_envelope(seed:usize, nodes:&[Super], adj:&[Vec<usize>],
                      p:&GroupingParams, taken:&[bool]) -> Grown {
    let mut best = Grown{members: vec![], z: 0};

    // consider seed + neighbors
    let mut cand: Vec<usize> = std::iter::once(seed)
        .chain(adj[seed].iter().copied())
        .filter(|&sid| !taken[sid])
        .collect();

    cand.sort_unstable_by(|&a,&b| nodes[a].mz_center.partial_cmp(&nodes[b].mz_center).unwrap());

    for z in p.z_min..=p.z_max {
        let target = 1.003355f32 / (z as f32);
        // ppm on Δm, with a small absolute floor (e.g., 20 mDa)
        let delta_abs = (target * (p.iso_ppm_tol * 1e-6)).max(0.02);

        let mut chain: Vec<usize> = vec![];
        for &sid in &cand {
            if chain.is_empty() {
                chain.push(sid);
            } else {
                let prev = *chain.last().unwrap();
                let dm = (nodes[sid].mz_center - nodes[prev].mz_center).abs();
                if (dm - target).abs() <= delta_abs {
                    chain.push(sid);
                }
            }
        }

        let sum_raw = chain.iter().map(|&sid| nodes[sid].raw_sum).sum::<f32>();
        let better = chain.len() > best.members.len()
            || (chain.len() == best.members.len()
            && sum_raw > best.members.iter().map(|&sid| nodes[sid].raw_sum).sum::<f32>());

        if better && chain.len() >= 2 {
            best = Grown { members: chain, z };
        }
    }
    best
}

// --- Main grouping (clean, single-pass envelope build) -----------------------

pub fn group_clusters_into_envelopes(
    clusters: &[ClusterResult],
    p: &GroupingParams,
) -> GroupingOutput {

    // Keep only clusters with a usable m/z fit & signal
    let good_ids: Vec<usize> = clusters.iter().enumerate()
        .filter(|(_, c)| c.raw_sum > 0.0 &&
            c.mz_fit.mu.is_finite() && c.mz_fit.mu > 50.0 &&
            c.mz_fit.sigma.is_finite() && c.mz_fit.sigma > 0.0)
        .map(|(i, _)| i)
        .collect();

    if good_ids.is_empty() {
        return GroupingOutput { envelopes: vec![], assignment: vec![None; clusters.len()], provisional: vec![] };
    }

    // Work on a compact Vec of “good” clusters
    let compact_vec: Vec<ClusterResult> = good_ids.iter().map(|&i| clusters[i].clone()).collect();
    let compact: &[ClusterResult] = &compact_vec;

    let m = compact.len();

    // A) DSU to merge near-duplicates
    let mut dsu = Dsu::new(m);
    for i in 0..m {
        for j in (i+1)..m {
            if is_near_duplicate(&compact[i], &compact[j]) {
                dsu.union(i, j);
            }
        }
    }

    // B) Super nodes
    let groups = dsu.groups(); // provisional (for inspection)
    let super_nodes: Vec<Super> = groups.iter().map(|g| summarize_super(g, compact)).collect();

    // C) Isotopic adjacency between supers
    let n = super_nodes.len();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for u in 0..n {
        for v in (u+1)..n {
            if is_isotopic_link(&super_nodes[u], &super_nodes[v], p) {
                adj[u].push(v);
                adj[v].push(u);
            }
        }
    }

    // D) Greedy seed order (strongest first)
    let mut seeds: Vec<usize> = (0..n).collect();
    seeds.sort_unstable_by(|&a,&b| super_nodes[b].raw_sum.partial_cmp(&super_nodes[a].raw_sum).unwrap());

    // E) Grow envelopes + assign
    let mut taken = vec![false; n];
    let mut envelopes: Vec<Envelope> = vec![];
    let mut assignment = vec![None; m];

    for seed in seeds {
        if taken[seed] { continue; }
        let best = grow_best_envelope(seed, &super_nodes, &adj, p, &taken);
        if best.members.is_empty() { continue; }

        let eid = envelopes.len();
        for &sid in &best.members { taken[sid] = true; }
        let env = envelope_from_supers(eid, &best.members, &super_nodes, Some(best.z));
        // write assignment for original clusters
        for &sid in &best.members {
            for &cid in &super_nodes[sid].member_cids {
                assignment[cid] = Some(eid);
            }
        }
        envelopes.push(env);
    }

    // F) Leftover singletons as one-member envelopes (optional, keeps total coverage)
    for sid in 0..n {
        if !taken[sid] {
            let eid = envelopes.len();
            let env = envelope_from_supers(eid, &[sid], &super_nodes, None);
            for &cid in &super_nodes[sid].member_cids {
                assignment[cid] = Some(eid);
            }
            envelopes.push(env);
        }
    }

    GroupingOutput { envelopes, assignment, provisional: groups }
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
    pub charge_hint: Option<u8>, // new
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