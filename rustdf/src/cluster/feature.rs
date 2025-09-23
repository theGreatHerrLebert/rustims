use rayon::prelude::*;
use std::sync::Arc;
use mscore::timstof::frame::TimsFrame;
use mscore::algorithm::isotope::generate_averagine_spectra;
use crate::cluster::cluster_eval::ClusterResult;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

static SLICES_CACHE: OnceLock<Mutex<HashMap<usize, Arc<Vec<ScanSlice>>>>> = OnceLock::new();

#[inline]
fn cached_scan_slices(fr: &Arc<TimsFrame>) -> Arc<Vec<ScanSlice>> {
    let key = Arc::as_ptr(fr) as usize;
    let map = SLICES_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(v) = map.lock().unwrap().get(&key) {
        return v.clone();
    }
    let built = Arc::new(build_scan_slices(fr));
    map.lock().unwrap().insert(key, built.clone());
    built
}

#[derive(Copy, Clone, Debug)]
struct ScanSlice {
    scan: usize,   // absolute scan index
    start: usize,  // inclusive
    end: usize,    // exclusive
}

// Build contiguous (scan, start, end) slices. Assumes points are sorted by scan first.
#[inline]
fn build_scan_slices(fr: &TimsFrame) -> Vec<ScanSlice> {
    let scv = &fr.scan;           // &[i32]
    let mut out: Vec<ScanSlice> = Vec::new();
    if scv.is_empty() { return out; }

    let mut s_cur = scv[0];
    let mut i_start = 0usize;

    for i in 1..scv.len() {
        if scv[i] != s_cur {
            if s_cur >= 0 {
                out.push(ScanSlice { scan: s_cur as usize, start: i_start, end: i });
            }
            s_cur = scv[i];
            i_start = i;
        }
    }
    if s_cur >= 0 {
        out.push(ScanSlice { scan: s_cur as usize, start: i_start, end: scv.len() });
    }
    out
}

#[inline]
fn lower_bound_in(mz: &[f64], start: usize, end: usize, x: f32) -> usize {
    let mut lo = start;
    let mut hi = end;
    let xf = x as f64;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if mz[mid] < xf { lo = mid + 1; } else { hi = mid; }
    }
    lo
}

#[inline]
fn upper_bound_in(mz: &[f64], start: usize, end: usize, x: f32) -> usize {
    let mut lo = start;
    let mut hi = end;
    let xf = x as f64;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if mz[mid] <= xf { lo = mid + 1; } else { hi = mid; }
    }
    lo
}

#[inline]
fn estimate_charge_from_env_hist(
    frames: &[Arc<TimsFrame>],
    rt_bounds: (usize, usize),
    im_bounds: (usize, usize),
    mz_mono_hint: f32,
    win_ppm: f32,     // e.g. 20.0
    bins: usize,      // e.g. 21
) -> Option<(u8, f32)> {
    let (axis, hist) = build_local_mz_histogram(frames, rt_bounds, im_bounds, mz_mono_hint, win_ppm, bins);
    if axis.is_empty() { return None; }
    estimate_charge_from_hist(&axis, &hist)
}

// ---- add to FeatureBuildParams ----
#[derive(Clone, Debug)]
pub struct FeatureBuildParams {
    pub k_max: usize,
    pub ppm_narrow: f32,
    pub min_members: usize,
    pub min_cosine: f32,
    pub max_points_per_slice: usize,
    pub min_hist_conf: f32,
    pub allow_unknown_charge: bool,
}

#[derive(Clone, Debug)]
pub struct Feature {
    pub envelope_id: usize,
    pub charge: u8,
    pub mz_mono: f32,
    pub neutral_mass: f32,
    pub rt_bounds: (usize, usize),
    pub im_bounds: (usize, usize),
    pub mz_center: f32,
    pub n_members: usize,
    pub iso: [f32; 8],           // L2-normalized observed
    pub cos_averagine: f32,
    pub raw_sum: f32,
    pub member_cluster_ids: Vec<usize>,
    pub repr_cluster_id: usize,
}

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
fn infer_charge_from_members(mzs: &[f32], z_min:u8, z_max:u8) -> Option<u8> {
    if mzs.len() < 2 { return None; }
    let mut m = mzs.to_vec();
    m.sort_by(|a,b| a.partial_cmp(b).unwrap());
    // collect neighbor deltas
    let mut deltas = Vec::with_capacity(m.len().saturating_sub(1));
    for w in m.windows(2) {
        let dm = (w[1]-w[0]).abs();
        if dm > 0.0 { deltas.push(dm); }
    }
    if deltas.is_empty() { return None; }
    let mut best = (0u8, f32::MAX);
    for z in z_min..=z_max {
        let t = 1.003355f32 / (z as f32);
        let err = deltas.iter().map(|&d| (d - t).abs()).sum::<f32>() / (deltas.len() as f32);
        if err < best.1 { best = (z, err); }
    }
    if best.0 == 0 { None } else { Some(best.0) }
}

pub fn build_features_from_envelopes(
    frames: &[Arc<TimsFrame>],          // RT-sorted, preloaded
    envelopes: &[Envelope],
    clusters: &[ClusterResult],         // to pull members’ mz/raw_sum
    lut: &AveragineLut,
    gp: &GroupingParams,                // for z range
    fp: &FeatureBuildParams,            // now includes min_hist_conf, allow_unknown_charge
) -> Vec<Feature> {
    const PROTON: f32 = 1.007_276_466_88_f32;

    let k_keep = fp.k_max.min(8);

    envelopes
        .par_iter()
        .filter_map(|env| {
            // ---- 1) robust mono seed from members (min μ over members) ----
            let mut mz_mono = f32::INFINITY;
            for &cid in &env.cluster_ids {
                let m = clusters[cid].mz_fit.mu;
                if m.is_finite() { mz_mono = mz_mono.min(m); }
            }
            if !mz_mono.is_finite() || mz_mono <= 50.0 {
                return None;
            }

            // ---- 2) charge inference (hint -> members -> histogram -> unknown) ----
            let z_opt: Option<u8> = if let Some(z) = env.charge_hint {
                Some(z)
            } else {
                // 2a) neighbor-delta from member μ's if we have enough members
                let mut z_from_members = None;
                if env.cluster_ids.len() >= fp.min_members.max(2) {
                    let mut mzs: Vec<f32> = Vec::with_capacity(env.cluster_ids.len());
                    for &cid in &env.cluster_ids {
                        let m = clusters[cid].mz_fit.mu;
                        if m.is_finite() { mzs.push(m); }
                    }
                    z_from_members = infer_charge_from_members(&mzs, gp.z_min, gp.z_max);
                }

                // 2b) fallback: m/z histogram spacing around mono seed within RT×IM
                if z_from_members.is_none() {
                    // pick a conservative window for spacing detection:
                    // widen a bit beyond the integration ppm, but clamp to a sane range
                    let win_ppm = (fp.ppm_narrow * 2.0).clamp(12.0, 40.0);
                    if let Some((zh, conf)) = estimate_charge_from_env_hist(
                        frames, env.rt_bounds, env.im_bounds, mz_mono, win_ppm, 21
                    ) {
                        if conf >= fp.min_hist_conf { Some(zh) } else { None }
                    } else {
                        None
                    }
                } else {
                    z_from_members
                }
            };

            // Decide how to proceed
            let z: u8 = match z_opt {
                Some(zz) if zz > 0 => zz,
                _ => {
                    if fp.allow_unknown_charge {
                        0 // keep feature with unknown charge; integrate mono only
                    } else {
                        return None; // drop if we require a known charge
                    }
                }
            };

            // ---- 3) integrate isotope stripes ----
            if z == 0 {
                // unknown charge: integrate mono only with a neutral “z=1” spacing (k=1)
                let iso_raw = integrate_isotope_series(
                    frames, env.rt_bounds, env.im_bounds,
                    mz_mono, /*z_for_window*/ 1, fp.ppm_narrow, 1,
                    fp.max_points_per_slice,
                );
                let mut iso = [0f32; 8];
                iso[0] = iso_raw[0];

                let mut raw_sum = 0f32;
                for &cid in &env.cluster_ids { raw_sum += clusters[cid].raw_sum; }

                let member_ids = env.cluster_ids.clone();

                // pick a representative cluster id (max raw_sum among members)
                let repr_cluster_id = member_ids
                    .iter()
                    .copied()
                    .max_by(|&a, &b| {
                        clusters[a].raw_sum
                            .partial_cmp(&clusters[b].raw_sum)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    // safe fallback: use first member if something is NaN/empty
                    .unwrap_or_else(|| member_ids[0]);

                return Some(Feature{
                    envelope_id: env.id,
                    charge: 0,
                    mz_mono,
                    neutral_mass: f32::NAN,
                    rt_bounds: env.rt_bounds,
                    im_bounds: env.im_bounds,
                    mz_center: env.mz_center,
                    n_members: env.cluster_ids.len(),
                    iso,
                    cos_averagine: f32::NAN,
                    raw_sum,
                    member_cluster_ids: member_ids,
                    repr_cluster_id,
                });
            }

            // known charge → integrate k peaks using that z
            let iso_raw = integrate_isotope_series(
                frames, env.rt_bounds, env.im_bounds,
                mz_mono, z, fp.ppm_narrow, k_keep,
                fp.max_points_per_slice,
            );

            // L2 normalize
            let mut iso = [0f32; 8];
            let mut norm = 0f32;
            for i in 0..k_keep { iso[i] = iso_raw[i]; norm += iso[i]*iso[i]; }
            if norm > 0.0 {
                let s = norm.sqrt();
                for x in &mut iso { *x /= s; }
            }

            // ---- 4) averagine cosine gate ----
            let neutral = (mz_mono - PROTON) * (z as f32);
            let avg = lut.lookup(neutral, z);
            let cos = cosine(&iso, &avg);
            if cos < fp.min_cosine { return None; }

            // ---- 5) aggregate raw_sum over members ----
            let mut raw_sum = 0f32;
            for &cid in &env.cluster_ids {
                raw_sum += clusters[cid].raw_sum;
            }

            Some(Feature{
                envelope_id: env.id,
                charge: z,
                mz_mono,
                neutral_mass: neutral,
                rt_bounds: env.rt_bounds,
                im_bounds: env.im_bounds,
                mz_center: env.mz_center,
                n_members: env.cluster_ids.len(),
                iso,
                cos_averagine: cos,
                raw_sum,
                member_cluster_ids: env.cluster_ids.clone(),
                repr_cluster_id: {
                    let member_ids = &env.cluster_ids;
                    // pick a representative cluster id (max raw_sum among members)
                    member_ids
                        .iter()
                        .copied()
                        .max_by(|&a, &b| {
                            clusters[a].raw_sum
                                .partial_cmp(&clusters[b].raw_sum)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        // safe fallback: use first member if something is NaN/empty
                        .unwrap_or_else(|| member_ids[0])
                },
            })
        })
        .collect()
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

// ---- Fast bucketing for near-duplicate DSU ---------------------------------

#[inline]
fn logppm_bin(mz: f32, ppm: f32) -> i32 {
    // Bin in log space so ppm distances become translationally invariant.
    let step = (1.0 + ppm * 1e-6).ln();
    (mz.ln() / step).floor() as i32
}

/// Build a DSU of near-duplicates in ~O(n), by bucketing in (RT, IM, m/z).
/// Uses your existing `is_near_duplicate` for exact checks.
fn make_neardup_groups(cl: &[ClusterResult], rt_pad: usize, im_pad: usize) -> Dsu {
    use std::collections::HashMap;

    // Coarse bins ~ half the overlap pad (>=1)
    let rt_bin = (rt_pad.max(1) as i32 / 2).max(1);
    let im_bin = (im_pad.max(1) as i32 / 2).max(1);
    let ppm_bin = 3.0_f32; // match `is_near_duplicate`'s 3 ppm gate

    // Bucket: (br, bi, bz) -> Vec<idx>
    let mut buckets: HashMap<(i32,i32,i32), Vec<usize>> = HashMap::new();
    for (i, c) in cl.iter().enumerate() {
        let br = (c.rt_fit.mu as i32) / rt_bin;
        let bi = (c.im_fit.mu as i32) / im_bin;
        let bz = logppm_bin(c.mz_fit.mu, ppm_bin);
        buckets.entry((br, bi, bz)).or_default().push(i);
    }

    // DSU and neighbor probing
    let mut dsu = Dsu::new(cl.len());

    // To avoid double work when probing neighbors,
    // iterate keys into a Vec so we can probe exact neighbor coords.
    let keys: Vec<(i32,i32,i32)> = buckets.keys().copied().collect();

    for &(br, bi, bz) in &keys {
        let ids = &buckets[&(br, bi, bz)];

        // 1) Within the same cell (small K)
        for a in 0..ids.len() {
            for b in (a+1)..ids.len() {
                let i = ids[a]; let j = ids[b];
                if is_near_duplicate(&cl[i], &cl[j]) { dsu.union(i, j); }
            }
        }

        // 2) Neighbor cells (3x3x3 – only forward to avoid double visiting)
        for dr in 0..=1 {
            for di in -1..=1 {
                for dz in -1..=1 {
                    if dr==0 && di<=0 && dz<=0 { continue; } // forward-only
                    let key = (br+dr, bi+di, bz+dz);
                    if let Some(nids) = buckets.get(&key) {
                        for &i in ids {
                            for &j in nids {
                                let (a,b) = if i<j { (i,j) } else { (j,i) };
                                if is_near_duplicate(&cl[a], &cl[b]) { dsu.union(a, b); }
                            }
                        }
                    }
                }
            }
        }
    }

    dsu
}

// ---- Isotopic adjacency in ~O(n) per charge (sorted two-pointer) -----------

fn two_pointer_join(idx:&[usize], nodes:&[Super], delta:f32, tol:f32,
                    rt_pad:usize, im_pad:usize, adj:&mut [Vec<usize>]) {
    let m: Vec<f32> = idx.iter().map(|&i| nodes[i].mz_center).collect();
    let mut j = 0usize;

    for i in 0..idx.len() {
        let mi = m[i];
        let mut lo = mi + delta - tol;
        if lo < 0.0 { lo = 0.0; }
        while j < idx.len() && m[j] < lo { j += 1; }

        let mut k = j;
        let hi = mi + delta + tol;
        while k < idx.len() && m[k] <= hi {
            let u = idx[i];
            let v = idx[k];
            if u != v {
                // Require some RT/IM co-location (with padding), like your link fn:
                let rt_ok = overlap1d(expand(nodes[u].rt_bounds, rt_pad),
                                      expand(nodes[v].rt_bounds, rt_pad)) > 0;
                let im_ok = overlap1d(expand(nodes[u].im_bounds, im_pad),
                                      expand(nodes[v].im_bounds, im_pad)) > 0;
                if rt_ok && im_ok {
                    adj[u].push(v);
                    adj[v].push(u);
                }
            }
            k += 1;
        }
    }
}

fn iso_edges_sorted(nodes: &[Super], p: &GroupingParams) -> Vec<Vec<usize>> {
    let mut idx: Vec<usize> = (0..nodes.len()).collect();
    idx.sort_unstable_by(|&a,&b| nodes[a].mz_center.partial_cmp(&nodes[b].mz_center).unwrap());

    let mut adj = vec![Vec::<usize>::new(); nodes.len()];

    for z in p.z_min..=p.z_max {
        let d1  = 1.003355f32 / (z as f32);
        let tol = (d1 * (p.iso_ppm_tol * 1e-6)).max(p.iso_abs_da);

        // first-neighbor isotope
        two_pointer_join(&idx, nodes, d1,      tol, p.rt_pad_overlap, p.im_pad_overlap, &mut adj);
        // optionally allow skip-one isotope too (kept from your logic)
        two_pointer_join(&idx, nodes, 2.0*d1,  tol, p.rt_pad_overlap, p.im_pad_overlap, &mut adj);
    }

    adj
}

// --- Super nodes --------------------------------------------------------------

struct Super {
    member_cids: Vec<usize>,
    rt_bounds: (usize,usize),
    im_bounds: (usize,usize),
    mz_center: f32,     // raw_sum-weighted
    raw_sum: f32,
}

fn summarize_super(group: &[usize], compact: &[ClusterResult], good_ids: &[usize]) -> Super {
    let mut rt_l = usize::MAX; let mut rt_r = 0usize;
    let mut im_l = usize::MAX; let mut im_r = 0usize;
    let mut wsum = 0f32; let mut mz_w = 0f32;
    let mut mz_min = f32::INFINITY; let mut mz_max = f32::NEG_INFINITY;
    let mut raw_sum = 0f32;

    for &cid_c in group {
        let c = &compact[cid_c];           // read metrics from compact
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

    // store original ids
    let member_cids: Vec<usize> = group.iter().map(|&cid_c| good_ids[cid_c]).collect();

    Super {
        member_cids,
        rt_bounds: (rt_l, rt_r),
        im_bounds: (im_l, im_r),
        mz_center: if wsum>0.0 { mz_w/wsum } else { (mz_min+mz_max)*0.5 },
        raw_sum,
    }
}

#[inline]
fn expand(w:(usize,usize), pad:usize) -> (usize,usize) {
    (w.0.saturating_sub(pad), w.1.saturating_add(pad))
}

#[inline]
fn overlap1d(a:(usize,usize), b:(usize,usize)) -> usize {
    if a.1 < b.0 || b.1 < a.0 { 0 } else { a.1.min(b.1) - a.0.max(b.0) + 1 }
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
        let d1  = 1.003355f32 / (z as f32);
        let tol = (d1 * (p.iso_ppm_tol * 1e-6)).max(p.iso_abs_da);

        let mut chain: Vec<usize> = vec![];
        for &sid in &cand {
            if chain.is_empty() {
                chain.push(sid);
            } else {
                let prev = *chain.last().unwrap();
                let dm = (nodes[sid].mz_center - nodes[prev].mz_center).abs();
                if (dm - d1).abs() <= tol || (dm - 2.0*d1).abs() <= tol {
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

    let dsu = make_neardup_groups(compact, p.rt_pad_overlap, p.im_pad_overlap);

    // B) Super nodes
    let groups = dsu.groups(); // provisional (for inspection)
    let super_nodes: Vec<Super> = groups.iter().map(|g| summarize_super(g, compact, &good_ids)).collect();

    // C) Isotopic adjacency between supers
    let n = super_nodes.len();
    let adj: Vec<Vec<usize>> = iso_edges_sorted(&super_nodes, p);

    // D) Greedy seed order (strongest first)
    let mut seeds: Vec<usize> = (0..n).collect();
    seeds.sort_unstable_by(|&a,&b| super_nodes[b].raw_sum.partial_cmp(&super_nodes[a].raw_sum).unwrap());

    // E) Grow envelopes + assign
    let mut taken = vec![false; n];
    let mut envelopes: Vec<Envelope> = vec![];
    let mut assignment = vec![None; clusters.len()];

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
                let norm = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt() as f32;
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
pub fn integrate_isotope_series(
    frames: &[Arc<TimsFrame>],
    rt_bounds: (usize, usize),
    im_bounds: (usize, usize),
    mz_mono: f32,
    z: u8,
    ppm_narrow: f32,
    k_max: usize,
    max_points_per_slice: usize,
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

    // Precompute isotope windows (μ ± ppm)
    let mut lo_k = [0f32; 8];
    let mut hi_k = [0f32; 8];
    for k in 0..k_keep {
        let mz_k = mz_mono + (k as f32) * dmz;
        let d = mz_k * ppm_narrow * 1e-6;
        lo_k[k] = mz_k - d;
        hi_k[k] = mz_k + d;
        if !lo_k[k].is_finite() || !hi_k[k].is_finite() || hi_k[k] <= lo_k[k] {
            lo_k[k] = 1.0;
            hi_k[k] = 0.0; // mark empty
        }
    }

    let rt_span = rt_r - rt_l + 1;
    let parallel = rt_span >= 256;

    // Reducer over an RT range. Uses cached scan slices + thinning compensation.
    let reducer = |range: std::ops::Range<usize>| -> [f32; 8] {
        let mut local = [0f32; 8];

        for f in range {
            let fr = &frames[f];
            let mz  = &fr.ims_frame.mz;         // &[f64]
            let it  = &fr.ims_frame.intensity;  // &[f32/f64]
            if mz.is_empty() { continue; }

            let slices = cached_scan_slices(fr);

            // iterate only scans in [im_l, im_r]
            for sl in slices.iter() {
                let s = sl.scan;
                if s < im_l || s > im_r { continue; }

                let start = sl.start;
                let end   = sl.end;

                // For each isotope window, do bounds inside [start, end)
                for k in 0..k_keep {
                    if hi_k[k] <= lo_k[k] { continue; }

                    let l = lower_bound_in(mz, start, end, lo_k[k]);
                    let r = upper_bound_in(mz, start, end, hi_k[k]);
                    if l >= r { continue; }

                    // Optional thinning with exact compensation
                    let mut stride = 1usize;
                    if max_points_per_slice > 0 {
                        let len = r - l;
                        if len > max_points_per_slice {
                            stride = ((len + max_points_per_slice - 1) / max_points_per_slice).max(1);
                        }
                    }
                    let weight = stride as f32;

                    let mut i = l;
                    while i < r {
                        // scan index matches within this slice; no re-check needed
                        local[k] += (it[i] as f32) * weight;
                        i += stride;
                    }
                }
            }
        }

        local
    };

    let acc = if parallel {
        use rayon::prelude::*;
        // Parallelize naturally over frame indices with Rayon; balanced even for awkward spans.
        (rt_l..=rt_r)
            .into_par_iter()
            .map(|f| reducer(f..(f+1)))
            .reduce(|| [0f32; 8], |mut a, b| { for k in 0..8 { a[k] += b[k]; } a })
    } else {
        reducer(rt_l..(rt_r + 1))
    };

    for k in 0..k_keep { isotopes[k] = acc[k]; }
    isotopes
}

/// Build a small m/z histogram around `mz_center` within the cluster's RT×IM
/// bounds. Returns (mz_axis_centers, hist_y). `bins` ~ 15–31 is plenty.
///
/// `win_ppm` is the total half-window (+/− ppm) around `mz_center`.
#[inline]
pub fn build_local_mz_histogram(
    frames: &[Arc<TimsFrame>], // RT-sorted, preloaded
    rt_bounds: (usize, usize),            // inclusive frame indices
    im_bounds: (usize, usize),            // inclusive absolute scan indices
    mz_center: f32,
    win_ppm: f32,                         // e.g. 20.0
    bins: usize,                          // e.g. 21
) -> (Vec<f32>, Vec<f32>) {
    let bins = bins.max(10);
    let d_da = mz_center * win_ppm * 1e-6;
    let lo = mz_center - d_da;
    let hi = mz_center + d_da;
    if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
        return (Vec::new(), Vec::new());
    }

    // Axis centers
    let width = (hi - lo) / (bins as f32);
    let mut axis = Vec::with_capacity(bins);
    for b in 0..bins {
        axis.push(lo + (b as f32 + 0.5) * width);
    }

    let n = frames.len();
    let rt_l = rt_bounds.0.min(n.saturating_sub(1));
    let rt_r = rt_bounds.1.min(n.saturating_sub(1));
    if rt_l > rt_r { return (axis, vec![0.0; bins]); }

    let (im_l, im_r) = im_bounds;
    if im_l > im_r { return (axis, vec![0.0; bins]); }

    // Accumulate using per-scan slices and per-slice binary bounds.
    let mut y = vec![0.0f32; bins];

    // Cap per-slice points for speed (compensated by stride).
    const MAX_POINTS_PER_SLICE: usize = 10_000;

    for f in rt_l..=rt_r {
        let fr = &frames[f];
        let mz  = &fr.ims_frame.mz;        // &[f64]
        let it  = &fr.ims_frame.intensity; // &[f32/f64]
        if mz.is_empty() { continue; }

        let slices = cached_scan_slices(fr);

        for sl in slices.iter() {
            let s = sl.scan;
            if s < im_l || s > im_r { continue; }

            // Bound to [lo, hi] in this slice
            let l = lower_bound_in(mz, sl.start, sl.end, lo);
            let r = upper_bound_in(mz, sl.start, sl.end, hi);
            if l >= r { continue; }

            let mut stride = 1usize;
            let len = r - l;
            if len > MAX_POINTS_PER_SLICE {
                stride = ((len + MAX_POINTS_PER_SLICE - 1) / MAX_POINTS_PER_SLICE).max(1);
            }
            let weight = stride as f32;

            let mut i = l;
            while i < r {
                let m = mz[i] as f32;
                // bin m (lo..hi) into 0..bins-1
                let mut b = ((m - lo) / (hi - lo) * (bins as f32)).floor() as isize;
                if b < 0 { b = 0; }
                if (b as usize) >= bins { b = (bins as isize) - 1; }
                y[b as usize] += (it[i] as f32) * weight;
                i += stride;
            }
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