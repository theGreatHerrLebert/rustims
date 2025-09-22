use rayon::prelude::*;
use std::sync::Arc;
use mscore::timstof::frame::TimsFrame;
use mscore::algorithm::isotope::generate_averagine_spectra;
use crate::cluster::cluster_eval::ClusterResult;

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
                if member_ids.is_empty() {
                    return None; // or skip making a Feature
                }
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

            let shifts = [-1, 0, 1];
            let mut best_cos = 0.0f32;
            for &s in &shifts {
                best_cos = best_cos.max(cosine_aligned(&iso, &avg, lut.k, s));
            }
            if best_cos < fp.min_cosine { return None; }
            let cos = best_cos;

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
fn cosine_aligned(a: &[f32; 8], b: &[f32; 8], k: usize, shift: isize) -> f32 {
    // overlap: indices i with 0<=i<k and 0<=i+shift<k
    let k = k.min(8) as isize;
    let i0 = 0isize.max(-shift);
    let i1 = k.min(k - shift); // exclusive
    if i0 >= i1 { return 0.0; }

    let mut dot = 0f32;
    let mut na  = 0f32;
    let mut nb  = 0f32;

    for i in i0..i1 {
        let ii = i as usize;
        let jj = (i + shift) as usize;
        let xa = a[ii];
        let xb = b[jj];
        dot += xa * xb;
        na  += xa * xa;
        nb  += xb * xb;
    }

    if na == 0.0 || nb == 0.0 { return 0.0; }

    // Light coverage factor: proportion of peaks actually overlapped
    let overlap = (i1 - i0) as f32;
    let cov = overlap / (k as f32);
    cov * (dot / (na.sqrt() * nb.sqrt()))
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
    // optional: set to 0 to disable; else thin very large per-scan slices
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

    // Precompute isotope windows
    let mut lo_k = [0f32; 8];
    let mut hi_k = [0f32; 8];
    for k in 0..k_keep {
        let mz_k = mz_mono + (k as f32) * dmz;
        let d = mz_k * ppm_narrow * 1e-6;
        let (lo, hi) = (mz_k - d, mz_k + d);
        if lo.is_finite() && hi.is_finite() && hi > lo {
            lo_k[k] = lo;
            hi_k[k] = hi;
        } else {
            // mark empty window
            lo_k[k] = 1.0;
            hi_k[k] = 0.0;
        }
    }

    // Parallelize over frames if the RT span is large
    let rt_span = rt_r - rt_l + 1;
    let parallel = rt_span >= 256;

    let reducer = |range: std::ops::Range<usize>| -> [f32; 8] {
        let mut local = [0f32; 8];

        for f in range {
            let fr = &frames[f];
            let mz  = &fr.ims_frame.mz;         // &[f64]
            let it  = &fr.ims_frame.intensity;  // &[f32/f64]
            if mz.is_empty() { continue; }

            // Build (scan, start, end) slices once per frame
            let slices = build_scan_slices(fr);

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

                    // Optional thinning with compensation
                    let mut stride = 1usize;
                    if max_points_per_slice > 0 {
                        let len = r - l;
                        if len > max_points_per_slice {
                            stride = ((len + max_points_per_slice - 1) / max_points_per_slice).max(1);
                        }
                    }
                    let weight = if stride > 1 { stride as f32 } else { 1.0 };

                    let mut i = l;
                    while i < r {
                        // scv[i] == s within this slice; IM bounds already satisfied
                        local[k] += (it[i] as f32) * weight;   // ← compensate for thinning
                        i += stride;
                    }
                }
            }
        }

        local
    };

    let acc = if parallel {
        let chunks = 8usize;
        let chunk = (rt_span + chunks - 1) / chunks;
        (0..chunks)
            .into_par_iter()
            .map(|c| {
                let a = rt_l + c * chunk;
                let b = (a + chunk).min(rt_r + 1);
                if a >= b { [0f32; 8] } else { reducer(a..b) }
            })
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

// --- NEW: global (per-box) lattice + DP + packing ----------------------------

#[derive(Clone, Debug)]
struct Seed {
    z: u8,
    mz_mono: f32,   // m0
    delta: f32,     // 1.003355/z
}

/// A candidate feature built from one seed (before global packing)
#[derive(Clone, Debug)]
struct FeatureCand {
    seed: Seed,
    // assignment: cluster index -> Some(k) or None (kept only for assigned members)
    assigned: Vec<(usize /*cluster id*/, usize /*slot k*/)>,

    // bookkeeping
    score: f32,
    rt_bounds: (usize,usize),
    im_bounds: (usize,usize),
    mz_center: f32,
    // observed envelope (unnormalized) length K<=8
    #[allow(dead_code)]
    env_obs: [f32; 8],
    env_k: usize,
}

/// Per-pair score weights (tweak!)
const ALPHA_MZ: f32 = 1.0;
const ALPHA_RT: f32 = 0.6;
const ALPHA_IM: f32 = 0.6;
const BETA_GAP: f32 = 0.4;         // penalty per internal missing slot
const BETA_JUMP: f32 = 0.25;       // penalty for jumping over slots when assigning next
const BONUS_CONTIG: f32 = 0.15;    // small nudge to extend a contiguous chain (k = last+1)
const MAX_K: usize = 8;

// Smooth Gaussian on ppm (recommended)
#[inline]
fn gauss_ppm(ppm: f32, sigma_ppm: f32) -> f32 {
    (-0.5 * (ppm / sigma_ppm).powi(2)).exp()
}

#[inline]
fn jaccard_overlap(a: (usize,usize), b: (usize,usize)) -> f32 {
    let l = a.0.max(b.0);
    let r = a.1.min(b.1);
    if r < l { return 0.0; }
    let inter = (r - l + 1) as f32;
    let la = (a.1 - a.0 + 1) as f32;
    let lb = (b.1 - b.0 + 1) as f32;
    inter / (la + lb - inter).max(1.0)
}

#[inline]
fn pair_score(
    c: &ClusterResult,
    k: usize,
    seed: &Seed,
    feat_rt_bounds: (usize,usize),
    feat_im_bounds: (usize,usize),
) -> f32 {
    let m0 = seed.mz_mono;
    let mk = m0 + (k as f32) * seed.delta;
    let ppm = ppm_between(c.mz_fit.mu, mk).abs();

    // Choose one of the following two — Gaussian recommended
    // let s_mz = huber_ppm(ppm, 12.0);      // slightly looser than 8 ppm
    let s_mz = gauss_ppm(ppm, 12.0);         // smooth tolerance in ppm

    let s_rt = jaccard_overlap(c.rt_window, feat_rt_bounds);
    let s_im = jaccard_overlap(c.im_window, feat_im_bounds);

    // weight by log(1+intensity) to stabilize huge peaks
    let w = (1.0 + c.raw_sum).ln();

    w * (ALPHA_MZ * s_mz + ALPHA_RT * s_rt + ALPHA_IM * s_im)
}

/// Build basic feature RT/IM bounds from members used so far (or seed estimate).
#[inline]
fn union_bounds(rt:(usize,usize), im:(usize,usize), c:&ClusterResult) -> ((usize,usize),(usize,usize)) {
    ((rt.0.min(c.rt_window.0), rt.1.max(c.rt_window.1)),
     (im.0.min(c.im_window.0), im.1.max(c.im_window.1)))
}

/// From cluster μ’s inside box, propose a few (z, m0) seeds.
fn propose_seeds_for_box(
    box_ids: &[usize],
    clusters: &[ClusterResult],
    z_min: u8, z_max: u8,
) -> Vec<Seed> {
    if box_ids.len() < 2 { return Vec::new(); }

    // 1) z candidates from μ spacings
    let mut mzs: Vec<f32> = box_ids.iter().map(|&i| clusters[i].mz_fit.mu).collect();
    mzs.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let z_guess = infer_charge_from_members(&mzs, z_min, z_max);

    let mut z_list: Vec<u8> = Vec::new();
    if let Some(z) = z_guess { z_list.push(z); }
    // add neighbors if ambiguous or small box
    for z in z_min..=z_max {
        if z_list.len() >= 3 { break; }
        if !z_list.contains(&z) { z_list.push(z); }
    }

    // 2) m0 per z via LS on lattice offset (1–2 small iterations)
    let mut seeds = Vec::new();
    for &z in &z_list {
        let d = 1.003355f32 / (z as f32);
        // initial m0 near smallest μ
        let mut m0 = mzs[0];
        for _ in 0..2 {
            // estimate nearest slot for each μ, then re-fit m0 = mean(μ_i - j_i * d)
            let mut sum = 0.0f32; let mut cnt = 0usize;
            for &m in &mzs {
                let j = ((m - m0) / d).round() as i32;
                sum += m - (j as f32) * d; cnt += 1;
            }
            if cnt > 0 { m0 = sum / (cnt as f32); }
        }
        // add ±0.5Δ neighbors to cover off-by-one
        seeds.push(Seed{z, mz_mono:m0, delta:d});
        seeds.push(Seed{z, mz_mono:m0 - 0.5*d, delta:d});
        seeds.push(Seed{z, mz_mono:m0 + 0.5*d, delta:d});
    }

    // de-dup close m0 per z
    seeds.sort_by(|a,b| a.z.cmp(&b.z).then_with(|| a.mz_mono.partial_cmp(&b.mz_mono).unwrap()));
    let mut uniq = Vec::new();
    for s in seeds {
        if uniq.last().map_or(true, |p:&Seed| p.z!=s.z || (ppm_between(p.mz_mono,s.mz_mono) > 2.0)) {
            uniq.push(s);
        }
    }
    uniq
}

/// Compute a dynamic slot limit per seed based on the mass span of the box.
fn k_limit_for_seed(seed: &Seed, box_ids: &[usize], clusters: &[ClusterResult], k_max: usize) -> usize {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &cid in box_ids {
        let m = clusters[cid].mz_fit.mu;
        lo = lo.min(m); hi = hi.max(m);
    }
    if !lo.is_finite() || !hi.is_finite() || hi <= lo { return 2; }
    let span = hi - lo;
    let slots = 1 + (span / seed.delta).floor() as usize; // inclusive count
    slots.clamp(2, MAX_K.min(k_max.max(2)))
}

/// DP assignment for one seed: best chain with gaps penalized.
/// Returns FeatureCand with assignment and score.
fn dp_assign_for_seed(
    seed: &Seed,
    box_ids: &[usize],
    clusters: &[ClusterResult],
    k_max: usize,
    lut: &AveragineLut,
) -> FeatureCand {
    let k_lim = k_limit_for_seed(seed, box_ids, clusters, k_max);
    // sort by μ
    let mut ids = box_ids.to_vec();
    ids.sort_by(|&a,&b| clusters[a].mz_fit.mu.partial_cmp(&clusters[b].mz_fit.mu).unwrap());

    // precompute nearest lattice index for each cluster
    let mut nearest_k: Vec<usize> = Vec::with_capacity(ids.len());
    for &cid in &ids {
        let m = clusters[cid].mz_fit.mu;
        let j = ((m - seed.mz_mono) / seed.delta).round();
        let k = j.clamp(0.0, k_lim as f32 - 1.0) as usize;
        nearest_k.push(k);
    }

    // DP state: best score up to i with last used slot = k_used (or k_used = usize::MAX for none)
    let none = usize::MAX;
    let mut dp: Vec<Vec<f32>> = vec![vec![f32::NEG_INFINITY; k_lim+1]; ids.len()+1];
    let mut back: Vec<Vec<(usize,usize,bool)>> = vec![vec![(0,none,false); k_lim+1]; ids.len()+1];
    // base
    dp[0][k_lim] = 0.0; // k_lim stands for "none yet"

    // allow a slightly wider neighborhood around the nearest slot
    const DK_TRY: [isize; 5] = [-2, -1, 0, 1, 2];

    for i in 0..ids.len() {
        for prev in 0..=k_lim {
            let cur_best = dp[i][prev];
            if !cur_best.is_finite() { continue; }

            // 1) skip cluster i
            if cur_best > dp[i+1][prev] {
                dp[i+1][prev] = cur_best;
                back[i+1][prev] = (prev, none, false);
            }

            // 2) try assign cluster i to k in {k0+[-2..2]} if ≥ prev+1 (monotone)
            let cid = ids[i];
            let k0 = nearest_k[i] as isize;
            for &dk in &DK_TRY {
                let kk = k0 + dk;
                if kk < 0 || kk >= k_lim as isize { continue; }
                let k = kk as usize;

                // enforce monotone increasing slots (and one per slot)
                let last_slot = if prev==k_lim { none } else { prev };
                if last_slot != none && k <= last_slot { continue; }

                // gap penalty if we jump more than 1 slot
                let gap = if last_slot==none { 0 } else { k as isize - last_slot as isize - 1 };
                let gap_pen = if gap > 0 { (gap as f32) * BETA_JUMP } else { 0.0 };

                // small bonus for extending a contiguous chain
                let contig_bonus = if last_slot!=none && k == last_slot + 1 { BONUS_CONTIG } else { 0.0 };

                // pair score: use overlap vs the candidate's own windows as proxy
                let ps = pair_score(&clusters[cid], k, seed, clusters[cid].rt_window, clusters[cid].im_window);

                let cand = cur_best + ps - gap_pen + contig_bonus;
                if cand > dp[i+1][k] {
                    dp[i+1][k] = cand;
                    back[i+1][k] = (prev, cid, true);
                }
            }
        }
    }

    // best terminal
    let mut best_val = f32::NEG_INFINITY;
    let mut best_k = k_lim;
    for k in 0..=k_lim {
        if dp[ids.len()][k] > best_val { best_val = dp[ids.len()][k]; best_k = k; }
    }

    // reconstruct assignment, union bounds, env
    let mut assigned_rev: Vec<(usize,usize)> = Vec::new();
    let mut rtb = (usize::MAX, 0usize);
    let mut imb = (usize::MAX, 0usize);

    let mut i = ids.len();
    let mut kcur = best_k;
    while i > 0 {
        let (kprev, cid, took) = back[i][kcur];
        if took {
            // we assigned ids[i-1] == cid to slot = kcur
            assigned_rev.push((cid, kcur));
            let c = &clusters[cid];
            let u = union_bounds(rtb, imb, c);
            rtb = u.0; imb = u.1;
            kcur = kprev;
        } else {
            // skipped
            kcur = kprev;
        }
        i -= 1;
    }
    assigned_rev.reverse();

    if rtb.0 == usize::MAX { rtb = (0,0); }
    if imb.0 == usize::MAX { imb = (0,0); }

    // internal gap penalty on occupied slot range
    let mut occ = vec![false; k_lim];
    for &(_,k) in &assigned_rev { occ[k] = true; }
    let (mut kmin, mut kmax) = (usize::MAX, 0usize);
    for k in 0..k_lim { if occ[k] { kmin = kmin.min(k); kmax = kmax.max(k); } }
    let mut internal_gaps = 0usize;
    if kmin != usize::MAX && kmax > kmin {
        for k in (kmin+1)..kmax { if !occ[k] { internal_gaps += 1; } }
    }
    let gap_pen_total = (internal_gaps as f32) * BETA_GAP;

    // observed envelope vector from raw_sum aggregated per slot
    let mut env_obs = [0f32; 8];
    for &(cid,k) in &assigned_rev {
        env_obs[k.min(7)] += clusters[cid].raw_sum.max(0.0);
    }
    let env_k = k_lim;

    // shape gain vs averagine (with ±1 shift), weight by coverage
    let shape_gain = envelope_shape_gain(&env_obs, env_k, seed, &assigned_rev, clusters, lut);

    // final score
    let score = best_val - gap_pen_total + shape_gain;

    // center ~ average μ of assigned
    let mut mz_center = 0f32; let mut wsum = 0f32;
    for &(cid,_) in &assigned_rev {
        let c = &clusters[cid];
        let w = c.raw_sum.max(1.0);
        mz_center += w * c.mz_fit.mu; wsum += w;
    }
    if wsum > 0.0 { mz_center /= wsum; }

    FeatureCand {
        seed: seed.clone(),
        assigned: assigned_rev,
        score,
        rt_bounds: rtb,
        im_bounds: imb,
        mz_center,
        env_obs,
        env_k,
    }
}

#[inline]
fn envelope_shape_gain(
    env_obs: &[f32;8],
    k: usize,
    seed: &Seed,
    assigned: &[(usize,usize)],
    _clusters: &[ClusterResult],
    lut: &AveragineLut,
) -> f32 {
    if k == 0 { return 0.0; }
    // neutral mass from seed m0 and z
    let neutral = (seed.mz_mono - 1.007_276_466_88_f32) * (seed.z as f32);

    // expected (unit L2) from LUT
    let exp = lut.lookup(neutral, seed.z);
    // observed → L2 normalize (avoid all-zero)
    let mut obs = [0f32; 8];
    let mut nn = 0f32;
    for i in 0..k.min(8) { obs[i] = env_obs[i]; nn += obs[i]*obs[i]; }
    if nn <= 0.0 { return 0.0; }
    let s = nn.sqrt();
    for i in 0..k.min(8) { obs[i] /= s; }

    // allow small shift {-1,0,1}
    let shifts = [-1, 0, 1];
    let mut best = 0.0f32;
    for &sft in &shifts {
        best = best.max(cosine_aligned(&obs, &exp, k.min(8), sft));
    }

    // coverage factor: fraction of occupied slots inside [kmin..kmax]
    let mut occ = vec![false; k.min(8)];
    for &(_,kk) in assigned { if kk < occ.len() { occ[kk] = true; } }
    let mut kmin = usize::MAX; let mut kmax = 0usize; let mut used = 0usize;
    for i in 0..occ.len() { if occ[i] { kmin = kmin.min(i); kmax = kmax.max(i); used += 1; } }
    let cov = if kmin==usize::MAX { 0.0 } else { used as f32 / (kmax - kmin + 1) as f32 };

    // scale shape gain modestly to avoid dominating lattice score
    0.8f32 * best * cov
}

/// Greedy set packing across candidates: pick highest score, drop those that share any cluster.
fn pack_features_greedy(cands: &mut Vec<FeatureCand>) -> Vec<FeatureCand> {
    cands.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut picked = Vec::new();
    'next: for f in cands.iter() {
        for &(cid,_) in &f.assigned {
            if used.contains(&cid) { continue 'next; }
        }
        for &(cid,_) in &f.assigned { used.insert(cid); }
        picked.push(f.clone());
    }
    picked
}

// helper: connected components over adjacency
fn connected_components(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut seen = vec![false; n];
    let mut comps: Vec<Vec<usize>> = Vec::new();
    for s in 0..n {
        if seen[s] { continue; }
        let mut stack = vec![s];
        let mut comp = Vec::new();
        seen[s] = true;
        while let Some(u) = stack.pop() {
            comp.push(u);
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        comps.push(comp);
    }
    comps
}

/// Replace the greedy chain grower with global DP+packing inside each **isotopic** box.
/// Pipeline:
///   - DSU collapse near-duplicates -> super nodes
///   - build isotopic adjacency between supers (Δm ≈ 1.003355/z within tol)
///   - connected components on that graph = boxes
///   - map each box back to original cluster ids
///   - per-box: seeds -> DP -> greedy packing -> envelopes
pub fn group_clusters_into_envelopes_global(
    clusters: &[ClusterResult],
    p: &GroupingParams,
    lut: &AveragineLut,
    k_max: usize,
) -> GroupingOutput {
    // 0) keep only usable clusters (same filter as before)
    let good_ids: Vec<usize> = clusters.iter().enumerate()
        .filter(|(_, c)| c.raw_sum > 0.0 &&
            c.mz_fit.mu.is_finite() && c.mz_fit.mu > 50.0 &&
            c.mz_fit.sigma.is_finite() && c.mz_fit.sigma > 0.0)
        .map(|(i, _)| i)
        .collect();

    if good_ids.is_empty() {
        return GroupingOutput { envelopes: vec![], assignment: vec![None; clusters.len()], provisional: vec![] };
    }

    // A) Compact copy for fast work + DSU near-duplicate collapse
    let compact_vec: Vec<ClusterResult> = good_ids.iter().map(|&i| clusters[i].clone()).collect();
    let compact: &[ClusterResult] = &compact_vec;

    let dsu = make_neardup_groups(compact, p.rt_pad_overlap, p.im_pad_overlap);
    let near_groups = dsu.groups(); // groups are indices in "compact" space

    // B) Build "super nodes" by summarizing each near-duplicate group
    let super_nodes: Vec<Super> = near_groups.iter()
        .map(|g| summarize_super(g, compact, &good_ids))
        .collect();

    // Early out: if no supers, nothing to do
    if super_nodes.is_empty() {
        return GroupingOutput { envelopes: vec![], assignment: vec![None; clusters.len()], provisional: vec![] };
    }

    // C) Build isotopic adjacency between supers (uses your tolerance/charge grid)
    let adj: Vec<Vec<usize>> = iso_edges_sorted(&super_nodes, p);

    // D) Connected components over isotopic graph -> true "boxes"
    let super_boxes: Vec<Vec<usize>> = connected_components(&adj);

    let mut envelopes: Vec<Envelope> = Vec::new();
    let mut assignment = vec![None; clusters.len()];

    // E) For each isotopic box, collect original cluster ids, then run DP+packing
    for sbox in super_boxes {
        // Gather all original cluster ids inside this super-node component
        let mut box_orig: Vec<usize> = Vec::new();
        for &sid in &sbox {
            box_orig.extend_from_slice(&super_nodes[sid].member_cids);
        }
        // Deduplicate & sort for stability
        box_orig.sort_unstable();
        box_orig.dedup();

        if box_orig.is_empty() {
            continue;
        }

        // 1) seeds from the box members
        let seeds = propose_seeds_for_box(&box_orig, clusters, p.z_min, p.z_max);
        if seeds.is_empty() {
            // fallback: singletons (retain coverage)
            for &cid in &box_orig {
                if assignment[cid].is_some() { continue; }
                let eid = envelopes.len();
                envelopes.push(Envelope{
                    id: eid,
                    cluster_ids: vec![cid],
                    rt_bounds: clusters[cid].rt_window,
                    im_bounds: clusters[cid].im_window,
                    mz_center: clusters[cid].mz_fit.mu,
                    mz_span_da: 0.0,
                    charge_hint: None,
                });
                assignment[cid] = Some(eid);
            }
            continue;
        }

        // 2) per-seed DP candidates
        let mut cands: Vec<FeatureCand> = seeds.iter()
            .map(|s| dp_assign_for_seed(s, &box_orig, clusters, k_max, lut))
            .filter(|f| !f.assigned.is_empty() && f.score.is_finite())
            .collect();

        if cands.is_empty() {
            // fallback as above
            for &cid in &box_orig {
                if assignment[cid].is_some() { continue; }
                let eid = envelopes.len();
                envelopes.push(Envelope{
                    id: eid,
                    cluster_ids: vec![cid],
                    rt_bounds: clusters[cid].rt_window,
                    im_bounds: clusters[cid].im_window,
                    mz_center: clusters[cid].mz_fit.mu,
                    mz_span_da: 0.0,
                    charge_hint: None,
                });
                assignment[cid] = Some(eid);
            }
            continue;
        }

        // 3) Greedy set packing across candidates (avoid reusing the same cluster)
        let chosen = pack_features_greedy(&mut cands);

        // 4) Emit envelopes & write assignments
        for feat in chosen {
            let eid = envelopes.len();
            let mut member_ids: Vec<usize> = feat.assigned.iter().map(|(cid,_)| *cid).collect();
            member_ids.sort_unstable();
            member_ids.dedup();

            let mz_lo = feat.seed.mz_mono;
            let mz_hi = feat.seed.mz_mono + (feat.env_k.saturating_sub(1) as f32) * feat.seed.delta;

            envelopes.push(Envelope{
                id: eid,
                cluster_ids: member_ids.clone(),
                rt_bounds: feat.rt_bounds,
                im_bounds: feat.im_bounds,
                mz_center: feat.mz_center,
                mz_span_da: (mz_hi - mz_lo).abs(),
                charge_hint: Some(feat.seed.z),
            });

            for cid in member_ids {
                assignment[cid] = Some(eid);
            }
        }
    }

    GroupingOutput { envelopes, assignment, provisional: near_groups }
}