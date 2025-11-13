
/*
use crate::cluster::cluster::ClusterResult1D;


#[derive(Clone, Debug)]
pub struct AveragineLut {
    pub masses: Vec<f32>,      // neutral-mass grid (Da)
    pub z_min: u8,
    pub z_max: u8,
    pub k: usize,              // kept peaks (<=8), zero-padded to 8
    pub envs: Vec<[f32; 8]>,   // flattened by (z, mass_index)
}

impl AveragineLut {
    #[inline]
    fn clamp_resolution_decimals(resolution: i32) -> i32 {
        // Preserve “decimals” meaning from the old impl; 0..=6 is already plenty.
        resolution.clamp(0, 6)
    }

    #[inline]
    fn clamp_threads(n: usize) -> usize {
        n.clamp(1, 32)
    }

    pub fn build(
        mass_min: f32,
        mass_max: f32,
        step: f32,        // e.g. 25–50 Da
        z_min: u8,
        z_max: u8,
        k: usize,         // keep first k peaks (<=8)
        resolution: i32,  // interpreted as *decimals* like the old code
        num_threads: usize,
    ) -> Self {
        // ---- grid & parameter guards -------------------------------------------------
        let mass_min = mass_min.max(50.0);
        let mass_max = mass_max.max(mass_min + 1.0);
        let step     = step.max(1.0);
        let z_min    = z_min.max(1);
        let z_max    = z_max.max(z_min);
        let k        = k.clamp(1, 8);

        let mut masses: Vec<f32> = Vec::new();
        let mut m = mass_min;
        while m <= mass_max + 1e-6 {
            masses.push(m);
            m += step;
        }

        // If someone asks for a pathological grid, refuse early.
        const MAX_GRID_POINTS: usize = 200_000; // generous hard-stop
        if masses.len() > MAX_GRID_POINTS {
            panic!("AveragineLut grid too large: {} points (> {})", masses.len(), MAX_GRID_POINTS);
        }

        // ---- prepare storage ---------------------------------------------------------
        let per_z = masses.len();
        let n_env = per_z * (z_max - z_min + 1) as usize;
        let mut envs: Vec<[f32; 8]> = Vec::with_capacity(n_env);

        // ---- clamp heavy knobs -------------------------------------------------------
        let res_dec = Self::clamp_resolution_decimals(resolution);
        let threads = Self::clamp_threads(num_threads);

        // ---- CHUNKED generation to bound memory -------------------------------------
        // We never build all spectra at once; do it in slices per charge.
        const CHUNK: usize = 512;

        // Reusable scratch buffers to avoid re-allocs in the loop.
        let mut masses_f64: Vec<f64> = Vec::with_capacity(CHUNK);
        let mut charges:     Vec<i32> = Vec::with_capacity(CHUNK);

        for z in z_min..=z_max {
            let zi = z as i32;

            let mut start = 0;
            while start < masses.len() {
                let end = (start + CHUNK).min(masses.len());

                // fill scratch
                masses_f64.clear();
                masses_f64.extend(masses[start..end].iter().map(|&x| x as f64));

                charges.clear();
                charges.resize(end - start, zi);

                // generate only for this chunk
                let specs = generate_averagine_spectra(
                    masses_f64.clone(), // (the API takes owned Vecs)
                    charges.clone(),
                    /*min_intensity*/ 1,
                    /*k*/ k as i32,
                    /*resolution(decimals)*/ res_dec,
                    /*centroid*/ true,
                    /*threads*/  threads,
                    /*amp*/ None,
                );

                // compact & normalize to unit-length k vector, zero-padded to 8
                for sp in specs {
                    let mut v = [0f32; 8];
                    let keep = k.min(sp.intensity.len());
                    for i in 0..keep {
                        v[i] = sp.intensity[i] as f32;
                    }
                    // L2 normalize
                    let norm = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32;
                    if norm > 0.0 {
                        for x in &mut v { *x /= norm; }
                    }
                    envs.push(v);
                }

                start = end;
            }
        }

        Self { masses, z_min, z_max, k, envs }
    }

    #[inline]
    pub fn lookup(&self, neutral_mass: f32, z: u8) -> [f32; 8] {
        if z < self.z_min || z > self.z_max || self.masses.is_empty() {
            return [0.0; 8];
        }
        let zi = (z - self.z_min) as usize;
        let per_z = self.masses.len();
        // nearest-neighbor on mass grid
        let i = match self.masses.binary_search_by(|m| m.partial_cmp(&neutral_mass).unwrap_or(Ordering::Equal)) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1).min(per_z.saturating_sub(1)),
        };
        self.envs[zi * per_z + i]
    }
}


#[derive(Clone, Debug)]
pub struct GroupingParams {
    pub rt_pad_overlap: usize,   // pad windows for edge gating
    pub im_pad_overlap: usize,
    pub mz_ppm_tol: f32,         // tight (~3–6 ppm) for near-dup merge
    pub iso_ppm_tol: f32,        // 8–12 ppm for isotopic spacing
    pub iso_abs_da: f32,         // 0.002–0.005 Da safety floor
    pub z_min: u8,               // 1
    pub z_max: u8,               // 6
}

#[derive(Clone, Debug)]
pub struct FeatureBuildParams {
    pub k_max: usize,            // 3–6 typical
    pub min_members: usize,      // ≥2 isotopes required
    pub min_cosine: f32,         // 0.85–0.92 if LUT provided
    // DP/edge scoring weights
    pub w_spacing: f32,          // α
    pub w_coelute: f32,          // β
    pub w_monotonic: f32,        // γ (soft)
    pub penalty_skip_one: f32,   // λ_gap
    // auction
    pub steal_delta: f32,        // improvement needed to steal a node (not used in simple version)
    // seed hygiene
    pub require_lowest_is_mono: bool, // guard: seed must be local mono
}

// ----- Output types ----------------------------------------------------------

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
    pub iso_raw: [f32; 8],       // from member cluster raw_sum (proxy)
    pub iso_l2: [f32; 8],        // L2-normalized
    pub cos_averagine: f32,      // NaN if LUT not provided
    pub member_cluster_ids: Vec<usize>, // original cluster indices
}

#[derive(Clone, Debug)]
pub struct Envelope {
    pub id: usize,
    pub cluster_ids: Vec<usize>, // original cluster indices
    pub rt_bounds: (usize, usize),
    pub im_bounds: (usize, usize),
    pub mz_center: f32,          // raw_sum-weighted
    pub mz_span_da: f32,
    pub charge_hint: Option<u8>, // from spacing if available
}

#[derive(Clone, Debug)]
pub struct GroupingOutput {
    pub envelopes: Vec<Envelope>,
    pub assignment: Vec<Option<usize>>, // cluster_i -> Some(envelope_id)
}

// ----- Helpers ---------------------------------------------------------------

#[inline]
fn ppm_between(a: f32, b: f32) -> f32 {
    let dm = (a - b).abs();
    let center = ((a + b) * 0.5).abs().max(1e-6);
    1.0e6 * dm / center
}
#[inline]
fn expand(w: (usize, usize), pad: usize) -> (usize, usize) {
    (w.0.saturating_sub(pad), w.1.saturating_add(pad))
}
#[inline]
fn overlap1d(a: (usize, usize), b: (usize, usize)) -> usize {
    if a.1 < b.0 || b.1 < a.0 { 0 } else { a.1.min(b.1) - a.0.max(b.0) + 1 }
}
#[inline]
fn frac_overlap(a: (usize, usize), b: (usize, usize)) -> f32 {
    let l = a.0.max(b.0);
    let r = a.1.min(b.1);
    if r < l { 0.0 } else { (r - l + 1) as f32 / ((a.1 - a.0 + 1).max(b.1 - b.0 + 1) as f32) }
}
#[inline]
fn gaussian_overlap_ok(mu_a: f32, sig_a: f32, mu_b: f32, sig_b: f32, c: f32) -> bool {
    let pooled = (sig_a*sig_a + sig_b*sig_b).sqrt().max(1e-6);
    (mu_a - mu_b).abs() <= c * pooled
}
#[inline]
fn coelution_score(u_rt: (f32, f32), u_im: (f32, f32), v_rt: (f32, f32), v_im: (f32, f32)) -> f32 {
    let d_rt = (u_rt.0 - v_rt.0).abs() / (u_rt.1.hypot(v_rt.1)).max(1e-6);
    let d_im = (u_im.0 - v_im.0).abs() / (u_im.1.hypot(v_im.1)).max(1e-6);
    (-(d_rt + d_im)).exp() // (0,1]
}
#[inline]
fn cosine(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let mut dot = 0f32; let mut na = 0f32; let mut nb = 0f32;
    for i in 0..8 { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt()*nb.sqrt()) }
}

fn infer_charge_from_members(mzs: &[f32], z_min:u8, z_max:u8) -> Option<u8> {
    if mzs.len() < 2 { return None; }
    let mut m = mzs.to_vec();
    m.sort_by(|a,b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // neighbor deltas
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

    if best.0 == 0 { return None; }

    // accept only if both absolute and relative error are small
    let t = 1.003355f32 / (best.0 as f32);
    let abs_ok = best.1 < 0.02;          // < 0.02 Da mean abs error
    let rel_ok = (best.1 / t) < 0.12;    // < 12% of target spacing
    if abs_ok && rel_ok { Some(best.0) } else { None }
}

// ----- Near-duplicate merge (DSU) -------------------------------------------

#[derive(Clone, Debug)]
struct Dsu { parent: Vec<usize>, size: Vec<usize> }
impl Dsu {
    fn new(n: usize) -> Self { Self { parent: (0..n).collect(), size: vec![1; n] } }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    fn union(&mut self, a: usize, b: usize) -> bool {
        let mut ra = self.find(a); let mut rb = self.find(b);
        if ra == rb { return false; }
        if self.size[ra] < self.size[rb] { std::mem::swap(&mut ra, &mut rb); }
        self.parent[rb] = ra; self.size[ra] += self.size[rb]; true
    }
    fn groups(mut self) -> Vec<Vec<usize>> {
        let n = self.parent.len();
        let mut root_of = vec![0usize; n];
        for i in 0..n { root_of[i] = self.find(i); }
        let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, r) in root_of.into_iter().enumerate() { buckets.entry(r).or_default().push(i); }
        let mut out: Vec<Vec<usize>> = buckets.into_values().collect();
        out.sort_by_key(|g| std::cmp::Reverse(g.len()));
        out
    }
}

fn is_near_duplicate(ci:&ClusterResult1D, cj:&ClusterResult1D) -> bool {
    let rt = frac_overlap(ci.rt_window, cj.rt_window) >= 0.6
        || gaussian_overlap_ok(ci.rt_fit.mu, ci.rt_fit.sigma, cj.rt_fit.mu, cj.rt_fit.sigma, 2.0);
    let im = frac_overlap(ci.im_window, cj.im_window) >= 0.6
        || gaussian_overlap_ok(ci.im_fit.mu, ci.im_fit.sigma, cj.im_fit.mu, cj.im_fit.sigma, 2.0);
    let mz_close = ppm_between(ci.mz_fit.mu, cj.mz_fit.mu) <= 3.0;
    rt && im && mz_close
}

#[derive(Clone, Debug)]
struct Super {
    member_cids: Vec<usize>,      // original cluster indices
    rt_bounds: (usize,usize),
    im_bounds: (usize,usize),
    mz_center: f32,               // raw_sum-weighted μ
    raw_sum: f32,
    rt_mu: f32, rt_sig: f32,
    im_mu: f32, im_sig: f32,
}

fn summarize_super(group: &[usize], cl: &[ClusterResult1D], good_ids: &[usize]) -> Super {
    let mut rt_l = usize::MAX; let mut rt_r = 0usize;
    let mut im_l = usize::MAX; let mut im_r = 0usize;
    let mut wsum = 0f32; let mut mz_w = 0f32; let mut raw_sum = 0f32;
    let mut mz_min = f32::INFINITY; let mut mz_max = f32::NEG_INFINITY;

    let mut rt_mu_w = 0f32; let mut im_mu_w = 0f32;

    for &i in group {
        let c = &cl[i];
        rt_l = rt_l.min(c.rt_window.0); rt_r = rt_r.max(c.rt_window.1);
        im_l = im_l.min(c.im_window.0); im_r = im_r.max(c.im_window.1);
        let w = c.raw_sum.max(1.0);
        wsum += w; mz_w += w * c.mz_fit.mu;
        rt_mu_w += w * c.rt_fit.mu; im_mu_w += w * c.im_fit.mu;
        raw_sum += c.raw_sum;
        mz_min = mz_min.min(c.mz_fit.mu); mz_max = mz_max.max(c.mz_fit.mu);
    }
    // map compact indices to original indices
    let member_cids: Vec<usize> = group.iter().map(|&cid_c| good_ids[cid_c]).collect();
    let mz_center = if wsum>0.0 { mz_w/wsum } else { (mz_min+mz_max)*0.5 };
    let (rt_mu, im_mu) = if wsum>0.0 {(rt_mu_w/wsum, im_mu_w/wsum)} else { (0.0,0.0) };

    // pooled sigmas across members
    let (mut rt_var, mut im_var) = (0f32, 0f32);
    for &i in group {
        let c = &cl[i];
        rt_var += c.rt_fit.sigma * c.rt_fit.sigma;
        im_var += c.im_fit.sigma * c.im_fit.sigma;
    }
    let m = group.len().max(1) as f32;
    let (rt_sig, im_sig) = ((rt_var/m).sqrt(), (im_var/m).sqrt());

    Super { member_cids, rt_bounds: (rt_l, rt_r), im_bounds: (im_l, im_r),
        mz_center, raw_sum, rt_mu, rt_sig, im_mu, im_sig }
}

// ----- Isotopic adjacency & edges -------------------------------------------

#[derive(Clone, Debug)]
struct Edge { v: usize, z: u8, w: f32 }

fn build_edges(super_nodes: &[Super], p: &GroupingParams, fb: &FeatureBuildParams) -> Vec<Vec<Edge>> {
    let mut idx: Vec<usize> = (0..super_nodes.len()).collect();
    idx.sort_unstable_by(|&a,&b| super_nodes[a].mz_center.partial_cmp(&super_nodes[b].mz_center).unwrap_or(Ordering::Equal));

    let mut adj: Vec<Vec<Edge>> = vec![Vec::new(); super_nodes.len()];
    let m: Vec<f32> = idx.iter().map(|&i| super_nodes[i].mz_center).collect();

    for z in p.z_min..=p.z_max {
        let d1 = 1.003355f32 / (z as f32);
        let tol = (d1 * (p.iso_ppm_tol * 1e-6)).max(p.iso_abs_da);

        for &delta in &[d1, 2.0*d1] {
            let skip_one = delta > d1 * 1.5;
            let mut j = 0usize;

            for i_local in 0..idx.len() {
                let mi = m[i_local];
                let lo = (mi + delta - tol).max(0.0);
                while j < idx.len() && m[j] < lo { j += 1; }
                let mut k = j;
                let hi = mi + delta + tol;

                while k < idx.len() && m[k] <= hi {
                    let u = idx[i_local]; let v = idx[k];
                    if u != v {
                        let u_s = &super_nodes[u]; let v_s = &super_nodes[v];

                        let rt_ok = overlap1d(expand(u_s.rt_bounds, p.rt_pad_overlap), expand(v_s.rt_bounds, p.rt_pad_overlap)) > 0
                            && gaussian_overlap_ok(u_s.rt_mu, u_s.rt_sig, v_s.rt_mu, v_s.rt_sig, 2.5);
                        let im_ok = overlap1d(expand(u_s.im_bounds, p.im_pad_overlap), expand(v_s.im_bounds, p.im_pad_overlap)) > 0
                            && gaussian_overlap_ok(u_s.im_mu, u_s.im_sig, v_s.im_mu, v_s.im_sig, 2.5);

                        if rt_ok && im_ok {
                            let spacing_err = ((m[k]-m[i_local]).abs() - delta).abs();
                            let tau = tol.max(1e-6);
                            let s_spacing = (-spacing_err / tau).exp();
                            let s_co = coelution_score((u_s.rt_mu,u_s.rt_sig),(u_s.im_mu,u_s.im_sig),
                                                       (v_s.rt_mu,v_s.rt_sig),(v_s.im_mu,v_s.im_sig));
                            let s_mono = 1.0;

                            let mut w = fb.w_spacing * s_spacing + fb.w_coelute * s_co + fb.w_monotonic * s_mono;
                            if skip_one { w -= fb.penalty_skip_one; } // stronger, always subtract

                            adj[u].push(Edge{ v, z, w });
                        }
                    }
                    k += 1;
                }
            }
        }
    }
    adj
}

// ----- DP best chain per seed × charge --------------------------------------

#[derive(Clone, Debug)]
struct Chain { path: Vec<usize>, score: f32 }

fn dp_best_chain(seed: usize, z: u8, adj: &[Vec<Edge>], k_max: usize) -> Chain {
    use std::collections::VecDeque;

    // best_score[v] = (score, prev)
    let mut best_score: HashMap<usize, (f32, Option<usize>)> = HashMap::new();
    best_score.insert(seed, (0.0, None));

    let mut q = VecDeque::new();
    q.push_back(seed);

    while let Some(u) = q.pop_front() {
        let (s_u, _) = *best_score.get(&u).unwrap();
        for e in adj[u].iter().filter(|e| e.z == z) {
            let v = e.v;
            let ns = s_u + e.w;
            let entry = best_score.entry(v).or_insert((f32::NEG_INFINITY, None));
            if ns > entry.0 {
                *entry = (ns, Some(u));
                q.push_back(v);
            }
        }
    }

    // choose best endpoint with lexicographic (length desc, score desc) under k_max
    let mut best: Option<(usize, usize, f32)> = None; // (end, len, score)
    for (&v, &(s, prev)) in &best_score {
        // reconstruct length (bounded)
        let mut len = 1usize;
        let mut t = prev;
        while let Some(p) = t {
            len += 1;
            if len > k_max { break; }
            t = best_score.get(&p).and_then(|x| x.1);
        }
        if len > k_max { continue; }

        match best {
            None => best = Some((v, len, s)),
            Some((_, bl, bs)) => {
                if len > bl || (len == bl && s > bs) {
                    best = Some((v, len, s));
                }
            }
        }
    }

    if let Some((end, _len, s)) = best {
        // reconstruct path
        let mut path = Vec::<usize>::new();
        let mut cur = end;
        path.push(cur);
        while let Some((_, Some(prev))) = best_score.get(&cur).copied() {
            path.push(prev);
            cur = prev;
        }
        path.reverse();
        if path.len() > k_max { path.truncate(k_max); }
        Chain { path, score: s }
    } else {
        Chain { path: vec![seed], score: 0.0 }
    }
}

fn chain_to_envelope(
    eid: usize,
    chain: &Chain,
    supers: &[Super],
    charge_hint: Option<u8>,
) -> Envelope {
    let mut rt_min = usize::MAX; let mut rt_max = 0;
    let mut im_min = usize::MAX; let mut im_max = 0;
    let mut mz_lo = f32::MAX; let mut mz_hi = f32::MIN;
    let mut all = Vec::<usize>::new();
    let mut mz_weighted = 0f64; let mut wsum = 0f64;

    for &sid in &chain.path {
        let s = &supers[sid];
        rt_min = rt_min.min(s.rt_bounds.0); rt_max = rt_max.max(s.rt_bounds.1);
        im_min = im_min.min(s.im_bounds.0); im_max = im_max.max(s.im_bounds.1);
        mz_lo = mz_lo.min(s.mz_center); mz_hi = mz_hi.max(s.mz_center);
        mz_weighted += (s.mz_center as f64) * (s.raw_sum as f64);
        wsum += s.raw_sum as f64;
        all.extend_from_slice(&s.member_cids);
    }
    let mz_center = if wsum>0.0 {(mz_weighted/wsum) as f32} else {(mz_lo+mz_hi)*0.5};

    // Only trust a hint if there’s enough structure (≥3 nodes => ≥2 spacings)
    let safe_hint = if chain.path.len() >= 3 { charge_hint } else { None };

    Envelope {
        id: eid,
        cluster_ids: all,
        rt_bounds: (rt_min, rt_max),
        im_bounds: (im_min, im_max),
        mz_center,
        mz_span_da: mz_hi - mz_lo,
        charge_hint: safe_hint,
    }
}
fn envelope_iso_vector_from_clusters(env: &Envelope, clusters: &[ClusterResult1D], mz_mono: f32, z: u8, k_max: usize) -> [f32; 8] {
    let mut iso = [0f32; 8];
    if z == 0 { // mono only
        // collect members closest to mono
        let mut best = (f32::MAX, 0f32);
        for &cid in &env.cluster_ids {
            let c = &clusters[cid];
            let dm = (c.mz_fit.mu - mz_mono).abs();
            if dm < best.0 { best = (dm, c.raw_sum); }
        }
        iso[0] = best.1;
        return iso;
    }
    let k_keep = k_max.min(8);
    let dmz = 1.003355f32 / (z as f32);
    // For each member cluster, assign it to nearest isotope index
    for &cid in &env.cluster_ids {
        let c = &clusters[cid];
        let kf = ((c.mz_fit.mu - mz_mono) / dmz).round();
        if kf.is_finite() {
            let k = kf as i32;
            if k >= 0 && (k as usize) < k_keep {
                iso[k as usize] += c.raw_sum.max(0.0);
            }
        }
    }
    iso
}

fn l2_norm8(v: &[f32;8]) -> f32 {
    (v.iter().map(|x| (*x as f64)*(*x as f64)).sum::<f64>()).sqrt() as f32
}

// ----- Auction (simple non-stealing version) --------------------------------

fn auction_assign(
    mut envs: Vec<(Envelope, u8, f32, f32)>, // (env, z, score, cosine)
    clusters: &[ClusterResult1D],
    _fp: &FeatureBuildParams,
) -> Vec<Feature> {
    // Sort by score desc
    envs.sort_by(|a,b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

    let mut claimed = vec![false; clusters.len()];
    let mut features = Vec::<Feature>::new();

    for (env, z, _score, cos) in envs.into_iter() {
        // compute mz_mono as min member μ
        let mut mz_mono = f32::INFINITY;
        for &cid in &env.cluster_ids {
            let m = clusters[cid].mz_fit.mu;
            if m.is_finite() { mz_mono = mz_mono.min(m); }
        }
        if !mz_mono.is_finite() || mz_mono <= 50.0 { continue; }

        // Claim clusters (no stealing)
        if env.cluster_ids.iter().any(|&cid| claimed[cid]) {
            continue;
        }
        for &cid in &env.cluster_ids { claimed[cid] = true; }

        // intensities
        let iso_raw = envelope_iso_vector_from_clusters(&env, clusters, mz_mono, z, 8);
        let mut iso_l2 = iso_raw;
        let norm = l2_norm8(&iso_l2);
        if norm > 0.0 {
            for i in 0..8 { iso_l2[i] /= norm; }
        }

        // finalize center
        let mut mz_weighted = 0f64; let mut wsum = 0f64;
        for &cid in &env.cluster_ids {
            let c = &clusters[cid];
            let w = c.raw_sum.max(1.0) as f64;
            mz_weighted += (c.mz_fit.mu as f64) * w;
            wsum += w;
        }
        let mz_center = if wsum>0.0 {(mz_weighted/wsum) as f32} else { env.mz_center };

        let neutral = if z>0 { (mz_mono - 1.007_276_5_f32) * (z as f32) } else { f32::NAN };

        features.push(Feature{
            envelope_id: env.id,
            charge: z,
            mz_mono,
            neutral_mass: neutral,
            rt_bounds: env.rt_bounds,
            im_bounds: env.im_bounds,
            mz_center,
            n_members: env.cluster_ids.len(),
            iso_raw,
            iso_l2,
            cos_averagine: if cos.is_finite() { cos } else { f32::NAN },
            member_cluster_ids: env.cluster_ids.clone(),
        });
    }

    features
}

// ----- Public API ------------------------------------------------------------

pub struct BuildResult {
    pub features: Vec<Feature>,
    pub grouping: GroupingOutput,
}

/// Main entry: build features from ClusterResult1D only.
pub fn build_features_from_clusters(
    clusters: &[ClusterResult1D],
    gp: &GroupingParams,
    fp: &FeatureBuildParams,
    lut: Option<&AveragineLut>,
) -> BuildResult {
    // filter usable
    let good_ids: Vec<usize> = clusters.iter().enumerate()
        .filter(|(_, c)| c.raw_sum > 0.0
            && c.mz_fit.mu.is_finite() && c.mz_fit.mu > 50.0
            && c.mz_fit.sigma.is_finite() && c.mz_fit.sigma > 0.0)
        .map(|(i, _)| i)
        .collect();

    if good_ids.is_empty() {
        return BuildResult {
            features: vec![],
            grouping: GroupingOutput { envelopes: vec![], assignment: vec![None; clusters.len()] },
        };
    }

    // compact vector view
    let compact: Vec<ClusterResult1D> = good_ids.iter().map(|&i| clusters[i].clone()).collect();

    // A) DSU near-duplicates (sigma-aware RT/IM + tight m/z) with coarse bucketing
    let mut buckets: HashMap<(i32,i32,i32), Vec<usize>> = HashMap::new();
    let rt_bin: i32 = 4; // frames; tune
    let im_bin: i32 = 8; // scans; tune
    let ppm_bin: f32 = gp.mz_ppm_tol.max(3.0);
    let step = (1.0 + ppm_bin * 1e-6).ln();
    let logppm = |mz: f32| -> i32 { (mz.ln() / step).floor() as i32 };

    for (i, c) in compact.iter().enumerate() {
        let br = (c.rt_fit.mu as i32) / rt_bin.max(1);
        let bi = (c.im_fit.mu as i32) / im_bin.max(1);
        let bz = logppm(c.mz_fit.mu);
        buckets.entry((br,bi,bz)).or_default().push(i);
    }

    let mut dsu = Dsu::new(compact.len());
    let keys: Vec<(i32,i32,i32)> = buckets.keys().copied().collect();
    for &(br,bi,bz) in &keys {
        let ids = &buckets[&(br,bi,bz)];
        for a in 0..ids.len() {
            for b in (a+1)..ids.len() {
                if is_near_duplicate(&compact[ids[a]], &compact[ids[b]]) { dsu.union(ids[a], ids[b]); }
            }
        }
        for dr in 0..=1 {
            for di in -1..=1 {
                for dz in -1..=1 {
                    if dr==0 && di<=0 && dz<=0 { continue; }
                    let key = (br+dr, bi+di, bz+dz);
                    if let Some(nids) = buckets.get(&key) {
                        for &i in ids {
                            for &j in nids {
                                let (a,b) = if i<j {(i,j)} else {(j,i)};
                                if is_near_duplicate(&compact[a], &compact[b]) { dsu.union(a,b); }
                            }
                        }
                    }
                }
            }
        }
    }

    let groups = dsu.groups();
    let supers: Vec<Super> = groups.iter().map(|g| summarize_super(g, &compact, &good_ids)).collect();

    // B) edges
    let adj = build_edges(&supers, gp, fp);

    // C) seeds = supers sorted by strength
    let mut seeds: Vec<usize> = (0..supers.len()).collect();
    seeds.sort_unstable_by(|&a,&b| supers[b].raw_sum.partial_cmp(&supers[a].raw_sum).unwrap_or(Ordering::Equal));

    // D) build candidate envelopes via DP for best z
    let mut candidates: Vec<(Envelope, u8, f32, f32)> = Vec::new(); // env, z, score, cosine

    for &seed in &seeds {
        // rough charge hint from member μ’s
        let mut mzs = Vec::<f32>::new();
        for &cid in &supers[seed].member_cids {
            let m = clusters[cid].mz_fit.mu;
            if m.is_finite() { mzs.push(m); }
        }
        let hint = infer_charge_from_members(&mzs, gp.z_min, gp.z_max);

        // if no hint, we still consider full range
        let z_list: Vec<u8> = if let Some(z) = hint { vec![z] } else { (gp.z_min..=gp.z_max).collect() };

        for z in z_list {
            let chain = dp_best_chain(seed, z, &adj, fp.k_max);
            if chain.path.len() < fp.min_members { continue; }

            let env_id = candidates.len();
            let env = chain_to_envelope(env_id, &chain, &supers, hint);

            // mz_mono
            let mut mz_mono = f32::INFINITY;
            for &cid in &env.cluster_ids {
                let m = clusters[cid].mz_fit.mu;
                if m.is_finite() { mz_mono = mz_mono.min(m); }
            }
            if !mz_mono.is_finite() || mz_mono <= 50.0 { continue; }

            // iso vectors
            let iso_raw = envelope_iso_vector_from_clusters(&env, clusters, mz_mono, z, fp.k_max);
            let mut iso_l2 = iso_raw;
            let norm = l2_norm8(&iso_l2);
            if norm > 0.0 {
                for i in 0..8 { iso_l2[i] /= norm; }
            }

            // cosine (optional)
            let cos = if let Some(lut) = lut {
                if z >= lut.z_min && z <= lut.z_max {
                    let neutral = (mz_mono - 1.007_276_5_f32) * (z as f32);
                    let avg = lut.lookup(neutral, z);
                    cosine(&iso_l2, &avg)
                } else { f32::NAN }
            } else { f32::NAN };

            // scoring:
            // - lexicographic preference to longer chains happens earlier (dp_best_chain),
            //   but we reinforce here by adding a small length bonus.
            // - mild prior penalty against very high z to curb random tight spacings.
            let len_bonus = 0.05 * (chain.path.len() as f32);          // small positive bonus
            let z_prior   = ((z as i32 - 3).max(0) as f32) * 0.02;     // penalty for z>3
            let score = chain.score + len_bonus + if cos.is_finite() { 0.5 * cos } else { 0.0 } - z_prior;

            candidates.push((env, z, score, cos));
        }
    }

    // E) auction assignment (simple non-steal by default)
    let features = auction_assign(candidates, clusters, fp);

    // F) assignment map (cluster -> envelope)
    let mut assignment = vec![None; clusters.len()];
    for feat in &features {
        for &cid in &feat.member_cluster_ids {
            assignment[cid] = Some(feat.envelope_id);
        }
    }

    // envelopes back from features (optional, but keep for API symmetry)
    let mut env_map: HashMap<usize, Envelope> = HashMap::new();
    for f in &features {
        env_map.insert(f.envelope_id, Envelope {
            id: f.envelope_id,
            cluster_ids: f.member_cluster_ids.clone(),
            rt_bounds: f.rt_bounds,
            im_bounds: f.im_bounds,
            mz_center: f.mz_center,
            mz_span_da: 0.0, // not tracked here post-hoc
            charge_hint: Some(f.charge).filter(|&z| z>0),
        });
    }
    let mut envelopes: Vec<Envelope> = env_map.into_values().collect();
    envelopes.sort_by_key(|e| e.id);

    BuildResult {
        features,
        grouping: GroupingOutput { envelopes, assignment },
    }
}
 */