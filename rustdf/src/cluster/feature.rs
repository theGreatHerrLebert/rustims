use std::cmp::Ordering;
use mscore::algorithm::isotope::generate_averagine_spectra;
use crate::cluster::cluster::ClusterResult1D;
use serde::{Deserialize, Serialize};

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleFeature {
    pub feature_id: usize,
    pub charge: u8,
    pub mz_mono: f32,
    pub neutral_mass: f32,
    pub rt_bounds: (usize, usize),
    pub im_bounds: (usize, usize),
    pub mz_center: f32,
    pub n_members: usize,

    /// Indices into the *input slice* of clusters (for local lookups).
    pub member_cluster_indices: Vec<usize>,

    /// Stable cluster IDs copied from `ClusterResult1D::cluster_id`.
    pub member_cluster_ids: Vec<u64>,

    pub raw_sum: f32,

    /// Cosine similarity vs. Averagine envelope (if LUT was provided; 0 otherwise).
    pub cos_averagine: f32,

    /// Optional: the actual MS1 member clusters that make up this feature.
    /// This lets us score a feature directly against an MS2 cluster without
    /// needing the global MS1 slice.
    pub member_clusters: Vec<ClusterResult1D>,
}

#[derive(Clone, Debug)]
pub struct SimpleFeatureParams {
    pub z_min: u8,
    pub z_max: u8,
    pub iso_ppm_tol: f32,        // e.g. 10.0
    pub iso_abs_da: f32,         // e.g. 0.003
    pub min_members: usize,      // at least this many isotopes
    pub max_members: usize,      // cap chain length
    pub min_raw_sum: f32,        // minimal cluster raw_sum
    pub min_mz: f32,             // minimal m/z to consider
    pub min_rt_overlap_frac: f32, // e.g. 0.3
    pub min_im_overlap_frac: f32, // e.g. 0.3
    /// If > 0 and LUT is provided, chains with cosine < min_cosine are dropped.
    pub min_cosine: f32,

    /// Weight for spacing penalty (0.0 = disabled, higher = penalize bad spacing more).
    ///
    /// Penalty is computed as average |Δmz_observed - Δmz_theory(z)| in *units of tolerance*
    /// (i.e. divided by `iso_tolerance_da(mid_mz)`), then multiplied by this weight and
    /// subtracted from the raw_sum-based chain score.
    pub w_spacing_penalty: f32,
}

impl Default for SimpleFeatureParams {
    fn default() -> Self {
        Self {
            z_min: 1,
            z_max: 5,
            iso_ppm_tol: 10.0,
            iso_abs_da: 0.003,
            min_members: 2,
            max_members: 5,
            min_raw_sum: 0.0,
            min_mz: 100.0,
            min_rt_overlap_frac: 0.3,
            min_im_overlap_frac: 0.3,
            min_cosine: 0.0,
            w_spacing_penalty: 0.0,
        }
    }
}

fn cluster_mz_mu(c: &ClusterResult1D) -> Option<f32> {
    if let Some(ref f) = c.mz_fit {
        if f.mu.is_finite() && f.mu > 0.0 {
            return Some(f.mu);
        }
    }
    if let Some((lo, hi)) = c.mz_window {
        let mu = 0.5 * (lo + hi);
        if mu.is_finite() && mu > 0.0 {
            return Some(mu);
        }
    }
    None
}

fn _sigma_from_fwhm(fwhm: f32) -> Option<f32> {
    if !fwhm.is_finite() || fwhm <= 0.0 {
        return None;
    }
    let two_sqrt_two_ln2 = 2.0 * 2.0_f32.ln().sqrt();
    let sigma = fwhm / two_sqrt_two_ln2;
    if sigma.is_finite() && sigma > 0.0 {
        Some(sigma)
    } else {
        None
    }
}

fn _cluster_mz_sigma(c: &ClusterResult1D) -> Option<f32> {
    if let Some(ref f) = c.mz_fit {
        if f.sigma.is_finite() && f.sigma > 0.0 {
            return Some(f.sigma);
        }
    }
    if let Some((lo, hi)) = c.mz_window {
        let fwhm = (hi - lo).abs();
        if let Some(s) = _sigma_from_fwhm(fwhm) {
            return Some(s);
        }
    }
    None
}

#[inline]
fn frac_overlap(a: (usize, usize), b: (usize, usize)) -> f32 {
    let l = a.0.max(b.0);
    let r = a.1.min(b.1);
    if r < l {
        0.0
    } else {
        let len = (r - l + 1) as f32;
        let len_a = (a.1 - a.0 + 1) as f32;
        let len_b = (b.1 - b.0 + 1) as f32;
        len / len_a.max(len_b)
    }
}

#[inline]
fn iso_tolerance_da(mz: f32, p: &SimpleFeatureParams) -> f32 {
    let ppm_term = mz.abs().max(1e-6) * (p.iso_ppm_tol * 1e-6);
    ppm_term.max(p.iso_abs_da)
}

/// Cosine between raw isotope vector (from chain) and LUT envelope.
fn cosine_iso_lut(
    neutral_mass: f32,
    z: u8,
    iso_raw: &[f32],
    lut: &AveragineLut,
) -> f32 {
    if iso_raw.is_empty() {
        return 0.0;
    }

    let env = lut.lookup(neutral_mass, z); // already L2-normalized, zero-padded to 8
    let mut v = [0.0f32; 8];

    let keep = iso_raw.len().min(8);
    let mut norm2 = 0.0f32;
    for i in 0..keep {
        let x = iso_raw[i].max(0.0); // just in case
        v[i] = x;
        norm2 += x * x;
    }

    if norm2 <= 0.0 {
        return 0.0;
    }
    let inv_norm = norm2.sqrt().recip();

    let mut dot = 0.0f32;
    for i in 0..8 {
        let x = v[i] * inv_norm;
        dot += x * env[i];
    }
    dot
}

#[derive(Clone, Debug)]
struct GoodCluster {
    /// index in the original `clusters` slice
    orig_idx: usize,

    /// stable ID from `ClusterResult1D`
    pub cluster_id: u64,

    mz_mu: f32,
    _rt_mu: f32,
    rt_win: (usize, usize),
    _im_mu: f32,
    im_win: (usize, usize),
    raw_sum: f32,
}

// ---------------------------- Core algorithm ------------------------------
// ---------------------------- Core algorithm ------------------------------

pub fn build_simple_features_from_clusters(
    clusters: &[ClusterResult1D],
    params: &SimpleFeatureParams,
    lut: Option<&AveragineLut>,
) -> Vec<SimpleFeature> {
    if clusters.is_empty() {
        return Vec::new();
    }

    // 1) Filter to usable clusters and precompute a light-weight view
    let mut good: Vec<GoodCluster> = clusters
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            if c.raw_sum < params.min_raw_sum {
                return None;
            }
            let mz_mu = cluster_mz_mu(c)?;
            if mz_mu <= params.min_mz {
                return None;
            }
            if !c.rt_fit.mu.is_finite() || !c.im_fit.mu.is_finite() {
                return None;
            }
            Some(GoodCluster {
                orig_idx: i,
                cluster_id: c.cluster_id,
                mz_mu,
                _rt_mu: c.rt_fit.mu,
                rt_win: c.rt_window,
                _im_mu: c.im_fit.mu,
                im_win: c.im_window,
                raw_sum: c.raw_sum,
            })
        })
        .collect();

    if good.is_empty() {
        return Vec::new();
    }

    // 2) Sort by m/z for fast neighborhood search
    good.sort_by(|a, b| a.mz_mu.partial_cmp(&b.mz_mu).unwrap_or(Ordering::Equal));

    // Precompute mz array for binary search
    let mzs: Vec<f32> = good.iter().map(|g| g.mz_mu).collect();

    // Also build a "seed order" by descending raw_sum
    let mut seed_indices: Vec<usize> = (0..good.len()).collect();
    seed_indices.sort_unstable_by(|&a, &b| {
        good[b]
            .raw_sum
            .partial_cmp(&good[a].raw_sum)
            .unwrap_or(Ordering::Equal)
    });

    // track assignment
    let mut assigned = vec![false; good.len()];
    let mut features: Vec<SimpleFeature> = Vec::new();

    // 3) Main loop over seeds
    for seed_idx in seed_indices {
        if assigned[seed_idx] {
            continue;
        }

        let mut best_chain: Vec<usize> = Vec::new();
        let mut best_z: u8 = 0;
        let mut best_score: f32 = f32::NEG_INFINITY;

        for z in params.z_min..=params.z_max {
            let delta = 1.003_355_f32 / (z as f32);

            // ------------------------------------------------------------
            // Precompute reference envelope for this (seed, z) if LUT present
            // ------------------------------------------------------------
            let env_opt: Option<[f32; 8]> = if let Some(lut_ref) = lut {
                let seed_mz = good[seed_idx].mz_mu;
                if seed_mz.is_finite() && seed_mz > params.min_mz {
                    let neutral_seed =
                        (seed_mz - 1.007_276_5_f32).max(0.0) * (z as f32);
                    Some(lut_ref.lookup(neutral_seed, z))
                } else {
                    None
                }
            } else {
                None
            };

            // Small slack so we don't overreact to noise
            const DIR_EPS_ENV: f32 = 1e-3; // tiny threshold for env direction
            const SLACK_UP: f32 = 1.05;    // require ≥ +5% when env rises
            const SLACK_DOWN: f32 = 0.95;  // require ≤ -5% when env falls

            // ---- UPWARD extension -------------------------------------------------
            let mut chain_up: Vec<usize> = Vec::new();
            chain_up.push(seed_idx);

            // keep observed raw sums for shape check
            let mut iso_raw_up: Vec<f32> = Vec::new();
            iso_raw_up.push(good[seed_idx].raw_sum.max(0.0));

            let mut cur_up = seed_idx;
            'upward: while chain_up.len() < params.max_members {
                let cur = &good[cur_up];
                let target_m = cur.mz_mu + delta;
                let tol = iso_tolerance_da(target_m, params);

                let lo = target_m - tol;
                let hi = target_m + tol;

                // binary search window [lo, hi] in sorted mzs
                let left = mzs
                    .binary_search_by(|x| x.partial_cmp(&lo).unwrap_or(Ordering::Equal))
                    .unwrap_or_else(|i| i);
                let mut right = match mzs.binary_search_by(|x| x.partial_cmp(&hi).unwrap_or(Ordering::Equal)) {
                    Ok(i) => i + 1,
                    Err(i) => i,
                };
                if right > mzs.len() {
                    right = mzs.len();
                }

                let mut best_candidate: Option<(usize, f32)> = None; // (idx, score)

                for k in left..right {
                    if assigned[k] || chain_up.contains(&k) {
                        continue;
                    }
                    let cand = &good[k];

                    // co-elution constraints (simple v1: overlap fractions)
                    let rt_ov = frac_overlap(cur.rt_win, cand.rt_win);
                    if rt_ov < params.min_rt_overlap_frac {
                        continue;
                    }
                    let im_ov = frac_overlap(cur.im_win, cand.im_win);
                    if im_ov < params.min_im_overlap_frac {
                        continue;
                    }

                    // simple scoring: weighted by raw_sum and overlap
                    let score = cand.raw_sum * (0.5 * (rt_ov + im_ov));

                    match best_candidate {
                        None => best_candidate = Some((k, score)),
                        Some((_, best_s)) => {
                            if score > best_s {
                                best_candidate = Some((k, score));
                            }
                        }
                    }
                }

                if let Some((best_k, _)) = best_candidate {
                    // ----------------- SHAPE CHECK vs. ENVELOPE (UPWARD) -----------------
                    if let Some(env) = env_opt {
                        let obs_prev = *iso_raw_up.last().unwrap_or(&0.0);
                        let obs_new = good[best_k].raw_sum.max(0.0);

                        // position of the *new* isotope in this direction
                        let env_idx = iso_raw_up.len().min(env.len().saturating_sub(1));

                        if env_idx > 0 && env_idx < env.len() {
                            let env_prev = env[env_idx - 1];
                            let env_cur = env[env_idx];

                            // determine expected direction from envelope
                            if env_cur > env_prev * (1.0 + DIR_EPS_ENV) {
                                // Envelope says: this isotope should go UP
                                if obs_new < obs_prev * SLACK_UP {
                                    // observed does NOT go up enough -> stop extending
                                    break 'upward;
                                }
                            } else if env_cur < env_prev * (1.0 - DIR_EPS_ENV) {
                                // Envelope says: this isotope should go DOWN
                                if obs_new > obs_prev * SLACK_DOWN {
                                    // observed does NOT go down enough -> stop extending
                                    break 'upward;
                                }
                            }
                            // If envelope is flat-ish, we don't enforce direction.
                        }

                        iso_raw_up.push(obs_new);
                    }

                    chain_up.push(best_k);
                    cur_up = best_k;
                } else {
                    break;
                }
            }

            // ---- DOWNWARD extension -----------------------------------------------
            let mut chain_down: Vec<usize> = Vec::new();
            let mut iso_raw_down: Vec<f32> = Vec::new();
            iso_raw_down.push(good[seed_idx].raw_sum.max(0.0));

            let mut cur_down = seed_idx;

            'downward: while (chain_down.len() + chain_up.len()) < params.max_members {
                let cur = &good[cur_down];
                let target_m = cur.mz_mu - delta;
                if target_m <= 0.0 {
                    break;
                }
                let tol = iso_tolerance_da(target_m, params);
                let lo = target_m - tol;
                let hi = target_m + tol;

                let left = mzs
                    .binary_search_by(|x| x.partial_cmp(&lo).unwrap_or(Ordering::Equal))
                    .unwrap_or_else(|i| i);
                let mut right = match mzs.binary_search_by(|x| x.partial_cmp(&hi).unwrap_or(Ordering::Equal)) {
                    Ok(i) => i + 1,
                    Err(i) => i,
                };
                if right > mzs.len() {
                    right = mzs.len();
                }

                let mut best_candidate: Option<(usize, f32)> = None; // (idx, score)

                for k in left..right {
                    if assigned[k] || chain_up.contains(&k) || chain_down.contains(&k) {
                        continue;
                    }
                    let cand = &good[k];

                    let rt_ov = frac_overlap(cur.rt_win, cand.rt_win);
                    if rt_ov < params.min_rt_overlap_frac {
                        continue;
                    }
                    let im_ov = frac_overlap(cur.im_win, cand.im_win);
                    if im_ov < params.min_im_overlap_frac {
                        continue;
                    }

                    let score = cand.raw_sum * (0.5 * (rt_ov + im_ov));

                    match best_candidate {
                        None => best_candidate = Some((k, score)),
                        Some((_, best_s)) => {
                            if score > best_s {
                                best_candidate = Some((k, score));
                            }
                        }
                    }
                }

                if let Some((best_k, _)) = best_candidate {
                    // ----------------- SHAPE CHECK vs. ENVELOPE (DOWNWARD) ---------------
                    if let Some(env) = env_opt {
                        let obs_prev = *iso_raw_down.last().unwrap_or(&0.0);
                        let obs_new = good[best_k].raw_sum.max(0.0);

                        // treat downward chain similarly: index along envelope
                        let env_idx = iso_raw_down.len().min(env.len().saturating_sub(1));

                        if env_idx > 0 && env_idx < env.len() {
                            let env_prev = env[env_idx - 1];
                            let env_cur = env[env_idx];

                            if env_cur > env_prev * (1.0 + DIR_EPS_ENV) {
                                // Envelope says: go UP here
                                if obs_new < obs_prev * SLACK_UP {
                                    break 'downward;
                                }
                            } else if env_cur < env_prev * (1.0 - DIR_EPS_ENV) {
                                // Envelope says: go DOWN here
                                if obs_new > obs_prev * SLACK_DOWN {
                                    break 'downward;
                                }
                            }
                        }

                        iso_raw_down.push(obs_new);
                    }

                    chain_down.push(best_k);
                    cur_down = best_k;
                } else {
                    break;
                }
            }

            chain_down.reverse(); // so that m/z increases
            let mut chain: Vec<usize> = chain_down;
            chain.extend(chain_up.into_iter());

            if chain.len() < params.min_members {
                continue;
            }

            // --- spacing penalty (optional) ---------------------------------------
            let mut spacing_penalty = 0.0f32;
            if params.w_spacing_penalty > 0.0 && chain.len() >= 2 {
                let mut spacing_err_sum = 0.0f32;
                let mut spacing_links = 0usize;

                for win in chain.windows(2) {
                    let i = win[0];
                    let j = win[1];
                    let gi = &good[i];
                    let gj = &good[j];

                    let observed = gj.mz_mu - gi.mz_mu;
                    let ideal = delta;
                    let mid_mz = 0.5 * (gi.mz_mu + gj.mz_mu);
                    let tol_da = iso_tolerance_da(mid_mz, params);

                    if tol_da > 0.0 {
                        let err_units = (observed - ideal).abs() / tol_da; // in “tolerances”
                        spacing_err_sum += err_units;
                        spacing_links += 1;
                    }
                }

                if spacing_links > 0 {
                    spacing_penalty = spacing_err_sum / (spacing_links as f32);
                }
            }

            // chain score: sum of raw_sum minus spacing penalty
            let score_raw: f32 = chain.iter().map(|&i| good[i].raw_sum).sum();
            let score = if params.w_spacing_penalty > 0.0 {
                score_raw - params.w_spacing_penalty * spacing_penalty
            } else {
                score_raw
            };

            if score > best_score {
                best_score = score;
                best_chain = chain;
                best_z = z;
            }
        }

        if best_chain.len() < params.min_members || best_z == 0 {
            continue;
        }

        // --- Averagine cosine gating (before assignment) -----------------
        let mut cos_averagine = 0.0f32;
        if let Some(lut_ref) = lut {
            // Build iso_raw = [raw_sum_0, raw_sum_1, ...] in m/z order (chain is monotonic)
            let mut mz_mono_tmp = f32::INFINITY;
            let mut iso_raw: Vec<f32> = Vec::with_capacity(best_chain.len());

            for &idx in &best_chain {
                let g = &good[idx];
                mz_mono_tmp = mz_mono_tmp.min(g.mz_mu);
                iso_raw.push(g.raw_sum.max(0.0));
            }

            if !mz_mono_tmp.is_finite() || mz_mono_tmp <= params.min_mz {
                continue;
            }

            let neutral_tmp = (mz_mono_tmp - 1.007_276_5_f32).max(0.0) * (best_z as f32);
            cos_averagine = cosine_iso_lut(neutral_tmp, best_z, &iso_raw, lut_ref);

            if params.min_cosine > 0.0 && cos_averagine < params.min_cosine {
                // Reject this chain; do NOT mark anything as assigned.
                continue;
            }
        }

        // 4) create feature and mark members as assigned
        for &idx in &best_chain {
            assigned[idx] = true;
        }

        // derive mz_mono as the smallest m/z in the chain (now can be < seed for high z)
        let mut mz_mono = f32::INFINITY;
        let mut mz_weighted = 0.0_f64;
        let mut wsum = 0.0_f64;
        let mut rt_min = usize::MAX;
        let mut rt_max = 0usize;
        let mut im_min = usize::MAX;
        let mut im_max = 0usize;
        let mut raw_sum_total = 0.0_f32;

        let mut member_cluster_indices: Vec<usize> =
            Vec::with_capacity(best_chain.len());
        let mut member_cluster_ids: Vec<u64> =
            Vec::with_capacity(best_chain.len());

        for &idx in &best_chain {
            let g = &good[idx];
            mz_mono = mz_mono.min(g.mz_mu);
            let w = g.raw_sum.max(1.0) as f64;
            mz_weighted += (g.mz_mu as f64) * w;
            wsum += w;
            rt_min = rt_min.min(g.rt_win.0);
            rt_max = rt_max.max(g.rt_win.1);
            im_min = im_min.min(g.im_win.0);
            im_max = im_max.max(g.im_win.1);
            raw_sum_total += g.raw_sum;

            member_cluster_indices.push(g.orig_idx);
            member_cluster_ids.push(g.cluster_id);
        }

        if !mz_mono.is_finite() || mz_mono <= params.min_mz {
            continue;
        }
        let mz_center = if wsum > 0.0 {
            (mz_weighted / wsum) as f32
        } else {
            mz_mono
        };

        let neutral_mass = (mz_mono - 1.007_276_5_f32).max(0.0) * (best_z as f32);

        // If we had no LUT, cos_averagine is 0.0; otherwise it’s already set.
        let feat_id = features.len();
        features.push(SimpleFeature {
            feature_id: feat_id,
            charge: best_z,
            mz_mono,
            neutral_mass,
            rt_bounds: (rt_min, rt_max),
            im_bounds: (im_min, im_max),
            mz_center,
            n_members: best_chain.len(),
            member_cluster_indices,
            member_cluster_ids,
            raw_sum: raw_sum_total,
            cos_averagine,
            member_clusters: best_chain
                .iter()
                .map(|&i| clusters[good[i].orig_idx].clone())
                .collect(),
        });
    }

    features
}