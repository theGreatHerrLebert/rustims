use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;

use crate::cluster::cluster::ClusterResult1D;
use crate::cluster::feature::SimpleFeature;
use crate::cluster::io::load_parquet;
use crate::cluster::pseudo::{cluster_mz_mu, PseudoFragment};
use crate::cluster::scoring::{
    query_precursor_scored, query_precursors_scored_par, MatchScoreMode, PrecursorLike, ScoredHit,
    XicScoreOpts,
};
use crate::data::dia::{DiaIndex};

pub use crate::cluster::pseudo::{
    PseudoBuildResult,
    build_pseudo_spectra_all_pairs,
    build_pseudo_spectra_end_to_end,
    build_pseudo_spectra_end_to_end_xic,
};

// ---------------------------------------------------------------------------
// Candidate / query knobs
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Scoring model for geometric assignment (PUBLIC API expected by scoring.rs)
// ---------------------------------------------------------------------------

/// Compact feature bundle per MS1–MS2 pair.
#[derive(Clone, Copy, Debug)]
pub struct PairFeatures {
    pub jacc_rt: f32,
    pub rt_apex_delta_s: f32,
    pub im_apex_delta_scans: f32,
    pub im_overlap_scans: u32,
    pub im_union_scans: u32,
    pub ms1_raw_sum: f32,
    pub shape_ok: bool,
    pub z_rt: f32,
    pub z_im: f32,
    pub s_shape: f32,
}

#[derive(Clone, Debug)]
pub struct ScoreOpts {
    pub w_jacc_rt: f32,
    pub w_shape: f32,
    pub w_rt_apex: f32,
    pub w_im_apex: f32,
    pub w_im_overlap: f32,
    pub w_ms1_intensity: f32,

    pub rt_apex_scale_s: f32,
    pub im_apex_scale_scans: f32,

    pub shape_neutral: f32,

    pub min_sigma_rt: f32,
    pub min_sigma_im: f32,

    pub w_shape_rt_inner: f32,
    pub w_shape_im_inner: f32,
}

impl Default for ScoreOpts {
    fn default() -> Self {
        Self {
            w_jacc_rt: 1.0,
            w_shape: 1.0,
            w_rt_apex: 0.75,
            w_im_apex: 0.75,
            w_im_overlap: 0.5,
            w_ms1_intensity: 0.25,
            rt_apex_scale_s: 0.75,
            im_apex_scale_scans: 3.0,
            shape_neutral: 0.6,
            min_sigma_rt: 0.05,
            min_sigma_im: 0.5,
            w_shape_rt_inner: 1.0,
            w_shape_im_inner: 1.0,
        }
    }
}

#[inline]
fn im_overlap_and_union(a: (usize, usize), b: (usize, usize)) -> (u32, u32) {
    let lo = a.0.max(b.0);
    let hi = a.1.min(b.1);
    let inter = if hi >= lo { (hi - lo + 1) as u32 } else { 0 };
    let a_len = if a.1 >= a.0 { (a.1 - a.0 + 1) as u32 } else { 0 };
    let b_len = if b.1 >= b.0 { (b.1 - b.0 + 1) as u32 } else { 0 };
    let union = a_len + b_len - inter;
    (inter, union.max(1))
}

#[inline]
fn pooled_sigma(s1: f32, s2: f32) -> Option<f32> {
    let v = s1 * s1 + s2 * s2;
    if v.is_finite() && v > 0.0 { Some(v.sqrt()) } else { None }
}

#[inline]
pub fn exp_decay(delta: f32, scale: f32) -> f32 {
    if !delta.is_finite() || !scale.is_finite() || scale <= 0.0 { return 0.0; }
    (-delta / scale).exp()
}

#[inline]
pub fn safe_log1p(x: f32) -> f32 {
    if x.is_finite() && x >= 0.0 { (1.0 + x as f64).ln() as f32 } else { 0.0 }
}

#[inline]
pub fn build_features(ms1: &ClusterResult1D,
                      ms2: &ClusterResult1D,
                      opts: &ScoreOpts) -> PairFeatures {
    // RT bounds from mu±3σ (assumes your mu is in seconds already)
    let (rt1_lo, rt1_hi) = (
        ms1.rt_fit.mu as f64 - (ms1.rt_fit.sigma as f64) * 3.0,
        ms1.rt_fit.mu as f64 + (ms1.rt_fit.sigma as f64) * 3.0,
    );
    let (rt2_lo, rt2_hi) = (
        ms2.rt_fit.mu as f64 - (ms2.rt_fit.sigma as f64) * 3.0,
        ms2.rt_fit.mu as f64 + (ms2.rt_fit.sigma as f64) * 3.0,
    );

    let jacc_rt = crate::cluster::scoring::jaccard_time(rt1_lo, rt1_hi, rt2_lo, rt2_hi).clamp(0.0, 1.0);

    let rt_apex_delta_s = (ms1.rt_fit.mu - ms2.rt_fit.mu).abs();
    let im_apex_delta_scans = (ms1.im_fit.mu - ms2.im_fit.mu).abs();

    let (im_inter, im_union) = im_overlap_and_union(ms1.im_window, ms2.im_window);

    let s1_rt = ms1.rt_fit.sigma.max(opts.min_sigma_rt);
    let s2_rt = ms2.rt_fit.sigma.max(opts.min_sigma_rt);
    let s1_im = ms1.im_fit.sigma.max(opts.min_sigma_im);
    let s2_im = ms2.im_fit.sigma.max(opts.min_sigma_im);

    let (mut shape_ok, mut z_rt, mut z_im, mut s_shape) = (false, 0.0, 0.0, 0.0);
    if let (Some(sig_rt), Some(sig_im)) = (pooled_sigma(s1_rt, s2_rt), pooled_sigma(s1_im, s2_im)) {
        if sig_rt.is_finite() && sig_rt > 0.0 && sig_im.is_finite() && sig_im > 0.0 {
            z_rt = rt_apex_delta_s / sig_rt;
            z_im = im_apex_delta_scans / sig_im;
            let q = -0.5_f32 * (opts.w_shape_rt_inner * z_rt * z_rt
                + opts.w_shape_im_inner * z_im * z_im);
            s_shape = q.exp();
            shape_ok = s_shape.is_finite();
        }
    }

    PairFeatures {
        jacc_rt,
        rt_apex_delta_s,
        im_apex_delta_scans,
        im_overlap_scans: im_inter,
        im_union_scans: im_union,
        ms1_raw_sum: ms1.raw_sum,
        shape_ok,
        z_rt,
        z_im,
        s_shape,
    }
}

#[derive(Clone, Debug)]
pub struct CandidateOpts {
    pub min_rt_jaccard: f32,
    pub ms2_rt_guard_sec: f64,
    pub rt_bucket_width: f64,
    pub max_ms1_rt_span_sec: Option<f64>,
    pub max_ms2_rt_span_sec: Option<f64>,
    pub min_raw_sum: f32,

    pub max_rt_apex_delta_sec: Option<f32>,
    pub max_scan_apex_delta: Option<usize>,
    pub min_im_overlap_scans: usize,

    pub reject_frag_inside_precursor_tile: bool,
}

impl Default for CandidateOpts {
    fn default() -> Self {
        Self {
            min_rt_jaccard: 0.0,
            ms2_rt_guard_sec: 0.0,
            rt_bucket_width: 1.0,
            max_ms1_rt_span_sec: Some(60.0),
            max_ms2_rt_span_sec: Some(60.0),
            min_raw_sum: 1.0,

            max_rt_apex_delta_sec: Some(2.0),
            max_scan_apex_delta: Some(6),
            min_im_overlap_scans: 1,

            reject_frag_inside_precursor_tile: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FragmentGroupIndex {
    pub ms2_indices: Vec<usize>, // indices into ms2_storage
    pub rt_apex: Vec<f32>,       // same order as ms2_indices
}

#[derive(Clone, Debug)]
pub struct FragmentQueryOpts {
    pub max_rt_apex_delta_sec: Option<f32>,
    pub max_scan_apex_delta: Option<usize>,
    pub min_im_overlap_scans: usize,
    pub require_tile_compat: bool,
    pub reject_frag_inside_precursor_tile: bool,
}

impl Default for FragmentQueryOpts {
    fn default() -> Self {
        Self {
            max_rt_apex_delta_sec: Some(2.0),
            max_scan_apex_delta: Some(6),
            min_im_overlap_scans: 1,
            require_tile_compat: true,
            reject_frag_inside_precursor_tile: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ThinPrecursor {
    pub mz_mu: f32,
    pub rt_mu: f32, // seconds
    pub im_mu: f32, // scan space
    pub im_window: (usize, usize),
}

// ---------------------------------------------------------------------------
// SlimCluster - minimal representation for streaming index (~52 bytes)
// ---------------------------------------------------------------------------

/// Minimal cluster representation for StreamingFragmentIndex queries.
/// Contains only the fields needed for candidate enumeration and scoring.
/// Size: ~52 bytes vs ~280 bytes for full ClusterResult1D.
#[derive(Clone, Debug)]
pub struct SlimCluster {
    pub cluster_id: u64,           // 8 bytes - for id_to_idx lookup
    pub window_group: Option<u32>, // 8 bytes - filtering by DIA window
    pub ms_level: u8,              // 1 byte - filtering (MS1 vs MS2)
    pub rt_mu: f32,                // 4 bytes - RT apex in seconds
    pub im_mu: f32,                // 4 bytes - IM apex in scan space
    pub mz_mu: f32,                // 4 bytes - m/z apex
    pub rt_sigma: f32,             // 4 bytes - RT width for scoring
    pub im_sigma: f32,             // 4 bytes - IM width for scoring
    pub im_window: (u16, u16),     // 4 bytes - IM bounds (scans rarely > 65k)
    pub raw_sum: f32,              // 4 bytes - intensity for filtering
}

/// Reference to locate a cluster in original parquet files for on-demand loading.
/// Only needed if full cluster data is required (e.g., XIC traces for scoring).
#[derive(Clone, Debug)]
pub struct ClusterFileRef {
    pub file_index: u16,    // Index into file path list (supports up to 65k files)
    pub row_offset: u32,    // Row index within that file
}

// ---------------------------------------------------------------------------
// ClusterResult1D -> SlimCluster conversion
// ---------------------------------------------------------------------------

impl ClusterResult1D {
    /// Convert to SlimCluster for memory-efficient storage.
    ///
    /// SlimCluster contains only the fields needed for candidate enumeration
    /// and geometric scoring (~52 bytes vs ~280 bytes for full ClusterResult1D).
    #[inline]
    pub fn to_slim(&self) -> SlimCluster {
        SlimCluster {
            cluster_id: self.cluster_id,
            window_group: self.window_group,
            ms_level: self.ms_level,
            rt_mu: self.rt_fit.mu,
            im_mu: self.im_fit.mu,
            mz_mu: self.mz_fit.as_ref().map(|f| f.mu).unwrap_or_else(|| {
                // Fallback to mz_window center if no mz_fit
                self.mz_window
                    .map(|(lo, hi)| 0.5 * (lo + hi))
                    .unwrap_or(0.0)
            }),
            rt_sigma: self.rt_fit.sigma,
            im_sigma: self.im_fit.sigma,
            im_window: (self.im_window.0 as u16, self.im_window.1 as u16),
            raw_sum: self.raw_sum,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: load all parquet files from a directory
// ---------------------------------------------------------------------------

/// Load all parquet files from a directory into full ClusterResult1D structs.
fn load_all_parquet_full(dir: &Path) -> io::Result<Vec<ClusterResult1D>> {
    let mut all: Vec<ClusterResult1D> = Vec::new();

    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("parquet") {
            continue;
        }
        paths.push(path);
    }
    paths.sort();

    for path in paths {
        let path_str = path
            .to_str()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "non-UTF8 path"))?;

        let mut part = load_parquet(path_str)?;
        all.append(&mut part);
    }

    Ok(all)
}

// ---------------------------------------------------------------------------
// FragmentIndex - Hybrid storage for memory efficiency
// ---------------------------------------------------------------------------

/// Fragment index for matching precursors to MS2 fragment clusters.
///
/// Uses a hybrid storage strategy:
/// - **Primary**: `ms2_slim` - SlimCluster (~52 bytes/cluster) for indexing and geometric scoring
/// - **Optional**: `ms2_full` - Full ClusterResult1D (~280 bytes/cluster) for XIC scoring
/// - **Lazy load**: `parquet_dir` - Path to parquet files for on-demand full data loading
///
/// This reduces RAM by ~80% compared to storing full ClusterResult1D for all clusters.
/// For 25M clusters: ~7GB → ~1.3GB when using slim-only storage.
#[derive(Clone, Debug)]
pub struct FragmentIndex {
    dia_index: Arc<DiaIndex>,

    // PRIMARY: Slim storage for indexing (~52 bytes/cluster)
    ms2_slim: Vec<SlimCluster>,

    // OPTIONAL: Full data for XIC scoring (lazy-loaded or explicitly set)
    ms2_full: Option<Arc<[ClusterResult1D]>>,

    // OPTIONAL: Parquet directory for on-demand full data loading
    parquet_dir: Option<PathBuf>,

    // Precomputed index structures
    ms2_keep: Vec<bool>,

    // group -> (rt-sorted ms2 indices, rt_apex values)
    by_group: HashMap<u32, FragmentGroupIndex>,

    // cluster_id -> index into ms2_slim
    id_to_idx: HashMap<u64, usize>,
}

impl FragmentIndex {
    // -------------------------------------------------------------------------
    // Accessor methods for slim data (always available, fast)
    // -------------------------------------------------------------------------

    /// Get slim cluster by index (always available, fast, no I/O)
    #[inline]
    pub fn get_slim(&self, idx: usize) -> Option<&SlimCluster> {
        self.ms2_slim.get(idx)
    }

    /// Get slim cluster by cluster_id (always available, fast, no I/O)
    #[inline]
    pub fn get_slim_by_id(&self, cid: u64) -> Option<&SlimCluster> {
        self.id_to_idx.get(&cid).and_then(|&i| self.ms2_slim.get(i))
    }

    /// Number of MS2 clusters in the index
    #[inline]
    pub fn len(&self) -> usize {
        self.ms2_slim.len()
    }

    /// Check if the index is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ms2_slim.is_empty()
    }

    // -------------------------------------------------------------------------
    // Accessor methods for full data (optional, may require loading)
    // -------------------------------------------------------------------------

    /// Get full cluster data slice (only available if `has_full_data()` is true)
    #[inline]
    pub fn ms2_slice(&self) -> Option<&[ClusterResult1D]> {
        self.ms2_full.as_ref().map(|arc| arc.as_ref())
    }

    /// Get full cluster by index (only available if `has_full_data()` is true)
    #[inline]
    pub fn get_ms2_by_index(&self, idx: usize) -> Option<&ClusterResult1D> {
        self.ms2_full.as_ref().and_then(|arc| arc.get(idx))
    }

    /// Get full cluster by cluster_id (only available if `has_full_data()` is true)
    #[inline]
    pub fn get_ms2_by_cluster_id(&self, cid: u64) -> Option<&ClusterResult1D> {
        self.id_to_idx.get(&cid).and_then(|&i| self.get_ms2_by_index(i))
    }

    /// Check if full cluster data is loaded (needed for XIC scoring)
    #[inline]
    pub fn has_full_data(&self) -> bool {
        self.ms2_full.is_some()
    }

    /// Check if parquet path is set for lazy loading
    #[inline]
    pub fn can_load_full_data(&self) -> bool {
        self.parquet_dir.is_some()
    }

    /// Load full data from parquet directory (if path is set).
    /// After calling this, `has_full_data()` will return true.
    pub fn load_full_data(&mut self) -> io::Result<()> {
        if self.ms2_full.is_some() {
            return Ok(()); // Already loaded
        }

        let dir = self.parquet_dir.as_ref()
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::NotFound,
                "No parquet directory set for lazy loading"
            ))?;

        // Load full clusters from parquet
        let full_clusters = load_all_parquet_full(dir)?;
        self.ms2_full = Some(Arc::from(full_clusters.into_boxed_slice()));
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Internal builder from slim data
    // -------------------------------------------------------------------------

    /// Build index from slim cluster data.
    /// This is the core builder used by all constructors.
    fn build_from_slim(
        dia_index: Arc<DiaIndex>,
        ms2_slim: Vec<SlimCluster>,
        ms2_full: Option<Arc<[ClusterResult1D]>>,
        parquet_dir: Option<PathBuf>,
        opts: &CandidateOpts,
    ) -> Self {
        // 1) Build keep mask from slim data (parallel for large datasets)
        let ms2_keep: Vec<bool> = ms2_slim
            .par_iter()
            .map(|c| {
                if c.ms_level != 2 { return false; }
                if c.window_group.is_none() { return false; }
                if c.raw_sum < opts.min_raw_sum { return false; }
                if !c.rt_mu.is_finite() || c.rt_mu <= 0.0 { return false; }
                true
            })
            .collect();

        // 2) Build by_group and id_to_idx in single pass
        let mut by_group_raw: HashMap<u32, Vec<(f32, usize)>> = HashMap::new();
        let mut id_to_idx: HashMap<u64, usize> = HashMap::with_capacity(ms2_slim.len());

        for (j, c) in ms2_slim.iter().enumerate() {
            // Always build id_to_idx
            id_to_idx.insert(c.cluster_id, j);

            // Only index valid clusters
            if !ms2_keep[j] { continue; }

            let g = match c.window_group {
                Some(g) => g,
                None => continue,
            };

            let rt = c.rt_mu;
            if !rt.is_finite() { continue; }

            by_group_raw.entry(g).or_default().push((rt, j));
        }

        // 3) Sort each group by RT (for binary search during queries)
        let mut by_group: HashMap<u32, FragmentGroupIndex> = HashMap::with_capacity(by_group_raw.len());
        for (g, mut v) in by_group_raw {
            v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let (rt_apex, ms2_indices): (Vec<f32>, Vec<usize>) = v.into_iter().unzip();
            by_group.insert(g, FragmentGroupIndex { ms2_indices, rt_apex });
        }

        Self {
            dia_index,
            ms2_slim,
            ms2_full,
            parquet_dir,
            ms2_keep,
            by_group,
            id_to_idx,
        }
    }

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /// Create from a single parquet file.
    /// Uses slim storage by default (low RAM). Call `load_full_data()` for XIC scoring.
    pub fn from_parquet_file(dia_index: Arc<DiaIndex>, parquet_path: impl AsRef<Path>, opts: &CandidateOpts) -> io::Result<Self> {
        let path_str = parquet_path
            .as_ref()
            .to_str()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "non-UTF8 path"))?;

        let clusters = load_parquet(path_str)?;
        Ok(Self::from_owned(dia_index, clusters, opts))
    }

    /// Create from a directory of parquet files.
    /// Uses slim storage by default (low RAM). Call `load_full_data()` for XIC scoring.
    pub fn from_parquet_dir(dia_index: Arc<DiaIndex>, dir: impl AsRef<Path>, opts: &CandidateOpts) -> io::Result<Self> {
        let dir_path = dir.as_ref().to_path_buf();
        let all = load_all_parquet_full(&dir_path)?;
        Ok(Self::from_owned_with_full(dia_index, all, Some(dir_path), opts))
    }

    /// Create from a borrowed slice of clusters.
    /// Uses slim storage by default (low RAM). Full data is NOT retained.
    pub fn from_slice(dia_index: Arc<DiaIndex>, ms2: &[ClusterResult1D], opts: &CandidateOpts) -> Self {
        let ms2_slim: Vec<SlimCluster> = ms2.iter().map(|c| c.to_slim()).collect();
        Self::build_from_slim(dia_index, ms2_slim, None, None, opts)
    }

    /// Create from owned clusters (slim storage only, drops full data).
    /// Uses slim storage by default (low RAM). Full data is NOT retained.
    /// For XIC scoring, use `from_owned_with_full()` instead.
    pub fn from_owned(dia_index: Arc<DiaIndex>, ms2: Vec<ClusterResult1D>, opts: &CandidateOpts) -> Self {
        let ms2_slim: Vec<SlimCluster> = ms2.iter().map(|c| c.to_slim()).collect();
        Self::build_from_slim(dia_index, ms2_slim, None, None, opts)
    }

    /// Create from owned clusters, retaining full data for XIC scoring.
    /// Higher RAM usage but enables XIC scoring without loading from parquet.
    pub fn from_owned_with_full(
        dia_index: Arc<DiaIndex>,
        ms2: Vec<ClusterResult1D>,
        parquet_dir: Option<PathBuf>,
        opts: &CandidateOpts,
    ) -> Self {
        let ms2_slim: Vec<SlimCluster> = ms2.iter().map(|c| c.to_slim()).collect();
        let ms2_full = Arc::from(ms2.into_boxed_slice());
        Self::build_from_slim(dia_index, ms2_slim, Some(ms2_full), parquet_dir, opts)
    }

    /// Load from parquet directory using streaming (lowest RAM).
    ///
    /// This method:
    /// 1. Streams parquet files row-group by row-group (minimal peak RAM)
    /// 2. Stores only SlimCluster data (~52 bytes/cluster instead of ~280)
    /// 3. Stores parquet path for lazy loading via `load_full_data()`
    ///
    /// Note: XIC scoring requires calling `load_full_data()` first.
    pub fn from_parquet_dir_slim(
        dia_index: Arc<DiaIndex>,
        dir: impl AsRef<Path>,
        opts: &CandidateOpts,
    ) -> io::Result<Self> {
        use crate::cluster::io::load_parquet_slim_streaming;

        let dir_path = dir.as_ref().to_path_buf();
        let mut all_slim: Vec<SlimCluster> = Vec::new();

        // Collect and sort parquet files for deterministic ordering
        let mut paths: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(&dir_path)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("parquet") {
                continue;
            }
            paths.push(path);
        }
        paths.sort();

        // Stream each file
        for path in paths {
            let path_str = path
                .to_str()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "non-UTF8 path"))?;

            // Stream load slim clusters (row-group at a time)
            let slim_batch = load_parquet_slim_streaming(path_str)?;
            all_slim.extend(slim_batch);
        }

        // Store parquet path for lazy loading of full data
        Ok(Self::build_from_slim(
            dia_index,
            all_slim,
            None,
            Some(dir_path),
            opts,
        ))
    }

    fn precursor_rt_apex_sec(&self, prec: &ClusterResult1D) -> Option<f32> {
        let mu = prec.rt_fit.mu;
        if mu.is_finite() && mu > 0.0 {
            return Some(mu);
        }

        let ft = &self.dia_index.frame_time;

        let mut sum = 0.0f64;
        let mut n = 0usize;
        for fid in &prec.frame_ids_used {
            if let Some(&t) = ft.get(fid) {
                if t.is_finite() {
                    sum += t;
                    n += 1;
                }
            }
        }
        if n > 0 {
            return Some((sum / n as f64) as f32);
        }

        let (rt_lo, rt_hi) = prec.rt_window;
        if rt_hi >= rt_lo {
            let mut sum = 0.0f64;
            let mut n = 0usize;
            for fid in rt_lo as u32..=rt_hi as u32 {
                if let Some(&t) = ft.get(&fid) {
                    if t.is_finite() {
                        sum += t;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                return Some((sum / n as f64) as f32);
            }
        }

        None
    }

    pub fn make_thin_precursor(&self, prec: &ClusterResult1D) -> Option<ThinPrecursor> {
        let prec_mz = if let Some(m) = cluster_mz_mu(prec) {
            if m.is_finite() && m > 0.0 {
                m
            } else {
                match prec.mz_window {
                    Some((lo, hi)) if lo.is_finite() && hi.is_finite() && hi > lo => 0.5 * (lo + hi),
                    _ => return None,
                }
            }
        } else {
            match prec.mz_window {
                Some((lo, hi)) if lo.is_finite() && hi.is_finite() && hi > lo => 0.5 * (lo + hi),
                _ => return None,
            }
        };

        let rt_mu = self.precursor_rt_apex_sec(prec)?;
        let im_window = prec.im_window;

        let im_mu = if prec.im_fit.mu.is_finite() {
            prec.im_fit.mu
        } else {
            let (lo, hi) = im_window;
            if hi > lo { ((lo + hi) as f32) * 0.5 } else { return None; }
        };

        Some(ThinPrecursor {
            mz_mu: prec_mz,
            rt_mu,
            im_mu,
            im_window,
        })
    }

    pub fn enumerate_candidates_for_precursor_thin(
        &self,
        prec: &ThinPrecursor,
        window_groups: Option<&[u32]>,
        opts: &FragmentQueryOpts,
    ) -> Vec<usize> {
        let mut out = Vec::<usize>::new();

        let prec_mz = prec.mz_mu;
        let prec_rt = prec.rt_mu;
        let prec_im = prec.im_mu;
        let prec_im_win = prec.im_window;
        let prec_scan_win = (prec_im_win.0 as u32, prec_im_win.1 as u32);

        let groups: Vec<u32> = match window_groups {
            Some(gs) if !gs.is_empty() => gs.to_vec(),
            _ => self.dia_index.groups_for_precursor(prec_mz, prec_im),
        };
        if groups.is_empty() {
            return out;
        }

        let require_tile = opts.require_tile_compat;
        let reject_inside = opts.reject_frag_inside_precursor_tile;

        for g in groups {
            let fg = match self.by_group.get(&g) {
                Some(fg) => fg,
                None => continue,
            };

            let (lo_idx, hi_idx) = if let Some(dt) = opts.max_rt_apex_delta_sec {
                if dt > 0.0 {
                    let lo_t = prec_rt - dt;
                    let hi_t = prec_rt + dt;
                    (lower_bound(&fg.rt_apex, lo_t), upper_bound(&fg.rt_apex, hi_t))
                } else {
                    (0, fg.ms2_indices.len())
                }
            } else {
                (0, fg.ms2_indices.len())
            };

            let slices = self.dia_index.program_slices_for_group(g);

            'ms2_loop: for k in lo_idx..hi_idx {
                let j = fg.ms2_indices[k];
                if !self.ms2_keep[j] {
                    continue;
                }

                let slim = &self.ms2_slim[j];
                let im2 = (slim.im_window.0 as usize, slim.im_window.1 as usize);
                let frag_scan_win = (slim.im_window.0 as u32, slim.im_window.1 as u32);

                let im_overlap = {
                    let lo = prec_im_win.0.max(im2.0);
                    let hi = prec_im_win.1.min(im2.1);
                    hi.saturating_sub(lo).saturating_add(1)
                };
                if im_overlap < opts.min_im_overlap_scans {
                    continue;
                }

                if let Some(max_d) = opts.max_scan_apex_delta {
                    let s2 = slim.im_mu;
                    if s2.is_finite() {
                        let d = (prec_im - s2).abs();
                        if d > max_d as f32 {
                            continue 'ms2_loop;
                        }
                    } else {
                        continue 'ms2_loop;
                    }
                }

                if require_tile || reject_inside {
                    let mut tile_compatible = false;
                    let mut inside_prec_tile = false;
                    let frag_mz = slim.mz_mu;

                    for s in &slices {
                        let tile_scans = (s.scan_lo, s.scan_hi);

                        let prec_hits_tile = ranges_overlap_u32(prec_scan_win, tile_scans);
                        let frag_hits_tile = ranges_overlap_u32(frag_scan_win, tile_scans);
                        if !(prec_hits_tile && frag_hits_tile) {
                            continue;
                        }

                        tile_compatible = true;

                        if reject_inside && frag_mz.is_finite() {
                            let prec_in_tile = prec_mz >= s.mz_lo as f32 && prec_mz <= s.mz_hi as f32;
                            let frag_in_tile = frag_mz >= s.mz_lo as f32 && frag_mz <= s.mz_hi as f32;
                            if prec_in_tile && frag_in_tile {
                                inside_prec_tile = true;
                                break;
                            }
                        }
                    }

                    if require_tile && !tile_compatible {
                        continue 'ms2_loop;
                    }
                    if reject_inside && inside_prec_tile {
                        continue 'ms2_loop;
                    }
                }

                out.push(j);
            }
        }

        out.sort_unstable();
        out.dedup();
        out
    }

    /// Score candidate fragments for a precursor.
    ///
    /// Requires full cluster data to be loaded. If using slim-only index,
    /// call `load_full_data()` first.
    ///
    /// Returns empty Vec if full data is not available.
    pub fn query_precursor_scored(
        &self,
        prec: &ClusterResult1D,
        window_groups: Option<&[u32]>,
        opts: &FragmentQueryOpts,
        mode: MatchScoreMode,
        geom_opts: &ScoreOpts,
        xic_opts: &XicScoreOpts,
        min_score: f32,
    ) -> Vec<ScoredHit> {
        // Require full data for scoring
        let ms2_data = match self.ms2_slice() {
            Some(data) => data,
            None => return Vec::new(), // No full data available
        };

        let thin = match self.make_thin_precursor(prec) {
            Some(t) => t,
            None => return Vec::new(),
        };

        let candidate_ids = self.enumerate_candidates_for_precursor_thin(&thin, window_groups, opts);

        query_precursor_scored(
            PrecursorLike::Cluster(prec),
            ms2_data,
            &candidate_ids,
            mode,
            geom_opts,
            xic_opts,
            min_score,
        )
    }

    /// Score candidate fragments for multiple precursors in parallel.
    ///
    /// Requires full cluster data to be loaded. If using slim-only index,
    /// call `load_full_data()` first.
    ///
    /// Returns empty Vec for each precursor if full data is not available.
    pub fn query_precursors_scored_par(
        &self,
        precs: &[ClusterResult1D],
        opts: &FragmentQueryOpts,
        mode: MatchScoreMode,
        geom_opts: &crate::cluster::scoring::ScoreOpts,
        xic_opts: &XicScoreOpts,
        min_score: f32,
    ) -> Vec<Vec<ScoredHit>> {
        // Require full data for scoring
        let ms2_data = match self.ms2_slice() {
            Some(data) => data,
            None => return vec![Vec::new(); precs.len()], // No full data available
        };

        let thin_precs: Vec<Option<ThinPrecursor>> = precs.iter().map(|c| self.make_thin_precursor(c)).collect();

        let all_candidates: Vec<Vec<usize>> = thin_precs
            .iter()
            .map(|p| p.as_ref().map(|tp| self.enumerate_candidates_for_precursor_thin(tp, None, opts)).unwrap_or_default())
            .collect();

        let prec_like: Vec<PrecursorLike<'_>> = precs.iter().map(|c| PrecursorLike::Cluster(c)).collect();

        query_precursors_scored_par(
            &prec_like,
            ms2_data,
            &all_candidates,
            mode,
            geom_opts,
            xic_opts,
            min_score,
        )
    }

    pub fn enumerate_candidates_for_feature(&self, feat: &SimpleFeature, opts: &FragmentQueryOpts) -> Vec<usize> {
        let prec_cluster = match feature_representative_cluster(feat) {
            Some(c) => c,
            None => return Vec::new(),
        };
        let thin = match self.make_thin_precursor(prec_cluster) {
            Some(t) => t,
            None => return Vec::new(),
        };
        self.enumerate_candidates_for_precursor_thin(&thin, None, opts)
    }

    /// Score candidate fragments for a feature.
    ///
    /// Requires full cluster data to be loaded. If using slim-only index,
    /// call `load_full_data()` first.
    ///
    /// Returns empty Vec if full data is not available.
    pub fn query_feature_scored(
        &self,
        feat: &SimpleFeature,
        opts: &FragmentQueryOpts,
        mode: MatchScoreMode,
        geom_opts: &crate::cluster::scoring::ScoreOpts,
        xic_opts: &XicScoreOpts,
        min_score: f32,
    ) -> Vec<ScoredHit> {
        // Require full data for scoring
        let ms2_data = match self.ms2_slice() {
            Some(data) => data,
            None => return Vec::new(), // No full data available
        };

        let candidate_ids = self.enumerate_candidates_for_feature(feat, opts);

        query_precursor_scored(
            PrecursorLike::Feature(feat),
            ms2_data,
            &candidate_ids,
            mode,
            geom_opts,
            xic_opts,
            min_score,
        )
    }
}

// ---------------------------------------------------------------------------
// StreamingFragmentIndex - memory-efficient version (~52 bytes/cluster)
// ---------------------------------------------------------------------------

/// Memory-efficient FragmentIndex that stores only query-essential data.
/// Uses SlimCluster (~52 bytes) instead of full ClusterResult1D (~280 bytes).
/// Reduces RAM by ~80% for large datasets (e.g., 25M clusters: 7GB → 1.3GB).
#[derive(Clone, Debug)]
pub struct StreamingFragmentIndex {
    dia_index: Arc<DiaIndex>,

    /// Core data: slim clusters only (~52 bytes each)
    ms2_slim: Vec<SlimCluster>,

    /// Keep mask for filtering (1 byte per cluster)
    ms2_keep: Vec<bool>,

    /// Group index: group -> (rt-sorted ms2 indices, rt_apex values)
    by_group: HashMap<u32, FragmentGroupIndex>,

    /// cluster_id -> index into ms2_slim
    id_to_idx: HashMap<u64, usize>,

    /// Optional: file references for on-demand full data loading
    file_refs: Option<Vec<ClusterFileRef>>,
    file_paths: Option<Vec<String>>,
}

impl StreamingFragmentIndex {
    /// Get slim cluster by index (fast, no I/O)
    #[inline]
    pub fn get_slim(&self, idx: usize) -> Option<&SlimCluster> {
        self.ms2_slim.get(idx)
    }

    /// Get slim cluster by cluster_id (fast, no I/O)
    #[inline]
    pub fn get_slim_by_id(&self, cid: u64) -> Option<&SlimCluster> {
        self.id_to_idx.get(&cid).and_then(|&i| self.ms2_slim.get(i))
    }

    /// Number of MS2 clusters in the index
    #[inline]
    pub fn len(&self) -> usize {
        self.ms2_slim.len()
    }

    /// Check if the index is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ms2_slim.is_empty()
    }

    /// Stream-load from a directory of parquet files with minimal RAM.
    /// Processes one file at a time using direct parquet crate (bypasses Polars overhead).
    /// Uses row-group level streaming for minimal peak memory.
    pub fn from_parquet_dir_streaming(
        dia_index: Arc<DiaIndex>,
        dir: impl AsRef<Path>,
        opts: &CandidateOpts,
        store_file_refs: bool,
    ) -> io::Result<Self> {
        // Use the ultra-low-memory streaming reader that bypasses Polars
        use crate::cluster::io::load_parquet_slim_streaming;

        let dir = dir.as_ref();

        // 1. Collect parquet file paths
        let mut file_paths_buf: Vec<std::path::PathBuf> = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() { continue; }
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("parquet") { continue; }
            file_paths_buf.push(path);
        }

        // Sort for deterministic ordering
        file_paths_buf.sort();

        // 2. Stream each file using direct parquet reader (no Polars overhead)
        let mut all_slim: Vec<SlimCluster> = Vec::new();
        let mut all_refs: Vec<ClusterFileRef> = Vec::new();

        for (file_idx, path) in file_paths_buf.iter().enumerate() {
            let path_str = path.to_str()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "non-UTF8 path"))?;

            // Load using row-group streaming (minimal RAM per file)
            let slim_batch = load_parquet_slim_streaming(path_str)?;
            let batch_len = slim_batch.len();

            // Store file references if requested
            if store_file_refs {
                for row_offset in 0..batch_len {
                    all_refs.push(ClusterFileRef {
                        file_index: file_idx as u16,
                        row_offset: row_offset as u32,
                    });
                }
            }

            // Append slim clusters (batch is dropped after this)
            all_slim.extend(slim_batch);
        }

        // 3. Build the index from slim data
        let file_paths_str = if store_file_refs {
            Some(file_paths_buf.iter().map(|p| p.to_string_lossy().into_owned()).collect())
        } else {
            None
        };

        Ok(Self::build_from_slim(
            dia_index,
            all_slim,
            opts,
            if store_file_refs { Some(all_refs) } else { None },
            file_paths_str,
        ))
    }

    /// Build index from slim cluster data in a single pass.
    /// Avoids creating intermediate vectors - builds by_group and id_to_idx directly.
    fn build_from_slim(
        dia_index: Arc<DiaIndex>,
        ms2_slim: Vec<SlimCluster>,
        opts: &CandidateOpts,
        file_refs: Option<Vec<ClusterFileRef>>,
        file_paths: Option<Vec<String>>,
    ) -> Self {
        // 1. Build keep mask (parallel for large datasets)
        let ms2_keep: Vec<bool> = ms2_slim
            .par_iter()
            .map(|c| {
                if c.ms_level != 2 { return false; }
                if c.window_group.is_none() { return false; }
                if c.raw_sum < opts.min_raw_sum { return false; }
                if !c.rt_mu.is_finite() || c.rt_mu <= 0.0 { return false; }
                true
            })
            .collect();

        // 2. Build by_group and id_to_idx in single pass
        let mut by_group_raw: HashMap<u32, Vec<(f32, usize)>> = HashMap::new();
        let mut id_to_idx: HashMap<u64, usize> = HashMap::with_capacity(ms2_slim.len());

        for (j, c) in ms2_slim.iter().enumerate() {
            // Always build id_to_idx
            id_to_idx.insert(c.cluster_id, j);

            // Only index valid clusters
            if !ms2_keep[j] { continue; }

            let g = match c.window_group {
                Some(g) => g,
                None => continue,
            };

            let rt = c.rt_mu;
            if !rt.is_finite() { continue; }

            by_group_raw.entry(g).or_default().push((rt, j));
        }

        // 3. Sort each group by RT (for binary search during queries)
        let mut by_group: HashMap<u32, FragmentGroupIndex> = HashMap::with_capacity(by_group_raw.len());
        for (g, mut v) in by_group_raw {
            v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let (rt_apex, ms2_indices): (Vec<f32>, Vec<usize>) = v.into_iter().unzip();
            by_group.insert(g, FragmentGroupIndex { ms2_indices, rt_apex });
        }

        Self {
            dia_index,
            ms2_slim,
            ms2_keep,
            by_group,
            id_to_idx,
            file_refs,
            file_paths,
        }
    }

    /// Enumerate candidate fragment indices for a precursor.
    /// Uses slim data only - no I/O required.
    pub fn enumerate_candidates_for_precursor_thin(
        &self,
        prec: &ThinPrecursor,
        window_groups: Option<&[u32]>,
        opts: &FragmentQueryOpts,
    ) -> Vec<usize> {
        let mut out = Vec::<usize>::new();

        let prec_mz = prec.mz_mu;
        let prec_rt = prec.rt_mu;
        let prec_im = prec.im_mu;
        let prec_im_win = prec.im_window;
        let prec_scan_win = (prec_im_win.0 as u32, prec_im_win.1 as u32);

        // Determine which groups to search
        let groups: Vec<u32> = match window_groups {
            Some(gs) if !gs.is_empty() => gs.to_vec(),
            _ => self.dia_index.groups_for_precursor(prec_mz, prec_im),
        };

        if groups.is_empty() { return out; }

        let require_tile = opts.require_tile_compat;
        let reject_inside = opts.reject_frag_inside_precursor_tile;

        for g in groups {
            let fg = match self.by_group.get(&g) {
                Some(fg) => fg,
                None => continue,
            };

            // Binary search for RT window
            let (lo_idx, hi_idx) = if let Some(dt) = opts.max_rt_apex_delta_sec {
                if dt > 0.0 {
                    let lo_t = prec_rt - dt;
                    let hi_t = prec_rt + dt;
                    (lower_bound(&fg.rt_apex, lo_t), upper_bound(&fg.rt_apex, hi_t))
                } else {
                    (0, fg.ms2_indices.len())
                }
            } else {
                (0, fg.ms2_indices.len())
            };

            let slices = self.dia_index.program_slices_for_group(g);

            'ms2_loop: for k in lo_idx..hi_idx {
                let j = fg.ms2_indices[k];
                if !self.ms2_keep[j] { continue; }

                let slim = &self.ms2_slim[j];

                // IM window overlap check using slim data
                let im2 = (slim.im_window.0 as usize, slim.im_window.1 as usize);
                let im_overlap = {
                    let lo = prec_im_win.0.max(im2.0);
                    let hi = prec_im_win.1.min(im2.1);
                    hi.saturating_sub(lo).saturating_add(1)
                };
                if im_overlap < opts.min_im_overlap_scans { continue; }

                // IM apex delta check using slim data
                if let Some(max_d) = opts.max_scan_apex_delta {
                    let d = (prec_im - slim.im_mu).abs();
                    if d > max_d as f32 { continue; }
                }

                // Tile compatibility check (if required)
                if require_tile || reject_inside {
                    let frag_mz = slim.mz_mu;
                    let frag_scan = (slim.im_window.0 as u32, slim.im_window.1 as u32);

                    let mut tile_compatible = false;
                    let mut inside_prec_tile = false;

                    for s in &slices {
                        // Check scan overlap
                        if !ranges_overlap_u32(prec_scan_win, (s.scan_lo, s.scan_hi)) {
                            continue;
                        }
                        if !ranges_overlap_u32(frag_scan, (s.scan_lo, s.scan_hi)) {
                            continue;
                        }

                        tile_compatible = true;

                        // Check if fragment's m/z is inside precursor's isolation tile
                        if reject_inside && frag_mz.is_finite() {
                            let prec_in_tile = prec_mz >= s.mz_lo as f32 && prec_mz <= s.mz_hi as f32;
                            let frag_in_tile = frag_mz >= s.mz_lo as f32 && frag_mz <= s.mz_hi as f32;
                            if prec_in_tile && frag_in_tile {
                                inside_prec_tile = true;
                            }
                        }
                    }

                    if require_tile && !tile_compatible {
                        continue 'ms2_loop;
                    }
                    if reject_inside && inside_prec_tile {
                        continue 'ms2_loop;
                    }
                }

                out.push(j);
            }
        }

        out.sort_unstable();
        out.dedup();
        out
    }
}

// ---------------------------------------------------------------------------
// Helpers (local)
// ---------------------------------------------------------------------------

fn lower_bound(xs: &[f32], x: f32) -> usize {
    let mut lo = 0;
    let mut hi = xs.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if xs[mid] < x { lo = mid + 1; } else { hi = mid; }
    }
    lo
}

fn upper_bound(xs: &[f32], x: f32) -> usize {
    let mut lo = 0;
    let mut hi = xs.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x { lo = mid + 1; } else { hi = mid; }
    }
    lo
}

#[inline]
fn ranges_overlap_u32(a: (u32, u32), b: (u32, u32)) -> bool {
    let lo = a.0.max(b.0);
    let hi = a.1.min(b.1);
    hi >= lo
}

pub fn fragment_from_cluster(c: &ClusterResult1D) -> Option<PseudoFragment> {
    let mz = if let Some(fit) = &c.mz_fit {
        if fit.mu.is_finite() && fit.mu > 0.0 {
            fit.mu
        } else if let Some((lo, hi)) = c.mz_window {
            0.5 * (lo + hi)
        } else {
            return None;
        }
    } else if let Some((lo, hi)) = c.mz_window {
        0.5 * (lo + hi)
    } else {
        return None;
    };

    if !mz.is_finite() {
        return None;
    }

    Some(PseudoFragment {
        mz,
        intensity: c.raw_sum,
        ms2_cluster_index: 0, // if you actually have the index, set it properly
        ms2_cluster_id: c.cluster_id,
        window_group: c.window_group.unwrap_or(0),
    })
}

fn feature_representative_cluster<'a>(feat: &'a SimpleFeature) -> Option<&'a ClusterResult1D> {
    if feat.member_clusters.is_empty() {
        return None;
    }
    feat.member_clusters
        .iter()
        .max_by(|a, b| a.raw_sum.partial_cmp(&b.raw_sum).unwrap_or(std::cmp::Ordering::Equal))
}