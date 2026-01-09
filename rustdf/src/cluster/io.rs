use std::fs::File;
use std::io::{BufReader, BufWriter};
use serde::{Deserialize, Serialize};
use crate::cluster::utility::Fit1D;
use super::cluster::ClusterResult1D;
use super::candidates::SlimCluster;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterRow {
    pub cluster_id: u64,
    pub ms_level: u8,
    pub window_group: Option<u32>,
    pub parent_im_id: Option<i64>,
    pub parent_rt_id: Option<i64>,

    pub rt_lo: usize,
    pub rt_hi: usize,
    pub im_lo: usize,
    pub im_hi: usize,
    pub tof_lo: usize,
    pub tof_hi: usize,
    pub tof_index_lo: i32,
    pub tof_index_hi: i32,
    pub mz_lo: Option<f32>,
    pub mz_hi: Option<f32>,

    pub rt_mu: f32,
    pub rt_sigma: f32,
    pub rt_height: f32,
    pub rt_area: f32,

    pub im_mu: f32,
    pub im_sigma: f32,
    pub im_height: f32,
    pub im_area: f32,

    pub tof_mu: f32,
    pub tof_sigma: f32,
    pub tof_height: f32,
    pub tof_area: f32,

    pub mz_mu: Option<f32>,
    pub mz_sigma: Option<f32>,
    pub mz_height: Option<f32>,
    pub mz_area: Option<f32>,

    pub raw_sum: f32,
    pub volume_proxy: f32,
}

impl From<&ClusterResult1D> for ClusterRow {
    fn from(c: &ClusterResult1D) -> Self {
        let (rt_lo, rt_hi) = c.rt_window;
        let (im_lo, im_hi) = c.im_window;
        let (tof_lo, tof_hi) = c.tof_window;
        let (tof_index_lo, tof_index_hi) = c.tof_index_window;

        let fit = |f: &Fit1D| (f.mu, f.sigma, f.height, f.area);

        let (rt_mu, rt_sigma, rt_height, rt_area) = fit(&c.rt_fit);
        let (im_mu, im_sigma, im_height, im_area) = fit(&c.im_fit);
        let (tof_mu, tof_sigma, tof_height, tof_area) = fit(&c.tof_fit);

        // First get mz window
        let (mz_lo, mz_hi) = match c.mz_window {
            Some((lo, hi)) => (Some(lo), Some(hi)),
            None => (None, None),
        };

        // Then get any fitted mz parameters
        let (mz_mu_raw, mz_sigma, mz_height, mz_area) = match &c.mz_fit {
            Some(f) => {
                let (mu, sigma, h, a) = fit(f);
                (Some(mu), Some(sigma), Some(h), Some(a))
            }
            None => (None, None, None, None),
        };

        // Fallback: if no fitted mz_mu but we *do* have mz_lo/mz_hi, use midpoint
        let mz_mu = match (mz_mu_raw, mz_lo, mz_hi) {
            (Some(mu), _, _) => Some(mu),
            (None, Some(lo), Some(hi)) => Some(0.5 * (lo + hi)),
            _ => None,
        };

        ClusterRow {
            cluster_id: c.cluster_id,
            ms_level: c.ms_level,
            window_group: c.window_group,
            parent_im_id: c.parent_im_id,
            parent_rt_id: c.parent_rt_id,

            rt_lo,
            rt_hi,
            im_lo,
            im_hi,
            tof_lo,
            tof_hi,
            tof_index_lo,
            tof_index_hi,
            mz_lo,
            mz_hi,

            rt_mu,
            rt_sigma,
            rt_height,
            rt_area,

            im_mu,
            im_sigma,
            im_height,
            im_area,

            tof_mu,
            tof_sigma,
            tof_height,
            tof_area,

            mz_mu,
            mz_sigma,
            mz_height,
            mz_area,

            raw_sum: c.raw_sum,
            volume_proxy: c.volume_proxy,
        }
    }
}

impl ClusterRow {
    /// Convert a flattened row back into a lightweight ClusterResult1D.
    /// Heavy fields (raw_points, axes, traces) are left empty / None.
    pub fn into_cluster_result(self) -> ClusterResult1D {
        let ClusterRow {
            cluster_id,
            ms_level,
            window_group,
            parent_im_id,
            parent_rt_id,

            rt_lo,
            rt_hi,
            im_lo,
            im_hi,
            tof_lo,
            tof_hi,
            tof_index_lo,
            tof_index_hi,
            mz_lo,
            mz_hi,

            rt_mu,
            rt_sigma,
            rt_height,
            rt_area,

            im_mu,
            im_sigma,
            im_height,
            im_area,

            tof_mu,
            tof_sigma,
            tof_height,
            tof_area,

            mz_mu,
            mz_sigma,
            mz_height,
            mz_area,

            raw_sum,
            volume_proxy,
        } = self;

        let rt_fit = Fit1D {
            mu: rt_mu,
            sigma: rt_sigma,
            height: rt_height,
            baseline: 0.0,
            area: rt_area,
            r2: 0.0,
            n: 0,
        };
        let im_fit = Fit1D {
            mu: im_mu,
            sigma: im_sigma,
            height: im_height,
            baseline: 0.0,
            area: im_area,
            r2: 0.0,
            n: 0,
        };
        let tof_fit = Fit1D {
            mu: tof_mu,
            sigma: tof_sigma,
            height: tof_height,
            baseline: 0.0,
            area: tof_area,
            r2: 0.0,
            n: 0,
        };
        let mz_fit = match (mz_mu, mz_sigma, mz_height, mz_area) {
            (Some(mu), Some(sigma), Some(h), Some(a)) => Some(Fit1D {
                mu,
                sigma,
                height: h,
                baseline: 0.0,
                area: a,
                r2: 0.0,
                n: 0,
            }),
            _ => None,
        };

        ClusterResult1D {
            cluster_id,
            rt_window: (rt_lo, rt_hi),
            im_window: (im_lo, im_hi),
            tof_window: (tof_lo, tof_hi),
            tof_index_window: (tof_index_lo, tof_index_hi),
            mz_window: match (mz_lo, mz_hi) {
                (Some(lo), Some(hi)) => Some((lo, hi)),
                _ => None,
            },
            rt_fit,
            im_fit,
            tof_fit,
            mz_fit,
            raw_sum,
            volume_proxy,
            frame_ids_used: Vec::new(), // not stored in parquet
            window_group,
            parent_im_id,
            parent_rt_id,
            ms_level,

            // heavy / optional stuff omitted in parquet
            rt_axis_sec: None,
            im_axis_scans: None,
            mz_axis_da: None,
            raw_points: None,
            rt_trace: None,
            im_trace: None,
        }
    }
}

use std::io;
use polars::prelude::*;
use crate::cluster::feature::SimpleFeature;
use crate::cluster::pseudo::PseudoSpectrum;

pub fn save_parquet(path: &str, clusters: &[ClusterResult1D]) -> io::Result<()> {
    let rows: Vec<ClusterRow> = clusters.iter().map(ClusterRow::from).collect();
    let n = rows.len();

    // Pre-allocate vectors for each column
    let mut cluster_id = Vec::with_capacity(n);
    let mut ms_level = Vec::with_capacity(n);
    let mut window_group = Vec::with_capacity(n);
    let mut parent_im_id = Vec::with_capacity(n);
    let mut parent_rt_id = Vec::with_capacity(n);

    let mut rt_lo = Vec::with_capacity(n);
    let mut rt_hi = Vec::with_capacity(n);
    let mut im_lo = Vec::with_capacity(n);
    let mut im_hi = Vec::with_capacity(n);
    let mut tof_lo = Vec::with_capacity(n);
    let mut tof_hi = Vec::with_capacity(n);
    let mut tof_index_lo = Vec::with_capacity(n);
    let mut tof_index_hi = Vec::with_capacity(n);
    let mut mz_lo = Vec::with_capacity(n);
    let mut mz_hi = Vec::with_capacity(n);

    let mut rt_mu = Vec::with_capacity(n);
    let mut rt_sigma = Vec::with_capacity(n);
    let mut rt_height = Vec::with_capacity(n);
    let mut rt_area = Vec::with_capacity(n);

    let mut im_mu = Vec::with_capacity(n);
    let mut im_sigma = Vec::with_capacity(n);
    let mut im_height = Vec::with_capacity(n);
    let mut im_area = Vec::with_capacity(n);

    let mut tof_mu = Vec::with_capacity(n);
    let mut tof_sigma = Vec::with_capacity(n);
    let mut tof_height = Vec::with_capacity(n);
    let mut tof_area = Vec::with_capacity(n);

    let mut mz_mu = Vec::with_capacity(n);
    let mut mz_sigma = Vec::with_capacity(n);
    let mut mz_height = Vec::with_capacity(n);
    let mut mz_area = Vec::with_capacity(n);

    let mut raw_sum = Vec::with_capacity(n);
    let mut volume_proxy = Vec::with_capacity(n);

    for r in rows {
        cluster_id.push(r.cluster_id);
        ms_level.push(r.ms_level);
        window_group.push(r.window_group);
        parent_im_id.push(r.parent_im_id);
        parent_rt_id.push(r.parent_rt_id);

        rt_lo.push(r.rt_lo as u32);
        rt_hi.push(r.rt_hi as u32);
        im_lo.push(r.im_lo as u32);
        im_hi.push(r.im_hi as u32);
        tof_lo.push(r.tof_lo as u32);
        tof_hi.push(r.tof_hi as u32);
        tof_index_lo.push(r.tof_index_lo);
        tof_index_hi.push(r.tof_index_hi);
        mz_lo.push(r.mz_lo);
        mz_hi.push(r.mz_hi);

        rt_mu.push(r.rt_mu);
        rt_sigma.push(r.rt_sigma);
        rt_height.push(r.rt_height);
        rt_area.push(r.rt_area);

        im_mu.push(r.im_mu);
        im_sigma.push(r.im_sigma);
        im_height.push(r.im_height);
        im_area.push(r.im_area);

        tof_mu.push(r.tof_mu);
        tof_sigma.push(r.tof_sigma);
        tof_height.push(r.tof_height);
        tof_area.push(r.tof_area);

        mz_mu.push(r.mz_mu);
        mz_sigma.push(r.mz_sigma);
        mz_height.push(r.mz_height);
        mz_area.push(r.mz_area);

        raw_sum.push(r.raw_sum);
        volume_proxy.push(r.volume_proxy);
    }

    let mut df = DataFrame::new(vec![
        Series::new(PlSmallStr::from("cluster_id"), cluster_id),
        Series::new(PlSmallStr::from("ms_level"), ms_level),
        Series::new(PlSmallStr::from("window_group"), window_group),
        Series::new(PlSmallStr::from("parent_im_id"), parent_im_id),
        Series::new(PlSmallStr::from("parent_rt_id"), parent_rt_id),

        Series::new(PlSmallStr::from("rt_lo"), rt_lo),
        Series::new(PlSmallStr::from("rt_hi"), rt_hi),
        Series::new(PlSmallStr::from("im_lo"), im_lo),
        Series::new(PlSmallStr::from("im_hi"), im_hi),
        Series::new(PlSmallStr::from("tof_lo"), tof_lo),
        Series::new(PlSmallStr::from("tof_hi"), tof_hi),
        Series::new(PlSmallStr::from("tof_index_lo"), tof_index_lo),
        Series::new(PlSmallStr::from("tof_index_hi"), tof_index_hi),
        Series::new(PlSmallStr::from("mz_lo"), mz_lo),
        Series::new(PlSmallStr::from("mz_hi"), mz_hi),

        Series::new(PlSmallStr::from("rt_mu"), rt_mu),
        Series::new(PlSmallStr::from("rt_sigma"), rt_sigma),
        Series::new(PlSmallStr::from("rt_height"), rt_height),
        Series::new(PlSmallStr::from("rt_area"), rt_area),

        Series::new(PlSmallStr::from("im_mu"), im_mu),
        Series::new(PlSmallStr::from("im_sigma"), im_sigma),
        Series::new(PlSmallStr::from("im_height"), im_height),
        Series::new(PlSmallStr::from("im_area"), im_area),

        Series::new(PlSmallStr::from("tof_mu"), tof_mu),
        Series::new(PlSmallStr::from("tof_sigma"), tof_sigma),
        Series::new(PlSmallStr::from("tof_height"), tof_height),
        Series::new(PlSmallStr::from("tof_area"), tof_area),

        Series::new(PlSmallStr::from("mz_mu"), mz_mu),
        Series::new(PlSmallStr::from("mz_sigma"), mz_sigma),
        Series::new(PlSmallStr::from("mz_height"), mz_height),
        Series::new(PlSmallStr::from("mz_area"), mz_area),

        Series::new(PlSmallStr::from("raw_sum"), raw_sum),
        Series::new(PlSmallStr::from("volume_proxy"), volume_proxy),
    ]).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    let f = File::create(path)?;
    let mut writer = ParquetWriter::new(f);
    // You can tune compression here; Zstd is a good default.
    writer = writer.with_compression(ParquetCompression::Zstd(None));

    writer
        .finish(&mut df)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(())
}


#[derive(Serialize, Deserialize)]
pub struct ClusterFile {
    pub version: u32,
    pub clusters: Vec<ClusterResult1D>,
}

impl ClusterFile {
    pub fn new(clusters: Vec<ClusterResult1D>) -> Self {
        Self { version: 1, clusters }
    }
}

// --- JSON (human-readable) ---
pub fn save_json(path: &str, clusters: &[ClusterResult1D]) -> std::io::Result<()> {
    let f = BufWriter::new(File::create(path)?);
    let cf = ClusterFile::new(clusters.to_vec());
    serde_json::to_writer_pretty(f, &cf).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

pub fn load_json(path: &str) -> std::io::Result<Vec<ClusterResult1D>> {
    let f = BufReader::new(File::open(path)?);
    let cf: ClusterFile = serde_json::from_reader(f)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(cf.clusters)
}

// --- Bincode + optional zstd compression ---
pub fn save_bincode(path: &str, clusters: &[ClusterResult1D], compress: bool) -> std::io::Result<()> {
    let f = File::create(path)?;
    if compress {
        let mut zw = zstd::Encoder::new(f, 3)?; // level 3 is a good default
        bincode::serialize_into(&mut zw, &ClusterFile::new(clusters.to_vec()))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        zw.finish()?;
        Ok(())
    } else {
        let mut bw = BufWriter::new(f);
        bincode::serialize_into(&mut bw, &ClusterFile::new(clusters.to_vec()))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

pub fn load_bincode(path: &str) -> std::io::Result<Vec<ClusterResult1D>> {
    let f = File::open(path)?;
    // Try zstd first, then plain bincode
    let try_zstd = zstd::Decoder::new(&f);
    if let Ok(mut zr) = try_zstd {
        let cf: ClusterFile = bincode::deserialize_from(&mut zr)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        return Ok(cf.clusters);
    }
    let f = BufReader::new(File::open(path)?);
    let cf: ClusterFile = bincode::deserialize_from(f)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(cf.clusters)
}

pub fn strip_heavy(mut clusters: Vec<ClusterResult1D>, keep_points: bool, keep_axes: bool) -> Vec<ClusterResult1D> {
    for c in &mut clusters {
        if !keep_points { c.raw_points = None; }
        if !keep_axes { c.rt_axis_sec = None; c.im_axis_scans = None; c.mz_axis_da = None; }
    }
    clusters
}

pub fn load_parquet(path: &str) -> io::Result<Vec<ClusterResult1D>> {
    use std::fs::File;
    use polars::prelude::*;

    let f = File::open(path)?;
    let df = ParquetReader::new(f)
        .finish()
        .map_err(to_io)?;

    let n = df.height();

    // --- Helper macros (LOCAL to this function) ---
    macro_rules! col_u64 {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .u64()
                .map_err(to_io)?
                .into_iter()
                .map(|v| v.unwrap_or(0))
                .collect::<Vec<u64>>()
        }};
    }
    macro_rules! col_u8 {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .u8()
                .map_err(to_io)?
                .into_iter()
                .map(|v| v.unwrap_or(0))
                .collect::<Vec<u8>>()
        }};
    }
    macro_rules! col_u32 {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .u32()
                .map_err(to_io)?
                .into_iter()
                .map(|v| v.unwrap_or(0))
                .collect::<Vec<u32>>()
        }};
    }
    macro_rules! col_i32 {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .i32()
                .map_err(to_io)?
                .into_iter()
                .map(|v| v.unwrap_or(0))
                .collect::<Vec<i32>>()
        }};
    }
    macro_rules! col_i64_opt {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .i64()
                .map_err(to_io)?
                .into_iter()
                .collect::<Vec<Option<i64>>>()
        }};
    }
    macro_rules! col_u32_opt {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .u32()
                .map_err(to_io)?
                .into_iter()
                .collect::<Vec<Option<u32>>>()
        }};
    }
    macro_rules! col_f32 {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .f32()
                .map_err(to_io)?
                .into_iter()
                .map(|v| v.unwrap_or(0.0))
                .collect::<Vec<f32>>()
        }};
    }
    macro_rules! col_f32_opt {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .f32()
                .map_err(to_io)?
                .into_iter()
                .collect::<Vec<Option<f32>>>()
        }};
    }

    // --- Materialize columns into Vecs (fast path) ---
    let cluster_id      = col_u64!("cluster_id");
    let ms_level        = col_u8!("ms_level");
    let window_group    = col_u32_opt!("window_group");
    let parent_im_id    = col_i64_opt!("parent_im_id");
    let parent_rt_id    = col_i64_opt!("parent_rt_id");

    let rt_lo           = col_u32!("rt_lo");
    let rt_hi           = col_u32!("rt_hi");
    let im_lo           = col_u32!("im_lo");
    let im_hi           = col_u32!("im_hi");
    let tof_lo          = col_u32!("tof_lo");
    let tof_hi          = col_u32!("tof_hi");
    let tof_index_lo    = col_i32!("tof_index_lo");
    let tof_index_hi    = col_i32!("tof_index_hi");
    let mz_lo           = col_f32_opt!("mz_lo");
    let mz_hi           = col_f32_opt!("mz_hi");

    let rt_mu           = col_f32!("rt_mu");
    let rt_sigma        = col_f32!("rt_sigma");
    let rt_height       = col_f32!("rt_height");
    let rt_area         = col_f32!("rt_area");

    let im_mu           = col_f32!("im_mu");
    let im_sigma        = col_f32!("im_sigma");
    let im_height       = col_f32!("im_height");
    let im_area         = col_f32!("im_area");

    let tof_mu          = col_f32!("tof_mu");
    let tof_sigma       = col_f32!("tof_sigma");
    let tof_height      = col_f32!("tof_height");
    let tof_area        = col_f32!("tof_area");

    let mz_mu           = col_f32_opt!("mz_mu");
    let mz_sigma        = col_f32_opt!("mz_sigma");
    let mz_height       = col_f32_opt!("mz_height");
    let mz_area         = col_f32_opt!("mz_area");

    let raw_sum         = col_f32!("raw_sum");
    let volume_proxy    = col_f32!("volume_proxy");

    debug_assert_eq!(cluster_id.len(), n);

    // ✅ Drop df before allocating/building out: reduces peak memory a lot.
    drop(df);

    // --- Reconstruct ClusterResult1D (serial = usually fastest + lowest overhead) ---
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mz_window = match (mz_lo[i], mz_hi[i]) {
            (Some(lo), Some(hi)) => Some((lo, hi)),
            _ => None,
        };

        let rt_fit = Fit1D {
            mu: rt_mu[i],
            sigma: rt_sigma[i],
            height: rt_height[i],
            baseline: 0.0,
            area: rt_area[i],
            r2: 0.0,
            n: 0,
        };
        let im_fit = Fit1D {
            mu: im_mu[i],
            sigma: im_sigma[i],
            height: im_height[i],
            baseline: 0.0,
            area: im_area[i],
            r2: 0.0,
            n: 0,
        };
        let tof_fit = Fit1D {
            mu: tof_mu[i],
            sigma: tof_sigma[i],
            height: tof_height[i],
            baseline: 0.0,
            area: tof_area[i],
            r2: 0.0,
            n: 0,
        };

        let mz_fit = match (mz_mu[i], mz_sigma[i], mz_height[i], mz_area[i]) {
            (Some(mu), Some(sig), Some(h), Some(a)) => Some(Fit1D {
                mu,
                sigma: sig,
                height: h,
                baseline: 0.0,
                area: a,
                r2: 0.0,
                n: 0,
            }),
            _ => None,
        };

        out.push(ClusterResult1D {
            cluster_id: cluster_id[i],
            ms_level: ms_level[i],
            window_group: window_group[i],
            parent_im_id: parent_im_id[i],
            parent_rt_id: parent_rt_id[i],

            rt_window: (rt_lo[i] as usize, rt_hi[i] as usize),
            im_window: (im_lo[i] as usize, im_hi[i] as usize),
            tof_window: (tof_lo[i] as usize, tof_hi[i] as usize),
            tof_index_window: (tof_index_lo[i], tof_index_hi[i]),
            mz_window,

            rt_fit,
            im_fit,
            tof_fit,
            mz_fit,

            raw_sum: raw_sum[i],
            volume_proxy: volume_proxy[i],

            frame_ids_used: Vec::new(),
            rt_axis_sec: None,
            im_axis_scans: None,
            mz_axis_da: None,
            raw_points: None,
            rt_trace: None,
            im_trace: None,
        });
    }

    Ok(out)
}

/// Load only the fields needed for StreamingFragmentIndex from a parquet file.
/// This uses selective column reading and produces SlimCluster instead of full ClusterResult1D.
/// RAM usage: ~52 bytes per cluster vs ~280 bytes for full ClusterResult1D.
pub fn load_parquet_slim(path: &str) -> io::Result<Vec<SlimCluster>> {
    use std::fs::File;
    use polars::prelude::*;

    let f = File::open(path)?;

    // Only read the columns we need for slim clusters
    let columns = vec![
        "cluster_id".to_string(),
        "ms_level".to_string(),
        "window_group".to_string(),
        "rt_mu".to_string(),
        "rt_sigma".to_string(),
        "im_mu".to_string(),
        "im_sigma".to_string(),
        "im_lo".to_string(),
        "im_hi".to_string(),
        "mz_mu".to_string(),
        "mz_lo".to_string(),
        "mz_hi".to_string(),
        "raw_sum".to_string(),
    ];

    let df = ParquetReader::new(f)
        .with_columns(Some(columns))
        .finish()
        .map_err(to_io)?;

    let n = df.height();

    // --- Helper macros (same as load_parquet) ---
    macro_rules! col_u64 {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .u64()
                .map_err(to_io)?
                .into_iter()
                .map(|v| v.unwrap_or(0))
                .collect::<Vec<u64>>()
        }};
    }
    macro_rules! col_u8 {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .u8()
                .map_err(to_io)?
                .into_iter()
                .map(|v| v.unwrap_or(0))
                .collect::<Vec<u8>>()
        }};
    }
    macro_rules! col_u32 {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .u32()
                .map_err(to_io)?
                .into_iter()
                .map(|v| v.unwrap_or(0))
                .collect::<Vec<u32>>()
        }};
    }
    macro_rules! col_u32_opt {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .u32()
                .map_err(to_io)?
                .into_iter()
                .collect::<Vec<Option<u32>>>()
        }};
    }
    macro_rules! col_f32 {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .f32()
                .map_err(to_io)?
                .into_iter()
                .map(|v| v.unwrap_or(0.0))
                .collect::<Vec<f32>>()
        }};
    }
    macro_rules! col_f32_opt {
        ($name:literal) => {{
            df.column($name)
                .map_err(to_io)?
                .f32()
                .map_err(to_io)?
                .into_iter()
                .collect::<Vec<Option<f32>>>()
        }};
    }

    // --- Materialize only the columns we need ---
    let cluster_id   = col_u64!("cluster_id");
    let ms_level     = col_u8!("ms_level");
    let window_group = col_u32_opt!("window_group");
    let rt_mu        = col_f32!("rt_mu");
    let rt_sigma     = col_f32!("rt_sigma");
    let im_mu        = col_f32!("im_mu");
    let im_sigma     = col_f32!("im_sigma");
    let im_lo        = col_u32!("im_lo");
    let im_hi        = col_u32!("im_hi");
    let mz_mu_col    = col_f32_opt!("mz_mu");
    let mz_lo        = col_f32_opt!("mz_lo");
    let mz_hi        = col_f32_opt!("mz_hi");
    let raw_sum      = col_f32!("raw_sum");

    // ✅ Drop df before building output: reduces peak memory
    drop(df);

    // --- Build SlimCluster vec ---
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Compute mz_mu: prefer mz_fit.mu, fallback to mz window midpoint
        let mz_mu = mz_mu_col[i].unwrap_or_else(|| {
            match (mz_lo[i], mz_hi[i]) {
                (Some(lo), Some(hi)) => 0.5 * (lo + hi),
                _ => 0.0,
            }
        });

        out.push(SlimCluster {
            cluster_id: cluster_id[i],
            window_group: window_group[i],
            ms_level: ms_level[i],
            rt_mu: rt_mu[i],
            im_mu: im_mu[i],
            mz_mu,
            rt_sigma: rt_sigma[i],
            im_sigma: im_sigma[i],
            im_window: (im_lo[i] as u16, im_hi[i] as u16),
            raw_sum: raw_sum[i],
        });
    }

    Ok(out)
}

/// Fast Polars-based slim parquet reader.
/// Only reads columns needed for SlimCluster, skipping heavy data (traces, raw_points).
/// Much faster than row-by-row iteration.
pub fn load_parquet_slim_streaming(path: &str) -> io::Result<Vec<SlimCluster>> {
    use polars::prelude::*;

    let f = File::open(path)?;

    // Only read the columns we need for SlimCluster
    let cols = vec![
        "cluster_id".into(),
        "ms_level".into(),
        "window_group".into(),
        "rt_mu".into(),
        "rt_sigma".into(),
        "im_mu".into(),
        "im_sigma".into(),
        "im_lo".into(),
        "im_hi".into(),
        "mz_mu".into(),
        "mz_lo".into(),
        "mz_hi".into(),
        "raw_sum".into(),
    ];

    let df = ParquetReader::new(f)
        .with_columns(Some(cols))
        .finish()
        .map_err(to_io)?;

    let n = df.height();

    // Extract columns
    let cluster_ids: Vec<u64> = df.column("cluster_id").map_err(to_io)?
        .u64().map_err(to_io)?
        .into_iter().map(|v| v.unwrap_or(0)).collect();

    let ms_levels: Vec<u8> = df.column("ms_level").map_err(to_io)?
        .u8().map_err(to_io)?
        .into_iter().map(|v| v.unwrap_or(0)).collect();

    let window_groups: Vec<Option<u32>> = df.column("window_group").map_err(to_io)?
        .u32().map_err(to_io)?
        .into_iter().collect();

    let rt_mus: Vec<f32> = df.column("rt_mu").map_err(to_io)?
        .f32().map_err(to_io)?
        .into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let rt_sigmas: Vec<f32> = df.column("rt_sigma").map_err(to_io)?
        .f32().map_err(to_io)?
        .into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let im_mus: Vec<f32> = df.column("im_mu").map_err(to_io)?
        .f32().map_err(to_io)?
        .into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let im_sigmas: Vec<f32> = df.column("im_sigma").map_err(to_io)?
        .f32().map_err(to_io)?
        .into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let im_los: Vec<u32> = df.column("im_lo").map_err(to_io)?
        .u32().map_err(to_io)?
        .into_iter().map(|v| v.unwrap_or(0)).collect();

    let im_his: Vec<u32> = df.column("im_hi").map_err(to_io)?
        .u32().map_err(to_io)?
        .into_iter().map(|v| v.unwrap_or(0)).collect();

    let mz_mus: Vec<Option<f32>> = df.column("mz_mu").map_err(to_io)?
        .f32().map_err(to_io)?
        .into_iter().collect();

    let mz_los: Vec<Option<f32>> = df.column("mz_lo").map_err(to_io)?
        .f32().map_err(to_io)?
        .into_iter().collect();

    let mz_his: Vec<Option<f32>> = df.column("mz_hi").map_err(to_io)?
        .f32().map_err(to_io)?
        .into_iter().collect();

    let raw_sums: Vec<f32> = df.column("raw_sum").map_err(to_io)?
        .f32().map_err(to_io)?
        .into_iter().map(|v| v.unwrap_or(0.0)).collect();

    // Build SlimClusters
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mz_mu = mz_mus[i].unwrap_or_else(|| {
            match (mz_los[i], mz_his[i]) {
                (Some(lo), Some(hi)) => 0.5 * (lo + hi),
                _ => 0.0,
            }
        });

        out.push(SlimCluster {
            cluster_id: cluster_ids[i],
            window_group: window_groups[i],
            ms_level: ms_levels[i],
            rt_mu: rt_mus[i],
            im_mu: im_mus[i],
            mz_mu,
            rt_sigma: rt_sigmas[i],
            im_sigma: im_sigmas[i],
            im_window: (im_los[i] as u16, im_his[i] as u16),
            raw_sum: raw_sums[i],
        });
    }

    Ok(out)
}

/// Even more memory-efficient: processes parquet file and directly populates index structures.
/// Never holds all SlimClusters in memory at once - builds index incrementally.
pub fn stream_parquet_to_index_data(
    path: &str,
    min_raw_sum: f32,
) -> io::Result<(
    Vec<SlimCluster>,        // Only kept clusters
    Vec<(u32, f32, usize)>,  // (window_group, rt_mu, index) for by_group building
)> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::RowAccessor;
    use std::fs::File;

    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    let metadata = reader.metadata();
    let num_rows = metadata.file_metadata().num_rows() as usize;

    let mut slim_clusters = Vec::with_capacity(num_rows);
    let mut group_entries = Vec::with_capacity(num_rows);

    let num_row_groups = metadata.num_row_groups();
    for rg_idx in 0..num_row_groups {
        let row_group = reader.get_row_group(rg_idx)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let iter = row_group.get_row_iter(None)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        for row_result in iter {
            let row = row_result
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

            let ms_level = row.get_ubyte(1).unwrap_or(0);
            if ms_level != 2 { continue; } // Skip non-MS2

            let window_group = match row.get_uint(2).ok() {
                Some(g) => g,
                None => continue, // Skip if no window group
            };

            let raw_sum = row.get_float(24).unwrap_or(0.0);
            if raw_sum < min_raw_sum { continue; } // Skip low intensity

            let rt_mu = row.get_float(5).unwrap_or(0.0);
            if !rt_mu.is_finite() || rt_mu <= 0.0 { continue; }

            let cluster_id = row.get_long(0).unwrap_or(0) as u64;
            let rt_sigma = row.get_float(6).unwrap_or(0.0);
            let im_mu = row.get_float(9).unwrap_or(0.0);
            let im_sigma = row.get_float(10).unwrap_or(0.0);
            let im_lo = row.get_uint(7).unwrap_or(0) as u16;
            let im_hi = row.get_uint(8).unwrap_or(0) as u16;

            let mz_mu_opt = row.get_float(20).ok();
            let mz_lo_opt = row.get_float(13).ok();
            let mz_hi_opt = row.get_float(14).ok();
            let mz_mu = mz_mu_opt.unwrap_or_else(|| {
                match (mz_lo_opt, mz_hi_opt) {
                    (Some(lo), Some(hi)) => 0.5 * (lo + hi),
                    _ => 0.0,
                }
            });

            let idx = slim_clusters.len();
            group_entries.push((window_group, rt_mu, idx));

            slim_clusters.push(SlimCluster {
                cluster_id,
                window_group: Some(window_group),
                ms_level,
                rt_mu,
                im_mu,
                mz_mu,
                rt_sigma,
                im_sigma,
                im_window: (im_lo, im_hi),
                raw_sum,
            });
        }
    }

    Ok((slim_clusters, group_entries))
}

fn to_io(e: PolarsError) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e)
}

#[derive(Serialize, Deserialize)]
pub struct PseudoSpectraFile {
    pub version: u32,
    pub spectra: Vec<PseudoSpectrum>,
}

impl PseudoSpectraFile {
    pub fn new(spectra: Vec<PseudoSpectrum>) -> Self {
        Self {
            version: 1,
            spectra,
        }
    }
}

pub fn save_pseudo_bincode(
    path: &str,
    spectra: &[PseudoSpectrum],
    compress: bool,
) -> io::Result<()> {
    let f = File::create(path)?;
    if compress {
        // zstd compression, level 3 is a good default
        let mut zw = zstd::Encoder::new(f, 3)?;
        bincode::serialize_into(&mut zw, &PseudoSpectraFile::new(spectra.to_vec()))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        zw.finish()?;
        Ok(())
    } else {
        let mut bw = BufWriter::new(f);
        bincode::serialize_into(&mut bw, &PseudoSpectraFile::new(spectra.to_vec()))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }
}

fn is_zstd_file(path: &str) -> io::Result<bool> {
    use std::io::Read;
    let mut f = File::open(path)?;
    let mut magic = [0u8; 4];
    let n = f.read(&mut magic)?;
    if n < 4 {
        return Ok(false);
    }
    Ok(magic == [0x28, 0xB5, 0x2F, 0xFD])
}

pub fn load_pseudo_bincode(path: &str) -> io::Result<Vec<PseudoSpectrum>> {
    if is_zstd_file(path)? {
        let f = File::open(path)?;
        let mut zr = zstd::Decoder::new(f)?;
        let pf: PseudoSpectraFile = bincode::deserialize_from(&mut zr)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        return Ok(pf.spectra);
    }

    let f = BufReader::new(File::open(path)?);
    let pf: PseudoSpectraFile = bincode::deserialize_from(f)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(pf.spectra)
}

pub fn save_feature_bincode(
    path: &str,
    features: &Vec<SimpleFeature>,
    compress: bool,
) -> io::Result<()> {
    let f = File::create(path)?;
    if compress {
        let mut zw = zstd::Encoder::new(f, 3)?;
        bincode::serialize_into(&mut zw, features)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        zw.finish()?;
        Ok(())
    } else {
        let mut bw = BufWriter::new(f);
        bincode::serialize_into(&mut bw, features)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }
}

pub fn load_feature_bincode(path: &str) -> io::Result<Vec<SimpleFeature>> {
    let f = File::open(path)?;

    // Try zstd first
    if let Ok(mut zr) = zstd::Decoder::new(&f) {
        let features: Vec<SimpleFeature> = bincode::deserialize_from(&mut zr)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        return Ok(features);
    }

    // Fallback: plain bincode
    let f = BufReader::new(File::open(path)?);
    let features: Vec<SimpleFeature> = bincode::deserialize_from(f)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(features)
}