use std::fs::File;
use std::io::{BufReader, BufWriter};
use serde::{Deserialize, Serialize};
use crate::cluster::utility::Fit1D;
use super::cluster::ClusterResult1D;

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

#[inline]
fn col<'a>(df: &'a DataFrame, name: &str) -> io::Result<&'a Series> {
    df.column(name).map_err(to_io)
}

pub fn load_parquet(path: &str) -> io::Result<Vec<ClusterResult1D>> {
    let f = File::open(path)?;

    // Full DataFrame read (still “en bloc”), but we avoid cloning columns into Vecs afterwards.
    let df = ParquetReader::new(f)
        .finish()
        .map_err(to_io)?;

    let n = df.height();

    // --- typed views (borrowed) ---
    let cluster_id   = col(&df, "cluster_id")?.u64().map_err(to_io)?;
    let ms_level     = col(&df, "ms_level")?.u8().map_err(to_io)?;
    let window_group = col(&df, "window_group")?.u32().map_err(to_io)?;
    let parent_im_id = col(&df, "parent_im_id")?.i64().map_err(to_io)?;
    let parent_rt_id = col(&df, "parent_rt_id")?.i64().map_err(to_io)?;

    let rt_lo        = col(&df, "rt_lo")?.u32().map_err(to_io)?;
    let rt_hi        = col(&df, "rt_hi")?.u32().map_err(to_io)?;
    let im_lo        = col(&df, "im_lo")?.u32().map_err(to_io)?;
    let im_hi        = col(&df, "im_hi")?.u32().map_err(to_io)?;
    let tof_lo       = col(&df, "tof_lo")?.u32().map_err(to_io)?;
    let tof_hi       = col(&df, "tof_hi")?.u32().map_err(to_io)?;
    let tof_index_lo = col(&df, "tof_index_lo")?.i32().map_err(to_io)?;
    let tof_index_hi = col(&df, "tof_index_hi")?.i32().map_err(to_io)?;
    let mz_lo        = col(&df, "mz_lo")?.f32().map_err(to_io)?;
    let mz_hi        = col(&df, "mz_hi")?.f32().map_err(to_io)?;

    let rt_mu        = col(&df, "rt_mu")?.f32().map_err(to_io)?;
    let rt_sigma     = col(&df, "rt_sigma")?.f32().map_err(to_io)?;
    let rt_height    = col(&df, "rt_height")?.f32().map_err(to_io)?;
    let rt_area      = col(&df, "rt_area")?.f32().map_err(to_io)?;

    let im_mu        = col(&df, "im_mu")?.f32().map_err(to_io)?;
    let im_sigma     = col(&df, "im_sigma")?.f32().map_err(to_io)?;
    let im_height    = col(&df, "im_height")?.f32().map_err(to_io)?;
    let im_area      = col(&df, "im_area")?.f32().map_err(to_io)?;

    let tof_mu       = col(&df, "tof_mu")?.f32().map_err(to_io)?;
    let tof_sigma    = col(&df, "tof_sigma")?.f32().map_err(to_io)?;
    let tof_height   = col(&df, "tof_height")?.f32().map_err(to_io)?;
    let tof_area     = col(&df, "tof_area")?.f32().map_err(to_io)?;

    let mz_mu        = col(&df, "mz_mu")?.f32().map_err(to_io)?;
    let mz_sigma     = col(&df, "mz_sigma")?.f32().map_err(to_io)?;
    let mz_height    = col(&df, "mz_height")?.f32().map_err(to_io)?;
    let mz_area      = col(&df, "mz_area")?.f32().map_err(to_io)?;

    let raw_sum      = col(&df, "raw_sum")?.f32().map_err(to_io)?;
    let volume_proxy = col(&df, "volume_proxy")?.f32().map_err(to_io)?;

    // --- build output without column Vec copies ---
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let cid = cluster_id.get(i).unwrap_or(0);
        let lvl = ms_level.get(i).unwrap_or(0);

        let wg  = window_group.get(i);       // Option<u32>
        let pim = parent_im_id.get(i);       // Option<i64>
        let prt = parent_rt_id.get(i);       // Option<i64>

        let rt_w = (rt_lo.get(i).unwrap_or(0) as usize, rt_hi.get(i).unwrap_or(0) as usize);
        let im_w = (im_lo.get(i).unwrap_or(0) as usize, im_hi.get(i).unwrap_or(0) as usize);
        let tf_w = (tof_lo.get(i).unwrap_or(0) as usize, tof_hi.get(i).unwrap_or(0) as usize);
        let tf_i = (tof_index_lo.get(i).unwrap_or(0), tof_index_hi.get(i).unwrap_or(0));

        let mz_lo_v = mz_lo.get(i);          // Option<f32>
        let mz_hi_v = mz_hi.get(i);          // Option<f32>
        let mz_window = match (mz_lo_v, mz_hi_v) {
            (Some(lo), Some(hi)) => Some((lo, hi)),
            _ => None,
        };

        let rt_fit = Fit1D {
            mu: rt_mu.get(i).unwrap_or(0.0),
            sigma: rt_sigma.get(i).unwrap_or(0.0),
            height: rt_height.get(i).unwrap_or(0.0),
            baseline: 0.0,
            area: rt_area.get(i).unwrap_or(0.0),
            r2: 0.0,
            n: 0,
        };

        let im_fit = Fit1D {
            mu: im_mu.get(i).unwrap_or(0.0),
            sigma: im_sigma.get(i).unwrap_or(0.0),
            height: im_height.get(i).unwrap_or(0.0),
            baseline: 0.0,
            area: im_area.get(i).unwrap_or(0.0),
            r2: 0.0,
            n: 0,
        };

        let tof_fit = Fit1D {
            mu: tof_mu.get(i).unwrap_or(0.0),
            sigma: tof_sigma.get(i).unwrap_or(0.0),
            height: tof_height.get(i).unwrap_or(0.0),
            baseline: 0.0,
            area: tof_area.get(i).unwrap_or(0.0),
            r2: 0.0,
            n: 0,
        };

        let mz_fit = match (mz_mu.get(i), mz_sigma.get(i), mz_height.get(i), mz_area.get(i)) {
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
            cluster_id: cid,
            ms_level: lvl,
            window_group: wg,
            parent_im_id: pim,
            parent_rt_id: prt,

            rt_window: rt_w,
            im_window: im_w,
            tof_window: tf_w,
            tof_index_window: tf_i,
            mz_window,

            rt_fit,
            im_fit,
            tof_fit,
            mz_fit,

            raw_sum: raw_sum.get(i).unwrap_or(0.0),
            volume_proxy: volume_proxy.get(i).unwrap_or(0.0),

            // not stored in parquet
            frame_ids_used: Vec::new(),

            // heavy fields omitted
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