use std::fs::File;
use std::io::{BufReader, BufWriter};
use serde::{Deserialize, Serialize};
use super::cluster::ClusterResult1D;
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