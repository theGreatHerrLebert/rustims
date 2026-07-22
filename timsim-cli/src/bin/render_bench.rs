//! `timsim-render-bench` — the §7 streaming-render prototype and memory benchmark.
//!
//! It proves the central claim of `TIMSIM_V2_RENDER.md`: that a sweep-line render's working set is
//! bounded by the **elution window, not the run length**. It streams the Parquet feature space, drives
//! the shared [`timsim_cli::render::stream_render`] sweep (the same code the oracle tests in
//! `src/render.rs` verify bin-for-bin), accumulates a real per-frame sparse `(scan, tof)` buffer from
//! each active precursor's isotope envelope (the MS1 path), and reports:
//!
//!   - peak **active-set** size (should be ~ elution width, independent of total precursors),
//!   - peak per-frame **buffer** entries,
//!   - the **sweep working set** (active heap + frame buffer) — the bounded quantity,
//!   - peak **RSS** (target < 1 GB) — dominated here by the O(total) load, a prototype artifact,
//!   - throughput (frames/s, peaks/s).
//!
//! Conservation is NOT re-checked here: it is proven independently and with teeth by the oracle in
//! `src/render.rs` (an ion-major reference render compared to this sweep at every bin, plus
//! metamorphic tests). Run it with `cargo test -p timsim-cli --lib render`.

use anyhow::Result;
use arrow::array::{Float64Array, ListArray, UInt64Array, UInt8Array};
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;
use timsim_cli::render::{stream_render, stream_render_flat, Geometry, Ion};
use timsim_schema::tables::{peptide_rt as RT, precursors as PRE};

#[derive(Parser)]
#[command(name = "timsim-render-bench", about = "streaming sweep-line render prototype + memory benchmark")]
struct Args {
    #[arg(long)]
    precursors: PathBuf,
    #[arg(long)]
    peptide_rt: PathBuf,
    /// Number of frames (TIMS ramps) in the run. Must be >= 1.
    #[arg(long, default_value_t = 36_000, value_parser = clap::value_parser!(u32).range(1..))]
    n_frames: u32,
    /// Number of mobility scans per frame. Must be >= 1 (0 would underflow the scan window).
    #[arg(long, default_value_t = 700, value_parser = clap::value_parser!(u32).range(1..))]
    n_scans: u32,
    /// Elution peak width, in frames (Gaussian sigma). ~7 s at a 100 ms cycle ≈ 70 frames.
    #[arg(long, default_value_t = 30.0)]
    sigma_frames: f64,
    /// Mobility peak width, in scans (Gaussian sigma).
    #[arg(long, default_value_t = 4.0)]
    sigma_scans: f64,
    /// Truncate each peak's window at this many sigma (the target_p analog).
    #[arg(long, default_value_t = 3.0)]
    n_sigma: f64,
    /// Cap on precursors read (0 = all). For scaling curves.
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Peaks per ion. Isotope envelopes are ~2-6 peaks (MS1); pass a larger value to emulate the
    /// point density of MS2 fragment spectra (a conservative upper bound — real diagonal transmission
    /// only removes points, so it lowers cost). Extra peaks are appended at stepped tof bins.
    #[arg(long, default_value_t = 0)]
    fragments: usize,
    /// Accumulator: "hashmap" (dedup while accumulating) or "flat" (append, dedup at encode).
    #[arg(long, default_value = "hashmap")]
    accumulator: String,
    /// Also run the REAL Bruker TDF per-frame block encoder on each frame (zstd + delta-TOF) and
    /// write blocks to a sink, measuring end-to-end render+encode throughput and compressed size.
    /// Requires building with `--features tdf`.
    #[arg(long, default_value_t = false)]
    encode: bool,
    /// Intensity quantisation scale: emitted f64 → u32 as `(value * scale)`. Encode path only.
    #[arg(long, default_value_t = 1000.0)]
    encode_scale: f64,
    /// Threads for the encode stage. The TDF block encoder is per-frame independent (embarrassingly
    /// parallel); frames are batched and encoded with a rayon pool of this size. 1 = serial baseline.
    #[arg(long, default_value_t = 1)]
    encode_threads: usize,
    /// Frames of raw triples buffered before a parallel encode flush. Bounds encode-stage memory
    /// (batch × triples/frame × 16 B) — lower for dense MS2, raise on a big-RAM box for more encode
    /// parallelism per flush.
    #[arg(long, default_value_t = 64)]
    encode_batch: usize,
    /// Parallel render-by-chunk: split the run into this many contiguous frame ranges and render them
    /// on a rayon pool of the same size, then report the parallel render wall. 0 = the serial sweep
    /// path (with the encode stage). >0 measures render SCALING (render-only; encode composes on top,
    /// measured separately by --encode-threads).
    #[arg(long, default_value_t = 0)]
    render_chunks: usize,
}

/// One frame's raw `(scan, tof, value)` contributions, queued for the parallel encode stage. Values
/// may repeat a `(scan, tof)` key (the flat accumulator appends co-eluting contributions); the encode
/// worker sums them in f64 before quantising. Carrying raw triples — not pre-summed arrays — is what
/// lets the DEDUP run inside the parallel worker instead of the serial render callback. Memory is
/// bounded by `encode_batch` frames of triples.
type FrameArrays = (u32, Vec<(u32, u32, f64)>);

/// Sum a frame's contributions per `(scan, tof)` in f64, then quantise to integer arrays, dropping
/// sub-quantum peaks. Summing BEFORE quantising is essential: quantising each contribution first would
/// floor co-eluting sub-quantum signal to zero. This is the render's per-frame dedup; it runs inside
/// the parallel encode worker.
fn dedup_and_quantise(triples: &[(u32, u32, f64)], scale: f64) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut summed: HashMap<(u32, u32), f64> = HashMap::with_capacity(triples.len());
    for &(scan, tof, v) in triples {
        *summed.entry((scan, tof)).or_insert(0.0) += v;
    }
    let (mut scans, mut tofs, mut ints) = (Vec::new(), Vec::new(), Vec::new());
    for ((scan, tof), v) in summed {
        let q = (v * scale) as u32;
        if q == 0 {
            continue;
        }
        scans.push(scan);
        tofs.push(tof);
        ints.push(q);
    }
    (scans, tofs, ints)
}

/// The encode stage: buffers per-frame integer arrays and flushes them through the real TDF block
/// encoder in parallel batches. Gap-fills empty blocks for frames the renderer skipped (no active
/// ions), so the block stream is 1:1 with acquisition frames — as a real `.d` writer must be.
struct EncodeState {
    enabled: bool,
    batch: Vec<FrameArrays>,
    batch_size: usize,
    next_frame: u32,
    n_scans: u32,
    scale: f64,
    encode_s: f64,
    compressed_bytes: u64,
    n_blocks: u64,
    sink: Option<std::io::BufWriter<std::fs::File>>,
}

impl EncodeState {
    fn flush(&mut self, pool: &rayon::ThreadPool) -> Result<()> {
        use rayon::prelude::*;
        if self.batch.is_empty() {
            return Ok(());
        }
        let (n_scans, scale) = (self.n_scans, self.scale);
        let t = std::time::Instant::now();
        // The whole per-frame back end — dedup, quantise, and the real TDF block encode — runs in the
        // parallel worker, so the render's dominant cost (deduping every contribution) scales with
        // cores rather than serialising in the render callback.
        let blocks: Vec<Result<Vec<u8>>> = pool.install(|| {
            self.batch
                .par_drain(..)
                .map(|(_frame, triples)| {
                    let (scans, tofs, ints) = dedup_and_quantise(&triples, scale);
                    encode_block(scans, tofs, ints, n_scans)
                })
                .collect()
        });
        self.encode_s += t.elapsed().as_secs_f64();
        for b in blocks {
            let block = b?;
            self.compressed_bytes += block.len() as u64;
            if let Some(w) = self.sink.as_mut() {
                use std::io::Write;
                w.write_all(&block)?;
            }
        }
        Ok(())
    }

    /// Queue empty blocks for `[next_frame, up_to)`.
    fn fill_gap(&mut self, up_to: u32, pool: &rayon::ThreadPool) -> Result<()> {
        while self.next_frame < up_to {
            self.batch.push((self.next_frame, Vec::new()));
            self.n_blocks += 1;
            self.next_frame += 1;
            if self.batch.len() >= self.batch_size {
                self.flush(pool)?;
            }
        }
        Ok(())
    }

    /// Queue one signal-bearing frame's raw triples, gap-filling any frames the renderer skipped.
    fn push(&mut self, frame: u32, triples: Vec<(u32, u32, f64)>, pool: &rayon::ThreadPool) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.fill_gap(frame, pool)?;
        self.batch.push((frame, triples));
        self.n_blocks += 1;
        self.next_frame = frame + 1;
        if self.batch.len() >= self.batch_size {
            self.flush(pool)?;
        }
        Ok(())
    }

    /// Trailing empty frames to `total_frames`, then a final flush and sink flush.
    fn finish(&mut self, total_frames: u32, pool: &rayon::ThreadPool) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.fill_gap(total_frames, pool)?;
        self.flush(pool)?;
        if let Some(w) = self.sink.as_mut() {
            use std::io::Write;
            w.flush()?;
        }
        Ok(())
    }
}

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

/// The REAL Bruker TDF per-frame block encoder (sort-dedup by (scan,tof), delta-TOF, interleave,
/// zstd, `len+total_scans` header) — so the throughput proof measures the actual compressor.
#[cfg(feature = "tdf")]
fn encode_block(scans: Vec<u32>, tofs: Vec<u32>, ints: Vec<u32>, total_scans: u32) -> Result<Vec<u8>> {
    ms_io::data::utility::reconstruct_compressed_data(scans, tofs, ints, total_scans, 1)
        .map_err(|e| anyhow::anyhow!("TDF block encode failed: {e}"))
}

#[cfg(not(feature = "tdf"))]
fn encode_block(_: Vec<u32>, _: Vec<u32>, _: Vec<u32>, _: u32) -> Result<Vec<u8>> {
    anyhow::bail!("--encode requires building with `--features tdf`")
}

/// A real on-disk sink for the encoded blocks (append-only, mirrors the tdf_bin write path). We only
/// need to measure write throughput and size, so a temp file is fine — but it must be per-process and
/// created without following/truncating an existing path, so concurrent runs don't corrupt each other
/// and a pre-planted symlink in a shared temp dir can't redirect the write.
fn tempfile_sink() -> Result<std::fs::File> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let name = format!("timsim_render_bench.{}.{}.tdf_bin", std::process::id(), nanos);
    let p = std::env::temp_dir().join(name);
    // create_new = O_CREAT|O_EXCL: fails rather than following a symlink or truncating an existing file.
    Ok(std::fs::OpenOptions::new().write(true).create_new(true).open(&p)?)
}

/// Per-chunk stats returned from a parallel render, reduced into the run totals.
struct ChunkStat {
    ions: usize,
    max_active: usize,
    max_buf: usize,
    total_peaks: u64,
    wall_s: f64,
}

/// Parallel render-by-chunk: split the run into `render_chunks` contiguous frame ranges, bucket the
/// ions into the chunks they overlap, and render the chunks concurrently on a rayon pool. Measures the
/// render-scaling — the wall of the parallel block vs the serial sum of per-chunk walls (the speedup).
/// Render-only (flat accumulator); the encode stage composes on top and is parallelised separately.
fn run_chunked(a: &Args, g: &Geometry, ions: &[Ion]) -> Result<()> {
    use rayon::prelude::*;
    use timsim_cli::render::{bucket_ions, frame_chunks, stream_render_flat_range};

    let chunks = frame_chunks(a.n_frames, a.render_chunks);
    // frame_chunks clamps to at most one chunk per frame, so the effective parallelism is chunks.len(),
    // NOT the raw --render-chunks. Size the pool from the clamped count, or `--render-chunks 1000000
    // --n-frames 10` would try to spawn a million OS threads for ten chunks.
    let k = chunks.len();
    let t_bucket = std::time::Instant::now();
    let buckets = bucket_ions(ions, g, &chunks);
    let bucket_s = t_bucket.elapsed().as_secs_f64();
    let dup_factor: f64 =
        buckets.iter().map(|b| b.len()).sum::<usize>() as f64 / (ions.len().max(1) as f64);

    let pool = rayon::ThreadPoolBuilder::new().num_threads(k).build()?;
    let t = std::time::Instant::now();
    let stats: Vec<ChunkStat> = pool.install(|| {
        chunks
            .par_iter()
            .zip(buckets.par_iter())
            .map(|(&(lo, hi), bucket)| {
                let sub: Vec<Ion> = bucket.iter().map(|&i| ions[i].clone()).collect();
                let (mut max_active, mut max_buf, mut total_peaks) = (0usize, 0usize, 0u64);
                let tc = std::time::Instant::now();
                stream_render_flat_range(&sub, g, lo, hi, |e| {
                    max_active = max_active.max(e.active);
                    max_buf = max_buf.max(e.triples.len());
                    total_peaks += e.triples.len() as u64;
                });
                ChunkStat {
                    ions: sub.len(),
                    max_active,
                    max_buf,
                    total_peaks,
                    wall_s: tc.elapsed().as_secs_f64(),
                }
            })
            .collect()
    });
    let parallel_s = t.elapsed().as_secs_f64();

    let serial_sum_s: f64 = stats.iter().map(|s| s.wall_s).sum();
    let total_peaks: u64 = stats.iter().map(|s| s.total_peaks).sum();
    let max_active = stats.iter().map(|s| s.max_active).max().unwrap_or(0);
    let max_buf = stats.iter().map(|s| s.max_buf).max().unwrap_or(0);
    let slowest = stats.iter().map(|s| s.wall_s).fold(0.0, f64::max);
    let (ion_min, ion_max) = (
        stats.iter().map(|s| s.ions).min().unwrap_or(0),
        stats.iter().map(|s| s.ions).max().unwrap_or(0),
    );
    // Load-balance: ideal parallel time is serial_sum / k; the slowest chunk sets the real floor.
    let balance = if slowest > 0.0 { (serial_sum_s / k as f64) / slowest } else { 1.0 };

    println!();
    println!("  ── parallel render-by-chunk (render scaling) ───────────────");
    println!("  precursors            : {:>12}", ions.len());
    println!("  frames                : {:>12}", a.n_frames);
    println!("  chunks / threads      : {:>12}", k);
    println!("  ion bucketing         : {:>12.3} s   ({:.3}× ions after boundary dups; per-chunk \
             {}–{})", bucket_s, dup_factor, ion_min, ion_max);
    println!("  peak ACTIVE (max chunk): {:>11}   (~ per-chunk elution width)", max_active);
    println!("  peak buffer (max chunk): {:>11}   (scan,tof,val) triples", max_buf);
    println!("  total peaks emitted   : {:>12}", total_peaks);
    println!("  peak RSS              : {:>12.1} MB", peak_rss_mb());
    println!("  Σ per-chunk render    : {:>12.2} s   (serial-equivalent work)", serial_sum_s);
    println!("  PARALLEL render wall  : {:>12.2} s   ({:.0} frames/s, {:.1}M peaks/s)",
             parallel_s, a.n_frames as f64 / parallel_s.max(1e-9), total_peaks as f64 / parallel_s.max(1e-9) / 1e6);
    println!("  speedup               : {:>12.1}×   ({:.0}% of {}× ideal; slowest chunk {:.2} s, \
             balance {:.2})", serial_sum_s / parallel_s.max(1e-9),
             100.0 * (serial_sum_s / parallel_s.max(1e-9)) / k as f64, k, slowest, balance);
    let full_s = parallel_s / a.n_frames as f64 * 36_000.0;
    println!("  ⇒ extrapolated full 36k-frame render at this density: {:.2} s ({:.1} min)",
             full_s, full_s / 60.0);
    println!("  conservation          : frame-partition proven — \
             `cargo test -p timsim-cli --lib render` (frame_range_partition_equals_whole)");
    Ok(())
}

fn main() -> Result<()> {
    let a = Args::parse();
    let g = Geometry {
        n_frames: a.n_frames,
        n_scans: a.n_scans,
        sigma_frames: a.sigma_frames,
        sigma_scans: a.sigma_scans,
        n_sigma: a.n_sigma,
    };
    let t_load = std::time::Instant::now();

    // peptide_id -> rt_index (apex, normalized). We map rt_index to an apex frame via its rank so the
    // bench does not need the gradient calibration; it only needs the peaks spread across the run.
    let mut rt: HashMap<u64, f64> = HashMap::new();
    for b in timsim_schema::read(&a.peptide_rt, RT::TABLE)? {
        let id: &UInt64Array = b.column_by_name(RT::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let idx: &Float64Array = b.column_by_name(RT::RT_INDEX).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if arrow::array::Array::is_valid(idx, i) {
                rt.insert(id.value(i), idx.value(i));
            }
        }
    }
    // Normalize rt_index to [0, n_frames].
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in rt.values() {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let span = (hi - lo).max(1e-9);

    let mut ions: Vec<Ion> = Vec::new();
    let mz_min = 100.0;
    let tof_scale = 4.0; // tof bins per m/z — a stand-in calibration
    'outer: for b in timsim_schema::read(&a.precursors, PRE::TABLE)? {
        let pid: &UInt64Array = b.column_by_name(PRE::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let mz: &Float64Array = b.column_by_name(PRE::MZ).unwrap().as_any().downcast_ref().unwrap();
        let chg: &UInt8Array = b.column_by_name(PRE::CHARGE).unwrap().as_any().downcast_ref().unwrap();
        let iso: &ListArray = b.column_by_name(PRE::ISOTOPE_INTENSITY).unwrap().as_any().downcast_ref().unwrap();
        let frac: &arrow::array::Float32Array =
            b.column_by_name(PRE::CHARGE_FRACTION).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let Some(&apex_idx) = rt.get(&pid.value(i)) else { continue };
            let apex_frame = (apex_idx - lo) / span * a.n_frames as f64;
            let m = mz.value(i);
            // A rough mobility ~ m/z placement so peaks spread across the scan axis.
            let scan_center = (((m - 300.0) / 900.0).clamp(0.0, 1.0)) * (a.n_scans as f64 - 1.0);
            // Isotope envelope -> (tof, intensity) peaks. tof steps by the isotope spacing / charge.
            let env = iso.value(i);
            let env: &arrow::array::Float32Array = env.as_any().downcast_ref().unwrap();
            let tof0 = ((m - mz_min).max(0.0) * tof_scale) as u32;
            let tof_step = (1.0033 / chg.value(i).max(1) as f64 * tof_scale).round().max(1.0) as u32;
            let mut peaks: Vec<(u32, f32)> = (0..arrow::array::Array::len(env))
                .map(|j| (tof0 + j as u32 * tof_step, env.value(j)))
                .collect();
            // Emulate MS2 fragment density: append `fragments` extra peaks at stepped tof bins.
            for k in 0..a.fragments {
                let tof = tof0.wrapping_add((100 + k as u32 * 37) % 30_000);
                peaks.push((tof, 0.3));
            }
            ions.push(Ion {
                apex_frame,
                scan_center,
                abundance: frac.value(i) as f64, // unit peptide amount × charge_fraction
                peaks,
            });
            if a.limit > 0 && ions.len() >= a.limit {
                break 'outer;
            }
        }
    }
    let load_s = t_load.elapsed().as_secs_f64();
    eprintln!("  loaded {} precursors + {} RT entries in {:.1}s  (post-load RSS {:.1} MB — parquet \
               decode + the O(total) RT index map + materialized Vec<Ion>; a real streaming render \
               would hold none of these whole)",
              ions.len(), rt.len(), load_s, peak_rss_mb());

    // ── parallel render-by-chunk (render-scaling measurement) ────────────────
    if a.render_chunks >= 1 {
        return run_chunked(&a, &g, &ions);
    }

    // ── the serial sweep ─────────────────────────────────────────────────────
    let t_sweep = std::time::Instant::now();
    let mut max_active = 0usize;
    let mut max_buf = 0usize;
    let mut total_peaks: u64 = 0;
    let mut n_frames_emitted: u64 = 0;

    // Encode stage: the TDF block encoder is per-frame independent, so frames are buffered and flushed
    // in parallel batches. `enc.encode_s` measures ONLY the wall spent inside the (possibly parallel)
    // encode, a true stage cost even at N threads. Both accumulators feed the encoder the SAME
    // f64-summed-then-quantised spectrum, so the encoded output is accumulator-independent.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(a.encode_threads.max(1))
        .build()?;
    let mut enc = EncodeState {
        enabled: a.encode,
        batch: Vec::new(),
        batch_size: a.encode_batch,
        next_frame: 0,
        n_scans: a.n_scans,
        scale: a.encode_scale,
        encode_s: 0.0,
        compressed_bytes: 0,
        n_blocks: 0,
        sink: if a.encode { Some(std::io::BufWriter::new(tempfile_sink()?)) } else { None },
    };

    let mut err: Result<()> = Ok(());
    match a.accumulator.as_str() {
        "hashmap" => {
            stream_render(&ions, &g, |e| {
                max_active = max_active.max(e.active);
                max_buf = max_buf.max(e.buffer.len());
                total_peaks += e.buffer.len() as u64;
                n_frames_emitted += 1;
                if a.encode && err.is_ok() {
                    // Already unique per (scan, tof); the worker's dedup is then a no-op.
                    let triples: Vec<(u32, u32, f64)> =
                        e.buffer.iter().map(|(&(s, t), &v)| (s, t, v)).collect();
                    err = enc.push(e.frame, triples, &pool);
                }
            });
        }
        "flat" => {
            stream_render_flat(&ions, &g, |e| {
                max_active = max_active.max(e.active);
                max_buf = max_buf.max(e.triples.len());
                total_peaks += e.triples.len() as u64;
                n_frames_emitted += 1;
                if a.encode && err.is_ok() {
                    // Raw (duplicated) triples — the parallel worker sums them in f64.
                    err = enc.push(e.frame, e.triples.to_vec(), &pool);
                }
            });
        }
        other => anyhow::bail!("unknown --accumulator {other:?} (want 'hashmap' or 'flat')"),
    }
    err?;
    enc.finish(a.n_frames, &pool)?;
    let (encode_s, compressed_bytes, n_blocks) = (enc.encode_s, enc.compressed_bytes, enc.n_blocks);
    let sweep_s = t_sweep.elapsed().as_secs_f64();

    // The sweep's OWN working set: active heap (16 B per Reverse<(u32,usize)>) + sparse frame buffer
    // (~32 B per entry incl. HashMap control bytes). This is the bounded quantity — independent of the
    // O(total) load above, and the number that must stay flat as the run gets longer.
    let sweep_ws_mb = (max_active * 16 + max_buf * 32) as f64 / (1024.0 * 1024.0);

    let avg_peaks = if ions.is_empty() { 0.0 } else {
        ions.iter().map(|i| i.peaks.len()).sum::<usize>() as f64 / ions.len() as f64
    };

    println!();
    println!("  ── streaming render throughput benchmark ───────────────────");
    println!("  precursors            : {:>12}   ({:.1} peaks/ion avg)", ions.len(), avg_peaks);
    println!("  frames                : {:>12}   ({} emitted non-empty)", a.n_frames, n_frames_emitted);
    println!("  accumulator           : {:>12}", a.accumulator);
    println!("  peak ACTIVE set       : {:>12}   (~ elution width, NOT total precursors)", max_active);
    println!("  peak frame buffer     : {:>12}   {}", max_buf,
             if a.accumulator == "flat" { "(scan,tof,val) triples, pre-dedup" } else { "(scan,tof) entries" });
    println!("  SWEEP working set     : {:>12.2} MB   (active heap + frame buffer — bounded by elution \
             DENSITY, not run length)", sweep_ws_mb);
    println!("  total peaks emitted   : {:>12}", total_peaks);
    println!("  peak RSS              : {:>12.1} MB   (dominated by the O(total) load, not the sweep)",
             peak_rss_mb());

    // Separate the render cost from the encode cost so the two throughput ceilings are visible.
    let render_s = sweep_s - encode_s;
    println!("  render wall           : {:>12.2} s   ({:.0} frames/s, {:.1}M peaks/s)",
             render_s, n_frames_emitted as f64 / render_s.max(1e-9), total_peaks as f64 / render_s.max(1e-9) / 1e6);
    if a.encode {
        let gib = compressed_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        // n_blocks counts EVERY acquisition frame (signal-bearing + gap-filled empties), so the encode
        // throughput and the extrapolation are honest on sparse runs, not just dense ones.
        println!("  ENCODE wall           : {:>12.2} s   ({:.0} frames/s, {} threads)   [real Bruker \
                 TDF block encoder, {} blocks incl. empties]", encode_s,
                 n_blocks as f64 / encode_s.max(1e-9), a.encode_threads, n_blocks);
        println!("  compressed output     : {:>12.3} GiB  ({:.0} bytes/frame)", gib,
                 compressed_bytes as f64 / n_blocks.max(1) as f64);
        // Extrapolate a full 36 000-frame run at THIS density. Every frame gets a block, so the rate
        // is over total frames (a.n_frames), not just the signal-bearing ones.
        let e2e_fps = a.n_frames as f64 / sweep_s.max(1e-9);
        let full_run_s = 36_000.0 / e2e_fps;
        println!("  END-TO-END            : {:>12.2} s   ({:.0} frames/s render+encode)", sweep_s, e2e_fps);
        println!("  ⇒ extrapolated full 36k-frame run at this density: {:.1} min ({:.2} h)",
                 full_run_s / 60.0, full_run_s / 3600.0);
    } else {
        println!("  encode                :      (skipped; pass --encode with --features tdf \
                 for the real TDF block encoder + end-to-end throughput)");
    }
    println!("  conservation          : proven separately by the oracle — \
             `cargo test -p timsim-cli --lib render`");
    Ok(())
}
