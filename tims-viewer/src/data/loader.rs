//! Background streaming loader.
//!
//! Runs on its own thread, streams frames (real or synthetic), downsamples to the
//! point budget, packs `GpuPoint`s, and sends them to the render thread over a bounded
//! channel so the event loop never blocks on I/O and the cloud builds progressively.
//!
//! Sampling is **stratified**: a global systematic stride provides a count-bounded,
//! RT-unbiased density base (each kept point weighted `1/p` for brightness-invariant
//! additive rendering), and a coarse spatial grid additionally keeps the brightest
//! point in every occupied cell so sparse high-intensity features always survive.

use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};

use mscore::data::spectrum::MsType;
use rustdf::data::acquisition::AcquisitionMode;
use rustdf::data::dataset::TimsDataset;
use rustdf::data::handle::{IndexConverter, TimsData};
use rustdf::data::meta::{
    read_dda_precursor_meta, read_dia_ms_ms_info, read_dia_ms_ms_windows, read_meta_data_sql,
};

use super::demo::DemoSource;
use super::point::{AxisBounds, GpuPoint};
use crate::render::annotation::LineVertex;

/// Messages from the loader thread to the render thread.
///
/// `generation` is carried now but only consumed once region-refinement lands
/// (Phase 1.5); the single Phase-1 load uses generation 0.
#[allow(dead_code)]
pub enum LoadMsg {
    /// A batch of systematic (density-base) points belonging to `generation`.
    Chunk { generation: u64, points: Vec<GpuPoint> },
    /// A batch of peak points (per-cell brightest) — display-only enrichment for the
    /// point cloud; consumers must NOT add these to the volume density (they would
    /// double-count) and they are not part of the systematic density base.
    PeakChunk { points: Vec<GpuPoint> },
    /// Fractional load progress in `[0, 1]`.
    Progress(f32),
    /// Robust intensity percentiles (p1/p50/p99) for transfer-function defaults.
    Stats { i_min: f32, i_max: f32, i_med: f32 },
    /// Per-axis distribution histograms (each `HIST_BINS` long) for the levels-style filter
    /// UI. m/z·1/K0·RT bin linearly over the cube bounds; `intensity` is data-tight log,
    /// spanning `[i_lo, i_hi]` (the kept sample's min/max intensity).
    Histograms {
        mz: Vec<u32>,
        im: Vec<u32>,
        rt: Vec<u32>,
        intensity: Vec<u32>,
        i_lo: f32,
        i_hi: f32,
        /// 2D density projections (each `PROJ_BINS * PROJ_BINS`, row-major `x + PROJ_BINS*y`)
        /// for the minimaps: m/z×1/K0, m/z×RT, 1/K0×RT.
        proj_mz_im: Vec<u32>,
        proj_mz_rt: Vec<u32>,
        proj_im_rt: Vec<u32>,
    },
    /// Annotation overlay geometry: colored line-list vertices (pairs = segments) in the
    /// normalized cube (DDA precursor crosses / DIA isolation-window footprints). `groups`
    /// is a parallel per-vertex window-group id (u32::MAX = ungrouped); `n_groups` is the
    /// number of DIA/MIDIA window groups, for the per-group visibility UI.
    Annotations { lines: Vec<LineVertex>, groups: Vec<u32>, n_groups: u32 },
    /// All frames for `generation` have been streamed.
    Done { generation: u64 },
    /// Fatal error; loading stopped.
    Error(String),
}

/// Commands from the render thread to the loader thread.
pub enum LoadCmd {
    /// Abandon the current load and exit.
    Cancel,
}

/// Optional real-unit m/z + 1/K0 + intensity cull applied per-point during a region refinement
/// load (frame selection already handles the RT window). Points outside are skipped at the source,
/// so the budget concentrates on the focused region (the 4D lens — see FOCUS_LENS_PLAN.md).
#[derive(Clone, Copy)]
pub struct RegionFilter {
    pub mz: (f64, f64),
    pub im: (f64, f64),
    /// Per-point intensity floor (real counts); points below are dropped at the source.
    pub intensity_min: f32,
}

/// What to stream.
pub enum LoaderMode {
    Real { path: String, frame_ids: Vec<u32> },
    Demo(DemoSource),
}

/// Owns the loader thread and the channels to talk to it.
pub struct LoaderHandle {
    pub rx: Receiver<LoadMsg>,
    cmd_tx: Sender<LoadCmd>,
    handle: Option<JoinHandle<()>>,
}

impl LoaderHandle {
    pub fn spawn(
        mode: LoaderMode,
        bounds: AxisBounds,
        total_estimate: u64,
        budget: usize,
        filter: Option<RegionFilter>,
    ) -> Self {
        let (msg_tx, msg_rx) = bounded::<LoadMsg>(8);
        let (cmd_tx, cmd_rx) = bounded::<LoadCmd>(4);
        let handle = std::thread::Builder::new()
            .name("tims-loader".into())
            .spawn(move || {
                run_loader(mode, bounds, total_estimate, budget, filter, &msg_tx, &cmd_rx);
            })
            .expect("failed to spawn loader thread");
        LoaderHandle {
            rx: msg_rx,
            cmd_tx,
            handle: Some(handle),
        }
    }
}

impl Drop for LoaderHandle {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(LoadCmd::Cancel);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

const GENERATION: u64 = 0; // single generation in Phase 1
const CHUNK_POINTS: usize = 300_000;
const RESERVOIR_CAP: usize = 200_000;

/// Sampling stride derived from the budget.
///
/// Uses **ceiling** division so the kept count never exceeds the budget: keeping every
/// `stride`-th point yields `~total/stride <= budget` points. (Rounding instead would
/// let e.g. total=149, budget=100 round to stride 1 and emit all 149, biasing the view
/// toward early frames once the GPU buffer truncates the tail.)
pub fn stride_for(total_estimate: u64, budget: usize) -> usize {
    if total_estimate == 0 || budget == 0 {
        return 1;
    }
    let budget = budget as u64;
    if budget >= total_estimate {
        1
    } else {
        total_estimate.div_ceil(budget).max(1) as usize
    }
}

/// Coarse spatial grid for peak preservation (m/z, 1/K0, RT cells).
const PEAK_DIMS: [u32; 3] = [128, 64, 128];
/// Placeholder point for the peak-grid HashMap entry API.
const ZERO_POINT: GpuPoint = GpuPoint {
    pos: [0.0; 3],
    intensity: 0.0,
    weight: 1.0,
    flags: 0,
    _pad: [GpuPoint::NO_CLUSTER, 0],
};

/// Number of bins in each per-axis distribution histogram (for the levels-style filter UI).
pub const HIST_BINS: usize = 80;
/// Fine intermediate bins for the intensity log histogram before rebinning to the data range.
const LOGI_FINE: usize = 256;
/// Upper bound of the fixed intensity log range (log10), generous for timsTOF intensities.
const LOGI_HI: f32 = 8.0;

/// Bin a normalized-cube coordinate in `[-1, 1]` into `[0, HIST_BINS)`.
#[inline]
fn hbin(n: f32) -> usize {
    (((n * 0.5 + 0.5).clamp(0.0, 1.0)) * (HIST_BINS as f32 - 1.0)) as usize
}

/// Side length of each 2D projection minimap (m/z·1/K0·RT plane heatmaps).
pub const PROJ_BINS: usize = 96;
/// Bin a normalized-cube coordinate in `[-1, 1]` into `[0, PROJ_BINS)`.
#[inline]
fn pbin(n: f32) -> usize {
    (((n * 0.5 + 0.5).clamp(0.0, 1.0)) * (PROJ_BINS as f32 - 1.0)) as usize
}

/// Map a normalized-cube position to a coarse peak-grid cell index.
#[inline]
fn peak_cell(pos: [f32; 3]) -> u32 {
    let axis = |n: f32, dim: u32| -> u32 {
        (((n * 0.5 + 0.5).clamp(0.0, 1.0) * dim as f32) as u32).min(dim - 1)
    };
    let x = axis(pos[0], PEAK_DIMS[0]);
    let y = axis(pos[1], PEAK_DIMS[1]);
    let z = axis(pos[2], PEAK_DIMS[2]);
    x + PEAK_DIMS[0] * (y + PEAK_DIMS[1] * z)
}

#[allow(unused_assignments)] // final flush! reassigns `chunk` which is then dropped
fn run_loader(
    mode: LoaderMode,
    bounds: AxisBounds,
    total_estimate: u64,
    budget: usize,
    filter: Option<RegionFilter>,
    tx: &Sender<LoadMsg>,
    cmd_rx: &Receiver<LoadCmd>,
) {
    // Stratified sampling: spend ~85% of the budget on a systematic density base, and
    // reserve the rest for per-cell intensity PEAKS so sparse high-intensity features
    // always survive (pure systematic sampling can statistically miss them).
    let systematic_budget = (budget.saturating_mul(85) / 100).max(1);
    let stride = stride_for(total_estimate, systematic_budget);
    let weight = stride as f32;
    let mut systematic_count: u64 = 0;
    // Flat peak grid indexed by cell: (best intensity, source index, best point) over
    // ALL points. A direct-indexed Vec (no hashing) is far faster than a HashMap across
    // the hundreds of millions of per-point updates. Sentinel intensity -1 = unoccupied;
    // the source index lets us drop peaks the systematic sampler already emitted.
    let n_cells = (PEAK_DIMS[0] * PEAK_DIMS[1] * PEAK_DIMS[2]) as usize;
    let mut peaks: Vec<(f32, u64, GpuPoint)> = vec![(-1.0, 0, ZERO_POINT); n_cells];

    // Subsample intensities into a reservoir for percentile defaults.
    let mut reservoir: Vec<f32> = Vec::with_capacity(RESERVOIR_CAP);
    let mut kept_counter: u64 = 0;
    let sample_every = (budget / RESERVOIR_CAP).max(1) as u64;

    // Distribution histograms (over the kept systematic sample) for the levels-style filter
    // UI. m/z·1/K0·RT bin over the normalized cube; intensity accumulates into a fine fixed
    // log range, then rebins to the data's actual range at the end (so it is data-tight).
    let mut hist_mz = [0u32; HIST_BINS];
    let mut hist_im = [0u32; HIST_BINS];
    let mut hist_rt = [0u32; HIST_BINS];
    let mut logi_fine = [0u32; LOGI_FINE];
    let mut i_lo_seen = f32::MAX;
    let mut i_hi_seen = 0.0f32;
    // 2D density projections onto the coordinate planes (the "you are here" minimaps).
    let mut proj_mz_im = vec![0u32; PROJ_BINS * PROJ_BINS]; // x = m/z, y = 1/K0
    let mut proj_mz_rt = vec![0u32; PROJ_BINS * PROJ_BINS]; // x = m/z, y = RT
    let mut proj_im_rt = vec![0u32; PROJ_BINS * PROJ_BINS]; // x = 1/K0, y = RT

    let mut chunk: Vec<GpuPoint> = Vec::with_capacity(CHUNK_POINTS);

    macro_rules! cancelled {
        () => {
            matches!(cmd_rx.try_recv(), Ok(LoadCmd::Cancel))
        };
    }

    // Send a message, blocking politely if the channel is full but still watching for
    // cancellation so the loader can never wedge on a full channel during shutdown.
    macro_rules! send_cancellable {
        ($msg:expr) => {{
            let mut m = $msg;
            loop {
                match tx.try_send(m) {
                    Ok(()) => break,
                    Err(TrySendError::Full(back)) => {
                        if cancelled!() {
                            return;
                        }
                        m = back;
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(_) => return, // receiver gone
                }
            }
        }};
    }

    // Flush a chunk, blocking politely if the channel is full but still watching for
    // cancellation so the loader can never wedge.
    macro_rules! flush {
        () => {{
            if !chunk.is_empty() {
                let mut payload = std::mem::take(&mut chunk);
                loop {
                    match tx.try_send(LoadMsg::Chunk {
                        generation: GENERATION,
                        points: payload,
                    }) {
                        Ok(()) => break,
                        Err(TrySendError::Full(LoadMsg::Chunk { points, .. })) => {
                            if cancelled!() {
                                return;
                            }
                            payload = points;
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Err(_) => return, // receiver gone
                    }
                }
                chunk = Vec::with_capacity(CHUNK_POINTS);
            }
        }};
    }

    let n_frames = match &mode {
        LoaderMode::Real { frame_ids, .. } => frame_ids.len(),
        LoaderMode::Demo(d) => d.num_frames(),
    };
    if n_frames == 0 {
        let _ = tx.send(LoadMsg::Error("run has no frames".into()));
        return;
    }

    // Open the real dataset once (SDK-free, thread-safe converter).
    let dataset = match &mode {
        LoaderMode::Real { path, .. } => Some(TimsDataset::new("NO_SDK", path, false, false)),
        LoaderMode::Demo(_) => None,
    };

    let mut last_progress = -1.0f32;
    // Cumulative point index across the whole run; keeping every `stride`-th is global
    // systematic sampling (count-bounded, RT-unbiased).
    let mut global_i: u64 = 0;
    let stride_u64 = stride as u64;

    // Per-point handler shared by both sources: cull against the region filter first, then update
    // the peak grid for every SURVIVING point and emit every stride-th into the systematic base.
    macro_rules! handle_point {
        ($mz:expr, $im:expr, $rt:expr, $it:expr, $ms2:expr) => {{
            let intensity = $it;
            // Region cull (the 4D lens): drop points outside the m/z·1/K0·intensity window at the
            // source so the budget concentrates on the focused region. RT is handled by frame
            // selection. `global_i` is not advanced for culled points, so the systematic stride
            // counts only survivors.
            if let Some(f) = &filter {
                if $mz < f.mz.0 || $mz > f.mz.1 || $im < f.im.0 || $im > f.im.1
                    || intensity < f.intensity_min
                {
                    continue;
                }
            }
            let pos = bounds.normalize($mz, $im, $rt);
            let flags = if $ms2 { GpuPoint::MS2_FLAG } else { 0 };
            // Peak: keep the highest-intensity point per coarse cell.
            let slot = &mut peaks[peak_cell(pos) as usize];
            if intensity > slot.0 {
                slot.0 = intensity;
                slot.1 = global_i;
                slot.2 = GpuPoint { pos, intensity, weight: 1.0, flags, _pad: [GpuPoint::NO_CLUSTER, 0] };
            }
            // Systematic density base.
            if global_i % stride_u64 == 0 {
                chunk.push(GpuPoint { pos, intensity, weight, flags, _pad: [GpuPoint::NO_CLUSTER, 0] });
                systematic_count += 1;
                reservoir_push(&mut reservoir, &mut kept_counter, sample_every, intensity);
                // Distribution histograms over the kept (representative) sample.
                hist_mz[hbin(pos[0])] += 1;
                hist_im[hbin(pos[1])] += 1;
                hist_rt[hbin(pos[2])] += 1;
                logi_fine[((intensity.max(1.0).log10() / LOGI_HI) * LOGI_FINE as f32)
                    .clamp(0.0, (LOGI_FINE - 1) as f32) as usize] += 1;
                proj_mz_im[pbin(pos[0]) + PROJ_BINS * pbin(pos[1])] += 1;
                proj_mz_rt[pbin(pos[0]) + PROJ_BINS * pbin(pos[2])] += 1;
                proj_im_rt[pbin(pos[1]) + PROJ_BINS * pbin(pos[2])] += 1;
                i_lo_seen = i_lo_seen.min(intensity);
                i_hi_seen = i_hi_seen.max(intensity);
                if chunk.len() >= CHUNK_POINTS {
                    flush!();
                }
            }
            global_i += 1;
        }};
    }

    match &mode {
        LoaderMode::Real { frame_ids, .. } => {
            use rayon::prelude::*;
            let ds = dataset.as_ref().unwrap();
            // Decode frames in parallel batches (the heavy, CPU-bound part — every core helps),
            // then fold the decoded points in sequentially so the systematic sampler, peak grid
            // and histograms stay exactly correct. One batch is in flight to bound memory.
            const DECODE_BATCH: usize = 64;
            let mut start = 0;
            while start < n_frames {
                if cancelled!() {
                    return;
                }
                let end = (start + DECODE_BATCH).min(n_frames);
                let frames: Vec<_> = frame_ids[start..end]
                    .par_iter()
                    .map(|&fid| ds.get_frame(fid))
                    .collect();
                for frame in &frames {
                    let rt = frame.ims_frame.retention_time;
                    let is_ms2 = !matches!(frame.ms_type, MsType::Precursor);
                    let mz = frame.ims_frame.mz.as_slice();
                    let im = frame.ims_frame.mobility.as_slice();
                    let it = frame.ims_frame.intensity.as_slice();
                    // Validate parallel arrays — never zip past the shortest.
                    let n = mz.len().min(im.len()).min(it.len());
                    for j in 0..n {
                        handle_point!(mz[j], im[j], rt, it[j] as f32, is_ms2);
                    }
                }
                start = end;
                let progress = end as f32 / n_frames as f32;
                if progress - last_progress >= 0.01 {
                    let _ = tx.try_send(LoadMsg::Progress(progress));
                    last_progress = progress;
                }
            }
        }
        LoaderMode::Demo(d) => {
            for idx in 0..n_frames {
                if cancelled!() {
                    return;
                }
                for p in d.frame(idx) {
                    handle_point!(p.mz, p.im, p.rt, p.intensity as f32, p.is_ms2);
                }
                let progress = (idx + 1) as f32 / n_frames as f32;
                if progress - last_progress >= 0.01 {
                    let _ = tx.try_send(LoadMsg::Progress(progress));
                    last_progress = progress;
                }
            }
        }
    }

    flush!();

    // Emit the brightest peaks to fill the reserved budget headroom, but only peaks the
    // systematic sampler did NOT already emit (source index not a multiple of stride) so
    // they add genuinely-new sparse points rather than duplicates. Sort by intensity so
    // the strongest survive if occupied cells exceed the headroom.
    let peak_room = budget.saturating_sub(systematic_count as usize);
    if peak_room > 0 && stride_u64 > 1 {
        let mut peak_pts: Vec<GpuPoint> = peaks
            .into_iter()
            .filter(|(i, gi, _)| *i >= 0.0 && gi % stride_u64 != 0)
            .map(|(_, _, gp)| gp)
            .collect();
        // Strongest first.
        peak_pts.sort_by(|a, b| {
            b.intensity
                .partial_cmp(&a.intensity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        peak_pts.truncate(peak_room);
        // Fold peaks into the distributions + intensity range too: they are displayed points
        // and are often the brightest in the run, so excluding them would make the default
        // intensity filter (which uses i_lo/i_hi) cull the very peaks preservation kept.
        for gp in &peak_pts {
            hist_mz[hbin(gp.pos[0])] += 1;
            hist_im[hbin(gp.pos[1])] += 1;
            hist_rt[hbin(gp.pos[2])] += 1;
            logi_fine[((gp.intensity.max(1.0).log10() / LOGI_HI) * LOGI_FINE as f32)
                .clamp(0.0, (LOGI_FINE - 1) as f32) as usize] += 1;
            proj_mz_im[pbin(gp.pos[0]) + PROJ_BINS * pbin(gp.pos[1])] += 1;
            proj_mz_rt[pbin(gp.pos[0]) + PROJ_BINS * pbin(gp.pos[2])] += 1;
            proj_im_rt[pbin(gp.pos[1]) + PROJ_BINS * pbin(gp.pos[2])] += 1;
            i_lo_seen = i_lo_seen.min(gp.intensity);
            i_hi_seen = i_hi_seen.max(gp.intensity);
        }
        for batch in peak_pts.chunks(CHUNK_POINTS) {
            send_cancellable!(LoadMsg::PeakChunk {
                points: batch.to_vec(),
            });
        }
    }

    // Annotation overlay (real data only — needs the dataset's scan->mobility converter
    // and the DDA/DIA metadata tables).
    if let (LoaderMode::Real { path, .. }, Some(ds)) = (&mode, &dataset) {
        let (lines, groups, n_groups) = build_annotations(ds, path, &bounds);
        if !lines.is_empty() {
            send_cancellable!(LoadMsg::Annotations { lines, groups, n_groups });
        }
    }

    // Robust intensity range from the reservoir.
    if !reservoir.is_empty() {
        reservoir.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p = |q: f32| {
            let i = ((reservoir.len() as f32 - 1.0) * q).round() as usize;
            reservoir[i.min(reservoir.len() - 1)]
        };
        let i_min = p(0.01).max(1.0);
        let i_med = p(0.50).max(i_min);
        let i_max = p(0.99).max(i_med * 1.0001);
        send_cancellable!(LoadMsg::Stats { i_min, i_max, i_med });
    }

    // Rebin the fine fixed-range intensity histogram into the data's actual log range so the
    // levels strip is data-tight, and ship all four distributions to the UI.
    if i_hi_seen > 0.0 {
        let i_lo = i_lo_seen.max(1.0);
        let i_hi = i_hi_seen.max(i_lo * 1.0001);
        let (llo, lhi) = (i_lo.log10(), i_hi.log10());
        let span = (lhi - llo).max(1e-6);
        let mut hist_i = vec![0u32; HIST_BINS];
        for (src, &c) in logi_fine.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let src_log = ((src as f32 + 0.5) / LOGI_FINE as f32) * LOGI_HI;
            let t = (((src_log - llo) / span) * HIST_BINS as f32) as i32;
            if t >= 0 && (t as usize) < HIST_BINS {
                hist_i[t as usize] += c;
            }
        }
        send_cancellable!(LoadMsg::Histograms {
            mz: hist_mz.to_vec(),
            im: hist_im.to_vec(),
            rt: hist_rt.to_vec(),
            intensity: hist_i,
            i_lo,
            i_hi,
            proj_mz_im,
            proj_mz_rt,
            proj_im_rt,
        });
    }

    send_cancellable!(LoadMsg::Done {
        generation: GENERATION,
    });
}

/// Build annotation-overlay line geometry (normalized cube) from the run's DDA/DIA
/// metadata. DDA precursors -> small 3D crosses; DIA isolation windows -> wireframe
/// boxes. Returns line-list vertices (consecutive pairs are segments).
fn build_annotations(
    ds: &TimsDataset,
    path: &str,
    bounds: &AxisBounds,
) -> (Vec<LineVertex>, Vec<u32>, u32) {
    // frame_id -> retention time (ids are contiguous 1..=N, validated at load).
    let meta = match read_meta_data_sql(path) {
        Ok(m) => m,
        Err(_) => return (Vec::new(), Vec::new(), 0),
    };
    let max_id = meta.iter().map(|m| m.id).max().unwrap_or(0).max(0) as usize;
    let mut rt_by = vec![0f64; max_id + 1];
    for m in &meta {
        if m.id >= 0 {
            rt_by[m.id as usize] = m.time;
        }
    }
    let rt_of = |fid: u32| rt_by.get(fid as usize).copied().unwrap_or(0.0);

    // `groups` is a per-vertex window-group id, parallel to `lines`, for CPU-side group
    // toggling. u32::MAX means "ungrouped, always shown" (DDA precursor crosses).
    let mut lines: Vec<LineVertex> = Vec::new();
    let mut groups: Vec<u32> = Vec::new();
    let mut n_groups_out: u32 = 0;
    match ds.get_acquisition_mode() {
        AcquisitionMode::DDA | AcquisitionMode::PRECURSOR => {
            if let Ok(precursors) = read_dda_precursor_meta(path) {
                for p in precursors {
                    let fid = p.precursor_frame_id.max(0) as u32;
                    let scan = p.precursor_average_scan_number.round().max(0.0) as u32;
                    let im = ds
                        .scan_to_inverse_mobility(fid, &vec![scan])
                        .first()
                        .copied()
                        .unwrap_or(0.0);
                    let pos = bounds.normalize(p.precursor_mz_highest_intensity, im, rt_of(fid));
                    push_cross(&mut lines, &mut groups, pos, 0.012, [0.1, 0.95, 0.95], u32::MAX);
                }
            }
        }
        AcquisitionMode::DIA => {
            if let (Ok(windows), Ok(info)) =
                (read_dia_ms_ms_windows(path), read_dia_ms_ms_info(path))
            {
                // A representative frame per window group, for scan->mobility conversion.
                let mut grp_frame: std::collections::HashMap<u32, u32> = Default::default();
                for di in &info {
                    grp_frame.entry(di.window_group).or_insert(di.frame_id);
                }
                // The isolation scheme repeats every cycle. Drawing each window as one
                // full-RT box piles every window face onto the two RT end-walls (leaving only
                // thin edges through the middle), which reads as a slab beside the data.
                // Instead draw the (m/z, mobility) selection footprint at several evenly
                // spaced RT slices, so the recurring selection sits on the precursor signal
                // through the whole run.
                // The fine diagonal has ~950 touching windows per group; drawing every one
                // fills the band solid and hides the data. Subsample to leave gaps you can
                // see the cloud through, and span each drawn rect across the skipped scans so
                // it stays a visible window outline rather than a sliver.
                const N_SLICES: usize = 6;
                const TARGET_WINDOWS: usize = 800;
                // Subsample so a fine MIDIA diagonal (~15k windows) stays legible, but a
                // conventional DIA scheme (tens of windows) keeps every box (stride == 1).
                let stride = (windows.len() / TARGET_WINDOWS).max(1);
                // Color each window by its group, so the interleaved isolation diagonals
                // that tile the precursor space read as distinct bands.
                let n_groups = windows.iter().map(|w| w.window_group).max().unwrap_or(1).max(1);
                n_groups_out = n_groups;
                let max_scan = windows.iter().map(|w| w.scan_num_end).max().unwrap_or(0);
                let (rt_lo, rt_hi) = (bounds.rt.min, bounds.rt.max);
                for w in windows.iter().step_by(stride) {
                    let fid = grp_frame.get(&w.window_group).copied().unwrap_or(1);
                    // Widen SYMMETRICALLY across the skipped scans so the rect stays a visible
                    // window AND stays centered on the real window's mobility: widening only the
                    // high-scan end would shift the box's 1/K0 center toward lower mobility for a
                    // subsampled MIDIA scheme. stride == 1 (conventional DIA) leaves it exact.
                    let half = stride as u32 / 2;
                    let scan_lo = w.scan_num_begin.saturating_sub(half);
                    let scan_hi = (w.scan_num_end + half).min(max_scan.max(w.scan_num_end));
                    let ims = ds.scan_to_inverse_mobility(fid, &vec![scan_lo, scan_hi]);
                    let im0 = ims.first().copied().unwrap_or(0.0);
                    let im1 = ims.get(1).copied().unwrap_or(im0);
                    let mz0 = w.isolation_mz - w.isolation_width * 0.5;
                    let mz1 = w.isolation_mz + w.isolation_width * 0.5;
                    let color = crate::render::colormap::group_color(w.window_group, n_groups);
                    for i in 0..N_SLICES {
                        let rt =
                            rt_lo + (rt_hi - rt_lo) * (i as f64 + 0.5) / N_SLICES as f64;
                        push_rect_mz_im(
                            &mut lines, &mut groups, bounds, (mz0, mz1), (im0, im1), rt, color,
                            w.window_group,
                        );
                    }
                }
            }
        }
        AcquisitionMode::Unknown => {}
    }
    (lines, groups, n_groups_out)
}

/// Append a 3-segment axis-aligned cross centered at `c` (half-length `h`), in `color`,
/// tagging each vertex with `group`.
fn push_cross(
    lines: &mut Vec<LineVertex>,
    groups: &mut Vec<u32>,
    c: [f32; 3],
    h: f32,
    color: [f32; 3],
    group: u32,
) {
    let mut v = |p: [f32; 3]| {
        lines.push(LineVertex::new(p, color));
        groups.push(group);
    };
    v([c[0] - h, c[1], c[2]]);
    v([c[0] + h, c[1], c[2]]);
    v([c[0], c[1] - h, c[2]]);
    v([c[0], c[1] + h, c[2]]);
    v([c[0], c[1], c[2] - h]);
    v([c[0], c[1], c[2] + h]);
}

/// Append the 4 edges of an axis-aligned rectangle in the m/z x mobility plane at a fixed
/// retention time — one isolation window's footprint, drawn at a given RT slice.
fn push_rect_mz_im(
    lines: &mut Vec<LineVertex>,
    groups: &mut Vec<u32>,
    bounds: &AxisBounds,
    mz: (f64, f64),
    im: (f64, f64),
    rt: f64,
    color: [f32; 3],
    group: u32,
) {
    let corner = |a: f64, b: f64| bounds.normalize(a, b, rt);
    let c = [
        corner(mz.0, im.0),
        corner(mz.1, im.0),
        corner(mz.1, im.1),
        corner(mz.0, im.1),
    ];
    const EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];
    for (a, b) in EDGES {
        lines.push(LineVertex::new(c[a], color));
        lines.push(LineVertex::new(c[b], color));
        groups.push(group);
        groups.push(group);
    }
}

#[inline]
fn reservoir_push(reservoir: &mut Vec<f32>, counter: &mut u64, sample_every: u64, v: f32) {
    if *counter % sample_every == 0 && reservoir.len() < RESERVOIR_CAP {
        reservoir.push(v);
    }
    *counter = counter.wrapping_add(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::point::{AxisBounds, AxisTransform};
    use std::time::Duration;

    fn demo_bounds() -> AxisBounds {
        AxisBounds {
            mz: AxisTransform::new(100.0, 1700.0),
            im: AxisTransform::new(0.6, 1.6),
            rt: AxisTransform::new(0.0, 10.0),
        }
    }

    #[test]
    fn stride_matches_budget_ratio() {
        assert_eq!(stride_for(100, 100), 1);
        assert_eq!(stride_for(100, 1000), 1); // budget >= total -> keep all
        assert_eq!(stride_for(1000, 100), 10);
        assert_eq!(stride_for(0, 100), 1); // no divide-by-zero
        // Ceiling, not rounding: 149/100 must give 2 (keeps ~75 <= budget), never 1
        // (which would emit all 149 and truncate the tail at the GPU buffer).
        assert_eq!(stride_for(149, 100), 2);
        // Ceiling never lets total/stride exceed the budget.
        for (total, budget) in [(149u64, 100usize), (1_000_000, 25_000), (7, 3)] {
            let s = stride_for(total, budget) as u64;
            assert!(total / s <= budget as u64, "total={total} budget={budget} s={s}");
        }
    }

    #[test]
    fn cumulative_systematic_sampling_is_bounded() {
        // The loader keeps every stride-th point by a cumulative global index, which is
        // global systematic sampling: kept == ceil(total/stride), independent of how the
        // points are split into frames.
        for stride in [2u64, 3, 5, 7] {
            let frame_lens = [1u64, 2, 1, 0, 9, 4, 3, 1, 1, 1];
            let total: u64 = frame_lens.iter().sum();
            let mut global_i = 0u64;
            let mut kept = 0u64;
            for &n in &frame_lens {
                for _ in 0..n {
                    if global_i % stride == 0 {
                        kept += 1;
                    }
                    global_i += 1;
                }
            }
            let bound = total.div_ceil(stride);
            assert_eq!(kept, bound, "stride={stride}: kept {kept} != ceil bound {bound}");
        }
    }

    #[test]
    fn peak_cell_in_range_and_distinct() {
        // Corners map inside the grid; opposite corners differ.
        let lo = peak_cell([-1.0, -1.0, -1.0]);
        let hi = peak_cell([1.0, 1.0, 1.0]);
        let max = PEAK_DIMS[0] * PEAK_DIMS[1] * PEAK_DIMS[2];
        assert_eq!(lo, 0);
        assert!(hi < max && hi != lo);
        // Out-of-range clamps rather than overflowing.
        assert!(peak_cell([5.0, -5.0, 2.0]) < max);
    }

    #[test]
    fn demo_loader_streams_and_respects_budget() {
        let total = 100_000u64;
        let budget = 50_000usize;
        let demo = DemoSource::new(20, total);
        let handle =
            LoaderHandle::spawn(LoaderMode::Demo(demo), demo_bounds(), total, budget, None);

        let mut points = 0usize;
        let mut saw_stats = false;
        loop {
            match handle.rx.recv_timeout(Duration::from_secs(15)) {
                Ok(LoadMsg::Chunk { points: p, .. }) => {
                    // Every packed point must be finite and inside the cube.
                    for gp in &p {
                        assert!(gp.pos.iter().all(|c| c.is_finite()));
                        assert!(gp.weight >= 1.0);
                    }
                    points += p.len();
                }
                Ok(LoadMsg::Stats { i_min, i_max, i_med }) => {
                    assert!(i_min > 0.0 && i_max > i_min);
                    assert!(i_med >= i_min && i_med <= i_max);
                    saw_stats = true;
                }
                // The only non-panic exit: reaching past the loop proves Done arrived.
                Ok(LoadMsg::Done { .. }) => break,
                Ok(LoadMsg::PeakChunk { points: p }) => {
                    points += p.len();
                }
                Ok(LoadMsg::Error(e)) => panic!("loader error: {e}"),
                Ok(LoadMsg::Progress(_))
                | Ok(LoadMsg::Annotations { .. })
                | Ok(LoadMsg::Histograms { .. }) => {}
                Err(e) => panic!("loader timed out / disconnected: {e}"),
            }
        }
        assert!(saw_stats, "loader never sent intensity Stats");
        assert!(points > 0, "loader produced no points");
        // ~budget points (stride=2 over 100k); allow generous slack.
        assert!(points <= 70_000, "downsample exceeded budget: {points}");
    }

    /// Drain a loader run, returning (systematic base count, peak count).
    fn drain(handle: LoaderHandle) -> (usize, usize) {
        let (mut sys, mut pk) = (0usize, 0usize);
        loop {
            match handle.rx.recv_timeout(Duration::from_secs(30)) {
                Ok(LoadMsg::Chunk { points, .. }) => sys += points.len(),
                Ok(LoadMsg::PeakChunk { points }) => pk += points.len(),
                Ok(LoadMsg::Done { .. }) => break,
                Ok(LoadMsg::Error(e)) => panic!("loader error: {e}"),
                Ok(_) => {}
                Err(e) => panic!("loader timed out / disconnected: {e}"),
            }
        }
        (sys, pk)
    }

    /// FOCUS_LENS_PLAN.md blocker-1 proof: a region load must spend its budget on the *region*,
    /// not stay over-strided on the full-run estimate. Drives the systematic-base count.
    #[test]
    fn region_estimate_concentrates_budget() {
        let total = 200_000u64;
        let demo = || DemoSource::new(30, total);
        // m/z sub-range; im wide open so only m/z culls (isolates a clean strict subset).
        let region = Some(RegionFilter { mz: (100.0, 500.0), im: (-10.0, 10.0), intensity_min: 0.0 });

        // True survivors in the region (stride 1: budget exceeds everything).
        let (survivors, _) =
            drain(LoaderHandle::spawn(LoaderMode::Demo(demo()), demo_bounds(), total, total as usize * 2, region));
        assert!(survivors > 1000, "region too small to test: {survivors}");
        assert!((survivors as u64) < total, "region should be a strict subset of the run");

        let budget = (survivors / 2).max(2000); // below survivors so sampling is actually active

        // Region-relative estimate (= survivors): the systematic base should ~fill the budget.
        let (sys_region, _) =
            drain(LoaderHandle::spawn(LoaderMode::Demo(demo()), demo_bounds(), survivors as u64, budget, region));
        // Full-run estimate (the bug): stride stays coarse -> budget badly under-spent.
        let (sys_full, _) =
            drain(LoaderHandle::spawn(LoaderMode::Demo(demo()), demo_bounds(), total, budget, region));

        // Exact: the systematic base keeps every `stride`-th SURVIVOR, where the stride is sized to
        // the survivor count. This equals ceil(survivors/stride) only if `global_i` advances solely
        // for survivors — it would fail if culled points still advanced the index.
        let systematic_budget = (budget * 85 / 100).max(1);
        let stride = stride_for(survivors as u64, systematic_budget) as u64;
        let expected = (survivors as u64).div_ceil(stride) as usize;
        assert_eq!(
            sys_region, expected,
            "region systematic count must be exactly ceil(survivors/stride): \
             survivors={survivors} stride={stride}"
        );
        // The full-run estimate sizes the stride to ~total, spending only ~(survivors/total) of the
        // budget — far less than the region estimate. This is the blocker-1 regression guard.
        assert!(
            (sys_full as f64) < 0.5 * sys_region as f64,
            "full-run estimate did not under-spend (blocker-1 unfixed): full={sys_full} region={sys_region}"
        );
    }
}
