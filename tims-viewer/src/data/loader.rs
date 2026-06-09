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
use rustdf::data::dataset::TimsDataset;
use rustdf::data::handle::TimsData;

use super::demo::DemoSource;
use super::point::{AxisBounds, GpuPoint};

/// Messages from the loader thread to the render thread.
///
/// `generation` is carried now but only consumed once region-refinement lands
/// (Phase 1.5); the single Phase-1 load uses generation 0.
#[allow(dead_code)]
pub enum LoadMsg {
    /// A batch of packed points belonging to `generation`.
    Chunk { generation: u64, points: Vec<GpuPoint> },
    /// Fractional load progress in `[0, 1]`.
    Progress(f32),
    /// Robust intensity range (p1/p99) for transfer-function defaults.
    Stats { i_min: f32, i_max: f32 },
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
    ) -> Self {
        let (msg_tx, msg_rx) = bounded::<LoadMsg>(8);
        let (cmd_tx, cmd_rx) = bounded::<LoadCmd>(4);
        let handle = std::thread::Builder::new()
            .name("tims-loader".into())
            .spawn(move || {
                run_loader(mode, bounds, total_estimate, budget, &msg_tx, &cmd_rx);
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
    _pad: [0, 0],
};

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
    tx: &Sender<LoadMsg>,
    cmd_rx: &Receiver<LoadCmd>,
) {
    // Stratified sampling: spend ~85% of the budget on a systematic density base, and
    // reserve the rest for per-cell intensity PEAKS so sparse high-intensity features
    // always survive (pure systematic sampling can statistically miss them).
    let systematic_budget = ((budget * 85) / 100).max(1);
    let stride = stride_for(total_estimate, systematic_budget);
    let weight = stride as f32;
    let mut systematic_count: u64 = 0;
    // Flat peak grid indexed by cell: (best intensity, best point) over ALL points.
    // A direct-indexed Vec (no hashing) is far faster than a HashMap across the
    // hundreds of millions of per-point updates. Sentinel intensity -1 = unoccupied.
    let n_cells = (PEAK_DIMS[0] * PEAK_DIMS[1] * PEAK_DIMS[2]) as usize;
    let mut peaks: Vec<(f32, GpuPoint)> = vec![(-1.0, ZERO_POINT); n_cells];

    // Subsample intensities into a reservoir for percentile defaults.
    let mut reservoir: Vec<f32> = Vec::with_capacity(RESERVOIR_CAP);
    let mut kept_counter: u64 = 0;
    let sample_every = (budget / RESERVOIR_CAP).max(1) as u64;

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

    // Per-point handler shared by both sources: update the peak grid for EVERY point,
    // and emit every stride-th into the systematic density base.
    macro_rules! handle_point {
        ($mz:expr, $im:expr, $rt:expr, $it:expr, $ms2:expr) => {{
            let intensity = $it;
            let pos = bounds.normalize($mz, $im, $rt);
            let flags = if $ms2 { GpuPoint::MS2_FLAG } else { 0 };
            // Peak: keep the highest-intensity point per coarse cell.
            let slot = &mut peaks[peak_cell(pos) as usize];
            if intensity > slot.0 {
                slot.0 = intensity;
                slot.1 = GpuPoint { pos, intensity, weight: 1.0, flags, _pad: [0, 0] };
            }
            // Systematic density base.
            if global_i % stride_u64 == 0 {
                chunk.push(GpuPoint { pos, intensity, weight, flags, _pad: [0, 0] });
                systematic_count += 1;
                reservoir_push(&mut reservoir, &mut kept_counter, sample_every, intensity);
                if chunk.len() >= CHUNK_POINTS {
                    flush!();
                }
            }
            global_i += 1;
        }};
    }

    for idx in 0..n_frames {
        if cancelled!() {
            return;
        }
        match &mode {
            LoaderMode::Real { frame_ids, .. } => {
                let ds = dataset.as_ref().unwrap();
                let frame = ds.get_frame(frame_ids[idx]);
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
            LoaderMode::Demo(d) => {
                for p in d.frame(idx) {
                    handle_point!(p.mz, p.im, p.rt, p.intensity as f32, p.is_ms2);
                }
            }
        }

        let progress = (idx + 1) as f32 / n_frames as f32;
        if progress - last_progress >= 0.01 {
            let _ = tx.try_send(LoadMsg::Progress(progress));
            last_progress = progress;
        }
    }

    flush!();

    // Emit the brightest peaks to fill the reserved budget headroom. Sort by intensity
    // so the strongest survive if there are more occupied cells than headroom.
    let peak_room = budget.saturating_sub(systematic_count as usize);
    if peak_room > 0 {
        let mut peak_pts: Vec<(f32, GpuPoint)> =
            peaks.into_iter().filter(|(i, _)| *i >= 0.0).collect();
        peak_pts.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, gp) in peak_pts.into_iter().take(peak_room) {
            chunk.push(gp);
            if chunk.len() >= CHUNK_POINTS {
                flush!();
            }
        }
        flush!();
    }

    // Robust intensity range from the reservoir.
    if !reservoir.is_empty() {
        reservoir.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p = |q: f32| {
            let i = ((reservoir.len() as f32 - 1.0) * q).round() as usize;
            reservoir[i.min(reservoir.len() - 1)]
        };
        let i_min = p(0.01).max(1.0);
        let i_max = p(0.99).max(i_min * 1.0001);
        send_cancellable!(LoadMsg::Stats { i_min, i_max });
    }

    send_cancellable!(LoadMsg::Done {
        generation: GENERATION,
    });
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
            LoaderHandle::spawn(LoaderMode::Demo(demo), demo_bounds(), total, budget);

        let mut points = 0usize;
        let mut saw_done = false;
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
                Ok(LoadMsg::Stats { i_min, i_max }) => {
                    assert!(i_min > 0.0 && i_max > i_min);
                    saw_stats = true;
                }
                Ok(LoadMsg::Done { .. }) => {
                    saw_done = true;
                    break;
                }
                Ok(LoadMsg::Error(e)) => panic!("loader error: {e}"),
                Ok(LoadMsg::Progress(_)) => {}
                Err(e) => panic!("loader timed out / disconnected: {e}"),
            }
        }
        assert!(saw_done, "loader never signaled Done");
        assert!(saw_stats, "loader never sent intensity Stats");
        assert!(points > 0, "loader produced no points");
        // ~budget points (stride=2 over 100k); allow generous slack.
        assert!(points <= 70_000, "downsample exceeded budget: {points}");
    }
}
