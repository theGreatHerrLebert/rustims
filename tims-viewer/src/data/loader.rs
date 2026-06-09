//! Background streaming loader.
//!
//! Runs on its own thread, streams frames (real or synthetic), downsamples to the
//! point budget, packs `GpuPoint`s, and sends them to the render thread over a bounded
//! channel so the event loop never blocks on I/O and the cloud builds progressively.
//!
//! Phase 1 uses **per-frame stride sampling** with an exact per-frame quota (so the
//! total can never exceed the budget) and stores each kept point's weight `1/p` for
//! brightness-invariant additive rendering. Stratified, peak-preserving sampling is the
//! Phase 1.5 / Phase 3 upgrade (see the plan); the message/generation plumbing here is
//! already shaped for it.

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

/// Offset of the first kept index in a frame, given how many source points were already
/// seen (`phase = points_seen % stride`). Keeping `offset, offset+stride, ...` in every
/// frame is exactly global systematic sampling over the concatenated point stream, so
/// the total kept is `ceil(total/stride) <= budget` and each kept point truly stands in
/// for `stride` source points (weight `1/p`).
#[inline]
fn systematic_offset(phase: usize, stride: usize) -> usize {
    (stride - phase % stride) % stride
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
    let stride = stride_for(total_estimate, budget);
    let weight = stride as f32;
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
    // Phase carried ACROSS frames so sampling is global systematic, not per-frame
    // (per-frame resets overshoot the budget on short frames and bias the kept tail).
    let mut phase = 0usize;

    for idx in 0..n_frames {
        if cancelled!() {
            return;
        }

        // Gather this frame's points as (pos_norm, intensity, is_ms2).
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
                let offset = systematic_offset(phase, stride);
                phase = (phase + n) % stride;
                let mut j = offset;
                while j < n {
                    push_point(
                        &mut chunk,
                        &bounds,
                        mz[j],
                        im[j],
                        rt,
                        it[j] as f32,
                        weight,
                        is_ms2,
                    );
                    reservoir_push(&mut reservoir, &mut kept_counter, sample_every, it[j] as f32);
                    if chunk.len() >= CHUNK_POINTS {
                        flush!();
                    }
                    j += stride;
                }
            }
            LoaderMode::Demo(d) => {
                let pts = d.frame(idx);
                let n = pts.len();
                let offset = systematic_offset(phase, stride);
                phase = (phase + n) % stride;
                let mut j = offset;
                while j < n {
                    let p = pts[j];
                    push_point(
                        &mut chunk,
                        &bounds,
                        p.mz,
                        p.im,
                        p.rt,
                        p.intensity as f32,
                        weight,
                        p.is_ms2,
                    );
                    reservoir_push(
                        &mut reservoir,
                        &mut kept_counter,
                        sample_every,
                        p.intensity as f32,
                    );
                    if chunk.len() >= CHUNK_POINTS {
                        flush!();
                    }
                    j += stride;
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
fn push_point(
    chunk: &mut Vec<GpuPoint>,
    bounds: &AxisBounds,
    mz: f64,
    im: f64,
    rt: f64,
    intensity: f32,
    weight: f32,
    is_ms2: bool,
) {
    let pos = bounds.normalize(mz, im, rt);
    chunk.push(GpuPoint {
        pos,
        intensity,
        weight,
        flags: if is_ms2 { GpuPoint::MS2_FLAG } else { 0 },
        _pad: [0, 0],
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
    fn systematic_sampling_bounded_over_uneven_frames() {
        // Carrying phase across frames keeps exactly ceil(total/stride) points, even
        // with short/empty frames — per-frame offset resets would overshoot this.
        for stride in [2usize, 3, 5, 7] {
            let frame_lens = [1usize, 2, 1, 0, 9, 4, 3, 1, 1, 1];
            let total: usize = frame_lens.iter().sum();
            let mut phase = 0usize;
            let mut kept = 0usize;
            for &n in &frame_lens {
                let offset = systematic_offset(phase, stride);
                phase = (phase + n) % stride;
                let mut j = offset;
                while j < n {
                    kept += 1;
                    j += stride;
                }
            }
            let bound = total.div_ceil(stride);
            assert_eq!(kept, bound, "stride={stride}: kept {kept} != ceil bound {bound}");
        }
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
