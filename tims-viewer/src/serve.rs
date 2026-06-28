//! Point-serving HTTP server with on-demand region queries (FOCUS_LENS_PLAN.md A1).
//!
//! Holds the full run's metadata and answers region-scoped loads on demand:
//!
//! ```text
//!   GET /points[?n=&mz0=&mz1=&im0=&im1=&rt0=&rt1=&imin=]  -> packed little-endian GpuPoint bytes
//!   GET /meta  [same query]                               -> JSON (region bounds, percentiles, hists)
//! ```
//!
//! No query => the full run. A query selects a 4D region (RT realized by frame selection; m/z·1/K0·
//! intensity culled at the source) and the budget concentrates on it. Each built load is cached by
//! `(region, budget)` and shared between `/points` and `/meta`. Several worker threads serve
//! concurrently so a multi-second region load can't stall other requests. Localhost only.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::app::Plan;
use crate::data::demo::DemoSource;
use crate::data::loader::{stride_for, LoadMsg, LoaderHandle, LoaderMode, RegionFilter};
use crate::data::meta::MetaIndex;
use crate::data::point::{AxisBounds, AxisTransform, GpuPoint};

/// Worker threads pulling requests off the shared server (so one slow load can't block others).
const WORKERS: usize = 4;
const I_HIST_BINS: usize = 80;

/// A 4D region of the run in real units (RT is realized via frame selection).
#[derive(Clone, Copy)]
struct Region4D {
    mz: (f64, f64),
    im: (f64, f64),
    rt: (f64, f64),
    imin: f32,
}

/// A built, cached load: shared bytes for `/points` and `/meta`, plus the point count.
struct LoadResult {
    points: Arc<[u8]>,
    meta: Arc<[u8]>,
    n_points: usize,
}

/// Output of a build: the cacheable result plus the intensity reference the full-run build seeds
/// for later region survivor estimates.
struct Built {
    result: LoadResult,
    i_hist: Vec<u32>,
    i_p99: f32,
}

/// Quantized cache key (regions are f64, so not directly Hash/Eq).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct LoadKey {
    mz0: i64,
    mz1: i64,
    im0: i64,
    im1: i64,
    rt0: i64,
    rt1: i64,
    imin: i64,
    budget: u64,
}

impl LoadKey {
    fn of(r: &Region4D, budget: usize) -> Self {
        let q = |x: f64, s: f64| (x * s).round() as i64;
        LoadKey {
            mz0: q(r.mz.0, 1e3),
            mz1: q(r.mz.1, 1e3),
            im0: q(r.im.0, 1e6), // 1/K0 spans ~1 unit — quantize finely
            im1: q(r.im.1, 1e6),
            rt0: q(r.rt.0, 1e3),
            rt1: q(r.rt.1, 1e3),
            imin: q(r.imin as f64, 1e3),
            budget: budget as u64,
        }
    }
}

struct State {
    meta: MetaIndex,
    is_demo: bool,
    default_budget: usize,
    /// Full-run intensity distribution, for estimating region survivor counts vs an intensity floor.
    full_i_hist: Vec<u32>,
    full_i_p99: f32,
    cache: Mutex<HashMap<LoadKey, Arc<LoadResult>>>,
}

/// Build the full run eagerly (fast first paint + the estimate reference), then serve region
/// queries on demand from `WORKERS` threads.
pub fn serve(plan: Plan, port: u16) -> Result<()> {
    anyhow::ensure!(
        cfg!(target_endian = "little"),
        "the point wire format is little-endian; --serve is unsupported on this big-endian target"
    );
    let Plan { meta, is_demo, budget } = plan;
    let full = full_region(&meta);

    // Eager full-run build (filter=None): seeds the cache and the intensity reference.
    let built = build_load_result(&meta, is_demo, &full, budget, None)?;
    let n0 = built.result.n_points;
    let bytes0 = built.result.points.len();
    let state = Arc::new(State {
        meta,
        is_demo,
        default_budget: budget,
        full_i_hist: built.i_hist,
        full_i_p99: built.i_p99,
        cache: Mutex::new(HashMap::new()),
    });
    state
        .cache
        .lock()
        .unwrap()
        .insert(LoadKey::of(&full, budget), Arc::new(built.result));

    let server = Arc::new(
        Server::http(("127.0.0.1", port)).map_err(|e| anyhow::anyhow!("bind {port}: {e}"))?,
    );
    log::info!(
        "serving {n0} points ({:.1} MB) + on-demand region queries on http://localhost:{port} (localhost only)",
        bytes0 as f64 / 1e6,
    );

    let mut handles = Vec::with_capacity(WORKERS);
    for _ in 0..WORKERS {
        let (server, state) = (server.clone(), state.clone());
        handles.push(std::thread::spawn(move || worker(&server, &state)));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

fn worker(server: &Server, state: &State) {
    while let Ok(req) = server.recv() {
        if let Err(e) = handle(req, state) {
            log::warn!("HTTP respond failed: {e}");
        }
    }
}

fn handle(req: Request, state: &State) -> std::io::Result<()> {
    if *req.method() == Method::Options {
        // CORS preflight (the trunk page is served from a different origin).
        return req.respond(
            Response::empty(204)
                .with_header(cors())
                .with_header(header("Access-Control-Allow-Methods", "GET, OPTIONS"))
                .with_header(header("Access-Control-Allow-Headers", "*")),
        );
    }
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("");
    let (want_points, want_meta) = (path == "/points", path == "/meta");
    if *req.method() == Method::Get && (want_points || want_meta) {
        let (region, budget) = parse_query(&url, state);
        return match get_or_build(state, &region, budget) {
            Ok(lr) if want_points => {
                respond_bytes(req, lr.points.clone(), "application/octet-stream")
            }
            Ok(lr) => respond_bytes(req, lr.meta.clone(), "application/json"),
            Err(e) => req.respond(
                Response::from_string(format!("load failed: {e}"))
                    .with_status_code(500)
                    .with_header(cors()),
            ),
        };
    }
    req.respond(
        Response::from_string(
            "tims-viewer point server\n\
             GET /points|/meta[?n=&mz0=&mz1=&im0=&im1=&rt0=&rt1=&imin=]\n",
        )
        .with_header(cors()),
    )
}

/// Parse the region + budget from a query string, defaulting to the full run. Ranges are clamped to
/// the full bounds and ordered (lo <= hi).
fn parse_query(url: &str, state: &State) -> (Region4D, usize) {
    let full = full_region(&state.meta);
    let mut r = full;
    let mut budget = state.default_budget;
    if let Some(q) = url.split('?').nth(1) {
        for kv in q.split('&') {
            let Some((k, v)) = kv.split_once('=') else { continue };
            match k {
                "n" => {
                    if let Ok(n) = v.parse::<usize>() {
                        budget = n.max(1);
                    }
                }
                "imin" => {
                    if let Ok(x) = v.parse::<f32>() {
                        r.imin = x.max(0.0);
                    }
                }
                _ => {
                    if let Ok(x) = v.parse::<f64>() {
                        match k {
                            "mz0" => r.mz.0 = x,
                            "mz1" => r.mz.1 = x,
                            "im0" => r.im.0 = x,
                            "im1" => r.im.1 = x,
                            "rt0" => r.rt.0 = x,
                            "rt1" => r.rt.1 = x,
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    r.mz = clamp_range(r.mz, full.mz);
    r.im = clamp_range(r.im, full.im);
    r.rt = clamp_range(r.rt, full.rt);
    (r, budget)
}

/// Look up a cached load or build (and cache) it. The build runs outside the lock so concurrent
/// requests for *other* regions are not blocked; a rare double-build of the same region is harmless.
fn get_or_build(state: &State, region: &Region4D, budget: usize) -> Result<Arc<LoadResult>> {
    let key = LoadKey::of(region, budget);
    if let Some(lr) = state.cache.lock().unwrap().get(&key).cloned() {
        return Ok(lr);
    }
    let built = build_load_result(
        &state.meta,
        state.is_demo,
        region,
        budget,
        Some((&state.full_i_hist, state.full_i_p99)),
    )?;
    let lr = Arc::new(built.result);
    state.cache.lock().unwrap().insert(key, lr.clone());
    Ok(lr)
}

/// Run one region load: select frames, size the stride to the region survivor estimate, stream
/// through the loader (normalized to the region so it fills the cube), shuffle, and pack the
/// `/points` bytes + `/meta` JSON.
fn build_load_result(
    meta: &MetaIndex,
    is_demo: bool,
    region: &Region4D,
    budget: usize,
    i_ref: Option<(&[u32], f32)>,
) -> Result<Built> {
    let full = full_region(meta);

    // RT window -> frame ids + the RT-region source size (all m/z·1/K0, pre-cull).
    let mut frame_ids: Vec<u32> = Vec::new();
    let mut rt_total: u64 = 0;
    for f in &meta.frames {
        if f.retention_time >= region.rt.0 && f.retention_time <= region.rt.1 {
            frame_ids.push(f.id);
            rt_total = rt_total.saturating_add(f.num_peaks);
        }
    }

    // Region-relative survivor estimate -> stride (FOCUS_LENS_PLAN.md blocker-1): scale the RT
    // source size by the m/z·1/K0 window fraction and the intensity floor's surviving fraction.
    let span = |r: (f64, f64), f: (f64, f64)| ((r.1 - r.0) / (f.1 - f.0).max(1e-9)).clamp(0.0, 1.0);
    let i_frac = intensity_frac(region.imin, i_ref);
    let estimate =
        (((rt_total as f64) * span(region.mz, full.mz) * span(region.im, full.im) * i_frac).ceil()
            as u64)
            .max(1);

    // Re-normalize: the loader maps points to [-1,1] over the region bounds (so the region fills
    // the view). For the full run this equals the run bounds (unchanged behavior).
    let bounds = AxisBounds {
        mz: AxisTransform::new(region.mz.0, region.mz.1),
        im: AxisTransform::new(region.im.0, region.im.1),
        rt: AxisTransform::new(region.rt.0, region.rt.1),
    };

    // No-op spatial culls (the full run) skip the per-point filter entirely.
    const EPS: f64 = 1e-6;
    let no_spatial = region.mz.0 <= full.mz.0 + EPS
        && region.mz.1 >= full.mz.1 - EPS
        && region.im.0 <= full.im.0 + EPS
        && region.im.1 >= full.im.1 - EPS;
    let filter = if no_spatial && region.imin <= 0.0 {
        None
    } else {
        Some(RegionFilter {
            mz: region.mz,
            im: region.im,
            intensity_min: region.imin,
        })
    };

    let mode = if is_demo {
        LoaderMode::Demo(DemoSource::new(frame_ids.len(), rt_total))
    } else {
        LoaderMode::Real {
            path: meta.data_path.clone(),
            frame_ids,
        }
    };

    let (mut points, stats, hist) = collect(mode, bounds, estimate, budget, filter)?;
    shuffle_points(&mut points);
    let n_points = points.len();
    let stride = stride_for(estimate, budget.max(1));
    let (i_p1, i_p50, i_p99) = stats.unwrap_or_else(|| intensity_percentiles(&points));
    let i_hist = intensity_linear_hist(&points, i_p99);

    let body: Arc<[u8]> =
        Arc::from(bytemuck::cast_slice::<GpuPoint, u8>(&points).to_vec().into_boxed_slice());
    drop(points);

    let fin = |x: f64| if x.is_finite() { x } else { 0.0 };
    let meta_json = serde_json::json!({
        "version": 1,
        "point_stride": std::mem::size_of::<GpuPoint>(),
        "n_points": n_points,
        "downsample_stride": stride,
        // Region bounds in real units — the client re-normalizes axes/crops/strips to these.
        "bounds": {
            "mz": [fin(region.mz.0), fin(region.mz.1)],
            "im": [fin(region.im.0), fin(region.im.1)],
            "rt": [fin(region.rt.0), fin(region.rt.1)],
        },
        "intensity": {
            "p1": fin(i_p1 as f64), "p50": fin(i_p50 as f64), "p99": fin(i_p99 as f64),
            "hist": i_hist.clone(),
        },
        // Sample-based per-axis density histograms (over the kept systematic sample + peak tail).
        "hist": hist.as_ref().map(|h| serde_json::json!({
            "mz": h.mz, "im": h.im, "rt": h.rt,
        })),
    })
    .to_string();

    Ok(Built {
        result: LoadResult {
            points: body,
            meta: Arc::from(meta_json.into_bytes().into_boxed_slice()),
            n_points,
        },
        i_hist,
        i_p99,
    })
}

/// Surviving fraction of points at or above an intensity floor, estimated from the full-run linear
/// intensity histogram over `[0, p99]`. `imin <= 0` (or no reference) => everything survives.
fn intensity_frac(imin: f32, i_ref: Option<(&[u32], f32)>) -> f64 {
    if imin <= 0.0 {
        return 1.0;
    }
    let Some((hist, p99)) = i_ref else { return 1.0 };
    let total: u64 = hist.iter().map(|&c| c as u64).sum();
    if total == 0 || hist.is_empty() {
        return 1.0;
    }
    let bin = ((imin / p99.max(1.0)).clamp(0.0, 1.0) * (hist.len() as f32 - 1.0)) as usize;
    let above: u64 = hist[bin..].iter().map(|&c| c as u64).sum();
    // Never 0: a tiny estimate yields stride 1 and we keep all (few) survivors anyway.
    (above as f64 / total as f64).max(1.0 / total as f64)
}

/// Serve a shared byte buffer without copying it per request (Cursor over the `Arc`).
fn respond_bytes(req: Request, body: Arc<[u8]>, content_type: &str) -> std::io::Result<()> {
    let len = body.len();
    let resp = Response::new(
        StatusCode(200),
        vec![cors(), header("Content-Type", content_type)],
        Cursor::new(body),
        Some(len),
        None,
    );
    req.respond(resp)
}

/// Per-axis density histograms from the loader (each `HIST_BINS` bins over the region range).
struct Hist {
    mz: Vec<u32>,
    im: Vec<u32>,
    rt: Vec<u32>,
}

/// Drain the loader (with `total_estimate` already sized to the region) into a bounded point
/// buffer, capturing the intensity `Stats` and per-axis histograms.
fn collect(
    mode: LoaderMode,
    bounds: AxisBounds,
    estimate: u64,
    budget: usize,
    filter: Option<RegionFilter>,
) -> Result<(Vec<GpuPoint>, Option<(f32, f32, f32)>, Option<Hist>)> {
    let capacity = budget.max(1);
    let loader = LoaderHandle::spawn(mode, bounds, estimate, capacity, filter);

    let mut points: Vec<GpuPoint> = Vec::with_capacity(capacity.min(1 << 20));
    let mut stats: Option<(f32, f32, f32)> = None;
    let mut hist: Option<Hist> = None;
    let mut done = false;
    loop {
        match loader.rx.recv() {
            Ok(LoadMsg::Chunk { points: pts, .. }) | Ok(LoadMsg::PeakChunk { points: pts }) => {
                let room = capacity.saturating_sub(points.len());
                if room > 0 {
                    points.extend(pts.into_iter().take(room));
                }
            }
            Ok(LoadMsg::Stats { i_min, i_max, i_med }) => stats = Some((i_min, i_med, i_max)),
            Ok(LoadMsg::Histograms { mz, im, rt, .. }) => hist = Some(Hist { mz, im, rt }),
            Ok(LoadMsg::Done { .. }) => {
                done = true;
                break;
            }
            Ok(LoadMsg::Error(e)) => anyhow::bail!("loader error: {e}"),
            Ok(_) => {}
            Err(_) => break,
        }
    }
    anyhow::ensure!(done, "loader thread ended before completing the load");
    Ok((points, stats, hist))
}

fn full_region(meta: &MetaIndex) -> Region4D {
    let b = &meta.bounds;
    Region4D {
        mz: (b.mz.min, b.mz.max),
        im: (b.im.min, b.im.max),
        rt: (b.rt.min, b.rt.max),
        imin: 0.0,
    }
}

/// Clamp a range into `full` and order it (lo <= hi).
fn clamp_range(r: (f64, f64), full: (f64, f64)) -> (f64, f64) {
    let lo = r.0.clamp(full.0, full.1);
    let hi = r.1.clamp(full.0, full.1);
    if lo <= hi {
        (lo, hi)
    } else {
        (hi, lo)
    }
}

/// Intensity (p1, p50, p99), nearest-rank, fallback when the loader Stats are absent.
fn intensity_percentiles(points: &[GpuPoint]) -> (f32, f32, f32) {
    let mut v: Vec<f32> = points.iter().map(|p| p.intensity).filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return (1.0, 1.0, 1.0);
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let pct = |q: f32| {
        let idx = ((v.len() as f32 * q).ceil() as usize).saturating_sub(1).min(v.len() - 1);
        v[idx].max(1.0)
    };
    (pct(0.01), pct(0.50), pct(0.99))
}

/// In-place Fisher-Yates shuffle with a seeded xorshift64 (deterministic, no `rand` dependency).
/// Decorrelates the loader's RT-ordered emission so a prefix of the array is a uniform subsample.
fn shuffle_points(points: &mut [GpuPoint]) {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for i in (1..points.len()).rev() {
        points.swap(i, (next() % (i as u64 + 1)) as usize);
    }
}

/// Linear intensity histogram over `[0, hi]` (real counts); values above `hi` clamp to the last bin.
fn intensity_linear_hist(points: &[GpuPoint], hi: f32) -> Vec<u32> {
    let hi = hi.max(1.0);
    let mut h = vec![0u32; I_HIST_BINS];
    for p in points {
        let i = p.intensity;
        let bin = if i.is_finite() && i > 0.0 {
            ((i / hi).clamp(0.0, 1.0) * (I_HIST_BINS as f32 - 1.0)) as usize
        } else {
            0
        };
        h[bin] += 1;
    }
    h
}

fn cors() -> Header {
    header("Access-Control-Allow-Origin", "*")
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("valid header")
}
