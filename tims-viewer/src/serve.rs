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

use std::collections::{HashMap, VecDeque};
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::app::Plan;
use crate::data::demo::DemoSource;
use crate::data::loader::{stride_for, LoadMsg, LoaderHandle, LoaderMode, RegionFilter, PROJ_BINS};
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

/// Max distinct (region, budget) loads kept resident. Each can be hundreds of MB, so bound it; the
/// pinned full-run base is never evicted (instant Reset).
const CACHE_MAX: usize = 8;

/// Bounded FIFO cache of built loads. The pinned key (full run) is never evicted.
struct Cache {
    map: HashMap<LoadKey, Arc<LoadResult>>,
    order: VecDeque<LoadKey>,
    pinned: LoadKey,
}

impl Cache {
    fn get(&self, k: &LoadKey) -> Option<Arc<LoadResult>> {
        self.map.get(k).cloned()
    }
    fn insert(&mut self, k: LoadKey, v: Arc<LoadResult>) {
        if self.map.insert(k, v).is_none() {
            self.order.push_back(k);
        }
        while self.map.len() > CACHE_MAX {
            // Evict the oldest non-pinned entry; drop stale/pinned keys from the order queue.
            let mut evicted = false;
            while let Some(old) = self.order.pop_front() {
                if old != self.pinned && self.map.remove(&old).is_some() {
                    evicted = true;
                    break;
                }
            }
            if !evicted {
                break;
            }
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
    /// Precomputed `/windows` JSON: the run's DIA isolation-window footprints in real units (static;
    /// the web re-normalizes them to whatever region it has focused). Empty for demo / non-DIA runs.
    windows_json: Arc<[u8]>,
    cache: Mutex<Cache>,
}

/// What the server can offer: a single synthetic dataset, or a set of discovered `.d` paths
/// (confined to the launch root) the frontend can choose between via `/datasets` + `?dataset=N`.
pub enum ServeSource {
    Demo(MetaIndex),
    Datasets(Vec<PathBuf>),
}

/// One selectable dataset in the registry (the paths themselves live in `Hub::source`).
struct DatasetEntry {
    name: String,
}

/// Max datasets kept resident at once — each holds a full-run load (hundreds of MB) plus its region
/// cache, so switching evicts the least-recently-built (it rebuilds in seconds if revisited).
const MAX_RESIDENT_DATASETS: usize = 2;

/// LRU of built per-dataset `State`s, keyed by dataset id.
#[derive(Default)]
struct StateCache {
    map: HashMap<usize, Arc<State>>,
    order: VecDeque<usize>,
}

impl StateCache {
    fn get(&mut self, id: usize) -> Option<Arc<State>> {
        let s = self.map.get(&id).cloned();
        if s.is_some() {
            self.order.retain(|&x| x != id);
            self.order.push_back(id); // mark most-recently-used
        }
        s
    }
    fn insert(&mut self, id: usize, s: Arc<State>) {
        if self.map.insert(id, s).is_none() {
            self.order.push_back(id);
        }
        while self.map.len() > MAX_RESIDENT_DATASETS {
            match self.order.pop_front() {
                Some(old) if old != id => {
                    self.map.remove(&old);
                }
                Some(_) | None => break,
            }
        }
    }
}

/// The multi-dataset server: a registry + a lazily-built, LRU-bounded set of per-dataset states.
struct Hub {
    source: ServeSource,
    entries: Vec<DatasetEntry>,
    budget: Option<usize>,
    /// zstd-compress the /points payload when the client advertises Accept-Encoding: zstd.
    compress: bool,
    states: Mutex<StateCache>,
    /// Per-dataset build locks: a dataset switch fires `/meta` + `/points` + `/windows` near-
    /// simultaneously, so serialize concurrent builds of the *same* id (the rest re-check the cache
    /// and hit it) — without serializing builds of *different* datasets.
    build_locks: Mutex<HashMap<usize, Arc<Mutex<()>>>>,
    datasets_json: Arc<[u8]>,
}

impl Hub {
    /// Build the load Plan for dataset `id` (loads its metadata; cheap relative to the full build).
    fn plan_for(&self, id: usize) -> Result<Plan> {
        match &self.source {
            ServeSource::Demo(meta) => Ok(Plan::new(meta.clone(), true, self.budget)),
            ServeSource::Datasets(paths) => {
                let p = paths
                    .get(id)
                    .ok_or_else(|| anyhow::anyhow!("dataset {id} out of range"))?;
                let meta = MetaIndex::load(&p.display().to_string())?;
                Ok(Plan::new(meta, false, self.budget))
            }
        }
    }
}

/// Resolve the `State` for dataset `id`, building (and caching, LRU-evicting) it on first request.
/// The build runs outside the lock so a multi-second load can't block other datasets' requests.
fn get_or_build_state(hub: &Hub, id: usize) -> Result<Arc<State>> {
    if let Some(s) = hub.states.lock().unwrap().get(id) {
        return Ok(s);
    }
    // Single-flight per dataset: hold this id's build lock so concurrent first-touches (a switch's
    // /meta + /points + /windows) don't each build the same hundreds-of-MB state.
    let build_lock = hub
        .build_locks
        .lock()
        .unwrap()
        .entry(id)
        .or_default()
        .clone();
    let _guard = build_lock.lock().unwrap();
    // Re-check: another thread may have built it while we waited on the build lock.
    if let Some(s) = hub.states.lock().unwrap().get(id) {
        return Ok(s);
    }
    let state = Arc::new(build_state(hub.plan_for(id)?)?);
    hub.states.lock().unwrap().insert(id, state.clone());
    Ok(state)
}

/// Eagerly build one dataset's `State`: the full-run load (first paint + intensity reference), the
/// DIA window footprints, and a fresh region cache pinned on the full run.
fn build_state(plan: Plan) -> Result<State> {
    let Plan { meta, is_demo, budget } = plan;
    // Snap the eager full region onto the cache grid so a client's explicit full-bounds query
    // (also snapped in parse_query) resolves to this same cached entry instead of rebuilding.
    let full = snap_region(full_region(&meta));
    let built = build_load_result(&meta, is_demo, &full, budget, None)?;
    log::info!(
        "built dataset '{}': {} points ({:.1} MB)",
        meta.data_path,
        built.result.n_points,
        built.result.points.len() as f64 / 1e6,
    );
    let full_key = LoadKey::of(&full, budget);
    let mut map = HashMap::new();
    map.insert(full_key, Arc::new(built.result));
    let windows_json = build_windows_json(is_demo, &meta.data_path);
    Ok(State {
        meta,
        is_demo,
        default_budget: budget,
        full_i_hist: built.i_hist,
        full_i_p99: built.i_p99,
        windows_json,
        cache: Mutex::new(Cache {
            map,
            order: VecDeque::new(),
            pinned: full_key,
        }),
    })
}

/// Display name for a dataset path: the `.d` folder's stem (e.g. `/data/run5.d` → `run5`).
fn dataset_name(path: &std::path::Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| path.display().to_string())
}

/// `/datasets` JSON: `[{"id":0,"name":"run5"}, ...]`. Built via serde_json so any folder name
/// (control chars, quotes, unicode) is escaped correctly.
fn build_datasets_json(entries: &[DatasetEntry]) -> Arc<[u8]> {
    let items: Vec<serde_json::Value> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| serde_json::json!({ "id": i, "name": e.name }))
        .collect();
    serde_json::to_vec(&items).unwrap_or_else(|_| b"[]".to_vec()).into()
}

/// Build the registry, eagerly build the default dataset (fast first paint), then serve region
/// queries — and dataset switches — on demand from `WORKERS` threads.
pub fn serve(source: ServeSource, budget: Option<usize>, port: u16, compress: bool) -> Result<()> {
    anyhow::ensure!(
        cfg!(target_endian = "little"),
        "the point wire format is little-endian; --serve is unsupported on this big-endian target"
    );
    let entries: Vec<DatasetEntry> = match &source {
        ServeSource::Demo(_) => vec![DatasetEntry { name: "DEMO".into() }],
        ServeSource::Datasets(paths) => {
            anyhow::ensure!(!paths.is_empty(), "no datasets to serve");
            paths.iter().map(|p| DatasetEntry { name: dataset_name(p) }).collect()
        }
    };
    let datasets_json = build_datasets_json(&entries);
    let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    log::info!("registry: {} dataset(s): {}", entries.len(), names.join(", "));

    let hub = Arc::new(Hub {
        source,
        entries,
        budget,
        compress,
        states: Mutex::new(StateCache::default()),
        build_locks: Mutex::new(HashMap::new()),
        datasets_json,
    });
    if compress {
        log::info!("over-the-wire zstd compression enabled for /points (Accept-Encoding negotiated)");
    }

    // Eagerly build the default dataset so its first paint is instant.
    get_or_build_state(&hub, 0)?;

    let server = Arc::new(
        Server::http(("127.0.0.1", port)).map_err(|e| anyhow::anyhow!("bind {port}: {e}"))?,
    );
    log::info!(
        "serving default '{}' + on-demand region/dataset queries on http://localhost:{port} (localhost only)",
        hub.entries[0].name,
    );

    let mut handles = Vec::with_capacity(WORKERS);
    for _ in 0..WORKERS {
        let (server, hub) = (server.clone(), hub.clone());
        handles.push(std::thread::spawn(move || worker(&server, &hub)));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

fn worker(server: &Server, hub: &Hub) {
    while let Ok(req) = server.recv() {
        if let Err(e) = handle(req, hub) {
            log::warn!("HTTP respond failed: {e}");
        }
    }
}

/// Dataset id from `?dataset=N` (default 0), clamped to the registry size.
fn parse_dataset_id(url: &str, n_datasets: usize) -> usize {
    let q = url.split('?').nth(1).unwrap_or("");
    q.split('&')
        .find_map(|kv| kv.strip_prefix("dataset=").and_then(|v| v.parse::<usize>().ok()))
        .unwrap_or(0)
        .min(n_datasets.saturating_sub(1))
}

fn handle(req: Request, hub: &Hub) -> std::io::Result<()> {
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
    if *req.method() == Method::Get && path == "/datasets" {
        // The selectable-dataset registry (the frontend renders this as a dropdown).
        return respond_bytes(req, hub.datasets_json.clone(), "application/json");
    }
    let want_windows = path == "/windows";
    let (want_points, want_meta) = (path == "/points", path == "/meta");
    if *req.method() == Method::Get && (want_windows || want_points || want_meta) {
        // Resolve (and lazily build) the selected dataset's state.
        let id = parse_dataset_id(&url, hub.entries.len());
        let state = match get_or_build_state(hub, id) {
            Ok(s) => s,
            Err(e) => {
                return req.respond(
                    Response::from_string(format!("dataset load failed: {e}"))
                        .with_status_code(StatusCode(500))
                        .with_header(cors()),
                );
            }
        };
        if want_windows {
            // Run-level DIA isolation-window footprints (static JSON; region-independent).
            return respond_bytes(req, state.windows_json.clone(), "application/json");
        }
        let (region, budget) = parse_query(&url, &state);
        return match get_or_build(&state, &region, budget) {
            Ok(lr) if want_points => {
                respond_points(req, lr.points.clone(), hub.compress)
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
                        budget = n;
                    }
                }
                "imin" => {
                    if let Ok(x) = v.parse::<f32>() {
                        if x.is_finite() {
                            r.imin = x.max(0.0);
                        }
                    }
                }
                _ => {
                    // Ignore non-finite (NaN/inf) coordinates rather than letting them reach the
                    // cube transform / cache key / normalization.
                    if let Some(x) = v.parse::<f64>().ok().filter(|x| x.is_finite()) {
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
    // The demo loader generates frames 0..N starting at RT 0 and ignores frame ids, so it can't
    // honor an RT sub-window — use the full RT range for demo sessions (m/z·1/K0·intensity focus
    // still works). Real data realizes RT via frame selection.
    if state.is_demo {
        r.rt = full.rt;
    }
    r.mz = clamp_range(r.mz, full.mz);
    r.im = clamp_range(r.im, full.im);
    r.rt = clamp_range(r.rt, full.rt);
    // Snap to the cache-key grid so the reported bounds exactly match the key (no aliasing where
    // two distinct requests round to the same key but expect different results).
    r = snap_region(r);
    // Cap the budget to the server pool (also guards the downstream `budget * 85` from overflow).
    (r, budget.clamp(1, state.default_budget))
}

/// Round a region onto the same grid the cache key quantizes to, so key and bounds stay consistent.
fn snap_region(mut r: Region4D) -> Region4D {
    let snap = |x: f64, s: f64| (x * s).round() / s;
    r.mz = (snap(r.mz.0, 1e3), snap(r.mz.1, 1e3));
    r.im = (snap(r.im.0, 1e6), snap(r.im.1, 1e6));
    r.rt = (snap(r.rt.0, 1e3), snap(r.rt.1, 1e3));
    r.imin = snap(r.imin as f64, 1e3) as f32;
    r
}

/// Look up a cached load or build (and cache) it. The build runs outside the lock so concurrent
/// requests for *other* regions are not blocked; a rare double-build of the same region is harmless.
fn get_or_build(state: &State, region: &Region4D, budget: usize) -> Result<Arc<LoadResult>> {
    let key = LoadKey::of(region, budget);
    // Poison-safe: a panicked worker must not wedge the others.
    let lock = || state.cache.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(lr) = lock().get(&key) {
        return Ok(lr);
    }
    // Build outside the lock (a multi-second load mustn't block other regions). Builds are
    // deterministic, so a rare concurrent double-build of the same key yields identical bytes —
    // `/meta` and `/points` stay coherent regardless of which build each request observes.
    let built = build_load_result(
        &state.meta,
        state.is_demo,
        region,
        budget,
        Some((&state.full_i_hist, state.full_i_p99)),
    )?;
    let lr = Arc::new(built.result);
    lock().insert(key, lr.clone());
    Ok(lr)
}

/// Run one region load: select frames, size the stride to the region survivor estimate, stream
/// through the loader (normalized to the region so it fills the cube), shuffle, and pack the
/// `/points` bytes + `/meta` JSON.
/// Build the static `/windows` JSON: `{ max_window_group, windows: [[group, mz0, mz1, im0, im1], …] }`
/// in real units. Demo or non-DIA runs yield an empty set.
fn build_windows_json(is_demo: bool, data_path: &str) -> Arc<[u8]> {
    let json = if is_demo {
        serde_json::json!({ "max_window_group": 0, "windows": [] })
    } else {
        let ds = rustdf::data::dataset::TimsDataset::new("NO_SDK", data_path, false, false);
        let (rects, max_group) = crate::data::loader::dia_window_rects(&ds, data_path);
        let windows: Vec<[f64; 5]> = rects
            .iter()
            .map(|r| [r.group as f64, r.mz0, r.mz1, r.im0, r.im1])
            .collect();
        serde_json::json!({ "max_window_group": max_group, "windows": windows })
    };
    Arc::from(json.to_string().into_bytes().into_boxed_slice())
}

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

    // A region with no frames in its RT window is valid — return an empty result (the loader would
    // otherwise error on a zero-frame run). Frames-but-zero-survivors flows through normally below.
    if frame_ids.is_empty() {
        return Ok(empty_result(region));
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

    // Cycle duration: the run-level mean RT gap between consecutive MS1 (precursor) frames. The web
    // uses it to cluster RT in cycle units, so precursor points across cycles stay density-connected
    // regardless of how tightly the RT range is focused. 0 when unknown (demo / no MS1 frames).
    let ms1_rt: Vec<f64> = meta.frames.iter().filter(|f| !f.is_ms2).map(|f| f.retention_time).collect();
    let cycle_duration = if ms1_rt.len() >= 2 {
        (ms1_rt[ms1_rt.len() - 1] - ms1_rt[0]) / (ms1_rt.len() - 1) as f64
    } else {
        0.0
    };

    let fin = |x: f64| if x.is_finite() { x } else { 0.0 };
    // Run-level 1/K0 per TIMS scan: the FULL-run mobility span over the ramp length. Run-level (not
    // region-scoped) so the client's IM clustering reach is focus-independent.
    let im_per_scan = if meta.num_scans > 1 {
        (meta.bounds.im.max - meta.bounds.im.min).abs() / (meta.num_scans as f64 - 1.0)
    } else {
        0.0
    };
    let meta_json = serde_json::json!({
        "version": 1,
        "point_stride": std::mem::size_of::<GpuPoint>(),
        "n_points": n_points,
        "downsample_stride": stride,
        "cycle_duration": fin(cycle_duration),
        "im_per_scan_1k0": fin(im_per_scan),
        "num_scans": meta.num_scans,
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
        // Sample-based 2D density projections for the box-select minimaps.
        "proj": hist.as_ref().map(|h| serde_json::json!({
            "bins": PROJ_BINS,
            "mz_im": h.proj_mz_im, "mz_rt": h.proj_mz_rt, "im_rt": h.proj_im_rt,
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

/// Does the request actually accept `zstd`? Token-correct: matches the `zstd` coding exactly (not a
/// substring like `x-zstd`) across a `gzip, zstd, br` list and rejects an explicit `q=0`.
fn accepts_zstd(req: &Request) -> bool {
    req.headers().iter().filter(|h| h.field.equiv("Accept-Encoding")).any(|h| {
        h.value.as_str().split(',').any(|tok| {
            let mut parts = tok.split(';').map(str::trim);
            let coding = parts.next().unwrap_or("");
            coding.eq_ignore_ascii_case("zstd")
                // accept unless an explicit q=0 disqualifies it
                && parts.all(|p| {
                    p.strip_prefix("q=")
                        .or_else(|| p.strip_prefix("Q="))
                        .and_then(|v| v.parse::<f64>().ok())
                        .map_or(true, |q| q > 0.0)
                })
        })
    })
}

/// Serve the `/points` payload, zstd-compressing it when `--compress` is on AND the client advertises
/// `Accept-Encoding: zstd` (the browser then decompresses `Content-Encoding: zstd` transparently — no
/// client code). When compression is enabled the response carries `Vary: Accept-Encoding` so a shared
/// cache/proxy can't hand a zstd body to a client that didn't ask. Falls back to raw on encode error.
fn respond_points(req: Request, body: Arc<[u8]>, compress: bool) -> std::io::Result<()> {
    if !compress {
        return respond_bytes(req, body, "application/octet-stream");
    }
    if accepts_zstd(&req) {
        match zstd::encode_all(&body[..], 3) {
            Ok(z) => {
                let len = z.len();
                return req.respond(Response::new(
                    StatusCode(200),
                    vec![
                        cors(),
                        header("Content-Type", "application/octet-stream"),
                        header("Content-Encoding", "zstd"),
                        header("Vary", "Accept-Encoding"),
                    ],
                    Cursor::new(z),
                    Some(len),
                    None,
                ));
            }
            Err(e) => log::warn!("zstd encode failed ({e}); sending raw points"),
        }
    }
    // Raw, but still Vary — this route negotiates, so caches must key on Accept-Encoding.
    let len = body.len();
    req.respond(Response::new(
        StatusCode(200),
        vec![cors(), header("Content-Type", "application/octet-stream"), header("Vary", "Accept-Encoding")],
        Cursor::new(body),
        Some(len),
        None,
    ))
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

/// Per-axis density histograms (each `HIST_BINS` bins over the region range) plus the three 2D
/// density projections (`PROJ_BINS²`, row-major `x + bins*y`) for the box-select minimaps.
struct Hist {
    mz: Vec<u32>,
    im: Vec<u32>,
    rt: Vec<u32>,
    proj_mz_im: Vec<u32>,
    proj_mz_rt: Vec<u32>,
    proj_im_rt: Vec<u32>,
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
    // Reservoir-sample the emitted stream to `capacity` rather than truncating its prefix: if the
    // region estimate under-counts and the loader over-emits, truncation would keep an RT-ordered
    // prefix (bias). A reservoir keeps a uniform sample regardless of how wrong the estimate was.
    let mut seen: u64 = 0;
    let mut rng: u64 = 0xD1B5_4A32_D192_ED03;
    loop {
        match loader.rx.recv() {
            Ok(LoadMsg::Chunk { points: pts, .. }) | Ok(LoadMsg::PeakChunk { points: pts }) => {
                for p in pts {
                    seen += 1;
                    if points.len() < capacity {
                        points.push(p);
                    } else {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        let j = (rng % seen) as usize;
                        if j < capacity {
                            points[j] = p;
                        }
                    }
                }
            }
            Ok(LoadMsg::Stats { i_min, i_max, i_med }) => stats = Some((i_min, i_med, i_max)),
            Ok(LoadMsg::Histograms {
                mz, im, rt, proj_mz_im, proj_mz_rt, proj_im_rt, ..
            }) => {
                hist = Some(Hist { mz, im, rt, proj_mz_im, proj_mz_rt, proj_im_rt });
            }
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

/// A valid empty load for a region with no points (e.g. an RT window selecting no frames).
fn empty_result(region: &Region4D) -> Built {
    let fin = |x: f64| if x.is_finite() { x } else { 0.0 };
    let meta_json = serde_json::json!({
        "version": 1,
        "point_stride": std::mem::size_of::<GpuPoint>(),
        "n_points": 0,
        "downsample_stride": 1,
        "bounds": {
            "mz": [fin(region.mz.0), fin(region.mz.1)],
            "im": [fin(region.im.0), fin(region.im.1)],
            "rt": [fin(region.rt.0), fin(region.rt.1)],
        },
        "intensity": { "p1": 1.0, "p50": 1.0, "p99": 1.0, "hist": vec![0u32; I_HIST_BINS] },
        "hist": serde_json::Value::Null,
        "proj": serde_json::Value::Null,
    })
    .to_string();
    Built {
        result: LoadResult {
            points: Arc::from(Vec::<u8>::new().into_boxed_slice()),
            meta: Arc::from(meta_json.into_bytes().into_boxed_slice()),
            n_points: 0,
        },
        i_hist: vec![0u32; I_HIST_BINS],
        i_p99: 1.0,
    }
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
