//! Point-streaming server (NATIVE_WEB.md Phase 2).
//!
//! Drains the streaming loader into a `Vec<GpuPoint>` (no GPU needed — the loader is pure CPU
//! and packs points already normalized to the `[-1, 1]` cube) and serves the raw bytes over
//! HTTP so the browser shell (`tims-viewer-web`) can `fetch()` and render them. A `.d` path or
//! `DEMO` both work, so this is testable without Bruker data.
//!
//! Wire format (v1, deliberately minimal): `GET /points` returns the raw little-endian
//! `GpuPoint` array (`application/octet-stream`); the client reinterprets it as 32-byte points.
//! `GET /meta` returns small JSON (point count + cube bounds, for axis labels later). Bound to
//! localhost only. Phase 2.x will frame this as a proper message protocol (stats/histograms/
//! annotations, version/stride negotiation, websocket) — see NATIVE_WEB.md.

use std::io::Cursor;
use std::sync::Arc;

use anyhow::Result;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::app::Plan;
use crate::data::demo::DemoSource;
use crate::data::loader::{LoadMsg, LoaderHandle, LoaderMode};
use crate::data::point::GpuPoint;

/// Collect the whole slice as packed points, then serve it on `127.0.0.1:port` until killed.
pub fn serve(plan: Plan, port: u16) -> Result<()> {
    anyhow::ensure!(
        cfg!(target_endian = "little"),
        "the point wire format is little-endian; --serve is unsupported on this big-endian target"
    );

    let bounds = plan.meta.bounds;
    let total = plan.meta.total_points_estimate;
    let capacity = plan.budget.max(1).min((total as usize).max(1));
    let stride = crate::data::loader::stride_for(total, capacity) as u64;
    let (points, stats) = collect_points(plan)?;
    let n_points = points.len();
    // Prefer the loader's systematic-base percentiles (matches the native viewer's Stats); fall
    // back to computing over the served points (incl. the peak tail) only if Stats never arrived.
    let (i_p1, i_p50, i_p99) = stats.unwrap_or_else(|| intensity_percentiles(&points));
    // One copy into an owned byte buffer, then share it (no per-request copy) via Arc + Cursor.
    let body: Arc<[u8]> =
        Arc::from(bytemuck::cast_slice::<GpuPoint, u8>(&points).to_vec().into_boxed_slice());
    drop(points); // keep only the byte body

    // Real-unit + exposure context for the client (axis labels, auto-transfer). serde_json emits
    // non-finite floats as null, so guard every value to keep the JSON valid.
    let fin = |x: f64| if x.is_finite() { x } else { 0.0 };
    let meta = serde_json::json!({
        "version": 1,
        "point_stride": std::mem::size_of::<GpuPoint>(),
        "n_points": n_points,
        "downsample_stride": stride,
        "bounds": {
            "mz": [fin(bounds.mz.min), fin(bounds.mz.max)],
            "im": [fin(bounds.im.min), fin(bounds.im.max)],
            "rt": [fin(bounds.rt.min), fin(bounds.rt.max)],
        },
        "intensity": {
            "p1": fin(i_p1 as f64), "p50": fin(i_p50 as f64), "p99": fin(i_p99 as f64),
        },
    })
    .to_string();
    let meta: Arc<[u8]> = Arc::from(meta.into_bytes().into_boxed_slice());

    let server = Server::http(("127.0.0.1", port)).map_err(|e| anyhow::anyhow!("bind {port}: {e}"))?;
    log::info!(
        "serving {n_points} points ({:.1} MB) on http://localhost:{port}/points (localhost only)",
        body.len() as f64 / 1e6,
    );

    for req in server.incoming_requests() {
        let is_get = *req.method() == Method::Get;
        // Route by path only — ignore any `?query` (owned so `req` can move into the responder).
        let path = req.url().split('?').next().unwrap_or("").to_string();
        let result = if path == "/points" && is_get {
            respond_bytes(req, body.clone(), "application/octet-stream")
        } else if path == "/meta" && is_get {
            respond_bytes(req, meta.clone(), "application/json")
        } else if *req.method() == Method::Options {
            // CORS preflight (the trunk page is served from a different port).
            req.respond(
                Response::empty(204)
                    .with_header(cors())
                    .with_header(header("Access-Control-Allow-Methods", "GET, OPTIONS"))
                    .with_header(header("Access-Control-Allow-Headers", "*")),
            )
        } else {
            req.respond(
                Response::from_string(
                    "tims-viewer point server\nGET /points -> GpuPoint bytes\nGET /meta -> JSON\n",
                )
                .with_header(cors()),
            )
        };
        if let Err(e) = result {
            log::warn!("HTTP respond failed: {e}");
        }
    }
    Ok(())
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

/// Drain the loader into a point buffer; also capture the loader's intensity `Stats`
/// (systematic-base p1/p50/p99 as `(i_min, i_med, i_max)`) for the client's auto-transfer.
fn collect_points(plan: Plan) -> Result<(Vec<GpuPoint>, Option<(f32, f32, f32)>)> {
    let total = plan.meta.total_points_estimate;
    let bounds = plan.meta.bounds;
    let capacity = plan.budget.max(1).min((total as usize).max(1));

    let mode = if plan.is_demo {
        LoaderMode::Demo(DemoSource::new(plan.meta.frames.len(), total))
    } else {
        LoaderMode::Real {
            path: plan.meta.data_path.clone(),
            frame_ids: plan.meta.frames.iter().map(|f| f.id).collect(),
            filter: None,
        }
    };
    let loader = LoaderHandle::spawn(mode, bounds, total, capacity);

    let mut points: Vec<GpuPoint> = Vec::with_capacity(capacity);
    let mut stats: Option<(f32, f32, f32)> = None;
    let mut done = false;
    loop {
        match loader.rx.recv() {
            // Both carry `points`; bound the buffer at `capacity` but keep draining so the loader
            // thread can finish (and signal Done) instead of blocking on a full channel.
            Ok(LoadMsg::Chunk { points: pts, .. }) | Ok(LoadMsg::PeakChunk { points: pts }) => {
                let room = capacity.saturating_sub(points.len());
                if room > 0 {
                    points.extend(pts.into_iter().take(room));
                }
            }
            Ok(LoadMsg::Stats { i_min, i_max, i_med }) => stats = Some((i_min, i_med, i_max)),
            Ok(LoadMsg::Done { .. }) => {
                done = true;
                break;
            }
            Ok(LoadMsg::Error(e)) => anyhow::bail!("loader error: {e}"),
            Ok(_) => {} // Histograms/Annotations/Progress: not part of the v1 point stream
            Err(_) => break,
        }
    }
    anyhow::ensure!(done, "loader thread ended before completing the load");
    Ok((points, stats))
}

/// Intensity (p1, p50, p99), nearest-rank, for the client's auto-transfer/exposure (mirrors the
/// native viewer). Non-finite values are dropped.
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

fn cors() -> Header {
    header("Access-Control-Allow-Origin", "*")
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("valid header")
}
