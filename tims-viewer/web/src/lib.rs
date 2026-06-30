//! Browser shell for the tims-viewer GPU point-cloud renderer (NATIVE_WEB.md Phase 1).
//!
//! Compiles the render-safe half of `tims-viewer` to WASM and drives it on a `<canvas>` via
//! WebGPU (WebGL2 fallback). For now it renders a synthetic demo cloud with an auto-orbiting
//! camera — the proof that the native renderer runs in a browser. Server-streamed `GpuPoint`s
//! (Phase 2) will replace `demo_cloud()`.
//!
//! Everything is gated to `wasm32`: on a native workspace build this file is empty.
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;

use bytemuck::Zeroable;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use tims_viewer::camera::{OrbitCamera, ROLL_UP_AXIS};
use tims_viewer::data::point::{AxisTransform, GpuPoint};
use tims_viewer::render::annotation::{AnnotationRenderer, LineVertex};
use tims_viewer::ticks::{fmt_tick, ticks_for, Axis, RT_MINUTES_SPAN};
use tims_viewer::render::colormap::{sample as colormap_sample, COLORMAP_NAMES};
use tims_viewer::cluster::{cluster_axis_scales, dbscan, dbscan_with_progress, AxisScales, ScaleInputs};
use tims_viewer::render::point_cloud::{PointCloudRenderer, PointMode};
use tims_viewer::render::uniforms::{ParamsUniform, VolumeUniform};
use tims_viewer::render::volume::{VolumeGrid, VolumeRenderer, VOLUME_DIMS};

/// Max points DBSCAN runs on (the in-wasm worker, or the Python service). Beyond this, Focus a region
/// first. Large runs go to the off-main-thread worker with a progress bar.
const CLUSTER_CAP: usize = 6_000_000;

/// Below this many filtered points, DBSCAN runs on the main thread (instant, no worker spin-up);
/// at/above it, the off-main-thread worker is used so the tab doesn't freeze.
const WORKER_THRESHOLD: usize = 300_000;

/// Default for the RT-cycles lever (`Gfx::cluster_rt_cycles`): how many MS1 cycles `eps` bridges
/// along RT. Precursor RT is quantized at the cycle period, so RT is scaled to span this many cycles
/// regardless of the RT zoom.
const CLUSTER_RT_CYCLES: f64 = 2.0;

/// Default for the 1/K0-scans lever (`Gfx::cluster_im_scans`): how many TIMS scans `eps` bridges
/// along mobility. ~2 scans matches the old MIDIA scan-index reach (`1.7·2^0.4 ≈ 2.2`).
const CLUSTER_IM_SCANS: f64 = 2.0;

/// Reference eps the RT/m/z axis scalings are calibrated to (the slider default). The scalings use
/// this fixed value — not the live eps — so the live eps still scales the reach on every axis
/// proportionally (otherwise eps cancels out on the transformed axes).
const CLUSTER_EPS_REF: f64 = 0.012;

/// Nominal TOF resolution for the m/z peak-width transform (peak width ≈ m/z / resolution).
const MZ_RESOLUTION: f64 = 50_000.0;
/// Default for the m/z-peak-width lever (`Gfx::cluster_mz_peak_widths`): how many m/z peak-widths
/// `eps` may span. The m/z axis is expanded so `eps` never bridges more than this, keeping adjacent
/// isotopes (tens of peak-widths apart) as separate clusters.
const CLUSTER_MZ_PEAK_WIDTHS: f64 = 1.5;

/// Hard ceiling on points loaded into the (32-bit) wasm heap, regardless of the server `--budget` or
/// the GPU buffer limit: ~32M × 32 B ≈ 1 GB resident, well clear of the ~4 GB wasm address space.
const MAX_WEB_POINTS: usize = 32_000_000;

/// RT slices each DIA window footprint is drawn at (mirrors the native loader's `DIA_WINDOW_RT_SLICES`,
/// which lives in the native-only loader module).
const DIA_WINDOW_RT_SLICES: usize = 6;

/// Parameters of a clustering Run, snapshotted at gather time (so the exported JSON reflects what
/// was actually used, not later slider edits while the worker runs).
#[derive(Clone, Copy)]
struct ClusterRunParams {
    dataset_id: usize, // name (a String) is resolved at serialize time — keeps this struct Copy
    method: ClusterMethod,
    python: bool,
    eps: f32,
    min_pts: usize,
    min_cluster_size: usize,
    hdb_min_samples: usize,
    selection_eps: f64,
    rt_cycles: f64,
    im_scans: f64,
    mz_peak_widths: f64,
    ms_mask: u32,
    floor: f32,
    region: Option<[(f64, f64); 3]>,
    cycle_duration: f64,
    im_per_scan: f64,
}

/// A DIA isolation-window footprint in real units (from `/windows`); re-normalized to the focused
/// region each load to draw the precursor-selection overlay in the m/z × 1/K0 (scan) plane.
#[derive(Clone, Copy)]
struct WinRect {
    g: u32,
    mz0: f64,
    mz1: f64,
    im0: f64,
    im1: f64,
}

/// Points (the splat cloud) vs Volume (raymarched density grid).
#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    Points,
    Volume,
}

/// Clustering algorithm. The in-wasm path is DBSCAN only; the Python service does both.
#[derive(Clone, Copy, PartialEq)]
enum ClusterMethod {
    Dbscan,
    Hdbscan,
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const CANVAS_ID: &str = "tims-canvas";
/// Default point-server port (`tims-viewer DEMO --serve 8090`). Override with `?port=N` in the URL.
const DEFAULT_SERVER_PORT: u16 = 8090;
/// Default Python (sklearn) clustering-service port (`cluster_service.py --port 8091`). Override with
/// `?clusterport=N`, or a full URL via `?cluster=<url>`.
const DEFAULT_CLUSTER_PORT: u16 = 8091;
/// Default acquisition-sidecar port (`acquisition_service.py --port 8092`). Override with
/// `?acqport=N`, or a full base URL via `?acq=<url>`.
const DEFAULT_ACQ_PORT: u16 = 8092;
/// Trailing-window size for acquisition playback: the renderer keeps only the last this-many points
/// (ring buffer), so an unbounded stream can't grind the GPU. Clamped to the GPU cap.
const ACQ_RING_POINTS: usize = 1_200_000;
/// Frames per `/acq/frames` prefetch request (built in parallel server-side).
const ACQ_BATCH: u32 = 64;
/// Kick the next prefetch when fewer than this many decoded frames remain buffered.
const ACQ_LOOKAHEAD: usize = 128;

/// wasm entry point — Trunk/wasm-bindgen call this on load.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    // Off the main thread (the clustering Web Worker re-inits this same wasm) there is no DOM —
    // `window()` is None. Skip the renderer; the worker only calls `cluster_dbscan_flat`.
    if web_sys::window().is_none() {
        return;
    }
    let _ = console_log::init_with_level(log::Level::Info);
    wasm_bindgen_futures::spawn_local(async {
        if let Err(e) = run().await {
            log::error!("{e}");
            show_status(&format!("tims-viewer could not start: {e}"));
        }
    });
}

/// DBSCAN over a flat `[x,y,z, x,y,z, …]` buffer, returning per-point labels (`-1` = noise). The
/// Web Worker calls this; exported so its wasm instance has it too. `progress` (if a JS function) is
/// called `(visited, total)` ~every 1% so the worker can post progress back to the main thread.
#[wasm_bindgen]
pub fn cluster_dbscan_flat(flat: &[f32], min_pts: usize, eps: f32, progress: &JsValue) -> Vec<i32> {
    let pts: Vec<[f32; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let total = pts.len() as f64;
    let cb = progress.dyn_ref::<js_sys::Function>();
    dbscan_with_progress(&pts, eps, min_pts, |visited| {
        if let Some(f) = cb {
            let _ =
                f.call2(&JsValue::NULL, &JsValue::from_f64(visited as f64), &JsValue::from_f64(total));
        }
    })
    .0
}

/// Summary of a clustering result, for the MIDIA-style stats panel.
struct ClusterStats {
    k: usize,
    signal_pts: u64,
    noise_pts: u64,
    signal_int: f64,
    noise_int: f64,
    /// Per-cluster point count, intensity sum, and real-unit extent on each axis.
    sizes: Vec<f64>,
    int_sum: Vec<f64>,
    mz_extent: Vec<f64>,
    im_extent: Vec<f64>,
    rt_extent: Vec<f64>,
}

/// A 4D region of the run in real units (the focus lens). `imin` is the intensity floor in counts.
#[derive(Clone, Copy)]
struct Region {
    mz: (f64, f64),
    im: (f64, f64),
    rt: (f64, f64),
    imin: f32,
}

/// Live GPU + scene state, shared (single-threaded) between the render loop and resize handler.
struct Gfx {
    canvas: web_sys::HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    renderer: PointCloudRenderer,
    /// Wireframe cube + axis edges drawn around the cloud for spatial orientation.
    axes: AnnotationRenderer,
    /// DIA isolation-window overlay (separate line buffer from the axes), with the run's footprints
    /// in real units re-normalized to the focused region.
    windows: AnnotationRenderer,
    window_rects: Vec<WinRect>,
    max_window_group: u32,
    show_windows: bool,
    /// Per-group visibility bitmask (bit g-1 = group g, for g in 1..=32); groups >32 and ungrouped
    /// are always shown. Mirrors the native `group_mask`.
    window_group_mask: u32,
    /// DOM tick/axis labels (element + its world position in the `[-1,1]` cube), projected each
    /// frame onto the canvas. Empty when real-unit bounds are unknown (demo fallback).
    labels: Vec<(web_sys::HtmlElement, [f32; 3])>,
    camera: OrbitCamera,
    /// Live render parameters, mutated by the control panel and uploaded each frame.
    params: ParamsUniform,
    /// Draw mode (kept in sync with `params.render_mode`).
    point_mode: PointMode,
    /// Auto-orbit until the user first interacts, then hand control to the mouse.
    auto_rotate: bool,
    /// Mouse button currently held for dragging (0 = left/orbit, else pan); `None` = idle.
    dragging: Option<i16>,
    /// Smoothed FPS + last RAF timestamp + a tick counter to throttle the HUD update.
    fps_ema: f32,
    fps_t: f64,
    hud_tick: u32,
    /// Real-unit axis ranges `[mz, im, rt]` from `/meta`, for crop labels; `None` => show %.
    axis_bounds: Option<[(f64, f64); 3]>,
    /// Mean RT seconds per cycle (from `/meta`); 0 if unknown. Clusters RT in cycle units.
    cycle_duration: f64,
    // ---- reload state (focus-lens A3): re-fetch a region/budget and rebuild the GPU buffer ----
    /// Base `/points` URL of the server (query appended per load); empty in the demo fallback.
    points_base: String,
    /// Max points the GPU buffer can hold (`max_buffer_size / stride`); the load budget cap.
    n_cap: usize,
    supports_compaction: bool,
    /// Resident CPU copy of the uploaded points (retained for the displayed-count recount, B).
    cpu_points: Vec<GpuPoint>,
    /// Current intensity p99 — the linear floor range; read live by the floor binding.
    floor_hi: f64,
    /// True while a reload is in flight (debounces overlapping Load/Focus requests).
    reloading: bool,
    /// Bumped whenever `cpu_points` is replaced (a load). Captured at clustering Run so a deferred
    /// DBSCAN that resolves after a reload can detect the stale point set and abort.
    data_gen: u64,
    /// Set when the active filter or draw count changes; the next frame recomputes the displayed
    /// count (CPU-side over `cpu_points`) — camera-independent, so it need not run every frame.
    filter_dirty: bool,
    /// Focus stack of `(region, budget)` (`last()` = current view; index 0 = the full run). The
    /// root's budget is `None` => loaded via no-query so it hits the server's pinned full-run cache.
    /// Focus pushes; Back pops. Empty in the demo fallback (focus disabled).
    focus_stack: Vec<(Region, Option<usize>)>,
    /// Whether the projection maps reflect the current view (valid grids drawn); box-select is gated
    /// on it so a drag never focuses against stale/blank maps.
    maps_ok: bool,
    // ---- volume rendering (VOLUME_WEB_IDEA.md) ----
    view_mode: ViewMode,
    /// The raymarcher; `None` when the GPU can't host a 256³-ish 3D R16Float texture.
    volume: Option<VolumeRenderer>,
    /// CPU density grid, allocated on first volume use; rebuilt from `cpu_points` when stale.
    vol_grid: Option<VolumeGrid>,
    /// Rebuild the grid from `cpu_points` (set on load and MS-mask change).
    vol_needs_grid: bool,
    /// Raymarch samples; composite (0) vs max-intensity-projection (1).
    vol_steps: u32,
    vol_style: u32,
    /// Volume density transfer range (raw density percentiles), auto-ranged on grid build.
    vol_i_min: f32,
    vol_i_max: f32,
    // ---- clustering (CLUSTERING_WEB_PLAN.md) ----
    cluster_eps: f32,
    cluster_min_pts: usize,
    /// User lever: how many MS1 cycles `eps` bridges along RT (integer; higher = looser RT).
    cluster_rt_cycles: f64,
    /// User lever: how many TIMS scans `eps` bridges along 1/K0 (higher = looser mobility).
    cluster_im_scans: f64,
    /// User lever: max m/z peak-widths `eps` may span (lower = isotopes separate harder).
    cluster_mz_peak_widths: f64,
    /// Run-level 1/K0 per TIMS scan (from `/meta`); 0 if unknown. Anchors the 1/K0 cluster reach.
    im_per_scan: f64,
    /// True while a DBSCAN result is colouring the cloud (`params.color_mode == 1`). Invalidated on
    /// reload / focus / filter change.
    clustered: bool,
    /// The last clustering result's stats (for the right-hand panel); cleared on invalidate.
    cluster_stats: Option<ClusterStats>,
    /// Parameters captured at the last Run (gather time); the export JSON is built from this, not
    /// live fields. Cleared on invalidate.
    cluster_run_params: Option<ClusterRunParams>,
    /// JSON of the parameters used for the last clustering (built at finish from cluster_run_params);
    /// exported alongside the per-cluster CSV. Cleared on invalidate.
    cluster_params_json: Option<String>,
    /// Retained DBSCAN result for isolation: the survivor `cpu_points` indices + their labels, and
    /// which cluster is currently isolated (`None` = show all). Cleared on invalidate.
    cluster_idx: Vec<usize>,
    cluster_labels: Vec<i32>,
    cluster_sel: Option<i32>,
    /// When clustered + colouring, hide the un-clustered (noise) points (color_mode 2 vs 1).
    hide_noise: bool,
    /// Acquisition-playback metadata from the sidecar's `/acq/meta` (None = no sidecar). Populated by
    /// the startup probe; drives the live playback engine.
    acq_meta: Option<AcqMeta>,
    /// Live-acquisition playback state. `acq_playing` gates the async play-loop; `acq_cursor` is how
    /// many frames have been displayed (the fetch frontier is `acq_fetch_next`); `acq_speed`
    /// multiplies the real device cadence.
    acq_playing: bool,
    acq_cursor: u32,
    acq_speed: f64,
    /// Prefetch ring: decoded frames waiting to be displayed (front = next). Filled ahead of the
    /// cursor by a detached batch fetch so the play-loop never blocks on a build.
    acq_buffer: std::collections::VecDeque<Vec<GpuPoint>>,
    /// Next frame index to request (the prefetch frontier); `acq_cursor` is how many we've displayed.
    acq_fetch_next: u32,
    /// A batch fetch is in flight (single-flight — the sidecar serializes builds anyway).
    acq_fetching: bool,
    /// Bumped each time playback (re)starts, so an in-flight prefetch from a previous session can tell
    /// it's stale and drop its result instead of contaminating the new run's buffer.
    acq_gen: u32,
    /// Route Run through the Python (sklearn) service instead of the in-wasm DBSCAN. Auto-enabled by
    /// the startup probe when the service is reachable; toggleable in the Cluster tab.
    use_python_cluster: bool,
    /// Selected dataset's display name (from `/datasets`), for the Data summary + exported config.
    dataset_name: String,
    /// Resolved (registry-clamped) dataset id — what the server actually loaded. Seeded from the URL,
    /// corrected by `populate_datasets`; exports use this, not the raw `?dataset=` value.
    dataset_id: usize,
    /// Set once the user picks an algorithm from the dropdown, so the async service probe won't
    /// stomp their choice with its auto-select.
    cluster_algo_user_set: bool,
    /// Algorithm (Python service only; wasm is always DBSCAN) + its HDBSCAN parameters.
    cluster_method: ClusterMethod,
    cluster_min_cluster_size: usize,
    cluster_hdb_min_samples: usize, // 0 = auto (defaults to min_cluster_size)
    cluster_selection_eps: f64,
    /// Persistent DBSCAN worker (lazily created; `None` if workers are unavailable → main-thread
    /// fallback), and the in-flight job's (survivor idx, data generation, job id) awaiting its reply.
    /// `job_id` distinguishes a superseded job's late reply; `worker_failed` latches a runtime failure
    /// so we stop recreating a broken worker and use the main thread.
    cluster_worker: Option<web_sys::Worker>,
    cluster_pending: Option<(Vec<usize>, u64, u64)>,
    next_cluster_job: u64,
    cluster_worker_failed: bool,
    /// Declared last so it drops AFTER the surface/device — wgpu requires the instance to outlive
    /// everything created from it.
    _instance: wgpu::Instance,
}

impl Gfx {
    /// Match the surface (and depth texture) to the canvas's CSS size × device-pixel-ratio, so
    /// the cloud stays crisp on HiDPI displays and tracks window resizes.
    fn resize_to_display(&mut self) {
        let (w, h) = display_size(&self.canvas);
        if w == 0 || h == 0 || (w == self.config.width && h == self.config.height) {
            return;
        }
        self.canvas.set_width(w);
        self.canvas.set_height(h);
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth(&self.device, w, h);
    }

    /// Project each axis label's cube position to the canvas and position its DOM element.
    fn update_labels(&self, aspect: f32) {
        if self.labels.is_empty() {
            return;
        }
        let vp = self.camera.view_proj(aspect); // glam::Mat4, with the axis-roll baked in
        let (cw, ch) = (self.canvas.client_width() as f32, self.canvas.client_height() as f32);
        for (el, w) in &self.labels {
            let clip = vp * glam::Vec4::new(w[0], w[1], w[2], 1.0);
            let style = el.style();
            if clip.w <= 0.0 {
                let _ = style.set_property("display", "none"); // behind the camera
                continue;
            }
            let x = (clip.x / clip.w * 0.5 + 0.5) * cw;
            let y = (1.0 - (clip.y / clip.w * 0.5 + 0.5)) * ch;
            let _ = style.set_property("display", "block");
            let _ = style.set_property(
                "transform",
                &format!("translate({x:.1}px,{y:.1}px) translate(-50%,-50%)"),
            );
        }
    }

    /// Rebuild the density grid from `cpu_points` if stale, upload it, and refresh the volume
    /// uniform for this frame. No-op when volume is unsupported.
    fn ensure_volume(&mut self, aspect: f32) {
        if self.volume.is_none() {
            return;
        }
        if self.vol_grid.is_none() {
            self.vol_grid = Some(VolumeGrid::new(VOLUME_DIMS));
            self.vol_needs_grid = true;
        }
        let grid = self.vol_grid.as_mut().unwrap();
        if self.vol_needs_grid {
            // Density of all resident points, filtered by the current MS mask (the spatial crop is
            // applied in the shader via box_min/max). Deposit intensity*weight so the density is
            // independent of the downsample ratio.
            grid.clear();
            let ms = self.params.ms_mask;
            let floor = self.params.filter_min[3]; // intensity floor: dim points don't deposit
            for pt in &self.cpu_points {
                if pt.intensity < floor {
                    continue;
                }
                let is_ms2 = pt.flags & GpuPoint::MS2_FLAG != 0;
                if if is_ms2 { ms & 0b10 != 0 } else { ms & 0b01 != 0 } {
                    grid.deposit(pt.pos, pt.intensity * pt.weight);
                }
            }
            // Auto-range the density transfer to the grid's own percentiles (median..p99.9) — the
            // point intensity range is a different scale, which is why the volume looked flat.
            let (lo, hi) = grid.density_percentiles();
            self.vol_i_min = lo;
            self.vol_i_max = hi;
            self.volume.as_ref().unwrap().upload(&self.queue, grid.to_f16_scaled());
            self.vol_needs_grid = false;
            show_status(""); // clear the "building volume…" notice
        }
        let p = &self.params;
        let inv = self.camera.view_proj(aspect).inverse().to_cols_array_2d();
        let vu = VolumeUniform {
            inv_view_proj: inv,
            box_min: [p.filter_min[0], p.filter_min[1], p.filter_min[2], 0.0],
            box_max: [p.filter_max[0], p.filter_max[1], p.filter_max[2], 0.0],
            // Log over the density percentiles + exposure 1.0 (mirrors the native auto-range).
            transfer: [2.0, self.vol_i_min, self.vol_i_max, 1.0],
            steps: self.vol_steps.max(1),
            style: self.vol_style,
            colormap_id: p.colormap_id,
            n_colormaps: p.n_colormaps.max(1),
            density_scale: grid.density_scale(),
            focus: 0.0,
            _pad: [0.0; 2],
        };
        self.volume.as_ref().unwrap().update_uniform(&self.queue, &vu);
    }

    /// Render one frame. `now` is the RAF timestamp (ms). Returns `false` if the loop should stop.
    fn frame(&mut self, now: f64) -> bool {
        // Smoothed FPS, pushed to the HUD a few times a second.
        if self.fps_t > 0.0 {
            let dt = (now - self.fps_t) as f32;
            if dt > 0.0 {
                let inst = 1000.0 / dt;
                self.fps_ema = if self.fps_ema == 0.0 { inst } else { self.fps_ema * 0.9 + inst * 0.1 };
            }
        }
        self.fps_t = now;
        self.hud_tick = self.hud_tick.wrapping_add(1);
        if self.hud_tick % 12 == 0 {
            set_text("hud-fps", &format!("{:.0} fps", self.fps_ema));
        }
        // Recompute the displayed (filter-surviving) count when the filter/draw-count changed.
        if self.filter_dirty {
            let shown = recount_displayed(self);
            let resident = self.renderer.resident() as usize;
            let txt = if self.points_base.is_empty() {
                format!("{} · demo", group_short(shown))
            } else if shown >= resident {
                group_short(resident) // nothing hidden — show the loaded pool
            } else {
                format!("{} / {}", group_short(shown), group_short(resident))
            };
            set_text("hud-points", &txt);
            self.filter_dirty = false;
        }

        if self.auto_rotate {
            self.camera.yaw += 0.005;
        }
        let (w, h) = (self.config.width as f32, self.config.height as f32);
        let cam = self.camera.to_uniform(w / h, [w, h]);
        self.renderer.update_camera(&self.queue, &cam);
        self.axes.update_camera(&self.queue, &cam);
        self.windows.update_camera(&self.queue, &cam);
        if self.show_windows {
            // Cull the windows to the live crop box (mirror the point filter), like the native overlay.
            self.windows
                .update_filter(&self.queue, self.params.filter_min, self.params.filter_max, 0.0);
        }
        self.update_labels(w / h);
        let volume_view = self.view_mode == ViewMode::Volume && self.volume.is_some();
        if volume_view {
            self.ensure_volume(w / h);
        } else {
            self.renderer.update_params(&self.queue, &self.params);
        }

        let surface_tex = match self.surface.get_current_texture() {
            Ok(t) => t,
            // Transient: just skip this frame.
            Err(wgpu::SurfaceError::Timeout) => return true,
            // Stale/lost (e.g. resize, tab restore): reconfigure and retry once next frame.
            Err(wgpu::SurfaceError::Outdated) | Err(wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.config);
                return true;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                log::error!("surface out of memory — stopping render loop");
                show_status("GPU out of memory — render loop stopped.");
                return false;
            }
        };
        let view = surface_tex.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        if !volume_view {
            self.renderer.prepare(&self.queue, &mut enc);
        }
        {
            let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if volume_view {
                self.volume.as_ref().unwrap().render(&mut rpass);
            } else {
                // Cluster hues mush under additive blending — force opaque while cluster-colored.
                let pm = if self.params.color_mode >= 1 {
                    PointMode::StructuralOpaque
                } else {
                    self.point_mode
                };
                self.renderer.render(&mut rpass, pm);
            }
            self.axes.render(&mut rpass);
            if self.show_windows {
                self.windows.render(&mut rpass);
            }
        }
        self.queue.submit(std::iter::once(enc.finish()));
        surface_tex.present();
        true
    }
}

async fn run() -> Result<(), String> {
    let window = web_sys::window().ok_or("no window")?;
    let canvas = window
        .document()
        .ok_or("no document")?
        .get_element_by_id(CANVAS_ID)
        .ok_or_else(|| format!("no #{CANVAS_ID} canvas"))?
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| "#tims-canvas is not a <canvas>".to_string())?;
    let (width, height) = display_size(&canvas);
    canvas.set_width(width);
    canvas.set_height(height);

    // Choose a backend by actually creating a device. wgpu 22 asks WebGPU for an obsolete limit
    // (`maxInterStageShaderComponents`) that current Chrome rejects, so WebGPU `requestDevice`
    // fails there; we then fall back to WebGL2, which the capability-split renderer fully supports.
    // The surface is created only after a backend wins, so the canvas context matches it.
    let (instance, surface, adapter, device, queue, is_webgl) = init_gpu(&canvas).await?;

    let info = adapter.get_info();
    log::info!("using {} ({:?})", info.name, info.backend);
    set_text("hud-backend", if is_webgl { "WebGL2" } else { "WebGPU" });
    set_text("hud-adapter", &info.name);

    let downlevel = adapter.get_downlevel_capabilities().flags;
    let supports_compaction = downlevel.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        && downlevel.contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION);

    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .or_else(|| caps.formats.first().copied())
        .ok_or("surface reported no supported formats")?;
    let alpha_mode = caps
        .alpha_modes
        .first()
        .copied()
        .unwrap_or(wgpu::CompositeAlphaMode::Auto);
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);
    let depth_view = create_depth(&device, width, height);

    // Phase 2: stream real (or server-side DEMO) points from the tims-viewer point server; fall
    // back to a synthetic cloud so the page still works standalone.
    let url = points_url();
    // Cap the request to what the GPU buffer can hold AND a wasm-memory-safe ceiling: the initial
    // load must request `n=n_cap`, or a large server `--budget` (e.g. 90M pts ≈ 2.9 GB) is fetched
    // whole into the 32-bit wasm heap and OOMs before the later GPU-cap clamp — a black screen.
    let n_cap = gpu_point_cap(&device, supports_compaction).min(MAX_WEB_POINTS);
    let initial = with_query(&url, &format!("n={n_cap}"));
    // Fetch /meta first to validate the wire contract and pick up real-unit axis/intensity ranges,
    // then the binary points; fall back to a synthetic cloud if the server is absent/incompatible.
    let mut axis_bounds: Option<[(f64, f64); 3]> = None;
    let mut server_meta: Option<MetaInfo> = None;
    let (pts, is_demo_fallback) = match fetch_meta(&meta_url(&initial)).await {
        Ok(m) => match fetch_points(&initial).await {
            Ok(p) if !p.is_empty() => {
                axis_bounds = m.bounds;
                server_meta = Some(m);
                (p, false)
            }
            Ok(_) => {
                show_status("server returned no points — showing synthetic demo cloud");
                (demo_cloud(), true)
            }
            Err(e) => {
                log::warn!("point fetch failed ({e}); using synthetic demo");
                show_status("point fetch failed — showing synthetic demo cloud");
                (demo_cloud(), true)
            }
        },
        Err(e) => {
            log::warn!("no/incompatible point server ({e}); using synthetic demo");
            show_status(&format!("no point server at {url} — showing synthetic demo cloud"));
            (demo_cloud(), true)
        }
    };
    // `n_cap` (computed above, before the fetch) already bounds the request; clamp the buffer too.
    let capacity = pts.len().min(n_cap).max(1) as u32;
    if (capacity as usize) < pts.len() {
        log::warn!("capping {} -> {capacity} points (GPU buffer limit {} MB)",
            pts.len(), device.limits().max_buffer_size / (1 << 20));
    }
    let mut renderer = PointCloudRenderer::new(
        &device,
        &queue,
        format,
        DEPTH_FORMAT,
        capacity,
        supports_compaction,
    );
    let n = renderer.append(&queue, &pts);
    // The HUD point count is owned by the per-frame displayed-count recount (filter_dirty starts
    // true, so the first frame fills it).
    log::info!("uploaded {n} points (compaction: {supports_compaction})");

    // Volume raymarcher, if the GPU can host the 3D R16Float density texture.
    let vol_max = *VOLUME_DIMS.iter().max().unwrap();
    let volume = if device.limits().max_texture_dimension_3d >= vol_max {
        Some(VolumeRenderer::new(&device, &queue, format, DEPTH_FORMAT, VOLUME_DIMS))
    } else {
        log::warn!(
            "volume disabled: max_texture_dimension_3d {} < {vol_max}",
            device.limits().max_texture_dimension_3d
        );
        None
    };

    // Native-matching defaults: MS1-only, sqrt transfer, additive density. When the server gave
    // intensity percentiles, auto-expose so the cloud isn't blown out (mirrors the native viewer).
    let mut params = ParamsUniform::default();
    params.ms_mask = 0b01; // MS1 only
    params.transfer[0] = 1.0; // sqrt
    params.colormap_id = 1; // Inferno
    params.n_colormaps = COLORMAP_NAMES.len() as u32; // match the multi-row LUT so colormaps switch
    if let Some(m) = &server_meta {
        apply_auto_transfer(&mut params, m);
    }

    // The orientation box (cube + axis edges) + projected DOM tick labels.
    let mut axes = AnnotationRenderer::new(&device, format, DEPTH_FORMAT);
    axes.upload(&device, &cube_box_verts());
    axes.update_filter(&queue, [-2.0, -2.0, -2.0, 0.0], [2.0, 2.0, 2.0, 0.0], 0.0);
    // DIA window overlay: culled to the cube so off-region windows clip (filled on the /windows fetch).
    let windows = AnnotationRenderer::new(&device, format, DEPTH_FORMAT);
    windows.update_filter(&queue, [-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 0.0], 0.0);
    let labels = create_axis_labels(axis_bounds);

    let canvas_for_input = canvas.clone();
    let mut cpu_points = pts;
    cpu_points.truncate(capacity as usize);
    let floor_hi = server_meta.as_ref().map(|m| m.i_p99).filter(|x| *x > 1.0).unwrap_or(1000.0);
    let gfx = Rc::new(RefCell::new(Gfx {
        _instance: instance,
        canvas,
        surface,
        device,
        queue,
        config,
        depth_view,
        renderer,
        axes,
        windows,
        window_rects: Vec::new(),
        max_window_group: 0,
        show_windows: false,
        window_group_mask: u32::MAX,
        labels,
        camera: OrbitCamera::default(),
        params,
        point_mode: PointMode::AdditiveDensity,
        auto_rotate: true,
        dragging: None,
        fps_ema: 0.0,
        fps_t: 0.0,
        hud_tick: 0,
        axis_bounds,
        cycle_duration: server_meta.as_ref().map(|m| m.cycle_duration).unwrap_or(0.0),
        points_base: if is_demo_fallback { String::new() } else { url.clone() },
        n_cap,
        supports_compaction,
        cpu_points,
        floor_hi,
        reloading: false,
        data_gen: 0,
        filter_dirty: true,
        focus_stack: axis_bounds
            .map(|b| vec![(Region { mz: b[0], im: b[1], rt: b[2], imin: 0.0 }, None)])
            .unwrap_or_default(),
        maps_ok: false, // set by the initial render_maps below
        view_mode: ViewMode::Points,
        volume,
        vol_grid: None,
        vol_needs_grid: true,
        vol_steps: 256,
        vol_style: 0,
        vol_i_min: 1.0,
        vol_i_max: 2.0,
        cluster_eps: 0.012,
        cluster_min_pts: 8,
        cluster_rt_cycles: CLUSTER_RT_CYCLES,
        cluster_im_scans: CLUSTER_IM_SCANS,
        cluster_mz_peak_widths: CLUSTER_MZ_PEAK_WIDTHS,
        im_per_scan: server_meta.as_ref().map(|m| m.im_per_scan).unwrap_or(0.0),
        clustered: false,
        cluster_stats: None,
        cluster_run_params: None,
        cluster_params_json: None,
        cluster_idx: Vec::new(),
        cluster_labels: Vec::new(),
        cluster_sel: None,
        hide_noise: false,
        acq_meta: None,
        acq_playing: false,
        acq_cursor: 0,
        acq_speed: 1.0, // real-time (the device's true frame rate); the speed selector changes it live
        acq_buffer: std::collections::VecDeque::new(),
        acq_fetch_next: 0,
        acq_fetching: false,
        acq_gen: 0,
        dataset_name: String::new(),
        dataset_id: dataset_id(), // raw URL value; clamped to the registry by populate_datasets
        use_python_cluster: false,
        cluster_algo_user_set: false,
        cluster_method: ClusterMethod::Dbscan,
        cluster_min_cluster_size: 7,
        cluster_hdb_min_samples: 0,
        cluster_selection_eps: 0.0,
        cluster_worker: None,
        cluster_pending: None,
        next_cluster_job: 0,
        cluster_worker_failed: false,
    }));

    wire_input(&gfx, &window, &canvas_for_input);
    wire_controls(&gfx);
    // Scale the linear intensity-floor controls (slider + number, both real counts) to the data.
    let floor_hi = server_meta.as_ref().map(|m| m.i_p99).filter(|x| *x > 1.0).unwrap_or(1000.0);
    for id in ["floor", "floor-n"] {
        if let Some(inp) = by_id::<web_sys::HtmlInputElement>(id) {
            let _ = inp.set_attribute("max", &format!("{floor_hi:.0}"));
            let _ = inp.set_attribute("step", &format!("{:.0}", (floor_hi / 200.0).max(1.0)));
        }
    }
    // Push code state onto every DOM control (defeats the browser's cross-reload form restore).
    sync_controls(&gfx);
    // Distribution strips: per-axis crops + the linear intensity floor + the projection maps.
    if let Some(m) = &server_meta {
        if let Some(h) = &m.hist {
            draw_hist_backdrops(h);
        }
        if let Some(ih) = &m.i_hist {
            set_hist_svg("floor-hist", ih);
        }
        let ok = render_maps(m);
        gfx.borrow_mut().maps_ok = ok;
    }
    // Runtime detail/performance control over how many of the loaded points are drawn.
    bind_display(&gfx);
    // Load-budget control: re-fetch a different number of points from the server.
    bind_load(&gfx);
    // Focus / Back: drill into the crop+floor region at full budget, and pop back out.
    bind_focus(&gfx);
    // 2D projection minimaps: drag a box to focus a region.
    bind_maps(&gfx);
    // Points/Volume view mode + volume controls.
    bind_volume(&gfx);
    // Clustering (DBSCAN on the focused set).
    bind_cluster(&gfx);
    // DIA isolation-window overlay.
    bind_windows(&gfx);
    // Probe the Python clustering service; auto-enable the backend toggle if it's up.
    wasm_bindgen_futures::spawn_local(probe_cluster_service(gfx.clone()));
    // Probe the acquisition sidecar; reveals the Live-acquisition control if it's up.
    wasm_bindgen_futures::spawn_local(probe_acq_service(gfx.clone()));
    if let Some(btn) = by_id::<web_sys::HtmlElement>("acq-play") {
        let gfx = gfx.clone();
        add_listener(btn.as_ref(), "click", move |_e: web_sys::Event| toggle_acquisition(&gfx));
    }
    if let Some(sel) = by_id::<web_sys::HtmlSelectElement>("acq-speed") {
        let gfx = gfx.clone();
        // Live: the play-loop reads acq_speed every tick, so this retimes playback mid-run. "Max"
        // (a big multiplier) drives the cadence toward 0 so it runs as fast as frames are fetched.
        add_listener(sel.as_ref(), "change", move |_e: web_sys::Event| {
            if let Some(s) = by_id::<web_sys::HtmlSelectElement>("acq-speed") {
                if let Ok(v) = s.value().parse::<f64>() {
                    gfx.borrow_mut().acq_speed = v.max(0.05);
                }
            }
        });
    }
    // ① Data: fill the meta summary now (points/ranges/cycle); the dataset name arrives via /datasets.
    fill_data_summary(server_meta.as_ref());
    wasm_bindgen_futures::spawn_local(populate_datasets(gfx.clone()));
    // Collapsible section headers + tab switching.
    bind_panel_chrome(&gfx);

    // Fetch the run-level DIA windows once; build the overlay + legend when they arrive. The toggle
    // stays clickable regardless (it reports "no windows" on click if the fetch failed / non-DIA).
    if !url.is_empty() {
        let gfx = gfx.clone();
        let wurl = windows_url(&url);
        wasm_bindgen_futures::spawn_local(async move {
            match fetch_windows(&wurl).await {
                Ok((rects, max_group)) => {
                    {
                        let mut g = gfx.borrow_mut();
                        g.window_rects = rects;
                        g.max_window_group = max_group;
                    }
                    rebuild_window_overlay(&gfx);
                    render_window_legend(&gfx);
                }
                Err(e) => log::warn!("/windows fetch failed (restart the point server?): {e}"),
            }
        });
    }

    // Track CSS/DPR changes (window resize fires on zoom + monitor moves too).
    {
        let gfx = gfx.clone();
        let cb = Closure::wrap(Box::new(move || gfx.borrow_mut().resize_to_display()) as Box<dyn FnMut()>);
        window
            .add_event_listener_with_callback("resize", cb.as_ref().unchecked_ref())
            .map_err(|_| "failed to add resize listener".to_string())?;
        cb.forget(); // lives for the page; cancellation is a Phase-4 (embedding) concern
    }

    // requestAnimationFrame render loop (the callback receives the frame timestamp).
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move |now: f64| {
        if gfx.borrow_mut().frame(now) {
            request_animation_frame(f.borrow().as_ref().unwrap());
        }
    }) as Box<dyn FnMut(f64)>));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}

type Gpu = (wgpu::Instance, wgpu::Surface<'static>, wgpu::Adapter, wgpu::Device, wgpu::Queue, bool);

fn device_desc(limits: wgpu::Limits) -> wgpu::DeviceDescriptor<'static> {
    wgpu::DeviceDescriptor {
        label: Some("tims-web-device"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::Performance,
    }
}

/// Downlevel limits for the backend, but with the buffer-size limits raised to the adapter's real
/// maximum so large point budgets (e.g. 12M points = 384 MB) can allocate where the GPU allows.
fn web_limits(adapter: &wgpu::Adapter, is_webgl: bool) -> wgpu::Limits {
    let base = if is_webgl {
        wgpu::Limits::downlevel_webgl2_defaults()
    } else {
        wgpu::Limits::downlevel_defaults()
    };
    let a = adapter.limits();
    let mut l = base.using_resolution(a.clone());
    l.max_buffer_size = a.max_buffer_size;
    l.max_storage_buffer_binding_size = a.max_storage_buffer_binding_size;
    l
}

/// Create a wgpu surface + device, preferring WebGPU and falling back to WebGL2 if its
/// `requestDevice` fails (see the `maxInterStageShaderComponents` note in `run`). WebGPU enumerates
/// its adapter globally (no canvas context yet), so we only create the WebGPU surface once the
/// device succeeds; WebGL2 instead needs the canvas surface to find its adapter at all.
async fn init_gpu(canvas: &web_sys::HtmlCanvasElement) -> Result<Gpu, String> {
    // 1) WebGPU — global adapter; create the surface only if the device is actually obtained.
    let wgpu_inst = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });
    let webgpu_err: Option<String> = match wgpu_inst
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
    {
        Some(adapter) => {
            let limits = web_limits(&adapter, false);
            match adapter.request_device(&device_desc(limits), None).await {
                // Surface creation can also fail here — treat that like any WebGPU failure and
                // fall through to WebGL2 rather than giving up (the canvas has no context yet).
                Ok((device, queue)) => match wgpu_inst
                    .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
                {
                    Ok(surface) => return Ok((wgpu_inst, surface, adapter, device, queue, false)),
                    Err(e) => Some(format!("WebGPU create_surface: {e}")),
                },
                Err(e) => Some(format!("WebGPU requestDevice: {e:?}")),
            }
        }
        None => Some("no WebGPU adapter".to_string()),
    };
    if let Some(e) = &webgpu_err {
        log::warn!("{e}; falling back to WebGL2");
    }

    // 2) WebGL2 — needs the canvas surface to enumerate its adapter.
    let gl_inst = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL,
        ..Default::default()
    });
    let surface = gl_inst
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
        .map_err(|e| format!("create_surface (webgl2): {e}"))?;
    let adapter = gl_inst
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .ok_or_else(|| format!("no WebGL2 adapter (after {webgpu_err:?})"))?;
    let limits = web_limits(&adapter, true);
    let (device, queue) = adapter
        .request_device(&device_desc(limits), None)
        .await
        .map_err(|e| format!("WebGL2 requestDevice: {e:?} (after {webgpu_err:?})"))?;
    Ok((gl_inst, surface, adapter, device, queue, true))
}

/// Backing-store size = canvas CSS size × device-pixel-ratio (min 1×1).
fn display_size(canvas: &web_sys::HtmlCanvasElement) -> (u32, u32) {
    let dpr = web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .unwrap_or(1.0)
        .max(1.0);
    let w = ((canvas.client_width().max(1) as f64) * dpr).round() as u32;
    let h = ((canvas.client_height().max(1) as f64) * dpr).round() as u32;
    (w.max(1), h.max(1))
}

fn request_animation_frame(f: &Closure<dyn FnMut(f64)>) {
    if let Some(w) = web_sys::window() {
        let _ = w.request_animation_frame(f.as_ref().unchecked_ref());
    }
}

/// Show a message in the page's `#status` element (if present) — used for fatal errors so the
/// page reports instead of silently going blank.
fn show_status(msg: &str) {
    if let Some(el) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("status"))
    {
        el.set_text_content(Some(msg));
    }
}

/// Selected dataset id from `?dataset=N` (default 0). The frontend switches datasets by reloading
/// the page with this param, so a fresh `run()` re-inits cleanly for the new dataset.
fn dataset_id() -> usize {
    web_sys::window()
        .and_then(|w| w.location().search().ok())
        .unwrap_or_default()
        .trim_start_matches('?')
        .split('&')
        .find_map(|kv| kv.strip_prefix("dataset=").and_then(|v| v.parse::<usize>().ok()))
        .unwrap_or(0)
}

/// The point-server `/points` URL for the selected dataset (base + `?dataset=N`). `/meta` inherits
/// the query via `meta_url`; `/windows` re-appends it.
fn points_url() -> String {
    with_query(&points_base_url(), &format!("dataset={}", dataset_id()))
}

/// The point-server base URL. `?points=<url>` overrides it outright (for proxied/same-origin
/// deploys); else `?port=N` selects `http://localhost:N/points`; else [`DEFAULT_SERVER_PORT`].
fn points_base_url() -> String {
    let search = web_sys::window()
        .and_then(|w| w.location().search().ok())
        .unwrap_or_default();
    let params = search.trim_start_matches('?');
    // Explicit full-URL override (percent-decoded). Strip any baked-in `n=` (so our budget caps stay
    // authoritative) and `dataset=` (so `points_url` is the sole source of the dataset param — else
    // /points and /windows could disagree, and picker switches wouldn't take).
    if let Some(v) = params.split('&').find_map(|kv| kv.strip_prefix("points=")) {
        if !v.is_empty() {
            let decoded = js_sys::decode_uri_component(v)
                .ok()
                .and_then(|s| s.as_string())
                .unwrap_or_else(|| v.to_string());
            return match decoded.split_once('?') {
                Some((path, query)) => {
                    let kept: Vec<&str> = query
                        .split('&')
                        .filter(|kv| !kv.starts_with("n=") && !kv.starts_with("dataset="))
                        .collect();
                    if kept.is_empty() {
                        path.to_string()
                    } else {
                        format!("{path}?{}", kept.join("&"))
                    }
                }
                None => decoded,
            };
        }
    }
    let port = params
        .split('&')
        .find_map(|kv| kv.strip_prefix("port=").and_then(|v| v.parse::<u16>().ok()))
        .unwrap_or(DEFAULT_SERVER_PORT);
    format!("http://localhost:{port}/points")
}

/// The Python clustering service URL: a `?cluster=<url>` override, else the default
/// `http://localhost:<DEFAULT_CLUSTER_PORT>/cluster` (or `?clusterport=N`). Always returns a URL so
/// the Python backend is reachable from the plain page — a startup probe decides if it's actually up.
fn cluster_service_url() -> String {
    let search = web_sys::window().and_then(|w| w.location().search().ok()).unwrap_or_default();
    let params = search.trim_start_matches('?');
    if let Some(v) = params.split('&').find_map(|kv| kv.strip_prefix("cluster=")).filter(|v| !v.is_empty())
    {
        if let Some(decoded) = js_sys::decode_uri_component(v).ok().and_then(|s| s.as_string()) {
            return decoded;
        }
    }
    let port = params
        .split('&')
        .find_map(|kv| kv.strip_prefix("clusterport=").and_then(|v| v.parse::<u16>().ok()))
        .unwrap_or(DEFAULT_CLUSTER_PORT);
    format!("http://localhost:{port}/cluster")
}

/// Acquisition-playback metadata from the sidecar's `/acq/meta` (real-unit axis ranges as
/// `AxisTransform`s for normalizing streamed frame points, plus the run's frame count / cadence).
#[derive(Clone)]
struct AcqMeta {
    n_frames: u32,
    rt_cycle_length: f64,
    mz: AxisTransform,
    im: AxisTransform,
    rt: AxisTransform,
    // Used by the playback engine (next slice): auto-exposure reference + per-frame MS1/MS2 timeline.
    #[allow(dead_code)]
    i_p99: f32,
    #[allow(dead_code)]
    ms_types: Vec<i32>, // per-frame ms_type (0 = MS1, 9 = MS2)
}

/// Acquisition-sidecar base URL: `?acq=<url>` override, else `http://localhost:<DEFAULT_ACQ_PORT>`
/// (or `?acqport=N`). Endpoints are `<base>/acq/meta` and `<base>/acq/frame?id=N`.
fn acq_service_url() -> String {
    let search = web_sys::window().and_then(|w| w.location().search().ok()).unwrap_or_default();
    let params = search.trim_start_matches('?');
    if let Some(v) = params.split('&').find_map(|kv| kv.strip_prefix("acq=")).filter(|v| !v.is_empty()) {
        if let Some(decoded) = js_sys::decode_uri_component(v).ok().and_then(|s| s.as_string()) {
            return decoded;
        }
    }
    let port = params
        .split('&')
        .find_map(|kv| kv.strip_prefix("acqport=").and_then(|v| v.parse::<u16>().ok()))
        .unwrap_or(DEFAULT_ACQ_PORT);
    format!("http://localhost:{port}")
}

/// Fetch `<base>/acq/meta` and parse it; `None` if the sidecar is absent/incompatible.
async fn fetch_acq_meta(base: &str) -> Option<AcqMeta> {
    let window = web_sys::window()?;
    let resp_val = wasm_bindgen_futures::JsFuture::from(
        window.fetch_with_str(&format!("{base}/acq/meta")),
    )
    .await
    .ok()?;
    let resp: web_sys::Response = resp_val.dyn_into().ok()?;
    if !resp.ok() {
        return None;
    }
    let text = wasm_bindgen_futures::JsFuture::from(resp.text().ok()?).await.ok()?.as_string()?;
    let v = js_sys::JSON::parse(&text).ok()?;
    let num = |k: &str| jnum(&v, k);
    let arr = js_sys::Array::from(&jget(&v, "ms_types"));
    let ms_types = (0..arr.length()).map(|i| arr.get(i).as_f64().unwrap_or(0.0) as i32).collect();
    Some(AcqMeta {
        n_frames: num("n_frames").unwrap_or(0.0).max(0.0) as u32,
        rt_cycle_length: num("rt_cycle_length").filter(|x| x.is_finite() && *x > 0.0).unwrap_or(0.1),
        i_p99: num("i_p99").filter(|x| x.is_finite() && *x > 0.0).unwrap_or(1000.0) as f32,
        mz: AxisTransform::new(num("mz_min").unwrap_or(100.0), num("mz_max").unwrap_or(1700.0)),
        im: AxisTransform::new(num("im_min").unwrap_or(0.6), num("im_max").unwrap_or(1.6)),
        rt: AxisTransform::new(num("rt_min").unwrap_or(0.0), num("rt_max").unwrap_or(1.0)),
        ms_types,
    })
}

/// Decode one 24-byte wire record into a normalized `GpuPoint` (`_pad[0]` = peptide_id, `flags` bit0 =
/// fragment). `c` must be exactly 24 bytes.
fn decode_acq_record(c: &[u8], m: &AcqMeta) -> GpuPoint {
    let f = |o: usize| f32::from_le_bytes([c[o], c[o + 1], c[o + 2], c[o + 3]]);
    let u = |o: usize| u32::from_le_bytes([c[o], c[o + 1], c[o + 2], c[o + 3]]);
    GpuPoint {
        pos: [
            m.mz.normalize(f(0) as f64),
            m.im.normalize(f(4) as f64),
            m.rt.normalize(f(8) as f64),
        ],
        intensity: f(12),
        weight: 1.0,
        flags: u(20),
        _pad: [u(16), 0],
    }
}

/// GET a binary acquisition endpoint and return the raw bytes. Aborts after 20s so a hung/half-open
/// sidecar can't leave a prefetch in flight forever (which would stall the play-loop).
async fn fetch_acq_bytes(url: &str) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or("no window")?;
    let opts = web_sys::RequestInit::new();
    if let Ok(controller) = web_sys::AbortController::new() {
        opts.set_signal(Some(&controller.signal()));
        let cb = wasm_bindgen::closure::Closure::once_into_js(move || controller.abort());
        let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(cb.unchecked_ref(), 20_000);
    }
    let resp_val = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str_and_init(url, &opts))
        .await
        .map_err(|e| format!("fetch failed: {e:?}"))?;
    let resp: web_sys::Response = resp_val.dyn_into().map_err(|_| "not a Response".to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let buf = wasm_bindgen_futures::JsFuture::from(
        resp.array_buffer().map_err(|e| format!("array_buffer: {e:?}"))?,
    )
    .await
    .map_err(|e| format!("body read failed: {e:?}"))?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

/// Fetch a single frame (`<base>/acq/frame?id=N`, `id` a 0-based frame index) as `GpuPoint`s.
#[allow(dead_code)] // kept for debugging / single-frame probes; playback uses the batch path
async fn fetch_acq_frame(base: &str, id: u32, m: &AcqMeta) -> Result<Vec<GpuPoint>, String> {
    let bytes = fetch_acq_bytes(&format!("{base}/acq/frame?id={id}")).await?;
    if bytes.len() % 24 != 0 {
        return Err(format!("acq frame body not 24-byte aligned ({} bytes)", bytes.len()));
    }
    Ok(bytes.chunks_exact(24).map(|c| decode_acq_record(c, m)).collect())
}

/// Fetch a parallel batch (`<base>/acq/frames?start=N&count=K`) and split it into per-frame
/// `GpuPoint` lists. Wire layout: `u32 count`, `count × [u32 frame_index, u32 n_peaks]`, then the
/// concatenated 24-byte records for each frame in order.
async fn fetch_acq_frames(base: &str, start: u32, count: u32, m: &AcqMeta) -> Result<Vec<Vec<GpuPoint>>, String> {
    let bytes = fetch_acq_bytes(&format!("{base}/acq/frames?start={start}&count={count}")).await?;
    if bytes.len() < 4 {
        return Err("acq batch response too short".into());
    }
    let rd_u32 = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    // The server returns exactly the frames we asked for; validating up front bounds `n` (so the
    // header arithmetic below can't overflow) and stops a malformed/short batch from desyncing the
    // fetch frontier (we advanced `acq_fetch_next` by `count` already).
    let n = rd_u32(0) as usize;
    if n != count as usize {
        return Err(format!("acq batch returned {n} frames, requested {count}"));
    }
    let table_end = 4 + n * 8; // n == count <= ACQ_BATCH, so no overflow
    if bytes.len() < table_end {
        return Err("acq batch header truncated".into());
    }
    let mut counts = Vec::with_capacity(n);
    for i in 0..n {
        let idx = rd_u32(4 + i * 8);
        if idx != start + i as u32 {
            return Err(format!("acq batch frame_index {idx} != expected {}", start + i as u32));
        }
        counts.push(rd_u32(4 + i * 8 + 4) as usize);
    }
    let recs = &bytes[table_end..];
    let total: usize = counts.iter().sum();
    if total.checked_mul(24) != Some(recs.len()) {
        return Err(format!("acq batch body {} B != {total} records", recs.len()));
    }
    let mut frames = Vec::with_capacity(n);
    let mut off = 0usize;
    for c in counts {
        let mut pts = Vec::with_capacity(c);
        for k in 0..c {
            pts.push(decode_acq_record(&recs[off + k * 24..off + k * 24 + 24], m));
        }
        off += c * 24;
        frames.push(pts);
    }
    Ok(frames)
}

/// Await `ms` milliseconds (wasm has no blocking sleep): a `setTimeout`-backed Promise.
async fn sleep_ms(ms: i32) {
    let Some(window) = web_sys::window() else { return };
    let p = js_sys::Promise::new(&mut |resolve, _reject| {
        let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms.max(0));
    });
    let _ = wasm_bindgen_futures::JsFuture::from(p).await;
}

/// Enter live-acquisition playback: clear the cloud, colour by peptide id (`_pad[0]`), and start the
/// async play-loop. Toggling again stops it. (Re-uses the cluster-colour shader path; the cube bounds
/// stay as-is for now — points are already normalized to the acq ranges by `fetch_acq_frame`.)
fn toggle_acquisition(gfx: &Rc<RefCell<Gfx>>) {
    let start = {
        let g = &mut *gfx.borrow_mut();
        if g.acq_meta.is_none() {
            return;
        }
        g.acq_playing = !g.acq_playing;
        if g.acq_playing {
            // Fresh ring buffer bounded to a trailing window: the initial buffer was sized to the
            // loaded (or demo) cloud, and unbounded accumulation would grind the GPU after ~1 min.
            let cap = g.n_cap.min(ACQ_RING_POINTS).max(1) as u32;
            g.renderer =
                PointCloudRenderer::new(&g.device, &g.queue, g.config.format, DEPTH_FORMAT, cap, g.supports_compaction);
            g.renderer.enable_ring(); // append wraps + overwrites oldest -> bounded resident
            g.renderer.set_draw_count(u32::MAX);
            g.cpu_points.clear(); // acquisition drives the GPU buffer directly
            g.view_mode = ViewMode::Points; // acquisition is a point cloud; never the volume path
            // Acquisition replaces cpu_points/_pad[0], so any clustering is now invalid: a stale
            // cluster Run or a clicked cluster row would index the emptied cpu_points and panic.
            // Invalidate + clear it (and bump data_gen so an already-posted worker result is dropped).
            g.data_gen = g.data_gen.wrapping_add(1);
            invalidate_clusters_mut(g);
            g.params.color_mode = 3; // colour by _pad[0] = peptide_id + size-by-intensity (acq shader path)
            g.params.ms_mask = 0b11; // show BOTH MS1 (precursors) and MS2 (fragments) — the whole point
            // Intensity transfer so faint peaks shrink/recede: sqrt over [1, i_p99].
            let i_p99 = g.acq_meta.as_ref().map(|m| m.i_p99.max(2.0)).unwrap_or(1000.0);
            g.params.transfer = [1.0, 1.0, i_p99, 1.0];
            // Sync speed to whatever the selector currently shows (covers browser form-restore where
            // the <select> value survives a reload but g.acq_speed reset to its default).
            g.acq_speed = by_id::<web_sys::HtmlSelectElement>("acq-speed")
                .and_then(|s| s.value().parse::<f64>().ok())
                .map_or(1.0, |v| v.max(0.05));
            g.acq_cursor = 0;
            g.acq_buffer.clear();
            g.acq_fetch_next = 0;
            g.acq_fetching = false;
            g.acq_gen = g.acq_gen.wrapping_add(1); // invalidate any in-flight prefetch from a prior run
            g.filter_dirty = true;
        }
        g.acq_playing
    };
    if start {
        // Tear down the cluster panel/results UI and the save-ready state alongside the state reset.
        set_body_class("has-clusters", false);
        set_save_ready(false);
    }
    set_body_class("cluster-color", start); // the intensity colorbar is meaningless while peptide-coloured
    set_text("acq-play", if start { "⏸ Stop playback" } else { "▶ Live acquisition" });
    if start {
        wasm_bindgen_futures::spawn_local(play_loop(gfx.clone()));
    } else {
        // Stop leaves the acquisition cloud frozen; a full in-place return to the static dataset is a
        // chunk-2d transition — for now, reload/switch dataset to get it back.
        show_status("playback stopped — switch dataset or reload to return to the static view");
    }
}

/// Detached prefetch: fetch one parallel batch and push its decoded frames into `acq_buffer`, then
/// clear the in-flight flag so the play-loop can request the next. Single-flight (the sidecar
/// serializes builds anyway). Forward-only: the streamed frames are consumed as displayed.
async fn acq_prefetch(gfx: Rc<RefCell<Gfx>>, start: u32, count: u32, meta: AcqMeta, gen: u32) {
    let base = acq_service_url();
    let result = fetch_acq_frames(&base, start, count, &meta).await;
    let g = &mut *gfx.borrow_mut();
    if g.acq_gen != gen {
        return; // a newer playback session started while we were in flight — drop this result
    }
    g.acq_fetching = false;
    match result {
        Ok(frames) => {
            if g.acq_playing {
                g.acq_buffer.extend(frames);
            }
        }
        Err(e) => log::warn!("acq batch [{start},{}) failed: {e}", start + count),
    }
}

/// The async playback loop. Each tick it (1) kicks a prefetch batch if the buffer is running low and
/// none is in flight, and (2) pops one ready frame and appends it at the (speed-scaled) device
/// cadence. Fetching overlaps display, so playback isn't build-limited — it runs at real time with
/// headroom to fast-forward, bounded by the prefetch throughput.
async fn play_loop(gfx: Rc<RefCell<Gfx>>) {
    let gen = gfx.borrow().acq_gen; // this session's id; a newer toggle supersedes us
    loop {
        enum Next {
            Displayed { cur: u32, n_frames: u32, rt: f64, resident: u32, cadence_ms: i32 },
            Wait,
            Done(u32),
        }
        let (prefetch, next) = {
            let g = &mut *gfx.borrow_mut();
            if !g.acq_playing || g.acq_gen != gen {
                return;
            }
            let Some(meta) = g.acq_meta.clone() else { return };
            // (1) keep the buffer filled ahead of the cursor.
            let prefetch = if !g.acq_fetching
                && g.acq_buffer.len() < ACQ_LOOKAHEAD
                && g.acq_fetch_next < meta.n_frames
            {
                let start = g.acq_fetch_next;
                let count = ACQ_BATCH.min(meta.n_frames - start);
                g.acq_fetch_next += count;
                g.acq_fetching = true;
                Some((start, count, meta.clone()))
            } else {
                None
            };
            // (2) display one buffered frame.
            let next = if let Some(frame) = g.acq_buffer.pop_front() {
                if !frame.is_empty() {
                    g.renderer.append(&g.queue, &frame);
                    g.renderer.set_draw_count(u32::MAX);
                    g.filter_dirty = true;
                }
                g.acq_cursor += 1;
                Next::Displayed {
                    cur: g.acq_cursor,
                    n_frames: meta.n_frames,
                    rt: g.acq_cursor as f64 * meta.rt_cycle_length,
                    resident: g.renderer.resident(),
                    cadence_ms: (meta.rt_cycle_length * 1000.0 / g.acq_speed.max(0.05)) as i32,
                }
            } else if g.acq_fetch_next >= meta.n_frames && !g.acq_fetching {
                Next::Done(meta.n_frames) // buffer drained and nothing left to fetch
            } else {
                Next::Wait // buffer momentarily empty — prefetch is catching up
            };
            (prefetch, next)
        };
        if let Some((start, count, meta)) = prefetch {
            wasm_bindgen_futures::spawn_local(acq_prefetch(gfx.clone(), start, count, meta, gen));
        }
        match next {
            Next::Displayed { cur, n_frames, rt, resident, cadence_ms } => {
                show_status(&format!(
                    "▶ frame {cur} / {n_frames} · RT {rt:.0}s · {} pts",
                    group_short(resident as usize),
                ));
                sleep_ms(cadence_ms).await;
            }
            Next::Wait => sleep_ms(8).await,
            Next::Done(n) => {
                gfx.borrow_mut().acq_playing = false;
                set_body_class("cluster-color", false); // restore the intensity colorbar (manual stop does too)
                set_text("acq-play", "▶ Live acquisition");
                show_status(&format!("acquisition complete — {n} frames"));
                return;
            }
        }
    }
}

/// Probe the acquisition sidecar (GET its health route); on success stash `/acq/meta` on `Gfx` so the
/// playback engine (next slice) can offer "Live acquisition" mode.
async fn probe_acq_service(gfx: Rc<RefCell<Gfx>>) {
    let base = acq_service_url();
    let Some(window) = web_sys::window() else { return };
    // Abort after 3s — the sidecar is single-threaded, so a probe issued while it's mid-build (or a
    // half-open service) must not pin detection forever.
    let opts = web_sys::RequestInit::new();
    if let Ok(controller) = web_sys::AbortController::new() {
        opts.set_signal(Some(&controller.signal()));
        let cb = wasm_bindgen::closure::Closure::once_into_js(move || controller.abort());
        let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(cb.unchecked_ref(), 3000);
    }
    let healthy = match wasm_bindgen_futures::JsFuture::from(window.fetch_with_str_and_init(&base, &opts)).await {
        Ok(v) => match v.dyn_into::<web_sys::Response>() {
            Ok(r) if r.ok() => match r.text() {
                Ok(p) => wasm_bindgen_futures::JsFuture::from(p)
                    .await
                    .ok()
                    .and_then(|t| t.as_string())
                    .is_some_and(|s| s.contains("acquisition service")),
                Err(_) => false,
            },
            _ => false,
        },
        Err(_) => false,
    };
    if !healthy {
        return;
    }
    if let Some(meta) = fetch_acq_meta(&base).await {
        if meta.n_frames == 0 {
            log::warn!("acquisition sidecar reports 0 frames — not offering Live");
            return;
        }
        log::info!(
            "acquisition sidecar: {} frames, cadence {:.3}s/frame, mz [{:.0},{:.0}] 1/K0 [{:.3},{:.3}]",
            meta.n_frames, meta.rt_cycle_length, meta.mz.min, meta.mz.max, meta.im.min, meta.im.max
        );
        gfx.borrow_mut().acq_meta = Some(meta);
        // Reveal the Live-acquisition control now that the sidecar is confirmed.
        if let Some(row) = by_id::<web_sys::HtmlElement>("acq-row") {
            let _ = row.style().set_property("display", "flex");
        }
    }
}

/// Probe the cluster service (GET its health route); on success enable + auto-select the Python
/// backend so it's a ready option from the plain page (no `?cluster=` needed).
async fn probe_cluster_service(gfx: Rc<RefCell<Gfx>>) {
    let Some(window) = web_sys::window() else {
        return;
    };
    // Abort the probe after 3s so a half-open service (200 then a stalled body) can't pin the UI at
    // "checking…" forever.
    let opts = web_sys::RequestInit::new();
    if let Ok(controller) = web_sys::AbortController::new() {
        opts.set_signal(Some(&controller.signal()));
        let cb = wasm_bindgen::closure::Closure::once_into_js(move || controller.abort());
        let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(cb.unchecked_ref(), 3000);
    }
    // Verify it's actually our service (a 200 from some other process on the port isn't enough).
    let ok = match wasm_bindgen_futures::JsFuture::from(
        window.fetch_with_str_and_init(&cluster_service_url(), &opts),
    )
    .await
    {
        Ok(v) => match v.dyn_into::<web_sys::Response>() {
            Ok(r) if r.ok() => match r.text() {
                Ok(p) => wasm_bindgen_futures::JsFuture::from(p)
                    .await
                    .ok()
                    .and_then(|t| t.as_string())
                    .is_some_and(|s| s.contains("cluster service")),
                Err(_) => false,
            },
            _ => false,
        },
        Err(_) => false,
    };
    if ok {
        set_disabled("opt-sklearn", false);
        set_disabled("opt-hdbscan", false);
        // Prefer sklearn DBSCAN when the service is up — unless the user already picked an algorithm.
        if !gfx.borrow().cluster_algo_user_set {
            if let Some(sel) = by_id::<web_sys::HtmlSelectElement>("cl-algo") {
                sel.set_value("sklearn");
            }
            apply_cluster_algo(&gfx, "sklearn");
        }
        set_text("cl-algo-status", "sklearn ✓");
    } else {
        set_text("cl-algo-status", "offline · built-in");
    }
}

/// POST the (already axis-scaled) flat positions to the Python clustering service and read back the
/// per-point labels. Same wire format as the wasm worker: float32 xyz triples in, int32 labels out.
async fn fetch_cluster(svc: &str, flat: &[f32], query: &str) -> Result<Vec<i32>, String> {
    let window = web_sys::window().ok_or("no window")?;
    let arr = js_sys::Float32Array::from(flat);
    let opts = web_sys::RequestInit::new();
    opts.set_method("POST");
    opts.set_body(&arr.buffer());
    let url = with_query(svc, query);
    let resp_val = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str_and_init(&url, &opts))
        .await
        .map_err(|e| format!("fetch failed: {e:?}"))?;
    let resp: web_sys::Response = resp_val.dyn_into().map_err(|_| "not a Response".to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let buf = wasm_bindgen_futures::JsFuture::from(
        resp.array_buffer().map_err(|e| format!("array_buffer: {e:?}"))?,
    )
    .await
    .map_err(|e| format!("body read failed: {e:?}"))?;
    if let Some(ab) = buf.dyn_ref::<js_sys::ArrayBuffer>() {
        if ab.byte_length() % 4 != 0 {
            return Err("cluster response not int32-aligned".to_string());
        }
    }
    Ok(js_sys::Int32Array::new(&buf).to_vec())
}

/// Fetch packed `GpuPoint` bytes from the server and reinterpret them (the server already
/// normalized positions to the `[-1, 1]` cube, so they upload as-is).
async fn fetch_points(url: &str) -> Result<Vec<GpuPoint>, String> {
    let window = web_sys::window().ok_or("no window")?;
    let resp_val = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|e| format!("fetch failed: {e:?}"))?;
    let resp: web_sys::Response = resp_val.dyn_into().map_err(|_| "not a Response".to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let buf = wasm_bindgen_futures::JsFuture::from(
        resp.array_buffer().map_err(|e| format!("array_buffer: {e:?}"))?,
    )
    .await
    .map_err(|e| format!("body read failed: {e:?}"))?;

    let bytes = js_sys::Uint8Array::new(&buf);
    let len = bytes.length();
    let stride = std::mem::size_of::<GpuPoint>() as u32;
    if len == 0 {
        return Ok(Vec::new()); // a valid empty region (the caller decides what to show)
    }
    if len % stride != 0 {
        return Err(format!("body {len} bytes is not a multiple of the {stride}-byte point stride"));
    }
    let mut pts = vec![GpuPoint::zeroed(); (len / stride) as usize];
    // dst is exactly `len` bytes and 4-aligned (it's a Vec<GpuPoint>); copy_to requires equal len.
    let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut pts);
    bytes.copy_to(dst);
    Ok(pts)
}

/// Real-unit + exposure context from `/meta`: axis ranges `[mz, im, rt]` (None if absent/malformed),
/// intensity percentiles, and the downsample stride (per-point weight) for auto-exposure.
struct MetaInfo {
    bounds: Option<[(f64, f64); 3]>,
    i_p1: f64,
    i_p50: f64,
    i_p99: f64,
    stride: f64,
    /// Per-axis density histograms `[mz, im, rt]` (each over the full axis range), for the crop
    /// distribution strips. `None` if the server didn't provide them.
    hist: Option<[Vec<u32>; 3]>,
    /// Linear intensity distribution over `[0, p99]` (real counts), for the floor strip.
    i_hist: Option<Vec<u32>>,
    /// 2D density projections `[mz×im, mz×rt, im×rt]` (each `proj_bins²`) for the box-select maps.
    proj: Option<[Vec<u32>; 3]>,
    proj_bins: usize,
    /// Mean RT gap (seconds) between consecutive MS1 frames; 0 if unknown. Used to cluster RT in
    /// cycle units so precursors stay connected across cycles regardless of focus.
    cycle_duration: f64,
    /// Total resident points the server holds for this region/run (for the Data summary).
    n_points: f64,
    /// Run-level 1/K0 per TIMS scan (full mobility span / ramp length); 0 if unknown. Anchors the
    /// clustering's 1/K0 reach to a fixed number of scans, focus-independent.
    im_per_scan: f64,
}

/// Derive the `/meta` URL from the `/points` URL: replace only the trailing path endpoint and keep
/// any query string. Non-`/points` overrides are left unchanged (the meta fetch then just fails).
fn meta_url(points: &str) -> String {
    let (path, query) = match points.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (points, None),
    };
    let meta_path = path
        .strip_suffix("/points")
        .map(|base| format!("{base}/meta"))
        .unwrap_or_else(|| path.to_string());
    match query {
        Some(q) => format!("{meta_path}?{q}"),
        None => meta_path,
    }
}

/// The run-level `/windows` URL (region-independent, but per-dataset: re-append `?dataset=N`).
fn windows_url(points: &str) -> String {
    let path = points.split('?').next().unwrap_or(points);
    let base = path
        .strip_suffix("/points")
        .map(|base| format!("{base}/windows"))
        .unwrap_or_else(|| path.to_string());
    with_query(&base, &format!("dataset={}", dataset_id()))
}

/// The `/datasets` registry URL (derived from the points base; dataset-independent).
fn datasets_url() -> String {
    let base = points_base_url();
    let path = base.split('?').next().unwrap_or(&base);
    path.strip_suffix("/points")
        .map(|b| format!("{b}/datasets"))
        .unwrap_or_else(|| path.to_string())
}

/// Switch the active dataset by reloading the page with `?dataset=N` — a fresh `run()` re-inits the
/// scene (camera, focus, clusters, bounds) cleanly for the new dataset.
fn switch_dataset(id: usize) {
    let Some(w) = web_sys::window() else { return };
    let loc = w.location();
    let search = loc.search().unwrap_or_default();
    let mut parts: Vec<String> = search
        .trim_start_matches('?')
        .split('&')
        .filter(|kv| !kv.is_empty() && !kv.starts_with("dataset="))
        .map(|s| s.to_string())
        .collect();
    parts.push(format!("dataset={id}"));
    let _ = loc.set_search(&parts.join("&")); // navigates -> fresh load for the new dataset
}

/// Fetch the dataset registry: always stash the active dataset's name (for the Data summary +
/// config export), and — if more than one — reveal + populate the picker (selecting one reloads
/// with `?dataset=N`). Silent no-op when the server is absent.
async fn populate_datasets(gfx: Rc<RefCell<Gfx>>) {
    let Some(window) = web_sys::window() else { return };
    let Some(doc) = window.document() else { return };
    let resp_val = match wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(&datasets_url())).await
    {
        Ok(v) => v,
        Err(_) => return,
    };
    let Ok(resp) = resp_val.dyn_into::<web_sys::Response>() else { return };
    if !resp.ok() {
        return;
    }
    let text = match resp.text() {
        Ok(p) => match wasm_bindgen_futures::JsFuture::from(p).await {
            Ok(t) => t.as_string().unwrap_or_default(),
            Err(_) => return,
        },
        Err(_) => return,
    };
    let Ok(v) = js_sys::JSON::parse(&text) else { return };
    let arr = js_sys::Array::from(&v);
    if arr.length() == 0 {
        return;
    }
    // The server clamps an out-of-range ?dataset to the last id, so clamp here too — else the picker
    // would highlight the wrong (first) option while a different dataset is actually loaded.
    let cur = dataset_id().min(arr.length() as usize - 1);
    // Stash + show the active dataset's name BEFORE the single-dataset early return, so even a
    // one-entry registry fills the Data summary.
    let cur_name = jget(&arr.get(cur as u32), "name").as_string().unwrap_or_default();
    {
        let mut g = gfx.borrow_mut();
        g.dataset_id = cur; // the registry-clamped id the server actually loaded
        if !cur_name.is_empty() {
            set_text("data-name", &cur_name);
            g.dataset_name = cur_name;
        }
    }
    if arr.length() <= 1 {
        return; // only one — name is set, but no picker to show
    }
    let Some(sel) = by_id::<web_sys::HtmlSelectElement>("dataset-sel") else { return };
    sel.set_inner_html("");
    for i in 0..arr.length() {
        let item = arr.get(i);
        let id = jnum(&item, "id").unwrap_or(i as f64) as usize;
        let name = jget(&item, "name").as_string().unwrap_or_else(|| format!("dataset {id}"));
        if let Ok(opt) = doc.create_element("option") {
            opt.set_attribute("value", &id.to_string()).ok();
            opt.set_text_content(Some(&name));
            if id == cur {
                opt.set_attribute("selected", "selected").ok();
            }
            let _ = sel.append_child(&opt);
        }
    }
    // Reveal the picker (hidden by default for the single-dataset case).
    if let Some(pick) = by_id::<web_sys::HtmlElement>("dspick") {
        let _ = pick.style().set_property("display", "flex");
    }
    let s2 = sel.clone();
    add_listener(sel.as_ref(), "change", move |_e: web_sys::Event| {
        if let Ok(id) = s2.value().parse::<usize>() {
            switch_dataset(id);
        }
    });
}

/// Fill the ① Data summary (points / ranges / cycle) from the loaded meta. The dataset NAME is set
/// separately by `populate_datasets` (it comes from `/datasets`).
fn fill_data_summary(meta: Option<&MetaInfo>) {
    let Some(m) = meta else {
        for id in ["data-points", "data-mz", "data-im", "data-rt", "data-cycle"] {
            set_text(id, "—");
        }
        return;
    };
    set_text("data-points", &group(m.n_points as usize));
    if let Some(b) = m.bounds {
        set_text("data-mz", &format!("{:.1} – {:.1}", b[0].0, b[0].1));
        set_text("data-im", &format!("{:.3} – {:.3}", b[1].0, b[1].1));
        set_text("data-rt", &format!("{:.1} – {:.1} s", b[2].0, b[2].1));
    }
    if m.cycle_duration > 0.0 {
        set_text("data-cycle", &format!("{:.3} s", m.cycle_duration));
    }
}

/// Fetch the run's DIA isolation-window footprints (real units) + the max group id.
async fn fetch_windows(url: &str) -> Result<(Vec<WinRect>, u32), String> {
    let window = web_sys::window().ok_or("no window")?;
    let resp_val = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|e| format!("fetch failed: {e:?}"))?;
    let resp: web_sys::Response = resp_val.dyn_into().map_err(|_| "not a Response".to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let text = wasm_bindgen_futures::JsFuture::from(resp.text().map_err(|e| format!("text: {e:?}"))?)
        .await
        .map_err(|e| format!("body read failed: {e:?}"))?
        .as_string()
        .ok_or("windows body is not text")?;
    let v = js_sys::JSON::parse(&text).map_err(|_| "windows is not valid JSON".to_string())?;
    let max_group = jnum(&v, "max_window_group").unwrap_or(0.0).max(0.0) as u32;
    let arr = js_sys::Array::from(&jget(&v, "windows"));
    let mut rects = Vec::with_capacity(arr.length() as usize);
    for i in 0..arr.length() {
        let row = js_sys::Array::from(&arr.get(i));
        if row.length() < 5 {
            continue;
        }
        let f = |k: u32| row.get(k).as_f64().unwrap_or(0.0);
        rects.push(WinRect { g: f(0) as u32, mz0: f(1), mz1: f(2), im0: f(3), im1: f(4) });
    }
    Ok((rects, max_group))
}

/// Build the window-overlay line vertices: each footprint as an `(m/z × 1/K0)` rectangle at
/// `DIA_WINDOW_RT_SLICES` RT slices, normalized to the focused region's `bounds` and colored by group.
fn build_window_verts(
    rects: &[WinRect],
    bounds: [(f64, f64); 3],
    max_group: u32,
    mask: u32,
) -> Vec<LineVertex> {
    let norm = |val: f64, axis: usize| -> f32 {
        let (lo, hi) = bounds[axis];
        (((val - lo) / (hi - lo).max(1e-12)) * 2.0 - 1.0) as f32
    };
    let mut v: Vec<LineVertex> = Vec::new();
    let denom = max_group.max(1);
    for r in rects {
        // Per-group visibility: groups 1..=32 gate on their mask bit; >32 and group 0 always show.
        if r.g >= 1 && r.g <= 32 && (mask & (1u32 << (r.g - 1))) == 0 {
            continue;
        }
        // Order each axis first: a footprint can arrive descending (1/K0 falls as scan rises, so
        // the scan_lo/scan_hi -> 1/K0 conversion yields im0 > im1). Without this the clamp below
        // produces hi < lo and the next check drops EVERY window — the overlay vanishes while the
        // legend (which only needs max_window_group) still shows.
        let (rmz0, rmz1) = (r.mz0.min(r.mz1), r.mz0.max(r.mz1));
        let (rim0, rim1) = (r.im0.min(r.im1), r.im0.max(r.im1));
        // Clamp the axis-aligned footprint to the region so a partly-off-region window clips cleanly
        // (the per-vertex shader cull would otherwise leave stubs); drop windows fully outside.
        let (mz0r, mz1r) = (rmz0.max(bounds[0].0), rmz1.min(bounds[0].1));
        let (im0r, im1r) = (rim0.max(bounds[1].0), rim1.min(bounds[1].1));
        if mz1r <= mz0r || im1r <= im0r {
            continue;
        }
        let color = tims_viewer::render::colormap::group_color(r.g, denom);
        let (mz0, mz1) = (norm(mz0r, 0), norm(mz1r, 0));
        let (im0, im1) = (norm(im0r, 1), norm(im1r, 1));
        for i in 0..DIA_WINDOW_RT_SLICES {
            let rt_real =
                bounds[2].0 + (bounds[2].1 - bounds[2].0) * (i as f64 + 0.5) / DIA_WINDOW_RT_SLICES as f64;
            let rt = norm(rt_real, 2);
            let corners = [[mz0, im0, rt], [mz1, im0, rt], [mz1, im1, rt], [mz0, im1, rt]];
            for e in 0..4 {
                v.push(LineVertex::new(corners[e], color));
                v.push(LineVertex::new(corners[(e + 1) % 4], color));
            }
        }
    }
    v
}

/// Re-normalize the run's window footprints to the current region and upload the overlay. A no-op-ish
/// empty upload when there are no windows or no real-unit bounds.
fn rebuild_window_overlay(gfx: &Rc<RefCell<Gfx>>) {
    let verts = {
        let g = gfx.borrow();
        match g.axis_bounds {
            Some(b) if !g.window_rects.is_empty() => {
                build_window_verts(&g.window_rects, b, g.max_window_group, g.window_group_mask)
            }
            _ => Vec::new(),
        }
    };
    let mut g = gfx.borrow_mut();
    let g = &mut *g;
    g.windows.upload(&g.device, &verts);
}

/// CSS color matching the rendered window's `group_color` (so the legend swatch is exact).
fn group_css(g: u32, n: u32) -> String {
    let [r, gg, b] = tims_viewer::render::colormap::group_color(g, n.max(1));
    format!("rgb({},{},{})", (r * 255.0) as u8, (gg * 255.0) as u8, (b * 255.0) as u8)
}

/// (Re)render the per-group window legend: a toggleable colored chip per group (capped at 32, like
/// the native mask). Groups beyond 32 are always shown.
fn render_window_legend(gfx: &Rc<RefCell<Gfx>>) {
    let (max_g, mask) = {
        let g = gfx.borrow();
        (g.max_window_group, g.window_group_mask)
    };
    if max_g == 0 {
        set_html("win-groups", "");
        return;
    }
    let n = max_g.min(32);
    let mut html = String::new();
    for g in 1..=n {
        let on = (mask & (1u32 << (g - 1))) != 0;
        let cls = if on { "wg on" } else { "wg" };
        html.push_str(&format!(
            "<button class=\"{cls}\" data-g=\"{g}\"><span class=\"wg-dot\" style=\"background:{}\"></span>{g}</button>",
            group_css(g, max_g)
        ));
    }
    if max_g > 32 {
        html.push_str(&format!("<span class=\"cs-more\">+{} more (shown)</span>", max_g - 32));
    }
    set_html("win-groups", &html);
}

fn jget(obj: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(obj, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
}
fn jnum(obj: &JsValue, key: &str) -> Option<f64> {
    jget(obj, key).as_f64()
}

/// Fetch `/meta` and validate the wire contract (version + point stride) before trusting `/points`.
async fn fetch_meta(url: &str) -> Result<MetaInfo, String> {
    let window = web_sys::window().ok_or("no window")?;
    let resp_val = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|e| format!("fetch failed: {e:?}"))?;
    let resp: web_sys::Response = resp_val.dyn_into().map_err(|_| "not a Response".to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let text = wasm_bindgen_futures::JsFuture::from(resp.text().map_err(|e| format!("text: {e:?}"))?)
        .await
        .map_err(|e| format!("body read failed: {e:?}"))?
        .as_string()
        .ok_or("meta body is not text")?;
    let v = js_sys::JSON::parse(&text).map_err(|_| "meta is not valid JSON".to_string())?;

    if jnum(&v, "version") != Some(1.0) {
        return Err("unsupported or missing meta version (expected 1)".to_string());
    }
    // Exact stride check — no `as usize` truncation (32.9 must NOT pass as 32).
    let expect = std::mem::size_of::<GpuPoint>() as f64;
    if jnum(&v, "point_stride") != Some(expect) {
        return Err(format!("point stride mismatch (expected {expect})"));
    }

    // Bounds are best-effort: render points regardless, but only label in real units if every axis
    // is a valid [lo, hi] with lo < hi; otherwise the crop labels fall back to percentages.
    let b = jget(&v, "bounds");
    let axis = |k: &str| -> Option<(f64, f64)> {
        let a = jget(&b, k);
        if !a.is_object() {
            return None; // arrays are objects; this also rejects null/number/missing
        }
        let arr = js_sys::Array::from(&a);
        if arr.length() < 2 {
            return None;
        }
        let (lo, hi) = (arr.get(0).as_f64()?, arr.get(1).as_f64()?);
        (lo.is_finite() && hi.is_finite() && hi > lo).then_some((lo, hi))
    };
    let bounds = match (axis("mz"), axis("im"), axis("rt")) {
        (Some(mz), Some(im), Some(rt)) => Some([mz, im, rt]),
        _ => None,
    };
    let it = jget(&v, "intensity");
    let pos = |key: &str| jnum(&it, key).filter(|x| x.is_finite() && *x > 0.0);
    let stride = jnum(&v, "downsample_stride")
        .filter(|x| x.is_finite() && *x >= 1.0)
        .unwrap_or(1.0);

    // Optional per-axis density histograms + the intensity (floor) histogram.
    let h = jget(&v, "hist");
    let hist = match (js_u32_array(&h, "mz"), js_u32_array(&h, "im"), js_u32_array(&h, "rt")) {
        (Some(mz), Some(im), Some(rt)) => Some([mz, im, rt]),
        _ => None,
    };
    let i_hist = js_u32_array(&it, "hist");

    // 2D projections for the box-select maps. Use the server's explicit `bins` and require all
    // three grids to be exactly `bins²`, else drop them (so the maps never show a malformed grid).
    let pj = jget(&v, "proj");
    let bins = jnum(&pj, "bins").filter(|x| x.is_finite() && *x >= 1.0).map(|x| x as usize);
    let proj = match (
        bins,
        js_u32_array(&pj, "mz_im"),
        js_u32_array(&pj, "mz_rt"),
        js_u32_array(&pj, "im_rt"),
    ) {
        (Some(b), Some(a), Some(c), Some(d))
            if a.len() == b * b && c.len() == b * b && d.len() == b * b =>
        {
            Some([a, c, d])
        }
        _ => None,
    };
    let proj_bins = if proj.is_some() { bins.unwrap_or(0) } else { 0 };

    Ok(MetaInfo {
        bounds,
        i_p1: pos("p1").unwrap_or(0.0),
        i_p50: pos("p50").unwrap_or(0.0),
        i_p99: pos("p99").unwrap_or(0.0),
        stride,
        hist,
        i_hist,
        proj,
        proj_bins,
        cycle_duration: jnum(&v, "cycle_duration").filter(|x| x.is_finite() && *x > 0.0).unwrap_or(0.0),
        n_points: jnum(&v, "n_points").filter(|x| x.is_finite() && *x >= 0.0).unwrap_or(0.0),
        im_per_scan: jnum(&v, "im_per_scan_1k0").filter(|x| x.is_finite() && *x > 0.0).unwrap_or(0.0),
    })
}

/// Read a JSON `u32` array field (`obj[key]`); `None` if absent or not a non-empty array.
fn js_u32_array(obj: &JsValue, key: &str) -> Option<Vec<u32>> {
    let a = jget(obj, key);
    if !a.is_object() {
        return None;
    }
    let arr = js_sys::Array::from(&a);
    if arr.length() == 0 {
        return None;
    }
    Some((0..arr.length()).map(|i| arr.get(i).as_f64().unwrap_or(0.0).max(0.0) as u32).collect())
}

/// Label for a per-axis crop: real units when `/meta` bounds are known, else a percentage.
fn crop_label(bounds: Option<[(f64, f64); 3]>, axis: usize, a: f32, b: f32) -> String {
    if a <= -0.999 && b >= 0.999 {
        return "full".into();
    }
    let (fa, fb) = ((a * 0.5 + 0.5) as f64, (b * 0.5 + 0.5) as f64);
    match bounds {
        Some(bnd) => {
            let (lo, hi) = bnd[axis];
            let (ra, rb) = (lo + fa * (hi - lo), lo + fb * (hi - lo));
            let p = if axis == 1 { 3 } else { 0 }; // 1/K0 needs decimals; m/z & RT are integer-ish
            format!("{ra:.*}–{rb:.*}", p, p)
        }
        None => format!("{:.0}–{:.0}%", fa * 100.0, fb * 100.0),
    }
}

/// Auto-set the transfer function + exposure from intensity percentiles, mirroring the native
/// viewer's `auto_transfer_points`, so the additive cloud is legible (not blown out) on load.
fn apply_auto_transfer(params: &mut ParamsUniform, m: &MetaInfo) {
    if m.i_p50 <= 0.0 {
        return;
    }
    let p1 = m.i_p1.max(1.0) as f32;
    let p50 = m.i_p50.max(m.i_p1) as f32;
    let p99 = m.i_p99.max(m.i_p50 * 1.0001) as f32;
    params.transfer[0] = 1.0; // sqrt: lifts the mid-range without flattening bright peaks
    params.transfer[1] = p1; // i_min
    params.transfer[2] = (p99 * 4.0).max(p50 * 1.0001); // i_max — headroom above p99 for peaks
    // Exposure solves a single median splat to ~0.35 alpha; weight = downsample stride.
    let weight = (m.stride as f32).max(1.0);
    params.transfer[3] = (0.35 / (params.opacity.max(0.05) * weight)).clamp(0.02, 4.0);
}

/// The 12 edges of the normalized `[-1, 1]` cube around the data, with the three principal edges
/// (meeting at the `(-1,-1,-1)` corner) tinted by axis: m/z orange, 1/K₀ cyan, RT green.
fn cube_box_verts() -> Vec<LineVertex> {
    const DIM: [f32; 3] = [0.30, 0.36, 0.48];
    const MZ: [f32; 3] = [1.0, 0.59, 0.35];
    const IM: [f32; 3] = [0.47, 0.86, 1.0];
    const RT: [f32; 3] = [0.59, 0.92, 0.59];
    let c = [-1.0f32, 1.0f32];
    let mut v = Vec::with_capacity(24);
    let mut seg = |a: [f32; 3], b: [f32; 3], col: [f32; 3]| {
        v.push(LineVertex::new(a, col));
        v.push(LineVertex::new(b, col));
    };
    for &j in &[0usize, 1] {
        for &k in &[0usize, 1] {
            let col = if j == 0 && k == 0 { MZ } else { DIM };
            seg([c[0], c[j], c[k]], [c[1], c[j], c[k]], col); // edges along x (m/z)
        }
    }
    for &i in &[0usize, 1] {
        for &k in &[0usize, 1] {
            let col = if i == 0 && k == 0 { IM } else { DIM };
            seg([c[i], c[0], c[k]], [c[i], c[1], c[k]], col); // edges along y (1/K0)
        }
    }
    for &i in &[0usize, 1] {
        for &j in &[0usize, 1] {
            let col = if i == 0 && j == 0 { RT } else { DIM };
            seg([c[i], c[j], c[0]], [c[i], c[j], c[1]], col); // edges along z (RT)
        }
    }
    v
}

const MZ_CSS: &str = "#ff9659";
const IM_CSS: &str = "#78dcff";
const RT_CSS: &str = "#97eb97";

/// Tick + axis-name labels in real units, as `(text, cube-position, css-color)`. Ticks ride the
/// three principal cube edges (the colored ones); positions are pushed slightly outside the box.
fn axis_label_specs(bounds: [(f64, f64); 3]) -> Vec<(String, [f32; 3], &'static str)> {
    let (mz, im, rt) = (bounds[0], bounds[1], bounds[2]);
    let (txm, txi, txr) = (
        AxisTransform::new(mz.0, mz.1),
        AxisTransform::new(im.0, im.1),
        AxisTransform::new(rt.0, rt.1),
    );
    let mut out = Vec::new();
    // m/z (x) ticks along the y=-1,z=-1 edge, label pushed down (-y).
    for t in ticks_for(mz.0, mz.1, 5, |v| txm.normalize(v)) {
        out.push((fmt_tick(Axis::Mz, t.value, (mz.1 - mz.0).abs()), [t.norm, -1.14, -1.0], MZ_CSS));
    }
    // 1/K0 (y) ticks along the x=-1,z=-1 edge, label pushed out (-x).
    for t in ticks_for(im.0, im.1, 5, |v| txi.normalize(v)) {
        out.push((fmt_tick(Axis::Im, t.value, (im.1 - im.0).abs()), [-1.14, t.norm, -1.0], IM_CSS));
    }
    // RT (z) ticks along the x=-1,y=-1 edge, label pushed down (-y).
    for t in ticks_for(rt.0, rt.1, 5, |v| txr.normalize(v)) {
        out.push((fmt_tick(Axis::Rt, t.value, (rt.1 - rt.0).abs()), [-1.0, -1.14, t.norm], RT_CSS));
    }
    // Axis names (with the RT unit) at the far ends of each colored edge.
    let rt_name = if (rt.1 - rt.0).abs() > RT_MINUTES_SPAN { "RT (min)" } else { "RT (s)" };
    out.push(("m/z".into(), [1.28, -1.14, -1.0], MZ_CSS));
    out.push(("1/K₀".into(), [-1.14, 1.28, -1.0], IM_CSS));
    out.push((rt_name.into(), [-1.0, -1.14, 1.28], RT_CSS));
    out
}

/// Max points the GPU buffer can hold = `max_buffer_size / stride` (also bounded by the storage-
/// bind limit on the compaction path). The hard ceiling on a load budget.
fn gpu_point_cap(device: &wgpu::Device, supports_compaction: bool) -> usize {
    let stride = std::mem::size_of::<GpuPoint>() as u64;
    let mut cap = (device.limits().max_buffer_size / stride) as usize;
    if supports_compaction {
        cap = cap.min((device.limits().max_storage_buffer_binding_size as u64 / stride) as usize);
    }
    cap.max(1)
}

/// Append a query string to a URL (`?q` or `&q`).
fn with_query(base: &str, query: &str) -> String {
    if base.contains('?') {
        format!("{base}&{query}")
    } else {
        format!("{base}?{query}")
    }
}

/// Scale the linear intensity-floor controls (slider + number) to a new data range.
fn set_floor_range(floor_hi: f64) {
    for id in ["floor", "floor-n"] {
        if let Some(inp) = by_id::<web_sys::HtmlInputElement>(id) {
            let _ = inp.set_attribute("max", &format!("{floor_hi:.0}"));
            let _ = inp.set_attribute("step", &format!("{:.0}", (floor_hi / 200.0).max(1.0)));
        }
    }
}

/// Reset the per-axis crop sliders to the full cube (thumbs, fills, readouts, and the filter).
fn reset_crops(gfx: &Rc<RefCell<Gfx>>) {
    for (axis, prefix) in [(0usize, "cmz"), (1, "cim"), (2, "crt")] {
        if let Some(lo) = by_id::<web_sys::HtmlInputElement>(&format!("{prefix}-lo")) {
            lo.set_value("0");
        }
        if let Some(hi) = by_id::<web_sys::HtmlInputElement>(&format!("{prefix}-hi")) {
            hi.set_value("1000");
        }
        crop_apply(gfx, axis, prefix, 0.0, 1000.0);
    }
}

/// Reset the Display control to 100% of the (new) resident pool.
fn reset_display(gfx: &Rc<RefCell<Gfx>>, n: usize) {
    {
        let mut g = gfx.borrow_mut();
        g.renderer.set_draw_count(u32::MAX);
        g.filter_dirty = true; // HUD displayed-count recomputed next frame
    }
    if let Some(s) = by_id::<web_sys::HtmlInputElement>("disp") {
        s.set_value("100");
    }
    if let Some(num) = by_id::<web_sys::HtmlInputElement>("disp-n") {
        num.set_value(&n.to_string());
        let _ = num.set_attribute("max", &n.to_string());
    }
}

/// Apply a freshly fetched load in place: rebuild the GPU buffer at the new capacity, retain the
/// CPU copy, re-apply auto-transfer + bounds, reset crops/floor/Display, and rebind the DOM. Used
/// by every reload (Load budget now; Focus/Back next). Returns the resident count.
fn apply_load(gfx: &Rc<RefCell<Gfx>>, meta: Option<MetaInfo>, pts: Vec<GpuPoint>) {
    let n = {
        let mut g = gfx.borrow_mut();
        let cap = pts.len().min(g.n_cap).max(1) as u32;
        let mut renderer = PointCloudRenderer::new(
            &g.device,
            &g.queue,
            g.config.format,
            DEPTH_FORMAT,
            cap,
            g.supports_compaction,
        );
        let n = renderer.append(&g.queue, &pts);
        g.renderer = renderer;
        let mut cpu = pts;
        cpu.truncate(cap as usize);
        g.cpu_points = cpu;
        g.data_gen = g.data_gen.wrapping_add(1); // invalidate any in-flight clustering Run
        g.vol_needs_grid = true; // the volume density grid is rebuilt from the new points
        g.params.color_mode = 0; // the fresh buffer has no cluster ids
        g.clustered = false;
        g.cluster_stats = None;
        g.cluster_params_json = None;
        g.cluster_run_params = None;
        set_save_ready(false); // ④ Save: results gone with the clustering
        g.cluster_idx = Vec::new();
        g.cluster_labels = Vec::new();
        g.cluster_sel = None;
        // Terminate a worker job in flight (a reload abandons it) so it doesn't keep computing with
        // the next Run queued behind it; the next Run rebuilds a fresh worker.
        if g.cluster_pending.take().is_some() {
            if let Some(w) = g.cluster_worker.take() {
                w.terminate();
            }
        }
        // Re-apply auto-transfer (exposure + floor range) and bounds from the new meta; keep the
        // user's colormap / point size / opacity / MS mask.
        if let Some(m) = &meta {
            apply_auto_transfer(&mut g.params, m);
            g.floor_hi = m.i_p99.max(1.0);
            g.axis_bounds = m.bounds;
            g.cycle_duration = m.cycle_duration;
            g.im_per_scan = m.im_per_scan;
        }
        // The new load defines the view: reset spatial crops + the intensity floor to "show all".
        for a in 0..3 {
            g.params.filter_min[a] = -1.0;
            g.params.filter_max[a] = 1.0;
        }
        g.params.filter_min[3] = 0.0;
        let bounds = g.axis_bounds;
        g.labels = create_axis_labels(bounds); // clears the container first, then rebuilds
        n
    };
    // DOM rebind (outside the borrow).
    if let Some(m) = &meta {
        set_floor_range(m.i_p99.max(1.0));
        if let Some(h) = &m.hist {
            draw_hist_backdrops(h);
        }
        if let Some(ih) = &m.i_hist {
            set_hist_svg("floor-hist", ih);
        }
        let ok = render_maps(m);
        gfx.borrow_mut().maps_ok = ok;
    }
    for id in ["floor", "floor-n"] {
        if let Some(inp) = by_id::<web_sys::HtmlInputElement>(id) {
            inp.set_value("0");
        }
    }
    reset_crops(gfx);
    reset_display(gfx, n);
    set_text("cl-readout", "—"); // clustering is invalidated by a new load
    set_checked("cl-color", true); // back to the default (applies once re-clustered)
    set_body_class("has-clusters", false);
    set_body_class("cluster-color", false);
    set_cluster_progress(None); // a load mid-run abandons it — clear the bar + Run button
    set_cluster_running(false);
    rebuild_window_overlay(gfx); // re-normalize the DIA windows to the new region
}

/// Build the region + budget query string for `/points` and `/meta`.
fn region_query(r: &Region, budget: usize) -> String {
    format!(
        "mz0={}&mz1={}&im0={}&im1={}&rt0={}&rt1={}&imin={}&n={budget}",
        r.mz.0, r.mz.1, r.im.0, r.im.1, r.rt.0, r.rt.1, r.imin,
    )
}

/// Which way `load_region` mutates the focus stack — applied only AFTER a successful load.
#[derive(Clone, Copy)]
enum StackOp {
    Push,
    Pop,
    Keep,
}

/// The focus-lens load primitive: re-fetch a region and rebuild the GPU buffer. `budget == None`
/// (the root stack entry) clamps to `n_cap` — the request is always a capped region query, never a
/// bare `/points`, so Back-to-root can't pull the server's full `--budget` and OOM the wasm heap.
/// The stack op is applied only after the load succeeds, so a debounced or failed load never desyncs
/// the stack from the displayed view. Debounced via `reloading`.
async fn load_region(gfx: Rc<RefCell<Gfx>>, region: Region, budget: Option<usize>, op: StackOp) {
    {
        let mut g = gfx.borrow_mut();
        if g.acq_playing {
            // A region load rebuilds the renderer; doing that under the running play-loop would
            // desync the two. Make the user stop playback first.
            show_status("stop live acquisition before focusing a region");
            return;
        }
        if g.reloading || g.focus_stack.is_empty() {
            return;
        }
        g.reloading = true;
    }
    refresh_controls(&gfx); // disable focus/back/load while a load is in flight
    let (base, n_cap) = {
        let g = gfx.borrow();
        (g.points_base.clone(), g.n_cap)
    };
    // Always cap n (None = the root entry) so Back-to-root can't fetch the server's full --budget
    // uncapped and OOM the wasm heap, mirroring the initial load.
    let q = region_query(&region, budget.unwrap_or(n_cap).min(n_cap));
    let purl = with_query(&base, &q);
    let murl = with_query(&meta_url(&base), &q);
    show_status("loading…");
    // Require BOTH meta and points: applying points without their matching meta would leave
    // axis_bounds stale while the buffer is normalized to the new region (breaks nested Focus).
    let meta = fetch_meta(&murl).await;
    let pts = fetch_points(&purl).await;
    gfx.borrow_mut().reloading = false;
    match (meta, pts) {
        (Ok(meta), Ok(pts)) if !pts.is_empty() => {
            apply_load(&gfx, Some(meta), pts);
            // Maintain the stack from the SERVER-canonical bounds (apply_load set axis_bounds), so
            // `stack.last()` always matches the displayed view.
            let canonical = gfx.borrow().axis_bounds.map(|b| Region {
                mz: b[0],
                im: b[1],
                rt: b[2],
                imin: region.imin,
            });
            {
                let mut g = gfx.borrow_mut();
                match op {
                    StackOp::Push => {
                        if let Some(c) = canonical {
                            g.focus_stack.push((c, budget));
                        }
                    }
                    StackOp::Pop => {
                        if g.focus_stack.len() > 1 {
                            g.focus_stack.pop();
                        }
                    }
                    StackOp::Keep => {
                        if let (Some(c), Some(top)) = (canonical, g.focus_stack.last_mut()) {
                            *top = (c, budget);
                        }
                    }
                }
            }
            show_status("");
        }
        (Ok(_), Ok(_)) => show_status("region is empty — nothing matched"),
        (Err(e), _) | (_, Err(e)) => show_status(&format!("load failed: {e}")),
    }
    refresh_controls(&gfx);
}

/// The current crop box + intensity floor, mapped from the normalized cube back to real units of
/// the current view — the region to focus on. `None` for the demo (no real bounds) or a degenerate
/// (zero/negative-width or non-finite) box that would divide-by-zero when re-normalized.
fn compute_focus_region(g: &Gfx) -> Option<Region> {
    let b = g.axis_bounds?;
    let p = &g.params;
    let (fmin, fmax, ms) = (p.filter_min, p.filter_max, p.ms_mask);
    let limit = (g.renderer.drawn() as usize).min(g.cpu_points.len());
    // Fit the box to the points that actually survive the active 4D filter (window + intensity
    // floor + MS) — NOT to the crop sliders. Otherwise points removed by the intensity floor still
    // "hold the box open" to the full extent. Survivors already lie inside any active spatial crop.
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    let mut n = 0u64;
    for pt in &g.cpu_points[..limit] {
        let pos = pt.pos;
        if pos[0] < fmin[0]
            || pos[0] > fmax[0]
            || pos[1] < fmin[1]
            || pos[1] > fmax[1]
            || pos[2] < fmin[2]
            || pos[2] > fmax[2]
            || pt.intensity < fmin[3]
        {
            continue;
        }
        let is_ms2 = (pt.flags & GpuPoint::MS2_FLAG) != 0;
        if !(if is_ms2 { ms & 0b10 != 0 } else { ms & 0b01 != 0 }) {
            continue;
        }
        for a in 0..3 {
            lo[a] = lo[a].min(pos[a]);
            hi[a] = hi[a].max(pos[a]);
        }
        n += 1;
    }
    if n == 0 {
        return None; // nothing visible to focus on
    }
    // Normalized survivor bbox -> real units, with a small margin (and a floor so a thin axis can't
    // collapse to zero width).
    let lerp = |rng: (f64, f64), norm: f32| rng.0 + ((norm as f64 + 1.0) * 0.5) * (rng.1 - rng.0);
    let mk = |axis: usize| {
        let m = ((hi[axis] - lo[axis]) * 0.02).max(0.01);
        let (a, c) = ((lo[axis] - m).max(-1.0), (hi[axis] + m).min(1.0));
        (lerp(b[axis], a), lerp(b[axis], c))
    };
    let (mz, im, rt) = (mk(0), mk(1), mk(2));
    let ok = |r: (f64, f64)| r.0.is_finite() && r.1.is_finite() && r.1 > r.0;
    if !ok(mz) || !ok(im) || !ok(rt) {
        return None;
    }
    Some(Region { mz, im, rt, imin: p.filter_min[3].max(0.0) })
}

/// Focus on the current crop+floor box: re-load it at full budget and push it on success.
fn do_focus(gfx: &Rc<RefCell<Gfx>>) {
    if gfx.borrow().reloading {
        return;
    }
    let region = match compute_focus_region(&gfx.borrow()) {
        Some(r) => r,
        None => {
            show_status("crop a region (or set a floor) before focusing");
            return;
        }
    };
    let budget = gfx.borrow().n_cap;
    wasm_bindgen_futures::spawn_local(load_region(gfx.clone(), region, Some(budget), StackOp::Push));
}

/// Re-load the region one level down the stack; pop only after it succeeds.
fn do_back(gfx: &Rc<RefCell<Gfx>>) {
    let target = {
        let g = gfx.borrow();
        if g.reloading || g.focus_stack.len() <= 1 {
            return;
        }
        g.focus_stack[g.focus_stack.len() - 2] // peek; load_region pops on success
    };
    wasm_bindgen_futures::spawn_local(load_region(gfx.clone(), target.0, target.1, StackOp::Pop));
}

/// Set/clear the `disabled` attribute on an element.
fn set_disabled(id: &str, disabled: bool) {
    if let Some(el) = by_id::<web_sys::HtmlElement>(id) {
        if disabled {
            let _ = el.set_attribute("disabled", "true");
        } else {
            let _ = el.remove_attribute("disabled");
        }
    }
}

/// Reflect the current load/focus state onto the controls: Focus/Load active only with a server +
/// a non-empty stack and no load in flight; Back active only when focused; show the focus depth.
fn refresh_controls(gfx: &Rc<RefCell<Gfx>>) {
    let (busy, has_server, depth, committed_imin) = {
        let g = gfx.borrow();
        (
            g.reloading,
            !g.points_base.is_empty(),
            g.focus_stack.len(),
            g.focus_stack.last().map(|(r, _)| r.imin).unwrap_or(0.0),
        )
    };
    let active = has_server && depth >= 1 && !busy;
    for id in ["focus-go", "load-go", "load-n"] {
        set_disabled(id, !active);
    }
    set_disabled("focus-back", busy || !has_server || depth <= 1);
    set_text(
        "focus-depth",
        &if depth > 1 { format!("L{}", depth - 1) } else { String::new() },
    );
    // When the current view was loaded with an intensity cutoff, the low-intensity points aren't
    // resident — make that explicit so a floor parked at 0 isn't mistaken for "all points".
    set_text(
        "floor-src",
        &if committed_imin > 0.0 {
            format!("· baked ≥ {committed_imin:.0} (Back to lower)")
        } else {
            String::new()
        },
    );
}

/// Apply a clustering-algorithm dropdown value: set the engine (wasm vs Python) + method, and reveal
/// that algorithm's parameter rows.
fn apply_cluster_algo(gfx: &Rc<RefCell<Gfx>>, value: &str) {
    let (python, method) = match value {
        "sklearn" => (true, ClusterMethod::Dbscan),
        "hdbscan" => (true, ClusterMethod::Hdbscan),
        _ => (false, ClusterMethod::Dbscan), // "wasm"
    };
    {
        let mut g = gfx.borrow_mut();
        g.use_python_cluster = python;
        g.cluster_method = method;
    }
    set_body_class("algo-hdb", method == ClusterMethod::Hdbscan);
}

/// Flip the Run/Stop button: while a (worker) clustering is in flight it becomes a red "Stop".
fn set_cluster_running(running: bool) {
    set_text("cl-run", if running { "⬡ Stop clustering" } else { "⬡ Run clustering" });
    set_class("cl-run", "stop", running);
}

/// Cancel an in-flight worker clustering: terminate the worker (taking it so the next Run rebuilds a
/// fresh one), drop the pending job, and reset the UI.
fn stop_clustering(gfx: &Rc<RefCell<Gfx>>) {
    {
        let mut g = gfx.borrow_mut();
        if let Some(w) = g.cluster_worker.take() {
            w.terminate();
        }
        g.cluster_pending = None;
    }
    set_cluster_progress(None);
    set_cluster_running(false);
    show_status("clustering stopped");
}

/// Show the clustering progress bar at `frac` (0..1), or hide it with `None`.
fn set_cluster_progress(frac: Option<f32>) {
    if let Some(el) = by_id::<web_sys::HtmlElement>("cl-prog") {
        let _ = el.style().set_property("display", if frac.is_some() { "block" } else { "none" });
    }
    if let Some(f) = frac {
        if let Some(bar) = by_id::<web_sys::HtmlElement>("cl-bar") {
            let _ = bar.style().set_property("width", &format!("{:.0}%", f.clamp(0.0, 1.0) * 100.0));
        }
    }
}

/// Toggle the `volmode` body class (reveals the volume-only controls).
fn set_volmode(on: bool) {
    if let Some(body) = document().and_then(|d| d.body()) {
        let _ = body.class_list().toggle_with_force("volmode", on);
    }
}

/// Wire the Points/Volume view-mode toggle + volume controls (style, steps). Disables Volume when
/// the GPU can't host the 3D density texture.
fn bind_volume(gfx: &Rc<RefCell<Gfx>>) {
    // Force the View toggle + style to the code defaults (Points / Composite), so a browser-restored
    // radio can't desync from view_mode (starts Points) / vol_style (starts Composite) — otherwise
    // the toggle shows Volume/MIP while the code renders Points/Composite.
    set_checked("v-pts", true);
    set_checked("v-vol", false);
    set_checked("vs-comp", true);
    set_checked("vs-mip", false);
    set_volmode(false);
    if gfx.borrow().volume.is_none() {
        set_disabled("v-vol", true);
        return; // leave Points selected; volume unsupported on this GPU
    }
    on_toggle("v-pts", gfx, |g, on| {
        if on {
            g.view_mode = ViewMode::Points;
            set_volmode(false);
        }
    });
    on_toggle("v-vol", gfx, |g, on| {
        if on {
            g.view_mode = ViewMode::Volume;
            g.vol_needs_grid = true;
            set_volmode(true);
            show_status("building volume…");
        }
    });
    on_toggle("vs-comp", gfx, |g, on| if on { g.vol_style = 0 });
    on_toggle("vs-mip", gfx, |g, on| if on { g.vol_style = 1 });
    bind_value(gfx, "vsteps", "vsteps-n", 0, |g, v| g.vol_steps = (v as u32).max(1));
    // The intensity floor feeds the volume deposit; rebuild when it SETTLES (slider release /
    // number commit), not on every drag tick — a rebuild is O(resident) and would lag a drag.
    for id in ["floor", "floor-n"] {
        if let Some(inp) = by_id::<web_sys::HtmlInputElement>(id) {
            let gfx = gfx.clone();
            add_listener(inp.as_ref(), "change", move |_e: web_sys::Event| {
                let mut g = gfx.borrow_mut();
                // Only flag in Volume mode; entering Volume already rebuilds, so a Points-mode
                // floor change needs no flag.
                if g.view_mode == ViewMode::Volume {
                    g.vol_needs_grid = true;
                }
            });
        }
    }
}

/// Revert cluster colouring (a filter/load changed the set, so the labels are stale). Cheap — just
/// flips `color_mode` back to intensity; the stale `_pad[0]` ids stay unused until the next Run.
fn invalidate_clusters_mut(g: &mut Gfx) {
    // Abandon any in-flight worker job AND terminate the worker, so it doesn't keep computing a
    // now-stale result with the next Run queued behind it. The next Run rebuilds a fresh worker.
    if g.cluster_pending.take().is_some() {
        if let Some(w) = g.cluster_worker.take() {
            w.terminate();
        }
    }
    set_cluster_progress(None);
    set_cluster_running(false);
    if g.clustered {
        g.params.color_mode = 0;
        g.clustered = false;
        g.cluster_stats = None;
        g.cluster_params_json = None;
        g.cluster_run_params = None;
        set_save_ready(false); // ④ Save: results gone with the clustering
        g.cluster_idx = Vec::new();
        g.cluster_labels = Vec::new();
        g.cluster_sel = None;
        set_text("cl-readout", "—");
        set_checked("cl-color", false);
        set_body_class("has-clusters", false);
        set_body_class("cluster-color", false);
    }
}

/// Run DBSCAN on the filtered resident points, write cluster ids into the GPU buffer, and colour by
/// cluster. Synchronous (blocks the main thread) — Focus first if the filtered set is large.
fn run_clustering(gfx: &Rc<RefCell<Gfx>>) {
    if gfx.borrow().acq_playing {
        show_status("stop live acquisition before clustering");
        return;
    }
    if gfx.borrow().reloading {
        show_status("loading — try clustering again in a moment");
        return;
    }
    if gfx.borrow().cluster_pending.is_some() {
        show_status("clustering already running…");
        return;
    }
    // MS1 (precursors) and MS2 (fragments) live in different m/z spaces — clustering them together is
    // meaningless. Require a single MS level.
    if gfx.borrow().params.ms_mask == 0b11 {
        show_status("pick MS1 or MS2 to cluster — clustering both levels together isn't meaningful");
        return;
    }
    // Gather the filtered survivors (spatial crop + intensity floor + MS) and their cube positions.
    let (idx, positions, eps, min_pts, gen, run_params) = {
        let g = gfx.borrow();
        let p = &g.params;
        let (fmin, fmax, ms) = (p.filter_min, p.filter_max, p.ms_mask);
        let mut idx: Vec<usize> = Vec::new();
        let mut positions: Vec<[f32; 3]> = Vec::new();
        for (i, pt) in g.cpu_points.iter().enumerate() {
            let pos = pt.pos;
            if pos[0] < fmin[0]
                || pos[0] > fmax[0]
                || pos[1] < fmin[1]
                || pos[1] > fmax[1]
                || pos[2] < fmin[2]
                || pos[2] > fmax[2]
                || pt.intensity < fmin[3]
            {
                continue;
            }
            let is_ms2 = pt.flags & GpuPoint::MS2_FLAG != 0;
            if !(if is_ms2 { ms & 0b10 != 0 } else { ms & 0b01 != 0 }) {
                continue;
            }
            idx.push(i);
            positions.push(pos);
        }
        // Per-axis equi-distancing (the shared, unit-tested helper in tims_viewer::cluster): calibrated
        // to CLUSTER_EPS_REF so the live eps scales the reach proportionally, and — crucially —
        // region-INDEPENDENT, so eps spans a fixed physical reach on every axis (cycles / scans /
        // peak-widths) regardless of focus crop. (The metric only; labels/stats use unscaled points.)
        let eps = g.cluster_eps;
        let scales = match g.axis_bounds {
            Some(bounds) => cluster_axis_scales(&ScaleInputs {
                bounds,
                eps_ref: CLUSTER_EPS_REF,
                cycle_duration: g.cycle_duration,
                im_per_scan: g.im_per_scan,
                mz_resolution: MZ_RESOLUTION,
                rt_cycles: g.cluster_rt_cycles,
                im_scans: g.cluster_im_scans,
                mz_peak_widths: g.cluster_mz_peak_widths,
            }),
            None => AxisScales { mz: 1.0, im: 1.0, rt: 1.0 },
        };
        for p in positions.iter_mut() {
            p[0] *= scales.mz;
            p[1] *= scales.im;
            p[2] *= scales.rt;
        }
        // Snapshot the parameters used (for the export JSON), so a slider edit mid-run can't skew it.
        let run_params = ClusterRunParams {
            dataset_id: g.dataset_id, // resolved/clamped id, not the raw URL value
            method: g.cluster_method,
            python: g.use_python_cluster,
            eps,
            min_pts: g.cluster_min_pts,
            min_cluster_size: g.cluster_min_cluster_size,
            hdb_min_samples: g.cluster_hdb_min_samples,
            selection_eps: g.cluster_selection_eps,
            rt_cycles: g.cluster_rt_cycles,
            im_scans: g.cluster_im_scans,
            mz_peak_widths: g.cluster_mz_peak_widths,
            ms_mask: g.params.ms_mask,
            floor: g.params.filter_min[3],
            region: g.axis_bounds,
            cycle_duration: g.cycle_duration,
            im_per_scan: g.im_per_scan,
        };
        (idx, positions, eps, g.cluster_min_pts, g.data_gen, run_params)
    };
    if idx.is_empty() {
        show_status("nothing to cluster — adjust the filter");
        return;
    }
    if idx.len() > CLUSTER_CAP {
        show_status(&format!(
            "{} points — Focus a region first (cap {})",
            group(idx.len()),
            group(CLUSTER_CAP)
        ));
        return;
    }
    gfx.borrow_mut().cluster_run_params = Some(run_params); // before dispatch, for the export JSON

    // Python backend (when the "Python (sklearn)" toggle is on — auto-enabled by the startup probe):
    // POST the scaled points to the service and colour by its labels. Else the wasm path below.
    if gfx.borrow().use_python_cluster {
        let svc = cluster_service_url();
        let flat: Vec<f32> = positions.iter().flatten().copied().collect();
        let (job, query) = {
            let mut g = gfx.borrow_mut();
            g.next_cluster_job = g.next_cluster_job.wrapping_add(1);
            let q = match g.cluster_method {
                ClusterMethod::Hdbscan => format!(
                    "method=hdbscan&mcs={}&ms={}&cse={}",
                    g.cluster_min_cluster_size, g.cluster_hdb_min_samples, g.cluster_selection_eps
                ),
                ClusterMethod::Dbscan => format!("method=dbscan&eps={eps}&min={min_pts}"),
            };
            g.cluster_pending = Some((idx, gen, g.next_cluster_job));
            (g.next_cluster_job, q)
        };
        set_cluster_running(true);
        show_status("clustering (python)…");
        let gfx2 = gfx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let result = fetch_cluster(&svc, &flat, &query).await;
            // Only consume the reply if it's still THIS job — a Stop / invalidate / newer Run
            // supersedes it (the un-aborted fetch would otherwise steal the new pending job).
            let mut g = gfx2.borrow_mut();
            if !matches!(&g.cluster_pending, Some((_, _, pj)) if *pj == job) {
                return;
            }
            let (idx, gen, _) = g.cluster_pending.take().unwrap();
            drop(g);
            match result {
                Ok(labels) if labels.len() == idx.len() => {
                    let k = labels.iter().copied().max().map_or(0, |m| (m + 1).max(0) as usize);
                    finish_clustering(&gfx2, idx, labels, k, gen);
                }
                Ok(_) => {
                    set_cluster_running(false);
                    show_status("python clustering: label/point count mismatch");
                }
                Err(e) => {
                    set_cluster_running(false);
                    show_status(&format!("python clustering failed: {e}"));
                }
            }
        });
        return;
    }

    show_status("clustering…");
    // Small inputs cluster in well under a second, so run them on the main thread (deferred one
    // macrotask so the status paints): instant, with none of the worker's one-time wasm spin-up
    // latency. Only large inputs — where DBSCAN would freeze the tab for seconds — go to the worker,
    // whose init cost is then negligible against the run and whose progress bar actually has time to
    // move.
    if idx.len() > WORKER_THRESHOLD {
        if let Some(worker) = ensure_cluster_worker(gfx) {
            let flat: Vec<f32> = positions.iter().flatten().copied().collect();
            let job = {
                let mut g = gfx.borrow_mut();
                g.next_cluster_job = g.next_cluster_job.wrapping_add(1);
                g.next_cluster_job
            };
            if post_cluster_job(&worker, &flat, eps, min_pts, job) {
                gfx.borrow_mut().cluster_pending = Some((idx, gen, job));
                set_cluster_progress(Some(0.0)); // worker reports ticks back; bar advances live
                set_cluster_running(true); // button becomes "Stop clustering"
                return;
            }
        }
    }
    // Main thread (small input, or worker unavailable): defer one macrotask so the status paints,
    // then run the (brief) blocking DBSCAN.
    let gfx = gfx.clone();
    defer(move || {
        let (labels, k) = dbscan(&positions, eps, min_pts);
        finish_clustering(&gfx, idx, labels, k, gen);
    });
}

/// JS module source for the DBSCAN worker — re-imports this same wasm (URLs injected) and runs
/// `cluster_dbscan_flat` per posted job. Top-level await holds the message queue until wasm is ready.
const CLUSTER_WORKER_SRC: &str = r#"
import init, { cluster_dbscan_flat } from '__GLUE__';
await init({ module_or_path: '__WASM__' });
self.onmessage = (e) => {
  const { flat, eps, min_pts, job } = e.data;
  const onProgress = (visited, total) => self.postMessage({ progress: visited / total, job });
  const labels = cluster_dbscan_flat(new Float32Array(flat), min_pts, eps, onProgress);
  self.postMessage({ labels, job }, [labels.buffer]);
};
"#;

/// Return the persistent DBSCAN worker, creating it on first use. `None` (→ main-thread fallback) if
/// workers are unavailable or the wasm/glue URLs can't be discovered.
fn ensure_cluster_worker(gfx: &Rc<RefCell<Gfx>>) -> Option<web_sys::Worker> {
    {
        let g = gfx.borrow();
        if g.cluster_worker_failed {
            return None; // a prior runtime failure latched — stay on the main thread
        }
        if let Some(w) = g.cluster_worker.clone() {
            return Some(w);
        }
    }
    let w = create_cluster_worker(gfx)?;
    gfx.borrow_mut().cluster_worker = Some(w.clone());
    Some(w)
}

/// Build the DBSCAN worker from a Blob module that re-imports the Trunk-built wasm (URLs read from
/// the page's preload links, so Trunk's content hashing is handled).
fn create_cluster_worker(gfx: &Rc<RefCell<Gfx>>) -> Option<web_sys::Worker> {
    let doc = document()?;
    // The glue JS + wasm URLs from Trunk's preload links (absolute via the element `href` property).
    let link_href = |sel: &str| -> Option<String> {
        doc.query_selector(sel)
            .ok()
            .flatten()
            .and_then(|e| e.dyn_into::<web_sys::HtmlLinkElement>().ok())
            .map(|l| l.href())
    };
    let glue = link_href("link[rel='modulepreload']")?;
    let wasm = link_href("link[rel='preload'][as='fetch']")?;
    let src = CLUSTER_WORKER_SRC.replace("__GLUE__", &glue).replace("__WASM__", &wasm);

    let parts = js_sys::Array::of1(&JsValue::from_str(&src));
    let bag = web_sys::BlobPropertyBag::new();
    bag.set_type("text/javascript"); // module workers require a JS MIME on the script
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &bag).ok()?;
    let url = web_sys::Url::create_object_url_with_blob(&blob).ok()?;
    let opts = web_sys::WorkerOptions::new();
    opts.set_type(web_sys::WorkerType::Module);
    let worker = web_sys::Worker::new_with_options(&url, &opts).ok()?;
    let _ = web_sys::Url::revoke_object_url(&url); // the worker has captured the script

    // Reply handler: only finish the reply whose job id matches the pending one (a superseded job's
    // late reply carries a stale id and is ignored).
    let gfx_msg = gfx.clone();
    let cb = wasm_bindgen::closure::Closure::<dyn FnMut(web_sys::MessageEvent)>::new(
        move |e: web_sys::MessageEvent| {
            let data = e.data();
            let get = |k: &str| js_sys::Reflect::get(&data, &JsValue::from_str(k)).ok();
            let reply_job = get("job").and_then(|v| v.as_f64()).map(|f| f as u64);
            let job_matches = |g: &Gfx| {
                matches!((&g.cluster_pending, reply_job), (Some((_, _, pj)), Some(rj)) if *pj == rj)
            };
            // Progress tick (no labels): advance the bar if it's for the current job.
            if let Some(p) = get("progress").and_then(|v| v.as_f64()) {
                if job_matches(&gfx_msg.borrow()) {
                    set_cluster_progress(Some(p as f32));
                }
                return;
            }
            let Some(labels) = get("labels")
                .and_then(|v| v.dyn_into::<js_sys::Int32Array>().ok())
                .map(|a| a.to_vec())
            else {
                return;
            };
            if !job_matches(&gfx_msg.borrow()) {
                return; // no pending / superseded job — drop this reply
            }
            if let Some((idx, gen, _)) = gfx_msg.borrow_mut().cluster_pending.take() {
                if labels.len() != idx.len() {
                    return; // malformed reply — never index past the survivor set
                }
                let k = labels.iter().copied().max().map_or(0, |m| (m + 1).max(0) as usize);
                finish_clustering(&gfx_msg, idx, labels, k, gen);
            }
        },
    );
    worker.set_onmessage(Some(cb.as_ref().unchecked_ref()));
    cb.forget(); // the worker lives for the app's lifetime

    // A runtime failure (e.g. the worker's wasm can't load, or a message can't deserialize) latches
    // `cluster_worker_failed` so every later Run uses the main thread instead of rebuilding a broken
    // worker; also release any stuck pending job.
    for ev in ["error", "messageerror"] {
        let gfx_err = gfx.clone();
        add_listener(worker.as_ref(), ev, move |_e: web_sys::Event| {
            let mut g = gfx_err.borrow_mut();
            g.cluster_pending = None;
            g.cluster_worker = None;
            g.cluster_worker_failed = true;
            drop(g);
            set_cluster_progress(None);
            set_cluster_running(false);
            show_status("clustering worker unavailable — runs on the main thread");
        });
    }
    Some(worker)
}

/// Post a clustering job (the flat positions buffer is transferred, not copied) to the worker.
fn post_cluster_job(worker: &web_sys::Worker, flat: &[f32], eps: f32, min_pts: usize, job: u64) -> bool {
    let arr = js_sys::Float32Array::from(flat);
    let msg = js_sys::Object::new();
    let set = |k: &str, v: &JsValue| js_sys::Reflect::set(&msg, &JsValue::from_str(k), v).is_ok();
    if !set("flat", &arr.buffer()) || !set("eps", &JsValue::from_f64(eps as f64)) {
        return false;
    }
    if !set("min_pts", &JsValue::from_f64(min_pts as f64)) || !set("job", &JsValue::from_f64(job as f64)) {
        return false;
    }
    let transfer = js_sys::Array::of1(&arr.buffer());
    worker.post_message_with_transfer(&msg, &transfer).is_ok()
}

/// Re-upload the resident `cpu_points` to the GPU (the only path to refresh the cluster ids).
fn reupload_points(g: &mut Gfx) {
    let pts = std::mem::take(&mut g.cpu_points);
    g.renderer.reset();
    g.renderer.append(&g.queue, &pts);
    g.cpu_points = pts;
}

/// Write DBSCAN labels into the GPU buffer + recolor by cluster, compute stats, then update the UI.
fn finish_clustering(gfx: &Rc<RefCell<Gfx>>, idx: Vec<usize>, labels: Vec<i32>, k: usize, gen: u64) {
    if labels.len() != idx.len() {
        show_status(""); // malformed result — never index past the survivor set
        return;
    }
    let noise = labels.iter().filter(|&&l| l < 0).count();
    let n_in = idx.len();
    {
        let mut gb = gfx.borrow_mut();
        let g: &mut Gfx = &mut gb; // &mut Gfx so renderer/queue/cpu_points field-split
        if g.data_gen != gen {
            show_status(""); // the point set changed under us — drop this stale result
            return;
        }
        let stats = compute_cluster_stats(&idx, &labels, k, &g.cpu_points, g.axis_bounds);
        for pt in g.cpu_points.iter_mut() {
            pt._pad[0] = GpuPoint::NO_CLUSTER;
        }
        for (j, &i) in idx.iter().enumerate() {
            if labels[j] >= 0 {
                g.cpu_points[i]._pad[0] = labels[j] as u32;
            }
        }
        reupload_points(g);
        g.params.color_mode = if g.hide_noise { 2 } else { 1 };
        g.clustered = true;
        g.cluster_params_json = g.cluster_run_params.map(|p| cluster_params_json(&p, k, noise, n_in));
        g.cluster_stats = Some(stats);
        g.cluster_idx = idx;
        g.cluster_labels = labels;
        g.cluster_sel = None;
        g.view_mode = ViewMode::Points; // cluster colour is a point concept
        g.filter_dirty = true; // refresh the displayed-count HUD
    }
    set_checked("v-pts", true);
    set_checked("v-vol", false);
    set_volmode(false);
    set_checked("cl-color", true);
    set_text("cl-readout", &format!("{k} clusters · {} noise · {} pts", group(noise), group(n_in)));
    set_body_class("cluster-color", true); // cluster coloring active -> hide the intensity colorbar
    set_cluster_progress(None); // done — hide the progress bar
    set_cluster_running(false); // restore the Run button
    set_save_ready(true); // ④ Save: results are now exportable
    update_cluster_panel(gfx); // we ran from the Cluster tab, so this shows + renders the panel
    show_status("");
}

/// Aggregate per-cluster stats from the DBSCAN labels (intensity weighted by `weight` so it's
/// downsample-independent; extents in real units via `axis_bounds`).
fn compute_cluster_stats(
    idx: &[usize],
    labels: &[i32],
    k: usize,
    pts: &[GpuPoint],
    bounds: Option<[(f64, f64); 3]>,
) -> ClusterStats {
    let mut sizes = vec![0f64; k];
    let mut int_sum = vec![0f64; k];
    let mut lo = vec![[f32::INFINITY; 3]; k];
    let mut hi = vec![[f32::NEG_INFINITY; 3]; k];
    let (mut signal_pts, mut noise_pts) = (0u64, 0u64);
    let (mut signal_int, mut noise_int) = (0f64, 0f64);
    for (j, &l) in labels.iter().enumerate() {
        let pt = &pts[idx[j]];
        let contrib = pt.intensity as f64 * pt.weight as f64; // widen before multiply
        if l < 0 {
            noise_pts += 1;
            noise_int += contrib;
        } else {
            let c = l as usize;
            sizes[c] += 1.0;
            int_sum[c] += contrib;
            signal_pts += 1;
            signal_int += contrib;
            for a in 0..3 {
                lo[c][a] = lo[c][a].min(pt.pos[a]);
                hi[c][a] = hi[c][a].max(pt.pos[a]);
            }
        }
    }
    // Normalized [-1,1] span -> real-unit extent on each axis.
    let extent = |a: usize| -> Vec<f64> {
        (0..k)
            .map(|c| {
                if sizes[c] == 0.0 {
                    return 0.0;
                }
                let span = (hi[c][a] - lo[c][a]).max(0.0) as f64;
                match bounds {
                    Some(b) => span * 0.5 * (b[a].1 - b[a].0).abs(),
                    None => span,
                }
            })
            .collect()
    };
    ClusterStats {
        k,
        signal_pts,
        noise_pts,
        signal_int,
        noise_int,
        mz_extent: extent(0),
        im_extent: extent(1),
        rt_extent: extent(2),
        sizes,
        int_sum,
    }
}

/// CSS color matching the shader's per-cluster hue (`hsv2rgb(fract(id*0.618), 0.65, 1.0)`).
fn cluster_css_color(id: usize) -> String {
    let hue = (id as f64 * 0.618_033_988_75).fract() * 360.0;
    format!("hsl({hue:.0}, 100%, 68%)")
}

/// Fill the cluster stats panel: summary table, four per-cluster histograms, and the clickable
/// cluster list (top clusters by size; `sel` marks the isolated one).
fn render_cluster_panel(s: &ClusterStats, sel: Option<i32>) {
    let total_pts = s.signal_pts + s.noise_pts;
    let total_int = s.signal_int + s.noise_int;
    set_text("cs-pts-t", &group_short(total_pts as usize));
    set_text("cs-pts-c", &group_short(s.signal_pts as usize));
    set_text("cs-pts-n", &group_short(s.noise_pts as usize));
    set_text("cs-int-t", &fmt_count(total_int));
    set_text("cs-int-c", &fmt_count(s.signal_int));
    set_text("cs-int-n", &fmt_count(s.noise_int));
    set_text("cs-k", &s.k.to_string());
    let ratio = |sig: f64, noise: f64| {
        if noise > 0.0 {
            format!("{:.1}×", sig / noise)
        } else {
            "∞".into()
        }
    };
    set_text("cs-ratio", &ratio(s.signal_pts as f64, s.noise_pts as f64));
    set_text("cs-iratio", &ratio(s.signal_int, s.noise_int));
    set_text("cs-sub", &format!("{} clusters · {} pts", s.k, group_short(total_pts as usize)));
    set_hist_svg("csh-size", &hist_bins(&s.sizes, 24));
    set_hist_svg("csh-mz", &hist_bins(&s.mz_extent, 24));
    set_hist_svg("csh-im", &hist_bins(&s.im_extent, 24));
    set_hist_svg("csh-rt", &hist_bins(&s.rt_extent, 24));
    // X-axis right edge = the max extent (the bins span [0, max]); show it in real units so the
    // spread is physically meaningful.
    let max_of = |v: &[f64]| v.iter().cloned().fold(0.0f64, f64::max);
    set_text("csx-size", &format!("{} pts", max_of(&s.sizes) as u64));
    set_text("csx-mz", &format!("{:.3} m/z", max_of(&s.mz_extent)));
    set_text("csx-im", &format!("{:.4} 1/K₀", max_of(&s.im_extent)));
    set_text("csx-rt", &format!("{:.1} s", max_of(&s.rt_extent)));

    // Clickable cluster list — top 50 by size, plus a "Show all" row when isolated.
    let mut order: Vec<usize> = (0..s.k).collect();
    order.sort_by(|&a, &b| s.sizes[b].total_cmp(&s.sizes[a]));
    let mut html = String::new();
    if sel.is_some() {
        html.push_str("<button class=\"cs-row show-all\" data-cl=\"-1\">← Show all clusters</button>");
    }
    for &c in order.iter().take(50) {
        let active = if sel == Some(c as i32) { " active" } else { "" };
        html.push_str(&format!(
            "<button class=\"cs-row{active}\" data-cl=\"{c}\"><span class=\"cs-dot\" style=\"background:{}\"></span>#{c}<span class=\"cs-meta\">{} · {}</span></button>",
            cluster_css_color(c),
            fmt_count(s.sizes[c]),
            fmt_count(s.int_sum[c]),
        ));
    }
    if s.k > 50 {
        html.push_str(&format!("<div class=\"cs-more\">+{} more</div>", s.k - 50));
    }
    set_html("cs-list", &html);
}

/// Bin non-negative finite values into `n` equal-width bins over `[0, max]`. Everything collapses to
/// bin 0 when every value is 0.
fn hist_bins(values: &[f64], n: usize) -> Vec<u32> {
    let n = n.max(1);
    let mut h = vec![0u32; n];
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite() && *v >= 0.0).collect();
    let max = finite.iter().copied().fold(0.0f64, f64::max);
    if max <= 0.0 {
        h[0] = finite.len() as u32;
        return h;
    }
    for &v in &finite {
        // equal-width: scale by n then clamp to n-1 (vs *(n-1), which leaves the last bin only for
        // exact maxima).
        let b = ((v / max * n as f64) as usize).min(n - 1);
        h[b] += 1;
    }
    h
}

/// Compact display of a (possibly huge / non-finite) f64 sum — avoids the silent `as usize`
/// saturation when intensity totals are large.
fn fmt_count(x: f64) -> String {
    if !x.is_finite() {
        return "—".into();
    }
    if x >= 1e9 {
        format!("{:.1}G", x / 1e9)
    } else if x >= 1e6 {
        format!("{:.1}M", x / 1e6)
    } else if x >= 1e4 {
        format!("{:.0}k", x / 1e3)
    } else {
        format!("{:.0}", x)
    }
}

/// Toggle a class on `<body>`.
fn set_body_class(class: &str, on: bool) {
    if let Some(body) = document().and_then(|d| d.body()) {
        let _ = body.class_list().toggle_with_force(class, on);
    }
}

/// Run a closure on the next macrotask (after a browser paint) via `setTimeout(0)`.
fn defer<F: FnOnce() + 'static>(f: F) {
    let cb = wasm_bindgen::closure::Closure::once_into_js(f);
    if let Some(w) = web_sys::window() {
        let _ = w.set_timeout_with_callback_and_timeout_and_arguments_0(cb.unchecked_ref(), 0);
    }
}

/// Wire the Cluster tab: eps / min-pts, Run, and the Color-by-cluster toggle.
fn bind_cluster(gfx: &Rc<RefCell<Gfx>>) {
    bind_value(gfx, "cl-eps", "cl-eps-n", 3, |g, v| g.cluster_eps = v as f32);
    bind_value(gfx, "cl-min", "cl-min-n", 0, |g, v| g.cluster_min_pts = (v as usize).max(2));
    bind_value(gfx, "cl-rtc", "cl-rtc-n", 0, |g, v| g.cluster_rt_cycles = v.round().max(1.0));
    bind_value(gfx, "cl-ims", "cl-ims-n", 1, |g, v| g.cluster_im_scans = v.max(0.1));
    bind_value(gfx, "cl-mzw", "cl-mzw-n", 1, |g, v| g.cluster_mz_peak_widths = v.max(0.1));
    // HDBSCAN parameters (Python only).
    bind_value(gfx, "cl-mcs", "cl-mcs-n", 0, |g, v| g.cluster_min_cluster_size = (v as usize).max(2));
    bind_value(gfx, "cl-hms", "cl-hms-n", 0, |g, v| g.cluster_hdb_min_samples = v.max(0.0) as usize);
    bind_value(gfx, "cl-cse", "cl-cse-n", 3, |g, v| g.cluster_selection_eps = v.max(0.0));
    // Algorithm dropdown: maps to (engine, method) + reveals that algorithm's params.
    if let Some(sel) = by_id::<web_sys::HtmlSelectElement>("cl-algo") {
        let (gfx, el) = (gfx.clone(), sel.clone());
        add_listener(sel.as_ref(), "change", move |_e: web_sys::Event| {
            gfx.borrow_mut().cluster_algo_user_set = true; // the probe must not override a manual pick
            apply_cluster_algo(&gfx, &el.value());
        });
    }
    if let Some(btn) = by_id::<web_sys::HtmlElement>("cl-run") {
        let gfx = gfx.clone();
        add_listener(btn.as_ref(), "click", move |_e: web_sys::Event| {
            // Run, or Stop if a worker clustering is in flight.
            if gfx.borrow().cluster_pending.is_some() {
                stop_clustering(&gfx);
            } else {
                run_clustering(&gfx);
            }
        });
    }
    on_toggle("cl-color", gfx, |g, on| {
        if g.clustered {
            g.params.color_mode = if on {
                if g.hide_noise { 2 } else { 1 }
            } else {
                0
            };
            set_body_class("cluster-color", on); // toggles the intensity colorbar back on when off
        }
    });
    on_toggle("hide-noise", gfx, |g, on| {
        g.hide_noise = on;
        // Only meaningful while cluster-colouring; flips between color_mode 1 (show) and 2 (hide).
        if g.clustered && g.params.color_mode != 0 {
            g.params.color_mode = if on { 2 } else { 1 };
        }
    });
    // Click a row in the cluster list to isolate it (data-cl="-1" = show all).
    if let Some(list) = by_id::<web_sys::HtmlElement>("cs-list") {
        let gfx = gfx.clone();
        add_listener(list.as_ref(), "click", move |e: web_sys::Event| {
            let Some(t) = e.target().and_then(|t| t.dyn_into::<web_sys::Element>().ok()) else {
                return;
            };
            if let Ok(Some(row)) = t.closest(".cs-row") {
                if let Some(cl) = row.get_attribute("data-cl").and_then(|s| s.parse::<i32>().ok()) {
                    isolate_cluster(&gfx, if cl < 0 { None } else { Some(cl) });
                }
            }
        });
    }
    if let Some(btn) = by_id::<web_sys::HtmlElement>("cl-export") {
        let gfx = gfx.clone();
        add_listener(btn.as_ref(), "click", move |_e: web_sys::Event| {
            export_clusters(&gfx);
        });
    }
    if let Some(btn) = by_id::<web_sys::HtmlElement>("cfg-export") {
        let gfx = gfx.clone();
        add_listener(btn.as_ref(), "click", move |_e: web_sys::Event| {
            export_config(&gfx);
        });
    }
}

/// Wire the "Show DIA windows" toggle + the per-group legend (enabled once `/windows` returns groups).
fn bind_windows(gfx: &Rc<RefCell<Gfx>>) {
    if let Some(inp) = by_id::<web_sys::HtmlInputElement>("show-windows") {
        let (gfx, el) = (gfx.clone(), inp.clone());
        add_listener(inp.as_ref(), "change", move |_e: web_sys::Event| {
            let on = el.checked();
            gfx.borrow_mut().show_windows = on;
            set_body_class("show-windows", on); // reveals the group legend
            if on {
                if gfx.borrow().window_rects.is_empty() {
                    show_status("no DIA windows for this run (not DIA, or restart the point server)");
                } else {
                    render_window_legend(&gfx);
                }
            }
        });
    }
    // Click a group chip to toggle that group's windows.
    if let Some(list) = by_id::<web_sys::HtmlElement>("win-groups") {
        let gfx = gfx.clone();
        add_listener(list.as_ref(), "click", move |e: web_sys::Event| {
            let Some(t) = e.target().and_then(|t| t.dyn_into::<web_sys::Element>().ok()) else {
                return;
            };
            if let Ok(Some(chip)) = t.closest(".wg") {
                if let Some(g) = chip.get_attribute("data-g").and_then(|s| s.parse::<u32>().ok()) {
                    if (1..=32).contains(&g) {
                        gfx.borrow_mut().window_group_mask ^= 1u32 << (g - 1);
                    }
                    rebuild_window_overlay(&gfx);
                    render_window_legend(&gfx);
                }
            }
        });
    }
}

/// Wire the Focus / Back buttons; control enable/disable is driven by `refresh_controls`.
fn bind_focus(gfx: &Rc<RefCell<Gfx>>) {
    if let Some(btn) = by_id::<web_sys::HtmlElement>("focus-go") {
        let gfx = gfx.clone();
        add_listener(btn.as_ref(), "click", move |_e: web_sys::Event| do_focus(&gfx));
    }
    if let Some(btn) = by_id::<web_sys::HtmlElement>("focus-back") {
        let gfx = gfx.clone();
        add_listener(btn.as_ref(), "click", move |_e: web_sys::Event| do_back(&gfx));
    }
    refresh_controls(gfx);
}

/// Wire the Load control: show N_cap, initialize to the resident count, and re-fetch the run at the
/// typed budget (clamped to N_cap) on click / Enter. Disabled in the demo fallback (no server).
fn bind_load(gfx: &Rc<RefCell<Gfx>>) {
    let (Some(num), Some(btn)) = (
        by_id::<web_sys::HtmlInputElement>("load-n"),
        by_id::<web_sys::HtmlElement>("load-go"),
    ) else {
        return;
    };
    let (n0, n_cap, has_server) = {
        let g = gfx.borrow();
        (g.renderer.resident(), g.n_cap, !g.points_base.is_empty())
    };
    num.set_value(&n0.to_string());
    let _ = num.set_attribute("max", &n_cap.to_string());
    set_text("load-cap", &format!("≤ {}", group(n_cap)));
    if !has_server {
        let _ = num.set_attribute("disabled", "true");
        let _ = btn.set_attribute("disabled", "true");
        return;
    }
    let trigger = {
        let (gfx, num) = (gfx.clone(), num.clone());
        move || {
            let v = num.value_as_number();
            if v.is_finite() && v >= 1.0 {
                let budget = (v as usize).min(n_cap).max(1);
                // Re-load the CURRENT region (focus stack top) at the new budget — Keep updates the
                // top entry's budget, no push/pop.
                let region = gfx.borrow().focus_stack.last().map(|(r, _)| *r);
                if let Some(region) = region {
                    wasm_bindgen_futures::spawn_local(load_region(
                        gfx.clone(),
                        region,
                        Some(budget),
                        StackOp::Keep,
                    ));
                }
            }
        }
    };
    {
        let trigger = trigger.clone();
        add_listener(btn.as_ref(), "click", move |_e: web_sys::Event| trigger());
    }
    add_listener(num.as_ref(), "change", move |_e: web_sys::Event| trigger());
}

/// Toggle a class on an element by id.
fn set_class(id: &str, class: &str, on: bool) {
    if let Some(el) = by_id::<web_sys::HtmlElement>(id) {
        let _ = el.class_list().toggle_with_force(class, on);
    }
}

/// Show one flow tab pane + highlight its button; hide the others.
fn switch_tab(name: &str) {
    for t in ["data", "region", "cluster", "save", "display"] {
        set_class(&format!("tabbtn-{t}"), "active", t == name);
        set_class(&format!("tab-{t}"), "active", t == name);
    }
}

/// Wire tab switching + collapsible section headers via one delegated click listener.
fn bind_panel_chrome(gfx: &Rc<RefCell<Gfx>>) {
    let Some(doc) = document() else {
        return;
    };
    let gfx = gfx.clone();
    add_listener(doc.as_ref(), "click", move |e: web_sys::Event| {
        let Some(t) = e.target().and_then(|t| t.dyn_into::<web_sys::Element>().ok()) else {
            return;
        };
        if let Ok(Some(tab)) = t.closest(".tab") {
            if let Some(name) = tab.get_attribute("data-tab") {
                switch_tab(&name);
                update_cluster_panel(&gfx); // stats panel shows only on the Cluster tab
            }
        } else if let Ok(Some(ey)) = t.closest(".ey") {
            if let Some(sec) = ey.parent_element() {
                let _ = sec.class_list().toggle("collapsed");
            }
        }
    });
}

/// Show the cluster stats panel iff the Cluster tab is active and a result exists (so switching tabs
/// is the natural dismiss, and the right edge is free for the colorbar elsewhere).
fn update_cluster_panel(gfx: &Rc<RefCell<Gfx>>) {
    let on_cluster_tab = by_id::<web_sys::HtmlElement>("tabbtn-cluster")
        .map(|b| b.class_list().contains("active"))
        .unwrap_or(false);
    let show = on_cluster_tab && gfx.borrow().clustered;
    set_body_class("has-clusters", show);
    if show {
        let g = gfx.borrow();
        if let Some(s) = g.cluster_stats.as_ref() {
            render_cluster_panel(s, g.cluster_sel);
        }
    }
}

/// Isolate one cluster on the cloud (others greyed to noise), or show all (`sel = None`). Clicking
/// the already-isolated cluster toggles back to all.
fn isolate_cluster(gfx: &Rc<RefCell<Gfx>>, sel: Option<i32>) {
    {
        let mut gb = gfx.borrow_mut();
        let g: &mut Gfx = &mut gb;
        if !g.clustered {
            return;
        }
        let sel = if g.cluster_sel == sel { None } else { sel }; // toggle off if re-clicked
        for (j, &i) in g.cluster_idx.iter().enumerate() {
            let l = g.cluster_labels[j];
            let shown = l >= 0 && sel.map_or(true, |s| l == s);
            g.cpu_points[i]._pad[0] = if shown { l as u32 } else { GpuPoint::NO_CLUSTER };
        }
        reupload_points(g);
        g.cluster_sel = sel;
        g.params.color_mode = if g.hide_noise { 2 } else { 1 }; // isolate needs cluster colouring

        g.view_mode = ViewMode::Points; // cluster ids are invisible in Volume
    }
    set_checked("cl-color", true);
    set_body_class("cluster-color", true);
    set_checked("v-pts", true);
    set_checked("v-vol", false);
    set_volmode(false);
    update_cluster_panel(gfx); // re-render the list with the active row marked
}

/// JSON snapshot of the parameters used for a clustering run, so the export is reproducible.
fn cluster_params_json(p: &ClusterRunParams, k: usize, noise: usize, n_in: usize) -> String {
    let obj = js_sys::Object::new();
    let set = |key: &str, v: &JsValue| {
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str(key), v);
    };
    set("dataset_id", &JsValue::from_f64(p.dataset_id as f64));
    let ms_level = if p.ms_mask == 0b10 { "MS2" } else { "MS1" };
    set("ms_level", &JsValue::from_str(ms_level));
    let (algo, engine) = match (p.method, p.python) {
        (ClusterMethod::Hdbscan, _) => ("hdbscan", "python-sklearn"),
        (ClusterMethod::Dbscan, true) => ("dbscan", "python-sklearn"),
        (ClusterMethod::Dbscan, false) => ("dbscan", "wasm"),
    };
    set("algorithm", &JsValue::from_str(algo));
    set("engine", &JsValue::from_str(engine));
    // Each algorithm exports only its own parameters.
    if p.method == ClusterMethod::Hdbscan {
        set("min_cluster_size", &JsValue::from_f64(p.min_cluster_size as f64));
        set("min_samples", &JsValue::from_f64(p.hdb_min_samples as f64)); // 0 = auto
        set("cluster_selection_epsilon", &JsValue::from_f64(p.selection_eps));
    } else {
        set("eps", &JsValue::from_f64(p.eps as f64));
        set("min_points", &JsValue::from_f64(p.min_pts as f64));
    }
    set("rt_cycles", &JsValue::from_f64(p.rt_cycles));
    set("im_scans", &JsValue::from_f64(p.im_scans));
    set("mz_peak_widths", &JsValue::from_f64(p.mz_peak_widths));
    set("mz_resolution", &JsValue::from_f64(MZ_RESOLUTION));
    set("cycle_duration_s", &JsValue::from_f64(p.cycle_duration));
    set("im_per_scan_1k0", &JsValue::from_f64(p.im_per_scan));
    set("intensity_floor", &JsValue::from_f64(p.floor as f64));
    set("clusters", &JsValue::from_f64(k as f64));
    set("noise_points", &JsValue::from_f64(noise as f64));
    set("input_points", &JsValue::from_f64(n_in as f64));
    if let Some(b) = p.region {
        let region = js_sys::Object::new();
        let pair = |lo: f64, hi: f64| {
            let a = js_sys::Array::new();
            a.push(&JsValue::from_f64(lo));
            a.push(&JsValue::from_f64(hi));
            a
        };
        let _ = js_sys::Reflect::set(&region, &JsValue::from_str("mz"), &pair(b[0].0, b[0].1));
        let _ = js_sys::Reflect::set(&region, &JsValue::from_str("im"), &pair(b[1].0, b[1].1));
        let _ = js_sys::Reflect::set(&region, &JsValue::from_str("rt"), &pair(b[2].0, b[2].1));
        set("region", &region);
    }
    js_sys::JSON::stringify_with_replacer_and_space(&obj, &JsValue::NULL, &JsValue::from_f64(2.0))
        .ok()
        .and_then(|s| s.as_string())
        .unwrap_or_default()
}

/// Download the per-cluster summary as CSV plus the run's parameters as JSON (two data-URL anchors).
fn export_clusters(gfx: &Rc<RefCell<Gfx>>) {
    let g = gfx.borrow();
    let Some(s) = g.cluster_stats.as_ref() else {
        return;
    };
    let mut csv = String::from("cluster,size,intensity,mz_extent,im_extent,rt_extent\n");
    for c in 0..s.k {
        csv.push_str(&format!(
            "{c},{},{},{:.6},{:.6},{:.6}\n",
            s.sizes[c] as u64, s.int_sum[c], s.mz_extent[c], s.im_extent[c], s.rt_extent[c]
        ));
    }
    download_text("clusters.csv", "text/csv", &csv);
    if let Some(pj) = g.cluster_params_json.as_ref() {
        download_text("cluster_params.json", "application/json", pj);
    }
}

/// Enable the results export + reflect Save status. Results need a finished clustering; the config
/// export is always available.
fn set_save_ready(ready: bool) {
    set_disabled("cl-export", !ready);
    set_text("save-status", if ready { "results ready" } else { "no clustering yet" });
}

/// The reproducible *recipe* as JSON — dataset + region + algorithm + params — built live from the
/// current state, so it works before any clustering (unlike the post-run `cluster_params.json`).
fn cluster_config_json(g: &Gfx) -> String {
    let obj = js_sys::Object::new();
    let set = |key: &str, v: &JsValue| {
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str(key), v);
    };
    let dataset = js_sys::Object::new();
    let name = if g.dataset_name.is_empty() {
        format!("dataset {}", g.dataset_id) // /datasets not resolved yet / demo — never emit ""
    } else {
        g.dataset_name.clone()
    };
    let _ = js_sys::Reflect::set(&dataset, &"id".into(), &JsValue::from_f64(g.dataset_id as f64));
    let _ = js_sys::Reflect::set(&dataset, &"name".into(), &JsValue::from_str(&name));
    set("dataset", &dataset);
    let ms_level = match g.params.ms_mask {
        0b10 => "MS2",
        0b01 => "MS1",
        _ => "both",
    };
    set("ms_level", &JsValue::from_str(ms_level));
    set("intensity_floor", &JsValue::from_f64(g.params.filter_min[3] as f64));
    let (algo, engine) = match (g.cluster_method, g.use_python_cluster) {
        (ClusterMethod::Hdbscan, _) => ("hdbscan", "python-sklearn"),
        (ClusterMethod::Dbscan, true) => ("dbscan", "python-sklearn"),
        (ClusterMethod::Dbscan, false) => ("dbscan", "wasm"),
    };
    set("algorithm", &JsValue::from_str(algo));
    set("engine", &JsValue::from_str(engine));
    if g.cluster_method == ClusterMethod::Hdbscan {
        set("min_cluster_size", &JsValue::from_f64(g.cluster_min_cluster_size as f64));
        set("min_samples", &JsValue::from_f64(g.cluster_hdb_min_samples as f64));
        set("cluster_selection_epsilon", &JsValue::from_f64(g.cluster_selection_eps));
    } else {
        set("eps", &JsValue::from_f64(g.cluster_eps as f64));
        set("min_points", &JsValue::from_f64(g.cluster_min_pts as f64));
    }
    set("rt_cycles", &JsValue::from_f64(g.cluster_rt_cycles));
    set("im_scans", &JsValue::from_f64(g.cluster_im_scans));
    set("mz_peak_widths", &JsValue::from_f64(g.cluster_mz_peak_widths));
    set("mz_resolution", &JsValue::from_f64(MZ_RESOLUTION));
    // The run-level metric anchors, so the config reproduces the exact scaling on its own.
    set("cycle_duration_s", &JsValue::from_f64(g.cycle_duration));
    set("im_per_scan_1k0", &JsValue::from_f64(g.im_per_scan));
    if let Some(b) = g.axis_bounds {
        let region = js_sys::Object::new();
        let pair = |lo: f64, hi: f64| {
            let a = js_sys::Array::new();
            a.push(&JsValue::from_f64(lo));
            a.push(&JsValue::from_f64(hi));
            a
        };
        let _ = js_sys::Reflect::set(&region, &"mz".into(), &pair(b[0].0, b[0].1));
        let _ = js_sys::Reflect::set(&region, &"im".into(), &pair(b[1].0, b[1].1));
        let _ = js_sys::Reflect::set(&region, &"rt".into(), &pair(b[2].0, b[2].1));
        set("region", &region);
    }
    js_sys::JSON::stringify_with_replacer_and_space(&obj, &JsValue::NULL, &JsValue::from_f64(2.0))
        .ok()
        .and_then(|s| s.as_string())
        .unwrap_or_default()
}

/// Download the live config recipe (`config.json`).
fn export_config(gfx: &Rc<RefCell<Gfx>>) {
    let json = cluster_config_json(&gfx.borrow());
    download_text("config.json", "application/json", &json);
}

/// Trigger a file download of `content` via a `data:` URL anchor (attached to the DOM for the click
/// so browsers honor it, then removed).
fn download_text(filename: &str, mime: &str, content: &str) {
    let Some(doc) = document() else {
        return;
    };
    let Ok(el) = doc.create_element("a") else {
        return;
    };
    let Ok(a) = el.dyn_into::<web_sys::HtmlElement>() else {
        return;
    };
    let encoded = String::from(js_sys::encode_uri_component(content));
    let _ = a.set_attribute("href", &format!("data:{mime};charset=utf-8,{encoded}"));
    let _ = a.set_attribute("download", filename);
    if let Some(body) = doc.body() {
        let _ = body.append_child(&a);
        a.click();
        let _ = body.remove_child(&a);
    } else {
        a.click();
    }
}

/// Set an element's inner HTML by id.
fn set_html(id: &str, html: &str) {
    if let Some(el) = by_id::<web_sys::HtmlElement>(id) {
        el.set_inner_html(html);
    }
}

/// Render a 2D density projection to its `<canvas>` with the inferno colormap (sqrt-scaled). The
/// projection is row-major `x + bins*y`; y is flipped so the low bin sits at the bottom.
fn render_heatmap(canvas_id: &str, data: &[u32], bins: usize) {
    if bins == 0 || data.len() < bins * bins {
        return;
    }
    let Some(canvas) = by_id::<web_sys::HtmlCanvasElement>(canvas_id) else {
        return;
    };
    canvas.set_width(bins as u32);
    canvas.set_height(bins as u32);
    let Some(ctx) = canvas
        .get_context("2d")
        .ok()
        .flatten()
        .and_then(|c| c.dyn_into::<web_sys::CanvasRenderingContext2d>().ok())
    else {
        return;
    };
    let max = data.iter().copied().max().unwrap_or(1).max(1) as f32;
    let mut buf = vec![0u8; bins * bins * 4];
    for y in 0..bins {
        let row = bins - 1 - y; // flip: low bin at the bottom
        for x in 0..bins {
            let t = (data[x + bins * y] as f32 / max).sqrt();
            let rgb = colormap_sample(1, t); // inferno
            let i = (row * bins + x) * 4;
            buf[i] = rgb[0];
            buf[i + 1] = rgb[1];
            buf[i + 2] = rgb[2];
            buf[i + 3] = 255;
        }
    }
    if let Ok(img) = web_sys::ImageData::new_with_u8_clamped_array_and_sh(
        wasm_bindgen::Clamped(&buf),
        bins as u32,
        bins as u32,
    ) {
        let _ = ctx.put_image_data(&img, 0.0, 0.0);
    }
}

/// Render all three projection minimaps from a load's meta. Returns whether valid maps were drawn;
/// when the meta lacks usable projections the stale canvases are cleared (so box-select can be
/// disabled rather than acting on a previous view).
fn render_maps(meta: &MetaInfo) -> bool {
    match &meta.proj {
        Some(proj) if meta.proj_bins > 0 => {
            for (i, id) in ["map-0", "map-1", "map-2"].iter().enumerate() {
                render_heatmap(id, &proj[i], meta.proj_bins);
            }
            true
        }
        _ => {
            for id in ["map-0", "map-1", "map-2"] {
                clear_canvas(id);
            }
            false
        }
    }
}

/// Clear a canvas (so it can't show a stale view from a prior load).
fn clear_canvas(id: &str) {
    let Some(canvas) = by_id::<web_sys::HtmlCanvasElement>(id) else {
        return;
    };
    if let Some(ctx) = canvas
        .get_context("2d")
        .ok()
        .flatten()
        .and_then(|c| c.dyn_into::<web_sys::CanvasRenderingContext2d>().ok())
    {
        ctx.clear_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);
    }
}

/// Map a box on projection `proj` (x,y fractions of the map) to a 4D region of the current view.
/// The map renders low-at-bottom, so the y fractions flip; the unselected axis keeps the full view
/// range, and the current intensity floor carries through.
fn region_from_box(g: &Gfx, proj: usize, x: (f64, f64), y: (f64, f64)) -> Option<Region> {
    let b = g.axis_bounds?;
    let lerp = |rng: (f64, f64), f: f64| rng.0 + f * (rng.1 - rng.0);
    let (yl, yh) = (1.0 - y.1, 1.0 - y.0); // top->bottom px, but low-at-bottom display
    let imin = g.params.filter_min[3].max(0.0);
    let (mz, im, rt) = match proj {
        0 => ((lerp(b[0], x.0), lerp(b[0], x.1)), (lerp(b[1], yl), lerp(b[1], yh)), b[2]),
        1 => ((lerp(b[0], x.0), lerp(b[0], x.1)), b[1], (lerp(b[2], yl), lerp(b[2], yh))),
        2 => (b[0], (lerp(b[1], x.0), lerp(b[1], x.1)), (lerp(b[2], yl), lerp(b[2], yh))),
        _ => return None,
    };
    Some(Region { mz, im, rt, imin })
}

/// Mouse position within a map-wrap as fractions `[0,1]` (x left→right, y top→bottom).
fn map_frac(mw: &web_sys::HtmlElement, e: &web_sys::MouseEvent) -> (f64, f64) {
    let rect = mw.get_bounding_client_rect();
    let nx = ((e.client_x() as f64 - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
    let ny = ((e.client_y() as f64 - rect.top()) / rect.height().max(1.0)).clamp(0.0, 1.0);
    (nx, ny)
}

/// Position (or, with `show=false`, hide) the selection-box overlay on map `proj`.
fn set_sel_box(proj: usize, x0: f64, y0: f64, x1: f64, y1: f64, show: bool) {
    let Some(sel) = by_id::<web_sys::HtmlElement>(&format!("sel-{proj}")) else {
        return;
    };
    let st = sel.style();
    if !show {
        let _ = st.set_property("display", "none");
        return;
    }
    let (l, r) = (x0.min(x1), x0.max(x1));
    let (t, b) = (y0.min(y1), y0.max(y1));
    let _ = st.set_property("display", "block");
    let _ = st.set_property("left", &format!("{:.2}%", l * 100.0));
    let _ = st.set_property("top", &format!("{:.2}%", t * 100.0));
    let _ = st.set_property("width", &format!("{:.2}%", (r - l) * 100.0));
    let _ = st.set_property("height", &format!("{:.2}%", (b - t) * 100.0));
}

/// Wire box-select on the three projection maps: drag a rectangle → Focus that 4D region.
fn bind_maps(gfx: &Rc<RefCell<Gfx>>) {
    if gfx.borrow().points_base.is_empty() {
        return; // demo fallback: maps render but can't drive a server focus
    }
    let Some(window) = web_sys::window() else {
        return;
    };
    // Shared drag state: (proj index, start-x frac, start-y frac).
    let drag: Rc<RefCell<Option<(usize, f64, f64)>>> = Rc::new(RefCell::new(None));

    for proj in 0..3usize {
        let Some(mw) = by_id::<web_sys::HtmlElement>(&format!("mw-{proj}")) else {
            continue;
        };
        let (drag, mw2) = (drag.clone(), mw.clone());
        add_listener(mw.as_ref(), "mousedown", move |e: web_sys::MouseEvent| {
            e.prevent_default();
            let (nx, ny) = map_frac(&mw2, &e);
            *drag.borrow_mut() = Some((proj, nx, ny));
            set_sel_box(proj, nx, ny, nx, ny, true);
        });
    }
    {
        let drag = drag.clone();
        add_listener(window.as_ref(), "mousemove", move |e: web_sys::MouseEvent| {
            let Some((proj, x0, y0)) = *drag.borrow() else {
                return;
            };
            // No button held => the mouseup happened off-window; cancel the drag + overlay.
            if e.buttons() == 0 {
                *drag.borrow_mut() = None;
                set_sel_box(proj, 0.0, 0.0, 0.0, 0.0, false);
                return;
            }
            if let Some(mw) = by_id::<web_sys::HtmlElement>(&format!("mw-{proj}")) {
                let (nx, ny) = map_frac(&mw, &e);
                set_sel_box(proj, x0, y0, nx, ny, true);
            }
        });
    }
    {
        let (drag, gfx) = (drag.clone(), gfx.clone());
        add_listener(window.as_ref(), "mouseup", move |e: web_sys::MouseEvent| {
            let Some((proj, x0, y0)) = drag.borrow_mut().take() else {
                return;
            };
            set_sel_box(proj, 0.0, 0.0, 0.0, 0.0, false);
            let Some(mw) = by_id::<web_sys::HtmlElement>(&format!("mw-{proj}")) else {
                return;
            };
            let (nx, ny) = map_frac(&mw, &e);
            let (xa, xb) = (x0.min(nx), x0.max(nx));
            let (ya, yb) = (y0.min(ny), y0.max(ny));
            if (xb - xa) < 0.02 || (yb - ya) < 0.02 {
                return; // ignore clicks / tiny drags
            }
            {
                let g = gfx.borrow();
                if g.reloading || !g.maps_ok {
                    return; // mid-load, or maps don't reflect the current view
                }
            }
            if let Some(r) = region_from_box(&gfx.borrow(), proj, (xa, xb), (ya, yb)) {
                let budget = gfx.borrow().n_cap;
                wasm_bindgen_futures::spawn_local(load_region(
                    gfx.clone(),
                    r,
                    Some(budget),
                    StackOp::Push,
                ));
            }
        });
    }
}

/// Create the DOM label elements inside `#axis-labels`; returns `(element, cube-position)` pairs
/// for per-frame projection. Empty when bounds are unknown (demo) or the container is missing.
fn create_axis_labels(bounds: Option<[(f64, f64); 3]>) -> Vec<(web_sys::HtmlElement, [f32; 3])> {
    let (Some(bounds), Some(doc)) = (bounds, document()) else {
        return Vec::new();
    };
    let Some(container) = doc.get_element_by_id("axis-labels") else {
        return Vec::new();
    };
    container.set_inner_html(""); // clear any stale labels
    let mut out = Vec::new();
    for (text, world, color) in axis_label_specs(bounds) {
        if let Some(el) = doc
            .create_element("div")
            .ok()
            .and_then(|e| e.dyn_into::<web_sys::HtmlElement>().ok())
        {
            el.set_class_name("axlabel");
            el.set_text_content(Some(&text));
            let _ = el.style().set_property("color", color);
            let _ = container.append_child(&el);
            out.push((el, world));
        }
    }
    out
}

// ---- control-panel wiring ----------------------------------------------------------------

fn document() -> Option<web_sys::Document> {
    web_sys::window().and_then(|w| w.document())
}

fn by_id<T: JsCast>(id: &str) -> Option<T> {
    document()?.get_element_by_id(id)?.dyn_into::<T>().ok()
}

fn set_text(id: &str, s: &str) {
    if let Some(el) = document().and_then(|d| d.get_element_by_id(id)) {
        el.set_text_content(Some(s));
    }
}

/// Reflect a state change back onto a checkbox (e.g. uncheck Auto-orbit when the mouse takes over).
fn set_checked(id: &str, v: bool) {
    if let Some(inp) = by_id::<web_sys::HtmlInputElement>(id) {
        inp.set_checked(v);
    }
}

/// Group an integer with thin spaces for the HUD readout, e.g. 300000 -> "300 000".
/// Compact count for the HUD: `3.7M`, `379k`, or grouped digits under 10k. Keeps the
/// `displayed / resident` readout from overflowing the stat cell.
fn group_short(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 10_000 {
        format!("{:.0}k", n as f64 / 1e3)
    } else {
        group(n)
    }
}

fn group(n: usize) -> String {
    let s = n.to_string();
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in b.iter().enumerate() {
        if i > 0 && (b.len() - i) % 3 == 0 {
            out.push('\u{2009}'); // thin space
        }
        out.push(*c as char);
    }
    out
}

/// Count points surviving the active 4D filter (spatial window + intensity floor + MS mask) among
/// the drawn prefix of the resident pool. Camera-independent (frustum culling excluded), so it is
/// recomputed only when the filter or draw count changes — not every frame.
fn recount_displayed(g: &Gfx) -> usize {
    let p = &g.params;
    let limit = (g.renderer.drawn() as usize).min(g.cpu_points.len());
    let (fmin, fmax, ms) = (p.filter_min, p.filter_max, p.ms_mask);
    let mut shown = 0usize;
    for pt in &g.cpu_points[..limit] {
        let pos = pt.pos;
        if pos[0] < fmin[0]
            || pos[0] > fmax[0]
            || pos[1] < fmin[1]
            || pos[1] > fmax[1]
            || pos[2] < fmin[2]
            || pos[2] > fmax[2]
        {
            continue;
        }
        if pt.intensity < fmin[3] {
            continue;
        }
        let is_ms2 = (pt.flags & GpuPoint::MS2_FLAG) != 0;
        let pass = if is_ms2 { ms & 0b10 != 0 } else { ms & 0b01 != 0 };
        if pass {
            shown += 1;
        }
    }
    shown
}

/// Bind the Display slider (1..100 %) to the renderer's runtime draw count — trade detail for
/// performance live, with no re-upload. Points are shuffled server-side, so the drawn prefix is a
/// representative subsample at any fraction. Also reflects the drawn count in the HUD.
fn bind_display(gfx: &Rc<RefCell<Gfx>>) {
    let (Some(slider), Some(num)) = (
        by_id::<web_sys::HtmlInputElement>("disp"),
        by_id::<web_sys::HtmlInputElement>("disp-n"),
    ) else {
        return;
    };
    // slider(%) -> count: update the count field + HUD
    {
        let (gfx, s, n) = (gfx.clone(), slider.clone(), num.clone());
        add_listener(slider.as_ref(), "input", move |_e: web_sys::Event| {
            let total = gfx.borrow().renderer.resident().max(1);
            let pct = (s.value_as_number() / 100.0).clamp(0.0, 1.0);
            let (k, _) = display_apply(&gfx, ((total as f64) * pct).round() as u32);
            n.set_value(&k.to_string());
        });
    }
    // count -> %: update the slider + HUD
    {
        let (gfx, s, n) = (gfx.clone(), slider.clone(), num.clone());
        add_listener(num.as_ref(), "change", move |_e: web_sys::Event| {
            let raw = n.value_as_number();
            if raw.is_nan() {
                return;
            }
            let (k, total) = display_apply(&gfx, raw.max(1.0) as u32);
            s.set_value(&format!("{:.0}", 100.0 * (k as f64) / (total as f64)));
            n.set_value(&k.to_string());
        });
    }
    // Initialize draw count + count field from the slider's current value.
    let total = gfx.borrow().renderer.resident().max(1);
    let pct = (slider.value_as_number() / 100.0).clamp(0.0, 1.0);
    let (k0, _) = display_apply(gfx, ((total as f64) * pct).round() as u32);
    num.set_value(&k0.to_string());
    let _ = num.set_attribute("max", &total.to_string());
}

/// Set the renderer's draw count (clamped to the live resident pool) and update the HUD.
/// Returns `(applied_count, pool_total)`.
fn display_apply(gfx: &Rc<RefCell<Gfx>>, k: u32) -> (u32, u32) {
    let mut g = gfx.borrow_mut();
    let total = g.renderer.resident().max(1);
    let kk = k.clamp(1, total);
    g.renderer.set_draw_count(kk);
    g.filter_dirty = true; // the displayed count is recomputed next frame
    (kk, total)
}

/// Bind a slider + its editable number field bidirectionally to a render parameter: dragging the
/// slider updates the param and the number; typing a number clamps it to the slider's min/max,
/// moves the slider, and updates the param. `prec` is the displayed decimal count.
fn bind_value(
    gfx: &Rc<RefCell<Gfx>>,
    slider_id: &str,
    num_id: &str,
    prec: usize,
    apply: impl Fn(&mut Gfx, f64) + 'static,
) {
    let (Some(slider), Some(num)) = (
        by_id::<web_sys::HtmlInputElement>(slider_id),
        by_id::<web_sys::HtmlInputElement>(num_id),
    ) else {
        return;
    };
    let apply = std::rc::Rc::new(apply);
    {
        let (gfx, apply, s, n) = (gfx.clone(), apply.clone(), slider.clone(), num.clone());
        add_listener(slider.as_ref(), "input", move |_e: web_sys::Event| {
            let v = s.value_as_number();
            apply(&mut gfx.borrow_mut(), v);
            n.set_value(&format!("{v:.prec$}"));
        });
    }
    {
        let (gfx, apply, s, n) = (gfx.clone(), apply.clone(), slider.clone(), num.clone());
        add_listener(num.as_ref(), "change", move |_e: web_sys::Event| {
            let raw = n.value_as_number();
            if raw.is_nan() {
                return; // empty / non-numeric: leave state as-is
            }
            let lo = s.min().parse::<f64>().unwrap_or(f64::NEG_INFINITY);
            let hi = s.max().parse::<f64>().unwrap_or(f64::INFINITY);
            let v = raw.clamp(lo, hi);
            n.set_value(&format!("{v:.prec$}"));
            s.set_value(&v.to_string());
            apply(&mut gfx.borrow_mut(), v);
        });
    }
}

/// Set a slider + its number field to `v` without firing handlers (for startup sync).
fn set_value_pair(slider_id: &str, num_id: &str, v: f64, prec: usize) {
    if let Some(s) = by_id::<web_sys::HtmlInputElement>(slider_id) {
        s.set_value(&v.to_string());
    }
    if let Some(n) = by_id::<web_sys::HtmlInputElement>(num_id) {
        n.set_value(&format!("{v:.prec$}"));
    }
}

/// Push the current Rust render state onto every DOM control. Browsers restore a control's prior
/// value across reloads, which can silently desync the radios/checkbox/select/sliders from the
/// actual params (a checked-but-stale radio won't emit `change` when clicked); calling this after
/// wiring re-asserts the truth so the panel always reflects what's rendering.
fn sync_controls(gfx: &Rc<RefCell<Gfx>>) {
    let (auto, rmode, xf, ms, cmap, psize, opac, expo, floor, cl_eps, cl_min, cl_rtc, cl_ims, cl_mzw, cl_mcs, cl_hms, cl_cse) = {
        let g = gfx.borrow();
        (
            g.auto_rotate,
            g.params.render_mode,
            g.params.transfer[0] as i32,
            g.params.ms_mask,
            g.params.colormap_id,
            g.params.point_size as f64,
            g.params.opacity as f64,
            g.params.transfer[3] as f64,
            g.params.filter_min[3] as f64,
            g.cluster_eps as f64,
            g.cluster_min_pts as f64,
            g.cluster_rt_cycles,
            g.cluster_im_scans,
            g.cluster_mz_peak_widths,
            g.cluster_min_cluster_size as f64,
            g.cluster_hdb_min_samples as f64,
            g.cluster_selection_eps,
        )
    };
    set_checked("ar", auto);
    set_checked("m-add", rmode == 0);
    set_checked("m-opq", rmode == 1);
    set_checked("xf-lin", xf == 0);
    set_checked("xf-sqrt", xf == 1);
    set_checked("xf-log", xf == 2);
    set_checked("ms-1", ms == 0b01);
    set_checked("ms-2", ms == 0b10);
    set_checked("ms-b", ms == 0b11);
    if let Some(sel) = by_id::<web_sys::HtmlSelectElement>("cmap") {
        sel.set_value(&cmap.to_string());
    }
    set_cbar(cmap);
    set_value_pair("psize", "psize-n", psize, 1);
    set_value_pair("opac", "opac-n", opac, 2);
    set_value_pair("expo", "expo-n", expo, 2);
    set_value_pair("floor", "floor-n", floor, 0);
    // Cluster controls (defeat browser form-restore; no result yet, so colour off).
    set_value_pair("cl-eps", "cl-eps-n", cl_eps, 3);
    set_value_pair("cl-min", "cl-min-n", cl_min, 0);
    set_value_pair("cl-rtc", "cl-rtc-n", cl_rtc, 0);
    set_value_pair("cl-ims", "cl-ims-n", cl_ims, 1);
    set_value_pair("cl-mzw", "cl-mzw-n", cl_mzw, 1);
    set_value_pair("cl-mcs", "cl-mcs-n", cl_mcs, 0);
    set_value_pair("cl-hms", "cl-hms-n", cl_hms, 0);
    set_value_pair("cl-cse", "cl-cse-n", cl_cse, 3);
    set_checked("cl-color", true); // colour by cluster is on by default (applies after a Run)
    set_checked("hide-noise", false);
    set_checked("show-windows", false); // off by default (defeats browser form-restore)
    // Clustering algorithm: default to built-in; the Python options are enabled by the startup probe.
    if let Some(sel) = by_id::<web_sys::HtmlSelectElement>("cl-algo") {
        sel.set_value("wasm");
    }
    set_disabled("opt-sklearn", true);
    set_disabled("opt-hdbscan", true);
    set_body_class("algo-hdb", false);
    // Crops start at the full range (thumbs, fills, readouts).
    reset_crops(gfx);
}

/// Bind a radio/checkbox: call `f(g, checked)` on change.
fn on_toggle(id: &str, gfx: &Rc<RefCell<Gfx>>, mut f: impl FnMut(&mut Gfx, bool) + 'static) {
    if let Some(inp) = by_id::<web_sys::HtmlInputElement>(id) {
        let (gfx, el) = (gfx.clone(), inp.clone());
        add_listener(inp.as_ref(), "change", move |_e: web_sys::Event| {
            f(&mut gfx.borrow_mut(), el.checked());
        });
    }
}

/// Bind a `<button>` click.
fn on_click(id: &str, gfx: &Rc<RefCell<Gfx>>, mut f: impl FnMut(&mut Gfx) + 'static) {
    if let Some(el) = document().and_then(|d| d.get_element_by_id(id)) {
        let gfx = gfx.clone();
        add_listener(el.as_ref(), "click", move |_e: web_sys::Event| f(&mut gfx.borrow_mut()));
    }
}

fn set_cbar(colormap_id: u32) {
    if let Some(el) = document().and_then(|d| d.get_element_by_id("cbar")) {
        el.set_class_name(&format!("cbar v{colormap_id}"));
    }
}

/// Wire every panel control to a live render parameter.
fn wire_controls(gfx: &Rc<RefCell<Gfx>>) {
    // View
    on_toggle("ar", gfx, |g, on| g.auto_rotate = on);
    on_click("reset", gfx, |g| g.camera.reset());
    on_click("roll", gfx, |g| {
        g.camera.roll_axis();
        set_text("roll", &format!("up: {}", ROLL_UP_AXIS[(g.camera.roll % 3) as usize]));
    });

    // Render
    on_toggle("m-add", gfx, |g, on| {
        if on {
            g.point_mode = PointMode::AdditiveDensity;
            g.params.render_mode = 0;
        }
    });
    on_toggle("m-opq", gfx, |g, on| {
        if on {
            g.point_mode = PointMode::StructuralOpaque;
            g.params.render_mode = 1;
        }
    });
    if let Some(sel) = by_id::<web_sys::HtmlSelectElement>("cmap") {
        let (gfx, el) = (gfx.clone(), sel.clone());
        add_listener(sel.as_ref(), "change", move |_e: web_sys::Event| {
            // Read the option's own value (don't rely on DOM order matching COLORMAP_NAMES).
            let id = el.value().parse::<u32>().unwrap_or(0).min(COLORMAP_NAMES.len() as u32 - 1);
            gfx.borrow_mut().params.colormap_id = id;
            set_cbar(id);
        });
    }
    bind_value(gfx, "psize", "psize-n", 1, |g, v| g.params.point_size = v as f32);
    bind_value(gfx, "opac", "opac-n", 2, |g, v| g.params.opacity = v as f32);

    // Intensity transfer
    on_toggle("xf-lin", gfx, |g, on| if on { g.params.transfer[0] = 0.0; });
    on_toggle("xf-sqrt", gfx, |g, on| if on { g.params.transfer[0] = 1.0; });
    on_toggle("xf-log", gfx, |g, on| if on { g.params.transfer[0] = 2.0; });
    bind_value(gfx, "expo", "expo-n", 2, |g, v| g.params.transfer[3] = v as f32);
    // Floor is linear in real counts; its slider/number range is scaled to the data in run().
    bind_value(gfx, "floor", "floor-n", 0, |g, v| {
        g.params.filter_min[3] = v as f32;
        g.filter_dirty = true;
        invalidate_clusters_mut(g);
    });

    // Filters
    on_toggle("ms-1", gfx, |g, on| if on { g.params.ms_mask = 0b01; g.filter_dirty = true; g.vol_needs_grid = true; invalidate_clusters_mut(g); });
    on_toggle("ms-2", gfx, |g, on| if on { g.params.ms_mask = 0b10; g.filter_dirty = true; g.vol_needs_grid = true; invalidate_clusters_mut(g); });
    on_toggle("ms-b", gfx, |g, on| if on { g.params.ms_mask = 0b11; g.filter_dirty = true; g.vol_needs_grid = true; invalidate_clusters_mut(g); });
    bind_crop(gfx, "cmz", 0);
    bind_crop(gfx, "cim", 1);
    bind_crop(gfx, "crt", 2);
}

/// Bind a lo/hi range-slider pair to `filter_min/max[axis]` (normalized cube `[-1, 1]`). The two
/// thumbs push each other rather than crossing: dragging `lo` past `hi` bumps `hi` up (and vice
/// versa), so the lower bound can never exceed the upper.
fn bind_crop(gfx: &Rc<RefCell<Gfx>>, prefix: &str, axis: usize) {
    let (lo, hi) = match (
        by_id::<web_sys::HtmlInputElement>(&format!("{prefix}-lo")),
        by_id::<web_sys::HtmlInputElement>(&format!("{prefix}-hi")),
    ) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => return,
    };
    // Lower thumb: if it passes the upper, push the upper up to meet it.
    {
        let (gfx_c, lo_c, hi_c, p) = (gfx.clone(), lo.clone(), hi.clone(), prefix.to_string());
        add_listener(lo.as_ref(), "input", move |_e: web_sys::Event| {
            let _ = lo_c.style().set_property("z-index", "5"); // active thumb on top
            let _ = hi_c.style().set_property("z-index", "4");
            if lo_c.value_as_number() > hi_c.value_as_number() {
                hi_c.set_value(&lo_c.value());
            }
            crop_apply(&gfx_c, axis, &p, lo_c.value_as_number(), hi_c.value_as_number());
        });
    }
    // Upper thumb: if it drops below the lower, push the lower down to meet it.
    {
        let (gfx_c, lo_c, hi_c, p) = (gfx.clone(), lo.clone(), hi.clone(), prefix.to_string());
        add_listener(hi.as_ref(), "input", move |_e: web_sys::Event| {
            let _ = hi_c.style().set_property("z-index", "5"); // active thumb on top
            let _ = lo_c.style().set_property("z-index", "4");
            if hi_c.value_as_number() < lo_c.value_as_number() {
                lo_c.set_value(&hi_c.value());
            }
            crop_apply(&gfx_c, axis, &p, lo_c.value_as_number(), hi_c.value_as_number());
        });
    }
}

/// Apply a crop pair (slider units 0..1000) to `filter_min/max[axis]`, refresh the real-unit
/// readout, and move the highlighted fill bar to sit between the two thumbs.
fn crop_apply(gfx: &Rc<RefCell<Gfx>>, axis: usize, prefix: &str, lo_val: f64, hi_val: f64) {
    let a = (lo_val / 1000.0) as f32 * 2.0 - 1.0;
    let b = (hi_val / 1000.0) as f32 * 2.0 - 1.0;
    let bounds = {
        let mut g = gfx.borrow_mut();
        g.params.filter_min[axis] = a;
        g.params.filter_max[axis] = b;
        g.filter_dirty = true;
        invalidate_clusters_mut(&mut g);
        g.axis_bounds
    };
    set_text(&format!("{prefix}-v"), &crop_label(bounds, axis, a, b));
    if let Some(fill) = document()
        .and_then(|d| d.get_element_by_id(&format!("{prefix}-fill")))
        .and_then(|e| e.dyn_into::<web_sys::HtmlElement>().ok())
    {
        let st = fill.style();
        let _ = st.set_property("left", &format!("{:.2}%", lo_val / 10.0));
        let _ = st.set_property("width", &format!("{:.2}%", (hi_val - lo_val).max(0.0) / 10.0));
    }
}

/// Inject an SVG area chart of `bins` into element `id` (sqrt height so low bins stay visible).
fn set_hist_svg(id: &str, bins: &[u32]) {
    let Some(el) = document().and_then(|d| d.get_element_by_id(id)) else {
        return;
    };
    if bins.len() < 2 {
        return;
    }
    let max = (*bins.iter().max().unwrap_or(&1)).max(1) as f32;
    let last = bins.len() - 1;
    let mut d = String::from("M0,100");
    for (i, &c) in bins.iter().enumerate() {
        let h = (c as f32 / max).sqrt() * 100.0;
        d.push_str(&format!(" L{i},{:.1}", 100.0 - h));
    }
    d.push_str(&format!(" L{last},100 Z"));
    el.set_inner_html(&format!(
        "<svg viewBox=\"0 0 {last} 100\" preserveAspectRatio=\"none\">\
         <path d=\"{d}\" fill=\"rgba(120,150,190,0.30)\" stroke=\"rgba(150,180,220,0.55)\" stroke-width=\"0.6\"/>\
         </svg>"
    ));
}

/// Draw the per-axis distribution strips above the crop sliders, aligned to their range.
fn draw_hist_backdrops(hist: &[Vec<u32>; 3]) {
    for (axis, prefix) in [(0usize, "cmz"), (1, "cim"), (2, "crt")] {
        set_hist_svg(&format!("{prefix}-hist"), &hist[axis]);
    }
}

/// Wire mouse controls into the camera: left-drag orbits, wheel zooms, shift/right-drag pans.
/// First interaction stops the auto-orbit. `mousemove`/`mouseup` live on the window so a drag
/// keeps working when the cursor leaves the canvas.
fn wire_input(gfx: &Rc<RefCell<Gfx>>, window: &web_sys::Window, canvas: &web_sys::HtmlCanvasElement) {
    {
        let gfx = gfx.clone();
        add_listener(canvas.as_ref(), "mousedown", move |e: web_sys::MouseEvent| {
            let mut g = gfx.borrow_mut();
            g.auto_rotate = false;
            g.dragging = Some(e.button());
            set_checked("ar", false); // keep the Auto-orbit switch in sync
        });
    }
    {
        let gfx = gfx.clone();
        add_listener(window.as_ref(), "mousemove", move |e: web_sys::MouseEvent| {
            let mut g = gfx.borrow_mut();
            if let Some(btn) = g.dragging {
                let (dx, dy) = (e.movement_x() as f32, e.movement_y() as f32);
                if btn == 0 && !e.shift_key() {
                    g.camera.orbit(dx, dy);
                } else {
                    g.camera.pan(dx, dy);
                }
            }
        });
    }
    {
        let gfx = gfx.clone();
        add_listener(window.as_ref(), "mouseup", move |_e: web_sys::MouseEvent| {
            gfx.borrow_mut().dragging = None;
        });
    }
    {
        let gfx = gfx.clone();
        add_listener(canvas.as_ref(), "wheel", move |e: web_sys::WheelEvent| {
            e.prevent_default(); // don't scroll the page
            let mut g = gfx.borrow_mut();
            g.auto_rotate = false; // zoom is an interaction too — hand control to the user
            g.camera.zoom(-(e.delta_y() as f32) / 100.0);
            set_checked("ar", false);
        });
    }
    // Suppress the context menu so right-drag pan works.
    add_listener(canvas.as_ref(), "contextmenu", |e: web_sys::MouseEvent| e.prevent_default());
}

/// Attach a typed event listener and leak the closure (it lives for the page; cancellation is a
/// Phase-4 embedding concern).
fn add_listener<E>(target: &web_sys::EventTarget, event: &str, handler: impl FnMut(E) + 'static)
where
    E: wasm_bindgen::convert::FromWasmAbi + 'static,
{
    let cb = Closure::wrap(Box::new(handler) as Box<dyn FnMut(E)>);
    let _ = target.add_event_listener_with_callback(event, cb.as_ref().unchecked_ref());
    cb.forget();
}

fn create_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// A synthetic point cloud in the normalized cube `[-1, 1]^3`: a helix plus two gaussian blobs,
/// intensities spanning the default log transfer range. Deterministic (seeded LCG, no `rand`).
fn demo_cloud() -> Vec<GpuPoint> {
    const N: usize = 250_000;
    let mut seed: u32 = 0x9E37_79B9;
    let mut rng = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 24) as f32 // [0, 1)
    };
    let blob = |c: [f32; 3], s: f32, r: &mut dyn FnMut() -> f32| {
        // crude gaussian via the central limit theorem (mean of 3 uniforms).
        let g = |r: &mut dyn FnMut() -> f32| (r() + r() + r() - 1.5) * 2.0 * s;
        [c[0] + g(r), c[1] + g(r), c[2] + g(r)]
    };

    let mut pts = Vec::with_capacity(N);
    for i in 0..N {
        let m = (rng() * 3.0) as u32;
        let pos = match m {
            0 => {
                let t = i as f32 / N as f32;
                let ang = t * 40.0;
                [0.6 * ang.cos(), 0.6 * ang.sin(), t * 2.0 - 1.0]
            }
            1 => blob([-0.4, 0.3, -0.2], 0.16, &mut rng),
            _ => blob([0.5, -0.4, 0.4], 0.20, &mut rng),
        };
        let u = rng();
        let intensity = 50.0 + u * u * 50_000.0; // skew toward low, span ~[50, 5e4]
        pts.push(GpuPoint {
            pos,
            intensity,
            weight: 1.0,
            flags: (m == 0) as u32, // helix tagged MS2 for variety; both shown via ms_mask
            _pad: [GpuPoint::NO_CLUSTER, 0],
        });
    }
    pts
}
