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
use tims_viewer::render::colormap::COLORMAP_NAMES;
use tims_viewer::render::point_cloud::{PointCloudRenderer, PointMode};
use tims_viewer::render::uniforms::ParamsUniform;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const CANVAS_ID: &str = "tims-canvas";
/// Default point-server port (`tims-viewer DEMO --serve 8090`). Override with `?port=N` in the URL.
const DEFAULT_SERVER_PORT: u16 = 8090;

/// wasm entry point — Trunk/wasm-bindgen call this on load.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
    wasm_bindgen_futures::spawn_local(async {
        if let Err(e) = run().await {
            log::error!("{e}");
            show_status(&format!("tims-viewer could not start: {e}"));
        }
    });
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
    /// Set when the active filter or draw count changes; the next frame recomputes the displayed
    /// count (CPU-side over `cpu_points`) — camera-independent, so it need not run every frame.
    filter_dirty: bool,
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
        self.update_labels(w / h);
        self.renderer.update_params(&self.queue, &self.params);

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
        self.renderer.prepare(&self.queue, &mut enc);
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
            self.renderer.render(&mut rpass, self.point_mode);
            self.axes.render(&mut rpass);
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
    // Fetch /meta first to validate the wire contract and pick up real-unit axis/intensity ranges,
    // then the binary points; fall back to a synthetic cloud if the server is absent/incompatible.
    let mut axis_bounds: Option<[(f64, f64); 3]> = None;
    let mut server_meta: Option<MetaInfo> = None;
    let (pts, is_demo_fallback) = match fetch_meta(&meta_url(&url)).await {
        Ok(m) => match fetch_points(&url).await {
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
    // Cap to what the GPU buffer can hold (master is a vertex+storage buffer): on a 12M budget
    // = 384 MB, clamp to the device's max so the allocation can't fail.
    let n_cap = gpu_point_cap(&device, supports_compaction);
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
        points_base: if is_demo_fallback { String::new() } else { url.clone() },
        n_cap,
        supports_compaction,
        cpu_points,
        floor_hi,
        reloading: false,
        filter_dirty: true,
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
    // Distribution strips: per-axis crops + the linear intensity floor.
    if let Some(m) = &server_meta {
        if let Some(h) = &m.hist {
            draw_hist_backdrops(h);
        }
        if let Some(ih) = &m.i_hist {
            set_hist_svg("floor-hist", ih);
        }
    }
    // Runtime detail/performance control over how many of the loaded points are drawn.
    bind_display(&gfx);
    // Load-budget control: re-fetch a different number of points from the server.
    bind_load(&gfx);

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

/// The point-server URL. `?points=<url>` overrides it outright (for proxied/same-origin
/// deploys); else `?port=N` selects `http://localhost:N/points`; else [`DEFAULT_SERVER_PORT`].
fn points_url() -> String {
    let search = web_sys::window()
        .and_then(|w| w.location().search().ok())
        .unwrap_or_default();
    let params = search.trim_start_matches('?');
    // Explicit full-URL override (percent-decoded).
    if let Some(v) = params.split('&').find_map(|kv| kv.strip_prefix("points=")) {
        if !v.is_empty() {
            return js_sys::decode_uri_component(v)
                .ok()
                .and_then(|s| s.as_string())
                .unwrap_or_else(|| v.to_string());
        }
    }
    let port = params
        .split('&')
        .find_map(|kv| kv.strip_prefix("port=").and_then(|v| v.parse::<u16>().ok()))
        .unwrap_or(DEFAULT_SERVER_PORT);
    format!("http://localhost:{port}/points")
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
        return Err("empty body".into());
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

    Ok(MetaInfo {
        bounds,
        i_p1: pos("p1").unwrap_or(0.0),
        i_p50: pos("p50").unwrap_or(0.0),
        i_p99: pos("p99").unwrap_or(0.0),
        stride,
        hist,
        i_hist,
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
        // Re-apply auto-transfer (exposure + floor range) and bounds from the new meta; keep the
        // user's colormap / point size / opacity / MS mask.
        if let Some(m) = &meta {
            apply_auto_transfer(&mut g.params, m);
            g.floor_hi = m.i_p99.max(1.0);
            g.axis_bounds = m.bounds;
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
    }
    for id in ["floor", "floor-n"] {
        if let Some(inp) = by_id::<web_sys::HtmlInputElement>(id) {
            inp.set_value("0");
        }
    }
    reset_crops(gfx);
    reset_display(gfx, n);
}

/// Re-fetch the run at a new budget (full bounds) and rebuild the GPU buffer. Debounced via
/// `reloading`. Focus/Back will drive the same path with a region query.
async fn reload(gfx: Rc<RefCell<Gfx>>, budget: usize) {
    {
        let mut g = gfx.borrow_mut();
        if g.reloading || g.points_base.is_empty() {
            return;
        }
        g.reloading = true;
    }
    let base = gfx.borrow().points_base.clone();
    let q = format!("n={budget}");
    let purl = with_query(&base, &q);
    let murl = with_query(&meta_url(&base), &q);
    show_status("loading…");
    let meta = fetch_meta(&murl).await.ok();
    let result = fetch_points(&purl).await;
    gfx.borrow_mut().reloading = false;
    match result {
        Ok(pts) if !pts.is_empty() => {
            apply_load(&gfx, meta, pts);
            show_status("");
        }
        Ok(_) => show_status("load returned no points"),
        Err(e) => show_status(&format!("load failed: {e}")),
    }
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
                wasm_bindgen_futures::spawn_local(reload(gfx.clone(), budget));
            }
        }
    };
    {
        let trigger = trigger.clone();
        add_listener(btn.as_ref(), "click", move |_e: web_sys::Event| trigger());
    }
    add_listener(num.as_ref(), "change", move |_e: web_sys::Event| trigger());
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
    let (auto, rmode, xf, ms, cmap, psize, opac, expo, floor) = {
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
    });

    // Filters
    on_toggle("ms-1", gfx, |g, on| if on { g.params.ms_mask = 0b01; g.filter_dirty = true; });
    on_toggle("ms-2", gfx, |g, on| if on { g.params.ms_mask = 0b10; g.filter_dirty = true; });
    on_toggle("ms-b", gfx, |g, on| if on { g.params.ms_mask = 0b11; g.filter_dirty = true; });
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
