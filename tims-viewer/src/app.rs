//! Application orchestrator: window, wgpu, egui, camera, and the render loop that
//! drains the streaming loader.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::ModifiersState;
use winit::window::{Window, WindowId};

use crate::camera::OrbitCamera;
use crate::data::demo::DemoSource;
use crate::data::loader::{LoadMsg, LoaderHandle, LoaderMode};
use crate::data::meta::MetaIndex;
use crate::render::colormap::COLORMAP_NAMES;
use crate::render::annotation::{AnnotationRenderer, LineVertex};
use crate::render::point_cloud::PointCloudRenderer;
use crate::render::volume::{VolumeGrid, VolumeRenderer, VOLUME_DIMS};
use crate::state::{AppState, ViewMode};
use crate::ui;

const DEFAULT_BUDGET: usize = 12_000_000;

/// How the run will be loaded once the window is up.
pub struct Plan {
    pub meta: MetaIndex,
    pub is_demo: bool,
    pub budget: usize,
}

impl Plan {
    pub fn new(meta: MetaIndex, is_demo: bool, budget: Option<usize>) -> Self {
        Plan {
            meta,
            is_demo,
            budget: budget.unwrap_or(DEFAULT_BUDGET),
        }
    }
}

/// Filter signature for caching the DBSCAN-input count: the three window ranges, the intensity
/// range, the two MS toggles, and the resident point count.
type CountSig = (f64, f64, f64, f64, f64, f64, f64, f64, bool, bool, usize);

/// Everything that exists only once the window/GPU are live.
struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,

    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,

    points: PointCloudRenderer,
    volume: VolumeRenderer,
    /// Display density grid (folded from the MS1/MS2 grids per the MS toggle).
    grid: VolumeGrid,
    /// Per-MS-level density grids, deposited separately so the volume can filter MS1/MS2.
    grid_ms1: VolumeGrid,
    grid_ms2: VolumeGrid,
    /// Last MS toggle the display grid was folded for (to detect changes).
    last_ms: (bool, bool),
    annotations: AnnotationRenderer,
    /// Full annotation geometry + parallel per-vertex group ids, retained so the overlay
    /// can be re-uploaded filtered by the per-group visibility mask without a reload.
    anno_lines: Vec<LineVertex>,
    anno_groups: Vec<u32>,
    last_group_mask: u32,
    /// Wireframe of the full data cube + axis ticks (+ optional back-face grid), drawn for
    /// spatial orientation.
    axes: AnnotationRenderer,
    /// Numeric tick labels for the overlay text pass, rebuilt only when the shown ranges,
    /// focus, or the back-face-grid toggle change (avoids per-frame string allocs).
    tick_labels: Vec<ui::TickLabel>,
    /// Cached (mz, im, rt) real ranges + grid-toggle the `axes` geometry was built for.
    shown_ranges: ((f64, f64), (f64, f64), (f64, f64)),
    shown_grid: bool,
    camera: OrbitCamera,
    state: AppState,
    loader: LoaderHandle,
    /// The full run's metadata, retained so a refined region can revert to the whole run.
    full_meta: MetaIndex,
    /// True for the built-in synthetic DEMO source: respawns must stay in demo mode rather
    /// than trying to open a real `.d` at the placeholder data path.
    is_demo: bool,
    /// CPU copy of the resident points (bounded by GPU capacity) so DBSCAN can run on a
    /// filtered subset and write per-point cluster ids back into the GPU buffer.
    cpu_points: Vec<crate::data::point::GpuPoint>,
    /// Cached filter signature; the filtered DBSCAN-input count is recomputed only when this
    /// changes (avoids rescanning millions of points every frame).
    last_count_sig: Option<CountSig>,

    // interaction
    modifiers: ModifiersState,
    left_down: bool,
    right_down: bool,
    middle_down: bool,
    last_cursor: Option<PhysicalPosition<f64>>,

    last_frame: Instant,
    fps_smooth: f32,
    /// Set when a fatal GPU error (e.g. surface OOM) should end the event loop.
    pending_exit: bool,
    /// Previous frame's view mode, to auto-range the transfer fn on Points->Volume.
    last_view_mode: ViewMode,
}

pub struct App {
    plan: Option<Plan>,
    gfx: Option<Gfx>,
}

impl App {
    pub fn new(plan: Plan) -> Self {
        App {
            plan: Some(plan),
            gfx: None,
        }
    }
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Neutral steel for the cube wireframe, distinct from the window overlay colors.
const FRAME_COLOR: [f32; 3] = [0.45, 0.55, 0.7];
/// Dimmed frame color for the short tick segments.
const TICK_COLOR: [f32; 3] = [0.6, 0.7, 0.85];
/// Very dim back-face gridlines (faintness via low RGB, the pipeline has no alpha blend).
const GRID_COLOR: [f32; 3] = [0.18, 0.21, 0.27];
/// Tick length in cube units (the segments sit just outside the near edges).
const TICK_LEN: f32 = 0.04;

/// The 12 edges of the normalized data cube `[-1, 1]^3` as a line list (pairs = segments),
/// for the orientation wireframe.
fn cube_edges() -> Vec<LineVertex> {
    let c = [-1.0f32, 1.0f32];
    let corner = |i: usize, j: usize, k: usize| LineVertex::new([c[i], c[j], c[k]], FRAME_COLOR);
    let mut v = Vec::with_capacity(24);
    for &j in &[0usize, 1] {
        for &k in &[0usize, 1] {
            v.push(corner(0, j, k));
            v.push(corner(1, j, k)); // edges along x (m/z)
        }
    }
    for &i in &[0usize, 1] {
        for &k in &[0usize, 1] {
            v.push(corner(i, 0, k));
            v.push(corner(i, 1, k)); // edges along y (1/K0)
        }
    }
    for &i in &[0usize, 1] {
        for &j in &[0usize, 1] {
            v.push(corner(i, j, 0));
            v.push(corner(i, j, 1)); // edges along z (RT)
        }
    }
    v
}

/// Build the cube wireframe + short 3D tick segments (+ optional back-face gridlines) and
/// the matching numeric tick labels. `mz/im/rt` are the (lo,hi) real ranges currently shown
/// (window if focus, else bounds). All geometry rides the always-on-top `axes` renderer.
fn axis_geometry(
    mz: (f64, f64),
    im: (f64, f64),
    rt: (f64, f64),
    show_grid: bool,
) -> (Vec<LineVertex>, Vec<ui::TickLabel>) {
    use crate::data::point::AxisTransform;
    use crate::ticks::{fmt_tick, ticks_for, Axis};

    // Axis label colors mirror ui::draw_axis_labels (m/z orange, 1/K0 cyan, RT green).
    const MZ_COL: egui::Color32 = egui::Color32::from_rgb(255, 150, 90);
    const IM_COL: egui::Color32 = egui::Color32::from_rgb(120, 220, 255);
    const RT_COL: egui::Color32 = egui::Color32::from_rgb(150, 235, 150);

    let mut verts = cube_edges();
    let mut labels = Vec::new();

    let tx_mz = AxisTransform::new(mz.0, mz.1);
    let tx_im = AxisTransform::new(im.0, im.1);
    let tx_rt = AxisTransform::new(rt.0, rt.1);
    let mz_ticks = ticks_for(mz.0, mz.1, 5, |v| tx_mz.normalize(v));
    let im_ticks = ticks_for(im.0, im.1, 5, |v| tx_im.normalize(v));
    let rt_ticks = ticks_for(rt.0, rt.1, 5, |v| tx_rt.normalize(v));
    let mz_span = (mz.1 - mz.0).abs();
    let im_span = (im.1 - im.0).abs();
    let rt_span = (rt.1 - rt.0).abs();

    let mut seg = |a: [f32; 3], b: [f32; 3], color: [f32; 3]| {
        verts.push(LineVertex::new(a, color));
        verts.push(LineVertex::new(b, color));
    };

    // m/z (x) axis: edge at y=-1, z=-1; tick + label extend in -y (just outside).
    for t in &mz_ticks {
        let n = t.norm;
        seg([n, -1.0, -1.0], [n, -1.0 - TICK_LEN, -1.0], TICK_COLOR);
        labels.push(ui::TickLabel {
            world: glam::vec3(n, -1.0 - 2.2 * TICK_LEN, -1.0),
            text: fmt_tick(Axis::Mz, t.value, mz_span),
            axis_color: MZ_COL,
        });
    }
    // 1/K0 (y) axis: edge at x=-1, z=-1; tick + label extend in -x.
    for t in &im_ticks {
        let n = t.norm;
        seg([-1.0, n, -1.0], [-1.0 - TICK_LEN, n, -1.0], TICK_COLOR);
        labels.push(ui::TickLabel {
            world: glam::vec3(-1.0 - 2.2 * TICK_LEN, n, -1.0),
            text: fmt_tick(Axis::Im, t.value, im_span),
            axis_color: IM_COL,
        });
    }
    // RT (z) axis: edge at x=-1, y=-1; tick + label extend in -y (reads against the floor).
    for t in &rt_ticks {
        let n = t.norm;
        seg([-1.0, -1.0, n], [-1.0, -1.0 - TICK_LEN, n], TICK_COLOR);
        labels.push(ui::TickLabel {
            world: glam::vec3(-1.0, -1.0 - 2.2 * TICK_LEN, n),
            text: fmt_tick(Axis::Rt, t.value, rt_span),
            axis_color: RT_COL,
        });
    }

    // Optional faint gridlines on the three faces meeting at the max-corner (+1,+1,+1),
    // which sit "behind" the cloud for the default orbit. Skip near-edge ticks so the
    // gridlines don't double the cube wireframe.
    if show_grid {
        let interior = |n: f32| n > -0.999 && n < 0.999;
        // Face z=+1 (m/z × 1/K0 plane).
        for t in &mz_ticks {
            if interior(t.norm) {
                seg([t.norm, -1.0, 1.0], [t.norm, 1.0, 1.0], GRID_COLOR);
            }
        }
        for t in &im_ticks {
            if interior(t.norm) {
                seg([-1.0, t.norm, 1.0], [1.0, t.norm, 1.0], GRID_COLOR);
            }
        }
        // Face y=+1 (m/z × RT plane).
        for t in &mz_ticks {
            if interior(t.norm) {
                seg([t.norm, 1.0, -1.0], [t.norm, 1.0, 1.0], GRID_COLOR);
            }
        }
        for t in &rt_ticks {
            if interior(t.norm) {
                seg([-1.0, 1.0, t.norm], [1.0, 1.0, t.norm], GRID_COLOR);
            }
        }
        // Face x=+1 (1/K0 × RT plane).
        for t in &im_ticks {
            if interior(t.norm) {
                seg([1.0, t.norm, -1.0], [1.0, t.norm, 1.0], GRID_COLOR);
            }
        }
        for t in &rt_ticks {
            if interior(t.norm) {
                seg([1.0, -1.0, t.norm], [1.0, 1.0, t.norm], GRID_COLOR);
            }
        }
    }

    (verts, labels)
}

/// Render a 2D projection (row-major `x + PROJ_BINS*y` counts) into an egui image: log-scaled
/// through inferno, with the vertical axis flipped so its low end sits at the image bottom.
fn proj_to_color_image(data: &[u32]) -> egui::ColorImage {
    let n = crate::data::loader::PROJ_BINS;
    let maxc = data.iter().copied().max().unwrap_or(1).max(1) as f32;
    let lmax = (maxc + 1.0).ln();
    let mut pixels = vec![egui::Color32::from_gray(16); n * n];
    for y in 0..n {
        for x in 0..n {
            let c = data[x + n * y];
            if c > 0 {
                let t = ((c as f32 + 1.0).ln() / lmax).clamp(0.0, 1.0);
                let rgb = crate::render::colormap::sample(1, t); // inferno
                pixels[x + n * (n - 1 - y)] = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            }
        }
    }
    egui::ColorImage { size: [n, n], pixels }
}

/// Whether a point passes the active filter (normalized window + intensity range + MS mask).
/// Mirrors the point-cloud shader cull so the DBSCAN input matches what is displayed.
#[inline]
fn point_passes(
    p: &crate::data::point::GpuPoint,
    fmin: [f32; 3],
    fmax: [f32; 3],
    irange: (f32, f32),
    ms_mask: u32,
) -> bool {
    let pos = p.pos;
    if pos[0] < fmin[0] || pos[0] > fmax[0] || pos[1] < fmin[1] || pos[1] > fmax[1]
        || pos[2] < fmin[2] || pos[2] > fmax[2]
    {
        return false;
    }
    if p.intensity < irange.0 || p.intensity > irange.1 {
        return false;
    }
    let is_ms2 = p.flags & crate::data::point::GpuPoint::MS2_FLAG != 0;
    if is_ms2 {
        ms_mask & 2 != 0
    } else {
        ms_mask & 1 != 0
    }
}

fn create_depth(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

impl App {
    fn init_gfx(&mut self, event_loop: &ActiveEventLoop) -> Result<Gfx> {
        let plan = self.plan.take().expect("plan already consumed");

        let attrs = Window::default_attributes()
            .with_title("tims-viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0));
        let window = Arc::new(event_loop.create_window(attrs)?);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY | wgpu::Backends::GL,
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone())?;

        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            },
        ))
        .ok_or_else(|| anyhow::anyhow!("no suitable GPU adapter found"))?;

        log::info!("adapter: {:?}", adapter.get_info());
        let limits = adapter.limits();
        // GPU compaction needs compute + indirect execution; fall back to draw-all
        // (vertex-shader culling) on backends that lack them (e.g. some GL drivers).
        let dl_flags = adapter.get_downlevel_capabilities().flags;
        let supports_compaction = dl_flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            && dl_flags.contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION);
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("tims-viewer-device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        let depth_view = create_depth(&device, &config);

        // Derive a safe capacity from device limits and the requested budget.
        let bytes_per_point = std::mem::size_of::<crate::data::point::GpuPoint>() as u64;
        let max_by_buffer = (limits.max_buffer_size / bytes_per_point) as usize;
        let total = plan.meta.total_points_estimate;
        let mut cap = plan.budget.min(max_by_buffer).min((total as usize).max(1));
        // When compaction is on, the master + compacted buffers are bound as storage,
        // which is capped by max_storage_buffer_binding_size (often much smaller than
        // max_buffer_size, e.g. 128 MiB) — exceeding it fails bind-group validation.
        if supports_compaction {
            let max_by_storage =
                (limits.max_storage_buffer_binding_size as u64 / bytes_per_point) as usize;
            cap = cap.min(max_by_storage);
        }
        let capacity = cap.max(1) as u32;
        log::info!(
            "point budget={} capacity={} compaction={} (device max_buffer_size={} MiB)",
            plan.budget,
            capacity,
            supports_compaction,
            limits.max_buffer_size / (1024 * 1024)
        );

        let points = PointCloudRenderer::new(
            &device,
            &queue,
            format,
            DEPTH_FORMAT,
            capacity,
            supports_compaction,
        );
        let volume = VolumeRenderer::new(&device, &queue, format, DEPTH_FORMAT, VOLUME_DIMS);
        let grid = VolumeGrid::new(VOLUME_DIMS);
        let grid_ms1 = VolumeGrid::new(VOLUME_DIMS);
        let grid_ms2 = VolumeGrid::new(VOLUME_DIMS);
        let annotations = AnnotationRenderer::new(&device, format, DEPTH_FORMAT);

        let n_colormaps = COLORMAP_NAMES.len() as u32;
        let mut state = AppState::new(plan.meta.bounds, total, n_colormaps);
        state.capacity = capacity;
        state.downsample_stride =
            crate::data::loader::stride_for(total, capacity as usize) as u32;

        // Data-cube wireframe + axis ticks for orientation, with the clip window opened past
        // the cube so the frame and the slightly-out-of-cube ticks are never culled.
        let init_ranges = (
            (plan.meta.bounds.mz.min, plan.meta.bounds.mz.max),
            (plan.meta.bounds.im.min, plan.meta.bounds.im.max),
            (plan.meta.bounds.rt.min, plan.meta.bounds.rt.max),
        );
        let mut axes = AnnotationRenderer::new(&device, format, DEPTH_FORMAT);
        let (init_verts, init_labels) = axis_geometry(
            init_ranges.0,
            init_ranges.1,
            init_ranges.2,
            state.show_grid_backfaces,
        );
        axes.upload(&device, &init_verts);
        // The axis frame outlines the cube itself, so it never re-fits (focus = 0).
        axes.update_filter(&queue, [-2.0, -2.0, -2.0, 0.0], [2.0, 2.0, 2.0, 0.0], 0.0);

        // egui plumbing.
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, true);

        // Keep the full run's metadata so a refined region can revert to the whole run.
        let full_meta = plan.meta.clone();

        // Spawn the loader.
        let mode = if plan.is_demo {
            LoaderMode::Demo(DemoSource::new(plan.meta.frames.len(), total))
        } else {
            let frame_ids = plan.meta.frames.iter().map(|f| f.id).collect();
            LoaderMode::Real {
                path: plan.meta.data_path.clone(),
                frame_ids,
            }
        };
        let loader = LoaderHandle::spawn(mode, plan.meta.bounds, total, capacity as usize, None);

        Ok(Gfx {
            window,
            surface,
            device,
            queue,
            config,
            depth_view,
            egui_ctx,
            egui_state,
            egui_renderer,
            points,
            volume,
            grid,
            grid_ms1,
            grid_ms2,
            last_ms: (true, true),
            annotations,
            anno_lines: Vec::new(),
            anno_groups: Vec::new(),
            last_group_mask: u32::MAX,
            axes,
            tick_labels: init_labels,
            shown_ranges: init_ranges,
            shown_grid: state.show_grid_backfaces,
            camera: OrbitCamera::default(),
            state,
            loader,
            full_meta,
            is_demo: plan.is_demo,
            cpu_points: Vec::new(),
            last_count_sig: None,
            modifiers: ModifiersState::empty(),
            left_down: false,
            right_down: false,
            middle_down: false,
            last_cursor: None,
            last_frame: Instant::now(),
            fps_smooth: 0.0,
            pending_exit: false,
            last_view_mode: ViewMode::Points,
        })
    }
}

impl Gfx {
    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth(&self.device, &self.config);
    }

    /// Drain the loader channel, bounding work per frame.
    fn pump_loader(&mut self) {
        // Cap chunks per frame so a fast loader can't stall the UI with uploads.
        const MAX_CHUNKS_PER_FRAME: usize = 16;
        for _ in 0..MAX_CHUNKS_PER_FRAME {
            match self.loader.rx.try_recv() {
                Ok(LoadMsg::Chunk { points, .. }) => {
                    // Volume density: deposit intensity*weight so the density is
                    // independent of the downsample ratio (weight = 1/p = stride). Deposit
                    // into the per-MS-level grid so the volume can filter MS1/MS2 like points.
                    for p in &points {
                        let g = if p.flags & crate::data::point::GpuPoint::MS2_FLAG != 0 {
                            &mut self.grid_ms2
                        } else {
                            &mut self.grid_ms1
                        };
                        g.deposit(p.pos, p.intensity * p.weight);
                    }
                    self.points.append(&self.queue, &points);
                    // Retain a CPU copy of every resident point (bounded by GPU capacity) so
                    // DBSCAN can run on any FILTERED subset — you filter the region down to a
                    // clusterable count instead of shrinking the spatial window.
                    let cap = self.state.capacity as usize;
                    if self.cpu_points.len() < cap {
                        let room = cap - self.cpu_points.len();
                        self.cpu_points.extend(points.iter().take(room).copied());
                    }
                }
                Ok(LoadMsg::PeakChunk { points }) => {
                    // Peaks are display-only enrichment; append to the cloud but do NOT
                    // add to the volume density (would double-count dense cells).
                    self.points.append(&self.queue, &points);
                    // Retain a CPU copy of every resident point (bounded by GPU capacity) so
                    // DBSCAN can run on any FILTERED subset — you filter the region down to a
                    // clusterable count instead of shrinking the spatial window.
                    let cap = self.state.capacity as usize;
                    if self.cpu_points.len() < cap {
                        let room = cap - self.cpu_points.len();
                        self.cpu_points.extend(points.iter().take(room).copied());
                    }
                }
                Ok(LoadMsg::Progress(p)) => self.state.load_progress = p,
                Ok(LoadMsg::Stats { i_min, i_max, i_med }) => {
                    // Stash the raw, mode-independent percentiles so any later mode switch
                    // or Auto button can re-run the heuristic from scratch.
                    self.state.i_p1 = i_min;
                    self.state.i_med = i_med;
                    self.state.i_p99 = i_max;
                    // Auto-expose the point cloud the moment the range is known (the cloud
                    // snaps legible without slider tweaks). Volume re-ranges on Done.
                    if self.state.view_mode == ViewMode::Points && self.state.may_auto_transfer()
                    {
                        self.state.auto_transfer_points();
                    }
                }
                Ok(LoadMsg::Histograms {
                    mz,
                    im,
                    rt,
                    intensity,
                    i_lo,
                    i_hi,
                    proj_mz_im,
                    proj_mz_rt,
                    proj_im_rt,
                }) => {
                    // Distribution strips for the levels-style filters; the intensity slider
                    // spans the data range, default filter = full (no cull).
                    self.state.hist_mz = mz;
                    self.state.hist_im = im;
                    self.state.hist_rt = rt;
                    self.state.hist_intensity = intensity;
                    self.state.i_data_lo = i_lo;
                    self.state.i_data_hi = i_hi;
                    self.state.intensity_window = crate::state::Window {
                        min: i_lo as f64,
                        max: i_hi as f64,
                    };
                    // Upload the 2D projection minimaps as egui textures.
                    let opt = egui::TextureOptions::LINEAR;
                    self.state.proj_mz_im = Some(self.egui_ctx.load_texture(
                        "proj_mz_im",
                        proj_to_color_image(&proj_mz_im),
                        opt,
                    ));
                    self.state.proj_mz_rt = Some(self.egui_ctx.load_texture(
                        "proj_mz_rt",
                        proj_to_color_image(&proj_mz_rt),
                        opt,
                    ));
                    self.state.proj_im_rt = Some(self.egui_ctx.load_texture(
                        "proj_im_rt",
                        proj_to_color_image(&proj_im_rt),
                        opt,
                    ));
                }
                Ok(LoadMsg::Annotations { lines, groups, n_groups }) => {
                    // Show the selection overlay in refined regions too: it is normalized to the
                    // current bounds, and the per-vertex window cull clips off-region boxes to
                    // the cube, so only the windows overlapping the region remain.
                    self.anno_lines = lines;
                    self.anno_groups = groups;
                    self.state.n_window_groups = n_groups;
                    self.reupload_annotations();
                }
                Ok(LoadMsg::Done { .. }) => {
                    self.state.load_progress = 1.0;
                    // Recompute the volume range against the now-complete grid.
                    if self.state.view_mode == ViewMode::Volume && self.state.may_auto_transfer()
                    {
                        self.refresh_volume_grid();
                        let (lo, hi) = self.grid.density_percentiles();
                        self.state.auto_transfer_volume(lo, hi);
                    }
                }
                Ok(LoadMsg::Error(e)) => {
                    log::error!("loader error: {e}");
                    self.state.load_failed = true;
                    self.state.load_progress = 1.0;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Normal completion drops the sender after Done (progress==1.0).
                    // A disconnect before that means the loader thread died (panic).
                    if self.state.load_progress < 1.0 {
                        log::error!("loader thread ended before completing the load");
                        self.state.load_failed = true;
                        self.state.load_progress = 1.0;
                    }
                    break;
                }
            }
        }
        self.state.resident_points = self.points.resident();
    }

    /// Fold the MS1/MS2 density grids into the display grid per the MS toggle, when the
    /// grids or the toggle changed. Returns true if the MS toggle itself changed (so the
    /// caller can re-range the transfer function).
    fn refresh_volume_grid(&mut self) -> bool {
        let ms = (self.state.show_ms1, self.state.show_ms2);
        let ms_changed = ms != self.last_ms;
        if !self.grid_ms1.dirty() && !self.grid_ms2.dirty() && !ms_changed {
            return false;
        }
        self.grid.combine(
            &self.grid_ms1,
            &self.grid_ms2,
            ms.0 as u8 as f32,
            ms.1 as u8 as f32,
        );
        self.grid_ms1.clear_dirty();
        self.grid_ms2.clear_dirty();
        self.last_ms = ms;
        ms_changed
    }

    /// Re-upload the selection overlay, keeping only vertices whose window group is enabled
    /// in `state.group_mask` (ungrouped vertices, u32::MAX, are always kept).
    fn reupload_annotations(&mut self) {
        let mask = self.state.group_mask;
        // Ungrouped (u32::MAX), an out-of-range group id (0, or > 32 for the 32-bit mask)
        // are always shown; otherwise gate on the group's bit (groups are 1-based).
        let visible = |g: u32| g == u32::MAX || g == 0 || g > 32 || (mask & (1u32 << (g - 1))) != 0;
        let filtered: Vec<LineVertex> = self
            .anno_lines
            .iter()
            .zip(self.anno_groups.iter())
            .filter(|(_, &g)| visible(g))
            .map(|(v, _)| *v)
            .collect();
        self.annotations.upload(&self.device, &filtered);
        self.last_group_mask = mask;
    }

    /// Rebuild the axis tick geometry + numeric labels when the shown ranges (focus toggle
    /// or a window slider moved while focused) or the back-face-grid toggle change. Mirrors
    /// the focus branch of `ui::draw_axis_labels`.
    fn refresh_axis_geometry(&mut self) {
        let ranges = if self.state.focus {
            (
                (self.state.mz_window.min, self.state.mz_window.max),
                (self.state.im_window.min, self.state.im_window.max),
                (self.state.rt_window.min, self.state.rt_window.max),
            )
        } else {
            (
                (self.state.bounds.mz.min, self.state.bounds.mz.max),
                (self.state.bounds.im.min, self.state.bounds.im.max),
                (self.state.bounds.rt.min, self.state.bounds.rt.max),
            )
        };
        let grid = self.state.show_grid_backfaces;
        // Epsilon compare so float jitter on the sliders doesn't rebuild every frame.
        let close = |a: (f64, f64), b: (f64, f64)| {
            (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6
        };
        let unchanged = grid == self.shown_grid
            && close(ranges.0, self.shown_ranges.0)
            && close(ranges.1, self.shown_ranges.1)
            && close(ranges.2, self.shown_ranges.2);
        if unchanged {
            return;
        }
        let (verts, labels) = axis_geometry(ranges.0, ranges.1, ranges.2, grid);
        self.axes.upload(&self.device, &verts);
        self.tick_labels = labels;
        self.shown_ranges = ranges;
        self.shown_grid = grid;
    }

    /// Re-spawn the loader over `frame_ids` with `bounds`, `total` estimate and an optional
    /// per-point region filter, resetting the point buffer, volume grids and load/transfer
    /// state so the new selection streams in from scratch. Shared by refine and revert.
    fn respawn_loader(
        &mut self,
        frame_ids: Vec<u32>,
        bounds: crate::data::point::AxisBounds,
        total: u64,
        filter: Option<crate::data::loader::RegionFilter>,
    ) {
        let capacity = self.state.capacity as usize;
        // Stay in the source's mode: a DEMO session must respawn a synthetic source, not try
        // to open a real `.d` at the placeholder path.
        let mode = if self.is_demo {
            LoaderMode::Demo(DemoSource::new(frame_ids.len(), total))
        } else {
            LoaderMode::Real {
                path: self.full_meta.data_path.clone(),
                frame_ids,
            }
        };
        self.loader = LoaderHandle::spawn(mode, bounds, total, capacity, filter);
        // Reset GPU/CPU buffers for a fresh stream.
        self.points.reset();
        self.cpu_points.clear();
        // Drop stale distributions; the new load re-sends them and resets the filter.
        self.state.hist_mz.clear();
        self.state.hist_im.clear();
        self.state.hist_rt.clear();
        self.state.hist_intensity.clear();
        self.state.proj_mz_im = None;
        self.state.proj_mz_rt = None;
        self.state.proj_im_rt = None;
        self.state.intensity_window = crate::state::Window {
            min: 0.0,
            max: f64::INFINITY,
        };
        self.state.color_mode = crate::state::ColorMode::Intensity;
        self.state.cluster_count = 0;
        self.state.cluster_noise = 0;
        self.grid_ms1.clear();
        self.grid_ms2.clear();
        self.grid.clear();
        self.last_ms = (true, true);
        self.anno_lines.clear();
        self.anno_groups.clear();
        self.annotations.upload(&self.device, &[]);
        self.last_group_mask = u32::MAX;
        // Reset per-load state; re-derive the cube + transfer for the new bounds.
        self.state.bounds = bounds;
        self.state.reset_windows();
        self.state.focus = false;
        self.state.transfer_user_dirty = false;
        self.state.n_window_groups = 0;
        self.state.group_mask = u32::MAX;
        self.state.load_progress = 0.0;
        self.state.load_failed = false;
        self.state.resident_points = 0;
        self.state.downsample_stride = crate::data::loader::stride_for(total, capacity) as u32;
        // Force a transfer re-range next frame and an axis-geometry rebuild (NaN never matches).
        self.last_view_mode = ViewMode::Points;
        self.shown_ranges = ((f64::NAN, 0.0), (0.0, 0.0), (0.0, 0.0));
    }

    /// Re-stream just the current RT/m-z/1-K0 window at full resolution. Frame selection
    /// handles the RT window; a per-point m/z·1/K0 cull handles the rest, and the window
    /// becomes the new cube — so the fixed budget now buys (often stride-1) detail there.
    fn refine_to_window(&mut self) {
        use crate::data::point::{AxisBounds, AxisTransform};
        let (rt0, rt1) = (self.state.rt_window.min, self.state.rt_window.max);
        let (mz0, mz1) = (self.state.mz_window.min, self.state.mz_window.max);
        let (im0, im1) = (self.state.im_window.min, self.state.im_window.max);
        // A zero-width axis would make the new AxisTransform divide by zero (NaN positions).
        if rt1 <= rt0 || mz1 <= mz0 || im1 <= im0 {
            return;
        }
        let mut frame_ids = Vec::new();
        let mut total: u64 = 0;
        for f in &self.full_meta.frames {
            if f.retention_time >= rt0 && f.retention_time <= rt1 {
                frame_ids.push(f.id);
                total = total.saturating_add(f.num_peaks);
            }
        }
        if frame_ids.is_empty() {
            return;
        }
        // num_peaks counts every m/z·1/K0 in each in-RT frame, but the loader culls to the
        // m/z·1/K0 window. Scale the estimate by the window's fraction of the full run so the
        // stride is sized to the surviving points — otherwise it is far too coarse and the
        // region streams sparse instead of the intended (often stride-1) detail.
        let b = &self.full_meta.bounds;
        let mz_frac = ((mz1 - mz0) / (b.mz.max - b.mz.min).max(1e-9)).clamp(0.0, 1.0);
        let im_frac = ((im1 - im0) / (b.im.max - b.im.min).max(1e-9)).clamp(0.0, 1.0);
        let total = (((total as f64) * mz_frac * im_frac).ceil() as u64).max(1);
        let bounds = AxisBounds {
            mz: AxisTransform::new(mz0, mz1),
            im: AxisTransform::new(im0, im1),
            rt: AxisTransform::new(rt0, rt1),
        };
        let filter = Some(crate::data::loader::RegionFilter {
            mz: (mz0, mz1),
            im: (im0, im1),
            intensity_min: 0.0, // native region refinement does not cull intensity at the source
        });
        self.respawn_loader(frame_ids, bounds, total, filter);
        self.state.refined = true;
    }

    /// The active point filter, mirroring the shader cull: normalized window min/max, the
    /// intensity range (real units), and the MS bit mask.
    fn active_filter(&self) -> ([f32; 3], [f32; 3], (f32, f32), u32) {
        let s = &self.state;
        let fmin = s
            .bounds
            .normalize(s.mz_window.min, s.im_window.min, s.rt_window.min);
        let fmax = s
            .bounds
            .normalize(s.mz_window.max, s.im_window.max, s.rt_window.max);
        let ms_mask = (s.show_ms1 as u32) | ((s.show_ms2 as u32) << 1);
        (
            fmin,
            fmax,
            (s.intensity_window.min as f32, s.intensity_window.max as f32),
            ms_mask,
        )
    }

    /// Signature of everything the filtered count depends on; used to skip the rescan when
    /// nothing changed.
    fn count_sig(&self) -> CountSig {
        let s = &self.state;
        (
            s.mz_window.min,
            s.mz_window.max,
            s.im_window.min,
            s.im_window.max,
            s.rt_window.min,
            s.rt_window.max,
            s.intensity_window.min,
            s.intensity_window.max,
            s.show_ms1,
            s.show_ms2,
            self.cpu_points.len(),
        )
    }

    /// Count the resident points that pass the active filter (the DBSCAN input size). Parallel
    /// for speed; called only when the filter signature changes (see `count_sig`).
    fn count_filtered(&self) -> usize {
        use rayon::prelude::*;
        let (fmin, fmax, irange, ms) = self.active_filter();
        self.cpu_points
            .par_iter()
            .filter(|p| point_passes(p, fmin, fmax, irange, ms))
            .count()
    }

    /// Run DBSCAN on the FILTERED resident points (intensity + window + MS filters), write per-
    /// point cluster ids into the GPU buffer, and switch to cluster coloring. Filtered-out points
    /// are cleared to NO_CLUSTER (the shader culls them anyway). No-op while streaming, with no
    /// filtered points, or with more than CLUSTER_CAP in the filter (tighten the filters).
    fn run_clustering(&mut self) {
        if self.state.load_progress < 1.0
            || self.cpu_points.is_empty()
            || self.cpu_points.len() as u32 != self.state.resident_points
        {
            return; // still streaming or partial retention
        }
        let (fmin, fmax, irange, ms) = self.active_filter();
        let mut idx: Vec<usize> = Vec::new();
        let mut positions: Vec<[f32; 3]> = Vec::new();
        for (i, p) in self.cpu_points.iter().enumerate() {
            if point_passes(p, fmin, fmax, irange, ms) {
                idx.push(i);
                positions.push(p.pos);
            }
        }
        if positions.is_empty() || positions.len() > crate::state::CLUSTER_CAP as usize {
            return; // nothing in the filter, or still too many — tighten the filters
        }
        let (labels, k) = crate::cluster::dbscan(
            &positions,
            self.state.cluster_eps,
            self.state.cluster_min_pts as usize,
        );
        for p in self.cpu_points.iter_mut() {
            p._pad[0] = crate::data::point::GpuPoint::NO_CLUSTER;
        }
        let mut noise = 0usize;
        for (j, &pi) in idx.iter().enumerate() {
            self.cpu_points[pi]._pad[0] = if labels[j] < 0 {
                noise += 1;
                crate::data::point::GpuPoint::NO_CLUSTER
            } else {
                labels[j] as u32
            };
        }
        self.points.reset();
        self.points.append(&self.queue, &self.cpu_points);
        self.state.resident_points = self.points.resident();
        self.state.cluster_count = k;
        self.state.cluster_noise = noise;
        self.state.cluster_input_count = positions.len();
        self.state.color_mode = crate::state::ColorMode::Cluster;
        self.state.view_mode = ViewMode::Points;
        // Cluster coloring renders opaque in the shader regardless of point_mode, so the
        // user's additive/opaque choice is left untouched.
    }

    /// Revert a refinement: re-stream the whole run at the global downsample.
    fn load_full_run(&mut self) {
        let frame_ids: Vec<u32> = self.full_meta.frames.iter().map(|f| f.id).collect();
        let bounds = self.full_meta.bounds;
        let total = self.full_meta.total_points_estimate;
        self.respawn_loader(frame_ids, bounds, total, None);
        self.state.refined = false;
    }

    fn render(&mut self) {
        self.pump_loader();
        // Level-of-detail: consume any pending refine/revert request from the UI.
        if let Some(action) = self.state.refine_request.take() {
            match action {
                crate::state::RefineAction::Refine => self.refine_to_window(),
                crate::state::RefineAction::FullRun => self.load_full_run(),
            }
        }
        // Live DBSCAN input size = points passing the active filter (so the Cluster gate +
        // readout react as you drag the sliders). Recompute only when the filter or point set
        // changes — rescanning millions of points every frame would stall the render loop.
        if self.state.load_progress >= 1.0 {
            let sig = self.count_sig();
            if self.last_count_sig != Some(sig) {
                self.state.cluster_input_count = self.count_filtered();
                self.last_count_sig = Some(sig);
            }
        }
        // Clustering: run DBSCAN on the filtered resident points and color by cluster id.
        if self.state.cluster_request {
            self.state.cluster_request = false;
            self.run_clustering();
        }
        // Re-range when switching Volume->Points. The two modes live in completely different
        // value ranges (per-point intensity vs summed density), so a transfer range pinned in
        // one mode is meaningless in the other: always re-range on the switch and clear the
        // manual-edit latch so the new mode starts from a sensible auto view.
        if self.last_view_mode == ViewMode::Volume && self.state.view_mode == ViewMode::Points {
            self.state.reset_transfer_auto((0.0, 0.0));
        }
        // Re-filter the selection overlay if the per-group visibility changed.
        if self.state.group_mask != self.last_group_mask {
            self.reupload_annotations();
        }

        // FPS (exponential smoothing).
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().max(1e-4);
        self.last_frame = now;
        let inst = 1.0 / dt;
        self.fps_smooth = if self.fps_smooth == 0.0 {
            inst
        } else {
            self.fps_smooth * 0.9 + inst * 0.1
        };
        self.state.fps = self.fps_smooth;

        // Acquire the surface BEFORE any egui texture upload, so a surface error
        // returns without leaking uploaded-but-never-freed egui textures (texture
        // `set` and `free` are then always paired within one successful frame).
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Err(wgpu::SurfaceError::Timeout) => return,
            Err(wgpu::SurfaceError::OutOfMemory) => {
                log::error!("surface out of memory; exiting");
                self.pending_exit = true;
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // egui frame. The "Auto" transfer button needs the volume density percentiles, but
        // computing them scans + sorts every non-zero voxel (~6M), so pass a CLOSURE that
        // ui::build invokes only on click — never on the per-frame render path.
        let raw_input = self.egui_state.take_egui_input(self.window.as_ref());
        let tick_labels = &self.tick_labels;
        let grid = &self.grid;
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            ui::build(ctx, &mut self.state, &mut self.camera, tick_labels, || {
                grid.density_percentiles()
            });
        });
        self.egui_state
            .handle_platform_output(self.window.as_ref(), full_output.platform_output);
        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }

        // Rebuild axis ticks/labels (and back-face grid) AFTER ui::build, so a same-frame
        // toggle of "Focus to window", a window slider, or the back-face grid is reflected
        // by the tick geometry in the SAME frame the point cloud / volume box snaps to it
        // (params() below also reads the post-build state). Mirrors ui::draw_axis_labels.
        self.refresh_axis_geometry();

        // Uniforms — computed AFTER egui so they reflect this frame's UI/camera changes
        // (otherwise a Points->Volume switch renders one frame with a stale matrix and
        // an un-uploaded texture).
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let cam = self.camera.to_uniform(
            aspect,
            [self.config.width as f32, self.config.height as f32],
        );
        let params = self.state.params();
        self.points.update_camera(&self.queue, &cam);
        self.points.update_params(&self.queue, &params);
        self.annotations.update_camera(&self.queue, &cam);
        self.annotations
            .update_filter(&self.queue, params.filter_min, params.filter_max, params.focus);
        self.axes.update_camera(&self.queue, &cam);

        // Volume mode: update the raycaster uniform (incl. the density scale) and
        // (re)upload the grid if it grew.
        if self.state.view_mode == ViewMode::Volume {
            // Fold MS1/MS2 density per the toggle (rebuilds the display grid if changed).
            let ms_changed = self.refresh_volume_grid();
            // Entering Volume always re-ranges to the density scale and clears the manual-edit
            // latch (a range pinned in Points mode is meaningless for density — this is what
            // made the volume render flat after a Points-mode transfer tweak). An MS toggle
            // within Volume re-ranges too, but respects a manual pin set in Volume mode.
            let entering_volume = self.last_view_mode != ViewMode::Volume;
            if entering_volume {
                let (lo, hi) = self.grid.density_percentiles();
                self.state.reset_transfer_auto((lo, hi));
            } else if ms_changed && self.state.may_auto_transfer() {
                let (lo, hi) = self.grid.density_percentiles();
                self.state.auto_transfer_volume(lo, hi);
            }
            let inv_vp = self.camera.inv_view_proj(aspect);
            let mut vu = self.state.volume_uniform(inv_vp);
            vu.density_scale = self.grid.density_scale();
            self.volume.update_uniform(&self.queue, &vu);
            if self.grid.dirty() {
                self.volume.upload(&self.queue, self.grid.to_f16_scaled());
                self.grid.clear_dirty();
            }
        }
        self.last_view_mode = self.state.view_mode;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-encoder"),
            });

        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        let user_bufs =
            self.egui_renderer
                .update_buffers(&self.device, &self.queue, &mut encoder, &tris, &screen_desc);

        // Cull + compact visible points (compute) before the scene pass (points mode only).
        if self.state.view_mode == ViewMode::Points {
            self.points.prepare(&self.queue, &mut encoder);
        }

        // 3D pass.
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.035,
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
            match self.state.view_mode {
                ViewMode::Points => {
                    // Cluster coloring must use the opaque pipeline (REPLACE blend + depth
                    // write): the additive pipeline would sum cluster hues regardless of the
                    // fragment's alpha. Don't mutate point_mode, so the user's additive/opaque
                    // choice is restored when they switch back to intensity coloring.
                    let mode = if self.state.color_mode == crate::state::ColorMode::Cluster {
                        crate::render::point_cloud::PointMode::StructuralOpaque
                    } else {
                        self.state.point_mode
                    };
                    self.points.render(&mut rpass, mode);
                }
                ViewMode::Volume => self.volume.render(&mut rpass),
            }
            if self.state.show_axes {
                self.axes.render(&mut rpass);
            }
            if self.state.show_annotations {
                self.annotations.render(&mut rpass);
            }
        }

        // egui pass (load existing color, no depth).
        {
            let mut rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut rpass, &tris, &screen_desc);
        }

        self.queue
            .submit(user_bufs.into_iter().chain(std::iter::once(encoder.finish())));
        frame.present();

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }

    fn handle_cursor(&mut self, pos: PhysicalPosition<f64>) {
        if let Some(last) = self.last_cursor {
            let dx = (pos.x - last.x) as f32;
            let dy = (pos.y - last.y) as f32;
            let shift = self.modifiers.shift_key();
            if self.middle_down || self.right_down || (self.left_down && shift) {
                self.camera.pan(dx, dy);
            } else if self.left_down {
                self.camera.orbit(dx, dy);
            }
        }
        self.last_cursor = Some(pos);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        match self.init_gfx(event_loop) {
            Ok(gfx) => self.gfx = Some(gfx),
            Err(e) => {
                log::error!("failed to initialize graphics: {e:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(gfx) = self.gfx.as_mut() else {
            return;
        };

        // egui gets first dibs.
        let resp = gfx.egui_state.on_window_event(gfx.window.as_ref(), &event);
        if resp.repaint {
            gfx.window.request_redraw();
        }
        let consumed = resp.consumed;

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => gfx.resize(size.width, size.height),
            WindowEvent::ModifiersChanged(m) => gfx.modifiers = m.state(),
            WindowEvent::RedrawRequested => gfx.render(),
            // Handle button state BEFORE the consumed-guard: a release that egui
            // consumes (drag ended over a panel) must still clear the button, or the
            // camera keeps orbiting on the next move. Presses only start a drag when
            // egui did not consume them (so dragging a slider doesn't orbit).
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                let set = if down { !consumed } else { false };
                match button {
                    MouseButton::Left => gfx.left_down = set,
                    MouseButton::Right => gfx.right_down = set,
                    MouseButton::Middle => gfx.middle_down = set,
                    _ => {}
                }
                if !down {
                    gfx.last_cursor = None;
                }
            }
            _ if consumed => {
                // egui handled it; don't drive the camera.
                gfx.last_cursor = None;
            }
            WindowEvent::CursorMoved { position, .. } => gfx.handle_cursor(position),
            WindowEvent::CursorLeft { .. } => gfx.last_cursor = None,
            WindowEvent::MouseWheel { delta, .. } => {
                let s = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 50.0,
                };
                gfx.camera.zoom(s);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(gfx) = self.gfx.as_ref() {
            if gfx.pending_exit {
                event_loop.exit();
                return;
            }
            // Drive continuous redraws so streaming data animates in.
            gfx.window.request_redraw();
        }
    }
}
