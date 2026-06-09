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
    grid: VolumeGrid,
    camera: OrbitCamera,
    state: AppState,
    loader: LoaderHandle,

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

        let n_colormaps = COLORMAP_NAMES.len() as u32;
        let mut state = AppState::new(plan.meta.bounds, total, n_colormaps);
        state.capacity = capacity;
        state.downsample_stride =
            crate::data::loader::stride_for(total, capacity as usize) as u32;

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
        let loader = LoaderHandle::spawn(mode, plan.meta.bounds, total, capacity as usize);

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
            camera: OrbitCamera::default(),
            state,
            loader,
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
                    // Voxelize into the CPU max-grid (for volume mode) and upload to GPU.
                    for p in &points {
                        self.grid.deposit(p.pos, p.intensity);
                    }
                    self.points.append(&self.queue, &points);
                }
                Ok(LoadMsg::Progress(p)) => self.state.load_progress = p,
                Ok(LoadMsg::Stats { i_min, i_max }) => {
                    self.state.i_min = i_min;
                    self.state.i_max = i_max;
                }
                Ok(LoadMsg::Done { .. }) => self.state.load_progress = 1.0,
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

    fn render(&mut self) {
        self.pump_loader();

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

        // egui frame.
        let raw_input = self.egui_state.take_egui_input(self.window.as_ref());
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            ui::build(ctx, &mut self.state, &mut self.camera);
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

        // Uniforms — computed AFTER egui so they reflect this frame's UI/camera changes
        // (otherwise a Points->Volume switch renders one frame with a stale matrix and
        // an un-uploaded texture).
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let cam = self.camera.to_uniform(
            aspect,
            [self.config.width as f32, self.config.height as f32],
        );
        self.points.update_camera(&self.queue, &cam);
        self.points.update_params(&self.queue, &self.state.params());

        // Volume mode: update the raycaster uniform (incl. the density scale) and
        // (re)upload the grid if it grew.
        if self.state.view_mode == ViewMode::Volume {
            // On entering volume mode, auto-range the transfer fn to the density
            // distribution (density sums differ in range from per-point intensity).
            if self.last_view_mode != ViewMode::Volume {
                let (lo, hi) = self.grid.density_percentiles();
                self.state.i_min = lo;
                self.state.i_max = hi;
            }
            let inv_vp = self.camera.inv_view_proj(aspect);
            let mut vu = self.state.volume_uniform(inv_vp);
            vu.density_scale = self.grid.density_scale();
            self.volume.update_uniform(&self.queue, &vu);
            if self.grid.dirty() {
                self.volume.upload(&self.queue, &self.grid.to_f16_scaled());
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
                ViewMode::Points => self.points.render(&mut rpass, self.state.point_mode),
                ViewMode::Volume => self.volume.render(&mut rpass),
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
