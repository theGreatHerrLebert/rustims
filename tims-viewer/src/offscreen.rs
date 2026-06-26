//! Headless one-frame renderer: load a run, render the point cloud or volume to an
//! offscreen texture, and write a PNG. Works without a display (used for screenshots on
//! headless machines and to preview the volume raycaster).

use std::path::Path;

use anyhow::{Context, Result};

use crate::app::Plan;
use crate::camera::OrbitCamera;
use crate::data::demo::DemoSource;
use crate::data::loader::{LoadMsg, LoaderHandle, LoaderMode};
use crate::data::point::GpuPoint;
use crate::render::colormap::COLORMAP_NAMES;
use crate::render::annotation::AnnotationRenderer;
use crate::render::point_cloud::PointCloudRenderer;
use crate::render::volume::{VolumeGrid, VolumeRenderer, VOLUME_DIMS};
use crate::state::{AppState, ViewMode, VolStyle};

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Which MS level to render.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MsFilter {
    All,
    Ms1,
    Ms2,
}

pub struct Options {
    pub width: u32,
    pub height: u32,
    pub volume: bool,
    pub mip: bool,
    pub ms: MsFilter,
    pub annotations: bool,
    /// Optional transfer-function overrides (else taken from data percentiles).
    pub i_min: Option<f32>,
    pub i_max: Option<f32>,
    pub exposure: Option<f32>,
}

/// Render one frame of `plan`'s run and write a PNG to `out`.
pub fn render_png(plan: Plan, out: &Path, opts: &Options) -> Result<()> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok_or_else(|| anyhow::anyhow!("no GPU adapter available for offscreen render"))?;
    log::info!("offscreen adapter: {:?}", adapter.get_info());

    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("offscreen-device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))?;

    // Validate output dimensions: nonzero (else invalid texture / infinite aspect) and
    // within the device's max texture size (also bounds the row/size arithmetic below).
    anyhow::ensure!(
        opts.width > 0 && opts.height > 0,
        "render dimensions must be > 0 (got {}x{})",
        opts.width,
        opts.height
    );
    let max_dim = limits.max_texture_dimension_2d;
    anyhow::ensure!(
        opts.width <= max_dim && opts.height <= max_dim,
        "render dimensions {}x{} exceed the device max texture size {max_dim}",
        opts.width,
        opts.height
    );

    let flags = adapter.get_downlevel_capabilities().flags;
    let supports_compaction = flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        && flags.contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION);

    let bytes_per_point = std::mem::size_of::<GpuPoint>() as u64;
    let total = plan.meta.total_points_estimate;
    let mut cap = plan
        .budget
        .min((limits.max_buffer_size / bytes_per_point) as usize)
        .min((total as usize).max(1));
    if supports_compaction {
        cap = cap.min((limits.max_storage_buffer_binding_size as u64 / bytes_per_point) as usize);
    }
    let capacity = cap.max(1) as u32;

    let mut points =
        PointCloudRenderer::new(&device, &queue, COLOR_FORMAT, DEPTH_FORMAT, capacity, supports_compaction);
    let volume = VolumeRenderer::new(&device, &queue, COLOR_FORMAT, DEPTH_FORMAT, VOLUME_DIMS);
    let mut grid = VolumeGrid::new(VOLUME_DIMS);
    let mut annotations = AnnotationRenderer::new(&device, COLOR_FORMAT, DEPTH_FORMAT);

    let bounds = plan.meta.bounds;
    let mut state = AppState::new(bounds, total, COLORMAP_NAMES.len() as u32);
    state.capacity = capacity;
    state.downsample_stride =
        crate::data::loader::stride_for(total, capacity as usize) as u32;
    state.view_mode = if opts.volume { ViewMode::Volume } else { ViewMode::Points };
    state.vol_style = if opts.mip { VolStyle::Mip } else { VolStyle::Composite };
    // MS selection is applied at upload time via `opts.ms`; let the shader pass everything
    // that was uploaded (don't double-filter with the interactive MS1-only default).
    state.show_ms2 = true;

    // Load the whole run synchronously by draining the streaming loader.
    let mode = if plan.is_demo {
        LoaderMode::Demo(DemoSource::new(plan.meta.frames.len(), total))
    } else {
        LoaderMode::Real {
            path: plan.meta.data_path.clone(),
            frame_ids: plan.meta.frames.iter().map(|f| f.id).collect(),
            filter: None,
        }
    };
    let loader = LoaderHandle::spawn(mode, bounds, total, capacity as usize);
    let keep = |p: &GpuPoint| match opts.ms {
        MsFilter::All => true,
        MsFilter::Ms1 => p.flags & GpuPoint::MS2_FLAG == 0,
        MsFilter::Ms2 => p.flags & GpuPoint::MS2_FLAG != 0,
    };
    // Reservoir of RETAINED intensities so the auto transfer range matches what is
    // actually rendered (the loader's Stats span both MS levels).
    let mut reservoir: Vec<f32> = Vec::new();
    let sample_every = (capacity as usize / 200_000).max(1) as u64;
    let mut seen: u64 = 0;
    let mut done = false;
    loop {
        match loader.rx.recv() {
            Ok(LoadMsg::Chunk { points: pts, .. }) => {
                // Filter to the requested MS level at the source so both the volume grid
                // and the point buffer render only that level.
                let pts: Vec<GpuPoint> = if opts.ms == MsFilter::All {
                    pts
                } else {
                    pts.into_iter().filter(&keep).collect()
                };
                for p in &pts {
                    // Deposit intensity*weight so density is downsample-invariant.
                    grid.deposit(p.pos, p.intensity * p.weight);
                    if seen % sample_every == 0 && reservoir.len() < 200_000 {
                        reservoir.push(p.intensity);
                    }
                    seen += 1;
                }
                points.append(&queue, &pts);
            }
            Ok(LoadMsg::PeakChunk { points: pts }) => {
                // Display-only peaks: append to the cloud, never to the volume density.
                let pts: Vec<GpuPoint> = if opts.ms == MsFilter::All {
                    pts
                } else {
                    pts.into_iter().filter(&keep).collect()
                };
                points.append(&queue, &pts);
            }
            Ok(LoadMsg::Stats { .. }) => {} // recomputed below from retained points
            Ok(LoadMsg::Histograms { .. }) => {} // UI-only; headless ignores
            Ok(LoadMsg::Annotations { lines, .. }) => {
                if opts.annotations {
                    annotations.upload(&device, &lines);
                }
            }
            Ok(LoadMsg::Done { .. }) => {
                done = true;
                break;
            }
            Ok(LoadMsg::Error(e)) => anyhow::bail!("loader error: {e}"),
            Ok(LoadMsg::Progress(_)) => {}
            Err(_) => break,
        }
    }
    // A disconnect before Done means the loader thread died (e.g. panicked) — don't
    // silently write a partial/empty image.
    anyhow::ensure!(done, "loader thread ended before completing the load");

    // Default transfer range. Volume density sums live in a different range than
    // per-point intensity, so take it from the voxel density distribution; the point
    // cloud uses the retained-intensity percentiles.
    if opts.volume {
        let (lo, hi) = grid.density_percentiles();
        state.auto_transfer_volume(lo, hi);
    } else if !reservoir.is_empty() {
        reservoir.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pct = |q: f32| reservoir[(((reservoir.len() - 1) as f32) * q) as usize];
        // Feed the same auto-exposure heuristic the interactive viewer uses, so the
        // headless render is legible (median mid-LUT, additive cloud not blown out).
        state.i_p1 = pct(0.01).max(1.0);
        state.i_med = pct(0.50).max(state.i_p1);
        state.i_p99 = pct(0.99).max(state.i_med * 1.0001);
        state.auto_transfer_points();
    }
    // Apply transfer-function overrides (e.g. raise i_min to threshold out noise).
    if let Some(v) = opts.i_min {
        state.i_min = v;
    }
    if let Some(v) = opts.i_max {
        state.i_max = v;
    }
    if let Some(v) = opts.exposure {
        state.exposure = v;
    }
    log::info!(
        "loaded {} points (i_min={:.0} i_max={:.0} exposure={})",
        points.resident(),
        state.i_min,
        state.i_max,
        state.exposure
    );

    // Camera + uniforms.
    let camera = OrbitCamera::default();
    let aspect = opts.width as f32 / opts.height as f32;
    let cam_uniform = camera.to_uniform(aspect, [opts.width as f32, opts.height as f32]);
    let params = state.params();
    points.update_camera(&queue, &cam_uniform);
    points.update_params(&queue, &params);
    annotations.update_camera(&queue, &cam_uniform);
    annotations.update_filter(&queue, params.filter_min, params.filter_max, params.focus);
    if opts.volume {
        let mut vu = state.volume_uniform(camera.inv_view_proj(aspect));
        vu.density_scale = grid.density_scale();
        volume.update_uniform(&queue, &vu);
        volume.upload(&queue, grid.to_f16_scaled());
    }

    // Offscreen targets.
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen-color"),
        size: wgpu::Extent3d {
            width: opts.width,
            height: opts.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen-depth"),
        size: wgpu::Extent3d {
            width: opts.width,
            height: opts.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color.create_view(&Default::default());
    let depth_view = depth.create_view(&Default::default());

    // Padded readback buffer (bytes_per_row must be 256-aligned for texture->buffer).
    let unpadded_bpr = opts.width * 4;
    let padded_bpr = unpadded_bpr.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded_bpr * opts.height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("offscreen") });
    if !opts.volume {
        points.prepare(&queue, &mut encoder);
    }
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("offscreen-scene"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
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
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if opts.volume {
            volume.render(&mut rpass);
        } else {
            points.render(&mut rpass, state.point_mode);
        }
        if opts.annotations {
            annotations.render(&mut rpass);
        }
    }
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &readback,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(opts.height),
            },
        },
        wgpu::Extent3d {
            width: opts.width,
            height: opts.height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    // Map and read back.
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .context("map_async channel closed")?
        .context("buffer map failed")?;

    let data = slice.get_mapped_range();
    let mut rgba = Vec::with_capacity((opts.width * opts.height * 4) as usize);
    for row in 0..opts.height {
        let start = (row * padded_bpr) as usize;
        rgba.extend_from_slice(&data[start..start + unpadded_bpr as usize]);
    }
    drop(data);
    readback.unmap();

    // Encode PNG (Rgba8UnormSrgb bytes are already sRGB).
    let file = std::fs::File::create(out).with_context(|| format!("creating {}", out.display()))?;
    let w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, opts.width, opts.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut writer = encoder.write_header().context("png header")?;
    writer.write_image_data(&rgba).context("png write")?;

    Ok(())
}
