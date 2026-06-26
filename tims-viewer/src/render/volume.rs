//! Volume rendering: a CPU max-intensity voxel grid (built incrementally as points
//! stream in) uploaded to an anisotropic R16Float 3D texture, plus a fullscreen-triangle
//! raycaster that shares the point cloud's transfer-function + colormap.
//!
//! MVP semantics: each voxel stores the MAX intensity of points falling in it (nearest
//! voxel). Max keeps voxel values in the same range as raw per-point intensity, so the
//! identical transfer function (i_min/i_max/log) and colormap apply unchanged — one
//! shared exposure model across points and volume. Trilinear cloud-in-cell *density*
//! deposition is the documented Phase-3 refinement.

use half::f16;

use super::colormap;
use super::uniforms::VolumeUniform;

/// Anisotropic grid dimensions (m/z, 1/K0, RT). m/z gets the most bins (finest features).
pub const VOLUME_DIMS: [u32; 3] = [256, 128, 192];

/// Target peak stored f16 value; the grid is normalized so its densest voxel maps
/// here, well under the f16 ceiling (65504) so nothing is clamped/lost.
const F16_TARGET_MAX: f32 = 60000.0;

/// CPU-side density voxel grid: trilinear cloud-in-cell accumulation of intensity.
pub struct VolumeGrid {
    dims: [u32; 3],
    /// Accumulated density per voxel, indexed x + nx*(y + ny*z).
    data: Vec<f32>,
    /// Largest accumulated voxel density (drives the normalization scale).
    max_density: f32,
    dirty: bool,
}

impl VolumeGrid {
    pub fn new(dims: [u32; 3]) -> Self {
        assert!(
            dims.iter().all(|&d| d > 0),
            "volume dims must be non-zero, got {dims:?}"
        );
        let n = (dims[0] as usize)
            .checked_mul(dims[1] as usize)
            .and_then(|m| m.checked_mul(dims[2] as usize))
            .expect("volume dimensions overflow usize");
        VolumeGrid {
            dims,
            data: vec![0.0; n],
            max_density: 0.0,
            dirty: false,
        }
    }

    pub fn dirty(&self) -> bool {
        self.dirty
    }

    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Deposit a point (normalized cube position in [-1,1], raw intensity) using
    /// trilinear cloud-in-cell weighting: the intensity is split across the 8 nearest
    /// voxels by their interpolation weights and ADDED. This is conservative (the 8
    /// weights sum to 1, so total deposited == intensity) and avoids the blocky
    /// aliasing of nearest-voxel binning. Edges clamp onto the boundary voxels.
    #[inline]
    pub fn deposit(&mut self, pos: [f32; 3], intensity: f32) {
        let [nx, ny, nz] = self.dims;
        // Cell-centered continuous coordinate in [-0.5, dim-0.5].
        let g = |norm: f32, dim: u32| (norm * 0.5 + 0.5).clamp(0.0, 1.0) * dim as f32 - 0.5;
        let (gx, gy, gz) = (g(pos[0], nx), g(pos[1], ny), g(pos[2], nz));
        let split = |gc: f32, dim: u32| -> (usize, usize, f32) {
            let base = gc.floor();
            let frac = gc - base;
            let lo = (base as i64).clamp(0, dim as i64 - 1) as usize;
            let hi = ((base as i64) + 1).clamp(0, dim as i64 - 1) as usize;
            (lo, hi, frac)
        };
        let (x0, x1, fx) = split(gx, nx);
        let (y0, y1, fy) = split(gy, ny);
        let (z0, z1, fz) = split(gz, nz);
        let nxu = nx as usize;
        let nyu = ny as usize;
        let mut add = |x: usize, y: usize, z: usize, w: f32| {
            let idx = x + nxu * (y + nyu * z);
            let v = self.data[idx] + intensity * w;
            self.data[idx] = v;
            if v > self.max_density {
                self.max_density = v;
            }
        };
        add(x0, y0, z0, (1.0 - fx) * (1.0 - fy) * (1.0 - fz));
        add(x1, y0, z0, fx * (1.0 - fy) * (1.0 - fz));
        add(x0, y1, z0, (1.0 - fx) * fy * (1.0 - fz));
        add(x1, y1, z0, fx * fy * (1.0 - fz));
        add(x0, y0, z1, (1.0 - fx) * (1.0 - fy) * fz);
        add(x1, y0, z1, fx * (1.0 - fy) * fz);
        add(x0, y1, z1, (1.0 - fx) * fy * fz);
        add(x1, y1, z1, fx * fy * fz);
        self.dirty = true;
    }

    /// Factor the shader multiplies the sampled (normalized) value by to recover raw
    /// density, chosen so the densest voxel maps to `F16_TARGET_MAX`.
    pub fn density_scale(&self) -> f32 {
        if self.max_density > 0.0 {
            self.max_density / F16_TARGET_MAX
        } else {
            1.0
        }
    }

    /// p1/p99 of the non-empty voxel densities — sensible default transfer range for
    /// the volume (density sums live in a different range than per-point intensity).
    pub fn density_percentiles(&self) -> (f32, f32) {
        let mut nz: Vec<f32> = self.data.iter().copied().filter(|&v| v > 0.0).collect();
        if nz.is_empty() {
            return (1.0, 2.0);
        }
        nz.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pct = |q: f32| nz[(((nz.len() - 1) as f32) * q) as usize];
        let lo = pct(0.50).max(1e-6); // median: voxels below it are sparse fringe
        let hi = pct(0.999).max(lo * 1.0001);
        (lo, hi)
    }

    /// Rebuild this grid as `wa*a + wb*b` of two same-dim source grids. Used to fold the
    /// separate MS1 / MS2 density grids into the displayed volume per the MS1/MS2 toggle
    /// (weights are 1.0 or 0.0), so the volume raycaster honors the same filter as points.
    pub fn combine(&mut self, a: &VolumeGrid, b: &VolumeGrid, wa: f32, wb: f32) {
        assert!(self.data.len() == a.data.len() && self.data.len() == b.data.len());
        let mut max = 0.0f32;
        for (i, d) in self.data.iter_mut().enumerate() {
            let v = a.data[i] * wa + b.data[i] * wb;
            *d = v;
            if v > max {
                max = v;
            }
        }
        self.max_density = max;
        self.dirty = true;
    }

    /// Normalize by `density_scale` and convert to f16. The shader multiplies back by
    /// `density_scale` before the transfer function.
    pub fn to_f16_scaled(&self) -> Vec<f16> {
        let inv = 1.0 / self.density_scale();
        self.data
            .iter()
            .map(|&v| f16::from_f32((v * inv).min(F16_TARGET_MAX)))
            .collect()
    }
}

pub struct VolumeRenderer {
    texture: wgpu::Texture,
    dims: [u32; 3],
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    _lut_texture: wgpu::Texture,
}

impl VolumeRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        dims: [u32; 3],
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume-texture"),
            size: wgpu::Extent3d {
                width: dims[0],
                height: dims[1],
                depth_or_array_layers: dims[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let (lut_texture, lut_view, lut_sampler) = colormap::create_lut_texture(device, queue);

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume-uniform"),
            size: std::mem::size_of::<VolumeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&VolumeUniform::default()));

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume-bind-group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&lut_sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volume-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/volume.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volume-pipeline-layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volume-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    // Premultiplied alpha: the composite fragment outputs (acc, alpha)
                    // with acc already premultiplied, so low-opacity rays let the scene
                    // clear color show through. MIP returns alpha=1 and stays opaque.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            // Scene pass owns a depth attachment; declare it but neither test nor write.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        VolumeRenderer {
            texture,
            dims,
            uniform_buf,
            bind_group,
            pipeline,
            _lut_texture: lut_texture,
        }
    }

    pub fn update_uniform(&self, queue: &wgpu::Queue, u: &VolumeUniform) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(u));
    }

    /// Upload the (f16) grid into the 3D texture. `data` length must equal the grid.
    pub fn upload(&self, queue: &wgpu::Queue, data: &[f16]) {
        let [nx, ny, nz] = self.dims;
        debug_assert_eq!(data.len(), nx as usize * ny as usize * nz as usize);
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(nx * 2), // R16Float = 2 bytes/texel
                rows_per_image: Some(ny),
            },
            wgpu::Extent3d {
                width: nx,
                height: ny,
                depth_or_array_layers: nz,
            },
        );
    }

    pub fn render(&self, rpass: &mut wgpu::RenderPass<'_>) {
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::uniforms::VolumeUniform;

    #[test]
    fn trilinear_deposit_conserves_total() {
        // Trilinear deposit splits intensity across the 8 nearest voxels with weights
        // summing to 1, so the total grid mass equals the deposited intensity.
        let mut g = VolumeGrid::new([8, 8, 8]);
        g.deposit([0.13, -0.27, 0.61], 1000.0);
        assert!(g.dirty());
        let total: f32 = g.data.iter().sum();
        assert!((total - 1000.0).abs() < 0.5, "total mass {total} != 1000");
        // A second point at the same place doubles the local mass (additive density).
        g.deposit([0.13, -0.27, 0.61], 1000.0);
        let total2: f32 = g.data.iter().sum();
        assert!((total2 - 2000.0).abs() < 1.0, "total mass {total2} != 2000");
    }

    #[test]
    fn density_scale_and_f16_recover_peak() {
        let mut g = VolumeGrid::new([4, 4, 4]);
        // Deposit on an exact voxel center so its full intensity lands in one voxel.
        g.deposit([0.0, 0.0, 0.0], 1e9);
        assert!(g.density_scale() > 1.0); // huge peak -> scaled down for f16
        let f = g.to_f16_scaled();
        assert_eq!(f.len(), 64);
        let peak = f.iter().map(|h| h.to_f32()).fold(0.0f32, f32::max);
        assert!(peak > 0.0 && peak <= 65504.0);
        // Recovering raw density (stored * density_scale) reconstructs the peak.
        let recovered = peak * g.density_scale();
        assert!((recovered - g.max_density).abs() / g.max_density < 0.01);
    }

    #[test]
    fn density_percentiles_are_ordered() {
        let mut g = VolumeGrid::new([16, 16, 16]);
        for i in 0..200 {
            let t = (i as f32 / 200.0) * 2.0 - 1.0;
            g.deposit([t, 0.0, 0.0], 100.0 + i as f32);
        }
        let (lo, hi) = g.density_percentiles();
        assert!(lo > 0.0 && hi > lo, "percentiles lo={lo} hi={hi}");
    }

    /// Build the volume pipeline (compiles volume.wgsl), upload a grid, and raycast
    /// offscreen in both styles. Skips if no GPU adapter is present.
    #[test]
    fn offscreen_volume_smoke() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY | wgpu::Backends::GL,
            ..Default::default()
        });
        let Some(adapter) = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        )) else {
            eprintln!("no GPU adapter; skipping offscreen volume smoke test");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("vol-smoke-device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .expect("request_device failed");

        let color_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let dims = [16u32, 16, 16];
        let r = VolumeRenderer::new(&device, &queue, color_format, depth_format, dims);

        let mut grid = VolumeGrid::new(dims);
        for i in 0..16 {
            let t = i as f32 / 15.0 * 2.0 - 1.0;
            grid.deposit([t, 0.0, 0.0], 500.0 + i as f32 * 100.0);
        }
        r.upload(&queue, &grid.to_f16_scaled());
        let cam = crate::camera::OrbitCamera::default();
        let mut u = VolumeUniform {
            inv_view_proj: cam.inv_view_proj(1.0),
            ..Default::default()
        };

        let make_tex = |format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 32,
                    height: 32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = make_tex(color_format, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth = make_tex(depth_format, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let cview = color.create_view(&Default::default());
        let dview = depth.create_view(&Default::default());

        for style in [0u32, 1] {
            u.style = style;
            r.update_uniform(&queue, &u);
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &cview,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &dview,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                r.render(&mut rpass);
            }
            queue.submit(std::iter::once(enc.finish()));
            device.poll(wgpu::Maintain::Wait);
        }
    }
}
