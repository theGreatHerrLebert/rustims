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

/// Target maximum stored f16 value; the grid is normalized so its peak maps here,
/// well under the f16 ceiling (65504) so no value is clamped/lost.
const F16_TARGET_MAX: f32 = 60000.0;

/// CPU-side max-intensity voxel grid.
pub struct VolumeGrid {
    dims: [u32; 3],
    /// Max intensity per voxel, indexed x + nx*(y + ny*z).
    data: Vec<f32>,
    /// Largest intensity deposited so far (drives the normalization scale).
    max_intensity: f32,
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
            max_intensity: 0.0,
            dirty: false,
        }
    }

    pub fn dirty(&self) -> bool {
        self.dirty
    }

    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Deposit a point (normalized cube position in [-1,1], raw intensity) as a max into
    /// its containing voxel.
    #[inline]
    pub fn deposit(&mut self, pos: [f32; 3], intensity: f32) {
        let [nx, ny, nz] = self.dims;
        let vx = voxel_index(pos[0], nx);
        let vy = voxel_index(pos[1], ny);
        let vz = voxel_index(pos[2], nz);
        let idx = vx + nx as usize * (vy + ny as usize * vz);
        let slot = &mut self.data[idx];
        if intensity > *slot {
            *slot = intensity;
        }
        if intensity > self.max_intensity {
            self.max_intensity = intensity;
        }
        self.dirty = true;
    }

    /// Factor the shader multiplies the sampled (normalized) value by to recover raw
    /// intensity. `>= 1`; chosen so the grid peak maps to `F16_TARGET_MAX`.
    pub fn density_scale(&self) -> f32 {
        (self.max_intensity / F16_TARGET_MAX).max(1.0)
    }

    /// Normalize by `density_scale` and convert to f16 — preserves the full intensity
    /// range (incl. values far above 65504) instead of clamping it away. The shader
    /// multiplies back by `density_scale` before applying the shared transfer function.
    pub fn to_f16_scaled(&self) -> Vec<f16> {
        let inv = 1.0 / self.density_scale();
        self.data
            .iter()
            .map(|&v| f16::from_f32((v * inv).min(F16_TARGET_MAX)))
            .collect()
    }
}

#[inline]
fn voxel_index(norm: f32, dim: u32) -> usize {
    // Map [-1, 1] -> [0, dim-1].
    let t = (norm * 0.5 + 0.5).clamp(0.0, 1.0);
    ((t * dim as f32) as u32).min(dim - 1) as usize
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
    fn voxel_index_maps_cube_to_dims() {
        assert_eq!(voxel_index(-1.0, 8), 0);
        assert_eq!(voxel_index(1.0, 8), 7); // clamped to dim-1, not 8
        assert_eq!(voxel_index(0.0, 8), 4);
        assert_eq!(voxel_index(2.0, 8), 7); // out-of-range clamps
        assert_eq!(voxel_index(-9.0, 8), 0);
    }

    #[test]
    fn grid_deposit_keeps_max_and_clamps_f16() {
        let mut g = VolumeGrid::new([4, 4, 4]);
        g.deposit([0.0, 0.0, 0.0], 100.0);
        g.deposit([0.0, 0.0, 0.0], 50.0); // smaller -> ignored (max)
        g.deposit([0.0, 0.0, 0.0], 1e9); // huge -> normalized, not clamped away
        assert!(g.dirty());
        // Scale normalizes the 1e9 peak down under the f16 ceiling.
        assert!(g.density_scale() > 1.0);
        let f = g.to_f16_scaled();
        assert_eq!(f.len(), 64);
        let center = f[voxel_index(0.0, 4) + 4 * (voxel_index(0.0, 4) + 4 * voxel_index(0.0, 4))];
        // Stored value is finite and well under the f16 ceiling; recovering raw
        // intensity (value * density_scale) reconstructs the 1e9 peak.
        let stored = center.to_f32();
        assert!(stored > 0.0 && stored <= 65504.0);
        let recovered = stored * g.density_scale();
        assert!((recovered - 1e9).abs() / 1e9 < 0.02, "recovered {recovered}");
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
