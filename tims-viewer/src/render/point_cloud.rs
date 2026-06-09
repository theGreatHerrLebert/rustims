//! Point-cloud renderer: a pre-allocated instance buffer plus two pipelines
//! (additive-density and structural-opaque) sharing one shader and bind group.

use wgpu::util::DeviceExt;

use crate::data::point::GpuPoint;

use super::colormap;
use super::uniforms::{CameraUniform, ParamsUniform};

/// Render mode selector matching `ParamsUniform.render_mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointMode {
    AdditiveDensity,
    StructuralOpaque,
}

pub struct PointCloudRenderer {
    /// Instance buffer sized to the point budget.
    instances: wgpu::Buffer,
    capacity: u32,
    resident: u32,

    camera_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    pipeline_additive: wgpu::RenderPipeline,
    pipeline_opaque: wgpu::RenderPipeline,

    // Keep the LUT alive for the lifetime of the bind group.
    _lut_texture: wgpu::Texture,
}

impl PointCloudRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        capacity: u32,
    ) -> Self {
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point-instances"),
            size: capacity as u64 * std::mem::size_of::<GpuPoint>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera-uniform"),
            contents: bytemuck::bytes_of(&CameraUniform::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params-uniform"),
            contents: bytemuck::bytes_of(&ParamsUniform::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (lut_texture, lut_view, lut_sampler) = colormap::create_lut_texture(device, queue);

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point-bind-group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&lut_sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point-cloud-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/point_cloud.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("point-pipeline-layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });

        let additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let make_pipeline = |blend: Option<wgpu::BlendState>, depth_write: bool, depth_cmp| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("point-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[GpuPoint::layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: depth_write,
                    depth_compare: depth_cmp,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        // Additive: order-independent, no depth writes, always blend.
        let pipeline_additive =
            make_pipeline(Some(additive_blend), false, wgpu::CompareFunction::Always);
        // Opaque: depth-tested and depth-written for true 3D structure.
        let pipeline_opaque =
            make_pipeline(Some(wgpu::BlendState::REPLACE), true, wgpu::CompareFunction::Less);

        PointCloudRenderer {
            instances,
            capacity,
            resident: 0,
            camera_buf,
            params_buf,
            bind_group,
            pipeline_additive,
            pipeline_opaque,
            _lut_texture: lut_texture,
        }
    }

    pub fn resident(&self) -> u32 {
        self.resident
    }

    /// Total point capacity (used by region-refinement / reload, Phase 1.5).
    #[allow(dead_code)]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Clear resident points (used by region-refinement / reload, Phase 1.5).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.resident = 0;
    }

    /// Append packed points at the running offset, clamping to capacity. Returns the
    /// number actually written.
    pub fn append(&mut self, queue: &wgpu::Queue, points: &[GpuPoint]) -> usize {
        if points.is_empty() || self.resident >= self.capacity {
            return 0;
        }
        let room = (self.capacity - self.resident) as usize;
        let n = points.len().min(room);
        let stride = std::mem::size_of::<GpuPoint>() as u64;
        let offset = self.resident as u64 * stride; // checked-range: resident <= capacity
        queue.write_buffer(&self.instances, offset, bytemuck::cast_slice(&points[..n]));
        self.resident += n as u32;
        n
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, cam: &CameraUniform) {
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(cam));
    }

    pub fn update_params(&self, queue: &wgpu::Queue, params: &ParamsUniform) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(params));
    }

    pub fn render(&self, rpass: &mut wgpu::RenderPass<'_>, mode: PointMode) {
        if self.resident == 0 {
            return;
        }
        let pipeline = match mode {
            PointMode::AdditiveDensity => &self.pipeline_additive,
            PointMode::StructuralOpaque => &self.pipeline_opaque,
        };
        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, self.instances.slice(..));
        // 4 vertices per instance (triangle strip quad), one instance per point.
        rpass.draw(0..4, 0..self.resident);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::point::GpuPoint;
    use crate::render::uniforms::{CameraUniform, ParamsUniform};

    /// Build both pipelines (forcing naga to compile the WGSL), upload points, and
    /// render offscreen in both modes. Skips cleanly if no GPU adapter is available
    /// (e.g. a headless CI box without a software rasterizer).
    #[test]
    fn offscreen_render_smoke() {
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
            eprintln!("no GPU adapter available; skipping offscreen render smoke test");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("smoke-device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .expect("request_device failed");

        let color_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let depth_format = wgpu::TextureFormat::Depth32Float;
        // Pipeline creation here compiles the WGSL and validates bind/layout.
        let mut r = PointCloudRenderer::new(&device, &queue, color_format, depth_format, 1024);

        let pts: Vec<GpuPoint> = (0..200)
            .map(|i| GpuPoint {
                pos: [(i as f32 / 200.0) * 2.0 - 1.0, 0.0, 0.0],
                intensity: 100.0 + i as f32,
                weight: 1.0,
                flags: (i % 2) as u32,
                _pad: [0, 0],
            })
            .collect();
        assert_eq!(r.append(&queue, &pts), 200);
        r.update_camera(&queue, &CameraUniform::default());
        r.update_params(&queue, &ParamsUniform::default());

        let make_tex = |format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
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

        for mode in [PointMode::AdditiveDensity, PointMode::StructuralOpaque] {
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
                r.render(&mut rpass, mode);
            }
            queue.submit(std::iter::once(enc.finish()));
            device.poll(wgpu::Maintain::Wait);
        }
    }
}
