//! Point-cloud renderer: a pre-allocated instance buffer plus two pipelines
//! (additive-density and structural-opaque) sharing one shader and bind group.
//!
//! When the device supports compute + indirect execution, a compaction pass culls each
//! resident point (window + MS mask + frustum) into a compacted buffer and writes the
//! indirect-draw count, so the draw call processes only the *visible* set instead of
//! every resident instance. Otherwise it falls back to drawing all resident points and
//! culling in the vertex shader (the Phase-1 path).

use wgpu::util::DeviceExt;

use crate::data::point::GpuPoint;

use super::colormap;
use super::uniforms::{CameraUniform, CompactionUniform, ParamsUniform};

/// Render mode selector matching `ParamsUniform.render_mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointMode {
    AdditiveDensity,
    StructuralOpaque,
}

const WORKGROUP_SIZE: u32 = 256;

pub struct PointCloudRenderer {
    /// Master instance buffer (all resident points), sized to the budget.
    master: wgpu::Buffer,

    capacity: u32,
    resident: u32,
    /// How many of the resident points to actually draw (perf vs. detail at runtime). `u32::MAX`
    /// means "all". Points are stored shuffled, so any prefix is a representative subsample.
    draw_count: u32,

    camera_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    compaction_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    pipeline_additive: wgpu::RenderPipeline,
    pipeline_opaque: wgpu::RenderPipeline,

    /// Present only when compaction is supported.
    compute: Option<ComputeStage>,

    // Keep the LUT alive for the lifetime of the bind group.
    _lut_texture: wgpu::Texture,
}

/// Compaction resources, present only when the device supports compute + indirect execution.
/// These carry `STORAGE`/`INDIRECT` usage, which the WebGL2 fallback cannot allocate — so they
/// (and `master`'s `STORAGE` usage) live here and are simply never created on that path.
struct ComputeStage {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    /// Compacted visible points written by the compute pass.
    compacted: wgpu::Buffer,
    /// Indirect draw args: [vertex_count, instance_count, first_vertex, first_instance].
    draw_args: wgpu::Buffer,
}

impl PointCloudRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        capacity: u32,
        supports_compaction: bool,
    ) -> Self {
        let point_bytes = capacity as u64 * std::mem::size_of::<GpuPoint>() as u64;
        // Master holds all resident points, always drawn as an instance buffer. With compaction
        // it is ALSO read as a storage buffer by the compute pass; WebGL2 can't allocate storage
        // buffers, so request STORAGE only on the compaction path.
        let master_usage = wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST
            | if supports_compaction {
                wgpu::BufferUsages::STORAGE
            } else {
                wgpu::BufferUsages::empty()
            };
        let master = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point-master"),
            size: point_bytes,
            usage: master_usage,
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
        let compaction_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("compaction-uniform"),
            contents: bytemuck::bytes_of(&CompactionUniform::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (lut_texture, lut_view, lut_sampler) = colormap::create_lut_texture(device, queue);

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point-bind-layout"),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/point_cloud.wgsl").into()),
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
        let pipeline_opaque = make_pipeline(
            Some(wgpu::BlendState::REPLACE),
            true,
            wgpu::CompareFunction::Less,
        );

        let compute = if supports_compaction {
            // STORAGE/INDIRECT buffers: created only here, so the WebGL2 fallback allocates none.
            let compacted = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("point-compacted"),
                size: point_bytes.max(std::mem::size_of::<GpuPoint>() as u64),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let draw_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("point-draw-args"),
                // vertex_count=4 (quad strip), instance_count=0, first_vertex=0, first_instance=0
                contents: bytemuck::cast_slice(&[4u32, 0u32, 0u32, 0u32]),
                usage: wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            });
            Some(build_compute_stage(
                device,
                &master,
                compacted,
                draw_args,
                &camera_buf,
                &params_buf,
                &compaction_buf,
            ))
        } else {
            None
        };

        PointCloudRenderer {
            master,
            capacity,
            resident: 0,
            draw_count: u32::MAX,
            camera_buf,
            params_buf,
            compaction_buf,
            bind_group,
            pipeline_additive,
            pipeline_opaque,
            compute,
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
        queue.write_buffer(&self.master, offset, bytemuck::cast_slice(&points[..n]));
        self.resident += n as u32;
        n
    }

    /// Set how many resident points to draw (clamped to resident when rendering).
    pub fn set_draw_count(&mut self, n: u32) {
        self.draw_count = n;
    }

    /// Points actually drawn this frame = `min(draw_count, resident)`.
    fn effective(&self) -> u32 {
        self.draw_count.min(self.resident)
    }

    /// Public view of `effective()` (the drawn-instance count) for HUD/displayed-count accounting.
    pub fn drawn(&self) -> u32 {
        self.effective()
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, cam: &CameraUniform) {
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(cam));
    }

    pub fn update_params(&self, queue: &wgpu::Queue, params: &ParamsUniform) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(params));
    }

    /// Run the compaction compute pass (if supported). Must be called on the encoder
    /// BEFORE the scene render pass; uniforms must already be updated for this frame.
    pub fn prepare(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        let Some(compute) = &self.compute else {
            return;
        };
        let n = self.effective();
        if n == 0 {
            return;
        }
        // Reset the survivor counter every frame (atomicAdd accumulates otherwise);
        // keep vertex_count=4 for the quad strip.
        queue.write_buffer(&compute.draw_args, 0, bytemuck::cast_slice(&[4u32, 0u32, 0u32, 0u32]));
        // 2D dispatch grid: a single dispatch dimension is capped at 65535 workgroups, which
        // a 1D dispatch exceeds once the point count passes ~65535*256 (~16.8M). Spread the
        // workgroups across x and y and let the shader rebuild the linear index from
        // row_stride (= invocations per row). Only the first `n` points are considered, so the
        // shader's `i >= point_count` cull also enforces the runtime draw count.
        const MAX_DIM: u32 = 65535;
        let groups = n.div_ceil(WORKGROUP_SIZE);
        let groups_x = groups.min(MAX_DIM);
        let groups_y = groups.div_ceil(groups_x);
        let row_stride = groups_x * WORKGROUP_SIZE;
        queue.write_buffer(
            &self.compaction_buf,
            0,
            bytemuck::bytes_of(&CompactionUniform {
                point_count: n,
                row_stride,
                _pad: [0; 2],
            }),
        );
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("point-compaction"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute.pipeline);
        cpass.set_bind_group(0, &compute.bind_group, &[]);
        cpass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    pub fn render(&self, rpass: &mut wgpu::RenderPass<'_>, mode: PointMode) {
        let n = self.effective();
        if n == 0 {
            return;
        }
        let pipeline = match mode {
            PointMode::AdditiveDensity => &self.pipeline_additive,
            PointMode::StructuralOpaque => &self.pipeline_opaque,
        };
        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        if let Some(compute) = &self.compute {
            // Draw only the compacted visible set; count comes from the indirect args.
            rpass.set_vertex_buffer(0, compute.compacted.slice(..));
            rpass.draw_indirect(&compute.draw_args, 0);
        } else {
            // Fallback (e.g. WebGL2): draw the first `n` resident points, cull in the vertex shader.
            rpass.set_vertex_buffer(0, self.master.slice(..));
            rpass.draw(0..4, 0..n);
        }
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn compute_uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_compute_stage(
    device: &wgpu::Device,
    master: &wgpu::Buffer,
    compacted: wgpu::Buffer,
    draw_args: wgpu::Buffer,
    camera_buf: &wgpu::Buffer,
    params_buf: &wgpu::Buffer,
    compaction_buf: &wgpu::Buffer,
) -> ComputeStage {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compaction-bind-layout"),
        entries: &[
            storage_entry(0, true),  // src
            storage_entry(1, false), // dst
            storage_entry(2, false), // draw_args
            compute_uniform_entry(3),
            compute_uniform_entry(4),
            compute_uniform_entry(5),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compaction-bind-group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: master.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: compacted.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: draw_args.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: camera_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: compaction_buf.as_entire_binding(),
            },
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compaction-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compact.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compaction-pipeline-layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compaction-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "cs_main",
        compilation_options: Default::default(),
        cache: None,
    });
    ComputeStage {
        pipeline,
        bind_group,
        compacted,
        draw_args,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::point::GpuPoint;
    use crate::render::uniforms::ParamsUniform;

    /// Build the pipelines (forcing WGSL compilation for both the draw and, when
    /// supported, the compaction compute shader), upload points, run prepare() +
    /// render offscreen in both modes. Skips cleanly if no GPU adapter is available.
    ///
    /// `force_compaction`: `None` uses the device's real capability; `Some(false)` exercises the
    /// WebGL2-shaped fallback (no storage/indirect buffers, plain instanced draw).
    fn run_smoke(force_compaction: Option<bool>) {
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

        let flags = adapter.get_downlevel_capabilities().flags;
        let device_compaction = flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            && flags.contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION);
        let supports_compaction = force_compaction.unwrap_or(device_compaction);

        let color_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let mut r = PointCloudRenderer::new(
            &device,
            &queue,
            color_format,
            depth_format,
            1024,
            supports_compaction,
        );

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
        // Camera looking at the cube so the frustum cull keeps points.
        let cam = crate::camera::OrbitCamera::default().to_uniform(1.0, [64.0, 64.0]);
        r.update_camera(&queue, &cam);
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
            r.prepare(&queue, &mut enc);
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

    /// Smoke test on the device's real capabilities (compaction path where supported).
    #[test]
    fn offscreen_render_smoke() {
        run_smoke(None);
    }

    /// Force the no-compute fallback (the WebGL2-shaped path): the renderer must build with no
    /// storage/indirect buffers and the plain instanced draw must render without validation errors.
    #[test]
    fn offscreen_render_smoke_webgl2_fallback() {
        run_smoke(Some(false));
    }
}
