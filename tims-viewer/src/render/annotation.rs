//! Annotation overlay renderer: line-list geometry (DDA precursor crosses / DIA
//! isolation-window boxes) drawn over the scene, always visible (depth-test Always).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::uniforms::CameraUniform;

/// Normalized-cube window the overlay is clipped to (xyz; w unused).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FilterUniform {
    min: [f32; 4],
    max: [f32; 4],
}

impl Default for FilterUniform {
    fn default() -> Self {
        FilterUniform {
            min: [-1.0, -1.0, -1.0, 0.0],
            max: [1.0, 1.0, 1.0, 0.0],
        }
    }
}

pub struct AnnotationRenderer {
    vbuf: Option<wgpu::Buffer>,
    vcount: u32,
    camera_buf: wgpu::Buffer,
    filter_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl AnnotationRenderer {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("annotation-camera"),
            contents: bytemuck::bytes_of(&CameraUniform::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let filter_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("annotation-filter"),
            contents: bytemuck::bytes_of(&FilterUniform::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("annotation-bind-layout"),
            entries: &[uniform_entry(0), uniform_entry(1)],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("annotation-bind-group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: filter_buf.as_entire_binding(),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("annotation-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/annotation.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("annotation-pipeline-layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("annotation-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            // Scene pass owns a depth attachment; draw on top without testing/writing.
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
        AnnotationRenderer {
            vbuf: None,
            vcount: 0,
            camera_buf,
            filter_buf,
            bind_group,
            pipeline,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, cam: &CameraUniform) {
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(cam));
    }

    /// Set the normalized-cube window the overlay is clipped to.
    pub fn update_filter(&self, queue: &wgpu::Queue, min: [f32; 4], max: [f32; 4]) {
        queue.write_buffer(
            &self.filter_buf,
            0,
            bytemuck::bytes_of(&FilterUniform { min, max }),
        );
    }

    /// Upload line-list vertices (pairs = segments). Recreates the buffer.
    pub fn upload(&mut self, device: &wgpu::Device, verts: &[[f32; 3]]) {
        if verts.is_empty() {
            self.vbuf = None;
            self.vcount = 0;
            return;
        }
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("annotation-vertices"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.vbuf = Some(buf);
        self.vcount = verts.len() as u32;
    }

    pub fn render(&self, rpass: &mut wgpu::RenderPass<'_>) {
        let Some(vbuf) = &self.vbuf else { return };
        if self.vcount == 0 {
            return;
        }
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, vbuf.slice(..));
        rpass.draw(0..self.vcount, 0..1);
    }
}
