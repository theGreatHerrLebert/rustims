//! GPU uniform structs shared between the renderers and WGSL shaders.

use bytemuck::{Pod, Zeroable};

/// Per-frame camera data. 16-byte aligned throughout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    /// Camera right vector in world space (for billboard orientation).
    pub right: [f32; 4],
    /// Camera up vector in world space.
    pub up: [f32; 4],
    /// Framebuffer size in pixels (x, y) + padding.
    pub viewport: [f32; 4],
}

impl Default for CameraUniform {
    fn default() -> Self {
        CameraUniform {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            right: [1.0, 0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0, 0.0],
            viewport: [1.0, 1.0, 0.0, 0.0],
        }
    }
}

/// Rendering parameters bound to the UI controls.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ParamsUniform {
    /// Normalized-cube window minimum (xyz) + unused w.
    pub filter_min: [f32; 4],
    /// Normalized-cube window maximum (xyz) + unused w.
    pub filter_max: [f32; 4],
    /// Transfer function: x=mode (0 lin/1 sqrt/2 log), y=i_min, z=i_max, w=exposure.
    pub transfer: [f32; 4],
    pub point_size: f32,
    pub opacity: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    /// Bit mask: bit0 show MS1, bit1 show MS2.
    pub ms_mask: u32,
    pub colormap_id: u32,
    /// 0 = additive density, 1 = structural opaque.
    pub render_mode: u32,
    pub n_colormaps: u32,
}

impl Default for ParamsUniform {
    fn default() -> Self {
        ParamsUniform {
            filter_min: [-1.0, -1.0, -1.0, 0.0],
            filter_max: [1.0, 1.0, 1.0, 0.0],
            transfer: [2.0, 1.0, 1e5, 1.0],
            point_size: 2.5,
            opacity: 0.6,
            _pad0: 0.0,
            _pad1: 0.0,
            ms_mask: 0b11,
            colormap_id: 0,
            render_mode: 0,
            n_colormaps: 1,
        }
    }
}
