//! Application UI state and its mapping to GPU params.

use crate::data::point::AxisBounds;
use crate::render::point_cloud::PointMode;
use crate::render::uniforms::{ParamsUniform, VolumeUniform};

/// Top-level visualization mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    Points,
    Volume,
}

/// Volume raycast style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolStyle {
    Composite,
    Mip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferMode {
    Linear,
    Sqrt,
    Log,
}

impl TransferMode {
    fn as_f32(self) -> f32 {
        match self {
            TransferMode::Linear => 0.0,
            TransferMode::Sqrt => 1.0,
            TransferMode::Log => 2.0,
        }
    }
}

/// Inclusive real-unit window on one axis.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub min: f64,
    pub max: f64,
}

pub struct AppState {
    pub bounds: AxisBounds,

    pub view_mode: ViewMode,
    pub point_mode: PointMode,
    pub vol_style: VolStyle,
    pub vol_steps: u32,

    // Transfer function / appearance
    pub transfer: TransferMode,
    pub i_min: f32,
    pub i_max: f32,
    pub exposure: f32,
    pub colormap_id: u32,
    pub n_colormaps: u32,
    pub point_size: f32,
    pub opacity: f32,

    // Filters
    pub show_ms1: bool,
    pub show_ms2: bool,
    /// Show the DDA-precursor / DIA-window annotation overlay.
    pub show_annotations: bool,
    /// Show the data-cube wireframe + axis labels (m/z, 1/K0, RT).
    pub show_axes: bool,
    pub rt_window: Window,
    pub mz_window: Window,
    pub im_window: Window,
    /// Re-fit the window box to the full cube (zoom to selection) instead of just culling.
    pub focus: bool,

    // Readouts
    pub fps: f32,
    pub resident_points: u32,
    pub capacity: u32,
    pub total_estimate: u64,
    pub load_progress: f32,
    pub downsample_stride: u32,
    /// Set if the loader thread ended before signaling completion (e.g. panic).
    pub load_failed: bool,
}

impl AppState {
    pub fn new(bounds: AxisBounds, total_estimate: u64, n_colormaps: u32) -> Self {
        AppState {
            bounds,
            view_mode: ViewMode::Points,
            point_mode: PointMode::AdditiveDensity,
            vol_style: VolStyle::Composite,
            vol_steps: 256,
            transfer: TransferMode::Log,
            i_min: 1.0,
            i_max: 1e5,
            exposure: 1.0,
            colormap_id: 1, // Inferno
            n_colormaps,
            point_size: 2.5,
            opacity: 0.5,
            show_ms1: true,
            show_ms2: true,
            show_annotations: true,
            show_axes: true,
            rt_window: Window {
                min: bounds.rt.min,
                max: bounds.rt.max,
            },
            mz_window: Window {
                min: bounds.mz.min,
                max: bounds.mz.max,
            },
            im_window: Window {
                min: bounds.im.min,
                max: bounds.im.max,
            },
            focus: false,
            fps: 0.0,
            resident_points: 0,
            capacity: 0,
            total_estimate,
            load_progress: 0.0,
            downsample_stride: 1,
            load_failed: false,
        }
    }

    pub fn reset_windows(&mut self) {
        self.rt_window = Window {
            min: self.bounds.rt.min,
            max: self.bounds.rt.max,
        };
        self.mz_window = Window {
            min: self.bounds.mz.min,
            max: self.bounds.mz.max,
        };
        self.im_window = Window {
            min: self.bounds.im.min,
            max: self.bounds.im.max,
        };
    }

    /// Build the GPU params uniform from the current UI state.
    pub fn params(&self) -> ParamsUniform {
        let fmin = self.bounds.normalize(
            self.mz_window.min,
            self.im_window.min,
            self.rt_window.min,
        );
        let fmax = self.bounds.normalize(
            self.mz_window.max,
            self.im_window.max,
            self.rt_window.max,
        );
        let ms_mask = (self.show_ms1 as u32) | ((self.show_ms2 as u32) << 1);
        let render_mode = match self.point_mode {
            PointMode::AdditiveDensity => 0,
            PointMode::StructuralOpaque => 1,
        };
        ParamsUniform {
            filter_min: [fmin[0], fmin[1], fmin[2], 0.0],
            filter_max: [fmax[0], fmax[1], fmax[2], 0.0],
            transfer: [self.transfer.as_f32(), self.i_min, self.i_max, self.exposure],
            point_size: self.point_size,
            opacity: self.opacity,
            focus: if self.focus { 1.0 } else { 0.0 },
            _pad1: 0.0,
            ms_mask,
            colormap_id: self.colormap_id,
            render_mode,
            n_colormaps: self.n_colormaps.max(1),
        }
    }

    /// Build the volume raycaster uniform; `inv_view_proj` comes from the camera.
    pub fn volume_uniform(&self, inv_view_proj: [[f32; 4]; 4]) -> VolumeUniform {
        let bmin = self.bounds.normalize(
            self.mz_window.min,
            self.im_window.min,
            self.rt_window.min,
        );
        let bmax = self.bounds.normalize(
            self.mz_window.max,
            self.im_window.max,
            self.rt_window.max,
        );
        let style = match self.vol_style {
            VolStyle::Composite => 0,
            VolStyle::Mip => 1,
        };
        VolumeUniform {
            inv_view_proj,
            box_min: [bmin[0], bmin[1], bmin[2], 0.0],
            box_max: [bmax[0], bmax[1], bmax[2], 0.0],
            transfer: [self.transfer.as_f32(), self.i_min, self.i_max, self.exposure],
            steps: self.vol_steps.max(1),
            style,
            colormap_id: self.colormap_id,
            n_colormaps: self.n_colormaps.max(1),
            // Overwritten by the app from VolumeGrid::density_scale() each frame.
            density_scale: 1.0,
            focus: if self.focus { 1.0 } else { 0.0 },
            _pad: [0.0; 2],
        }
    }
}
