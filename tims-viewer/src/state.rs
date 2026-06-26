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

/// A request from the UI for the app to re-stream the loader (level-of-detail refinement).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefineAction {
    /// Re-stream only the current RT/m-z/1-K0 window at full resolution.
    Refine,
    /// Re-stream the whole run (revert a refinement).
    FullRun,
    /// Re-stream the current scope (used when the intensity-priority sampling toggles).
    Restream,
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
    /// Raw p1 of retained point intensity, from LoadMsg::Stats. 0.0 until known.
    /// Stored unstretched so a mode switch can re-run the heuristic from scratch.
    pub i_p1: f32,
    /// p50 (median) of retained point intensity, from LoadMsg::Stats. 0.0 until known.
    pub i_med: f32,
    /// Raw p99 of retained point intensity, from LoadMsg::Stats. 0.0 until known.
    pub i_p99: f32,
    /// True once the user has manually edited transfer/exposure/range this session;
    /// suppresses further auto-ranging so we never clobber manual edits.
    pub transfer_user_dirty: bool,
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
    /// Draw faint gridlines across the three far cube faces for depth reference.
    pub show_grid_backfaces: bool,
    /// Draw the intensity colorbar overlay (active colormap + range + transfer tag).
    pub show_colorbar: bool,
    /// Per-window-group visibility bitmask (bit g-1 = group g); used by the selection overlay.
    pub group_mask: u32,
    /// Number of DIA/MIDIA window groups in the loaded run (0 until known).
    pub n_window_groups: u32,
    pub rt_window: Window,
    pub mz_window: Window,
    pub im_window: Window,
    /// Re-fit the window box to the full cube (zoom to selection) instead of just culling.
    pub focus: bool,
    /// True while showing a re-streamed sub-region (vs the full run).
    pub refined: bool,
    /// Bias the load toward high-intensity points (brightest fill the budget first) instead
    /// of density-faithful uniform sampling. Applied at load time; toggling re-streams.
    pub intensity_priority: bool,
    /// Pending level-of-detail request from the UI; the app consumes it each frame.
    pub refine_request: Option<RefineAction>,

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
            i_p1: 0.0,
            i_med: 0.0,
            i_p99: 0.0,
            transfer_user_dirty: false,
            colormap_id: 1, // Inferno
            n_colormaps,
            point_size: 2.5,
            opacity: 0.5,
            show_ms1: true,
            show_ms2: true,
            show_annotations: true,
            show_axes: true,
            show_grid_backfaces: false,
            show_colorbar: true,
            group_mask: u32::MAX,
            n_window_groups: 0,
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
            refined: false,
            intensity_priority: false,
            refine_request: None,
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

    /// Auto-set transfer mode + range + exposure for POINT mode from the stored intensity
    /// percentiles (`i_p1`/`i_med`/`i_p99`) so the median maps to a legible mid-tone and the
    /// additive cloud never saturates. No-op until percentiles are known (`i_med > 0`).
    pub fn auto_transfer_points(&mut self) {
        if self.i_med <= 0.0 {
            return;
        }
        let p1 = self.i_p1.max(1.0);
        // Derive the working median into a LOCAL: `i_med` is the pristine raw p50 from
        // LoadMsg::Stats and must stay untouched so a later mode switch / Auto click can
        // re-run this heuristic from scratch (see field doc). Writing back here would
        // ratchet the anchor upward if p1 ever exceeds the raw median.
        let p50 = self.i_med.max(p1);
        let p99 = self.i_p99.max(p50 * 1.0001);
        self.i_min = p1;
        // Headroom above p99 so the few brightest peaks still gain color but the bulk maps
        // below 1.0. 4x ~= one extra decade for lognormal tails.
        self.i_max = (p99 * 4.0).max(p50 * 1.0001);

        // timsTOF intensity is lognormal -> Log transfer puts the median mid-LUT.
        self.transfer = TransferMode::Log;

        // Exposure controls additive COVERAGE (alpha), independent of the LUT index.
        // contrib = falloff(center=1) * opacity * exposure * weight, with weight == stride
        // (the loader deposits weight = 1/p = stride). Solve exposure so a single median
        // splat lands at ~TARGET_ALPHA so overlapping medians ramp toward (not past) white.
        let weight = self.downsample_stride.max(1) as f32;
        const TARGET_ALPHA: f32 = 0.35;
        self.exposure =
            (TARGET_ALPHA / (self.opacity.max(0.05) * weight)).clamp(0.02, 4.0);
    }

    /// Auto-set transfer range for VOLUME mode from density percentiles
    /// (p50 .. p99.9 from VolumeGrid::density_percentiles). Volume composite alpha is
    /// integrated in-shader, so exposure stays at its tuned default.
    pub fn auto_transfer_volume(&mut self, lo: f32, hi: f32) {
        self.transfer = TransferMode::Log;
        self.i_min = lo.max(1e-6);
        self.i_max = hi.max(self.i_min * 1.0001);
        self.exposure = 1.0;
    }

    /// True when auto-ranging is allowed (no manual transfer edits yet).
    pub fn may_auto_transfer(&self) -> bool {
        !self.transfer_user_dirty
    }

    /// Re-enable auto-ranging and immediately re-apply the heuristic for the active mode.
    /// `volume_range` is the density percentile pair, used only in volume mode.
    pub fn reset_transfer_auto(&mut self, volume_range: (f32, f32)) {
        self.transfer_user_dirty = false;
        match self.view_mode {
            ViewMode::Points => self.auto_transfer_points(),
            ViewMode::Volume => self.auto_transfer_volume(volume_range.0, volume_range.1),
        }
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
