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

/// How points are colored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    /// Intensity through the active colormap.
    Intensity,
    /// DBSCAN cluster id (golden-ratio hues, noise grey).
    Cluster,
}

impl ColorMode {
    pub fn as_u32(self) -> u32 {
        match self {
            ColorMode::Intensity => 0,
            ColorMode::Cluster => 1,
        }
    }
}

/// Max resident points DBSCAN will run on; above this, clustering is disabled (refine first).
pub const CLUSTER_CAP: u32 = 600_000;

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
    /// Intensity range filter (real units, log-scaled UI). Points outside are culled.
    pub intensity_window: Window,
    /// Data intensity range (kept-sample min/max), the slider bounds; set from Histograms.
    pub i_data_lo: f32,
    pub i_data_hi: f32,
    /// Per-axis distribution histograms for the levels-style filters (empty until loaded).
    pub hist_mz: Vec<u32>,
    pub hist_im: Vec<u32>,
    pub hist_rt: Vec<u32>,
    pub hist_intensity: Vec<u32>,
    /// 2D projection minimaps (uploaded egui textures), the "you are here" overview planes.
    pub proj_mz_im: Option<egui::TextureHandle>,
    pub proj_mz_rt: Option<egui::TextureHandle>,
    pub proj_im_rt: Option<egui::TextureHandle>,
    /// Re-fit the window box to the full cube (zoom to selection) instead of just culling.
    pub focus: bool,
    /// True while showing a re-streamed sub-region (vs the full run).
    pub refined: bool,

    // Clustering (color points by DBSCAN cluster id)
    pub color_mode: ColorMode,
    /// DBSCAN neighborhood radius in normalized-cube units.
    pub cluster_eps: f32,
    /// DBSCAN minimum points to form a dense region.
    pub cluster_min_pts: u32,
    /// Last clustering result: number of clusters and noise points (0 until run).
    pub cluster_count: usize,
    pub cluster_noise: usize,
    /// Set by the UI "Cluster" button; the app runs DBSCAN and consumes it.
    pub cluster_request: bool,
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
            transfer: TransferMode::Sqrt,
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
            // Default to MS1-only: the precursor map is the cleaner first view; MS2 fragments
            // can be toggled on in the Filters panel.
            show_ms2: false,
            // The DIA/MIDIA window overlay is off by default (toggle on in the Filters panel).
            show_annotations: false,
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
            // Intensity range unknown until the first Histograms message; default pass-all.
            intensity_window: Window {
                min: 0.0,
                max: f64::INFINITY,
            },
            i_data_lo: 1.0,
            i_data_hi: 1.0e6,
            hist_mz: Vec::new(),
            hist_im: Vec::new(),
            hist_rt: Vec::new(),
            hist_intensity: Vec::new(),
            proj_mz_im: None,
            proj_mz_rt: None,
            proj_im_rt: None,
            focus: false,
            refined: false,
            color_mode: ColorMode::Intensity,
            cluster_eps: 0.012,
            cluster_min_pts: 8,
            cluster_count: 0,
            cluster_noise: 0,
            cluster_request: false,
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
        self.intensity_window = Window {
            min: self.i_data_lo as f64,
            max: self.i_data_hi as f64,
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

        // Sqrt transfer is the default for point intensity: gentler than log, it lifts the
        // mid-range without flattening the bright peaks.
        self.transfer = TransferMode::Sqrt;

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
        // Intensity range filter rides the unused .w of the spatial filter vectors (real
        // intensity units; the shader culls points outside).
        ParamsUniform {
            filter_min: [fmin[0], fmin[1], fmin[2], self.intensity_window.min as f32],
            filter_max: [fmax[0], fmax[1], fmax[2], self.intensity_window.max as f32],
            transfer: [self.transfer.as_f32(), self.i_min, self.i_max, self.exposure],
            point_size: self.point_size,
            opacity: self.opacity,
            focus: if self.focus { 1.0 } else { 0.0 },
            color_mode: self.color_mode.as_u32(),
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
