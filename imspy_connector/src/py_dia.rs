use std::sync::Arc;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyList, PySlice};
use numpy::{PyArray1, PyArray2};
use numpy::ndarray::{Array2, ShapeBuilder};

use rustdf::cluster::io as cio;

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use rustdf::cluster::candidates::{CandidateOpts, PrecursorSearchIndex};
use rustdf::cluster::cluster::{Attach1DOptions, BuildSpecOpts, ClusterResult1D, Eval1DOpts, Fit1D, RawPoints};
use rustdf::cluster::cluster_scoring::{best_ms1_for_each_ms2, score_pairs, ScoreOpts};
use rustdf::data::dia::TimsDatasetDIA;
use rustdf::data::handle::TimsData;
use rustdf::cluster::peak::{MzScanWindowGrid, FrameBinView, build_frame_bin_view, ImPeak1D, RtPeak1D, RtExpandParams, expand_many_im_peaks_along_rt};
use rustdf::cluster::utility::{MzScale, scan_mz_range, smooth_vector_gaussian, MobilityFn, trapezoid_area_fractional, quad_subsample, find_im_peaks_row, im_peak_id, blur_mz_all_frames};
use crate::py_tims_frame::PyTimsFrame;
use crate::py_tims_slice::PyTimsSlice;

#[pyfunction]
#[pyo3(signature = (path, clusters, compress=true, strip_points=false, strip_axes=false))]
pub fn save_clusters_bin(
    path: &str,
    clusters: Vec<Py<PyClusterResult1D>>,
    compress: bool,
    strip_points: bool,
    strip_axes: bool,
) -> PyResult<()> {
    let mut rust_clusters: Vec<ClusterResult1D> = clusters
        .into_iter()
        .map(|c| Python::with_gil(|py| c.borrow(py).inner.clone()))
        .collect();
    if strip_points || strip_axes {
        rust_clusters = rustdf::cluster::io::strip_heavy(rust_clusters, !strip_points, !strip_axes);
    }
    cio::save_bincode(path, &rust_clusters, compress)
        .map_err(|e| exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
pub fn load_clusters_bin(py: Python<'_>, path: &str) -> PyResult<Vec<Py<PyClusterResult1D>>> {
    let clusters = cio::load_bincode(path)
        .map_err(|e| exceptions::PyIOError::new_err(e.to_string()))?;
    clusters.into_iter()
        .map(|c| Py::new(py, PyClusterResult1D { inner: c }))
        .collect()
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyFit1D { pub inner: Fit1D }

#[pymethods]
impl PyFit1D {
    #[getter] fn mu(&self) -> f32 { self.inner.mu }
    #[getter] fn sigma(&self) -> f32 { self.inner.sigma }
    #[getter] fn height(&self) -> f32 { self.inner.height }
    #[getter] fn baseline(&self) -> f32 { self.inner.baseline }
    #[getter] fn area(&self) -> f32 { self.inner.area }
    #[getter] fn r2(&self) -> f32 { self.inner.r2 }
    #[getter] fn n(&self) -> usize { self.inner.n }

    pub fn __repr__(&self) -> String {
        format!("Fit1D(mu={:.6}, sigma={:.6}, area={:.3}, r2={:.4}, n={})",
                self.inner.mu, self.inner.sigma, self.inner.area, self.inner.r2, self.inner.n)
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct PyRawPoints { pub inner: RawPoints }

#[pymethods]
impl PyRawPoints {
    #[getter] fn len(&self) -> usize { self.inner.mz.len() }

    /// True if no points were collected
    #[getter]
    fn is_empty(&self) -> bool { self.inner.mz.is_empty() }

    /// Unique frame ids (sorted ascending)
    fn unique_frames<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<u32>>> {
        let mut v = self.inner.frame.clone();
        v.sort_unstable();
        v.dedup();
        Ok(PyArray1::from_vec_bound(py, v).unbind())
    }

    /// Unique scan indices (sorted ascending)
    fn unique_scans<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<u32>>> {
        let mut v = self.inner.scan.clone();
        v.sort_unstable();
        v.dedup();
        Ok(PyArray1::from_vec_bound(py, v).unbind())
    }

    /// (mz_min, mz_max)
    fn mz_min_max(&self) -> Option<(f32, f32)> {
        if self.inner.mz.is_empty() { return None; }
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in &self.inner.mz { if x < lo { lo = x; } else if x > hi { hi = x; } }
        Some((lo, hi))
    }

    /// (rt_min, rt_max)
    fn rt_min_max(&self) -> Option<(f32, f32)> {
        if self.inner.rt.is_empty() { return None; }
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in &self.inner.rt { if x < lo { lo = x; } else if x > hi { hi = x; } }
        Some((lo, hi))
    }

    /// (im_min, im_max)
    fn im_min_max(&self) -> Option<(f32, f32)> {
        if self.inner.im.is_empty() { return None; }
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in &self.inner.im { if x < lo { lo = x; } else if x > hi { hi = x; } }
        Some((lo, hi))
    }

    /// Sum and max of intensities
    fn intensity_sum_max(&self) -> (f32, f32) {
        let mut s = 0.0f32;
        let mut m = 0.0f32;
        for &y in &self.inner.intensity { s += y; if y > m { m = y; } }
        (s, m)
    }

    /// Return numpy arrays (mz, rt, im, scan, intensity, tof, frame) — no copy on the Python side
    fn to_arrays<'py>(&self, py: Python<'py>)
                      -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<f32>>, Py<PyArray1<f32>>,
                                   Py<PyArray1<u32>>, Py<PyArray1<f32>>, Py<PyArray1<i32>>, Py<PyArray1<u32>>)>
    {
        Ok((
            PyArray1::from_vec_bound(py, self.inner.mz.clone()).unbind(),
            PyArray1::from_vec_bound(py, self.inner.rt.clone()).unbind(),
            PyArray1::from_vec_bound(py, self.inner.im.clone()).unbind(),
            PyArray1::from_vec_bound(py, self.inner.scan.clone()).unbind(),
            PyArray1::from_vec_bound(py, self.inner.intensity.clone()).unbind(),
            PyArray1::from_vec_bound(py, self.inner.tof.clone()).unbind(),
            PyArray1::from_vec_bound(py, self.inner.frame.clone()).unbind(),
        ))
    }

    pub fn __repr__(&self) -> String {
        format!("RawPoints(n={})", self.inner.mz.len())
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyClusterResult1D { pub inner: ClusterResult1D }

#[pymethods]
impl PyClusterResult1D {
    // windows
    #[getter] fn rt_window(&self) -> (usize, usize) { self.inner.rt_window }
    #[getter] fn im_window(&self) -> (usize, usize) { self.inner.im_window }
    #[getter] fn mz_window(&self) -> (f32, f32) { self.inner.mz_window }

    // fits
    #[getter] fn rt_fit(&self) -> PyFit1D { PyFit1D { inner: self.inner.rt_fit.clone() } }
    #[getter] fn im_fit(&self) -> PyFit1D { PyFit1D { inner: self.inner.im_fit.clone() } }
    #[getter] fn mz_fit(&self) -> PyFit1D { PyFit1D { inner: self.inner.mz_fit.clone() } }

    // stats
    #[getter] fn raw_sum(&self) -> f32 { self.inner.raw_sum }
    #[getter] fn volume_proxy(&self) -> f32 { self.inner.volume_proxy }
    #[getter] fn ms_level(&self) -> u8 { self.inner.ms_level }
    #[getter] fn window_group(&self) -> Option<u32> { self.inner.window_group }
    #[getter] fn parent_im_id(&self) -> Option<i64> { self.inner.parent_im_id }
    #[getter] fn parent_rt_id(&self) -> Option<i64> { self.inner.parent_rt_id }

    // axes / frame IDs (optional arrays)
    fn frame_ids_used<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<u32>>> {
        Ok(PyArray1::from_vec_bound(py, self.inner.frame_ids_used.clone()).unbind())
    }
    fn rt_axis_sec<'py>(&self, py: Python<'py>) -> Option<Py<PyArray1<f32>>> {
        self.inner.rt_axis_sec.as_ref().map(|v|
            PyArray1::from_vec_bound(py, v.clone()).unbind()
        )
    }
    fn im_axis_scans<'py>(&self, py: Python<'py>) -> Option<Py<PyArray1<usize>>> {
        self.inner.im_axis_scans.as_ref().map(|v|
            PyArray1::from_vec_bound(py, v.clone()).unbind()
        )
    }
    fn mz_axis_da<'py>(&self, py: Python<'py>) -> Option<Py<PyArray1<f32>>> {
        self.inner.mz_axis_da.as_ref().map(|v|
            PyArray1::from_vec_bound(py, v.clone()).unbind()
        )
    }

    // raw points (optional)
    fn raw_points(&self) -> Option<PyRawPoints> {
        self.inner.raw_points.as_ref().map(|rp| PyRawPoints { inner: rp.clone() })
    }

    pub fn __repr__(&self) -> String {
        let (rt0, rt1) = self.inner.rt_window;
        let (im0, im1) = self.inner.im_window;
        format!("ClusterResult1D(ms_level={}, rt=({},{}) im=({},{}) raw_sum={:.1})",
                self.inner.ms_level, rt0, rt1, im0, im1, self.inner.raw_sum)
    }
}

#[inline]
fn owned_copy_im(p: &ImPeak1D) -> ImPeak1D {
    ImPeak1D {
        mz_row: p.mz_row,
        mz_center: p.mz_center,
        mz_bounds: p.mz_bounds,
        rt_bounds: p.rt_bounds,
        frame_id_bounds: p.frame_id_bounds,
        window_group: p.window_group,

        // scan geometry
        scan: p.scan,
        left: p.left,
        right: p.right,
        scan_abs: p.scan_abs,       // NEW
        left_abs: p.left_abs,       // NEW
        right_abs: p.right_abs,     // NEW
        left_x: p.left_x,
        right_x: p.right_x,
        width_scans: p.width_scans,

        mobility: p.mobility,
        apex_smoothed: p.apex_smoothed,
        apex_raw: p.apex_raw,
        prominence: p.prominence,
        area_raw: p.area_raw,
        subscan: p.subscan,
        id: p.id,
    }
}

#[pyclass]
pub struct PyRtPeak1D { pub inner: RtPeak1D }

#[pymethods]
impl PyRtPeak1D {
    // --- geometry in RT index/time
    #[getter] fn rt_idx(&self) -> usize { self.inner.rt_idx }
    #[getter] fn rt_sec(&self) -> Option<f32> { self.inner.rt_sec }
    #[getter] fn apex_smoothed(&self) -> f32 { self.inner.apex_smoothed }
    #[getter] fn apex_raw(&self) -> f32 { self.inner.apex_raw }
    #[getter] fn prominence(&self) -> f32 { self.inner.prominence }
    #[getter] fn left_x(&self) -> f32 { self.inner.left_x }
    #[getter] fn right_x(&self) -> f32 { self.inner.right_x }
    #[getter] fn width_frames(&self) -> usize { self.inner.width_frames }
    #[getter] fn area_raw(&self) -> f32 { self.inner.area_raw }
    #[getter] fn subframe(&self) -> f32 { self.inner.subframe }

    // --- provenance / bounds
    #[getter] fn rt_bounds_frames(&self) -> (usize, usize) { self.inner.rt_bounds_frames }
    #[getter] fn frame_id_bounds(&self) -> (u32, u32) { self.inner.frame_id_bounds }
    #[getter] fn window_group(&self) -> Option<u32> { self.inner.window_group }

    // --- m/z context
    #[getter] fn mz_row(&self) -> usize { self.inner.mz_row }
    #[getter] fn mz_center(&self) -> f32 { self.inner.mz_center }
    #[getter] fn mz_bounds(&self) -> (f32, f32) { self.inner.mz_bounds }

    // --- linkage
    #[getter] fn parent_im_id(&self) -> Option<i64> { self.inner.parent_im_id }
    #[getter] fn id(&self) -> i64 { self.inner.id }

    pub fn __repr__(&self) -> String {
        format!(
            "RtPeak1D(rt_idx={}, rt_sec={:?}, apex={:.3}, prom={:.3}, frames={:?}, mz_row={})",
            self.inner.rt_idx, self.inner.rt_sec, self.inner.apex_smoothed,
            self.inner.prominence, self.inner.rt_bounds_frames, self.inner.mz_row
        )
    }
}

#[pyclass]
pub struct PyImPeak1D { pub inner: Arc<ImPeak1D> }

#[pymethods]
impl PyImPeak1D {
    #[new]
    #[pyo3(signature = (
        mz_row,
        mz_center,
        mz_bounds,
        rt_bounds,
        frame_id_bounds,
        window_group,
        scan,
        left,
        right,
        scan_abs,
        left_abs,
        right_abs,
        mobility,
        apex_smoothed,
        apex_raw,
        prominence,
        left_x,
        right_x,
        width_scans,
        area_raw,
        subscan,
        id
    ))]
    pub fn new(
        mz_row: usize,
        mz_center: f32,
        mz_bounds: (f32, f32),
        rt_bounds: (usize, usize),
        frame_id_bounds: (u32, u32),
        window_group: Option<u32>,

        scan: usize,
        left: usize,
        right: usize,

        scan_abs: usize,
        left_abs: usize,
        right_abs: usize,

        mobility: Option<f32>,
        apex_smoothed: f32,
        apex_raw: f32,
        prominence: f32,
        left_x: f32,
        right_x: f32,
        width_scans: usize,
        area_raw: f32,
        subscan: f32,
        id: i64,
    ) -> Self {
        let inner = ImPeak1D {
            mz_row,
            mz_center,
            mz_bounds,
            rt_bounds,
            frame_id_bounds,
            window_group,

            scan,
            left,
            right,

            scan_abs,
            left_abs,
            right_abs,

            mobility,
            apex_smoothed,
            apex_raw,
            prominence,
            left_x,
            right_x,
            width_scans,
            area_raw,
            subscan,
            id,
        };
        Self { inner: Arc::new(inner) }
    }
    /// Stable 64-bit identity of this IM peak
    #[getter] fn id(&self) -> i64 { self.inner.id }
    /// Prefer this going forward (same value as rt_row)
    #[getter] fn mz_row(&self) -> usize { self.inner.mz_row }

    #[getter] fn mz_center(&self) -> f32 { self.inner.mz_center }

    #[getter] fn mz_bounds(&self) -> (f32, f32) { self.inner.mz_bounds }

    /// RT column bounds (inclusive) in the source RT grid for this row
    #[getter] fn rt_bounds(&self) -> (usize, usize) { self.inner.rt_bounds }

    /// Materialized frame-id bounds (inclusive) backing rt_bounds
    #[getter] fn frame_id_bounds(&self) -> (u32, u32) { self.inner.frame_id_bounds }

    /// Optional DIA window group id (if applicable)
    #[getter] fn window_group(&self) -> Option<u32> { self.inner.window_group }

    #[getter] fn scan(&self) -> usize { self.inner.scan }
    #[getter] fn mobility(&self) -> Option<f32> { self.inner.mobility }
    #[getter] fn apex_smoothed(&self) -> f32 { self.inner.apex_smoothed }
    #[getter] fn apex_raw(&self) -> f32 { self.inner.apex_raw }
    #[getter] fn prominence(&self) -> f32 { self.inner.prominence }
    #[getter] pub fn left(&self) -> usize { self.inner.left }
    #[getter] pub fn right(&self) -> usize { self.inner.right }
    #[getter] fn left_x(&self) -> f32 { self.inner.left_x }
    #[getter] fn right_x(&self) -> f32 { self.inner.right_x }
    #[getter] fn width_scans(&self) -> usize { self.inner.width_scans }
    #[getter] fn area_raw(&self) -> f32 { self.inner.area_raw }
    #[getter] fn subscan(&self) -> f32 { self.inner.subscan }
    #[getter] fn scan_abs(&self) -> usize { self.inner.scan_abs }
    #[getter] fn left_abs(&self) -> usize { self.inner.left_abs }
    #[getter] fn right_abs(&self) -> usize { self.inner.right_abs }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyMzScanWindowGrid {
    pub inner: MzScanWindowGrid,
}

#[pymethods]
impl PyMzScanWindowGrid {
    #[getter]
    fn rt_range_frames(&self) -> (usize, usize) { self.inner.rt_range_frames }

    #[getter]
    fn rt_range_sec(&self) -> (f32, f32) { self.inner.rt_range_sec }

    #[getter]
    fn rows(&self) -> usize { self.inner.rows }

    #[getter]
    fn cols(&self) -> usize { self.inner.cols }

    #[getter]
    fn frame_id_bounds(&self) -> (u32, u32) { self.inner.frame_id_bounds }

    #[getter]
    fn window_group(&self) -> Option<u32> { self.inner.window_group }

    /// Global scan axis (0..global_num_scans-1) as u32 for stable dtype.
    #[getter]
    fn scans<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<u32>>> {
        let v: Vec<u32> = self.inner.scans.iter().map(|&s| s as u32).collect();
        Ok(PyArray1::from_vec_bound(py, v).unbind())
    }

    /// Dense window matrix (smoothed if smoothing was applied), Fortran order (rows, cols).
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray2<f32>>> {
        let arr_f = Array2::from_shape_vec((self.inner.rows, self.inner.cols).f(), self.inner.data.clone())
            .map_err(|e| exceptions::PyValueError::new_err(format!("shape error: {e}")))?;
        Ok(PyArray2::from_owned_array_bound(py, arr_f).unbind())
    }

    /// Optional raw (pre-smoothing) matrix, Fortran order (rows, cols).
    #[getter]
    fn data_raw<'py>(&self, py: Python<'py>) -> Option<Py<PyArray2<f32>>> {
        self.inner.data_raw.as_ref().map(|raw| {
            let arr_f = Array2::from_shape_vec((self.inner.rows, self.inner.cols).f(), raw.clone()).unwrap();
            PyArray2::from_owned_array_bound(py, arr_f).unbind()
        })
    }

    pub fn __repr__(&self) -> String {
        let (l, r) = self.inner.rt_range_frames;
        let (tl, tr) = self.inner.rt_range_sec;
        format!(
            "MzScanWindowGrid(frames=({l},{r}), rt=({:.3},{:.3})s, shape=({}, {}))",
            tl, tr, self.inner.rows, self.inner.cols
        )
    }
    #[getter]
    fn mz_centers<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec_bound(py, self.inner.scale.centers.clone()).unbind())
    }
    #[getter]
    fn mz_edges<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec_bound(py, self.inner.scale.edges.clone()).unbind())
    }
    /// Pick 1D IM peaks in each m/z row of this window.
    /// Returns a nested list: List[List[PyImPeak1D]] with outer index = row (m/z bin).
    // ── REPLACE the entire pick_im_peaks method in impl PyMzScanWindowGrid with this ──
    #[pyo3(signature = (
    min_prom=50.0,
    min_distance_scans=2,
    min_width_scans=2,
    use_mobility=false
))]
    pub fn pick_im_peaks<'py>(
        &self,
        py: Python<'py>,
        min_prom: f32,
        min_distance_scans: usize,
        min_width_scans: usize,
        use_mobility: bool,
    ) -> PyResult<Vec<Vec<Py<PyImPeak1D>>>> {
    let rows = self.inner.rows;
    let cols = self.inner.cols;

    let mob_fn: MobilityFn = if use_mobility { Some(|scan| scan as f32) } else { None };

    let rows_rs: Vec<Vec<ImPeak1D>> = (0..rows).map(|r| {
    let mut y_s = Vec::with_capacity(cols);
    let mut y_r = Vec::with_capacity(cols);
    for s in 0..cols {
    let val_s = self.inner.data[s * rows + r];
    y_s.push(val_s);
    let val_r = self.inner
    .data_raw
    .as_ref()
    .map(|dr| dr[s * rows + r])
    .unwrap_or(val_s);
    y_r.push(val_r);
    }

    find_im_peaks_row_nocontext(
    &y_s,
    &y_r,
    r, // mz_row
    self.inner.mz_center_for_row(r),
    self.inner.mz_bounds_for_row(r),
    self.inner.rt_range_frames,
    self.inner.frame_id_bounds,
    self.inner.window_group,
    &self.inner.scans,            // <-- NEW ARG: absolute scan axis
    mob_fn,
    min_prom,
    min_distance_scans,
    min_width_scans,
    )
    }).collect();

    impeaks_to_py_nested(py, rows_rs)
    }
}

#[pyclass]
pub struct PyMzScanPlanGroup {
    ds: Py<PyAny>,
    window_group: u32,

    // planned axes + schedule
    scale: Arc<MzScale>,
    frame_ids_sorted: Vec<u32>,
    frame_times: Vec<f32>,
    windows_idx: Vec<(usize, usize)>,
    rows: usize,
    global_num_scans: usize,

    // exec params
    maybe_sigma_scans: Option<f32>,
    maybe_sigma_mz_bins: Option<f32>,
    truncate: f32,
    num_threads: usize,

    // optional accel
    views: Option<Vec<FrameBinView>>,

    // iterator state
    cur: usize,
}

#[pymethods]
impl PyMzScanPlanGroup {
    #[new]
    #[pyo3(signature = (
        ds,
        window_group,
        ppm_per_bin,
        mz_pad_ppm,
        rt_window_sec,
        rt_hop_sec,
        num_threads=4,
        maybe_sigma_scans=None,
        maybe_sigma_mz_bins=None,
        truncate=3.0,
        precompute_views=false,
        clamp_mz_to_group=true
    ))]
    pub fn new(
        py: Python<'_>,
        ds: Py<PyAny>,
        window_group: u32,
        ppm_per_bin: f32,
        mz_pad_ppm: f32,
        rt_window_sec: f32,
        rt_hop_sec: f32,
        num_threads: usize,
        maybe_sigma_scans: Option<f32>,
        maybe_sigma_mz_bins: Option<f32>,
        truncate: f32,
        precompute_views: bool,
        clamp_mz_to_group: bool,
    ) -> PyResult<Self> {
        // --- collect frames/times and discover scale
        let (frame_ids_sorted, frame_times, scale, rows, global_num_scans, views) = {
            let ds_bound = ds.bind(py);
            let ds_obj: &Bound<PyTimsDatasetDIA> = ds_bound.downcast()?;
            let ds_ref = ds_obj.borrow();

            // 1) RT-sort MS2 frames for this group (from DiaFrameMsMsInfo)
            let (frame_ids_sorted, frame_times) =
                ds_ref.fragment_frame_ids_and_times_for_group(window_group);

            // 2) materialize frames once to discover mz range + global scan max
            let frames = ds_ref.inner.get_slice(frame_ids_sorted.clone(), num_threads).frames;

            // 3) m/z range: prefer DIA-window clamp; else scan actual frames
            let (mut mz_min, mut mz_max) = if clamp_mz_to_group {
                ds_ref
                    .mz_bounds_for_group(window_group)
                    .or_else(|| scan_mz_range(&frames))
                    .ok_or_else(|| exceptions::PyRuntimeError::new_err("no m/z found for group"))?
            } else {
                scan_mz_range(&frames).ok_or_else(|| {
                    exceptions::PyRuntimeError::new_err("no m/z found for group")
                })?
            };

            if mz_pad_ppm > 0.0 {
                let f = 1.0 + mz_pad_ppm * 1e-6;
                mz_min /= f; mz_max *= f;
            }
            let scale = MzScale::build(mz_min.max(10.0), mz_max, ppm_per_bin);
            let rows = scale.num_bins();

            // 4) global scan axis size
            let global_num_scans = ds_ref.max_global_num_scans();

            // 5) optional precomputed per-frame views
            let views = if precompute_views {
                Some((0..frames.len())
                    .into_par_iter()
                    .map(|i| build_frame_bin_view(frames[i].clone(), &scale, global_num_scans))
                    .collect())
            } else { None };

            (frame_ids_sorted, frame_times, scale, rows, global_num_scans, views)
        };

        // 6) Build RT window schedule (same logic as MS1 plan)
        let fps = if frame_times.len() < 2 { 1.0 } else {
            let mut d: Vec<f32> = frame_times.windows(2).map(|w| w[1] - w[0]).collect();
            d.sort_by(|a,b| a.partial_cmp(&b).unwrap());
            1.0 / d[d.len()/2].max(1e-6)
        };
        let mut win_len = (rt_window_sec * fps).max(1.0).round() as usize;
        let hop_len = (rt_hop_sec   * fps).max(1.0).round() as usize;
        let n_frames = frame_ids_sorted.len().max(1);
        win_len = win_len.min(n_frames);
        let hop_len = hop_len.max(1);

        let mut windows_idx: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        while start < n_frames {
            let end = (start + win_len - 1).min(n_frames - 1);
            windows_idx.push((start, end));
            if end + 1 >= n_frames { break; }
            start = (start + hop_len).min(n_frames - 1);
            if start == 0 { break; }
        }

        Ok(Self {
            ds,
            window_group,
            scale: Arc::new(scale),
            frame_ids_sorted,
            frame_times,
            windows_idx,
            rows,
            global_num_scans,
            maybe_sigma_scans,
            maybe_sigma_mz_bins,
            truncate,
            num_threads,
            views,
            cur: 0,
        })
    }

    // ---- Introspection (parity with PyMzScanPlan)
    #[getter] fn rows(&self) -> usize { self.rows }
    #[getter] fn global_num_scans(&self) -> usize { self.global_num_scans }
    #[getter] fn num_windows(&self) -> usize { self.windows_idx.len() }
    #[getter] fn window_group(&self) -> u32 { self.window_group }

    #[getter]
    fn frame_times<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec_bound(py, self.frame_times.clone()).unbind())
    }
    #[getter]
    fn frame_ids<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<u32>>> {
        Ok(PyArray1::from_vec_bound(py, self.frame_ids_sorted.clone()).unbind())
    }
    #[getter]
    fn mz_centers<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec_bound(py, self.scale.centers.clone()).unbind())
    }

    pub fn bounds(&self, i: usize) -> Option<(usize, usize)> {
        self.windows_idx.get(i).copied()
    }
    pub fn bounds_frame_ids(&self, i: usize) -> Option<(u32, u32)> {
        self.windows_idx.get(i).map(|(lo, hi)| (self.frame_ids_sorted[*lo], self.frame_ids_sorted[*hi]))
    }
    #[getter]
    fn fragment_frame_id_bounds(&self) -> Option<(u32, u32)> {
        if self.frame_ids_sorted.is_empty() { None } else { Some((self.frame_ids_sorted[0], *self.frame_ids_sorted.last().unwrap())) }
    }

    // --- Python iteration protocol + __getitem__ (Bound<PyAny> to avoid deprecation)
    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> { slf.cur = 0; slf }
    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<Py<PyMzScanWindowGrid>>> {
        if slf.cur >= slf.windows_idx.len() { return Ok(None); }
        let i = slf.cur; slf.cur += 1;
        let grid = slf.build_window(py, i)?;
        Ok(Some(Py::new(py, PyMzScanWindowGrid { inner: grid })?))
    }
    fn __len__(&self) -> usize { self.windows_idx.len() }

    fn __getitem__(&self, py: Python<'_>, idx: &Bound<PyAny>) -> PyResult<PyObject> {
        if let Ok(i_signed) = idx.extract::<isize>() {
            let n = self.windows_idx.len() as isize;
            let j = if i_signed < 0 { n + i_signed } else { i_signed };
            if j < 0 || j >= n { return Err(exceptions::PyIndexError::new_err("index out of range")); }
            let grid = self.build_window(py, j as usize)?;
            let obj = Py::new(py, PyMzScanWindowGrid { inner: grid })?;
            return Ok(obj.into_py(py));
        }
        if let Ok(slice) = idx.downcast::<PySlice>() {
            let indices = slice.indices(self.windows_idx.len() as isize)?;
            let (start, stop, step) = (indices.start, indices.stop, indices.step);
            let out = PyList::empty_bound(py);
            let mut i = start;
            if step > 0 {
                while i < stop {
                    let grid = self.build_window(py, i as usize)?;
                    out.append(Py::new(py, PyMzScanWindowGrid { inner: grid })?)?;
                    i += step;
                }
            } else if step < 0 {
                while i > stop {
                    let grid = self.build_window(py, i as usize)?;
                    out.append(Py::new(py, PyMzScanWindowGrid { inner: grid })?)?;
                    i += step;
                }
            }
            return Ok(out.into_py(py));
        }
        Err(exceptions::PyTypeError::new_err("indices must be int or slice"))
    }
    pub fn get_batch(
        &self,
        py: Python<'_>,
        start: usize,
        count: usize,
    ) -> PyResult<Vec<Py<PyMzScanWindowGrid>>> {
        let n = self.windows_idx.len();
        if start >= n { return Ok(Vec::new()); }
        let end = (start + count).min(n);
        let mut out = Vec::with_capacity(end - start);
        for i in start..end {
            let grid = self.build_window(py, i)?;
            out.push(Py::new(py, PyMzScanWindowGrid { inner: grid })?);
        }
        Ok(out)
    }

    #[pyo3(signature = (indices, min_prom=50.0, min_distance_scans=2, min_width_scans=10, use_mobility=false))]
    pub fn pick_im_peaks_for_indices(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
        min_prom: f32,
        min_distance_scans: usize,
        min_width_scans: usize,
        use_mobility: bool,
    ) -> PyResult<Vec<Vec<Vec<Py<PyImPeak1D>>>>>
    {
        let results: Vec<Vec<Vec<ImPeak1D>>> = py.allow_threads(|| {
            indices.into_par_iter()
                .filter_map(|i| {
                    if i >= self.windows_idx.len() { return None; }
                    let grid = match self.build_window(unsafe { Python::assume_gil_acquired() }, i) {
                        Ok(g) => g,
                        Err(_) => return None,
                    };
                    Some(pick_im_peaks_rows_from_grid(
                        &grid, min_prom, min_distance_scans, min_width_scans, use_mobility
                    ))
                })
                .collect()
        });

        let mut out = Vec::with_capacity(results.len());
        for rows in results {
            let mut rows_py = Vec::with_capacity(rows.len());
            for row in rows {
                let mut row_py = Vec::with_capacity(row.len());
                for p in row {
                    row_py.push(Py::new(py, PyImPeak1D { inner: Arc::new(p) })?);
                }
                rows_py.push(row_py);
            }
            out.push(rows_py);
        }
        Ok(out)
    }
}

impl PyMzScanPlanGroup {
    fn build_window(&self, py: Python<'_>, i: usize) -> PyResult<MzScanWindowGrid> {
        let (lo, hi) = self.windows_idx[i];
        let rows = self.rows;
        let cols = self.global_num_scans;
        let do_smooth = self.maybe_sigma_scans.unwrap_or(0.0) > 0.0;
        let do_blur_mz = self.maybe_sigma_mz_bins.unwrap_or(0.0) > 0.0;

        // Views for this window (group frames only)
        let mut views_local: Vec<FrameBinView> = if let Some(ref views) = self.views {
            (lo..=hi).map(|k| views[k].clone()).collect()
        } else {
            let fids = self.frame_ids_sorted[lo..=hi].to_vec();
            let frames = {
                let ds_bound = self.ds.bind(py);
                let ds_obj: &Bound<PyTimsDatasetDIA> = ds_bound.downcast()?;
                let ds_ref = ds_obj.borrow();
                ds_ref.inner.get_slice(fids, self.num_threads).frames
            };
            frames.into_par_iter()
                .map(|fr| build_frame_bin_view(fr, &self.scale, cols))
                .collect()
        };

        if do_blur_mz {
            let sigma_bins = self.maybe_sigma_mz_bins.unwrap();
            let trunc = self.truncate;
            views_local = blur_mz_all_frames(&views_local, sigma_bins, trunc);
        }

        // Accumulate onto the **global** scan axis (same as MS1 plan)
        let mut raw = vec![0.0f32; rows * cols];
        for v in &views_local {
            for b_i in 0..v.unique_bins.len() {
                let row = v.unique_bins[b_i];
                let start = v.offsets[b_i];
                let end   = v.offsets[b_i + 1];
                for j in start..end {
                    let s_phys = v.scan_idx[j] as usize;
                    if s_phys < cols {
                        raw[s_phys * rows + row] += v.intensity[j];
                    }
                }
            }
        }

        let data = if do_smooth {
            let mut sm = raw.clone();
            for r in 0..rows {
                let mut y: Vec<f32> = (0..cols).map(|c| sm[c * rows + r]).collect();
                smooth_vector_gaussian(&mut y, self.maybe_sigma_scans.unwrap(), self.truncate);
                for c in 0..cols { sm[c * rows + r] = y[c]; }
            }
            sm
        } else {
            raw.clone()
        };

        let (lo, hi) = self.windows_idx[i];
        let frame_id_bounds = (
            self.frame_ids_sorted[lo],
            self.frame_ids_sorted[hi],
        );

        Ok(MzScanWindowGrid {
            scale: self.scale.clone(),
            rt_range_frames: (lo, hi),
            rt_range_sec: (self.frame_times[lo], self.frame_times[hi]),
            frame_id_bounds,
            window_group: Some(self.window_group),
            scans: (0..cols).collect(),
            data,
            rows,
            cols,
            data_raw: if do_smooth || do_blur_mz { Some(raw) } else { None },
        })
    }
}

#[pyclass]
pub struct PyMzScanPlan {
    ds: Py<PyAny>,                 // keeps dataset alive; downcast when needed

    // planned axes + schedule
    scale: Arc<MzScale>,
    frame_ids_sorted: Vec<u32>,
    frame_times: Vec<f32>,
    windows_idx: Vec<(usize, usize)>,
    rows: usize,
    global_num_scans: usize,

    // exec params
    maybe_sigma_scans: Option<f32>,
    maybe_sigma_mz_bins: Option<f32>,
    truncate: f32,
    num_threads: usize,

    // optional accel
    views: Option<Vec<FrameBinView>>,

    // iterator state
    cur: usize,
}

#[pymethods]
impl PyMzScanPlan {
    #[new]
    #[pyo3(signature = (
        ds,
        ppm_per_bin,
        mz_pad_ppm,
        rt_window_sec,
        rt_hop_sec,
        num_threads=4,
        maybe_sigma_scans=None,
        maybe_sigma_mz_bins=None,
        truncate=3.0,
        precompute_views=false
    ))]
    pub fn new(
        py: Python<'_>,
        ds: Py<PyAny>,
        ppm_per_bin: f32,
        mz_pad_ppm: f32,
        rt_window_sec: f32,
        rt_hop_sec: f32,
        num_threads: usize,
        maybe_sigma_scans: Option<f32>,
        maybe_sigma_mz_bins: Option<f32>,
        truncate: f32,
        precompute_views: bool,
    ) -> PyResult<Self> {

        // ---- Borrow only inside this block; collect everything we need, then drop the borrow.
        let (frame_ids_sorted, frame_times, scale, rows, global_num_scans, views) = {
            let ds_bound = ds.bind(py);
            let ds_obj: &Bound<PyTimsDatasetDIA> = ds_bound.downcast()?;
            let ds_ref = ds_obj.borrow();

            // 1) RT-sort MS1 frames
            let (frame_ids_sorted, frame_times) = ds_ref.precursor_frame_ids_and_times();

            // 2) Discover scale
            let frames = ds_ref.inner.get_slice(frame_ids_sorted.clone(), num_threads).frames;
            let (mut mz_min, mut mz_max) =
                scan_mz_range(&frames).ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no m/z found"))?;
            if mz_pad_ppm > 0.0 {
                let f = 1.0 + mz_pad_ppm * 1e-6;
                mz_min /= f; mz_max *= f;
            }
            let scale = MzScale::build(mz_min.max(10.0), mz_max, ppm_per_bin);
            let rows = scale.num_bins();
            let global_num_scans = ds_ref.max_global_num_scans();

            // 4) Optional precomputed views
            let views = if precompute_views {
                Some((0..frames.len())
                    .into_par_iter()
                    .map(|i| build_frame_bin_view(frames[i].clone(), &scale, global_num_scans))
                    .collect())
            } else { None };

            (frame_ids_sorted, frame_times, scale, rows, global_num_scans, views)
        }; // <— borrow dropped here

        // 3) RT window schedule (can be done after borrow has been dropped)
        let fps = if frame_times.len() < 2 { 1.0 } else {
            let mut d: Vec<f32> = frame_times.windows(2).map(|w| w[1] - w[0]).collect();
            d.sort_by(|a,b| a.partial_cmp(&b).unwrap());
            1.0 / d[d.len()/2].max(1e-6)
        };
        let mut win_len = (rt_window_sec * fps).max(1.0).round() as usize;
        let hop_len = (rt_hop_sec   * fps).max(1.0).round() as usize;
        let n_frames = frame_ids_sorted.len().max(1);
        win_len = win_len.min(n_frames);
        let hop_len = hop_len.max(1);

        let mut windows_idx: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        while start < n_frames {
            let end = (start + win_len - 1).min(n_frames - 1);
            windows_idx.push((start, end));
            if end + 1 >= n_frames { break; }
            start = (start + hop_len).min(n_frames - 1);
            if start == 0 { break; }
        }

        Ok(Self {
            ds,                      // now safe to move
            scale: Arc::new(scale),
            frame_ids_sorted,
            frame_times,
            windows_idx,
            rows,
            global_num_scans,
            maybe_sigma_scans,
            maybe_sigma_mz_bins,
            truncate,
            num_threads,
            views,
            cur: 0,
        })
    }

    // ---- Introspection
    #[getter] fn rows(&self) -> usize { self.rows }
    #[getter] fn global_num_scans(&self) -> usize { self.global_num_scans }
    #[getter] fn num_windows(&self) -> usize { self.windows_idx.len() }

    #[getter]
    fn frame_times<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec_bound(py, self.frame_times.clone()).unbind())
    }

    #[getter]
    fn frame_ids<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<u32>>> {
        Ok(numpy::PyArray1::from_vec_bound(py, self.frame_ids_sorted.clone()).unbind())
    }

    #[getter]
    fn mz_centers<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(numpy::PyArray1::from_vec_bound(py, self.scale.centers.clone()).unbind())
    }

    pub fn bounds(&self, i: usize) -> Option<(usize, usize)> {
        self.windows_idx.get(i).copied()
    }

    // ---- Materialization (random access + Python iteration protocol)
    pub fn get(&self, py: Python<'_>, i: usize) -> PyResult<Option<Py<PyMzScanWindowGrid>>> {
        if i >= self.windows_idx.len() { return Ok(None); }
        let grid = self.build_window(py, i)?;
        Ok(Some(Py::new(py, PyMzScanWindowGrid { inner: grid })?))
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> { slf.cur = 0; slf }
    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<Py<PyMzScanWindowGrid>>> {
        if slf.cur >= slf.windows_idx.len() { return Ok(None); }
        let i = slf.cur; slf.cur += 1;
        let grid = slf.build_window(py, i)?;
        Ok(Some(Py::new(py, PyMzScanWindowGrid { inner: grid })?))
    }
    // Python len(plan)
    fn __len__(&self) -> usize { self.windows_idx.len() }

    /// Bounds in *frame IDs* for window i (precursor space)
    pub fn bounds_frame_ids(&self, i: usize) -> Option<(u32, u32)> {
        self.windows_idx.get(i).map(|(lo, hi)| (self.frame_ids_sorted[*lo], self.frame_ids_sorted[*hi]))
    }

    /// Overall precursor frame-id bounds across the plan
    #[getter]
    fn precursor_frame_id_bounds(&self) -> Option<(u32, u32)> {
        if self.frame_ids_sorted.is_empty() {
            None
        } else {
            Some((self.frame_ids_sorted[0], *self.frame_ids_sorted.last().unwrap()))
        }
    }

    /// plan[i] or plan[i:j:k]
    fn __getitem__(&self, py: Python<'_>, idx: &Bound<PyAny>) -> PyResult<PyObject> {
        // Try integer first (supports negative)
        if let Ok(i_signed) = idx.extract::<isize>() {
            let n = self.windows_idx.len() as isize;
            let j = if i_signed < 0 { n + i_signed } else { i_signed };
            if j < 0 || j >= n {
                return Err(exceptions::PyIndexError::new_err("index out of range"));
            }
            let grid = self.build_window(py, j as usize)?;
            let obj = Py::new(py, PyMzScanWindowGrid { inner: grid })?;
            return Ok(obj.into_py(py));
        }

        // Then try slice
        if let Ok(slice) = idx.downcast::<PySlice>() {
            let indices = slice.indices(self.windows_idx.len() as isize)?;
            let (start, stop, step) = (indices.start, indices.stop, indices.step);
            let out = PyList::empty_bound(py);

            let mut i = start;
            if step > 0 {
                while i < stop {
                    let grid = self.build_window(py, i as usize)?;
                    out.append(Py::new(py, PyMzScanWindowGrid { inner: grid })?)?;
                    i += step;
                }
            } else if step < 0 {
                while i > stop {
                    let grid = self.build_window(py, i as usize)?;
                    out.append(Py::new(py, PyMzScanWindowGrid { inner: grid })?)?;
                    i += step; // step is negative
                }
            }
            return Ok(out.into_py(py));
        }

        Err(exceptions::PyTypeError::new_err(
            "indices must be int or slice",
        ))
    }

    /// Materialize a batch of windows [start, start+count)
    pub fn get_batch(
        &self,
        py: Python<'_>,
        start: usize,
        count: usize,
    ) -> PyResult<Vec<Py<PyMzScanWindowGrid>>> {
        let n = self.windows_idx.len();
        if start >= n { return Ok(Vec::new()); }
        let end = (start + count).min(n);
        let mut out = Vec::with_capacity(end - start);
        for i in start..end {
            let grid = self.build_window(py, i)?;
            out.push(Py::new(py, PyMzScanWindowGrid { inner: grid })?);
        }
        Ok(out)
    }

    /// Pick IM peaks for a set of window indices in one Rust call.
    /// Returns: List[ List[ List[PyImPeak1D] ] ]  ==> windows × rows × peaks
    #[pyo3(signature = (indices, min_prom=50.0, min_distance_scans=2, min_width_scans=2, use_mobility=false))]
    pub fn pick_im_peaks_for_indices(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
        min_prom: f32,
        min_distance_scans: usize,
        min_width_scans: usize,
        use_mobility: bool,
    ) -> PyResult<Vec<Vec<Vec<Py<PyImPeak1D>>>>>
    {
        // Heavy work without GIL
        let results: Vec<Vec<Vec<ImPeak1D>>> = py.allow_threads(|| {
            indices.into_par_iter()
                .filter_map(|i| {
                    if i >= self.windows_idx.len() { return None; }
                    // Build window (this borrows the dataset briefly inside build_window)
                    // Safe to call here; build_window does its own short borrows.
                    let grid = match self.build_window(unsafe { Python::assume_gil_acquired() }, i) {
                        Ok(g) => g,
                        Err(_) => return None,
                    };
                    let rows = pick_im_peaks_rows_from_grid(
                        &grid, min_prom, min_distance_scans, min_width_scans, use_mobility
                    );
                    Some(rows)
                })
                .collect()
        });

        // Wrap back to Python objects
        let mut out = Vec::with_capacity(results.len());
        for rows in results {
            let mut rows_py = Vec::with_capacity(rows.len());
            for row in rows {
                let mut row_py = Vec::with_capacity(row.len());
                for p in row {
                    row_py.push(Py::new(py, PyImPeak1D { inner: Arc::new(p) })?);
                }
                rows_py.push(row_py);
            }
            out.push(rows_py);
        }
        Ok(out)
    }
}

impl PyMzScanPlan {
    fn build_window(&self, py: Python<'_>, i: usize) -> PyResult<MzScanWindowGrid> {
        let (lo, hi) = self.windows_idx[i];
        let rows = self.rows;
        let cols = self.global_num_scans;
        let do_smooth = self.maybe_sigma_scans.unwrap_or(0.0) > 0.0;
        let do_blur_mz = self.maybe_sigma_mz_bins.unwrap_or(0.0) > 0.0;

        // Views for this window; borrow the dataset only around get_slice
        let mut views_local: Vec<FrameBinView> = if let Some(ref views) = self.views {
            (lo..=hi).map(|k| views[k].clone()).collect()
        } else {
            let fids = self.frame_ids_sorted[lo..=hi].to_vec();
            let frames = {
                let ds_bound = self.ds.bind(py);
                let ds_obj: &Bound<PyTimsDatasetDIA> = ds_bound.downcast()?;
                let ds_ref = ds_obj.borrow();
                ds_ref.inner.get_slice(fids, self.num_threads).frames
            };
            frames.into_par_iter()
                .map(|fr| build_frame_bin_view(fr, &self.scale, cols))
                .collect()
        };

        // 2) OPTIONAL: m/z blur on sparse frames (separable, stable)
        if do_blur_mz {
            let sigma_bins = self.maybe_sigma_mz_bins.unwrap();
            let trunc = self.truncate;
            views_local = blur_mz_all_frames(&views_local, sigma_bins, trunc);
        }

        // Accumulate onto the fixed global scan axis
        let mut raw = vec![0.0f32; rows * cols];
        for v in &views_local {
            for b_i in 0..v.unique_bins.len() {
                let row = v.unique_bins[b_i];
                let start = v.offsets[b_i];
                let end   = v.offsets[b_i + 1];
                for j in start..end {
                    let s_phys = v.scan_idx[j] as usize;
                    if s_phys < cols {
                        raw[s_phys * rows + row] += v.intensity[j];
                    }
                }
            }
        }

        let data = if do_smooth {
            let mut sm = raw.clone();
            for r in 0..rows {
                let mut y: Vec<f32> = (0..cols).map(|c| sm[c * rows + r]).collect();
                smooth_vector_gaussian(&mut y, self.maybe_sigma_scans.unwrap(), self.truncate);
                for c in 0..cols { sm[c * rows + r] = y[c]; }
            }
            sm
        } else {
            raw.clone()
        };

        let (lo, hi) = self.windows_idx[i];
        let frame_id_bounds = (
            self.frame_ids_sorted[lo],
            self.frame_ids_sorted[hi],
        );

        Ok(MzScanWindowGrid {
            scale: self.scale.clone(),
            rt_range_frames: (lo, hi),
            rt_range_sec: (self.frame_times[lo], self.frame_times[hi]),
            frame_id_bounds,           // NEW
            window_group: None,        // NEW
            scans: (0..cols).collect(),
            data,
            rows,
            cols,
            data_raw: if do_smooth || do_blur_mz { Some(raw) } else { None },
        })
    }
}

#[pyclass]
pub struct PyTimsDatasetDIA {
    inner: TimsDatasetDIA,
}

#[pymethods]
impl PyTimsDatasetDIA {
    #[new]
    pub fn new(data_path: &str, bruker_lib_path: &str, in_memory: bool, use_bruker_sdk: bool) -> Self {
        let dataset = TimsDatasetDIA::new(bruker_lib_path, data_path, in_memory, use_bruker_sdk);
        PyTimsDatasetDIA { inner: dataset }
    }

    /// Enumerate (ms2_idx, ms1_idx) candidate pairs using DIA program + RT/IM constraints.
    ///
    /// Returns a list of (ms2_idx, ms1_idx) integer tuples.
    ///
    /// Notes:
    /// - This clones the passed ClusterResult1D objects. For best memory behavior,
    ///   avoid attaching raw points to clusters before calling this method.
    /// - Heavy work runs without the GIL.
    #[pyo3(signature = (
    ms1, ms2,
    // RT coarse guards
    min_rt_jaccard = 0.10_f32,
    rt_guard_sec = 0.0_f64,
    rt_bucket_width = 1.0_f64,
    // sanity limits
    max_ms1_rt_span_sec = Some(60.0_f64),
    max_ms2_rt_span_sec = Some(60.0_f64),
    min_raw_sum = 1.0_f32,
    // ---- NEW tight guards (keyword-only in Python) ----
    max_rt_apex_delta_sec = Some(2.0_f32),
    max_scan_apex_delta = Some(6_usize),
    min_im_overlap_scans = 1_usize,
))]
    pub fn enumerate_ms2_ms1_pairs(
        &self,
        py: Python<'_>,
        ms1: Vec<Py<PyClusterResult1D>>,
        ms2: Vec<Py<PyClusterResult1D>>,
        // RT coarse guards
        min_rt_jaccard: f32,
        rt_guard_sec: f64,
        rt_bucket_width: f64,
        // sanity limits
        max_ms1_rt_span_sec: Option<f64>,
        max_ms2_rt_span_sec: Option<f64>,
        min_raw_sum: f32,
        // NEW tight guards
        max_rt_apex_delta_sec: Option<f32>,
        max_scan_apex_delta: Option<usize>,
        min_im_overlap_scans: usize,
    ) -> PyResult<Vec<(usize, usize)>> {

        // 1) Options (now includes tight guards)
        let opts = CandidateOpts {
            min_rt_jaccard,
            ms2_rt_guard_sec: rt_guard_sec,
            rt_bucket_width,
            max_ms1_rt_span_sec,
            max_ms2_rt_span_sec,
            min_raw_sum,

            // NEW:
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            min_im_overlap_scans,
        };

        // 2) Extract owned Rust clusters (clone, still cheap if no raw points attached)
        let ms1_rust: Vec<ClusterResult1D> = ms1
            .into_iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();
        let ms2_rust: Vec<ClusterResult1D> = ms2
            .into_iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();

        // 3) Build index + enumerate without the GIL
        let pairs = py.allow_threads(|| {
            let idx = PrecursorSearchIndex::build(&self.inner, &ms1_rust, &opts);
            idx.enumerate_pairs(&ms1_rust, &ms2_rust, &opts)
        });

        Ok(pairs)
    }

    /// Score candidate (ms2_idx, ms1_idx) pairs using the built-in scoring model.
    ///
    /// You can pass optional weight overrides; all unspecified parameters fall back to defaults.
    #[pyo3(signature = (
        ms1, ms2, pairs,
        w_jacc_rt = 1.0_f32,
        w_shape = 1.0_f32,
        w_rt_apex = 0.75_f32,
        w_im_apex = 0.75_f32,
        w_im_overlap = 0.5_f32,
        w_ms1_intensity = 0.25_f32,
        rt_apex_scale_s = 0.75_f32,
        im_apex_scale_scans = 3.0_f32,
        shape_neutral = 0.6_f32,
        min_sigma_rt = 0.05_f32,
        min_sigma_im = 0.5_f32,
        w_shape_rt_inner = 1.0_f32,
        w_shape_im_inner = 1.0_f32,
        return_features = false,
    ))]
    pub fn score_ms2_ms1_pairs(
        &self,
        py: Python<'_>,
        ms1: Vec<Py<PyClusterResult1D>>,
        ms2: Vec<Py<PyClusterResult1D>>,
        pairs: Vec<(usize, usize)>,
        // optional scoring weights / parameters
        w_jacc_rt: f32,
        w_shape: f32,
        w_rt_apex: f32,
        w_im_apex: f32,
        w_im_overlap: f32,
        w_ms1_intensity: f32,
        rt_apex_scale_s: f32,
        im_apex_scale_scans: f32,
        shape_neutral: f32,
        min_sigma_rt: f32,
        min_sigma_im: f32,
        w_shape_rt_inner: f32,
        w_shape_im_inner: f32,
        return_features: bool,
    ) -> PyResult<Vec<PyObject>> {
        let sopts = ScoreOpts {
            w_jacc_rt,
            w_shape,
            w_rt_apex,
            w_im_apex,
            w_im_overlap,
            w_ms1_intensity,
            rt_apex_scale_s,
            im_apex_scale_scans,
            shape_neutral,
            min_sigma_rt,
            min_sigma_im,
            w_shape_rt_inner,
            w_shape_im_inner,
        };

        let ms1_rust: Vec<ClusterResult1D> =
            ms1.into_iter().map(|p| p.borrow(py).inner.clone()).collect();
        let ms2_rust: Vec<ClusterResult1D> =
            ms2.into_iter().map(|p| p.borrow(py).inner.clone()).collect();

        let scored = py.allow_threads(|| score_pairs(&ms1_rust, &ms2_rust, &pairs, &sopts));

        let mut out = Vec::with_capacity(scored.len());
        for (j, i, feats, s) in scored {
            if return_features {
                let d = PyDict::new_bound(py);
                d.set_item("ms2_idx", j)?;
                d.set_item("ms1_idx", i)?;
                d.set_item("score", s)?;
                d.set_item("rt_apex_delta_s", feats.rt_apex_delta_s)?;
                d.set_item("im_apex_delta_scans", feats.im_apex_delta_scans)?;
                d.set_item("jacc_rt", feats.jacc_rt)?;
                d.set_item("im_overlap", feats.im_overlap_scans)?;
                d.set_item("shape_ok", feats.shape_ok)?;
                d.set_item("s_shape", feats.s_shape)?;
                out.push(d.into_py(py));
            } else {
                out.push((j, i, s).into_py(py));
            }
        }

        Ok(out)
    }

    #[pyo3(signature = (
    ms1, ms2, pairs,
    w_jacc_rt = 1.0_f32,
    w_shape = 1.0_f32,
    w_rt_apex = 0.75_f32,
    w_im_apex = 0.75_f32,
    w_im_overlap = 0.5_f32,
    w_ms1_intensity = 0.25_f32,
    rt_apex_scale_s = 0.75_f32,
    im_apex_scale_scans = 3.0_f32,
    shape_neutral = 0.6_f32,
    min_sigma_rt = 0.05_f32,
    min_sigma_im = 0.5_f32,
    w_shape_rt_inner = 1.0_f32,
    w_shape_im_inner = 1.0_f32,
))]
    pub fn best_ms1_per_ms2(
        &self,
        py: Python<'_>,
        ms1: Vec<Py<PyClusterResult1D>>,
        ms2: Vec<Py<PyClusterResult1D>>,
        pairs: Vec<(usize, usize)>,
        // same params as above
        w_jacc_rt: f32,
        w_shape: f32,
        w_rt_apex: f32,
        w_im_apex: f32,
        w_im_overlap: f32,
        w_ms1_intensity: f32,
        rt_apex_scale_s: f32,
        im_apex_scale_scans: f32,
        shape_neutral: f32,
        min_sigma_rt: f32,
        min_sigma_im: f32,
        w_shape_rt_inner: f32,
        w_shape_im_inner: f32,
    ) -> PyResult<Vec<Option<usize>>> {
        let sopts = ScoreOpts {
            w_jacc_rt,
            w_shape,
            w_rt_apex,
            w_im_apex,
            w_im_overlap,
            w_ms1_intensity,
            rt_apex_scale_s,
            im_apex_scale_scans,
            shape_neutral,
            min_sigma_rt,
            min_sigma_im,
            w_shape_rt_inner,
            w_shape_im_inner,
        };

        let ms1_rust: Vec<ClusterResult1D> =
            ms1.into_iter().map(|p| p.borrow(py).inner.clone()).collect();
        let ms2_rust: Vec<ClusterResult1D> =
            ms2.into_iter().map(|p| p.borrow(py).inner.clone()).collect();

        let winners =
            py.allow_threads(|| best_ms1_for_each_ms2(&ms1_rust, &ms2_rust, &pairs, &sopts));
        Ok(winners)
    }

    pub fn get_frame(&self, frame_id: u32) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.get_frame(frame_id) }
    }

    pub fn get_slice(&self, frame_ids: Vec<u32>, num_threads: usize) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.get_slice(frame_ids, num_threads) }
    }

    pub fn get_acquisition_mode(&self) -> String {
        self.inner.get_acquisition_mode().to_string()
    }

    pub fn get_frame_count(&self) -> i32 {
        self.inner.get_frame_count()
    }

    pub fn get_data_path(&self) -> &str {
        self.inner.get_data_path()
    }
    
    pub fn sample_precursor_signal(&self, num_frames: usize, max_intensity: f64, take_probability: f64) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.sample_precursor_signal(num_frames, max_intensity, take_probability) }
    }
    
    pub fn sample_fragment_signal(&self, num_frames: usize, window_group: u32, max_intensity: f64, take_probability: f64) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.sample_fragment_signal(num_frames, window_group, max_intensity, take_probability) }
    }

    /// Convenience helper for Python-side planner
    #[getter]
    pub fn max_global_num_scans(&self) -> usize {
        self.inner.meta_data
            .iter()
            .map(|m| (m.num_scans + 1) as usize)
            .max()
            .unwrap_or(0)
    }

    /// (Optional) Helper if you want a clean way to fetch MS1 ids+times
    pub fn precursor_frame_ids_and_times(&self) -> (Vec<u32>, Vec<f32>) {
        let mut v: Vec<(u32, f32)> = self.inner
            .meta_data
            .iter()
            .filter(|m| m.ms_ms_type == 0)
            .map(|m| (m.id as u32, m.time as f32))
            .collect();
        v.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        let (ids, times): (Vec<_>, Vec<_>) = v.into_iter().unzip();
        (ids, times)
    }

    /// List available DIA window groups (sorted).
    pub fn dia_window_groups(&self) -> Vec<u32> {
        self.inner.dia_window_groups()
    }

    /// RT-sorted fragment frames and their times for a specific group.
    pub fn fragment_frame_ids_and_times_for_group(&self, window_group: u32) -> (Vec<u32>, Vec<f32>) {
        self.inner.fragment_frame_ids_and_times_for_group_core(window_group)
    }

    /// Merged scan unions for this group (global scan numbers, inclusive).
    pub fn scan_unions_for_group(&self, window_group: u32) -> Option<Vec<(usize, usize)>> {
        self.inner.scan_unions_for_window_group_core(window_group)
    }

    pub fn mz_bounds_for_group(&self, window_group: u32) -> Option<(f32, f32)> {
        self.inner.mz_bounds_for_window_group_core(window_group)
    }

    #[pyo3(signature = (
    window_group,
    im_peaks,
    bin_pad=0,
    smooth_sigma_sec=1.25,
    smooth_trunc_k=3.0,
    min_prom=50.0,
    min_sep_sec=2.0,
    min_width_sec=2.0,
    ppm_per_bin=5.0,
    fallback_if_frames_lt=5,
    fallback_frac_width=0.5
))]
    pub fn expand_rt_for_im_peaks_in_group(
        &self,
        py: Python<'_>,
        window_group: u32,
        im_peaks: Vec<Py<PyImPeak1D>>,
        bin_pad: usize,
        smooth_sigma_sec: f32,
        smooth_trunc_k: f32,
        min_prom: f32,
        min_sep_sec: f32,
        min_width_sec: f32,
        ppm_per_bin: f32,
        fallback_if_frames_lt: usize,
        fallback_frac_width: f32,
    ) -> PyResult<Vec<Vec<Py<PyRtPeak1D>>>> {
        if im_peaks.is_empty() {
            return Ok(Vec::new());
        }

        let im_rs: Vec<ImPeak1D> = im_peaks
            .iter()
            .map(|p| p.borrow(py).inner.as_ref().clone())
            .collect();
        debug_assert!(im_rs.iter().all(|p| p.window_group == Some(window_group)));

        let rt_frames = self.inner.make_rt_frames_for_group(window_group, ppm_per_bin);
        if !rt_frames.is_consistent() {
            return Err(exceptions::PyRuntimeError::new_err("inconsistent RT frames layout"));
        }
        let ctx = rt_frames.ctx();

        let p = RtExpandParams {
            bin_pad,
            smooth_sigma_sec,
            smooth_trunc_k,
            min_prom,
            min_sep_sec,
            min_width_sec,
            fallback_if_frames_lt,
            fallback_frac_width,
        };

        let nested: Vec<Vec<RtPeak1D>> = py.allow_threads(|| {
            expand_many_im_peaks_along_rt(
                &im_rs,
                &rt_frames.frames,
                ctx,
                rt_frames.scale.as_ref(),
                p,
            )
        });

        rtpeaks_to_py_nested(py, nested)
    }

    #[pyo3(signature = (
    im_peaks,
    bin_pad=0,
    smooth_sigma_sec=1.25,
    smooth_trunc_k=3.0,
    min_prom=50.0,
    min_sep_sec=2.0,
    min_width_sec=2.0,
    ppm_per_bin=10.0,
    fallback_if_frames_lt=5,
    fallback_frac_width=0.5
))]
    pub fn expand_rt_for_im_peaks_in_precursor(
        &self,
        py: Python<'_>,
        im_peaks: Vec<Py<PyImPeak1D>>,
        bin_pad: usize,
        smooth_sigma_sec: f32,
        smooth_trunc_k: f32,
        min_prom: f32,
        min_sep_sec: f32,
        min_width_sec: f32,
        ppm_per_bin: f32,
        fallback_if_frames_lt: usize,
        fallback_frac_width: f32,
    ) -> PyResult<Vec<Vec<Py<PyRtPeak1D>>>> {
        if im_peaks.is_empty() {
            return Ok(Vec::new());
        }

        let im_rs: Vec<ImPeak1D> = im_peaks
            .iter()
            .map(|p| p.borrow(py).inner.as_ref().clone())
            .collect();
        debug_assert!(im_rs.iter().all(|p| p.window_group.is_none()));

        let rt_frames = self.inner.make_rt_frames_for_precursor(ppm_per_bin);
        if !rt_frames.is_consistent() {
            return Err(exceptions::PyRuntimeError::new_err("inconsistent RT frames layout"));
        }
        let ctx = rt_frames.ctx();

        let p = RtExpandParams {
            bin_pad,
            smooth_sigma_sec,
            smooth_trunc_k,
            min_prom,
            min_sep_sec,
            min_width_sec,
            fallback_if_frames_lt,
            fallback_frac_width,
        };

        let nested: Vec<Vec<RtPeak1D>> = py.allow_threads(|| {
            expand_many_im_peaks_along_rt(
                &im_rs,
                &rt_frames.frames,
                ctx,
                rt_frames.scale.as_ref(),
                p,
            )
        });

        rtpeaks_to_py_nested(py, nested)
    }

    #[pyo3(signature = (
    window_group,
    ppm_per_bin,
    im_peaks,
    // RtExpandParams (already in seconds)
    bin_pad=0,
    smooth_sigma_sec=1.25,
    smooth_trunc_k=3.0,
    min_prom=50.0,
    min_sep_sec=2.0,
    min_width_sec=2.0,
    fallback_if_frames_lt=5,
    fallback_frac_width=0.5,
    // BuildSpecOpts
    extra_rt_pad=0,
    extra_im_pad=0,
    mz_ppm_pad=5.0,
    mz_hist_bins=64,
    // Eval1DOpts
    refine_mz_once=true,
    refine_k_sigma=3.0,
    attach_axes=true,
    attach_points=false,
    attach_max_points=None,
    // matching constraint + threads
    require_rt_overlap=true,
    num_threads=0,
    min_im_span=12,
))]
    pub fn clusters_for_group(
        &self,
        py: Python<'_>,
        window_group: u32,
        ppm_per_bin: f32,
        im_peaks: Vec<Py<PyImPeak1D>>,
        bin_pad: usize,
        smooth_sigma_sec: f32, smooth_trunc_k: f32,
        min_prom: f32, min_sep_sec: f32, min_width_sec: f32,
        fallback_if_frames_lt: usize, fallback_frac_width: f32,
        extra_rt_pad: usize, extra_im_pad: usize, mz_ppm_pad: f32, mz_hist_bins: usize,
        refine_mz_once: bool, refine_k_sigma: f32,
        attach_axes: bool,
        attach_points: bool, attach_max_points: Option<usize>,
        require_rt_overlap: bool, num_threads: usize,
        min_im_span: usize,
    ) -> PyResult<Vec<Py<PyClusterResult1D>>> {
        let im_rs: Vec<ImPeak1D> = im_peaks.iter()
            .map(|p| p.borrow(py).inner.as_ref().clone())
            .collect();
        debug_assert!(im_rs.iter().all(|p| p.window_group == Some(window_group)));

        let rt_params = RtExpandParams {
            bin_pad,
            smooth_sigma_sec,
            smooth_trunc_k,
            min_prom,
            min_sep_sec,
            min_width_sec,
            fallback_if_frames_lt,
            fallback_frac_width,
        };
        let build_opts = BuildSpecOpts {
            extra_rt_pad,
            extra_im_pad,
            mz_ppm_pad,
            mz_hist_bins,
            ms_level: 2,
            min_im_span,
        };
        let eval_opts = Eval1DOpts {
            refine_mz_once,
            refine_k_sigma,
            attach_axes,
            attach: Attach1DOptions {
                attach_points,
                attach_axes,
                max_points: attach_max_points,
            },
        };

        let results = py.allow_threads(|| {
            self.inner.clusters_for_group(
                window_group,
                ppm_per_bin,
                &im_rs,
                rt_params,
                &build_opts,
                &eval_opts,
                require_rt_overlap,
                num_threads,
            )
        });

        results_to_py(py, results)
    }

    #[pyo3(signature = (
    ppm_per_bin,
    im_peaks,
    bin_pad=0,
    smooth_sigma_sec=1.25, smooth_trunc_k=3.0,
    min_prom=50.0, min_sep_sec=2.0, min_width_sec=2.0,
    fallback_if_frames_lt=5, fallback_frac_width=0.5,
    extra_rt_pad=0, extra_im_pad=0, mz_ppm_pad=5.0, mz_hist_bins=64,
    refine_mz_once=true, refine_k_sigma=3.0,
    attach_axes=true,
    attach_points=false, attach_max_points=None,
    require_rt_overlap=true, num_threads=0,
    min_im_span=12,
))]
    pub fn clusters_for_precursor(
        &self,
        py: Python<'_>,
        ppm_per_bin: f32,
        im_peaks: Vec<Py<PyImPeak1D>>,
        bin_pad: usize,
        smooth_sigma_sec: f32, smooth_trunc_k: f32,
        min_prom: f32, min_sep_sec: f32, min_width_sec: f32,
        fallback_if_frames_lt: usize, fallback_frac_width: f32,
        extra_rt_pad: usize, extra_im_pad: usize, mz_ppm_pad: f32, mz_hist_bins: usize,
        refine_mz_once: bool, refine_k_sigma: f32,
        attach_axes: bool,
        attach_points: bool, attach_max_points: Option<usize>,
        require_rt_overlap: bool, num_threads: usize,
        min_im_span: usize,
    ) -> PyResult<Vec<Py<PyClusterResult1D>>> {
        let im_rs: Vec<ImPeak1D> = im_peaks.iter()
            .map(|p| p.borrow(py).inner.as_ref().clone())
            .collect();
        debug_assert!(im_rs.iter().all(|p| p.window_group.is_none()));

        let rt_params = RtExpandParams {
            bin_pad,
            smooth_sigma_sec,
            smooth_trunc_k,
            min_prom,
            min_sep_sec,
            min_width_sec,
            fallback_if_frames_lt,
            fallback_frac_width,
        };
        let build_opts = BuildSpecOpts {
            extra_rt_pad, extra_im_pad, mz_ppm_pad, mz_hist_bins,
            ms_level: 1,
            min_im_span,
        };
        let eval_opts = Eval1DOpts {
            refine_mz_once,
            refine_k_sigma,
            attach_axes,
            attach: Attach1DOptions {
                attach_points,
                attach_axes,
                max_points: attach_max_points,
            },
        };

        let results = py.allow_threads(|| {
            self.inner.clusters_for_precursor(
                ppm_per_bin,
                &im_rs,
                rt_params,
                &build_opts,
                &eval_opts,
                require_rt_overlap,
                num_threads,
            )
        });

        results_to_py(py, results)
    }
}

fn impeaks_to_py_nested(py: Python<'_>, rows: Vec<Vec<ImPeak1D>>) -> PyResult<Vec<Vec<Py<PyImPeak1D>>>> {
    Ok(rows.into_iter().map(|row| {
        row.into_iter()
            .map(|p| Py::new(py, PyImPeak1D { inner: Arc::new(p) }).unwrap())
            .collect()
    }).collect())
}

// ── REPLACE the helper pick_im_peaks_rows_from_grid with this ──
#[inline]
fn pick_im_peaks_rows_from_grid(
    grid: &MzScanWindowGrid,
    min_prom: f32,
    min_distance_scans: usize,
    min_width_scans: usize,
    use_mobility: bool,
) -> Vec<Vec<ImPeak1D>> {
    use rustdf::cluster::utility::MobilityFn;

    let rows = grid.rows;
    let cols = grid.cols;
    let mob_fn: MobilityFn = if use_mobility { Some(|scan| scan as f32) } else { None };

    (0..rows).map(|r| {
        // gather row across scans (column-major)
        let mut y_s = Vec::with_capacity(cols);
        let mut y_r = Vec::with_capacity(cols);
        for s in 0..cols {
            let val_s = grid.data[s * rows + r];
            y_s.push(val_s);
            let val_r = grid.data_raw.as_ref().map(|dr| dr[s * rows + r]).unwrap_or(val_s);
            y_r.push(val_r);
        }

        let mz_center = grid.mz_center_for_row(r);
        let mz_bounds = grid.mz_bounds_for_row(r);

        find_im_peaks_row(
            &y_s,
            &y_r,
            r,
            mz_center,
            mz_bounds,
            grid.rt_range_frames,
            grid.frame_id_bounds,
            grid.window_group,
            mob_fn,
            min_prom,
            min_distance_scans,   // <-- distance first
            min_width_scans,      // <-- then width
            &grid.scans,          // <-- scan_axis LAST
        )
    }).collect()
}

#[derive(Clone, Copy)]
pub struct StitchParams {
    pub min_overlap_frames: usize,
    pub max_scan_delta: usize,
    pub jaccard_min: f32,
    pub max_mz_row_delta: usize,   // NEW, e.g. 0 (current), 1 or 2
    pub allow_cross_groups: bool,  // NEW if you want to stitch across groups
}

fn same_row_or_close(p:&ImPeak1D, q:&ImPeak1D, d:usize) -> bool {
    p.mz_row.abs_diff(q.mz_row) <= d
}

fn jaccard((a0,a1):(usize,usize),(b0,b1):(usize,usize)) -> f32 {
    let inter = rt_overlap((a0,a1),(b0,b1)) as f32;
    if inter == 0.0 { return 0.0; }
    let len_a = (a1 - a0 + 1) as f32;
    let len_b = (b1 - b0 + 1) as f32;
    inter / (len_a + len_b - inter)
}

// A compact key; bucket scans to avoid too many keys
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct Key { wg: Option<u32>, mz_row: usize, scan_bin: usize }

#[inline]
fn compatible_fast(p:&ImPeak1D, q:&ImPeak1D, sp:&StitchParams) -> bool {
    if !same_row_or_close(p, q, sp.max_mz_row_delta) { return false; }
    if !sp.allow_cross_groups && p.window_group != q.window_group { return false; }
    if (p.scan as isize - q.scan as isize).abs() as usize > sp.max_scan_delta { return false; }
    let ov = rt_overlap(p.rt_bounds, q.rt_bounds);
    if ov < sp.min_overlap_frames { return false; }
    if sp.jaccard_min > 0.0 && jaccard(p.rt_bounds, q.rt_bounds) < sp.jaccard_min { return false; }
    true
}
#[inline]
fn rt_overlap(a: (usize, usize), b: (usize, usize)) -> usize {
    let lo = a.0.max(b.0);
    let hi = a.1.min(b.1);
    hi.saturating_sub(lo).saturating_add(1)
}

#[inline]
fn merge_into(a: &mut ImPeak1D, b: &ImPeak1D) {
    a.rt_bounds = (a.rt_bounds.0.min(b.rt_bounds.0), a.rt_bounds.1.max(b.rt_bounds.1));
    a.frame_id_bounds = (a.frame_id_bounds.0.min(b.frame_id_bounds.0),
                         a.frame_id_bounds.1.max(b.frame_id_bounds.1));
    let w0 = a.apex_smoothed.max(1e-6);
    let w1 = b.apex_smoothed.max(1e-6);
    a.subscan = (a.subscan * w0 + b.subscan * w1) / (w0 + w1);
    a.scan = ((a.scan as f32 * w0 + b.scan as f32 * w1) / (w0 + w1)).round() as usize;
    a.left = a.left.min(b.left);
    a.right = a.right.max(b.right);
    a.width_scans = a.right.saturating_sub(a.left).saturating_add(1);
    a.apex_raw      = a.apex_raw.max(b.apex_raw);
    a.apex_smoothed = a.apex_smoothed.max(b.apex_smoothed);
    a.prominence    = a.prominence.max(b.prominence);
    a.area_raw     += b.area_raw;
    if a.mobility.is_none() { a.mobility = b.mobility; }
}

#[inline]
fn opt_u32_to_i64(x: Option<u32>) -> i64 { x.map(|v| v as i64).unwrap_or(-1) }

#[pyfunction]
#[pyo3(signature = (batched, min_overlap_frames=1, max_scan_delta=1, jaccard_min=0.0, max_mz_row_delta=0, allow_cross_groups=false))]
pub fn stitch_im_peaks_batched_streaming(
    py: Python<'_>,
    batched: Vec<Vec<Vec<Py<PyImPeak1D>>>>, // windows × rows × peaks
    min_overlap_frames: usize,
    max_scan_delta: usize,
    jaccard_min: f32,
    max_mz_row_delta: usize,
    allow_cross_groups: bool,
) -> PyResult<Vec<Py<PyImPeak1D>>> {
    let params = StitchParams {
        min_overlap_frames,
        max_scan_delta: max_scan_delta.max(1),
        jaccard_min,
        max_mz_row_delta,
        allow_cross_groups,
    };

    // Active accumulators keyed by (wg, mz_row, scan_bin)
    let mut active: FxHashMap<Key, ImPeak1D> = FxHashMap::default();
    // Final output
    let mut out: Vec<Py<PyImPeak1D>> = Vec::new();

    // Process windows in given order (RT order); for each row, process its peaks
    for win in batched.into_iter() {
        for row in win.into_iter() {
            let mut row_arcs: Vec<Arc<ImPeak1D>> = row
                .into_iter()
                .map(|p| p.borrow(py).inner.clone())
                .collect();

            row_arcs.sort_unstable_by(|a, b| {
                a.scan.cmp(&b.scan).then(a.rt_bounds.0.cmp(&b.rt_bounds.0))
            });

            for p_arc in row_arcs.into_iter() {
                let p = p_arc.as_ref(); // &ImPeak1D
                let scan_bin = p.scan / params.max_scan_delta;
                let key = Key { wg: p.window_group, mz_row: p.mz_row, scan_bin };

                if let Some(cur) = active.get_mut(&key) {
                    if compatible_fast(cur, p, &params) {
                        merge_into(cur, p);
                        continue;
                    }
                    // finalize current and start new run
                    let flushed = active.remove(&key).unwrap();
                    out.push(Py::new(py, PyImPeak1D { inner: Arc::new(flushed) })?);
                    active.insert(key, owned_copy_im(p)); // <-- owned copy only here
                } else {
                    active.insert(key, owned_copy_im(p)); // <-- first time for this bucket
                }
            }
        }

        // Optional window-boundary compaction:
        // If you have window RT bounds, you can flush accumulators whose rt_right
        // is far behind to reduce the hashmap even further.
        // (Left as a hook; needs passing per-window (rt_lo, rt_hi) if desired.)
    }

    // Flush remaining accumulators
    for (_, v) in active.into_iter() {
        out.push(Py::new(py, PyImPeak1D { inner: Arc::new(v) })?);
    }
    Ok(out)
}

/// IM 1D peak picker without ImRowContext.
/// You provide the few metadata fields directly.
fn find_im_peaks_row_nocontext(
    y_smoothed: &[f32],
    y_raw: &[f32],
    mz_row: usize,
    mz_center: f32,
    mz_bounds: (f32, f32),
    rt_bounds: (usize, usize),
    frame_id_bounds: (u32, u32),
    window_group: Option<u32>,
    scans_axis: &[usize],            // NEW: absolute scan numbers (len == y_* len)
    mobility_of: MobilityFn,
    min_prom: f32,
    min_distance_scans: usize,
    min_width_scans: usize,
) -> Vec<ImPeak1D> {
    let n = y_smoothed.len();
    if n < 3 { return Vec::new(); }

    let row_max = y_raw.iter().copied().fold(0.0f32, f32::max);
    if row_max < min_prom { return Vec::new(); }

    // strict local maxima
    let mut cands = Vec::new();
    for i in 1..n-1 {
        let yi = y_smoothed[i];
        if yi > y_smoothed[i-1] && yi >= y_smoothed[i+1] { cands.push(i); }
    }

    let mut peaks: Vec<ImPeak1D> = Vec::new();
    for &i in &cands {
        let apex = y_smoothed[i];

        // prominence baseline
        let mut l = i; let mut left_min = apex;
        while l > 0 { l -= 1; left_min = left_min.min(y_smoothed[l]); if y_smoothed[l] > apex { break; } }
        let mut r = i; let mut right_min = apex;
        while r + 1 < n { r += 1; right_min = right_min.min(y_smoothed[r]); if y_smoothed[r] > apex { break; } }

        let baseline = left_min.max(right_min);
        let prom = apex - baseline;
        if prom < min_prom { continue; }

        // half-prom crossings (fractional)
        let half = baseline + 0.5 * prom;

        // left crossing
        let mut wl = i;
        while wl > 0 && y_smoothed[wl] > half { wl -= 1; }
        let left_x = if wl < i && wl + 1 < n {
            let y0 = y_smoothed[wl]; let y1 = y_smoothed[wl + 1];
            wl as f32 + if y1 != y0 { (half - y0) / (y1 - y0) } else { 0.0 }
        } else { wl as f32 };

        // right crossing
        let mut wr = i;
        while wr + 1 < n && y_smoothed[wr] > half { wr += 1; }
        let right_x = if wr > i && wr < n {
            let y0 = y_smoothed[wr - 1]; let y1 = y_smoothed[wr];
            (wr - 1) as f32 + if y1 != y0 { (half - y0) / (y1 - y0) } else { 0.0 }
        } else { wr as f32 };

        let width = (right_x - left_x).max(0.0);
        let width_scans = width.round() as usize;
        if width_scans < min_width_scans { continue; }

        // sub-scan apex offset
        let sub = if i > 0 && i + 1 < n {
            quad_subsample(y_smoothed[i - 1], y_smoothed[i], y_smoothed[i + 1]).clamp(-0.5, 0.5)
        } else { 0.0 };

        // NMS by min_distance_scans
        if let Some(last) = peaks.last() {
            if i.abs_diff(last.scan) < min_distance_scans {
                if apex <= last.apex_smoothed { continue; }
                peaks.pop();
            }
        }

        // area on raw between fractional bounds
        let left_idx  = left_x.floor().clamp(0.0, (n-1) as f32) as usize;
        let right_idx = right_x.ceil().clamp(0.0, (n-1) as f32) as usize;
        let area = trapezoid_area_fractional(y_raw, left_x.max(0.0), right_x.min((n-1) as f32));

        let scan_abs  = *scans_axis.get(i).unwrap_or(&i);
        let left_abs  = *scans_axis.get(left_idx).unwrap_or(&left_idx);
        let right_abs = *scans_axis.get(right_idx).unwrap_or(&right_idx);

        let mobility = mobility_of.map(|f| f(i));

        let mut peak = ImPeak1D{
            mz_row,
            mz_center,
            mz_bounds,
            rt_bounds,
            frame_id_bounds,
            window_group,

            scan: i,
            scan_abs,                       // NEW
            left_abs,                       // NEW
            right_abs,
            mobility,
            apex_smoothed: apex,
            apex_raw: y_raw[i],
            prominence: prom,
            left: left_idx,
            right: right_idx,
            left_x,
            right_x,
            width_scans,
            area_raw: area,
            subscan: sub,
            id: 0,
        };

        peak.id = im_peak_id(&peak);
        peaks.push(peak);
    }
    peaks
}

fn rtpeaks_to_py_nested(py: Python<'_>, nested: Vec<Vec<RtPeak1D>>) -> PyResult<Vec<Vec<Py<PyRtPeak1D>>>> {
    nested.into_iter().map(|v| {
        v.into_iter()
            .map(|r| Py::new(py, PyRtPeak1D { inner: r }))
            .collect::<PyResult<Vec<_>>>()
    }).collect()
}

fn results_to_py(py: Python<'_>, v: Vec<ClusterResult1D>) -> PyResult<Vec<Py<PyClusterResult1D>>> {
    v.into_iter()
        .map(|r| Py::new(py, PyClusterResult1D { inner: r }))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (clusters, include_raw_stats=None))]
pub fn export_cluster_arrays<'py>(
    py: Python<'py>,
    clusters: Bound<'py, PyAny>,                // <-- Bound API (by value)
    include_raw_stats: Option<bool>,
) -> PyResult<Py<PyDict>> {
    let include_raw = include_raw_stats.unwrap_or(false);

    // 1) Collect inner results (under GIL), no PyCell/deprecation: extract PyRef directly
    let inners: Vec<ClusterResult1D> = {
        let it = PyIterator::from_bound_object(&clusters)?;
        let mut v = Vec::new();
        for obj in it {
            let any = obj?;                                       // Bound<'_, PyAny>
            let cref: PyRef<PyClusterResult1D> = any.extract()?;  // borrow wrapper directly
            v.push(cref.inner.clone());                           // push the actual inner
        }
        v
    };

    let n = inners.len();

    // 2) Pre-allocate columns
    let mut ms_level:           Vec<i64>   = Vec::with_capacity(n);
    let mut window_group:       Vec<i64>   = Vec::with_capacity(n);
    let mut parent_im_id:       Vec<i64>   = Vec::with_capacity(n);
    let mut parent_rt_id:       Vec<i64>   = Vec::with_capacity(n);

    let mut rt_lo:              Vec<i64>   = Vec::with_capacity(n);
    let mut rt_hi:              Vec<i64>   = Vec::with_capacity(n);
    let mut im_lo:              Vec<i64>   = Vec::with_capacity(n);
    let mut im_hi:              Vec<i64>   = Vec::with_capacity(n);
    let mut mz_lo:              Vec<f32>   = Vec::with_capacity(n);
    let mut mz_hi:              Vec<f32>   = Vec::with_capacity(n);

    let mut raw_sum:            Vec<f32>   = Vec::with_capacity(n);
    let mut volume_proxy:       Vec<f32>   = Vec::with_capacity(n);
    let mut frame_count:        Vec<i64>   = Vec::with_capacity(n);

    let mut has_rt_axis:        Vec<u8>    = Vec::with_capacity(n);
    let mut has_im_axis:        Vec<u8>    = Vec::with_capacity(n);
    let mut has_mz_axis:        Vec<u8>    = Vec::with_capacity(n);

    let mut empty_rt:           Vec<u8>    = Vec::with_capacity(n);
    let mut empty_im:           Vec<u8>    = Vec::with_capacity(n);
    let mut empty_mz:           Vec<u8>    = Vec::with_capacity(n);
    let mut any_empty_dim:      Vec<u8>    = Vec::with_capacity(n);

    // Fit fields
    let mut rt_mu:     Vec<f32> = Vec::with_capacity(n);
    let mut rt_sigma:  Vec<f32> = Vec::with_capacity(n);
    let mut rt_height: Vec<f32> = Vec::with_capacity(n);
    let mut rt_base:   Vec<f32> = Vec::with_capacity(n);
    let mut rt_area:   Vec<f32> = Vec::with_capacity(n);
    let mut rt_r2:     Vec<f32> = Vec::with_capacity(n);
    let mut rt_n:      Vec<i64> = Vec::with_capacity(n);

    let mut im_mu:     Vec<f32> = Vec::with_capacity(n);
    let mut im_sigma:  Vec<f32> = Vec::with_capacity(n);
    let mut im_height: Vec<f32> = Vec::with_capacity(n);
    let mut im_base:   Vec<f32> = Vec::with_capacity(n);
    let mut im_area:   Vec<f32> = Vec::with_capacity(n);
    let mut im_r2:     Vec<f32> = Vec::with_capacity(n);
    let mut im_n:      Vec<i64> = Vec::with_capacity(n);

    let mut mz_mu:     Vec<f32> = Vec::with_capacity(n);
    let mut mz_sigma:  Vec<f32> = Vec::with_capacity(n);
    let mut mz_height: Vec<f32> = Vec::with_capacity(n);
    let mut mz_base:   Vec<f32> = Vec::with_capacity(n);
    let mut mz_area:   Vec<f32> = Vec::with_capacity(n);
    let mut mz_r2:     Vec<f32> = Vec::with_capacity(n);
    let mut mz_n:      Vec<i64> = Vec::with_capacity(n);

    // Raw-points status
    let mut raw_points_attached: Vec<u8>   = Vec::with_capacity(n);
    let mut raw_points_n:        Vec<i64>  = Vec::with_capacity(n);
    let mut raw_empty:           Vec<u8>   = Vec::with_capacity(n);

    // Optional raw aggregates
    let mut raw_intensity_sum:   Vec<f32>  = Vec::with_capacity(n);
    let mut raw_intensity_max:   Vec<f32>  = Vec::with_capacity(n);
    let mut raw_n_frames:        Vec<i64>  = Vec::with_capacity(n);
    let mut raw_n_scans:         Vec<i64>  = Vec::with_capacity(n);
    let mut raw_mz_min:          Vec<f32>  = Vec::with_capacity(n);
    let mut raw_mz_max:          Vec<f32>  = Vec::with_capacity(n);
    let mut raw_rt_min:          Vec<f32>  = Vec::with_capacity(n);
    let mut raw_rt_max:          Vec<f32>  = Vec::with_capacity(n);
    let mut raw_im_min:          Vec<f32>  = Vec::with_capacity(n);
    let mut raw_im_max:          Vec<f32>  = Vec::with_capacity(n);

    // 3) Fill (outside GIL)
    py.allow_threads(|| {
        for cr in inners.into_iter() {
            // windows
            rt_lo.push(cr.rt_window.0 as i64);
            rt_hi.push(cr.rt_window.1 as i64);
            im_lo.push(cr.im_window.0 as i64);
            im_hi.push(cr.im_window.1 as i64);
            mz_lo.push(cr.mz_window.0);
            mz_hi.push(cr.mz_window.1);

            // provenance / stats
            ms_level.push(cr.ms_level as i64);
            window_group.push(opt_u32_to_i64(cr.window_group));
            parent_im_id.push(cr.parent_im_id.unwrap_or(-1));
            parent_rt_id.push(cr.parent_rt_id.unwrap_or(-1));

            raw_sum.push(cr.raw_sum);
            volume_proxy.push(cr.volume_proxy);
            frame_count.push(cr.frame_ids_used.len() as i64);

            // axes present?
            has_rt_axis.push(if cr.rt_axis_sec.as_ref().map_or(false, |v| !v.is_empty()) { 1 } else { 0 });
            has_im_axis.push(if cr.im_axis_scans.as_ref().map_or(false, |v| !v.is_empty()) { 1 } else { 0 });
            has_mz_axis.push(if cr.mz_axis_da.as_ref().map_or(false, |v| !v.is_empty()) { 1 } else { 0 });

            // fits
            let push_fit = |f: &Fit1D,
                                mu: &mut Vec<f32>, sigma: &mut Vec<f32>, height: &mut Vec<f32>,
                                base: &mut Vec<f32>, area: &mut Vec<f32>, r2: &mut Vec<f32>, n_: &mut Vec<i64>| {
                mu.push(f.mu);
                sigma.push(f.sigma);
                height.push(f.height);
                base.push(f.baseline);
                area.push(f.area);
                r2.push(f.r2);
                n_.push(f.n as i64);
            };
            push_fit(&cr.rt_fit, &mut rt_mu, &mut rt_sigma, &mut rt_height, &mut rt_base, &mut rt_area, &mut rt_r2, &mut rt_n);
            push_fit(&cr.im_fit, &mut im_mu, &mut im_sigma, &mut im_height, &mut im_base, &mut im_area, &mut im_r2, &mut im_n);
            push_fit(&cr.mz_fit, &mut mz_mu, &mut mz_sigma, &mut mz_height, &mut mz_base, &mut mz_area, &mut mz_r2, &mut mz_n);

            // empties (derived)
            let is_empty_rt = (cr.rt_fit.n == 0) || (cr.rt_fit.area.abs() <= 0.0);
            let is_empty_im = (cr.im_fit.n == 0) || (cr.im_fit.area.abs() <= 0.0);
            let is_empty_mz = (cr.mz_fit.n == 0) || (cr.mz_fit.area.abs() <= 0.0);

            empty_rt.push(if is_empty_rt { 1 } else { 0 });
            empty_im.push(if is_empty_im { 1 } else { 0 });
            empty_mz.push(if is_empty_mz { 1 } else { 0 });
            any_empty_dim.push(if is_empty_rt || is_empty_im || is_empty_mz { 1 } else { 0 });

            // raw points + optional aggregates
            match &cr.raw_points {
                Some(rp) if !rp.mz.is_empty() => {
                    raw_points_attached.push(1);
                    raw_points_n.push(rp.mz.len() as i64);
                    raw_empty.push(0);

                    if include_raw {
                        // intensity aggregates
                        let mut sum_i = 0.0f64;
                        let mut max_i = 0.0f64;
                        for &v in &rp.intensity {
                            let vf = v as f64;
                            sum_i += vf;
                            if vf > max_i { max_i = vf; }
                        }
                        raw_intensity_sum.push(sum_i as f32);
                        raw_intensity_max.push(max_i as f32);

                        // unique frames/scans
                        let mut frames: std::collections::HashSet<u32> = std::collections::HashSet::new();
                        let mut scans:  std::collections::HashSet<u32> = std::collections::HashSet::new();
                        for &f in &rp.frame { frames.insert(f); }
                        for &s in &rp.scan  { scans.insert(s); }
                        raw_n_frames.push(frames.len() as i64);
                        raw_n_scans.push(scans.len() as i64);

                        // min/max over mz/rt/im
                        let (mut mzmin, mut mzmax) = (f32::INFINITY, f32::NEG_INFINITY);
                        let (mut rtmin, mut rtmax) = (f32::INFINITY, f32::NEG_INFINITY);
                        let (mut immin, mut immax) = (f32::INFINITY, f32::NEG_INFINITY);
                        for i in 0..rp.mz.len() {
                            let mzv = rp.mz[i];
                            let rtv = rp.rt[i];
                            let imv = rp.im[i];
                            if mzv < mzmin { mzmin = mzv; } else if mzv > mzmax { mzmax = mzv; }
                            if rtv < rtmin { rtmin = rtv; } else if rtv > rtmax { rtmax = rtv; }
                            if imv < immin { immin = imv; } else if imv > immax { immax = imv; }
                        }
                        raw_mz_min.push(mzmin); raw_mz_max.push(mzmax);
                        raw_rt_min.push(rtmin); raw_rt_max.push(rtmax);
                        raw_im_min.push(immin); raw_im_max.push(immax);
                    }
                }
                _ => {
                    raw_points_attached.push(0);
                    raw_points_n.push(0);
                    raw_empty.push(1);
                    if include_raw {
                        raw_intensity_sum.push(0.0);
                        raw_intensity_max.push(0.0);
                        raw_n_frames.push(0);
                        raw_n_scans.push(0);
                        raw_mz_min.push(f32::NAN); raw_mz_max.push(f32::NAN);
                        raw_rt_min.push(f32::NAN); raw_rt_max.push(f32::NAN);
                        raw_im_min.push(f32::NAN); raw_im_max.push(f32::NAN);
                    }
                }
            }
        }
    });

    // 4) Build dict of numpy arrays
    let out = PyDict::new_bound(py);
    let arr_i64 = |v: Vec<i64>| PyArray1::from_vec_bound(py, v);
    let arr_u8  = |v: Vec<u8 >| PyArray1::from_vec_bound(py, v);
    let arr_f32 = |v: Vec<f32>| PyArray1::from_vec_bound(py, v);

    out.set_item("ms_level",            arr_i64(ms_level))?;
    out.set_item("window_group",        arr_i64(window_group))?;
    out.set_item("parent_im_id",        arr_i64(parent_im_id))?;
    out.set_item("parent_rt_id",        arr_i64(parent_rt_id))?;

    out.set_item("rt_lo",               arr_i64(rt_lo))?;
    out.set_item("rt_hi",               arr_i64(rt_hi))?;
    out.set_item("im_lo",               arr_i64(im_lo))?;
    out.set_item("im_hi",               arr_i64(im_hi))?;
    out.set_item("mz_lo",               arr_f32(mz_lo))?;
    out.set_item("mz_hi",               arr_f32(mz_hi))?;

    out.set_item("raw_sum",             arr_f32(raw_sum))?;
    out.set_item("volume_proxy",        arr_f32(volume_proxy))?;
    out.set_item("frame_count",         arr_i64(frame_count))?;

    out.set_item("has_rt_axis",         arr_u8 (has_rt_axis))?;
    out.set_item("has_im_axis",         arr_u8 (has_im_axis))?;
    out.set_item("has_mz_axis",         arr_u8 (has_mz_axis))?;

    out.set_item("empty_rt",            arr_u8 (empty_rt))?;
    out.set_item("empty_im",            arr_u8 (empty_im))?;
    out.set_item("empty_mz",            arr_u8 (empty_mz))?;
    out.set_item("any_empty_dim",       arr_u8 (any_empty_dim))?;

    out.set_item("rt_mu",               arr_f32(rt_mu))?;
    out.set_item("rt_sigma",            arr_f32(rt_sigma))?;
    out.set_item("rt_height",           arr_f32(rt_height))?;
    out.set_item("rt_baseline",         arr_f32(rt_base))?;
    out.set_item("rt_area",             arr_f32(rt_area))?;
    out.set_item("rt_r2",               arr_f32(rt_r2))?;
    out.set_item("rt_n",                arr_i64(rt_n))?;

    out.set_item("im_mu",               arr_f32(im_mu))?;
    out.set_item("im_sigma",            arr_f32(im_sigma))?;
    out.set_item("im_height",           arr_f32(im_height))?;
    out.set_item("im_baseline",         arr_f32(im_base))?;
    out.set_item("im_area",             arr_f32(im_area))?;
    out.set_item("im_r2",               arr_f32(im_r2))?;
    out.set_item("im_n",                arr_i64(im_n))?;

    out.set_item("mz_mu",               arr_f32(mz_mu))?;
    out.set_item("mz_sigma",            arr_f32(mz_sigma))?;
    out.set_item("mz_height",           arr_f32(mz_height))?;
    out.set_item("mz_baseline",         arr_f32(mz_base))?;
    out.set_item("mz_area",             arr_f32(mz_area))?;
    out.set_item("mz_r2",               arr_f32(mz_r2))?;
    out.set_item("mz_n",                arr_i64(mz_n))?;

    out.set_item("raw_points_attached", arr_u8 (raw_points_attached))?;
    out.set_item("raw_points_n",        arr_i64(raw_points_n))?;
    out.set_item("raw_empty",           arr_u8 (raw_empty))?;

    if include_raw {
        out.set_item("raw_intensity_sum", arr_f32(raw_intensity_sum))?;
        out.set_item("raw_intensity_max", arr_f32(raw_intensity_max))?;
        out.set_item("raw_n_frames",      arr_i64(raw_n_frames))?;
        out.set_item("raw_n_scans",       arr_i64(raw_n_scans))?;
        out.set_item("raw_mz_min",        arr_f32(raw_mz_min))?;
        out.set_item("raw_mz_max",        arr_f32(raw_mz_max))?;
        out.set_item("raw_rt_min",        arr_f32(raw_rt_min))?;
        out.set_item("raw_rt_max",        arr_f32(raw_rt_max))?;
        out.set_item("raw_im_min",        arr_f32(raw_im_min))?;
        out.set_item("raw_im_max",        arr_f32(raw_im_max))?;
    }

    Ok(out.unbind())
}

#[pymodule]
pub fn py_dia(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimsDatasetDIA>()?;
    m.add_class::<PyMzScanPlan>()?;
    m.add_class::<PyMzScanPlanGroup>()?;
    m.add_class::<PyMzScanWindowGrid>()?;
    m.add_class::<PyImPeak1D>()?;
    m.add_class::<PyRtPeak1D>()?;
    m.add_class::<PyFit1D>()?;
    m.add_class::<PyRawPoints>()?;
    m.add_class::<PyClusterResult1D>()?;
    m.add_function(wrap_pyfunction!(stitch_im_peaks_batched_streaming, m)?)?;
    m.add_function(wrap_pyfunction!(save_clusters_bin, m)?)?;
    m.add_function(wrap_pyfunction!(load_clusters_bin, m)?)?;
    m.add_function(wrap_pyfunction!(export_cluster_arrays, m)?)?;
    Ok(())
}