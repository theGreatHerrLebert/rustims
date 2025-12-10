use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PySlice};
use numpy::{Ix1, Ix2, PyArray, PyArray1, PyArray2, PyReadonlyArray1};
use numpy::ndarray::{Array2, ShapeBuilder};
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use rustdf::cluster::candidates::{fragment_from_cluster, AssignmentResult, CandidateOpts, FragmentIndex, FragmentQueryOpts, PseudoBuildResult, ScoreOpts};
use rustdf::cluster::cluster::{merge_clusters_by_distance, Attach1DOptions, BuildSpecOpts, ClusterMergeDistancePolicy, ClusterResult1D, Eval1DOpts, RawPoints};
use rustdf::cluster::feature::SimpleFeature;
use rustdf::data::dia::{DiaIndex, TimsDatasetDIA};
use rustdf::data::handle::TimsData;
use rustdf::cluster::peak::{TofScanWindowGrid, FrameBinView, build_frame_bin_view, ImPeak1D, RtPeak1D, RtExpandParams, TofRtGrid};
use rustdf::cluster::utility::{TofScale, smooth_vector_gaussian, Fit1D, blur_tof_all_frames, stitch_im_peaks_flat_unordered_impl, StitchParams};
use crate::py_tims_frame::PyTimsFrame;
use crate::py_tims_slice::PyTimsSlice;

#[pyclass]
#[derive(Clone)]
pub struct PyRawPoints {
    pub inner: RawPoints,
}

#[pymethods]
impl PyRawPoints {
    #[getter]
    fn mz(&self) -> Vec<f32> {
        self.inner.mz.clone()
    }

    #[getter]
    fn rt(&self) -> Vec<f32> {
        self.inner.rt.clone()
    }

    #[getter]
    fn im(&self) -> Vec<f32> {
        self.inner.im.clone()
    }

    #[getter]
    fn scan(&self) -> Vec<u32> {
        self.inner.scan.clone()
    }

    #[getter]
    fn intensity(&self) -> Vec<f32> {
        self.inner.intensity.clone()
    }

    #[getter]
    fn tof(&self) -> Vec<i32> {
        self.inner.tof.clone()
    }

    #[getter]
    fn frame(&self) -> Vec<u32> {
        self.inner.frame.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RawPoints(n={})",
            self.inner.mz.len()
        )
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyClusterResult1D {
    pub inner: ClusterResult1D,
}

#[pymethods]
impl PyClusterResult1D {
    /// We don’t want Python constructing these directly; they come from Rust.
    #[new]
    fn new() -> PyResult<Self> {
        Err(exceptions::PyRuntimeError::new_err(
            "PyClusterResult1D objects are created by the clustering APIs, not directly.",
        ))
    }

    fn __repr__(&self) -> String {
        let (rt_lo, rt_hi) = self.inner.rt_window;
        let (im_lo, im_hi) = self.inner.im_window;
        let (tof_lo, tof_hi) = self.inner.tof_window;
        let ms = self.inner.ms_level;
        format!(
            "ClusterResult1D(ms_level={}, rt=[{}..{}], im=[{}..{}], tof=[{}..{}], raw_sum={:.1})",
            ms, rt_lo, rt_hi, im_lo, im_hi, tof_lo, tof_hi, self.inner.raw_sum
        )
    }

    // -------- windows ----------------------------------------------------

    #[getter]
    pub fn cluster_id(&self) -> u64 { self.inner.cluster_id }

    #[getter]
    fn rt_window(&self) -> (usize, usize) {
        self.inner.rt_window
    }

    #[getter]
    fn im_window(&self) -> (usize, usize) {
        self.inner.im_window
    }

    #[getter]
    fn tof_window(&self) -> (i32, i32) {
        self.inner.tof_index_window
    }

    #[getter]
    fn mz_window(&self) -> Option<(f32, f32)> {
        self.inner.mz_window
    }

    // -------- fits (return PyFit1D) --------------------------------------

    #[getter]
    fn rt_fit(&self) -> PyFit1D {
        PyFit1D {
            inner: self.inner.rt_fit.clone(),
        }
    }

    #[getter]
    fn im_fit(&self) -> PyFit1D {
        PyFit1D {
            inner: self.inner.im_fit.clone(),
        }
    }

    #[getter]
    fn tof_fit(&self) -> PyFit1D {
        PyFit1D {
            inner: self.inner.tof_fit.clone(),
        }
    }

    #[getter]
    fn mz_fit(&self) -> Option<PyFit1D> {
        self.inner
            .mz_fit
            .clone()
            .map(|f| PyFit1D { inner: f })
    }

    // And keep the scalar shortcuts if you like:

    #[getter]
    fn rt_mu(&self) -> f32 {
        self.inner.rt_fit.mu
    }

    #[getter]
    fn rt_sigma(&self) -> f32 {
        self.inner.rt_fit.sigma
    }

    #[getter]
    fn rt_height(&self) -> f32 {
        self.inner.rt_fit.height
    }

    #[getter]
    fn rt_area(&self) -> f32 {
        self.inner.rt_fit.area
    }

    #[getter]
    fn im_mu(&self) -> f32 {
        self.inner.im_fit.mu
    }

    #[getter]
    fn im_sigma(&self) -> f32 {
        self.inner.im_fit.sigma
    }

    #[getter]
    fn im_height(&self) -> f32 {
        self.inner.im_fit.height
    }

    #[getter]
    fn im_area(&self) -> f32 {
        self.inner.im_fit.area
    }

    #[getter]
    fn tof_mu(&self) -> f32 {
        self.inner.tof_fit.mu
    }

    #[getter]
    fn tof_sigma(&self) -> f32 {
        self.inner.tof_fit.sigma
    }

    #[getter]
    fn tof_height(&self) -> f32 {
        self.inner.tof_fit.height
    }

    #[getter]
    fn tof_area(&self) -> f32 {
        self.inner.tof_fit.area
    }

    #[getter]
    fn mz_mu(&self) -> Option<f32> {
        self.inner.mz_fit.as_ref().map(|f| f.mu)
    }

    #[getter]
    fn mz_sigma(&self) -> Option<f32> {
        self.inner.mz_fit.as_ref().map(|f| f.sigma)
    }

    #[getter]
    fn mz_height(&self) -> Option<f32> {
        self.inner.mz_fit.as_ref().map(|f| f.height)
    }

    #[getter]
    fn mz_area(&self) -> Option<f32> {
        self.inner.mz_fit.as_ref().map(|f| f.area)
    }

    // -------- scalar meta ------------------------------------------------

    #[getter]
    fn raw_sum(&self) -> f32 {
        self.inner.raw_sum
    }

    #[getter]
    fn volume_proxy(&self) -> f32 {
        self.inner.volume_proxy
    }

    #[getter]
    fn ms_level(&self) -> u8 {
        self.inner.ms_level
    }

    #[getter]
    fn window_group(&self) -> Option<u32> {
        self.inner.window_group
    }

    #[getter]
    fn parent_im_id(&self) -> Option<i64> {
        self.inner.parent_im_id
    }

    #[getter]
    fn parent_rt_id(&self) -> Option<i64> {
        self.inner.parent_rt_id
    }

    // -------- arrays / axes ----------------------------------------------

    #[getter]
    fn frame_ids_used(&self) -> Vec<u32> {
        self.inner.frame_ids_used.clone()
    }

    #[getter]
    fn rt_axis_sec(&self) -> Option<Vec<f32>> {
        self.inner.rt_axis_sec.clone()
    }

    #[getter]
    fn im_axis_scans(&self) -> Option<Vec<usize>> {
        self.inner.im_axis_scans.clone()
    }

    #[getter]
    fn mz_axis_da(&self) -> Option<Vec<f32>> {
        self.inner.mz_axis_da.clone()
    }

    // -------- raw points -------------------------------------------------

    #[getter]
    fn has_raw_points(&self) -> bool {
        self.inner.raw_points.is_some()
    }

    #[getter]
    fn raw_points(&self) -> Option<PyRawPoints> {
        self.inner
            .raw_points
            .clone()
            .map(|rp| PyRawPoints { inner: rp })
    }

    // Optional RT XIC, aligned with rt_window and rt_axis_sec (if present).
    #[getter]
    fn rt_xic(&self) -> Option<Vec<f32>> {
        self.inner.rt_trace.clone()
    }

    /// Optional IM trace, aligned with im_window and im_axis_scans (if present).
    #[getter]
    fn im_xic(&self) -> Option<Vec<f32>> {
        self.inner.im_trace.clone()
    }

    pub fn drop_raw_data(&mut self) {
        self.inner.raw_points = None;
    }
}

#[pyclass]
pub struct PyTofScanPlanGroup {
    ds: Py<PyAny>,
    window_group: u32,

    // planned axes + schedule
    scale: Arc<TofScale>,
    frame_ids_sorted: Vec<u32>,
    frame_times: Vec<f32>,
    windows_idx: Vec<(usize, usize)>,
    rows: usize,
    global_num_scans: usize,

    // exec params
    maybe_sigma_scans: Option<f32>,
    maybe_sigma_tof_bins: Option<f32>,
    truncate: f32,
    num_threads: usize,

    // optional accel
    views: Option<Vec<FrameBinView>>,

    // iterator state
    cur: usize,
}

#[pymethods]
impl PyTofScanPlanGroup {
    #[new]
    #[pyo3(signature = (
        ds,
        window_group,
        tof_step,
        rt_window_sec,
        rt_hop_sec,
        num_threads=4,
        maybe_sigma_scans=None,
        maybe_sigma_tof_bins=None,
        truncate=3.0,
        precompute_views=false
    ))]
    pub fn new(
        py: Python<'_>,
        ds: Py<PyAny>,
        window_group: u32,
        tof_step: i32,
        rt_window_sec: f32,
        rt_hop_sec: f32,
        num_threads: usize,
        maybe_sigma_scans: Option<f32>,
        maybe_sigma_tof_bins: Option<f32>,
        truncate: f32,
        precompute_views: bool,
    ) -> PyResult<Self> {
        // --- collect frames/times and discover TOF scale
        let (frame_ids_sorted, frame_times, scale, rows, global_num_scans, views) = {
            let ds_bound = ds.bind(py);
            let ds_obj: &Bound<PyTimsDatasetDIA> = ds_bound.downcast()?;
            let ds_ref = ds_obj.borrow();

            // 1) RT-sort MS2 frames for this group
            let (frame_ids_sorted, frame_times) =
                ds_ref.fragment_frame_ids_and_times_for_group(window_group);

            // 2) materialize frames once to discover TOF range + global scan max
            let frames = ds_ref.inner.get_slice(frame_ids_sorted.clone(), num_threads).frames;

            // Derive TOF scale from frames (adjust builder to your TofScale API)
            let scale = TofScale::build_from_frames(&frames, tof_step).ok_or_else(|| {
                exceptions::PyRuntimeError::new_err("no TOF range found for group")
            })?;
            let rows = scale.num_bins();

            // 3) global scan axis size
            let global_num_scans = ds_ref.max_global_num_scans();

            // 4) optional precomputed per-frame views
            let views = if precompute_views {
                Some((0..frames.len())
                    .into_par_iter()
                    .map(|i| build_frame_bin_view(frames[i].clone(), &scale, global_num_scans))
                    .collect())
            } else {
                None
            };

            (frame_ids_sorted, frame_times, scale, rows, global_num_scans, views)
        };

        // 5) Build RT window schedule (same logic as MS1 plan)
        let fps = if frame_times.len() < 2 {
            1.0
        } else {
            let mut d: Vec<f32> = frame_times.windows(2).map(|w| w[1] - w[0]).collect();
            d.sort_by(|a, b| a.partial_cmp(&b).unwrap());
            1.0 / d[d.len() / 2].max(1e-6)
        };
        let mut win_len = (rt_window_sec * fps).max(1.0).round() as usize;
        let hop_len = (rt_hop_sec * fps).max(1.0).round() as usize;
        let n_frames = frame_ids_sorted.len().max(1);
        win_len = win_len.min(n_frames);
        let hop_len = hop_len.max(1);

        let mut windows_idx: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        while start < n_frames {
            let end = (start + win_len - 1).min(n_frames - 1);
            windows_idx.push((start, end));
            if end + 1 >= n_frames {
                break;
            }
            start = (start + hop_len).min(n_frames - 1);
            if start == 0 {
                break;
            }
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
            maybe_sigma_tof_bins,
            truncate,
            num_threads,
            views,
            cur: 0,
        })
    }

    // ---- Introspection (parity with PyTofScanPlan)
    #[getter]
    fn rows(&self) -> usize { self.rows }

    #[getter]
    fn global_num_scans(&self) -> usize { self.global_num_scans }

    #[getter]
    fn num_windows(&self) -> usize { self.windows_idx.len() }

    #[getter]
    fn window_group(&self) -> u32 { self.window_group }

    #[getter]
    fn frame_times<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec_bound(py, self.frame_times.clone()).unbind())
    }

    #[getter]
    fn frame_ids<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<u32>>> {
        Ok(PyArray1::from_vec_bound(py, self.frame_ids_sorted.clone()).unbind())
    }

    #[getter]
    fn tof_centers<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec_bound(py, self.scale.centers.clone()).unbind())
    }

    pub fn bounds(&self, i: usize) -> Option<(usize, usize)> {
        self.windows_idx.get(i).copied()
    }

    pub fn bounds_frame_ids(&self, i: usize) -> Option<(u32, u32)> {
        self.windows_idx
            .get(i)
            .map(|(lo, hi)| (self.frame_ids_sorted[*lo], self.frame_ids_sorted[*hi]))
    }

    #[getter]
    fn fragment_frame_id_bounds(&self) -> Option<(u32, u32)> {
        if self.frame_ids_sorted.is_empty() {
            None
        } else {
            Some((
                self.frame_ids_sorted[0],
                *self.frame_ids_sorted.last().unwrap(),
            ))
        }
    }

    // --- Python iteration protocol + __getitem__
    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.cur = 0;
        slf
    }

    fn __next__(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PyTofScanWindowGrid>>> {
        if slf.cur >= slf.windows_idx.len() {
            return Ok(None);
        }
        let i = slf.cur;
        slf.cur += 1;
        let grid = slf.build_window(py, i)?;
        Ok(Some(Py::new(py, PyTofScanWindowGrid { inner: grid })?))
    }

    fn __len__(&self) -> usize {
        self.windows_idx.len()
    }

    fn __getitem__(&self, py: Python<'_>, idx: &Bound<PyAny>) -> PyResult<PyObject> {
        // int (supports negative)
        if let Ok(i_signed) = idx.extract::<isize>() {
            let n = self.windows_idx.len() as isize;
            let j = if i_signed < 0 { n + i_signed } else { i_signed };
            if j < 0 || j >= n {
                return Err(exceptions::PyIndexError::new_err("index out of range"));
            }
            let grid = self.build_window(py, j as usize)?;
            let obj = Py::new(py, PyTofScanWindowGrid { inner: grid })?;
            return Ok(obj.into_py(py));
        }

        // slice
        if let Ok(slice) = idx.downcast::<PySlice>() {
            let indices = slice.indices(self.windows_idx.len() as isize)?;
            let (start, stop, step) = (indices.start, indices.stop, indices.step);
            let out = PyList::empty_bound(py);
            let mut i = start;
            if step > 0 {
                while i < stop {
                    let grid = self.build_window(py, i as usize)?;
                    out.append(Py::new(py, PyTofScanWindowGrid { inner: grid })?)?;
                    i += step;
                }
            } else if step < 0 {
                while i > stop {
                    let grid = self.build_window(py, i as usize)?;
                    out.append(Py::new(py, PyTofScanWindowGrid { inner: grid })?)?;
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
    ) -> PyResult<Vec<Py<PyTofScanWindowGrid>>> {
        let n = self.windows_idx.len();
        if start >= n {
            return Ok(Vec::new());
        }
        let end = (start + count).min(n);
        let mut out = Vec::with_capacity(end - start);
        for i in start..end {
            let grid = self.build_window(py, i)?;
            out.push(Py::new(py, PyTofScanWindowGrid { inner: grid })?);
        }
        Ok(out)
    }

    #[pyo3(name = "get_batch_par")]
    pub fn get_batch_par(
        &self,
        py: Python<'_>,
        start: usize,
        count: usize,
    ) -> PyResult<Vec<Py<PyTofScanWindowGrid>>> {
        let n = self.windows_idx.len();
        if start >= n {
            return Ok(Vec::new());
        }
        let end = (start + count).min(n);

        let indices: Vec<usize> = (start..end).collect();

        // Build all grids without holding the GIL
        let built: Vec<Result<TofScanWindowGrid, String>> = py.allow_threads(|| {
            if self.views.is_some() {
                // Fully GIL-free path when views are precomputed
                indices
                    .par_iter()
                    .map(|&i| Ok(self.build_window_from_views(i)))
                    .collect()
            } else {
                // Fall back to calling build_window under an assumed GIL.
                indices
                    .par_iter()
                    .map(|&i| {
                        let py = unsafe { Python::assume_gil_acquired() };
                        self.build_window(py, i).map_err(|e| format!("{e}"))
                    })
                    .collect()
            }
        });

        // Convert to Python objects, preserving order and surfacing any errors
        let mut out = Vec::with_capacity(built.len());
        for item in built {
            match item {
                Ok(grid) => out.push(Py::new(py, PyTofScanWindowGrid { inner: grid })?),
                Err(msg) => return Err(exceptions::PyRuntimeError::new_err(msg)),
            }
        }
        Ok(out)
    }

    /// Continuous TOF center for a fractional grid index mu (float).
    #[pyo3(name = "tof_center_at")]
    pub fn tof_center_at(&self, mu: f32) -> f32 {
        self.scale.center_at(mu)
    }

    #[getter]
    pub fn tof_min(&self) -> i32 {
        self.scale.tof_min
    }

    #[getter]
    pub fn tof_max(&self) -> i32 {
        self.scale.tof_max
    }

    #[pyo3(name = "tof_index_of")]
    pub fn tof_index_of(&self, tof: i32) -> Option<usize> {
        self.scale.index_of_tof(tof)
    }

    #[pyo3(name = "tof_center_for_row")]
    pub fn tof_center_for_row(&self, row: usize) -> f32 {
        self.scale.center(row)
    }
}

impl PyTofScanPlanGroup {
    fn build_window(&self, py: Python<'_>, i: usize) -> PyResult<TofScanWindowGrid> {
        if self.views.is_some() {
            // Fast path: no dataset access, no GIL work
            return Ok(self.build_window_from_views(i));
        }

        let (lo, hi) = self.windows_idx[i];
        let rows = self.rows;
        let cols = self.global_num_scans;
        let do_smooth = self.maybe_sigma_scans.unwrap_or(0.0) > 0.0;
        let do_blur_tof = self.maybe_sigma_tof_bins.unwrap_or(0.0) > 0.0;

        // Views for this window (group frames only)
        let mut views_local: Vec<FrameBinView> = {
            let fids = self.frame_ids_sorted[lo..=hi].to_vec();
            let frames = {
                let ds_bound = self.ds.bind(py);
                let ds_obj: &Bound<PyTimsDatasetDIA> = ds_bound.downcast()?;
                let ds_ref = ds_obj.borrow();
                ds_ref.inner.get_slice(fids, self.num_threads).frames
            };
            frames
                .into_par_iter()
                .map(|fr| build_frame_bin_view(fr, &self.scale, cols))
                .collect()
        };

        if do_blur_tof {
            let sigma_bins = self.maybe_sigma_tof_bins.unwrap();
            let trunc = self.truncate;
            views_local = blur_tof_all_frames(&views_local, sigma_bins, trunc);
        }

        // Accumulate onto the **global** scan axis
        let mut raw = vec![0.0f32; rows * cols];
        for v in &views_local {
            for b_i in 0..v.unique_bins.len() {
                let row = v.unique_bins[b_i];
                let start = v.offsets[b_i];
                let end = v.offsets[b_i + 1];
                for j in start..end {
                    let s_phys = v.scan_idx[j] as usize;
                    if s_phys < cols {
                        raw[s_phys * rows + row] += v.intensity[j];
                    }
                }
            }
        }

        let data = if do_smooth {
            smooth_rows_parallel(
                &raw,
                rows,
                cols,
                self.maybe_sigma_scans.unwrap(),
                self.truncate,
            )
        } else {
            raw.clone()
        };

        let frame_id_bounds = (self.frame_ids_sorted[lo], self.frame_ids_sorted[hi]);

        Ok(TofScanWindowGrid {
            scale: self.scale.clone(),
            rt_range_frames: (lo, hi),
            rt_range_sec: (self.frame_times[lo], self.frame_times[hi]),
            frame_id_bounds,
            window_group: Some(self.window_group),
            scans: (0..cols).collect(),
            data: Some(data),
            rows,
            cols,
            data_raw: if do_smooth || do_blur_tof { Some(raw) } else { None },
        })
    }

    #[inline]
    fn build_window_from_views(&self, i: usize) -> TofScanWindowGrid {
        let (lo, hi) = self.windows_idx[i];
        let rows = self.rows;
        let cols = self.global_num_scans;
        let do_smooth = self.maybe_sigma_scans.unwrap_or(0.0) > 0.0;
        let do_blur_tof = self.maybe_sigma_tof_bins.unwrap_or(0.0) > 0.0;

        // Views for this window are already precomputed; no Python / dataset needed.
        let mut views_local: Vec<FrameBinView> =
            (lo..=hi).map(|k| self.views.as_ref().unwrap()[k].clone()).collect();

        if do_blur_tof {
            let sigma_bins = self.maybe_sigma_tof_bins.unwrap();
            let trunc = self.truncate;
            views_local = blur_tof_all_frames(&views_local, sigma_bins, trunc);
        }

        // Accumulate onto the fixed global scan axis
        let mut raw = vec![0.0f32; rows * cols];
        for v in &views_local {
            for b_i in 0..v.unique_bins.len() {
                let row = v.unique_bins[b_i];
                let start = v.offsets[b_i];
                let end = v.offsets[b_i + 1];
                for j in start..end {
                    let s_phys = v.scan_idx[j] as usize;
                    if s_phys < cols {
                        raw[s_phys * rows + row] += v.intensity[j];
                    }
                }
            }
        }

        let data = if do_smooth {
            smooth_rows_parallel(
                &raw,
                rows,
                cols,
                self.maybe_sigma_scans.unwrap(),
                self.truncate,
            )
        } else {
            raw.clone()
        };

        let frame_id_bounds = (self.frame_ids_sorted[lo], self.frame_ids_sorted[hi]);

        TofScanWindowGrid {
            scale: self.scale.clone(),
            rt_range_frames: (lo, hi),
            rt_range_sec: (self.frame_times[lo], self.frame_times[hi]),
            frame_id_bounds,
            window_group: Some(self.window_group),
            scans: (0..cols).collect(),
            data: Some(data),
            rows,
            cols,
            data_raw: if do_smooth || do_blur_tof { Some(raw) } else { None },
        }
    }
}

#[pyclass]
pub struct PyTofScanPlan {
    ds: Py<PyAny>,                 // keeps dataset alive; downcast when needed

    // planned axes + schedule
    scale: Arc<TofScale>,
    frame_ids_sorted: Vec<u32>,
    frame_times: Vec<f32>,
    windows_idx: Vec<(usize, usize)>,
    rows: usize,
    global_num_scans: usize,

    // exec params
    maybe_sigma_scans: Option<f32>,
    maybe_sigma_tof_bins: Option<f32>,
    truncate: f32,
    num_threads: usize,

    // optional accel
    views: Option<Vec<FrameBinView>>,

    // iterator state
    cur: usize,
}

#[pymethods]
impl PyTofScanPlan {
    #[new]
    #[pyo3(signature = (
        ds,
        tof_step,
        rt_window_sec,
        rt_hop_sec,
        num_threads=4,
        maybe_sigma_scans=None,
        maybe_sigma_tof_bins=None,
        truncate=3.0,
        precompute_views=false
    ))]
    pub fn new(
        py: Python<'_>,
        ds: Py<PyAny>,
        tof_step: i32,
        rt_window_sec: f32,
        rt_hop_sec: f32,
        num_threads: usize,
        maybe_sigma_scans: Option<f32>,
        maybe_sigma_tof_bins: Option<f32>,
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

            // 2) Discover TOF scale from all precursor frames
            let frames = ds_ref.inner.get_slice(frame_ids_sorted.clone(), num_threads).frames;
            let scale = TofScale::build_from_frames(&frames, tof_step).ok_or_else(|| {
                exceptions::PyRuntimeError::new_err("no TOF range found for precursor frames")
            })?;
            let rows = scale.num_bins();
            let global_num_scans = ds_ref.max_global_num_scans();

            // 3) Optional precomputed views
            let views = if precompute_views {
                Some(
                    (0..frames.len())
                        .into_par_iter()
                        .map(|i| build_frame_bin_view(frames[i].clone(), &scale, global_num_scans))
                        .collect(),
                )
            } else {
                None
            };

            (frame_ids_sorted, frame_times, scale, rows, global_num_scans, views)
        }; // <— borrow dropped here

        // 4) RT window schedule
        let fps = if frame_times.len() < 2 {
            1.0
        } else {
            let mut d: Vec<f32> = frame_times.windows(2).map(|w| w[1] - w[0]).collect();
            d.sort_by(|a, b| a.partial_cmp(&b).unwrap());
            1.0 / d[d.len() / 2].max(1e-6)
        };
        let mut win_len = (rt_window_sec * fps).max(1.0).round() as usize;
        let hop_len = (rt_hop_sec * fps).max(1.0).round() as usize;
        let n_frames = frame_ids_sorted.len().max(1);
        win_len = win_len.min(n_frames);
        let hop_len = hop_len.max(1);

        let mut windows_idx: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        while start < n_frames {
            let end = (start + win_len - 1).min(n_frames - 1);
            windows_idx.push((start, end));
            if end + 1 >= n_frames {
                break;
            }
            start = (start + hop_len).min(n_frames - 1);
            if start == 0 {
                break;
            }
        }

        Ok(Self {
            ds,
            scale: Arc::new(scale),
            frame_ids_sorted,
            frame_times,
            windows_idx,
            rows,
            global_num_scans,
            maybe_sigma_scans,
            maybe_sigma_tof_bins,
            truncate,
            num_threads,
            views,
            cur: 0,
        })
    }

    // ---- Introspection
    #[getter]
    fn rows(&self) -> usize { self.rows }

    #[getter]
    fn global_num_scans(&self) -> usize { self.global_num_scans }

    #[getter]
    fn num_windows(&self) -> usize { self.windows_idx.len() }

    #[getter]
    fn frame_times<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec_bound(py, self.frame_times.clone()).unbind())
    }

    #[getter]
    fn frame_ids<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<u32>>> {
        Ok(PyArray1::from_vec_bound(py, self.frame_ids_sorted.clone()).unbind())
    }

    #[getter]
    fn tof_centers<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec_bound(py, self.scale.centers.clone()).unbind())
    }

    pub fn bounds(&self, i: usize) -> Option<(usize, usize)> {
        self.windows_idx.get(i).copied()
    }

    /// Continuous TOF center for a fractional grid index mu (float).
    #[pyo3(name = "tof_center_at")]
    pub fn tof_center_at(&self, mu: f32) -> f32 {
        self.scale.center_at(mu)
    }

    #[getter]
    pub fn tof_min(&self) -> i32 {
        self.scale.tof_min
    }

    #[getter]
    pub fn tof_max(&self) -> i32 {
        self.scale.tof_max
    }

    #[pyo3(name = "tof_index_of")]
    pub fn tof_index_of(&self, tof: i32) -> Option<usize> {
        self.scale.index_of_tof(tof)
    }

    #[pyo3(name = "tof_center_for_row")]
    pub fn tof_center_for_row(&self, row: usize) -> f32 {
        self.scale.center(row)
    }

    pub fn get(
        &self,
        py: Python<'_>,
        i: usize,
    ) -> PyResult<Option<Py<PyTofScanWindowGrid>>> {
        if i >= self.windows_idx.len() {
            return Ok(None);
        }
        let grid = self.build_window(py, i)?;  // <-- note the `?`
        Ok(Some(Py::new(py, PyTofScanWindowGrid { inner: grid })?))
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.cur = 0;
        slf
    }

    fn __next__(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PyTofScanWindowGrid>>> {
        if slf.cur >= slf.windows_idx.len() {
            return Ok(None);
        }
        let i = slf.cur;
        slf.cur += 1;
        let grid = slf.build_window(py, i)?;  // <-- `?`
        Ok(Some(Py::new(py, PyTofScanWindowGrid { inner: grid })?))
    }

    fn __len__(&self) -> usize { self.windows_idx.len() }

    pub fn bounds_frame_ids(&self, i: usize) -> Option<(u32, u32)> {
        self.windows_idx
            .get(i)
            .map(|(lo, hi)| (self.frame_ids_sorted[*lo], self.frame_ids_sorted[*hi]))
    }

    #[getter]
    fn precursor_frame_id_bounds(&self) -> Option<(u32, u32)> {
        if self.frame_ids_sorted.is_empty() {
            None
        } else {
            Some((
                self.frame_ids_sorted[0],
                *self.frame_ids_sorted.last().unwrap(),
            ))
        }
    }

    fn __getitem__(&self, py: Python<'_>, idx: &Bound<PyAny>) -> PyResult<PyObject> {
        // integer first
        if let Ok(i_signed) = idx.extract::<isize>() {
            let n = self.windows_idx.len() as isize;
            let j = if i_signed < 0 { n + i_signed } else { i_signed };
            if j < 0 || j >= n {
                return Err(exceptions::PyIndexError::new_err("index out of range"));
            }
            let grid = self.build_window(py, j as usize)?;  // <-- `?`
            let obj = Py::new(py, PyTofScanWindowGrid { inner: grid })?;
            return Ok(obj.into_py(py));
        }

        // slice
        if let Ok(slice) = idx.downcast::<PySlice>() {
            let indices = slice.indices(self.windows_idx.len() as isize)?;
            let (start, stop, step) = (indices.start, indices.stop, indices.step);
            let out = PyList::empty_bound(py);

            let mut i = start;
            if step > 0 {
                while i < stop {
                    let grid = self.build_window(py, i as usize)?;  // <-- `?`
                    out.append(Py::new(py, PyTofScanWindowGrid { inner: grid })?)?;
                    i += step;
                }
            } else if step < 0 {
                while i > stop {
                    let grid = self.build_window(py, i as usize)?;  // <-- `?`
                    out.append(Py::new(py, PyTofScanWindowGrid { inner: grid })?)?;
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
    ) -> PyResult<Vec<Py<PyTofScanWindowGrid>>> {
        let n = self.windows_idx.len();
        if start >= n {
            return Ok(Vec::new());
        }
        let end = (start + count).min(n);
        let mut out = Vec::with_capacity(end - start);
        for i in start..end {
            let grid = self.build_window(py, i)?;  // <-- `?`
            out.push(Py::new(py, PyTofScanWindowGrid { inner: grid })?);
        }
        Ok(out)
    }

    #[pyo3(name = "get_batch_par")]
    pub fn get_batch_par(
        &self,
        py: Python<'_>,
        start: usize,
        count: usize,
    ) -> PyResult<Vec<Py<PyTofScanWindowGrid>>> {
        let n = self.windows_idx.len();
        if start >= n {
            return Ok(Vec::new());
        }
        let end = (start + count).min(n);
        let indices: Vec<usize> = (start..end).collect();

        let built: Vec<Result<TofScanWindowGrid, String>> = py.allow_threads(|| {
            if self.views.is_some() {
                indices
                    .par_iter()
                    .map(|&i| Ok(self.build_window_from_views(i)))
                    .collect()
            } else {
                indices
                    .par_iter()
                    .map(|&i| {
                        let py = unsafe { Python::assume_gil_acquired() };
                        // now build_window returns Result, so map_err makes sense
                        self.build_window(py, i).map_err(|e| format!("{e}"))
                    })
                    .collect()
            }
        });

        let mut out = Vec::with_capacity(built.len());
        for item in built {
            match item {
                Ok(grid) => out.push(Py::new(py, PyTofScanWindowGrid { inner: grid })?),
                Err(msg) => return Err(exceptions::PyRuntimeError::new_err(msg)),
            }
        }
        Ok(out)
    }
}

impl PyTofScanPlan {
    fn build_window(&self, py: Python<'_>, i: usize) -> PyResult<TofScanWindowGrid> {
        if self.views.is_some() {
            return Ok(self.build_window_from_views(i));
        }

        let (lo, hi) = self.windows_idx[i];
        let rows = self.rows;
        let cols = self.global_num_scans;
        let do_smooth   = self.maybe_sigma_scans.unwrap_or(0.0) > 0.0;
        let do_blur_tof = self.maybe_sigma_tof_bins.unwrap_or(0.0) > 0.0;

        let mut views_local: Vec<FrameBinView> = {
            let fids = self.frame_ids_sorted[lo..=hi].to_vec();
            let frames = {
                let ds_bound = self.ds.bind(py);
                let ds_obj: &Bound<PyTimsDatasetDIA> = ds_bound.downcast()?;  // <-- `?` now valid
                let ds_ref = ds_obj.borrow();
                ds_ref.inner.get_slice(fids, self.num_threads).frames
            };
            frames
                .into_par_iter()
                .map(|fr| build_frame_bin_view(fr, &self.scale, cols))
                .collect()
        };

        if do_blur_tof {
            let sigma_bins = self.maybe_sigma_tof_bins.unwrap();
            let trunc      = self.truncate;
            views_local = blur_tof_all_frames(&views_local, sigma_bins, trunc);
        }

        let mut raw = vec![0.0f32; rows * cols];
        for v in &views_local {
            for b_i in 0..v.unique_bins.len() {
                let row   = v.unique_bins[b_i];
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
            smooth_rows_parallel(
                &raw,
                rows,
                cols,
                self.maybe_sigma_scans.unwrap(),
                self.truncate,
            )
        } else {
            raw.clone()
        };

        let frame_id_bounds = (self.frame_ids_sorted[lo], self.frame_ids_sorted[hi]);

        Ok(TofScanWindowGrid {
            scale: self.scale.clone(),
            rt_range_frames: (lo, hi),
            rt_range_sec: (self.frame_times[lo], self.frame_times[hi]),
            frame_id_bounds,
            window_group: None,
            scans: (0..cols).collect(),
            data: Some(data),
            rows,
            cols,
            data_raw: if do_smooth || do_blur_tof { Some(raw) } else { None },
        })
    }

    #[inline]
    fn build_window_from_views(&self, i: usize) -> TofScanWindowGrid {
        let (lo, hi) = self.windows_idx[i];
        let rows = self.rows;
        let cols = self.global_num_scans;
        let do_smooth   = self.maybe_sigma_scans.unwrap_or(0.0) > 0.0;
        let do_blur_tof = self.maybe_sigma_tof_bins.unwrap_or(0.0) > 0.0;

        let mut views_local: Vec<FrameBinView> =
            (lo..=hi).map(|k| self.views.as_ref().unwrap()[k].clone()).collect();

        if do_blur_tof {
            let sigma_bins = self.maybe_sigma_tof_bins.unwrap();
            let trunc      = self.truncate;
            views_local = blur_tof_all_frames(&views_local, sigma_bins, trunc);
        }

        let mut raw = vec![0.0f32; rows * cols];
        for v in &views_local {
            for b_i in 0..v.unique_bins.len() {
                let row   = v.unique_bins[b_i];
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
            smooth_rows_parallel(
                &raw,
                rows,
                cols,
                self.maybe_sigma_scans.unwrap(),
                self.truncate,
            )
        } else {
            raw.clone()
        };

        let frame_id_bounds = (self.frame_ids_sorted[lo], self.frame_ids_sorted[hi]);

        TofScanWindowGrid {
            scale: self.scale.clone(),
            rt_range_frames: (lo, hi),
            rt_range_sec: (self.frame_times[lo], self.frame_times[hi]),
            frame_id_bounds,
            window_group: None,
            scans: (0..cols).collect(),
            data: Some(data),
            rows,
            cols,
            data_raw: if do_smooth || do_blur_tof { Some(raw) } else { None },
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyFit1D {
    pub inner: Fit1D,
}

#[pymethods]
impl PyFit1D {
    #[new]
    fn new() -> PyResult<Self> {
        Err(exceptions::PyRuntimeError::new_err(
            "PyFit1D objects are created internally, not directly from Python.",
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "Fit1D(mu={:.4}, sigma={:.4}, height={:.4}, area={:.4}, n={:?})",
            self.inner.mu,
            self.inner.sigma,
            self.inner.height,
            self.inner.area,
            self.inner.n,
        )
    }

    #[getter]
    fn mu(&self) -> f32 {
        self.inner.mu
    }

    #[getter]
    fn sigma(&self) -> f32 {
        self.inner.sigma
    }

    #[getter]
    fn height(&self) -> f32 {
        self.inner.height
    }

    #[getter]
    fn area(&self) -> f32 {
        self.inner.area
    }

    /// Optional sample count; may be None.
    #[getter]
    fn n(&self) -> Option<usize> {
        Some(self.inner.n)
    }
}

#[pyclass]
pub struct PyRtPeak1D { pub inner: RtPeak1D }

#[pymethods]
impl PyRtPeak1D {
    #[new]
    #[pyo3(signature = (
        rt_idx,
        rt_sec,
        apex_smoothed,
        apex_raw,
        prominence,
        left_x,
        right_x,
        width_frames,
        area_raw,
        subframe,
        rt_bounds_frames,
        frame_id_bounds,
        window_group,
        tof_row,
        tof_center,
        tof_bounds,
        parent_im_id,
        id
    ))]
    pub fn new(
        rt_idx: usize,
        rt_sec: Option<f32>,
        apex_smoothed: f32,
        apex_raw: f32,
        prominence: f32,
        left_x: f32,
        right_x: f32,
        width_frames: usize,
        area_raw: f32,
        subframe: f32,
        rt_bounds_frames: (usize, usize),
        frame_id_bounds: (u32, u32),
        window_group: Option<u32>,
        tof_row: usize,
        tof_center: i32,
        tof_bounds: (i32, i32),
        parent_im_id: Option<i64>,
        id: i64,
    ) -> Self {
        let inner = RtPeak1D {
            rt_idx,
            rt_sec,
            apex_smoothed,
            apex_raw,
            prominence,
            left_x,
            right_x,
            width_frames,
            area_raw,
            subframe,
            rt_bounds_frames,
            frame_id_bounds,
            window_group,
            tof_row,
            tof_center,
            tof_bounds,
            parent_im_id,
            id,
        };
        Self { inner }
    }

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

    // --- TOF context
    #[getter] fn tof_row(&self) -> usize { self.inner.tof_row }
    #[getter] fn tof_center(&self) -> i32 { self.inner.tof_center }
    #[getter] fn tof_bounds(&self) -> (i32, i32) { self.inner.tof_bounds }

    // --- linkage
    #[getter] fn parent_im_id(&self) -> Option<i64> { self.inner.parent_im_id }
    #[getter] fn id(&self) -> i64 { self.inner.id }

    #[staticmethod]
    #[pyo3(signature = (
        grid,
        mu_rt,
        sigma_rt,
        mu_tof,
        sigma_tof,
        amplitude,
        baseline,
        area,
        tof_row,
        rt_idx_raw,
        k_sigma,
        min_width
    ))]
    pub fn from_batch_detected<'py>(
        py: Python<'py>,
        grid: &PyTofRtGrid,
        mu_rt: PyReadonlyArray1<'py, f32>,
        sigma_rt: PyReadonlyArray1<'py, f32>,
        mu_tof: PyReadonlyArray1<'py, f32>,
        sigma_tof: PyReadonlyArray1<'py, f32>,
        amplitude: PyReadonlyArray1<'py, f32>,
        baseline: PyReadonlyArray1<'py, f32>,
        area: PyReadonlyArray1<'py, f32>,
        tof_row: PyReadonlyArray1<'py, f32>,
        rt_idx_raw: PyReadonlyArray1<'py, f32>,
        k_sigma: f32,
        min_width: usize,
    ) -> PyResult<Vec<Py<PyRtPeak1D>>> {
        // ---- Borrow NumPy arrays as slices (no unsafe needed) ------------
        let mu_rt       = mu_rt.as_slice()?;
        let sigma_rt    = sigma_rt.as_slice()?;
        let mu_tof      = mu_tof.as_slice()?;
        let sigma_tof   = sigma_tof.as_slice()?;
        let amplitude   = amplitude.as_slice()?;
        let baseline    = baseline.as_slice()?;
        let area        = area.as_slice()?;
        let tof_row     = tof_row.as_slice()?;
        let rt_idx_raw  = rt_idx_raw.as_slice()?;

        let n = mu_rt.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // sanity check lengths
        let check_len = |name: &str, len: usize| -> PyResult<()> {
            if len != n {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "array {name} has len={len}, expected {n}"
                )))
            } else {
                Ok(())
            }
        };

        check_len("sigma_rt", sigma_rt.len())?;
        check_len("mu_tof", mu_tof.len())?;
        check_len("sigma_tof", sigma_tof.len())?;
        check_len("amplitude", amplitude.len())?;
        check_len("baseline", baseline.len())?;
        check_len("area", area.len())?;
        check_len("tof_row", tof_row.len())?;
        check_len("rt_idx_raw", rt_idx_raw.len())?;

        // ---- Snapshot grid info so parallel region is GIL-free -----------
        let rows = grid.inner.rows;
        let cols = grid.inner.cols;
        if rows == 0 || cols == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "TofRtGrid has zero size",
            ));
        }

        let frame_ids = grid.inner.frame_ids.clone();
        let frame_id_bounds_default = grid.inner.frame_id_bounds;
        let rt_times = grid.inner.rt_times.clone();
        let rt_range_sec = grid.inner.rt_range_sec;
        let window_group = grid.inner.window_group;

        let k_sigma = k_sigma.max(0.0);
        let min_width = min_width.max(1);

        // ---- Heavy work in parallel, GIL released ------------------------
        let peaks: Vec<RtPeak1D> = py.allow_threads(|| {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let mu_rt_i      = mu_rt[i];
                    let sigma_rt_i   = sigma_rt[i].max(1e-3);
                    let mu_tof_i     = mu_tof[i];
                    let sigma_tof_i  = sigma_tof[i].max(1e-3);
                    let amp_i        = amplitude[i];
                    let base_i       = baseline[i];
                    let area_i       = area[i];
                    let tof_row_i    = tof_row[i];
                    let _rt_idx_raw_i = rt_idx_raw[i];

                    // ----------------- RT geometry ------------------------
                    let mut rt_idx = mu_rt_i.round() as isize;
                    if rt_idx < 0 {
                        rt_idx = 0;
                    }
                    if rt_idx as usize >= cols {
                        rt_idx = (cols - 1) as isize;
                    }
                    let rt_idx_u = rt_idx as usize;

                    let subframe = mu_rt_i - (rt_idx_u as f32);

                    // rt_sec
                    let rt_sec = if !rt_times.is_empty() {
                        rt_times[rt_idx_u]
                    } else {
                        let (t0, t1) = rt_range_sec;
                        if cols > 1 {
                            let frac = (rt_idx_u as f32) / ((cols - 1) as f32);
                            t0 + frac * (t1 - t0)
                        } else {
                            t0
                        }
                    };

                    // bounds in frame index space
                    let half_from_sigma = (k_sigma * sigma_rt_i).ceil() as isize;
                    let half_from_min = ((min_width.saturating_sub(1)) / 2) as isize;
                    let half = half_from_sigma.max(half_from_min).max(0);

                    let mut rt_lo = rt_idx - half;
                    let mut rt_hi = rt_idx + half;
                    if rt_lo < 0 {
                        rt_lo = 0;
                    }
                    if rt_hi as usize >= cols {
                        rt_hi = (cols - 1) as isize;
                    }
                    let rt_lo_u = rt_lo as usize;
                    let rt_hi_u = rt_hi as usize;
                    let width_frames = rt_hi_u.saturating_sub(rt_lo_u).saturating_add(1);

                    // frame id bounds
                    let frame_id_bounds = if !frame_ids.is_empty() {
                        (frame_ids[rt_lo_u], frame_ids[rt_hi_u])
                    } else {
                        frame_id_bounds_default
                    };

                    let rt_bounds_frames = (rt_lo_u, rt_hi_u);
                    let left_x = rt_lo_u as f32;
                    let right_x = rt_hi_u as f32;

                    // ----------------- TOF geometry -----------------------
                    let mut tof_row_idx = tof_row_i.round() as isize;
                    if tof_row_idx < 0 {
                        tof_row_idx = 0;
                    }
                    if tof_row_idx as usize >= rows {
                        tof_row_idx = (rows - 1) as isize;
                    }
                    let tof_row_u = tof_row_idx as usize;

                    let mut tof_center_idx = mu_tof_i.round() as isize;
                    if tof_center_idx < 0 {
                        tof_center_idx = 0;
                    }
                    if tof_center_idx as usize >= rows {
                        tof_center_idx = (rows - 1) as isize;
                    }
                    let tof_center_u = tof_center_idx as usize;

                    let half_tof = (k_sigma * sigma_tof_i).ceil().max(1.0) as isize;
                    let mut tof_lo_idx = tof_center_idx - half_tof;
                    let mut tof_hi_idx = tof_center_idx + half_tof;
                    if tof_lo_idx < 0 {
                        tof_lo_idx = 0;
                    }
                    if tof_hi_idx as usize >= rows {
                        tof_hi_idx = (rows - 1) as isize;
                    }

                    let tof_bounds = (tof_lo_idx as i32, tof_hi_idx as i32);
                    let tof_center = tof_center_u as i32;

                    // ----------------- shape / intensity -------------------
                    let apex_raw = amp_i + base_i;
                    let apex_smoothed = apex_raw;
                    let prominence = amp_i;
                    let area_raw = area_i;

                    RtPeak1D {
                        rt_idx: rt_idx_u,
                        rt_sec: Some(rt_sec),
                        apex_smoothed,
                        apex_raw,
                        prominence,
                        left_x,
                        right_x,
                        width_frames,
                        area_raw,
                        subframe,
                        rt_bounds_frames,
                        frame_id_bounds,
                        window_group,
                        tof_row: tof_row_u,
                        tof_center,
                        tof_bounds,
                        parent_im_id: None,
                        id: i as i64,
                    }
                })
                .collect()
        });

        // ---- Wrap into PyRtPeak1D under the GIL ---------------------------
        let mut out = Vec::with_capacity(peaks.len());
        for p in peaks {
            out.push(Py::new(py, PyRtPeak1D { inner: p })?);
        }
        Ok(out)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "RtPeak1D(rt_idx={}, rt_sec={:?}, apex={:.3}, prom={:.3}, frames={:?}, tof_row={})",
            self.inner.rt_idx, self.inner.rt_sec, self.inner.apex_smoothed,
            self.inner.prominence, self.inner.rt_bounds_frames, self.inner.tof_row
        )
    }
}

#[pyclass]
pub struct PyImPeak1D { pub inner: Arc<ImPeak1D> }

#[pymethods]
impl PyImPeak1D {
    #[new]
    #[pyo3(signature = (
        tof_row,
        tof_center,
        tof_bounds,
        rt_bounds,
        frame_id_bounds,
        window_group,
        scan,
        left,
        right,
        scan_abs,
        left_abs,
        right_abs,
        scan_sigma,
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
        tof_row: usize,
        tof_center: i32,
        tof_bounds: (i32, i32),
        rt_bounds: (usize, usize),
        frame_id_bounds: (u32, u32),
        window_group: Option<u32>,

        scan: usize,
        left: usize,
        right: usize,

        scan_abs: usize,
        left_abs: usize,
        right_abs: usize,

        scan_sigma: Option<f32>,
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
            tof_row,
            tof_center,
            tof_bounds,
            rt_bounds,
            frame_id_bounds,
            window_group,

            scan,
            left,
            right,

            scan_abs,
            left_abs,
            right_abs,

            scan_sigma,
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

    #[getter] fn tof_row(&self) -> usize { self.inner.tof_row }
    #[getter] fn tof_center(&self) -> i32 { self.inner.tof_center }
    #[getter] fn tof_bounds(&self) -> (i32, i32) { self.inner.tof_bounds }

    /// RT column bounds (inclusive) in the source RT grid
    #[getter] fn rt_bounds(&self) -> (usize, usize) { self.inner.rt_bounds }

    /// Materialized frame-id bounds (inclusive)
    #[getter] fn frame_id_bounds(&self) -> (u32, u32) { self.inner.frame_id_bounds }

    /// Optional DIA window group id (if applicable)
    #[getter] fn window_group(&self) -> Option<u32> { self.inner.window_group }

    #[getter] fn scan(&self) -> usize { self.inner.scan }
    #[getter] fn scan_sigma(&self) -> Option<f32> { self.inner.scan_sigma }
    #[getter] fn mobility(&self) -> Option<f32> { self.inner.mobility }
    #[getter] fn apex_smoothed(&self) -> f32 { self.inner.apex_smoothed }
    #[getter] fn apex_raw(&self) -> f32 { self.inner.apex_raw }
    #[getter] fn prominence(&self) -> f32 { self.inner.prominence }
    #[getter] fn left(&self) -> usize { self.inner.left }
    #[getter] fn right(&self) -> usize { self.inner.right }
    #[getter] fn left_x(&self) -> f32 { self.inner.left_x }
    #[getter] fn right_x(&self) -> f32 { self.inner.right_x }
    #[getter] fn width_scans(&self) -> usize { self.inner.width_scans }
    #[getter] fn area_raw(&self) -> f32 { self.inner.area_raw }
    #[getter] fn subscan(&self) -> f32 { self.inner.subscan }
    #[getter] fn scan_abs(&self) -> usize { self.inner.scan_abs }
    #[getter] fn left_abs(&self) -> usize { self.inner.left_abs }
    #[getter] fn right_abs(&self) -> usize { self.inner.right_abs }

    pub fn __repr__(&self) -> String {
        let p = &self.inner;
        format!(
            "ImPeak1D(tof_row={}, rt_bounds={:?}, scan_abs={}, width_scans={}, apex={:.3}, prom={:.3})",
            p.tof_row, p.rt_bounds, p.scan_abs, p.width_scans, p.apex_smoothed, p.prominence
        )
    }

    /// Vectorized construction from detector outputs (dict-of-arrays).
    ///
    /// All arrays must have the same length N.
    /// Geometry from the window grid is passed in once.
    #[staticmethod]
    #[pyo3(signature = (
        mu_scan,
        sigma_scan,
        amplitude,
        baseline,
        area,
        tof_row,
        scan_idx,
        rows,
        scans_global,
        tof_centers,
        tof_edges,
        rt_bounds,
        frame_id_bounds,
        window_group,
        k_sigma,
        min_width,
    ))]
    pub fn batch_from_detected(
        mu_scan: PyReadonlyArray1<f32>,
        sigma_scan: PyReadonlyArray1<f32>,
        amplitude: PyReadonlyArray1<f32>,
        baseline: PyReadonlyArray1<f32>, // currently unused, kept for future
        area: PyReadonlyArray1<f32>,
        tof_row: PyReadonlyArray1<f32>,
        scan_idx: PyReadonlyArray1<f32>,

        rows: usize,
        scans_global: PyReadonlyArray1<u32>,
        tof_centers: PyReadonlyArray1<f32>,
        tof_edges: PyReadonlyArray1<f32>,
        rt_bounds: (usize, usize),
        frame_id_bounds: (u32, u32),
        window_group: Option<u32>,

        k_sigma: f32,
        min_width: usize,
    ) -> PyResult<Vec<PyImPeak1D>> {
        let mu_scan = mu_scan.as_slice()?;
        let sigma_scan = sigma_scan.as_slice()?;
        let amplitude = amplitude.as_slice()?;
        let _baseline = baseline.as_slice()?; // not used yet
        let area = area.as_slice()?;
        let tof_row_arr = tof_row.as_slice()?;
        let scan_idx_arr = scan_idx.as_slice()?;

        let scans_global = scans_global.as_slice()?;
        let tof_centers = tof_centers.as_slice()?;
        let tof_edges = tof_edges.as_slice()?;

        let n = mu_scan.len();
        if sigma_scan.len() != n
            || amplitude.len() != n
            || area.len() != n
            || tof_row_arr.len() != n
            || scan_idx_arr.len() != n
        {
            return Err(PyValueError::new_err(
                "All input arrays must have the same length",
            ));
        }

        let cols = scans_global.len();
        if tof_centers.len() < rows || tof_edges.len() < rows {
            return Err(PyValueError::new_err(
                "tof_centers/tof_edges shorter than rows",
            ));
        }

        let wg_val: u32 = window_group.unwrap_or(0);
        let fid_lo: u32 = frame_id_bounds.0;

        let mut out = Vec::with_capacity(n);

        for i in 0..n {
            let mu_scan_v = mu_scan[i];
            let sigma_scan_v = sigma_scan[i].max(1e-3); // guard
            let amp_v = amplitude[i];
            let area_v = area[i];

            // indices in original grid orientation
            let tof_row_f = tof_row_arr[i];
            let scan_idx_f = scan_idx_arr[i];

            let tof_row_i = tof_row_f.round() as usize;
            let scan_idx_i = scan_idx_f.round() as usize;

            if tof_row_i >= rows {
                return Err(PyValueError::new_err(format!(
                    "tof_row={} outside grid rows={}",
                    tof_row_i, rows
                )));
            }
            if scan_idx_i >= cols {
                return Err(PyValueError::new_err(format!(
                    "scan_idx={} outside grid cols={}",
                    scan_idx_i, cols
                )));
            }

            // tof geometry
            let tof_center_f = tof_centers[tof_row_i];
            let tof_center = tof_center_f.round() as i32;

            let (tof_min, tof_max) = if tof_row_i + 1 < tof_edges.len() {
                let lo = tof_edges[tof_row_i];
                let hi = tof_edges[tof_row_i + 1];
                (lo.round() as i32, hi.round() as i32)
            } else {
                let e_last = tof_edges[tof_row_i];
                let width = if tof_row_i > 0 {
                    e_last - tof_edges[tof_row_i - 1]
                } else {
                    1.0
                };
                let lo = e_last - width;
                (lo.round() as i32, e_last.round() as i32)
            };

            let tof_bounds = (tof_min, tof_max);

            // absolute scan ids
            let scan_abs_u32 = scans_global[scan_idx_i];
            let scan_abs = scan_abs_u32 as usize;

            // left/right bounds from sigma_scan
            let half = f32::max(min_width as f32 / 2.0, k_sigma * sigma_scan_v);
            let left = scan_idx_i.saturating_sub(half.floor() as usize);
            let right = std::cmp::min(cols.saturating_sub(1), scan_idx_i + half.ceil() as usize);
            let width_scans = right.saturating_sub(left).saturating_add(1);

            let left_abs = scans_global[left] as usize;
            let right_abs = scans_global[right] as usize;

            // scan_sigma as Option<f32>
            let scan_sigma = if sigma_scan_v.is_finite() {
                Some(sigma_scan_v)
            } else {
                None
            };

            let mobility = None;
            let apex_smoothed = amp_v;
            let apex_raw = amp_v;
            let prominence = amp_v;

            let left_x = left_abs as f32;
            let right_x = right_abs as f32;
            let subscan = mu_scan_v;

            // peak id (same bit packing as Python)
            let peak_id: i64 = (((wg_val & 0xFF) as i64) << 56)
                | (((fid_lo & 0xFFFF) as i64) << 40)
                | (((tof_row_i as u32 & 0xFFFF) as i64) << 24)
                | (((scan_abs as u32 & 0xFFFFFF) as i64));

            let inner = ImPeak1D {
                tof_row: tof_row_i,
                tof_center,
                tof_bounds,
                rt_bounds,
                frame_id_bounds,
                window_group,

                scan: scan_idx_i,
                left,
                right,

                scan_abs,
                left_abs,
                right_abs,

                scan_sigma,
                mobility,
                apex_smoothed,
                apex_raw,
                prominence,
                left_x,
                right_x,
                width_scans,
                area_raw: area_v,
                subscan,
                id: peak_id,
            };

            out.push(PyImPeak1D {
                inner: Arc::new(inner),
            });
        }

        Ok(out)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTofScanWindowGrid {
    pub inner: TofScanWindowGrid,
}

#[pymethods]
impl PyTofScanWindowGrid {
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
    fn data<'py>(&mut self, py: Python<'py>) -> PyResult<Py<PyArray2<f32>>> {
        let v = self.inner.data
            .take()
            .ok_or_else(|| exceptions::PyRuntimeError::new_err("data already moved"))?;
        let arr = PyArray2::from_owned_array_bound(
            py,
            Array2::from_shape_vec((self.inner.rows, self.inner.cols).f(), v)
                .map_err(|e| exceptions::PyValueError::new_err(format!("shape error: {e}")))?,
        );
        Ok(arr.unbind())
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
            "TofScanWindowGrid(frames=({l},{r}), rt=({:.3},{:.3})s, shape=({}, {}))",
            tl, tr, self.inner.rows, self.inner.cols
        )
    }

    #[getter]
    fn tof_centers<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec_bound(py, self.inner.scale.centers.clone()).unbind())
    }

    #[getter]
    fn tof_edges<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec_bound(py, self.inner.scale.edges.clone()).unbind())
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
        tof_step,
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
        // BuildSpecOpts (TOF-based now)
        extra_rt_pad=0,
        extra_im_pad=0,
        tof_bin_pad=0,
        tof_hist_bins=64,
        // Eval1DOpts
        refine_tof_once=true,
        refine_k_sigma=3.0,
        attach_axes=true,
        attach_points=false,
        attach_max_points=None,
        attach_im_trace=false,
        attach_rt_trace=false,
        // matching constraint + threads
        require_rt_overlap=true,
        compute_mz_from_tof=true,
        pad_rt_frames=0,
        pad_im_scans=0,
        pad_tof_bins=0,
        num_threads=0,
        min_im_span=12,
        rt_pad_frames=5,
        // NEW: distance-based merge of duplicates
        merge_duplicates=false,
        max_rt_center_delta=0.0,
        max_im_center_delta=0.0,
        max_tof_center_delta=0.0,
    ))]
        pub fn clusters_for_group(
            &self,
            py: Python<'_>,
            window_group: u32,
            tof_step: i32,
            im_peaks: Vec<Py<PyImPeak1D>>,
            // RtExpandParams
            bin_pad: usize,
            smooth_sigma_sec: f32,
            smooth_trunc_k: f32,
            min_prom: f32,
            min_sep_sec: f32,
            min_width_sec: f32,
            fallback_if_frames_lt: usize,
            fallback_frac_width: f32,
            // BuildSpecOpts
            extra_rt_pad: usize,
            extra_im_pad: usize,
            tof_bin_pad: usize,
            tof_hist_bins: usize,
            // Eval1DOpts
            refine_tof_once: bool,
            refine_k_sigma: f32,
            attach_axes: bool,
            attach_points: bool,
            attach_max_points: Option<usize>,
            attach_im_trace: bool,
            attach_rt_trace: bool,
            // matching + threads
            require_rt_overlap: bool,
            compute_mz_from_tof: bool,
            pad_rt_frames: usize,
            pad_im_scans: usize,
            pad_tof_bins: usize,
            num_threads: usize,
            min_im_span: usize,
            rt_pad_frames: usize,
            // distance-based merge
            merge_duplicates: bool,
            max_rt_center_delta: f32,
            max_im_center_delta: f32,
            max_tof_center_delta: f32,
        ) -> PyResult<Vec<Py<PyClusterResult1D>>> {
            // ---- convert Python IM peaks into Rust ImPeak1D ------------------
            let im_rs: Vec<ImPeak1D> = im_peaks
                .iter()
                .map(|p| p.borrow(py).inner.as_ref().clone())
                .collect();
            debug_assert!(
                im_rs.iter().all(|p| p.window_group == Some(window_group)),
                "clusters_for_group: some IM peaks have wrong or missing window_group"
            );

            // ---- RT expansion params -----------------------------------------
            let rt_params = RtExpandParams {
                bin_pad,
                smooth_sigma_sec,
                smooth_trunc_k,
                min_prom,
                min_sep_sec,
                min_width_sec,
                fallback_if_frames_lt,
                fallback_frac_width,
                rt_pad_frames,
            };

            // ---- BuildSpecOpts: TOF-based, no ppm ----------------------------
            let build_opts = BuildSpecOpts {
                extra_rt_pad,
                extra_im_pad,
                tof_bin_pad,
                tof_hist_bins,
                ms_level: 2,         // DIA fragments
                min_im_span,
                im_k_sigma: 3.0,     // keep hard-coded for now
            };

            // ---- Eval1DOpts: TOF refine, no mz_ppm_cap here ------------------
            let eval_opts = Eval1DOpts {
                refine_tof_once,
                refine_k_sigma,
                attach_axes,
                attach: Attach1DOptions {
                    attach_points,
                    attach_axes,
                    max_points: attach_max_points,
                },
                compute_mz_from_tof,
                attach_im_trace,
                attach_rt_trace,
                pad_rt_frames,
                pad_im_scans,
                pad_tof_bins,
            };

            // ---- Run the core DIA clustering --------------------------------
            let mut results = py.allow_threads(|| {
                self.inner.clusters_for_group(
                    window_group,
                    tof_step,
                    &im_rs,
                    rt_params,
                    &build_opts,
                    &eval_opts,
                    require_rt_overlap,
                    num_threads,
                )
            });

            // ---- Optional distance-based de-duplication ----------------------
            if merge_duplicates {
                let dist = ClusterMergeDistancePolicy {
                    max_rt_center_delta,
                    max_im_center_delta,
                    max_tof_center_delta,
                    max_mz_center_delta_da: 0.0, // can be exposed later if useful
                };
                results = merge_clusters_by_distance(results, &dist);
            }

            results_to_py(py, results)
        }

        #[pyo3(signature = (
        tof_step,
        im_peaks,
        bin_pad=0,
        smooth_sigma_sec=1.25, smooth_trunc_k=3.0,
        min_prom=50.0, min_sep_sec=2.0, min_width_sec=2.0,
        fallback_if_frames_lt=5, fallback_frac_width=0.5,
        extra_rt_pad=0, extra_im_pad=0, tof_bin_pad=0, tof_hist_bins=64,
        refine_tof_once=true,
        refine_k_sigma=3.0,
        attach_axes=true,
        attach_points=false,
        attach_max_points=None,
        attach_im_trace=false,
        attach_rt_trace=false,
        require_rt_overlap=true,
        compute_mz_from_tof=true,
        pad_rt_frames=0,
        pad_im_scans=0,
        pad_tof_bins=0,
        num_threads=0,
        min_im_span=12,
        rt_pad_frames=5,
        // NEW: distance-based merge of duplicates
        merge_duplicates=false,
        max_rt_center_delta=0.0,
        max_im_center_delta=0.0,
        max_tof_center_delta=0.0,
    ))]
        pub fn clusters_for_precursor(
            &self,
            py: Python<'_>,
            tof_step: i32,
            im_peaks: Vec<Py<PyImPeak1D>>,
            // RtExpandParams
            bin_pad: usize,
            smooth_sigma_sec: f32,
            smooth_trunc_k: f32,
            min_prom: f32,
            min_sep_sec: f32,
            min_width_sec: f32,
            fallback_if_frames_lt: usize,
            fallback_frac_width: f32,
            // BuildSpecOpts
            extra_rt_pad: usize,
            extra_im_pad: usize,
            tof_bin_pad: usize,
            tof_hist_bins: usize,
            // Eval1DOpts
            refine_tof_once: bool,
            refine_k_sigma: f32,
            attach_axes: bool,
            attach_points: bool,
            attach_max_points: Option<usize>,
            attach_im_trace: bool,
            attach_rt_trace: bool,
            // matching + threads
            require_rt_overlap: bool,
            compute_mz_from_tof: bool,
            pad_rt_frames: usize,
            pad_im_scans: usize,
            pad_tof_bins: usize,
            num_threads: usize,
            min_im_span: usize,
            rt_pad_frames: usize,
            // distance-based merge
            merge_duplicates: bool,
            max_rt_center_delta: f32,
            max_im_center_delta: f32,
            max_tof_center_delta: f32,
        ) -> PyResult<Vec<Py<PyClusterResult1D>>> {
            // ---- convert Python IM peaks into Rust ImPeak1D ------------------
            let im_rs: Vec<ImPeak1D> = im_peaks
                .iter()
                .map(|p| p.borrow(py).inner.as_ref().clone())
                .collect();
            debug_assert!(
                im_rs.iter().all(|p| p.window_group.is_none()),
                "clusters_for_precursor: IM peaks unexpectedly carry a window_group"
            );

            // ---- RT expansion params -----------------------------------------
            let rt_params = RtExpandParams {
                bin_pad,
                smooth_sigma_sec,
                smooth_trunc_k,
                min_prom,
                min_sep_sec,
                min_width_sec,
                fallback_if_frames_lt,
                fallback_frac_width,
                rt_pad_frames,
            };

            // ---- BuildSpecOpts: MS1, TOF-based -------------------------------
            let build_opts = BuildSpecOpts {
                extra_rt_pad,
                extra_im_pad,
                tof_bin_pad,
                tof_hist_bins,
                ms_level: 1,         // precursor
                min_im_span,
                im_k_sigma: 3.0,
            };

            // ---- Eval1DOpts --------------------------------------------------
            let eval_opts = Eval1DOpts {
                refine_tof_once,
                refine_k_sigma,
                attach_axes,
                attach: Attach1DOptions {
                    attach_points,
                    attach_axes,
                    max_points: attach_max_points,
                },
                compute_mz_from_tof,
                attach_im_trace,
                attach_rt_trace,
                pad_rt_frames,
                pad_im_scans,
                pad_tof_bins,
            };

            // ---- Run the core MS1 clustering --------------------------------
            let mut results = py.allow_threads(|| {
                self.inner.clusters_for_precursor(
                    tof_step,
                    &im_rs,
                    rt_params,
                    &build_opts,
                    &eval_opts,
                    require_rt_overlap,
                    num_threads,
                )
            });

            // ---- Optional distance-based de-duplication ----------------------
            if merge_duplicates {
                let dist = ClusterMergeDistancePolicy {
                    max_rt_center_delta,
                    max_im_center_delta,
                    max_tof_center_delta,
                    max_mz_center_delta_da: 0.0, // can be exposed later if useful
                };
                results = merge_clusters_by_distance(results, &dist);
            }

            results_to_py(py, results)
        }

    #[pyo3(signature = (
        ms1_clusters,
        ms2_clusters,
        features=None,

        // ---- CandidateOpts (with defaults matching `CandidateOpts::default()`) ----
        min_rt_jaccard = 0.0,
        ms2_rt_guard_sec = 0.0,
        rt_bucket_width = 1.0,
        max_ms1_rt_span_sec = Some(60.0),
        max_ms2_rt_span_sec = Some(60.0),
        min_raw_sum = 1.0,
        max_rt_apex_delta_sec = Some(2.0),
        max_scan_apex_delta = Some(6),
        min_im_overlap_scans = 1,

        // ---- ScoreOpts (defaults from `ScoreOpts::default()`) ----
        w_jacc_rt = 1.0,
        w_shape = 1.0,
        w_rt_apex = 0.75,
        w_im_apex = 0.75,
        w_im_overlap = 0.5,
        w_ms1_intensity = 0.25,
        rt_apex_scale_s = 0.75,
        im_apex_scale_scans = 3.0,
        shape_neutral = 0.6,
        min_sigma_rt = 0.05,
        min_sigma_im = 0.5,
        w_shape_rt_inner = 1.0,
        w_shape_im_inner = 1.0,

        // ---- PseudoSpecOpts ----
        top_n_fragments = 500,
    ))]
    pub fn build_pseudo_spectra_from_clusters_geom(
        &self,
        py: Python<'_>,
        ms1_clusters: Vec<Py<PyClusterResult1D>>,
        ms2_clusters: Vec<Py<PyClusterResult1D>>,
        features: Option<Vec<Py<PySimpleFeature>>>,

        // CandidateOpts
        min_rt_jaccard: f32,
        ms2_rt_guard_sec: f64,
        rt_bucket_width: f64,
        max_ms1_rt_span_sec: Option<f64>,
        max_ms2_rt_span_sec: Option<f64>,
        min_raw_sum: f32,
        max_rt_apex_delta_sec: Option<f32>,
        max_scan_apex_delta: Option<usize>,
        min_im_overlap_scans: usize,

        // ScoreOpts
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

        // PseudoSpecOpts
        top_n_fragments: usize,
    ) -> PyResult<PyPseudoBuildResult> {
        // Unwrap clusters
        let rust_ms1: Vec<ClusterResult1D> = ms1_clusters
            .into_iter()
            .map(|c| c.borrow(py).inner.clone())
            .collect();

        let rust_ms2: Vec<ClusterResult1D> = ms2_clusters
            .into_iter()
            .map(|c| c.borrow(py).inner.clone())
            .collect();

        // Unwrap features (optional)
        let rust_feats: Option<Vec<SimpleFeature>> = features.map(|vec_f| {
            vec_f
                .into_iter()
                .map(|pf| pf.borrow(py).inner.clone())
                .collect()
        });

        // Build CandidateOpts from primitives
        let cand_opts = CandidateOpts {
            min_rt_jaccard,
            ms2_rt_guard_sec,
            rt_bucket_width,
            max_ms1_rt_span_sec,
            max_ms2_rt_span_sec,
            min_raw_sum,
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            min_im_overlap_scans,
            reject_frag_inside_precursor_tile: true,
        };

        // Build ScoreOpts from primitives
        let score_opts = ScoreOpts {
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

        // PseudoSpecOpts (top_n_fragments is the one knob we expose now)
        let pseudo_opts = PseudoSpecOpts {
            top_n_fragments,
            ..PseudoSpecOpts::default()
        };

        // Call the dataset method
        let res = self.inner.build_pseudo_spectra_from_clusters_geom(
            &rust_ms1,
            &rust_ms2,
            rust_feats.as_deref(),
            &cand_opts,
            &score_opts,
            &pseudo_opts,
        );

        Ok(PyPseudoBuildResult { inner: res })
    }

    #[pyo3(signature = (
        ms1_clusters,
        ms2_clusters,
        features=None,

        // ---- CandidateOpts (with defaults matching `CandidateOpts::default()`) ----
        min_rt_jaccard = 0.0,
        ms2_rt_guard_sec = 0.0,
        rt_bucket_width = 1.0,
        max_ms1_rt_span_sec = Some(60.0),
        max_ms2_rt_span_sec = Some(60.0),
        min_raw_sum = 1.0,
        max_rt_apex_delta_sec = Some(2.0),
        max_scan_apex_delta = Some(6),
        min_im_overlap_scans = 1,

        // ---- XicScoreOpts (defaults from `XicScoreOpts::default()`) ----
        w_rt = 0.45,
        w_im = 0.45,
        w_intensity = 0.10,
        intensity_tau = 1.5,
        min_total_score = 0.0,
        use_rt = true,
        use_im = true,
        use_intensity = true,

        // ---- PseudoSpecOpts ----
        top_n_fragments = 500,
    ))]
    pub fn build_pseudo_spectra_from_clusters_xic(
        &self,
        py: Python<'_>,
        ms1_clusters: Vec<Py<PyClusterResult1D>>,
        ms2_clusters: Vec<Py<PyClusterResult1D>>,
        features: Option<Vec<Py<PySimpleFeature>>>,

        // CandidateOpts
        min_rt_jaccard: f32,
        ms2_rt_guard_sec: f64,
        rt_bucket_width: f64,
        max_ms1_rt_span_sec: Option<f64>,
        max_ms2_rt_span_sec: Option<f64>,
        min_raw_sum: f32,
        max_rt_apex_delta_sec: Option<f32>,
        max_scan_apex_delta: Option<usize>,
        min_im_overlap_scans: usize,

        // XicScoreOpts
        w_rt: f32,
        w_im: f32,
        w_intensity: f32,
        intensity_tau: f32,
        min_total_score: f32,
        use_rt: bool,
        use_im: bool,
        use_intensity: bool,

        // PseudoSpecOpts
        top_n_fragments: usize,
    ) -> PyResult<PyPseudoBuildResult> {
        // Unwrap clusters
        let rust_ms1: Vec<ClusterResult1D> = ms1_clusters
            .into_iter()
            .map(|c| c.borrow(py).inner.clone())
            .collect();

        let rust_ms2: Vec<ClusterResult1D> = ms2_clusters
            .into_iter()
            .map(|c| c.borrow(py).inner.clone())
            .collect();

        // Unwrap features (optional)
        let rust_feats: Option<Vec<SimpleFeature>> = features.map(|vec_f| {
            vec_f
                .into_iter()
                .map(|pf| pf.borrow(py).inner.clone())
                .collect()
        });

        // CandidateOpts from primitives
        let cand_opts = CandidateOpts {
            min_rt_jaccard,
            ms2_rt_guard_sec,
            rt_bucket_width,
            max_ms1_rt_span_sec,
            max_ms2_rt_span_sec,
            min_raw_sum,
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            min_im_overlap_scans,
            reject_frag_inside_precursor_tile: true,
        };

        // XicScoreOpts from primitives
        let xic_opts = XicScoreOpts {
            w_rt,
            w_im,
            w_intensity,
            intensity_tau,
            min_total_score,
            use_rt,
            use_im,
            use_intensity,
        };

        // PseudoSpecOpts (top_n_fragments is the one knob we expose now)
        let pseudo_opts = PseudoSpecOpts {
            top_n_fragments,
            ..PseudoSpecOpts::default()
        };

        let res = self.inner.build_pseudo_spectra_from_clusters_xic(
            &rust_ms1,
            &rust_ms2,
            rust_feats.as_deref(),
            &cand_opts,
            &xic_opts,
            &pseudo_opts,
        );

        Ok(PyPseudoBuildResult { inner: res })
    }

    /// Non-competitive "all pairs" builder.
    ///
    /// MS2 clusters may be linked to multiple precursors. Intended for
    /// diagnostics / visualization only. For proper analysis, use
    /// `build_pseudo_spectra_from_clusters` (competitive).
    #[pyo3(signature = (
        ms1_clusters,
        ms2_clusters,
        features=None,
        top_n_fragments=500,
    ))]
    pub fn build_pseudo_spectra_all_pairs_from_clusters(
        &self,
        py: Python<'_>,
        ms1_clusters: Vec<Py<PyClusterResult1D>>,
        ms2_clusters: Vec<Py<PyClusterResult1D>>,
        features: Option<Vec<Py<PySimpleFeature>>>,
        top_n_fragments: usize,
    ) -> PyResult<PyPseudoBuildResult> {
        // Unwrap MS1 / MS2
        let rust_ms1: Vec<ClusterResult1D> = ms1_clusters
            .into_iter()
            .map(|c| c.borrow(py).inner.clone())
            .collect();

        let rust_ms2: Vec<ClusterResult1D> = ms2_clusters
            .into_iter()
            .map(|c| c.borrow(py).inner.clone())
            .collect();

        // Unwrap features
        let rust_feats: Option<Vec<SimpleFeature>> = features.map(|vec_f| {
            vec_f
                .into_iter()
                .map(|pf| pf.borrow(py).inner.clone())
                .collect()
        });

        let pseudo_opts = PseudoSpecOpts {
            top_n_fragments,
            ..PseudoSpecOpts::default()
        };

        // Call dataset-level all-pairs builder
        let result = self.inner.build_pseudo_spectra_all_pairs_from_clusters(
            &rust_ms1,
            &rust_ms2,
            rust_feats.as_deref(),
            &pseudo_opts,
        );

        Ok(PyPseudoBuildResult { inner: result })
    }

    /// Build a dense TOF×RT grid over all PRECURSOR (MS1) frames.
    ///
    /// `tof_step` > 0, where 1 = full TOF resolution, >1 = downsampled.
    #[pyo3(signature = (tof_step = 1))]
    pub fn tof_rt_grid_precursor(&self, tof_step: i32) -> PyTofRtGrid {
        let ds: &TimsDatasetDIA = &self.inner;
        let grid = ds.tof_rt_grid_precursor(tof_step);
        PyTofRtGrid { inner: grid }
    }

    /// Build a dense TOF×RT grid over FRAGMENT (MS2) frames for a DIA window group.
    ///
    /// `tof_step` > 0, where 1 = full TOF resolution, >1 = downsampled.
    #[pyo3(signature = (window_group, tof_step = 1))]
    pub fn tof_rt_grid_for_group(&self, window_group: u32, tof_step: i32) -> PyTofRtGrid {
        let ds: &TimsDatasetDIA = &self.inner;
        let grid = ds.tof_rt_grid_for_group(window_group, tof_step);
        PyTofRtGrid { inner: grid }
    }

    #[pyo3(signature = (prec_mz, im_apex))]
    pub fn window_groups_for_precursor(&self, prec_mz: f32, im_apex: f32) -> Vec<u32> {
        self.inner.window_groups_for_precursor(prec_mz, im_apex)
    }

    #[pyo3(signature = (
        clusters,
        window_group = None,
        tof_step = 1,
        max_points = 1024,
        num_threads = 1,
    ))]
    pub fn debug_extract_raw_for_clusters(
        &self,
        py: Python<'_>,
        clusters: Vec<Py<PyClusterResult1D>>,
        window_group: Option<u32>,
        tof_step: i32,
        max_points: Option<usize>,
        num_threads: usize,
    ) -> Vec<PyRawPoints> {
        // Unwrap clusters
        let rust_clusters: Vec<ClusterResult1D> = clusters
            .iter()
            .map(|c| c.borrow(py).inner.clone())
            .collect();

        let raws = self.inner.debug_extract_raw_for_clusters(
            &rust_clusters,
            window_group,
            tof_step,
            max_points,
            num_threads,
        );

        // Wrap back into PyRawPoints
        raws.into_iter()
            .map(|rp| PyRawPoints { inner: rp })
            .collect()
    }
}

#[pyfunction]
#[pyo3(signature = (
    flat,
    min_overlap_frames=1,
    max_scan_delta=1,
    jaccard_min=0.0,
    max_tof_row_delta=0,
    allow_cross_groups=false,
    // IM-specific:
    min_im_overlap_scans=1,
    im_jaccard_min=0.0,
    require_mutual_apex_inside=true,
))]
pub fn stitch_im_peaks_flat_unordered(
    py: Python<'_>,
    flat: Vec<Py<PyImPeak1D>>,
    min_overlap_frames: usize,
    max_scan_delta: usize,
    jaccard_min: f32,
    max_tof_row_delta: usize,
    allow_cross_groups: bool,
    min_im_overlap_scans: usize,
    im_jaccard_min: f32,
    require_mutual_apex_inside: bool,
) -> PyResult<Vec<Py<PyImPeak1D>>> {
    let params = StitchParams {
        min_overlap_frames,
        max_scan_delta: max_scan_delta.max(1),
        jaccard_min,
        max_tof_row_delta,
        allow_cross_groups,
        min_im_overlap_scans,
        im_jaccard_min,
        require_mutual_apex_inside,
    };

    if flat.is_empty() {
        return Ok(Vec::new());
    }

    // Convert to Vec<Arc<ImPeak1D>>
    let arcs: Vec<Arc<ImPeak1D>> = flat
        .into_iter()
        .map(|p| p.borrow(py).inner.clone())
        .collect();

    let stitched = stitch_im_peaks_flat_unordered_impl(arcs, &params);

    // Wrap back into PyImPeak1D
    let mut out = Vec::with_capacity(stitched.len());
    for v in stitched.into_iter() {
        out.push(Py::new(py, PyImPeak1D { inner: Arc::new(v) })?);
    }
    Ok(out)
}

/// Smooth along the **scan axis** for each TOF row in parallel.
///
/// Layout convention:
///   raw[c * rows + r]  == intensity at (tof_row = r, scan = c)
fn smooth_rows_parallel(
    raw: &[f32],
    rows: usize,
    cols: usize,
    sigma_scans: f32,
    truncate: f32,
) -> Vec<f32> {
    // 1) Build a per-row view: Vec< Vec<f32> >, each row = all scans for that TOF row.
    let mut per_row: Vec<Vec<f32>> = (0..rows)
        .map(|r| {
            let mut row_vec = Vec::with_capacity(cols);
            for c in 0..cols {
                row_vec.push(raw[c * rows + r]);
            }
            row_vec
        })
        .collect();

    // 2) Smooth each row independently, in parallel.
    per_row
        .par_iter_mut()
        .for_each(|row_vec| smooth_vector_gaussian(row_vec, sigma_scans, truncate));

    // 3) Flatten back to the original layout (scan-major, TOF-fast).
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row_vec = &per_row[r];
        for c in 0..cols {
            out[c * rows + r] = row_vec[c];
        }
    }
    out
}

fn results_to_py(py: Python<'_>, v: Vec<ClusterResult1D>) -> PyResult<Vec<Py<PyClusterResult1D>>> {
    v.into_iter()
        .map(|r| Py::new(py, PyClusterResult1D { inner: r }))
        .collect()
}

use rustdf::cluster::io as cio;
use rustdf::cluster::pseudo::{PseudoFragment, PseudoSpecOpts, PseudoSpectrum};
use rustdf::cluster::scoring::{MatchScoreMode, ScoredHit, XicScoreOpts};
use crate::py_feature::PySimpleFeature;
use crate::py_pseudo::PyPseudoSpectrum;

#[pyfunction]
#[pyo3(signature = (path, clusters, compress=true, strip_points=false, strip_axes=false))]
pub fn save_clusters_bin(
    path: &str,
    clusters: Vec<Py<PyClusterResult1D>>,
    compress: bool,
    strip_points: bool,
    strip_axes: bool,
) -> PyResult<()> {
    let mut rust_clusters = Vec::with_capacity(clusters.len());

    Python::with_gil(|py| {
        for c in clusters {
            let inner = &c.borrow(py).inner;

            if strip_points || strip_axes {
                // manually clone only the cheap fields
                let light = ClusterResult1D {
                    cluster_id: inner.cluster_id,
                    rt_window: inner.rt_window,
                    im_window: inner.im_window,
                    tof_window: inner.tof_window,
                    tof_index_window: inner.tof_index_window,
                    mz_window: inner.mz_window,

                    rt_fit: inner.rt_fit.clone(),
                    im_fit: inner.im_fit.clone(),
                    tof_fit: inner.tof_fit.clone(),
                    mz_fit: inner.mz_fit.clone(),

                    raw_sum: inner.raw_sum,
                    volume_proxy: inner.volume_proxy,

                    frame_ids_used: inner.frame_ids_used.clone(),
                    window_group: inner.window_group,
                    parent_im_id: inner.parent_im_id,
                    parent_rt_id: inner.parent_rt_id,
                    ms_level: inner.ms_level,

                    rt_axis_sec: if strip_axes { None } else { inner.rt_axis_sec.clone() },
                    im_axis_scans: if strip_axes { None } else { inner.im_axis_scans.clone() },
                    mz_axis_da: if strip_axes { None } else { inner.mz_axis_da.clone() },

                    raw_points: if strip_points { None } else { inner.raw_points.clone() },

                    rt_trace: inner.rt_trace.clone(),
                    im_trace: inner.im_trace.clone(),
                };
                rust_clusters.push(light);
            } else {
                // keep full cluster, including heavy stuff
                rust_clusters.push(inner.clone());
            }
        }
    });

    cio::save_bincode(path, &rust_clusters, compress)
        .map_err(|e| exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (path, clusters, strip_points=false, strip_axes=false))]
pub fn save_clusters_parquet(
    path: &str,
    clusters: Vec<Py<PyClusterResult1D>>,
    strip_points: bool,
    strip_axes: bool,
) -> PyResult<()> {
    let mut rust_clusters = Vec::with_capacity(clusters.len());

    Python::with_gil(|py| {
        for c in clusters {
            let inner = &c.borrow(py).inner;

            // As with save_clusters_bin: optionally strip heavy fields
            let light = ClusterResult1D {
                cluster_id: inner.cluster_id,
                rt_window: inner.rt_window,
                im_window: inner.im_window,
                tof_window: inner.tof_window,
                tof_index_window: inner.tof_index_window,
                mz_window: inner.mz_window,

                rt_fit: inner.rt_fit.clone(),
                im_fit: inner.im_fit.clone(),
                tof_fit: inner.tof_fit.clone(),
                mz_fit: inner.mz_fit.clone(),

                raw_sum: inner.raw_sum,
                volume_proxy: inner.volume_proxy,

                frame_ids_used: inner.frame_ids_used.clone(),
                window_group: inner.window_group,
                parent_im_id: inner.parent_im_id,
                parent_rt_id: inner.parent_rt_id,
                ms_level: inner.ms_level,

                rt_axis_sec: if strip_axes { None } else { inner.rt_axis_sec.clone() },
                im_axis_scans: if strip_axes { None } else { inner.im_axis_scans.clone() },
                mz_axis_da: if strip_axes { None } else { inner.mz_axis_da.clone() },

                raw_points: if strip_points { None } else { inner.raw_points.clone() },

                rt_trace: inner.rt_trace.clone(),
                im_trace: inner.im_trace.clone(),
            };

            rust_clusters.push(light);
        }
    });

    cio::save_parquet(path, &rust_clusters)
        .map_err(|e| exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
pub fn load_clusters_parquet(
    py: Python<'_>,
    path: &str,
) -> PyResult<Vec<Py<PyClusterResult1D>>> {
    let clusters = cio::load_parquet(path)
        .map_err(|e| exceptions::PyIOError::new_err(e.to_string()))?;

    clusters
        .into_iter()
        .map(|c| Py::new(py, PyClusterResult1D { inner: c }))
        .collect()
}

#[pyfunction]
pub fn load_clusters_bin(py: Python<'_>, path: &str) -> PyResult<Vec<Py<PyClusterResult1D>>> {
    let clusters = cio::load_bincode(path)
        .map_err(|e| exceptions::PyIOError::new_err(e.to_string()))?;
    clusters.into_iter()
        .map(|c| Py::new(py, PyClusterResult1D { inner: c }))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (path, spectra, compress=true))]
pub fn save_pseudo_spectra_bin(
    path: &str,
    spectra: Vec<Py<PyPseudoSpectrum>>,
    compress: bool,
) -> PyResult<()> {
    let mut rust_spectra = Vec::with_capacity(spectra.len());

    Python::with_gil(|py| {
        for s in spectra {
            let inner = &s.borrow(py).inner;
            // No heavy fields to strip, just clone the whole thing
            rust_spectra.push(inner.clone());
        }
    });

    cio::save_pseudo_bincode(path, &rust_spectra, compress)
        .map_err(|e| exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (path))]
pub fn load_pseudo_spectra_bin(
    path: &str,
) -> PyResult<Vec<Py<PyPseudoSpectrum>>> {
    let spectra = cio::load_pseudo_bincode(path)
        .map_err(|e| exceptions::PyIOError::new_err(e.to_string()))?;

    Python::with_gil(|py| {
        let mut out = Vec::with_capacity(spectra.len());
        for spec in spectra {
            let obj = Py::new(py, PyPseudoSpectrum { inner: spec })
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
            out.push(obj);
        }
        Ok(out)
    })
}

#[pyclass]
pub struct PyTofRtGrid {
    pub inner: TofRtGrid,
}

#[pymethods]
impl PyTofRtGrid {
    #[getter]
    pub fn rows(&self) -> usize {
        self.inner.rows
    }

    #[getter]
    pub fn cols(&self) -> usize {
        self.inner.cols
    }

    #[getter]
    pub fn rt_range_frames(&self) -> (usize, usize) {
        self.inner.rt_range_frames
    }

    #[getter]
    pub fn rt_range_sec(&self) -> (f32, f32) {
        self.inner.rt_range_sec
    }

    #[getter]
    pub fn frame_id_bounds(&self) -> (u32, u32) {
        self.inner.frame_id_bounds
    }

    #[getter]
    pub fn window_group(&self) -> Option<u32> {
        self.inner.window_group
    }

    /// Dense TOF×RT matrix as NumPy array (rows = TOF bins, cols = RT frames).
    pub fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f32, Ix2>> {
        let a = Array2::from_shape_vec(
            (self.inner.rows, self.inner.cols),
            self.inner.data.clone(),
        )
            .expect("TofRtGrid: shape mismatch");
        PyArray2::from_array_bound(py, &a)
    }

    /// RT axis (seconds), length = cols.
    pub fn rt_times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f32, Ix1>> {
        PyArray1::from_iter_bound(py, self.inner.rt_times.clone().into_iter())
    }

    /// Frame IDs for each RT column, length = cols.
    pub fn frame_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<u32, Ix1>> {
        PyArray1::from_iter_bound(py, self.inner.frame_ids.clone().into_iter())
    }

    /// TOF centers for each row; convert to m/z on Python side if desired.
    pub fn tof_centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f32, Ix1>> {
        let v = (0..self.inner.rows).map(|r| self.inner.scale.center(r));
        PyArray1::from_iter_bound(py, v)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyAssignmentResult {
    pub inner: AssignmentResult,
}

#[pymethods]
impl PyAssignmentResult {
    #[getter]
    pub fn pairs(&self) -> Vec<(usize, usize)> {
        self.inner.pairs.clone()
    }

    #[getter]
    pub fn ms2_best_ms1(&self) -> Vec<Option<usize>> {
        self.inner.ms2_best_ms1.clone()
    }

    #[getter]
    pub fn ms1_to_ms2(&self) -> Vec<Vec<usize>> {
        self.inner.ms1_to_ms2.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "AssignmentResult(pairs={}, ms2_best_ms1={}, ms1_to_ms2={})",
            self.inner.pairs.len(),
            self.inner.ms2_best_ms1.len(),
            self.inner.ms1_to_ms2.len(),
        )
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyPseudoBuildResult {
    pub inner: PseudoBuildResult,
}

#[pymethods]
impl PyPseudoBuildResult {
    /// Accessor: assignment info (pairs, ms2→best ms1, ms1→ms2 lists).
    #[getter]
    pub fn assignment(&self) -> PyAssignmentResult {
        PyAssignmentResult {
            inner: self.inner.assignment.clone(),
        }
    }

    /// Accessor: pseudo-MS/MS spectra as PyPseudoSpectrum list.
    #[getter]
    pub fn pseudo_spectra(&self) -> Vec<PyPseudoSpectrum> {
        self.inner
            .pseudo_spectra
            .iter()
            .cloned()
            .map(|s| PyPseudoSpectrum { inner: s })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PseudoBuildResult(num_spectra={}, num_pairs={})",
            self.inner.pseudo_spectra.len(),
            self.inner.assignment.pairs.len(),
        )
    }
}

#[pyclass]
pub struct PyScoredHit {
    pub inner: ScoredHit,
}

#[pymethods]
impl PyScoredHit {
    #[getter]
    fn frag_idx(&self) -> usize {
        self.inner.frag_idx
    }

    #[getter]
    fn score(&self) -> f32 {
        self.inner.score
    }

    // Geom fields: return None when geom is not present (e.g. XIC mode)

    #[getter]
    fn jacc_rt(&self) -> Option<f32> {
        self.inner.geom.as_ref().map(|g| g.jacc_rt)
    }

    #[getter]
    fn rt_apex_delta_s(&self) -> Option<f32> {
        self.inner.geom.as_ref().map(|g| g.rt_apex_delta_s)
    }

    #[getter]
    fn im_apex_delta_scans(&self) -> Option<f32> {
        self.inner.geom.as_ref().map(|g| g.im_apex_delta_scans)
    }

    #[getter]
    fn im_overlap_scans(&self) -> Option<u32> {
        self.inner.geom.as_ref().map(|g| g.im_overlap_scans)
    }

    #[getter]
    fn im_union_scans(&self) -> Option<u32> {
        self.inner.geom.as_ref().map(|g| g.im_union_scans)
    }

    #[getter]
    fn ms1_raw_sum(&self) -> Option<f32> {
        self.inner.geom.as_ref().map(|g| g.ms1_raw_sum)
    }

    #[getter]
    fn shape_ok(&self) -> Option<bool> {
        self.inner.geom.as_ref().map(|g| g.shape_ok)
    }

    #[getter]
    fn z_rt(&self) -> Option<f32> {
        self.inner.geom.as_ref().map(|g| g.z_rt)
    }

    #[getter]
    fn z_im(&self) -> Option<f32> {
        self.inner.geom.as_ref().map(|g| g.z_im)
    }

    #[getter]
    fn s_shape(&self) -> Option<f32> {
        self.inner.geom.as_ref().map(|g| g.s_shape)
    }

    #[getter]
    fn xic_s_rt(&self) -> Option<f32> {
        self.inner.xic.as_ref().and_then(|x| x.s_rt)
    }

    #[getter]
    fn xic_s_im(&self) -> Option<f32> {
        self.inner.xic.as_ref().and_then(|x| x.s_im)
    }

    #[getter]
    fn xic_s_intensity(&self) -> Option<f32> {
        self.inner.xic.as_ref().and_then(|x| x.s_intensity)
    }

    #[getter]
    fn xic_r_rt(&self) -> Option<f32> {
        self.inner.xic.as_ref().and_then(|x| x.r_rt)
    }

    #[getter]
    fn xic_r_im(&self) -> Option<f32> {
        self.inner.xic.as_ref().and_then(|x| x.r_im)
    }
}

use std::sync::Arc;
#[pyclass]
#[derive(Clone)]
pub struct PyCandidateOpts {
    pub(crate) inner: CandidateOpts,
}

#[pymethods]
impl PyCandidateOpts {
    /// Create candidate-enumeration options.
    ///
    /// Parameters (Python side):
    /// - min_rt_jaccard: float
    /// - ms2_rt_guard_sec: float
    /// - rt_bucket_width: float
    /// - max_ms1_rt_span_sec: Optional[float]
    /// - max_ms2_rt_span_sec: Optional[float]
    /// - min_raw_sum: float
    /// - max_rt_apex_delta_sec: Optional[float]
    /// - max_scan_apex_delta: Optional[int]
    /// - min_im_overlap_scans: int
    /// - reject_frag_inside_precursor_tile: bool
    #[new]
    #[pyo3(signature = (
        min_rt_jaccard = 0.0,
        ms2_rt_guard_sec = 1.0,
        rt_bucket_width = 0.5,
        max_ms1_rt_span_sec = None,
        max_ms2_rt_span_sec = None,
        min_raw_sum = 0.0,
        max_rt_apex_delta_sec = None,
        max_scan_apex_delta = None,
        min_im_overlap_scans = 1,
        reject_frag_inside_precursor_tile = true,
    ))]
    pub fn new(
        min_rt_jaccard: f32,
        ms2_rt_guard_sec: f64,
        rt_bucket_width: f64,
        max_ms1_rt_span_sec: Option<f64>,
        max_ms2_rt_span_sec: Option<f64>,
        min_raw_sum: f32,
        max_rt_apex_delta_sec: Option<f32>,
        max_scan_apex_delta: Option<usize>,
        min_im_overlap_scans: usize,
        reject_frag_inside_precursor_tile: bool,
    ) -> Self {
        PyCandidateOpts {
            inner: CandidateOpts {
                min_rt_jaccard,
                ms2_rt_guard_sec,
                rt_bucket_width,
                max_ms1_rt_span_sec,
                max_ms2_rt_span_sec,
                min_raw_sum,
                max_rt_apex_delta_sec,
                max_scan_apex_delta,
                min_im_overlap_scans,
                reject_frag_inside_precursor_tile,
            },
        }
    }

    // You can add getters here later if you want Python-side introspection.
}
#[pyclass]
#[allow(dead_code)]
pub struct PyFragmentIndex {
    /// Fragment index with owned MS2 storage.
    pub(crate) inner: FragmentIndex,
}

#[pymethods]
impl PyFragmentIndex {
    #[new]
    #[pyo3(signature = (ds, ms2_clusters, cand_opts))]
    pub fn new(
        ds: &PyTimsDatasetDIA,
        ms2_clusters: Vec<Py<PyClusterResult1D>>,
        cand_opts: &PyCandidateOpts,
        py: Python<'_>,
    ) -> PyResult<Self> {
        // Convert Python MS2 objects → Vec<ClusterResult1D>
        let ms2_vec: Vec<ClusterResult1D> = ms2_clusters
            .into_iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();

        let dia_arc: Arc<DiaIndex> = ds.inner.dia_index.clone().into();
        let inner = FragmentIndex::from_owned(dia_arc, ms2_vec, &cand_opts.inner);

        Ok(PyFragmentIndex { inner })
    }

    /// Build a FragmentIndex directly from a directory of fragment-cluster Parquet files.
    ///
    /// Python signature:
    ///   PyFragmentIndex.from_parquet_dir(ds, parquet_dir, cand_opts)
    #[staticmethod]
    #[pyo3(signature = (ds, parquet_dir, cand_opts))]
    pub fn from_parquet_dir(
        ds: &PyTimsDatasetDIA,
        parquet_dir: String,
        cand_opts: &PyCandidateOpts,
    ) -> PyResult<Self> {
        let dia_arc: Arc<DiaIndex> = ds.inner.dia_index.clone().into();

        let idx = FragmentIndex::from_parquet_dir(dia_arc, &parquet_dir, &cand_opts.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))?;

        Ok(PyFragmentIndex { inner: idx })
    }

    #[pyo3(signature = (
        prec,
        window_groups = None,
        max_rt_apex_delta_sec = Some(2.0),
        max_scan_apex_delta = Some(6),
        min_im_overlap_scans = 1,
        require_tile_compat = true,
        reject_frag_inside_precursor_tile = true,
    ))]
    pub fn query_precursor(
        &self,
        prec: &PyClusterResult1D,
        window_groups: Option<Vec<u32>>,
        max_rt_apex_delta_sec: Option<f32>,
        max_scan_apex_delta: Option<usize>,
        min_im_overlap_scans: usize,
        require_tile_compat: bool,
        reject_frag_inside_precursor_tile: bool,
    ) -> PyResult<Vec<u64>> {
        let opts = FragmentQueryOpts {
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            min_im_overlap_scans,
            require_tile_compat,
            reject_frag_inside_precursor_tile,
        };
        let prec_rust = &prec.inner;

        let groups_opt = window_groups.as_ref().map(|v| v.as_slice());
        let hits = self.inner.query_precursor(prec_rust, groups_opt, &opts);
        Ok(hits)
    }

    #[pyo3(signature = (
        precs,
        max_rt_apex_delta_sec = Some(2.0),
        max_scan_apex_delta = Some(6),
        min_im_overlap_scans = 1,
        require_tile_compat = true,
        reject_frag_inside_precursor_tile = true,
        num_threads = 0,
    ))]
    pub fn query_precursors_par(
        &self,
        precs: Vec<Py<PyClusterResult1D>>,
        max_rt_apex_delta_sec: Option<f32>,
        max_scan_apex_delta: Option<usize>,
        min_im_overlap_scans: usize,
        require_tile_compat: bool,
        reject_frag_inside_precursor_tile: bool,
        num_threads: usize,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<u64>>> {
        let opts = FragmentQueryOpts {
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            min_im_overlap_scans,
            require_tile_compat,
            reject_frag_inside_precursor_tile,
        };

        // Still clones precursors here, but that's much smaller than cloning all MS2.
        let precs_rust: Vec<ClusterResult1D> = precs
            .into_iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();

        let all_hits = self.inner.query_precursors_par(&precs_rust, &opts, num_threads);
        Ok(all_hits)
    }

    // ------------------------------------------------------------------
    // 1) Cluster-based: single precursor, scored
    // ------------------------------------------------------------------
    #[pyo3(signature = (
        prec,
        window_groups = None,
        mode = "geom",
        min_score = 0.0,
        reject_frag_inside_precursor_tile = true,
        max_rt_apex_delta_sec = Some(2.0),
        max_scan_apex_delta = Some(6),
        min_im_overlap_scans = 1,
        require_tile_compat = true,
    ))]
    pub fn query_precursor_scored(
        &self,
        prec: &PyClusterResult1D,
        window_groups: Option<Vec<u32>>,
        mode: &str,
        min_score: f32,
        reject_frag_inside_precursor_tile: bool,
        max_rt_apex_delta_sec: Option<f32>,
        max_scan_apex_delta: Option<usize>,
        min_im_overlap_scans: usize,
        require_tile_compat: bool,
    ) -> PyResult<Vec<PyScoredHit>> {
        let prec_rust: &ClusterResult1D = &prec.inner;

        let mode_rs = parse_match_score_mode(mode)?;
        let geom_opts = ScoreOpts::default();
        let xic_opts  = XicScoreOpts::default();

        let frag_opts = FragmentQueryOpts {
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            min_im_overlap_scans,
            require_tile_compat,
            reject_frag_inside_precursor_tile,
        };

        let groups_opt = window_groups.as_ref().map(|v| v.as_slice());

        let hits: Vec<ScoredHit> = self.inner.query_precursor_scored(
            prec_rust,
            groups_opt,
            &frag_opts,
            mode_rs,
            &geom_opts,
            &xic_opts,
            min_score,
        );

        Ok(hits.into_iter().map(|h| PyScoredHit { inner: h }).collect())
    }

    // ------------------------------------------------------------------
    // 2) Cluster-based: many precursors, scored in parallel
    // ------------------------------------------------------------------
    #[pyo3(signature = (
        precs,
        mode = "geom",
        min_score = 0.0,
        reject_frag_inside_precursor_tile = true,
        max_rt_apex_delta_sec = Some(2.0),
        max_scan_apex_delta = Some(6),
        min_im_overlap_scans = 1,
        require_tile_compat = true,
    ))]
    pub fn query_precursors_scored_par(
        &self,
        precs: Vec<Py<PyClusterResult1D>>,
        mode: &str,
        min_score: f32,
        reject_frag_inside_precursor_tile: bool,
        max_rt_apex_delta_sec: Option<f32>,
        max_scan_apex_delta: Option<usize>,
        min_im_overlap_scans: usize,
        require_tile_compat: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<PyScoredHit>>> {
        let precs_rust: Vec<ClusterResult1D> = precs
            .into_iter()
            // remove the raw data to avoid killing the memory
            .map(|p| {
                let mut c = p.borrow(py).inner.clone();
                c.rt_axis_sec = None;
                c.im_axis_scans = None;
                c.mz_axis_da = None;
                c.raw_points = None;
                c.rt_trace = None;
                c.im_trace = None;
                c
            }
            )
            .collect();

        let mode_rs = parse_match_score_mode(mode)?;
        let geom_opts = ScoreOpts::default();
        let xic_opts  = XicScoreOpts::default();

        let frag_opts = FragmentQueryOpts {
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            min_im_overlap_scans,
            require_tile_compat,
            reject_frag_inside_precursor_tile,
        };

        let all_hits: Vec<Vec<ScoredHit>> = self.inner.query_precursors_scored_par(
            &precs_rust,
            &frag_opts,
            mode_rs,
            &geom_opts,
            &xic_opts,
            min_score,
        );

        let wrapped: Vec<Vec<PyScoredHit>> = all_hits
            .into_iter()
            .map(|hits| {
                hits.into_iter()
                    .map(|h| PyScoredHit { inner: h })
                    .collect()
            })
            .collect();

        Ok(wrapped)
    }

    // ------------------------------------------------------------------
    // 3) Feature-based: single feature, scored
    // ------------------------------------------------------------------
    #[pyo3(signature = (
        feat,
        mode = "geom",
        min_score = 0.0,
        reject_frag_inside_precursor_tile = true,
        max_rt_apex_delta_sec = Some(2.0),
        max_scan_apex_delta = Some(6),
        min_im_overlap_scans = 1,
        require_tile_compat = true,
    ))]
    pub fn score_feature(
        &self,
        feat: Py<PySimpleFeature>,
        mode: &str,
        min_score: f32,
        reject_frag_inside_precursor_tile: bool,
        max_rt_apex_delta_sec: Option<f32>,
        max_scan_apex_delta: Option<usize>,
        min_im_overlap_scans: usize,
        require_tile_compat: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<PyScoredHit>> {
        let feat_rust: SimpleFeature = feat.borrow(py).inner.clone();
        let mode_rs = parse_match_score_mode(mode)?;
        let geom_opts = ScoreOpts::default();
        let xic_opts  = XicScoreOpts::default();

        let frag_opts = FragmentQueryOpts {
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            min_im_overlap_scans,
            require_tile_compat,
            reject_frag_inside_precursor_tile,
        };

        let hits = self.inner.query_feature_scored(
            &feat_rust,
            &frag_opts,
            mode_rs,
            &geom_opts,
            &xic_opts,
            min_score,
        );

        Ok(hits.into_iter().map(|h| PyScoredHit { inner: h }).collect())
    }

    /// Score many SimpleFeatures in parallel.
    ///
    /// mode: "geom" | "xic"
    /// min_score: keep only hits with score >= min_score
    #[pyo3(signature = (
        feats,
        mode = "geom",
        min_score = 0.0,
        reject_frag_inside_precursor_tile = true,
        max_rt_apex_delta_sec = Some(2.0),
        max_scan_apex_delta = Some(6),
        min_im_overlap_scans = 1,
        require_tile_compat = true,
    ))]
    pub fn score_features_par(
        &self,
        feats: Vec<Py<PySimpleFeature>>,
        mode: &str,
        min_score: f32,
        reject_frag_inside_precursor_tile: bool,
        max_rt_apex_delta_sec: Option<f32>,
        max_scan_apex_delta: Option<usize>,
        min_im_overlap_scans: usize,
        require_tile_compat: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<PyScoredHit>>> {
        let feats_rust: Vec<SimpleFeature> = feats
            .into_iter()
            .map(|f| f.borrow(py).inner.clone())
            .collect();

        let mode_rs = parse_match_score_mode(mode)?;
        let geom_opts = ScoreOpts::default();
        let xic_opts  = XicScoreOpts::default();

        let frag_opts = FragmentQueryOpts {
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            min_im_overlap_scans,
            require_tile_compat,
            reject_frag_inside_precursor_tile,
        };

        let all_hits: Vec<Vec<ScoredHit>> = self.inner.query_features_scored_par(
            &feats_rust,
            &frag_opts,
            mode_rs,
            &geom_opts,
            &xic_opts,
            min_score,
        );

        let wrapped: Vec<Vec<PyScoredHit>> = all_hits
            .into_iter()
            .map(|hits| {
                hits.into_iter()
                    .map(|h| PyScoredHit { inner: h })
                    .collect()
            })
            .collect();

        Ok(wrapped)
    }

    #[pyo3(signature = (
        feats,
        mode = "geom",
        min_score = 0.0,
        reject_frag_inside_precursor_tile = true,
        max_rt_apex_delta_sec = Some(2.0),
        max_scan_apex_delta = Some(6),
        min_im_overlap_scans = 1,
        require_tile_compat = true,
        min_fragments = 4,
    ))]
    pub fn score_features_to_pseudospectra_par(
        &self,
        feats: Vec<Py<PySimpleFeature>>,
        mode: &str,
        min_score: f32,
        reject_frag_inside_precursor_tile: bool,
        max_rt_apex_delta_sec: Option<f32>,
        max_scan_apex_delta: Option<usize>,
        min_im_overlap_scans: usize,
        require_tile_compat: bool,
        min_fragments: usize,
        py: Python<'_>,
    ) -> PyResult<Vec<PyPseudoSpectrum>> {
        let feats_rust: Vec<SimpleFeature> = feats
            .into_iter()
            .map(|f| f.borrow(py).inner.clone())
            .collect();

        let mode_rs = parse_match_score_mode(mode)?;
        let geom_opts = ScoreOpts::default();
        let xic_opts  = XicScoreOpts::default();

        let frag_opts = FragmentQueryOpts {
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            min_im_overlap_scans,
            require_tile_compat,
            reject_frag_inside_precursor_tile,
        };

        // Parallel scoring inside FragmentIndex
        let all_hits: Vec<Vec<ScoredHit>> = self.inner.query_features_scored_par(
            &feats_rust,
            &frag_opts,
            mode_rs,
            &geom_opts,
            &xic_opts,
            min_score,
        );

        let ms2 = self.inner.ms2_slice();

        // Build PseudoSpectra on the Rust side
        let mut out: Vec<PyPseudoSpectrum> = Vec::new();
        for (feat, hits) in feats_rust.iter().zip(all_hits.iter()) {
            if hits.len() < min_fragments {
                continue;
            }
            if let Some(ps) = pseudospectrum_from_feature_and_hits(feat, hits, ms2) {
                out.push(PyPseudoSpectrum { inner: ps });
            }
        }

        Ok(out)
    }

    /// Score many precursor clusters in parallel and build PseudoSpectra.
    ///
    /// Returns one PseudoSpectrum per precursor cluster with >= min_fragments hits.
    #[pyo3(signature = (
        precs,
        mode = "geom",
        min_score = 0.0,
        reject_frag_inside_precursor_tile = true,
        max_rt_apex_delta_sec = Some(2.0),
        max_scan_apex_delta = Some(6),
        min_im_overlap_scans = 1,
        require_tile_compat = true,
        min_fragments = 4,
    ))]
    pub fn query_precursors_to_pseudospectra_par(
        &self,
        precs: Vec<Py<PyClusterResult1D>>,
        mode: &str,
        min_score: f32,
        reject_frag_inside_precursor_tile: bool,
        max_rt_apex_delta_sec: Option<f32>,
        max_scan_apex_delta: Option<usize>,
        min_im_overlap_scans: usize,
        require_tile_compat: bool,
        min_fragments: usize,
        py: Python<'_>,
    ) -> PyResult<Vec<PyPseudoSpectrum>> {
        let precs_rust: Vec<ClusterResult1D> = precs
            .into_iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();

        let mode_rs   = parse_match_score_mode(mode)?;
        let geom_opts = ScoreOpts::default();
        let xic_opts  = XicScoreOpts::default();

        let frag_opts = FragmentQueryOpts {
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            min_im_overlap_scans,
            require_tile_compat,
            reject_frag_inside_precursor_tile,
        };

        let all_hits: Vec<Vec<ScoredHit>> = self.inner.query_precursors_scored_par(
            &precs_rust,
            &frag_opts,
            mode_rs,
            &geom_opts,
            &xic_opts,
            min_score,
        );

        let ms2 = self.inner.ms2_slice();

        let mut out: Vec<PyPseudoSpectrum> = Vec::new();
        for (prec, hits) in precs_rust.iter().zip(all_hits.iter()) {
            if hits.len() < min_fragments {
                continue;
            }
            if let Some(ps) = pseudospectrum_from_cluster_and_hits(prec, hits, ms2) {
                out.push(PyPseudoSpectrum { inner: ps });
            }
        }

        Ok(out)
    }
}

// unchanged
fn parse_match_score_mode(mode: &str) -> PyResult<MatchScoreMode> {
    let m = mode.to_ascii_lowercase();
    match m.as_str() {
        "geom" | "geometric" => Ok(MatchScoreMode::Geom),
        "xic" | "extracted_chromatogram"   => Ok(MatchScoreMode::Xic),
        other => Err(PyValueError::new_err(format!(
            "Unknown match score mode '{other}'. \
             Expected one of: 'geom', 'xic'."
        ))),
    }
}

/// Build a PseudoSpectrum from a single precursor cluster and its scored hits.
///
/// - `prec` : precursor cluster (MS1)
/// - `hits` : scored fragment hits (ScoredHit::frag_idx indexes into `ms2`)
/// - `ms2`  : full MS2 cluster slice owned by FragmentIndex
pub fn pseudospectrum_from_cluster_and_hits(
    prec: &ClusterResult1D,
    hits: &[ScoredHit],
    ms2: &[ClusterResult1D],
) -> Option<PseudoSpectrum> {
    use std::cmp::Ordering;

    let mut frags: Vec<PseudoFragment> = Vec::new();
    let mut window_groups: Vec<u32> = Vec::new();

    for h in hits {
        let j = h.frag_idx;
        if j >= ms2.len() {
            continue;
        }
        let c2 = &ms2[j];

        // Only keep MS2 clusters as fragments
        if c2.ms_level != 2 {
            continue;
        }

        if let Some(wg) = c2.window_group {
            if !window_groups.contains(&wg) {
                window_groups.push(wg);
            }
        }

        if let Some(pf) = fragment_from_cluster(c2) {
            frags.push(pf);
        }
    }

    if frags.is_empty() {
        return None;
    }

    // Deterministic ordering
    window_groups.sort_unstable();
    frags.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap_or(Ordering::Equal));

    // Precursor m/z: prefer mz_fit, fallback to window midpoint
    let precursor_mz = if let Some(fit) = &prec.mz_fit {
        fit.mu
    } else if let Some((lo, hi)) = prec.mz_window {
        0.5 * (lo + hi)
    } else {
        return None;
    };

    let rt_apex = prec.rt_fit.mu;
    let im_apex = prec.im_fit.mu;

    Some(PseudoSpectrum {
        precursor_mz,
        precursor_charge: 0, // not known from cluster
        rt_apex,
        im_apex,
        feature_id: None,
        window_groups,
        precursor_cluster_ids: vec![prec.cluster_id],
        fragments: frags,
        precursor_cluster_indices: vec![],
    })
}

/// Build a PseudoSpectrum from a SimpleFeature and its scored hits.
pub fn pseudospectrum_from_feature_and_hits(
    feat: &SimpleFeature,
    hits: &[ScoredHit],
    ms2: &[ClusterResult1D],
) -> Option<PseudoSpectrum> {
    use std::cmp::Ordering;

    let mut frags: Vec<PseudoFragment> = Vec::new();
    let mut window_groups: Vec<u32> = Vec::new();

    for h in hits {
        let j = h.frag_idx;
        if j >= ms2.len() {
            continue;
        }
        let c2 = &ms2[j];

        if c2.ms_level != 2 {
            continue;
        }

        if let Some(wg) = c2.window_group {
            if !window_groups.contains(&wg) {
                window_groups.push(wg);
            }
        }

        if let Some(pf) = fragment_from_cluster(c2) {
            frags.push(pf);
        }
    }

    if frags.is_empty() {
        return None;
    }

    window_groups.sort_unstable();
    frags.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap_or(Ordering::Equal));

    let precursor_mz = feat.mz_mono;
    let precursor_charge = feat.charge;
    let feature_id = Some(feat.feature_id);

    let mut _rt_apex: f32;
    let mut _im_apex: f32;
    let mut precursor_cluster_ids: Vec<u64> = Vec::new();

    if let Some(top) = feat
        .member_clusters
        .iter()
        .max_by(|a, b| {
            a.raw_sum
                .partial_cmp(&b.raw_sum)
                .unwrap_or(Ordering::Equal)
        })
    {
        _rt_apex = top.rt_fit.mu;
        _im_apex = top.im_fit.mu;
        precursor_cluster_ids.push(top.cluster_id);
    } else {
        let (rt_lo, rt_hi) = feat.rt_bounds;
        let (im_lo, im_hi) = feat.im_bounds;
        _rt_apex = 0.5 * (rt_lo as f32 + rt_hi as f32);
        _im_apex = 0.5 * (im_lo as f32 + im_hi as f32);
    }

    Some(PseudoSpectrum {
        precursor_mz,
        precursor_charge,
        rt_apex: _rt_apex,
        im_apex: _im_apex,
        feature_id,
        window_groups,
        precursor_cluster_ids,
        fragments: frags,
        precursor_cluster_indices: feat.member_cluster_indices.clone(),
    })
}

#[pymodule]
pub fn py_dia(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimsDatasetDIA>()?;
    m.add_class::<PyTofScanPlan>()?;
    m.add_class::<PyTofScanPlanGroup>()?;
    m.add_class::<PyTofScanWindowGrid>()?;
    m.add_class::<PyImPeak1D>()?;
    m.add_class::<PyRtPeak1D>()?;
    m.add_class::<PyFit1D>()?;
    m.add_class::<PyClusterResult1D>()?;
    m.add_class::<PyTofRtGrid>()?;
    m.add_class::<PyAssignmentResult>()?;
    m.add_class::<PyPseudoBuildResult>()?;
    m.add_class::<PyFragmentIndex>()?;
    m.add_class::<PyScoredHit>()?;
    m.add_class::<PyCandidateOpts>()?;
    m.add_function(wrap_pyfunction!(stitch_im_peaks_flat_unordered, m)?)?;
    m.add_function(wrap_pyfunction!(save_clusters_bin, m)?)?;
    m.add_function(wrap_pyfunction!(load_clusters_bin, m)?)?;
    m.add_function(wrap_pyfunction!(save_clusters_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(load_clusters_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(save_pseudo_spectra_bin, m)?)?;
    m.add_function(wrap_pyfunction!(load_pseudo_spectra_bin, m)?)?;
    Ok(())
}