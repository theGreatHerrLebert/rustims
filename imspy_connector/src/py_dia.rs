use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PySlice};
use numpy::{PyArray1, PyArray2};
use numpy::ndarray::{Array2, ShapeBuilder};

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use rustdf::data::dia::TimsDatasetDIA;
use rustdf::data::handle::TimsData;
use rustdf::cluster::peak::{MzScanWindowGrid, FrameBinView, build_frame_bin_view, ImPeak1D};
use rustdf::cluster::utility::{MzScale, scan_mz_range, smooth_vector_gaussian};
use crate::py_tims_frame::PyTimsFrame;
use crate::py_tims_slice::PyTimsSlice;

#[pyclass]
pub struct PyImPeak1D { pub inner: ImPeak1D }

#[pymethods]
impl PyImPeak1D {
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
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("shape error: {e}")))?;
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

    /// Pick 1D IM peaks in each m/z row of this window.
    /// Returns a nested list: List[List[PyImPeak1D]] with outer index = row (m/z bin).
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
        use rustdf::cluster::utility::{find_im_peaks_row, ImRowContext, MobilityFn};

        let rows = self.inner.rows;
        let cols = self.inner.cols;

        let mob_fn: MobilityFn = if use_mobility {
            // TODO: plug your real scan->1/K0 converter
            Some(|scan| scan as f32)
        } else {
            None
        };

        // Build per-row contexts from the window metadata once.
        let ctx_template = ImRowContext {
            mz_row: 0, // will overwrite per row
            mz_center: 0.0, // dummy
            mz_bounds: (0.0, 0.0),                     // dummy
            rt_bounds: self.inner.rt_range_frames,         // <- window’s frame-index range
            frame_id_bounds: self.inner.frame_id_bounds,   // <- actual frame IDs
            window_group: self.inner.window_group,         // <- DIA group if any
        };

        // For each m/z row, gather the scan profile and run the row picker with context.
        let rows_rs: Vec<Vec<ImPeak1D>> = (0..rows).map(|r| {
            // Gather this row across scans (column-major)
            let mut y_s = Vec::with_capacity(cols);
            let mut y_r = Vec::with_capacity(cols);
            for s in 0..cols {
                let val_s = self.inner.data[s * rows + r];
                y_s.push(val_s);
                let val_r = self.inner.data_raw
                    .as_ref()
                    .map(|dr| dr[s * rows + r])
                    .unwrap_or(val_s);
                y_r.push(val_r);
            }

            let mut ctx = ctx_template;
            ctx.mz_row = r;

            find_im_peaks_row(
                &y_s, &y_r, &ctx, mob_fn,
                min_prom, min_distance_scans, min_width_scans
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
    scale: MzScale,
    frame_ids_sorted: Vec<u32>,
    frame_times: Vec<f32>,
    windows_idx: Vec<(usize, usize)>,
    rows: usize,
    global_num_scans: usize,

    // exec params
    maybe_sigma_scans: Option<f32>,
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
            scale,
            frame_ids_sorted,
            frame_times,
            windows_idx,
            rows,
            global_num_scans,
            maybe_sigma_scans,
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
                    row_py.push(Py::new(py, PyImPeak1D { inner: p })?);
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

        // Views for this window (group frames only)
        let views_local: Vec<FrameBinView> = if let Some(ref views) = self.views {
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
            rt_range_frames: (lo, hi),
            rt_range_sec: (self.frame_times[lo], self.frame_times[hi]),
            frame_id_bounds,                           // NEW
            window_group: Some(self.window_group),     // NEW
            scans: (0..cols).collect(),
            data,
            rows,
            cols,
            data_raw: if do_smooth { Some(raw) } else { None },
        })
    }
}

#[pyclass]
pub struct PyMzScanPlan {
    ds: Py<PyAny>,                 // keeps dataset alive; downcast when needed

    // planned axes + schedule
    scale: MzScale,
    frame_ids_sorted: Vec<u32>,
    frame_times: Vec<f32>,
    windows_idx: Vec<(usize, usize)>,
    rows: usize,
    global_num_scans: usize,

    // exec params
    maybe_sigma_scans: Option<f32>,
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
            scale,
            frame_ids_sorted,
            frame_times,
            windows_idx,
            rows,
            global_num_scans,
            maybe_sigma_scans,
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
                    row_py.push(Py::new(py, PyImPeak1D { inner: p })?);
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

        // Views for this window; borrow the dataset only around get_slice
        let views_local: Vec<FrameBinView> = if let Some(ref views) = self.views {
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
            rt_range_frames: (lo, hi),
            rt_range_sec: (self.frame_times[lo], self.frame_times[hi]),
            frame_id_bounds,           // NEW
            window_group: None,        // NEW
            scans: (0..cols).collect(),
            data,
            rows,
            cols,
            data_raw: if do_smooth { Some(raw) } else { None },
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
}

fn impeaks_to_py_nested(py: Python<'_>, rows: Vec<Vec<ImPeak1D>>) -> PyResult<Vec<Vec<Py<PyImPeak1D>>>> {
    Ok(rows.into_iter().map(|row| {
        row.into_iter().map(|p| Py::new(py, PyImPeak1D{ inner: p }).unwrap()).collect()
    }).collect())
}

#[inline]
fn pick_im_peaks_rows_from_grid(
    grid: &MzScanWindowGrid,
    min_prom: f32,
    min_distance_scans: usize,
    min_width_scans: usize,
    use_mobility: bool,
) -> Vec<Vec<ImPeak1D>> {
    use rustdf::cluster::utility::{find_im_peaks_row, ImRowContext, MobilityFn};

    let rows = grid.rows;
    let cols = grid.cols;

    let mob_fn: MobilityFn = if use_mobility {
        Some(|scan| scan as f32)
    } else {
        None
    };

    let ctx_template = ImRowContext {
        mz_row: 0,
        mz_center: 0.0,
        mz_bounds: (0.0, 0.0),
        rt_bounds: grid.rt_range_frames,
        frame_id_bounds: grid.frame_id_bounds,
        window_group: grid.window_group,
    };

    (0..rows).map(|r| {
        // gather row across scans (column-major)
        let mut y_s = Vec::with_capacity(cols);
        let mut y_r = Vec::with_capacity(cols);
        for s in 0..cols {
            let val_s = grid.data[s * rows + r];
            y_s.push(val_s);
            let val_r = grid.data_raw
                .as_ref()
                .map(|dr| dr[s * rows + r])
                .unwrap_or(val_s);
            y_r.push(val_r);
        }

        let mut ctx = ctx_template;
        ctx.mz_row = r;

        find_im_peaks_row(
            &y_s, &y_r, &ctx, mob_fn,
            min_prom, min_distance_scans, min_width_scans
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
            // Small sort per row keeps merges stable and cheap
            let mut row_rs: Vec<ImPeak1D> = row
                .into_iter()
                .map(|p| p.borrow(py).inner.clone())
                .collect();
            row_rs.sort_unstable_by(|a,b| a.scan.cmp(&b.scan).then(a.rt_bounds.0.cmp(&b.rt_bounds.0)));

            for p in row_rs.into_iter() {
                let scan_bin = p.scan / params.max_scan_delta;
                let key = Key { wg: p.window_group, mz_row: p.mz_row, scan_bin };

                if let Some(cur) = active.get_mut(&key) {
                    if compatible_fast(cur, &p, &params) {
                        merge_into(cur, &p);
                        continue;
                    }
                    // If the new peak starts clearly after the current can no longer overlap,
                    // flush and replace. This keeps only one accumulator per key.
                    if p.rt_bounds.0 > cur.rt_bounds.1 + params.min_overlap_frames {
                        let flushed = active.remove(&key).unwrap();
                        out.push(Py::new(py, PyImPeak1D { inner: flushed })?);
                        active.insert(key, p);
                    } else {
                        // Overlapping-but-incompatible; finalize current and start a new run.
                        let flushed = active.remove(&key).unwrap();
                        out.push(Py::new(py, PyImPeak1D { inner: flushed })?);
                        active.insert(key, p);
                    }
                } else {
                    active.insert(key, p);
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
        out.push(Py::new(py, PyImPeak1D { inner: v })?);
    }
    Ok(out)
}

#[pymodule]
pub fn py_dia(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimsDatasetDIA>()?;
    m.add_class::<PyMzScanPlan>()?;
    m.add_class::<PyMzScanPlanGroup>()?;
    m.add_class::<PyMzScanWindowGrid>()?;
    m.add_class::<PyImPeak1D>()?;
    m.add_function(wrap_pyfunction!(stitch_im_peaks_batched_streaming, m)?)?;
    Ok(())
}