use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2};
use numpy::ndarray::{Array2, ShapeBuilder};
use std::sync::Arc;
use rayon::prelude::*;
use rustdf::data::dia::{annotate_precursor_groups, TimsDatasetDIA};
use rustdf::data::handle::TimsData;

use rustdf::cluster::utility::{RtPeak1D, ImPeak1D, RtIndex, ImIndex, MobilityFn, pick_im_peaks_on_imindex, build_dense_im_by_rtpeaks_ppm, smooth_vector_gaussian, MzScale, scan_mz_range, FrameBinView, build_frame_bin_view};

use rustdf::cluster::im_centric::{MzScanWindowGrid};
use rustdf::cluster::cluster_eval::{evaluate_clusters_3d, ClusterResult, ClusterSpec, EvalOptions};
use rustdf::cluster::feature::{Envelope, Feature};
use crate::py_tims_frame::PyTimsFrame;
use crate::py_tims_slice::PyTimsSlice;
use crate::py_cluster::{PyClusterResult, PyClusterSpec, PyEvalOptions};
use crate::py_feature::{PyAveragineLut, PyEnvelope, PyFeature, PyFeatureBuildParams, PyGroupingParams};

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

        Ok(MzScanWindowGrid {
            rt_range_frames: (lo, hi),
            rt_range_sec: (self.frame_times[lo], self.frame_times[hi]),
            scans: (0..cols).collect(),
            data,
            rows,
            cols,
            data_raw: if do_smooth { Some(raw) } else { None },
        })
    }
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
}
pub fn impeaks_to_py_nested(py: Python<'_>, rows: Vec<Vec<ImPeak1D>>) -> PyResult<Vec<Vec<Py<PyImPeak1D>>>> {
    Ok(rows.into_iter().map(|row| {
        row.into_iter().map(|p| Py::new(py, PyImPeak1D{ inner: p }).unwrap()).collect()
    }).collect())
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyRtIndex {
    pub inner: Arc<RtIndex>,
}

#[pymethods]
impl PyRtIndex {
    #[getter]
    fn centers<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec_bound(py, self.inner.scale.centers.clone()).unbind())
    }
    #[getter]
    fn frame_ids<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<u32>>> {
        Ok(PyArray1::from_vec_bound(py, self.inner.frames.clone()).unbind())
    }
    #[getter]
    fn frame_times<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec_bound(py, self.inner.frame_times.clone()).unbind())
    }
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray2<f32>>> {
        let rt = &self.inner;
        let arr_f = Array2::from_shape_vec((rt.rows, rt.cols).f(), rt.data.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("shape error: {e}")))?;
        Ok(PyArray2::from_owned_array_bound(py, arr_f).unbind())
    }
    #[getter]
    fn data_raw<'py>(&self, py: Python<'py>) -> Option<Py<PyArray2<f32>>> {
        self.inner.data_raw.as_ref().map(|raw| {
            let arr_f = Array2::from_shape_vec((self.inner.rows, self.inner.cols).f(), raw.clone()).unwrap();
            PyArray2::from_owned_array_bound(py, arr_f).unbind()
        })
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyImIndex {
    pub inner: Arc<ImIndex>,
}

#[pymethods]
impl PyImIndex {
    /// IM scan indices (0..cols-1). We expose as u32 for stable dtype across platforms.
    #[getter]
    fn scans<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<u32>>> {
        // Cast usize -> u32 (safe if your scan count fits u32; typical for TIMS)
        let scans_u32: Vec<u32> = self.inner.scans.iter().map(|&s| s as u32).collect();
        Ok(PyArray1::from_vec_bound(py, scans_u32).unbind())
    }

    /// Dense IM matrix (smoothed if smoothing was applied), column-major, shape (rows, cols).
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray2<f32>>> {
        let im = &self.inner;
        let arr_f = Array2::from_shape_vec((im.rows, im.cols).f(), im.data.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("shape error: {e}")))?;
        Ok(PyArray2::from_owned_array_bound(py, arr_f).unbind())
    }

    /// Optional raw (pre-smoothing) IM matrix, same layout/shape as `data`.
    #[getter]
    fn data_raw<'py>(&self, py: Python<'py>) -> Option<Py<PyArray2<f32>>> {
        self.inner.data_raw.as_ref().map(|raw| {
            let arr_f = Array2::from_shape_vec((self.inner.rows, self.inner.cols).f(), raw.clone())
                .expect("internal shape invariant (rows*cols) violated");
            PyArray2::from_owned_array_bound(py, arr_f).unbind()
        })
    }

    /// Convenience—row/col sizes (kept for parity with PyRtIndex pattern).
    #[getter] fn rows(&self) -> usize { self.inner.rows }
    #[getter] fn cols(&self) -> usize { self.inner.cols }
}


#[pyclass]
pub struct PyImPeak1D { pub inner: ImPeak1D }

#[pymethods]
impl PyImPeak1D {
    #[getter] fn rt_row(&self) -> usize { self.inner.rt_row }
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

pub fn impeaks_to_py(py: Python<'_>, rows: Vec<Vec<ImPeak1D>>) -> PyResult<Vec<Vec<Py<PyImPeak1D>>>> {
    Ok(rows.into_iter().map(|row| {
        row.into_iter().map(|p| Py::new(py, PyImPeak1D{ inner: p }).unwrap()).collect()
    }).collect())
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyRtPeak1D {
    pub inner: RtPeak1D,
}

#[pymethods]
impl PyRtPeak1D {
    #[new]
    pub fn new(
        mz_row: usize,
        mz_center: f32,
        rt_col: usize,
        rt_time: f32,
        apex_smoothed: f32,
        apex_raw: f32,
        prominence: f32,
        left: usize,
        right: usize,
        width_frames: usize,
        area_raw: f32,
        subcol: f32,
        left_x: f32,
        right_x: f32,
        left_padded: usize,      // NEW
        right_padded: usize,     // NEW
        area_padded: f32,      // NEW
    ) -> Self {
        PyRtPeak1D {
            inner: RtPeak1D {
                mz_row,
                mz_center,
                rt_col,
                rt_time,
                apex_smoothed,
                apex_raw,
                prominence,
                left,
                right,
                width_frames,
                area_raw,
                subcol,
                left_x,
                right_x,
                left_padded,
                right_padded,
                area_padded,
            }
        }
    }

    // --- getters mirroring your style ---

    #[getter] pub fn mz_row(&self) -> usize { self.inner.mz_row }
    #[getter] pub fn mz_center(&self) -> f32 { self.inner.mz_center }
    #[getter] pub fn rt_col(&self) -> usize { self.inner.rt_col }
    #[getter] pub fn rt_time(&self) -> f32 { self.inner.rt_time }
    #[getter] pub fn apex_smoothed(&self) -> f32 { self.inner.apex_smoothed }
    #[getter] pub fn apex_raw(&self) -> f32 { self.inner.apex_raw }
    #[getter] pub fn prominence(&self) -> f32 { self.inner.prominence }
    #[getter] pub fn left(&self) -> usize { self.inner.left }
    #[getter] pub fn right(&self) -> usize { self.inner.right }
    #[getter] pub fn width_frames(&self) -> usize { self.inner.width_frames }
    #[getter] pub fn area_raw(&self) -> f32 { self.inner.area_raw }
    #[getter] pub fn subcol(&self) -> f32 { self.inner.subcol }
    #[getter] pub fn left_x(&self) -> f32 { self.inner.left_x }
    #[getter] pub fn right_x(&self) -> f32 { self.inner.right_x }
    #[getter] pub fn left_padded(&self) -> usize { self.inner.left_padded }      // NEW
    #[getter] pub fn right_padded(&self) -> usize { self.inner.right_padded }    // NEW
    #[getter] pub fn area_padded(&self) -> f32 { self.inner.area_padded }      // NEW

    pub fn __repr__(&self) -> String {
        format!(
            "PyPeak1D(mz_row={}, rt_col={}, rt_time={:.4}, apex_smoothed={:.3}, prominence={:.3}, width_frames={}, area_raw={:.3}, left={}, right={}, subcol={:.3}, left_x={:.3}, right_x={:.3}, left_padded={}, right_padded={}, area_padded={:.3})",

            self.inner.mz_row, self.inner.rt_col, self.inner.rt_time,
            self.inner.apex_smoothed, self.inner.prominence, self.inner.width_frames,
            self.inner.area_raw, self.inner.left, self.inner.right, self.inner.subcol,
            self.inner.left_x, self.inner.right_x, self.inner.left_padded, self.inner.right_padded, self.inner.area_padded
        )
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

    #[pyo3(signature = (truncate, maybe_sigma_frames=None, ppm_per_bin=25.0, mz_pad_ppm=50.0, num_threads=4, min_prom=100.0, min_distance=2, min_width=2, pad_left=1, pad_right=2))]
    pub fn build_dense_rt_by_mz_and_pick(
        &self,
        truncate: f32,
        maybe_sigma_frames: Option<f32>,
        ppm_per_bin: f32,
        mz_pad_ppm: f32,
        num_threads: usize,
        min_prom: f32,
        min_distance: usize,
        min_width: usize,
        pad_left: usize,
        pad_right: usize,
        py: Python<'_>,
    ) -> PyResult<(PyRtIndex, Vec<Py<PyRtPeak1D>>)> {
        let (rt, peaks_rs) = self.inner.pick_peaks_dense(
            maybe_sigma_frames, truncate, ppm_per_bin, mz_pad_ppm, num_threads,
            min_prom, min_distance, min_width, pad_left, pad_right,
        );
        let py_rt = PyRtIndex { inner: Arc::new(rt) };
        let peaks_py = peaks_to_py(py, peaks_rs)?;
        Ok((py_rt, peaks_py))
    }
    #[pyo3(signature = (window_group,maybe_sigma_frames=None,truncate=3.0,ppm_per_bin=25.0,mz_pad_ppm=50.0,clamp_to_group=true,num_threads=4,min_prom=100.0,min_distance=2,min_width=2,pad_left=1,pad_right=2))]
    pub fn build_dense_rt_by_mz_and_window_group_and_pick(
        &self,
        window_group: u32,
        maybe_sigma_frames: Option<f32>,
        truncate: f32,
        ppm_per_bin: f32,
        mz_pad_ppm: f32,
        clamp_to_group: bool,
        num_threads: usize,
        min_prom: f32,
        min_distance: usize,
        min_width: usize,
        pad_left: usize,
        pad_right: usize,
        py: Python<'_>,
    ) -> PyResult<(PyRtIndex, Vec<Py<PyRtPeak1D>>)> {
        let (rt, peaks_rs) = self.inner.pick_peaks_dense_for_group(
            window_group, maybe_sigma_frames, truncate, ppm_per_bin, mz_pad_ppm, clamp_to_group,
            num_threads, min_prom, min_distance, min_width, pad_left, pad_right,
        );
        let py_rt = PyRtIndex { inner: Arc::new(rt) };
        let peaks_py = peaks_to_py(py, peaks_rs)?;
        Ok((py_rt, peaks_py))
    }

    pub fn group_mz_unions(&self) -> std::collections::HashMap<u32, Vec<(f32,f32)>> {
        let map =self.inner.group_mz_unions();
        map.into_iter().collect()
    }

    pub fn groups_covering_mz(&self, mz: f32) -> Vec<u32> {
        let unions = self.inner.group_mz_unions();
        self.inner.groups_covering_mz(mz, &unions)
    }

    // Convenience: build IM index and immediately pick IM peaks per row (simple non-adaptive)
    #[pyo3(signature = (rt_index,peaks,num_threads=4,mz_ppm_window=10.0,rt_extra_pad=0,maybe_sigma_scans=None,truncate=3.0,min_prom=50.0,min_distance_scans=2,min_width_scans=2,use_mobility=false))]
    pub fn build_dense_im_by_rtpeaks_ppm_and_pick(
        &self,
        rt_index: PyRtIndex,
        peaks: Vec<Py<PyRtPeak1D>>,
        num_threads: usize,
        mz_ppm_window: f32,
        rt_extra_pad: usize,
        maybe_sigma_scans: Option<f32>,
        truncate: f32,
        min_prom: f32,
        min_distance_scans: usize,
        min_width_scans: usize,
        use_mobility: bool,
        py: Python<'_>,
    ) -> PyResult<(PyImIndex, Vec<Vec<Py<PyImPeak1D>>>)> {

        // 1) Build
        let peaks_rs: Vec<RtPeak1D> = peaks
            .into_iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();

        let im = build_dense_im_by_rtpeaks_ppm( &self.inner,
                                                peaks_rs,
                                                &rt_index.inner,
                                                num_threads,
                                                mz_ppm_window,
                                                rt_extra_pad,
                                                maybe_sigma_scans,
                                                truncate,
        );

        // 2) Pick IM peaks (operate on column-major buffers already in `im`)
        let mob_fn: MobilityFn = if use_mobility {

        // TODO: wire your real converter here
        Some(|scan| scan as f32)
        } else { None };

        let rows = im.rows;
        let cols = im.cols;
        let rows_rs = pick_im_peaks_on_imindex(
            im.data.as_slice(),
            im.data_raw.as_deref(),
            rows,
            cols,
            min_prom,
            min_distance_scans,
            min_width_scans,
            mob_fn
        );

        let im_py = PyImIndex { inner: Arc::new(im) };
        let peaks_py = impeaks_to_py_nested(py, rows_rs)?;
        Ok((im_py, peaks_py))
    }

    #[pyo3(signature = (
    window_group,
    rt_index,
    peaks,
    num_threads=4,
    mz_ppm_window=10.0,
    rt_extra_pad=0,
    maybe_sigma_scans=None,
    truncate=3.0,
    min_prom=50.0,
    min_distance_scans=2,
    min_width_scans=2,
    clamp_scans_to_group=true,
    use_mobility=false
    ))]
    pub fn build_dense_im_by_rtpeaks_ppm_for_group_and_pick(
        &self,
        window_group: u32,
        rt_index: PyRtIndex,
        peaks: Vec<Py<PyRtPeak1D>>,
        num_threads: usize,
        mz_ppm_window: f32,
        rt_extra_pad: usize,
        maybe_sigma_scans: Option<f32>,
        truncate: f32,
        min_prom: f32,
        min_distance_scans: usize,
        min_width_scans: usize,
        clamp_scans_to_group: bool,
        use_mobility: bool,
        py: Python<'_>,
    ) -> PyResult<(PyImIndex, Vec<Vec<Py<PyImPeak1D>>>)> {
        // 1) unwrap peaks into Rust
        let peaks_rs: Vec<RtPeak1D> = peaks.into_iter().map(|p| p.borrow(py).inner.clone()).collect();

        // 2) build group IM index
        let im = self.inner.get_dense_im_by_rtpeaks_ppm_for_group(
            &rt_index.inner,
            peaks_rs,
            window_group,
            num_threads,
            mz_ppm_window,
            rt_extra_pad,
            maybe_sigma_scans,
            truncate,
            clamp_scans_to_group,
        );

        // 3) IM peak picking (same as MS1 method)
        let mob_fn: MobilityFn = if use_mobility {
            // TODO: replace with a real scan->1/K0 converter if desired
            Some(|scan| scan as f32)
        } else {
            None
        };

        let rows_rs = pick_im_peaks_on_imindex(
            im.data.as_slice(),
            im.data_raw.as_deref(),
            im.rows,
            im.cols,
            min_prom,
            min_distance_scans,
            min_width_scans,
            mob_fn,
        );

        let im_py = PyImIndex { inner: Arc::new(im) };
        let rows_py = impeaks_to_py_nested(py, rows_rs)?;
        Ok((im_py, rows_py))
    }

    /// Evaluate 3D clusters (RT × IM with m/z marginal) for the given specs.
    ///
    /// Python signature:
    ///   evaluate_clusters_3d(rt_index, specs, opts=None) -> List[PyClusterResult]
    #[pyo3(signature = (rt_index, specs, opts=None, num_threads=4))]
    pub fn evaluate_clusters_3d<'py>(
        &self,
        py: Python<'py>,
        rt_index: PyRtIndex,
        specs: Vec<Py<PyClusterSpec>>,
        opts: Option<PyEvalOptions>,
        num_threads: usize,
    ) -> PyResult<Vec<Py<PyClusterResult>>> {
        // Convert inputs
        let rt_rs = &rt_index.inner; // Arc<RtIndex>
        let specs_rs: Vec<ClusterSpec> = specs
            .into_iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();
        let eval_opts: EvalOptions = opts.map(|o| o.inner).unwrap_or_default();

        // Core evaluation
        let mut results_rs: Vec<ClusterResult> =
            evaluate_clusters_3d(&self.inner, rt_rs, &specs_rs, eval_opts, num_threads);

        annotate_precursor_groups(&self.inner, &mut results_rs);

        // Wrap back to Python
        let results_py: Vec<Py<PyClusterResult>> = results_rs
            .into_iter()
            .map(|r| Py::new(py, PyClusterResult { inner: r }))
            .collect::<PyResult<_>>()?;

        Ok(results_py)
    }
    /// Build features from envelopes using preloaded precursor frames internally.
    pub fn build_features_from_envelopes(
        &self,
        envelopes: Vec<PyEnvelope>,
        clusters: Vec<PyClusterResult>,
        lut: PyAveragineLut,
        gp: PyGroupingParams,
        fp: PyFeatureBuildParams,
    ) -> PyResult<Vec<PyFeature>> {
        // Unwrap inner
        let envs: Vec<Envelope> = envelopes.into_iter().map(|e| e.inner).collect();
        let clus: Vec<ClusterResult> = clusters.into_iter().map(|c| c.inner).collect();

        // Call the Rust method on TimsDatasetDIA
        let feats: Vec<Feature> = self
            .inner
            .build_features_from_envelopes(
                &envs,
                &clus,
                &lut.inner,
                &gp.inner,
                &fp.inner,
            );

        // Wrap back for Python
        Ok(feats.into_iter().map(|f| PyFeature { inner: f }).collect())
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
}

fn peaks_to_py(py: Python, peaks: Vec<RtPeak1D>) -> PyResult<Vec<Py<PyRtPeak1D>>> {
    let mut out = Vec::with_capacity(peaks.len());
    for p in peaks {
        out.push(Py::new(py, PyRtPeak1D { inner: p })?);
    }
    Ok(out)
}

#[pymodule]
pub fn py_dia(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimsDatasetDIA>()?;
    m.add_class::<PyRtPeak1D>()?;
    m.add_class::<PyImPeak1D>()?;
    m.add_class::<PyRtIndex>()?;
    m.add_class::<PyImIndex>()?;
    m.add_class::<PyMzScanWindowGrid>()?;
    m.add_class::<PyMzScanPlan>()?;
    Ok(())
}