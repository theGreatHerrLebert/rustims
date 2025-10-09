use pyo3::prelude::*;
use rustdf::data::dia::TimsDatasetDIA;
use rustdf::data::handle::TimsData;
use rustdf::cluster::utility::{RtPeak1D, ImPeak1D, RtIndex, ImIndex, MobilityFn, pick_im_peaks_on_imindex, build_dense_im_by_rtpeaks_ppm};

use crate::py_tims_frame::PyTimsFrame;
use crate::py_tims_slice::PyTimsSlice;
use numpy::{PyArray1, PyArray2};
use numpy::ndarray::{Array2, ShapeBuilder};
use std::sync::Arc;
use rustdf::cluster::cluster_eval::{evaluate_clusters_3d, ClusterResult, ClusterSpec, EvalOptions};
use rustdf::cluster::feature::{Envelope, Feature};
use crate::py_cluster::{PyClusterResult, PyClusterSpec, PyEvalOptions};
use crate::py_feature::{PyAveragineLut, PyEnvelope, PyFeature, PyFeatureBuildParams, PyGroupingParams};

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
        let results_rs: Vec<ClusterResult> =
            evaluate_clusters_3d(&self.inner, rt_rs, &specs_rs, eval_opts, num_threads);

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
}

fn peaks_to_py<'py>(py: Python<'py>, peaks: Vec<RtPeak1D>) -> PyResult<Vec<Py<PyRtPeak1D>>> {
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
    Ok(())
}