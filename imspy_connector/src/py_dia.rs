use pyo3::prelude::*;
use rustdf::data::dia::TimsDatasetDIA;
use rustdf::data::handle::TimsData;
use rustdf::cluster::utility::{RtPeak1D, ImPeak1D, pick_im_peaks_on_imindex, pick_im_peaks_on_imindex_adaptive, MobilityFn, ImAdaptivePolicy, FallbackMode};
use rustdf::cluster::utility::ImIndex;
use crate::py_tims_frame::PyTimsFrame;
use crate::py_tims_slice::PyTimsSlice;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{Array2, ShapeBuilder};

fn impeaks_to_py_nested(py: Python<'_>, rows: Vec<Vec<ImPeak1D>>) -> PyResult<Vec<Vec<Py<PyImPeak1D>>>> {
    Ok(rows.into_iter().map(|row| {
        row.into_iter().map(|p| Py::new(py, PyImPeak1D{ inner: p }).unwrap()).collect()
    }).collect())
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
    #[getter] fn left(&self) -> usize { self.inner.left }
    #[getter] fn right(&self) -> usize { self.inner.right }
    #[getter] fn left_x(&self) -> f32 { self.inner.left_x }
    #[getter] fn right_x(&self) -> f32 { self.inner.right_x }
    #[getter] fn width_scans(&self) -> usize { self.inner.width_scans }
    #[getter] fn area_raw(&self) -> f32 { self.inner.area_raw }
    #[getter] fn subscan(&self) -> f32 { self.inner.subscan }
}

fn impeaks_to_py(py: Python<'_>, rows: Vec<Vec<ImPeak1D>>) -> PyResult<Vec<Vec<Py<PyImPeak1D>>>> {
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

    #[pyo3(signature = (resolution, num_threads, truncate, maybe_sigma_frames=None))]
    pub fn build_dense_rt_by_mz(
        &self,
        resolution: usize,
        num_threads: usize,
        truncate: f32,
        maybe_sigma_frames: Option<f32>,
        py: Python<'_>,
    ) -> PyResult<(
        Py<PyArray1<u32>>,
        Py<PyArray1<u32>>,
        Py<PyArray2<f32>>,
    )> {
        let rt = self.inner.get_dense_rt_by_mz(maybe_sigma_frames, truncate, resolution, num_threads);

        let bins_py   = PyArray1::from_vec_bound(py, rt.bins).unbind();
        let frames_py = PyArray1::from_vec_bound(py, rt.frames).unbind();

        let arr_f: Array2<f32> = Array2::from_shape_vec((rt.rows, rt.cols).f(), rt.data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("shape error: {e}")))?;
        let data_py = PyArray2::from_owned_array_bound(py, arr_f).unbind();

        Ok((bins_py, frames_py, data_py))
    }

    #[pyo3(signature = (resolution, num_threads, truncate, maybe_sigma_frames=None, min_prom=100.0, min_distance=2, min_width=2, pad_left=1, pad_right=2))]
    pub fn build_dense_rt_by_mz_and_pick(
        &self,
        resolution: usize,
        num_threads: usize,
        truncate: f32,
        maybe_sigma_frames: Option<f32>,
        min_prom: f32,
        min_distance: usize,
        min_width: usize,
        pad_left: usize,      // NEW
        pad_right: usize,     // NEW
        py: Python<'_>,
    ) -> PyResult<(
        Py<PyArray1<u32>>, // bins
        Py<PyArray1<u32>>, // frames
        Py<PyArray2<f32>>, // data (Fortran/column-major)
        Vec<Py<PyRtPeak1D>>, // peaks
    )> {
        let (rt, peaks_rs) = self.inner.pick_peaks_dense(
            maybe_sigma_frames, truncate, resolution, num_threads,
            min_prom, min_distance, min_width, pad_left, pad_right
        );

        let bins_py   = PyArray1::from_vec_bound(py, rt.bins).unbind();
        let frames_py = PyArray1::from_vec_bound(py, rt.frames).unbind();

        let arr_f: Array2<f32> = Array2::from_shape_vec((rt.rows, rt.cols).f(), rt.data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("shape error: {e}")))?;
        let data_py = PyArray2::from_owned_array_bound(py, arr_f).unbind();

        let peaks_py = peaks_to_py(py, peaks_rs)?;

        Ok((bins_py, frames_py, data_py, peaks_py))
    }

    #[pyo3(signature = (peaks,rt_bins,rt_frames,resolution,num_threads,mz_ppm = 10.0, rt_extra_pad = 0,im_sigma_scans = None,truncate = 3.0))]
    pub fn build_dense_im_by_rtpeaks(
        &self,
        peaks: Vec<Py<PyRtPeak1D>>,
        rt_bins: PyReadonlyArray1<u32>,
        rt_frames: PyReadonlyArray1<u32>,
        resolution: usize,
        num_threads: usize,
        mz_ppm: f32,
        rt_extra_pad: usize,
        im_sigma_scans: Option<f32>,
        truncate: f32,
        py: Python<'_>,
    ) -> PyResult<(
        Py<PyArray1<u32>>,     // scans (0..num_scans-1)
        Py<PyArray2<f32>>,     // data (rows x scans), Fortran/column-major
    )> {
        // 1) convert peaks to Rust
        let peaks_rs: Vec<RtPeak1D> = py_peaks_to_rust(py, peaks);

        // 2) borrow RT bins/frames from numpy
        let bins_slice: &[u32]   = rt_bins.as_slice()?;   // safe, read-only
        let frames_slice: &[u32] = rt_frames.as_slice()?; // safe, read-only

        // 3) build IM index
        let imx: ImIndex = self.inner.get_dense_im_by_rtpeaks(
            peaks_rs,
            bins_slice,
            frames_slice,
            resolution,
            num_threads,
            mz_ppm,
            rt_extra_pad,
            im_sigma_scans,
            truncate,
        );

        // 4) numpy outputs
        // scans: 0..num_scans-1 (cast to u32)
        let scans_vec: Vec<u32> = imx.scans.iter().map(|&s| s as u32).collect();
        let scans_py = PyArray1::from_vec_bound(py, scans_vec).unbind();

        // data: (rows, cols) Fortran-order
        let arr_f: Array2<f32> = Array2::from_shape_vec((imx.rows, imx.cols).f(), imx.data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("shape error: {e}")))?;
        let data_py = PyArray2::from_owned_array_bound(py, arr_f).unbind();

        // Echo peaks back unchanged (already Py objects)
        Ok((scans_py, data_py))
    }

    /// Given the IM matrix you just built (or build it internally), pick IM peaks per row.
    #[pyo3(signature = (data, data_raw=None, min_prom=50.0, min_distance_scans=2, min_width_scans=2, use_mobility=false))]
    pub fn pick_im_peaks_on_matrix(
        &self,
        data: PyReadonlyArray2<f32>,              // Fortran-order (rows, cols)
        data_raw: Option<PyReadonlyArray2<f32>>, // optional raw matrix
        min_prom: f32,
        min_distance_scans: usize,
        min_width_scans: usize,
        use_mobility: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyImPeak1D>>>> {
        use numpy::ndarray::ArrayView2;
        let a: ArrayView2<f32> = data.as_array();
        let (rows, cols) = a.dim();

        // We need column-major; ndarray may be Fortran, but to be safe flatten explicitly:
        let mut cm = vec![0.0f32; rows * cols];
        for s in 0..cols {
            for r in 0..rows {
                cm[s * rows + r] = a[(r, s)];
            }
        }

        let raw_slice: Option<Vec<f32>> = data_raw.map(|dr| {
            let ar = dr.as_array();
            let mut v = vec![0.0f32; rows * cols];
            for s in 0..cols {
                for r in 0..rows {
                    v[s * rows + r] = ar[(r, s)];
                }
            }
            v
        });

        // Optional mobility converter via loader
        let mob_fn: MobilityFn = if use_mobility {
            // Example: simple closure using your index converter; adjust as needed:
            Some(|scan| scan as f32) // TODO: replace with real scan->mobility
        } else { None };

        let rows_rs = pick_im_peaks_on_imindex(
            &cm,
            raw_slice.as_deref(),
            rows, cols,
            min_prom, min_distance_scans, min_width_scans,
            mob_fn,
        );
        impeaks_to_py(py, rows_rs)
    }

    /// Adaptive IM-peak picking with fallback strategies.
    #[pyo3(signature = (
        im_matrix,                   // np.ndarray[f32] Fortran/C, shape (rows, scans)
        im_matrix_raw=None,          // optional raw (same shape)
        min_distance_scans=4,
        strategy="active_range",     // "none" | "full" | "active_range" | "apex_window"
        low_thresh=100.0,
        mid_thresh=200.0,
        sigma_lo=4.0,
        sigma_hi=2.0,
        min_prom_lo=25.0,
        min_prom_hi=50.0,
        min_width_lo=6,
        min_width_hi=3,
        abs_thr=5.0,
        rel_thr=0.03,
        pad=2,
        active_min_width=6,
        apex_half_width=15,
        use_mobility=false,
    ))]
    pub fn pick_im_peaks_on_matrix_adaptive(
        &self,
        im_matrix: PyReadonlyArray2<f32>,
        im_matrix_raw: Option<PyReadonlyArray2<f32>>,
        min_distance_scans: usize,
        strategy: &str,
        low_thresh: f32,
        mid_thresh: f32,
        sigma_lo: f32,
        sigma_hi: f32,
        min_prom_lo: f32,
        min_prom_hi: f32,
        min_width_lo: usize,
        min_width_hi: usize,
        abs_thr: f32,
        rel_thr: f32,
        pad: usize,
        active_min_width: usize,
        apex_half_width: usize,
        use_mobility: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyImPeak1D>>>> {
        use numpy::ndarray::ArrayView2;

        // 1) flatten to column-major
        let a: ArrayView2<f32> = im_matrix.as_array();
        let (rows, cols) = a.dim();
        let mut cm = vec![0.0f32; rows * cols];
        for s in 0..cols { for r in 0..rows { cm[s*rows + r] = a[(r,s)]; } }

        let raw_cm: Option<Vec<f32>> = im_matrix_raw.map(|dr| {
            let ar = dr.as_array();
            let mut v = vec![0.0f32; rows * cols];
            for s in 0..cols { for r in 0..rows { v[s*rows + r] = ar[(r,s)]; } }
            v
        });

        // 2) mobility mapping (stub here, wire your converter if desired)
        let _mob_fn = if use_mobility { Some(|scan: usize| scan as f32) } else { None };

        // 3) build policy + fallback
        let fallback = match strategy {
            "none" => FallbackMode::None,
            "full" => FallbackMode::FullWindow,
            "active_range" => FallbackMode::ActiveRange { abs_thr, rel_thr, pad, min_width: active_min_width },
            "apex_window" => FallbackMode::ApexWindow { half_width: apex_half_width },
            other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown strategy: {}", other))),
        };

        let policy = ImAdaptivePolicy {
            low_thresh, mid_thresh,
            sigma_lo, sigma_hi,
            min_prom_lo, min_prom_hi,
            min_width_lo, min_width_hi,
            fallback_mode: fallback,
        };

        // 4) run
        let rows_rs = pick_im_peaks_on_imindex_adaptive(
            &cm, raw_cm.as_deref(),
            rows, cols,
            min_distance_scans,
            None,
            policy,
        );
        impeaks_to_py_nested(py, rows_rs)
    }
}

fn peaks_to_py<'py>(py: Python<'py>, peaks: Vec<RtPeak1D>) -> PyResult<Vec<Py<PyRtPeak1D>>> {
    let mut out = Vec::with_capacity(peaks.len());
    for p in peaks {
        out.push(Py::new(py, PyRtPeak1D { inner: p })?);
    }
    Ok(out)
}

fn py_peaks_to_rust(
    py: Python<'_>,
    peaks_py: Vec<Py<PyRtPeak1D>>,
) -> Vec<RtPeak1D> {
    peaks_py.into_iter()
        .map(|p| p.borrow(py).inner.clone())
        .collect()
}

#[pymodule]
pub fn py_dia(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimsDatasetDIA>()?;
    m.add_class::<PyRtPeak1D>()?;
    m.add_class::<PyImPeak1D>()?;
    Ok(())
}