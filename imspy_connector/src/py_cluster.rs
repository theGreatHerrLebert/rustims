use pyo3::{pymodule, Bound, PyResult, Python, pyclass, pymethods, Py, pyfunction, wrap_pyfunction};
use rustdf::cluster::cluster_eval::{AttachOptions, ClusterFit1D, ClusterResult, ClusterSpec, EvalOptions};
use pyo3::prelude::{PyModule, PyModuleMethods};
use crate::py_dia::{PyImPeak1D, PyRtPeak1D};
use rayon::prelude::*;

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct PyRawPoints { pub inner: rustdf::cluster::cluster_eval::RawPoints }

#[pymethods]
impl PyRawPoints {
    #[getter] fn mz(&self) -> Vec<f32> { self.inner.mz.clone() }
    #[getter] fn rt(&self) -> Vec<f32> { self.inner.rt.clone() }
    #[getter] fn im(&self) -> Vec<f32> { self.inner.im.clone() }
    #[getter] fn scan(&self) -> Vec<u32> { self.inner.scan.clone() }
    #[getter] fn intensity(&self) -> Vec<f32> { self.inner.intensity.clone() }
    #[getter] fn tof(&self) -> Vec<i32> { self.inner.tof.clone() }
    #[getter] fn frame(&self) -> Vec<u32> { self.inner.frame.clone() }

    fn __repr__(&self) -> String {
        let n = self.inner.mz.len();
        format!("RawPoints(n={})", n)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyClusterSpec { pub inner: ClusterSpec }

#[pymethods]
impl PyClusterSpec {
    #[new]
    #[pyo3(signature = (
        rt_left, rt_right,
        im_left, im_right,
        mz_center_hint, mz_ppm_window,
        extra_rt_pad=0, extra_im_pad=0, mz_hist_bins=64
    ))]
    fn new(
        rt_left: usize, rt_right: usize,
        im_left: usize, im_right: usize,
        mz_center_hint: f32, mz_ppm_window: f32,
        extra_rt_pad: usize, extra_im_pad: usize, mz_hist_bins: usize,
    ) -> Self {
        Self { inner: ClusterSpec {
            rt_left, rt_right, im_left, im_right,
            mz_center_hint, mz_ppm_window,
            extra_rt_pad, extra_im_pad, mz_hist_bins,
            // if you have this field in ClusterSpec:
            mz_window_da_override: None,
        }}
    }

    // --- getters used by your Python thin wrapper ---
    #[getter] fn rt_left(&self) -> usize { self.inner.rt_left }
    #[getter] fn rt_right(&self) -> usize { self.inner.rt_right }
    #[getter] fn im_left(&self) -> usize { self.inner.im_left }
    #[getter] fn im_right(&self) -> usize { self.inner.im_right }
    #[getter] fn mz_center_hint(&self) -> f32 { self.inner.mz_center_hint }
    #[getter] fn mz_ppm_window(&self) -> f32 { self.inner.mz_ppm_window }
    #[getter] fn extra_rt_pad(&self) -> usize { self.inner.extra_rt_pad }
    #[getter] fn extra_im_pad(&self) -> usize { self.inner.extra_im_pad }
    #[getter] fn mz_hist_bins(&self) -> usize { self.inner.mz_hist_bins }

    // (optional) expose DA override if you kept it in ClusterSpec
    #[getter]
    fn mz_window_da_override(&self) -> Option<(f32, f32)> {
        self.inner.mz_window_da_override
    }
    #[setter]
    fn set_mz_window_da_override(&mut self, val: Option<(f32, f32)>) {
        self.inner.mz_window_da_override = val;
    }

    fn __repr__(&self) -> String {
        format!(
            "ClusterSpec(rt=[{},{}], im=[{},{}], mz≈{:.5}±{}ppm, pads(rt={},im={}), mz_bins={})",
            self.inner.rt_left, self.inner.rt_right,
            self.inner.im_left, self.inner.im_right,
            self.inner.mz_center_hint, self.inner.mz_ppm_window,
            self.inner.extra_rt_pad, self.inner.extra_im_pad,
            self.inner.mz_hist_bins
        )
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct PyAttachOptions { pub inner: AttachOptions }

#[pymethods]
impl PyAttachOptions {
    #[new]
    #[pyo3(signature = (
        attach_frames=true,
        attach_scans=true,
        attach_mz_axis=true,
        attach_points=false,
        max_points=None
    ))]
    fn new(
        attach_frames: bool,
        attach_scans: bool,
        attach_mz_axis: bool,
        attach_points: bool,
        max_points: Option<usize>,
    ) -> Self {
        Self { inner: AttachOptions {
            attach_frames,
            attach_scans,
            attach_mz_axis,
            attach_points,
            max_points,
        }}
    }

    #[getter] fn attach_frames(&self) -> bool { self.inner.attach_frames }
    #[getter] fn attach_scans(&self) -> bool { self.inner.attach_scans }
    #[getter] fn attach_mz_axis(&self) -> bool { self.inner.attach_mz_axis }
    #[getter] fn attach_points(&self) -> bool { self.inner.attach_points }
    #[getter] fn max_points(&self) -> Option<usize> { self.inner.max_points }

    fn __repr__(&self) -> String {
        format!(
            "AttachOptions(frames={}, scans={}, mz_axis={}, points={}, max_points={:?})",
            self.inner.attach_frames,
            self.inner.attach_scans,
            self.inner.attach_mz_axis,
            self.inner.attach_points,
            self.inner.max_points
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyEvalOptions { pub inner: EvalOptions }

#[pymethods]
impl PyEvalOptions {
    #[new]
    #[pyo3(signature = (
        attach,
        refine_mz_once=false,
        refine_k_sigma=3.0,
        im_k_sigma=None,
        im_min_width=1
    ))]
    fn new(
        attach: PyAttachOptions,
        refine_mz_once: bool,
        refine_k_sigma: f32,
        im_k_sigma: Option<f32>,
        im_min_width: usize,
    ) -> Self {
        Self { inner: EvalOptions {
            attach: attach.inner,
            refine_mz_once,
            refine_k_sigma,
            im_k_sigma,
            im_min_width,
            min_num_points: None
        }}
    }

    #[getter] fn refine_mz_once(&self) -> bool { self.inner.refine_mz_once }
    #[getter] fn refine_k_sigma(&self) -> f32 { self.inner.refine_k_sigma }
    #[getter] fn im_k_sigma(&self) -> Option<f32> { self.inner.im_k_sigma }
    #[getter] fn im_min_width(&self) -> usize { self.inner.im_min_width }

    fn __repr__(&self) -> String {
        format!(
            "EvalOptions({}, refine_mz_once={}, k_sigma={}, im_k_sigma={:?}, im_min_width={})",
            PyAttachOptions { inner: self.inner.attach.clone() }.__repr__(),
            self.inner.refine_mz_once,
            self.inner.refine_k_sigma,
            self.inner.im_k_sigma,
            self.inner.im_min_width
        )
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct PyClusterFit1D { pub inner: ClusterFit1D }

#[pymethods]
impl PyClusterFit1D {
    #[getter] fn mu(&self) -> f32 { self.inner.mu }
    #[getter] fn sigma(&self) -> f32 { self.inner.sigma }
    #[getter] fn height(&self) -> f32 { self.inner.height }
    #[getter] fn baseline(&self) -> f32 { self.inner.baseline }
    #[getter] fn area(&self) -> f32 { self.inner.area }
    #[getter] fn r2(&self) -> f32 { self.inner.r2 }
    #[getter] fn n(&self) -> usize { self.inner.n }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyClusterResult { pub inner: ClusterResult }

#[pymethods]
impl PyClusterResult {
    #[getter] fn id(&self) -> usize { self.inner.id }
    #[getter] fn rt_window(&self) -> (usize, usize) { self.inner.rt_window }
    #[getter] fn im_window(&self) -> (usize, usize) { self.inner.im_window }
    #[getter] fn mz_window_da(&self) -> (f32, f32) { self.inner.mz_window_da }

    #[getter] fn rt_fit(&self) -> PyClusterFit1D { PyClusterFit1D { inner: self.inner.rt_fit.clone() } }
    #[getter] fn im_fit(&self) -> PyClusterFit1D { PyClusterFit1D { inner: self.inner.im_fit.clone() } }
    #[getter] fn mz_fit(&self) -> PyClusterFit1D { PyClusterFit1D { inner: self.inner.mz_fit.clone() } }

    #[getter] fn raw_sum(&self) -> f32 { self.inner.raw_sum }
    #[getter] fn fit_volume(&self) -> f32 { self.inner.fit_volume }

    #[getter] fn rt_peak_id(&self) -> usize { self.inner.rt_peak_id }
    #[getter] fn im_peak_id(&self) -> usize { self.inner.im_peak_id }
    #[getter] fn mz_center_hint(&self) -> f32 { self.inner.mz_center_hint }

    #[getter] fn frame_ids_used(&self) -> Vec<u32> { self.inner.frame_ids_used.clone() }

    #[getter] fn frames_axis(&self) -> Option<Vec<u32>> { self.inner.frames_axis.clone() }
    #[getter] fn scans_axis(&self) -> Option<Vec<usize>> { self.inner.scans_axis.clone() }
    #[getter] fn mz_axis(&self) -> Option<Vec<f32>> { self.inner.mz_axis.clone() }

    // NEW: expose raw points as a nested class, or None
    #[getter]
    fn raw_points(&self) -> Option<PyRawPoints> {
        self.inner.raw_points.as_ref().map(|rp| PyRawPoints { inner: rp.clone() })
    }

    fn __repr__(&self) -> String {
        let n_pts = self.inner.raw_points.as_ref().map(|p| p.mz.len()).unwrap_or(0);
        format!(
            "ClusterResult(id={}, rt=[{},{}], im=[{},{}], mz=[{:.5},{:.5}], points={})",
            self.inner.id,
            self.inner.rt_window.0, self.inner.rt_window.1,
            self.inner.im_window.0, self.inner.im_window.1,
            self.inner.mz_window_da.0, self.inner.mz_window_da.1,
            n_pts
        )
    }
}

#[pyfunction]
#[pyo3(signature = (rt_peaks, im_rows, mz_ppm_window=15.0, extra_rt_pad=0, extra_im_pad=0, mz_hist_bins=64))]
pub fn make_cluster_specs_from_peaks(
    py: Python<'_>,
    rt_peaks: Vec<Py<PyRtPeak1D>>,
    im_rows: Vec<Vec<Py<PyImPeak1D>>>, // same row order as rt_peaks
    mz_ppm_window: f32,
    extra_rt_pad: usize,
    extra_im_pad: usize,
    mz_hist_bins: usize,
) -> PyResult<Vec<Py<PyClusterSpec>>> {
    // 1) Snapshot the few needed values from Python objects under the GIL
    //    to Rust-owned plain types we can use without the GIL.
    #[derive(Clone, Copy)]
    struct RtRowSnap {
        rt_l: usize,
        rt_r: usize,
        mz_center: f32,
    }
    #[derive(Clone, Copy)]
    struct ImSnap {
        im_l: usize,
        im_r: usize,
    }

    let mut rt_rows: Vec<RtRowSnap> = Vec::with_capacity(rt_peaks.len());
    let mut im_rows_snap: Vec<Vec<ImSnap>> = Vec::with_capacity(im_rows.len());

    for (row_idx, rt_p) in rt_peaks.iter().enumerate() {
        let rt = rt_p.borrow(py);
        let mz_center = rt.mz_center();
        let rt_l = rt.left_padded().saturating_sub(extra_rt_pad);
        let rt_r = rt.right_padded().saturating_add(extra_rt_pad);
        rt_rows.push(RtRowSnap { rt_l, rt_r, mz_center });

        let mut row_vec: Vec<ImSnap> = Vec::with_capacity(im_rows[row_idx].len());
        for im_p in &im_rows[row_idx] {
            let im = im_p.borrow(py);
            let im_l = im.left().saturating_sub(extra_im_pad);
            let im_r = im.right().saturating_add(extra_im_pad);
            row_vec.push(ImSnap { im_l, im_r });
        }
        im_rows_snap.push(row_vec);
    }

    // 2) Release GIL and build ClusterSpec in parallel
    let specs: Vec<ClusterSpec> = py.allow_threads(|| {
        rt_rows
            .par_iter()
            .enumerate()
            .flat_map_iter(|(row_idx, rt)| {
                im_rows_snap[row_idx]
                    .iter()
                    .map(move |im| ClusterSpec {
                        rt_left: rt.rt_l,
                        rt_right: rt.rt_r,
                        im_left: im.im_l,
                        im_right: im.im_r,
                        mz_center_hint: rt.mz_center,
                        mz_ppm_window,
                        extra_rt_pad: 0,
                        extra_im_pad: 0,
                        mz_hist_bins,
                        mz_window_da_override: None,
                    })
            })
            .collect()
    });

    // 3) Reacquire GIL only to wrap specs into Python objects
    let mut specs_py = Vec::with_capacity(specs.len());
    for s in specs {
        specs_py.push(Py::new(py, PyClusterSpec { inner: s })?);
    }
    Ok(specs_py)
}

#[pymodule]
pub fn py_cluster(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClusterSpec>()?;
    m.add_class::<PyAttachOptions>()?;
    m.add_class::<PyEvalOptions>()?;
    m.add_class::<PyClusterFit1D>()?;
    m.add_class::<PyClusterResult>()?;
    m.add_class::<PyRawPoints>()?;
    m.add_function(wrap_pyfunction!(make_cluster_specs_from_peaks, m)?)?;
    Ok(())
}