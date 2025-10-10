use pyo3::{pymodule, Bound, PyResult, Python, pyclass, pymethods, Py, pyfunction, wrap_pyfunction};
use rustdf::cluster::cluster_eval::{AttachOptions, ClusterFit1D, ClusterResult, ClusterSpec, EvalOptions, CapAnchor};
use pyo3::prelude::{PyModule, PyModuleMethods};
use crate::py_dia::{PyImPeak1D, PyRtPeak1D};
use rayon::prelude::*;
use rustdf::cluster::io as cio;

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
        im_min_width=1,
        max_rt_span_frames=100,
        max_im_span_scans=100,
        ms_level=0,
        window_group_hint=None
    ))]
    fn new(
        attach: PyAttachOptions,
        refine_mz_once: bool,
        refine_k_sigma: f32,
        im_k_sigma: Option<f32>,
        im_min_width: usize,
        max_rt_span_frames: usize,
        max_im_span_scans: usize,
        ms_level: u8,
        window_group_hint: Option<u32>,
    ) -> Self {
        Self { inner: EvalOptions {
            attach: attach.inner,
            refine_mz_once,
            refine_k_sigma,
            im_k_sigma,
            im_min_width,
            max_rt_span_frames: Some(max_rt_span_frames),
            max_im_span_scans: Some( max_im_span_scans),
            cap_anchor: CapAnchor::RequestedMid,
            ms_level,
            window_group_hint,
        }}
    }

    #[getter] fn refine_mz_once(&self) -> bool { self.inner.refine_mz_once }
    #[getter] fn refine_k_sigma(&self) -> f32 { self.inner.refine_k_sigma }
    #[getter] fn im_k_sigma(&self) -> Option<f32> { self.inner.im_k_sigma }
    #[getter] fn im_min_width(&self) -> usize { self.inner.im_min_width }
    #[getter] fn max_rt_span_frames(&self) -> Option<usize> { self.inner.max_rt_span_frames }
    #[getter] fn max_im_span_scans(&self) -> Option<usize> {
        self.inner.max_im_span_scans
    }

    #[getter] fn ms_level(&self) -> u8 { self.inner.ms_level }
    #[getter] fn window_group_hint(&self) -> Option<u32> { self.inner.window_group_hint }
    #[getter] fn cap_anchor(&self) -> String {
        match self.inner.cap_anchor {
            CapAnchor::RequestedMid => "RequestedMid".to_string(),
        }
    }
    #[getter] fn attach(&self) -> PyAttachOptions {
        PyAttachOptions { inner: self.inner.attach.clone() }
    }

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
    #[getter] fn raw_points(&self) -> Option<PyRawPoints> {
        self.inner.raw_points.as_ref().map(|rp| PyRawPoints { inner: rp.clone() })
    }
    #[getter] fn ms_level(&self) -> u8 {
        self.inner.ms_level
    }

    #[getter] fn window_group(&self) -> Option<u32> {
        self.inner.window_group
    }
    #[getter]
    fn window_groups_covering_mz(&self) -> Option<Vec<u32>> {
        self.inner.window_groups_covering_mz.clone()
    }

    fn __repr__(&self) -> String {
        let n_pts = self.inner.raw_points.as_ref().map(|p| p.mz.len()).unwrap_or(0);
        format!(
            "ClusterResult#{}(rt=[{},{}], im=[{},{}], mz=[{:.5},{:.5}] Da,\
             rt_peak_id={}, im_peak_id={}, mz_hint={:.5}, points={}, ms_level={}, window_group={:?},
                window_groups_covering_mz={:?})",
            self.inner.id,
            self.inner.rt_window.0, self.inner.rt_window.1,
            self.inner.im_window.0, self.inner.im_window.1,
            self.inner.mz_window_da.0, self.inner.mz_window_da.1,
            self.inner.rt_peak_id,
            self.inner.im_peak_id,
            self.inner.mz_center_hint,
            n_pts,
            self.inner.ms_level,
            self.inner.window_group,
            self.inner.window_groups_covering_mz
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyLinkCandidate { pub inner: rustdf::cluster::cluster_eval::LinkCandidate }
#[pymethods]
impl PyLinkCandidate {
    #[getter] fn ms1_idx(&self) -> usize { self.inner.ms1_idx }
    #[getter] fn ms2_idx(&self) -> usize { self.inner.ms2_idx }
    #[getter] fn score(&self) -> f32 { self.inner.score }
    #[getter] fn group(&self) -> u32 { self.inner.group }
    fn __repr__(&self) -> String {
        format!("LinkCandidate(ms1_idx={}, ms2_idx={}, score={:.5})",
            self.inner.ms1_idx, self.inner.ms2_idx, self.inner.score)
    }
}

#[pyfunction]
pub fn build_precursor_fragment_annotation(
    py: Python<'_>,
    ms1: Vec<Py<PyClusterResult>>,
    ms2: Vec<Py<PyClusterResult>>,
    candidates: Vec<Py<PyLinkCandidate>>,
    min_score: f32,
) -> PyResult<Vec<(Py<PyClusterResult>, Vec<Py<PyClusterResult>>)>> {
    let rust_ms1: Vec<ClusterResult> = ms1
        .into_iter()
        .map(|c| c.borrow(py).inner.clone())
        .collect();
    let rust_ms2: Vec<ClusterResult> = ms2
        .into_iter()
        .map(|c| c.borrow(py).inner.clone())
        .collect();
    let rust_candidates: Vec<rustdf::cluster::cluster_eval::LinkCandidate> = candidates
        .into_iter()
        .map(|c| c.borrow(py).inner.clone())
        .collect();
    let annotated = rustdf::cluster::matching::build_precursor_fragment_annotation(
        &rust_ms1,
        &rust_ms2,
        &rust_candidates,
        min_score,
    );
    let mut out: Vec<(Py<PyClusterResult>, Vec<Py<PyClusterResult>>)> = Vec::with_capacity(annotated.len());
    for (prec, frags) in annotated {
        let prec_py = Py::new(py, PyClusterResult { inner: prec })?;
        let mut frags_py: Vec<Py<PyClusterResult>> = Vec::with_capacity(frags.len());
        for f in frags {
            frags_py.push(Py::new(py, PyClusterResult { inner: f })?);
        }
        out.push((prec_py, frags_py));
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (rt_peaks, im_rows, im_scans=None, mz_ppm_window=15.0, extra_rt_pad=0, extra_im_pad=0, mz_hist_bins=64))]
pub fn make_cluster_specs_from_peaks(
    py: Python<'_>,
    rt_peaks: Vec<Py<PyRtPeak1D>>,
    im_rows: Vec<Vec<Py<PyImPeak1D>>>,   // same row order as rt_peaks
    im_scans: Option<Vec<u32>>,          // <-- NEW: IM axis (absolute scan numbers)
    mz_ppm_window: f32,
    extra_rt_pad: usize,
    extra_im_pad: usize,
    mz_hist_bins: usize,
) -> PyResult<Vec<Py<PyClusterSpec>>> {

    #[derive(Clone, Copy)]
    struct RtRowSnap { rt_l: usize, rt_r: usize, mz_center: f32 }

    #[derive(Clone, Copy)]
    struct ImSnap { im_l: usize, im_r: usize }

    // If provided, convert to usize and we’ll use it to map indices->absolute scans
    let im_scans_abs: Option<Vec<usize>> = im_scans.map(|v| v.into_iter().map(|x| x as usize).collect());

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

            // indices within the IM row
            let i_l = im.left().saturating_sub(extra_im_pad);
            let i_r = im.right().saturating_add(extra_im_pad);

            // map to absolute scan numbers if axis available; else keep indices (MS1)
            let (im_l, im_r) = if let Some(ref axis) = im_scans_abs {
                let il = i_l.min(axis.len().saturating_sub(1));
                let ir = i_r.min(axis.len().saturating_sub(1));
                (axis[il], axis[ir])
            } else {
                (i_l, i_r)
            };

            row_vec.push(ImSnap { im_l, im_r });
        }
        im_rows_snap.push(row_vec);
    }

    let specs: Vec<ClusterSpec> = py.allow_threads(|| {
        rt_rows.par_iter().enumerate().flat_map_iter(|(row_idx, rt)| {
            im_rows_snap[row_idx].iter().map(move |im| ClusterSpec {
                rt_left: rt.rt_l,
                rt_right: rt.rt_r,
                im_left: im.im_l,      // now absolute if axis provided
                im_right: im.im_r,     // now absolute if axis provided
                mz_center_hint: rt.mz_center,
                mz_ppm_window,
                extra_rt_pad: 0,
                extra_im_pad: 0,
                mz_hist_bins,
                mz_window_da_override: None,
            })
        }).collect()
    });

    let mut specs_py = Vec::with_capacity(specs.len());
    for s in specs {
        specs_py.push(Py::new(py, PyClusterSpec { inner: s })?);
    }
    Ok(specs_py)
}

#[pyfunction]
#[pyo3(signature = (path, clusters, strip_points=false, strip_axes=false, compressed=false))]
#[allow(unused_variables)]
fn save_clusters_json(
    path: &str,
    clusters: Vec<Py<PyClusterResult>>,
    strip_points: bool,
    strip_axes: bool,
    compressed: bool,   // kept for parity; ignored for JSON
) -> PyResult<()> {
    let mut rust_clusters: Vec<ClusterResult> = clusters
        .into_iter()
        .map(|c| Python::with_gil(|py| c.borrow(py).inner.clone()))
        .collect();
    if strip_points || strip_axes {
        rust_clusters = rustdf::cluster::io::strip_heavy(rust_clusters, !strip_points, !strip_axes);
    }
    cio::save_json(path, &rust_clusters)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
fn load_clusters_json(py: Python<'_>, path: &str) -> PyResult<Vec<Py<PyClusterResult>>> {
    let clusters = cio::load_json(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    clusters.into_iter()
        .map(|c| Py::new(py, PyClusterResult { inner: c }))
        .collect()
}
#[pyfunction]
#[pyo3(signature = (path, clusters, compress=true, strip_points=false, strip_axes=false))]
fn save_clusters_bin(
    path: &str,
    clusters: Vec<Py<PyClusterResult>>,
    compress: bool,
    strip_points: bool,
    strip_axes: bool,
) -> PyResult<()> {
    let mut rust_clusters: Vec<ClusterResult> = clusters
        .into_iter()
        .map(|c| Python::with_gil(|py| c.borrow(py).inner.clone()))
        .collect();
    if strip_points || strip_axes {
        rust_clusters = rustdf::cluster::io::strip_heavy(rust_clusters, !strip_points, !strip_axes);
    }
    cio::save_bincode(path, &rust_clusters, compress)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
fn load_clusters_bin(py: Python<'_>, path: &str) -> PyResult<Vec<Py<PyClusterResult>>> {
    let clusters = cio::load_bincode(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    clusters.into_iter()
        .map(|c| Py::new(py, PyClusterResult { inner: c }))
        .collect()
}

/*
/// Link MS2→MS1 with group compatibility and co-elution.
/// Returns sorted candidates (best score first).
pub fn link_ms2_to_ms1(
    ms1: &[ClusterResult],         // ms_level==1
    ms2: &[ClusterResult],         // ms_level==2, with window_group = Some(g)
    min_rt_jaccard: f32,           // e.g. 0.1–0.2
    max_rt_apex_sec: f32,          // e.g. 5.0–10.0
    max_im_apex_scans: Option<f32>,// e.g. Some(5.0) or None to ignore
) -> Vec<LinkCandidate> {
 */

#[pyfunction]
#[pyo3(signature = (ms1, ms2, min_rt_jaccard=0.1, max_rt_apex_sec=5.0, max_im_apex_scans=None))]
pub fn link_ms2_to_ms1(
    py: Python<'_>,
    ms1: Vec<Py<PyClusterResult>>,
    ms2: Vec<Py<PyClusterResult>>,
    min_rt_jaccard: f32,
    max_rt_apex_sec: f32,
    max_im_apex_scans: Option<f32>,
) -> PyResult<Vec<Py<PyLinkCandidate>>> {
    let rust_ms1: Vec<ClusterResult> = ms1
        .into_iter()
        .map(|c| c.borrow(py).inner.clone())
        .collect();
    let rust_ms2: Vec<ClusterResult> = ms2
        .into_iter()
        .map(|c| c.borrow(py).inner.clone())
        .collect();
    let links = rustdf::cluster::cluster_eval::link_ms2_to_ms1(
        &rust_ms1,
        &rust_ms2,
        min_rt_jaccard,
        max_rt_apex_sec,
        max_im_apex_scans,
    );
    let mut links_py: Vec<Py<PyLinkCandidate>> = Vec::with_capacity(links.len());
    for l in links {
        links_py.push(Py::new(py, PyLinkCandidate { inner: l })?);
    }
    Ok(links_py)
}

#[pymodule]
pub fn py_cluster(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClusterSpec>()?;
    m.add_class::<PyAttachOptions>()?;
    m.add_class::<PyEvalOptions>()?;
    m.add_class::<PyClusterFit1D>()?;
    m.add_class::<PyClusterResult>()?;
    m.add_class::<PyRawPoints>()?;
    m.add_class::<PyLinkCandidate>()?;
    m.add_function(wrap_pyfunction!(make_cluster_specs_from_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(save_clusters_json, m)?)?;
    m.add_function(wrap_pyfunction!(load_clusters_json, m)?)?;
    m.add_function(wrap_pyfunction!(save_clusters_bin, m)?)?;
    m.add_function(wrap_pyfunction!(load_clusters_bin, m)?)?;
    m.add_function(wrap_pyfunction!(link_ms2_to_ms1, m)?)?;
    m.add_function(wrap_pyfunction!(build_precursor_fragment_annotation, m)?)?;
    Ok(())
}