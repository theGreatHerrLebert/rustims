use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, Bound, pyfunction, wrap_pyfunction};
use rustdf::cluster::feature::{Feature, FeatureBuildParams};
use rustdf::cluster::feature::{group_clusters_into_envelopes, group_clusters_into_envelopes_global, AveragineLut, Envelope, GroupingOutput, GroupingParams};
use rustdf::cluster::cluster_eval::ClusterResult;
use crate::py_cluster::PyClusterResult;

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyFeatureBuildParams { pub inner: FeatureBuildParams }

#[pymethods]
impl PyFeatureBuildParams {
    #[new]
    #[pyo3(signature = (ppm_narrow, k_max, min_cosine, min_members, max_points_per_slice, min_hist_conf=0.7, allow_unknown_charge=false))]
    pub fn new(
        ppm_narrow: f32,
        k_max: usize,
        min_cosine: f32,
        min_members: usize,
        max_points_per_slice: usize,
        min_hist_conf: f32,
        allow_unknown_charge: bool,
    ) -> Self {
        Self { inner: FeatureBuildParams {
            ppm_narrow,
            k_max,
            min_cosine,
            min_members,
            max_points_per_slice,
            min_hist_conf,
            allow_unknown_charge,
        }}
    }

    // getters
    #[getter] fn ppm_narrow(&self)      -> f32 { self.inner.ppm_narrow }
    #[getter] fn k_max(&self)           -> usize { self.inner.k_max }
    #[getter] fn min_cosine(&self)      -> f32 { self.inner.min_cosine }
    #[getter] fn min_members(&self)     -> usize { self.inner.min_members }
    #[getter] fn max_points_per_slice(&self) -> usize { self.inner.max_points_per_slice }

    fn __repr__(&self) -> String {
        format!(
            "FeatureBuildParams(ppm_narrow={}, k_max={}, min_cosine={}, min_members={}, max_points_per_slice={})",
            self.inner.ppm_narrow, self.inner.k_max,
            self.inner.min_cosine, self.inner.min_members, self.inner.max_points_per_slice
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyGroupingParams { pub inner: GroupingParams }

#[pymethods]
impl PyGroupingParams {
    #[new]
    #[pyo3(signature = (rt_pad_overlap, im_pad_overlap, mz_ppm_tol, iso_ppm_tol, z_min, z_max, iso_abs_da=0.05))]
    fn new(
        rt_pad_overlap: usize,
        im_pad_overlap: usize,
        mz_ppm_tol: f32,
        iso_ppm_tol: f32,
        z_min: u8,
        z_max: u8,
        iso_abs_da: f32,
    ) -> Self {
        Self { inner: GroupingParams {
            rt_pad_overlap,
            im_pad_overlap,
            mz_ppm_tol,
            iso_ppm_tol,
            z_min,
            z_max,
            iso_abs_da,
        }}
    }

    // getters
    #[getter] fn rt_pad_overlap(&self) -> usize { self.inner.rt_pad_overlap }
    #[getter] fn im_pad_overlap(&self) -> usize { self.inner.im_pad_overlap }
    #[getter] fn mz_ppm_tol(&self) -> f32 { self.inner.mz_ppm_tol }
    #[getter] fn iso_ppm_tol(&self) -> f32 { self.inner.iso_ppm_tol }
    #[getter] fn z_min(&self) -> u8 { self.inner.z_min }
    #[getter] fn z_max(&self) -> u8 { self.inner.z_max }

    fn __repr__(&self) -> String {
        format!(
            "GroupingParams(rt_pad_overlap={}, im_pad_overlap={}, mz_ppm_tol={}, iso_ppm_tol={}, z_min={}, z_max={})",
            self.inner.rt_pad_overlap, self.inner.im_pad_overlap,
            self.inner.mz_ppm_tol, self.inner.iso_ppm_tol,
            self.inner.z_min, self.inner.z_max
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyEnvelope { pub inner: Envelope }

#[pymethods]
impl PyEnvelope {
    #[getter] fn id(&self) -> usize { self.inner.id }
    #[getter] fn cluster_ids(&self) -> Vec<usize> { self.inner.cluster_ids.clone() }
    #[getter] fn rt_bounds(&self) -> (usize, usize) { self.inner.rt_bounds }
    #[getter] fn im_bounds(&self) -> (usize, usize) { self.inner.im_bounds }
    #[getter] fn mz_center(&self) -> f32 { self.inner.mz_center }
    #[getter] fn mz_span_da(&self) -> f32 { self.inner.mz_span_da }
    #[getter] fn charge_hint(&self) -> Option<u8> { self.inner.charge_hint }

    fn __repr__(&self) -> String {
        let (rt_l, rt_r) = self.inner.rt_bounds;
        let (im_l, im_r) = self.inner.im_bounds;
        format!(
            "Envelope#{}(members={}, rt=[{},{}], im=[{},{}], mz_center={:.5}, mz_span={:.5} Da)",
            self.inner.id, self.inner.cluster_ids.len(),
            rt_l, rt_r, im_l, im_r, self.inner.mz_center, self.inner.mz_span_da
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyGroupingOutput { pub inner: GroupingOutput }

#[pymethods]
impl PyGroupingOutput {
    #[getter]
    fn envelopes(&self, py: Python<'_>) -> PyResult<Vec<Py<PyEnvelope>>> {
        self.inner.envelopes
            .iter()
            .cloned()
            .map(|e| Py::new(py, PyEnvelope { inner: e }))
            .collect()
    }

    /// cluster_id -> Some(envelope_id) or None if unassigned
    #[getter]
    fn assignment(&self) -> Vec<Option<usize>> {
        self.inner.assignment.clone()
    }

    /// Provisional groups (list of cluster id lists), prior to final conflict resolution
    #[getter]
    fn provisional(&self) -> Vec<Vec<usize>> {
        self.inner.provisional.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "GroupingOutput(envelopes={}, assigned={}, provisional={})",
            self.inner.envelopes.len(),
            self.inner.assignment.iter().filter(|x| x.is_some()).count(),
            self.inner.provisional.len()
        )
    }
}

#[pyfunction]
pub fn group_clusters_into_envelopes_py(
    py: Python<'_>,
    clusters: Vec<Py<PyClusterResult>>,
    params: PyGroupingParams,
) -> PyResult<PyGroupingOutput> {
    // unwrap ClusterResult inners
    let mut rs: Vec<ClusterResult> = Vec::with_capacity(clusters.len());
    for c in clusters {
        let r = c.borrow(py);
        rs.push(r.inner.clone());
    }
    let out = group_clusters_into_envelopes(&rs, &params.inner);
    Ok(PyGroupingOutput { inner: out })
}

#[pyfunction]
pub fn group_clusters_into_envelopes_global_py(
    py: Python<'_>,
    clusters: Vec<Py<PyClusterResult>>,
    params: PyGroupingParams,
    averagine_lut: PyAveragineLut,
    k_max: usize,
) -> PyResult<PyGroupingOutput> {
    // unwrap ClusterResult inners
    let mut rs: Vec<ClusterResult> = Vec::with_capacity(clusters.len());
    for c in clusters {
        let r = c.borrow(py);
        rs.push(r.inner.clone());
    }
    let out = group_clusters_into_envelopes_global(&rs, &params.inner, &averagine_lut.inner, k_max);
    Ok(PyGroupingOutput { inner: out })
}


#[pyclass]
#[derive(Clone, Debug)]
pub struct PyFeature { pub inner: Feature }

#[pymethods]
impl PyFeature {
    #[getter] fn envelope_id(&self) -> usize { self.inner.envelope_id }          // if present
    #[getter] fn charge(&self)      -> u8    { self.inner.charge }               // e.g. z
    #[getter] fn mz_mono(&self)     -> f32   { self.inner.mz_mono }              // monoisotopic m/z     // if you store apex
    #[getter] fn rt_left(&self)     -> usize { self.inner.rt_bounds.0 }
    #[getter] fn rt_right(&self)    -> usize { self.inner.rt_bounds.1 }
    #[getter] fn im_left(&self)     -> usize { self.inner.im_bounds.0 }
    #[getter] fn im_right(&self)    -> usize { self.inner.im_bounds.1 }     // total/area
    #[getter] fn cosine(&self)      -> f32   { self.inner.cos_averagine }               // averagine score
    #[getter] fn n_members(&self)   -> usize { self.inner.n_members }
    #[getter] fn cluster_ids(&self) -> Vec<usize> { self.inner.member_cluster_ids.clone() }// clusters linked
    #[getter] fn repr_cluster_id(&self)   -> usize   { self.inner.repr_cluster_id }            // sum of raw intensities

    fn __repr__(&self) -> String {
        format!("Feature(z={}, mz_mono={:.6}, rt=({},{}) im=({},{}) cos={:.3}), clusters={})",
                self.charge(), self.mz_mono(),
                self.rt_left(), self.rt_right(),
                self.im_left(), self.im_right(),
                self.cosine(), self.n_members()
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyAveragineLut {
    pub inner: AveragineLut,
}

#[pymethods]
impl PyAveragineLut {
    /// Build an averagine lookup table on a mass grid for a charge range.
    #[new]
    #[pyo3(signature = (mass_min, mass_max, step, z_min, z_max, k=6, resolution=3, num_threads=4))]
    fn new(
        mass_min: f32,
        mass_max: f32,
        step: f32,
        z_min: u8,
        z_max: u8,
        k: usize,
        resolution: i32,
        num_threads: usize,
    ) -> Self {
        let inner = AveragineLut::build(mass_min, mass_max, step, z_min, z_max, k, resolution, num_threads);
        PyAveragineLut { inner }
    }

    #[getter] fn masses(&self) -> Vec<f32> { self.inner.masses.clone() }
    #[getter] fn z_min(&self) -> u8 { self.inner.z_min }
    #[getter] fn z_max(&self) -> u8 { self.inner.z_max }
    #[getter] fn k(&self) -> usize { self.inner.k }

    /// Return the k-length normalized envelope (zero-padded to 8) for (mass, z).
    #[pyo3(signature = (neutral_mass, z))]
    fn lookup(&self, neutral_mass: f32, z: u8) -> Vec<f32> {
        self.inner.lookup(neutral_mass, z).to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "AveragineLut(masses={}, z=[{}..{}], k={})",
            self.inner.masses.len(), self.inner.z_min, self.inner.z_max, self.inner.k
        )
    }
}

#[pymodule]
pub fn py_feature(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFeature>()?;
    m.add_class::<PyAveragineLut>()?;
    m.add_class::<PyGroupingParams>()?;
    m.add_class::<PyEnvelope>()?;
    m.add_class::<PyGroupingOutput>()?;
    m.add_class::<PyFeatureBuildParams>()?;
    m.add_function(wrap_pyfunction!(group_clusters_into_envelopes_py, m)?)?;
    m.add_function(wrap_pyfunction!(group_clusters_into_envelopes_global_py, m)?)?;
    Ok(())
}