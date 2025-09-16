use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, Bound, pyfunction, wrap_pyfunction};
use rustdf::cluster::cluster_eval::{Feature};
use rustdf::cluster::feature::{group_clusters_into_envelopes, AveragineLut, Envelope, GroupingOutput, GroupingParams};
use rustdf::cluster::cluster_eval::ClusterResult;
use crate::py_cluster::PyClusterResult;

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

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyFeature {
    pub inner: Feature,
}

#[pymethods]
impl PyFeature {
    #[getter] fn rt_mu(&self) -> f32 { self.inner.rt_mu }
    #[getter] fn rt_sigma(&self) -> f32 { self.inner.rt_sigma }
    #[getter] fn im_mu(&self) -> f32 { self.inner.im_mu }
    #[getter] fn im_sigma(&self) -> f32 { self.inner.im_sigma }
    #[getter] fn mz_mono(&self) -> f32 { self.inner.mz_mono }
    #[getter] fn z(&self) -> u8 { self.inner.z }
    #[getter] fn iso_i(&self) -> Vec<f32> { self.inner.iso_i.to_vec() }
    #[getter] fn avg_score(&self) -> f32 { self.inner.avg_score }
    #[getter] fn z_conf(&self) -> f32 { self.inner.z_conf }
    #[getter] fn raw_sum(&self) -> f32 { self.inner.raw_sum }
    #[getter] fn fit_volume(&self) -> f32 { self.inner.fit_volume }
    #[getter] fn source_cluster_id(&self) -> usize { self.inner.source_cluster_id }

    fn __repr__(&self) -> String {
        format!(
            "Feature(rt=μ{:.2} σ{:.2}, im=μ{:.2} σ{:.2}, mz_mono={:.5}, z={}, avg={:.3}, raw_sum={:.1})",
            self.inner.rt_mu, self.inner.rt_sigma,
            self.inner.im_mu, self.inner.im_sigma,
            self.inner.mz_mono, self.inner.z, self.inner.avg_score, self.inner.raw_sum
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
    m.add_function(wrap_pyfunction!(group_clusters_into_envelopes_py, m)?)?;
    Ok(())
}