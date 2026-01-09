use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use numpy::{IntoPyArray, PyArray1};

use rustdf::cluster::cluster::ClusterResult1D;
use rustdf::cluster::feature::{
    AveragineLut,
    SimpleFeature,
    SimpleFeatureParams,
    SlimFeature,
    build_simple_features_from_clusters,
};
use rustdf::cluster::io as cio;

// NOTE: adjust this import path to wherever your PyClusterResult1D lives.
use crate::py_dia::PyClusterResult1D;

// ---------------------------------------------------------------
// PyAveragineLut – optional, but now used for cosine gating too
// ---------------------------------------------------------------

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyAveragineLut {
    pub inner: AveragineLut,
}

#[pymethods]
impl PyAveragineLut {
    /// Build an averagine lookup table.
    ///
    /// resolution: number of decimals (0..=6), interpreted like the old code.
    #[new]
    #[pyo3(signature = (
        mass_min,
        mass_max,
        step,
        z_min,
        z_max,
        k=6,
        resolution=3,
        num_threads=4
    ))]
    pub fn new(
        mass_min: f32,
        mass_max: f32,
        step: f32,
        z_min: u8,
        z_max: u8,
        k: usize,
        resolution: i32,
        num_threads: usize,
    ) -> Self {
        let inner = AveragineLut::build(
            mass_min,
            mass_max,
            step,
            z_min,
            z_max,
            k,
            resolution,
            num_threads,
        );
        PyAveragineLut { inner }
    }

    #[getter]
    pub fn masses(&self) -> Vec<f32> {
        self.inner.masses.clone()
    }

    #[getter]
    pub fn z_min(&self) -> u8 {
        self.inner.z_min
    }

    #[getter]
    pub fn z_max(&self) -> u8 {
        self.inner.z_max
    }

    #[getter]
    pub fn k(&self) -> usize {
        self.inner.k
    }

    /// Unit-length envelope (zero-padded to 8) for (neutral_mass, z).
    #[pyo3(signature = (neutral_mass, z))]
    pub fn lookup(&self, neutral_mass: f32, z: u8, py: Python<'_>) -> Py<PyArray1<f32>> {
        let v = self.inner.lookup(neutral_mass, z);
        v.to_vec().into_pyarray_bound(py).unbind()
    }

    fn __repr__(&self) -> String {
        format!(
            "AveragineLut(grid={}, z=[{}..{}], k={})",
            self.inner.masses.len(),
            self.inner.z_min,
            self.inner.z_max,
            self.inner.k
        )
    }
}

// ---------------------------------------------------------------
// PySimpleFeatureParams – controls the greedy builder
// ---------------------------------------------------------------

#[pyclass]
#[derive(Clone, Debug)]
pub struct PySimpleFeatureParams {
    pub inner: SimpleFeatureParams,
}

#[pymethods]
impl PySimpleFeatureParams {
    /// Create SimpleFeatureParams.
    ///
    /// All arguments have defaults matching Rust::Default,
    /// except min_cosine which defaults to 0.7 here (Python side).
    #[new]
    #[pyo3(signature = (
        z_min=1,
        z_max=5,
        iso_ppm_tol=10.0,
        iso_abs_da=0.003,
        min_members=2,
        max_members=5,
        min_raw_sum=0.0,
        min_mz=100.0,
        min_rt_overlap_frac=0.3,
        min_im_overlap_frac=0.3,
        min_cosine=0.7,
        w_spacing_penalty=1.0,
    ))]
    pub fn new(
        z_min: u8,
        z_max: u8,
        iso_ppm_tol: f32,
        iso_abs_da: f32,
        min_members: usize,
        max_members: usize,
        min_raw_sum: f32,
        min_mz: f32,
        min_rt_overlap_frac: f32,
        min_im_overlap_frac: f32,
        min_cosine: f32,
        w_spacing_penalty: f32,
    ) -> Self {
        PySimpleFeatureParams {
            inner: SimpleFeatureParams {
                z_min,
                z_max,
                iso_ppm_tol,
                iso_abs_da,
                min_members,
                max_members,
                min_raw_sum,
                min_mz,
                min_rt_overlap_frac,
                min_im_overlap_frac,
                min_cosine,
                w_spacing_penalty,
            },
        }
    }

    #[staticmethod]
    pub fn default() -> Self {
        PySimpleFeatureParams {
            inner: SimpleFeatureParams::default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SimpleFeatureParams(z=[{}..{}], iso_ppm_tol={}, iso_abs_da={}, \
             min_members={}, max_members={}, min_raw_sum={}, min_mz={}, \
             min_rt_overlap_frac={}, min_im_overlap_frac={}, min_cosine={},
             w_spacing_penalty={})",
            self.inner.z_min,
            self.inner.z_max,
            self.inner.iso_ppm_tol,
            self.inner.iso_abs_da,
            self.inner.min_members,
            self.inner.max_members,
            self.inner.min_raw_sum,
            self.inner.min_mz,
            self.inner.min_rt_overlap_frac,
            self.inner.min_im_overlap_frac,
            self.inner.min_cosine,
            self.inner.w_spacing_penalty,
        )
    }
}

// ---------------------------------------------------------------
// PySimpleFeature – thin wrapper around SimpleFeature
// ---------------------------------------------------------------

#[pyclass]
#[derive(Clone, Debug)]
pub struct PySimpleFeature {
    pub inner: SimpleFeature,
}

#[pymethods]
impl PySimpleFeature {
    #[getter]
    pub fn feature_id(&self) -> usize {
        self.inner.feature_id
    }

    #[getter]
    pub fn charge(&self) -> u8 {
        self.inner.charge
    }

    #[getter]
    pub fn mz_mono(&self) -> f32 {
        self.inner.mz_mono
    }

    #[getter]
    pub fn neutral_mass(&self) -> f32 {
        self.inner.neutral_mass
    }

    #[getter]
    pub fn rt_bounds(&self) -> (usize, usize) {
        self.inner.rt_bounds
    }

    #[getter]
    pub fn im_bounds(&self) -> (usize, usize) {
        self.inner.im_bounds
    }

    #[getter]
    pub fn mz_center(&self) -> f32 {
        self.inner.mz_center
    }

    #[getter]
    pub fn n_members(&self) -> usize {
        self.inner.n_members
    }

    /// Stable cluster IDs copied from ClusterResult1D::cluster_id (u64).
    #[getter]
    pub fn member_cluster_ids(&self) -> Vec<u64> {
        self.inner.member_cluster_ids.clone()
    }

    /// Indices into the *input slice* of clusters.
    #[getter]
    pub fn member_cluster_indices(&self) -> Vec<usize> {
        self.inner.member_cluster_indices.clone()
    }

    #[getter]
    pub fn raw_sum(&self) -> f32 {
        self.inner.raw_sum
    }

    /// Cosine similarity vs Averagine envelope (0.0 if no LUT used).
    #[getter]
    pub fn cos_averagine(&self) -> f32 {
        self.inner.cos_averagine
    }

    #[getter]
    pub fn precursors(&self) -> Vec<PyClusterResult1D> {
        self.inner
            .member_clusters
            .iter()
            .map(|c| {
                PyClusterResult1D { inner: c.clone() }
            })
            .collect()
    }

    #[getter]
    pub fn most_intense_precursor(&self) -> PyClusterResult1D {
        let most_intense = self.inner.member_clusters.iter().max_by(|a, b| a.raw_sum.partial_cmp(&b.raw_sum).unwrap()).unwrap();
        PyClusterResult1D { inner: most_intense.clone() }
    }

    fn __repr__(&self) -> String {
        format!(
            "SimpleFeature(id={}, z={}, mz_mono={:.4}, n_members={}, raw_sum={:.1}, cos_averagine={:.3})",
            self.inner.feature_id,
            self.inner.charge,
            self.inner.mz_mono,
            self.inner.n_members,
            self.inner.raw_sum,
            self.inner.cos_averagine,
        )
    }

    /// Convert to SlimFeature for memory-efficient storage.
    /// Pre-computes scoring parameters from the top (most intense) member cluster.
    pub fn to_slim(&self) -> PySlimFeature {
        PySlimFeature {
            inner: self.inner.to_slim(),
        }
    }
}

// ---------------------------------------------------------------
// PySlimFeature – memory-efficient feature representation
// ---------------------------------------------------------------

/// Minimal feature representation for memory-efficient storage (~80 bytes).
/// Contains pre-computed values from the top cluster for scoring and
/// pseudo-spectrum building, without storing full member cluster data.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PySlimFeature {
    pub inner: SlimFeature,
}

#[pymethods]
impl PySlimFeature {
    #[getter]
    pub fn feature_id(&self) -> usize {
        self.inner.feature_id
    }

    #[getter]
    pub fn charge(&self) -> u8 {
        self.inner.charge
    }

    #[getter]
    pub fn mz_mono(&self) -> f32 {
        self.inner.mz_mono
    }

    #[getter]
    pub fn neutral_mass(&self) -> f32 {
        self.inner.neutral_mass
    }

    #[getter]
    pub fn rt_mu(&self) -> f32 {
        self.inner.rt_mu
    }

    #[getter]
    pub fn rt_sigma(&self) -> f32 {
        self.inner.rt_sigma
    }

    #[getter]
    pub fn im_mu(&self) -> f32 {
        self.inner.im_mu
    }

    #[getter]
    pub fn im_sigma(&self) -> f32 {
        self.inner.im_sigma
    }

    #[getter]
    pub fn im_lo(&self) -> u16 {
        self.inner.im_window.0
    }

    #[getter]
    pub fn im_hi(&self) -> u16 {
        self.inner.im_window.1
    }

    #[getter]
    pub fn raw_sum(&self) -> f32 {
        self.inner.raw_sum
    }

    #[getter]
    pub fn top_cluster_id(&self) -> u64 {
        self.inner.top_cluster_id
    }

    #[getter]
    pub fn rt_bounds(&self) -> (u16, u16) {
        self.inner.rt_bounds
    }

    #[getter]
    pub fn im_bounds(&self) -> (u16, u16) {
        self.inner.im_bounds
    }

    #[getter]
    pub fn n_members(&self) -> u8 {
        self.inner.n_members
    }

    #[getter]
    pub fn member_cluster_ids(&self) -> Vec<u64> {
        self.inner.member_cluster_ids.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SlimFeature(id={}, z={}, mz_mono={:.4}, n_members={}, raw_sum={:.1})",
            self.inner.feature_id,
            self.inner.charge,
            self.inner.mz_mono,
            self.inner.n_members,
            self.inner.raw_sum,
        )
    }
}

// ---------------------------------------------------------------
// Top-level API: build_simple_features_from_clusters_py
// ---------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (clusters, params, lut=None))]
pub fn build_simple_features_from_clusters_py(
    py: Python<'_>,
    clusters: Vec<Py<PyClusterResult1D>>,
    params: &PySimpleFeatureParams,
    lut: Option<Py<PyAveragineLut>>,
) -> PyResult<Vec<PySimpleFeature>> {
    // Pull Rust clusters out of Py wrappers
    let rust_clusters: Vec<ClusterResult1D> = clusters
        .into_iter()
        .map(|c| {
            // borrow GIL-local, clone inner ClusterResult1D
            c.borrow(py).inner.clone()
        })
        .collect();

    // Call Rust builder with or without LUT, making sure the borrow
    // on PyAveragineLut lives for the duration of the call.
    let feats: Vec<SimpleFeature> = if let Some(py_lut) = lut {
        let lut_borrow = py_lut.borrow(py);
        build_simple_features_from_clusters(
            &rust_clusters,
            &params.inner,
            Some(&lut_borrow.inner),
        )
    } else {
        build_simple_features_from_clusters(&rust_clusters, &params.inner, None)
    };

    let py_feats: Vec<PySimpleFeature> = feats
        .into_iter()
        .map(|f| PySimpleFeature { inner: f })
        .collect();

    Ok(py_feats)
}

#[pyfunction]
pub fn save_features_bin(
    path: &str,
    features: Vec<Py<PySimpleFeature>>,
) -> PyResult<()> {
    let rust_features: Vec<SimpleFeature> = features
        .into_iter()
        .map(|f| {
            Python::with_gil(|py| {
                f.borrow(py).inner.clone()
            })
        })
        .collect();

    cio::save_feature_bincode(path, &rust_features, true)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
pub fn load_features_bin(py: Python<'_>, path: &str) -> PyResult<Vec<Py<PySimpleFeature>>> {
    let features = cio::load_feature_bincode(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    features.into_iter()
        .map(|f| Py::new(py, PySimpleFeature { inner: f }))
        .collect()
}


#[pymodule]
pub fn py_feature(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAveragineLut>()?;
    m.add_class::<PySimpleFeatureParams>()?;
    m.add_class::<PySimpleFeature>()?;
    m.add_class::<PySlimFeature>()?;

    m.add_function(wrap_pyfunction!(build_simple_features_from_clusters_py, m)?)?;
    m.add_function(wrap_pyfunction!(save_features_bin, m)?)?;
    m.add_function(wrap_pyfunction!(load_features_bin, m)?)?;

    // Silence unused variable warning for `py` if you don't use it directly
    let _ = py;

    Ok(())
}