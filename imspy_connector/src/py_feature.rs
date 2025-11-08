use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1};

use rustdf::cluster::cluster::ClusterResult1D;
use rustdf::cluster::feature::{
    build_features_from_clusters, AveragineLut, Feature, FeatureBuildParams, GroupingParams,
};

// NOTE: adjust this import path to wherever your PyClusterResult1D lives.
use crate::py_dia::PyClusterResult1D;

#[pyclass]
#[derive(Clone)]
pub struct PyAveragineLut {
    pub inner: AveragineLut,
}

#[pymethods]
impl PyAveragineLut {
    /// Construct and build an Averagine LUT.
    ///
    /// Example:
    ///   lut = PyAveragineLut(200.0, 6000.0, 25.0, 1, 6, 6, 60000, 8)
    ///
    /// - mass_min/mass_max in Da (neutral masses)
    /// - step in Da (e.g. 25–50)
    /// - z_min/z_max charge bounds
    /// - k: number of isotopic peaks kept (<=8)
    /// - resolution: instrument-ish resolution parameter passed to generator
    /// - num_threads: parallelism for the generator
    #[new]
    #[pyo3(signature = (mass_min, mass_max, step, z_min, z_max, k, resolution, num_threads))]
    pub fn new(mass_min: f32,
               mass_max: f32,
               step: f32,
               z_min: u8,
               z_max: u8,
               k: usize,
               resolution: i32,
               num_threads: usize) -> Self {
        let inner = AveragineLut::build(mass_min, mass_max, step, z_min, z_max, k, resolution, num_threads);
        PyAveragineLut { inner }
    }

    /// Convenience constructor with sensible defaults for proteomics.
    /// Grid: 200–6000 Da, step 25 Da, z=1..6, k=6, resolution=60k, threads = num_cpus.
    #[staticmethod]
    pub fn default_grid() -> Self {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let inner = AveragineLut::build(200.0, 6000.0, 25.0, 1, 6, 6, 60_000, threads);
        PyAveragineLut { inner }
    }

    /// Explicit alias if you still want a named constructor.
    #[staticmethod]
    #[pyo3(signature = (mass_min, mass_max, step, z_min, z_max, k, resolution, num_threads))]
    pub fn build(mass_min: f32,
                 mass_max: f32,
                 step: f32,
                 z_min: u8,
                 z_max: u8,
                 k: usize,
                 resolution: i32,
                 num_threads: usize) -> Self {
        let inner = AveragineLut::build(mass_min, mass_max, step, z_min, z_max, k, resolution, num_threads);
        PyAveragineLut { inner }
    }

    #[getter]
    pub fn z_min(&self) -> u8 { self.inner.z_min }
    #[getter]
    pub fn z_max(&self) -> u8 { self.inner.z_max }
    #[getter]
    pub fn k(&self) -> usize { self.inner.k }

    /// Nearest-neighbor lookup; returns a unit-length 8-vector (zero-padded).
    pub fn lookup(&self, neutral_mass: f32, z: u8, py: Python<'_>) -> Py<PyArray1<f32>> {
        self.inner.lookup(neutral_mass, z).to_vec().into_pyarray_bound(py).unbind()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyGroupingParams {
    pub inner: GroupingParams,
}

#[pymethods]
impl PyGroupingParams {
    #[new]
    #[pyo3(signature = (rt_pad_overlap, im_pad_overlap, mz_ppm_tol, iso_ppm_tol, iso_abs_da, z_min, z_max))]
    pub fn new(rt_pad_overlap: usize,
               im_pad_overlap: usize,
               mz_ppm_tol: f32,
               iso_ppm_tol: f32,
               iso_abs_da: f32,
               z_min: u8,
               z_max: u8) -> Self {
        PyGroupingParams {
            inner: GroupingParams {
                rt_pad_overlap,
                im_pad_overlap,
                mz_ppm_tol,
                iso_ppm_tol,
                iso_abs_da,
                z_min,
                z_max,
            }
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyFeatureBuildParams {
    pub inner: FeatureBuildParams,
}

#[pymethods]
impl PyFeatureBuildParams {
    #[new]
    #[pyo3(signature = (
        k_max, min_members, min_cosine,
        w_spacing, w_coelute, w_monotonic, penalty_skip_one,
        steal_delta, require_lowest_is_mono
    ))]
    pub fn new(k_max: usize,
               min_members: usize,
               min_cosine: f32,
               w_spacing: f32,
               w_coelute: f32,
               w_monotonic: f32,
               penalty_skip_one: f32,
               steal_delta: f32,
               require_lowest_is_mono: bool) -> Self {
        PyFeatureBuildParams {
            inner: FeatureBuildParams {
                k_max,
                min_members,
                min_cosine,
                w_spacing,
                w_coelute,
                w_monotonic,
                penalty_skip_one,
                steal_delta,
                require_lowest_is_mono,
            }
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyFeature {
    pub inner: Feature,
}

#[pymethods]
impl PyFeature {
    #[getter]
    pub fn envelope_id(&self) -> usize { self.inner.envelope_id }
    #[getter]
    pub fn charge(&self) -> u8 { self.inner.charge }
    #[getter]
    pub fn mz_mono(&self) -> f32 { self.inner.mz_mono }
    #[getter]
    pub fn neutral_mass(&self) -> f32 { self.inner.neutral_mass }
    #[getter]
    pub fn rt_bounds(&self) -> (usize, usize) { self.inner.rt_bounds }
    #[getter]
    pub fn im_bounds(&self) -> (usize, usize) { self.inner.im_bounds }
    #[getter]
    pub fn mz_center(&self) -> f32 { self.inner.mz_center }
    #[getter]
    pub fn n_members(&self) -> usize { self.inner.n_members }

    #[getter]
    pub fn member_cluster_ids(&self) -> Vec<usize> {
        self.inner.member_cluster_ids.clone()
    }

    #[getter]
    pub fn iso_raw(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        self.inner.iso_raw.to_vec().into_pyarray_bound(py).unbind()
    }

    #[getter]
    pub fn iso_l2(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        self.inner.iso_l2.to_vec().into_pyarray_bound(py).unbind()
    }

    #[getter]
    pub fn cos_averagine(&self) -> f32 { self.inner.cos_averagine }
}

#[pyclass]
#[derive(Clone)]
pub struct PyGroupingOutput {
    pub envelopes: Vec<(usize, (usize, usize), (usize, usize), f32, Option<u8>)>, // (id, rt, im, mz_center, charge_hint)
    pub assignment: Vec<Option<usize>>,
}

#[pymethods]
impl PyGroupingOutput {
    #[getter]
    pub fn envelopes(&self) -> Vec<(usize, (usize,usize), (usize,usize), f32, Option<u8>)> {
        self.envelopes.clone()
    }
    #[getter]
    pub fn assignment(&self) -> Vec<Option<usize>> {
        self.assignment.clone()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyBuildResult {
    pub features: Vec<PyFeature>,
    pub grouping: PyGroupingOutput,
}

#[pymethods]
impl PyBuildResult {
    #[getter]
    pub fn features(&self) -> Vec<PyFeature> { self.features.clone() }

    #[getter]
    pub fn grouping(&self) -> PyGroupingOutput { self.grouping.clone() }
}

//
// ---------- Top-level API: build_features_from_clusters ----------
//

#[pyfunction]
#[pyo3(signature = (clusters, gp, fp, lut=None))]
pub fn build_features_from_clusters_py(
    py: Python<'_>,
    clusters: Vec<Py<PyClusterResult1D>>,
    gp: &PyGroupingParams,
    fp: &PyFeatureBuildParams,
    lut: Option<&PyAveragineLut>,
) -> PyResult<PyBuildResult> {
    // Pull Rust clusters out of Py wrappers
    let rust_clusters: Vec<ClusterResult1D> = clusters
        .into_iter()
        .map(|c| c.borrow(py).inner.clone())
        .collect();

    let lut_ref = lut.map(|x| &x.inner);

    let out = build_features_from_clusters(
        &rust_clusters,
        &gp.inner,
        &fp.inner,
        lut_ref,
    );

    // Wrap features
    let py_features: Vec<PyFeature> = out.features.into_iter()
        .map(|f| PyFeature { inner: f })
        .collect();

    // Flatten grouping for Python-friendly consumption
    let envs_flat: Vec<(usize, (usize,usize), (usize,usize), f32, Option<u8>)> =
        out.grouping.envelopes.iter()
            .map(|e| (e.id, e.rt_bounds, e.im_bounds, e.mz_center, e.charge_hint))
            .collect();

    let py_grouping = PyGroupingOutput {
        envelopes: envs_flat,
        assignment: out.grouping.assignment,
    };

    Ok(PyBuildResult {
        features: py_features,
        grouping: py_grouping,
    })
}

#[pymodule]
pub fn py_feature(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAveragineLut>()?;
    m.add_class::<PyGroupingParams>()?;
    m.add_class::<PyFeatureBuildParams>()?;
    m.add_class::<PyFeature>()?;
    m.add_class::<PyGroupingOutput>()?;
    m.add_class::<PyBuildResult>()?;

    m.add_function(wrap_pyfunction!(build_features_from_clusters_py, m)?)?;
    Ok(())
}