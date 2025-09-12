use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult, Python};
use pyo3::prelude::PyModule;

use numpy::{PyArray1, PyArray2};
use numpy::ndarray::{Array1, Array2};

use rustdf::cluster::cluster_eval::{
    ClusterSpec, ClusterPatch, Gaussian1D, Separable2DFit, ClusterQuality, ClusterResult,
};

/// -------- PyClusterSpec --------
#[pyclass]
pub struct PyClusterSpec {
    pub inner: ClusterSpec,
}

#[pymethods]
impl PyClusterSpec {
    #[new]
    #[pyo3(signature = (
        rt_row, rt_left, rt_right, scan_left, scan_right, mz_ppm, resolution
    ))]
    pub fn new(
        rt_row: usize,
        rt_left: usize,
        rt_right: usize,
        scan_left: usize,
        scan_right: usize,
        mz_ppm: f32,
        resolution: usize,
    ) -> Self {
        let inner = ClusterSpec {
            rt_row, rt_left, rt_right, scan_left, scan_right, mz_ppm, resolution
        };
        Self { inner }
    }

    // Read-only getters (Pythonic ergonomics)
    #[getter] pub fn rt_row(&self) -> usize { self.inner.rt_row }
    #[getter] pub fn rt_left(&self) -> usize { self.inner.rt_left }
    #[getter] pub fn rt_right(&self) -> usize { self.inner.rt_right }
    #[getter] pub fn scan_left(&self) -> usize { self.inner.scan_left }
    #[getter] pub fn scan_right(&self) -> usize { self.inner.scan_right }
    #[getter] pub fn mz_ppm(&self) -> f32 { self.inner.mz_ppm }
    #[getter] pub fn resolution(&self) -> usize { self.inner.resolution }
}

/// -------- PyGaussian1D --------
#[pyclass]
pub struct PyGaussian1D {
    pub inner: Gaussian1D,
}
#[pymethods]
impl PyGaussian1D {
    #[getter] pub fn mu(&self) -> f32 { self.inner.mu }
    #[getter] pub fn sigma(&self) -> f32 { self.inner.sigma }
    #[getter] pub fn fwhm(&self) -> f32 { self.inner.fwhm }
}

/// -------- PySeparable2DFit --------
#[pyclass]
pub struct PySeparable2DFit {
    pub inner: Separable2DFit,
}
#[pymethods]
impl PySeparable2DFit {
    #[getter] pub fn rt(&self) -> PyGaussian1D { PyGaussian1D { inner: self.inner.rt.clone() } }
    #[getter] pub fn im(&self) -> PyGaussian1D { PyGaussian1D { inner: self.inner.im.clone() } }
    #[getter] pub fn A(&self) -> f32 { self.inner.A }
    #[getter] pub fn B(&self) -> f32 { self.inner.B }
}

/// -------- PyClusterQuality --------
#[pyclass]
pub struct PyClusterQuality {
    pub inner: ClusterQuality,
}
#[pymethods]
impl PyClusterQuality {
    #[getter] pub fn r2(&self) -> f32 { self.inner.r2 }
    #[getter] pub fn mse(&self) -> f32 { self.inner.mse }
    #[getter] pub fn snr_local(&self) -> f32 { self.inner.snr_local }
    #[getter] pub fn edge_mass_frac(&self) -> f32 { self.inner.edge_mass_frac }
}

/// -------- PyClusterPatch --------
#[pyclass]
pub struct PyClusterPatch {
    pub inner: ClusterPatch,
}

#[pymethods]
impl PyClusterPatch {
    #[getter] pub fn rows(&self) -> usize { self.inner.rows }
    #[getter] pub fn cols(&self) -> usize { self.inner.cols }

    pub fn rt_frames(&self, py: Python<'_>) -> Py<PyArray1<u32>> {
        PyArray1::from_vec_bound(py, self.inner.rt_frames.clone()).unbind()
    }

    pub fn scans(&self, py: Python<'_>) -> Py<PyArray1<u16>> {
        PyArray1::from_vec_bound(py, self.inner.scans.clone()).unbind()
    }

    pub fn rt_trace(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec_bound(py, self.inner.rt_trace.clone()).unbind()
    }

    pub fn im_trace(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec_bound(py, self.inner.im_trace.clone()).unbind()
    }

    pub fn patch(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        use numpy::ndarray::Array2;
        let arr: Array2<f32> = Array2::from_shape_vec(
            (self.inner.rows, self.inner.cols),
            self.inner.patch.clone()
        ).expect("shape error in patch");
        PyArray2::from_owned_array_bound(py, arr).unbind()
    }

    #[getter] pub fn total_area(&self) -> f32 { self.inner.total_area }
    #[getter] pub fn apex_value(&self) -> f32 { self.inner.apex_value }

    #[getter]
    pub fn apex_pos(&self) -> (usize, usize) {
        self.inner.apex_pos
    }
}

/// -------- PyClusterResult --------
#[pyclass]
pub struct PyClusterResult {
    pub inner: ClusterResult,
}

#[pymethods]
impl PyClusterResult {
    #[getter] pub fn spec(&self) -> PyClusterSpec { PyClusterSpec { inner: self.inner.spec.clone() } }
    #[getter] pub fn patch(&self) -> PyClusterPatch { PyClusterPatch { inner: self.inner.patch.clone() } }
    #[getter] pub fn fit(&self) -> PySeparable2DFit { PySeparable2DFit { inner: self.inner.fit.clone() } }
    #[getter] pub fn quality(&self) -> PyClusterQuality { PyClusterQuality { inner: self.inner.q.clone() } }

    /// Convenience dict for Pandas
    pub fn to_dict(&self) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let q = &self.inner.q;
            let rt = &self.inner.fit.rt;
            let im = &self.inner.fit.im;
            let d = pyo3::types::PyDict::new_bound(py);
            d.set_item("rt_row", self.inner.spec.rt_row)?;
            d.set_item("rt_left", self.inner.spec.rt_left)?;
            d.set_item("rt_right", self.inner.spec.rt_right)?;
            d.set_item("scan_left", self.inner.spec.scan_left)?;
            d.set_item("scan_right", self.inner.spec.scan_right)?;
            d.set_item("mz_ppm", self.inner.spec.mz_ppm)?;
            d.set_item("resolution", self.inner.spec.resolution)?;
            d.set_item("rows", self.inner.patch.rows)?;
            d.set_item("cols", self.inner.patch.cols)?;
            d.set_item("total_area", self.inner.patch.total_area)?;
            d.set_item("apex_value", self.inner.patch.apex_value)?;
            d.set_item("r2", q.r2)?;
            d.set_item("mse", q.mse)?;
            d.set_item("snr_local", q.snr_local)?;
            d.set_item("edge_mass_frac", q.edge_mass_frac)?;
            d.set_item("rt_mu", rt.mu)?;
            d.set_item("rt_sigma", rt.sigma)?;
            d.set_item("rt_fwhm", rt.fwhm)?;
            d.set_item("im_mu", im.mu)?;
            d.set_item("im_sigma", im.sigma)?;
            d.set_item("im_fwhm", im.fwhm)?;
            Ok(d.into())
        })
    }
}

#[pymodule]
pub fn py_cluster(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClusterSpec>()?;
    m.add_class::<PyGaussian1D>()?;
    m.add_class::<PySeparable2DFit>()?;
    m.add_class::<PyClusterQuality>()?;
    m.add_class::<PyClusterPatch>()?;
    m.add_class::<PyClusterResult>()?;
    Ok(())
}