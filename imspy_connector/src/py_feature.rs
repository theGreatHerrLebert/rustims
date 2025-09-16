use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, Bound};
use rustdf::cluster::cluster_eval::{BuildOpts, Feature};
use rustdf::cluster::feature::{AveragineLut};


// --------------------- PyFeature ---------------------

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

// --------------------- PyBuildOpts ---------------------

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyBuildOpts {
    pub inner: BuildOpts,
}

#[pymethods]
impl PyBuildOpts {
    #[new]
    #[pyo3(signature = (
        ppm_narrow=10.0,
        k_max=6,
        min_raw_sum=0.0,
        num_threads=4,
        charge_hist_ppm_window=None,
        charge_hist_bins=None
    ))]
    fn new(
        ppm_narrow: f32,
        k_max: usize,
        min_raw_sum: f32,
        num_threads: usize,
        charge_hist_ppm_window: Option<f32>,
        charge_hist_bins: Option<usize>,
    ) -> Self {
        PyBuildOpts {
            inner: BuildOpts {
                ppm_narrow,
                k_max,
                min_raw_sum,
                num_threads,
                charge_hist_ppm_window,
                charge_hist_bins,
            }
        }
    }

    #[getter] fn ppm_narrow(&self) -> f32 { self.inner.ppm_narrow }
    #[getter] fn k_max(&self) -> usize { self.inner.k_max }
    #[getter] fn min_raw_sum(&self) -> f32 { self.inner.min_raw_sum }
    #[getter] fn num_threads(&self) -> usize { self.inner.num_threads }
    #[getter] fn charge_hist_ppm_window(&self) -> Option<f32> { self.inner.charge_hist_ppm_window }
    #[getter] fn charge_hist_bins(&self) -> Option<usize> { self.inner.charge_hist_bins }

    fn __repr__(&self) -> String {
        format!(
            "BuildOpts(ppm_narrow={:.1}, k_max={}, min_raw_sum={:.1}, threads={}, hist_ppm={:?}, hist_bins={:?})",
            self.inner.ppm_narrow, self.inner.k_max, self.inner.min_raw_sum,
            self.inner.num_threads, self.inner.charge_hist_ppm_window, self.inner.charge_hist_bins
        )
    }
}

// --------------------- PyAveragineLut ---------------------

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

// --------------------- Module registration ---------------------

#[pymodule]
pub fn py_feature(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFeature>()?;
    m.add_class::<PyBuildOpts>()?;
    m.add_class::<PyAveragineLut>()?;
    Ok(())
}