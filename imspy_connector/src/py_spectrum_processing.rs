use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

use mscore::timstof::spectrum_processing::{
    PreprocessedSpectrum, SpectrumProcessingConfig,
    deisotope_spectrum, filter_top_n, process_spectrum,
};

#[pyclass]
#[derive(Clone)]
pub struct PySpectrumProcessingConfig {
    pub inner: SpectrumProcessingConfig,
}

#[pymethods]
impl PySpectrumProcessingConfig {
    #[new]
    #[pyo3(signature = (take_top_n=150, deisotope=true, deisotope_tolerance_ppm=10.0, deisotope_min_mz_diff=0.0005))]
    pub fn new(
        take_top_n: usize,
        deisotope: bool,
        deisotope_tolerance_ppm: f64,
        deisotope_min_mz_diff: f64,
    ) -> Self {
        PySpectrumProcessingConfig {
            inner: SpectrumProcessingConfig {
                take_top_n,
                deisotope,
                deisotope_tolerance_ppm,
                deisotope_min_mz_diff,
            },
        }
    }

    #[getter]
    pub fn take_top_n(&self) -> usize {
        self.inner.take_top_n
    }

    #[getter]
    pub fn deisotope(&self) -> bool {
        self.inner.deisotope
    }

    #[getter]
    pub fn deisotope_tolerance_ppm(&self) -> f64 {
        self.inner.deisotope_tolerance_ppm
    }

    #[getter]
    pub fn deisotope_min_mz_diff(&self) -> f64 {
        self.inner.deisotope_min_mz_diff
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SpectrumProcessingConfig(take_top_n={}, deisotope={}, deisotope_tolerance_ppm={}, deisotope_min_mz_diff={})",
            self.inner.take_top_n, self.inner.deisotope, self.inner.deisotope_tolerance_ppm, self.inner.deisotope_min_mz_diff
        )
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyPreprocessedSpectrum {
    pub inner: PreprocessedSpectrum,
}

#[pymethods]
impl PyPreprocessedSpectrum {
    #[new]
    #[pyo3(signature = (spec_id, precursor_mz, precursor_charge, precursor_intensity, inverse_ion_mobility, collision_energy, scan_start_time, total_ion_current, mz, intensity, isolation_mz, isolation_width))]
    pub fn new(
        spec_id: String,
        precursor_mz: f64,
        precursor_charge: Option<i32>,
        precursor_intensity: f64,
        inverse_ion_mobility: f64,
        collision_energy: f64,
        scan_start_time: f64,
        total_ion_current: f64,
        mz: Vec<f32>,
        intensity: Vec<f32>,
        isolation_mz: f64,
        isolation_width: f64,
    ) -> Self {
        PyPreprocessedSpectrum {
            inner: PreprocessedSpectrum::new(
                spec_id,
                precursor_mz,
                precursor_charge,
                precursor_intensity,
                inverse_ion_mobility,
                collision_energy,
                scan_start_time,
                total_ion_current,
                mz,
                intensity,
                isolation_mz,
                isolation_width,
            ),
        }
    }

    #[getter]
    pub fn spec_id(&self) -> &str {
        &self.inner.spec_id
    }

    #[getter]
    pub fn precursor_mz(&self) -> f64 {
        self.inner.precursor_mz
    }

    #[getter]
    pub fn precursor_charge(&self) -> Option<i32> {
        self.inner.precursor_charge
    }

    #[getter]
    pub fn precursor_intensity(&self) -> f64 {
        self.inner.precursor_intensity
    }

    #[getter]
    pub fn inverse_ion_mobility(&self) -> f64 {
        self.inner.inverse_ion_mobility
    }

    #[getter]
    pub fn collision_energy(&self) -> f64 {
        self.inner.collision_energy
    }

    #[getter]
    pub fn scan_start_time(&self) -> f64 {
        self.inner.scan_start_time
    }

    #[getter]
    pub fn total_ion_current(&self) -> f64 {
        self.inner.total_ion_current
    }

    #[getter]
    pub fn mz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.mz.clone())
    }

    #[getter]
    pub fn intensity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.intensity.clone())
    }

    #[getter]
    pub fn isolation_mz(&self) -> f64 {
        self.inner.isolation_mz
    }

    #[getter]
    pub fn isolation_width(&self) -> f64 {
        self.inner.isolation_width
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PreprocessedSpectrum(spec_id='{}', precursor_mz={:.4}, charge={:?}, peaks={}, collision_energy={:.1})",
            self.inner.spec_id, self.inner.precursor_mz, self.inner.precursor_charge,
            self.inner.mz.len(), self.inner.collision_energy
        )
    }
}

impl PyPreprocessedSpectrum {
    pub fn from_inner(inner: PreprocessedSpectrum) -> Self {
        PyPreprocessedSpectrum { inner }
    }
}

/// Deisotope a spectrum by removing peaks that are likely isotopes.
#[pyfunction]
#[pyo3(signature = (mz, intensity, tolerance_ppm=10.0))]
pub fn py_deisotope_spectrum<'py>(
    py: Python<'py>,
    mz: PyReadonlyArray1<f64>,
    intensity: PyReadonlyArray1<f64>,
    tolerance_ppm: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let mz_slice = mz.as_slice().unwrap();
    let intensity_slice = intensity.as_slice().unwrap();

    let (filtered_mz, filtered_intensity) = deisotope_spectrum(mz_slice, intensity_slice, tolerance_ppm);

    (
        PyArray1::from_vec(py, filtered_mz),
        PyArray1::from_vec(py, filtered_intensity),
    )
}

/// Filter spectrum to keep only the top N most intense peaks.
#[pyfunction]
#[pyo3(signature = (mz, intensity, top_n))]
pub fn py_filter_top_n<'py>(
    py: Python<'py>,
    mz: PyReadonlyArray1<f64>,
    intensity: PyReadonlyArray1<f64>,
    top_n: usize,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let mz_slice = mz.as_slice().unwrap();
    let intensity_slice = intensity.as_slice().unwrap();

    let (filtered_mz, filtered_intensity) = filter_top_n(mz_slice, intensity_slice, top_n);

    (
        PyArray1::from_vec(py, filtered_mz),
        PyArray1::from_vec(py, filtered_intensity),
    )
}

/// Process a spectrum: flatten, optionally deisotope, filter top N peaks.
#[pyfunction]
#[pyo3(signature = (tof, mz, intensity, config))]
pub fn py_process_spectrum<'py>(
    py: Python<'py>,
    tof: PyReadonlyArray1<i32>,
    mz: PyReadonlyArray1<f64>,
    intensity: PyReadonlyArray1<f64>,
    config: &PySpectrumProcessingConfig,
) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>) {
    let tof_slice = tof.as_slice().unwrap();
    let mz_slice = mz.as_slice().unwrap();
    let intensity_slice = intensity.as_slice().unwrap();

    let (processed_mz, processed_intensity) = process_spectrum(
        tof_slice,
        mz_slice,
        intensity_slice,
        &config.inner,
    );

    (
        PyArray1::from_vec(py, processed_mz),
        PyArray1::from_vec(py, processed_intensity),
    )
}

#[pymodule]
pub fn py_spectrum_processing(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpectrumProcessingConfig>()?;
    m.add_class::<PyPreprocessedSpectrum>()?;
    m.add_function(wrap_pyfunction!(py_deisotope_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(py_filter_top_n, m)?)?;
    m.add_function(wrap_pyfunction!(py_process_spectrum, m)?)?;
    Ok(())
}
