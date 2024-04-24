use pyo3::prelude::*;
use numpy::{PyArray1, IntoPyArray};
use rustms::ms::spectrum::{MzSpectrum};
#[pyclass]
#[derive(Clone)]
pub struct PyMzSpectrum {
    pub inner: MzSpectrum,
}

#[pymethods]

impl PyMzSpectrum {
    #[new]
    pub unsafe fn new(mz: &PyArray1<f64>, intensity: &PyArray1<f64>) -> PyResult<Self> {
        // convert mz and intensity to slices
        let mz = mz.as_slice()?;
        let intensity = intensity.as_slice()?;

        // sort mz and intensity by mz
        let mut mz_intensity: Vec<(f64, f64)> = mz.iter().zip(intensity.iter()).map(|(m, i)| (*m, *i)).collect();
        mz_intensity.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        Ok(PyMzSpectrum {
            inner: MzSpectrum {
                mz: mz_intensity.iter().map(|(m, _)| *m).collect(),
                intensity: mz_intensity.iter().map(|(_, i)| *i).collect(),
            }
        })
    }

    #[getter]
    pub fn mz(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.mz.clone().into_pyarray_bound(py).unbind()
    }
    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.intensity.clone().into_pyarray_bound(py).unbind()
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min: f64, intensity_max: f64) -> PyResult<PyMzSpectrum> {
        let filtered = self.inner.filter_ranged(mz_min, mz_max, intensity_min, intensity_max);
        let py_filtered = PyMzSpectrum {
            inner: filtered,
        };
        Ok(py_filtered)
    }
    pub fn __add__(&self, other: PyMzSpectrum) -> PyResult<PyMzSpectrum> {
        Ok(PyMzSpectrum { inner: self.inner.clone() + other.inner })
    }
    pub fn __mul__(&self, scale: f64) -> PyResult<PyMzSpectrum> {
        Ok(PyMzSpectrum { inner: self.inner.clone() * scale })
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.inner).unwrap()
    }
}

#[pymodule]
pub fn mz_spectrum(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMzSpectrum>()?;
    Ok(())
}