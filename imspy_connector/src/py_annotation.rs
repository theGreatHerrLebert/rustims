use std::collections::BTreeMap;
use mscore::data::spectrum::MsType;
use pyo3::prelude::*;
use mscore::simulation::annotation::{SourceType, SignalAttributes, ContributionSource, MzSpectrumAnnotated, PeakAnnotation, TimsFrameAnnotated, TimsSpectrumAnnotated};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

#[pyclass]
#[derive(Clone)]
pub struct PyTimsSpectrumAnnotated {
    pub inner: TimsSpectrumAnnotated,
}

#[pymethods]
impl PyTimsSpectrumAnnotated {
    #[new]
    pub unsafe fn new(frame_id: i32,
                      retention_time: f64,
                      scan_id: u32,
                      inv_mobility: f64,
                      ms_type: i32,
                      tof: &Bound<'_, PyArray1<u32>>,
                      mz: &Bound<'_, PyArray1<f64>>,
                      intensity: &Bound<'_, PyArray1<f64>>,
                      annotations: Vec<PyPeakAnnotation>) -> PyResult<Self> {

        let ms_type = MsType::new(ms_type);
        let annotations = annotations.iter().map(|x| PeakAnnotation {
            contributions: x.inner.clone()
        }).collect();

        Ok(PyTimsSpectrumAnnotated {
            inner: TimsSpectrumAnnotated {
                frame_id,
                retention_time,
                scan: scan_id,
                mobility: inv_mobility,
                ms_type,
                tof: tof.as_slice()?.to_vec(),
                spectrum: MzSpectrumAnnotated {
                    mz: mz.as_slice()?.to_vec(),
                    intensity: intensity.as_slice()?.to_vec(),
                    annotations,
                }
            },
        })
    }

    #[getter]
    pub fn frame_id(&self) -> i32 { self.inner.frame_id }

    #[getter]
    pub fn retention_time(&self) -> f64 { self.inner.retention_time }

    #[getter]
    pub fn scan_id(&self) -> u32 { self.inner.scan }

    #[getter]
    pub fn inv_mobility(&self) -> f64 { self.inner.mobility }

    #[getter]
    pub fn mz(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.spectrum.mz.clone().into_pyarray(py).unbind()
    }

    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.spectrum.intensity.clone().into_pyarray(py).unbind()
    }

    #[getter]
    pub fn annotations(&self) -> Vec<PyPeakAnnotation> {
        self.inner.spectrum.annotations.iter().map(|x| PyPeakAnnotation { inner: x.contributions.clone() }).collect()
    }

    #[getter]
    pub fn get_annotated_spectrum(&self) -> PyMzSpectrumAnnotated {
        PyMzSpectrumAnnotated { inner: self.inner.spectrum.clone() }
    }
    #[setter]
    pub unsafe fn set_tof(&mut self, tof: &Bound<'_, PyArray1<u32>>) {
        self.inner.tof = tof.as_slice().unwrap().to_vec();
    }

    pub fn __add__(&self, other: PyTimsSpectrumAnnotated) -> PyResult<PyTimsSpectrumAnnotated> {
        Ok(PyTimsSpectrumAnnotated { inner: self.inner.clone() + other.inner })
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min: f64, intensity_max: f64) -> PyTimsSpectrumAnnotated {
        PyTimsSpectrumAnnotated { inner: self.inner.clone().filter_ranged(mz_min, mz_max, intensity_min, intensity_max) }
    }

    pub fn to_windows(&self, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64) -> BTreeMap<i32, PyTimsSpectrumAnnotated> {
        self.inner.to_windows(window_length, overlapping, min_peaks, min_intensity)
            .into_iter()
            .map(|(id, spectrum)| (id, PyTimsSpectrumAnnotated { inner: spectrum }))
            .collect()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyTimsFrameAnnotated {
    pub inner: TimsFrameAnnotated,
}

#[pymethods]
impl PyTimsFrameAnnotated {
    #[new]
    pub unsafe fn new(frame_id: i32,
                      retention_time: f64,
                      ms_type: i32,
                      tof: &Bound<'_, PyArray1<u32>>,
                      mz: &Bound<'_, PyArray1<f64>>,
                      scan: &Bound<'_, PyArray1<u32>>,
                      inv_mobility: &Bound<'_, PyArray1<f64>>,
                      intensity: &Bound<'_, PyArray1<f64>>,
                      annotations: Vec<PyPeakAnnotation>) -> PyResult<Self> {

        assert!(tof.len()? == mz.len()? && mz.len()? == scan.len()? && scan.len()? == inv_mobility.len()? && inv_mobility.len()? == intensity.len()? && intensity.len()? == annotations.len());

        let ms_type = MsType::new(ms_type);
        let annotations = annotations.iter().map(|x| PeakAnnotation {
            contributions: x.inner.clone()
        }).collect();

        Ok(PyTimsFrameAnnotated {
            inner: TimsFrameAnnotated {
                frame_id,
                retention_time,
                ms_type,
                tof: tof.as_slice()?.to_vec(),
                mz: mz.as_slice()?.to_vec(),
                scan: scan.as_slice()?.to_vec(),
                inv_mobility: inv_mobility.as_slice()?.to_vec(),
                intensity: intensity.as_slice()?.to_vec(),
                annotations,
            },
        })
    }

    #[getter]
    pub fn frame_id(&self) -> i32 { self.inner.frame_id }

    #[getter]
    pub fn retention_time(&self) -> f64 { self.inner.retention_time }
    #[getter]
    pub fn tof(&self, py: Python) -> Py<PyArray1<u32>> { self.inner.tof.clone().into_pyarray(py).unbind() }

    #[getter]
    pub fn mz(&self, py: Python) -> Py<PyArray1<f64>> { self.inner.mz.clone().into_pyarray(py).unbind() }

    #[getter]
    pub fn scan(&self, py: Python) -> Py<PyArray1<u32>> { self.inner.scan.clone().into_pyarray(py).unbind() }

    #[getter]
    pub fn inv_mobility(&self, py: Python) -> Py<PyArray1<f64>> { self.inner.inv_mobility.clone().into_pyarray(py).unbind() }

    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> { self.inner.intensity.clone().into_pyarray(py).unbind() }

    #[getter]
    pub fn annotations(&self) -> Vec<PyPeakAnnotation> {
        self.inner.annotations.iter().map(|x| PyPeakAnnotation { inner: x.contributions.clone() }).collect()
    }

    #[getter]
    pub fn peptide_ids_first_only(&self, py: Python) -> Py<PyArray1<i32>> {
        let data: Vec<_> = self.inner.annotations.iter().map(|x| {
            x.contributions.first().map_or(-1, |contribution| {
                contribution.signal_attributes.as_ref().map_or(-1, |signal_attributes| signal_attributes.peptide_id)
            })
        }).collect();
        data.into_pyarray(py).unbind()
    }

    #[getter]
    pub fn charge_states_first_only(&self, py: Python) ->  Py<PyArray1<i32>> {
        let data: Vec<_> =self.inner.annotations.iter().map(|x| {
            x.contributions.first().map_or(-1, |contribution| {
                contribution.signal_attributes.as_ref().map_or(-1, |signal_attributes| signal_attributes.charge_state)
            })
        }).collect();
        data.into_pyarray(py).unbind()
    }

    #[getter]
    pub fn isotope_peaks_first_only(&self, py: Python) ->  Py<PyArray1<i32>> {
        let data: Vec<_> =self.inner.annotations.iter().map(|x| {
            x.contributions.first().map_or(-1, |contribution| {
                contribution.signal_attributes.as_ref().map_or(-1, |signal_attributes| signal_attributes.isotope_peak)
            })
        }).collect();
        data.into_pyarray(py).unbind()
    }

    #[getter]
    pub fn ms_type_numeric(&self) -> i32 {
        self.inner.ms_type.ms_type_numeric()
    }

    #[setter]
    pub unsafe fn set_tof(&mut self, tof: &Bound<'_, PyArray1<u32>>) {
        self.inner.tof = tof.as_slice().unwrap().to_vec();
    }

    pub fn __add__(&self, other: PyTimsFrameAnnotated) -> PyResult<PyTimsFrameAnnotated> {
        Ok(PyTimsFrameAnnotated { inner: self.inner.clone() + other.inner })
    }

    pub fn to_tims_spectra_annotated(&self) -> Vec<PyTimsSpectrumAnnotated> {
        self.inner.to_tims_spectra_annotated().iter().map(|x| PyTimsSpectrumAnnotated { inner: x.clone() }).collect()
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, inv_mobility_min: f64, inv_mobility_max: f64, scan_min: u32, scan_max: u32, intensity_min: f64, intensity_max: f64) -> PyTimsFrameAnnotated {
        PyTimsFrameAnnotated { inner: self.inner.clone().filter_ranged(mz_min, mz_max, inv_mobility_min, inv_mobility_max, scan_min, scan_max, intensity_min, intensity_max) }
    }

    pub fn to_windows_indexed(
        &self,
        window_length: f64,
        overlapping: bool,
        min_peaks: usize,
        min_intensity: f64
    ) -> (Vec<u32>, Vec<i32>, Vec<PyTimsSpectrumAnnotated>) {
        let (ids, indices, spectra) = self.inner.to_windows_indexed(window_length, overlapping, min_peaks, min_intensity);
        let py_spectra = spectra.into_iter().map(|s| PyTimsSpectrumAnnotated { inner: s }).collect();
        (ids, indices, py_spectra)
    }

    pub fn to_windows(
        &self,
        window_length: f64,
        overlapping: bool,
        min_peaks: usize,
        min_intensity: f64
    ) -> Vec<PyTimsSpectrumAnnotated> {
        self.inner.to_windows(window_length, overlapping, min_peaks, min_intensity)
            .into_iter()
            .map(|s| PyTimsSpectrumAnnotated { inner: s })
            .collect()
    }

    pub fn to_dense_windows(
        &self,
        window_length: f64,
        overlapping: bool,
        min_peaks: usize,
        min_intensity: f64,
        resolution: i32
    ) -> (Vec<f64>, Vec<i32>, Vec<i32>, usize, usize) {
        self.inner.to_dense_windows(window_length, overlapping, min_peaks, min_intensity, resolution)
    }
    pub fn to_dense_windows_with_labels(
        &self,
        window_length: f64,
        overlapping: bool,
        min_peaks: usize,
        min_intensity: f64,
        resolution: i32,
    ) -> (
        Vec<f64>,    // intensities
        Vec<u32>,    // scan index per row
        Vec<i32>,    // window key per row
        Vec<f64>,    // mz values
        Vec<f64>,    // inv_mobility values
        usize,       // n_rows
        usize,       // n_cols
        Vec<i32>,    // isotopologue labels
        Vec<i32>,    // charge_state labels
        Vec<i32>,    // feature_id labels
    ) {
        self.inner.to_dense_windows_with_labels(window_length, overlapping, min_peaks, min_intensity, resolution)
    }

    pub fn fold_along_scan_axis(&self, fold_width: usize) -> PyTimsFrameAnnotated {
        let folded_frame = self.inner.clone().fold_along_scan_axis(fold_width);
        PyTimsFrameAnnotated { inner: folded_frame }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyMzSpectrumAnnotated {
    pub inner: MzSpectrumAnnotated,
}

#[pymethods]
impl PyMzSpectrumAnnotated {
    #[new]
    pub unsafe fn new(mz: &Bound<'_, PyArray1<f64>>, intensity:&Bound<'_, PyArray1<f64>>, annotations: Vec<PyPeakAnnotation>) -> PyResult<Self> {
        assert!(mz.len()? == intensity.len()? && intensity.len()? == annotations.len());
        let annotations = annotations.iter().map(|x| PeakAnnotation {
            contributions: x.inner.clone()
        }).collect();
        Ok(PyMzSpectrumAnnotated {
            inner: MzSpectrumAnnotated {
                mz: mz.as_slice()?.to_vec(),
                intensity: intensity.as_slice()?.to_vec(),
                annotations,
            },
        })
    }
    #[getter]
    pub fn mz(&self, py: Python) -> Py<PyArray1<f64>> { self.inner.mz.clone().into_pyarray(py).unbind() }

    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> { self.inner.intensity.clone().into_pyarray(py).unbind() }

    #[getter]
    pub fn annotations(&self) -> Vec<PyPeakAnnotation> {
        self.inner.annotations.iter().map(|x| PyPeakAnnotation { inner: x.contributions.clone() }).collect()
    }

    pub fn __add__(&self, other: PyMzSpectrumAnnotated) -> PyResult<PyMzSpectrumAnnotated> {
        Ok(PyMzSpectrumAnnotated { inner: self.inner.clone() + other.inner })
    }

    pub fn __mul__(&self, other: f64) -> PyResult<PyMzSpectrumAnnotated> {
        Ok(PyMzSpectrumAnnotated { inner: self.inner.clone() * other })
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, intensity_min: f64, intensity_max: f64) -> PyMzSpectrumAnnotated {
        PyMzSpectrumAnnotated { inner: self.inner.clone().filter_ranged(mz_min, mz_max, intensity_min, intensity_max) }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyPeakAnnotation {
    pub inner: Vec<ContributionSource>,
}

#[pymethods]
impl PyPeakAnnotation {
    #[new]
    pub fn new(contributions: Vec<PyContributionSource>) -> Self {
        PyPeakAnnotation {
            inner: contributions.iter().map(|x| x.inner.clone()).collect(),
        }
    }
    #[getter]
    pub fn contributions(&self) -> Vec<PyContributionSource> {
        self.inner.iter().map(|x| PyContributionSource { inner: x.clone() }).collect()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PySourceType {
    inner: SourceType,
}
#[pymethods]
impl PySourceType {
    #[new]
    pub fn new(source_type: i32) -> PyResult<Self> {
        Ok(PySourceType {
            inner: SourceType::new(source_type),
        })
    }
    #[getter]
    pub fn source_type(&self) -> String { self.inner.to_string() }
}

#[pyclass]
#[derive(Clone)]
pub struct PySignalAttributes {
    inner: SignalAttributes,
}
#[pymethods]
impl PySignalAttributes {
    #[new]
    #[pyo3(signature = (charge_state, peptide_id, isotope_peak, description=None))]
    pub fn new(charge_state: i32, peptide_id: i32, isotope_peak: i32, description: Option<String>) -> PyResult<Self> {
        Ok(PySignalAttributes {
            inner: SignalAttributes {
                charge_state,
                peptide_id,
                isotope_peak,
                description,
            },
        })
    }
    #[getter]
    pub fn charge_state(&self) -> i32 { self.inner.charge_state }
    #[getter]
    pub fn peptide_id(&self) -> i32 { self.inner.peptide_id }
    #[getter]
    pub fn isotope_peak(&self) -> i32 { self.inner.isotope_peak }
    #[getter]
    pub fn description(&self) -> Option<String> { self.inner.description.clone() }
}

#[pyclass]
#[derive(Clone)]
pub struct PyContributionSource {
    pub inner: ContributionSource,
}

#[pymethods]
impl PyContributionSource {
    #[new]
    #[pyo3(signature = (intensity_contribution, source_type, signal_attributes=None))]
    pub fn new(intensity_contribution: f64, source_type: PySourceType, signal_attributes: Option<PySignalAttributes>) -> Self {
        PyContributionSource {
            inner: ContributionSource {
                intensity_contribution,
                source_type: source_type.inner.clone(),
                signal_attributes: signal_attributes.map(|x| x.inner.clone()),
            },
        }
    }

    #[getter]
    pub fn intensity_contribution(&self) -> f64 { self.inner.intensity_contribution }

    #[getter]
    pub fn source_type(&self) -> PySourceType { PySourceType { inner: self.inner.source_type.clone() } }

    #[getter]
    pub fn signal_attributes(&self) -> Option<PySignalAttributes> {
        self.inner.signal_attributes.as_ref().map(|x| PySignalAttributes { inner: x.clone() })
    }
}

#[pymodule]
pub fn py_annotation(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySourceType>()?;
    m.add_class::<PySignalAttributes>()?;
    m.add_class::<PyContributionSource>()?;
    m.add_class::<PyPeakAnnotation>()?;
    m.add_class::<PyMzSpectrumAnnotated>()?;
    m.add_class::<PyTimsSpectrumAnnotated>()?;
    m.add_class::<PyTimsFrameAnnotated>()?;
    Ok(())
}