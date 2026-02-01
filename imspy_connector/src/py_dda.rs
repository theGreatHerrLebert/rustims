use pyo3::prelude::*;
use numpy::{PyArray1, IntoPyArray};

use rustdf::data::dda::{PASEFDDAFragment, TimsDatasetDDA, PrecursorCoord, PrecursorMS1Signal, SignalMoments};
use rustdf::data::handle::{TimsData, IndexConverter};
use rustdf::data::meta::DDAPrecursor;
use crate::py_tims_frame::PyTimsFrame;
use crate::py_tims_slice::PyTimsSlice;
use crate::py_spectrum_processing::{PyPreprocessedSpectrum, PySpectrumProcessingConfig};

#[pyclass]
pub struct PyDDAPrecursor {
    inner: DDAPrecursor,
}

#[pymethods]
impl PyDDAPrecursor {
    #[new]
    #[pyo3(signature = (frame_id, precursor_id, highest_intensity_mz, average_mz, inverse_ion_mobility, collision_energy, precuror_total_intensity, isolation_mz, isolation_width, mono_mz=None, charge=None))]
    pub fn new(
        frame_id: i64,
        precursor_id: i64,
        highest_intensity_mz: f64,
        average_mz: f64,
        inverse_ion_mobility: f64,
        collision_energy: f64,
        precuror_total_intensity: f64,
        isolation_mz: f64,
        isolation_width: f64,
        mono_mz: Option<f64>,
        charge: Option<i64>,
    ) -> Self {
        let precursor = DDAPrecursor {
            frame_id,
            precursor_id,
            mono_mz,
            highest_intensity_mz,
            average_mz,
            charge,
            inverse_ion_mobility,
            collision_energy,
            precuror_total_intensity,
            isolation_mz,
            isolation_width,
        };
        PyDDAPrecursor { inner: precursor }
    }

    #[getter]
    pub fn frame_id(&self) -> i64 { self.inner.frame_id }

    #[getter]
    pub fn precursor_id(&self) -> i64 { self.inner.precursor_id }

    #[getter]
    pub fn mono_mz(&self) -> Option<f64> { self.inner.mono_mz }

    #[getter]
    pub fn highest_intensity_mz(&self) -> f64 { self.inner.highest_intensity_mz }

    #[getter]
    pub fn average_mz(&self) -> f64 { self.inner.average_mz }

    #[getter]
    pub fn charge(&self) -> Option<i64> { self.inner.charge }

    #[getter]
    pub fn inverse_ion_mobility(&self) -> f64 { self.inner.inverse_ion_mobility }

    #[getter]
    pub fn collision_energy(&self) -> f64 { self.inner.collision_energy }

    #[getter]
    pub fn precuror_total_intensity(&self) -> f64 { self.inner.precuror_total_intensity }

    #[getter]
    pub fn isolation_mz(&self) -> f64 { self.inner.isolation_mz }

    #[getter]
    pub fn isolation_width(&self) -> f64 { self.inner.isolation_width }
}

#[pyclass]
pub struct PyTimsDatasetDDA {
    inner: TimsDatasetDDA,
}

#[pymethods]
impl PyTimsDatasetDDA {
    #[new]
    pub fn new(data_path: &str, bruker_lib_path: &str, in_memory: bool, use_bruker_sdk: bool) -> Self {
        let dataset = TimsDatasetDDA::new(bruker_lib_path, data_path, in_memory, use_bruker_sdk);
        PyTimsDatasetDDA { inner: dataset }
    }

    /// Create a DDA dataset with pre-computed ion mobility calibration lookup table.
    ///
    /// This enables accurate ion mobility calibration with fast parallel extraction.
    /// The im_lookup array should be pre-computed using the Bruker SDK via
    /// extract_calibration.py or by opening a dataset with use_bruker_sdk=True
    /// and calling scan_to_inverse_mobility for all scan indices.
    ///
    /// # Arguments
    /// * `data_path` - Path to the .d folder
    /// * `in_memory` - Whether to load all data into memory
    /// * `im_lookup` - Pre-computed scan→1/K0 lookup table (list or numpy array)
    ///
    /// # Returns
    /// A new PyTimsDatasetDDA with accurate IM calibration and thread-safe parallel access
    ///
    /// # Example
    /// ```python
    /// import numpy as np
    /// from imspy_connector import py_dda
    ///
    /// # Load pre-computed calibration
    /// im_lookup = np.load("sample.im_calibration.npy")
    ///
    /// # Create dataset with calibration (fast + accurate)
    /// dataset = py_dda.PyTimsDatasetDDA.with_calibration("sample.d", False, im_lookup.tolist())
    /// ```
    #[staticmethod]
    pub fn with_calibration(data_path: &str, in_memory: bool, im_lookup: Vec<f64>) -> Self {
        let dataset = TimsDatasetDDA::new_with_calibration(data_path, in_memory, im_lookup);
        PyTimsDatasetDDA { inner: dataset }
    }

    /// Check if the Bruker SDK is being used for index conversion.
    /// Returns false for Simple and Lookup converters (thread-safe).
    /// Returns true only for BrukerLib converter (NOT thread-safe).
    pub fn uses_bruker_sdk(&self) -> bool {
        self.inner.uses_bruker_sdk()
    }

    /// Convert scan indices to inverse mobility (1/K0) values.
    ///
    /// # Arguments
    /// * `frame_id` - Frame ID (typically not used for calibrated datasets)
    /// * `scan_values` - List of scan indices to convert
    ///
    /// # Returns
    /// List of 1/K0 values corresponding to the scan indices
    pub fn scan_to_inverse_mobility(&self, frame_id: u32, scan_values: Vec<u32>) -> Vec<f64> {
        self.inner.scan_to_inverse_mobility(frame_id, &scan_values)
    }

    /// Convert inverse mobility (1/K0) values to scan indices.
    ///
    /// # Arguments
    /// * `frame_id` - Frame ID (typically not used for calibrated datasets)
    /// * `im_values` - List of 1/K0 values to convert
    ///
    /// # Returns
    /// List of scan indices corresponding to the 1/K0 values
    pub fn inverse_mobility_to_scan(&self, frame_id: u32, im_values: Vec<f64>) -> Vec<u32> {
        self.inner.inverse_mobility_to_scan(frame_id, &im_values)
    }

    pub fn get_frame(&self, frame_id: u32) -> PyTimsFrame {
        PyTimsFrame::from_inner(self.inner.get_frame(frame_id))
    }

    pub fn get_slice(&self, frame_ids: Vec<u32>, num_threads: usize) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.get_slice(frame_ids, num_threads) }
    }

    pub fn get_acquisition_mode(&self) -> String {
        self.inner.get_acquisition_mode().to_string()
    }

    pub fn get_frame_count(&self) -> i32 {
        self.inner.get_frame_count()
    }

    pub fn get_data_path(&self) -> &str {
        self.inner.get_data_path()
    }

    pub fn get_pasef_fragments(&self, num_threads: usize) -> Vec<PyTimsFragmentDDA> {
        let pasef_fragments = self.inner.get_pasef_fragments(num_threads);
        pasef_fragments.iter().map(|pasef_fragment| PyTimsFragmentDDA { inner: pasef_fragment.clone() }).collect()
    }

    /// Get fragment spectra for specific precursor IDs only.
    /// More memory-efficient for batched processing - only loads frames for requested precursors.
    ///
    /// # Arguments
    /// * `precursor_ids` - List of precursor IDs to extract fragments for
    /// * `num_threads` - Number of threads for parallel processing
    ///
    /// # Returns
    /// List of PyTimsFragmentDDA, one per PASEF event for the requested precursors
    pub fn get_pasef_fragments_for_precursors(
        &self,
        precursor_ids: Vec<u32>,
        num_threads: usize,
    ) -> Vec<PyTimsFragmentDDA> {
        let pasef_fragments = self.inner.get_pasef_fragments_for_precursors(
            Some(&precursor_ids),
            num_threads,
        );
        pasef_fragments.iter().map(|pasef_fragment| PyTimsFragmentDDA { inner: pasef_fragment.clone() }).collect()
    }

    pub fn get_selected_precursors(&self) -> Vec<PyDDAPrecursor> {
        let pasef_precursor_meta = self.inner.get_selected_precursors();
        pasef_precursor_meta.iter().map(|precursor_meta| PyDDAPrecursor { inner: precursor_meta.clone() }).collect()
    }

    pub fn get_precursor_frames(&self, min_intensity: f64, max_peaks: usize, num_threads: usize) -> Vec<PyTimsFrame> {
        let precursor_frames = self.inner.get_precursor_frames(min_intensity, max_peaks, num_threads);
        precursor_frames.iter().map(|frame| PyTimsFrame::from_inner(frame.clone())).collect()
    }
    
    pub fn sample_pasef_fragments_random(&self, target_scan_apex_values: Vec<i32>, experiment_max_scan: i32, ) -> PyTimsFrame {
        let tims_frame = self.inner.sample_pasef_fragments_random(
            target_scan_apex_values,
            experiment_max_scan,
        );
        PyTimsFrame::from_inner(tims_frame)
    }

    pub fn sample_precursor_signal(&self, num_frames: usize, max_intensity: f64, take_probability: f64) -> PyTimsFrame {
        PyTimsFrame::from_inner(self.inner.sample_precursor_signal(num_frames, max_intensity, take_probability))
    }

    /// Get preprocessed PASEF fragments ready for database search.
    /// This method performs parallel processing of all fragment spectra.
    ///
    /// # Arguments
    /// * `dataset_name` - Name of the dataset for generating spec_ids
    /// * `config` - Spectrum processing configuration
    /// * `num_threads` - Number of threads for parallel processing
    ///
    /// # Returns
    /// List of preprocessed spectra ready for Sage search
    #[pyo3(signature = (dataset_name, config, num_threads=16))]
    pub fn get_preprocessed_pasef_fragments(
        &self,
        dataset_name: &str,
        config: &PySpectrumProcessingConfig,
        num_threads: usize,
    ) -> Vec<PyPreprocessedSpectrum> {
        let preprocessed = self.inner.get_preprocessed_pasef_fragments(
            dataset_name,
            config.inner.clone(),
            num_threads,
        );
        preprocessed
            .into_iter()
            .map(|spec| PyPreprocessedSpectrum::from_inner(spec))
            .collect()
    }

    /// Extract MS1 precursor signals for a batch of precursors in parallel.
    ///
    /// For each precursor, extracts XIC (chromatographic profile), mobilogram,
    /// and isotope envelope from MS1 frames in the specified RT window.
    /// Also calculates statistical moments (mean, variance, skewness, apex, FWHM).
    ///
    /// # Arguments
    /// * `precursor_coords` - List of PyPrecursorCoord objects
    /// * `rt_window_sec` - RT window in seconds (total width, default 10.0)
    /// * `mz_tol_ppm` - m/z tolerance in ppm (default 20.0)
    /// * `im_window` - IM window in 1/K0 units (default 0.05)
    /// * `n_isotopes` - Number of isotope peaks to extract (default 5)
    /// * `num_threads` - Number of threads for parallel processing (default 16)
    ///
    /// # Returns
    /// List of PyPrecursorMS1Signal, one per input precursor
    #[pyo3(signature = (precursor_coords, rt_window_sec=10.0, mz_tol_ppm=20.0, im_window=0.05, n_isotopes=5, num_threads=16))]
    pub fn extract_precursor_ms1_signals(
        &self,
        precursor_coords: Vec<PyPrecursorCoord>,
        rt_window_sec: f64,
        mz_tol_ppm: f64,
        im_window: f64,
        n_isotopes: usize,
        num_threads: usize,
    ) -> Vec<PyPrecursorMS1Signal> {
        let coords: Vec<PrecursorCoord> = precursor_coords
            .into_iter()
            .map(|c| c.inner)
            .collect();

        let signals = self.inner.extract_precursor_ms1_signals(
            coords,
            rt_window_sec,
            mz_tol_ppm,
            im_window,
            n_isotopes,
            num_threads,
        );

        signals
            .into_iter()
            .map(|s| PyPrecursorMS1Signal { inner: s })
            .collect()
    }
}

#[pyclass]
pub struct PyTimsFragmentDDA {
    inner: PASEFDDAFragment,
}

#[pymethods]
impl PyTimsFragmentDDA {
    #[new]
    pub fn new(frame_id: u32, precursor_id: u32, collision_energy: f64, selected_fragment: &PyTimsFrame) -> PyResult<Self> {

        let pasef_fragment = PASEFDDAFragment {
            frame_id,
            precursor_id,
            collision_energy,
            selected_fragment: selected_fragment.inner.clone(),
        };

        Ok(PyTimsFragmentDDA { inner: pasef_fragment })
    }

    #[getter]
    pub fn frame_id(&self) -> u32 { self.inner.frame_id }

    #[getter]
    pub fn precursor_id(&self) -> u32 { self.inner.precursor_id }

    #[getter]
    pub fn selected_fragment(&self) -> PyTimsFrame { PyTimsFrame::from_inner(self.inner.selected_fragment.clone()) }

    #[getter]
    pub fn collision_energy(&self) -> f64 { self.inner.collision_energy }
}

/// Statistical moments of a signal distribution
#[pyclass]
#[derive(Clone)]
pub struct PySignalMoments {
    inner: SignalMoments,
}

#[pymethods]
impl PySignalMoments {
    #[getter]
    pub fn mean(&self) -> f64 { self.inner.mean }

    #[getter]
    pub fn variance(&self) -> f64 { self.inner.variance }

    #[getter]
    pub fn skewness(&self) -> f64 { self.inner.skewness }

    #[getter]
    pub fn apex(&self) -> f64 { self.inner.apex }

    #[getter]
    pub fn fwhm(&self) -> f64 { self.inner.fwhm }

    #[getter]
    pub fn total_intensity(&self) -> f64 { self.inner.total_intensity }
}

/// MS1 precursor signal extracted from surrounding frames
#[pyclass]
pub struct PyPrecursorMS1Signal {
    inner: PrecursorMS1Signal,
}

#[pymethods]
impl PyPrecursorMS1Signal {
    #[getter]
    pub fn precursor_id(&self) -> u32 { self.inner.precursor_id }

    #[getter]
    pub fn rt_coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.rt_coords.clone().into_pyarray(py)
    }

    #[getter]
    pub fn rt_intensities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.rt_intensities.clone().into_pyarray(py)
    }

    #[getter]
    pub fn rt_moments(&self) -> PySignalMoments {
        PySignalMoments { inner: self.inner.rt_moments.clone() }
    }

    #[getter]
    pub fn im_coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.im_coords.clone().into_pyarray(py)
    }

    #[getter]
    pub fn im_intensities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.im_intensities.clone().into_pyarray(py)
    }

    #[getter]
    pub fn im_moments(&self) -> PySignalMoments {
        PySignalMoments { inner: self.inner.im_moments.clone() }
    }

    #[getter]
    pub fn isotope_mz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.isotope_mz.clone().into_pyarray(py)
    }

    #[getter]
    pub fn isotope_intensity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.isotope_intensity.clone().into_pyarray(py)
    }

    #[getter]
    pub fn mz_moments(&self) -> PySignalMoments {
        PySignalMoments { inner: self.inner.mz_moments.clone() }
    }

    // Raw 2D data (merged from all MS1 frames in RT window)
    #[getter]
    pub fn raw_rt<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.raw_rt.clone().into_pyarray(py)
    }

    #[getter]
    pub fn raw_mz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.raw_mz.clone().into_pyarray(py)
    }

    #[getter]
    pub fn raw_mobility<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.raw_mobility.clone().into_pyarray(py)
    }

    #[getter]
    pub fn raw_intensity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.raw_intensity.clone().into_pyarray(py)
    }
}

/// Input coordinates for MS1 extraction
#[pyclass]
#[derive(Clone)]
pub struct PyPrecursorCoord {
    inner: PrecursorCoord,
}

#[pymethods]
impl PyPrecursorCoord {
    #[new]
    pub fn new(precursor_id: u32, mz: f64, rt_seconds: f64, mobility: f64, charge: i32) -> Self {
        PyPrecursorCoord {
            inner: PrecursorCoord {
                precursor_id,
                mz,
                rt_seconds,
                mobility,
                charge,
            }
        }
    }

    #[getter]
    pub fn precursor_id(&self) -> u32 { self.inner.precursor_id }

    #[getter]
    pub fn mz(&self) -> f64 { self.inner.mz }

    #[getter]
    pub fn rt_seconds(&self) -> f64 { self.inner.rt_seconds }

    #[getter]
    pub fn mobility(&self) -> f64 { self.inner.mobility }

    #[getter]
    pub fn charge(&self) -> i32 { self.inner.charge }
}

#[pymodule]
pub fn py_dda(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimsDatasetDDA>()?;
    m.add_class::<PyTimsFragmentDDA>()?;
    m.add_class::<PyDDAPrecursor>()?;
    m.add_class::<PyPrecursorCoord>()?;
    m.add_class::<PyPrecursorMS1Signal>()?;
    m.add_class::<PySignalMoments>()?;
    Ok(())
}