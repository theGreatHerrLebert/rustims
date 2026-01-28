use std::collections::BTreeMap;
use mscore::timstof::collision::TimsTofCollisionEnergy;
use pyo3::prelude::*;
use rustdf::sim::containers::{IsotopeTransmissionConfig, IsotopeTransmissionMode};
use rustdf::sim::dda::TimsTofSyntheticsFrameBuilderDDA;
use rustdf::sim::dia::TimsTofSyntheticsFrameBuilderDIA;
use rustdf::sim::lazy_builder::TimsTofLazyFrameBuilderDIA;
use rustdf::sim::precursor::TimsTofSyntheticsPrecursorFrameBuilder;
use rustdf::sim::handle::TimsTofSyntheticsDataHandle;
use crate::py_annotation::PyTimsFrameAnnotated;
use crate::py_mz_spectrum::PyMzSpectrum;
use crate::py_peptide::PyPeptideProductIonSeriesCollection;
use crate::py_quadrupole::PyPasefMeta;
use crate::py_tims_frame::PyTimsFrame;

/// Configuration for quad-selection dependent isotope transmission.
///
/// # Arguments
///
/// * `mode` - The isotope transmission mode:
///   - "none" (default): No transmission-dependent calculation
///   - "precursor_scaling": Calculate transmission factor from precursor isotope distribution
///     and apply uniform scaling to all fragment intensities (computationally efficient)
///   - "per_fragment": Calculate transmission-dependent isotope distribution for each
///     individual fragment ion based on its complementary fragment (more accurate but slower)
/// * `min_probability` - Minimum probability threshold for isotope transmission (default: 0.5)
/// * `max_isotopes` - Maximum number of isotope peaks to consider (default: 10)
#[pyclass]
#[derive(Clone)]
pub struct PyIsotopeTransmissionConfig {
    pub inner: IsotopeTransmissionConfig,
}

#[pymethods]
impl PyIsotopeTransmissionConfig {
    #[new]
    #[pyo3(signature = (mode="none", min_probability=0.5, max_isotopes=10, precursor_survival_min=0.0, precursor_survival_max=0.0))]
    pub fn new(
        mode: &str,
        min_probability: f64,
        max_isotopes: usize,
        precursor_survival_min: f64,
        precursor_survival_max: f64,
    ) -> Self {
        let transmission_mode = match mode.to_lowercase().as_str() {
            "precursor_scaling" | "precursor" => IsotopeTransmissionMode::PrecursorScaling,
            "per_fragment" | "fragment" => IsotopeTransmissionMode::PerFragment,
            _ => IsotopeTransmissionMode::None,
        };
        PyIsotopeTransmissionConfig {
            inner: IsotopeTransmissionConfig::new(
                transmission_mode,
                min_probability,
                max_isotopes,
                precursor_survival_min,
                precursor_survival_max,
            ),
        }
    }

    /// Create config with precursor scaling mode
    #[staticmethod]
    #[pyo3(signature = (min_probability=0.5))]
    pub fn precursor_scaling(min_probability: f64) -> Self {
        PyIsotopeTransmissionConfig {
            inner: IsotopeTransmissionConfig::precursor_scaling(min_probability),
        }
    }

    /// Create config with per-fragment mode
    #[staticmethod]
    #[pyo3(signature = (min_probability=0.5, max_isotopes=10))]
    pub fn per_fragment(min_probability: f64, max_isotopes: usize) -> Self {
        PyIsotopeTransmissionConfig {
            inner: IsotopeTransmissionConfig::per_fragment(min_probability, max_isotopes),
        }
    }

    #[getter]
    pub fn mode(&self) -> String {
        match self.inner.mode {
            IsotopeTransmissionMode::None => "none".to_string(),
            IsotopeTransmissionMode::PrecursorScaling => "precursor_scaling".to_string(),
            IsotopeTransmissionMode::PerFragment => "per_fragment".to_string(),
        }
    }

    #[getter]
    pub fn enabled(&self) -> bool {
        self.inner.is_enabled()
    }

    #[getter]
    pub fn min_probability(&self) -> f64 {
        self.inner.min_probability
    }

    #[getter]
    pub fn max_isotopes(&self) -> usize {
        self.inner.max_isotopes
    }

    #[getter]
    pub fn precursor_survival_min(&self) -> f64 {
        self.inner.precursor_survival_min
    }

    #[getter]
    pub fn precursor_survival_max(&self) -> f64 {
        self.inner.precursor_survival_max
    }
}

#[pyclass(unsendable)]
pub struct PyTimsTofSyntheticsDataHandle {
    pub inner: TimsTofSyntheticsDataHandle,
}

#[pymethods]
impl PyTimsTofSyntheticsDataHandle {
    #[new]
    pub fn new(db_path: &str) -> Self {
        let path = std::path::Path::new(db_path);
        PyTimsTofSyntheticsDataHandle { inner: TimsTofSyntheticsDataHandle::new(path).unwrap() }
    }

    #[pyo3(signature = (num_threads=None, dda=None))]
    pub fn get_transmitted_ions(&self, num_threads: Option<usize>, dda: Option<bool>) -> (Vec<i32>, Vec<i32>, Vec<String>, Vec<i8>, Vec<f32>) {
        let threads = num_threads.unwrap_or(4);
        self.inner.get_transmitted_ions(threads, dda.unwrap_or(false))
    }

    /// Get transmitted ions for a specific frame range (lazy loading version).
    /// This reduces memory usage by only loading peptides and ions relevant to the frame range.
    ///
    /// # Arguments
    ///
    /// * `frame_min` - Minimum frame ID to include (inclusive)
    /// * `frame_max` - Maximum frame ID to include (inclusive)
    /// * `num_threads` - Number of threads to use for parallel processing
    /// * `dda` - If true, use DDA transmission; if false, use DIA transmission
    ///
    /// # Returns
    ///
    /// Tuple of (peptide_ids, ion_ids, sequences, charges, collision_energies)
    #[pyo3(signature = (frame_min, frame_max, num_threads=None, dda=None))]
    pub fn get_transmitted_ions_for_frame_range(
        &self,
        frame_min: u32,
        frame_max: u32,
        num_threads: Option<usize>,
        dda: Option<bool>,
    ) -> (Vec<i32>, Vec<i32>, Vec<String>, Vec<i8>, Vec<f32>) {
        let threads = num_threads.unwrap_or(4);
        self.inner.get_transmitted_ions_for_frame_range(frame_min, frame_max, threads, dda.unwrap_or(false))
    }
}

#[pyclass]
pub struct PyTimsTofSyntheticsPrecursorFrameBuilder {
    pub inner: TimsTofSyntheticsPrecursorFrameBuilder,
}

#[pymethods]
impl PyTimsTofSyntheticsPrecursorFrameBuilder {
    #[new]
    pub fn new(db_path: &str) -> Self {
        let path = std::path::Path::new(db_path);
        PyTimsTofSyntheticsPrecursorFrameBuilder { inner: TimsTofSyntheticsPrecursorFrameBuilder::new(path).unwrap() }
    }

    pub fn build_precursor_frame(&self, frame_id: u32, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, right_drag: bool) -> PyTimsFrame {
        PyTimsFrame::from_inner(self.inner.build_precursor_frame(frame_id, mz_noise_precursor, uniform, precursor_noise_ppm, right_drag))
    }

    pub fn build_precursor_frames(&self, frame_ids: Vec<u32>, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, right_drag: bool, num_threads: usize) -> Vec<PyTimsFrame> {
        let frames = self.inner.build_precursor_frames(frame_ids, mz_noise_precursor, uniform, precursor_noise_ppm, right_drag, num_threads);
        // Use into_iter() to move frames instead of cloning
        frames.into_iter().map(|x| PyTimsFrame::from_inner(x)).collect()
    }

    pub fn build_precursor_frame_annotated(&self, frame_id: u32, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, right_drag: bool) -> PyTimsFrameAnnotated {
        PyTimsFrameAnnotated { inner: self.inner.build_precursor_frame_annotated(frame_id, mz_noise_precursor, uniform, precursor_noise_ppm, right_drag) }
    }

    pub fn build_precursor_frames_annotated(&self, frame_ids: Vec<u32>, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, right_drag: bool, num_threads: usize) -> Vec<PyTimsFrameAnnotated> {
        let frames = self.inner.build_precursor_frames_annotated(frame_ids, mz_noise_precursor, uniform, precursor_noise_ppm, right_drag, num_threads);
        frames.into_iter().map(|x| PyTimsFrameAnnotated { inner: x }).collect()
    }
    pub fn frame_to_abundances(&self) -> BTreeMap<u32, (Vec<u32>, Vec<f32>)> {
        self.inner.frame_to_abundances.clone()
    }
}

#[pyclass(unsendable)]
pub struct PyTimsTofSyntheticsFrameBuilderDIA {
    pub inner: TimsTofSyntheticsFrameBuilderDIA,
}

#[pymethods]
impl PyTimsTofSyntheticsFrameBuilderDIA {
    #[new]
    #[pyo3(signature = (db_path, with_annotations, num_threads, isotope_config=None))]
    pub fn new(
        db_path: &str,
        with_annotations: bool,
        num_threads: usize,
        isotope_config: Option<PyIsotopeTransmissionConfig>,
    ) -> Self {
        let path = std::path::Path::new(db_path);
        let config = isotope_config.map(|c| c.inner).unwrap_or_default();
        PyTimsTofSyntheticsFrameBuilderDIA {
            inner: TimsTofSyntheticsFrameBuilderDIA::new_with_config(
                path,
                with_annotations,
                num_threads,
                config,
            ).unwrap()
        }
    }

    pub fn build_frame(&self, frame_id: u32, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool) -> PyTimsFrame {
        let frames = self.inner.build_frames(vec![frame_id], fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, 1);
        // Use into_iter().next() to move first element instead of cloning
        PyTimsFrame::from_inner(frames.into_iter().next().unwrap())
    }

    pub fn build_frame_annotated(&self, frame_id: u32, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool) -> PyTimsFrameAnnotated {
        let frames = self.inner.build_frames_annotated(vec![frame_id], fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, 1);
        PyTimsFrameAnnotated { inner: frames.into_iter().next().unwrap() }
    }

    pub fn build_frames(&self, frame_ids: Vec<u32>, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool, num_threads: usize) -> Vec<PyTimsFrame> {
        let frames = self.inner.build_frames(frame_ids, fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, num_threads);
        frames.into_iter().map(|x| PyTimsFrame::from_inner(x)).collect()
    }

    pub fn build_frames_annotated(&self, frame_ids: Vec<u32>, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool, num_threads: usize) -> Vec<PyTimsFrameAnnotated> {
        let frames = self.inner.build_frames_annotated(frame_ids, fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, num_threads);
        frames.into_iter().map(|x| PyTimsFrameAnnotated { inner: x }).collect()
    }

    pub fn get_collision_energy(&self, frame_id: i32, scan_id: i32) -> f64 {
        self.inner.get_collision_energy(frame_id, scan_id)
    }

    pub fn get_collision_energies(&self, frame_ids: Vec<i32>, scan_ids: Vec<i32>) -> Vec<f64> {
        let mut result = Vec::with_capacity(frame_ids.len());
        for (frame_id, scan_id) in frame_ids.iter().zip(scan_ids.iter()) {
            result.push(self.inner.get_collision_energy(*frame_id, *scan_id));
        }
        result
    }

    pub fn get_ion_transmission_matrix(&self, peptide_id: u32, charge: i8, include_precursor_frames: bool) -> Vec<Vec<f32>> {
        self.inner.get_ion_transmission_matrix(peptide_id, charge, include_precursor_frames)
    }

    pub fn count_number_transmissions(&self, peptide_id: u32, charge: i8) -> (usize, usize) {
        self.inner.count_number_transmissions(peptide_id, charge)
    }

    pub fn count_number_transmissions_parallel(&self, peptide_ids: Vec<u32>, charge: Vec<i8>, num_threads: usize) -> Vec<(usize, usize)> {
        self.inner.count_number_transmissions_parallel(peptide_ids, charge, num_threads)
    }

    pub fn get_fragment_ions_map(&self) -> BTreeMap<(u32, i8, i32), (PyPeptideProductIonSeriesCollection, Vec<PyMzSpectrum>)> {
        let mut result = BTreeMap::new();
        if let Some(ref fragment_ions) = self.inner.fragment_ions {
            for (key, (peptide_ions, mz_spectra)) in fragment_ions.iter() {
                let peptide_ions = PyPeptideProductIonSeriesCollection { inner: peptide_ions.clone() };
                let mz_spectra = mz_spectra.iter().map(|x| PyMzSpectrum::from_inner(x.clone())).collect::<Vec<_>>();
                result.insert(*key, (peptide_ions, mz_spectra));
            }
        }
        result
    }
}

#[pyclass(unsendable)]
pub struct PyTimsTofSyntheticsFrameBuilderDDA {
    pub inner: TimsTofSyntheticsFrameBuilderDDA,
}

#[pymethods]
impl PyTimsTofSyntheticsFrameBuilderDDA {
    #[new]
    #[pyo3(signature = (db_path, with_annotations, num_threads, isotope_config=None))]
    pub fn new(
        db_path: &str,
        with_annotations: bool,
        num_threads: usize,
        isotope_config: Option<PyIsotopeTransmissionConfig>,
    ) -> Self {
        let path = std::path::Path::new(db_path);
        let config = isotope_config.map(|c| c.inner);
        PyTimsTofSyntheticsFrameBuilderDDA {
            inner: TimsTofSyntheticsFrameBuilderDDA::new(path, with_annotations, num_threads, config),
        }
    }

    pub fn build_frame(&self, frame_id: u32, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool) -> PyTimsFrame {
        let frames = self.inner.build_frames(vec![frame_id], fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, 1);
        PyTimsFrame::from_inner(frames.into_iter().next().unwrap())
    }

    pub fn build_frame_annotated(&self, frame_id: u32, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool) -> PyTimsFrameAnnotated {
        let frames = self.inner.build_frames_annotated(vec![frame_id], fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, 1);
        PyTimsFrameAnnotated { inner: frames.into_iter().next().unwrap() }
    }

    pub fn build_frames(&self, frame_ids: Vec<u32>, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool, num_threads: usize) -> Vec<PyTimsFrame> {
        let frames = self.inner.build_frames(frame_ids, fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, num_threads);
        frames.into_iter().map(|x| PyTimsFrame::from_inner(x)).collect()
    }

    pub fn build_frames_annotated(&self, frame_ids: Vec<u32>, fragmentation: bool, mz_noise_precursor: bool, uniform: bool, precursor_noise_ppm: f64, mz_noise_fragment: bool, fragment_noise_ppm: f64, right_drag: bool, num_threads: usize) -> Vec<PyTimsFrameAnnotated> {
        let frames = self.inner.build_frames_annotated(frame_ids, fragmentation, mz_noise_precursor, uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, num_threads);
        frames.into_iter().map(|x| PyTimsFrameAnnotated { inner: x }).collect()
    }

    pub fn get_collision_energy(&self, frame_id: i32, scan_id: i32) -> f64 {
        self.inner.get_collision_energy(frame_id, scan_id)
    }

    pub fn get_collision_energies(&self, frame_ids: Vec<i32>, scan_ids: Vec<i32>) -> Vec<f64> {
        let mut result = Vec::with_capacity(frame_ids.len());
        for (frame_id, scan_id) in frame_ids.iter().zip(scan_ids.iter()) {
            result.push(self.inner.get_collision_energy(*frame_id, *scan_id));
        }
        result
    }

    pub fn get_pasef_meta(&self) -> Vec<PyPasefMeta> {
        // Estimate capacity: sum of all value list lengths
        let total_len: usize = self.inner.transmission_settings.pasef_meta.values().map(|v| v.len()).sum();
        let mut result = Vec::with_capacity(total_len);
        for values in self.inner.transmission_settings.pasef_meta.values() {
            for value in values.iter() {
                result.push(PyPasefMeta { inner: value.clone() });
            }
        }
        result
    }

    pub fn get_fragment_frames(&self) -> Vec<i32> {
        // extract the keys from the pasef_meta to get all fragment frames sorted by frame_id ascending
        self.inner.transmission_settings.pasef_meta.keys().copied().collect()
    }

    pub fn get_fragment_ions_map(&self) -> BTreeMap<(u32, i8, i32), (PyPeptideProductIonSeriesCollection, Vec<PyMzSpectrum>)> {
        let mut result = BTreeMap::new();
        if let Some(ref fragment_ions) = self.inner.fragment_ions {
            for (key, (peptide_ions, mz_spectra)) in fragment_ions.iter() {
                let peptide_ions = PyPeptideProductIonSeriesCollection { inner: peptide_ions.clone() };
                let mz_spectra = mz_spectra.iter().map(|x| PyMzSpectrum::from_inner(x.clone())).collect::<Vec<_>>();
                result.insert(*key, (peptide_ions, mz_spectra));
            }
        }
        result
    }
}

/// Lazy frame builder for DIA experiments.
///
/// This builder only loads peptide/ion data for the frames being built,
/// reducing memory usage for large simulations.
#[pyclass(unsendable)]
pub struct PyTimsTofLazyFrameBuilderDIA {
    pub inner: TimsTofLazyFrameBuilderDIA,
}

#[pymethods]
impl PyTimsTofLazyFrameBuilderDIA {
    #[new]
    #[pyo3(signature = (db_path, num_threads=4))]
    pub fn new(db_path: &str, num_threads: usize) -> Self {
        let path = std::path::Path::new(db_path);
        PyTimsTofLazyFrameBuilderDIA {
            inner: TimsTofLazyFrameBuilderDIA::new(path, num_threads).unwrap(),
        }
    }

    /// Build frames for the specified frame IDs using lazy loading.
    ///
    /// Only loads peptide/ion data for the frames being built, then releases it.
    #[pyo3(signature = (
        frame_ids,
        fragmentation=true,
        mz_noise_precursor=true,
        uniform=false,
        precursor_noise_ppm=5.0,
        mz_noise_fragment=true,
        fragment_noise_ppm=5.0,
        right_drag=false
    ))]
    pub fn build_frames_lazy(
        &self,
        frame_ids: Vec<u32>,
        fragmentation: bool,
        mz_noise_precursor: bool,
        uniform: bool,
        precursor_noise_ppm: f64,
        mz_noise_fragment: bool,
        fragment_noise_ppm: f64,
        right_drag: bool,
    ) -> Vec<PyTimsFrame> {
        let frames = self.inner.build_frames_lazy(
            frame_ids,
            fragmentation,
            mz_noise_precursor,
            uniform,
            precursor_noise_ppm,
            mz_noise_fragment,
            fragment_noise_ppm,
            right_drag,
        );
        frames.into_iter().map(|x| PyTimsFrame::from_inner(x)).collect()
    }

    /// Get total number of frames.
    pub fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    /// Get all frame IDs.
    pub fn frame_ids(&self) -> Vec<u32> {
        self.inner.frame_ids()
    }

    /// Get precursor frame IDs.
    pub fn precursor_frame_ids(&self) -> Vec<u32> {
        self.inner.precursor_frame_ids()
    }

    /// Get fragment frame IDs.
    pub fn fragment_frame_ids(&self) -> Vec<u32> {
        self.inner.fragment_frame_ids()
    }

    /// Get collision energy for a specific frame and scan.
    pub fn get_collision_energy(&self, frame_id: i32, scan_id: i32) -> f64 {
        self.inner.get_collision_energy(frame_id, scan_id)
    }
}

#[pymodule]
pub fn py_simulation(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIsotopeTransmissionConfig>()?;
    m.add_class::<PyTimsTofSyntheticsDataHandle>()?;
    m.add_class::<PyTimsTofSyntheticsPrecursorFrameBuilder>()?;
    m.add_class::<PyTimsTofSyntheticsFrameBuilderDIA>()?;
    m.add_class::<PyTimsTofSyntheticsFrameBuilderDDA>()?;
    m.add_class::<PyTimsTofLazyFrameBuilderDIA>()?;
    Ok(())
}