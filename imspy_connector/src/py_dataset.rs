use pyo3::prelude::*;
use rustdf::data::dataset::TimsDataset;
use rustdf::data::handle::{TimsData, AcquisitionMode, zstd_compress, zstd_decompress, parse_decompressed_bruker_binary_data, reconstruct_compressed_data, compress_collection};

use crate::py_tims_frame::{PyTimsFrame};
use crate::py_tims_slice::PyTimsSlice;
use numpy::{IntoPyArray, PyArray1};
use pyo3::types::PyList;
use pyo3::{PyResult, Python, PyObject};

#[pyclass]
pub struct PyTimsDataset {
    inner: TimsDataset,
}

#[pymethods]
impl PyTimsDataset {
    #[new]
    pub fn new(data_path: &str, bruker_lib_path: &str) -> Self {
        let dataset = TimsDataset::new(bruker_lib_path, data_path);
        PyTimsDataset { inner: dataset }
    }

    pub fn get_frame(&self, frame_id: u32) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.get_frame(frame_id) }
    }

    pub fn get_slice(&self, frame_ids: Vec<u32>) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.get_slice(frame_ids) }
    }

    pub fn get_acquisition_mode(&self) -> String {
        self.inner.get_aquisition_mode().to_string()
    }

    pub fn get_acquisition_mode_numeric(&self) -> i32 {
        self.inner.get_aquisition_mode().to_i32()
    }

    pub fn get_frame_count(&self) -> i32 {
        self.inner.get_frame_count()
    }

    pub fn get_data_path(&self) -> &str {
        self.inner.get_data_path()
    }

    pub fn get_bruker_lib_path(&self) -> &str {
        self.inner.get_bruker_lib_path()
    }

    pub fn frame_count(&self) -> i32 {
        self.inner.get_frame_count()
    }

    // TODO: make this more efficient
    pub fn mz_to_tof(&self, frame_id: u32, mz_values: Vec<f64>) -> Vec<u32> {
        self.inner.mz_to_tof(frame_id, &mz_values.clone())
    }

    // TODO: make this more efficient
    pub fn tof_to_mz(&self, frame_id: u32, tof_values: Vec<u32>) -> Vec<f64> {
        self.inner.tof_to_mz(frame_id, &tof_values.clone())
    }

    #[staticmethod]
    pub fn compress_bytes_zstd(bytes: Vec<u8>) -> Vec<u8> {
        let result = zstd_compress(&bytes).unwrap();
        result
    }

    #[staticmethod]
    pub fn decompress_bytes_zstd(bytes: Vec<u8>) -> Vec<u8> {
        let result = zstd_decompress(&bytes).unwrap();
        result
    }

    #[staticmethod]
    pub fn scan_tof_intensities_to_compressed_u8(py: Python<'_>, scan_values: Vec<u32>, tof_values: Vec<u32>, intensity_values: Vec<u32>, total_scans: u32) -> Py<PyArray1<u8>> {
        let result = reconstruct_compressed_data(scan_values, tof_values, intensity_values, total_scans).unwrap();
        result.into_pyarray(py).to_owned()
    }

    #[staticmethod]
    pub fn u8_to_scan_tof_intensities(py: Python<'_>, data: Vec<u8>) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<u32>>, Py<PyArray1<u32>>)> {
        let (scan_counts, tof, intensities) = parse_decompressed_bruker_binary_data(&data).unwrap();

        // Expand the scan counts
        let mut expanded_scans = Vec::new();
        for (index, &count) in scan_counts.iter().enumerate() {
            expanded_scans.extend(std::iter::repeat(index as u32).take(count as usize));
        }

        Ok((
            expanded_scans.into_pyarray(py).to_owned(),
            tof.into_pyarray(py).to_owned(),
            intensities.into_pyarray(py).to_owned(),
        ))
    }

    #[staticmethod]
    pub fn compress_collection(py: Python<'_>, frames: Vec<PyTimsFrame>, total_scans: u32, num_threads: usize) -> PyResult<PyObject> {
        let compressed_frames = compress_collection(frames.iter().map(|x| x.inner.clone()).collect::<Vec<_>>(), total_scans, num_threads);

        let py_list = PyList::empty(py);

        for frame in compressed_frames {
            let np_array = frame.into_pyarray(py).to_owned();
            py_list.append(np_array)?;
        }

        Ok(py_list.into())
    }
}

#[pyclass]
pub struct PyAcquisitionMode {
    inner: AcquisitionMode,
}

#[pymethods]
impl PyAcquisitionMode {
    #[new]
    pub fn new(acquisition_mode: &str) -> Self {
        PyAcquisitionMode { inner: AcquisitionMode::from(acquisition_mode) }
    }

    #[staticmethod]
    pub fn from_numeric(acquisition_mode: i32) -> Self {
        PyAcquisitionMode { inner: AcquisitionMode::from(acquisition_mode) }
    }

    #[staticmethod]
    pub fn from_string(acquisition_mode: &str) -> Self {
        PyAcquisitionMode { inner: AcquisitionMode::from(acquisition_mode) }
    }

    #[getter]
    pub fn acquisition_mode(&self) -> String {
        self.inner.to_str().to_string()
    }

    #[getter]
    pub fn acquisition_mode_numeric(&self) -> i32 {
        self.inner.to_i32()
    }
}