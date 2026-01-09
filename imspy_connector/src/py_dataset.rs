use mscore::timstof::frame::TimsFrame;
use pyo3::prelude::*;
use rustdf::data::dataset::TimsDataset;
use rustdf::data::utility::{zstd_compress, zstd_decompress, reconstruct_compressed_data, compress_collection, parse_decompressed_bruker_binary_data};

use crate::py_tims_frame::{PyTimsFrame};
use crate::py_tims_slice::PyTimsSlice;
use numpy::{IntoPyArray, PyArray1};
use pyo3::types::PyList;
use pyo3::{PyResult, Python, PyObject};
use rustdf::data::acquisition::AcquisitionMode;
use rustdf::data::handle::TimsData;

#[pyclass]
pub struct PyTimsDataset {
    inner: TimsDataset,
}

#[pymethods]
impl PyTimsDataset {
    #[new]
    pub fn new(data_path: &str, bruker_lib_path: &str, in_memory: bool, use_bruker_sdk: bool) -> PyResult<Self> {
        let dataset = TimsDataset::new(bruker_lib_path, data_path, in_memory, use_bruker_sdk)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(PyTimsDataset { inner: dataset })
    }

    pub fn get_frame(&self, frame_id: u32) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.get_frame(frame_id) }
    }

    pub fn get_slice(&self, frame_ids: Vec<u32>, num_threads: usize) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.get_slice(frame_ids, num_threads) }
    }

    pub fn get_acquisition_mode(&self) -> String {
        self.inner.get_acquisition_mode().to_string()
    }

    pub fn get_acquisition_mode_numeric(&self) -> i32 {
        self.inner.get_acquisition_mode().to_i32()
    }

    pub fn get_frame_count(&self) -> i32 {
        self.inner.get_frame_count()
    }

    pub fn get_data_path(&self) -> &str {
        self.inner.get_data_path()
    }

    pub fn frame_count(&self) -> i32 {
        self.inner.get_frame_count()
    }

    pub fn mz_to_tof(&self, frame_id: u32, mz_values: Vec<f64>) -> Vec<u32> {
        self.inner.loader.get_index_converter().mz_to_tof(frame_id, &mz_values.clone())
    }

    pub fn tof_to_mz(&self, frame_id: u32, tof_values: Vec<u32>) -> Vec<f64> {
        self.inner.loader.get_index_converter().tof_to_mz(frame_id, &tof_values.clone())
    }

    pub fn scan_to_inverse_mobility(&self, frame_id: u32, scan_values: Vec<u32>) -> Vec<f64> {
        self.inner.loader.get_index_converter().scan_to_inverse_mobility(frame_id, &scan_values.clone())
    }

    pub fn inverse_mobility_to_scan(&self, frame_id: u32, inverse_mobility_values: Vec<f64>) -> Vec<u32> {
        self.inner.loader.get_index_converter().inverse_mobility_to_scan(frame_id, &inverse_mobility_values.clone())
    }

    #[staticmethod]
    pub fn compress_bytes_zstd(bytes: Vec<u8>, compression_level: i32) -> Vec<u8> {
        let result = zstd_compress(&bytes, compression_level).unwrap();
        result
    }

    #[staticmethod]
    pub fn decompress_bytes_zstd(bytes: Vec<u8>) -> Vec<u8> {
        let result = zstd_decompress(&bytes).unwrap();
        result
    }

    pub fn scan_tof_intensities_to_compressed_u8(&self, py: Python<'_>, scan_values: Vec<u32>, tof_values: Vec<u32>, intensity_values: Vec<u32>, total_scans: u32, compression_level: i32) -> Py<PyArray1<u8>> {
        let result = reconstruct_compressed_data(scan_values, tof_values, intensity_values, total_scans, compression_level).unwrap();
        result.into_pyarray_bound(py).unbind()
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
            expanded_scans.into_pyarray_bound(py).unbind(),
            tof.into_pyarray_bound(py).unbind(),
            intensities.into_pyarray_bound(py).unbind(),
        ))
    }

    #[pyo3(signature = (frame, total_scans, use_frame_id=None, compression_level=None))]
    pub fn compress_frame(&self, py: Python<'_>, frame: PyTimsFrame, total_scans: u32, use_frame_id: Option<bool>, compression_level: Option<i32>) -> PyResult<PyObject> {

        let compression_level = compression_level.unwrap_or(0);

        let frame_id = if use_frame_id.unwrap_or(false) {
            frame.inner.frame_id
        } else {
            1
        };

        let mz = frame.inner.ims_frame.mz.clone();
        let tof = self.inner.loader.get_index_converter().mz_to_tof(frame_id as u32, &mz);
        let inv_mob = frame.inner.ims_frame.mobility.clone();
        let scan = self.inner.loader.get_index_converter().inverse_mobility_to_scan(1, &inv_mob).iter().map(|x| *x as u32).collect::<Vec<_>>();

        let compressed_frame = reconstruct_compressed_data(
            scan,
            tof,
            frame.inner.ims_frame.intensity.clone().iter().map(|x| *x as u32).collect::<Vec<_>>(),
            total_scans, compression_level).unwrap();

        let py_array: Bound<'_, PyArray1<u8>> = compressed_frame.into_pyarray_bound(py);
        Ok(py_array.unbind().into())
    }

    #[pyo3(signature = (frames, total_scans, num_threads, use_frame_id=None, compression_level=None))]
    pub fn compress_frames(&self, py: Python<'_>, frames: Vec<PyTimsFrame>, total_scans: u32, num_threads: usize, use_frame_id: Option<bool>, compression_level: Option<i32>) -> PyResult<PyObject> {

        let mut filled_tims_frames = Vec::with_capacity(frames.len());
        let compression_level = compression_level.unwrap_or(0);

        // translate mz to tof, inv_mob to scan for each frame
        for frame in frames {
            let frame_id = if use_frame_id.unwrap_or(false) {
                frame.inner.frame_id
            } else {
                1
            };

            let mz = frame.inner.ims_frame.mz.clone();
            let tof = self.inner.loader.get_index_converter().mz_to_tof(frame_id as u32, &mz).iter().map(|x| *x as i32).collect::<Vec<_>>();
            let inv_mob = frame.inner.ims_frame.mobility.clone();
            let scan = self.inner.loader.get_index_converter().inverse_mobility_to_scan(1, &inv_mob).iter().map(|x| *x as i32).collect::<Vec<_>>();

            let frame = TimsFrame::new(
                frame.inner.frame_id,
                frame.inner.ms_type.clone(),
                frame.inner.ims_frame.retention_time,
                scan,
                inv_mob,
                tof,
                mz,
                frame.inner.ims_frame.intensity,
            );

            filled_tims_frames.push(frame);
        }

        // compress the frames
        let compressed_frames = compress_collection(filled_tims_frames, total_scans, compression_level, num_threads);

        // convert the compressed frames to a python list of numpy arrays
        let py_list = PyList::empty_bound(py);

        for frame in compressed_frames {
            let np_array = frame.into_pyarray_bound(py).unbind();
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

// pub fn get_peak_cnts(&self, total_scans: u32, scans: Vec<u32>) -> Vec<u32> {
//         self.inner.get_peak_cnts(total_scans, &scans)
//     }

#[pyfunction]
pub fn get_peak_cnts(total_scans: u32, scans: Vec<u32>) -> Vec<u32> {
    rustdf::data::utility::get_peak_cnts(total_scans, &scans)
}

// pub fn modify_tofs(tofs: &mut [u32], scans: &[u32]) {
#[pyfunction]
pub fn modify_tofs(tofs: Vec<u32>, scans: Vec<u32>) -> Vec<u32> {
    // Create a mutable copy of `tofs` that can be modified.
    let mut mutable_tofs = tofs.clone();

    // Directly pass the mutable reference of the cloned `tofs`.
    rustdf::data::utility::modify_tofs(&mut mutable_tofs, &scans);

    // Return the modified `mutable_tofs`.
    mutable_tofs
}
#[pyfunction]
pub fn get_realdata(peak_cnts: Vec<u32>, interleaved: Vec<u32>) -> Vec<u8> {
    rustdf::data::utility::get_realdata(&peak_cnts, &interleaved)
}

#[pyfunction]
pub fn get_data_for_compression(tofs: Vec<u32>, scans: Vec<u32>, intensities: Vec<u32>, max_scans: u32) -> Vec<u8> {
    rustdf::data::utility::get_data_for_compression(&tofs, &scans, &intensities, max_scans)
}

#[pyfunction]
pub fn get_data_for_compression_par(tofs: Vec<Vec<u32>>, scans: Vec<Vec<u32>>, intensities: Vec<Vec<u32>>, max_scans: u32, num_threads: usize) -> Vec<Vec<u8>> {
    rustdf::data::utility::get_data_for_compression_par(tofs, scans, intensities, max_scans, num_threads)
}

#[pymodule]
pub fn py_dataset(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimsDataset>()?;
    m.add_class::<PyAcquisitionMode>()?;
    m.add_function(wrap_pyfunction!(get_peak_cnts, m)?)?;
    m.add_function(wrap_pyfunction!(modify_tofs, m)?)?;
    m.add_function(wrap_pyfunction!(get_realdata, m)?)?;
    m.add_function(wrap_pyfunction!(get_data_for_compression, m)?)?;
    m.add_function(wrap_pyfunction!(get_data_for_compression_par, m)?)?;
    Ok(())
}