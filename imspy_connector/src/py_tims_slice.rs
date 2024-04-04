use pyo3::prelude::*;
use mscore::data::spectrum::{MsType};
use mscore::timstof::slice::{TimsPlane, TimsSlice, TimsSliceVectorized};
use pyo3::types::{PyList};
use numpy::{IntoPyArray, PyArray1};
use crate::py_mz_spectrum::{PyTimsSpectrum};

use crate::py_tims_frame::{PyTimsFrame, PyTimsFrameVectorized};

#[pyclass]
#[derive(Clone)]
pub struct PyTimsSlice {
    pub inner: TimsSlice,
}


#[pymethods]
impl PyTimsSlice {
    #[new]
    pub unsafe fn new(
        _py: Python,
        frame_ids: &PyArray1<i32>,
        scans: &PyArray1<i32>,
        tofs: &PyArray1<i32>,
        retention_times: &PyArray1<f64>,
        mobilities: &PyArray1<f64>,
        mzs: &PyArray1<f64>,
        intensities: &PyArray1<f64>,
    ) -> PyResult<Self> {
        let frame_ids_vec = frame_ids.as_slice()?.to_vec();
        let scans_vec = scans.as_slice()?.to_vec();
        let tofs_vec = tofs.as_slice()?.to_vec();
        let retention_times_vec = retention_times.as_slice()?.to_vec();
        let mobilities_vec = mobilities.as_slice()?.to_vec();
        let mzs_vec = mzs.as_slice()?.to_vec();
        let intensities_vec = intensities.as_slice()?.to_vec();

        // Now call your Rust function from_flat_slice
        let tims_slice = TimsSlice::from_flat_slice(
            frame_ids_vec,
            scans_vec,
            tofs_vec,
            retention_times_vec,
            mobilities_vec,
            mzs_vec,
            intensities_vec
        );

        Ok(PyTimsSlice { inner: tims_slice })
    }
    #[getter]
    pub fn first_frame_id(&self) -> i32 { self.inner.frames.first().unwrap().frame_id }

    #[getter]
    pub fn last_frame_id(&self) -> i32 { self.inner.frames.last().unwrap().frame_id }

    #[getter]
    pub fn frame_count(&self) -> i32 { self.inner.frames.len() as i32 }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, inv_mob_min: f64, inv_mob_max: f64, intensity_min: f64, intensity_max: f64, num_threads: usize) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.filter_ranged(mz_min, mz_max, scan_min, scan_max, inv_mob_min, inv_mob_max, intensity_min, intensity_max, num_threads) }
    }

    pub fn filter_ranged_ms_type_specific(&self,
                                          mz_min_ms1: f64,
                                          mz_max_ms1: f64,
                                          scan_min_ms1: i32,
                                          scan_max_ms1: i32,
                                          inv_mob_min_ms1: f64,
                                          inv_mob_max_ms1: f64,
                                          intensity_min_ms1: f64,
                                          intensity_max_ms1: f64,
                                          mz_min_ms2: f64,
                                          mz_max_ms2: f64,
                                          scan_min_ms2: i32,
                                          scan_max_ms2: i32,
                                          inv_mob_min_ms2: f64,
                                          inv_mob_max_ms2: f64,
                                          intensity_min_ms2: f64,
                                          intensity_max_ms2: f64,
                                          num_threads: usize) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.filter_ranged_ms_type_specific(
            mz_min_ms1, mz_max_ms1, scan_min_ms1, scan_max_ms1, inv_mob_min_ms1, inv_mob_max_ms1, intensity_min_ms1, intensity_max_ms1,
            mz_min_ms2, mz_max_ms2, scan_min_ms2, scan_max_ms2, inv_mob_min_ms2, inv_mob_max_ms2, intensity_min_ms2, intensity_max_ms2,
            num_threads) }
    }

    pub fn get_frames(&self, py: Python) -> PyResult<Py<PyList>> {
        let frames = &self.inner.frames;
        let list: Py<PyList> = PyList::empty(py).into();

        for frame in frames {
            let py_tims_frame = Py::new(py, PyTimsFrame { inner: frame.clone() })?;
            list.as_ref(py).append(py_tims_frame)?;
        }

        Ok(list.into())
    }

    pub fn get_precursors(&self) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.get_slice_by_type(MsType::Precursor) }
    }

    pub fn get_fragments_dda(&self) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.get_slice_by_type(MsType::FragmentDda) }
    }

    pub fn get_fragments_dia(&self) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.get_slice_by_type(MsType::FragmentDia) }
    }

    pub fn to_windows(&self, py: Python, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64, num_threads: usize) -> PyResult<Py<PyList>> {

        let windows = self.inner.to_windows(window_length, overlapping, min_peaks, min_intensity, num_threads);
        let list: Py<PyList> = PyList::empty(py).into();

        for window in windows {
            let py_mz_spectrum = Py::new(py, PyTimsSpectrum { inner: window })?;
            list.as_ref(py).append(py_mz_spectrum)?;
        }

        Ok(list.into())
    }

    pub fn to_dense_windows(&self, window_length: f64, overlapping: bool, min_peaks: usize, min_intensity: f64, resolution: i32, num_threads: usize) -> Vec<(Vec<f64>, Vec<i32>, Vec<i32>, usize, usize)> {
        self.inner.to_dense_windows(window_length, overlapping, min_peaks, min_intensity, resolution, num_threads)
    }

    pub fn get_frame_at_index(&self, index: i32) -> PyTimsFrame {
        PyTimsFrame { inner: self.inner.frames[index as usize].clone() }
    }

    pub fn to_resolution(&self, resolution: i32, num_threads: usize) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.to_resolution(resolution, num_threads) }
    }

    fn to_arrays(&self, py: Python) -> PyResult<(PyObject, PyObject, PyObject, PyObject, PyObject, PyObject, PyObject)> {

        let flat_frame = self.inner.flatten();

        let frame_ids_np = flat_frame.frame_ids.into_pyarray(py);
        let scans_np = flat_frame.scans.into_pyarray(py);
        let tofs_np = flat_frame.tofs.into_pyarray(py);
        let retention_times_np = flat_frame.retention_times.into_pyarray(py);
        let mobilities_np = flat_frame.mobilities.into_pyarray(py);
        let mzs_np = flat_frame.mzs.into_pyarray(py);
        let intensities_np = flat_frame.intensities.into_pyarray(py);

        Ok((frame_ids_np.to_object(py), scans_np.to_object(py), tofs_np.to_object(py),
            retention_times_np.to_object(py), mobilities_np.to_object(py), mzs_np.to_object(py),
            intensities_np.to_object(py)))
    }

    pub fn to_tims_planes(&self, py: Python, tof_max_value: i32, num_chunks: i32, num_threads: i32) -> PyResult<Py<PyList>> {

        let planes = self.inner.to_tims_planes(tof_max_value, num_chunks, num_threads as usize);
        let list: Py<PyList> = PyList::empty(py).into();

        for plane in planes {
            let py_plane = Py::new(py, PyTimsPlane { inner: plane })?;
            list.as_ref(py).append(py_plane)?;
        }

        Ok(list.into())
    }

    pub fn vectorized(&self, resolution: i32, num_threads: usize) -> PyTimsSliceVectorized {
        let vectorized = self.inner.vectorized(resolution, num_threads);
        let py_vectorized = PyTimsSliceVectorized {
            inner: vectorized,
        };
        py_vectorized
    }

    #[staticmethod]
    pub fn from_frames(frames: Vec<PyTimsFrame>) -> PyTimsSlice {
        PyTimsSlice { inner: TimsSlice::new(frames.iter().map(|frame| frame.inner.clone()).collect()) }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyTimsSliceVectorized {
    pub inner: TimsSliceVectorized,
}

#[pymethods]
impl PyTimsSliceVectorized {
    #[getter]
    pub fn get_vectorized_frames(&self, py: Python) -> PyResult<Py<PyList>> {
        let frames = &self.inner.frames;
        let list: Py<PyList> = PyList::empty(py).into();

        for frame in frames {
            let py_tims_frame = Py::new(py, PyTimsFrameVectorized { inner: frame.clone() })?;
            list.as_ref(py).append(py_tims_frame)?;
        }

        Ok(list.into())
    }

    pub fn get_frame_at_index(&self, index: i32) -> PyTimsFrameVectorized {
        PyTimsFrameVectorized { inner: self.inner.frames[index as usize].clone() }
    }

    #[getter]
    pub fn frame_count(&self) -> i32 {
        self.inner.frames.len() as i32
    }

    #[getter]
    pub fn first_frame_id(&self) -> i32 {
        self.inner.frames.first().unwrap().frame_id
    }

    #[getter]
    pub fn last_frame_id(&self) -> i32 {
        self.inner.frames.last().unwrap().frame_id
    }


    pub fn to_arrays(&self, py: Python) -> PyResult<(PyObject, PyObject, PyObject, PyObject, PyObject, PyObject, PyObject)> {

        let flat_frame = self.inner.flatten();

        let frame_ids_np = flat_frame.frame_ids.into_pyarray(py);
        let scans_np = flat_frame.scans.into_pyarray(py);
        let tofs_np = flat_frame.tofs.into_pyarray(py);
        let retention_times_np = flat_frame.retention_times.into_pyarray(py);
        let mobilities_np = flat_frame.mobilities.into_pyarray(py);
        let indices_np = flat_frame.indices.into_pyarray(py);
        let intensities_np = flat_frame.intensities.into_pyarray(py);

        Ok((frame_ids_np.to_object(py), scans_np.to_object(py), tofs_np.to_object(py),
            retention_times_np.to_object(py), mobilities_np.to_object(py), indices_np.to_object(py),
            intensities_np.to_object(py)))
    }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, inv_mob_min: f64, inv_mob_max: f64, intensity_min: f64, intensity_max: f64, num_threads: usize) -> PyTimsSliceVectorized {
        PyTimsSliceVectorized { inner: self.inner.filter_ranged(mz_min, mz_max, scan_min, scan_max, inv_mob_min, inv_mob_max, intensity_min, intensity_max, num_threads) }
    }
}

#[pyclass]
pub struct PyTimsPlane {
    pub inner: TimsPlane,
}

#[pymethods]
impl PyTimsPlane {

    #[getter]
    pub fn mz_mean(&self) -> f64 {
        self.inner.mz_mean
    }
    #[getter]
    pub fn mz_std(&self) -> f64 {
        self.inner.mz_std
    }

    #[getter]
    pub fn tof_mean(&self) -> f64 {
        self.inner.tof_mean
    }

    #[getter]
    pub fn tof_std(&self) -> f64 {
        self.inner.tof_std
    }

    #[getter]
    pub fn scans(&self, py: Python) -> Py<PyArray1<i32>> {
        self.inner.scan.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn mobilities(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.mobility.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn frame_ids(&self, py: Python) -> Py<PyArray1<i32>> {
        self.inner.frame_id.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn retention_times(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.retention_time.clone().into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn intensity(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.intensity.clone().into_pyarray(py).to_owned()
    }
}

#[pymodule]
pub fn tims_slice(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTimsSlice>()?;
    m.add_class::<PyTimsSliceVectorized>()?;
    m.add_class::<PyTimsPlane>()?;
    Ok(())
}
