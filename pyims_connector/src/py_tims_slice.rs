use pyo3::prelude::*;
use mscore::TimsSlice;
use pyo3::types::PyList;

use crate::py_tims_frame::{PyTimsFrame};

#[pyclass]
pub struct PySliceIterator {
    inner: PyTimsSlice,
    index: usize,
}

#[pymethods]
impl PySliceIterator {
    #[new]
    pub fn new(inner: PyTimsSlice) -> Self {
        PySliceIterator {
            inner,
            index: 0,
        }
    }

    fn __iter__(slf: PyRef<PySliceIterator>) -> Py<PySliceIterator> {
        slf.into()
    }

    fn __next__(mut slf: PyRefMut<PySliceIterator>) -> PyResult<Option<PyTimsFrame>> {
        if slf.index < slf.inner.inner.frames.len() {
            let index = slf.index;
            slf.index += 1;

            let item = &slf.inner.inner.frames[index];
            let py_item = PyTimsFrame { inner: item.clone() };
            Ok(Some(py_item))
        } else {
            Ok(None) // End of the iterator, return None instead of using PyStopIteration.
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyTimsSlice {
    pub inner: TimsSlice,
}


#[pymethods]
impl PyTimsSlice {
    #[getter]
    pub fn first_frame_id(&self) -> i32 { self.inner.frames.first().unwrap().frame_id }

    #[getter]
    pub fn last_frame_id(&self) -> i32 { self.inner.frames.last().unwrap().frame_id }

    pub fn filter_ranged(&self, mz_min: f64, mz_max: f64, scan_min: i32, scan_max: i32, intensity_min: f64) -> PyTimsSlice {
        PyTimsSlice { inner: self.inner.filter_ranged(mz_min, mz_max, scan_min, scan_max, intensity_min) }
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

    fn iter(&mut self) -> PySliceIterator {
        PySliceIterator {
            inner: self.clone(),
            index: 0,
        }
    }
}
