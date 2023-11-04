mod py_handle;
mod py_mz_spectrum;
mod py_tims_frame;
mod py_tims_slice;

use pyo3::prelude::*;
use crate::py_handle::PyTimsDataHandle;
use crate::py_mz_spectrum::{PyMzSpectrum, PyIndexedMzSpectrum, PyTimsSpectrum, PyMzSpectrumVectorized};
use crate::py_tims_frame::{PyTimsFrame, PyTimsFrameVectorized};
use crate::py_tims_slice::{PyTimsPlane, PyTimsSlice, PyTimsSliceVectorized};

#[pymodule]
fn pyims_connector(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTimsDataHandle>()?;
    m.add_class::<PyMzSpectrum>()?;
    m.add_class::<PyMzSpectrumVectorized>()?;
    m.add_class::<PyIndexedMzSpectrum>()?;
    m.add_class::<PyTimsSpectrum>()?;
    m.add_class::<PyTimsFrame>()?;
    m.add_class::<PyTimsFrameVectorized>()?;
    m.add_class::<PyTimsSlice>()?;
    m.add_class::<PyTimsSliceVectorized>()?;
    m.add_class::<PyTimsPlane>()?;
    Ok(())
}
