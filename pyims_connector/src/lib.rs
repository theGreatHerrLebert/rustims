mod pyhandle;
mod py_mz_spectrum;
mod py_tims_frame;
mod py_tims_slice;

use pyo3::prelude::*;
use crate::pyhandle::PyTimsDataHandle;
use crate::py_mz_spectrum::{PyMzSpectrum, PyIndexedMzSpectrum, PyTimsSpectrum};
use crate::py_tims_frame::{PyTimsFrame};
use crate::py_tims_slice::{PyTimsSlice};

#[pymodule]
fn pyims_connector(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTimsDataHandle>()?;
    m.add_class::<PyMzSpectrum>()?;
    m.add_class::<PyIndexedMzSpectrum>()?;
    m.add_class::<PyTimsSpectrum>()?;
    m.add_class::<PyTimsFrame>()?;
    m.add_class::<PyTimsSlice>()?;
    Ok(())
}
