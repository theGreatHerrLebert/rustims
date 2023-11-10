mod py_dataset;
mod py_mz_spectrum;
mod py_tims_frame;
mod py_tims_slice;
mod py_dda;
mod py_dia;

use pyo3::prelude::*;
use crate::py_dataset::PyTimsDataset;
use crate::py_mz_spectrum::{PyMzSpectrum, PyIndexedMzSpectrum, PyTimsSpectrum, PyMzSpectrumVectorized};
use crate::py_tims_frame::{PyTimsFrame, PyTimsFrameVectorized, PyRawTimsFrame};
use crate::py_tims_slice::{PyTimsPlane, PyTimsSlice, PyTimsSliceVectorized};
use crate::py_dda::{PyTimsDatasetDDA, PyTimsFragmentDDA};

#[pymodule]
fn imspy_connector(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTimsDataset>()?;
    m.add_class::<PyTimsDatasetDDA>()?;
    m.add_class::<PyMzSpectrum>()?;
    m.add_class::<PyMzSpectrumVectorized>()?;
    m.add_class::<PyIndexedMzSpectrum>()?;
    m.add_class::<PyTimsSpectrum>()?;
    m.add_class::<PyTimsFrame>()?;
    m.add_class::<PyRawTimsFrame>()?;
    m.add_class::<PyTimsFrameVectorized>()?;
    m.add_class::<PyTimsSlice>()?;
    m.add_class::<PyTimsSliceVectorized>()?;
    m.add_class::<PyTimsPlane>()?;
    m.add_class::<PyTimsFragmentDDA>()?;
    Ok(())
}
