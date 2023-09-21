mod pyhandle;
mod py_mz_spectrum;

use pyo3::prelude::*;
use crate::pyhandle::PyTimsDataset;
use crate::py_mz_spectrum::PyMzSpectrum;

/// A Python module implemented in Rust.
#[pymodule]
fn pyims(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTimsDataset>()?;
    m.add_class::<PyMzSpectrum>()?;
    Ok(())
}