mod py_dataset;
mod py_mz_spectrum;
mod py_tims_frame;
mod py_tims_slice;
mod py_dda;
mod py_dia;
mod py_simulation;
mod py_chemistry;
mod py_quadrupole;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::py_dataset::{PyTimsDataset, PyAcquisitionMode};
use crate::py_mz_spectrum::{PyMzSpectrum, PyIndexedMzSpectrum, PyTimsSpectrum, PyMzSpectrumVectorized};
use crate::py_tims_frame::{PyTimsFrame, PyTimsFrameVectorized, PyRawTimsFrame};
use crate::py_tims_slice::{PyTimsPlane, PyTimsSlice, PyTimsSliceVectorized};
use crate::py_dda::{PyTimsDatasetDDA, PyTimsFragmentDDA};
use crate::py_simulation::{PyTimsTofSyntheticsPrecursorFrameBuilder, PyTimsTofSyntheticsFrameBuilderDIA};
pub use py_chemistry::{generate_precursor_spectrum, generate_precursor_spectra, calculate_monoisotopic_mass, calculate_b_y_ion_series, simulate_charge_state_for_sequence, simulate_charge_states_for_sequences};
use crate::py_quadrupole::{PyTimsTransmissionDIA, apply_transmission, PyTimsTofCollisionEnergyDIA};


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
    m.add_class::<PyAcquisitionMode>()?;
    m.add_class::<PyTimsTofSyntheticsPrecursorFrameBuilder>()?;
    m.add_class::<PyTimsTransmissionDIA>()?;
    m.add_class::<PyTimsTofSyntheticsFrameBuilderDIA>()?;
    m.add_class::<PyTimsTofCollisionEnergyDIA>()?;
    m.add_function(wrap_pyfunction!(generate_precursor_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(generate_precursor_spectra, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_monoisotopic_mass, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_b_y_ion_series, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_charge_state_for_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_charge_states_for_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(apply_transmission, m)?)?;
    Ok(())
}
