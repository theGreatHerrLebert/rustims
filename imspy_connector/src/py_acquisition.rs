use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustdf::sim::scheme::AcquisitionScheme;

/// Python wrapper over the vendor-neutral DIA acquisition scheme
/// (`rustdf::sim::scheme::AcquisitionScheme`).
///
/// Exposes the vendor extractors and the Bruker backward-compat adapter, so the
/// Python simulator can build its DIA layout from a real Bruker/Thermo/SCIEX run
/// (or an injected window table) and render the Bruker `DiaFrameMsMsWindows` /
/// `DiaFrameMsMsInfo` tables. The `from_thermo_raw`/`from_sciex_wiff` constructors
/// require the connector to be built with the `thermo`/`sciex` feature
/// (see [`has_thermo`]/[`has_sciex`]).
#[pyclass]
pub struct PyAcquisitionScheme {
    pub inner: AcquisitionScheme,
}

/// Validation/rendering failures (a `String`) -> `ValueError`.
fn val_err<E: ToString>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pymethods]
impl PyAcquisitionScheme {
    /// Extract the scheme from a Bruker timsTOF DIA `.d` folder.
    #[staticmethod]
    pub fn from_bruker_d(py: Python<'_>, path: &str) -> PyResult<Self> {
        // I/O errors map to OSError/FileNotFoundError via PyErr::from; release the
        // GIL while reading the .d's SQLite metadata.
        let inner = py
            .allow_threads(|| AcquisitionScheme::from_bruker_d(path))
            .map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Extract from a Thermo Orbitrap `.raw` (requires the `thermo` feature).
    #[cfg(feature = "thermo")]
    #[staticmethod]
    pub fn from_thermo_raw(py: Python<'_>, path: &str) -> PyResult<Self> {
        let inner = py
            .allow_threads(|| AcquisitionScheme::from_thermo_raw(path))
            .map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Extract from a SCIEX `.wiff` SWATH method (requires the `sciex` feature).
    /// SCIEX rolling CE isn't stored per-window, so it's left `Unknown` unless a
    /// linear rolling-CE model is supplied; supplying only one of the two CE
    /// coefficients is an error (never silently dropped).
    #[cfg(feature = "sciex")]
    #[staticmethod]
    #[pyo3(signature = (path, cycle_time_s, gradient_length_s, *, ce_intercept=None, ce_slope_per_mz=None))]
    pub fn from_sciex_wiff(
        py: Python<'_>,
        path: &str,
        cycle_time_s: f64,
        gradient_length_s: f64,
        ce_intercept: Option<f64>,
        ce_slope_per_mz: Option<f64>,
    ) -> PyResult<Self> {
        use rustdf::sim::scheme::CollisionEnergyPolicy;
        let ce = match (ce_intercept, ce_slope_per_mz) {
            (Some(intercept), Some(slope_per_mz)) => {
                CollisionEnergyPolicy::Linear { intercept, slope_per_mz }
            }
            (None, None) => CollisionEnergyPolicy::Unknown,
            _ => {
                return Err(PyValueError::new_err(
                    "supply both ce_intercept and ce_slope_per_mz, or neither",
                ))
            }
        };
        let inner = py
            .allow_threads(|| {
                AcquisitionScheme::from_sciex_wiff(path, cycle_time_s, gradient_length_s, ce)
            })
            .map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Instrument kind (canonical enum variant name, e.g. `"TimsTofDia"`).
    #[getter]
    pub fn instrument(&self) -> String {
        format!("{:?}", self.inner.instrument)
    }

    #[getter]
    pub fn mz_range(&self) -> (f64, f64) {
        self.inner.mz_range
    }

    #[getter]
    pub fn provenance(&self) -> String {
        format!(
            "{:?}: {}",
            self.inner.provenance.source, self.inner.provenance.notes
        )
    }

    pub fn ms1_count(&self) -> usize {
        self.inner.ms1_count()
    }

    pub fn n_windows(&self) -> usize {
        self.inner.windows().count()
    }

    pub fn num_cycles(&self) -> Option<u64> {
        self.inner.num_cycles()
    }

    pub fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(val_err)
    }

    /// Bruker `DiaFrameMsMsWindows` as a column dict of NumPy arrays, ready for
    /// `pandas.DataFrame(...)`: keys `window_group, scan_start, scan_end,
    /// isolation_mz, isolation_width, collision_energy`.
    pub fn to_bruker_windows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let rows = self.inner.to_bruker_windows().map_err(val_err)?;
        let n = rows.len();
        let (mut wg, mut ss, mut se) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut mz, mut wd, mut ce) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for r in &rows {
            wg.push(r.window_group);
            ss.push(r.scan_num_begin);
            se.push(r.scan_num_end);
            mz.push(r.isolation_mz);
            wd.push(r.isolation_width);
            ce.push(r.collision_energy);
        }
        let d = PyDict::new(py);
        d.set_item("window_group", wg.into_pyarray(py))?;
        d.set_item("scan_start", ss.into_pyarray(py))?;
        d.set_item("scan_end", se.into_pyarray(py))?;
        d.set_item("isolation_mz", mz.into_pyarray(py))?;
        d.set_item("isolation_width", wd.into_pyarray(py))?;
        d.set_item("collision_energy", ce.into_pyarray(py))?;
        Ok(d)
    }

    /// Bruker `DiaFrameMsMsInfo` as a column dict of NumPy arrays for a
    /// `num_frames`-frame run: keys `frame, window_group`.
    pub fn to_bruker_info<'py>(
        &self,
        py: Python<'py>,
        num_frames: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rows = self.inner.to_bruker_info(num_frames).map_err(val_err)?;
        let mut frame = Vec::with_capacity(rows.len());
        let mut wg = Vec::with_capacity(rows.len());
        for r in &rows {
            frame.push(r.frame_id);
            wg.push(r.window_group);
        }
        let d = PyDict::new(py);
        d.set_item("frame", frame.into_pyarray(py))?;
        d.set_item("window_group", wg.into_pyarray(py))?;
        Ok(d)
    }
}

/// Whether this build includes the Thermo `.raw` extractor.
#[pyfunction]
pub fn has_thermo() -> bool {
    cfg!(feature = "thermo")
}

/// Whether this build includes the SCIEX `.wiff` extractor.
#[pyfunction]
pub fn has_sciex() -> bool {
    cfg!(feature = "sciex")
}

#[pymodule]
pub fn py_acquisition(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAcquisitionScheme>()?;
    m.add_function(wrap_pyfunction!(has_thermo, m)?)?;
    m.add_function(wrap_pyfunction!(has_sciex, m)?)?;
    Ok(())
}
