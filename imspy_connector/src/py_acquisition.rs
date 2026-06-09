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

    /// Number of events in one cycle (1 MS1 + N MS2 frames) — the Bruker
    /// `precursor_every` (frames per cycle).
    pub fn cycle_length(&self) -> usize {
        self.inner.cycle.len()
    }

    /// Number of MS2 frames (window groups) per cycle.
    pub fn n_ms2_frames(&self) -> usize {
        self.inner.cycle.len() - self.inner.ms1_count()
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

/// Thermo `.raw` writer (template-mutation), exposed to the Python simulator.
///
/// Open a real Astral/Orbitrap `.raw` template, push MS1/MS2 scans in acquisition
/// order, and `finalize`. `Replace` rewrites template scans; `Overlay`
/// (`overlay_merge_tol_ppm`) adds simulated peaks onto the real template signal
/// (real⊕sim). Requires the `thermo` feature.
#[cfg(feature = "thermo")]
#[pyclass]
pub struct PyThermoRawWriter {
    inner: rustdf::sim::acquisition::ThermoRawWriter,
}

#[cfg(feature = "thermo")]
#[pymethods]
impl PyThermoRawWriter {
    /// Open `template` and prepare to author into `out`. Pass
    /// `overlay_merge_tol_ppm` to overlay onto the real template signal instead
    /// of replacing it.
    #[staticmethod]
    #[pyo3(signature = (template, out, overlay_merge_tol_ppm=None))]
    pub fn from_template(
        py: Python<'_>,
        template: &str,
        out: &str,
        overlay_merge_tol_ppm: Option<f64>,
    ) -> PyResult<Self> {
        use rustdf::sim::acquisition::{ThermoRawWriter, WriteMode};
        let mut w = py
            .allow_threads(|| ThermoRawWriter::from_template(template, out))
            .map_err(PyErr::from)?;
        if let Some(tol) = overlay_merge_tol_ppm {
            w = w.with_mode(WriteMode::Overlay { merge_tol_ppm: tol });
        }
        Ok(Self { inner: w })
    }

    /// Template (MS1, MS2) scan capacity, so a run can be checked to fit.
    pub fn capacity(&self) -> (usize, usize) {
        self.inner.capacity()
    }

    /// Author an MS1 scan from peak arrays (intensity cast to f32).
    pub fn write_ms1(
        &mut self,
        retention_time: f64,
        mz: Vec<f64>,
        intensity: Vec<f64>,
    ) -> PyResult<()> {
        use rustdf::sim::acquisition::{AcquisitionWriter, ScanDescriptor};
        if mz.len() != intensity.len() {
            return Err(PyValueError::new_err("mz and intensity must have equal length"));
        }
        let peaks = mz.iter().zip(&intensity).map(|(&m, &i)| (m, i as f32)).collect();
        let d = ScanDescriptor {
            ms_level: 1,
            retention_time,
            isolation: None,
            peaks,
        };
        self.inner.write_scan(&d).map_err(PyErr::from)
    }

    /// Author an MS2 scan with its isolation window + collision energy.
    pub fn write_ms2(
        &mut self,
        retention_time: f64,
        isolation_center: f64,
        isolation_width: f64,
        collision_energy: f64,
        mz: Vec<f64>,
        intensity: Vec<f64>,
    ) -> PyResult<()> {
        use rustdf::sim::acquisition::{AcquisitionWriter, IsolationWindow, ScanDescriptor};
        if mz.len() != intensity.len() {
            return Err(PyValueError::new_err("mz and intensity must have equal length"));
        }
        let peaks = mz.iter().zip(&intensity).map(|(&m, &i)| (m, i as f32)).collect();
        let d = ScanDescriptor {
            ms_level: 2,
            retention_time,
            isolation: Some(IsolationWindow {
                center_mz: isolation_center,
                width_mz: isolation_width,
                collision_energy,
            }),
            peaks,
        };
        self.inner.write_scan(&d).map_err(PyErr::from)
    }

    /// Recompute the checksum and write the `.raw` to disk.
    pub fn finalize(&mut self, py: Python<'_>) -> PyResult<()> {
        use rustdf::sim::acquisition::AcquisitionWriter;
        py.allow_threads(|| self.inner.finalize()).map_err(PyErr::from)
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
    #[cfg(feature = "thermo")]
    m.add_class::<PyThermoRawWriter>()?;
    m.add_function(wrap_pyfunction!(has_thermo, m)?)?;
    m.add_function(wrap_pyfunction!(has_sciex, m)?)?;
    Ok(())
}
