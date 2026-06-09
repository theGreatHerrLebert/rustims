use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustdf::sim::scheme::AcquisitionScheme;

/// Python wrapper over the vendor-neutral DIA acquisition scheme
/// (`rustdf::sim::scheme::AcquisitionScheme`).
///
/// Exposes the vendor extractors and the Bruker backward-compat adapter, so the
/// Python simulator can build its DIA layout from a real Bruker/Thermo/SCIEX run
/// (or an injected window table) and render the Bruker `DiaFrameMsMsWindows` /
/// `DiaFrameMsMsInfo` tables. The `from_thermo_raw`/`from_sciex_wiff` constructors
/// require the connector to be built with the `thermo`/`sciex` feature.
#[pyclass]
pub struct PyAcquisitionScheme {
    pub inner: AcquisitionScheme,
}

fn val_err<E: ToString>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pymethods]
impl PyAcquisitionScheme {
    /// Extract the scheme from a Bruker timsTOF DIA `.d` folder.
    #[staticmethod]
    pub fn from_bruker_d(path: &str) -> PyResult<Self> {
        AcquisitionScheme::from_bruker_d(path)
            .map(|inner| Self { inner })
            .map_err(val_err)
    }

    /// Extract from a Thermo Orbitrap `.raw` (requires the `thermo` feature).
    #[cfg(feature = "thermo")]
    #[staticmethod]
    pub fn from_thermo_raw(path: &str) -> PyResult<Self> {
        AcquisitionScheme::from_thermo_raw(path)
            .map(|inner| Self { inner })
            .map_err(val_err)
    }

    /// Extract from a SCIEX `.wiff` SWATH method (requires the `sciex` feature).
    /// SCIEX rolling CE isn't stored per-window, so it's left `Unknown` unless a
    /// linear rolling-CE model is supplied via `(ce_intercept, ce_slope_per_mz)`.
    #[cfg(feature = "sciex")]
    #[staticmethod]
    #[pyo3(signature = (path, cycle_time_s, gradient_length_s, ce_intercept=None, ce_slope_per_mz=None))]
    pub fn from_sciex_wiff(
        path: &str,
        cycle_time_s: f64,
        gradient_length_s: f64,
        ce_intercept: Option<f64>,
        ce_slope_per_mz: Option<f64>,
    ) -> PyResult<Self> {
        use rustdf::sim::scheme::CollisionEnergyPolicy;
        let ce = match (ce_intercept, ce_slope_per_mz) {
            (Some(intercept), Some(slope_per_mz)) => CollisionEnergyPolicy::Linear {
                intercept,
                slope_per_mz,
            },
            _ => CollisionEnergyPolicy::Unknown,
        };
        AcquisitionScheme::from_sciex_wiff(path, cycle_time_s, gradient_length_s, ce)
            .map(|inner| Self { inner })
            .map_err(val_err)
    }

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

    /// Bruker `DiaFrameMsMsWindows` as column lists, ready for a DataFrame:
    /// `(window_group, scan_start, scan_end, isolation_mz, isolation_width, collision_energy)`.
    pub fn to_bruker_windows(
        &self,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let rows = self.inner.to_bruker_windows().map_err(val_err)?;
        let mut wg = Vec::with_capacity(rows.len());
        let mut ss = Vec::with_capacity(rows.len());
        let mut se = Vec::with_capacity(rows.len());
        let mut mz = Vec::with_capacity(rows.len());
        let mut wd = Vec::with_capacity(rows.len());
        let mut ce = Vec::with_capacity(rows.len());
        for r in &rows {
            wg.push(r.window_group);
            ss.push(r.scan_num_begin);
            se.push(r.scan_num_end);
            mz.push(r.isolation_mz);
            wd.push(r.isolation_width);
            ce.push(r.collision_energy);
        }
        Ok((wg, ss, se, mz, wd, ce))
    }

    /// Bruker `DiaFrameMsMsInfo` as column lists for a `num_frames`-frame run:
    /// `(frame, window_group)`.
    pub fn to_bruker_info(&self, num_frames: u32) -> PyResult<(Vec<u32>, Vec<u32>)> {
        let rows = self.inner.to_bruker_info(num_frames).map_err(val_err)?;
        let mut frame = Vec::with_capacity(rows.len());
        let mut wg = Vec::with_capacity(rows.len());
        for r in &rows {
            frame.push(r.frame_id);
            wg.push(r.window_group);
        }
        Ok((frame, wg))
    }
}

#[pymodule]
pub fn py_acquisition(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAcquisitionScheme>()?;
    Ok(())
}
