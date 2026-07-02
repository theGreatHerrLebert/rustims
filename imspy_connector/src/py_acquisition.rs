use numpy::IntoPyArray;
#[cfg(feature = "thermo")]
use pyo3::exceptions::PyRuntimeError;
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
            .detach(|| AcquisitionScheme::from_bruker_d(path))
            .map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Extract from a Thermo Orbitrap `.raw` (requires the `thermo` feature).
    #[cfg(feature = "thermo")]
    #[staticmethod]
    pub fn from_thermo_raw(py: Python<'_>, path: &str) -> PyResult<Self> {
        let inner = py
            .detach(|| AcquisitionScheme::from_thermo_raw(path))
            .map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// The template's FULL per-scan schedule (build-from-template timeline, P6e):
    /// one row per scan as `(scan, retention_time_s, ms_level, isolation_center_mz,
    /// isolation_width_mz, collision_energy)`. `isolation_*`/`collision_energy` are
    /// `None` for MS1. An Astral run mirrors these frames 1:1 so the trunk is
    /// simulated on the template's REAL (non-uniform) scan retention times rather
    /// than a recomputed fixed cycle. Requires the `thermo` feature.
    #[cfg(feature = "thermo")]
    #[staticmethod]
    pub fn thermo_frame_schedule(
        py: Python<'_>,
        path: &str,
    ) -> PyResult<Vec<(u32, f64, u8, Option<f64>, Option<f64>, Option<f64>)>> {
        let sched = py
            .detach(|| AcquisitionScheme::thermo_frame_schedule(path))
            .map_err(PyErr::from)?;
        Ok(sched
            .into_iter()
            .map(|s| {
                (
                    s.scan,
                    s.retention_time_s,
                    s.ms_level,
                    s.isolation.map(|w| w.center_mz),
                    s.isolation.map(|w| w.width_mz),
                    s.collision_energy,
                )
            })
            .collect())
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
            .detach(|| {
                AcquisitionScheme::from_sciex_wiff(path, cycle_time_s, gradient_length_s, ce)
            })
            .map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Synthesize the per-frame SWATH schedule for a SCIEX `.wiff` over the gradient —
    /// the SCIEX analogue of `thermo_frame_schedule`. Reads the `.wiff` SWATH method,
    /// then expands the cycle (MS1 + one MS2 frame per window) over `num_cycles`. Returns
    /// rows `(scan, retention_time_s, ms_level, isolation_center_mz, isolation_width_mz,
    /// collision_energy)`; MS1 rows have `None` isolation/CE.
    #[cfg(feature = "sciex")]
    #[staticmethod]
    #[pyo3(signature = (path, cycle_time_s, gradient_length_s, *, ce_intercept=None, ce_slope_per_mz=None))]
    pub fn sciex_frame_schedule(
        py: Python<'_>,
        path: &str,
        cycle_time_s: f64,
        gradient_length_s: f64,
        ce_intercept: Option<f64>,
        ce_slope_per_mz: Option<f64>,
    ) -> PyResult<Vec<(u32, f64, u8, Option<f64>, Option<f64>, Option<f64>)>> {
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
        let rows = py
            .detach(|| {
                AcquisitionScheme::from_sciex_wiff(path, cycle_time_s, gradient_length_s, ce)
                    .map(|s| s.dia_frame_schedule())
            })
            .map_err(PyErr::from)?;
        if rows.is_empty() {
            return Err(PyValueError::new_err(
                "empty SWATH schedule (check cycle_time_s/gradient_length_s)",
            ));
        }
        Ok(rows)
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
    /// Set after a successful `finalize`. Guards against write-after-finalize
    /// (which would leave the on-disk `.raw` stale) and double-finalize.
    finalized: bool,
}

/// Caller-fault writer errors (`InvalidInput`/`InvalidData` — e.g. malformed
/// peaks, exhausted template capacity) map to `ValueError`; genuine filesystem
/// failures fall through to PyO3's default `io::Error` -> `OSError`.
#[cfg(feature = "thermo")]
fn writer_err(e: std::io::Error) -> PyErr {
    use std::io::ErrorKind::{InvalidData, InvalidInput};
    match e.kind() {
        InvalidInput | InvalidData => PyValueError::new_err(e.to_string()),
        _ => PyErr::from(e),
    }
}

/// Validate and pack `(mz, intensity)` peak arrays into the writer's tuple form,
/// rejecting the casts that would silently corrupt the output (NaN/inf m/z or
/// intensity, non-positive m/z, negative intensity, overflow past `f32::MAX`).
#[cfg(feature = "thermo")]
fn pack_peaks(mz: &[f64], intensity: &[f64]) -> PyResult<Vec<(f64, f32)>> {
    if mz.len() != intensity.len() {
        return Err(PyValueError::new_err("mz and intensity must have equal length"));
    }
    // An EMPTY peak list is allowed: a sparse Astral acquisition legitimately has
    // windows that transmit no precursor / no fragment, and the zero-residual
    // contract requires authoring an empty (cleared) scan into that template slot
    // rather than aborting or leaving the template's real signal behind.
    let mut peaks = Vec::with_capacity(mz.len());
    for (i, (&m, &inten)) in mz.iter().zip(intensity).enumerate() {
        if !m.is_finite() || m <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "mz[{i}] must be finite and > 0 (got {m})"
            )));
        }
        if !inten.is_finite() || inten < 0.0 {
            return Err(PyValueError::new_err(format!(
                "intensity[{i}] must be finite and >= 0 (got {inten})"
            )));
        }
        if inten > f32::MAX as f64 {
            return Err(PyValueError::new_err(format!(
                "intensity[{i}] overflows f32 (got {inten})"
            )));
        }
        peaks.push((m, inten as f32));
    }
    Ok(peaks)
}

#[cfg(feature = "thermo")]
#[pymethods]
impl PyThermoRawWriter {
    /// Open `template` and prepare to author into `out`. Pass
    /// `overlay_merge_tol_ppm` (finite, > 0) to overlay onto the real template
    /// signal instead of replacing it. Pass `allow_partial=True` to permit
    /// `finalize` even when fewer scans than the template holds were authored —
    /// otherwise a Replace run that does not fill every slot fails loudly (the
    /// zero-residual contract: unauthored slots keep the template's real signal).
    #[staticmethod]
    #[pyo3(signature = (template, out, overlay_merge_tol_ppm=None, allow_partial=false))]
    pub fn from_template(
        py: Python<'_>,
        template: &str,
        out: &str,
        overlay_merge_tol_ppm: Option<f64>,
        allow_partial: bool,
    ) -> PyResult<Self> {
        use rustdf::sim::acquisition::{ThermoRawWriter, WriteMode};
        if let Some(tol) = overlay_merge_tol_ppm {
            if !tol.is_finite() || tol <= 0.0 {
                return Err(PyValueError::new_err(
                    "overlay_merge_tol_ppm must be finite and > 0",
                ));
            }
        }
        let mut w = py
            .detach(|| ThermoRawWriter::from_template(template, out))
            .map_err(PyErr::from)?
            .with_allow_partial(allow_partial);
        if let Some(tol) = overlay_merge_tol_ppm {
            w = w.with_mode(WriteMode::Overlay { merge_tol_ppm: tol });
        }
        Ok(Self { inner: w, finalized: false })
    }

    /// Template (MS1, MS2) scan capacity, so a run can be checked to fit.
    pub fn capacity(&self) -> (usize, usize) {
        self.inner.capacity()
    }

    /// Author an MS1 scan from peak arrays (intensity cast to f32).
    ///
    /// `retention_time` is ordering/provenance metadata only — the template's
    /// own scan retention times are preserved, so this value is not written to
    /// the `.raw`. It is accepted to keep MS1/MS2 call sites symmetric.
    pub fn write_ms1(
        &mut self,
        retention_time: f64,
        mz: Vec<f64>,
        intensity: Vec<f64>,
    ) -> PyResult<()> {
        use rustdf::sim::acquisition::{AcquisitionWriter, ScanDescriptor};
        self.ensure_open()?;
        let peaks = pack_peaks(&mz, &intensity)?;
        let d = ScanDescriptor {
            ms_level: 1,
            retention_time,
            isolation: None,
            peaks,
        };
        self.inner.write_scan(&d).map_err(writer_err)
    }

    /// Author an MS2 scan with its isolation window + collision energy.
    ///
    /// `retention_time` is ordering/provenance metadata only (see `write_ms1`).
    /// In `Overlay` mode the isolation/CE are left as the template's; the
    /// supplied values only take effect in `Replace` mode.
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
        self.ensure_open()?;
        if !isolation_center.is_finite() || isolation_center <= 0.0 {
            return Err(PyValueError::new_err("isolation_center must be finite and > 0"));
        }
        if !isolation_width.is_finite() || isolation_width <= 0.0 {
            return Err(PyValueError::new_err("isolation_width must be finite and > 0"));
        }
        if !collision_energy.is_finite() || collision_energy < 0.0 {
            return Err(PyValueError::new_err("collision_energy must be finite and >= 0"));
        }
        let peaks = pack_peaks(&mz, &intensity)?;
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
        self.inner.write_scan(&d).map_err(writer_err)
    }

    /// Recompute the checksum and write the `.raw` to disk. May be called once;
    /// subsequent writes or finalizes raise `RuntimeError`.
    pub fn finalize(&mut self, py: Python<'_>) -> PyResult<()> {
        use rustdf::sim::acquisition::AcquisitionWriter;
        if self.finalized {
            return Err(PyRuntimeError::new_err("writer already finalized"));
        }
        py.detach(|| self.inner.finalize())
            .map_err(writer_err)?;
        self.finalized = true;
        Ok(())
    }
}

#[cfg(feature = "thermo")]
impl PyThermoRawWriter {
    /// Reject writes after a successful `finalize` (the on-disk `.raw` would go
    /// stale otherwise).
    fn ensure_open(&self) -> PyResult<()> {
        if self.finalized {
            return Err(PyRuntimeError::new_err(
                "writer already finalized; no further scans can be written",
            ));
        }
        Ok(())
    }
}

/// LegacyCompat frame (time) projection — the exact Rust path
/// (`rustdf::sim::projector::project_time_legacy`), exposed so parity tests
/// drive the real projector rather than the bare kernels. Inputs are the legacy
/// f64 EMG params + the full frames table; returns one list of
/// `(frame_id, abundance)` per peptide.
#[pyfunction]
#[pyo3(signature = (rt_mus, rt_sigmas, rt_lambdas, frame_ids, frame_times, rt_cycle_length, target_p, step_size, remove_epsilon, n_steps=None, num_threads=4))]
pub fn legacy_frame_projection(
    py: Python<'_>,
    rt_mus: Vec<f64>,
    rt_sigmas: Vec<f64>,
    rt_lambdas: Vec<f64>,
    frame_ids: Vec<u32>,
    frame_times: Vec<f64>,
    rt_cycle_length: f64,
    target_p: f64,
    step_size: f64,
    remove_epsilon: f64,
    n_steps: Option<usize>,
    num_threads: usize,
) -> Vec<Vec<(u32, f64)>> {
    py.detach(|| {
        rustdf::sim::projector::project_time_legacy(
            &rt_mus, &rt_sigmas, &rt_lambdas, &frame_ids, &frame_times, rt_cycle_length, target_p,
            step_size, n_steps, remove_epsilon, num_threads,
        )
    })
}

/// LegacyCompat scan (mobility) projection for one ion — the exact Rust path
/// (`rustdf::sim::projector::project_mobility_ion_legacy`). `mean`/`sigma` are
/// the original 1/K0 mean+std; `scan_mobilities` ascending, `scan_ids` aligned.
/// Returns `(scan, abundance)`.
#[pyfunction]
#[pyo3(signature = (mean, sigma, scan_ids, scan_mobilities, im_cycle_length, target_p, step_size))]
pub fn legacy_scan_projection(
    mean: f64,
    sigma: f64,
    scan_ids: Vec<u32>,
    scan_mobilities: Vec<f64>,
    im_cycle_length: f64,
    target_p: f64,
    step_size: f64,
) -> Vec<(i32, f64)> {
    rustdf::sim::projector::project_mobility_ion_legacy(
        mean, sigma, &scan_ids, &scan_mobilities, im_cycle_length, target_p, step_size,
    )
}

/// Batched + parallel LegacyCompat scan projection over many ions
/// (`rustdf::sim::projector::project_mobility_legacy_par`) — the kernels the
/// legacy job used. `means`/`sigmas` per ion; `scan_ids`/`scan_mobilities`
/// ascending+aligned, shared. Returns one `(scan, abundance)` list per ion.
#[pyfunction]
#[pyo3(signature = (means, sigmas, scan_ids, scan_mobilities, im_cycle_length, target_p, step_size, num_threads=4))]
pub fn legacy_scan_projection_par(
    py: Python<'_>,
    means: Vec<f64>,
    sigmas: Vec<f64>,
    scan_ids: Vec<u32>,
    scan_mobilities: Vec<f64>,
    im_cycle_length: f64,
    target_p: f64,
    step_size: f64,
    num_threads: usize,
) -> Vec<Vec<(i32, f64)>> {
    py.detach(|| {
        rustdf::sim::projector::project_mobility_legacy_par(
            &means, &sigmas, &scan_ids, &scan_mobilities, im_cycle_length, target_p, step_size,
            num_threads,
        )
    })
}

/// Accurate frame (time) projection — the EMG integrated over each event's
/// explicit `[start, end]` interval (`mscore::project_emg_over_events_par`),
/// with RT-support truncation. `event_starts`/`event_ends` define the event
/// timeline; returns one `(event_index, abundance)` list per peptide (index =
/// position in the timeline, which the caller maps to a frame id).
#[pyfunction]
#[pyo3(signature = (rt_mus, rt_sigmas, rt_lambdas, event_starts, event_ends, target_p, step_size, num_threads=4, n_steps=None))]
pub fn accurate_frame_projection(
    py: Python<'_>,
    rt_mus: Vec<f64>,
    rt_sigmas: Vec<f64>,
    rt_lambdas: Vec<f64>,
    event_starts: Vec<f64>,
    event_ends: Vec<f64>,
    target_p: f64,
    step_size: f64,
    num_threads: usize,
    n_steps: Option<usize>,
) -> PyResult<Vec<Vec<(usize, f64)>>> {
    if event_starts.len() != event_ends.len() {
        return Err(PyValueError::new_err("event_starts/event_ends length mismatch"));
    }
    let intervals: Vec<(f64, f64)> =
        event_starts.into_iter().zip(event_ends).collect();
    Ok(py.detach(|| {
        mscore::algorithm::utility::project_emg_over_events_par(
            &intervals, rt_mus, rt_sigmas, rt_lambdas, target_p, step_size,
            num_threads.max(1), n_steps,
        )
    }))
}

/// Accurate scan (mobility) projection over many ions
/// (`rustdf::sim::projector::project_mobility_accurate_par`): a Gaussian per ion
/// onto the ascending `mobility_grid`, with per-scan midpoint bins (correct on
/// non-uniform grids). Returns one `(ascending_scan_index, abundance)` list per
/// ion; the caller maps the ascending index to a native scan id.
#[pyfunction]
#[pyo3(signature = (means, sigmas, mobility_grid, target_p, step_size, num_threads=4))]
pub fn accurate_scan_projection(
    py: Python<'_>,
    means: Vec<f64>,
    sigmas: Vec<f64>,
    mobility_grid: Vec<f64>,
    target_p: f64,
    step_size: f64,
    num_threads: usize,
) -> Vec<Vec<(i32, f64)>> {
    py.detach(|| {
        rustdf::sim::projector::project_mobility_accurate_par(
            &means, &sigmas, &mobility_grid, target_p, step_size, num_threads,
        )
    })
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

/// Vendor-neutral activation/collision-energy policy (P5c). The Bruker timsTOF
/// DDA-PASEF policy reproduces the legacy `ce_bias + ce_slope*scan` exactly; the
/// DDA selection job uses this instead of the inline formula, so the CE model is
/// an instrument decision (a Thermo policy plugs in at P6 without touching the
/// selection code).
#[pyclass(unsendable)]
pub struct PyActivationPolicy {
    pub inner: rustdf::sim::scheme::ActivationPolicy,
}

#[pymethods]
impl PyActivationPolicy {
    /// Bruker timsTOF DDA-PASEF policy (collisional/HCD, eV, CE linear in scan).
    #[staticmethod]
    pub fn bruker_pasef(ce_bias: f64, ce_slope: f64) -> Self {
        PyActivationPolicy {
            inner: rustdf::sim::scheme::ActivationPolicy::bruker_pasef(ce_bias, ce_slope),
        }
    }

    /// Collision energy (eV) applied at a Bruker mobility scan. Errors if the
    /// policy is not scan-parameterised (e.g. a per-window/Thermo model).
    pub fn collision_energy_for_scan(&self, scan: u32) -> PyResult<f64> {
        self.inner.collision_energy_for_scan(scan).ok_or_else(|| {
            PyValueError::new_err(
                "this activation policy is not scan-parameterised (no IMS); \
                 collision energy is driven by m/z, not scan",
            )
        })
    }

    /// Vectorised: collision energies for a list of mobility scans.
    pub fn collision_energies_for_scans(&self, scans: Vec<u32>) -> PyResult<Vec<f64>> {
        scans
            .iter()
            .map(|&s| {
                self.inner.collision_energy_for_scan(s).ok_or_else(|| {
                    PyValueError::new_err(
                        "this activation policy is not scan-parameterised (no IMS)",
                    )
                })
            })
            .collect()
    }

    #[getter]
    pub fn activation_method(&self) -> String {
        format!("{:?}", self.inner.method).to_lowercase()
    }

    #[getter]
    pub fn energy_unit(&self) -> String {
        match self.inner.unit {
            rustdf::sim::scheme::EnergyUnit::ElectronVolt => "ev".to_string(),
            rustdf::sim::scheme::EnergyUnit::NormalizedCe => "nce".to_string(),
            rustdf::sim::scheme::EnergyUnit::Unknown => "unknown".to_string(),
        }
    }
}

/// Render an Astral build-from-template `synthetic_data.db` to a Thermo `.raw`
/// (P6e dispatch). Walks the template slot manifest, renders each frame (MS1
/// profile / MS2 per-window fragments), and authors it into its slot. Returns
/// `(scans, ms1, ms2, ms2_nonempty, overflow_cleared, checksum_valid)`. Requires
/// the `thermo` feature; the DB must have been built from this template (frames
/// 1:1 with template slots), else a structured error is raised. `superimpose_ppm`
/// `0` (default) replaces the template signal (pure simulated output); `>0` keeps
/// the template's real signal and overlays simulated peaks on top (real⊕sim), using
/// the value as the MS2 centroid merge tolerance in ppm.
#[cfg(feature = "thermo")]
#[pyfunction]
#[pyo3(signature = (db_path, template_path, out_path, num_threads=4, quad_k=15.0, max_ms1_peaks=400, max_ms2_peaks=120, precursor_noise_ppm=0.0, fragment_noise_ppm=0.0, superimpose_ppm=0.0))]
pub fn write_astral_raw(
    py: Python<'_>,
    db_path: &str,
    template_path: &str,
    out_path: &str,
    num_threads: usize,
    quad_k: f64,
    max_ms1_peaks: usize,
    max_ms2_peaks: usize,
    precursor_noise_ppm: f64,
    fragment_noise_ppm: f64,
    superimpose_ppm: f64,
) -> PyResult<(usize, usize, usize, usize, usize, bool)> {
    use rustdf::sim::astral_dispatch::{write_astral_raw as run, AstralWriteOptions};
    use std::path::Path;
    for (name, v) in [
        ("precursor_noise_ppm", precursor_noise_ppm),
        ("fragment_noise_ppm", fragment_noise_ppm),
        ("superimpose_ppm", superimpose_ppm),
    ] {
        if !v.is_finite() || v < 0.0 {
            return Err(PyValueError::new_err(format!("{name} must be finite and >= 0, got {v}")));
        }
    }
    let opts = AstralWriteOptions {
        num_threads,
        quad_k,
        max_ms1_peaks,
        max_ms2_peaks,
        precursor_noise_ppm,
        fragment_noise_ppm,
        superimpose_ppm,
    };
    let s = py
        .detach(|| {
            run(Path::new(db_path), Path::new(template_path), Path::new(out_path), opts)
        })
        .map_err(PyValueError::new_err)?;
    Ok((s.scans, s.ms1, s.ms2, s.ms2_nonempty, s.overflow_cleared, s.checksum_valid))
}

/// Re-window a Thermo DIA template (Tier-2 3a): write `src` → `dst` with every MS2
/// isolation window set to `isolation_width` (centers + CE kept; same window count).
/// Use as a pre-authoring step, then point the run's template at `dst`. Returns the
/// number of MS2 scans re-windowed.
#[cfg(feature = "thermo")]
#[pyfunction]
pub fn rewindow_thermo_template(
    py: Python<'_>,
    src_path: &str,
    dst_path: &str,
    isolation_width: f64,
) -> PyResult<usize> {
    use rustdf::sim::acquisition::rewindow_thermo_template as run;
    use std::path::Path;
    py.detach(|| run(Path::new(src_path), Path::new(dst_path), isolation_width))
        .map_err(PyValueError::new_err)
}

/// Render a no-IM DIA `synthetic_data.db` to open **mzML** (the SCIEX / vendor-neutral
/// output). Walks the DB frames, renders MS1 (precursor marginal) + MS2 (per-window
/// fragments), and writes mzML via mzdata. Returns `(scans, ms1, ms2, ms2_nonempty)`.
/// Requires the `mzml` feature.
#[cfg(feature = "mzml")]
#[pyfunction]
#[pyo3(signature = (db_path, out_path, num_threads=4, quad_k=15.0))]
pub fn render_dia_mzml(
    py: Python<'_>,
    db_path: &str,
    out_path: &str,
    num_threads: usize,
    quad_k: f64,
) -> PyResult<(usize, usize, usize, usize)> {
    use std::path::Path;
    let s = py
        .detach(|| {
            rustdf::sim::mzml::render_db_to_mzml(
                Path::new(db_path),
                Path::new(out_path),
                num_threads,
                quad_k,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((s.scans, s.ms1, s.ms2, s.ms2_nonempty))
}

/// Whether the connector was built with the `mzml` feature (open mzML writer).
#[pyfunction]
pub fn has_mzml() -> bool {
    cfg!(feature = "mzml")
}

#[pymodule]
pub fn py_acquisition(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAcquisitionScheme>()?;
    m.add_class::<PyActivationPolicy>()?;
    #[cfg(feature = "thermo")]
    m.add_class::<PyThermoRawWriter>()?;
    #[cfg(feature = "thermo")]
    m.add_function(wrap_pyfunction!(write_astral_raw, m)?)?;
    #[cfg(feature = "thermo")]
    m.add_function(wrap_pyfunction!(rewindow_thermo_template, m)?)?;
    m.add_function(wrap_pyfunction!(legacy_frame_projection, m)?)?;
    m.add_function(wrap_pyfunction!(legacy_scan_projection, m)?)?;
    m.add_function(wrap_pyfunction!(legacy_scan_projection_par, m)?)?;
    m.add_function(wrap_pyfunction!(accurate_frame_projection, m)?)?;
    m.add_function(wrap_pyfunction!(accurate_scan_projection, m)?)?;
    m.add_function(wrap_pyfunction!(has_thermo, m)?)?;
    m.add_function(wrap_pyfunction!(has_sciex, m)?)?;
    #[cfg(feature = "mzml")]
    m.add_function(wrap_pyfunction!(render_dia_mzml, m)?)?;
    m.add_function(wrap_pyfunction!(has_mzml, m)?)?;
    Ok(())
}
