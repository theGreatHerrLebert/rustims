Reading additional input from stdin...
OpenAI Codex v0.136.0
--------
workdir: /scratch/timsim-demo/SUBMISSION/rustims
model: gpt-5.5
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: medium
reasoning summaries: none
session id: 019eab5a-5941-7341-b762-61b408639746
--------
user
Review this PyO3 (pyo3 0.23) module that exposes a Rust DIA-acquisition-scheme type to Python in a mass-spec toolkit. It wraps rustdf::sim::scheme::AcquisitionScheme (already reviewed) and is the Python entry point for the simulator to build a DIA layout and render Bruker DiaFrameMsMsWindows/DiaFrameMsMsInfo tables.

Trusted context: AcquisitionScheme::from_bruker_d/from_thermo_raw/from_sciex_wiff return io::Result; to_bruker_windows/to_bruker_info return Result<Vec<row>, String>; the connector forwards 'thermo'/'sciex' cargo features to rustdf. Verified from Python on a real .d: from_bruker_d -> to_bruker_windows/to_bruker_info reproduce the source SQLite tables exactly (36 / 16543 rows). The column-list return is consumed as pandas DataFrames.

Focus, ranked: 1) PyO3 correctness/idiom for pyo3 0.23 — #[pyclass]/#[pymethods]/#[staticmethod]/#[getter], the cfg-gated methods INSIDE one #[pymethods] block (is that valid without the multiple-pymethods feature?), the #[pyo3(signature=...)] with Option args, returning large tuples of Vec<u32>/Vec<f64> (correctness + are they materialized as Python lists efficiently/should they be numpy?). 2) error mapping (io/String -> PyValueError) — any lossy or misleading cases; should some be different exception types. 3) GIL/perf: to_bruker_info can return millions of rows as 2 Python lists while holding the GIL — is that a problem, and is releasing the GIL or returning numpy advisable? 4) API ergonomics for the DataFrame consumer (parallel column lists vs a single structured return; risk of column/length desync). 5) the feature-gated constructors: if built without thermo/sciex, are they cleanly absent (no broken symbols) and is that the intended degradation? 6) anything unsound, panics, or memory blowups. Concrete, ~500 words.

<stdin>
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
</stdin>
codex
I’ll verify the PyO3 0.23 macro and conversion behavior against its official documentation/source, then give findings ordered by severity with concrete recommendations.
web search: 
web search: site:pyo3.rs/v0.23.0 pymethods cfg feature multiple-pymethods
web search: 
web search: site:pyo3.rs/v0.23.4 allow_threads pymethod GIL release
web search: 
web search: site:docs.rs/pyo3/0.23.5 "From<io::Error>" PyErr
codex
**Findings**

1. **High: `to_bruker_info` has serious GIL and memory scaling risk.**  
   `Vec<u32>` correctly becomes `list[int]` in PyO3 0.23, but conversion allocates millions of Python integer objects while holding the GIL. The Rust work also runs with the GIL held even though no `Python` argument appears. For large runs this can block all Python threads and use far more memory than the underlying 8 bytes per row.

   Prefer returning NumPy `uint32` arrays using `numpy::PyArray1::from_vec`, which transfers the Rust allocation into NumPy. If NumPy cannot be mandatory, offer a separate array-oriented method. Adding `py: Python<'_>` and wrapping Rust-only generation in `py.allow_threads(...)` helps computation, but Python-list conversion still requires the GIL. Constructors performing vendor I/O are also candidates for `allow_threads`.

2. **Medium: I/O errors should not become `ValueError`.**  
   `from_bruker_d`, `from_thermo_raw`, and `from_sciex_wiff` return `io::Result`; mapping every failure through `to_string()` loses the error kind, errno, and Python’s useful `FileNotFoundError`, `PermissionError`, etc. PyO3 provides `From<std::io::Error> for PyErr`, mapping these to `OSError` or its subclasses:

   ```rust
   .map_err(PyErr::from)
   ```

   `ValueError` is reasonable for validation and rendering failures represented only as `String`, assuming those strings describe invalid/inconsistent schemes rather than I/O or internal failures.

3. **Medium: partial SCIEX CE input is silently discarded.**  
   Supplying only `ce_intercept` or only `ce_slope_per_mz` produces `Unknown`. That is surprising and can silently change simulation output. Return `ValueError` unless both are supplied or both omitted. Consider also rejecting non-finite values here if the underlying constructor does not.

4. **Low: the PyO3 macro arrangement is valid.**  
   `#[cfg]`-gated methods inside one `#[pymethods]` block are fine. The `multiple-pymethods` feature is required only for multiple separate `#[pymethods]` blocks on one class. Disabled Thermo/SCIEX constructors therefore disappear cleanly with no unresolved symbols. This is reasonable degradation, though exposing module-level capability flags would improve discoverability.

5. **Low: signatures and getters are otherwise correct.**  
   `#[staticmethod]`, `#[getter]`, and the explicit defaults for trailing `Option<f64>` arguments are idiomatic for 0.23. Consider making CE arguments keyword-only:

   ```rust
   signature = (path, cycle_time_s, gradient_length_s, *, ce_intercept=None, ce_slope_per_mz=None)
   ```

6. **Low: positional column tuples are fragile.**  
   The implementation itself cannot desynchronize lengths because every column is pushed in the same loop. The Python caller can, however, mislabel or reorder tuple positions. A mapping such as `{column_name: numpy_array}` is directly accepted by pandas and self-describing. `instrument` and `provenance` also rely on Rust `Debug` formatting, which is not a stable Python API; expose canonical strings or structured properties instead.

No direct unsoundness or explicit panic path is present. The main operational hazard is list materialization for multi-million-row info tables.

Sources: [PyO3 signatures](https://pyo3.rs/v0.23.3/function/signature.html), [PyO3 classes](https://pyo3.rs/v0.23.1/class.html), [PyO3 vector conversion](https://pyo3.rs/main/doc/src/pyo3/conversions/std/vec.rs), [NumPy ownership transfer](https://docs.rs/numpy/latest/numpy/array/struct.PyArray.html).
tokens used
28,680
**Findings**

1. **High: `to_bruker_info` has serious GIL and memory scaling risk.**  
   `Vec<u32>` correctly becomes `list[int]` in PyO3 0.23, but conversion allocates millions of Python integer objects while holding the GIL. The Rust work also runs with the GIL held even though no `Python` argument appears. For large runs this can block all Python threads and use far more memory than the underlying 8 bytes per row.

   Prefer returning NumPy `uint32` arrays using `numpy::PyArray1::from_vec`, which transfers the Rust allocation into NumPy. If NumPy cannot be mandatory, offer a separate array-oriented method. Adding `py: Python<'_>` and wrapping Rust-only generation in `py.allow_threads(...)` helps computation, but Python-list conversion still requires the GIL. Constructors performing vendor I/O are also candidates for `allow_threads`.

2. **Medium: I/O errors should not become `ValueError`.**  
   `from_bruker_d`, `from_thermo_raw`, and `from_sciex_wiff` return `io::Result`; mapping every failure through `to_string()` loses the error kind, errno, and Python’s useful `FileNotFoundError`, `PermissionError`, etc. PyO3 provides `From<std::io::Error> for PyErr`, mapping these to `OSError` or its subclasses:

   ```rust
   .map_err(PyErr::from)
   ```

   `ValueError` is reasonable for validation and rendering failures represented only as `String`, assuming those strings describe invalid/inconsistent schemes rather than I/O or internal failures.

3. **Medium: partial SCIEX CE input is silently discarded.**  
   Supplying only `ce_intercept` or only `ce_slope_per_mz` produces `Unknown`. That is surprising and can silently change simulation output. Return `ValueError` unless both are supplied or both omitted. Consider also rejecting non-finite values here if the underlying constructor does not.

4. **Low: the PyO3 macro arrangement is valid.**  
   `#[cfg]`-gated methods inside one `#[pymethods]` block are fine. The `multiple-pymethods` feature is required only for multiple separate `#[pymethods]` blocks on one class. Disabled Thermo/SCIEX constructors therefore disappear cleanly with no unresolved symbols. This is reasonable degradation, though exposing module-level capability flags would improve discoverability.

5. **Low: signatures and getters are otherwise correct.**  
   `#[staticmethod]`, `#[getter]`, and the explicit defaults for trailing `Option<f64>` arguments are idiomatic for 0.23. Consider making CE arguments keyword-only:

   ```rust
   signature = (path, cycle_time_s, gradient_length_s, *, ce_intercept=None, ce_slope_per_mz=None)
   ```

6. **Low: positional column tuples are fragile.**  
   The implementation itself cannot desynchronize lengths because every column is pushed in the same loop. The Python caller can, however, mislabel or reorder tuple positions. A mapping such as `{column_name: numpy_array}` is directly accepted by pandas and self-describing. `instrument` and `provenance` also rely on Rust `Debug` formatting, which is not a stable Python API; expose canonical strings or structured properties instead.

No direct unsoundness or explicit panic path is present. The main operational hazard is list materialization for multi-million-row info tables.

Sources: [PyO3 signatures](https://pyo3.rs/v0.23.3/function/signature.html), [PyO3 classes](https://pyo3.rs/v0.23.1/class.html), [PyO3 vector conversion](https://pyo3.rs/main/doc/src/pyo3/conversions/std/vec.rs), [NumPy ownership transfer](https://docs.rs/numpy/latest/numpy/array/struct.PyArray.html).
