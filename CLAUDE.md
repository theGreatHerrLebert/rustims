# CLAUDE.md - AI Assistant Guide for rustims

This document provides guidance for AI assistants working with the rustims codebase.

## Project Overview

**rustims** is a framework for processing raw data from Ion-Mobility Spectrometry (IMS) in proteomics mass spectrometry. It provides efficient algorithm implementations and robust data structures using Rust as the backend, with Python bindings via PyO3.

- **License**: MIT
- **Author**: David Teschner
- **Python Version**: 3.11 (required due to TensorFlow compatibility)
- **Rust Version**: 1.84+

## Repository Structure

```
rustims/
├── mscore/              # Core Rust library - data structures & algorithms
├── rustdf/              # Rust TDF file reader/writer (Bruker timsTOF format)
├── rustms/              # Additional Rust MS utilities (chemistry, proteomics)
├── imspy_connector/     # PyO3 Python bindings (compiles to wheel)
├── imspy/               # High-level Python package for end-users
├── imsjl_connector/     # Julia bindings via FFI (experimental)
├── IMSJL/               # Julia code (experimental)
└── .github/workflows/   # CI/CD pipelines
```

### Architecture

The project follows a layered architecture:

1. **Rust Core Layer** (`mscore`, `rustdf`, `rustms`): Low-level, high-performance implementations
2. **Binding Layer** (`imspy_connector`): PyO3 bindings exposing Rust to Python
3. **Python API Layer** (`imspy`): User-friendly Python package with additional ML/DL features

## Key Crates and Packages

### mscore (Rust)
Core library containing:
- `data/` - Spectrum, peptide, SMILES data structures
- `chemistry/` - Elements, amino acids, UNIMOD, sum formulas
- `algorithm/` - Isotope calculations, peptide utilities
- `timstof/` - Frame, slice, spectrum structures for timsTOF data
- `simulation/` - Annotation and simulation utilities

Key types: `MzSpectrum`, `MsType`, `PeptideSequence`, `ImsFrame`, `TimsFrame`

### rustdf (Rust)
TDF file I/O library:
- `data/` - Dataset reading, DDA/DIA handling, raw data access
- `sim/` - Simulation containers, precursor handling, synthetic data generation

### rustms (Rust)
Additional MS utilities:
- `chemistry/` - Chemical formulas, elements, UNIMOD
- `proteomics/` - Amino acids, peptides
- `algorithm/` - Isotope and peptide algorithms
- `ms/` - Spectrum utilities

### imspy_connector (Rust/PyO3)
Python bindings organized as submodules:
- `py_mz_spectrum`, `py_peptide`, `py_tims_frame`, etc.
- Each `py_*.rs` file wraps corresponding Rust types
- Uses `#[pyclass]` and `#[pymethods]` attributes

### imspy (Python)
High-level Python package:
- `timstof/` - TimsDataset, TimsDatasetDDA, TimsDatasetDIA
- `data/` - Python wrappers for Rust data structures
- `chemistry/` - Elements, sum formulas
- `algorithm/` - ML models, prediction algorithms
- `simulation/` - TimSim simulation tools
- `vis/` - Visualization utilities

CLI tools: `imspy_dda`, `timsim`, `timsim_gui`, `imspy_ccs`, `imspy_rescore_sage`

## Development Workflows

### Building Rust Crates

```bash
# Build specific crate
cd mscore && cargo build --release
cd rustdf && cargo build --release

# Run tests
cargo test --verbose
```

### Building Python Bindings

```bash
# Install maturin
pip install maturin[patchelf]

# Build wheel
cd imspy_connector
maturin build --release

# Install wheel
pip install --force-reinstall ./target/wheels/<filename>.whl
```

### Building Python Package

```bash
# Install poetry
pip install poetry

# Install imspy
cd imspy
poetry install
```

### Running Tests

```bash
# Rust tests
cd mscore && cargo test --verbose
cd rustdf && cargo test --verbose

# Python tests (limited)
cd imspy && pytest tests/
```

## CI/CD Pipelines

- **rust.yml**: Builds and tests `mscore` and `rustdf` on push/PR to main
- **imspy-connector-publish.yml**: Builds wheels for multiple platforms on release
- **imspy-publish.yml**: Publishes Python package via Poetry on release
- **docs.yml**: Builds Rust docs (cargo doc) and Python docs (Sphinx)

## Code Conventions

### Rust

- Use `#[derive(Clone, Debug, Serialize, Deserialize)]` for data structures
- Add `Encode, Decode` from bincode for binary serialization
- Document public APIs with `///` doc comments
- Use `rayon` for parallelism
- Tests go in `#[cfg(test)]` modules at end of files

```rust
/// Brief description of the function.
///
/// # Arguments
///
/// * `param` - Description of the parameter.
///
/// # Examples
///
/// ```
/// use mscore::...;
/// // example code
/// ```
pub fn example_function(param: Type) -> ReturnType {
    // implementation
}
```

### PyO3 Bindings

Pattern for wrapping Rust types:

```rust
#[pyclass]
#[derive(Clone)]
pub struct PyRustType {
    pub inner: RustType,
}

#[pymethods]
impl PyRustType {
    #[new]
    pub fn new(/* params */) -> PyResult<Self> {
        Ok(PyRustType { inner: RustType::new(/* params */) })
    }

    #[getter]
    pub fn property(&self) -> Type {
        self.inner.property.clone()
    }
}
```

### Python

- Use type hints for all function signatures
- Wrapper classes implement `RustWrapperObject` pattern with `get_py_ptr()` and `from_py_ptr()` methods
- Use `imspy_connector` submodules: `ims = imspy_connector.py_<module>`

```python
class PythonWrapper(RustWrapperObject):
    def __init__(self, param: Type):
        self.__py_ptr = ims.PyRustType(param)

    @classmethod
    def from_py_ptr(cls, ptr):
        instance = cls.__new__(cls)
        instance.__py_ptr = ptr
        return instance

    def get_py_ptr(self):
        return self.__py_ptr
```

## Key Data Structures

### Spectra
- `MzSpectrum`: m/z and intensity vectors
- `IndexedMzSpectrum`: MzSpectrum with index
- `TimsSpectrum`: IMS spectrum with scan/mobility
- `MzSpectrumAnnotated`: Annotated spectrum with peak metadata

### Frames
- `ImsFrame`: Ion mobility frame (retention_time, mobility, mz, intensity)
- `TimsFrame`: timsTOF frame with frame_id and ms_type
- `RawTimsFrame`: Raw frame with scan/tof indices

### Peptides
- `PeptideSequence`: Sequence with modifications (UNIMOD format)
- `PeptideIon`: Peptide with charge and intensity
- `PeptideProductIonSeries`: Fragment ion series (b/y ions)

## Common Tasks

### Adding a New Rust Function

1. Implement in appropriate `mscore` or `rustdf` module
2. Add Python binding in `imspy_connector/src/py_<module>.rs`
3. Register in module's `#[pymodule]` function
4. Create Python wrapper in `imspy/<subpackage>/`

### Modifying Data Structures

1. Update Rust struct in `mscore` or `rustdf`
2. Update PyO3 wrapper in `imspy_connector`
3. Update Python wrapper in `imspy`
4. Run tests: `cargo test` and `pytest`

### Version Updates

Versions are maintained in:
- `mscore/Cargo.toml`
- `rustdf/Cargo.toml`
- `imspy_connector/Cargo.toml`
- `imspy/pyproject.toml`

Dependencies between crates reference specific versions (e.g., `mscore = { version = "0.3.3" }`).

## Documentation

- Rust docs: https://thegreatherrlebert.github.io/rustims/main/mscore/
- Python docs: https://thegreatherrlebert.github.io/rustims/main/imspy/
- Examples: `imspy/examples/` (Jupyter notebooks)

## Important Notes

1. **Python 3.11 Required**: Due to TensorFlow 2.15 dependency
2. **Bruker SDK**: Optional but recommended for accurate mass/mobility calibration
3. **GPU Support**: Install `tensorflow[and-cuda]==2.15.*` for CUDA support
4. **Local Development**: Use `path = "../mscore"` in Cargo.toml for local development (commented out in production)

## Dependencies

### Key Rust Dependencies
- `pyo3`: Python bindings
- `numpy`: NumPy array interop
- `rayon`: Parallelism
- `serde`/`bincode`: Serialization
- `rusqlite`: SQLite for TDF files
- `zstd`/`lzf`: Compression

### Key Python Dependencies
- `tensorflow`: Deep learning models
- `sagepy`: Peptide search
- `numba`: JIT compilation
- `pandas`/`numpy`: Data handling
- `mokapot`: Statistical validation

## File Patterns

- Rust source: `<crate>/src/**/*.rs`
- Python source: `imspy/imspy/**/*.py`
- Tests (Rust): Inline in source files
- Tests (Python): `imspy/tests/`
- Examples: `imspy/examples/*.ipynb`
- CI: `.github/workflows/*.yml`
