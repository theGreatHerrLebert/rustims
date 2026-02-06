# CLAUDE.md - AI Assistant Guide for rustims

This document provides guidance for AI assistants working with the rustims codebase.

## Project Overview

**rustims** is a framework for processing raw data from Ion-Mobility Spectrometry (IMS) in proteomics mass spectrometry. It provides efficient algorithm implementations and robust data structures using Rust as the backend, with Python bindings via PyO3.

- **License**: MIT
- **Author**: David Teschner
- **Python Version**: >=3.11, <3.14
- **Rust Version**: 1.84+

## Repository Structure

```
rustims/
├── mscore/              # Core Rust library - data structures & algorithms
├── rustdf/              # Rust TDF file reader/writer (Bruker timsTOF format)
├── rustms/              # Additional Rust MS utilities (chemistry, proteomics)
├── imspy_connector/     # PyO3 Python bindings (compiles to wheel)
├── packages/            # Modular Python packages
│   ├── imspy-core/          # Base data structures, timsTOF dataset access
│   ├── imspy-predictors/    # PyTorch models for CCS, RT, fragment intensities
│   ├── imspy-dia/           # DIA-PASEF clustering and feature extraction
│   ├── imspy-search/        # Database search integration (sagepy, mokapot)
│   ├── imspy-simulation/    # TimSim synthetic data generation
│   └── imspy-vis/           # Visualization tools
├── imsjl_connector/     # Julia bindings via FFI (experimental)
├── IMSJL/               # Julia code (experimental)
└── .github/workflows/   # CI/CD pipelines
```

### Architecture

The project follows a layered architecture:

1. **Rust Core Layer** (`mscore`, `rustdf`, `rustms`): Low-level, high-performance implementations
2. **Binding Layer** (`imspy_connector`): PyO3 bindings exposing Rust to Python
3. **Python API Layer** (`packages/`): Modular Python packages with ML/DL features

## Key Crates and Packages

### mscore (Rust) — v0.4.1
Core library containing:
- `data/` - Spectrum, peptide, SMILES data structures
- `chemistry/` - Elements, amino acids, UNIMOD, sum formulas
- `algorithm/` - Isotope calculations, peptide utilities
- `timstof/` - Frame, slice, spectrum structures for timsTOF data
- `simulation/` - Annotation and simulation utilities

Key types: `MzSpectrum`, `MsType`, `PeptideSequence`, `ImsFrame`, `TimsFrame`

### rustdf (Rust) — v0.4.1
TDF file I/O library:
- `data/` - Dataset reading, DDA/DIA handling, raw data access
- `sim/` - Simulation containers, precursor handling, synthetic data generation

### rustms (Rust) — v0.1.0
Additional MS utilities:
- `chemistry/` - Chemical formulas, elements, UNIMOD
- `proteomics/` - Amino acids, peptides
- `algorithm/` - Isotope and peptide algorithms
- `ms/` - Spectrum utilities

### imspy_connector (Rust/PyO3) — v0.4.1
Python bindings organized as 21 submodules:
- `py_mz_spectrum`, `py_peptide`, `py_tims_frame`, `py_tims_slice`, `py_dataset`
- `py_dda`, `py_dia`, `py_quadrupole`, `py_feature`, `py_pseudo`
- `py_chemistry`, `py_elements`, `py_sumformula`, `py_amino_acids`, `py_unimod`, `py_constants`
- `py_annotation`, `py_simulation`, `py_ml_utility`, `py_spectrum_processing`, `py_utility`
- Each `py_*.rs` file wraps corresponding Rust types
- Uses `#[pyclass]` and `#[pymethods]` attributes

### Python Packages (under `packages/`)

| Package | Version | Module | Description |
|---------|---------|--------|-------------|
| **imspy-core** | 0.4.0 | `imspy_core` | Base data structures, timsTOF dataset access |
| **imspy-predictors** | 0.5.0 | `imspy_predictors` | PyTorch models for CCS, RT, fragment intensities |
| **imspy-dia** | 0.4.0 | `imspy_dia` | DIA-PASEF clustering and feature extraction |
| **imspy-search** | 0.4.0 | `imspy_search` | Database search integration (sagepy, mokapot) |
| **imspy-simulation** | 0.4.0 | `imspy_simulation` | TimSim synthetic data generation & EVAL pipeline |
| **imspy-vis** | 0.4.0 | `imspy_vis` | Visualization and plotting tools |

#### imspy-core
- `core/` - RustWrapperObject base class
- `data/` - MzSpectrum, PeptideSequence wrappers
- `chemistry/` - Elements, amino acids, UNIMOD, mobility, constants
- `timstof/` - TimsDataset, TimsDatasetDDA, TimsDatasetDIA, TimsFrame, TimsSlice
- `utility/` - General helpers, sequence processing

#### imspy-predictors
- `ccs/` - Collisional cross-section predictors (`PyTorchCCSPredictor`)
- `rt/` - Retention time predictors (`PyTorchRTPredictor`)
- `intensity/` - Ion intensity predictors (`Prosit2023TimsTofWrapper`, `DeepPeptideIntensityPredictor`)
- `ionization/` - Charge state predictors
- `koina_models/` - Koina remote prediction service
- `models/` - Neural network model definitions
- `pretrained/` - Pretrained model management
- `utilities/` - Tokenizers and utilities

#### imspy-dia
- `clustering/` - DIA clustering utilities
- `pipeline/` - DIA processing pipeline (cluster_pipeline, cluster_report)

#### imspy-search
- `cli/` - Command-line interfaces (imspy_dda, imspy_ccs, imspy_rescore_sage)
- `configs/` - Configuration files
- `dda_extensions.py` - Sage extension methods for PrecursorDDA/TimsDatasetDDA
- `rescoring.py` - PSM rescoring
- `utility.py`, `mgf.py`, `sage_output_utility.py`

#### imspy-simulation
- `timsim/` - TimSim simulator, GUI, jobs, validation, integration tests (EVAL pipeline)
- `builders/` - Simulation builders
- `core/` - Core simulation logic
- `data/` - Simulation data structures
- `annotation.py`, `experiment.py`, `acquisition.py`, `tdf.py`

#### imspy-vis
- `frame_rendering.py` - Frame rendering
- `pointcloud.py` - Point cloud visualization

### CLI Tools

| Entry Point | Package | Function |
|-------------|---------|----------|
| `imspy-dda` | imspy-search | `imspy_search.cli.imspy_dda:main` |
| `imspy-ccs` | imspy-search | `imspy_search.cli.imspy_ccs:main` |
| `imspy-rescore-sage` | imspy-search | `imspy_search.cli.imspy_rescore_sage:main` |
| `open_tracer` | imspy-dia | `imspy_dia.pipeline.cluster_pipeline:main` |
| `imspy-cluster-report` | imspy-dia | `imspy_dia.pipeline.cluster_report:main` |
| `timsim` | imspy-simulation | `imspy_simulation.timsim.simulator:main` |
| `timsim-gui` | imspy-simulation | `imspy_simulation.timsim.gui:main` |

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

### Installing Python Packages

```bash
# Install from packages/ directory
cd packages
pip install -e ./imspy-core
pip install -e ./imspy-predictors
pip install -e ./imspy-dia
pip install -e ./imspy-search
pip install -e ./imspy-simulation
pip install -e ./imspy-vis
```

### Running Tests

```bash
# Rust tests
cd mscore && cargo test --verbose
cd rustdf && cargo test --verbose

# Python tests
cd packages/imspy-predictors && pytest tests/
```

## CI/CD Pipelines

- **rust.yml**: Builds and tests `mscore` and `rustdf` on push/PR to main
- **imspy-connector-publish.yml**: Builds wheels for multiple platforms on release
- **imspy-publish.yml**: Publishes Python packages on release
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

1. Implement in appropriate `mscore`, `rustdf`, or `rustms` module
2. Add Python binding in `imspy_connector/src/py_<module>.rs`
3. Register in module's `#[pymodule]` function
4. Create Python wrapper in the appropriate package under `packages/`

### Modifying Data Structures

1. Update Rust struct in `mscore` or `rustdf`
2. Update PyO3 wrapper in `imspy_connector`
3. Update Python wrapper in the appropriate package under `packages/`
4. Run tests: `cargo test` and `pytest`

### Version Updates

Versions are maintained in:
- `mscore/Cargo.toml` (0.4.1)
- `rustdf/Cargo.toml` (0.4.1)
- `rustms/Cargo.toml` (0.1.0)
- `imspy_connector/Cargo.toml` (0.4.1)
- `packages/imspy-core/pyproject.toml` (0.4.0)
- `packages/imspy-predictors/pyproject.toml` (0.5.0)
- `packages/imspy-dia/pyproject.toml` (0.4.0)
- `packages/imspy-search/pyproject.toml` (0.4.0)
- `packages/imspy-simulation/pyproject.toml` (0.4.0)
- `packages/imspy-vis/pyproject.toml` (0.4.0)

Dependencies between Rust crates reference specific versions (e.g., `mscore = { version = "0.4.1" }`).

## Documentation

- Rust docs: https://thegreatherrlebert.github.io/rustims/main/mscore/
- Python docs: https://thegreatherrlebert.github.io/rustims/main/imspy/
- TimSim docs: `packages/imspy-simulation/SIMULATOR_README.md`
- EVAL pipeline: `packages/imspy-simulation/src/imspy_simulation/timsim/integration/VALIDATION_README.md`

## Important Notes

1. **Python >=3.11**: Required by all packages (upper bound <3.14)
2. **Bruker SDK**: Optional but recommended for accurate mass/mobility calibration
3. **GPU Support**: PyTorch ships with CUDA support (`pip install torch`)
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
- `torch`: Deep learning models (PyTorch)
- `sagepy`: Peptide search (imspy-search)
- `mokapot`: Statistical validation (imspy-search)
- `numba`: JIT compilation
- `pandas`/`numpy`: Data handling
- `koinapy`: Remote prediction service (optional, imspy-predictors)

## File Patterns

- Rust source: `<crate>/src/**/*.rs`
- Python source: `packages/<package>/src/<module>/**/*.py`
- Tests (Rust): Inline in source files
- Tests (Python): `packages/<package>/tests/`
- CI: `.github/workflows/*.yml`
