# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**imspy** is the high-level Python package for working with Bruker timsTOF ion mobility spectrometry data. It wraps Rust implementations via PyO3 bindings (`imspy_connector`) and provides ML/DL algorithms for peptide property prediction.

See the root `../CLAUDE.md` for the full rustims project architecture.

## Build & Development Commands

```bash
# Install with Poetry (development mode)
poetry install

# Build distribution
poetry build

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_spectrum.py -v

# Run a specific test
pytest tests/test_spectrum.py::TestMzSpectrum::test_create_simple_spectrum -v

# Run with coverage
pytest tests/ --cov=imspy

# Build documentation (Sphinx)
cd docs && make html
```

## Rebuilding Rust Bindings

When modifying Rust code in `mscore`, `rustdf`, or `imspy_connector`:

```bash
cd ../imspy_connector
maturin build --release
pip install --force-reinstall ./target/wheels/*.whl
```

## Package Structure

```
imspy/
├── core/base.py          # RustWrapperObject ABC
├── data/                 # Spectrum, peptide data structures
├── timstof/              # TimsDataset, TimsDatasetDDA, TimsDatasetDIA
├── chemistry/            # Elements, sum formulas, amino acids
├── algorithm/            # ML predictors (CCS, RT, intensity)
│   ├── ccs/             # Collision cross-section prediction
│   ├── rt/              # Retention time prediction
│   ├── intensity/       # Fragment intensity (Prosit wrapper)
│   └── koina_models/    # Pre-trained model access
├── simulation/           # TimSim synthetic data generation
│   └── timsim/          # CLI simulator and GUI
└── vis/                  # Visualization utilities
```

## Key Patterns

### RustWrapperObject Pattern

All Python classes wrapping Rust objects inherit from `RustWrapperObject` and implement:

```python
from imspy.core.base import RustWrapperObject
import imspy_connector
ims = imspy_connector.py_spectrum  # or other submodule

class MyWrapper(RustWrapperObject):
    def __init__(self, data):
        self.__py_ptr = ims.PyRustType(data)

    @classmethod
    def from_py_ptr(cls, ptr):
        instance = cls.__new__(cls)
        instance.__py_ptr = ptr
        return instance

    def get_py_ptr(self):
        return self.__py_ptr
```

### Connector Submodule Import Pattern

```python
import imspy_connector
ims = imspy_connector.py_spectrum      # Spectrum types
ims = imspy_connector.py_peptide       # Peptide types
ims = imspy_connector.py_tims_frame    # Frame types
ims = imspy_connector.py_chemistry     # Chemistry utilities
ims = imspy_connector.py_dataset       # Dataset access
```

## CLI Tools

```bash
imspy_dda --help              # DDA data analysis pipeline
timsim --help                 # Synthetic raw data generation
timsim_gui                    # GUI for timsim
imspy_ccs --help              # CCS calibration
imspy_rescore_sage --help     # Sage rescoring
timsim_validate --help        # Simulation validation
```

## Testing

Tests use pytest with fixtures defined in `tests/conftest.py`. Key fixtures:
- `simple_mz_array`, `simple_intensity_array` - Basic spectrum data
- `simple_peptide_sequence`, `modified_peptide_sequence` - Peptide sequences with UNIMOD mods
- `frame_data`, `precursor_frame_data` - TimsFrame test data
- `amino_acid_masses`, `tolerance` - Chemistry constants

Test files follow the pattern `test_<module>.py` with class-based organization:
```python
class TestMzSpectrum:
    def test_create_simple_spectrum(self, simple_mz_array, simple_intensity_array):
        spectrum = MzSpectrum(simple_mz_array, simple_intensity_array)
        assert len(spectrum.mz) == 5
```

## Common Data Types

| Python Class | Rust Type (mscore) | Connector Module |
|--------------|-------------------|------------------|
| `MzSpectrum` | `MzSpectrum` | `py_spectrum` |
| `TimsSpectrum` | `TimsSpectrum` | `py_spectrum` |
| `PeptideSequence` | `PeptideSequence` | `py_peptide` |
| `TimsFrame` | `TimsFrame` | `py_tims_frame` |
| `SumFormula` | `SumFormula` | `py_sumformula` |

## Adding New Functionality

1. If wrapping Rust code: implement in `mscore`/`rustdf` → add PyO3 binding in `imspy_connector` → create Python wrapper here
2. If pure Python: add to appropriate subpackage with type hints
3. Add tests in `tests/test_<module>.py` using existing fixtures

## Dependencies

- **imspy-connector**: Required Rust bindings (must match version)
- **tensorflow 2.20**: Deep learning models
- **sagepy**: Peptide search engine integration
- **mokapot**: Statistical validation
- **koinapy**: Pre-trained model access (Python <3.13 only)
