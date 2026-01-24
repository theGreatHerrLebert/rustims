# Plan: Split imspy Monolith into Modular Sub-packages

## User Preferences
- **Package count**: 5 packages (full dependency isolation)
- **Backward compatibility**: Low priority (clean break)
- **Repository structure**: Monorepo (`rustims/packages/`)
- **Pain points**: All - TensorFlow, GUI deps, sagepy/mokapot

## Problem Statement

The `imspy` package (~35,000 lines) requires all dependencies even for simple use cases:
- TensorFlow 2.20 (~2GB) - needed only for ML predictions
- PyQt5/VTK (~500MB) - needed only for GUI
- sagepy/mokapot - needed only for database search
- mendeleev, dlomix, koinapy - various specialized uses

Users who just want to read timsTOF data must install the entire dependency tree.

## Proposed Architecture: 5-Package Split

```
                      imspy (meta-package)
                             |
          +------------------+------------------+
          |                  |                  |
    imspy-search      imspy-simulation    imspy-gui
          |                  |                  |
          +--------+---------+                  |
                   |                            |
             imspy-predictors-------------------+
                   |
             imspy-core
                   |
            imspy-connector (unchanged)
```

### Package 1: imspy-core (~5,000 lines)

**Contents:**
- `core/` - RustWrapperObject base class
- `data/` - MzSpectrum, PeptideSequence wrappers
- `chemistry/` - Elements, amino acids, UNIMOD, mobility, constants
- `timstof/` - TimsDataset, TimsFrame, TimsSlice (excluding dbsearch/)
- `utility/` - General helpers

**Dependencies:** imspy-connector, numpy, pandas, scipy, mendeleev

**Refactoring required:**
- `timstof/dda.py`: Move sagepy imports to lazy loading or extension module
  - Lines 11-14 import sagepy at module level
  - `to_sage_precursor()` method (lines 88-100) needs sagepy
  - `get_sage_processed_precursors()` method (lines 208-261) needs sagepy

### Package 2: imspy-predictors (~3,000 lines)

**Contents:**
- `algorithm/ccs/` - CCS/ion mobility predictors
- `algorithm/rt/` - Retention time predictors
- `algorithm/intensity/` - Prosit intensity predictor
- `algorithm/ionization/` - Charge state predictors
- `algorithm/koina_models/` - Koina API access
- `algorithm/utility.py`, `hashing.py`, `mixture.py`

**Dependencies:** imspy-core, tensorflow==2.20.*, dlomix, koinapy (optional)

### Package 3: imspy-search (~3,500 lines)

**Contents:**
- `timstof/dbsearch/` - All search modules
- `algorithm/rescoring.py` - PSM rescoring
- Sage extension methods for `PrecursorDDA` and `TimsDatasetDDA`

**Dependencies:** imspy-predictors, sagepy, mokapot, scikit-learn, numba

**Entry points:** imspy_dda, imspy_search, imspy_ccs, imspy_rescore_sage

### Package 4: imspy-simulation (~5,000 lines)

**Contents:**
- `simulation/` - All simulation modules
  - `annotation.py`, `experiment.py`, `acquisition.py`
  - `builders/`, `data/`, `core/`
  - `timsim/` - simulator, jobs/, validate/, integration/
- `vis/` - Visualization utilities (optional: could be separate)

**Dependencies:** imspy-predictors, sagepy (for digestion), zstd, pyarrow

**Entry points:** timsim, timsim_validate, timsim_compare, timsim_integration_*

### Package 5: imspy-gui (~2,000 lines)

**Contents:**
- `simulation/timsim/timsim_gui.py`
- `vis/frame_rendering.py`, `vis/pointcloud.py`

**Dependencies:** imspy-simulation, PyQt5, VTK, qdarkstyle, matplotlib

**Entry point:** timsim_gui

### Package 6: imspy (Optional Meta-package)

Simple convenience package that installs all sub-packages (no re-exports):
```toml
# pyproject.toml
[project]
name = "imspy"
dependencies = [
    "imspy-core",
    "imspy-predictors",
    "imspy-search",
    "imspy-simulation",
]

[project.optional-dependencies]
gui = ["imspy-gui"]
all = ["imspy-gui"]
```

Users who want everything can `pip install imspy[all]`, but must use new import paths.

## Key Refactoring Tasks

### 1. Decouple sagepy from dda.py

Current problem (`timstof/dda.py`):
```python
from sagepy.core import Precursor, Tolerance, ProcessedSpectrum, ...  # Line 11
```

Solution options:
- **Option A**: Lazy imports - move sagepy imports inside methods that use them
- **Option B**: Extension pattern - keep base classes in core, add Sage methods via mixins in imspy-search
- **Option C**: Protocol pattern - define abstract interface, concrete implementation in search package

### 2. Handle circular import in peptide.py

Already solved with lazy import pattern (lines 9-18 of `data/peptide.py`):
```python
if TYPE_CHECKING:
    from imspy.simulation.annotation import MzSpectrumAnnotated

def _get_mz_spectrum_annotated():
    from imspy.simulation.annotation import MzSpectrumAnnotated
    return MzSpectrumAnnotated
```

This pattern should be replicated for other cross-package imports.

### 3. Separate ML model weights

Move pretrained weights from `algorithm/pretrained/` to package resources using `importlib.resources`.

## Installation Scenarios

| Use Case | Install Command | Size |
|----------|-----------------|------|
| Read timsTOF data only | `pip install imspy-core` | ~200MB |
| + ML predictions | `pip install imspy-predictors` | ~2.2GB |
| + Database search | `pip install imspy-search` | ~2.5GB |
| + Simulation | `pip install imspy-simulation` | ~2.5GB |
| + GUI | `pip install imspy-gui` | ~3GB |
| Everything (legacy) | `pip install imspy[all]` | ~3GB |

## Repository Structure (Monorepo)

```
rustims/
├── packages/
│   ├── imspy-core/
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── imspy_core/
│   │           ├── __init__.py
│   │           ├── core/
│   │           ├── data/
│   │           ├── chemistry/
│   │           ├── timstof/
│   │           └── utility/
│   ├── imspy-predictors/
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── imspy_predictors/
│   │           ├── __init__.py
│   │           ├── ccs/
│   │           ├── rt/
│   │           ├── intensity/
│   │           ├── ionization/
│   │           └── koina_models/
│   ├── imspy-search/
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── imspy_search/
│   │           ├── __init__.py
│   │           ├── dbsearch/
│   │           └── rescoring.py
│   ├── imspy-simulation/
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── imspy_simulation/
│   │           ├── __init__.py
│   │           ├── annotation.py
│   │           ├── builders/
│   │           ├── timsim/
│   │           └── ...
│   ├── imspy-gui/
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── imspy_gui/
│   │           ├── __init__.py
│   │           ├── timsim_gui.py
│   │           └── vis/
│   └── imspy/  (meta-package, optional)
│       └── pyproject.toml
├── imspy/  (OLD - to be archived/deleted after migration)
├── mscore/
├── rustdf/
└── imspy_connector/
```

Benefits of this structure:
- Single CI/CD pipeline can build/test all packages
- Atomic releases with version synchronization
- Easy cross-package development and testing
- `src/` layout follows Python best practices

## Migration Path

### Phase 1: Create imspy-core
1. Extract core modules
2. Refactor `dda.py` to remove sagepy dependency from base classes
3. Publish to PyPI as v0.4.0

### Phase 2: Create imspy-predictors
1. Extract algorithm modules
2. Add lazy TensorFlow loading
3. Publish to PyPI

### Phase 3: Create imspy-search
1. Extract dbsearch modules
2. Add Sage extension methods
3. Publish to PyPI

### Phase 4: Create imspy-simulation
1. Extract simulation modules
2. Publish to PyPI

### Phase 5: Create imspy-gui
1. Extract GUI module
2. Publish to PyPI

### Phase 6: Create meta-package
1. Create imspy meta-package with re-exports
2. Add deprecation warnings for old import paths
3. Publish as v0.5.0

## Breaking Changes (Clean Break)

### Import paths change
| Old | New |
|-----|-----|
| `from imspy.data.spectrum import MzSpectrum` | `from imspy_core.data.spectrum import MzSpectrum` |
| `from imspy.data.peptide import PeptideSequence` | `from imspy_core.data.peptide import PeptideSequence` |
| `from imspy.timstof.data import TimsDataset` | `from imspy_core.timstof.data import TimsDataset` |
| `from imspy.timstof.dda import TimsDatasetDDA` | `from imspy_core.timstof.dda import TimsDatasetDDA` |
| `from imspy.chemistry.mobility import ...` | `from imspy_core.chemistry.mobility import ...` |
| `from imspy.algorithm.ccs import ...` | `from imspy_predictors.ccs import ...` |
| `from imspy.algorithm.rt import ...` | `from imspy_predictors.rt import ...` |
| `from imspy.algorithm.intensity import ...` | `from imspy_predictors.intensity import ...` |
| `from imspy.timstof.dbsearch import ...` | `from imspy_search.dbsearch import ...` |
| `from imspy.simulation import ...` | `from imspy_simulation import ...` |
| `from imspy.simulation.timsim import ...` | `from imspy_simulation.timsim import ...` |

### CLI commands unchanged
All CLI entry points keep their names but are installed by different packages:
- `imspy_dda`, `imspy_ccs`, `imspy_search`, `imspy_rescore_sage` → imspy-search
- `timsim`, `timsim_validate`, `timsim_compare` → imspy-simulation
- `timsim_gui` → imspy-gui

### Migration guide for users
Users will need to:
1. Update `pip install imspy` to `pip install imspy-core` (or appropriate package)
2. Find-and-replace imports using the table above
3. No code logic changes required - just import paths

## Verification Plan

### Per-package verification:
```bash
# imspy-core
cd packages/imspy-core
pip install -e .
python -c "from imspy_core.data.spectrum import MzSpectrum; print('OK')"
python -c "from imspy_core.timstof.dda import TimsDatasetDDA; print('OK')"
pytest tests/

# imspy-predictors (verify TensorFlow loads)
cd packages/imspy-predictors
pip install -e .
python -c "from imspy_predictors.ccs.predictors import DeepPeptideIonMobilityApex; print('OK')"
pytest tests/

# imspy-search (verify CLI works)
cd packages/imspy-search
pip install -e .
imspy_dda --help
imspy_ccs --help

# imspy-simulation
cd packages/imspy-simulation
pip install -e .
timsim --help
timsim_validate --help

# imspy-gui
cd packages/imspy-gui
pip install -e .
python -c "from imspy_gui.timsim_gui import main; print('OK')"
```

### Integration test:
```bash
# Install all packages together
pip install packages/imspy-core packages/imspy-predictors packages/imspy-search packages/imspy-simulation packages/imspy-gui

# Run existing integration tests
pytest imspy/tests/  # Original test suite should pass with new imports
```

### Dependency isolation test:
```bash
# Verify core works without TensorFlow
python -m venv test_core_only
source test_core_only/bin/activate
pip install packages/imspy-core
python -c "import imspy_core; print('Core works without TensorFlow')"
pip list | grep -i tensorflow  # Should show nothing
```

## Implementation Steps

### Step 1: Create packages/ directory structure
```bash
mkdir -p packages/{imspy-core,imspy-predictors,imspy-search,imspy-simulation,imspy-gui}
```

### Step 2: Create imspy-core package
1. Create `packages/imspy-core/pyproject.toml`
2. Copy modules: `core/`, `data/`, `chemistry/`, `timstof/` (without dbsearch/), `utility/`
3. **Critical refactor**: Remove sagepy imports from `timstof/dda.py`
   - Move `to_sage_precursor()` and `get_sage_processed_precursors()` to imspy-search
   - Keep base `PrecursorDDA`, `TimsDatasetDDA`, `FragmentDDA` classes
4. Update all internal imports to use `imspy_core.*`
5. Add tests from `tests/test_spectrum.py`, `tests/test_peptide.py`, `tests/test_tims_frame.py`

### Step 3: Create imspy-predictors package
1. Create `packages/imspy-predictors/pyproject.toml` with tensorflow dependency
2. Copy modules: `algorithm/ccs/`, `algorithm/rt/`, `algorithm/intensity/`, `algorithm/ionization/`, `algorithm/koina_models/`, `algorithm/utility.py`, `algorithm/hashing.py`, `algorithm/mixture.py`
3. Update imports to use `imspy_core.*` for base types
4. Add tests from `tests/test_sage_interface.py` (predictor parts)

### Step 4: Create imspy-search package
1. Create `packages/imspy-search/pyproject.toml` with sagepy, mokapot
2. Copy modules: `timstof/dbsearch/`
3. Add Sage extension methods (moved from dda.py):
   ```python
   # imspy_search/dda_extensions.py
   from imspy_core.timstof.dda import PrecursorDDA, TimsDatasetDDA
   from sagepy.core import Precursor, ...

   def to_sage_precursor(precursor: PrecursorDDA) -> Precursor:
       ...  # Code from original dda.py lines 88-100

   def get_sage_processed_precursors(dataset: TimsDatasetDDA, ...) -> List[ProcessedSpectrum]:
       ...  # Code from original dda.py lines 208-261
   ```
4. Move `algorithm/rescoring.py` here
5. Add CLI entry points

### Step 5: Create imspy-simulation package
1. Create `packages/imspy-simulation/pyproject.toml`
2. Copy modules: entire `simulation/` directory
3. Update imports to use `imspy_core.*` and `imspy_predictors.*`
4. Add CLI entry points
5. Handle `data/peptide.py` → `simulation/annotation.py` circular import (already solved with lazy import)

### Step 6: Create imspy-gui package
1. Create `packages/imspy-gui/pyproject.toml` with PyQt5, VTK
2. Move: `simulation/timsim/timsim_gui.py`, `vis/`
3. Add CLI entry point for `timsim_gui`

### Step 7: Create meta-package (optional)
1. Create `packages/imspy/pyproject.toml` with all packages as dependencies
2. No code, just dependency aggregation

### Step 8: Update CI/CD
1. Create GitHub Actions workflow for building each package
2. Add version synchronization (all packages share same version)
3. Add cross-package integration tests

## Critical Files

- `imspy/pyproject.toml` - Current dependency list
- `imspy/imspy/timstof/dda.py` - Needs refactoring for sagepy decoupling
- `imspy/imspy/data/peptide.py` - Has lazy import pattern to follow
- `imspy/imspy/algorithm/ccs/predictors.py` - TensorFlow-dependent code
