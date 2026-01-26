# imspy-core

Core data structures and utilities for processing timsTOF ion mobility spectrometry data.

## Installation

```bash
pip install imspy-core
```

## Features

- **Data Structures**: MzSpectrum, TimsSpectrum, PeptideSequence, and more
- **Chemistry Utilities**: Elements, amino acids, UNIMOD modifications, CCS/mobility conversions
- **TimsTOF Readers**: Read DDA and DIA datasets from Bruker timsTOF instruments
- **Low Dependencies**: Only essential packages (numpy, pandas, scipy, numba)

## Quick Start

```python
from imspy_core.timstof import TimsDatasetDDA
from imspy_core.data import PeptideSequence

# Read a DDA dataset
dataset = TimsDatasetDDA("/path/to/data.d")
frame = dataset.get_tims_frame(1)
print(frame)

# Work with peptides
peptide = PeptideSequence("PEPTIDEK")
print(f"Mass: {peptide.mono_isotopic_mass}")
```

## Related Packages

- **imspy-predictors**: ML-based predictions (CCS, RT, intensity) - requires TensorFlow
- **imspy-search**: Database search functionality - requires sagepy, mokapot
- **imspy-simulation**: Simulation tools for timsTOF data
- **imspy-vis**: Visualization tools - requires Plotly, Matplotlib

## License

MIT License - see LICENSE file for details.
