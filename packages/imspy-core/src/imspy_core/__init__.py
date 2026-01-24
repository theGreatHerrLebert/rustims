"""
imspy_core - Core data structures and utilities for processing timsTOF ion mobility spectrometry data.

This package provides the foundational components for working with timsTOF data:
- Data structures for spectra and peptides
- Chemistry utilities (elements, amino acids, UNIMOD)
- TimsTOF dataset readers (DDA and DIA)
- General utilities

For ML-based predictions, install imspy-predictors.
For database search functionality, install imspy-search.
For simulation tools, install imspy-simulation.
"""

__version__ = "0.4.0"

# Core base classes
from imspy_core.core import RustWrapperObject

# Data structures
from imspy_core.data import (
    MzSpectrum, IndexedMzSpectrum, MzSpectrumVectorized, TimsSpectrum,
    PeptideSequence, PeptideIon, PeptideProductIon,
    PeptideProductIonSeries, PeptideProductIonSeriesCollection
)

# Chemistry
from imspy_core.chemistry import (
    MASS_PROTON, MASS_NEUTRON, MASS_ELECTRON, MASS_WATER,
    AMINO_ACID_MASSES, AMINO_ACIDS,
    UNIMOD_MASSES,
    one_over_k0_to_ccs, ccs_to_one_over_k0,
    SumFormula, calculate_mz
)

# TimsTOF
from imspy_core.timstof import (
    TimsDataset, AcquisitionMode,
    TimsFrame, TimsFrameVectorized,
    TimsSlice, TimsSliceVectorized,
    TimsDatasetDDA, PrecursorDDA, FragmentDDA,
    TimsDatasetDIA,
    TimsTofQuadrupoleDDA, TimsTofQuadrupoleDIA
)

# Utility
from imspy_core.utility import (
    re_index_indices, tokenize_unimod_sequence
)

__all__ = [
    # Version
    '__version__',
    # Core
    'RustWrapperObject',
    # Data
    'MzSpectrum', 'IndexedMzSpectrum', 'MzSpectrumVectorized', 'TimsSpectrum',
    'PeptideSequence', 'PeptideIon', 'PeptideProductIon',
    'PeptideProductIonSeries', 'PeptideProductIonSeriesCollection',
    # Chemistry
    'MASS_PROTON', 'MASS_NEUTRON', 'MASS_ELECTRON', 'MASS_WATER',
    'AMINO_ACID_MASSES', 'AMINO_ACIDS',
    'UNIMOD_MASSES',
    'one_over_k0_to_ccs', 'ccs_to_one_over_k0',
    'SumFormula', 'calculate_mz',
    # TimsTOF
    'TimsDataset', 'AcquisitionMode',
    'TimsFrame', 'TimsFrameVectorized',
    'TimsSlice', 'TimsSliceVectorized',
    'TimsDatasetDDA', 'PrecursorDDA', 'FragmentDDA',
    'TimsDatasetDIA',
    'TimsTofQuadrupoleDDA', 'TimsTofQuadrupoleDIA',
    # Utility
    're_index_indices', 'tokenize_unimod_sequence',
]
