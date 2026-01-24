"""
Data module for imspy_core.

Contains spectrum and peptide data structures.
"""

from .spectrum import MzSpectrum, IndexedMzSpectrum, MzSpectrumVectorized, TimsSpectrum
from .peptide import (
    PeptideSequence, PeptideIon, PeptideProductIon,
    PeptideProductIonSeries, PeptideProductIonSeriesCollection
)

__all__ = [
    'MzSpectrum', 'IndexedMzSpectrum', 'MzSpectrumVectorized', 'TimsSpectrum',
    'PeptideSequence', 'PeptideIon', 'PeptideProductIon',
    'PeptideProductIonSeries', 'PeptideProductIonSeriesCollection'
]
