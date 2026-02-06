"""
TimsTOF module for imspy_core.

Contains classes for reading and processing timsTOF data.
"""

from .data import TimsDataset, AcquisitionMode
from .frame import TimsFrame, TimsFrameVectorized
from .slice import TimsSlice, TimsSliceVectorized, TimsPlane
from .dda import TimsDatasetDDA, PrecursorDDA, FragmentDDA
from .dia import TimsDatasetDIA
from .quadrupole import (
    TimsTofQuadrupoleDDA, TimsTofQuadrupoleDIA, PasefMeta
)
from .collision import TimsTofCollisionEnergy, TimsTofCollisionEnergyDIA

__all__ = [
    'TimsDataset', 'AcquisitionMode',
    'TimsFrame', 'TimsFrameVectorized',
    'TimsSlice', 'TimsSliceVectorized', 'TimsPlane',
    'TimsDatasetDDA', 'PrecursorDDA', 'FragmentDDA',
    'TimsDatasetDIA',
    'TimsTofQuadrupoleDDA', 'TimsTofQuadrupoleDIA', 'PasefMeta',
    'TimsTofCollisionEnergy', 'TimsTofCollisionEnergyDIA',
]
