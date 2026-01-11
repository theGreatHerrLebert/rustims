"""Simulation module for timsTOF data generation.

This module provides tools for simulating timsTOF mass spectrometry data,
including peptide ionization, quadrupole transmission, and frame building.

New API (recommended):
    - builders.create_frame_builder: Factory for creating frame builders
    - builders.DIAFrameBuilder: Unified DIA frame builder
    - builders.DDAFrameBuilder: DDA frame builder
    - data.SimulationDatabase: Database access for simulation data
    - data.TransmissionHandle: Ion transmission calculations

Legacy API (deprecated):
    - TimsTofSyntheticFrameBuilderDIA (use DIAFrameBuilder instead)
    - TimsTofSyntheticFrameBuilderDDA (use DDAFrameBuilder instead)
    - TimsTofLazyFrameBuilderDIA (use DIAFrameBuilder(lazy=True) instead)
    - SyntheticExperimentDataHandle (use SimulationDatabase instead)
    - TimsTofSyntheticsDataHandleRust (use TransmissionHandle instead)
"""

# New clean API
from .core import FrameBuilder, AnnotatedFrameBuilder, PyO3Wrapper
from .builders import (
    create_frame_builder,
    AcquisitionMode,
    LoadingStrategy,
    DIAFrameBuilder,
    DDAFrameBuilder,
)
from .data import SimulationDatabase, SimulationDatabaseDIA, TransmissionHandle

# Legacy imports for backward compatibility
# These are re-exported from their original modules
from .experiment import (
    TimsTofSyntheticFrameBuilderDIA,
    TimsTofSyntheticFrameBuilderDDA,
    TimsTofLazyFrameBuilderDIA,
    TimsTofSyntheticPrecursorFrameBuilder,
    SyntheticExperimentDataHandle,
    SyntheticExperimentDataHandleDIA,
)
from .handle import TimsTofSyntheticsDataHandleRust

__all__ = [
    # New API
    "FrameBuilder",
    "AnnotatedFrameBuilder",
    "PyO3Wrapper",
    "create_frame_builder",
    "AcquisitionMode",
    "LoadingStrategy",
    "DIAFrameBuilder",
    "DDAFrameBuilder",
    "SimulationDatabase",
    "SimulationDatabaseDIA",
    "TransmissionHandle",
    # Legacy (for backward compatibility)
    "TimsTofSyntheticFrameBuilderDIA",
    "TimsTofSyntheticFrameBuilderDDA",
    "TimsTofLazyFrameBuilderDIA",
    "TimsTofSyntheticPrecursorFrameBuilder",
    "SyntheticExperimentDataHandle",
    "SyntheticExperimentDataHandleDIA",
    "TimsTofSyntheticsDataHandleRust",
]
