"""
imspy_simulation - TimsTOF data simulation tools for proteomics.

This package provides tools for simulating timsTOF mass spectrometry data,
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

__version__ = "0.4.0"

# New clean API
from imspy_simulation.core import FrameBuilder, AnnotatedFrameBuilder, PyO3Wrapper
from imspy_simulation.builders import (
    create_frame_builder,
    AcquisitionMode,
    LoadingStrategy,
    DIAFrameBuilder,
    DDAFrameBuilder,
)
from imspy_simulation.data import SimulationDatabase, SimulationDatabaseDIA, TransmissionHandle

# Legacy imports for backward compatibility
from imspy_simulation.experiment import (
    TimsTofSyntheticFrameBuilderDIA,
    TimsTofSyntheticFrameBuilderDDA,
    TimsTofLazyFrameBuilderDIA,
    TimsTofSyntheticPrecursorFrameBuilder,
    SyntheticExperimentDataHandle,
    SyntheticExperimentDataHandleDIA,
)
from imspy_simulation.handle import TimsTofSyntheticsDataHandleRust

# Annotation classes
from imspy_simulation.annotation import (
    SourceType,
    SignalAttributes,
    ContributionSource,
    PeakAnnotation,
    MzSpectrumAnnotated,
    TimsFrameAnnotated,
    TimsSpectrumAnnotated,
)

# Utility functions
from imspy_simulation.utility import (
    flatten_prosit_array,
    sequences_to_all_ions,
    sequence_to_all_ions,
    get_acquisition_builder_resource_path,
    get_dilution_factors,
    get_ms_ms_window_layout_resource_path,
    read_acquisition_config,
    get_native_dataset_path,
    calculate_number_frames,
    calculate_mobility_spacing,
    iter_frame_batches,
)

__all__ = [
    # Version
    '__version__',
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
    # Annotation
    "SourceType",
    "SignalAttributes",
    "ContributionSource",
    "PeakAnnotation",
    "MzSpectrumAnnotated",
    "TimsFrameAnnotated",
    "TimsSpectrumAnnotated",
    # Utility
    "flatten_prosit_array",
    "sequences_to_all_ions",
    "sequence_to_all_ions",
    "get_acquisition_builder_resource_path",
    "get_dilution_factors",
    "get_ms_ms_window_layout_resource_path",
    "read_acquisition_config",
    "get_native_dataset_path",
    "calculate_number_frames",
    "calculate_mobility_spacing",
    "iter_frame_batches",
]
