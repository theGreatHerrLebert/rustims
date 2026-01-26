"""
Unified PyTorch models for peptide property prediction.

This module provides a transformer-based architecture for predicting:
- Collision Cross Section (CCS) / Ion Mobility
- Retention Time (RT)
- Charge State Distribution
- Fragment Ion Intensities

The architecture uses a shared encoder pre-trained on fragment intensity prediction
(31M samples) with task-specific heads for each prediction task.

Supports instrument-specific predictions via instrument type encoding.
"""

from imspy_predictors.models.transformer import (
    PeptideTransformer,
    PositionalEncoding,
    PeptideTransformerConfig,
)
from imspy_predictors.models.heads import (
    CCSHead,
    RTHead,
    ChargeHead,
    IntensityHead,
    SquareRootProjectionLayer,
    # Instrument utilities
    INSTRUMENT_TYPES,
    INSTRUMENT_TO_ID,
    NUM_INSTRUMENT_TYPES,
    get_instrument_id,
)
from imspy_predictors.models.unified import (
    UnifiedPeptideModel,
    TaskLoss,
)

__all__ = [
    # Base encoder
    "PeptideTransformer",
    "PositionalEncoding",
    "PeptideTransformerConfig",
    # Task heads
    "CCSHead",
    "RTHead",
    "ChargeHead",
    "IntensityHead",
    "SquareRootProjectionLayer",
    # Unified model
    "UnifiedPeptideModel",
    "TaskLoss",
    # Instrument utilities
    "INSTRUMENT_TYPES",
    "INSTRUMENT_TO_ID",
    "NUM_INSTRUMENT_TYPES",
    "get_instrument_id",
]
