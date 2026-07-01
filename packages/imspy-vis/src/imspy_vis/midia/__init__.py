"""MIDIA / DIA-PASEF interactive clustering visualization on the imspy-core backend.

A modern reconstruction of the proteolizard-midia Voila tool: load a retention-time slice of
a run, then tune HDBSCAN parameters live to cluster precursor (3D) or MIDIA fragment (4D)
point clouds. No C++/opentims or proteolizard dependencies — everything is derived from the
run's own metadata via :class:`imspy_core.timstof.dia.TimsDatasetDIA`.
"""

from .data import MidiaExperiment, MidiaSlice
from .clustering import (
    cluster_precursors_dbscan,
    cluster_precursors_hdbscan,
    cluster_midia_hdbscan,
    calculate_statistics,
)
from .transforms import peak_width_preserving_mz_transform, calculate_mz_tick_spacing
from .widgets import (
    MidiaVis,
    DataPanel,
    PrecursorHDBSCANPanel,
    MidiaHDBSCANPanel,
)

__all__ = [
    "MidiaExperiment", "MidiaSlice",
    "cluster_precursors_dbscan", "cluster_precursors_hdbscan", "cluster_midia_hdbscan",
    "calculate_statistics",
    "peak_width_preserving_mz_transform", "calculate_mz_tick_spacing",
    "MidiaVis", "DataPanel", "PrecursorHDBSCANPanel", "MidiaHDBSCANPanel",
]
