"""
imspy_dia - DIA processing and clustering for timsTOF IMS data.

This package provides tools for:
- DIA dataset handling with clustering capabilities
- Hierarchical feature extraction from MS1 and MS2 data
- Pseudo spectrum construction
- Isotopic feature grouping

For basic timsTOF data handling, use imspy-core.
For ML-based predictions, use imspy-predictors.
"""

__version__ = "0.4.0"

# DIA dataset with clustering
from imspy_dia.dia import (
    TimsDatasetDIAClustering,
    CandidateOpts,
    ScoredHit,
    FragmentIndex,
    # I/O helpers
    save_clusters_bin,
    load_clusters_bin,
    save_clusters_parquet,
    load_clusters_parquet,
    save_pseudo_spectra_bin,
    load_pseudo_spectra_bin,
)

# Clustering data structures
from imspy_dia.clustering import (
    Fit1D,
    RtPeak1D,
    ImPeak1D,
    TofScanWindowGrid,
    TofScanPlan,
    TofScanPlanGroup,
    RawPoints,
    ClusterResult1D,
    TofRtGrid,
    AssignmentResult,
    PseudoBuildResult,
    PseudoFragment,
    PseudoSpectrum,
)

# Feature extraction
from imspy_dia.feature import (
    AveragineLut,
    SimpleFeatureParams,
    SimpleFeature,
    build_simple_features_from_clusters,
)

__all__ = [
    # Version
    '__version__',
    # DIA Dataset
    'TimsDatasetDIAClustering',
    'CandidateOpts',
    'ScoredHit',
    'FragmentIndex',
    # I/O
    'save_clusters_bin',
    'load_clusters_bin',
    'save_clusters_parquet',
    'load_clusters_parquet',
    'save_pseudo_spectra_bin',
    'load_pseudo_spectra_bin',
    # Clustering data
    'Fit1D',
    'RtPeak1D',
    'ImPeak1D',
    'TofScanWindowGrid',
    'TofScanPlan',
    'TofScanPlanGroup',
    'RawPoints',
    'ClusterResult1D',
    'TofRtGrid',
    'AssignmentResult',
    'PseudoBuildResult',
    'PseudoFragment',
    'PseudoSpectrum',
    # Feature extraction
    'AveragineLut',
    'SimpleFeatureParams',
    'SimpleFeature',
    'build_simple_features_from_clusters',
]
