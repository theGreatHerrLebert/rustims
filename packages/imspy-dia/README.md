# imspy-dia

DIA processing and clustering for timsTOF IMS data.

## Overview

`imspy-dia` provides tools for Data-Independent Acquisition (DIA) data analysis from Bruker timsTOF instruments:

- **Hierarchical DIA feature extraction** - Extract MS1 precursor and MS2 fragment clusters
- **Pseudo spectrum construction** - Build pseudo spectra from correlated precursor-fragment pairs
- **Isotopic feature grouping** - Group isotope clusters into features with charge state assignment

## Installation

```bash
pip install imspy-dia
```

## Dependencies

- `imspy-core>=0.4.0` - Core data structures
- `imspy-connector>=0.4.0` - Rust bindings
- `pandas>=2.0.0`
- `numpy>=1.24.0`

Optional dependencies:
- `matplotlib` - For cluster reports (`pip install imspy-dia[reporting]`)
- `torch` - For neural network-based extraction (`pip install imspy-dia[torch]`)

## Quick Start

```python
from imspy_dia import TimsDatasetDIAClustering, build_simple_features_from_clusters

# Open a DIA dataset with clustering capabilities
ds = TimsDatasetDIAClustering("/path/to/data.d")

# Plan TOF-scan windows for MS1
plan = ds.plan_tof_scan_windows(tof_step=8, rt_window_sec=30.0, rt_hop_sec=15.0)

# Extract precursor clusters (requires peak detection)
# ... see full documentation for peak detection workflow

# Build isotopic features from precursor clusters
features = build_simple_features_from_clusters(precursor_clusters)

# Build pseudo spectra
pseudo_spectra = ds.build_pseudo_spectra_geom(
    ms1_clusters=precursor_clusters,
    ms2_clusters=fragment_clusters,
    features=features,
)
```

## CLI Tools

### Cluster Report

Generate a PDF report summarizing cluster statistics:

```bash
imspy-cluster-report --in clusters.parquet --out report.pdf --json summary.json
```

## License

MIT License - see LICENSE file for details.
