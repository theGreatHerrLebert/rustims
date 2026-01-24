# imspy-search

Database search functionality for timsTOF proteomics data using sagepy.

## Installation

```bash
pip install imspy-search
```

## Features

- **Database Search**: SAGE-based database search for timsTOF DDA data
- **PSM Rescoring**: Machine learning-based rescoring of peptide-spectrum matches
- **FDR Control**: Target-decoy competition and q-value estimation
- **MGF Support**: Parse and search Bruker DataAnalysis MGF files
- **CLI Tools**: Command-line interfaces for common workflows

## Quick Start

```python
from imspy_search import (
    extract_timstof_dda_data,
    get_searchable_spec,
    generate_balanced_rt_dataset,
    generate_balanced_im_dataset,
)

# Extract DDA data for database search
fragments = extract_timstof_dda_data(
    path="path/to/data.d",
    num_threads=16,
)
```

## CLI Tools

### imspy-dda
Full DDA search pipeline with intensity prediction and rescoring:
```bash
imspy-dda /path/to/data /path/to/fasta.fasta --config config.toml
```

### imspy-ccs
Extract CCS values from DDA data for machine learning:
```bash
imspy-ccs --raw_data_path /path/to/data --fasta_path /path/to/fasta.fasta
```

### imspy-rescore-sage
Rescore SAGE search results with deep learning features:
```bash
imspy-rescore-sage results.tsv fragments.tsv /output/path
```

## Submodules

- **utility**: Core utility functions for database search
- **sage_output_utility**: SAGE output processing and rescoring
- **mgf**: MGF file parsing for sagepy queries
- **rescoring**: PSM rescoring with deep learning features
- **dda_extensions**: TimsDatasetDDA extensions for sagepy
- **cli/**: Command-line interface tools

## Dependencies

- **imspy-core**: Core data structures (required)
- **imspy-predictors**: ML predictors for CCS, RT, intensity (required)
- **sagepy**: SAGE database search framework (required)
- **mokapot**: Machine learning for PSM scoring (required)

## Related Packages

- **imspy-core**: Core data structures and timsTOF readers
- **imspy-predictors**: ML-based predictors
- **imspy-simulation**: Simulation tools for timsTOF data
- **imspy-gui**: GUI visualization tools

## License

MIT License - see LICENSE file for details.
