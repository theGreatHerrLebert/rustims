# imspy-simulation

TimsTOF data simulation tools for proteomics.

## Installation

```bash
pip install imspy-simulation
```

For search integration (validation workflows):

```bash
pip install imspy-simulation[search]
```

## Features

- **Frame Builders**: DIA and DDA frame simulation with annotation support
- **TimSim**: Complete simulation pipeline for synthetic timsTOF data
- **Validation**: Tools for validating simulated data against search results
- **Isotope Simulation**: Accurate isotope distribution generation
- **TDF Writing**: Write simulated data to Bruker TDF format

## Quick Start

```python
from imspy_simulation import (
    DIAFrameBuilder,
    DDAFrameBuilder,
    SimulationDatabase,
    TransmissionHandle,
    create_frame_builder,
    AcquisitionMode,
)

# Create a DIA frame builder
frame_builder = DIAFrameBuilder(
    database_path="path/to/synthetic_data.db",
    num_threads=16,
)

# Build frames
frames = frame_builder.build_frames([1, 2, 3])
```

## CLI Tools

### timsim
Full simulation pipeline:
```bash
timsim --config config.toml --output /path/to/output
```

## Submodules

- **builders/**: Frame builder implementations (DIA, DDA)
- **core/**: Core protocols and wrappers
- **data/**: Simulation database and transmission handling
- **timsim/**: TimSim simulation pipeline
  - **jobs/**: Individual simulation steps
  - **integration/**: Integration workflows
  - **validate/**: Validation tools

## Dependencies

- **imspy-core**: Core data structures (required)
- **imspy-predictors**: ML predictors for CCS, RT, intensity (required)
- **imspy-search**: Database search for validation (optional)

## Related Packages

- **imspy-core**: Core data structures and timsTOF readers
- **imspy-predictors**: ML-based predictors
- **imspy-search**: Database search functionality
- **imspy-gui**: GUI visualization tools

## License

MIT License - see LICENSE file for details.
