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

For KOINA remote model support (optional):

```bash
pip install imspy-predictors[koina]
```

## Features

- **Frame Builders**: DIA and DDA frame simulation with annotation support
- **TimSim**: Complete simulation pipeline for synthetic timsTOF data
- **Prediction Models**: Local PyTorch models with optional KOINA remote model support (see [Prediction Models](#prediction-models))
- **Validation**: Tools for validating simulated data against search results
- **Integration Testing (EVAL)**: Automated validation against DiaNN, FragPipe, and Sage (see [Integration Testing](#integration-testing))
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
timsim config.toml
timsim config.toml --save-path output.d --reference-path reference.d --fasta-path proteome.fasta
```

## Prediction Models

TimSim uses deep learning models for retention time, ion mobility (CCS), and fragment intensity prediction. By default, local PyTorch models are used. Optionally, remote models can be accessed via [KOINA](https://koina.wilhelmlab.org) servers:

```toml
[models]
rt_model = ""              # "" = local (default), or e.g. "Deeplc_hela_hf"
ccs_model = ""             # "" = local (default), or e.g. "AlphaPeptDeep_ccs_generic"
intensity_model = ""       # "" = local (default), or e.g. "prosit", "alphapeptdeep"
```

Requires `pip install imspy-predictors[koina]` for remote models. Falls back to local models if KOINA is unreachable. See [SIMULATOR_README.md](SIMULATOR_README.md) for the full list of available models.

## Integration Testing

The EVAL pipeline validates simulated datasets against production proteomics search engines:

```bash
python -m imspy_simulation.timsim.integration.sim --env env.toml --list
python -m imspy_simulation.timsim.integration.sim --env env.toml --test IT-DIA-HELA
python -m imspy_simulation.timsim.integration.eval --env env.toml --test IT-DIA-HELA
```

See the [Validation README](src/imspy_simulation/timsim/integration/VALIDATION_README.md) for setup, available tests, and configuration details.

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
- **imspy-vis**: Visualization tools

## License

MIT License - see LICENSE file for details.
