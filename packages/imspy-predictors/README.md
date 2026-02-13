# imspy-predictors

ML-based predictors for CCS, retention time, and fragment intensity in mass spectrometry.

## Installation

```bash
pip install imspy-predictors
```

For remote model access via Koina servers:

```bash
pip install imspy-predictors[koina]
```

## Features

- **CCS Prediction**: Deep learning models for collision cross section / ion mobility prediction
- **Retention Time Prediction**: GRU-based retention time predictors
- **Fragment Intensity Prediction**: Prosit 2023 timsTOF intensity predictor
- **Charge State Prediction**: Binomial and deep learning charge state distribution models
- **Koina Integration**: Access remote prediction models via [Koina](https://koina.wilhelmlab.org) servers (optional)

### Available Koina Remote Models

When using `pip install imspy-predictors[koina]`, remote models can be configured in TimSim's `[models]` TOML section:

| Task | Model Name |
|------|-----------|
| **RT** | `Deeplc_hela_hf`, `Chronologer_RT`, `AlphaPeptDeep_rt_generic`, `Prosit_2019_irt` |
| **CCS** | `AlphaPeptDeep_ccs_generic`, `IM2Deep` |
| **Intensity** | `prosit`, `alphapeptdeep`, `ms2pip` |

```toml
[models]
rt_model = "AlphaPeptDeep_rt_generic"
ccs_model = ""                          # "" = local PyTorch model
intensity_model = "prosit"
```

## Quick Start

```python
from imspy_predictors import (
    load_deep_ccs_predictor,
    load_deep_retention_time_predictor,
    Prosit2023TimsTofWrapper,
)

# Load CCS predictor
ccs_model = load_deep_ccs_predictor()

# Load RT predictor
rt_model = load_deep_retention_time_predictor()

# Load intensity predictor
intensity_model = Prosit2023TimsTofWrapper()
```

## Submodules

- **ccs/**: CCS / ion mobility prediction
- **rt/**: Retention time prediction
- **intensity/**: Fragment intensity prediction (Prosit)
- **ionization/**: Charge state distribution prediction
- **koina_models/**: Koina remote model access (requires `koinapy`)
- **utilities/**: Tokenizers for ML models

## Dependencies

- **imspy-core**: Core data structures (required)
- **PyTorch**: Deep learning framework (required)
- **koinapy**: Koina API client (optional, for remote models)

## Optional Dependencies

Some functionality requires additional packages:

- **imspy-search**: For PSM-based predictions using sagepy
- **imspy-simulation**: For simulation utilities (e.g., flatten_prosit_array)

## Related Packages

- **imspy-core**: Core data structures and timsTOF readers
- **imspy-search**: Database search functionality
- **imspy-simulation**: Simulation tools for timsTOF data
- **imspy-vis**: Visualization tools

## License

MIT License - see LICENSE file for details.
