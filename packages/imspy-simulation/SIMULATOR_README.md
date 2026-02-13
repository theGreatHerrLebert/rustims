# TIMSIM - timsTOF Mass Spectrometry Simulator

A high-fidelity proteomics simulation engine for Bruker timsTOF instruments. Generate realistic synthetic DIA-PASEF and DDA-PASEF datasets with full control over experimental parameters.

## Quick Start

### 1. Installation

```bash
# From PyPI (recommended)
pip install imspy-simulation

# With KOINA remote model support (optional)
pip install imspy-predictors[koina]
```

**Docker** (includes all dependencies + GPU support):
```bash
docker build -t rustims .
docker run --rm --gpus all -v /data:/workspace rustims timsim /workspace/config.toml
```

<details>
<summary>From source</summary>

```bash
source /path/to/your/env/bin/activate

# Install the Rust backend (requires maturin)
cd /path/to/rustims/imspy_connector
maturin develop --release

# Install Python packages
pip install -e /path/to/rustims/packages/imspy-core
pip install -e /path/to/rustims/packages/imspy-predictors
pip install -e /path/to/rustims/packages/imspy-simulation
```
</details>

### 2. Create a Configuration File

Create `my_simulation.toml`:

```toml
[paths]
save_path = "/path/to/output"
reference_path = "/path/to/blank_reference.d"
fasta_path = "/path/to/proteome.fasta"

[experiment]
experiment_name = "MyExperiment"
acquisition_type = "DIA"  # or "DDA"
gradient_length = 3600.0  # seconds
apply_fragmentation = true

[digestion]
n_proteins = 20000
num_peptides_total = 500000
num_sample_peptides = 150000
sample_peptides = true
sample_seed = 42
missed_cleavages = 2
min_len = 7
max_len = 30

[performance]
num_threads = -1  # auto-detect
use_gpu = true
```

### 3. Run Simulation

```bash
python -m imspy_simulation.timsim.simulator my_simulation.toml
```

## Key Concepts

### Reference .d Files

TIMSIM uses a **blank reference acquisition** to define the instrument layout:
- Frame structure (precursor/fragment frames)
- Scan geometry (mobility range, scan count)
- DIA windows or DDA selection parameters

You need a real timsTOF `.d` file as a template. The simulator will populate it with synthetic peptide signals.

### Simulation Pipeline

```
FASTA → Digestion → Model Selection → RT Prediction → IM Prediction →
                     (local/KOINA)
Fragment Intensity Prediction → Frame Assembly → .d File
```

1. **Digestion**: In-silico tryptic digest of proteins
2. **Model Selection**: Choose local PyTorch or KOINA remote models (see [Prediction Model Selection](#prediction-model-selection-koina))
3. **RT Prediction**: Deep learning model predicts retention times
4. **IM Prediction**: CCS/mobility prediction for each peptide ion
5. **Intensity Prediction**: Fragment ion intensities (local or KOINA models)
6. **Frame Assembly**: Signals placed into timsTOF frame structure

### Output Files

```
output_dir/
├── ExperimentName/
│   └── ExperimentName.d/      # Simulated timsTOF data
│       ├── analysis.tdf       # SQLite metadata
│       └── analysis.tdf_bin   # Binary peak data
├── synthetic_data.db          # Ground truth database
└── ExperimentName_preview.mp4 # Optional preview video
```

## Configuration Reference

### Essential Sections

#### `[paths]`
| Parameter | Description |
|-----------|-------------|
| `save_path` | Output directory |
| `reference_path` | Blank .d file for layout |
| `fasta_path` | Proteome FASTA file |

#### `[experiment]`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `experiment_name` | required | Name for output files |
| `acquisition_type` | `"DIA"` | `"DIA"` or `"DDA"` |
| `gradient_length` | `3600.0` | LC gradient in seconds |
| `apply_fragmentation` | `true` | Generate MS2 spectra |

#### `[digestion]`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_proteins` | `20000` | Max proteins to digest |
| `num_sample_peptides` | `150000` | Peptides in final dataset |
| `sample_seed` | `42` | Random seed for reproducibility |
| `missed_cleavages` | `2` | Allowed missed cleavages |
| `min_len` / `max_len` | `7` / `30` | Peptide length range |

### Prediction Model Selection (KOINA)

TimSim supports both local PyTorch models and remote [KOINA](https://koina.wilhelmlab.org) models for predictions. Configure via the `[models]` section:

```toml
[models]
rt_model = ""              # "" or "local" = local PyTorch (default)
ccs_model = ""             # "" or "local" = local PyTorch (default)
intensity_model = ""       # "" or "local" = local PyTorch (default)
```

**Available remote models:**

| Task | Model Name | Notes |
|------|-----------|-------|
| **RT** | `"Deeplc_hela_hf"` | DeepLC HeLa model |
| | `"Chronologer_RT"` | Chronologer RT predictor |
| | `"AlphaPeptDeep_rt_generic"` | AlphaPeptDeep generic RT |
| | `"Prosit_2019_irt"` | Prosit indexed RT |
| **CCS** | `"AlphaPeptDeep_ccs_generic"` | AlphaPeptDeep generic CCS |
| | `"IM2Deep"` | IM2Deep predictor |
| **Intensity** | `"prosit"` | Prosit 2023 timsTOF (max 30 AA, limited mods) |
| | `"alphapeptdeep"` | AlphaPeptDeep generic (supports phospho) |
| | `"ms2pip"` | ms2pip timsTOF 2024 |

**Prerequisites**: `pip install imspy-predictors[koina]`

If a KOINA server is unreachable, the simulator automatically falls back to local models.

### Advanced Features

#### Partial Fragmentation (Unfragmented Precursors)

Simulate incomplete fragmentation where some precursors survive intact:

```toml
[quad_transmission]
precursor_survival_min = 0.0   # Min fraction unfragmented
precursor_survival_max = 0.3   # Max fraction unfragmented (30%)
```

#### Noise from Real Data

Add realistic background noise sampled from reference acquisitions:

```toml
[noise]
add_real_data_noise = true
reference_noise_intensity_max = 150000
precursor_sample_fraction = 0.2
fragment_sample_fraction = 0.2
```

#### DDA-Specific Settings

```toml
[dda]
precursors_every = 10           # MS2 trigger frequency
precursor_intensity_threshold = 500
max_precursors = 25             # TopN
exclusion_width = 25            # Dynamic exclusion (scans)
selection_mode = "topN"
```

#### Multi-Species (HYE) Simulation

```toml
[proteome_mix]
proteome_mix = true

[paths]
fasta_path = "/path/to/fasta_dir"  # Directory with HUMAN.fasta, YEAST.fasta, etc.
```

#### Phosphoproteomics

```toml
[phosphorylation]
phospho_mode = true
```

#### Video Preview

```toml
[video_settings]
generate_preview_video = true
preview_video_max_frames = 100
preview_video_fps = 10
```

### Performance Tuning

```toml
[performance]
num_threads = -1        # -1 = auto-detect
batch_size = 256        # Peptide batch size
frame_batch_size = 500  # Frames per write batch
use_gpu = true          # Use CUDA for ML models
gpu_memory_limit_gb = 4
lazy_frame_assembly = false  # true for large datasets
```

## Ground Truth Database

The `synthetic_data.db` SQLite file contains complete ground truth:

```sql
-- Peptide information
SELECT * FROM peptides;

-- Ion-level data (charge states, m/z, mobility)
SELECT * FROM ions;

-- Fragment ions with predicted intensities
SELECT * FROM fragments;

-- Frame-level occurrence data
SELECT * FROM frame_occurrences;
```

Use this for validation against search engine results.

## Example Configurations

### Standard DIA (HeLa, 1hr)
```toml
[experiment]
acquisition_type = "DIA"
gradient_length = 3600.0

[digestion]
num_sample_peptides = 150000
```

### Fast DIA (15min)
```toml
[experiment]
acquisition_type = "DIA"
gradient_length = 900.0

[digestion]
num_sample_peptides = 50000
```

### DDA TopN
```toml
[experiment]
acquisition_type = "DDA"
gradient_length = 3600.0

[dda]
selection_mode = "topN"
max_precursors = 25
```

### Immunopeptidomics (HLA)
```toml
[digestion]
min_len = 8
max_len = 14
cleave_at = ""  # Non-specific

[charge_states]
binomial_charge_model = true
charge_state_one_probability = 0.15
```

## Integration Testing (EVAL Pipeline)

Validate simulated datasets against production proteomics search engines (DiaNN, FragPipe, Sage):

```bash
# List available integration tests
python -m imspy_simulation.timsim.integration.sim --env env.toml --list

# Run a simulation
python -m imspy_simulation.timsim.integration.sim --env env.toml --test IT-DIA-HELA

# Analyze and validate against ground truth
python -m imspy_simulation.timsim.integration.eval --env env.toml --test IT-DIA-HELA
```

Third-party analysis tools (DiaNN, FragPipe, Sage) must be installed separately — they are not bundled due to licensing. See the full [Validation README](src/imspy_simulation/timsim/integration/VALIDATION_README.md) for setup instructions and available test scenarios.

## Troubleshooting

### "Bruker SDK not found"
Install `opentims-bruker-bridge`:
```bash
pip install opentims-bruker-bridge
```

### Out of Memory
- Reduce `num_sample_peptides`
- Enable `lazy_frame_assembly = true`
- Reduce `batch_size`

### Slow Simulation
- Enable GPU: `use_gpu = true`
- Increase `num_threads`
- Use `lazy_frame_assembly = true` for large datasets

## Citation

If you use TIMSIM in your research, please cite:
```
[Citation pending]
```

## License

MIT License - see LICENSE file for details.
