# TIMSIM Integration Test & Validation Pipeline

Automated validation framework for TIMSIM simulations. Generate synthetic datasets, analyze with production proteomics tools, and validate against ground truth.

## Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Simulate   │ ──▶ │   Analyze   │ ──▶ │  Validate   │ ──▶ │   Report    │
│  (timsim)   │     │ (DiaNN/FP)  │     │ (vs truth)  │     │  (HTML/JSON)│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
# Create/activate Python environment
source /path/to/your/env/bin/activate

# Install required packages
pip install -e /path/to/rustims/packages/imspy-simulation
pip install -e /path/to/rustims/packages/imspy-core

# Rebuild Rust backend if needed
cd /path/to/rustims/imspy_connector
maturin develop --release
```

### 2. Configure Environment File

Create `env.toml` in your working directory:

```toml
# Analysis tools
[tools]
diann_path = "/path/to/diann-linux"
fragpipe_path = "/path/to/fragpipe/bin/fragpipe"
fragpipe_tools = "/path/to/fragpipe/tools"
fragpipe_workflow_dia = "/path/to/workflows/DIA_SpecLib_Quant_diaPASEF.workflow"
fragpipe_workflow_dda = "/path/to/workflows/LFQ-noMBR.workflow"
sage_path = "/path/to/sage"

# Output and reference data
[paths]
output_base = "/path/to/output"
reference_dia = "/path/to/blank_dia.d"
reference_dda = "/path/to/blank_dda.d"
fasta_hela = "/path/to/hela.fasta"
fasta_hela_decoys = "/path/to/hela-decoys.fasta"

# Performance settings
[performance]
num_threads = -1
use_gpu = true
```

### 3. Run Tests

```bash
# List available tests
python -m imspy_simulation.timsim.integration.sim --env env.toml --list

# Run a single simulation
python -m imspy_simulation.timsim.integration.sim --env env.toml --test IT-DIA-HELA

# Run evaluation (analysis + validation)
python -m imspy_simulation.timsim.integration.eval --env env.toml --test IT-DIA-HELA

# Run all tests
python -m imspy_simulation.timsim.integration.sim --env env.toml --all
python -m imspy_simulation.timsim.integration.eval --env env.toml --all
```

## Available Integration Tests

| Test ID | Mode | Sample | Description |
|---------|------|--------|-------------|
| `IT-DIA-HELA` | DIA | HeLa | Standard proteomics (150K peptides) |
| `IT-DIA-HYE` | DIA | HYE | Multi-species quantification |
| `IT-DIA-HYE-A/B` | DIA | HYE | Fold-change benchmark (paired) |
| `IT-DIA-PHOS` | DIA | HeLa | Phosphoproteomics |
| `IT-DIA-PHOS-A/B` | DIA | HeLa | PTM site localization benchmark |
| `IT-DIA-PARTIAL-FRAG` | DIA | HeLa | Partial fragmentation (30% unfrag) |
| `IT-DDA-TOPN` | DDA | HeLa | TopN DDA (250K peptides) |
| `IT-DDA-HLA` | DDA | HeLa | Immunopeptidomics |
| `IT-DDA-PARTIAL-FRAG` | DDA | HeLa | DDA with partial fragmentation |

## Pipeline Details

### Phase 1: Simulation (`sim.py`)

Generates synthetic timsTOF datasets:

```bash
python -m imspy_simulation.timsim.integration.sim --env env.toml --test IT-DIA-HELA
```

**Output:**
```
output/IT-DIA-HELA/
├── SIM_SUCCESS                    # Status marker
├── IT-DIA-HELA_config.toml        # Resolved configuration
├── IT-DIA-HELA/
│   └── IT-DIA-HELA.d/             # Simulated timsTOF data
├── synthetic_data.db              # Ground truth database
└── IT-DIA-HELA_preview.mp4        # Preview video (if enabled)
```

### Phase 2: Evaluation (`eval.py`)

Runs analysis tools and validates results:

```bash
python -m imspy_simulation.timsim.integration.eval --env env.toml --test IT-DIA-HELA
```

**Steps:**
1. Run DiaNN analysis
2. Run FragPipe analysis
3. Run Sage analysis (DDA only)
4. Extract identifications from each tool
5. Match against ground truth
6. Calculate metrics (ID rate, RT/IM correlation)
7. Check against pass/fail thresholds
8. Generate reports

**Output:**
```
output/IT-DIA-HELA/
├── EVAL_PASS                      # Status marker (or EVAL_FAIL)
├── diann/
│   ├── report.parquet             # DiaNN results
│   └── report.log.txt
├── fragpipe/
│   ├── psm.tsv
│   └── peptide.tsv
├── sage/                          # DDA only
│   └── results.sage.tsv
└── validation/
    ├── validation_metrics.json    # Detailed metrics
    ├── validation_report.html     # Visual report
    └── plots/                     # Comparison plots
```

## Validation Metrics

### Core Metrics (All Tests)

| Metric | Description | Typical Threshold |
|--------|-------------|-------------------|
| **ID Rate** | Ground truth peptides identified | ≥ 25-30% |
| **RT Correlation** | Expected vs observed retention time | ≥ 0.95 |
| **IM Correlation** | Expected vs observed ion mobility | ≥ 0.95 |

### HYE-Specific Metrics

| Metric | Description | Typical Threshold |
|--------|-------------|-------------------|
| **Species Ratio Error** | Deviation from expected H:Y:E ratios | ≤ 20% |
| **Fold Change Error** | Deviation from expected fold changes | ≤ 30% |

### Phospho-Specific Metrics

| Metric | Description | Typical Threshold |
|--------|-------------|-------------------|
| **PTM Site Accuracy** | Correctly localized phosphosites | ≥ 80% |

## Test Configuration

Test configs are in `configs/` directory. Each `.toml` file defines:

```toml
[test_metadata]
test_id = "IT-DIA-HELA"
description = "Standard HeLa DIA-PASEF benchmark"
acquisition_type = "DIA"
sample_type = "hela"
analysis_tools = ["diann", "fragpipe"]

[thresholds]
min_id_rate = 0.28
min_rt_correlation = 0.95
min_im_correlation = 0.95

[paths]
save_path = "${output_base}/IT-DIA-HELA"
reference_path = "${reference_dia}"
fasta_path = "${fasta_hela}"

# ... simulation parameters ...
```

### Path Placeholders

Use `${variable}` syntax for machine-specific paths:
- `${output_base}` → from `env.toml [paths] output_base`
- `${reference_dia}` → from `env.toml [paths] reference_dia`
- `${fasta_hela}` → from `env.toml [paths] fasta_hela`

## Adding a New Test

### 1. Create Test Config

Create `configs/IT-NEW-TEST.toml`:

```toml
[test_metadata]
test_id = "IT-NEW-TEST"
description = "My new test"
acquisition_type = "DIA"  # or "DDA"
sample_type = "hela"
analysis_tools = ["diann", "fragpipe"]

[thresholds]
min_id_rate = 0.25
min_rt_correlation = 0.95
min_im_correlation = 0.95

[paths]
save_path = "${output_base}/IT-NEW-TEST"
reference_path = "${reference_dia}"
fasta_path = "${fasta_hela}"

[experiment]
experiment_name = "IT-NEW-TEST"
acquisition_type = "DIA"
gradient_length = 3600.0

# ... add other sections as needed ...
```

### 2. Register the Test

Edit `sim.py` and `eval.py`, add to `AVAILABLE_TESTS`:

```python
AVAILABLE_TESTS = [
    "IT-DIA-HELA",
    # ...
    "IT-NEW-TEST",  # Add here
]
```

### 3. Run the Test

```bash
python -m imspy_simulation.timsim.integration.sim --env env.toml --test IT-NEW-TEST
python -m imspy_simulation.timsim.integration.eval --env env.toml --test IT-NEW-TEST
```

## Command Reference

### Simulation Commands

```bash
# List tests
python -m imspy_simulation.timsim.integration.sim --env env.toml --list

# Run single test
python -m imspy_simulation.timsim.integration.sim --env env.toml --test IT-DIA-HELA

# Run multiple tests
python -m imspy_simulation.timsim.integration.sim --env env.toml --tests IT-DIA-HELA,IT-DDA-TOPN

# Run all tests
python -m imspy_simulation.timsim.integration.sim --env env.toml --all
```

### Evaluation Commands

```bash
# Run with all tools
python -m imspy_simulation.timsim.integration.eval --env env.toml --test IT-DIA-HELA

# Run with specific tool
python -m imspy_simulation.timsim.integration.eval --env env.toml --test IT-DIA-HELA --tool diann

# Skip analysis (validate existing results)
python -m imspy_simulation.timsim.integration.eval --env env.toml --test IT-DIA-HELA --skip-analysis

# Run all evaluations
python -m imspy_simulation.timsim.integration.eval --env env.toml --all
```

## Output Summary

After running all tests, check:

```
output/
├── evaluation_summary.json   # Machine-readable results
├── index.html                # Dashboard with pass/fail status
└── full_report.log           # Complete execution log
```

### Pass/Fail Markers

Each test directory contains status markers:
- `SIM_SUCCESS` / `SIM_FAILED` - Simulation status
- `EVAL_PASS` / `EVAL_FAIL` - Evaluation status

## Troubleshooting

### DiaNN Fails

```bash
# Check DiaNN is executable
chmod +x /path/to/diann-linux

# Verify FASTA has decoys (or let DiaNN generate them)
# Check diann/report.log.txt for errors
```

### FragPipe Fails

```bash
# Ensure Java is available
java -version

# Check workflow file path in env.toml
# Verify fragpipe_tools path is correct
```

### Low ID Rates

- Check FASTA file matches sample type
- Verify reference .d file has correct acquisition mode
- Review simulation parameters (noise, gradient length)
- Consider lowering threshold for initial testing

### Simulation Crashes

- Check available disk space
- Reduce `num_sample_peptides`
- Enable `lazy_frame_assembly = true`
- Check GPU memory if using CUDA

## File Locations

```
imspy_simulation/timsim/integration/
├── sim.py              # Simulation runner
├── eval.py             # Evaluation runner
├── configs/            # Test configurations
│   ├── IT-DIA-HELA.toml
│   ├── IT-DDA-TOPN.toml
│   └── ...
└── VALIDATION_README.md  # This file
```

## Requirements

### Analysis Tools

| Tool | Version | Purpose |
|------|---------|---------|
| DiaNN | 1.8+ | DIA/DDA analysis |
| FragPipe | 21+ | DIA/DDA analysis |
| Sage | 0.14+ | DDA analysis (optional) |

### Python Packages

```
imspy-simulation
imspy-core
imspy-connector (Rust backend)
pandas
numpy
toml
```

### Hardware

- **CPU**: 8+ cores recommended
- **RAM**: 32GB+ for large datasets
- **GPU**: NVIDIA CUDA-capable (optional, speeds up ML models)
- **Disk**: 50GB+ free space per test

## License

MIT License
