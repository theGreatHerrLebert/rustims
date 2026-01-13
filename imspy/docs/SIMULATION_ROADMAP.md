# IMSPY Simulation Package - Roadmap & Status

## Overview

The `imspy.simulation.timsim` package provides a comprehensive simulation framework for generating synthetic timsTOF mass spectrometry data. This document outlines the current capabilities, validation infrastructure, and planned improvements for establishing a robust integration testing pipeline.

---

## Goals

### Primary Objective
Create a **comprehensive integration test pipeline** that:
1. Simulates datasets across all supported acquisition modes and configurations
2. Analyzes simulated data with industry-standard tools (DiaNN, FragPipe)
3. Compares tool outputs against simulation ground truth
4. Generates standardized reports for quality assurance when simulator changes are made

### Why This Matters
- **Regression Detection**: Catch breaking changes in the simulator before release
- **Performance Benchmarking**: Track identification rates across simulator versions
- **Tool Compatibility**: Ensure simulated data works correctly with downstream analysis tools
- **Configuration Validation**: Verify all acquisition modes produce valid outputs

---

## Current State

### Simulation Capabilities

| Feature | Status | Notes |
|---------|--------|-------|
| DIA-PASEF | Complete | Multiple window schemes supported |
| DDA-PASEF | Complete | TopN and random selection modes |
| Thunder-PASEF | Complete | Optimized for immunopeptidomics |
| Retention time prediction | Complete | GRU model + Koina integration |
| Ion mobility prediction | Complete | GRU-based prediction |
| Fragment intensity | Complete | Prosit and PeptDeep models |
| Isotopic patterns | Complete | Configurable isotope count |
| Quadrupole transmission | Complete | Three modes: none, precursor_scaling, per_fragment |
| Noise modeling | Complete | m/z noise + real data noise injection |
| Phosphoproteomics | Complete | Variable modification support |
| Immunopeptidomics | Complete | HLA modifications config (no carbamidomethylation) |
| Multi-proteome mixing | Complete | FASTA mixing with dilution factors |
| Binomial charge model | Complete | Proper charge 1+ representation for short peptides |

### Validation Infrastructure

| Component | Status | Location |
|-----------|--------|----------|
| `timsim_validate` CLI | Complete | Single-tool validation (DiaNN or FragPipe) |
| `timsim_compare` CLI | Complete | Multi-tool comparison reports |
| `--analysis-tool both` | Complete | Run both tools in single workflow |
| DiaNN executor | Complete | Automated DiaNN analysis (DIA + DDA modes) |
| FragPipe executor | Complete | Automated FragPipe analysis (DIA + DDA modes) |
| Text reports | Complete | Human-readable validation summaries |
| JSON reports | Complete | Machine-parseable metrics with tool versions |
| HTML reports | Complete | Self-contained reports with embedded plots |
| Comparison plots | Complete | 9 plot types (Venn, correlations, breakdowns) |
| Per-tool plots | Complete | Summary grids, RT/IM correlations per tool |
| Tool version display | Complete | Versions shown in plots and reports |

### Metrics Captured

- **Identification metrics**: ID rate, precision, FDR, true/false positives
- **Correlation metrics**: RT and IM Pearson R vs ground truth
- **Intensity breakdown**: ID rates per intensity quintile
- **Charge state breakdown**: ID rates per charge state (2+, 3+, 4+)
- **Pairwise tool comparisons**: Jaccard index, overlap counts
- **Mass accuracy**: PPM error statistics

---

## Existing Benchmark Datasets

The following comprehensive datasets have been generated for benchmarking studies (complexity ramps, multiple replicates):

| Dataset ID | Acquisition | Sample Type | Complexity | Repl. | Source | Primary Use Case |
|------------|-------------|-------------|------------|-------|--------|------------------|
| DIA-H01 | dia-PASEF | HeLa | 5k–250k | 2 | In silico | ID + FDR |
| DIA-M01 | dia-PASEF | HYE | 10k–250k | 2 | In silico | Quantification |
| DIA-PH01 | dia-PASEF | HeLa-Phos | 5k–125k | 2 | In silico | PTM localization |
| DDA-H01 | dda-PASEF | HeLa | 150k–300k | 12 | In silico | MBR benchmark |
| DDA-HLA01 | thunder-PASEF | HLA1 | 10k, 100k | 3 | Experimental | ID + FDR |

### Dataset Locations

| Dataset | Path |
|---------|------|
| DIA HeLa (plain) | `/scratch/timsim/submission/dia/hela/plain/` |
| DIA HeLa (phospho) | `/scratch/timsim/submission/dia/hela/phospho/` |
| DIA HYE | `/scratch/timsim/submission/dia/hye/` |
| DDA HeLa | `/scratch/timsim/submission/dda/hela/` |
| DDA HLA | `/scratch/timsim/submission/dda/hla/` |

### Acquisition Modes Covered

| Mode | Description | Datasets |
|------|-------------|----------|
| dia-PASEF | Standard DIA with isolation windows | DIA-H01, DIA-M01, DIA-PH01 |
| dda-PASEF | Data-dependent with TopN selection | DDA-H01 |
| thunder-PASEF | Optimized for short immunopeptides | DDA-HLA01 |

---

## Integration Test Architecture

### Design Principles

The integration test system is split into **two independent phases**:

1. **Simulation Phase** (`timsim-integration-sim`): Generates synthetic datasets
2. **Evaluation Phase** (`timsim-integration-eval`): Runs analysis tools and validates results

This separation allows:
- Running simulation on GPU machines, evaluation on CPU machines
- Re-running evaluation with new tool versions without re-simulating
- Parallel execution of independent test cases
- Easier debugging and maintenance

### Directory Structure

```
imspy/simulation/timsim/integration/
├── __init__.py
├── configs/                    # Test-specific simulation configs
│   ├── IT-DIA-HELA.toml
│   ├── IT-DIA-HYE.toml
│   ├── IT-DIA-PHOS.toml
│   ├── IT-DDA-TOPN.toml
│   └── IT-DDA-HLA.toml
├── env.toml.template           # Machine-specific paths (copy and customize)
├── sim.py                      # Simulation runner script
└── eval.py                     # Evaluation runner script
```

### Environment Configuration

Each machine requires an `env.toml` file with local paths:

```toml
[paths]
# Base output directory for all integration tests
output_base = "/path/to/integration-tests"

# Reference .d files (blank acquisitions)
reference_dia = "/path/to/blanks/dia_blank.d"
reference_dda = "/path/to/blanks/dda_blank.d"
reference_thunder = "/path/to/blanks/thunder_blank.d"

# FASTA files
fasta_hela = "/path/to/fasta/human.fasta"
fasta_hye = "/path/to/fasta/hye_mixed.fasta"
fasta_hla = "/path/to/fasta/hla_ligands.fasta"

[tools]
# Analysis tool paths
diann_path = "/path/to/diann-2.3.1/diann-linux"
fragpipe_path = "/path/to/fragpipe-24.0/bin/fragpipe"
fragpipe_tools = "/path/to/fragpipe-24.0/tools"
fragpipe_workflow_dia = "/path/to/fragpipe-24.0/workflows/DIA_SpecLib_Quant_diaPASEF.workflow"
fragpipe_workflow_dda = "/path/to/fragpipe-24.0/workflows/LFQ-MBR.workflow"

[performance]
num_threads = -1
use_gpu = true
gpu_memory_limit_gb = 4
```

---

## Integration Test Matrix

### Final Test Cases

| Test ID | Acquisition | Sample | Complexity | Analysis Tools | Focus |
|---------|-------------|--------|------------|----------------|-------|
| IT-DIA-HELA | DIA-PASEF | HeLa | 25k | DiaNN, FragPipe | Standard ID benchmark |
| IT-DIA-HYE | DIA-PASEF | HYE | 25k | DiaNN, FragPipe | Quantification (species ratios) |
| IT-DIA-PHOS | DIA-PASEF | HeLa+Phospho | 25k | DiaNN, FragPipe | PTM localization |
| IT-DDA-TOPN | DDA-PASEF (TopN) | HeLa | 25k | DiaNN, FragPipe | DDA TopN selection |
| IT-DDA-HLA | DDA thunder | HLA | 10k | DiaNN, FragPipe | Immunopeptidomics |

### Dimension Coverage

| Dimension | Values Covered |
|-----------|----------------|
| Acquisition mode | DIA-PASEF, DDA-PASEF (TopN), DDA thunder |
| Sample type | Standard proteome, Mixed species, Phospho, HLA |
| Fragment model | Prosit (default) |
| Noise | Real data noise injection enabled |
| Analysis tools | DiaNN 2.3+, FragPipe 24+ |

### Pass/Fail Criteria

| Test ID | Min ID Rate | Min RT Corr | Min IM Corr | Additional |
|---------|-------------|-------------|-------------|------------|
| IT-DIA-HELA | 35% | 0.95 | 0.95 | - |
| IT-DIA-HYE | 30% | 0.95 | 0.95 | Species ratio error <20% |
| IT-DIA-PHOS | 25% | 0.95 | 0.95 | PTM site accuracy >80% |
| IT-DDA-TOPN | 30% | 0.95 | 0.95 | - |
| IT-DDA-HLA | 20% | 0.90 | 0.95 | 8-11mer coverage >50% |

---

## Workflow

### Phase 1: Simulation

```bash
# Run all simulations
timsim-integration-sim --env env.toml --all

# Run specific test
timsim-integration-sim --env env.toml --test IT-DIA-HELA

# Run subset
timsim-integration-sim --env env.toml --tests IT-DIA-HELA,IT-DIA-HYE
```

Output structure:
```
output_base/
├── IT-DIA-HELA/
│   ├── IT-DIA-HELA.d/           # Simulated .d folder
│   ├── synthetic_data.db         # Ground truth database
│   └── simulation.log
├── IT-DIA-HYE/
│   └── ...
└── ...
```

### Phase 2: Evaluation

```bash
# Evaluate all simulations
timsim-integration-eval --env env.toml --all

# Evaluate specific test with specific tool
timsim-integration-eval --env env.toml --test IT-DIA-HELA --tool diann

# Evaluate with both tools
timsim-integration-eval --env env.toml --test IT-DIA-HELA --tool both
```

Output structure:
```
output_base/
├── IT-DIA-HELA/
│   ├── IT-DIA-HELA/
│   │   ├── IT-DIA-HELA.d/        # Simulated .d folder
│   │   └── synthetic_data.db     # Ground truth database
│   ├── diann/
│   │   ├── report.parquet
│   │   └── diann.log
│   ├── fragpipe/
│   │   ├── psm.tsv
│   │   ├── peptide.tsv
│   │   └── ...
│   ├── validation/
│   │   ├── tool_comparison_report.txt
│   │   ├── tool_comparison_report.json
│   │   ├── validation_report.html    # Self-contained HTML report
│   │   ├── plots/                    # Comparison plots
│   │   │   ├── comparison_summary.png
│   │   │   └── ...
│   │   ├── DIA-NN_plots/             # Per-tool validation plots
│   │   │   ├── summary_grid.png
│   │   │   ├── rt_correlation.png
│   │   │   └── ...
│   │   └── FragPipe_plots/
│   │       └── ...
│   └── EVAL_PASS / EVAL_FAIL        # Status file
└── ...
```

### Phase 3: Reporting

```bash
# Generate summary report across all tests
timsim-integration-report --env env.toml --output report.html
```

---

## Success Criteria

### Per-Tool Thresholds
| Metric | DiaNN | FragPipe |
|--------|-------|----------|
| Identification Rate | >= 30% | >= 20% |
| RT Correlation | >= 0.95 | >= 0.90 |
| IM Correlation | >= 0.95 | >= 0.95 |
| Precision | >= 90% | >= 80% |

### Overall Pass Criteria
- All individual tool thresholds met
- No regression > 5% from baseline
- All acquisition modes produce valid outputs

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Validation infrastructure (`timsim_validate`, `timsim_compare`) | Complete |
| Phase 1 | Benchmark dataset generation | Complete |
| Phase 2 | Integration test configs | Complete |
| Phase 3 | Simulation script (`timsim-integration-sim`) | Complete |
| Phase 4 | Evaluation script (`timsim-integration-eval`) | Complete |
| Phase 5 | CI/CD integration | Planned |

### Completed Test Results (IT-DDA-HLA Example)

| Tool | Version | ID Rate | Precision | RT Corr | IM Corr | Status |
|------|---------|---------|-----------|---------|---------|--------|
| DIA-NN | 2.3.1 | 56.5% | 98.9% | 0.9999 | 1.0000 | PASS |
| FragPipe | 24.0 | 64.4% | 99.9% | 0.9999 | 1.0000 | PASS |

---

## References

- Simulator entry point: `imspy.simulation.timsim.simulator:main`
- Validation CLI: `imspy.simulation.timsim.validate.cli:main`
- Comparison CLI: `imspy.simulation.timsim.validate.tool_comparison:main_compare`
- HTML report generator: `imspy.simulation.timsim.validate.html_report`
- Example configs: `imspy/simulation/timsim/configs/`
- HLA modifications: `imspy/simulation/timsim/configs/modifications_hla.toml`
- Integration tests: `imspy/simulation/timsim/integration/`
- Integration sim: `imspy.simulation.timsim.integration.sim:main`
- Integration eval: `imspy.simulation.timsim.integration.eval:main`
