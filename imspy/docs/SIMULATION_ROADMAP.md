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
| Retention time prediction | Complete | GRU model + Koina integration |
| Ion mobility prediction | Complete | GRU-based prediction |
| Fragment intensity | Complete | Prosit and PeptDeep models |
| Isotopic patterns | Complete | Configurable isotope count |
| Quadrupole transmission | Complete | Three modes: none, precursor_scaling, per_fragment |
| Noise modeling | Complete | m/z noise + real data noise injection |
| Phosphoproteomics | Complete | Variable modification support |
| Multi-proteome mixing | Complete | FASTA mixing with dilution factors |

### Validation Infrastructure

| Component | Status | Location |
|-----------|--------|----------|
| `timsim_validate` CLI | Complete | Single-tool validation (DiaNN or FragPipe) |
| `timsim_compare` CLI | Complete | Multi-tool comparison reports |
| `--analysis-tool both` | Complete | Run both tools in single workflow |
| DiaNN executor | Complete | Automated DiaNN analysis |
| FragPipe executor | Complete | Automated FragPipe analysis |
| Text reports | Complete | Human-readable validation summaries |
| JSON reports | Complete | Machine-parseable metrics |
| Comparison plots | Complete | 7 plot types (Venn, correlations, breakdowns) |

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

### Dataset Details

- **DIA-H01**: Standard HeLa proteome at varying complexity levels for identification rate and FDR benchmarking
- **DIA-M01**: HYE (Human-Yeast-Ecoli) mixed species for quantification accuracy testing with known ratios
- **DIA-PH01**: Phosphoproteomics simulation for PTM site localization validation
- **DDA-H01**: High-complexity DDA with many replicates for match-between-runs (MBR) algorithm testing
- **DDA-HLA01**: Immunopeptidomics (HLA class I) with thunder-PASEF acquisition for short peptide identification

### Acquisition Modes Covered

| Mode | Description | Datasets |
|------|-------------|----------|
| dia-PASEF | Standard DIA with isolation windows | DIA-H01, DIA-M01, DIA-PH01 |
| dda-PASEF | Data-dependent with TopN selection | DDA-H01 |
| thunder-PASEF | Optimized for short immunopeptides | DDA-HLA01 |

---

## What Comes Next

### Phase 1: Validation Pipeline Execution

Run the validation pipeline on all existing datasets:

```
validation-outputs/
├── DIA-H01/
│   ├── 5k/
│   │   ├── rep1/
│   │   │   ├── diann/
│   │   │   ├── fragpipe/
│   │   │   └── comparison_report.json
│   │   └── rep2/
│   ├── 25k/
│   ├── 100k/
│   └── 250k/
├── DIA-M01/
│   └── ...
├── DIA-PH01/
│   └── ...
├── DDA-H01/
│   └── ...
└── DDA-HLA01/
    └── ...
```

### Phase 2: Automation & Reporting

#### Executor Script
A master script (`timsim-validate-grid`) that:
1. Iterates through all dataset directories
2. Locates simulation databases and analysis outputs
3. Runs `timsim_compare` for each complexity/replicate combination
4. Aggregates results into a summary report

#### Report Aggregation
- **Per-dataset reports**: Individual JSON/text reports per complexity level
- **Summary dashboard**: Single markdown/HTML overview across all datasets
- **Complexity curves**: ID rate vs. complexity plots per dataset
- **Cross-dataset comparison**: Compare DIA vs DDA, standard vs phospho, etc.

### Phase 3: CI/CD Integration

```yaml
# Example GitHub Actions workflow
validate-simulation-datasets:
  runs-on: self-hosted
  steps:
    - name: Validate DIA-H01 (quick)
      run: |
        timsim_compare --database DIA-H01/5k/rep1/synthetic_data.db \
                       --diann DIA-H01/5k/rep1/diann/report.parquet \
                       --output validation/DIA-H01/5k/rep1

    - name: Check thresholds
      run: |
        python check_validation_thresholds.py --min-id-rate 0.30
```

---

## Integration Test Matrix

For the integration test pipeline, we create **one dataset per configuration** (no complexity ramps):

| Test ID | Acquisition | Sample | Complexity | Analysis Tools | Focus Metrics |
|---------|-------------|--------|------------|----------------|---------------|
| IT-DIA-HELA | dia-PASEF | HeLa | 25k | DiaNN, FragPipe | ID rate, RT/IM correlation |
| IT-DIA-HYE | dia-PASEF | HYE | 25k | DiaNN, FragPipe | Quant correlation, species ratios |
| IT-DIA-PHOS | dia-PASEF | HeLa-Phos | 25k | DiaNN, FragPipe | PTM localization |
| IT-DDA-HELA | dda-PASEF | HeLa | 25k | DiaNN, FragPipe | ID rate, precursor selection |
| IT-DDA-HLA | thunder-PASEF | HLA1 | 10k | DiaNN | Short peptide ID rate |

### Pass/Fail Criteria per Test

| Test ID | Min ID Rate | Min RT Corr | Min IM Corr | Additional |
|---------|-------------|-------------|-------------|------------|
| IT-DIA-HELA | 35% | 0.95 | 0.95 | - |
| IT-DIA-HYE | 30% | 0.95 | 0.95 | Species ratio error <20% |
| IT-DIA-PHOS | 25% | 0.95 | 0.95 | PTM site accuracy >80% |
| IT-DDA-HELA | 30% | 0.95 | 0.95 | - |
| IT-DDA-HLA | 20% | 0.90 | 0.95 | 8-11mer coverage >50% |

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

## Open Questions

1. **Dataset locations**: Where are the simulation outputs stored? (paths needed for automation)
2. **Tool versions**: Which DiaNN/FragPipe versions were used for analysis?
3. **Quant metrics**: For DIA-M01, what species ratios should we validate?
4. **PTM metrics**: For DIA-PH01, how do we score localization accuracy?
5. **MBR metrics**: For DDA-H01, what defines successful MBR transfer?

---

## Timeline

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Validation infrastructure (`timsim_validate`, `timsim_compare`) | Complete |
| Phase 1 | Dataset generation (DIA-H01, DIA-M01, DIA-PH01, DDA-H01, DDA-HLA01) | Complete |
| Phase 2 | Validation pipeline execution on all datasets | Next |
| Phase 3 | Report aggregation & summary dashboard | Planned |
| Phase 4 | CI/CD integration | Planned |

---

## References

- Simulator entry point: `imspy.simulation.timsim.simulator:main`
- Validation CLI: `imspy.simulation.timsim.validate.cli:main`
- Comparison CLI: `imspy.simulation.timsim.validate.tool_comparison:main_compare`
- Example configs: `imspy/simulation/timsim/configs/`
