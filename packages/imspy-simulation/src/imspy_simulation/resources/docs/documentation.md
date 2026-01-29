# TimSim: Proteomics Experiment Simulation on timsTOF

Welcome to the **`timsim` user manual**.
This guide explains how to configure and run proteomics simulations on a virtual timsTOF platform.

---

## Quick Start

### Installation

```bash
# 1. Activate your Python environment
source /path/to/your/env/bin/activate

# 2. Install the Rust backend (requires maturin)
cd /path/to/rustims/imspy_connector
maturin develop --release

# 3. Install Python packages
pip install -e /path/to/rustims/packages/imspy-core
pip install -e /path/to/rustims/packages/imspy-simulation
pip install -e /path/to/rustims/packages/imspy-predictors
```

### Minimal Configuration

Create `config.toml`:

```toml
[paths]
save_path = "/path/to/output"
reference_path = "/path/to/blank_reference.d"
fasta_path = "/path/to/proteome.fasta"

[experiment]
experiment_name = "MyExperiment"
acquisition_type = "DIA"  # or "DDA"
gradient_length = 3600.0

[digestion]
num_sample_peptides = 50000
sample_seed = 42

[performance]
num_threads = -1
use_gpu = true
```

### Run Simulation

```bash
# Command line
python -m imspy_simulation.timsim.simulator config.toml

# Or use the GUI
python -m imspy_simulation.timsim.gui
```

### Output

```
output/
├── MyExperiment/
│   └── MyExperiment.d/    # Simulated timsTOF data
├── synthetic_data.db      # Ground truth database
└── MyExperiment_preview.mp4  # Optional preview video
```

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Main Settings](#main-settings)
3. [Peptide Digestion Settings](#peptide-digestion-settings)
4. [Peptide Intensity Settings](#peptide-intensity-settings)
5. [Isotopic Pattern Settings](#isotopic-pattern-settings)
6. [Signal Distribution Settings](#signal-distribution-settings)
7. [Noise Settings](#noise-settings)
8. [Property Variation Settings](#property-variation-settings)
9. [DDA Settings](#dda-settings)
10. [Charge State Probabilities](#charge-state-probabilities)
11. [Quad Transmission Settings](#quad-transmission-settings)
12. [Video Settings](#video-settings)
13. [Performance Settings](#performance-settings)
14. [Console and Execution](#console-and-execution)

---

## Main Settings

### Parameters
| Option | Description |
|--------|-------------|
| **Save Path** | Directory where all outputs, reports, and logs are stored. |
| **Reference Dataset Path** | Path to a real timsTOF dataset that provides the template layout. |
| **FASTA File Path** | Location of the FASTA file containing protein sequences. |
| **Experiment Name** | Unique identifier used in output file names. |
| **Acquisition Type** | Choose among **DIA**, **DDA**, **SYNCHRO**, **SLICE**, or **MIDIA**. |

### Options
* **Use Reference Layout** – Mimic realistic instrument parameters from the reference dataset.
* **Load Reference into Memory** – Pre-load dataset into RAM for faster access.
* **Sample Peptides** – Random sampling of peptides after digestion.
* **Generate Decoys** – Simulate inverted (decoy) peptides.
* **Silent Mode** – Minimal console output.
* **Apply Fragmentation to Ions** – Perform ion fragmentation (enabled by default).
* **Proteome Mixture** – Mix multiple proteomes for complex samples.
* **Phospho Mode** – Generate a phospho-enriched dataset.

### Configuration Example

```toml
[experiment]
experiment_name = "MyExperiment"
acquisition_type = "DIA"
gradient_length = 3600.0
use_reference_layout = true
apply_fragmentation = true
silent_mode = false
```

---

## Peptide Digestion Settings

| Parameter | Description |
|-----------|-------------|
| **Number of Sampled Peptides** | Total peptides drawn from the in-silico digest. |
| **Missed Cleavages** | Allowed missed cleavage sites. |
| **Minimum Peptide Length** | Shortest peptide length (aa). |
| **Maximum Peptide Length** | Longest peptide length (aa). |
| **Cleave At** | Cleavage residues (e.g. `KR`). |
| **Restrict** | Residues that block cleavage (e.g. `P`). |
| **Amino Acid Modifications** | Path to TOML file with fixed/variable mods. |
| **Sample Occurrences Randomly** | Toggle random sampling of peptide occurrences. |

### Configuration Example

```toml
[digestion]
n_proteins = 20000
num_peptides_total = 500000
num_sample_peptides = 150000
sample_peptides = true
sample_seed = 42
missed_cleavages = 2
min_len = 7
max_len = 30
cleave_at = "KR"
restrict = "P"
```

---

## Peptide Intensity Settings

*Currently simulated with a fixed procedure.*
For custom intensity profiles, edit the simulation database directly and rerun `timsim` with that database as reference.

---

## Isotopic Pattern Settings

| Parameter | Description |
|-----------|-------------|
| **Maximum Number of Isotopes** | How many isotopic peaks to simulate. |
| **Minimum Isotope Intensity** | Threshold below which isotopes are skipped. |
| **Centroid Isotopes** | Average peaks to simplified centroids (enabled by default). |

### Configuration Example

```toml
[isotopic_pattern]
isotope_k = 8
isotope_min_intensity = 1
isotope_centroid = true
```

---

## Signal Distribution Settings

| Parameter | Description |
|-----------|-------------|
| **Gradient Length** | Total chromatographic gradient length (s). |
| **Mean Std RT / Variance Std RT** | Mean and variance of RT peak widths. |
| **Mean Skewness / Variance Skewness** | Mean and variance of RT peak asymmetry. |
| **Z-Score** | Fraction of total intensity to cover before integration stops. |
| **Target Percentile** | Percentile that defines high-density RT regions. |
| **Sampling Step Size** | RT sampling resolution; smaller = finer detail, slower run. |

### Configuration Example

```toml
[retention_time]
min_rt_percent = 2.0
target_p = 0.999
sampling_step_size = 0.0001

[ion_mobility]
use_inverse_mobility_std_mean = true
inverse_mobility_std_mean = 0.009
```

---

## Noise Settings

| Parameter | Description |
|-----------|-------------|
| **Add Noise to Signals** | Inject random intensity noise. |
| **Add Precursor M/Z Noise** | Variability in precursor m/z values. |
| **Precursor Noise PPM** | Noise level for precursors (ppm). |
| **Add Fragment M/Z Noise** | Variability in fragment m/z values. |
| **Fragment Noise PPM** | Noise level for fragments (ppm). |
| **Use Uniform Distribution for M/Z Noise** | Uniform instead of Gaussian noise. |
| **Add Real Data Noise** | Use noise profiles from real datasets. |
| **Reference Noise Intensity Max** | Max intensity for reference-derived noise. |
| **Fragment Downsample Factor** | Probability to keep fragments inversely ∝ intensity. |
| **Add Noise to Frame / Scan Abundance** | Toggle extra abundance noise. |

### Configuration Example

```toml
[noise]
mz_noise_precursor = true
precursor_noise_ppm = 6.5
mz_noise_fragment = true
fragment_noise_ppm = 6.5
add_real_data_noise = true
reference_noise_intensity_max = 150000
precursor_sample_fraction = 0.2
fragment_sample_fraction = 0.2
```

---

## Property Variation Settings

Feature-level Gaussian jitter applied *after* deterministic peak shaping and noise modeling.

| Parameter | Default | Description |
|-----------|---------|-------------|
| **RT Variation σ (s)** | `15` | Std-dev of retention-time apex jitter. |
| **Ion-Mobility Variation σ (1/K0)** | `0.008` | Std-dev of ion-mobility jitter. |
| **Intensity Variation σ (relative)** | `0.02` | Relative std-dev applied multiplicatively (≈ ±2 %). |

Set all three to `0` to disable feature jitter completely.

---

## DDA Settings

*Only active when **Acquisition Type** is set to **DDA**.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Precursors Every** | `7` | Distance between MS¹ frames in one cycle. |
| **Precursor Intensity Threshold** | `500` | Minimum MS¹ intensity for eligibility. |
| **Max Precursors** | `7` | Hard cap on selected precursors per fragment frame. |
| **Exclusion Width** | `25` | Frames before same m/z can be re-selected. |
| **Selection Mode** | `topN` | `topN` (by intensity) or `random`. |

### Configuration Example

```toml
[dda]
precursors_every = 10
precursor_intensity_threshold = 500
max_precursors = 25
exclusion_width = 25
selection_mode = "topN"
```

---

## Charge State Probabilities

| Parameter | Description |
|-----------|-------------|
| **Binomial Charge Model** | Uses binomial distribution for charge states. |
| **Probability of Charge** | Success probability in binomial model. |
| **Minimum Charge Contribution** | Min relative contribution for a charge state. |
| **Maximum Charge** | Highest simulated charge state. |
| **Normalize Charge States** | Ensure kept charge states sum to 1. |

### Configuration Example

```toml
[charge_states]
p_charge = 0.5
max_charge = 4
min_charge_contrib = 0.25
binomial_charge_model = false
normalize_charge_states = true
charge_state_one_probability = 0.0
```

---

## Quad Transmission Settings

Advanced settings for quadrupole-dependent isotope transmission and **partial fragmentation** (precursor survival).

| Parameter | Default | Description |
|-----------|---------|-------------|
| **quad_isotope_transmission_mode** | `"none"` | `"none"`, `"precursor_scaling"`, or `"per_fragment"` |
| **quad_transmission_min_probability** | `0.5` | Min probability threshold for isotope transmission |
| **quad_transmission_max_isotopes** | `10` | Max isotope peaks to consider |
| **precursor_survival_min** | `0.0` | Min fraction of precursors remaining unfragmented (0.0-1.0) |
| **precursor_survival_max** | `0.0` | Max fraction of precursors remaining unfragmented (0.0-1.0) |

### Partial Fragmentation Example

Simulate 0-30% of precursors surviving fragmentation intact:

```toml
[quad_transmission]
quad_isotope_transmission_mode = "none"
precursor_survival_min = 0.0
precursor_survival_max = 0.3
```

---

## Video Settings

Generate preview videos for visual quality control.

| Parameter | Default | Description |
|-----------|---------|-------------|
| **generate_preview_video** | `false` | Enable video generation |
| **preview_video_max_frames** | `100` | Max frames to include in video |
| **preview_video_fps** | `10` | Frames per second |
| **preview_video_dpi** | `80` | Resolution (DPI) |
| **preview_video_annotate** | `true` | Add annotations to frames |

### Configuration Example

```toml
[video_settings]
generate_preview_video = true
preview_video_max_frames = 100
preview_video_fps = 10
preview_video_dpi = 80
preview_video_annotate = true
```

---

## Performance Settings

| Parameter | Description |
|-----------|-------------|
| **Number of Threads** | Parallel threads (`-1` = all cores). |
| **Batch Size** | Number of peptides to process in parallel. |
| **Frame Batch Size** | Number of frames to write in one batch. |
| **Use GPU** | Enable CUDA acceleration for ML models. |
| **GPU Memory Limit** | Max GPU memory in GB. |
| **Lazy Frame Assembly** | Build frames on-demand (lower memory, recommended for large datasets). |

### Configuration Example

```toml
[performance]
num_threads = -1
batch_size = 256
frame_batch_size = 500
use_gpu = true
gpu_memory_limit_gb = 4
lazy_frame_assembly = false
```

---

## Console and Execution (GUI Mode)

* **Console Output** – Real-time logs and progress.
* **Run Button** – Starts the simulation with current settings.
* **Cancel Button** – Terminates a running simulation.
* **Save Config** – Store current settings as a TOML file.
* **Load Config** – Restore settings from a saved TOML file.

---

## Complete Configuration Template

```toml
[paths]
save_path = "/path/to/output"
reference_path = "/path/to/reference.d"
fasta_path = "/path/to/proteome.fasta"

[experiment]
experiment_name = "MyExperiment"
acquisition_type = "DIA"
gradient_length = 3600.0
use_reference_layout = true
apply_fragmentation = true
silent_mode = false

[digestion]
n_proteins = 20000
num_peptides_total = 500000
num_sample_peptides = 150000
sample_peptides = true
sample_seed = 42
missed_cleavages = 2
min_len = 7
max_len = 30
cleave_at = "KR"
restrict = "P"

[retention_time]
min_rt_percent = 2.0
target_p = 0.999

[ion_mobility]
use_inverse_mobility_std_mean = true
inverse_mobility_std_mean = 0.009

[charge_states]
p_charge = 0.5
max_charge = 4
min_charge_contrib = 0.25

[isotopic_pattern]
isotope_k = 8
isotope_min_intensity = 1
isotope_centroid = true

[noise]
mz_noise_precursor = true
precursor_noise_ppm = 6.5
mz_noise_fragment = true
fragment_noise_ppm = 6.5
add_real_data_noise = true

[quad_transmission]
quad_isotope_transmission_mode = "none"
precursor_survival_min = 0.0
precursor_survival_max = 0.0

[video_settings]
generate_preview_video = true
preview_video_max_frames = 100
preview_video_fps = 10

[performance]
num_threads = -1
batch_size = 256
use_gpu = true
```

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "Bruker SDK not found" | Install `opentims-bruker-bridge`: `pip install opentims-bruker-bridge` |
| Out of memory | Enable `lazy_frame_assembly = true`, reduce `num_sample_peptides` |
| Slow simulation | Enable GPU (`use_gpu = true`), increase `num_threads` |
| No output .d file | Check `save_path` is writable, verify `reference_path` exists |

---

For help or troubleshooting, visit our [GitHub repository](https://github.com/theGreatHerrLebert/rustims).
