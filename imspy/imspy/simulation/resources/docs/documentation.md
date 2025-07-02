# TimSim: Proteomics Experiment Simulation on timsTOF

Welcome to the **`timsim` user manual**.  
This guide explains how to configure and run proteomics simulations on a virtual timsTOF platform.  
Each section details a parameter group, how it influences the simulation, and how to tune it to match your experiment.

---

## Table of Contents

1. [Main Settings](#main-settings)  
2. [Peptide Digestion Settings](#peptide-digestion-settings)  
3. [Peptide Intensity Settings](#peptide-intensity-settings)  
4. [Isotopic Pattern Settings](#isotopic-pattern-settings)  
5. [Signal Distribution Settings](#signal-distribution-settings)  
6. [Noise Settings](#noise-settings)  
7. [Property Variation Settings](#property-variation-settings)  
8. [DDA Settings](#dda-settings)  
9. [Charge State Probabilities](#charge-state-probabilities)  
10. [Performance Settings](#performance-settings)  
11. [Console and Execution](#console-and-execution)  

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

---

## Property Variation Settings

Feature-level Gaussian jitter applied *after* deterministic peak shaping and noise modeling. Will only be applied

| Parameter | Default | Description |
|-----------|---------|-------------|
| **RT Variation σ (s)** | `15` | Std-dev of retention-time apex jitter. |
| **Ion-Mobility Variation σ (1/K0)** | `0.008` | Std-dev of ion-mobility jitter. |
| **Intensity Variation σ (relative)** | `0.02` | Relative std-dev applied multiplicatively (≈ ±2 %). |

Set all three to `0` to disable feature jitter completely.

---

## DDA Settings

*Only active when **Acquisition Type** is set to **DDA**.*

| Parameter | Default | Description                                                                                                   |
|-----------|---------|---------------------------------------------------------------------------------------------------------------|
| **Precursors Every** | `7` | Distance between MS¹  frames in one cycle (7 = 1 precursor frame followed by 6 fragment frames).              |
| **Precursor Intensity Threshold** | `500` | Minimum MS¹ intensity for eligibility.                                                                        |
| **Max Precursors** | `7` | Hard cap on selected precursors per fragment frame.                                                           |
| **Exclusion Width** | `25` | Number of frames the re-acquisition of the same mz range that had been selected should not be selected again. |
| **Selection Mode** | `topN` | `topN` (by intensity) or `random`.                                                                            |

> *Features in precursor space are currently not detected from raw-data but taken from the simulation tables, and therefore feature detection is de-facto perfect. 
> For a more realistic procedure, we are planning to include a feature detection step in the future.*

---

## Charge State Probabilities

| Parameter | Description |
|-----------|-------------|
| **Binomial Charge Model** | Uses binomial distribution for charge states. |
| **Probability of Charge** | Success probability in binomial model. |
| **Minimum Charge Contribution** | Min relative contribution for a charge state. |
| **Maximum Charge** | Highest simulated charge state. |
| **Normalize Charge States** | Ensure kept charge states sum to 1. |

---

## Performance Settings

| Parameter | Description                                   |
|-----------|-----------------------------------------------|
| **Number of Threads** | Parallel threads (`-1` = all cores).          |
| **Batch Size** | Number of TimsFrames to be build in parallel. |

---

## Console and Execution (GUI Mode)

* **Console Output** – Real-time logs and progress.  
* **Run Button** – Starts the simulation with current settings.  
* **Cancel Button** – Terminates a running simulation.  
* **Save Config** – Store current settings as a TOML file.  
* **Load Config** – Restore settings from a saved TOML file.  

---

For help or troubleshooting, hover over the in-app tool-tips or visit our  
[GitHub repository](https://github.com/theGreatHerrLebert/rustims).
