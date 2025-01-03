# TimSim: Proteomics Experiment Simulation on timsTOF

Welcome to the TimSim user manual. This document provides detailed instructions on using the GUI to configure and run simulations. Each section corresponds to a group of parameters available in the GUI, with explanations for how these influence the simulation.

---

## Table of Contents

1. [Main Settings](#main-settings)
2. [Peptide Digestion Settings](#peptide-digestion-settings)
3. [Peptide Intensity Settings](#peptide-intensity-settings)
4. [Isotopic Pattern Settings](#isotopic-pattern-settings)
5. [Signal Distribution Settings](#signal-distribution-settings)
6. [Noise Settings](#noise-settings)
7. [Charge State Probabilities](#charge-state-probabilities)
8. [Performance Settings](#performance-settings)
9. [Console and Execution](#console-and-execution)

---

## Main Settings

### Parameters
- **Save Path**: Directory to save the simulation outputs.
- **Reference Dataset Path**: Path to the reference dataset used as a simulation template.
- **FASTA File Path**: Path to the FASTA file containing protein sequences for digestion.
- **Experiment Name**: A unique name for your experiment.
- **Acquisition Type**: Select the method for data acquisition (DIA, SYNCHRO, SLICE, MIDIA).

### Options
- **Use Reference Layout**: Use the layout of the reference dataset as a template.
- **Load Reference into Memory**: Speeds up access by loading the reference dataset into memory.
- **Sample Peptides**: Enable to randomly sample peptides for a subset of the dataset.
- **Generate Decoys**: Simulate inverted decoy peptides (non-biological).
- **Silent Mode**: Suppress output messages during the simulation.
- **Apply Fragmentation to Ions**: Disable to skip ion fragmentation but retain quadrupole selection.
- **Proteome Mixture**: Use a mixture of proteomes for more complex samples.

---

## Peptide Digestion Settings

### Parameters
- **Number of Sampled Peptides**: Total number of peptides to simulate.
- **Missed Cleavages**: Maximum number of missed cleavages allowed during digestion.
- **Minimum Peptide Length**: Minimum length of peptides in amino acids.
- **Maximum Peptide Length**: Maximum length of peptides in amino acids.
- **Cleave At**: Residues at which cleavage occurs (e.g., "KR" for trypsin).
- **Restrict**: Residues where cleavage is restricted (e.g., "P" to prevent cleavage after proline).
- **Amino Acid Modifications**: Path to a TOML file defining fixed and variable modifications.

---

## Peptide Intensity Settings

### Parameters
- **Mean Intensity**: Average intensity for simulated peptides (power of 10).
- **Minimum Intensity**: Lowest possible intensity for peptides (power of 10).
- **Maximum Intensity**: Highest possible intensity for peptides (power of 10).
- **Sample Occurrences Randomly**: Random sampling of peptide occurrences.
- **Fixed Intensity Value**: Set a fixed intensity for peptides when random sampling is disabled.

---

## Isotopic Pattern Settings

### Parameters
- **Maximum Number of Isotopes**: Specify the number of isotopes to simulate for each peptide.
- **Minimum Isotope Intensity**: Threshold intensity for including isotopes in the simulation.
- **Centroid Isotopes**: Enable to average peak positions for simplified patterns.

---

## Signal Distribution Settings

### Parameters
- **Gradient Length**: Total length of the simulated chromatographic gradient (seconds).
- **Mean Std RT**: Mean standard deviation for retention time.
- **Variance Std RT**: Variance of the standard deviation of retention time.
- **Mean Skewness**: Average skewness for retention time distribution.
- **Variance Skewness**: Variance of skewness in retention time distribution.
- **Standard Deviation IM**: Standard deviation of ion mobility.
- **Variance Std IM**: Variance of standard deviation for ion mobility.
- **Z-Score**: Threshold for filtering data by signal intensity.
- **Target Percentile**: Target percentile for retention time density.
- **Sampling Step Size**: Step size for data sampling, impacting resolution.

---

## Noise Settings

### Parameters
- **Add Noise to Signals**: Enable random noise for signal intensities.
- **Add Precursor M/Z Noise**: Add noise to precursor m/z values.
- **Precursor Noise PPM**: Noise level (ppm) for precursor m/z values.
- **Add Fragment M/Z Noise**: Add noise to fragment m/z values.
- **Fragment Noise PPM**: Noise level (ppm) for fragment m/z values.
- **Use Uniform Distribution for M/Z Noise**: Use a uniform distribution for m/z noise instead of Gaussian.
- **Add Real Data Noise**: Incorporate noise derived from real datasets.
- **Reference Noise Intensity Max**: Maximum intensity for reference data noise.
- **Fragment Downsample Factor**: Downsampling probability for fragment ions.

---

## Charge State Probabilities

### Parameters
- **Probability of Charge**: Likelihood of peptides adopting a specific charge state.
- **Minimum Charge Contribution**: Minimum contribution of peptides with the specified charge state.

---

## Performance Settings

### Parameters
- **Number of Threads**: Number of threads for parallel processing (-1 for auto-detect).
- **Batch Size**: Number of data points processed per batch.

---

## Console and Execution

### Features
- **Console Output**: Displays logs, warnings, and progress messages.
- **Run Button**: Starts the simulation with the configured settings.
- **Cancel Button**: Cancels a running simulation.
- **Save Config**: Save your settings as a TOML file for reuse.
- **Load Config**: Load a previously saved TOML configuration file.

---

For further assistance or troubleshooting, consult the application's built-in tooltips or 
[contact us via GitHub](https://github.com/theGreatHerrLebert/rustims).
