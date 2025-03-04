# TimSim: Proteomics Experiment Simulation on timsTOF

Welcome to the `timsim` user manual. This document provides guidance on how to configure and run 
proteomics simulations on a virtual timsTOF platform with `timsim`. Each section details a specific group of parameters, 
explaining how they influence the simulation and how you can adjust them to model your experiment accurately.

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
- **Save Path**: The directory where all simulation outputs, reports, and logs will be stored.
- **Reference Dataset Path**: The path to a real timsTOF dataset that serves as the template for the simulation’s layout.
- **FASTA File Path**: The location of the FASTA file containing protein sequences to be digested.
- **Experiment Name**: A unique identifier for your simulation. This name is used to label output files and reports.
- **Acquisition Type**: Choose the data acquisition method to simulate. Options include DIA, SYNCHRO, SLICE, and MIDIA.

### Options
- **Use Reference Layout**: Use the layout from the reference dataset to mimic realistic instrument parameters.
- **Load Reference into Memory**: Pre-load the reference dataset into RAM to speed up data access during simulation.
- **Sample Peptides**: When enabled, peptides are randomly sampled from the digestion process.
- **Generate Decoys**: Simulate decoy (inverted) peptides as well. Why would you want this? We still search an application!.
- **Silent Mode**: Run the simulation with minimal console output.
- **Apply Fragmentation to Ions**: Perform ion fragmentation during the simulation. This option is enabled by default.
- **Proteome Mixture**: Simulate complex samples by incorporating a mixture of proteomes.
- **Phospho Mode**: Enable phospho mode to generate a phospho-enriched dataset for testing phosphosite localization algorithms.

---

## Peptide Digestion Settings

### Parameters
- **Number of Sampled Peptides**: The total number of peptides generated from the in-silico digestion process.
- **Missed Cleavages**: The maximum number of allowed missed cleavage sites during digestion.
- **Minimum Peptide Length**: The shortest acceptable peptide length (in amino acids).
- **Maximum Peptide Length**: The longest acceptable peptide length (in amino acids).
- **Cleave At**: Specifies the amino acid residues at which the protein is cleaved (e.g., `"KR"` for trypsin).
- **Restrict**: Specifies residues that inhibit cleavage (e.g., `"P"` to prevent cleavage immediately after a cleavage site).
- **Amino Acid Modifications**: The path to a TOML file that defines fixed and variable modifications applied during digestion.

---

## Peptide Intensity Settings

### Parameters
Peptide intensity settings are currently simulated with a fixed procedure. If you want to vary the intensity of peptides, 
connect to a simulation database and adjust the intensity values there directly. Afterward you can re-run the simulation
using the adapted database as reference.

## Isotopic Pattern Settings

### Parameters
- **Maximum Number of Isotopes**: The number of isotopic peaks to simulate for each peptide.
- **Minimum Isotope Intensity**: The threshold intensity below which isotopic peaks are not included.
- **Centroid Isotopes**: When enabled, the simulation averages peak positions to generate a simplified isotopic pattern. This parameter is enabled by default.

---

## Signal Distribution Settings

### Parameters
- **Gradient Length**: The total duration of the simulated chromatographic gradient (in seconds).
- **Mean Std RT**: The average standard deviation of the retention time (RT) distribution, which affects peak widths.
- **Variance Std RT**: The variance in the RT standard deviation, influencing the variability of the collection of all peak widths.
- **Mean Skewness**: The average skewness of the RT distribution, which determines the asymmetry of peaks.
- **Variance Skewness**: The variance in the skewness, influencing the variability of the asymmetry of all peaks.
- **Z-Score**: Total amount of signal intensity in the chromatogram that needs to be covered before the numerical integration stops.
- **Target Percentile**: The percentile used to select high-density regions within the RT distribution.
- **Sampling Step Size**: The resolution for sampling the RT distribution; smaller values yield finer detail but require more computational resources.
---

## Noise Settings

### Parameters
- **Add Noise to Signals**: Enable this option to introduce random noise into signal intensities, mimicking experimental variability.
- **Add Precursor M/Z Noise**: Adds variability to precursor m/z values to simulate instrument measurement precision.
- **Precursor Noise PPM**: Specifies the noise level (in parts per million) for precursor m/z values.
- **Add Fragment M/Z Noise**: Adds noise to fragment m/z values to simulate spectral variation.
- **Fragment Noise PPM**: Specifies the noise level (in parts per million) for fragment m/z values.
- **Use Uniform Distribution for M/Z Noise**: When enabled, m/z noise is sampled from a uniform distribution instead of a Gaussian.
- **Add Real Data Noise**: Incorporate noise profiles derived from real experimental data.
- **Reference Noise Intensity Max**: The maximum intensity threshold for noise derived from reference data.
- **Fragment Downsample Factor**: Sets the relative amount fragment ions are downsampled. Take probability is inversely proportional to their intensity.

---

## Charge State Probabilities

### Parameters
- **Binomial Charge Model**: When enabled, the charge state distribution is modeled using a binomial distribution. Only if enabled, the following parameters are considered.
- **Probability of Charge**: The likelihood that a peptide will adopt a particular charge state.
- **Minimum Charge Contribution**: The minimum relative contribution required for peptides with a given charge state to be considered in the simulation.
- **Maximum Charge**: The highest charge state to simulate for peptides.

---

## Performance Settings

### Parameters
- **Number of Threads**: The number of parallel threads used for the simulation. Use `-1` to auto-detect all available cores.
- **Batch Size**: The number of data points processed in each computational batch.

---

## Console and Execution (GUI Mode)

### Features
- **Console Output**: Displays real-time logs, warnings, and progress updates during the simulation.
- **Run Button**: Initiates the simulation with the current configuration settings.
- **Cancel Button**: Provides the ability to terminate a running simulation.
- **Save Config**: Allows you to save your current settings to a TOML file for future use.
- **Load Config**: Load a previously saved TOML configuration file to quickly restore your simulation settings.

---

For additional assistance or troubleshooting, please refer to the tooltips within the GUI application or visit our [GitHub repository](https://github.com/theGreatHerrLebert/rustims) for support.