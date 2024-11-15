import numpy as np
import pandas as pd

from .utility import check_path
from imspy.simulation.proteome import PeptideDigest

from imspy.algorithm.rt.predictors import DeepChromatographyApex, load_deep_retention_time_predictor
from imspy.algorithm.utility import load_tokenizer_from_resources


def digest_fasta(
        fasta_file_path: str,
        missed_cleavages: int = 2,
        min_len: int = 6,
        max_len: int = 30,
        cleave_at: str = 'KR',
        restrict: str = None,
        decoys: bool = False,
        verbose: bool = False,
        job_name: str = "digest_fasta",
        static_mods: dict[str, str] = {"C": "[UNIMOD:4]"},
        variable_mods: dict[str, list[str]] = {"M": ["[UNIMOD:35]"], "[": ["[UNIMOD:1]"]},
        exclude_accumulated_gradient_start: bool = True,
        min_rt_percent: float = 2.0,
        gradient_length: float = 60 * 60,
) -> PeptideDigest:
    """Digest a fasta file.

    Args:
        fasta_file_path: Path to the fasta file.
        missed_cleavages: Number of missed cleavages.
        min_len: Minimum peptide length.
        max_len: Maximum peptide length.
        cleave_at: Cleavage sites.
        restrict: Restrict to specific proteins.
        decoys: Generate decoys.
        verbose: Verbosity.
        job_name: Job name.
        static_mods: Static modifications.
        variable_mods: Variable modifications.
        exclude_accumulated_gradient_start: Exclude low retention times.
        min_rt_percent: Minimum retention time in percent.
        gradient_length: Gradient length in seconds (in seconds).

    Returns:
        PeptideDigest: Peptide digest object.
    """
    if verbose:
        print("Digesting peptides...")

    peptides = PeptideDigest(
        check_path(fasta_file_path),
        missed_cleavages=missed_cleavages,
        min_len=min_len,
        max_len=max_len,
        cleave_at=cleave_at,
        restrict=restrict,
        generate_decoys=decoys,
        verbose=verbose,
        variable_mods=variable_mods,
        static_mods=static_mods
    )

    # synthetic peptides tend to accumulate at the start of the gradient, exclude them
    if exclude_accumulated_gradient_start:

        assert 0 <= min_rt_percent <= 100, "min_rt_percent must be between 0 and 100"

        # create RTColumn instance
        RTColumn = DeepChromatographyApex(
            model=load_deep_retention_time_predictor(),
            tokenizer=load_tokenizer_from_resources(tokenizer_name='tokenizer-ptm'),
            verbose=False
        )

        if verbose:
            print("Simulating retention times for exclusion of low retention times...")

        # predict rts
        peptide_rt = RTColumn.simulate_separation_times_pandas(
            data=peptides.peptides,
            gradient_length=gradient_length,
        )

        # Define the bin size
        bin_size = 0.1

        # Create bins
        bins = np.arange(0, peptide_rt['retention_time_gru_predictor'].max() + bin_size, bin_size)
        peptide_rt["rt_bins"] = pd.cut(peptide_rt['retention_time_gru_predictor'], bins)

        # Count rows in each bin
        binned_counts = peptide_rt["rt_bins"].value_counts().sort_index()

        # Calculate the target count per bin (mean or median can be used)
        target_count = int(binned_counts.mean())  # or use binned_counts.median()

        # Smooth the distribution
        adjusted_rt_filter = peptide_rt.index.isin([])  # Start with an empty filter
        for bin_range, count in binned_counts.items():
            bin_indices = peptide_rt[peptide_rt["rt_bins"] == bin_range].index
            if count > target_count:
                # Randomly sample excess peptides from overpopulated bins
                sampled_indices = np.random.choice(bin_indices, size=target_count, replace=False)
            else:
                # Keep all peptides in underpopulated bins
                sampled_indices = bin_indices
            adjusted_rt_filter = adjusted_rt_filter | peptide_rt.index.isin(sampled_indices)

        # Apply the adjusted filter to peptides
        peptides.peptides = peptides.peptides[adjusted_rt_filter]

        if verbose:
            print(f"Smoothed distribution: Targeted ~{target_count} peptides per bin.")

    return peptides
