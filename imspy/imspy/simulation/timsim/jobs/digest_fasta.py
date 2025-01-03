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

    if exclude_accumulated_gradient_start:
        assert 0 <= min_rt_percent <= 100, "min_rt_percent must be between 0 and 100"

        RTColumn = DeepChromatographyApex(
            model=load_deep_retention_time_predictor(),
            tokenizer=load_tokenizer_from_resources(tokenizer_name='tokenizer-ptm'),
            verbose=False
        )

        if verbose:
            print("Simulating retention times for exclusion of low retention times...")

        peptide_rt = RTColumn.simulate_separation_times_pandas(
            # deep copy the data to avoid modifying the original data
            data=peptides.peptides.copy(),
            gradient_length=gradient_length,
        )

        min_rt = gradient_length * min_rt_percent / 100
        rt_filter = peptide_rt['retention_time_gru_predictor'] > min_rt

        false_indices = np.where(peptide_rt['retention_time_gru_predictor'] <= min_rt)[0]

        # Define the bin size
        bin_size = 0.1
        bins = np.arange(0, peptide_rt['retention_time_gru_predictor'].max() + bin_size, bin_size)
        peptide_rt["rt_bins"] = pd.cut(peptide_rt['retention_time_gru_predictor'], bins)

        # Count rows in each bin
        binned_counts = peptide_rt["rt_bins"].value_counts().sort_index()

        # Get the median count
        median_rt_count = binned_counts.median()

        # Get the number of bins covered by the minimum retention time
        bins_covered = int(min_rt / bin_size)
        rt_count = bins_covered * median_rt_count

        if verbose:
            print(f"Minimum retention time: {min_rt}")
            print(f"Number of peptides to sample in min_rt range: {rt_count}")

        # Randomly select indices to set to true
        random_indices = np.random.choice(false_indices, size=int(rt_count), replace=False)

        # Set the selected indices to true
        rt_filter[random_indices] = True

        if verbose:
            print(f"Excluded {len(peptides.peptides) - len(peptides.peptides[rt_filter])} peptides with low retention times.")

        # Apply the filter
        peptides.peptides = peptides.peptides[rt_filter]

    return peptides