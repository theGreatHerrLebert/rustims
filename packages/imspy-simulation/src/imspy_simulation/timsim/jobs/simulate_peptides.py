import numpy as np
import pandas as pd
from typing import Optional
from collections import Counter
from imspy_core.data.peptide import PeptideSequence

from imspy_predictors.rt.predictors import DeepChromatographyApex, load_deep_retention_time_predictor
from imspy_predictors.ionization.predictors import predict_peptide_flyability_with_koina
from imspy_predictors.utility import load_tokenizer_from_resources

def sample_peptides_from_proteins(table, num_peptides_total=250_000, down_sample: bool = True):

    # Sample peptides from proteins, otherwise take all
    if down_sample:
        indices = np.concatenate([np.repeat(x, y) for x, y in zip(range(len(table)), table.num_peptides)])
        indices_chosen = np.random.choice(indices, size=num_peptides_total, replace=True)

    else:
        indices_chosen = np.concatenate([np.repeat(x, y) for x, y in zip(range(len(table)), table.num_peptides)])

    peptide_samples = []
    index_counts = Counter(indices_chosen)

    for index, count in index_counts.items():
        row = table.iloc[index]
        num_peptides = row["num_peptides"]

        # if sample peptide amount is larger than peptides, take all
        if count > num_peptides:
            peptide_samples.append((index, row.peptides))
        else:
            peptide_samples.append((index, np.random.choice(row.peptides, count, replace=False)))

    peptides_sorted = dict(sorted(peptide_samples, key=lambda x: x[0]))

    for i in range(0, len(table)):
        if i in peptides_sorted:
            continue
        else:
            peptides_sorted[i] = []

    tmp = sorted(list([(k, v) for k, v in peptides_sorted.items()]), key=lambda x: x[0])

    return [x[1] for x in tmp]


def generate_normal_efficiency(n, mean_log=-2, std_log=0.6, min_val=0.0001, max_val=1):
    efficiency = []
    while len(efficiency) < n:
        # Generate more samples than needed to reduce iterations
        samples = 10 ** np.random.normal(loc=mean_log, scale=std_log, size=n)

        # Filter only the values within the desired range
        valid_samples = samples[(samples >= min_val) & (samples <= max_val)]

        # Append valid samples to the list
        efficiency.extend(valid_samples)

    # Truncate to exactly `n` values
    return np.array(efficiency[:n])


def simulate_peptides(
        protein_table: pd.DataFrame,
        num_peptides_total: int = 250_000,
        verbose: bool = True,
        exclude_accumulated_gradient_start: bool = True,
        min_rt_percent: float = 2.0,
        gradient_length: float = 60 * 60,
        down_sample: bool = True,
        min_length: int = 7,
        max_length: int = 30,
        proteome_mix: bool = False,
        use_koina_model: Optional[str] = None,
) -> pd.DataFrame:
    """
    Simulate peptides from a protein table.
    Args:
        protein_table: DataFrame with protein sequences and events.
        num_peptides_total: Total number of peptides to simulate.
        verbose: Print progress.
        exclude_accumulated_gradient_start: Exclude peptides with low retention times.
        min_rt_percent: Minimum retention time percentage to keep.
        gradient_length: Length of the gradient in seconds.
        down_sample: Down sample the peptides from the proteins.
        min_length: Minimum length of the peptides.
        max_length: Maximum length of the peptides.
        proteome_mix: If True, simulate a proteome mix.
        use_koina_model: If not None, use the Koina model to predict peptide flyability, i.e. chances of being measured in the experiment, currently only supports pfly.
    Returns:
        DataFrame with simulated peptides.
    """
    protein_table["peptides_sampled"] = sample_peptides_from_proteins(protein_table, num_peptides_total, down_sample)
    protein_table = protein_table[[len(l) > 0 for l in protein_table.peptides_sampled]]

    peptide_id, names, events, sequences, decoys, missed_cleavages, n_term, c_term, masses, protein_id = [], [], [], [], [], [], [], [], [], []

    i = 0

    for (index, row) in protein_table.iterrows():
        for peptide in row.peptides_sampled:

            if len(peptide) < min_length or len(peptide) > max_length:
                continue

            masses.append(PeptideSequence(peptide).mono_isotopic_mass)
            missed_cleavages.append(0)
            decoys.append(0)
            n_term.append(None)
            c_term.append(None)
            names.append(row["protein"])
            events.append(row.events)
            sequences.append(peptide)
            peptide_id.append(i)
            protein_id.append(row.protein_id)
            i += 1

    peptide_table = pd.DataFrame({
        "protein_id": protein_id,
        "peptide_id": peptide_id,
        "sequence": sequences,
        "protein": names,
        "decoys": decoys,
        "missed_cleavages": missed_cleavages,
        "n_term": n_term,
        "c_term": c_term,
        "monoisotopic-mass": masses,
        "events": events}
    )

    # Simulate peptide efficiency using rejection sampling in log-normal space
    efficiency = generate_normal_efficiency(
        n=len(peptide_table),
        mean_log=-2,  # Center the log values around log10(0.01)
        std_log=1,  # Spread the values across 3 orders of magnitude
        min_val=0.0001,  # Minimum value
        max_val=1  # Maximum value
    )

    # Simulate peptide efficiency
    if use_koina_model is not None and use_koina_model != "":
        try:
            if verbose:
                print(f"Using Koina model: {use_koina_model}")

            # Simulate flyability using Koina model
            efficiency = predict_peptide_flyability_with_koina(
                model_name=use_koina_model,
                data=peptide_table,
            )

        except Exception as e:
            print(f"Failed to predict peptide flyability with KOINA: {e}")
            print("Falling back to normal distribution ...")

    peptide_table["events"] = (peptide_table.events * efficiency).astype(np.int32)

    if proteome_mix:
        peptide_table["total_events"] = peptide_table["events"]

    # Retention time filtering (optional)
    if exclude_accumulated_gradient_start:
        assert 0 <= min_rt_percent <= 100, "min_rt_percent must be between 0 and 100"

        if verbose:
            print("Simulating retention times for peptides and filtering ...")

        RTColumn = DeepChromatographyApex(
            model=load_deep_retention_time_predictor(),
            tokenizer=load_tokenizer_from_resources(tokenizer_name='tokenizer-ptm'),
            verbose=False
        )

        # rename column for compatibility
        peptide_table.rename(columns={"decoys": "decoy"}, inplace=True)

        peptide_rt = RTColumn.simulate_separation_times_pandas(
            data=peptide_table.copy(),
            gradient_length=gradient_length,
        )

        min_rt = gradient_length * min_rt_percent / 100
        rt_filter = peptide_rt['retention_time_gru_predictor'] > min_rt

        false_indices = np.where(peptide_rt['retention_time_gru_predictor'] <= min_rt)[0]

        bin_size = 0.1
        bins = np.arange(0, peptide_rt['retention_time_gru_predictor'].max() + bin_size, bin_size)
        peptide_rt["rt_bins"] = pd.cut(peptide_rt['retention_time_gru_predictor'], bins)

        binned_counts = peptide_rt["rt_bins"].value_counts().sort_index()
        median_rt_count = binned_counts.median()

        bins_covered = int(min_rt / bin_size)
        rt_count = bins_covered * median_rt_count

        random_indices = np.random.choice(false_indices, size=int(rt_count), replace=False)
        rt_filter[random_indices] = True

        peptide_table = peptide_table[rt_filter]

        if verbose:
            print(f"Excluded {len(peptide_table) - rt_filter.sum()} peptides with low retention times.")

    return peptide_table
