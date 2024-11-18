import numpy as np
import pandas as pd

from collections import Counter
from imspy.data.peptide import PeptideSequence

from imspy.algorithm.rt.predictors import DeepChromatographyApex, load_deep_retention_time_predictor
from imspy.algorithm.utility import load_tokenizer_from_resources

def sample_peptides_from_proteins(table, num_peptides_total=100_000):
    indices = np.concatenate([np.repeat(x, y) for x, y in zip(range(len(table)), table.num_peptides)])
    indices_chosen = np.random.choice(indices, size=num_peptides_total, replace=True)

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


def simulate_peptides(
        protein_table: pd.DataFrame,
        num_peptides_total: int = 100_000,
        verbose: bool = True,
        exclude_accumulated_gradient_start: bool = True,
        min_rt_percent: float = 2.0,
        gradient_length: float = 60 * 60,
) -> pd.DataFrame:
    protein_table["peptides_sampled"] = sample_peptides_from_proteins(protein_table, num_peptides_total)
    protein_table = protein_table[[len(l) > 0 for l in protein_table.peptides_sampled]]

    peptide_id, names, events, sequences, decoys, missed_cleavages, n_term, c_term, masses, protein_id = [], [], [], [], [], [], [], [], [], []

    i = 0

    for (index, row) in protein_table.iterrows():
        for peptide in row.peptides_sampled:
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

    # simulate peptide efficiency by sampling from a normal distribution
    efficiency = np.random.normal(loc=0, scale=1, size=len(peptide_table))

    # normalize all values to be between 0 and 1
    efficiency = (efficiency - np.min(efficiency)) / (np.max(efficiency) - np.min(efficiency))

    peptide_table["events"] = (peptide_table.events * efficiency).astype(np.int32)

    # Retention time filtering (optional)
    if exclude_accumulated_gradient_start:
        assert 0 <= min_rt_percent <= 100, "min_rt_percent must be between 0 and 100"

        if verbose:
            print("Simulating retention times for peptides and filtering...")

        RTColumn = DeepChromatographyApex(
            model=load_deep_retention_time_predictor(),
            tokenizer=load_tokenizer_from_resources(tokenizer_name='tokenizer-ptm'),
            verbose=False
        )

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