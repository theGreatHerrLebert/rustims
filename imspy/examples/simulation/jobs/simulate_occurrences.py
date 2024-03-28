import pandas as pd
import numpy as np

from imspy.simulation.utility import generate_events


def simulate_peptide_occurrences(
        peptides: pd.DataFrame,
        intensity_mean: float,
        intensity_min: float,
        intensity_max: float,
        verbose: bool = False,
        sample_occurrences: bool = True,
        intensity_value: float = 1e6,
) -> pd.DataFrame:

    if verbose:
        print("Simulating peptide occurrences...")

    if sample_occurrences:
        peptides['events'] = generate_events(
            n=peptides.shape[0],
            mean=intensity_mean,
            min_val=intensity_min,
            max_val=intensity_max
        )
    else:
        peptides['events'] = intensity_value
    return peptides
