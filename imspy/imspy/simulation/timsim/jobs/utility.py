import os
import argparse
from typing import Tuple, List


# check if the path exists
def check_path(p: str) -> str:
    if not os.path.exists(p):
        raise argparse.ArgumentTypeError(f"Invalid path: {p}")
    return p


def phosphorylation_sizes(sequence: str) -> Tuple[int, List[int]]:
    """
    Checks if a sequence contains potential phosphorylation sites (S, T, or Y),
    and returns the count of sites and their indices.

    Args:
        sequence (str): The input sequence string, e.g., "IC[UNIMOD:4]RQHTK".

    Returns:
        tuple: A tuple containing:
            - int: The number of phosphorylation sites.
            - list: A list of indices where the sites are found.
    """
    # Define the phosphorylation sites
    phosphorylation_sites = {'S', 'T', 'Y'}

    # List to store the indices of phosphorylation sites
    indices = []

    # Iterate over the sequence and collect indices of phosphorylation sites
    for index, char in enumerate(sequence):
        if char in phosphorylation_sites:
            indices.append(index)

    # Return the count of sites and the list of indices
    return len(indices), indices

import numpy as np

def add_normal_noise_softclip(values: np.ndarray,
                              variation_std: float = 10.0,
                              softclip_tau: float = 5.0) -> np.ndarray:
    """
    Add Gaussian noise to values and apply a soft-pull toward bounds [0, max_val].
    Args:
        values: np.ndarray
            Array of values to which noise will be added.
        variation_std: float
            Standard deviation of the Gaussian noise to be added.
        softclip_tau: float
            Controls the steepness of the soft-pull function.
            Higher values make the pull more gradual.

    Returns:
        np.ndarray
            Array of values with added noise and soft-pull applied.
    """
    values = np.asarray(values)
    max_val = np.max(values)

    # Add Gaussian noise
    noisy = values + np.random.normal(0, variation_std, size=values.shape)

    # Soft-pull toward bounds
    # Define a smooth function that asymptotically approaches [0, max_val]
    # This uses a scaled logistic function for mapping out-of-bounds values
    def soft_pull(x):
        scaled = (x - 0) / (max_val - 0)  # normalize to [0,1] space
        pulled = 1 / (1 + np.exp(-softclip_tau * (scaled - 0.5)))  # sigmoid centered at 0.5
        return pulled * max_val

    return soft_pull(noisy)

import numpy as np

def add_log_normal_noise(
    intensities: np.ndarray,
    log_noise_std: float = 0.02,
) -> np.ndarray:
    """
    Add log-normal noise to a set of intensities.
    Args:
        intensities: np.ndarray
            Array of intensities to which noise will be added.
        log_noise_std: float
            Standard deviation of the log-normal noise to be added.
    Returns:
        np.ndarray
            Array of intensities with added log-normal noise.
    """
    intensities = np.asarray(intensities)
    assert np.all(intensities >= 0), "All intensities must be ≥ 0"

    # Log-transform safely
    log_vals = np.log1p(intensities)

    # Add noise
    noisy_log_vals = log_vals + np.random.normal(0, log_noise_std, size=log_vals.shape)

    # Transform back and ensure non-negativity
    noised_intensities = np.expm1(noisy_log_vals)
    return np.clip(noised_intensities, 0, None)
