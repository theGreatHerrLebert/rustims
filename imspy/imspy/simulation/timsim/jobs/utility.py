import numpy as np
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


def add_normal_noise(
        values: np.ndarray,
        variation_std: float = 10.0
) -> np.ndarray:
    """
    Add Gaussian noise to the input values and hard clip to [0, max(values)].
    Args:
        values: np.ndarray of non-negative values
        variation_std: standard deviation of the noise to be added

    Returns:
        np.ndarray of noise-added and clipped values
    """
    values = np.asarray(values)
    noisy = values + np.random.normal(0, variation_std, size=values.shape)
    return np.clip(noisy, 0, np.max(values))


def add_log_normal_noise(
        intensities: np.ndarray,
        log_noise_std: float = 0.2
) -> np.ndarray:
    """
    Add log-normal noise to the input intensities and hard clip to [0, max(intensities)].
    Args:
        intensities: np.ndarray of non-negative intensities
        log_noise_std: standard deviation of the log-normal noise to be added

    Returns:
        np.ndarray of noise-added and clipped intensities
    """
    intensities = np.asarray(intensities)
    assert np.all(intensities >= 0), "All intensities must be ≥ 0"

    log_vals = np.log1p(intensities)
    noisy_log_vals = log_vals + np.random.normal(0, log_noise_std, size=log_vals.shape)
    noised = np.expm1(noisy_log_vals)

    return np.clip(noised, 0, np.max(intensities))
