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
