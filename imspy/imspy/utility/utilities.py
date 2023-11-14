import numpy as np


def re_index_indices(ids):
    """Re-index indices, i.e. replace gaps in indices with consecutive numbers.
    Can be used, e.g., to re-index frame IDs from precursors for visualization.
    Args:
        ids: Indices.
    Returns:
        Indices.
    """
    _, inverse = np.unique(ids, return_inverse=True)
    return inverse
