import numpy as np


def re_index_indices(ids):
    _, inverse = np.unique(ids, return_inverse=True)
    return inverse
