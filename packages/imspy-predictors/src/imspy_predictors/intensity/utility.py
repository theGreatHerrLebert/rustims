"""
Utility functions for fragment intensity prediction.

This module provides utilities for:
- Sequence tokenization for Prosit-style models
- Postprocessing of predicted fragment spectra
- Fragment ion label generation
"""

from numpy.typing import NDArray
from typing import List, Dict
import numba
import numpy as np
import pandas as pd

from imspy_core.utility import tokenize_unimod_sequence


# Lazy import for sagepy IonType (optional, requires imspy-search)
def _get_ion_type():
    """Lazy import of IonType. Requires sagepy (via imspy-search package)."""
    try:
        from sagepy.core import IonType
        return IonType
    except ImportError:
        raise ImportError(
            "IonType requires sagepy. Install imspy-search package for this functionality."
        )


@numba.njit
def _log_factorial(n: int, k: int) -> float:
    k = max(k, 2)
    result = 0.0
    for i in range(n, k - 1, -1):
        result += np.log(i)
    return result


def beta_score(fragments_observed, fragments_predicted) -> float:
    """
    The beta score is a variant of the OpenMS proposed score calculation, using predicted intensities instead of a constant value for the expected intensity.

    Note: This function requires sagepy (via imspy-search package).

    Args:
        fragments_observed: The Sage Fragment object containing the observed intensities
        fragments_predicted: The Sage Fragment object containing the predicted intensities, e.g. from Prosit

    Returns:
        float: The beta score, hyper score variant using predicted intensities instead of a constant value for the expected intensity
    """
    IonType = _get_ion_type()

    intensity = np.dot(fragments_observed.intensities, fragments_predicted.intensities)

    len_b, len_y = 0, 0

    b_type = IonType("b")
    y_type = IonType("y")

    for t in fragments_observed.ion_types:
        if t == b_type:
            len_b += 1
        elif t == y_type:
            len_y += 1

    i_min = min(len_b, len_y)
    i_max = max(len_b, len_y)

    return np.log1p(intensity) + 2.0 * _log_factorial(int(i_min), 2) + _log_factorial(int(i_max), int(i_min) + 1)


# Prosit alphabet mapping (standard amino acids with common modifications)
ALPHABET_UNMOD: Dict[str, int] = {
    '': 0,
    'A': 1,
    'C': 2,
    '[UNIMOD:4]': 3,  # Carbamidomethyl
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'K': 10,
    'L': 11,
    'M': 12,
    '[UNIMOD:35]': 13,  # Oxidation
    'N': 14,
    'P': 15,
    'Q': 16,
    'R': 17,
    'S': 18,
    'T': 19,
    'V': 20,
    'W': 21,
    'Y': 22,
}


def seq_to_index(seq: str, max_length: int = 30) -> NDArray:
    """Convert a sequence to a list of indices into the alphabet.

    Args:
        seq: A string representing a sequence of amino acids.
        max_length: The maximum length of the sequence to allow.

    Returns:
        A list of integers, each representing an index into the alphabet.
    """
    ret_arr = np.zeros(max_length, dtype=np.int32)
    tokenized_seq = tokenize_unimod_sequence(seq)[1:-1]
    assert len(tokenized_seq) <= max_length, f"Allowed sequence length is {max_length}, but got {len(tokenized_seq)}"

    aa_indices = []

    for s in tokenized_seq:
        if s in ALPHABET_UNMOD:
            aa_indices.append(ALPHABET_UNMOD[s])
        else:
            aa_indices.append(0)

    ret_arr[:len(aa_indices)] = aa_indices
    return ret_arr


def reshape_dims(intensities: NDArray) -> NDArray:
    """Reshape flat intensities to 3D (batch, seq_len, ion_types).

    Args:
        intensities: Flat array of shape (batch, 174) where 174 = 29 * 6

    Returns:
        Reshaped array of shape (batch, 29, 6)
    """
    if len(intensities.shape) == 1:
        # Single sample
        return intensities.reshape(29, 6)
    else:
        # Batch
        return intensities.reshape(-1, 29, 6)


def reshape_flat(intensities: NDArray) -> NDArray:
    """Reshape 3D intensities back to flat format.

    Args:
        intensities: Array of shape (batch, 29, 6) or (29, 6)

    Returns:
        Flat array of shape (batch, 174) or (174,)
    """
    if len(intensities.shape) == 2:
        return intensities.reshape(174)
    else:
        return intensities.reshape(-1, 174)


def normalize_base_peak(intensities: NDArray) -> NDArray:
    """Normalize intensities by base peak (max value).

    Args:
        intensities: Intensity array

    Returns:
        Normalized intensities (max = 1.0)
    """
    if len(intensities.shape) == 1:
        max_val = np.max(intensities[intensities >= 0])
        if max_val > 0:
            intensities = np.where(intensities >= 0, intensities / max_val, intensities)
    else:
        # Batch processing
        for i in range(len(intensities)):
            valid = intensities[i] >= 0
            max_val = np.max(intensities[i][valid]) if np.any(valid) else 0
            if max_val > 0:
                intensities[i] = np.where(valid, intensities[i] / max_val, intensities[i])
    return intensities


def mask_outofrange(intensities: NDArray, sequence_lengths: NDArray) -> NDArray:
    """Mask fragment ions that are out of sequence range.

    Args:
        intensities: 3D array of shape (batch, 29, 6)
        sequence_lengths: Array of sequence lengths

    Returns:
        Masked intensities with -1.0 for out-of-range positions
    """
    for i, seq_len in enumerate(sequence_lengths):
        # Fragment positions start at 1, max position is seq_len - 1
        max_frag_pos = seq_len - 1
        if max_frag_pos < 29:
            intensities[i, max_frag_pos:, :] = -1.0
    return intensities


def mask_outofcharge(intensities: NDArray, charges: NDArray) -> NDArray:
    """Mask fragment ions that exceed precursor charge.

    Args:
        intensities: 3D array of shape (batch, 29, 6)
        charges: Array of precursor charges

    Returns:
        Masked intensities with -1.0 for out-of-charge fragments
    """
    # Ion types in order: y+1, y+2, y+3, b+1, b+2, b+3
    for i, charge in enumerate(charges):
        # y ions: indices 0, 1, 2 (charges 1, 2, 3)
        # b ions: indices 3, 4, 5 (charges 1, 2, 3)
        for ion_charge in range(1, 4):
            if ion_charge > charge:
                # y ion index
                intensities[i, :, ion_charge - 1] = -1.0
                # b ion index
                intensities[i, :, ion_charge + 2] = -1.0
    return intensities


def post_process_predicted_fragment_spectra(data_pred: pd.DataFrame) -> NDArray:
    """
    Post process the predicted fragment intensities using pure NumPy.

    Args:
        data_pred: DataFrame containing the predicted fragment intensities
            Required columns: sequence_length, intensity_raw, charge

    Returns:
        numpy array of fragment intensities
    """
    # Get sequence length for masking out of sequence
    sequence_lengths = data_pred["sequence_length"].values

    # Get data
    intensities = np.stack(data_pred["intensity_raw"].to_numpy()).astype(np.float32)

    # Set negative intensity values to 0
    intensities[intensities < 0] = 0

    # Reshape to 3D
    intensities = reshape_dims(intensities)

    # Mask out of sequence and out of charge
    intensities = mask_outofrange(intensities, sequence_lengths)
    intensities = mask_outofcharge(intensities, data_pred.charge.values)

    # Reshape back to flat
    intensities = reshape_flat(intensities)

    # Save indices of -1.0 values, will be altered by intensity normalization
    m_idx = intensities == -1.0

    # Normalize to base peak
    intensities = normalize_base_peak(intensities)
    intensities[m_idx] = -1.0

    return intensities


def to_prosit_tensor(sequences: List) -> NDArray:
    """
    Translate a list of sequences to a numpy array of indices.

    Args:
        sequences: List of peptide sequences

    Returns:
        numpy array of shape (n_sequences, max_length)
    """
    return np.array([seq_to_index(s) for s in sequences], dtype=np.int32)


def get_prosit_intensity_flat_labels() -> List[str]:
    """
    Get the list of fragment ion labels for Prosit.
    Returns:
        List of fragment ion labels, giving the returned order of fragment intensities.
    """
    return [
        "y1+1",
        "y1+2",
        "y1+3",
        "b1+1",
        "b1+2",
        "b1+3",
        "y2+1",
        "y2+2",
        "y2+3",
        "b2+1",
        "b2+2",
        "b2+3",
        "y3+1",
        "y3+2",
        "y3+3",
        "b3+1",
        "b3+2",
        "b3+3",
        "y4+1",
        "y4+2",
        "y4+3",
        "b4+1",
        "b4+2",
        "b4+3",
        "y5+1",
        "y5+2",
        "y5+3",
        "b5+1",
        "b5+2",
        "b5+3",
        "y6+1",
        "y6+2",
        "y6+3",
        "b6+1",
        "b6+2",
        "b6+3",
        "y7+1",
        "y7+2",
        "y7+3",
        "b7+1",
        "b7+2",
        "b7+3",
        "y8+1",
        "y8+2",
        "y8+3",
        "b8+1",
        "b8+2",
        "b8+3",
        "y9+1",
        "y9+2",
        "y9+3",
        "b9+1",
        "b9+2",
        "b9+3",
        "y10+1",
        "y10+2",
        "y10+3",
        "b10+1",
        "b10+2",
        "b10+3",
        "y11+1",
        "y11+2",
        "y11+3",
        "b11+1",
        "b11+2",
        "b11+3",
        "y12+1",
        "y12+2",
        "y12+3",
        "b12+1",
        "b12+2",
        "b12+3",
        "y13+1",
        "y13+2",
        "y13+3",
        "b13+1",
        "b13+2",
        "b13+3",
        "y14+1",
        "y14+2",
        "y14+3",
        "b14+1",
        "b14+2",
        "b14+3",
        "y15+1",
        "y15+2",
        "y15+3",
        "b15+1",
        "b15+2",
        "b15+3",
        "y16+1",
        "y16+2",
        "y16+3",
        "b16+1",
        "b16+2",
        "b16+3",
        "y17+1",
        "y17+2",
        "y17+3",
        "b17+1",
        "b17+2",
        "b17+3",
        "y18+1",
        "y18+2",
        "y18+3",
        "b18+1",
        "b18+2",
        "b18+3",
        "y19+1",
        "y19+2",
        "y19+3",
        "b19+1",
        "b19+2",
        "b19+3",
        "y20+1",
        "y20+2",
        "y20+3",
        "b20+1",
        "b20+2",
        "b20+3",
        "y21+1",
        "y21+2",
        "y21+3",
        "b21+1",
        "b21+2",
        "b21+3",
        "y22+1",
        "y22+2",
        "y22+3",
        "b22+1",
        "b22+2",
        "b22+3",
        "y23+1",
        "y23+2",
        "y23+3",
        "b23+1",
        "b23+2",
        "b23+3",
        "y24+1",
        "y24+2",
        "y24+3",
        "b24+1",
        "b24+2",
        "b24+3",
        "y25+1",
        "y25+2",
        "y25+3",
        "b25+1",
        "b25+2",
        "b25+3",
        "y26+1",
        "y26+2",
        "y26+3",
        "b26+1",
        "b26+2",
        "b26+3",
        "y27+1",
        "y27+2",
        "y27+3",
        "b27+1",
        "b27+2",
        "b27+3",
        "y28+1",
        "y28+2",
        "y28+3",
        "b28+1",
        "b28+2",
        "b28+3",
        "y29+1",
        "y29+2",
        "y29+3",
        "b29+1",
        "b29+2",
        "b29+3",
    ]
