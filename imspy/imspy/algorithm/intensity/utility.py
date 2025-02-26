import os

from numpy.typing import NDArray
from typing import List, Tuple
import numba
import numpy as np
import pandas as pd

import tensorflow as tf

from dlomix.constants import PTMS_ALPHABET, ALPHABET_UNMOD

from dlomix.reports.postprocessing import (reshape_flat, reshape_dims,
                                           normalize_base_peak, mask_outofcharge, mask_outofrange)
from sagepy.core import IonType

import re
from imspy.utility import tokenize_unimod_sequence


def remove_unimod_annotation(sequence: str) -> str:
    """
    Remove the unimod annotation from a peptide sequence.
    Args:
        sequence: a peptide sequence

    Returns:
        str: the peptide sequence without unimod annotation
    """

    pattern = r'\[UNIMOD:\d+\]'
    return re.sub(pattern, '', sequence)

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
    Args:
        fragments_observed: The Sage Fragment object containing the observed intensities
        fragments_predicted: The Sage Fragment object containing the predicted intensities, e.g. from Prosit

    Returns:
        float: The beta score, hyper score variant using predicted intensities instead of a constant value for the expected intensity
    """

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


# Your existing code for data preparation, with modifications to name the inputs
def generate_prosit_intensity_prediction_dataset(
        sequences: List[str],
        charges: NDArray,
        collision_energies: NDArray | None = None,
        remove_mods: bool = True,
):
    """
    Generate a dataset for predicting fragment intensities using Prosit.
    Args:
        sequences: A list of peptide sequences.
        charges: A numpy array of precursor charges.
        collision_energies: A numpy array of collision energies.
        remove_mods: Whether to remove modifications from the sequences.

    Returns:
        A tf.data.Dataset object that yields batches of data in the format expected by the model.
    """


    # set default for unprovided collision_energies
    if collision_energies is None:
        collision_energies = np.expand_dims(np.repeat([0.35], len(charges)), 1)

    # check for 1D collision_energies, need to be expanded to 2D
    elif len(collision_energies.shape) == 1:
        collision_energies = np.expand_dims(collision_energies, 1)

    charges = tf.one_hot(charges - 1, depth=6)

    if remove_mods:
        sequences = tf.cast([seq_to_index(remove_unimod_annotation(s)) for s in sequences], dtype=tf.int32)
    else:
        sequences = tf.cast([seq_to_index(s) for s in sequences], dtype=tf.int32)

    # Create a dataset that yields batches in the format expected by the model??
    dataset = tf.data.Dataset.from_tensor_slices((
        {"peptides_in": sequences,
         "precursor_charge_in": charges,
         "collision_energy_in": tf.cast(collision_energies, dtype=tf.float32)}
    ))

    return dataset


def unpack_dict(features):
    """Unpack the dictionary of features into the expected inputs for the model.

    Args:
        features: A dictionary of features, with keys 'peptides_in', 'precursor_charge_in', and 'collision_energy_in'

    Returns:
        A tuple of the expected inputs for the model: (peptides, precursor_charge, collision_energy)
    """
    return features['peptides_in'], features['precursor_charge_in'], features['collision_energy_in']


def post_process_predicted_fragment_spectra(data_pred: pd.DataFrame) -> NDArray:
    """
    post process the predicted fragment intensities
    Args:
        data_pred: dataframe containing the predicted fragment intensities

    Returns:
        numpy array of fragment intensities
    """
    # get sequence length for masking out of sequence
    sequence_lengths = data_pred["sequence_length"]

    # get data
    intensities = np.stack(data_pred["intensity_raw"].to_numpy()).astype(np.float32)
    # set negative intensity values to 0
    intensities[intensities < 0] = 0
    intensities = reshape_dims(intensities)

    # mask out of sequence and out of charge
    intensities = mask_outofrange(intensities, sequence_lengths)
    intensities = mask_outofcharge(intensities, data_pred.charge)
    intensities = reshape_flat(intensities)

    # save indices of -1.0 values, will be altered by intensity normalization
    m_idx = intensities == -1.0
    # normalize to base peak
    intensities = normalize_base_peak(intensities)
    intensities[m_idx] = -1.0

    return intensities


def to_prosit_tensor(sequences: List) -> tf.Tensor:
    """
    translate a list of fixed length numpy arrays into a tensorflow tensor
    Args:
        sequences: list of numpy arrays, representing peptide sequences

    Returns:
        tensorflow tensor
    """
    return tf.convert_to_tensor(sequences, dtype=tf.string)


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
        "b29+3"
    ]
