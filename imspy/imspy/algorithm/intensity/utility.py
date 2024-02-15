from numpy.typing import NDArray
from typing import List

import numpy as np

import pandas as pd

import tensorflow as tf
from imspy.simulation.utility import sequence_to_numpy
from dlomix.constants import ALPHABET_UNMOD

from tensorflow.keras.layers.experimental import preprocessing

from dlomix.reports.postprocessing import (reshape_flat, reshape_dims,
                                           normalize_base_peak, mask_outofcharge, mask_outofrange)


def seq_to_index(seq: str) -> List[int]:
    """Convert a sequence to a list of indices into the alphabet.

    Args:
        seq: A string representing a sequence of amino acids.

    Returns:
        A list of integers, each representing an index into the alphabet.
    """
    return [ALPHABET_UNMOD[s] for s in seq]


# Your existing code for data preparation, with modifications to name the inputs
def generate_prosit_intensity_prediction_dataset(
        sequences: List[str],
        charges: NDArray,
        collision_energies: NDArray | None = None):

    # set default for unprovided collision_energies
    if collision_energies is None:
        collision_energies = np.expand_dims(np.repeat([0.35], len(charges)), 1)

    # check for 1D collision_energies, need to be expanded to 2D
    elif len(collision_energies.shape) == 1:
        collision_energies = np.expand_dims(collision_energies, 1)

    # Create a string lookup layer to convert sequences to indices, and one-hot encode charges
    string_lookup = preprocessing.StringLookup(
        vocabulary=list(ALPHABET_UNMOD.keys())
    )

    charges = tf.one_hot(charges - 1, depth=6)
    sequences = tf.cast(string_lookup([sequence_to_numpy(s) for s in sequences]), dtype=tf.int32)

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

    return np.squeeze(reshape_dims(intensities))


def to_prosit_tensor(sequences: List) -> tf.Tensor:
    """
    translate a list of fixed length numpy arrays into a tensorflow tensor
    Args:
        sequences: list of numpy arrays, representing peptide sequences

    Returns:
        tensorflow tensor
    """
    return tf.convert_to_tensor(sequences, dtype=tf.string)
