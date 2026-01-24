import json
import re
import importlib.resources as resources

from typing import List, Dict

import numpy as np

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

from imspy_predictors.ccs.model_std import GRUCCSPredictorStd
from imspy_predictors.ccs.predictors import SquareRootProjectionLayer
from imspy_predictors.utility import get_model_path
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model

def load_tokenizer_from_resources(tokenizer_name: str = "unimod-vocab") -> Dict[str, int]:
    """ Load a tokenizer from resources

    Returns:
        The pretrained tokenizer
    """
    path = resources.files('imspy_predictors.pretrained').joinpath(f'{tokenizer_name}.json')

    # Load from file
    with open(path, "r") as f:
        tokenizer = json.load(f)

    return tokenizer

def token_list_from_sequence(sequence: str) -> List[str]:
    """ Tokenize a peptide sequence into a list of single-letter amino acids and modifications.

    Args:
        sequence (str): The peptide sequence to tokenize.

    Returns:
        List[str]: List of tokens.
    """

    # Split the sequence into tokens
    tokens = re.findall(r'\[.*?]|\w', sequence)

    return ['<SOS>'] + tokens + ['<EOS>']

def tokenize_and_pad(token_list: List[str], tokenizer: Dict[str, int], target_len: int = 50, post: bool = True) -> List[int]:
    """
    Tokenizes a list of strings and pads or truncates to a target length.

    Args:
        token_list (List[str]): List of strings to tokenize.
        tokenizer (dict): Dictionary mapping tokens to their indices.
        target_len (int): The target length for padding/truncating. Default is 50.
        post (bool): If True, pad/truncate at the end. If False, pad/truncate at the beginning.

    Returns:
        List[int]: Tokenized and padded/truncated list of integers.
    """
    # Convert tokens to indices using the tokenizer dictionary
    token_indices = [tokenizer.get(token, tokenizer.get('<UNK>', 0)) for token in token_list]

    # Truncate if the token list exceeds the target length
    if len(token_indices) > target_len:
        if post:
            token_indices = token_indices[:target_len]
        else:
            token_indices = token_indices[-target_len:]

    # Calculate padding length
    padding_length = target_len - len(token_indices)

    # Pad with PAD token (3) to the target length
    if padding_length > 0:
        padding = [3] * padding_length  # PAD token is 3
        if post:
            token_indices.extend(padding)
        else:
            token_indices = padding + token_indices

    return token_indices


@keras.saving.register_keras_serializable()
class CustomLossMean(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        """
        MSE loss for the primary output (mean CCS prediction)
        """
        return tf.reduce_mean(tf.square(y_true - y_pred))

@keras.saving.register_keras_serializable()
class CustomLossStd(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        """
        MSE loss for the secondary output (predicted CCS standard deviation),
        with masking for invalid `-1` values in y_true.
        """
        # Create a mask for valid values (y_true != -1)
        mask = tf.not_equal(y_true, -1.0)
        # Apply the mask to y_true and y_pred
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        # Compute the MSE only on valid values
        return tf.reduce_mean(tf.square(y_true_masked - y_pred_masked))


def load_deep_ccs_std_predictor() -> tf.keras.models.Model:

    path = get_model_path('ccs/ionmob-std-20-01-2025.keras')

    custom_objects = {
        'SquareRootProjectionLayer': SquareRootProjectionLayer,
        'GRUCCSPredictorStd': GRUCCSPredictorStd,
        'CustomLossMean': CustomLossMean,
        'CustomLossStd': CustomLossStd,
    }

    return load_model(path, custom_objects=custom_objects, compile=False)


def to_tf_dataset_with_variance(
        mz: np.ndarray,
        charge: np.ndarray,
        sequences: np.ndarray,
        ccs: np.ndarray = None,
        ccs_std: np.ndarray = None,
        tokenizer: Dict[str, int] = None,
        batch: bool = True,
        batch_size: int = 2048,
        shuffle: bool = True,
        remove_unimod: bool = False
):
    """
    Create a TensorFlow dataset from input data, including CCS standard deviation
    Args:
        mz: a numpy array of m/z values
        charge: a numpy array of charge values
        sequences: a numpy array of peptide sequences
        ccs: a numpy array of CCS values, if available
        ccs_std: a numpy array of CCS standard deviation values, if available
        tokenizer: a dictionary mapping tokens to their indices, default is the UniMod vocabulary
        batch: if True, return a batched dataset
        batch_size: the batch size for the dataset
        shuffle: if True, shuffle the dataset
        remove_unimod: if True, remove UniMod annotations from sequences

    Returns:
        A TensorFlow dataset, when ccs and ccs_std are available, the dataset will include both CCS and CCS standard deviation for training,
        otherwise, dataset will only be for prediction
    """
    if tokenizer is None:
        tokenizer = load_tokenizer_from_resources()

    # prepare masses, charges, sequences
    masses = np.expand_dims(mz, 1)
    charges_one_hot = tf.one_hot(charge - 1, 4)

    seq_padded = []

    for seq in sequences:
        if remove_unimod:
            seq = remove_unimod_annotation(seq)
        seq_padded.append(tokenize_and_pad(token_list_from_sequence(seq), tokenizer))

    # prepare ccs
    if ccs is not None:
        ccs = np.expand_dims(ccs, 1)
    else:
        ccs = np.zeros_like(masses)

    # prepare ccs_std
    if ccs_std is not None:
        ccs_std = np.expand_dims(ccs_std, 1)
    else:
        ccs_std = np.zeros_like(ccs)

    # generate dataset
    ds = tf.data.Dataset.from_tensor_slices(((masses, charges_one_hot, np.array(seq_padded)), (ccs, ccs_std, ccs_std)))

    if shuffle:
        ds = ds.shuffle(len(sequences))

    if batch:
        return ds.batch(batch_size)
    return ds
