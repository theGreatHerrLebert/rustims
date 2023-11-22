import json
import numpy as np
import tensorflow as tf


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


def tokenizer_to_json(tokenizer: tf.keras.preprocessing.text.Tokenizer, path: str):
    """
    save a fit keras tokenizer to json for later use
    :param tokenizer: fit keras tokenizer to save
    :param path: path to save json to
    """
    tokenizer_json = tokenizer.to_json()
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


def tokenizer_from_json(path: str):
    """
    load a pre-fit tokenizer from a json file
    :param path: path to tokenizer as json file
    :return: a keras tokenizer loaded from json
    """
    with open(path) as f:
        data = json.load(f)
    return tf.keras.preprocessing.text.tokenizer_from_json(data)
