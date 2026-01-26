"""
Utility functions for CCS prediction.

This module provides tokenization and data preparation utilities for CCS models.
"""

import json
import importlib.resources as resources
from typing import List, Dict


def load_tokenizer_from_resources(tokenizer_name: str = "unimod-vocab") -> Dict[str, int]:
    """Load a tokenizer from resources.

    Args:
        tokenizer_name: Name of the tokenizer file (without .json extension)

    Returns:
        The pretrained tokenizer as a dictionary mapping tokens to indices
    """
    path = resources.files('imspy_predictors.pretrained').joinpath(f'{tokenizer_name}.json')

    # Load from file
    with open(path, "r") as f:
        tokenizer = json.load(f)

    return tokenizer


def token_list_from_sequence(sequence: str) -> List[str]:
    """Tokenize a peptide sequence into a list of single-letter amino acids and modifications.

    Args:
        sequence (str): The peptide sequence to tokenize.

    Returns:
        List[str]: List of tokens.
    """
    # Split the sequence into tokens
    tokens = re.findall(r'\[.*?]|\w', sequence)

    return ['<SOS>'] + tokens + ['<EOS>']


def tokenize_and_pad(
    token_list: List[str],
    tokenizer: Dict[str, int],
    target_len: int = 50,
    post: bool = True
) -> List[int]:
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
