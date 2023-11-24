import re
from typing import List


def tokenize_unimod_sequence(unimod_sequence: str) -> List[str]:
    """
    Tokenizes a sequence of modified amino acids.
    Args:
        unimod_sequence: A string representing the sequence of amino acids with modifications.

    Returns:
        A list of tokenized amino acids.
    """
    token_pattern = r'[A-Z](?:\[UNIMOD:\d+\])?'

    # Special case handling for [UNIMOD:1] at the beginning
    if unimod_sequence.startswith("[UNIMOD:1]"):
        special_token = "<START>[UNIMOD:1]"
        rest_of_string = unimod_sequence[len("[UNIMOD:1]"):]
        other_tokens = re.findall(token_pattern, rest_of_string)
        return [special_token] + other_tokens + ['<END>']
    else:
        tokens = re.findall(token_pattern, unimod_sequence)
        return ['<START>'] + tokens + ['<END>']
