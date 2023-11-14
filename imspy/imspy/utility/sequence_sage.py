import numpy as np
from numpy.typing import NDArray


def mass_to_modification(mass: float) -> str:
    """ Convert a mass to a UNIMOD modification annotation.

    Args:
        mass: a mass in Da

    Returns:
        a UNIMOD modification annotation
    """
    maybe_key = int(np.round(mass))
    # TODO: find a better way to do the map-back
    mod_dict = {
        16: '[UNIMOD:35]',
        42: '[UNIMOD:1]',
        57: '[UNIMOD:4]',
        80: '[UNIMOD:21]',
        119: '[UNIMOD:312]',
    }
    # try to translate to UNIMOD annotation
    try:
        return mod_dict[maybe_key]
    except KeyError:
        raise KeyError(f"Rounded mass not in dict: {maybe_key}, known mods: {mod_dict}")


def tokenize_sage_sequence(sequence: str, modifications: NDArray) -> list[str]:
    """ Tokenize a sequence with modifications into a list of tokens that are ionmob compatible.

    Args:
        sequence: a string of amino acids
        modifications: a numpy array of modifications, where 0 is no modification and mods are given as masses

    Returns:
        a list of tokens that are ionmob compatible
    """
    assert len(sequence) == len(modifications), "Sequence and modifications list need to be same length."
    seq_list = []
    # TODO: Acetyl N-term is not handled here
    start, end = ['<START>'], ['<END>']
    for char, mod in zip(sequence, modifications):
        if mod == 0:
            seq_list.append(char)
        else:
            seq_list.append(char + mass_to_modification(mod))

    return start + seq_list + end
