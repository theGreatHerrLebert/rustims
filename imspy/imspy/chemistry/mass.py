import re
from collections import defaultdict
import numpy as np

MASS_PROTON = 1.007276466583
MASS_NEUTRON = 1.00866491597
MASS_ELECTRON = 0.00054857990946
MASS_H2O = 18.0105646863

AMINO_ACIDS = {
    'Lysine': 'K',
    'Alanine': 'A',
    'Glycine': 'G',
    'Valine': 'V',
    'Tyrosine': 'Y',
    'Arginine': 'R',
    'Glutamic Acid': 'E',
    'Phenylalanine': 'F',
    'Tryptophan': 'W',
    'Leucine': 'L',
    'Threonine': 'T',
    'Cysteine': 'C',
    'Serine': 'S',
    'Glutamine': 'Q',
    'Methionine': 'M',
    'Isoleucine': 'I',
    'Asparagine': 'N',
    'Proline': 'P',
    'Histidine': 'H',
    'Aspartic Acid': 'D'
}

AMINO_ACID_MASSES = {
    'A': 71.037114,
    'R': 156.101111,
    'N': 114.042927,
    'D': 115.026943,
    'C': 103.009185,
    'E': 129.042593,
    'Q': 128.058578,
    'G': 57.021464,
    'H': 137.058912,
    'I': 113.084064,
    'L': 113.084064,
    'K': 128.094963,
    'M': 131.040485,
    'F': 147.068414,
    'P': 97.052764,
    'S': 87.032028,
    'T': 101.047679,
    'W': 186.079313,
    'Y': 163.063329,
    'V': 99.068414
}

MODIFICATIONS_MZ = {
    # currently unclear if correct, TODO: check
    '[UNIMOD:58]': 56.026215, '[UNIMOD:408]': 148.037173,
    # correct
    '[UNIMOD:43]': 203.079373, '[UNIMOD:7]': 0.984016,
    '[UNIMOD:1]': 42.010565, '[UNIMOD:35]': 15.994915, '[UNIMOD:1289]': 70.041865,
    '[UNIMOD:3]': 226.077598, '[UNIMOD:1363]': 68.026215, '[UNIMOD:36]': 28.031300,
    '[UNIMOD:122]': 27.994915, '[UNIMOD:1848]': 114.031694, '[UNIMOD:1849]': 86.036779,
    '[UNIMOD:64]': 100.016044, '[UNIMOD:37]': 42.046950, '[UNIMOD:121]': 114.042927,
    '[UNIMOD:747]': 86.000394, '[UNIMOD:34]': 14.015650, '[UNIMOD:354]': 44.985078,
    '[UNIMOD:4]': 57.021464, '[UNIMOD:21]': 79.966331, '[UNIMOD:312]': 119.004099
}


MODIFICATIONS_MZ_NUMERICAL = {
    # currently unclear if correct, TODO: check
    58: 56.026215, 408: 148.037173,
    # correct
    43: 203.079373, 7: 0.984016,
    1: 42.010565, 35: 15.994915, 1289: 70.041865,
    3: 226.077598, 1363: 68.026215, 36: 28.031300,
    122: 27.994915, 1848: 114.031694, 1849: 86.036779,
    64: 100.016044, 37: 42.046950, 121: 114.042927,
    747: 86.000394, 34: 14.015650, 354: 44.985078,
    4: 57.021464, 21: 79.966331, 312: 119.004099
}


def tokenize_amino_acids(sequence):
    """
    Tokenizes a sequence of modified amino acids.

    Each character stands for itself, and if a modification is at the beginning,
    the modification stands for itself. Otherwise, it should be the suffix of the amino acid.

    Args:
    sequence (str): A string representing the sequence of amino acids with modifications.

    Returns:
    List[str]: A list of tokenized amino acids.
    """
    # Regular expression pattern to match amino acids with or without modifications
    pattern = r'(\[UNIMOD:\d+\])?([A-Z])(\[[A-Z]+:\d+\])?'

    # Find all matches using the regular expression
    matches = re.findall(pattern, sequence)

    # Process the matches to form the tokenized list
    tokens = []
    for match in matches:
        mod1, aa, mod2 = match
        if mod1:
            # If there's a modification at the beginning, it stands for itself
            tokens.append(mod1)
        # Add the amino acid, with or without the suffix modification
        tokens.append(aa + mod2 if mod2 else aa)

    return tokens


def calculate_monoisotopic_mass(sequence):
    """
    Calculates the monoisotopic mass of a sequence of amino acids with modifications.

    Args:
    sequence (str): A string representing the sequence of amino acids with modifications.

    Returns:
    float: The monoisotopic mass of the sequence.
    """
    # Regex pattern to find modifications in the format [UNIMOD:number]
    pattern = r"\[UNIMOD:(\d+)\]"

    # Find all occurrences of the pattern
    modifications = re.findall(pattern, sequence)

    # Count occurrences of each modification number
    mod_counts = defaultdict(int)
    for mod in modifications:
        mod_counts[int(mod)] += 1

    # Remove the modifications from the sequence
    sequence = re.sub(pattern, '', sequence)

    # Count occurrences of each amino acid
    aa_counts = defaultdict(int)
    for char in sequence:
        aa_counts[char] += 1

    # mass of amino acids and modifications
    mass_sequence = np.sum([AMINO_ACID_MASSES[amino_acid] * count for amino_acid, count in aa_counts.items()])
    mass_modifics = np.sum([MODIFICATIONS_MZ_NUMERICAL[mod] * count for mod, count in mod_counts.items()])

    return mass_sequence + mass_modifics + MASS_H2O


def calculate_mz(monoisotopic_mass, charge):
    """
    Calculates the m/z of a sequence of amino acids with modifications.

    Args:
    sequence (str): A string representing the sequence of amino acids with modifications.

    Returns:
    float: The m/z of the sequence.
    """

    return (monoisotopic_mass + charge * MASS_PROTON) / charge


def calculate_mz_from_sequence(sequence, charge):
    """
    Calculates the m/z of a sequence of amino acids with modifications.

    Args:
    sequence (str): A string representing the sequence of amino acids with modifications.

    Returns:
    float: The m/z of the sequence.
    """
    return calculate_mz(calculate_monoisotopic_mass(sequence), charge)
