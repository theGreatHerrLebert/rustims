import re
from collections import defaultdict
import numpy as np
import mendeleev as me

MASS_PROTON = 1.007276466583
MASS_NEUTRON = 1.00866491597
MASS_ELECTRON = 0.00054857990946
MASS_WATER = 18.0105646863

# IUPAC standard in Kelvin
STANDARD_TEMPERATURE = 273.15
# IUPAC standard in Pa
STANDARD_PRESSURE = 1e5
# IUPAC elementary charge
ELEMENTARY_CHARGE = 1.602176634e-19
# IUPAC BOLTZMANN'S CONSTANT
K_BOLTZMANN = 1.380649e-23

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

    return mass_sequence + mass_modifics + MASS_WATER


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


def get_monoisotopic_token_weight(token:str):
    """
    Gets monoisotopic weight of token

    :param token: Token of aa sequence e.g. "<START>[UNIMOD:1]"
    :type token: str
    :return: Weight in Dalton.
    :rtype: float
    """
    splits = token.split("[")
    for i in range(1, len(splits)):
        splits[i] = "["+splits[i]

    mass = 0
    for split in splits:
        mass += AMINO_ACID_MASSES[split]
    return mass


def get_mono_isotopic_weight(sequence_tokenized: list[str]) -> float:
    mass = 0
    for token in sequence_tokenized:
        mass += get_monoisotopic_token_weight(token)
    return mass + MASS_WATER


def get_mass_over_charge(mass: float, charge: int) -> float:
    return (mass / charge) + MASS_PROTON


def get_num_protonizable_sites(sequence: str) -> int:
    """
    Gets number of sites that can be protonized.
    This function does not yet account for PTMs.

    :param sequence: Amino acid sequence
    :type sequence: str
    :return: Number of protonizable sites
    :rtype: int
    """
    sites = 1 # n-terminus
    for s in sequence:
        if s in ["H", "R", "K"]:
            sites += 1
    return sites


class ChemicalCompound:

    def _calculate_molecular_mass(self):
        mass = 0
        for (atom, abundance) in self.element_composition.items():
            mass += me.element(atom).atomic_weight * abundance
        return mass

    def __init__(self, formula):
        self.element_composition = self.get_composition(formula)
        self.mass = self._calculate_molecular_mass()

    def get_composition(self, formula: str):
        """
        Parse chemical formula into Dict[str:int] with
        atoms as keys and the respective counts as values.

        :param formula: Chemical formula of compound e.g. 'C6H12O6'
        :type formula: str
        :return: Dictionary Atom: Count
        :rtype: Dict[str:int]
        """
        if formula.startswith("("):
            assert formula.endswith(")")
            formula = formula[1:-1]

        tmp_group = ""
        tmp_group_count = ""
        depth = 0
        comp_list = []
        comp_counts = []

        # extract components: everything in brackets and atoms
        # extract component counts: number behind component or 1
        for (i, e) in enumerate(formula):
            if e == "(":
                depth += 1
                if depth == 1:
                    if tmp_group != "":
                        comp_list.append(tmp_group)
                        tmp_group = ""
                        if tmp_group_count == "":
                            comp_counts.append(1)
                        else:
                            comp_counts.append(int(tmp_group_count))
                            tmp_group_count = ""
                tmp_group += e
                continue
            if e == ")":
                depth -= 1
                tmp_group += e
                continue
            if depth > 0:
                tmp_group += e
                continue
            if e.isupper():
                if tmp_group != "":
                    comp_list.append(tmp_group)
                    tmp_group = ""
                    if tmp_group_count == "":
                        comp_counts.append(1)
                    else:
                        comp_counts.append(int(tmp_group_count))
                        tmp_group_count = ""
                tmp_group += e
                continue
            if e.islower():
                tmp_group += e
                continue
            if e.isnumeric():
                tmp_group_count += e
        if tmp_group != "":
            comp_list.append(tmp_group)
            if tmp_group_count == "":
                comp_counts.append(1)
            else:
                comp_counts.append(int(tmp_group_count))

        # assemble dictionary from component lists
        atom_dict = {}
        for (comp, count) in zip(comp_list, comp_counts):
            if not comp.startswith("("):
                atom_dict[comp] = count
            else:
                atom_dicts_depth = self.get_composition(comp)
                for atom in atom_dicts_depth:
                    atom_dicts_depth[atom] *= count
                    if atom in atom_dict:
                        atom_dict[atom] += atom_dicts_depth[atom]
                    else:
                        atom_dict[atom] = atom_dicts_depth[atom]
                atom_dicts_depth = {}
        return atom_dict


class BufferGas(ChemicalCompound):

    def __init__(self, formula: str):
        super().__init__(formula)
