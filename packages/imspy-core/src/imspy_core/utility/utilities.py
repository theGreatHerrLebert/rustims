import io
import json
import math
import numba
import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import tensorflow as tf

from numpy.typing import ArrayLike


@numba.jit(nopython=True)
def normal_pdf(x: ArrayLike, mass: float, s: float = 0.001,
               inv_sqrt_2pi: float = 0.3989422804014327, normalize: bool = False):
    """
    Args:
        x:
        mass:
        s:
        inv_sqrt_2pi:
        normalize:
    """
    a = (x - mass) / s
    if normalize:
        return np.exp(-0.5 * np.power(a, 2))
    else:
        return inv_sqrt_2pi / s * np.exp(-0.5 * np.power(a, 2))


@numba.jit(nopython=True)
def gaussian(x, μ: float = 0, σ: float = 1):
    """
    Gaussian function
    :param x:
    :param μ:
    :param σ:
    :return:
    """
    A = 1 / np.sqrt(2 * np.pi * np.power(σ, 2))
    B = np.exp(- (np.power(x - μ, 2) / 2 * np.power(σ, 2)))

    return A * B


@numba.jit(nopython=True)
def exp_distribution(x, λ: float = 1):
    """
    Exponential function
    :param x:
    :param λ:
    :return:
    """
    if x > 0:
        return λ * np.exp(-λ * x)
    return 0


@numba.jit(nopython=True)
def exp_gaussian(x, μ: float = -3, σ: float = 1, λ: float = .25):
    """
    laplacian distribution with exponential decay
    :param x:
    :param μ:
    :param σ:
    :param λ:
    :return:
    """
    A = λ / 2 * np.exp(λ / 2 * (2 * μ + λ * np.power(σ, 2) - 2 * x))
    B = math.erfc((μ + λ * np.power(σ, 2) - x) / (np.sqrt(2) * σ))
    return A * B


class NormalDistribution:
    def __init__(self, μ: float, σ: float):
        self.μ = μ
        self.σ = σ

    def __call__(self, x):
        return gaussian(x, self.μ, self.σ)


class ExponentialGaussianDistribution:
    def __init__(self, μ: float = -3, σ: float = 1, λ: float = .25):
        self.μ = μ
        self.σ = σ
        self.λ = λ

    def __call__(self, x):
        return exp_gaussian(x, self.μ, self.σ, self.λ)


def _from_jsons(jsons: str):
    return json.loads(jsons)


class TokenSequence:

    def __init__(self, sequence_tokenized: Optional[List[str]] = None, jsons: Optional[str] = None):
        if jsons is not None:
            self.sequence_tokenized = _from_jsons(jsons)
            self._jsons = jsons
        else:
            self.sequence_tokenized = sequence_tokenized
            self._jsons = self._to_jsons()

    def _to_jsons(self):
        json_dict = self.sequence_tokenized
        return json.dumps(json_dict)

    @property
    def jsons(self):
        return self._jsons


def is_unimod_start(char: str):
    """
    Tests if char is start of unimod
    bracket

    :param char: Character of a proForma formatted aa sequence
    :type char: str
    :return: Whether char is start of unimod bracket
    :rtype: bool
    """
    if char in ["(", "[", "{"]:
        return True
    else:
        return False


def is_unimod_end(char: str):
    """
    Tests if char is end of unimod
    bracket

    :param char: Character of a proForma formatted aa sequence
    :type char: str
    :return: Whether char is end of unimod bracket
    :rtype: bool
    """
    if char in [")", "]", "}"]:
        return True
    else:
        return False


def tokenize_proforma_sequence(sequence: str):
    """
    Tokenize a ProForma formatted sequence string.

    :param sequence: Sequence string (ProForma formatted)
    :type sequence: str
    :return: List of tokens
    :rtype: List
    """
    sequence = sequence.upper().replace("(", "[").replace(")", "]")
    token_list = ["<START>"]
    in_unimod_bracket = False
    tmp_token = ""

    for aa in sequence:
        if is_unimod_start(aa):
            in_unimod_bracket = True
        if in_unimod_bracket:
            if is_unimod_end(aa):
                in_unimod_bracket = False
            tmp_token += aa
            continue
        if tmp_token != "":
            token_list.append(tmp_token)
            tmp_token = ""
        tmp_token += aa

    if tmp_token != "":
        token_list.append(tmp_token)

    if len(token_list) > 1:
        if token_list[1].find("UNIMOD:1") != -1:
            token_list[1] = "<START>"+token_list[1]
            token_list = token_list[1:]
    token_list.append("<END>")

    return token_list


def get_aa_num_proforma_sequence(sequence: str):
    """
    get number of amino acids in sequence

    :param sequence: proforma formatted aa sequence
    :type sequence: str
    :return: Number of amino acids
    :rtype: int
    """
    num_aa = 0
    inside_bracket = False

    for aa in sequence:
        if is_unimod_start(aa):
            inside_bracket = True
        if inside_bracket:
            if is_unimod_end(aa):
                inside_bracket = False
            continue
        num_aa += 1
    return num_aa


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


def tokenizer_to_json(tokenizer: "tf.keras.preprocessing.text.Tokenizer", path: str):
    """
    save a fit keras tokenizer to json for later use

    Note:
        Requires tensorflow to be installed.

    :param tokenizer: fit keras tokenizer to save
    :param path: path to save json to
    """
    tokenizer_json = tokenizer.to_json()
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


def tokenizer_from_json(path: str):
    """
    load a pre-fit tokenizer from a json file

    Note:
        Requires tensorflow to be installed.

    :param path: path to tokenizer as json file
    :return: a keras tokenizer loaded from json
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "tokenizer_from_json requires tensorflow. "
            "Install it with: pip install tensorflow"
        )

    with open(path) as f:
        data = json.load(f)
    return tf.keras.preprocessing.text.tokenizer_from_json(data)
