import io
import json
import numpy as np
import tensorflow as tf
from typing import List, Optional


class TokenSequence:

    def __init__(self, sequence_tokenized: Optional[List[str]] = None, jsons:Optional[str] = None):
        if jsons is not None:
            self.sequence_tokenized = self._from_jsons(jsons)
            self._jsons = jsons
        else :
            self.sequence_tokenized = sequence_tokenized
            self._jsons = self._to_jsons()

    def _to_jsons(self):
        json_dict = self.sequence_tokenized
        return json.dumps(json_dict)

    def _from_jsons(self, jsons:str):
        return json.loads(jsons)

    @property
    def jsons(self):
        return self._jsons


def is_unimod_start(char:str):
    """
    Tests if char is start of unimod
    bracket

    :param char: Character of a proForma formatted aa sequence
    :type char: str
    :return: Wether char is start of unimod bracket
    :rtype: bool
    """
    if char in ["(","[","{"]:
        return True
    else:
        return False


def is_unimod_end(char:str):
    """
    Tests if char is end of unimod
    bracket

    :param char: Character of a proForma formatted aa sequence
    :type char: str
    :return: Wether char is end of unimod bracket
    :rtype: bool
    """
    if char in [")","]","}"]:
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
    sequence = sequence.upper().replace("(","[").replace(")","]")
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

def get_aa_num_proforma_sequence(sequence:str):
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
