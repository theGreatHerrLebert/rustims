import io
import json
from typing import List, Optional
import tensorflow as tf
import numpy as np
from numpy.typing import ArrayLike

import numba
import math
import pandas as pd

from tqdm import tqdm
from pyims.data import MzSpectrum
# TODO bring to pyims
#from proteolizardalgo.hashing import ReferencePattern


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

def proteome_from_fasta(path: str) -> pd.DataFrame:
    """
    Read a fasta file and return a dataframe with the protein name and sequence.
    :param path: path to the fasta file
    :return: a dataframe with the protein name and sequence
    """
    d = {}
    with open(path) as infile:
        gene = ''
        header = ''
        for i, line in enumerate(infile):
            if line.find('>') == -1:
                gene += line
            elif line.find('>') != -1 and i > 0:
                header = line
                d[header.replace('\n', '')[4:]] = gene.replace('\n', '')
                gene = ''
            elif i == 0:
                header = line

    row_list = []

    for key, value in d.items():
        split_index = key.find(' ')
        gene_id = key[:split_index].split('|')[0]
        rest = key[split_index:]
        row = {'id': gene_id, 'meta_data': rest, 'sequence': value}
        row_list.append(row)

    return pd.DataFrame(row_list)


def peak_width_preserving_mz_transform(
        mz: np.array,
        M0: float = 500,
        resolution: float = 50_000):
    """
    Transform values into an index that fixes the width of the peak at half full width.
    Arguments:
        mz (np.array): An array of mass to charge ratios to transform.
        M0 (float): The m/z value at which TOFs resolution is reported.
        resolution (float): The resolution of the TOF instrument.
    """
    return (np.log(mz) - np.log(M0)) / np.log1p(1 / resolution)


def bins_to_mz(mz_bin, win_len):
    return np.abs(mz_bin) * win_len + (int(mz_bin < 0) * (0.5 * win_len))


def get_ref_pattern_as_spectra(ref_patterns):
    """
    Get the reference patterns as a list of spectra.
    :param ref_patterns: the reference patterns
    :return: a list of spectra
    """
    spec_list = []

    for index, row in ref_patterns.iterrows():

        if 150 <= (row['m'] + 1) / row['z'] <= 1700:
            last_contrib = row['last_contrib']

            pattern = row['pattern'][:1000]

            mz_bin = np.arange(1000)

            both = list(zip(mz_bin, pattern))

            both = list(filter(lambda x: x[1] >= 0.001, both))

            mz = [x[0] for x in both]
            i = [x[1] for x in both]

            first_peak = np.where(np.diff(mz) > 1)[0][0]
            max_i = np.argmax(i[:first_peak])
            mono_mz_max = mz[max_i]

            mz = (mz - mono_mz_max) / 100 + row['m'] / row['z']

            mz = np.round(np.array(mz), 2) + 1.0
            i = np.array([int(x) for x in i])

            spectum = MzSpectrum(None, -1, -1, mz, i)
            spec_list.append((row['m'], row['z'], spectum, last_contrib))

    return spec_list


def get_refspec_list(ref_patterns, win_len=5):
    """

    """

    d_list = []

    for spec in tqdm(ref_patterns, desc='creating reference patterns', ncols=100):
        m = spec[0]
        z = spec[1]
        sp = spec[2]
        lc = spec[3]
        peak_width = 0.04
        mz_mono = np.round((m + 1.001) / z, 2) + peak_width
        mz_bin = np.floor(mz_mono / win_len)
        ref_spec = ReferencePattern(mz_bin=mz_bin, mz_mono=mz_mono, charge=z, mz_last_contrib=lc,
                                    spectrum=sp, window_length=win_len)

        d_list.append((mz_bin, ref_spec))

    return d_list


def create_reference_dict(D):
    # create members
    keys = set([t[0] for t in D])
    ref_d = dict([(k, []) for k in keys])

    for b, p in D:
        ref_d[b].append(p)

    tmp_dict = {}
    # go over all keys
    for key, values in ref_d.items():

        # create charges
        r = np.arange(5) + 1

        # go over all charge states
        for c in r:

            # go over all reference pattern
            for ca in values:

                # append exactly one charge state per reference bin
                if c == ca.charge:
                    # if key already exists, append
                    if key in tmp_dict:
                        tmp_dict[key].append(ca)
                        break

                    # if key does not exist, create new list
                    else:
                        tmp_dict[key] = [ca]
                        break

    return tmp_dict

@numba.jit(nopython=True)
def normal_pdf(x: ArrayLike, mass: float, s: float = 0.001, inv_sqrt_2pi: float = 0.3989422804014327, normalize: bool = False):
    """
    :param inv_sqrt_2pi:
    :param x:
    :param mass:
    :param s:
    :return:
    """
    a = (x - mass) / s
    if normalize:
        return np.exp(-0.5 * np.power(a,2))
    else:
        return inv_sqrt_2pi / s * np.exp(-0.5 * np.power(a,2))


@numba.jit
def gaussian(x, μ=0, σ=1):
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


@numba.jit
def exp_distribution(x, λ=1):
    """
    Exponential function
    :param x:
    :param λ:
    :return:
    """
    if x > 0:
        return λ * np.exp(-λ * x)
    return 0


@numba.jit
def exp_gaussian(x, μ=-3, σ=1, λ=.25):
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


def preprocess_max_quant_evidence(exp: pd.DataFrame) -> pd.DataFrame:
    """
    select columns from evidence txt, rename to ionmob naming convention and transform to raw data rt in seconds
    Args:
        exp: a MaxQuant evidence dataframe from evidence.txt table
    Returns: cleaned evidence dataframe, columns renamed to ionmob naming convention
    """

    # select columns
    exp = exp[['m/z', 'Mass', 'Charge', 'Modified sequence', 'Retention time',
               'Retention length', 'Ion mobility index', 'Ion mobility length', '1/K0', '1/K0 length',
               'Number of isotopic peaks', 'Max intensity m/z 0', 'Intensity', 'Raw file', 'CCS']].rename(
        # rename columns to ionmob naming convention
        columns={'m/z': 'mz', 'Mass': 'mass',
                 'Charge': 'charge', 'Modified sequence': 'sequence', 'Retention time': 'rt',
                 'Retention length': 'rt_length', 'Ion mobility index': 'im', 'Ion mobility length': 'im_length',
                 '1/K0': 'inv_ion_mob', '1/K0 length': 'inv_ion_mob_length', 'CCS': 'ccs',
                 'Number of isotopic peaks': 'num_peaks', 'Max intensity m/z 0': 'mz_max_intensity',
                 'Intensity': 'intensity', 'Raw file': 'raw'}).dropna()

    # transform retention time from minutes to seconds as stored in tdf raw data
    exp['rt'] = exp.apply(lambda r: r['rt'] * 60, axis=1)
    exp['rt_length'] = exp.apply(lambda r: r['rt_length'] * 60, axis=1)
    exp['rt_start'] = exp.apply(lambda r: r['rt'] - r['rt_length'] / 2, axis=1)
    exp['rt_stop'] = exp.apply(lambda r: r['rt'] + r['rt_length'] / 2, axis=1)

    exp['im_start'] = exp.apply(lambda r: int(np.round(r['im'] - r['im_length'] / 2)), axis=1)
    exp['im_stop'] = exp.apply(lambda r: int(np.round(r['im'] + r['im_length'] / 2)), axis=1)

    exp['inv_ion_mob_start'] = exp.apply(lambda r: r['inv_ion_mob'] - r['inv_ion_mob_length'] / 2, axis=1)
    exp['inv_ion_mob_stop'] = exp.apply(lambda r: r['inv_ion_mob'] + r['inv_ion_mob_length'] / 2, axis=1)

    # remove duplicates
    exp = exp.drop_duplicates(['sequence', 'charge', 'rt', 'ccs'])

    return exp

def max_quant_to_proforma(s:str, old_annotation:bool=False):
    """
    Convert MaxQuant amino acid sequences (with PTM) to
    proForma format.

    :param s: amino acid sequence
    :type s: str
    :param old_annotation: Wether old annotation is used in `s`, defaults to False
    :type old_annotation: bool, optional
    """
    seq = s.strip("_")

    proforma_seq = ""
    if old_annotation:
        proforma_seq = (seq
                        .replace("(ox)", "[UNIMOD:35]")
                        .replace("(ac)", "[UNIMOD:1]")
        )
    else:
        proforma_seq = (seq
                        .replace("(Oxidation (M))", "[UNIMOD:35]")
                        .replace("(Phospho (STY))", "[UNIMOD:21]")
                        .replace("(Acetyl (Protein N-term))", "[UNIMOD:1]")
                        )
    return f"_{proforma_seq}_"

def preprocess_max_quant_sequence(s, old_annotation=False):
    """
    :param s:
    :param old_annotation:
    """

    seq = s[1:-1]

    is_acc = False

    if old_annotation:
        seq = seq.replace('(ox)', '$')

        if seq.find('(ac)') != -1:
            is_acc = True
            seq = seq.replace('(ac)', '')

    else:
        seq = seq.replace('(Oxidation (M))', '$')
        seq = seq.replace('(Phospho (STY))', '&')

        if seq.find('(Acetyl (Protein N-term))') != -1:
            is_acc = True
            seq = seq.replace('(Acetyl (Protein N-term))', '')

    # form list from string
    slist = list(seq)

    tmp_list = []

    for item in slist:
        if item == '$':
            tmp_list.append('<OX>')

        elif item == '&':
            tmp_list.append('<PH>')

        else:
            tmp_list.append(item)

    slist = tmp_list

    r_list = []

    for i, char in enumerate(slist):

        if char == '<OX>':
            C = slist[i - 1]
            C = C + '-<OX>'
            r_list = r_list[:-1]
            r_list.append(C)

        elif char == 'C':
            r_list.append('C-<CM>')

        elif char == '<PH>':
            C = slist[i - 1]
            C = C + '-<PH>'
            r_list = r_list[:-1]
            r_list.append(C)

        else:
            r_list.append(char)

    if is_acc:
        return ['<START>-<AC>'] + r_list + ['<END>']

    return ['<START>'] + r_list + ['<END>']

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

