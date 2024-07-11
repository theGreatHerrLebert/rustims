import os
import re
from typing import List, Tuple

import numba
import pandas as pd
from sagepy.core.ion_series import IonType
from tqdm import tqdm

import numpy as np
from typing import Optional
from sagepy.core import Precursor, RawSpectrum, ProcessedSpectrum, SpectrumProcessor, Representation
from sagepy.core.scoring import PeptideSpectrumMatch, associate_fragment_ions_with_prosit_predicted_intensities

from imspy.algorithm.intensity.predictors import Prosit2023TimsTofWrapper
from imspy.timstof.frame import TimsFrame

from sagepy.utility import get_features
from sagepy.qfdr.tdc import target_decoy_competition_pandas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from sagepy.utility import peptide_spectrum_match_list_to_pandas
from numpy.typing import NDArray

from sagepy.core.scoring import merge_psm_dicts
from numba import jit


@jit(nopython=True)
def linear_map(value, old_min, old_max, new_min=0.0, new_max=60.0):
    scale = (new_max - new_min) / (old_max - old_min)
    offset = new_min - old_min * scale

    new_value = value * scale + offset
    return new_value


def merge_dicts_with_merge_dict(dicts):
    d = None
    for i, item in enumerate(dicts):
        if i == 0:
            d = item
        else:
            d = merge_psm_dicts(item, d)

    return d


def map_to_domain(data, gradient_length: float = 120.0):
    """
    Maps the input data linearly into the domain [0, l].

    Parameters:
    - data: list or numpy array of numerical values
    - l: float, the upper limit of the target domain [0, l]

    Returns:
    - mapped_data: list of values mapped into the domain [0, l]
    """
    min_val = min(data)
    max_val = max(data)

    if max_val == min_val:
        raise ValueError("All elements in data are the same. Linear mapping is not possible.")

    mapped_data = [(gradient_length * (x - min_val) / (max_val - min_val)) for x in data]

    return mapped_data


def sanitize_charge(charge: Optional[float]) -> Optional[int]:
    """Sanitize charge value.
    Args:
        charge: Charge value as float.

    Returns:
        int: Charge value as int.
    """
    if charge is not None and not np.isnan(charge):
        return int(charge)
    return None


def sanitize_mz(mz: Optional[float], mz_highest: float) -> Optional[float]:
    """Sanitize mz value.
    Args:
        mz: Mz value as float.
        mz_highest: Highest mz value.

    Returns:
        float: Mz value as float.
    """
    if mz is not None and not np.isnan(mz):
        return mz
    return mz_highest


def split_fasta(fasta: str, num_splits: int = 16, randomize: bool = True) -> List[str]:
    """ Split a fasta file into multiple fasta files.
    Args:
        fasta: Fasta file as string.
        num_splits: Number of splits fasta file should be split into.
        randomize: Whether to randomize the order of sequences before splitting.

    Returns:
        List of fasta files as strings, will contain num_splits fasta files with equal number of sequences.
    """

    if num_splits == 1:
        return [fasta]

    split_strings = re.split(r'\n>', fasta)

    print(f"Total number of sequences: {len(split_strings)} ...")

    if randomize:
        np.random.shuffle(split_strings)

    if not split_strings[0].startswith('>'):
        split_strings[0] = '>' + split_strings[0]

    total_items = len(split_strings)
    items_per_batch = total_items // num_splits
    remainder = total_items % num_splits

    fastas = []
    start_index = 0

    for i in range(num_splits):
        extra = 1 if i < remainder else 0
        stop_index = start_index + items_per_batch + extra

        if start_index >= total_items:
            break

        batch = '\n>'.join(split_strings[start_index:stop_index])

        if not batch.startswith('>'):
            batch = '>' + batch

        fastas.append(batch)
        start_index = stop_index

    return fastas


def get_searchable_spec(precursor: Precursor,
                        raw_fragment_data: TimsFrame,
                        spec_processor: SpectrumProcessor,
                        time: float,
                        spec_id: str,
                        file_id: int = 0,
                        ms_level: int = 2) -> ProcessedSpectrum:
    """
    Get SAGE searchable spectrum from raw data.
    Args:
        precursor: Precursor object
        raw_fragment_data: TimsFrame object
        time: float
        spec_processor: SpectrumProcessor object
        spec_id: str
        file_id: int
        ms_level: int

    Returns:
        ProcessedSpectrum: ProcessedSpectrum object
    """

    flat_spec = raw_fragment_data.to_indexed_mz_spectrum()

    spec = RawSpectrum(
            file_id=file_id,
            ms_level=ms_level,
            spec_id=spec_id,
            representation=Representation(),
            precursors=[precursor],
            scan_start_time=time,
            ion_injection_time=time,
            total_ion_current=np.sum(flat_spec.intensity),
            mz=flat_spec.mz.astype(np.float32),
            intensity=flat_spec.intensity.astype(np.float32)
        )

    processed_spec = spec_processor.process(spec)
    return processed_spec


def get_collision_energy_calibration_factor(
        sample: List[PeptideSpectrumMatch],
        model: Prosit2023TimsTofWrapper,
        lower: int = -30,
        upper: int = 30,
        verbose: bool = False,
) -> Tuple[float, List[float]]:
    """
    Get the collision energy calibration factor for a given sample.
    Args:
        sample: a list of PeptideSpectrumMatch objects
        model: a Prosit2023TimsTofWrapper object
        lower: lower bound for the search
        upper: upper bound for the search
        verbose: whether to print progress

    Returns:
        Tuple[float, List[float]]: the collision energy calibration factor and the cosine similarities
    """
    cos_target, cos_decoy = [], []

    if verbose:
        print(f"Searching for collision energy calibration factor between {lower} and {upper} ...")

    for i in tqdm(range(lower, upper), disable=not verbose, desc='calibrating CE', ncols=100):
        I = model.predict_intensities(
            [p.sequence for p in sample],
            np.array([p.charge for p in sample]),
            [p.collision_energy + i for p in sample],
            batch_size=2048,
            flatten=True
        )

        psm_i = associate_fragment_ions_with_prosit_predicted_intensities(sample, I)
        target = list(filter(lambda x: not x.decoy, psm_i))
        decoy = list(filter(lambda x: x.decoy, psm_i))

        cos_target.append((i, np.mean([x.cosine_similarity for x in target])))
        cos_decoy.append((i, np.mean([x.cosine_similarity for x in decoy])))

    return cos_target[np.argmax([x[1] for x in cos_target])][0], [x[1] for x in cos_target]


def write_psms_binary(byte_array, folder_path: str, file_name: str, total: bool = False):
    """ Write PSMs to binary file.
    Args:
        byte_array: Byte array
        folder_path: Folder path
        file_name: File name
        total: Whether to write to total folder
    """
    # create folder if it doesn't exist
    if not os.path.exists(f'{folder_path}/imspy/psm'):
        os.makedirs(f'{folder_path}/imspy/psm')

    # remove existing file, if any
    if os.path.exists(f'{folder_path}/imspy/psm/{file_name}.bin'):
        os.remove(f'{folder_path}/imspy/psm/{file_name}.bin')

    if not total:
        file = open(f'{folder_path}/imspy/psm/{file_name}.bin', 'wb')
    else:
        file = open(f'{folder_path}/imspy/{file_name}.bin', 'wb')
    try:
        file.write(bytearray(byte_array))
    finally:
        file.close()


def generate_training_data(psms: List[PeptideSpectrumMatch], method: str = "psm", q_max: float = 0.01,
                           balance: bool = True) -> Tuple[NDArray, NDArray]:
    """ Generate training data.
    Args:
        psms: List of PeptideSpectrumMatch objects
        method: Method to use for training data generation
        q_max: Maximum q-value allowed for positive examples
        balance: Whether to balance the dataset

    Returns:
        Tuple[NDArray, NDArray]: X_train and Y_train
    """
    # create pandas table from psms
    PSM_pandas = peptide_spectrum_match_list_to_pandas(psms)

    # calculate q-values to get inital "good" hits
    PSM_q = target_decoy_competition_pandas(PSM_pandas, method=method)
    PSM_pandas = PSM_pandas.drop(columns=["q_value", "score"])

    # merge data with q-values
    TDC = pd.merge(PSM_q, PSM_pandas, left_on=["spec_idx", "match_idx", "decoy"],
                   right_on=["spec_idx", "match_idx", "decoy"])

    # select best positive examples
    TARGET = TDC[(TDC.decoy == False) & (TDC.q_value <= q_max)]
    X_target, Y_target = get_features(TARGET)

    # select all decoys
    DECOY = TDC[TDC.decoy]
    X_decoy, Y_decoy = get_features(DECOY)

    # balance the dataset such that the number of target and decoy examples are equal
    if balance:
        num_target = np.min((len(DECOY), len(TARGET)))
        target_indices = np.random.choice(np.arange(len(X_target)), size=num_target)
        X_target = X_target[target_indices, :]
        Y_target = Y_target[target_indices]

    X_train = np.vstack((X_target, X_decoy))
    Y_train = np.hstack((Y_target, Y_decoy))

    return X_train, Y_train


def split_psms(psms: List[PeptideSpectrumMatch], num_splits: int = 10) -> List[List[PeptideSpectrumMatch]]:
    """ Split PSMs into multiple splits.

    Args:
        psms: List of PeptideSpectrumMatch objects
        num_splits: Number of splits

    Returns:
        List[List[PeptideSpectrumMatch]]: List of splits
    """
    # floor devision by num_splits
    split_size = len(psms) // num_splits
    # remainder for last split
    remainder = len(psms) % num_splits
    splits = []
    start_index = 0
    for i in range(num_splits):
        end_index = start_index + split_size + (1 if i < remainder else 0)
        splits.append(psms[start_index:end_index])
        start_index = end_index

    return splits


def re_score_psms(
        psms: List[PeptideSpectrumMatch],
        num_splits: int = 10,
        verbose: bool = True,
        balance: bool = True,
        score: str = "hyper_score",
) -> List[PeptideSpectrumMatch]:
    """ Re-score PSMs using LDA.
    Args:
        psms: List of PeptideSpectrumMatch objects
        num_splits: Number of splits
        verbose: Whether to print progress
        balance: Whether to balance the dataset
        score: Score to use for re-scoring

    Returns:
        List[PeptideSpectrumMatch]: List of PeptideSpectrumMatch objects
    """

    scaler = StandardScaler()
    X_all, _ = get_features(peptide_spectrum_match_list_to_pandas(psms), score=score)
    scaler.fit(X_all)

    splits = split_psms(psms=psms, num_splits=num_splits)
    predictions = []

    for i in tqdm(range(num_splits), disable=not verbose, desc='Re-scoring PSMs', ncols=100):

        target = splits[i]
        features = []

        for j in range(num_splits):
            if j != i:
                features.extend(splits[j])

        X_train, Y_train = generate_training_data(features, balance=balance)
        X, _ = get_features(peptide_spectrum_match_list_to_pandas(target))

        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
        lda.fit(scaler.transform(X_train), Y_train)

        try:
            # check for flip sign of LDA classification return to be compatible with good score ascending
            score_flip = 1.0 if Y_train[np.argmax(np.squeeze(lda.transform(scaler.transform(X_train))))] == 1.0 else -1.0
        except:
            score_flip = 1.0

        Y_pred = np.squeeze(lda.transform(scaler.transform(X))) * score_flip
        predictions.extend(Y_pred)

    for score, match in zip(predictions, psms):
        match.re_score = score

    return psms


def generate_balanced_rt_dataset(psms):
    # generate good hits
    PSM_pandas = peptide_spectrum_match_list_to_pandas(psms, re_score=False)
    PSM_q = target_decoy_competition_pandas(PSM_pandas, method="psm")
    PSM_pandas_dropped = PSM_pandas.drop(columns=["q_value", "score"])

    # merge data with q-values
    TDC = pd.merge(PSM_q, PSM_pandas_dropped, left_on=["spec_idx", "match_idx", "decoy"],
                   right_on=["spec_idx", "match_idx", "decoy"])
    TDC = TDC[(TDC.decoy == False) & (TDC.q_value <= 0.01)].drop_duplicates(subset="sequence")

    id_set = set(TDC.spec_idx.values)

    r_list = list(filter(lambda p: p.spec_idx in id_set and p.rank == 1, psms))

    return r_list


def generate_balanced_im_dataset(psms):
    # generate good hits
    PSM_pandas = peptide_spectrum_match_list_to_pandas(psms, re_score=False)
    PSM_q = target_decoy_competition_pandas(PSM_pandas, method="psm")
    PSM_pandas_dropped = PSM_pandas.drop(columns=["q_value", "score"])

    # merge data with q-values
    TDC = pd.merge(PSM_q, PSM_pandas_dropped, left_on=["spec_idx", "match_idx", "decoy"],
                   right_on=["spec_idx", "match_idx", "decoy"])
    TDC = TDC[(TDC.decoy == False) & (TDC.q_value <= 0.01)].drop_duplicates(subset=["sequence", "charge"])
    id_set = set(TDC.spec_idx.values)

    im_list = list(filter(lambda p: p.spec_idx in id_set and p.rank == 1, psms))

    return im_list


@numba.njit
def log_factorial(n: int, k: int) -> float:
    k = max(k, 2)
    result = 0.0
    for i in range(n, k - 1, -1):
        result += np.log(i)
    return result


def beta_score(fragments_observed, fragments_predicted) -> float:

    intensity = np.dot(fragments_observed.intensities, fragments_predicted.intensities)

    len_b, len_y = 0, 0

    b_type = IonType("b")
    y_type = IonType("y")

    for t in fragments_observed.ion_types:
        if t == b_type:
            len_b += 1
        elif t == y_type:
            len_y += 1

    i_min = min(len_b, len_y)
    i_max = max(len_b, len_y)

    return np.log1p(intensity) + 2.0 * log_factorial(int(i_min), 2) + log_factorial(int(i_max), int(i_min) + 1)
