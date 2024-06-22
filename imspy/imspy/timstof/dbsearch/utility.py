import os
import re
from typing import List, Tuple

import pandas as pd
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

from sagepy.utility import peptide_spectrum_match_list_to_pandas
from numpy.typing import NDArray

from sagepy.core.scoring import merge_psm_dicts


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


def generate_training_data(psms: List[PeptideSpectrumMatch], method: str = "psm", q_max: float = 0.01) -> Tuple[NDArray, NDArray]:
    """ Generate training data.
    Args:
        psms: List of PeptideSpectrumMatch objects
        method: Method to use for training data generation
        q_max: Maximum q-value allowed for positive examples

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
) -> List[PeptideSpectrumMatch]:
    """ Re-score PSMs using LDA.
    Args:
        psms: List of PeptideSpectrumMatch objects
        num_splits: Number of splits
        verbose: Whether to print progress

    Returns:
        List[PeptideSpectrumMatch]: List of PeptideSpectrumMatch objects
    """

    splits = split_psms(psms=psms, num_splits=num_splits)
    predictions = []

    for i in tqdm(range(num_splits), disable=not verbose, desc='Re-scoring PSMs', ncols=100):

        target = splits[i]
        features = []

        for j in range(num_splits):
            if j != i:
                features.extend(splits[j])

        X_train, Y_train = generate_training_data(features)
        X, _ = get_features(peptide_spectrum_match_list_to_pandas(target))

        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
        lda.fit(X_train, Y_train)

        score_flip = None
        try:
            # check for flip sign of LDA classification return to be compatible with good score ascending
            score_flip = 1.0 if Y_train[np.argmax(np.squeeze(lda.transform(X_train)))] == 1.0 else -1.0
        except:
            score_flip = 1.0

        Y_pred = np.squeeze(lda.transform(X)) * score_flip
        predictions.extend(Y_pred)

    for score, match in zip(predictions, psms):
        match.re_score = score

    return psms


def generate_balanced_rt_dataset(psms, num_bins=128, hits_per_bin=64, rt_min=0.0, rt_max=60.0):
    bin_width = (rt_max - rt_min) / (num_bins - 1)
    bins = [rt_min + i * bin_width for i in range(num_bins)]

    r_list = []

    for i in range(len(bins) - 1):
        rt_lower = bins[i]
        rt_upper = bins[i + 1]

        psm = list(sorted(key=filter(
            lambda match: rt_lower <= match.retention_time_observed <= rt_upper and match.decoy is False, psms)))

        # sort by hyper_score descending
        psm = sorted(psm, key=lambda x: x.hyper_score, reverse=True)[:hits_per_bin]
        r_list.extend(psm)

    return r_list


def generate_balanced_im_dataset(psms, min_charge=1, max_charge=4, hits_per_charge=2048):

    im_list = []

    for charge in range(min_charge, max_charge + 1):

        psm = list(filter(lambda match: match.charge == charge and match.decoy is False, psms))

        # sort by hyper_score descending
        psm = sorted(psm, key=lambda x: x.hyper_score, reverse=True)[:hits_per_charge]
        im_list.extend(psm)

    return im_list
