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

    for i in tqdm(range(lower, upper), disable=verbose):
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


def write_psms_binary(byte_array, folder_path: str, file_name: str):
    """ Write PSMs to binary file.
    Args:
        byte_array: Byte array
        folder_path: Folder path
        file_name: File name
    """
    # create folder if it doesn't exist
    if not os.path.exists(f'{folder_path}/imspy/psm'):
        os.makedirs(f'{folder_path}/imspy/psm')

    file = open(f'{folder_path}/imspy/psm/{file_name}.bin', 'wb')
    try:
        file.write(bytearray(byte_array))
    finally:
        file.close()


def generate_training_data(psms: List[PeptideSpectrumMatch]) -> Tuple[NDArray, NDArray]:
    """ Generate training data.
    Args:
        psms: List of PeptideSpectrumMatch objects

    Returns:
        Tuple[NDArray, NDArray]: X_train and Y_train
    """

    # create pandas table from psms
    PSM_pandas = peptide_spectrum_match_list_to_pandas(psms)

    # calculate q-values to get inital "good" hits
    PSM_q = target_decoy_competition_pandas(PSM_pandas)
    PSM_pandas = PSM_pandas.drop(columns=["q_value", "score"])

    # merge data with q-values
    TDC = pd.merge(PSM_q, PSM_pandas, left_on=["spec_idx", "match_idx", "decoy"],
                   right_on=["spec_idx", "match_idx", "decoy"])

    # select best positive examples
    TARGET = TDC[(TDC.decoy == False) & (TDC.q_value <= 0.001)]
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


def re_score_psms(psms: List[PeptideSpectrumMatch], num_splits: int = 10,
                  verbose: bool = True) -> List[PeptideSpectrumMatch]:
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

    for i in tqdm(range(num_splits), disable=not verbose):

        target = splits[i]
        features = []

        for j in range(num_splits):
            if j != i:
                features.extend(splits[j])

        X_train, Y_train = generate_training_data(features)
        X, _ = get_features(peptide_spectrum_match_list_to_pandas(target))

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, Y_train)
        Y_pred = np.squeeze(lda.transform(X))
        predictions.extend(Y_pred)

    for score, match in zip(predictions, psms):
        match.re_score = score

    return psms
