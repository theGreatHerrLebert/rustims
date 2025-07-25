import os
import warnings
from typing import List, Tuple, Union, Dict

import pandas as pd

import numpy as np
from typing import Optional
from sagepy.core import Precursor, RawSpectrum, ProcessedSpectrum, SpectrumProcessor, Representation, Tolerance, \
    ProcessedIMSpectrum
from sagepy.core.scoring import Psm

from imspy.timstof import TimsDatasetDDA
from imspy.timstof.frame import TimsFrame

from sagepy.utility import get_features
from sagepy.qfdr.tdc import target_decoy_competition_pandas

from sagepy.utility import psm_collection_to_pandas
from numpy.typing import NDArray

from sagepy.core.scoring import merge_psm_dicts
from numba import jit

import ast
import re

def check_memory(
        limit_in_gb: int = 16,
        msg: str = "Warning: System has only {total_ram_gb:.2f}GB of RAM, which is below the recommended {limit_in_gb}GB."):

    if hasattr(os, "sysconf"):
        total_ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        total_ram_gb = total_ram_bytes / (1024 ** 3)  # Convert to GB
        if total_ram_gb < limit_in_gb:
            msg = msg.format(total_ram_gb=total_ram_gb, limit_in_gb=limit_in_gb)
            warnings.warn(msg)
    else:
        warnings.warn("Unable to determine system memory.")

# Run the check
check_memory()

def peptide_length(peptide: str):
    """
    Takes a peptide sequence as a string and returns its length,
    excluding [UNIMOD:X] modifications.

    Parameters:
        peptide (str): A peptide sequence with possible UNIMOD modifications.

    Returns:
        int: The length of the peptide without modifications.
    """
    cleaned_peptide = re.sub(r'\[UNIMOD:\d+\]', '', peptide)
    return len(cleaned_peptide)

def parse_string_list(input_str: str):
    """
    Takes a string representation of a list and converts it into an actual list of strings.

    Parameters:
        input_str (str): A string containing a list representation.

    Returns:
        list: A list of strings parsed from the input string.
    """
    if isinstance(input_str, list):
        return input_str

    try:
        return ast.literal_eval(input_str)
    except (SyntaxError, ValueError):
        raise ValueError("Invalid list format")


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

def get_ms1_ims_spectrum(
    raw_spectrum: TimsFrame,
    spec_processor: SpectrumProcessor,
    time: float,
    spec_id: str,
    file_id: int = 0,
    ms_level: int = 1) -> ProcessedIMSpectrum:
    """
    Get SAGE searchable spectrum from raw data.
    Args:
        raw_spectrum: TimsFrame object
        time: float
        spec_processor: SpectrumProcessor object
        spec_id: str
        file_id: int
        ms_level: int

    Returns:
        ProcessedSpectrum: ProcessedSpectrum object
    """

    spec = RawSpectrum(
        file_id=file_id,
        ms_level=ms_level,
        spec_id=spec_id,
        precursors=[],
        representation=Representation(),
        scan_start_time=time,
        ion_injection_time=time,
        total_ion_current=np.sum(raw_spectrum.intensity),
        mz=raw_spectrum.mz.astype(np.float32),
        intensity=raw_spectrum.intensity.astype(np.float32),
        mobility=raw_spectrum.mobility.astype(np.float32),
    )

    return spec_processor.process_with_mobility(spec)


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


def generate_training_data(psms: List[Psm], method: str = "psm", q_max: float = 0.01,
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
    PSM_pandas = psm_collection_to_pandas(psms)

    # calculate q-values to get inital "good" hits
    PSM_q = target_decoy_competition_pandas(PSM_pandas, method=method)
    PSM_pandas = PSM_pandas.drop(columns=["hyperscore"])

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


def split_psms(psms: List[Psm], num_splits: int = 10) -> List[List[Psm]]:
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


def generate_balanced_rt_dataset(psms: Union[List[Psm], Dict[str, List[Psm]]]) -> List[Psm]:

    psm_list = []
    if isinstance(psms, dict):
        for key in psms:
            psm_list.extend(psms[key])
    else:
        psm_list = psms

    # generate good hits
    PSM_pandas = psm_collection_to_pandas(psm_list)
    PSM_q = target_decoy_competition_pandas(PSM_pandas, method="psm", score="hyperscore")
    PSM_pandas_dropped = PSM_pandas.drop(columns=["hyperscore"])

    # merge data with q-values
    TDC = pd.merge(PSM_q, PSM_pandas_dropped, left_on=["spec_idx", "match_idx", "decoy"],
                   right_on=["spec_idx", "match_idx", "decoy"])
    TDC = TDC[(TDC.decoy == False) & (TDC.q_value <= 0.01)].drop_duplicates(subset="sequence")

    id_set = set(TDC.spec_idx.values)

    r_list = list(filter(lambda p: p.spec_idx in id_set and p.rank == 1, psm_list))

    return r_list


def generate_balanced_im_dataset(psms: Union[List[Psm], Dict[str, List[Psm]]]) -> List[Psm]:

    psm_list = []
    if isinstance(psms, dict):
        for key in psms:
            psm_list.extend(psms[key])

    else:
        psm_list = psms

    # generate good hits
    PSM_pandas = psm_collection_to_pandas(psm_list)
    PSM_q = target_decoy_competition_pandas(PSM_pandas, method="psm", score="hyperscore")
    PSM_pandas_dropped = PSM_pandas.drop(columns=["hyperscore"])

    # merge data with q-values
    TDC = pd.merge(PSM_q, PSM_pandas_dropped, left_on=["spec_idx", "match_idx", "decoy"],
                   right_on=["spec_idx", "match_idx", "decoy"])
    TDC = TDC[(TDC.decoy == False) & (TDC.q_value <= 0.01)].drop_duplicates(subset=["sequence", "charge"])
    id_set = set(TDC.spec_idx.values)

    im_list = list(filter(lambda p: p.spec_idx in id_set and p.rank == 1, psm_list))

    return im_list


def extract_timstof_dda_data(path: str,
                             in_memory: bool = False,
                             use_bruker_sdk: bool = False,
                             isolation_window_lower: float = -3.0,
                             isolation_window_upper: float = 3.0,
                             take_top_n: int = 100,
                             num_threads: int = 16,
                             ) -> pd.DataFrame:
    """
    Extract TIMSTOF DDA data from bruker timsTOF TDF file.
    Args:
        path: Path to TIMSTOF DDA data
        in_memory: Whether to load data in memory
        use_bruker_sdk: Whether to use bruker SDK for data extraction
        isolation_window_lower: Lower bound for isolation window (Da)
        isolation_window_upper: Upper bound for isolation window (Da)
        take_top_n: Number of top peaks to take
        num_threads: Number of threads to use

    Returns:
        pd.DataFrame: DataFrame containing timsTOF DDA data
    """
    ds_name = os.path.basename(path)

    dataset = TimsDatasetDDA(path, in_memory=in_memory, use_bruker_sdk=use_bruker_sdk)
    fragments = dataset.get_pasef_fragments(num_threads=num_threads)

    fragments = fragments.groupby('precursor_id').agg({
        'frame_id': 'first',
        'time': 'first',
        'precursor_id': 'first',
        'raw_data': 'sum',
        'scan_begin': 'first',
        'scan_end': 'first',
        'isolation_mz': 'first',
        'isolation_width': 'first',
        'collision_energy': 'first',
        'largest_peak_mz': 'first',
        'average_mz': 'first',
        'monoisotopic_mz': 'first',
        'charge': 'first',
        'average_scan': 'first',
        'intensity': 'first',
        'parent_id': 'first',
    })

    mobility = fragments.apply(lambda r: r.raw_data.get_inverse_mobility_along_scan_marginal(), axis=1)
    fragments['mobility'] = mobility

    # generate random string for for spec_id
    spec_id = fragments.apply(lambda r: str(r['frame_id']) + '-' + str(r['precursor_id']) + '-' + ds_name, axis=1)
    fragments['spec_id'] = spec_id

    sage_precursor = fragments.apply(lambda r: Precursor(
        mz=sanitize_mz(r['monoisotopic_mz'], r['largest_peak_mz']),
        intensity=r['intensity'],
        charge=sanitize_charge(r['charge']),
        isolation_window=Tolerance(da=(isolation_window_lower, isolation_window_upper)),
        collision_energy=r.collision_energy,
        inverse_ion_mobility=r.mobility,
    ), axis=1)

    fragments['sage_precursor'] = sage_precursor

    processed_spec = fragments.apply(
        lambda r: get_searchable_spec(
            precursor=r.sage_precursor,
            raw_fragment_data=r.raw_data,
            spec_processor=SpectrumProcessor(take_top_n=take_top_n),
            spec_id=r.spec_id,
            time=r['time'],
        ),
        axis=1
    )

    fragments['processed_spec'] = processed_spec

    return fragments


def transform_psm_to_pin(psm_df):
    columns_map = {
        'spec_idx': 'SpecId',
        'decoy': 'Label',
        'charge': 'Charge',
        'sequence': 'Peptide',
        # feature mapping for re-scoring
        'hyperscore': 'Feature1',
        'isotope_error': 'Feature2',
        'delta_mass': 'Feature3',
        'delta_rt': 'Feature4',
        'delta_ims': 'Feature5',
        'matched_peaks': 'Feature6',
        'matched_intensity_pct': 'Feature7',
        'intensity_ms1': 'Feature8',
        'intensity_ms2': 'Feature9',
        'average_ppm': 'Feature1ß',
        'poisson': 'Feature11',
        'spectral_entropy_similarity': 'Feature12',
        'spectral_correlation_similarity_pearson': 'Feature13',
        'spectral_correlation_similarity_spearman': 'Feature14',
        'spectral_normalized_intensity_difference': 'Feature15',
        'collision_energy': 'Feature16',
        'delta_next': 'Feature17',
        'delta_best': 'Feature18',
        'longest_b': 'Feature19',
        'longest_y': 'Feature20',
        'longest_y_pct': 'Feature21',
    }

    psm_df = psm_df[list(columns_map.keys())]
    df_pin = psm_df.rename(columns=columns_map)
    df_pin_clean = df_pin.dropna(axis=1, how='all')
    df_pin_clean = df_pin_clean.dropna()

    df_pin_clean['Label'] = df_pin_clean['Label'].apply(lambda x: -1 if x else 1)
    df_pin_clean['ScanNr'] = range(1, len(df_pin_clean) + 1)

    return df_pin_clean

full_renaming_scheme = {
    'spec_idx': 'psm_id',
    'sequence_modified': 'peptide',
    'ims': 'ion_mobility',
    'predicted_ims': 'predicted_mobility',
    'delta_ims': 'delta_mobility',
    'discriminant_score': 'sage_discriminant_score',
    'delta_mass': 'precursor_ppm',
    'average_ppm': 'fragment_ppm'
}

sage_target_columns = ['psm_id', 'peptide', 'proteins', 'num_proteins', 'filename', 'scannr',
       'rank', 'label', 'expmass', 'calcmass', 'charge', 'peptide_len',
       'missed_cleavages', 'semi_enzymatic', 'isotope_error', 'precursor_ppm',
       'fragment_ppm', 'hyperscore', 'delta_next', 'delta_best', 'rt',
       'aligned_rt', 'predicted_rt', 'delta_rt_model', 'ion_mobility',
       'predicted_mobility', 'delta_mobility', 'matched_peaks', 'longest_b',
       'longest_y', 'longest_y_pct', 'matched_intensity_pct',
       'scored_candidates', 'poisson', 'sage_discriminant_score',
       'posterior_error', 'spectrum_q', 'peptide_q', 'protein_q',
       'ms2_intensity']

def list_to_semicolon_string(value):
    """Converts a list of proteins into a semicolon-separated string."""
    if isinstance(value, list):
        return ";".join(value)
    return value

def parse_to_tims2rescore(TDC, from_mgf: bool = False, file_name: str = None):
    TDC_tmp = TDC.copy()
    TDC_tmp["filename"] = file_name if from_mgf else TDC_tmp.spec_idx.apply(lambda s: '-'.join(s.split('-')[3:]) + ".d")
    TDC_tmp["scannr"] = TDC_tmp.spec_idx.apply(lambda i: int(i.split("-")[1]) - 1) if from_mgf else TDC_tmp.spec_idx.apply(lambda s: int(s.split('-')[2]) - 1)
    TDC_tmp["num_proteins"] = TDC_tmp.proteins.apply(lambda protein: len(parse_string_list(protein)))
    TDC_tmp["label"] = TDC_tmp.decoy.apply(lambda b: - 1 if b else 1)
    TDC_tmp["peptide_len"] = TDC_tmp.sequence.apply(peptide_length)
    TDC_tmp["semi_enzymatic"] = False
    TDC_tmp = TDC_tmp.rename(columns=full_renaming_scheme)
    TDC_tmp = TDC_tmp[sage_target_columns]
    TDC_tmp["rank"] = TDC_tmp["rank"].astype(int)
    TDC_tmp["charge"] = TDC_tmp["charge"].astype(int)
    TDC_tmp["missed_cleavages"] = TDC_tmp["missed_cleavages"].astype(int)
    TDC_tmp["semi_enzymatic"] = TDC_tmp["semi_enzymatic"].astype(int)
    TDC_tmp["scored_candidates"] = TDC_tmp["scored_candidates"].astype(int)
    TDC_tmp["matched_peaks"] = TDC_tmp["matched_peaks"].astype(int)
    TDC_tmp["longest_b"] = TDC_tmp["longest_b"].astype(int)
    TDC_tmp["longest_y"] = TDC_tmp["longest_y"].astype(int)
    TDC_tmp["proteins"] = TDC_tmp.proteins.apply(list_to_semicolon_string)
    TDC_tmp["proteins"] = TDC_tmp["proteins"].astype(str)

    # sort by spectrum_q
    TDC_tmp = TDC_tmp.sort_values(by="spectrum_q", ascending=True)

    # set PSM id from 1 to n
    TDC_tmp["psm_id"] = range(1, len(TDC_tmp) + 1)

    def add_rev_prefix(protein, label):
        if label == -1:
            return f"rev_{protein}"
        return protein

    TDC_tmp["proteins"] = TDC_tmp.apply(lambda row: add_rev_prefix(row["proteins"], row["label"]), axis=1)

    return TDC_tmp


def get_ms1_ims_spectrum(
    raw_spectrum: TimsFrame,
    spec_processor: SpectrumProcessor,
    time: float,
    spec_id: str,
    file_id: int = 0,
    ms_level: int = 1) -> ProcessedIMSpectrum:
    """
    Get SAGE searchable spectrum from raw data.
    Args:
        raw_spectrum: TimsFrame object
        time: float
        spec_processor: SpectrumProcessor object
        spec_id: str
        file_id: int
        ms_level: int
    Returns:
        ProcessedSpectrum: ProcessedSpectrum object
    """
    spec = RawSpectrum(
        file_id=file_id,
        ms_level=ms_level,
        spec_id=spec_id,
        precursors=[],
        representation=Representation(),
        scan_start_time=time,
        ion_injection_time=time,
        total_ion_current=np.sum(raw_spectrum.intensity),
        mz=raw_spectrum.mz.astype(np.float32),
        intensity=raw_spectrum.intensity.astype(np.float32),
        mobility=raw_spectrum.mobility.astype(np.float32),
    )
    return spec_processor.process_with_mobility(spec)
