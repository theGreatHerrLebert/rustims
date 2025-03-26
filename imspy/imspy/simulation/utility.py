import argparse
import json
import math
import os

import pandas as pd
import toml
import numpy as np
import importlib.resources as resources

from typing import List, Tuple, Any, Dict
from numpy.typing import NDArray

from importlib.abc import Traversable

import numba
from numba import jit
from scipy.stats import norm

from imspy.chemistry.constants import MASS_WATER
from imspy.chemistry.amino_acids import AMINO_ACID_MASSES
from imspy.chemistry.utility import calculate_mz

import imspy_connector
ims = imspy_connector.py_chemistry


def get_fasta_file_paths(fasta_path) -> Dict[str, str]:
    """
    Check if the provided fasta path is a folder or file, if its a folder, check if it exists and return all fasta
    Args:
        fasta_path:  Path to the fasta file or folder containing fasta files

    Returns:
        List of fasta file paths
    """

    # check if provided fasta path is a folder or file, if its a folder, check if it exists
    if os.path.isdir(fasta_path):
        fastas = {f: os.path.join(fasta_path, f) for f in os.listdir(fasta_path) if f.endswith('.fasta')}

        # check if there are any fasta files in the folder
        if len(fastas) == 0:
            raise argparse.ArgumentTypeError(f"No fasta files found in folder: {fasta_path}")

    # if the fasta path is a file, check if it is a fasta file
    else:
        if not fasta_path.endswith('.fasta'):
            raise argparse.ArgumentTypeError(f"Invalid fasta file: {fasta_path}")

        fastas = {os.path.basename(fasta_path): fasta_path}

    return fastas


@jit(nopython=True)
def add_uniform_noise(rt_abu: np.ndarray, noise_level: float = 1.0):
    noise = np.random.uniform(0, noise_level, len(rt_abu))
    noise_relative = (noise * rt_abu)
    noised_signal = rt_abu + noise_relative
    return (noised_signal / np.sum(noised_signal)) * np.sum(rt_abu)


@jit(nopython=True)
def flatten_prosit_array(array):
    array_return = np.zeros(174)

    ptr = 0

    for c in range(3):
        i_y = array[:, 0, c]
        array_return[ptr:ptr + 29] = i_y
        ptr += 29

        i_b = array[:, 1, c]
        array_return[ptr:ptr + 29] = i_b
        ptr += 29

    return array_return


def sequences_to_all_ions(
        sequences: List[str],
        charges: List[int],
        intensities_flat: List[List[float]],
        normalized: bool = True,
        half_charge_one: bool = True,
        num_threads: int = 4) -> List[str]:
    """
    Simulate ion intensities for a list of peptide sequences, charges, and collision energies.
    Args:
        sequences: List of peptide sequences
        charges: List of peptide charges
        intensities_flat: List of intensities
        normalized: Whether to normalize the intensities
        half_charge_one: Whether to divide the intensity by 2 if the charge is 1
        num_threads: Number of threads to use for the calculation

    Returns:
        NDArray: Array of ion intensities
    """
    return ims.sequence_to_all_ions_par(sequences, charges, intensities_flat, normalized, half_charge_one, num_threads)


def sequence_to_all_ions(
        sequence: str,
        charge: int,
        intensities_flat: List[float],
        normalized: bool = True,
        half_charge_one: bool = True) -> List[str]:
    """
    Simulate ion intensities for a peptide sequence, charge, and collision energy.
    Args:
        sequence: Peptide sequence
        charge: Peptide charge
        intensities_flat: List of intensities
        normalized: Whether to normalize the intensities
        half_charge_one: Whether to divide the intensity by 2 if the charge is 1

    Returns:
        NDArray: Array of ion intensities
    """
    return ims.sequence_to_all_ions_ims(sequence, charge, intensities_flat, normalized, half_charge_one)


def get_acquisition_builder_resource_path(acquisition_mode: str = 'dia') -> Traversable:
    """ Get the path to a pretrained model

    Args:
        acquisition_mode: The name of the model to load

    Returns:
        The path to the pretrained model
    """
    acquisition_mode = acquisition_mode.lower()
    assert acquisition_mode in ['dia', 'midia', 'slice', 'synchro'], \
        f"acquisition_mode needs to be one of 'dia', 'midia', 'slice', 'synchro', was: {acquisition_mode}"

    return resources.files('imspy.simulation.resources.configs').joinpath(acquisition_mode + 'pasef.toml')


def get_dilution_factors(path: str = None) -> Dict[str, float]:

    if path is not None:
        table = pd.read_csv(path)
    else:
        table = pd.read_csv(str(resources.files('imspy.simulation.resources.configs').joinpath('dilution_factors.csv')))

    dilution_dict = {}

    for _, row in table.iterrows():
        dilution_dict[row["proteome"]] = row["dilution_factor"]

    return dilution_dict


def get_ms_ms_window_layout_resource_path(acquisition_mode: str) -> Traversable:
    """ Get the path to a pretrained model

    Returns:
        The path to the pretrained model
    """

    assert acquisition_mode in ['dia', 'midia', 'slice', 'synchro'], \
        f"acquisition_mode needs to be one of 'dia', 'midia', 'slice', 'synchro', was: {acquisition_mode}"

    return resources.files('imspy.simulation.resources.configs').joinpath(acquisition_mode + '_ms_ms_windows.csv')


def read_acquisition_config(acquisition_name: str = 'dia') -> Dict[str, Any]:

    file_path = get_acquisition_builder_resource_path(acquisition_name)

    with open(file_path, 'r') as config_file:
        config_data = toml.load(config_file)
    return config_data


# Function to convert a list (or a pandas series) to a JSON string
def python_list_to_json_string(lst, as_float=True, num_decimals: int = 4) -> str:
    if as_float:
        return json.dumps([float(np.round(x, num_decimals)) for x in lst])
    return json.dumps([int(x) for x in lst])


# load peptides and ions
def json_string_to_python_list(json_string):
    return json.loads(json_string)


def sequence_to_numpy(sequence: str, max_length: int = 30) -> NDArray:
    """
    translate a peptide sequence given as python string into a numpy array of characters with a fixed length
    Args:
        sequence: the peptide sequence
        max_length: the maximum length a sequence can have

    Returns:
        numpy array of characters
    """
    arr = np.full((1, max_length), fill_value="", dtype=object)
    for i, char in enumerate(sequence):
        arr[0, i] = char
    return np.squeeze(arr)


def calculate_b_y_fragment_mz(sequence: str, modifications: NDArray, is_y: bool = False, charge: int = 1) -> float:
    """
    Calculate the m/z value of a b or y fragment.
    Args:
        sequence: the peptide sequence
        modifications: potential modifications
        is_y: is the fragment a y ion
        charge: the charge state of the peptide precursor

    Returns:
        m/z value of the fragment
    """
    # return mz of empty sequence
    if len(sequence) == 0:
        return 0.0

    # add up raw amino acid masses and potential modifications
    mass = np.sum([AMINO_ACID_MASSES[s] for s in sequence]) + np.sum(modifications)

    # if sequence is n-terminal, add water mass and calculate mz
    if is_y:
        return calculate_mz(mass + MASS_WATER, charge)

    # otherwise, calculate mz
    return calculate_mz(mass, charge)


def calculate_b_y_ion_series_ims(sequence: str, modifications: NDArray, charge: int = 1) -> Tuple[List, List]:
    """
    Calculate the b and y ion series for a given peptide sequence.
    Args:
        sequence: the peptide sequence
        modifications: potential modifications
        charge: the charge state of the peptide precursor

    Returns:
        b ion series, y ion series
    """
    return ims.calculate_b_y_ion_series(sequence, modifications, charge)


def get_native_dataset_path(ds_name: str = 'NATIVE.d') -> str:
    """ Get the path to a pretrained model

    Args:
        ds_name: The name of the dataset to load

    Returns:
        The path to the pretrained model
    """
    return str(resources.files('imspy.simulation.resources').joinpath(ds_name))


@jit(nopython=True)
def generate_events(n, mean, min_val, max_val, mixture_contribution=1.0):
    generated_values = np.random.exponential(scale=mean, size=n * 10)
    filtered_values = np.empty(n, dtype=np.float32)
    count = 0

    for val in generated_values:
        if min_val <= val <= max_val:
            filtered_values[count] = float(int(val))
            count += 1
            if count == n:
                break

    return filtered_values[:count] * mixture_contribution


@jit(nopython=True)
def custom_cdf(x, mean, std_dev):
    """
    Custom implementation of the CDF for a normal distribution.
    """
    z = (x - mean) / std_dev
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


@jit(nopython=True)
def accumulated_intensity_cdf_numba(sample_start, sample_end, mean, std_dev):
    """
    Calculate the accumulated intensity between two points using the custom CDF.
    """
    cdf_start = custom_cdf(sample_start, mean, std_dev)
    cdf_end = custom_cdf(sample_end, mean, std_dev)
    return cdf_end - cdf_start


@jit(nopython=True)
def irt_to_rts_numba(irt: NDArray, new_min=0, new_max=120):
    """
    Scale an array of values from the original range (min_val, max_val) to a new range (new_min, new_max).

    Parameters
    ----------
    irt : NDArray
        Array of values to be scaled.
    new_min : float
        Minimum value of the new range.
    new_max : float
        Maximum value of the new range.

    Returns
    -------
    NDArray
        Array of scaled values.
    """
    min_val = np.min(irt)
    max_val = np.max(irt)
    scaled_values = new_min + (irt - min_val) * (new_max - new_min) / (max_val - min_val)
    return scaled_values


@jit(nopython=True)
def calculate_number_frames(gradient_length: float, rt_cycle_length: float) -> int:
    """ Calculate the number of frames that will be taken during the acquisition

    Parameters
    ----------
    gradient_length : float
        Length of the gradient in seconds
    rt_cycle_length : float
        Length of the RT cycle in seconds

    Returns
    -------
    int
        Number of frames that will be taken during the acquisition
    """
    return int(gradient_length / rt_cycle_length)


@jit(nopython=True)
def calculate_mobility_spacing(mobility_min: float, mobility_max: float, num_scans: int) -> float:
    """ Calculate the amount of mobility that will be occupied by a single scan

    Parameters
    ----------
    mobility_min : float
        Minimum mobility value
    mobility_max : float
        Maximum mobility value
    num_scans : int
        Number of scans that will be taken during the acquisition

    Returns
    -------
    float
        Mobility spacing
    """
    length = float(mobility_max) - float(mobility_min)
    return length / num_scans


def get_z_score_for_percentile(target_score=0.95):
    assert 0 <= target_score <= 1.0, f"target_score needs to be between 0 and 1, was: {target_score}"
    return norm.ppf(1 - (1 - target_score) / 2)


@jit(nopython=True)
def calculate_bounds_numba(mean, std, z_score):
    """
    Calculate the bounds of a normal distribution for a given z-score.

    Parameters
    ----------
    mean : float
        Mean of the normal distribution
    std : float
        Standard deviation of the normal distribution
    z_score : float
        Z-score of the normal distribution

    Returns
    -------
    float
        Lower bound of the normal distribution
    """
    return mean - z_score * std, mean + z_score * std

@numba.jit(cache=True, nopython=True)
def get_peak_cnts(total_scans, scans):
    peak_cnts = [total_scans]
    ii = 0
    for scan_id in range(1, total_scans):
        counter = 0
        while ii < len(scans) and scans[ii] < scan_id:
            ii += 1
            counter += 1
        peak_cnts.append(counter * 2)
    peak_cnts = np.array(peak_cnts, np.uint32)
    return peak_cnts


@numba.jit(cache=True, nopython=True)
def modify_tofs(tofs, scans):
    last_tof = -1
    last_scan = 0
    for ii in range(len(tofs)):
        if last_scan != scans[ii]:
            last_tof = -1
            last_scan = scans[ii]
        val = tofs[ii]
        tofs[ii] = val - last_tof
        last_tof = val


@numba.jit(nopython=True)
def np_zip(xx, yy):
    res = np.empty(2 * len(xx), dtype=xx.dtype)
    i = 0
    for x, y in zip(xx, yy):
        res[i] = x
        i += 1
        res[i] = y
        i += 1
    return res


@numba.njit
def get_realdata_loop(peak_cnts, interleaved, back_data, real_data):
    reminder = 0
    bd_idx = 0
    for rd_idx in range(len(back_data)):
        if bd_idx >= len(back_data):
            reminder += 1
            bd_idx = reminder
        real_data[rd_idx] = back_data[bd_idx]
        bd_idx += 4


def get_realdata(peak_cnts, interleaved):
    back_data = peak_cnts.tobytes() + interleaved.tobytes()
    real_data = bytearray(len(back_data))
    get_realdata_loop(peak_cnts, interleaved, back_data, real_data)
    return real_data


def get_compressible_data(tofs, scans, intensities, num_scans):
    peak_counts = get_peak_cnts(num_scans, scans)
    tofs = np.copy(tofs)
    modify_tofs(tofs, scans)
    interleaved = np_zip(tofs, intensities)
    return np.array(get_realdata(peak_counts, interleaved))


@jit(nopython=True)
def flat_intensity_to_sparse(intensity_flat: NDArray, num_elements: int = 174):
    flat_intensity = np.round(intensity_flat, 6)

    for i in range(num_elements):
        if flat_intensity[i] < 0:
            flat_intensity[i] = 0

    nonzero = np.count_nonzero(flat_intensity)
    indices = np.zeros(nonzero)
    values = np.zeros(nonzero)

    counter = 0

    for i in range(num_elements):
        value = intensity_flat[i]
        if value > 0:
            indices[counter] = i
            values[counter] = value
            counter += 1

    return indices.astype(np.int32), values


def set_percentage_to_zero(row, percentage):
    """
    Sets a given percentage of the non-zero elements of a numpy vector to zero,
    where the probability of being set to 0 is inversely proportional to the element's value
    relative to other values in the vector.

    Parameters:
        row (np.ndarray): Input vector of arbitrary length
        percentage (float): Percentage of non-zero elements to set to zero (between 0 and 1)

    Returns:
        np.ndarray: Modified vector with the specified percentage of non-zero elements set to zero
    """
    if percentage < 0 or percentage > 1:
        raise ValueError("Percentage must be between 0 and 1")

    if row.ndim != 1:
        raise ValueError("Input must be a 1D vector")

    # Copy the input row and replace -1 values with 0
    result = row.copy()
    result[result < 0] = 0

    # Calculate the total number of non-zero elements to be set to zero
    total_non_zero_elements = np.count_nonzero(result)
    num_elements_to_zero = int(total_non_zero_elements * percentage)

    if total_non_zero_elements == 0:
        return result  # No non-zero elements to set to zero

    # Create a probability array inversely proportional to the values in the row
    inverse_values = np.zeros_like(result, dtype=float)
    non_zero_mask = result != 0

    # Inverse of non-zero, non-negative values
    inverse_values[non_zero_mask] = 1.0 / (result[non_zero_mask] * 10_000.00)

    row_sum = inverse_values.sum()
    if row_sum == 0:
        return result  # No non-zero elements to set to zero

    probabilities = inverse_values / row_sum

    # Ensure all probabilities are non-negative
    if not np.all(probabilities >= 0):
        raise ValueError("Probabilities contain negative values")

    # Find non-zero elements
    non_zero_indices = np.nonzero(result)[0]

    # Choose indices to be zeroed based on the calculated probabilities
    chosen_indices = np.random.choice(non_zero_indices, num_elements_to_zero, replace=False,
                                      p=probabilities[non_zero_indices])

    # Set the chosen elements to zero
    result[chosen_indices] = 0

    return result
