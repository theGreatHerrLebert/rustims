from numba import jit
import numpy as np
from scipy.stats import norm
from numpy.typing import NDArray
import math
import importlib.resources as resources
from importlib.abc import Traversable


def get_native_dataset_path(ds_name: str = 'NATIVE.d') -> Traversable:
    """ Get the path to a pretrained model

    Args:
        ds_name: The name of the dataset to load

    Returns:
        The path to the pretrained model
    """
    return resources.files('imspy.simulation.resources').joinpath(ds_name)


@jit(nopython=True)
def generate_events(n, mean, min_val, max_val):
    generated_values = np.random.exponential(scale=mean, size=n * 10)
    filtered_values = np.empty(n, dtype=np.float32)
    count = 0

    for val in generated_values:
        if min_val <= val <= max_val:
            filtered_values[count] = float(int(val))
            count += 1
            if count == n:
                break

    return filtered_values[:count]


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
def calculate_number_frames(gradient_length: float = 90 * 60, rt_cycle_length: float = .108) -> int:
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
def calculate_mobility_spacing(mobility_min: float, mobility_max: float, num_scans=500) -> float:
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


@jit(nopython=True)
def get_frames_numba(rt_value, times_array, std_rt, z_score):
    """
    Get the frames that will be acquired for a given retention time value.
    Parameters
    ----------
    rt_value : float
        Retention time value
    times_array : NDArray
        Array of retention times
    std_rt : float
        Standard deviation of the retention time
    z_score : float
        Z-score of the normal distribution

    Returns
    -------
    NDArray
        Array of frame indices
    """
    rt_min, rt_max = calculate_bounds_numba(rt_value, std_rt, z_score)
    first_frame = np.argmin(np.abs(times_array - rt_min)) + 1
    last_frame = np.argmin(np.abs(times_array - rt_max)) + 1
    rt_frames = np.arange(first_frame, last_frame + 1)

    return rt_frames


@jit(nopython=True)
def get_scans_numba(im_value, ims_array, scans_array, std_im, z_score):
    """
    Get the scans that will be acquired for a given ion mobility value.
    """
    im_min, im_max = calculate_bounds_numba(im_value, std_im, z_score)
    im_start = np.argmin(np.abs(ims_array - im_max))
    im_end = np.argmin(np.abs(ims_array - im_min))

    scan_start = scans_array[im_start]
    scan_end = scans_array[im_end]

    # Generate scan indices in the correct order given the inverse relationship
    im_scans = np.arange(scan_start, scan_end + 1)
    return im_scans
