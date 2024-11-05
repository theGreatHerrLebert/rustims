import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm

from imspy.simulation.utility import get_z_score_for_percentile, python_list_to_json_string, get_scans_numba, \
    accumulated_intensity_cdf_numba, add_uniform_noise


def simulate_scan_distributions(
        ions: pd.DataFrame,
        scans: pd.DataFrame,
        z_score: float,
        mean_std_im: float = 0.01,
        variance_std_im: float = 0.0,
        verbose: bool = False,
        add_noise: bool = False,
        normalize: bool = False,
        from_existing: bool = False,
        std_means: NDArray = None,
) -> pd.DataFrame:
    """Simulate scan distributions for ions.

    Args:
        ions: Ions DataFrame.
        scans: Scan DataFrame.
        z_score: Z-score.
        mean_std_im: Standard deviation of ion mobility.
        variance_std_im: Variance of standard deviation of ion mobility.
        verbose: Verbosity.
        add_noise: Add noise.
        normalize: Normalize scan abundance.
        from_existing: Use existing parameters.
        std_means: Standard deviations.

    Returns:
        pd.DataFrame: Ions DataFrame with scan distributions.
    """

    im_cycle_length = np.mean(np.diff(scans.mobility))

    # distribution parameters
    z_score = get_z_score_for_percentile(target_score=z_score)
    im_dict = dict(zip(scans.scan, scans.mobility))

    mobility_np = scans.mobility.values
    scans_np = scans.scan.values

    im_scans = []
    im_contributions = []

    # Generate random standard deviations for ion mobility, if not from_existing
    if not from_existing:
        std_im = np.random.normal(loc=mean_std_im, scale=np.sqrt(variance_std_im), size=ions.shape[0])
    else:
        std_im = std_means

    # Add standard deviation deviations to ions DataFrame
    ions['std_im'] = std_im

    if verbose:
        print("Calculating scan distributions...")

    for index, (_, row) in enumerate(tqdm(ions.iterrows(), total=ions.shape[0], desc='scan distribution', ncols=100)):
        scan_occurrence, scan_abundance = [], []

        im_value = row.inv_mobility_gru_predictor
        contributing_scans = get_scans_numba(im_value, mobility_np, scans_np, std_im[index], z_score)

        for scan in contributing_scans:
            im = im_dict[scan]
            start = im - im_cycle_length
            i = accumulated_intensity_cdf_numba(start, im, im_value, std_im[index])
            scan_occurrence.append(scan)
            scan_abundance.append(i)

        if add_noise:
            noise_level = np.random.uniform(0.0, 2.0)
            scan_abundance = add_uniform_noise(np.array(scan_abundance), noise_level)

        im_scans.append(scan_occurrence)

        if normalize:
            scan_abundance = scan_abundance / np.sum(scan_abundance)

        im_contributions.append(scan_abundance)

    if verbose:
        print("Serializing scan distributions to json...")

    ions['scan_occurrence'] = [list(x) for x in im_scans]
    ions['scan_abundance'] = [list(x) for x in im_contributions]

    ions['scan_occurrence'] = ions['scan_occurrence'].apply(lambda x: python_list_to_json_string(x, as_float=False))
    ions['scan_abundance'] = ions['scan_abundance'].apply(python_list_to_json_string)

    return ions
