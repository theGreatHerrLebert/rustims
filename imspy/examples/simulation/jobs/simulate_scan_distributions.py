import pandas as pd
from tqdm import tqdm

from imspy.simulation.utility import get_z_score_for_percentile, python_list_to_json_string, get_scans_numba, \
    accumulated_intensity_cdf_numba


def simulate_scan_distributions(
        ions: pd.DataFrame,
        scans: pd.DataFrame,
        z_score: float,
        std_im: float,
        im_cycle_length: float,
        verbose: bool = False
) -> pd.DataFrame:

    # distribution parameters
    z_score = get_z_score_for_percentile(target_score=z_score)
    im_dict = dict(zip(scans.scan, scans.mobility))

    mobility_np = scans.mobility.values
    scans_np = scans.scan.values

    im_scans = []
    im_contributions = []

    for _, row in tqdm(ions.iterrows(), total=ions.shape[0], desc='scan distribution', ncols=100):
        scan_occurrence, scan_abundance = [], []

        im_value = row.mobility_gru_predictor
        contributing_scans = get_scans_numba(im_value, mobility_np, scans_np, std_im, z_score)

        for scan in contributing_scans:
            im = im_dict[scan]
            start = im - im_cycle_length
            i = accumulated_intensity_cdf_numba(start, im, im_value, std_im)

            # TODO: ADD NOISE HERE AS WELL?

            scan_occurrence.append(scan)
            scan_abundance.append(i)

        im_scans.append(scan_occurrence)
        im_contributions.append(scan_abundance)

    if verbose:
        print("Saving scan distributions...")

    ions['scan_occurrence'] = [list(x) for x in im_scans]
    ions['scan_abundance'] = [list(x) for x in im_contributions]

    ions['scan_occurrence'] = ions['scan_occurrence'].apply(lambda x: python_list_to_json_string(x, as_float=False))
    ions['scan_abundance'] = ions['scan_abundance'].apply(python_list_to_json_string)

    return ions