import numpy as np
import pandas as pd
from imspy_connector import py_utility as ims

from imspy.simulation.utility import python_list_to_json_string, add_uniform_noise


def simulate_scan_distributions_with_variance(
        ions: pd.DataFrame,
        scans: pd.DataFrame,
        p_target: float = 0.9995,
        verbose: bool = False,
        add_noise: bool = False,
        num_threads: int = 16,
) -> pd.DataFrame:
    """Simulate scan distributions for ions.

    Args:
        ions: Ions DataFrame.
        scans: Scan DataFrame.
        p_target: Target percentile.
        verbose: Verbosity.
        add_noise: Add noise to the scan distributions.
        num_threads: Number of threads.

    Returns:
        pd.DataFrame: Ions DataFrame with scan distributions.
    """

    im_cycle_length = np.mean(np.diff(scans.mobility))

    # inv_mobility_gru_predictor and inv_mobility_gru_predictor_std need to be in ions, scan and mobility in scans
    assert "scan" in scans.columns, "scan column is missing"
    assert "mobility" in scans.columns, "mobility column is missing"
    assert "inv_mobility_gru_predictor" in ions.columns, "inv_mobility_gru_predictor column is missing"
    assert "inv_mobility_gru_predictor_std" in ions.columns, "inv_mobility_gru_predictor_std column is missing"

    if verbose:
        print("Calculating scan occurrences...")

    occurrence = ims.calculate_scan_occurrences_gaussian_par(
        times=scans.mobility,
        means=ions.inv_mobility_gru_predictor,
        sigmas=ions.inv_mobility_gru_predictor_std,
        target_p=p_target,
        step_size=0.0001,
        n_lower_start=5,
        n_upper_start=5,
        num_threads=num_threads
    )

    if verbose:
        print("Calculating scan abundances...")

    abundances = ims.calculate_scan_abundances_gaussian_par(
        indices=scans.scan,
        times=scans.mobility,
        occurrences=occurrence,
        means=ions.inv_mobility_gru_predictor,
        sigmas=ions.inv_mobility_gru_predictor_std,
        cycle_length=im_cycle_length,
        num_threads=num_threads,
    )

    if verbose:
        print("Serializing scan distributions to json...")

    if add_noise:
        # TODO: Make noise model configurable
        noise_levels = np.random.uniform(0.0, 2.0, len(abundances))
        abundances = [add_uniform_noise(np.array(abundance), noise_level) for abundance, noise_level in zip(abundances, noise_levels)]
        # Normalize the abundances
        abundances = [scan_abundance / np.sum(scan_abundance) for scan_abundance in abundances]

    ions['scan_occurrence'] = [list(x) for x in occurrence]
    ions['scan_abundance'] = [list(x) for x in abundances]

    ions['scan_occurrence'] = ions['scan_occurrence'].apply(lambda x: python_list_to_json_string(x, as_float=False))
    ions['scan_abundance'] = ions['scan_abundance'].apply(python_list_to_json_string)

    # remove rows where scan_abundance is empty
    ions = ions[ions['scan_abundance'].apply(lambda x: len(x) > 2)]

    ions = ions[['peptide_id', 'sequence', 'charge', 'mz', 'relative_abundance', 'inv_mobility_gru_predictor',
                 'inv_mobility_gru_predictor_std', 'simulated_spectrum', 'scan_occurrence', 'scan_abundance']]

    ion_id = ions.index
    ions.insert(0, 'ion_id', ion_id)

    return ions