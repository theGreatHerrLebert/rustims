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

    # Check required columns
    assert "scan" in scans.columns, "scan column is missing"
    assert "mobility" in scans.columns, "mobility column is missing"
    assert "inv_mobility_gru_predictor" in ions.columns, "inv_mobility_gru_predictor column is missing"
    assert "inv_mobility_gru_predictor_std" in ions.columns, "inv_mobility_gru_predictor_std column is missing"

    # Calculate occurrences and abundances
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
        print("Serializing scan distributions to JSON...")

    # Optionally add noise and normalize
    if add_noise:
        noise_levels = np.random.uniform(0.0, 2.0, len(abundances))
        abundances = [
            add_uniform_noise(np.array(ab), level)
            for ab, level in zip(abundances, noise_levels)
        ]
        abundances = [ab / np.sum(ab) if np.sum(ab) > 0 else ab for ab in abundances]

    # Filter out zero-abundance entries
    filtered = [
        (
            [o for o, a in zip(occ, ab) if a > 1e-6],
            [a for a in ab if a > 1e-6]
        )
        for occ, ab in zip(occurrence, abundances)
    ]

    # Ensure each entry is at least an empty list
    scan_occurrence, scan_abundance = zip(*[
        (o, a) if len(o) > 0 else ([], [])
        for o, a in filtered
    ])

    # Serialize to JSON strings
    ions['scan_occurrence'] = [python_list_to_json_string(x, as_float=False) for x in scan_occurrence]
    ions['scan_abundance'] = [python_list_to_json_string(x, as_float=True) for x in scan_abundance]

    # Remove rows where scan_abundance is an empty list (length <= 2 due to '[]' or '[]\n')
    ions = ions[ions['scan_abundance'].apply(lambda x: len(x) > 2)]

    # Reorder and assign ion_id
    ions = ions[['peptide_id', 'sequence', 'charge', 'mz', 'relative_abundance',
                 'inv_mobility_gru_predictor', 'inv_mobility_gru_predictor_std',
                 'simulated_spectrum', 'scan_occurrence', 'scan_abundance']]

    ion_id = ions.index
    ions.insert(0, 'ion_id', ion_id)

    return ions