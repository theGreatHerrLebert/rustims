import numpy as np
import pandas as pd
from imspy.simulation.utility import python_list_to_json_string, add_uniform_noise
import imspy_connector
ims = imspy_connector.py_utility


def sample_parameters_rejection(sigma_mean, sigma_variance, lambda_mean, lambda_variance, n):
    sigmas = np.random.normal(loc=sigma_mean, scale=np.sqrt(sigma_variance), size=n)
    lambdas = np.random.normal(loc=lambda_mean, scale=np.sqrt(lambda_variance), size=n)

    # Re-sample any negative values
    while any(sigmas < 0):
        sigmas[sigmas < 0] = np.random.normal(loc=sigma_mean, scale=np.sqrt(sigma_variance), size=np.sum(sigmas < 0))
    while any(lambdas <= 0.01):
        lambdas[lambdas <= 0.01] = np.random.normal(loc=lambda_mean, scale=np.sqrt(lambda_variance),
                                                    size=np.sum(lambdas < 0.01))

    return sigmas, lambdas


def simulate_frame_distributions_emg(
        peptides: pd.DataFrame,
        frames: pd.DataFrame,
        mean_std_rt: float,
        variance_std_rt: float,
        mean_scewness: float,
        variance_scewness: float,
        target_p: float,
        step_size: float,
        rt_cycle_length: float,
        verbose: bool = False,
        add_noise: bool = False,
        normalize: bool = False,
        n_steps: int = 1000,
        num_threads: int = 4,
        from_existing: bool = False,
        sigmas: np.ndarray = None,
        lambdas: np.ndarray = None,
) -> pd.DataFrame:
    """Simulate frame distributions for peptides.

    Args:
        peptides: Peptide DataFrame.
        frames: Frame DataFrame.
        mean_std_rt: mean retention time.
        variance_std_rt: variance retention time.
        mean_scewness: mean scewness.
        variance_scewness: variance scewness.
        target_p: target p.
        step_size: step size.
        rt_cycle_length: Retention time cycle length in seconds.
        verbose: Verbosity.
        add_noise: Add noise.
        normalize: Normalize frame abundance.
        n_steps: number of steps.
        num_threads: number of threads.
        from_existing: Use existing parameters.
        sigmas: sigmas.
        lambdas: lambdas.

    Returns:
        pd.DataFrame: Peptide DataFrame with frame distributions.
    """

    frames_np = frames.frame_id.values
    times_np = frames.time.values
    peptide_rt = peptides

    if verbose:
        print("Calculating frame distributions...")

    n = peptides.shape[0]

    if not from_existing:
        sigmas, lambdas = sample_parameters_rejection(mean_std_rt, variance_std_rt, mean_scewness, variance_scewness, n)

    peptide_rt['rt_sigma'] = sigmas
    peptide_rt['rt_lambda'] = lambdas

    occurrences = ims.calculate_frame_occurrences_emg_par(
        times_np,
        peptides.retention_time_gru_predictor,
        sigmas,
        lambdas,
        target_p,
        step_size,
        num_threads=num_threads,
        n_steps=n_steps,
    )

    abundances = ims.calculate_frame_abundances_emg_par(
        frames_np,
        times_np,
        occurrences,
        peptides.retention_time_gru_predictor,
        sigmas,
        lambdas,
        rt_cycle_length,
        num_threads=num_threads,
        n_steps=n_steps,
    )

    if verbose:
        print("Serializing frame distributions to json...")

    first_occurrence = [occurrence[0] for occurrence in occurrences]
    last_occurrence = [occurrence[-1] for occurrence in occurrences]

    peptide_rt['frame_occurrence_start'] = first_occurrence
    peptide_rt['frame_occurrence_end'] = last_occurrence

    peptide_rt['frame_occurrence'] = [list(x) for x in occurrences]

    if add_noise:
        noise_levels = np.random.uniform(0.0, 2.0, len(abundances))
        abundances = [add_uniform_noise(np.array(abundance), noise_level) for abundance, noise_level in zip(abundances, noise_levels)]

    if normalize:
        abundances = [frame_abundance / np.sum(frame_abundance) for frame_abundance in abundances]

    peptide_rt['frame_abundance'] = [list(x) for x in abundances]

    # remove entries where frame_abundance is empty
    peptide_rt = peptide_rt[peptide_rt['frame_abundance'].apply(lambda l: len(l) > 0)]

    # print type of frame_occurrence and frame_abundance BEFORE applying the lambda functions
    print(type(peptide_rt['frame_occurrence']))
    print(type(peptide_rt['frame_abundance']))

    peptide_rt['frame_occurrence'] = peptide_rt['frame_occurrence'].apply(
        lambda r: python_list_to_json_string(r, as_float=False)
    )

    peptide_rt['frame_abundance'] = peptide_rt['frame_abundance'].apply(
        lambda r: python_list_to_json_string(r, as_float=True)
    )

    # print type of frame_occurrence and frame_abundance AFTER applying the lambda functions
    print(type(peptide_rt['frame_occurrence']))
    print(type(peptide_rt['frame_abundance']))

    peptide_rt = peptides.sort_values(by=['frame_occurrence_start', 'frame_occurrence_end'])

    return peptide_rt
