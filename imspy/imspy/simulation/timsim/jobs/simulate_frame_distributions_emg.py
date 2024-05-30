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
    while any(lambdas <= 0.05):
        lambdas[lambdas <= 0.05] = np.random.normal(loc=lambda_mean, scale=np.sqrt(lambda_variance),
                                                    size=np.sum(lambdas < 0.05))

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
) -> pd.DataFrame:

    frames_np = frames.frame_id.values
    times_np = frames.time.values
    peptide_rt = peptides

    if verbose:
        print("Calculating frame distributions...")

    n = peptides.shape[0]

    sigmas, lambdas = sample_parameters_rejection(mean_std_rt, variance_std_rt, mean_scewness, variance_scewness, n)

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

    peptide_rt['frame_occurrence'] = occurrences

    if add_noise:
        noise_levels = np.random.uniform(0.0, 2.0, len(abundances))
        abundances = [add_uniform_noise(np.array(abundance), noise_level) for abundance, noise_level in zip(abundances, noise_levels)]

    if normalize:
        abundances = [frame_abundance / np.sum(frame_abundance) for frame_abundance in abundances]

    peptide_rt['frame_abundance'] = [list(x) for x in abundances]

    peptide_rt['frame_occurrence'] = peptide_rt['frame_occurrence'].apply(
        lambda r: python_list_to_json_string(r, as_float=False)
    )

    peptide_rt['frame_abundance'] = peptide_rt['frame_abundance'].apply(
        lambda r: python_list_to_json_string(r, as_float=True)
    )

    peptide_rt = peptides.sort_values(by=['frame_occurrence_start', 'frame_occurrence_end'])

    return peptide_rt
