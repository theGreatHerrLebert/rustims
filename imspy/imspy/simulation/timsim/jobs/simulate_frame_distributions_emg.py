from typing import Tuple, Union
import numpy as np
import pandas as pd
from scipy.special import erfcx
from numpy.typing import ArrayLike
from imspy.simulation.utility import python_list_to_json_string, add_uniform_noise
import imspy_connector
ims = imspy_connector.py_utility


def calculate_rt_defaults(gradient_length: float) -> dict:
    """Calculates 'sigma_lower_rt' and 'sigma_upper_rt', if these
    are not provided by the user. The calculation is based
    on the gradient length.

    Args:
        gradient_length (float): Length of the LC gradient in seconds.

    Returns:
        dict: Parameter dictionary with calculated values.
    """

    sigma_middle_rt = gradient_length / (60 * 60) * 0.75 + 1.125
    sigma_lower_rt = sigma_middle_rt - sigma_middle_rt * 0.25
    sigma_upper_rt = sigma_middle_rt + sigma_middle_rt * 0.25
    return {
        'sigma_lower_rt': sigma_lower_rt,
        'sigma_upper_rt': sigma_upper_rt
    }


def sample_sigma_lambda_emg(sigma_lower:ArrayLike, 
                            sigma_upper:ArrayLike, 
                            sigma_alpha:ArrayLike, 
                            sigma_beta:ArrayLike, 
                            lambda_lower:ArrayLike, 
                            lambda_upper:ArrayLike, 
                            lambda_alpha:ArrayLike, 
                            lambda_beta:ArrayLike, 
                            n:int)->Tuple[ArrayLike,ArrayLike]:
    r"""
    Sample :math:`\sigma` and :math:`\lambda` from scaled beta distributions:
    
    .. math::
        \begin{aligned}
        \sigma &= \sigma_{\text{lower}} + \hat{\sigma} \cdot (\sigma_{\text{upper}} - \sigma_{\text{lower}}) \\
        \hat{\sigma} &\sim \text{Beta}(\alpha_{\sigma}, \beta_{\sigma})
        \end{aligned}
    
    
    This function is currently not used in the codebase.
    It is kept in case we want to use the EMG parametrization
    with :math:`\sigma` and :math:`\lambda` instead of :math:`\sigma` and :math:`k`.

    Args:
        sigma_lower (ArrayLike): The lower bound for :math:`\sigma`.
        sigma_upper (ArrayLike): The upper bound for :math:`\sigma`.
        sigma_alpha (ArrayLike): The :math:`\alpha` parameter for the beta distribution for :math:`\hat{sigma}`.
        sigma_beta (ArrayLike): The :math:`\beta` parameter for the beta distribution for :math:`\hat{sigma}`.
        lambda_lower (ArrayLike): The lower bound for :math:`\lambda`.
        lambda_upper (ArrayLike): The upper bound for :math:`\lambda`.
        lambda_alpha (ArrayLike): The :math:`\alpha` parameter for the beta distribution for :math:`\hat{\lambda}`.
        lambda_beta (ArrayLike): The :math:`\beta` parameter for the beta distribution for :math:`\hat{\lambda}`.
        n (int): Number of samples.

    Returns:
        Tuple[ArrayLike, ArrayLike]: The sampled :math:`\sigma` and :math:`\lambda`.
    """
    # TODO use rng
    sigma_hat = np.random.beta(a=sigma_alpha, b=sigma_beta, size=n)
    lambda_hat = np.random.beta(a=lambda_alpha, b=lambda_beta, size=n)
    sigmas = sigma_lower + sigma_hat * (sigma_upper - sigma_lower)
    lambdas = lambda_lower + lambda_hat * (lambda_upper - lambda_lower)
    
    return sigmas, lambdas

def sample_sigma_k_emg(sigma_lower: ArrayLike, 
                        sigma_upper: ArrayLike, 
                        sigma_alpha: ArrayLike, 
                        sigma_beta: ArrayLike, 
                        k_lower: ArrayLike, 
                        k_upper: ArrayLike, 
                        k_alpha: ArrayLike, 
                        k_beta: ArrayLike, 
                        n:int)->Tuple[ArrayLike,ArrayLike]:
    r"""
    Sample :math:`\sigma` and :math:`k` from scaled beta distributions:
    
    .. math::
        \begin{aligned}
        \sigma &= \sigma_{\text{lower}} + \hat{\sigma} \cdot (\sigma_{\text{upper}} - \sigma_{\text{lower}}) \\
        \hat{\sigma} &\sim \text{Beta}(\alpha_{\sigma}, \beta_{\sigma}) \\
        \end{aligned}
    
    This function samples :math:`\sigma` and :math:`k` parameters for the exponentially modified Gaussian (EMG) distribution
    with: 
    
    .. math::
        k=\frac{1}{\sigma\lambda}
    
    Args:
        sigma_lower (ArrayLike): The lower bound for :math:`\sigma`.
        sigma_upper (ArrayLike): The upper bound for :math:`\sigma`.
        sigma_alpha (ArrayLike): The :math:`\alpha` parameter for the beta distribution for :math:`\hat{sigma}`.
        sigma_beta (ArrayLike): The :math:`\beta` parameter for the beta distribution for :math:`\hat{sigma}`.
        k_lower (ArrayLike): The lower bound for :math:`k`.
        k_upper (ArrayLike): The upper bound for :math:`k`.
        k_alpha (ArrayLike): The :math:`\alpha` parameter for the beta distribution for :math:`\hat{k}`.
        k_beta (ArrayLike): The :math:`\beta` parameter for the beta distribution for :math:`\hat{k}`.
        n (int): Number of samples.

    Returns:
        Tuple[ArrayLike, ArrayLike]: The sampled :math:`\sigma` and :math:`k`.
    """
    # TODO use rng
    sigma_hat = np.random.beta(a=sigma_alpha, b=sigma_beta, size=n)
    k_hat = np.random.beta(a=k_alpha, b=k_beta, size=n)
    sigmas = sigma_lower + sigma_hat * (sigma_upper - sigma_lower)
    ks = k_lower + k_hat * (k_upper - k_lower)
    
    return sigmas, ks

def erfcxinv(y:ArrayLike, n:int=10)->ArrayLike:
    """
    Calculates the inverse of the scaled complementary error function (erfcx) 
    via the Newton-Raphson method.
    
    Args:
        y (ArrayLike): The value(s) for which the inverse is to be calculated.    
        n (int, optional): Number of iterations for the Newton-Raphson method. Default is 10.    
    Returns:
        ArrayLike: The inverse of the scaled complementary error function at y.
    """
    assert np.all(y > 0), "y must be positive, as erfcx only maps to positive values."
    # assert that y is an array of np.float64 
    y = np.array(y).astype(np.float64)
    # start value depends on the value of y
    xn_start = np.where(y<2,1/(y*np.sqrt(np.pi)), -np.sqrt(np.log(y/2, out=np.zeros_like(y), where=y>=2)))
    xn = xn_start
    # Newton-Raphson method
    for i in range(n):
        xn = xn - (erfcx(xn)-y)/(2*xn*erfcx(xn)-2/np.sqrt(np.pi))
    return xn

def estimate_mu_from_mode_emg(mode: ArrayLike, sigma: ArrayLike, lambda_: ArrayLike)->ArrayLike:
    r"""
    Estimate the parameter :math:`\mu` of an EMG distribution from the mode (vectorized).
    The function uses the following formula 
    (adapted from en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution)
    
    .. math::
        \mu = x_m + \sqrt{2}\sigma\text{erfcx}^{-1}\left(\frac{1}{\lambda\sigma}\sqrt{\frac{2}{\pi}}\right)-\sigma^2\lambda
    
    
    Args:
        mode (ArrayLike): The modes of the EMG distributions.
        sigma (ArrayLike): EMG parameters :math:`\sigma`.
        lambda_ (ArrayLike): EMG parameters :math:`\lambda`.
    
    Returns:
        ArrayLike: The estimated parameters :math:`\mu`.
    """
    return mode + np.sqrt(2)*sigma*erfcxinv(1/(lambda_*sigma)*np.sqrt(2/np.pi))-np.power(sigma,2)*lambda_


def simulate_frame_distributions_emg(
        peptides: pd.DataFrame,
        frames: pd.DataFrame,
        sigma_lower_rt: Union[float, None],
        sigma_upper_rt: Union[float, None],
        sigma_alpha_rt: float,
        sigma_beta_rt: float,
        k_lower_rt: float,
        k_upper_rt: float,
        k_alpha_rt: float,
        k_beta_rt: float,
        target_p: float,
        step_size: float,
        rt_cycle_length: float,
        verbose: bool = False,
        add_noise: bool = False,
        n_steps: int = 1000,
        num_threads: int = 4,
        from_existing: bool = False,
        sigmas: np.ndarray = None,
        lambdas: np.ndarray = None,
        gradient_length: float = None,
        remove_epsilon: float = 1e-4,
) -> pd.DataFrame:
    """Simulate frame distributions for peptides.

    Args:
        peptides: Peptide DataFrame.
        frames: Frame DataFrame.
        sigma_lower_rt: Lower bound for sigma of an EMG chromatographic peak.
        sigma_upper_rt: Upper bound for sigma of an EMG chromatographic peak.
        sigma_alpha_rt: Alpha for beta distribution for sigma_hat that is then scaled to sigma in (sigma_lower_rt, sigma_upper_rt).
        sigma_beta_rt: Beta for beta distribution for sigma_hat that is then scaled to sigma in (sigma_lower_rt, sigma_upper_rt).
        k_lower_rt: Lower bound for k of an EMG chromatographic peak.
        k_upper_rt: Upper bound for k of an EMG chromatographic peak.
        k_alpha_rt: Alpha for beta distribution for k_hat that is then scaled to k in (k_lower_rt, k_upper_rt).
        k_beta_rt: Beta for beta distribution for k_hat that is then scaled to k in (k_lower_rt, k_upper_rt).
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
        gradient_length: Length of the LC gradient in seconds.
        remove_epsilon: Remove values with a probability lower than this value.

    Returns:
        pd.DataFrame: Peptide DataFrame with frame distributions.
    """

    frames_np = frames.frame_id.values
    times_np = frames.time.values
    peptide_rt = peptides

    if sigma_lower_rt is None:
        if gradient_length is None:
            raise ValueError("gradient_length must be provided if sigma_lower_rt is None")
        defaults = calculate_rt_defaults(gradient_length)
        sigma_lower_rt = defaults['sigma_lower_rt']
        sigma_upper_rt = defaults['sigma_upper_rt']

    if verbose and not from_existing:
        print("Calculating frame distributions ...")
        print(f"sigma_lower: {sigma_lower_rt}, sigma_upper: {sigma_upper_rt}, sigma_alpha: {sigma_alpha_rt}, sigma_beta: {sigma_beta_rt}")

    n = peptides.shape[0]

    if not from_existing:
        sigmas, ks = sample_sigma_k_emg(sigma_lower_rt, 
                                        sigma_upper_rt, 
                                        sigma_alpha_rt,
                                        sigma_beta_rt,
                                        k_lower_rt,
                                        k_upper_rt,
                                        k_alpha_rt,
                                        k_beta_rt,
                                        n)
        lambdas = 1 / (ks * sigmas)

    mus = estimate_mu_from_mode_emg(peptide_rt.retention_time_gru_predictor, sigmas, lambdas)
    peptide_rt['rt_sigma'] = sigmas
    peptide_rt['rt_lambda'] = lambdas
    
    occurrences = ims.calculate_frame_occurrences_emg_par(
        times_np,
        mus,
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
        mus,
        sigmas,
        lambdas,
        rt_cycle_length,
        num_threads=num_threads,
        n_steps=n_steps,
    )

    if verbose:
        print("Serializing frame distributions to json ...")

    # filter occurrences and abundances
    occurrences = [list(np.array(occurrence)[np.array(abundance) > remove_epsilon]) for occurrence, abundance in zip(occurrences, abundances)]
    abundances = [list(np.array(abundance)[np.array(abundance) > remove_epsilon]) for abundance in abundances]

    # this could now fail, if all values are removed because they are below remove_epsilon

    first_occurrence, last_occurrence = [], []

    for oc, ab in zip(occurrences, abundances):
        if len(oc) == 0 or len(ab) == 0:
            first_occurrence.append(-1)
            last_occurrence.append(-1)
        else:
            first_occurrence.append(oc[0])
            last_occurrence.append(oc[-1])

    peptide_rt['frame_occurrence_start'] = first_occurrence
    peptide_rt['frame_occurrence_end'] = last_occurrence

    peptide_rt['frame_occurrence'] = occurrences

    if add_noise:
        # TODO: make noise model configurable
        noise_levels = np.random.uniform(0.0, 2.0, len(abundances))
        abundances = [add_uniform_noise(np.array(abundance), noise_level) for abundance, noise_level in zip(abundances, noise_levels)]
        # Normalize frame abundance
        abundances = [frame_abundance / np.sum(frame_abundance) for frame_abundance in abundances]

    peptide_rt['frame_abundance'] = [list(x) for x in abundances]

    peptide_rt['frame_occurrence'] = peptide_rt['frame_occurrence'].apply(
        lambda r: python_list_to_json_string(r, as_float=False)
    )

    # replace NaN values with 0.0 in frame abundance
    peptide_rt['frame_abundance'] = [np.nan_to_num(np.array(x), nan=0.0).tolist() for x in abundances]

    peptide_rt['frame_abundance'] = peptide_rt['frame_abundance'].apply(
        lambda r: python_list_to_json_string(r, as_float=True)
    )

    peptide_rt['rt_mu'] = mus

    # print how many rows get removed, if any
    if verbose:
        num_removed = np.sum((peptide_rt['frame_occurrence_start'] == -1) | (peptide_rt['frame_occurrence_end'] == -1))
        if num_removed > 0:
            print(f"Removing {num_removed} peptides that do not elute in any frame.")

    # remove lists where frame_start is -1, or frame_end is -1
    peptide_rt = peptide_rt[(peptide_rt['frame_occurrence_start'] != -1) & (peptide_rt['frame_occurrence_end'] != -1)]

    # remove empty lists, that are now cast to strings, for frame_abundance
    peptide_rt_filtered = peptide_rt[peptide_rt['frame_abundance'].apply(len) > 2]

    peptide_rt_filtered = peptide_rt_filtered.sort_values(by=['frame_occurrence_start', 'frame_occurrence_end'])

    return peptide_rt_filtered
