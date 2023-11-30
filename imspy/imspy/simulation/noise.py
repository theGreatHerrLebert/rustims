""" This module contains several noise models,
    such as detection noise, shot noise and baseline noise.
"""
import numpy as np
import numba
from typing import Callable, Optional
from numpy.typing import ArrayLike
from imspy.core import MzSpectrum
from imspy.utility.utilities import normal_pdf


@numba.jit(nopython=True)
def mu_function_normal_default(intensity: ArrayLike) -> ArrayLike:
    return intensity


@numba.jit(nopython=True)
def sigma_function_normal_default(intensity:ArrayLike) -> ArrayLike:
    return np.sqrt(intensity)


@numba.jit(nopython=True)
def mu_function_poisson_default(intensity:ArrayLike) -> ArrayLike:
    offset = 0
    return intensity + offset


@numba.jit(nopython=True)
def detection_noise(signal: ArrayLike,
                    method: str = "poisson",
                    custom_mu_function: Optional[Callable] = None,
                    custom_sigma_function: Optional[Callable] = None) -> ArrayLike:
    """

    :param signal:
    :param method:
    :param custom_mu_function:
    :param custom_sigma_function:
    """
    if method == "normal":
        if custom_sigma_function is None:
            sigma_function:Callable = sigma_function_normal_default
        else:
            sigma_function:Callable = custom_sigma_function
        if custom_mu_function is None:
            mu_function:Callable = mu_function_normal_default
        else:
            mu_function:Callable = custom_mu_function

        sigmas = sigma_function(signal)
        mus = mu_function(signal)
        noised_signal = np.zeros_like(mus)

        for i in range(noised_signal.size):
            s = sigmas[i]
            m = mus[i]
            n = np.random.normal()*s+m
            noised_signal[i] = n

    elif method == "poisson":
        if custom_sigma_function is not None:
            raise ValueError("Sigma function is not used if method is 'poisson'")
        if custom_mu_function is None:
            mu_function:Callable = mu_function_poisson_default
        else:
            mu_function:Callable = custom_mu_function
        mus = mu_function(signal)
        noised_signal = np.zeros_like(mus)

        for i in range(noised_signal.size):
            mu = mus[i]
            n = np.random.poisson(lam = mu)
            noised_signal[i] = n

    else:
        raise NotImplementedError("This method is not implemented, choose 'normal' or 'poisson'.")

    return noised_signal


@numba.jit(nopython=True)
def generate_noise_peak(pos:float, sigma: float, intensity: float, min_intensity:int = 0, resolution:float = 3):
    step_size = min(sigma/10,1/np.power(10,resolution))
    lower = int((pos-4*sigma)//step_size)
    upper = int((pos+4*sigma)//step_size)
    mzs = np.arange(lower,upper)*step_size
    intensities = normal_pdf(mzs,pos,sigma,normalize=True)*intensity*step_size
    return (mzs[intensities>=min_intensity], intensities[intensities>=min_intensity].astype(np.int64))


def baseline_shot_noise_window(window:MzSpectrum,
                               window_theoretical_mz_min:float,
                               window_theoretical_mz_max:float,
                               expected_noise_peaks: int = 5,
                               expected_noise_intensity: float = 10,
                               expected_noise_sigma:float = 0.001,
                               resolution:float = 3) -> MzSpectrum:
    """

    """
    num_noise_peaks = np.random.poisson(lam=expected_noise_peaks)
    noised_window = MzSpectrum([],[])
    for i in range(num_noise_peaks):
        location_i = np.random.uniform(window_theoretical_mz_min,window_theoretical_mz_max)
        intensity_i = np.random.exponential(expected_noise_intensity)
        sigma_i = np.random.exponential(expected_noise_sigma)
        noise_mz, noise_intensity = generate_noise_peak(location_i,sigma_i,intensity_i,resolution=resolution)
        noise_peak = MzSpectrum(noise_mz, noise_intensity)
        noised_window = noised_window+noise_peak
    return noised_window


def baseline_shot_noise(spectrum:MzSpectrum,window_size:float=50,expected_noise_peaks_per_Th:int=10,min_intensity:int = 5, resolution = 3):
    """


    """
    min_mz = spectrum.mz.min()-0.1
    max_mz = spectrum.mz.max()+0.1
    bins,windows = spectrum.windows(window_size,overlapping=False,min_peaks=0,min_intensity=0)
    noised_spectrum = spectrum
    for b,w in zip(bins,windows):
        lower = b*window_size
        upper = (b+1)*window_size
        noised_spectrum = noised_spectrum+baseline_shot_noise_window(w,lower,upper,window_size*expected_noise_peaks_per_Th)
    return noised_spectrum.to_resolution(resolution).filter(min_mz,max_mz,min_intensity)


def baseline_noise():
    pass
