from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod

from imspy.data.spectrum import MzSpectrum
from imspy.utility.utilities import gaussian, exp_gaussian, normal_pdf
import numba
import pyopenms

from imspy.simulation.noise import detection_noise
import imspy_connector
ims = imspy_connector.py_chemistry

MASS_PROTON = 1.007276466621
MASS_NEUTRON = 1.00866491595


def simulate_precursor_spectrum(sequence: str, charge: int) -> MzSpectrum:
    return MzSpectrum.from_py_mz_spectrum(ims.simulate_precursor_spectrum(sequence, charge))


def simulate_precursor_spectra(sequences: NDArray, charges: NDArray, num_threads: int) -> List[MzSpectrum]:
    spectra = ims.simulate_precursor_spectra(sequences, charges, num_threads)
    return [MzSpectrum.from_py_mz_spectrum(x) for x in spectra]


@numba.jit(nopython=True)
def factorial(n: int):
    if n <= 0:
        return 1
    return n * factorial(n - 1)


# linear dependency of mass
@numba.jit(nopython=True)
def lam(mass: float, slope: float = 0.000594, intercept: float = -0.03091):
    """
    :param intercept:
    :param slope:
    :param mass:
    :return:
    """
    return slope * mass + intercept


@numba.jit(nopython=True)
def weight(mass: float, peak_nums: NDArray, normalize: bool = True):
    """
    :param peak_nums:
    :param mass:
    :param normalize:
    :return:
    """
    factorials = np.zeros_like(peak_nums)
    norm = 1
    for i, k in enumerate(peak_nums):
        factorials[i] = factorial(k)
    weights = np.exp(-lam(mass)) * np.power(lam(mass), peak_nums) / factorials
    if normalize:
        norm = weights.sum()
    return weights / norm


def get_pyopenms_weights(sequence: str, peak_nums: ArrayLike, generator: pyopenms.CoarseIsotopePatternGenerator):
    """
    Gets hypothetical intensities of isotopic peaks.

    :param peak_nums:
    :param sequence: Peptide sequence in proForma format.
    :type sequence: str
    :param generator: pyopenms generator for isotopic pattern calculation
    :type generator: pyopenms.CoarseIsotopePatternGenerator
    :return: List of isotopic peak intensities
    :rtype: List
    """

    n = peak_nums.shape[0]
    generator.setMaxIsotope(n)
    aa_sequence = sequence.strip("_")
    peptide = pyopenms.AASequence().fromString(aa_sequence.replace("[", "(").replace("]", ")"))
    formula = peptide.getFormula()
    distribution = generator.run(formula)
    intensities = [i.getIntensity() for i in distribution.getContainer()]
    return intensities


@numba.jit(nopython=True)
def iso(x: NDArray, mass: float, charge: float, sigma: float, amp: float, K: int, step_size: float,
        add_detection_noise: bool = False, mass_neutron: float = MASS_NEUTRON):
    """
    :param mass_neutron:
    :param x:
    :param mass:
    :param charge:
    :param sigma:
    :param amp:
    :param K:4
    :param step_size:
    :param add_detection_noise:
    :return:
    """
    k = np.arange(K)
    means = ((mass + mass_neutron * k) / charge).reshape((1, -1))
    weights = weight(mass, k).reshape((1, -1))
    intensities = np.sum(weights * normal_pdf(x.reshape((-1, 1)), means, sigma), axis=1) * step_size
    if add_detection_noise:
        return detection_noise(intensities * amp)
    else:
        return intensities * amp


@numba.jit(nopython=True)
def numba_generate_pattern(
        lower_bound: float,
        upper_bound: float,
        mass: float,
        charge: float,
        amp: float,
        k: int,
        sigma: float = 0.008492569002123142,
        resolution: int = 3) -> Tuple[ArrayLike, ArrayLike]:
    """
    :param lower_bound:
    :param upper_bound:
    :param mass:
    :param charge:
    :param amp:
    :param k:
    :param sigma:
    :param resolution:
    :return:
    """
    step_size = min(sigma / 10, 1 / np.power(10, resolution))
    size = int((upper_bound - lower_bound) // step_size + 1)
    mzs = np.linspace(lower_bound, upper_bound, size)
    intensities = iso(mzs, mass, charge, sigma, amp, k, step_size)

    return mzs + MASS_PROTON, intensities


def numba_ion_sampler(mass: float, charge: int, sigma: ArrayLike, k: int, ion_count: int, intensity_per_ion: int):
    sigma = np.atleast_1d(sigma)
    if sigma.size == 1:
        sigma = np.repeat(sigma, k)
    if sigma.size != k:
        raise ValueError("Sigma must be either length 1 or k")

    us = np.zeros(ion_count)
    devs_std = np.zeros(ion_count)
    for ion in range(ion_count):
        u = np.random.uniform()
        dev_std = np.random.normal()
        us[ion] = u
        devs_std[ion] = dev_std

    weights = weight(mass, np.arange(k)).cumsum()

    def _get_component(u: float, weights_cum_sum: ArrayLike = weights) -> int:
        for idx, weight_sum in enumerate(weights_cum_sum):
            if u < weight_sum:
                return idx

    comps = np.zeros_like(us)
    devs = np.zeros_like(us)
    for idx, u in enumerate(us):
        comp = _get_component(u)
        comps[idx] = comp
        devs[idx] = sigma[comp] * devs_std[idx]

    mz = ((comps * MASS_NEUTRON) + mass) / charge + MASS_PROTON + devs
    i = np.ones_like(mz) * intensity_per_ion
    return mz, i


@numba.jit(nopython=True)
def create_initial_feature_distribution(num_rt: int, num_im: int,
                                        rt_lower: float = -9,
                                        rt_upper: float = 18,
                                        im_lower: float = -4,
                                        im_upper: float = 4,
                                        distr_im=gaussian,
                                        distr_rt=exp_gaussian) -> np.array:
    I = np.ones((num_rt, num_im)).astype(np.float32)

    for i, x in enumerate(np.linspace(im_lower, im_upper, num_im)):
        for j, y in enumerate(np.linspace(rt_lower, rt_upper, num_rt)):
            I[j, i] *= (distr_im(x) * distr_rt(y))

    return I


class IsotopePatternGenerator(ABC):
    @abstractmethod
    def generate_pattern(self, mass: float, charge: int) -> Tuple[ArrayLike, ArrayLike]:
        pass

    @abstractmethod
    def generate_spectrum(self, mass: int, charge: int, min_intensity: int) -> MzSpectrum:
        pass


class AveragineGenerator(IsotopePatternGenerator):
    def __init__(self):
        super().__init__()
        self.default_abundance = 1e4

    def generate_pattern(self, mass: float, charge: int, k: int = 7,
                         amp: Optional[float] = None, resolution: float = 3,
                         min_intensity: int = 5) -> Tuple[ArrayLike, ArrayLike]:
        pass

    def generate_spectrum(self, mass: int, charge: int, min_intensity: int = 5, k: int = 7,
                          amp: Optional[float] = None, resolution: int = 3, centroided: bool = True) -> MzSpectrum:
        if amp is None:
            amp = self.default_abundance

        lb = mass / charge - .2
        ub = mass / charge + k + .2

        mz, i = numba_generate_pattern(
            lower_bound=lb,
            upper_bound=ub,
            mass=mass,
            charge=charge,
            amp=amp,
            k=k,
            resolution=resolution
        )

        # TODO: implement centroid method
        if centroided:
            return MzSpectrum(mz, i) \
                .to_resolution(resolution).filter(lb, ub, min_intensity) \
                .to_centroided(baseline_noise_level=max(min_intensity, 1), sigma=1 / np.power(10, resolution - 1))

        return MzSpectrum(mz, i).to_resolution(resolution).filter(lb, ub, min_intensity)

    @staticmethod
    def generate_spectrum_by_sampling(
            mass: float,
            charge: int,
            k: int = 7,
            sigma: ArrayLike = 0.008492569002123142,
            ion_count: int = 1000000,
            resolution: int = 3,
            intensity_per_ion: int = 1,
            centroided: bool = True,
            min_intensity: int = 5) -> MzSpectrum:

        assert 100 <= mass / charge <= 2000, f"m/z should be between 100 and 2000, was: {mass / charge}"
        mz, i = numba_ion_sampler(mass, charge, sigma, k, ion_count, intensity_per_ion)

        if centroided:
            return (MzSpectrum(mz, i)
                    .to_resolution(resolution).filter(-1, -1, min_intensity)
                    .to_centroided(baseline_noise_level=max(min_intensity, 1), sigma=1 / np.power(10, resolution - 1))
                    )
        return MzSpectrum(mz, i) \
            .to_resolution(resolution).filter(-1, -1, min_intensity)


def generate_isotope_pattern_rust(mass: float, charge: int, min_intensity: float = 150, k: int = 7, resolution: int = 3,
                                  centroid: bool = True):
    return MzSpectrum.from_py_mz_spectrum(ims.generate_precursor_spectrum(
        mass=mass,
        charge=charge,
        min_intensity=min_intensity,
        k=k,
        resolution=resolution,
        centroid=centroid
    ))


def generate_isotope_patterns_rust(masses: NDArray, charges: NDArray, min_intensity: float = 0, k: int = 7,
                                   resolution: int = 3, centroid: bool = True, num_threads: int = 4):
    return [MzSpectrum.from_py_mz_spectrum(x) for x in ims.generate_precursor_spectra(
        masses=masses,
        charges=charges,
        min_intensity=min_intensity,
        k=k,
        resolution=resolution,
        centroid=centroid,
        num_threads=num_threads,
    )]
