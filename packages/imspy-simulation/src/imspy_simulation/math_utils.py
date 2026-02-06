"""
Mathematical utility functions for simulation.

This module provides numba-jitted mathematical distribution functions
used throughout the simulation package.
"""

import math
import numba
import numpy as np
from numpy.typing import ArrayLike


@numba.jit(nopython=True)
def normal_pdf(
    x: ArrayLike,
    mass: float,
    s: float = 0.001,
    inv_sqrt_2pi: float = 0.3989422804014327,
    normalize: bool = False,
):
    """
    Normal probability density function.

    Args:
        x: Input values
        mass: Mean (center) of the distribution
        s: Standard deviation
        inv_sqrt_2pi: Precomputed 1/sqrt(2*pi) for efficiency
        normalize: If True, return normalized values (max=1)

    Returns:
        PDF values at x
    """
    a = (x - mass) / s
    if normalize:
        return np.exp(-0.5 * np.power(a, 2))
    else:
        return inv_sqrt_2pi / s * np.exp(-0.5 * np.power(a, 2))


@numba.jit(nopython=True)
def gaussian(x, μ: float = 0, σ: float = 1):
    """
    Gaussian (normal) distribution function.

    Args:
        x: Input values
        μ: Mean of the distribution
        σ: Standard deviation

    Returns:
        Gaussian PDF values at x
    """
    A = 1 / np.sqrt(2 * np.pi * np.power(σ, 2))
    B = np.exp(-(np.power(x - μ, 2) / 2 * np.power(σ, 2)))
    return A * B


@numba.jit(nopython=True)
def exp_distribution(x, λ: float = 1):
    """
    Exponential distribution function.

    Args:
        x: Input value
        λ: Rate parameter

    Returns:
        Exponential PDF value at x
    """
    if x > 0:
        return λ * np.exp(-λ * x)
    return 0


@numba.jit(nopython=True)
def exp_gaussian(x, μ: float = -3, σ: float = 1, λ: float = 0.25):
    """
    Exponentially modified Gaussian (EMG) distribution.

    Also known as the exponential-Gaussian convolution, commonly used
    to model chromatographic peak shapes.

    Args:
        x: Input values
        μ: Mean of the Gaussian component
        σ: Standard deviation of the Gaussian component
        λ: Rate parameter of the exponential component

    Returns:
        EMG PDF values at x
    """
    A = λ / 2 * np.exp(λ / 2 * (2 * μ + λ * np.power(σ, 2) - 2 * x))
    B = math.erfc((μ + λ * np.power(σ, 2) - x) / (np.sqrt(2) * σ))
    return A * B


class NormalDistribution:
    """Callable wrapper for the Gaussian distribution."""

    def __init__(self, μ: float, σ: float):
        self.μ = μ
        self.σ = σ

    def __call__(self, x):
        return gaussian(x, self.μ, self.σ)


class ExponentialGaussianDistribution:
    """Callable wrapper for the exponentially modified Gaussian distribution."""

    def __init__(self, μ: float = -3, σ: float = 1, λ: float = 0.25):
        self.μ = μ
        self.σ = σ
        self.λ = λ

    def __call__(self, x):
        return exp_gaussian(x, self.μ, self.σ, self.λ)
