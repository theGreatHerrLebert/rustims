import os
from numpy.typing import NDArray
import imspy_connector
ims = imspy_connector.py_chemistry


def one_over_k0_to_ccs(one_over_k0, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """Convert reduced ion mobility (1/k0) to CCS.

    Args:
        one_over_k0: reduced ion mobility
        mz: mass-over-charge of the ion
        charge: charge state of the ion
        mass_gas: mass of drift gas
        temp: temperature of the drift gas in C°
        t_diff: factor to translate from C° to K

    Returns:
        float: collision cross-section
    """
    return ims.one_over_reduced_mobility_to_ccs(one_over_k0, mz, charge, mass_gas, temp, t_diff)


def ccs_to_one_over_k0(ccs, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """Convert CCS to reduced ion mobility (1/k0).

    Args:
        ccs: collision cross-section
        mz: mass-over-charge of the ion
        charge: charge state of the ion
        mass_gas: mass of drift gas
        temp: temperature of the drift gas in C°
        t_diff: factor to translate from C° to K

    Returns:
        float: reduced ion mobility
    """
    return ims.ccs_to_one_over_reduced_mobility(ccs, mz, charge, mass_gas, temp, t_diff)

def ccs_to_one_over_k0_par(ccs: NDArray, mz: NDArray, charge: NDArray, mass_gas: float = 28.013, temp: float = 31.85,
                          t_diff: float = 273.15, num_threads: int = -1) -> NDArray:
    """Convert CCS to reduced ion mobility (1/k0) in parallel.

    Args:
        ccs: collision cross-section
        mz: mass-over-charge of the ion
        charge: charge state of the ion
        mass_gas: mass of drift gas
        temp: temperature of the drift gas in C°
        t_diff: factor to translate from C° to K
        num_threads: number of threads

    Returns:
        NDArray: reduced ion mobility
    """

    if num_threads == -1:
        num_threads = os.cpu_count()

    return ims.ccs_to_one_over_reduced_mobility_par(ccs, mz, charge, mass_gas, temp, t_diff, num_threads)

def one_over_k0_to_ccs_par(one_over_k0: NDArray, mz: NDArray, charge: NDArray, mass_gas: float = 28.013, temp: float = 31.85,
                            t_diff: float = 273.15, num_threads: int = -1) -> NDArray:
     """Convert reduced ion mobility (1/k0) to CCS in parallel.

     Args:
          one_over_k0: reduced ion mobility
          mz: mass-over-charge of the ion
          charge: charge state of the ion
          mass_gas: mass of drift gas
          temp: temperature of the drift gas in C°
          t_diff: factor to translate from C° to K
          num_threads: number of threads

     Returns:
          NDArray: collision cross-section
     """

     if num_threads == -1:
          num_threads = os.cpu_count()

     return ims.one_over_reduced_mobility_to_ccs_par(one_over_k0, mz, charge, mass_gas, temp, t_diff, num_threads)
