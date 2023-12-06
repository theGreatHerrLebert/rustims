import numpy as np

SUMMARY_CONSTANT = 18509.8632163405


def one_over_k0_to_ccs(one_over_k0, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """
    convert reduced ion mobility (1/k0) to CCS
    :param one_over_k0: reduced ion mobility
    :param charge: charge state of the ion
    :param mz: mass-over-charge of the ion
    :param mass_gas: mass of drift gas
    :param temp: temperature of the drift gas in C°
    :param t_diff: factor to translate from C° to K
    """
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return (SUMMARY_CONSTANT * charge) / (np.sqrt(reduced_mass * (temp + t_diff)) * 1 / one_over_k0)


def ccs_to_one_over_k0(ccs, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """
    convert CCS to 1 over reduced ion mobility (1/k0)
    :param ccs: collision cross-section
    :param charge: charge state of the ion
    :param mz: mass-over-charge of the ion
    :param mass_gas: mass of drift gas (N2)
    :param temp: temperature of the drift gas in C°
    :param t_diff: factor to translate from C° to K
    """
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return ((np.sqrt(reduced_mass * (temp + t_diff))) * ccs) / (SUMMARY_CONSTANT * charge)
