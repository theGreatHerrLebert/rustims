import imspy_connector

from imspy.data.spectrum import MzSpectrum

ims = imspy_connector.py_chemistry


def calculate_mz(mass: float, charge: int) -> float:
    """Calculate m/z value.

    Args:
        mass (float): Mass.
        charge (int): Charge.

    Returns:
        float: m/z value.
    """
    return ims.calculate_mz(mass, charge)


def calculate_transmission_dependent_fragment_ion_isotope_distribution(
        target_spec: MzSpectrum,
        complement_spec: MzSpectrum,
        transmitted_isotopes: MzSpectrum,
        max_isotope: int) -> MzSpectrum:
    """Calculate transmission dependent fragment ion isotope distribution.

    Args:
        target_spec (MzSpectrum): Target spectrum.
        complement_spec (MzSpectrum): Complement spectrum.
        transmitted_isotopes (MzSpectrum): Transmitted isotopes.
        max_isotope (int): Maximum isotope.

    Returns:
        MzSpectrum: Transmission dependent fragment ion isotope distribution.
    """
    return MzSpectrum.from_py_ptr(
        ims.calculate_transmission_dependent_fragment_ion_isotope_distribution(
            target_spec.get_py_ptr(),
            complement_spec.get_py_ptr(),
            transmitted_isotopes.get_py_ptr(), max_isotope
        )
    )
