from typing import Dict, List
import imspy_connector
ims = imspy_connector.py_elements

ELEMENTAL_MONO_ISOTOPIC_MASSES = ims.get_elemental_mono_isotopic_weight_map()
ELEMENTAL_ISOTOPIC_MASSES = ims.get_elemental_isotope_weight_map()
ELEMENTAL_ISOTOPIC_ABUNDANCES = ims.get_elemental_isotope_abundance_map()


def get_elemental_isotopes_abundance_map() -> Dict[str, List[float]]:
    """Get the isotopic abundances of all elements.
    Returns:
        The isotopic abundances of all elements.
    """
    return ims.get_elemental_isotope_abundance_map()


def get_elemental_isotopes_weight_map() -> Dict[str, List[float]]:
    """Get the isotopic weights of all elements.
    Returns:
        The isotopic weights of all elements.
    """
    return ims.get_elemental_isotope_weight_map()


def get_elemental_mono_isotopic_weight_map() -> Dict[str, float]:
    """Get the mono isotopic weights of all elements.
    Returns:
        The mono isotopic weights of all elements.
    """
    return ims.get_elemental_mono_isotopic_weight_map()
