import rustms_connector
rmsc = rustms_connector.py_elements

ELEMENTAL_MONO_ISOTOPIC_MASSES = rmsc.get_elemental_mono_isotopic_weight_map()
ELEMENTAL_ISOTOPIC_MASSES = rmsc.get_elemental_isotope_weight_map()
ELEMENTAL_ISOTOPIC_ABUNDANCES = rmsc.get_elemental_isotope_abundance_map()
