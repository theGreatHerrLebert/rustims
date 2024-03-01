import imspy_connector
ims = imspy_connector.py_elements

ELEMENTAL_MONO_ISOTOPIC_MASSES = ims.get_elemental_mono_isotopic_weight_map()
ELEMENTAL_ISOTOPIC_MASSES = ims.get_elemental_isotope_weight_map()
ELEMENTAL_ISOTOPIC_ABUNDANCES = ims.get_elemental_isotope_abundance_map()
