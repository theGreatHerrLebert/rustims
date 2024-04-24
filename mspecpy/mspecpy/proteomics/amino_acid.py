import rustms_connector
rmsc = rustms_connector.py_amino_acids


AMINO_ACID_MASSES = rmsc.get_amino_acid_mono_isotopic_masses()
AMINO_ACIDS = rmsc.get_amino_acids()
AMINO_ACID_ATOMIC_COMPOSITIONS = rmsc.get_amino_acid_atomic_compositions()
