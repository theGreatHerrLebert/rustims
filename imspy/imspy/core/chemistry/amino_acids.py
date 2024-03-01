import imspy_connector
ims = imspy_connector.py_amino_acids


AMINO_ACID_MASSES = ims.get_amino_acid_mono_isotopic_masses()
AMINO_ACIDS = ims.get_amino_acids()
AMINO_ACID_ATOMIC_COMPOSITIONS = ims.get_amino_acid_atomic_compositions()
