import imspy_connector as ims

from imspy import MzSpectrum


class PeptideSequence:
    def __init__(self, sequence: str):
        self.__ptr = ims.PyPeptideSequence(sequence)

    @property
    def sequence(self) -> str:
        return self.__ptr.sequence

    @property
    def mono_isotopic_mass(self) -> float:
        return self.__ptr.mono_isotopic_mass

    @property
    def atomic_composition(self):
        return self.__ptr.atomic_composition()

    def get_ptr(self):
        return self.__ptr

    @classmethod
    def fom_py_ptr(cls, seq: ims.PyPeptideSequence):
        instance = cls.__new__(cls)
        instance.__ptr = seq
        return instance

    def __repr__(self):
        return f"AminoAcidSequence(sequence={self.sequence}, mono_isotopic_mass={self.mono_isotopic_mass})"
