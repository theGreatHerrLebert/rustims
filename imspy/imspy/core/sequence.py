import imspy_connector as ims

from imspy import MzSpectrum


class AminoAcidSequence:
    def __init__(self, sequence: str):
        self.__ptr = ims.PyAminoAcidSequence(sequence)

    @property
    def sequence(self) -> str:
        return self.__ptr.sequence

    @property
    def monoisotopic_mass(self) -> float:
        return self.__ptr.monoisotopic_mass

    def get_mz(self, charge: int) -> float:
        return self.__ptr.get_mz(charge)

    def get_ptr(self):
        return self.__ptr

    @classmethod
    def fom_py_ptr(cls, seq: ims.PyAminoAcidSequence):
        instance = cls.__new__(cls)
        instance.__ptr = seq
        return instance

    def precursor_spectrum_averagine(self, charge: int = 1, min_intensity: int = 1, k: int = 10,
                                     resolution: int = 3, centroid: bool = True) -> MzSpectrum:
        return MzSpectrum.from_py_mz_spectrum(self.__ptr.precursor_spectrum_averagine(
            charge, min_intensity, k, resolution, centroid
        ))

    def precursor_spectrum_from_atomic_composition(self, charge,
                                                   mass_tolerance: float = 1e-6,
                                                   abundance_threshold: float = 1e-7,
                                                   max_result: int = 200) -> MzSpectrum:
        return MzSpectrum.from_py_mz_spectrum(self.__ptr.precursor_spectrum_from_atomic_composition(
            charge, mass_tolerance, abundance_threshold, max_result
        ))

    def get_atomic_composition(self):
        return self.__ptr.get_atomic_composition()

    def __repr__(self):
        return f"AminoAcidSequence(sequence={self.sequence}, monoisotopic_mass={self.monoisotopic_mass})"
