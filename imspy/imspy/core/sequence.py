from typing import List, Tuple
import imspy_connector as ims


class ProductIon:
    def __init__(self, kind: str, sequence: str, charge: int = 1, intensity: float = 1.0):
        assert kind in ['a', 'b', 'c', 'x', 'y', 'z'], (f"Invalid kind: {kind}, "
                                                        f"must be one of 'a', 'b', 'c', 'x', 'y', 'z'")
        self.__ptr = ims.PyPeptideProductIon(
            kind=kind,
            sequence=sequence,
            charge=charge,
            intensity=intensity
        )

    @property
    def kind(self) -> str:
        return self.__ptr.kind

    @property
    def sequence(self) -> str:
        return self.__ptr.sequence

    @property
    def charge(self) -> int:
        return self.__ptr.charge

    @property
    def intensity(self) -> float:
        return self.__ptr.intensity

    @property
    def mono_isotopic_mass(self) -> float:
        return self.__ptr.mono_isotopic_mass()

    @property
    def mz(self) -> float:
        return self.__ptr.mz

    def atomic_composition(self):
        return self.__ptr.atomic_composition()

    def isotope_distribution(self, mass_tolerance: float = 1e-3, abundance_threshold: float = 1e-8,
                             max_result: int = 200, intensity_min: float = 1e-4) -> List[Tuple[float, float]]:
        return self.__ptr.isotope_distribution(mass_tolerance, abundance_threshold, max_result, intensity_min)

    def get_ptr(self):
        return self.__ptr

    @classmethod
    def from_py_ptr(cls, product_ion: ims.PyPeptideProductIon):
        instance = cls.__new__(cls)
        instance.__ptr = product_ion
        return instance

    def __repr__(self):
        return (f"ProductIon(kind={self.kind}, sequence={self.sequence}, charge={self.charge}, mz={self.mz}, "
                f" intensity={self.intensity})")


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

    def to_tokens(self, group_modifications: bool = True) -> List[str]:
        return self.__ptr.to_tokens(group_modifications)

    def to_sage_representation(self) -> Tuple[str, List[float]]:
        return self.__ptr.to_sage_representation()

    def get_ptr(self):
        return self.__ptr

    def calculate_b_y_product_ion_series(self, charge: int = 1) -> Tuple[List[ProductIon], List[ProductIon]]:
        b_ions, y_ions = self.__ptr.calculate_b_y_product_ion_series(charge)
        return [ProductIon.from_py_ptr(ion) for ion in b_ions], [ProductIon.from_py_ptr(ion) for ion in y_ions][::-1]

    @classmethod
    def fom_py_ptr(cls, seq: ims.PyPeptideSequence):
        instance = cls.__new__(cls)
        instance.__ptr = seq
        return instance

    def __repr__(self):
        return f"AminoAcidSequence(sequence={self.sequence}, mono_isotopic_mass={self.mono_isotopic_mass})"
