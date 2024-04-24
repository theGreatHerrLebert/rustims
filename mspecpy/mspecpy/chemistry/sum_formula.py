from mspecpy.ms.spectrum import MzSpectrum
from mspecpy.proteomics.peptide import RustWrapper

import rustms_connector
rmsc = rustms_connector.py_sumformula


class SumFormula(RustWrapper):
    def __init__(self, sum_formula: str):
        self.__py_ptr = rmsc.PySumFormula(sum_formula)

    @property
    def formula(self) -> str:
        return self.__py_ptr.formula

    @property
    def formula_dict(self) -> dict:
        return self.__py_ptr.elements

    @property
    def monoisotopic_mass(self) -> float:
        return self.__py_ptr.monoisotopic_mass

    def __repr__(self):
        return f"SumFormula('{self.formula}')"

    @classmethod
    def from_py_ptr(cls, py_ptr):
        instance = cls.__new__(cls)
        instance.__py_ptr = py_ptr
        return instance

    def get_py_ptr(self):
        return self.__py_ptr

    def generate_isotope_distribution(self, charge: int) -> 'MzSpectrum':
        return MzSpectrum.from_py_ptr(self.__py_ptr.generate_isotope_distribution(charge))
