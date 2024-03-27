import imspy_connector

from imspy.data.spectrum import MzSpectrum

ims = imspy_connector.py_sumformula


class SumFormula:
    def __init__(self, sum_formula: str):
        self.__ptr = ims.PySumFormula(sum_formula)

    @property
    def formula(self) -> str:
        return self.__ptr.formula

    @property
    def formula_dict(self) -> dict:
        return self.__ptr.elements

    @property
    def monoisotopic_mass(self) -> float:
        return self.__ptr.monoisotopic_mass

    def __repr__(self):
        return f"SumFormula('{self.formula}')"

    @classmethod
    def from_py_ptr(cls, py_ptr):
        instance = cls.__new__(cls)
        instance.__ptr = py_ptr
        return instance

    def get_py_ptr(self):
        return self.__ptr

    def generate_isotope_distribution(self, charge: int) -> 'MzSpectrum':
        return MzSpectrum.from_py_ptr(self.__ptr.generate_isotope_distribution(charge))
