import numpy as np
import pandas as pd

import imspy_connector as pims

from imspy.spectrum import IndexedMzSpectrum


class FragmentDDA:
    def __init__(self, precursor_id, selected_fragment: IndexedMzSpectrum):
        self._fragment_ptr = pims.PyTimsFragmentDDA(precursor_id, selected_fragment.get_spec_ptr())

    @classmethod
    def from_py_tims_fragment_dda(cls, fragment: pims.PyTimsFragmentDDA):
        instance = cls.__new__(cls)
        instance._fragment_ptr = fragment
        return instance

    @property
    def precursor_id(self) -> int:
        return self._fragment_ptr.precursor_id

    @property
    def selected_fragment(self) -> IndexedMzSpectrum:
        return IndexedMzSpectrum.from_py_indexed_mz_spectrum(self._fragment_ptr.selected_fragment)

    def __repr__(self):
        return f"FragmentDDA({self.precursor_id}, {self.selected_fragment})"

    def get_fragment_ptr(self):
        return self._fragment_ptr

