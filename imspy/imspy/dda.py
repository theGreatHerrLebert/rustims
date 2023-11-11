import numpy as np
import pandas as pd

import imspy_connector as pims

from imspy.frame import TimsFrame


class FragmentDDA:
    def __init__(self, frame_id: int, precursor_id: int, selected_fragment: TimsFrame):
        self._fragment_ptr = pims.PyTimsFragmentDDA(frame_id, precursor_id, selected_fragment.get_fragment_ptr())

    @classmethod
    def from_py_tims_fragment_dda(cls, fragment: pims.PyTimsFragmentDDA):
        instance = cls.__new__(cls)
        instance._fragment_ptr = fragment
        return instance

    @property
    def frame_id(self) -> int:
        return self._fragment_ptr.frame_id

    @property
    def precursor_id(self) -> int:
        return self._fragment_ptr.precursor_id

    @property
    def selected_fragment(self) -> TimsFrame:
        return TimsFrame.from_py_tims_frame(self._fragment_ptr.selected_fragment)

    def __repr__(self):
        return f"FragmentDDA(frame_id={self.frame_id}, precursor_id={self.precursor_id}, " \
               f"selected_fragment={self.selected_fragment})"

    def get_fragment_ptr(self):
        return self._fragment_ptr

