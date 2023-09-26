import numpy as np
from numpy.typing import NDArray

import pyims_connector as pims

class TimsSlice:
    def __int__(self):
        pass

    @classmethod
    def from_py_tims_slice(cls, slice: pims.PyTimsSlice):
        """Create a TimsSlice from a PyTimsSlice.

        Args:
            slice (pims.PyTimsSlice): PyTimsSlice to create the TimsSlice from.

        Returns:
            TimsSlice: TimsSlice created from the PyTimsSlice.
        """
        instance = cls.__new__(cls)
        instance.__slice_ptr = slice
        return instance
