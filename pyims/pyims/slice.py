import numpy as np
from numpy.typing import NDArray

import pyims_connector as pims


class TimsSlice:
    def __int__(self):
        self.__slice_ptr = None

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

    @property
    def first_frame_id(self) -> int:
        """First frame ID.

        Returns:
            int: First frame ID.
        """
        return self.__slice_ptr.first_frame_id

    @property
    def last_frame_id(self) -> int:
        """Last frame ID.

        Returns:
            int: Last frame ID.
        """
        return self.__slice_ptr.last_frame_id

    def __repr__(self):
        return f"TimsSlice({self.first_frame_id}, {self.last_frame_id})"

    def filter_ranged(self, mz_min: float, mz_max: float, scan_min: int, scan_max: int, intensity_min: float) -> 'TimsSlice':
        """Filter the spectrum by m/z range and intensity.

        Args:
            mz_min (float): Minimum m/z.
            mz_max (float): Maximum m/z.
            intensity_min (float): Minimum intensity.

        Returns:
            NDArray[np.float64]: Filtered spectrum.
        """
        return TimsSlice.from_py_tims_slice(self.__slice_ptr.filter_ranged(mz_min, mz_max, scan_min, scan_max, intensity_min))
