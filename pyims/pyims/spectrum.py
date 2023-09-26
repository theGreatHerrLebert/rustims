import numpy as np
from numpy.typing import NDArray

import pyims_connector as pims


class MzSpectrum:
    def __init__(self, mz: NDArray[np.float64], intensity: NDArray[np.float64]):
        """MzSpectrum class.

        Args:
            mz (NDArray[np.float64]): m/z.
            intensity (NDArray[np.float64]): Intensity.

        Raises:
            AssertionError: If the length of the mz and intensity arrays are not equal.
        """
        assert len(mz) == len(intensity), "The length of the mz and intensity arrays must be equal."
        self.__spec_ptr = pims.PyMzSpectrum(mz, intensity)

    @classmethod
    def from_py_mz_spectrum(cls, spec: pims.PyMzSpectrum):
        """Create a MzSpectrum from a PyMzSpectrum.

        Args:
            spec (pims.PyMzSpectrum): PyMzSpectrum to create the MzSpectrum from.

        Returns:
            MzSpectrum: MzSpectrum created from the PyMzSpectrum.
        """
        instance = cls.__new__(cls)
        instance.__spec_ptr = spec
        return instance

    def filter_ranged(self, mz_min: float, mz_max: float, intensity_min: float = 0.0) -> 'MzSpectrum':
        """Filter the spectrum for a given m/z range and intensity range.

        Args:
            mz_min (float): Minimum m/z value.
            mz_max (float): Maximum m/z value.
            intensity_min (float, optional): Minimum intensity value. Defaults to 0.0.

        Returns:
            MzSpectrum: Filtered spectrum.
        """
        return MzSpectrum.from_py_mz_spectrum(self.__spec_ptr.filter_ranged(mz_min, mz_max, intensity_min))


class TimsSpectrum:
    def __init__(self, frame_id: int, scan: int, retention_time: float, inv_mobility: float, ms_type: int, index: NDArray[np.int32], mz: NDArray[np.float64], intensity: NDArray[np.float64]):
        """TimsSpectrum class.

        Args:
            index (NDArray[np.int32]): Index.
            mz (NDArray[np.float64]): m/z.
            intensity (NDArray[np.float64]): Intensity.

        Raises:
            AssertionError: If the length of the index, mz and intensity arrays are not equal.
        """
        assert len(index) == len(mz) == len(intensity), ("The length of the index, mz and intensity arrays must be "
                                                         "equal.")
        self.__spec_ptr = pims.PyTimsSpectrum(frame_id, scan, retention_time, inv_mobility, ms_type, index, mz, intensity)

    @classmethod
    def from_py_tims_spectrum(cls, spec: pims.PyTimsSpectrum):
        """Create a TimsSpectrum from a PyTimsSpectrum.

        Args:
            spec (pims.PyTimsSpectrum): PyTimsSpectrum to create the TimsSpectrum from.

        Returns:
            TimsSpectrum: TimsSpectrum created from the PyTimsSpectrum.
        """
        instance = cls.__new__(cls)
        instance.__spec_ptr = spec
        return instance

    @property
    def index(self) -> NDArray[np.int32]:
        """Index.

        Returns:
            NDArray[np.int32]: Index.
        """
        return self.__spec_ptr.index

    @property
    def mz(self) -> NDArray[np.float64]:
        """m/z.

        Returns:
            NDArray[np.float64]: m/z.
        """
        return self.__spec_ptr.mz

    @property
    def intensity(self) -> NDArray[np.float64]:
        """Intensity.

        Returns:
            NDArray[np.float64]: Intensity.
        """
        return self.__spec_ptr.intensity

    @property
    def ms_type(self) -> str:
        """MS type.

        Returns:
            str: MS type.
        """
        return self.__spec_ptr.ms_type

    @property
    def inv_mobility(self) -> float:
        """Inverse mobility.

        Returns:
            float: Inverse mobility.
        """
        return self.__spec_ptr.inv_mobility

    @property
    def scan(self) -> int:
        """Scan.

        Returns:
            int: Scan.
        """
        return self.__spec_ptr.scan

    @property
    def retention_time(self) -> float:
        """Retention time.

        Returns:
            float: Retention time.
        """
        return self.__spec_ptr.retention_time

    @property
    def frame_id(self) -> int:
        """Frame ID.

        Returns:
            int: Frame ID.
        """
        return self.__spec_ptr.frame_id

    def __repr__(self):
        return f"TimsSpectrum(id={self.frame_id}, retention_time={np.round(self.retention_time, 2)}, scan={self.scan}, inv_mobility={np.round(self.inv_mobility, 2)}, ms_type={self.ms_type}, num_peaks={len(self.index)})"
