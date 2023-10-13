import numpy as np
from typing import List, Tuple
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

    def __repr__(self):
        return f"MzSpectrum(num_peaks={len(self.mz)})"

    def to_windows(self, window_length: float = 10, overlapping: bool = True, min_num_peaks: int = 5, min_intensity: float = 1) -> Tuple[NDArray, List['MzSpectrum']]:
        """Convert the spectrum to a list of windows.

        Args:
            window_length (float, optional): Window length. Defaults to 10.
            overlapping (bool, optional): Whether the windows should overlap. Defaults to True.
            min_num_peaks (int, optional): Minimum number of peaks in a window. Defaults to 5.
            min_intensity (float, optional): Minimum intensity of a peak in a window. Defaults to 1.

        Returns:
            Tuple[NDArray, List[MzSpectrum]]: List of windows.
        """

        indices, windows = self.__spec_ptr.to_windows(window_length, overlapping, min_num_peaks, min_intensity)
        return indices, [MzSpectrum.from_py_mz_spectrum(window) for window in windows]

    def filter(self, mz_min: float = 0.0, mz_max: float = 2000.0, intensity_min: float = 0.0, intensity_max: float = 1e9) -> 'MzSpectrum':
        """Filter the spectrum for a given m/z range and intensity range.

        Args:
            mz_min (float): Minimum m/z value.
            mz_max (float): Maximum m/z value.
            intensity_min (float, optional): Minimum intensity value. Defaults to 0.0.
            intensity_max (float, optional): Maximum intensity value. Defaults to 1e9.

        Returns:
            MzSpectrum: Filtered spectrum.
        """
        return MzSpectrum.from_py_mz_spectrum(self.__spec_ptr.filter_ranged(mz_min, mz_max, intensity_min, intensity_max))

    def vectorized(self, resolution: int = 2) -> 'MzSpectrumVectorized':
        """Convert the spectrum to a vectorized spectrum.

        Args:
            resolution (int, optional): Resolution. Defaults to 2.

        Returns:
            MzSpectrumVectorized: Vectorized spectrum.
        """
        return MzSpectrumVectorized.from_py_mz_spectrum_vectorized(self.__spec_ptr.vectorized(resolution))

    def __repr__(self):
        return f"MzSpectrum(num_peaks={len(self.mz)})"


class MzSpectrumVectorized:
    def __init__(self, indices: NDArray[np.int32], values: NDArray[np.float64], resolution: int):
        """MzSpectrum class.

        Args:
            mz (NDArray[np.float64]): m/z.
            values (NDArray[np.float64]): Intensity.

        Raises:
            AssertionError: If the length of the mz and intensity arrays are not equal.
        """
        assert len(indices) == len(values), "The length of the mz and intensity arrays must be equal."
        self.__spec_ptr = pims.PyMzSpectrumVectorized(indices, values, resolution)

    @classmethod
    def from_py_mz_spectrum_vectorized(cls, spec: pims.PyMzSpectrumVectorized):
        """Create a MzSpectrum from a PyMzSpectrum.

        Args:
            spec (pims.PyMzSpectrum): PyMzSpectrum to create the MzSpectrum from.

        Returns:
            MzSpectrum: MzSpectrum created from the PyMzSpectrum.
        """
        instance = cls.__new__(cls)
        instance.__spec_ptr = spec
        return instance

    @property
    def resolution(self) -> float:
        """Resolution.

        Returns:
            float: Resolution.
        """
        return self.__spec_ptr.resolution

    @property
    def indices(self) -> NDArray[np.int32]:
        """m/z.

        Returns:
            NDArray[np.float64]: m/z.
        """
        return self.__spec_ptr.indices

    @property
    def values(self) -> NDArray[np.float64]:
        """Intensity.

        Returns:
            NDArray[np.float64]: Intensity.
        """
        return self.__spec_ptr.values

    def __repr__(self):
        return f"MzSpectrumVectorized(num_values={len(self.values)})"


class TimsSpectrum:
    def __init__(self, frame_id: int, scan: int, retention_time: float, mobility: float, ms_type: int, index: NDArray[np.int32], mz: NDArray[np.float64], intensity: NDArray[np.float64]):
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
        self.__spec_ptr = pims.PyTimsSpectrum(frame_id, scan, retention_time, mobility, ms_type, index, mz, intensity)

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
    def mobility(self) -> float:
        """Inverse mobility.

        Returns:
            float: Inverse mobility.
        """
        return self.__spec_ptr.mobility

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

    @property
    def mz_spectrum(self) -> MzSpectrum:
        """Get the MzSpectrum.

        Returns:
            MzSpectrum: Spectrum.
        """
        return MzSpectrum.from_py_mz_spectrum(self.__spec_ptr.mz_spectrum)

    def __repr__(self):
        return (f"TimsSpectrum(id={self.frame_id}, retention_time={np.round(self.retention_time, 2)}, "
                f"scan={self.scan}, mobility={np.round(self.mobility, 2)}, ms_type={self.ms_type}, "
                f"num_peaks={len(self.index)})")
