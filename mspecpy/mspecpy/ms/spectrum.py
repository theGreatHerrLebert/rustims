import json
import numpy as np

from typing import List, Tuple
import pandas as pd
from numpy.typing import NDArray

import rustms_connector
rmsc = rustms_connector.py_spectrum


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
        self.__spec_ptr = rmsc.PyMzSpectrum(mz, intensity)

    @classmethod
    def from_jsons(cls, jsons: str) -> 'MzSpectrum':
        json_dict: dict = json.loads(jsons)
        mz = json_dict["mz"]
        intensity = json_dict["intensity"]
        return cls(np.array(mz, dtype=np.float64), np.array(intensity, dtype=np.float64))

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
    def df(self) -> pd.DataFrame:
        """Data.

        Returns:
            pd.DataFrame: Data.
        """

        return pd.DataFrame({'mz': self.mz, 'intensity': self.intensity})

    @classmethod
    def from_py_mz_spectrum(cls, spec: rmsc.PyMzSpectrum):
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

    def __add__(self, other: 'MzSpectrum') -> 'MzSpectrum':
        """Overwrite + operator for adding of spectra

        Args:
            other (MzSpectrum): Other spectrum.

        Returns:
            MzSpectrum: Sum of spectra
        """
        return self.from_py_mz_spectrum(self.__spec_ptr + other.__spec_ptr)

    def __mul__(self, scale) -> 'MzSpectrum':
        """Overwrite * operator for scaling of spectrum

        Args:
            scale (float): Scale.

        Returns:
            MzSpectrum: Scaled spectrum
        """
        return self.from_py_mz_spectrum(self.__spec_ptr * scale)

    def to_windows(self, window_length: float = 10, overlapping: bool = True, min_num_peaks: int = 5,
                   min_intensity: float = 1) -> Tuple[NDArray, List['MzSpectrum']]:
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

    def filter(self, mz_min: float = 0.0, mz_max: float = 2000.0, intensity_min: float = 0.0,
               intensity_max: float = 1e9) -> 'MzSpectrum':
        """Filter the spectrum for a given m/z range and intensity range.

        Args:
            mz_min (float): Minimum m/z value.
            mz_max (float): Maximum m/z value.
            intensity_min (float, optional): Minimum intensity value. Defaults to 0.0.
            intensity_max (float, optional): Maximum intensity value. Defaults to 1e9.

        Returns:
            MzSpectrum: Filtered spectrum.
        """
        return MzSpectrum.from_py_mz_spectrum(
            self.__spec_ptr.filter_ranged(mz_min, mz_max, intensity_min, intensity_max))

    def to_jsons(self) -> str:
        """
        generates json string representation of MzSpectrum
        """
        json_dict = {}
        json_dict["mz"] = self.mz.tolist()
        json_dict["intensity"] = self.intensity.tolist()

        return json.dumps(json_dict)
