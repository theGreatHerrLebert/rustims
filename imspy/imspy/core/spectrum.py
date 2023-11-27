from __future__ import annotations
import json
import numpy as np
from typing import List, Tuple, Callable
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import find_peaks
import imspy_connector as pims


def get_peak_integral(peaks: NDArray[np.int32], peak_info: dict) -> NDArray[np.float64]:
    """Calculates the integral of the peaks in a spectrum.

    Args:
        peaks (NDArray[np.int32]): Peak indices.
        peak_info (dict): Peak info.

    Returns:
        NDArray[np.float64]: Peak integrals.
    """
    integrals = np.zeros(len(peaks), dtype=np.float64)
    FWHM = peak_info['widths']
    h = peak_info['prominences']
    integrals = np.sqrt(2*np.pi) * h * FWHM / (2*np.sqrt(2*np.log(2)))
    return integrals


class IndexedMzSpectrum:
    def __init__(self, index: NDArray[np.int32], mz: NDArray[np.float64], intensity: NDArray[np.float64]):
        """IndexedMzSpectrum class.

        Args:
            index (NDArray[np.int32]): Index.
            mz (NDArray[np.float64]): m/z.
            intensity (NDArray[np.float64]): Intensity.

        Raises:
            AssertionError: If the length of the index, mz and intensity arrays are not equal.
        """
        assert len(index) == len(mz) == len(intensity), ("The length of the index, mz and intensity arrays must be "
                                                         "equal.")
        self.__spec_ptr = pims.PyIndexedMzSpectrum(index, mz, intensity)

    @classmethod
    def from_py_indexed_mz_spectrum(cls, spec: pims.PyIndexedMzSpectrum):
        """Create a IndexedMzSpectrum from a PyIndexedMzSpectrum.

        Args:
            spec (pims.PyIndexedMzSpectrum): PyIndexedMzSpectrum to create the IndexedMzSpectrum from.

        Returns:
            IndexedMzSpectrum: IndexedMzSpectrum created from the PyIndexedMzSpectrum.
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

    def filter(self, mz_min: float = 0.0, mz_max: float = 2000.0, intensity_min: float = 0.0,
               intensity_max: float = 1e9) -> 'IndexedMzSpectrum':
        """Filter the spectrum for a given m/z range and intensity range.

        Args:
            mz_min (float): Minimum m/z value.
            mz_max (float): Maximum m/z value.
            intensity_min (float, optional): Minimum intensity value. Defaults to 0.0.
            intensity_max (float, optional): Maximum intensity value. Defaults to 1e9.

        Returns:
            IndexedMzSpectrum: Filtered spectrum.
        """
        return IndexedMzSpectrum.from_py_indexed_mz_spectrum(
            self.__spec_ptr.filter_ranged(mz_min, mz_max, intensity_min, intensity_max))

    @property
    def df(self) -> pd.DataFrame:
        """Data.

        Returns:
            pd.DataFrame: Data.
        """

        return pd.DataFrame({'index': self.index, 'mz': self.mz, 'intensity': self.intensity})

    def get_spec_ptr(self) -> pims.PyIndexedMzSpectrum:
        """Get the spec_ptr.

        Returns:
            pims.PyIndexedMzSpectrum: spec_ptr.
        """
        return self.__spec_ptr

    def __repr__(self):
        return f"IndexedMzSpectrum(num_peaks={len(self.index)})"


class MzSpectrum:
    
    @classmethod
    def from_jsons(cls, jsons: str) -> MzSpectrum:
        json_dict:dict = json.loads(jsons)
        mz = json_dict["mz"]
        intensity = json_dict["intensity"]
        return cls(np.array(mz, dtype=np.float64), np.array(intensity, dtype=np.float64))
    
    @classmethod
    def from_mz_spectra_list(cls, spectra_list:List[MzSpectrum], resolution: int)->MzSpectrum:
        """Generates a convoluted mass spectrum by adding all spectra in the given list.

        Args:
            spectra_list (List[MzSpectrum]): List of mass spectra.
            resolution (int): Desired resolution of returned spectrum.

        Returns:
            MzSpectrum: Convoluted spectrum.
        """
        return cls.from_py_mz_spectrum(pims.PyMzSpectrum.from_mzspectra_list([spectrum.__spec_ptr for spectrum in spectra_list], resolution))
    
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

    @property
    def df(self) -> pd.DataFrame:
        """Data.

        Returns:
            pd.DataFrame: Data.
        """

        return pd.DataFrame({'mz': self.mz, 'intensity': self.intensity})

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
    
    def __mul__(self, scale) -> MzSpectrum:
        """Overwrite * operator for scaling of spectrum

        Args:
            scale (float): Scale.

        Returns:
            MzSpectrum: Scaled spectrum
        """
        tmp: pims.PyMzSpectrum =  self.__spec_ptr * scale
        return self.from_py_mz_spectrum(tmp)

    def to_windows(self, window_length: float = 10, overlapping: bool = True, min_num_peaks: int = 5,
                   min_intensity: float = 1) -> Tuple[NDArray, List[MzSpectrum]]:
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

    def to_resolution(self, resolution: int) -> MzSpectrum:
        """Bins the spectrum's m/z values to a 
        given resolution and sums the intensities.

        Args:
            resolution (int): Negative decadic logarithm of bin size.

        Returns:
            MzSpectrum: A new `MzSpectrum` where m/z values are binned according to the given resolution.
        """
        return MzSpectrum.from_py_mz_spectrum(self.__spec_ptr.to_resolution(resolution))
    
    def filter(self, mz_min: float = 0.0, mz_max: float = 2000.0, intensity_min: float = 0.0,
               intensity_max: float = 1e9) -> MzSpectrum:
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

    def vectorized(self, resolution: int = 2) -> MzSpectrumVectorized:
        """Convert the spectrum to a vectorized spectrum.

        Args:
            resolution (int, optional): Resolution. Defaults to 2.

        Returns:
            MzSpectrumVectorized: Vectorized spectrum.
        """
        return MzSpectrumVectorized.from_py_mz_spectrum_vectorized(self.__spec_ptr.vectorized(resolution))
    
    def to_jsons(self) -> str:
        """
        generates json string representation of MzSpectrum
        """
        json_dict = {}
        json_dict["mz"] = self.mz.tolist()
        json_dict["intensity"] = self.intensity.tolist()

        return json.dumps(json_dict)

    
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

    def to_centroided(self, integrate_method: Callable = get_peak_integral) -> MzSpectrum:
        """Convert the spectrum to a centroided spectrum.

        Returns:
            MzSpectrum: Centroided spectrum.
        """
        # first generate dense spectrum
        dense_spectrum = MzSpectrumVectorized.from_py_mz_spectrum_vectorized(self.__spec_ptr.to_dense_spectrum(None))
        # find peaks in the dense spectrum and widths with scipy
        peaks, peak_info = find_peaks(dense_spectrum.values, height=0, width=(0,0.5))
        # then get the peak integrals
        integrals = integrate_method(peaks, peak_info)
        # then create a new spectrum with the peak indices and the integrals
        return MzSpectrum.from_py_mz_spectrum(pims.PyMzSpectrum(dense_spectrum.indices[peaks]/np.power(10,dense_spectrum.resolution), integrals))

    def __repr__(self):
        return f"MzSpectrumVectorized(num_values={len(self.values)})"


class TimsSpectrum:
    def __init__(self, frame_id: int, scan: int, retention_time: float, mobility: float, ms_type: int,
                 index: NDArray[np.int32], mz: NDArray[np.float64], intensity: NDArray[np.float64]):
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

    @property
    def df(self) -> pd.DataFrame:
        """Data.
        
        Returns:
            pd.DataFrame: Data.
        """

        return pd.DataFrame({
            'frame': np.repeat(self.frame_id, len(self.index)),
            'retention_time': np.repeat(self.retention_time, len(self.index)),
            'scan': np.repeat(self.scan, len(self.index)),
            'mobility': np.repeat(self.mobility, len(self.index)),
            'tof': self.index,
            'mz': self.mz,
            'intensity': self.intensity
        })

    def __repr__(self):
        return (f"TimsSpectrum(id={self.frame_id}, retention_time={np.round(self.retention_time, 2)}, "
                f"scan={self.scan}, mobility={np.round(self.mobility, 2)}, ms_type={self.ms_type}, "
                f"num_peaks={len(self.index)})")

    def get_spec_ptr(self) -> pims.PyTimsSpectrum:
        """Get the spec_ptr.

        Returns:
            pims.PyTimsSpectrum: spec_ptr.
        """
        return self.__spec_ptr
