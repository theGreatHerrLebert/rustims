import pandas as pd

from typing import List, Tuple, Optional, TYPE_CHECKING
from numpy.typing import NDArray

import numpy as np
from imspy.data.spectrum import TimsSpectrum, IndexedMzSpectrum
from imspy.core.base import RustWrapperObject
from imspy.utility.utilities import re_index_indices

if TYPE_CHECKING:
    from imspy.simulation.annotation import TimsFrameAnnotated

import imspy_connector
ims = imspy_connector.py_tims_frame


def _get_tims_frame_annotated():
    """Lazy import of TimsFrameAnnotated to avoid circular imports."""
    from imspy.simulation.annotation import TimsFrameAnnotated
    return TimsFrameAnnotated


class TimsFrame(RustWrapperObject):
    def __init__(self, frame_id: int, ms_type: int, retention_time: float, scan: NDArray[np.int32],
                 mobility: NDArray[np.float64], tof: NDArray[np.int32],
                 mz: NDArray[np.float64], intensity: NDArray[np.float64]):
        """TimsFrame class.

        Args:
            frame_id (int): Frame ID.
            ms_type (int): MS type.
            retention_time (float): Retention time.
            scan (NDArray[np.int32]): Scan.
            mobility (NDArray[np.float64]): Inverse mobility.
            tof (NDArray[np.int32]): Time of flight.
            mz (NDArray[np.float64]): m/z.
            intensity (NDArray[np.float64]): Intensity.

        Raises:
            AssertionError: If the length of the scan, mobility, tof, mz and intensity arrays are not equal.
        """

        assert len(scan) == len(mobility) == len(tof) == len(mz) == len(intensity), \
            "The length of the scan, mobility, tof, mz and intensity arrays must be equal."

        self.__frame_ptr = ims.PyTimsFrame(frame_id, ms_type, retention_time, scan, mobility, tof, mz, intensity)

    def __add__(self, other: 'TimsFrame') -> 'TimsFrame':
        """Add two TimsFrames together.

        Args:
            other (TimsFrame): TimsFrame to add.

        Returns:
            TimsFrame: Sum of the two TimsFrames.
        """
        return TimsFrame.from_py_ptr(self.__frame_ptr + other.__frame_ptr)

    @classmethod
    def from_py_ptr(cls, frame: ims.PyTimsFrame):
        """Create a TimsFrame from a PyTimsFrame.

        Args:
            frame (pims.PyTimsFrame): PyTimsFrame to create the TimsFrame from.

        Returns:
            TimsFrame: TimsFrame created from the PyTimsFrame.
        """
        instance = cls.__new__(cls)
        instance.__frame_ptr = frame
        return instance

    @property
    def frame_id(self) -> int:
        """Frame ID.

        Returns:
            int: Frame ID.
        """
        return self.__frame_ptr.frame_id

    @property
    def ms_type_as_string(self) -> str:
        """MS type.

        Returns:
            int: MS type.
        """
        return self.__frame_ptr.ms_type

    @property
    def ms_type(self) -> int:
        """MS type.

        Returns:
            int: MS type.
        """
        return self.__frame_ptr.ms_type_numeric

    @property
    def retention_time(self) -> float:
        """Retention time.

        Returns:
            float: Retention time.
        """
        return self.__frame_ptr.retention_time

    @property
    def scan(self) -> NDArray[np.int32]:
        """Scan.

        Returns:
            NDArray[np.int32]: Scan.
        """
        return self.__frame_ptr.scan

    @property
    def mobility(self) -> NDArray[np.float64]:
        """Inverse mobility.

        Returns:
            NDArray[np.float64]: Inverse mobility.
        """
        return self.__frame_ptr.mobility

    @property
    def tof(self) -> NDArray[np.int32]:
        """Time of flight.

        Returns:
            NDArray[np.int32]: Time of flight.
        """
        return self.__frame_ptr.tof

    @tof.setter
    def tof(self, tof: NDArray[np.int32]):
        self.__frame_ptr.tof = tof

    @property
    def mz(self) -> NDArray[np.float64]:
        """m/z.

        Returns:
            NDArray[np.float64]: m/z.
        """
        return self.__frame_ptr.mz

    @property
    def intensity(self) -> NDArray[np.float64]:
        """Intensity.

        Returns:
            NDArray[np.float64]: Intensity.
        """
        return self.__frame_ptr.intensity

    @property
    def df(self) -> pd.DataFrame:
        """ Data as a pandas DataFrame.

        Returns:
            pd.DataFrame: Data.
        """

        return pd.DataFrame({
            'frame': np.repeat(self.frame_id, len(self.scan)),
            'retention_time': np.repeat(self.retention_time, len(self.scan)),
            'scan': self.scan,
            'mobility': self.mobility,
            'tof': self.tof,
            'mz': self.mz,
            'intensity': self.intensity})

    def filter(self,
               mz_min: float = 0.0,
               mz_max: float = 2000.0,
               scan_min: int = 0,
               scan_max: int = 1000,
               mobility_min: float = 0.0,
               mobility_max: float = 2.0,
               intensity_min: float = 0.0,
               intensity_max: float = 1e9,
               tof_min: int = 0,
               tof_max: int = int(1e9),
               ) -> 'TimsFrame':
        """Filter the frame for a given m/z range, scan range and intensity range.

        Args:
            mz_min (float): Minimum m/z value.
            mz_max (float): Maximum m/z value.
            scan_min (int, optional): Minimum scan value. Defaults to 0.
            scan_max (int, optional): Maximum scan value. Defaults to 1000.
            mobility_min (float, optional): Minimum inverse mobility value. Defaults to 0.0.
            mobility_max (float, optional): Maximum inverse mobility value. Defaults to 2.0.
            intensity_min (float, optional): Minimum intensity value. Defaults to 0.0.
            intensity_max (float, optional): Maximum intensity value. Defaults to 1e9.
            tof_min (int, optional): Minimum tof value. Defaults to 0.
            tof_max (int, optional): Maximum tof value. Defaults to 1e6.

        Returns:
            TimsFrame: Filtered frame.
        """

        return TimsFrame.from_py_ptr(self.__frame_ptr.filter_ranged(mz_min, mz_max, scan_min, scan_max, mobility_min, mobility_max,
                                                                    intensity_min, intensity_max, tof_min, tof_max))

    def to_indexed_mz_spectrum(self) -> 'IndexedMzSpectrum':
        """Convert the frame to an IndexedMzSpectrum.

        Returns:
            IndexedMzSpectrum: IndexedMzSpectrum.
        """
        return IndexedMzSpectrum.from_py_ptr(self.__frame_ptr.to_indexed_mz_spectrum())

    def to_resolution(self, resolution: int) -> 'TimsFrame':
        """Convert the frame to a given resolution.

        Args:
            resolution (int): Resolution.

        Returns:
            TimsFrame: Frame with the given resolution.
        """
        return TimsFrame.from_py_ptr(self.__frame_ptr.to_resolution(resolution))

    def vectorized(self, resolution: int = 2) -> 'TimsFrameVectorized':
        """Convert the frame to a vectorized frame.

        Args:
            resolution (int, optional): Resolution. Defaults to 2.

        Returns:
            TimsFrameVectorized: Vectorized frame.
        """
        return TimsFrameVectorized.from_py_ptr(self.__frame_ptr.vectorized(resolution))

    def to_tims_spectra(self) -> List['TimsSpectrum']:
        """Convert the frame to a list of TimsSpectrum.

        Returns:
            List[TimsSpectrum]: List of TimsSpectrum.
        """
        return [TimsSpectrum.from_py_tims_spectrum(spec) for spec in self.__frame_ptr.to_tims_spectra()]

    def to_windows(self, window_length: float = 10, overlapping: bool = True, min_num_peaks: int = 5,
                   min_intensity: float = 1) -> List[TimsSpectrum]:
        """Convert the frame to a list of windows.

        Args:
            window_length (float, optional): Window length. Defaults to 10.
            overlapping (bool, optional): Whether the windows should overlap. Defaults to True.
            min_num_peaks (int, optional): Minimum number of peaks in a window. Defaults to 5.
            min_intensity (float, optional): Minimum intensity of a peak in a window. Defaults to 1.

        Returns:
            List[MzSpectrum]: List of windows.
        """
        return [TimsSpectrum.from_py_tims_spectrum(spec) for spec in self.__frame_ptr.to_windows(
            window_length, overlapping, min_num_peaks, min_intensity)]

    @classmethod
    def from_windows(cls, windows: List[TimsSpectrum]) -> 'TimsFrame':
        """Create a TimsFrame from a list of windows.

        Args:
            windows (List[TimsSpectrum]): List of windows.

        Returns:
            TimsFrame: TimsFrame created from the windows.
        """
        return TimsFrame.from_py_ptr(ims.PyTimsFrame.from_windows(
            [spec.get_py_ptr() for spec in windows]
        ))

    @classmethod
    def from_tims_spectra(cls, spectra: List[TimsSpectrum]) -> 'TimsFrame':
        """Create a TimsFrame from a list of TimsSpectrum.

        Args:
            spectra (List[TimsSpectrum]): List of TimsSpectrum.

        Returns:
            TimsFrame: TimsFrame created from the TimsSpectrum.
        """
        return TimsFrame.from_py_ptr(ims.PyTimsFrame.from_tims_spectra(
            [spec.get_py_ptr() for spec in spectra]
        ))

    def to_dense_windows(self, window_length: float = 10, resolution: int = 1, overlapping: bool = True,
                         min_num_peaks: int = 5, min_intensity: float = 0.0):

        rows, cols, values, ims, mzs, scans, window_indices = self.__frame_ptr.to_dense_windows(window_length, resolution,
                                                                                      overlapping, min_num_peaks,
                                                                                      min_intensity)

        return ims, mzs, scans, window_indices, np.reshape(values, (rows, cols))

    def to_noise_annotated_tims_frame(self) -> 'TimsFrameAnnotated':
        """Convert the frame to a noise annotated frame.

        Returns:
            TimsFrameAnnotated: Noise annotated frame.
        """
        TimsFrameAnnotated = _get_tims_frame_annotated()
        return TimsFrameAnnotated.from_py_ptr(self.__frame_ptr.to_noise_annotated_tims_frame())

    def get_inverse_mobility_along_scan_marginal(self) -> float:
        """Get the inverse mobility along the scan marginal.

        Returns:
            float: Inverse mobility.
        """
        return self.__frame_ptr.get_inverse_mobility_along_scan_marginal()

    def get_mobility_mean_and_variance(self) -> Tuple[float, float]:
        """Get the mean and variance of the inverse mobility.

        Returns:
            Tuple[float, float]: Mean and variance of the inverse mobility.
        """
        return self.__frame_ptr.get_mobility_mean_and_variance()

    def get_py_ptr(self):
        return self.__frame_ptr

    def __repr__(self):
        return (f"TimsFrame(frame_id={self.__frame_ptr.frame_id}, ms_type={self.__frame_ptr.ms_type}, "
                f"num_peaks={len(self.__frame_ptr.mz)}, intensity_sum={np.round(np.sum(self.__frame_ptr.intensity))})")

    def random_subsample_frame(self, take_probability: float) -> 'TimsFrame':
        """Randomly subsample the frame.

            Args:
            take_probability (float): Take probability.

            Returns:
            TimsFrame: Subsampled frame.
        """

        assert 0.0 <= take_probability <= 1.0, "The take probability must be between 0 and 1."
        return TimsFrame.from_py_ptr(self.__frame_ptr.random_subsample_frame(take_probability))

    def fold_along_scan_axis(self, fold_width: int = 4) -> 'TimsFrame':
        """Fold the frame along the scan axis.

        Args:
            fold_width (int): Width of the fold. Default to 4.

        Returns:
            TimsFrame: Folded frame.
        """
        return TimsFrame.from_py_ptr(self.__frame_ptr.fold_along_scan_axis(fold_width))

    def __getitem__(self, index: int) -> Optional[TimsSpectrum]:
        """Get the TimsSpectrum at a given index.

        Args:
            index (int): Index.

        Returns:
            TimsSpectrum: TimsSpectrum at the index, or None if the index is out of bounds.
        """
        maybe_spectrum = self.__frame_ptr.get_tims_spectrum(index)
        if maybe_spectrum is None:
            return None
        else:
            return TimsSpectrum.from_py_ptr(maybe_spectrum)


class TimsFrameVectorized(RustWrapperObject):
    def __init__(self, frame_id: int, ms_type: int, retention_time: float, scan: NDArray[np.int32],
                 mobility: NDArray[np.float64], tof: NDArray[np.int32],
                 indices: NDArray[np.int32], intensity: NDArray[np.float64]):
        """TimsFrameVectorized class.

        Args:
            frame_id (int): Frame ID.
            ms_type (int): MS type.
            retention_time (float): Retention time.
            scan (NDArray[np.int32]): Scan.
            mobility (NDArray[np.float64]): Inverse mobility.
            tof (NDArray[np.int32]): Time of flight.
            indices (NDArray[np.int32]): Indices.
            intensity (NDArray[np.float64]): Intensity.

        Raises:
            AssertionError: If the length of the scan, mobility, tof, indices and intensity arrays are not equal.
        """

        assert len(scan) == len(mobility) == len(tof) == len(indices) == len(intensity), \
            "The length of the scan, mobility, tof, indices and intensity arrays must be equal."

        self.__frame_ptr = ims.PyTimsFrameVectorized(frame_id, ms_type, retention_time, scan, mobility, tof, indices,
                                                      intensity)

    @classmethod
    def from_py_ptr(cls, frame: ims.PyTimsFrameVectorized):
        """Create a TimsFrameVectorized from a PyTimsFrameVectorized.

        Args:
            frame (pims.PyTimsFrameVectorized): PyTimsFrameVectorized to create the TimsFrameVectorized from.

        Returns:
            TimsFrameVectorized: TimsFrameVectorized created from the PyTimsFrameVectorized.
        """
        instance = cls.__new__(cls)
        instance.__frame_ptr = frame
        return instance

    @property
    def frame_id(self) -> int:
        """Frame ID.

        Returns:
            int: Frame ID.
        """
        return self.__frame_ptr.frame_id

    @property
    def ms_type(self) -> str:
        """MS type.

        Returns:
            int: MS type.
        """
        return self.__frame_ptr.ms_type_as_string

    @property
    def retention_time(self) -> float:
        """Retention time.

        Returns:
            float: Retention time.
        """
        return self.__frame_ptr.retention_time

    @property
    def scan(self) -> NDArray[np.int32]:
        """Scan.

        Returns:
            NDArray[np.int32]: Scan.
        """
        return self.__frame_ptr.scan

    @property
    def mobility(self) -> NDArray[np.float64]:
        """Inverse mobility.

        Returns:
            NDArray[np.float64]: Inverse mobility.
        """
        return self.__frame_ptr.mobility

    @property
    def tof(self) -> NDArray[np.int32]:
        """Time of flight.

        Returns:
            NDArray[np.int32]: Time of flight.
        """
        return self.__frame_ptr.tof

    @property
    def indices(self) -> NDArray[np.int32]:
        """Indices.

        Returns:
            NDArray[np.int32]: Indices.
        """
        return self.__frame_ptr.indices

    @property
    def intensity(self) -> NDArray[np.float64]:
        """Intensity.

        Returns:
            NDArray[np.float64]: Intensity.
        """
        return self.__frame_ptr.values

    @property
    def df(self) -> pd.DataFrame:
        """ Data as a pandas DataFrame.

        Returns:
            pd.DataFrame: Data.
        """

        return pd.DataFrame({
            'frame': np.repeat(self.frame_id, len(self.scan)),
            'retention_time': np.repeat(self.retention_time, len(self.scan)),
            'scan': self.scan,
            'mobility': self.mobility,
            'tof': self.tof,
            'indices': self.indices,
            'intensity': self.intensity})

    def __repr__(self):
        return (f"TimsFrameVectorized(frame_id={self.__frame_ptr.frame_id}, ms_type={self.__frame_ptr.ms_type}, "
                f"num_peaks={len(self.__frame_ptr.indices)})")

    def get_tensor_repr(self, dense=True, zero_indexed=True, re_index=True, scan_max=None, index_max=None):
        from tensorflow import sparse as sp

        s = self.scan
        f = self.indices
        i = self.intensity

        if zero_indexed:
            s = s - np.min(s)
            f = f - np.min(f)

        if re_index:
            f = re_index_indices(f)

        if scan_max is None:
            m_s = np.max(s) + 1
        else:
            m_s = scan_max + 1

        if index_max is None:
            m_f = np.max(f) + 1
        else:
            m_f = index_max + 1

        sv = sp.reorder(sp.SparseTensor(indices=np.c_[s, f], values=i, dense_shape=(m_s, m_f)))

        if dense:
            return sp.to_dense(sv)
        else:
            return sv

    def filter(self,
               mz_min: float = 0.0,
               mz_max: float = 2000.0,
               scan_min: int = 0,
               scan_max: int = 1000,
               mobility_min: float = 0.0,
               mobility_max: float = 2.0,
               intensity_min: float = 0.0,
               intensity_max: float = 1e9,
               ) -> 'TimsFrameVectorized':
        """Filter the frame for a given m/z range, scan range and intensity range.

        Args:
            mz_min (float): Minimum m/z value.
            mz_max (float): Maximum m/z value.
            scan_min (int, optional): Minimum scan value. Defaults to 0.
            scan_max (int, optional): Maximum scan value. Defaults to 1000.
            mobility_min (float, optional): Minimum inverse mobility value. Defaults to 0.0.
            mobility_max (float, optional): Maximum inverse mobility value. Defaults to 2.0.
            intensity_min (float, optional): Minimum intensity value. Defaults to 0.0.
            intensity_max (float, optional): Maximum intensity value. Defaults to 1e9.

        Returns:
            TimsFrameVectorized: Filtered frame.
        """

        return TimsFrameVectorized.from_py_ptr(self.__frame_ptr.filter_ranged(
            mz_min, mz_max, scan_min, scan_max, mobility_min, mobility_max, intensity_min, intensity_max))

    def get_py_ptr(self):
        return self.__frame_ptr

    def get_arrays_at_index(self, index: int) -> Tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
        """Get the arrays at a given index.

        Args:
            index (int): Index.

        Returns:
            NDArray[np.float64]: Arrays at the index.
        """
        return self.__frame_ptr.get_arrays_at_index(index)
