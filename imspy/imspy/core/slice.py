import numpy as np
import pandas as pd
from typing import List, Tuple, Any

from numpy.typing import NDArray
from tensorflow import sparse as sp

from imspy.utility.utilities import re_index_indices

import imspy_connector as pims

from imspy.core.frame import TimsFrame, TimsFrameVectorized
from imspy.core.spectrum import MzSpectrum, TimsSpectrum


class TimsSlice:
    def __init__(self,
                 frame_id: NDArray[np.int32],
                 scan: NDArray[np.int32],
                 tof: NDArray[np.int32],
                 retention_time: NDArray[np.float64],
                 mobility: NDArray[np.float64],
                 mz: NDArray[np.float64],
                 intensity: NDArray[np.float64]):
        """Create a TimsSlice.

        Args:
            frame_id (NDArray[np.int32]): Frame ID.
            scan (NDArray[np.int32]): Scan.
            tof (NDArray[np.int32]): TOF.
            retention_time (NDArray[np.float64]): Retention time.
            mobility (NDArray[np.float64]): Mobility.
            mz (NDArray[np.float64]): m/z.
            intensity (NDArray[np.float64]): Intensity.
        """

        assert len(frame_id) == len(scan) == len(tof) == len(retention_time) == len(mobility) == len(mz) == len(
            intensity), "All arrays must have the same length."

        self.__slice_ptr = pims.PyTimsSlice(
            frame_id, scan, tof, retention_time, mobility, mz, intensity
        )
        self.__current_index = 0

    @classmethod
    def from_py_tims_slice(cls, tims_slice: pims.PyTimsSlice):
        """Create a TimsSlice from a PyTimsSlice.

        Args:
            tims_slice (pims.PyTimsSlice): PyTimsSlice to create the TimsSlice from.

        Returns:
            TimsSlice: TimsSlice created from the PyTimsSlice.
        """
        instance = cls.__new__(cls)
        instance.__slice_ptr = tims_slice
        instance.__current_index = 0
        return instance

    @classmethod
    def from_frames(cls, frames: List[TimsFrame]):
        """Create a TimsSlice from a list of TimsFrames.

        Args:
            frames (List[TimsFrame]): List of TimsFrames.

        Returns:
            TimsSlice: TimsSlice created from the list of TimsFrames.
        """
        return cls.from_py_tims_slice(pims.PyTimsSlice.from_frames([frame.get_frame_ptr() for frame in frames]))

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

    @property
    def precursors(self):
        return TimsSlice.from_py_tims_slice(self.__slice_ptr.get_precursors())

    @property
    def fragments(self):
        return TimsSlice.from_py_tims_slice(self.__slice_ptr.get_fragments_dda())

    def filter(self, mz_min: float = 0.0, mz_max: float = 2000.0, scan_min: int = 0, scan_max: int = 1000,
               mobility_min: float = 0.0,
               mobility_max: float = 2.0,
               intensity_min: float = 0.0, intensity_max: float = 1e9, num_threads: int = 4) -> 'TimsSlice':
        """Filter the slice by m/z, scan and intensity.

        Args:
            mz_min (float): Minimum m/z value.
            mz_max (float): Maximum m/z value.
            scan_min (int, optional): Minimum scan value. Defaults to 0.
            scan_max (int, optional): Maximum scan value. Defaults to 1000.
            mobility_min (float, optional): Minimum inverse mobility value. Defaults to 0.0.
            mobility_max (float, optional): Maximum inverse mobility value. Defaults to 2.0.
            intensity_min (float, optional): Minimum intensity value. Defaults to 0.0.
            intensity_max (float, optional): Maximum intensity value. Defaults to 1e9.
            num_threads (int, optional): Number of threads to use. Defaults to 4.

        Returns:
            TimsSlice: Filtered slice.
        """
        return TimsSlice.from_py_tims_slice(
            self.__slice_ptr.filter_ranged(mz_min, mz_max, scan_min, scan_max, mobility_min, mobility_max,
                                           intensity_min, intensity_max, num_threads))

    def filter_by_type(self,
                       mz_min_ms1: float = 0,
                       mz_max_ms1: float = 2000,
                       scan_min_ms1: int = 0,
                       scan_max_ms1: int = 1000,
                       inv_mob_min_ms1: float = 0,
                       inv_mob_max_ms1: float = 2,
                       intensity_min_ms1: float = 0,
                       intensity_max_ms1: float = 1e9,
                       mz_min_ms2: float = 0,
                       mz_max_ms2: float = 2000,
                       scan_min_ms2: int = 0,
                       scan_max_ms2: int = 1000,
                       inv_mob_min_ms2: float = 0,
                       inv_mob_max_ms2: float = 2,
                       intensity_min_ms2: float = 0,
                       intensity_max_ms2: float = 1e9,
                       num_threads: int = 4) -> 'TimsSlice':

        """Filter the slice by m/z, scan and intensity, for MS1 and MS2 with different ranges.

        Args:
            mz_min_ms1 (float, optional): Minimum m/z value for MS1. Defaults to 0.
            mz_max_ms1 (float, optional): Maximum m/z value for MS1. Defaults to 2000.
            scan_min_ms1 (int, optional): Minimum scan value for MS1. Defaults to 0.
            scan_max_ms1 (int, optional): Maximum scan value for MS1. Defaults to 1000.
            inv_mob_min_ms1 (float, optional): Minimum inverse mobility value for MS1. Defaults to 0.
            inv_mob_max_ms1 (float, optional): Maximum inverse mobility value for MS1. Defaults to 2.
            intensity_min_ms1 (float, optional): Minimum intensity value for MS1. Defaults to 0.
            intensity_max_ms1 (float, optional): Maximum intensity value for MS1. Defaults to 1e9.
            mz_min_ms2 (float, optional): Minimum m/z value for MS2. Defaults to 0.
            mz_max_ms2 (float, optional): Maximum m/z value for MS2. Defaults to 2000.
            scan_min_ms2 (int, optional): Minimum scan value for MS2. Defaults to 0.
            scan_max_ms2 (int, optional): Maximum scan value for MS2. Defaults to 1000.
            inv_mob_min_ms2 (float, optional): Minimum inverse mobility value for MS2. Defaults to 0.
            inv_mob_max_ms2 (float, optional): Maximum inverse mobility value for MS2. Defaults to 2.
            intensity_min_ms2 (float, optional): Minimum intensity value for MS2. Defaults to 0.
            intensity_max_ms2 (float, optional): Maximum intensity value for MS2. Defaults to 1e9.
            num_threads (int, optional): Number of threads to use. Defaults to 4.

        Returns:
            TimsSlice: Filtered slice.
        """
        return TimsSlice.from_py_tims_slice(
            self.__slice_ptr.filter_ranged_ms_type_specific(
                mz_min_ms1, mz_max_ms1, scan_min_ms1, scan_max_ms1, inv_mob_min_ms1,
                inv_mob_max_ms1, intensity_min_ms1, intensity_max_ms1, mz_min_ms2, mz_max_ms2,
                scan_min_ms2, scan_max_ms2, inv_mob_min_ms2, inv_mob_max_ms2, intensity_min_ms2,
                intensity_max_ms2, num_threads))

    @property
    def frames(self) -> List[TimsFrame]:
        """Get the frames.

        Returns:
            List[TimsFrame]: Frames.
        """
        return [TimsFrame.from_py_tims_frame(frame) for frame in self.__slice_ptr.get_frames()]

    def to_resolution(self, resolution: int, num_threads: int = 4) -> 'TimsSlice':
        """Convert the slice to a given resolution.

        Args:
            resolution (int): Resolution.
            num_threads (int, optional): Number of threads to use. Defaults to 4.

        Returns:
            TimsSlice: Slice with given resolution.
        """
        return TimsSlice.from_py_tims_slice(self.__slice_ptr.to_resolution(resolution, num_threads))

    def to_windows(self, window_length: float = 10, overlapping: bool = True, min_num_peaks: int = 5,
                   min_intensity: float = 1, num_threads: int = 4) -> List[TimsSpectrum]:
        """Convert the slice to a list of windows.

        Args:
            window_length (float, optional): Window length. Defaults to 10.
            overlapping (bool, optional): Whether the windows should overlap. Defaults to True.
            min_num_peaks (int, optional): Minimum number of peaks in a window. Defaults to 5.
            min_intensity (float, optional): Minimum intensity of a peak in a window. Defaults to 1.
            num_threads (int, optional): Number of threads to use. Defaults to 1.

        Returns:
            List[MzSpectrum]: List of windows.
        """
        return [TimsSpectrum.from_py_tims_spectrum(spec) for spec in self.__slice_ptr.to_windows(
            window_length, overlapping, min_num_peaks, min_intensity, num_threads)]

    def to_dense_windows(self, window_length: float = 10, resolution: int = 1, overlapping: bool = True,
                         min_num_peaks: int = 5, min_intensity: float = 0.0, num_theads: int = 4) -> (
            tuple)[list[NDArray], list[NDArray], list[NDArray]]:

        DW = self.__slice_ptr.to_dense_windows(window_length, overlapping, min_num_peaks, min_intensity, resolution,
                                               num_theads)

        scan_list, window_indices_list, values_list = [], [], []

        for values, scans, bins, row, col in DW:
            W = np.reshape(values, (row, col))
            scan_list.append(scans)
            window_indices_list.append(bins)
            values_list.append(W)

        return scan_list, window_indices_list, values_list

    @property
    def df(self) -> pd.DataFrame:
        """Get the data as a pandas DataFrame.

        Returns:
            pd.DataFrame: Data.
        """
        columns = ['frame', 'scan', 'tof', 'retention_time', 'mobility', 'mz', 'intensity']
        return pd.DataFrame({c: v for c, v in zip(columns, self.__slice_ptr.to_arrays())})

    def __iter__(self):
        return self

    def __next__(self):
        if self.__current_index < self.__slice_ptr.frame_count:
            frame_ptr = self.__slice_ptr.get_frame_at_index(self.__current_index)
            self.__current_index += 1
            if frame_ptr is not None:
                return TimsFrame.from_py_tims_frame(frame_ptr)
            else:
                raise ValueError("Frame pointer is None for valid index.")
        else:
            self.__current_index = 0  # Reset for next iteration
            raise StopIteration

    def vectorized(self, resolution: int = 2, num_threads: int = 4) -> 'TimsSliceVectorized':
        """Get a vectorized version of the slice.

        Args:
            resolution (int, optional): Resolution. Defaults to 2.
            num_threads (int, optional): Number of threads to use. Defaults to 4.

        Returns:
            TimsSliceVectorized: Vectorized version of the slice.
        """
        return TimsSliceVectorized.from_vectorized_py_tims_slice(self.__slice_ptr.vectorized(resolution, num_threads))

    def get_tims_planes(self, tof_max_value: int = 400_000, num_chunks: int = 7, num_threads: int = 4) -> List[
        'TimsPlane']:
        return [TimsPlane.from_py_tims_plane(plane) for plane in
                self.__slice_ptr.to_tims_planes(tof_max_value, num_chunks, num_threads)]


class TimsSliceVectorized:
    def __init__(self):
        self.__slice_ptr = None
        self.__current_index = 0

    @classmethod
    def from_vectorized_py_tims_slice(cls, tims_slice: pims.PyTimsSliceVectorized):
        """Create a TimsSlice from a PyTimsSlice.

        Args:
            tims_slice (pims.PyTimsSlice): PyTimsSlice to create the TimsSlice from.

        Returns:
            TimsSlice: TimsSlice created from the PyTimsSlice.
        """
        instance = cls.__new__(cls)
        instance.__slice_ptr = tims_slice
        instance.__current_index = 0
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

    @property
    def precursors(self):
        return TimsSlice.from_py_tims_slice(self.__slice_ptr.get_precursors())

    @property
    def fragments(self):
        return TimsSlice.from_py_tims_slice(self.__slice_ptr.get_fragments_dda())

    @property
    def frames(self) -> List[TimsFrameVectorized]:
        """Get the frames.

        Returns:
            List[TimsFrame]: Frames.
        """
        return [TimsFrameVectorized.from_py_tims_frame_vectorized(frame) for frame in
                self.__slice_ptr.get_vectorized_frames()]

    @property
    def df(self) -> pd.DataFrame:
        """Get the data as a pandas DataFrame.

        Returns:
            pd.DataFrame: Data.
        """
        columns = ['frame', 'scan', 'tof', 'retention_time', 'mobility', 'index', 'intensity']
        return pd.DataFrame({c: v for c, v in zip(columns, self.__slice_ptr.to_arrays())})

    def __iter__(self):
        return self

    def __next__(self):
        if self.__current_index < self.__slice_ptr.frame_count:
            frame_ptr = self.__slice_ptr.get_frame_at_index(self.__current_index)
            self.__current_index += 1
            if frame_ptr is not None:
                return TimsFrameVectorized.from_py_tims_frame_vectorized(frame_ptr)
            else:
                raise ValueError("Frame pointer is None for valid index.")
        else:
            self.__current_index = 0
            raise StopIteration

    def __repr__(self):
        return f"TimsSliceVectorized({self.first_frame_id}, {self.last_frame_id})"

    def get_tensor_repr(self, dense=True, zero_index=True, re_index=True, frame_max=None, scan_max=None,
                        index_max=None):

        frames, scans, _, _, _, indices, intensities = self.__slice_ptr.to_arrays()

        if zero_index:
            scans = scans - np.min(scans)
            frames = frames - np.min(frames)
            indices = indices - np.min(indices)

        if re_index:
            frames = re_index_indices(frames)

        if scan_max is None:
            m_s = np.max(scans) + 1
        else:
            m_s = scan_max + 1

        if index_max is None:
            m_i = np.max(indices) + 1
        else:
            m_i = index_max + 1

        if frame_max is None:
            m_f = np.max(frames) + 1
        else:
            m_f = frame_max + 1

        sv = sp.reorder(
            sp.SparseTensor(indices=np.c_[frames, scans, indices], values=intensities, dense_shape=(m_f, m_s, m_i)))

        if dense:
            return sp.to_dense(sv)
        else:
            return sv


class TimsPlane:
    def __init__(self):
        self.__plane_ptr = None

    @classmethod
    def from_py_tims_plane(cls, plane: pims.PyTimsPlane):
        """Create a TimsPlane from a PyTimsPlane.

        Args:
            plane (pims.PyTimsPlane): PyTimsPlane to create the TimsPlane from.

        Returns:
            TimsPlane: TimsPlane created from the PyTimsPlane.
        """
        instance = cls.__new__(cls)
        instance.__plane_ptr = plane
        return instance

    @property
    def mz_mean(self):
        return self.__plane_ptr.mz_mean

    @property
    def mz_std(self):
        return self.__plane_ptr.mz_std

    @property
    def tof_mean(self):
        return self.__plane_ptr.tof_mean

    @property
    def tof_std(self):
        return self.__plane_ptr.tof_std

    @property
    def frame_ids(self):
        return self.__plane_ptr.frame_ids

    @property
    def scans(self):
        return self.__plane_ptr.scans

    @property
    def intensities(self):
        return self.__plane_ptr.intensity

    @property
    def retention_times(self):
        return self.__plane_ptr.retention_times

    @property
    def mobilities(self):
        return self.__plane_ptr.mobilities

    @property
    def num_points(self):
        return len(self.frame_ids)

    @property
    def df(self):
        return pd.DataFrame({
            'frame': self.frame_ids,
            'scan': self.scans,
            'retention_time': self.retention_times,
            'mobility': self.mobilities,
            'intensity': self.intensities
        })

    def __repr__(self):
        return (f"TimsPlane(mz_mean: "
                f"{np.round(self.mz_mean, 4)}, "
                f"mz_std: {np.round(self.mz_std, 4)},"
                f" tof_mean: {np.round(self.tof_mean, 4)}, "
                f"tof_std: {np.round(self.tof_std, 4)}, "
                f"num_points: {len(self.frame_ids)})")
