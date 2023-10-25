import numpy as np
import pandas as pd
from typing import List

import pyims_connector as pims
from pyims.frame import TimsFrame
from pyims.spectrum import MzSpectrum


class TimsSlice:
    def __int__(self):
        self.__slice_ptr = None
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
        return TimsSlice.from_py_tims_slice(self.__slice_ptr.filter_ranged(mz_min, mz_max, scan_min, scan_max, mobility_min, mobility_max,
                                                                           intensity_min, intensity_max, num_threads))

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

    def to_windows(self, window_length: float = 10, overlapping: bool = True, min_num_peaks: int = 5, min_intensity: float = 1, num_threads: int = 4) -> List[MzSpectrum]:
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
        return [MzSpectrum.from_py_mz_spectrum(spec) for spec in self.__slice_ptr.to_windows(
            window_length, overlapping, min_num_peaks, min_intensity, num_threads)]

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

    def get_tims_planes(self, tof_max_value: int = 400_000, num_chunks: int = 7, num_threads: int = 4) -> List['TimsPlane']:
        return [TimsPlane.from_py_tims_plane(plane) for plane in self.__slice_ptr.to_tims_planes(tof_max_value, num_chunks, num_threads)]


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
