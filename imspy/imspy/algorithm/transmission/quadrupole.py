import numpy as np
from abc import abstractmethod
from typing import Callable, Dict

import pandas as pd

from imspy.core import TimsFrame, MzSpectrum
from numpy.typing import NDArray
from imspy.algorithm.transmission.utility import ion_transition_function_midpoint

import imspy_connector as ims


class TimsTofQuadrupoleDIA:
    def __init__(self, frame: NDArray, frame_window_group: NDArray, window_group: NDArray, scan_start: NDArray,
                 scan_end: NDArray, isolation_mz: NDArray, isolation_width: NDArray, k: float | None = None):
        self.handle = ims.PyTimsTransmissionDIA(
            frame, frame_window_group, window_group, scan_start, scan_end, isolation_mz, isolation_width, k
        )

    def apply_transmission(self, frame_id: int, scan_id: int, mz: NDArray) -> NDArray:
        return self.handle.apply_transmission(frame_id, scan_id, mz)

    def transmit_spectrum(self, frame_id: int, scan_id: int, spectrum: MzSpectrum, min_probability: float | None = None) -> MzSpectrum:
        return MzSpectrum.from_py_mz_spectrum(self.handle.transmit_spectrum(frame_id, scan_id, spectrum.get_spec_ptr(), min_probability))

    def transmit_frame(self, frame: TimsFrame, min_probability: float | None = None) -> TimsFrame:
        return TimsFrame.from_py_tims_frame(self.handle.transmit_tims_frame(frame.get_frame_ptr(), min_probability))

    def frame_to_window_group(self, frame_id: int) -> int:
        return self.handle.frame_to_window_group(frame_id)

    def is_transmitted(self, frame_id: int, scan_id: int, mz: float, min_proba: float | None = None) -> bool:
        return self.handle.is_transmitted(frame_id, scan_id, mz, min_proba)

    def any_transmitted(self, frame_id: int, scan_id: int, mz: NDArray, min_proba: float | None = None) -> bool:
        return self.handle.any_transmitted(frame_id, scan_id, mz, min_proba)


class TimsTofQuadrupoleSetting:
    @abstractmethod
    def get_transmission_function(self, frame_id: int, scan_id: int) -> Callable[[NDArray], NDArray]:
        pass

    @abstractmethod
    def filter_spectrum(self, frame_id: int, scan_id: int, spectrum: MzSpectrum) -> MzSpectrum:
        pass

    @abstractmethod
    def filter_frame(self, frame: TimsFrame) -> TimsFrame:
        pass

    @abstractmethod
    def apply_transmission(self, frame_id: int, scan_id: int, mz: NDArray) -> NDArray:
        pass


class TransmissionDDA(TimsTofQuadrupoleSetting):
    def get_transmission_function(self, frame_id: int, scan_id: int) -> Callable[[NDArray], NDArray]:
        pass

    def filter_spectrum(self, frame_id: int, scan_id: int, spectrum: MzSpectrum) -> MzSpectrum:
        pass

    def filter_frame(self, frame: TimsFrame) -> TimsFrame:
        pass

    def apply_transmission(self, frame_id: int, scan_id: int, mz: NDArray) -> NDArray:
        pass


class TransmissionDIA(TimsTofQuadrupoleSetting):
    def __init__(self, frame_to_window_group: pd.DataFrame, window_group_settings: pd.DataFrame):
        self.frame_to_window_group = self._setup_frame_to_window_group(frame_to_window_group)
        self.window_group_settings = self._setup_window_group_settings(window_group_settings)
        self.transmission_functions = None
        self._setup()

    @staticmethod
    def _setup_frame_to_window_group(frame_to_window_group: pd.DataFrame) -> Dict[int, int]:
        frame_ids = frame_to_window_group.frame.values
        window_groups = frame_to_window_group.window_group.values
        return dict(zip(frame_ids, window_groups))

    @staticmethod
    def _setup_window_group_settings(window_group_settings: pd.DataFrame) -> Dict[tuple, tuple]:
        window_group_dict = {}

        for _, row in window_group_settings.iterrows():
            window_group = row.window_group
            scan_start = int(row.scan_start)
            scan_end = int(row.scan_end)
            isolation_mz = row.isolation_mz
            isolation_width = row.isolation_width
            for scan in range(scan_start, scan_end + 1):
                window_group_dict[(window_group, scan)] = (isolation_mz, isolation_width)

        return window_group_dict

    def _setup(self):
        transmission_dict = {}
        for (window_group, scan), (mz_mid, mz_length) in self.window_group_settings.items():
            transmission_dict[(window_group, scan)] = ion_transition_function_midpoint(midpoint=mz_mid,
                                                                                       window_length=mz_length)

        self.transmission_functions = transmission_dict

    def get_transmission_function(self, frame_id: int, scan_id: int) -> Callable[[NDArray], NDArray]:
        window_group = self.frame_to_window_group[frame_id]
        return self.transmission_functions[(window_group, scan_id)]

    def filter_frame(self, frame: TimsFrame) -> TimsFrame:

            spectra = frame.to_tims_spectra()
            spec_list = []

            for spectrum in spectra:

                f = self.get_transmission_function(frame.frame_id, spectrum.scan)
                mz = spectrum.mz
                mz_len = len(mz)
                transmission = f(mz)

                if np.sum(transmission > 0.001) > 0:
                    first_index = np.argmax(transmission > 0.001)
                    last_index = mz_len - np.argmax(transmission[::-1] > 0) - 1
                    mz_min = mz[first_index]
                    mz_max = mz[last_index]
                    spec_list.append(spectrum.filter(mz_min=mz_min, mz_max=mz_max))

            if len(spec_list) > 0:
                return TimsFrame.from_tims_spectra(spec_list)

            return TimsFrame(
                frame_id=frame.frame_id,
                ms_type=frame.ms_type,
                retention_time=frame.retention_time,
                scan=np.array([], dtype=np.int32),
                mobility=np.array([], dtype=np.float64),
                tof=np.array([], dtype=np.int32),
                mz=np.array([], dtype=np.float64),
                intensity=np.array([], dtype=np.float64)
            )

    def apply_transmission(self, frame_id: int, scan_id: int, mz: NDArray) -> NDArray:
        """
        Apply transmission function to mz array
        :param frame_id: frame of spectrum
        :param scan_id: scan of spectrum
        :param mz: array of mz values
        :return: transmission probability for each mz value
        """
        f = self.get_transmission_function(frame_id, scan_id)
        return f(mz)

    def filter_spectrum(self, frame_id: int, scan_id: int, spectrum: MzSpectrum) -> MzSpectrum:
        """
        Filter spectrum based on transmission function
        :param frame_id: frame of spectrum
        :param scan_id: scan of spectrum
        :param spectrum: spectrum to filter
        :return: filtered spectrum, removing non-transmitted mz values
        """

        f = self.get_transmission_function(frame_id, scan_id)
        mz = spectrum.mz
        mz_len = len(mz)
        transmission = f(mz)

        if np.sum(transmission > 0.001) > 0:
            first_index = np.argmax(transmission > 0.001)
            last_index = mz_len - np.argmax(transmission[::-1] > 0) - 1
            mz_min = mz[first_index]
            mz_max = mz[last_index]
            return spectrum.filter(mz_min=mz_min, mz_max=mz_max)

        else:
            return MzSpectrum(
                mz=np.array([1000.0], dtype=np.float64),
                intensity=np.array([1.0], dtype=np.float64)
            )


class TransmissionMIDIA(TimsTofQuadrupoleSetting):
    def __init__(self, frame_to_window_group: pd.DataFrame, window_group_settings: pd.DataFrame):
        self.frame_to_window_group = self._setup_frame_to_window_group(frame_to_window_group)
        self.window_group_settings = self._setup_window_group_settings(window_group_settings)
        self.transmission_functions = None
        self._setup()

    @staticmethod
    def _setup_frame_to_window_group(frame_to_window_group: pd.DataFrame) -> Dict[int, int]:
        frame_ids = frame_to_window_group.frame.values
        window_groups = frame_to_window_group.window_group.values
        return dict(zip(frame_ids, window_groups))

    @staticmethod
    def _setup_window_group_settings(window_group_settings: pd.DataFrame) -> Dict[tuple, tuple]:
        window_groups = window_group_settings.window_group.values
        scans = window_group_settings.scan_start.values
        mz_mid = window_group_settings.isolation_mz.values
        mz_length = window_group_settings.isolation_width.values
        return dict(zip(zip(window_groups, scans), zip(mz_mid, mz_length)))

    def _setup(self):
        transmission_dict = {}
        for (window_group, scan), (mz_mid, mz_length) in self.window_group_settings.items():
            transmission_dict[(window_group, scan)] = ion_transition_function_midpoint(midpoint=mz_mid,
                                                                                       window_length=mz_length)

        self.transmission_functions = transmission_dict

    def get_transmission_function(self, frame_id: int, scan_id: int) -> Callable[[NDArray], NDArray]:
        window_group = self.frame_to_window_group[frame_id]
        return self.transmission_functions[(window_group, scan_id)]

    def apply_transmission(self, frame_id: int, scan_id: int, mz: NDArray) -> NDArray:
        f = self.get_transmission_function(frame_id, scan_id)
        return f(mz)

    def filter_spectrum(self, frame_id: int, scan_id: int, spectrum: MzSpectrum) -> MzSpectrum:

        f = self.get_transmission_function(frame_id, scan_id)

        mz = spectrum.mz
        mz_len = len(mz)
        transmission = f(mz)

        if np.sum(transmission > 0.001) > 0:
            first_index = np.argmax(transmission > 0.001)
            last_index = mz_len - np.argmax(transmission[::-1] > 0) - 1
            mz_min = mz[first_index]
            mz_max = mz[last_index]
            return spectrum.filter(mz_min=mz_min, mz_max=mz_max)

        else:
            return MzSpectrum(
                mz=np.array([1000.0], dtype=np.float64),
                intensity=np.array([1.0], dtype=np.float64)
            )

    def filter_frame(self, frame: TimsFrame) -> TimsFrame:

        spectra = frame.to_tims_spectra()
        spec_list = []

        for spectrum in spectra:

            f = self.get_transmission_function(frame.frame_id, spectrum.scan)
            mz = spectrum.mz
            mz_len = len(mz)
            transmission = f(mz)

            if np.sum(transmission > 0.001) > 0:
                first_index = np.argmax(transmission > 0.001)
                last_index = mz_len - np.argmax(transmission[::-1] > 0) - 1
                mz_min = mz[first_index]
                mz_max = mz[last_index]
                spec_list.append(spectrum.filter(mz_min=mz_min, mz_max=mz_max))

        if len(spec_list) > 0:
            return TimsFrame.from_tims_spectra(spec_list)

        return TimsFrame(
            frame_id=frame.frame_id,
            ms_type=frame.ms_type,
            retention_time=frame.retention_time,
            scan=np.array([], dtype=np.int32),
            mobility=np.array([], dtype=np.float64),
            tof=np.array([], dtype=np.int32),
            mz=np.array([], dtype=np.float64),
            intensity=np.array([], dtype=np.float64)
        )
