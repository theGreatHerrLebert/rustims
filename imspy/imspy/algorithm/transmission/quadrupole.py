import numpy as np
from abc import abstractmethod
from typing import Callable, Dict

import pandas as pd

from imspy.core import TimsFrame
from numpy.typing import NDArray
from imspy.algorithm.transmission.utility import ion_transition_function_midpoint


class TimsTofQuadrupoleSetting:
    @abstractmethod
    def get_transmission_function(self, frame_id: int, scan_id: int) -> Callable[[NDArray], NDArray]:
        pass

    @abstractmethod
    def apply_transmission(self, frame: TimsFrame) -> TimsFrame:
        pass


class TransmissionDDA(TimsTofQuadrupoleSetting):
    def get_transmission_function(self, frame_id: int, scan_id: int) -> Callable[[NDArray], NDArray]:
        pass

    def apply_transmission(self, frame: TimsFrame) -> TimsFrame:
        pass


class TransmissionDIA(TimsTofQuadrupoleSetting):
    def get_transmission_function(self, frame_id: int, scan_id: int) -> Callable[[NDArray], NDArray]:
        pass

    def apply_transmission(self, frame: TimsFrame) -> TimsFrame:
        pass


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

    def apply_transmission(self, frame: TimsFrame) -> TimsFrame:

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
