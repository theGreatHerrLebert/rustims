import sqlite3
from typing import List

from imspy.simulation.annotation import RustWrapperObject
from imspy.timstof.data import TimsDataset
import pandas as pd

import imspy_connector

from imspy.timstof.frame import TimsFrame

ims = imspy_connector.py_dia


class TimsDatasetDIA(TimsDataset, RustWrapperObject):
    def __init__(self, data_path: str, in_memory: bool = False, use_bruker_sdk: bool = True):
        super().__init__(data_path=data_path, in_memory=in_memory, use_bruker_sdk=use_bruker_sdk)
        self.__dataset = ims.PyTimsDatasetDIA(self.data_path, self.binary_path, in_memory, self.use_bruker_sdk)

    @property
    def dia_ms_ms_windows(self):
        """Get PASEF meta data for DIA.

        Returns:
            pd.DataFrame: PASEF meta data.
        """
        return pd.read_sql_query("SELECT * from DiaFrameMsMsWindows",
                                 sqlite3.connect(self.data_path + "/analysis.tdf"))

    @property
    def dia_ms_ms_info(self):
        """Get DIA MS/MS info.

        Returns:
            pd.DataFrame: DIA MS/MS info.
        """
        return pd.read_sql_query("SELECT * from DiaFrameMsMsInfo",
                                 sqlite3.connect(self.data_path + "/analysis.tdf"))

    def sample_precursor_signal(self, num_frames: int, max_intensity: float = 25.0, take_probability: float = 0.5) -> TimsFrame:
        """Sample precursor signal.

        Args:
            num_frames: Number of frames.
            max_intensity: Maximum intensity.
            take_probability: Probability to take signals from sampled frames.

        Returns:
            TimsFrame: Frame.
        """

        assert num_frames > 0, "Number of frames must be greater than 0."
        assert 0 < take_probability <= 1, " Probability to take signals from sampled frames must be between 0 and 1."

        return TimsFrame.from_py_ptr(self.__dataset.sample_precursor_signal(num_frames, max_intensity, take_probability))

    def sample_fragment_signal(self, num_frames: int, window_group: int, max_intensity: float = 25.0, take_probability: float = 0.5) -> TimsFrame:
        """Sample fragment signal.

        Args:
            num_frames: Number of frames.
            window_group: Window group to take frames from.
            max_intensity: Maximum intensity.
            take_probability: Probability to take signals from sampled frames.

        Returns:
            TimsFrame: Frame.
        """

        assert num_frames > 0, "Number of frames must be greater than 0."
        assert 0 < take_probability <= 1, " Probability to take signals from sampled frames must be between 0 and 1."

        return TimsFrame.from_py_ptr(self.__dataset.sample_fragment_signal(num_frames, window_group, max_intensity, take_probability))

    def read_compressed_data_full(self) -> List[bytes]:
        """Read compressed data.

        Returns:
            List[bytes]: Compressed data.
        """
        return self.__dataset.read_compressed_data_full()

    @classmethod
    def from_py_ptr(cls, obj):
        instance = cls.__new__(cls)
        instance.__dataset = obj
        return instance

    def get_py_ptr(self):
        return self.__dataset

    def plan_tof_scan_windows(
            self,
            tof_step: int,
            rt_window_sec: float,
            rt_hop_sec: float,
            *,
            num_threads: int = 4,
            maybe_sigma_scans: float | None = None,
            maybe_sigma_tof_bins: float | None = None,
            truncate: float = 3.0,
            precompute_views: bool = False,
    ) -> "TofScanPlan":
        from imspy.timstof.clustering.data import TofScanPlan
        py_plan = ims.PyTofScanPlan(
            self.__dataset,
            tof_step,
            rt_window_sec,
            rt_hop_sec,
            num_threads,
            maybe_sigma_scans,
            maybe_sigma_tof_bins,
            truncate,
            precompute_views,
        )
        return TofScanPlan.from_py_ptr(py_plan)

    def plan_tof_scan_windows_for_group(
            self,
            window_group: int,
            tof_step: int,
            rt_window_sec: float,
            rt_hop_sec: float,
            *,
            num_threads: int = 4,
            maybe_sigma_scans: float | None = None,
            maybe_sigma_tof_bins: float | None = None,
            truncate: float = 3.0,
            precompute_views: bool = False,
    ) -> "TofScanPlanGroup":
        from imspy.timstof.clustering.data import TofScanPlanGroup
        py_plan = ims.PyTofScanPlanGroup(
            self.__dataset,
            int(window_group),
            tof_step,
            rt_window_sec,
            rt_hop_sec,
            num_threads,
            maybe_sigma_scans,
            maybe_sigma_tof_bins,
            truncate,
            precompute_views,
        )
        return TofScanPlanGroup.from_py_ptr(py_plan)