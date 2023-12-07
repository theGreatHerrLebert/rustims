from typing import Dict

import numpy as np
import pandas as pd
import sqlite3
from numpy.typing import NDArray

import imspy_connector as pims
import opentims_bruker_bridge as obb

from abc import ABC

from imspy.core.frame import TimsFrame
from imspy.core.slice import TimsSlice


class AcquisitionMode:
    def __init__(self, mode: str):
        """AcquisitionMode class.

        Args:
            mode (str): Acquisition mode.
        """
        allowed_modes = ["DDA", "DIA", "MIDIA", "UNKNOWN", "PRECURSOR"]
        assert mode in allowed_modes, f"Unknown acquisition mode, use one of {allowed_modes}"
        self.__mode_ptr = pims.PyAcquisitionMode.from_string(mode)

    @property
    def mode(self) -> str:
        """Get the acquisition mode.

        Returns:
            str: Acquisition mode.
        """
        return self.__mode_ptr.acquisition_mode

    @classmethod
    def from_ptr(cls, ptr: pims.PyAcquisitionMode):
        """Get an AcquisitionMode from a pointer.

        Args:
            ptr (pims.AcquisitionMode): Pointer to an acquisition mode.

        Returns:
            AcquisitionMode: Acquisition mode.
        """
        instance = cls.__new__(cls)
        instance.__mode_ptr = ptr
        return instance

    def __repr__(self):
        return f"AcquisitionMode({self.mode})"


class TimsDataset(ABC):
    def __init__(self, data_path: str):
        """TimsDataHandle class.

        Args:
            data_path (str): Path to the data.
        """
        self.__dataset = None
        self.binary_path = None

        self.data_path = data_path
        self.meta_data = self.__load_meta_data()
        self.global_meta_data = self.__load_global_meta_data()
        self.precursor_frames = self.meta_data[self.meta_data["MsMsType"] == 0].Id.values.astype(np.int32)
        self.fragment_frames = self.meta_data[self.meta_data["MsMsType"] > 0].Id.values.astype(np.int32)
        self.__current_index = 1

        # Try to load the data with the first binary found
        appropriate_found = False
        for so_path in obb.get_so_paths():
            try:
                self.__dataset = pims.PyTimsDataset(self.data_path, so_path)
                self.binary_path = so_path
                appropriate_found = True
                break
            except Exception:
                continue
        assert appropriate_found is True, ("No appropriate bruker binary could be found, please check if your "
                                           "operating system is supported by open-tims-bruker-bridge.")

    @property
    def acquisition_mode(self) -> str:
        """Get the acquisition mode.

        Returns:
            str: Acquisition mode.
        """
        return self.__dataset.get_acquisition_mode_as_string()

    @property
    def acquisition_mode_numerical(self) -> int:
        """Get the acquisition mode as a numerical value.

        Returns:
            int: Acquisition mode as a numerical value.
        """
        return self.__dataset.get_acquisition_mode()

    @property
    def frame_count(self) -> int:
        """Get the number of frames.

        Returns:
            int: Number of frames.
        """
        return self.__dataset.frame_count()

    def __load_meta_data(self) -> pd.DataFrame:
        """Get the meta data.

        Returns:
            pd.DataFrame: Meta data.
        """
        return pd.read_sql_query("SELECT * from Frames", sqlite3.connect(self.data_path + "/analysis.tdf"))

    def __load_global_meta_data(self) -> Dict[str, str]:
        """Get the global meta data.

        Returns:
            pd.DataFrame: Global meta data.
        """
        d = pd.read_sql_query("SELECT * from GlobalMetaData", sqlite3.connect(self.data_path + "/analysis.tdf"))
        return dict(zip(d.Key, d.Value))

    @property
    def im_lower(self):
        return float(self.global_meta_data["OneOverK0AcqRangeLower"])

    @property
    def im_upper(self):
        return float(self.global_meta_data["OneOverK0AcqRangeUpper"])

    @property
    def mz_lower(self):
        return float(self.global_meta_data["MzAcqRangeLower"])

    @property
    def mz_upper(self):
        return float(self.global_meta_data["MzAcqRangeUpper"])

    @property
    def average_cycle_length(self) -> float:
        return np.mean(np.diff(self.meta_data.Time.values))

    def get_tims_frame(self, frame_id: int) -> TimsFrame:
        """Get a TimsFrame.

        Args:
            frame_id (int): Frame ID.

        Returns:
            TimsFrame: TimsFrame.
        """
        return TimsFrame.from_py_tims_frame(self.__dataset.get_frame(frame_id))

    def get_tims_slice(self, frame_ids: NDArray[np.int32]) -> TimsSlice:
        """Get a TimsFrame.

        Args:
            frame_ids (int): Frame ID.

        Returns:
            TimsFrame: TimsFrame.
        """
        return TimsSlice.from_py_tims_slice(self.__dataset.get_slice(frame_ids))

    def __iter__(self):
        return self

    def __next__(self):
        if self.__current_index <= self.frame_count:
            frame_ptr = self.__dataset.get_frame(self.__current_index)
            self.__current_index += 1
            if frame_ptr is not None:
                return TimsFrame.from_py_tims_frame(frame_ptr)
            else:
                raise ValueError(f"Frame pointer is None for valid index: {self.__current_index}")
        else:
            self.__current_index = 1  # Reset for next iteration
            raise StopIteration

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.get_tims_slice(np.arange(index.start, index.stop, index.step).astype(np.int32))
        return self.get_tims_frame(index)
