from typing import List

import numpy as np
import pandas as pd
import sqlite3
from numpy.typing import NDArray

import pyims_connector as pims
import opentims_bruker_bridge as obb

from abc import ABC

from pyims.frame import TimsFrame
from pyims.slice import TimsSlice


class TimsDataset(ABC):
    def __init__(self, data_path: str):
        """TimsDataHandle class.

        Args:
            data_path (str): Path to the data.
        """
        self.data_path = data_path
        self.bp: List[str] = obb.get_so_paths()
        self.meta_data = self.__load_meta_data()
        self.precursor_frames = self.meta_data[self.meta_data["MsMsType"] == 0].Id.values.astype(np.int32)
        self.fragment_frames = self.meta_data[self.meta_data["MsMsType"] > 0].Id.values.astype(np.int32)
        self.__handle = None
        self.__current_index = 1

        # Try to load the data with the first binary found
        appropriate_found = False
        for so_path in self.bp:
            try:
                self.__handle = pims.PyTimsDataHandle(self.data_path, so_path)
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
        return self.__handle.get_acquisition_mode_as_string()

    @property
    def acquisition_mode_numerical(self) -> int:
        """Get the acquisition mode as a numerical value.

        Returns:
            int: Acquisition mode as a numerical value.
        """
        return self.__handle.get_acquisition_mode()

    @property
    def frame_count(self) -> int:
        """Get the number of frames.

        Returns:
            int: Number of frames.
        """
        return self.__handle.frame_count

    def __load_meta_data(self) -> pd.DataFrame:
        """Get the meta data.

        Returns:
            pd.DataFrame: Meta data.
        """
        return pd.read_sql_query("SELECT * from Frames", sqlite3.connect(self.data_path + "/analysis.tdf"))

    def get_tims_frame(self, frame_id: int) -> TimsFrame:
        """Get a TimsFrame.

        Args:
            frame_id (int): Frame ID.

        Returns:
            TimsFrame: TimsFrame.
        """
        return TimsFrame.from_py_tims_frame(self.__handle.get_tims_frame(frame_id))

    def get_tims_slice(self, frame_ids: NDArray[np.int32]) -> TimsSlice:
        """Get a TimsFrame.

        Args:
            frame_ids (int): Frame ID.

        Returns:
            TimsFrame: TimsFrame.
        """
        return TimsSlice.from_py_tims_slice(self.__handle.get_tims_slice(frame_ids))

    def __iter__(self):
        return self

    def __next__(self):
        if self.__current_index <= self.frame_count:
            frame_ptr = self.__handle.get_tims_frame(self.__current_index)
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


class TimsDatasetDDA(TimsDataset):
    @property
    def selected_precursors(self):
        """Get precursors selected for fragmentation.

        Returns:
            pd.DataFrame: Precursors selected for fragmentation.
        """
        return pd.read_sql_query("SELECT * from Precursors", sqlite3.connect(self.data_path + "/analysis.tdf"))

    @property
    def pasef_meta_data(self):
        """Get PASEF meta data for DDA.

        Returns:
            pd.DataFrame: PASEF meta data.
        """
        return pd.read_sql_query("SELECT * from PasefFrameMsMsInfo",
                                 sqlite3.connect(self.data_path + "/analysis.tdf"))


class TimsDatasetDIA(TimsDataset):
    @property
    def pasef_meta_data(self):
        """Get PASEF meta data for DIA.

        Returns:
            pd.DataFrame: PASEF meta data.
        """
        return pd.read_sql_query("SELECT * from DiaFrameMsMsWindows",
                                 sqlite3.connect(self.data_path + "/analysis.tdf"))
