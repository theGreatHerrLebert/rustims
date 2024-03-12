import numpy as np
import pandas as pd
import sqlite3

from typing import Dict, List

from numpy.typing import NDArray
import opentims_bruker_bridge as obb

from abc import ABC

from imspy.timstof.frame import TimsFrame
from imspy.timstof.slice import TimsSlice

import imspy_connector
ims = imspy_connector.py_dataset


class AcquisitionMode:
    def __init__(self, mode: str):
        """AcquisitionMode class.

        Args:
            mode (str): Acquisition mode.
        """
        allowed_modes = ["DDA", "DIA", "UNKNOWN", "PRECURSOR"]
        assert mode in allowed_modes, f"Unknown acquisition mode, use one of {allowed_modes}"
        self.__mode_ptr = ims.PyAcquisitionMode.from_string(mode)

    @property
    def mode(self) -> str:
        """Get the acquisition mode.

        Returns:
            str: Acquisition mode.
        """
        return self.__mode_ptr.acquisition_mode

    @classmethod
    def from_ptr(cls, ptr: ims.PyAcquisitionMode):
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
                self.__dataset = ims.PyTimsDataset(self.data_path, so_path)
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
        return self.__dataset.get_acquisition_mode()

    @property
    def acquisition_mode_numeric(self) -> int:
        """Get the acquisition mode as a numerical value.

        Returns:
            int: Acquisition mode as a numerical value.
        """
        return self.__dataset.get_acquisition_mode_numeric()

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
        d = pd.read_sql_query("SELECT * from GlobalMetadata", sqlite3.connect(self.data_path + "/analysis.tdf"))
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

    @property
    def description(self) -> str:
        return self.global_meta_data["Description"]

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

    def tof_to_mz(self, frame_id: int, tof_values: NDArray[np.int32]) -> NDArray[np.float64]:
        """Convert TOF values to m/z values.

        Args:
            frame_id (int): Frame ID.
            tof_values (NDArray[np.int32]): TOF values.

        Returns:
            NDArray[np.float64]: m/z values.
        """
        return self.__dataset.tof_to_mz(frame_id, tof_values)

    def mz_to_tof(self, frame_id: int, mz_values: NDArray[np.float64]) -> NDArray[np.int32]:
        """Convert m/z values to TOF values.

        Args:
            frame_id (int): Frame ID.
            mz_values (NDArray[np.float64]): m/z values.

        Returns:
            NDArray[np.int32]: TOF values.
        """
        return self.__dataset.mz_to_tof(frame_id, mz_values)

    def scan_to_inverse_mobility(self, frame_id: int, scan_values: NDArray[np.int32]) -> NDArray[np.float64]:
        """Convert scan values to inverse mobility values.

        Args:
            frame_id (int): Frame ID.
            scan_values (NDArray[np.int32]): Scan values.

        Returns:
            NDArray[np.float64]: Inverse mobility values.
        """
        return self.__dataset.scan_to_inverse_mobility(frame_id, scan_values)

    def inverse_mobility_to_scan(self, frame_id: int, im_values: NDArray[np.float64]) -> NDArray[np.int32]:
        """Convert inverse mobility values to scan values.

        Args:
            frame_id (int): Frame ID.
            im_values (NDArray[np.float64]): Inverse mobility values.

        Returns:
            NDArray[np.int32]: Scan values.
        """
        return self.__dataset.inverse_mobility_to_scan(frame_id, im_values)

    def compress_zstd(self, values: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Compress values using ZSTD.

        Args:
            values (NDArray[np.float64]): Values to compress.

        Returns:
            NDArray[np.uint8]: Compressed values.
        """
        return self.__dataset.compress_bytes_zstd(values)

    def decompress_zstd(self, values: NDArray[np.uint8], ignore_first_n: int = 8) -> NDArray[np.uint8]:
        """Decompress values using ZSTD.

        Args:
            values (NDArray[np.float64]): Values to decompress.
            ignore_first_n (int): Number of bytes to ignore.

        Returns:
            NDArray[np.uint8]: Decompressed values.
        """
        return self.__dataset.decompress_bytes_zstd(values[ignore_first_n:])

    def indexed_values_to_compressed_bytes(self, scan_values: NDArray[np.int32], tof_values: NDArray[np.int32],
                                intensity_values: NDArray[np.float64], total_scans: int) -> NDArray[np.uint8]:
        """Convert scan and intensity values to bytes.

        Args:
            scan_values (NDArray[np.int32]): Scan values.
            tof_values (NDArray[np.int32]): TOF values.
            intensity_values (NDArray[np.float64]): Intensity values.
            total_scans (int): Total number of scans.

        Returns:
            NDArray[np.uint8]: Bytes.
        """
        return self.__dataset.scan_tof_intensities_to_compressed_u8(
            scan_values, 
            tof_values,
            intensity_values.astype(np.int32),
            total_scans
        )

    def compress_frames(self, frames: List[TimsFrame],
                                  total_scans: int, num_threads: int = 4) -> List[NDArray[np.uint8]]:
        """Compress a collection of frames.

        Args:
            frames (List[TimsFrame]): List of frames.
            total_scans (int): Total number of scans.
            num_threads (int): Number of threads to use.

        Returns:
            List[NDArray[np.uint8]]: List of compressed bytes.
        """
        return self.__dataset.compress_frames([f.get_frame_ptr() for f in frames], total_scans, num_threads)

    def bytes_to_indexed_values(self, values: NDArray[np.uint8]) \
            -> (NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]):
        """Convert bytes to scan, tof, and intensity values.

        Args:
            values (NDArray[np.uint8]): Bytes.

        Returns:
            NDArray[np.int32]: Scan values.
            NDArray[np.int32]: TOF values.
            NDArray[np.float64]: Intensity values.
        """
        scan_values, tof_values, intensity_values = self.__dataset.u8_to_scan_tof_intensities(values)
        return scan_values, tof_values, intensity_values.astype(np.float64)

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

    def __repr__(self):
        return f"TimsDataset({self.data_path})"
