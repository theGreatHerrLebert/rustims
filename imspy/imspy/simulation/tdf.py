import sqlite3
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
from imspy.timstof.frame import TimsFrame
from imspy.timstof import TimsDataset
import zstd

import imspy_connector
ims = imspy_connector.py_dataset

class TDFWriter:
    def __init__(
            self,
            helper_handle: TimsDataset,
            path: str = "./",
            exp_name: str = "RAW.d",
            offset_bytes: int = 64,
            ) -> None:

        self.path = Path(path)
        self.exp_name = exp_name
        self.full_path = Path(path) / exp_name
        self.position = 0
        self.binary_file = self.full_path / "analysis.tdf_bin"
        self.frame_meta_data = []
        self.conn = None
        self.helper_handle = helper_handle
        self.offset_bytes = offset_bytes

        self.__conn_native = None
        self._setup_connections()

    def _setup_connections(self) -> None:
        # Create the directory and connect to DB
        self.full_path.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(f'{self.full_path}/analysis.tdf')

        frame_ms_ms_info = self.helper_handle.get_table("FrameMsmsInfo")
        segments = self.helper_handle.get_table("Segments")
        last_frame = self.helper_handle.meta_data.Id.max()
        segments.iloc[0, segments.columns.get_loc("LastFrame")] = last_frame

        # Save table to analysis.tdf
        self._create_table(self.conn, self.helper_handle.mz_calibration, "MzCalibration")
        self._create_table(self.conn, self.helper_handle.tims_calibration, "TimsCalibration")
        self._create_table(self.conn, self.helper_handle.global_meta_data_pandas, "GlobalMetadata")
        self._create_table(self.conn, frame_ms_ms_info, "FrameMsmsInfo")
        self._create_table(self.conn, segments, "Segments")

        with open(self.binary_file, "wb") as bin_file:
            bin_file.write(b'\x00' * self.offset_bytes)
            self.position = bin_file.tell()

    @staticmethod
    def _get_table(conn, table_name: str) -> pd.DataFrame:
        # Get a table as a pandas DataFrame
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)

    @staticmethod
    def _create_table(conn, table, table_name: str) -> None:
        # Create a table from a pandas DataFrame
        table.to_sql(table_name, conn, if_exists='replace', index=False)

    def mz_to_tof(self, frame_id, mzs):
        return self.helper_handle.mz_to_tof(frame_id, mzs)

    def tof_to_mz(self, frame_id, tofs):
        return self.helper_handle.tof_to_mz(frame_id, tofs)

    def inv_mobility_to_scan(self, frame_id, inv_mobs):
        return self.helper_handle.inverse_mobility_to_scan(frame_id, inv_mobs)

    def scan_to_inv_mobility(self, frame_id, scans):
        return self.helper_handle.scan_to_inverse_mobility(frame_id, scans)

    def __repr__(self) -> str:
        return f"TDFWriter(path={self.path}, db_name={self.exp_name}, num_scans={self.helper_handle.num_scans}, " \
               f"im_lower={self.helper_handle.im_lower}, im_upper={self.helper_handle.im_upper}, mz_lower={self.helper_handle.mz_lower}, " \
               f"mz_upper={self.helper_handle.mz_upper})"

    def build_frame_meta_row(self, frame: TimsFrame, scan_mode: int, frame_start_pos: int, only_frame_one: bool = False):
        r = self.helper_handle.meta_data.iloc[0, :].copy()
        if not only_frame_one:
            r = self.helper_handle.meta_data.iloc[frame.frame_id - 1, :].copy()

        r.Id = frame.frame_id
        r.Time = frame.retention_time
        r.ScanMode = scan_mode
        r.MsMsType = frame.ms_type
        r.TimsId = frame_start_pos
        r.MaxIntensity = int(np.max(frame.intensity)) if len(frame.intensity) > 0 else 0
        r.SummedIntensities = int(np.sum(frame.intensity)) if len(frame.intensity) > 0 else 0
        r.NumScans = self.helper_handle.num_scans
        r.NumPeaks = len(frame.mz)

        return r

    def compress_frame(self, frame: TimsFrame, only_frame_one: bool = False) -> bytes:
        # calculate TOF using the DH of the other frame
        # TODO: move translation of mz -> tof and inv_mob -> scan to the helper handle
        i = 1 if only_frame_one else frame.frame_id

        tof = self.mz_to_tof(i, frame.mz)
        scan = self.inv_mobility_to_scan(i, frame.mobility)
        return self.helper_handle.indexed_values_to_compressed_bytes(scan, tof,
                                                                     frame.intensity,
                                                                     total_scans=self.helper_handle.num_scans)

    def compress_frames(self, frames: List[TimsFrame], num_threads: int = 4) -> List[bytes]:
        return self.helper_handle.compress_frames(frames, num_threads=num_threads)

    def write_frame(self, frame: TimsFrame, scan_mode: int, only_frame_one: bool = False) -> None:
        self.frame_meta_data.append(self.build_frame_meta_row(frame, scan_mode, self.position, only_frame_one))
        # compressed_data = self.compress_frame(frame, only_frame_one)

        i = 1 if only_frame_one else frame.frame_id
        tof = self.mz_to_tof(i, frame.mz).astype(np.uint32)
        scan = self.inv_mobility_to_scan(i, frame.mobility).astype(np.uint32)
        intensity = frame.intensity.astype(np.uint32)

        real_data = imspy_connector.get_data_for_compression(tof, scan, intensity, self.helper_handle.num_scans)
        compressed_data = compressed_data = zstd.ZSTD_compress(bytes(real_data), 1)

        with open(self.binary_file, "ab") as bin_file:
            bin_file.write(compressed_data)
            self.position = bin_file.tell()

    def write_frames(self, frames: List[TimsFrame], scan_mode: int, num_threads: int = 4) -> None:

        # compress frames
        compressed_data = self.helper_handle.compress_frames(
            frames,
            num_threads=num_threads
        )

        # write to binary file
        with open(self.binary_file, "ab") as bin_file:
            # write compressed data to binary file, and add frame meta data to list
            for frame, data in zip(frames, compressed_data):
                self.frame_meta_data.append(self.build_frame_meta_row(frame, scan_mode, self.position))
                bin_file.write(data)
                self.position = bin_file.tell()

    def get_frame_meta_data(self) -> pd.DataFrame:
        return pd.DataFrame(self.frame_meta_data)

    def write_frame_meta_data(self) -> None:
        self._create_table(self.conn, self.get_frame_meta_data(), "Frames")

    def write_dia_ms_ms_info(self, dia_ms_ms_info: pd.DataFrame) -> None:
        out = dia_ms_ms_info.rename(columns={
            'frame': 'Frame',
            'window_group': 'WindowGroup',
        })

        self._create_table(self.conn, out, "DiaFrameMsMsInfo")

    def write_dia_ms_ms_windows(self, dia_ms_ms_windows: pd.DataFrame) -> None:
        out = dia_ms_ms_windows.rename(columns={
            'window_group': 'WindowGroup',
            'scan_start': 'ScanNumBegin',
            'scan_end': 'ScanNumEnd',
            'isolation_mz': 'IsolationMz',
            'isolation_width': 'IsolationWidth',
            'collision_energy': 'CollisionEnergy',
        })

        self._create_table(self.conn, out, "DiaFrameMsMsWindows")