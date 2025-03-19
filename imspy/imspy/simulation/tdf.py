import sqlite3
import pandas as pd
import numpy as np

from pathlib import Path

from imspy.simulation.utility import get_compressible_data
from imspy.timstof import TimsDataset
from imspy.timstof.frame import TimsFrame
import zstd

import imspy_connector
ims = imspy_connector.py_dataset


class TDFWriter:
    def __init__(self, helper_handle: TimsDataset, path: str = "./", exp_name: str = "RAW.d", offset_bytes: int = 64, verbose: bool=False) -> None:

        self.path = Path(path)
        self.exp_name = exp_name
        self.full_path = Path(path) / exp_name
        self.position = 0
        self.binary_file = self.full_path / "analysis.tdf_bin"
        self.frame_meta_data = []
        self.conn = None
        self.helper_handle = helper_handle
        self.offset_bytes = offset_bytes
        self.verbose = verbose

        self.__conn_native = None
        self._setup_connections()

    def _setup_connections(self) -> None:
        # Create the directory and connect to DB
        self.full_path.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(f'{self.full_path}/analysis.tdf')

        # Create the tables for the analysis.tdf
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

        # Create the binary file and add the offset bytes
        # TODO: check if this is necessary
        with open(self.binary_file, "wb") as bin_file:
            bin_file.write(b'\x00' * self.offset_bytes)
            self.position = bin_file.tell()

        if self.verbose:
            print(f"Setting up TDF file meta data, created: {self.full_path}/analysis.tdf and {self.full_path}/analysis.tdf_bin")

    @staticmethod
    def _get_table(conn, table_name: str) -> pd.DataFrame:
        # Get a table as a pandas DataFrame
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)

    @staticmethod
    def _create_table(conn, table, table_name: str) -> None:
        # Create a table from a pandas DataFrame
        table.to_sql(table_name, conn, if_exists='replace', index=False)

    def mz_to_tof(self, frame_id, mzs):
        """Convert m/z values to TOF values for a given frame using the helper handle.
        # CAUTION: This will use the calibration data from the reference handle.
        """
        max_ref_frame_id = self.helper_handle.meta_data.Id.max()
        if frame_id > max_ref_frame_id:
            frame_id = max_ref_frame_id

        return np.array(self.helper_handle.mz_to_tof(frame_id, mzs))

    def tof_to_mz(self, frame_id, tofs):
        """Convert TOF values to m/z values for a given frame using the helper handle.
        # CAUTION: This will use the calibration data from the reference handle.
        """
        max_ref_frame_id = self.helper_handle.meta_data.Id.max()
        if frame_id > max_ref_frame_id:
            frame_id = max_ref_frame_id
        return np.array(self.helper_handle.tof_to_mz(frame_id, tofs))

    def inv_mobility_to_scan(self, frame_id, inv_mobs):
        """Convert inverse mobility values to scan values for a given frame using the helper handle.
        # CAUTION: This will use the calibration data from the reference handle.
        """
        max_ref_frame_id = self.helper_handle.meta_data.Id.max()
        if frame_id > max_ref_frame_id:
            frame_id = max_ref_frame_id
        return np.array(self.helper_handle.inverse_mobility_to_scan(frame_id, inv_mobs))

    def scan_to_inv_mobility(self, frame_id, scans):
        """Convert scan values to inverse mobility values for a given frame using the helper handle.
        # CAUTION: This will use the calibration data from the reference handle.
        """
        max_ref_frame_id = self.helper_handle.meta_data.Id.max()
        if frame_id > max_ref_frame_id:
            frame_id = max_ref_frame_id
        return np.array(self.helper_handle.scan_to_inverse_mobility(frame_id, scans))

    def __repr__(self) -> str:
        return f'TDFWriter(path={self.path}, db_name={self.exp_name}, num_scans={self.helper_handle.num_scans}, ' \
               f'im_lower={self.helper_handle.im_lower}, im_upper={self.helper_handle.im_upper}, mz_lower={self.helper_handle.mz_lower}, ' \
               f'mz_upper={self.helper_handle.mz_upper})'

    def build_frame_meta_row(self, frame: TimsFrame, scan_mode: int, frame_start_pos: int, only_frame_one: bool = False):
        """Build a row for the frame meta data table from a TimsFrame object.
            Arguments:
                frame: TimsFrame object
                scan_mode: int
                frame_start_pos: int
                only_frame_one: bool
        """
        max_index = self.helper_handle.meta_data.Id.max()

        r = self.helper_handle.meta_data.iloc[0, :].copy()
        if not only_frame_one:
            # check for index out of bounds since ref data handle might not hold same number of frames
            if frame.frame_id > max_index:
                r = self.helper_handle.meta_data.iloc[max_index - 1, :].copy()
            else:
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
        """Compress a single frame using zstd.
            Arguments:
                frame: TimsFrame object
                only_frame_one: bool

            Returns:
                bytes: compressed data
        """
        # either use frame 1 or the ref handle frame for writing of calibration data and call to conversion function
        i = 1 if only_frame_one else frame.frame_id
        max_index = self.helper_handle.meta_data.Id.max()
        if frame.frame_id > max_index and not only_frame_one:
            i = max_index

        # transform mz and mobility to tof and scan
        tof = self.mz_to_tof(i, frame.mz).astype(np.uint32)
        scan = self.inv_mobility_to_scan(i, frame.mobility).astype(np.uint32)
        intensity = frame.intensity.astype(np.uint32)
        # get the real data as interleaved bytes
        real_data = get_compressible_data(tof, scan, intensity, self.helper_handle.num_scans)
        # compress the data
        return zstd.ZSTD_compress(bytes(real_data), 0)

    def write_frame(self, frame: TimsFrame, scan_mode: int, only_frame_one: bool = False) -> None:
        """Write a single frame to the binary file.
            Arguments:
                frame: TimsFrame object
                scan_mode: int
                only_frame_one: bool
        """
        self.frame_meta_data.append(self.build_frame_meta_row(frame, scan_mode, self.position, only_frame_one))
        compressed_data = self.compress_frame(frame, only_frame_one)

        with open(self.binary_file, "ab") as bin_file:
            bin_file.write(
                (len(compressed_data) + 8).to_bytes(4, "little", signed=False)
            )
            bin_file.write(int(self.helper_handle.num_scans).to_bytes(4, "little", signed=False))
            bin_file.write(compressed_data)
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

    def write_precursor_table(self, precursor_table: pd.DataFrame) -> None:
        out = precursor_table.rename(columns={
            'id': 'Id',
            'largest_peak_mz': 'LargestPeakMz',
            'average_mz': 'AverageMz',
            'monoisotopic_mz': 'MonoisotopicMz',
            'charge': 'Charge',
            'scan_number': 'ScanNumber',
            'intensity': 'Intensity',
            'parent': 'Parent',
        })
        self._create_table(self.conn, out, "Precursors")

    def write_pasef_meta_table(self, pasef_meta_table: pd.DataFrame) -> None:
        out = pasef_meta_table.rename(columns={
            'frame': 'Frame',
            'scan_start': 'ScanNumBegin',
            'scan_end': 'ScanNumEnd',
            'isolation_mz': 'IsolationMz',
            'isolation_width': 'IsolationWidth',
            'collision_energy': 'CollisionEnergy',
            'precursor': 'Precursor',
        })
        self._create_table(self.conn, out, "PasefMeta")

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

        # TODO: these methods needs to be debugged
        """
        def compress_frames(self, frames: List[TimsFrame], only_frame_one: bool = False, num_threads: int = 4) -> List[bytes]:
            # same as compress_frame but for multiple frames
            tofs, scans, intensities = [], [], []
            for frame in frames:
                i = 1 if only_frame_one else frame.frame_id
                tofs.append(self.mz_to_tof(i, frame.mz).astype(np.uint32))
                scans.append(self.inv_mobility_to_scan(i, frame.mobility).astype(np.uint32))
                intensities.append(frame.intensity.astype(np.uint32))

            real_data = ims.get_data_for_compression_par(tofs, scans, intensities, self.helper_handle.num_scans, num_threads)
            return [zstd.ZSTD_compress(bytes(data), 1) for data in real_data]

        def write_frames(self, frames: List[TimsFrame], scan_mode: int, only_frame_one: bool = False, num_threads: int = 4) -> None:

            compressed_data = self.compress_frames(frames, only_frame_one, num_threads=num_threads)

            for i, data in enumerate(compressed_data):

                self.frame_meta_data.append(self.build_frame_meta_row(frames[i], scan_mode, self.position, only_frame_one))

                with open(self.binary_file, "ab") as bin_file:
                    bin_file.write(
                        (len(data) + 8).to_bytes(4, "little", signed=False)
                    )
                    bin_file.write(int(self.helper_handle.num_scans).to_bytes(4, "little", signed=False))
                    bin_file.write(data)
                    self.position = bin_file.tell()
        """