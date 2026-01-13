import sqlite3
import pandas as pd
import numpy as np

from pathlib import Path

from numpy._typing import NDArray

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

        try:
            last_frame = self.helper_handle.meta_data.Id.max()
        except AttributeError as e:
            last_frame = self.helper_handle.meta_data.frame_id.max()

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

        try:
            max_ref_frame_id = self.helper_handle.meta_data.Id.max()

        except AttributeError as e:
            max_ref_frame_id = self.helper_handle.meta_data.frame_id.max()

        if frame_id > max_ref_frame_id:
            frame_id = max_ref_frame_id

        return np.array(self.helper_handle.mz_to_tof(frame_id, mzs))

    def tof_to_mz(self, frame_id, tofs):
        """Convert TOF values to m/z values for a given frame using the helper handle.
        # CAUTION: This will use the calibration data from the reference handle.
        """

        try:
            max_ref_frame_id = self.helper_handle.meta_data.Id.max()
        except AttributeError as e:
            max_ref_frame_id = self.helper_handle.meta_data.frame_id.max()
        if frame_id > max_ref_frame_id:
            frame_id = max_ref_frame_id
        return np.array(self.helper_handle.tof_to_mz(frame_id, tofs))

    def inv_mobility_to_scan(self, frame_id, inv_mobs):
        """Convert inverse mobility values to scan values for a given frame using the helper handle.
        # CAUTION: This will use the calibration data from the reference handle.
        """
        try:
            max_ref_frame_id = self.helper_handle.meta_data.Id.max()
        except AttributeError as e:
            max_ref_frame_id = self.helper_handle.meta_data.frame_id.max()
        if frame_id > max_ref_frame_id:
            frame_id = max_ref_frame_id
        return np.array(self.helper_handle.inverse_mobility_to_scan(frame_id, inv_mobs))

    def scan_to_inv_mobility(self, frame_id, scans):
        """Convert scan values to inverse mobility values for a given frame using the helper handle.
        # CAUTION: This will use the calibration data from the reference handle.
        """
        try:
            max_ref_frame_id = self.helper_handle.meta_data.Id.max()
        except AttributeError as e:
            max_ref_frame_id = self.helper_handle.meta_data.frame_id.max()
        if frame_id > max_ref_frame_id:
            frame_id = max_ref_frame_id
        return np.array(self.helper_handle.scan_to_inverse_mobility(frame_id, scans))

    def __repr__(self) -> str:
        return f'TDFWriter(path={self.path}, db_name={self.exp_name}, num_scans={self.helper_handle.num_scans}, ' \
               f'im_lower={self.helper_handle.im_lower}, im_upper={self.helper_handle.im_upper}, mz_lower={self.helper_handle.mz_lower}, ' \
               f'mz_upper={self.helper_handle.mz_upper})'

    def build_frame_meta_row(
            self,
            intensity: NDArray,
            frame: TimsFrame,
            scan_mode: int,
            frame_start_pos: int,
            only_frame_one: bool = False
    ):
        """Build a row for the frame meta data table from a TimsFrame object.
            Arguments:
                intensity: NDArray
                frame: TimsFrame object
                scan_mode: int
                frame_start_pos: int
                only_frame_one: bool
        """
        try:
            max_index = self.helper_handle.meta_data.Id.max()
        except AttributeError as e:
            max_index = self.helper_handle.meta_data.frame_id.max()

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
        r.MaxIntensity = int(np.max(intensity)) if len(intensity) > 0 else 0
        r.SummedIntensities = int(np.sum(intensity)) if len(intensity) > 0 else 0
        r.NumScans = self.helper_handle.num_scans
        r.NumPeaks = len(intensity)

        return r

    def compress_frame(self, frame: TimsFrame, only_frame_one: bool = False) -> (NDArray, bytes):
        """Compress a single frame using zstd.
            Arguments:
                frame: TimsFrame object
                only_frame_one: bool

            Returns:
                bytes: intensities, compressed data
        """
        # either use frame 1 or the ref handle frame for writing of calibration data and call to conversion function
        i = 1 if only_frame_one else frame.frame_id

        try:
            max_index = self.helper_handle.meta_data.Id.max()
        except AttributeError as e:
            max_index = self.helper_handle.meta_data.frame_id.max()

        if frame.frame_id > max_index and not only_frame_one:
            i = max_index

        # transform mz and mobility to tof and scan
        tof = self.mz_to_tof(i, frame.mz).astype(np.uint32)
        scan = self.inv_mobility_to_scan(i, frame.mobility).astype(np.uint32)
        intensity = frame.intensity.astype(np.uint32)

        # Since, mz -> tof is not bijective, we need to check for duplicates
        # stack scan and tof to form a 2D array for unique grouping
        scan_tof = np.stack((scan, tof), axis=1)

        # get unique (scan, tof) pairs and their inverse indices
        unique_pairs, inverse_indices = np.unique(scan_tof, axis=0, return_inverse=True)

        # sum intensities for each unique (scan, tof) pair
        summed_intensity = np.bincount(inverse_indices, weights=intensity)

        # now split back scan and tof
        unique_scan = unique_pairs[:, 0]
        unique_tof = unique_pairs[:, 1]

        # sort first by scan, then by tof
        sort_idx = np.lexsort((unique_tof, unique_scan))

        # final sorted arrays
        scan = unique_scan[sort_idx]
        tof = unique_tof[sort_idx]
        intensity = summed_intensity[sort_idx].astype(np.uint32)

        # get the real data as interleaved bytes
        real_data = get_compressible_data(tof, scan, intensity, self.helper_handle.num_scans)
        # compress the data
        return intensity, zstd.ZSTD_compress(bytes(real_data), 0)

    def write_frame(self, frame: TimsFrame, scan_mode: int, only_frame_one: bool = False) -> None:
        """Write a single frame to the binary file.
            Arguments:
                frame: TimsFrame object
                scan_mode: int
                only_frame_one: bool
        """
        intensity, compressed_data = self.compress_frame(frame, only_frame_one)

        self.frame_meta_data.append(
            self.build_frame_meta_row(
                intensity,
                frame,
                scan_mode,
                self.position,
                only_frame_one
        ))

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

    def write_calibration_info(self, mz_standard_deviation_ppm: float = 0.15) -> None:
        try:
            table = self.helper_handle.get_table("CalibrationInfo")
            table.iloc[5].Value = str(mz_standard_deviation_ppm)
            self._create_table(self.conn, table, "CalibrationInfo")

        except Exception as e:
            print(f"Error writing calibration info table: {e}")

    def write_pasef_frame_ms_ms_info(self) -> None:
        try:
            self._create_table(self.conn, self.helper_handle.get_table("PasefFrameMsMsInfo"), "PasefFrameMsMsInfo")
        except Exception as e:
            print(f"Error writing PasefFrameMsMsInfo table: {e}. In most cases, this is not a problem, since the table is empty in DIA mode.")

    def write_prm_frame_ms_ms_info(self) -> None:
        try:
            self._create_table(self.conn, self.helper_handle.get_table("PrmFrameMsMsInfo"), "PrmFrameMsMsInfo")
        except Exception as e:
            print(f"Error writing PrmFrameMsMsInfo table: {e}")

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
        self._create_table(self.conn, out, "PasefFrameMsMsInfo")

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

    def compress_frames_batch(
            self,
            frames: list,
            only_frame_one: bool = False,
            num_threads: int = 4,
            compression_level: int = 0
    ) -> list:
        """Compress multiple frames using parallel Rust implementation with duplicate merging.

        This method uses the Rust `compress_frames_with_merge` function which:
        1. Merges duplicate (scan, tof) pairs by summing intensities
        2. Sorts by scan, then tof
        3. Compresses frames in parallel using ZSTD

        Arguments:
            frames: List of TimsFrame objects
            only_frame_one: If True, use frame 1 calibration for all frames
            num_threads: Number of parallel threads for compression
            compression_level: ZSTD compression level (0 = default)

        Returns:
            List of compressed frame data (bytes), already includes header with size and scan count
        """
        # Convert frames to (scan, tof, intensity) tuples
        frame_data = []
        for frame in frames:
            i = 1 if only_frame_one else frame.frame_id

            try:
                max_index = self.helper_handle.meta_data.Id.max()
            except AttributeError:
                max_index = self.helper_handle.meta_data.frame_id.max()

            if frame.frame_id > max_index and not only_frame_one:
                i = max_index

            tof = self.mz_to_tof(i, frame.mz).astype(np.uint32)
            scan = self.inv_mobility_to_scan(i, frame.mobility).astype(np.uint32)
            intensity = frame.intensity.astype(np.uint32)

            # Convert to lists for Rust
            frame_data.append((
                scan.tolist(),
                tof.tolist(),
                intensity.tolist()
            ))

        # Parallel compression in Rust with duplicate merging
        compressed = ims.compress_frames_with_merge(
            frame_data,
            self.helper_handle.num_scans,
            compression_level,
            num_threads
        )
        return compressed

    def write_frames_batch(
            self,
            frames: list,
            scan_mode: int,
            only_frame_one: bool = False,
            num_threads: int = 4,
            compression_level: int = 0
    ) -> None:
        """Write multiple frames using parallel Rust compression.

        Arguments:
            frames: List of TimsFrame objects
            scan_mode: Scan mode value for frame metadata
            only_frame_one: If True, use frame 1 calibration for all frames
            num_threads: Number of parallel threads for compression
            compression_level: ZSTD compression level (0 = default)
        """
        if not frames:
            return

        # Compress all frames in parallel using Rust
        compressed_batch = self.compress_frames_batch(
            frames, only_frame_one, num_threads, compression_level
        )

        # Build metadata for all frames (needed for intensity stats)
        # Do this before writing since we need intensity values
        metadata_rows = []
        for frame in frames:
            i = 1 if only_frame_one else frame.frame_id

            try:
                max_index = self.helper_handle.meta_data.Id.max()
            except AttributeError:
                max_index = self.helper_handle.meta_data.frame_id.max()

            if frame.frame_id > max_index and not only_frame_one:
                i = max_index

            tof = self.mz_to_tof(i, frame.mz).astype(np.uint32)
            scan = self.inv_mobility_to_scan(i, frame.mobility).astype(np.uint32)
            intensity = frame.intensity.astype(np.uint32)

            # Merge duplicates for accurate intensity stats (using Rust)
            merged_scan, merged_tof, merged_intensity = ims.merge_and_sort_peaks(
                scan.tolist(), tof.tolist(), intensity.tolist()
            )

            metadata_rows.append((frame, merged_intensity))

        # Write all compressed data sequentially (I/O is fast, no need to parallelize)
        with open(self.binary_file, "ab") as bin_file:
            for i, compressed_data in enumerate(compressed_batch):
                frame, merged_intensity = metadata_rows[i]

                # Build frame metadata with position before writing
                self.frame_meta_data.append(
                    self.build_frame_meta_row(
                        merged_intensity,
                        frame,
                        scan_mode,
                        self.position,
                        only_frame_one
                    )
                )

                # Write the compressed data (it already has the header from Rust)
                bin_file.write(bytes(compressed_data))
                self.position = bin_file.tell()