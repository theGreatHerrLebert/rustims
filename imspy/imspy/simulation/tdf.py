import sqlite3
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
from imspy.simulation.utility import get_native_dataset_path
from imspy.core.frame import TimsFrame
from imspy.timstof import TimsDataset
import shutil


class TDFWriter:
    def __init__(
            self,
            path: str = "./",
            exp_name: str = "RAW.d",
            num_scans: int = 917,
            im_lower: float = 0.6,
            im_upper: float = 1.6,
            mz_lower: float = 100.0,
            mz_upper: float = 1700.0)\
            -> None:

        self.path = Path(path)
        self.exp_name = exp_name
        self.full_path = Path(path) / exp_name
        self.num_scans = num_scans
        self.im_lower = im_lower
        self.im_upper = im_upper
        self.mz_lower = mz_lower
        self.mz_upper = mz_upper
        self.position = 0
        self.binary_file = self.full_path / "analysis.tdf_bin"
        self.frame_meta_data = []
        self.conn = None
        self.__helper_handle = None
        self.__conn_native = None
        self._setup_connections()

    def _setup_connections(self) -> None:
        # Create the directory and connect to DB
        self.full_path.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(f'{self.full_path}/analysis.tdf')

        # generate tables for analysis.tdf DB
        tims_calib_name, tims_calib_tbl = self._create_tims_calibration_table(num_scans=self.num_scans)
        mz_calib_name, mz_calib_tbl = self._create_mz_calibration_table()
        global_meta_name, global_meta_tbl = self._create_global_meta_data_table(im_lower=self.im_lower,
                                                                                im_upper=self.im_upper,
                                                                                mz_lower=self.mz_lower,
                                                                                mz_upper=self.mz_upper)
        # Save table to analysis.tdf
        self._create_table(self.conn, mz_calib_tbl, mz_calib_name)
        self._create_table(self.conn, tims_calib_tbl, tims_calib_name)
        self._create_table(self.conn, global_meta_tbl, global_meta_name)

        # Build reference for correct translation of mz -> tof and inv_mob -> scan
        native = self.full_path / ".native/"
        native.mkdir(parents=True, exist_ok=True)
        ref_location = get_native_dataset_path()
        source_file = ref_location
        destination_file = self.full_path / ".native/"
        shutil.copytree(source_file, destination_file, dirs_exist_ok=True)
        self.__conn_native = sqlite3.connect(f'{native}/analysis.tdf')
        self._create_table(self.__conn_native, mz_calib_tbl, mz_calib_name)
        self._create_table(self.__conn_native, tims_calib_tbl, tims_calib_name)
        self._create_table(self.__conn_native, global_meta_tbl, global_meta_name)

        # setup handle for function calls
        self.__helper_handle = TimsDataset(str(native))

    @staticmethod
    def _get_table(conn, table_name: str) -> pd.DataFrame:
        # Get a table as a pandas DataFrame
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)

    @staticmethod
    def _create_table(conn, table, table_name: str) -> None:
        # Create a table from a pandas DataFrame
        table.to_sql(table_name, conn, if_exists='replace', index=False)

    @staticmethod
    def _create_mz_calibration_table() -> (str, pd.DataFrame):
        col_names = ['Id', 'ModelType', 'DigitizerTimebase', 'DigitizerDelay', 'T1', 'T2',
                     'dC1', 'dC2', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
                     'C9', 'C10', 'C11', 'C12', 'C13', 'C14']

        values = [1.0, 2.0, 0.19999999999999998, 25726.199999999997, 25.66893460672872, 25.10638483329421, 20.0, 0.0,
                  313.51167546836524, 154833.4542880268, -7.3593189552069465e-06, 313.51167546836524,
                  -7.3593189552069465e-06,
                  225.951491, 1519.712539, 7.0, 0.058519441907584555, -0.0005411344569873044, 1.8634987450927556e-06,
                  -3.1297916646308927e-09,
                  2.7471965457754794e-12, -1.207997933655641e-15, 2.095848318930801e-19]

        return "MzCalibration", pd.DataFrame([values], columns=col_names)

    @staticmethod
    def _create_tims_calibration_table(num_scans: int = 917) -> (str, pd.DataFrame):
        col_names = ['Id', 'ModelType', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

        values = [1, 2, 1, num_scans, 218.720487393, 73.404989154, 33.027522936, 1.0, 0.042931657,
                  127.310271708, 12.676243115, 4414.816879989]

        return "TimsCalibration", pd.DataFrame([values], columns=col_names)

    @staticmethod
    def _create_global_meta_data_table(im_lower: float, im_upper: float,
                                       mz_lower: float, mz_upper) -> (str, pd.DataFrame):
        global_meta = {'SchemaType': 'TDF',
                       'SchemaVersionMajor': '3',
                       'SchemaVersionMinor': '5',
                       'AcquisitionSoftwareVendor': 'Bruker',
                       'InstrumentVendor': 'Bruker',
                       'ClosedProperly': '1',
                       'TimsCompressionType': '2',
                       'MaxNumPeaksPerScan': '1661',
                       'AnalysisId': '00000000-0000-0000-0000-000000000000',
                       'DigitizerNumSamples': '396854',
                       'MzAcqRangeLower': f'{mz_lower}',
                       'MzAcqRangeUpper': f'{mz_upper}',
                       'AcquisitionSoftware': 'timsTOF',
                       'AcquisitionSoftwareVersion': '2.0.18.0',
                       'AcquisitionFirmwareVersion': 'I4IT-9.481.28.81; ITPT-9.481.28.81; ITET-9.481.28.81; '
                                                     'FXM3-0.0.1.6; MXMC-0.0.3.4; MXIF-0.0.1.1; MXRF-0.0.1.1; '
                                                     'RFXS-0.1.3.1; RFXE-NOT_PRESENT',
                       'AcquisitionDateTime': '2021-01-15T16:15:32.327+01:00',
                       'InstrumentName': 'timsTOF Pro',
                       'InstrumentFamily': '9',
                       'InstrumentRevision': '3',
                       'InstrumentSourceType': '11',
                       'InstrumentSerialNumber': '1854399.00271',
                       'OperatorName': 'Admin',
                       'Description': 'HeLa 200ng/uL',
                       'SampleName': 'M210115_001',
                       'MethodName': '20210113_DDA PASEF-standard_1.1sec_cycletime_1600V.m',
                       'DenoisingEnabled': '0',
                       'PeakWidthEstimateValue': '0.000025',
                       'PeakWidthEstimateType': '1',
                       'PeakListIndexScaleFactor': '1',
                       'OneOverK0AcqRangeLower': f'{im_lower}',
                       'OneOverK0AcqRangeUpper': f'{im_upper}'}

        return "GlobalMetaData", pd.DataFrame(global_meta.items(), columns=['Key', 'Value'])

    def mz_to_tof(self, mzs):
        return self.__helper_handle.mz_to_tof(1, mzs)

    def tof_to_mz(self, tofs):
        return self.__helper_handle.tof_to_mz(1, tofs)

    def inv_mobility_to_scan(self, inv_mobs):
        return self.__helper_handle.inverse_mobility_to_scan(1, inv_mobs)

    def scan_to_inv_mobility(self, scans):
        return self.__helper_handle.scan_to_inverse_mobility(1, scans)

    def __repr__(self) -> str:
        return f"TDFWriter(path={self.path}, db_name={self.exp_name}, num_scans={self.num_scans}, " \
               f"im_lower={self.im_lower}, im_upper={self.im_upper}, mz_lower={self.mz_lower}, " \
               f"mz_upper={self.mz_upper})"

    def build_frame_meta_row(self, frame: TimsFrame, scan_mode: int, frame_start_pos: int):
        r = self.__helper_handle.meta_data.iloc[0, :].copy()
        r.Id = frame.frame_id
        r.Time = frame.retention_time
        r.ScanMode = scan_mode
        r.MsMsType = frame.ms_type
        r.TimsId = frame_start_pos
        r.MaxIntensity = int(np.max(frame.intensity)) if len(frame.intensity) > 0 else 0
        r.SummedIntensities = int(np.sum(frame.intensity)) if len(frame.intensity) > 0 else 0
        r.NumScans = self.num_scans
        r.NumPeaks = len(frame.mz)

        return r

    def compress_frame(self, frame: TimsFrame) -> bytes:
        # calculate TOF using the DH of the other frame
        # TODO: move translation of mz -> tof and inv_mob -> scan to the helper handle
        tof = self.mz_to_tof(frame.mz)
        scan = self.inv_mobility_to_scan(frame.mobility)
        return self.__helper_handle.indexed_values_to_compressed_bytes(scan, tof, frame.intensity,
                                                                       total_scans=self.num_scans)

    def compress_frames(self, frames: List[TimsFrame], num_threads: int = 4) -> List[bytes]:
        return self.__helper_handle.compress_frames(frames, total_scans=self.num_scans, num_threads=num_threads)

    def write_frame(self, frame: TimsFrame, scan_mode: int) -> None:
        self.frame_meta_data.append(self.build_frame_meta_row(frame, scan_mode, self.position))
        compressed_data = self.compress_frame(frame)

        with open(self.binary_file, "ab") as bin_file:
            bin_file.write(compressed_data)
            self.position = bin_file.tell()

    def write_frames(self, frames: List[TimsFrame], scan_mode: int, num_threads: int = 4) -> None:

        # generate meta data
        meta_data = [self.build_frame_meta_row(frame, scan_mode, self.position) for frame in frames]

        # append to frame meta data table
        for data in meta_data:
            self.frame_meta_data.append(data)

        compressed_data = self.__helper_handle.compress_frame_collection(
            frames,
            total_scans=self.num_scans,
            num_threads=num_threads
        )

        # write to binary file
        with open(self.binary_file, "ab") as bin_file:
            for data in compressed_data:
                bin_file.write(data)
                self.position = bin_file.tell()

    def get_frame_meta_data(self) -> pd.DataFrame:
        return pd.DataFrame(self.frame_meta_data)

    def write_frame_meta_data(self) -> None:
        self._create_table(self.conn, self.get_frame_meta_data(), "Frames")
