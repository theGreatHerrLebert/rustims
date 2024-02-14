import sqlite3
import os
from abc import ABC
from typing import List

import pandas as pd
import imspy_connector as pims

from imspy.core import TimsFrame


class TimsTofSyntheticFrameBuilderDIA:
    def __init__(self, db_path: str):
        self.handle = pims.PyTimsTofSyntheticsFrameBuilderDIA(db_path)

    def build_frame(self, frame_id: int, fragment: bool = True) -> TimsFrame:
        frame = self.handle.build_frame(frame_id, fragment)
        return TimsFrame.from_py_tims_frame(frame)

    def build_frames(self, frame_ids: List[int], fragment: bool = True, num_threads: int = 4) -> List[TimsFrame]:
        frames = self.handle.build_frames(frame_ids, fragment, num_threads)
        return [TimsFrame.from_py_tims_frame(frame) for frame in frames]

    def get_collision_energy(self, frame_id: int, scan_id: int) -> float:
        return self.handle.get_collision_energy(frame_id, scan_id)

    def get_collision_energies(self, frame_ids: List[int], scan_ids: List[int]) -> List[float]:
        return self.handle.get_collision_energies(frame_ids, scan_ids)


class TimsTofSyntheticFrameBuilder:
    def __init__(self, db_path: str):
        self.handle = pims.PyTimsTofSyntheticsFrameBuilder(db_path)

    def build_frame(self, frame_id: int):
        frame = self.handle.build_frame(frame_id)
        return TimsFrame.from_py_tims_frame(frame)

    def build_frames(self, frame_ids: List[int], num_threads: int = 4):
        frames = self.handle.build_frames(frame_ids, num_threads)
        return [TimsFrame.from_py_tims_frame(frame) for frame in frames]


class SyntheticExperimentDataHandle:
    def __init__(self,
                 database_path: str,
                 database_name: str = 'synthetic_data.db',
                 verbose: bool = True,
                 ):
        self.verbose = verbose
        self.base_path = database_path
        self.database_path = os.path.join(self.base_path, database_name)
        self.conn = None

        self._setup()

    def _setup(self):
        if not os.path.exists(self.base_path):
            if self.verbose:
                print(f"Creating data directory: {self.base_path}")
            os.makedirs(self.base_path)
        if self.verbose:
            print(f"Connecting to database: {self.database_path}")
        self.conn = sqlite3.connect(self.database_path)

    def create_table(self, table_name: str, table: pd.DataFrame):
        # Create a table from a pandas DataFrame
        table.to_sql(table_name, self.conn, if_exists='replace', index=False)

    def create_table_sql(self, sql):
        # Create a table as per the provided SQL statement
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")

    def close(self):
        # Close the database connection
        if self.conn:
            self.conn.close()

    def get_table(self, table_name: str) -> pd.DataFrame:
        # Get a table as a pandas DataFrame
        return pd.read_sql(f"SELECT * FROM {table_name}", self.conn)


class SyntheticExperimentDataHandleDIA(SyntheticExperimentDataHandle, ABC):
    def __init__(self,
                 database_path: str,
                 database_name: str = 'synthetic_data.db',
                 verbose: bool = True,):
        super().__init__(database_path, database_name, verbose)
        self.dia_ms_ms_info = None
        self.dia_ms_ms_windows = None

        self._additional_setup()

    def _additional_setup(self):
        self.dia_ms_ms_info = self.get_table('dia_ms_ms_info')
        self.dia_ms_ms_windows = self.get_table('dia_ms_ms_windows')

    def get_frame_to_window_group(self):
        return dict(zip(self.dia_ms_ms_info.frame, self.dia_ms_ms_info.window_group))

    def get_window_group_settings(self):
        window_group_settings = {}

        for _, row in self.dia_ms_ms_windows.iterrows():
            key = (row.window_group, row.scan_start)
            value = (row.mz_mid, row.mz_width)
            window_group_settings[key] = value

        return window_group_settings


if __name__ == '__main__':

    # Example usage
    path = '/path/to/directory'
    db_name = 'experiment_data.db'
    handle = SyntheticExperimentDataHandle(path, db_name)

    # Create a table, for example
    sql_create_peptides_table = '''
    CREATE TABLE IF NOT EXISTS peptides (
        peptide_id INTEGER PRIMARY KEY,
        sequence TEXT NOT NULL,
        monoisotopic_mass REAL)
    '''
    handle.create_table(sql_create_peptides_table)

    # Close the connection when done
    handle.close()
