import sqlite3
import os
from abc import ABC
from typing import List

import numpy as np
import pandas as pd
import imspy_connector as pims

from imspy.core import TimsFrame


class TimsTofSyntheticAcquisitionBuilderDDA:
    def __init__(self, db_path: str):
        self.handle = pims.PyTimsTofSyntheticsDDA(db_path)

    def get_frame(self, frame_id: int):
        frame = self.handle.build_frame(frame_id)
        return TimsFrame.from_py_tims_frame(frame)

    def build_frames(self, frame_ids: List[int], num_threads: int = 4):
        frames = self.handle.build_frames(frame_ids, num_threads)
        return [TimsFrame.from_py_tims_frame(frame) for frame in frames]


class ExperimentDataHandle:
    def __init__(self,
                 database_path: str,
                 database_name: str = 'experiment_data.db',
                 verbose: bool = True,
                 ):
        self.verbose = verbose
        self.base_path = database_path
        self.database_path = os.path.join(self.base_path, database_name)
        self.raw_data_path = os.path.join(self.base_path, 'raw_data')
        self.conn = None

        self._setup()

    def _setup(self):
        # Create raw_data directory if it doesn't exist
        if not os.path.exists(self.raw_data_path):
            if self.verbose:
                print(f"Creating raw data directory: {self.raw_data_path}")
            os.makedirs(self.raw_data_path)

        # Connect to the SQLite database or create it if it doesn't exist
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


class ExperimentDataHandleDIA(ExperimentDataHandle, ABC):
    def __init__(self,
                 database_path: str,
                 database_name: str = 'experiment_data.db',
                 verbose: bool = True,):
        super().__init__(database_path, database_name, verbose)
        self.dia_ms_ms_info = None
        self.dia_ms_ms_windows = None

        self._additional_setup()

    def _additional_setup(self):
        self.dia_ms_ms_info = self.get_table('dia_ms_ms_info')
        self.dia_ms_ms_windows = self.get_table('dia_ms_ms_windows')


if __name__ == '__main__':

    # Example usage
    path = '/path/to/directory'
    db_name = 'experiment_data.db'
    handle = ExperimentDataHandle(path, db_name)

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
