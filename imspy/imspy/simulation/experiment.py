import sqlite3
import os
from abc import ABC
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

import pandas as pd

from imspy.simulation.annotation import TimsFrameAnnotated, RustWrapperObject
from imspy.timstof.frame import TimsFrame

import imspy_connector
ims = imspy_connector.py_simulation


class TimsTofSyntheticFrameBuilderDDA(RustWrapperObject):
    def __init__(self, db_path: str, with_annotations: bool = False, num_threads: int = 4):
        """Initializes the TimsTofSyntheticFrameBuilderDIA.

        Args:
            db_path (str): Path to the raw data file.
            with_annotations (bool): If true, frame annotations can be created during frame building, but this will slow down the process and needs a lot of extra memory, use with caution.
            num_threads (int): Number of threads.
        """
        self.path = db_path
        self.__py_ptr = ims.PyTimsTofSyntheticsFrameBuilderDDA(db_path, with_annotations, num_threads)

    def build_frame(self,
                    frame_id: int,
                    fragment: bool = True,
                    mz_noise_precursor: bool = False,
                    mz_noise_uniform: bool = False,
                    precursor_noise_ppm: float = 5.,
                    mz_noise_fragment: bool = False,
                    fragment_noise_ppm: float = 5.,
                    right_drag: bool = True) -> TimsFrame:
        """Build a frame.

        Args:
            frame_id (int): Frame ID.
            fragment (bool): if true, frame will undergo synthetic fragmentation if it is a fragment frame,
            otherwise quadrupole isolation will still be applied but no fragmentation.
            mz_noise_precursor (bool): if true, noise will be added to the precursor m/z values.
            mz_noise_uniform (bool): if true, noise will be added to the precursor m/z values uniformly.
            precursor_noise_ppm (float): PPM of the precursor noise.
            mz_noise_fragment (bool): if true, noise will be added to the fragment m/z values.
            fragment_noise_ppm (float): PPM of the fragment noise.
            right_drag (bool): if true, the noise will be shifted to the right.

        Returns:
            TimsFrame: Frame.
        """
        frame = self.__py_ptr.build_frame(frame_id, fragment, mz_noise_precursor, mz_noise_uniform,
                                          precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag)

        return TimsFrame.from_py_ptr(frame)

    def build_frames(self,
                     frame_ids: List[int],
                     fragment: bool = True,
                     mz_noise_precursor: bool = False,
                     mz_noise_uniform: bool = False,
                     precursor_noise_ppm: float = 5.,
                     mz_noise_fragment: bool = False,
                     fragment_noise_ppm: float = 5.,
                     right_drag: bool = True,
                     num_threads: int = 4) -> List[TimsFrame]:
        """Build frames.

        Args:
            frame_ids (List[int]): Frame IDs.
            fragment (bool): if true, frame will undergo synthetic fragmentation if it is a fragment frame,
            otherwise quadrupole isolation will still be applied but no fragmentation.
            mz_noise_precursor (bool): if true, noise will be added to the precursor m/z values.
            mz_noise_uniform (bool): if true, noise will be added to the precursor m/z values uniformly.
            precursor_noise_ppm (float): PPM of the precursor noise.
            mz_noise_fragment (bool): if true, noise will be added to the fragment m/z values.
            fragment_noise_ppm (float): PPM of the fragment noise.
            right_drag (bool): if true, the noise will be shifted to the right.
            num_threads (int): Number of threads.

        Returns:
            List[TimsFrame]: Frames.
        """
        frames = self.__py_ptr.build_frames(frame_ids, fragment, mz_noise_precursor, mz_noise_uniform,
                                            precursor_noise_ppm,
                                            mz_noise_fragment, fragment_noise_ppm, right_drag, num_threads)
        return [TimsFrame.from_py_ptr(frame) for frame in frames]

    def build_frame_annotated(self, frame_id: int, fragment: bool = True, mz_noise_precursor: bool = False,
                              mz_noise_uniform: bool = False, precursor_noise_ppm: float = 5.,
                              mz_noise_fragment: bool = False,
                              fragment_noise_ppm: float = 5., right_drag: bool = True) -> TimsFrameAnnotated:
        """Build a frame. The frame will be annotated.

        Args:
            frame_id (int): Frame ID.
            fragment (bool): if true, frame will undergo synthetic fragmentation if it is a fragment frame,
            otherwise quadrupole isolation will still be applied but no fragmentation.
            mz_noise_precursor (bool): if true, noise will be added to the precursor m/z values.
            mz_noise_uniform (bool): if true, noise will be added to the precursor m/z values uniformly.
            precursor_noise_ppm (float): PPM of the precursor noise.
            mz_noise_fragment (bool): if true, noise will be added to the fragment m/z values.
            fragment_noise_ppm (float): PPM of the fragment noise.
            right_drag (bool): if true, the noise will be shifted to the right.

        Returns:
            TimsFrameAnnotated: Frame.
        """
        frame = self.__py_ptr.build_frame_annotated(frame_id, fragment, mz_noise_precursor, mz_noise_uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag)
        return TimsFrameAnnotated.from_py_ptr(frame)

    def build_frames_annotated(self, frame_ids: List[int], fragment: bool = True, mz_noise_precursor: bool = False,
                               mz_noise_uniform: bool = False, precursor_noise_ppm: float = 5.,
                               mz_noise_fragment: bool = False, fragment_noise_ppm: float = 5.,
                               right_drag: bool = True, num_threads: int = 4) -> List[TimsFrameAnnotated]:
        """Build frames. The frames will be annotated.

        Args:
            frame_ids (List[int]): Frame IDs.
            fragment (bool): if true, frame will undergo synthetic fragmentation if it is a fragment frame,
            otherwise quadrupole isolation will still be applied but no fragmentation.
            mz_noise_precursor (bool): if true, noise will be added to the precursor m/z values.
            mz_noise_uniform (bool): if true, noise will be added to the precursor m/z values uniformly.
            precursor_noise_ppm (float): PPM of the precursor noise.
            mz_noise_fragment (bool): if true, noise will be added to the fragment m/z values.
            fragment_noise_ppm (float): PPM of the fragment noise.
            right_drag (bool): if true, the noise will be shifted to the right.
            num_threads (int): Number of threads.

        Returns:
            List[TimsFrameAnnotated]: Frames.
        """
        frames = self.__py_ptr.build_frames_annotated(frame_ids, fragment, mz_noise_precursor, mz_noise_uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, num_threads)
        return [TimsFrameAnnotated.from_py_ptr(frame) for frame in frames]

    def get_collision_energy(self, frame_id: int, scan_id: int) -> float:
        return self.__py_ptr.get_collision_energy(frame_id, scan_id)

    def get_collision_energies(self, frame_ids: List[int], scan_ids: List[int]) -> List[float]:
        return self.__py_ptr.get_collision_energies(frame_ids, scan_ids)

    def __repr__(self):
        return f"TimsTofSyntheticFrameBuilderDDA(path={self.path})"

    @classmethod
    def from_py_ptr(cls, py_ptr: ims.PyTimsTofSyntheticsFrameBuilderDDA) -> 'TimsTofSyntheticFrameBuilderDDA':
        """Create a TimsTofSyntheticFrameBuilderDDA from a PyTimsTofSyntheticsFrameBuilderDDA.

        Args:
            py_ptr (ims.PyTimsTofSyntheticsFrameBuilderDDA): PyTimsTofSyntheticsFrameBuilderDDA.

        Returns:
            TimsTofSyntheticFrameBuilderDDA: TimsTofSyntheticFrameBuilderDDA.
        """
        builder = cls.__new__(cls)
        builder.__py_ptr = py_ptr
        return builder

    def get_py_ptr(self) -> ims.PyTimsTofSyntheticsFrameBuilderDDA:
        return self.__py_ptr

class TimsTofSyntheticFrameBuilderDIA(RustWrapperObject):
    def __init__(self, db_path: str, with_annotations: bool = False, num_threads: int = 4):
        """Initializes the TimsTofSyntheticFrameBuilderDIA.

        Args:
            db_path (str): Path to the raw data file.
            with_annotations (bool): If true, frame annotations can be created during frame building, but this will slow down the process and needs a lot of extra memory, use with caution.
            num_threads (int): Number of threads.
        """
        self.path = db_path
        self.__py_ptr = ims.PyTimsTofSyntheticsFrameBuilderDIA(db_path, with_annotations, num_threads)

    def build_frame(self,
                    frame_id: int,
                    fragment: bool = True,
                    mz_noise_precursor: bool = False,
                    mz_noise_uniform: bool = False,
                    precursor_noise_ppm: float = 5.,
                    mz_noise_fragment: bool = False,
                    fragment_noise_ppm: float = 5.,
                    right_drag: bool = True) -> TimsFrame:
        """Build a frame.

        Args:
            frame_id (int): Frame ID.
            fragment (bool): if true, frame will undergo synthetic fragmentation if it is a fragment frame,
            otherwise quadrupole isolation will still be applied but no fragmentation.
            mz_noise_precursor (bool): if true, noise will be added to the precursor m/z values.
            mz_noise_uniform (bool): if true, noise will be added to the precursor m/z values uniformly.
            precursor_noise_ppm (float): PPM of the precursor noise.
            mz_noise_fragment (bool): if true, noise will be added to the fragment m/z values.
            fragment_noise_ppm (float): PPM of the fragment noise.
            right_drag (bool): if true, the noise will be shifted to the right.

        Returns:
            TimsFrame: Frame.
        """
        frame = self.__py_ptr.build_frame(frame_id, fragment, mz_noise_precursor, mz_noise_uniform,
                                          precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag)

        return TimsFrame.from_py_ptr(frame)

    def build_frames(self,
                     frame_ids: List[int],
                     fragment: bool = True,
                     mz_noise_precursor: bool = False,
                     mz_noise_uniform: bool = False,
                     precursor_noise_ppm: float = 5.,
                     mz_noise_fragment: bool = False,
                     fragment_noise_ppm: float = 5.,
                     right_drag: bool = True,
                     num_threads: int = 4) -> List[TimsFrame]:
        """Build frames.

        Args:
            frame_ids (List[int]): Frame IDs.
            fragment (bool): if true, frame will undergo synthetic fragmentation if it is a fragment frame,
            otherwise quadrupole isolation will still be applied but no fragmentation.
            mz_noise_precursor (bool): if true, noise will be added to the precursor m/z values.
            mz_noise_uniform (bool): if true, noise will be added to the precursor m/z values uniformly.
            precursor_noise_ppm (float): PPM of the precursor noise.
            mz_noise_fragment (bool): if true, noise will be added to the fragment m/z values.
            fragment_noise_ppm (float): PPM of the fragment noise.
            right_drag (bool): if true, the noise will be shifted to the right.
            num_threads (int): Number of threads.

        Returns:
            List[TimsFrame]: Frames.
        """
        frames = self.__py_ptr.build_frames(frame_ids, fragment, mz_noise_precursor, mz_noise_uniform,
                                            precursor_noise_ppm,
                                            mz_noise_fragment, fragment_noise_ppm, right_drag, num_threads)
        return [TimsFrame.from_py_ptr(frame) for frame in frames]

    def build_frame_annotated(self, frame_id: int, fragment: bool = True, mz_noise_precursor: bool = False,
                              mz_noise_uniform: bool = False, precursor_noise_ppm: float = 5.,
                              mz_noise_fragment: bool = False,
                              fragment_noise_ppm: float = 5., right_drag: bool = True) -> TimsFrameAnnotated:
        """Build a frame. The frame will be annotated.

        Args:
            frame_id (int): Frame ID.
            fragment (bool): if true, frame will undergo synthetic fragmentation if it is a fragment frame,
            otherwise quadrupole isolation will still be applied but no fragmentation.
            mz_noise_precursor (bool): if true, noise will be added to the precursor m/z values.
            mz_noise_uniform (bool): if true, noise will be added to the precursor m/z values uniformly.
            precursor_noise_ppm (float): PPM of the precursor noise.
            mz_noise_fragment (bool): if true, noise will be added to the fragment m/z values.
            fragment_noise_ppm (float): PPM of the fragment noise.
            right_drag (bool): if true, the noise will be shifted to the right.

        Returns:
            TimsFrameAnnotated: Frame.
        """
        frame = self.__py_ptr.build_frame_annotated(frame_id, fragment, mz_noise_precursor, mz_noise_uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag)
        return TimsFrameAnnotated.from_py_ptr(frame)

    def build_frames_annotated(self, frame_ids: List[int], fragment: bool = True, mz_noise_precursor: bool = False,
                               mz_noise_uniform: bool = False, precursor_noise_ppm: float = 5.,
                               mz_noise_fragment: bool = False, fragment_noise_ppm: float = 5.,
                               right_drag: bool = True, num_threads: int = 4) -> List[TimsFrameAnnotated]:
        """Build frames. The frames will be annotated.

        Args:
            frame_ids (List[int]): Frame IDs.
            fragment (bool): if true, frame will undergo synthetic fragmentation if it is a fragment frame,
            otherwise quadrupole isolation will still be applied but no fragmentation.
            mz_noise_precursor (bool): if true, noise will be added to the precursor m/z values.
            mz_noise_uniform (bool): if true, noise will be added to the precursor m/z values uniformly.
            precursor_noise_ppm (float): PPM of the precursor noise.
            mz_noise_fragment (bool): if true, noise will be added to the fragment m/z values.
            fragment_noise_ppm (float): PPM of the fragment noise.
            right_drag (bool): if true, the noise will be shifted to the right.
            num_threads (int): Number of threads.

        Returns:
            List[TimsFrameAnnotated]: Frames.
        """
        frames = self.__py_ptr.build_frames_annotated(frame_ids, fragment, mz_noise_precursor, mz_noise_uniform, precursor_noise_ppm, mz_noise_fragment, fragment_noise_ppm, right_drag, num_threads)
        return [TimsFrameAnnotated.from_py_ptr(frame) for frame in frames]

    def get_collision_energy(self, frame_id: int, scan_id: int) -> float:
        return self.__py_ptr.get_collision_energy(frame_id, scan_id)

    def get_collision_energies(self, frame_ids: List[int], scan_ids: List[int]) -> List[float]:
        return self.__py_ptr.get_collision_energies(frame_ids, scan_ids)

    def __repr__(self):
        return f"TimsTofSyntheticFrameBuilderDIA(path={self.path})"

    @classmethod
    def from_py_ptr(cls, py_ptr: ims.PyTimsTofSyntheticsFrameBuilderDIA) -> 'TimsTofSyntheticFrameBuilderDIA':
        """Create a TimsTofSyntheticFrameBuilderDIA from a PyTimsTofSyntheticsFrameBuilderDIA.

        Args:
            py_ptr (ims.PyTimsTofSyntheticsFrameBuilderDIA): PyTimsTofSyntheticsFrameBuilderDIA.

        Returns:
            TimsTofSyntheticFrameBuilderDIA: TimsTofSyntheticFrameBuilderDIA.
        """
        builder = cls.__new__(cls)
        builder.__py_ptr = py_ptr
        return builder

    def get_py_ptr(self) -> ims.PyTimsTofSyntheticsFrameBuilderDIA:
        return self.__py_ptr

    def get_ion_transmission_matrix(self, peptide_id: int, charge: int, include_precursor_frames: bool = False) -> NDArray:
        return np.array(self.__py_ptr.get_ion_transmission_matrix(peptide_id, charge, include_precursor_frames))

    def count_number_transmissions(self, peptide_id: int, charge: int) -> (int, int):
        return self.__py_ptr.count_number_transmissions(peptide_id, charge)

    def count_number_transmissions_parallel(self, peptide_ids: List[int], charges: List[int], num_threads: int = 4) -> List[Tuple[int, int]]:
        return self.__py_ptr.count_number_transmissions_parallel(peptide_ids, charges, num_threads)


class TimsTofSyntheticPrecursorFrameBuilder(RustWrapperObject):
    def __init__(self, db_path: str):
        self.__py_ptr = ims.PyTimsTofSyntheticsPrecursorFrameBuilder(db_path)

    def build_precursor_frame(self, frame_id: int, mz_noise_precursor: bool = False, mz_noise_uniform: bool = False, precursor_noise_ppm: float = 5., right_drag: bool = True) -> TimsFrame:
        frame = self.__py_ptr.build_precursor_frame(frame_id, mz_noise_precursor, mz_noise_uniform, precursor_noise_ppm, right_drag)
        return TimsFrame.from_py_ptr(frame)

    def build_precursor_frames(self, frame_ids: List[int], mz_noise_precursor: bool = False, mz_noise_uniform: bool = False, precursor_noise_ppm: float = 5.,
                               right_drag: bool = True,
                               num_threads: int = 4):
        frames = self.__py_ptr.build_precursor_frames(frame_ids, mz_noise_precursor, mz_noise_uniform,
                                                      precursor_noise_ppm,
                                                      right_drag,
                                                      num_threads)
        return [TimsFrame.from_py_ptr(frame) for frame in frames]

    def build_precursor_frame_annotated(self, frame_id: int, mz_noise_precursor: bool = False, mz_noise_uniform: bool = False, precursor_noise_ppm: float = 5., right_drag: bool = True) -> TimsFrameAnnotated:
        frame = self.__py_ptr.build_precursor_frame_annotated(frame_id, mz_noise_precursor, mz_noise_uniform, precursor_noise_ppm, right_drag)
        return TimsFrameAnnotated.from_py_ptr(frame)

    def build_precursor_frames_annotated(self, frame_ids: List[int], mz_noise_precursor: bool = False, mz_noise_uniform: bool = False, precursor_noise_ppm: float = 5., right_drag: bool = True, num_threads: int = 4) -> List[TimsFrameAnnotated]:
        frames = self.__py_ptr.build_precursor_frames_annotated(frame_ids, mz_noise_precursor, mz_noise_uniform, precursor_noise_ppm, right_drag, num_threads)
        return [TimsFrameAnnotated.from_py_ptr(frame) for frame in frames]

    def __repr__(self):
        return f"TimsTofSyntheticPrecursorFrameBuilder()"

    @classmethod
    def from_py_ptr(cls, py_ptr: ims.PyTimsTofSyntheticsPrecursorFrameBuilder) -> 'TimsTofSyntheticPrecursorFrameBuilder':
        builder = cls.__new__(cls)
        builder.__py_ptr = py_ptr
        return builder

    def get_py_ptr(self) -> ims.PyTimsTofSyntheticsPrecursorFrameBuilder:
        return self.__py_ptr


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

    def append_table(self, table_name: str, table: pd.DataFrame):
        # Append a table to an existing table in the database
        table.to_sql(table_name, self.conn, if_exists='append', index=False)

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

    def list_tables(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            return [table[0] for table in tables]

    def list_columns(self, table_name):
        if table_name not in self.list_tables():
            raise ValueError(f"Table '{table_name}' does not exist in the database.")

        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            return [column[1] for column in columns]

    def __repr__(self):
        return f"SyntheticExperimentDataHandle(database_path={self.database_path})"


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
