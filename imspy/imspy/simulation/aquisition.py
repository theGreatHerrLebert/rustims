import os
import sqlite3

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from abc import abstractmethod, ABC
from imspy.timstof.data import AcquisitionMode
from imspy.simulation.utility import calculate_number_frames, calculate_mobility_spacing


class TimsTofAcquisitionBuilder:
    def __init__(self,
                 path: str,
                 rt_cycle_length: float,
                 gradient_length: float,
                 im_lower: float,
                 im_upper: float,
                 num_scans: int):
        """ Base class for building TimsTOF experiments
        Parameters
        ----------
        path : str
            Path to the experiment folder
        rt_cycle_length : float
            Length of the RT cycle in seconds
        gradient_length : float
            Length of the gradient in seconds
        im_lower : float
            Lower bound of the ion mobility range
        im_upper : float
            Upper bound of the ion mobility range
        num_scans : int
            Number of scans to be acquired
        """

        self.path = path
        self.raw_data_path = f"{self.path}/raw_data/"
        self.database_path = f"{self.path}/experiment_database.db"

        self.gradient_length = gradient_length
        self.im_lower = im_lower
        self.im_upper = im_upper
        self.num_scans = num_scans
        self.im_cycle_length = calculate_mobility_spacing(im_lower, im_upper, num_scans)
        self.num_frames = calculate_number_frames(gradient_length, rt_cycle_length)
        self.num_scans = num_scans

        frames = []
        for i in range(self.num_frames):
            frame_id = i + 1
            time = frame_id * rt_cycle_length
            frames.append({'frame_id': frame_id, 'time': time})

        scans = []
        for i in range(self.num_scans):
            scan_id = i
            time = im_lower + i * self.im_cycle_length
            scans.append({'scan': scan_id, 'mobility': time})

        self.frames = pd.DataFrame(frames)
        self.scans = pd.DataFrame(scans)

    @abstractmethod
    def calculate_frame_types(self, **kwargs) -> NDArray:
        pass


class TimsTofDDAAcquisitionBuilder(TimsTofAcquisitionBuilder, ABC):
    def __init__(self,
                 path: str,
                 verbose: bool = True,
                 precursor_every: int = 7,
                 rt_cycle_length=0.109,
                 gradient_length=120 * 60,
                 im_lower=0.6,
                 im_upper=1.6,
                 num_scans=917):
        super().__init__(path, rt_cycle_length, gradient_length, im_lower, im_upper, num_scans)
        self.acquisition_mode = AcquisitionMode('DDA')

        # calculate frame types
        if verbose:
            print("Generating TimsTOF DDA acquisition with parameters:")
            print(self)
            print("Calculating frame types...")
        self.frames['ms_type'] = self._calculate_frame_types(precursor_every)

        # create folder for output files if it does not exist
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # create folder for raw data if it does not exist
        if not os.path.exists(self.raw_data_path):
            os.makedirs(self.raw_data_path)

        # check if database exists, if not create it
        if not os.path.exists(self.database_path):
            if verbose:
                print("Creating database...")
            conn = sqlite3.connect(self.database_path)
            conn.close()

        # write tables to database
        if verbose:
            print("Writing frame table to database...")
        self._frame_table_to_sqlite_db()

        if verbose:
            print("Writing scan table to database...")
        self._scan_table_to_sqlite_db()

    def _frame_table_to_sqlite_db(self):
        self.frames.to_sql('frames', sqlite3.connect(self.database_path), if_exists='replace', index=False)

    def _scan_table_to_sqlite_db(self):
        self.scans.to_sql('scans', sqlite3.connect(self.database_path), if_exists='replace', index=False)

    def _calculate_frame_types(self, precursor_every: int = 7) -> NDArray:
        return np.array([0 if (x - 1) % precursor_every == 0 else 8 for x in self.frames.frame_id])

    def __repr__(self):
        return (f"TimsTofDDAAcquisitionBuilder(path={self.path}, gradient_length={np.round(self.gradient_length / 60)} "
                f"min, mobility_range: {self.im_lower}-{self.im_upper}, "
                f"num_frames: {self.num_frames}, num_scans: {self.num_scans})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_path', type=str, help='Path to the dataset')
    parser.add_argument('--gradient_length', type=float, default=120 * 60, help='Length of the gradient in seconds')
    parser.add_argument('--im_lower', type=float, default=0.6, help='Lower bound of the ion mobility range')
    parser.add_argument('--im_upper', type=float, default=1.6, help='Upper bound of the ion mobility range')
    parser.add_argument('--num_scans', type=int, default=917, help='Number of scans to be acquired')
    parser.add_argument('--precursor_every', type=int, default=7, help='Number of frames between precursors')
    args = parser.parse_args()

    builder = TimsTofDDAAcquisitionBuilder('/home/administrator/Documents/promotion/rust/notebook/TEST',
                                           gradient_length=args.gradient_length,
                                           im_lower=args.im_lower,
                                           im_upper=args.im_upper,
                                           num_scans=args.num_scans,
                                           precursor_every=args.precursor_every)
