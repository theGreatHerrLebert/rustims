import numpy as np
import pandas as pd
from numpy.typing import NDArray
from abc import abstractmethod, ABC

from imspy.timstof.data import AcquisitionMode
from imspy.simulation.utility import calculate_number_frames, calculate_mobility_spacing


class TimsTofAcquisitionBuilder:
    def __init__(self, gradient_length: float, rt_cycle_length: float, im_lower: float, im_upper: float,
                 num_scans: int):
        """ Base class for building TimsTOF experiments
        Parameters
        ----------
        gradient_length : float
            Length of the gradient in seconds
        rt_cycle_length : float
            Length of the RT cycle in seconds
        im_lower : float
            Lower bound of the ion mobility range (IM first scan)
        im_upper : float
            Upper bound of the ion mobility range (IM last scan)
        num_scans : int
            Number of scans (IM) to be acquired
        """
        self.gradient_length = gradient_length
        self.rt_cycle_length = rt_cycle_length
        self.im_lower = im_lower
        self.im_upper = im_upper
        self.num_scans = num_scans
        self.im_cycle_length = calculate_mobility_spacing(im_lower, im_upper, num_scans)
        self.num_frames = calculate_number_frames(gradient_length, rt_cycle_length)
        self.num_scans = num_scans

    def generate_frame_table(self, verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print('generating frame layout...')
        frames = []
        for i in range(self.num_frames):
            frame_id = i + 1
            time = frame_id * self.rt_cycle_length
            frames.append({'frame_id': frame_id, 'time': time})

        return pd.DataFrame(frames)

    def generate_scan_table(self, verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print('Generating scan layout...')
        scans = []

        # Generating scan_ids in ascending order
        scan_ids = list(range(self.num_scans))
        for index, value in enumerate(scan_ids):
            scan_id = value
            # Start with the highest mobility value at the lowest scan index and decrease
            mobility = self.im_upper - index * self.im_cycle_length
            scans.append({'scan': scan_id, 'mobility': mobility})

        return pd.DataFrame(scans)

    def __repr__(self):
        return (f"TimsTofAcquisitionBuilder(path={self.path}, gradient_length={np.round(self.gradient_length / 60)} "
                f"min, mobility_range: {self.im_lower}-{self.im_upper}, "
                f"num_frames: {self.num_frames}, num_scans: {self.num_scans})")

    @abstractmethod
    def calculate_frame_types(self, *args) -> NDArray:
        pass


class TimsTofAcquisitionBuilderDDA(TimsTofAcquisitionBuilder, ABC):
    def __init__(self,
                 verbose: bool = True,
                 precursor_every: int = 7,
                 gradient_length=120 * 60,
                 rt_cycle_length=0.109,
                 im_lower=0.6,
                 im_upper=1.6,
                 num_scans=917,
                 mz_lower: float = 150,
                 mz_upper: float = 1700):
        super().__init__(gradient_length, rt_cycle_length, im_lower, im_upper, num_scans)
        self.scan_table = None
        self.frame_table = None
        self.precursor_every = precursor_every
        self.acquisition_mode = AcquisitionMode('DDA')
        self.verbose = verbose
        self.mz_lower = mz_lower
        self.mz_upper = mz_upper

        self._setup(verbose=verbose)

    def calculate_frame_types(self, table: pd.DataFrame, precursor_every: int = 7, verbose: bool = True) -> NDArray:
        if verbose:
            print(f'calculating frame types, precursor frame will be taken every {precursor_every} rt cycles.')
        return np.array([0 if (x - 1) % (precursor_every + 1) == 0 else 8 for x in table.frame_id])

    def _setup(self, verbose: bool = True):
        self.frame_table = self.generate_frame_table(verbose=verbose)
        self.scan_table = self.generate_scan_table(verbose=verbose)
        self.frame_table['ms_type'] = self.calculate_frame_types(table=self.frame_table,
                                                                 precursor_every=self.precursor_every, verbose=verbose)


class TimsTofAcquisitionBuilderDIA(TimsTofAcquisitionBuilder, ABC):
    def __init__(self,
                 verbose: bool = True,
                 precursor_every: int = 17,
                 gradient_length=50 * 60,
                 rt_cycle_length=0.1054,
                 im_lower=0.6,
                 im_upper=1.5,
                 num_scans=927,
                 mz_lower: float = 100,
                 mz_upper: float = 1700):
        super().__init__(gradient_length, rt_cycle_length, im_lower, im_upper, num_scans)
        self.scan_table = None
        self.frame_table = None
        self.acquisition_mode = AcquisitionMode('DIA')
        self.verbose = verbose
        self.mz_lower = mz_lower
        self.mz_upper = mz_upper
        self.precursor_every = precursor_every

        self._setup(verbose=verbose)

    def calculate_frame_types(self, table: pd.DataFrame, verbose: bool = True) -> NDArray:
        if verbose:
            print(f'calculating frame types, precursor frame will be taken every {self.precursor_every} rt cycles.')
        return np.array([0 if (x - 1) % (self.precursor_every + 1) == 0 else 9 for x in table.frame_id])

    def _setup(self, verbose: bool = True):
        self.frame_table = self.generate_frame_table(verbose=verbose)
        self.scan_table = self.generate_scan_table(verbose=verbose)
        self.frame_table['ms_type'] = self.calculate_frame_types(table=self.frame_table, verbose=verbose)


class TimsTofAcquisitionBuilderMIDIA(TimsTofAcquisitionBuilder, ABC):
    def __init__(self,
                 verbose: bool = True,
                 precursor_every: int = 20,
                 gradient_length=50 * 60,
                 rt_cycle_length=0.056,
                 im_lower=0.6,
                 im_upper=1.5,
                 num_scans=451,
                 mz_lower: float = 150,
                 mz_upper: float = 1200):
        super().__init__(gradient_length, rt_cycle_length, im_lower, im_upper, num_scans)
        self.scan_table = None
        self.frame_table = None
        self.acquisition_mode = AcquisitionMode('MIDIA')
        self.verbose = verbose
        self.mz_lower = mz_lower
        self.mz_upper = mz_upper
        self.precursor_every = precursor_every

        self._setup(verbose=verbose)

    def calculate_frame_types(self, table: pd.DataFrame, verbose: bool = True) -> NDArray:
        if verbose:
            print(f'calculating frame types, precursor frame will be taken every {self.precursor_every} rt cycles.')
        return np.array([0 if (x - 1) % (self.precursor_every + 1) == 0 else 9 for x in table.frame_id])

    def _setup(self, verbose: bool = True):
        self.frame_table = self.generate_frame_table(verbose=verbose)
        self.scan_table = self.generate_scan_table(verbose=verbose)
        self.frame_table['ms_type'] = self.calculate_frame_types(table=self.frame_table, verbose=verbose)
