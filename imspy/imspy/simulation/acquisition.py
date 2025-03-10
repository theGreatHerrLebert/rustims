from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from abc import abstractmethod, ABC

from imspy.simulation.experiment import SyntheticExperimentDataHandle
from imspy.timstof import TimsDatasetDIA
from imspy.timstof.data import AcquisitionMode, TimsDataset
from imspy.simulation.utility import calculate_number_frames, get_ms_ms_window_layout_resource_path
from imspy.simulation.tdf import TDFWriter


class TimsTofAcquisitionBuilder:
    def __init__(
            self,
            path: str,
            reference_ds: TimsDataset,
            gradient_length: float,
            rt_cycle_length: float,
            exp_name: str = "RAW.d",
    ):
        """ Base class for building TimsTOF experiments
        Parameters
        ----------
        path : str
            Path to the experiment directory
        gradient_length : float
            Length of the gradient in seconds
        rt_cycle_length : float
            Length of the RT cycle in seconds
        """

        self.path = path
        self.gradient_length = gradient_length
        self.rt_cycle_length = rt_cycle_length
        self.num_frames = calculate_number_frames(gradient_length, rt_cycle_length)
        # Create the TDFWriter, used to deal with bruker binary format writing and metadata for libtimsdata.so
        self.tdf_writer = TDFWriter(
            path=self.path,
            helper_handle=reference_ds,
            exp_name=exp_name,
        )
        # Create the SyntheticExperimentDataHandle, which is used to deal with the sqlite database of synthetic data
        self.synthetics_handle = SyntheticExperimentDataHandle(database_path=self.path)
        self.frame_table = None
        self.scan_table = None

    def generate_frame_table(self, verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print('Generating frame layout.')
        frames = []
        for i in range(self.num_frames):
            frame_id = i + 1
            time = frame_id * self.rt_cycle_length
            frames.append({'frame_id': frame_id, 'time': time})

        return pd.DataFrame(frames)

    def generate_scan_table(self, verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print('Generating scan layout.')

        scans = np.arange(self.tdf_writer.helper_handle.num_scans)[::-1]
        mobilities = self.tdf_writer.scan_to_inv_mobility(1, scans)

        return pd.DataFrame({'scan': scans, 'mobility': mobilities})

    def __repr__(self):
        return (f"TimsTofAcquisitionBuilder(path={self.path}, gradient_length={np.round(self.gradient_length / 60)} "
                f"min, mobility_range: {self.tdf_writer.helper_handle.im_lower}-{self.tdf_writer.helper_handle.im_upper}, "
                f"num_frames: {self.num_frames}, num_scans: {self.tdf_writer.helper_handle.num_scans})")

    @abstractmethod
    def calculate_frame_types(self, *args) -> NDArray:
        pass

    @classmethod
    def from_existing(cls, path: str, reference_ds: TimsDataset) -> "TimsTofAcquisitionBuilder":
        """ Create an instance from existing data without calling __init__.
        """
        # Create a new instance without calling __init__
        instance = cls.__new__(cls)
        instance.synthetics_handle = SyntheticExperimentDataHandle(database_path=path)

        # Manually set fields from data dictionary
        instance.path = path
        instance.frame_table = instance.synthetics_handle.get_table('frames')
        instance.scan_table = instance.synthetics_handle.get_table('scans')
        instance.gradient_length = np.round(instance.frame_table.time.max())
        instance.rt_cycle_length = np.mean(np.diff(instance.frame_table.time))
        instance.num_frames = instance.frame_table.shape[0]

        # extract experiment name from the path
        exp_name = Path(path).name

        # Set up the TDFWriter and SyntheticExperimentDataHandle
        instance.tdf_writer = TDFWriter(
            path=instance.path,
            helper_handle=reference_ds,
            exp_name=exp_name,
        )

        return instance


class TimsTofAcquisitionBuilderDDA(TimsTofAcquisitionBuilder, ABC):
    def __init__(self,
                 path: str,
                 reference_ds: TimsDataset,
                 verbose: bool = True,
                 gradient_length=120 * 60,
                 rt_cycle_length=0.109,
                 exp_name: str = "RAW.d",
                 ):
        super().__init__(path,
                         gradient_length,
                         rt_cycle_length,
                         reference_ds.im_lower,
                         reference_ds.im_upper,
                         reference_ds.mz_lower,
                         reference_ds.mz_upper,
                         reference_ds.num_scans,
                         exp_name=exp_name)

        self.scan_table = None
        self.frame_table = None
        self.pasef_meta = None
        self.selected_precursors = None
        self.acquisition_mode = AcquisitionMode('DDA')
        self.verbose = verbose
        self._setup(verbose=verbose)

    def _setup(self, verbose: bool = True):
        self.frame_table = self.generate_frame_table(verbose=verbose)
        self.scan_table = self.generate_scan_table(verbose=verbose)
        self.synthetics_handle.create_table(
            table_name='frames',
            table=self.frame_table
        )
        self.synthetics_handle.create_table(
            table_name='scans',
            table=self.scan_table
        )

    def calculate_frame_types(self, frame_types: NDArray):
        assert len(frame_types) == self.frame_table.shape[0], "frame_types must have the same length as the frame table."
        assert set(frame_types).issubset({0, 8}), f"frame_types must be a list of 0s and 8s, got {set(frame_types)}."
        self.frame_table['ms_type'] = frame_types
        self.synthetics_handle.create_table(
            table_name='frames',
            table=self.frame_table
        )

class TimsTofAcquisitionBuilderDIA(TimsTofAcquisitionBuilder, ABC):
    def __init__(self,
                 path: str,
                 reference_ds: TimsDatasetDIA,
                 window_group_file: str,
                 acquisition_name: str = "dia",
                 exp_name: str = "RAW",
                 verbose: bool = True,
                 precursor_every: int = 17,
                 gradient_length=50 * 60,
                 rt_cycle_length=0.1054,
                 use_reference_ds_layout: bool = True,
                 round_collision_energy: bool = True,
                 collision_energy_decimals: int = 1
                 ):

        super().__init__(path, reference_ds, gradient_length, rt_cycle_length,
                         exp_name=exp_name)
        # TODO: check this, could be missing replacement of reference layout of windows
        if use_reference_ds_layout:
            rt_cycle_length = np.mean(np.diff(reference_ds.meta_data.Time))
            if verbose:
                print('Using reference dataset cycle length:', np.round(rt_cycle_length, 4))
            self.rt_cycle_length = rt_cycle_length

        self.acquisition_name = acquisition_name
        self.scan_table = None
        self.frame_table = None
        self.frames_to_window_groups = None
        self.dia_ms_ms_windows = pd.read_csv(window_group_file)
        self.use_reference_ds_layout = use_reference_ds_layout
        self.reference = reference_ds
        self.round_collision_energy = round_collision_energy
        self.collision_energy_decimals = collision_energy_decimals

        # TODO: check if the number of scans in the window group file matches the number of scans in the experiment

        self.acquisition_mode = AcquisitionMode('DIA')
        self.verbose = verbose
        self.precursor_every = precursor_every

        self._setup(verbose=verbose)

    def calculate_frame_types(self, verbose: bool = True) -> NDArray:
        if verbose:
            print(f'Calculating frame types, precursor frame will be taken every {self.precursor_every} rt cycles.')
        return np.array([0 if (x - 1) % self.precursor_every == 0 else 9 for x in self.frame_table.frame_id])

    def generate_frame_to_window_group_table(self, verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print(f'generating frame to window group table, precursors every {self.precursor_every} frames.')

        table_list = []
        for index, row in self.frame_table.iterrows():
            frame_id, ms_type = row.frame_id, row.ms_type
            wg = index % self.precursor_every
            if ms_type > 0:
                table_list.append({'frame': int(frame_id), 'window_group': wg})

        return pd.DataFrame(table_list)

    def _setup(self, verbose: bool = True):
        self.frame_table = self.generate_frame_table(verbose=verbose)
        self.scan_table = self.generate_scan_table(verbose=verbose)

        if self.use_reference_ds_layout:
            self.precursor_every = int(np.diff(self.reference.precursor_frames)[0])
            self.dia_ms_ms_windows = self.reference.dia_ms_ms_windows.rename(
                columns={
                    'WindowGroup': 'window_group',
                    'ScanNumBegin': 'scan_start',
                    'ScanNumEnd': 'scan_end',
                    'IsolationMz': 'isolation_mz',
                    'IsolationWidth': 'isolation_width',
                    'CollisionEnergy': 'collision_energy',
                }
            )

        self.frame_table['ms_type'] = self.calculate_frame_types(verbose=verbose)
        self.frames_to_window_groups = self.generate_frame_to_window_group_table(verbose=verbose)

        if self.round_collision_energy:
            self.dia_ms_ms_windows['collision_energy'] = np.round(self.dia_ms_ms_windows['collision_energy'].values,
                                                                  decimals=self.collision_energy_decimals)

        self.synthetics_handle.create_table(
            table_name='frames',
            table=self.frame_table
        )
        self.synthetics_handle.create_table(
            table_name='scans',
            table=self.scan_table
        )
        self.synthetics_handle.create_table(
            table_name='dia_ms_ms_info',
            table=self.frames_to_window_groups
        )
        self.synthetics_handle.create_table(
            table_name='dia_ms_ms_windows',
            table=self.dia_ms_ms_windows
        )

    @staticmethod
    def from_config(
            path: str,
            reference_ds: TimsDatasetDIA,
            exp_name: str,
            config: Dict[str, any],
            verbose: bool = True,
            use_reference_layout: bool = True,
            round_collision_energy: bool = True,
            collision_energy_decimals: int = 1
    ) -> 'TimsTofAcquisitionBuilderDIA':

        acquisition_name = config['name'].lower().replace('pasef', '')
        window_group_file = get_ms_ms_window_layout_resource_path(acquisition_name)

        return TimsTofAcquisitionBuilderDIA(
            path=str(Path(path) / exp_name),
            reference_ds=reference_ds,
            window_group_file=str(window_group_file),
            exp_name=exp_name + ".d",
            verbose=verbose,
            acquisition_name=acquisition_name,
            precursor_every=config['precursor_every'],
            gradient_length=config['gradient_length'],
            rt_cycle_length=config['rt_cycle_length'],
            use_reference_ds_layout=use_reference_layout,
            round_collision_energy=round_collision_energy,
            collision_energy_decimals=collision_energy_decimals
        )

    @classmethod
    def from_existing(cls,
                      path: str,
                      reference_ds: TimsDatasetDIA,
                      use_reference_ds_layout: bool = True,
                      verbose: bool = True) -> "TimsTofAcquisitionBuilderDIA":
        """ Create an instance from existing data for DIA without calling __init__."""

        # Create an instance without calling __init__
        instance = cls.__new__(cls)

        # Initialize shared attributes as done in the parent’s from_existing
        instance.synthetics_handle = SyntheticExperimentDataHandle(database_path=path)
        instance.path = path
        instance.frame_table = instance.synthetics_handle.get_table('frames')
        instance.scan_table = instance.synthetics_handle.get_table('scans')
        instance.gradient_length = np.round(instance.frame_table.time.max())
        instance.rt_cycle_length = np.mean(np.diff(instance.frame_table.time))
        instance.num_frames = instance.frame_table.shape[0]
        exp_name = Path(path).name
        instance.tdf_writer = TDFWriter(
            path=instance.path,
            helper_handle=reference_ds,
            exp_name=exp_name,
        )

        # Ensure use_reference_ds_layout is supported
        assert use_reference_ds_layout, "Only use_reference_ds_layout=True is supported for existing DIA acquisitions"

        # Set DIA-specific fields
        instance.reference = reference_ds
        instance.verbose = verbose
        instance.window_group_file = None
        instance.precursor_every = int(np.diff(instance.reference.precursor_frames)[0])
        instance.round_collision_energy = True
        instance.collision_energy_decimals = 1
        instance.use_reference_ds_layout = use_reference_ds_layout

        # Load additional tables for DIA-specific data
        instance.frames_to_window_groups = instance.synthetics_handle.get_table('dia_ms_ms_info')
        instance.dia_ms_ms_windows = instance.synthetics_handle.get_table('dia_ms_ms_windows')

        return instance

    def __repr__(self):
        return (f"TimsTofAcquisitionBuilderDIA(name={self.acquisition_name}, path={self.path}, "
                f"gradient_length={np.round(self.gradient_length / 60)} min, mobility_range: "
                f"{self.tdf_writer.helper_handle.im_lower}-{self.tdf_writer.helper_handle.im_upper}, "
                f"num_frames: {self.num_frames}, num_scans: {self.tdf_writer.helper_handle.num_scans})")
