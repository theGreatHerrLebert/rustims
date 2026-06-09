Reading additional input from stdin...
OpenAI Codex v0.136.0
--------
workdir: /scratch/timsim-demo/SUBMISSION/rustims
model: gpt-5.5
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: medium
reasoning summaries: none
session id: 019eab6b-8a3f-7dc2-8f2a-1ddea8c9d30f
--------
user
Review the NEWLY CHANGED code in this Python mass-spec simulator (rest is context): in TimsTofAcquisitionBuilderDIA — (a) the __init__ num_frames recompute under use_reference_ds_layout, (b) the new _layout_from_scheme() helper, (c) the rewired _setup() that uses the scheme layout (with legacy fallback), and (d) the parity test.

Trusted context: imspy_connector.py_acquisition.PyAcquisitionScheme.from_bruker_d(path) wraps a Rust extractor; to_bruker_windows() returns a dict of numpy arrays {window_group, scan_start, scan_end, isolation_mz, isolation_width, collision_energy}; to_bruker_info(num_frames) returns {frame, window_group}; cycle_length() = frames per cycle (precursor_every). The Rust side is golden-tested to reproduce the reference .d's DiaFrameMsMsWindows/DiaFrameMsMsInfo exactly. The LEGACY path reads reference.dia_ms_ms_windows (renamed) and generate_frame_to_window_group_table uses wg = global_frame_index % precursor_every (emitting ms2 frames only). calculate_frame_types sets ms_type=0 when (frame_id-1)%precursor_every==0 else 9. CE is rounded to collision_energy_decimals after.

Focus, ranked: 1) is the migrated _setup BEHAVIOR-EQUIVALENT to legacy for a real reference .d — same dia_ms_ms_windows (column names/dtypes/CE rounding) and same dia_ms_ms_info (frame ids, window_group values, row count, ordering), given scheme group ids vs the legacy POSITION-based wg=index%precursor_every (do they agree only when reference WindowGroups are 1..N contiguous in acquisition order, and is that now relied upon)? 2) the num_frames recompute fix — correct and sufficient? any other stale-state from rt_cycle_length changing (scan_table, frame_table already generated before _setup)? 3) dtype regressions from numpy uint32 columns vs the legacy int columns when written to SQLite (to_sql) and read back by the simulator — could downstream code break on uint32 / column order? 4) the fallback logic (connector without py_acquisition) — correctness and silent-degradation risk. 5) the parity test — does it actually prove backward-compat, what's NOT covered (CE rounding, dtypes, the full builder, non-1..N groups). 6) any bug. Concrete, ranked, ~550 words.

<stdin>
===== FILE: acquisition.py (focus: TimsTofAcquisitionBuilderDIA changes) =====
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from abc import abstractmethod, ABC

from imspy_simulation.experiment import SyntheticExperimentDataHandle
from imspy_core.timstof import TimsDatasetDIA, TimsDatasetDDA
from imspy_core.timstof.data import AcquisitionMode, TimsDataset
from imspy_simulation.utility import calculate_number_frames, get_ms_ms_window_layout_resource_path
from imspy_simulation.tdf import TDFWriter


class TimsTofAcquisitionBuilder:
    def __init__(
            self,
            path: str,
            reference_ds: TimsDataset,
            gradient_length: float,
            rt_cycle_length: float,
            exp_name: str = "RAW.d",
            verbose: bool = False,
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
        exp_name : str
            Name of the experiment
        verbose : bool
            Print verbose output
        """

        self.path = path
        self.gradient_length = gradient_length
        self.rt_cycle_length = rt_cycle_length
        self.num_frames = calculate_number_frames(gradient_length, rt_cycle_length)
        self.verbose = verbose
        # Create the TDFWriter, used to deal with bruker binary format writing and metadata for libtimsdata.so
        self.tdf_writer = TDFWriter(
            path=self.path,
            helper_handle=reference_ds,
            exp_name=exp_name,
            verbose=verbose,
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
                 reference_ds: TimsDatasetDDA,
                 verbose: bool = False,
                 gradient_length=60 * 60,
                 rt_cycle_length=0.109,
                 exp_name: str = "T001.d",
                 ):
        super().__init__(path, reference_ds, gradient_length, rt_cycle_length, exp_name=exp_name, verbose=verbose)

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
        # assert set(frame_types).issubset({0, 8}), f"frame_types must be a list of 0s and 8s, got {set(frame_types)}."
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
            # Recompute the frame count: the base __init__ computed it from the
            # passed rt_cycle_length before this reference-derived value replaced it.
            self.num_frames = calculate_number_frames(self.gradient_length, rt_cycle_length)

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

    def _layout_from_scheme(self, verbose: bool = True):
        """Derive the DIA window + frame→group tables from the reference `.d` via
        the vendor-neutral ``AcquisitionScheme`` (imspy_connector.py_acquisition).

        Returns ``(dia_ms_ms_windows, frames_to_window_groups, precursor_every)``
        or ``None`` if the connector lacks the scheme module (legacy fallback).
        """
        try:
            import imspy_connector
            acq = imspy_connector.py_acquisition
        except (ImportError, AttributeError):
            if verbose:
                print('py_acquisition unavailable; using legacy reference layout')
            return None

        scheme = acq.PyAcquisitionScheme.from_bruker_d(self.reference.data_path)
        precursor_every = scheme.cycle_length()
        windows = pd.DataFrame(scheme.to_bruker_windows())
        info = pd.DataFrame(scheme.to_bruker_info(int(self.num_frames)))
        if verbose:
            print(f'Using AcquisitionScheme layout: {len(windows)} windows, '
                  f'{scheme.n_ms2_frames()} groups, precursor_every={precursor_every}')
        return windows, info, precursor_every

    def _setup(self, verbose: bool = True):
        self.frame_table = self.generate_frame_table(verbose=verbose)
        self.scan_table = self.generate_scan_table(verbose=verbose)

        scheme_layout = self._layout_from_scheme(verbose=verbose) if self.use_reference_ds_layout else None

        if scheme_layout is not None:
            # Vendor-neutral path: windows + frame→group come from the scheme.
            self.dia_ms_ms_windows, frames_to_window_groups, self.precursor_every = scheme_layout
            self.frame_table['ms_type'] = self.calculate_frame_types(verbose=verbose)
            self.frames_to_window_groups = frames_to_window_groups
        else:
            # Legacy path: copy the reference .d's window table directly.
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

===== FILE: tests/test_acquisition_scheme_parity.py =====
"""Golden parity: the vendor-neutral AcquisitionScheme (via imspy_connector)
must reproduce the Bruker DIA layout the legacy `use_reference_ds_layout` path
copies from the reference `.d` — both the `DiaFrameMsMsWindows` window table and
the per-frame `DiaFrameMsMsInfo` (frame→group) table.

Gated on TIMSIM_BRUKER_DIA_D pointing at a real DIA-PASEF `.d`.
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import imspy_connector

acq = imspy_connector.py_acquisition
DIA_D = os.environ.get("TIMSIM_BRUKER_DIA_D")
pytestmark = pytest.mark.skipif(
    not DIA_D, reason="set TIMSIM_BRUKER_DIA_D to a real DIA-PASEF .d folder"
)


def _norm(df, cols):
    return df[cols].astype(float).round(6).sort_values(cols).reset_index(drop=True)


def test_scheme_windows_match_reference():
    scheme = acq.PyAcquisitionScheme.from_bruker_d(DIA_D)
    windows = pd.DataFrame(scheme.to_bruker_windows())

    con = sqlite3.connect(DIA_D + "/analysis.tdf")
    ref = pd.read_sql(
        "SELECT WindowGroup window_group, ScanNumBegin scan_start, ScanNumEnd scan_end, "
        "IsolationMz isolation_mz, IsolationWidth isolation_width, "
        "CollisionEnergy collision_energy FROM DiaFrameMsMsWindows",
        con,
    )
    cols = list(ref.columns)
    assert len(windows) == len(ref)
    pd.testing.assert_frame_equal(_norm(windows, cols), _norm(ref, cols))


def test_scheme_info_matches_reference_and_legacy_formula():
    scheme = acq.PyAcquisitionScheme.from_bruker_d(DIA_D)
    con = sqlite3.connect(DIA_D + "/analysis.tdf")
    num_frames = int(con.execute("SELECT MAX(Id) FROM Frames").fetchone()[0])

    info = pd.DataFrame(scheme.to_bruker_info(num_frames))

    # (a) matches the .d's own DiaFrameMsMsInfo exactly
    ref = pd.read_sql("SELECT Frame frame, WindowGroup window_group FROM DiaFrameMsMsInfo", con)
    pd.testing.assert_frame_equal(
        _norm(info, ["frame", "window_group"]), _norm(ref, ["frame", "window_group"])
    )

    # (b) matches the legacy position formula (index % precursor_every, ms2 only)
    pe = scheme.cycle_length()
    legacy = pd.DataFrame(
        [
            {"frame": f, "window_group": (f - 1) % pe}
            for f in range(1, num_frames + 1)
            if (f - 1) % pe != 0
        ]
    )
    pd.testing.assert_frame_equal(
        _norm(info, ["frame", "window_group"]), _norm(legacy, ["frame", "window_group"])
    )
</stdin>
codex
I’ll inspect the actual diff and nearby persistence/read paths so the review distinguishes changed behavior from context. I’ll then rank concrete findings with file/line references and assess the test’s real coverage.
exec
/bin/bash -lc 'git diff -- acquisition.py tests/test_acquisition_scheme_parity.py' in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
exec
/bin/bash -lc "nl -ba acquisition.py | sed -n '120,360p'; nl -ba tests/test_acquisition_scheme_parity.py" in /scratch/timsim-demo/SUBMISSION/rustims
 exited 1 in 0ms:
nl: acquisition.py: No such file or directory
nl: tests/test_acquisition_scheme_parity.py: No such file or directory

exec
/bin/bash -lc "rg -n \"dia_ms_ms_info|dia_ms_ms_windows|create_table\\(|get_table\\(\" -S ." in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
./MIGRATION.codex-review.md:16:Trusted context: imspy_connector.py_acquisition.PyAcquisitionScheme.from_bruker_d(path) wraps a Rust extractor; to_bruker_windows() returns a dict of numpy arrays {window_group, scan_start, scan_end, isolation_mz, isolation_width, collision_energy}; to_bruker_info(num_frames) returns {frame, window_group}; cycle_length() = frames per cycle (precursor_every). The Rust side is golden-tested to reproduce the reference .d's DiaFrameMsMsWindows/DiaFrameMsMsInfo exactly. The LEGACY path reads reference.dia_ms_ms_windows (renamed) and generate_frame_to_window_group_table uses wg = global_frame_index % precursor_every (emitting ms2 frames only). calculate_frame_types sets ms_type=0 when (frame_id-1)%precursor_every==0 else 9. CE is rounded to collision_energy_decimals after.
./MIGRATION.codex-review.md:18:Focus, ranked: 1) is the migrated _setup BEHAVIOR-EQUIVALENT to legacy for a real reference .d — same dia_ms_ms_windows (column names/dtypes/CE rounding) and same dia_ms_ms_info (frame ids, window_group values, row count, ordering), given scheme group ids vs the legacy POSITION-based wg=index%precursor_every (do they agree only when reference WindowGroups are 1..N contiguous in acquisition order, and is that now relied upon)? 2) the num_frames recompute fix — correct and sufficient? any other stale-state from rt_cycle_length changing (scan_table, frame_table already generated before _setup)? 3) dtype regressions from numpy uint32 columns vs the legacy int columns when written to SQLite (to_sql) and read back by the simulator — could downstream code break on uint32 / column order? 4) the fallback logic (connector without py_acquisition) — correctness and silent-degradation risk. 5) the parity test — does it actually prove backward-compat, what's NOT covered (CE rounding, dtypes, the full builder, non-1..N groups). 6) any bug. Concrete, ranked, ~550 words.
./MIGRATION.codex-review.md:118:        instance.frame_table = instance.synthetics_handle.get_table('frames')
./MIGRATION.codex-review.md:119:        instance.scan_table = instance.synthetics_handle.get_table('scans')
./MIGRATION.codex-review.md:159:        self.synthetics_handle.create_table(
./MIGRATION.codex-review.md:163:        self.synthetics_handle.create_table(
./MIGRATION.codex-review.md:172:        self.synthetics_handle.create_table(
./MIGRATION.codex-review.md:209:        self.dia_ms_ms_windows = pd.read_csv(window_group_file)
./MIGRATION.codex-review.md:245:        Returns ``(dia_ms_ms_windows, frames_to_window_groups, precursor_every)``
./MIGRATION.codex-review.md:273:            self.dia_ms_ms_windows, frames_to_window_groups, self.precursor_every = scheme_layout
./MIGRATION.codex-review.md:280:                self.dia_ms_ms_windows = self.reference.dia_ms_ms_windows.rename(
./MIGRATION.codex-review.md:294:            self.dia_ms_ms_windows['collision_energy'] = np.round(self.dia_ms_ms_windows['collision_energy'].values,
./MIGRATION.codex-review.md:297:        self.synthetics_handle.create_table(
./MIGRATION.codex-review.md:301:        self.synthetics_handle.create_table(
./MIGRATION.codex-review.md:305:        self.synthetics_handle.create_table(
./MIGRATION.codex-review.md:306:            table_name='dia_ms_ms_info',
./MIGRATION.codex-review.md:309:        self.synthetics_handle.create_table(
./MIGRATION.codex-review.md:310:            table_name='dia_ms_ms_windows',
./MIGRATION.codex-review.md:311:            table=self.dia_ms_ms_windows
./MIGRATION.codex-review.md:358:        instance.frame_table = instance.synthetics_handle.get_table('frames')
./MIGRATION.codex-review.md:359:        instance.scan_table = instance.synthetics_handle.get_table('scans')
./MIGRATION.codex-review.md:383:        instance.frames_to_window_groups = instance.synthetics_handle.get_table('dia_ms_ms_info')
./MIGRATION.codex-review.md:384:        instance.dia_ms_ms_windows = instance.synthetics_handle.get_table('dia_ms_ms_windows')
./EXTRACTORS.codex-review.md:16:Trusted context: rustdf reads Bruker TDF natively — read_dia_ms_ms_windows(folder)->Vec<DiaMsMsWindow{window_group,scan_num_begin,scan_num_end,isolation_mz,isolation_width,collision_energy}> and read_meta_data_sql(folder)->Vec<FrameMeta{id,time:f64,ms_ms_type:i64,...}> (MsMsType==0 = MS1/precursor frame). from_bruker_d groups windows by window_group into DiaMs2Frame{TimsMobility windows} and derives cycle time from precursor-frame spacing; verified to extract 15 frames/36 windows from a real DIA-PASEF .d. from_sciex_wiff maps SWATH windows to single MzOnly frames with CE Unknown (SCIEX rolling CE not in the method) and caller-supplied timing; verified 60 windows on a real ZenoTOF .wiff. validate() (already reviewed) requires exactly one MS1 first then >=1 MS2 frame, finite/positive widths, window edges within mz_range, CE finite>=0 for Value/resolved-Linear, multi-window frames need TimsMobility on every window and no within-frame m/z+mobility overlap, start<=gradient. sciexwiff::read_method opens a .wiff OLE2 (cfb crate) and parses SWATHMethod (20B records from off 40) + TOFCalibrationData.
./EXTRACTORS.codex-review.md:453:        use crate::data::meta::{read_dia_ms_ms_windows, read_meta_data_sql};
./EXTRACTORS.codex-review.md:459:        let windows = read_dia_ms_ms_windows(&folder).map_err(to_io)?;
./EXTRACTORS.codex-review.md:1253:   432	        use crate::data::meta::{read_dia_ms_ms_windows, read_meta_data_sql};
./EXTRACTORS.codex-review.md:1259:   438	        let windows = read_dia_ms_ms_windows(&folder).map_err(to_io)?;
./EXTRACTORS.codex-review.md:1412:/bin/bash -lc "rg -n \"struct FrameMeta|enum MsMsType|ms_ms_type|read_dia_ms_ms_windows|SWATHMethod|TOFCalibrationData|read_method\" rustdf sciexwiff -g '*.rs' -g '*.md'" in /scratch/timsim-demo/SUBMISSION/rustims
./EXTRACTORS.codex-review.md:1415:rustdf/src/sim/scheme.rs:432:        use crate::data::meta::{read_dia_ms_ms_windows, read_meta_data_sql};
./EXTRACTORS.codex-review.md:1416:rustdf/src/sim/scheme.rs:438:        let windows = read_dia_ms_ms_windows(&folder).map_err(to_io)?;
./EXTRACTORS.codex-review.md:1422:rustdf/src/data/meta.rs:377:pub fn read_dia_ms_ms_windows(
./EXTRACTORS.codex-review.md:1431:rustdf/src/data/dia.rs:6:    read_dia_ms_ms_info, read_dia_ms_ms_windows, read_global_meta_sql, read_meta_data_sql,
./EXTRACTORS.codex-review.md:1432:rustdf/src/data/dia.rs:443:        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
./EXTRACTORS.codex-review.md:1433:rustdf/src/data/dia.rs:503:        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
./EXTRACTORS.codex-review.md:1601:   377	pub fn read_dia_ms_ms_windows(
./EXTRACTORS.codex-review.md:1748:   442	        let dia_ms_mis_info = read_dia_ms_ms_info(data_path).unwrap();
./EXTRACTORS.codex-review.md:1749:   443	        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
./EXTRACTORS.codex-review.md:1784:   478	        let dia_index = DiaIndex::new(&meta_data, &dia_ms_mis_info, &dia_ms_ms_windows);
./EXTRACTORS.codex-review.md:1790:   484	            dia_ms_ms_info: dia_ms_mis_info,
./EXTRACTORS.codex-review.md:1791:   485	            dia_ms_ms_windows,
./EXTRACTORS.codex-review.md:1808:   502	        let dia_ms_mis_info = read_dia_ms_ms_info(data_path).unwrap();
./EXTRACTORS.codex-review.md:1809:   503	        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
./EXTRACTORS.codex-review.md:1834:   528	        let dia_index = DiaIndex::new(&meta_data, &dia_ms_mis_info, &dia_ms_ms_windows);
./EXTRACTORS.codex-review.md:1860:   723	            .dia_ms_ms_info
./EXTRACTORS.codex-review.md:1880:   743	            .dia_ms_ms_windows
./EXTRACTORS.codex-review.md:2070:   743	            .dia_ms_ms_windows
./EXTRACTORS.codex-review.md:2090:   763	        for w in &self.dia_ms_ms_windows {
./ACQUISITION_SCHEME_PLAN.md:39:is timsTOF-coupled: it holds a pandas `dia_ms_ms_windows` table
./ACQUISITION_SCHEME_PLAN.md:44:`dia_ms_ms_windows` + `dia_ms_ms_info` (frame→group) → the output `.d`.
./ACQUISITION_SCHEME_PLAN.md:47:- **Bruker** — `reference_ds.dia_ms_ms_windows` (Python, today).
./ACQUISITION_SCHEME_PLAN.md:175:`dia_ms_ms_info` frame→group sequencing.
./ACQUISITION_SCHEME_PLAN.md:177:- Add `scheme.to_bruker_tables() -> (dia_ms_ms_windows, dia_ms_ms_info)` that
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:57:is timsTOF-coupled: it holds a pandas `dia_ms_ms_windows` table
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:62:`dia_ms_ms_windows` + `dia_ms_ms_info` (frame→group) → the output `.d`.
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:65:- **Bruker** — `reference_ds.dia_ms_ms_windows` (Python, today).
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:193:`dia_ms_ms_info` frame→group sequencing.
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:195:- Add `scheme.to_bruker_tables() -> (dia_ms_ms_windows, dia_ms_ms_info)` that
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:294:- `dia_ms_ms_info` depends on total frame count and cycle phase, so `to_bruker_tables()` needs run timing/frame context, not merely the scheme as currently shown.
./ACQUISITION_SCHEME_PLAN.codex-review-v2.md:365:- `dia_ms_ms_info` depends on total frame count and cycle phase, so `to_bruker_tables()` needs run timing/frame context, not merely the scheme as currently shown.
./BRUKER_INFO.codex-review.md:610:        use crate::data::meta::{read_dia_ms_ms_windows, read_meta_data_sql};
./BRUKER_INFO.codex-review.md:616:        let windows = read_dia_ms_ms_windows(&folder).map_err(to_io)?;
./BRUKER_INFO.codex-review.md:1189:        let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
./BRUKER_INFO.codex-review.md:1296:        let original = crate::data::meta::read_dia_ms_ms_info(&d).expect("read source");
./TO_BRUKER_V2.codex-review.md:521:        use crate::data::meta::{read_dia_ms_ms_windows, read_meta_data_sql};
./TO_BRUKER_V2.codex-review.md:527:        let windows = read_dia_ms_ms_windows(&folder).map_err(to_io)?;
./TO_BRUKER_V2.codex-review.md:1046:        let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
./TO_BRUKER_V2.codex-review.md:1164:/bin/bash -lc 'rg -n "struct DiaMsMsWindow|read_dia_ms_ms_windows|DiaFrameMsMsInfo|WindowGroup|window_group" src tests crates 2>/dev/null' in /scratch/timsim-demo/SUBMISSION/rustims
./TO_BRUKER_V2.codex-review.md:1167:./TO_BRUKER.codex-review.md:16:Trusted context: crate::data::meta::DiaMsMsWindow { window_group:u32, scan_num_begin:u32, scan_num_end:u32, isolation_mz:f64, isolation_width:f64, collision_energy:f64 } and read_dia_ms_ms_windows(folder)->Vec<DiaMsMsWindow> read the Bruker SQLite DiaFrameMsMsWindows table (no ORDER BY). from_bruker_d builds the scheme by grouping those rows by window_group (BTreeMap ascending) into one DiaMs2Frame per group with TimsMobility geometry. The scheme model: cycle is an ordered Vec<AcquisitionEvent> = Ms1 | DiaMs2Frame(windows:Vec<DiaWindow{isolation:{center_mz,width_mz}, collision_energy:CollisionEnergyPolicy{Value|Linear|Unknown}, geometry:DiaGeometry{MzOnly|TimsMobility{scan_start,scan_end}}}>). to_bruker_windows numbers window groups 1..N in cycle (frame) order; the round-trip test normalizes window-group ids to canonical 1..N on both sides and compares sorted exact-bit field tuples.
./TO_BRUKER_V2.codex-review.md:1221:./EXTRACTORS.codex-review.md:16:Trusted context: rustdf reads Bruker TDF natively — read_dia_ms_ms_windows(folder)->Vec<DiaMsMsWindow{window_group,scan_num_begin,scan_num_end,isolation_mz,isolation_width,collision_energy}> and read_meta_data_sql(folder)->Vec<FrameMeta{id,time:f64,ms_ms_type:i64,...}> (MsMsType==0 = MS1/precursor frame). from_bruker_d groups windows by window_group into DiaMs2Frame{TimsMobility windows} and derives cycle time from precursor-frame spacing; verified to extract 15 frames/36 windows from a real DIA-PASEF .d. from_sciex_wiff maps SWATH windows to single MzOnly frames with CE Unknown (SCIEX rolling CE not in the method) and caller-supplied timing; verified 60 windows on a real ZenoTOF .wiff. validate() (already reviewed) requires exactly one MS1 first then >=1 MS2 frame, finite/positive widths, window edges within mz_range, CE finite>=0 for Value/resolved-Linear, multi-window frames need TimsMobility on every window and no within-frame m/z+mobility overlap, start<=gradient. sciexwiff::read_method opens a .wiff OLE2 (cfb crate) and parses SWATHMethod (20B records from off 40) + TOFCalibrationData.
./TO_BRUKER_V2.codex-review.md:1312:./ACQUISITION_SCHEME_PLAN.codex-review.md:179:- `from_bruker_d(ref)` round-trips the existing `dia_ms_ms_windows` columns
./TO_BRUKER_V2.codex-review.md:1551:        use crate::data::meta::{read_dia_ms_ms_windows, read_meta_data_sql};
./TO_BRUKER_V2.codex-review.md:1557:        let windows = read_dia_ms_ms_windows(&folder).map_err(to_io)?;
./TO_BRUKER_V2.codex-review.md:1632:        let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
./TO_BRUKER_V2.codex-review.md:1910:        let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
./TO_BRUKER_V2.codex-review.md:2104:         let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
./TO_BRUKER_V2.codex-review.md:2507:  1028	        let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
./rustdf/src/sim/scheme.rs:601:        use crate::data::meta::{read_dia_ms_ms_info, read_dia_ms_ms_windows, read_meta_data_sql};
./rustdf/src/sim/scheme.rs:607:        let windows = read_dia_ms_ms_windows(&folder).map_err(to_io)?;
./rustdf/src/sim/scheme.rs:639:        let info = read_dia_ms_ms_info(&folder).unwrap_or_default();
./rustdf/src/sim/scheme.rs:1200:        let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
./rustdf/src/sim/scheme.rs:1357:        let original = crate::data::meta::read_dia_ms_ms_info(&d).expect("read source");
./rustdf/examples/dump_bruker_centroids.rs:13:// definitions from `dia_ms_ms_windows` and `dia_ms_ms_info`.
./packages/imspy-dia/src/imspy_dia/clustering/helpers.py:93:    wg = getattr(ds, "dia_ms_ms_info", None)
./packages/imspy-dia/src/imspy_dia/clustering/helpers.py:95:        raise ValueError("Dataset has no dia_ms_ms_info.WindowGroup")
./packages/imspy-dia/src/imspy_dia/clustering/helpers.py:337:        if not hasattr(ds, "dia_ms_ms_info"):
./packages/imspy-dia/src/imspy_dia/clustering/helpers.py:338:            raise AttributeError("Dataset missing 'dia_ms_ms_info'")
./packages/imspy-dia/src/imspy_dia/clustering/helpers.py:339:        msms = ds.dia_ms_ms_info
./packages/imspy-dia/src/imspy_dia/clustering/helpers.py:341:            raise AttributeError("dia_ms_ms_info must have 'WindowGroup' and 'Frame' columns")
./ACQUISITION_SCHEME_PLAN.codex-review.md:16:Focus on: 1) is the AcquisitionScheme data model right — does Option<mobility_scan_range> on DiaWindow cleanly capture the timsTOF-vs-(Thermo/SCIEX) difference, or should mobility be split out? 2) the Rust-vs-Python boundary decision (scheme in Rust/rustdf exposed via connector, Python builder migrates onto it) — risks to backward compatibility with the existing dia_ms_ms_windows table and the Bruker reference path; 3) the 6 open questions — which are real blockers vs deferrable, and any the doc should have answered already; 4) anything missing for a faithful per-instrument acquisition design (e.g. MS1 handling, overlapping windows, rolling CE, cycle/RT modeling); 5) is the input(AcquisitionScheme)/output(AcquisitionWriter) split the right factoring? Concrete, specific, cap ~800 words.
./ACQUISITION_SCHEME_PLAN.codex-review.md:52:- Holds the scheme as a pandas `dia_ms_ms_windows` table with columns
./ACQUISITION_SCHEME_PLAN.codex-review.md:57:  `DiaFrameMsMsWindows` (`reference_ds.dia_ms_ms_windows`).
./ACQUISITION_SCHEME_PLAN.codex-review.md:59:  tables; persists to `synthetic_data.db` (`dia_ms_ms_windows`, `dia_ms_ms_info`)
./ACQUISITION_SCHEME_PLAN.codex-review.md:69:- **Bruker** — `reference_ds.dia_ms_ms_windows` (Python, today).
./ACQUISITION_SCHEME_PLAN.codex-review.md:137:  `reference_ds.dia_ms_ms_windows` directly and instead consumes an
./ACQUISITION_SCHEME_PLAN.codex-review.md:141:  `dia_ms_ms_windows` table is produced from the scheme.
./ACQUISITION_SCHEME_PLAN.codex-review.md:179:- `from_bruker_d(ref)` round-trips the existing `dia_ms_ms_windows` columns
./ACQUISITION_SCHEME_PLAN.codex-review.md:285:        instance.frame_table = instance.synthetics_handle.get_table('frames')
./ACQUISITION_SCHEME_PLAN.codex-review.md:286:        instance.scan_table = instance.synthetics_handle.get_table('scans')
./ACQUISITION_SCHEME_PLAN.codex-review.md:326:        self.synthetics_handle.create_table(
./ACQUISITION_SCHEME_PLAN.codex-review.md:330:        self.synthetics_handle.create_table(
./ACQUISITION_SCHEME_PLAN.codex-review.md:339:        self.synthetics_handle.create_table(
./ACQUISITION_SCHEME_PLAN.codex-review.md:373:        self.dia_ms_ms_windows = pd.read_csv(window_group_file)
./ACQUISITION_SCHEME_PLAN.codex-review.md:411:            self.dia_ms_ms_windows = self.reference.dia_ms_ms_windows.rename(
./ACQUISITION_SCHEME_PLAN.codex-review.md:426:            self.dia_ms_ms_windows['collision_energy'] = np.round(self.dia_ms_ms_windows['collision_energy'].values,
./ACQUISITION_SCHEME_PLAN.codex-review.md:429:        self.synthetics_handle.create_table(
./ACQUISITION_SCHEME_PLAN.codex-review.md:433:        self.synthetics_handle.create_table(
./ACQUISITION_SCHEME_PLAN.codex-review.md:437:        self.synthetics_handle.create_table(
./ACQUISITION_SCHEME_PLAN.codex-review.md:438:            table_name='dia_ms_ms_info',
./ACQUISITION_SCHEME_PLAN.codex-review.md:441:        self.synthetics_handle.create_table(
./ACQUISITION_SCHEME_PLAN.codex-review.md:442:            table_name='dia_ms_ms_windows',
./ACQUISITION_SCHEME_PLAN.codex-review.md:443:            table=self.dia_ms_ms_windows
./ACQUISITION_SCHEME_PLAN.codex-review.md:702:/bin/bash -lc "rg -n \"dia_ms_ms_windows|DiaFrameMsMsWindows|precursor_every|ScanDescriptor|AcquisitionWriter\" packages rustdf imspy_connector -g '*.py' -g '*.rs'" in /scratch/timsim-demo/SUBMISSION/rustims
./ACQUISITION_SCHEME_PLAN.codex-review.md:713:rustdf/src/sim/handle.rs:182:        let mut stmt = self.connection.prepare("SELECT * FROM dia_ms_ms_windows")?;
./ACQUISITION_SCHEME_PLAN.codex-review.md:714:rustdf/src/data/meta.rs:377:pub fn read_dia_ms_ms_windows(
./ACQUISITION_SCHEME_PLAN.codex-review.md:716:rustdf/src/data/dia.rs:6:    read_dia_ms_ms_info, read_dia_ms_ms_windows, read_global_meta_sql, read_meta_data_sql,
./ACQUISITION_SCHEME_PLAN.codex-review.md:717:rustdf/src/data/dia.rs:428:    pub dia_ms_ms_windows: Vec<DiaMsMsWindow>,
./ACQUISITION_SCHEME_PLAN.codex-review.md:718:rustdf/src/data/dia.rs:443:        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
./ACQUISITION_SCHEME_PLAN.codex-review.md:719:rustdf/src/data/dia.rs:478:        let dia_index = DiaIndex::new(&meta_data, &dia_ms_mis_info, &dia_ms_ms_windows);
./ACQUISITION_SCHEME_PLAN.codex-review.md:720:rustdf/src/data/dia.rs:485:            dia_ms_ms_windows,
./ACQUISITION_SCHEME_PLAN.codex-review.md:721:rustdf/src/data/dia.rs:503:        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
./ACQUISITION_SCHEME_PLAN.codex-review.md:722:rustdf/src/data/dia.rs:528:        let dia_index = DiaIndex::new(&meta_data, &dia_ms_mis_info, &dia_ms_ms_windows);
./ACQUISITION_SCHEME_PLAN.codex-review.md:723:rustdf/src/data/dia.rs:535:            dia_ms_ms_windows,
./ACQUISITION_SCHEME_PLAN.codex-review.md:725:rustdf/src/data/dia.rs:547:        self.dia_ms_ms_windows
./ACQUISITION_SCHEME_PLAN.codex-review.md:727:rustdf/src/data/dia.rs:743:            .dia_ms_ms_windows
./ACQUISITION_SCHEME_PLAN.codex-review.md:729:rustdf/src/data/dia.rs:763:        for w in &self.dia_ms_ms_windows {
./ACQUISITION_SCHEME_PLAN.codex-review.md:731:rustdf/examples/dump_bruker_centroids.rs:13:// definitions from `dia_ms_ms_windows` and `dia_ms_ms_info`.
./ACQUISITION_SCHEME_PLAN.codex-review.md:734:packages/imspy-core/src/imspy_core/timstof/dia.py:21:    def dia_ms_ms_windows(self):
./ACQUISITION_SCHEME_PLAN.codex-review.md:736:packages/imspy-vis/src/imspy_vis/frame_rendering.py:289:        self.windows = self.handle.dia_ms_ms_windows.copy()
./ACQUISITION_SCHEME_PLAN.codex-review.md:737:packages/imspy-simulation/src/imspy_simulation/experiment.py:635:        self.dia_ms_ms_windows = None
./ACQUISITION_SCHEME_PLAN.codex-review.md:738:packages/imspy-simulation/src/imspy_simulation/experiment.py:641:        self.dia_ms_ms_windows = self.get_table('dia_ms_ms_windows')
./ACQUISITION_SCHEME_PLAN.codex-review.md:739:packages/imspy-simulation/src/imspy_simulation/experiment.py:649:        for _, row in self.dia_ms_ms_windows.iterrows():
./ACQUISITION_SCHEME_PLAN.codex-review.md:741:packages/imspy-simulation/src/imspy_simulation/acquisition.py:185:        self.dia_ms_ms_windows = pd.read_csv(window_group_file)
./ACQUISITION_SCHEME_PLAN.codex-review.md:748:packages/imspy-simulation/src/imspy_simulation/acquisition.py:223:            self.dia_ms_ms_windows = self.reference.dia_ms_ms_windows.rename(
./ACQUISITION_SCHEME_PLAN.codex-review.md:749:packages/imspy-simulation/src/imspy_simulation/acquisition.py:238:            self.dia_ms_ms_windows['collision_energy'] = np.round(self.dia_ms_ms_windows['collision_energy'].values,
./ACQUISITION_SCHEME_PLAN.codex-review.md:750:packages/imspy-simulation/src/imspy_simulation/acquisition.py:254:            table_name='dia_ms_ms_windows',
./ACQUISITION_SCHEME_PLAN.codex-review.md:751:packages/imspy-simulation/src/imspy_simulation/acquisition.py:255:            table=self.dia_ms_ms_windows
./ACQUISITION_SCHEME_PLAN.codex-review.md:754:packages/imspy-simulation/src/imspy_simulation/acquisition.py:328:        instance.dia_ms_ms_windows = instance.synthetics_handle.get_table('dia_ms_ms_windows')
./ACQUISITION_SCHEME_PLAN.codex-review.md:755:packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:20:        config_name: Name of the config file (e.g., 'dia_ms_ms_windows.csv')
./ACQUISITION_SCHEME_PLAN.codex-review.md:756:packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:504:            # dia-PASEF: Load all windows from actual dia_ms_ms_windows.csv
./ACQUISITION_SCHEME_PLAN.codex-review.md:757:packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:506:            window_df = load_window_config("dia_ms_ms_windows.csv")
./ACQUISITION_SCHEME_PLAN.codex-review.md:758:packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:580:            # midia-PASEF: Load windows from actual midia_ms_ms_windows.csv
./ACQUISITION_SCHEME_PLAN.codex-review.md:759:packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:581:            window_df = load_window_config("midia_ms_ms_windows.csv")
./ACQUISITION_SCHEME_PLAN.codex-review.md:760:packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py:179:        acquisition_builder.tdf_writer.write_dia_ms_ms_windows(
./ACQUISITION_SCHEME_PLAN.codex-review.md:761:packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py:180:            acquisition_builder.synthetics_handle.get_table('dia_ms_ms_windows'))
./ACQUISITION_SCHEME_PLAN.codex-review.md:762:packages/imspy-simulation/src/imspy_simulation/data/database.py:217:        dia_ms_ms_windows: Cached DIA MS/MS windows DataFrame.
./ACQUISITION_SCHEME_PLAN.codex-review.md:763:packages/imspy-simulation/src/imspy_simulation/data/database.py:235:        self.dia_ms_ms_windows: Optional[pd.DataFrame] = None
./ACQUISITION_SCHEME_PLAN.codex-review.md:764:packages/imspy-simulation/src/imspy_simulation/data/database.py:242:            self.dia_ms_ms_windows = self.get_table("dia_ms_ms_windows")
./ACQUISITION_SCHEME_PLAN.codex-review.md:765:packages/imspy-simulation/src/imspy_simulation/data/database.py:266:        if self.dia_ms_ms_windows is None:
./ACQUISITION_SCHEME_PLAN.codex-review.md:766:packages/imspy-simulation/src/imspy_simulation/data/database.py:269:        if self.dia_ms_ms_windows is None:
./ACQUISITION_SCHEME_PLAN.codex-review.md:767:packages/imspy-simulation/src/imspy_simulation/data/database.py:273:        for _, row in self.dia_ms_ms_windows.iterrows():
./ACQUISITION_SCHEME_PLAN.codex-review.md:770:packages/imspy-simulation/src/imspy_simulation/tdf.py:496:    def write_dia_ms_ms_windows(self, dia_ms_ms_windows: pd.DataFrame) -> None:
./ACQUISITION_SCHEME_PLAN.codex-review.md:771:packages/imspy-simulation/src/imspy_simulation/tdf.py:497:        out = dia_ms_ms_windows.rename(columns={
./ACQUISITION_SCHEME_PLAN.codex-review.md:772:packages/imspy-simulation/src/imspy_simulation/tdf.py:506:        self._create_table(self.conn, out, "DiaFrameMsMsWindows")
./ACQUISITION_SCHEME_PLAN.codex-review.md:793:        self.dia_ms_ms_info = None
./ACQUISITION_SCHEME_PLAN.codex-review.md:794:        self.dia_ms_ms_windows = None
./ACQUISITION_SCHEME_PLAN.codex-review.md:799:        self.dia_ms_ms_info = self.get_table('dia_ms_ms_info')
./ACQUISITION_SCHEME_PLAN.codex-review.md:800:        self.dia_ms_ms_windows = self.get_table('dia_ms_ms_windows')
./ACQUISITION_SCHEME_PLAN.codex-review.md:803:        return dict(zip(self.dia_ms_ms_info.frame, self.dia_ms_ms_info.window_group))
./ACQUISITION_SCHEME_PLAN.codex-review.md:808:        for _, row in self.dia_ms_ms_windows.iterrows():
./ACQUISITION_SCHEME_PLAN.codex-review.md:830:    handle.create_table(sql_create_peptides_table)
./ACQUISITION_SCHEME_PLAN.codex-review.md:860:        instance.frame_table = instance.synthetics_handle.get_table('frames')
./ACQUISITION_SCHEME_PLAN.codex-review.md:861:        instance.scan_table = instance.synthetics_handle.get_table('scans')
./ACQUISITION_SCHEME_PLAN.codex-review.md:885:        instance.frames_to_window_groups = instance.synthetics_handle.get_table('dia_ms_ms_info')
./ACQUISITION_SCHEME_PLAN.codex-review.md:886:        instance.dia_ms_ms_windows = instance.synthetics_handle.get_table('dia_ms_ms_windows')
./ACQUISITION_SCHEME_PLAN.codex-review.md:900:pub fn read_dia_ms_ms_info(
./ACQUISITION_SCHEME_PLAN.codex-review.md:926:pub fn read_dia_ms_ms_windows(
./ACQUISITION_SCHEME_PLAN.codex-review.md:992:        let mut stmt = self.connection.prepare("SELECT * FROM dia_ms_ms_windows")?;
./ACQUISITION_SCHEME_PLAN.codex-review.md:1011:        let mut stmt = self.connection.prepare("SELECT * FROM dia_ms_ms_info")?;
./ACQUISITION_SCHEME_PLAN.codex-review.md:1051:    def write_dia_ms_ms_windows(self, dia_ms_ms_windows: pd.DataFrame) -> None:
./ACQUISITION_SCHEME_PLAN.codex-review.md:1052:        out = dia_ms_ms_windows.rename(columns={
./ACQUISITION_SCHEME_PLAN.codex-review.md:1061:        self._create_table(self.conn, out, "DiaFrameMsMsWindows")
./ACQUISITION_SCHEME_PLAN.codex-review.md:1092:        acquisition_builder.tdf_writer.write_dia_ms_ms_info(
./ACQUISITION_SCHEME_PLAN.codex-review.md:1093:            acquisition_builder.synthetics_handle.get_table('dia_ms_ms_info'))
./ACQUISITION_SCHEME_PLAN.codex-review.md:1095:        acquisition_builder.tdf_writer.write_dia_ms_ms_windows(
./ACQUISITION_SCHEME_PLAN.codex-review.md:1096:            acquisition_builder.synthetics_handle.get_table('dia_ms_ms_windows'))
./ACQUISITION_SCHEME_PLAN.codex-review.md:1161:   - `dia_ms_ms_info` frame-to-group sequencing.
./ACQUISITION_SCHEME_PLAN.codex-review.md:1231:   - `dia_ms_ms_info` frame-to-group sequencing.
./rustdf/src/sim/handle.rs:182:        let mut stmt = self.connection.prepare("SELECT * FROM dia_ms_ms_windows")?;
./rustdf/src/sim/handle.rs:201:        let mut stmt = self.connection.prepare("SELECT * FROM dia_ms_ms_info")?;
./packages/imspy-core/src/imspy_core/timstof/data.py:229:    def get_table(self, table_name: str) -> pd.DataFrame:
./TO_BRUKER.codex-review.md:16:Trusted context: crate::data::meta::DiaMsMsWindow { window_group:u32, scan_num_begin:u32, scan_num_end:u32, isolation_mz:f64, isolation_width:f64, collision_energy:f64 } and read_dia_ms_ms_windows(folder)->Vec<DiaMsMsWindow> read the Bruker SQLite DiaFrameMsMsWindows table (no ORDER BY). from_bruker_d builds the scheme by grouping those rows by window_group (BTreeMap ascending) into one DiaMs2Frame per group with TimsMobility geometry. The scheme model: cycle is an ordered Vec<AcquisitionEvent> = Ms1 | DiaMs2Frame(windows:Vec<DiaWindow{isolation:{center_mz,width_mz}, collision_energy:CollisionEnergyPolicy{Value|Linear|Unknown}, geometry:DiaGeometry{MzOnly|TimsMobility{scan_start,scan_end}}}>). to_bruker_windows numbers window groups 1..N in cycle (frame) order; the round-trip test normalizes window-group ids to canonical 1..N on both sides and compares sorted exact-bit field tuples.
./TO_BRUKER.codex-review.md:510:        use crate::data::meta::{read_dia_ms_ms_windows, read_meta_data_sql};
./TO_BRUKER.codex-review.md:516:        let windows = read_dia_ms_ms_windows(&folder).map_err(to_io)?;
./TO_BRUKER.codex-review.md:1029:        let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
./rustdf/src/data/meta.rs:351:pub fn read_dia_ms_ms_info(
./rustdf/src/data/meta.rs:377:pub fn read_dia_ms_ms_windows(
./rustdf/src/data/dia.rs:6:    read_dia_ms_ms_info, read_dia_ms_ms_windows, read_global_meta_sql, read_meta_data_sql,
./rustdf/src/data/dia.rs:427:    pub dia_ms_ms_info: Vec<DiaMsMisInfo>,
./rustdf/src/data/dia.rs:428:    pub dia_ms_ms_windows: Vec<DiaMsMsWindow>,
./rustdf/src/data/dia.rs:442:        let dia_ms_mis_info = read_dia_ms_ms_info(data_path).unwrap();
./rustdf/src/data/dia.rs:443:        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
./rustdf/src/data/dia.rs:478:        let dia_index = DiaIndex::new(&meta_data, &dia_ms_mis_info, &dia_ms_ms_windows);
./rustdf/src/data/dia.rs:484:            dia_ms_ms_info: dia_ms_mis_info,
./rustdf/src/data/dia.rs:485:            dia_ms_ms_windows,
./rustdf/src/data/dia.rs:502:        let dia_ms_mis_info = read_dia_ms_ms_info(data_path).unwrap();
./rustdf/src/data/dia.rs:503:        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
./rustdf/src/data/dia.rs:528:        let dia_index = DiaIndex::new(&meta_data, &dia_ms_mis_info, &dia_ms_ms_windows);
./rustdf/src/data/dia.rs:534:            dia_ms_ms_info: dia_ms_mis_info,
./rustdf/src/data/dia.rs:535:            dia_ms_ms_windows,
./rustdf/src/data/dia.rs:547:        self.dia_ms_ms_windows
./rustdf/src/data/dia.rs:618:            .dia_ms_ms_info
./rustdf/src/data/dia.rs:697:            .dia_ms_ms_info
./rustdf/src/data/dia.rs:723:            .dia_ms_ms_info
./rustdf/src/data/dia.rs:743:            .dia_ms_ms_windows
./rustdf/src/data/dia.rs:763:        for w in &self.dia_ms_ms_windows {
./packages/imspy-core/src/imspy_core/timstof/dia.py:21:    def dia_ms_ms_windows(self):
./packages/imspy-core/src/imspy_core/timstof/dia.py:31:    def dia_ms_ms_info(self):
./packages/imspy-vis/src/imspy_vis/frame_rendering.py:289:        self.windows = self.handle.dia_ms_ms_windows.copy()
./packages/imspy-vis/src/imspy_vis/frame_rendering.py:290:        self.frame_to_group = dict(zip(self.handle.dia_ms_ms_info.Frame, self.handle.dia_ms_ms_info.WindowGroup))
./packages/imspy-dia/src/imspy_dia/pipeline/cluster_pipeline.py:559:    info = ds.dia_ms_ms_info
./packages/imspy-simulation/src/imspy_simulation/experiment.py:545:    def create_table(self, table_name: str, table: pd.DataFrame):
./packages/imspy-simulation/src/imspy_simulation/experiment.py:595:    def get_table(self, table_name: str) -> pd.DataFrame:
./packages/imspy-simulation/src/imspy_simulation/experiment.py:605:        return self.get_table("frames")
./packages/imspy-simulation/src/imspy_simulation/experiment.py:634:        self.dia_ms_ms_info = None
./packages/imspy-simulation/src/imspy_simulation/experiment.py:635:        self.dia_ms_ms_windows = None
./packages/imspy-simulation/src/imspy_simulation/experiment.py:640:        self.dia_ms_ms_info = self.get_table('dia_ms_ms_info')
./packages/imspy-simulation/src/imspy_simulation/experiment.py:641:        self.dia_ms_ms_windows = self.get_table('dia_ms_ms_windows')
./packages/imspy-simulation/src/imspy_simulation/experiment.py:644:        return dict(zip(self.dia_ms_ms_info.frame, self.dia_ms_ms_info.window_group))
./packages/imspy-simulation/src/imspy_simulation/experiment.py:649:        for _, row in self.dia_ms_ms_windows.iterrows():
./packages/imspy-simulation/src/imspy_simulation/experiment.py:671:    handle.create_table(sql_create_peptides_table)
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:97:        instance.frame_table = instance.synthetics_handle.get_table('frames')
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:98:        instance.scan_table = instance.synthetics_handle.get_table('scans')
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:138:        self.synthetics_handle.create_table(
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:142:        self.synthetics_handle.create_table(
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:151:        self.synthetics_handle.create_table(
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:188:        self.dia_ms_ms_windows = pd.read_csv(window_group_file)
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:224:        Returns ``(dia_ms_ms_windows, frames_to_window_groups, precursor_every)``
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:252:            self.dia_ms_ms_windows, frames_to_window_groups, self.precursor_every = scheme_layout
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:259:                self.dia_ms_ms_windows = self.reference.dia_ms_ms_windows.rename(
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:273:            self.dia_ms_ms_windows['collision_energy'] = np.round(self.dia_ms_ms_windows['collision_energy'].values,
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:276:        self.synthetics_handle.create_table(
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:280:        self.synthetics_handle.create_table(
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:284:        self.synthetics_handle.create_table(
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:285:            table_name='dia_ms_ms_info',
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:288:        self.synthetics_handle.create_table(
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:289:            table_name='dia_ms_ms_windows',
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:290:            table=self.dia_ms_ms_windows
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:337:        instance.frame_table = instance.synthetics_handle.get_table('frames')
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:338:        instance.scan_table = instance.synthetics_handle.get_table('scans')
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:362:        instance.frames_to_window_groups = instance.synthetics_handle.get_table('dia_ms_ms_info')
./packages/imspy-simulation/src/imspy_simulation/acquisition.py:363:        instance.dia_ms_ms_windows = instance.synthetics_handle.get_table('dia_ms_ms_windows')
./packages/imspy-simulation/src/imspy_simulation/data/database.py:77:    def create_table(self, table_name: str, table: pd.DataFrame) -> None:
./packages/imspy-simulation/src/imspy_simulation/data/database.py:142:    def get_table(self, table_name: str) -> pd.DataFrame:
./packages/imspy-simulation/src/imspy_simulation/data/database.py:159:        return self.get_table("frames")
./packages/imspy-simulation/src/imspy_simulation/data/database.py:216:        dia_ms_ms_info: Cached DIA MS/MS info DataFrame.
./packages/imspy-simulation/src/imspy_simulation/data/database.py:217:        dia_ms_ms_windows: Cached DIA MS/MS windows DataFrame.
./packages/imspy-simulation/src/imspy_simulation/data/database.py:234:        self.dia_ms_ms_info: Optional[pd.DataFrame] = None
./packages/imspy-simulation/src/imspy_simulation/data/database.py:235:        self.dia_ms_ms_windows: Optional[pd.DataFrame] = None
./packages/imspy-simulation/src/imspy_simulation/data/database.py:241:            self.dia_ms_ms_info = self.get_table("dia_ms_ms_info")
./packages/imspy-simulation/src/imspy_simulation/data/database.py:242:            self.dia_ms_ms_windows = self.get_table("dia_ms_ms_windows")
./packages/imspy-simulation/src/imspy_simulation/data/database.py:252:        if self.dia_ms_ms_info is None:
./packages/imspy-simulation/src/imspy_simulation/data/database.py:255:        if self.dia_ms_ms_info is None:
./packages/imspy-simulation/src/imspy_simulation/data/database.py:258:        return dict(zip(self.dia_ms_ms_info.frame, self.dia_ms_ms_info.window_group))
./packages/imspy-simulation/src/imspy_simulation/data/database.py:266:        if self.dia_ms_ms_windows is None:
./packages/imspy-simulation/src/imspy_simulation/data/database.py:269:        if self.dia_ms_ms_windows is None:
./packages/imspy-simulation/src/imspy_simulation/data/database.py:273:        for _, row in self.dia_ms_ms_windows.iterrows():
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/dda_selection_scheme.py:62:    scan_max = acquisition_builder.synthetics_handle.get_table("scans").scan.max()
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/dda_selection_scheme.py:75:    ms_1_frames = set(handle.get_table("frames")[handle.get_table("frames").ms_type == 0].frame_id.values)
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/dda_selection_scheme.py:78:    peptides = handle.get_table("peptides")
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/dda_selection_scheme.py:79:    ions = handle.get_table("ions")
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/dda_selection_scheme.py:129:    acquisition_builder.synthetics_handle.create_table(
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py:176:        acquisition_builder.tdf_writer.write_dia_ms_ms_info(
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py:177:            acquisition_builder.synthetics_handle.get_table('dia_ms_ms_info'))
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py:179:        acquisition_builder.tdf_writer.write_dia_ms_ms_windows(
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py:180:            acquisition_builder.synthetics_handle.get_table('dia_ms_ms_windows'))
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py:203:            id_mapping_table = acquisition_builder.synthetics_handle.get_table('precursor_id_mapping')
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py:210:            acquisition_builder.synthetics_handle.get_table('precursors'),
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py:216:            acquisition_builder.synthetics_handle.get_table('pasef_meta'),
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/simulate_fragment_intensities.py:396:            acquisition_builder.synthetics_handle.create_table(
./packages/imspy-simulation/src/imspy_simulation/timsim/jobs/simulate_fragment_intensities.py:550:                acquisition_builder.synthetics_handle.create_table(
./packages/imspy-simulation/src/imspy_simulation/tdf.py:83:        frame_ms_ms_info = self.helper_handle.get_table("FrameMsmsInfo")
./packages/imspy-simulation/src/imspy_simulation/tdf.py:84:        segments = self.helper_handle.get_table("Segments")
./packages/imspy-simulation/src/imspy_simulation/tdf.py:94:        self._create_table(self.conn, self.helper_handle.mz_calibration, "MzCalibration")
./packages/imspy-simulation/src/imspy_simulation/tdf.py:95:        self._create_table(self.conn, self.helper_handle.tims_calibration, "TimsCalibration")
./packages/imspy-simulation/src/imspy_simulation/tdf.py:96:        self._create_table(self.conn, self.helper_handle.global_meta_data_pandas, "GlobalMetadata")
./packages/imspy-simulation/src/imspy_simulation/tdf.py:97:        self._create_table(self.conn, frame_ms_ms_info, "FrameMsmsInfo")
./packages/imspy-simulation/src/imspy_simulation/tdf.py:98:        self._create_table(self.conn, segments, "Segments")
./packages/imspy-simulation/src/imspy_simulation/tdf.py:110:    def _get_table(conn, table_name: str) -> pd.DataFrame:
./packages/imspy-simulation/src/imspy_simulation/tdf.py:115:    def _create_table(conn, table, table_name: str) -> None:
./packages/imspy-simulation/src/imspy_simulation/tdf.py:324:        self._create_table(self.conn, meta_df, "Frames")
./packages/imspy-simulation/src/imspy_simulation/tdf.py:341:            table = self.helper_handle.get_table("CalibrationInfo")
./packages/imspy-simulation/src/imspy_simulation/tdf.py:343:            self._create_table(self.conn, table, "CalibrationInfo")
./packages/imspy-simulation/src/imspy_simulation/tdf.py:350:            self._create_table(self.conn, self.helper_handle.get_table("PasefFrameMsMsInfo"), "PasefFrameMsMsInfo")
./packages/imspy-simulation/src/imspy_simulation/tdf.py:356:            self._create_table(self.conn, self.helper_handle.get_table("PrmFrameMsMsInfo"), "PrmFrameMsMsInfo")
./packages/imspy-simulation/src/imspy_simulation/tdf.py:360:    def write_dia_ms_ms_info(self, dia_ms_ms_info: pd.DataFrame) -> None:
./packages/imspy-simulation/src/imspy_simulation/tdf.py:361:        out = dia_ms_ms_info.rename(columns={
./packages/imspy-simulation/src/imspy_simulation/tdf.py:366:        self._create_table(self.conn, out, "DiaFrameMsMsInfo")
./packages/imspy-simulation/src/imspy_simulation/tdf.py:496:    def write_dia_ms_ms_windows(self, dia_ms_ms_windows: pd.DataFrame) -> None:
./packages/imspy-simulation/src/imspy_simulation/tdf.py:497:        out = dia_ms_ms_windows.rename(columns={
./packages/imspy-simulation/src/imspy_simulation/tdf.py:506:        self._create_table(self.conn, out, "DiaFrameMsMsWindows")
./packages/imspy-simulation/src/imspy_simulation/timsim/simulator.py:906:        peptides = existing_sim_handle.get_table('peptides')
./packages/imspy-simulation/src/imspy_simulation/timsim/simulator.py:907:        proteins = existing_sim_handle.get_table('proteins')
./packages/imspy-simulation/src/imspy_simulation/timsim/simulator.py:908:        ions = existing_sim_handle.get_table('ions')
./packages/imspy-simulation/src/imspy_simulation/timsim/simulator.py:1184:        acquisition_builder.synthetics_handle.create_table(table_name='proteins', table=proteins)
./packages/imspy-simulation/src/imspy_simulation/timsim/simulator.py:1198:        acquisition_builder.synthetics_handle.create_table(table_name='peptides', table=peptides)
./packages/imspy-simulation/src/imspy_simulation/timsim/simulator.py:1270:        acquisition_builder.synthetics_handle.create_table(table_name='ions', table=ions)
./packages/imspy-simulation/src/imspy_simulation/timsim/simulator.py:1283:            acquisition_builder.synthetics_handle.create_table(table_name='pasef_meta', table=pasef_meta)
./packages/imspy-simulation/src/imspy_simulation/timsim/simulator.py:1284:            acquisition_builder.synthetics_handle.create_table(table_name='precursors', table=precursors)
./packages/imspy-simulation/src/imspy_simulation/timsim/simulator.py:1298:        acquisition_builder.synthetics_handle.create_table(table_name='proteins', table=proteins)
./packages/imspy-simulation/src/imspy_simulation/timsim/simulator.py:1299:        acquisition_builder.synthetics_handle.create_table(table_name='peptides', table=peptides)
./packages/imspy-simulation/src/imspy_simulation/timsim/simulator.py:1300:        acquisition_builder.synthetics_handle.create_table(table_name='ions', table=ions)
./packages/imspy-simulation/src/imspy_simulation/timsim/simulator.py:1302:            acquisition_builder.synthetics_handle.create_table(table_name='pasef_meta', table=pasef_meta)
./packages/imspy-simulation/src/imspy_simulation/timsim/simulator.py:1304:            acquisition_builder.synthetics_handle.create_table(table_name='precursors', table=precursors)
./packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:20:        config_name: Name of the config file (e.g., 'dia_ms_ms_windows.csv')
./packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:504:            # dia-PASEF: Load all windows from actual dia_ms_ms_windows.csv
./packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:506:            window_df = load_window_config("dia_ms_ms_windows.csv")
./packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:580:            # midia-PASEF: Load windows from actual midia_ms_ms_windows.csv
./packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:581:            window_df = load_window_config("midia_ms_ms_windows.csv")
./packages/imspy-simulation/src/imspy_simulation/timsim/validate/comparison.py:48:    peptides = handle.get_table("peptides")
./packages/imspy-simulation/src/imspy_simulation/timsim/validate/comparison.py:49:    ions = handle.get_table("ions")
./packages/imspy-simulation/src/imspy_simulation/timsim/validate/comparison.py:54:        fragment_ions = handle.get_table("fragment_ions")

codex
The submitted paths are under `packages/imspy-simulation`, and the working tree diff is empty for the shorthand paths. I found the live implementation and am checking SQLite serialization plus the Rust/Python consumers, where dtype and ordering assumptions become observable.
exec
/bin/bash -lc 'git status --short; git log -1 --oneline -- packages/imspy-simulation/src/imspy_simulation/acquisition.py tests/test_acquisition_scheme_parity.py' in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
 M imspy_connector/src/py_acquisition.rs
?? MIGRATION.codex-review.md
?? packages/imspy-predictors/scripts/compare_libraries.py
?? packages/imspy-predictors/scripts/generate_calibrated_library.py
?? primitives/
?? rustdf/examples/build_bruker_pseudo_ms2.rs
?? rustdf/examples/build_bruker_pseudo_ms2_v3.rs
?? rustdf/examples/dump_bruker_ms1_centroids.rs
?? scripts/compare_diann_to_truth.py
?? scripts/reread_rustw_smoke.py
ea07bcf1 feat(imspy-sim): migrate TimsTofAcquisitionBuilderDIA onto AcquisitionScheme

exec
/bin/bash -lc "nl -ba packages/imspy-simulation/src/imspy_simulation/acquisition.py | sed -n '160,310p'; nl -ba tests/test_acquisition_scheme_parity.py 2>/dev/null || true" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
   160	                 window_group_file: str,
   161	                 acquisition_name: str = "dia",
   162	                 exp_name: str = "RAW",
   163	                 verbose: bool = True,
   164	                 precursor_every: int = 17,
   165	                 gradient_length=50 * 60,
   166	                 rt_cycle_length=0.1054,
   167	                 use_reference_ds_layout: bool = True,
   168	                 round_collision_energy: bool = True,
   169	                 collision_energy_decimals: int = 1
   170	                 ):
   171	
   172	        super().__init__(path, reference_ds, gradient_length, rt_cycle_length,
   173	                         exp_name=exp_name)
   174	        # TODO: check this, could be missing replacement of reference layout of windows
   175	        if use_reference_ds_layout:
   176	            rt_cycle_length = np.mean(np.diff(reference_ds.meta_data.Time))
   177	            if verbose:
   178	                print('Using reference dataset cycle length:', np.round(rt_cycle_length, 4))
   179	            self.rt_cycle_length = rt_cycle_length
   180	            # Recompute the frame count: the base __init__ computed it from the
   181	            # passed rt_cycle_length before this reference-derived value replaced it.
   182	            self.num_frames = calculate_number_frames(self.gradient_length, rt_cycle_length)
   183	
   184	        self.acquisition_name = acquisition_name
   185	        self.scan_table = None
   186	        self.frame_table = None
   187	        self.frames_to_window_groups = None
   188	        self.dia_ms_ms_windows = pd.read_csv(window_group_file)
   189	        self.use_reference_ds_layout = use_reference_ds_layout
   190	        self.reference = reference_ds
   191	        self.round_collision_energy = round_collision_energy
   192	        self.collision_energy_decimals = collision_energy_decimals
   193	
   194	        # TODO: check if the number of scans in the window group file matches the number of scans in the experiment
   195	
   196	        self.acquisition_mode = AcquisitionMode('DIA')
   197	        self.verbose = verbose
   198	        self.precursor_every = precursor_every
   199	
   200	        self._setup(verbose=verbose)
   201	
   202	    def calculate_frame_types(self, verbose: bool = True) -> NDArray:
   203	        if verbose:
   204	            print(f'Calculating frame types, precursor frame will be taken every {self.precursor_every} rt cycles.')
   205	        return np.array([0 if (x - 1) % self.precursor_every == 0 else 9 for x in self.frame_table.frame_id])
   206	
   207	    def generate_frame_to_window_group_table(self, verbose: bool = True) -> pd.DataFrame:
   208	        if verbose:
   209	            print(f'generating frame to window group table, precursors every {self.precursor_every} frames.')
   210	
   211	        table_list = []
   212	        for index, row in self.frame_table.iterrows():
   213	            frame_id, ms_type = row.frame_id, row.ms_type
   214	            wg = index % self.precursor_every
   215	            if ms_type > 0:
   216	                table_list.append({'frame': int(frame_id), 'window_group': wg})
   217	
   218	        return pd.DataFrame(table_list)
   219	
   220	    def _layout_from_scheme(self, verbose: bool = True):
   221	        """Derive the DIA window + frame→group tables from the reference `.d` via
   222	        the vendor-neutral ``AcquisitionScheme`` (imspy_connector.py_acquisition).
   223	
   224	        Returns ``(dia_ms_ms_windows, frames_to_window_groups, precursor_every)``
   225	        or ``None`` if the connector lacks the scheme module (legacy fallback).
   226	        """
   227	        try:
   228	            import imspy_connector
   229	            acq = imspy_connector.py_acquisition
   230	        except (ImportError, AttributeError):
   231	            if verbose:
   232	                print('py_acquisition unavailable; using legacy reference layout')
   233	            return None
   234	
   235	        scheme = acq.PyAcquisitionScheme.from_bruker_d(self.reference.data_path)
   236	        precursor_every = scheme.cycle_length()
   237	        windows = pd.DataFrame(scheme.to_bruker_windows())
   238	        info = pd.DataFrame(scheme.to_bruker_info(int(self.num_frames)))
   239	        if verbose:
   240	            print(f'Using AcquisitionScheme layout: {len(windows)} windows, '
   241	                  f'{scheme.n_ms2_frames()} groups, precursor_every={precursor_every}')
   242	        return windows, info, precursor_every
   243	
   244	    def _setup(self, verbose: bool = True):
   245	        self.frame_table = self.generate_frame_table(verbose=verbose)
   246	        self.scan_table = self.generate_scan_table(verbose=verbose)
   247	
   248	        scheme_layout = self._layout_from_scheme(verbose=verbose) if self.use_reference_ds_layout else None
   249	
   250	        if scheme_layout is not None:
   251	            # Vendor-neutral path: windows + frame→group come from the scheme.
   252	            self.dia_ms_ms_windows, frames_to_window_groups, self.precursor_every = scheme_layout
   253	            self.frame_table['ms_type'] = self.calculate_frame_types(verbose=verbose)
   254	            self.frames_to_window_groups = frames_to_window_groups
   255	        else:
   256	            # Legacy path: copy the reference .d's window table directly.
   257	            if self.use_reference_ds_layout:
   258	                self.precursor_every = int(np.diff(self.reference.precursor_frames)[0])
   259	                self.dia_ms_ms_windows = self.reference.dia_ms_ms_windows.rename(
   260	                    columns={
   261	                        'WindowGroup': 'window_group',
   262	                        'ScanNumBegin': 'scan_start',
   263	                        'ScanNumEnd': 'scan_end',
   264	                        'IsolationMz': 'isolation_mz',
   265	                        'IsolationWidth': 'isolation_width',
   266	                        'CollisionEnergy': 'collision_energy',
   267	                    }
   268	                )
   269	            self.frame_table['ms_type'] = self.calculate_frame_types(verbose=verbose)
   270	            self.frames_to_window_groups = self.generate_frame_to_window_group_table(verbose=verbose)
   271	
   272	        if self.round_collision_energy:
   273	            self.dia_ms_ms_windows['collision_energy'] = np.round(self.dia_ms_ms_windows['collision_energy'].values,
   274	                                                                  decimals=self.collision_energy_decimals)
   275	
   276	        self.synthetics_handle.create_table(
   277	            table_name='frames',
   278	            table=self.frame_table
   279	        )
   280	        self.synthetics_handle.create_table(
   281	            table_name='scans',
   282	            table=self.scan_table
   283	        )
   284	        self.synthetics_handle.create_table(
   285	            table_name='dia_ms_ms_info',
   286	            table=self.frames_to_window_groups
   287	        )
   288	        self.synthetics_handle.create_table(
   289	            table_name='dia_ms_ms_windows',
   290	            table=self.dia_ms_ms_windows
   291	        )
   292	
   293	    @staticmethod
   294	    def from_config(
   295	            path: str,
   296	            reference_ds: TimsDatasetDIA,
   297	            exp_name: str,
   298	            config: Dict[str, any],
   299	            verbose: bool = True,
   300	            use_reference_layout: bool = True,
   301	            round_collision_energy: bool = True,
   302	            collision_energy_decimals: int = 1
   303	    ) -> 'TimsTofAcquisitionBuilderDIA':
   304	
   305	        acquisition_name = config['name'].lower().replace('pasef', '')
   306	        window_group_file = get_ms_ms_window_layout_resource_path(acquisition_name)
   307	
   308	        return TimsTofAcquisitionBuilderDIA(
   309	            path=str(Path(path) / exp_name),
   310	            reference_ds=reference_ds,

exec
/bin/bash -lc "nl -ba packages/imspy-simulation/src/imspy_simulation/experiment.py | sed -n '530,665p'; nl -ba packages/imspy-simulation/src/imspy_simulation/tdf.py | sed -n '105,125p;350,375p;490,515p'; nl -ba rustdf/src/sim/handle.rs | sed -n '170,220p'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
   530	        self.base_path = database_path
   531	        self.database_path = os.path.join(self.base_path, database_name)
   532	        self.conn = None
   533	
   534	        self._setup()
   535	
   536	    def _setup(self):
   537	        if not os.path.exists(self.base_path):
   538	            if self.verbose:
   539	                print(f"Creating data directory: {self.base_path}")
   540	            os.makedirs(self.base_path)
   541	        if self.verbose:
   542	            print(f"Connecting to database: {self.database_path}")
   543	        self.conn = sqlite3.connect(self.database_path)
   544	
   545	    def create_table(self, table_name: str, table: pd.DataFrame):
   546	        # Create a table from a pandas DataFrame
   547	        table.to_sql(table_name, self.conn, if_exists='replace', index=False)
   548	        # Create indexes for efficient lazy loading queries
   549	        self._create_indexes_for_table(table_name)
   550	
   551	    def _create_indexes_for_table(self, table_name: str):
   552	        """Create indexes for efficient lazy loading queries based on table name."""
   553	        indexes = {
   554	            'peptides': [
   555	                "CREATE INDEX IF NOT EXISTS idx_peptides_frame_range ON peptides(frame_occurrence_start, frame_occurrence_end)",
   556	                "CREATE INDEX IF NOT EXISTS idx_peptides_peptide_id ON peptides(peptide_id)",
   557	            ],
   558	            'ions': [
   559	                "CREATE INDEX IF NOT EXISTS idx_ions_peptide_id ON ions(peptide_id)",
   560	            ],
   561	            'fragment_ions': [
   562	                "CREATE INDEX IF NOT EXISTS idx_fragment_ions_peptide_id ON fragment_ions(peptide_id)",
   563	                "CREATE INDEX IF NOT EXISTS idx_fragment_ions_lookup ON fragment_ions(peptide_id, charge)",
   564	            ],
   565	        }
   566	
   567	        if table_name in indexes:
   568	            cursor = self.conn.cursor()
   569	            for sql in indexes[table_name]:
   570	                try:
   571	                    cursor.execute(sql)
   572	                except sqlite3.Error as e:
   573	                    if self.verbose:
   574	                        print(f"Warning: Could not create index: {e}")
   575	            self.conn.commit()
   576	
   577	    def append_table(self, table_name: str, table: pd.DataFrame):
   578	        # Append a table to an existing table in the database
   579	        table.to_sql(table_name, self.conn, if_exists='append', index=False)
   580	
   581	    def create_table_sql(self, sql):
   582	        # Create a table as per the provided SQL statement
   583	        try:
   584	            cursor = self.conn.cursor()
   585	            cursor.execute(sql)
   586	            self.conn.commit()
   587	        except sqlite3.Error as e:
   588	            print(f"An error occurred: {e}")
   589	
   590	    def close(self):
   591	        # Close the database connection
   592	        if self.conn:
   593	            self.conn.close()
   594	
   595	    def get_table(self, table_name: str) -> pd.DataFrame:
   596	        # Get a table as a pandas DataFrame
   597	        return pd.read_sql(f"SELECT * FROM {table_name}", self.conn)
   598	
   599	    def get_frame_meta_data(self) -> pd.DataFrame:
   600	        """Get frame metadata table.
   601	
   602	        Returns:
   603	            Frame metadata DataFrame.
   604	        """
   605	        return self.get_table("frames")
   606	
   607	    def list_tables(self):
   608	        with self.conn:
   609	            cursor = self.conn.cursor()
   610	            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
   611	            tables = cursor.fetchall()
   612	            return [table[0] for table in tables]
   613	
   614	    def list_columns(self, table_name):
   615	        if table_name not in self.list_tables():
   616	            raise ValueError(f"Table '{table_name}' does not exist in the database.")
   617	
   618	        with self.conn:
   619	            cursor = self.conn.cursor()
   620	            cursor.execute(f"PRAGMA table_info({table_name});")
   621	            columns = cursor.fetchall()
   622	            return [column[1] for column in columns]
   623	
   624	    def __repr__(self):
   625	        return f"SyntheticExperimentDataHandle(database_path={self.database_path})"
   626	
   627	
   628	class SyntheticExperimentDataHandleDIA(SyntheticExperimentDataHandle, ABC):
   629	    def __init__(self,
   630	                 database_path: str,
   631	                 database_name: str = 'synthetic_data.db',
   632	                 verbose: bool = True,):
   633	        super().__init__(database_path, database_name, verbose)
   634	        self.dia_ms_ms_info = None
   635	        self.dia_ms_ms_windows = None
   636	
   637	        self._additional_setup()
   638	
   639	    def _additional_setup(self):
   640	        self.dia_ms_ms_info = self.get_table('dia_ms_ms_info')
   641	        self.dia_ms_ms_windows = self.get_table('dia_ms_ms_windows')
   642	
   643	    def get_frame_to_window_group(self):
   644	        return dict(zip(self.dia_ms_ms_info.frame, self.dia_ms_ms_info.window_group))
   645	
   646	    def get_window_group_settings(self):
   647	        window_group_settings = {}
   648	
   649	        for _, row in self.dia_ms_ms_windows.iterrows():
   650	            key = (row.window_group, row.scan_start)
   651	            value = (row.mz_mid, row.mz_width)
   652	            window_group_settings[key] = value
   653	
   654	        return window_group_settings
   655	
   656	
   657	if __name__ == '__main__':
   658	
   659	    # Example usage
   660	    path = '/path/to/directory'
   661	    db_name = 'experiment_data.db'
   662	    handle = SyntheticExperimentDataHandle(path, db_name)
   663	
   664	    # Create a table, for example
   665	    sql_create_peptides_table = '''
   105	
   106	        if self.verbose:
   107	            print(f"Setting up TDF file meta data, created: {self.full_path}/analysis.tdf and {self.full_path}/analysis.tdf_bin")
   108	
   109	    @staticmethod
   110	    def _get_table(conn, table_name: str) -> pd.DataFrame:
   111	        # Get a table as a pandas DataFrame
   112	        return pd.read_sql(f"SELECT * FROM {table_name}", conn)
   113	
   114	    @staticmethod
   115	    def _create_table(conn, table, table_name: str) -> None:
   116	        # Create a table from a pandas DataFrame
   117	        table.to_sql(table_name, conn, if_exists='replace', index=False)
   118	
   119	    def mz_to_tof(self, frame_id, mzs):
   120	        """Convert m/z values to TOF values for a given frame using the helper handle.
   121	        # CAUTION: This will use the calibration data from the reference handle.
   122	        """
   123	
   124	        try:
   125	            max_ref_frame_id = self.helper_handle.meta_data.Id.max()
   350	            self._create_table(self.conn, self.helper_handle.get_table("PasefFrameMsMsInfo"), "PasefFrameMsMsInfo")
   351	        except Exception as e:
   352	            print(f"Error writing PasefFrameMsMsInfo table: {e}. In most cases, this is not a problem, since the table is empty in DIA mode.")
   353	
   354	    def write_prm_frame_ms_ms_info(self) -> None:
   355	        try:
   356	            self._create_table(self.conn, self.helper_handle.get_table("PrmFrameMsMsInfo"), "PrmFrameMsMsInfo")
   357	        except Exception as e:
   358	            print(f"Error writing PrmFrameMsMsInfo table: {e}")
   359	
   360	    def write_dia_ms_ms_info(self, dia_ms_ms_info: pd.DataFrame) -> None:
   361	        out = dia_ms_ms_info.rename(columns={
   362	            'frame': 'Frame',
   363	            'window_group': 'WindowGroup',
   364	        })
   365	
   366	        self._create_table(self.conn, out, "DiaFrameMsMsInfo")
   367	
   368	    def write_precursor_table(self, precursor_table: pd.DataFrame, id_mapping_table: pd.DataFrame = None) -> dict:
   369	        """Write the Precursors table with proper vendor schema.
   370	
   371	        Args:
   372	            precursor_table: DataFrame with precursor data
   373	            id_mapping_table: Optional DataFrame with 'ion_id' and 'tdf_precursor_id' columns.
   374	                             If None, creates sequential mapping from existing IDs.
   375	
   490	                float(row['IsolationWidth']),
   491	                float(row['CollisionEnergy']),
   492	                precursor_id
   493	            ))
   494	        self.conn.commit()
   495	
   496	    def write_dia_ms_ms_windows(self, dia_ms_ms_windows: pd.DataFrame) -> None:
   497	        out = dia_ms_ms_windows.rename(columns={
   498	            'window_group': 'WindowGroup',
   499	            'scan_start': 'ScanNumBegin',
   500	            'scan_end': 'ScanNumEnd',
   501	            'isolation_mz': 'IsolationMz',
   502	            'isolation_width': 'IsolationWidth',
   503	            'collision_energy': 'CollisionEnergy',
   504	        })
   505	
   506	        self._create_table(self.conn, out, "DiaFrameMsMsWindows")
   507	
   508	        # TODO: these methods needs to be debugged
   509	        """
   510	        def compress_frames(self, frames: List[TimsFrame], only_frame_one: bool = False, num_threads: int = 4) -> List[bytes]:
   511	            # same as compress_frame but for multiple frames
   512	            tofs, scans, intensities = [], [], []
   513	            for frame in frames:
   514	                i = 1 if only_frame_one else frame.frame_id
   515	                tofs.append(self.mz_to_tof(i, frame.mz).astype(np.uint32))
   170	
   171	    pub fn read_ions(&self) -> rusqlite::Result<Vec<IonSim>> {
   172	        let mut stmt = self.connection.prepare("SELECT * FROM ions")?;
   173	        let ions_iter = stmt.query_map([], |row| Self::ion_from_row(row))?;
   174	        let mut ions = Vec::new();
   175	        for ion in ions_iter {
   176	            ions.push(ion?);
   177	        }
   178	        Ok(ions)
   179	    }
   180	
   181	    pub fn read_window_group_settings(&self) -> rusqlite::Result<Vec<WindowGroupSettingsSim>> {
   182	        let mut stmt = self.connection.prepare("SELECT * FROM dia_ms_ms_windows")?;
   183	        let window_group_settings_iter = stmt.query_map([], |row| {
   184	            Ok(WindowGroupSettingsSim::new(
   185	                row.get("window_group")?,
   186	                row.get("scan_start")?,
   187	                row.get("scan_end")?,
   188	                row.get("isolation_mz")?,
   189	                row.get("isolation_width")?,
   190	                row.get("collision_energy")?,
   191	            ))
   192	        })?;
   193	        let mut window_group_settings = Vec::new();
   194	        for window_group_setting in window_group_settings_iter {
   195	            window_group_settings.push(window_group_setting?);
   196	        }
   197	        Ok(window_group_settings)
   198	    }
   199	
   200	    pub fn read_frame_to_window_group(&self) -> rusqlite::Result<Vec<FrameToWindowGroupSim>> {
   201	        let mut stmt = self.connection.prepare("SELECT * FROM dia_ms_ms_info")?;
   202	        let frame_to_window_group_iter = stmt.query_map([], |row| {
   203	            Ok(FrameToWindowGroupSim::new(
   204	                row.get("frame")?,
   205	                row.get("window_group")?,
   206	            ))
   207	        })?;
   208	
   209	        let mut frame_to_window_groups: Vec<FrameToWindowGroupSim> = Vec::new();
   210	        for frame_to_window_group in frame_to_window_group_iter {
   211	            frame_to_window_groups.push(frame_to_window_group?);
   212	        }
   213	
   214	        Ok(frame_to_window_groups)
   215	    }
   216	
   217	    pub fn read_pasef_meta(&self) -> rusqlite::Result<Vec<PASEFMeta>> {
   218	        let mut stmt = self.connection.prepare(
   219	            "SELECT frame, scan_start, scan_end, isolation_mz, isolation_width, collision_energy, precursor FROM pasef_meta"
   220	        )?;

exec
/bin/bash -lc 'rg -n "def calculate_number_frames|calculate_number_frames" packages/imspy-simulation/src -S; rg -n "to_bruker_info|to_bruker_windows|cycle_length|n_ms2_frames" imspy_connector rustdf -S' in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
packages/imspy-simulation/src/imspy_simulation/acquisition.py:12:from imspy_simulation.utility import calculate_number_frames, get_ms_ms_window_layout_resource_path
packages/imspy-simulation/src/imspy_simulation/acquisition.py:44:        self.num_frames = calculate_number_frames(gradient_length, rt_cycle_length)
packages/imspy-simulation/src/imspy_simulation/acquisition.py:182:            self.num_frames = calculate_number_frames(self.gradient_length, rt_cycle_length)
packages/imspy-simulation/src/imspy_simulation/utility.py:325:def calculate_number_frames(gradient_length: float, rt_cycle_length: float) -> int:
packages/imspy-simulation/src/imspy_simulation/__init__.py:67:    calculate_number_frames,
packages/imspy-simulation/src/imspy_simulation/__init__.py:112:    "calculate_number_frames",
rustdf/src/sim/scheme.rs:445:    pub fn to_bruker_windows(&self) -> Result<Vec<crate::data::meta::DiaMsMsWindow>, String> {
rustdf/src/sim/scheme.rs:489:    /// scheme. This is the single source of truth shared by `to_bruker_windows`
rustdf/src/sim/scheme.rs:490:    /// and `to_bruker_info`, so the two tables always agree.
rustdf/src/sim/scheme.rs:537:    /// same ids `to_bruker_windows` emits, so the two tables reference the same
rustdf/src/sim/scheme.rs:544:    pub fn to_bruker_info(
rustdf/src/sim/scheme.rs:584:        Ok((self.to_bruker_windows()?, self.to_bruker_info(num_frames)?))
rustdf/src/sim/scheme.rs:1133:    // Unconditional: to_bruker_windows must preserve explicit vendor_group_ids
rustdf/src/sim/scheme.rs:1138:    fn to_bruker_windows_group_ids() {
rustdf/src/sim/scheme.rs:1174:            s.to_bruker_windows().map(|r| r.iter().map(|w| w.window_group).collect::<Vec<_>>())
rustdf/src/sim/scheme.rs:1199:        let regenerated = scheme.to_bruker_windows().expect("to_bruker_windows");
rustdf/src/sim/scheme.rs:1243:    fn to_bruker_info_tiling() {
rustdf/src/sim/scheme.rs:1280:            .to_bruker_info(6)
rustdf/src/sim/scheme.rs:1331:            .to_bruker_info(3)
rustdf/src/sim/scheme.rs:1338:        let w3 = s2.to_bruker_windows().unwrap();
rustdf/src/sim/scheme.rs:1356:        let regenerated = scheme.to_bruker_info(num_frames).expect("to_bruker_info");
imspy_connector/src/py_acquisition.rs:117:    pub fn cycle_length(&self) -> usize {
imspy_connector/src/py_acquisition.rs:122:    pub fn n_ms2_frames(&self) -> usize {
imspy_connector/src/py_acquisition.rs:133:    pub fn to_bruker_windows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
imspy_connector/src/py_acquisition.rs:134:        let rows = self.inner.to_bruker_windows().map_err(val_err)?;
imspy_connector/src/py_acquisition.rs:158:    pub fn to_bruker_info<'py>(
imspy_connector/src/py_acquisition.rs:163:        let rows = self.inner.to_bruker_info(num_frames).map_err(val_err)?;
imspy_connector/src/py_utility.rs:43:#[pyo3(signature = (frame_ids, retention_times, frame_occurrences, rt, sigma, lambda_, rt_cycle_length, n_steps=None))]
imspy_connector/src/py_utility.rs:44:pub fn calculate_frame_abundance_emg(frame_ids: Vec<i32>, retention_times: Vec<f64>, frame_occurrences: Vec<i32>, rt: f64, sigma: f64, lambda_: f64, rt_cycle_length: f64, n_steps: Option<usize>) -> Vec<f64> {
imspy_connector/src/py_utility.rs:46:    mscore::algorithm::utility::calculate_frame_abundance_emg(&time_map, &frame_occurrences, rt, sigma, lambda_, rt_cycle_length, n_steps)
imspy_connector/src/py_utility.rs:56:#[pyo3(signature = (frame_ids, retention_times, frame_occurrences, rts, sigmas, lambdas, rt_cycle_length, num_threads, n_steps=None))]
imspy_connector/src/py_utility.rs:57:pub fn calculate_frame_abundances_emg_par(frame_ids: Vec<i32>, retention_times: Vec<f64>, frame_occurrences: Vec<Vec<i32>>, rts: Vec<f64>, sigmas: Vec<f64>, lambdas: Vec<f64>, rt_cycle_length: f64, num_threads: usize, n_steps: Option<usize>) -> Vec<Vec<f64>> {
imspy_connector/src/py_utility.rs:59:    mscore::algorithm::utility::calculate_frame_abundances_emg_par(&time_map, frame_occurrences, rts, sigmas, lambdas, rt_cycle_length, num_threads, n_steps)
imspy_connector/src/py_utility.rs:68:pub fn calculate_abundance_gaussian(indices: Vec<i32>, times: Vec<f64>, occurrences: Vec<i32>, mean: f64, sigma: f64, cycle_length: f64) -> Vec<f64> {
imspy_connector/src/py_utility.rs:70:    mscore::algorithm::utility::calculate_abundance_gaussian(&time_map, &occurrences, mean, sigma, cycle_length)
imspy_connector/src/py_utility.rs:79:pub fn calculate_scan_abundances_gaussian_par(indices: Vec<i32>, times: Vec<f64>, occurrences: Vec<Vec<i32>>, means: Vec<f64>, sigmas: Vec<f64>, cycle_length: f64, num_threads: usize) -> Vec<Vec<f64>> {
imspy_connector/src/py_utility.rs:81:    mscore::algorithm::utility::calculate_scan_abundances_gaussian_par(&time_map, occurrences, means, sigmas, cycle_length, num_threads)

codex
A key distinction is emerging: direct scheme output can be internally correct while still not matching the legacy builder’s positional group formula. I’m tracing whether the extractor preserves vendor group IDs and whether the new test masks order/dtype differences before finalizing severity.
exec
/bin/bash -lc "nl -ba packages/imspy-simulation/src/imspy_simulation/utility.py | sed -n '315,340p'; rg --files | rg 'test_acquisition_scheme_parity\\.py"'$|acquisition.*test|test.*acquisition'"'" in /scratch/timsim-demo/SUBMISSION/rustims
exec
/bin/bash -lc 'git show --stat --oneline ea07bcf1; git show --format=fuller --find-renames ea07bcf1 -- packages/imspy-simulation/src/imspy_simulation/acquisition.py packages/imspy-simulation/tests/test_acquisition_scheme_parity.py tests/test_acquisition_scheme_parity.py' in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
   315	    NDArray
   316	        Array of scaled values.
   317	    """
   318	    min_val = np.min(irt)
   319	    max_val = np.max(irt)
   320	    scaled_values = new_min + (irt - min_val) * (new_max - new_min) / (max_val - min_val)
   321	    return scaled_values
   322	
   323	
   324	@jit(nopython=True)
   325	def calculate_number_frames(gradient_length: float, rt_cycle_length: float) -> int:
   326	    """ Calculate the number of frames that will be taken during the acquisition
   327	
   328	    Parameters
   329	    ----------
   330	    gradient_length : float
   331	        Length of the gradient in seconds
   332	    rt_cycle_length : float
   333	        Length of the RT cycle in seconds
   334	
   335	    Returns
   336	    -------
   337	    int
   338	        Number of frames that will be taken during the acquisition
   339	    """
   340	    return int(gradient_length / rt_cycle_length)
packages/imspy-simulation/tests/test_acquisition_scheme_parity.py

 succeeded in 0ms:
ea07bcf1 feat(imspy-sim): migrate TimsTofAcquisitionBuilderDIA onto AcquisitionScheme
 .../src/imspy_simulation/acquisition.py            | 65 ++++++++++++++++-----
 .../tests/test_acquisition_scheme_parity.py        | 68 ++++++++++++++++++++++
 2 files changed, 118 insertions(+), 15 deletions(-)
commit ea07bcf18478fb4c502061d3f548bd6459dc3a24
Author:     theGreatHerrLebert <davidteschner@googlemail.com>
AuthorDate: Tue Jun 9 10:03:21 2026 +0200
Commit:     theGreatHerrLebert <davidteschner@googlemail.com>
CommitDate: Tue Jun 9 10:03:21 2026 +0200

    feat(imspy-sim): migrate TimsTofAcquisitionBuilderDIA onto AcquisitionScheme
    
    use_reference_ds_layout now sources the DIA window table + frame->group table from
    the vendor-neutral AcquisitionScheme via imspy_connector.py_acquisition
    (PyAcquisitionScheme.from_bruker_d(reference.data_path) -> to_bruker_windows /
    to_bruker_info), with precursor_every = scheme.cycle_length(). Falls back to the
    legacy direct reference read if the connector lacks py_acquisition.
    
    Also fixes a latent bug (flagged in the scheme design review): num_frames was
    computed in the base __init__ from the passed rt_cycle_length BEFORE the
    reference-derived cycle length replaced it, leaving a stale frame count; now
    recomputed.
    
    Golden parity test (TIMSIM_BRUKER_DIA_D): the scheme-derived DiaFrameMsMsWindows
    and DiaFrameMsMsInfo match the reference .d's tables exactly AND the legacy
    position formula. The Bruker layout now flows through the shared Rust scheme.

diff --git a/packages/imspy-simulation/src/imspy_simulation/acquisition.py b/packages/imspy-simulation/src/imspy_simulation/acquisition.py
index 1c9948cd..2e840c18 100644
--- a/packages/imspy-simulation/src/imspy_simulation/acquisition.py
+++ b/packages/imspy-simulation/src/imspy_simulation/acquisition.py
@@ -177,6 +177,9 @@ class TimsTofAcquisitionBuilderDIA(TimsTofAcquisitionBuilder, ABC):
             if verbose:
                 print('Using reference dataset cycle length:', np.round(rt_cycle_length, 4))
             self.rt_cycle_length = rt_cycle_length
+            # Recompute the frame count: the base __init__ computed it from the
+            # passed rt_cycle_length before this reference-derived value replaced it.
+            self.num_frames = calculate_number_frames(self.gradient_length, rt_cycle_length)
 
         self.acquisition_name = acquisition_name
         self.scan_table = None
@@ -214,25 +217,57 @@ class TimsTofAcquisitionBuilderDIA(TimsTofAcquisitionBuilder, ABC):
 
         return pd.DataFrame(table_list)
 
+    def _layout_from_scheme(self, verbose: bool = True):
+        """Derive the DIA window + frame→group tables from the reference `.d` via
+        the vendor-neutral ``AcquisitionScheme`` (imspy_connector.py_acquisition).
+
+        Returns ``(dia_ms_ms_windows, frames_to_window_groups, precursor_every)``
+        or ``None`` if the connector lacks the scheme module (legacy fallback).
+        """
+        try:
+            import imspy_connector
+            acq = imspy_connector.py_acquisition
+        except (ImportError, AttributeError):
+            if verbose:
+                print('py_acquisition unavailable; using legacy reference layout')
+            return None
+
+        scheme = acq.PyAcquisitionScheme.from_bruker_d(self.reference.data_path)
+        precursor_every = scheme.cycle_length()
+        windows = pd.DataFrame(scheme.to_bruker_windows())
+        info = pd.DataFrame(scheme.to_bruker_info(int(self.num_frames)))
+        if verbose:
+            print(f'Using AcquisitionScheme layout: {len(windows)} windows, '
+                  f'{scheme.n_ms2_frames()} groups, precursor_every={precursor_every}')
+        return windows, info, precursor_every
+
     def _setup(self, verbose: bool = True):
         self.frame_table = self.generate_frame_table(verbose=verbose)
         self.scan_table = self.generate_scan_table(verbose=verbose)
 
-        if self.use_reference_ds_layout:
-            self.precursor_every = int(np.diff(self.reference.precursor_frames)[0])
-            self.dia_ms_ms_windows = self.reference.dia_ms_ms_windows.rename(
-                columns={
-                    'WindowGroup': 'window_group',
-                    'ScanNumBegin': 'scan_start',
-                    'ScanNumEnd': 'scan_end',
-                    'IsolationMz': 'isolation_mz',
-                    'IsolationWidth': 'isolation_width',
-                    'CollisionEnergy': 'collision_energy',
-                }
-            )
-
-        self.frame_table['ms_type'] = self.calculate_frame_types(verbose=verbose)
-        self.frames_to_window_groups = self.generate_frame_to_window_group_table(verbose=verbose)
+        scheme_layout = self._layout_from_scheme(verbose=verbose) if self.use_reference_ds_layout else None
+
+        if scheme_layout is not None:
+            # Vendor-neutral path: windows + frame→group come from the scheme.
+            self.dia_ms_ms_windows, frames_to_window_groups, self.precursor_every = scheme_layout
+            self.frame_table['ms_type'] = self.calculate_frame_types(verbose=verbose)
+            self.frames_to_window_groups = frames_to_window_groups
+        else:
+            # Legacy path: copy the reference .d's window table directly.
+            if self.use_reference_ds_layout:
+                self.precursor_every = int(np.diff(self.reference.precursor_frames)[0])
+                self.dia_ms_ms_windows = self.reference.dia_ms_ms_windows.rename(
+                    columns={
+                        'WindowGroup': 'window_group',
+                        'ScanNumBegin': 'scan_start',
+                        'ScanNumEnd': 'scan_end',
+                        'IsolationMz': 'isolation_mz',
+                        'IsolationWidth': 'isolation_width',
+                        'CollisionEnergy': 'collision_energy',
+                    }
+                )
+            self.frame_table['ms_type'] = self.calculate_frame_types(verbose=verbose)
+            self.frames_to_window_groups = self.generate_frame_to_window_group_table(verbose=verbose)
 
         if self.round_collision_energy:
             self.dia_ms_ms_windows['collision_energy'] = np.round(self.dia_ms_ms_windows['collision_energy'].values,
diff --git a/packages/imspy-simulation/tests/test_acquisition_scheme_parity.py b/packages/imspy-simulation/tests/test_acquisition_scheme_parity.py
new file mode 100644
index 00000000..98457c75
--- /dev/null
+++ b/packages/imspy-simulation/tests/test_acquisition_scheme_parity.py
@@ -0,0 +1,68 @@
+"""Golden parity: the vendor-neutral AcquisitionScheme (via imspy_connector)
+must reproduce the Bruker DIA layout the legacy `use_reference_ds_layout` path
+copies from the reference `.d` — both the `DiaFrameMsMsWindows` window table and
+the per-frame `DiaFrameMsMsInfo` (frame→group) table.
+
+Gated on TIMSIM_BRUKER_DIA_D pointing at a real DIA-PASEF `.d`.
+"""
+import os
+import sqlite3
+
+import numpy as np
+import pandas as pd
+import pytest
+
+import imspy_connector
+
+acq = imspy_connector.py_acquisition
+DIA_D = os.environ.get("TIMSIM_BRUKER_DIA_D")
+pytestmark = pytest.mark.skipif(
+    not DIA_D, reason="set TIMSIM_BRUKER_DIA_D to a real DIA-PASEF .d folder"
+)
+
+
+def _norm(df, cols):
+    return df[cols].astype(float).round(6).sort_values(cols).reset_index(drop=True)
+
+
+def test_scheme_windows_match_reference():
+    scheme = acq.PyAcquisitionScheme.from_bruker_d(DIA_D)
+    windows = pd.DataFrame(scheme.to_bruker_windows())
+
+    con = sqlite3.connect(DIA_D + "/analysis.tdf")
+    ref = pd.read_sql(
+        "SELECT WindowGroup window_group, ScanNumBegin scan_start, ScanNumEnd scan_end, "
+        "IsolationMz isolation_mz, IsolationWidth isolation_width, "
+        "CollisionEnergy collision_energy FROM DiaFrameMsMsWindows",
+        con,
+    )
+    cols = list(ref.columns)
+    assert len(windows) == len(ref)
+    pd.testing.assert_frame_equal(_norm(windows, cols), _norm(ref, cols))
+
+
+def test_scheme_info_matches_reference_and_legacy_formula():
+    scheme = acq.PyAcquisitionScheme.from_bruker_d(DIA_D)
+    con = sqlite3.connect(DIA_D + "/analysis.tdf")
+    num_frames = int(con.execute("SELECT MAX(Id) FROM Frames").fetchone()[0])
+
+    info = pd.DataFrame(scheme.to_bruker_info(num_frames))
+
+    # (a) matches the .d's own DiaFrameMsMsInfo exactly
+    ref = pd.read_sql("SELECT Frame frame, WindowGroup window_group FROM DiaFrameMsMsInfo", con)
+    pd.testing.assert_frame_equal(
+        _norm(info, ["frame", "window_group"]), _norm(ref, ["frame", "window_group"])
+    )
+
+    # (b) matches the legacy position formula (index % precursor_every, ms2 only)
+    pe = scheme.cycle_length()
+    legacy = pd.DataFrame(
+        [
+            {"frame": f, "window_group": (f - 1) % pe}
+            for f in range(1, num_frames + 1)
+            if (f - 1) % pe != 0
+        ]
+    )
+    pd.testing.assert_frame_equal(
+        _norm(info, ["frame", "window_group"]), _norm(legacy, ["frame", "window_group"])
+    )

exec
/bin/bash -lc "nl -ba rustdf/src/sim/scheme.rs | sed -n '430,570p;590,670p;1120,1215p;1230,1375p'; nl -ba imspy_connector/src/py_acquisition.rs | sed -n '105,180p'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
   430	impl AcquisitionScheme {
   431	    /// Render the scheme's MS2 windows as Bruker `DiaFrameMsMsWindows` rows — the
   432	    /// backward-compatibility adapter for the existing timsTOF write path.
   433	    ///
   434	    /// Each MS2 frame is one window group; its windows must carry `TimsMobility`
   435	    /// geometry (the Bruker-grid scan ranges). The `WindowGroup` id is the frame's
   436	    /// preserved [`DiaMs2Frame::vendor_group_id`] when present (so a Bruker
   437	    /// round-trip keeps the ids that `DiaFrameMsMsInfo` references), else a 1..N
   438	    /// fallback in cycle order. A `Linear` CE policy is resolved at the window
   439	    /// center (a lossy materialization, not an inverse); `Unknown` CE and
   440	    /// non-timsTOF schemes are rejected. The scheme is validated first.
   441	    ///
   442	    /// Row order is not meaningful (the SQLite table has none); the companion
   443	    /// frame→group table `DiaFrameMsMsInfo` needs the run frame schedule and is a
   444	    /// separate step.
   445	    pub fn to_bruker_windows(&self) -> Result<Vec<crate::data::meta::DiaMsMsWindow>, String> {
   446	        let layout = self.bruker_group_layout()?;
   447	        let mut rows = Vec::new();
   448	        for (ev, slot) in self.cycle.iter().zip(&layout) {
   449	            if let (AcquisitionEvent::DiaMs2Frame(frame), Some(group)) = (ev, slot) {
   450	                for w in &frame.windows {
   451	                    let (scan_num_begin, scan_num_end) = match w.geometry {
   452	                        DiaGeometry::TimsMobility {
   453	                            scan_start,
   454	                            scan_end,
   455	                        } => (scan_start, scan_end),
   456	                        DiaGeometry::MzOnly => {
   457	                            return Err("timsTOF window lacks mobility geometry".into())
   458	                        }
   459	                    };
   460	                    let collision_energy = match w.collision_energy {
   461	                        CollisionEnergyPolicy::Value(v) => v,
   462	                        CollisionEnergyPolicy::Linear { .. } => w
   463	                            .collision_energy
   464	                            .at(w.isolation.center_mz)
   465	                            .ok_or("could not resolve linear CE")?,
   466	                        CollisionEnergyPolicy::Unknown => {
   467	                            return Err("window has unknown collision energy".into())
   468	                        }
   469	                    };
   470	                    rows.push(crate::data::meta::DiaMsMsWindow {
   471	                        window_group: *group,
   472	                        scan_num_begin,
   473	                        scan_num_end,
   474	                        isolation_mz: w.isolation.center_mz,
   475	                        isolation_width: w.isolation.width_mz,
   476	                        collision_energy,
   477	                    });
   478	                }
   479	            }
   480	        }
   481	        Ok(rows)
   482	    }
   483	
   484	    /// Per-cycle-position window-group id: `None` for an MS1 event, `Some(group)`
   485	    /// for an MS2 frame. The group id is the frame's preserved `vendor_group_id`
   486	    /// (Bruker `WindowGroup`) when set, else a collision-safe positional id drawn
   487	    /// from the unused set (so preserved and fallback ids never collide).
   488	    /// Duplicate explicit ids are rejected. Validates first; requires a timsTOF
   489	    /// scheme. This is the single source of truth shared by `to_bruker_windows`
   490	    /// and `to_bruker_info`, so the two tables always agree.
   491	    fn bruker_group_layout(&self) -> Result<Vec<Option<u32>>, String> {
   492	        self.validate()?;
   493	        if self.instrument != InstrumentKind::TimsTofDia {
   494	            return Err("Bruker tables require a timsTOF (TimsTofDia) scheme".into());
   495	        }
   496	        use std::collections::HashSet;
   497	        let mut reserved: HashSet<u32> = HashSet::new();
   498	        for ev in &self.cycle {
   499	            if let AcquisitionEvent::DiaMs2Frame(f) = ev {
   500	                if let Some(g) = f.vendor_group_id {
   501	                    if !reserved.insert(g) {
   502	                        return Err(format!("duplicate vendor window-group id {g}"));
   503	                    }
   504	                }
   505	            }
   506	        }
   507	        let mut layout = Vec::with_capacity(self.cycle.len());
   508	        let mut assigned: HashSet<u32> = HashSet::new();
   509	        let mut next_seq: u32 = 1;
   510	        for ev in &self.cycle {
   511	            match ev {
   512	                AcquisitionEvent::Ms1(_) => layout.push(None),
   513	                AcquisitionEvent::DiaMs2Frame(frame) => {
   514	                    let group = match frame.vendor_group_id {
   515	                        Some(g) => g,
   516	                        None => {
   517	                            while reserved.contains(&next_seq) || assigned.contains(&next_seq) {
   518	                                next_seq += 1;
   519	                            }
   520	                            next_seq
   521	                        }
   522	                    };
   523	                    if !assigned.insert(group) {
   524	                        return Err(format!("window-group id {group} collides"));
   525	                    }
   526	                    layout.push(Some(group));
   527	                }
   528	            }
   529	        }
   530	        Ok(layout)
   531	    }
   532	
   533	    /// Generate the Bruker `DiaFrameMsMsInfo` (frame → window group) rows for a run
   534	    /// of `num_frames` total frames, tiling the scheme's cycle. Cycle position
   535	    /// `(frame_id - 1) % cycle_len` selects the event (1-based frame ids), MS1
   536	    /// frames produce no row, and each MS2 frame maps to its window-group id (the
   537	    /// same ids `to_bruker_windows` emits, so the two tables reference the same
   538	    /// groups). The final cycle may be partial.
   539	    ///
   540	    /// Precondition: this models a **clean generated run** — frame 1 is the
   541	    /// cycle's leading MS1 and frame ids are contiguous (as TimSim produces). It
   542	    /// is not a reproducer for arbitrary real files with prefix/calibration
   543	    /// frames, gaps, or acquisition starting mid-cycle.
   544	    pub fn to_bruker_info(
   545	        &self,
   546	        num_frames: u32,
   547	    ) -> Result<Vec<crate::data::meta::DiaMsMisInfo>, String> {
   548	        // Bound the work so an absurd frame count errors cleanly instead of OOM.
   549	        const MAX_FRAMES: u32 = 100_000_000;
   550	        if num_frames > MAX_FRAMES {
   551	            return Err(format!("num_frames {num_frames} exceeds the {MAX_FRAMES} limit"));
   552	        }
   553	        let layout = self.bruker_group_layout()?;
   554	        let fpc = layout.len() as u32;
   555	        if fpc == 0 {
   556	            return Err("empty cycle".into());
   557	        }
   558	        let mut rows = Vec::new();
   559	        for frame_id in 1..=num_frames {
   560	            let pos = ((frame_id - 1) % fpc) as usize;
   561	            if let Some(group) = layout[pos] {
   562	                rows.push(crate::data::meta::DiaMsMisInfo {
   563	                    frame_id,
   564	                    window_group: group,
   565	                });
   566	            }
   567	        }
   568	        Ok(rows)
   569	    }
   570	
   590	    /// becomes one [`DiaMs2Frame`] holding its mobility-partitioned windows
   591	    /// (`TimsMobility` geometry — the scan ranges are Bruker-grid coordinates),
   592	    /// preceded by a single MS1 event. Cycle timing comes from the spacing of
   593	    /// the precursor (MS1) frames. The returned scheme is validated.
   594	    ///
   595	    /// Window groups (frames) are ordered by their **first occurrence in
   596	    /// `DiaFrameMsMsInfo`** (ascending frame id), i.e. the real acquisition order
   597	    /// within a cycle — not merely ascending `WindowGroup` id — so the cycle
   598	    /// faithfully represents permuted/reused group numbering. (If the info table
   599	    /// is absent, falls back to ascending group id.)
   600	    pub fn from_bruker_d<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
   601	        use crate::data::meta::{read_dia_ms_ms_info, read_dia_ms_ms_windows, read_meta_data_sql};
   602	        use std::collections::BTreeMap;
   603	
   604	        let folder = path.as_ref().to_string_lossy().into_owned();
   605	        let to_io =
   606	            |e: Box<dyn std::error::Error>| io::Error::new(io::ErrorKind::InvalidData, e.to_string());
   607	        let windows = read_dia_ms_ms_windows(&folder).map_err(to_io)?;
   608	        let frames = read_meta_data_sql(&folder).map_err(to_io)?;
   609	        if windows.is_empty() {
   610	            return Err(io::Error::new(
   611	                io::ErrorKind::InvalidData,
   612	                "no DiaFrameMsMsWindows rows (not a DIA .d?)",
   613	            ));
   614	        }
   615	
   616	        // Group windows by window group (ascending); each group is a frame.
   617	        let mut by_group: BTreeMap<u32, Vec<DiaWindow>> = BTreeMap::new();
   618	        let mut lo = f64::INFINITY;
   619	        let mut hi = f64::NEG_INFINITY;
   620	        for w in &windows {
   621	            let dw = DiaWindow {
   622	                isolation: IsolationWindow {
   623	                    center_mz: w.isolation_mz,
   624	                    width_mz: w.isolation_width,
   625	                },
   626	                collision_energy: CollisionEnergyPolicy::Value(w.collision_energy),
   627	                geometry: DiaGeometry::TimsMobility {
   628	                    scan_start: w.scan_num_begin,
   629	                    scan_end: w.scan_num_end,
   630	                },
   631	            };
   632	            lo = lo.min(dw.isolation.lower());
   633	            hi = hi.max(dw.isolation.upper());
   634	            by_group.entry(w.window_group).or_default().push(dw);
   635	        }
   636	
   637	        // Order groups by first occurrence (ascending frame id) in DiaFrameMsMsInfo
   638	        // = real intra-cycle acquisition order. Missing info -> ascending group id.
   639	        let info = read_dia_ms_ms_info(&folder).unwrap_or_default();
   640	        let mut first_frame: BTreeMap<u32, u32> = BTreeMap::new();
   641	        for r in &info {
   642	            first_frame
   643	                .entry(r.window_group)
   644	                .and_modify(|f| {
   645	                    if r.frame_id < *f {
   646	                        *f = r.frame_id
   647	                    }
   648	                })
   649	                .or_insert(r.frame_id);
   650	        }
   651	        let mut ordered_groups: Vec<u32> = by_group.keys().copied().collect();
   652	        // Stable sort by first-occurrence frame; groups absent from the info table
   653	        // (key u32::MAX) keep their ascending-id relative order.
   654	        ordered_groups.sort_by_key(|g| first_frame.get(g).copied().unwrap_or(u32::MAX));
   655	
   656	        let mut cycle = vec![AcquisitionEvent::Ms1(Ms1Event {
   657	            analyzer: Analyzer::Tof,
   658	            data_mode: DataMode::Centroid,
   659	            mz_range: None,
   660	            duration_s: None,
   661	        })];
   662	        let n_groups = ordered_groups.len();
   663	        for group in ordered_groups {
   664	            let ws = by_group.remove(&group).expect("group present");
   665	            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
   666	                windows: ws,
   667	                analyzer: Analyzer::Tof,
   668	                data_mode: DataMode::Centroid,
   669	                duration_s: None,
   670	                vendor_group_id: Some(group), // preserve Bruker WindowGroup identity
  1120	            assert!(w.collision_energy.at(w.isolation.center_mz).is_some());
  1121	        }
  1122	        // At least one frame should be mobility-partitioned (>1 window).
  1123	        let multi = s.cycle.iter().any(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(f) if f.windows.len() > 1));
  1124	        let n_frames = s.cycle.iter().filter(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_))).count();
  1125	        eprintln!(
  1126	            "from_bruker_d OK: TimsTofDia, {} frames, {} windows ({}), mz {:.1}..{:.1}",
  1127	            n_frames, n_win, if multi { "mobility-partitioned" } else { "single-window" },
  1128	            s.mz_range.0, s.mz_range.1
  1129	        );
  1130	    }
  1131	
  1132	    // Gated: set TIMSIM_SCIEX_WIFF to a real ZenoTOF .wiff (OLE2 method).
  1133	    // Unconditional: to_bruker_windows must preserve explicit vendor_group_ids
  1134	    // (incl. non-canonical), allocate non-colliding ids for None frames, and
  1135	    // reject duplicate explicit ids. (The gated round-trip uses a real .d whose
  1136	    // ids are already 1..N, so it can't prove non-canonical preservation alone.)
  1137	    #[test]
  1138	    fn to_bruker_windows_group_ids() {
  1139	        let frame = |gid: Option<u32>, center: f64| {
  1140	            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
  1141	                windows: vec![DiaWindow {
  1142	                    isolation: IsolationWindow { center_mz: center, width_mz: 10.0 },
  1143	                    collision_energy: CollisionEnergyPolicy::Value(25.0),
  1144	                    geometry: DiaGeometry::TimsMobility { scan_start: 0, scan_end: 100 },
  1145	                }],
  1146	                analyzer: Analyzer::Tof,
  1147	                data_mode: DataMode::Centroid,
  1148	                duration_s: None,
  1149	                vendor_group_id: gid,
  1150	            })
  1151	        };
  1152	        let mk = |frames: Vec<AcquisitionEvent>| {
  1153	            let mut cycle = vec![AcquisitionEvent::Ms1(Ms1Event {
  1154	                analyzer: Analyzer::Tof,
  1155	                data_mode: DataMode::Centroid,
  1156	                mz_range: None,
  1157	                duration_s: None,
  1158	            })];
  1159	            cycle.extend(frames);
  1160	            AcquisitionScheme {
  1161	                version: SCHEME_VERSION,
  1162	                instrument: InstrumentKind::TimsTofDia,
  1163	                cycle,
  1164	                repeat: RepeatPolicy::FixedCycleTime {
  1165	                    cycle_time_s: 1.0,
  1166	                    gradient_length_s: 600.0,
  1167	                    start_time_s: 0.0,
  1168	                },
  1169	                mz_range: (300.0, 900.0),
  1170	                provenance: Provenance { source: SchemeSource::Programmatic, notes: String::new() },
  1171	            }
  1172	        };
  1173	        let ids = |s: &AcquisitionScheme| {
  1174	            s.to_bruker_windows().map(|r| r.iter().map(|w| w.window_group).collect::<Vec<_>>())
  1175	        };
  1176	
  1177	        // preserved non-canonical ids
  1178	        assert_eq!(ids(&mk(vec![frame(Some(7), 500.0), frame(Some(42), 600.0)])).unwrap(), vec![7, 42]);
  1179	        // all-None -> sequential 1..N
  1180	        assert_eq!(ids(&mk(vec![frame(None, 500.0), frame(None, 600.0)])).unwrap(), vec![1, 2]);
  1181	        // mixed: None allocates a free id (1), avoiding the reserved 7
  1182	        assert_eq!(ids(&mk(vec![frame(Some(7), 500.0), frame(None, 600.0)])).unwrap(), vec![7, 1]);
  1183	        // duplicate explicit ids rejected
  1184	        assert!(ids(&mk(vec![frame(Some(5), 500.0), frame(Some(5), 600.0)])).is_err());
  1185	    }
  1186	
  1187	    // Gated: round-trip a real Bruker DIA .d through the scheme and back to the
  1188	    // DiaFrameMsMsWindows rows; the regenerated table must match the source.
  1189	    #[test]
  1190	    fn bruker_windows_round_trip() {
  1191	        let d = match std::env::var("TIMSIM_BRUKER_DIA_D") {
  1192	            Ok(p) => p,
  1193	            Err(_) => {
  1194	                eprintln!("SKIP bruker_windows_round_trip: set TIMSIM_BRUKER_DIA_D");
  1195	                return;
  1196	            }
  1197	        };
  1198	        let scheme = AcquisitionScheme::from_bruker_d(&d).expect("extract");
  1199	        let regenerated = scheme.to_bruker_windows().expect("to_bruker_windows");
  1200	        let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
  1201	        assert_eq!(regenerated.len(), original.len(), "row count differs");
  1202	
  1203	        // Compare as a sorted multiset of EXACT field tuples, including the
  1204	        // WindowGroup id (preserved via vendor_group_id) — no normalization, so a
  1205	        // real id mismatch would fail. Row order in the table is not guaranteed.
  1206	        fn tuples(
  1207	            rows: &[crate::data::meta::DiaMsMsWindow],
  1208	        ) -> Vec<(u32, u32, u32, u64, u64, u64)> {
  1209	            let mut out: Vec<_> = rows
  1210	                .iter()
  1211	                .map(|r| {
  1212	                    (
  1213	                        r.window_group,
  1214	                        r.scan_num_begin,
  1215	                        r.scan_num_end,
  1230	        eprintln!(
  1231	            "bruker_windows_round_trip OK: {} rows match across {} groups",
  1232	            original.len(),
  1233	            scheme
  1234	                .cycle
  1235	                .iter()
  1236	                .filter(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_)))
  1237	                .count()
  1238	        );
  1239	    }
  1240	
  1241	    // Unconditional: DiaFrameMsMsInfo tiling + windows/info group-id consistency.
  1242	    #[test]
  1243	    fn to_bruker_info_tiling() {
  1244	        let frame = |c: f64| {
  1245	            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
  1246	                windows: vec![DiaWindow {
  1247	                    isolation: IsolationWindow { center_mz: c, width_mz: 10.0 },
  1248	                    collision_energy: CollisionEnergyPolicy::Value(25.0),
  1249	                    geometry: DiaGeometry::TimsMobility { scan_start: 0, scan_end: 100 },
  1250	                }],
  1251	                analyzer: Analyzer::Tof,
  1252	                data_mode: DataMode::Centroid,
  1253	                duration_s: None,
  1254	                vendor_group_id: None,
  1255	            })
  1256	        };
  1257	        let s = AcquisitionScheme {
  1258	            version: SCHEME_VERSION,
  1259	            instrument: InstrumentKind::TimsTofDia,
  1260	            cycle: vec![
  1261	                AcquisitionEvent::Ms1(Ms1Event {
  1262	                    analyzer: Analyzer::Tof,
  1263	                    data_mode: DataMode::Centroid,
  1264	                    mz_range: None,
  1265	                    duration_s: None,
  1266	                }),
  1267	                frame(500.0),
  1268	                frame(600.0),
  1269	            ],
  1270	            repeat: RepeatPolicy::FixedCycleTime {
  1271	                cycle_time_s: 1.0,
  1272	                gradient_length_s: 600.0,
  1273	                start_time_s: 0.0,
  1274	            },
  1275	            mz_range: (300.0, 900.0),
  1276	            provenance: Provenance { source: SchemeSource::Programmatic, notes: String::new() },
  1277	        };
  1278	        // cycle = [MS1, g1, g2]; num_frames=6: 1->skip, 2->g1, 3->g2, 4->skip, 5->g1, 6->g2
  1279	        let info: Vec<(u32, u32)> = s
  1280	            .to_bruker_info(6)
  1281	            .unwrap()
  1282	            .iter()
  1283	            .map(|r| (r.frame_id, r.window_group))
  1284	            .collect();
  1285	        assert_eq!(info, vec![(2, 1), (3, 2), (5, 1), (6, 2)]);
  1286	        // The two tables must reference the same set of window-group ids.
  1287	        let (windows, info2) = s.to_bruker_tables(6).unwrap();
  1288	        let wg: std::collections::BTreeSet<u32> = windows.iter().map(|w| w.window_group).collect();
  1289	        let ig: std::collections::BTreeSet<u32> = info2.iter().map(|r| r.window_group).collect();
  1290	        assert_eq!(wg, ig, "windows/info group ids disagree");
  1291	
  1292	        // Non-ascending explicit ids + a multi-window (mobility-partitioned) frame:
  1293	        // info must follow CYCLE order (not sorted id) and emit ONE row per frame.
  1294	        let win = |center: f64, s0: u32, s1: u32| DiaWindow {
  1295	            isolation: IsolationWindow { center_mz: center, width_mz: 10.0 },
  1296	            collision_energy: CollisionEnergyPolicy::Value(20.0),
  1297	            geometry: DiaGeometry::TimsMobility { scan_start: s0, scan_end: s1 },
  1298	        };
  1299	        let dia = |gid: u32, ws: Vec<DiaWindow>| {
  1300	            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
  1301	                windows: ws,
  1302	                analyzer: Analyzer::Tof,
  1303	                data_mode: DataMode::Centroid,
  1304	                duration_s: None,
  1305	                vendor_group_id: Some(gid),
  1306	            })
  1307	        };
  1308	        let s2 = AcquisitionScheme {
  1309	            version: SCHEME_VERSION,
  1310	            instrument: InstrumentKind::TimsTofDia,
  1311	            cycle: vec![
  1312	                AcquisitionEvent::Ms1(Ms1Event {
  1313	                    analyzer: Analyzer::Tof,
  1314	                    data_mode: DataMode::Centroid,
  1315	                    mz_range: None,
  1316	                    duration_s: None,
  1317	                }),
  1318	                dia(7, vec![win(500.0, 0, 50), win(500.0, 51, 100)]), // 2 mobility windows
  1319	                dia(2, vec![win(600.0, 0, 100)]),
  1320	            ],
  1321	            repeat: RepeatPolicy::FixedCycleTime {
  1322	                cycle_time_s: 1.0,
  1323	                gradient_length_s: 600.0,
  1324	                start_time_s: 0.0,
  1325	            },
  1326	            mz_range: (300.0, 900.0),
  1327	            provenance: Provenance { source: SchemeSource::Programmatic, notes: String::new() },
  1328	        };
  1329	        // cycle len 3; frame2->g7 (first), frame3->g2 — acquisition order, NOT sorted.
  1330	        let info3: Vec<(u32, u32)> = s2
  1331	            .to_bruker_info(3)
  1332	            .unwrap()
  1333	            .iter()
  1334	            .map(|r| (r.frame_id, r.window_group))
  1335	            .collect();
  1336	        assert_eq!(info3, vec![(2, 7), (3, 2)], "info must follow cycle order, one row/frame");
  1337	        // windows: the 2-window frame emits 2 rows (group 7), the other 1 (group 2).
  1338	        let w3 = s2.to_bruker_windows().unwrap();
  1339	        assert_eq!(w3.iter().filter(|w| w.window_group == 7).count(), 2);
  1340	        assert_eq!(w3.iter().filter(|w| w.window_group == 2).count(), 1);
  1341	    }
  1342	
  1343	    // Gated: regenerate DiaFrameMsMsInfo for the .d's full frame count and match.
  1344	    #[test]
  1345	    fn bruker_info_round_trip() {
  1346	        let d = match std::env::var("TIMSIM_BRUKER_DIA_D") {
  1347	            Ok(p) => p,
  1348	            Err(_) => {
  1349	                eprintln!("SKIP bruker_info_round_trip: set TIMSIM_BRUKER_DIA_D");
  1350	                return;
  1351	            }
  1352	        };
  1353	        let scheme = AcquisitionScheme::from_bruker_d(&d).expect("extract");
  1354	        let frames = crate::data::meta::read_meta_data_sql(&d).expect("frames");
  1355	        let num_frames = frames.iter().map(|f| f.id).max().unwrap_or(0) as u32;
  1356	        let regenerated = scheme.to_bruker_info(num_frames).expect("to_bruker_info");
  1357	        let original = crate::data::meta::read_dia_ms_ms_info(&d).expect("read source");
  1358	        let key = |r: &crate::data::meta::DiaMsMisInfo| (r.frame_id, r.window_group);
  1359	        let mut a: Vec<_> = regenerated.iter().map(key).collect();
  1360	        let mut b: Vec<_> = original.iter().map(key).collect();
  1361	        a.sort_unstable();
  1362	        b.sort_unstable();
  1363	        assert_eq!(a, b, "DiaFrameMsMsInfo round-trip differs from source");
  1364	        eprintln!("bruker_info_round_trip OK: {} (frame, group) rows over {num_frames} frames", a.len());
  1365	    }
  1366	
  1367	    #[cfg(feature = "sciex")]
  1368	    #[test]
  1369	    fn from_sciex_wiff_extracts_windows() {
  1370	        let wiff = match std::env::var("TIMSIM_SCIEX_WIFF") {
  1371	            Ok(p) => p,
  1372	            Err(_) => {
  1373	                eprintln!("SKIP from_sciex_wiff_extracts_windows: set TIMSIM_SCIEX_WIFF=<.wiff>");
  1374	                return;
  1375	            }
   105	    }
   106	
   107	    pub fn n_windows(&self) -> usize {
   108	        self.inner.windows().count()
   109	    }
   110	
   111	    pub fn num_cycles(&self) -> Option<u64> {
   112	        self.inner.num_cycles()
   113	    }
   114	
   115	    /// Number of events in one cycle (1 MS1 + N MS2 frames) — the Bruker
   116	    /// `precursor_every` (frames per cycle).
   117	    pub fn cycle_length(&self) -> usize {
   118	        self.inner.cycle.len()
   119	    }
   120	
   121	    /// Number of MS2 frames (window groups) per cycle.
   122	    pub fn n_ms2_frames(&self) -> usize {
   123	        self.inner.cycle.len() - self.inner.ms1_count()
   124	    }
   125	
   126	    pub fn validate(&self) -> PyResult<()> {
   127	        self.inner.validate().map_err(val_err)
   128	    }
   129	
   130	    /// Bruker `DiaFrameMsMsWindows` as a column dict of NumPy arrays, ready for
   131	    /// `pandas.DataFrame(...)`: keys `window_group, scan_start, scan_end,
   132	    /// isolation_mz, isolation_width, collision_energy`.
   133	    pub fn to_bruker_windows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
   134	        let rows = self.inner.to_bruker_windows().map_err(val_err)?;
   135	        let n = rows.len();
   136	        let (mut wg, mut ss, mut se) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
   137	        let (mut mz, mut wd, mut ce) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
   138	        for r in &rows {
   139	            wg.push(r.window_group);
   140	            ss.push(r.scan_num_begin);
   141	            se.push(r.scan_num_end);
   142	            mz.push(r.isolation_mz);
   143	            wd.push(r.isolation_width);
   144	            ce.push(r.collision_energy);
   145	        }
   146	        let d = PyDict::new(py);
   147	        d.set_item("window_group", wg.into_pyarray(py))?;
   148	        d.set_item("scan_start", ss.into_pyarray(py))?;
   149	        d.set_item("scan_end", se.into_pyarray(py))?;
   150	        d.set_item("isolation_mz", mz.into_pyarray(py))?;
   151	        d.set_item("isolation_width", wd.into_pyarray(py))?;
   152	        d.set_item("collision_energy", ce.into_pyarray(py))?;
   153	        Ok(d)
   154	    }
   155	
   156	    /// Bruker `DiaFrameMsMsInfo` as a column dict of NumPy arrays for a
   157	    /// `num_frames`-frame run: keys `frame, window_group`.
   158	    pub fn to_bruker_info<'py>(
   159	        &self,
   160	        py: Python<'py>,
   161	        num_frames: u32,
   162	    ) -> PyResult<Bound<'py, PyDict>> {
   163	        let rows = self.inner.to_bruker_info(num_frames).map_err(val_err)?;
   164	        let mut frame = Vec::with_capacity(rows.len());
   165	        let mut wg = Vec::with_capacity(rows.len());
   166	        for r in &rows {
   167	            frame.push(r.frame_id);
   168	            wg.push(r.window_group);
   169	        }
   170	        let d = PyDict::new(py);
   171	        d.set_item("frame", frame.into_pyarray(py))?;
   172	        d.set_item("window_group", wg.into_pyarray(py))?;
   173	        Ok(d)
   174	    }
   175	}
   176	
   177	/// Whether this build includes the Thermo `.raw` extractor.
   178	#[pyfunction]
   179	pub fn has_thermo() -> bool {
   180	    cfg!(feature = "thermo")

exec
/bin/bash -lc "nl -ba rustdf/src/sim/types.rs 2>/dev/null | sed -n '1,160p'; rg -n \"struct (WindowGroupSettingsSim|FrameToWindowGroupSim)|impl .*WindowGroupSettingsSim|window_group:\" rustdf/src/sim -S" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
rustdf/src/sim/containers.rs:126:pub struct WindowGroupSettingsSim {
rustdf/src/sim/containers.rs:127:    pub window_group: u32,
rustdf/src/sim/containers.rs:135:impl WindowGroupSettingsSim {
rustdf/src/sim/containers.rs:137:        window_group: u32,
rustdf/src/sim/containers.rs:156:pub struct FrameToWindowGroupSim {
rustdf/src/sim/containers.rs:158:    pub window_group: u32,
rustdf/src/sim/containers.rs:162:    pub fn new(frame_id: u32, window_group: u32) -> Self {
rustdf/src/sim/scheme.rs:471:                        window_group: *group,
rustdf/src/sim/scheme.rs:564:                    window_group: group,

codex
The frame-count recompute occurs before `_setup`, so `frame_table`, `scan_table`, and scheme info are generated from the updated value; I see no stale generated table. I’m doing one focused serialization check because NumPy unsigned integer handling differs by width in pandas/SQLite and should not be guessed.
exec
/bin/bash -lc "python - <<'PY'
import sqlite3, pandas as pd, numpy as np
for dt in [np.uint32, np.int64]:
    con=sqlite3.connect(':memory:')
    df=pd.DataFrame({'frame':np.array([2,3],dtype=dt),'window_group':np.array([1,2],dtype=dt)})
    df.to_sql('x',con,index=False)
    print(dt.__name__, con.execute('pragma table_info(x)').fetchall(), pd.read_sql('select * from x',con).dtypes.to_dict())
PY" in /scratch/timsim-demo/SUBMISSION/rustims
 exited 127 in 0ms:
/bin/bash: line 1: python: command not found

exec
/bin/bash -lc "rg -n \"frames_to_window_groups|dia_ms_ms_windows\" packages/imspy-simulation/src/imspy_simulation -g '*.py' | head -80; nl -ba packages/imspy-simulation/tests/test_acquisition_scheme_parity.py" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
packages/imspy-simulation/src/imspy_simulation/experiment.py:635:        self.dia_ms_ms_windows = None
packages/imspy-simulation/src/imspy_simulation/experiment.py:641:        self.dia_ms_ms_windows = self.get_table('dia_ms_ms_windows')
packages/imspy-simulation/src/imspy_simulation/experiment.py:649:        for _, row in self.dia_ms_ms_windows.iterrows():
packages/imspy-simulation/src/imspy_simulation/acquisition.py:187:        self.frames_to_window_groups = None
packages/imspy-simulation/src/imspy_simulation/acquisition.py:188:        self.dia_ms_ms_windows = pd.read_csv(window_group_file)
packages/imspy-simulation/src/imspy_simulation/acquisition.py:224:        Returns ``(dia_ms_ms_windows, frames_to_window_groups, precursor_every)``
packages/imspy-simulation/src/imspy_simulation/acquisition.py:252:            self.dia_ms_ms_windows, frames_to_window_groups, self.precursor_every = scheme_layout
packages/imspy-simulation/src/imspy_simulation/acquisition.py:254:            self.frames_to_window_groups = frames_to_window_groups
packages/imspy-simulation/src/imspy_simulation/acquisition.py:259:                self.dia_ms_ms_windows = self.reference.dia_ms_ms_windows.rename(
packages/imspy-simulation/src/imspy_simulation/acquisition.py:270:            self.frames_to_window_groups = self.generate_frame_to_window_group_table(verbose=verbose)
packages/imspy-simulation/src/imspy_simulation/acquisition.py:273:            self.dia_ms_ms_windows['collision_energy'] = np.round(self.dia_ms_ms_windows['collision_energy'].values,
packages/imspy-simulation/src/imspy_simulation/acquisition.py:286:            table=self.frames_to_window_groups
packages/imspy-simulation/src/imspy_simulation/acquisition.py:289:            table_name='dia_ms_ms_windows',
packages/imspy-simulation/src/imspy_simulation/acquisition.py:290:            table=self.dia_ms_ms_windows
packages/imspy-simulation/src/imspy_simulation/acquisition.py:362:        instance.frames_to_window_groups = instance.synthetics_handle.get_table('dia_ms_ms_info')
packages/imspy-simulation/src/imspy_simulation/acquisition.py:363:        instance.dia_ms_ms_windows = instance.synthetics_handle.get_table('dia_ms_ms_windows')
packages/imspy-simulation/src/imspy_simulation/data/database.py:217:        dia_ms_ms_windows: Cached DIA MS/MS windows DataFrame.
packages/imspy-simulation/src/imspy_simulation/data/database.py:235:        self.dia_ms_ms_windows: Optional[pd.DataFrame] = None
packages/imspy-simulation/src/imspy_simulation/data/database.py:242:            self.dia_ms_ms_windows = self.get_table("dia_ms_ms_windows")
packages/imspy-simulation/src/imspy_simulation/data/database.py:266:        if self.dia_ms_ms_windows is None:
packages/imspy-simulation/src/imspy_simulation/data/database.py:269:        if self.dia_ms_ms_windows is None:
packages/imspy-simulation/src/imspy_simulation/data/database.py:273:        for _, row in self.dia_ms_ms_windows.iterrows():
packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:20:        config_name: Name of the config file (e.g., 'dia_ms_ms_windows.csv')
packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:504:            # dia-PASEF: Load all windows from actual dia_ms_ms_windows.csv
packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:506:            window_df = load_window_config("dia_ms_ms_windows.csv")
packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:580:            # midia-PASEF: Load windows from actual midia_ms_ms_windows.csv
packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:581:            window_df = load_window_config("midia_ms_ms_windows.csv")
packages/imspy-simulation/src/imspy_simulation/tdf.py:496:    def write_dia_ms_ms_windows(self, dia_ms_ms_windows: pd.DataFrame) -> None:
packages/imspy-simulation/src/imspy_simulation/tdf.py:497:        out = dia_ms_ms_windows.rename(columns={
packages/imspy-simulation/src/imspy_simulation/timsim/jobs/add_noise_from_real_data.py:102:        d = acquisition_builder.frames_to_window_groups
packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py:179:        acquisition_builder.tdf_writer.write_dia_ms_ms_windows(
packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py:180:            acquisition_builder.synthetics_handle.get_table('dia_ms_ms_windows'))
     1	"""Golden parity: the vendor-neutral AcquisitionScheme (via imspy_connector)
     2	must reproduce the Bruker DIA layout the legacy `use_reference_ds_layout` path
     3	copies from the reference `.d` — both the `DiaFrameMsMsWindows` window table and
     4	the per-frame `DiaFrameMsMsInfo` (frame→group) table.
     5	
     6	Gated on TIMSIM_BRUKER_DIA_D pointing at a real DIA-PASEF `.d`.
     7	"""
     8	import os
     9	import sqlite3
    10	
    11	import numpy as np
    12	import pandas as pd
    13	import pytest
    14	
    15	import imspy_connector
    16	
    17	acq = imspy_connector.py_acquisition
    18	DIA_D = os.environ.get("TIMSIM_BRUKER_DIA_D")
    19	pytestmark = pytest.mark.skipif(
    20	    not DIA_D, reason="set TIMSIM_BRUKER_DIA_D to a real DIA-PASEF .d folder"
    21	)
    22	
    23	
    24	def _norm(df, cols):
    25	    return df[cols].astype(float).round(6).sort_values(cols).reset_index(drop=True)
    26	
    27	
    28	def test_scheme_windows_match_reference():
    29	    scheme = acq.PyAcquisitionScheme.from_bruker_d(DIA_D)
    30	    windows = pd.DataFrame(scheme.to_bruker_windows())
    31	
    32	    con = sqlite3.connect(DIA_D + "/analysis.tdf")
    33	    ref = pd.read_sql(
    34	        "SELECT WindowGroup window_group, ScanNumBegin scan_start, ScanNumEnd scan_end, "
    35	        "IsolationMz isolation_mz, IsolationWidth isolation_width, "
    36	        "CollisionEnergy collision_energy FROM DiaFrameMsMsWindows",
    37	        con,
    38	    )
    39	    cols = list(ref.columns)
    40	    assert len(windows) == len(ref)
    41	    pd.testing.assert_frame_equal(_norm(windows, cols), _norm(ref, cols))
    42	
    43	
    44	def test_scheme_info_matches_reference_and_legacy_formula():
    45	    scheme = acq.PyAcquisitionScheme.from_bruker_d(DIA_D)
    46	    con = sqlite3.connect(DIA_D + "/analysis.tdf")
    47	    num_frames = int(con.execute("SELECT MAX(Id) FROM Frames").fetchone()[0])
    48	
    49	    info = pd.DataFrame(scheme.to_bruker_info(num_frames))
    50	
    51	    # (a) matches the .d's own DiaFrameMsMsInfo exactly
    52	    ref = pd.read_sql("SELECT Frame frame, WindowGroup window_group FROM DiaFrameMsMsInfo", con)
    53	    pd.testing.assert_frame_equal(
    54	        _norm(info, ["frame", "window_group"]), _norm(ref, ["frame", "window_group"])
    55	    )
    56	
    57	    # (b) matches the legacy position formula (index % precursor_every, ms2 only)
    58	    pe = scheme.cycle_length()
    59	    legacy = pd.DataFrame(
    60	        [
    61	            {"frame": f, "window_group": (f - 1) % pe}
    62	            for f in range(1, num_frames + 1)
    63	            if (f - 1) % pe != 0
    64	        ]
    65	    )
    66	    pd.testing.assert_frame_equal(
    67	        _norm(info, ["frame", "window_group"]), _norm(legacy, ["frame", "window_group"])
    68	    )

exec
/bin/bash -lc 'git show ea07bcf1''^:packages/imspy-simulation/src/imspy_simulation/acquisition.py | nl -ba | sed -n '"'165,260p'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
   165	                 gradient_length=50 * 60,
   166	                 rt_cycle_length=0.1054,
   167	                 use_reference_ds_layout: bool = True,
   168	                 round_collision_energy: bool = True,
   169	                 collision_energy_decimals: int = 1
   170	                 ):
   171	
   172	        super().__init__(path, reference_ds, gradient_length, rt_cycle_length,
   173	                         exp_name=exp_name)
   174	        # TODO: check this, could be missing replacement of reference layout of windows
   175	        if use_reference_ds_layout:
   176	            rt_cycle_length = np.mean(np.diff(reference_ds.meta_data.Time))
   177	            if verbose:
   178	                print('Using reference dataset cycle length:', np.round(rt_cycle_length, 4))
   179	            self.rt_cycle_length = rt_cycle_length
   180	
   181	        self.acquisition_name = acquisition_name
   182	        self.scan_table = None
   183	        self.frame_table = None
   184	        self.frames_to_window_groups = None
   185	        self.dia_ms_ms_windows = pd.read_csv(window_group_file)
   186	        self.use_reference_ds_layout = use_reference_ds_layout
   187	        self.reference = reference_ds
   188	        self.round_collision_energy = round_collision_energy
   189	        self.collision_energy_decimals = collision_energy_decimals
   190	
   191	        # TODO: check if the number of scans in the window group file matches the number of scans in the experiment
   192	
   193	        self.acquisition_mode = AcquisitionMode('DIA')
   194	        self.verbose = verbose
   195	        self.precursor_every = precursor_every
   196	
   197	        self._setup(verbose=verbose)
   198	
   199	    def calculate_frame_types(self, verbose: bool = True) -> NDArray:
   200	        if verbose:
   201	            print(f'Calculating frame types, precursor frame will be taken every {self.precursor_every} rt cycles.')
   202	        return np.array([0 if (x - 1) % self.precursor_every == 0 else 9 for x in self.frame_table.frame_id])
   203	
   204	    def generate_frame_to_window_group_table(self, verbose: bool = True) -> pd.DataFrame:
   205	        if verbose:
   206	            print(f'generating frame to window group table, precursors every {self.precursor_every} frames.')
   207	
   208	        table_list = []
   209	        for index, row in self.frame_table.iterrows():
   210	            frame_id, ms_type = row.frame_id, row.ms_type
   211	            wg = index % self.precursor_every
   212	            if ms_type > 0:
   213	                table_list.append({'frame': int(frame_id), 'window_group': wg})
   214	
   215	        return pd.DataFrame(table_list)
   216	
   217	    def _setup(self, verbose: bool = True):
   218	        self.frame_table = self.generate_frame_table(verbose=verbose)
   219	        self.scan_table = self.generate_scan_table(verbose=verbose)
   220	
   221	        if self.use_reference_ds_layout:
   222	            self.precursor_every = int(np.diff(self.reference.precursor_frames)[0])
   223	            self.dia_ms_ms_windows = self.reference.dia_ms_ms_windows.rename(
   224	                columns={
   225	                    'WindowGroup': 'window_group',
   226	                    'ScanNumBegin': 'scan_start',
   227	                    'ScanNumEnd': 'scan_end',
   228	                    'IsolationMz': 'isolation_mz',
   229	                    'IsolationWidth': 'isolation_width',
   230	                    'CollisionEnergy': 'collision_energy',
   231	                }
   232	            )
   233	
   234	        self.frame_table['ms_type'] = self.calculate_frame_types(verbose=verbose)
   235	        self.frames_to_window_groups = self.generate_frame_to_window_group_table(verbose=verbose)
   236	
   237	        if self.round_collision_energy:
   238	            self.dia_ms_ms_windows['collision_energy'] = np.round(self.dia_ms_ms_windows['collision_energy'].values,
   239	                                                                  decimals=self.collision_energy_decimals)
   240	
   241	        self.synthetics_handle.create_table(
   242	            table_name='frames',
   243	            table=self.frame_table
   244	        )
   245	        self.synthetics_handle.create_table(
   246	            table_name='scans',
   247	            table=self.scan_table
   248	        )
   249	        self.synthetics_handle.create_table(
   250	            table_name='dia_ms_ms_info',
   251	            table=self.frames_to_window_groups
   252	        )
   253	        self.synthetics_handle.create_table(
   254	            table_name='dia_ms_ms_windows',
   255	            table=self.dia_ms_ms_windows
   256	        )
   257	
   258	    @staticmethod
   259	    def from_config(
   260	            path: str,

exec
/bin/bash -lc "python3 - <<'PY'
import sqlite3, pandas as pd, numpy as np
for dt in [np.uint32, np.int64]:
    con=sqlite3.connect(':memory:')
    df=pd.DataFrame({'frame':np.array([2,3],dtype=dt),'window_group':np.array([1,2],dtype=dt)})
    df.to_sql('x',con,index=False)
    print(dt.__name__, con.execute('pragma table_info(x)').fetchall(), pd.read_sql('select * from x',con).dtypes.to_dict())
PY" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 350ms:
uint32 [(0, 'frame', 'INTEGER', 0, None, 0), (1, 'window_group', 'INTEGER', 0, None, 0)] {'frame': dtype('int64'), 'window_group': dtype('int64')}
int64 [(0, 'frame', 'INTEGER', 0, None, 0), (1, 'window_group', 'INTEGER', 0, None, 0)] {'frame': dtype('int64'), 'window_group': dtype('int64')}

codex
**Findings**

1. **Medium: exact legacy equivalence depends on canonical group IDs.**  
   [acquisition.py:252](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/src/imspy_simulation/acquisition.py:252) now uses preserved reference `WindowGroup` IDs. Legacy generated `window_group = frame_position % precursor_every` at [acquisition.py:214](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/src/imspy_simulation/acquisition.py:214). These agree only when cycle position 1 maps to group 1, position 2 to group 2, etc. Contiguous but permuted IDs also differ. For non-canonical references the new result is internally and reference-correct, while legacy was inconsistent with its copied window table, but this is still a backward-compatibility change.

2. **Medium: window row ordering is not proven equivalent.**  
   The scheme reconstructs rows grouped in acquisition-cycle order; legacy retained whatever order `reference.dia_ms_ms_windows` supplied. The test sorts every column at [test_acquisition_scheme_parity.py:25](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/tests/test_acquisition_scheme_parity.py:25), masking ordering changes. SQLite itself promises no order without `ORDER BY`, so consumers should not rely on it, but the requested byte-for-behavior parity is not established. `dia_ms_ms_info` should remain frame-ascending because `to_bruker_info()` emits frames sequentially.

3. **Low: the parity test does not exercise the migrated builder.**  
   It tests connector conversion directly, using the reference file’s existing frame count, not the builder’s recomputed synthetic frame count. It therefore misses:
   - `_setup()` branch selection and persistence;
   - CE rounding at [acquisition.py:272](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/src/imspy_simulation/acquisition.py:272);
   - column order and dtypes;
   - generated frame count and partial final cycles;
   - fallback behavior;
   - non-`1..N` or permuted groups.

   Moreover, the legacy-formula assertion at [test_acquisition_scheme_parity.py:57](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/tests/test_acquisition_scheme_parity.py:57) would reject a valid non-canonical reference even though the new implementation correctly preserves it.

4. **Low: fallback can conceal deployment skew.**  
   [acquisition.py:227](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/src/imspy_simulation/acquisition.py:227) silently falls back when the module is unavailable, with notification only under `verbose=True`. That preserves legacy behavior, but an unexpectedly old/mispackaged connector can go unnoticed. A warning would be preferable. Other extraction/API failures correctly propagate rather than degrade silently.

5. **Low: in-memory integer dtypes change, but persisted behavior is safe.**  
   Scheme columns are `uint32`; legacy-generated columns are normally `int64`. Pandas SQLite serialization maps both `uint32` and `int64` to SQLite `INTEGER`, and `read_sql` returns `int64`. Rust consumers read by column name into `u32`, so persisted simulator behavior should not break. Explicit casts would still make the builder’s public in-memory tables stable. Dictionary insertion also currently produces the expected column order.

The `num_frames` recompute at [acquisition.py:182](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/src/imspy_simulation/acquisition.py:182) is correct and sufficient for generated state: `_setup()` creates both frame and scan tables afterward. Scan layout is independent of RT timing, and no other precomputed RT-dependent table is retained.
tokens used
44,854
**Findings**

1. **Medium: exact legacy equivalence depends on canonical group IDs.**  
   [acquisition.py:252](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/src/imspy_simulation/acquisition.py:252) now uses preserved reference `WindowGroup` IDs. Legacy generated `window_group = frame_position % precursor_every` at [acquisition.py:214](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/src/imspy_simulation/acquisition.py:214). These agree only when cycle position 1 maps to group 1, position 2 to group 2, etc. Contiguous but permuted IDs also differ. For non-canonical references the new result is internally and reference-correct, while legacy was inconsistent with its copied window table, but this is still a backward-compatibility change.

2. **Medium: window row ordering is not proven equivalent.**  
   The scheme reconstructs rows grouped in acquisition-cycle order; legacy retained whatever order `reference.dia_ms_ms_windows` supplied. The test sorts every column at [test_acquisition_scheme_parity.py:25](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/tests/test_acquisition_scheme_parity.py:25), masking ordering changes. SQLite itself promises no order without `ORDER BY`, so consumers should not rely on it, but the requested byte-for-behavior parity is not established. `dia_ms_ms_info` should remain frame-ascending because `to_bruker_info()` emits frames sequentially.

3. **Low: the parity test does not exercise the migrated builder.**  
   It tests connector conversion directly, using the reference file’s existing frame count, not the builder’s recomputed synthetic frame count. It therefore misses:
   - `_setup()` branch selection and persistence;
   - CE rounding at [acquisition.py:272](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/src/imspy_simulation/acquisition.py:272);
   - column order and dtypes;
   - generated frame count and partial final cycles;
   - fallback behavior;
   - non-`1..N` or permuted groups.

   Moreover, the legacy-formula assertion at [test_acquisition_scheme_parity.py:57](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/tests/test_acquisition_scheme_parity.py:57) would reject a valid non-canonical reference even though the new implementation correctly preserves it.

4. **Low: fallback can conceal deployment skew.**  
   [acquisition.py:227](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/src/imspy_simulation/acquisition.py:227) silently falls back when the module is unavailable, with notification only under `verbose=True`. That preserves legacy behavior, but an unexpectedly old/mispackaged connector can go unnoticed. A warning would be preferable. Other extraction/API failures correctly propagate rather than degrade silently.

5. **Low: in-memory integer dtypes change, but persisted behavior is safe.**  
   Scheme columns are `uint32`; legacy-generated columns are normally `int64`. Pandas SQLite serialization maps both `uint32` and `int64` to SQLite `INTEGER`, and `read_sql` returns `int64`. Rust consumers read by column name into `u32`, so persisted simulator behavior should not break. Explicit casts would still make the builder’s public in-memory tables stable. Dictionary insertion also currently produces the expected column order.

The `num_frames` recompute at [acquisition.py:182](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/src/imspy_simulation/acquisition.py:182) is correct and sufficient for generated state: `_setup()` creates both frame and scan tables afterward. Scan layout is independent of RT timing, and no other precomputed RT-dependent table is retained.
