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
session id: 019ea7e0-5e0c-7b73-b4e0-9c20d7b61fd0
--------
user
Review this design doc as an independent engineer. It proposes a vendor-neutral AcquisitionScheme for a DIA mass-spec simulator (TimSim) that already supports Bruker timsTOF and is being extended to Thermo Orbitrap Astral and SCIEX ZenoTOF. Context you can trust: the team has working, oracle-verified Rust readers/writers for Thermo .raw (thermorawfile: scan_event gives per-MS2 isolation center/width/CE; author_profile/author_centroids) and a SCIEX .wiff method reader (sciexwiff: SWATHMethod windows). The existing timsTOF acquisition layout is Python, mobility-coupled (windows carry mobility scan ranges), and sourced by copying a real Bruker reference .d.

Focus on: 1) is the AcquisitionScheme data model right — does Option<mobility_scan_range> on DiaWindow cleanly capture the timsTOF-vs-(Thermo/SCIEX) difference, or should mobility be split out? 2) the Rust-vs-Python boundary decision (scheme in Rust/rustdf exposed via connector, Python builder migrates onto it) — risks to backward compatibility with the existing dia_ms_ms_windows table and the Bruker reference path; 3) the 6 open questions — which are real blockers vs deferrable, and any the doc should have answered already; 4) anything missing for a faithful per-instrument acquisition design (e.g. MS1 handling, overlapping windows, rolling CE, cycle/RT modeling); 5) is the input(AcquisitionScheme)/output(AcquisitionWriter) split the right factoring? Concrete, specific, cap ~800 words.

<stdin>
# AcquisitionScheme — vendor-neutral DIA acquisition design for TimSim

**Status:** design draft (for review)
**Author:** drafted for David Teschner
**Scope:** the *input/design* side of multi-instrument TimSim — the DIA window
schedule the simulator generates against — complementary to the already-built
`AcquisitionWriter` (the *output* side, `rustdf/src/sim/acquisition.rs`).

---

## 1. Problem

TimSim simulates DIA acquisitions. To extend it beyond Bruker timsTOF to Thermo
(Orbitrap Astral) and SCIEX (ZenoTOF), the simulator must generate scans for the
**target instrument's acquisition scheme** — its isolation-window layout, cycle
structure, and collision-energy plan. Those differ materially per vendor:

- **timsTOF DIA-PASEF**: 2-D windows — (m/z isolation) × (ion-mobility scan
  range). Cycle = 1 precursor (MS1) frame every `precursor_every` frames.
- **Orbitrap Astral**: 1-D narrow m/z windows, **no mobility**. Cycle = 1 MS1
  (FTMS) every ~16 MS2 (ASTMS), precursor m/z stepping, fixed/rolling CE.
- **SCIEX ZenoTOF**: 1-D variable-width SWATH windows (~60), **no mobility**,
  Zeno pulsing.

We want to be able to **inject a real instrument's window layout into the sim**
(e.g. replicate a specific Astral method), not just hand-author windows.

## 2. Current state (what exists today)

The only acquisition abstraction is **`TimsTofAcquisitionBuilderDIA`**
(`packages/imspy-simulation/src/imspy_simulation/acquisition.py`), which is
**timsTOF-coupled**:

- Holds the scheme as a pandas `dia_ms_ms_windows` table with columns
  `window_group, scan_start, scan_end, isolation_mz, isolation_width,
  collision_energy` — note `scan_start/scan_end` are **mobility scan ranges**.
- Sources the layout either from a `window_group_file` CSV or (default,
  `use_reference_ds_layout=True`) by **copying a real Bruker reference `.d`'s**
  `DiaFrameMsMsWindows` (`reference_ds.dia_ms_ms_windows`).
- Also derives `precursor_every` from the reference and builds `frame`/`scan`
  tables; persists to `synthetic_data.db` (`dia_ms_ms_windows`, `dia_ms_ms_info`)
  → written into the output `.d`'s `DiaFrameMsMsWindows` table.

So the "inject the real layout" idea **already exists for Bruker** (copy the
reference `.d`'s windows). The gap: it's mobility-shaped and Bruker-only.

### Readers we already have (this session)

We can already *extract* the window layout from each vendor's real files:

- **Bruker** — `reference_ds.dia_ms_ms_windows` (Python, today).
- **Thermo** — `thermorawfile::RawFile::scan_event(scan)` returns
  `ScanEvent { ms_order, analyzer, isolation_center, isolation_width,
  collision_energy }` per MS2 scan. Walking one template DIA cycle recovers the
  exact Astral window schedule.
- **SCIEX** — `sciexwiff` extracts `SWATHMethod` → ~60 variable windows
  `[lower, upper]` + TOF calibration (CE per window is rolling / not in
  `SWATHMethod`; trailer = 0).

## 3. Proposed model

A vendor-neutral **`AcquisitionScheme`** describing one DIA cycle and how it
repeats. Lives in Rust (`rustdf/src/sim/scheme.rs`), next to `AcquisitionWriter`.

```rust
/// One DIA isolation window in a cycle.
pub struct DiaWindow {
    pub window_group: u32,
    pub isolation_mz: f64,       // center
    pub isolation_width: f64,
    pub collision_energy: f64,
    /// timsTOF only: the mobility scan range. None for Thermo/SCIEX.
    pub mobility_scan_range: Option<(u32, u32)>,
}

pub enum InstrumentKind { TimsTofDia, OrbitrapAstral, SciexZenoTof }

pub struct AcquisitionScheme {
    pub instrument: InstrumentKind,
    pub windows: Vec<DiaWindow>,
    /// MS1 cadence: one precursor scan every `precursor_every` cycle positions.
    pub precursor_every: u32,
    /// Nominal cycle time (s) and gradient length (s) — drive RT sampling.
    pub cycle_time_s: f64,
    pub gradient_length_s: f64,
    /// m/z acquisition range (low, high), for MS1 and window coverage checks.
    pub mz_range: (f64, f64),
}
```

### Extractors (populate the scheme from a real run)

```rust
impl AcquisitionScheme {
    pub fn from_bruker_d(path: &str) -> io::Result<Self>;       // DiaFrameMsMsWindows
    #[cfg(feature = "thermo")]
    pub fn from_thermo_raw(path: &str) -> io::Result<Self>;     // walk MS2 scan_event()
    #[cfg(feature = "sciex")]
    pub fn from_sciex_wiff(path: &str) -> io::Result<Self>;     // SWATHMethod
    pub fn from_window_table(...) -> Self;                       // hand-authored / CSV
}
```

- `from_thermo_raw`: iterate the template's first full cycle of MS2 scans, read
  `scan_event()` per scan → `DiaWindow { isolation_mz=center, width, CE,
  mobility_scan_range=None }`; `precursor_every` = count of MS2 between MS1s;
  `cycle_time_s` from scan RTs.
- `from_sciex_wiff`: `SWATHMethod` windows → `DiaWindow` (center = (lo+hi)/2,
  width = hi-lo). CE: rolling — derive from a default model or per-window if
  recoverable (open question).

## 4. How it plugs in

- **Generation**: the per-vendor frame/scan builders consume the
  `AcquisitionScheme` to decide which MS2 windows exist, their CE, and (for
  timsTOF) the mobility ranges. The neutral `ScanDescriptor`s the sim emits feed
  the matching `AcquisitionWriter`.
- **Migration of `TimsTofAcquisitionBuilderDIA`**: it stops reading
  `reference_ds.dia_ms_ms_windows` directly and instead consumes an
  `AcquisitionScheme` (via the connector). For Bruker this is
  `from_bruker_d(reference)` → identical behavior (the `mobility_scan_range`
  carries `scan_start/scan_end`). Backwards-compatible: the existing
  `dia_ms_ms_windows` table is produced from the scheme.
- **Symmetry**: `AcquisitionScheme` (in) and `AcquisitionWriter` (out) are the
  two halves. A round trip is: `from_thermo_raw(template)` → generate → author
  via `ThermoRawWriter(template)` → `.raw`.

## 5. Boundary / location decision

The new readers are Rust (`thermorawfile`, `sciexwiff`); the existing builder is
Python. Proposal: `AcquisitionScheme` lives in `rustdf/src/sim`, exposed through
`imspy_connector` as a `PyAcquisitionScheme`. Python `TimsTofAcquisitionBuilderDIA`
consumes it via the binding rather than reading the reference `.d` itself. The
Bruker extractor can stay Python-side initially (it already works) and move to
Rust later, as long as the neutral scheme is the contract.

## 6. Open questions

1. **CE model for SCIEX** rolling collision energy — derive from m/z (a linear
   CE(m/z) model) when not stored, or require a user-provided CE table?
2. **Mobility for timsTOF** — keep `mobility_scan_range` on `DiaWindow`, or split
   into a `TimsTofExtras` side-struct so the neutral type stays 1-D?
3. **Variable cycle / overlapping windows** — Astral/SCIEX windows can overlap
   (1 Th overlap observed in SCIEX). Does the generator need explicit overlap
   handling, or is per-window isolation sufficient?
4. **Where Bruker extraction lives** (Rust vs Python) given mobility coupling.
5. **MS1 representation** — Astral MS1 is FTMS *profile*; the scheme is about MS2
   windows. Does MS1 need any scheme-level config (mass range only?), or is it
   purely the writer's concern?
6. **scan_event location for arbitrary scans** — `from_thermo_raw` needs to read
   `scan_event()` for MS2 scans beyond the first; thermorawfile's event locator
   currently relies on a uniform stride that may not hold for mixed MS1/MS2
   (MS1 events are longer). Does `from_thermo_raw` need the variable-stride walk
   first?

## 7. Test plan

- `from_thermo_raw(astral_template)` → assert N windows, monotonic centers,
  fixed CE=25, `precursor_every` ≈ 16, all `mobility_scan_range == None`.
- `from_sciex_wiff(zenotof_wiff)` → assert ~60 windows covering 399.5–899.9.
- `from_bruker_d(ref)` round-trips the existing `dia_ms_ms_windows` columns
  (parity vs today's Python path).
- End-to-end: `from_thermo_raw(template)` → generate a 1-MS1 + N-MS2 cycle →
  `ThermoRawWriter` → read back via thermorawfile + Thermo oracle.
</stdin>
codex
I’ll verify the proposed boundary against the current Python builder and Rust writer interfaces, then give a findings-first review capped near 800 words.
exec
/bin/bash -lc "sed -n '1,280p' packages/imspy-simulation/src/imspy_simulation/acquisition.py" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
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

exec
/bin/bash -lc "sed -n '1,280p' rustdf/src/sim/acquisition.rs" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
//! Vendor-neutral acquisition-writer abstraction.
//!
//! [`AcquisitionWriter`] turns vendor-neutral [`ScanDescriptor`]s into a vendor
//! raw file. The first implementation is [`ThermoRawWriter`] (Thermo `.raw`, via
//! the `thermo` feature + the `thermorawfile` crate); a Bruker `.d` adapter over
//! the existing write path is a planned follow-up.

use std::io;

/// MS2 quadrupole isolation window.
#[derive(Clone, Copy, Debug)]
pub struct IsolationWindow {
    pub center_mz: f64,
    pub width_mz: f64,
    pub collision_energy: f64,
}

/// One vendor-neutral scan: a peak list plus acquisition metadata.
///
/// Intentionally has **no ion-mobility dimension** — it is the
/// Bruker∩Thermo∩SCIEX common denominator. IMS-bearing Bruker frames are handled
/// by the Bruker writer, which can flatten or carry mobility separately.
#[derive(Clone, Debug)]
pub struct ScanDescriptor {
    pub ms_level: u8,
    pub retention_time: f64,
    /// Present for MS2 (the precursor isolation window).
    pub isolation: Option<IsolationWindow>,
    /// `(m/z, intensity)` pairs; need not be sorted (writers sort).
    pub peaks: Vec<(f64, f32)>,
}

/// A sink that writes vendor-neutral scans into a vendor raw file. Scans are
/// written in acquisition order; `finalize` flushes the file (and fixes any
/// integrity checksum).
pub trait AcquisitionWriter {
    fn write_scan(&mut self, scan: &ScanDescriptor) -> io::Result<()>;
    fn finalize(&mut self) -> io::Result<()>;
}

#[cfg(feature = "thermo")]
pub use thermo::ThermoRawWriter;

#[cfg(feature = "thermo")]
mod thermo {
    use super::*;
    use std::path::{Path, PathBuf};
    use thermorawfile::{Calibration, RawFile};

    /// Authors scans into a Thermo `.raw` by **template-mutation**: open a real
    /// `.raw`, overwrite template scans of matching type in acquisition order
    /// (MS1 → profile via `author_profile`, MS2 → centroids via
    /// `author_centroids`), and save on [`AcquisitionWriter::finalize`].
    ///
    /// The template supplies the frequency grid + calibration; synthetic peaks
    /// are placed at their exact m/z within each packet's byte budget. The
    /// calibration is read once from the first scan (it is ≈constant across a
    /// run); each scan's own frequency grid is taken from its profile.
    pub struct ThermoRawWriter {
        raw: RawFile,
        out_path: PathBuf,
        calib: Calibration,
        /// Template scans that carry a profile (treated as MS1 targets).
        profile_scans: Vec<u32>,
        /// Centroid-only template scans (treated as MS2 targets).
        centroid_scans: Vec<u32>,
        prof_cur: usize,
        cent_cur: usize,
    }

    impl ThermoRawWriter {
        /// Open `template` and prepare to author into `out`.
        pub fn from_template<P: AsRef<Path>, Q: AsRef<Path>>(
            template: P,
            out: Q,
        ) -> io::Result<Self> {
            let raw = RawFile::open(template)?;
            let calib = raw
                .calibration_at_event(raw.scantrailer_addr as usize + 4)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "no MS1 calibration in template")
                })?;

            // Classify template scans by packet type: a non-zero profile section
            // marks an MS1-style (FTMS profile) scan; centroid-only is MS2 (ASTMS).
            let mut profile_scans = Vec::new();
            let mut centroid_scans = Vec::new();
            for (i, e) in raw.index.iter().enumerate() {
                let pkt = (raw.data_addr + e.offset) as usize;
                if pkt + 8 > raw.bytes.len() {
                    continue;
                }
                let profile_size =
                    u32::from_le_bytes(raw.bytes[pkt + 4..pkt + 8].try_into().unwrap());
                let scan = raw.first_scan + i as u32;
                if profile_size > 0 {
                    profile_scans.push(scan);
                } else {
                    centroid_scans.push(scan);
                }
            }
            Ok(Self {
                raw,
                out_path: out.as_ref().to_path_buf(),
                calib,
                profile_scans,
                centroid_scans,
                prof_cur: 0,
                cent_cur: 0,
            })
        }

        /// Template MS1/MS2 capacity, so callers can check a run fits.
        pub fn capacity(&self) -> (usize, usize) {
            (self.profile_scans.len(), self.centroid_scans.len())
        }
    }

    impl AcquisitionWriter for ThermoRawWriter {
        fn write_scan(&mut self, scan: &ScanDescriptor) -> io::Result<()> {
            if scan.ms_level <= 1 {
                let t = *self.profile_scans.get(self.prof_cur).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "template exhausted of MS1 (profile) scans",
                    )
                })?;
                self.prof_cur += 1;
                self.raw.author_profile(t, &scan.peaks, &self.calib)
            } else {
                let t = *self.centroid_scans.get(self.cent_cur).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "template exhausted of MS2 (centroid) scans",
                    )
                })?;
                self.cent_cur += 1;
                self.raw.author_centroids(t, &scan.peaks)
            }
        }

        fn finalize(&mut self) -> io::Result<()> {
            let path = self.out_path.clone();
            self.raw.save(path)
        }
    }
}

#[cfg(all(test, feature = "thermo"))]
mod tests {
    use super::*;

    // Gated: set TIMSIM_ASTRAL_TEMPLATE to a real Orbitrap Astral .raw to run.
    // `cargo test --features thermo -- --nocapture thermo_roundtrip`
    #[test]
    fn thermo_roundtrip() {
        let template = match std::env::var("TIMSIM_ASTRAL_TEMPLATE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP thermo_roundtrip: set TIMSIM_ASTRAL_TEMPLATE=<astral .raw>");
                return;
            }
        };
        let out = std::env::temp_dir().join("rustdf_thermo_roundtrip.raw");

        let mut w = ThermoRawWriter::from_template(&template, &out).expect("open template");
        let (n_ms1, n_ms2) = w.capacity();
        assert!(n_ms1 > 0 && n_ms2 > 0, "template has no MS1/MS2 scans");

        let ms1 = ScanDescriptor {
            ms_level: 1,
            retention_time: 0.0,
            isolation: None,
            peaks: vec![(500.0, 1.0e6), (700.0, 5.0e5)],
        };
        let ms2 = ScanDescriptor {
            ms_level: 2,
            retention_time: 0.01,
            isolation: Some(IsolationWindow {
                center_mz: 500.0,
                width_mz: 2.0,
                collision_energy: 25.0,
            }),
            peaks: vec![(150.1, 3.0e4), (420.2, 8.0e4), (610.3, 5.0e4)],
        };
        w.write_scan(&ms1).expect("write MS1");
        w.write_scan(&ms2).expect("write MS2");
        w.finalize().expect("finalize");

        // Read back through thermorawfile and confirm the authored peaks.
        let rf = thermorawfile::RawFile::open(&out).expect("reopen");
        assert!(rf.checksum_valid(), "checksum invalid");

        // The writer used the first profile scan for MS1 and first centroid scan for MS2.
        let cal = rf
            .calibration_at_event(rf.scantrailer_addr as usize + 4)
            .unwrap();
        let mut prof_scan = None;
        let mut cent_scan = None;
        for i in 0..rf.index.len() {
            let scan = rf.first_scan + i as u32;
            let pkt = (rf.data_addr + rf.index[i].offset) as usize;
            let psize = u32::from_le_bytes(rf.bytes[pkt + 4..pkt + 8].try_into().unwrap());
            if psize > 0 && prof_scan.is_none() {
                prof_scan = Some(scan);
            }
            if psize == 0 && cent_scan.is_none() {
                cent_scan = Some(scan);
            }
        }
        let prof = rf.profile(prof_scan.unwrap()).expect("ms1 profile");
        assert_eq!(prof.chunks.len(), 2, "MS1 peak count");
        let ms1_mz: Vec<f64> = prof
            .chunks
            .iter()
            .map(|c| prof.mz_of_bin(c.first_bin, &cal))
            .collect();
        assert!((ms1_mz[0] - 500.0).abs() < 0.01 && (ms1_mz[1] - 700.0).abs() < 0.01);

        let cents = rf.centroid_peaks(cent_scan.unwrap());
        assert_eq!(cents.len(), 3, "MS2 peak count");
        assert!((cents[0].mz - 150.1).abs() < 0.01);
        assert!((cents[2].mz - 610.3).abs() < 0.01);

        eprintln!("thermo_roundtrip OK: MS1 {:?}, MS2 {} peaks", ms1_mz, cents.len());
    }
}

exec
/bin/bash -lc "rg -n \"dia_ms_ms_windows|DiaFrameMsMsWindows|precursor_every|ScanDescriptor|AcquisitionWriter\" packages rustdf imspy_connector -g '*.py' -g '*.rs'" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
rustdf/src/sim/acquisition.rs:3://! [`AcquisitionWriter`] turns vendor-neutral [`ScanDescriptor`]s into a vendor
rustdf/src/sim/acquisition.rs:24:pub struct ScanDescriptor {
rustdf/src/sim/acquisition.rs:36:pub trait AcquisitionWriter {
rustdf/src/sim/acquisition.rs:37:    fn write_scan(&mut self, scan: &ScanDescriptor) -> io::Result<()>;
rustdf/src/sim/acquisition.rs:53:    /// `author_centroids`), and save on [`AcquisitionWriter::finalize`].
rustdf/src/sim/acquisition.rs:119:    impl AcquisitionWriter for ThermoRawWriter {
rustdf/src/sim/acquisition.rs:120:        fn write_scan(&mut self, scan: &ScanDescriptor) -> io::Result<()> {
rustdf/src/sim/acquisition.rs:170:        let ms1 = ScanDescriptor {
rustdf/src/sim/acquisition.rs:176:        let ms2 = ScanDescriptor {
rustdf/src/sim/handle.rs:182:        let mut stmt = self.connection.prepare("SELECT * FROM dia_ms_ms_windows")?;
rustdf/src/data/meta.rs:377:pub fn read_dia_ms_ms_windows(
rustdf/src/data/meta.rs:393:    let query = format!("SELECT {} FROM DiaFrameMsMsWindows", rows.join(", "));
rustdf/src/data/dia.rs:6:    read_dia_ms_ms_info, read_dia_ms_ms_windows, read_global_meta_sql, read_meta_data_sql,
rustdf/src/data/dia.rs:428:    pub dia_ms_ms_windows: Vec<DiaMsMsWindow>,
rustdf/src/data/dia.rs:443:        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
rustdf/src/data/dia.rs:478:        let dia_index = DiaIndex::new(&meta_data, &dia_ms_mis_info, &dia_ms_ms_windows);
rustdf/src/data/dia.rs:485:            dia_ms_ms_windows,
rustdf/src/data/dia.rs:503:        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();
rustdf/src/data/dia.rs:528:        let dia_index = DiaIndex::new(&meta_data, &dia_ms_mis_info, &dia_ms_ms_windows);
rustdf/src/data/dia.rs:535:            dia_ms_ms_windows,
rustdf/src/data/dia.rs:544:    /// Collect all program slices for a window group from DiaFrameMsMsWindows.
rustdf/src/data/dia.rs:547:        self.dia_ms_ms_windows
rustdf/src/data/dia.rs:739:    /// Merged, sorted **global scan** unions for this DIA group, from DiaFrameMsMsWindows.
rustdf/src/data/dia.rs:743:            .dia_ms_ms_windows
rustdf/src/data/dia.rs:758:    /// m/z unions (min..max) for the group from DiaFrameMsMsWindows (wide clamp).
rustdf/src/data/dia.rs:763:        for w in &self.dia_ms_ms_windows {
rustdf/examples/build_bruker_pseudo_ms2_v3.rs:102:         FROM DiaFrameMsMsWindows ORDER BY WindowGroup, ScanNumBegin",
rustdf/examples/dump_bruker_centroids.rs:13:// definitions from `dia_ms_ms_windows` and `dia_ms_ms_info`.
rustdf/examples/dump_bruker_centroids.rs:44:         FROM DiaFrameMsMsWindows ORDER BY WindowGroup, ScanNumBegin",
rustdf/examples/build_bruker_pseudo_ms2.rs:84:         FROM DiaFrameMsMsWindows ORDER BY WindowGroup, ScanNumBegin",
packages/imspy-core/src/imspy_core/timstof/dia.py:21:    def dia_ms_ms_windows(self):
packages/imspy-core/src/imspy_core/timstof/dia.py:27:        return pd.read_sql_query("SELECT * from DiaFrameMsMsWindows",
packages/imspy-vis/src/imspy_vis/frame_rendering.py:289:        self.windows = self.handle.dia_ms_ms_windows.copy()
packages/imspy-simulation/src/imspy_simulation/experiment.py:635:        self.dia_ms_ms_windows = None
packages/imspy-simulation/src/imspy_simulation/experiment.py:641:        self.dia_ms_ms_windows = self.get_table('dia_ms_ms_windows')
packages/imspy-simulation/src/imspy_simulation/experiment.py:649:        for _, row in self.dia_ms_ms_windows.iterrows():
packages/imspy-simulation/src/imspy_simulation/acquisition.py:164:                 precursor_every: int = 17,
packages/imspy-simulation/src/imspy_simulation/acquisition.py:185:        self.dia_ms_ms_windows = pd.read_csv(window_group_file)
packages/imspy-simulation/src/imspy_simulation/acquisition.py:195:        self.precursor_every = precursor_every
packages/imspy-simulation/src/imspy_simulation/acquisition.py:201:            print(f'Calculating frame types, precursor frame will be taken every {self.precursor_every} rt cycles.')
packages/imspy-simulation/src/imspy_simulation/acquisition.py:202:        return np.array([0 if (x - 1) % self.precursor_every == 0 else 9 for x in self.frame_table.frame_id])
packages/imspy-simulation/src/imspy_simulation/acquisition.py:206:            print(f'generating frame to window group table, precursors every {self.precursor_every} frames.')
packages/imspy-simulation/src/imspy_simulation/acquisition.py:211:            wg = index % self.precursor_every
packages/imspy-simulation/src/imspy_simulation/acquisition.py:222:            self.precursor_every = int(np.diff(self.reference.precursor_frames)[0])
packages/imspy-simulation/src/imspy_simulation/acquisition.py:223:            self.dia_ms_ms_windows = self.reference.dia_ms_ms_windows.rename(
packages/imspy-simulation/src/imspy_simulation/acquisition.py:238:            self.dia_ms_ms_windows['collision_energy'] = np.round(self.dia_ms_ms_windows['collision_energy'].values,
packages/imspy-simulation/src/imspy_simulation/acquisition.py:254:            table_name='dia_ms_ms_windows',
packages/imspy-simulation/src/imspy_simulation/acquisition.py:255:            table=self.dia_ms_ms_windows
packages/imspy-simulation/src/imspy_simulation/acquisition.py:280:            precursor_every=config['precursor_every'],
packages/imspy-simulation/src/imspy_simulation/acquisition.py:321:        instance.precursor_every = int(np.diff(instance.reference.precursor_frames)[0])
packages/imspy-simulation/src/imspy_simulation/acquisition.py:328:        instance.dia_ms_ms_windows = instance.synthetics_handle.get_table('dia_ms_ms_windows')
packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:20:        config_name: Name of the config file (e.g., 'dia_ms_ms_windows.csv')
packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:504:            # dia-PASEF: Load all windows from actual dia_ms_ms_windows.csv
packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:506:            window_df = load_window_config("dia_ms_ms_windows.csv")
packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:580:            # midia-PASEF: Load windows from actual midia_ms_ms_windows.csv
packages/imspy-simulation/src/imspy_simulation/timsim/gui/components/plots.py:581:            window_df = load_window_config("midia_ms_ms_windows.csv")
packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py:179:        acquisition_builder.tdf_writer.write_dia_ms_ms_windows(
packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py:180:            acquisition_builder.synthetics_handle.get_table('dia_ms_ms_windows'))
packages/imspy-simulation/src/imspy_simulation/data/database.py:217:        dia_ms_ms_windows: Cached DIA MS/MS windows DataFrame.
packages/imspy-simulation/src/imspy_simulation/data/database.py:235:        self.dia_ms_ms_windows: Optional[pd.DataFrame] = None
packages/imspy-simulation/src/imspy_simulation/data/database.py:242:            self.dia_ms_ms_windows = self.get_table("dia_ms_ms_windows")
packages/imspy-simulation/src/imspy_simulation/data/database.py:266:        if self.dia_ms_ms_windows is None:
packages/imspy-simulation/src/imspy_simulation/data/database.py:269:        if self.dia_ms_ms_windows is None:
packages/imspy-simulation/src/imspy_simulation/data/database.py:273:        for _, row in self.dia_ms_ms_windows.iterrows():
packages/imspy-simulation/src/imspy_simulation/timsim/validate/runner.py:46:    Checks for presence of DIA-specific tables (DiaFrameMsMsInfo, DiaFrameMsMsWindows)
packages/imspy-simulation/src/imspy_simulation/timsim/validate/runner.py:70:        has_dia_tables = "DiaFrameMsMsInfo" in tables or "DiaFrameMsMsWindows" in tables
packages/imspy-simulation/src/imspy_simulation/tdf.py:496:    def write_dia_ms_ms_windows(self, dia_ms_ms_windows: pd.DataFrame) -> None:
packages/imspy-simulation/src/imspy_simulation/tdf.py:497:        out = dia_ms_ms_windows.rename(columns={
packages/imspy-simulation/src/imspy_simulation/tdf.py:506:        self._create_table(self.conn, out, "DiaFrameMsMsWindows")

codex
The current code confirms a key compatibility hazard: `window_group` is not merely metadata; it drives Bruker frame assignment, while the stored table is consumed row-wise with mandatory scan bounds. I’m checking those consumers before ranking the design issues.
exec
/bin/bash -lc "sed -n '620,680p' packages/imspy-simulation/src/imspy_simulation/experiment.py" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
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

exec
/bin/bash -lc "sed -n '280,350p' packages/imspy-simulation/src/imspy_simulation/acquisition.py" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
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

exec
/bin/bash -lc "sed -n '350,420p' rustdf/src/data/meta.rs && sed -n '160,210p' rustdf/src/sim/handle.rs" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:

pub fn read_dia_ms_ms_info(
    bruker_d_folder_name: &str,
) -> Result<Vec<DiaMsMisInfo>, Box<dyn std::error::Error>> {
    // Connect to the database
    let db_path = Path::new(bruker_d_folder_name).join("analysis.tdf");
    let conn = Connection::open(db_path)?;

    // prepare the query
    let rows: Vec<&str> = vec!["Frame", "WindowGroup"];
    let query = format!("SELECT {} FROM DiaFrameMsMsInfo", rows.join(", "));

    // execute the query
    let frames_rows: Result<Vec<DiaMsMisInfo>, _> = conn
        .prepare(&query)?
        .query_map([], |row| {
            Ok(DiaMsMisInfo {
                frame_id: row.get(0)?,
                window_group: row.get(1)?,
            })
        })?
        .collect();

    // return the frames
    Ok(frames_rows?)
}

pub fn read_dia_ms_ms_windows(
    bruker_d_folder_name: &str,
) -> Result<Vec<DiaMsMsWindow>, Box<dyn std::error::Error>> {
    // Connect to the database
    let db_path = Path::new(bruker_d_folder_name).join("analysis.tdf");
    let conn = Connection::open(db_path)?;

    // prepare the query
    let rows: Vec<&str> = vec![
        "WindowGroup",
        "ScanNumBegin",
        "ScanNumEnd",
        "IsolationMz",
        "IsolationWidth",
        "CollisionEnergy",
    ];
    let query = format!("SELECT {} FROM DiaFrameMsMsWindows", rows.join(", "));

    // execute the query
    let frames_rows: Result<Vec<DiaMsMsWindow>, _> = conn
        .prepare(&query)?
        .query_map([], |row| {
            Ok(DiaMsMsWindow {
                window_group: row.get(0)?,
                scan_num_begin: row.get(1)?,
                scan_num_end: row.get(2)?,
                isolation_mz: row.get(3)?,
                isolation_width: row.get(4)?,
                collision_energy: row.get(5)?,
            })
        })?
        .collect();

    // return the frames
    Ok(frames_rows?)
}

/// Read m/z calibration data from the MzCalibration table.
/// This provides the coefficients needed for accurate TOF to m/z conversion
/// without requiring the Bruker SDK.
///
/// The calibration formula is:
///   tof_time = (tof_index + 0.5) * digitizer_timebase + digitizer_delay
///   sqrt(mz) = c0 + c1*tof_time + c2*tof_time^2 + ...

        Ok(FragmentIonSim::new(
            row.get("peptide_id")?,
            row.get("ion_id")?,
            row.get("collision_energy")?,
            row.get("charge")?,
            indices,
            values,
        ))
    }

    pub fn read_ions(&self) -> rusqlite::Result<Vec<IonSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM ions")?;
        let ions_iter = stmt.query_map([], |row| Self::ion_from_row(row))?;
        let mut ions = Vec::new();
        for ion in ions_iter {
            ions.push(ion?);
        }
        Ok(ions)
    }

    pub fn read_window_group_settings(&self) -> rusqlite::Result<Vec<WindowGroupSettingsSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM dia_ms_ms_windows")?;
        let window_group_settings_iter = stmt.query_map([], |row| {
            Ok(WindowGroupSettingsSim::new(
                row.get("window_group")?,
                row.get("scan_start")?,
                row.get("scan_end")?,
                row.get("isolation_mz")?,
                row.get("isolation_width")?,
                row.get("collision_energy")?,
            ))
        })?;
        let mut window_group_settings = Vec::new();
        for window_group_setting in window_group_settings_iter {
            window_group_settings.push(window_group_setting?);
        }
        Ok(window_group_settings)
    }

    pub fn read_frame_to_window_group(&self) -> rusqlite::Result<Vec<FrameToWindowGroupSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM dia_ms_ms_info")?;
        let frame_to_window_group_iter = stmt.query_map([], |row| {
            Ok(FrameToWindowGroupSim::new(
                row.get("frame")?,
                row.get("window_group")?,
            ))
        })?;

        let mut frame_to_window_groups: Vec<FrameToWindowGroupSim> = Vec::new();
        for frame_to_window_group in frame_to_window_group_iter {

exec
/bin/bash -lc "sed -n '470,520p' packages/imspy-simulation/src/imspy_simulation/tdf.py && sed -n '160,195p' packages/imspy-simulation/src/imspy_simulation/timsim/jobs/assemble_frames.py" in /scratch/timsim-demo/SUBMISSION/rustims
 succeeded in 0ms:
                FOREIGN KEY(Frame) REFERENCES Frames(Id),
                FOREIGN KEY(Precursor) REFERENCES Precursors(Id)
            ) WITHOUT ROWID
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS PasefFrameMsMsInfoPrecursorIndex ON PasefFrameMsMsInfo (Precursor)")

        # Insert data with remapped precursor IDs if mapping provided
        for _, row in out.iterrows():
            precursor_id = int(row['Precursor']) if pd.notna(row['Precursor']) else None
            if id_mapping is not None and precursor_id is not None:
                precursor_id = id_mapping.get(precursor_id, precursor_id)

            cursor.execute("""
                INSERT INTO PasefFrameMsMsInfo (Frame, ScanNumBegin, ScanNumEnd, IsolationMz, IsolationWidth, CollisionEnergy, Precursor)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                int(row['Frame']),
                int(row['ScanNumBegin']),
                int(row['ScanNumEnd']),
                float(row['IsolationMz']),
                float(row['IsolationWidth']),
                float(row['CollisionEnergy']),
                precursor_id
            ))
        self.conn.commit()

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
            )

        for frame in built_frames:
            if acquisition_builder.acquisition_mode.mode == 'DDA':
                acquisition_builder.tdf_writer.write_frame(frame, scan_mode=8)
            else:
                acquisition_builder.tdf_writer.write_frame(frame, scan_mode=9)

    logger.info('Writing frame meta data to database ...')

    # write frame meta data to database
    acquisition_builder.tdf_writer.write_frame_meta_data()

    if acquisition_builder.acquisition_mode.mode == 'DIA':
        logger.info("Writing DIA specific meta data to database ...")
        # write frame ms/ms info to database
        acquisition_builder.tdf_writer.write_dia_ms_ms_info(
            acquisition_builder.synthetics_handle.get_table('dia_ms_ms_info'))
        # write frame ms/ms windows to database
        acquisition_builder.tdf_writer.write_dia_ms_ms_windows(
            acquisition_builder.synthetics_handle.get_table('dia_ms_ms_windows'))
        # write prm frame ms ms info
        acquisition_builder.tdf_writer.write_prm_frame_ms_ms_info()
        # write pasef frame ms ms info
        acquisition_builder.tdf_writer.write_pasef_frame_ms_ms_info()
        # write calibration info to database

        if mz_noise_precursor:
            acquisition_builder.tdf_writer.write_calibration_info(
                mz_standard_deviation_ppm=precursor_noise_ppm)

        elif mz_noise_fragment:
            acquisition_builder.tdf_writer.write_calibration_info(
                mz_standard_deviation_ppm=fragment_noise_ppm)

        else:

codex
**Key Findings**

1. **The proposed model is too window-centric.** A flat `Vec<DiaWindow>` plus `precursor_every` cannot unambiguously represent:
   - timsTOF: one frame containing several mobility-partitioned windows sharing a `window_group`.
   - Astral/SCIEX: an ordered sequence of individual MS1/MS2 scans.
   - Variable cycles, staggered schemes, multiple MS1 types, calibration scans, or per-event timing.

   Model an ordered cycle explicitly:

   ```rust
   struct AcquisitionScheme {
       cycle: Vec<AcquisitionEvent>,
       repeat: RepeatPolicy,
       run_duration_s: f64,
   }

   enum AcquisitionEvent {
       Ms1(Ms1Event),
       DiaMs2(DiaMs2Event),
   }
   ```

   Each event should carry relative timing or duration, analyzer/type, m/z range, isolation, and CE policy. For timsTOF, an MS2 event can contain multiple mobility slices belonging to one frame.

2. **Split mobility from the common isolation window.** `Option<(u32,u32)>` is mechanically workable but semantically weak. Scan numbers are Bruker-grid coordinates requiring the reference dataset’s calibration and scan count; they are not a physical mobility range. It also permits invalid states such as mobility on Astral.

   Prefer:

   ```rust
   enum DiaGeometry {
       MzOnly,
       TimsMobility {
           scan_start: u32,
           scan_end: u32,
           // optionally inverse-mobility bounds
       },
   }
   ```

   Better still, put geometry on a timsTOF frame/event rather than every generic window. Keep `IsolationWindow` strictly m/z-based.

3. **Backward compatibility needs an explicit adapter, not an assertion of parity.** Existing code requires six concrete columns and uses `window_group` to assign frames, not merely identify windows ([acquisition.py](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/src/imspy_simulation/acquisition.py:199), [handle.rs](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/handle.rs:181)). The migration must preserve:
   - Exact group IDs and row order.
   - Inclusive/exclusive meaning of `scan_start`/`scan_end`.
   - Integer types and column names.
   - CE rounding behavior.
   - Existing CSV and `use_reference_ds_layout` constructor paths.
   - `dia_ms_ms_info` frame-to-group sequencing.

   Add `scheme.to_bruker_tables()`, and retain the existing Python API as a compatibility wrapper. Golden-test both SQLite tables byte-for-value against the current path. Also fix or test the current ordering bug where `num_frames` is calculated before reference-derived `rt_cycle_length` replaces it.

4. **MS1 is part of acquisition design, not purely writer behavior.** It determines cadence, analyzer, resolution/transient time, profile/centroid representation, mass range, and cycle duration. At minimum, add `Ms1Event { analyzer, mz_range, duration_s, data_mode }`. “One precursor every N positions” is unsuitable terminology outside Bruker and becomes redundant with an explicit event sequence.

5. **CE must be a policy, not mandatory `f64`.** Represent `CollisionEnergyPolicy::{Fixed, PerWindow, Linear{...}, VendorModel{...}, Unknown}`. Silently inventing SCIEX CE would make “real-method replication” misleading. Require a user model/table when extraction cannot recover it, unless an explicitly labeled default simulation mode is selected.

6. **The input/output split is correct, but the current writer contract is incomplete.** `AcquisitionScheme` describes intended events; `AcquisitionWriter` materializes scans. However, the existing Thermo writer ignores descriptor RT, isolation, and CE and only replaces peak packets ([acquisition.rs](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/acquisition.rs:119)). Therefore the claimed arbitrary round trip is only valid when the generated scheme exactly matches the template schedule. Writers need either:
   - Metadata-authoring support, or
   - `validate_scheme(&scheme)` that rejects incompatible event count/order/metadata.

**Open Questions**

- **Blockers now:** 2 (core representation), 5 (MS1/cycle semantics), 6 (correct Thermo extraction), and 1 for faithful SCIEX output.
- **Must be specified now but implementation can follow:** 3. Overlap needs no special isolation algorithm; windows should independently include ions in their bounds, including duplicated coverage. Define boundary conventions and preserve acquisition order.
- **Deferrable:** 4. Bruker extraction may remain Python, provided both paths produce the same versioned Rust DTO and golden tables.

**Missing Design Elements**

Add event-level timing, cycle ordering, CE policy, analyzer/data mode, isolation boundary conventions, optional dwell/transient time, validation rules, scheme versioning, and provenance indicating whether values were extracted, inferred, or user-supplied. `cycle_time_s` alone cannot faithfully model RT sampling or instrument duty cycle.
tokens used
17,539
**Key Findings**

1. **The proposed model is too window-centric.** A flat `Vec<DiaWindow>` plus `precursor_every` cannot unambiguously represent:
   - timsTOF: one frame containing several mobility-partitioned windows sharing a `window_group`.
   - Astral/SCIEX: an ordered sequence of individual MS1/MS2 scans.
   - Variable cycles, staggered schemes, multiple MS1 types, calibration scans, or per-event timing.

   Model an ordered cycle explicitly:

   ```rust
   struct AcquisitionScheme {
       cycle: Vec<AcquisitionEvent>,
       repeat: RepeatPolicy,
       run_duration_s: f64,
   }

   enum AcquisitionEvent {
       Ms1(Ms1Event),
       DiaMs2(DiaMs2Event),
   }
   ```

   Each event should carry relative timing or duration, analyzer/type, m/z range, isolation, and CE policy. For timsTOF, an MS2 event can contain multiple mobility slices belonging to one frame.

2. **Split mobility from the common isolation window.** `Option<(u32,u32)>` is mechanically workable but semantically weak. Scan numbers are Bruker-grid coordinates requiring the reference dataset’s calibration and scan count; they are not a physical mobility range. It also permits invalid states such as mobility on Astral.

   Prefer:

   ```rust
   enum DiaGeometry {
       MzOnly,
       TimsMobility {
           scan_start: u32,
           scan_end: u32,
           // optionally inverse-mobility bounds
       },
   }
   ```

   Better still, put geometry on a timsTOF frame/event rather than every generic window. Keep `IsolationWindow` strictly m/z-based.

3. **Backward compatibility needs an explicit adapter, not an assertion of parity.** Existing code requires six concrete columns and uses `window_group` to assign frames, not merely identify windows ([acquisition.py](/scratch/timsim-demo/SUBMISSION/rustims/packages/imspy-simulation/src/imspy_simulation/acquisition.py:199), [handle.rs](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/handle.rs:181)). The migration must preserve:
   - Exact group IDs and row order.
   - Inclusive/exclusive meaning of `scan_start`/`scan_end`.
   - Integer types and column names.
   - CE rounding behavior.
   - Existing CSV and `use_reference_ds_layout` constructor paths.
   - `dia_ms_ms_info` frame-to-group sequencing.

   Add `scheme.to_bruker_tables()`, and retain the existing Python API as a compatibility wrapper. Golden-test both SQLite tables byte-for-value against the current path. Also fix or test the current ordering bug where `num_frames` is calculated before reference-derived `rt_cycle_length` replaces it.

4. **MS1 is part of acquisition design, not purely writer behavior.** It determines cadence, analyzer, resolution/transient time, profile/centroid representation, mass range, and cycle duration. At minimum, add `Ms1Event { analyzer, mz_range, duration_s, data_mode }`. “One precursor every N positions” is unsuitable terminology outside Bruker and becomes redundant with an explicit event sequence.

5. **CE must be a policy, not mandatory `f64`.** Represent `CollisionEnergyPolicy::{Fixed, PerWindow, Linear{...}, VendorModel{...}, Unknown}`. Silently inventing SCIEX CE would make “real-method replication” misleading. Require a user model/table when extraction cannot recover it, unless an explicitly labeled default simulation mode is selected.

6. **The input/output split is correct, but the current writer contract is incomplete.** `AcquisitionScheme` describes intended events; `AcquisitionWriter` materializes scans. However, the existing Thermo writer ignores descriptor RT, isolation, and CE and only replaces peak packets ([acquisition.rs](/scratch/timsim-demo/SUBMISSION/rustims/rustdf/src/sim/acquisition.rs:119)). Therefore the claimed arbitrary round trip is only valid when the generated scheme exactly matches the template schedule. Writers need either:
   - Metadata-authoring support, or
   - `validate_scheme(&scheme)` that rejects incompatible event count/order/metadata.

**Open Questions**

- **Blockers now:** 2 (core representation), 5 (MS1/cycle semantics), 6 (correct Thermo extraction), and 1 for faithful SCIEX output.
- **Must be specified now but implementation can follow:** 3. Overlap needs no special isolation algorithm; windows should independently include ions in their bounds, including duplicated coverage. Define boundary conventions and preserve acquisition order.
- **Deferrable:** 4. Bruker extraction may remain Python, provided both paths produce the same versioned Rust DTO and golden tables.

**Missing Design Elements**

Add event-level timing, cycle ordering, CE policy, analyzer/data mode, isolation boundary conventions, optional dwell/transient time, validation rules, scheme versioning, and provenance indicating whether values were extracted, inferred, or user-supplied. `cycle_time_s` alone cannot faithfully model RT sampling or instrument duty cycle.
