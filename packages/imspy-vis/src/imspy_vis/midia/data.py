"""MIDIA data layer reconstructed on top of imspy-core's DIA reader.

This replaces the old C++/opentims ``proteolizard-midia`` stack (``MidiaExperimentCursor``,
``MidiaSlice``, ``FragmentCoords4D`` ...). The acquisition is read with
:class:`imspy_core.timstof.dia.TimsDatasetDIA`; everything MIDIA-specific (cycle grouping,
the 4D fragment coordinate system, and the extraction-window map) is derived here in pure
Python/NumPy from the run's own metadata tables — no separate ``extraction_windows.h5``.

A MIDIA cycle is one precursor (MS1) frame followed by ``K`` fragment frames, each fragment
frame being one quadrupole **step** (its ``WindowGroup``). The 4 fragment dimensions are
therefore ``(cycle, step, scan, mz)`` carrying ``intensity``; precursors are ``(cycle, scan,
mz)``. The extraction window maps a ``(step, scan)`` pair back to the precursor isolation m/z
that produced it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from imspy_core.timstof.dia import TimsDatasetDIA


class MidiaExperiment:
    """A MIDIA/DIA-PASEF run, with cycle structure and extraction windows derived once."""

    def __init__(self, data_path: str, use_bruker_sdk: bool = False):
        self.data_path = data_path
        self.dataset = TimsDatasetDIA(data_path, in_memory=False, use_bruker_sdk=use_bruker_sdk)

        # Per-frame metadata (Id is contiguous 1..N). Time is retention time in seconds.
        meta = self.dataset.get_table("Frames")[["Id", "Time", "MsMsType"]].copy()
        meta = meta.sort_values("Id").reset_index(drop=True)
        self.meta = meta

        # Fragment frame -> quadrupole step (its WindowGroup). MS1 frames are absent here.
        info = self.dataset.dia_ms_ms_info
        self.frame_to_step = dict(zip(info.Frame.to_numpy(), info.WindowGroup.to_numpy()))

        # Extraction windows: (WindowGroup, scan range) -> isolation m/z. Kept as per-group
        # sorted arrays for a vectorized scan->isolation lookup (see ``isolation_left_bound``).
        self.windows = self.dataset.dia_ms_ms_windows
        self._win_by_group: dict[int, dict[str, np.ndarray]] = {}
        for grp, g in self.windows.groupby("WindowGroup"):
            g = g.sort_values("ScanNumBegin")
            self._win_by_group[int(grp)] = {
                "begin": g.ScanNumBegin.to_numpy(),
                "end": g.ScanNumEnd.to_numpy(),
                "left": (g.IsolationMz - g.IsolationWidth / 2.0).to_numpy(),
            }

        # Cycle index per frame id. A cycle opens on each precursor (MS1) frame and runs until
        # the next one; the precursor and all its fragment frames share that cycle index.
        is_precursor = meta.MsMsType.to_numpy() == 0
        ids = meta.Id.to_numpy()
        cycle_of_id = np.cumsum(is_precursor) - 1  # 0-based, aligned to sorted ids
        self._cycle_of_frame = dict(zip(ids, cycle_of_id))
        self.precursor_frame_ids = ids[is_precursor]
        self.rt_of_frame = dict(zip(ids, meta.Time.to_numpy()))
        self.n_cycles = int(cycle_of_id.max()) + 1

    # -- frame selection ----------------------------------------------------------------
    def frames_in_rt(self, rt_start_s: float, rt_stop_s: float) -> np.ndarray:
        """All frame ids whose retention time falls in ``[rt_start_s, rt_stop_s]``."""
        m = self.meta
        mask = (m.Time >= rt_start_s) & (m.Time <= rt_stop_s)
        return m.loc[mask, "Id"].to_numpy().astype(np.int32)

    def get_slice_retention_time(self, rt_start_s: float, rt_stop_s: float) -> "MidiaSlice":
        frame_ids = self.frames_in_rt(rt_start_s, rt_stop_s)
        return MidiaSlice(self, frame_ids)

    # -- extraction-window lookup -------------------------------------------------------
    def isolation_left_bound(self, step: int, scans: np.ndarray) -> np.ndarray:
        """Vectorized ``(step, scan) -> precursor isolation-window left bound``.

        Returns ``NaN`` for scans that fall outside every window of the step.
        """
        win = self._win_by_group.get(int(step))
        out = np.full(scans.shape, np.nan, dtype=np.float64)
        if win is None:
            return out
        # Windows within a group are disjoint scan ranges; assign each scan the window that
        # contains it. Few windows per group (<=3 here) so a small loop is clearest and fast.
        for begin, end, left in zip(win["begin"], win["end"], win["left"]):
            m = (scans >= begin) & (scans <= end)
            out[m] = left
        return out


class MidiaSlice:
    """A retention-time slice of a :class:`MidiaExperiment`, loaded lazily.

    Exposes the two coordinate views the clustering needs:
    :meth:`precursor_coords3D` -> ``cycle, scan, mz, intensity`` and
    :meth:`fragment_coords4D` -> ``cycle, step, scan, mz, intensity``.
    """

    def __init__(self, experiment: MidiaExperiment, frame_ids: np.ndarray):
        self.exp = experiment
        self.frame_ids = np.asarray(frame_ids, dtype=np.int32)
        self._df: pd.DataFrame | None = None

    # -- raw point loading --------------------------------------------------------------
    def _load(self) -> pd.DataFrame:
        """Load all points for the slice once and tag each with cycle + step."""
        if self._df is not None:
            return self._df
        if len(self.frame_ids) == 0:
            self._df = _empty_points()
            return self._df
        # Batched, multi-threaded read of the whole slice in one call.
        df = self.exp.dataset.get_tims_slice(self.frame_ids).df
        df = df[["frame", "scan", "mz", "intensity"]].copy()
        df["cycle"] = df["frame"].map(self.exp._cycle_of_frame).astype("int64")
        # step = quadrupole window group for fragment frames, 0 for precursor (MS1) frames.
        df["step"] = df["frame"].map(self.exp.frame_to_step).fillna(0).astype("int64")
        self._df = df
        return df

    # -- coordinate views ---------------------------------------------------------------
    def precursor_coords3D(self) -> pd.DataFrame:
        """MS1 points as ``cycle, scan, mz, intensity`` (step == 0)."""
        df = self._load()
        out = df[df["step"] == 0][["cycle", "scan", "mz", "intensity"]].reset_index(drop=True)
        return out

    def fragment_coords4D(self) -> pd.DataFrame:
        """Fragment points as ``cycle, step, scan, mz, intensity`` (step >= 1)."""
        df = self._load()
        out = df[df["step"] >= 1][["cycle", "step", "scan", "mz", "intensity"]].reset_index(drop=True)
        return out

    # -- filtering ----------------------------------------------------------------------
    def filtered(self, mz_min: float = 0.0, mz_max: float = 2000.0,
                 scan_min: int = 0, scan_max: int = 1_000,
                 intensity_min: float = 1.0) -> "MidiaSlice":
        """Return a new slice sharing the loaded points but range-filtered."""
        df = self._load()
        m = ((df.mz >= mz_min) & (df.mz <= mz_max) &
             (df.scan >= scan_min) & (df.scan <= scan_max) &
             (df.intensity >= intensity_min))
        out = MidiaSlice(self.exp, self.frame_ids)
        out._df = df[m].reset_index(drop=True)
        return out


def _empty_points() -> pd.DataFrame:
    return pd.DataFrame({c: [] for c in ["frame", "scan", "mz", "intensity", "cycle", "step"]})
