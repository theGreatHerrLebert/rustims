import sqlite3
from typing import List

from imspy.timstof.slice import TimsSlice

from imspy.simulation.annotation import RustWrapperObject
from imspy.timstof.data import TimsDataset
import pandas as pd
import numpy as np
import warnings
from typing import Optional

import imspy_connector

from imspy.timstof.frame import TimsFrame

ims = imspy_connector.py_dia

class Fit1D:
    """Light wrapper around ims.PyFit1D (read-only)."""
    def __init__(self, py):
        self._py = py

    @property
    def mu(self) -> float: return self._py.mu
    @property
    def sigma(self) -> float: return self._py.sigma
    @property
    def height(self) -> float: return self._py.height
    @property
    def baseline(self) -> float: return self._py.baseline
    @property
    def area(self) -> float: return self._py.area
    @property
    def r2(self) -> float: return self._py.r2
    @property
    def n(self) -> int: return self._py.n

    def to_dict(self, prefix: str):
        return {
            f"{prefix}_mu": self.mu,
            f"{prefix}_sigma": self.sigma,
            f"{prefix}_height": self.height,
            f"{prefix}_baseline": self.baseline,
            f"{prefix}_area": self.area,
            f"{prefix}_r2": self.r2,
            f"{prefix}_n": self.n,
        }

    def __repr__(self):
        return repr(self._py)


class RawPoints:
    """Wrapper around ims.PyRawPoints."""
    def __init__(self, py):
        self._py = py

    @property
    def n(self) -> int:
        return self._py.len

    def arrays(self):
        """Return numpy arrays (mz, rt, im, scan, intensity, tof, frame)."""
        return tuple(np.asarray(arr) for arr in self._py.to_arrays())

    def to_dataframe(self) -> pd.DataFrame:
        mz, rt, im, scan, inten, tof, frame = self.arrays()
        return pd.DataFrame({
            "mz": mz, "rt": rt, "im": im, "scan": scan,
            "intensity": inten, "tof": tof, "frame": frame
        })

    def to_tims_slice(self) -> TimsSlice:
        mz, rt, im, scan, intensity, tof, frame = self.arrays()
        return TimsSlice(
            frame_id=frame.astype(np.int32),
            scan=scan.astype(np.int32),
            tof=tof.astype(np.int32),
            retention_time=rt.astype(np.float64),
            mobility=im.astype(np.float64),
            mz=mz.astype(np.float64),
            intensity=intensity.astype(np.float64),
        )

    def __repr__(self):
        return f"RawPoints(n={self.n})"


class ClusterResult1D:
    """Ergonomic wrapper around ims.PyClusterResult1D."""
    def __init__(self, py):
        self._py = py

    # windows
    @property
    def rt_window(self): return self._py.rt_window
    @property
    def im_window(self): return self._py.im_window
    @property
    def mz_window(self): return self._py.mz_window

    # fits
    @property
    def rt_fit(self) -> Fit1D: return Fit1D(self._py.rt_fit)
    @property
    def im_fit(self) -> Fit1D: return Fit1D(self._py.im_fit)
    @property
    def mz_fit(self) -> Fit1D: return Fit1D(self._py.mz_fit)

    # stats / provenance
    @property
    def raw_sum(self) -> float: return self._py.raw_sum
    @property
    def volume_proxy(self) -> float: return self._py.volume_proxy
    @property
    def ms_level(self) -> int: return self._py.ms_level
    @property
    def window_group(self): return self._py.window_group
    @property
    def parent_im_id(self): return self._py.parent_im_id
    @property
    def parent_rt_id(self): return self._py.parent_rt_id

    # axes (may be None)
    def frame_ids_used(self) -> np.ndarray:
        return np.asarray(self._py.frame_ids_used())
    def rt_axis_sec(self) -> np.ndarray | None:
        arr = self._py.rt_axis_sec()
        return None if arr is None else np.asarray(arr, dtype=np.float32)
    def im_axis_scans(self) -> np.ndarray | None:
        arr = self._py.im_axis_scans()
        return None if arr is None else np.asarray(arr, dtype=np.int64)
    def mz_axis_da(self) -> np.ndarray | None:
        arr = self._py.mz_axis_da()
        return None if arr is None else np.asarray(arr, dtype=np.float32)

    # optional raw points
    def raw_points(self) -> RawPoints | None:
        rp = self._py.raw_points()
        return None if rp is None else RawPoints(rp)

    def to_dict(self) -> dict:
        d = {
            "ms_level": self.ms_level,
            "window_group": self.window_group,
            "parent_im_id": self.parent_im_id,
            "parent_rt_id": self.parent_rt_id,
            "rt_lo": self.rt_window[0],
            "rt_hi": self.rt_window[1],
            "im_lo": self.im_window[0],
            "im_hi": self.im_window[1],
            "mz_lo": self.mz_window[0],
            "mz_hi": self.mz_window[1],
            "raw_sum": self.raw_sum,
            "volume_proxy": self.volume_proxy,
        }
        d.update(self.rt_fit.to_dict("rt"))
        d.update(self.im_fit.to_dict("im"))
        d.update(self.mz_fit.to_dict("mz"))
        return d

    def to_dataframe_row(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def __repr__(self):
        return repr(self._py)

def iter_window_batches(plan, batch_size: int):
    """Yield successive batches of window indices from the plan."""
    n = plan.num_windows
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        yield list(range(i, j))
        i = j

class MzScanWindowGrid(RustWrapperObject):
    """Python wrapper for ims.PyMzScanWindowGrid."""

    def __init__(self, *a, **k):
        raise RuntimeError("Use MzScanWindowGrid.from_py_ptr()")

    @classmethod
    def from_py_ptr(cls, p: ims.PyMzScanWindowGrid) -> "MzScanWindowGrid":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    # --- properties mapped to Rust getters ---
    @property
    def rt_range_frames(self) -> tuple[int, int]:
        return self.__py_ptr.rt_range_frames

    @property
    def rt_range_sec(self) -> tuple[float, float]:
        return self.__py_ptr.rt_range_sec

    @property
    def rows(self) -> int:
        return self.__py_ptr.rows

    @property
    def cols(self) -> int:
        return self.__py_ptr.cols

    @property
    def scans(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.scans, dtype=np.uint32)

    @property
    def data(self) -> np.ndarray:
        # Fortran order on Rust side; keep order='F' for cheap reshape-friendly view
        return np.asarray(self.__py_ptr.data, dtype=np.float32, order="F")

    @property
    def data_raw(self) -> np.ndarray | None:
        raw = self.__py_ptr.data_raw
        if raw is None:
            return None
        return np.asarray(raw, dtype=np.float32, order="F")

    @property
    def frame_id_bounds(self) -> tuple[int, int]:
        return self.__py_ptr.frame_id_bounds

    @property
    def window_group(self) -> int | None:
        return self.__py_ptr.window_group

    def pick_im_peaks(self, **kw):
        rows = self.__py_ptr.pick_im_peaks(**kw)
        return [[ImPeak1D.from_py_ptr(p) for p in row] for row in rows]

    def __repr__(self) -> str:
        (l, r) = self.rt_range_frames
        (t0, t1) = self.rt_range_sec
        (fid_lo, fid_hi) = self.frame_id_bounds
        wg = self.window_group
        return (f"MzScanWindowGrid(frames=({l},{r}) ids=({fid_lo},{fid_hi}) "
                f"group={wg}, rt=({t0:.3f},{t1:.3f})s, shape=({self.rows},{self.cols}))")

class MzScanPlanGroup(RustWrapperObject):
    """Python wrapper for ims.PyMzScanPlanGroup (iterable over MzScanWindowGrid)."""

    def __init__(self, *a, **k):
        raise RuntimeError("Use TimsDatasetDIA.plan_mz_scan_windows_for_group(...) or MzScanPlanGroup.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: ims.PyMzScanPlanGroup) -> "MzScanPlanGroup":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    # ---- metadata exposed by Rust getters ----
    @property
    def rows(self) -> int:
        return self.__py_ptr.rows

    @property
    def global_num_scans(self) -> int:
        return self.__py_ptr.global_num_scans

    @property
    def num_windows(self) -> int:
        return self.__py_ptr.num_windows

    @property
    def window_group(self) -> int:
        return self.__py_ptr.window_group

    @property
    def frame_times(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.frame_times, dtype=np.float32)

    @property
    def frame_ids(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.frame_ids, dtype=np.uint32)

    @property
    def mz_centers(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.mz_centers, dtype=np.float32)

    def bounds(self, i: int) -> tuple[int, int] | None:
        return self.__py_ptr.bounds(i)

    def bounds_frame_ids(self, i: int) -> tuple[int, int] | None:
        return self.__py_ptr.bounds_frame_ids(i)

    @property
    def fragment_frame_id_bounds(self) -> tuple[int, int] | None:
        return self.__py_ptr.fragment_frame_id_bounds

    # ---- materialization / iteration ----
    def get(self, i: int) -> MzScanWindowGrid | None:
        w = self.__py_ptr.get(i)
        if w is None:
            return None
        return MzScanWindowGrid.from_py_ptr(w)

    def __iter__(self):
        for w in self.__py_ptr:
            yield MzScanWindowGrid.from_py_ptr(w)

    def __len__(self) -> int:
        return self.__py_ptr.__len__()

    def __getitem__(self, key):
        res = self.__py_ptr.__getitem__(key)
        if isinstance(res, list):
            return [MzScanWindowGrid.from_py_ptr(w) for w in res]
        return MzScanWindowGrid.from_py_ptr(res)

    def get_batch(self, start: int, count: int) -> list["MzScanWindowGrid"]:
        grids = self.__py_ptr.get_batch(int(start), int(count))
        return [MzScanWindowGrid.from_py_ptr(g) for g in grids]

    def pick_im_peaks_batched(
            self,
            indices: list[int],
            *,
            min_prom: float = 50.0,
            min_distance_scans: int = 2,
            min_width_scans: int = 2,
            use_mobility: bool = False,
    ) -> list[list[list["ImPeak1D"]]]:
        rows = self.__py_ptr.pick_im_peaks_for_indices(
            [int(i) for i in indices],
            float(min_prom),
            int(min_distance_scans),
            int(min_width_scans),
            bool(use_mobility),
        )
        return [[[ImPeak1D.from_py_ptr(p) for p in row] for row in win] for win in rows]

class MzScanPlan(RustWrapperObject):
    """Python wrapper for ims.PyMzScanPlan (iterable over MzScanWindowGrid)."""

    def __init__(self, *a, **k):
        raise RuntimeError("Use TimsDatasetDIA.plan_mz_scan_windows(...) or MzScanPlan.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: ims.PyMzScanPlan) -> "MzScanPlan":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    # ---- metadata from Rust getters on the plan ----
    @property
    def rows(self) -> int:
        return self.__py_ptr.rows

    @property
    def global_num_scans(self) -> int:
        return self.__py_ptr.global_num_scans

    @property
    def num_windows(self) -> int:
        return self.__py_ptr.num_windows

    @property
    def frame_times(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.frame_times, dtype=np.float32)

    @property
    def frame_ids(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.frame_ids, dtype=np.uint32)

    @property
    def mz_centers(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.mz_centers, dtype=np.float32)

    def bounds(self, i: int) -> tuple[int, int] | None:
        return self.__py_ptr.bounds(i)

    # ---- materialization / iteration ----
    def get(self, i: int) -> MzScanWindowGrid | None:
        w = self.__py_ptr.get(i)
        if w is None:
            return None
        return MzScanWindowGrid.from_py_ptr(w)

    def __iter__(self):
        # The Rust class implements __iter__/__next__, yielding PyMzScanWindowGrid.
        for w in self.__py_ptr:
            yield MzScanWindowGrid.from_py_ptr(w)

    # passthroughs
    def __len__(self) -> int:
        return self.__py_ptr.__len__()  # or just len(self.__py_ptr)

    def __getitem__(self, key):
        """Return MzScanWindowGrid or list[MzScanWindowGrid] depending on index/slice."""
        res = self.__py_ptr.__getitem__(key)
        if isinstance(res, list):
            return [MzScanWindowGrid.from_py_ptr(w) for w in res]
        return MzScanWindowGrid.from_py_ptr(res)

    # new helpers
    def bounds_frame_ids(self, i: int) -> tuple[int, int] | None:
        return self.__py_ptr.bounds_frame_ids(i)

    @property
    def precursor_frame_id_bounds(self) -> tuple[int, int] | None:
        return self.__py_ptr.precursor_frame_id_bounds

    def get_batch(self, start: int, count: int) -> list["MzScanWindowGrid"]:
        grids = self.__py_ptr.get_batch(int(start), int(count))
        return [MzScanWindowGrid.from_py_ptr(g) for g in grids]

    def pick_im_peaks_batched(
            self,
            indices: list[int],
            *,
            min_prom: float = 50.0,
            min_distance_scans: int = 2,
            min_width_scans: int = 2,
            use_mobility: bool = False,
    ) -> list[list[list["ImPeak1D"]]]:
        rows = self.__py_ptr.pick_im_peaks_for_indices(
            [int(i) for i in indices],
            float(min_prom),
            int(min_distance_scans),
            int(min_width_scans),
            bool(use_mobility),
        )
        # wrap Py objects
        return [[[ImPeak1D.from_py_ptr(p) for p in row] for row in win] for win in rows]

class RtPeak1D(RustWrapperObject):
    """Python wrapper for Rust PyRtPeak1D (read-only)."""
    def __init__(self, *a, **k):
        raise RuntimeError("RtPeak1D is created in Rust; use RtPeak1D.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: ims.PyRtPeak1D) -> "RtPeak1D":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    # --- geometry in RT index/time
    @property
    def rt_idx(self) -> int: return self.__py_ptr.rt_idx
    @property
    def rt_sec(self) -> float | None: return self.__py_ptr.rt_sec
    @property
    def apex_smoothed(self) -> float: return self.__py_ptr.apex_smoothed
    @property
    def apex_raw(self) -> float: return self.__py_ptr.apex_raw
    @property
    def prominence(self) -> float: return self.__py_ptr.prominence
    @property
    def left_x(self) -> float: return self.__py_ptr.left_x
    @property
    def right_x(self) -> float: return self.__py_ptr.right_x
    @property
    def width_frames(self) -> int: return self.__py_ptr.width_frames
    @property
    def area_raw(self) -> float: return self.__py_ptr.area_raw
    @property
    def subframe(self) -> float: return self.__py_ptr.subframe

    # --- provenance / bounds
    @property
    def rt_bounds_frames(self) -> tuple[int, int]: return self.__py_ptr.rt_bounds_frames
    @property
    def frame_id_bounds(self) -> tuple[int, int]: return self.__py_ptr.frame_id_bounds
    @property
    def window_group(self) -> int | None: return self.__py_ptr.window_group

    # --- m/z context
    @property
    def mz_row(self) -> int: return self.__py_ptr.mz_row
    @property
    def mz_center(self) -> float: return self.__py_ptr.mz_center
    @property
    def mz_bounds(self) -> tuple[float, float]: return self.__py_ptr.mz_bounds

    # --- linkage
    @property
    def parent_im_id(self) -> int | None: return self.__py_ptr.parent_im_id
    @property
    def id(self) -> int: return self.__py_ptr.id

    def get_py_ptr(self):
        return self.__py_ptr

    def __repr__(self):
        return repr(self.__py_ptr)

class ImPeak1D(RustWrapperObject):
    """Python wrapper for Rust PyImPeak1D (read-only)."""
    def __init__(self, *a, **k):
        raise RuntimeError("ImPeak1D is created in Rust; use ImPeak1D.from_py_ptr().")

    @property
    def id(self) -> int: return self.__py_ptr.id
    # --- New preferred / extra context fields ---
    @property
    def mz_row(self) -> int: return self.__py_ptr.mz_row

    @property
    def mz_center(self) -> float: return self.__py_ptr.mz_center

    @property
    def mz_bounds(self) -> tuple[float, float]: return self.__py_ptr.mz_bounds

    @property
    def rt_bounds(self) -> tuple[int, int]: return self.__py_ptr.rt_bounds
    @property
    def frame_id_bounds(self) -> tuple[int, int]: return self.__py_ptr.frame_id_bounds
    @property
    def window_group(self) -> int | None: return self.__py_ptr.window_group

    # --- Existing fields (note: mobility is Optional now) ---
    @property
    def scan(self) -> int: return self.__py_ptr.scan
    @property
    def mobility(self) -> float | None: return self.__py_ptr.mobility
    @property
    def apex_smoothed(self) -> float: return self.__py_ptr.apex_smoothed
    @property
    def apex_raw(self) -> float: return self.__py_ptr.apex_raw
    @property
    def prominence(self) -> float: return self.__py_ptr.prominence
    @property
    def left(self) -> int: return self.__py_ptr.left
    @property
    def right(self) -> int: return self.__py_ptr.right
    @property
    def left_x(self) -> float: return self.__py_ptr.left_x
    @property
    def right_x(self) -> float: return self.__py_ptr.right_x
    @property
    def width_scans(self) -> int: return self.__py_ptr.width_scans
    @property
    def area_raw(self) -> float: return self.__py_ptr.area_raw
    @property
    def subscan(self) -> float: return self.__py_ptr.subscan

    def get_py_ptr(self):
        return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, p: ims.PyImPeak1D) -> "ImPeak1D":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def __repr__(self):
        rt_lo, rt_hi = self.rt_bounds
        fid_lo, fid_hi = self.frame_id_bounds
        return (
            "ImPeak1D(mz_row={mz_row}, scan={scan}, mobility={mob}, "
            "apex_smoothed={apx:.1f}, prominence={prom:.1f}, width_scans={w}, area_raw={area:.1f}, "
            "left={l}, right={r}, left_x={lx:.3f}, right_x={rx:.3f}, subscan={sub:.3f}, "
            "rt_bounds=({rtl},{rth}), frame_id_bounds=({fl},{fh}), "
            "window_group={wg})"
        ).format(
            mz_row=self.mz_row, scan=self.scan, mob=self.mobility,
            apx=self.apex_smoothed, prom=self.prominence, w=self.width_scans, area=self.area_raw,
            l=self.left, r=self.right, lx=self.left_x, rx=self.right_x, sub=self.subscan,
            rtl=rt_lo, rth=rt_hi, fl=fid_lo, fh=fid_hi,
            wg=self.window_group,
        )


class TimsDatasetDIA(TimsDataset, RustWrapperObject):
    def __init__(self, data_path: str, in_memory: bool = False, use_bruker_sdk: bool = True):
        super().__init__(data_path=data_path, in_memory=in_memory, use_bruker_sdk=use_bruker_sdk)
        self.__dataset = ims.PyTimsDatasetDIA(self.data_path, self.binary_path, in_memory, self.use_bruker_sdk)

    @property
    def dia_ms_ms_windows(self):
        """Get PASEF meta data for DIA.

        Returns:
            pd.DataFrame: PASEF meta data.
        """
        return pd.read_sql_query("SELECT * from DiaFrameMsMsWindows",
                                 sqlite3.connect(self.data_path + "/analysis.tdf"))

    @property
    def dia_ms_ms_info(self):
        """Get DIA MS/MS info.

        Returns:
            pd.DataFrame: DIA MS/MS info.
        """
        return pd.read_sql_query("SELECT * from DiaFrameMsMsInfo",
                                 sqlite3.connect(self.data_path + "/analysis.tdf"))

    def sample_precursor_signal(self, num_frames: int, max_intensity: float = 25.0, take_probability: float = 0.5) -> TimsFrame:
        """Sample precursor signal.

        Args:
            num_frames: Number of frames.
            max_intensity: Maximum intensity.
            take_probability: Probability to take signals from sampled frames.

        Returns:
            TimsFrame: Frame.
        """

        assert num_frames > 0, "Number of frames must be greater than 0."
        assert 0 < take_probability <= 1, " Probability to take signals from sampled frames must be between 0 and 1."

        return TimsFrame.from_py_ptr(self.__dataset.sample_precursor_signal(num_frames, max_intensity, take_probability))

    def sample_fragment_signal(self, num_frames: int, window_group: int, max_intensity: float = 25.0, take_probability: float = 0.5) -> TimsFrame:
        """Sample fragment signal.

        Args:
            num_frames: Number of frames.
            window_group: Window group to take frames from.
            max_intensity: Maximum intensity.
            take_probability: Probability to take signals from sampled frames.

        Returns:
            TimsFrame: Frame.
        """

        assert num_frames > 0, "Number of frames must be greater than 0."
        assert 0 < take_probability <= 1, " Probability to take signals from sampled frames must be between 0 and 1."

        return TimsFrame.from_py_ptr(self.__dataset.sample_fragment_signal(num_frames, window_group, max_intensity, take_probability))

    def read_compressed_data_full(self) -> List[bytes]:
        """Read compressed data.

        Returns:
            List[bytes]: Compressed data.
        """
        return self.__dataset.read_compressed_data_full()

    @classmethod
    def from_py_ptr(cls, obj):
        instance = cls.__new__(cls)
        instance.__dataset = obj
        return instance

    def get_py_ptr(self):
        return self.__dataset

    def plan_mz_scan_windows(
        self,
        *,
        ppm_per_bin: float,
        mz_pad_ppm: float,
        rt_window_sec: float,
        rt_hop_sec: float,
        num_threads: int = 4,
        im_sigma_scans: Optional[float] = None,
        truncate: float = 3.0,
        precompute_views: bool = False,
    ) -> MzScanPlan:
        """Build a planner for sliding RT windows with a constant-ppm m/z grid."""
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, forcing num_threads=1.")
            num_threads = 1

        py_plan = ims.PyMzScanPlan(
            self.__dataset,        # pass the PyTimsDatasetDIA handle
            ppm_per_bin,
            mz_pad_ppm,
            rt_window_sec,
            rt_hop_sec,
            num_threads,
            im_sigma_scans,
            truncate,
            precompute_views,
        )
        return MzScanPlan.from_py_ptr(py_plan)

    def plan_mz_scan_windows_for_group(
        self,
        window_group: int,
        *,
        ppm_per_bin: float,
        mz_pad_ppm: float,
        rt_window_sec: float,
        rt_hop_sec: float,
        num_threads: int = 4,
        im_sigma_scans: Optional[float] = None,
        truncate: float = 3.0,
        precompute_views: bool = False,
        clamp_mz_to_group: bool = True,
    ) -> MzScanPlanGroup:
        """
        Build a planner for sliding RT windows on fragment (MS2) frames belonging
        to a specific DIA window group, using a constant-ppm m/z grid.

        Returns:
            MzScanPlanGroup: iterable of MzScanWindowGrid
        """
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, forcing num_threads=1.")
            num_threads = 1

        py_plan = ims.PyMzScanPlanGroup(
            self.__dataset,          # PyTimsDatasetDIA handle
            int(window_group),
            float(ppm_per_bin),
            float(mz_pad_ppm),
            float(rt_window_sec),
            float(rt_hop_sec),
            int(num_threads),
            im_sigma_scans,          # None or float
            float(truncate),
            bool(precompute_views),
            bool(clamp_mz_to_group),
        )
        return MzScanPlanGroup.from_py_ptr(py_plan)

    def expand_rt_for_im_peaks_in_precursor(
            self,
            im_peaks: list["ImPeak1D"],
            *,
            bin_pad: int = 0,
            smooth_sigma: float = 1.25,
            smooth_trunc: float = 3.0,
            min_prom: float = 50.0,
            min_sep_frames: int = 2,
            min_width_frames: int = 2,
            ppm_per_bin: float = 10.0,
            fallback_if_frames_lt: int = 3,
            fallback_frac_width: float = 0.10
    ) -> list[list["RtPeak1D"]]:
        nested = self.__dataset.expand_rt_for_im_peaks_in_precursor(
            [p.get_py_ptr() for p in im_peaks],
            int(bin_pad),
            float(smooth_sigma),
            float(smooth_trunc),
            float(min_prom),
            int(min_sep_frames),
            int(min_width_frames),
            float(ppm_per_bin),
            int(fallback_if_frames_lt),
            float(fallback_frac_width)
        )
        return [[RtPeak1D.from_py_ptr(r) for r in row] for row in nested]

    def expand_rt_for_im_peaks_in_group(
            self,
            window_group: int,
            im_peaks: list["ImPeak1D"],
            *,
            bin_pad: int = 0,
            smooth_sigma: float = 1.25,
            smooth_trunc: float = 3.0,
            min_prom: float = 50.0,
            min_sep_frames: int = 2,
            min_width_frames: int = 2,
            ppm_per_bin: float = 5.0,
            fallback_if_frames_lt: int = 3,
            fallback_frac_width: float = 0.10
    ) -> list[list["RtPeak1D"]]:
        nested = self.__dataset.expand_rt_for_im_peaks_in_group(
            int(window_group),
            [p.get_py_ptr() for p in im_peaks],
            int(bin_pad),
            float(smooth_sigma),
            float(smooth_trunc),
            float(min_prom),
            int(min_sep_frames),
            int(min_width_frames),
            float(ppm_per_bin),
            int(fallback_if_frames_lt),
            float(fallback_frac_width)
        )
        return [[RtPeak1D.from_py_ptr(r) for r in row] for row in nested]

    def clusters_for_group(
            self,
            window_group: int,
            im_peaks: list["ImPeak1D"],
            *,
            ppm_per_bin: float = 5.0,
            # RtExpandParams
            bin_pad: int = 0,
            smooth_sigma: float = 1.25,
            smooth_trunc: float = 3.0,
            min_prom: float = 50.0,
            min_sep_frames: int = 2,
            min_width_frames: int = 2,
            fallback_if_frames_lt: int = 5,
            fallback_frac_width: float = 0.5,
            # BuildSpecOpts
            extra_rt_pad: int = 0,
            extra_im_pad: int = 0,
            mz_ppm_pad: float = 5.0,
            mz_hist_bins: int = 64,
            # Eval1DOpts
            refine_mz_once: bool = True,
            refine_k_sigma: float = 3.0,
            attach_axes: bool = True,
            attach_points: bool = False,
            attach_max_points: int | None = None,
            # pairing + threads
            require_rt_overlap: bool = True,
            num_threads: int = 0,
    ):
        """Cluster in MS2 space for one DIA window group."""
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, forcing num_threads=1.")
            num_threads = 1

        py_results = self.__dataset.clusters_for_group(
            int(window_group),
            float(ppm_per_bin),
            [p.get_py_ptr() for p in im_peaks],
            int(bin_pad),
            float(smooth_sigma), float(smooth_trunc),
            float(min_prom), int(min_sep_frames), int(min_width_frames),
            int(fallback_if_frames_lt), float(fallback_frac_width),
            int(extra_rt_pad), int(extra_im_pad), float(mz_ppm_pad), int(mz_hist_bins),
            bool(refine_mz_once), float(refine_k_sigma),
            bool(attach_axes),
            bool(attach_points), attach_max_points,
            bool(require_rt_overlap), int(num_threads),
        )
        return [ClusterResult1D(r) for r in py_results]

    def clusters_for_precursor(
            self,
            im_peaks: list["ImPeak1D"],
            *,
            ppm_per_bin: float = 10.0,
            # RtExpandParams
            bin_pad: int = 0,
            smooth_sigma: float = 1.25,
            smooth_trunc: float = 3.0,
            min_prom: float = 50.0,
            min_sep_frames: int = 2,
            min_width_frames: int = 2,
            fallback_if_frames_lt: int = 5,
            fallback_frac_width: float = 0.5,
            # BuildSpecOpts
            extra_rt_pad: int = 0,
            extra_im_pad: int = 0,
            mz_ppm_pad: float = 5.0,
            mz_hist_bins: int = 64,
            # Eval1DOpts
            refine_mz_once: bool = True,
            refine_k_sigma: float = 3.0,
            attach_axes: bool = True,
            attach_points: bool = False,
            attach_max_points: int | None = None,
            # pairing + threads
            require_rt_overlap: bool = True,
            num_threads: int = 0,
    ):
        """Cluster in MS1 precursor space."""
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, forcing num_threads=1.")
            num_threads = 1

        py_results = self.__dataset.clusters_for_precursor(
            float(ppm_per_bin),
            [p.get_py_ptr() for p in im_peaks],
            int(bin_pad),
            float(smooth_sigma), float(smooth_trunc),
            float(min_prom), int(min_sep_frames), int(min_width_frames),
            int(fallback_if_frames_lt), float(fallback_frac_width),
            int(extra_rt_pad), int(extra_im_pad), float(mz_ppm_pad), int(mz_hist_bins),
            bool(refine_mz_once), float(refine_k_sigma),
            bool(attach_axes),
            bool(attach_points), attach_max_points,
            bool(require_rt_overlap), int(num_threads),
        )
        return [ClusterResult1D(r) for r in py_results]

from collections.abc import Iterable
from typing import List, Sequence

def stitch_im_peaks(
    peaks: Sequence[Sequence[Sequence['ImPeak1D']]],  # windows × rows × peaks
    min_overlap_frames: int = 1,
    max_scan_delta: int = 1,
    jaccard_min: float = 0.0,
    max_mz_row_delta: int = 0,
    allow_cross_groups: bool = False,
) -> List['ImPeak1D']:
    """
    Stitch IM 1D peaks across overlapping RT windows.

    Accepts either:
      - flat:   List[ImPeak1D]
      - nested: List[List[List[ImPeak1D]]] (windows × rows × peaks)

    Returns:
      flat, deduplicated List[ImPeak1D]
    """
    from .dia import ImPeak1D

    # empty fast-path
    if not peaks:
        return []

    batched_ptrs = [
        [[p.get_py_ptr() for p in row] for row in win]
        for win in peaks
    ]
    stitched_py = ims.stitch_im_peaks_batched_streaming(
        batched_ptrs,
        int(min_overlap_frames),
        int(max_scan_delta),
        float(jaccard_min),
        int(max_mz_row_delta),
        bool(allow_cross_groups),
    )
    return [ImPeak1D.from_py_ptr(p) for p in stitched_py]
