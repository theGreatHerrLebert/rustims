from __future__ import annotations

from typing import Iterator
import numpy as np
import pandas as pd
from imspy.simulation.annotation import RustWrapperObject

import imspy_connector
ims = imspy_connector.py_dia

class Fit1D(RustWrapperObject):
    """Python wrapper around ims.PyFit1D."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use Fit1D.from_py_ptr(...) instead.")

    @classmethod
    def from_py_ptr(cls, p: ims.PyFit1D) -> "Fit1D":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self) -> ims.PyFit1D:
        return self.__py_ptr

    @property
    def mu(self) -> float:
        return self.__py_ptr.mu

    @property
    def sigma(self) -> float:
        return self.__py_ptr.sigma

    @property
    def height(self) -> float:
        return self.__py_ptr.height

    @property
    def baseline(self) -> float:
        return self.__py_ptr.baseline

    @property
    def area(self) -> float:
        return self.__py_ptr.area

    @property
    def r2(self) -> float:
        return self.__py_ptr.r2

    @property
    def n(self) -> int:
        return self.__py_ptr.n

    def __repr__(self) -> str:
        return repr(self.__py_ptr)

class RtPeak1D(RustWrapperObject):
    """Python wrapper around ims.PyRtPeak1D."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use RtPeak1D.from_py_ptr(...) instead.")

    @classmethod
    def from_py_ptr(cls, p: ims.PyRtPeak1D) -> "RtPeak1D":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self) -> ims.PyRtPeak1D:
        return self.__py_ptr

    # geometry
    @property
    def rt_idx(self) -> int:
        return self.__py_ptr.rt_idx

    @property
    def rt_sec(self) -> Optional[float]:
        return self.__py_ptr.rt_sec

    @property
    def apex_smoothed(self) -> float:
        return self.__py_ptr.apex_smoothed

    @property
    def apex_raw(self) -> float:
        return self.__py_ptr.apex_raw

    @property
    def prominence(self) -> float:
        return self.__py_ptr.prominence

    @property
    def left_x(self) -> float:
        return self.__py_ptr.left_x

    @property
    def right_x(self) -> float:
        return self.__py_ptr.right_x

    @property
    def width_frames(self) -> int:
        return self.__py_ptr.width_frames

    @property
    def area_raw(self) -> float:
        return self.__py_ptr.area_raw

    @property
    def subframe(self) -> float:
        return self.__py_ptr.subframe

    # provenance / bounds
    @property
    def rt_bounds_frames(self) -> Tuple[int, int]:
        return self.__py_ptr.rt_bounds_frames

    @property
    def frame_id_bounds(self) -> Tuple[int, int]:
        return self.__py_ptr.frame_id_bounds

    @property
    def window_group(self) -> Optional[int]:
        return self.__py_ptr.window_group

    # TOF context
    @property
    def tof_row(self) -> int:
        return self.__py_ptr.tof_row

    @property
    def tof_center(self) -> int:
        return self.__py_ptr.tof_center

    @property
    def tof_bounds(self) -> Tuple[int, int]:
        return self.__py_ptr.tof_bounds

    # linkage
    @property
    def parent_im_id(self) -> Optional[int]:
        return self.__py_ptr.parent_im_id

    @property
    def id(self) -> int:
        return self.__py_ptr.id

    def __repr__(self) -> str:
        return repr(self.__py_ptr)

from typing import Dict, List, Optional, Tuple

class ImPeak1D(RustWrapperObject):
    """Python wrapper around ims.PyImPeak1D."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use ImPeak1D.from_py_ptr(...) or ImPeak1D.from_detected(...).")

    @classmethod
    def from_py_ptr(cls, p: ims.PyImPeak1D) -> "ImPeak1D":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self) -> ims.PyImPeak1D:
        return self.__py_ptr

    @classmethod
    def from_detected(
            cls,
            peaks: Dict[str, np.ndarray],
            idx: int,
            *,
            window_grid,
            plan_group=None,
            k_sigma: float = 3.0,
            min_width: int = 3,
    ) -> "ImPeak1D":
        """
        Build a single ImPeak1D from the dict-of-arrays produced by
        `iter_detect_peaks_from_blurred(...)`.

        Accepts either:
          - peaks["tof_row"], peaks["scan_idx"]
          - or peaks["i"], peaks["j"] for backward compatibility.

        Assumes:
          - peaks["mu_scan"], peaks["sigma_scan"]
          - peaks["amplitude"], peaks["baseline"], peaks["area"]
        """
        mu_scan = peaks["mu_scan"]
        sigma_scan = peaks["sigma_scan"]
        amp = peaks["amplitude"]
        base = peaks["baseline"]
        area = peaks["area"]

        # support both new and old names
        tof_row_arr = peaks.get("tof_row", peaks.get("i"))
        scan_idx_arr = peaks.get("scan_idx", peaks.get("j"))

        if tof_row_arr is None or scan_idx_arr is None:
            raise KeyError(
                "Expected tof_row/scan_idx or i/j in peaks dict, "
                f"got keys={list(peaks.keys())}"
            )

        n = mu_scan.shape[0]
        if idx < 0 or idx >= n:
            raise IndexError(f"idx {idx} out of range for {n} detected peaks")

        mu_scan_v = float(mu_scan[idx])
        sigma_scan_v = float(sigma_scan[idx])
        amp_v = float(amp[idx])
        base_v = float(base[idx])  # unused for now
        area_v = float(area[idx])

        tof_row = int(round(float(tof_row_arr[idx])))
        scan_idx = int(round(float(scan_idx_arr[idx])))

        # --- grid-level geometry -------------------------------------------
        rows = window_grid.rows
        scans_global = np.asarray(window_grid.scans, dtype=np.uint32)
        cols = scans_global.shape[0]

        if not (0 <= tof_row < rows):
            raise ValueError(f"tof_row={tof_row} outside grid rows={rows}")
        if not (0 <= scan_idx < cols):
            raise ValueError(f"scan_idx={scan_idx} outside grid cols={cols}")

        tof_centers = window_grid.tof_centers
        tof_edges = window_grid.tof_edges

        tof_center = int(round(float(tof_centers[tof_row])))

        if tof_row + 1 < tof_edges.shape[0]:
            tof_min = int(round(float(tof_edges[tof_row])))
            tof_max = int(round(float(tof_edges[tof_row + 1])))
        else:
            e_last = float(tof_edges[tof_row])
            width = (
                float(tof_edges[tof_row] - tof_edges[tof_row - 1])
                if tof_row > 0 else 1.0
            )
            tof_min = int(round(e_last - width))
            tof_max = int(round(e_last))

        tof_bounds: Tuple[int, int] = (tof_min, tof_max)

        rt_bounds: Tuple[int, int] = window_grid.rt_range_frames
        frame_id_bounds: Tuple[int, int] = window_grid.frame_id_bounds
        window_group = window_grid.window_group

        scan_abs = int(scans_global[scan_idx])

        # --- left/right from sigma_scan ------------------------------------
        half = max(min_width / 2.0, k_sigma * max(sigma_scan_v, 1e-3))
        left = int(max(0, np.floor(scan_idx - half)))
        right = int(min(cols - 1, np.ceil(scan_idx + half)))
        width_scans = max(1, right - left + 1)

        left_abs = int(scans_global[left])
        right_abs = int(scans_global[right])

        mobility: Optional[float] = None
        scan_sigma = float(sigma_scan_v) if np.isfinite(sigma_scan_v) else None

        apex_smoothed = amp_v
        apex_raw = amp_v
        prominence = amp_v

        left_x = float(left_abs)
        right_x = float(right_abs)
        subscan = float(mu_scan_v)

        wg_val = int(window_group) if window_group is not None else 0
        fid_lo = int(frame_id_bounds[0])
        peak_id = (
                (wg_val & 0xFF) << 56 |
                (fid_lo & 0xFFFF) << 40 |
                (tof_row & 0xFFFF) << 24 |
                (scan_abs & 0xFFFFFF)
        )

        py_peak = ims.PyImPeak1D(
            tof_row,
            tof_center,
            tof_bounds,
            rt_bounds,
            frame_id_bounds,
            window_group,
            scan_idx,
            left,
            right,
            scan_abs,
            left_abs,
            right_abs,
            scan_sigma,
            mobility,
            float(apex_smoothed),
            float(apex_raw),
            float(prominence),
            float(left_x),
            float(right_x),
            int(width_scans),
            float(area_v),
            float(subscan),
            int(peak_id),
        )

        return cls.from_py_ptr(py_peak)

    @classmethod
    def batch_from_detected(
            cls,
            peaks: Dict[str, np.ndarray],
            *,
            window_grid,
            plan_group=None,
            k_sigma: float = 3.0,
            min_width: int = 3,
    ) -> List["ImPeak1D"]:
        n = int(peaks["mu_scan"].shape[0])
        if n == 0:
            return []
        return [
            cls.from_detected(
                peaks,
                i,
                window_grid=window_grid,
                plan_group=plan_group,
                k_sigma=k_sigma,
                min_width=min_width,
            )
            for i in range(n)
        ]

    @property
    def id(self) -> int:
        return self.__py_ptr.id

    @property
    def tof_row(self) -> int:
        return self.__py_ptr.tof_row

    @property
    def tof_center(self) -> int:
        return self.__py_ptr.tof_center

    @property
    def tof_bounds(self) -> Tuple[int, int]:
        return self.__py_ptr.tof_bounds

    @property
    def rt_bounds(self) -> Tuple[int, int]:
        return self.__py_ptr.rt_bounds

    @property
    def frame_id_bounds(self) -> Tuple[int, int]:
        return self.__py_ptr.frame_id_bounds

    @property
    def window_group(self) -> Optional[int]:
        return self.__py_ptr.window_group

    @property
    def scan(self) -> int:
        return self.__py_ptr.scan

    @property
    def scan_sigma(self) -> Optional[float]:
        return self.__py_ptr.scan_sigma

    @property
    def mobility(self) -> Optional[float]:
        return self.__py_ptr.mobility

    @property
    def apex_smoothed(self) -> float:
        return self.__py_ptr.apex_smoothed

    @property
    def apex_raw(self) -> float:
        return self.__py_ptr.apex_raw

    @property
    def prominence(self) -> float:
        return self.__py_ptr.prominence

    @property
    def left(self) -> int:
        return self.__py_ptr.left

    @property
    def right(self) -> int:
        return self.__py_ptr.right

    @property
    def left_x(self) -> float:
        return self.__py_ptr.left_x

    @property
    def right_x(self) -> float:
        return self.__py_ptr.right_x

    @property
    def width_scans(self) -> int:
        return self.__py_ptr.width_scans

    @property
    def area_raw(self) -> float:
        return self.__py_ptr.area_raw

    @property
    def subscan(self) -> float:
        return self.__py_ptr.subscan

    @property
    def scan_abs(self) -> int:
        return self.__py_ptr.scan_abs

    @property
    def left_abs(self) -> int:
        return self.__py_ptr.left_abs

    @property
    def right_abs(self) -> int:
        return self.__py_ptr.right_abs

    def __repr__(self) -> str:
        return repr(self.__py_ptr)

class TofScanWindowGrid(RustWrapperObject):
    """Python wrapper around ims.PyTofScanWindowGrid."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use TofScanWindowGrid.from_py_ptr(...) instead.")

    @classmethod
    def from_py_ptr(cls, p: ims.PyTofScanWindowGrid) -> "TofScanWindowGrid":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self) -> ims.PyTofScanWindowGrid:
        return self.__py_ptr

    # meta
    @property
    def rt_range_frames(self) -> Tuple[int, int]:
        return self.__py_ptr.rt_range_frames

    @property
    def rt_range_sec(self) -> Tuple[float, float]:
        return self.__py_ptr.rt_range_sec

    @property
    def rows(self) -> int:
        return self.__py_ptr.rows

    @property
    def cols(self) -> int:
        return self.__py_ptr.cols

    @property
    def frame_id_bounds(self) -> Tuple[int, int]:
        return self.__py_ptr.frame_id_bounds

    @property
    def window_group(self) -> Optional[int]:
        return self.__py_ptr.window_group

    @property
    def scans(self) -> np.ndarray:
        """Global scan indices (uint32)."""
        return np.asarray(self.__py_ptr.scans)

    @property
    def data(self) -> np.ndarray:
        """
        Dense TOF×scan matrix (rows, cols), possibly smoothed.

        Note: calling this moves the data out of the Rust object.
        """
        return np.asarray(self.__py_ptr.data)

    @property
    def data_raw(self) -> Optional[np.ndarray]:
        """Optional raw (pre-smoothing) matrix."""
        arr = self.__py_ptr.data_raw
        return None if arr is None else np.asarray(arr)

    @property
    def tof_centers(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.tof_centers)

    @property
    def tof_edges(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.tof_edges)

    def __repr__(self) -> str:
        return repr(self.__py_ptr)

class TofScanPlan(RustWrapperObject):
    """Python wrapper around ims.PyTofScanPlan (MS1 / precursor)."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use TofScanPlan.from_py_ptr(...) or TimsDatasetDIA.plan_tof_scan_windows(...)")

    @classmethod
    def from_py_ptr(cls, p: ims.PyTofScanPlan) -> "TofScanPlan":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self) -> ims.PyTofScanPlan:
        return self.__py_ptr

    # basic meta
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
        return np.asarray(self.__py_ptr.frame_times)

    @property
    def frame_ids(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.frame_ids)

    @property
    def tof_centers(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.tof_centers)

    @property
    def tof_min(self) -> int:
        return self.__py_ptr.tof_min

    @property
    def tof_max(self) -> int:
        return self.__py_ptr.tof_max

    def tof_center_at(self, mu: float) -> float:
        return self.__py_ptr.tof_center_at(mu)

    def tof_index_of(self, tof: int) -> Optional[int]:
        return self.__py_ptr.tof_index_of(tof)

    def tof_center_for_row(self, row: int) -> float:
        return self.__py_ptr.tof_center_for_row(row)

    # bounds
    def bounds(self, i: int) -> Optional[Tuple[int, int]]:
        return self.__py_ptr.bounds(i)

    def bounds_frame_ids(self, i: int) -> Optional[Tuple[int, int]]:
        return self.__py_ptr.bounds_frame_ids(i)

    @property
    def precursor_frame_id_bounds(self) -> Optional[Tuple[int, int]]:
        return self.__py_ptr.precursor_frame_id_bounds

    # iteration / indexing
    def __len__(self) -> int:
        return len(self.__py_ptr)

    def __getitem__(self, idx) -> TofScanWindowGrid | list[TofScanWindowGrid]:
        res = self.__py_ptr.__getitem__(idx)
        # Could be single grid or a list
        if isinstance(res, ims.PyTofScanWindowGrid):
            return TofScanWindowGrid.from_py_ptr(res)
        # assume it's a Python list
        return [TofScanWindowGrid.from_py_ptr(g) for g in res]

    def __iter__(self) -> Iterator[TofScanWindowGrid]:
        for g in self.__py_ptr:
            yield TofScanWindowGrid.from_py_ptr(g)

    def get(self, i: int) -> Optional[TofScanWindowGrid]:
        res = self.__py_ptr.get(i)
        if res is None:
            return None
        return TofScanWindowGrid.from_py_ptr(res)

    def get_batch(self, start: int, count: int) -> list[TofScanWindowGrid]:
        grids = self.__py_ptr.get_batch(start, count)
        return [TofScanWindowGrid.from_py_ptr(g) for g in grids]

    def get_batch_par(self, start: int, count: int) -> list[TofScanWindowGrid]:
        grids = self.__py_ptr.get_batch_par(start, count)
        return [TofScanWindowGrid.from_py_ptr(g) for g in grids]

    def __repr__(self) -> str:
        return f"TofScanPlan(num_windows={self.num_windows}, rows={self.rows}, scans={self.global_num_scans})"

class TofScanPlanGroup(RustWrapperObject):
    """Python wrapper around ims.PyTofScanPlanGroup (DIA window group)."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use TofScanPlanGroup.from_py_ptr(...) or TimsDatasetDIA.plan_tof_scan_windows_for_group(...)")

    @classmethod
    def from_py_ptr(cls, p: ims.PyTofScanPlanGroup) -> "TofScanPlanGroup":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self) -> ims.PyTofScanPlanGroup:
        return self.__py_ptr

    # meta
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
        return np.asarray(self.__py_ptr.frame_times)

    @property
    def frame_ids(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.frame_ids)

    @property
    def tof_centers(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.tof_centers)

    @property
    def tof_min(self) -> int:
        return self.__py_ptr.tof_min

    @property
    def tof_max(self) -> int:
        return self.__py_ptr.tof_max

    def tof_center_at(self, mu: float) -> float:
        return self.__py_ptr.tof_center_at(mu)

    def tof_index_of(self, tof: int) -> Optional[int]:
        return self.__py_ptr.tof_index_of(tof)

    def tof_center_for_row(self, row: int) -> float:
        return self.__py_ptr.tof_center_for_row(row)

    def bounds(self, i: int) -> Optional[Tuple[int, int]]:
        return self.__py_ptr.bounds(i)

    def bounds_frame_ids(self, i: int) -> Optional[Tuple[int, int]]:
        return self.__py_ptr.bounds_frame_ids(i)

    @property
    def fragment_frame_id_bounds(self) -> Optional[Tuple[int, int]]:
        return self.__py_ptr.fragment_frame_id_bounds

    # iteration / indexing
    def __len__(self) -> int:
        return len(self.__py_ptr)

    def __getitem__(self, idx) -> TofScanWindowGrid | list[TofScanWindowGrid]:
        res = self.__py_ptr.__getitem__(idx)
        if isinstance(res, ims.PyTofScanWindowGrid):
            return TofScanWindowGrid.from_py_ptr(res)
        return [TofScanWindowGrid.from_py_ptr(g) for g in res]

    def __iter__(self) -> Iterator[TofScanWindowGrid]:
        for g in self.__py_ptr:
            yield TofScanWindowGrid.from_py_ptr(g)

    def get_batch(self, start: int, count: int) -> list[TofScanWindowGrid]:
        grids = self.__py_ptr.get_batch(start, count)
        return [TofScanWindowGrid.from_py_ptr(g) for g in grids]

    def get_batch_par(self, start: int, count: int) -> list[TofScanWindowGrid]:
        grids = self.__py_ptr.get_batch_par(start, count)
        return [TofScanWindowGrid.from_py_ptr(g) for g in grids]

    def __repr__(self) -> str:
        return f"TofScanPlanGroup(group={self.window_group}, num_windows={self.num_windows}, rows={self.rows})"


class RawPoints:
    """Ergonomic wrapper around ims.PyRawPoints."""

    def __init__(self, py):
        self._py = py

    # ---- basic size / emptiness -----------------------------------------

    @property
    def n(self) -> int:
        # length of the arrays; mz is fine
        return len(self._py.mz)

    @property
    def is_empty(self) -> bool:
        return self.n == 0

    # ---- array access ---------------------------------------------------

    def arrays(self):
        """
        Return numpy arrays (mz, rt, im, scan, intensity, tof, frame).

        All arrays are copies from the underlying Rust vectors, so you can
        mutate them freely on the Python side.
        """
        mz = np.asarray(self._py.mz, dtype=np.float32)
        rt = np.asarray(self._py.rt, dtype=np.float32)
        im = np.asarray(self._py.im, dtype=np.float32)
        scan = np.asarray(self._py.scan, dtype=np.uint32)
        intensity = np.asarray(self._py.intensity, dtype=np.float32)
        tof = np.asarray(self._py.tof, dtype=np.int32)
        frame = np.asarray(self._py.frame, dtype=np.uint32)
        return mz, rt, im, scan, intensity, tof, frame

    # ---- diagnostics (pure Python, cheap enough) ------------------------

    @property
    def unique_frames(self) -> np.ndarray:
        _, _, _, _, _, _, frame = self.arrays()
        return np.unique(frame.astype(np.uint32))

    @property
    def unique_scans(self) -> np.ndarray:
        _, _, _, scan, _, _, _ = self.arrays()
        return np.unique(scan.astype(np.uint32))

    @property
    def mz_min_max(self) -> tuple[float, float] | None:
        mz, *_ = self.arrays()
        if mz.size == 0:
            return None
        return float(mz.min()), float(mz.max())

    @property
    def rt_min_max(self) -> tuple[float, float] | None:
        _, rt, *_ = self.arrays()
        if rt.size == 0:
            return None
        return float(rt.min()), float(rt.max())

    @property
    def im_min_max(self) -> tuple[float, float] | None:
        *_, im, _, _, _, _ = self.arrays()
        if im.size == 0:
            return None
        return float(im.min()), float(im.max())

    @property
    def intensity_sum_max(self) -> tuple[float, float]:
        *_, intensity, _, _ = self.arrays()
        if intensity.size == 0:
            return 0.0, 0.0
        s = float(np.sum(intensity, dtype=np.float64))
        m = float(np.max(intensity))
        return s, m

    # ---- conversions ----------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        mz, rt, im, scan, inten, tof, frame = self.arrays()
        return pd.DataFrame(
            {
                "mz": mz,
                "rt": rt,
                "im": im,
                "scan": scan,
                "intensity": inten,
                "tof": tof,
                "frame": frame,
            }
        )

    def to_tims_slice(self) -> "TimsSlice":
        from imspy.timstof.slice import TimsSlice
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

    def to_dict(self) -> dict:
        """Small summary for dataframe merging / diagnostics."""
        (i_sum, i_max) = self.intensity_sum_max
        uf = self.unique_frames
        us = self.unique_scans
        mz_minmax = self.mz_min_max
        rt_minmax = self.rt_min_max
        im_minmax = self.im_min_max
        return {
            "raw_n": self.n,
            "raw_empty": self.is_empty,
            "raw_intensity_sum": i_sum,
            "raw_intensity_max": i_max,
            "raw_n_frames": int(uf.size),
            "raw_n_scans": int(us.size),
            "raw_mz_min": None if mz_minmax is None else mz_minmax[0],
            "raw_mz_max": None if mz_minmax is None else mz_minmax[1],
            "raw_rt_min": None if rt_minmax is None else rt_minmax[0],
            "raw_rt_max": None if rt_minmax is None else rt_minmax[1],
            "raw_im_min": None if im_minmax is None else im_minmax[0],
            "raw_im_max": None if im_minmax is None else im_minmax[1],
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"RawPoints(n={self.n})"


class ClusterResult1D:
    """Ergonomic wrapper around ims.PyClusterResult1D."""

    def __init__(self, py):
        self._py = py

    # --- windows ---------------------------------------------------------
    @property
    def rt_window(self):
        return self._py.rt_window

    @property
    def im_window(self):
        return self._py.im_window

    @property
    def tof_window(self):
        return self._py.tof_window

    @property
    def mz_window(self):
        # Optional[(float, float)]
        return self._py.mz_window

    # --- fits ------------------------------------------------------------
    @property
    def rt_fit(self) -> Fit1D:
        return Fit1D(self._py.rt_fit)

    @property
    def im_fit(self) -> Fit1D:
        return Fit1D(self._py.im_fit)

    @property
    def tof_fit(self) -> Fit1D:
        return Fit1D(self._py.tof_fit)

    @property
    def mz_fit(self) -> Fit1D | None:
        f = self._py.mz_fit
        return None if f is None else Fit1D(f)

    # convenience scalar shortcuts
    @property
    def rt_mu(self) -> float:        return float(self.rt_fit.mu)
    @property
    def rt_sigma(self) -> float:     return float(self.rt_fit.sigma)
    @property
    def rt_height(self) -> float:    return float(self.rt_fit.height)
    @property
    def rt_area(self) -> float:      return float(self.rt_fit.area)

    @property
    def im_mu(self) -> float:        return float(self.im_fit.mu)
    @property
    def im_sigma(self) -> float:     return float(self.im_fit.sigma)
    @property
    def im_height(self) -> float:    return float(self.im_fit.height)
    @property
    def im_area(self) -> float:      return float(self.im_fit.area)

    @property
    def tof_mu(self) -> float:       return float(self.tof_fit.mu)
    @property
    def tof_sigma(self) -> float:    return float(self.tof_fit.sigma)
    @property
    def tof_height(self) -> float:   return float(self.tof_fit.height)
    @property
    def tof_area(self) -> float:     return float(self.tof_fit.area)

    @property
    def mz_mu(self) -> float | None:
        return None if self.mz_fit is None else float(self.mz_fit.mu)

    @property
    def mz_sigma(self) -> float | None:
        return None if self.mz_fit is None else float(self.mz_fit.sigma)

    @property
    def mz_height(self) -> float | None:
        return None if self.mz_fit is None else float(self.mz_fit.height)

    @property
    def mz_area(self) -> float | None:
        return None if self.mz_fit is None else float(self.mz_fit.area)

    # --- stats / provenance ----------------------------------------------
    @property
    def raw_sum(self) -> float:
        return float(self._py.raw_sum)

    @property
    def volume_proxy(self) -> float:
        return float(self._py.volume_proxy)

    @property
    def ms_level(self) -> int:
        return int(self._py.ms_level)

    @property
    def window_group(self):
        return self._py.window_group

    @property
    def parent_im_id(self):
        return self._py.parent_im_id

    @property
    def parent_rt_id(self):
        return self._py.parent_rt_id

    # --- axes (may be None) ----------------------------------------------
    def frame_ids_used(self) -> np.ndarray:
        # NOTE: now a property, *not* a method
        return np.asarray(self._py.frame_ids_used, dtype=np.uint32)

    def rt_axis_sec(self) -> np.ndarray | None:
        arr = self._py.rt_axis_sec  # property, may be None
        return None if arr is None else np.asarray(arr, dtype=np.float32)

    def im_axis_scans(self) -> np.ndarray | None:
        arr = self._py.im_axis_scans  # property, may be None
        return None if arr is None else np.asarray(arr, dtype=np.int64)

    def mz_axis_da(self) -> np.ndarray | None:
        arr = self._py.mz_axis_da  # property, may be None
        return None if arr is None else np.asarray(arr, dtype=np.float32)

    # --- optional raw points ---------------------------------------------
    def raw_points(self) -> RawPoints | None:
        # raw_points is still a *method* on the PyO3 side
        rp = self._py.raw_points()
        return None if rp is None else RawPoints(rp)

    def get_py_ptr(self):
        return self._py

    # ---- diagnostics flags -----------------------------------------------
    @property
    def has_rt_axis(self) -> bool:
        return self.rt_axis_sec() is not None

    @property
    def has_im_axis(self) -> bool:
        return self.im_axis_scans() is not None

    @property
    def has_mz_axis(self) -> bool:
        return self.mz_axis_da() is not None

    @property
    def frame_count(self) -> int:
        fids = self.frame_ids_used()
        return int(fids.size)

    @property
    def raw_points_attached(self) -> bool:
        rp = self.raw_points()
        return (rp is not None) and (rp.n > 0)

    @property
    def raw_points_n(self) -> int:
        rp = self.raw_points()
        return 0 if rp is None else int(rp.n)

    @property
    def empty_rt(self) -> bool:
        f = self.rt_fit
        return (f.n == 0) or (abs(getattr(f, "area", 0.0)) <= 0.0)

    @property
    def empty_im(self) -> bool:
        f = self.im_fit
        return (f.n is None or f.n == 0) or (abs(getattr(f, "area", 0.0)) <= 0.0)

    @property
    def empty_tof(self) -> bool:
        f = self.tof_fit
        return (f.n is None or f.n == 0) or (abs(getattr(f, "area", 0.0)) <= 0.0)

    @property
    def empty_mz(self) -> bool:
        f = self.mz_fit
        if f is None:
            return True
        return (f.n is None or f.n == 0) or (abs(getattr(f, "area", 0.0)) <= 0.0)

    @property
    def any_empty_dim(self) -> bool:
        return self.empty_rt or self.empty_im or self.empty_tof or self.empty_mz

    @property
    def raw_empty(self) -> bool:
        return not self.raw_points_attached

    # --- flatten to dict / dataframe -------------------------------------
    def to_dict(self) -> dict:
        """Flatten cluster stats + optional raw diagnostics into a dict."""
        d: dict[str, object] = {
            "ms_level": self.ms_level,
            "window_group": self.window_group,
            "parent_im_id": self.parent_im_id,
            "parent_rt_id": self.parent_rt_id,
            "rt_lo": self.rt_window[0],
            "rt_hi": self.rt_window[1],
            "im_lo": self.im_window[0],
            "im_hi": self.im_window[1],
            "tof_lo": self.tof_window[0],
            "tof_hi": self.tof_window[1],
            "mz_lo": None if self.mz_window is None else self.mz_window[0],
            "mz_hi": None if self.mz_window is None else self.mz_window[1],
            "raw_sum": float(self.raw_sum),
            "volume_proxy": float(self.volume_proxy),
            "frame_count": self.frame_count,
            "has_rt_axis": self.has_rt_axis,
            "has_im_axis": self.has_im_axis,
            "has_mz_axis": self.has_mz_axis,
            "raw_points_attached": self.raw_points_attached,
            "raw_points_n": self.raw_points_n,
            "raw_empty": self.raw_empty,
            "empty_rt": self.empty_rt,
            "empty_im": self.empty_im,
            "empty_tof": self.empty_tof,
            "empty_mz": self.empty_mz,
            "any_empty_dim": self.any_empty_dim,
            # RT fit
            "rt_mu": self.rt_mu,
            "rt_sigma": self.rt_sigma,
            "rt_height": self.rt_height,
            "rt_area": self.rt_area,
            # IM fit
            "im_mu": self.im_mu,
            "im_sigma": self.im_sigma,
            "im_height": self.im_height,
            "im_area": self.im_area,
            # TOF fit
            "tof_mu": self.tof_mu,
            "tof_sigma": self.tof_sigma,
            "tof_height": self.tof_height,
            "tof_area": self.tof_area,
            # m/z fit (optional)
            "mz_mu": self.mz_mu,
            "mz_sigma": self.mz_sigma,
            "mz_height": self.mz_height,
            "mz_area": self.mz_area,
        }

        rp = self.raw_points()
        if rp is not None:
            d.update(rp.to_dict())
        else:
            d.update({
                "raw_n": 0,
                "raw_empty": True,
                "raw_intensity_sum": 0.0,
                "raw_intensity_max": 0.0,
                "raw_n_frames": 0,
                "raw_n_scans": 0,
                "raw_mz_min": None, "raw_mz_max": None,
                "raw_rt_min": None, "raw_rt_max": None,
                "raw_im_min": None, "raw_im_max": None,
            })
        return d

    def to_dataframe_row(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def __repr__(self):
        return (
            f"ClusterResult1D(ms_level={self.ms_level}, "
            f"window_group={self.window_group}, "
            f"raw_points_n={self.raw_points_n}, "
            f"rt_window={self.rt_window}, im_window={self.im_window}, "
            f"tof_window={self.tof_window}, mz_window={self.mz_window}, "
            f"rt_fit={self.rt_fit}, im_fit={self.im_fit}, "
            f"tof_fit={self.tof_fit}, mz_fit={self.mz_fit}, "
            f"raw_sum={self.raw_sum}, volume_proxy={self.volume_proxy}, "
            f"parent_im_id={self.parent_im_id}, "
            f"parent_rt_id={self.parent_rt_id})"
        )