from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Sequence, Tuple
import numpy as np
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

class ImPeak1D(RustWrapperObject):
    """Python wrapper around ims.PyImPeak1D."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use ImPeak1D.from_py_ptr(...) instead.")

    @classmethod
    def from_py_ptr(cls, p: ims.PyImPeak1D) -> "ImPeak1D":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self) -> ims.PyImPeak1D:
        return self.__py_ptr

    # identity
    @property
    def id(self) -> int:
        return self.__py_ptr.id

    # TOF / IM geometry
    @property
    def tof_row(self) -> int:
        return self.__py_ptr.tof_row

    @property
    def tof_center(self) -> int:
        return self.__py_ptr.tof_center

    @property
    def tof_bounds(self) -> Tuple[int, int]:
        return self.__py_ptr.tof_bounds

    # RT / frame bounds
    @property
    def rt_bounds(self) -> Tuple[int, int]:
        return self.__py_ptr.rt_bounds

    @property
    def frame_id_bounds(self) -> Tuple[int, int]:
        return self.__py_ptr.frame_id_bounds

    @property
    def window_group(self) -> Optional[int]:
        return self.__py_ptr.window_group

    # scan geometry
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