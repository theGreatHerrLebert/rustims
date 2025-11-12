from __future__ import annotations
import sqlite3

from imspy.timstof.slice import TimsSlice

from imspy.simulation.annotation import RustWrapperObject
from imspy.timstof.data import TimsDataset
import pandas as pd
import numpy as np
from typing import Optional, Dict

import imspy_connector

from imspy.timstof.frame import TimsFrame

ims = imspy_connector.py_dia

import os
import tempfile
import warnings
from pathlib import Path
from typing import List, Sequence, Union, Tuple
# --- helpers ---------------------------------------------------------------

import warnings

_BIN_SUFFIX = ".bin"
_BINZ_SUFFIX = ".binz"  # suggest: compressed files end with .binz

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _suffix_for(compress: bool) -> str:
    return _BINZ_SUFFIX if compress else _BIN_SUFFIX

def _infer_compress_from_suffix(path: Path, default: bool) -> bool:
    if path.suffix.lower() == _BINZ_SUFFIX:
        return True
    if path.suffix.lower() == _BIN_SUFFIX:
        return False
    return default

def _normalize_path_and_compress(
    path: Union[str, Path],
    compress: bool,
    allow_suffix_inference: bool = True,
) -> tuple[Path, bool]:
    """
    - Accept Path or str
    - If suffix is .binz/.bin, allow it to override `compress` (when allowed)
    - If no/unknown suffix, append the correct one
    - If suffix disagrees with `compress`, warn and rename to match `compress`
    """
    p = Path(path)

    # 1) infer from suffix if user explicitly provided one
    eff_compress = _infer_compress_from_suffix(p, compress) if allow_suffix_inference else compress

    # 2) normalize suffix
    desired_suffix = _suffix_for(eff_compress)
    if p.suffix.lower() not in {_BIN_SUFFIX, _BINZ_SUFFIX}:
        # add the expected suffix if missing/unknown
        p = p.with_suffix(p.suffix + desired_suffix if p.suffix else desired_suffix)
    elif p.suffix.lower() != desired_suffix:
        warnings.warn(
            f"Provided suffix '{p.suffix}' does not match compress={eff_compress}. "
            f"Using '{desired_suffix}' instead.",
            stacklevel=2,
        )
        p = p.with_suffix(desired_suffix)

    return p, eff_compress

def _assert_clusters(seq: Sequence["ClusterResult1D"]) -> None:
    if not isinstance(seq, Sequence):
        raise TypeError("clusters must be a sequence of ClusterResult1D")
    for i, c in enumerate(seq):
        # Be strict here; it prevents passing raw PyO3 handles by accident
        from types import SimpleNamespace  # noqa: F401  # only for fallback typing
        if not hasattr(c, "_py"):
            raise TypeError(f"clusters[{i}] is not a ClusterResult1D (missing ._py)")
        # Optional: very cheap sanity check to avoid mixing types
        if c.__class__.__name__ != "ClusterResult1D":
            warnings.warn(
                f"clusters[{i}] is a {c.__class__.__name__}, expected ClusterResult1D.",
                stacklevel=2,
            )

# --- public API ------------------------------------------------------------

def save_clusters_bin(
    path: Union[str, Path],
    clusters: Sequence["ClusterResult1D"],
    compress: bool = True,
    strip_points: bool = False,
    strip_axes: bool = False,
    *,
    overwrite: bool = True,
    atomic: bool = True,
) -> None:
    """
    Save clusters to a bincode file (.bin for uncompressed, .binz for compressed).

    Args:
        path: target path (str or Path). If no suffix is given, one is added
              based on `compress` (.binz if True, .bin if False).
              If a conflicting suffix is given, it is overridden with a warning.
        clusters: sequence of ClusterResult1D
        compress: gzip-like compression (controls suffix normalization)
        strip_points: drop raw point arrays before saving (smaller file)
        strip_axes: drop axes arrays before saving (smaller file)
        overwrite: allow overwriting existing files
        atomic: write to a temporary file and atomically replace target
    """
    _assert_clusters(clusters)

    # Normalize path & compression based on suffix conventions
    path, compress = _normalize_path_and_compress(path, compress, allow_suffix_inference=True)

    if not overwrite and Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    if strip_points or strip_axes:
        kept = []
        if not strip_points:
            kept.append("points")
        if not strip_axes:
            kept.append("axes")
        warnings.warn(
            "Stripping heavy fields for smaller file size; "
            f"kept: {', '.join(kept) if kept else 'none'}.",
            stacklevel=2,
        )

    rust_clusters = [c._py for c in clusters]  # stable interface to the PyO3 side

    _ensure_dir(Path(path))

    if atomic:
        # create a temp file in the same directory for atomic replace
        tmp_dir = str(Path(path).parent)
        suffix = Path(path).suffix
        with tempfile.NamedTemporaryFile(prefix=".tmp_", suffix=suffix, dir=tmp_dir, delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # write to temp via PyO3
            ims.save_clusters_bin(
                str(tmp_path),
                rust_clusters,
                bool(compress),
                bool(strip_points),
                bool(strip_axes),
            )
            # atomic replace
            os.replace(str(tmp_path), str(path))
        except Exception:
            # if something goes wrong, clean up the temp file
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            finally:
                raise
    else:
        ims.save_clusters_bin(
            str(path),
            rust_clusters,
            bool(compress),
            bool(strip_points),
            bool(strip_axes),
        )

def load_clusters_bin(path: Union[str, Path]) -> List["ClusterResult1D"]:
    """
    Load clusters from a bincode file (.bin or .binz).

    Args:
        path: file to load (str or Path). Suffix must be .bin or .binz.

    Returns:
        list[ClusterResult1D]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")
    if p.suffix.lower() not in {_BIN_SUFFIX, _BINZ_SUFFIX}:
        warnings.warn(
            f"Unexpected suffix '{p.suffix}'. Expected '{_BIN_SUFFIX}' or '{_BINZ_SUFFIX}'. "
            "Attempting to load anyway.",
            stacklevel=2,
        )

    rust_clusters = ims.load_clusters_bin(str(p))
    # Wrap back into your Python class
    return [ClusterResult1D(c) for c in rust_clusters]

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

    @property
    def is_empty(self) -> bool:
        return bool(self._py.is_empty)

    def arrays(self):
        """Return numpy arrays (mz, rt, im, scan, intensity, tof, frame)."""
        return tuple(np.asarray(arr) for arr in self._py.to_arrays())

    # ---- diagnostics from Rust (cheap) ----
    @property
    def unique_frames(self) -> np.ndarray:
        return np.asarray(self._py.unique_frames(), dtype=np.uint32)

    @property
    def unique_scans(self) -> np.ndarray:
        return np.asarray(self._py.unique_scans(), dtype=np.uint32)

    @property
    def mz_min_max(self) -> tuple[float, float] | None:
        t = self._py.mz_min_max()
        return None if t is None else (float(t[0]), float(t[1]))

    @property
    def rt_min_max(self) -> tuple[float, float] | None:
        t = self._py.rt_min_max()
        return None if t is None else (float(t[0]), float(t[1]))

    @property
    def im_min_max(self) -> tuple[float, float] | None:
        t = self._py.im_min_max()
        return None if t is None else (float(t[0]), float(t[1]))

    @property
    def intensity_sum_max(self) -> tuple[float, float]:
        s, m = self._py.intensity_sum_max()
        return float(s), float(m)

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

    def to_dict(self) -> dict:
        """Small summary for dataframe merging."""
        (i_sum, i_max) = self.intensity_sum_max
        uf = self.unique_frames
        us = self.unique_scans
        return {
            "raw_n": self.n,
            "raw_empty": self.is_empty,
            "raw_intensity_sum": i_sum,
            "raw_intensity_max": i_max,
            "raw_n_frames": int(uf.size),
            "raw_n_scans": int(us.size),
            "raw_mz_min": None if self.mz_min_max is None else self.mz_min_max[0],
            "raw_mz_max": None if self.mz_min_max is None else self.mz_min_max[1],
            "raw_rt_min": None if self.rt_min_max is None else self.rt_min_max[0],
            "raw_rt_max": None if self.rt_min_max is None else self.rt_min_max[1],
            "raw_im_min": None if self.im_min_max is None else self.im_min_max[0],
            "raw_im_max": None if self.im_min_max is None else self.im_min_max[1],
        }

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

    def get_py_ptr(self):
        return self._py

    # ---- diagnostics ----
    @property
    def has_rt_axis(self) -> bool: return self.rt_axis_sec() is not None
    @property
    def has_im_axis(self) -> bool: return self.im_axis_scans() is not None
    @property
    def has_mz_axis(self) -> bool: return self.mz_axis_da() is not None

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
    def empty_mz(self) -> bool:
        f = self.mz_fit
        return (f.n is None or f.n == 0) or (abs(getattr(f, "area", 0.0)) <= 0.0)

    @property
    def any_empty_dim(self) -> bool:
        return self.empty_rt or self.empty_im or self.empty_mz

    @property
    def raw_empty(self) -> bool:
        return not self.raw_points_attached

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
            "empty_mz": self.empty_mz,
            "any_empty_dim": self.any_empty_dim,
        }
        d.update(self.rt_fit.to_dict("rt"))
        d.update(self.im_fit.to_dict("im"))
        d.update(self.mz_fit.to_dict("mz"))

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
        """
        Return a string representation of the ClusterResult1D object.
        Returns:
        str: A string representation of the ClusterResult1D object.
        """

        return (f"ClusterResult1D(ms_level={self.ms_level}, window_group={self.window_group}, "
                f"raw_points_n={self.raw_points_n}, rt_window={self.rt_window}, im_window={self.im_window},"
                f" mz_window={self.mz_window}, rt_fit={self.rt_fit}, im_fit={self.im_fit}, "
                f"mz_fit={self.mz_fit}, raw_sum={self.raw_sum}, "
                f"volume_proxy={self.volume_proxy}, parent_im_id={self.parent_im_id}, "
                f"parent_rt_id={self.parent_rt_id})")

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

    def get_batch_par(self, start: int, count: int, num_threads: int) -> list["MzScanWindowGrid"]:
        grids = self.__py_ptr.get_batch_par(int(start), int(count))
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

    def get_batch_par(self, start: int, count: int, num_threads: int) -> list["MzScanWindowGrid"]:
        grids = self.__py_ptr.get_batch_par(int(start), int(count))
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

def _fnv1a64_signed(parts):
    h = 0xcbf29ce484222325
    for x in parts:
        x &= 0xFFFFFFFFFFFFFFFF
        h ^= x
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h - (1 << 64) if (h & (1 << 63)) else h

def _mz_bounds_from_centers(
    mz_centers: np.ndarray,
    row: int,
    pad_ppm: float = 0.0,
    pad_abs: float = 0.0,   # <-- NEW absolute Th padding
) -> tuple[float, float]:

    c = float(mz_centers[row])

    # --- apply ppm-based symmetric expansion first ---
    if pad_ppm > 0.0:
        f = 1.0 + pad_ppm * 1e-6
        lo, hi = c / f, c * f
    else:
        # --- fallback: half-neighbour spacing ---
        if row + 1 < len(mz_centers):
            dz = abs(float(mz_centers[row + 1]) - c)
        else:
            dz = abs(c - float(mz_centers[max(row - 1, 0)]))
        dz = dz if dz > 0 else 1.0
        lo, hi = c - 0.5 * dz, c + 0.5 * dz

    # --- NEW: absolute padding ---
    if pad_abs > 0.0:
        lo -= pad_abs
        hi += pad_abs

    # --- guards: no negative m/z ---
    lo = max(0.0, lo)
    if hi <= lo:
        hi = lo + 1e-6

    return float(lo), float(hi)


class ImPeak1D(RustWrapperObject):
    """Python wrapper for Rust PyImPeak1D."""
    def __init__(
            self,
            mz_row: int,
            mz_center: float,
            mz_bounds: tuple[float, float],
            rt_bounds: tuple[int, int],
            frame_id_bounds: tuple[int, int],
            window_group: int | None,
            scan: int,
            left: int,
            right: int,
            scan_abs: int,
            left_abs: int,
            right_abs: int,
            scan_sigma: float | None,
            mobility: float | None,
            apex_smoothed: float,
            apex_raw: float,
            prominence: float,
            left_x: float,
            right_x: float,
            width_scans: int,
            area_raw: float,
            subscan: float,
            id: int,
    ):
        """
        Direct constructor that creates a Rust-side PyImPeak1D
        using only primitive arguments and tuples.
        """
        # Call the Rust @new constructor
        p = ims.PyImPeak1D(
            mz_row,
            mz_center,
            mz_bounds,
            rt_bounds,
            frame_id_bounds,
            window_group,
            scan,
            left,
            right,
            scan_abs,
            left_abs,
            right_abs,
            scan_sigma,
            mobility,
            apex_smoothed,
            apex_raw,
            prominence,
            left_x,
            right_x,
            width_scans,
            area_raw,
            subscan,
            id,
        )
        # assign raw pointer for all getters
        self.__py_ptr = p

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
    def scan_sigma(self) -> float | None: return self.__py_ptr.scan_sigma
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
    @property
    def scan_abs(self) -> int:
        return self.__py_ptr.scan_abs
    @property
    def left_abs(self) -> int:
        return self.__py_ptr.left_abs
    @property
    def right_abs(self) -> int:
        return self.__py_ptr.right_abs

    def get_py_ptr(self):
        return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, p: ims.PyImPeak1D) -> "ImPeak1D":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    @classmethod
    def from_detected(cls,
                      det_row,
                      # dict | pandas.Series with keys: mu_scan, mu_mz, sigma_scan, sigma_mz, amplitude, baseline, area, i, j
                      *,
                      window_grid,  # MzScanWindowGrid
                      mz_centers=None,  # np.ndarray | None (if None, will try to read from a plan/plan-group)
                      plan=None,  # optional MzScanPlan (for mz_centers)
                      plan_group=None,  # optional MzScanPlanGroup (for mz_centers)
                      k_sigma: float = 3.0,
                      min_width: int = 3,
                      clamp_to_grid: bool = True,
                      mz_bounds_pad_ppm: float = 0.0,
                      mz_bounds_pad_abs: float = 0.0
                      ):

        # normalize row access
        if isinstance(det_row, pd.Series):
            r = det_row
        else:
            r = pd.Series(det_row)

        # get mz_centers
        if mz_centers is None:
            if plan is not None:
                mz_centers = plan.mz_centers
            elif plan_group is not None:
                mz_centers = plan_group.mz_centers
            else:
                raise ValueError("Provide mz_centers or (plan | plan_group).")
        mz_centers = np.asarray(mz_centers, dtype=np.float32)

        # grid metadata
        rt_lo, rt_hi = window_grid.rt_range_frames
        fid_lo, fid_hi = window_grid.frame_id_bounds
        wg = window_grid.window_group
        n_rows = window_grid.rows
        n_cols = window_grid.cols

        # indices from detector (i=row=m/z bin, j=col=scan)
        mz_row = int(r["i"])
        j_scan = int(r["j"])
        if clamp_to_grid:
            mz_row = max(0, min(n_rows - 1, mz_row))
            j_scan = max(0, min(n_cols - 1, j_scan))

        mu_scan = float(r["mu_scan"])
        s_scan = float(r["sigma_scan"])
        amp = float(r["amplitude"])
        base = float(r["baseline"])
        area = float(r["area"])

        # integer window bounds in scan from σ
        half = int(np.ceil(k_sigma * max(1e-6, s_scan)))
        left = max(0, j_scan - half)
        right = min(n_cols - 1, j_scan + half)
        width = max(min_width, right - left + 1)
        if width > (right - left + 1):
            extra = width - (right - left + 1)
            dl = extra // 2
            dr = extra - dl
            left = max(0, left - dl)
            right = min(n_cols - 1, right + dr)

        left_x = float(j_scan) - k_sigma * s_scan
        right_x = float(j_scan) + k_sigma * s_scan

        mz_center = float(mz_centers[mz_row])
        mz_bounds = _mz_bounds_from_centers(mz_centers, mz_row, pad_ppm=mz_bounds_pad_ppm, pad_abs=mz_bounds_pad_abs)

        apex_smoothed = amp + base
        apex_raw = apex_smoothed
        prominence = amp
        subscan = mu_scan - float(j_scan)

        scan_abs = j_scan
        left_abs, right_abs = left, right

        pid = int(_fnv1a64_signed([
            int(wg) if wg is not None else 0,
            mz_row,
            scan_abs,
            left_abs,
            right_abs,
            int(np.round(mz_center * 1e3)),
        ]))

        # Rust-side construction (tuples & primitives only)
        p = ims.PyImPeak1D(
            int(mz_row),
            float(mz_center),
            (float(mz_bounds[0]), float(mz_bounds[1])),
            (int(rt_lo), int(rt_hi)),
            (int(fid_lo), int(fid_hi)),
            (int(wg) if wg is not None else None),
            int(j_scan),
            int(left),
            int(right),
            int(scan_abs),
            int(left_abs),
            int(right_abs),
            float(s_scan) if s_scan > 0.0 else None,
            None,  # mobility: Option<f32>
            float(apex_smoothed),
            float(apex_raw),
            float(prominence),
            float(left_x),
            float(right_x),
            int(width),
            float(area),
            float(subscan),
            int(pid),
        )
        return cls.from_py_ptr(p)

    @classmethod
    def batch_from_detected(cls,
                            detected_dict_or_df,
                            *,
                            window_grid,
                            mz_centers=None,
                            plan=None,
                            plan_group=None,
                            k_sigma: float = 3.0,
                            min_width: int = 3,
                            clamp_to_grid: bool = True,
                            mz_bounds_pad_ppm: float = 0.0,
                            mz_bounds_pad_abs: float = 0.0
                            ) -> list["ImPeak1D"]:
        """
        Vectorized convenience: dict-of-arrays/DataFrame -> list[ImPeak1D].
        """
        D = pd.DataFrame(detected_dict_or_df)
        out = []
        for _, row in D.iterrows():
            out.append(cls.from_detected(
                row,
                window_grid=window_grid,
                mz_centers=mz_centers,
                plan=plan,
                plan_group=plan_group,
                k_sigma=k_sigma,
                min_width=min_width,
                clamp_to_grid=clamp_to_grid,
                mz_bounds_pad_ppm=mz_bounds_pad_ppm,
                mz_bounds_pad_abs=mz_bounds_pad_abs
            ))
        return out

    def __repr__(self):
        rt_lo, rt_hi = self.rt_bounds
        fid_lo, fid_hi = self.frame_id_bounds
        return (
            "ImPeak1D(mz_row={mz_row}, scan={scan}, mobility={mob}, "
            "apex_smoothed={apx:.1f}, prominence={prom:.1f}, width_scans={w}, area_raw={area:.1f}, "
            "left={l}, right={r}, left_x={lx:.3f}, right_x={rx:.3f}, subscan={sub:.3f}, "
            "rt_bounds=({rtl},{rth}), frame_id_bounds=({fl},{fh}), "
            "scan_abs={scan_abs}, left_abs={left_abs}, right_abs={right_abs}, "
            "window_group={wg})"
        ).format(
            mz_row=self.mz_row, scan=self.scan, mob=self.mobility,
            apx=self.apex_smoothed, prom=self.prominence, w=self.width_scans, area=self.area_raw,
            l=self.left, r=self.right, lx=self.left_x, rx=self.right_x, sub=self.subscan,
            rtl=rt_lo, rth=rt_hi, fl=fid_lo, fh=fid_hi,
            scan_abs=self.scan_abs, left_abs=self.left_abs, right_abs=self.right_abs,
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
        mz_sigma_bins: Optional[float] = None,
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
            mz_sigma_bins,
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
        mz_sigma_bins: Optional[float] = None,
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
            mz_sigma_bins,           # None or float
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
            smooth_sigma_sec: float = 1.25,
            smooth_trunc_k: float = 3.0,
            min_prom: float = 50.0,
            min_sep_sec: float = 2.0,
            min_width_sec: float = 2.0,
            ppm_per_bin: float = 10.0,
            fallback_if_frames_lt: int = 5,
            fallback_frac_width: float = 0.50,
    ) -> list[list["RtPeak1D"]]:
        """
        Expand IM peaks along RT in **MS1 precursor** space.

        All time parameters are in **seconds** (not frames).
        """
        nested = self.__dataset.expand_rt_for_im_peaks_in_precursor(
            [p.get_py_ptr() for p in im_peaks],
            int(bin_pad),
            float(smooth_sigma_sec),
            float(smooth_trunc_k),
            float(min_prom),
            float(min_sep_sec),
            float(min_width_sec),
            float(ppm_per_bin),
            int(fallback_if_frames_lt),
            float(fallback_frac_width),
        )
        return [[RtPeak1D.from_py_ptr(r) for r in row] for row in nested]

    def expand_rt_for_im_peaks_in_group(
            self,
            window_group: int,
            im_peaks: list["ImPeak1D"],
            *,
            bin_pad: int = 0,
            smooth_sigma_sec: float = 1.25,
            smooth_trunc_k: float = 3.0,
            min_prom: float = 50.0,
            min_sep_sec: float = 2.0,
            min_width_sec: float = 2.0,
            ppm_per_bin: float = 5.0,
            fallback_if_frames_lt: int = 5,
            fallback_frac_width: float = 0.50,
    ) -> list[list["RtPeak1D"]]:
        """
        Expand IM peaks along RT in **MS2 (one DIA window group)**.

        All time parameters are in **seconds** (not frames).
        """
        nested = self.__dataset.expand_rt_for_im_peaks_in_group(
            int(window_group),
            [p.get_py_ptr() for p in im_peaks],
            int(bin_pad),
            float(smooth_sigma_sec),
            float(smooth_trunc_k),
            float(min_prom),
            float(min_sep_sec),
            float(min_width_sec),
            float(ppm_per_bin),
            int(fallback_if_frames_lt),
            float(fallback_frac_width),
        )
        return [[RtPeak1D.from_py_ptr(r) for r in row] for row in nested]

    def clusters_for_group(
            self,
            window_group: int,
            im_peaks: list["ImPeak1D"],
            *,
            ppm_per_bin: float = 5.0,
            # RtExpandParams (seconds)
            bin_pad: int = 0,
            smooth_sigma_sec: float = 1.25,
            smooth_trunc_k: float = 3.0,
            min_prom: float = 50.0,
            min_sep_sec: float = 2.0,
            min_width_sec: float = 2.0,
            fallback_if_frames_lt: int = 5,
            fallback_frac_width: float = 0.50,
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
            min_im_span: int = 10,
    ):
        """Cluster in **MS2** for one DIA window group. Time params in seconds."""
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, forcing num_threads=1.")
            num_threads = 1

        py_results = self.__dataset.clusters_for_group(
            int(window_group),
            float(ppm_per_bin),
            [p.get_py_ptr() for p in im_peaks],
            int(bin_pad),
            float(smooth_sigma_sec), float(smooth_trunc_k),
            float(min_prom), float(min_sep_sec), float(min_width_sec),
            int(fallback_if_frames_lt), float(fallback_frac_width),
            int(extra_rt_pad), int(extra_im_pad), float(mz_ppm_pad), int(mz_hist_bins),
            bool(refine_mz_once), float(refine_k_sigma),
            bool(attach_axes),
            bool(attach_points), attach_max_points,
            bool(require_rt_overlap), int(num_threads),
            int(min_im_span)
        )
        return [ClusterResult1D(r) for r in py_results]

    def clusters_for_precursor(
            self,
            im_peaks: list["ImPeak1D"],
            *,
            ppm_per_bin: float = 10.0,
            # RtExpandParams (seconds)
            bin_pad: int = 0,
            smooth_sigma_sec: float = 1.25,
            smooth_trunc_k: float = 3.0,
            min_prom: float = 50.0,
            min_sep_sec: float = 2.0,
            min_width_sec: float = 2.0,
            fallback_if_frames_lt: int = 5,
            fallback_frac_width: float = 0.50,
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
            min_im_span: int = 10,
    ):
        """Cluster in **MS1 precursor** space. Time params in seconds."""
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, forcing num_threads=1.")
            num_threads = 1

        py_results = self.__dataset.clusters_for_precursor(
            float(ppm_per_bin),
            [p.get_py_ptr() for p in im_peaks],
            int(bin_pad),
            float(smooth_sigma_sec), float(smooth_trunc_k),
            float(min_prom), float(min_sep_sec), float(min_width_sec),
            int(fallback_if_frames_lt), float(fallback_frac_width),
            int(extra_rt_pad), int(extra_im_pad), float(mz_ppm_pad), int(mz_hist_bins),
            bool(refine_mz_once), float(refine_k_sigma),
            bool(attach_axes),
            bool(attach_points), attach_max_points,
            bool(require_rt_overlap), int(num_threads),
            int(min_im_span)
        )
        return [ClusterResult1D(r) for r in py_results]

    def enumerate_ms2_ms1_pairs(
            self,
            ms1_clusters: List["ClusterResult1D"],
            ms2_clusters: List["ClusterResult1D"],
            *,
            # coarse RT guards (match Rust defaults)
            min_rt_jaccard: float = 0.10,
            rt_guard_sec: float = 0.0,
            rt_bucket_width: float = 1.0,
            max_ms1_rt_span_sec: float | None = 60.0,
            max_ms2_rt_span_sec: float | None = 60.0,
            min_raw_sum: float = 1.0,
            # ---- NEW tight guards (keyword-only) ----
            max_rt_apex_delta_sec: float | None = 2.0,
            max_scan_apex_delta: int | None = 6,
            min_im_overlap_scans: int = 1,
            # Back-compat shim: accept old kwargs but ignore them
            **deprecated_kwargs,
    ) -> List[Tuple[int, int]]:
        """
        Return (ms2_idx, ms1_idx) pairs based on:
          • RT overlap (Jaccard >= min_rt_jaccard with MS2 guard),
          • Group reachability (mz∩isolation AND scans∩program),
          • Tight guards: IM window overlap, IM apex proximity (scans), RT apex proximity (s).

        All arguments are keyword-only; extra deprecated kwargs are ignored with a warning.
        """
        if deprecated_kwargs:
            import warnings
            warnings.warn(
                "enumerate_ms2_ms1_pairs: ignoring deprecated kwargs: "
                + ", ".join(sorted(deprecated_kwargs.keys())),
                RuntimeWarning,
                stacklevel=2,
            )

        py_ms1 = [c._py for c in ms1_clusters]
        py_ms2 = [c._py for c in ms2_clusters]

        # IMPORTANT: positional order must match the Rust signature exactly.
        return self.__dataset.enumerate_ms2_ms1_pairs(
            py_ms1,
            py_ms2,
            # coarse RT guards
            float(min_rt_jaccard),
            float(rt_guard_sec),
            float(rt_bucket_width),
            max_ms1_rt_span_sec,
            max_ms2_rt_span_sec,
            float(min_raw_sum),
            # tight guards
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            int(min_im_overlap_scans),
        )

    # Optional: tiny helper if you sometimes want objects instead of indices
    def materialize_pairs(
            self,
            pairs: List[Tuple[int, int]],
            ms1_clusters: List["ClusterResult1D"],
            ms2_clusters: List["ClusterResult1D"],
    ) -> List[Tuple["ClusterResult1D", "ClusterResult1D"]]:
        """
        Convert index pairs back to (ms2_obj, ms1_obj) for convenience.
        """
        out: list[tuple["ClusterResult1D", "ClusterResult1D"]] = []
        for j_ms2, i_ms1 in pairs:
            out.append((ms2_clusters[j_ms2], ms1_clusters[i_ms1]))
        return out

    def best_ms1_per_ms2(
            self,
            ms1_clusters: List["ClusterResult1D"],
            ms2_clusters: List["ClusterResult1D"],
            pairs: List[Tuple[int, int]],
            *,
            # scoring weights / params (passed straight through to Rust; same defaults)
            w_jacc_rt: float = 1.0,
            w_shape: float = 1.0,
            w_rt_apex: float = 0.75,
            w_im_apex: float = 0.75,
            w_im_overlap: float = 0.5,
            w_ms1_intensity: float = 0.25,
            rt_apex_scale_s: float = 0.75,
            im_apex_scale_scans: float = 3.0,
            shape_neutral: float = 0.6,
            min_sigma_rt: float = 0.05,
            min_sigma_im: float = 0.5,
            w_shape_rt_inner: float = 1.0,
            w_shape_im_inner: float = 1.0,
    ) -> List[Optional[int]]:
        """
        For each MS2 (by index), return the index of the best MS1 (or None).
        Uses the same scoring formulation as `score_ms2_ms1_pairs` internally.
        """
        py_ms1 = [c._py for c in ms1_clusters]
        py_ms2 = [c._py for c in ms2_clusters]

        return self.__dataset.best_ms1_per_ms2(
            py_ms1, py_ms2, pairs,
            w_jacc_rt, w_shape, w_rt_apex, w_im_apex, w_im_overlap, w_ms1_intensity,
            rt_apex_scale_s, im_apex_scale_scans, shape_neutral, min_sigma_rt, min_sigma_im,
            w_shape_rt_inner, w_shape_im_inner,
        )

    def group_ms2_by_ms1(
            self,
            winners: List[Optional[int]],
            ms1_clusters: List["ClusterResult1D"],
            ms2_clusters: List["ClusterResult1D"],
            scores: Optional[List[Optional[float]]] = None,
    ) -> Dict["ClusterResult1D", List["ClusterResult1D"]]:
        """
        Build a mapping: precursor MS1 cluster -> list of its assigned MS2 clusters.

        Parameters
        ----------
        winners : List[Optional[int]]
            Output of `best_ms1_per_ms2` (length == len(ms2_clusters)); each entry
            is the chosen ms1_idx for that MS2 or None if no valid assignment.
        scores : Optional[List[Optional[float]]]
            Optional per-MS2 score (same order as `winners`). If provided, each
            MS2 list per MS1 is sorted by descending score; otherwise preserves input order.

        Returns
        -------
        Dict[ClusterResult1D, List[ClusterResult1D]]
        """
        if scores is not None and len(scores) != len(winners):
            raise ValueError("scores must be same length as winners if provided")

        # Use object identity as keys; if you prefer indices, change key to int
        out: Dict["ClusterResult1D", List[Tuple[float, "ClusterResult1D"]]] = {}

        for j, ms1_idx in enumerate(winners):
            if ms1_idx is None:
                continue
            ms1_obj = ms1_clusters[ms1_idx]
            ms2_obj = ms2_clusters[j]
            s = scores[j] if scores is not None and scores[j] is not None else float("nan")
            out.setdefault(ms1_obj, []).append((s, ms2_obj))

        # Sort per-precursor by score if provided; strip scores
        grouped: Dict["ClusterResult1D", List["ClusterResult1D"]] = {}
        for ms1_obj, pairs in out.items():
            if scores is not None:
                pairs.sort(key=lambda t: (-(t[0] if t[0] == t[0] else float("-inf"))))  # NaN-safe
            grouped[ms1_obj] = [m for _, m in pairs]

        return grouped

    def enumerate_score_assign_group(
            self,
            ms1_clusters: List["ClusterResult1D"],
            ms2_clusters: List["ClusterResult1D"],
            *,
            # enumeration guards
            min_rt_jaccard: float = 0.10,
            rt_guard_sec: float = 0.0,
            rt_bucket_width: float = 1.0,
            max_ms1_rt_span_sec: float | None = 60.0,
            max_ms2_rt_span_sec: float | None = 60.0,
            min_raw_sum: float = 1.0,
            max_rt_apex_delta_sec: float | None = 2.0,
            max_scan_apex_delta: int | None = 6,
            min_im_overlap_scans: int = 1,
            # scoring knobs
            w_jacc_rt: float = 1.0,
            w_shape: float = 1.0,
            w_rt_apex: float = 0.75,
            w_im_apex: float = 0.75,
            w_im_overlap: float = 0.5,
            w_ms1_intensity: float = 0.25,
            rt_apex_scale_s: float = 0.75,
            im_apex_scale_scans: float = 3.0,
            shape_neutral: float = 0.6,
            min_sigma_rt: float = 0.05,
            min_sigma_im: float = 0.5,
            w_shape_rt_inner: float = 1.0,
            w_shape_im_inner: float = 1.0,
            return_scores: bool = False,
    ):
        """
        Convenience: enumerate → pick best MS1 per MS2 → group MS2 by MS1.
        Returns (grouped, winners, optional_scores) for downstream analysis.
        """
        pairs = self.enumerate_ms2_ms1_pairs(
            ms1_clusters, ms2_clusters,
            min_rt_jaccard=min_rt_jaccard,
            rt_guard_sec=rt_guard_sec,
            rt_bucket_width=rt_bucket_width,
            max_ms1_rt_span_sec=max_ms1_rt_span_sec,
            max_ms2_rt_span_sec=max_ms2_rt_span_sec,
            min_raw_sum=min_raw_sum,
            max_rt_apex_delta_sec=max_rt_apex_delta_sec,
            max_scan_apex_delta=max_scan_apex_delta,
            min_im_overlap_scans=min_im_overlap_scans,
        )

        winners = self.best_ms1_per_ms2(
            ms1_clusters, ms2_clusters, pairs,
            w_jacc_rt=w_jacc_rt, w_shape=w_shape, w_rt_apex=w_rt_apex, w_im_apex=w_im_apex,
            w_im_overlap=w_im_overlap, w_ms1_intensity=w_ms1_intensity,
            rt_apex_scale_s=rt_apex_scale_s, im_apex_scale_scans=im_apex_scale_scans,
            shape_neutral=shape_neutral, min_sigma_rt=min_sigma_rt, min_sigma_im=min_sigma_im,
            w_shape_rt_inner=w_shape_rt_inner, w_shape_im_inner=w_shape_im_inner,
        )

        scores = None
        if return_scores:
            # Use the same scoring call but ask Rust to return features/scores per pair,
            # then collapse to per-MS2 best (aligned with `winners`) for sorting.
            triplets_or_feats = self.__dataset.score_ms2_ms1_pairs(
                [c._py for c in ms1_clusters],
                [c._py for c in ms2_clusters],
                pairs,
                w_jacc_rt, w_shape, w_rt_apex, w_im_apex, w_im_overlap, w_ms1_intensity,
                rt_apex_scale_s, im_apex_scale_scans, shape_neutral, min_sigma_rt, min_sigma_im,
                w_shape_rt_inner, w_shape_im_inner,
                True,  # return_features
            )
            # triplets_or_feats is a list of dicts with "ms2_idx", "ms1_idx", "score", ...
            best_score_per_ms2 = [None] * len(ms2_clusters)
            for d in triplets_or_feats:
                j = int(d["ms2_idx"])
                i = int(d["ms1_idx"])
                s = float(d["score"])
                if winners[j] is None or winners[j] != i:
                    continue
                best_score_per_ms2[j] = s
            scores = best_score_per_ms2

        grouped = self.group_ms2_by_ms1(winners, ms1_clusters, ms2_clusters, scores=scores)
        return (grouped, winners, scores) if return_scores else (grouped, winners)

from collections.abc import Iterable
from typing import List, Sequence

def stitch_im_peaks(
    peaks: Sequence[Sequence[Sequence['ImPeak1D']]] ,  # windows × rows × peaks
    min_overlap_frames: int = 1,
    max_scan_delta: int = 1,
    jaccard_min: float = 0.0,             # RT Jaccard (unchanged)
    max_mz_row_delta: int = 0,
    allow_cross_groups: bool = False,
    # --- NEW IM constraints ---
    min_im_overlap_scans: int = 1,
    im_jaccard_min: float = 0.0,
    require_mutual_apex_inside: bool = True,
) -> List['ImPeak1D']:
    """
    Stitch IM 1D peaks across overlapping RT windows.

    Parameters
    ----------
    peaks : windows × rows × peaks (nested)
    min_overlap_frames : int
        Minimum RT frame overlap (inclusive).
    max_scan_delta : int
        Maximum allowed distance between IM apex scans.
    jaccard_min : float
        Minimum RT Jaccard; set 0.0 to disable.
    max_mz_row_delta : int
        Allowed |Δ mz_row|.
    allow_cross_groups : bool
        Permit stitching across DIA window groups.
    min_im_overlap_scans : int
        Minimum IM (half-prom) overlap in scans (inclusive).
    im_jaccard_min : float
        Minimum IM Jaccard on [left,right]; set 0.0 to disable.
    require_mutual_apex_inside : bool
        Require each apex scan to lie inside the other's half-prom interval.

    Returns
    -------
    List[ImPeak1D]
        Flat, deduplicated stitched peaks.
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
        # NEW args (must match Rust order)
        int(min_im_overlap_scans),
        float(im_jaccard_min),
        bool(require_mutual_apex_inside),
    )
    return [ImPeak1D.from_py_ptr(p) for p in stitched_py]


def stitch_im_peaks_flat(
    peaks,                   # Sequence[ImPeak1D]
    min_overlap_frames: int = 1,
    max_scan_delta: int = 1,
    jaccard_min: float = 0.0,
    max_mz_row_delta: int = 0,
    allow_cross_groups: bool = False,
    # --- NEW IM constraints ---
    min_im_overlap_scans: int = 1,
    im_jaccard_min: float = 0.0,
    require_mutual_apex_inside: bool = True,
):
    """
    Stitch IM 1D peaks from a flat list (unordered).

    See `stitch_im_peaks` for parameter semantics.
    """
    from .dia import ImPeak1D
    if not peaks:
        return []
    ptrs = [p.get_py_ptr() for p in peaks]
    stitched_py = ims.stitch_im_peaks_flat_unordered(
        ptrs,
        int(min_overlap_frames),
        int(max_scan_delta),
        float(jaccard_min),
        int(max_mz_row_delta),
        bool(allow_cross_groups),
        # NEW args (must match Rust order)
        int(min_im_overlap_scans),
        float(im_jaccard_min),
        bool(require_mutual_apex_inside),
    )
    return [ImPeak1D.from_py_ptr(p) for p in stitched_py]
_DEFAULT_ORDER = [
    # provenance / windows
    "ms_level", "window_group", "parent_im_id", "parent_rt_id",
    "rt_lo", "rt_hi", "im_lo", "im_hi", "mz_lo", "mz_hi",
    # summary stats
    "raw_sum", "volume_proxy", "frame_count",
    # present/empty flags
    "has_rt_axis", "has_im_axis", "has_mz_axis",
    "empty_rt", "empty_im", "empty_mz", "any_empty_dim",
    # fits
    "rt_mu", "rt_sigma", "rt_height", "rt_baseline", "rt_area", "rt_r2", "rt_n",
    "im_mu", "im_sigma", "im_height", "im_baseline", "im_area", "im_r2", "im_n",
    "mz_mu", "mz_sigma", "mz_height", "mz_baseline", "mz_area", "mz_r2", "mz_n",
    # raw attachment
    "raw_points_attached", "raw_points_n", "raw_empty",
    # optional raw aggregates (only present if include_raw_stats=True)
    "raw_intensity_sum", "raw_intensity_max",
    "raw_n_frames", "raw_n_scans",
    "raw_mz_min", "raw_mz_max",
    "raw_rt_min", "raw_rt_max",
    "raw_im_min", "raw_im_max",
]

_BOOL_COLS = [
    "has_rt_axis", "has_im_axis", "has_mz_axis",
    "empty_rt", "empty_im", "empty_mz", "any_empty_dim",
    "raw_points_attached", "raw_empty",
]


def clusters_to_dataframe(
    clusters,
    include_raw_stats: bool = False,
    as_bool_flags: bool = True,
    column_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert an iterable of ims.PyClusterResult1D (or your Python wrapper objects
    that expose the same pyobject) into a pandas DataFrame using the fast
    Rust-side column exporter.

    Parameters
    ----------
    clusters : Iterable[PyClusterResult1D or wrapper]
        Anything whose underlying pyobject is the `PyClusterResult1D` class
        from the PyO3 extension (your wrapper works since it’s extractable).
    include_raw_stats : bool, default False
        Include raw_* aggregate columns (requires raw points to be attached).
    as_bool_flags : bool, default True
        Convert uint8 indicator columns to boolean dtype.
    column_order : list[str] | None
        Reorder columns; by default uses a sensible order with optional raw_* at the end.

    Returns
    -------
    pandas.DataFrame
    """
    # Delegates to the PyO3 function; returns a dict[str, np.ndarray]-like
    arrs = ims.export_cluster_arrays(
        [c._py for c in clusters],
        include_raw_stats
    )

    # Defensive: ensure plain ndarray views
    data = {k: np.asarray(v) for k, v in arrs.items()}

    # Optional: convert flag columns to bool
    if as_bool_flags:
        for k in _BOOL_COLS:
            if k in data:
                # uint8 -> bool is fast and compact
                data[k] = data[k].astype(bool, copy=False)

    # Build DataFrame
    df = pd.DataFrame(data)

    # Optional reorder (skip any missing keys to stay robust whether raw_* present or not)
    if column_order is None:
        order = [c for c in _DEFAULT_ORDER if c in df.columns] + \
                [c for c in df.columns if c not in _DEFAULT_ORDER]
    else:
        order = [c for c in column_order if c in df.columns] + \
                [c for c in df.columns if c not in column_order]
    df = df.reindex(columns=order)

    return df
