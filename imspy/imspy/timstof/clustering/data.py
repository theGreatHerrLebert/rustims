from __future__ import annotations

from typing import Iterator
import numpy as np
import pandas as pd
from imspy.simulation.annotation import RustWrapperObject

import imspy_connector
ims = imspy_connector.py_dia

class Fit1D(RustWrapperObject):
    """Python wrapper around ims.PyFit1D."""

    def __init__(self, ptr):
        self.__py_ptr = ptr

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

    @property
    def id(self) -> int:
        return self.__py_ptr.id

    @property
    def rt_idx(self) -> int:
        return self.__py_ptr.rt_idx

    @property
    def rt_sec(self) -> float:
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

    @property
    def rt_bounds_frames(self) -> Tuple[int, int]:
        return self.__py_ptr.rt_bounds_frames

    @property
    def frame_id_bounds(self) -> Tuple[int, int]:
        return self.__py_ptr.frame_id_bounds

    @property
    def window_group(self) -> Optional[int]:
        return self.__py_ptr.window_group

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
    def parent_im_id(self) -> Optional[int]:
        return self.__py_ptr.parent_im_id

    # --- existing properties omitted for brevity ---

    # ------------------------------------------------------------------
    # NEW: construct a single RtPeak1D from detector output
    # ------------------------------------------------------------------
    @classmethod
    def from_detected(
        cls,
        peaks: Dict[str, np.ndarray],
        idx: int,
        *,
        window_grid: "TofRtGrid",
        plan_group=None,      # kept for API symmetry, unused here
        k_sigma: float = 3.0,
        min_width: int = 3,
        id_override: int | None = None,
    ) -> "RtPeak1D":
        """
        Build a single RtPeak1D from the dict-of-arrays produced by a
        TOF×RT detector.

        Accepts either:
          - peaks["tof_row"], peaks["scan_idx"]
          - or peaks["i"], peaks["j"] for backward compatibility.

        Assumes:
          - peaks["mu_rt"] (or "mu_scan"), peaks["sigma_rt"] (or "sigma_scan")
          - peaks["mu_tof"], peaks["sigma_tof"]
          - peaks["amplitude"], peaks["baseline"], peaks["area"]
        """
        g = window_grid  # TofRtGrid wrapper
        rows = g.rows
        cols = g.cols
        if cols == 0 or rows == 0:
            raise ValueError("TofRtGrid has zero size")

        # ---- get arrays with fallback names --------------------------------
        def arr(name: str, alt: str | None = None) -> np.ndarray:
            if name in peaks:
                return peaks[name]
            if alt and alt in peaks:
                return peaks[alt]
            raise KeyError(f"Required key {name!r} (or {alt!r}) missing from peaks")

        mu_rt = arr("mu_rt", "mu_scan")
        sigma_rt = arr("sigma_rt", "sigma_scan")
        mu_tof = arr("mu_tof")
        sigma_tof = arr("sigma_tof")
        amp = arr("amplitude")
        base = arr("baseline")
        area = arr("area")

        tof_row = arr("tof_row", "i")
        rt_idx_raw = arr("scan_idx", "j")

        # sanity
        n = mu_rt.shape[0]
        if not (
            sigma_rt.shape[0]
            == mu_tof.shape[0]
            == sigma_tof.shape[0]
            == amp.shape[0]
            == base.shape[0]
            == area.shape[0]
            == tof_row.shape[0]
            == rt_idx_raw.shape[0]
            == n
        ):
            raise ValueError("peaks arrays have inconsistent lengths")

        # ---- extract scalar values for this idx -----------------------------
        mu_rt_i = float(mu_rt[idx])
        sigma_rt_i = float(sigma_rt[idx])
        mu_tof_i = float(mu_tof[idx])
        sigma_tof_i = float(sigma_tof[idx])
        amp_i = float(amp[idx])
        base_i = float(base[idx])
        area_i = float(area[idx])
        tof_row_i = float(tof_row[idx])
        rt_idx_raw_i = float(rt_idx_raw[idx])

        # ---------------------------------------------------------------
        # RT geometry (indices, time, bounds)
        # ---------------------------------------------------------------
        # index in RT-frame space (0..cols-1); derived from mu_rt
        rt_idx = int(round(mu_rt_i))
        rt_idx = max(0, min(cols - 1, rt_idx))

        # subframe = fine position around that index
        subframe = mu_rt_i - float(rt_idx)

        # RT time in seconds from grid
        rt_times = g.rt_times
        if rt_times.size > 0:
            rt_sec = float(rt_times[rt_idx])
        else:
            # fallback: interpolate from rt_range_sec
            t0, t1 = g.rt_range_sec
            if cols > 1:
                frac = rt_idx / (cols - 1.0)
                rt_sec = float(t0 + frac * (t1 - t0))
            else:
                rt_sec = float(t0)

        # bounds in "frames" (RT index space)
        sigma_frames = max(sigma_rt_i, 1e-3)
        half_from_sigma = int(np.ceil(k_sigma * sigma_frames))
        half_from_min = max((min_width - 1) // 2, 0)
        half = max(half_from_sigma, half_from_min)

        rt_lo = max(0, rt_idx - half)
        rt_hi = min(cols - 1, rt_idx + half)
        width_frames = rt_hi - rt_lo + 1

        # map to frame IDs
        frame_ids = g.frame_ids
        if frame_ids.size > 0:
            frame_lo = int(frame_ids[rt_lo])
            frame_hi = int(frame_ids[rt_hi])
            frame_id_bounds = (frame_lo, frame_hi)
        else:
            frame_id_bounds = g.frame_id_bounds

        rt_bounds_frames = (rt_lo, rt_hi)

        # left/right_x can be interpreted as RT-frame coordinates
        left_x = float(rt_lo)
        right_x = float(rt_hi)

        # ---------------------------------------------------------------
        # TOF context
        # ---------------------------------------------------------------
        # primary TOF row index; clamp into valid range
        tof_row_idx = int(round(tof_row_i))
        tof_row_idx = max(0, min(rows - 1, tof_row_idx))

        # we treat mu_tof/sigma_tof in TOF-bin index units
        sigma_tof_bins = max(sigma_tof_i, 1e-3)
        half_tof = int(np.ceil(k_sigma * sigma_tof_bins))

        tof_center_idx = int(round(mu_tof_i))
        tof_center_idx = max(0, min(rows - 1, tof_center_idx))

        tof_lo_idx = max(0, tof_center_idx - half_tof)
        tof_hi_idx = min(rows - 1, tof_center_idx + half_tof)

        tof_bounds = (int(tof_lo_idx), int(tof_hi_idx))

        # for tof_center we can either:
        #   - store the index, or
        #   - convert to physical TOF using tof_centers
        # Keeping it consistent with ImPeak1D: treat it as index.
        tof_center = int(tof_center_idx)

        # ---------------------------------------------------------------
        # Peak shape / intensity proxies
        # ---------------------------------------------------------------
        # Here we only have "area" and amplitude/baseline from detector
        apex_raw = amp_i + base_i
        apex_smoothed = apex_raw
        prominence = amp_i  # simple proxy

        area_raw = area_i

        # provenance
        window_group = g.window_group
        parent_im_id = None  # RT-first; no IM parent at this point

        # ID: caller can override; otherwise use idx
        peak_id = int(id_override if id_override is not None else idx)

        # ---------------------------------------------------------------
        # Call into PyO3 constructor with the exact signature:
        # (rt_idx, rt_sec, apex_smoothed, apex_raw, prominence,
        #  left_x, right_x, width_frames, area_raw, subframe,
        #  rt_bounds_frames, frame_id_bounds, window_group,
        #  tof_row, tof_center, tof_bounds, parent_im_id, id)
        # ---------------------------------------------------------------
        py_peak = ims.PyRtPeak1D(
            rt_idx,
            rt_sec,
            apex_smoothed,
            apex_raw,
            prominence,
            left_x,
            right_x,
            int(width_frames),
            area_raw,
            subframe,
            rt_bounds_frames,
            frame_id_bounds,
            window_group,
            int(tof_row_idx),
            int(tof_center),
            tof_bounds,
            parent_im_id,
            int(peak_id),
        )

        return cls.from_py_ptr(py_peak)

    @classmethod
    def from_batch_detected(
            cls,
            peaks: Dict[str, np.ndarray],
            *,
            window_grid: "TofRtGrid",
            plan_group=None,
            k_sigma: float = 3.0,
            min_width: int = 3,
    ) -> list["RtPeak1D"]:
        """
        Build many RtPeak1D instances from detector output using a
        fast parallel Rust implementation.
        """
        # choose length from mu_rt / mu_scan
        if "mu_rt" in peaks:
            mu_rt = peaks["mu_rt"]
        elif "mu_scan" in peaks:
            mu_rt = peaks["mu_scan"]
        else:
            return []

        n = int(mu_rt.shape[0])
        if n == 0:
            return []

        def arr(name: str, alt: str | None = None) -> np.ndarray:
            if name in peaks:
                return peaks[name]
            if alt and alt in peaks:
                return peaks[alt]
            raise KeyError(f"Required key {name!r} (or {alt!r}) missing from peaks")

        mu_rt_arr = arr("mu_rt", "mu_scan")
        sigma_rt_arr = arr("sigma_rt", "sigma_scan")
        mu_tof_arr = arr("mu_tof")
        sigma_tof_arr = arr("sigma_tof")
        amp_arr = arr("amplitude")
        base_arr = arr("baseline")
        area_arr = arr("area")
        tof_row_arr = arr("tof_row", "i")
        rt_idx_raw_arr = arr("scan_idx", "j")

        # Call into the Rust static method.
        peaks_py = ims.PyRtPeak1D.from_batch_detected(
            window_grid.get_py_ptr(),
            np.asarray(mu_rt_arr, dtype=np.float32),
            np.asarray(sigma_rt_arr, dtype=np.float32),
            np.asarray(mu_tof_arr, dtype=np.float32),
            np.asarray(sigma_tof_arr, dtype=np.float32),
            np.asarray(amp_arr, dtype=np.float32),
            np.asarray(base_arr, dtype=np.float32),
            np.asarray(area_arr, dtype=np.float32),
            np.asarray(tof_row_arr, dtype=np.float32),
            np.asarray(rt_idx_raw_arr, dtype=np.float32),
            float(k_sigma),
            int(min_width),
        )

        return [cls.from_py_ptr(p) for p in peaks_py]

    def __repr__(self):
        return repr(self.__py_ptr)

    def to_dict(self) -> Dict[str, any]:
        return {
            "id": self.id,
            "rt_idx": self.rt_idx,
            "rt_sec": self.rt_sec,
            "apex_smoothed": self.apex_smoothed,
            "apex_raw": self.apex_raw,
            "prominence": self.prominence,
            "left_x": self.left_x,
            "right_x": self.right_x,
            "width_frames": self.width_frames,
            "area_raw": self.area_raw,
            "subframe": self.subframe,
            "rt_bounds_frames": self.rt_bounds_frames,
            "frame_id_bounds": self.frame_id_bounds,
            "window_group": self.window_group,
            "tof_row": self.tof_row,
            "tof_center": self.tof_center,
            "tof_bounds": self.tof_bounds,
            "parent_im_id": self.parent_im_id,
        }

from typing import Dict, List

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

    def to_dict(self) -> Dict[str, any]:
        return {
            "id": self.id,
            "tof_row": self.tof_row,
            "tof_center": self.tof_center,
            "tof_bounds": self.tof_bounds,
            "rt_bounds": self.rt_bounds,
            "frame_id_bounds": self.frame_id_bounds,
            "window_group": self.window_group,
            "scan": self.scan,
            "scan_sigma": self.scan_sigma,
            "mobility": self.mobility,
            "apex_smoothed": self.apex_smoothed,
            "apex_raw": self.apex_raw,
            "prominence": self.prominence,
            "left": self.left,
            "right": self.right,
            "left_x": self.left_x,
            "right_x": self.right_x,
            "width_scans": self.width_scans,
            "area_raw": self.area_raw,
            "subscan": self.subscan,
            "scan_abs": self.scan_abs,
            "left_abs": self.left_abs,
            "right_abs": self.right_abs,
        }

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

    @property
    def cluster_id(self):
        return self._py.cluster_id

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

    @property
    def rt_xic(self) -> Optional[np.ndarray]:
        arr = self._py.rt_xic  # property, may be None
        return None if arr is None else np.asarray(arr, dtype=np.float32)

    @property
    def im_xic(self) -> Optional[np.ndarray]:
        arr = self._py.im_xic  # property, may be None
        return None if arr is None else np.asarray(arr, dtype=np.float32)

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
        rp = self._py.raw_points
        return None if rp is None else RawPoints(rp)

    def get_py_ptr(self):
        return self._py

    @classmethod
    def from_py_ptr(cls, p: ims.PyClusterResult1D) -> "ClusterResult1D":
        return cls(p)

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
            "cluster_id": self.cluster_id,
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

    def drop_raw_data(self):
        """Drop attached raw points to save memory."""
        self._py.drop_raw_data()

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

from typing import Optional, Tuple, Union

Index = Union[int, slice, np.ndarray]

class TofRtGrid(RustWrapperObject):
    """
    Python wrapper around ims.PyTofRtGrid: dense TOF × RT matrix.

    - rows: TOF bins
    - cols: RT frames
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Use TofRtGrid.from_py_ptr(...) or TimsDatasetDIA.tof_rt_grid_* methods."
        )

    @classmethod
    def from_py_ptr(cls, p: ims.PyTofRtGrid) -> "TofRtGrid":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self) -> ims.PyTofRtGrid:
        return self.__py_ptr

    # --- basic meta ---

    @property
    def rows(self) -> int:
        return self.__py_ptr.rows

    @property
    def cols(self) -> int:
        return self.__py_ptr.cols

    @property
    def rt_range_frames(self) -> Tuple[int, int]:
        return self.__py_ptr.rt_range_frames

    @property
    def rt_range_sec(self) -> Tuple[float, float]:
        return self.__py_ptr.rt_range_sec

    @property
    def frame_id_bounds(self) -> Tuple[int, int]:
        return self.__py_ptr.frame_id_bounds

    @property
    def window_group(self) -> Optional[int]:
        return self.__py_ptr.window_group

    # --- dense data / axes ---

    @property
    def data(self) -> np.ndarray:
        """
        Dense TOF × RT matrix (shape = (rows, cols), dtype=float32).
        """
        return np.asarray(self.__py_ptr.data())

    @property
    def rt_times(self) -> np.ndarray:
        """
        RT axis (seconds), length = cols.
        """
        return np.asarray(self.__py_ptr.rt_times())

    @property
    def frame_ids(self) -> np.ndarray:
        """
        Frame IDs for each RT column, length = cols.
        """
        return np.asarray(self.__py_ptr.frame_ids())

    @property
    def tof_centers(self) -> np.ndarray:
        """
        TOF center for each TOF row, length = rows.
        Convert to m/z using ds.tof_to_mz on the Python side if needed.
        """
        return np.asarray(self.__py_ptr.tof_centers())

    # --- small conveniences ---

    def __repr__(self) -> str:
        wg = self.window_group
        if wg is None:
            return f"TofRtGrid(MS1, shape=({self.rows}, {self.cols}))"
        return f"TofRtGrid(group={wg}, shape=({self.rows}, {self.cols}))"

    def __getitem__(
            self,
            key: Union[Index, tuple[Index, Index]],
    ) -> np.ndarray:
        """
        Fast numpy-style slicing on the dense matrix:

        Examples
        --------
        grid[5000:5005, 2500:2510]   # TOF rows 5000–5004, RT cols 2500–2509
        grid[:, 100:200]             # all TOF, RT window
        grid[1000, :]                # one TOF row (XIC over RT)
        """
        return self.data[key]

    # --- XIC helper if you want a 1D trace directly ---

    def xic_by_index(self, tof_slice: Index, rt_slice: Index) -> np.ndarray:
        """
        Sum over TOF within `tof_slice` → 1D XIC vs RT in the given RT window.
        """
        block = self.data[tof_slice, rt_slice]
        # axis=0: sum over TOF rows → RT axis
        return block.sum(axis=0)

    def row_slice_for_tof_window(self, tof_lo: float, tof_hi: float) -> slice:
        centers = self.tof_centers
        lo = int(np.searchsorted(centers, tof_lo, side="left"))
        hi = int(np.searchsorted(centers, tof_hi, side="right"))
        return slice(lo, hi)

    def col_slice_for_rt_window(self, rt_lo: float, rt_hi: float) -> slice:
        t = self.rt_times
        lo = int(np.searchsorted(t, rt_lo, side="left"))
        hi = int(np.searchsorted(t, rt_hi, side="right"))
        return slice(lo, hi)

class AssignmentResult:
    """
    Thin Python wrapper around ims.PyAssignmentResult.

    Exposes:
      - pairs: list[(ms2_idx, ms1_idx)]
      - ms2_best_ms1: list[Optional[int]]
      - ms1_to_ms2: list[list[int]]
    """

    def __init__(self, inner: "ims.PyAssignmentResult") -> None:
        self._inner = inner

    @property
    def pairs(self) -> list[tuple[int, int]]:
        # Already a list of tuples on the Rust side, just copy
        return list(self._inner.pairs)

    @property
    def ms2_best_ms1(self) -> list[int | None]:
        return list(self._inner.ms2_best_ms1)

    @property
    def ms1_to_ms2(self) -> list[list[int]]:
        # Vec<Vec<usize>> from Rust is seen as list[list[int]]
        return [list(v) for v in self._inner.ms1_to_ms2]

    def __len__(self) -> int:
        return len(self._inner.pairs)

    def __repr__(self) -> str:
        return (
            f"AssignmentResult("
            f"num_pairs={len(self._inner.pairs)}, "
            f"num_ms2={len(self._inner.ms2_best_ms1)}, "
            f"num_ms1={len(self._inner.ms1_to_ms2)})"
        )


class PseudoBuildResult:
    """
    High-level Python wrapper around ims.PyPseudoBuildResult.

    Attributes:
      - pseudo_spectra: list[PseudoSpectrum]
      - assignment: AssignmentResult
    """

    def __init__(self, inner: "ims.PyPseudoBuildResult") -> None:
        self._inner = inner

    @property
    def pseudo_spectra(self) -> list["PseudoSpectrum"]:
        from imspy.timstof.clustering.pseudo import PseudoSpectrum
        # Wrap each PyPseudoSpectrum in your existing PseudoSpectrum wrapper
        return [PseudoSpectrum(s) for s in self._inner.pseudo_spectra]

    @property
    def assignment(self) -> AssignmentResult:
        return AssignmentResult(self._inner.assignment)

    def __len__(self) -> int:
        return len(self._inner.pseudo_spectra)

    def __iter__(self) -> Iterator["PseudoSpectrum"]:
        # Iterate over spectra, so you can do: for s in result: ...
        for s in self.pseudo_spectra:
            yield s

    def __repr__(self) -> str:
        return (
            f"PseudoBuildResult("
            f"num_spectra={len(self._inner.pseudo_spectra)}, "
            f"num_pairs={len(self._inner.assignment.pairs)})"
        )