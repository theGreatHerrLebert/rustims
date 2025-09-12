import sqlite3
import warnings
from typing import List, Optional

import numpy as np

from imspy.timstof.cluster import ClusterResult, ClusterSpec
from imspy.timstof.data import TimsDataset
import pandas as pd

import imspy_connector
from imspy.timstof.frame import TimsFrame
ims = imspy_connector.py_dia
from imspy.simulation.annotation import RustWrapperObject


class ImPeak1D(RustWrapperObject):
    """Python wrapper for Rust PyImPeak1D (read-only properties)."""

    def __init__(
        self,
        rt_row: int,
        scan: int,
        mobility: float,
        apex_smoothed: float,
        apex_raw: float,
        prominence: float,
        left: int,
        right: int,
        left_x: float,
        right_x: float,
        width_scans: int,
        area_raw: float,
        subscan: float,
    ):
        # Typically not constructed from Python; parity with pattern:
        self.__py_ptr = ims.PyImPeak1D(
            rt_row, scan, mobility, apex_smoothed, apex_raw, prominence,
            left, right, left_x, right_x, width_scans, area_raw, subscan
        )

    @property
    def rt_row(self) -> int: return self.__py_ptr.rt_row
    @property
    def scan(self) -> int: return self.__py_ptr.scan
    @property
    def mobility(self) -> float: return self.__py_ptr.mobility
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
        return (
            f"ImPeak1D(rt_row={self.rt_row}, scan={self.scan}, mobility={self.mobility}, "
            f"apex_smoothed={self.apex_smoothed:.1f}, prominence={self.prominence:.1f}, "
            f"width_scans={self.width_scans}, area_raw={self.area_raw:.1f}, "
            f"left={self.left}, right={self.right}, left_x={self.left_x:.3f}, "
            f"right_x={self.right_x:.3f}, subscan={self.subscan:.3f})"
        )


class RtPeak1D(RustWrapperObject):
    """Python wrapper for Rust PyPeak1D (read-only properties)."""

    def __init__(
        self,
        mz_row: int,
        rt_col: int,
        rt_time: float,
        apex_smoothed: float,
        apex_raw: float,
        prominence: float,
        left: int,
        right: int,
        width_frames: int,
        area_raw: float,
        subcol: float,
        right_x: float,
        left_x: float,
        left_padded: int = 0,
        right_padded: int = 0,
        area_padded: float = 0.0,
    ):
        # You’ll rarely construct peaks from Python, but keeping parity with your pattern:
        self.__py_ptr = ims.PyRtPeak1D(
            mz_row, rt_col, rt_time, apex_smoothed, apex_raw, prominence,
            left, right, width_frames, area_raw, subcol, right_x, left_x,
            left_padded, right_padded, area_padded
        )

    # --- properties mapping 1:1 to Rust getters ---

    @property
    def mz_row(self) -> int:
        return self.__py_ptr.mz_row

    @property
    def rt_col(self) -> int:
        return self.__py_ptr.rt_col

    @property
    def rt_time(self) -> float:
        return self.__py_ptr.rt_time

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
    def width_frames(self) -> int:
        return self.__py_ptr.width_frames

    @property
    def area_raw(self) -> float:
        return self.__py_ptr.area_raw

    @property
    def subcol(self) -> float:
        return self.__py_ptr.subcol

    @property
    def right_x(self) -> float:
        return self.__py_ptr.right_x

    @property
    def left_x(self) -> float:
        return self.__py_ptr.left_x

    @property
    def left_padded(self) -> int:
        return self.__py_ptr.left_padded

    @property
    def right_padded(self) -> int:
        return self.__py_ptr.right_padded

    @property
    def area_padded(self) -> float:
        return self.__py_ptr.area_padded

    def get_py_ptr(self):
        return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, peak: ims.PyRtPeak1D) -> "RtPeak1D":
        inst = cls.__new__(cls)
        inst.__py_ptr = peak
        return inst

    def __repr__(self):
        return (
            f"Peak1D(mz_row={self.mz_row}, rt_col={self.rt_col}, rt_time={self.rt_time:.3f}, "
            f"apex_smoothed={self.apex_smoothed:.1f}, prominence={self.prominence:.1f}, "
            f"width_frames={self.width_frames}, area_raw={self.area_raw:.1f}, left={self.left}, right={self.right}, "
            f"left_x={self.left_x:.3f}, right_x={self.right_x:.3f}, subcol={self.subcol:.3f}, "
            f"left_padded={self.left_padded}, right_padded={self.right_padded}, area_padded={self.area_padded:.1f})"
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

    def get_dense_mz_vs_rt(self, resolution: int = 1, num_threads: int = 4, sigma_frames = None, truncate = 3.0):
        """Get dense m/z vs RT matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: m/z values, RT values, intensity matrix.
        """

        # need to set num_threads to 1 when not using bruker sdk
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, setting num_threads to 1.")
            num_threads = 1

        return self.__dataset.build_dense_rt_by_mz(resolution, num_threads, truncate, sigma_frames)

    def get_dense_mz_vs_rt_and_pick(
            self,
            resolution: int = 1,
            num_threads: int = 4,
            sigma_frames: Optional[float] = None,
            truncate: float = 3.0,
            min_prom: float = 100.0,
            min_distance: int = 2,
            min_width: int = 2,
            left_pad : int = 1,
            right_pad : int = 2,
    ):
        """Build dense matrix (optionally smoothed) AND pick peaks.

        Returns:
            bins (np.ndarray[uint32]),
            frames (np.ndarray[uint32]),
            matrix (np.ndarray[float32], Fortran-contiguous),
            peaks (List[Peak1D])
        """
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, setting num_threads=1.")
            num_threads = 1

        bins, frames, mat, peaks_py = self.__dataset.build_dense_rt_by_mz_and_pick(
            resolution, num_threads, truncate, sigma_frames, min_prom, min_distance, min_width,
            left_pad, right_pad
        )

        # Convert list of PyRtPeak1D -> list of Peak1D wrappers
        peaks = [RtPeak1D.from_py_ptr(p) for p in peaks_py]
        return bins, frames, mat, peaks

    def get_dense_im_by_rtpeaks(
            self,
            peaks: List[RtPeak1D],
            bins: "np.ndarray[np.uint32]",  # from get_dense_mz_vs_rt_and_pick
            frames: "np.ndarray[np.uint32]",  # from get_dense_mz_vs_rt_and_pick
            resolution: int = 1,
            num_threads: int = 4,
            mz_ppm: float = 10.0,
            rt_extra_pad: int = 0,
            im_sigma_scans: Optional[float] = None,
            truncate: float = 3.0,
    ):
        """
        Build dense IM matrix conditioned on RT peaks.

        Args:
            peaks: List of RtPeak1D (row order defines matrix row order).
            bins:  np.ndarray[uint32], m/z bins from RT build (for mapping mz_row -> m/z).
            frames: np.ndarray[uint32], RT-sorted frame IDs from RT build.
            resolution: m/z quantization (same you used in RT build).
            num_threads: worker threads (set to 1 if using Bruker SDK).
            mz_ppm: ± ppm window around each peak's m/z center.
            rt_extra_pad: extra frames to include beyond each peak's padded [left,right].
            im_sigma_scans: optional Gaussian sigma (in scans) for IM smoothing.
            truncate: gaussian truncation (sigmas).

        Returns:
            scans: np.ndarray[uint32], shape (num_scans,)
            im_matrix: np.ndarray[float32], shape (len(peaks), num_scans), Fortran/column-major
        """
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, setting num_threads=1.")
            num_threads = 1

        # Extract underlying Rust peak objects for the bridge call
        peaks_py = [p.get_py_ptr() for p in peaks]

        scans, im_mat = self.__dataset.build_dense_im_by_rtpeaks(
            peaks_py,
            bins, frames,
            resolution, num_threads,
            mz_ppm, rt_extra_pad,
            im_sigma_scans, truncate,
        )
        return scans, im_mat

    def pick_im_peaks_on_im_matrix(
            self,
            im_matrix,  # np.ndarray[float32], shape (n_rows, n_scans), Fortran or C is fine
            im_matrix_raw=None,  # optional raw matrix (same shape) if you smoothed
            min_prom: float = 50.0,
            min_distance_scans: int = 2,
            min_width_scans: int = 2,
            use_mobility: bool = False,
    ):
        """
        Find IM peaks per RT-peak row on the IM matrix.

        Returns:
            List[List[ImPeak1D]]: length = n_rows; each entry is the list of IM peaks for that row.
        """
        # Call into Rust; it returns List[List[PyImPeak1D]]
        peak_rows_py = self.__dataset.pick_im_peaks_on_matrix(
            im_matrix,
            im_matrix_raw,
            min_prom,
            min_distance_scans,
            min_width_scans,
            use_mobility,
        )
        # Wrap to Python ImPeak1D
        return [[ImPeak1D.from_py_ptr(p) for p in row] for row in peak_rows_py]

    def pick_im_peaks_on_im_matrix_adaptive(
            self,
            im_matrix,
            im_matrix_raw=None,
            *,
            strategy: str = "active_range",  # "none" | "full" | "active_range" | "apex_window"
            low_thresh: float = 100.0,
            mid_thresh: float = 200.0,
            sigma_lo: float = 4.0,
            sigma_hi: float = 2.0,
            min_prom_lo: float = 25.0,
            min_prom_hi: float = 50.0,
            min_width_lo: int = 6,
            min_width_hi: int = 3,
            abs_thr: float = 5.0,
            rel_thr: float = 0.03,
            pad: int = 2,
            active_min_width: int = 6,
            apex_half_width: int = 15,
            min_distance_scans: int = 4,
            use_mobility: bool = False,
    ):
        rows_py = self.__dataset.pick_im_peaks_on_matrix_adaptive(
            im_matrix, im_matrix_raw,
            min_distance_scans,
            strategy,
            low_thresh, mid_thresh,
            sigma_lo, sigma_hi,
            min_prom_lo, min_prom_hi,
            min_width_lo, min_width_hi,
            abs_thr, rel_thr, pad, active_min_width,
            apex_half_width,
            use_mobility,
        )
        return [[ImPeak1D.from_py_ptr(p) for p in row] for row in rows_py]

    def evaluate_clusters_separable(
            self,
            specs: List[ClusterSpec],
            bins: "np.ndarray[np.uint32]",
            frames: "np.ndarray[np.uint32]",
            num_threads: int = 4,
    ) -> List[ClusterResult]:
        """
        Extract RT×IM patches for given ClusterSpec, fit a separable 2D Gaussian,
        and return ClusterResult objects (spec, patch, fit, quality).

        Args:
            specs: List[ClusterSpec] (row/column windows + mz_ppm/resolution).
            bins:  np.ndarray[uint32] from get_dense_mz_vs_rt(_and_pick).
            frames: np.ndarray[uint32] RT-sorted frame IDs from get_dense_mz_vs_rt(_and_pick).
            num_threads: worker threads (will be forced to 1 for Bruker SDK).

        Returns:
            List[ClusterResult]
        """
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, setting num_threads=1.")
            num_threads = 1

        specs_py = [s.get_py_ptr() for s in specs]
        res_py = self.__dataset.evaluate_clusters_separable(
            specs_py, bins, frames, num_threads
        )
        return [ClusterResult.from_py_ptr(r) for r in res_py]
