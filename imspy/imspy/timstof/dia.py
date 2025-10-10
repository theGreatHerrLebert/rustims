import sqlite3
import warnings
from typing import List, Optional, Sequence

import numpy as np

from .cluster import ClusterResult, ClusterSpec, EvalOptions
from .data import TimsDataset
import pandas as pd

import imspy_connector
from imspy.timstof.frame import TimsFrame
ims = imspy_connector.py_dia
from imspy.simulation.annotation import RustWrapperObject

class ImIndex(RustWrapperObject):
    """Python wrapper for Rust PyImIndex (IM matrix around RT-picked peaks)."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "ImIndex objects are constructed in Rust; use ImIndex.from_py_ptr()."
        )

    @classmethod
    def from_py_ptr(cls, p: ims.PyImIndex) -> "ImIndex":
        """Wrap an existing PyImIndex pointer coming from Rust."""
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        """Return the underlying PyImIndex pointer for Rust calls."""
        return self.__py_ptr

    # ---------- Properties matching PyImIndex getters ----------

    @property
    def scans(self) -> np.ndarray:
        """Scan indices (0..num_scans-1) as a 1-D uint32 NumPy array."""
        return np.asarray(self.__py_ptr.scans, dtype=np.uint32)

    @property
    def data(self) -> np.ndarray:
        """
        Dense IM matrix (smoothed if smoothing was applied).
        Shape: (rows, cols), Fortran-contiguous.
        """
        return np.asarray(self.__py_ptr.data, dtype=np.float32, order="F")

    @property
    def data_raw(self) -> np.ndarray | None:
        """
        Optional raw (pre-smoothing) IM matrix, same shape/layout as `data`.
        Returns None if no smoothing was performed.
        """
        raw = self.__py_ptr.data_raw
        if raw is None:
            return None
        return np.asarray(raw, dtype=np.float32, order="F")

    @property
    def rows(self) -> int:
        """Number of RT-peak rows."""
        return self.__py_ptr.rows

    @property
    def cols(self) -> int:
        """Number of scan columns."""
        return self.__py_ptr.cols

    def __repr__(self) -> str:
        return (
            f"ImIndex(rows={self.rows}, cols={self.cols}, "
            f"scans={len(self.scans)}, "
            f"data_shape=({self.rows}, {self.cols}))"
        )

class RtIndex(RustWrapperObject):
    """
    Python wrapper for Rust PyRtIndex (read-only).
    Exposes m/z bin centers (constant-ppm grid), frame IDs/times, and the dense matrix.

    Matrix layout is Fortran/column-major with shape (rows, cols) = (num_mz_bins, num_frames).
    """

    def __init__(self, *_args, **_kwargs):
        # Not meant to be constructed directly; use from_py_ptr
        raise RuntimeError("RtIndex cannot be constructed directly; use RtIndex.from_py_ptr(...)")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyRtIndex") -> "RtIndex":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    # --- read-only properties (NumPy arrays are views/copies from Rust getters) ---
    @property
    def centers(self) -> np.ndarray:
        """m/z bin centers (float32), length == rows."""
        return self.__py_ptr.centers

    @property
    def frame_ids(self) -> np.ndarray:
        """RT-sorted frame IDs (uint32), length == cols."""
        return self.__py_ptr.frame_ids

    @property
    def frame_times(self) -> np.ndarray:
        """RT-sorted frame times (float32), same length/order as frame_ids."""
        return self.__py_ptr.frame_times

    @property
    def data(self) -> np.ndarray:
        """Dense matrix (float32), Fortran/column-major, shape (rows, cols)."""
        return self.__py_ptr.data

    @property
    def data_raw(self) -> Optional[np.ndarray]:
        """
        UnsMoothed dense matrix if smoothing was applied in RT build, else None.
        Fortran/column-major, shape (rows, cols).
        """
        return self.__py_ptr.data_raw

    # --- convenience ---
    @property
    def rows(self) -> int:
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        return self.data.shape[1]

    def as_arrays(self):
        """
        Convenience unpack identical to old tuple returns (but centers are float):
        Returns (centers, frame_ids, data[, frame_times])
        """
        return self.centers, self.frame_ids, self.data, self.frame_times

    def __repr__(self):
        return f"RtIndex(rows={self.rows}, cols={self.cols}, ppm_bin≈{self._ppm_hint():.3f})"

    def _ppm_hint(self) -> float:
        # Best-effort: estimate ppm per bin from first two centers
        c = self.centers
        if c.size >= 2 and c[0] > 0:
            return (c[1] / c[0] - 1.0) * 1e6
        return float("nan")


class ImPeak1D(RustWrapperObject):
    """Python wrapper for Rust PyImPeak1D (read-only)."""
    def __init__(self, *a, **k):
        raise RuntimeError("ImPeak1D is created in Rust; use ImPeak1D.from_py_ptr().")

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

    def __init__(self, *a, **k):
        raise RuntimeError("RtPeak1D is created in Rust; use RtPeak1D.from_py_ptr().")

    # --- properties mapping 1:1 to Rust getters ---

    @property
    def mz_row(self) -> int:
        return self.__py_ptr.mz_row

    @property
    def mz_center(self) -> float:
        return self.__py_ptr.mz_center

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
            f"Peak1D(mz_row={self.mz_row}, mz_center={self.mz_center}, rt_col={self.rt_col}, rt_time={self.rt_time:.3f}, "
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

    def get_dense_mz_vs_rt_and_pick(
            self,
            *,
            ppm_per_bin: float = 25.0,
            mz_pad_ppm: float = 50.0,
            num_threads: int = 4,
            sigma_frames: Optional[float] = None,
            truncate: float = 3.0,
            min_prom: float = 100.0,
            min_distance: int = 2,
            min_width: int = 2,
            left_pad: int = 1,
            right_pad: int = 2,
    ):
        """
        Build dense m/z×RT matrix on a constant-ppm grid (optionally smoothed) and pick RT peaks.

        Returns:
            rt_index (RtIndex): wrapper with centers, frame_ids/times, data(, data_raw)
            peaks (List[RtPeak1D])
        """
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, forcing num_threads=1.")
            num_threads = 1

        # Rust now returns (PyRtIndex, List[PyRtPeak1D])
        py_rt, peaks_py = self.__dataset.build_dense_rt_by_mz_and_pick(
            truncate,  # f32
            sigma_frames,  # Option<f32>
            ppm_per_bin,  # f32
            mz_pad_ppm,  # f32
            num_threads,  # usize
            min_prom,  # f32
            min_distance,  # usize
            min_width,  # usize
            left_pad,  # usize
            right_pad,  # usize
        )

        rt_index = RtIndex.from_py_ptr(py_rt)
        peaks = [RtPeak1D.from_py_ptr(p) for p in peaks_py]
        return rt_index, peaks

    def get_dense_mz_vs_rt_by_window_group_and_pick(
            self,
            window_group: int,
            sigma_frames: Optional[float] = None,
            truncate: float = 3.0,
            ppm_per_bin: float = 25.0,
            mz_pad_ppm: float = 50.0,
            clamp_to_group: bool = True,
            num_threads: int = 4,
            min_prom: float = 100.0,
            min_distance: int = 2,
            min_width: int = 2,
            left_pad: int = 0,
            right_pad: int = 0,
    ):
        """
        Build dense m/z×RT matrix on a constant-ppm grid (optionally smoothed) for a specific window group and pick RT peaks.

        Returns:
            rt_index (RtIndex): wrapper with centers, frame_ids/times, data(, data_raw)
            peaks (List[RtPeak1D])
        """
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, forcing num_threads=1.")
            num_threads = 1

        # Rust now returns (PyRtIndex, List[PyRtPeak1D])
        py_rt, peaks_py = self.__dataset.build_dense_rt_by_mz_and_window_group_and_pick(
            window_group,
            sigma_frames,
            truncate,  # f32
            ppm_per_bin,  # f32
            mz_pad_ppm,  # f32
            clamp_to_group,
            num_threads,  # usize
            min_prom,  # f32
            min_distance,  # usize
            min_width,  # usize
            left_pad,  # usize
            right_pad,  # usize
        )

        rt_index = RtIndex.from_py_ptr(py_rt)
        peaks = [RtPeak1D.from_py_ptr(p) for p in peaks_py]
        return rt_index, peaks

    def build_dense_im_by_rtpeaks_ppm(
        self,
        rt_index: "RtIndex",
        peaks: List["RtPeak1D"],
        *,
        num_threads: int = 4,
        mz_ppm_window: float = 10.0,
        rt_extra_pad: int = 0,
        im_sigma_scans: Optional[float] = None,
        truncate: float = 3.0,
    ) -> ImIndex:
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, forcing num_threads=1.")
            num_threads = 1

        py_im = self.__dataset.build_dense_im_by_rtpeaks_ppm(
            rt_index.get_py_ptr(),
            [p.get_py_ptr() for p in peaks],
            num_threads,
            mz_ppm_window,
            rt_extra_pad,
            im_sigma_scans,
            truncate,
        )
        return ImIndex.from_py_ptr(py_im)

    def build_dense_im_by_rtpeaks_ppm_and_pick(
        self,
        rt_index: "RtIndex",
        peaks: List["RtPeak1D"],
        *,
        num_threads: int = 4,
        mz_ppm_window: float = 10.0,
        rt_extra_pad: int = 0,
        im_sigma_scans: Optional[float] = None,
        truncate: float = 3.0,
        min_prom: float = 50.0,
        min_distance_scans: int = 2,
        min_width_scans: int = 2,
        use_mobility: bool = False,
    ):
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, forcing num_threads=1.")
            num_threads = 1

        py_im, rows_py = self.__dataset.build_dense_im_by_rtpeaks_ppm_and_pick(
            rt_index.get_py_ptr(),
            [p.get_py_ptr() for p in peaks],
            num_threads,
            mz_ppm_window,
            rt_extra_pad,
            im_sigma_scans,
            truncate,
            min_prom,
            min_distance_scans,
            min_width_scans,
            use_mobility,
        )
        im_index = ImIndex.from_py_ptr(py_im)
        im_peaks = [[ImPeak1D.from_py_ptr(p) for p in row] for row in rows_py]
        return im_index, im_peaks

    def build_dense_im_by_rtpeaks_ppm_for_group_and_pick(
        self,
        window_group: int,
        rt_index: "RtIndex",
        peaks: List["RtPeak1D"],
        *,
        num_threads: int = 4,
        mz_ppm_window: float = 10.0,
        rt_extra_pad: int = 0,
        im_sigma_scans: Optional[float] = None,
        truncate: float = 3.0,
        min_prom: float = 50.0,
        min_distance_scans: int = 2,
        min_width_scans: int = 2,
        clamp_scans_to_group: bool = True,
        use_mobility: bool = False,
    ):
        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, forcing num_threads=1.")
            num_threads = 1

        py_im, rows_py = self.__dataset.build_dense_im_by_rtpeaks_ppm_for_group_and_pick(
            window_group,
            rt_index.get_py_ptr(),
            [p.get_py_ptr() for p in peaks],
            num_threads,
            mz_ppm_window,
            rt_extra_pad,
            im_sigma_scans,
            truncate,
            min_prom,
            min_distance_scans,
            min_width_scans,
            clamp_scans_to_group,
            use_mobility,
        )
        im_index = ImIndex.from_py_ptr(py_im)
        im_peaks = [[ImPeak1D.from_py_ptr(p) for p in row] for row in rows_py]
        return im_index, im_peaks

    def evaluate_clusters_3d(
            self,
            rt_index: "RtIndex",
            specs: Sequence[ClusterSpec],
            opts: Optional[EvalOptions] = None,
            num_threads: int = 4,
    ) -> List[ClusterResult]:
        """
        Evaluate a list of ClusterSpec against the dataset.

        Returns a list of ClusterResult. Use EvalOptions(AttachOptions(...)) to
        attach axes/patch if you need raw data alongside fits.
        """
        specs_py = [s.get_py_ptr() for s in specs]
        if opts is None:
            opts_py = ims.PyEvalOptions(ims.PyAttachOptions(True, True, True, False), False, 3.0)
        else:
            opts_py = opts.get_py_ptr()

        res_py = self.__dataset.evaluate_clusters_3d(rt_index.get_py_ptr(), specs_py, opts_py, num_threads)
        return [ClusterResult.from_py_ptr(r) for r in res_py]

    def group_mz_unions(self) -> dict[int, List[tuple[float, float]]]:
        """Get m/z unions for each window group.

        Returns:
            dict[int, List[tuple[float, float]]]: Dictionary mapping window group to list of (min_mz, max_mz) tuples.
        """
        return self.__dataset.group_mz_unions()

    """
    /// Build features from envelopes using preloaded precursor frames internally.
    pub fn build_features_from_envelopes(
        &self,
        envelopes: Vec<PyEnvelope>,
        clusters: Vec<PyClusterResult>,
        lut: PyAveragineLut,
        gp: PyGroupingParams,
        fp: PyFeatureBuildParams,
    ) -> PyResult<Vec<PyFeature>> {
        // Unwrap inner
        let envs: Vec<Envelope> = envelopes.into_iter().map(|e| e.inner).collect();
        let clus: Vec<ClusterResult> = clusters.into_iter().map(|c| c.inner).collect();

        // Call the Rust method on TimsDatasetDIA
        let feats: Vec<Feature> = self
            .inner
            .build_features_from_envelopes(
                &envs,
                &clus,
                &lut.inner,
                &gp.inner,
                &fp.inner,
            );

        // Wrap back for Python
        Ok(feats.into_iter().map(|f| PyFeature { inner: f }).collect())
    }
    """
    def build_features_from_envelopes(
        self,
        envelopes: Sequence["Envelope"],
        clusters: Sequence["ClusterResult"],
        lut: "AveragineLut",
        gp: "GroupingParams",
        fp: "FeatureBuildParams",
    ) -> List["Feature"]:
        from .feature import Feature
        envelopes_py = [e.get_py_ptr() for e in envelopes]
        clusters_py = [c.get_py_ptr() for c in clusters]
        lut_py = lut.get_py_ptr()
        gp_py = gp.get_py_ptr()
        fp_py = fp.get_py_ptr()

        feats_py = self.__dataset.build_features_from_envelopes(
            envelopes_py,
            clusters_py,
            lut_py,
            gp_py,
            fp_py,
        )
        return [Feature.from_py_ptr(f) for f in feats_py]