import sqlite3
from typing import List, Sequence, Any, Optional

from imspy.simulation.annotation import RustWrapperObject
from imspy.timstof.data import TimsDataset
import pandas as pd

import imspy_connector

from imspy.timstof.frame import TimsFrame

ims = imspy_connector.py_dia

class FragmentIndex(RustWrapperObject):
    """
    Python wrapper for ims.PyFragmentIndex.

    Built from a TimsDatasetDIA instance:

        idx = ds.build_fragment_index(
            fragment_clusters,
            min_raw_sum=10.0,
        )

    Then queried per-precursor:

        wgs = ds.groups_for_precursor(prec_mz, prec_im)
        cand_ms2 = idx.query_precursor(
            precursor_cluster,
            window_groups=wgs,
            max_rt_apex_delta_sec=2.0,
            max_scan_apex_delta=6,
            min_im_overlap_scans=1,
            require_tile_compat=True,
        )
    """

    def __init__(self, *a, **k):
        raise RuntimeError(
            "Use TimsDatasetDIA.build_fragment_index(...) "
            "or FragmentIndex.from_py_ptr(...)"
        )

    @classmethod
    def from_py_ptr(cls, p: ims.PyFragmentIndex) -> "FragmentIndex":
        inst = cls.__new__(cls)
        inst._py = p
        return inst

    def get_py_ptr(self) -> ims.PyFragmentIndex:
        return self._py

    def query_precursor(
        self,
        precursor_cluster,
        window_groups: list[int] | None = None,
        *,
        max_rt_apex_delta_sec: float | None = 2.0,
        max_scan_apex_delta: int | None = 6,
        min_im_overlap_scans: int = 1,
        require_tile_compat: bool = True,
    ) -> list[int]:
        return self._py.query_precursor(
            precursor_cluster.get_py_ptr(),
            window_groups,  # None => let Rust compute from DiaIndex
            max_rt_apex_delta_sec=max_rt_apex_delta_sec,
            max_scan_apex_delta=max_scan_apex_delta,
            min_im_overlap_scans=min_im_overlap_scans,
            require_tile_compat=require_tile_compat,
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

    def clusters_for_group(
            self,
            window_group: int,
            im_peaks: list["ImPeak1D"],
            *,
            tof_step: int = 1,
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
            tof_bin_pad: int = 1,
            tof_hist_pad: int = 64,
            # Eval1DOpts
            refine_tof_once: bool = True,
            refine_k_sigma: float = 3.0,
            attach_axes: bool = True,
            attach_points: bool = False,
            attach_max_points: int | None = None,
            attach_im_xic: bool = False,
            attach_rt_xic: bool = False,
            # pairing + threads
            require_rt_overlap: bool = True,
            compute_mz_from_tof: bool = True,
            pad_rt_frames: int = 0,
            pad_im_scans: int = 0,
            pad_tof_bins: int = 0,
            num_threads: int = 0,
            min_im_span: int = 10,
            rt_pad_frames: int = 5,
    ):
        from imspy.timstof.clustering.data import ClusterResult1D
        """
        Cluster in **MS2** for one DIA window group.

        Parameters
        ----------
        window_group : int
            DIA window_group ID.
        im_peaks : list[ImPeak1D]
            IM 1D peaks for this window_group (must all carry this group ID).
        tof_step : int, default 1
            TOF binning factor for the CSR grid.
            1 = full TOF resolution, 2 = every 2nd TOF index, etc.
        Time-related parameters are in seconds.
        """
        if tof_step <= 0:
            raise ValueError(f"tof_step must be > 0, got {tof_step}")

        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, forcing num_threads=1.")
            num_threads = 1

        py_results = self.__dataset.clusters_for_group(
            int(window_group),
            int(tof_step),
            [p.get_py_ptr() for p in im_peaks],
            int(bin_pad),
            float(smooth_sigma_sec),
            float(smooth_trunc_k),
            float(min_prom),
            float(min_sep_sec),
            float(min_width_sec),
            int(fallback_if_frames_lt),
            float(fallback_frac_width),
            int(extra_rt_pad),
            int(extra_im_pad),
            int(tof_bin_pad),
            int(tof_hist_pad),
            bool(refine_tof_once),
            float(refine_k_sigma),
            bool(attach_axes),
            bool(attach_points),
            attach_max_points,
            bool(attach_im_xic),
            bool(attach_rt_xic),
            bool(require_rt_overlap),
            bool(compute_mz_from_tof),
            int(pad_rt_frames),
            int(pad_im_scans),
            int(pad_tof_bins),
            int(num_threads),
            int(min_im_span),
            int(rt_pad_frames),
        )
        return [ClusterResult1D(r) for r in py_results]

    def clusters_for_precursor(
            self,
            im_peaks: list["ImPeak1D"],
            *,
            tof_step: int = 1,
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
            tof_bin_pad: int = 1,
            tof_hist_bins: int = 64,
            # Eval1DOpts
            refine_tof_once: bool = True,
            refine_k_sigma: float = 3.0,
            attach_axes: bool = True,
            attach_points: bool = False,
            attach_max_points: int | None = None,
            attach_im_xic: bool = False,
            attach_rt_xic: bool = False,
            # pairing + threads
            require_rt_overlap: bool = True,
            compute_mz_from_tof: bool = True,
            pad_rt_frames: int = 0,
            pad_im_scans: int = 0,
            pad_tof_bins: int = 0,
            num_threads: int = 0,
            min_im_span: int = 10,
            rt_pad_frames: int = 5,
    ):
        from imspy.timstof.clustering.data import ClusterResult1D
        """
        Cluster in **MS1 precursor** space.

        Parameters
        ----------
        im_peaks : list[ImPeak1D]
            IM 1D peaks for MS1 (must have window_group=None).
        tof_step : int, default 1
            TOF binning factor for the CSR grid.
            1 = full TOF resolution, 2 = every 2nd TOF index, etc.
        Time-related parameters are in seconds.
        """
        if tof_step <= 0:
            raise ValueError(f"tof_step must be > 0, got {tof_step}")

        if self.use_bruker_sdk:
            warnings.warn("Using Bruker SDK, forcing num_threads=1.")
            num_threads = 1

        py_results = self.__dataset.clusters_for_precursor(
            int(tof_step),
            [p.get_py_ptr() for p in im_peaks],
            int(bin_pad),
            float(smooth_sigma_sec),
            float(smooth_trunc_k),
            float(min_prom),
            float(min_sep_sec),
            float(min_width_sec),
            int(fallback_if_frames_lt),
            float(fallback_frac_width),
            int(extra_rt_pad),
            int(extra_im_pad),
            int(tof_bin_pad),
            int(tof_hist_bins),
            bool(refine_tof_once),
            float(refine_k_sigma),
            bool(attach_axes),
            bool(attach_points),
            attach_max_points,
            bool(attach_im_xic),
            bool(attach_rt_xic),
            bool(require_rt_overlap),
            bool(compute_mz_from_tof),
            int(pad_rt_frames),
            int(pad_im_scans),
            int(pad_tof_bins),
            int(num_threads),
            int(min_im_span),
            int(rt_pad_frames),
        )
        return [ClusterResult1D(r) for r in py_results]

    @classmethod
    def from_py_ptr(cls, obj):
        instance = cls.__new__(cls)
        instance.__dataset = obj
        return instance

    def get_py_ptr(self):
        return self.__dataset

    def plan_tof_scan_windows(
            self,
            tof_step: int,
            rt_window_sec: float,
            rt_hop_sec: float,
            *,
            num_threads: int = 4,
            maybe_sigma_scans: float | None = None,
            maybe_sigma_tof_bins: float | None = None,
            truncate: float = 3.0,
            precompute_views: bool = False,
    ) -> "TofScanPlan":
        from imspy.timstof.clustering.data import TofScanPlan
        py_plan = ims.PyTofScanPlan(
            self.__dataset,
            tof_step,
            rt_window_sec,
            rt_hop_sec,
            num_threads,
            maybe_sigma_scans,
            maybe_sigma_tof_bins,
            truncate,
            precompute_views,
        )
        return TofScanPlan.from_py_ptr(py_plan)

    def plan_tof_scan_windows_for_group(
            self,
            window_group: int,
            tof_step: int,
            rt_window_sec: float,
            rt_hop_sec: float,
            *,
            num_threads: int = 4,
            maybe_sigma_scans: float | None = None,
            maybe_sigma_tof_bins: float | None = None,
            truncate: float = 3.0,
            precompute_views: bool = False,
    ) -> "TofScanPlanGroup":
        from imspy.timstof.clustering.data import TofScanPlanGroup
        py_plan = ims.PyTofScanPlanGroup(
            self.__dataset,
            int(window_group),
            tof_step,
            rt_window_sec,
            rt_hop_sec,
            num_threads,
            maybe_sigma_scans,
            maybe_sigma_tof_bins,
            truncate,
            precompute_views,
        )
        return TofScanPlanGroup.from_py_ptr(py_plan)

    def build_pseudo_spectra_geom(
            self,
            ms1_clusters: Sequence[Any],
            ms2_clusters: Sequence[Any],
            features: Sequence["SimpleFeature"] | None = None,
            *,
            top_n_fragments: int = 500,
            # CandidateOpts
            min_rt_jaccard: float = 0.0,
            ms2_rt_guard_sec: float = 0.0,
            rt_bucket_width: float = 1.0,
            max_ms1_rt_span_sec: float | None = 60.0,
            max_ms2_rt_span_sec: float | None = 60.0,
            min_raw_sum: float = 1.0,
            max_rt_apex_delta_sec: float | None = 2.0,
            max_scan_apex_delta: int | None = 6,
            min_im_overlap_scans: int = 1,
            # ScoreOpts
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
    ) -> "PseudoBuildResult":
        from imspy.timstof.clustering.data import PseudoBuildResult
        """
        High-level DIA → pseudo-DDA builder.

        Returns a PseudoBuildResult:
          - .pseudo_spectra → list[PseudoSpectrum]
          - .assignment     → AssignmentResult
        """
        ms1_ptrs = [c.get_py_ptr() for c in ms1_clusters]
        ms2_ptrs = [c.get_py_ptr() for c in ms2_clusters]

        feats_ptrs = None
        if features is not None:
            feats_ptrs = [f.get_py_ptr() for f in features]

        inner_res = self.__dataset.build_pseudo_spectra_from_clusters_geom(
            ms1_ptrs,
            ms2_ptrs,
            feats_ptrs,
            float(min_rt_jaccard),
            float(ms2_rt_guard_sec),
            float(rt_bucket_width),
            max_ms1_rt_span_sec,
            max_ms2_rt_span_sec,
            float(min_raw_sum),
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            int(min_im_overlap_scans),
            float(w_jacc_rt),
            float(w_shape),
            float(w_rt_apex),
            float(w_im_apex),
            float(w_im_overlap),
            float(w_ms1_intensity),
            float(rt_apex_scale_s),
            float(im_apex_scale_scans),
            float(shape_neutral),
            float(min_sigma_rt),
            float(min_sigma_im),
            float(w_shape_rt_inner),
            float(w_shape_im_inner),
            int(top_n_fragments),
        )

        # Wrap the PyPseudoBuildResult into the Python-level PseudoBuildResult
        return PseudoBuildResult(inner_res)

    def build_pseudo_spectra_xic(
            self,
            ms1_clusters: Sequence[Any],
            ms2_clusters: Sequence[Any],
            features: Sequence["SimpleFeature"] | None = None,
            *,
            top_n_fragments: int = 500,
            # CandidateOpts
            min_rt_jaccard: float = 0.0,
            ms2_rt_guard_sec: float = 0.0,
            rt_bucket_width: float = 1.0,
            max_ms1_rt_span_sec: float | None = 60.0,
            max_ms2_rt_span_sec: float | None = 60.0,
            min_raw_sum: float = 1.0,
            max_rt_apex_delta_sec: float | None = 2.0,
            max_scan_apex_delta: int | None = 6,
            min_im_overlap_scans: int = 1,
            # XicScoreOpts
            w_rt: float = 0.45,
            w_im: float = 0.45,
            w_intensity: float = 0.10,
            intensity_tau: float = 1.5,
            min_total_score: float = 0.0,
            use_rt: bool = True,
            use_im: bool = True,
            use_intensity: bool = True,
    ) -> "PseudoBuildResult":
        """
        High-level DIA → pseudo-DDA builder (XIC-based scoring).

        Returns a PseudoBuildResult:
          - .pseudo_spectra → list[PseudoSpectrum]
          - .assignment     → AssignmentResult
        """

        from imspy.timstof.clustering.data import PseudoBuildResult

        ms1_ptrs = [c.get_py_ptr() for c in ms1_clusters]
        ms2_ptrs = [c.get_py_ptr() for c in ms2_clusters]

        feats_ptrs = None
        if features is not None:
            feats_ptrs = [f.get_py_ptr() for f in features]

        inner_res = self.__dataset.build_pseudo_spectra_from_clusters_xic(
            ms1_ptrs,
            ms2_ptrs,
            feats_ptrs,
            float(min_rt_jaccard),
            float(ms2_rt_guard_sec),
            float(rt_bucket_width),
            max_ms1_rt_span_sec,
            max_ms2_rt_span_sec,
            float(min_raw_sum),
            max_rt_apex_delta_sec,
            max_scan_apex_delta,
            int(min_im_overlap_scans),
            float(w_rt),
            float(w_im),
            float(w_intensity),
            float(intensity_tau),
            float(min_total_score),
            bool(use_rt),
            bool(use_im),
            bool(use_intensity),
            int(top_n_fragments),
        )

        return PseudoBuildResult(inner_res)

    def build_pseudo_spectra_all_pairs(
            self,
            ms1_clusters: Sequence[Any],
            ms2_clusters: Sequence[Any],
            features: Sequence["SimpleFeature"] | None = None,
            *,
            top_n_fragments: int = 500,
    ) -> "PseudoBuildResult":
        """
        NON-competitive DIA → pseudo-DDA builder (debugging / visualization).

        - Uses a very loose candidate definition:
            * same window group
            * any RT overlap
            * any IM overlap (≥ 1 scan)
          plus the program-legal tile checks in Rust.
        - MS2 clusters may be linked to multiple precursors.

        Returns:
          PseudoBuildResult with all possible pairings.
        """
        from imspy.timstof.clustering.data import PseudoBuildResult

        ms1_ptrs = [c.get_py_ptr() for c in ms1_clusters]
        ms2_ptrs = [c.get_py_ptr() for c in ms2_clusters]

        feats_ptrs = None
        if features is not None:
            feats_ptrs = [f.get_py_ptr() for f in features]

        result = self.__dataset.build_pseudo_spectra_all_pairs_from_clusters(
            ms1_ptrs,
            ms2_ptrs,
            feats_ptrs,
            int(top_n_fragments),
        )

        return PseudoBuildResult(result)

    def window_groups_for_precursor(self, prec_mz: float, im_apex: float) -> List[int]:
        """
        Get DIA window groups that may contain the given precursor m/z and IM apex.

        Parameters
        ----------
        prec_mz : float
            Precursor m/z.
        im_apex : float
            IM apex (1/k0).

        Returns
        -------
        List[int]
            List of window group IDs that may contain the precursor.
        """
        return self.__dataset.window_groups_for_precursor(float(prec_mz), float(im_apex))

    def tof_rt_grid_precursor(self, tof_step: int = 1) -> "TofRtGrid":
        from imspy.timstof.clustering.data import TofRtGrid
        """
        Build a dense TOF × RT grid over all PRECURSOR (MS1) frames.

        Parameters
        ----------
        tof_step : int, default 1
            TOF binning step; 1 = max TOF resolution, larger values downsample.

        Returns
        -------
        TofRtGrid
            Grid over all MS1 frames (window_group = None).
        """
        grid = self.__dataset.tof_rt_grid_precursor(int(tof_step))
        return TofRtGrid.from_py_ptr(grid)

    def tof_rt_grid_for_group(self, window_group: int, tof_step: int = 1) -> "TofRtGrid":
        from imspy.timstof.clustering.data import TofRtGrid
        """
        Build a dense TOF × RT grid over FRAGMENT (MS2) frames for a DIA window group.

        Parameters
        ----------
        window_group : int
            DIA window_group ID.
        tof_step : int, default 1
            TOF binning step; 1 = max TOF resolution, larger values downsample.

        Returns
        -------
        TofRtGrid
            Grid over all MS2 frames in the given group.
        """
        grid = self.__dataset.tof_rt_grid_for_group(int(window_group), int(tof_step))
        return TofRtGrid.from_py_ptr(grid)

    def build_fragment_index(
            self,
            ms2_clusters: list["ClusterResult1D"],
            *,
            min_raw_sum: float = 1.0,
            ms2_rt_guard_sec: float = 0.0,
            max_ms2_rt_span_sec: float | None = 60.0,
            rt_bucket_width: float = 1.0,
    ) -> FragmentIndex:
        """
        Build a fragment index (FragmentIndex) over the given MS2 clusters.

        Parameters mirror the Rust CandidateOpts MS2-side fields.
        """
        py_ms2 = [c.get_py_ptr() for c in ms2_clusters]
        py_idx = self.get_py_ptr().build_fragment_index(
            py_ms2,
            min_raw_sum=min_raw_sum,
            ms2_rt_guard_sec=ms2_rt_guard_sec,
            max_ms2_rt_span_sec=max_ms2_rt_span_sec,
            rt_bucket_width=rt_bucket_width,
        )
        return FragmentIndex.from_py_ptr(py_idx)

import os
import tempfile
from pathlib import Path
from typing import List, Sequence, Union
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
    from imspy.timstof.clustering.data import ClusterResult1D
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