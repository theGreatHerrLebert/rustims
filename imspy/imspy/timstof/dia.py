import sqlite3
from typing import List, Sequence, Any, Optional

from imspy.simulation.annotation import RustWrapperObject
from imspy.timstof.clustering.data import ClusterResult1D
from imspy.timstof.data import TimsDataset
import pandas as pd

import imspy_connector

from imspy.timstof.frame import TimsFrame

ims = imspy_connector.py_dia

class CandidateOpts(RustWrapperObject):
    """
    Python wrapper for ims.PyCandidateOpts.

    Encapsulates the candidate-enumeration options used when building a
    FragmentIndex. You can reuse a CandidateOpts instance for multiple
    indexes if you like.
    """

    def __init__(
        self,
        *,
        min_rt_jaccard: float = 0.0,
        ms2_rt_guard_sec: float = 0.0,
        rt_bucket_width: float = 1.0,
        max_ms1_rt_span_sec: float | None = None,
        max_ms2_rt_span_sec: float | None = 60.0,
        min_raw_sum: float = 1.0,
        max_rt_apex_delta_sec: float | None = None,
        max_scan_apex_delta: int | None = None,
        min_im_overlap_scans: int = 1,
        reject_frag_inside_precursor_tile: bool = True,
    ) -> None:
        # Directly construct the PyO3 object
        self._py = ims.PyCandidateOpts(
            min_rt_jaccard=min_rt_jaccard,
            ms2_rt_guard_sec=ms2_rt_guard_sec,
            rt_bucket_width=rt_bucket_width,
            max_ms1_rt_span_sec=max_ms1_rt_span_sec,
            max_ms2_rt_span_sec=max_ms2_rt_span_sec,
            min_raw_sum=min_raw_sum,
            max_rt_apex_delta_sec=max_rt_apex_delta_sec,
            max_scan_apex_delta=max_scan_apex_delta,
            min_im_overlap_scans=min_im_overlap_scans,
            reject_frag_inside_precursor_tile=reject_frag_inside_precursor_tile,
        )

    def get_py_ptr(self) -> ims.PyCandidateOpts:
        return self._py

    @classmethod
    def from_py_ptr(cls, p: ims.PyCandidateOpts) -> "CandidateOpts":
        inst = cls.__new__(cls)
        inst._py = p
        return inst

class ScoredHit(RustWrapperObject):
    """
    Thin Python wrapper around ims.PyScoredHit.

    Exposes read-only properties like:
      - ms2_index: index of the fragment cluster (into the fragment cluster array)
      - score: overall match score

    Extend with more properties if PyScoredHit exposes them (e.g. rt_score, xic_score, mode, ...).
    """

    def __init__(self, *a, **k):
        raise RuntimeError(
            "Use ScoreHit.from_py_ptr(...) instead of constructing directly."
        )

    @classmethod
    def from_py_ptr(cls, p: ims.PyScoredHit) -> "ScoredHit":
        inst = cls.__new__(cls)
        inst._py = p
        return inst

    def get_py_ptr(self) -> ims.PyScoredHit:
        return self._py

    # --- properties, mapped 1:1 to PyScoredHit getters ----

    @property
    def ms2_index(self) -> int:
        """Index into the fragment-cluster array used to build the FragmentIndex."""
        return self._py.frag_idx

    @property
    def score(self) -> float:
        """Overall match score (higher is better)."""
        return self._py.score

    # Geometric features (None in XIC mode)

    @property
    def jacc_rt(self) -> float | None:
        return self._py.jacc_rt

    @property
    def rt_apex_delta_s(self) -> float | None:
        return self._py.rt_apex_delta_s

    @property
    def im_apex_delta_scans(self) -> float | None:
        return self._py.im_apex_delta_scans

    @property
    def im_overlap_scans(self) -> int | None:
        return self._py.im_overlap_scans

    @property
    def im_union_scans(self) -> int | None:
        return self._py.im_union_scans

    @property
    def ms1_raw_sum(self) -> float | None:
        return self._py.ms1_raw_sum

    @property
    def shape_ok(self) -> bool | None:
        return self._py.shape_ok

    @property
    def z_rt(self) -> float | None:
        return self._py.z_rt

    @property
    def z_im(self) -> float | None:
        return self._py.z_im

    @property
    def s_shape(self) -> float | None:
        return self._py.s_shape

    # XIC details (None in Geom mode)

    @property
    def xic_s_rt(self) -> float | None:
        """RT XIC similarity in [0,1], or None if not used."""
        return self._py.xic_s_rt

    @property
    def xic_s_im(self) -> float | None:
        """IM XIC similarity in [0,1], or None if not used."""
        return self._py.xic_s_im

    @property
    def xic_s_intensity(self) -> float | None:
        """Intensity-ratio term in (0,1], or None if not used."""
        return self._py.xic_s_intensity

    @property
    def xic_r_rt(self) -> float | None:
        """Raw RT Pearson correlation in [-1,1], or None."""
        return self._py.xic_r_rt

    @property
    def xic_r_im(self) -> float | None:
        """Raw IM Pearson correlation in [-1,1], or None."""
        return self._py.xic_r_im

    def __repr__(self) -> str:
        return f"ScoreHit(ms2_index={self.ms2_index}, score={self.score:.4f})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ms2_index": self.ms2_index,
            "score": self.score,
            "jacc_rt": self.jacc_rt,
            "rt_apex_delta_s": self.rt_apex_delta_s,
            "im_apex_delta_scans": self.im_apex_delta_scans,
            "im_overlap_scans": self.im_overlap_scans,
            "im_union_scans": self.im_union_scans,
            "ms1_raw_sum": self.ms1_raw_sum,
            "shape_ok": self.shape_ok,
            "z_rt": self.z_rt,
            "z_im": self.z_im,
            "s_shape": self.s_shape,
            "xic_s_rt": self.xic_s_rt,
            "xic_s_im": self.xic_s_im,
            "xic_s_intensity": self.xic_s_intensity,
            "xic_r_rt": self.xic_r_rt,
            "xic_r_im": self.xic_r_im,
        }


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

    # ------------------------------------------------------------------
    # Unscored candidate queries
    # ------------------------------------------------------------------

    def query_precursor(
        self,
        precursor_cluster: "ClusterResult1D",
        window_groups: list[int] | None = None,
        *,
        max_rt_apex_delta_sec: float | None = 2.0,
        max_scan_apex_delta: int | None = 6,
        min_im_overlap_scans: int = 1,
        require_tile_compat: bool = True,
        reject_frag_inside_precursor_tile: bool = True,
    ) -> list[int]:
        return self._py.query_precursor(
            precursor_cluster.get_py_ptr(),
            window_groups,  # None => let Rust compute from DiaIndex
            max_rt_apex_delta_sec=max_rt_apex_delta_sec,
            max_scan_apex_delta=max_scan_apex_delta,
            min_im_overlap_scans=min_im_overlap_scans,
            require_tile_compat=require_tile_compat,
            reject_frag_inside_precursor_tile=reject_frag_inside_precursor_tile,
        )

    def query_precursors(
        self,
        precursor_clusters: list["ClusterResult1D"],
        *,
        max_rt_apex_delta_sec: float | None = 2.0,
        max_scan_apex_delta: int | None = 6,
        min_im_overlap_scans: int = 1,
        require_tile_compat: bool = True,
        reject_frag_inside_precursor_tile: bool = True,
        num_threads: int = 4,
    ) -> list[list[int]]:
        return self._py.query_precursors_par(
            [c.get_py_ptr() for c in precursor_clusters],
            max_rt_apex_delta_sec=max_rt_apex_delta_sec,
            max_scan_apex_delta=max_scan_apex_delta,
            min_im_overlap_scans=min_im_overlap_scans,
            require_tile_compat=require_tile_compat,
            reject_frag_inside_precursor_tile=reject_frag_inside_precursor_tile,
            num_threads=num_threads,
        )

    # ------------------------------------------------------------------
    # Scored queries – now with tile-reject knob
    # ------------------------------------------------------------------

    def query_precursor_scored(
            self,
            precursor_cluster: "ClusterResult1D",
            window_groups: list[int] | None = None,
            *,
            mode: str = "geom",  # or "xic"
            min_score: float = 0.0,
            reject_frag_inside_precursor_tile: bool = True,
            max_rt_apex_delta_sec: float | None = 2.0,
            max_scan_apex_delta: int | None = 6,
            min_im_overlap_scans: int = 1,
            require_tile_compat: bool = True,
    ) -> list[ScoredHit]:
        hits_py = self._py.query_precursor_scored(
            precursor_cluster.get_py_ptr(),
            window_groups,
            mode=mode,
            min_score=min_score,
            reject_frag_inside_precursor_tile=reject_frag_inside_precursor_tile,
            max_rt_apex_delta_sec=max_rt_apex_delta_sec,
            max_scan_apex_delta=max_scan_apex_delta,
            min_im_overlap_scans=min_im_overlap_scans,
            require_tile_compat=require_tile_compat,
        )
        return [ScoredHit.from_py_ptr(h) for h in hits_py]

    def query_precursors_scored(
            self,
            precursor_clusters: list["ClusterResult1D"],
            *,
            mode: str = "geom",
            min_score: float = 0.0,
            reject_frag_inside_precursor_tile: bool = True,
            max_rt_apex_delta_sec: float | None = 2.0,
            max_scan_apex_delta: int | None = 6,
            min_im_overlap_scans: int = 1,
            require_tile_compat: bool = True,
    ) -> list[list[ScoredHit]]:
        hits_nested = self._py.query_precursors_scored_par(
            [c.get_py_ptr() for c in precursor_clusters],
            mode=mode,
            min_score=min_score,
            reject_frag_inside_precursor_tile=reject_frag_inside_precursor_tile,
            max_rt_apex_delta_sec=max_rt_apex_delta_sec,
            max_scan_apex_delta=max_scan_apex_delta,
            min_im_overlap_scans=min_im_overlap_scans,
            require_tile_compat=require_tile_compat,
        )
        return [
            [ScoredHit.from_py_ptr(h) for h in hits_row]
            for hits_row in hits_nested
        ]

    def score_feature(
            self,
            feature: "SimpleFeature",
            *,
            mode: str = "geom",
            min_score: float = 0.0,
            reject_frag_inside_precursor_tile: bool = True,
            max_rt_apex_delta_sec: float | None = 2.0,
            max_scan_apex_delta: int | None = 6,
            min_im_overlap_scans: int = 1,
            require_tile_compat: bool = True,
    ) -> list["ScoredHit"]:
        hits_py = self._py.score_feature(
            feature.get_py_ptr(),
            mode=mode,
            min_score=min_score,
            reject_frag_inside_precursor_tile=reject_frag_inside_precursor_tile,
            max_rt_apex_delta_sec=max_rt_apex_delta_sec,
            max_scan_apex_delta=max_scan_apex_delta,
            min_im_overlap_scans=min_im_overlap_scans,
            require_tile_compat=require_tile_compat,
        )
        return [ScoredHit.from_py_ptr(h) for h in hits_py]

    def score_features(
            self,
            features: list["SimpleFeature"],
            *,
            mode: str = "geom",
            min_score: float = 0.0,
            reject_frag_inside_precursor_tile: bool = True,
            max_rt_apex_delta_sec: float | None = 2.0,
            max_scan_apex_delta: int | None = 6,
            min_im_overlap_scans: int = 1,
            require_tile_compat: bool = True,
    ) -> list[list["ScoredHit"]]:
        hits_nested = self._py.score_features_par(
            [f.get_py_ptr() for f in features],
            mode=mode,
            min_score=min_score,
            reject_frag_inside_precursor_tile=reject_frag_inside_precursor_tile,
            max_rt_apex_delta_sec=max_rt_apex_delta_sec,
            max_scan_apex_delta=max_scan_apex_delta,
            min_im_overlap_scans=min_im_overlap_scans,
            require_tile_compat=require_tile_compat,
        )
        return [
            [ScoredHit.from_py_ptr(h) for h in hits_row]
            for hits_row in hits_nested
        ]

    def score_features_to_pseudospectra(
            self,
            features: list["SimpleFeature"],
            *,
            mode: str = "geom",
            min_score: float = 0.0,
            reject_frag_inside_precursor_tile: bool = True,
            max_rt_apex_delta_sec: float | None = 2.0,
            max_scan_apex_delta: int | None = 6,
            min_im_overlap_scans: int = 1,
            require_tile_compat: bool = True,
            min_fragments: int = 4,
            max_fragments: int = 512,
    ) -> list["PseudoSpectrum"]:
        from imspy.timstof.clustering.pseudo import PseudoSpectrum
        """
        Parallel scoring of SimpleFeatures AND construction of PseudoSpectra.

        Returns one PseudoSpectrum per feature with at least `min_fragments`
        surviving fragment hits.
        """
        py_specs = self._py.score_features_to_pseudospectra_par(
            [f.get_py_ptr() for f in features],
            mode=mode,
            min_score=min_score,
            reject_frag_inside_precursor_tile=reject_frag_inside_precursor_tile,
            max_rt_apex_delta_sec=max_rt_apex_delta_sec,
            max_scan_apex_delta=max_scan_apex_delta,
            min_im_overlap_scans=min_im_overlap_scans,
            require_tile_compat=require_tile_compat,
            min_fragments=min_fragments,
            max_fragments=max_fragments,
        )
        return [PseudoSpectrum(ps) for ps in py_specs]

    def score_precursors_to_pseudospectra(
            self,
            precursor_clusters: list["ClusterResult1D"],
            *,
            mode: str = "geom",
            min_score: float = 0.0,
            reject_frag_inside_precursor_tile: bool = True,
            max_rt_apex_delta_sec: float | None = 2.0,
            max_scan_apex_delta: int | None = 6,
            min_im_overlap_scans: int = 1,
            require_tile_compat: bool = True,
            min_fragments: int = 4,
            max_fragments: int = 512,
    ) -> list["PseudoSpectrum"]:
        from imspy.timstof.clustering.pseudo import PseudoSpectrum
        """
        Parallel scoring of precursor clusters AND construction of PseudoSpectra.

        Returns one PseudoSpectrum per cluster with at least `min_fragments`
        surviving fragment hits.
        """
        py_specs = self._py.query_precursors_to_pseudospectra_par(
            [c.get_py_ptr() for c in precursor_clusters],
            mode=mode,
            min_score=min_score,
            reject_frag_inside_precursor_tile=reject_frag_inside_precursor_tile,
            max_rt_apex_delta_sec=max_rt_apex_delta_sec,
            max_scan_apex_delta=max_scan_apex_delta,
            min_im_overlap_scans=min_im_overlap_scans,
            require_tile_compat=require_tile_compat,
            min_fragments=min_fragments,
            max_fragments=max_fragments,
        )
        return [PseudoSpectrum(ps) for ps in py_specs]

    @classmethod
    def from_parquet_dir(
            cls,
            ds: "TimsDatasetDIA",
            parquet_dir: str,
            cand_opts: "CandidateOpts",
    ) -> "FragmentIndex":
        """
        Build a FragmentIndex directly from a directory of fragment cluster parquet files.
        """
        p = ims.PyFragmentIndex.from_parquet_dir(
            ds.get_py_ptr(),
            parquet_dir,
            cand_opts.get_py_ptr(),  # or cand_opts.inner depending on your pattern
        )
        return cls.from_py_ptr(p)


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
            # NEW: distance-based merge of duplicates (within same WG)
            merge_duplicates: bool = False,
            max_rt_center_delta: float = 0.1,
            max_im_center_delta: float = 5.0,
            max_tof_center_delta: float = 2.0,
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
        merge_duplicates : bool, default False
            If True, merge clusters that are very close in RT/IM/TOF centers
            within the same window_group according to the *_center_delta.
        max_*_center_delta : float
            Center-difference thresholds in index units (frames/scans/bins).
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
            bool(merge_duplicates),
            float(max_rt_center_delta),
            float(max_im_center_delta),
            float(max_tof_center_delta),
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
            # NEW: distance-based merge of duplicates (within same WG)
            merge_duplicates: bool = False,
            max_rt_center_delta: float = 0.1,
            max_im_center_delta: float = 5.0,
            max_tof_center_delta: float = 2.0,
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
            bool(merge_duplicates),
            float(max_rt_center_delta),
            float(max_im_center_delta),
            float(max_tof_center_delta),
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

    def debug_extract_raw_for_clusters(
            self,
            clusters: list["ClusterResult1D"],
            window_group: Optional[int] = None,
            tof_step: int = 1,
            max_points: Optional[int] = None,
            tof_pad: Optional[int] = None,
            rt_pad: Option[int] = None,
            scan_pad: Optional[int] = None,
            num_threads: int = 4,
    ) -> list["RawPoints"]:
        from imspy.timstof.clustering.data import RawPoints

        raw_points = self.get_py_ptr().debug_extract_raw_for_clusters(
            [c.get_py_ptr() for c in clusters],
            window_group,
            tof_step,
            max_points,
            tof_pad,
            rt_pad,
            scan_pad,
            num_threads,
        )

        return [RawPoints(rp) for rp in raw_points]

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

_PARQUET_SUFFIX = ".parquet"

def _normalize_parquet_path(path: Union[str, Path]) -> Path:
    """
    Ensure a .parquet suffix. If a different suffix is given, warn and replace it.
    """
    p = Path(path)
    if p.suffix == "":
        return p.with_suffix(_PARQUET_SUFFIX)
    if p.suffix.lower() != _PARQUET_SUFFIX:
        warnings.warn(
            f"Unexpected suffix '{p.suffix}'. Using '{_PARQUET_SUFFIX}' instead.",
            stacklevel=2,
        )
        return p.with_suffix(_PARQUET_SUFFIX)
    return p

def save_clusters_parquet(
    path: Union[str, Path],
    clusters: Sequence["ClusterResult1D"],
    strip_points: bool = True,
    strip_axes: bool = True,
    *,
    overwrite: bool = True,
    atomic: bool = True,
) -> None:
    """
    Save clusters to a Parquet file (.parquet).

    Notes
    -----
    Parquet only stores the lightweight, tabular fields:
      - windows, fits, intensities, IDs, ms_level, etc.
    Heavy fields (raw_points, axes, traces) are *never* persisted, even if
    `strip_points=False` / `strip_axes=False`; those flags only control whether
    they are kept in memory before serialization.

    Args:
        path:
            Target path (str or Path). Any suffix is normalized to ``.parquet``.
        clusters:
            Sequence of ClusterResult1D instances.
        strip_points:
            If True, drop `raw_points` before serializing (recommended).
        strip_axes:
            If True, drop `rt_axis_sec`, `im_axis_scans`, `mz_axis_da` before
            serializing (recommended).
        overwrite:
            If False, raise FileExistsError if the target already exists.
        atomic:
            If True, write to a temporary file in the same directory and
            atomically replace the target.
    """
    from imspy.timstof.clustering.data import ClusterResult1D  # for isinstance in _assert_clusters if needed

    _assert_clusters(clusters)

    p = _normalize_parquet_path(path)

    if not overwrite and p.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {p}")

    if strip_points or strip_axes:
        kept = []
        if not strip_points:
            kept.append("points")
        if not strip_axes:
            kept.append("axes")
        warnings.warn(
            "Saving clusters to Parquet without heavy fields; "
            f"kept in-memory: {', '.join(kept) if kept else 'none'}.",
            stacklevel=2,
        )

    # Underlying PyO3 wrapper objects
    rust_clusters = [c._py for c in clusters]

    _ensure_dir(p)

    if atomic:
        tmp_dir = str(p.parent)
        with tempfile.NamedTemporaryFile(
            prefix=".tmp_", suffix=_PARQUET_SUFFIX, dir=tmp_dir, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            ims.save_clusters_parquet(
                str(tmp_path),
                rust_clusters,
                bool(strip_points),
                bool(strip_axes),
            )
            os.replace(str(tmp_path), str(p))
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            finally:
                raise
    else:
        ims.save_clusters_parquet(
            str(p),
            rust_clusters,
            bool(strip_points),
            bool(strip_axes),
        )

def load_clusters_parquet(path: Union[str, Path]) -> List["ClusterResult1D"]:
    """
    Load clusters from a Parquet file (.parquet).

    This reconstructs lightweight ``ClusterResult1D`` objects:
    raw_points, axes, and traces are always ``None`` / empty.

    Args:
        path:
            File to load (str or Path). Any suffix is accepted but will emit
            a warning if it is not ``.parquet``.

    Returns:
        list[ClusterResult1D]
    """
    from imspy.timstof.clustering.data import ClusterResult1D

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")
    if p.suffix.lower() != _PARQUET_SUFFIX:
        warnings.warn(
            f"Unexpected suffix '{p.suffix}'. Expected '{_PARQUET_SUFFIX}'. "
            "Attempting to load anyway.",
            stacklevel=2,
        )

    rust_clusters = ims.load_clusters_parquet(str(p))
    return [ClusterResult1D(c) for c in rust_clusters]

def _assert_pseudo_spectra(spectra: Sequence["PseudoSpectrum"]) -> None:
    from imspy.timstof.clustering.pseudo import PseudoSpectrum
    if not isinstance(spectra, (list, tuple)):
        raise TypeError(f"Expected a sequence of PseudoSpectrum, got {type(spectra)!r}")
    for s in spectra:
        if not isinstance(s, PseudoSpectrum):
            raise TypeError(f"Expected PseudoSpectrum, got {type(s)!r}")


def save_pseudo_spectra_bin(
    path: Union[str, Path],
    spectra: Sequence["PseudoSpectrum"],
    compress: bool = True,
    *,
    overwrite: bool = True,
    atomic: bool = True,
) -> None:
    """
    Save pseudo spectra to a bincode file (.bin / .binz).

    Args:
        path:
            Target path. If no suffix is given, one is added based on
            `compress` (.binz if True, .bin if False). If a conflicting
            suffix is given, it is normalized with a warning.

        spectra:
            Sequence of PseudoSpectrum objects.

        compress:
            Whether to use zstd compression (controls suffix normalization).

        overwrite:
            If False, refuse to overwrite existing files.

        atomic:
            If True, write to a temporary file in the same directory and
            atomically replace the target.
    """

    from imspy.timstof.clustering.pseudo import PseudoSpectrum

    _assert_pseudo_spectra(spectra)

    # Normalize suffix and possibly infer compress from it
    path, compress = _normalize_path_and_compress(
        path,
        compress,
        allow_suffix_inference=True,
    )

    p = Path(path)

    if not overwrite and p.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {p}")

    rust_spectra = [s._py for s in spectra]  # mirror the ClusterResult1D pattern

    _ensure_dir(p)

    if atomic:
        tmp_dir = str(p.parent)
        suffix = p.suffix
        with tempfile.NamedTemporaryFile(
            prefix=".tmp_",
            suffix=suffix,
            dir=tmp_dir,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            ims.save_pseudo_spectra_bin(str(tmp_path), rust_spectra, bool(compress))
            os.replace(str(tmp_path), str(p))
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            finally:
                raise
    else:
        ims.save_pseudo_spectra_bin(str(p), rust_spectra, bool(compress))


def load_pseudo_spectra_bin(path: Union[str, Path]) -> List["PseudoSpectrum"]:
    """
    Load pseudo spectra from a bincode file (.bin / .binz).

    Args:
        path:
            File to load.

    Returns:
        list[PseudoSpectrum]
    """

    from imspy.timstof.clustering.pseudo import PseudoSpectrum

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    if p.suffix.lower() not in {_BIN_SUFFIX, _BINZ_SUFFIX}:
        warnings.warn(
            f"Unexpected suffix '{p.suffix}'. Expected '{_BIN_SUFFIX}' or '{_BINZ_SUFFIX}'. "
            "Attempting to load anyway.",
            stacklevel=2,
        )

    rust_spectra = ims.load_pseudo_spectra_bin(str(p))
    # Wrap back into Python PseudoSpectrum objects
    return [PseudoSpectrum(s) for s in rust_spectra]