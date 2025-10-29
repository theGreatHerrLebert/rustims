# imspy/cluster.py
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

import imspy_connector

from imspy.timstof.slice import TimsSlice

ims = imspy_connector.py_cluster
from imspy.simulation.annotation import RustWrapperObject

class RawPoints(RustWrapperObject):
    """Thin wrapper around Rust RawPoints (SoA)."""
    def __init__(self, *a, **k):
        raise RuntimeError("RawPoints is created in Rust; use via ClusterResult.raw_points.")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyRawPoints") -> "RawPoints":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    # numpy accessors
    @property
    def mz(self) -> np.ndarray:        return np.asarray(self.__py_ptr.mz, dtype=np.float32)
    @property
    def rt(self) -> np.ndarray:        return np.asarray(self.__py_ptr.rt, dtype=np.float32)
    @property
    def im(self) -> np.ndarray:        return np.asarray(self.__py_ptr.im, dtype=np.float32)   # mobility (1/K0) if provided
    @property
    def scan(self) -> np.ndarray:      return np.asarray(self.__py_ptr.scan, dtype=np.uint32)
    @property
    def intensity(self) -> np.ndarray: return np.asarray(self.__py_ptr.intensity, dtype=np.float32)
    @property
    def tof(self) -> np.ndarray:       return np.asarray(self.__py_ptr.tof, dtype=np.int32)
    @property
    def frame(self) -> np.ndarray:     return np.asarray(self.__py_ptr.frame, dtype=np.uint32)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "mz": self.mz,
            "rt": self.rt,
            "im": self.im,
            "scan": self.scan,
            "intensity": self.intensity,
            "tof": self.tof,
            "frame": self.frame,
        })

    def to_tims_slice(self) -> "TimsSlice":
        return TimsSlice(
            frame_id=self.frame.astype(np.uint32),
            scan=self.scan.astype(np.uint32),
            tof=self.tof.astype(np.int32),
            retention_time=self.rt.astype(np.float32),
            mobility=self.im.astype(np.float32),
            mz=self.mz.astype(np.float32),
            intensity=self.intensity.astype(np.float32),
        )

    def __len__(self) -> int:
        return int(self.mz.shape[0])

    def __repr__(self) -> str:
        return f"RawPoints(n={len(self)})"

class ClusterSpec(RustWrapperObject):
    """Thin wrapper for Rust PyClusterSpec."""
    def __init__(self, *args, **kwargs):
        # prefer from_py_ptr; constructing directly uses the rust ctor:
        rt_left      = kwargs.get("rt_left")
        rt_right     = kwargs.get("rt_right")
        im_left      = kwargs.get("im_left")
        im_right     = kwargs.get("im_right")
        mz_center    = kwargs.get("mz_center_hint")
        mz_ppm       = kwargs.get("mz_ppm_window")
        extra_rt_pad = kwargs.get("extra_rt_pad", 0)
        extra_im_pad = kwargs.get("extra_im_pad", 0)
        mz_bins      = kwargs.get("mz_hist_bins", 64)
        if None in (rt_left, rt_right, im_left, im_right, mz_center, mz_ppm):
            raise RuntimeError("Use ClusterSpec.from_py_ptr(...) or provide all required args.")
        self.__py_ptr = ims.PyClusterSpec(
            rt_left, rt_right, im_left, im_right,
            float(mz_center), float(mz_ppm),
            int(extra_rt_pad), int(extra_im_pad), int(mz_bins)
        )

    @classmethod
    def from_py_ptr(cls, p: "ims.PyClusterSpec") -> "ClusterSpec":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    # properties (read back from Rust object)
    @property
    def rt_left(self) -> int: return self.__py_ptr.rt_left
    @property
    def rt_right(self) -> int: return self.__py_ptr.rt_right
    @property
    def im_left(self) -> int: return self.__py_ptr.im_left
    @property
    def im_right(self) -> int: return self.__py_ptr.im_right
    @property
    def mz_center_hint(self) -> float: return self.__py_ptr.mz_center_hint
    @property
    def mz_ppm_window(self) -> float: return self.__py_ptr.mz_ppm_window
    @property
    def extra_rt_pad(self) -> int: return self.__py_ptr.extra_rt_pad
    @property
    def extra_im_pad(self) -> int: return self.__py_ptr.extra_im_pad
    @property
    def mz_hist_bins(self) -> int: return self.__py_ptr.mz_hist_bins

    def __repr__(self) -> str:
        return (f"ClusterSpec(rt=[{self.rt_left},{self.rt_right}], "
                f"im=[{self.im_left},{self.im_right}], "
                f"mz≈{self.mz_center_hint:.5f}±{self.mz_ppm_window}ppm, "
                f"pads(rt={self.extra_rt_pad}, im={self.extra_im_pad}), "
                f"mz_bins={self.mz_hist_bins})")


class AttachOptions(RustWrapperObject):
    """Thin wrapper for Rust PyAttachOptions."""
    def __init__(
        self,
        attach_frames: bool = True,
        attach_scans: bool = True,
        attach_mz_axis: bool = True,
        attach_points: bool = False,
        max_points: Optional[int] = None,
    ):
        self.__py_ptr = ims.PyAttachOptions(
            bool(attach_frames),
            bool(attach_scans),
            bool(attach_mz_axis),
            bool(attach_points),
            None if max_points is None else int(max_points),
        )

    @classmethod
    def from_py_ptr(cls, p: "ims.PyAttachOptions") -> "AttachOptions":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    @property
    def attach_frames(self) -> bool: return self.__py_ptr.attach_frames
    @property
    def attach_scans(self) -> bool: return self.__py_ptr.attach_scans
    @property
    def attach_mz_axis(self) -> bool: return self.__py_ptr.attach_mz_axis

    # NEW:
    @property
    def attach_points(self) -> bool: return self.__py_ptr.attach_points
    @property
    def max_points(self) -> Optional[int]: return self.__py_ptr.max_points

    def __repr__(self) -> str:
        return (f"AttachOptions(frames={self.attach_frames}, scans={self.attach_scans}, "
                f"mz_axis={self.attach_mz_axis}, points={self.attach_points}, "
                f"max_points={self.max_points})")


class EvalOptions(RustWrapperObject):
    """Thin wrapper for Rust PyEvalOptions."""
    def __init__(
        self,
        attach: AttachOptions | None = None,
        refine_mz_once: bool = False,
        refine_k_sigma: float = 3.0,
        im_k_sigma: float | None = None,
        im_min_width: int = 1,
        max_rt_span_frames: int = 50,
        max_im_span_scans: int = 100,
        ms_level: int = 0,
        window_group_hint: None | int = None,
    ):
        if attach is None:
            attach = AttachOptions()
        self.__py_ptr = ims.PyEvalOptions(
            attach.get_py_ptr(),
            bool(refine_mz_once),
            float(refine_k_sigma),
            None if im_k_sigma is None else float(im_k_sigma),
            int(im_min_width),
            int(max_rt_span_frames),
            int(max_im_span_scans),
            int(ms_level),
            None if window_group_hint is None else int(window_group_hint),
        )

    @classmethod
    def from_py_ptr(cls, p: "ims.PyEvalOptions") -> "EvalOptions":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    @property
    def refine_mz_once(self) -> bool:
        return self.__py_ptr.refine_mz_once

    @property
    def refine_k_sigma(self) -> float:
        return self.__py_ptr.refine_k_sigma

    @property
    def im_k_sigma(self) -> float | None:
        return self.__py_ptr.im_k_sigma

    @property
    def im_min_width(self) -> int:
        return self.__py_ptr.im_min_width

    @property
    def max_rt_span_frames(self) -> int:
        return self.__py_ptr.max_rt_span_frames

    @property
    def max_im_span_scans(self) -> int:
        return self.__py_ptr.max_im_span_scans

    @property
    def ms_level(self) -> int:
        return self.__py_ptr.ms_level

    @property
    def window_group_hint(self) -> Optional[int]:
        return self.__py_ptr.window_group_hint

    def __repr__(self) -> str:
        return (
            f"EvalOptions(refine_mz_once={self.refine_mz_once}, "
            f"k={self.refine_k_sigma}, im_k_sigma={self.im_k_sigma}, "
            f"im_min_width={self.im_min_width}, "
            f"max_rt_span_frames={self.max_rt_span_frames}, "
            f"max_im_span_scans={self.max_im_span_scans}, "
            f"ms_level={self.ms_level}, "f"window_group_hint={self.window_group_hint}, "
            f"attach={AttachOptions.from_py_ptr(self.__py_ptr.attach)})"
        )


class ClusterFit1D(RustWrapperObject):
    """Thin wrapper for Rust ClusterFit1D (read-only)."""
    def __init__(self, *a, **k):
        raise RuntimeError("ClusterFit1D is created in Rust; use ClusterFit1D.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyClusterFit1D") -> "ClusterFit1D":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    # fields
    @property
    def mu(self) -> float: return self.__py_ptr.mu
    @property
    def sigma(self) -> float: return self.__py_ptr.sigma
    @property
    def height(self) -> float: return self.__py_ptr.height
    @property
    def baseline(self) -> float: return self.__py_ptr.baseline
    @property
    def area(self) -> float: return self.__py_ptr.area
    @property
    def r2(self) -> float: return self.__py_ptr.r2
    @property
    def n(self) -> int: return self.__py_ptr.n

    def __repr__(self) -> str:
        return f"Fit1D(mu={self.mu:.3f}, σ={self.sigma:.3f}, h={self.height:.1f}, base={self.baseline:.1f}, area={self.area:.1f})"

    def get_py_ptr(self):
        return self.__py_ptr


class ClusterResult(RustWrapperObject):
    """Thin wrapper for Rust ClusterResult (read-only)."""
    def __init__(self, *a, **k):
        raise RuntimeError("ClusterResult is created in Rust; use ClusterResult.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyClusterResult") -> "ClusterResult":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    @property
    def cluster_id(self) -> int: return self.__py_ptr.id
    # windows / meta
    @property
    def rt_window(self) -> Tuple[int, int]: return tuple(self.__py_ptr.rt_window)
    @property
    def im_window(self) -> Tuple[int, int]: return tuple(self.__py_ptr.im_window)
    @property
    def mz_window_da(self) -> Tuple[float, float]: return tuple(self.__py_ptr.mz_window_da)

    # fits
    @property
    def rt_fit(self) -> "ClusterFit1D": return ClusterFit1D.from_py_ptr(self.__py_ptr.rt_fit)
    @property
    def im_fit(self) -> "ClusterFit1D": return ClusterFit1D.from_py_ptr(self.__py_ptr.im_fit)
    @property
    def mz_fit(self) -> "ClusterFit1D": return ClusterFit1D.from_py_ptr(self.__py_ptr.mz_fit)

    # intensity summaries
    @property
    def raw_sum(self) -> float: return self.__py_ptr.raw_sum
    @property
    def fit_volume(self) -> float: return self.__py_ptr.fit_volume

    # provenance
    @property
    def rt_peak_id(self) -> int: return self.__py_ptr.rt_peak_id
    @property
    def im_peak_id(self) -> int: return self.__py_ptr.im_peak_id
    @property
    def mz_center_hint(self) -> float: return self.__py_ptr.mz_center_hint
    @property
    def ms_level(self) -> int: return self.__py_ptr.ms_level
    @property
    def window_group(self) -> Optional[int]:
        wg = self.__py_ptr.window_group
        return None if wg is None else int(wg)

    @property
    def window_groups_covering_mz(self) -> Optional[np.ndarray]:
        wgc = self.__py_ptr.window_groups_covering_mz
        return None if wgc is None else np.asarray(wgc, dtype=np.uint32)

    # attached axes
    @property
    def frame_ids_used(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.frame_ids_used, dtype=np.uint32)

    @property
    def frames_axis(self) -> Optional[np.ndarray]:
        fa = self.__py_ptr.frames_axis
        return None if fa is None else np.asarray(fa, dtype=np.uint32)

    @property
    def scans_axis(self) -> Optional[np.ndarray]:
        sa = self.__py_ptr.scans_axis
        return None if sa is None else np.asarray(sa, dtype=np.int64)

    @property
    def mz_axis(self) -> Optional[np.ndarray]:
        ma = self.__py_ptr.mz_axis
        return None if ma is None else np.asarray(ma, dtype=np.float32)

    # NEW: raw points
    @property
    def raw_points(self) -> Optional[RawPoints]:
        rp = self.__py_ptr.raw_points
        return None if rp is None else RawPoints.from_py_ptr(rp)

    def __repr__(self) -> str:
        (rt_l, rt_r) = self.rt_window
        (im_l, im_r) = self.im_window
        npts = 0 if self.raw_points is None else len(self.raw_points)
        return (f"ClusterResult(id={self.cluster_id}, "
                f"rt=[{rt_l},{rt_r}], im=[{im_l},{im_r}], "
                f"mz=({self.mz_window_da[0]:.4f},{self.mz_window_da[1]:.4f}), "
                f"sum={self.raw_sum:.1f}, points={npts}, "
                f"rt_fit={self.rt_fit}, im_fit={self.im_fit}, mz_fit={self.mz_fit}, "
                f"rt_peak_id={self.rt_peak_id}, im_peak_id={self.im_peak_id}, "
                f"mz_center_hint={self.mz_center_hint:.4f}, ms_level={self.ms_level}, "
                f"window_group={self.window_group}, "
                f"window_groups_covering_mz={self.window_groups_covering_mz}, "
                f"frames_axis={'yes' if self.frames_axis is not None else 'no'}, "
                f"scans_axis={'yes' if self.scans_axis is not None else 'no'}, "
                f"mz_axis={'yes' if self.mz_axis is not None else 'no'}, "
                f"raw_points={'yes' if self.raw_points is not None else 'no'})")

class LinkCandidate:
    """Thin wrapper for Rust LinkCandidate (read-only)."""
    def __init__(self, *a, **k):
        raise RuntimeError("LinkCandidate is created in Rust; use LinkCandidate.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyLinkCandidate") -> "LinkCandidate":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    @property
    def ms1_idx(self) -> int: return self.__py_ptr.ms1_idx
    @property
    def ms2_idx(self) -> int: return self.__py_ptr.ms2_idx
    @property
    def ms1_id(self) -> int: return self.__py_ptr.ms1_id
    @property
    def ms2_id(self) -> int: return self.__py_ptr.ms2_id
    @property
    def score(self) -> float: return self.__py_ptr.score
    @property
    def group(self) -> int: return self.__py_ptr.group

    def __repr__(self) -> str:
        return (f"LinkCandidate(ms1_idx={self.ms1_idx}, ms2_idx={self.ms2_idx}, "
                f"ms1_id={self.ms1_id}, ms2_id={self.ms2_id}, "
                f"score={self.score:.5f}, group={self.group})")

def link_ms2_to_ms1(
    ms1: Sequence["ClusterResult"],
    ms2: Sequence["ClusterResult"],
    *,
    min_rt_jaccard: float = 0.1,
    max_rt_apex_sec: float = 5.0,
    max_im_apex_scans: Optional[float] = None,
) -> List["LinkCandidate"]:
    """
    Link MS2 clusters to MS1 clusters based on RT/IM overlap and apex proximity.

    Args
    ----
    ms1 : sequence of ClusterResult (MS1 clusters)
    ms2 : sequence of ClusterResult (MS2 clusters)
    min_rt_jaccard : minimum Jaccard index for RT overlap (0.0-1.0)
    max_rt_apex_sec : maximum RT difference between apexes (in seconds)
    max_im_apex_scans : optional maximum IM scan difference between apexes

    Returns
    -------
    List[LinkCandidate]

    Note
    ----
    This function assumes that the RT units in the ClusterResult are in seconds.
    If your RT units are in frames, you need to convert `max_rt_apex_sec` accordingly.
    """
    ms1_py = [c.get_py_ptr() for c in ms1]
    ms2_py = [c.get_py_ptr() for c in ms2]

    out_py = ims.link_ms2_to_ms1(
        ms1_py, ms2_py,
        float(min_rt_jaccard),
        float(max_rt_apex_sec),
        None if max_im_apex_scans is None else float(max_im_apex_scans),
    )
    return [LinkCandidate.from_py_ptr(p) for p in out_py]

# --- Convenience function: build specs from peaks (fully wrapped) ------------

def make_cluster_specs_from_peaks(
    rt_peaks: Sequence["RtPeak1D"],
    im_rows: Sequence[Sequence["ImPeak1D"]],
    im_scans: List[int],
    *,
    mz_ppm_window: float = 15.0,
    extra_rt_pad: int = 0,
    extra_im_pad: int = 0,
    mz_hist_bins: int = 64,
) -> List[ClusterSpec]:
    """
    Build ClusterSpec objects from aligned RT peaks and their IM peak rows.

    Args
    ----
    rt_peaks : sequence of RtPeak1D, length = R
    im_rows  : sequence (length R) of sequences of ImPeak1D (one list per RT row)
    im_scans : list of int, the IM scan indices corresponding to the im_rows
    mz_ppm_window : ±ppm window around each rt_peak.mz_center_hint
    extra_rt_pad  : extra frames to include around the RT padded window
    extra_im_pad  : extra scans to include around the IM window
    mz_hist_bins  : number of bins for the m/z marginal histogram inside the cluster

    Returns
    -------
    List[ClusterSpec]

    Args:
        im_scans:
    """
    if len(rt_peaks) != len(im_rows):
        raise ValueError("rt_peaks and im_rows must have the same length (row-aligned).")

    # Convert to raw PyO3 objects (no user poking at pointers)
    rt_py = [p.get_py_ptr() for p in rt_peaks]
    im_py = [[q.get_py_ptr() for q in row] for row in im_rows]

    out_py = ims.make_cluster_specs_from_peaks(
        rt_py, im_py, im_scans,
        mz_ppm_window, extra_rt_pad, extra_im_pad, mz_hist_bins
    )
    return [ClusterSpec.from_py_ptr(p) for p in out_py]

def build_precursor_fragment_annotation(
    ms1: Sequence["ClusterResult"],
    ms2: Sequence["ClusterResult"],
    candidates: Sequence["LinkCandidate"],
    *,
    min_score: float = 0.0,
) -> List[Tuple["ClusterResult", List["ClusterResult"]]]:
    """
    Build precursor-fragment annotations from MS1/MS2 clusters and link candidates.

    Args
    ----
    ms1 : sequence of ClusterResult (MS1 clusters)
    ms2 : sequence of ClusterResult (MS2 clusters)
    candidates : sequence of LinkCandidate (from link_ms2_to_ms1)
    min_score : minimum score to consider a candidate link
    one_to_one : if True, enforce one-to-one mapping (greedy by score)
    top_k_per_ms2 : max number of MS1 links to keep per MS2 cluster
    top_k_per_ms1 : optional max number of MS2 links to keep per MS1 cluster

    Returns
    -------
    List of (ms1_cluster, [ms2_clusters]) tuples

    Note
    ----
    Each MS1 cluster may appear at most once in the output list.
    An MS1 cluster may have zero or more linked MS2 clusters.
    """
    ms1_py = [c.get_py_ptr() for c in ms1]
    ms2_py = [c.get_py_ptr() for c in ms2]
    cand_py = [c.get_py_ptr() for c in candidates]

    out_py = ims.build_precursor_fragment_annotation_py(
        ms1_py, ms2_py, cand_py,
        float(min_score),
    )
    out = []
    for (pms1, pms2_list) in out_py:
        cms1 = ClusterResult.from_py_ptr(pms1)
        cms2_list = [ClusterResult.from_py_ptr(p) for p in pms2_list]
        out.append((cms1, cms2_list))
    return out

# --- I/O helpers --------------------------------------------------------------

def save_clusters(
    path: str,
    clusters: Sequence["ClusterResult"],
    *,
    fmt: str | None = None,
    compress: bool = True,
    strip_points: bool = True,
    strip_axes: bool = False,
) -> None:
    """
    Save a list of ClusterResult to disk.

    Parameters
    ----------
    path : str
        Output path. If `fmt` is None, we infer from extension:
        - .json  -> JSON (pretty)
        - .bin   -> bincode (no compression)
        - .zst / .bin.zst -> bincode + zstd
    fmt : {"json","bin"} | None
        Explicit format. If None, inferred from `path`.
    compress : bool
        Only relevant for bincode; ignored for JSON. If True, use zstd.
    strip_points : bool
        Drop raw_points before saving (recommended; files get much smaller).
    strip_axes : bool
        Drop frames_axis/scans_axis/mz_axis before saving.
    """
    # Pull the underlying PyO3 objects
    py_list = [c.get_py_ptr() for c in clusters]

    # Infer format unless given
    f = (fmt or "").lower()
    ext = path.lower()
    if not f:
        if ext.endswith(".json"):
            f = "json"
        elif ext.endswith(".zst") or ext.endswith(".bin.zst"):
            f = "bin"
            compress = True
        elif ext.endswith(".bin"):
            f = "bin"
        else:
            # default to compressed bin if unknown
            f = "bin"
            if not ext.endswith(".bin") and not ext.endswith(".zst"):
                path = path + ".bin.zst"
                compress = True

    if f == "json":
        ims.save_clusters_json(path, py_list, strip_points, strip_axes, False)
    elif f == "bin":
        ims.save_clusters_bin(path, py_list, compress, strip_points, strip_axes)
    else:
        raise ValueError(f"Unknown fmt={fmt!r}. Use 'json' or 'bin'.")


def load_clusters(path: str) -> list["ClusterResult"]:
    """
    Load clusters from disk. Format inferred from extension.
    Returns a list of ClusterResult wrappers.
    """
    ext = path.lower()
    if ext.endswith(".json"):
        out_py = ims.load_clusters_json(path)
    elif ext.endswith(".zst") or ext.endswith(".bin.zst") or ext.endswith(".bin"):
        out_py = ims.load_clusters_bin(path)
    else:
        # Try bincode+zstd first, then JSON path if user omitted extension
        try:
            out_py = ims.load_clusters_bin(path)
        except Exception:
            out_py = ims.load_clusters_json(path)

    return [ClusterResult.from_py_ptr(p) for p in out_py]


def clusters_to_dataframe(results, rt_index=None):
    """
    Convert a list of ClusterResult into a tidy DataFrame.

    If `rt_index` is provided (your wrapped RtIndex), we compute an
    approximate `rt_time_mu` by mapping the local-μ frame within each
    cluster to the global frame_time via `frame_ids_used`.

    Columns (selection):
      - windows: rt_left/right, im_left/right, mz_min/max
      - rt_* / im_* / mz_*: mu, sigma, height, baseline, area, n
      - intensities: raw_sum, fit_volume (with log1p_* convenience columns)
      - mz_center_hint, mz_ppm_error
      - axes/meta: frames, scans, has_frames_axis, has_scans_axis, has_mz_axis
      - raw points: has_points, n_points
      - (optional) rt_time_mu
    """
    import numpy as np
    import pandas as pd

    rows = []

    # Optional: frame_id -> frame_time lookup from RtIndex
    frame_time_lookup = None
    if rt_index is not None:
        fid = np.asarray(rt_index.frame_ids, dtype=np.uint32)
        fti = np.asarray(rt_index.frame_times, dtype=np.float32)
        frame_time_lookup = {int(fid[i]): float(fti[i]) for i in range(len(fid))}

    for cid, cr in enumerate(results):
        # windows
        rt_left, rt_right = cr.rt_window
        im_left, im_right = cr.im_window
        mz_min, mz_max = cr.mz_window_da

        # fits
        rt = cr.rt_fit
        im = cr.im_fit
        mz = cr.mz_fit

        # axes presence / sizes
        fa = cr.frames_axis
        sa = cr.scans_axis
        ma = cr.mz_axis

        has_frames_axis = fa is not None
        has_scans_axis = sa is not None
        has_mz_axis = ma is not None

        # robust frame/scan counts:
        # - prefer attached axes if present, else fall back to window length
        frames = int(len(cr.frame_ids_used))  # always present and aligned to local frames
        scans = int(len(sa)) if has_scans_axis else int(max(im_right - im_left + 1, 0))

        # raw points summary
        rp = cr.raw_points
        has_points = rp is not None
        n_points = int(len(rp)) if has_points else 0

        # ppm error against hint (guard divide)
        mz_center_hint = float(cr.mz_center_hint)
        if np.isfinite(mz_center_hint) and mz_center_hint > 0.0:
            mz_ppm_error = (float(mz.mu) - mz_center_hint) / mz_center_hint * 1e6
        else:
            mz_ppm_error = np.nan

        # optional absolute RT time at μ (use frame_ids_used to map local μ)
        rt_time_mu = np.nan
        if frame_time_lookup is not None and frames > 0 and np.isfinite(rt.mu):
            mu_local = int(np.clip(round(rt.mu), 0, max(frames - 1, 0)))
            frame_ids = np.asarray(cr.frame_ids_used, dtype=np.uint32)
            if 0 <= mu_local < len(frame_ids):
                fid_mu = int(frame_ids[mu_local])
                rt_time_mu = frame_time_lookup.get(fid_mu, np.nan)

        rows.append({
            # IDs / provenance
            "cluster_id": cid,
            "rt_peak_id": cr.rt_peak_id,
            "im_peak_id": cr.im_peak_id,

            # windows
            "rt_left": rt_left, "rt_right": rt_right,
            "im_left": im_left, "im_right": im_right,
            "mz_min": mz_min, "mz_max": mz_max,

            # rt fit (frames)
            "rt_mu_frame": float(rt.mu),
            "rt_sigma_frames": float(rt.sigma),
            "rt_height": float(rt.height),
            "rt_baseline": float(rt.baseline),
            "rt_area": float(rt.area),
            "rt_n": int(rt.n),

            # im fit (scans)
            "im_mu_scan": float(im.mu),
            "im_sigma_scans": float(im.sigma),
            "im_height": float(im.height),
            "im_baseline": float(im.baseline),
            "im_area": float(im.area),
            "im_n": int(im.n),

            # mz fit (Da)
            "mz_mu_da": float(mz.mu),
            "mz_sigma_da": float(mz.sigma),
            "mz_height": float(mz.height),
            "mz_baseline": float(mz.baseline),
            "mz_area": float(mz.area),
            "mz_n": int(mz.n),

            # intensity summaries
            "raw_sum": float(cr.raw_sum),
            "fit_volume": float(cr.fit_volume),

            # hints / deltas
            "mz_center_hint": mz_center_hint,
            "mz_ppm_error": float(mz_ppm_error),

            # axes/meta
            "frames": frames,
            "scans": scans,
            "has_frames_axis": bool(has_frames_axis),
            "has_scans_axis": bool(has_scans_axis),
            "has_mz_axis": bool(has_mz_axis),

            # raw points
            "has_points": bool(has_points),
            "n_points": n_points,

            # optional absolute RT time
            "rt_time_mu": float(rt_time_mu),
        })

    df = pd.DataFrame(rows)

    # handy derived logs for plotting
    for col in ("raw_sum", "fit_volume", "rt_area", "im_area", "mz_area"):
        if col in df.columns:
            df[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))

    return df

from collections.abc import Iterable
from typing import List, Sequence, Union

def _is_impeak1d(x) -> bool:
    from .dia import ImPeak1D  # your wrapper class
    return isinstance(x, ImPeak1D)

def _is_nested(seq) -> bool:
    """Detect windows×rows×peaks shape: list of lists whose leaves are ImPeak1D."""
    if not isinstance(seq, Iterable) or isinstance(seq, (str, bytes)):
        return False
    for a in seq:
        if a:  # first non-empty
            return isinstance(a, Iterable) and not _is_impeak1d(a)
    return False


def stitch_im_peaks(
    peaks: Union[
        Sequence['ImPeak1D'],
        Sequence[Sequence[Sequence['ImPeak1D']]]  # windows × rows × peaks
    ],
    min_overlap_frames: int = 1,
    max_scan_delta: int = 1,
    jaccard_min: float = 0.0,
    pivot_log_intensity: float = 8.0,
    alpha_relax: float = 0.5,
    relax_max: float = 3.0,
    k_scan: float = 1.0,
    k_overlap: float = 1.0,
    k_jaccard: float = 0.1,
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

    if _is_nested(peaks):
        # windows × rows × peaks  ->  call the *streaming* Rust function
        batched_ptrs = [
            [[p.get_py_ptr() for p in row] for row in win]
            for win in peaks
        ]
        stitched_py = ims.stitch_im_peaks_batched_streaming(
            batched_ptrs,
            int(min_overlap_frames),
            int(max_scan_delta),
            float(jaccard_min),
            float(pivot_log_intensity),
            float(alpha_relax),
            float(relax_max),
            float(k_scan),
            float(k_overlap),
            float(k_jaccard),
            int(max_mz_row_delta),
            bool(allow_cross_groups),
        )
        return [ImPeak1D.from_py_ptr(p) for p in stitched_py]

    # Otherwise: flat list (kept with the non-streaming flat Rust path; typically small)
    flat = [p for p in peaks if _is_impeak1d(p)]
    if not flat:
        return []
    stitched_py = ims.stitch_im_peaks_across_windows(
        [p.get_py_ptr() for p in flat],
        int(min_overlap_frames),
        int(max_scan_delta),
        float(jaccard_min),
    )
    return [ImPeak1D.from_py_ptr(p) for p in stitched_py]