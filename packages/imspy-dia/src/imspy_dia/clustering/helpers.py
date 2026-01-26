# imspy_pipelines.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Sequence, Dict, Any
import numpy as np
import pandas as pd

from imspy_dia.clustering.utility import stitch_im_peaks

try:
    from tqdm import tqdm as _tqdm
    TQDM = True
except Exception:
    TQDM = False

from imspy_core.timstof.dia import (
    TimsDatasetDIA,
)

# ---------------------------- Params ----------------------------

@dataclass(frozen=True)
class PlanningParams:
    ppm_per_bin: float = 25.0
    mz_pad_ppm: Optional[float] = None  # default -> 2 * ppm_per_bin
    rt_window_sec: float = 3.0
    rt_hop_sec: Optional[float] = None  # default -> rt_window_sec / 2
    num_threads: int = 128
    im_sigma_scans: float = 3.0
    truncate: float = 3.0
    precompute_views: bool = True
    def resolved(self) -> dict:
        mz_pad = self.mz_pad_ppm if self.mz_pad_ppm is not None else 2.0 * self.ppm_per_bin
        rt_hop = self.rt_hop_sec if self.rt_hop_sec is not None else (self.rt_window_sec / 2.0)
        return dict(
            ppm_per_bin=self.ppm_per_bin,
            mz_pad_ppm=mz_pad,
            rt_window_sec=self.rt_window_sec,
            rt_hop_sec=rt_hop,
            num_threads=self.num_threads,
            im_sigma_scans=self.im_sigma_scans,
            truncate=self.truncate,
            precompute_views=self.precompute_views,
        )

@dataclass(frozen=True)
class PeakPickingParams:
    min_prom: float = 100.0
    min_distance_scans: int = 2
    min_width_scans: int = 7
    batch_size: int = 256

@dataclass(frozen=True)
class StitchParams:
    min_overlap_frames: int = 1
    max_scan_delta: int = 5
    jaccard_min: float = 0.2
    max_mz_row_delta: int = 1

@dataclass(frozen=True)
class ClusterParams:
    ppm_per_bin: float = 25.0
    # seconds-based knobs (align with Rust API)
    smooth_sigma_sec: float = 1.25
    smooth_trunc_k: float = 3.0
    min_sep_sec: float = 2.0
    min_width_sec: float = 2.0
    # extras
    attach_points: bool = False
    include_raw_stats: bool = False   # DF export only
    # fallback for RT fallback peak logic
    fallback_if_frames_lt: int = 5
    fallback_frac_width: float = 0.50

# ---------------------------- Utilities ----------------------------

def _iter_batches(n: int, batch_size: int) -> Iterable[Sequence[int]]:
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        yield range(i, j)
        i = j

def _maybe_tqdm(iterable, total: Optional[int], desc: str, ncols: int = 80):
    if TQDM:
        return _tqdm(iterable, total=total, desc=desc, ncols=ncols)
    return iterable

def _unique_window_groups(ds) -> List[int]:
    if hasattr(ds, "dia_window_groups"):
        groups = ds.dia_window_groups()
        return sorted(int(x) for x in groups)
    wg = getattr(ds, "dia_ms_ms_info", None)
    if wg is None or not hasattr(wg, "WindowGroup"):
        raise ValueError("Dataset has no dia_ms_ms_info.WindowGroup")
    vals = np.asarray(wg.WindowGroup)
    return sorted(int(x) for x in np.unique(vals))

# ---------------------------- Planning ----------------------------

def plan_windows(ds, planning: PlanningParams):
    return ds.plan_mz_scan_windows(**planning.resolved())

def plan_windows_for_group(ds, window_group: int, planning: PlanningParams):
    return ds.plan_mz_scan_windows_for_group(window_group=window_group, **planning.resolved())

# ---------------------------- Peak picking ----------------------------

def _pick_im_peaks_for_plan(plan, picking: PeakPickingParams, desc: str = "pick peaks"):
    num = int(getattr(plan, "num_windows", len(plan)))
    def _batches(n, bs):
        i = 0
        while i < n:
            j = min(i + bs, n)
            yield list(range(i, j))
            i = j
    out_windows = []
    iterator = _maybe_tqdm(
        _batches(num, picking.batch_size),
        total=(num + picking.batch_size - 1) // picking.batch_size,
        desc=desc,
        ncols=80,
    )
    has_for_indices = hasattr(plan, "pick_im_peaks_for_indices")
    has_batched     = hasattr(plan, "pick_im_peaks_batched")
    for idx_batch in iterator:
        if has_for_indices:
            win_rows = plan.pick_im_peaks_for_indices(
                idx_batch,
                min_prom=picking.min_prom,
                min_distance_scans=picking.min_distance_scans,
                min_width_scans=picking.min_width_scans,
            )
            out_windows.extend(win_rows)
        elif has_batched:
            win_rows = plan.pick_im_peaks_batched(
                idx_batch,
                min_prom=picking.min_prom,
                min_distance_scans=picking.min_distance_scans,
                min_width_scans=picking.min_width_scans,
            )
            out_windows.extend(win_rows)
        else:
            for i in idx_batch:
                grid = plan[i]
                rows = grid.pick_im_peaks(
                    min_prom=picking.min_prom,
                    min_distance_scans=picking.min_distance_scans,
                    min_width_scans=picking.min_width_scans,
                    use_mobility=False,
                )
                out_windows.append(rows)
    return out_windows

pick_im_peaks_for_plan = _pick_im_peaks_for_plan

def stitch_peaks(peaks: list, stitch: StitchParams):
    return stitch_im_peaks(
        peaks,
        min_overlap_frames=stitch.min_overlap_frames,
        max_scan_delta=stitch.max_scan_delta,
        jaccard_min=stitch.jaccard_min,
        max_tof_row_delta=stitch.max_mz_row_delta,
    )

# ---------------------------- Pipelines ----------------------------

def build_precursor_clusters(
    ds,
    planning: PlanningParams = PlanningParams(),
    picking: PeakPickingParams = PeakPickingParams(),
    stitch: StitchParams = StitchParams(),
    cluster: ClusterParams = ClusterParams(),
    *,
    return_intermediates: bool = False,
):
    plan = plan_windows(ds, planning)
    peaks = _pick_im_peaks_for_plan(plan, picking, desc="IM peaks (MS1)")
    stitched = stitch_peaks(peaks, stitch)
    clusters = ds.clusters_for_precursor(
        stitched,
        ppm_per_bin=cluster.ppm_per_bin,
        # RtExpandParams (seconds)
        bin_pad=0,
        smooth_sigma_sec=cluster.smooth_sigma_sec,
        smooth_trunc_k=cluster.smooth_trunc_k,
        min_prom=picking.min_prom,
        min_sep_sec=cluster.min_sep_sec,
        min_width_sec=cluster.min_width_sec,
        fallback_if_frames_lt=cluster.fallback_if_frames_lt,
        fallback_frac_width=cluster.fallback_frac_width,
        # BuildSpecOpts / Eval1DOpts
        extra_rt_pad=0,
        extra_im_pad=0,
        mz_ppm_pad=5.0,
        mz_hist_bins=64,
        refine_mz_once=True,
        refine_k_sigma=3.0,
        attach_axes=True,
        attach_points=cluster.attach_points,
        attach_max_points=None,
        # pairing + threads
        require_rt_overlap=True,
        num_threads=planning.num_threads,
        min_im_span=12,
    )
    if return_intermediates:
        return dict(plan=plan, peaks=peaks, stitched=stitched, clusters=clusters)
    return clusters

def build_fragment_clusters_per_group(
    ds,
    window_group: int,
    planning: PlanningParams,
    picking: PeakPickingParams,
    stitch: StitchParams,
):
    plan_g = plan_windows_for_group(ds, window_group, planning)
    peaks_g = _pick_im_peaks_for_plan(plan_g, picking, desc=f"IM peaks (wg={window_group})")
    stitched_g = stitch_peaks(peaks_g, stitch)
    return stitched_g, plan_g

def build_fragment_clusters(
    ds,
    planning: PlanningParams = PlanningParams(rt_window_sec=6.0),
    picking: PeakPickingParams = PeakPickingParams(min_prom=50, min_width_scans=10, batch_size=128),
    stitch: StitchParams = StitchParams(),
    cluster: ClusterParams = ClusterParams(),
    *,
    return_intermediates: bool = False,
):
    groups = _unique_window_groups(ds)
    all_clusters = []
    intermediates = {"plans": {}, "peaks": {}, "stitched": {}, "clusters_by_group": {}}
    for wg in groups:  # ← run all groups
        plan_g = plan_windows_for_group(ds, wg, planning)
        intermediates["plans"][wg] = plan_g
        peaks_g = _pick_im_peaks_for_plan(plan_g, picking, desc=f"IM peaks (wg={wg})")
        intermediates["peaks"][wg] = peaks_g
        stitched_g = stitch_peaks(peaks_g, stitch)
        intermediates["stitched"][wg] = stitched_g
        if not all(getattr(p, "window_group", None) == wg for p in stitched_g):
            raise ValueError(f"Stitched IM peaks contain peaks not in window_group={wg}")
        clusters_g = ds.clusters_for_group(
            wg,
            stitched_g,
            ppm_per_bin=cluster.ppm_per_bin,
            # RtExpandParams (seconds)
            bin_pad=0,
            smooth_sigma_sec=cluster.smooth_sigma_sec,
            smooth_trunc_k=cluster.smooth_trunc_k,
            min_prom=picking.min_prom,
            min_sep_sec=cluster.min_sep_sec,
            min_width_sec=cluster.min_width_sec,
            fallback_if_frames_lt=cluster.fallback_if_frames_lt,
            fallback_frac_width=cluster.fallback_frac_width,
            # BuildSpecOpts / Eval1DOpts
            extra_rt_pad=0,
            extra_im_pad=0,
            mz_ppm_pad=5.0,
            mz_hist_bins=64,
            refine_mz_once=True,
            refine_k_sigma=3.0,
            attach_axes=True,
            attach_points=cluster.attach_points,
            attach_max_points=None,
            # matching + threads
            require_rt_overlap=True,
            num_threads=planning.num_threads,
            min_im_span=12,
        )
        if not all(getattr(c, "ms_level", None) == 2 for c in clusters_g):
            raise AssertionError("Expected ms_level=2 for fragment clusters")
        if not all(getattr(c, "window_group", None) == wg for c in clusters_g):
            raise AssertionError("Fragment clusters missing window_group assignment")
        intermediates["clusters_by_group"][wg] = clusters_g
        all_clusters.extend(clusters_g)
    if return_intermediates:
        return {"clusters": all_clusters, **intermediates}
    return all_clusters

# ---------------------------- DF convenience ----------------------------

def clusters_to_df(
    clusters,
):
    return pd.DataFrame([c.to_dict() for c in clusters]).set_index("ms_level")

# --------- small helpers used by the CLI ----------

def to_plain_dict(dc_obj) -> Dict[str, Any]:
    return asdict(dc_obj)

def run_ms1(ds: TimsDatasetDIA,
            planning: PlanningParams,
            picking: PeakPickingParams,
            stitch: StitchParams,
            cluster: ClusterParams):
    cls = build_precursor_clusters(ds, planning, picking, stitch, cluster)
    df = clusters_to_df(cls, include_raw_stats=cluster.include_raw_stats)
    return cls, df

def run_ms2(ds: TimsDatasetDIA,
            planning: PlanningParams,
            picking: PeakPickingParams,
            stitch: StitchParams,
            cluster: ClusterParams):
    cls = build_fragment_clusters(ds, planning, picking, stitch, cluster)
    df = clusters_to_df(cls, include_raw_stats=cluster.include_raw_stats)
    return cls, df

from typing import Optional

def get_slice_filtered_cluster(
    r,
    ds,
    *,
    is_precursor: bool = True,
    mz_pad: float = 0.0,
) -> Optional[np.ndarray]:
    """
    Build a dense RT×IM slice for the cluster's cuboid.

    Notes
    -----
    - Rust frame bounds (rt_lo/rt_hi) are inclusive indices into the
      *group-specific* frame list.
    - Python slicing is end-exclusive, so we slice [: rt_hi+1].
    - im_lo/im_hi are absolute scan indices (inclusive).
    """
    # 1) Resolve frame ids per space
    if is_precursor:
        frame_ids = getattr(ds, "precursor_frames", None)
        if frame_ids is None:
            raise AttributeError("Dataset missing 'precursor_frames'")
    else:
        if not hasattr(ds, "dia_ms_ms_info"):
            raise AttributeError("Dataset missing 'dia_ms_ms_info'")
        msms = ds.dia_ms_ms_info
        if "WindowGroup" not in msms.columns or "Frame" not in msms.columns:
            raise AttributeError("dia_ms_ms_info must have 'WindowGroup' and 'Frame' columns")
        frame_ids = msms.loc[msms.WindowGroup == r.window_group, "Frame"].values

    if frame_ids is None or len(frame_ids) == 0:
        return None

    # 2) Clamp & convert bounds (inclusive → exclusive)
    n = len(frame_ids)
    rt_lo = max(0, int(r.rt_lo))
    rt_hi = min(n - 1, int(r.rt_hi))
    if rt_lo > rt_hi:
        return None

    sel_fids = frame_ids[rt_lo: rt_hi + 1]  # inclusive hi

    # 3) Fetch slice and filter
    try:
        # Prefer your TIMS-slice API; fall back if needed
        if hasattr(ds, "get_tims_slice"):
            S = ds.get_tims_slice(sel_fids)
        elif hasattr(ds, "get_slice"):
            S = ds.get_slice(sel_fids)  # maybe pass num_threads=...
        else:
            raise AttributeError("Dataset missing get_tims_slice/get_slice API")

        S = S.filter(
            scan_min=int(r.im_lo),
            scan_max=int(r.im_hi),
            mz_min=float(r.mz_lo) - mz_pad,
            mz_max=float(r.mz_hi) + mz_pad,
        )

        # Vectorize → dense tensor (rt × im × mz)
        T = S.vectorized(4).get_tensor_repr(dense=True, re_index=True).numpy()
        if T.ndim != 3:
            # Unexpected layout; try to coerce
            return None

        return T

    except Exception as e:
        return None
