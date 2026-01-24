from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:
    from tqdm import tqdm as _tqdm
    TQDM = True
except Exception:
    TQDM = False

from imspy.timstof.dia import stitch_im_peaks

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
    batch_size: int = 256  # how many windows per pick_im_peaks_batched call


@dataclass(frozen=True)
class StitchParams:
    min_overlap_frames: int = 1
    max_scan_delta: int = 5
    jaccard_min: float = 0.2
    max_mz_row_delta: int = 1


@dataclass(frozen=True)
class ClusterParams:
    ppm_per_bin: float = 25.0
    smooth_sigma: float = 1.5
    attach_points: bool = False
    include_raw_stats: bool = False   # only matters if you later export DataFrame


# ---------------------------- Utilities ----------------------------

def _iter_batches(n: int, batch_size: int) -> Iterable[Sequence[int]]:
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        yield range(i, j)
        i = j


def _maybe_tqdm(iterable, total: Optional[int], desc: str):
    if TQDM:
        return _tqdm(iterable, total=total, desc=desc)
    return iterable


def _unique_window_groups(ds) -> List[int]:
    """
    Extract sorted unique DIA window groups from the dataset.
    Assumes ds.dia_ms_ms_info.WindowGroup is array-like / pandas Series–like.
    """
    wg = getattr(ds, "dia_ms_ms_info", None)
    if wg is None or not hasattr(wg, "WindowGroup"):
        raise ValueError("Dataset has no dia_ms_ms_info.WindowGroup")
    vals = np.asarray(wg.WindowGroup)
    return sorted(int(x) for x in np.unique(vals))


# ---------------------------- Planning ----------------------------

def plan_windows(ds, planning: PlanningParams):
    """Plan windows over the full DIA grid."""
    return ds.plan_mz_scan_windows(**planning.resolved())


def plan_windows_for_group(ds, window_group: int, planning: PlanningParams):
    """Plan windows for a specific DIA window group."""
    return ds.plan_mz_scan_windows_for_group(window_group=window_group, **planning.resolved())


# ---------------------------- Peak picking ----------------------------

def pick_im_peaks_for_plan(plan, picking: PeakPickingParams, desc: str = "pick peaks"):
    """Pick IM peaks for one plan (precursor or a fragment group)."""
    num = int(getattr(plan, "num_windows"))
    peaks: list = []
    batches = _iter_batches(num, picking.batch_size)
    total = (num + picking.batch_size - 1) // picking.batch_size
    for idx_batch in _maybe_tqdm(batches, total=total, desc=desc):
        rows = plan.pick_im_peaks_batched(
            list(idx_batch),
            min_prom=picking.min_prom,
            min_distance_scans=picking.min_distance_scans,
            min_width_scans=picking.min_width_scans,
        )
        peaks.extend(rows)
    return peaks


def stitch_peaks(peaks: list, stitch: StitchParams):
    """Run your stitching in a single place."""
    return stitch_im_peaks(
        peaks,
        min_overlap_frames=stitch.min_overlap_frames,
        max_scan_delta=stitch.max_scan_delta,
        jaccard_min=stitch.jaccard_min,
        max_mz_row_delta=stitch.max_mz_row_delta,
    )


# ---------------------------- Cluster builders ----------------------------

def build_precursor_clusters(
    ds,
    planning: PlanningParams = PlanningParams(),
    picking: PeakPickingParams = PeakPickingParams(),
    stitch: StitchParams = StitchParams(),
    cluster: ClusterParams = ClusterParams(),
    *,
    return_intermediates: bool = False,
):
    """
    Full pipeline: plan -> pick -> stitch -> cluster (precursors).
    """
    plan = plan_windows(ds, planning)
    peaks = pick_im_peaks_for_plan(plan, picking, desc="precursor peaks")
    stitched = stitch_peaks(peaks, stitch)
    clusters = ds.clusters_for_precursor(
        stitched,
        ppm_per_bin=cluster.ppm_per_bin,
        smooth_sigma=cluster.smooth_sigma,
        attach_points=cluster.attach_points,
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
    """
    One fragment group: plan -> pick -> stitch.
    Returns (stitched_peaks, plan) for that group.
    """
    plan_g = plan_windows_for_group(ds, window_group, planning)
    peaks_g = pick_im_peaks_for_plan(plan_g, picking, desc=f"frag peaks wg={window_group}")
    stitched_g = stitch_peaks(peaks_g, stitch)
    return stitched_g, plan_g


def build_fragment_clusters(
    ds,
    planning: PlanningParams = PlanningParams(rt_window_sec=6.0),   # your fragment default
    picking: PeakPickingParams = PeakPickingParams(min_prom=50, min_width_scans=10, batch_size=128),
    stitch: StitchParams = StitchParams(),
    cluster: ClusterParams = ClusterParams(),
    *,
    return_intermediates: bool = False,
):
    """
    Full pipeline for *all* fragment window groups:
        per group: plan -> pick -> stitch
        combine stitched peaks
        -> cluster (use your ds.clusters_for_precursor unless you have a dedicated fragment variant)
    """
    all_groups = _unique_window_groups(ds)
    stitched_all: list = []
    plans = {}

    for wg in all_groups:
        stitched_g, plan_g = build_fragment_clusters_per_group(ds, wg, planning, picking, stitch)
        stitched_all.extend(stitched_g)
        plans[wg] = plan_g

    # If you have a dedicated clustering entry point for fragments, replace below.
    clusters = ds.clusters_for_precursor(
        stitched_all,
        ppm_per_bin=cluster.ppm_per_bin,
        smooth_sigma=cluster.smooth_sigma,
        attach_points=cluster.attach_points,
    )
    if return_intermediates:
        return dict(plans=plans, stitched=stitched_all, clusters=clusters)
    return clusters


# ---------------------------- DF convenience ----------------------------

def clusters_to_df(
    clusters,
    include_raw_stats: bool = False,
    as_bool_flags: bool = True,
    column_order: Optional[list[str]] = None,
):
    from imspy.timstof.dia import clusters_to_dataframe

    return clusters_to_dataframe(
        clusters,
        include_raw_stats=include_raw_stats,
        as_bool_flags=as_bool_flags,
        column_order=column_order,
    )