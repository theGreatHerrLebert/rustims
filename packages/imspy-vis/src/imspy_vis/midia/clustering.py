"""Density clustering of precursor (3D) and MIDIA fragment (4D) point clouds.

Ported from ``proteolizardalgo.clustering`` / ``proteolizardmidia.clustering`` onto the
imspy-core data layer. Each function takes a plain coordinate DataFrame (as produced by
:mod:`imspy_vis.midia.data`) and returns the same points with ``label`` (and, for HDBSCAN,
``probability``) columns appended.

The axes are scaled before clustering so a density clusterer sees comparable distances in
retention time, ion mobility and m/z: cycle/scan are divided by ``2**scaling`` and m/z is
mapped through :func:`peak_width_preserving_mz_transform`. The MIDIA variant adds a fourth
axis, the precursor isolation m/z recovered from the quadrupole step via the extraction
window (the "MIDIA dimension").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN

from .transforms import peak_width_preserving_mz_transform

if TYPE_CHECKING:
    from pandas.core.groupby import DataFrameGroupBy
    from .data import MidiaExperiment


def cluster_precursors_dbscan(points: pd.DataFrame,
                              epsilon: float = 1.7,
                              min_samples: int = 7,
                              metric: str = "euclidean",
                              cycle_scaling: float = -0.4,
                              scan_scaling: float = 0.4,
                              resolution: int = 50_000) -> pd.DataFrame:
    """DBSCAN over precursor points (``cycle, scan, mz, intensity``)."""
    rt = points.cycle.to_numpy() / np.power(2, cycle_scaling)
    dt = points.scan.to_numpy() / np.power(2, scan_scaling)
    mz = peak_width_preserving_mz_transform(points.mz.to_numpy(), resolution=resolution)

    labels = DBSCAN(eps=epsilon, min_samples=min_samples, n_jobs=-1,
                    metric=metric).fit(np.vstack([rt, dt, mz]).T).labels_

    out = points[["cycle", "scan", "mz", "intensity"]].copy()
    out["label"] = labels
    return out


def cluster_precursors_hdbscan(points: pd.DataFrame,
                               algorithm: str = "best",
                               alpha: float = 1.0,
                               approx_min_span_tree: bool = True,
                               gen_min_span_tree: bool = True,
                               leaf_size: int = 40,
                               min_cluster_size: int = 7,
                               min_samples: int = 7,
                               p=None,
                               metric: str = "euclidean",
                               cycle_scaling: float = -0.4,
                               scan_scaling: float = 0.4,
                               resolution: int = 50_000,
                               mz_scaling: float = 0.0) -> pd.DataFrame:
    """HDBSCAN over precursor points (``cycle, scan, mz, intensity``)."""
    rt = points.cycle.to_numpy() / np.power(2, cycle_scaling)
    dt = points.scan.to_numpy() / np.power(2, scan_scaling)
    mz = peak_width_preserving_mz_transform(points.mz.to_numpy(), resolution=resolution) / np.power(2, mz_scaling)

    clusters = HDBSCAN(algorithm=algorithm, alpha=alpha,
                       approx_min_span_tree=approx_min_span_tree,
                       gen_min_span_tree=gen_min_span_tree,
                       leaf_size=leaf_size, metric=metric,
                       min_cluster_size=min_cluster_size,
                       min_samples=min_samples, p=p).fit(np.vstack([rt, dt, mz]).T)

    out = points[["cycle", "scan", "mz", "intensity"]].copy()
    out["label"] = clusters.labels_
    out["probability"] = clusters.probabilities_
    return out


def cluster_midia_hdbscan(points: pd.DataFrame,
                          experiment,
                          algorithm: str = "best",
                          cluster_selection_method: str = "eom",
                          cluster_selection_epsilon: float = 1.0,
                          alpha: float = 1.0,
                          approx_min_span_tree: bool = True,
                          gen_min_span_tree: bool = True,
                          leaf_size: int = 40,
                          min_cluster_size: int = 7,
                          min_samples: int = 2,
                          p=None,
                          metric: str = "manhattan",
                          cycle_scaling: float = 0.2,
                          scan_scaling: float = 0.6,
                          resolution: int = 46_900,
                          mz_scaling: float = 0.0,
                          extraction_scaling: float = 36.0,
                          use_midia_dimension: bool = True) -> pd.DataFrame:
    """HDBSCAN over MIDIA fragment points (``cycle, step, scan, mz, intensity``).

    The "MIDIA dimension" ``mc`` is the precursor isolation m/z recovered from each point's
    quadrupole ``step`` and ``scan`` via the run's extraction windows, scaled by
    ``extraction_scaling``. With ``use_midia_dimension`` the clustering is 4D
    ``(rt, mc, dt, mz)``; otherwise it falls back to 3D ``(rt, dt, mz)``.
    """
    cycle = points.cycle.to_numpy()
    step = points.step.to_numpy()
    scan = points.scan.to_numpy()
    mz_raw = points.mz.to_numpy()
    intensity = points.intensity.to_numpy()
    mc = _midia_dimension(experiment, step, scan)  # NaN where (step, scan) hits no window

    if use_midia_dimension:
        # Keep only fragments that fall inside an isolation window. Window-less points have
        # no meaningful MIDIA coordinate; folding them in (as 0) collapses them onto a fake
        # plane and lets HDBSCAN form a spurious cluster.
        valid = np.isfinite(mc)
        cycle, step, scan, mz_raw, intensity, mc = (
            arr[valid] for arr in (cycle, step, scan, mz_raw, intensity, mc)
        )
    else:
        mc = np.nan_to_num(mc, nan=0.0)  # unused in 3D mode; keep the returned column finite

    if len(cycle) == 0:
        # No fragments survived the window filter; HDBSCAN.fit would raise on 0 samples.
        return pd.DataFrame(
            columns=["cycle", "step", "mc_dim", "scan", "mz", "intensity", "label", "probability"]
        )

    rt = cycle / np.power(2, cycle_scaling)
    mc_scaled = mc / extraction_scaling
    dt = scan / np.power(2, scan_scaling)
    mz = peak_width_preserving_mz_transform(mz_raw, resolution=resolution) / np.power(2, mz_scaling)

    clusters = HDBSCAN(algorithm=algorithm, alpha=alpha,
                       cluster_selection_method=cluster_selection_method,
                       cluster_selection_epsilon=cluster_selection_epsilon,
                       approx_min_span_tree=approx_min_span_tree,
                       gen_min_span_tree=gen_min_span_tree,
                       leaf_size=leaf_size, metric=metric,
                       min_cluster_size=min_cluster_size,
                       min_samples=min_samples, p=p)

    feats = np.vstack([rt, mc_scaled, dt, mz]).T if use_midia_dimension else np.vstack([rt, dt, mz]).T
    clusters.fit(feats)

    return pd.DataFrame({
        "cycle": cycle, "step": step, "mc_dim": mc_scaled, "scan": scan,
        "mz": mz_raw, "intensity": intensity,
        "label": clusters.labels_, "probability": clusters.probabilities_,
    })


def _midia_dimension(experiment: "MidiaExperiment", step: np.ndarray, scan: np.ndarray) -> np.ndarray:
    """Per-point precursor isolation m/z from (step, scan); NaN where no window applies."""
    mc = np.full(step.shape, np.nan, dtype=np.float64)
    for s in np.unique(step):
        m = step == s
        mc[m] = experiment.isolation_left_bound(int(s), scan[m])
    return mc


def calculate_statistics(
    clusters: pd.DataFrame, noise: pd.DataFrame
) -> tuple[pd.DataFrame, DataFrameGroupBy]:
    """Summary table (points/intensity/clusters x total/cluster/noise/ratio) + per-label group."""
    sum_int_clusters = clusters.groupby(["label"])["intensity"].sum().sum()
    sum_int_noise = noise.groupby(["label"])["intensity"].sum().sum()
    intensity_ratio = np.round(sum_int_clusters / sum_int_noise, 3) if sum_int_noise else np.inf

    n_clusters_pts = clusters.shape[0]
    n_noise_pts = noise.shape[0]
    points_ratio = np.round(n_clusters_pts / n_noise_pts, 3) if n_noise_pts else np.inf
    # Labels are 0-based and contiguous, so the count is max(label) + 1.
    num_clusters = int(clusters.label.max()) + 1 if n_clusters_pts else 0

    summary_table = pd.DataFrame(
        {"Total": [n_clusters_pts + n_noise_pts, sum_int_clusters + sum_int_noise, num_clusters],
         "Cluster": [n_clusters_pts, sum_int_clusters, num_clusters],
         "Noise": [n_noise_pts, sum_int_noise, 0],
         "Ratio": [points_ratio, intensity_ratio, 0]},
        index=["Points", "Intensity", "Clusters"])
    return summary_table, clusters.groupby("label")
