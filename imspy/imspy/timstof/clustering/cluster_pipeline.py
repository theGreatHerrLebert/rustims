#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IM peak detection → stitching → clustering (TOML-configured) with logging and CLI overrides.

Updated for:
- TimsDatasetDIA.plan_tof_scan_windows / plan_tof_scan_windows_for_group
- new torch extractor (pool_tof, tol_tof, Gaussian blur, topk_per_tile, patch batching)
- new clustering API (tof_step + RT/IM/TOF params, no ppm_per_bin)
- optional per-window-group fragment output
"""
from __future__ import annotations

import argparse
import gc
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from imspy.timstof.clustering.utility import stitch_im_peaks

# Python 3.11+ has tomllib; otherwise optional tomli fallback
try:
    import tomllib as toml
except Exception:
    print("Warning: tomllib not found, falling back to tomli")
    import tomli as toml  # type: ignore

from imspy.timstof.clustering.torch_extractor import iter_im_peaks_batches
from imspy.timstof.dia import (
    save_clusters_bin,
    TimsDatasetDIA,
)

# --------------------------- logging ------------------------------------------
_LOGGER_NAME = "t-tracer"
_logger = logging.getLogger(_LOGGER_NAME)


def setup_logging(
    log_file: str | os.PathLike | None,
    level: str = "INFO",
    also_console: bool = True,
    rotate_bytes: int = 50 * 1024 * 1024,  # 50 MB
    backup_count: int = 5,
) -> None:
    """Initialize rotating file logging + optional console."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(process)d | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file:
        log_path = Path(log_file)
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_path, maxBytes=rotate_bytes, backupCount=backup_count)
        fh.setFormatter(fmt)
        fh.setLevel(logger.level)
        logger.addHandler(fh)

    if also_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(logger.level)
        logger.addHandler(ch)

    # capture unhandled exceptions into log
    def _excepthook(exc_type, exc, tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook


def log(msg: str, level: int = logging.INFO) -> None:
    _logger.log(level, msg)


# ------------------------ config summary --------------------------------------
def print_config_summary(cfg: dict) -> None:
    ds_cfg = cfg.get("dataset", {})
    out_cfg = cfg.get("output", {})
    run_cfg = cfg.get("run", {})
    det_cfg = cfg.get("detector", {})
    st_cfg = cfg.get("stitch", {})
    plan_cfg = cfg.get("plans", {})
    clus_cfg = cfg.get("cluster", {})

    lines: list[str] = []
    lines.append("──────────────── CONFIG SUMMARY ────────────────")
    # Dataset
    lines.append("[dataset]")
    lines.append(f"  path                 : {ds_cfg.get('path')}")
    lines.append(f"  use_bruker_sdk       : {bool(ds_cfg.get('use_bruker_sdk', False))}")
    # Run
    lines.append("[run]")
    lines.append(f"  stage                : {run_cfg.get('stage')}")
    lines.append(f"  device               : {run_cfg.get('device')}")
    lines.append(f"  batch_size           : {int(run_cfg.get('batch_size', 0))}")
    lines.append(f"  precompute_views     : {bool(run_cfg.get('precompute_views', False))}")
    lines.append(f"  fragments_enabled    : {bool(run_cfg.get('fragments_enabled', False))}")
    lines.append(f"  attach_raw_data      : {bool(run_cfg.get('attach_raw_data', False))}")
    # Output
    lines.append("[output]")
    lines.append(f"  dir                  : {out_cfg.get('dir')}")
    if "precursor_file" in out_cfg:
        lines.append(f"  precursor_file       : {out_cfg.get('precursor_file')}")
    if "fragment_file" in out_cfg:
        lines.append(f"  fragment_file        : {out_cfg.get('fragment_file')}")
    pq_enabled = bool(out_cfg.get("parquet_enabled", False))
    lines.append(f"  parquet_enabled      : {pq_enabled}")
    if pq_enabled:
        if "precursor_parquet" in out_cfg:
            lines.append(f"  precursor_parquet    : {out_cfg.get('precursor_parquet')}")
        if "fragment_parquet" in out_cfg:
            lines.append(f"  fragment_parquet     : {out_cfg.get('fragment_parquet')}")
    lines.append(f"  compress_bin         : {bool(out_cfg.get('compress_bin', True))}")
    # fragment-per-WG options
    lines.append(f"  fragment_split_by_wg : {bool(out_cfg.get('fragment_split_by_wg', False))}")
    if "fragment_file_pattern" in out_cfg and out_cfg.get("fragment_file_pattern") is not None:
        lines.append(f"  fragment_file_pattern: {out_cfg.get('fragment_file_pattern')}")
    if "fragment_parquet_pattern" in out_cfg and out_cfg.get("fragment_parquet_pattern") is not None:
        lines.append(f"  fragment_parquet_pattern: {out_cfg.get('fragment_parquet_pattern')}")

    # Plans + Stitch
    for sec in ("precursor", "fragment"):
        if sec in plan_cfg:
            p = plan_cfg[sec]
            lines.append(f"[plans.{sec}]")
            lines.append(f"  tof_step             : {int(p.get('tof_step', 1))}")
            lines.append(f"  rt_window_sec        : {float(p.get('rt_window_sec', 0.0))}")
            lines.append(f"  rt_hop_sec           : {float(p.get('rt_hop_sec', 0.0))}")
            lines.append(f"  sigma_scans          : {p.get('sigma_scans')}")
            lines.append(f"  sigma_tof_bins       : {p.get('sigma_tof_bins')}")
            lines.append(f"  truncate             : {float(p.get('truncate', 3.0))}")
            lines.append(f"  num_threads          : {int(p.get('num_threads', 4))}")
        if sec in st_cfg:
            s = st_cfg[sec]
            lines.append(f"[stitch.{sec}]")
            lines.append(f"  max_tof_row_delta    : {int(s.get('max_tof_row_delta', 0))}")
            lines.append(f"  max_scan_delta       : {int(s.get('max_scan_delta', 0))}")
            lines.append(f"  min_overlap_frames   : {int(s.get('min_overlap_frames', 1))}")
            lines.append(f"  min_im_overlap_scans : {int(s.get('min_im_overlap_scans', 1))}")
            lines.append(f"  jaccard_min          : {float(s.get('jaccard_min', 0.0))}")
            lines.append(f"  im_jaccard_min       : {float(s.get('im_jaccard_min', 0.0))}")
            lines.append(f"  allow_cross_groups   : {bool(s.get('allow_cross_groups', False))}")
            lines.append(f"  require_mutual_apex  : {bool(s.get('require_mutual_apex_inside', True))}")
            lines.append(f"  use_batch_stitch     : {bool(s.get('use_batch_stitch', False))}")

    # Detector
    if det_cfg:
        lines.append("[detector]")
        lines.append(f"  pool_scan            : {int(det_cfg.get('pool_scan', 0))}")
        lines.append(f"  pool_tof             : {int(det_cfg.get('pool_tof', 0))}")
        lines.append(f"  min_intensity_scaled : {float(det_cfg.get('min_intensity_scaled', 0.0))}")
        lines.append(f"  tile_rows            : {int(det_cfg.get('tile_rows', 0))}")
        lines.append(f"  tile_overlap         : {int(det_cfg.get('tile_overlap', 0))}")
        lines.append(f"  fit_h / fit_w        : {int(det_cfg.get('fit_h', 0))} / {int(det_cfg.get('fit_w', 0))}")
        lines.append(f"  refine               : {det_cfg.get('refine')}")
        lines.append(f"  refine_iters         : {int(det_cfg.get('refine_iters', 0))}")
        lines.append(f"  refine_lr            : {float(det_cfg.get('refine_lr', 0.0))}")
        lines.append(f"  refine_mask_k        : {float(det_cfg.get('refine_mask_k', 0.0))}")
        lines.append(
            "  refine_scan/tof      : "
            f"{bool(det_cfg.get('refine_scan', True))} / {bool(det_cfg.get('refine_tof', True))}"
        )
        lines.append(
            "  refine_sigma_scan/tof: "
            f"{bool(det_cfg.get('refine_sigma_scan', True))} / "
            f"{bool(det_cfg.get('refine_sigma_tof', True))}"
        )
        lines.append(f"  scale                : {det_cfg.get('scale')}")
        lines.append(f"  output_units         : {det_cfg.get('output_units')}")
        lines.append(f"  gn_float64           : {bool(det_cfg.get('gn_float64', False))}")
        lines.append(f"  do_dedup             : {bool(det_cfg.get('do_dedup', True))}")
        lines.append(
            f"  tol_scan / tol_tof   : {float(det_cfg.get('tol_scan', 0.0))} / "
            f"{float(det_cfg.get('tol_tof', 0.0))}"
        )
        lines.append(f"  k_sigma              : {float(det_cfg.get('k_sigma', 0.0))}")
        lines.append(f"  min_width            : {int(det_cfg.get('min_width', 0))}")
        lines.append(f"  topk_per_tile        : {det_cfg.get('topk_per_tile')}")
        lines.append(f"  patch_batch_target_mb: {int(det_cfg.get('patch_batch_target_mb', 0))}")
        lines.append(f"  blur_sigma_scan      : {float(det_cfg.get('blur_sigma_scan', 0.0))}")
        lines.append(f"  blur_sigma_tof       : {float(det_cfg.get('blur_sigma_tof', 0.0))}")
        lines.append(f"  blur_truncate        : {float(det_cfg.get('blur_truncate', 3.0))}")

    # Cluster
    if "cluster" in cfg:
        for sec in ("precursor", "fragment"):
            if sec in clus_cfg:
                cc = clus_cfg[sec]
                lines.append(f"[cluster.{sec}]")
                lines.append(f"  tof_step             : {int(cc.get('tof_step', 1))}")
                lines.append(f"  bin_pad              : {float(cc.get('bin_pad', 0.0))}")
                lines.append(f"  smooth_sigma_sec     : {float(cc.get('smooth_sigma_sec', 0.0))}")
                lines.append(f"  smooth_trunc_k       : {float(cc.get('smooth_trunc_k', 0.0))}")
                lines.append(f"  min_prom             : {float(cc.get('min_prom', 0.0))}")
                lines.append(f"  min_sep_sec          : {float(cc.get('min_sep_sec', 0.0))}")
                lines.append(f"  min_width_sec        : {float(cc.get('min_width_sec', 0.0))}")
                lines.append(f"  fallback_if_frames_lt: {int(cc.get('fallback_if_frames_lt', 0))}")
                lines.append(f"  fallback_frac_width  : {float(cc.get('fallback_frac_width', 0.0))}")
                lines.append(f"  extra_rt_pad         : {int(cc.get('extra_rt_pad', 0))}")
                lines.append(f"  extra_im_pad         : {int(cc.get('extra_im_pad', 0))}")
                lines.append(f"  tof_bin_pad          : {int(cc.get('tof_bin_pad', 0))}")
                if sec == "precursor":
                    lines.append(f"  tof_hist_bins        : {int(cc.get('tof_hist_bins', 0))}")
                else:
                    lines.append(f"  tof_hist_pad         : {int(cc.get('tof_hist_pad', 0))}")
                lines.append(f"  refine_tof_once      : {bool(cc.get('refine_tof_once', True))}")
                lines.append(f"  refine_k_sigma       : {float(cc.get('refine_k_sigma', 0.0))}")
                lines.append(f"  attach_max_points    : {int(cc.get('attach_max_points', 0))}")
                lines.append(f"  require_rt_overlap   : {bool(cc.get('require_rt_overlap', True))}")
                lines.append(f"  compute_mz_from_tof  : {bool(cc.get('compute_mz_from_tof', True))}")
                lines.append(f"  num_threads          : {int(cc.get('num_threads', 0))}")
                lines.append(f"  min_im_span          : {int(cc.get('min_im_span', 0))}")

    # Light sanity notes
    lines.append("──────────────── NOTES ─────────────────────────")
    cuda_avail = torch.cuda.is_available()
    lines.append(f"  CUDA available       : {cuda_avail}")
    dev = run_cfg.get("device")
    if dev == "cpu" and cuda_avail:
        lines.append("  ⚠ You selected CPU but CUDA is available.")
    if dev == "cuda" and not cuda_avail:
        lines.append("  ⚠ CUDA requested but not available; this will fall back or fail.")

    # Output existence hints
    out_dir = out_cfg.get("dir")
    if out_dir and not Path(out_dir).exists():
        lines.append(f"  ℹ Output dir will be created: {out_dir}")

    # Dataset path check
    ds_path = ds_cfg.get("path")
    if ds_path and not Path(ds_path).exists():
        lines.append(f"  ⚠ Dataset path does not exist: {ds_path}")

    lines.append("───────────────────────────────────────────────")
    for ln in lines:
        log(ln)


# ---------- small helpers -----------------------------------------------------
def ensure_dir_for_file(path: str | os.PathLike) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def cuda_gc() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------- core runners ------------------------------------------------------
def build_plan(ds, plan_cfg: dict, precompute_views: bool, *, for_group: int | None = None):
    """
    Build a TofScanPlan (precursor) or TofScanPlanGroup (fragment WG) using the
    TimsDatasetDIA.plan_tof_scan_windows* methods.
    """
    tof_step = int(plan_cfg["tof_step"])
    rt_window_sec = float(plan_cfg["rt_window_sec"])
    rt_hop_sec = float(plan_cfg["rt_hop_sec"])
    num_threads = int(plan_cfg.get("num_threads", 4))

    sigma_scans = plan_cfg.get("sigma_scans", None)
    sigma_tof_bins = plan_cfg.get("sigma_tof_bins", None)
    truncate = float(plan_cfg.get("truncate", 3.0))

    if sigma_scans is not None:
        sigma_scans = float(sigma_scans)
    if sigma_tof_bins is not None:
        sigma_tof_bins = float(sigma_tof_bins)

    if for_group is None:
        return ds.plan_tof_scan_windows(
            tof_step=tof_step,
            rt_window_sec=rt_window_sec,
            rt_hop_sec=rt_hop_sec,
            num_threads=num_threads,
            maybe_sigma_scans=sigma_scans,
            maybe_sigma_tof_bins=sigma_tof_bins,
            truncate=truncate,
            precompute_views=bool(precompute_views),
        )
    else:
        return ds.plan_tof_scan_windows_for_group(
            window_group=int(for_group),
            tof_step=tof_step,
            rt_window_sec=rt_window_sec,
            rt_hop_sec=rt_hop_sec,
            num_threads=num_threads,
            maybe_sigma_scans=sigma_scans,
            maybe_sigma_tof_bins=sigma_tof_bins,
            truncate=truncate,
            precompute_views=bool(precompute_views),
        )


def detect_and_stitch_for_plan(
    plan,
    run_cfg: dict,
    det_cfg: dict,
    stitch_cfg: dict,
):
    """Return stitched peaks list for a plan with optional batch-stitch RAM saver."""
    use_batch_stitch = bool(stitch_cfg.get("use_batch_stitch", False))
    if use_batch_stitch:
        stitched_running: list = []
    else:
        collected: list = []

    num_batches = (len(plan) + int(run_cfg["batch_size"]) - 1) // int(run_cfg["batch_size"])
    log(f"[detect] plan has {len(plan)} tof-scan planes -> (~{num_batches} batches)")

    # Handle topk_per_tile semantics: 0 or negative => None
    topk_per_tile_cfg = det_cfg.get("topk_per_tile", None)
    if topk_per_tile_cfg is not None:
        topk_per_tile = int(topk_per_tile_cfg)
        if topk_per_tile <= 0:
            topk_per_tile = None
    else:
        topk_per_tile = None

    patch_batch_target_mb = int(det_cfg.get("patch_batch_target_mb", 128))

    blur_sigma_scan = float(det_cfg.get("blur_sigma_scan", 0.0))
    blur_sigma_tof = float(det_cfg.get("blur_sigma_tof", 0.0))
    blur_truncate = float(det_cfg.get("blur_truncate", 3.0))

    for peak_batch in iter_im_peaks_batches(
        plan,
        batch_size=int(run_cfg["batch_size"]),
        device=str(run_cfg["device"]),
        # detector / refinement
        pool_scan=int(det_cfg["pool_scan"]),
        pool_tof=int(det_cfg["pool_tof"]),
        min_intensity_scaled=float(det_cfg["min_intensity_scaled"]),
        tile_rows=int(det_cfg["tile_rows"]),
        tile_overlap=int(det_cfg["tile_overlap"]),
        fit_h=int(det_cfg["fit_h"]),
        fit_w=int(det_cfg["fit_w"]),
        refine=str(det_cfg["refine"]),
        refine_iters=int(det_cfg["refine_iters"]),
        refine_lr=float(det_cfg["refine_lr"]),
        refine_mask_k=float(det_cfg["refine_mask_k"]),
        refine_scan=bool(det_cfg["refine_scan"]),
        refine_tof=bool(det_cfg["refine_tof"]),
        refine_sigma_scan=bool(det_cfg["refine_sigma_scan"]),
        refine_sigma_tof=bool(det_cfg["refine_sigma_tof"]),
        scale=str(det_cfg["scale"]),
        output_units=str(det_cfg["output_units"]),
        gn_float64=bool(det_cfg["gn_float64"]),
        do_dedup=bool(det_cfg["do_dedup"]),
        tol_scan=float(det_cfg["tol_scan"]),
        tol_tof=float(det_cfg["tol_tof"]),
        k_sigma=float(det_cfg["k_sigma"]),
        min_width=int(det_cfg["min_width"]),
        # NEW: performance + blur knobs
        topk_per_tile=topk_per_tile,
        patch_batch_target_mb=patch_batch_target_mb,
        blur_sigma_scan=blur_sigma_scan,
        blur_sigma_tof=blur_sigma_tof,
        blur_truncate=blur_truncate,
    ):
        if use_batch_stitch:
            stitched_batch = stitch_im_peaks(
                peak_batch,
                min_overlap_frames=int(stitch_cfg.get("min_overlap_frames", 1)),
                max_scan_delta=int(stitch_cfg["max_scan_delta"]),
                jaccard_min=float(stitch_cfg.get("jaccard_min", 0.0)),
                max_tof_row_delta=int(stitch_cfg["max_tof_row_delta"]),
                allow_cross_groups=bool(stitch_cfg.get("allow_cross_groups", False)),
                min_im_overlap_scans=int(stitch_cfg.get("min_im_overlap_scans", 1)),
                im_jaccard_min=float(stitch_cfg.get("im_jaccard_min", 0.0)),
                require_mutual_apex_inside=bool(
                    stitch_cfg.get("require_mutual_apex_inside", True)
                ),
            )
            stitched_running = stitch_im_peaks(
                stitched_running + stitched_batch,
                min_overlap_frames=int(stitch_cfg.get("min_overlap_frames", 1)),
                max_scan_delta=int(stitch_cfg["max_scan_delta"]),
                jaccard_min=float(stitch_cfg.get("jaccard_min", 0.0)),
                max_tof_row_delta=int(stitch_cfg["max_tof_row_delta"]),
                allow_cross_groups=bool(stitch_cfg.get("allow_cross_groups", False)),
                min_im_overlap_scans=int(stitch_cfg.get("min_im_overlap_scans", 1)),
                im_jaccard_min=float(stitch_cfg.get("im_jaccard_min", 0.0)),
                require_mutual_apex_inside=bool(
                    stitch_cfg.get("require_mutual_apex_inside", True)
                ),
            )
            del peak_batch, stitched_batch
        else:
            collected.extend(peak_batch)

        cuda_gc()

    if use_batch_stitch:
        return stitched_running
    else:
        return stitch_im_peaks(
            collected,
            min_overlap_frames=int(stitch_cfg.get("min_overlap_frames", 1)),
            max_scan_delta=int(stitch_cfg["max_scan_delta"]),
            jaccard_min=float(stitch_cfg.get("jaccard_min", 0.0)),
            max_tof_row_delta=int(stitch_cfg["max_tof_row_delta"]),
            allow_cross_groups=bool(stitch_cfg.get("allow_cross_groups", False)),
            min_im_overlap_scans=int(stitch_cfg.get("min_im_overlap_scans", 1)),
            im_jaccard_min=float(stitch_cfg.get("im_jaccard_min", 0.0)),
            require_mutual_apex_inside=bool(
                stitch_cfg.get("require_mutual_apex_inside", True)
            ),
        )


def run_precursor(ds, cfg):

    log("[stage] precursor")
    precompute_views = bool(cfg["run"]["precompute_views"])
    plan = build_plan(ds, cfg["plans"]["precursor"], precompute_views, for_group=None)

    stitched = detect_and_stitch_for_plan(
        plan, cfg["run"], cfg["detector"], cfg["stitch"]["precursor"]
    )

    attach_raw = bool(cfg["run"].get("attach_raw_data", False))
    c = cfg["cluster"]["precursor"]

    clusters = ds.clusters_for_precursor(
        stitched,
        tof_step=int(c.get("tof_step", 1)),
        bin_pad=float(c.get("bin_pad", 10.0)),
        smooth_sigma_sec=float(c.get("smooth_sigma_sec", 1.25)),
        smooth_trunc_k=float(c.get("smooth_trunc_k", 3.0)),
        min_prom=float(c.get("min_prom", 50.0)),
        min_sep_sec=float(c.get("min_sep_sec", 2.0)),
        min_width_sec=float(c.get("min_width_sec", 2.0)),
        fallback_if_frames_lt=int(c.get("fallback_if_frames_lt", 5)),
        fallback_frac_width=float(c.get("fallback_frac_width", 0.50)),
        extra_rt_pad=int(c.get("extra_rt_pad", 0)),
        extra_im_pad=int(c.get("extra_im_pad", 0)),
        tof_bin_pad=int(c.get("tof_bin_pad", 1)),
        tof_hist_bins=int(c.get("tof_hist_bins", 64)),
        refine_tof_once=bool(c.get("refine_tof_once", True)),
        refine_k_sigma=float(c.get("refine_k_sigma", 3.0)),
        attach_axes=True,
        attach_points=attach_raw,
        attach_max_points=int(c.get("attach_max_points", 512)),
        require_rt_overlap=bool(c.get("require_rt_overlap", True)),
        compute_mz_from_tof=bool(c.get("compute_mz_from_tof", True)),
        num_threads=int(c.get("num_threads", 0)),
        min_im_span=int(c.get("min_im_span", 10)),
        attach_im_xic=True,
        attach_rt_xic=True,
        pad_tof_bins=0,
        pad_im_scans=0,
        pad_rt_frames=1,
    )

    # ---- precursor output dirs: <root>/precursor/ ----
    out_root = Path(cfg["output"]["dir"])
    prec_dir = out_root / "precursor"

    # ---- binary save ----
    out_bin = prec_dir / cfg["output"]["precursor_file"]
    ensure_dir_for_file(out_bin)
    compress = bool(cfg["output"].get("compress_bin", True))
    save_clusters_bin(path=str(out_bin), clusters=clusters, compress=compress)
    log(f"[ok] wrote precursor clusters -> {out_bin} (compress={compress})")

    # ---- optional parquet ----
    if bool(cfg["output"].get("parquet_enabled", False)):
        out_parq = prec_dir / cfg["output"]["precursor_parquet"]
        ensure_dir_for_file(out_parq)
        df = pd.DataFrame([c.to_dict() for c in clusters])
        df.to_parquet(out_parq, index=False)
        log(f"[ok] wrote precursor parquet -> {out_parq}")

    del stitched, clusters
    cuda_gc()


def run_fragments(ds, cfg):

    log("[stage] fragments")

    if "fragment" not in cfg.get("plans", {}) or "fragment" not in cfg.get("stitch", {}):
        log("[skip] fragments: no [plans.fragment] or [stitch.fragment] in config")
        return

    precompute_views = bool(cfg["run"]["precompute_views"])
    plan_cfg = cfg["plans"]["fragment"]
    stitch_cfg = cfg["stitch"]["fragment"]
    c = cfg["cluster"]["fragment"]

    out_cfg = cfg["output"]
    base_dir = Path(out_cfg["dir"])
    compress = bool(out_cfg.get("compress_bin", True))
    parquet_enabled = bool(out_cfg.get("parquet_enabled", False))

    fragment_split_by_wg = bool(out_cfg.get("fragment_split_by_wg", False))
    frag_file = out_cfg.get("fragment_file", "fragment_clusters.binz")
    frag_parq = out_cfg.get("fragment_parquet", "fragment_clusters.parquet")

    frag_file_pattern = out_cfg.get("fragment_file_pattern")
    frag_parq_pattern = out_cfg.get("fragment_parquet_pattern")

    # fixed structure under <root>/fragment/…
    frag_root = base_dir / "fragment"
    frag_bin_dir = frag_root / "bin"       # per-WG .binz
    frag_parq_dir = frag_root / "parquet"  # per-WG .parquet

    # Use DIA MS/MS info table to get window groups
    info = ds.dia_ms_ms_info
    wgs = sorted(int(w) for w in info.WindowGroup.unique())
    log(f"[fragments] {len(wgs)} window-groups")

    if fragment_split_by_wg:
        # ---- per-WG writing mode ------------------------------------------
        for wg in wgs:
            log(f"[fragment] WG={wg}")
            plan = build_plan(ds, plan_cfg, precompute_views, for_group=wg)

            stitched_wg = detect_and_stitch_for_plan(
                plan, cfg["run"], cfg["detector"], stitch_cfg
            )

            attach_raw = bool(cfg["run"].get("attach_raw_data", False))

            clusters_wg = ds.clusters_for_group(
                window_group=int(wg),
                im_peaks=stitched_wg,
                tof_step=int(c.get("tof_step", 1)),
                bin_pad=float(c.get("bin_pad", 30.0)),
                smooth_sigma_sec=float(c.get("smooth_sigma_sec", 1.25)),
                smooth_trunc_k=float(c.get("smooth_trunc_k", 3.0)),
                min_prom=float(c.get("min_prom", 25.0)),
                min_sep_sec=float(c.get("min_sep_sec", 2.0)),
                min_width_sec=float(c.get("min_width_sec", 2.0)),
                fallback_if_frames_lt=int(c.get("fallback_if_frames_lt", 5)),
                fallback_frac_width=float(c.get("fallback_frac_width", 0.50)),
                extra_rt_pad=int(c.get("extra_rt_pad", 0)),
                extra_im_pad=int(c.get("extra_im_pad", 0)),
                tof_bin_pad=int(c.get("tof_bin_pad", 1)),
                tof_hist_pad=int(c.get("tof_hist_pad", 64)),
                refine_tof_once=bool(c.get("refine_tof_once", True)),
                refine_k_sigma=float(c.get("refine_k_sigma", 3.0)),
                attach_axes=True,
                attach_points=attach_raw,
                attach_max_points=int(c.get("attach_max_points", 512)),
                require_rt_overlap=bool(c.get("require_rt_overlap", True)),
                compute_mz_from_tof=bool(c.get("compute_mz_from_tof", True)),
                num_threads=int(c.get("num_threads", 0)),
                min_im_span=int(c.get("min_im_span", 10)),
                attach_im_xic=True,
                attach_rt_xic=True,
                pad_tof_bins=0,
                pad_im_scans=0,
                pad_rt_frames=1,
            )

            # ---- per-WG binary save ----
            if frag_file_pattern:
                frag_fname = frag_file_pattern.format(wg=wg)
            else:
                p = Path(frag_file)
                frag_fname = f"{p.stem}_WG{wg:03d}{p.suffix}"

            out_bin = frag_bin_dir / frag_fname
            ensure_dir_for_file(out_bin)
            save_clusters_bin(path=str(out_bin), clusters=clusters_wg, compress=compress)
            log(f"[ok] wrote fragment clusters WG={wg} -> {out_bin} (compress={compress})")

            # ---- per-WG parquet ----
            if parquet_enabled:
                if frag_parq_pattern:
                    parq_fname = frag_parq_pattern.format(wg=wg)
                else:
                    p = Path(frag_parq)
                    parq_fname = f"{p.stem}_WG{wg:03d}{p.suffix}"

                out_parq = frag_parq_dir / parq_fname
                ensure_dir_for_file(out_parq)
                df = pd.DataFrame([c.to_dict() for c in clusters_wg])
                df.to_parquet(out_parq, index=False)
                log(f"[ok] wrote fragment parquet WG={wg} -> {out_parq}")

            del stitched_wg, clusters_wg
            cuda_gc()

        log("[fragments] per-WG writing completed")
        return

    # ---- original all-in-one mode ------------------------------------------
    all_clusters: list = []

    for wg in wgs:
        log(f"[fragment] WG={wg}")
        plan = build_plan(ds, plan_cfg, precompute_views, for_group=wg)

        stitched_wg = detect_and_stitch_for_plan(
            plan, cfg["run"], cfg["detector"], stitch_cfg
        )

        attach_raw = bool(cfg["run"].get("attach_raw_data", False))

        clusters_wg = ds.clusters_for_group(
            window_group=int(wg),
            im_peaks=stitched_wg,
            tof_step=int(c.get("tof_step", 1)),
            bin_pad=float(c.get("bin_pad", 30.0)),
            smooth_sigma_sec=float(c.get("smooth_sigma_sec", 1.25)),
            smooth_trunc_k=float(c.get("smooth_trunc_k", 3.0)),
            min_prom=float(c.get("min_prom", 25.0)),
            min_sep_sec=float(c.get("min_sep_sec", 2.0)),
            min_width_sec=float(c.get("min_width_sec", 2.0)),
            fallback_if_frames_lt=int(c.get("fallback_if_frames_lt", 5)),
            fallback_frac_width=float(c.get("fallback_frac_width", 0.50)),
            extra_rt_pad=int(c.get("extra_rt_pad", 0)),
            extra_im_pad=int(c.get("extra_im_pad", 0)),
            tof_bin_pad=int(c.get("tof_bin_pad", 1)),
            tof_hist_pad=int(c.get("tof_hist_pad", 64)),
            refine_tof_once=bool(c.get("refine_tof_once", True)),
            refine_k_sigma=float(c.get("refine_k_sigma", 3.0)),
            attach_axes=True,
            attach_points=attach_raw,
            attach_max_points=int(c.get("attach_max_points", 512)),
            require_rt_overlap=bool(c.get("require_rt_overlap", True)),
            compute_mz_from_tof=bool(c.get("compute_mz_from_tof", True)),
            num_threads=int(c.get("num_threads", 0)),
            min_im_span=int(c.get("min_im_span", 10)),
        )
        all_clusters.extend(clusters_wg)

        del stitched_wg, clusters_wg
        cuda_gc()

    # ---- binary save (all) ----
    out_bin = frag_root / frag_file
    ensure_dir_for_file(out_bin)
    save_clusters_bin(path=str(out_bin), clusters=all_clusters, compress=compress)
    log(f"[ok] wrote fragment clusters -> {out_bin} (compress={compress})")

    # ---- optional parquet (all) ----
    if parquet_enabled:
        out_parq = frag_root / frag_parq
        ensure_dir_for_file(out_parq)
        df = pd.DataFrame([c.to_dict() for c in all_clusters])
        df.to_parquet(out_parq, index=False)
        log(f"[ok] wrote fragment parquet -> {out_parq}")

    del all_clusters
    cuda_gc()


# ---------- CLI & config ------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        cfg = toml.load(f)

    # light validation / defaults
    for section in ("dataset", "output", "run", "detector", "stitch", "plans"):
        if section not in cfg:
            raise ValueError(f"Missing [{section}] in config.")

    # output
    cfg["output"].setdefault("compress_bin", True)
    # fragment-per-WG defaults
    out = cfg["output"]
    out.setdefault("fragment_split_by_wg", False)
    out.setdefault("fragment_file_pattern", None)
    out.setdefault("fragment_parquet_pattern", None)

    # run defaults
    cfg["run"].setdefault("stage", "both")
    cfg["run"].setdefault("device", "cuda")
    cfg["run"].setdefault("batch_size", 64)
    cfg["run"].setdefault("precompute_views", True)
    cfg["run"].setdefault("fragments_enabled", False)  # default OFF (opt-in)
    cfg["run"].setdefault("attach_raw_data", False)

    # stitching defaults
    cfg["stitch"].setdefault("precursor", {})
    cfg["stitch"].setdefault("fragment", {})
    for s in ("precursor", "fragment"):
        cfg["stitch"][s].setdefault("max_tof_row_delta", 0)
        cfg["stitch"][s].setdefault("max_scan_delta", 5)
        cfg["stitch"][s].setdefault("min_overlap_frames", 1)
        cfg["stitch"][s].setdefault("min_im_overlap_scans", 1)
        cfg["stitch"][s].setdefault("jaccard_min", 0.0)
        cfg["stitch"][s].setdefault("im_jaccard_min", 0.0)
        cfg["stitch"][s].setdefault("allow_cross_groups", False)
        cfg["stitch"][s].setdefault("require_mutual_apex_inside", True)
        cfg["stitch"][s].setdefault("use_batch_stitch", False)

    # detector defaults (TOF-based)
    cfg.setdefault("detector", {})
    d = cfg["detector"]
    d.setdefault("pool_scan", 11)
    d.setdefault("pool_tof", 5)
    d.setdefault("min_intensity_scaled", 12.0)
    d.setdefault("tile_rows", 65_536)
    d.setdefault("tile_overlap", 64)
    d.setdefault("fit_h", 35)
    d.setdefault("fit_w", 11)
    d.setdefault("refine", "adam")  # "none" | "adam" | "gn" | "gauss_newton"
    d.setdefault("refine_iters", 8)
    d.setdefault("refine_lr", 0.2)
    d.setdefault("refine_mask_k", 2.5)
    d.setdefault("refine_scan", True)
    d.setdefault("refine_tof", True)
    d.setdefault("refine_sigma_scan", True)
    d.setdefault("refine_sigma_tof", True)
    d.setdefault("scale", "sqrt")        # "none" | "sqrt" | "cbrt" | "log1p"
    d.setdefault("output_units", "original")
    d.setdefault("gn_float64", False)
    d.setdefault("do_dedup", True)
    d.setdefault("tol_scan", 0.75)
    d.setdefault("tol_tof", 0.25)
    d.setdefault("k_sigma", 3.0)
    d.setdefault("min_width", 3)
    d.setdefault("topk_per_tile", None)          # if set <=0, treated as None
    d.setdefault("patch_batch_target_mb", 128)
    d.setdefault("blur_sigma_scan", 0.0)
    d.setdefault("blur_sigma_tof", 0.0)
    d.setdefault("blur_truncate", 3.0)

    # plans defaults (TOF-based)
    cfg.setdefault("plans", {})
    cfg["plans"].setdefault("precursor", {})
    cfg["plans"].setdefault("fragment", {})

    pp = cfg["plans"]["precursor"]
    pp.setdefault("tof_step", 1)
    pp.setdefault("rt_window_sec", 6.0)
    pp.setdefault("rt_hop_sec", 3.0)
    pp.setdefault("sigma_scans", 9.0)
    pp.setdefault("sigma_tof_bins", 1.0)
    pp.setdefault("truncate", 3.0)
    pp.setdefault("num_threads", 4)

    pf = cfg["plans"]["fragment"]
    pf.setdefault("tof_step", 1)
    pf.setdefault("rt_window_sec", 3.0)
    pf.setdefault("rt_hop_sec", 1.5)
    pf.setdefault("sigma_scans", 5.0)
    pf.setdefault("sigma_tof_bins", 1.0)
    pf.setdefault("truncate", 3.0)
    pf.setdefault("num_threads", 4)

    # clustering defaults (new API)
    cfg.setdefault("cluster", {})
    cfg["cluster"].setdefault("precursor", {})
    cfg["cluster"].setdefault("fragment", {})

    # precursor clustering defaults
    cp = cfg["cluster"]["precursor"]
    cp.setdefault("tof_step", 1)
    cp.setdefault("bin_pad", 10.0)
    cp.setdefault("smooth_sigma_sec", 1.25)
    cp.setdefault("smooth_trunc_k", 3.0)
    cp.setdefault("min_prom", 50.0)
    cp.setdefault("min_sep_sec", 2.0)
    cp.setdefault("min_width_sec", 2.0)
    cp.setdefault("fallback_if_frames_lt", 5)
    cp.setdefault("fallback_frac_width", 0.50)
    cp.setdefault("extra_rt_pad", 0)
    cp.setdefault("extra_im_pad", 0)
    cp.setdefault("tof_bin_pad", 1)
    cp.setdefault("tof_hist_bins", 64)
    cp.setdefault("refine_tof_once", True)
    cp.setdefault("refine_k_sigma", 3.0)
    cp.setdefault("attach_max_points", 512)
    cp.setdefault("require_rt_overlap", True)
    cp.setdefault("compute_mz_from_tof", True)
    cp.setdefault("num_threads", 0)
    cp.setdefault("min_im_span", 10)

    # fragment clustering defaults
    cf = cfg["cluster"]["fragment"]
    cf.setdefault("tof_step", 1)
    cf.setdefault("bin_pad", 30.0)
    cf.setdefault("smooth_sigma_sec", 1.25)
    cf.setdefault("smooth_trunc_k", 3.0)
    cf.setdefault("min_prom", 25.0)
    cf.setdefault("min_sep_sec", 2.0)
    cf.setdefault("min_width_sec", 2.0)
    cf.setdefault("fallback_if_frames_lt", 5)
    cf.setdefault("fallback_frac_width", 0.50)
    cf.setdefault("extra_rt_pad", 0)
    cf.setdefault("extra_im_pad", 0)
    cf.setdefault("tof_bin_pad", 1)
    cf.setdefault("tof_hist_pad", 64)
    cf.setdefault("refine_tof_once", True)
    cf.setdefault("refine_k_sigma", 3.0)
    cf.setdefault("attach_max_points", 512)
    cf.setdefault("require_rt_overlap", True)
    cf.setdefault("compute_mz_from_tof", True)
    cf.setdefault("num_threads", 0)
    cf.setdefault("min_im_span", 10)

    return cfg


def _default_log_path_from_cfg(cfg: dict) -> Path | None:
    out_dir = cfg.get("output", {}).get("dir")
    if not out_dir:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(out_dir) / "logs" / f"timsim_{ts}.log"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="IM peak detection → stitching → clustering (TOML-configured)."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config.toml")
    parser.add_argument("--stage", choices=["precursor", "fragments", "both"],
                        help="Override run.stage from config")
    parser.add_argument("--device", help="Override run.device (e.g., 'cuda', 'cpu')")

    # explicit opt-in/out for fragments
    parser.add_argument("--fragments", dest="fragments", action="store_true",
                        help="Opt-in to fragment processing (overrides config)")
    parser.add_argument("--no-fragments", dest="fragments", action="store_false",
                        help="Opt-out of fragment processing (overrides config)")
    parser.set_defaults(fragments=None)

    # logging flags
    parser.add_argument("--log-file", default=None,
                        help="Log file path (defaults to [output]/logs/timsim_*.log)")
    parser.add_argument("--log-level", default="INFO",
                        help="Log level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--no-console-log", action="store_true",
                        help="Disable console logging")

    # simple clustering overrides
    parser.add_argument("--prec-bin-pad", type=float,
                        help="Override cluster.precursor.bin_pad")
    parser.add_argument("--prec-min-prom", type=float,
                        help="Override cluster.precursor.min_prom")
    parser.add_argument("--prec-attach-max", type=int,
                        help="Override cluster.precursor.attach_max_points")
    parser.add_argument("--frag-bin-pad", type=float,
                        help="Override cluster.fragment.bin_pad")
    parser.add_argument("--frag-min-prom", type=float,
                        help="Override cluster.fragment.min_prom")
    parser.add_argument("--frag-attach-max", type=int,
                        help="Override cluster.fragment.attach_max_points")

    # output compression toggle
    parser.add_argument("--compress-bin", dest="compress_bin", action="store_true",
                        help="Enable compression for .binz outputs")
    parser.add_argument("--no-compress-bin", dest="compress_bin", action="store_false",
                        help="Disable compression for .binz outputs")
    parser.set_defaults(compress_bin=None)

    args = parser.parse_args(argv)

    # load config first (we need output dir to auto-pick log path)
    cfg = load_config(args.config)

    # decide log file path & init logging
    log_file = args.log_file or _default_log_path_from_cfg(cfg)
    setup_logging(
        log_file=log_file,
        level=args.log_level,
        also_console=not args.no_console_log,
    )
    if log_file:
        log(f"[log] writing logfile -> {log_file}")

    # reflect CLI overrides
    if args.stage:
        cfg["run"]["stage"] = args.stage
    if args.device:
        cfg["run"]["device"] = args.device
    if args.fragments is not None:
        cfg["run"]["fragments_enabled"] = bool(args.fragments)

    # clustering overrides
    if args.prec_bin_pad is not None:
        cfg["cluster"]["precursor"]["bin_pad"] = float(args.prec_bin_pad)
    if args.prec_min_prom is not None:
        cfg["cluster"]["precursor"]["min_prom"] = float(args.prec_min_prom)
    if args.prec_attach_max is not None:
        cfg["cluster"]["precursor"]["attach_max_points"] = int(args.prec_attach_max)

    if args.frag_bin_pad is not None:
        cfg["cluster"]["fragment"]["bin_pad"] = float(args.frag_bin_pad)
    if args.frag_min_prom is not None:
        cfg["cluster"]["fragment"]["min_prom"] = float(args.frag_min_prom)
    if args.frag_attach_max is not None:
        cfg["cluster"]["fragment"]["attach_max_points"] = int(args.frag_attach_max)

    # output compression override
    if args.compress_bin is not None:
        cfg["output"]["compress_bin"] = bool(args.compress_bin)

    # environment preamble
    log(f"[env] Python {sys.version.split()[0]}")
    log(f"[env] Torch {torch.__version__} | CUDA available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            log(f"[env] CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    print_config_summary(cfg)

    # open dataset
    ds = TimsDatasetDIA(
        cfg["dataset"]["path"],
        use_bruker_sdk=bool(cfg["dataset"].get("use_bruker_sdk", False)),
    )

    # Decide which stages to run
    stage = cfg["run"].get("stage", "both")
    fragments_enabled = bool(cfg["run"].get("fragments_enabled", False))

    # Precursor: always allowed unless stage=="fragments"
    if stage in ("precursor", "both"):
        run_precursor(ds, cfg)

    # Fragments: run if (stage says so) AND (opt-in true)
    if stage in ("fragments", "both") and fragments_enabled:
        run_fragments(ds, cfg)
    elif stage in ("fragments", "both") and not fragments_enabled:
        log("[skip] fragments disabled (opt-in required)")

    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())