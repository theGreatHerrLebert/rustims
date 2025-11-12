#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IM peak detection → stitching → clustering (TOML-configured) with logging and CLI overrides.
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
import torch

# Python 3.11+ has tomllib; otherwise optional tomli fallback
try:
    import tomllib as toml
except Exception:
    print("Warning: tomllib not found, falling back to tomli")
    import tomli as toml  # type: ignore

from imspy.timstof.clustering.torch_extractor import iter_im_peaks_batches
from imspy.timstof.dia import (
    stitch_im_peaks_flat,
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
        # also pass to default hook (so behavior stays familiar)
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

    lines = []
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

    # Plans + Stitch
    for sec in ("precursor", "fragment"):
        if sec in plan_cfg:
            p = plan_cfg[sec]
            lines.append(f"[plans.{sec}]")
            lines.append(f"  ppm_per_bin          : {float(p.get('ppm_per_bin'))}")
            lines.append(f"  mz_pad_ppm           : {float(p.get('mz_pad_ppm'))}")
            lines.append(f"  rt_window_sec        : {float(p.get('rt_window_sec'))}")
            lines.append(f"  rt_hop_sec           : {float(p.get('rt_hop_sec'))}")
            lines.append(f"  im_sigma_scans       : {float(p.get('im_sigma_scans'))}")
            lines.append(f"  mz_sigma_bins        : {float(p.get('mz_sigma_bins'))}")
        if sec in st_cfg:
            s = st_cfg[sec]
            lines.append(f"[stitch.{sec}]")
            lines.append(f"  max_mz_row_delta     : {int(s.get('max_mz_row_delta', 0))}")
            lines.append(f"  max_scan_delta       : {int(s.get('max_scan_delta', 0))}")
            lines.append(f"  jaccard_min          : {float(s.get('jaccard_min', 0.0))}")
            lines.append(f"  im_jaccard_min       : {float(s.get('im_jaccard_min', 0.0))}")
            lines.append(f"  use_batch_stitch     : {bool(s.get('use_batch_stitch', False))}")

    # Cluster (new)
    if "cluster" in cfg:
        for sec in ("precursor", "fragment"):
            if sec in clus_cfg:
                cc = clus_cfg[sec]
                lines.append(f"[cluster.{sec}]")
                lines.append(f"  ppm_per_bin          : {float(cc.get('ppm_per_bin'))}")
                lines.append(f"  bin_pad              : {float(cc.get('bin_pad'))}")
                lines.append(f"  min_prom             : {float(cc.get('min_prom'))}")
                lines.append(f"  attach_max_points    : {int(cc.get('attach_max_points'))}")

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
    if for_group is None:
        return ds.plan_mz_scan_windows(
            ppm_per_bin=float(plan_cfg["ppm_per_bin"]),
            mz_pad_ppm=float(plan_cfg["mz_pad_ppm"]),
            rt_window_sec=float(plan_cfg["rt_window_sec"]),
            rt_hop_sec=float(plan_cfg["rt_hop_sec"]),
            im_sigma_scans=float(plan_cfg["im_sigma_scans"]),
            mz_sigma_bins=float(plan_cfg["mz_sigma_bins"]),
            precompute_views=bool(precompute_views),
        )
    else:
        return ds.plan_mz_scan_windows_for_group(
            window_group=int(for_group),
            ppm_per_bin=float(plan_cfg["ppm_per_bin"]),
            mz_pad_ppm=float(plan_cfg["mz_pad_ppm"]),
            rt_window_sec=float(plan_cfg["rt_window_sec"]),
            rt_hop_sec=float(plan_cfg["rt_hop_sec"]),
            im_sigma_scans=float(plan_cfg["im_sigma_scans"]),
            mz_sigma_bins=float(plan_cfg["mz_sigma_bins"]),
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
        stitched_running = []
    else:
        collected = []

    num_batches = (len(plan) + int(run_cfg["batch_size"]) - 1) // int(run_cfg["batch_size"])
    log(f"[detect] plan has {len(plan)} mz-im planes -> (~{num_batches} batches)")

    for peak_batch in iter_im_peaks_batches(
        plan,
        batch_size=int(run_cfg["batch_size"]),
        device=str(run_cfg["device"]),
        # detector / refinement
        pool_scan=int(det_cfg["pool_scan"]),
        pool_mz=int(det_cfg["pool_mz"]),
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
        refine_mz=bool(det_cfg["refine_mz"]),
        refine_sigma_scan=bool(det_cfg["refine_sigma_scan"]),
        refine_sigma_mz=bool(det_cfg["refine_sigma_mz"]),
        scale=str(det_cfg["scale"]),
        output_units=str(det_cfg["output_units"]),
        gn_float64=bool(det_cfg["gn_float64"]),
        do_dedup=bool(det_cfg["do_dedup"]),
        tol_scan=float(det_cfg["tol_scan"]),
        tol_mz=float(det_cfg["tol_mz"]),
        k_sigma=float(det_cfg["k_sigma"]),
        min_width=int(det_cfg["min_width"]),
        mz_bounds_pad_ppm=float(det_cfg["mz_bounds_pad_ppm"]),
        mz_bounds_pad_abs=float(det_cfg["mz_bounds_pad_abs"]),
    ):
        if use_batch_stitch:
            stitched_batch = stitch_im_peaks_flat(
                peak_batch,
                max_mz_row_delta=int(stitch_cfg["max_mz_row_delta"]),
                max_scan_delta=int(stitch_cfg["max_scan_delta"]),
                jaccard_min=float(stitch_cfg.get("jaccard_min", 0.0)),
                im_jaccard_min=float(stitch_cfg.get("im_jaccard_min", 0.0)),
            )
            stitched_running = stitch_im_peaks_flat(
                stitched_running + stitched_batch,
                max_mz_row_delta=int(stitch_cfg["max_mz_row_delta"]),
                max_scan_delta=int(stitch_cfg["max_scan_delta"]),
                jaccard_min=float(stitch_cfg.get("jaccard_min", 0.0)),
                im_jaccard_min=float(stitch_cfg.get("im_jaccard_min", 0.0)),
            )
            del peak_batch, stitched_batch
        else:
            collected.extend(peak_batch)

        cuda_gc()

    if use_batch_stitch:
        return stitched_running
    else:
        return stitch_im_peaks_flat(
            collected,
            max_mz_row_delta=int(stitch_cfg["max_mz_row_delta"]),
            max_scan_delta=int(stitch_cfg["max_scan_delta"]),
            jaccard_min=float(stitch_cfg.get("jaccard_min", 0.0)),
            im_jaccard_min=float(stitch_cfg.get("im_jaccard_min", 0.0)),
        )


def run_precursor(ds, cfg):
    from imspy.timstof.dia import clusters_to_dataframe

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
        ppm_per_bin=float(c["ppm_per_bin"]),
        bin_pad=float(c["bin_pad"]),
        min_prom=float(c["min_prom"]),
        attach_points=attach_raw,
        attach_axes=attach_raw,
        attach_max_points=int(c["attach_max_points"]),
    )

    # ---- binary save ----
    out_bin = Path(cfg["output"]["dir"]) / cfg["output"]["precursor_file"]
    ensure_dir_for_file(out_bin)
    compress = bool(cfg["output"].get("compress_bin", True))
    save_clusters_bin(clusters=clusters, path=str(out_bin), compress=compress)
    log(f"[ok] wrote precursor clusters -> {out_bin} (compress={compress})")

    # ---- optional parquet ----
    if bool(cfg["output"].get("parquet_enabled", False)):
        out_parq = Path(cfg["output"]["dir"]) / cfg["output"]["precursor_parquet"]
        ensure_dir_for_file(out_parq)
        df = clusters_to_dataframe(clusters)
        df.to_parquet(out_parq, index=False)
        log(f"[ok] wrote precursor parquet -> {out_parq}")

    del stitched, clusters
    cuda_gc()


def run_fragments(ds, cfg):
    from imspy.timstof.dia import clusters_to_dataframe

    log("[stage] fragments")

    if "fragment" not in cfg.get("plans", {}) or "fragment" not in cfg.get("stitch", {}):
        log("[skip] fragments: no [plans.fragment] or [stitch.fragment] in config")
        return

    precompute_views = bool(cfg["run"]["precompute_views"])
    plan_cfg = cfg["plans"]["fragment"]
    stitch_cfg = cfg["stitch"]["fragment"]
    c = cfg["cluster"]["fragment"]

    all_clusters = []
    wgs = sorted(set(ds.dia_ms_ms_info.WindowGroup))
    log(f"[fragments] {len(wgs)} window-groups")

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
            ppm_per_bin=float(c["ppm_per_bin"]),
            bin_pad=float(c["bin_pad"]),
            min_prom=float(c["min_prom"]),
            attach_points=attach_raw,
            attach_axes=attach_raw,
            attach_max_points=int(c["attach_max_points"]),
        )
        all_clusters.extend(clusters_wg)

        del stitched_wg, clusters_wg
        cuda_gc()

    # ---- binary save ----
    out_bin = Path(cfg["output"]["dir"]) / cfg["output"]["fragment_file"]
    ensure_dir_for_file(out_bin)
    compress = bool(cfg["output"].get("compress_bin", True))
    save_clusters_bin(clusters=all_clusters, path=str(out_bin), compress=compress)
    log(f"[ok] wrote fragment clusters -> {out_bin} (compress={compress})")

    # ---- optional parquet ----
    if bool(cfg["output"].get("parquet_enabled", False)):
        out_parq = Path(cfg["output"]["dir"]) / cfg["output"]["fragment_parquet"]
        ensure_dir_for_file(out_parq)
        df = clusters_to_dataframe(all_clusters)
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
        cfg["stitch"][s].setdefault("max_mz_row_delta", 0)
        cfg["stitch"][s].setdefault("max_scan_delta", 5)
        cfg["stitch"][s].setdefault("jaccard_min", 0.0)
        cfg["stitch"][s].setdefault("im_jaccard_min", 0.0)
        cfg["stitch"][s].setdefault("use_batch_stitch", False)

    # clustering defaults (NEW)
    cfg.setdefault("cluster", {})
    cfg["cluster"].setdefault("precursor", {})
    cfg["cluster"].setdefault("fragment", {})

    # precursor clustering defaults
    cfg["cluster"]["precursor"].setdefault("ppm_per_bin", 15.0)
    cfg["cluster"]["precursor"].setdefault("bin_pad", 30.0)
    cfg["cluster"]["precursor"].setdefault("min_prom", 50.0)
    cfg["cluster"]["precursor"].setdefault("attach_max_points", 5000)

    # fragment clustering defaults
    cfg["cluster"]["fragment"].setdefault("ppm_per_bin", 15.0)
    cfg["cluster"]["fragment"].setdefault("bin_pad", 30.0)
    cfg["cluster"]["fragment"].setdefault("min_prom", 25.0)
    cfg["cluster"]["fragment"].setdefault("attach_max_points", 1000)

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

    # clustering overrides
    # precursor
    parser.add_argument("--prec-ppm-per-bin", type=float,
                        help="Override cluster.precursor.ppm_per_bin")
    parser.add_argument("--prec-bin-pad", type=float,
                        help="Override cluster.precursor.bin_pad")
    parser.add_argument("--prec-min-prom", type=float,
                        help="Override cluster.precursor.min_prom")
    parser.add_argument("--prec-attach-max", type=int,
                        help="Override cluster.precursor.attach_max_points")
    # fragment
    parser.add_argument("--frag-ppm-per-bin", type=float,
                        help="Override cluster.fragment.ppm_per_bin")
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
    if args.prec_ppm_per_bin is not None:
        cfg["cluster"]["precursor"]["ppm_per_bin"] = float(args.prec_ppm_per_bin)
    if args.prec_bin_pad is not None:
        cfg["cluster"]["precursor"]["bin_pad"] = float(args.prec_bin_pad)
    if args.prec_min_prom is not None:
        cfg["cluster"]["precursor"]["min_prom"] = float(args.prec_min_prom)
    if args.prec_attach_max is not None:
        cfg["cluster"]["precursor"]["attach_max_points"] = int(args.prec_attach_max)

    if args.frag_ppm_per_bin is not None:
        cfg["cluster"]["fragment"]["ppm_per_bin"] = float(args.frag_ppm_per_bin)
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