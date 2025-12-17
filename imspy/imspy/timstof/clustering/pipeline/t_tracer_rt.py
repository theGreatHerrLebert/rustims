#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RT peak detection → clustering (TOML-configured) with logging and CLI overrides.

This is the RT-centric sibling of the IM-first pipeline:
- Build dense TOF×RT grids (precursor and optionally per fragment window-group)
- Detect RT peaks via detect_rt_peaks_for_grid (torch extractor)
- Cluster via:
    - ds.clusters_for_precursor_from_rt_peaks(...)
    - ds.clusters_for_group_from_rt_peaks(...)   (if available)
      OR fallback to ds.clusters_for_group(...) if you implement the RT->IM expansion there.

Notes:
- No IM stitching stage here; RT peaks are already “global” in RT.
- Keep the same output structure: <out>/precursor, <out>/fragment.
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
from contextlib import contextmanager
import time

import torch

def assert_cuda_healthy(device: str) -> None:
    if device != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    try:
        # minimal alloc + sync to catch ECC immediately
        _ = torch.empty((1,), device="cuda")
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(
            "CUDA device is not healthy (often ECC). "
            "Try CUDA_VISIBLE_DEVICES=1 or run on CPU."
        ) from e

# Python 3.11+ tomllib; fallback tomli
try:
    import tomllib as toml
except Exception:
    print("Warning: tomllib not found, falling back to tomli")
    import tomli as toml  # type: ignore

from imspy.timstof.dia import (
    save_clusters_parquet,
    TimsDatasetDIA,
)
from imspy.timstof.clustering.pipeline.torch_extractor import detect_rt_peaks_for_grid


# --------------------------- logging ------------------------------------------
_LOGGER_NAME = "t-tracer"
_logger = logging.getLogger(_LOGGER_NAME)

@contextmanager
def log_timing(label: str):
    _logger.info(f"[timing] {label}: start")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        _logger.info(f"[timing] {label}: done in {dt:0.3f} s")


def setup_logging(
    log_file: str | os.PathLike | None,
    level: str = "INFO",
    also_console: bool = True,
    rotate_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
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

    def _excepthook(exc_type, exc, tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook


def log(msg: str, level: int = logging.INFO) -> None:
    _logger.log(level, msg)


def ensure_dir_for_file(path: str | os.PathLike) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def cuda_gc() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ------------------------ config helpers --------------------------------------
def get_detector_cfg(cfg: dict, role: str) -> dict:
    """
    Build an *effective* RT detector config for a given role ("precursor" or "fragment").

    Rules:
      - Start from [rt_detector] as shared defaults.
      - If [rt_detector.<role>] exists, overlay/override those keys.
    """
    det_root = cfg.get("rt_detector", {}) or {}
    eff = dict(det_root)
    role_cfg = det_root.get(role)
    if isinstance(role_cfg, dict):
        eff.update(role_cfg)
    return eff


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        cfg = toml.load(f)

    # minimal validation
    for section in ("dataset", "output", "run"):
        if section not in cfg:
            raise ValueError(f"Missing [{section}] in config.")

    # output defaults
    out = cfg["output"]
    out.setdefault("precursor_parquet", "precursor_clusters.parquet")
    out.setdefault("fragment_parquet", "fragment_clusters.parquet")
    out.setdefault("fragment_split_by_wg", False)
    out.setdefault("fragment_parquet_pattern", None)

    # run defaults
    cfg["run"].setdefault("stage", "both")
    cfg["run"].setdefault("device", "cuda")
    cfg["run"].setdefault("fragments_enabled", False)
    cfg["run"].setdefault("attach_raw_data", False)

    # RT detector defaults
    cfg.setdefault("rt_detector", {})
    d = cfg["rt_detector"]
    d.setdefault("tof_step", 1)

    d.setdefault("pool_rt", 15)
    d.setdefault("pool_tof", 3)
    d.setdefault("min_intensity_scaled", 5.0)
    d.setdefault("tile_rows", 50_000)
    d.setdefault("tile_overlap", 32)
    d.setdefault("fit_h", 15)
    d.setdefault("fit_w", 5)

    d.setdefault("refine", "adam")
    d.setdefault("refine_iters", 8)
    d.setdefault("refine_lr", 0.2)
    d.setdefault("refine_mask_k", 2.5)
    d.setdefault("refine_rt", True)
    d.setdefault("refine_tof", True)
    d.setdefault("refine_sigma_rt", True)
    d.setdefault("refine_sigma_tof", True)

    d.setdefault("scale", "sqrt")
    d.setdefault("output_units", "original")
    d.setdefault("gn_float64", False)

    d.setdefault("do_dedup", True)
    d.setdefault("tol_rt", 0.5)
    d.setdefault("tol_tof", 0.25)

    d.setdefault("k_sigma", 2.0)
    d.setdefault("min_width_frames", 3)

    d.setdefault("blur_sigma_rt", 0.5)
    d.setdefault("blur_sigma_tof", 0.5)
    d.setdefault("blur_truncate", 3.0)

    d.setdefault("topk_per_tile", None)
    d.setdefault("patch_batch_target_mb", 128)

    cfg["rt_detector"].setdefault("precursor", {})
    cfg["rt_detector"].setdefault("fragment", {})

    # clustering defaults for RT-centric path
    cfg.setdefault("rt_cluster", {})
    cfg["rt_cluster"].setdefault("precursor", {})
    cfg["rt_cluster"].setdefault("fragment", {})

    cp = cfg["rt_cluster"]["precursor"]
    cp.setdefault("num_threads", 0)
    cp.setdefault("min_im_span", 15)

    cp.setdefault("tof_step", 1)
    cp.setdefault("pad_rt_frames", 2)
    cp.setdefault("pad_tof_bins", 2)

    cp.setdefault("im_smooth_sigma_scans", 10.0)
    cp.setdefault("im_min_prom", 20.0)

    cf = cfg["rt_cluster"]["fragment"]
    cf.setdefault("num_threads", 0)
    cf.setdefault("min_im_span", 10)

    cf.setdefault("tof_step", 1)
    cf.setdefault("pad_rt_frames", 0)
    cf.setdefault("pad_tof_bins", 2)

    cf.setdefault("im_smooth_sigma_scans", 10.0)
    cf.setdefault("im_min_prom", 15.0)

    return cfg


def print_config_summary(cfg: dict) -> None:
    ds_cfg = cfg.get("dataset", {})
    out_cfg = cfg.get("output", {})
    run_cfg = cfg.get("run", {})

    lines: list[str] = []
    lines.append("──────────────── CONFIG SUMMARY (RT) ───────────")
    lines.append("[dataset]")
    lines.append(f"  path                 : {ds_cfg.get('path')}")
    lines.append(f"  use_bruker_sdk       : {bool(ds_cfg.get('use_bruker_sdk', False))}")

    lines.append("[run]")
    lines.append(f"  stage                : {run_cfg.get('stage')}")
    lines.append(f"  device               : {run_cfg.get('device')}")
    lines.append(f"  fragments_enabled    : {bool(run_cfg.get('fragments_enabled', False))}")
    lines.append(f"  attach_raw_data      : {bool(run_cfg.get('attach_raw_data', False))}")

    lines.append("[output]")
    lines.append(f"  dir                  : {out_cfg.get('dir')}")
    lines.append(f"  precursor_parquet    : {out_cfg.get('precursor_parquet')}")
    lines.append(f"  fragment_parquet     : {out_cfg.get('fragment_parquet')}")
    lines.append(f"  fragment_split_by_wg : {bool(out_cfg.get('fragment_split_by_wg', False))}")

    for sec in ("precursor", "fragment"):
        eff = get_detector_cfg(cfg, sec)
        if eff:
            lines.append(f"[rt_detector.{sec}] (effective)")
            keys = [
                "tof_step","pool_rt","pool_tof","min_intensity_scaled",
                "tile_rows","tile_overlap","fit_h","fit_w",
                "refine","refine_iters","refine_lr","refine_mask_k",
                "refine_rt","refine_tof","refine_sigma_rt","refine_sigma_tof",
                "scale","output_units","gn_float64",
                "do_dedup","tol_rt","tol_tof",
                "k_sigma","min_width_frames",
                "blur_sigma_rt","blur_sigma_tof","blur_truncate",
                "topk_per_tile","patch_batch_target_mb",
            ]
            for k in keys:
                if k in eff:
                    lines.append(f"  {k:20s}: {eff.get(k)}")

    lines.append("──────────────── NOTES ─────────────────────────")
    cuda_avail = torch.cuda.is_available()
    lines.append(f"  CUDA available       : {cuda_avail}")
    dev = run_cfg.get("device")
    if dev == "cpu" and cuda_avail:
        lines.append("  ⚠ You selected CPU but CUDA is available.")
    if dev == "cuda" and not cuda_avail:
        lines.append("  ⚠ CUDA requested but not available; this will fall back or fail.")

    out_dir = out_cfg.get("dir")
    if out_dir and not Path(out_dir).exists():
        lines.append(f"  ℹ Output dir will be created: {out_dir}")

    ds_path = ds_cfg.get("path")
    if ds_path and not Path(ds_path).exists():
        lines.append(f"  ⚠ Dataset path does not exist: {ds_path}")

    lines.append("───────────────────────────────────────────────")
    for ln in lines:
        log(ln)


def _default_log_path_from_cfg(cfg: dict) -> Path | None:
    out_dir = cfg.get("output", {}).get("dir")
    if not out_dir:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(out_dir) / "logs" / f"timsim_rt_{ts}.log"


# --------------------------- RT detection -------------------------------------
def detect_rt_peaks(ds: TimsDatasetDIA, *, window_group: int | None, det_cfg: dict, device: str):
    tof_step = int(det_cfg.get("tof_step", 1))

    with log_timing(f"grid.build.{('precursor' if window_group is None else f'WG{window_group:03d}')}.tof_step={tof_step}"):
        if window_group is None:
            grid = ds.tof_rt_grid_precursor(tof_step=tof_step)
        else:
            grid = ds.tof_rt_grid_for_group(window_group=int(window_group), tof_step=tof_step)

    # topk semantics
    topk = det_cfg.get("topk_per_tile", None)
    if topk is not None:
        topk = int(topk)
        if topk <= 0:
            topk = None

    peaks = None
    with log_timing(f"rt.detect.{('precursor' if window_group is None else f'WG{window_group:03d}')}"):
        peaks = detect_rt_peaks_for_grid(
            grid,
            device=device,
            pool_rt=int(det_cfg["pool_rt"]),
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
            refine_rt=bool(det_cfg["refine_rt"]),
            refine_tof=bool(det_cfg["refine_tof"]),
            refine_sigma_rt=bool(det_cfg["refine_sigma_rt"]),
            refine_sigma_tof=bool(det_cfg["refine_sigma_tof"]),
            scale=str(det_cfg["scale"]),
            output_units=str(det_cfg["output_units"]),
            gn_float64=bool(det_cfg["gn_float64"]),
            do_dedup=bool(det_cfg["do_dedup"]),
            tol_rt=float(det_cfg["tol_rt"]),
            tol_tof=float(det_cfg["tol_tof"]),
            k_sigma=float(det_cfg["k_sigma"]),
            min_width_frames=int(det_cfg["min_width_frames"]),
            blur_sigma_rt=float(det_cfg["blur_sigma_rt"]),
            blur_sigma_tof=float(det_cfg["blur_sigma_tof"]),
            blur_truncate=float(det_cfg["blur_truncate"]),
            topk_per_tile=topk,
            patch_batch_target_mb=int(det_cfg["patch_batch_target_mb"]),
        )

    log(f"[rt] peaks={len(peaks)} for {('precursor' if window_group is None else f'WG={window_group}')}")
    return peaks


# --------------------------- stages -------------------------------------------
def run_precursor(ds: TimsDatasetDIA, cfg: dict):
    log("[stage] precursor (RT-centric)")

    det = get_detector_cfg(cfg, "precursor")
    cl = cfg.get("rt_cluster", {}).get("precursor", {})

    device = str(cfg["run"]["device"])
    rt_peaks = detect_rt_peaks(ds, window_group=None, det_cfg=det, device=device)

    attach_raw = bool(cfg["run"].get("attach_raw_data", False))

    with log_timing("cluster.precursor.from_rt_peaks"):
        clusters = ds.clusters_for_precursor_from_rt_peaks(
            rt_peaks=rt_peaks,
            num_threads=int(cl.get("num_threads", 0)),
            min_im_span=int(cl.get("min_im_span", 15)),
            tof_step=int(cl.get("tof_step", det.get("tof_step", 1))),
            im_smooth_sigma_scans=float(cl.get("im_smooth_sigma_scans", 10.0)),
            im_min_prom=float(cl.get("im_min_prom", 20.0)),
            pad_rt_frames=int(cl.get("pad_rt_frames", 2)),
            pad_tof_bins=int(cl.get("pad_tof_bins", 2)),
            attach_axes=True,
            attach_points=attach_raw,
        )

    out_root = Path(cfg["output"]["dir"])
    prec_dir = out_root / "precursor"
    out_parq = prec_dir / cfg["output"]["precursor_parquet"]

    with log_timing("write.precursor.parquet"):
        ensure_dir_for_file(out_parq)
        save_clusters_parquet(
            path=str(out_parq),
            clusters=clusters,
            strip_points=True,
            strip_axes=True,
        )
        log(f"[ok] wrote precursor clusters (Parquet) -> {out_parq}")

    del rt_peaks, clusters
    cuda_gc()


def run_fragments(ds: TimsDatasetDIA, cfg: dict):
    log("[stage] fragments (RT-centric)")

    out_cfg = cfg["output"]
    base_dir = Path(out_cfg["dir"])
    frag_root = base_dir / "fragment"

    det = get_detector_cfg(cfg, "fragment")
    cl = cfg.get("rt_cluster", {}).get("fragment", {})

    fragment_split_by_wg = bool(out_cfg.get("fragment_split_by_wg", False))
    frag_parq = out_cfg.get("fragment_parquet", "fragment_clusters.parquet")
    frag_parq_pattern = out_cfg.get("fragment_parquet_pattern")

    info = ds.dia_ms_ms_info
    wgs = sorted(int(w) for w in info.WindowGroup.unique())
    log(f"[fragments] {len(wgs)} window-groups")

    device = str(cfg["run"]["device"])
    attach_raw = bool(cfg["run"].get("attach_raw_data", False))

    # We require a group-level RT-peaks clustering method. If you don't have it,
    # implement it (recommended) or route via “expand RT→IM then ds.clusters_for_group”.
    have_group_from_rt = hasattr(ds, "clusters_for_group_from_rt_peaks")

    if fragment_split_by_wg:
        for wg in wgs:
            log(f"[fragment] WG={wg}")
            rt_peaks = detect_rt_peaks(ds, window_group=wg, det_cfg=det, device=device)

            if not have_group_from_rt:
                raise RuntimeError(
                    "Dataset is missing clusters_for_group_from_rt_peaks(window_group, rt_peaks, ...). "
                    "Either add it or switch to the IM-first script for fragments."
                )

            with log_timing(f"cluster.fragment.WG{wg:03d}.from_rt_peaks"):
                clusters_wg = ds.clusters_for_group_from_rt_peaks(
                    window_group=int(wg),
                    rt_peaks=rt_peaks,
                    num_threads=int(cl.get("num_threads", 0)),
                    min_im_span=int(cl.get("min_im_span", 10)),
                    tof_step=int(cl.get("tof_step", det.get("tof_step", 1))),
                    im_smooth_sigma_scans=float(cl.get("im_smooth_sigma_scans", 10.0)),
                    im_min_prom=float(cl.get("im_min_prom", 15.0)),
                    pad_rt_frames=int(cl.get("pad_rt_frames", 0)),
                    pad_tof_bins=int(cl.get("pad_tof_bins", 2)),
                    attach_axes=True,
                    attach_points=attach_raw,
                )

            if frag_parq_pattern:
                parq_fname = frag_parq_pattern.format(wg=wg)
            else:
                p = Path(frag_parq)
                parq_fname = f"{p.stem}_WG{wg:03d}{p.suffix}"

            with log_timing(f"write.fragment.parquet.WG{wg:03d}"):
                out_parq = frag_root / parq_fname
                ensure_dir_for_file(out_parq)
                save_clusters_parquet(
                    path=str(out_parq),
                    clusters=clusters_wg,
                    strip_points=True,
                    strip_axes=True,
                )
                log(f"[ok] wrote fragment clusters WG={wg} (Parquet) -> {out_parq}")

            del rt_peaks, clusters_wg
            cuda_gc()

        log("[fragments] per-WG writing completed")
        return

    # all-in-one
    all_clusters: list = []
    for wg in wgs:
        log(f"[fragment] WG={wg}")
        rt_peaks = detect_rt_peaks(ds, window_group=wg, det_cfg=det, device=device)

        if not have_group_from_rt:
            raise RuntimeError(
                "Dataset is missing clusters_for_group_from_rt_peaks(window_group, rt_peaks, ...). "
                "Either add it or switch to the IM-first script for fragments."
            )

        with log_timing(f"cluster.fragment.WG{wg:03d}.from_rt_peaks"):
            clusters_wg = ds.clusters_for_group_from_rt_peaks(
                window_group=int(wg),
                rt_peaks=rt_peaks,
                num_threads=int(cl.get("num_threads", 0)),
                min_im_span=int(cl.get("min_im_span", 10)),
                tof_step=int(cl.get("tof_step", det.get("tof_step", 1))),
                im_smooth_sigma_scans=float(cl.get("im_smooth_sigma_scans", 10.0)),
                im_min_prom=float(cl.get("im_min_prom", 15.0)),
                pad_rt_frames=int(cl.get("pad_rt_frames", 0)),
                pad_tof_bins=int(cl.get("pad_tof_bins", 2)),
                attach_axes=True,
                attach_points=attach_raw,
            )

        all_clusters.extend(clusters_wg)
        del rt_peaks, clusters_wg
        cuda_gc()

    with log_timing("write.fragment.parquet.all"):
        out_parq = frag_root / frag_parq
        ensure_dir_for_file(out_parq)
        save_clusters_parquet(
            path=str(out_parq),
            clusters=all_clusters,
            strip_points=True,
            strip_axes=True,
        )
        log(f"[ok] wrote fragment clusters (Parquet) -> {out_parq}")

    del all_clusters
    cuda_gc()


# --------------------------- CLI ----------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="RT peak detection → clustering (TOML-configured)."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config.toml")
    parser.add_argument("--stage", choices=["precursor", "fragments", "both"],
                        help="Override run.stage from config")
    parser.add_argument("--device", help="Override run.device (e.g., 'cuda', 'cpu')")

    parser.add_argument("--fragments", dest="fragments", action="store_true",
                        help="Opt-in to fragment processing (overrides config)")
    parser.add_argument("--no-fragments", dest="fragments", action="store_false",
                        help="Opt-out of fragment processing (overrides config)")
    parser.set_defaults(fragments=None)

    parser.add_argument("--log-file", default=None,
                        help="Log file path (defaults to [output]/logs/timsim_rt_*.log)")
    parser.add_argument("--log-level", default="INFO",
                        help="Log level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--no-console-log", action="store_true",
                        help="Disable console logging")

    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    log_file = args.log_file or _default_log_path_from_cfg(cfg)
    setup_logging(
        log_file=log_file,
        level=args.log_level,
        also_console=not args.no_console_log,
    )
    if log_file:
        log(f"[log] writing logfile -> {log_file}")

    assert_cuda_healthy(str(cfg["run"]["device"]))

    if args.stage:
        cfg["run"]["stage"] = args.stage
    if args.device:
        cfg["run"]["device"] = args.device
    if args.fragments is not None:
        cfg["run"]["fragments_enabled"] = bool(args.fragments)

    log(f"[env] Python {sys.version.split()[0]}")
    log(f"[env] Torch {torch.__version__} | CUDA available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            log(f"[env] CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    print_config_summary(cfg)

    ds = TimsDatasetDIA(
        cfg["dataset"]["path"],
        use_bruker_sdk=bool(cfg["dataset"].get("use_bruker_sdk", False)),
    )

    stage = cfg["run"].get("stage", "both")
    fragments_enabled = bool(cfg["run"].get("fragments_enabled", False))

    if stage in ("precursor", "both"):
        with log_timing("stage.precursor"):
            run_precursor(ds, cfg)

    if stage in ("fragments", "both") and fragments_enabled:
        with log_timing("stage.fragments"):
            run_fragments(ds, cfg)
    elif stage in ("fragments", "both") and not fragments_enabled:
        log("[skip] fragments disabled (opt-in required)")

    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())