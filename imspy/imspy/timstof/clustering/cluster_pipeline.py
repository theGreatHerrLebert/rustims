#!/usr/bin/env python3
import argparse
import os
import sys
import gc
from pathlib import Path

import numpy as np
import torch

# Python 3.11+ has tomllib; otherwise optional tomli fallback
try:
    import tomllib as toml
except Exception as e:
    print("Warning: tomllib not found, using default toml")
    import tomli as toml  # type: ignore

from imspy.timstof.clustering.torch_extractor import iter_im_peaks_batches
from imspy.timstof.dia import (
    stitch_im_peaks_flat,
    save_clusters_bin,
    TimsDatasetDIA,
)

# ---------- small helpers -----------------------------------------------------
def ensure_dir_for_file(path: str | os.PathLike) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

def cuda_gc() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def log(msg: str) -> None:
    print(msg, flush=True)

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

    clusters = ds.clusters_for_precursor(
        stitched,
        ppm_per_bin=15.0,
        bin_pad=30.0,
        min_prom=50,
        attach_raw_data=attach_raw,
        attach_axes=attach_raw,
        attach_max_points=5000,
    )

    # ---- binary save ----
    out_bin = Path(cfg["output"]["dir"]) / cfg["output"]["precursor_file"]
    ensure_dir_for_file(out_bin)
    save_clusters_bin(clusters=clusters, path=str(out_bin), compress=True)
    log(f"[ok] wrote precursor clusters -> {out_bin}")

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
            ppm_per_bin=15.0,
            bin_pad=30.0,
            min_prom=25,
            attach_raw_data=attach_raw,
            attach_axes=attach_raw,
            attach_max_points=5000,
        )
        all_clusters.extend(clusters_wg)

        del stitched_wg, clusters_wg
        cuda_gc()

    # ---- binary save ----
    out_bin = Path(cfg["output"]["dir"]) / cfg["output"]["fragment_file"]
    ensure_dir_for_file(out_bin)
    save_clusters_bin(clusters=all_clusters, path=str(out_bin), compress=True)
    log(f"[ok] wrote fragment clusters -> {out_bin}")

    # ---- optional parquet ----
    if bool(cfg["output"].get("parquet_enabled", False)):
        out_parq = Path(cfg["output"]["dir"]) / cfg["output"]["fragment_parquet"]
        ensure_dir_for_file(out_parq)
        df = clusters_to_dataframe(all_clusters)
        df.to_parquet(out_parq, index=False)
        log(f"[ok] wrote fragment parquet -> {out_parq}")

    del all_clusters
    cuda_gc()

# ---------- CLI ---------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        cfg = toml.load(f)

    # light validation / defaults
    for section in ("dataset", "output", "run", "detector", "stitch", "plans"):
        if section not in cfg:
            raise ValueError(f"Missing [{section}] in config.")

    # defaults
    cfg["run"].setdefault("stage", "both")
    cfg["run"].setdefault("device", "cuda")
    cfg["run"].setdefault("batch_size", 64)
    cfg["run"].setdefault("precompute_views", True)
    cfg["run"].setdefault("fragments_enabled", False)  # <-- default OFF (opt-in)
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

    return cfg

def main(argv=None):
    parser = argparse.ArgumentParser(description="IM peak detection → stitching → clustering (TOML-configured).")
    parser.add_argument("-c", "--config", required=True, help="Path to config.toml")
    parser.add_argument("--stage", choices=["precursor", "fragments", "both"],
                        help="Override run.stage from config")
    parser.add_argument("--device", help="Override run.device (e.g., 'cuda', 'cpu')")
    # --- NEW: explicit opt-in/out for fragments ---
    parser.add_argument("--fragments", dest="fragments", action="store_true",
                        help="Opt-in to fragment processing (overrides config)")
    parser.add_argument("--no-fragments", dest="fragments", action="store_false",
                        help="Opt-out of fragment processing (overrides config)")
    parser.set_defaults(fragments=None)

    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.stage:
        cfg["run"]["stage"] = args.stage
    if args.device:
        cfg["run"]["device"] = args.device
    if args.fragments is not None:
        cfg["run"]["fragments_enabled"] = bool(args.fragments)

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