# imspy_pipeline_cli.py
from __future__ import annotations
from .helpers import *

import argparse
import os
import sys
import datetime as dt
from dataclasses import replace
from typing import Any, Dict, List, Optional

import tomllib  # Python 3.11+

from imspy.timstof.dia import TimsDatasetDIA

def _load_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)

def _merge_dataclass(dc, overrides: Dict[str, Any]):
    # Ignore unknown keys; only replace fields that exist on the dataclass
    fields = {f.name for f in dc.__dataclass_fields__.values()}  # type: ignore
    filtered = {k: v for k, v in overrides.items() if k in fields}
    return replace(dc, **filtered)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")

def _dataset_from_config(cfg_ds: Dict[str, Any]) -> TimsDatasetDIA:
    path = cfg_ds.get("path")
    if not path:
        raise ValueError("[dataset] must set 'path'")
    use_bruker_sdk = bool(cfg_ds.get("use_bruker_sdk", False))
    # Optional: pass additional kwargs transparently if your constructor supports them
    return TimsDatasetDIA(path, use_bruker_sdk=use_bruker_sdk)

def _build_param_objects(cfg: Dict[str, Any]):
    planning = PlanningParams()
    if "planning" in cfg:
        planning = _merge_dataclass(planning, cfg["planning"])

    picking = PeakPickingParams()
    if "picking" in cfg:
        picking = _merge_dataclass(picking, cfg["picking"])

    stitch = StitchParams()
    if "stitch" in cfg:
        stitch = _merge_dataclass(stitch, cfg["stitch"])

    cluster = ClusterParams()
    if "cluster" in cfg:
        cluster = _merge_dataclass(cluster, cfg["cluster"])

    return planning, picking, stitch, cluster

def _run_one(config: Dict[str, Any], outdir: str, run_name: Optional[str] = None) -> str:
    """
    Executes a single run; returns path to the written Parquet file.
    """
    ds_cfg = config.get("dataset", {})
    ds = _dataset_from_config(ds_cfg)

    planning, picking, stitch, cluster = _build_param_objects(config)

    mode = config.get("mode", "ms1").lower()
    if mode not in ("ms1", "ms2"):
        raise ValueError("[mode] must be 'ms1' or 'ms2'")

    _ensure_dir(outdir)

    base = run_name or config.get("name") or f"run-{_ts()}"
    base = base.replace(" ", "_")

    if mode == "ms1":
        clusters, df = run_ms1(ds, planning, picking, stitch, cluster)
        outfile = os.path.join(outdir, f"{base}.ms1.parquet")
        df.to_parquet(outfile, index=False)
        return outfile

    # ms2
    clusters, df = run_ms2(ds, planning, picking, stitch, cluster)
    outfile = os.path.join(outdir, f"{base}.ms2.parquet")
    df.to_parquet(outfile, index=False)
    return outfile

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="imspy-pipeline",
        description="Run timsTOF IM-peak→stitch→cluster pipelines (MS1 or MS2) from TOML configs."
    )
    ap.add_argument("-c", "--config", required=True, help="Path to TOML config.")
    ap.add_argument("-o", "--outdir", default="./out", help="Output directory (Parquet files).")
    ap.add_argument("--list", action="store_true", help="List runs found in config and exit.")
    ap.add_argument("--run", type=str, default=None, help="Run name/index to execute (for multi-run configs).")
    args = ap.parse_args(argv)

    cfg = _load_toml(args.config)

    # Two styles supported:
    # 1) Single run config with top-level [dataset], [planning], [picking], [stitch], [cluster], mode="ms1|ms2"
    # 2) Multi-run config with [[runs]] blocks, each containing the same structure.
    runs = cfg.get("runs", None)

    if runs is None:
        # single-run file
        if args.list:
            print("Single-run config:")
            print("  name:", cfg.get("name", "(none)"))
            print("  mode:", cfg.get("mode", "ms1"))
            print("  dataset.path:", cfg.get("dataset", {}).get("path"))
            return 0
        out = _run_one(cfg, args.outdir, cfg.get("name"))
        print(out)
        return 0

    # multi-run
    if not isinstance(runs, list) or len(runs) == 0:
        print("ERROR: [[runs]] must be a non-empty array in the TOML.", file=sys.stderr)
        return 2

    # list mode
    if args.list:
        print("Runs in config:")
        for i, r in enumerate(runs):
            rname = r.get("name", f"runs[{i}]")
            mode = r.get("mode", "ms1")
            ds_path = r.get("dataset", {}).get("path")
            print(f"  - {rname}  (mode={mode}, dataset={ds_path})")
        return 0

    # choose which run(s)
    selected: List[Dict[str, Any]] = []
    if args.run is None:
        selected = runs
    else:
        # allow index or name
        try:
            idx = int(args.run)
            selected = [runs[idx]]
        except ValueError:
            # name match
            matched = [r for r in runs if r.get("name") == args.run]
            if not matched:
                print(f"ERROR: No run with name '{args.run}'", file=sys.stderr)
                return 3
            selected = matched

    # execute
    paths = []
    for i, r in enumerate(selected):
        rname = r.get("name", f"runs[{i}]")
        out = _run_one(r, args.outdir, rname)
        print(out)
        paths.append(out)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())