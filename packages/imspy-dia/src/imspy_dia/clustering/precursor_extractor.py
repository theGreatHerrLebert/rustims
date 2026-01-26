# imspy_pipeline_cli.py
from __future__ import annotations

from .helpers import *               # expects run_ms1, run_ms2, etc.
from .cluster_report import *        # expects save_run_report, save_sweep_index

import argparse
import sys
import datetime as dt
from dataclasses import replace
from typing import Any, Dict, List, Optional

import tomllib  # Python 3.11+

from pathlib import Path
from imspy_core.timstof.dia import TimsDatasetDIA


# ------------------------- arg parsing -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="imspy-pipeline",
        description="Run timsTOF IM-peak→stitch→cluster pipelines (MS1 or MS2) from TOML configs."
    )
    p.add_argument("-c", "--config", required=True, help="Path to TOML config.")
    p.add_argument("-o", "--outdir", default="./out", help="Output directory.")
    p.add_argument("--list", action="store_true", help="List runs found in config and exit.")
    p.add_argument("--run", type=str, default=None,
                   help="Run name or index to execute (for multi-run configs).")
    # reporting
    p.add_argument("--report", action="store_true",
                   help="Generate HTML/PNG reports per run (report.html in each run dir).")
    p.add_argument("--report-title", default=None,
                   help="Optional report title for single-run; per-run name is used otherwise.")
    return p


# ------------------------- helpers -------------------------

def _load_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)

def _merge_dataclass(dc, overrides: Dict[str, Any]):
    # Ignore unknown keys; only replace fields that exist on the dataclass
    fields = {f.name for f in dc.__dataclass_fields__.values()}  # type: ignore
    filtered = {k: v for k, v in overrides.items() if k in fields}
    return replace(dc, **filtered)

def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

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


# ------------------------- runner -------------------------

def _run_one(
    config: Dict[str, Any],
    outdir: str | Path,
    run_name: Optional[str] = None,
    *,
    do_report: bool = False,
    report_title: Optional[str] = None,
) -> str:
    """
    Executes a single run.

    Returns:
        Path to the legacy top-level Parquet file (string).
    Side-effects:
        - Writes a per-run subdir with clusters.parquet, summary.json, report.html (if --report).
        - Still writes the legacy top-level parquet: <outdir>/<name>.(ms1|ms2).parquet
    """
    ds_cfg = config.get("dataset", {})
    ds = _dataset_from_config(ds_cfg)

    planning, picking, stitch, cluster = _build_param_objects(config)

    mode = str(config.get("mode", "ms1")).lower()
    if mode not in ("ms1", "ms2"):
        raise ValueError("[mode] must be 'ms1' or 'ms2'")

    _ensure_dir(outdir)
    outdir = Path(outdir)

    base = (run_name or config.get("name") or f"run-{_ts()}").replace(" ", "_")

    # Per-run directory for reports/artifacts
    run_dir = outdir / base
    _ensure_dir(run_dir)

    # Execute pipeline
    if mode == "ms1":
        clusters, df = run_ms1(ds, planning, picking, stitch, cluster)
        legacy_parquet = outdir / f"{base}.ms1.parquet"
    else:
        clusters, df = run_ms2(ds, planning, picking, stitch, cluster)
        legacy_parquet = outdir / f"{base}.ms2.parquet"

    # Write artifacts
    # 1) Legacy parquet (keep old behavior)
    df.to_parquet(legacy_parquet, index=False)

    # 2) Run-local parquet
    (run_dir / "clusters.parquet").write_bytes(legacy_parquet.read_bytes())

    # 3) Optional report
    if do_report:
        title = report_title or f"{base} ({mode.upper()})"
        save_run_report(
            df,
            out_dir=run_dir,  # write report next to clusters.parquet
            title=title,
            mode=mode,
            extra_meta={"run": base, "dataset": getattr(ds, "path", None)},
            ds=ds,
            slice_extractor=get_slice_filtered_cluster,  # your function
            n_examples=8,
            example_mz_pad=2.0,
        )

    return str(legacy_parquet)


# ------------------------- CLI main -------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    cfg = _load_toml(args.config)
    out_root = Path(args.outdir)
    _ensure_dir(out_root)

    # Support two styles:
    # 1) Single run config with top-level [dataset], [planning], [picking], [stitch], [cluster], mode="ms1|ms2"
    # 2) Multi-run config with [[runs]] blocks, each containing the same structure.
    runs = cfg.get("runs", None)

    # Single-run path
    if runs is None:
        if args.list:
            print("Single-run config:")
            print("  name:", cfg.get("name", "(none)"))
            print("  mode:", cfg.get("mode", "ms1"))
            print("  dataset.path:", cfg.get("dataset", {}).get("path"))
            return 0

        out = _run_one(
            cfg,
            out_root,
            cfg.get("name"),
            do_report=args.report,
            report_title=args.report_title,
        )
        print(out)
        return 0

    # Multi-run path
    if not isinstance(runs, list) or len(runs) == 0:
        print("ERROR: [[runs]] must be a non-empty array in the TOML.", file=sys.stderr)
        return 2

    if args.list:
        print("Runs in config:")
        for i, r in enumerate(runs):
            rname = r.get("name", f"runs[{i}]")
            mode = r.get("mode", "ms1")
            ds_path = r.get("dataset", {}).get("path")
            print(f"  - {rname}  (mode={mode}, dataset={ds_path})")
        return 0

    # choose which run(s)
    if args.run is None:
        selected: List[Dict[str, Any]] = runs
    else:
        try:
            idx = int(args.run)
            selected = [runs[idx]]
        except ValueError:
            matched = [r for r in runs if r.get("name") == args.run]
            if not matched:
                print(f"ERROR: No run with name '{args.run}'", file=sys.stderr)
                return 3
            selected = matched

    # execute
    paths = []
    for i, r in enumerate(selected):
        rname = r.get("name", f"runs[{i}]")
        out = _run_one(
            r,
            out_root,
            rname,
            do_report=args.report,
            report_title=None,  # per-run name already in title
        )
        print(out)
        paths.append(out)

    # Build an index if reports were requested
    if args.report:
        save_sweep_index(out_root)
        print((out_root / "index.html").as_posix())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())