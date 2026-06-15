"""Multi-vendor benchmark *test grid*: expand one shared base config + per-vendor
cells into matched per-vendor TimSim configs, and emit the manifest the
``multi_vendor`` consolidator consumes.

The point of the grid is a fair, reproducible cross-vendor comparison: every cell
shares the bulk of the configuration (FASTA, peptide counts, m/z noise) so the only
differences are the instrument and its acquisition source. A grid spec (JSON)::

    {
      "save_root": "/tmp/grid",
      "base": {                       # shared across every cell
        "fasta_path": "/data/human.fasta",
        "acquisition_type": "DIA",
        "num_peptides_total": 30000,
        "num_sample_peptides": 5000,
        "num_threads": 6,
        "apply_fragmentation": true
      },
      "cells": [
        {"label": "bruker", "engine": "diann",
         "config": {"instrument": "bruker_timstof", "reference_path": "/data/ref.d",
                    "gradient_length": 1800}},
        {"label": "thermo", "engine": "alphadia",
         "config": {"instrument": "orbitrap_exploris", "template_path": "/data/tmpl.raw",
                    "intensity_model": "prosit_hcd", "gradient_length": 2700}},
        {"label": "sciex", "engine": "alphadia",
         "config": {"instrument": "sciex_zenotof", "template_path": "/data/m.wiff",
                    "intensity_model": "prosit_hcd", "gradient_length": 1800}},
        {"label": "waters", "engine": "diann",
         "config": {"instrument": "waters_synapt_xs", "intensity_model": "prosit_hcd",
                    "gradient_length": 1800, "waters_cycle_time_s": 1.5}}
      ]
    }

``expand`` merges ``base`` + each cell's ``config`` into a full TimSim config (adding
``save_path = <save_root>/<label>`` and ``experiment_name``). ``write_configs`` writes
``<label>.toml`` per cell. ``manifest_for`` produces the ``multi_vendor`` manifest whose
each cell's ``eval_json`` points at ``<save_root>/<label>/eval.json`` — the JSON you
produce by running, per cell::

    timsim <out_dir>/<label>.toml                          # sim -> output + synthetic_data.db
    <engine> on the output                                 # DiaNN (.d/.mzML) | alphaDIA (.raw/.mzML)
    python -m imspy_simulation.timsim.groundtruth_eval \\
        --db <save_root>/<label>/<EXP>/synthetic_data.db --report <report> \\
        --engine <engine> --precursor-fdp --out <save_root>/<label>/eval.json

Then ``python -m imspy_simulation.timsim.benchmark.multi_vendor --manifest <manifest>``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import toml


def _drop_none(d: dict) -> dict:
    """TOML has no null; drop keys whose value is None (e.g. an unset window_step),
    recursing into nested tables so a None buried in a sub-table can't reach toml.dump."""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        out[k] = _drop_none(v) if isinstance(v, dict) else v
    return out


def expand(spec: dict) -> dict[str, dict]:
    """Merge ``base`` + each cell's ``config`` into a full per-cell TimSim config.

    Adds ``save_path = <save_root>/<label>`` and a derived ``experiment_name`` unless
    the cell sets its own. Cell config overrides base on key collision.
    """
    save_root = spec.get("save_root", "grid")
    base = spec.get("base", {})
    cells = spec.get("cells", [])
    if not cells:
        raise ValueError("grid spec has no 'cells'")
    out: dict[str, dict] = {}
    seen = set()
    for cell in cells:
        label = cell.get("label")
        if not label:
            raise ValueError(f"cell missing 'label': {cell!r}")
        if label in seen:
            raise ValueError(f"duplicate cell label {label!r}")
        seen.add(label)
        cfg = dict(base)
        cfg.update(cell.get("config", {}))
        if "instrument" not in cfg:
            raise ValueError(f"cell {label!r} has no 'instrument' (in cell.config or base)")
        cfg.setdefault("save_path", os.path.join(save_root, label))
        cfg.setdefault("experiment_name", label.upper().replace("_", "-"))
        out[label] = _drop_none(cfg)
    # Two cells writing to the same save_path would clobber each other's output (can
    # happen if cells set/override save_path, or labels collide post-normalisation).
    by_save_path: dict[str, str] = {}
    for label, cfg in out.items():
        sp = cfg["save_path"]
        if sp in by_save_path:
            raise ValueError(
                f"cells {by_save_path[sp]!r} and {label!r} resolve to the same "
                f"save_path {sp!r}; give them distinct save_path / labels"
            )
        by_save_path[sp] = label
    return out


def write_configs(expanded: dict[str, dict], out_dir: str) -> dict[str, str]:
    """Write one ``<label>.toml`` per cell into ``out_dir``; return {label: path}."""
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}
    for label, cfg in expanded.items():
        path = os.path.join(out_dir, f"{label}.toml")
        with open(path, "w") as fh:
            toml.dump(cfg, fh)
        paths[label] = path
    return paths


def manifest_for(spec: dict, eval_filename: str = "eval.json") -> dict:
    """Build the ``multi_vendor`` manifest for this grid.

    Derived from ``expand(spec)`` so it shares the same validation (missing instrument,
    duplicate labels, save_path collisions) and the SAME resolved ``save_path`` as the
    written configs — each cell's ``eval_json`` is ``<resolved save_path>/<eval_filename>``,
    so the manifest can't drift from where the run actually wrote its output. ``q`` carries
    through from the spec.
    """
    expanded = expand(spec)
    cell_by_label = {c["label"]: c for c in spec.get("cells", [])}
    vendors = []
    for label, cfg in expanded.items():
        cell = cell_by_label[label]
        vendors.append({
            "instrument": cfg["instrument"],
            "engine": cell.get("engine", "?"),
            "eval_json": os.path.join(cfg["save_path"], eval_filename),
            "source": cell.get("source", "live"),
        })
    return {"q": spec.get("q", 0.01), "vendors": vendors}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, help="path to the grid spec JSON")
    ap.add_argument("--out-dir", required=True, help="directory to write the per-cell TOML configs")
    ap.add_argument("--write-manifest", help="optional path to write the multi_vendor manifest JSON")
    args = ap.parse_args(argv)

    with open(args.spec) as fh:
        spec = json.load(fh)
    expanded = expand(spec)
    paths = write_configs(expanded, args.out_dir)
    print(f"wrote {len(paths)} grid configs to {args.out_dir}:")
    for label, p in paths.items():
        print(f"  {label:10s} -> {p}  (instrument={expanded[label]['instrument']})")
    if args.write_manifest:
        manifest = manifest_for(spec)
        with open(args.write_manifest, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"wrote consolidation manifest -> {args.write_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
