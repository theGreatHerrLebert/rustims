"""vm.direction.*: is the sim easier or harder than real, feature by feature (evident/claims/direction.yaml).

Reads the per-run quantiles that vm_discrepancy.py stores, so it needs no report access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vm_common import REAL_RUNS, dump, write_sidecar

# feature -> (statistic, easier direction). Spread is q95 - q05.
FEATURES = {
    "ms1_ppm": ("spread", "lower"),
    "fr_ppm": ("spread", "lower"),
    "rt_resid": ("spread", "lower"),
    "im_resid": ("spread", "lower"),
    "fwhm": ("q50", "lower"),
    "rt_width": ("q50", "lower"),
    "evidence": ("q50", "higher"),
    "ms1_profile_corr": ("q50", "higher"),
    "mass_evidence": ("q50", "higher"),
    "pep": ("q95", "lower"),
}


def stat(t: dict, kind: str) -> float:
    return t["q95"] - t["q05"] if kind == "spread" else t[kind]


def classify(sim: float, real: list[float], easier: str) -> str:
    lo, hi = min(real), max(real)
    if lo <= sim <= hi:
        return "comparable"
    below = sim < lo
    return "easier" if below == (easier == "lower") else "harder"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="analysis-v1 directory holding results/discrepancy.json")
    a = ap.parse_args()
    disc = json.loads((a.out / "results" / "discrepancy.json").read_text())
    matched = json.loads((a.out / "matched_complexity.json").read_text())
    view = disc["by_complexity"][str(matched["matched_complexity"])]["all"]
    per = {}
    for f, (kind, easier) in FEATURES.items():
        tails = view[f]["tails"]
        real = [stat(tails[r], kind) for r in REAL_RUNS]
        sim = float(np.median([stat(tails[r], kind) for r in matched["matched_runs"]]))
        if not all(np.isfinite([*real, sim])):
            raise ValueError(f"non-finite statistic for {f}")
        per[f] = {"statistic": kind, "easier_is": easier, "sim": sim, "real_min": min(real),
                  "real_max": max(real), "class": classify(sim, real, easier)}
    n_h = sum(v["class"] == "harder" for v in per.values())
    n_e = sum(v["class"] == "easier" for v in per.values())
    dump({"matched_complexity": matched["matched_complexity"], "features": per,
          "n_harder": n_h, "n_easier": n_e}, a.out / "results" / "direction.json")
    write_sidecar(a.out, {"vm.direction.sim-not-harder.diann": {"n_harder": n_h},
                          "vm.direction.sim-not-easier.diann": {"n_easier": n_e}})
    for f, v in per.items():
        print(f"{f:18s} {v['class']:11s} sim {v['sim']:.4g}  real [{v['real_min']:.4g}, {v['real_max']:.4g}]")
    print(f"harder {n_h}, easier {n_e}")


if __name__ == "__main__":
    main()
