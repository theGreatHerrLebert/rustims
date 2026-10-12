"""claims/marginals.yaml: RT, IM and m/z marginals, view (ii) identified and view (i) all MS1 ions.

View (iii), the simulated ions, is carried into the output for the figure and has no claim.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vm_common import REAL_RUNS, Q_MAX, dump, write_sidecar
from vm_discrepancy import ratio, w1_equal_n

IDENTIFIED = {"rt": "RT", "im": "IM", "mz": "Precursor.Mz"}


def w1_binned(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> float:
    """W1 between two histograms on the same bins, each normalised to unit mass."""
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("empty histogram")
    ca, cb = np.cumsum(a / a.sum()), np.cumsum(b / b.sum())
    return float(np.sum(np.abs(ca - cb)[:-1] * np.diff(edges)[:-1]))


def pairwise(values: dict, real: list[str], sim: list[str], dist) -> dict:
    rr = {f"{a}|{b}": dist(values[a], values[b]) for a, b in itertools.combinations(real, 2)}
    sr = {f"{s}|{r}": dist(values[s], values[r]) for s in sim for r in real}
    rr_max, sr_med = max(rr.values()), float(np.median(list(sr.values())))
    return {"real_real": rr, "sim_real": sr, "real_real_max": rr_max, "sim_real_median": sr_med, "ratio": ratio(sr_med, rr_max)}


def identified(recipe: Path, sim_runs: list[str], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    frames = {}
    for side, runs in (("real10", REAL_RUNS), ("sim", sim_runs)):
        for r in runs:
            d = pd.read_parquet(recipe / side / r / "plain" / "report.parquet", columns=[*IDENTIFIED.values(), "Q.Value", "Decoy"])
            frames[r] = d[(d["Decoy"] == 0) & (d["Q.Value"] <= Q_MAX)]
    out = {}
    for dim, col in IDENTIFIED.items():
        vals = {r: f[col][np.isfinite(f[col])].to_numpy() for r, f in frames.items()}
        out[dim] = pairwise(vals, REAL_RUNS, sim_runs, lambda a, b: w1_equal_n(a, b, rng))
    return out


def allions(raw: Path, sim_runs: list[str]) -> tuple[dict, dict]:
    docs = {r: json.loads((raw / "real10" / f"{r}.json").read_text()) for r in REAL_RUNS}
    docs |= {r: json.loads((raw / "sim" / f"{r}.json").read_text()) for r in sim_runs}
    edges = {k: np.asarray(v) for k, v in docs[REAL_RUNS[0]]["bins"].items()}
    out = {}
    for dim in ("rt", "im", "mz"):
        h = {r: np.asarray(d["raw"]["hist"][dim]) for r, d in docs.items()}
        out[dim] = pairwise(h, REAL_RUNS, sim_runs, lambda a, b, e=edges[dim]: w1_binned(a, b, e))
        out[dim]["out_of_range_share"] = {r: d["raw"]["out_of_range_intensity"][dim] / d["raw"]["total_intensity"] for r, d in docs.items()}
    figure = {"bins": {k: v.tolist() for k, v in edges.items()},
              "raw": {r: d["raw"]["hist"] for r, d in docs.items()},
              "truth": {r: docs[r]["truth"] for r in sim_runs}}
    return out, figure


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True, help="searches/vm/raw-v1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    sim_runs = json.loads((a.out / "matched_complexity.json").read_text())["matched_runs"]
    ident = identified(a.recipe, sim_runs, a.seed)
    raw, figure = allions(a.raw, sim_runs)
    dump({"identified": ident, "allions": raw}, a.out / "results" / "marginals.json")
    dump(figure, a.out / "results" / "marginals_figure.json")
    vals = {f"vm.marginal.{d}.identified.diann": {"ratio": v["ratio"]} for d, v in ident.items()}
    vals |= {f"vm.marginal.{d}.allions.raw": {"ratio": v["ratio"]} for d, v in raw.items()}
    write_sidecar(a.out, vals)
    for k, v in vals.items():
        print(f"{k:38s} {v['ratio']:.2f}")


if __name__ == "__main__":
    main()
