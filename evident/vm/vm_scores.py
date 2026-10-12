"""claims/scores.yaml: DIA-NN q-value distribution, FragPipe score distributions and the FragPipe direction rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vm_common import REAL_RUNS, complexity_of, dump, write_sidecar
from vm_direction import classify
from vm_marginals import pairwise
from vm_discrepancy import w1_equal_n

Q_FLOOR = 1e-10  # log10 of a reported q of exactly 0
FP_SCORES = {"hyperscore": "Hyperscore", "spectral_sim": "SpectralSim", "rt_score": "RTScore",
             "im_score": "IMScore", "log10_expectation": "Expectation", "log10_qvalue": "Qvalue"}


def fp_exp(run: str) -> str:
    return run.replace("-", "_")


def auc(target: np.ndarray, decoy: np.ndarray) -> float:
    """P(target score > decoy score), ties counted half (Mann-Whitney)."""
    allv = np.concatenate([target, decoy])
    ranks = pd.Series(allv).rank().to_numpy()
    rt = ranks[: len(target)].sum()
    return float((rt - len(target) * (len(target) + 1) / 2) / (len(target) * len(decoy)))


def fragpipe_run(arm_dir: Path, exp: str) -> dict:
    psm = pd.read_csv(arm_dir / exp / "psm.tsv", sep="\t")
    vals = {}
    for k, col in FP_SCORES.items():
        x = psm[col].astype(float).to_numpy()
        vals[k] = np.log10(np.clip(x, Q_FLOOR, None)) if k.startswith("log10") else x
        vals[k] = vals[k][np.isfinite(vals[k])]
    tgt = [p for p in arm_dir.rglob("*percolator_target_psms.tsv") if p.name.startswith(exp)]
    dec = [p for p in arm_dir.rglob("*percolator_decoy_psms.tsv") if p.name.startswith(exp)]
    if len(tgt) != 1 or len(dec) != 1:
        raise FileNotFoundError(f"{arm_dir}/{exp}: expected one Percolator target and decoy table, got {len(tgt)}/{len(dec)}")
    t = pd.read_csv(tgt[0], sep="\t", usecols=["score"])["score"].to_numpy()
    d = pd.read_csv(dec[0], sep="\t", usecols=["score"])["score"].to_numpy()
    direction = {"percolator_auc": auc(t, d), "median_hyperscore": float(np.median(vals["hyperscore"])),
                 "median_spectral_sim": float(np.median(vals["spectral_sim"])),
                 "median_log10_expectation": float(np.median(vals["log10_expectation"]))}
    return {"values": vals, "direction": direction, "n_psm": int(len(psm)), "n_target": int(len(t)), "n_decoy": int(len(d))}


FP_DIRECTION = {"percolator_auc": "higher", "median_hyperscore": "higher",
                "median_spectral_sim": "higher", "median_log10_expectation": "lower"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--fragpipe", type=Path, required=True, help="searches/vm/fragpipe-v1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    m = json.loads((a.out / "matched_complexity.json").read_text())
    sim_runs = m["matched_runs"]
    res, side_vals = {}, {}

    # DIA-NN: log10 Q.Value of reported targets.
    q = {}
    for side, runs in (("real10", REAL_RUNS), ("sim", sim_runs)):
        for r in runs:
            d = pd.read_parquet(a.recipe / side / r / "plain" / "report.parquet", columns=["Q.Value", "Decoy"])
            q[r] = np.log10(np.clip(d.loc[d["Decoy"] == 0, "Q.Value"].to_numpy(), Q_FLOOR, None))
    res["diann_qvalue"] = pairwise(q, REAL_RUNS, sim_runs, lambda x, y: w1_equal_n(x, y, rng))
    side_vals["vm.scores.qvalue.diann"] = {"ratio": res["diann_qvalue"]["ratio"]}

    # FragPipe: per-run score marginals and direction features.
    arm = f"sim{complexity_of(sim_runs[0]) // 1000}k"
    runs = {r: fragpipe_run(a.fragpipe / "real10", r) for r in REAL_RUNS}
    runs |= {r: fragpipe_run(a.fragpipe / arm, fp_exp(r)) for r in sim_runs}
    res["fragpipe"] = {}
    for k in FP_SCORES:
        res["fragpipe"][k] = pairwise({r: v["values"][k] for r, v in runs.items()}, REAL_RUNS, sim_runs,
                                      lambda x, y: w1_equal_n(x, y, rng))
        side_vals[f"vm.scores.{k}.fragpipe"] = {"ratio": res["fragpipe"][k]["ratio"]}
    feats = {}
    for f, easier in FP_DIRECTION.items():
        real = [runs[r]["direction"][f] for r in REAL_RUNS]
        sim = float(np.median([runs[r]["direction"][f] for r in sim_runs]))
        feats[f] = {"easier_is": easier, "sim": sim, "real_min": min(real), "real_max": max(real), "class": classify(sim, real, easier)}
    n_h = sum(v["class"] == "harder" for v in feats.values())
    n_e = sum(v["class"] == "easier" for v in feats.values())
    res["fragpipe_direction"] = {"features": feats, "n_harder": n_h, "n_easier": n_e}
    res["fragpipe_counts"] = {r: {k: v[k] for k in ("n_psm", "n_target", "n_decoy")} for r, v in runs.items()}
    side_vals["vm.direction.scores.sim-not-harder.fragpipe"] = {"n_harder": n_h}
    side_vals["vm.direction.scores.sim-not-easier.fragpipe"] = {"n_easier": n_e}
    dump(res, a.out / "results" / "scores.json")
    write_sidecar(a.out, side_vals)
    for k, v in side_vals.items():
        print(k, v)


if __name__ == "__main__":
    main()
