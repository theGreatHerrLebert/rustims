"""Descriptive fit of TimSim's local RT model and Chronologer to real LS10 (evident/vm/stress_settings.yaml).

Reported without tolerance; it chooses nothing. Runs in the timsim container with the pinned Chronologer on
PYTHONPATH. Never touches a sim or an FDP.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vm_stress_prepare import chronologer_hi

N_FIT = 5000
SEED = 2026
ALLOWED_MODS = re.compile(r"^(?:\(UniMod:1\))?(?:[A-Z]|C\(UniMod:4\)|M\(UniMod:35\))+$")


def robust_sd(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def fit_metric(pred: np.ndarray, obs: np.ndarray) -> float:
    coef = np.polyfit(pred, obs, 3)
    return robust_sd(obs - np.polyval(coef, pred))


def to_unimod(s: str) -> str:
    return re.sub(r"\(UniMod:(\d+)\)", r"[UNIMOD:\1]", s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--chronologer-model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from imspy_predictors.rt.predictors import DeepChromatographyApex, load_deep_retention_time_predictor
    from imspy_predictors.utility import load_tokenizer_from_resources

    d = pd.read_parquet(a.recipe / "real10" / "LS10" / "plain" / "report.parquet",
                        columns=["Modified.Sequence", "Precursor.Charge", "RT", "Q.Value", "Decoy"])
    d = d[(d["Decoy"] == 0) & (d["Q.Value"] <= 0.01)].drop_duplicates("Modified.Sequence")
    d = d[d["Modified.Sequence"].str.match(ALLOWED_MODS)].sample(n=N_FIT, random_state=SEED)
    seqs = d["Modified.Sequence"].map(to_unimod).tolist()
    obs = d["RT"].to_numpy() * 60.0

    local = DeepChromatographyApex(model=load_deep_retention_time_predictor(),
                                   tokenizer=load_tokenizer_from_resources(tokenizer_name="tokenizer-ptm"))
    pl = local.simulate_separation_times_pandas(pd.DataFrame({"sequence": seqs, "decoy": False}), gradient_length=7202)
    pred = {"timsim_local": pl.filter(like="retention_time").iloc[:, 0].to_numpy(float),
            "chronologer": chronologer_hi(seqs, a.chronologer_model)}
    res = {"n_fit": int(len(d)), "unit": "s", "robust_sd": {k: fit_metric(v, obs) for k, v in pred.items()},
           "spearman_vs_observed": {k: float(pd.Series(v).corr(pd.Series(obs), method="spearman")) for k, v in pred.items()}}
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
