"""Section-3 discrepancy: sim vs real DIA-NN observables, read against real-vs-real replicate distance.

Uses plain-library reports only and never computes FDP. Views:
  all     every accepted precursor at run-level Q.Value <= 0.01
  shared  precursors accepted on both sides (per-precursor median, equal weight), with coverage reported
The raw-signal panel (view 3) needs the .d files and lives elsewhere.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vm_common import Q_MAX, check_cohort, complexity_of, dump, mod_key

SUBSAMPLE_REPS = 20
MAD_K = 3.0
MZ_BINS = [-np.inf, 300, 500, 700, 900, 1200, 1800, np.inf]  # outer bins catch out-of-range precursors
CHARGES = [2, 3, 4]

# name -> (function of the report frame, plain-words definition)
OBSERVABLES = {
    "ms1_ppm": (lambda d: d["Ms1.Apex.Mz.Delta"] / d["Precursor.Mz"] * 1e6, "MS1 apex m/z delta, ppm (Th / m/z)"),
    "fr_ppm": (lambda d: d["Best.Fr.Mz.Delta"] / d["Best.Fr.Mz"] * 1e6, "best fragment m/z delta, ppm"),
    "fwhm": (lambda d: d["FWHM"], "elution peak FWHM, min"),
    "rt_width": (lambda d: d["RT.Stop"] - d["RT.Start"], "RT.Stop - RT.Start, min"),
    "rt_resid": (lambda d: d["RT"] - d["Predicted.RT"], "RT - Predicted.RT after DIA-NN alignment, min"),
    "im_resid": (lambda d: d["IM"] - d["Predicted.IM"], "IM - Predicted.IM after DIA-NN calibration, 1/K0"),
    "log10_ms1_area": (lambda d: np.log10(d["Ms1.Area"].where(d["Ms1.Area"] > 0)), "log10 Ms1.Area"),
    "evidence": (lambda d: d["Evidence"], "DIA-NN Evidence (chromatographic, not spectral)"),
    "ms1_profile_corr": (lambda d: d["Ms1.Profile.Corr"], "MS1-MS2 elution agreement (not isotopes)"),
    "mass_evidence": (lambda d: d["Mass.Evidence"], "DIA-NN Mass.Evidence"),
    "pep": (lambda d: d["PEP"], "DIA-NN posterior error probability (score marginal)"),
}


def w1(a: np.ndarray, b: np.ndarray) -> float:
    """Exact 1-Wasserstein distance between two empirical distributions."""
    a, b = np.sort(a), np.sort(b)
    allv = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(a, allv[:-1], "right") / len(a)
    fb = np.searchsorted(b, allv[:-1], "right") / len(b)
    return float(np.sum(np.abs(fa - fb) * np.diff(allv)))


def w1_equal_n(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> float:
    n = min(len(a), len(b))
    return float(np.mean([w1(rng.choice(a, n, replace=False), rng.choice(b, n, replace=False)) for _ in range(SUBSAMPLE_REPS)]))


def tails(x: np.ndarray, centre: float, mad: float) -> dict:
    return {"q05": float(np.quantile(x, 0.05)), "q50": float(np.median(x)), "q95": float(np.quantile(x, 0.95)),
            "frac_beyond_kmad": float(np.mean(np.abs(x - centre) > MAD_K * mad)) if mad > 0 else float("nan"), "n": int(len(x))}


def ratio(num: float, den: float) -> float:
    if den > 0:
        return num / den
    return float("nan") if num == 0 else float("inf")


def tvd(p: pd.Series | None, q: pd.Series | None) -> float:
    """Total variation distance; a bin populated on one side only counts as fully different."""
    if p is None or q is None:
        return 0.0 if p is None and q is None else 1.0
    idx = p.index.union(q.index)
    return float(0.5 * np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum())


def load(report: Path) -> pd.DataFrame:
    d = pd.read_parquet(report)
    if "Decoy" in d.columns:
        d = d[d["Decoy"] == 0]
    d = d[d["Q.Value"] <= Q_MAX].copy()
    d["key"] = list(zip(d["Modified.Sequence"].map(mod_key), d["Precursor.Charge"].astype(int)))
    for name, (f, _) in OBSERVABLES.items():
        d[name] = f(d)
    return d.drop_duplicates("key")


def charge_profile(d: pd.DataFrame) -> dict:
    overall = d["Precursor.Charge"].value_counts(normalize=True)
    mzbin = pd.cut(d["Precursor.Mz"], MZ_BINS)
    by_bin = {str(b): g["Precursor.Charge"].value_counts(normalize=True) for b, g in d.groupby(mzbin, observed=True)}
    return {"overall": overall, "by_mz": by_bin, "bin_weights": mzbin.value_counts(normalize=True)}


def charge_distance(a: dict, b: dict, weights: pd.Series) -> dict:
    by = {k: tvd(a["by_mz"].get(k), b["by_mz"].get(k)) for k in set(a["by_mz"]) | set(b["by_mz"])}
    w = weights.rename(index=str).reindex(list(by)).fillna(0)
    return {"tvd_overall": tvd(a["overall"], b["overall"]),
            "tvd_by_mz_weighted": float(sum(by[k] * w[k] for k in by) / w.sum()) if w.sum() else float("nan")}


def compare(real: dict[str, pd.DataFrame], sim: dict[str, pd.DataFrame], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pooled_real = pd.concat(real.values())
    out = {}
    for name, (_, desc) in OBSERVABLES.items():
        vals = {k: v[name][np.isfinite(v[name])].to_numpy() for k, v in {**real, **sim}.items()}
        if any(len(x) == 0 for x in vals.values()):
            out[name] = {"description": desc, "skipped": "no finite values on at least one run"}
            continue
        pr = pooled_real[name][np.isfinite(pooled_real[name])].to_numpy()
        centre, mad = float(np.median(pr)), float(np.median(np.abs(pr - np.median(pr))))
        rr = {f"{a}|{b}": w1_equal_n(vals[a], vals[b], rng) for a, b in itertools.combinations(real, 2)}
        sr = {f"{s}|{r}": w1_equal_n(vals[s], vals[r], rng) for s in sim for r in real}
        rr_max = max(rr.values())
        sr_med = float(np.median(list(sr.values())))
        out[name] = {
            "description": desc, "w1_real_real": rr, "w1_sim_real": sr,
            "w1_real_real_max": rr_max, "w1_sim_real_median": sr_med,
            "ratio": ratio(sr_med, rr_max),
            "tails": {k: tails(v, centre, mad) for k, v in vals.items()},
        }
    profiles = {k: charge_profile(v) for k, v in {**real, **sim}.items()}
    weights = charge_profile(pooled_real)["bin_weights"]
    rr = {f"{a}|{b}": charge_distance(profiles[a], profiles[b], weights) for a, b in itertools.combinations(real, 2)}
    sr = {f"{s}|{r}": charge_distance(profiles[s], profiles[r], weights) for s in sim for r in real}
    rr_max = max(v["tvd_by_mz_weighted"] for v in rr.values())
    sr_med = float(np.median([v["tvd_by_mz_weighted"] for v in sr.values()]))
    out["charge"] = {"description": "charge-state fractions, TVD within fixed m/z bins weighted by pooled-real bin share",
                     "real_real": rr, "sim_real": sr, "tvd_real_real_max": rr_max, "tvd_sim_real_median": sr_med,
                     "ratio": ratio(sr_med, rr_max)}
    return out


def shared_panel(real: dict[str, pd.DataFrame], sim: dict[str, pd.DataFrame]) -> dict:
    r_all, s_all = pd.concat(real.values()), pd.concat(sim.values())
    shared = set(r_all["key"]) & set(s_all["key"])
    out = {"n_shared": len(shared),
           "coverage_of_real": len(shared) / r_all["key"].nunique(),
           "coverage_of_sim": len(shared) / s_all["key"].nunique()}
    for name in OBSERVABLES:
        r = r_all[r_all["key"].isin(shared) & np.isfinite(r_all[name])].groupby("key")[name].median()
        s = s_all[s_all["key"].isin(shared) & np.isfinite(s_all[name])].groupby("key")[name].median()
        both = r.index.intersection(s.index)
        if len(both) == 0:
            continue
        diff = (s[both] - r[both]).to_numpy()
        out[name] = {"w1": w1(s[both].to_numpy(), r[both].to_numpy()), "n": int(len(both)),
                     "paired_median_diff": float(np.median(diff)), "paired_mad_diff": float(np.median(np.abs(diff - np.median(diff))))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    matched = json.loads((a.out / "matched_complexity.json").read_text())
    real_runs, sim_runs = check_cohort(a.recipe)
    real = {r: load(a.recipe / "real10" / r / "plain" / "report.parquet") for r in real_runs}
    sims = {r: a.recipe / "sim" / r for r in sim_runs}
    res = {"matched_complexity": matched["matched_complexity"], "by_complexity": {}}
    for c in sorted({complexity_of(n) for n in sims}):
        sim = {n: load(p / "plain" / "report.parquet") for n, p in sims.items() if complexity_of(n) == c}
        res["by_complexity"][str(c)] = {"all": compare(real, sim, a.seed), "shared": shared_panel(real, sim)}
    dump(res, a.out / "results" / "discrepancy.json")
    m = res["by_complexity"][str(matched["matched_complexity"])]["all"]
    print("observable  sim-real/real-real  (matched complexity)")
    for k, v in m.items():
        r = v.get("ratio")
        print(f"{k:18s} {'n/a' if r is None else f'{r:8.2f}'}")


if __name__ == "__main__":
    main()
