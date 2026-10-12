"""Truth-derived and entrapment-estimated FDP for the frozen claims in evident/claims/.

Modes map to claims:
  estimator-bias      entrap.sim.estimator-bias.diann
  search-space-shift  entrap.sim.search-space-shift.diann
  verdict             entrap.verdict.diann.{real10,sim-matched}, entrap.sim-vs-real.verdict-agree.diann
  sim-vs-real         entrap.sim-vs-real.diann
  baseline            vm.baseline.fdp-{bound,nominal}.diann
  all                 everything above, then the band-or-class diagnostic decision
  stress              only --claim vm.baseline.*, which routes to baseline; stress rows are not frozen yet

Writes results/<mode>.json and a last_verified sidecar for typed-trust.
Any missing, non-finite or inconsistent input stops the run instead of shrinking an aggregate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vm_common import (Q_MAX, accepted_from_report, check_cohort, complexity_of, dump, fdp_stripped, fdrbench_at_threshold,
                       find_truth_db, finite, is_entrapment, label_false, modal, read_pairs, read_truth, verdict,
                       write_sidecar)

class Ctx:
    def __init__(self, recipe: Path, sim_root: Path, out: Path):
        self.recipe, self.sim_root, self.out = recipe, sim_root, out
        self.pairs = read_pairs(recipe / "entrap" / "human_entrap_pep.txt")
        self.real_runs, self.sim_runs = check_cohort(recipe)
        self._truth: dict[str, tuple[set, set]] = {}

    def truth(self, run: str) -> tuple[set, set]:
        if run not in self._truth:
            self._truth[run] = read_truth(find_truth_db(self.sim_root / run))
        return self._truth[run]

    def matched(self) -> dict:
        p = self.out / "matched_complexity.json"
        if not p.exists():
            raise SystemExit("run vm_match_complexity.py first (vm.protocol.matched-complexity)")
        m = json.loads(p.read_text())
        if not set(m["matched_runs"]) <= set(self.sim_runs) or len(m["matched_runs"]) != 3:
            raise SystemExit("matched_complexity.json does not match the current sim cohort")
        return m

    def estimates(self, side: str, run: str) -> tuple[pd.DataFrame, dict]:
        """FDRBench accepted set and estimates, reconciled with its own counts and with DIA-NN."""
        acc, est = fdrbench_at_threshold(self.recipe / side / run / "entrap" / "fdp_precursor.csv")
        entrap = is_entrapment(acc["stripped"], self.pairs, acc["protein"])
        if int(entrap.sum()) != est["n_p"] or int((~entrap).sum()) != est["n_t"]:
            raise ValueError(f"{side}/{run}: labels give {int(entrap.sum())} entrapments, FDRBench n_p={est['n_p']}")
        diann = accepted_from_report(self.recipe / side / run / "entrap" / "report.parquet")
        a, b = set(zip(acc["mod"], acc["charge"])), set(zip(diann["mod"], diann["charge"]))
        est["only_fdrbench"], est["only_diann"] = len(a - b), len(b - a)
        acc["entrap"] = entrap
        return acc, est


def fdp(flags) -> float:
    flags = np.asarray(flags)
    if len(flags) == 0:
        raise ValueError("FDP of an empty accepted set")
    return float(flags.mean())


def mean_strict(xs, what: str) -> float:
    return float(np.mean([finite(x, what) for x in xs]))


def estimator_bias(c: Ctx) -> dict:
    rows = []
    for run in c.sim_runs:
        acc, est = c.estimates("sim", run)
        lab = label_false(acc, *c.truth(run), acc["entrap"].to_numpy())
        true = fdp(lab["false_mod"])
        rows.append({
            "run": run, "complexity": complexity_of(run), **est,
            "true_fdp": true, "true_fdp_stripped": fdp_stripped(lab),
            "paired_minus_true": est["paired"] - true, "combined_minus_true": est["combined"] - true,
            "bracketed": bool(est["lower"] <= true <= est["paired"]),
        })
    df = pd.DataFrame(rows)
    return {
        "per_run": rows,
        "per_complexity": df.groupby("complexity")[["paired_minus_true", "combined_minus_true", "true_fdp"]].mean().to_dict("index"),
        "values": {
            "entrap.sim.estimator-bias.diann": {
                "paired_bias_mean": mean_strict(df["paired_minus_true"], "paired bias"),
                "combined_bias_mean": mean_strict(df["combined_minus_true"], "combined bias"),
                "bracket_frequency": float(df["bracketed"].mean()),
            }
        },
    }


def search_space_shift(c: Ctx) -> dict:
    rows = []
    for run in c.sim_runs:
        tm, ts = c.truth(run)
        res = {}
        for lib in ("plain", "entrap"):
            acc = accepted_from_report(c.recipe / "sim" / run / lib / "report.parquet")
            tgt = acc[~is_entrapment(acc["stripped"], c.pairs, acc["protein"])]
            lab = label_false(tgt, tm, ts, np.zeros(len(tgt), bool))
            res[lib] = (fdp(lab["false_mod"]), fdp_stripped(lab), set(zip(lab["mod"], lab["charge"])))
        a, b = res["plain"][2], res["entrap"][2]
        rows.append({
            "run": run, "fdp_t_plain": res["plain"][0], "fdp_t_entrap": res["entrap"][0],
            "fdp_t_plain_stripped": res["plain"][1], "fdp_t_entrap_stripped": res["entrap"][1],
            "abs_shift": abs(res["entrap"][0] - res["plain"][0]),
            "n_plain": len(a), "n_entrap": len(b), "count_ratio": len(b) / len(a),
            "lost": len(a - b), "gained": len(b - a), "jaccard": len(a & b) / len(a | b),
        })
    df = pd.DataFrame(rows)
    return {
        "per_run": rows,
        "values": {
            "entrap.sim.search-space-shift.diann": {
                "target_fdp_shift_mean_abs": mean_strict(df["abs_shift"], "target FDP shift"),
                "target_count_ratio": float(df["count_ratio"].median()),
            }
        },
    }


def side_verdict(paired_max: float, lower_max: float) -> str:
    """Claim-level verdict from the maxima: every run valid, some run invalid, or inconclusive."""
    if paired_max <= Q_MAX:
        return "valid"
    if lower_max > Q_MAX:
        return "invalid"
    return "inconclusive"


def verdicts(c: Ctx) -> dict:
    sides = {"real10": ("real10", c.real_runs), "sim-matched": ("sim", c.matched()["matched_runs"])}
    out, vals = {}, {}
    for name, (side, runs) in sides.items():
        per = []
        for run in runs:
            _, e = c.estimates(side, run)
            per.append({"run": run, **e, "class": verdict(e["lower"], e["paired"])})
        pmax = max(finite(p["paired"], "paired") for p in per)
        lmax = max(finite(p["lower"], "lower") for p in per)
        out[name] = {"per_run": per, "modal": modal([p["class"] for p in per]), "side_verdict": side_verdict(pmax, lmax)}
        vals[f"entrap.verdict.diann.{name}"] = {"paired_upper_max": pmax, "lower_bound_max": lmax}
    agree = out["real10"]["modal"] == out["sim-matched"]["modal"]
    vals["entrap.sim-vs-real.verdict-agree.diann"] = {"class_agreement": 1.0 if agree else 0.0}
    out["no_positive_transfer_evidence"] = agree and out["real10"]["modal"] == "inconclusive"
    out["values"] = vals
    return out


def sim_vs_real(c: Ctx) -> dict:
    m = c.matched()
    est = {"real10": [c.estimates("real10", r)[1] for r in c.real_runs],
           "sim": [c.estimates("sim", r)[1] for r in m["matched_runs"]]}
    mean = {s: {k: mean_strict([e[k] for e in v], f"{s} {k}") for k in ("lower", "combined", "paired")} for s, v in est.items()}
    return {
        "matched_complexity": m["matched_complexity"], "poor_match": m["poor_match"], "means": mean,
        "values": {
            "entrap.sim-vs-real.diann": {
                "paired_abs_diff": abs(mean["sim"]["paired"] - mean["real10"]["paired"]),
                "combined_abs_diff": abs(mean["sim"]["combined"] - mean["real10"]["combined"]),
            }
        },
    }


def baseline(c: Ctx) -> dict:
    rows = []
    for run in c.sim_runs:
        acc = accepted_from_report(c.recipe / "sim" / run / "plain" / "report.parquet")
        ent = is_entrapment(acc["stripped"], c.pairs, acc["protein"])
        if ent.any():
            raise ValueError(f"{run}: entrapment precursors in a plain-library search")
        lab = label_false(acc, *c.truth(run), ent)
        rows.append({"run": run, "complexity": complexity_of(run), "accepted": len(lab),
                     "false": int(lab["false_mod"].sum()), "fdp": fdp(lab["false_mod"]),
                     "fdp_stripped": fdp_stripped(lab)})
    df = pd.DataFrame(rows)
    fdps = [finite(x, "baseline FDP") for x in df["fdp"]]
    return {
        "per_run": rows,
        "per_complexity": df.groupby("complexity")["fdp"].agg(["mean", "std", "max"]).to_dict("index"),
        "values": {
            "vm.baseline.fdp-bound.diann": {"fdp_max": max(fdps)},
            "vm.baseline.fdp-nominal.diann": {"fdp_mean": float(np.mean(fdps))},
        },
    }


def diagnostic(results: dict) -> dict:
    """entrap.sim-vs-real.diann: the diagnostic agrees if the 0.5 pp band holds or the classes agree."""
    band = all(v <= 0.005 for v in results["sim-vs-real"]["values"]["entrap.sim-vs-real.diann"].values())
    cls = results["verdict"]["values"]["entrap.sim-vs-real.verdict-agree.diann"]["class_agreement"] == 1.0
    return {"band_holds": band, "class_agrees": cls, "diagnostic_agrees": band or cls,
            "no_positive_transfer_evidence": results["verdict"]["no_positive_transfer_evidence"]}


MODES = {"estimator-bias": estimator_bias, "search-space-shift": search_space_shift,
         "verdict": verdicts, "sim-vs-real": sim_vs_real, "baseline": baseline}
REPORT_ONLY = {"bracket_frequency", "target_count_ratio"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=[*MODES, "all", "stress"], required=True)
    ap.add_argument("--recipe", type=Path, required=True, help="searches/vm/recipe-v1")
    ap.add_argument("--sim-root", type=Path, required=True, help="searches/vm/sim-baseline")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--side", help="accepted for the claim commands; verdict always computes both sides")
    ap.add_argument("--claim", help="stress mode: claim id")
    a = ap.parse_args()
    modes = list(MODES) if a.mode == "all" else [a.mode]
    if a.mode == "stress":
        if not (a.claim or "").startswith("vm.baseline."):
            raise SystemExit("stress settings are not frozen yet (evident/vm/stress_settings.yaml)")
        modes = ["baseline"]
    c = Ctx(a.recipe, a.sim_root, a.out)
    if a.mode == "all":
        c.matched()  # fail before any FDP is computed
    results = {}
    for mode in modes:
        results[mode] = MODES[mode](c)
        dump(results[mode], a.out / "results" / f"{mode}.json")
        write_sidecar(a.out, results[mode]["values"], REPORT_ONLY)
        print(mode, json.dumps(results[mode]["values"]))
    if a.mode == "all":
        d = diagnostic(results)
        dump(d, a.out / "results" / "diagnostic.json")
        print("diagnostic", json.dumps(d))


if __name__ == "__main__":
    main()
