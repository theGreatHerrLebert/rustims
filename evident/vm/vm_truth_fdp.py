"""Truth-derived and entrapment-estimated FDP for the frozen claims in evident/claims/.

Modes map to claims:
  estimator-bias      entrap.sim.estimator-bias.diann
  search-space-shift  entrap.sim.search-space-shift.diann
  verdict             entrap.verdict.diann.{real10,sim-matched}, entrap.sim-vs-real.verdict-agree.diann
  sim-vs-real         entrap.sim-vs-real.diann
  baseline            vm.baseline.fdp-{bound,nominal}.diann
  all                 everything above, then the band-or-class diagnostic decision
  stress              --claim vm.baseline.* routes to baseline; --claim vm.<row>.* runs that frozen stress row
                      (predictor-swap, rt-spread, mass-accuracy; evident/vm/stress_settings*.yaml) on
                      --stress-recipe / --stress-root

Writes results/<mode>.json and a last_verified sidecar for typed-trust.
Any missing, non-finite or inconsistent input stops the run instead of shrinking an aggregate.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from vm_common import (Q_MAX, accepted_from_report, check_cohort, complexity_of, dump, fdp_stripped, fdrbench_at_threshold,
                       find_truth_db, finite, is_entrapment, label_false, mod_key, modal, read_pairs, read_truth, verdict,
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


STRESS_SEEDS = ("jitter-250k-r1", "jitter-250k-r2", "jitter-250k-r3")
STRESS_SETTINGS = ("rt-chronologer", "im-std", "joint")  # claim scope; replay-control is reported only
# Claim scope per row. rt-spread reuses predictor-swap's replay-control replays (stress_settings_rt_spread.yaml);
# mass-accuracy has its own control and two settings reported without tolerance (stress_settings_mass_accuracy.yaml).
STRESS_ROWS = {"predictor-swap": STRESS_SETTINGS, "rt-spread": ("rt-spread",),
               "mass-accuracy": ("ms1-real", "ms2-real", "ms12-real")}
REPLAY_CONTROL = "replay-control"
ROW_CONTROL = {"mass-accuracy": "replay-control-ma"}
ROW_REPORTED_ONLY = {"mass-accuracy": ("ms12-centred", "ms12-2x")}
MZ_BIAS = ("mz_bias_ppm_ms1", "mz_bias_ppm_ms2")
_DIANN_LOG = {"rt_window_min": r"RT window set to ([0-9.eE+-]+)", "im_window": r"IM window set to ([0-9.eE+-]+)",
              "ms1_acc_ppm": r"Recommended MS1 mass accuracy setting: ([0-9.eE+-]+)"}


def diann_windows(stdout: Path) -> dict:
    """Last value DIA-NN logged for each calibration window; NaN when it logged none."""
    txt = stdout.read_text(errors="replace") if stdout.exists() else ""
    out = {}
    for k, pat in _DIANN_LOG.items():
        hits = re.findall(pat, txt)
        out[k] = float(hits[-1]) if hits else float("nan")
    return out


def identified_rt_quantiles(report: Path) -> list:
    """5/50/95% RT (min) of accepted target precursors; NaN when the report has no RT column."""
    d = pd.read_parquet(report)
    if "RT" not in d.columns:
        return [float("nan")] * 3
    if "Decoy" in d.columns:
        d = d[d["Decoy"] == 0]
    rt = d.loc[d["Q.Value"] <= 0.01, "RT"].to_numpy(float)
    return np.quantile(rt, [0.05, 0.5, 0.95]).tolist() if len(rt) else [float("nan")] * 3


MA_BIAS_COLS = {"replay-control-ma": (), "ms1-real": ("mz_bias_ppm_ms1",), "ms2-real": ("mz_bias_ppm_ms2",),
                "ms12-real": MZ_BIAS, "ms12-centred": MZ_BIAS, "ms12-2x": MZ_BIAS}


def replay_checked(run_dir: Path, mass_accuracy: bool = False) -> str:
    """A replay counts only with done.flag and a --verify result naming both apex columns. Returns done.flag's commit.

    mass-accuracy replays also need: the verify result bound to this replay's truth DB, the ion-key check, exactly the
    setting's bias columns, and (outside the control) the comparison with the same seed's replay-control-ma.
    """
    v = run_dir / "verify.json"
    if not (run_dir / "done.flag").exists() or not v.exists():
        raise SystemExit(f"{run_dir.name}: no done.flag or verify.json, the replay was not checked")
    res = json.loads(v.read_text())
    if not {"retention_time_gru_predictor", "inv_mobility_gru_predictor"} <= set(res):
        raise SystemExit(f"{run_dir.name}: verify.json lacks the apex checks")
    if mass_accuracy:
        rid, setting = run_dir.name.split("-", 3)[:3], run_dir.name.split("-", 3)[3]
        control = f"{'-'.join(rid)}-{ROW_CONTROL['mass-accuracy']}"
        if Path(res.get("replay_db", "")).resolve() != find_truth_db(run_dir).resolve():
            raise SystemExit(f"{run_dir.name}: verify.json does not name this replay's truth DB")
        if "ion_keys" not in res:
            raise SystemExit(f"{run_dir.name}: verify.json lacks the ion-key check")
        want = set(MA_BIAS_COLS[setting])
        if {k for k in MZ_BIAS if k in res} != want or not all(res[k].get("all_equal") is True for k in want):
            raise SystemExit(f"{run_dir.name}: verify.json bias checks {sorted(set(MZ_BIAS) & set(res))} != {sorted(want)}")
        if want and f"/{control}/" not in res["ion_keys"].get("same_as_control", ""):
            raise SystemExit(f"{run_dir.name}: not compared with {control}")
    return (run_dir / "done.flag").read_text().strip()


def truth_fdp_row(c: Ctx, report_dir: Path, sim_dir: Path) -> dict:
    acc = accepted_from_report(report_dir / "plain" / "report.parquet")
    ent = is_entrapment(acc["stripped"], c.pairs, acc["protein"])
    if ent.any():
        raise ValueError(f"{report_dir}: entrapment precursors in a plain-library search")
    truth_mod, truth_stripped = read_truth(find_truth_db(sim_dir))
    lab = label_false(acc, truth_mod, truth_stripped, ent)
    false = lab["false_mod"]
    form = false & ~lab["false_stripped"]  # sequence simulated at that charge, modification state wrong
    no_ox = {(k.replace("{35}", ""), z) for k, z in truth_mod}
    met_ox = form & np.fromiter(((k.replace("{35}", ""), z) in no_ox for k, z in zip(lab["mod"], lab["charge"])),
                                bool, len(lab))
    share = (lambda m: float(m.sum() / false.sum()) if false.any() else float("nan"))
    # an empty accepted set has undefined FDP (NaN); rows other than mass-accuracy then stop in mean_fdp
    return {"accepted": len(lab), "false": int(false.sum()), "fdp": fdp(false) if len(lab) else float("nan"),
            "fdp_stripped": fdp_stripped(lab),
            "mod_form_share_of_false": share(form), "met_ox_share_of_false": share(met_ox),
            "identified_rt_q05_q50_q95_min": identified_rt_quantiles(report_dir / "plain" / "report.parquet"),
            **diann_windows(report_dir / "plain" / "diann.stdout")}


def mass_deltas(report: Path) -> dict:
    """DIA-NN MS1 and best-fragment m/z deltas (ppm, as vm_discrepancy.py) of accepted targets: q05/q50/q95 and the
    90% width (numpy default linear quantile); non-finite values dropped and counted; NaN when none remain."""
    from vm_discrepancy import OBSERVABLES
    d = pd.read_parquet(report)
    if "Decoy" in d.columns:
        d = d[d["Decoy"] == 0]
    d = d[d["Q.Value"] <= Q_MAX]
    out = {}
    for name in ("ms1_ppm", "fr_ppm"):
        x = OBSERVABLES[name][0](d).to_numpy(float) if len(d) else np.array([])
        ok = np.isfinite(x)
        q = np.quantile(x[ok], [0.05, 0.5, 0.95]).tolist() if ok.any() else [float("nan")] * 3
        out[name] = {"q05": q[0], "q50": q[1], "q95": q[2], "width90": q[2] - q[0], "n": int(ok.sum()),
                     "n_nonfinite": int((~ok).sum())}
    return out


LOSS_EDGES = (0, 2, 4, 6, 8, 10, 15, 20, np.inf)


def loss_by_offset(report_dir: Path, sim_dir: Path) -> dict:
    """Descriptive: ions accepted at 1% / ions simulated, by |injected offset| (ppm) bin, per bias column."""
    with sqlite3.connect(f"file:{find_truth_db(sim_dir)}?mode=ro", uri=True) as con:
        cols = [r[1] for r in con.execute("pragma table_info(ions)")]
        have = [col for col in MZ_BIAS if col in cols]
        if not have:
            return {}
        ions = pd.read_sql(f"select sequence, charge, {', '.join(have)} from ions", con)
    acc = accepted_from_report(report_dir / "plain" / "report.parquet")
    hit = set(zip(acc["mod"], acc["charge"]))
    found = np.fromiter(((mod_key(q), int(z)) in hit for q, z in zip(ions["sequence"], ions["charge"])), bool, len(ions))
    out = {}
    for col in have:
        b = pd.cut(ions[col].abs().fillna(0.0), LOSS_EDGES, right=False)
        g = pd.DataFrame({"bin": b, "found": found}).groupby("bin", observed=False)["found"]
        out[col] = {str(k): {"simulated": int(n), "accepted": int(a), "share": float(a / n) if n else float("nan")}
                    for (k, n), a in zip(g.size().items(), g.sum().to_numpy())}
    return out


# stress_settings_mass_accuracy.yaml, interpretation (frozen with the settings)
MA_MIN_WIDTH = {"ms1_ppm": 4.5, "fr_ppm": 9.2}
MA_MIN_DEPTH = 0.70
MA_REPLAY_FDP = 0.0025
MA_REPLAY_DEPTH = (0.95, 1.05)


def _fin(x) -> bool:
    return x is not None and bool(np.isfinite(x))


def ma_interpretation(values: dict, base: dict, rows: dict, deltas: dict, control: str) -> dict:
    """Robustness to the injected-offset construction is supported only if (0) and (a)-(c) all hold."""
    bound = values["vm.mass-accuracy.fdp-bound.diann"]["fdp_max"]
    delta = values["vm.mass-accuracy.fdp-delta.diann"]["fdp_delta_max"]
    c0 = _fin(bound) and _fin(delta) and bound <= 0.015 and delta <= 0.005
    a = {rid: {k: deltas["ms12-real"][rid][k]["width90"] for k in MA_MIN_WIDTH} for rid in STRESS_SEEDS}
    ca = all(_fin(w[k]) and w[k] >= MA_MIN_WIDTH[k] for w in a.values() for k in MA_MIN_WIDTH)

    def ratio(n, d):
        return n / d if d else float("nan")
    b = {s: {rid: ratio(rows[s][rid]["accepted"], base[rid]["accepted"]) for rid in STRESS_SEEDS}
         for s in STRESS_ROWS["mass-accuracy"]}
    cb = all(_fin(v) and v >= MA_MIN_DEPTH for d in b.values() for v in d.values())
    c = {rid: {"fdp_diff": rows[control][rid]["fdp"] - base[rid]["fdp"],
               "depth_ratio": ratio(rows[control][rid]["accepted"], base[rid]["accepted"])} for rid in STRESS_SEEDS}
    cc = all(_fin(v["fdp_diff"]) and abs(v["fdp_diff"]) <= MA_REPLAY_FDP and _fin(v["depth_ratio"])
             and MA_REPLAY_DEPTH[0] <= v["depth_ratio"] <= MA_REPLAY_DEPTH[1] for v in c.values())
    failing = [n for n, ok in (("0 claims", c0), ("a manipulation", ca), ("b depth", cb), ("c replay", cc)) if not ok]
    return {"claims_pass": c0, "manipulation_width90_ms12_real": a, "manipulation_holds": ca,
            "depth_ratio": b, "depth_holds": cb, "replay_vs_baseline": c, "replay_holds": cc,
            "conclusion": "supports robustness to this construction" if not failing
            else "uninformative about robustness to this construction", "failing": failing}


def stress(c: Ctx, stress_recipe: Path, stress_root: Path, row: str = "predictor-swap") -> dict:
    """vm.<row>.fdp-{bound,delta}.diann: paired by baseline seed, delta against the baseline sims."""
    settings = STRESS_ROWS[row]
    control = ROW_CONTROL.get(row, REPLAY_CONTROL)
    extra = ROW_REPORTED_ONLY.get(row, ())
    ma = row == "mass-accuracy"
    m = c.matched()
    if sorted(m["matched_runs"]) != list(STRESS_SEEDS):
        raise SystemExit(f"matched runs {m['matched_runs']} are not the stress baselines {STRESS_SEEDS}")
    every = (control, *settings, *extra)
    done = {replay_checked(stress_root / f"{rid}-{s}", ma) for s in every for rid in STRESS_SEEDS}
    if ma and len(done) != 1:
        raise SystemExit(f"mass-accuracy replays were done at different commits: {sorted(done)}")
    base = {rid: truth_fdp_row(c, c.recipe / "sim" / rid, c.sim_root / rid) for rid in STRESS_SEEDS}
    rows = {s: {rid: truth_fdp_row(c, stress_recipe / "sim" / f"{rid}-{s}", stress_root / f"{rid}-{s}")
                for rid in STRESS_SEEDS} for s in every}

    def mean_fdp(d: dict) -> float:
        if ma:  # undefined FDP propagates as NaN, so the interpretation is still reported
            return float(np.mean([r["fdp"] for r in d.values()]))
        return mean_strict([finite(r["fdp"], "stress FDP") for r in d.values()], "FDP")

    def max_defined(xs) -> float:
        xs = list(xs)
        return max(xs) if all(np.isfinite(xs)) else float("nan")

    b = mean_fdp(base)
    per_setting = {s: {"fdp_mean": mean_fdp(rows[s]), "delta_vs_baseline": mean_fdp(rows[s]) - b,
                       "delta_vs_replay_control": mean_fdp(rows[s]) - mean_fdp(rows[control]),
                       "accepted_mean": float(np.mean([r["accepted"] for r in rows[s].values()]))}
                   for s in every}
    paired = {s: {rid: {"minus_baseline": rows[s][rid]["fdp"] - base[rid]["fdp"],
                        "minus_replay_control": rows[s][rid]["fdp"] - rows[control][rid]["fdp"]}
                  for rid in STRESS_SEEDS} for s in every}
    res = {
        "row": row, "control": control, "claim_settings": list(settings), "reported_only": list(extra),
        "baseline": base, "per_seed": rows, "per_seed_differences": paired, "per_setting": per_setting,
        "baseline_fdp_mean": b,
        "values": {
            f"vm.{row}.fdp-bound.diann": {"fdp_max": max_defined(rows[s][rid]["fdp"] for s in settings
                                                                       for rid in STRESS_SEEDS)},
            f"vm.{row}.fdp-delta.diann": {"fdp_delta_max": max_defined(abs(per_setting[s]["delta_vs_baseline"])
                                                                             for s in settings)},
        },
    }
    if ma:
        rep = lambda d: d / "plain" / "report.parquet"  # noqa: E731
        deltas = {"real": {r: mass_deltas(rep(c.recipe / "real10" / r)) for r in c.real_runs},
                  "baseline": {rid: mass_deltas(rep(c.recipe / "sim" / rid)) for rid in STRESS_SEEDS},
                  **{s: {rid: mass_deltas(rep(stress_recipe / "sim" / f"{rid}-{s}")) for rid in STRESS_SEEDS}
                     for s in every}}
        res["mass_deltas_ppm"] = deltas
        res["loss_by_offset"] = {s: {rid: loss_by_offset(stress_recipe / "sim" / f"{rid}-{s}",
                                                         stress_root / f"{rid}-{s}") for rid in STRESS_SEEDS}
                                 for s in (*settings, *extra)}
        prep = stress_root / "prepared-mass-accuracy"
        res["done_commit"] = done.pop()
        res["real_accepted"] = {r: len(accepted_from_report(rep(c.recipe / "real10" / r))) for r in c.real_runs}
        res["injected"] = {rid: json.loads(p.read_text()) if (p := prep / rid / "prepare.json").exists() else "MISSING"
                           for rid in STRESS_SEEDS}
        res["tdf_bin_sha256"] = {rid: json.loads(p.read_text()) if p.exists() else "MISSING"
                                 for rid in STRESS_SEEDS
                                 for p in [stress_root / f"{rid}-{control}" / "tdf_bin_sha256.json"]}
        res["interpretation"] = ma_interpretation(res["values"], base, rows, deltas, control)
    return res


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
    ap.add_argument("--stress-recipe", type=Path, help="stress mode: searches/vm/recipe-v1-stress")
    ap.add_argument("--stress-root", type=Path, help="stress mode: searches/vm/sim-stress")
    a = ap.parse_args()
    modes = list(MODES) if a.mode == "all" else [a.mode]
    c = Ctx(a.recipe, a.sim_root, a.out)
    if a.mode == "stress":
        claim = a.claim or ""
        row = next((r for r in STRESS_ROWS if claim.startswith(f"vm.{r}.")), None)
        if row:
            if not (a.stress_recipe and a.stress_root):
                raise SystemExit(f"--stress-recipe and --stress-root are required for the {row} row")
            res = stress(c, a.stress_recipe, a.stress_root, row)
            dump(res, a.out / "results" / f"stress-{row}.json")
            undefined = [k for k, v in res["values"].items() for x in v.values() if not np.isfinite(x)]
            if undefined:  # results and interpretation are written; the claims are not assessed
                raise SystemExit(f"undefined claim values (an empty accepted set), not assessed: {undefined}")
            write_sidecar(a.out, res["values"], REPORT_ONLY)
            print("stress", json.dumps(res["values"]), json.dumps(res["per_setting"]))
            return
        if not claim.startswith("vm.baseline."):
            raise SystemExit(f"no frozen stress settings for {claim!r}")
        modes = ["baseline"]
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
