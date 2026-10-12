"""claims/protein.yaml: protein-level entrapment, truth per vm.protocol.protein-estimand."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from vm_common import (ENTRAP_TAG, Q_MAX, REAL_RUNS, check_cohort, dump, find_truth_db, finite, mod_key, modal, read_truth,
                       write_sidecar)


def accession(header: str) -> str:
    parts = header.split("|")
    return parts[1] if len(parts) > 2 else header.split()[0]


def simulated_proteins(db: Path) -> set[str]:
    """Target proteins with at least one simulated ion."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        rows = c.execute("select distinct p.protein from ions i join peptides p on p.peptide_id = i.peptide_id").fetchall()
    return {accession(r[0]) for r in rows}


def at_threshold(csv: Path) -> tuple[pd.DataFrame, dict]:
    f = pd.read_csv(csv)
    acc = f[f["q_value"] <= Q_MAX].copy()
    if acc.empty:
        raise ValueError(f"no accepted protein groups in {csv}")
    n = acc["n_t"] + acc["n_p"]
    ends = acc[n == n.max()]
    if len(ends.drop_duplicates(["lower_bound_fdp", "combined_fdp"])) > 1:
        raise ValueError(f"tied end rows disagree in {csv}")
    if int(n.max()) != len(acc):
        raise ValueError(f"{csv}: {len(acc)} accepted groups but n_t + n_p = {int(n.max())}")
    last = ends.iloc[0]
    est = {"n_t": int(last["n_t"]), "n_p": int(last["n_p"]),
           "lower": finite(last["lower_bound_fdp"], "lower"), "combined": finite(last["combined_fdp"], "combined")}
    return acc[["protein"]].reset_index(drop=True), est


def verdict(lower: float, combined: float) -> str:
    if combined <= Q_MAX:
        return "valid"
    return "invalid" if lower > Q_MAX else "inconclusive"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=Path, required=True, help="searches/vm/recipe-v1 (for the cohort check)")
    ap.add_argument("--protein", type=Path, required=True, help="searches/vm/recipe-v1-protein")
    ap.add_argument("--sim-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    _, sim_runs = check_cohort(a.recipe)
    matched = json.loads((a.out / "matched_complexity.json").read_text())["matched_runs"]

    rows = []
    for run in sim_runs:
        acc, est = at_threshold(a.protein / "sim" / run / "fdp_protein.csv")
        db = find_truth_db(a.sim_root / run)
        sim_prot = simulated_proteins(db)
        accs = acc["protein"].str.split(";")
        entrap_only = accs.map(lambda g: all(ENTRAP_TAG in p for p in g))
        if int(entrap_only.sum()) > est["n_p"]:
            raise ValueError(f"{run}: more entrapment-only groups than FDRBench n_p")
        false = accs.map(lambda g: not any(p in sim_prot for p in g if ENTRAP_TAG not in p))
        true_fdp = float(false.mean())
        # Diagnostic, not the frozen rule: a group is true if one of its accepted precursors was simulated.
        truth_mod, _ = read_truth(db)
        d = pd.read_parquet(a.protein / "sim" / run / "report.parquet", columns=["Protein.Group", "Modified.Sequence",
                                                                                   "Precursor.Charge", "Q.Value", "Decoy"])
        d = d[(d["Decoy"] == 0) & (d["Q.Value"] <= Q_MAX)]
        d["hit"] = [(mod_key(s), int(z)) in truth_mod for s, z in zip(d["Modified.Sequence"], d["Precursor.Charge"])]
        evid = d.groupby("Protein.Group")["hit"].any()
        alt_false = acc["protein"].map(lambda g: not bool(evid.get(g, False)))
        rows.append({"run": run, **est, "true_fdp": true_fdp, "true_fdp_evidence_rule": float(alt_false.mean()),
                     "combined_minus_true": est["combined"] - true_fdp, "class": verdict(est["lower"], est["combined"])})
    df = pd.DataFrame(rows)

    real = []
    for run in REAL_RUNS:
        _, est = at_threshold(a.protein / "real10" / run / "fdp_protein.csv")
        real.append({"run": run, **est, "class": verdict(est["lower"], est["combined"])})
    sim_m = [r for r in rows if r["run"] in matched]

    vals = {
        "entrap.protein.sim.estimator-bias.diann": {"combined_bias_mean": float(df["combined_minus_true"].mean())},
        "entrap.protein.verdict.diann.real10": {"combined_upper_max": max(r["combined"] for r in real),
                                                "lower_bound_max": max(r["lower"] for r in real)},
        "entrap.protein.verdict.diann.sim-matched": {"combined_upper_max": max(r["combined"] for r in sim_m),
                                                     "lower_bound_max": max(r["lower"] for r in sim_m)},
        "entrap.protein.sim-vs-real.diann": {"combined_abs_diff": abs(np.mean([r["combined"] for r in sim_m])
                                                                       - np.mean([r["combined"] for r in real]))},
    }
    dump({"sim": rows, "real10": real,
          "modal_class": {"real10": modal([r["class"] for r in real]), "sim-matched": modal([r["class"] for r in sim_m])},
          "values": vals}, a.out / "results" / "protein.json")
    write_sidecar(a.out, vals)
    for k, v in vals.items():
        print(k, v)


if __name__ == "__main__":
    main()
