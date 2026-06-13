"""Ground-truth evaluation of a simulated run against a search-engine report.

Given a TimSim ``synthetic_data.db`` (the ground truth: exactly which peptides, at
which charges, retention times, and abundances were simulated) and a DIA search report
(DiaNN or alphaDIA), compute the validation metrics:

- **RT correlation**     — reported retention time vs simulated apex RT (Pearson +
  median absolute error). Tests that the render places elution at the intended RT.
- **Charge accuracy**    — fraction of identified precursors whose charge was actually
  simulated for that peptide backbone.
- **Empirical FDR**      — entrapment: identified backbones that were NOT simulated /
  total identified. Validates the engine's FDR control on the data (meaningful only
  when searching a library larger than the simulated set, e.g. the full proteome).
- **Quant correlation**  — reported quantity vs simulated abundance
  (``events × relative_abundance``), log10–log10 Pearson, across the dynamic range.
- **Recall by abundance**— detection rate per simulated-abundance quartile (a flat
  curve implies an interference/intensity-model ceiling; a rising curve implies a
  realistic dynamic-range limit).

Instrument-agnostic: works for Bruker, Astral, or Orbitrap runs alike — it only needs
the DB + a report. See also the integration framework (``timsim.integration.eval``)
for the CI-style pass/fail harness; this module is the reusable metric core.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

_UNIMOD = re.compile(r"\[UNIMOD:\d+\]")


def _strip_mods(seq: str) -> str:
    """Backbone (bare residues) of a UNIMOD-bracket sequence, e.g.
    ``M[UNIMOD:35]C[UNIMOD:4]K`` -> ``MCK``."""
    return _UNIMOD.sub("", seq).replace("[", "").replace("]", "")


@dataclass
class GroundTruthMetrics:
    identified: int                      # identified precursors matched to a simulated backbone+charge
    reported_total: int                  # total precursors in the report
    simulated_ions: int                  # simulated (target) precursor ions
    rt_pearson: float
    rt_median_abs_error: float           # in the report's RT unit (minutes)
    charge_accuracy: float               # of matched-backbone IDs, fraction at a simulated charge
    empirical_fdr: float                 # identified backbones not in the simulated set / reported_total
    empirical_fdr_n_false: int
    quant_log_pearson: float
    quant_log_spearman: float
    quant_dynamic_range_orders: float
    recall_by_abundance_quartile: Dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_simulated_truth(db_path: str) -> pd.DataFrame:
    """Load the simulated (non-decoy) precursor ions with backbone, charge, apex RT
    (converted seconds->minutes), and abundance (``events × relative_abundance``)."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT i.sequence, i.charge, i.relative_abundance, "
        "p.events, p.retention_time_gru_predictor AS rt_s "
        "FROM ions i JOIN peptides p ON i.peptide_id = p.peptide_id "
        "WHERE p.decoy = 0",
        con,
    )
    con.close()
    df["bb"] = df["sequence"].map(_strip_mods)
    df["sim_rt"] = df["rt_s"] / 60.0                      # DB stores seconds; reports use minutes
    df["sim_quantity"] = df["events"] * df["relative_abundance"]
    return df


def load_report(path: str, engine: str = "diann") -> pd.DataFrame:
    """Normalize a search report to columns: ``bb, charge, rt, quantity``.

    engine='diann': a DiaNN ``report.parquet``/``.tsv``
    (Stripped.Sequence, Precursor.Charge, RT, Precursor.Quantity).
    engine='alphadia': an alphaDIA ``precursor.tsv``/parquet
    (sequence, charge, rt, ...)."""
    engine = engine.lower()
    if engine == "diann":
        r = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, sep="\t")
        return pd.DataFrame({
            "bb": r["Stripped.Sequence"].astype(str),
            "charge": r["Precursor.Charge"].astype(int),
            "rt": r["RT"].astype(float),
            "quantity": r.get("Precursor.Quantity", pd.Series(np.nan, index=r.index)).astype(float),
        })
    if engine == "alphadia":
        r = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, sep="\t")
        # alphaDIA precursor table: sequence (stripped), charge, rt (seconds), intensity
        seq_col = next(c for c in ("sequence", "Sequence", "stripped_sequence") if c in r.columns)
        rt_col = next((c for c in ("rt", "rt_observed", "rt_calibrated") if c in r.columns), None)
        q_col = next((c for c in ("intensity", "quantity", "sum_b_ion_intensity") if c in r.columns), None)
        rt = r[rt_col].astype(float) if rt_col else pd.Series(np.nan, index=r.index)
        if rt_col and rt.max() > 200:   # heuristic: alphaDIA rt in seconds -> minutes
            rt = rt / 60.0
        return pd.DataFrame({
            "bb": r[seq_col].map(_strip_mods),
            "charge": r["charge"].astype(int),
            "rt": rt,
            "quantity": r[q_col].astype(float) if q_col else pd.Series(np.nan, index=r.index),
        })
    raise ValueError(f"unknown engine '{engine}' (expected 'diann' or 'alphadia')")


def evaluate(truth: pd.DataFrame, report: pd.DataFrame) -> GroundTruthMetrics:
    """Compute ground-truth metrics from a simulated-truth frame and a normalized report."""
    from scipy.stats import pearsonr, spearmanr

    sim_bb = set(truth["bb"])
    sim_bc = set(zip(truth["bb"], truth["charge"]))
    bb_charges: Dict[str, set] = {}
    for b, c in zip(truth["bb"], truth["charge"]):
        bb_charges.setdefault(b, set()).add(int(c))
    sim_rt = dict(zip(truth["bb"], truth["sim_rt"]))            # apex RT per backbone
    sim_q = truth.groupby(["bb", "charge"])["sim_quantity"].sum().to_dict()

    n = len(report)
    rep_pairs = list(zip(report["bb"], report["charge"]))
    false_bb = [b not in sim_bb for b in report["bb"]]
    empirical_fdr = float(np.mean(false_bb)) if n else 0.0

    matched = report[report["bb"].isin(sim_bb)].copy()
    # RT correlation on matched backbones
    matched["sim_rt"] = matched["bb"].map(sim_rt)
    rt_ok = matched.dropna(subset=["rt", "sim_rt"])
    if len(rt_ok) >= 2 and rt_ok["rt"].std() > 0 and rt_ok["sim_rt"].std() > 0:
        rt_pearson = float(pearsonr(rt_ok["rt"], rt_ok["sim_rt"])[0])
        rt_mae = float(np.median(np.abs(rt_ok["rt"] - rt_ok["sim_rt"])))
    else:
        rt_pearson, rt_mae = float("nan"), float("nan")
    # charge accuracy
    charge_acc = float(np.mean([
        c in bb_charges.get(b, set()) for b, c in zip(matched["bb"], matched["charge"])
    ])) if len(matched) else float("nan")
    # quant correlation (log-log) on matched backbone+charge with positive quantity
    matched["sim_q"] = [sim_q.get((b, c)) for b, c in zip(matched["bb"], matched["charge"])]
    q = matched.dropna(subset=["sim_q", "quantity"])
    q = q[(q["sim_q"] > 0) & (q["quantity"] > 0)]
    if len(q) >= 2:
        lr, ls = np.log10(q["quantity"].values), np.log10(q["sim_q"].values)
        quant_p = float(pearsonr(lr, ls)[0])
        quant_s = float(spearmanr(lr, ls)[0])
        dyn = float(np.log10(q["sim_q"].max() / q["sim_q"].min()))
    else:
        quant_p = quant_s = dyn = float("nan")
    # recall by simulated-abundance quartile (over all simulated ions)
    ident_pairs = set(rep_pairs)
    t = truth.copy()
    t["found"] = [(b, c) in ident_pairs for b, c in zip(t["bb"], t["charge"])]
    recall_q: Dict[str, float] = {}
    if len(t) >= 4:
        t["q"] = pd.qcut(t["sim_quantity"].rank(method="first"), 4,
                         labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        recall_q = {str(k): round(float(v), 4)
                    for k, v in t.groupby("q", observed=True)["found"].mean().items()}

    return GroundTruthMetrics(
        identified=int(sum((b, c) in sim_bc for b, c in rep_pairs)),
        reported_total=n,
        simulated_ions=len(truth),
        rt_pearson=rt_pearson,
        rt_median_abs_error=rt_mae,
        charge_accuracy=charge_acc,
        empirical_fdr=empirical_fdr,
        empirical_fdr_n_false=int(sum(false_bb)),
        quant_log_pearson=quant_p,
        quant_log_spearman=quant_s,
        quant_dynamic_range_orders=dyn,
        recall_by_abundance_quartile=recall_q,
    )


def evaluate_run(db_path: str, report_path: str, engine: str = "diann") -> GroundTruthMetrics:
    """Convenience: load DB + report and evaluate."""
    return evaluate(load_simulated_truth(db_path), load_report(report_path, engine))


def main() -> None:
    ap = argparse.ArgumentParser(description="Ground-truth eval of a TimSim run vs a DIA search report.")
    ap.add_argument("--db", required=True, help="path to synthetic_data.db")
    ap.add_argument("--report", required=True, help="path to the search report (DiaNN parquet/tsv, or alphaDIA)")
    ap.add_argument("--engine", default="diann", choices=["diann", "alphadia"])
    ap.add_argument("--out", default=None, help="optional path to write metrics JSON")
    a = ap.parse_args()
    m = evaluate_run(a.db, a.report, a.engine)
    print(m.to_json())
    if a.out:
        with open(a.out, "w") as f:
            f.write(m.to_json())


if __name__ == "__main__":
    main()
