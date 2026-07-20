"""``timsim-v2-thermo-eval`` — score a DiaNN DIA search of a synthetic Thermo Astral ``.raw`` against the
render's answer key, with a **hierarchical, present+detectable denominator** and a recall-vs-abundance
curve (per the M3 low-IDs investigation + claudex).

The lesson from M3: a flat ``has_ms2 & in_any_window`` denominator counts precursors that are not in the
sample (``abundance == 0``) or too faint to detect, so recall looks absurdly low. Recall is only
meaningful against precursors that are **present, in DIA coverage, and detectable**. This harness reports:

  - raw IDs, correct, and **FDP** (engine calls not matching truth ÷ calls) — the calibration check;
  - **recall over a hierarchy** of ever-stricter denominators (all → present → present+in-window →
    present+in-window+has-fragments → +above an abundance floor), so the drop-off is visible;
  - a **recall-vs-abundance-decile curve** — the interpretable "how deep into the dynamic range does the
    render/search reach" number, rather than one threshold.

Truth join: DiaNN ``Stripped.Sequence`` + ``Precursor.Charge`` ↔ the answer key's ``peptide_id`` (mapped
to sequence via ``peptides.parquet``) + ``charge``. The answer key carries ``abundance``, ``has_ms2``,
``in_any_window`` (emitted by ``timsim-render-thermo``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _diann_ids(report_path: str, q_threshold: float) -> set[tuple[str, int]]:
    d = pq.read_table(report_path).to_pandas()
    seqcol = "Stripped.Sequence" if "Stripped.Sequence" in d.columns else next(c for c in d.columns if "Stripped" in c)
    qcol = "Q.Value" if "Q.Value" in d.columns else next(c for c in d.columns if c.endswith("Q.Value"))
    d = d[d[qcol] <= q_threshold]
    return set(zip(d[seqcol].astype(str), d["Precursor.Charge"].astype(int)))


def score_thermo_dia(report_path, truth_path, peptides_path, q_threshold=0.01, abundance_floor=1e-3):
    diann = _diann_ids(report_path, q_threshold)

    truth = pq.read_table(truth_path).to_pandas()
    pep = pq.read_table(peptides_path, columns=["peptide_id", "sequence"]).to_pandas()
    truth = truth.merge(pep, on="peptide_id", how="left")
    truth["seq"] = truth["sequence"].astype(str)
    truth["key"] = list(zip(truth["seq"], truth["charge"].astype(int)))

    all_keys = set(truth["key"])
    correct = diann & all_keys
    false = diann - all_keys
    fdp = len(false) / max(1, len(diann))

    # Hierarchical denominators (each strictly narrower than the last).
    def denom(mask):
        return set(truth.loc[mask, "key"])

    present = truth["abundance"] > 0
    inwin = present & truth["in_any_window"]
    withms2 = inwin & truth["has_ms2"]
    detect = withms2 & (truth["abundance"] > abundance_floor)
    levels = [
        ("all answer-key precursors", denom(pd.Series(True, index=truth.index))),
        ("present (abundance>0)", denom(present)),
        ("present & in a DIA window", denom(inwin)),
        ("present & in-window & has fragments", denom(withms2)),
        (f"present & in-window & has-frags & abundance>{abundance_floor:g}", denom(detect)),
    ]
    hierarchy = []
    for name, dset in levels:
        found = len(diann & dset)
        hierarchy.append({"denominator": name, "size": len(dset), "found": found,
                          "recall": found / max(1, len(dset))})

    # Recall vs abundance decile, over the detectable set (the interpretable curve).
    det = truth[detect].copy()
    curve = []
    if len(det) >= 10:
        det["decile"] = pd.qcut(det["abundance"].rank(method="first"), 10, labels=False)
        det["found"] = [k in diann for k in det["key"]]
        for dec, g in det.groupby("decile"):
            curve.append({"decile": int(dec) + 1,
                          "abundance_median": float(g["abundance"].median()),
                          "n": int(len(g)), "recall": float(g["found"].mean())})

    return {
        "diann_ids": len(diann),
        "correct": len(correct),
        "false": len(false),
        "fdp": fdp,
        "q_threshold": q_threshold,
        "abundance_floor": abundance_floor,
        "hierarchy": hierarchy,
        "recall_by_abundance_decile": curve,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="timsim-v2-thermo-eval",
                                 description="score DiaNN DIA search of a synthetic Astral .raw vs the answer key")
    ap.add_argument("--report", required=True, type=Path, help="DiaNN report.parquet")
    ap.add_argument("--truth", required=True, type=Path, help="render's thermo answer key parquet")
    ap.add_argument("--peptides", required=True, type=Path, help="peptides.parquet (peptide_id -> sequence)")
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--abundance-floor", type=float, default=1e-3,
                    help="detectability floor on abundance for the strictest denominator")
    ap.add_argument("--out", type=Path, help="write metrics JSON here")
    a = ap.parse_args(argv)

    m = score_thermo_dia(str(a.report), str(a.truth), str(a.peptides), q_threshold=a.fdr,
                         abundance_floor=a.abundance_floor)
    print("timsim v2 Thermo Astral DIA eval — DiaNN vs answer key")
    print(f"  DiaNN IDs (q<={a.fdr}): {m['diann_ids']:,}   correct {m['correct']:,} / false {m['false']:,}")
    print(f"  FDP: {m['fdp']*100:.2f}%   (engine calls not matching truth)")
    print("  recall over a hierarchy of denominators (present -> detectable):")
    for h in m["hierarchy"]:
        print(f"    {h['recall']*100:5.1f}%  ({h['found']:,}/{h['size']:,})  {h['denominator']}")
    if m["recall_by_abundance_decile"]:
        print("  recall by abundance decile (detectable set; 1=lowest .. 10=highest):")
        cells = "  ".join(f"D{c['decile']}:{c['recall']*100:.0f}%" for c in m["recall_by_abundance_decile"])
        print(f"    {cells}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(m, indent=2))
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
