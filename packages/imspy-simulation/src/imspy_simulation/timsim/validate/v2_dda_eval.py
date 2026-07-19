"""``timsim-v2-dda-eval`` — score a **Sage** DDA-PASEF search against timsim v2's answer key.

The DDA analogue of ``v2_eval`` (which scores DiaNN for DIA). The render emits a per-event answer key
(``dda_selected.parquet``: one row per PASEF selection, with `tdf_precursor_id`, `peptide_id`, `charge`,
locator). Sage searches the ``.d`` and reports PSMs by a 0-based spectrum ``scannr``. The empirically
established locator mapping (validated 89/89 on gen10k) is::

    Sage scannr + 1  ==  our tdf_precursor_id

(Sage's timsTOF reader aggregates a precursor's PASEF bands into one spectrum and numbers spectra 0-based
in ``Precursors.Id`` order; our ids are 1-based and contiguous.) A PSM is CORRECT if that precursor exists
in the answer key AND Sage's `(stripped sequence, charge)` matches the truth for it.

Metrics (see ``timsim-cli/DDA_PLAN.md``): **conditional ID recall** = correctly identified ÷ selected
precursors (what DDA chose to fragment), and **FDP** = engine calls not matching truth ÷ calls. Selection
recall (selected ÷ eligible) needs the render to also emit the eligible candidate pool — a follow-up.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

_MOD = re.compile(r"\[[^\]]*\]")


def _strip(seq: str) -> str:
    return _MOD.sub("", seq).replace("(", "").replace(")", "")


def score_dda(sage_tsv, truth_parquet, peptides_parquet, q_threshold=0.01):
    """Return a metrics dict scoring a Sage run against the v2 DDA answer key."""
    truth = pq.read_table(truth_parquet).to_pandas()
    peptides = pq.read_table(peptides_parquet, columns=["peptide_id", "sequence"]).to_pandas()
    truth = truth.merge(peptides, on="peptide_id", how="left")
    # One canonical row per selected precursor (the answer key is per-event; identity is per precursor).
    per_prec = truth.drop_duplicates("tdf_precursor_id")[["tdf_precursor_id", "sequence", "charge"]]
    per_prec = per_prec.set_index("tdf_precursor_id")
    n_selected = truth["tdf_precursor_id"].nunique()

    sage = pd.read_csv(sage_tsv, sep="\t")
    # Targets only (label == 1), passing the q-value cutoff (spectrum-level).
    sage = sage[sage["label"] == 1]
    if "spectrum_q" in sage.columns:
        sage = sage[sage["spectrum_q"] <= q_threshold]
    sage = sage.copy()
    sage["seq"] = sage["peptide"].map(_strip)
    sage["tdf"] = sage["scannr"] + 1  # the locator mapping

    n_psm = len(sage)
    correct = 0
    for r in sage.itertuples():
        t = per_prec.loc[r.tdf] if r.tdf in per_prec.index else None
        if t is not None and r.seq == t["sequence"] and int(r.charge) == int(t["charge"]):
            correct += 1

    correct_precursors = {r.tdf for r in sage.itertuples()
                          if r.tdf in per_prec.index
                          and r.seq == per_prec.loc[r.tdf]["sequence"]
                          and int(r.charge) == int(per_prec.loc[r.tdf]["charge"])}
    return {
        "selected_precursors": int(n_selected),
        "sage_psms": int(n_psm),
        "correct_psms": int(correct),
        "false_psms": int(n_psm - correct),
        "conditional_id_recall": correct / n_selected if n_selected else 0.0,
        "precursor_id_recall": len(correct_precursors) / n_selected if n_selected else 0.0,
        "fdp": (n_psm - correct) / n_psm if n_psm else 0.0,
        "q_threshold": q_threshold,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="timsim-v2-dda-eval",
                                 description="score a Sage DDA search against timsim v2 answer keys")
    ap.add_argument("--sage", required=True, type=Path, help="Sage results.sage.tsv")
    ap.add_argument("--truth", required=True, type=Path, help="render's dda_selected.parquet answer key")
    ap.add_argument("--peptides", required=True, type=Path, help="peptides.parquet (peptide_id -> sequence)")
    ap.add_argument("--fdr", type=float, default=0.01, help="spectrum q-value cutoff")
    ap.add_argument("--out", type=Path, help="write metrics JSON here")
    a = ap.parse_args(argv)

    m = score_dda(str(a.sage), str(a.truth), str(a.peptides), q_threshold=a.fdr)
    print("timsim v2 DDA eval — Sage vs answer key")
    print(f"  selected precursors : {m['selected_precursors']:,}   (what DDA fragmented)")
    print(f"  Sage PSMs (q<={a.fdr}) : {m['sage_psms']:,}")
    print(f"  correct / false     : {m['correct_psms']:,} / {m['false_psms']:,}")
    print(f"  conditional ID recall: {m['conditional_id_recall']*100:.1f}%   (correct PSMs / selected)")
    print(f"  precursor ID recall  : {m['precursor_id_recall']*100:.1f}%   (distinct precursors identified / selected)")
    print(f"  FDP                  : {m['fdp']*100:.2f}%   (false PSMs / PSMs)")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(m, indent=2))
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
