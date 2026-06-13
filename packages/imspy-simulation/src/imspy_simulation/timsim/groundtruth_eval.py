"""Ground-truth evaluation of a simulated run against a search-engine report.

Given a TimSim ``synthetic_data.db`` (the ground truth: exactly which peptides, at
which charges, retention times, and abundances were simulated) and a DIA search report
(DiaNN or alphaDIA), compute the validation metrics:

- **RT correlation**     — reported retention time vs simulated apex RT (Pearson +
  median absolute error). Tests that the render places elution at the intended RT.
- **Charge accuracy**    — fraction of identified precursors whose charge was actually
  simulated for that peptide backbone.
- **FDP** (false discovery proportion) — identified backbones NOT in the simulated set
  / total identified. This is the *realized* false fraction measured against the
  COMPLETE ground truth (we know exactly what was simulated) — NOT an FDR estimate, and
  NOT entrapment. (FDR is the expected FDP an estimator targets; entrapment estimates
  FDR by spiking known non-target sequences as a separate axis — neither is needed here
  since the truth is known.) Meaningful only when the search space exceeds the simulated
  set (library-free / full proteome); assumes the run has no blank/noise-derived real
  IDs (true today — only simulated-peptide signal is rendered).
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


# ----------------------------------------------------------------------------
# Precursor-level FDP (peptidoform-aware). The bare-backbone metrics above are a
# diagnostic; FDR is controlled per peptidoform+charge, so the realized FDP must
# be measured on a canonical modified-sequence key consistent across the truth DB
# (UNIMOD brackets), DiaNN (`(UniMod:n)` in Modified.Sequence) and alphaDIA
# (stripped sequence + `mods` names + `mod_sites`). Validated: the same
# peptidoform yields an identical key from all three sources.
# ----------------------------------------------------------------------------

# alphaDIA mod NAME (before '@') -> UNIMOD id.
_MOD_NAME_TO_UNIMOD = {
    "Oxidation": 35, "Carbamidomethyl": 4, "Acetyl": 1,
    "Phospho": 21, "Deamidated": 7, "GlyGly": 121,
}
_INLINE_MOD = re.compile(r"[\(\[](?:UniMod|UNIMOD):(\d+)[\)\]]", re.I)


def _il_norm(s: str) -> str:
    """Collapse the isobaric I/L ambiguity (I->L) so mass-equal calls don't count
    as backbone-distinct."""
    return s.replace("I", "L")


def _canon_inline(seq: str) -> tuple:
    """Parse inline UNIMOD notation — ``M[UNIMOD:35]K`` (truth) or
    ``M(UniMod:35)K`` (DiaNN) — into ``(stripped, {pos_1based: unimod_id})``.
    A mod token before any residue is N-term (position 0)."""
    mods, stripped, pos, i = {}, [], 0, 0
    while i < len(seq):
        m = _INLINE_MOD.match(seq, i)
        if m:
            mods[pos] = int(m.group(1))
            i = m.end()
        elif seq[i].isalpha():
            pos += 1
            stripped.append(seq[i])
            i += 1
        else:
            i += 1
    return "".join(stripped), mods


def _canon_alphadia(stripped: str, mods_str, sites_str) -> tuple:
    """alphaDIA (stripped, ``mods`` names, ``mod_sites`` 1-based) ->
    ``(stripped, {pos: unimod_id})``. Unknown mod names map to -1 (a sentinel that
    cannot match a truth id, so it counts as a peptidoform mismatch rather than
    silently matching)."""
    mods = {}
    if isinstance(mods_str, str) and mods_str:
        for nm, st in zip(mods_str.split(";"), str(sites_str).split(";")):
            uid = _MOD_NAME_TO_UNIMOD.get(nm.split("@")[0], -1)
            try:
                mods[int(st)] = uid
            except ValueError:
                continue
    return stripped, mods


def _pf_key(stripped: str, mods: Dict[int, int], il: bool = True) -> str:
    s = _il_norm(stripped) if il else stripped
    return s + "/" + ";".join(f"{p}:{i}" for p, i in sorted(mods.items()))


def _bb_key(stripped: str, il: bool = True) -> str:
    return _il_norm(stripped) if il else stripped


@dataclass
class GroundTruthMetrics:
    identified: int                      # identified precursors matched to a simulated backbone+charge
    reported_total: int                  # total precursors in the report
    simulated_ions: int                  # simulated (target) precursor ions
    rt_pearson: float
    rt_median_abs_error: float           # in the report's RT unit (minutes)
    charge_accuracy: float               # of matched-backbone IDs, fraction at a simulated charge
    fdp: float                           # false discovery PROPORTION: backbones not simulated / reported_total
    n_false_discoveries: int             # reported IDs whose backbone was never simulated
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
        # alphaDIA `precursors.parquet`: dotted columns (precursor.sequence [stripped],
        # precursor.charge, precursor.rt.observed [seconds], precursor.intensity [often
        # NaN in the precursor table — quant lives in precursor.matrix], precursor.decoy,
        # precursor.qval). Fall back to bare names for older/other layouts.
        r = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, sep="\t")
        col = lambda *names: next((c for c in names if c in r.columns), None)
        seq_c = col("precursor.sequence", "sequence", "Sequence", "stripped_sequence")
        chg_c = col("precursor.charge", "charge")
        rt_c = col("precursor.rt.observed", "precursor.rt.calibrated", "rt_observed", "rt")
        q_c = col("precursor.intensity", "intensity", "quantity")
        dec_c = col("precursor.decoy", "decoy")
        qv_c = col("precursor.qval", "qval", "q_value")
        if seq_c is None or chg_c is None:
            raise ValueError(f"alphaDIA report missing sequence/charge columns; saw {list(r.columns)[:12]}")
        if dec_c is not None:
            r = r[r[dec_c] == 0]
        if qv_c is not None:
            r = r[r[qv_c] <= 0.01]
        rt = r[rt_c].astype(float) if rt_c else pd.Series(np.nan, index=r.index)
        if rt_c is not None and len(rt) and rt.max() > 200:   # seconds -> minutes
            rt = rt / 60.0
        return pd.DataFrame({
            "bb": r[seq_c].map(_strip_mods),
            "charge": r[chg_c].astype(int),
            "rt": rt.values,
            "quantity": r[q_c].astype(float).values if q_c else np.full(len(r), np.nan),
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
    # FDP: a reported peptide whose backbone was never simulated is a realized false
    # discovery (its signal is not in the .raw). NOTE: assumes no blank/noise-derived
    # real IDs (true while only simulated signal is rendered). I/L isobars make this a
    # mild upper bound (an L-variant of a simulated I-peptide is mass-equal but counts
    # false under exact string match).
    false_bb = [b not in sim_bb for b in report["bb"]]
    fdp = float(np.mean(false_bb)) if n else 0.0

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
        fdp=fdp,
        n_false_discoveries=int(sum(false_bb)),
        quant_log_pearson=quant_p,
        quant_log_spearman=quant_s,
        quant_dynamic_range_orders=dyn,
        recall_by_abundance_quartile=recall_q,
    )


def evaluate_run(db_path: str, report_path: str, engine: str = "diann") -> GroundTruthMetrics:
    """Convenience: load DB + report and evaluate."""
    return evaluate(load_simulated_truth(db_path), load_report(report_path, engine))


def load_precursor_truth(db_path: str, il_normalize: bool = True) -> Dict[str, object]:
    """Rendered-observable simulated precursors as canonical keys. 'Rendered-observable'
    = a non-decoy ion that has >=1 fragment row (it was actually fragmented/rendered),
    not mere DB membership. Returns sets of (backbone, charge) and (peptidoform, charge),
    plus a {(backbone,charge): set(mod-id-multiset)} map to separate wrong-site
    localization (right backbone + right mod composition, wrong position — unresolvable)
    from genuine wrong-modification calls."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT DISTINCT i.sequence, i.charge FROM ions i "
        "JOIN fragment_ions f ON i.ion_id = f.ion_id "
        "JOIN peptides p ON i.peptide_id = p.peptide_id WHERE p.decoy = 0",
        con,
    )
    con.close()
    bb, pf = set(), set()
    bb_multiset: Dict[tuple, set] = {}
    for seq, ch in zip(df["sequence"], df["charge"]):
        st, mods = _canon_inline(seq)
        c = int(ch)
        bk = (_bb_key(st, il_normalize), c)
        bb.add(bk)
        pf.add((_pf_key(st, mods, il_normalize), c))
        bb_multiset.setdefault(bk, set()).add(tuple(sorted(mods.values())))
    return {"bb": bb, "pf": pf, "bb_multiset": bb_multiset, "n": len(df)}


def load_precursor_report(path: str, engine: str = "diann") -> pd.DataFrame:
    """Normalize a report to per-row canonical keys + q-value, WITHOUT a q-filter (the
    caller thresholds for the curve). Columns: bb, pf, charge, mod_multiset, q."""
    engine = engine.lower()
    rows = []
    if engine == "diann":
        d = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, sep="\t")
        for ms, ch, q in zip(d["Modified.Sequence"].astype(str),
                             d["Precursor.Charge"].astype(int), d["Q.Value"].astype(float)):
            st, mods = _canon_inline(ms)
            rows.append((_bb_key(st), _pf_key(st, mods), int(ch),
                         tuple(sorted(mods.values())), float(q)))
    elif engine == "alphadia":
        a = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, sep="\t")
        if "precursor.decoy" in a.columns:
            a = a[a["precursor.decoy"] == 0]
        for st, mo, si, ch, q in zip(a["precursor.sequence"].astype(str), a["precursor.mods"],
                                     a["precursor.mod_sites"], a["precursor.charge"].astype(int),
                                     a["precursor.qval"].astype(float)):
            _, mods = _canon_alphadia(st, mo, si)
            rows.append((_bb_key(st), _pf_key(st, mods), int(ch),
                         tuple(sorted(mods.values())), float(q)))
    else:
        raise ValueError(f"unknown engine '{engine}' (expected 'diann' or 'alphadia')")
    return pd.DataFrame(rows, columns=["bb", "pf", "charge", "mod_multiset", "q"])


def precursor_fdp_breakdown(truth: Dict[str, object], report: pd.DataFrame,
                            q_thresholds=(0.001, 0.005, 0.01, 0.02, 0.05)) -> list:
    """Realized precursor FDP at each q threshold, dedup on (peptidoform, charge), with
    the genuine / localization-ambiguous / wrong-backbone split. 'genuine' excludes
    unresolvable localization (right backbone + same mod-id multiset, wrong site)."""
    tbb, tpf, tms = truth["bb"], truth["pf"], truth["bb_multiset"]
    out = []
    for q in q_thresholds:
        r = report[report["q"] <= q].drop_duplicates(subset=["pf", "charge"])
        n = len(r)
        correct = wrong_bb = wrong_mod = loc_ambig = 0
        for bb, pf, ch, ms in zip(r["bb"], r["pf"], r["charge"], r["mod_multiset"]):
            if (pf, ch) in tpf:
                correct += 1
            elif (bb, ch) not in tbb:
                wrong_bb += 1
            elif ms in tms.get((bb, ch), set()):
                loc_ambig += 1          # right backbone + mod composition, wrong site
            else:
                wrong_mod += 1
        genuine = wrong_bb + wrong_mod
        bbr = report[report["q"] <= q].drop_duplicates(subset=["bb", "charge"])
        nbb = len(bbr)
        bb_false = sum((b, c) not in tbb for b, c in zip(bbr["bb"], bbr["charge"]))
        out.append({
            "q": q, "n_precursors": n,
            "fdp_backbone": bb_false / nbb if nbb else float("nan"),
            "fdp_peptidoform_strict": (n - correct) / n if n else float("nan"),
            "fdp_peptidoform_genuine": genuine / n if n else float("nan"),
            "localization_ambiguous": loc_ambig,
            "wrong_backbone": wrong_bb, "wrong_mod_composition": wrong_mod,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Ground-truth eval of a TimSim run vs a DIA search report.")
    ap.add_argument("--db", required=True, help="path to synthetic_data.db")
    ap.add_argument("--report", required=True, help="path to the search report (DiaNN parquet/tsv, or alphaDIA)")
    ap.add_argument("--engine", default="diann", choices=["diann", "alphadia"])
    ap.add_argument("--out", default=None, help="optional path to write metrics JSON")
    ap.add_argument("--precursor-fdp", action="store_true",
                    help="precursor-level FDP-vs-q (peptidoform-aware) with the genuine / "
                         "localization-ambiguous / wrong-backbone breakdown")
    a = ap.parse_args()
    if a.precursor_fdp:
        truth = load_precursor_truth(a.db)
        report = load_precursor_report(a.report, a.engine)
        bd = precursor_fdp_breakdown(truth, report)
        payload = {"rendered_observable_precursors": truth["n"],
                   "report_rows": int(len(report)), "engine": a.engine, "curve": bd}
        print(json.dumps(payload, indent=2))
        if a.out:
            with open(a.out, "w") as f:
                json.dump(payload, f, indent=2)
        return
    m = evaluate_run(a.db, a.report, a.engine)
    print(m.to_json())
    if a.out:
        with open(a.out, "w") as f:
            f.write(m.to_json())


if __name__ == "__main__":
    main()
