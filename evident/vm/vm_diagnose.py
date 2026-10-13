"""Descriptive diagnostics for the baseline sims. Reported only: no claim, tolerance or verdict reads these.

  depth      which simulated precursors DIA-NN accepts, by simulated intensity decile, charge, RT and m/z
  met-ox     anatomy of the false hits that differ from a simulated precursor only in Met oxidation
  entrap     true FDP split into error classes; entrapment can only see wrong-sequence errors
  q-curve    true FDP against DIA-NN's run-level Q.Value cut, 0.1% to 5%
  never-sim  wrong-sequence false hits: nearest simulated precursor m/z, anagrams, Hamming neighbours

Uses the frozen truth rule (vm_common.label_false). Writes <out>/results/diagnose.json.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from vm_common import (Q_MAX, _ANY_MOD, check_cohort, complexity_of, dump, fdp_stripped, fdrbench_at_threshold,
                       find_truth_db, is_entrapment, label_false, mod_key, read_pairs)

OX = "{35}"
Q_GRID = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
SEARCH_Q = 0.05  # --qvalue of the recipe-v1 searches; reports hold nothing above it
RT_BIN_MIN = 10.0
MZ_BINS = (0, 400, 500, 600, 700, 800, 1000, 1200, np.inf)
SCORE_COLS = ("Q.Value", "PEP", "Evidence", "Mass.Evidence", "Ms1.Profile.Corr", "Ms1.Area")


def read_truth_table(db: Path) -> pd.DataFrame:
    """One row per simulated (modified key, charge): summed intensity, peptide apex RT (min), m/z, IM."""
    q = """select i.sequence, i.charge, i.mz, i.inv_mobility_gru_predictor as im,
                  i.relative_abundance * p.events as intensity, p.retention_time_gru_predictor as rt
           from ions i join peptides p on i.peptide_id = p.peptide_id"""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        t = pd.read_sql(q, c)
    if t.empty or t[["mz", "im", "intensity", "rt"]].isna().any().any():
        raise ValueError(f"{db}: empty truth table or missing values")
    t["mod"] = t["sequence"].map(mod_key)
    t["charge"] = t["charge"].astype(int)
    t["rt_min"] = t["rt"] / 60.0  # TimSim stores the apex in seconds
    # I->L can merge simulated ions into one key: intensity is summed, coordinates averaged, and the spread is kept.
    t = t.groupby(["mod", "charge"], as_index=False).agg(
        intensity=("intensity", "sum"), rt_min=("rt_min", "mean"), mz=("mz", "mean"), im=("im", "mean"),
        n_ions=("rt_min", "size"), rt_spread_min=("rt_min", lambda x: float(x.max() - x.min())))
    t["stripped"] = t["mod"].str.replace(_ANY_MOD, "", regex=True)
    return t


def read_report(report: Path, q_max: float = Q_MAX) -> pd.DataFrame:
    """Target precursors of a DIA-NN report at Q.Value <= q_max, best row per (mod, charge)."""
    d = pd.read_parquet(report)
    if "Decoy" in d.columns:
        d = d[d["Decoy"] == 0]
    d = d[d["Q.Value"] <= q_max].sort_values("Q.Value")
    d = d.assign(mod=d["Modified.Sequence"].map(mod_key), stripped=d["Stripped.Sequence"].str.replace("I", "L"),
                 charge=d["Precursor.Charge"].astype(int), protein=d["Protein.Ids"].astype(str))
    return d.drop_duplicates(["mod", "charge"]).reset_index(drop=True)


def truth_sets(t: pd.DataFrame) -> tuple[set, set]:
    return set(zip(t["mod"], t["charge"])), set(zip(t["stripped"], t["charge"]))


def classify(lab: pd.DataFrame, truth_mod: set) -> pd.DataFrame:
    """Error class per accepted row: true, entrapment, never-simulated sequence, sequence simulated only at
    another charge, Met-ox form confusion, other modification-form confusion."""
    sim_seqs = {_ANY_MOD.sub("", k) for k, _ in truth_mod}
    no_ox = {(k.replace(OX, ""), z) for k, z in truth_mod}
    cls = np.full(len(lab), "true", dtype=object)
    form = lab["false_mod"] & ~lab["false_stripped"]
    met = form & np.fromiter(((k.replace(OX, ""), z) in no_ox for k, z in zip(lab["mod"], lab["charge"])), bool, len(lab))
    seq_known = lab["stripped"].isin(sim_seqs).to_numpy()
    cls[form & ~met] = "form-other"
    cls[met] = "form-met-ox"
    cls[lab["false_stripped"] & ~lab["entrap"] & seq_known] = "seq-other-charge"
    cls[lab["false_stripped"] & ~lab["entrap"] & ~seq_known] = "seq-never-simulated"
    cls[lab["entrap"]] = "entrapment"
    return lab.assign(cls=cls)


def recall_by(t: pd.DataFrame, hit: np.ndarray, by: pd.Series) -> dict:
    g = pd.DataFrame({"by": by.astype(str), "hit": hit}).groupby("by", observed=True, sort=False)["hit"]
    return {k: {"n": int(n), "recall": float(r)} for (k, n), r in zip(g.size().items(), g.mean())}


def depth(t: pd.DataFrame, lab: pd.DataFrame) -> dict:
    true = set(zip(lab.loc[~lab["false_mod"], "mod"], lab.loc[~lab["false_mod"], "charge"]))
    hit = np.fromiter(((k, z) in true for k, z in zip(t["mod"], t["charge"])), bool, len(t))
    pos = t["intensity"] > 0
    dec = pd.Series("zero", index=t.index, dtype=object)
    if pos.any():  # deciles by rank, so ties and constant intensities still split into ten bins
        r = t.loc[pos, "intensity"].rank(method="first").to_numpy()
        dec[pos] = [f"d{int(i):02d}" for i in np.ceil(r / len(r) * 10).clip(1, 10)]
    rt_bin = (np.floor(t["rt_min"] / RT_BIN_MIN) * RT_BIN_MIN).astype(int)
    return {"simulated": len(t), "merged_keys": int((t["n_ions"] > 1).sum()),
            "merged_rt_spread_max_min": float(t["rt_spread_min"].max()), "simulated_zero_intensity": int((~pos).sum()), "accepted_true": int(hit.sum()),
            "recall": float(hit.mean()),
            "by_intensity_decile": recall_by(t, hit, dec), "by_charge": recall_by(t, hit, t["charge"]),
            "by_rt_bin_min": recall_by(t, hit, rt_bin), "by_mz": recall_by(t, hit, pd.cut(t["mz"], MZ_BINS)),
            "accepted_true_intensity_decile_share": pd.Series(dec[hit]).value_counts(normalize=True).sort_index().to_dict()}


def score_medians(d: pd.DataFrame) -> dict:
    return {c: float(d[c].median()) if c in d.columns and len(d) else None for c in SCORE_COLS}


def met_ox(t: pd.DataFrame, rep: pd.DataFrame, lab: pd.DataFrame) -> dict:
    """False Met-ox hits against their twin: a simulated form at the same charge that differs only in Met oxidation.

    A hit can have several twins (oxidised at different positions); the twin's apex and IM are their mean, and the
    count with more than one twin is reported. The hit itself is never its own twin (it is not simulated).
    """
    deox = lambda k: k.replace(OX, "")  # noqa: E731
    twins: dict[tuple, list] = {}
    for k, z in zip(t["mod"], t["charge"]):
        twins.setdefault((deox(k), z), []).append((k, z))
    tt = t.set_index(["mod", "charge"])
    acc = rep.set_index(["mod", "charge"])
    m = lab[lab["cls"] == "form-met-ox"]
    keys = list(zip(m["mod"], m["charge"]))
    cand = [[c for c in twins.get((deox(k), z), []) if c != (k, z)] for k, z in keys]
    if any(not c for c in cand):
        raise ValueError("a form-met-ox hit without a simulated twin")
    nan = float("nan")
    twin_apex = np.array([tt.loc[c, "rt_min"].mean() for c in cand], float)
    twin_im = np.array([tt.loc[c, "im"].mean() for c in cand], float)
    twin_obs = [[x for x in c if x in acc.index] for c in cand]
    twin_rt_obs = np.array([acc.loc[o, "RT"].mean() if o else nan for o in twin_obs], float)
    hit_rt = acc.loc[keys, "RT"].to_numpy(float) if keys else np.array([])
    hit_im = acc.loc[keys, "IM"].to_numpy(float) if keys else np.array([])
    paired = np.isfinite(twin_rt_obs) & np.isfinite(hit_rt)
    false_groups = {(deox(k), z) for k, z in keys}
    true_m = lab[(lab["cls"] == "true") & lab["stripped"].str.contains("M")]
    q = lambda x: np.quantile(x[np.isfinite(x)], [0.05, 0.5, 0.95]).tolist() if np.isfinite(x).any() else [None] * 3  # noqa: E731
    sim_ox = t["mod"].str.contains(OX, regex=False)
    hit_ox = m["mod"].str.contains(OX, regex=False)
    return {
        "n": len(m), "hit_oxidised_share": float(hit_ox.mean()) if len(m) else None,
        "multiple_twins": int(sum(len(c) > 1 for c in cand)),
        "simulated_precursors_with_ox": int(sim_ox.sum()), "simulated_precursors": len(t),
        "accepted_share_with_ox": float(rep["mod"].str.contains(OX, regex=False).mean()),
        "twin_also_accepted": float(np.mean([bool(o) for o in twin_obs])) if keys else None,
        "hit_rt_minus_twin_apex_min_q05_q50_q95": q(hit_rt - twin_apex),
        "hit_im_minus_twin_im_q05_q50_q95": q(hit_im - twin_im),
        "paired_with_accepted_twin": int(paired.sum()),
        "abs_hit_rt_minus_twin_rt_min_q05_q50_q95": q(np.abs(hit_rt - twin_rt_obs)),
        "same_peak_share_0p1min_of_paired": float(np.mean(np.abs(hit_rt - twin_rt_obs)[paired] <= 0.1))
        if paired.any() else None,
        "true_m_containing_with_false_met_ox_sibling": float(np.mean(
            [(deox(k), z) in false_groups for k, z in zip(true_m["mod"], true_m["charge"])])) if len(true_m) else None,
        "scores_false_met_ox": score_medians(acc.loc[keys]) if keys else None,
        "scores_true": score_medians(acc.loc[list(zip(lab.loc[lab["cls"] == "true", "mod"],
                                                      lab.loc[lab["cls"] == "true", "charge"]))]),
    }


def never_sim(t: pd.DataFrame, rep: pd.DataFrame, lab: pd.DataFrame) -> dict:
    """Wrong-sequence false hits (not entrapment): how close is the nearest simulated precursor?"""
    m = lab[lab["cls"].isin(["seq-never-simulated", "seq-other-charge"])]
    acc = rep.set_index(["mod", "charge"])
    by_len: dict[int, list[str]] = {}
    for s in t["stripped"].unique():
        by_len.setdefault(len(s), []).append(s)
    anagrams = {"".join(sorted(s)) for s in t["stripped"].unique()}
    mz_by_z = {z: np.sort(g["mz"].to_numpy(float)) for z, g in t.groupby("charge")}
    rows = []
    for k, z, s, c in zip(m["mod"], m["charge"], m["stripped"], m["cls"]):
        mz = float(acc.loc[(k, z), "Precursor.Mz"])
        arr = mz_by_z.get(z, np.array([]))
        i = np.searchsorted(arr, mz)
        near = min((abs(arr[j] - mz) for j in (i - 1, i) if 0 <= j < len(arr)), default=np.inf)
        same = by_len.get(len(s), []) if c == "seq-never-simulated" else []
        ham = min((sum(a != b for a, b in zip(s, o)) for o in same), default=None)
        rows.append({"cls": c, "len": len(s), "nearest_same_charge_ppm": near / mz * 1e6,
                     "anagram_of_simulated": "".join(sorted(s)) in anagrams, "min_hamming_same_length": ham})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n": 0}
    out = {"n": len(df), "by_class": df["cls"].value_counts().to_dict()}
    for c, g in df.groupby("cls"):  # other-charge hits are simulated sequences: anagram and Hamming 0 by definition
        out[c] = {"n": len(g), "within_10ppm_share": float((g["nearest_same_charge_ppm"] <= 10).mean()),
                  "nearest_ppm_q50": float(g["nearest_same_charge_ppm"].median()),
                  "anagram_share": float(g["anagram_of_simulated"].mean()),
                  "hamming_le2_share": float((g["min_hamming_same_length"].fillna(99) <= 2).mean())}
    return out


def q_curve(report: Path, truth_mod: set, truth_stripped: set, pairs: dict, search_q: float) -> list:
    d = read_report(report, q_max=1.0)
    if float(d["Q.Value"].max()) > search_q:
        raise ValueError(f"{report}: Q.Value above the search cut {search_q}")
    out = []
    for q in Q_GRID:
        if q > search_q:
            out.append({"q": q, "covered": False})
            continue
        s = d[d["Q.Value"] <= q]
        lab = classify(label_false(s, truth_mod, truth_stripped, is_entrapment(s["stripped"], pairs, s["protein"])),
                       truth_mod)
        out.append({"q": q, "covered": True, "accepted": len(lab), "fdp": float(lab["false_mod"].mean()),
                    "fdp_stripped": fdp_stripped(lab), "met_ox_fdp": float((lab["cls"] == "form-met-ox").mean())})
    return out


def entrap_classes(recipe: Path, run: str, truth_mod: set, truth_stripped: set, pairs: dict) -> dict:
    acc, est = fdrbench_at_threshold(recipe / "sim" / run / "entrap" / "fdp_precursor.csv")
    ent = is_entrapment(acc["stripped"], pairs, acc["protein"])
    lab = classify(label_false(acc, truth_mod, truth_stripped, ent), truth_mod)
    share = lab["cls"].value_counts(normalize=True).to_dict()
    # Shares of the accepted set per error class, beside FDRBench's estimates. The estimates are model-based, not
    # class counts; this only shows which classes are large enough to explain the gap to the true FDP.
    f = lambda *c: float(lab["cls"].isin(c).mean())  # noqa: E731
    return {"accepted": len(lab), "true_fdp": float(lab["false_mod"].mean()),
            "fdp_entrapment_or_never_simulated": f("entrapment", "seq-never-simulated"),
            "fdp_other_charge": f("seq-other-charge"), "fdp_form": f("form-met-ox", "form-other"),
            "paired": est["paired"], "combined": est["combined"], "lower": est["lower"], "class_shares": share}


def run_sim(recipe: Path, sim_root: Path, run: str, pairs: dict, search_q: float) -> dict:
    t = read_truth_table(find_truth_db(sim_root / run))
    tm, ts = truth_sets(t)
    report = recipe / "sim" / run / "plain" / "report.parquet"
    rep = read_report(report)
    ent = is_entrapment(rep["stripped"], pairs, rep["protein"])
    if ent.any():
        raise ValueError(f"{run}: entrapment precursors in a plain-library search")
    lab = classify(label_false(rep, tm, ts, ent), tm)
    return {"complexity": complexity_of(run), "accepted": len(lab), "fdp": float(lab["false_mod"].mean()),
            "class_counts": lab["cls"].value_counts().to_dict(),
            "depth": depth(t, lab), "met_ox": met_ox(t, rep, lab), "never_simulated": never_sim(t, rep, lab),
            "q_curve": q_curve(report, tm, ts, pairs, search_q), "entrapment_classes": entrap_classes(recipe, run, tm, ts, pairs)}


def run_real(recipe: Path, run: str, search_q: float) -> dict:
    rep = read_report(recipe / "real10" / run / "plain" / "report.parquet")
    full = read_report(recipe / "real10" / run / "plain" / "report.parquet", q_max=1.0)
    return {"accepted": len(rep), "accepted_share_with_ox": float(rep["mod"].str.contains(OX, regex=False).mean()),
            "m_containing_share": float(rep["stripped"].str.contains("M").mean()),
            "accepted_by_q": {q: int((full["Q.Value"] <= q).sum()) for q in Q_GRID if q <= search_q},
            "scores_accepted": score_medians(rep)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=Path, required=True, help="searches/vm/recipe-v1")
    ap.add_argument("--sim-root", type=Path, required=True, help="searches/vm/sim-baseline")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--search-q", type=float, default=SEARCH_Q, help="the searches' --qvalue")
    a = ap.parse_args()
    pairs = read_pairs(a.recipe / "entrap" / "human_entrap_pep.txt")
    real, sim = check_cohort(a.recipe)
    res = {"reported_only": True,
           "real": {r: run_real(a.recipe, r, a.search_q) for r in real},
           "sim": {r: run_sim(a.recipe, a.sim_root, r, pairs, a.search_q) for r in sim}}
    dump(res, a.out / "results" / "diagnose.json")
    for r, v in res["sim"].items():
        print(r, "recall", round(v["depth"]["recall"], 4), "fdp", round(v["fdp"], 4), v["class_counts"])


if __name__ == "__main__":
    main()
