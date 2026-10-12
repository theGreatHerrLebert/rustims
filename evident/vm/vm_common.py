"""Shared loaders and sequence keys for the validity-matrix analysis.

Implements the frozen rules in evident/claims/protocol.yaml. Change a rule only through a new claim id.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

Q_MAX = 0.01  # vm.protocol.estimand: run-level Q.Value <= 0.01

_MOD = re.compile(r"[\(\[]UN[IL]MOD:(\d+)[\)\]]", re.IGNORECASE)  # UnLMod: I->L applied to a whole string
_LEFTOVER = re.compile(r"[\(\)\[\]]")
ENTRAP_TAG = "_p_target"
_ANY_MOD = re.compile(r"\{\d+\}")
_COMPLEXITY = re.compile(r"-(\d+)k-r(\d+)$")


def mod_key(seq: str) -> str:
    """One key for DIA-NN `(UniMod:4)` and TimSim `[UNIMOD:4]` notation, I->L on residues only."""
    k = _MOD.sub(r"{\1}", seq).replace("I", "L")
    if _LEFTOVER.search(k):
        raise ValueError(f"unrecognised modification token in {seq!r}")
    return k


def stripped_key(seq: str) -> str:
    return _ANY_MOD.sub("", mod_key(seq))


def read_pairs(pep_txt: Path) -> dict[str, str]:
    """FDRBench pair file -> {I->L stripped sequence: 'target' | 'p_target'}."""
    p = pd.read_csv(pep_txt, sep="\t", usecols=["sequence", "peptide_type"])
    p["sequence"] = p["sequence"].str.replace("I", "L")
    bad = set(p["peptide_type"]) - {"target", "p_target"}
    if bad:
        raise ValueError(f"unexpected peptide_type values {bad}")
    conflict = p.groupby("sequence")["peptide_type"].nunique()
    if (conflict > 1).any():
        raise ValueError(f"{int((conflict > 1).sum())} sequences are both target and entrapment after I->L")
    return dict(zip(p["sequence"], p["peptide_type"]))


def is_entrapment(stripped: pd.Series, pairs: dict[str, str], protein: pd.Series | None = None) -> np.ndarray:
    """Entrapment by the FDRBench pair file. Sequences missing from it fall back to the `_p_target`
    protein tag. A pair-file label that contradicts the protein tag is an error."""
    lab = stripped.map(pairs)
    if protein is None:
        if lab.isna().any():
            raise ValueError(f"{int(lab.isna().sum())} accepted sequences missing from the pair file and no protein column")
        return lab.eq("p_target").to_numpy()
    tag = protein.astype(str).str.contains(ENTRAP_TAG, regex=False)
    known = lab.notna()
    # A target peptide shared with a tagged entrapment protein is still a target. Only an entrapment
    # without the tag is impossible.
    clash = known & lab.eq("p_target") & ~tag
    if clash.any():
        raise ValueError(f"{int(clash.sum())} pair-file entrapments without the {ENTRAP_TAG} protein tag")
    return np.where(known, lab.eq("p_target"), tag).astype(bool)


def find_truth_db(sim_run_dir: Path) -> Path:
    hits = sorted((sim_run_dir / "out").rglob("synthetic_data.db"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one synthetic_data.db under {sim_run_dir}/out, found {len(hits)}")
    return hits[0]


def read_truth(db: Path) -> tuple[set, set]:
    """Simulated ions -> (modified keys, stripped keys), each a set of (key, charge)."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        ions = pd.read_sql("select sequence, charge from ions", c)
    mod = set(zip(ions["sequence"].map(mod_key), ions["charge"].astype(int)))
    stripped = {(_ANY_MOD.sub("", k), z) for k, z in mod}
    return mod, stripped


def accepted_from_report(report: Path) -> pd.DataFrame:
    """DIA-NN report -> accepted precursors (targets and entrapments, no engine decoys), one row each."""
    d = pd.read_parquet(report, columns=None)
    if "Decoy" in d.columns:
        d = d[d["Decoy"] == 0]
    d = d[d["Q.Value"] <= Q_MAX]
    out = pd.DataFrame({
        "mod": d["Modified.Sequence"].map(mod_key),
        "stripped": d["Stripped.Sequence"].str.replace("I", "L"),
        "charge": d["Precursor.Charge"].astype(int),
        "protein": d["Protein.Ids"].astype(str),
    })
    return out.drop_duplicates(["mod", "charge"]).reset_index(drop=True)


def label_false(df: pd.DataFrame, truth_mod: set, truth_stripped: set, entrap: np.ndarray) -> pd.DataFrame:
    """vm.protocol.truth-rule: false if entrapment, or a target whose (key, charge) was not simulated."""
    df = df.copy()
    df["entrap"] = entrap
    present_mod = np.fromiter(((k, z) in truth_mod for k, z in zip(df["mod"], df["charge"])), bool, len(df))
    present_str = np.fromiter(((k, z) in truth_stripped for k, z in zip(df["stripped"], df["charge"])), bool, len(df))
    df["false_mod"] = df["entrap"] | ~present_mod
    df["false_stripped"] = df["entrap"] | ~present_str
    return df


def fdp_stripped(lab: pd.DataFrame) -> float:
    """Secondary estimand: one vote per (stripped sequence, charge), false if every row for it is false."""
    g = lab.groupby(["stripped", "charge"])["false_stripped"].all()
    return float(g.mean()) if len(g) else float("nan")


def fdrbench_at_threshold(csv: Path) -> tuple[pd.DataFrame, dict]:
    """FDRBench precursor output -> (accepted rows, estimates at the 1% cut).

    Estimates are cumulative, so they are read on the accepted row with the largest n_t + n_p.
    This avoids taking a row that only looks last because of q-value ties.
    """
    f = pd.read_csv(csv)
    acc = f[f["q_value"] <= Q_MAX].copy()
    if acc.empty:
        raise ValueError(f"no accepted rows in {csv}")
    n = acc["n_t"] + acc["n_p"]
    ends = acc[n == n.max()]
    cols = ["lower_bound_fdp", "combined_fdp", "paired_fdp"]
    if len(ends.drop_duplicates(cols)) > 1:
        raise ValueError(f"tied end rows disagree on estimates in {csv}")
    last = ends.iloc[0]
    if int(n.max()) != len(acc):
        raise ValueError(f"{csv}: {len(acc)} accepted rows but n_t + n_p = {int(n.max())}")
    est = {
        "n_t": int(last["n_t"]),
        "n_p": int(last["n_p"]),
        "lower": float(last["lower_bound_fdp"]),
        "combined": float(last["combined_fdp"]),
        "paired": float(last["paired_fdp"]),
    }
    acc = pd.DataFrame({
        "mod": acc["mod_peptide"].map(mod_key),
        "stripped": acc["peptide"].str.replace("I", "L"),
        "charge": acc["charge"].astype(int),
        "protein": acc["protein"].astype(str),
    })
    if acc.duplicated(["mod", "charge"]).any():
        raise ValueError(f"{csv}: duplicate precursors among accepted rows")
    if not all(np.isfinite(v) for v in (est["lower"], est["combined"], est["paired"])):
        raise ValueError(f"{csv}: non-finite estimate at the 1% cut")
    return acc.reset_index(drop=True), est


REAL_RUNS = ["LS10", "LS11", "LS12"]
SIM_COMPLEXITIES = [50000, 125000, 250000]
SEEDS_PER_COMPLEXITY = 3


def check_cohort(recipe: Path) -> tuple[list[str], list[str]]:
    """The frozen cohort: real LS10/LS11/LS12 and sims at 50k/125k/250k with 3 seeds each. Anything else fails."""
    real = sorted(p.name for p in (recipe / "real10").iterdir() if p.is_dir())
    sim = sorted(p.name for p in (recipe / "sim").iterdir() if p.is_dir())
    if real != REAL_RUNS:
        raise ValueError(f"real runs {real} != {REAL_RUNS}")
    counts = pd.Series([complexity_of(r) for r in sim]).value_counts().to_dict()
    if counts != {c: SEEDS_PER_COMPLEXITY for c in SIM_COMPLEXITIES}:
        raise ValueError(f"sim cohort {counts} != 3 seeds x {SIM_COMPLEXITIES}")
    return real, sim


def complexity_of(run_id: str) -> int:
    m = _COMPLEXITY.search(run_id)
    if not m:
        raise ValueError(f"cannot parse complexity from run id {run_id!r}")
    return int(m.group(1)) * 1000


def verdict(lower: float, paired: float) -> str:
    """Wen et al. 2025 class at 1%, paired estimate as the upper bound."""
    if paired <= Q_MAX:
        return "valid"
    if lower > Q_MAX:
        return "invalid"
    return "inconclusive"


def modal(classes: list[str]) -> str:
    """Most common class. A tie counts as inconclusive (vm claims)."""
    s = pd.Series(classes).value_counts()
    if len(s) > 1 and s.iloc[0] == s.iloc[1]:
        return "inconclusive"
    return str(s.index[0])


def finite(x: float, what: str) -> float:
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"non-finite {what}")
    return x


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dump(obj, path: Path) -> None:
    """Strict JSON: NaN and infinity become null."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, sort_keys=True, allow_nan=False) + "\n")
