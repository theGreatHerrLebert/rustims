"""Build and verify the from_existing inputs for the predictor-swap, rt-spread and mass-accuracy stress rows
(evident/vm/stress_settings.yaml, stress_settings_rt_spread.yaml, stress_settings_mass_accuracy.yaml).

prepare (default): for one baseline sim, writes <out>/<setting>/synthetic_data.db holding only the tables a
  from_existing replay reads (peptides, ions, proteins, dia_ms_ms_info, dia_ms_ms_windows), with the RT and/or
  IM apex columns rewritten per setting, plus <out>/prepare.json with the guard statistics. Never touches the
  baseline DB.
--verify <replay db> --prepared <setting dir>: after a replay, checks that the replay's own truth DB carries the
  prepared apex values, so the intervention survived the replay. mass-accuracy also checks the m/z bias columns and,
  with --control-replay, that the replay kept the same ions as that seed's replay-control-ma.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SETTINGS = {"replay-control": (False, False), "rt-chronologer": (True, False), "im-std": (False, True), "joint": (True, True)}
RT_SPREAD = "rt-spread"
MIN_REAL_RT = 10_000        # distinct modified peptides in the real sample
MAX_NEW_TIES = 0.01         # share of distinct old RTs the map may merge into one value
KEEP = ("peptides", "ions", "proteins", "dia_ms_ms_info", "dia_ms_ms_windows")
UNIMOD_TO_CHRONOLOGER = {"[UNIMOD:4]": "[+57.02146]", "[UNIMOD:35]": "[+15.99491]"}
IM_SEED_OFFSET = 7


def to_chronologer(seq: str) -> str:
    out = seq
    if out.startswith("[UNIMOD:1]"):
        out = "[+42.01057]" + out[len("[UNIMOD:1]"):]
    for k, v in UNIMOD_TO_CHRONOLOGER.items():
        out = out.replace(k, v)
    if "UNIMOD" in out:
        raise ValueError(f"unmappable modification for Chronologer: {seq}")
    # Amendment 1 (stress_settings.yaml): Chronologer has no selenocysteine token; predict U as C.
    return out.replace("U", "C")


def chronologer_hi(seqs: list[str], model_file: Path, batch: int = 2048) -> np.ndarray:
    import torch
    from chronologer.chronologer_utils.tensorize import codedseq_to_array, modseq_to_codedseq
    from chronologer.model import initialize_chronologer_model

    coded = [modseq_to_codedseq(to_chronologer(s)) for s in seqs]
    bad = [s for s, c in zip(seqs, coded) if c is None]
    if bad:
        raise ValueError(f"{len(bad)} sequences Chronologer cannot encode, e.g. {bad[:3]}")
    arr = np.asarray([codedseq_to_array(c) for c in coded], dtype="int64")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = initialize_chronologer_model(str(model_file)).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(arr), batch):
            out.append(model(torch.from_numpy(arr[i:i + batch]).to(dev)).cpu().numpy()[:, 0])
    hi = np.concatenate(out).astype(float)
    if len(hi) != len(seqs) or not np.all(np.isfinite(hi)):
        raise ValueError("Chronologer returned missing or non-finite predictions")
    return hi


def min_max(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """TimSim's Koina RT rescaling: linear map of [min, max] onto [lo, hi]."""
    if not np.all(np.isfinite(x)) or x.max() <= x.min():
        raise ValueError("RT predictions are non-finite or constant")
    return lo + (x - x.min()) / (x.max() - x.min()) * (hi - lo)


def _ids(ids: list, what: str) -> None:
    if not ids or any(i is None for i in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{what}: empty, null or duplicate ids")


def _finite(x: np.ndarray, what: str) -> None:
    if len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError(f"{what}: empty or non-finite values")


def slim_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    with sqlite3.connect(dst) as c:
        c.execute("attach database ? as src", (f"file:{src}?mode=ro",))
        for t in KEEP:  # the baseline's own DDL, so column types match
            ddl = c.execute("select sql from src.sqlite_master where type = 'table' and name = ?", (t,)).fetchone()
            if ddl is None:
                raise SystemExit(f"baseline DB lacks table {t}")
            c.execute(ddl[0])
            c.execute(f"insert into main.{t} select * from src.{t}")
        # the tables carry no keys; without these the row updates below are table scans
        c.execute("create index main.vm_peptide_id on peptides(peptide_id)")
        c.execute("create index main.vm_ion_id on ions(ion_id)")
        c.commit()
        c.execute("detach database src")


def change_stats(old: np.ndarray, new: np.ndarray, width: np.ndarray) -> dict:
    """Size of the rewrite, absolute and relative to the per-row peak width (RT sigma or IM std)."""
    d = np.abs(new - old)
    return {"n": int(len(d)), "n_changed": int(np.sum(d > 0)), "median_abs_change": float(np.median(d)),
            "p95_abs_change": float(np.quantile(d, 0.95)), "median_abs_change_over_width": float(np.median(d / width))}


def real_rt_sample(report: Path) -> np.ndarray:
    """RT (s) of the real run's accepted targets (Q.Value <= 0.01, Decoy == 0), median per Modified.Sequence.

    One value per modified peptide (charge states pooled), matching the peptide-level apex it is mapped onto.
    """
    import pandas as pd
    d = pd.read_parquet(report, columns=["Modified.Sequence", "RT", "Q.Value", "Decoy"])
    d = d[(d["Decoy"] == 0) & (d["Q.Value"] <= 0.01)]
    if d["Modified.Sequence"].isna().any() or not np.all(np.isfinite(d["RT"].to_numpy(float))):
        raise SystemExit("guard: accepted real rows with a missing sequence or non-finite RT")
    rt = d.groupby("Modified.Sequence")["RT"].median().to_numpy(float) * 60.0
    if len(rt) < MIN_REAL_RT:
        raise SystemExit(f"guard: real RT sample has {len(rt)} modified peptides, fewer than {MIN_REAL_RT}")
    return rt


def spread_map(rt_old: np.ndarray, real_rt: np.ndarray) -> np.ndarray:
    """new = Q_real(F_sim(old)), F_sim the mid-rank empirical CDF of the sim apexes.

    Amendment 1 (stress_settings_rt_spread.yaml): DIA-NN reports RT on a ~0.17 s grid, so the real sample repeats
    values. Q_real is linear interpolation through its distinct values at their mid-point CDF, which is strictly
    increasing; only F_sim beyond the outermost knots is held at the real minimum or maximum.
    """
    import pandas as pd
    f = (pd.Series(rt_old).rank(method="average").to_numpy() - 0.5) / len(rt_old)
    x, n = np.unique(real_rt, return_counts=True)
    p = (np.cumsum(n) - n / 2) / n.sum()
    return np.interp(f, p, x)


def order_preserved(old: np.ndarray, new: np.ndarray) -> bool:
    """new is non-decreasing in old, and tied old values share one new value."""
    o = np.lexsort((new, old))
    same = np.diff(old[o]) == 0
    return bool(np.all(np.diff(new[o]) >= 0) and np.all(np.diff(new[o])[same] == 0))


def new_tie_share(old: np.ndarray, new: np.ndarray) -> float:
    """Share of adjacent distinct old RTs that the map sends to one value (plateaus in the real sample)."""
    o = np.argsort(old, kind="stable")
    distinct = np.diff(old[o]) != 0
    return float(np.mean(np.diff(new[o])[distinct] == 0)) if distinct.any() else 0.0


def read_column(db: Path, table: str, id_col: str, col: str) -> dict:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        return dict(c.execute(f"select {id_col}, {col} from {table}").fetchall())


UNTOUCHED = {"peptides": ("peptide_id", ("sequence", "events", "rt_sigma", "rt_lambda")),
             "ions": ("ion_id", ("sequence", "charge", "relative_abundance", "inv_mobility_gru_predictor_std"))}


def read_back(db: Path, src: Path, expect: dict) -> None:
    """The prepared DB holds exactly the intended apex values, and every other checked column is the baseline's."""
    for (table, col), (ids, values) in expect.items():
        got = read_column(db, table, UNTOUCHED[table][0], col)
        if len(got) != len(ids) or not np.allclose([got[i] for i in ids], values, rtol=0, atol=1e-9):
            raise SystemExit(f"guard: read-back of {db} {table}.{col} does not match")
    for table, (id_col, cols) in UNTOUCHED.items():
        for col in cols:
            if read_column(db, table, id_col, col) != read_column(src, table, id_col, col):
                raise SystemExit(f"guard: {db} {table}.{col} differs from the baseline")


def read_keyed(db: Path, table: str, key: tuple, col: str) -> dict:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        rows = c.execute(f"select {', '.join(key)}, {col} from {table}").fetchall()
    out = {r[:-1]: r[-1] for r in rows}
    if len(out) != len(rows):
        raise SystemExit(f"guard: {db} {table} key {key} is not unique")
    return out


# A replay renumbers ion_id (TimSim sets it from the reloaded frame's index), so ions are matched on
# (peptide_id, charge); peptide_id is carried over unchanged.
REPLAY_KEYS = (("peptides", ("peptide_id",), "retention_time_gru_predictor"),
               ("ions", ("peptide_id", "charge"), "inv_mobility_gru_predictor"))


def verify_replay(replay_db: Path, prepared: Path, out: Path, control: Path | None = None) -> None:
    """Replay truth DB vs prepared DB: same apex values on every row the replay kept."""
    prep_db = prepared / "synthetic_data.db"
    res = {"replay_db": str(replay_db), "prepared": str(prepared)}
    for table, key, col in REPLAY_KEYS:
        want, got = read_keyed(prep_db, table, key, col), read_keyed(replay_db, table, key, col)
        if set(got) - set(want):
            raise SystemExit(f"guard: replay {table} has ids the prepared DB lacks")
        vals = np.asarray([*got.values(), *want.values()], float)
        if not np.all(np.isfinite(vals)):
            raise SystemExit(f"guard: non-finite {col} in the replay or prepared DB")
        common = sorted(got)
        diff = np.abs(np.asarray([got[i] for i in common], float) - np.asarray([want[i] for i in common], float))
        res[col] = {"n_prepared": len(want), "n_replay": len(got), "kept_share": len(got) / len(want),
                    "max_abs_diff": float(diff.max()) if len(diff) else float("nan")}
        if len(got) < 0.99 * len(want) or not len(diff) or diff.max() > 1e-4 * max(1.0, np.abs(list(want.values())).max()):
            raise SystemExit(f"guard: replay {col} does not carry the prepared values: {res[col]}")
    prep_json = prepared.parent / "prepare.json"
    if prep_json.exists() and json.loads(prep_json.read_text()).get("row") == MASS_ACCURACY:
        verify_mass_accuracy(replay_db, prepared, control, res)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res))


MASS_ACCURACY = "mass-accuracy"
MA_SEED_OFFSET = 11
MIN_DONORS = 10_000
MA_CONTROL = "replay-control-ma"
MS1, MS2 = "mz_bias_ppm_ms1", "mz_bias_ppm_ms2"
# setting -> (ms1 applied, ms2 applied, transform of the donor deltas)
MA_SETTINGS = {MA_CONTROL: (False, False, "real"), "ms1-real": (True, False, "real"),
               "ms2-real": (False, True, "real"), "ms12-real": (True, True, "real"),
               "ms12-centred": (True, True, "centred"), "ms12-2x": (True, True, "2x")}


def real_mass_donors(report: Path) -> tuple[pd.DataFrame, int]:
    """Accepted real targets -> (Precursor.Id, ms1_ppm, fr_ppm), converted exactly as vm_discrepancy.py does."""
    from vm_discrepancy import OBSERVABLES
    d = pd.read_parquet(report, columns=["Precursor.Id", "Ms1.Apex.Mz.Delta", "Precursor.Mz", "Best.Fr.Mz.Delta",
                                         "Best.Fr.Mz", "Q.Value", "Decoy"])
    d = d[(d["Decoy"] == 0) & (d["Q.Value"] <= 0.01)]
    out = pd.DataFrame({"precursor_id": d["Precursor.Id"].to_numpy(),
                        "ms1_ppm": OBSERVABLES["ms1_ppm"][0](d).to_numpy(float),
                        "fr_ppm": OBSERVABLES["fr_ppm"][0](d).to_numpy(float)})
    ok = np.isfinite(out["ms1_ppm"]) & np.isfinite(out["fr_ppm"]) & out["precursor_id"].notna()
    out = out[ok]
    if out["precursor_id"].duplicated().any():
        raise SystemExit("guard: a Precursor.Id occurs twice among the donors")
    if len(out) < MIN_DONORS:
        raise SystemExit(f"guard: {len(out)} donors, fewer than {MIN_DONORS}")
    return out.sort_values("precursor_id", kind="stable").reset_index(drop=True), int((~ok).sum())


def ma_recipients(db: Path) -> list[tuple[int, int]]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        keys = c.execute("select peptide_id, charge from ions order by peptide_id, charge").fetchall()
    if not keys or any(None in k for k in keys) or len(set(keys)) != len(keys):
        raise SystemExit("guard: empty, null or duplicate (peptide_id, charge) among the recipients")
    return keys


def ma_assign(donors: pd.DataFrame, keys: list, seed: int) -> pd.DataFrame:
    """One donor per recipient, uniform with replacement; the (MS1, fragment) pair travels together."""
    idx = np.random.default_rng(seed + MA_SEED_OFFSET).integers(0, len(donors), size=len(keys))
    d = donors.iloc[idx].reset_index(drop=True)
    return pd.DataFrame({"peptide_id": [k[0] for k in keys], "charge": [k[1] for k in keys],
                         "donor": d["precursor_id"], "ms1_ppm": d["ms1_ppm"], "fr_ppm": d["fr_ppm"]})


def ma_values(a: pd.DataFrame, donors: pd.DataFrame, transform: str) -> tuple[np.ndarray, np.ndarray]:
    ms1, fr = a["ms1_ppm"].to_numpy(float), a["fr_ppm"].to_numpy(float)
    if transform == "centred":   # minus the donor-population median of each delta
        return ms1 - float(np.median(donors["ms1_ppm"])), fr - float(np.median(donors["fr_ppm"]))
    if transform == "2x":
        return 2 * ms1, 2 * fr
    return ms1, fr


def columns_of(db: Path, table: str) -> list[str]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        return [r[1] for r in c.execute(f"pragma table_info({table})")]


def tables_equal(db: Path, src: Path, table: str, cols: list[str]) -> bool:
    """Same multiset of rows over cols (EXCEPT both ways plus equal counts)."""
    sel = ", ".join(f'"{c}"' for c in cols)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        c.execute("attach database ? as src", (f"file:{src}?mode=ro",))
        n = c.execute(f"select (select count(*) from main.{table}), (select count(*) from src.{table})").fetchone()
        a = c.execute(f"select count(*) from (select {sel} from main.{table} except select {sel} from src.{table})").fetchone()[0]
        b = c.execute(f"select count(*) from (select {sel} from src.{table} except select {sel} from main.{table})").fetchone()[0]
    return n[0] == n[1] and a == 0 and b == 0


def ma_read_back(db: Path, src: Path, keys: list, want: dict) -> None:
    """The prepared DB holds exactly the intended bias columns and values; every baseline column is unchanged."""
    cols = columns_of(db, "ions")
    for col in (MS1, MS2):
        if (col in cols) != (col in want):
            raise SystemExit(f"guard: {db} ions.{col} present={col in cols}, intended={col in want}")
        if col in want:
            got = read_keyed(db, "ions", ("peptide_id", "charge"), col)
            if [got.get(k) for k in keys] != want[col].tolist():
                raise SystemExit(f"guard: read-back of {db} ions.{col} does not match")
    for t in KEEP:
        if not tables_equal(db, src, t, columns_of(src, t)):
            raise SystemExit(f"guard: {db} {t} differs from the baseline")


def _q(x: np.ndarray) -> list:
    return np.quantile(x, (0.05, 0.5, 0.95)).tolist()


def prepare_mass_accuracy(a) -> None:
    """stress_settings_mass_accuracy.yaml: each ion takes one real donor's (MS1, fragment) m/z delta."""
    if MS1 in columns_of(a.baseline_db, "ions") or MS2 in columns_of(a.baseline_db, "ions"):
        raise SystemExit("guard: the baseline DB already has m/z bias columns")
    donors, n_dropped = real_mass_donors(a.real_report)
    keys = ma_recipients(a.baseline_db)
    assign = ma_assign(donors, keys, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    table = a.out / "assignment.tsv"
    assign.to_csv(table, sep="\t", index=False)
    report = {"row": MASS_ACCURACY, "baseline_db": str(a.baseline_db), "real_report": str(a.real_report),
              "real_report_sha256": _sha256(a.real_report), "seed": a.seed, "rng": f"default_rng({a.seed} + {MA_SEED_OFFSET})",
              "numpy": np.__version__, "n_donors": len(donors), "n_donors_dropped_nonfinite": n_dropped,
              "n_recipients": len(keys), "n_distinct_donors_used": int(assign["donor"].nunique()),
              "donor_median_ppm": {"ms1": float(np.median(donors["ms1_ppm"])), "fr": float(np.median(donors["fr_ppm"]))},
              "donor_q05_q50_q95_ppm": {"ms1": _q(donors["ms1_ppm"]), "fr": _q(donors["fr_ppm"])},
              "assignment_sha256": _sha256(table), "settings": {}}
    for name, (do1, do2, transform) in MA_SETTINGS.items():
        ms1, fr = ma_values(assign, donors, transform)
        want = {**({MS1: ms1} if do1 else {}), **({MS2: fr} if do2 else {})}
        for col, v in want.items():
            if not np.all(np.isfinite(v)) or np.any(1 + v * 1e-6 <= 0):
                raise SystemExit(f"guard: {name} {col} has non-finite or non-positive factors")
        db = a.out / name / "synthetic_data.db"
        slim_copy(a.baseline_db, db)
        with sqlite3.connect(db) as c:
            c.execute("create index main.vm_ion_key on ions(peptide_id, charge)")
            for col, v in want.items():
                c.execute(f"alter table ions add column {col} REAL")
                c.executemany(f"update ions set {col} = ? where peptide_id = ? and charge = ?",
                              zip(v.tolist(), (k[0] for k in keys), (k[1] for k in keys)))
            c.commit()
        ma_read_back(db, a.baseline_db, keys, want)
        report["settings"][name] = {col: {"q05_q50_q95_ppm": _q(v), "share_zero": float(np.mean(v == 0)),
                                          "mean_abs_ppm": float(np.mean(np.abs(v)))} for col, v in want.items()}
    (a.out / "prepare.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


def verify_mass_accuracy(replay_db: Path, prepared: Path, control: Path | None, res: dict) -> None:
    """Bias columns carried as prepared, absent where not prepared; same ions as the seed's replay-control-ma."""
    prep_db = prepared / "synthetic_data.db"
    pcols, rcols = columns_of(prep_db, "ions"), columns_of(replay_db, "ions")
    key = ("peptide_id", "charge")
    for col in (MS1, MS2):
        if col not in pcols:
            if col in rcols:
                raise SystemExit(f"guard: replay has ions.{col}, the prepared DB does not")
            continue
        if col not in rcols:
            raise SystemExit(f"guard: replay dropped ions.{col}")
        want, got = read_keyed(prep_db, "ions", key, col), read_keyed(replay_db, "ions", key, col)
        if set(got) - set(want):
            raise SystemExit(f"guard: replay ions have keys the prepared DB lacks")
        bad = [k for k in got if got[k] is None or not np.isfinite(got[k]) or got[k] != want[k]]
        if bad:
            raise SystemExit(f"guard: replay ions.{col} differs from the prepared values on {len(bad)} ions, e.g. {bad[:3]}")
        res[col] = {"n_prepared": len(want), "n_replay": len(got), "all_equal": True}
    with sqlite3.connect(f"file:{replay_db}?mode=ro", uri=True) as c:
        keys = sorted(c.execute("select peptide_id, charge from ions").fetchall())
    import hashlib
    res["ion_keys"] = {"n": len(keys), "sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest()}
    if prepared.name == MA_CONTROL:
        return
    if control is None:
        raise SystemExit("guard: mass-accuracy --verify needs --control-replay (the seed's replay-control-ma)")
    with sqlite3.connect(f"file:{control}?mode=ro", uri=True) as c:
        ckeys = sorted(c.execute("select peptide_id, charge from ions").fetchall())
    if ckeys != keys:
        raise SystemExit(f"guard: replay keeps {len(keys)} ions, replay-control-ma {len(ckeys)}; the sets differ")
    res["ion_keys"]["same_as_control"] = str(control)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-db", type=Path, help="baseline synthetic_data.db")
    ap.add_argument("--gradient", type=float, help="gradient length (s), from the baseline plan.json")
    ap.add_argument("--seed", type=int, help="the baseline sample_seed")
    ap.add_argument("--expect-seed", type=int, help="the seed stress_settings.yaml pairs with this baseline")
    ap.add_argument("--chronologer-model", type=Path)
    ap.add_argument("--row", choices=["predictor-swap", RT_SPREAD, MASS_ACCURACY], default="predictor-swap")
    ap.add_argument("--real-report", type=Path, help="rt-spread, mass-accuracy: real LS10 plain report.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify", type=Path, help="replay synthetic_data.db to check against --prepared")
    ap.add_argument("--prepared", type=Path)
    ap.add_argument("--control-replay", type=Path, help="mass-accuracy --verify: the seed's replay-control-ma DB")
    a = ap.parse_args()
    if a.verify:
        return verify_replay(a.verify, a.prepared, a.out, a.control_replay)
    if a.expect_seed is not None and a.seed != a.expect_seed:
        raise SystemExit(f"guard: baseline seed {a.seed} != frozen seed {a.expect_seed}")
    if not a.gradient or a.gradient <= 0:
        raise SystemExit("guard: gradient must be positive")

    with sqlite3.connect(f"file:{a.baseline_db}?mode=ro", uri=True) as c:
        pep = c.execute("select peptide_id, sequence, retention_time_gru_predictor, rt_sigma from peptides "
                        "order by peptide_id").fetchall()
        ion = c.execute("select ion_id, inv_mobility_gru_predictor, inv_mobility_gru_predictor_std from ions "
                        "order by ion_id").fetchall()
        im_lo, im_hi = c.execute("select min(mobility), max(mobility) from scans").fetchone()
    pep_id, ion_id = [r[0] for r in pep], [r[0] for r in ion]
    _ids(pep_id, "peptides")
    _ids(ion_id, "ions")
    rt_old = np.asarray([r[2] for r in pep], float)
    rt_sigma = np.asarray([r[3] for r in pep], float)
    im_old = np.asarray([r[1] for r in ion], float)
    im_std = np.asarray([r[2] for r in ion], float)
    for x, w in ((rt_old, "rt"), (rt_sigma, "rt_sigma"), (im_old, "im"), (im_std, "im_std")):
        _finite(x, w)
    if np.any(im_std <= 0) or np.any(rt_sigma <= 0) or im_lo is None or not im_lo < im_hi:
        raise SystemExit("guard: non-positive widths or invalid IM range")

    if a.row == MASS_ACCURACY:
        return prepare_mass_accuracy(a)
    if a.row == RT_SPREAD:
        return prepare_rt_spread(a, pep_id, rt_old, rt_sigma, ion_id, im_old)
    hi = chronologer_hi([r[1] for r in pep], a.chronologer_model)
    n_sec = sum("U" in re.sub(r"\[UNIMOD:\d+\]", "", r[1]) for r in pep)
    rt_new = min_max(hi, 0.0, a.gradient)
    rng = np.random.default_rng(a.seed + IM_SEED_OFFSET)
    im_draw = rng.normal(im_old, im_std)
    im_new = np.clip(im_draw, im_lo, im_hi)

    report = {"baseline_db": str(a.baseline_db), "seed": a.seed, "gradient_s": a.gradient,
              "im_range": [im_lo, im_hi], "im_clipped_share": float(np.mean(im_draw != im_new)),
              "n_peptides": len(pep_id), "n_ions": len(ion_id), "n_peptides_selenocysteine_as_c": int(n_sec),
              "rt_spearman_local_vs_chronologer": float(np.corrcoef(np.argsort(np.argsort(rt_old)),
                                                                    np.argsort(np.argsort(rt_new)))[0, 1]),
              "im_std_mean": float(im_std.mean()), "settings": {}}
    for name, (do_rt, do_im) in SETTINGS.items():
        db = a.out / name / "synthetic_data.db"
        slim_copy(a.baseline_db, db)
        stats, expect = {}, {("peptides", "retention_time_gru_predictor"): (pep_id, rt_old),
                             ("ions", "inv_mobility_gru_predictor"): (ion_id, im_old)}
        with sqlite3.connect(db) as c:
            if do_rt:
                c.executemany("update peptides set retention_time_gru_predictor = ? where peptide_id = ?",
                              zip(rt_new.tolist(), pep_id))
                stats["rt"] = change_stats(rt_old, rt_new, rt_sigma)
                expect[("peptides", "retention_time_gru_predictor")] = (pep_id, rt_new)
            if do_im:
                c.executemany("update ions set inv_mobility_gru_predictor = ? where ion_id = ?",
                              zip(im_new.tolist(), ion_id))
                stats["im"] = change_stats(im_old, im_new, im_std)
                expect[("ions", "inv_mobility_gru_predictor")] = (ion_id, im_new)
            c.commit()
        for k, st in stats.items():
            if not st["median_abs_change"] > 0:
                raise SystemExit(f"guard: {name} {k} rewrite changed nothing")
        read_back(db, a.baseline_db, expect)
        report["settings"][name] = stats
    (a.out / "prepare.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))

def _sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _near_edge(rt: np.ndarray, sigma: np.ndarray, gradient: float) -> float:
    return float(np.mean((rt - 3 * sigma < 0) | (rt + 3 * sigma > gradient)))


def prepare_rt_spread(a, pep_id: list, rt_old: np.ndarray, rt_sigma: np.ndarray, ion_id: list,
                      im_old: np.ndarray) -> None:
    """stress_settings_rt_spread.yaml: RT apexes keep their order and take the real run's spread."""
    real = real_rt_sample(a.real_report)
    rt_new = spread_map(rt_old, real)
    _finite(rt_new, "rt-spread RT")
    ties = new_tie_share(rt_old, rt_new)
    if not order_preserved(rt_old, rt_new) or ties > MAX_NEW_TIES:
        raise SystemExit(f"guard: rt-spread map does not preserve the RT order (new tie share {ties:.4f})")
    if rt_new.min() < 0 or rt_new.max() > a.gradient:
        raise SystemExit(f"guard: rt-spread RT outside [0, {a.gradient}]")
    a.out.mkdir(parents=True, exist_ok=True)
    db = a.out / RT_SPREAD / "synthetic_data.db"
    slim_copy(a.baseline_db, db)
    with sqlite3.connect(db) as c:
        c.executemany("update peptides set retention_time_gru_predictor = ? where peptide_id = ?",
                      zip(rt_new.tolist(), pep_id))
        c.commit()
    stats = {"rt": change_stats(rt_old, rt_new, rt_sigma)}
    if not stats["rt"]["median_abs_change"] > 0:
        raise SystemExit("guard: rt-spread rewrite changed nothing")
    read_back(db, a.baseline_db, {("peptides", "retention_time_gru_predictor"): (pep_id, rt_new),
                                  ("ions", "inv_mobility_gru_predictor"): (ion_id, im_old)})
    q = (0.05, 0.5, 0.95)
    report = {"row": RT_SPREAD, "baseline_db": str(a.baseline_db), "real_report": str(a.real_report),
              "seed": a.seed, "gradient_s": a.gradient, "n_peptides": len(pep_id), "n_real_precursors": len(real),
              "rt_quantiles_s": {"real": np.quantile(real, q).tolist(), "sim_before": np.quantile(rt_old, q).tolist(),
                                 "sim_after": np.quantile(rt_new, q).tolist()},
              "new_tie_share": ties, "real_report_sha256": _sha256(a.real_report),
              # descriptive: peaks whose apex sits within 3 sigma of either end of the acquisition
              "near_boundary_share": {"before": _near_edge(rt_old, rt_sigma, a.gradient),
                                      "after": _near_edge(rt_new, rt_sigma, a.gradient)},
              "settings": {RT_SPREAD: stats}}
    (a.out / "prepare.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
