"""Fixture tests for the validity-matrix analysis. Run: pytest evident/vm/test_vm.py"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from vm_common import fdrbench_at_threshold, mod_key, modal, stripped_key, verdict  # noqa: E402

# Pairs: (target, entrapment). Targets 0-3 are simulated, target 4 (KLMNLLR) is not.
PAIRS = [("PEPTLDEK", "EPTPLDEK"), ("ACDEFGHK", "CDAEGFHK"), ("MLLNQSTR", "LNLMQTSR"),
         ("VVVYAAAK", "YVAVAVAK"), ("KLMNLLR", "LMKNLLR")]
ENTRAP = {e for _, e in PAIRS}
SIM_RUNS = [f"jitter-{c}k-r{i}" for c in (50, 125, 250) for i in (1, 2, 3)]
SIM_IONS = [("PEPTIDEK", 2), ("AC[UNIMOD:4]DEFGHK", 2), ("[UNIMOD:1]M[UNIMOD:35]ILNQSTR", 3), ("VVVYAAAK", 2)]


def test_mod_key_maps_both_notations_and_keeps_unimod_intact():
    assert mod_key("(UniMod:1)M(UniMod:35)ILNQSTR") == mod_key("[UNIMOD:1]M[UNIMOD:35]ILNQSTR") == "{1}M{35}LLNQSTR"
    assert stripped_key("AC[UNIMOD:4]DEFGHK") == "ACDEFGHK"


def test_verdict_and_modal():
    assert verdict(0.002, 0.009) == "valid"
    assert verdict(0.011, 0.02) == "invalid"
    assert verdict(0.004, 0.015) == "inconclusive"
    assert modal(["valid", "valid", "invalid"]) == "valid"
    assert modal(["valid", "invalid", "inconclusive"]) == "inconclusive"


def test_fdrbench_reads_cumulative_row_not_last_tied_row(tmp_path):
    f = tmp_path / "fdp.csv"
    pd.DataFrame({
        "peptide": ["A", "B", "C", "D"], "mod_peptide": ["A", "B", "C", "D"], "charge": [2] * 4,
        "q_value": [0.001, 0.01, 0.01, 0.02],
        "protein": "P", "n_t": [1, 2, 2, 4], "n_p": [0, 1, 0, 1],
        "lower_bound_fdp": [0, 0.25, 0.33, 0.2], "combined_fdp": [0, 0.5, 0.66, 0.4], "paired_fdp": [0, 0.45, 0.6, 0.38],
    }).to_csv(f, index=False)
    acc, est = fdrbench_at_threshold(f)
    assert len(acc) == 3 and est["n_t"] + est["n_p"] == 3 and est["paired"] == 0.45


def _report(path: Path, rows: list[tuple[str, str, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Run": "r", "Modified.Sequence": [r[0] for r in rows], "Stripped.Sequence": [r[1] for r in rows],
        "Precursor.Charge": [r[2] for r in rows], "Q.Value": [r[3] for r in rows],
        "Protein.Ids": ["P_p_target" if r[1] in ENTRAP else "P" for r in rows],
    }).to_parquet(path)


def _fdrbench(path: Path, rows: list[tuple[str, str, int, float]], entrap: set[str]) -> None:
    rows = sorted(rows, key=lambda r: r[3])
    n_t = n_p = 0
    out = []
    for mod, s, z, q in rows:
        is_p = s.replace("I", "L") in entrap
        n_p += is_p
        n_t += not is_p
        out.append({"peptide": s.replace("I", "L"), "mod_peptide": mod.replace("I", "L"), "charge": z, "q_value": q,
                    "protein": "P_p_target" if is_p else "P",
                    "n_t": n_t, "n_p": n_p, "lower_bound_fdp": n_p / (n_t + n_p),
                    "combined_fdp": 2 * n_p / (n_t + n_p), "paired_fdp": 1.5 * n_p / (n_t + n_p)})
    pd.DataFrame(out).to_csv(path, index=False)


@pytest.fixture
def tree(tmp_path):
    recipe, sim_root = tmp_path / "recipe-v1", tmp_path / "sim-baseline"
    (recipe / "entrap").mkdir(parents=True)
    pd.DataFrame(
        [{"sequence": t, "decoy": "No", "proteins": "P", "peptide_type": "target", "peptide_pair_index": i} for i, (t, _) in enumerate(PAIRS)]
        + [{"sequence": e, "decoy": "No", "proteins": "P_p_target", "peptide_type": "p_target", "peptide_pair_index": i} for i, (_, e) in enumerate(PAIRS)]
    ).to_csv(recipe / "entrap" / "human_entrap_pep.txt", sep="\t", index=False)
    entrap = {e for _, e in PAIRS}

    # Accepted at 1%: 3 true targets, 1 unsimulated target, 1 entrapment. Plus one row above the cut.
    rows = [("PEPTIDEK", "PEPTIDEK", 2, 0.001), ("AC(UniMod:4)DEFGHK", "ACDEFGHK", 2, 0.002),
            ("(UniMod:1)M(UniMod:35)ILNQSTR", "MILNQSTR", 3, 0.003), ("KLMNLLR", "KLMNLLR", 2, 0.004),
            ("EPTPLDEK", "EPTPLDEK", 2, 0.005), ("VVVYAAAK", "VVVYAAAK", 2, 0.04)]
    for side, runs in (("real10", ["LS10", "LS11", "LS12"]), ("sim", SIM_RUNS)):
        for run in runs:
            _report(recipe / side / run / "plain" / "report.parquet", rows[:4] + rows[5:])
            _report(recipe / side / run / "entrap" / "report.parquet", rows)
            _fdrbench(recipe / side / run / "entrap" / "fdp_precursor.csv", rows, entrap)
            if side == "sim":
                db = sim_root / run / "out" / "x" / "synthetic_data.db"
                db.parent.mkdir(parents=True)
                with sqlite3.connect(db) as c:
                    c.execute("create table ions (sequence text, charge integer)")
                    c.executemany("insert into ions values (?, ?)", SIM_IONS)
    return recipe, sim_root, tmp_path / "out"


def _run(script: str, *args) -> None:
    subprocess.run([sys.executable, str(HERE / script), *map(str, args)], check=True, cwd=HERE)


def test_end_to_end(tree):
    recipe, sim_root, out = tree
    _run("vm_match_complexity.py", "--recipe", recipe, "--out", out / "matched_complexity.json")
    m = json.loads((out / "matched_complexity.json").read_text())
    assert m["real_median"] == 4 and m["matched_complexity"] == 250000  # three-way tie -> highest complexity
    _run("vm_truth_fdp.py", "--mode", "all", "--recipe", recipe, "--sim-root", sim_root, "--out", out)
    side = json.loads((out / "last_verified.json").read_text())

    # 5 accepted in the entrapment search: 1 entrapment + 1 unsimulated target false -> true FDP 0.4.
    # Fixture estimates: paired 1.5 * 1/5 = 0.3, combined 0.4.
    bias = side["entrap.sim.estimator-bias.diann"]["values"]
    assert bias["paired_bias_mean"] == pytest.approx(0.3 - 0.4)
    assert bias["combined_bias_mean"] == pytest.approx(0.0)
    assert "bracket_frequency" not in bias
    # Targets only: 1 of 4 false in both searches -> no shift.
    assert side["entrap.sim.search-space-shift.diann"]["values"]["target_fdp_shift_mean_abs"] == pytest.approx(0)
    assert side["vm.baseline.fdp-bound.diann"]["values"]["fdp_max"] == pytest.approx(0.25)
    assert side["entrap.verdict.diann.real10"]["values"]["paired_upper_max"] == pytest.approx(0.3)
    assert side["entrap.sim-vs-real.verdict-agree.diann"]["values"]["class_agreement"] == 1.0
    assert side["entrap.sim-vs-real.diann"]["values"]["paired_abs_diff"] == pytest.approx(0)
    verdicts = json.loads((out / "results" / "verdict.json").read_text())
    assert verdicts["real10"]["modal"] == "invalid"  # lower bound 0.2 > 0.01


def test_w1_matches_scipy():
    import numpy as np
    from scipy.stats import wasserstein_distance
    from vm_discrepancy import w1
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 500), rng.normal(0.3, 1.4, 321)
    assert w1(a, b) == pytest.approx(wasserstein_distance(a, b))


def test_discrepancy_flags_shifted_sim(tmp_path):
    import numpy as np
    rng = np.random.default_rng(1)
    recipe, out = tmp_path / "recipe-v1", tmp_path / "out"

    def rep(path, n, ppm_shift):
        path.parent.mkdir(parents=True, exist_ok=True)
        mz = rng.uniform(350, 1100, n)
        rt = rng.uniform(5, 110, n)
        pd.DataFrame({
            "Modified.Sequence": [f"PEP{i}K" for i in range(n)], "Stripped.Sequence": [f"PEP{i}K" for i in range(n)],
            "Precursor.Charge": rng.choice([2, 3, 4], n, p=[0.7, 0.26, 0.04]), "Precursor.Mz": mz, "Q.Value": 0.001,
            "Ms1.Apex.Mz.Delta": (rng.normal(2 + ppm_shift, 2, n)) * mz / 1e6, "Best.Fr.Mz": rng.uniform(200, 1500, n),
            "Best.Fr.Mz.Delta": rng.normal(0, 3e-3, n), "FWHM": rng.gamma(4, 0.025, n), "RT": rt,
            "RT.Start": rt - 0.1, "RT.Stop": rt + 0.1, "Predicted.RT": rt + rng.normal(0, 0.5, n), "IM": 1.0,
            "Predicted.IM": 1.0 + rng.normal(0, 0.01, n), "Ms1.Area": 10 ** rng.normal(6, 1, n),
            "Evidence": rng.uniform(0, 6, n), "Ms1.Profile.Corr": rng.uniform(0, 1, n),
            "Mass.Evidence": rng.uniform(0, 1, n), "PEP": rng.uniform(0, 0.01, n),
        }).to_parquet(path)

    for r in ("LS10", "LS11", "LS12"):
        rep(recipe / "real10" / r / "plain" / "report.parquet", 3000, 0)
    for s in SIM_RUNS:
        rep(recipe / "sim" / s / "plain" / "report.parquet", 2000, 3)
    out.mkdir()
    (out / "matched_complexity.json").write_text(json.dumps({"matched_complexity": 50000}))
    _run("vm_discrepancy.py", "--recipe", recipe, "--out", out)
    res = json.loads((out / "results" / "discrepancy.json").read_text())["by_complexity"]["50000"]
    assert res["all"]["ms1_ppm"]["ratio"] > 5        # shifted by 3 ppm
    assert res["all"]["fwhm"]["ratio"] < 5           # same distribution
    assert 0 < res["shared"]["n_shared"] and 0 < res["shared"]["coverage_of_real"] <= 1


def test_cohort_and_label_guards(tree, tmp_path):
    from vm_common import check_cohort, is_entrapment
    recipe, _, _ = tree
    assert check_cohort(recipe)[0] == ["LS10", "LS11", "LS12"]
    (recipe / "sim" / "jitter-50k-r4").mkdir()
    with pytest.raises(ValueError, match="sim cohort"):
        check_cohort(recipe)
    pairs = {"EPTPLDEK": "p_target", "PEPTLDEK": "target"}
    with pytest.raises(ValueError, match="without the"):
        is_entrapment(pd.Series(["EPTPLDEK"]), pairs, pd.Series(["P"]))
    got = is_entrapment(pd.Series(["UNKNOWNK", "PEPTLDEK"]), pairs, pd.Series(["X_p_target", "P"]))
    assert got.tolist() == [True, False]
    with pytest.raises(ValueError, match="unrecognised"):
        mod_key("PEP(Phospho)TK")


def test_stripped_estimand_counts_each_backbone_once():
    from vm_common import fdp_stripped
    lab = pd.DataFrame({"stripped": ["A", "A", "A", "B"], "charge": [2, 2, 2, 2],
                        "false_stripped": [False, False, False, True]})
    assert fdp_stripped(lab) == 0.5


def test_direction_classify_and_end_to_end(tmp_path):
    from vm_direction import FEATURES, classify
    assert classify(1.0, [2, 3, 4], "lower") == "easier"
    assert classify(5.0, [2, 3, 4], "lower") == "harder"
    assert classify(5.0, [2, 3, 4], "higher") == "easier"
    assert classify(3.0, [2, 3, 4], "higher") == "comparable"
    runs = ["LS10", "LS11", "LS12", "s1", "s2", "s3"]
    tails = {r: {"q05": -1.0, "q50": 1.0, "q95": 1.0} for r in runs}
    for r in ("LS10", "LS11", "LS12"):
        tails[r] = {"q05": -2.0 - 0.1 * runs.index(r), "q50": 1.0 + 0.1 * runs.index(r), "q95": 2.0}
    view = {f: {"tails": tails} for f in FEATURES}
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "discrepancy.json").write_text(json.dumps({"by_complexity": {"50000": {"all": view}}}))
    (tmp_path / "matched_complexity.json").write_text(json.dumps({"matched_complexity": 50000, "matched_runs": ["s1", "s2", "s3"]}))
    _run("vm_direction.py", "--out", tmp_path)
    d = json.loads((tmp_path / "results" / "direction.json").read_text())
    # sim spread 2.0 < real [4.0, 4.2]: easier on 4 spread features; sim q50 1.0 within [1.0, 1.2]: comparable;
    # sim q95 1.0 < real 2.0 with lower easier: pep easier.
    assert d["n_easier"] == 5 and d["n_harder"] == 0
    side = json.loads((tmp_path / "last_verified.json").read_text())
    assert side["vm.direction.sim-not-harder.diann"]["values"] == {"n_harder": 0.0}


def test_w1_binned_and_auc():
    import numpy as np
    from vm_marginals import w1_binned
    from vm_scores import auc
    e = np.arange(0, 11.0)
    a = np.zeros(10); a[2] = 1
    b = np.zeros(10); b[5] = 1
    assert w1_binned(a, b, e) == pytest.approx(3.0)
    assert w1_binned(a, a * 7, e) == 0
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0]), np.array([1.0])) == 0.5


def test_protein_truth_and_estimates(tree):
    recipe, sim_root, out = tree
    prot = recipe.parent / "recipe-v1-protein"
    out.mkdir(exist_ok=True)
    (out / "matched_complexity.json").write_text(json.dumps({"matched_runs": SIM_RUNS[-3:]}))
    # Peptides in the fixture DB carry no protein; give them P1 so P1 is the simulated protein.
    for run in SIM_RUNS:
        db = next((sim_root / run / "out").rglob("synthetic_data.db"))
        with sqlite3.connect(db) as c:
            c.execute("alter table ions add column peptide_id integer")
            c.execute("update ions set peptide_id = rowid")
            c.execute("create table peptides (peptide_id integer, protein text)")
            c.executemany("insert into peptides values (?, ?)", [(i, "sp|P1|X_HUMAN desc") for i in range(1, 5)])
    groups = [("P1", 0.001, 1, 0), ("P9", 0.002, 2, 0), ("P1_p_target", 0.003, 2, 1), ("P2", 0.5, 3, 1)]
    for side, runs in (("real10", ["LS10", "LS11", "LS12"]), ("sim", SIM_RUNS)):
        for run in runs:
            d = prot / side / run
            d.mkdir(parents=True)
            pd.DataFrame([{"protein": g, "q_value": q, "n_t": t, "n_p": p, "lower_bound_fdp": p / (t + p),
                           "combined_fdp": 2 * p / (t + p)} for g, q, t, p in groups]).to_csv(d / "fdp_protein.csv", index=False)
            pd.DataFrame({"Protein.Group": ["P1", "P9"], "Modified.Sequence": ["PEPTIDEK", "KLMNLLR"],
                          "Precursor.Charge": [2, 2], "Q.Value": [0.001, 0.001], "Decoy": [0, 0]}).to_parquet(d / "report.parquet")
    _run("vm_protein.py", "--recipe", recipe, "--protein", prot, "--sim-root", sim_root, "--out", out)
    res = json.loads((out / "results" / "protein.json").read_text())
    # 3 accepted groups: P1 true, P9 not simulated, entrapment-only false -> true FDP 2/3; combined 2/3.
    assert res["sim"][0]["true_fdp"] == pytest.approx(2 / 3)
    assert res["values"]["entrap.protein.sim.estimator-bias.diann"]["combined_bias_mean"] == pytest.approx(0)
    assert res["modal_class"]["real10"] == "invalid"
