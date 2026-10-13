"""Fixture tests for the validity-matrix analysis. Run: pytest evident/vm/test_vm.py"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
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



def _ions_db(db: Path, ions) -> None:
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as c:
        c.execute("create table ions (sequence text, charge integer)")
        c.executemany("insert into ions values (?, ?)", ions)


VERIFY_OK = json.dumps({"retention_time_gru_predictor": {}, "inv_mobility_gru_predictor": {}})


def _checked(run_dir: Path, verify: str = VERIFY_OK) -> None:
    (run_dir / "verify.json").write_text(verify)
    (run_dir / "done.flag").write_text("frozen\n")


def test_predictor_swap_stress(tree, tmp_path):
    """Seeds differ in truth; rt-chronologer moves FDP up on r1 and down on r2, im-std adds a Met-ox confusion."""
    recipe, sim_root, out = tree
    _run("vm_match_complexity.py", "--recipe", recipe, "--out", out / "matched_complexity.json")
    srec, sroot = tmp_path / "recipe-v1-stress", tmp_path / "sim-stress"
    seeds = ("jitter-250k-r1", "jitter-250k-r2", "jitter-250k-r3")
    rows4 = [("PEPTIDEK", "PEPTIDEK", 2, 0.001), ("AC(UniMod:4)DEFGHK", "ACDEFGHK", 2, 0.002),
             ("(UniMod:1)M(UniMod:35)ILNQSTR", "MILNQSTR", 3, 0.003), ("KLMNLLR", "KLMNLLR", 2, 0.004)]
    unox = ("(UniMod:1)MILNQSTR", "MILNQSTR", 3, 0.003)       # simulated only as M(ox): Met-ox confusion
    vvv = ("VVVYAAAK", "VVVYAAAK", 2, 0.002)
    # r2's truth also contains KLMNLLR, so the shared rows are 1/4 false on r1/r3 and 0/4 on r2.
    truth = {"jitter-250k-r1": SIM_IONS, "jitter-250k-r2": SIM_IONS + [("KLMNLLR", 2)], "jitter-250k-r3": SIM_IONS}
    for rid in seeds:
        (recipe / "sim" / rid / "plain" / "diann.stdout").write_text("RT window set to 15.7\nIM window set to 0.109\n")
        db = sim_root / rid / "out" / "x" / "synthetic_data.db"
        db.unlink()
        _ions_db(db, truth[rid])
        for st in ("replay-control", "rt-chronologer", "im-std", "joint"):
            rows = list(rows4)
            if st == "im-std":
                rows += [vvv, unox]
            if st == "rt-chronologer" and rid == "jitter-250k-r1":
                rows += [("NEVERSIMK", "NEVERSIMK", 2, 0.005)]        # up: 2/5
            if st == "rt-chronologer" and rid == "jitter-250k-r2":
                rows += [vvv]                                         # flat: 0/5 vs baseline 0/4
            if st == "rt-chronologer" and rid == "jitter-250k-r3":
                rows += [vvv]                                         # down: 1/5 vs baseline 1/4
            _report(srec / "sim" / f"{rid}-{st}" / "plain" / "report.parquet", rows)
            (srec / "sim" / f"{rid}-{st}" / "plain" / "diann.stdout").write_text("RT window set to 9.1\nRT window set to 8.0\n")
            _ions_db(sroot / f"{rid}-{st}" / "out" / "x" / "synthetic_data.db", truth[rid])
    args = ["--mode", "stress", "--claim", "vm.predictor-swap.fdp-delta.diann", "--recipe", recipe, "--sim-root",
            sim_root, "--out", out, "--stress-recipe", srec, "--stress-root", sroot]
    with pytest.raises(subprocess.CalledProcessError):  # no verify.json yet
        _run("vm_truth_fdp.py", *args)
    for rid in seeds:
        for st in ("replay-control", "rt-chronologer", "im-std", "joint"):
            _checked(sroot / f"{rid}-{st}")
    _run("vm_truth_fdp.py", *args)
    res = json.loads((out / "results" / "stress-predictor-swap.json").read_text())
    assert [res["baseline"][r]["fdp"] for r in seeds] == pytest.approx([0.25, 0.0, 0.25])
    d = res["per_seed_differences"]["rt-chronologer"]
    assert d["jitter-250k-r1"]["minus_baseline"] == pytest.approx(0.4 - 0.25)
    assert d["jitter-250k-r2"]["minus_baseline"] == pytest.approx(0.0)
    assert d["jitter-250k-r3"]["minus_baseline"] == pytest.approx(0.2 - 0.25)
    assert res["per_setting"]["rt-chronologer"]["delta_vs_baseline"] == pytest.approx((0.15 + 0 - 0.05) / 3)
    # im-std: r1/r3 2/6 false, r2 1/6 false (KLMNLLR simulated there); baseline mean 1/6.
    im = res["per_setting"]["im-std"]
    assert im["fdp_mean"] == pytest.approx((2 / 6 + 1 / 6 + 2 / 6) / 3)
    assert res["per_setting"]["replay-control"]["delta_vs_baseline"] == pytest.approx(0)
    assert res["values"]["vm.predictor-swap.fdp-delta.diann"]["fdp_delta_max"] == pytest.approx(
        (2 / 6 + 1 / 6 + 2 / 6) / 3 - 1 / 6)
    assert res["values"]["vm.predictor-swap.fdp-bound.diann"]["fdp_max"] == pytest.approx(0.4)
    r = res["per_seed"]["im-std"]["jitter-250k-r2"]
    assert r["mod_form_share_of_false"] == pytest.approx(1.0) and r["met_ox_share_of_false"] == pytest.approx(1.0)
    assert res["per_seed"]["rt-chronologer"]["jitter-250k-r1"]["met_ox_share_of_false"] == pytest.approx(0)
    assert r["rt_window_min"] == 8.0 and res["baseline"]["jitter-250k-r1"]["rt_window_min"] == 15.7
    assert "vm.predictor-swap.fdp-bound.diann" in json.loads((out / "last_verified.json").read_text())

    # rt-spread: its own setting, predictor-swap's replay-control as control
    args = ["--mode", "stress", "--claim", "vm.rt-spread.fdp-bound.diann", "--recipe", recipe, "--sim-root",
            sim_root, "--out", out, "--stress-recipe", srec, "--stress-root", sroot]
    for rid in seeds:
        rows = list(rows4) + ([("NEVERSIMK", "NEVERSIMK", 2, 0.005)] if rid == "jitter-250k-r2" else [])
        _report(srec / "sim" / f"{rid}-rt-spread" / "plain" / "report.parquet", rows)
        _ions_db(sroot / f"{rid}-rt-spread" / "out" / "x" / "synthetic_data.db", truth[rid])
    with pytest.raises(subprocess.CalledProcessError):  # no verify.json yet
        _run("vm_truth_fdp.py", *args)
    for rid in seeds:
        _checked(sroot / f"{rid}-rt-spread", "{}")
    with pytest.raises(subprocess.CalledProcessError):  # an empty verify.json is not a check
        _run("vm_truth_fdp.py", *args)
    for rid in seeds:
        _checked(sroot / f"{rid}-rt-spread")
    _run("vm_truth_fdp.py", *args)
    rs = json.loads((out / "results" / "stress-rt-spread.json").read_text())
    assert set(rs["per_setting"]) == {"replay-control", "rt-spread"} and rs["row"] == "rt-spread"
    assert rs["per_seed"]["rt-spread"]["jitter-250k-r2"]["fdp"] == pytest.approx(1 / 5)
    assert rs["values"]["vm.rt-spread.fdp-bound.diann"]["fdp_max"] == pytest.approx(0.25)
    assert rs["values"]["vm.rt-spread.fdp-delta.diann"]["fdp_delta_max"] == pytest.approx((0.25 + 0.2 + 0.25) / 3 - 1 / 6)
    sidecar = json.loads((out / "last_verified.json").read_text())
    assert "vm.rt-spread.fdp-bound.diann" in sidecar and "vm.predictor-swap.fdp-bound.diann" in sidecar


def _mass_report(path: Path, rows, ms1_ppm, fr_ppm) -> None:
    """A plain report with DIA-NN m/z delta columns; delta = ppm * m/z * 1e-6 at m/z 500 (MS1) and 1000 (fragment)."""
    _report(path, rows)
    d = pd.read_parquet(path)
    d["Decoy"], d["Precursor.Mz"], d["Best.Fr.Mz"] = 0, 500.0, 1000.0
    d["Ms1.Apex.Mz.Delta"] = np.resize(np.asarray(ms1_ppm, float), len(d)) * 500e-6
    d["Best.Fr.Mz.Delta"] = np.resize(np.asarray(fr_ppm, float), len(d)) * 1000e-6
    d.to_parquet(path)


def test_mass_accuracy_stress(tree, tmp_path):
    """Claim values over ms1/ms2/ms12-real only; manipulation, depth and replay checks decide the interpretation."""
    recipe, sim_root, out = tree
    _run("vm_match_complexity.py", "--recipe", recipe, "--out", out / "matched_complexity.json")
    srec, sroot = tmp_path / "recipe-v1-stress", tmp_path / "sim-stress"
    seeds = ("jitter-250k-r1", "jitter-250k-r2", "jitter-250k-r3")
    rows4 = [("PEPTIDEK", "PEPTIDEK", 2, 0.001), ("AC(UniMod:4)DEFGHK", "ACDEFGHK", 2, 0.002),
             ("(UniMod:1)M(UniMod:35)ILNQSTR", "MILNQSTR", 3, 0.003), ("VVVYAAAK", "VVVYAAAK", 2, 0.004)]
    wide = (np.linspace(-10, 5, 4), np.linspace(-15, 10, 4))     # widths 13.5 and 22.5 ppm (> 4.5 / 9.2)
    narrow = (np.linspace(-1, 1, 4), np.linspace(-2, 2, 4))
    for r in ("LS10", "LS11", "LS12"):
        _mass_report(recipe / "real10" / r / "plain" / "report.parquet", rows4, *wide)
    for rid in seeds:
        _mass_report(recipe / "sim" / rid / "plain" / "report.parquet", rows4, *narrow)   # 0/4 false
        db = sim_root / rid / "out" / "x" / "synthetic_data.db"
        db.unlink()
        _ions_db(db, SIM_IONS)
    settings = ("replay-control-ma", "ms1-real", "ms2-real", "ms12-real", "ms12-centred", "ms12-2x")
    bias = {"ms1-real": ("mz_bias_ppm_ms1",), "ms2-real": ("mz_bias_ppm_ms2",)}

    def build(ms12_rows, ctrl_rows=rows4, ms12_deltas=wide, x2_rows=None):
        for rid in seeds:
            for st in settings:
                rows = {"ms12-real": ms12_rows, "replay-control-ma": ctrl_rows,
                        "ms12-2x": rows4[:1] + [("NEVERSIMK", "NEVERSIMK", 2, 0.005)] if x2_rows is None
                        else x2_rows}.get(st, rows4)
                _mass_report(srec / "sim" / f"{rid}-{st}" / "plain" / "report.parquet", rows,
                             *(ms12_deltas if st == "ms12-real" else wide))
                db = sroot / f"{rid}-{st}" / "out" / "x" / "synthetic_data.db"
                if db.exists():
                    db.unlink()
                db.parent.mkdir(parents=True, exist_ok=True)
                cols = () if st == "replay-control-ma" else bias.get(st, ("mz_bias_ppm_ms1", "mz_bias_ppm_ms2"))
                with sqlite3.connect(db) as c:
                    c.execute("create table ions (sequence text, charge integer"
                              + "".join(f", {k} real" for k in cols) + ")")
                    for (q, z), off in zip(SIM_IONS, (1.0, 3.0, 12.0, 25.0)):
                        c.execute(f"insert into ions values (?, ?{', ?' * len(cols)})", (q, z, *([off] * len(cols))))
                v = {"retention_time_gru_predictor": {}, "inv_mobility_gru_predictor": {}, "ion_keys": {"n": 4},
                     "replay_db": str(db)}
                if st != "replay-control-ma":
                    v["ion_keys"]["same_as_control"] = str(sroot / f"{rid}-replay-control-ma" / "out" / "x" / "s.db")
                    v.update({k: {"all_equal": True} for k in cols})
                _checked(sroot / f"{rid}-{st}", json.dumps(v))
    build(rows4)
    args = ["--mode", "stress", "--claim", "vm.mass-accuracy.fdp-bound.diann", "--recipe", recipe, "--sim-root",
            sim_root, "--out", out, "--stress-recipe", srec, "--stress-root", sroot]
    _run("vm_truth_fdp.py", *args)
    res = json.loads((out / "results" / "stress-mass-accuracy.json").read_text())
    assert res["control"] == "replay-control-ma" and res["reported_only"] == ["ms12-centred", "ms12-2x"]
    # ms12-2x has FDP 1/2 but is outside the claim scope
    assert res["per_seed"]["ms12-2x"]["jitter-250k-r1"]["fdp"] == pytest.approx(0.5)
    assert res["values"]["vm.mass-accuracy.fdp-bound.diann"]["fdp_max"] == pytest.approx(0)
    md = res["mass_deltas_ppm"]
    assert md["real"]["LS10"]["ms1_ppm"]["width90"] == pytest.approx(np.quantile(wide[0], .95) - np.quantile(wide[0], .05))
    assert md["baseline"]["jitter-250k-r1"]["fr_ppm"]["q50"] == pytest.approx(0)
    loss = res["loss_by_offset"]["ms12-real"]["jitter-250k-r1"]["mz_bias_ppm_ms1"]
    assert loss["[20.0, inf)"] == {"simulated": 1, "accepted": 1, "share": 1.0}     # VVVYAAAK, offset 25
    assert loss["[2.0, 4.0)"]["simulated"] == 1 and loss["[4.0, 6.0)"]["simulated"] == 0
    assert set(res["loss_by_offset"]["ms1-real"]["jitter-250k-r1"]) == {"mz_bias_ppm_ms1"}
    it = res["interpretation"]
    assert it["conclusion"] == "supports robustness to this construction" and it["failing"] == []

    # (a) fails when the stressed run's deltas stay narrow; (b) when ms12-real loses depth; (c) when the replay moves
    build(rows4, ms12_deltas=narrow)
    _run("vm_truth_fdp.py", *args)
    it = json.loads((out / "results" / "stress-mass-accuracy.json").read_text())["interpretation"]
    assert it["failing"] == ["a manipulation"] and it["conclusion"].startswith("uninformative")
    build(rows4[:2], ms12_deltas=([-10.0, 5.0], [-15.0, 10.0]))
    _run("vm_truth_fdp.py", *args)
    it = json.loads((out / "results" / "stress-mass-accuracy.json").read_text())["interpretation"]
    assert it["failing"] == ["b depth"] and it["depth_ratio"]["ms12-real"]["jitter-250k-r2"] == pytest.approx(0.5)
    build(rows4, ctrl_rows=rows4 + [("NEVERSIMK", "NEVERSIMK", 2, 0.005)])
    _run("vm_truth_fdp.py", *args)
    it = json.loads((out / "results" / "stress-mass-accuracy.json").read_text())["interpretation"]
    assert it["failing"] == ["c replay"] and it["replay_vs_baseline"]["jitter-250k-r3"]["fdp_diff"] == pytest.approx(0.2)
    # (0): a claim-scope FDP above 1.5% fails the claims and therefore the interpretation
    build(rows4 + [("KLMNLLR", "KLMNLLR", 2, 0.005)])                          # 1/5 false on ms12-real
    _run("vm_truth_fdp.py", *args)
    it = json.loads((out / "results" / "stress-mass-accuracy.json").read_text())["interpretation"]
    assert it["failing"][0] == "0 claims" and not it["claims_pass"]
    # an empty reported-only run is reported as undefined and does not block the assessment
    build(rows4, x2_rows=[("VVVYAAAK", "VVVYAAAK", 2, 0.04)])
    _run("vm_truth_fdp.py", *args)
    res = json.loads((out / "results" / "stress-mass-accuracy.json").read_text())
    assert res["per_seed"]["ms12-2x"]["jitter-250k-r1"]["fdp"] is None and res["interpretation"]["failing"] == []
    # an empty claim-scope run: the results and interpretation are written, the claims are not assessed
    build([("VVVYAAAK", "VVVYAAAK", 2, 0.04)])
    with pytest.raises(subprocess.CalledProcessError):
        _run("vm_truth_fdp.py", *args)
    res = json.loads((out / "results" / "stress-mass-accuracy.json").read_text())
    assert res["values"]["vm.mass-accuracy.fdp-bound.diann"]["fdp_max"] is None
    assert {"a manipulation", "b depth"} <= set(res["interpretation"]["failing"])
    # weak or mismatched verify evidence is not counted
    build(rows4)
    _run("vm_truth_fdp.py", *args)
    f = sroot / "jitter-250k-r1-ms12-real" / "verify.json"
    good = f.read_text()
    for broken in ({"drop": "same_as_control"}, {"control": "jitter-250k-r2"}, {"bias": "mz_bias_ppm_ms2"},
                   {"db": "elsewhere"}):
        v = json.loads(good)
        if "drop" in broken:
            del v["ion_keys"]["same_as_control"]
        if "control" in broken:
            v["ion_keys"]["same_as_control"] = str(sroot / "jitter-250k-r2-replay-control-ma" / "out" / "x" / "s.db")
        if "bias" in broken:
            del v["mz_bias_ppm_ms2"]
        if "db" in broken:
            v["replay_db"] = "/elsewhere/synthetic_data.db"
        f.write_text(json.dumps(v))
        with pytest.raises(subprocess.CalledProcessError):
            _run("vm_truth_fdp.py", *args)
    f.write_text(good)
    (sroot / "jitter-250k-r2-ms1-real" / "done.flag").write_text("othercommit\n")     # one replay at another commit
    with pytest.raises(subprocess.CalledProcessError):
        _run("vm_truth_fdp.py", *args)


def _stress_base_db(src: Path) -> None:
    with sqlite3.connect(src) as c:
        c.execute("create table peptides (peptide_id integer, sequence text, events integer, rt_sigma real, "
                  "rt_lambda real, retention_time_gru_predictor real)")
        c.executemany("insert into peptides values (?,?,?,?,?,?)",
                      [(2, "AC[UNIMOD:4]K", 5, 2.0, 0.1, 20.0), (1, "[UNIMOD:1]M[UNIMOD:35]K", 5, 2.0, 0.1, 10.0),
                       (3, "PEPK", 5, 2.0, 0.1, 30.0)])
        c.execute("create table ions (ion_id integer, peptide_id integer, sequence text, charge integer, "
                  "relative_abundance real, inv_mobility_gru_predictor real, inv_mobility_gru_predictor_std real)")
        c.executemany("insert into ions values (?,?,?,?,?,?,?)",
                      [(1, 1, "a", 2, 1.0, 0.9, 0.01), (2, 2, "b", 2, 1.0, 1.0, 0.01), (3, 3, "c", 2, 1.0, 1.1, 0.01)])
        c.execute("create table scans (scan integer, mobility real)")
        c.executemany("insert into scans values (?,?)", [(1, 0.6), (2, 1.6)])
        for t in ("proteins", "dia_ms_ms_info", "dia_ms_ms_windows"):
            c.execute(f"create table {t} (x integer)")


def test_stress_prepare_rewrites_only_its_columns(tmp_path, monkeypatch):
    import numpy as np
    import vm_stress_prepare as sp

    src = tmp_path / "base.db"
    _stress_base_db(src)
    seen = []
    monkeypatch.setattr(sp, "chronologer_hi", lambda seqs, model: seen.extend(seqs) or np.array([3.0, 1.0, 2.0]))
    argv = ["x", "--baseline-db", str(src), "--gradient", "100", "--seed", "4101", "--chronologer-model", "m.pt",
            "--out", str(tmp_path / "p")]
    monkeypatch.setattr(sys, "argv", argv + ["--expect-seed", "4102"])
    with pytest.raises(SystemExit):
        sp.main()
    monkeypatch.setattr(sys, "argv", argv + ["--expect-seed", "4101"])
    sp.main()
    assert sp.to_chronologer(seen[0]) == "[+42.01057]M[+15.99491]K"  # ordered by peptide_id

    def col(setting, q):
        with sqlite3.connect(tmp_path / "p" / setting / "synthetic_data.db") as c:
            return [r[0] for r in c.execute(q)]
    rt_q = "select retention_time_gru_predictor from peptides order by peptide_id"
    im_q = "select inv_mobility_gru_predictor from ions order by ion_id"
    assert col("replay-control", rt_q) == [10.0, 20.0, 30.0] and col("replay-control", im_q) == [0.9, 1.0, 1.1]
    assert col("rt-chronologer", rt_q) == [100.0, 0.0, 50.0] and col("rt-chronologer", im_q) == [0.9, 1.0, 1.1]
    assert col("im-std", rt_q) == [10.0, 20.0, 30.0] and col("im-std", im_q) != [0.9, 1.0, 1.1]
    assert col("joint", rt_q) == [100.0, 0.0, 50.0] and col("joint", im_q) == col("im-std", im_q)
    rep = json.loads((tmp_path / "p" / "prepare.json").read_text())
    assert rep["settings"]["rt-chronologer"]["rt"]["median_abs_change_over_width"] == pytest.approx(20.0 / 2.0)
    with pytest.raises(ValueError):
        sp.to_chronologer("PEPS[UNIMOD:21]K")
    assert sp.to_chronologer("[UNIMOD:1]GUC[UNIMOD:4]K") == "[+42.01057]GCC[+57.02146]K"  # amendment 1

    # verify: a replay that kept the prepared values passes, one that re-predicted RT fails
    prepared = tmp_path / "p" / "joint"
    replay = tmp_path / "replay.db"
    replay.write_bytes((prepared / "synthetic_data.db").read_bytes())
    monkeypatch.setattr(sys, "argv", ["x", "--verify", str(replay), "--prepared", str(prepared), "--out",
                                      str(tmp_path / "verify.json")])
    sp.main()
    with sqlite3.connect(replay) as c:  # a replay renumbers ion_id; matching is on (peptide_id, charge)
        c.execute("update ions set ion_id = ion_id + 1000")
    sp.main()
    with sqlite3.connect(replay) as c:
        c.execute("update peptides set retention_time_gru_predictor = 42.0")
    with pytest.raises(SystemExit):
        sp.main()


def test_stress_prepare_rt_spread(tmp_path, monkeypatch):
    """rt-spread rewrites only peptide RT, keeps the order, takes the real spread, and survives --verify."""
    import numpy as np
    import pandas as pd
    import vm_stress_prepare as sp

    src = tmp_path / "base.db"
    _stress_base_db(src)
    real = tmp_path / "LS10.parquet"
    pd.DataFrame({"Modified.Sequence": ["A", "A", "B", "C", "D", "E"], "RT": [1.0, 3.0, 4.0, 5.0, 6.0, 99.0],
                  "Q.Value": [0.001] * 5 + [0.5], "Decoy": [0] * 6}).to_parquet(real)
    argv = ["x", "--row", "rt-spread", "--baseline-db", str(src), "--gradient", "600", "--seed", "1",
            "--real-report", str(real), "--out", str(tmp_path / "p")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):                     # 4 real precursors < MIN_REAL_RT
        sp.main()
    monkeypatch.setattr(sp, "MIN_REAL_RT", 4)
    monkeypatch.setattr(sp, "chronologer_hi", lambda *a: (_ for _ in ()).throw(AssertionError("no Chronologer")))
    sp.main()
    db = tmp_path / "p" / "rt-spread" / "synthetic_data.db"
    with sqlite3.connect(db) as c:
        rt = [r[0] for r in c.execute("select retention_time_gru_predictor from peptides order by peptide_id")]
        im = [r[0] for r in c.execute("select inv_mobility_gru_predictor from ions order by ion_id")]
    # real RT (s): A median 2 min, B 4, C 5, D 6 -> knots [120, 240, 300, 360] at mid-point CDF 1/8 .. 7/8;
    # sim mid-ranks 1/6, 3/6, 5/6
    assert rt == pytest.approx(list(np.interp([1 / 6, 0.5, 5 / 6], [0.125, 0.375, 0.625, 0.875],
                                              [120.0, 240.0, 300.0, 360.0])))
    assert rt == sorted(rt) and im == [0.9, 1.0, 1.1]
    rep = json.loads((tmp_path / "p" / "prepare.json").read_text())
    assert rep["row"] == "rt-spread" and rep["n_real_precursors"] == 4 and list(rep["settings"]) == ["rt-spread"]
    assert rep["new_tie_share"] == 0 and len(rep["real_report_sha256"]) == 64
    replay = tmp_path / "replay.db"
    replay.write_bytes(db.read_bytes())
    monkeypatch.setattr(sys, "argv", ["x", "--verify", str(replay), "--prepared", str(db.parent), "--out",
                                      str(tmp_path / "verify.json")])
    sp.main()
    # the map: ties share a value, the order holds, a reversal is caught
    old = np.array([3.0, 1.0, 3.0, 2.0])
    new = sp.spread_map(old, np.array([0.0, 10.0, 20.0, 30.0]))
    assert new[0] == new[2] and new[1] < new[3] < new[0] and sp.order_preserved(old, new)
    assert not sp.order_preserved(old, new[::-1])
    # a real sample on a grid (repeated values) no longer merges distinct sim RTs (amendment 1)
    grid = np.repeat(60.0 + 0.2 * np.arange(500), 3)     # 500 grid points, each value repeated
    sim = np.linspace(0.0, 1.0, 1000)
    assert sp.new_tie_share(sim, sp.spread_map(sim, grid)) < 0.01
    assert sp.new_tie_share(sim, np.quantile(grid, (np.arange(1000) + 0.5) / 1000)) > 0.1   # the frozen rule did
    flat = sp.spread_map(np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 5.0, 5.0, 5.0]))   # one value merges all
    assert sp.order_preserved(np.array([1.0, 2.0, 3.0, 4.0]), flat)
    assert sp.new_tie_share(np.array([1.0, 2.0, 3.0, 4.0]), flat) == 1.0
    # verify rejects a NaN apex in the replay
    with sqlite3.connect(replay) as c:
        c.execute("update peptides set retention_time_gru_predictor = NULL where peptide_id = 1")
    with pytest.raises(SystemExit, match="non-finite"):
        sp.main()
    # a missing real RT fails loudly instead of being skipped
    pd.DataFrame({"Modified.Sequence": ["A", "B", "C", "D"], "RT": [1.0, None, 3.0, 4.0],
                  "Q.Value": [0.001] * 4, "Decoy": [0] * 4}).to_parquet(real)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="non-finite"):
        sp.main()
    real.unlink()
    pd.DataFrame({"Modified.Sequence": ["A", "A", "B", "C", "D"], "RT": [1.0, 3.0, 4.0, 5.0, 6.0],
                  "Q.Value": [0.001] * 5, "Decoy": [0] * 5}).to_parquet(real)
    monkeypatch.setattr(sys, "argv", argv[:6] + ["10"] + argv[7:])     # gradient 10 s: mapped RT outside it
    with pytest.raises(SystemExit, match="outside"):
        sp.main()


def _real_mass_report(path: Path, n: int, rng_seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(rng_seed)
    d = pd.DataFrame({"Precursor.Id": [f"P{i:03d}2" for i in range(n)][::-1], "Precursor.Mz": r.uniform(400, 900, n),
                      "Best.Fr.Mz": r.uniform(200, 1200, n), "Q.Value": 0.001, "Decoy": 0})
    d["Ms1.Apex.Mz.Delta"] = r.normal(-3, 3, n) * d["Precursor.Mz"] * 1e-6
    d["Best.Fr.Mz.Delta"] = r.normal(-3, 6, n) * d["Best.Fr.Mz"] * 1e-6
    d.to_parquet(path)
    return d


def test_stress_prepare_mass_accuracy(tmp_path, monkeypatch):
    """Each ion takes one donor's (MS1, fragment) ppm pair; settings add only their columns; --verify checks them."""
    import vm_stress_prepare as sp

    src = tmp_path / "base.db"
    _stress_base_db(src)
    real = tmp_path / "LS10.parquet"
    d = _real_mass_report(real, 8)
    d.loc[0, "Ms1.Apex.Mz.Delta"] = np.nan                                   # dropped and counted
    d.loc[1, "Q.Value"] = 0.5                                                 # not accepted
    d.to_parquet(real)
    argv = ["x", "--row", "mass-accuracy", "--baseline-db", str(src), "--gradient", "600", "--seed", "4101",
            "--expect-seed", "4101", "--real-report", str(real), "--out", str(tmp_path / "p")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="donors"):                          # 6 donors < MIN_DONORS
        sp.main()
    monkeypatch.setattr(sp, "MIN_DONORS", 6)
    sp.main()
    rep = json.loads((tmp_path / "p" / "prepare.json").read_text())
    assert rep["n_donors"] == 6 and rep["n_donors_dropped_nonfinite"] == 1 and rep["n_recipients"] == 3
    assert list(rep["settings"]) == list(sp.MA_SETTINGS)
    # the draw, recomputed from the frozen rule
    ok = d[(d["Q.Value"] <= 0.01) & d["Ms1.Apex.Mz.Delta"].notna()].sort_values("Precursor.Id")
    ms1 = (ok["Ms1.Apex.Mz.Delta"] / ok["Precursor.Mz"] * 1e6).to_numpy()
    fr = (ok["Best.Fr.Mz.Delta"] / ok["Best.Fr.Mz"] * 1e6).to_numpy()
    idx = np.random.default_rng(4101 + 11).integers(0, 6, size=3)
    a = pd.read_csv(tmp_path / "p" / "assignment.tsv", sep="\t")
    assert a["donor"].tolist() == ok["Precursor.Id"].to_numpy()[idx].tolist()
    assert rep["assignment_sha256"] == sp._sha256(tmp_path / "p" / "assignment.tsv")

    def cols(setting):
        db = tmp_path / "p" / setting / "synthetic_data.db"
        with sqlite3.connect(db) as c:
            names = [r[1] for r in c.execute("pragma table_info(ions)")]
            got = {k: [r[0] for r in c.execute(f"select {k} from ions order by peptide_id, charge")]
                   for k in names if k.startswith("mz_bias")}
        return got
    assert cols("replay-control-ma") == {}
    assert cols("ms1-real") == {"mz_bias_ppm_ms1": ms1[idx].tolist()}                 # exact
    assert cols("ms2-real") == {"mz_bias_ppm_ms2": fr[idx].tolist()}
    assert cols("ms12-real") == {"mz_bias_ppm_ms1": ms1[idx].tolist(), "mz_bias_ppm_ms2": fr[idx].tolist()}
    c = cols("ms12-centred")
    assert c["mz_bias_ppm_ms1"] == pytest.approx((ms1[idx] - np.median(ms1)).tolist())
    assert c["mz_bias_ppm_ms2"] == pytest.approx((fr[idx] - np.median(fr)).tolist())
    assert cols("ms12-2x")["mz_bias_ppm_ms2"] == pytest.approx((2 * fr[idx]).tolist())

    # verify: control first (no columns), then a biased replay against it
    out = tmp_path / "verify.json"
    ctrl = tmp_path / "ctrl.db"
    ctrl.write_bytes((tmp_path / "p" / "replay-control-ma" / "synthetic_data.db").read_bytes())
    monkeypatch.setattr(sys, "argv", ["x", "--verify", str(ctrl), "--prepared", str(tmp_path / "p" / "replay-control-ma"),
                                      "--out", str(out)])
    sp.main()
    assert json.loads(out.read_text())["ion_keys"]["n"] == 3
    prepared = tmp_path / "p" / "ms12-real"
    replay = tmp_path / "replay.db"
    replay.write_bytes((prepared / "synthetic_data.db").read_bytes())
    with sqlite3.connect(replay) as c:                                       # a replay renumbers ion_id
        c.execute("update ions set ion_id = ion_id + 1000")
    base_args = ["x", "--verify", str(replay), "--prepared", str(prepared), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", base_args)
    with pytest.raises(SystemExit, match="control-replay"):                 # biased settings need the control
        sp.main()
    monkeypatch.setattr(sys, "argv", base_args + ["--control-replay", str(ctrl)])
    sp.main()
    v = json.loads(out.read_text())
    assert v["mz_bias_ppm_ms1"]["all_equal"] and v["ion_keys"]["same_as_control"] == str(ctrl)
    with sqlite3.connect(ctrl) as c:                                         # control kept an ion the replay lost
        c.execute("insert into ions (ion_id, peptide_id, charge) values (9, 9, 2)")
    with pytest.raises(SystemExit, match="sets differ"):
        sp.main()
    with sqlite3.connect(ctrl) as c:
        c.execute("delete from ions where peptide_id = 9")
    with sqlite3.connect(replay) as c:                                       # the bias did not survive
        c.execute("update ions set mz_bias_ppm_ms2 = mz_bias_ppm_ms2 + 1 where peptide_id = 2")
    with pytest.raises(SystemExit, match="differs from the prepared"):
        sp.main()
    with sqlite3.connect(replay) as c:
        c.execute("alter table ions drop column mz_bias_ppm_ms2")
    with pytest.raises(SystemExit, match="dropped"):
        sp.main()
    # a duplicate donor id fails; a baseline that already has a bias column fails
    d2 = _real_mass_report(real, 8)
    d2.loc[2, "Precursor.Id"] = d2.loc[3, "Precursor.Id"]
    d2.to_parquet(real)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="twice"):
        sp.main()
    _real_mass_report(real, 8)
    with sqlite3.connect(src) as c:                                          # two ions share (peptide_id, charge)
        c.execute("insert into ions values (4, 3, 'd', 2, 1.0, 1.2, 0.01)")
    with pytest.raises(SystemExit, match="recipients"):
        sp.main()
    with sqlite3.connect(src) as c:
        c.execute("delete from ions where ion_id = 4")
        c.execute("alter table ions add column mz_bias_ppm_ms1 REAL")
    with pytest.raises(SystemExit, match="already"):
        sp.main()


def test_stress_prepare_rejects_constant_predictions(tmp_path, monkeypatch):
    import numpy as np
    import vm_stress_prepare as sp

    src = tmp_path / "base.db"
    _stress_base_db(src)
    monkeypatch.setattr(sp, "chronologer_hi", lambda seqs, model: np.ones(len(seqs)))
    monkeypatch.setattr(sys, "argv", ["x", "--baseline-db", str(src), "--gradient", "100", "--seed", "1",
                                      "--chronologer-model", "m.pt", "--out", str(tmp_path / "p")])
    with pytest.raises(ValueError):
        sp.main()


def test_diagnose_classes_and_twins(tree):
    """vm_diagnose (reported only): error classes, Met-ox twin, depth recall, q-curve coverage."""
    recipe, sim_root, out = tree
    ions = [("PEPTIDEK", 2, 400.0, 1e5), ("AC[UNIMOD:4]DEFGHK", 2, 500.0, 1e3), ("MLLNQSTR", 3, 330.0, 1e4),
            ("VVVYAAAK", 2, 440.0, 1e2), ("GGGGKR", 2, 290.0, 1e1)]
    rows = [("PEPTIDEK", "PEPTIDEK", 2, 0.001, 30.0), ("AC(UniMod:4)DEFGHK", "ACDEFGHK", 2, 0.002, 40.0),
            ("M(UniMod:35)ILNQSTR", "MILNQSTR", 3, 0.003, 50.05), ("MILNQSTR", "MILNQSTR", 3, 0.004, 50.0),
            ("KLMNLLR", "KLMNLLR", 2, 0.005, 60.0), ("PEPTIDEK", "PEPTIDEK", 3, 0.006, 30.0),
            ("VVVYAAAK", "VVVYAAAK", 2, 0.015, 70.0)]
    for run in SIM_RUNS:
        db = sim_root / run / "out" / "x" / "synthetic_data.db"
        db.unlink()
        with sqlite3.connect(db) as c:
            c.execute("create table peptides (peptide_id integer, events real, retention_time_gru_predictor real)")
            c.execute("create table ions (peptide_id integer, sequence text, charge integer, mz real, "
                      "relative_abundance real, inv_mobility_gru_predictor real)")
            for i, (s, z, mz, ev) in enumerate(ions):
                rt = {"PEPTIDEK": 30, "MLLNQSTR": 50}.get(s, 20) * 60.0 + 1000  # seconds
                c.execute("insert into peptides values (?, ?, ?)", (i, ev, rt))
                c.execute("insert into ions values (?, ?, ?, ?, 1.0, 0.9)", (i, s, z, mz))
        p = recipe / "sim" / run / "plain" / "report.parquet"
        pd.DataFrame({"Modified.Sequence": [r[0] for r in rows], "Stripped.Sequence": [r[1] for r in rows],
                      "Precursor.Charge": [r[2] for r in rows], "Q.Value": [r[3] for r in rows],
                      "RT": [r[4] for r in rows], "IM": 0.9, "Precursor.Mz": 450.0, "Protein.Ids": "P",
                      "Decoy": 0}).to_parquet(p)
    _run("vm_diagnose.py", "--recipe", recipe, "--sim-root", sim_root, "--out", out, "--search-q", "0.02")
    r = json.loads((out / "results" / "diagnose.json").read_text())["sim"]["jitter-250k-r1"]
    assert r["class_counts"] == {"true": 3, "form-met-ox": 1, "seq-never-simulated": 1, "seq-other-charge": 1}
    assert r["met_ox"]["twin_also_accepted"] == 1.0 and r["met_ox"]["same_peak_share_0p1min_of_paired"] == 1.0
    assert r["met_ox"]["true_m_containing_with_false_met_ox_sibling"] == 1.0
    assert r["met_ox"]["hit_rt_minus_twin_apex_min_q05_q50_q95"][1] == pytest.approx(50.05 - (50 * 60.0 + 1000) / 60)
    assert r["depth"]["simulated"] == 5 and r["depth"]["accepted_true"] == 3
    assert r["never_simulated"]["by_class"] == {"seq-never-simulated": 1, "seq-other-charge": 1}
    assert r["depth"]["by_intensity_decile"]["d10"]["recall"] == 1.0  # PEPTIDEK/2, the most intense
    curve = {c["q"]: c for c in r["q_curve"]}
    # A 2% search cut: the 5% point is marked not covered rather than undercounted.
    assert curve[0.01]["accepted"] == 6 and curve[0.02]["accepted"] == 7 and not curve[0.05]["covered"]


def test_diagnose_met_ox_hit_is_not_its_own_twin():
    """Simulated only AM(ox)K: an accepted unoxidised AMK is a false hit whose twin is AM(ox)K, not itself."""
    from vm_diagnose import classify, met_ox
    from vm_common import label_false
    t = pd.DataFrame({"mod": ["AM{35}K"], "charge": [2], "rt_min": [10.0], "im": [0.8], "intensity": [1.0],
                      "stripped": ["AMK"]})
    rep = pd.DataFrame({"mod": ["AMK"], "stripped": ["AMK"], "charge": [2], "protein": ["P"], "RT": [10.02], "IM": [0.8]})
    tm = {("AM{35}K", 2)}
    lab = classify(label_false(rep, tm, {("AMK", 2)}, np.zeros(1, bool)), tm)
    assert list(lab["cls"]) == ["form-met-ox"]
    r = met_ox(t, rep, lab)
    assert r["twin_also_accepted"] == 0.0 and r["paired_with_accepted_twin"] == 0
    assert r["same_peak_share_0p1min_of_paired"] is None
    assert r["hit_rt_minus_twin_apex_min_q05_q50_q95"][1] == pytest.approx(0.02)
