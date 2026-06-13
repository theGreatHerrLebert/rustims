"""Unit tests for the ground-truth eval metric core (engine/instrument-agnostic)."""
import numpy as np
import pandas as pd

from imspy_simulation.timsim.groundtruth_eval import evaluate, _strip_mods


def _truth():
    # 4 simulated precursors over 3 backbones; B is simulated at z2 AND z3.
    return pd.DataFrame({
        "bb":           ["AAAK", "BBBR", "BBBR", "CCCK"],
        "charge":       [2,      2,      3,      2],
        "sim_rt":       [5.0,    10.0,   10.0,   15.0],   # minutes
        "sim_quantity": [1e3,    1e5,    1e4,    1e7],
    })


def test_strip_mods():
    assert _strip_mods("M[UNIMOD:35]C[UNIMOD:4]K") == "MCK"
    assert _strip_mods("[UNIMOD:1]PEPK") == "PEPK"


def test_perfect_report_metrics():
    truth = _truth()
    # report identifies AAAK/z2, BBBR/z2, CCCK/z2 — all real, correct charges, exact RT,
    # quantity proportional to simulated abundance (log-correlated).
    report = pd.DataFrame({
        "bb":       ["AAAK", "BBBR", "CCCK"],
        "charge":   [2,      2,      2],
        "rt":       [5.0,    10.0,   15.0],
        "quantity": [2e3,    2e5,    2e7],   # 2x sim -> perfect log-log correlation
    })
    m = evaluate(truth, report)
    assert m.identified == 3 and m.reported_total == 3 and m.simulated_ions == 4
    assert m.empirical_fdr == 0.0 and m.empirical_fdr_n_false == 0
    assert m.charge_accuracy == 1.0
    assert m.rt_pearson > 0.999 and m.rt_median_abs_error < 1e-9
    assert m.quant_log_pearson > 0.999


def test_entrapment_fdr_and_charge_error():
    truth = _truth()
    # 4 reported: 2 real (AAAK/z2, BBBR/z3), 1 wrong-charge (CCCK/z4 — backbone real,
    # charge not simulated), 1 entrapment false (XXXK not simulated at all).
    report = pd.DataFrame({
        "bb":       ["AAAK", "BBBR", "CCCK", "XXXK"],
        "charge":   [2,      3,      4,      2],
        "rt":       [5.0,    10.0,   15.0,   20.0],
        "quantity": [1e3,    1e4,    1e7,    1e2],
    })
    m = evaluate(truth, report)
    # empirical FDR = backbones not simulated / total = 1 (XXXK) / 4
    assert m.empirical_fdr_n_false == 1
    assert abs(m.empirical_fdr - 0.25) < 1e-9
    # matched-backbone IDs = AAAK,BBBR,CCCK (3); CCCK@z4 not a simulated charge -> 2/3
    assert abs(m.charge_accuracy - (2 / 3)) < 1e-9
    # only backbone+charge that match a simulated ion count as 'identified': AAAK/z2, BBBR/z3
    assert m.identified == 2


def test_recall_by_abundance_rises_with_quantity():
    # 8 simulated precursors spanning abundance; report finds only the brightest 4.
    truth = pd.DataFrame({
        "bb": [f"PEP{i}K" for i in range(8)],
        "charge": [2] * 8,
        "sim_rt": list(np.linspace(1, 30, 8)),
        "sim_quantity": [10 ** e for e in range(8)],   # 1e0 .. 1e7
    })
    bright = [f"PEP{i}K" for i in range(4, 8)]          # top-4 abundance
    report = pd.DataFrame({"bb": bright, "charge": [2] * 4,
                           "rt": [1.0] * 4, "quantity": [1e6] * 4})
    m = evaluate(truth, report)
    q = m.recall_by_abundance_quartile
    assert q["Q1_low"] == 0.0 and q["Q4_high"] == 1.0   # detection tracks abundance
