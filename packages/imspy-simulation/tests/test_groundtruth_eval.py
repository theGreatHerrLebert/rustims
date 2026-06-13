"""Unit tests for the ground-truth eval metric core (engine/instrument-agnostic)."""
import numpy as np
import pandas as pd

from imspy_simulation.timsim.groundtruth_eval import (
    evaluate, _strip_mods,
    _canon_inline, _canon_alphadia, _pf_key, _bb_key,
    precursor_fdp_breakdown,
)


def test_canonical_peptidoform_consistent_across_sources():
    # The SAME peptidoform must yield an identical canonical key whether it comes from
    # the truth DB (UNIMOD brackets), DiaNN ((UniMod:n) parens), or alphaDIA
    # (stripped + mod names + 1-based sites). Oxidation@M == UNIMOD:35 at residue 11.
    st_t, m_t = _canon_inline("AAALLAKQAEM[UNIMOD:35]EVK")     # truth
    st_d, m_d = _canon_inline("AAALLAKQAEM(UniMod:35)EVK")     # DiaNN
    st_a, m_a = _canon_alphadia("AAALLAKQAEMEVK", "Oxidation@M", "11")  # alphaDIA
    assert _pf_key(st_t, m_t) == _pf_key(st_d, m_d) == _pf_key(st_a, m_a) == "AAALLAKQAEMEVK/11:35"
    # I/L normalization collapses the isobaric ambiguity at backbone + peptidoform level.
    assert _bb_key("PEPTIDEK") == _bb_key("PEPTLDEK") == "PEPTLDEK"
    # Unknown alphaDIA mod -> -1 sentinel (cannot match a real truth id).
    _, m_u = _canon_alphadia("PEPK", "Glubbery@P", "1")
    assert m_u == {1: -1}


def test_precursor_fdp_breakdown_buckets():
    # Truth: PEPTLDEK/z2 with oxidation at M-less... use a clear case.
    # Simulated peptidoforms: ACDEK/z2 (no mod), MCDEK/z2 with ox@M1.
    truth = {
        "bb": {("ACDEK", 2), ("MCDEK", 2)},
        "pf": {("ACDEK/", 2), ("MCDEK/1:35", 2)},
        "bb_multiset": {("ACDEK", 2): {()}, ("MCDEK", 2): {(35,)}},
        "n": 2,
    }
    # Report: 1 correct, 1 wrong-backbone (genuine FP), 1 right-bb+ox wrong SITE
    # (localization-ambiguous: MCDEK has ox, but reported on... only one M, so emulate
    # via a 2-M backbone), 1 right-bb wrong-mod-composition (genuine).
    truth["bb"].add(("MMDEK", 2)); truth["pf"].add(("MMDEK/1:35", 2))
    truth["bb_multiset"][("MMDEK", 2)] = {(35,)}
    report = pd.DataFrame({
        "bb":  ["ACDEK",   "ZZZEK",      "MMDEK",      "MMDEK"],
        "pf":  ["ACDEK/",  "ZZZEK/",     "MMDEK/2:35", "MMDEK/1:4"],
        "charge":       [2,        2,            2,            2],
        "mod_multiset": [(),       (),           (35,),        (4,)],
        "q":            [0.001,    0.001,        0.001,        0.001],
    })
    bd = precursor_fdp_breakdown(truth, report, q_thresholds=(0.01,))[0]
    assert bd["n_precursors"] == 4
    assert bd["wrong_backbone"] == 1           # ZZZEK
    assert bd["localization_ambiguous"] == 1   # MMDEK ox wrong site (right mod multiset)
    assert bd["wrong_mod_composition"] == 1    # MMDEK carbamidomethyl (not a simulated mod set)
    # strict counts all 3 non-correct; genuine excludes the localization-ambiguous one.
    assert abs(bd["fdp_peptidoform_strict"] - 3 / 4) < 1e-9
    assert abs(bd["fdp_peptidoform_genuine"] - 2 / 4) < 1e-9


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
    assert m.fdp == 0.0 and m.n_false_discoveries == 0
    assert m.charge_accuracy == 1.0
    assert m.rt_pearson > 0.999 and m.rt_median_abs_error < 1e-9
    assert m.quant_log_pearson > 0.999


def test_fdp_and_charge_error():
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
    # FDP = backbones not simulated / total = 1 (XXXK) / 4
    assert m.n_false_discoveries == 1
    assert abs(m.fdp - 0.25) < 1e-9
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
