"""The multi-vendor benchmark consolidator: eval-JSON summary + table rendering."""

import json

import pytest

from imspy_simulation.timsim.benchmark.multi_vendor import (
    summarize_eval,
    build_rows,
    render_markdown,
)


def _write_eval(tmp_path, name, observable, curve):
    p = tmp_path / name
    p.write_text(json.dumps({"rendered_observable_precursors": observable,
                             "engine": "diann", "curve": curve}))
    return str(p)


def test_summarize_eval_picks_the_requested_q_and_computes_recall(tmp_path):
    path = _write_eval(tmp_path, "w.json", 1000, [
        {"q": 0.001, "n_precursors": 700, "fdp_backbone": 0.0, "fdp_peptidoform_genuine": 0.0},
        {"q": 0.01, "n_precursors": 800, "fdp_backbone": 0.0, "fdp_peptidoform_genuine": 0.0},
        {"q": 0.05, "n_precursors": 810, "fdp_backbone": 0.01, "fdp_peptidoform_genuine": 0.0},
    ])
    s = summarize_eval(path, q=0.01)
    assert s["observable"] == 1000 and s["ids"] == 800
    assert s["recall_pct"] == pytest.approx(80.0)
    assert s["fdp_backbone"] == 0.0


def test_summarize_eval_picks_largest_q_at_or_below_target(tmp_path):
    # FDR semantics: never report a point ABOVE the requested threshold. With points
    # below (0.001, 0.005) and above (0.02) the target 0.01, pick 0.005 (largest <= 0.01),
    # NOT the numerically-nearest 0.02.
    path = _write_eval(tmp_path, "w.json", 100, [
        {"q": 0.001, "n_precursors": 30, "fdp_backbone": 0.0, "fdp_peptidoform_genuine": 0.0},
        {"q": 0.005, "n_precursors": 40, "fdp_backbone": 0.0, "fdp_peptidoform_genuine": 0.0},
        {"q": 0.02, "n_precursors": 90, "fdp_backbone": 0.0, "fdp_peptidoform_genuine": 0.0},
    ])
    s = summarize_eval(path, q=0.01)
    assert s["q"] == 0.005 and s["ids"] == 40


def test_summarize_eval_falls_back_when_curve_starts_above_target(tmp_path):
    # No point at/below 0.01 -> fall back to the smallest q (best available), and report it.
    path = _write_eval(tmp_path, "w.json", 100, [
        {"q": 0.02, "n_precursors": 60, "fdp_backbone": 0.0, "fdp_peptidoform_genuine": 0.0},
        {"q": 0.05, "n_precursors": 80, "fdp_backbone": 0.0, "fdp_peptidoform_genuine": 0.0},
    ])
    s = summarize_eval(path, q=0.01)
    assert s["q"] == 0.02


def test_build_rows_engine_falls_back_to_eval_json(tmp_path):
    # Manifest omits 'engine' -> use the engine recorded in the eval JSON ('diann').
    path = _write_eval(tmp_path, "w.json", 100, [
        {"q": 0.01, "n_precursors": 50, "fdp_backbone": 0.0, "fdp_peptidoform_genuine": 0.0},
    ])
    rows = build_rows({"vendors": [{"instrument": "waters_synapt_xs", "eval_json": path}]})
    assert rows[0].engine == "diann"


def test_build_rows_handles_live_and_recorded(tmp_path):
    path = _write_eval(tmp_path, "sciex.json", 6049, [
        {"q": 0.01, "n_precursors": 4957, "fdp_backbone": 0.0, "fdp_peptidoform_genuine": 0.0},
    ])
    manifest = {"q": 0.01, "vendors": [
        {"instrument": "sciex_zenotof", "engine": "alphadia", "eval_json": path, "source": "live"},
        {"instrument": "bruker_timstof", "engine": "diann", "source": "recorded",
         "recorded": {"fdp_backbone": 0.0075, "fdp_peptidoform": 0.0085}, "note": "prior run"},
    ]}
    rows = build_rows(manifest)
    assert [r.instrument for r in rows] == ["sciex_zenotof", "bruker_timstof"]
    live = rows[0]
    assert live.ids == 4957 and live.recall_pct == pytest.approx(81.9, abs=0.1)
    assert live.display == "SCIEX ZenoTOF" and live.output == "mzML"
    rec = rows[1]
    assert rec.ids is None and rec.fdp_backbone == 0.0075 and rec.source == "recorded"

    md = render_markdown(rows, q=0.01)
    assert "SCIEX ZenoTOF" in md and "Waters" not in md
    assert "81.9%" in md and "0.00%" in md   # live recall + FDP
    assert "0.75%" in md                       # recorded FDP rendered as percent
    assert "prior run" in md                   # note rendered


def test_missing_metrics_source_raises():
    with pytest.raises(ValueError):
        build_rows({"vendors": [{"instrument": "waters_synapt_xs", "engine": "diann"}]})
