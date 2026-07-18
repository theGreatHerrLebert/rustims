"""``timsim-v2-eval`` — score a search-engine result against timsim **v2** answer keys.

Harness v1 (deliberately minimal): a DiaNN report + a v2 feature-space directory → recall /
precision / FDR, RT / IM / quant correlation, and pass/fail. It reuses ``validate/`` verbatim —
``parse_diann_report`` for the report, ``build_truth_from_v2`` for the truth, and the
``comparison``/``metrics`` scorers — so the metric definitions are shared with the v1 harness.
Matching is at the **backbone** (stripped-sequence + charge) level, exact for unmodified data;
peptidoform / FDR-calibration / PTM-localization scoring and Sage/DDA are later steps.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .comparison import (
    calculate_charge_state_metrics,
    calculate_correlation_metrics,
    calculate_identification_metrics,
    create_peptide_sets,
    match_results,
)
from .metrics import ValidationMetrics, ValidationThresholds, check_thresholds
from .parsing import parse_diann_report
from .v2_truth import build_truth_from_v2

_CORR_FIELDS = (
    "rt_pearson_r", "rt_pearson_p", "rt_mae_minutes", "rt_median_error_minutes",
    "im_pearson_r", "im_pearson_p", "im_mae", "im_median_error",
    "quant_pearson_r", "quant_pearson_p", "quant_spearman_r", "quant_spearman_p",
)


def aggregate_to_backbone(truth_df):
    """Collapse truth to one row per ``(sequence_normalized, charge)`` so the row-based scorers
    (match_results, correlations, charge breakdown) count precursors the same way the set-based
    identification does. Intensity is summed; the most-abundant row's RT/IM is the representative.
    A no-op for unmodified data (one modform per peptide); for modified data it makes backbone-level
    metrics well-defined, and it also merges I/L-equivalent backbones DiaNN can't tell apart.
    """
    df = truth_df.sort_values("intensity", ascending=False, kind="mergesort")
    agg = df.groupby(["sequence_normalized", "charge"], as_index=False).agg(
        sequence=("sequence", "first"),
        sequence_modified=("sequence_modified", "first"),
        sequence_modified_normalized=("sequence_modified_normalized", "first"),
        protein_id=("protein_id", "first"),
        rt=("rt", "first"),
        inverse_mobility=("inverse_mobility", "first"),
        intensity=("intensity", "sum"),
    )
    return agg


def recall_by_abundance(truth_df, identified_prec_set, n_bins=5):
    """Recall stratified into equal-COUNT quantiles of true abundance.

    Binning is rank-based (``qcut`` on the abundance rank), not on log-clipped intensity: v2's truth
    abundance (amol × propensities) is heavily tied near/below 1, so a value/log bin collapses. One row
    per identifiable precursor ``(sequence, charge)`` — duplicate backbones' abundances summed, matching
    how identification treats a precursor.
    """
    g = (
        truth_df[["sequence_normalized", "charge", "intensity"]]
        .groupby(["sequence_normalized", "charge"], as_index=False)["intensity"]
        .sum()
    )
    if len(g) == 0:
        return {}
    g["found"] = pd.Series(zip(g["sequence_normalized"], g["charge"])).isin(identified_prec_set).values
    n_bins = max(1, min(int(n_bins), len(g)))
    g["bin"] = pd.qcut(g["intensity"].rank(method="first"), n_bins, labels=False)
    out = {}
    for b in range(n_bins):
        sub = g[g["bin"] == b]
        lo, hi = float(sub["intensity"].min()), float(sub["intensity"].max())
        out[int(b)] = {
            "intensity_range": f"{lo:.1e}-{hi:.1e}",
            "abundance_range": [lo, hi],
            "ground_truth": int(len(sub)),
            "identified": int(sub["found"].sum()),
            "identification_rate": float(sub["found"].mean()) if len(sub) else 0.0,
        }
    return out


def score(report_df, truth_df, thresholds=None, n_abundance_bins=5):
    """Score a parsed report frame against a v2 truth frame → a ``ValidationMetrics``.

    Backbone matching: identification on ``(sequence, charge)`` sets; correlations on the
    ground-truth rows that DiaNN matched; recall stratified into ``n_abundance_bins`` quantiles.
    """
    thresholds = thresholds or ValidationThresholds()
    # Collapse to backbone precursors so every scorer counts precursors identically (see docstring).
    truth_df = aggregate_to_backbone(truth_df)
    _, gt_prec = create_peptide_sets(truth_df, use_modifications=False, normalize=True)
    _, id_prec = create_peptide_sets(report_df, use_modifications=False, normalize=True)
    idm = calculate_identification_metrics(gt_prec, id_prec)

    matched, unidentified, _fp = match_results(truth_df, report_df, match_on="sequence", normalize=True)
    corr = calculate_correlation_metrics(matched)

    m = ValidationMetrics(
        num_ground_truth=idm["true_positives"] + idm["false_negatives"],
        num_identified=idm["true_positives"],
        num_false_positives=idm["false_positives"],
        num_false_negatives=idm["false_negatives"],
        identification_rate=idm["identification_rate"],
        precision=idm["precision"],
        fdr=idm["fdr"],
        **{k: corr.get(k, np.nan) for k in _CORR_FIELDS},
    )
    # Recall stratified by true-abundance quantile — the interpretable form of recall when the truth
    # spans orders of magnitude (a flat recall is just the dynamic-range ceiling). Plus a per-charge
    # breakdown. Both reuse Stack B and populate fields ValidationMetrics already carries.
    m.intensity_bin_metrics = recall_by_abundance(truth_df, id_prec, n_bins=n_abundance_bins)
    m.charge_state_metrics = calculate_charge_state_metrics(truth_df, matched, unidentified)
    return check_thresholds(m, thresholds)


def _json_default(o):
    """Coerce numpy scalars (np.bool_/int/float from the metric math) to native JSON types."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def summary_text(m: ValidationMetrics) -> str:
    d = m.to_dict()
    i, rt, im, q = d["identification"], d["retention_time"], d["ion_mobility"], d["quantification"]
    lines = [
        "timsim v2 eval — DiaNN vs answer keys",
        f"  identifiable truth : {i['num_ground_truth']:,}",
        f"  identified (TP)    : {i['num_identified']:,}   recall {i['identification_rate']*100:.1f}%",
        f"  false positives    : {i['num_false_positives']:,}   precision {i['precision']*100:.1f}%   FDP {i['fdr']*100:.2f}%",
        f"  RT   Pearson r     : {_fmt(rt['pearson_r'])}   MAE {_fmt(rt['mae_minutes'])} min",
        f"  1/K0 Pearson r     : {_fmt(im['pearson_r'])}   MAE {_fmt(im['mae'], 4)}",
        f"  quant log Pearson  : {_fmt(q['pearson_r'])}   Spearman {_fmt(q['spearman_r'])}",
    ]
    bins = d.get("intensity_bin_breakdown") or {}
    if bins:
        lines.append("  recall by true-abundance quantile (low → high):")
        for b in sorted(bins, key=int):
            r = bins[b]
            lines.append(
                f"    bin {int(b)+1}  {r['intensity_range']:>13} :"
                f" {r['identification_rate']*100:5.1f}%   ({r['identified']:,}/{r['ground_truth']:,})"
            )
    checks = {k: bool(v) for k, v in d["threshold_checks"].items()}
    lines.append(f"  overall pass       : {'PASS' if d['overall_pass'] else 'FAIL'}  {checks}")
    return "\n".join(lines)


def _run_diann(script, d_path, out_dir):
    """Run a DiaNN wrapper ``script <d> <out_dir>`` (same interface as the pipeline's run_diann_*.sh)
    and return the produced report.parquet path."""
    subprocess.run(["bash", script, str(d_path), str(out_dir)], check=True)
    report = Path(out_dir) / "report.parquet"
    if not report.exists():
        raise FileNotFoundError(f"DiaNN did not produce {report}")
    return str(report)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="timsim-v2-eval",
        description="score a DiaNN result against timsim v2 answer keys (recall/FDR/RT/IM/quant)",
    )
    ap.add_argument("--truth-dir", required=True, type=Path,
                    help="v2 feature-space directory (precursors_inrange, peptides, peptide_rt, "
                         "precursor_ccs, peptide_quantities, ion_spectra_ce)")
    ap.add_argument("--report", type=Path,
                    help="existing DiaNN report.parquet to score (skips running DiaNN)")
    ap.add_argument("--d", type=Path, help="rendered .d to search (with --run-diann)")
    ap.add_argument("--run-diann", type=Path,
                    help="DiaNN wrapper script; invoked as `script <d> <out_dir>`")
    ap.add_argument("--n-frames", type=int, required=True, help="render --n-frames (RT map)")
    ap.add_argument("--cycle-seconds", type=float, required=True, help="render --cycle-seconds (RT map)")
    ap.add_argument("--sample", default=None, help="which design sample was rendered (default: first)")
    ap.add_argument("--fdr", type=float, default=0.01, help="report q-value cutoff")
    ap.add_argument("--abundance-bins", type=int, default=5,
                    help="equal-count true-abundance quantiles for the recall breakdown")
    ap.add_argument("--diann-v1", action="store_true", help="report is DiaNN 1.9 (default: 2.x)")
    ap.add_argument("--out", type=Path, help="write metrics JSON here")
    a = ap.parse_args(argv)

    if a.report is None:
        if a.d is None or a.run_diann is None:
            ap.error("provide --report, or both --d and --run-diann")
        out_dir = (a.out.parent if a.out else Path.cwd()) / "diann_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = _run_diann(str(a.run_diann), a.d, out_dir)
    else:
        report_path = str(a.report)

    report_df = parse_diann_report(report_path, diann_version_2=not a.diann_v1, fdr_threshold=a.fdr)
    truth_df = build_truth_from_v2(str(a.truth_dir), a.n_frames, a.cycle_seconds, sample=a.sample)
    metrics = score(report_df, truth_df, n_abundance_bins=a.abundance_bins)

    print(summary_text(metrics))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(metrics.to_dict(), indent=2, default=_json_default))
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
