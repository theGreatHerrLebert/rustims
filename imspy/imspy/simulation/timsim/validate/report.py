"""
Report generation for timsim-validate.
"""

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np

from .metrics import ValidationMetrics, ValidationThresholds
from .plots import PlotPaths


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_version() -> str:
    """Get imspy version."""
    try:
        from importlib.metadata import version
        return version("imspy")
    except Exception:
        return "unknown"


def generate_json_report(
    metrics: ValidationMetrics,
    thresholds: ValidationThresholds,
    output_dir: str,
    simulation_path: Optional[str] = None,
    diann_report_path: Optional[str] = None,
    fasta_path: Optional[str] = None,
    plot_paths: Optional[PlotPaths] = None,
) -> str:
    """
    Generate machine-parseable JSON validation report.

    Args:
        metrics: Computed validation metrics.
        thresholds: Threshold configuration used.
        output_dir: Directory to write the report.
        simulation_path: Path to the simulated .d folder.
        diann_report_path: Path to DiaNN report.tsv.
        fasta_path: Path to FASTA file used.
        plot_paths: Paths to generated plots.

    Returns:
        Path to the generated JSON report.
    """
    # Build plots dict from PlotPaths
    plots_dict = {}
    if plot_paths:
        if plot_paths.summary_plot:
            plots_dict["summary"] = plot_paths.summary_plot
        if plot_paths.rt_correlation:
            plots_dict["rt_correlation"] = plot_paths.rt_correlation
        if plot_paths.im_correlation:
            plots_dict["im_correlation"] = plot_paths.im_correlation
        if plot_paths.intensity_histogram:
            plots_dict["intensity_histogram"] = plot_paths.intensity_histogram
        if plot_paths.quant_correlation:
            plots_dict["quant_correlation"] = plot_paths.quant_correlation

    report = {
        "metadata": {
            "tool": "timsim-validate",
            "version": get_version(),
            "timestamp": datetime.now().isoformat(),
            "paths": {
                "simulation": simulation_path,
                "diann_report": diann_report_path,
                "fasta": fasta_path,
            },
            "plots": plots_dict,
        },
        "configuration": {
            "thresholds": thresholds.to_dict(),
        },
        "metrics": metrics.to_dict(),
        "status": "PASS" if metrics.overall_pass else "FAIL",
    }

    output_path = os.path.join(output_dir, "validation_report.json")
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    return output_path


def generate_text_summary(
    metrics: ValidationMetrics,
    thresholds: ValidationThresholds,
) -> str:
    """
    Generate human-readable validation summary.

    Args:
        metrics: Computed validation metrics.
        thresholds: Threshold configuration used.

    Returns:
        Formatted text summary string.
    """
    status = "PASS" if metrics.overall_pass else "FAIL"
    checks = metrics.threshold_checks

    def check_mark(passed: bool) -> str:
        return "[PASS]" if passed else "[FAIL]"

    def format_float(val: float, decimals: int = 4) -> str:
        if val is None or (isinstance(val, float) and (val != val)):  # NaN check
            return "N/A"
        return f"{val:.{decimals}f}"

    summary = f"""
================================================================================
                        TIMSIM VALIDATION REPORT
================================================================================

Status: {status}

IDENTIFICATION METRICS
----------------------
Ground Truth Precursors:   {metrics.num_ground_truth:,}
Identified by DiaNN:       {metrics.num_identified:,}
False Positives:           {metrics.num_false_positives:,}
False Negatives:           {metrics.num_false_negatives:,}

Identification Rate:       {metrics.identification_rate:.1%} {check_mark(checks.get('identification_rate', False))}
                           (threshold: >= {thresholds.min_identification_rate:.0%})
Precision:                 {metrics.precision:.1%}
FDR:                       {metrics.fdr:.1%}

RETENTION TIME CORRELATION
--------------------------
Pearson R:                 {format_float(metrics.rt_pearson_r)} {check_mark(checks.get('rt_correlation', False))}
                           (threshold: >= {thresholds.min_rt_correlation:.2f})
Mean Absolute Error:       {format_float(metrics.rt_mae_minutes, 2)} min {check_mark(checks.get('rt_mae', False))}
                           (threshold: <= {thresholds.max_rt_mae_minutes:.2f} min)
Median Error:              {format_float(metrics.rt_median_error_minutes, 2)} min

ION MOBILITY CORRELATION
------------------------
Pearson R:                 {format_float(metrics.im_pearson_r)} {check_mark(checks.get('im_correlation', False))}
                           (threshold: >= {thresholds.min_im_correlation:.2f})
Mean Absolute Error:       {format_float(metrics.im_mae)} 1/K0 {check_mark(checks.get('im_mae', False))}
                           (threshold: <= {thresholds.max_im_mae:.4f} 1/K0)
Median Error:              {format_float(metrics.im_median_error)} 1/K0

QUANTIFICATION METRICS (informational)
--------------------------------------
Pearson R (log):           {format_float(metrics.quant_pearson_r)} {check_mark(checks.get('quant_correlation', False))}
                           (threshold: >= {thresholds.min_quant_correlation:.2f})
Spearman R:                {format_float(metrics.quant_spearman_r)}
"""

    # Charge state breakdown
    if metrics.charge_state_metrics:
        summary += """
CHARGE STATE BREAKDOWN
----------------------
"""
        summary += f"  {'Charge':<8} {'Ground Truth':>14} {'Identified':>12} {'ID Rate':>10}\n"
        summary += f"  {'-'*8} {'-'*14} {'-'*12} {'-'*10}\n"
        for charge, data in sorted(metrics.charge_state_metrics.items()):
            summary += f"  {charge}+{'':<6} {data['ground_truth']:>14,} {data['identified']:>12,} {data['identification_rate']:>9.1%}\n"

    # Intensity bin breakdown
    if metrics.intensity_bin_metrics:
        summary += """
INTENSITY DEPENDENCE
--------------------
"""
        summary += f"  {'Bin':<5} {'Intensity Range':<18} {'Ground Truth':>14} {'Identified':>12} {'ID Rate':>10}\n"
        summary += f"  {'-'*5} {'-'*18} {'-'*14} {'-'*12} {'-'*10}\n"
        for bin_idx, data in sorted(metrics.intensity_bin_metrics.items()):
            summary += f"  {bin_idx:<5} {data['intensity_range']:<18} {data['ground_truth']:>14,} {data['identified']:>12,} {data['identification_rate']:>9.1%}\n"

    # Mass accuracy
    if metrics.mass_accuracy_metrics and metrics.mass_accuracy_metrics.get("mean_ppm_error") is not None:
        summary += f"""
MASS ACCURACY
-------------
Mean Error:                {format_float(metrics.mass_accuracy_metrics.get('mean_ppm_error'), 2)} ppm
Median Error:              {format_float(metrics.mass_accuracy_metrics.get('median_ppm_error'), 2)} ppm
Std Dev:                   {format_float(metrics.mass_accuracy_metrics.get('std_ppm_error'), 2)} ppm
MAE:                       {format_float(metrics.mass_accuracy_metrics.get('mae_ppm'), 2)} ppm
"""

    # Peptide length breakdown
    if metrics.peptide_property_metrics and metrics.peptide_property_metrics.get("by_length"):
        summary += """
PEPTIDE LENGTH BREAKDOWN
------------------------
"""
        summary += f"  {'Length':<10} {'Ground Truth':>14} {'Identified':>12} {'ID Rate':>10}\n"
        summary += f"  {'-'*10} {'-'*14} {'-'*12} {'-'*10}\n"
        for length_range, data in metrics.peptide_property_metrics["by_length"].items():
            if data['ground_truth'] > 0:
                summary += f"  {length_range:<10} {data['ground_truth']:>14,} {data['identified']:>12,} {data['identification_rate']:>9.1%}\n"

    # Missed cleavages breakdown
    if metrics.peptide_property_metrics and metrics.peptide_property_metrics.get("by_missed_cleavages"):
        summary += """
MISSED CLEAVAGES BREAKDOWN
--------------------------
"""
        summary += f"  {'MC':<5} {'Ground Truth':>14} {'Identified':>12} {'ID Rate':>10}\n"
        summary += f"  {'-'*5} {'-'*14} {'-'*12} {'-'*10}\n"
        for mc, data in sorted(metrics.peptide_property_metrics["by_missed_cleavages"].items()):
            if data['ground_truth'] > 0:
                summary += f"  {mc:<5} {data['ground_truth']:>14,} {data['identified']:>12,} {data['identification_rate']:>9.1%}\n"

    summary += """
================================================================================
                           THRESHOLD SUMMARY
================================================================================
"""

    for check_name, passed in checks.items():
        summary += f"  {check_name:25s} {check_mark(passed)}\n"

    summary += f"""
================================================================================
                             FINAL RESULT: {status}
================================================================================
"""

    return summary


def save_text_report(
    summary: str,
    output_dir: str,
) -> str:
    """
    Save text summary to file.

    Args:
        summary: Text summary string.
        output_dir: Directory to write the report.

    Returns:
        Path to the generated text report.
    """
    output_path = os.path.join(output_dir, "validation_report.txt")
    with open(output_path, 'w') as f:
        f.write(summary)
    return output_path
