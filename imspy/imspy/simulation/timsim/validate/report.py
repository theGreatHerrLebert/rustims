"""
Report generation for timsim-validate.
"""

import json
import os
from datetime import datetime
from typing import Optional

from .metrics import ValidationMetrics, ValidationThresholds


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

    Returns:
        Path to the generated JSON report.
    """
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
        },
        "configuration": {
            "thresholds": thresholds.to_dict(),
        },
        "metrics": metrics.to_dict(),
        "status": "PASS" if metrics.overall_pass else "FAIL",
    }

    output_path = os.path.join(output_dir, "validation_report.json")
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

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
