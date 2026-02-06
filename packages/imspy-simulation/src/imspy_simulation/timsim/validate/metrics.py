"""
Validation metrics and thresholds for timsim-validate.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


@dataclass
class ValidationThresholds:
    """Pass/fail thresholds for validation metrics."""

    # Identification thresholds
    min_identification_rate: float = 0.30  # 30% of peptides should be found

    # RT correlation thresholds
    min_rt_correlation: float = 0.90  # Pearson R
    max_rt_mae_minutes: float = 1.0  # Mean absolute error in minutes

    # IM correlation thresholds
    min_im_correlation: float = 0.90  # Pearson R
    max_im_mae: float = 0.05  # Mean absolute error in 1/K0 units

    # Quantification thresholds (informational, not required for pass)
    min_quant_correlation: float = 0.70  # Pearson R on log-transformed intensities

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "min_identification_rate": self.min_identification_rate,
            "min_rt_correlation": self.min_rt_correlation,
            "max_rt_mae_minutes": self.max_rt_mae_minutes,
            "min_im_correlation": self.min_im_correlation,
            "max_im_mae": self.max_im_mae,
            "min_quant_correlation": self.min_quant_correlation,
        }


@dataclass
class ValidationMetrics:
    """All validation metrics computed from comparison."""

    # Identification metrics
    num_ground_truth: int = 0
    num_identified: int = 0
    num_false_positives: int = 0
    num_false_negatives: int = 0
    identification_rate: float = 0.0
    precision: float = 0.0
    fdr: float = 0.0

    # RT correlation metrics
    rt_pearson_r: float = np.nan
    rt_pearson_p: float = np.nan
    rt_mae_minutes: float = np.nan
    rt_median_error_minutes: float = np.nan

    # IM correlation metrics
    im_pearson_r: float = np.nan
    im_pearson_p: float = np.nan
    im_mae: float = np.nan
    im_median_error: float = np.nan

    # Quantification metrics
    quant_pearson_r: float = np.nan
    quant_pearson_p: float = np.nan
    quant_spearman_r: float = np.nan
    quant_spearman_p: float = np.nan

    # Detailed breakdown metrics
    charge_state_metrics: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    intensity_bin_metrics: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    mass_accuracy_metrics: Dict[str, Any] = field(default_factory=dict)
    peptide_property_metrics: Dict[str, Any] = field(default_factory=dict)

    # Overall result
    overall_pass: bool = False
    threshold_checks: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "identification": {
                "num_ground_truth": self.num_ground_truth,
                "num_identified": self.num_identified,
                "num_false_positives": self.num_false_positives,
                "num_false_negatives": self.num_false_negatives,
                "identification_rate": self.identification_rate,
                "precision": self.precision,
                "fdr": self.fdr,
            },
            "retention_time": {
                "pearson_r": self.rt_pearson_r if not np.isnan(self.rt_pearson_r) else None,
                "pearson_p": self.rt_pearson_p if not np.isnan(self.rt_pearson_p) else None,
                "mae_minutes": self.rt_mae_minutes if not np.isnan(self.rt_mae_minutes) else None,
                "median_error_minutes": self.rt_median_error_minutes if not np.isnan(self.rt_median_error_minutes) else None,
            },
            "ion_mobility": {
                "pearson_r": self.im_pearson_r if not np.isnan(self.im_pearson_r) else None,
                "pearson_p": self.im_pearson_p if not np.isnan(self.im_pearson_p) else None,
                "mae": self.im_mae if not np.isnan(self.im_mae) else None,
                "median_error": self.im_median_error if not np.isnan(self.im_median_error) else None,
            },
            "quantification": {
                "pearson_r": self.quant_pearson_r if not np.isnan(self.quant_pearson_r) else None,
                "pearson_p": self.quant_pearson_p if not np.isnan(self.quant_pearson_p) else None,
                "spearman_r": self.quant_spearman_r if not np.isnan(self.quant_spearman_r) else None,
                "spearman_p": self.quant_spearman_p if not np.isnan(self.quant_spearman_p) else None,
            },
            "charge_state_breakdown": self.charge_state_metrics,
            "intensity_bin_breakdown": self.intensity_bin_metrics,
            "mass_accuracy": self.mass_accuracy_metrics,
            "peptide_properties": self.peptide_property_metrics,
            "overall_pass": self.overall_pass,
            "threshold_checks": self.threshold_checks,
        }


def check_thresholds(
    metrics: ValidationMetrics,
    thresholds: ValidationThresholds,
) -> ValidationMetrics:
    """
    Check metrics against thresholds and update the metrics object.

    Args:
        metrics: ValidationMetrics object with computed values.
        thresholds: ValidationThresholds defining pass/fail criteria.

    Returns:
        Updated ValidationMetrics with threshold_checks and overall_pass set.
    """
    checks = {}

    # Identification rate check
    checks["identification_rate"] = metrics.identification_rate >= thresholds.min_identification_rate

    # RT correlation check
    if not np.isnan(metrics.rt_pearson_r):
        checks["rt_correlation"] = metrics.rt_pearson_r >= thresholds.min_rt_correlation
    else:
        checks["rt_correlation"] = False

    # RT MAE check
    if not np.isnan(metrics.rt_mae_minutes):
        checks["rt_mae"] = metrics.rt_mae_minutes <= thresholds.max_rt_mae_minutes
    else:
        checks["rt_mae"] = False

    # IM correlation check
    if not np.isnan(metrics.im_pearson_r):
        checks["im_correlation"] = metrics.im_pearson_r >= thresholds.min_im_correlation
    else:
        checks["im_correlation"] = False

    # IM MAE check
    if not np.isnan(metrics.im_mae):
        checks["im_mae"] = metrics.im_mae <= thresholds.max_im_mae
    else:
        checks["im_mae"] = False

    # Quantification check (informational)
    if not np.isnan(metrics.quant_pearson_r):
        checks["quant_correlation"] = metrics.quant_pearson_r >= thresholds.min_quant_correlation
    else:
        checks["quant_correlation"] = False

    # Overall pass requires: identification rate, RT correlation, IM correlation
    # MAE checks are secondary
    required_checks = [
        checks["identification_rate"],
        checks["rt_correlation"],
        checks["im_correlation"],
    ]

    metrics.threshold_checks = checks
    metrics.overall_pass = all(required_checks)

    return metrics


def create_metrics_from_comparison(
    id_metrics: dict,
    correlation_metrics: dict,
    num_ground_truth: int,
) -> ValidationMetrics:
    """
    Create ValidationMetrics from comparison results.

    Args:
        id_metrics: Dictionary from calculate_identification_metrics.
        correlation_metrics: Dictionary from calculate_correlation_metrics.
        num_ground_truth: Total number of ground truth precursors.

    Returns:
        ValidationMetrics object with all values populated.
    """
    metrics = ValidationMetrics()

    # Identification metrics
    metrics.num_ground_truth = num_ground_truth
    metrics.num_identified = id_metrics.get("true_positives", 0)
    metrics.num_false_positives = id_metrics.get("false_positives", 0)
    metrics.num_false_negatives = id_metrics.get("false_negatives", 0)
    metrics.identification_rate = id_metrics.get("identification_rate", 0.0)
    metrics.precision = id_metrics.get("precision", 0.0)
    metrics.fdr = id_metrics.get("fdr", 0.0)

    # RT metrics
    metrics.rt_pearson_r = correlation_metrics.get("rt_pearson_r", np.nan)
    metrics.rt_pearson_p = correlation_metrics.get("rt_pearson_p", np.nan)
    metrics.rt_mae_minutes = correlation_metrics.get("rt_mae_minutes", np.nan)
    metrics.rt_median_error_minutes = correlation_metrics.get("rt_median_error_minutes", np.nan)

    # IM metrics
    metrics.im_pearson_r = correlation_metrics.get("im_pearson_r", np.nan)
    metrics.im_pearson_p = correlation_metrics.get("im_pearson_p", np.nan)
    metrics.im_mae = correlation_metrics.get("im_mae", np.nan)
    metrics.im_median_error = correlation_metrics.get("im_median_error", np.nan)

    # Quantification metrics
    metrics.quant_pearson_r = correlation_metrics.get("quant_pearson_r", np.nan)
    metrics.quant_pearson_p = correlation_metrics.get("quant_pearson_p", np.nan)
    metrics.quant_spearman_r = correlation_metrics.get("quant_spearman_r", np.nan)
    metrics.quant_spearman_p = correlation_metrics.get("quant_spearman_p", np.nan)

    return metrics
