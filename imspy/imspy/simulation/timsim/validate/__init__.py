"""
timsim-validate: Automated validation of timsim simulations using DiaNN.

This module provides a CLI tool for running integration tests that:
1. Run a small timsim simulation
2. Analyze with DiaNN via subprocess
3. Compare DiaNN results to simulation ground truth
4. Generate pass/fail validation reports with visualizations
"""

from .metrics import ValidationMetrics, ValidationThresholds
from .runner import ValidationRunner, ValidationResult
from .plots import PlotPaths, generate_all_plots, generate_summary_figure

__all__ = [
    "ValidationMetrics",
    "ValidationThresholds",
    "ValidationRunner",
    "ValidationResult",
    "PlotPaths",
    "generate_all_plots",
    "generate_summary_figure",
]
