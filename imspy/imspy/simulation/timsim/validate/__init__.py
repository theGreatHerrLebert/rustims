"""
timsim-validate: Automated validation of timsim simulations using DiaNN or FragPipe.

This module provides a CLI tool for running integration tests that:
1. Run a small timsim simulation
2. Analyze with DiaNN or FragPipe via subprocess
3. Compare results to simulation ground truth
4. Generate pass/fail validation reports with visualizations
"""

from .metrics import ValidationMetrics, ValidationThresholds
from .runner import ValidationRunner, ValidationResult
from .plots import PlotPaths, generate_all_plots, generate_summary_figure
from .diann_executor import DiannExecutor, DiannConfig, DiannResult
from .fragpipe_executor import FragPipeExecutor, FragPipeConfig, FragPipeResult
from .parsing import parse_diann_report, parse_fragpipe_psm, parse_fragpipe_combined
from .tool_comparison import (
    run_comparison,
    compare_tools,
    ComparisonResult,
    IntensityBinMetrics,
    ChargeStateMetrics,
    generate_comparison_plots,
    generate_comparison_text_report,
)

__all__ = [
    "ValidationMetrics",
    "ValidationThresholds",
    "ValidationRunner",
    "ValidationResult",
    "PlotPaths",
    "generate_all_plots",
    "generate_summary_figure",
    # DiaNN
    "DiannExecutor",
    "DiannConfig",
    "DiannResult",
    # FragPipe
    "FragPipeExecutor",
    "FragPipeConfig",
    "FragPipeResult",
    # Parsing
    "parse_diann_report",
    "parse_fragpipe_psm",
    "parse_fragpipe_combined",
    # Comparison
    "run_comparison",
    "compare_tools",
    "ComparisonResult",
    "IntensityBinMetrics",
    "ChargeStateMetrics",
    "generate_comparison_plots",
    "generate_comparison_text_report",
]
