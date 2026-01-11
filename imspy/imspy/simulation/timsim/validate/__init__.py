"""
timsim-validate: Automated validation of timsim simulations using DiaNN.

This module provides a CLI tool for running integration tests that:
1. Run a small timsim simulation
2. Analyze with DiaNN via subprocess
3. Compare DiaNN results to simulation ground truth
4. Generate pass/fail validation reports
"""

from .metrics import ValidationMetrics, ValidationThresholds
from .runner import ValidationRunner, ValidationResult

__all__ = [
    "ValidationMetrics",
    "ValidationThresholds",
    "ValidationRunner",
    "ValidationResult",
]
