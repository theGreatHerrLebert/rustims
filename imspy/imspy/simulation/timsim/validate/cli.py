"""
Command-line interface for timsim-validate.
"""

import argparse
import logging
import sys
import os

from .runner import ValidationRunner, SimulationConfig
from .diann_executor import DiannExecutor
from .metrics import ValidationThresholds


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging for the CLI."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def banner() -> str:
    """Return the application banner."""
    return """
================================================================================
                              TIMSIM-VALIDATE
================================================================================
  Automated validation of timsim simulations using DiaNN analysis
================================================================================
"""


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for timsim-validate."""
    description = """
Validate timsim simulation output by running DiaNN analysis and comparing
the identified peptides with the simulation ground truth.

This tool:
1. Runs a small timsim simulation (or uses existing)
2. Analyzes the simulated data with DiaNN
3. Compares DiaNN results to the simulation ground truth
4. Reports validation metrics (identification rate, RT/IM correlation)
"""

    epilog = """
Examples:
  timsim-validate /path/to/reference.d
  timsim-validate /path/to/reference.d --fasta /path/to/custom.fasta
  timsim-validate /path/to/reference.d --min-id-rate 0.4 --verbose

Exit codes:
  0: Validation PASSED
  1: Validation FAILED (threshold not met)
  2: Simulation error
  3: DiaNN error
  4: Comparison/parsing error
  5: Other error
"""

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "reference_path",
        type=str,
        help="Path to reference .d folder for instrument layout",
    )

    # Path arguments
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument(
        "--fasta",
        type=str,
        default=None,
        help="Custom FASTA file (default: bundled test proteome)",
    )
    path_group.add_argument(
        "--diann-path",
        type=str,
        default="diann",
        help="Path to DiaNN executable (default: 'diann' in PATH)",
    )
    path_group.add_argument(
        "--output-dir",
        type=str,
        default="./timsim-validate-output",
        help="Output directory for results (default: ./timsim-validate-output)",
    )

    # Simulation arguments
    sim_group = parser.add_argument_group("Simulation")
    sim_group.add_argument(
        "--num-peptides",
        type=int,
        default=5000,
        help="Number of peptides to simulate (default: 5000)",
    )
    sim_group.add_argument(
        "--gradient-length",
        type=float,
        default=1800.0,
        help="Gradient length in seconds (default: 1800 = 30 minutes)",
    )

    # Threshold arguments
    thresh_group = parser.add_argument_group("Validation Thresholds")
    thresh_group.add_argument(
        "--min-id-rate",
        type=float,
        default=0.30,
        help="Minimum identification rate threshold (default: 0.30)",
    )
    thresh_group.add_argument(
        "--min-rt-corr",
        type=float,
        default=0.90,
        help="Minimum RT correlation threshold (default: 0.90)",
    )
    thresh_group.add_argument(
        "--min-im-corr",
        type=float,
        default=0.90,
        help="Minimum IM correlation threshold (default: 0.90)",
    )
    thresh_group.add_argument(
        "--max-rt-mae",
        type=float,
        default=1.0,
        help="Maximum RT mean absolute error in minutes (default: 1.0)",
    )
    thresh_group.add_argument(
        "--max-im-mae",
        type=float,
        default=0.05,
        help="Maximum IM mean absolute error in 1/K0 (default: 0.05)",
    )

    # DiaNN arguments
    diann_group = parser.add_argument_group("DiaNN Options")
    diann_group.add_argument(
        "--diann-threads",
        type=int,
        default=4,
        help="Number of threads for DiaNN (default: 4)",
    )
    diann_group.add_argument(
        "--diann-timeout",
        type=int,
        default=3600,
        help="DiaNN timeout in seconds (default: 3600 = 1 hour)",
    )

    # Behavior arguments
    behavior_group = parser.add_argument_group("Behavior")
    behavior_group.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary simulation files",
    )
    behavior_group.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    behavior_group.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output, only final status",
    )

    return parser


def main() -> int:
    """Main entry point, returns exit code."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    # Print banner unless quiet
    if not args.quiet:
        print(banner())

    # Validate reference path
    if not os.path.exists(args.reference_path):
        print(f"Error: Reference path does not exist: {args.reference_path}", file=sys.stderr)
        return 5

    # Create configuration objects
    thresholds = ValidationThresholds(
        min_identification_rate=args.min_id_rate,
        min_rt_correlation=args.min_rt_corr,
        min_im_correlation=args.min_im_corr,
        max_rt_mae_minutes=args.max_rt_mae,
        max_im_mae=args.max_im_mae,
    )

    simulation_config = SimulationConfig(
        num_peptides=args.num_peptides,
        gradient_length=args.gradient_length,
    )

    diann_executor = DiannExecutor(
        executable_path=args.diann_path,
        threads=args.diann_threads,
        timeout_seconds=args.diann_timeout,
    )

    # Create and run validator
    runner = ValidationRunner(
        reference_path=args.reference_path,
        output_dir=args.output_dir,
        fasta_path=args.fasta,
        diann_executor=diann_executor,
        thresholds=thresholds,
        simulation_config=simulation_config,
        keep_temp=args.keep_temp,
        verbose=args.verbose,
    )

    result = runner.run()

    # Print final status
    if not args.quiet:
        if result.success:
            print("\n" + "=" * 80)
            print("  VALIDATION PASSED")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("  VALIDATION FAILED")
            if result.error_message:
                print(f"  Error: {result.error_message}")
            print("=" * 80)

        if result.report_path:
            print(f"\nReports saved to:")
            print(f"  JSON: {result.report_path}")
            if result.text_report_path:
                print(f"  Text: {result.text_report_path}")

    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
