#!/usr/bin/env python3
"""
timsim-integration-eval: Evaluation runner for integration tests.

This script runs analysis tools (DiaNN, FragPipe) on simulated datasets
and validates results against ground truth.

Usage:
    timsim-integration-eval --env env.toml --all
    timsim-integration-eval --env env.toml --test IT-DIA-HELA --tool diann
    timsim-integration-eval --env env.toml --test IT-DIA-HELA --tool both
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import toml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Available test IDs
AVAILABLE_TESTS = [
    "IT-DIA-HELA",
    "IT-DIA-HYE",
    "IT-DIA-PHOS",
    "IT-DDA-TOPN",
    "IT-DDA-HLA",
]


@dataclass
class TestThresholds:
    """Pass/fail thresholds for a test."""
    min_id_rate: float = 0.30
    min_rt_correlation: float = 0.90
    min_im_correlation: float = 0.90


def get_integration_dir() -> Path:
    """Get the integration test directory."""
    return Path(__file__).parent


def load_env_config(env_path: str) -> Dict:
    """Load environment configuration from TOML file."""
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Environment config not found: {env_path}")

    with open(env_path, "r") as f:
        config = toml.load(f)

    # Flatten the config
    flat = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat[key] = value
        else:
            flat[section] = value

    return flat


def load_test_config(test_id: str) -> Dict:
    """Load test configuration from TOML file."""
    config_path = get_integration_dir() / "configs" / f"{test_id}.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Test config not found: {config_path}")

    with open(config_path, "r") as f:
        return toml.load(f)


def get_test_thresholds(test_id: str) -> TestThresholds:
    """Get pass/fail thresholds for a test."""
    try:
        config = load_test_config(test_id)
        thresholds = config.get("thresholds", {})
        return TestThresholds(
            min_id_rate=thresholds.get("min_id_rate", 0.30),
            min_rt_correlation=thresholds.get("min_rt_correlation", 0.90),
            min_im_correlation=thresholds.get("min_im_correlation", 0.90),
        )
    except Exception:
        return TestThresholds()


def find_simulation_outputs(test_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find simulation outputs in test directory.

    Args:
        test_dir: Test output directory.

    Returns:
        Tuple of (d_folder_path, database_path) or (None, None) if not found.
    """
    # Look for .d folder
    d_folders = list(test_dir.glob("*.d"))
    d_folder = d_folders[0] if d_folders else None

    # Look for synthetic_data.db
    db_paths = list(test_dir.glob("**/synthetic_data.db"))
    db_path = db_paths[0] if db_paths else None

    # Also check inside .d folder
    if d_folder and not db_path:
        db_in_d = d_folder / "synthetic_data.db"
        if db_in_d.exists():
            db_path = db_in_d

    return d_folder, db_path


def run_diann_analysis(
    d_folder: Path,
    fasta_path: str,
    output_dir: Path,
    env_config: Dict,
    test_id: str,
    is_dda: bool = False,
) -> Optional[Path]:
    """
    Run DiaNN analysis on simulated data.

    Args:
        d_folder: Path to the .d folder.
        fasta_path: Path to FASTA file.
        output_dir: Output directory for results.
        env_config: Environment configuration.
        test_id: Test identifier for logging.
        is_dda: Whether this is DDA data.

    Returns:
        Path to DiaNN report file, or None if failed.
    """
    logger.info(f"[{test_id}] Running DiaNN analysis...")

    try:
        from imspy.simulation.timsim.validate.diann_executor import DiannExecutor, DiannConfig

        # Configure DiaNN
        config = DiannConfig(
            fasta_path=fasta_path,
            library_free=True,
            use_predictor=True,
            qvalue=env_config.get("diann_qvalue", 0.01),
            min_pep_len=7,
            max_pep_len=30,
            missed_cleavages=2,
        )

        executor = DiannExecutor(
            executable_path=env_config.get("diann_path", "diann"),
            threads=env_config.get("diann_threads", 8),
            timeout_seconds=env_config.get("diann_timeout", 7200),
            config=config,
        )

        diann_output = output_dir / "diann"
        diann_output.mkdir(parents=True, exist_ok=True)

        success = executor.run(
            raw_files=[str(d_folder)],
            output_dir=str(diann_output),
        )

        if success:
            # Find the report file
            report_files = list(diann_output.glob("report*.tsv")) + list(diann_output.glob("report*.parquet"))
            if report_files:
                logger.info(f"[{test_id}] DiaNN completed: {report_files[0]}")
                return report_files[0]

        logger.error(f"[{test_id}] DiaNN failed or no report generated")
        return None

    except Exception as e:
        logger.error(f"[{test_id}] DiaNN error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_fragpipe_analysis(
    d_folder: Path,
    fasta_path: str,
    output_dir: Path,
    env_config: Dict,
    test_id: str,
    is_dda: bool = False,
) -> Optional[Path]:
    """
    Run FragPipe analysis on simulated data.

    Args:
        d_folder: Path to the .d folder.
        fasta_path: Path to FASTA file.
        output_dir: Output directory for results.
        env_config: Environment configuration.
        test_id: Test identifier for logging.
        is_dda: Whether this is DDA data.

    Returns:
        Path to FragPipe output directory, or None if failed.
    """
    logger.info(f"[{test_id}] Running FragPipe analysis...")

    try:
        from imspy.simulation.timsim.validate.fragpipe_executor import FragPipeExecutor, FragPipeConfig

        # Select appropriate workflow
        if is_dda:
            workflow = env_config.get("fragpipe_workflow_dda")
        else:
            workflow = env_config.get("fragpipe_workflow_dia")

        if not workflow:
            logger.error(f"[{test_id}] No FragPipe workflow configured")
            return None

        config = FragPipeConfig(
            workflow_path=workflow,
            fasta_path=fasta_path,
            tools_folder=env_config.get("fragpipe_tools", ""),
            python_path=env_config.get("fragpipe_python", ""),
        )

        executor = FragPipeExecutor(
            executable_path=env_config.get("fragpipe_path", "fragpipe"),
            threads=env_config.get("num_threads", 8),
            timeout_seconds=env_config.get("fragpipe_timeout", 14400),
            config=config,
        )

        fragpipe_output = output_dir / "fragpipe"
        fragpipe_output.mkdir(parents=True, exist_ok=True)

        success = executor.run(
            raw_files=[str(d_folder)],
            output_dir=str(fragpipe_output),
        )

        if success:
            logger.info(f"[{test_id}] FragPipe completed: {fragpipe_output}")
            return fragpipe_output

        logger.error(f"[{test_id}] FragPipe failed")
        return None

    except Exception as e:
        logger.error(f"[{test_id}] FragPipe error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_validation(
    database_path: Path,
    diann_report: Optional[Path],
    fragpipe_output: Optional[Path],
    output_dir: Path,
    test_id: str,
    thresholds: TestThresholds,
) -> Tuple[bool, Dict]:
    """
    Run validation comparison against ground truth.

    Args:
        database_path: Path to synthetic_data.db.
        diann_report: Path to DiaNN report file.
        fragpipe_output: Path to FragPipe output directory.
        output_dir: Output directory for validation results.
        test_id: Test identifier for logging.
        thresholds: Pass/fail thresholds.

    Returns:
        Tuple of (passed, metrics_dict).
    """
    logger.info(f"[{test_id}] Running validation...")

    try:
        from imspy.simulation.timsim.validate.tool_comparison import run_comparison

        validation_output = output_dir / "validation"
        validation_output.mkdir(parents=True, exist_ok=True)

        result = run_comparison(
            database_path=str(database_path),
            diann_report_path=str(diann_report) if diann_report else None,
            fragpipe_output_dir=str(fragpipe_output) if fragpipe_output else None,
            output_dir=str(validation_output),
            generate_plots=True,
        )

        # Check thresholds
        passed = True
        metrics = {
            "ground_truth_precursors": result.ground_truth_precursors,
            "ground_truth_peptides": result.ground_truth_peptides,
            "thresholds": {
                "min_id_rate": thresholds.min_id_rate,
                "min_rt_correlation": thresholds.min_rt_correlation,
                "min_im_correlation": thresholds.min_im_correlation,
            },
            "tool_results": {},
        }

        for tool_name, tool_result in result.tool_results.items():
            id_rate = tool_result.precision  # or use id_count / ground_truth

            # Get correlations
            rt_corr = result.rt_correlations.get(tool_name, 0.0)
            im_corr = result.im_correlations.get(tool_name, 0.0)

            tool_passed = True
            failures = []

            if id_rate < thresholds.min_id_rate:
                tool_passed = False
                failures.append(f"ID rate {id_rate:.2%} < {thresholds.min_id_rate:.2%}")

            if rt_corr < thresholds.min_rt_correlation:
                tool_passed = False
                failures.append(f"RT correlation {rt_corr:.3f} < {thresholds.min_rt_correlation:.3f}")

            if im_corr < thresholds.min_im_correlation:
                tool_passed = False
                failures.append(f"IM correlation {im_corr:.3f} < {thresholds.min_im_correlation:.3f}")

            metrics["tool_results"][tool_name] = {
                "id_count": tool_result.id_count,
                "precision": tool_result.precision,
                "rt_correlation": rt_corr,
                "im_correlation": im_corr,
                "passed": tool_passed,
                "failures": failures,
            }

            if not tool_passed:
                passed = False

        # Save metrics
        metrics_file = validation_output / "validation_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"[{test_id}] Validation {'PASSED' if passed else 'FAILED'}")
        return passed, metrics

    except Exception as e:
        logger.error(f"[{test_id}] Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}


def run_test_evaluation(
    test_id: str,
    env_config: Dict,
    tool: str = "both",
    skip_analysis: bool = False,
) -> Tuple[bool, Dict]:
    """
    Run complete evaluation for a single test.

    Args:
        test_id: Test identifier.
        env_config: Environment configuration.
        tool: Which tool(s) to run ("diann", "fragpipe", or "both").
        skip_analysis: Skip analysis, only run validation on existing outputs.

    Returns:
        Tuple of (passed, metrics).
    """
    logger.info(f"=" * 60)
    logger.info(f"Evaluating test: {test_id}")
    logger.info(f"=" * 60)

    # Load test config for metadata
    try:
        test_config = load_test_config(test_id)
        test_meta = test_config.get("test_metadata", {})
        acquisition_type = test_meta.get("acquisition_type", "DIA")
        sample_type = test_meta.get("sample_type", "hela")
        is_dda = acquisition_type.upper() == "DDA"
    except Exception:
        is_dda = "DDA" in test_id
        sample_type = "hela"

    # Get thresholds
    thresholds = get_test_thresholds(test_id)

    # Find test output directory
    output_base = env_config.get("output_base", ".")
    test_dir = Path(output_base) / test_id

    if not test_dir.exists():
        logger.error(f"[{test_id}] Test directory not found: {test_dir}")
        logger.error(f"[{test_id}] Run simulation first with timsim-integration-sim")
        return False, {"error": "Test directory not found"}

    # Find simulation outputs
    d_folder, db_path = find_simulation_outputs(test_dir)

    if not d_folder:
        logger.error(f"[{test_id}] No .d folder found in {test_dir}")
        return False, {"error": "No .d folder found"}

    if not db_path:
        logger.error(f"[{test_id}] No synthetic_data.db found in {test_dir}")
        return False, {"error": "No database found"}

    logger.info(f"[{test_id}] Found simulation: {d_folder}")
    logger.info(f"[{test_id}] Found database: {db_path}")

    # Determine FASTA path
    fasta_key = f"fasta_{sample_type}"
    fasta_path = env_config.get(fasta_key, env_config.get("fasta_hela", ""))

    if not fasta_path:
        logger.error(f"[{test_id}] No FASTA path configured for {sample_type}")
        return False, {"error": f"No FASTA for {sample_type}"}

    # Run analysis tools
    diann_report = None
    fragpipe_output = None

    if not skip_analysis:
        if tool in ["diann", "both"]:
            diann_report = run_diann_analysis(
                d_folder, fasta_path, test_dir, env_config, test_id, is_dda
            )

        if tool in ["fragpipe", "both"]:
            fragpipe_output = run_fragpipe_analysis(
                d_folder, fasta_path, test_dir, env_config, test_id, is_dda
            )
    else:
        # Look for existing analysis outputs
        existing_diann = list(test_dir.glob("diann/report*.tsv")) + list(test_dir.glob("diann/report*.parquet"))
        if existing_diann:
            diann_report = existing_diann[0]
            logger.info(f"[{test_id}] Using existing DiaNN output: {diann_report}")

        existing_fragpipe = test_dir / "fragpipe"
        if existing_fragpipe.exists():
            fragpipe_output = existing_fragpipe
            logger.info(f"[{test_id}] Using existing FragPipe output: {fragpipe_output}")

    # Run validation
    if not diann_report and not fragpipe_output:
        logger.error(f"[{test_id}] No analysis results to validate")
        return False, {"error": "No analysis results"}

    passed, metrics = run_validation(
        db_path, diann_report, fragpipe_output, test_dir, test_id, thresholds
    )

    # Write status file
    status_file = test_dir / ("EVAL_PASS" if passed else "EVAL_FAIL")
    status_file.touch()

    # Clean up old status files
    for old_status in ["EVAL_PASS", "EVAL_FAIL"]:
        old_file = test_dir / old_status
        if old_file.exists() and old_file != status_file:
            old_file.unlink()

    return passed, metrics


def main():
    """Entry point for timsim-integration-eval."""
    parser = argparse.ArgumentParser(
        description="Run timsim integration test evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate all tests with both tools
    timsim-integration-eval --env env.toml --all

    # Evaluate specific test with DiaNN only
    timsim-integration-eval --env env.toml --test IT-DIA-HELA --tool diann

    # Evaluate with both tools
    timsim-integration-eval --env env.toml --test IT-DIA-HELA --tool both

    # Skip analysis, only run validation on existing outputs
    timsim-integration-eval --env env.toml --all --skip-analysis
        """,
    )

    parser.add_argument(
        "--env",
        required=True,
        help="Path to environment config file (env.toml)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all available tests",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Evaluate a single test by ID",
    )
    parser.add_argument(
        "--tests",
        type=str,
        help="Evaluate multiple tests (comma-separated IDs)",
    )
    parser.add_argument(
        "--tool",
        type=str,
        choices=["diann", "fragpipe", "both"],
        default="both",
        help="Which analysis tool(s) to run (default: both)",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip running analysis tools, only validate existing outputs",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests and exit",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available integration tests:")
        for test_id in AVAILABLE_TESTS:
            try:
                config = load_test_config(test_id)
                meta = config.get("test_metadata", {})
                desc = meta.get("description", "No description")
                tools = ", ".join(meta.get("analysis_tools", []))
                print(f"  {test_id}: {desc} [{tools}]")
            except FileNotFoundError:
                print(f"  {test_id}: (config not found)")
        return

    # Determine which tests to run
    tests_to_run: List[str] = []

    if args.all:
        tests_to_run = AVAILABLE_TESTS.copy()
    elif args.test:
        if args.test not in AVAILABLE_TESTS:
            logger.error(f"Unknown test: {args.test}")
            logger.info(f"Available tests: {', '.join(AVAILABLE_TESTS)}")
            sys.exit(1)
        tests_to_run = [args.test]
    elif args.tests:
        tests_to_run = [t.strip() for t in args.tests.split(",")]
        unknown = [t for t in tests_to_run if t not in AVAILABLE_TESTS]
        if unknown:
            logger.error(f"Unknown tests: {', '.join(unknown)}")
            logger.info(f"Available tests: {', '.join(AVAILABLE_TESTS)}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # Load environment config
    try:
        env_config = load_env_config(args.env)
        logger.info(f"Loaded environment config from: {args.env}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load environment config: {e}")
        sys.exit(1)

    # Run evaluations
    logger.info(f"Evaluating {len(tests_to_run)} test(s): {', '.join(tests_to_run)}")
    logger.info(f"Tool(s): {args.tool}")
    start_time = datetime.now()

    results = {}
    for test_id in tests_to_run:
        passed, metrics = run_test_evaluation(
            test_id, env_config, tool=args.tool, skip_analysis=args.skip_analysis
        )
        results[test_id] = {"passed": passed, "metrics": metrics}

    # Summary
    elapsed = datetime.now() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    passed_count = sum(1 for r in results.values() if r["passed"])
    failed_count = len(results) - passed_count

    for test_id, result in results.items():
        status = "PASS" if result["passed"] else "FAIL"
        logger.info(f"  {test_id}: {status}")

        # Print tool-level details
        tool_results = result.get("metrics", {}).get("tool_results", {})
        for tool_name, tool_metrics in tool_results.items():
            tool_status = "PASS" if tool_metrics.get("passed", False) else "FAIL"
            id_count = tool_metrics.get("id_count", 0)
            precision = tool_metrics.get("precision", 0)
            logger.info(f"    {tool_name}: {tool_status} (IDs: {id_count}, precision: {precision:.2%})")
            for failure in tool_metrics.get("failures", []):
                logger.info(f"      - {failure}")

    logger.info("")
    logger.info(f"Total: {passed_count} passed, {failed_count} failed")
    logger.info(f"Elapsed time: {elapsed}")

    # Write summary JSON
    output_base = env_config.get("output_base", ".")
    summary_file = Path(output_base) / "evaluation_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed.total_seconds(),
        "tool": args.tool,
        "passed": passed_count,
        "failed": failed_count,
        "results": results,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary written to: {summary_file}")

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
