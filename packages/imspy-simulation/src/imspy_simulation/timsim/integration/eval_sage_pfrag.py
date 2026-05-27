#!/usr/bin/env python3
"""
Dedicated evaluation script for Sage partial fragmentation tests.

Runs FragPipe + Sage (no DiaNN) on each test independently (no cross-sample MBR),
then produces a cross-condition comparison of sensitivity to partial fragmentation.

Usage:
    python -m imspy_simulation.timsim.integration.eval_sage_pfrag --env env.toml
    python -m imspy_simulation.timsim.integration.eval_sage_pfrag --env env.toml --skip-analysis
    python -m imspy_simulation.timsim.integration.eval_sage_pfrag --env env.toml --test IT-DDA-SAGE-PFRAG-LOW
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from imspy_simulation.timsim.integration.eval import (
    TestThresholds,
    find_simulation_outputs,
    get_test_thresholds,
    get_tool_versions,
    load_env_config,
    load_test_config,
    resolve_fasta_path,
    run_fragpipe_analysis,
    run_sage_analysis,
    run_validation,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Partial fragmentation test suite with metadata
PFRAG_TESTS = [
    {"test_id": "IT-DDA-SAGE-PFRAG-BASE", "label": "No partial frag", "survival_max": 0.0},
    {"test_id": "IT-DDA-SAGE-PFRAG-LOW",  "label": "Low (0-10%)",     "survival_max": 0.1},
    {"test_id": "IT-DDA-SAGE-PFRAG-MED",  "label": "Med (0-30%)",     "survival_max": 0.3},
    {"test_id": "IT-DDA-SAGE-PFRAG-HIGH", "label": "High (0-50%)",    "survival_max": 0.5},
]

PFRAG_TEST_IDS = [t["test_id"] for t in PFRAG_TESTS]


def run_pfrag_test_evaluation(
    test_id: str,
    env_config: Dict,
    skip_analysis: bool = False,
) -> Tuple[bool, Dict]:
    """
    Run evaluation for a single partial fragmentation test.

    Runs FragPipe + Sage independently (no additional_d_folders / MBR).

    Args:
        test_id: Test identifier (e.g., IT-DDA-SAGE-PFRAG-BASE).
        env_config: Environment configuration.
        skip_analysis: Skip analysis, only run validation on existing outputs.

    Returns:
        Tuple of (passed, metrics).
    """
    logger.info("=" * 60)
    logger.info(f"Evaluating test: {test_id}")
    logger.info("=" * 60)

    # Load test config for metadata
    try:
        test_config = load_test_config(test_id)
        test_meta = test_config.get("test_metadata", {})
        description = test_meta.get("description", "")
        sample_type = test_meta.get("sample_type", "hela")
    except Exception:
        description = ""
        sample_type = "hela"

    # Get thresholds
    thresholds = get_test_thresholds(test_id)

    # Find test output directory
    output_base = env_config.get("output_base", ".")
    test_dir = Path(output_base) / test_id

    if not test_dir.exists():
        logger.error(f"[{test_id}] Test directory not found: {test_dir}")
        logger.error(f"[{test_id}] Run simulation first")
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

    # Resolve FASTA paths
    fasta_path = resolve_fasta_path(env_config, f"fasta_{sample_type}")
    fasta_path_decoys = resolve_fasta_path(env_config, f"fasta_{sample_type}_decoys")

    if not fasta_path:
        logger.error(f"[{test_id}] No FASTA path configured for {sample_type}")
        return False, {"error": f"No FASTA for {sample_type}"}

    # Run analysis tools (FragPipe + Sage only, no additional_d_folders)
    fragpipe_output = None
    sage_results = None

    if not skip_analysis:
        # Sage (run first)
        sage_fasta = fasta_path_decoys if fasta_path_decoys else fasta_path
        if sage_fasta != fasta_path:
            logger.info(f"[{test_id}] Using decoy FASTA for Sage: {sage_fasta}")
        sage_results = run_sage_analysis(
            d_folder, sage_fasta, test_dir, env_config, test_id,
        )

        # FragPipe (run second)
        fragpipe_fasta = fasta_path_decoys if fasta_path_decoys else fasta_path
        if fragpipe_fasta != fasta_path:
            logger.info(f"[{test_id}] Using decoy FASTA for FragPipe: {fragpipe_fasta}")
        fragpipe_output = run_fragpipe_analysis(
            d_folder, fragpipe_fasta, test_dir, env_config, test_id, is_dda=True,
        )
    else:
        # Look for existing outputs
        existing_fragpipe = test_dir / "fragpipe"
        if existing_fragpipe.exists():
            fragpipe_output = existing_fragpipe
            logger.info(f"[{test_id}] Using existing FragPipe output: {fragpipe_output}")

        existing_sage = test_dir / "sage" / "results.sage.tsv"
        if existing_sage.exists():
            sage_results = existing_sage
            logger.info(f"[{test_id}] Using existing Sage output: {sage_results}")

    if not fragpipe_output and not sage_results:
        logger.error(f"[{test_id}] No analysis results to validate")
        return False, {"error": "No analysis results"}

    # Get tool versions
    tool_versions = get_tool_versions(env_config)

    # Build test metadata for HTML report
    test_metadata = {
        "test_id": test_id,
        "acquisition_type": "DDA",
        "sample_type": sample_type,
        "description": description,
    }

    passed, metrics = run_validation(
        db_path, None, fragpipe_output, sage_results, test_dir, test_id, thresholds,
        tool_versions=tool_versions,
        test_metadata=test_metadata,
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


def print_comparison_table(all_results: Dict[str, Dict]) -> None:
    """Print a cross-condition comparison table to the console."""
    # Build lookup from test_id to pfrag metadata
    pfrag_lookup = {t["test_id"]: t for t in PFRAG_TESTS}

    print()
    print("SAGE PARTIAL FRAGMENTATION COMPARISON")
    print("=" * 95)
    header = (
        f"{'Condition':<20s} | {'FragPipe ID Rate':>16s} | {'Sage ID Rate':>12s} "
        f"| {'FragPipe IDs':>12s} | {'Sage IDs':>10s}"
    )
    print(header)
    print("-" * 95)

    for test_info in PFRAG_TESTS:
        test_id = test_info["test_id"]
        label = test_info["label"]

        result = all_results.get(test_id)
        if not result:
            print(f"{label:<20s} |     (no results) |             |              |")
            continue

        metrics = result.get("metrics", {})
        tool_results = metrics.get("tool_results", {})

        fp = tool_results.get("FragPipe", {})
        sage = tool_results.get("Sage", {})

        fp_rate = f"{fp.get('id_rate', 0):.1%}" if fp else "N/A"
        sage_rate = f"{sage.get('id_rate', 0):.1%}" if sage else "N/A"
        fp_ids = f"{fp.get('id_count', 0):,}" if fp else "N/A"
        sage_ids = f"{sage.get('id_count', 0):,}" if sage else "N/A"

        print(
            f"{label:<20s} | {fp_rate:>16s} | {sage_rate:>12s} "
            f"| {fp_ids:>12s} | {sage_ids:>10s}"
        )

    print("=" * 95)
    print()


def save_comparison_json(all_results: Dict[str, Dict], output_base: str) -> str:
    """Save cross-condition comparison as JSON."""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "description": "Sage partial fragmentation sensitivity comparison (FragPipe vs Sage)",
        "conditions": [],
    }

    for test_info in PFRAG_TESTS:
        test_id = test_info["test_id"]
        result = all_results.get(test_id)

        condition = {
            "test_id": test_id,
            "label": test_info["label"],
            "survival_max": test_info["survival_max"],
            "passed": result["passed"] if result else None,
            "tools": {},
        }

        if result:
            tool_results = result.get("metrics", {}).get("tool_results", {})
            for tool_name, tool_metrics in tool_results.items():
                condition["tools"][tool_name] = {
                    "id_count": tool_metrics.get("id_count", 0),
                    "id_rate": tool_metrics.get("id_rate", 0),
                    "precision": tool_metrics.get("precision", 0),
                    "rt_correlation": tool_metrics.get("rt_correlation", 0),
                    "im_correlation": tool_metrics.get("im_correlation", 0),
                    "passed": tool_metrics.get("passed", False),
                }

        comparison["conditions"].append(condition)

    output_path = Path(output_base) / "sage_pfrag_comparison.json"
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Comparison saved to: {output_path}")
    return str(output_path)


def main():
    """Entry point for eval_sage_pfrag."""
    parser = argparse.ArgumentParser(
        description="Evaluate Sage partial fragmentation test suite (FragPipe + Sage)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate all 4 partial fragmentation conditions
    python -m imspy_simulation.timsim.integration.eval_sage_pfrag --env env.toml

    # Skip analysis, only re-run validation
    python -m imspy_simulation.timsim.integration.eval_sage_pfrag --env env.toml --skip-analysis

    # Evaluate a single condition (for debugging)
    python -m imspy_simulation.timsim.integration.eval_sage_pfrag --env env.toml --test IT-DDA-SAGE-PFRAG-LOW
        """,
    )

    parser.add_argument(
        "--env",
        required=True,
        help="Path to environment config file (env.toml)",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip running analysis tools, only validate existing outputs",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Evaluate a single test by ID (still prints comparison if prior results exist)",
    )

    args = parser.parse_args()

    # Load environment config
    try:
        env_config = load_env_config(args.env)
        logger.info(f"Loaded environment config from: {args.env}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load environment config: {e}")
        sys.exit(1)

    # Determine which tests to run
    if args.test:
        if args.test not in PFRAG_TEST_IDS:
            logger.error(f"Unknown test: {args.test}")
            logger.info(f"Available tests: {', '.join(PFRAG_TEST_IDS)}")
            sys.exit(1)
        tests_to_run = [args.test]
    else:
        tests_to_run = PFRAG_TEST_IDS

    # Run evaluations
    logger.info(f"Evaluating {len(tests_to_run)} partial fragmentation test(s)")
    start_time = datetime.now()

    results = {}
    for test_id in tests_to_run:
        passed, metrics = run_pfrag_test_evaluation(
            test_id, env_config, skip_analysis=args.skip_analysis,
        )
        results[test_id] = {"passed": passed, "metrics": metrics}

    # If running a single test, try to load prior results for the comparison table
    all_results = dict(results)
    if args.test:
        output_base = env_config.get("output_base", ".")
        for test_id in PFRAG_TEST_IDS:
            if test_id in all_results:
                continue
            # Try to load from existing validation_metrics.json
            metrics_file = Path(output_base) / test_id / "validation" / "validation_metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        prior_metrics = json.load(f)
                    # Determine pass/fail from status files
                    test_dir = Path(output_base) / test_id
                    prior_passed = (test_dir / "EVAL_PASS").exists()
                    all_results[test_id] = {"passed": prior_passed, "metrics": prior_metrics}
                    logger.info(f"Loaded prior results for {test_id}")
                except Exception:
                    pass

    # Print per-test summary
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

        tool_results = result.get("metrics", {}).get("tool_results", {})
        for tool_name, tool_metrics in tool_results.items():
            tool_status = "PASS" if tool_metrics.get("passed", False) else "FAIL"
            id_count = tool_metrics.get("id_count", 0)
            id_rate = tool_metrics.get("id_rate", 0)
            logger.info(f"    {tool_name}: {tool_status} (IDs: {id_count}, ID rate: {id_rate:.2%})")
            for failure in tool_metrics.get("failures", []):
                logger.info(f"      - {failure}")

    logger.info("")
    logger.info(f"Total: {passed_count} passed, {failed_count} failed")
    logger.info(f"Elapsed time: {elapsed}")

    # Print cross-condition comparison table
    print_comparison_table(all_results)

    # Save comparison JSON
    output_base = env_config.get("output_base", ".")
    save_comparison_json(all_results, output_base)

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
