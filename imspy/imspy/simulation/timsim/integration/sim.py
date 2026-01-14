#!/usr/bin/env python3
"""
timsim-integration-sim: Simulation runner for integration tests.

This script generates synthetic datasets for all configured integration tests.
It reads test configurations from the configs/ directory and resolves
machine-specific paths from env.toml.

Usage:
    timsim-integration-sim --env env.toml --all
    timsim-integration-sim --env env.toml --test IT-DIA-HELA
    timsim-integration-sim --env env.toml --tests IT-DIA-HELA,IT-DIA-HYE
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import toml

from imspy.vis.frame_rendering import generate_preview_gif

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


def get_integration_dir() -> Path:
    """Get the integration test directory."""
    return Path(__file__).parent


def load_env_config(env_path: str) -> Dict:
    """
    Load environment configuration from TOML file.

    Args:
        env_path: Path to env.toml file.

    Returns:
        Flattened dictionary of environment settings.

    Raises:
        FileNotFoundError: If env.toml doesn't exist.
        ValueError: If required paths are missing.
    """
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

    # Validate required paths
    required = ["output_base", "reference_dia", "fasta_hela"]
    missing = [k for k in required if not flat.get(k)]
    if missing:
        raise ValueError(f"Missing required env settings: {', '.join(missing)}")

    return flat


def load_test_config(test_id: str) -> Dict:
    """
    Load test configuration from TOML file.

    Args:
        test_id: Test identifier (e.g., "IT-DIA-HELA").

    Returns:
        Test configuration dictionary.

    Raises:
        FileNotFoundError: If test config doesn't exist.
    """
    config_path = get_integration_dir() / "configs" / f"{test_id}.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Test config not found: {config_path}")

    with open(config_path, "r") as f:
        return toml.load(f)


def resolve_placeholder_string(value: str, env_config: Dict) -> str:
    """
    Resolve all ${key} placeholders in a string.

    Args:
        value: String potentially containing ${key} placeholders.
        env_config: Environment configuration with actual values.

    Returns:
        String with all placeholders resolved.
    """
    import re

    def replacer(match):
        key = match.group(1)
        if key in env_config:
            return str(env_config[key])
        else:
            logger.warning(f"Unresolved placeholder: ${{{key}}}")
            return match.group(0)  # Keep original if not found

    # Match ${key} patterns
    return re.sub(r'\$\{([^}]+)\}', replacer, value)


def resolve_paths(test_config: Dict, env_config: Dict) -> Dict:
    """
    Resolve path placeholders in test config using env config.

    Replaces ${key} patterns with values from env_config.
    Supports string interpolation like "${output_base}/subdir".

    Args:
        test_config: Test configuration with placeholders.
        env_config: Environment configuration with actual paths.

    Returns:
        Test configuration with resolved paths.
    """
    result = {}

    for section, values in test_config.items():
        if isinstance(values, dict):
            result[section] = {}
            for key, value in values.items():
                if isinstance(value, str) and "${" in value:
                    result[section][key] = resolve_placeholder_string(value, env_config)
                else:
                    result[section][key] = value
        else:
            result[section] = values

    return result


def merge_configs(test_config: Dict, env_config: Dict) -> Dict:
    """
    Merge test config with environment config.

    Environment settings (like performance, gpu) override test defaults.

    Args:
        test_config: Resolved test configuration.
        env_config: Environment configuration.

    Returns:
        Merged configuration ready for simulation.
    """
    merged = {}

    # Copy test config sections (skip metadata and thresholds)
    skip_sections = ["test_metadata", "thresholds", "paths"]
    for section, values in test_config.items():
        if section not in skip_sections:
            merged[section] = values.copy() if isinstance(values, dict) else values

    # Handle paths section specially
    if "paths" in test_config:
        if "main_settings" not in merged:
            merged["main_settings"] = {}
        merged["main_settings"]["save_path"] = test_config["paths"].get("save_path", "")
        merged["main_settings"]["reference_path"] = test_config["paths"].get("reference_path", "")
        merged["main_settings"]["fasta_path"] = test_config["paths"].get("fasta_path", "")

    # Override performance settings from env
    if "performance_settings" not in merged:
        merged["performance_settings"] = {}

    perf_overrides = ["num_threads", "batch_size", "frame_batch_size", "use_gpu", "gpu_memory_limit_gb"]
    for key in perf_overrides:
        if key in env_config:
            merged["performance_settings"][key] = env_config[key]

    # Override Bruker SDK setting
    if "use_bruker_sdk" in env_config and "main_settings" in merged:
        merged["main_settings"]["use_bruker_sdk"] = env_config["use_bruker_sdk"]

    return merged


def write_temp_config(config: Dict, test_id: str, output_dir: Path) -> Path:
    """
    Write merged configuration to a temporary file.

    Args:
        config: Merged configuration dictionary.
        test_id: Test identifier.
        output_dir: Directory to write config file.

    Returns:
        Path to the temporary config file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"{test_id}_config.toml"

    with open(config_path, "w") as f:
        toml.dump(config, f)

    return config_path


def run_simulation(config_path: Path, test_id: str) -> bool:
    """
    Run timsim simulation with the given config.

    Uses subprocess to run timsim CLI, which ensures proper isolation
    and mimics actual user workflow.

    Args:
        config_path: Path to the simulation config file.
        test_id: Test identifier for logging.

    Returns:
        True if simulation succeeded, False otherwise.
    """
    logger.info(f"[{test_id}] Starting simulation...")
    logger.info(f"[{test_id}] Config: {config_path}")

    try:
        # Run timsim via subprocess using the CLI entry point
        cmd = [sys.executable, "-m", "imspy.simulation.timsim.simulator", str(config_path)]
        logger.info(f"[{test_id}] Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=False,  # Let output stream to console
            text=True,
        )

        if result.returncode == 0:
            logger.info(f"[{test_id}] Simulation completed successfully")
            return True
        else:
            logger.error(f"[{test_id}] Simulation failed with exit code {result.returncode}")
            return False

    except Exception as e:
        logger.error(f"[{test_id}] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_simulation_preview(
    test_output_dir: Path,
    test_id: str,
    acquisition_type: str,
    use_bruker_sdk: bool = False,
    max_frames: int = 50,
) -> Optional[str]:
    """
    Generate a preview GIF of the simulated data.

    Args:
        test_output_dir: Directory containing simulation output.
        test_id: Test identifier (used as experiment name).
        acquisition_type: 'DIA' or 'DDA'.
        use_bruker_sdk: Whether to use Bruker SDK for reading.
        max_frames: Maximum frames to include in GIF.

    Returns:
        Path to generated GIF, or None if generation failed.
    """
    # Find the .d folder
    d_folder = test_output_dir / test_id / f"{test_id}.d"
    if not d_folder.exists():
        logger.warning(f"Cannot find .d folder at {d_folder}")
        return None

    gif_path = test_output_dir / test_id / "simulation_preview.gif"

    try:
        mode = 'dia' if acquisition_type.upper() == 'DIA' else 'dda'
        logger.info(f"Generating preview GIF ({mode} mode, {max_frames} frames)...")

        generate_preview_gif(
            data_path=str(d_folder),
            output_path=str(gif_path),
            mode=mode,
            max_frames=max_frames,
            fps=5,
            dpi=60,
            annotate=True,
            use_bruker_sdk=use_bruker_sdk,
            show_progress=True,
        )

        logger.info(f"Preview GIF saved to: {gif_path}")
        return str(gif_path)

    except Exception as e:
        logger.warning(f"Failed to generate preview GIF: {e}")
        return None


def run_test(test_id: str, env_config: Dict, dry_run: bool = False) -> bool:
    """
    Run a single integration test simulation.

    Args:
        test_id: Test identifier.
        env_config: Environment configuration.
        dry_run: If True, only prepare config without running.

    Returns:
        True if successful, False otherwise.
    """
    logger.info(f"=" * 60)
    logger.info(f"Running test: {test_id}")
    logger.info(f"=" * 60)

    try:
        # Load test config
        test_config = load_test_config(test_id)
        logger.info(f"[{test_id}] Loaded test config")

        # Resolve path placeholders
        resolved_config = resolve_paths(test_config, env_config)
        logger.info(f"[{test_id}] Resolved paths")

        # Merge with env config
        merged_config = merge_configs(resolved_config, env_config)
        logger.info(f"[{test_id}] Merged configurations")

        # Create output directory
        output_base = env_config.get("output_base", ".")
        test_output_dir = Path(output_base) / test_id
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # Write temporary config
        config_path = write_temp_config(merged_config, test_id, test_output_dir)
        logger.info(f"[{test_id}] Wrote config to: {config_path}")

        if dry_run:
            logger.info(f"[{test_id}] Dry run - skipping simulation")
            return True

        # Run simulation
        success = run_simulation(config_path, test_id)

        # Generate preview GIF if simulation succeeded
        if success:
            acquisition_type = test_config.get("test_metadata", {}).get("acquisition_type", "DIA")
            use_bruker_sdk = env_config.get("use_bruker_sdk", False)
            generate_simulation_preview(
                test_output_dir=test_output_dir,
                test_id=test_id,
                acquisition_type=acquisition_type,
                use_bruker_sdk=use_bruker_sdk,
            )

        # Clean up old status files and write new one
        for old_status in ["SIM_SUCCESS", "SIM_FAILED"]:
            old_file = test_output_dir / old_status
            if old_file.exists():
                old_file.unlink()

        status_file = test_output_dir / ("SIM_SUCCESS" if success else "SIM_FAILED")
        status_file.touch()

        return success

    except Exception as e:
        logger.error(f"[{test_id}] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Entry point for timsim-integration-sim."""
    parser = argparse.ArgumentParser(
        description="Run timsim integration test simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tests
    timsim-integration-sim --env env.toml --all

    # Run specific test
    timsim-integration-sim --env env.toml --test IT-DIA-HELA

    # Run multiple tests
    timsim-integration-sim --env env.toml --tests IT-DIA-HELA,IT-DIA-HYE

    # Dry run (prepare configs only)
    timsim-integration-sim --env env.toml --all --dry-run
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
        help="Run all available tests",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Run a single test by ID",
    )
    parser.add_argument(
        "--tests",
        type=str,
        help="Run multiple tests (comma-separated IDs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare configs without running simulations",
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
                print(f"  {test_id}: {desc}")
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

    # Run tests
    logger.info(f"Running {len(tests_to_run)} test(s): {', '.join(tests_to_run)}")
    start_time = datetime.now()

    results = {}
    for test_id in tests_to_run:
        success = run_test(test_id, env_config, dry_run=args.dry_run)
        results[test_id] = success

    # Summary
    elapsed = datetime.now() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed

    for test_id, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"  {test_id}: {status}")

    logger.info("")
    logger.info(f"Total: {passed} passed, {failed} failed")
    logger.info(f"Elapsed time: {elapsed}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
