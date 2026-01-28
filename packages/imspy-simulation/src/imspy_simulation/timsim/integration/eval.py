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
    max_species_ratio_error: float = 0.20  # For HYE experiments
    min_ptm_site_accuracy: float = 0.80  # For phosphoproteomics experiments


def get_tool_versions(env_config: Dict) -> Dict[str, str]:
    """
    Get version strings for analysis tools from environment config paths.

    Args:
        env_config: Environment configuration dictionary.

    Returns:
        Dictionary mapping tool names to version strings.
    """
    import re
    versions = {}

    # Get DiaNN version
    diann_path = env_config.get("diann_path", "")
    if diann_path:
        try:
            from imspy_simulation.timsim.validate.diann_executor import DiannExecutor
            executor = DiannExecutor(executable_path=diann_path)
            versions["DIA-NN"] = executor.get_version()
        except Exception:
            # Fallback: extract from path
            match = re.search(r'diann[_-]?(\d+\.\d+(?:\.\d+)?)', diann_path)
            if match:
                versions["DIA-NN"] = match.group(1)

    # Get FragPipe version
    fragpipe_path = env_config.get("fragpipe_path", "")
    if fragpipe_path:
        try:
            from imspy_simulation.timsim.validate.fragpipe_executor import FragPipeExecutor
            executor = FragPipeExecutor(executable_path=fragpipe_path)
            versions["FragPipe"] = executor.get_version()
        except Exception:
            # Fallback: extract from path
            match = re.search(r'fragpipe[_-]?(\d+\.\d+(?:\.\d+)?)', fragpipe_path)
            if match:
                versions["FragPipe"] = match.group(1)

    # Get Sage version
    sage_path = env_config.get("sage_path", "")
    if sage_path:
        try:
            from imspy_simulation.timsim.validate.sage_executor import get_sage_version
            versions["Sage"] = get_sage_version(sage_path)
        except Exception:
            # Fallback: extract from path
            match = re.search(r'sage[_-]?(\d+\.\d+(?:\.\d+)?(?:-[a-z]+\.\d+)?)', sage_path)
            if match:
                versions["Sage"] = match.group(1)

    return versions


def get_integration_dir() -> Path:
    """Get the integration test directory."""
    return Path(__file__).parent


def get_bundled_data_dir() -> Path:
    """Get the bundled data directory containing FASTAs and configs."""
    return get_integration_dir() / "data"


def get_bundled_data_paths() -> Dict[str, str]:
    """
    Get paths to bundled data files (FASTAs, configs).

    Returns:
        Dictionary with keys for each data type and their absolute paths.
    """
    integration_dir = get_integration_dir()
    data_dir = get_bundled_data_dir()
    fasta_dir = data_dir / "fasta"
    hye_dir = fasta_dir / "hye"
    configs_dir = integration_dir / "configs"

    return {
        # HeLa proteome (small test set)
        "fasta_hela": str(fasta_dir / "hela-small.fasta"),
        "fasta_hela_decoys": str(fasta_dir / "hela-small-decoys.fasta"),
        # HLA (uses HeLa for now)
        "fasta_hla": str(fasta_dir / "hela-small.fasta"),
        "fasta_hla_decoys": str(fasta_dir / "hela-small-decoys.fasta"),
        # Phospho (uses HeLa)
        "fasta_phospho": str(fasta_dir / "hela-small.fasta"),
        "fasta_phospho_decoys": str(fasta_dir / "hela-small-decoys.fasta"),
        # HYE mixed proteome
        "fasta_hye_dir": str(hye_dir),
        "fasta_hye": str(hye_dir / "HYE_small.fasta"),
        "fasta_hye_decoys": str(hye_dir / "HYE_small_decoys.fasta"),
        # Config files
        "dilution_factors_hye": str(configs_dir / "dilution_factors_hye.csv"),
    }


# Backwards compatibility alias
get_bundled_fasta_paths = get_bundled_data_paths


def resolve_fasta_path(env_config: Dict, key: str) -> str:
    """
    Resolve a FASTA path from env_config, falling back to bundled data.

    Args:
        env_config: Environment configuration dictionary.
        key: The data key to resolve (e.g., "fasta_hela", "fasta_hye_dir").

    Returns:
        Absolute path to the FASTA file or directory.
    """
    # Get configured path
    path = env_config.get(key, "")

    # If empty or not specified, use bundled data
    if not path:
        bundled = get_bundled_data_paths()
        path = bundled.get(key, "")

    return path


def create_results_bundle(
    output_base: str,
    test_ids: List[str],
    timestamp: datetime,
) -> Optional[str]:
    """
    Create a zip bundle of all validation results.

    Includes:
    - index.html (dashboard)
    - evaluation_summary.json
    - Per-test validation folders (HTML reports, plots, JSON reports)

    Args:
        output_base: Base output directory containing test results.
        test_ids: List of test IDs that were evaluated.
        timestamp: Timestamp of the evaluation run.

    Returns:
        Path to the created zip file, or None if creation failed.
    """
    import zipfile

    output_path = Path(output_base)
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    bundle_name = f"timsim_validation_{timestamp_str}"
    zip_name = f"{bundle_name}.zip"
    zip_path = output_path / zip_name

    # Files to include at root level
    root_files = [
        "index.html",
        "evaluation_summary.json",
    ]

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add root files (inside the bundle folder)
            for filename in root_files:
                file_path = output_path / filename
                if file_path.exists():
                    # Add with bundle folder prefix so unzip creates a folder
                    arcname = f"{bundle_name}/{filename}"
                    zf.write(file_path, arcname)
                    logger.debug(f"Added to bundle: {arcname}")

            # Add validation results for each test
            for test_id in test_ids:
                test_dir = output_path / test_id
                if not test_dir.exists():
                    continue

                # Add validation folder contents
                validation_dir = test_dir / "validation"
                if validation_dir.exists():
                    for item in validation_dir.rglob("*"):
                        if item.is_file():
                            # Only include relevant files (HTML, plots, reports)
                            if item.suffix in [".html", ".png", ".txt", ".json"]:
                                # Add with bundle folder prefix
                                rel_path = item.relative_to(output_path)
                                arcname = f"{bundle_name}/{rel_path}"
                                zf.write(item, arcname)
                                logger.debug(f"Added to bundle: {arcname}")

        # Get bundle size
        bundle_size = zip_path.stat().st_size
        size_mb = bundle_size / (1024 * 1024)
        logger.info(f"Created results bundle: {zip_path} ({size_mb:.1f} MB)")

        return str(zip_path)

    except Exception as e:
        logger.warning(f"Failed to create results bundle: {e}")
        # Clean up partial zip if it exists
        if zip_path.exists():
            zip_path.unlink()
        return None


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
            max_species_ratio_error=thresholds.get("max_species_ratio_error", 0.20),
            min_ptm_site_accuracy=thresholds.get("min_ptm_site_accuracy", 0.80),
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
    # Look for .d folder recursively
    d_folders = list(test_dir.glob("**/*.d"))
    d_folder = d_folders[0] if d_folders else None

    # Look for synthetic_data.db recursively
    db_paths = list(test_dir.glob("**/synthetic_data.db"))
    db_path = db_paths[0] if db_paths else None

    return d_folder, db_path


# Paired tests that require A/B conditions to be analyzed together
PAIRED_TESTS = {
    "IT-DIA-HYE": {
        "condition_a": "IT-DIA-HYE-A",
        "condition_b": "IT-DIA-HYE-B",
        "expected_fold_changes": {
            "HUMAN": 1.0,      # 65%/65% = 1x
            "YEAST": 0.667,    # 20%/30% = 0.67x
            "ECOLI": 3.0,      # 15%/5% = 3x
        },
    },
    "IT-DIA-PHOS": {
        "condition_a": "IT-DIA-PHOS-A",
        "condition_b": "IT-DIA-PHOS-B",
        "expected_fold_changes": None,  # Phospho doesn't have fold-change validation
    },
}


def is_paired_test(test_id: str) -> bool:
    """Check if a test requires paired A/B conditions."""
    return test_id in PAIRED_TESTS


def find_paired_simulation_outputs(
    output_base: Path, test_id: str
) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """
    Find simulation outputs for paired A/B tests.

    Args:
        output_base: Base output directory.
        test_id: Test identifier (e.g., IT-DIA-HYE).

    Returns:
        Tuple of (d_folder_a, db_path_a, d_folder_b, db_path_b).
    """
    paired_config = PAIRED_TESTS.get(test_id)
    if not paired_config:
        return None, None, None, None

    # Find condition A outputs
    test_dir_a = output_base / paired_config["condition_a"]
    d_folder_a, db_path_a = find_simulation_outputs(test_dir_a)

    # Find condition B outputs
    test_dir_b = output_base / paired_config["condition_b"]
    d_folder_b, db_path_b = find_simulation_outputs(test_dir_b)

    return d_folder_a, db_path_a, d_folder_b, db_path_b


def run_diann_analysis(
    d_folder: Path,
    fasta_path: str,
    output_dir: Path,
    env_config: Dict,
    test_id: str,
    is_dda: bool = False,
    is_phospho: bool = False,
    additional_d_folders: Optional[List[Path]] = None,
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
        is_phospho: Whether this is phosphoproteomics data.
        additional_d_folders: Additional .d folders for multi-sample analysis
            (e.g., condition B for A/B fold-change experiments).

    Returns:
        Path to DiaNN report file, or None if failed.
    """
    if additional_d_folders:
        logger.info(f"[{test_id}] Running DiaNN analysis on {1 + len(additional_d_folders)} samples...")
    else:
        logger.info(f"[{test_id}] Running DiaNN analysis...")

    try:
        from imspy_simulation.timsim.validate.diann_executor import DiannExecutor, DiannConfig

        # Configure DiaNN
        config = DiannConfig(
            library_free=True,
            use_predictor=True,
            dda_mode=is_dda,  # Enable DDA mode for DDA data
            qvalue=env_config.get("diann_qvalue", 0.01),
            min_pep_len=7,
            max_pep_len=30,
            missed_cleavages=2,
            # Add phosphorylation as variable mod for phospho experiments
            var_mod="UniMod:21,79.966331,STY" if is_phospho else None,
            var_mods=3 if is_phospho else 2,  # Allow more var mods for phospho
        )

        executor = DiannExecutor(
            executable_path=env_config.get("diann_path", "diann"),
            threads=env_config.get("diann_threads", 8),
            timeout_seconds=env_config.get("diann_timeout", 7200),
            config=config,
        )

        diann_output = output_dir / "diann"
        diann_output.mkdir(parents=True, exist_ok=True)

        # Convert additional paths to strings
        additional_paths = None
        if additional_d_folders:
            additional_paths = [str(p) for p in additional_d_folders]

        result = executor.execute(
            data_path=str(d_folder),
            fasta_path=fasta_path,
            output_dir=str(diann_output),
            additional_data_paths=additional_paths,
        )

        if result.success:
            report_path = Path(result.report_path)
            if report_path.exists():
                logger.info(f"[{test_id}] DiaNN completed: {report_path}")
                return report_path

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
    additional_d_folders: Optional[List[Path]] = None,
    is_phospho: bool = False,
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
        additional_d_folders: Additional .d folders for multi-sample analysis.
        is_phospho: Whether this is a phosphoproteomics experiment.

    Returns:
        Path to FragPipe output directory, or None if failed.
    """
    if additional_d_folders:
        logger.info(f"[{test_id}] Running FragPipe analysis on {1 + len(additional_d_folders)} samples...")
    else:
        logger.info(f"[{test_id}] Running FragPipe analysis...")

    try:
        from imspy_simulation.timsim.validate.fragpipe_executor import FragPipeExecutor, FragPipeConfig

        # Select appropriate workflow (phospho-specific if available)
        if is_dda:
            if is_phospho:
                workflow = env_config.get("fragpipe_workflow_dda_phospho") or env_config.get("fragpipe_workflow_dda")
            else:
                workflow = env_config.get("fragpipe_workflow_dda")
        else:
            if is_phospho:
                workflow = env_config.get("fragpipe_workflow_dia_phospho") or env_config.get("fragpipe_workflow_dia")
            else:
                workflow = env_config.get("fragpipe_workflow_dia")

        if is_phospho:
            logger.info(f"[{test_id}] Using phospho-specific workflow: {workflow}")

        if not workflow:
            logger.error(f"[{test_id}] No FragPipe workflow configured")
            return None

        config = FragPipeConfig(
            workflow_path=workflow,
            tools_folder=env_config.get("fragpipe_tools", ""),
            python_path=env_config.get("fragpipe_python", ""),
            diann_path=env_config.get("diann_path", ""),
            threads=env_config.get("num_threads", 8),
        )

        executor = FragPipeExecutor(
            executable_path=env_config.get("fragpipe_path", "fragpipe"),
            threads=env_config.get("num_threads", 8),
            timeout_seconds=env_config.get("fragpipe_timeout", 14400),
            config=config,
        )

        fragpipe_output = output_dir / "fragpipe"
        fragpipe_output.mkdir(parents=True, exist_ok=True)

        # Convert additional paths to strings
        additional_paths = None
        if additional_d_folders:
            additional_paths = [str(p) for p in additional_d_folders]

        result = executor.execute(
            data_path=str(d_folder),
            fasta_path=fasta_path,
            output_dir=str(fragpipe_output),
            workflow_path=workflow,
            is_dda=is_dda,
            additional_data_paths=additional_paths,
        )

        if result.success:
            logger.info(f"[{test_id}] FragPipe completed: {fragpipe_output}")
            return fragpipe_output

        if result.partial_results:
            logger.warning(f"[{test_id}] FragPipe pipeline failed but ident results available (ident-only mode)")
            return fragpipe_output

        logger.error(f"[{test_id}] FragPipe failed")
        return None

    except Exception as e:
        logger.error(f"[{test_id}] FragPipe error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_sage_analysis(
    d_folder: Path,
    fasta_path: str,
    output_dir: Path,
    env_config: Dict,
    test_id: str,
) -> Optional[Path]:
    """
    Run Sage search on DDA data.

    Args:
        d_folder: Path to the .d folder.
        fasta_path: Path to FASTA file (should include decoys).
        output_dir: Output directory for results.
        env_config: Environment configuration.
        test_id: Test identifier for logging.

    Returns:
        Path to Sage results.sage.tsv file, or None if failed.
    """
    logger.info(f"[{test_id}] Running Sage analysis...")

    try:
        from imspy_simulation.timsim.validate.sage_executor import run_sage, SageConfig

        # Configure Sage
        config = SageConfig(
            fasta_path=fasta_path,
            missed_cleavages=env_config.get("sage_missed_cleavages", 2),
            min_peptide_len=7,
            max_peptide_len=50,
            precursor_tol_ppm=env_config.get("sage_precursor_ppm", 20.0),
            fragment_tol_ppm=env_config.get("sage_fragment_ppm", 20.0),
            min_charge=2,
            max_charge=4,
            decoy_tag="rev_",
            generate_decoys=False,  # Assume FASTA has decoys
        )

        sage_output = output_dir / "sage"
        sage_output.mkdir(parents=True, exist_ok=True)

        sage_path = env_config.get("sage_path")

        result = run_sage(
            d_folder=str(d_folder),
            fasta_path=fasta_path,
            output_dir=str(sage_output),
            config=config,
            sage_path=sage_path,
        )

        if result.success:
            results_path = Path(result.results_path)
            if results_path.exists():
                logger.info(f"[{test_id}] Sage completed: {result.psm_count} PSMs at 1% FDR")
                return results_path

        logger.error(f"[{test_id}] Sage failed")
        if result.log_path:
            logger.error(f"[{test_id}] Check log file: {result.log_path}")
        return None

    except Exception as e:
        logger.error(f"[{test_id}] Sage error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_validation(
    database_path: Path,
    diann_report: Optional[Path],
    fragpipe_output: Optional[Path],
    sage_results: Optional[Path],
    output_dir: Path,
    test_id: str,
    thresholds: TestThresholds,
    tool_versions: Optional[Dict[str, str]] = None,
    test_metadata: Optional[Dict[str, str]] = None,
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
        tool_versions: Dictionary mapping tool names to version strings.
        test_metadata: Test metadata for HTML report (test_id, acquisition_type, sample_type, description).

    Returns:
        Tuple of (passed, metrics_dict).
    """
    logger.info(f"[{test_id}] Running validation...")

    try:
        from imspy_simulation.timsim.validate.tool_comparison import run_comparison

        validation_output = output_dir / "validation"
        validation_output.mkdir(parents=True, exist_ok=True)

        result = run_comparison(
            database_path=str(database_path),
            diann_report_path=str(diann_report) if diann_report else None,
            fragpipe_output_dir=str(fragpipe_output) if fragpipe_output else None,
            sage_results_path=str(sage_results) if sage_results else None,
            output_dir=str(validation_output),
            generate_plots=True,
            tool_versions=tool_versions,
            test_metadata=test_metadata,
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
                "max_species_ratio_error": thresholds.max_species_ratio_error,
                "min_ptm_site_accuracy": thresholds.min_ptm_site_accuracy,
            },
            "tool_results": {},
        }

        # Check if this is an HYE experiment (species breakdown available)
        has_species_breakdown = result.species_breakdown is not None

        # Check if this is a phospho experiment (PTM metrics available)
        has_ptm_metrics = result.ptm_metrics is not None

        for tool_name, tool_result in result.tool_results.items():
            # Get overlap stats for precision and ID rate
            overlap = result.gt_overlaps.get(tool_name)
            id_rate = overlap.identification_rate if overlap else 0.0
            precision = overlap.precision if overlap else 0.0

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

            # Check species ratio error (for HYE experiments)
            species_ratio_error = None
            if has_species_breakdown:
                species_ratio_error = result.species_breakdown.max_ratio_error_per_tool.get(tool_name, 0.0)
                if species_ratio_error > thresholds.max_species_ratio_error:
                    tool_passed = False
                    failures.append(
                        f"Species ratio error {species_ratio_error:.2%} > "
                        f"{thresholds.max_species_ratio_error:.2%}"
                    )

            # Check PTM site accuracy (for phosphoproteomics experiments)
            ptm_site_accuracy = None
            if has_ptm_metrics:
                ptm_site_accuracy = result.ptm_metrics.site_accuracy_per_tool.get(tool_name, 0.0)
                if ptm_site_accuracy < thresholds.min_ptm_site_accuracy:
                    tool_passed = False
                    failures.append(
                        f"PTM site accuracy {ptm_site_accuracy:.2%} < "
                        f"{thresholds.min_ptm_site_accuracy:.2%}"
                    )

            metrics["tool_results"][tool_name] = {
                "id_count": tool_result.num_precursors,
                "precision": precision,
                "id_rate": id_rate,
                "rt_correlation": rt_corr,
                "im_correlation": im_corr,
                "species_ratio_error": species_ratio_error,
                "ptm_site_accuracy": ptm_site_accuracy,
                "passed": tool_passed,
                "failures": failures,
            }

            if not tool_passed:
                passed = False

        # Add species breakdown summary to metrics if available
        if has_species_breakdown:
            metrics["species_breakdown"] = {
                "expected_ratios": result.species_breakdown.expected_ratios,
                "ground_truth_counts": result.species_breakdown.ground_truth_counts,
                "observed_ratios_per_tool": result.species_breakdown.observed_ratios_per_tool,
                "max_ratio_error_per_tool": result.species_breakdown.max_ratio_error_per_tool,
            }

        # Add PTM metrics summary if available
        if has_ptm_metrics:
            metrics["ptm_metrics"] = {
                "ground_truth_phosphopeptides": result.ptm_metrics.ground_truth_phosphopeptides,
                "identified_phosphopeptides_per_tool": result.ptm_metrics.identified_phosphopeptides_per_tool,
                "correctly_localized_per_tool": result.ptm_metrics.correctly_localized_per_tool,
                "site_accuracy_per_tool": result.ptm_metrics.site_accuracy_per_tool,
            }

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


def run_paired_test_evaluation(
    test_id: str,
    env_config: Dict,
    tool: str = "both",
    skip_analysis: bool = False,
) -> Tuple[bool, Dict]:
    """
    Run evaluation for a paired A/B test (e.g., HYE fold-change experiment).

    Args:
        test_id: Test identifier (e.g., IT-DIA-HYE).
        env_config: Environment configuration.
        tool: Which tool(s) to run ("diann", "fragpipe", or "both").
        skip_analysis: Skip analysis, only run validation on existing outputs.

    Returns:
        Tuple of (passed, metrics).
    """
    logger.info(f"=" * 60)
    logger.info(f"Evaluating paired test: {test_id}")
    logger.info(f"=" * 60)

    paired_config = PAIRED_TESTS.get(test_id)
    if not paired_config:
        logger.error(f"[{test_id}] Not a paired test")
        return False, {"error": "Not a paired test"}

    condition_a = paired_config["condition_a"]
    condition_b = paired_config["condition_b"]

    # Load test config for metadata
    description = ""
    try:
        test_config = load_test_config(test_id)
        test_meta = test_config.get("test_metadata", {})
        acquisition_type = test_meta.get("acquisition_type", "DIA")
        sample_type = test_meta.get("sample_type", "hela")
        description = test_meta.get("description", "")
        is_dda = acquisition_type.upper() == "DDA"
    except Exception:
        is_dda = "DDA" in test_id
        sample_type = "hye" if "HYE" in test_id else "hela"
        acquisition_type = "DDA" if is_dda else "DIA"

    # Get thresholds
    thresholds = get_test_thresholds(test_id)

    # Find test output directories for both conditions
    output_base = Path(env_config.get("output_base", "."))

    # Find outputs for condition A
    test_dir_a = output_base / condition_a
    d_folder_a, db_path_a = find_simulation_outputs(test_dir_a)

    # Find outputs for condition B
    test_dir_b = output_base / condition_b
    d_folder_b, db_path_b = find_simulation_outputs(test_dir_b)

    # Check all required outputs exist
    if not d_folder_a:
        logger.error(f"[{test_id}] No .d folder found for {condition_a}")
        logger.error(f"[{test_id}] Run simulation with: --test {condition_a}")
        return False, {"error": f"No .d folder for {condition_a}"}

    if not d_folder_b:
        logger.error(f"[{test_id}] No .d folder found for {condition_b}")
        logger.error(f"[{test_id}] Run simulation with: --test {condition_b}")
        return False, {"error": f"No .d folder for {condition_b}"}

    if not db_path_a:
        logger.error(f"[{test_id}] No synthetic_data.db found for {condition_a}")
        return False, {"error": f"No database for {condition_a}"}

    logger.info(f"[{test_id}] Found condition A: {d_folder_a}")
    logger.info(f"[{test_id}] Found condition B: {d_folder_b}")
    logger.info(f"[{test_id}] Using database from condition A: {db_path_a}")

    # Create combined output directory
    test_dir = output_base / test_id
    test_dir.mkdir(parents=True, exist_ok=True)

    # Determine FASTA paths
    fasta_key = f"fasta_{sample_type}"
    fasta_path = resolve_fasta_path(env_config, fasta_key)

    fasta_key_decoys = f"fasta_{sample_type}_decoys"
    fasta_path_decoys = resolve_fasta_path(env_config, fasta_key_decoys)

    if not fasta_path:
        logger.error(f"[{test_id}] No FASTA path configured for {sample_type}")
        return False, {"error": f"No FASTA for {sample_type}"}

    # Run analysis tools with both samples
    diann_report = None
    fragpipe_output = None
    sage_results = None

    is_phospho = sample_type == "phospho"

    if not skip_analysis:
        if tool in ["diann", "both"]:
            # Run DiaNN with both conditions for proper MBR and quantification
            diann_report = run_diann_analysis(
                d_folder_a,
                fasta_path,
                test_dir,
                env_config,
                test_id,
                is_dda,
                is_phospho,
                additional_d_folders=[d_folder_b],
            )

        if tool in ["fragpipe", "both"]:
            # Run FragPipe with both conditions
            # FragPipe requires decoy FASTA for Philosopher steps (both DDA and DIA)
            # Note: diaTracer converts DIA to pseudo-DDA but doesn't add decoys to FASTA
            fragpipe_fasta = fasta_path_decoys if fasta_path_decoys else fasta_path
            if fragpipe_fasta != fasta_path:
                logger.info(f"[{test_id}] Using decoy FASTA for FragPipe: {fragpipe_fasta}")
            fragpipe_output = run_fragpipe_analysis(
                d_folder_a,
                fragpipe_fasta,
                test_dir,
                env_config,
                test_id,
                is_dda,
                additional_d_folders=[d_folder_b],
                is_phospho=is_phospho,
            )

    else:
        # Look for existing analysis outputs in combined directory
        diann_parquet = test_dir / "diann" / "report.parquet"
        diann_tsv = test_dir / "diann" / "report.tsv"
        if diann_parquet.exists():
            diann_report = diann_parquet
            logger.info(f"[{test_id}] Using existing DiaNN output: {diann_report}")
        elif diann_tsv.exists():
            diann_report = diann_tsv
            logger.info(f"[{test_id}] Using existing DiaNN output: {diann_report}")

        existing_fragpipe = test_dir / "fragpipe"
        if existing_fragpipe.exists():
            fragpipe_output = existing_fragpipe
            logger.info(f"[{test_id}] Using existing FragPipe output: {fragpipe_output}")

    # Run validation
    if not diann_report and not fragpipe_output and not sage_results:
        logger.error(f"[{test_id}] No analysis results to validate")
        return False, {"error": "No analysis results"}

    # Get tool versions for reporting
    tool_versions = get_tool_versions(env_config)

    # Build test metadata for HTML report
    test_metadata = {
        "test_id": test_id,
        "acquisition_type": acquisition_type,
        "sample_type": sample_type,
        "description": f"{description} (Paired A/B evaluation)",
        "condition_a": condition_a,
        "condition_b": condition_b,
    }

    passed, metrics = run_validation(
        db_path_a, diann_report, fragpipe_output, sage_results, test_dir, test_id, thresholds,
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
    # Check if this is a paired test that requires A/B conditions
    if is_paired_test(test_id):
        return run_paired_test_evaluation(test_id, env_config, tool, skip_analysis)

    logger.info(f"=" * 60)
    logger.info(f"Evaluating test: {test_id}")
    logger.info(f"=" * 60)

    # Load test config for metadata
    description = ""
    try:
        test_config = load_test_config(test_id)
        test_meta = test_config.get("test_metadata", {})
        acquisition_type = test_meta.get("acquisition_type", "DIA")
        sample_type = test_meta.get("sample_type", "hela")
        description = test_meta.get("description", "")
        is_dda = acquisition_type.upper() == "DDA"
    except Exception:
        is_dda = "DDA" in test_id
        sample_type = "hela"
        acquisition_type = "DDA" if is_dda else "DIA"

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

    # Determine FASTA paths
    # DiaNN can use regular FASTA (generates decoys internally)
    # FragPipe/Sage require decoys in the FASTA file
    # Use resolve_fasta_path to fall back to bundled FASTAs if not configured
    fasta_key = f"fasta_{sample_type}"
    fasta_path = resolve_fasta_path(env_config, fasta_key)

    # Try to get decoy FASTA for FragPipe/Sage
    fasta_key_decoys = f"fasta_{sample_type}_decoys"
    fasta_path_decoys = resolve_fasta_path(env_config, fasta_key_decoys)

    if not fasta_path:
        logger.error(f"[{test_id}] No FASTA path configured for {sample_type}")
        return False, {"error": f"No FASTA for {sample_type}"}

    # Run analysis tools
    diann_report = None
    fragpipe_output = None
    sage_results = None

    # Detect if this is a phospho experiment
    is_phospho = sample_type == "phospho"

    if not skip_analysis:
        if tool in ["diann", "both"]:
            diann_report = run_diann_analysis(
                d_folder, fasta_path, test_dir, env_config, test_id, is_dda, is_phospho
            )

        if tool in ["fragpipe", "both"]:
            # Use decoy FASTA for FragPipe if available
            fragpipe_fasta = fasta_path_decoys if fasta_path_decoys else fasta_path
            if fragpipe_fasta != fasta_path:
                logger.info(f"[{test_id}] Using decoy FASTA for FragPipe: {fragpipe_fasta}")
            fragpipe_output = run_fragpipe_analysis(
                d_folder, fragpipe_fasta, test_dir, env_config, test_id, is_dda,
                is_phospho=is_phospho,
            )

        # Run Sage for DDA tests (Sage is DDA-only)
        if is_dda and tool in ["sage", "both"]:
            # Sage requires FASTA with decoys
            sage_fasta = fasta_path_decoys if fasta_path_decoys else fasta_path
            if sage_fasta != fasta_path:
                logger.info(f"[{test_id}] Using decoy FASTA for Sage: {sage_fasta}")
            sage_results = run_sage_analysis(
                d_folder, sage_fasta, test_dir, env_config, test_id
            )
    else:
        # Look for existing analysis outputs
        # Prefer parquet over tsv, and avoid .stats.tsv files
        diann_parquet = test_dir / "diann" / "report.parquet"
        diann_tsv = test_dir / "diann" / "report.tsv"
        if diann_parquet.exists():
            diann_report = diann_parquet
            logger.info(f"[{test_id}] Using existing DiaNN output: {diann_report}")
        elif diann_tsv.exists():
            diann_report = diann_tsv
            logger.info(f"[{test_id}] Using existing DiaNN output: {diann_report}")

        existing_fragpipe = test_dir / "fragpipe"
        if existing_fragpipe.exists():
            fragpipe_output = existing_fragpipe
            logger.info(f"[{test_id}] Using existing FragPipe output: {fragpipe_output}")

        # Look for existing Sage results
        existing_sage = test_dir / "sage" / "results.sage.tsv"
        if existing_sage.exists():
            sage_results = existing_sage
            logger.info(f"[{test_id}] Using existing Sage output: {sage_results}")

    # Run validation
    if not diann_report and not fragpipe_output and not sage_results:
        logger.error(f"[{test_id}] No analysis results to validate")
        return False, {"error": "No analysis results"}

    # Get tool versions for reporting
    tool_versions = get_tool_versions(env_config)

    # Build test metadata for HTML report
    test_metadata = {
        "test_id": test_id,
        "acquisition_type": acquisition_type,
        "sample_type": sample_type,
        "description": description,
    }

    passed, metrics = run_validation(
        db_path, diann_report, fragpipe_output, sage_results, test_dir, test_id, thresholds,
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
        choices=["diann", "fragpipe", "sage", "both"],
        default="both",
        help="Which analysis tool(s) to run (default: both). Sage is only used for DDA tests.",
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

    # Generate meta HTML report (dashboard)
    try:
        from imspy_simulation.timsim.validate.html_report import generate_meta_report

        # Build benchmark results for meta report
        benchmark_results = []
        sample_type_labels = {
            "hela": "HeLa Proteome",
            "hye": "HYE Mixed Species (Human/Yeast/E.coli)",
            "phospho": "Phosphoproteomics (PTM Localization)",
            "hla": "HLA Immunopeptidomics",
        }

        for test_id, result in results.items():
            # Get test metadata
            try:
                test_config = load_test_config(test_id)
                test_meta = test_config.get("test_metadata", {})
                acquisition_type = test_meta.get("acquisition_type", "DIA")
                sample_type = test_meta.get("sample_type", "hela")
            except Exception:
                acquisition_type = "DDA" if "DDA" in test_id else "DIA"
                sample_type = "hela"

            sample_label = sample_type_labels.get(sample_type, sample_type.title())
            benchmark_type = f"{acquisition_type}-PASEF {sample_label}"

            # Build tool results for meta report
            tool_results_meta = {}
            for tool_name, tool_metrics in result.get("metrics", {}).get("tool_results", {}).items():
                tool_results_meta[tool_name] = {
                    "passed": tool_metrics.get("passed", False),
                    "id_rate": tool_metrics.get("id_rate", 0),
                    "precision": tool_metrics.get("precision", 0),
                }

            # Relative path to individual report
            report_path = f"{test_id}/validation/validation_report.html"

            benchmark_results.append({
                "test_id": test_id,
                "passed": result.get("passed", False),
                "benchmark_type": benchmark_type,
                "acquisition_type": acquisition_type,
                "sample_type": sample_type,
                "report_path": report_path,
                "tool_results": tool_results_meta,
            })

        # Generate meta report
        meta_report_path = Path(output_base) / "index.html"
        generate_meta_report(
            benchmark_results=benchmark_results,
            output_path=str(meta_report_path),
            title="TIMSIM Integration Test Dashboard",
        )
        logger.info(f"Dashboard written to: {meta_report_path}")

    except Exception as e:
        logger.warning(f"Failed to generate meta report: {e}")

    # Create zip bundle of all validation results
    try:
        zip_path = create_results_bundle(
            output_base=output_base,
            test_ids=tests_to_run,
            timestamp=start_time,
        )
        if zip_path:
            logger.info(f"Results bundle: {zip_path}")
    except Exception as e:
        logger.warning(f"Failed to create results bundle: {e}")

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
