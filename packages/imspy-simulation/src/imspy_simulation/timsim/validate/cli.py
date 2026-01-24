"""
Command-line interface for timsim-validate.
"""

import argparse
import logging
import sys
import os

import toml

from .runner import ValidationRunner, SimulationConfig
from .diann_executor import DiannExecutor, DiannConfig
from .fragpipe_executor import FragPipeExecutor, FragPipeConfig
from .metrics import ValidationThresholds


def get_default_config() -> dict:
    """Return default configuration values for timsim-validate."""
    return {
        # Paths
        "reference_path": None,
        "fasta_path": None,
        "diann_path": "diann",
        "output_dir": "./timsim-validate-output",
        "existing_simulation": None,  # Path to existing .d folder to skip simulation
        "database_path": None,  # Explicit path to synthetic_data.db

        # Simulation
        "num_peptides": 5000,
        "gradient_length": 1800.0,
        "acquisition_type": "DIA",
        "apply_fragmentation": True,
        "batch_size": 256,
        "num_threads": -1,

        # DiaNN basic
        "diann_threads": 4,
        "diann_timeout": 3600,
        "diann_qvalue": 0.01,
        "diann_library_free": True,
        "diann_use_predictor": True,

        # DiaNN mass accuracy
        "diann_mass_acc": None,
        "diann_mass_acc_ms1": None,

        # DiaNN peptide settings
        "diann_min_pep_len": 7,
        "diann_max_pep_len": 30,
        "diann_missed_cleavages": 2,
        "diann_enzyme": "K*,R*",
        "diann_met_excision": True,

        # DiaNN fragment settings
        "diann_min_fr_mz": None,
        "diann_max_fr_mz": None,

        # DiaNN modifications
        "diann_fixed_mod": None,
        "diann_var_mod": None,
        "diann_var_mods": 2,

        # Thresholds
        "min_identification_rate": 0.30,
        "min_rt_correlation": 0.90,
        "min_im_correlation": 0.90,
        "max_rt_mae_minutes": 1.0,
        "max_im_mae": 0.05,

        # Behavior
        "keep_temp": False,
        "verbose": False,
        "quiet": False,

        # Analysis tool selection
        "analysis_tool": "diann",  # "diann" or "fragpipe"

        # FragPipe settings
        "fragpipe_path": None,  # Path to fragpipe executable
        "fragpipe_workflow": None,  # Path to workflow file
        "fragpipe_tools_folder": None,  # FragPipe tools folder
        "fragpipe_python": None,  # Python executable for FragPipe
        "fragpipe_timeout": 7200,  # 2 hours
        "existing_fragpipe_output": None,  # Path to existing FragPipe output
    }


def load_toml_config(config_path: str) -> dict:
    """
    Load a TOML configuration file and flatten sections.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        Flattened dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        toml.TomlDecodeError: If the config file is invalid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_config = toml.load(f)

    # Flatten sections into a single dict
    flat_config = {}
    for section_key, section_value in raw_config.items():
        if isinstance(section_value, dict):
            # Map TOML section keys to flat config keys
            for key, value in section_value.items():
                # Handle key mapping from TOML naming to internal naming
                if section_key == "diann":
                    # Map diann section keys to diann_ prefixed config keys
                    diann_key_mapping = {
                        "executable_path": "diann_path",
                        "threads": "diann_threads",
                        "timeout_seconds": "diann_timeout",
                        "qvalue": "diann_qvalue",
                        "library_free": "diann_library_free",
                        "use_predictor": "diann_use_predictor",
                        "mass_acc": "diann_mass_acc",
                        "mass_acc_ms1": "diann_mass_acc_ms1",
                        "min_pep_len": "diann_min_pep_len",
                        "max_pep_len": "diann_max_pep_len",
                        "missed_cleavages": "diann_missed_cleavages",
                        "enzyme": "diann_enzyme",
                        "met_excision": "diann_met_excision",
                        "min_fr_mz": "diann_min_fr_mz",
                        "max_fr_mz": "diann_max_fr_mz",
                        "fixed_mod": "diann_fixed_mod",
                        "var_mod": "diann_var_mod",
                        "var_mods": "diann_var_mods",
                    }
                    if key in diann_key_mapping:
                        flat_config[diann_key_mapping[key]] = value
                    else:
                        flat_config[f"diann_{key}"] = value
                elif section_key == "paths":
                    # Map paths section keys directly
                    flat_config[key] = value
                elif section_key == "behavior":
                    # Behavior section maps directly
                    flat_config[key] = value
                elif section_key == "simulation":
                    # Simulation section maps directly
                    flat_config[key] = value
                elif section_key == "thresholds":
                    # Thresholds section maps directly
                    flat_config[key] = value
                elif section_key == "fragpipe":
                    # Map fragpipe section keys to fragpipe_ prefixed config keys
                    fragpipe_key_mapping = {
                        "executable_path": "fragpipe_path",
                        "workflow": "fragpipe_workflow",
                        "tools_folder": "fragpipe_tools_folder",
                        "python": "fragpipe_python",
                        "timeout_seconds": "fragpipe_timeout",
                        "existing_output": "existing_fragpipe_output",
                    }
                    if key in fragpipe_key_mapping:
                        flat_config[fragpipe_key_mapping[key]] = value
                    else:
                        flat_config[f"fragpipe_{key}"] = value
                else:
                    flat_config[key] = value
        else:
            flat_config[section_key] = section_value

    return flat_config


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """
    Merge TOML config with CLI arguments (CLI takes precedence).

    Args:
        config: Dictionary from TOML config.
        args: Parsed CLI arguments.

    Returns:
        Merged configuration dictionary.
    """
    merged = config.copy()

    # Map CLI arg names to config keys
    cli_mapping = {
        "reference_path": "reference_path",
        "fasta": "fasta_path",
        "diann_path": "diann_path",
        "output_dir": "output_dir",
        "existing_simulation": "existing_simulation",
        "database_path": "database_path",
        "num_peptides": "num_peptides",
        "gradient_length": "gradient_length",
        "min_id_rate": "min_identification_rate",
        "min_rt_corr": "min_rt_correlation",
        "min_im_corr": "min_im_correlation",
        "max_rt_mae": "max_rt_mae_minutes",
        "max_im_mae": "max_im_mae",
        "diann_threads": "diann_threads",
        "diann_timeout": "diann_timeout",
        "keep_temp": "keep_temp",
        "verbose": "verbose",
        "quiet": "quiet",
        # FragPipe options
        "analysis_tool": "analysis_tool",
        "fragpipe_path": "fragpipe_path",
        "fragpipe_workflow": "fragpipe_workflow",
        "fragpipe_tools_folder": "fragpipe_tools_folder",
        "fragpipe_python": "fragpipe_python",
        "fragpipe_timeout": "fragpipe_timeout",
        "existing_fragpipe_output": "existing_fragpipe_output",
    }

    for cli_key, config_key in cli_mapping.items():
        cli_value = getattr(args, cli_key, None)
        # Only override if CLI provided a non-default value
        # For boolean flags, they're False by default, so check explicitly
        if cli_key in ("keep_temp", "verbose", "quiet"):
            if cli_value:  # Only override if True
                merged[config_key] = cli_value
        elif cli_value is not None:
            merged[config_key] = cli_value

    return merged


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
  # Using TOML config file
  timsim-validate --config /path/to/config.toml

  # Using CLI arguments
  timsim-validate /path/to/reference.d
  timsim-validate /path/to/reference.d --fasta /path/to/custom.fasta

  # Mix: TOML config with CLI overrides
  timsim-validate --config config.toml --min-id-rate 0.4 --verbose

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

    # Config file argument
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        metavar="CONFIG_FILE",
        help="Path to TOML configuration file (CLI args override config values)",
    )

    # Reference path - optional if config is provided
    parser.add_argument(
        "reference_path",
        type=str,
        nargs="?",
        default=None,
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
        default=None,
        help="Path to DiaNN executable (default: 'diann' in PATH)",
    )
    path_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: ./timsim-validate-output)",
    )
    path_group.add_argument(
        "--existing-simulation",
        type=str,
        default=None,
        help="Path to existing .d folder to skip simulation step",
    )
    path_group.add_argument(
        "--database-path",
        type=str,
        default=None,
        help="Explicit path to synthetic_data.db (required with --existing-simulation if DB is not in simulation dir)",
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

    # Analysis tool selection
    tool_group = parser.add_argument_group("Analysis Tool")
    tool_group.add_argument(
        "--analysis-tool",
        type=str,
        choices=["diann", "fragpipe", "both"],
        default=None,
        help="Analysis tool to use: diann, fragpipe, or both (default: diann)",
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

    # FragPipe arguments
    fragpipe_group = parser.add_argument_group("FragPipe Options")
    fragpipe_group.add_argument(
        "--fragpipe-path",
        type=str,
        default=None,
        help="Path to FragPipe executable",
    )
    fragpipe_group.add_argument(
        "--fragpipe-workflow",
        type=str,
        default=None,
        help="Path to FragPipe workflow file (.workflow)",
    )
    fragpipe_group.add_argument(
        "--fragpipe-tools-folder",
        type=str,
        default=None,
        help="Path to FragPipe tools folder",
    )
    fragpipe_group.add_argument(
        "--fragpipe-python",
        type=str,
        default=None,
        help="Path to Python executable for FragPipe",
    )
    fragpipe_group.add_argument(
        "--fragpipe-timeout",
        type=int,
        default=7200,
        help="FragPipe timeout in seconds (default: 7200 = 2 hours)",
    )
    fragpipe_group.add_argument(
        "--existing-fragpipe-output",
        type=str,
        default=None,
        help="Path to existing FragPipe output directory (skip analysis)",
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

    # Build configuration: defaults -> TOML config -> CLI args
    config = get_default_config()

    # Load TOML config if provided
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
            return 5
        try:
            toml_config = load_toml_config(args.config)
            config.update(toml_config)
        except Exception as e:
            print(f"Error loading configuration file: {e}", file=sys.stderr)
            return 5

    # Merge CLI arguments (CLI takes precedence)
    config = merge_config_with_args(config, args)

    # Setup logging
    setup_logging(verbose=config["verbose"], quiet=config["quiet"])

    # Print banner unless quiet
    if not config["quiet"]:
        print(banner())

    # Validate paths
    reference_path = config.get("reference_path")
    existing_simulation = config.get("existing_simulation")

    # reference_path is only required if not using existing simulation
    if not existing_simulation:
        if not reference_path:
            print("Error: reference_path is required (provide via CLI or config file)", file=sys.stderr)
            return 5
        if not os.path.exists(reference_path):
            print(f"Error: Reference path does not exist: {reference_path}", file=sys.stderr)
            return 5

    # Validate existing simulation path if provided
    if existing_simulation:
        if not os.path.exists(existing_simulation):
            print(f"Error: Existing simulation path does not exist: {existing_simulation}", file=sys.stderr)
            return 5
        # fasta_path is required when using existing simulation
        if not config.get("fasta_path"):
            print("Error: fasta_path is required when using existing_simulation", file=sys.stderr)
            return 5

    # Create configuration objects
    thresholds = ValidationThresholds(
        min_identification_rate=config["min_identification_rate"],
        min_rt_correlation=config["min_rt_correlation"],
        min_im_correlation=config["min_im_correlation"],
        max_rt_mae_minutes=config["max_rt_mae_minutes"],
        max_im_mae=config["max_im_mae"],
    )

    simulation_config = SimulationConfig(
        num_peptides=config["num_peptides"],
        gradient_length=config["gradient_length"],
    )

    diann_config = DiannConfig(
        qvalue=config["diann_qvalue"],
        library_free=config["diann_library_free"],
        use_predictor=config["diann_use_predictor"],
        mass_acc=config["diann_mass_acc"],
        mass_acc_ms1=config["diann_mass_acc_ms1"],
        min_pep_len=config["diann_min_pep_len"],
        max_pep_len=config["diann_max_pep_len"],
        missed_cleavages=config["diann_missed_cleavages"],
        enzyme=config["diann_enzyme"],
        met_excision=config["diann_met_excision"],
        min_fr_mz=config["diann_min_fr_mz"],
        max_fr_mz=config["diann_max_fr_mz"],
        fixed_mod=config["diann_fixed_mod"],
        var_mod=config["diann_var_mod"],
        var_mods=config["diann_var_mods"],
    )

    diann_executor = DiannExecutor(
        executable_path=config["diann_path"],
        threads=config["diann_threads"],
        timeout_seconds=config["diann_timeout"],
        config=diann_config,
    )

    # Create FragPipe executor if needed
    fragpipe_executor = None
    analysis_tool = config.get("analysis_tool", "diann")

    if analysis_tool in ("fragpipe", "both") or config.get("existing_fragpipe_output"):
        # Validate FragPipe requirements for "both" mode
        if analysis_tool == "both":
            if not config.get("fragpipe_path") and not config.get("existing_fragpipe_output"):
                print("Error: --fragpipe-path is required when using --analysis-tool both", file=sys.stderr)
                return 5
            if not config.get("fragpipe_workflow") and not config.get("existing_fragpipe_output"):
                print("Error: --fragpipe-workflow is required when using --analysis-tool both", file=sys.stderr)
                return 5

        fragpipe_config = FragPipeConfig(
            workflow_path=config.get("fragpipe_workflow"),
            tools_folder=config.get("fragpipe_tools_folder"),
            diann_path=config.get("diann_path"),
            python_path=config.get("fragpipe_python"),
        )

        fragpipe_executor = FragPipeExecutor(
            executable_path=config.get("fragpipe_path") or "fragpipe",
            threads=config.get("diann_threads", 8),  # Reuse thread count
            timeout_seconds=config.get("fragpipe_timeout", 7200),
            config=fragpipe_config,
        )

    # Create and run validator
    runner = ValidationRunner(
        reference_path=reference_path,
        output_dir=config["output_dir"],
        fasta_path=config["fasta_path"],
        diann_executor=diann_executor,
        fragpipe_executor=fragpipe_executor,
        thresholds=thresholds,
        simulation_config=simulation_config,
        keep_temp=config["keep_temp"],
        verbose=config["verbose"],
        existing_simulation=config.get("existing_simulation"),
        database_path=config.get("database_path"),
        analysis_tool=analysis_tool,
        existing_fragpipe_output=config.get("existing_fragpipe_output"),
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

        if result.plot_paths and result.plot_paths.summary_plot:
            print(f"\nPlots saved to:")
            print(f"  Summary: {result.plot_paths.summary_plot}")
            if result.plot_paths.rt_correlation:
                print(f"  RT Correlation: {result.plot_paths.rt_correlation}")
            if result.plot_paths.im_correlation:
                print(f"  IM Correlation: {result.plot_paths.im_correlation}")
            if result.plot_paths.intensity_histogram:
                print(f"  Intensity Histogram: {result.plot_paths.intensity_histogram}")
            if result.plot_paths.quant_correlation:
                print(f"  Quant Correlation: {result.plot_paths.quant_correlation}")
            if result.plot_paths.charge_state_breakdown:
                print(f"  Charge State Breakdown: {result.plot_paths.charge_state_breakdown}")
            if result.plot_paths.intensity_id_rate:
                print(f"  Intensity ID Rate: {result.plot_paths.intensity_id_rate}")
            if result.plot_paths.peptide_length_breakdown:
                print(f"  Peptide Length: {result.plot_paths.peptide_length_breakdown}")
            if result.plot_paths.missed_cleavages_breakdown:
                print(f"  Missed Cleavages: {result.plot_paths.missed_cleavages_breakdown}")
            if result.plot_paths.mass_accuracy:
                print(f"  Mass Accuracy: {result.plot_paths.mass_accuracy}")

    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
