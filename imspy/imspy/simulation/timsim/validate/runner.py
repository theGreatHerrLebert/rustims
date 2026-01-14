"""
Main orchestration logic for timsim-validate.
"""

import os
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Literal

from .diann_executor import DiannExecutor, DiannError
from .fragpipe_executor import FragPipeExecutor, FragPipeError, FragPipeConfig
from .parsing import parse_diann_report, parse_fragpipe_psm, parse_fragpipe_combined
from .comparison import (
    load_ground_truth,
    create_peptide_sets,
    calculate_identification_metrics,
    match_results,
    calculate_correlation_metrics,
)
from .metrics import (
    ValidationMetrics,
    ValidationThresholds,
    check_thresholds,
    create_metrics_from_comparison,
)
from .report import (
    generate_json_report,
    generate_text_summary,
    save_text_report,
)
from .plots import generate_all_plots, PlotPaths
from .tool_comparison import (
    run_comparison as run_tool_comparison,
    generate_comparison_text_report,
    generate_comparison_plots,
)

logger = logging.getLogger(__name__)


def detect_acquisition_type(reference_path: str) -> str:
    """
    Auto-detect whether a reference .d folder is DDA or DIA.

    Checks for presence of DIA-specific tables (DiaFrameMsMsInfo, DiaFrameMsMsWindows)
    vs DDA-specific tables (PasefFrameMsMsInfo).

    Args:
        reference_path: Path to reference .d folder

    Returns:
        "DDA" or "DIA"
    """
    import sqlite3

    tdf_path = os.path.join(reference_path, "analysis.tdf")
    if not os.path.exists(tdf_path):
        logger.warning(f"Could not find analysis.tdf in {reference_path}, defaulting to DIA")
        return "DIA"

    try:
        conn = sqlite3.connect(tdf_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        # Check for DIA-specific tables
        has_dia_tables = "DiaFrameMsMsInfo" in tables or "DiaFrameMsMsWindows" in tables
        # Check for DDA-specific tables
        has_dda_tables = "PasefFrameMsMsInfo" in tables

        if has_dia_tables and not has_dda_tables:
            logger.info(f"Detected DIA acquisition type from reference")
            return "DIA"
        elif has_dda_tables and not has_dia_tables:
            logger.info(f"Detected DDA acquisition type from reference")
            return "DDA"
        elif has_dia_tables and has_dda_tables:
            # Both present - check which has data
            conn = sqlite3.connect(tdf_path)
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM DiaFrameMsMsInfo")
                dia_count = cursor.fetchone()[0]
            except:
                dia_count = 0
            try:
                cursor.execute("SELECT COUNT(*) FROM PasefFrameMsMsInfo")
                dda_count = cursor.fetchone()[0]
            except:
                dda_count = 0
            conn.close()

            if dda_count > dia_count:
                logger.info(f"Detected DDA acquisition type from reference (DDA entries: {dda_count}, DIA entries: {dia_count})")
                return "DDA"
            else:
                logger.info(f"Detected DIA acquisition type from reference (DIA entries: {dia_count}, DDA entries: {dda_count})")
                return "DIA"
        else:
            logger.warning(f"Could not detect acquisition type from reference tables, defaulting to DIA")
            return "DIA"

    except Exception as e:
        logger.warning(f"Error detecting acquisition type: {e}, defaulting to DIA")
        return "DIA"


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class SimulationError(ValidationError):
    """Error during simulation phase."""
    pass


class ComparisonError(ValidationError):
    """Error during comparison phase."""
    pass


@dataclass
class SimulationConfig:
    """Configuration for the validation simulation."""
    num_peptides: int = 5000
    gradient_length: float = 1800.0  # 30 minutes
    acquisition_type: str = "DIA"
    apply_fragmentation: bool = True
    batch_size: int = 256
    num_threads: int = -1
    silent_mode: bool = True


@dataclass
class ValidationResult:
    """Complete validation result."""
    success: bool
    metrics: Optional[ValidationMetrics] = None
    report_path: Optional[str] = None
    text_report_path: Optional[str] = None
    plot_paths: Optional[PlotPaths] = None
    exit_code: int = 0
    error_message: Optional[str] = None

    @staticmethod
    def from_error(message: str, exit_code: int) -> 'ValidationResult':
        """Create a failure result from an error."""
        return ValidationResult(
            success=False,
            exit_code=exit_code,
            error_message=message,
        )


class ValidationRunner:
    """Orchestrates the complete validation workflow."""

    def __init__(
        self,
        reference_path: str,
        output_dir: str,
        fasta_path: Optional[str] = None,
        diann_executor: Optional[DiannExecutor] = None,
        fragpipe_executor: Optional[FragPipeExecutor] = None,
        thresholds: Optional[ValidationThresholds] = None,
        simulation_config: Optional[SimulationConfig] = None,
        keep_temp: bool = False,
        verbose: bool = False,
        existing_simulation: Optional[str] = None,
        database_path: Optional[str] = None,
        analysis_tool: Literal["diann", "fragpipe", "both"] = "diann",
        existing_fragpipe_output: Optional[str] = None,
    ):
        """
        Initialize validation runner.

        Args:
            reference_path: Path to reference .d folder for instrument layout.
            output_dir: Output directory for results.
            fasta_path: Path to FASTA file (None = use bundled test FASTA).
            diann_executor: DiaNN executor instance.
            fragpipe_executor: FragPipe executor instance.
            thresholds: Validation thresholds.
            simulation_config: Simulation configuration.
            keep_temp: Keep temporary simulation files.
            verbose: Verbose output.
            existing_simulation: Path to existing .d folder to skip simulation.
            database_path: Explicit path to synthetic_data.db (required with existing_simulation).
            analysis_tool: Which analysis tool to use ("diann" or "fragpipe").
            existing_fragpipe_output: Path to existing FragPipe output directory (skip analysis).
        """
        self.reference_path = reference_path
        self.output_dir = output_dir
        self.fasta_path = fasta_path or self._get_bundled_fasta()
        self.diann_executor = diann_executor or DiannExecutor()
        self.fragpipe_executor = fragpipe_executor
        self.thresholds = thresholds or ValidationThresholds()
        self.simulation_config = simulation_config or SimulationConfig()
        self.keep_temp = keep_temp
        self.verbose = verbose
        self.existing_simulation = existing_simulation
        self.database_path_override = database_path
        self.analysis_tool = analysis_tool
        self.existing_fragpipe_output = existing_fragpipe_output

        # Auto-detect acquisition type from reference if using default
        if self.simulation_config.acquisition_type == "DIA":
            detected_type = detect_acquisition_type(reference_path)
            if detected_type != self.simulation_config.acquisition_type:
                logger.info(f"Auto-detected acquisition type: {detected_type} (overriding default DIA)")
                self.simulation_config.acquisition_type = detected_type

        # Will be set during run
        self._simulation_path: Optional[str] = None
        self._database_path: Optional[str] = None
        self._diann_report_path: Optional[str] = None
        self._fragpipe_result: Optional[Any] = None

    def _get_bundled_fasta(self) -> str:
        """Get path to bundled test FASTA."""
        # Look for bundled FASTA in integration module
        integration_fasta = Path(__file__).parent.parent / "integration" / "data" / "fasta" / "hela-small.fasta"

        if integration_fasta.exists():
            return str(integration_fasta)

        raise FileNotFoundError(
            "Bundled test FASTA not found. Please provide a FASTA file with --fasta."
        )

    def run(self) -> ValidationResult:
        """
        Execute complete validation workflow.

        Returns:
            ValidationResult with metrics and report paths.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        try:
            # Step 1: Run simulation (or use existing)
            if self.existing_simulation:
                logger.info("Step 1/6: Using existing simulation...")
                self._use_existing_simulation()
            else:
                logger.info("Step 1/6: Running timsim simulation...")
                self._run_simulation()

            # Step 2: Execute analysis tool (DiaNN, FragPipe, or both)
            if self.analysis_tool == "both":
                logger.info("Step 2/6: Executing DiaNN analysis...")
                self._run_diann()
                logger.info("Step 2.5/6: Executing FragPipe analysis...")
                self._run_fragpipe()
            elif self.analysis_tool == "fragpipe":
                logger.info("Step 2/6: Executing FragPipe analysis...")
                self._run_fragpipe()
            else:
                logger.info("Step 2/6: Executing DiaNN analysis...")
                self._run_diann()

            # Step 3: Load ground truth
            logger.info("Step 3/6: Loading ground truth data...")
            ground_truth = load_ground_truth(self._database_path)

            # Step 4: Parse results and compare
            logger.info("Step 4/6: Comparing results...")

            # Handle "both" mode with multi-tool comparison
            if self.analysis_tool == "both":
                return self._run_both_tools_comparison(ground_truth)

            if self.analysis_tool == "fragpipe":
                analysis_results = parse_fragpipe_combined(
                    psm_path=self._fragpipe_result.psm_path,
                    peptide_path=self._fragpipe_result.peptide_path,
                    protein_path=self._fragpipe_result.protein_path,
                    ion_path=self._fragpipe_result.ion_path,
                )
            else:
                analysis_results = parse_diann_report(self._diann_report_path)
            metrics, matched_df = self._compute_metrics(ground_truth, analysis_results)

            # Step 5: Generate plots
            logger.info("Step 5/6: Generating plots...")
            try:
                plot_paths = generate_all_plots(
                    ground_truth_df=ground_truth,
                    matched_df=matched_df,
                    metrics=metrics,
                    output_dir=self.output_dir,
                )
            except Exception as e:
                logger.warning(f"Failed to generate plots: {e}")
                plot_paths = PlotPaths()

            # Step 6: Generate reports
            logger.info("Step 6/6: Generating reports...")
            report_path = generate_json_report(
                metrics=metrics,
                thresholds=self.thresholds,
                output_dir=self.output_dir,
                simulation_path=self._simulation_path,
                diann_report_path=self._diann_report_path,
                fasta_path=self.fasta_path,
                plot_paths=plot_paths,
            )

            text_summary = generate_text_summary(metrics, self.thresholds)
            text_report_path = save_text_report(text_summary, self.output_dir)

            if self.verbose:
                print(text_summary)

            # Cleanup temp files if not keeping
            if not self.keep_temp and self._simulation_path:
                self._cleanup_simulation()

            exit_code = 0 if metrics.overall_pass else 1

            return ValidationResult(
                success=metrics.overall_pass,
                metrics=metrics,
                report_path=report_path,
                text_report_path=text_report_path,
                plot_paths=plot_paths,
                exit_code=exit_code,
            )

        except DiannError as e:
            logger.error(f"DiaNN error: {e}")
            return ValidationResult.from_error(str(e), exit_code=3)
        except FragPipeError as e:
            logger.error(f"FragPipe error: {e}")
            return ValidationResult.from_error(str(e), exit_code=6)
        except SimulationError as e:
            logger.error(f"Simulation error: {e}")
            return ValidationResult.from_error(str(e), exit_code=2)
        except ComparisonError as e:
            logger.error(f"Comparison error: {e}")
            return ValidationResult.from_error(str(e), exit_code=4)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return ValidationResult.from_error(str(e), exit_code=5)

    def _run_simulation(self) -> None:
        """Run timsim simulation."""
        # Create simulation output directory
        sim_dir = os.path.join(self.output_dir, "simulation")
        os.makedirs(sim_dir, exist_ok=True)

        # Create a minimal TOML config for the simulation
        config_path = os.path.join(sim_dir, "validate_config.toml")
        experiment_name = "timsim-validate"

        config_content = f"""
[paths]
save_path = "{sim_dir}"
reference_path = "{self.reference_path}"
fasta_path = "{self.fasta_path}"

[experiment]
experiment_name = "{experiment_name}"
acquisition_type = "{self.simulation_config.acquisition_type}"
gradient_length = {self.simulation_config.gradient_length}
apply_fragmentation = {str(self.simulation_config.apply_fragmentation).lower()}
silent_mode = {str(self.simulation_config.silent_mode).lower()}

[digestion]
num_sample_peptides = {self.simulation_config.num_peptides}
sample_peptides = true

[performance]
num_threads = {self.simulation_config.num_threads}
batch_size = {self.simulation_config.batch_size}
"""

        with open(config_path, 'w') as f:
            f.write(config_content)

        # Run timsim simulation
        try:
            from imspy.simulation.timsim.simulator import SimulationConfig, main as run_simulation
            import sys

            # Temporarily replace sys.argv to run the simulator
            old_argv = sys.argv
            sys.argv = ["timsim", config_path]

            try:
                # Import and run simulation directly
                # Note: This is a simplified approach - in production you might
                # want to call the simulator programmatically
                run_simulation()
            finally:
                sys.argv = old_argv

            # Set paths to simulation outputs
            self._simulation_path = os.path.join(sim_dir, experiment_name)
            self._database_path = os.path.join(self._simulation_path, "synthetic_data.db")

            if not os.path.exists(self._database_path):
                raise SimulationError(
                    f"Simulation completed but database not found: {self._database_path}"
                )

        except ImportError as e:
            raise SimulationError(f"Failed to import timsim simulator: {e}")
        except Exception as e:
            raise SimulationError(f"Simulation failed: {e}")

    def _use_existing_simulation(self) -> None:
        """Use an existing simulation instead of running a new one."""
        if not os.path.exists(self.existing_simulation):
            raise SimulationError(
                f"Existing simulation path not found: {self.existing_simulation}"
            )

        # The existing_simulation should be the .d folder path
        if self.existing_simulation.endswith(".d"):
            self._simulation_path = os.path.dirname(self.existing_simulation)
        else:
            self._simulation_path = self.existing_simulation

        # Use explicit database_path if provided, otherwise look in simulation directory
        if self.database_path_override:
            self._database_path = self.database_path_override
        else:
            self._database_path = os.path.join(self._simulation_path, "synthetic_data.db")

        if not os.path.exists(self._database_path):
            raise SimulationError(
                f"Database not found: {self._database_path}\n"
                f"When using existing_simulation, you may need to specify database_path "
                f"explicitly if the database is not in the simulation directory."
            )

        logger.info(f"  Using simulation: {self._simulation_path}")
        logger.info(f"  Database: {self._database_path}")

    def _run_diann(self) -> None:
        """Execute DiaNN analysis."""
        diann_output_dir = os.path.join(self.output_dir, "diann")

        # Find the .d folder in simulation output
        sim_d_folder = None
        for item in os.listdir(self._simulation_path):
            if item.endswith(".d"):
                sim_d_folder = os.path.join(self._simulation_path, item)
                break

        if sim_d_folder is None:
            # The simulation path might itself be the .d folder
            if self._simulation_path.endswith(".d"):
                sim_d_folder = self._simulation_path
            else:
                raise ComparisonError(
                    f"No .d folder found in simulation output: {self._simulation_path}"
                )

        result = self.diann_executor.execute(
            data_path=sim_d_folder,
            fasta_path=self.fasta_path,
            output_dir=diann_output_dir,
        )

        if not result.success:
            raise DiannError(f"DiaNN analysis failed. Check log at {result.log_path}")

        self._diann_report_path = result.report_path

    def _run_fragpipe(self) -> None:
        """Execute FragPipe analysis."""
        # Check if using existing FragPipe output
        if self.existing_fragpipe_output:
            logger.info(f"  Using existing FragPipe output: {self.existing_fragpipe_output}")
            self._fragpipe_result = self.fragpipe_executor.execute_from_existing_output(
                self.existing_fragpipe_output
            )
            return

        # Ensure FragPipe executor is configured
        if self.fragpipe_executor is None:
            raise FragPipeError(
                "FragPipe executor not configured. "
                "Provide --fragpipe-path and --fragpipe-workflow options."
            )

        fragpipe_output_dir = os.path.join(self.output_dir, "fragpipe")

        # Find the .d folder in simulation output
        sim_d_folder = None
        for item in os.listdir(self._simulation_path):
            if item.endswith(".d"):
                sim_d_folder = os.path.join(self._simulation_path, item)
                break

        if sim_d_folder is None:
            # The simulation path might itself be the .d folder
            if self._simulation_path.endswith(".d"):
                sim_d_folder = self._simulation_path
            else:
                raise ComparisonError(
                    f"No .d folder found in simulation output: {self._simulation_path}"
                )

        result = self.fragpipe_executor.execute(
            data_path=sim_d_folder,
            fasta_path=self.fasta_path,
            output_dir=fragpipe_output_dir,
        )

        if not result.success:
            raise FragPipeError(f"FragPipe analysis failed. Check log at {result.log_path}")

        self._fragpipe_result = result

    def _compute_metrics(
        self,
        ground_truth,
        diann_results,
    ) -> Tuple[ValidationMetrics, Any]:
        """Compute validation metrics from comparison."""
        from .comparison import (
            calculate_charge_state_metrics,
            calculate_intensity_bin_metrics,
            calculate_mass_accuracy_metrics,
            calculate_peptide_property_metrics,
        )

        # Create sets for identification metrics
        gt_peptides, gt_precursors = create_peptide_sets(
            ground_truth, use_modifications=False, normalize=True
        )
        diann_peptides, diann_precursors = create_peptide_sets(
            diann_results, use_modifications=False, normalize=True
        )

        # Calculate identification metrics at precursor level
        id_metrics = calculate_identification_metrics(gt_precursors, diann_precursors)

        # Match results for correlation analysis
        matched, unidentified, false_positives = match_results(
            ground_truth, diann_results, match_on="sequence", normalize=True
        )

        # Calculate correlation metrics
        correlation_metrics = calculate_correlation_metrics(matched)

        # Calculate detailed breakdown metrics
        charge_metrics = calculate_charge_state_metrics(ground_truth, matched, unidentified)
        intensity_metrics = calculate_intensity_bin_metrics(ground_truth, matched)
        mass_metrics = calculate_mass_accuracy_metrics(matched, diann_results)
        property_metrics = calculate_peptide_property_metrics(ground_truth, matched)

        # Create metrics object
        metrics = create_metrics_from_comparison(
            id_metrics=id_metrics,
            correlation_metrics=correlation_metrics,
            num_ground_truth=len(gt_precursors),
        )

        # Add detailed breakdown metrics
        metrics.charge_state_metrics = charge_metrics
        metrics.intensity_bin_metrics = intensity_metrics
        metrics.mass_accuracy_metrics = mass_metrics
        metrics.peptide_property_metrics = property_metrics

        # Check thresholds
        metrics = check_thresholds(metrics, self.thresholds)

        return metrics, matched

    def _run_both_tools_comparison(self, ground_truth) -> ValidationResult:
        """
        Run multi-tool comparison when both DiaNN and FragPipe are used.

        This generates a comprehensive comparison report showing how both tools
        perform against the simulation ground truth.
        """
        import json
        from datetime import datetime
        from .tool_comparison import compare_tools, ComparisonResult

        # Parse both tool results
        tool_results = {}

        if self._diann_report_path and os.path.exists(self._diann_report_path):
            logger.info("  Loading DiaNN results...")
            tool_results["DIA-NN"] = parse_diann_report(self._diann_report_path)

        if self._fragpipe_result and self._fragpipe_result.psm_path:
            logger.info("  Loading FragPipe results...")
            tool_results["FragPipe"] = parse_fragpipe_combined(
                psm_path=self._fragpipe_result.psm_path,
                peptide_path=self._fragpipe_result.peptide_path,
                protein_path=self._fragpipe_result.protein_path,
                ion_path=self._fragpipe_result.ion_path,
            )

        if not tool_results:
            raise ComparisonError("No tool results available for comparison")

        # Run multi-tool comparison
        logger.info("Step 4/6: Running multi-tool comparison...")
        comparison_result = compare_tools(ground_truth, tool_results)

        # Step 5: Generate comparison plots
        logger.info("Step 5/6: Generating comparison plots...")
        plot_dir = os.path.join(self.output_dir, "plots")
        try:
            comparison_plots = generate_comparison_plots(
                comparison_result, ground_truth, plot_dir
            )
        except Exception as e:
            logger.warning(f"Failed to generate comparison plots: {e}")
            comparison_plots = {}

        # Step 6: Generate reports
        logger.info("Step 6/6: Generating comparison reports...")

        # Generate text report
        text_report = generate_comparison_text_report(comparison_result)
        text_report_path = os.path.join(self.output_dir, "tool_comparison_report.txt")
        with open(text_report_path, 'w') as f:
            f.write(text_report)

        # Generate JSON report
        json_report = {
            "metadata": {
                "tool": "timsim-validate",
                "mode": "multi-tool-comparison",
                "version": "0.4.0",
                "timestamp": datetime.now().isoformat(),
                "paths": {
                    "simulation": self._simulation_path,
                    "database": self._database_path,
                    "diann_report": self._diann_report_path,
                    "fragpipe_output": self._fragpipe_result.output_dir if self._fragpipe_result else None,
                    "fasta": self.fasta_path,
                },
                "plots": comparison_plots,
            },
            "ground_truth": {
                "num_precursors": comparison_result.ground_truth_precursors,
                "num_peptides": comparison_result.ground_truth_peptides,
            },
            "tool_results": {
                name: {
                    "num_psms": tool.num_psms,
                    "num_peptides": tool.num_peptides,
                    "num_precursors": tool.num_precursors,
                    "identification_rate": comparison_result.gt_overlaps[name].identification_rate,
                    "precision": comparison_result.gt_overlaps[name].precision,
                    "true_positives": comparison_result.gt_overlaps[name].both,
                    "false_positives": comparison_result.gt_overlaps[name].tool_only,
                    "false_negatives": comparison_result.gt_overlaps[name].gt_only,
                }
                for name, tool in comparison_result.tool_results.items()
            },
            "pairwise_comparisons": {
                key: {
                    "tool1": comp.tool1_name,
                    "tool2": comp.tool2_name,
                    "common": comp.both,
                    "tool1_only": comp.tool1_only,
                    "tool2_only": comp.tool2_only,
                    "jaccard_index": comp.jaccard_index,
                }
                for key, comp in comparison_result.pairwise.items()
            },
            "correlations": {
                "rt": comparison_result.rt_correlations,
                "im": comparison_result.im_correlations,
            },
            "coverage": {
                "common_to_all_tools": comparison_result.common_to_all,
                "unique_to_ground_truth": comparison_result.unique_to_gt,
            },
            "intensity_breakdown": [
                {
                    "bin": bin_m.bin_index,
                    "intensity_range": bin_m.intensity_range,
                    "log_range": bin_m.log_range,
                    "ground_truth": bin_m.ground_truth_count,
                    "per_tool": bin_m.metrics_per_tool,
                }
                for bin_m in (comparison_result.intensity_breakdown or [])
            ],
            "charge_breakdown": [
                {
                    "charge": charge_m.charge,
                    "ground_truth": charge_m.ground_truth_count,
                    "per_tool": charge_m.metrics_per_tool,
                }
                for charge_m in (comparison_result.charge_breakdown or [])
            ],
        }

        json_report_path = os.path.join(self.output_dir, "tool_comparison_report.json")
        with open(json_report_path, 'w') as f:
            json.dump(json_report, f, indent=2)

        if self.verbose:
            print(text_report)

        # Cleanup temp files if not keeping
        if not self.keep_temp and self._simulation_path:
            self._cleanup_simulation()

        # Determine overall success based on both tools meeting thresholds
        all_pass = all(
            overlap.identification_rate >= self.thresholds.min_identification_rate
            for overlap in comparison_result.gt_overlaps.values()
        )

        return ValidationResult(
            success=all_pass,
            metrics=None,  # Multi-tool mode doesn't use single metrics
            report_path=json_report_path,
            text_report_path=text_report_path,
            plot_paths=None,
            exit_code=0 if all_pass else 1,
        )

    def _cleanup_simulation(self) -> None:
        """Clean up temporary simulation files."""
        if self._simulation_path and os.path.exists(self._simulation_path):
            try:
                shutil.rmtree(self._simulation_path)
                logger.debug(f"Cleaned up simulation directory: {self._simulation_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup simulation directory: {e}")
