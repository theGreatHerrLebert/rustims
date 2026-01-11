"""
Main orchestration logic for timsim-validate.
"""

import os
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

from .diann_executor import DiannExecutor, DiannError
from .parsing import parse_diann_report
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

logger = logging.getLogger(__name__)


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
        thresholds: Optional[ValidationThresholds] = None,
        simulation_config: Optional[SimulationConfig] = None,
        keep_temp: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize validation runner.

        Args:
            reference_path: Path to reference .d folder for instrument layout.
            output_dir: Output directory for results.
            fasta_path: Path to FASTA file (None = use bundled test FASTA).
            diann_executor: DiaNN executor instance.
            thresholds: Validation thresholds.
            simulation_config: Simulation configuration.
            keep_temp: Keep temporary simulation files.
            verbose: Verbose output.
        """
        self.reference_path = reference_path
        self.output_dir = output_dir
        self.fasta_path = fasta_path or self._get_bundled_fasta()
        self.diann_executor = diann_executor or DiannExecutor()
        self.thresholds = thresholds or ValidationThresholds()
        self.simulation_config = simulation_config or SimulationConfig()
        self.keep_temp = keep_temp
        self.verbose = verbose

        # Will be set during run
        self._simulation_path: Optional[str] = None
        self._database_path: Optional[str] = None
        self._diann_report_path: Optional[str] = None

    def _get_bundled_fasta(self) -> str:
        """Get path to bundled test FASTA."""
        # Look for bundled FASTA in resources directory
        resources_dir = Path(__file__).parent / "resources"
        bundled_fasta = resources_dir / "test_proteome.fasta"

        if bundled_fasta.exists():
            return str(bundled_fasta)

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
            # Step 1: Run simulation
            logger.info("Step 1/5: Running timsim simulation...")
            self._run_simulation()

            # Step 2: Execute DiaNN
            logger.info("Step 2/5: Executing DiaNN analysis...")
            self._run_diann()

            # Step 3: Load ground truth
            logger.info("Step 3/5: Loading ground truth data...")
            ground_truth = load_ground_truth(self._database_path)

            # Step 4: Parse DiaNN results and compare
            logger.info("Step 4/5: Comparing results...")
            diann_results = parse_diann_report(self._diann_report_path)
            metrics = self._compute_metrics(ground_truth, diann_results)

            # Step 5: Generate reports
            logger.info("Step 5/5: Generating reports...")
            report_path = generate_json_report(
                metrics=metrics,
                thresholds=self.thresholds,
                output_dir=self.output_dir,
                simulation_path=self._simulation_path,
                diann_report_path=self._diann_report_path,
                fasta_path=self.fasta_path,
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
                exit_code=exit_code,
            )

        except DiannError as e:
            logger.error(f"DiaNN error: {e}")
            return ValidationResult.from_error(str(e), exit_code=3)
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

    def _compute_metrics(
        self,
        ground_truth,
        diann_results,
    ) -> ValidationMetrics:
        """Compute validation metrics from comparison."""
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

        # Create metrics object
        metrics = create_metrics_from_comparison(
            id_metrics=id_metrics,
            correlation_metrics=correlation_metrics,
            num_ground_truth=len(gt_precursors),
        )

        # Check thresholds
        metrics = check_thresholds(metrics, self.thresholds)

        return metrics

    def _cleanup_simulation(self) -> None:
        """Clean up temporary simulation files."""
        if self._simulation_path and os.path.exists(self._simulation_path):
            try:
                shutil.rmtree(self._simulation_path)
                logger.debug(f"Cleaned up simulation directory: {self._simulation_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup simulation directory: {e}")
