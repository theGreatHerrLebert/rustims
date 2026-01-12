"""
DiaNN subprocess execution wrapper for timsim-validate.
"""

import os
import subprocess
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


class DiannError(Exception):
    """Base exception for DiaNN-related errors."""
    pass


class DiannNotFoundError(DiannError):
    """DiaNN executable not found."""
    pass


class DiannTimeoutError(DiannError):
    """DiaNN execution exceeded timeout."""
    pass


class DiannExecutionError(DiannError):
    """DiaNN returned non-zero exit code."""
    pass


@dataclass
class DiannResult:
    """Result of DiaNN execution."""
    report_path: str
    log_path: str
    return_code: int
    stdout: str
    stderr: str
    success: bool

    @property
    def stats_path(self) -> str:
        """Path to the stats file."""
        return self.report_path.replace(".tsv", ".stats.tsv")


@dataclass
class DiannConfig:
    """Configuration options for DiaNN analysis."""
    # Basic settings
    qvalue: float = 0.01
    library_free: bool = True
    use_predictor: bool = True

    # Mass accuracy
    mass_acc: Optional[float] = None  # Fragment mass accuracy in ppm
    mass_acc_ms1: Optional[float] = None  # MS1 mass accuracy in ppm

    # Peptide settings
    min_pep_len: int = 7
    max_pep_len: int = 30
    missed_cleavages: int = 2
    enzyme: str = "K*,R*"  # Trypsin
    met_excision: bool = True

    # Fragment settings
    min_fr_mz: Optional[float] = None
    max_fr_mz: Optional[float] = None

    # Modifications
    fixed_mod: Optional[str] = None  # e.g., "C,carbamidomethyl"
    var_mod: Optional[str] = None  # e.g., "M,oxidation"
    var_mods: int = 2  # Max variable modifications

    # Additional args
    extra_args: Optional[List[str]] = None


class DiannExecutor:
    """Execute DiaNN via subprocess."""

    def __init__(
        self,
        executable_path: str = "diann",
        threads: int = 4,
        timeout_seconds: int = 3600,
        extra_args: Optional[List[str]] = None,
        config: Optional[DiannConfig] = None,
    ):
        """
        Initialize DiaNN executor.

        Args:
            executable_path: Path to the DiaNN executable or just 'diann' if in PATH.
            threads: Number of threads for DiaNN to use.
            timeout_seconds: Maximum execution time in seconds.
            extra_args: Additional command-line arguments to pass to DiaNN.
            config: DiaNN configuration options.
        """
        self.executable_path = executable_path
        self.threads = threads
        self.timeout_seconds = timeout_seconds
        self.extra_args = extra_args or []
        self.config = config or DiannConfig()

    def verify_executable(self) -> bool:
        """
        Check if DiaNN executable exists and is runnable.

        Returns:
            True if DiaNN is available, False otherwise.
        """
        try:
            result = subprocess.run(
                [self.executable_path, "--help"],
                capture_output=True,
                timeout=30,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def build_command(
        self,
        data_path: str,
        fasta_path: str,
        output_path: str,
    ) -> List[str]:
        """
        Build DiaNN command line.

        Args:
            data_path: Path to the .d folder to analyze.
            fasta_path: Path to the FASTA file.
            output_path: Path for the output report.tsv.

        Returns:
            List of command-line arguments.
        """
        cfg = self.config

        cmd = [
            self.executable_path,
            "--f", data_path,
            "--fasta", fasta_path,
            "--out", output_path,
            "--threads", str(self.threads),
            "--qvalue", str(cfg.qvalue),
        ]

        # Library-free mode uses --fasta-search instead of --lib
        if cfg.library_free:
            cmd.append("--fasta-search")

        # Output as TSV (DiaNN 2.3+ defaults to parquet)
        cmd.append("--no-compression")

        # Predictor
        if cfg.use_predictor:
            cmd.append("--predictor")

        # Mass accuracy
        if cfg.mass_acc is not None:
            cmd.extend(["--mass-acc", str(cfg.mass_acc)])
        if cfg.mass_acc_ms1 is not None:
            cmd.extend(["--mass-acc-ms1", str(cfg.mass_acc_ms1)])

        # Peptide settings
        cmd.extend(["--min-pep-len", str(cfg.min_pep_len)])
        cmd.extend(["--max-pep-len", str(cfg.max_pep_len)])
        cmd.extend(["--missed-cleavages", str(cfg.missed_cleavages)])
        cmd.extend(["--cut", cfg.enzyme])

        if cfg.met_excision:
            cmd.append("--met-excision")

        # Fragment m/z range
        if cfg.min_fr_mz is not None:
            cmd.extend(["--min-fr-mz", str(cfg.min_fr_mz)])
        if cfg.max_fr_mz is not None:
            cmd.extend(["--max-fr-mz", str(cfg.max_fr_mz)])

        # Modifications
        if cfg.fixed_mod:
            cmd.extend(["--fixed-mod", cfg.fixed_mod])
        if cfg.var_mod:
            cmd.extend(["--var-mod", cfg.var_mod])
        cmd.extend(["--var-mods", str(cfg.var_mods)])

        # Add any extra arguments from config
        if cfg.extra_args:
            cmd.extend(cfg.extra_args)

        # Add any extra arguments from constructor
        cmd.extend(self.extra_args)

        return cmd

    def execute(
        self,
        data_path: str,
        fasta_path: str,
        output_dir: str,
        report_name: str = "report.tsv",
    ) -> DiannResult:
        """
        Execute DiaNN analysis.

        Args:
            data_path: Path to the .d folder to analyze.
            fasta_path: Path to the FASTA file.
            output_dir: Directory for output files.
            report_name: Name of the output report file.

        Returns:
            DiannResult containing paths and execution status.

        Raises:
            DiannNotFoundError: If DiaNN executable is not found.
            DiannTimeoutError: If DiaNN exceeds the timeout.
            DiannExecutionError: If DiaNN returns non-zero exit code.
        """
        # Verify executable first
        if not self.verify_executable():
            raise DiannNotFoundError(
                f"DiaNN executable not found at '{self.executable_path}'. "
                "Please install DiaNN or specify the path with --diann-path."
            )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Build paths
        output_path = os.path.join(output_dir, report_name)
        log_path = os.path.join(output_dir, "diann.log")

        # Build command
        cmd = self.build_command(
            data_path=data_path,
            fasta_path=fasta_path,
            output_path=output_path,
        )

        logger.info(f"Executing DiaNN: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=output_dir,
            )

            # Write log file
            with open(log_path, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n\n")
                f.write(f"Return code: {result.returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)

            success = result.returncode == 0 and os.path.exists(output_path)

            if not success and result.returncode != 0:
                logger.error(f"DiaNN failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr[:1000]}")  # First 1000 chars
                raise DiannExecutionError(
                    f"DiaNN failed with return code {result.returncode}. "
                    f"Check log at {log_path}"
                )

            if not success and not os.path.exists(output_path):
                raise DiannExecutionError(
                    f"DiaNN completed but output file not found: {output_path}"
                )

            return DiannResult(
                report_path=output_path,
                log_path=log_path,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                success=success,
            )

        except subprocess.TimeoutExpired as e:
            # Write partial log
            with open(log_path, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n\n")
                f.write(f"TIMEOUT after {self.timeout_seconds} seconds\n")
                if e.stdout:
                    f.write("STDOUT:\n")
                    f.write(e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout)
                if e.stderr:
                    f.write("\n\nSTDERR:\n")
                    f.write(e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr)

            raise DiannTimeoutError(
                f"DiaNN exceeded timeout of {self.timeout_seconds} seconds. "
                f"Increase with --diann-timeout. Log at {log_path}"
            )
