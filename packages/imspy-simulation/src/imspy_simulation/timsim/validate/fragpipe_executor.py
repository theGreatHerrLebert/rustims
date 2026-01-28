"""
FragPipe subprocess execution wrapper for timsim-validate.
"""

import os
import subprocess
import logging
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class FragPipeError(Exception):
    """Base exception for FragPipe-related errors."""
    pass


class FragPipeNotFoundError(FragPipeError):
    """FragPipe executable not found."""
    pass


class FragPipeTimeoutError(FragPipeError):
    """FragPipe execution exceeded timeout."""
    pass


class FragPipeExecutionError(FragPipeError):
    """FragPipe returned non-zero exit code."""
    pass


@dataclass
class FragPipeResult:
    """Result of FragPipe execution."""
    output_dir: str
    psm_path: str
    peptide_path: str
    protein_path: str
    ion_path: Optional[str]
    log_path: str
    return_code: int
    stdout: str
    stderr: str
    success: bool
    partial_results: bool = False  # True if pipeline failed but ident results available

    @property
    def diann_report_path(self) -> Optional[str]:
        """Path to DIA-NN report if available."""
        diann_dir = os.path.join(self.output_dir, "dia-quant-output")
        report_tsv = os.path.join(diann_dir, "report.tsv")
        if os.path.exists(report_tsv):
            return report_tsv
        report_parquet = os.path.join(diann_dir, "report.parquet")
        if os.path.exists(report_parquet):
            return report_parquet
        return None

    @property
    def msbooster_dir(self) -> Optional[str]:
        """Path to MSBooster output directory."""
        msbooster_dir = os.path.join(self.output_dir, "MSBooster")
        if os.path.exists(msbooster_dir):
            return msbooster_dir
        return None


@dataclass
class FragPipeConfig:
    """Configuration options for FragPipe analysis."""
    # Required paths
    workflow_path: Optional[str] = None  # Path to .workflow file
    tools_folder: Optional[str] = None  # FragPipe tools folder
    diann_path: Optional[str] = None  # DIA-NN executable
    python_path: Optional[str] = None  # Python executable

    # Optional workflow parameters to override
    database_path: Optional[str] = None  # FASTA database path
    qvalue: float = 0.01
    threads: int = 8
    ram: int = 0  # 0 = auto

    # Pipeline options
    run_msbooster: bool = True
    run_diann: bool = True
    run_ionquant: bool = True


class FragPipeExecutor:
    """Execute FragPipe via subprocess in headless mode."""

    def __init__(
        self,
        executable_path: str = "fragpipe",
        threads: int = 8,
        ram: int = 0,
        timeout_seconds: int = 7200,  # 2 hours default
        config: Optional[FragPipeConfig] = None,
    ):
        """
        Initialize FragPipe executor.

        Args:
            executable_path: Path to the FragPipe executable or just 'fragpipe' if in PATH.
            threads: Number of threads for FragPipe to use.
            ram: RAM limit in GB (0 = auto).
            timeout_seconds: Maximum execution time in seconds.
            config: FragPipe configuration options.
        """
        self.executable_path = executable_path
        self.threads = threads
        self.ram = ram
        self.timeout_seconds = timeout_seconds
        self.config = config or FragPipeConfig()
        self._version: Optional[str] = None

    def get_version(self) -> str:
        """
        Get FragPipe version string.

        Extracts version from executable path or directory structure.

        Returns:
            Version string (e.g., "24.0") or "unknown" if unavailable.
        """
        if self._version is not None:
            return self._version

        import re

        # Try to extract from executable path (e.g., fragpipe-24.0/bin/fragpipe)
        path_match = re.search(r'fragpipe[_-]?(\d+\.\d+(?:\.\d+)?)', self.executable_path)
        if path_match:
            self._version = path_match.group(1)
            return self._version

        # Try to find version in parent directories
        try:
            path = Path(self.executable_path).resolve()
            for parent in path.parents:
                match = re.search(r'fragpipe[_-]?(\d+\.\d+(?:\.\d+)?)', parent.name)
                if match:
                    self._version = match.group(1)
                    return self._version
        except Exception:
            pass

        self._version = "unknown"
        return self._version

    def verify_executable(self) -> bool:
        """
        Check if FragPipe executable exists and is runnable.

        Returns:
            True if FragPipe is available, False otherwise.
        """
        try:
            result = subprocess.run(
                [self.executable_path, "--help"],
                capture_output=True,
                timeout=30,
                text=True,
            )
            # FragPipe returns exit code 1 for --help, so check output instead
            return "FragPipe" in result.stdout or "FragPipe" in result.stderr
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def _create_manifest(
        self,
        data_path: str,
        manifest_path: str,
        is_dda: bool = False,
        additional_data_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Create a FragPipe manifest file for the input data.

        Args:
            data_path: Path to the .d folder to analyze.
            manifest_path: Path to write the manifest file.
            is_dda: If True, mark as DDA data type; otherwise DIA.
            additional_data_paths: Additional .d folders for multi-sample analysis.
        """
        data_type = "DDA" if is_dda else "DIA"

        # Collect all data paths
        all_paths = [data_path]
        if additional_data_paths:
            all_paths.extend(additional_data_paths)

        with open(manifest_path, 'w') as f:
            for i, path in enumerate(all_paths):
                # Extract experiment name from .d folder
                exp_name = os.path.basename(path).replace(".d", "")

                if is_dda:
                    # DDA with LFQ-MBR requires experiment annotation
                    # Format: path\texperiment\tbiorep\ttechrep\tdata_type
                    f.write(f"{path}\t{exp_name}\t{i+1}\t1\t{data_type}\n")
                else:
                    # DIA format: empty experiment fields for diaTracer pipeline
                    # Format: path\t\t\tdata_type (no experiment annotation)
                    f.write(f"{path}\t\t\t{data_type}\n")

    def _update_workflow(
        self,
        base_workflow_path: str,
        output_workflow_path: str,
        fasta_path: str,
        is_timstof: bool = True,
    ) -> None:
        """
        Update workflow file with new FASTA path and other settings.

        Args:
            base_workflow_path: Path to the base workflow file.
            output_workflow_path: Path to write the updated workflow file.
            fasta_path: Path to the FASTA database.
            is_timstof: If True, configure for timsTOF ion mobility data.
        """
        with open(base_workflow_path, 'r') as f:
            workflow_content = f.read()

        # Update FASTA path and other settings
        lines = workflow_content.split('\n')
        updated_lines = []
        found_db_path = False
        found_im_ms = False
        found_regular_ms = False

        for line in lines:
            if line.startswith('database.db-path='):
                updated_lines.append(f'database.db-path={fasta_path}')
                found_db_path = True
            elif is_timstof and line.startswith('workflow.input.data-type.im-ms='):
                updated_lines.append('workflow.input.data-type.im-ms=true')
                found_im_ms = True
            elif is_timstof and line.startswith('workflow.input.data-type.regular-ms='):
                updated_lines.append('workflow.input.data-type.regular-ms=false')
                found_regular_ms = True
            else:
                updated_lines.append(line)

        # Add database.db-path if it wasn't found
        if not found_db_path:
            # Insert after header comment if present, otherwise at the beginning
            insert_idx = 0
            for i, line in enumerate(updated_lines):
                if line.startswith('#'):
                    insert_idx = i + 1
                else:
                    break
            updated_lines.insert(insert_idx, f'database.db-path={fasta_path}')

        with open(output_workflow_path, 'w') as f:
            f.write('\n'.join(updated_lines))

    def _find_output_file(self, output_dir: str, filename: str) -> Optional[str]:
        """
        Find an output file in the output directory or its subdirectories.

        FragPipe DDA puts results in experiment subdirectories, while DIA
        puts them at the root. This method searches both locations.

        Args:
            output_dir: Base output directory.
            filename: Name of the file to find (e.g., "psm.tsv").

        Returns:
            Path to the file if found, None otherwise.
        """
        # First check root directory
        root_path = os.path.join(output_dir, filename)
        if os.path.exists(root_path):
            return root_path

        # Check for combined_* variant at root (DDA combined output)
        combined_name = f"combined_{filename}"
        combined_path = os.path.join(output_dir, combined_name)
        if os.path.exists(combined_path):
            return combined_path

        # Search in subdirectories (DDA experiment directories)
        for entry in os.listdir(output_dir):
            subdir = os.path.join(output_dir, entry)
            if os.path.isdir(subdir) and not entry.startswith('.'):
                sub_path = os.path.join(subdir, filename)
                if os.path.exists(sub_path):
                    return sub_path

        return None

    def build_command(
        self,
        workflow_path: str,
        manifest_path: str,
        output_dir: str,
    ) -> List[str]:
        """
        Build FragPipe command line.

        Args:
            workflow_path: Path to the workflow file.
            manifest_path: Path to the manifest file.
            output_dir: Path for output directory.

        Returns:
            List of command-line arguments.
        """
        cmd = [
            self.executable_path,
            "--headless",
            "--workflow", workflow_path,
            "--manifest", manifest_path,
            "--workdir", output_dir,
        ]

        # Only add threads if positive (FragPipe uses core-1 as default)
        if self.threads > 0:
            cmd.extend(["--threads", str(self.threads)])

        if self.ram > 0:
            cmd.extend(["--ram", str(self.ram)])

        # Add tool paths if configured
        if self.config.tools_folder:
            cmd.extend(["--config-tools-folder", self.config.tools_folder])

        if self.config.diann_path:
            cmd.extend(["--config-diann", self.config.diann_path])

        if self.config.python_path:
            cmd.extend(["--config-python", self.config.python_path])

        return cmd

    def execute(
        self,
        data_path: str,
        fasta_path: str,
        output_dir: str,
        workflow_path: Optional[str] = None,
        is_dda: bool = False,
        additional_data_paths: Optional[List[str]] = None,
    ) -> FragPipeResult:
        """
        Execute FragPipe analysis.

        Args:
            data_path: Path to the .d folder to analyze.
            fasta_path: Path to the FASTA file.
            output_dir: Directory for output files.
            workflow_path: Path to workflow file (uses config if not specified).
            is_dda: If True, configure for DDA data; otherwise DIA.
            additional_data_paths: Additional .d folders for multi-sample analysis.

        Returns:
            FragPipeResult containing paths and execution status.

        Raises:
            FragPipeNotFoundError: If FragPipe executable is not found.
            FragPipeTimeoutError: If FragPipe exceeds the timeout.
            FragPipeExecutionError: If FragPipe returns non-zero exit code.
        """
        # Verify executable first
        if not self.verify_executable():
            raise FragPipeNotFoundError(
                f"FragPipe executable not found at '{self.executable_path}'. "
                "Please install FragPipe or specify the path with --fragpipe-path."
            )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Use workflow from config if not specified
        workflow_to_use = workflow_path or self.config.workflow_path
        if not workflow_to_use:
            raise FragPipeError(
                "No workflow file specified. Provide via --fragpipe-workflow or config."
            )

        if not os.path.exists(workflow_to_use):
            raise FragPipeError(f"Workflow file not found: {workflow_to_use}")

        # Create manifest
        manifest_path = os.path.join(output_dir, "manifest.fp-manifest")
        self._create_manifest(data_path, manifest_path, is_dda=is_dda, additional_data_paths=additional_data_paths)

        # Create updated workflow with correct FASTA path
        updated_workflow_path = os.path.join(output_dir, "workflow.workflow")
        self._update_workflow(workflow_to_use, updated_workflow_path, fasta_path)

        # Build command
        cmd = self.build_command(
            workflow_path=updated_workflow_path,
            manifest_path=manifest_path,
            output_dir=output_dir,
        )

        log_path = os.path.join(output_dir, "fragpipe_runner.log")

        logger.info(f"Executing FragPipe: {' '.join(cmd)}")

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

            # Check for expected output files
            # For DDA, files may be in experiment subdirectories
            psm_path = self._find_output_file(output_dir, "psm.tsv")
            peptide_path = self._find_output_file(output_dir, "peptide.tsv")
            protein_path = self._find_output_file(output_dir, "protein.tsv")
            ion_path = self._find_output_file(output_dir, "ion.tsv")

            if result.returncode != 0:
                logger.error(f"FragPipe failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr[:1000]}")

                # Check for partial results (identification completed but quant/speclib failed)
                if psm_path:
                    logger.warning(
                        f"FragPipe pipeline failed but identification results found. "
                        f"Using partial results (ident-only mode)."
                    )
                    return FragPipeResult(
                        output_dir=output_dir,
                        psm_path=psm_path,
                        peptide_path=peptide_path or "",
                        protein_path=protein_path or "",
                        ion_path=ion_path,
                        log_path=log_path,
                        return_code=result.returncode,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        success=False,
                        partial_results=True,
                    )
                else:
                    raise FragPipeExecutionError(
                        f"FragPipe failed with return code {result.returncode}. "
                        f"Check log at {log_path}"
                    )

            if not psm_path:
                raise FragPipeExecutionError(
                    f"FragPipe completed but PSM file not found in: {output_dir}"
                )

            return FragPipeResult(
                output_dir=output_dir,
                psm_path=psm_path,
                peptide_path=peptide_path or "",
                protein_path=protein_path or "",
                ion_path=ion_path,
                log_path=log_path,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                success=True,
                partial_results=False,
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

            raise FragPipeTimeoutError(
                f"FragPipe exceeded timeout of {self.timeout_seconds} seconds. "
                f"Increase with --fragpipe-timeout. Log at {log_path}"
            )

    def execute_from_existing_output(self, output_dir: str) -> FragPipeResult:
        """
        Create a FragPipeResult from an existing FragPipe output directory.

        This is useful when reusing previous FragPipe runs for validation.

        Args:
            output_dir: Path to existing FragPipe output directory.

        Returns:
            FragPipeResult with paths to output files.

        Raises:
            FragPipeError: If required output files are not found.
        """
        # Search for files in root or subdirectories
        psm_path = self._find_output_file(output_dir, "psm.tsv")
        peptide_path = self._find_output_file(output_dir, "peptide.tsv")
        protein_path = self._find_output_file(output_dir, "protein.tsv")
        ion_path = self._find_output_file(output_dir, "ion.tsv")

        if not psm_path:
            raise FragPipeError(f"PSM file not found in: {output_dir}")

        # Find log file
        log_path = os.path.join(output_dir, "fragpipe_runner.log")
        if not os.path.exists(log_path):
            log_path = os.path.join(output_dir, "fragpipe.log")

        return FragPipeResult(
            output_dir=output_dir,
            psm_path=psm_path,
            peptide_path=peptide_path or "",
            protein_path=protein_path or "",
            ion_path=ion_path,
            log_path=log_path,
            return_code=0,
            stdout="",
            stderr="",
            success=True,
        )
