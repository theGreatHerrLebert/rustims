"""
Sage executor for timsim validation.

Runs Sage search engine on timsTOF DDA data.
"""

import json
import logging
import os
import subprocess
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class SageConfig:
    """Configuration for Sage search."""
    # Database settings
    fasta_path: str
    missed_cleavages: int = 2
    min_peptide_len: int = 7
    max_peptide_len: int = 50
    min_peptide_mass: float = 500.0
    max_peptide_mass: float = 5000.0

    # Static modifications (Carbamidomethyl C by default)
    static_mods: Dict[str, float] = None
    variable_mods: Dict[str, List[Any]] = None

    # Tolerances
    precursor_tol_ppm: float = 20.0
    fragment_tol_ppm: float = 20.0

    # Charge states
    min_charge: int = 2
    max_charge: int = 4

    # Search settings
    isotope_errors: List[int] = None
    min_peaks: int = 15
    max_peaks: int = 150
    min_matched_peaks: int = 4
    max_fragment_charge: int = 1

    # Decoy settings
    decoy_tag: str = "rev_"
    generate_decoys: bool = False  # Assume FASTA already has decoys

    # Output settings
    report_psms: int = 1
    write_pin: bool = False

    def __post_init__(self):
        if self.static_mods is None:
            self.static_mods = {"C": 57.021464}
        if self.variable_mods is None:
            self.variable_mods = {}
        if self.isotope_errors is None:
            self.isotope_errors = [-1, 3]

    def to_json(self, d_folder: str, output_dir: str) -> Dict:
        """Convert config to Sage JSON format."""
        return {
            "database": {
                "bucket_size": 32768,
                "enzyme": {
                    "missed_cleavages": self.missed_cleavages,
                    "min_len": self.min_peptide_len,
                    "max_len": self.max_peptide_len,
                    "cleave_at": "KR",
                    "restrict": "P",
                },
                "fragment_min_mz": 150.0,
                "fragment_max_mz": 2000.0,
                "peptide_min_mass": self.min_peptide_mass,
                "peptide_max_mass": self.max_peptide_mass,
                "ion_kinds": ["b", "y"],
                "min_ion_index": 2,
                "static_mods": self.static_mods,
                "variable_mods": self.variable_mods,
                "decoy_tag": self.decoy_tag,
                "generate_decoys": self.generate_decoys,
                "fasta": self.fasta_path,
            },
            "precursor_tol": {
                "ppm": [-self.precursor_tol_ppm, self.precursor_tol_ppm]
            },
            "fragment_tol": {
                "ppm": [-self.fragment_tol_ppm, self.fragment_tol_ppm]
            },
            "precursor_charge": [self.min_charge, self.max_charge],
            "isotope_errors": self.isotope_errors,
            "deisotope": True,
            "chimera": False,
            "wide_window": False,
            "predict_rt": False,
            "min_peaks": self.min_peaks,
            "max_peaks": self.max_peaks,
            "min_matched_peaks": self.min_matched_peaks,
            "max_fragment_charge": self.max_fragment_charge,
            "report_psms": self.report_psms,
            "mzml_paths": [d_folder],
        }


@dataclass
class SageResult:
    """Result of Sage execution."""
    output_dir: str
    results_path: str  # Path to results.sage.tsv
    config_path: str  # Path to config JSON used
    log_path: str
    success: bool
    psm_count: int = 0
    peptide_count: int = 0
    version: str = ""


def find_sage_executable() -> Optional[str]:
    """Find Sage executable in PATH or common locations."""
    # Check PATH first
    sage_path = shutil.which("sage")
    if sage_path:
        return sage_path

    # Check common locations
    common_paths = [
        os.path.expanduser("~/rust/sage/target/release/sage"),
        "/usr/local/bin/sage",
        "/opt/sage/sage",
    ]

    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def get_sage_version(sage_path: str) -> str:
    """Get Sage version string."""
    try:
        result = subprocess.run(
            [sage_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Output format: "sage X.Y.Z" or "sage X.Y.Z-beta.N"
        version_line = result.stdout.strip()
        if version_line.startswith("sage "):
            return version_line.split()[1]
        return version_line
    except Exception:
        return "unknown"


def run_sage(
    d_folder: str,
    fasta_path: str,
    output_dir: str,
    config: Optional[SageConfig] = None,
    sage_path: Optional[str] = None,
) -> SageResult:
    """
    Run Sage search on a timsTOF .d folder.

    Args:
        d_folder: Path to the .d folder containing TDF data.
        fasta_path: Path to FASTA file (should include decoys).
        output_dir: Directory to write Sage output.
        config: Optional SageConfig. If None, uses defaults.
        sage_path: Optional path to Sage executable. If None, searches PATH.

    Returns:
        SageResult with paths to output files and execution status.
    """
    # Find Sage
    if sage_path is None:
        sage_path = find_sage_executable()

    if sage_path is None:
        logger.error("Sage executable not found. Install Sage or add to PATH.")
        return SageResult(
            output_dir=output_dir,
            results_path="",
            config_path="",
            log_path="",
            success=False,
        )

    # Get version
    version = get_sage_version(sage_path)
    logger.info(f"Using Sage {version} at {sage_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create config
    if config is None:
        config = SageConfig(fasta_path=fasta_path)
    else:
        config.fasta_path = fasta_path

    # Write config JSON
    config_path = os.path.join(output_dir, "sage_config.json")
    config_dict = config.to_json(d_folder, output_dir)

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Sage config written to {config_path}")

    # Prepare log file
    log_path = os.path.join(output_dir, "sage_runner.log")

    # Run Sage
    cmd = [
        sage_path,
        config_path,
        "-o", output_dir,
    ]

    logger.info(f"Running Sage: {' '.join(cmd)}")

    try:
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1 hour timeout
                cwd=output_dir,
            )

        # Check for output file
        results_path = os.path.join(output_dir, "results.sage.tsv")

        if result.returncode != 0:
            logger.error(f"Sage failed with return code {result.returncode}")
            logger.error(f"Check log file: {log_path}")
            return SageResult(
                output_dir=output_dir,
                results_path=results_path if os.path.exists(results_path) else "",
                config_path=config_path,
                log_path=log_path,
                success=False,
                version=version,
            )

        if not os.path.exists(results_path):
            logger.error(f"Sage completed but results file not found: {results_path}")
            return SageResult(
                output_dir=output_dir,
                results_path="",
                config_path=config_path,
                log_path=log_path,
                success=False,
                version=version,
            )

        # Count results
        psm_count = 0
        peptide_count = 0
        try:
            import pandas as pd
            df = pd.read_csv(results_path, sep="\t")
            # Filter to 1% FDR
            df_filtered = df[df["spectrum_q"] <= 0.01]
            psm_count = len(df_filtered)
            peptide_count = df_filtered["peptide"].nunique()
        except Exception as e:
            logger.warning(f"Could not count Sage results: {e}")

        logger.info(f"Sage completed: {psm_count} PSMs, {peptide_count} peptides at 1% FDR")

        return SageResult(
            output_dir=output_dir,
            results_path=results_path,
            config_path=config_path,
            log_path=log_path,
            success=True,
            psm_count=psm_count,
            peptide_count=peptide_count,
            version=version,
        )

    except subprocess.TimeoutExpired:
        logger.error("Sage timed out after 1 hour")
        return SageResult(
            output_dir=output_dir,
            results_path="",
            config_path=config_path,
            log_path=log_path,
            success=False,
            version=version,
        )

    except Exception as e:
        logger.error(f"Sage execution failed: {e}")
        return SageResult(
            output_dir=output_dir,
            results_path="",
            config_path=config_path,
            log_path=log_path,
            success=False,
            version=version,
        )
