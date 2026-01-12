import os
import sys

# Silence verbose package outputs before importing them
os.environ["WANDB_SILENT"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import platform
import argparse
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional

import toml
import pandas as pd
from tabulate import tabulate

try:
    from importlib.metadata import version as get_version
    __version__ = get_version("imspy")
except Exception:
    __version__ = "0.3.23"  # Fallback


@dataclass
class SimulationTimer:
    """Track timing for simulation steps."""
    start_time: float = field(default_factory=time.time)
    step_times: Dict[str, float] = field(default_factory=dict)
    _current_step: Optional[str] = None
    _step_start: float = 0.0

    def start_step(self, name: str) -> None:
        """Start timing a step."""
        self._current_step = name
        self._step_start = time.time()

    def end_step(self) -> None:
        """End timing the current step."""
        if self._current_step:
            self.step_times[self._current_step] = time.time() - self._step_start
            self._current_step = None

    def total_elapsed(self) -> float:
        """Get total elapsed time."""
        return time.time() - self.start_time

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format seconds as human-readable duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


@dataclass
class SimulationStats:
    """Track statistics for simulation summary."""
    n_proteins: int = 0
    n_peptides: int = 0
    n_ions: int = 0
    n_frames: int = 0
    acquisition_type: str = ""
    experiment_name: str = ""
    output_path: str = ""

# imspy imports
from imspy.simulation.experiment import SyntheticExperimentDataHandleDIA, SyntheticExperimentDataHandle
from imspy.simulation.timsim.jobs.simulate_ion_mobilities_and_variance import simulate_ion_mobilities_and_variance
from imspy.simulation.timsim.jobs.simulate_peptides import simulate_peptides
from imspy.simulation.timsim.jobs.simulate_phosphorylation import simulate_phosphorylation
from imspy.simulation.timsim.jobs.simulate_proteins import simulate_proteins
from imspy.simulation.timsim.jobs.simulate_scan_distributions_with_variance import (
    simulate_scan_distributions_with_variance
)
from imspy.simulation.utility import get_fasta_file_paths, get_dilution_factors
from imspy.simulation.timsim.jobs.utility import check_path

from .jobs.utility import add_log_noise_variation, add_normal_noise

# Local imports
from .jobs.assemble_frames import assemble_frames
from .jobs.build_acquisition import build_acquisition
from .jobs.simulate_charge_states import simulate_charge_states
from .jobs.simulate_fragment_intensities import simulate_fragment_intensities
from .jobs.simulate_frame_distributions_emg import simulate_frame_distributions_emg
from .jobs.simulate_precursor_spectra import simulate_precursor_spectra_sequence
from .jobs.simulate_retention_time import simulate_retention_times
from .jobs.dda_selection_scheme import simulate_dda_pasef_selection_scheme

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Configure logging for the timsim application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs to console only.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    handlers.append(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        handlers.append(file_handler)

    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)


# ----------------------------------------------------------------------
# Environment setup and GPU configuration
# ----------------------------------------------------------------------

def configure_gpu_memory(memory_limit_gb: int = 4, use_gpu: bool = True) -> None:
    """
    Configures GPU usage for TensorFlow.

    Args:
        memory_limit_gb: Memory limit in gigabytes per GPU.
        use_gpu: If False, disables GPU and forces CPU-only mode.
    """
    # Must set CUDA_VISIBLE_DEVICES before importing TensorFlow
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("  GPU disabled via config, using CPU only")

    import tensorflow as tf

    if not use_gpu:
        return

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for i, _ in enumerate(gpus):
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[i],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * memory_limit_gb)]
                )
                logger.info(f"  GPU {i}: memory restricted to {memory_limit_gb}GB")
        except RuntimeError as e:
            logger.error(f"  GPU configuration error: {e}")
    else:
        logger.info("  No GPU detected, using CPU")


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

def load_toml_config(config_path: str) -> dict:
    """
    Loads a TOML configuration file into a dictionary.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        toml.TomlDecodeError: If the config file is invalid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as config_file:
        return toml.load(config_file)


def get_default_settings() -> dict:
    """
    Returns a dictionary of default values for all configuration options.

    Returns:
        Dictionary with default configuration values.
    """
    return {
        # Main settings
        'save_path': None,
        'reference_path': None,
        'fasta_path': None,
        'reference_in_memory': False,
        'silent_mode': False,
        'acquisition_type': 'DIA',
        'experiment_name': f'TIMSIM-[PLACEHOLDER]-{int(time.time())}',
        'use_reference_layout': True,
        'sample_peptides': True,
        'sample_seed': 41,
        'apply_fragmentation': False,
        'from_existing': False,
        'existing_path': None,
        'use_bruker_sdk': True,

        # Peptide digestion
        'num_sample_peptides': 25_000,
        'num_peptides_total': 250_000,
        'n_proteins': 20_000,
        'missed_cleavages': 2,
        'min_len': 7,
        'max_len': 30,
        'cleave_at': 'KR',
        'restrict': 'P',
        'decoys': False,
        'digest_proteins': True,
        'remove_degenerate_peptides': False,
        'upscale_factor': 100_000,
        'min_rt_percent': 2.0,
        'exclude_accumulated_gradient_start': True,

        # Modifications
        'modifications': None,

        # Isotopic pattern
        'isotope_k': 8,
        'isotope_min_intensity': 1,
        'isotope_centroid': True,

        # Occurrences
        'sample_occurrences': True,

        # Distribution settings
        'gradient_length': 3600,
        'koina_rt_model': None,
        'sigma_lower_rt': None,
        'sigma_upper_rt': None,
        'sigma_alpha_rt': 4,
        'sigma_beta_rt': 4,
        'k_lower_rt': 0,
        'k_upper_rt': 10,
        'k_alpha_rt': 1,
        'k_beta_rt': 20,
        'target_p': 0.999,
        'sampling_step_size': 0.001,
        'n_steps': 1000,
        'remove_epsilon': 1e-4,
        'use_inverse_mobility_std_mean': True,
        'inverse_mobility_std_mean': 0.009,

        # Acquisition settings
        'round_collision_energy': True,
        'collision_energy_decimals': 0,

        # Variation settings
        're_scale_rt': False,
        'rt_variation_std': None,
        'ion_mobility_variation_std': None,
        'intensity_variation_std': None,

        # Performance settings
        'num_threads': -1,
        'batch_size': 256,

        # Charge state settings
        'p_charge': 0.8,
        'min_charge_contrib': 0.005,
        'max_charge': 4,
        'binomial_charge_model': False,
        'normalize_charge_states': True,
        'charge_state_one_probability': 0.0,

        # Noise settings
        'noise_frame_abundance': False,
        'noise_scan_abundance': False,
        'mz_noise_precursor': False,
        'precursor_noise_ppm': 5.0,
        'mz_noise_fragment': False,
        'fragment_noise_ppm': 5.0,
        'mz_noise_uniform': False,
        'add_real_data_noise': False,
        'reference_noise_intensity_max': 30,
        'down_sample_factor': 0.5,

        # Fragment intensity model
        'fragment_intensity_model': None,  # None/"prosit" or "peptdeep"

        # Proteome mixture
        'proteome_mix': False,
        'multi_fasta_dilution': None,

        # Phosphorylation
        'phospho_mode': False,

        # Frame assembly
        'lazy_frame_assembly': False,
        'frame_batch_size': 500,

        # Real data noise settings
        'precursor_sample_fraction': 0.2,
        'fragment_sample_fraction': 0.2,
        'num_precursor_noise_frames': 5,
        'num_fragment_noise_frames': 5,

        # GPU settings
        'use_gpu': True,
        'gpu_memory_limit_gb': 4,

        # DDA settings
        'precursors_every': 10,
        'precursor_intensity_threshold': 500,
        'max_precursors': 25,
        'exclusion_width': 25,
        'selection_mode': 'topN',

        # Quad-dependent isotope transmission (DDA and DIA)
        # Mode options: "none", "precursor_scaling", "per_fragment"
        # - "none": Standard isotope patterns (default)
        # - "precursor_scaling": Fast mode - uniform scaling based on precursor transmission
        # - "per_fragment": Accurate mode - individual fragment ion recalculation
        'quad_isotope_transmission_mode': 'none',
        'quad_transmission_min_probability': 0.5,
        'quad_transmission_max_isotopes': 10,

        # Logging settings
        'log_level': 'INFO',
        'log_file': None,
    }


class SimulationConfig:
    """
    Configuration container for timsim simulation parameters.

    Loads configuration from a TOML file and provides attribute-style access
    to all settings with defaults applied.
    """

    def __init__(self, config_path: str):
        """
        Initialize configuration from a TOML file.

        Args:
            config_path: Path to the TOML configuration file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If required settings are missing or invalid.
        """
        self._config = get_default_settings()

        # Load TOML config and flatten sections
        raw_config = load_toml_config(config_path)
        for section_key, section_value in raw_config.items():
            if isinstance(section_value, dict):
                self._config.update(section_value)
            else:
                self._config[section_key] = section_value

        # Validate required settings
        self._validate()

    def _validate(self) -> None:
        """Validate that required configuration options are set."""
        required = ['save_path', 'reference_path', 'fasta_path']
        missing = [key for key in required if not self._config.get(key)]

        if missing:
            raise ValueError(f"Missing required configuration options: {', '.join(missing)}")

        # Validate sigma bounds
        sigma_lower = self._config.get('sigma_lower_rt')
        sigma_upper = self._config.get('sigma_upper_rt')
        if sigma_lower is not None and sigma_upper is not None:
            if sigma_lower >= sigma_upper:
                raise ValueError("sigma_lower_rt must be less than sigma_upper_rt")

        # Validate k bounds
        k_lower = self._config.get('k_lower_rt', 0)
        k_upper = self._config.get('k_upper_rt', 10)
        if k_lower >= k_upper:
            raise ValueError("k_lower_rt must be less than k_upper_rt")

        # Validate p_charge
        p_charge = self._config.get('p_charge', 0.8)
        if not (0.0 < p_charge < 1.0):
            raise ValueError(f"p_charge must be between 0 and 1, got {p_charge}")

    def __getattr__(self, name: str):
        """Allow attribute-style access to configuration values."""
        if name.startswith('_'):
            return super().__getattribute__(name)
        try:
            return self._config[name]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{name}'")

    def to_dict(self) -> dict:
        """Return configuration as a dictionary."""
        return self._config.copy()

    def __repr__(self) -> str:
        return f"SimulationConfig({self._config})"


def banner(use_unicode: bool = True) -> str:
    """Return the application banner."""
    if use_unicode:
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🦀💻  TIMSIM  🔬🐍                                                         ║
║                                                                              ║
║   Proteomics Simulation Engine for timsTOF                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    else:
        return """
+------------------------------------------------------------------------------+
|                                                                              |
|   TIMSIM - Proteomics Simulation Engine for timsTOF                          |
|                                                                              |
+------------------------------------------------------------------------------+
"""


def section_header(title: str, use_unicode: bool = True) -> str:
    """Return a formatted section header."""
    width = 78
    if use_unicode:
        return f"\n{'─' * width}\n  ▶ {title}\n{'─' * width}"
    else:
        return f"\n{'-' * width}\n  > {title}\n{'-' * width}"


def subsection_header(title: str) -> str:
    """Return a formatted subsection header."""
    return f"\n  ┌─ {title}"


def simulation_complete_banner(use_unicode: bool = True) -> str:
    """Return the simulation complete banner."""
    if use_unicode:
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ✅  SIMULATION COMPLETED SUCCESSFULLY                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    else:
        return """
+------------------------------------------------------------------------------+
|                                                                              |
|   SIMULATION COMPLETED SUCCESSFULLY                                          |
|                                                                              |
+------------------------------------------------------------------------------+
"""


def reorder_peptide_columns(peptides: pd.DataFrame, phospho_mode: bool = False,
                            proteome_mix: bool = False) -> pd.DataFrame:
    """
    Reorder peptide DataFrame columns based on simulation mode.

    Args:
        peptides: DataFrame containing peptide data.
        phospho_mode: Whether phosphorylation mode is enabled.
        proteome_mix: Whether proteome mixture mode is enabled.

    Returns:
        DataFrame with columns reordered appropriately.
    """
    if phospho_mode:
        columns = [
            'protein_id', 'peptide_id', 'sequence', 'protein', 'decoy',
            'missed_cleavages', 'n_term', 'c_term', 'monoisotopic-mass',
            'retention_time_gru_predictor', 'events', 'rt_sigma', 'rt_lambda',
            'frame_occurrence_start', 'frame_occurrence_end', 'frame_occurrence',
            'frame_abundance', 'rt_mu',
            'phospho_site_a', 'phospho_site_b', 'sequence_original',
        ]
        return peptides[columns]
    elif proteome_mix:
        columns = [
            'protein_id', 'peptide_id', 'sequence', 'protein', 'decoy',
            'missed_cleavages', 'n_term', 'c_term', 'monoisotopic-mass',
            'retention_time_gru_predictor', 'events', 'rt_sigma', 'rt_lambda',
            'frame_occurrence_start', 'frame_occurrence_end', 'frame_occurrence',
            'frame_abundance',
            'total_events', 'fasta'
        ]
        # Only include columns that exist in the DataFrame
        return peptides[[col for col in columns if col in peptides.columns]]
    else:
        return peptides


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build a minimal argument parser that only accepts a TOML config file.

    Returns:
        ArgumentParser configured for TOML-only configuration.
    """
    description = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   TIMSIM - Proteomics Simulation Engine for timsTOF                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run a proteomics experiment simulation with PASEF-like acquisition on a
BRUKER timsTOF instrument. All configuration is provided via a TOML file.
"""

    epilog = """
──────────────────────────────────────────────────────────────────────────────
  Configuration File Structure
──────────────────────────────────────────────────────────────────────────────

  The TOML configuration file should contain the following sections:

    [paths]
      save_path       = "/path/to/output"
      reference_path  = "/path/to/reference.d"
      fasta_path      = "/path/to/proteome.fasta"

    [experiment]
      experiment_name   = "MyExperiment"
      acquisition_type  = "DIA"  # or "DDA"
      gradient_length   = 3600

    [digestion]
      num_sample_peptides = 25000
      missed_cleavages    = 2
      min_len             = 7
      max_len             = 30

    [simulation]
      batch_size   = 256
      num_threads  = -1  # -1 for auto-detect

──────────────────────────────────────────────────────────────────────────────
  Examples
──────────────────────────────────────────────────────────────────────────────

    timsim config.toml
    timsim /path/to/my-experiment-config.toml

──────────────────────────────────────────────────────────────────────────────
  Author: David Teschner | License: MIT | GitHub: @theGreatHerrLebert
──────────────────────────────────────────────────────────────────────────────
"""

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config",
        type=str,
        metavar="CONFIG_FILE",
        help="Path to the TOML configuration file"
    )
    return parser


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def main():
    """
    Entry point for the proteomics simulation script.

    Parses the TOML configuration file path from command line arguments,
    loads the configuration, and runs the simulation pipeline.
    """
    # Parse command line for config file path
    parser = build_arg_parser()
    cli_args = parser.parse_args()

    # Load configuration from TOML file
    try:
        config = SimulationConfig(cli_args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup logging based on config
    log_file = config.log_file
    if log_file and config.save_path:
        # If log_file is relative, make it relative to save_path
        if not os.path.isabs(log_file):
            log_file = os.path.join(config.save_path, log_file)

    setup_logging(log_level=config.log_level, log_file=log_file)

    # Detect terminal capability
    use_unicode = sys.stdout.isatty()

    # Print banner
    print(banner(use_unicode))
    logger.info(f"Version: {__version__}")
    logger.info("Author: David Teschner, JGU Mainz, GitHub: @theGreatHerrLebert")
    logger.info("License: MIT")
    logger.info("")

    # Initialize timer and stats
    timer = SimulationTimer()
    stats = SimulationStats()

    # Configure GPU
    logger.info(section_header("GPU Configuration", use_unicode))
    configure_gpu_memory(memory_limit_gb=config.gpu_memory_limit_gb, use_gpu=config.use_gpu)

    # Handle macOS Bruker SDK limitation
    use_bruker_sdk = config.use_bruker_sdk
    if platform.system() == 'Darwin':
        logger.warning("Bruker SDK is not supported on macOS, disabling SDK usage")
        use_bruker_sdk = False

    # Load modifications config
    logger.info(section_header("Loading Configuration", use_unicode))
    script_dir = Path(__file__).parent
    modifications_path = config.modifications
    if not modifications_path or modifications_path == "":
        modifications_path = script_dir / "configs" / "modifications.toml"
    mod_config = load_toml_config(str(modifications_path))
    variable_modifications = mod_config.get('variable_modifications', {})
    static_modifications = mod_config.get('static_modifications', {})

    # Log configuration details
    if not config.silent_mode:
        logger.info("")
        logger.info(f"  Variable modifications: {variable_modifications}")
        logger.info(f"  Static modifications:   {static_modifications}")
        logger.info("")
        if config.apply_fragmentation:
            logger.info("  Fragmentation: ENABLED")
        else:
            logger.info("  Fragmentation: DISABLED")

    # Log configuration table in debug mode
    config_dict = config.to_dict()
    logger.debug("Configuration settings:")
    for key, value in config_dict.items():
        logger.debug(f"  {key}: {value}")

    # Prepare paths
    save_path = check_path(config.save_path)
    reference_path = check_path(config.reference_path)
    name = config.experiment_name.replace('[PLACEHOLDER]', f'{config.acquisition_type}').replace("'", "")

    # Save configuration for future reference
    table_data = [[key, value] for key, value in config_dict.items()]
    with open(os.path.join(save_path, f'arguments-{name}.txt'), 'w') as f:
        f.write(tabulate(table_data, headers=["Argument", "Value"], tablefmt="grid"))

    # Build the acquisition (frames, scans, etc.)
    logger.info(section_header("Building Acquisition", use_unicode))
    if not config.silent_mode:
        logger.info("")
        logger.info(f"  Experiment:  {name}")
        logger.info(f"  Output path: {save_path}")
        logger.info("")

    acquisition_builder = build_acquisition(
        path=save_path,
        reference_path=reference_path,
        exp_name=name,
        acquisition_type=config.acquisition_type,
        verbose=not config.silent_mode,
        gradient_length=config.gradient_length,
        use_reference_ds_layout=config.use_reference_layout,
        reference_in_memory=config.reference_in_memory,
        round_collision_energy=config.round_collision_energy,
        collision_energy_decimals=config.collision_energy_decimals,
        use_bruker_sdk=use_bruker_sdk,
    )

    if not config.silent_mode:
        logger.info(str(acquisition_builder))

    # Possibly re-used from existing
    rt_sigma = None
    rt_lambda = None
    peptides, proteins, ions = None, None, None

    # Proteome mix setup (if needed)
    if config.proteome_mix and config.multi_fasta_dilution:
        factors = get_dilution_factors(config.multi_fasta_dilution)
    else:
        factors = {}

    if config.from_existing:
        logger.info(section_header("Loading Existing Simulation", use_unicode))
        # Load existing simulation data
        if config.acquisition_type == 'DIA':
            existing_sim_handle = SyntheticExperimentDataHandleDIA(database_path=config.existing_path)
        else:
            existing_sim_handle = SyntheticExperimentDataHandle(database_path=config.existing_path)

        peptides = existing_sim_handle.get_table('peptides')
        proteins = existing_sim_handle.get_table('proteins')
        ions = existing_sim_handle.get_table('ions')

        rt_sigma = peptides['rt_sigma'].values
        rt_lambda = peptides['rt_lambda'].values

        if not config.silent_mode:
            logger.info("")
            logger.info(f"  Source:   {config.existing_path}")
            logger.info(f"  Peptides: {peptides.shape[0]}")
            logger.info(f"  Ions:     {ions.shape[0]}")

        if config.re_scale_rt:
            if not config.silent_mode:
                logger.info(f"Re-scaling retention times to gradient length of {config.gradient_length} seconds")

            # re-scale retention times by running rt simulation again
            peptides = simulate_retention_times(
                peptides=peptides,
                verbose=not config.silent_mode,
                gradient_length=config.gradient_length,
                use_koina_model=config.koina_rt_model,
            )

        if config.rt_variation_std is not None:
            if not config.silent_mode:
                logger.info(f"Applying RT variation with std={config.rt_variation_std} seconds")

            # find the column containing the word 'retention_time' in its name
            col_name = [col for col in peptides.columns if 'retention_time' in col][0]

            # Apply RT variation
            peptides[col_name] = add_normal_noise(
                values=peptides[col_name],
                variation_std=config.rt_variation_std,
            )

        if config.ion_mobility_variation_std is not None:
            if not config.silent_mode:
                logger.info(f"Applying ion mobility variation with std={config.ion_mobility_variation_std}")

            # find the column containing the word 'inv_mobility' in its name
            col_name = [col for col in ions.columns if 'inv_mobility' in col][0]

            # Apply ion mobility variation
            ions[col_name] = add_normal_noise(
                values=ions[col_name],
                variation_std=config.ion_mobility_variation_std,
            )

        if config.intensity_variation_std is not None:
            if not config.silent_mode:
                logger.info(f"Applying intensity variation with std={config.intensity_variation_std}")

            # Apply intensity variation
            peptides["events"] = add_log_noise_variation(
                intensities=peptides["events"].values,
                log_noise_std=config.intensity_variation_std,
            )

        # If mixing is enabled, apply dilution factors by FASTA
        if config.proteome_mix:
            for fasta, dilution_factor in factors.items():
                if not config.silent_mode:
                    logger.info(f"Applying dilution factor {dilution_factor} to {fasta}")
                mask = peptides['fasta'] == fasta

                # need to multiply total_events by dilution factor and save it as events
                peptides.loc[mask, 'events'] = peptides.loc[mask, 'total_events']
                peptides.loc[mask, 'events'] *= dilution_factor
                peptides.loc[mask, 'fasta'] = fasta

        if config.phospho_mode:
            # if config.from_existing is True, we need to set template to False
            peptides, ions = simulate_phosphorylation(
                peptides=peptides,
                ions=ions,
                pick_phospho_sites=2,
                template=False,
                verbose=not config.silent_mode
            )

        # Warn if gradient length mismatch is large
        rt_max = peptides['retention_time_gru_predictor'].max()
        if abs(rt_max - config.gradient_length) / config.gradient_length > 0.05:
            logger.warning(
                f"Existing simulation gradient length ({rt_max}s) differs by >5% "
                f"from configured gradient length ({config.gradient_length}s)"
            )

    # ----------------------------------------
    # FASTA processing if not from existing
    # ----------------------------------------
    fastas = get_fasta_file_paths(config.fasta_path)
    protein_list, peptide_list = [], []

    if not config.from_existing:
        logger.info(section_header("Processing FASTA Files", use_unicode))
        for fasta_name, fasta_path in fastas.items():
            if not config.silent_mode:
                logger.info("")
                logger.info(f"  Digesting: {fasta_name}")

            mixture_factor = 1.0
            if config.proteome_mix:
                mixture_factor = factors.get(fasta_name, 1.0)
                if not config.silent_mode:
                    logger.info(f"Using mixture factor {mixture_factor} for {fasta_name}")

            # JOB 0: Generate Protein Data
            proteins_tmp = simulate_proteins(
                fasta_file_path=fasta_path,
                n_proteins=config.n_proteins,
                upscale_factor=config.upscale_factor,
                cleave_at=config.cleave_at,
                restrict=config.restrict,
                missed_cleavages=config.missed_cleavages,
                min_len=config.min_len,
                max_len=config.max_len,
                generate_decoys=config.decoys,
                variable_mods=variable_modifications,
                static_mods=static_modifications,
                verbose=not config.silent_mode,
                digest=config.digest_proteins,
                remove_degenerate_peptides=config.remove_degenerate_peptides,
            )

            # JOB 1: Simulate peptides
            if not config.silent_mode:
                logger.info("  Creating peptides from proteins...")

            peptides_tmp = simulate_peptides(
                protein_table=proteins_tmp,
                num_peptides_total=config.num_peptides_total,
                verbose=not config.silent_mode,
                exclude_accumulated_gradient_start=config.exclude_accumulated_gradient_start,
                min_rt_percent=config.min_rt_percent,
                gradient_length=acquisition_builder.gradient_length,
                down_sample=config.sample_peptides,
                min_length=config.min_len,
                max_length=config.max_len,
                proteome_mix=config.proteome_mix,
            )

            if config.proteome_mix:
                # Scale by mixture factor
                peptides_tmp['events'] *= mixture_factor
                peptides_tmp['fasta'] = fasta_name

            protein_list.append(proteins_tmp)
            peptide_list.append(peptides_tmp)

        # Concatenate across all FASTA inputs
        proteins = pd.concat(protein_list)[["protein_id", "protein", "sequence", "events"]]
        peptides = pd.concat(peptide_list)

        # Phospho mode
        if config.phospho_mode:
            if not config.silent_mode:
                logger.info("Simulating phosphorylation")
            peptides, _ = simulate_phosphorylation(
                peptides=peptides,
                ions=None,
                pick_phospho_sites=2,
                template=True,
                verbose=not config.silent_mode
            )

        # Subsample peptides if needed
        if config.sample_peptides:
            try:
                peptides = peptides.sample(n=config.num_sample_peptides, random_state=config.sample_seed).reset_index(
                    drop=True)
            except ValueError:
                logger.warning(
                    f"Not enough peptides to sample {config.num_sample_peptides}, "
                    f"using all {peptides.shape[0]} peptides"
                )

        if not config.silent_mode:
            logger.info("")
            logger.info(f"  Total peptides to simulate: {peptides.shape[0]}")

        # JOB 3: Simulate retention times
        logger.info(section_header("Simulating Retention Times", use_unicode))
        peptides = simulate_retention_times(
            peptides=peptides,
            verbose=not config.silent_mode,
            gradient_length=acquisition_builder.gradient_length,
            use_koina_model=config.koina_rt_model,
        )

        # Workaround for the correct column ordering
        columns = list(peptides.columns)
        # The last two might be 'events' and 'retention_time_gru_predictor' or some similar swap:
        columns[-2], columns[-1] = columns[-1], columns[-2]
        peptides = peptides[columns]

    # Determine number of threads
    num_threads = config.num_threads
    if num_threads == -1:
        num_threads = os.cpu_count()

    # JOB 4: Frame distributions
    logger.info(section_header("Simulating Frame Distributions", use_unicode))
    peptides = simulate_frame_distributions_emg(
        peptides=peptides,
        frames=acquisition_builder.frame_table,
        sigma_lower_rt=config.sigma_lower_rt,
        sigma_upper_rt=config.sigma_upper_rt,
        sigma_alpha_rt=config.sigma_alpha_rt,
        sigma_beta_rt=config.sigma_beta_rt,
        k_lower_rt=config.k_lower_rt,
        k_upper_rt=config.k_upper_rt,
        k_alpha_rt=config.k_alpha_rt,
        k_beta_rt=config.k_beta_rt,
        rt_cycle_length=acquisition_builder.rt_cycle_length,
        target_p=config.target_p,
        step_size=config.sampling_step_size,
        verbose=not config.silent_mode,
        add_noise=config.noise_frame_abundance,
        n_steps=config.n_steps,
        num_threads=num_threads,
        from_existing=config.from_existing,
        sigmas=rt_sigma,
        lambdas=rt_lambda,
        gradient_length=acquisition_builder.gradient_length,
        remove_epsilon=config.remove_epsilon,
    )

    # Save proteins
    acquisition_builder.synthetics_handle.create_table(table_name='proteins', table=proteins)

    # Handle final column ordering for phospho or proteome mixes
    peptides = reorder_peptide_columns(
        peptides,
        phospho_mode=config.phospho_mode,
        proteome_mix=config.proteome_mix
    )

    if config.proteome_mix:
        # Remove duplicate sequences for proteome mix
        peptides = peptides.drop_duplicates(subset=['sequence'])

    # Save peptides
    acquisition_builder.synthetics_handle.create_table(table_name='peptides', table=peptides)

    # ------------------------------------------------------------------
    # Further steps if not from existing simulation
    # ------------------------------------------------------------------
    if not config.from_existing:
        logger.info(section_header("Simulating Ion Properties", use_unicode))
        # JOB 5: Charge states
        ions = simulate_charge_states(
            peptides=peptides,
            mz_lower=acquisition_builder.tdf_writer.helper_handle.mz_lower,
            mz_upper=acquisition_builder.tdf_writer.helper_handle.mz_upper,
            p_charge=config.p_charge,
            max_charge=config.max_charge,
            charge_state_one_probability=config.charge_state_one_probability,
            use_binomial=config.binomial_charge_model,
            min_charge_contrib=config.min_charge_contrib,
            normalize=config.normalize_charge_states,
            verbose=not config.silent_mode,
        )

        # need to drop duplicates by sequence and charge state for ions if proteome mix
        if config.proteome_mix:
            ions = ions.drop_duplicates(subset=['sequence', 'charge'])

        # JOB 6: Ion mobilities
        ions = simulate_ion_mobilities_and_variance(
            ions=ions,
            im_lower=acquisition_builder.tdf_writer.helper_handle.im_lower,
            im_upper=acquisition_builder.tdf_writer.helper_handle.im_upper,
            verbose=not config.silent_mode,
            remove_mods=True,
            use_target_mean_std=config.use_inverse_mobility_std_mean,
            target_std_mean=config.inverse_mobility_std_mean,
        )

        # JOB 7: Precursor isotopic distributions
        ions = simulate_precursor_spectra_sequence(
            ions=ions,
            num_threads=num_threads,
            verbose=not config.silent_mode,
        )

    # JOB 8: Scan distributions
    logger.info(section_header("Simulating Scan Distributions", use_unicode))
    ions = simulate_scan_distributions_with_variance(
        ions=ions,
        scans=acquisition_builder.scan_table,
        verbose=not config.silent_mode,
        p_target=config.target_p,
        add_noise=config.noise_scan_abundance,
        num_threads=num_threads,
    )

    # Remove ions where peptide_id is not in the peptides table
    ions = ions[ions['peptide_id'].isin(peptides['peptide_id'])].reset_index(drop=True)

    # Save ions
    acquisition_builder.synthetics_handle.create_table(table_name='ions', table=ions)

    pasef_meta = None

    if config.acquisition_type == 'DDA':
        logger.info(section_header("Simulating DDA-PASEF Selection", use_unicode))
        pasef_meta, precursors = simulate_dda_pasef_selection_scheme(
            acquisition_builder=acquisition_builder,
            verbose=not config.silent_mode,
            precursors_every=config.precursors_every,
            intensity_threshold=config.precursor_intensity_threshold,
            max_precursors=config.max_precursors,
            selection_mode=config.selection_mode,
            precursor_exclusion_width=config.exclusion_width,
        )
        acquisition_builder.synthetics_handle.create_table(table_name='pasef_meta', table=pasef_meta)
        acquisition_builder.synthetics_handle.create_table(table_name='precursors', table=precursors)

    # JOB 9: Simulate fragment intensities
    logger.info(section_header("Simulating Fragment Intensities", use_unicode))
    if config.fragment_intensity_model:
        logger.info(f"  Using intensity model: {config.fragment_intensity_model}")
    if config.lazy_frame_assembly:
        logger.info("  Using lazy loading for fragment intensity simulation")
    simulate_fragment_intensities(
        path=save_path,
        name=name,
        acquisition_builder=acquisition_builder,
        batch_size=config.batch_size,
        verbose=not config.silent_mode,
        num_threads=num_threads,
        down_sample_factor=config.down_sample_factor,
        dda=config.acquisition_type == 'DDA',
        model_name=config.fragment_intensity_model,
        lazy_loading=config.lazy_frame_assembly,
        frame_batch_size=config.frame_batch_size,
    )

    # JOB 10: Assemble frames
    logger.info(section_header("Assembling Frames", use_unicode))
    assemble_frames(
        acquisition_builder=acquisition_builder,
        frames=acquisition_builder.frame_table,
        batch_size=config.batch_size,
        verbose=not config.silent_mode,
        mz_noise_precursor=config.mz_noise_precursor,
        mz_noise_uniform=config.mz_noise_uniform,
        precursor_noise_ppm=config.precursor_noise_ppm,
        mz_noise_fragment=config.mz_noise_fragment,
        fragment_noise_ppm=config.fragment_noise_ppm,
        num_threads=num_threads,
        add_real_data_noise=config.add_real_data_noise,
        intensity_max_precursor=config.reference_noise_intensity_max,
        intensity_max_fragment=config.reference_noise_intensity_max,
        precursor_sample_fraction=config.precursor_sample_fraction,
        fragment_sample_fraction=config.fragment_sample_fraction,
        num_precursor_frames=config.num_precursor_noise_frames,
        num_fragment_frames=config.num_fragment_noise_frames,
        fragment=config.apply_fragmentation,
        pasef_meta=pasef_meta,
        lazy_loading=config.lazy_frame_assembly,
        quad_isotope_transmission_mode=config.quad_isotope_transmission_mode,
        quad_transmission_min_probability=config.quad_transmission_min_probability,
        quad_transmission_max_isotopes=config.quad_transmission_max_isotopes,
    )

    # Collect final statistics
    stats.n_proteins = len(proteins) if proteins is not None else 0
    stats.n_peptides = len(peptides) if peptides is not None else 0
    stats.n_ions = len(ions) if ions is not None else 0
    stats.n_frames = len(acquisition_builder.frame_table)
    stats.acquisition_type = config.acquisition_type
    stats.experiment_name = name
    stats.output_path = str(save_path)

    # Print completion banner and summary
    print(simulation_complete_banner(use_unicode))

    total_time = timer.format_duration(timer.total_elapsed())
    logger.info("  Simulation Summary")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  Experiment:    {stats.experiment_name}")
    logger.info(f"  Type:          {stats.acquisition_type}")
    logger.info(f"  Proteins:      {stats.n_proteins:,}")
    logger.info(f"  Peptides:      {stats.n_peptides:,}")
    logger.info(f"  Ions:          {stats.n_ions:,}")
    logger.info(f"  Frames:        {stats.n_frames:,}")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  Total time:    {total_time}")
    logger.info(f"  Output:        {stats.output_path}")
    logger.info("")


if __name__ == '__main__':
    main()
