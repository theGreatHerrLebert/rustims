import os
import sys
import math

# Silence verbose package outputs before importing them
os.environ["WANDB_SILENT"] = "true"

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
    __version__ = get_version("imspy-simulation")
except Exception:
    __version__ = "0.4.0"  # Fallback


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
from imspy_simulation.experiment import SyntheticExperimentDataHandleDIA, SyntheticExperimentDataHandle
from imspy_simulation.timsim.jobs.simulate_ion_mobilities_and_variance import simulate_ion_mobilities_and_variance
from imspy_simulation.timsim.jobs.simulate_peptides import simulate_peptides
from imspy_simulation.timsim.jobs.simulate_phosphorylation import simulate_phosphorylation
from imspy_simulation.timsim.jobs.simulate_proteins import simulate_proteins
from imspy_simulation.timsim.jobs.simulate_scan_distributions_with_variance import (
    simulate_scan_distributions_with_variance
)
from imspy_simulation.utility import get_fasta_file_paths, get_dilution_factors
from imspy_simulation.timsim.jobs.utility import check_path

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
from .jobs.load_findings import load_findings, FindingsResult
from .jobs import checkpoint

# Optional video generation import (requires imspy-vis)
try:
    from imspy_vis.frame_rendering import generate_preview_video
    VIDEO_GENERATION_AVAILABLE = True
except ImportError:
    VIDEO_GENERATION_AVAILABLE = False

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
    Configures GPU usage for PyTorch.

    Args:
        memory_limit_gb: Memory limit in gigabytes per GPU (used for logging only,
                        PyTorch manages memory dynamically).
        use_gpu: If False, disables GPU and forces CPU-only mode.
    """
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("  GPU disabled via config, using CPU only")
        return

    import torch

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f}GB total)")
        logger.info(f"  PyTorch will manage GPU memory dynamically")
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


_LEGACY_SECTION_NAMES = frozenset({
    'main_settings', 'peptide_digestion', 'peptide_intensity',
    'charge_state_probabilities', 'distribution_settings',
    'noise_settings', 'phosphorylation_settings', 'dda_settings',
    'performance_settings', 'property_variation_settings',
})

_LEGACY_IGNORED_KEYS = frozenset({
    'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_value',
    'mean_std_rt', 'variance_std_rt', 'mean_skewness', 'variance_skewness',
    'z_score', 'add_noise_to_signals',
})


def translate_legacy_config(raw_config: dict) -> tuple[dict, bool]:
    """Detect and translate old-format timsim configs to the current format.

    Legacy configs use section names like ``[main_settings]``,
    ``[peptide_digestion]``, etc. and have a few renamed / removed keys.

    Args:
        raw_config: The raw dict returned by ``toml.load``.

    Returns:
        A tuple of (flat_config_dict, is_legacy).  ``is_legacy`` is True when
        old-format markers were detected so that callers can log accordingly.
    """
    is_legacy = bool(_LEGACY_SECTION_NAMES & set(raw_config.keys()))

    # Flatten sections (same logic as before – section names are irrelevant)
    flat: dict = {}
    for key, value in raw_config.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value

    if not is_legacy:
        return flat, False

    logger.info("Detected legacy config format, translating...")

    # Key rename: add_decoys → decoys
    if 'add_decoys' in flat:
        flat['decoys'] = flat.pop('add_decoys')
        logger.info("  Renamed key: add_decoys -> decoys")

    # binomial_charge_model: keep new default True even for legacy configs
    # (the old non-binomial charge model is broken)
    if 'binomial_charge_model' not in flat:
        logger.info("  Note: binomial_charge_model not set in legacy config, "
                     "keeping new default (True)")

    # Warn about ignored deprecated keys
    for key in sorted(_LEGACY_IGNORED_KEYS & set(flat.keys())):
        logger.info(f"  Ignoring deprecated key: {key}")
        del flat[key]

    return flat, True


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
        'from_findings': False,
        'findings_path': None,
        'intensity_multiplier': 1.0,
        # Shared event-scaling denominator across conditions; None = per-sample
        # median (legacy). Set the SAME value for every condition (A/B/...) of a
        # multi-sample experiment to preserve cross-sample intensity ratios.
        'findings_reference_median': None,
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

        # Model selection (None/"local" = PyTorch, or Koina model name)
        # RT models: "local", "Deeplc_hela_hf", "Chronologer_RT", "AlphaPeptDeep_rt_generic", "Prosit_2019_irt"
        'rt_model': None,
        'koina_rt_model': None,  # Deprecated alias for rt_model
        # CCS models: "local", "AlphaPeptDeep_ccs_generic", "IM2Deep"
        'ccs_model': None,
        # Intensity models: "local"/"prosit", "alphapeptdeep", "ms2pip", or full Koina name
        'intensity_model': None,

        # P6d: instrument the run records fragments for. Drives the collision-energy
        # UNIT (Bruker timsTOF = absolute eV; Orbitrap Astral = normalized CE / NCE),
        # the fragment-model compatibility guard, and the prediction-set provenance.
        # Default 'bruker_timstof' preserves current behaviour exactly.
        'instrument': 'bruker_timstof',
        # Build-from-template: for any Thermo instrument (orbitrap_astral,
        # orbitrap_exploris), the Thermo .raw template the run is built from — its real
        # per-scan schedule + windows become the acquisition (no Bruker reference).
        # `template_path` is the generic name; `astral_template_path` is the historical
        # alias (both accepted; see thermo_template_path()). Required for a Thermo run.
        'template_path': None,
        'astral_template_path': None,

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
        # Instrument-dispatch projector (opt-in). 'off' (default) keeps the
        # legacy frame/scan distribution columns untouched; 'legacy_compat'
        # regenerates them via the Rust projector reproducing the legacy kernels;
        # 'accurate' uses the improved projection (event-interval time + per-scan
        # mobility bins). See jobs/project_distributions.py.
        'projection_mode': 'off',
        'use_inverse_mobility_std_mean': True,
        'inverse_mobility_std_mean': 0.009,

        # Acquisition settings
        'round_collision_energy': True,
        'collision_energy_decimals': 0,
        # P6d: for instrument='orbitrap_astral', the normalized collision energy
        # (NCE) that REPLACES the reference-derived Bruker eV window CE (required
        # for an Astral run). Ignored for Bruker timsTOF.
        'collision_energy_nce': None,

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
        'binomial_charge_model': True,
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
        'superimpose_on_reference': False,
        # MS2 centroid merge tolerance (ppm) used when superimposing simulated peaks
        # onto a Thermo .raw template's real signal (Astral/Orbitrap Overlay mode).
        # Ignored for the Bruker path and when superimpose_on_reference is False.
        'superimpose_merge_ppm': 15.0,
        'reference_noise_intensity_max': 30,
        'down_sample_factor': 0.5,

        # mzPROV provenance (Ed25519-signed self-disclosure: "this IS TimSim simulated
        # data", binds the output to the config + a signing key). Emitted for ALL four
        # vendors: Bruker .d (structural canonicalization), SCIEX/Waters mzML (mzML content
        # canonicalization), and Thermo .raw (opaque whole-file content hash, sidecar-only —
        # a vendor binary can't be embedded). Requires the optional `mzprov` package
        # (with .raw support); import-guarded.
        'emit_provenance': True,
        # True (default): EMBED the signed envelope INTO the output itself — the .d's
        # analysis.tdf (a provenance table) or the mzML's fileContent — so provenance
        # travels with the file and no extra file is produced. False: write a sibling
        # `<name>.provenance.json` sidecar instead. (A vendor .raw can't be embedded; that
        # path would fall back to a sidecar.)
        'provenance_embed': True,
        'provenance_key_path': None,      # None = default ~/.config/timsim/keys/ (auto-gen)

        # SCIEX ZenoTOF SWATH build-from-.wiff (instrument=sciex_zenotof; template_path = .wiff).
        # The .wiff method has no per-scan timing, so the SWATH schedule is synthesized from
        # these + gradient_length. Rolling CE = ce_intercept + ce_slope_per_mz * precursor_mz.
        'sciex_cycle_time_s': 3.5,
        'sciex_ce_intercept': 5.0,
        'sciex_ce_slope_per_mz': 0.045,
        # Author native .wiff.scan spectra into the template instead of open mzML. Requires the
        # connector built with the `sciex` feature; the sim schedule is matched to the template's
        # cycle count (authored cycles fill it, extras cleared). Off -> mzML (the portable path).
        'sciex_native': False,
        # Gaussian m/z noise (≈ppm at 3σ) added at native authoring, so a search engine has a
        # mass-error distribution to calibrate against (a zero-error spectrum is degenerate).
        'sciex_precursor_noise_ppm': 5.0,
        'sciex_fragment_noise_ppm': 8.0,
        # Spike-in / overlay: > 0 keeps the template's real peaks and adds the simulated ones on
        # top (real⊕sim); 0 = pure-synthetic (template peaks replaced).
        'sciex_overlay_ppm': 0.0,

        # Waters SONAR build-from-parameters (instrument=waters_synapt_xs; NO template file).
        # SONAR is a scanning-quadrupole DIA fully described by these; the schedule is
        # synthesized from them + gradient_length. window_step=None -> contiguous windows
        # (set < window_width for the faithful overlapping quad scan). Rolling CE = intercept
        # + slope_per_mz * window_center_mz.
        'waters_mz_start': 400.0,
        'waters_mz_end': 900.0,
        'waters_window_width': 20.0,
        'waters_window_step': None,
        'waters_cycle_time_s': 0.5,
        'waters_ce_intercept': 5.0,
        'waters_ce_slope_per_mz': 0.04,

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

        # Precursor survival settings - fraction of precursor ions that survive fragmentation intact
        # When both are 0.0, no unfragmented precursors are added (backward compatible)
        # For realistic simulation, try min=0.05, max=0.15
        'precursor_survival_min': 0.0,
        'precursor_survival_max': 0.0,

        # Logging settings
        'log_level': 'INFO',
        'log_file': None,

        # Video generation settings
        'generate_preview_video': False,
        'preview_video_max_frames': 100,
        'preview_video_fps': 10,
        'preview_video_dpi': 80,
        'preview_video_annotate': True,

        # Checkpointing
        'enable_checkpoints': False,
    }


def thermo_template_path(config) -> "str | None":
    """The Thermo ``.raw`` template path for a build-from-template run.

    Accepts the generic ``template_path`` and the historical ``astral_template_path``
    alias (the latter takes precedence if both are set)."""
    get = config.get if isinstance(config, dict) else (lambda k, d=None: getattr(config, k, d))
    return get('astral_template_path', None) or get('template_path', None)


def astral_nce_override(config) -> "float | None":
    """The DIA-window NCE override to apply for this run (P6d).

    Returns the configured normalized collision energy ONLY for a Thermo
    build-from-template run (Astral/Orbitrap); ``None`` (i.e. ignore it, keep the
    reference eV windows) for Bruker — so a stray ``collision_energy_nce`` on a Bruker
    config can never silently replace eV windows with an NCE while provenance still
    labels them eV."""
    from .jobs.register_prediction_set import is_thermo_template_instrument
    instrument = str(getattr(config, 'instrument', 'bruker_timstof')).lower()
    if is_thermo_template_instrument(instrument):
        return getattr(config, 'collision_energy_nce', None)
    return None


class SimulationConfig:
    """
    Configuration container for timsim simulation parameters.

    Loads configuration from a TOML file and provides attribute-style access
    to all settings with defaults applied.
    """

    def __init__(self, config_path: str, overrides: dict | None = None):
        """
        Initialize configuration from a TOML file.

        Args:
            config_path: Path to the TOML configuration file.
            overrides: Optional dict of key→value pairs that take precedence
                over values loaded from the TOML file (e.g. CLI path overrides).

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If required settings are missing or invalid.
        """
        self._config = get_default_settings()

        # Load TOML config, detect/translate legacy format, then flatten
        raw_config = load_toml_config(config_path)
        flat_config, is_legacy = translate_legacy_config(raw_config)
        self._config.update(flat_config)

        # Apply CLI overrides (take precedence over config file)
        if overrides:
            self._config.update(overrides)

        # Validate required settings
        self._validate()

    def _validate(self) -> None:
        """Validate that required configuration options are set."""
        if self._config.get('from_findings') and self._config.get('from_existing'):
            raise ValueError("Cannot use both 'from_findings' and 'from_existing' at the same time")

        # A Thermo build-from-template run (Astral/Orbitrap) is built from a Thermo .raw
        # template, NOT a Bruker reference .d — so it requires a template path in place
        # of reference_path (the lean, Bruker-independent build-from-template path).
        from .jobs.register_prediction_set import (
            is_thermo_template_instrument, is_sciex_instrument, is_waters_instrument,
        )
        instrument = str(self._config.get('instrument', 'bruker_timstof')).lower()
        is_thermo = is_thermo_template_instrument(instrument)
        # Reference-free runs build WITHOUT a Bruker reference .d: Thermo .raw and SCIEX
        # .wiff build from a vendor template, Waters SONAR is fully synthesized from
        # parameters. None of them require `reference_path`. (Thermo/SCIEX still require a
        # `template_path`/`astral_template_path` file, validated per-instrument below;
        # Waters requires no file at all.)
        is_template_based = is_thermo or is_sciex_instrument(instrument) or is_waters_instrument(instrument)
        ref_key = 'reference_path'
        if self._config.get('from_findings'):
            required = ['save_path', 'findings_path'] if is_template_based else ['save_path', ref_key, 'findings_path']
        else:
            required = ['save_path', 'fasta_path'] if is_template_based else ['save_path', ref_key, 'fasta_path']
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

        # Validate superimpose_on_reference is DIA-only
        if self._config.get('superimpose_on_reference', False) and self._config.get('acquisition_type') == 'DDA':
            raise ValueError("superimpose_on_reference is only supported for DIA acquisition mode")

        # For a Thermo build-from-template run, superimposition uses an MS2 centroid
        # merge tolerance (ppm). A 0/negative tolerance would silently fall back to
        # Replace (overwrite, destroying the reference signal the user asked to keep),
        # so require a strictly-positive, finite tolerance when superimposing onto a
        # Thermo template. (The Bruker path ignores this key.)
        if self._config.get('superimpose_on_reference', False) and is_thermo:
            raw_ppm = self._config.get('superimpose_merge_ppm')
            # Coerce ONCE; reject bool (float(True)==1.0 would slip through) and any
            # non-numeric value with the contextual error, not a bare float() exception.
            merge_ppm = None
            if not isinstance(raw_ppm, bool):
                try:
                    merge_ppm = float(raw_ppm)
                except (TypeError, ValueError):
                    merge_ppm = None
            if merge_ppm is None or not math.isfinite(merge_ppm) or merge_ppm <= 0.0:
                raise ValueError(
                    "superimpose_on_reference on a Thermo template requires "
                    f"superimpose_merge_ppm > 0 (got {raw_ppm!r}); a 0/negative tolerance "
                    "would overwrite the reference signal instead of overlaying onto it"
                )

        # Validate p_charge
        p_charge = self._config.get('p_charge', 0.8)
        if not (0.0 < p_charge < 1.0):
            raise ValueError(f"p_charge must be between 0 and 1, got {p_charge}")

        # Validate instrument-dispatch projector mode (fail fast on bad TOML,
        # before any expensive simulation runs).
        projection_mode = str(self._config.get('projection_mode', 'off')).lower()
        if projection_mode not in ('off', 'legacy_compat', 'accurate'):
            raise ValueError(
                f"projection_mode must be off|legacy_compat|accurate, got {projection_mode!r}"
            )
        if projection_mode != 'off':
            # The projector is deterministic, so it would silently overwrite any
            # configured abundance noise. Reject the combination explicitly.
            if self._config.get('noise_frame_abundance') or self._config.get('noise_scan_abundance'):
                raise ValueError(
                    "projection_mode is incompatible with noise_frame_abundance/"
                    "noise_scan_abundance (the projector is deterministic and would "
                    "drop the noise); disable the noise or projection_mode"
                )
            # The 'ions' checkpoint persists in-memory distributions; resuming would
            # restore the legacy (pre-projection) columns and diverge from a projected
            # run. Not yet supported together.
            if self._config.get('enable_checkpoints'):
                raise ValueError(
                    "projection_mode is not yet supported with enable_checkpoints "
                    "(checkpoint resume would restore pre-projection distributions)"
                )

        # P6d: validate the instrument selection at config load (fail fast, before
        # any expensive acquisition/simulation work leaves partial output).
        from .jobs.register_prediction_set import INSTRUMENT_ACTIVATION
        instrument = str(self._config.get('instrument', 'bruker_timstof')).lower()
        if instrument not in INSTRUMENT_ACTIVATION:
            raise ValueError(
                f"unknown instrument '{instrument}'. Supported: "
                f"{sorted(INSTRUMENT_ACTIVATION)}."
            )
        if is_thermo:
            # Thermo build-from-template instruments (Astral/Orbitrap) are DIA, no-IMS,
            # NCE. DDA-PASEF CE is Bruker scan-driven, and the run MUST supply a genuine
            # normalized collision energy — otherwise reference-derived Bruker eV
            # windows would be silently mislabelled as NCE and fed to the NCE predictor.
            if self._config.get('acquisition_type') == 'DDA':
                raise ValueError(
                    f"instrument '{instrument}' does not support DDA acquisition "
                    "(DDA-PASEF collision energy is Bruker scan-driven). Use DIA."
                )
            # collision_energy_nce is OPTIONAL for the build-from-template path: the
            # template already supplies a genuine per-window NCE, used by default. If
            # set, it OVERRIDES every window with that single NCE (a deliberate manual
            # choice); if set, it must be positive.
            nce = self._config.get('collision_energy_nce')
            if nce is not None and not (isinstance(nce, (int, float)) and nce > 0):
                raise ValueError(
                    "collision_energy_nce, if set, must be a positive normalized "
                    "collision energy (e.g. 27); it overrides the template's "
                    "per-window NCE for a Thermo build-from-template run."
                )
            # The Thermo MS2 render is deterministic; precursor survival is a stochastic
            # (per-scan random fraction) feature. Refuse rather than silently drop the
            # configured survival signal (it is not modelled on this path yet).
            if self._config.get('precursor_survival_max', 0.0) and float(
                self._config.get('precursor_survival_max', 0.0)
            ) > 0.0:
                raise ValueError(
                    f"instrument '{instrument}' does not support precursor_survival_* "
                    "(the Thermo MS2 render is deterministic; survival is stochastic). "
                    "Set precursor_survival_min/max to 0 for a Thermo run."
                )
            # Build-from-template: the run is built from a Thermo .raw template (its real
            # per-scan schedule + windows become the acquisition). Accept either
            # `template_path` or the `astral_template_path` alias.
            template = thermo_template_path(self._config)
            if not template:
                raise ValueError(
                    f"instrument '{instrument}' requires 'template_path' (or the "
                    "'astral_template_path' alias): a Thermo .raw whose per-scan "
                    "schedule + windows the run is built from."
                )
            if not os.path.exists(template):
                raise FileNotFoundError(f"template_path does not exist: {template}")
            # The Thermo .raw writer + dispatch live behind the connector's 'thermo'
            # feature, which the published wheels disable (the Thermo writer dependency
            # is private). Fail fast at config load instead of an AttributeError after a
            # full simulation.
            try:
                import imspy_connector
                thermo_ok = bool(imspy_connector.py_acquisition.has_thermo())
            except Exception:
                thermo_ok = False
            if not thermo_ok:
                raise ValueError(
                    f"instrument '{instrument}' requires imspy-connector built with the "
                    "'thermo' feature (maturin build --release --features thermo, then "
                    "reinstall). The published wheels disable it because the Thermo .raw "
                    "writer dependency is private."
                )

        if is_sciex_instrument(instrument):
            # SCIEX ZenoTOF SWATH is build-from-.wiff: DIA-only (the schedule is a
            # synthesized SWATH cycle), no IMS. DDA has no meaning on this path.
            if self._config.get('acquisition_type') == 'DDA':
                raise ValueError(
                    f"instrument '{instrument}' does not support DDA acquisition "
                    "(SCIEX SWATH is a DIA cycle). Use DIA."
                )
            # Build-from-.wiff: the run is built from a SCIEX .wiff SWATH method (its
            # windows + TOF cal become the acquisition; the schedule is synthesized).
            # Accept either `template_path` or the `astral_template_path` alias.
            template = thermo_template_path(self._config)
            if not template:
                raise ValueError(
                    f"instrument '{instrument}' requires 'template_path' (or the "
                    "'astral_template_path' alias): a SCIEX .wiff whose SWATH windows "
                    "the run is built from."
                )
            if not os.path.exists(template):
                raise FileNotFoundError(f"template_path does not exist: {template}")
            # The rolling-CE model must be finite/positive: CE = intercept + slope*mz
            # conditions fragment-intensity prediction, and a non-finite/negative value
            # would silently yield empty or unphysical MS2.
            for key in ('sciex_cycle_time_s', 'sciex_ce_intercept', 'sciex_ce_slope_per_mz'):
                val = self._config.get(key)
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(f"{key} must be a finite number, got {val!r}")
            if self._config.get('sciex_cycle_time_s') <= 0.0:
                raise ValueError(
                    f"sciex_cycle_time_s must be > 0, got {self._config.get('sciex_cycle_time_s')!r}"
                )
            # The mzML renderer lives behind the connector's 'mzml' feature; fail fast at
            # config load instead of an AttributeError after a full simulation.
            try:
                import imspy_connector
                mzml_ok = bool(imspy_connector.py_acquisition.has_mzml())
            except Exception:
                mzml_ok = False
            if not mzml_ok:
                raise ValueError(
                    f"instrument '{instrument}' requires imspy-connector built with the "
                    "'mzml' feature (maturin build --release --features mzml, then "
                    "reinstall) to render SCIEX SWATH output to open mzML."
                )

        if is_waters_instrument(instrument):
            # Waters SONAR is build-from-parameters: DIA-only (scanning-quadrupole cycle),
            # no IMS, no vendor file required. DDA has no meaning on this path.
            if self._config.get('acquisition_type') == 'DDA':
                raise ValueError(
                    f"instrument '{instrument}' does not support DDA acquisition "
                    "(SONAR is a scanning-quadrupole DIA cycle). Use DIA."
                )
            # The SONAR scan geometry + rolling-CE model must be finite/positive: a
            # non-finite/degenerate value would yield an empty or unphysical window scheme
            # or condition fragment-intensity prediction to empty MS2.
            mz_start = self._config.get('waters_mz_start')
            mz_end = self._config.get('waters_mz_end')
            for key in (
                'waters_mz_start', 'waters_mz_end', 'waters_window_width',
                'waters_cycle_time_s', 'waters_ce_intercept', 'waters_ce_slope_per_mz',
            ):
                val = self._config.get(key)
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(f"{key} must be a finite number, got {val!r}")
            if not (mz_end > mz_start):
                raise ValueError(
                    f"waters_mz_end must be > waters_mz_start (got {mz_start}..{mz_end})"
                )
            width = self._config.get('waters_window_width')
            if width <= 0.0:
                raise ValueError(f"waters_window_width must be > 0, got {width!r}")
            # A window wider than the scanned span is not a valid SONAR geometry.
            if width > (mz_end - mz_start):
                raise ValueError(
                    f"waters_window_width ({width}) must be <= the m/z span "
                    f"({mz_end - mz_start}); a window wider than the scanned range "
                    "is not a valid SONAR geometry"
                )
            if self._config.get('waters_cycle_time_s') <= 0.0:
                raise ValueError(
                    f"waters_cycle_time_s must be > 0, got {self._config.get('waters_cycle_time_s')!r}"
                )
            # gradient_length drives the cycle count; a non-finite/non-positive value would
            # silently collapse to a single cycle (or crash mid-build). Fail fast.
            grad = self._config.get('gradient_length')
            if isinstance(grad, bool) or not isinstance(grad, (int, float)) or not math.isfinite(grad) or grad <= 0.0:
                raise ValueError(f"gradient_length must be a finite number > 0, got {grad!r}")
            # window_step is optional (None -> contiguous); if set it must be finite, positive,
            # and <= window_width (a larger step would leave uncovered gaps between windows).
            step = self._config.get('waters_window_step')
            if step is not None:
                if (isinstance(step, bool) or not isinstance(step, (int, float))
                        or not math.isfinite(step) or step <= 0.0):
                    raise ValueError(
                        f"waters_window_step, if set, must be a finite number > 0, got {step!r}"
                    )
                if step > width:
                    raise ValueError(
                        f"waters_window_step ({step}) must be <= waters_window_width ({width}); "
                        "a larger step would leave uncovered gaps between windows"
                    )
            # Rolling CE must stay strictly positive across the scanned range, else fragment
            # prediction is conditioned on a non-positive NCE (empty/unphysical MS2). Check the
            # range ends (CE is linear in m/z, so the minimum is at one end).
            ce_b = self._config.get('waters_ce_intercept')
            ce_m = self._config.get('waters_ce_slope_per_mz')
            ce_ends = (ce_b + ce_m * mz_start, ce_b + ce_m * mz_end)
            if min(ce_ends) <= 0.0:
                raise ValueError(
                    f"waters rolling-CE model (intercept={ce_b}, slope={ce_m}) yields a "
                    f"non-positive collision energy over {mz_start}..{mz_end} m/z "
                    f"(ends {ce_ends[0]:.3f}, {ce_ends[1]:.3f}); use a positive CE model"
                )
            # The mzML renderer lives behind the connector's 'mzml' feature; fail fast.
            try:
                import imspy_connector
                mzml_ok = bool(imspy_connector.py_acquisition.has_mzml())
            except Exception:
                mzml_ok = False
            if not mzml_ok:
                raise ValueError(
                    f"instrument '{instrument}' requires imspy-connector built with the "
                    "'mzml' feature (maturin build --release --features mzml, then "
                    "reinstall) to render Waters SONAR output to open mzML."
                )

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
    parser.add_argument(
        "--save-path", "-s",
        type=str,
        default=None,
        help="Override save_path from config"
    )
    parser.add_argument(
        "--reference-path", "-r",
        type=str,
        default=None,
        help="Override reference_path from config"
    )
    parser.add_argument(
        "--fasta-path", "-f",
        type=str,
        default=None,
        help="Override fasta_path from config"
    )
    parser.add_argument(
        "--findings-path",
        type=str,
        default=None,
        help="Override findings_path from config (enables from_findings mode)"
    )
    parser.add_argument(
        "--intensity-multiplier",
        type=float,
        default=None,
        help="Multiply all findings intensities by this factor (e.g. 10 for 10x sensitivity, 0.1 for 10x reduction)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint (requires a previous run with enable_checkpoints=true)"
    )
    parser.add_argument(
        "--projection-mode",
        choices=["off", "legacy_compat", "accurate"],
        default=None,
        help="Instrument-dispatch projector for frame/scan distributions: 'off' (default) "
             "keeps the legacy columns; 'legacy_compat' reproduces them via the Rust projector; "
             "'accurate' uses the improved projection."
    )
    return parser


def emit_provenance_sidecar(d_path, db_path, config_path, experiment_name,
                            embed, key_path, logger) -> None:
    """Write an mzPROV Ed25519-signed provenance sidecar for a Bruker .d output —
    tamper-evident self-disclosure that this is TimSim-simulated data, bound to the
    config and a signing key. Import-guarded: a missing `mzprov` package or any signing
    error is logged as a warning and never fails the run. mzprov v0 canonicalizes
    .d/mzML only (not vendor .raw), so callers gate this to the Bruker path."""
    try:
        from mzprov.sign import sign_simulation_output
    except ImportError:
        logger.warning(
            "  provenance: `mzprov` not installed — skipping sidecar "
            "(pip install the mzprov python implementation to enable)"
        )
        return
    try:
        from imspy_simulation import __version__ as _sim_version
    except Exception:
        _sim_version = "unknown"
    try:
        gt = db_path if (db_path and os.path.exists(db_path)) else None
        out = sign_simulation_output(
            d_path=d_path,
            ground_truth_path=gt,
            config_path=config_path,
            experiment_name=experiment_name,
            simulator_version=_sim_version,
            private_key_path=key_path,
            embed=embed,
        )
        logger.info(f"  provenance: mzPROV sidecar -> {out}"
                    + (" (embedded in .d)" if embed else ""))
    except Exception as e:
        logger.warning(f"  provenance: mzPROV signing failed (non-fatal): {e}")


def emit_provenance_sidecar_mzml(mzml_path, config_path, experiment_name,
                                 embed, key_path, logger) -> None:
    """Write an mzPROV Ed25519-signed provenance sidecar for an mzML output (the SCIEX
    and Waters open-format paths) — tamper-evident self-disclosure that this is TimSim-
    simulated data, bound to the config + a signing key. Uses mzprov's first-class mzML
    signer (``sign_mzml_output``), which canonicalizes the mzML content (config + mzML
    content hash; note v0 does not bind the ground-truth DB the way the .d path does).
    Import-guarded: a missing ``mzprov`` package or any signing error is logged as a
    warning and never fails the run."""
    try:
        from mzprov.sign import sign_mzml_output
    except ImportError:
        logger.warning(
            "  provenance: `mzprov` not installed — skipping mzML sidecar "
            "(pip install the mzprov python implementation to enable)"
        )
        return
    try:
        from imspy_simulation import __version__ as _sim_version
    except Exception:
        _sim_version = "unknown"
    try:
        out = sign_mzml_output(
            mzml_path=mzml_path,
            config_path=config_path,
            experiment_name=experiment_name,
            tool_name="TimSim",
            tool_version=_sim_version,
            private_key_path=key_path,
            embed=embed,
        )
        logger.info(f"  provenance: mzPROV mzML sidecar -> {out}"
                    + (" (embedded in mzML)" if embed else ""))
    except Exception as e:
        logger.warning(f"  provenance: mzPROV mzML signing failed (non-fatal): {e}")


def emit_provenance_sidecar_raw(raw_path, config_path, experiment_name, key_path, logger) -> None:
    """Write an mzPROV Ed25519-signed provenance sidecar for a Thermo .raw output. A vendor
    .raw is an opaque proprietary binary with no safe injection point, so this is ALWAYS a
    JSON sidecar (regardless of the run's embed preference) and the attestation is an opaque
    whole-file content hash (sensitive to any byte change). Uses mzprov's ``sign_raw_output``.
    Import-guarded (covers both a missing ``mzprov`` and an older ``mzprov`` without raw
    support): any failure is logged as a warning and never fails the run."""
    try:
        from mzprov.sign import sign_raw_output
    except ImportError:
        logger.warning(
            "  provenance: `mzprov` (with .raw support) not available — skipping .raw "
            "sidecar (pip install/upgrade the mzprov python implementation to enable)"
        )
        return
    try:
        from imspy_simulation import __version__ as _sim_version
    except Exception:
        _sim_version = "unknown"
    try:
        out = sign_raw_output(
            raw_path=raw_path,
            config_path=config_path,
            experiment_name=experiment_name,
            tool_name="TimSim",
            tool_version=_sim_version,
            private_key_path=key_path,
        )
        logger.info(f"  provenance: mzPROV .raw sidecar -> {out}")
    except Exception as e:
        logger.warning(f"  provenance: mzPROV .raw signing failed (non-fatal): {e}")


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

    # Build path overrides from CLI arguments
    overrides = {}
    if cli_args.save_path:
        overrides['save_path'] = cli_args.save_path
    if cli_args.reference_path:
        overrides['reference_path'] = cli_args.reference_path
    if cli_args.fasta_path:
        overrides['fasta_path'] = cli_args.fasta_path
    if cli_args.findings_path:
        overrides['findings_path'] = cli_args.findings_path
        overrides['from_findings'] = True
    if cli_args.intensity_multiplier is not None:
        overrides['intensity_multiplier'] = cli_args.intensity_multiplier
    if cli_args.resume:
        overrides['enable_checkpoints'] = True
    if cli_args.projection_mode is not None:
        overrides['projection_mode'] = cli_args.projection_mode

    # Load configuration from TOML file
    try:
        config = SimulationConfig(cli_args.config, overrides=overrides)
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
    # An Astral run has no Bruker reference .d (it builds from a .raw template).
    reference_path = check_path(config.reference_path) if config.reference_path else None
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

    from .jobs.register_prediction_set import (
        is_thermo_template_instrument, is_sciex_instrument, is_waters_instrument,
    )
    instrument = str(getattr(config, 'instrument', 'bruker_timstof')).lower()
    if is_thermo_template_instrument(instrument):
        # Build-from-template (P6e option b): NO Bruker reference. The Thermo template's
        # real per-scan schedule + windows become the acquisition, so the trunk is
        # simulated on the template's true non-uniform timeline. Same path for Astral
        # and classic Orbitrap — the .raw writer handles the MS2 detector format.
        from .jobs.astral_acquisition import AstralAcquisitionBuilder
        _template = thermo_template_path(config)
        logger.info(f"  Thermo build-from-template ({instrument}): {_template}")
        acquisition_builder = AstralAcquisitionBuilder(
            str(Path(save_path) / name),
            _template,
            round_collision_energy=config.round_collision_energy,
            collision_energy_decimals=config.collision_energy_decimals,
            # Optional override: if set, forces a single NCE across all windows;
            # otherwise the template's genuine per-window NCE is used.
            collision_energy_nce=astral_nce_override(config),
            verbose=not config.silent_mode,
        )
    elif is_sciex_instrument(instrument):
        # Build-from-.wiff (SCIEX ZenoTOF SWATH): the .wiff SWATH method (windows + TOF
        # cal) becomes the acquisition; the schedule is SYNTHESIZED (the .wiff has no
        # timing) from gradient_length + sciex_cycle_time_s + a rolling-CE model. Output
        # is open mzML (the proprietary .wiff.scan is not authored).
        from .jobs.sciex_acquisition import SciexAcquisitionBuilder
        _wiff = thermo_template_path(config)
        logger.info(f"  SCIEX build-from-.wiff ({instrument}): {_wiff}")
        # Native authoring writes into the template's fixed block schedule, so match the sim's
        # cycle count to the template (codex #6): keep the user's gradient for the elution model
        # and set cycle_time = gradient / template_cycles, so every template cycle is filled.
        _sciex_cycle_time = config.sciex_cycle_time_s
        if config.sciex_native:
            import imspy_connector
            if imspy_connector.py_acquisition.has_sciex():
                _tcyc = imspy_connector.py_acquisition.sciex_template_cycles(_wiff)
                if _tcyc > 0:
                    _sciex_cycle_time = config.gradient_length / _tcyc
                    logger.info(
                        f"  native: matching {_tcyc} template cycles "
                        f"(cycle_time {_sciex_cycle_time:.4f}s over {config.gradient_length}s gradient)"
                    )
        acquisition_builder = SciexAcquisitionBuilder(
            str(Path(save_path) / name),
            _wiff,
            cycle_time_s=_sciex_cycle_time,
            gradient_length_s=config.gradient_length,
            ce_intercept=config.sciex_ce_intercept,
            ce_slope_per_mz=config.sciex_ce_slope_per_mz,
            round_collision_energy=config.round_collision_energy,
            collision_energy_decimals=config.collision_energy_decimals,
            verbose=not config.silent_mode,
        )
    elif is_waters_instrument(instrument):
        # Build-from-parameters (Waters SONAR): NO vendor file. SONAR is a scanning-
        # quadrupole DIA fully described by its scan range + window + cycle, so the
        # schedule is SYNTHESIZED from those + gradient_length + a rolling-CE model.
        # Output is open mzML.
        from .jobs.waters_acquisition import WatersSonarAcquisitionBuilder
        logger.info(f"  Waters SONAR build-from-parameters ({instrument})")
        acquisition_builder = WatersSonarAcquisitionBuilder(
            str(Path(save_path) / name),
            mz_start=config.waters_mz_start,
            mz_end=config.waters_mz_end,
            window_width=config.waters_window_width,
            window_step=config.waters_window_step,
            cycle_time_s=config.waters_cycle_time_s,
            gradient_length_s=config.gradient_length,
            ce_intercept=config.waters_ce_intercept,
            ce_slope_per_mz=config.waters_ce_slope_per_mz,
            round_collision_energy=config.round_collision_energy,
            collision_energy_decimals=config.collision_energy_decimals,
            verbose=not config.silent_mode,
        )
    else:
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
            # NCE override is Astral-only — never replace Bruker eV windows with an
            # NCE (that would silently mislabel eV as NCE). Ignored for Bruker.
            collision_energy_nce=astral_nce_override(config),
            use_bruker_sdk=use_bruker_sdk,
        )

    if not config.silent_mode:
        logger.info(str(acquisition_builder))

    # Possibly re-used from existing
    rt_sigma = None
    rt_lambda = None
    peptides, proteins, ions = None, None, None
    pasef_meta = None
    precursors = None

    # Check for resume from checkpoint
    resume_after = None
    if cli_args.resume:
        resume_after = checkpoint.latest(str(save_path))
        if resume_after:
            logger.info(section_header(f"Resuming from Checkpoint: {resume_after}", use_unicode))
            cp_data = checkpoint.load(str(save_path), resume_after)
            proteins = cp_data["proteins"]
            peptides = cp_data["peptides"]
            ions = cp_data.get("ions")
            pasef_meta = cp_data.get("pasef_meta")
            precursors = cp_data.get("precursors")
            if config.from_existing and "rt_sigma" in peptides.columns:
                rt_sigma = peptides["rt_sigma"].values
                rt_lambda = peptides["rt_lambda"].values
        else:
            logger.warning("--resume specified but no checkpoints found, running full pipeline")
            resume_after = None

    # Proteome mix setup (if needed)
    if config.proteome_mix and config.multi_fasta_dilution:
        factors = get_dilution_factors(config.multi_fasta_dilution)
    else:
        factors = {}

    if config.from_existing and not resume_after:
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
            # For an Astral build-from-template run the timeline comes from the template
            # (seconds, after the minutes->seconds conversion in AstralAcquisitionBuilder),
            # which can differ from config.gradient_length; rescaling to the config value
            # would smear elution peaks onto the wrong timeline. Use the template-derived
            # length there. Other instruments keep the configured gradient length (their
            # builder derives it from the config/reference) to avoid a silent behavior
            # change to the Bruker path.
            rescale_gradient = (
                acquisition_builder.gradient_length
                if is_thermo_template_instrument(instrument)
                else config.gradient_length
            )
            if not config.silent_mode:
                logger.info(f"Re-scaling retention times to gradient length of {rescale_gradient} seconds")

            # Support both rt_model and deprecated koina_rt_model
            rt_model = config.rt_model or config.koina_rt_model
            # re-scale retention times by running rt simulation again
            peptides = simulate_retention_times(
                peptides=peptides,
                verbose=not config.silent_mode,
                gradient_length=rescale_gradient,
                use_koina_model=rt_model,
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

        # Warn if gradient length mismatch is large. Compare against the template-derived
        # gradient for Astral (authoritative there) and the configured one otherwise
        # (matching the Bruker reference-layout expectation).
        rt_max = peptides['retention_time_gru_predictor'].max()
        grad = (
            acquisition_builder.gradient_length
            if is_thermo_template_instrument(instrument)
            else config.gradient_length
        )
        if abs(rt_max - grad) / grad > 0.05:
            logger.warning(
                f"Existing simulation gradient length ({rt_max}s) differs by >5% "
                f"from acquisition gradient length ({grad}s)"
            )

    # ----------------------------------------
    # Load from search engine findings
    # ----------------------------------------
    findings_result = None
    if config.from_findings and not resume_after:
        logger.info(section_header("Loading Search Engine Findings", use_unicode))
        rt_lower = acquisition_builder.frame_table['time'].min()
        rt_upper = acquisition_builder.frame_table['time'].max()
        findings_result = load_findings(
            findings_path=config.findings_path,
            rt_lower=rt_lower,
            rt_upper=rt_upper,
            mz_lower=acquisition_builder.tdf_writer.helper_handle.mz_lower,
            mz_upper=acquisition_builder.tdf_writer.helper_handle.mz_upper,
            im_lower=acquisition_builder.tdf_writer.helper_handle.im_lower,
            im_upper=acquisition_builder.tdf_writer.helper_handle.im_upper,
            upscale_factor=config.upscale_factor,
            inverse_mobility_std_mean=config.inverse_mobility_std_mean,
            intensity_multiplier=config.intensity_multiplier,
            verbose=not config.silent_mode,
            reference_median=config.findings_reference_median,
        )
        peptides = findings_result.peptides
        proteins = findings_result.proteins
        ions = findings_result.ions  # None if charge was absent

    # ----------------------------------------
    # FASTA processing if not from existing or findings
    # ----------------------------------------
    fastas = get_fasta_file_paths(config.fasta_path) if config.fasta_path else {}
    protein_list, peptide_list = [], []

    if not config.from_existing and not config.from_findings and not resume_after:
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
        # Support both rt_model and deprecated koina_rt_model
        rt_model = config.rt_model or config.koina_rt_model
        if rt_model:
            logger.info(f"  Using RT model: {rt_model}")
        peptides = simulate_retention_times(
            peptides=peptides,
            verbose=not config.silent_mode,
            gradient_length=acquisition_builder.gradient_length,
            use_koina_model=rt_model,
        )

    # Simulate RT for from_findings when not provided in the input
    if config.from_findings and findings_result is not None and not findings_result.has_rt and not resume_after:
        logger.info(section_header("Simulating Retention Times (not in findings)", use_unicode))
        rt_model = config.rt_model or config.koina_rt_model
        if rt_model:
            logger.info(f"  Using RT model: {rt_model}")
        peptides = simulate_retention_times(
            peptides=peptides,
            verbose=not config.silent_mode,
            gradient_length=acquisition_builder.gradient_length,
            use_koina_model=rt_model,
        )

    # Save proteome checkpoint (after data source + RT, before frame distributions)
    if config.enable_checkpoints and not resume_after:
        cp_data = {"proteins": proteins, "peptides": peptides}
        if ions is not None:
            cp_data["ions"] = ions
        checkpoint.save(str(save_path), "proteome", cp_data)

    # Determine number of threads
    num_threads = config.num_threads
    if num_threads == -1:
        num_threads = os.cpu_count()

    if resume_after != "ions":
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
        # Ion property simulation
        # ------------------------------------------------------------------
        # Determine which steps to run based on data source
        need_charge = not config.from_existing and (
            not config.from_findings or findings_result is None or not findings_result.has_charge
        )
        need_im = not config.from_existing and (
            not config.from_findings or findings_result is None or not findings_result.has_im
        )

        if need_charge:
            logger.info(section_header("Simulating Charge States", use_unicode))
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

            if config.proteome_mix:
                ions = ions.drop_duplicates(subset=['sequence', 'charge'])

        if need_im:
            logger.info(section_header("Simulating Ion Mobilities", use_unicode))
            # JOB 6: Ion mobilities
            if config.ccs_model:
                logger.info(f"  Using CCS model: {config.ccs_model}")
            ions = simulate_ion_mobilities_and_variance(
                ions=ions,
                im_lower=acquisition_builder.tdf_writer.helper_handle.im_lower,
                im_upper=acquisition_builder.tdf_writer.helper_handle.im_upper,
                verbose=not config.silent_mode,
                remove_mods=True,
                use_target_mean_std=config.use_inverse_mobility_std_mean,
                target_std_mean=config.inverse_mobility_std_mean,
                use_koina_model=config.ccs_model,
            )

        if not config.from_existing:
            # JOB 7: Precursor isotopic distributions (always needed)
            logger.info(section_header("Simulating Precursor Isotope Patterns", use_unicode))
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

        # JOB 8.5 (opt-in): regenerate the frame/scan distribution columns via the
        # instrument-dispatch projector, BEFORE DDA selection + fragment intensity +
        # assembly so every downstream step reads one consistent set of
        # distributions. Default 'off' is a hard no-op (the legacy columns written
        # above are the fallback). The config validator rejects projection combined
        # with abundance noise or checkpoints (see SimulationConfig._validate).
        projection_mode = str(getattr(config, 'projection_mode', 'off')).lower()
        if projection_mode != 'off':
            from imspy_simulation.timsim.jobs.project_distributions import (
                write_projected_distributions,
            )
            logger.info(section_header(f"Projecting Distributions ({projection_mode})", use_unicode))
            db_path = str(Path(acquisition_builder.path) / 'synthetic_data.db')
            summary = write_projected_distributions(
                db_path,
                mode=projection_mode,
                target_p=config.target_p,
                frame_step_size=config.sampling_step_size,
                scan_step_size=0.0001,  # the scan distribution job's fixed step
                n_steps=config.n_steps,
                remove_epsilon=config.remove_epsilon,
                num_threads=num_threads if num_threads and num_threads > 0 else 4,
            )
            logger.info(
                f"  projector ({projection_mode}): {summary['peptides']} peptides, "
                f"{summary['ions']} ions"
            )

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

        # Save ions checkpoint (before the expensive fragment intensity step)
        if config.enable_checkpoints:
            cp_data = {"proteins": proteins, "peptides": peptides, "ions": ions}
            if pasef_meta is not None:
                cp_data["pasef_meta"] = pasef_meta
            if precursors is not None:
                cp_data["precursors"] = precursors
            checkpoint.save(str(save_path), "ions", cp_data)

    else:
        # Resuming from ions checkpoint — populate DB tables
        logger.info(section_header("Restoring Database from Checkpoint", use_unicode))
        acquisition_builder.synthetics_handle.create_table(table_name='proteins', table=proteins)
        acquisition_builder.synthetics_handle.create_table(table_name='peptides', table=peptides)
        acquisition_builder.synthetics_handle.create_table(table_name='ions', table=ions)
        if pasef_meta is not None:
            acquisition_builder.synthetics_handle.create_table(table_name='pasef_meta', table=pasef_meta)
        if precursors is not None:
            acquisition_builder.synthetics_handle.create_table(table_name='precursors', table=precursors)
        logger.info(f"  Restored: proteins ({len(proteins)}), peptides ({len(peptides)}), ions ({len(ions)})")

    # JOB 9: Simulate fragment intensities
    logger.info(section_header("Simulating Fragment Intensities", use_unicode))
    # Support both intensity_model and deprecated fragment_intensity_model
    intensity_model = config.intensity_model or config.fragment_intensity_model
    if intensity_model:
        logger.info(f"  Using intensity model: {intensity_model}")
    if config.lazy_frame_assembly:
        logger.info("  Using lazy loading for fragment intensity simulation")

    # P6d: the instrument fixes the collision-energy UNIT the run stores/applies.
    # The fragment job validates the selected model's capability accepts it (an
    # eV timsTOF model for an NCE Astral run, or vice-versa, fails loudly).
    from .jobs.register_prediction_set import (
        register_prediction_set,
        resolve_instrument_activation,
        is_sciex_instrument,
        is_waters_instrument,
    )
    # Normalize once (lower-case) so the later exact dispatch comparison can't be
    # fooled by a case variant like 'ORBITRAP_ASTRAL'.
    instrument = str(getattr(config, 'instrument', 'bruker_timstof')).lower()
    activation_method, energy_unit = resolve_instrument_activation(instrument)
    # (instrument validity, Astral⇒DIA, and Astral⇒collision_energy_nce are all
    # enforced at config load in SimulationConfig._validate — fail fast.)
    if instrument.lower() != 'bruker_timstof':
        logger.info(
            f"  Instrument: {instrument} (activation={activation_method}, CE unit={energy_unit})"
        )

    effective_intensity_model = simulate_fragment_intensities(
        path=save_path,
        name=name,
        acquisition_builder=acquisition_builder,
        batch_size=config.batch_size,
        verbose=not config.silent_mode,
        num_threads=num_threads,
        down_sample_factor=config.down_sample_factor,
        dda=config.acquisition_type == 'DDA',
        model_name=intensity_model,
        lazy_loading=config.lazy_frame_assembly,
        frame_batch_size=config.frame_batch_size,
        phospho_mode=config.phospho_mode,
        activation_method=activation_method,
        energy_unit=energy_unit,
    )

    # JOB 9.6 (P5a): register the fragment prediction set — record HOW the
    # fragment intensities were produced (model / instrument / acquisition /
    # activation / CE encoding) and stamp fragment_ions.prediction_set_id.
    # Additive + idempotent: does not change fragment rows or values. A renderer
    # (P5b) uses this to verify stored fragments match the instrument it renders.
    register_prediction_set(
        str(Path(acquisition_builder.path) / 'synthetic_data.db'),
        predictor_model=effective_intensity_model,
        acquisition_type=config.acquisition_type,
        instrument=instrument,
        activation_method=activation_method,
        energy_unit=energy_unit,
        down_sample_factor=config.down_sample_factor,
    )

    # JOB 10: vendor output.
    if is_thermo_template_instrument(instrument):
        # The trunk is simulated on the Thermo template's timeline + windows (fragments
        # at per-window NCE). Author a Thermo .raw from the template (render each frame
        # -> its template slot) — NOT a Bruker .d. Same dispatch for Astral and classic
        # Orbitrap; the writer authors the template's native MS2 centroid format.
        import imspy_connector
        logger.info(section_header(f"Authoring Thermo .raw ({instrument})", use_unicode))
        db_path = acquisition_builder.synthetics_handle.database_path
        out_raw = str(Path(save_path) / f"{name}.raw")
        # Recording-stage m/z noise (Gaussian, ppm). Respect the same toggles the
        # Bruker path uses; 0 ppm = off. A few ppm of mass-error scatter gives a
        # downstream search engine a realistic (non-degenerate) error distribution
        # to mass-calibrate against.
        prec_noise_ppm = config.precursor_noise_ppm if config.mz_noise_precursor else 0.0
        frag_noise_ppm = config.fragment_noise_ppm if config.mz_noise_fragment else 0.0
        if prec_noise_ppm or frag_noise_ppm:
            logger.info(
                f"  m/z noise: precursor {prec_noise_ppm} ppm, fragment {frag_noise_ppm} ppm (Gaussian)"
            )
        # Superimpose simulated peaks onto the template's real signal (real⊕sim)
        # instead of replacing it. 0 ppm = Replace (default, pure simulated output);
        # >0 ppm = Overlay with that MS2 centroid merge tolerance.
        superimpose_ppm = float(config.superimpose_merge_ppm) if config.superimpose_on_reference else 0.0
        if superimpose_ppm:
            logger.info(
                f"  superimpose: overlaying simulated peaks on the template's real "
                f"signal (merge tolerance {superimpose_ppm} ppm)"
            )
        scans, n_ms1, n_ms2, n_ms2_nz, n_cleared, ok = (
            imspy_connector.py_acquisition.write_astral_raw(
                db_path, thermo_template_path(config), out_raw, num_threads,
                precursor_noise_ppm=prec_noise_ppm, fragment_noise_ppm=frag_noise_ppm,
                superimpose_ppm=superimpose_ppm,
            )
        )
        logger.info(
            f"  Astral .raw -> {out_raw}: {scans} scans | {n_ms1} MS1 | "
            f"{n_ms2} MS2 ({n_ms2_nz} non-empty) | {n_cleared} budget-cleared | "
            f"checksum_valid={ok}"
        )
        if not ok:
            raise RuntimeError(
                f"authored Astral .raw failed checksum validation: {out_raw}"
            )
        # mzPROV provenance: sign the authored .raw. Always a JSON sidecar (a vendor
        # .raw can't be embedded into), so the run's embed preference does not apply here.
        if config.emit_provenance:
            emit_provenance_sidecar_raw(
                raw_path=out_raw,
                config_path=cli_args.config,
                experiment_name=name,
                key_path=config.provenance_key_path,
                logger=logger,
            )
    elif is_sciex_instrument(instrument) and config.sciex_native:
        # SCIEX ZenoTOF SWATH, NATIVE: author the synthesized DIA frames into the real .wiff
        # template's .wiff.scan spectra (pure-synthetic native vendor file), instead of mzML.
        # Length-preserving in-place authoring (the template Idx stays valid); the sim schedule
        # fills the template's cycles, extras are cleared. Requires the `sciex` feature.
        import imspy_connector
        logger.info(section_header(f"Authoring native .wiff.scan ({instrument})", use_unicode))
        if not imspy_connector.py_acquisition.has_sciex():
            raise RuntimeError(
                "connector built without the `sciex` feature — rebuild with "
                "`maturin build --features sciex` for native .wiff.scan output, or set "
                "sciex_native=false for open mzML"
            )
        db_path = acquisition_builder.synthetics_handle.database_path
        _wiff = thermo_template_path(config)
        (n_scan, n_ms1, n_ms2, n_ms2_nz, n_auth, n_clear, n_pres, n_verb, n_cyc, n_win, _lp) = (
            imspy_connector.py_acquisition.write_sciex_wiff(
                db_path, _wiff, str(save_path), num_threads,
                fragment_noise_ppm=config.sciex_fragment_noise_ppm,
                precursor_noise_ppm=config.sciex_precursor_noise_ppm,
                overlay_ppm=config.sciex_overlay_ppm,
            )
        )
        out_wiff = str(Path(save_path) / Path(_wiff).name)
        logger.info(
            f"  SCIEX native -> {out_wiff}: {n_scan} scans | {n_ms1} MS1 | {n_ms2} MS2 "
            f"({n_ms2_nz} non-empty) | {n_cyc} cycles x {n_win} windows | "
            f"{n_auth} blocks authored, {n_clear} cleared"
            + (f", {n_verb} kept verbatim (real signal)" if n_verb else "")
        )
    elif is_sciex_instrument(instrument):
        # SCIEX ZenoTOF SWATH: render the synthesized DIA frames to open mzML (the
        # proprietary .wiff.scan spectra are not authored). mzML is readable by
        # DiaNN/alphaDIA and mzprov-signable.
        import imspy_connector
        logger.info(section_header(f"Rendering mzML ({instrument})", use_unicode))
        db_path = acquisition_builder.synthetics_handle.database_path
        out_mzml = str(Path(save_path) / f"{name}.mzML")
        if not imspy_connector.py_acquisition.has_mzml():
            raise RuntimeError(
                "connector built without the `mzml` feature — rebuild with "
                "`maturin build --features mzml` to render SCIEX mzML output"
            )
        scans, n_ms1, n_ms2, n_ms2_nz = imspy_connector.py_acquisition.render_dia_mzml(
            db_path, out_mzml, num_threads
        )
        logger.info(
            f"  SCIEX mzML -> {out_mzml}: {scans} scans | {n_ms1} MS1 | "
            f"{n_ms2} MS2 ({n_ms2_nz} non-empty)"
        )
        # mzPROV provenance: sign the rendered mzML (self-disclosure that this is
        # TimSim-simulated data) via mzprov's mzML signer.
        if config.emit_provenance:
            emit_provenance_sidecar_mzml(
                mzml_path=out_mzml,
                config_path=cli_args.config,
                experiment_name=name,
                embed=config.provenance_embed,
                key_path=config.provenance_key_path,
                logger=logger,
            )
    elif is_waters_instrument(instrument):
        # Waters SONAR: render the synthesized scanning-quadrupole DIA frames to open mzML
        # (no proprietary Waters .raw is authored). Unlike a real SONAR->mzML conversion
        # (whose isolation windows ProteoWizard leaves full-range), we write correct per-
        # window isolation, so the output is proper "normal DIA" mzML for DiaNN.
        import imspy_connector
        logger.info(section_header(f"Rendering mzML ({instrument})", use_unicode))
        db_path = acquisition_builder.synthetics_handle.database_path
        out_mzml = str(Path(save_path) / f"{name}.mzML")
        if not imspy_connector.py_acquisition.has_mzml():
            raise RuntimeError(
                "connector built without the `mzml` feature — rebuild with "
                "`maturin build --features mzml` to render Waters SONAR mzML output"
            )
        scans, n_ms1, n_ms2, n_ms2_nz = imspy_connector.py_acquisition.render_dia_mzml(
            db_path, out_mzml, num_threads
        )
        logger.info(
            f"  Waters SONAR mzML -> {out_mzml}: {scans} scans | {n_ms1} MS1 | "
            f"{n_ms2} MS2 ({n_ms2_nz} non-empty)"
        )
        # mzPROV provenance: sign the rendered mzML (self-disclosure that this is
        # TimSim-simulated data) via mzprov's mzML signer.
        if config.emit_provenance:
            emit_provenance_sidecar_mzml(
                mzml_path=out_mzml,
                config_path=cli_args.config,
                experiment_name=name,
                embed=config.provenance_embed,
                key_path=config.provenance_key_path,
                logger=logger,
            )
    else:
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
            superimpose_on_reference=config.superimpose_on_reference,
        )
        # mzPROV provenance: sign the authored Bruker .d (self-disclosure that this is
        # TimSim-simulated data). Bruker-only — mzprov v0 doesn't canonicalize vendor
        # .raw, so the Thermo build-from-template branch above is intentionally skipped.
        if config.emit_provenance:
            emit_provenance_sidecar(
                d_path=os.path.join(save_path, name, f"{name}.d"),
                db_path=str(Path(acquisition_builder.path) / 'synthetic_data.db'),
                config_path=cli_args.config,
                experiment_name=name,
                embed=config.provenance_embed,
                key_path=config.provenance_key_path,
                logger=logger,
            )

    # Collect final statistics
    stats.n_proteins = len(proteins) if proteins is not None else 0
    stats.n_peptides = len(peptides) if peptides is not None else 0
    stats.n_ions = len(ions) if ions is not None else 0
    stats.n_frames = len(acquisition_builder.frame_table)
    stats.acquisition_type = config.acquisition_type
    stats.experiment_name = name
    stats.output_path = str(save_path)

    # Optional: Generate preview video for visual inspection
    if config.generate_preview_video:
        if VIDEO_GENERATION_AVAILABLE:
            logger.info(section_header("Generating Preview Video", use_unicode))
            data_path = os.path.join(save_path, name, f"{name}.d")
            video_path = os.path.join(save_path, f"{name}_preview.mp4")
            mode = 'dda' if config.acquisition_type == 'DDA' else 'dia'
            try:
                generate_preview_video(
                    data_path=data_path,
                    output_path=video_path,
                    mode=mode,
                    max_frames=config.preview_video_max_frames,
                    fps=config.preview_video_fps,
                    dpi=config.preview_video_dpi,
                    annotate=config.preview_video_annotate,
                    use_bruker_sdk=use_bruker_sdk,
                    show_progress=not config.silent_mode,
                )
                logger.info(f"  Preview video saved to: {video_path}")
            except Exception as e:
                logger.warning(f"  Failed to generate preview video: {e}")
        else:
            logger.warning("  Video generation requested but imspy.vis.frame_rendering not available")

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
