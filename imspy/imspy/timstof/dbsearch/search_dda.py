"""
DDA-PASEF Database Search Pipeline

Clean, TOML-config-only pipeline for DDA database search on timsTOF data.
Designed to mirror the timsim simulator's configuration style.

Example usage:
    imspy_search config.toml

Configuration file structure:
    [paths]
    data_path = "/path/to/raw_data"
    fasta_path = "/path/to/proteome.fasta"
    output_path = "/path/to/output"

    [search]
    precursor_tolerance_ppm = 15.0
    fragment_tolerance_ppm = 20.0
    ...
"""

import os
import sys
import time
import logging
import argparse
import hashlib
import json

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import toml
import numpy as np

from tabulate import tabulate

# Suppress TensorFlow warnings before import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from importlib.metadata import version as get_version
    __version__ = get_version("imspy")
except Exception:
    __version__ = "0.3.23"

# Sagepy imports
from sagepy.core import (
    Precursor,
    Tolerance,
    SpectrumProcessor,
    Scorer,
    EnzymeBuilder,
    SageSearchConfiguration,
    PredictedIntensityStore,
    IndexedDatabase,
)
from sagepy.core.scoring import (
    associate_fragment_ions_with_prosit_predicted_intensities,
    ScoreType,
)
from sagepy.qfdr.tdc import target_decoy_competition_pandas
from sagepy.core.fdr import sage_fdr_psm
from sagepy.utility import psm_collection_to_pandas

# imspy imports
from imspy.timstof import TimsDatasetDDA
from imspy.algorithm.ccs.predictors import DeepPeptideIonMobilityApex, load_deep_ccs_predictor
from imspy.algorithm.utility import load_tokenizer_from_resources
from imspy.algorithm.rt.predictors import DeepChromatographyApex, load_deep_retention_time_predictor
from imspy.algorithm.intensity.predictors import (
    Prosit2023TimsTofWrapper,
    get_collision_energy_calibration_factor,
)

from imspy.timstof.dbsearch.utility import (
    sanitize_mz,
    sanitize_charge,
    get_searchable_spec,
    get_searchable_specs_batch,
    linear_map,
    check_memory,
)

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging for the search pipeline."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    handlers.append(console_handler)

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
# Timer and Stats tracking
# ----------------------------------------------------------------------

@dataclass
class SearchTimer:
    """Track timing for search pipeline steps."""
    start_time: float = field(default_factory=time.time)
    step_times: Dict[str, float] = field(default_factory=dict)
    _current_step: Optional[str] = None
    _step_start: float = 0.0

    def start_step(self, name: str) -> None:
        """Start timing a step."""
        self._current_step = name
        self._step_start = time.time()
        logger.info(f"  Starting: {name}")

    def end_step(self) -> None:
        """End timing the current step."""
        if self._current_step:
            elapsed = time.time() - self._step_start
            self.step_times[self._current_step] = elapsed
            logger.info(f"  Completed: {self._current_step} ({self.format_duration(elapsed)})")
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
class SearchStats:
    """Track statistics for search summary."""
    n_raw_files: int = 0
    n_spectra: int = 0
    n_psms_total: int = 0
    n_psms_filtered: int = 0
    n_peptides: int = 0
    experiment_name: str = ""
    output_path: str = ""


# ----------------------------------------------------------------------
# GPU Configuration
# ----------------------------------------------------------------------

def configure_gpu_memory(memory_limit_gb: int = 4, use_gpu: bool = True) -> None:
    """Configure TensorFlow GPU memory usage."""
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("  GPU disabled, using CPU only")
        return

    import tensorflow as tf

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
    """Load a TOML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as config_file:
        return toml.load(config_file)


def get_default_settings() -> dict:
    """Return default values for all configuration options."""
    return {
        # Paths
        'data_path': None,
        'fasta_path': None,
        'output_path': None,

        # Experiment
        'experiment_name': f'DDA-Search-{int(time.time())}',

        # Modifications
        'variable_modifications': {'M': ['[UNIMOD:35]'], '[': ['[UNIMOD:1]']},
        'static_modifications': {'C': '[UNIMOD:4]'},

        # Enzyme/Digest settings
        'missed_cleavages': 2,
        'min_peptide_len': 7,
        'max_peptide_len': 30,
        'cleave_at': 'KR',
        'restrict': 'P',
        'c_terminal': True,

        # Database settings
        'generate_decoys': True,
        'shuffle_decoys': False,
        'keep_ends': True,
        'bucket_size': 16384,
        'database_path': None,  # Path to save/load indexed database (.sagdb)

        # Search tolerances
        'precursor_tolerance_ppm': 15.0,
        'precursor_tolerance_da': None,
        'fragment_tolerance_ppm': 20.0,
        'fragment_tolerance_da': None,
        'isolation_window_lower': -3.0,
        'isolation_window_upper': 3.0,

        # Scoring (hyperscore, openmshyperscore, weightedhyperscore, weightedopenmshyperscore, betascore)
        'score_type': 'hyperscore',
        'report_psms': 5,
        'min_matched_peaks': 5,
        'max_fragment_charge': 2,
        'min_isotope_err': -1,
        'max_isotope_err': 3,

        # Preprocessing
        'take_top_n': 150,

        # FDR settings
        'fdr_threshold': 0.01,
        'fdr_method': 'psm',
        'remove_decoys_output': True,

        # Prediction settings
        'predict_intensities': True,
        'predict_rt': True,
        'predict_im': True,
        'refine_rt': False,
        'refine_im': False,
        'collision_energy_calibration_sample_size': 256,

        # Performance
        'num_threads': -1,
        'batch_size': 2048,

        # GPU
        'use_gpu': True,
        'gpu_memory_limit_gb': 4,

        # Data loading
        'in_memory': False,
        'use_bruker_sdk': True,

        # Logging
        'log_level': 'INFO',
        'log_file': None,
        'verbose': True,

        # Intensity Store Settings (for weighted scoring)
        'intensity_store_path': None,           # Path to existing .sagi file
        'intensity_prediction_charges': [2, 3], # Charge states for prediction
        'intensity_prediction_exclude_decoys': False,  # MUST be False for production!
        'intensity_cache_dir': None,            # Cache directory (default: output_path)
        'skip_intensity_cache': False,          # Force regeneration
        'use_v1_format': False,                 # Use V1 positional format (legacy)

        # Linear CE Model Settings (CE = intercept + slope * m/z)
        'use_linear_ce_model': False,           # Use linear CE model instead of constant CE
        'ce_intercept': 20.0,                   # CE intercept (NCE at m/z=0)
        'ce_slope': 0.015,                      # CE slope (NCE per Da)
        'extract_ce_from_data': False,          # Extract CE model from first raw file

        # Batch Processing (Rust parallel preprocessing)
        'use_batch_preprocessing': True,        # Use Rust parallel preprocessing (faster, ~24% speedup)
    }


class SearchConfig:
    """
    Configuration container for DDA search parameters.

    Loads configuration from a TOML file and provides attribute-style access.
    """

    def __init__(self, config_path: str):
        """
        Initialize configuration from a TOML file.

        Args:
            config_path: Path to the TOML configuration file.
        """
        self._config = get_default_settings()

        # Load TOML config and flatten sections
        raw_config = load_toml_config(config_path)
        for section_key, section_value in raw_config.items():
            if isinstance(section_value, dict):
                # Handle nested modification dicts specially
                if section_key in ('variable_modifications', 'static_modifications'):
                    self._config[section_key] = section_value
                else:
                    self._config.update(section_value)
            else:
                self._config[section_key] = section_value

        self._validate()

    def _validate(self) -> None:
        """Validate that required configuration options are set."""
        required = ['data_path', 'fasta_path']
        missing = [key for key in required if not self._config.get(key)]

        if missing:
            raise ValueError(f"Missing required configuration options: {', '.join(missing)}")

        # Validate paths exist
        if not os.path.exists(self._config['data_path']):
            raise ValueError(f"Data path does not exist: {self._config['data_path']}")
        if not os.path.exists(self._config['fasta_path']):
            raise ValueError(f"FASTA path does not exist: {self._config['fasta_path']}")

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
        return f"SearchConfig({self._config})"


# ----------------------------------------------------------------------
# UI Helpers
# ----------------------------------------------------------------------

def banner(use_unicode: bool = True) -> str:
    """Return the application banner."""
    if use_unicode:
        return """
+------------------------------------------------------------------------------+
|                                                                              |
|   IMSPY-SEARCH: DDA-PASEF Database Search Pipeline                           |
|                                                                              |
+------------------------------------------------------------------------------+
"""
    else:
        return """
+------------------------------------------------------------------------------+
|                                                                              |
|   IMSPY-SEARCH: DDA-PASEF Database Search Pipeline                           |
|                                                                              |
+------------------------------------------------------------------------------+
"""


def section_header(title: str) -> str:
    """Return a formatted section header."""
    width = 78
    return f"\n{'-' * width}\n  > {title}\n{'-' * width}"


def completion_banner() -> str:
    """Return the completion banner."""
    return """
+------------------------------------------------------------------------------+
|                                                                              |
|   SEARCH COMPLETED SUCCESSFULLY                                              |
|                                                                              |
+------------------------------------------------------------------------------+
"""


# ----------------------------------------------------------------------
# Core Search Functions
# ----------------------------------------------------------------------

def create_indexed_database(
    fasta_content: str,
    config: SearchConfig,
) -> IndexedDatabase:
    """Create an indexed database from FASTA content.

    With V2 key-based intensity stores, database indices no longer need to be
    stable between runs. The V2 format uses (raw_sequence, charge) as lookup keys
    instead of peptide indices, enabling:
    - Database regeneration each run (no need to save/load)
    - Intensity store reuse across different database configurations
    - Flexible parallel workflows

    If config.database_path is specified (legacy V1 support):
    - If the file exists, loads the database from disk
    - If the file doesn't exist, builds from FASTA and saves to that path

    Args:
        fasta_content: The FASTA file content as a string.
        config: Search configuration object.

    Returns:
        IndexedDatabase: The indexed database ready for searching.
    """
    # If database_path is provided and file exists, load from disk
    if config.database_path is not None and os.path.exists(config.database_path):
        logger.info(f"  Loading indexed database from: {config.database_path}")
        return IndexedDatabase.load(config.database_path)

    # Build database from FASTA
    enzyme_builder = EnzymeBuilder(
        missed_cleavages=config.missed_cleavages,
        min_len=config.min_peptide_len,
        max_len=config.max_peptide_len,
        cleave_at=config.cleave_at,
        restrict=config.restrict,
        c_terminal=config.c_terminal,
    )

    sage_config = SageSearchConfiguration(
        fasta=fasta_content,
        static_mods=config.static_modifications,
        variable_mods=config.variable_modifications,
        enzyme_builder=enzyme_builder,
        generate_decoys=config.generate_decoys,
        bucket_size=config.bucket_size,
        shuffle_decoys=config.shuffle_decoys,
        keep_ends=config.keep_ends,
    )

    indexed_db = sage_config.generate_indexed_database()

    # Save to disk if database_path is provided
    if config.database_path is not None:
        logger.info(f"  Saving indexed database to: {config.database_path}")
        indexed_db.save(config.database_path)

    return indexed_db


def create_scorer(config: SearchConfig) -> Scorer:
    """Create a Scorer instance from configuration."""
    # Precursor tolerance
    if config.precursor_tolerance_da is not None:
        prec_tol = Tolerance(da=(-abs(config.precursor_tolerance_da), abs(config.precursor_tolerance_da)))
    else:
        ppm = config.precursor_tolerance_ppm
        prec_tol = Tolerance(ppm=(-abs(ppm), abs(ppm)))

    # Fragment tolerance
    if config.fragment_tolerance_da is not None:
        frag_tol = Tolerance(da=(-abs(config.fragment_tolerance_da), abs(config.fragment_tolerance_da)))
    else:
        ppm = config.fragment_tolerance_ppm
        frag_tol = Tolerance(ppm=(-abs(ppm), abs(ppm)))

    score_type = ScoreType(config.score_type)

    return Scorer(
        precursor_tolerance=prec_tol,
        fragment_tolerance=frag_tol,
        report_psms=config.report_psms,
        min_matched_peaks=config.min_matched_peaks,
        max_fragment_charge=config.max_fragment_charge,
        score_type=score_type,
        variable_mods=config.variable_modifications,
        static_mods=config.static_modifications,
        min_isotope_err=config.min_isotope_err,
        max_isotope_err=config.max_isotope_err,
    )


def compute_database_hash(fasta_content: str, config: SearchConfig) -> str:
    """
    Compute a deterministic hash for the database configuration.

    This hash is used as a cache key for intensity store files, enabling
    reuse of predictions across searches with the same database.

    The hash includes:
    - FASTA content
    - Modification settings
    - Enzyme/digest settings
    - Decoy generation settings

    Args:
        fasta_content: The raw FASTA file content.
        config: Search configuration object.

    Returns:
        A hex string hash representing the database configuration.
    """
    hash_input = {
        'fasta_hash': hashlib.md5(fasta_content.encode()).hexdigest(),
        'static_mods': config.static_modifications,
        'variable_mods': config.variable_modifications,
        'missed_cleavages': config.missed_cleavages,
        'min_peptide_len': config.min_peptide_len,
        'max_peptide_len': config.max_peptide_len,
        'cleave_at': config.cleave_at,
        'restrict': config.restrict,
        'c_terminal': config.c_terminal,
        'generate_decoys': config.generate_decoys,
        'shuffle_decoys': config.shuffle_decoys,
        'keep_ends': config.keep_ends,
    }

    # Create deterministic JSON string
    json_str = json.dumps(hash_input, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def is_weighted_score_type(score_type: str) -> bool:
    """Check if the score type requires intensity predictions."""
    weighted_types = {'weightedhyperscore', 'weightedopenmshyperscore', 'betascore'}
    return score_type.lower() in weighted_types


def extract_expected_mods(config: SearchConfig) -> list:
    """Extract list of UNIMOD strings from config modifications.

    Args:
        config: Search configuration containing static and variable modifications.

    Returns:
        List of UNIMOD strings, e.g., ["[UNIMOD:4]", "[UNIMOD:35]", "[UNIMOD:1]"]
    """
    expected_mods = set()

    # Extract from static modifications (dict: residue -> unimod string)
    if config.static_modifications:
        for unimod in config.static_modifications.values():
            if isinstance(unimod, str) and '[UNIMOD:' in unimod:
                expected_mods.add(unimod)

    # Extract from variable modifications (dict: residue -> list of unimod strings)
    if config.variable_modifications:
        for unimod_list in config.variable_modifications.values():
            if isinstance(unimod_list, list):
                for unimod in unimod_list:
                    if isinstance(unimod, str) and '[UNIMOD:' in unimod:
                        expected_mods.add(unimod)
            elif isinstance(unimod_list, str) and '[UNIMOD:' in unimod_list:
                expected_mods.add(unimod_list)

    return list(expected_mods)


def load_or_generate_intensity_store(
    indexed_db: Any,
    fasta_content: str,
    config: SearchConfig,
    timer: SearchTimer,
) -> Optional[PredictedIntensityStore]:
    """
    Load an existing intensity store or generate a new one.

    Supports both V1 (positional) and V2 (key-based) formats:
    - V2 (default): Uses (modified_sequence, charge) as lookup keys, modification-aware
    - V1 (legacy): Uses peptide index as lookup key, requires stable DB indices

    Workflow:
    1. If a store path is explicitly provided, load it (auto-detects V1 or V2)
    2. If weighted scoring is enabled:
       a. Check for a cached store matching the database hash
       b. If cache exists and not skipped, load it
       c. Otherwise, generate predictions and save to cache
    3. If not using weighted scoring, return None

    IMPORTANT: For production use, decoys MUST have real predictions (not uniform 1.0)
    to ensure fair FDR estimation. See INTENSITY_PREDICTION.md for details.

    Args:
        indexed_db: The indexed database.
        fasta_content: The raw FASTA content (for hash computation).
        config: Search configuration.
        timer: Timer for logging.

    Returns:
        PredictedIntensityStore if weighted scoring is enabled, None otherwise.
    """
    use_v1 = getattr(config, 'use_v1_format', False)
    format_name = "V1 (positional)" if use_v1 else "V2 (key-based)"

    # If explicit store path is provided, load it (auto-detects V1 or V2)
    if config.intensity_store_path:
        timer.start_step("Loading intensity store from file")
        store = PredictedIntensityStore(config.intensity_store_path)
        timer.end_step()
        logger.info(f"  Loaded intensity store: {config.intensity_store_path}")
        is_v2 = getattr(store, 'is_key_based', False)
        logger.info(f"  Store format: {'V2 (key-based)' if is_v2 else 'V1 (positional)'}")
        if is_v2:
            logger.info(f"  Entries in store: {store.entry_count}")
        else:
            logger.info(f"  Peptides in store: {store.peptide_count}")
        return store

    # Check if weighted scoring is enabled
    if not is_weighted_score_type(config.score_type):
        return None

    logger.info(f"  Weighted scoring enabled - {format_name} intensity store required")

    # Compute database hash for caching
    db_hash = compute_database_hash(fasta_content, config)
    logger.info(f"  Database hash: {db_hash}")

    # Determine cache directory
    cache_dir = config.intensity_cache_dir or config.output_path
    if cache_dir is None:
        cache_dir = os.path.dirname(config.data_path)

    os.makedirs(cache_dir, exist_ok=True)

    # Cache path depends on format
    if use_v1:
        cache_path = os.path.join(cache_dir, f"intensity_store_v1_{db_hash}.sagi")
    else:
        cache_path = os.path.join(cache_dir, f"intensity_store_v2_{db_hash}.sagi")

    # Check for cached store
    if os.path.exists(cache_path) and not config.skip_intensity_cache:
        timer.start_step(f"Loading cached {format_name} intensity store")
        store = PredictedIntensityStore(cache_path)
        timer.end_step()
        logger.info(f"  Loaded cached store: {cache_path}")
        is_v2 = getattr(store, 'is_key_based', False)
        logger.info(f"  Store format: {'V2 (key-based)' if is_v2 else 'V1 (positional)'}")
        if is_v2:
            logger.info(f"  Entries in store: {store.entry_count}")
        else:
            logger.info(f"  Peptides in store: {store.peptide_count}")
        return store

    # Generate new intensity predictions
    logger.info(f"  ========== STARTING {format_name.upper()} INTENSITY STORE GENERATION ==========")
    logger.info(f"  Generating {format_name} intensity predictions...")
    timer.start_step(f"Generating {format_name} intensity predictions")

    # Get all peptide sequences (targets AND decoys)
    peptide_sequences = indexed_db.peptides_as_string()
    n_peptides = len(peptide_sequences)
    n_decoys = sum(1 for i in range(n_peptides) if indexed_db[i].decoy)
    n_targets = n_peptides - n_decoys

    logger.info(f"  Total peptides: {n_peptides} ({n_targets} targets, {n_decoys} decoys)")

    # Check decoy warning
    if config.intensity_prediction_exclude_decoys:
        logger.warning("  WARNING: exclude_decoys=True will produce UNFAIR FDR estimates!")
        logger.warning("  Set intensity_prediction_exclude_decoys=False for production use.")

    # Extract expected modifications for UNIMOD-annotated sequences
    expected_mods = extract_expected_mods(config)
    if expected_mods:
        logger.info(f"  Expected modifications: {expected_mods}")

    # Build CE model if using linear CE
    ce_model = None
    if getattr(config, 'use_linear_ce_model', False):
        from imspy.algorithm.intensity.collision_energy import CollisionEnergyModel
        ce_model = CollisionEnergyModel(
            intercept=config.ce_intercept,
            slope=config.ce_slope
        )
        logger.info(f"  Using linear CE model: {ce_model}")

    if use_v1:
        # V1: Use positional pipeline
        from imspy.algorithm.intensity.sage_interface import IntensityPredictionPipeline
        pipeline = IntensityPredictionPipeline(indexed_db, expected_mods=expected_mods)
        pipeline.predict_intensities(
            charges=config.intensity_prediction_charges,
            exclude_decoys=config.intensity_prediction_exclude_decoys,
        )
    else:
        # V2: Use key-based pipeline (modification-aware)
        from imspy.algorithm.intensity.sage_interface import IntensityPredictionPipelineV2
        pipeline = IntensityPredictionPipelineV2(indexed_db, expected_mods=expected_mods)
        pipeline.predict_intensities(
            charges=config.intensity_prediction_charges,
            exclude_decoys=config.intensity_prediction_exclude_decoys,
            collision_energy_model=ce_model,
        )

    # Get the intensity store
    store = pipeline.get_intensity_store()
    timer.end_step()

    # Save to cache
    timer.start_step(f"Saving {format_name} intensity store to cache")
    pipeline.save_intensity_store(cache_path)
    timer.end_step()
    logger.info(f"  Saved {format_name} intensity store: {cache_path}")

    if use_v1:
        logger.info(f"  Peptides in store: {store.peptide_count}")
    else:
        logger.info(f"  Entries in store: {store.entry_count}")

    logger.info(f"  ========== {format_name.upper()} INTENSITY STORE GENERATION COMPLETED ==========")

    return store


def load_predictors(config: SearchConfig):
    """Load ML predictors for intensity, RT, and IM."""
    prosit_model = None
    im_predictor = None
    rt_predictor = None

    if config.predict_intensities:
        prosit_model = Prosit2023TimsTofWrapper(verbose=False)

    if config.predict_im:
        im_predictor = DeepPeptideIonMobilityApex(
            load_deep_ccs_predictor(),
            load_tokenizer_from_resources("tokenizer-ptm")
        )

    if config.predict_rt:
        rt_predictor = DeepChromatographyApex(
            load_deep_retention_time_predictor(),
            load_tokenizer_from_resources("tokenizer-ptm"),
            verbose=False
        )

    return prosit_model, im_predictor, rt_predictor


def process_raw_file(
    raw_path: str,
    indexed_db: Any,
    scorer: Scorer,
    prosit_model: Optional[Prosit2023TimsTofWrapper],
    im_predictor: Optional[DeepPeptideIonMobilityApex],
    rt_predictor: Optional[DeepChromatographyApex],
    config: SearchConfig,
    timer: SearchTimer,
    intensity_store: Optional[PredictedIntensityStore] = None,
) -> tuple:
    """Process a single raw file through the search pipeline.

    Returns:
        Tuple of (n_spectra, psm_list) where n_spectra is the number of
        query spectra searched and psm_list is the list of PSMs found.
    """
    ds_name = os.path.basename(raw_path).replace(".d", "")
    logger.info(f"  Processing: {ds_name}")

    # Load dataset
    timer.start_step("Loading dataset")
    dataset = TimsDatasetDDA(
        str(raw_path),
        in_memory=config.in_memory,
        use_bruker_sdk=config.use_bruker_sdk
    )
    timer.end_step()

    rt_min = dataset.meta_data.Time.min() / 60.0
    rt_max = dataset.meta_data.Time.max() / 60.0

    # Preprocessing spectra
    num_threads = config.num_threads if config.num_threads > 0 else os.cpu_count()

    if config.use_batch_preprocessing:
        # Use Rust parallel batch processing (faster)
        timer.start_step("Loading and preprocessing spectra (batch mode)")
        processed_specs = get_searchable_specs_batch(
            dataset=dataset,
            ds_name=ds_name,
            take_top_n=config.take_top_n,
            deisotope=True,
            isolation_window_lower=config.isolation_window_lower,
            isolation_window_upper=config.isolation_window_upper,
            num_threads=num_threads,
        )
        timer.end_step()
        n_spectra = len(processed_specs)
        logger.info(f"  Spectra to score: {n_spectra}")
    else:
        # Sequential Python processing (original code path)
        timer.start_step("Loading PASEF fragments")
        # Bruker SDK requires single thread
        load_threads = 1 if config.use_bruker_sdk else num_threads

        fragments = dataset.get_pasef_fragments(num_threads=load_threads)

        # Aggregate re-fragmented PASEF frames
        fragments = fragments.groupby('precursor_id').agg({
            'frame_id': 'first',
            'time': 'first',
            'precursor_id': 'first',
            'raw_data': 'sum',
            'scan_begin': 'first',
            'scan_end': 'first',
            'isolation_mz': 'first',
            'isolation_width': 'first',
            'collision_energy': 'first',
            'largest_peak_mz': 'first',
            'average_mz': 'first',
            'monoisotopic_mz': 'first',
            'charge': 'first',
            'average_scan': 'first',
            'intensity': 'first',
            'parent_id': 'first',
        })

        # Calculate mobility
        mobility = fragments.apply(lambda r: r.raw_data.get_inverse_mobility_along_scan_marginal(), axis=1)
        fragments['mobility'] = mobility

        # Generate spec_id
        spec_id = fragments.apply(
            lambda r: f"{np.random.randint(int(1e6))}-{r['frame_id']}-{r['precursor_id']}-{ds_name}",
            axis=1
        )
        fragments['spec_id'] = spec_id
        timer.end_step()

        # Create Sage precursors
        timer.start_step("Creating precursor objects")
        iso_tol = Tolerance(da=(config.isolation_window_lower, config.isolation_window_upper))

        sage_precursor = fragments.apply(lambda r: Precursor(
            mz=sanitize_mz(r['monoisotopic_mz'], r['largest_peak_mz']),
            intensity=r['intensity'],
            charge=sanitize_charge(r['charge']),
            isolation_window=iso_tol,
            collision_energy=r.collision_energy,
            inverse_ion_mobility=r.mobility,
        ), axis=1)
        fragments['sage_precursor'] = sage_precursor
        timer.end_step()

        # Preprocess spectra
        timer.start_step("Preprocessing spectra")
        spec_processor = SpectrumProcessor(take_top_n=config.take_top_n, deisotope=True)

        processed_spec = fragments.apply(
            lambda r: get_searchable_spec(
                precursor=r.sage_precursor,
                raw_fragment_data=r.raw_data,
                spec_processor=spec_processor,
                spec_id=r.spec_id,
                time=r['time'],
            ),
            axis=1
        )
        fragments['processed_spec'] = processed_spec
        timer.end_step()

        # Convert to list for consistent API
        processed_specs = list(fragments['processed_spec'].values)
        n_spectra = len(processed_specs)
        logger.info(f"  Spectra to score: {n_spectra}")

    # Database search
    timer.start_step("Searching database")
    psm_dict = scorer.score_collection_psm(
        db=indexed_db,
        spectrum_collection=processed_specs,
        num_threads=num_threads,
        intensity_store=intensity_store,
    )
    timer.end_step()

    # Flatten PSMs
    psm = []
    for _, values in psm_dict.items():
        psm.extend(list(filter(lambda p: p.sage_feature.rank <= config.report_psms, values)))

    # Map RT to [0, 60] for prediction models
    for p in psm:
        p.retention_time_projected = linear_map(p.retention_time, rt_min, rt_max, 0.0, 60.0)

    logger.info(f"  PSMs found: {len(psm)}")

    # Intensity prediction with collision energy
    if prosit_model is not None and len(psm) > 0:
        timer.start_step("Predicting fragment intensities")

        if config.use_linear_ce_model:
            # Use linear CE model: CE = intercept + slope * m/z (no calibration)
            ce_intercept = config.ce_intercept
            ce_slope = config.ce_slope

            if ce_intercept == 0.0 and ce_slope == 0.0:
                # Special case: use raw CE values without modification
                logger.info("  Using raw CE values (no calibration)")
                for ps in psm:
                    ps.collision_energy_calibrated = ps.collision_energy
                logger.info(f"  CE range: {min(p.collision_energy for p in psm):.1f} - {max(p.collision_energy for p in psm):.1f}")
            else:
                # Apply linear model
                for ps in psm:
                    ps.collision_energy_calibrated = ce_intercept + ce_slope * ps.mono_mz_calculated
                logger.info(f"  Using linear CE model: CE = {ce_intercept} + {ce_slope} * m/z")
        else:
            # Calibrate collision energy with constant offset (original behavior)
            sample_size = min(config.collision_energy_calibration_sample_size, len(psm))
            sample = list(sorted(psm, key=lambda x: x.hyperscore, reverse=True))[:sample_size]

            ce_calibration_factor, ce_similarities = get_collision_energy_calibration_factor(
                list(filter(lambda m: m.decoy is not True, sample)),
                prosit_model,
                verbose=config.verbose,
            )

            # Log CE calibration results
            logger.info(f"  CE calibration offset: {ce_calibration_factor}")
            logger.info(f"  Best SA at calibration: {max(ce_similarities):.4f}")
            logger.info(f"  Raw CE range: {min(p.collision_energy for p in psm):.1f} - {max(p.collision_energy for p in psm):.1f}")
            logger.info(f"  Calibrated CE range: {min(p.collision_energy for p in psm) + ce_calibration_factor:.1f} - {max(p.collision_energy for p in psm) + ce_calibration_factor:.1f}")

            for ps in psm:
                ps.collision_energy_calibrated = ps.collision_energy + ce_calibration_factor

        # Predict intensities
        # NOTE: Use raw sequence (without UNIMOD) for ALL PSMs (targets AND decoys).
        # The Prosit model uses ALPHABET_UNMOD which doesn't support modified tokens
        # like C[UNIMOD:4]. Using sequence_modified would cause modified residues to
        # be encoded as index 0, corrupting predictions.
        # For decoys, p.sequence is the reversed sequence that matched (correct).
        intensity_pred = prosit_model.predict_intensities(
            [p.sequence for p in psm],
            np.array([p.charge for p in psm]),
            [p.collision_energy_calibrated for p in psm],
            batch_size=config.batch_size,
            flatten=True,
        )

        psm = associate_fragment_ions_with_prosit_predicted_intensities(
            psm, intensity_pred, num_threads=num_threads
        )
        timer.end_step()

    # Ion mobility prediction
    if im_predictor is not None and len(psm) > 0:
        timer.start_step("Predicting ion mobilities")

        inv_mob = im_predictor.simulate_ion_mobilities(
            sequences=[x.sequence_modified if not x.decoy else x.sequence_decoy_modified for x in psm],
            charges=[x.charge for x in psm],
            mz=[x.mono_mz_calculated for x in psm]
        )

        for mob, ps in zip(inv_mob, psm):
            ps.inverse_ion_mobility_predicted = mob

        # Calibration factor if not refining
        if not config.refine_im:
            im_calibration = np.mean([x.inverse_ion_mobility - x.inverse_ion_mobility_predicted for x in psm])
            for p in psm:
                p.inverse_ion_mobility_predicted += im_calibration

        timer.end_step()

    # Retention time prediction
    if rt_predictor is not None and len(psm) > 0:
        timer.start_step("Predicting retention times")

        rt_pred = rt_predictor.simulate_separation_times(
            sequences=[x.sequence_modified if not x.decoy else x.sequence_decoy_modified for x in psm],
        )

        for rt, p in zip(rt_pred, psm):
            p.retention_time_predicted = rt

        timer.end_step()

    # Set file name for each PSM
    for p in psm:
        p.file_name = ds_name

    return n_spectra, psm


def run_search_pipeline(config: SearchConfig) -> None:
    """Run the complete DDA search pipeline."""
    timer = SearchTimer()
    stats = SearchStats()

    use_unicode = sys.stdout.isatty()

    # Print banner
    print(banner(use_unicode))
    logger.info(f"Version: {__version__}")
    logger.info("")

    # Configure GPU
    logger.info(section_header("GPU Configuration"))
    configure_gpu_memory(
        memory_limit_gb=config.gpu_memory_limit_gb,
        use_gpu=config.use_gpu
    )

    # Check memory
    check_memory(limit_in_gb=16)

    # Find raw files
    logger.info(section_header("Discovering Raw Files"))
    data_path = config.data_path
    raw_paths = []

    if data_path.endswith(".d"):
        raw_paths = [data_path]
    else:
        for root, dirs, _ in os.walk(data_path):
            for d in dirs:
                if d.endswith(".d"):
                    raw_paths.append(os.path.join(root, d))

    logger.info(f"  Found {len(raw_paths)} raw file(s)")
    stats.n_raw_files = len(raw_paths)

    if not raw_paths:
        logger.error("No .d folders found in data path")
        sys.exit(1)

    # Load FASTA
    logger.info(section_header("Loading FASTA Database"))
    fasta_path = config.fasta_path

    if os.path.isdir(fasta_path):
        fasta_files = [os.path.join(fasta_path, f) for f in os.listdir(fasta_path) if f.endswith(".fasta")]
        fasta_content = ""
        for fasta_file in fasta_files:
            with open(fasta_file, 'r') as f:
                fasta_content += f.read()
    else:
        with open(fasta_path, 'r') as f:
            fasta_content = f.read()

    logger.info(f"  FASTA loaded from: {fasta_path}")

    # Create indexed database (or load from disk if database_path is provided)
    if config.database_path and os.path.exists(config.database_path):
        timer.start_step("Loading indexed database from disk")
    else:
        timer.start_step("Building indexed database")
    indexed_db = create_indexed_database(fasta_content, config)
    timer.end_step()
    logger.info(f"  Peptides in database: {len(indexed_db.peptides_as_string())}")
    if config.database_path:
        logger.info(f"  Database path: {config.database_path}")

    # Extract CE model from data if requested
    if config.use_linear_ce_model and config.extract_ce_from_data:
        logger.info(section_header("Extracting CE Model from Data"))
        timer.start_step("Extracting CE model")
        try:
            first_dataset = TimsDatasetDDA(
                raw_paths[0],
                in_memory=False,
                use_bruker_sdk=config.use_bruker_sdk
            )
            meta = first_dataset.pasef_meta_data
            if meta is not None and len(meta) > 0:
                mz = meta['isolation_mz'].values
                ce = meta['collision_energy'].values
                valid = (mz > 0) & (ce > 0) & np.isfinite(mz) & np.isfinite(ce)
                mz, ce = mz[valid], ce[valid]
                if len(mz) >= 2:
                    coeffs = np.polyfit(mz, ce, 1)
                    config._config['ce_slope'] = float(coeffs[0])
                    config._config['ce_intercept'] = float(coeffs[1])
                    logger.info(f"  Extracted CE model: CE = {config.ce_intercept:.2f} + {config.ce_slope:.5f} * m/z")
        except Exception as e:
            logger.warning(f"  Failed to extract CE model: {e}")
            logger.warning(f"  Using default: CE = {config.ce_intercept:.2f} + {config.ce_slope:.5f} * m/z")
        timer.end_step()

    # Load or generate intensity store (for weighted scoring)
    # Only show section header if weighted scoring is used or explicit path is provided
    if is_weighted_score_type(config.score_type) or config.intensity_store_path:
        logger.info(section_header("Intensity Store (V2)"))
        intensity_store = load_or_generate_intensity_store(
            indexed_db=indexed_db,
            fasta_content=fasta_content,
            config=config,
            timer=timer,
        )
        if intensity_store is not None:
            is_v2 = getattr(intensity_store, 'is_key_based', False)
            count = intensity_store.entry_count if is_v2 else intensity_store.peptide_count
            format_str = "V2 key-based" if is_v2 else "V1 positional"
            logger.info(f"  Intensity store: Active ({count} entries, {format_str})")
    else:
        intensity_store = None
        logger.info(f"\n  Scoring mode: Standard ({config.score_type})")

    # Create scorer
    logger.info(section_header("Creating Scorer"))
    scorer = create_scorer(config)
    logger.info(f"  Score type: {config.score_type}")
    logger.info(f"  Precursor tolerance: {config.precursor_tolerance_ppm} ppm")
    logger.info(f"  Fragment tolerance: {config.fragment_tolerance_ppm} ppm")

    # Load predictors
    logger.info(section_header("Loading ML Predictors"))
    prosit_model, im_predictor, rt_predictor = load_predictors(config)
    if prosit_model:
        logger.info("  Intensity predictor: Prosit2023TimsTof")
    if im_predictor:
        logger.info("  Ion mobility predictor: DeepPeptideIonMobilityApex")
    if rt_predictor:
        logger.info("  Retention time predictor: DeepChromatographyApex")

    # Setup output directory
    output_path = config.output_path or os.path.dirname(raw_paths[0])
    output_dir = os.path.join(output_path, "imspy_search")
    os.makedirs(output_dir, exist_ok=True)
    stats.output_path = output_dir

    # Process each raw file
    logger.info(section_header("Processing Raw Files"))
    all_psms = []

    for i, raw_path in enumerate(raw_paths):
        logger.info(f"\n  File {i+1}/{len(raw_paths)}: {os.path.basename(raw_path)}")
        n_spectra, psms = process_raw_file(
            raw_path=raw_path,
            indexed_db=indexed_db,
            scorer=scorer,
            prosit_model=prosit_model,
            im_predictor=im_predictor,
            rt_predictor=rt_predictor,
            config=config,
            timer=timer,
            intensity_store=intensity_store,
        )
        all_psms.extend(psms)
        stats.n_spectra += n_spectra

    stats.n_psms_total = len(all_psms)
    logger.info(f"\n  Total PSMs: {len(all_psms)}")

    # FDR calculation
    logger.info(section_header("FDR Calculation"))
    timer.start_step("Calculating FDR")

    # Use sage FDR
    sage_fdr_psm(all_psms, indexed_db, use_hyper_score=True)

    timer.end_step()

    # Convert to pandas
    psm_df = psm_collection_to_pandas(all_psms)

    # Apply FDR filter
    timer.start_step("Filtering by FDR threshold")
    psm_fdr = target_decoy_competition_pandas(
        psm_df,
        method=config.fdr_method,
        score='hyperscore'
    )

    if config.remove_decoys_output:
        psm_fdr = psm_fdr[psm_fdr.decoy == False]

    psm_fdr = psm_fdr[psm_fdr.q_value <= config.fdr_threshold]

    # Merge filtered results back with full PSM dataframe to get sequence info
    merge_cols = ['spec_idx', 'match_idx', 'decoy']
    psm_filtered = psm_df.merge(
        psm_fdr[merge_cols + ['q_value']],
        on=merge_cols,
        how='inner',
        suffixes=('', '_fdr')
    )
    # Use q_value from FDR calculation
    if 'q_value_fdr' in psm_filtered.columns:
        psm_filtered['q_value'] = psm_filtered['q_value_fdr']
        psm_filtered = psm_filtered.drop(columns=['q_value_fdr'])

    stats.n_psms_filtered = len(psm_filtered)
    stats.n_peptides = psm_filtered['sequence'].nunique() if 'sequence' in psm_filtered.columns else 0
    timer.end_step()

    logger.info(f"  PSMs at {config.fdr_threshold*100}% FDR: {stats.n_psms_filtered}")
    logger.info(f"  Unique peptides: {stats.n_peptides}")

    # Write output
    logger.info(section_header("Writing Output"))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(output_dir, f"PSMs_{timestamp}.csv")
    psm_filtered.to_csv(output_file, index=False)
    logger.info(f"  Written to: {output_file}")

    # Also write full results (before FDR filter)
    full_output_file = os.path.join(output_dir, f"PSMs_all_{timestamp}.csv")
    psm_df.to_csv(full_output_file, index=False)
    logger.info(f"  Full results: {full_output_file}")

    # Save config for reference
    config_output = os.path.join(output_dir, f"config_{timestamp}.txt")
    with open(config_output, 'w') as f:
        table_data = [[k, v] for k, v in config.to_dict().items()]
        f.write(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))

    # Print completion summary
    print(completion_banner())

    total_time = timer.format_duration(timer.total_elapsed())
    logger.info("  Search Summary")
    logger.info(f"  {'-' * 40}")
    logger.info(f"  Raw files:       {stats.n_raw_files}")
    logger.info(f"  Total spectra:   {stats.n_spectra:,}")
    logger.info(f"  PSMs (total):    {stats.n_psms_total:,}")
    logger.info(f"  PSMs (filtered): {stats.n_psms_filtered:,}")
    logger.info(f"  Peptides:        {stats.n_peptides:,}")
    logger.info(f"  {'-' * 40}")
    logger.info(f"  Total time:      {total_time}")
    logger.info(f"  Output:          {stats.output_path}")
    logger.info("")


# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for TOML-only configuration."""
    description = """
+------------------------------------------------------------------------------+
|                                                                              |
|   IMSPY-SEARCH: DDA-PASEF Database Search Pipeline                           |
|                                                                              |
+------------------------------------------------------------------------------+

Run a DDA database search on timsTOF PASEF data. All configuration is provided
via a TOML file.
"""

    epilog = """
------------------------------------------------------------------------------
  Configuration File Structure
------------------------------------------------------------------------------

  [paths]
    data_path   = "/path/to/raw_data"    # .d folder or directory of .d folders
    fasta_path  = "/path/to/proteins.fasta"
    output_path = "/path/to/output"      # optional

  [experiment]
    experiment_name = "MySearch"

  [modifications]
    # Variable modifications (list format)
    M = ["[UNIMOD:35]"]  # Oxidation
    "[" = ["[UNIMOD:1]"] # N-term acetylation

  [static_modifications]
    C = "[UNIMOD:4]"     # Carbamidomethylation

  [enzyme]
    missed_cleavages = 2
    min_peptide_len  = 7
    max_peptide_len  = 30
    cleave_at        = "KR"
    restrict         = "P"

  [database]
    database_path = "/path/to/database.sagdb"  # optional: save/load database

  [search]
    precursor_tolerance_ppm = 15.0
    fragment_tolerance_ppm  = 20.0
    # score_type options: hyperscore, openmshyperscore, weightedhyperscore, weightedopenmshyperscore, betascore
    score_type              = "hyperscore"

  [fdr]
    fdr_threshold        = 0.01
    remove_decoys_output = true

  [performance]
    num_threads = -1
    use_gpu     = true

------------------------------------------------------------------------------
  Example Usage
------------------------------------------------------------------------------

    imspy_search config.toml
    imspy_search /path/to/my-search-config.toml

------------------------------------------------------------------------------
  Author: David Teschner | License: MIT
------------------------------------------------------------------------------
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


def main():
    """Entry point for the DDA search pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load configuration
    try:
        config = SearchConfig(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup logging
    log_file = config.log_file
    if log_file and config.output_path:
        if not os.path.isabs(log_file):
            log_file = os.path.join(config.output_path, log_file)

    setup_logging(log_level=config.log_level, log_file=log_file)

    # Run the pipeline
    run_search_pipeline(config)


if __name__ == "__main__":
    main()
