"""State management for TimSim GUI configuration."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List
import json
import os
import toml


# Config directory for storing recent configs
CONFIG_DIR = Path.home() / ".timsim"
RECENT_CONFIGS_FILE = CONFIG_DIR / "recent_configs.json"
MAX_RECENT_CONFIGS = 10


@dataclass
class PathsConfig:
    """Paths configuration section."""
    save_path: str = ""
    reference_path: str = ""
    fasta_path: str = ""
    modifications: str = ""
    existing_path: str = ""


@dataclass
class ExperimentConfig:
    """Experiment configuration section."""
    experiment_name: str = "TIMSIM-[PLACEHOLDER]"
    acquisition_type: str = "DIA"
    gradient_length: float = 3600.0
    use_reference_layout: bool = True
    reference_in_memory: bool = False
    use_bruker_sdk: bool = True
    apply_fragmentation: bool = True
    from_existing: bool = False
    silent_mode: bool = False


@dataclass
class DigestionConfig:
    """Digestion configuration section."""
    n_proteins: int = 20000
    num_peptides_total: int = 250000
    num_sample_peptides: int = 25000
    sample_peptides: bool = True
    sample_seed: int = 41
    digest_proteins: bool = True
    remove_degenerate_peptides: bool = False
    cleave_at: str = "KR"
    restrict: str = "P"
    missed_cleavages: int = 2
    min_len: int = 7
    max_len: int = 30
    decoys: bool = False
    upscale_factor: int = 100000


@dataclass
class RetentionTimeConfig:
    """Retention time / chromatography configuration section."""
    koina_rt_model: str = ""
    min_rt_percent: float = 2.0
    exclude_accumulated_gradient_start: bool = True
    sigma_lower_rt: float = 1.0
    sigma_upper_rt: float = 2.5
    sigma_alpha_rt: float = 4.0
    sigma_beta_rt: float = 4.0
    k_lower_rt: float = 0.0
    k_upper_rt: float = 10.0
    k_alpha_rt: float = 1.0
    k_beta_rt: float = 20.0
    target_p: float = 0.999
    sampling_step_size: float = 0.001
    n_steps: int = 1000
    remove_epsilon: float = 1e-4


@dataclass
class IonMobilityConfig:
    """Ion mobility configuration section."""
    use_inverse_mobility_std_mean: bool = True
    inverse_mobility_std_mean: float = 0.009


@dataclass
class ChargeStatesConfig:
    """Charge state configuration section."""
    p_charge: float = 0.8
    max_charge: int = 4
    min_charge_contrib: float = 0.005
    binomial_charge_model: bool = False
    normalize_charge_states: bool = True
    charge_state_one_probability: float = 0.0


@dataclass
class IsotopicPatternConfig:
    """Isotopic pattern configuration section."""
    isotope_k: int = 8
    isotope_min_intensity: int = 1
    isotope_centroid: bool = True


@dataclass
class ModelsConfig:
    """Prediction model selection configuration section.

    Available options:
    - rt_model: "" (local PyTorch), "Deeplc_hela_hf", "Chronologer_RT",
                "AlphaPeptDeep_rt_generic", "Prosit_2019_irt"
    - ccs_model: "" (local PyTorch), "AlphaPeptDeep_ccs_generic", "IM2Deep"
    - intensity_model: "" or "prosit" (Prosit_2023_intensity_timsTOF via Koina),
                       "alphapeptdeep" (AlphaPeptDeep_ms2_generic - supports phospho!),
                       "ms2pip" (ms2pip_timsTOF2024)
    """
    rt_model: str = ""
    ccs_model: str = ""
    intensity_model: str = ""


@dataclass
class FragmentIntensityConfig:
    """Fragment intensity prediction configuration section."""
    fragment_intensity_model: str = ""  # Deprecated: use ModelsConfig.intensity_model
    down_sample_factor: float = 0.5


@dataclass
class QuadTransmissionConfig:
    """Quadrupole transmission configuration section."""
    quad_isotope_transmission_mode: str = "none"
    quad_transmission_min_probability: float = 0.5
    quad_transmission_max_isotopes: int = 10


@dataclass
class NoiseConfig:
    """Noise configuration section."""
    noise_frame_abundance: bool = False
    noise_scan_abundance: bool = False
    mz_noise_precursor: bool = False
    precursor_noise_ppm: float = 5.0
    mz_noise_fragment: bool = False
    fragment_noise_ppm: float = 5.0
    mz_noise_uniform: bool = False
    add_real_data_noise: bool = False
    reference_noise_intensity_max: float = 30.0
    precursor_sample_fraction: float = 0.2
    fragment_sample_fraction: float = 0.2
    num_precursor_noise_frames: int = 5
    num_fragment_noise_frames: int = 5


@dataclass
class VariationConfig:
    """Variation configuration section."""
    re_scale_rt: bool = False
    rt_variation_std: str = ""
    ion_mobility_variation_std: str = ""
    intensity_variation_std: str = ""


@dataclass
class AcquisitionConfig:
    """Acquisition configuration section."""
    round_collision_energy: bool = True
    collision_energy_decimals: int = 0


@dataclass
class DDAConfig:
    """DDA-specific configuration section."""
    precursors_every: int = 10
    precursor_intensity_threshold: int = 500
    max_precursors: int = 25
    exclusion_width: int = 25
    selection_mode: str = "topN"


@dataclass
class ProteomeMixConfig:
    """Proteome mix configuration section."""
    proteome_mix: bool = False
    multi_fasta_dilution: str = ""


@dataclass
class PhosphorylationConfig:
    """Phosphorylation configuration section."""
    phospho_mode: bool = False


@dataclass
class PerformanceConfig:
    """Performance configuration section."""
    num_threads: int = -1
    batch_size: int = 256
    frame_batch_size: int = 500
    lazy_frame_assembly: bool = False
    use_gpu: bool = True
    gpu_memory_limit_gb: int = 4


@dataclass
class LoggingConfig:
    """Logging configuration section."""
    log_level: str = "INFO"
    log_file: str = ""


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    digestion: DigestionConfig = field(default_factory=DigestionConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    retention_time: RetentionTimeConfig = field(default_factory=RetentionTimeConfig)
    ion_mobility: IonMobilityConfig = field(default_factory=IonMobilityConfig)
    charge_states: ChargeStatesConfig = field(default_factory=ChargeStatesConfig)
    isotopic_pattern: IsotopicPatternConfig = field(default_factory=IsotopicPatternConfig)
    fragment_intensity: FragmentIntensityConfig = field(default_factory=FragmentIntensityConfig)
    quad_transmission: QuadTransmissionConfig = field(default_factory=QuadTransmissionConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    variation: VariationConfig = field(default_factory=VariationConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    dda: DDAConfig = field(default_factory=DDAConfig)
    proteome_mix: ProteomeMixConfig = field(default_factory=ProteomeMixConfig)
    phosphorylation: PhosphorylationConfig = field(default_factory=PhosphorylationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_toml_dict(self) -> dict:
        """Convert configuration to TOML-compatible dictionary."""
        return {
            "paths": asdict(self.paths),
            "experiment": asdict(self.experiment),
            "digestion": asdict(self.digestion),
            "models": asdict(self.models),
            "retention_time": asdict(self.retention_time),
            "ion_mobility": asdict(self.ion_mobility),
            "charge_states": asdict(self.charge_states),
            "isotopic_pattern": asdict(self.isotopic_pattern),
            "fragment_intensity": asdict(self.fragment_intensity),
            "quad_transmission": asdict(self.quad_transmission),
            "noise": asdict(self.noise),
            "variation": asdict(self.variation),
            "acquisition": asdict(self.acquisition),
            "dda": asdict(self.dda),
            "proteome_mix": asdict(self.proteome_mix),
            "phosphorylation": asdict(self.phosphorylation),
            "performance": asdict(self.performance),
            "logging": asdict(self.logging),
        }

    def save_toml(self, path: str) -> None:
        """Save configuration to TOML file."""
        with open(path, "w") as f:
            toml.dump(self.to_toml_dict(), f)

    @classmethod
    def from_toml(cls, path: str) -> "SimulationConfig":
        """Load configuration from TOML file."""
        with open(path, "r") as f:
            data = toml.load(f)

        config = cls()

        # Map sections to config objects
        section_map = {
            "paths": (config.paths, PathsConfig),
            "experiment": (config.experiment, ExperimentConfig),
            "digestion": (config.digestion, DigestionConfig),
            "models": (config.models, ModelsConfig),
            "retention_time": (config.retention_time, RetentionTimeConfig),
            "ion_mobility": (config.ion_mobility, IonMobilityConfig),
            "charge_states": (config.charge_states, ChargeStatesConfig),
            "isotopic_pattern": (config.isotopic_pattern, IsotopicPatternConfig),
            "fragment_intensity": (config.fragment_intensity, FragmentIntensityConfig),
            "quad_transmission": (config.quad_transmission, QuadTransmissionConfig),
            "noise": (config.noise, NoiseConfig),
            "variation": (config.variation, VariationConfig),
            "acquisition": (config.acquisition, AcquisitionConfig),
            "dda": (config.dda, DDAConfig),
            "proteome_mix": (config.proteome_mix, ProteomeMixConfig),
            "phosphorylation": (config.phosphorylation, PhosphorylationConfig),
            "performance": (config.performance, PerformanceConfig),
            "logging": (config.logging, LoggingConfig),
        }

        # Also handle legacy section names from old configs
        legacy_map = {
            "main_settings": "paths",
            "peptide_digestion": "digestion",
            "peptide_intensity": "digestion",
            "distribution_settings": "retention_time",
            "charge_state_probabilities": "charge_states",
            "noise_settings": "noise",
            "variation_settings": "variation",
            "acquisition_settings": "acquisition",
            "phosphorylation_settings": "phosphorylation",
            "performance_settings": "performance",
            "logging_settings": "logging",
        }

        for section_name, section_data in data.items():
            # Handle legacy section names
            target_section = legacy_map.get(section_name, section_name)
            if target_section in section_map:
                config_obj, config_class = section_map[target_section]
                for key, value in section_data.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)

        return config


def get_recent_configs() -> List[dict]:
    """Get list of recent config files with metadata."""
    if not RECENT_CONFIGS_FILE.exists():
        return []

    try:
        with open(RECENT_CONFIGS_FILE, "r") as f:
            configs = json.load(f)
        # Filter out non-existent files
        return [c for c in configs if Path(c.get("path", "")).exists()]
    except Exception:
        return []


def add_recent_config(path: str, name: str = "") -> None:
    """Add a config to recent list."""
    CONFIG_DIR.mkdir(exist_ok=True)

    configs = get_recent_configs()

    # Remove if already exists
    configs = [c for c in configs if c.get("path") != path]

    # Add to front
    configs.insert(0, {
        "path": path,
        "name": name or Path(path).stem,
        "timestamp": str(Path(path).stat().st_mtime) if Path(path).exists() else "",
    })

    # Keep only recent
    configs = configs[:MAX_RECENT_CONFIGS]

    try:
        with open(RECENT_CONFIGS_FILE, "w") as f:
            json.dump(configs, f, indent=2)
    except Exception:
        pass


def find_config_in_path(path: str) -> Optional[str]:
    """Find a config file in a given path.

    Supports:
    - Direct TOML file path
    - Simulation output directory (looks for *_config.toml, config.toml)
    - .d folder (looks in parent for config)

    Returns:
        Path to config file if found, None otherwise.
    """
    p = Path(path)

    if not p.exists():
        return None

    # Direct TOML file
    if p.is_file() and p.suffix.lower() == ".toml":
        return str(p)

    # Directory - search for config files
    if p.is_dir():
        search_dirs = [p]

        # If it's a .d folder, also search parent
        if p.suffix == ".d":
            search_dirs.append(p.parent)

        for search_dir in search_dirs:
            # Look for config files in order of preference
            patterns = [
                "*_config.toml",  # e.g., test_config.toml
                "config.toml",
                "validate_config.toml",
                "*.toml",
            ]

            for pattern in patterns:
                matches = list(search_dir.glob(pattern))
                if matches:
                    # Return the most recently modified
                    return str(max(matches, key=lambda x: x.stat().st_mtime))

    return None


class SimulationState:
    """Shared state for the simulation GUI."""

    def __init__(self):
        self.config = SimulationConfig()
        self.mode: str = "simple"  # "simple" or "advanced"
        self.current_step: int = 0  # For wizard navigation
        self.is_running: bool = False
        self.log_lines: list[str] = []
        self.progress: float = 0.0
        self.preset: str = "standard"  # "quick", "standard", "high_fidelity"
        self.loaded_config_path: Optional[str] = None

    def apply_preset(self, preset: str) -> None:
        """Apply a preset configuration."""
        self.preset = preset

        if preset == "quick":
            # Quick mode: fewer peptides, faster settings
            self.config.digestion.num_sample_peptides = 5000
            self.config.digestion.n_proteins = 5000
            self.config.performance.batch_size = 128
            self.config.experiment.gradient_length = 1800.0
            self.config.noise.add_real_data_noise = False

        elif preset == "standard":
            # Standard mode: balanced settings
            self.config.digestion.num_sample_peptides = 25000
            self.config.digestion.n_proteins = 20000
            self.config.performance.batch_size = 256
            self.config.experiment.gradient_length = 3600.0
            self.config.noise.add_real_data_noise = False

        elif preset == "high_fidelity":
            # High fidelity mode: more peptides, noise enabled
            self.config.digestion.num_sample_peptides = 50000
            self.config.digestion.n_proteins = 50000
            self.config.performance.batch_size = 512
            self.config.experiment.gradient_length = 7200.0
            self.config.noise.add_real_data_noise = True
            self.config.noise.mz_noise_precursor = True
            self.config.noise.mz_noise_fragment = True

    def reset(self) -> None:
        """Reset state to defaults."""
        self.config = SimulationConfig()
        self.current_step = 0
        self.is_running = False
        self.log_lines = []
        self.progress = 0.0
        self.preset = "standard"
        self.loaded_config_path = None

    def load_config(self, path: str) -> tuple[bool, str]:
        """Load configuration from file or directory.

        Args:
            path: Path to TOML file, simulation directory, or .d folder

        Returns:
            Tuple of (success, message)
        """
        # Find config file if path is a directory
        config_path = find_config_in_path(path)

        if not config_path:
            return False, f"No config file found in: {path}"

        try:
            self.config = SimulationConfig.from_toml(config_path)
            self.loaded_config_path = config_path
            add_recent_config(config_path, Path(config_path).stem)
            return True, f"Loaded config from: {config_path}"
        except Exception as e:
            return False, f"Error loading config: {e}"

    def save_config(self, path: str) -> tuple[bool, str]:
        """Save configuration to file.

        Returns:
            Tuple of (success, message)
        """
        try:
            # Ensure .toml extension
            if not path.endswith(".toml"):
                path = path + ".toml"

            self.config.save_toml(path)
            self.loaded_config_path = path
            add_recent_config(path, Path(path).stem)
            return True, f"Config saved to: {path}"
        except Exception as e:
            return False, f"Error saving config: {e}"

    def validate(self) -> tuple[bool, List[str], List[str]]:
        """Validate the current configuration.

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # ===== Required Paths =====
        if not self.config.paths.save_path:
            errors.append("Output directory is required")
        elif not Path(self.config.paths.save_path).parent.exists():
            errors.append(f"Parent directory for output does not exist: {Path(self.config.paths.save_path).parent}")

        if not self.config.paths.reference_path:
            errors.append("Reference dataset is required")
        else:
            ref_path = Path(self.config.paths.reference_path)
            if not ref_path.exists():
                errors.append(f"Reference path does not exist: {ref_path}")
            elif not ref_path.is_dir():
                errors.append(f"Reference path must be a directory (.d folder): {ref_path}")
            elif ref_path.suffix != ".d":
                warnings.append(f"Reference path should be a .d folder: {ref_path}")

        if not self.config.paths.fasta_path:
            errors.append("FASTA file is required")
        else:
            fasta_path = Path(self.config.paths.fasta_path)
            if not fasta_path.exists():
                errors.append(f"FASTA path does not exist: {fasta_path}")
            elif fasta_path.is_file() and fasta_path.suffix.lower() not in [".fasta", ".fa", ".faa"]:
                warnings.append(f"FASTA file has unusual extension: {fasta_path.suffix}")

        # ===== Optional Paths =====
        if self.config.paths.modifications:
            mod_path = Path(self.config.paths.modifications)
            if not mod_path.exists():
                errors.append(f"Modifications file does not exist: {mod_path}")

        # ===== Numeric Validation =====
        if self.config.digestion.num_sample_peptides < 1:
            errors.append("Number of sample peptides must be at least 1")

        if self.config.digestion.num_sample_peptides > self.config.digestion.num_peptides_total:
            warnings.append("Sample peptides exceeds total peptides - will use all available")

        if self.config.digestion.min_len < 1:
            errors.append("Minimum peptide length must be at least 1")

        if self.config.digestion.max_len < self.config.digestion.min_len:
            errors.append("Maximum peptide length must be greater than minimum")

        if self.config.digestion.max_len > 60:
            warnings.append("Maximum peptide length > 60 may cause issues with some models")

        if self.config.digestion.missed_cleavages < 0:
            errors.append("Missed cleavages cannot be negative")

        if self.config.digestion.missed_cleavages > 5:
            warnings.append("More than 5 missed cleavages may significantly increase peptide count")

        # ===== Charge State Validation =====
        if self.config.charge_states.max_charge < 1:
            errors.append("Maximum charge state must be at least 1")

        if self.config.charge_states.max_charge > 6:
            warnings.append("Charge states > 6 are unusual for most peptides")

        if not 0 <= self.config.charge_states.p_charge <= 1:
            errors.append("Charge probability must be between 0 and 1")

        # ===== EMG Parameters =====
        if self.config.retention_time.sigma_lower_rt > self.config.retention_time.sigma_upper_rt:
            errors.append("Sigma lower bound must be less than upper bound")

        if self.config.retention_time.k_lower_rt > self.config.retention_time.k_upper_rt:
            errors.append("K lower bound must be less than upper bound")

        # ===== Noise Validation =====
        if self.config.noise.precursor_noise_ppm < 0:
            errors.append("Precursor noise PPM cannot be negative")

        if self.config.noise.fragment_noise_ppm < 0:
            errors.append("Fragment noise PPM cannot be negative")

        if self.config.noise.precursor_noise_ppm > 50:
            warnings.append("Precursor noise > 50 ppm is unusually high")

        if self.config.noise.fragment_noise_ppm > 50:
            warnings.append("Fragment noise > 50 ppm is unusually high")

        # ===== Performance Validation =====
        if self.config.performance.num_threads == 0:
            errors.append("Number of threads cannot be 0 (use -1 for auto)")

        if self.config.performance.batch_size < 1:
            errors.append("Batch size must be at least 1")

        if self.config.performance.gpu_memory_limit_gb < 1:
            warnings.append("GPU memory limit < 1 GB may cause issues")

        # ===== Gradient Length =====
        if self.config.experiment.gradient_length < 60:
            errors.append("Gradient length must be at least 60 seconds")

        if self.config.experiment.gradient_length > 36000:
            warnings.append("Gradient length > 10 hours is unusually long")

        # ===== Platform Warnings =====
        import platform
        if platform.system() == "Darwin" and self.config.experiment.use_bruker_sdk:
            warnings.append("Bruker SDK is not available on macOS - will be automatically disabled")

        return len(errors) == 0, errors, warnings
