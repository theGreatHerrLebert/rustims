"""
imspy_predictors - ML-based predictors for CCS, retention time, and fragment intensity in mass spectrometry.

This package provides machine learning models for predicting peptide properties:
- CCS (Collision Cross Section) / Ion Mobility
- Retention Time
- Fragment Intensity (via Koina/Prosit)
- Charge State / Ionization

All models use PyTorch as the deep learning backend.

Requires imspy-core for core data structures and Rust tokenizer.

Optional dependencies:
- koina: For remote model access via Koina servers (pip install imspy-predictors[koina])
- imspy-search: For sagepy-based PSM predictions
- imspy-simulation: For simulation utilities
"""

__version__ = "0.5.0"

# Track which components are available
_IMSPY_CORE_AVAILABLE = False
_TORCH_AVAILABLE = False

# Core utility functions (no external deps)
from imspy_predictors.utility import (
    get_model_path,
    load_tokenizer_from_resources,
    InMemoryCheckpoint,
    get_device,
    count_parameters,
    save_model_checkpoint,
    load_model_checkpoint,
)

# Hashing utilities (PyTorch only, no imspy_core)
from imspy_predictors.hashing import (
    CosimHasher,
    TimsHasher,
    SpectralHasher,
)

# Mixture models (PyTorch only, no imspy_core)
from imspy_predictors.mixture import (
    GaussianMixtureModel,
)

# CCS / Ion Mobility predictors (requires imspy_core)
try:
    from imspy_predictors.ccs import (
        PeptideIonMobilityApex,
        DeepPeptideIonMobilityApex,
        SquareRootProjectionLayer,
        load_deep_ccs_predictor,
        get_sqrt_slopes_and_intercepts,
        predict_inverse_ion_mobility_with_koina,
    )
    _IMSPY_CORE_AVAILABLE = True
except ImportError:
    PeptideIonMobilityApex = None
    DeepPeptideIonMobilityApex = None
    SquareRootProjectionLayer = None
    load_deep_ccs_predictor = None
    get_sqrt_slopes_and_intercepts = None
    predict_inverse_ion_mobility_with_koina = None

# Retention time predictors (requires imspy_core)
try:
    from imspy_predictors.rt import (
        PeptideChromatographyApex,
        DeepChromatographyApex,
        load_deep_retention_time_predictor,
        predict_retention_time_with_koina,
        linear_map,
    )
except ImportError:
    PeptideChromatographyApex = None
    DeepChromatographyApex = None
    load_deep_retention_time_predictor = None
    predict_retention_time_with_koina = None
    linear_map = None

# Fragment intensity predictors (requires imspy_core)
try:
    from imspy_predictors.intensity import (
        IonIntensityPredictor,
        Prosit2023TimsTofWrapper,
        get_collision_energy_calibration_factor,
        remove_unimod_annotation,
        predict_fragment_intensities_with_koina,
        post_process_predicted_fragment_spectra,
        get_prosit_intensity_flat_labels,
    )
except ImportError:
    IonIntensityPredictor = None
    Prosit2023TimsTofWrapper = None
    get_collision_energy_calibration_factor = None
    remove_unimod_annotation = None
    predict_fragment_intensities_with_koina = None
    post_process_predicted_fragment_spectra = None
    get_prosit_intensity_flat_labels = None

# Charge state / ionization predictors (requires imspy_core)
try:
    from imspy_predictors.ionization import (
        PeptideChargeStateDistribution,
        BinomialChargeStateDistributionModel,
        DeepChargeStateDistribution,
        load_deep_charge_state_predictor,
        charge_state_distribution_from_sequence_rust,
        charge_state_distributions_from_sequences_rust,
        predict_peptide_flyability_with_koina,
    )
except ImportError:
    PeptideChargeStateDistribution = None
    BinomialChargeStateDistributionModel = None
    DeepChargeStateDistribution = None
    load_deep_charge_state_predictor = None
    charge_state_distribution_from_sequence_rust = None
    charge_state_distributions_from_sequences_rust = None
    predict_peptide_flyability_with_koina = None

# Tokenizers (requires imspy_core/Rust bindings)
try:
    from imspy_predictors.utilities import ProformaTokenizer
except ImportError:
    ProformaTokenizer = None

# HFProformaTokenizer requires transformers (optional)
try:
    from imspy_predictors.utilities import HFProformaTokenizer
except (ImportError, TypeError):
    HFProformaTokenizer = None

# New PyTorch models (optional - requires torch)
try:
    from imspy_predictors.models import (
        PeptideTransformer,
        PeptideTransformerConfig,
        UnifiedPeptideModel,
        TaskLoss,
        CCSHead,
        RTHead,
        ChargeHead,
        IntensityHead,
        INSTRUMENT_TYPES,
        INSTRUMENT_TO_ID,
        get_instrument_id,
    )
    from imspy_predictors.data_utils import (
        PeptideDataset,
        HuggingFaceDatasetWrapper,
        create_dataloader,
        collate_peptide_batch,
        load_ionmob_dataset,
        load_prospect_rt_dataset,
        load_prospect_charge_dataset,
        load_prospect_ms2_dataset,
        load_timstof_ms2_dataset,
    )
    from imspy_predictors.training import (
        Trainer,
        TrainingConfig,
        EarlyStopping,
        MetricTracker,
        train_ccs_model,
        train_rt_model,
        train_intensity_model,
    )
    _TORCH_AVAILABLE = True
except ImportError:
    PeptideTransformer = None
    PeptideTransformerConfig = None
    UnifiedPeptideModel = None
    TaskLoss = None
    CCSHead = None
    RTHead = None
    ChargeHead = None
    IntensityHead = None
    INSTRUMENT_TYPES = None
    INSTRUMENT_TO_ID = None
    get_instrument_id = None
    PeptideDataset = None
    HuggingFaceDatasetWrapper = None
    create_dataloader = None
    collate_peptide_batch = None
    load_ionmob_dataset = None
    load_prospect_rt_dataset = None
    load_prospect_charge_dataset = None
    load_prospect_ms2_dataset = None
    load_timstof_ms2_dataset = None
    Trainer = None
    TrainingConfig = None
    EarlyStopping = None
    MetricTracker = None
    train_ccs_model = None
    train_rt_model = None
    train_intensity_model = None

__all__ = [
    # Version
    '__version__',
    # Utility
    'get_model_path',
    'load_tokenizer_from_resources',
    'InMemoryCheckpoint',
    'get_device',
    'count_parameters',
    'save_model_checkpoint',
    'load_model_checkpoint',
    # Hashing
    'CosimHasher',
    'TimsHasher',
    'SpectralHasher',
    # Mixture
    'GaussianMixtureModel',
    # CCS
    'PeptideIonMobilityApex',
    'DeepPeptideIonMobilityApex',
    'SquareRootProjectionLayer',
    'load_deep_ccs_predictor',
    'get_sqrt_slopes_and_intercepts',
    'predict_inverse_ion_mobility_with_koina',
    # RT
    'PeptideChromatographyApex',
    'DeepChromatographyApex',
    'load_deep_retention_time_predictor',
    'predict_retention_time_with_koina',
    'linear_map',
    # Intensity
    'IonIntensityPredictor',
    'Prosit2023TimsTofWrapper',
    'get_collision_energy_calibration_factor',
    'remove_unimod_annotation',
    'predict_fragment_intensities_with_koina',
    'post_process_predicted_fragment_spectra',
    'get_prosit_intensity_flat_labels',
    # Ionization
    'PeptideChargeStateDistribution',
    'BinomialChargeStateDistributionModel',
    'DeepChargeStateDistribution',
    'load_deep_charge_state_predictor',
    'charge_state_distribution_from_sequence_rust',
    'charge_state_distributions_from_sequences_rust',
    'predict_peptide_flyability_with_koina',
    # Tokenizers
    'ProformaTokenizer',
    'HFProformaTokenizer',
    # PyTorch models (new unified architecture)
    'PeptideTransformer',
    'PeptideTransformerConfig',
    'UnifiedPeptideModel',
    'TaskLoss',
    'CCSHead',
    'RTHead',
    'ChargeHead',
    'IntensityHead',
    'INSTRUMENT_TYPES',
    'INSTRUMENT_TO_ID',
    'get_instrument_id',
    # Data utilities
    'PeptideDataset',
    'HuggingFaceDatasetWrapper',
    'create_dataloader',
    'collate_peptide_batch',
    'load_ionmob_dataset',
    'load_prospect_rt_dataset',
    'load_prospect_charge_dataset',
    'load_prospect_ms2_dataset',
    'load_timstof_ms2_dataset',
    # Training utilities
    'Trainer',
    'TrainingConfig',
    'EarlyStopping',
    'MetricTracker',
    'train_ccs_model',
    'train_rt_model',
    'train_intensity_model',
]
