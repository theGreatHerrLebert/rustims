"""
imspy_predictors - ML-based predictors for CCS, retention time, and fragment intensity in mass spectrometry.

This package provides machine learning models for predicting peptide properties:
- CCS (Collision Cross Section) / Ion Mobility
- Retention Time
- Fragment Intensity (Prosit)
- Charge State / Ionization

Requires imspy-core for core data structures.

Optional dependencies:
- koina: For remote model access via Koina servers (pip install imspy-predictors[koina])
- imspy-search: For sagepy-based PSM predictions
- imspy-simulation: For simulation utilities
"""

__version__ = "0.4.0"

# Core utility functions
from imspy_predictors.utility import (
    get_model_path,
    load_tokenizer_from_resources,
    InMemoryCheckpoint,
)

# Hashing utilities
from imspy_predictors.hashing import (
    CosimHasher,
    TimsHasher,
)

# Mixture models
from imspy_predictors.mixture import (
    GaussianMixtureModel,
)

# CCS / Ion Mobility predictors
from imspy_predictors.ccs import (
    PeptideIonMobilityApex,
    DeepPeptideIonMobilityApex,
    GRUCCSPredictor,
    GRUCCSPredictorStd,
    SquareRootProjectionLayer,
    load_deep_ccs_predictor,
    load_deep_ccs_std_predictor,
    get_sqrt_slopes_and_intercepts,
    predict_inverse_ion_mobility_with_koina,
)

# Retention time predictors
from imspy_predictors.rt import (
    PeptideChromatographyApex,
    DeepChromatographyApex,
    GRURetentionTimePredictor,
    load_deep_retention_time_predictor,
    get_rt_train_set,
    get_rt_prediction_set,
    predict_retention_time_with_koina,
)

# Fragment intensity predictors
from imspy_predictors.intensity import (
    IonIntensityPredictor,
    Prosit2023TimsTofWrapper,
    load_prosit_2023_timsTOF_predictor,
    get_collision_energy_calibration_factor,
    remove_unimod_annotation,
    generate_prosit_intensity_prediction_dataset,
    predict_fragment_intensities_with_koina,
)

# Charge state / ionization predictors
from imspy_predictors.ionization import (
    PeptideChargeStateDistribution,
    BinomialChargeStateDistributionModel,
    DeepChargeStateDistribution,
    GRUChargeStatePredictor,
    load_deep_charge_state_predictor,
    charge_state_distribution_from_sequence_rust,
    charge_state_distributions_from_sequences_rust,
    predict_peptide_flyability_with_koina,
)

# Tokenizers
from imspy_predictors.utilities import ProformaTokenizer

# HFProformaTokenizer requires transformers (optional)
try:
    from imspy_predictors.utilities import HFProformaTokenizer
except (ImportError, TypeError):
    HFProformaTokenizer = None

__all__ = [
    # Version
    '__version__',
    # Utility
    'get_model_path',
    'load_tokenizer_from_resources',
    'InMemoryCheckpoint',
    # Hashing
    'CosimHasher',
    'TimsHasher',
    # Mixture
    'GaussianMixtureModel',
    # CCS
    'PeptideIonMobilityApex',
    'DeepPeptideIonMobilityApex',
    'GRUCCSPredictor',
    'GRUCCSPredictorStd',
    'SquareRootProjectionLayer',
    'load_deep_ccs_predictor',
    'load_deep_ccs_std_predictor',
    'get_sqrt_slopes_and_intercepts',
    'predict_inverse_ion_mobility_with_koina',
    # RT
    'PeptideChromatographyApex',
    'DeepChromatographyApex',
    'GRURetentionTimePredictor',
    'load_deep_retention_time_predictor',
    'get_rt_train_set',
    'get_rt_prediction_set',
    'predict_retention_time_with_koina',
    # Intensity
    'IonIntensityPredictor',
    'Prosit2023TimsTofWrapper',
    'load_prosit_2023_timsTOF_predictor',
    'get_collision_energy_calibration_factor',
    'remove_unimod_annotation',
    'generate_prosit_intensity_prediction_dataset',
    'predict_fragment_intensities_with_koina',
    # Ionization
    'PeptideChargeStateDistribution',
    'BinomialChargeStateDistributionModel',
    'DeepChargeStateDistribution',
    'GRUChargeStatePredictor',
    'load_deep_charge_state_predictor',
    'charge_state_distribution_from_sequence_rust',
    'charge_state_distributions_from_sequences_rust',
    'predict_peptide_flyability_with_koina',
    # Tokenizers
    'ProformaTokenizer',
    'HFProformaTokenizer',
]
